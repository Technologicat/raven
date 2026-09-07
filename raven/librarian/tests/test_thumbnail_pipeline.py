"""Attachment sidecar bytes to a drawn card, across `chat_controller` and `chatgraph_panel`.

Every other test of the chat graph's thumbnails hands the panel a provider that answers instantly with a
made-up texture. That covers the drawing and the bookkeeping and nothing else: the decode, the
letterboxing, the upload and the two `split_frame`s all live in `DPGChatController`, and a provider that
never says "not ready" also never exercises the waiting. This is the one that runs the real thing.

**It needs a mapped window, which is why it carries the `gui` marker.** `split_frame` waits for the render
loop to complete a frame, so there has to be a render loop — and this test *is* it, pumping frames on the
main thread while the preparation runs on a worker, exactly as the app does.

Named for the pipeline rather than for a module because it spans two, which is what makes it worth having:
each half is covered on its own and the seam between them is not.
"""

import concurrent.futures
import io
import threading
import time

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")
Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")
# The methods under test are the controller's, and importing it reaches the ML stack -- `hybridir` alone
# wants bm25s, chromadb and watchdog. Skipping on the module itself is what `test_chat_controller.py` does
# and for the same reason; naming a dependency instead would only name whichever one moved last. Nothing
# is lost in CI, which skips this module anyway for want of a display.
pytest.importorskip("raven.librarian.chat_controller")  # noqa: E402 -- see above

from raven.common import bgtask  # noqa: E402 -- after importorskip by design
from raven.common.gui import utils as guiutils  # noqa: E402 -- after importorskip by design
from raven.common.gui.xdotwidget import graph as xdotgraph  # noqa: E402 -- after importorskip by design

from raven.librarian import chatgraph_panel  # noqa: E402 -- after importorskip by design
from raven.librarian.chat_controller import DPGChatController  # noqa: E402 -- after importorskip by design
from raven.librarian.chattree import Forest  # noqa: E402 -- after importorskip by design

pytestmark = pytest.mark.gui

_payload_serial = 0


def payload(role, text, images=(), documents=()):
    global _payload_serial
    _payload_serial += 1
    content = [{"type": "text", "text": text}]
    content += [{"type": "image_url", "image_url": {"url": f"sidecar:{n}"}} for n in images]
    content += [{"type": "text_file", "text_file": {"url": f"sidecar:{n}", "name": n}} for n in documents]
    return {"message": {"role": role, "content": content},
            "general_metadata": {"persona": None, "timestamp": _payload_serial}}


def png_bytes(width, height, rgb):
    """Real encoded PNG bytes, which is what a sidecar holds and what the decode has to cope with."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), rgb).save(buffer, format="PNG")
    return buffer.getvalue()


class ThumbnailHost:
    """Just the attributes `get_graph_thumbnail_texture` touches, so the real method can be run.

    Constructing a `DPGChatController` needs an LLM backend and most of an app, and none of that is in the
    picture here. The two methods under test are bound to this instead, so what runs is the shipped code
    rather than a copy of it — and if either grows a dependency this class does not have, this test says so
    by failing rather than by drifting out of date.
    """

    # Re-wrapped: attribute access unwraps the descriptor, so a bare assignment would make the static
    # method an instance method and pass `self` as the filename.
    graph_thumbnail_identity = staticmethod(DPGChatController.graph_thumbnail_identity)
    get_graph_thumbnail_texture = DPGChatController.get_graph_thumbnail_texture
    _prepare_graph_thumbnail = DPGChatController._prepare_graph_thumbnail

    def __init__(self, datastore, registry, executor):
        self.datastore = datastore
        self._inline_image_texture_registry = registry
        self._graph_thumbnail_textures = {}
        self._graph_thumbnail_pending = set()
        self._graph_thumbnail_failed = set()
        self._graph_thumbnail_lock = threading.Lock()
        self.task_manager = bgtask.TaskManager(name="thumbnail_pipeline_test", mode="concurrent",
                                               executor=executor)


@pytest.fixture(scope="module")
def themes_and_fonts(mapped_gui_context):
    """One `bootup` for the module. It builds themes and font atlases into the context, and doing that per
    test leaves DPG's container stack in a state the next call cannot pop."""
    return guiutils.bootup(font_size=14)


@pytest.fixture
def pipeline(mapped_gui_context, themes_and_fonts):
    """A panel over a forest of real attachments, with the real preparation behind it.

    Yields `(panel, forest, names, pump)`, where `pump` renders frames until a predicate holds. Rendering
    is the fixture's whole reason for existing: the preparation calls `split_frame`, which waits for a
    frame, and nothing else here would ever produce one.
    """
    forest = Forest()
    root = forest.create_node(payload("system", "the card"), parent_id=None)
    names = {"wide": forest.store_sidecar(png_bytes(400, 100, (200, 60, 60)), "png"),
             "tall": forest.store_sidecar(png_bytes(100, 400, (60, 160, 90)), "png"),
             "document": forest.store_sidecar(b"a document's bytes, which nothing decodes", "pdf")}
    carrier = forest.create_node(payload("user", "look at these",
                                         images=[names["wide"], names["tall"]],
                                         documents=[names["document"]]),
                                 parent_id=root)
    app_state = {"HEAD": carrier}

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    registry = dpg.add_texture_registry()
    host = ThumbnailHost(forest, registry, executor)
    with dpg.window() as holder:
        panel = chatgraph_panel.DPGChatGraphPanel(
            gui_parent=holder, datastore=forest, app_state=app_state,
            themes_and_fonts=themes_and_fonts, width=600, height=400, show=True,
            thumbnail_for=host.get_graph_thumbnail_texture)

    def pump(predicate, timeout=20.0):
        """Render frames until `predicate()` or `timeout`, rebuilding as thumbnails land.

        The rebuild is what the panel's own animator hook does once per frame, and doing it here rather
        than by ticking the process-wide animator keeps this test from advancing every other animation in
        the session. Without it the picture freezes at whatever had landed by the last explicit refresh —
        which is one thumbnail of three, and reads exactly like a preparation that stalled.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            dpg.render_dearpygui_frame()
            if panel._is_stale():
                panel.refresh()
            if predicate():
                return True
        return False

    panel.refresh()
    yield panel, forest, names, carrier, pump

    panel.destroy()
    dpg.delete_item(holder)
    dpg.delete_item(registry)
    executor.shutdown(wait=False)


def cards(panel, node_name):
    """The cards fanned off one box, in the order the message carries its attachments.

    By drawing order reversed, which is what that order is: the deck is drawn back to front so the first
    attachment lies on top. **Not by position** — the pile lays its columns out as a Latin square, so for
    three cards the middle one is the rightmost, and sorting by x silently pairs each assertion with the
    wrong attachment.
    """
    node = panel._chat_graph.graph.get_node_by_name(node_name)
    centre_of_box = 0.5 * (node.get_bounding_box()[0] + node.get_bounding_box()[2])

    def centre(shape):
        box = shape.get_bounding_box()
        return 0.5 * (box[0] + box[2])
    return list(reversed([s for s in node.shapes
                          if isinstance(s, xdotgraph.ImageShape) and centre(s) > centre_of_box]))


def test_real_attachments_become_drawn_cards(pipeline):
    """The seam: bytes in a sidecar, out the other end as textures the graph has drawn."""
    panel, forest, names, carrier, pump = pipeline

    assert [c.texture for c in cards(panel, carrier)] == [None, None, None], \
        "something was ready before a frame had rendered, so this test never waits for anything"
    assert pump(lambda: all(c.texture is not None for c in cards(panel, carrier))), \
        f"not every card was filled: {[c.texture for c in cards(panel, carrier)]}"
    assert len(cards(panel, carrier)) == 3
    assert panel._awaited_thumbnails == set(), "the panel is still waiting for something it has"


def test_a_document_is_drawn_as_its_file_type(pipeline):
    """It has no picture of its own, and a message that is only attachments has no words either — so
    without this it is an `[empty]` box with nothing beside it."""
    panel, forest, names, carrier, pump = pipeline
    assert pump(lambda: all(c.texture is not None for c in cards(panel, carrier)))
    document_card = cards(panel, carrier)[-1]  # attached last, so drawn last in the fan
    assert "document" in str(document_card.texture), \
        f"the .pdf did not get the generic document icon: {document_card.texture}"


def test_the_pictures_keep_their_proportions(pipeline):
    """A 4:1 image and a 1:4 one, so a card that squared them would be caught either way round."""
    panel, forest, names, carrier, pump = pipeline
    assert pump(lambda: all(c.texture is not None for c in cards(panel, carrier)))
    wide, tall = cards(panel, carrier)[0], cards(panel, carrier)[1]

    def aspect(card):
        box = card.get_bounding_box()
        return (box[2] - box[0]) / (box[3] - box[1])
    assert aspect(wide) == pytest.approx(4.0, rel=0.05)
    assert aspect(tall) == pytest.approx(0.25, rel=0.05)


def test_one_texture_serves_every_document_of_a_type(pipeline):
    """Three attached PDFs are one icon, not three copies of it. Images stay their own, being content
    addressed already — the same picture attached twice decodes once for that reason instead."""
    panel, forest, names, carrier, pump = pipeline
    assert pump(lambda: all(c.texture is not None for c in cards(panel, carrier)))
    identity = DPGChatController.graph_thumbnail_identity
    assert identity("one.pdf") == identity("another.pdf")
    assert identity(names["wide"]) != identity(names["tall"]), \
        "two different images share an identity, so they would share a texture"


def test_a_picture_arrives_as_a_mip_chain(pipeline):
    """One prepared size cannot serve a graph that zooms, and this is the end that builds the rest.

    The declared sizes are checked against the textures DPG actually holds, because they are what the
    renderer chooses by: a chain whose levels claim sizes they do not have would pick wrongly while
    looking perfectly well formed.
    """
    panel, forest, names, carrier, pump = pipeline
    assert pump(lambda: all(c.texture is not None for c in cards(panel, carrier)))
    wide, tall = cards(panel, carrier)[0], cards(panel, carrier)[1]

    def size_of(texture):
        configuration = dpg.get_item_configuration(texture)
        return (configuration["width"], configuration["height"])

    for card in (wide, tall):
        assert card.mips, "the picture came back as a single texture, so there is no chain to draw from"
        for level in card.mips:
            assert (level.width, level.height) == size_of(level.texture), \
                f"a level claims {(level.width, level.height)} and holds {size_of(level.texture)}"
        sizes = [size_of(card.texture)] + [(level.width, level.height) for level in card.mips]
        for finer, coarser in zip(sizes, sizes[1:]):
            assert coarser[0] < finer[0] and coarser[1] < finer[1], \
                f"the chain does not descend: {sizes}"
