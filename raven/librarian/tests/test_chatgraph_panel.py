"""Tests for `chatgraph_panel` — what a click on the chat graph means, and when the picture rebuilds.

The panel's whole job is a state machine over a real widget: preview, preview again, commit; move a
window; notice that the tree changed. What is worth pinning is that **preview changes nothing** — the
non-destructive half of the interaction is the half a visitor depends on, and it is the half that would
break silently, since a commit that happens too eagerly looks exactly like one the reader asked for.

A real DPG context is needed (the panel builds widgets) but no rendered frame and no mapped window: nothing
here asks about layout, and a mapped window would steal the keyboard from whoever is running the suite.
"""

import time

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from raven.common.gui import animation as gui_animation  # noqa: E402 -- after importorskip by design
from raven.common.gui import utils as guiutils  # noqa: E402 -- after importorskip by design

from raven.librarian import chatgraph  # noqa: E402 -- after importorskip by design
from raven.librarian import chatgraph_panel  # noqa: E402 -- after importorskip by design
from raven.librarian.chattree import Forest  # noqa: E402 -- after importorskip by design


def payload(role: str, text: str) -> dict:
    """A chat node payload of the shape `chatutil` writes."""
    return {"message": {"role": role, "content": [{"type": "text", "text": text}]},
            "general_metadata": {"persona": None}}


@pytest.fixture(scope="module")
def dpg_context():
    """One DPG context for the whole module, with an unmapped viewport.

    Module-scoped, which is the house pattern for DPG tests here: a context is not cheap, DPG keeps global
    state, and recreating one per test is what took the `--run-gui` group down for eleven days.
    """
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    themes_and_fonts = guiutils.bootup(font_size=14)
    yield themes_and_fonts
    dpg.destroy_context()


class Calls:
    """Records what the panel asked its caller to do."""
    def __init__(self):
        self.previewed = []
        self.committed = []


@pytest.fixture
def panel(dpg_context):
    """A panel over a small forked forest, plus the handles a test needs to drive it.

        system -> user -> taken -> taken_tip        <- HEAD is here
                       -> not_taken -> not_taken_tip
    """
    themes_and_fonts = dpg_context
    forest = Forest()
    ids = {}
    ids["system"] = forest.create_node(payload("system", "you are helpful"), parent_id=None)
    ids["user"] = forest.create_node(payload("user", "which way"), parent_id=ids["system"])
    ids["taken"] = forest.create_node(payload("assistant", "this way"), parent_id=ids["user"])
    ids["taken_tip"] = forest.create_node(payload("user", "onwards"), parent_id=ids["taken"])
    ids["not_taken"] = forest.create_node(payload("assistant", "or that way"), parent_id=ids["user"])
    ids["not_taken_tip"] = forest.create_node(payload("user", "elsewhere"), parent_id=ids["not_taken"])

    app_state = {"HEAD": ids["taken_tip"], "new_chat_HEAD": ids["user"]}
    calls = Calls()

    with dpg.window() as holder:
        built = chatgraph_panel.DPGChatGraphPanel(
            gui_parent=holder,
            datastore=forest,
            app_state=app_state,
            themes_and_fonts=themes_and_fonts,
            width=400, height=300,
            on_preview=calls.previewed.append,
            on_commit=calls.committed.append,
            show=True)
    built.refresh()

    yield built, forest, app_state, ids, calls

    built.destroy()
    dpg.delete_item(holder)


def click(panel_obj, node_name: str) -> None:
    """Deliver a left click on the named graph box, the way the widget would.

    The widget hands over the `Node` it hit, not its name, so the lookup here is part of what is being
    tested: a panel reading the wrong field off it finds no ref and does nothing.
    """
    node = panel_obj._chat_graph.graph.get_node_by_name(node_name)
    assert node is not None, f"no box named '{node_name}' in the current picture"
    panel_obj._on_click(node, dpg.mvMouseButton_Left)


# ---------------------------------------------------------------------------
# Preview changes nothing
# ---------------------------------------------------------------------------

class TestPreview:
    def test_clicking_a_node_on_the_branch_scrolls_the_chat_log(self, panel):
        built, forest, app_state, ids, calls = panel
        click(built, ids["taken"])
        assert calls.previewed == [ids["taken"]]
        assert calls.committed == []

    def test_clicking_a_node_off_the_branch_moves_the_picture_instead(self, panel):
        # There is no message in the chat log for a node on another branch, so there is nothing to scroll
        # to; the graph redraws around it instead, bringing its siblings and children into view.
        built, forest, app_state, ids, calls = panel
        click(built, ids["not_taken"])
        assert calls.previewed == [], "a node off the branch has no message in the log to scroll to"
        assert calls.committed == []
        assert built._chat_graph.spine == tuple(forest.linearize_up(ids["not_taken"]))

    def test_no_preview_moves_head(self, panel):
        # The property a visitor depends on, stated once for every kind of box there is: browsing leaves
        # the session where it was found.
        built, forest, app_state, ids, calls = panel
        head_before = app_state["HEAD"]
        for name in (ids["taken"], ids["not_taken"], ids["system"]):
            click(built, name)
        assert calls.committed == []
        assert app_state["HEAD"] == head_before

    def test_the_previewed_box_is_highlighted(self, panel):
        # What a second click will act on has to be visible before it happens, or commit-on-second-click is
        # a trap rather than a gesture.
        built, forest, app_state, ids, calls = panel
        click(built, ids["taken"])
        assert built._widget.get_highlighted_nodes() == {ids["taken"]}


# ---------------------------------------------------------------------------
# Commit is the second act
# ---------------------------------------------------------------------------

class TestCommit:
    def test_a_second_click_on_the_same_box_commits(self, panel):
        built, forest, app_state, ids, calls = panel
        click(built, ids["taken"])
        click(built, ids["taken"])
        assert calls.committed == [ids["taken"]]

    def test_a_second_click_on_a_different_box_does_not(self, panel):
        # The control that keeps the test above honest: "clicked twice" must mean the same box twice, not
        # any two clicks. Without this, a panel that committed on every second click would pass it.
        built, forest, app_state, ids, calls = panel
        click(built, ids["taken"])
        click(built, ids["system"])
        assert calls.committed == []
        assert calls.previewed == [ids["taken"], ids["system"]]

    def test_the_toolbar_button_commits_the_previewed_node(self, panel):
        built, forest, app_state, ids, calls = panel
        click(built, ids["not_taken"])
        built._commit_previewed()
        assert calls.committed == [ids["not_taken"]]

    def test_the_toolbar_button_does_nothing_with_nothing_previewed(self, panel):
        built, forest, app_state, ids, calls = panel
        built._commit_previewed()
        assert calls.committed == []

    def test_committing_clears_the_preview(self, panel):
        # Otherwise the next click on that same box would read as a second click and commit again, which is
        # a move the reader did not ask for.
        built, forest, app_state, ids, calls = panel
        click(built, ids["taken"])
        click(built, ids["taken"])
        click(built, ids["taken"])
        assert calls.committed == [ids["taken"]], "the third click committed again"

    def test_going_home_abandons_the_preview(self, panel):
        built, forest, app_state, ids, calls = panel
        click(built, ids["not_taken"])
        assert built._widget.get_highlighted_nodes(), "nothing was previewed, so clearing it proves nothing"

        built.go_to_head()
        assert built._chat_graph.spine == tuple(forest.linearize_up(app_state["HEAD"]))
        assert built._previewed_node_id is None, "the preview survived, so a click would commit it"
        assert built._widget.get_highlighted_nodes() == set(), "the preview mark is still lit"

    def test_going_home_flashes_where_it_landed(self, panel):
        # The view slides and the zoom changes together, and HEAD is parked off-centre on purpose, so
        # nothing about the motion says which box was the destination.
        built, forest, app_state, ids, calls = panel
        built.go_to_head()
        highlight = built._widget._highlight
        flashed = {element.internal_name for element in highlight._fading}
        assert flashed == {app_state["HEAD"]}

    def test_the_flash_fades_rather_than_switching_off(self, panel):
        # Through the same fade-out a hover uses. Lighting a box and then cutting it reads as a glitch
        # beside a hover that fades, which is what the first version of this did.
        built, forest, app_state, ids, calls = panel
        built.go_to_head()
        highlight = built._widget._highlight
        assert highlight.is_animating(), "the flash is not fading, so it can only be switching off"

        # Wind its start time past the fade rather than sleeping through it.
        element = next(iter(highlight._fading))
        highlight._fading[element] = (time.monotonic_ns() - int(2e9 * highlight.fade_duration), 1.0)
        highlight.update()
        assert not highlight.is_animating()
        assert built._widget.get_highlighted_nodes() == set(), "the fade left a permanent mark behind"


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------

class TestFraming:
    def test_the_first_build_frames_the_branch(self, panel):
        built, forest, app_state, ids, calls = panel
        # The fixture already refreshed once. What that must have fitted is the branch, not the graph: a
        # windowed level is thousands of units wide against a panel of hundreds, and fitting *that* lands
        # at a zoom where the renderer stops drawing text at all.
        x1, y1, x2, y2 = built._chat_graph.spine_bbox
        assert x2 - x1 < built._chat_graph.graph.width, \
            "the branch is as wide as the whole picture here, so this fixture cannot tell the two fits apart"
        assert built._framed

    def test_later_rebuilds_do_not_reframe(self, panel):
        # The picture must stay still while a reply arrives. The tree gains a node per round, so a re-fit
        # per rebuild is a lurch per turn of the conversation.
        built, forest, app_state, ids, calls = panel
        before = built._widget.get_zoom()
        forest.create_node(payload("assistant", "a fresh reroll"), parent_id=ids["taken"])
        built.refresh()
        assert built._widget.get_zoom() == before

    def test_going_home_reframes(self, panel):
        # The one deliberate exception: asking to be returned to HEAD is asking for a framing.
        built, forest, app_state, ids, calls = panel
        built._widget.set_zoom(0.3, animate=False)
        built.go_to_head()
        assert built._widget._viewport.zoom.target == pytest.approx(1.0), \
            "the crosshair should return the reader to full size"


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------

class TestGaps:
    def _wide(self, dpg_context, width=30):
        themes_and_fonts = dpg_context
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        greeting = forest.create_node(payload("assistant", "hello!"), parent_id=root)
        chats = [forest.create_node(payload("user", f"chat {k}"), parent_id=greeting)
                 for k in range(width)]
        app_state = {"HEAD": chats[0], "new_chat_HEAD": greeting}
        with dpg.window() as holder:
            built = chatgraph_panel.DPGChatGraphPanel(
                gui_parent=holder, datastore=forest, app_state=app_state,
                themes_and_fonts=themes_and_fonts, width=400, height=300, show=True)
        built.refresh()
        return built, forest, chats, holder

    def test_a_sibling_gap_moves_the_window_onto_what_it_hid(self, dpg_context):
        built, forest, chats, holder = self._wide(dpg_context)
        try:
            gaps = [ref for ref in built._chat_graph.refs.values()
                    if isinstance(ref, chatgraph.SiblingGapRef)]
            assert gaps, "no sibling gap in this fixture, so there is nothing to click"
            target = gaps[0].recenter_on
            assert target not in built._chat_graph.refs, \
                "the gap's target is already on screen, so moving the window would prove nothing"

            click(built, gaps[0].name)
            assert target in built._chat_graph.refs, "the window did not move onto the hidden sibling"
        finally:
            built.destroy()
            dpg.delete_item(holder)

    def test_moving_the_window_is_not_a_preview(self, dpg_context):
        # A window move is navigation, not a choice of node, so it must not leave a box armed for commit.
        built, forest, chats, holder = self._wide(dpg_context)
        try:
            gaps = [ref for ref in built._chat_graph.refs.values()
                    if isinstance(ref, chatgraph.SiblingGapRef)]
            click(built, gaps[0].name)
            assert built._previewed_node_id is None
            assert built._widget.get_highlighted_nodes() == set()
        finally:
            built.destroy()
            dpg.delete_item(holder)

    def test_a_depth_gap_shows_more_of_the_branch(self, dpg_context):
        themes_and_fonts = dpg_context
        forest = Forest()
        node = None
        ids = []
        for k in range(40):
            node = forest.create_node(payload("user" if k % 2 == 0 else "assistant", f"m{k}"),
                                      parent_id=node)
            ids.append(node)
        app_state = {"HEAD": ids[-1], "new_chat_HEAD": None}
        with dpg.window() as holder:
            built = chatgraph_panel.DPGChatGraphPanel(
                gui_parent=holder, datastore=forest, app_state=app_state,
                themes_and_fonts=themes_and_fonts, width=400, height=300, show=True)
        try:
            built.refresh()
            before = sum(1 for ref in built._chat_graph.refs.values()
                         if isinstance(ref, chatgraph.ChatNodeRef))
            gaps = [ref for ref in built._chat_graph.refs.values()
                    if isinstance(ref, chatgraph.DepthGapRef)]
            assert gaps, "this branch is not long enough to be truncated"

            click(built, gaps[0].name)
            after = sum(1 for ref in built._chat_graph.refs.values()
                        if isinstance(ref, chatgraph.ChatNodeRef))
            assert after > before, f"clicking the depth gap showed no more of the branch ({before} -> {after})"
        finally:
            built.destroy()
            dpg.delete_item(holder)


# ---------------------------------------------------------------------------
# Keeping up with the forest
# ---------------------------------------------------------------------------

class TestStaleness:
    def test_a_new_node_makes_the_picture_stale(self, panel):
        built, forest, app_state, ids, calls = panel
        assert not built._is_stale(), "the fixture starts stale, so this test cannot tell staleness apart"
        forest.create_node(payload("assistant", "a fresh reroll"), parent_id=ids["taken"])
        assert built._is_stale()

    def test_a_moved_head_makes_the_picture_stale(self, panel):
        # The other half, and neither implies the other: a branch switch adds no nodes, so a check that
        # watched only the change counter would sit on the wrong branch indefinitely.
        built, forest, app_state, ids, calls = panel
        generation_before = forest.generation
        app_state["HEAD"] = ids["not_taken_tip"]
        assert forest.generation == generation_before, \
            "moving HEAD touched the datastore, so this fixture cannot isolate the two signals"
        assert built._is_stale()

    def test_a_refresh_settles_it(self, panel):
        built, forest, app_state, ids, calls = panel
        forest.create_node(payload("assistant", "a fresh reroll"), parent_id=ids["taken"])
        built.refresh()
        assert not built._is_stale()

    def test_the_frame_hook_rebuilds_a_stale_picture(self, panel):
        built, forest, app_state, ids, calls = panel
        new_node = forest.create_node(payload("assistant", "a fresh reroll"), parent_id=ids["taken"])
        assert new_node not in built._chat_graph.refs
        built.render_frame(0)
        assert new_node in built._chat_graph.refs

    def test_a_hidden_panel_does_not_rebuild(self, panel):
        # It refreshes on `show` instead. A panel nobody is looking at should not be paying for a rebuild
        # on every turn of a conversation.
        built, forest, app_state, ids, calls = panel
        built.hide()
        forest.create_node(payload("assistant", "a fresh reroll"), parent_id=ids["taken"])
        built.render_frame(0)
        assert built._is_stale(), "the hidden panel rebuilt anyway"
        built.show()
        assert not built._is_stale()

    def test_a_vanished_head_does_not_take_the_panel_with_it(self, panel):
        # A cleanup can delete the node HEAD names while the view is up.
        built, forest, app_state, ids, calls = panel
        app_state["HEAD"] = "no-such-node"
        built.refresh()  # must not raise
        assert built._chat_graph is not None


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_two_panels_can_coexist(self, dpg_context):
        # Per-instance tags, checked because the widget beneath had exactly this defect: fixed tags, and
        # the second instance died on "Alias already exists".
        themes_and_fonts = dpg_context
        forest = Forest()
        root = forest.create_node(payload("system", "hi"), parent_id=None)
        app_state = {"HEAD": root}
        with dpg.window() as holder:
            first = chatgraph_panel.DPGChatGraphPanel(
                gui_parent=holder, datastore=forest, app_state=app_state,
                themes_and_fonts=themes_and_fonts, width=200, height=200)
            second = chatgraph_panel.DPGChatGraphPanel(
                gui_parent=holder, datastore=forest, app_state=app_state,
                themes_and_fonts=themes_and_fonts, width=200, height=200)
        assert first.gui_uuid != second.gui_uuid
        first.destroy()
        second.destroy()
        dpg.delete_item(holder)

    def test_destroy_stops_the_animator_calling_back(self, panel):
        # A registered animation whose widgets are gone calls DPG on a freed item every frame, which is how
        # a teardown race becomes a segfault rather than an exception.
        built, forest, app_state, ids, calls = panel
        assert built in gui_animation.animator._animations, \
            "the panel never registered, so this test cannot tell an unregister from a no-op"
        built.destroy()
        assert built not in gui_animation.animator._animations
        built.destroy = lambda: None  # the fixture tears down too; once is enough
