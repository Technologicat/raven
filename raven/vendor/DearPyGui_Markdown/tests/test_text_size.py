"""`get_text_size` refuses an impossible wait instead of hanging on it.

Measuring text needs the face in DPG's font atlas, and a face reaches the atlas only *between frames*. So
a measurement taken before the first frame has to wait for one — and during GUI building the thread that
would render it is the thread waiting for it. That wait can never end.

It used to be an unbounded retry, which presented as a wedged app: no traceback, no log line, nothing to
bisect. These pin the named failure that replaced it. A regression would not fail a test so much as make
the suite stop responding, which is what makes them worth keeping.

**The `wrap` argument is the whole trigger.** Wrapping is what forces a measurement; unwrapped text is
never measured and so needs nothing from the atlas. That is also how an app preloads faces before its
first frame (`raven.visualizer.app`'s `markdown_font_loader_trigger_dummy`).

Nothing here maps a window: an unmapped viewport renders no frames, which is exactly the condition under
test.
"""

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from raven.common.gui import utils as guiutils  # noqa: E402 -- after importorskip by design
from raven.vendor import DearPyGui_Markdown as dpg_markdown  # noqa: E402 -- after importorskip by design


@pytest.fixture(scope="module")
def dpg_context():
    """One DPG context for the whole module, with an unmapped viewport.

    `bootup` rather than a bare context: the renderer needs its fonts configured before it will build
    anything, and that is what configures them. Module-scoped, the house pattern for every DPG test here.
    """
    dpg.create_context()
    guiutils.bootup(font_size=20)
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    yield
    # Building Markdown starts the renderer's worker threads, and they call into DPG. Destroying the
    # context under them segfaults the interpreter — which is how this module found the bug that
    # `dpg_markdown.shutdown` now fixes, so the call is load-bearing rather than tidy.
    dpg_markdown.shutdown()
    dpg.destroy_context()


def test_unwrapped_markdown_builds_before_the_first_frame(dpg_context):
    """The negative control: without this, the test below could be passing for the wrong reason.

    If *nothing* could be built before the first frame, a raise would say only that, and would keep saying
    it if the trigger silently became "any Markdown at all". This fixes which half of the pair is meant to
    work — and it is the half apps rely on to preload their faces.
    """
    window = dpg.add_window(width=80, height=80)
    assert dpg_markdown.add_text("plain words, **bold** and *italic*", parent=window)
    assert dpg_markdown.add_text("a second one, also unmeasured", parent=window)


def test_wrapped_markdown_before_the_first_frame_raises_rather_than_hanging(dpg_context):
    window = dpg.add_window(width=80, height=80)
    with pytest.raises(RuntimeError) as excinfo:
        dpg_markdown.add_text("plain words only, but wrapped", parent=window, wrap=60)
    message = str(excinfo.value)
    # The message is the entire diagnostic here — a reader arrives with a stalled app and no other clue —
    # so it is asserted rather than merely the exception type.
    assert "atlas" in message, message
    assert "wrap" in message, message
