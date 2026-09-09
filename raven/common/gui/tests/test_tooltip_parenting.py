"""Pins which DearPyGui item types will accept a tooltip as a child, and which will not.

Raven's chat messages draw their role glyph into a drawlist, and hovering that glyph names the speaker. A
drawlist cannot carry the tooltip: it accepts only draw items as children, so the glyph's drawlist is
wrapped in a group that exists for no other reason. Nothing at the call site fails loudly if somebody
removes that group — the breakage is at runtime, in a view rebuild, and what reaches Python is
`SystemError: <built-in function add_tooltip> returned a result with an exception set`, which names neither
tooltips nor drawlists. DPG's own readable complaint goes to its error handler and not into the exception.

So these assertions exist to notice a change in either direction. If a future DPG lets a drawlist parent a
tooltip, `test_a_drawlist_refuses_a_tooltip` starts failing, and that failure is the signal to go delete the
wrapper rather than a defect to work around.

What they cannot reach is whether the tooltip then *shows*, which is a question about hovering and needs a
real pointer over a mapped window. That the group half works was confirmed live (2026-09-09); see
`dpg-notes.md`, "A drawlist cannot carry a tooltip — wrap it in a group".
"""

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed (GUI toolkit absent in CI)")


@pytest.fixture(scope="module")
def dpg_context():
    """A DPG context with an unmapped viewport, torn down after the module."""
    dpg.create_context()
    dpg.create_viewport(width=400, height=300)  # never shown: these tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


@pytest.fixture
def window(dpg_context):
    """A window to hang items from, taken down after each test."""
    tag = dpg.add_window()
    yield tag
    dpg.delete_item(tag)


class TestWhatCanCarryATooltip:
    def test_a_group_accepts_a_tooltip(self, window):
        # The wrapper Raven's chat icons rely on. Asserted before the refusal below so that a context where
        # *nothing* accepts a tooltip cannot pass for a context where only the drawlist refuses.
        group = dpg.add_group(parent=window)
        tooltip = dpg.add_tooltip(group)
        dpg.add_text("who is speaking", parent=tooltip)
        assert dpg.does_item_exist(tooltip)

    def test_a_drawlist_refuses_a_tooltip(self, window):
        drawlist = dpg.add_drawlist(width=40, height=40, parent=window)
        with pytest.raises(SystemError):
            dpg.add_tooltip(drawlist)

    def test_a_drawlist_inside_a_group_is_the_way_round_that(self, window):
        # The whole shape, as `chat_controller.DPGChatMessage.build` builds it: the drawn content in the
        # drawlist, the caption on the group holding it.
        group = dpg.add_group(parent=window)
        drawlist = dpg.add_drawlist(width=40, height=40, parent=group)
        tooltip = dpg.add_tooltip(group)
        dpg.add_text("Aria", parent=tooltip)
        assert dpg.does_item_exist(drawlist) and dpg.does_item_exist(tooltip)
