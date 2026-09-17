"""Decorations drawn after their paragraph was deleted.

Underlines, strikethroughs, code backgrounds and link tooltips are added on the renderer's worker thread a
frame after the text they decorate, into containers the paragraph's build handed over. A view rebuild, or a
search swapping a paragraph for its highlighted version, can delete those containers in between. The
decoration must then quietly do nothing: the worker catches whatever escapes and prints a traceback.

Nothing here maps a window or renders a frame.
"""

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from raven.vendor.DearPyGui_Markdown import text_attributes  # noqa: E402 -- after importorskip by design


@pytest.fixture(scope="module")
def dpg_context():
    """One DPG context for the whole module, with an unmapped viewport."""
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    with dpg.window():
        pass
    yield
    dpg.destroy_context()


@pytest.fixture
def fixed_text_size(monkeypatch):
    """Measuring text needs a font atlas, which needs a rendered frame; these tests are about what comes after."""
    monkeypatch.setattr(text_attributes, "get_text_size", lambda text, font=None, **kwargs: (40, 20))


@pytest.mark.parametrize("decoration", [text_attributes.Underline, text_attributes.Strike])
def test_a_line_decoration_into_a_deleted_parent_does_nothing(dpg_context, fixed_text_size, decoration):
    window = dpg.add_window()
    text_group = dpg.add_group(parent=window)
    text = dpg.add_text("decorated", parent=text_group)
    attributes_group = dpg.add_group(parent=window)
    dpg.delete_item(attributes_group)
    try:
        assert not dpg.does_item_exist(attributes_group), "the parent still exists, so this fixture tests nothing"
        drawlist, line = decoration.render(text, text_group, parent=attributes_group)
        assert (drawlist, line) == (None, None)
    finally:
        dpg.delete_item(window)
