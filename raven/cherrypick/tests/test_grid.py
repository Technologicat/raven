"""Tests for `raven.cherrypick.grid` — that the triage grid still fits the base class it extends.

`TriageGrid` spells its constructor out rather than forwarding `**kwargs`, and it reads Cherrypick's config
itself. Both are deliberate, and together they mean a parameter added to `ThumbnailGrid` reaches it only if
someone remembers to thread it through. When that is missed the failure is a `TypeError` raised while the
app builds its window — invisible to every test of the base class, since none of them construct the
subclass the app actually uses. That happened on 2026-08-14, which is why this module exists.

These reach for private attributes on purpose: what is under test is the wiring between two of our own
classes, and the values are not otherwise observable without a font atlas and a rendered frame.

No window is mapped, so nothing here takes keyboard focus.
"""

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from raven.cherrypick import config  # noqa: E402 -- after importorskip by design
from raven.cherrypick.grid import TriageGrid  # noqa: E402 -- after importorskip by design
from raven.common.gui.thumbnailgrid import ThumbnailGrid  # noqa: E402 -- after importorskip by design


@pytest.fixture(scope="module")
def dpg_context():
    """One DPG context for the whole module, with an unmapped viewport."""
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


@pytest.fixture
def triage_grid(dpg_context, request):
    """A `TriageGrid` in a throwaway window, torn down after the test."""
    window = dpg.add_window(label="host", tag=f"host_{request.node.name}")
    grid = TriageGrid(parent=window, width=500, height=400)
    yield grid
    grid.destroy()
    dpg.delete_item(window)


def test_the_subclass_constructs(triage_grid):
    """The drift guard. A parameter the base gained and the subclass did not pass fails right here."""
    assert isinstance(triage_grid, ThumbnailGrid)


def test_cherrypicks_scrolling_settings_reach_the_base(triage_grid):
    """Config the subclass reads on the app's behalf has to arrive, not merely be read."""
    assert triage_grid._smooth_scrolling == config.SMOOTH_SCROLLING
    assert triage_grid._smooth_scrolling_step_parameter == config.SMOOTH_SCROLLING_STEP_PARAMETER


def test_entries_and_navigation_work_through_the_subclass(triage_grid):
    """The base's behaviour survives the override layer, which is the other half of "still fits"."""
    triage_grid.set_entries([f"image {i}" for i in range(8)])
    triage_grid._compute_layout()  # normally done by the rebuild in update()

    assert triage_grid.visible_count == 8
    assert triage_grid.current == 0

    triage_grid.navigate_next()

    assert triage_grid.current == 1
