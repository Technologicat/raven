"""Tests for `raven.common.gui.thumbnailgrid` — the grid's layout arithmetic and its bookkeeping.

The layout is *computed*, not read back from DPG, which is what buys O(1) hit detection from a mouse
position. The price is that a drifting constant breaks it silently: tiles still draw, clicks just land on
the wrong one. That is the failure these tests exist to catch, and it is why they check the arithmetic
against hand-worked numbers rather than against the code's own formula.

No window is mapped, so nothing here takes keyboard focus and none of it carries the `gui` marker. The
grid creates real DPG items, so it needs a context — but not a rendered frame, except where noted.
"""

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from raven.common.gui.thumbnailgrid import ThumbnailGrid  # noqa: E402 -- after importorskip by design

TILE = 100
# The layout arithmetic, worked by hand for TILE=100 and Raven's standard metrics:
#   col_width  = tile + item_spacing_x           = 100 + 8            = 108
#   text_h     = font_size + 2 * frame_padding_y = 20 + 6             = 26
#   row_height = tile + spacing_y + text_h + spacing_y = 100 + 4 + 26 + 4 = 134
EXPECTED_COL_WIDTH = 108
EXPECTED_ROW_HEIGHT = 134


@pytest.fixture(scope="module")
def dpg_context():
    """One DPG context for the whole module, with an unmapped viewport."""
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


@pytest.fixture
def make_grid(dpg_context, request):
    """Build a grid inside a throwaway window, with a per-test tag so the shared context stays clean."""
    built = []

    def build(n_entries: int = 12, width: int = 500, height: int = 400, **kwargs):
        window = dpg.add_window(label="host", tag=f"host_{request.node.name}_{len(built)}")
        grid = ThumbnailGrid(parent=window, width=width, height=height, tile_size=TILE, **kwargs)
        grid.set_entries([f"entry {i}" for i in range(n_entries)])
        grid._compute_layout()  # normally done by the rebuild in update()
        built.append((grid, window))
        return grid
    yield build
    for grid, window in built:
        grid.destroy()
        dpg.delete_item(window)


# --------------------------------------------------------------------------------
# Layout arithmetic

def test_column_and_row_geometry_match_the_hand_worked_numbers(make_grid):
    grid = make_grid()
    assert grid._col_width == EXPECTED_COL_WIDTH
    assert grid._row_height == EXPECTED_ROW_HEIGHT


def test_column_count_fits_the_panel_minus_the_scrollbar(make_grid):
    """The scrollbar's width is not usable for tiles, and forgetting it overflows the last column."""
    # width 500, scrollbar 14 -> 486 usable; 486 // 108 = 4 columns.
    assert make_grid(width=500).n_cols == 4
    # Exactly four columns' worth plus the scrollbar still fits four, not five.
    assert make_grid(width=4 * EXPECTED_COL_WIDTH + 14).n_cols == 4
    # One pixel short of five columns is still four.
    assert make_grid(width=5 * EXPECTED_COL_WIDTH + 14 - 1).n_cols == 4
    assert make_grid(width=5 * EXPECTED_COL_WIDTH + 14).n_cols == 5


def test_a_panel_narrower_than_one_tile_still_has_a_column(make_grid):
    """Zero columns would divide by zero in hit testing and show nothing; one is the floor."""
    assert make_grid(width=10).n_cols == 1


# --------------------------------------------------------------------------------
# Hit testing — the arithmetic the layout exists to serve

def test_hit_test_maps_positions_to_the_expected_tiles(make_grid):
    grid = make_grid(n_entries=12, width=500)  # 4 columns
    assert grid.hit_test_at(10, 10) == 0                                    # first tile
    assert grid.hit_test_at(EXPECTED_COL_WIDTH + 10, 10) == 1               # second column
    assert grid.hit_test_at(10, EXPECTED_ROW_HEIGHT + 10) == 4              # second row, first column
    assert grid.hit_test_at(3 * EXPECTED_COL_WIDTH + 10,
                            2 * EXPECTED_ROW_HEIGHT + 10) == 11             # last tile of a 12-entry grid


def test_hit_test_misses_the_gaps_between_tiles(make_grid):
    """A tile is `tile_size` wide inside a wider column; the remainder is spacing and must not select."""
    grid = make_grid(width=500)
    assert grid.hit_test_at(TILE + 4, 10) is None                # horizontal spacing after tile 0
    assert grid.hit_test_at(10, TILE + 4) is None                # the label strip below tile 0


def test_hit_test_misses_past_the_last_column_and_the_last_entry(make_grid):
    grid = make_grid(n_entries=6, width=500)  # 4 columns, so row 1 holds entries 4 and 5
    assert grid.hit_test_at(4 * EXPECTED_COL_WIDTH + 10, 10) is None             # a fifth column does not exist
    assert grid.hit_test_at(2 * EXPECTED_COL_WIDTH + 10,
                            EXPECTED_ROW_HEIGHT + 10) is None                    # row 1, column 2 -> entry 6, absent
    assert grid.hit_test_at(10, 40 * EXPECTED_ROW_HEIGHT) is None                # far below everything


def test_hit_test_follows_the_visible_list_not_the_entry_numbers(make_grid):
    """Positions index the *visible* sequence, so a filtered grid must not select by entry number."""
    grid = make_grid(n_entries=12, width=500)
    grid.set_visible([5, 7, 9])
    assert grid.hit_test_at(10, 10) == 5
    assert grid.hit_test_at(EXPECTED_COL_WIDTH + 10, 10) == 7
    assert grid.hit_test_at(2 * EXPECTED_COL_WIDTH + 10, 10) == 9
    assert grid.hit_test_at(3 * EXPECTED_COL_WIDTH + 10, 10) is None


# --------------------------------------------------------------------------------
# Paging

def test_a_page_is_the_whole_rows_that_fit(make_grid):
    """height 400 / row 134 -> 2 whole rows; a partial third row does not count as visible paging."""
    assert make_grid(height=400)._rows_per_page() == 2
    assert make_grid(height=2 * EXPECTED_ROW_HEIGHT - 1)._rows_per_page() == 1
    assert make_grid(height=10)._rows_per_page() == 1  # never zero: paging must always move


# --------------------------------------------------------------------------------
# Entries, visibility and selection bookkeeping

def test_setting_entries_shows_all_of_them_and_starts_at_the_first(make_grid):
    grid = make_grid(n_entries=5)
    assert grid.visible == [0, 1, 2, 3, 4]
    assert grid.visible_count == 5
    assert grid.current == 0
    assert grid.selected == set()


def test_an_empty_grid_has_no_current_entry(make_grid):
    assert make_grid(n_entries=0).current == -1


def test_selection_operations_act_on_the_visible_set(make_grid):
    grid = make_grid(n_entries=10)
    grid.set_visible([1, 3, 5])
    grid.select_all()
    assert grid.selected == {1, 3, 5}
    grid.invert_selection()
    assert grid.selected == set()
    grid.toggle_select(3)
    assert grid.selected == {3}
    grid.deselect_all()
    assert grid.selected == set()


def test_navigation_walks_the_visible_list(make_grid):
    grid = make_grid(n_entries=12, width=500)  # 4 columns
    grid.set_visible([2, 4, 6, 8, 10])
    grid.set_current(2)
    assert grid.navigate_next() == 4
    assert grid.navigate_row_down() == 10  # +4 positions in the visible list
    assert grid.navigate_first() == 2
    assert grid.navigate_last() == 10


class _RecordingFlasher:
    """Stands in for `ScrollEndFlasher`; only `show` is reached from the grid."""

    def __init__(self):
        self.shown = []

    def show(self, where):
        self.shown.append(where)


def test_navigation_refused_at_an_end_flashes_that_end(make_grid):
    """The gesture a scroll-side trigger cannot see.

    Navigation clamps and `set_current` returns early on an unchanged index, so pressing past the last row
    requests no scroll at all, and the flasher the scroll carries never fires.
    """
    flasher = _RecordingFlasher()
    grid = make_grid(n_entries=12, scroll_end_flasher=flasher)

    grid.navigate_prev()  # already on the first entry
    assert flasher.shown == ["top"]

    grid.navigate_last()  # a move, not a refusal
    grid.navigate_next()  # now there is no further
    assert flasher.shown == ["top", "bottom"]


def test_a_move_does_not_fire_the_refusal_flash(make_grid):
    """A move that succeeded is not a refusal, so this path stays quiet.

    Arriving at an end *is* announced — by the scroll animation, which flashes when it lands on the top or
    bottom. That needs rendered frames to complete and so is not exercised here; what this pins is that the
    refusal path does not double up on it.
    """
    flasher = _RecordingFlasher()
    grid = make_grid(n_entries=12, scroll_end_flasher=flasher)

    grid.navigate_next()
    grid.navigate_row_down()
    grid.navigate_last()  # lands on the last entry, having moved

    assert flasher.shown == []


def test_jumping_to_an_end_already_there_flashes(make_grid):
    """End-at-the-end is as much a refused request as Down-at-the-end."""
    flasher = _RecordingFlasher()
    grid = make_grid(n_entries=12, scroll_end_flasher=flasher)

    grid.navigate_first()  # already there
    grid.navigate_last()
    grid.navigate_last()  # already there

    assert flasher.shown == ["top", "bottom"]


def test_a_grid_without_a_flasher_navigates_normally(make_grid):
    """The flasher is optional, and refusing to move must not depend on having one."""
    grid = make_grid(n_entries=12)

    grid.navigate_prev()

    assert grid.current == 0


def test_replacing_the_entries_drops_the_thumbnails(make_grid):
    """Indices mean something else afterwards, so a kept texture would be a picture of the wrong file."""
    grid = make_grid(n_entries=3)
    grid.set_thumbnail(0, [0.0] * (TILE * TILE * 4))
    assert grid.has_thumbnail(0)
    grid.set_entries(["a", "b", "c"])
    assert not grid.has_thumbnail(0)


def test_a_thumbnail_of_the_wrong_size_is_discarded(make_grid):
    """Arrives when the tile size changed while a decode was in flight; it must not be shown or stored."""
    grid = make_grid(n_entries=3)
    grid.set_thumbnail(0, [0.0] * (TILE * TILE * 4 // 2))
    assert not grid.has_thumbnail(0)


# --------------------------------------------------------------------------------
# The extension hooks

def test_the_draw_hooks_are_called_for_every_tile_drawn(make_grid):
    """Both hooks, because the two exist to sit on opposite sides of the tile's own furniture."""
    seen = {"under": [], "over": []}

    class Decorated(ThumbnailGrid):
        def draw_underlay(self, idx, drawlist_tag):
            seen["under"].append(idx)

        def draw_overlay(self, idx, drawlist_tag):
            seen["over"].append(idx)

    window = dpg.add_window(label="host", tag="host_hooks")
    grid = Decorated(parent=window, width=500, height=400, tile_size=TILE)
    try:
        grid.set_entries(["a", "b", "c"])
        grid.update()  # performs the pending rebuild, which draws every tile
        assert seen["under"] == [0, 1, 2]
        assert seen["over"] == [0, 1, 2]
    finally:
        grid.destroy()
        dpg.delete_item(window)


def test_border_colour_comes_from_the_hook(make_grid):
    class Bordered(ThumbnailGrid):
        def border_color_for(self, idx):
            return (1, 2, 3, 4) if idx == 1 else (9, 9, 9, 9)

    window = dpg.add_window(label="host", tag="host_border")
    grid = Bordered(parent=window, width=500, height=400, tile_size=TILE)
    try:
        assert grid.border_color_for(1) == (1, 2, 3, 4)
        assert grid.border_color_for(0) == (9, 9, 9, 9)
    finally:
        grid.destroy()
        dpg.delete_item(window)
