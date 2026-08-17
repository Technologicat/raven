"""Tests for raven.common.gui.tablecursor.

The cursor's state machine needs no DPG context: with no scroll target and a recording paint callback,
every rule here can be stated directly. Scrolling and painting are the parts that need a real table, and
they are exercised where the dialog uses them.

The rules under test are shared with `FileGrid` through `gridnav.reanchor_cursor`, which is the point —
the file dialog shows one listing through two widgets, and a keypress has to mean the same thing in both.
"""

import inspect

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui")

from raven.common.gui.tablecursor import TableCursor  # noqa: E402 -- after importorskip by design
from raven.common.gui.thumbnailgrid import ThumbnailGrid  # noqa: E402 -- ditto


@pytest.fixture
def painted():
    """Records `(idx, is_cursor)` in call order, so a test can say which rows were repainted."""
    return []


@pytest.fixture
def make_cursor(painted):
    def factory(**kwargs):
        return TableCursor(on_paint=lambda idx, is_cursor: painted.append((idx, is_cursor)), **kwargs)
    return factory


# --------------------------------------------------------------------------------
# The interface itself

def test_the_table_cursor_answers_to_the_grid_s_navigation_interface():
    """The duck-typing this class exists for, pinned so a method added to one side is noticed on the other.

    The dialog's key handler picks a navigator once — the grid or this — and then calls the same names. A
    method appearing on the grid without appearing here would strand a key in one view only, which is
    exactly the kind of gap that reads as "the keyboard is flaky in thumbnail mode".
    """
    def nav_names(cls):
        return {name for name, _ in inspect.getmembers(cls, inspect.isfunction) if name.startswith("navigate_")}
    assert nav_names(ThumbnailGrid) == nav_names(TableCursor)


# --------------------------------------------------------------------------------
# Where the cursor starts and how it moves

def test_a_fresh_listing_starts_at_the_top(make_cursor):
    cursor = make_cursor()
    cursor.set_listing(["a", "b", "c"])
    assert cursor.current == 0
    assert cursor.current_key == "a"


def test_an_empty_listing_has_no_cursor(make_cursor):
    cursor = make_cursor()
    cursor.set_listing([])
    assert cursor.current == -1
    assert cursor.current_key is None


def test_navigation_moves_one_entry_at_a_time(make_cursor):
    cursor = make_cursor()
    cursor.set_listing(["a", "b", "c"])
    cursor.navigate_next()
    assert cursor.current_key == "b"
    cursor.navigate_prev()
    assert cursor.current_key == "a"


def test_navigation_clamps_at_both_ends(make_cursor):
    """Rather than wrapping. A file listing has a top and a bottom, and arriving at one should say so by
    staying there — wrapping from the last file to the first reads as the cursor having been lost."""
    cursor = make_cursor()
    cursor.set_listing(["a", "b"])
    cursor.navigate_prev()
    assert cursor.current == 0
    cursor.navigate_last()
    cursor.navigate_next()
    assert cursor.current == 1


def test_first_and_last_go_to_the_ends(make_cursor):
    cursor = make_cursor()
    cursor.set_listing(["a", "b", "c"])
    cursor.navigate_last()
    assert cursor.current_key == "c"
    cursor.navigate_first()
    assert cursor.current_key == "a"


def test_paging_without_a_measurable_view_still_moves(make_cursor):
    """The fallback matters: a page key that does nothing is worse than one that moves a single row."""
    cursor = make_cursor()
    cursor.set_listing(["a", "b", "c"])
    assert cursor.rows_per_page() == 1
    cursor.navigate_page_down()
    assert cursor.current == 1


# --------------------------------------------------------------------------------
# Surviving a rebuild — the rules shared with FileGrid

def test_the_cursor_follows_its_entry_across_a_re_sort(make_cursor):
    cursor = make_cursor()
    cursor.set_listing(["a", "b", "c"], listing_key="/dir")
    cursor.navigate_last()
    cursor.set_listing(["c", "b", "a"], listing_key="/dir")
    assert cursor.current_key == "c"


def test_the_cursor_keeps_its_place_when_its_entry_is_gone(make_cursor):
    cursor = make_cursor()
    cursor.set_listing(["a", "b", "c"], listing_key="/dir")
    cursor.navigate_next()
    cursor.set_listing(["a", "c"], listing_key="/dir")
    assert cursor.current_key == "c"


def test_the_cursor_goes_home_when_its_entry_comes_back(make_cursor):
    """Type a character that filters the cursor's file out, erase it, and the cursor returns.

    While the entry is missing the cursor holds its position, so a different entry sits under it — but the
    user never chose that one, the list moved under a stationary cursor.
    """
    cursor = make_cursor()
    cursor.set_listing(["a", "b", "c"], listing_key="/dir")
    cursor.set_current(0)

    cursor.set_listing(["b", "c"], listing_key="/dir")
    assert cursor.current_key == "b"

    cursor.set_listing(["a", "b", "c"], listing_key="/dir")
    assert cursor.current_key == "a"


def test_moving_the_cursor_adopts_the_new_entry(make_cursor):
    """The escape hatch: a deliberate move re-anchors, so the cursor stops trying to go home."""
    cursor = make_cursor()
    cursor.set_listing(["a", "b", "c"], listing_key="/dir")
    cursor.set_current(0)
    cursor.set_listing(["b", "c"], listing_key="/dir")
    cursor.navigate_next()  # deliberate: the user chose "c"

    cursor.set_listing(["a", "b", "c"], listing_key="/dir")
    assert cursor.current_key == "c"


def test_a_different_listing_starts_at_the_top(make_cursor):
    cursor = make_cursor()
    cursor.set_listing(["a", "b", "c"], listing_key="/here")
    cursor.navigate_last()
    cursor.set_listing(["x", "y", "z"], listing_key="/elsewhere")
    assert cursor.current_key == "x"


# --------------------------------------------------------------------------------
# Painting

def test_moving_repaints_the_row_left_and_the_row_arrived_at(make_cursor, painted):
    """Two rows, not the whole table: a listing can be thousands of rows and the cursor touches two."""
    cursor = make_cursor()
    cursor.set_listing(["a", "b", "c"])
    painted.clear()
    cursor.navigate_next()
    assert painted == [(0, False), (1, True)]


def test_a_paint_failure_does_not_stop_the_cursor():
    """A row that cannot be painted — deleted mid-rebuild, say — must not leave the cursor stuck."""
    def explode(idx, is_cursor):
        raise RuntimeError("no such row")
    cursor = TableCursor(on_paint=explode)
    cursor.set_listing(["a", "b"])
    cursor.navigate_next()
    assert cursor.current == 1


def test_the_owner_is_told_where_the_cursor_went(make_cursor):
    seen = []
    cursor = TableCursor(on_paint=lambda idx, is_cursor: None,
                         on_current_changed=seen.append)
    cursor.set_listing(["a", "b"])
    cursor.navigate_next()
    assert seen == [0, 1]
