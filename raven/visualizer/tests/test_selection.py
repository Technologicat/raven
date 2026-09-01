"""Unit tests for raven.visualizer.selection.

The selection algebra and the undo history are set operations over index arrays, so they can be stated
without a plot to draw them on -- which is the reason this module exists apart from the rest of the
Visualizer's GUI layer.

No DPG context is created here. `selection`'s DPG use is four calls (`enable_item`, `disable_item`,
`set_value`, `is_key_down`), so a stand-in records what the module *asked the GUI to do*, which is the
Raven-side decision worth pinning -- whether a disabled button then greys out is DPG's business. The
stand-in delegates everything else to the real module, so the key constants a test compares against are
DPG's own rather than numbers copied into a test.

The cross-module handshakes are stubbed the same way, since each of them is a whole subsystem: the info
panel, the plotter tooltip and the word cloud are told to refresh, and what the tests assert is that they
were told, with which arguments.
"""

import numpy as np

import pytest

selection = pytest.importorskip("raven.visualizer.selection")

from unpythonic import unbox  # noqa: E402 -- after importorskip by design
from unpythonic.env import env  # noqa: E402 -- ditto

from raven.visualizer.app_state import app_state  # noqa: E402 -- ditto

UNDO_BUTTON = "selection_undo_button"  # tag
REDO_BUTTON = "selection_redo_button"  # tag
HIGHLIGHT_SERIES = "my_selection_scatter_series"  # tag


class RecordingDPG:
    """Stands in for `dearpygui.dearpygui`, recording what `selection` does to the GUI.

    Anything not overridden here comes from the real module, so `mvKey_LShift` and friends are the
    toolkit's own constants: a test that pressed a keycode of its own invention would agree with the
    code under test about a number neither of them shares with DPG.
    """
    def __init__(self, real_dpg):
        self._real_dpg = real_dpg
        self.enabled = {}  # tag -> bool, as last set
        self.values = {}  # tag -> value, as last set
        self.keys_down = set()

    def __getattr__(self, name):
        return getattr(self._real_dpg, name)

    def enable_item(self, tag):
        self.enabled[tag] = True

    def disable_item(self, tag):
        self.enabled[tag] = False

    def set_value(self, tag, value):
        self.values[tag] = value

    def is_key_down(self, key):
        return key in self.keys_down


class Recorder:
    """Records the calls one of the cross-module GUI updaters received."""
    def __init__(self):
        self.calls = []  # list of `(args, kwargs)`

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    @property
    def called(self):
        return bool(self.calls)

    @property
    def last_kwargs(self):
        return self.calls[-1][1]


@pytest.fixture
def gui(monkeypatch):
    """A Visualizer with a dataset loaded, its GUI replaced by recorders, and a blank undo history.

    Yields the recorders, so a test can ask what the selection asked the rest of the app to do.
    """
    fake_dpg = RecordingDPG(selection.dpg)
    monkeypatch.setattr(selection, "dpg", fake_dpg)

    # Six data points on a diagonal, so an assertion about the highlight can name which ones it got.
    dataset = env(sorted_lowdim_data=np.array([[float(i), float(10 * i)] for i in range(6)]))
    monkeypatch.setattr(app_state, "dataset", dataset, raising=False)

    info_panel = Recorder()
    mouse_hover = Recorder()
    word_cloud = Recorder()
    monkeypatch.setattr(app_state, "update_info_panel", info_panel, raising=False)
    monkeypatch.setattr(app_state, "update_mouse_hover", mouse_hover, raising=False)
    monkeypatch.setattr(selection.word_cloud, "update", word_cloud)

    # The undo stack is module-local and outlives a test, so every test starts from a blank history.
    selection.reset_undo_history()

    return env(dpg=fake_dpg,
               dataset=dataset,
               info_panel=info_panel,
               mouse_hover=mouse_hover,
               word_cloud=word_cloud)


def current_selection():
    """The current selection as a set, since the combine modes go through `set` and do not promise order."""
    return set(int(i) for i in unbox(app_state.selection_data_idxs_box))


# --------------------------------------------------------------------------------
# The four combine modes

def test_replace_swaps_the_selection_wholesale(gui):
    selection.update([0, 1, 2])
    assert selection.update([3, 4], mode="replace") is True
    assert current_selection() == {3, 4}


def test_add_unions(gui):
    selection.update([0, 1])
    assert selection.update([1, 2], mode="add") is True
    assert current_selection() == {0, 1, 2}


def test_subtract_removes(gui):
    selection.update([0, 1, 2])
    assert selection.update([1, 5], mode="subtract") is True, "one of the two was selected, so this changes things"
    assert current_selection() == {0, 2}


def test_intersect_keeps_what_the_two_have_in_common(gui):
    selection.update([0, 1, 2])
    assert selection.update([2, 3, 4], mode="intersect") is True
    assert current_selection() == {2}


# --------------------------------------------------------------------------------
# When a mode decides nothing changed
#
# Each of these is the negative control for its counterpart above: the same mode, the same call shape,
# and the only difference is whether the operation would move any point in or out of the selection. A
# suite with only the four tests above would pass against a `selection.update` that never short-circuits.

def test_replacing_a_selection_with_itself_changes_nothing(gui):
    selection.update([0, 1, 2])
    gui.info_panel.calls.clear()
    assert selection.update([2, 1, 0], mode="replace") is False, "the same set in another order is the same set"
    assert not gui.info_panel.called, "an unchanged selection must not spend an info panel rebuild"


def test_adding_points_that_are_already_selected_changes_nothing(gui):
    selection.update([0, 1, 2])
    gui.info_panel.calls.clear()
    assert selection.update([1, 2], mode="add") is False
    assert not gui.info_panel.called


def test_subtracting_points_that_are_not_selected_changes_nothing(gui):
    selection.update([0, 1, 2])
    gui.info_panel.calls.clear()
    assert selection.update([4, 5], mode="subtract") is False
    assert not gui.info_panel.called


def test_intersecting_with_a_superset_changes_nothing(gui):
    selection.update([0, 1])
    gui.info_panel.calls.clear()
    assert selection.update([0, 1, 2, 3], mode="intersect") is False
    assert not gui.info_panel.called


def test_force_refreshes_the_gui_even_when_the_selection_stands_still(gui):
    # Loading a new dataset reuses the same indices while the data under them changes, so the GUI has to
    # be rebuilt for a selection that compares equal to the one already there.
    selection.update([0, 1, 2])
    gui.info_panel.calls.clear()
    assert selection.update([0, 1, 2], mode="replace", force=True) is True
    assert gui.info_panel.called


# --------------------------------------------------------------------------------
# Scroll anchoring

def test_the_items_surviving_a_selection_change_become_scroll_anchors(gui):
    # The info panel scrolls to keep something the reader was already looking at in view, so what it
    # needs is the overlap between the old selection and the new.
    selection.update([0, 1, 2])
    selection.update([2, 3, 4])
    assert app_state.selection_anchor_data_idxs_set == {2}
    assert app_state.selection_changed is True


def test_a_disjoint_selection_offers_no_anchors(gui):
    selection.update([0, 1])
    selection.update([4, 5])
    assert app_state.selection_anchor_data_idxs_set == set()


# --------------------------------------------------------------------------------
# The highlight series

def test_the_highlight_gets_the_coordinates_of_the_selected_points(gui):
    selection.update([1, 3])
    xs, ys = gui.dpg.values[HIGHLIGHT_SERIES]
    assert sorted(xs) == [1.0, 3.0]
    assert sorted(ys) == [10.0, 30.0]


def test_an_empty_selection_clears_the_highlight(gui):
    selection.update([1, 3])
    selection.update([])
    assert gui.dpg.values[HIGHLIGHT_SERIES] == [[], []]


# --------------------------------------------------------------------------------
# Undo history

def test_a_fresh_history_offers_neither_undo_nor_redo(gui):
    assert gui.dpg.enabled[UNDO_BUTTON] is False
    assert gui.dpg.enabled[REDO_BUTTON] is False


def test_undoing_at_the_start_of_history_does_nothing(gui):
    selection.undo()
    assert current_selection() == set()
    assert not gui.info_panel.called, "there was nothing to undo, so nothing should have been rebuilt"


def test_a_selection_change_offers_an_undo_but_not_yet_a_redo(gui):
    selection.update([0, 1])
    assert gui.dpg.enabled[UNDO_BUTTON] is True
    assert gui.dpg.enabled[REDO_BUTTON] is False


def test_undo_restores_exactly_the_previous_selection(gui):
    selection.update([0, 1, 2])
    selection.update([3, 4], mode="replace")
    selection.undo()
    assert current_selection() == {0, 1, 2}


def test_redo_puts_back_what_undo_took(gui):
    selection.update([0, 1, 2])
    selection.update([3, 4], mode="replace")
    selection.undo()
    selection.redo()
    assert current_selection() == {3, 4}


def test_walking_back_to_the_start_disables_undo(gui):
    selection.update([0, 1])
    selection.undo()
    assert current_selection() == set()
    assert gui.dpg.enabled[UNDO_BUTTON] is False
    assert gui.dpg.enabled[REDO_BUTTON] is True


def test_walking_forward_to_the_end_disables_redo(gui):
    selection.update([0, 1])
    selection.undo()
    selection.redo()
    assert gui.dpg.enabled[REDO_BUTTON] is False
    assert gui.dpg.enabled[UNDO_BUTTON] is True


def test_redoing_at_the_end_of_history_does_nothing(gui):
    selection.update([0, 1])
    gui.info_panel.calls.clear()
    selection.redo()
    assert current_selection() == {0, 1}
    assert not gui.info_panel.called


def test_selecting_something_new_after_an_undo_discards_the_redo_branch(gui):
    selection.update([0, 1])
    selection.update([2, 3])
    selection.undo()  # back to {0, 1}, with {2, 3} available to redo
    selection.update([4, 5])  # ...and now it is not
    assert gui.dpg.enabled[REDO_BUTTON] is False
    selection.redo()
    assert current_selection() == {4, 5}, "the abandoned branch must not be reachable going forward"
    # Nor backward, which is the assertion that needs making: a history that merely *appended* the new
    # selection also leaves the cursor at the end with redo disabled, so everything above agrees with it.
    # The difference shows one step back, where the abandoned {2, 3} would be sitting.
    selection.undo()
    assert current_selection() == {0, 1}, "undo should reach the selection that {4, 5} replaced"


def test_undo_walks_back_through_several_steps(gui):
    # Negative control for the stack arithmetic: with a single entry, "step back one" and "go to the
    # start" are the same move, so a one-change history cannot tell a cursor from a reset.
    selection.update([0])
    selection.update([1])
    selection.update([2])
    selection.undo()
    assert current_selection() == {1}
    selection.undo()
    assert current_selection() == {0}


def test_committing_an_unchanged_selection_is_a_no_op(gui):
    selection.update([0, 1])
    assert selection.commit_change_to_undo_history() is False
    selection.undo()
    assert current_selection() == set(), "the redundant commit must not have added a step to walk back through"


def test_an_uncommitted_selection_change_leaves_no_undo_step(gui):
    # Mouse-draw select updates the selection on every movement and commits once, on button release --
    # otherwise dragging across the plot would leave one undo entry per frame.
    selection.update([0, 1], update_selection_undo_history=False)
    assert current_selection() == {0, 1}
    selection.undo()
    assert current_selection() == {0, 1}, "an uncommitted change is not on the undo stack, so there is nothing to undo"


def test_committing_afterwards_makes_the_whole_drag_one_undo_step(gui):
    selection.update([0], update_selection_undo_history=False)
    selection.update([0, 1], update_selection_undo_history=False)
    selection.update([0, 1, 2], update_selection_undo_history=False)
    assert selection.commit_change_to_undo_history() is True
    selection.undo()
    assert current_selection() == set(), "the drag should undo in one step, not three"


def test_resetting_the_history_clears_the_selection_and_both_buttons(gui):
    # What happens when a new dataset is opened: the indices in the old selection mean different items
    # now, so the history is not carried over.
    selection.update([0, 1])
    selection.reset_undo_history()
    assert current_selection() == set()
    assert gui.dpg.enabled[UNDO_BUTTON] is False
    assert gui.dpg.enabled[REDO_BUTTON] is False
    selection.undo()
    assert current_selection() == set()


# --------------------------------------------------------------------------------
# Who gets told, and whether they are asked to wait

def test_a_selection_change_tells_the_info_panel_the_tooltip_and_the_word_cloud(gui):
    selection.update([0, 1])
    assert gui.info_panel.called
    assert gui.mouse_hover.called
    assert gui.word_cloud.called
    # The word cloud is expensive and usually not on screen, so it is told to skip the work when hidden.
    assert gui.word_cloud.last_kwargs["only_if_visible"] is True


def test_wait_is_passed_on_to_everything_the_selection_refreshes(gui):
    # Mouse-draw select asks for `wait=True` so that a drag does not start a panel rebuild per frame.
    # It is the caller's choice, so it has to reach all three of them or the drag stutters on whichever
    # one was left out.
    selection.update([0, 1], wait=True)
    assert gui.info_panel.last_kwargs["wait"] is True
    assert gui.mouse_hover.last_kwargs["wait"] is True
    assert gui.word_cloud.last_kwargs["wait"] is True


def test_an_ordinary_selection_change_asks_nobody_to_wait(gui):
    # Negative control for the test above: `wait` is forwarded rather than always set.
    selection.update([0, 1])
    assert gui.info_panel.last_kwargs["wait"] is False
    assert gui.mouse_hover.last_kwargs["wait"] is False
    assert gui.word_cloud.last_kwargs["wait"] is False


def test_undo_and_redo_always_ask_the_updaters_to_wait(gui):
    # Undo is a button and a hotkey, both of which get hammered, so each step defers its expensive work
    # until the user stops walking through the history.
    selection.update([0, 1])
    selection.update([2, 3])
    selection.undo()
    assert gui.info_panel.last_kwargs["wait"] is True
    assert gui.word_cloud.last_kwargs["wait"] is True
    selection.redo()
    assert gui.info_panel.last_kwargs["wait"] is True
    assert gui.word_cloud.last_kwargs["wait"] is True


# --------------------------------------------------------------------------------
# Keyboard modifiers → combine mode

def test_no_modifiers_replace_the_selection(gui):
    assert selection.keyboard_state_to_mode() == "replace"


def test_shift_adds(gui):
    gui.dpg.keys_down = {gui.dpg.mvKey_LShift}
    assert selection.keyboard_state_to_mode() == "add"


def test_ctrl_subtracts(gui):
    gui.dpg.keys_down = {gui.dpg.mvKey_RControl}
    assert selection.keyboard_state_to_mode() == "subtract"


def test_shift_and_ctrl_together_intersect(gui):
    gui.dpg.keys_down = {gui.dpg.mvKey_LShift, gui.dpg.mvKey_LControl}
    assert selection.keyboard_state_to_mode() == "intersect"


def test_either_side_of_a_modifier_key_counts(gui):
    # Negative control for the three above, each of which presses one particular key: the rule is about
    # the modifier, not about which of its two keys the hand happened to reach.
    gui.dpg.keys_down = {gui.dpg.mvKey_RShift}
    assert selection.keyboard_state_to_mode() == "add"
    gui.dpg.keys_down = {gui.dpg.mvKey_LControl}
    assert selection.keyboard_state_to_mode() == "subtract"
