"""Unit tests for raven.visualizer.annotation.

The plotter's mouse-hover tooltip. Most of the module is one long background worker that builds DPG
widgets, waits for the window to autosize, and swaps the new content in -- that part needs a running app,
not just a context, because it calls `split_frame` from a background thread and reads back geometry that
only exists once frames have been rendered. It is not tested here.

What is tested is everything that decides *whether* to show anything, which is where the module's
behaviour lives from a user's point of view:

  - `update`, the submitter, which also updates the plot highlight immediately rather than waiting for the
    background task -- and hides the tooltip the moment the mouse moves, so that the user can move onto
    where it was.
  - the worker's three early exits: a modal is up, the mouse has left the plot, or there is nothing under
    the cursor. Each of them has to hide the tooltip and build nothing.

The stand-in below raises if anything tries to build a widget, so a guard that stopped guarding fails
loudly rather than by wandering into DPG calls that would need a context.
"""

import numpy as np

import pytest

annotation = pytest.importorskip("raven.visualizer.annotation")

from unpythonic.env import env  # noqa: E402 -- after importorskip by design

from raven.visualizer.app_state import app_state  # noqa: E402 -- ditto

TOOLTIP = "annotation_tooltip_window"  # tag
HOVER_SERIES = "my_mouse_hover_scatter_series"  # tag


class RecordingDPG:
    """Stands in for `dearpygui.dearpygui`, recording what `annotation` does before it starts building.

    Every call the content build makes first raises instead, because every test here is about a path
    that must *not* reach that build. `split_frame` is on that list for a blunter reason than `add_group`:
    delegated to the real toolkit with no context, it does not fail, it segfaults -- so the difference
    between a red test and a dead test process is naming it here. (Learned by defeating the guard on
    purpose, which is what these tests are checked with.)
    """
    def __init__(self, real_dpg):
        self._real_dpg = real_dpg
        self.mouse_pos = [100.0, 100.0]
        self.hidden = []  # tags, in call order
        self.shown = []  # tags, in call order
        self.values = {}  # tag -> value, as last set

    def __getattr__(self, name):
        return getattr(self._real_dpg, name)

    def get_mouse_pos(self, local=True):
        return list(self.mouse_pos)

    def hide_item(self, tag):
        self.hidden.append(tag)

    def show_item(self, tag):
        self.shown.append(tag)

    def set_value(self, tag, value):
        self.values[tag] = value

    def split_frame(self, *args, **kwargs):
        raise AssertionError("the tooltip content build was reached; these tests are all about not reaching it")

    def add_group(self, *args, **kwargs):
        raise AssertionError("the tooltip content build was reached; these tests are all about not reaching it")

    def reset_log(self):
        self.hidden.clear()
        self.shown.clear()
        self.values.clear()


@pytest.fixture
def gui(monkeypatch):
    """A Visualizer with a dataset loaded, the mouse over the plot, and no modal window open."""
    fake_dpg = RecordingDPG(annotation.dpg)
    monkeypatch.setattr(annotation, "dpg", fake_dpg)

    dataset = env(sorted_lowdim_data=np.array([[float(i), float(10 * i)] for i in range(6)]))
    monkeypatch.setattr(app_state, "dataset", dataset, raising=False)
    monkeypatch.setattr(app_state, "is_any_modal_window_visible", lambda: False, raising=False)
    monkeypatch.setattr(app_state, "mouse_inside_plot_widget", lambda: True, raising=False)

    # What the plotter reports under the cursor. The decisions under test are this module's, not the
    # nearest-neighbour search's, which `test_plotter.py` covers.
    at_mouse = {"idxs": np.array([1, 3])}
    monkeypatch.setattr(annotation.plotter, "get_data_idxs_at_mouse",
                        lambda dataset=None: at_mouse["idxs"])

    submitted = []

    class FakeTaskManager:
        def submit(self, task, task_env):
            submitted.append((task, task_env))

    monkeypatch.setattr(annotation, "_get_task_manager", FakeTaskManager)

    # `data_idxs` is module-level and read by `app.py`'s right-click handler, so it outlives a test.
    monkeypatch.setattr(annotation, "data_idxs", [])

    return env(dpg=fake_dpg, dataset=dataset, at_mouse=at_mouse, submitted=submitted)


def prime_mouse_at(gui, position):
    """Run one `update` so the module's remembered mouse position is `position`, then forget what it did.

    `update` remembers the last mouse position in a `@dlet` env -- let over def, so the env is a closure
    cell created once and shared by every call, for the life of the function object. "The mouse has not
    moved since last time" is therefore a state that outlives a test, and one a test has to establish by
    calling rather than by assignment: the cell can be reached through `__closure__`, but that is not the
    function's API and a test has no business writing through it. Priming is also what makes these tests
    independent of the order they run in, which they otherwise would not be.
    """
    gui.dpg.mouse_pos = list(position)
    annotation.update()
    gui.submitted.clear()
    gui.dpg.reset_log()


# --------------------------------------------------------------------------------
# The plot highlight, which `update` does immediately rather than in the background

def test_the_datapoints_under_the_cursor_are_highlighted_at_once(gui):
    # The tooltip can afford to arrive a frame or two late; the highlight cannot, since it is the
    # feedback that says the plot is tracking the mouse at all.
    prime_mouse_at(gui, (100.0, 100.0))
    gui.dpg.mouse_pos = [120.0, 100.0]
    annotation.update()
    xs, ys = gui.dpg.values[HOVER_SERIES]
    assert sorted(xs) == [1.0, 3.0]
    assert sorted(ys) == [10.0, 30.0]


def test_the_highlight_is_cleared_when_the_cursor_is_over_nothing(gui):
    prime_mouse_at(gui, (100.0, 100.0))
    gui.at_mouse["idxs"] = np.array([], dtype=np.int64)
    gui.dpg.mouse_pos = [120.0, 100.0]
    annotation.update()
    assert gui.dpg.values[HOVER_SERIES] == [[], []]


def test_the_highlight_is_cleared_when_no_dataset_is_loaded(gui, monkeypatch):
    # The plot is empty, but its series still exist, so something has to say so.
    prime_mouse_at(gui, (100.0, 100.0))
    monkeypatch.setattr(app_state, "dataset", None, raising=False)
    gui.dpg.mouse_pos = [120.0, 100.0]
    annotation.update()
    assert gui.dpg.values[HOVER_SERIES] == [[], []]


def test_a_mouse_that_has_not_moved_does_not_redraw_the_highlight(gui):
    # It would be the same points in the same places, and this runs on every mouse event.
    prime_mouse_at(gui, (100.0, 100.0))
    annotation.update()
    assert HOVER_SERIES not in gui.dpg.values


def test_forcing_an_update_redraws_the_highlight_though_the_mouse_is_still(gui):
    # Negative control for the test above, and what `force` is for: the mouse wheel zooms the plot, so
    # the points under a stationary cursor change without any mouse movement to notice.
    prime_mouse_at(gui, (100.0, 100.0))
    annotation.update(force=True)
    assert HOVER_SERIES in gui.dpg.values


# --------------------------------------------------------------------------------
# Hiding the tooltip on movement

def test_moving_the_mouse_hides_the_tooltip(gui):
    # So that the user can move the cursor onto where the tooltip was. A window swallows the mouse across
    # its whole rect, and `get_plot_mouse_pos` then stops reporting plot coordinates.
    prime_mouse_at(gui, (100.0, 100.0))
    gui.dpg.mouse_pos = [120.0, 100.0]
    annotation.update()
    assert TOOLTIP in gui.dpg.hidden


def test_forcing_an_update_without_moving_the_mouse_leaves_the_tooltip_up(gui):
    # Negative control for the test above: the two conditions are separate, and `force` is not movement.
    # Hiding here would make the wheel flicker the tooltip away on every notch.
    prime_mouse_at(gui, (100.0, 100.0))
    annotation.update(force=True)
    assert TOOLTIP not in gui.dpg.hidden


# --------------------------------------------------------------------------------
# Submitting the rebuild

def test_every_update_submits_a_rebuild(gui):
    prime_mouse_at(gui, (100.0, 100.0))
    annotation.update()
    assert len(gui.submitted) == 1


def test_the_submitted_task_carries_the_wait_flag(gui):
    # The caller knows whether more input is likely -- mouse movement usually comes in bursts -- and a
    # wait is what lets a burst collapse into one rebuild.
    prime_mouse_at(gui, (100.0, 100.0))
    annotation.update(wait=False)
    assert gui.submitted[0][1].wait is False
    annotation.update(wait=True)
    assert gui.submitted[1][1].wait is True


# --------------------------------------------------------------------------------
# The worker's early exits

def run_worker():
    """Run the render worker directly, standing in for the task manager."""
    task_env = env(task_name="test_render", cancelled=False)
    annotation._render_worker(task_env=task_env)
    return task_env


def test_no_tooltip_is_built_while_a_modal_window_is_open(gui, monkeypatch):
    # The rest of the GUI is meant to be inactive behind a modal, and a tooltip is drawn over everything.
    monkeypatch.setattr(app_state, "is_any_modal_window_visible", lambda: True, raising=False)
    run_worker()
    assert TOOLTIP in gui.dpg.hidden


def test_no_tooltip_is_built_once_the_mouse_has_left_the_plot(gui, monkeypatch):
    # The task waits in a queue before it runs, so by the time it does, the cursor it was submitted for
    # may be somewhere else entirely.
    monkeypatch.setattr(app_state, "mouse_inside_plot_widget", lambda: False, raising=False)
    run_worker()
    assert TOOLTIP in gui.dpg.hidden


def test_no_tooltip_is_built_when_the_cursor_is_over_nothing(gui):
    gui.at_mouse["idxs"] = np.array([], dtype=np.int64)
    run_worker()
    assert TOOLTIP in gui.dpg.hidden


def test_the_list_of_shown_items_is_emptied_when_the_cursor_is_over_nothing(gui):
    # `data_idxs` says which items the tooltip is currently listing, and `app.py`'s right-click handler
    # reads it to decide whether the click can scroll the info panel to one of them. Left stale, a right
    # click acts on items that are no longer on screen.
    annotation.data_idxs.extend([1, 3])
    gui.at_mouse["idxs"] = np.array([], dtype=np.int64)
    run_worker()
    assert annotation.data_idxs == []


def test_the_guards_run_before_anything_is_built(gui, monkeypatch):
    # The stand-in raises from `add_group`, so this is really asserting that the three tests above are
    # testing the guards rather than a build that happens to be harmless. Stated once, explicitly,
    # because it is the property that makes this whole module testable without a DPG context.
    monkeypatch.setattr(app_state, "is_any_modal_window_visible", lambda: True, raising=False)
    run_worker()  # would raise AssertionError from `RecordingDPG.add_group`
    monkeypatch.setattr(app_state, "is_any_modal_window_visible", lambda: False, raising=False)
    gui.at_mouse["idxs"] = np.array([], dtype=np.int64)
    run_worker()


# --------------------------------------------------------------------------------
# Clearing

def test_clearing_the_hover_hides_the_tooltip_and_the_highlight(gui):
    # Both halves: the tooltip is a window and the highlight is a scatter series, and leaving either
    # behind leaves the plot claiming the cursor is somewhere it is not.
    annotation.clear_mouse_hover()
    assert TOOLTIP in gui.dpg.hidden
    assert gui.dpg.values[HOVER_SERIES] == [[], []]


def test_clearing_tasks_before_anything_has_been_rendered_is_harmless(monkeypatch):
    # Shutdown runs this whether or not the plot was ever hovered, and the task manager is created lazily.
    monkeypatch.setattr(annotation, "_task_manager", None)
    annotation.clear_tasks()


def test_clearing_tasks_reaches_the_task_manager_once_there_is_one(monkeypatch):
    # Negative control for the test above: the guard skips a missing manager rather than skipping always.
    cleared = []

    class FakeTaskManager:  # not an `env`: `clear` is one of its reserved names
        def clear(self, wait):
            cleared.append(wait)

    monkeypatch.setattr(annotation, "_task_manager", FakeTaskManager())
    annotation.clear_tasks(wait=True)
    assert cleared == [True]
