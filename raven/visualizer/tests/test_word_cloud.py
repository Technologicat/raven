"""Unit tests for raven.visualizer.word_cloud.

Three kinds of decision live here, and none of them needs a rendered window:

  - **when not to render.** The window is usually hidden and the selection usually has not moved, so the
    module has two guards -- `only_if_visible` at submission, and a "same dataset, same selection" check
    inside the worker -- that between them are what keep toggling the window cheap.
  - **what goes into the cloud.** Keyword counts are summed across the selected entries, which is the one
    piece of data handling in the module.
  - **what happens when the render does not finish.** A cancelled task must leave the cache saying the
    texture is stale, and the toolbar must go back to saying "Toggle word cloud window" whatever happened.

As in `test_selection.py`, DPG is a recording stand-in rather than a context: the module's calls are
commands (`show_item`, `set_value`, `set_item_label`) plus one query it is asked to answer
(`is_item_visible`), so what a test can assert is what the module told the GUI to do. `WordCloud` itself
is stubbed -- generating a real one is the expensive thing the guards exist to avoid, and its output is
the library's business rather than Raven's.
"""

import numpy as np

import pytest

word_cloud = pytest.importorskip("raven.visualizer.word_cloud")

from unpythonic import box, unbox  # noqa: E402 -- after importorskip by design
from unpythonic.env import env  # noqa: E402 -- ditto

from raven.visualizer import config as visualizer_config  # noqa: E402 -- ditto
from raven.visualizer.app_state import app_state  # noqa: E402 -- ditto

gui_config = visualizer_config.gui_config

WINDOW = "word_cloud_window"  # tag
TEXTURE = "word_cloud_texture"  # tag
BUTTON = "word_cloud_button"  # tag


class RecordingDPG:
    """Stands in for `dearpygui.dearpygui`, recording what `word_cloud` does to the GUI.

    `visible` is the one piece of GUI state the module reads back, so it is the one a test sets up.
    """
    def __init__(self, real_dpg):
        self._real_dpg = real_dpg
        self.visible = {}  # tag -> bool
        self.shown = []  # tags, in call order
        self.hidden = []  # tags, in call order
        self.labels = {}  # tag -> label, as last set
        self.values = {}  # tag -> value, as last set

    def __getattr__(self, name):
        return getattr(self._real_dpg, name)

    def is_item_visible(self, tag):
        return self.visible.get(tag, False)

    def show_item(self, tag):
        self.shown.append(tag)
        self.visible[tag] = True

    def hide_item(self, tag):
        self.hidden.append(tag)
        self.visible[tag] = False

    def set_item_label(self, tag, label):
        self.labels[tag] = label

    def set_value(self, tag, value):
        self.values[tag] = value


class FakeWordCloud:
    """Stands in for `wordcloud.WordCloud`, recording the frequencies it was asked to render."""
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.frequencies = None
        self.saved_to = None
        FakeWordCloud.instances.append(self)

    def generate_from_frequencies(self, frequencies):
        self.frequencies = dict(frequencies)

    def to_array(self):
        # The shape the worker expects: [h, w, 3], 0..255. A flat mid-grey, so an assertion can tell a
        # rendered texture from a cleared one.
        return np.full([gui_config.word_cloud_h, gui_config.word_cloud_w, 3], 128, dtype=np.uint8)

    def to_file(self, filename):
        self.saved_to = filename


def make_entry(**keywords):
    return env(keywords=keywords)


@pytest.fixture
def gui(monkeypatch):
    """A Visualizer whose word cloud has never been rendered, with the GUI replaced by recorders."""
    fake_dpg = RecordingDPG(word_cloud.dpg)
    monkeypatch.setattr(word_cloud, "dpg", fake_dpg)

    FakeWordCloud.instances = []
    monkeypatch.setattr(word_cloud, "WordCloud", FakeWordCloud)
    monkeypatch.setattr(word_cloud.gui_animation, "flash_button", lambda **kwargs: None)

    # The render cache is module-local and outlives a test, so every test starts with it cold.
    monkeypatch.setattr(word_cloud, "_last_dataset_addr", None)
    monkeypatch.setattr(word_cloud, "_last_data_idxs", set())
    monkeypatch.setattr(word_cloud, "_image_box",
                        box(np.ones([gui_config.word_cloud_h, gui_config.word_cloud_w, 4], dtype=np.float64)))
    monkeypatch.setattr(word_cloud, "_data_box", box(None))

    dataset = env(sorted_entries=[make_entry(laser=3, ablation=1),
                                  make_entry(laser=2, welding=5),
                                  make_entry(photocatalysis=7)])
    monkeypatch.setattr(app_state, "dataset", dataset, raising=False)
    monkeypatch.setattr(app_state, "word_cloud_tooltip", env(text=""), raising=False)
    monkeypatch.setattr(app_state, "word_cloud_save_tooltip", env(text=""), raising=False)

    return env(dpg=fake_dpg, dataset=dataset)


def render(data_idxs, *, cancelled=False, cancel_after=None):
    """Run the render worker directly, standing in for the task manager that normally would.

    `cancel_after`: cancel once this many entries have been visited, for the mid-render case.
    """
    counter = [0]

    class TaskEnv:
        task_name = "test_render"
        data_idxs = None

        @property
        def cancelled(self):
            if cancelled:
                return True
            if cancel_after is None:
                return False
            hit = counter[0] >= cancel_after
            counter[0] += 1
            return hit

    task_env = TaskEnv()
    task_env.data_idxs = data_idxs
    word_cloud._render_worker(task_env=task_env)
    return task_env


def texture_is_blank(gui):
    """Whether the last texture upload was the cleared (black) one."""
    uploaded = np.array(gui.dpg.values[TEXTURE])
    return bool(np.all(uploaded.reshape([-1, 4])[:, :3] == 0.0))


# --------------------------------------------------------------------------------
# When not to render: the submission guard

@pytest.fixture
def submissions(monkeypatch):
    """Replaces the render task manager, so a test can ask whether a render was submitted at all."""
    submitted = []

    class FakeTaskManager:
        def submit(self, task, task_env):
            submitted.append((task, task_env))

    monkeypatch.setattr(word_cloud, "_get_task_manager", FakeTaskManager)
    return submitted


def test_a_hidden_window_is_not_rendered_into_when_the_selection_changes(gui, submissions):
    # This is the common case by a wide margin: the word cloud is off, and every selection change would
    # otherwise pay for a render nobody is looking at.
    gui.dpg.visible[WINDOW] = False
    word_cloud.update([0, 1], only_if_visible=True)
    assert submissions == []


def test_a_visible_window_is_rendered_into_when_the_selection_changes(gui, submissions):
    # Negative control for the test above: with the window up, the same call does submit -- so the guard
    # is reading the window's visibility rather than declining always.
    gui.dpg.visible[WINDOW] = True
    word_cloud.update([0, 1], only_if_visible=True)
    assert len(submissions) == 1


def test_an_explicit_request_renders_even_into_a_hidden_window(gui, submissions):
    # Opening the window is exactly the case where it is not visible yet and a render is wanted.
    gui.dpg.visible[WINDOW] = False
    word_cloud.update([0, 1])
    assert len(submissions) == 1


def test_the_submitted_task_carries_the_selection_and_the_wait_flag(gui, submissions):
    gui.dpg.visible[WINDOW] = False
    word_cloud.update([0, 1], wait=True)
    _, task_env = submissions[0]
    assert list(task_env.data_idxs) == [0, 1]
    assert task_env.wait is True


# --------------------------------------------------------------------------------
# When not to render: the worker's cache

def test_the_same_selection_twice_is_rendered_once(gui):
    render([0, 1])
    assert len(FakeWordCloud.instances) == 1
    render([0, 1])
    assert len(FakeWordCloud.instances) == 1, "an unchanged selection should be shown, not recomputed"
    assert gui.dpg.shown.count(WINDOW) == 2, "...and shown both times, since that is what a toggle asks for"


def test_a_changed_selection_is_rendered_again(gui):
    # Negative control for the test above: it would pass against a worker that renders only once ever.
    render([0, 1])
    render([0, 2])
    assert len(FakeWordCloud.instances) == 2


def test_reordering_the_same_selection_is_not_a_change(gui):
    # The selection arrives as an index array whose order is not meaningful -- the combine modes in
    # `selection` go through Python sets -- so the cache compares sets.
    render([0, 1])
    render([1, 0])
    assert len(FakeWordCloud.instances) == 1


def test_a_new_dataset_with_the_same_selection_is_rendered_again(gui, monkeypatch):
    # Indices mean different items after a file is opened, so the identity of the dataset is half of the
    # cache key. Same indices, different data, and the cloud on screen would otherwise be the old one.
    render([0, 1])
    monkeypatch.setattr(app_state, "dataset",
                        env(sorted_entries=[make_entry(entirely=1), make_entry(different=1)]),
                        raising=False)
    render([0, 1])
    assert len(FakeWordCloud.instances) == 2


# --------------------------------------------------------------------------------
# What goes into the cloud

def test_keyword_counts_are_summed_across_the_selected_entries(gui):
    render([0, 1])
    assert FakeWordCloud.instances[0].frequencies == {"laser": 5, "ablation": 1, "welding": 5}


def test_unselected_entries_contribute_nothing(gui):
    # Negative control for the test above, which selects two of three: the third entry's keyword must be
    # absent, or the sum says nothing about which entries were read.
    render([0, 1])
    assert "photocatalysis" not in FakeWordCloud.instances[0].frequencies


def test_an_empty_selection_clears_the_cloud_rather_than_rendering_one(gui):
    render([])
    assert FakeWordCloud.instances == [], "there is nothing to render, and an empty cloud raises"
    assert texture_is_blank(gui)
    assert WINDOW in gui.dpg.shown


def test_with_no_dataset_loaded_the_cloud_is_cleared(gui, monkeypatch):
    monkeypatch.setattr(app_state, "dataset", None, raising=False)
    render([0, 1])
    assert FakeWordCloud.instances == []
    assert texture_is_blank(gui)


def test_a_rendered_cloud_reaches_the_texture(gui):
    # Negative control for the two above: a worker that cleared the texture unconditionally would satisfy
    # both of them.
    render([0, 1])
    assert not texture_is_blank(gui)


def test_the_rendered_cloud_is_kept_for_saving(gui):
    # `save_to_file` writes whatever was last rendered, so the worker has to leave it somewhere.
    render([0, 1])
    assert unbox(word_cloud._data_box) is FakeWordCloud.instances[0]


# --------------------------------------------------------------------------------
# When the render does not finish

def test_a_task_cancelled_before_starting_touches_nothing(gui):
    render([0, 1], cancelled=True)
    assert FakeWordCloud.instances == []
    assert TEXTURE not in gui.dpg.values
    assert WINDOW not in gui.dpg.shown


def test_a_task_cancelled_partway_leaves_the_cache_cold(gui):
    # The cache says "the texture already shows this selection", so a run that did not get as far as
    # updating the texture must not claim it did -- otherwise the next request for the same selection is
    # answered by showing a window with the *previous* cloud in it.
    render([0, 1], cancel_after=1)
    assert FakeWordCloud.instances == [], "cancelled while collecting keywords, before any rendering"
    render([0, 1])
    assert len(FakeWordCloud.instances) == 1, "the retry must actually render"


def test_the_toolbar_is_restored_even_when_the_render_fails(gui, monkeypatch):
    # The button and its tooltip say "working" for the duration, and a raise on the way through would
    # otherwise leave them saying it forever.
    class ExplodingWordCloud(FakeWordCloud):
        def generate_from_frequencies(self, frequencies):
            raise RuntimeError("word cloud generation exploded")

    monkeypatch.setattr(word_cloud, "WordCloud", ExplodingWordCloud)
    with pytest.raises(RuntimeError):
        render([0, 1])
    assert gui.dpg.labels[WINDOW] == "Word cloud"
    assert app_state.word_cloud_tooltip.text == "Toggle word cloud window [F10]"


# --------------------------------------------------------------------------------
# The window toggle

def test_toggling_a_visible_window_hides_it_without_rendering(gui, submissions):
    gui.dpg.visible[WINDOW] = True
    word_cloud.toggle_window()
    assert gui.dpg.hidden == [WINDOW]
    assert submissions == [], "hiding is not a reason to compute anything"


def test_toggling_a_hidden_window_renders_the_current_selection(gui, submissions, monkeypatch):
    # The window is shown by the worker when it finishes, so the toggle's job is to ask for the render.
    monkeypatch.setattr(app_state, "selection_data_idxs_box", box(np.array([0, 2])), raising=False)
    gui.dpg.visible[WINDOW] = False
    word_cloud.toggle_window()
    assert len(submissions) == 1
    _, task_env = submissions[0]
    assert list(task_env.data_idxs) == [0, 2]


# --------------------------------------------------------------------------------
# Saving

def test_showing_the_save_dialog_puts_the_app_into_modal_mode(gui, monkeypatch):
    # Entering modal mode is what takes the plotter's annotation tooltip off the screen and lifts the
    # info panel's keyboard mark, so the two go together: a dialog shown without it leaves a tooltip
    # floating over the dialog.
    shown = []
    monkeypatch.setattr(app_state, "filedialog_save",
                        env(show_file_dialog=lambda: shown.append("dialog")), raising=False)
    entered = []
    monkeypatch.setattr(app_state, "enter_modal_mode", lambda: entered.append(True), raising=False)
    word_cloud.show_save_dialog()
    assert shown == ["dialog"]
    assert entered == [True]


@pytest.fixture
def closing_the_dialog(monkeypatch):
    """Wires up what `save_callback` touches, and records both of the things it can do."""
    exited = []
    saved = []
    monkeypatch.setattr(app_state, "exit_modal_mode", lambda: exited.append(True), raising=False)
    monkeypatch.setattr(word_cloud, "save_to_file", lambda filename: saved.append(filename))
    return env(exited=exited, saved=saved)


def test_choosing_a_file_saves_to_it(closing_the_dialog):
    word_cloud.save_callback(["/tmp/cloud.png"])
    assert closing_the_dialog.saved == ["/tmp/cloud.png"]
    assert closing_the_dialog.exited == [True]


def test_cancelling_the_dialog_saves_nothing_but_still_leaves_modal_mode(closing_the_dialog):
    # Negative control for the test above, and the case that matters: a cancel that forgot to leave modal
    # mode would lock the plot's input handlers out for the rest of the session.
    word_cloud.save_callback([])
    assert closing_the_dialog.saved == []
    assert closing_the_dialog.exited == [True]


def test_more_than_one_file_is_refused(closing_the_dialog):
    # The dialog is built with `multi_selection=False`, so this cannot happen -- which is the reason it
    # raises rather than picking one: silently saving to whichever came first would hide the day the
    # dialog is rewired.
    with pytest.raises(ValueError):
        word_cloud.save_callback(["/tmp/a.png", "/tmp/b.png"])


def test_saving_writes_the_cloud_that_was_last_rendered(gui, monkeypatch):
    render([0, 1])
    written = []
    monkeypatch.setattr(app_state, "bg", env(submit=lambda task: written.append(task())), raising=False)
    word_cloud.save_to_file("/tmp/cloud.png")
    assert FakeWordCloud.instances[0].saved_to == "/tmp/cloud.png"


# --------------------------------------------------------------------------------
# Odds and ends with a stated contract

def test_the_save_dialog_is_not_visible_before_it_exists(monkeypatch):
    # The hotkey that asks this fires from the first frame, before `initialize_filedialogs` has run.
    monkeypatch.setattr(app_state, "filedialog_save", None, raising=False)
    assert word_cloud.is_save_dialog_visible() is False


def test_the_save_dialog_reports_its_own_visibility_once_it_exists(monkeypatch):
    # Negative control for the test above: the guard is about existence, not a hardcoded False.
    monkeypatch.setattr(app_state, "filedialog_save", env(is_visible=lambda: True), raising=False)
    assert word_cloud.is_save_dialog_visible() is True


def test_clearing_tasks_before_anything_has_been_rendered_is_harmless(monkeypatch):
    # Shutdown runs this whether or not the word cloud was ever opened, and the task manager is created
    # lazily on the first render.
    monkeypatch.setattr(word_cloud, "_task_manager", None)
    word_cloud.clear_tasks()


def test_clearing_tasks_reaches_the_task_manager_once_there_is_one(monkeypatch):
    # Negative control for the test above: the guard skips a missing manager rather than skipping always.
    cleared = []

    class FakeTaskManager:  # not an `env`: `clear` is one of its reserved names
        def clear(self, wait):
            cleared.append(wait)

    monkeypatch.setattr(word_cloud, "_task_manager", FakeTaskManager())
    word_cloud.clear_tasks(wait=True)
    assert cleared == [True]
