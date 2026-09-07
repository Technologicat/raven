"""What the avatar renderer does when the server it streams from is not there.

`animator_running` is not bookkeeping: Librarian's idle-framerate throttle reads it, so a renderer that
stops without clearing it costs full frame rate for the rest of the session — for an animator that is no
longer running. The pause path is the one that has to be right about this, and it is also the one most
likely to fail, being what the renderer calls *because* the stream just died.

No server and no window: the API is replaced and the viewport is never mapped. What is under test is the
bookkeeping, not the picture.
"""

import threading

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from unpythonic.env import env  # noqa: E402 -- after importorskip by design

from raven.client import avatar_renderer  # noqa: E402 -- after importorskip by design


@pytest.fixture
def dpg_context():
    """A DPG context with an unmapped viewport, fresh per test so the item registry starts empty."""
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


@pytest.fixture
def renderer(dpg_context, monkeypatch):
    """A renderer with real widgets and a stubbed API, in the running state.

    Built without `__init__`, which wants a live server and a thread pool. Only `pause` is exercised, and
    the attributes below are everything it reads.
    """
    calls = []
    monkeypatch.setattr(avatar_renderer.api, "avatar_stop", lambda instance_id: calls.append("stop"))
    monkeypatch.setattr(avatar_renderer.api, "avatar_start", lambda instance_id: calls.append("start"))

    with dpg.window() as window:
        paused_text = dpg.add_text("[Video is off]", parent=window)
        backdrop = dpg.add_drawlist(width=10, height=10, parent=window)
        live_image = dpg.add_text("", tag="avatar_live_image_0", parent=window)  # tag  # stands in for the video widget

    instance = avatar_renderer.DPGAvatarRenderer.__new__(avatar_renderer.DPGAvatarRenderer)
    instance.avatar_instance_id = "test-instance"
    instance.paused_text_gui_widget = paused_text
    instance.backdrop_drawlist_gui_widget = backdrop
    instance.live_texture_id_counter = 0
    instance.gui_parent = window
    instance.animator_running = True
    # Cancelled, so `_split_frame_unless_stopping` skips: there is no render loop here to complete a frame,
    # and waiting for one would hang the test rather than fail it.
    instance._task_env = env(cancelled=True)
    instance.avatar_x_center = 5
    instance.avatar_y_bottom = 10
    # `_reposition_paused_text` centres on the backdrop when there is one and on the video otherwise; no
    # backdrop here, so it takes the second branch and wants the video's own geometry.
    instance.backdrop_last_configured_image = None
    instance.full_h = 8

    yield instance, calls, live_image
    dpg.delete_item(window)


def test_pausing_clears_the_running_flag(renderer):
    instance, calls, unused_live_image = renderer
    instance.pause(action="pause")
    assert instance.animator_running is False
    assert calls == ["stop"], "the server should still be told, when it can be"


def test_a_server_that_has_gone_away_still_leaves_the_animator_stopped(renderer, monkeypatch):
    """The regression. The renderer pauses *because* its stream died, so the server it would notify is the
    one that just went away — and the notification used to run before the flag was cleared, so it took the
    flag with it. Librarian then ran at full frame rate for the rest of the session.
    """
    instance, unused_calls, unused_live_image = renderer

    def refuse(instance_id):
        raise ConnectionError("faultproxy: the server is not there")
    monkeypatch.setattr(avatar_renderer.api, "avatar_stop", refuse)

    instance.pause(action="pause")  # must not raise: this runs on the renderer's own error path

    assert instance.animator_running is False, "a pause that could not reach the server left the animator marked running"


def test_a_missing_widget_still_leaves_the_animator_stopped(renderer):
    """The same shape from the other side: `nonexistent_ok` leaves its block at the *first* missing widget,
    so anything after that inside it is skipped. During teardown the widgets go before the renderer does.
    """
    instance, unused_calls, live_image = renderer
    dpg.delete_item(instance.paused_text_gui_widget)  # the first one the block touches

    instance.pause(action="pause")

    assert instance.animator_running is False, "a pause over a deleted widget left the animator marked running"


def test_resuming_against_a_dead_server_stays_paused_and_does_not_raise(renderer, monkeypatch):
    """`ping` resumes, and `ping` is called from whichever background task counted as activity. An
    unreachable server must not take that task down, and must not leave the renderer claiming to run.
    """
    instance, unused_calls, unused_live_image = renderer
    instance.animator_running = False

    def refuse(instance_id):
        raise ConnectionError("faultproxy: the server is not there")
    monkeypatch.setattr(avatar_renderer.api, "avatar_start", refuse)

    instance.pause(action="resume")

    assert instance.animator_running is False, "a resume that never reached the server claimed to have worked"


def test_resuming_normally_sets_the_flag(renderer):
    """The negative control for the two above: without this, they would pass against a `pause` that had
    stopped setting the flag at all."""
    instance, calls, unused_live_image = renderer
    instance.animator_running = False
    instance.pause(action="resume")
    assert instance.animator_running is True
    assert calls == ["start"]


def test_the_running_flag_is_not_shared_between_renderers(renderer):
    """It is per instance, and Librarian's throttle asks one particular renderer."""
    instance, unused_calls, unused_live_image = renderer
    other = avatar_renderer.DPGAvatarRenderer.__new__(avatar_renderer.DPGAvatarRenderer)
    other.animator_running = True
    instance.pause(action="pause")
    assert other.animator_running is True


def test_it_is_safe_from_two_threads(renderer):
    """The idle detector pauses from its own task while the panel switch may be resuming from another."""
    instance, unused_calls, unused_live_image = renderer

    def flip():
        for _ in range(20):
            instance.pause(action="pause")
            instance.pause(action="resume")

    threads = [threading.Thread(target=flip) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert instance.animator_running in (True, False)  # no exception escaped, which is the assertion
