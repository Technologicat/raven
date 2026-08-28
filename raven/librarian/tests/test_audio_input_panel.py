"""Tests for `raven.librarian.audio_input_panel`.

The panel is DPG, but its logic is not: what a control does when it moves, what the *loudest
recently* reading is, and what *Measure the room* computes from it. Those run headless, against a
stub in place of the audio device — `Recorder` opens one on construction, and neither CI nor an
unattended dev run has a microphone to give it.

The stub is the recorder's *settings* surface, which is all the panel touches: three attributes it
reads and writes, the three state predicates, and the connect/start/stop calls. Anything the panel
did that needed real audio would fail here rather than passing quietly, since the stub captures no
sound.
"""

import collections
import inspect
import math

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed (GUI toolkit absent in CI)")

from raven.common.audio import recorder as audio_recorder  # noqa: E402 -- after importorskip by design
from raven.common.audio import silencegate  # noqa: E402 -- ditto
from raven.common.gui import utils as guiutils  # noqa: E402 -- ditto
from raven.librarian import audio_input_panel as aip  # noqa: E402 -- ditto

SECOND = 10**9  # ns, as `_record_level` timestamps in

# The sliders, by tag. A callback is handed the widget that fired it, and the panel writes the
# quantized value back through it — so a test passing `None` as the sender would exercise a path
# no drag ever takes.
THRESHOLD_SLIDER = "audio_input_threshold_slider"  # tag
AUTOSTOP_SLIDER = "audio_input_autostop_slider"  # tag
PEAK_HOLD_SLIDER = "audio_input_peak_hold_slider"  # tag

DEVICES = ["Built-in Microphone", "Webcam Microphone", "Monitor of Built-in Audio"]

CONFIGURED = {"stt_silence_threshold": -40.0,
              "stt_autostop_timeout": 1.5,
              "stt_vu_peak_hold": 1.0,
              "stt_capture_audio_device": DEVICES[0]}


class StubRecorder:
    """Everything the panel asks of a recorder, and no audio device."""

    def __init__(self):
        self.device_name = DEVICES[0]
        self.silence_threshold = CONFIGURED["stt_silence_threshold"]
        self.autostop_timeout = CONFIGURED["stt_autostop_timeout"]
        self.vu_peak_hold = CONFIGURED["stt_vu_peak_hold"]
        self.recording = False
        self.monitoring = False
        self.listeners = []
        self.starts = []

    def is_capturing(self):
        return self.recording or self.monitoring

    def is_recording(self):
        return self.recording

    def is_monitoring(self):
        return self.monitoring

    def connect_vu_readout(self, cb):
        if cb not in self.listeners:
            self.listeners.append(cb)

    def disconnect_vu_readout(self, cb):
        if cb in self.listeners:
            self.listeners.remove(cb)

    def start(self, on_autostop=None, monitor=False):
        self.starts.append(monitor)
        self.monitoring = monitor
        self.recording = not monitor

    def stop(self, wait=False, timeout=1.0):
        self.monitoring = self.recording = False
        return True

    def set_device(self, device_name):
        # Mirrors the real one: refuses mid-recording, raises for a device that is not there, and
        # resumes monitoring on the new device.
        if self.recording:
            return False
        if device_name not in DEVICES:
            raise ValueError(f"No such audio capture device {device_name!r}")
        was_monitoring = self.monitoring
        self.stop()
        self.device_name = device_name
        if was_monitoring:
            self.start(monitor=True)
        return True


@pytest.fixture(scope="module")
def dpg_context():
    """A DPG context with an unmapped viewport, torn down after the module."""
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: these tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


@pytest.fixture(scope="module")
def themes_and_fonts(dpg_context):
    return guiutils.bootup(font_size=20)


@pytest.fixture
def panel(dpg_context, themes_and_fonts, monkeypatch):
    """A built panel, with a stub recorder in place of the singleton. Not open, so nothing captures."""
    stub = StubRecorder()
    monkeypatch.setattr(audio_recorder, "instance", stub)
    monkeypatch.setattr(audio_recorder, "get_available_devices", lambda refresh=False: list(DEVICES))
    thepanel = aip.DPGAudioInputPanel(app_state=dict(CONFIGURED),
                                      configured_defaults=dict(CONFIGURED),
                                      themes_and_fonts=themes_and_fonts)
    thepanel._build_window()
    thepanel.recorder = stub  # for the tests to reach; the panel itself goes through `require()`
    yield thepanel
    with guiutils.nonexistent_ok():
        dpg.delete_item(thepanel.window_id)


class TestTheStubResemblesTheRealRecorder:
    """Everything else here is worth only as much as the stub's likeness to `Recorder`.

    A stub is free to drift, and a suite testing against a drifted one goes on passing while it stops
    saying anything about the code that ships. These assert the likeness at the points the panel
    actually leans on — checked against the real class, which imports without an audio device since
    `pvrecorder` is loaded only where one is opened.
    """

    @pytest.mark.parametrize("name", sorted(name for name in vars(StubRecorder) if not name.startswith("_")))
    def test_the_real_recorder_has_it_too(self, name):
        assert hasattr(audio_recorder.Recorder, name) or name in inspect.signature(audio_recorder.Recorder.__init__).parameters, \
            f"the stub offers {name!r}, which `Recorder` does not — the stub has drifted"

    @pytest.mark.parametrize("name,expected", [("start", {"on_autostop", "monitor"}),
                                               ("stop", {"wait", "timeout"})])
    def test_the_real_recorder_takes_the_same_arguments(self, name, expected):
        parameters = set(inspect.signature(getattr(audio_recorder.Recorder, name)).parameters)
        assert expected <= parameters, f"`Recorder.{name}` no longer takes {sorted(expected - parameters)}"

    @pytest.mark.parametrize("name", ["device_name", "silence_threshold", "autostop_timeout", "vu_peak_hold"])
    def test_the_settings_are_constructor_parameters_of_the_real_recorder(self, name):
        assert name in inspect.signature(audio_recorder.Recorder.__init__).parameters


class TestASliderOnlyEditsASettingThatIsInEffect:
    """A slider whose switch is off must not switch it back on.

    The panel refuses this by asking the recorder, not by trusting that DPG withholds input from a
    disabled widget — so these hold whatever the toolkit does with a disabled slider.
    """

    def test_the_autostop_slider_does_nothing_while_autostop_is_off(self, panel):
        panel._on_autostop_checkbox(None, False)
        assert panel.recorder.autostop_timeout is None
        panel._on_autostop_slider(AUTOSTOP_SLIDER,4.0)
        assert panel.recorder.autostop_timeout is None, "moving the slider switched autostop back on"

    def test_the_autostop_slider_works_while_autostop_is_on(self, panel):
        # The control for the test above: with the switch on, this very call does change the setting,
        # so that test is about the switch rather than about the slider being inert in general.
        panel._on_autostop_checkbox(None, True)
        panel._on_autostop_slider(AUTOSTOP_SLIDER,4.0)
        assert panel.recorder.autostop_timeout == 4.0

    def test_the_threshold_slider_does_nothing_while_autodetect_is_on(self, panel):
        panel._on_autodetect_checkbox(None, True)
        assert panel.recorder.silence_threshold is None
        panel._on_threshold_slider(THRESHOLD_SLIDER,-55.0)
        assert panel.recorder.silence_threshold is None, "moving the slider cancelled the autodetection"

    def test_the_threshold_slider_works_while_autodetect_is_off(self, panel):
        panel._on_autodetect_checkbox(None, False)
        panel._on_threshold_slider(THRESHOLD_SLIDER,-55.0)
        assert panel.recorder.silence_threshold == -55.0


class TestTheControlsWriteEverywhereTheyHaveTo:
    def test_the_threshold_reaches_the_recorder_the_state_and_the_meter(self, panel):
        panel._on_threshold_slider(THRESHOLD_SLIDER,-52.0)
        assert panel.recorder.silence_threshold == -52.0
        assert panel.app_state["stt_silence_threshold"] == -52.0
        assert panel.meter.threshold == -52.0

    def test_autodetect_clears_the_meters_line(self, panel):
        panel._on_autodetect_checkbox(None, True)
        assert panel.app_state["stt_silence_threshold"] is None
        assert panel.meter.threshold is None

    def test_the_autostop_timeout_reaches_the_recorder_and_the_state(self, panel):
        panel._on_autostop_slider(AUTOSTOP_SLIDER,2.5)
        assert panel.recorder.autostop_timeout == 2.5
        assert panel.app_state["stt_autostop_timeout"] == 2.5

    def test_peak_hold_reaches_the_recorder_and_the_state(self, panel):
        panel._on_peak_hold_slider(PEAK_HOLD_SLIDER,3.0)
        assert panel.recorder.vu_peak_hold == 3.0
        assert panel.app_state["stt_vu_peak_hold"] == 3.0


class TestASliderStoresThePrecisionItShows:
    """ImGui's float slider has no step, so `format="%.1f"` changes what the number looks like and not
    what it is. Left alone, a drag stores seventeen digits under a control that offered one.
    """

    # Values with digits past the format's, as a real drag produces — each inside its own slider's range.
    DRAGGED = [(THRESHOLD_SLIDER, "_on_threshold_slider", "silence_threshold", -42.5327194213867188, -42.5),
               (AUTOSTOP_SLIDER, "_on_autostop_slider", "autostop_timeout", 2.5327194213867188, 2.5),
               (PEAK_HOLD_SLIDER, "_on_peak_hold_slider", "vu_peak_hold", 2.5327194213867188, 2.5)]

    @pytest.mark.parametrize("slider,callback_name,setting,dragged,expected", DRAGGED)
    def test_a_dragged_value_is_rounded_to_one_decimal(self, panel, slider, callback_name, setting, dragged, expected):
        assert dragged != expected, "this fixture cannot tell a rounded value from an unrounded one"
        getattr(panel, callback_name)(slider, dragged)
        assert getattr(panel.recorder, setting) == expected

    @pytest.mark.parametrize("slider,callback_name,setting,dragged,expected", DRAGGED)
    def test_the_handle_snaps_onto_the_stored_value(self, panel, slider, callback_name, setting, dragged, expected):
        # Otherwise the widget keeps the unrounded value and the next drag starts from a number the
        # panel never stored — a drift the display would hide, since it rounds either way.
        getattr(panel, callback_name)(slider, dragged)
        assert dpg.get_value(slider) == pytest.approx(expected)


class TestTheOtherMeterIsKeptInStep:
    """The app's toolbar carries a second VU meter drawing the same threshold line.

    It is not the panel's to know about, so the panel announces the change and the app wires it up —
    and every route to a new threshold has to go through that announcement, which is what these check.
    Two meters disagreeing about where the threshold is would be worse than one.
    """

    @pytest.fixture
    def announced(self, panel):
        seen = []
        panel.on_threshold_changed = seen.append
        return seen

    def test_dragging_the_slider_announces(self, panel, announced):
        panel._on_threshold_slider(THRESHOLD_SLIDER, -55.0)
        assert announced == [-55.0]

    def test_measuring_the_room_announces(self, panel, announced):
        panel._record_level(-62.0)
        age_history(panel)
        panel._measure_the_room()
        assert announced == [-62.0 + silencegate.DEFAULT_SILENCE_MARGIN]

    def test_switching_to_autodetect_announces_that_there_is_no_line(self, panel, announced):
        panel._on_autodetect_checkbox(None, True)
        assert announced == [None]

    def test_resetting_announces(self, panel, announced):
        panel._on_threshold_slider(THRESHOLD_SLIDER, -55.0)
        announced.clear()
        panel._reset_to_configured_defaults()
        assert announced[-1] == CONFIGURED["stt_silence_threshold"]

    def test_opening_announces_what_the_recorder_holds(self, panel, announced):
        # Not because the toolbar's meter would otherwise start out wrong — the app applies the stored
        # settings before it builds that meter, so a session that never opens the panel is already
        # right. This is so that the panel never *introduces* a disagreement: it announces what it
        # finds, so the two are in step from the first open onward, whatever set the threshold before.
        panel.recorder.silence_threshold = -33.0
        panel.open()
        assert announced[-1] == -33.0


DEVICE_COMBO = "audio_input_device_combo"  # tag


class TestChoosingAMicrophone:
    def test_choosing_one_switches_the_recorder_and_is_remembered(self, panel):
        panel._on_device_combo(DEVICE_COMBO, DEVICES[1])
        assert panel.recorder.device_name == DEVICES[1]
        assert panel.app_state["stt_capture_audio_device"] == DEVICES[1]

    def test_the_reading_does_not_carry_over_from_the_previous_microphone(self, panel):
        panel._record_level(-20.0)
        age_history(panel)
        assert panel.floor == -20.0
        panel._on_device_combo(DEVICE_COMBO, DEVICES[1])
        assert panel.floor is None, "the loudest-recently reading still describes the old microphone"

    def test_switching_is_refused_while_a_message_is_being_recorded(self, panel):
        panel.recorder.recording = True
        panel._on_device_combo(DEVICE_COMBO, DEVICES[1])
        assert panel.recorder.device_name == DEVICES[0], "a recording was split across two microphones"
        assert dpg.get_value(DEVICE_COMBO) == DEVICES[0], "the combo is offering a device that is not in use"
        assert panel.app_state["stt_capture_audio_device"] != DEVICES[1]

    def test_a_device_that_vanished_between_offer_and_click_is_survived(self, panel):
        panel._on_device_combo(DEVICE_COMBO, "Microphone That Went Away")
        assert panel.recorder.device_name == DEVICES[0]
        assert dpg.get_value(DEVICE_COMBO) == DEVICES[0]

    def test_monitoring_follows_to_the_new_microphone(self, panel):
        panel.open()
        assert panel.recorder.is_monitoring()
        panel._on_device_combo(DEVICE_COMBO, DEVICES[1])
        assert panel.recorder.is_monitoring(), "the meter went dead after switching microphones"
        assert panel.recorder.device_name == DEVICES[1]

    def test_opening_re_asks_which_microphones_exist(self, panel, monkeypatch):
        # A microphone plugged in while the app was running is the whole reason to re-ask; the answer
        # from startup would never mention it.
        monkeypatch.setattr(audio_recorder, "get_available_devices",
                            lambda refresh=False: DEVICES + ["Microphone Plugged In Just Now"])
        panel.open()
        assert "Microphone Plugged In Just Now" in dpg.get_item_configuration(DEVICE_COMBO)["items"]

    def test_monitoring_devices_are_kept_out(self, panel):
        # DEVICES[2] is "Monitor of Built-in Audio". Pointing speech recognition at one would
        # transcribe whatever is being played, the AI's own voice included.
        panel.open()
        items = dpg.get_item_configuration(DEVICE_COMBO)["items"]
        assert DEVICES[2] not in items
        assert DEVICES[0] in items and DEVICES[1] in items, "the filter took the real microphones too"

    def test_a_configured_monitoring_device_stays_offerable_after_switching_away(self, panel, monkeypatch):
        # Filtering it away unconditionally is a one-way door: switch off it once and going back means
        # editing the configuration file.
        monkeypatch.setitem(panel.configured_defaults, "stt_capture_audio_device", DEVICES[2])
        panel.recorder.device_name = DEVICES[2]
        panel.open()
        assert DEVICES[2] in dpg.get_item_configuration(DEVICE_COMBO)["items"]

        panel.close()
        panel.recorder.device_name = DEVICES[0]  # the user picks a real microphone
        panel.open()
        assert DEVICES[2] in dpg.get_item_configuration(DEVICE_COMBO)["items"], \
            "the configured microphone is no longer reachable from the panel"

    def test_a_microphone_in_use_but_unplugged_is_listed_and_tagged(self, panel, monkeypatch):
        # Dropping it would leave the combo showing some other device as selected; listing it silently
        # would leave the reader wondering why it does not work.
        monkeypatch.setattr(audio_recorder, "get_available_devices", lambda refresh=False: DEVICES[1:])
        panel.open()
        items = dpg.get_item_configuration(DEVICE_COMBO)["items"]
        tagged = aip.device_label(DEVICES[0], DEVICES[1:])
        assert tagged in items
        assert tagged.startswith(DEVICES[0]) and tagged != DEVICES[0], "the entry is not marked as unavailable"
        assert dpg.get_value(DEVICE_COMBO) == tagged

    def test_a_present_microphone_is_not_tagged(self, panel):
        # The control for the test above: the tag has to mean something, so it must not be on everything.
        panel.open()
        assert dpg.get_value(DEVICE_COMBO) == DEVICES[0]
        assert aip.UNAVAILABLE_SUFFIX not in "".join(dpg.get_item_configuration(DEVICE_COMBO)["items"])

    def test_choosing_a_tagged_entry_asks_for_the_real_name(self, panel, monkeypatch):
        # The combo hands its callback the label it displays, tag and all; what reaches the recorder
        # has to be the microphone's actual name.
        monkeypatch.setattr(audio_recorder, "get_available_devices", lambda refresh=False: DEVICES[1:])
        panel.open()
        panel.recorder.device_name = DEVICES[1]  # move off it, so switching back is a real change
        panel._on_device_combo(DEVICE_COMBO, aip.device_label(DEVICES[0], DEVICES[1:]))
        assert panel.recorder.device_name == DEVICES[0]


def age_history(panel, seconds=None):
    """Pretend everything currently in the panel's level history happened `seconds` ago.

    The floor's window ends `FLOOR_LAG` before now, so a level just recorded is deliberately not in it
    yet — see `test_the_sound_of_pressing_the_button_is_not_measured`. Tests that want a level counted
    have to let it age in, and sleeping through half a second each time is not the way.
    """
    if seconds is None:
        seconds = aip.FLOOR_LAG + 0.1
    shift = int(seconds * SECOND)
    panel._levels = collections.deque((timestamp - shift, level) for timestamp, level in panel._levels)


def meter_line_ys(panel):
    """The y coordinates of the lines the panel's VU meter has drawn — its peak and threshold."""
    return [dpg.get_item_configuration(item)["p1"][1]
            for item in dpg.get_item_children(panel.meter.drawlist, slot=2)
            if dpg.get_item_type(item) == "mvAppItemType::mvDrawLine"]


class TestThePanelsOwnMeterIsFed:
    """The toolbar's meter is connected to the recorder by the app; this one is the panel's to drive,
    and nothing else does it. Asserted through what the meter draws rather than through its state.
    """

    def test_a_level_reaches_the_meter(self, panel):
        before = meter_line_ys(panel)
        panel._on_vu_update(-30.0, -20.0)
        after = meter_line_ys(panel)
        assert after != before, "the meter drew the same thing after a new level arrived"

    def test_the_peak_line_follows_the_peak(self, panel):
        panel._on_vu_update(-90.0, -80.0)
        low = meter_line_ys(panel)
        panel._on_vu_update(-90.0, -10.0)
        high = meter_line_ys(panel)
        # y counts down from the top, so a louder peak sits at a smaller y.
        assert min(high) < min(low), f"a louder peak did not move the line up: {low} then {high}"


class TestTheLoudestRecentlyReading:
    def test_nothing_heard_yet_reads_as_nothing(self, panel):
        assert panel.floor is None
        assert aip.format_dBFS(panel.floor) == "—"

    def test_it_is_the_maximum_rather_than_the_last_or_the_mean(self, panel):
        for level in (-70.0, -62.0, -68.0):
            panel._record_level(level)
        age_history(panel)
        assert panel.floor == -62.0

    def test_the_sound_of_pressing_the_button_is_not_measured(self, panel):
        """Pressing *Measure the room* makes a noise the microphone hears, so the window ends before now.

        Live case 2026-08-28: a room floor near -40 dBFS, and the mouse click alone read -27 — the
        operator's own hand setting the threshold 13 dB high, every time, since clicking that button is
        the whole gesture.
        """
        panel._record_level(-40.0)  # the room
        age_history(panel)  # ...a moment ago
        panel._record_level(-27.0)  # the click, just now
        assert panel.floor == -40.0, "the click set the floor"

    def test_a_level_inside_the_window_is_still_measured(self, panel):
        # The control: the same sample, old enough to have aged in, does count. Without this the test
        # above would pass against a floor that ignored everything.
        panel._record_level(-27.0)
        age_history(panel)
        assert panel.floor == -27.0

    def test_nothing_has_aged_in_yet_reads_as_nothing(self, panel):
        # Monitoring started less than the lag ago: there is a reading, but none of it is old enough to
        # be trusted, and a dash is the honest answer.
        panel._record_level(-40.0)
        assert panel.floor is None

    def test_a_level_older_than_the_window_is_forgotten(self, panel):
        # Timestamps come from `time.monotonic_ns` inside `_record_level`, so age the samples by
        # reaching into the history rather than by sleeping through the window.
        panel._record_level(-20.0)
        panel._record_level(-70.0)
        age_history(panel)
        assert panel.floor == -20.0, "this fixture cannot tell a forgotten level from one never counted"

        # Push the loud one out of the far end of the window, leaving the quiet one inside it.
        stale_ns, level = panel._levels[0]
        panel._levels[0] = (stale_ns - int((aip.FLOOR_WINDOW + 1.0) * SECOND), level)
        panel._record_level(-70.0)  # any append trims the window
        age_history(panel)
        assert panel.floor == -70.0, "a level older than the window is still being counted"

    def test_digital_silence_reads_as_a_dash(self, panel):
        panel._record_level(-math.inf)
        age_history(panel)
        assert panel.floor == -math.inf
        assert aip.format_dBFS(panel.floor) == "—"


class TestMeasureTheRoom:
    def test_it_sets_the_threshold_a_margin_above_the_loudest_recently(self, panel):
        for level in (-70.0, -62.0, -68.0):
            panel._record_level(level)
        age_history(panel)
        panel._measure_the_room()
        assert panel.recorder.silence_threshold == -62.0 + silencegate.DEFAULT_SILENCE_MARGIN

    def test_it_uses_the_maximum_and_not_the_latest(self, panel):
        # The control for the test above, and the property the design rests on: one loud moment has to
        # move the threshold, since one frame above it is enough to hold a recording open.
        for level in (-62.0, -70.0, -70.0):  # loudest first, so "latest" and "maximum" disagree
            panel._record_level(level)
        age_history(panel)
        panel._measure_the_room()
        assert panel.recorder.silence_threshold == -62.0 + silencegate.DEFAULT_SILENCE_MARGIN

    def test_it_cancels_autodetect(self, panel):
        panel._on_autodetect_checkbox(None, True)
        panel._record_level(-62.0)
        age_history(panel)
        panel._measure_the_room()
        assert panel.recorder.silence_threshold is not None
        assert dpg.get_value("audio_input_autodetect_checkbox") is False  # tag

    def test_it_stays_inside_the_meter(self, panel):
        panel._record_level(-1.0)  # + the margin would be above full scale
        age_history(panel)
        panel._measure_the_room()
        assert panel.recorder.silence_threshold == aip.METER_MAX

    def test_nothing_heard_yet_leaves_the_threshold_alone(self, panel):
        panel._on_threshold_slider(THRESHOLD_SLIDER,-33.0)
        panel._measure_the_room()
        assert panel.recorder.silence_threshold == -33.0

    def test_digital_silence_leaves_the_threshold_alone(self, panel):
        # An input sending nothing at all is a broken microphone, not a very quiet room, and -inf plus
        # a margin is still -inf. Better to leave the last usable value than to write that in.
        panel._on_threshold_slider(THRESHOLD_SLIDER,-33.0)
        panel._record_level(-math.inf)
        age_history(panel)
        panel._measure_the_room()
        assert panel.recorder.silence_threshold == -33.0


class TestResetToConfiguredDefaults:
    def test_it_restores_every_setting(self, panel):
        panel._on_threshold_slider(THRESHOLD_SLIDER,-55.0)
        panel._on_autostop_slider(AUTOSTOP_SLIDER,4.0)
        panel._on_peak_hold_slider(PEAK_HOLD_SLIDER,3.0)
        panel._reset_to_configured_defaults()
        assert panel.recorder.silence_threshold == CONFIGURED["stt_silence_threshold"]
        assert panel.recorder.autostop_timeout == CONFIGURED["stt_autostop_timeout"]
        assert panel.recorder.vu_peak_hold == CONFIGURED["stt_vu_peak_hold"]
        assert panel.app_state == CONFIGURED

    def test_it_restores_a_switched_off_autostop(self, panel):
        panel._on_autostop_checkbox(None, False)
        assert panel.recorder.autostop_timeout is None
        panel._reset_to_configured_defaults()
        assert panel.recorder.autostop_timeout == CONFIGURED["stt_autostop_timeout"]
        assert dpg.get_value("audio_input_autostop_checkbox") is True  # tag


class TestMonitoring:
    def test_opening_starts_monitoring_and_closing_stops_it(self, panel):
        panel.open()
        assert panel.is_open
        assert panel.recorder.starts == [True], "the panel captured in recording mode"
        assert panel.recorder.is_monitoring()
        assert panel._on_vu_update in panel.recorder.listeners

        panel.close()
        assert not panel.is_open
        assert not panel.recorder.is_capturing()
        assert panel._on_vu_update not in panel.recorder.listeners

    def test_it_refuses_to_open_over_a_recording(self, panel):
        panel.recorder.recording = True
        panel.open()
        assert not panel.is_open, "opening the panel took the microphone off a recording"
        assert panel.recorder.starts == []

    def test_monitoring_does_not_resume_while_the_panel_is_closed(self, panel):
        # What the app calls after a recording ends: the panel takes the device back only if the user
        # still has it open.
        panel.start_monitoring()
        assert panel.recorder.starts == []

    def test_closing_saves_the_app_state(self, panel):
        saved = []
        panel.save_app_state = lambda: saved.append(dict(panel.app_state))
        panel.open()
        panel._on_threshold_slider(THRESHOLD_SLIDER,-47.0)
        panel.close()
        assert saved and saved[-1]["stt_silence_threshold"] == -47.0
