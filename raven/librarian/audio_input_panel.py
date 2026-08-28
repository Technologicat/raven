"""The Librarian's "Audio input" panel: set the microphone's silence threshold while watching the room.

Speech input has to work in whatever room it is set up in, and the number that decides whether a
recording stops by itself — the silence threshold — depends on that room's noise floor rather than
on anything knowable in advance. So the panel exists to be used *in* the room: it meters the input
whenever it is open, without recording anything, and offers the threshold as a control rather than
as a configuration value.

Non-modal on purpose. The calibration that matters is watching the meter while somebody actually
speaks, and a modal would hide the conversation that is happening.

The decision the threshold feeds is `raven.common.audio.silencegate`, which is separate from the
recorder and unit-tested on its own; this module is the part that needs a render loop and an audio
device, and is verified by hand.
"""

__all__ = ["device_label", "device_name_from_label",
           "format_dBFS",
           "DPGAudioInputPanel"]

import logging
logger = logging.getLogger(__name__)

import collections
import math
import time
from collections.abc import Sequence
from typing import Callable, Optional, Union

import dearpygui.dearpygui as dpg

from ..common import numutils
from ..common.audio import recorder as audio_recorder
from ..common.audio import silencegate
from ..common.gui import utils as guiutils
from ..common.gui.vumeter import DPGVUMeter

from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa

from . import config as librarian_config

gui_config = librarian_config.gui_config  # shorthand, as in `chat_controller`

# The meter's range. 0 dBFS is full scale, and -90 is about as quiet as 16-bit audio gets.
METER_MIN = -90.0
METER_MAX = 0.0
METER_YELLOW_START = -24.0
METER_RED_START = -6.0

# How far back the "loudest recently" readout looks, and — the same number, deliberately — what
# "Measure the room" measures over. That way the readout is a preview of what the button will do.
FLOOR_WINDOW = 3.0  # seconds

# What the sliders will let the user ask for. The autostop range runs high because a speaker pausing
# to think is a real case, and cutting them off mid-question is the failure that reads as breakage.
AUTOSTOP_MIN = 0.2  # seconds
AUTOSTOP_MAX = 10.0  # seconds
PEAK_HOLD_MIN = 0.1  # seconds
PEAK_HOLD_MAX = 5.0  # seconds

# Decimals every slider here displays, and — via `_quantize` — the precision it actually stores.
# Keep in step with the `format=` strings below, which are all `%.1f`.
_SLIDER_DECIMALS = 1

DIM_TEXT = (180, 180, 180)

# Appended to a microphone the panel lists although the OS no longer reports it — one that was
# unplugged, or one named in the configuration that is not here today. Listing it silently would
# leave the reader wondering why it does not work; leaving it out is a one-way door.
UNAVAILABLE_SUFFIX = "  [unavailable]"


def device_label(device_name: str, available: Sequence[str]) -> str:
    """The combo entry for `device_name`, tagged if it is not among `available`."""
    return device_name if device_name in available else f"{device_name}{UNAVAILABLE_SUFFIX}"


def device_name_from_label(label: str) -> str:
    """Recover a microphone's real name from a combo entry `device_label` produced."""
    if label.endswith(UNAVAILABLE_SUFFIX):
        return label[:-len(UNAVAILABLE_SUFFIX)]
    return label


def format_dBFS(value: Optional[float]) -> str:
    """Format a signal level for display, rendering silence and "nothing heard yet" as a dash."""
    if value is None or value == -math.inf:
        return "—"
    return f"{value:0.1f} dBFS"


class DPGAudioInputPanel:
    """The "Audio input" panel: meters the microphone and tunes the silence detection.

    While open, the recorder captures in monitoring mode — the level is live and nothing is kept — so
    the room's noise floor can be read off without sending anything to the AI. Opening it while a
    message is being recorded is refused rather than interrupting that.

    `app_state`: the Librarian app state dict, which carries the tuned values between runs. The panel
                 writes into it as the controls move, and asks `save_app_state` to persist on close.
    `configured_defaults`: what the *reset* button puts back, keyed as in `app_state` — normally
                           `appstate._DEFAULT_SETTINGS`, i.e. what `client/config.py` says.
    `themes_and_fonts`: the app's `guiutils.bootup` result, for the icon font.
    `save_app_state`: optional zero-argument callable, run when the panel closes.
    `on_threshold_changed`: optional 1-argument callable, receiving the new silence threshold in dBFS,
                            or `None` when it is being measured per recording. For any *other* meter
                            drawing the same line — the app's toolbar carries one, and two meters
                            disagreeing about where the threshold is would be worse than one.
    `centering_reference_window`: DPG tag or ID to center on the first time the panel opens; the main
                                  window, normally. Later opens leave the panel where the user put it.
    """

    def __init__(self,
                 app_state: dict,
                 configured_defaults: dict,
                 themes_and_fonts,
                 save_app_state: Optional[Callable] = None,
                 on_threshold_changed: Optional[Callable] = None,
                 centering_reference_window: Optional[Union[int, str]] = None):
        self.app_state = app_state
        self.configured_defaults = dict(configured_defaults)
        self.themes_and_fonts = themes_and_fonts
        self.save_app_state = save_app_state
        self.on_threshold_changed = on_threshold_changed
        self.centering_reference_window = centering_reference_window

        self.is_open = False
        self.window_id = None
        self.meter = None
        self._has_been_positioned = False

        # Recent signal levels, as `(timestamp_ns, dBFS)`, trimmed to `FLOOR_WINDOW`. Appended by the
        # capture thread, read by whichever thread the user clicks on.
        self._levels = collections.deque()

    # ------------------------------------------------------------------------------
    # Opening and closing

    def open(self) -> None:
        """Show the panel and start metering the input. Safe to call when already open (no-op)."""
        if self.is_open:
            return
        if audio_recorder.require().is_recording():
            logger.info("DPGAudioInputPanel.open: a message is being recorded; not opening.")
            return

        if self.window_id is None:
            self._build_window()
        self._refresh_device_list()  # a microphone may have been plugged in since the last look
        self._sync_widgets_from_recorder()
        self.is_open = True

        if self.centering_reference_window is not None and not self._has_been_positioned:
            dpg.split_frame()  # let anything that is closing finish first, or ours may not appear
            guiutils.recenter_window(self.window_id, reference_window=self.centering_reference_window)  # this shows it
            self._has_been_positioned = True
        else:
            dpg.show_item(self.window_id)  # tag

        self.start_monitoring()

    def close(self) -> None:
        """Hide the panel, stop metering, and persist what was tuned."""
        if not self.is_open:
            return
        self.is_open = False
        self.stop_monitoring()
        with guiutils.nonexistent_ok():
            dpg.hide_item(self.window_id)  # tag
        if self.save_app_state is not None:
            self.save_app_state()

    def toggle(self) -> None:
        """Open the panel if closed, close it if open. What the hotkey and the toolbar button call."""
        if self.is_open:
            self.close()
        else:
            self.open()

    # ------------------------------------------------------------------------------
    # Monitoring
    #
    # The microphone is one device handle, so monitoring and recording a message cannot overlap. The
    # app hands the device over around a recording by calling these two, which is why they are public.

    def start_monitoring(self) -> None:
        """Start metering the input, if the panel is open and the device is free."""
        if not self.is_open:
            return
        rec = audio_recorder.require()
        if rec.is_capturing():
            return
        self._levels.clear()
        rec.connect_vu_readout(self._on_vu_update)
        rec.start(monitor=True)
        self._set_status_text()

    def stop_monitoring(self) -> None:
        """Stop metering, and let go of the device so a recording can have it."""
        rec = audio_recorder.require()
        rec.disconnect_vu_readout(self._on_vu_update)
        if rec.is_monitoring():
            rec.stop()
        self._set_status_text()

    def _on_vu_update(self, instant: float, peak: float) -> None:
        """Take one frame's levels from the recorder. Runs on the capture thread."""
        self._record_level(instant)
        with guiutils.nonexistent_ok():
            # The panel's meter, unlike the toolbar's, has no other feed: the app connects that one to
            # the readout directly, and this one is ours to drive.
            if self.meter is not None:
                self.meter.update(instant, peak)
            dpg.set_value("audio_input_now_text", format_dBFS(instant))  # tag
            dpg.set_value("audio_input_peak_text", format_dBFS(peak))  # tag
            dpg.set_value("audio_input_floor_text", format_dBFS(self.floor))  # tag

    def _record_level(self, level: float) -> None:
        """Add one level to the recent history, and drop what has aged out of the window."""
        now_ns = time.monotonic_ns()
        self._levels.append((now_ns, level))
        cutoff = now_ns - int(FLOOR_WINDOW * 10**9)
        while self._levels and self._levels[0][0] < cutoff:
            self._levels.popleft()

    def _get_floor(self) -> Optional[float]:
        # Snapshot rather than lock: the capture thread appends here every audio frame, and blocking it
        # to read a cosmetic number would be the wrong trade. One C-level pass cannot see a half-mutated
        # deque; what it can be is an instant out of date, which for a level readout is free.
        levels = tuple(self._levels)
        if not levels:
            return None
        return max(level for _, level in levels)
    floor = property(fget=_get_floor,
                     doc="The loudest the input has been within the recent-history window, in dBFS, or `None` with nothing heard yet. Read-only.")

    # ------------------------------------------------------------------------------
    # The controls
    #
    # Each writes three places: the recorder, which reads its settings live; the app state, so the
    # value survives a restart; and — for the threshold — the meter, where it is a line to aim.

    def _show_threshold(self, value: Optional[float]) -> None:
        """Point every threshold line at `value` — ours, and any the app keeps elsewhere."""
        if self.meter is not None:
            self.meter.threshold = value
        if self.on_threshold_changed is not None:
            self.on_threshold_changed(value)

    def _apply_threshold(self, value: Optional[float]) -> None:
        audio_recorder.require().silence_threshold = value
        self.app_state["stt_silence_threshold"] = value
        self._show_threshold(value)

    def _apply_autostop_timeout(self, value: Optional[float]) -> None:
        audio_recorder.require().autostop_timeout = value
        self.app_state["stt_autostop_timeout"] = value

    def _apply_peak_hold(self, value: float) -> None:
        audio_recorder.require().vu_peak_hold = value
        self.app_state["stt_vu_peak_hold"] = value

    # A slider only edits a value that is currently in effect, and it asks the recorder whether that is
    # so rather than asking its own checkbox. Two assumptions drop out that way: that DPG withholds
    # input from a disabled widget, and that a checkbox's widget value is in step with the `app_data`
    # its callback was handed. Neither has been measured here, and what a wrong guess switches back on
    # is precisely the setting the user turned off.
    def _quantize(self, sender, app_data) -> float:
        """Round a slider's value to the precision its `format` displays, and snap the handle onto it."""
        return guiutils.snap_slider(sender, app_data, decimals=_SLIDER_DECIMALS)

    def _on_threshold_slider(self, sender, app_data) -> None:
        if audio_recorder.require().silence_threshold is None:
            return  # measuring per recording; the slider is only saying what a fixed threshold would be
        self._apply_threshold(self._quantize(sender, app_data))

    def _on_autodetect_checkbox(self, sender, app_data) -> None:
        autodetect = bool(app_data)
        dpg.configure_item("audio_input_threshold_slider", enabled=not autodetect)  # tag
        self._apply_threshold(None if autodetect else float(dpg.get_value("audio_input_threshold_slider")))  # tag

    def _on_autostop_checkbox(self, sender, app_data) -> None:
        enabled = bool(app_data)
        dpg.configure_item("audio_input_autostop_slider", enabled=enabled)  # tag
        self._apply_autostop_timeout(float(dpg.get_value("audio_input_autostop_slider")) if enabled else None)  # tag

    def _on_autostop_slider(self, sender, app_data) -> None:
        if audio_recorder.require().autostop_timeout is None:
            return  # autostop is off; the slider is only saying how long it would wait
        self._apply_autostop_timeout(self._quantize(sender, app_data))

    def _on_peak_hold_slider(self, sender, app_data) -> None:
        self._apply_peak_hold(self._quantize(sender, app_data))

    def _on_device_combo(self, sender, app_data) -> None:
        """Switch microphones, and start metering the new one straight away."""
        rec = audio_recorder.require()
        chosen = device_name_from_label(str(app_data))
        try:
            switched = rec.set_device(chosen)
        except ValueError:  # unplugged between the list being offered and the choice being made
            logger.warning(f"DPGAudioInputPanel._on_device_combo: '{chosen}' is no longer there; staying on '{rec.device_name}'.")
            switched = False
        if not switched:
            # Rebuild rather than just putting the value back: the OS is worth re-asking after a switch
            # that did not take, and it is what decides which entries carry the unavailable tag.
            self._refresh_device_list()
            return
        self.app_state["stt_capture_audio_device"] = rec.device_name
        self._levels.clear()  # the reading belongs to the microphone that produced it
        self._set_status_text()

    def _refresh_device_list(self) -> None:
        """Re-ask the OS which microphones exist, and offer those. Called each time the panel opens."""
        # Monitoring devices record what is being played, so offering one by default would invite
        # transcribing the AI's own voice back into the chat. They are kept out of the list — except
        # for two that a user has already named, and would otherwise be unable to get back to:
        #
        #   - the one the recorder holds, so the combo shows the recorder's own answer rather than
        #     appearing to be set to some other device;
        #   - the configured one, because filtering that away is a one-way door. Switch off it once
        #     and it is gone from the combo, and going back means editing `client/config.py`.
        #
        # The first also covers a device the OS has stopped listing. What happens to a capture whose
        # device is unplugged mid-recording is not known here — the sound server may move the stream,
        # or reads may start failing — so the panel reports what it was asked for and does not guess.
        available = audio_recorder.get_available_devices(refresh=True)
        devices = [name for name in available if not audio_recorder.is_monitoring_device(name)]
        current = audio_recorder.require().device_name
        for name in (current, self.configured_defaults.get("stt_capture_audio_device")):
            if name is not None and name not in devices:
                devices.append(name)
        dpg.configure_item("audio_input_device_combo", items=[device_label(name, available) for name in devices])  # tag
        dpg.set_value("audio_input_device_combo", device_label(current, available))  # tag

    def _measure_the_room(self) -> None:
        """Set the threshold from what has been heard recently, plus a margin."""
        floor = self.floor
        if floor is None or floor == -math.inf:
            logger.info("DPGAudioInputPanel._measure_the_room: nothing heard yet; leaving the threshold alone.")
            return
        threshold = numutils.clamp(floor + silencegate.DEFAULT_SILENCE_MARGIN, METER_MIN, METER_MAX)
        logger.info(f"DPGAudioInputPanel._measure_the_room: loudest of the last {FLOOR_WINDOW:0.6g}s was {floor:0.2f}dBFS, setting the threshold to {threshold:0.2f}dBFS.")
        dpg.set_value("audio_input_autodetect_checkbox", False)  # tag
        dpg.configure_item("audio_input_threshold_slider", enabled=True)  # tag
        dpg.set_value("audio_input_threshold_slider", threshold)  # tag  # a programmatic set fires no callback
        self._apply_threshold(threshold)

    def _reset_to_configured_defaults(self) -> None:
        """Put back what `client/config.py` says, which is the way out of a tuning that went wrong."""
        logger.info("DPGAudioInputPanel._reset_to_configured_defaults: restoring the configured values.")
        rec = audio_recorder.require()
        self.app_state.update(self.configured_defaults)
        rec.silence_threshold = self.app_state["stt_silence_threshold"]
        rec.autostop_timeout = self.app_state["stt_autostop_timeout"]
        rec.vu_peak_hold = self.app_state["stt_vu_peak_hold"]
        self._sync_widgets_from_recorder()

    def _sync_widgets_from_recorder(self) -> None:
        """Point every control at what the recorder is actually set to."""
        rec = audio_recorder.require()

        threshold = rec.silence_threshold
        autodetect = threshold is None
        dpg.set_value("audio_input_autodetect_checkbox", autodetect)  # tag
        dpg.configure_item("audio_input_threshold_slider", enabled=not autodetect)  # tag
        if not autodetect:
            dpg.set_value("audio_input_threshold_slider", threshold)  # tag

        autostop = rec.autostop_timeout
        dpg.set_value("audio_input_autostop_checkbox", autostop is not None)  # tag
        dpg.configure_item("audio_input_autostop_slider", enabled=autostop is not None)  # tag
        if autostop is not None:
            dpg.set_value("audio_input_autostop_slider", autostop)  # tag

        dpg.set_value("audio_input_peak_hold_slider", rec.vu_peak_hold)  # tag

        self._show_threshold(threshold)

    def _set_status_text(self) -> None:
        rec = audio_recorder.require()
        if rec.is_monitoring():
            status = "Listening. Nothing is recorded or sent."
        elif rec.is_recording():
            status = "Recording a message."
        else:
            status = "Not listening."
        with guiutils.nonexistent_ok():
            dpg.set_value("audio_input_status_text", status)  # tag

    # ------------------------------------------------------------------------------
    # Layout

    def _build_window(self) -> None:
        """Build the panel, hidden. Built once and reused, so fixed tags are safe here.

        Every item names its parent. The container stack is one process-wide global and DPG lets any
        thread create widgets, so a `with` is only safe while building an app's main GUI, before the
        render loop starts and before there is a background task to race. This runs on first open,
        which is neither.
        """
        rec = audio_recorder.require()
        slider_w = 200

        window_id = dpg.add_window(label="Audio input",
                                   modal=False,
                                   show=False,
                                   no_collapse=True,
                                   autosize=True,
                                   tag="audio_input_panel_window",  # tag
                                   on_close=lambda: self.close())
        self.window_id = window_id

        device_row = dpg.add_group(horizontal=True, parent=window_id)
        dpg.add_text("Microphone", color=DIM_TEXT, parent=device_row)
        dpg.add_combo(items=audio_recorder.get_available_devices(),
                      default_value=rec.device_name,
                      width=-1,
                      callback=self._on_device_combo,
                      parent=device_row,
                      tag="audio_input_device_combo")  # tag
        dpg.add_text("", parent=window_id, tag="audio_input_status_text", color=DIM_TEXT)  # tag
        guiutils.add_section_separator(parent=window_id)

        meter_row = dpg.add_group(horizontal=True, parent=window_id)
        self.meter = DPGVUMeter(width=28,
                                height=150,
                                border=1,
                                min_value=METER_MIN,
                                max_value=METER_MAX,
                                yellow_start=METER_YELLOW_START,
                                red_start=METER_RED_START,
                                threshold_value=rec.silence_threshold,
                                line_thickness=2,  # one pixel is lost across a meter this wide
                                tooltip_text=("Microphone input level.\n"
                                              f"Yellow from {METER_YELLOW_START:0.6g}, red from {METER_RED_START:0.6g} dBFS.\n"
                                              "The gray line is the silence threshold."),
                                parent=meter_row)
        readouts = dpg.add_group(parent=meter_row)
        for label, tag in (("Now", "audio_input_now_text"),  # tag
                           ("Peak", "audio_input_peak_text"),  # tag
                           (f"Loudest in {FLOOR_WINDOW:0.6g} s", "audio_input_floor_text")):  # tag
            readout_row = dpg.add_group(horizontal=True, parent=readouts)
            dpg.add_text(f"{label}:", color=DIM_TEXT, parent=readout_row)
            dpg.add_text("—", parent=readout_row, tag=tag)
        dpg.add_spacer(height=8, parent=readouts)

        # Icon-only button with the words beside it, as elsewhere in the constellation: the icon font
        # carries no Latin glyphs, so a label mixing the two renders the text as boxes. The words stay
        # visible rather than living only in the tooltip — this is the control an operator who has never
        # seen the panel has to find, in a hurry.
        measure_row = dpg.add_group(horizontal=True, parent=readouts)
        dpg.add_button(label=fa.ICON_RULER,
                       callback=lambda: self._measure_the_room(),
                       width=gui_config.toolbutton_w,
                       parent=measure_row,
                       tag="audio_input_measure_button")  # tag
        dpg.bind_item_font("audio_input_measure_button", self.themes_and_fonts.icon_font_solid)  # tag
        dpg.add_text("Measure the room", parent=measure_row)
        # Plain DPG rather than `gui_tooltip.Tooltip`: this caption is written once and never changes,
        # so there is no resize for the class to protect against.
        measure_tooltip = dpg.add_tooltip("audio_input_measure_button")  # tag
        dpg.add_text(f"Set the silence threshold from the last {FLOOR_WINDOW:0.6g} seconds:\n"
                     f"the loudest moment in them, plus {silencegate.DEFAULT_SILENCE_MARGIN:0.6g} dB.\n\n"
                     "Ask the room to be quiet first. The figure it will use is\n"
                     f"the \"Loudest in {FLOOR_WINDOW:0.6g} s\" reading above.",
                     parent=measure_tooltip)

        # Peak hold sits with the meter rather than with the settings below: it governs how far back the
        # peak line lets you see, which is a property of the readout you are looking at.
        dpg.add_slider_float(label="Meter peak hold",
                             min_value=PEAK_HOLD_MIN,
                             max_value=PEAK_HOLD_MAX,
                             default_value=rec.vu_peak_hold,
                             format="%.1f s",
                             width=slider_w,
                             callback=self._on_peak_hold_slider,
                             parent=window_id,
                             tag="audio_input_peak_hold_slider")  # tag

        guiutils.add_section_separator(parent=window_id)

        dpg.add_slider_float(label="Silence level",
                             min_value=METER_MIN,
                             max_value=METER_MAX,
                             default_value=(rec.silence_threshold if rec.silence_threshold is not None
                                            else silencegate.DEFAULT_SILENCE_THRESHOLD),
                             format="%.1f dBFS",
                             width=slider_w,
                             callback=self._on_threshold_slider,
                             parent=window_id,
                             tag="audio_input_threshold_slider")  # tag
        dpg.add_checkbox(label="Measure at the start of each recording instead",
                         default_value=rec.silence_threshold is None,
                         callback=self._on_autodetect_checkbox,
                         parent=window_id,
                         tag="audio_input_autodetect_checkbox")  # tag

        dpg.add_spacer(height=6, parent=window_id)
        dpg.add_checkbox(label="Stop when the speaker falls silent",
                         default_value=rec.autostop_timeout is not None,
                         callback=self._on_autostop_checkbox,
                         parent=window_id,
                         tag="audio_input_autostop_checkbox")  # tag
        dpg.add_slider_float(label="...after",
                             min_value=AUTOSTOP_MIN,
                             max_value=AUTOSTOP_MAX,
                             default_value=(rec.autostop_timeout if rec.autostop_timeout is not None
                                            else silencegate.DEFAULT_AUTOSTOP_TIMEOUT),
                             format="%.1f s",
                             width=slider_w,
                             callback=self._on_autostop_slider,
                             parent=window_id,
                             tag="audio_input_autostop_slider")  # tag

        guiutils.add_section_separator(parent=window_id)
        # No Close button: the window's own title-bar X does it, and `on_close` routes that through the
        # same `close`.
        dpg.add_button(label="Reset to configured defaults",
                       callback=lambda: self._reset_to_configured_defaults(),
                       parent=window_id,
                       tag="audio_input_reset_button")  # tag
