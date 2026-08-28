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

__all__ = ["format_dBFS",
           "DPGAudioInputPanel"]

import logging
logger = logging.getLogger(__name__)

import collections
import math
import time
from typing import Callable, Optional, Union

import dearpygui.dearpygui as dpg

from ..common import numutils
from ..common.audio import recorder as audio_recorder
from ..common.audio import silencegate
from ..common.gui import utils as guiutils
from ..common.gui.vumeter import DPGVUMeter

from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa

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

DIM_TEXT = (180, 180, 180)


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
    `centering_reference_window`: DPG tag or ID to center on the first time the panel opens; the main
                                  window, normally. Later opens leave the panel where the user put it.
    """

    def __init__(self,
                 app_state: dict,
                 configured_defaults: dict,
                 themes_and_fonts,
                 save_app_state: Optional[Callable] = None,
                 centering_reference_window: Optional[Union[int, str]] = None):
        self.app_state = app_state
        self.configured_defaults = dict(configured_defaults)
        self.themes_and_fonts = themes_and_fonts
        self.save_app_state = save_app_state
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

    def _apply_threshold(self, value: Optional[float]) -> None:
        audio_recorder.require().silence_threshold = value
        self.app_state["stt_silence_threshold"] = value
        if self.meter is not None:
            self.meter.threshold = value

    def _apply_autostop_timeout(self, value: Optional[float]) -> None:
        audio_recorder.require().autostop_timeout = value
        self.app_state["stt_autostop_timeout"] = value

    def _apply_peak_hold(self, value: float) -> None:
        audio_recorder.require().vu_peak_hold = value
        self.app_state["stt_vu_peak_hold"] = value

    def _on_threshold_slider(self, sender, app_data) -> None:
        self._apply_threshold(float(app_data))

    def _on_autodetect_checkbox(self, sender, app_data) -> None:
        autodetect = bool(app_data)
        dpg.configure_item("audio_input_threshold_slider", enabled=not autodetect)  # tag
        self._apply_threshold(None if autodetect else float(dpg.get_value("audio_input_threshold_slider")))  # tag

    def _on_autostop_checkbox(self, sender, app_data) -> None:
        enabled = bool(app_data)
        dpg.configure_item("audio_input_autostop_slider", enabled=enabled)  # tag
        self._apply_autostop_timeout(float(dpg.get_value("audio_input_autostop_slider")) if enabled else None)  # tag

    def _on_autostop_slider(self, sender, app_data) -> None:
        self._apply_autostop_timeout(float(app_data))

    def _on_peak_hold_slider(self, sender, app_data) -> None:
        self._apply_peak_hold(float(app_data))

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

        if self.meter is not None:
            self.meter.threshold = threshold

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
        """Build the panel, hidden. Built once and reused, so fixed tags are safe here."""
        rec = audio_recorder.require()
        slider_w = 200

        with dpg.window(label="Audio input",
                        modal=False,
                        show=False,
                        no_collapse=True,
                        autosize=True,
                        tag="audio_input_panel_window",  # tag
                        on_close=lambda: self.close()) as window_id:
            self.window_id = window_id

            dpg.add_text(f"Microphone: {rec.device_name}", color=DIM_TEXT)
            dpg.add_text("", tag="audio_input_status_text", color=DIM_TEXT)  # tag
            dpg.add_separator()

            with dpg.group(horizontal=True):
                self.meter = DPGVUMeter(width=28,
                                        height=150,
                                        border=1,
                                        min_value=METER_MIN,
                                        max_value=METER_MAX,
                                        yellow_start=METER_YELLOW_START,
                                        red_start=METER_RED_START,
                                        threshold_value=rec.silence_threshold,
                                        tooltip_text=("Microphone input level.\n"
                                                      f"Yellow from {METER_YELLOW_START:0.6g}, red from {METER_RED_START:0.6g} dBFS.\n"
                                                      "The gray line is the silence threshold."))
                with dpg.group():
                    for label, tag in (("Now", "audio_input_now_text"),  # tag
                                       ("Peak", "audio_input_peak_text"),  # tag
                                       (f"Loudest in {FLOOR_WINDOW:0.6g} s", "audio_input_floor_text")):  # tag
                        with dpg.group(horizontal=True):
                            dpg.add_text(f"{label}:", color=DIM_TEXT)
                            dpg.add_text("—", tag=tag)
                    dpg.add_spacer(height=8)
                    measure_button = dpg.add_button(label=f"{fa.ICON_RULER}  Measure the room",
                                                    callback=lambda: self._measure_the_room(),
                                                    tag="audio_input_measure_button")  # tag
                    dpg.bind_item_font(measure_button, self.themes_and_fonts.icon_font_solid)
                    dpg.add_text("Sets the threshold just above\nthe loudest of those seconds.", color=DIM_TEXT)

            dpg.add_separator()

            dpg.add_slider_float(label="Silence below",
                                 min_value=METER_MIN,
                                 max_value=METER_MAX,
                                 default_value=(rec.silence_threshold if rec.silence_threshold is not None
                                                else silencegate.DEFAULT_SILENCE_THRESHOLD),
                                 format="%.1f dBFS",
                                 width=slider_w,
                                 callback=self._on_threshold_slider,
                                 tag="audio_input_threshold_slider")  # tag
            dpg.add_checkbox(label="Measure at the start of each recording instead",
                             default_value=rec.silence_threshold is None,
                             callback=self._on_autodetect_checkbox,
                             tag="audio_input_autodetect_checkbox")  # tag

            dpg.add_spacer(height=6)
            dpg.add_checkbox(label="Stop when the speaker falls silent",
                             default_value=rec.autostop_timeout is not None,
                             callback=self._on_autostop_checkbox,
                             tag="audio_input_autostop_checkbox")  # tag
            dpg.add_slider_float(label="...after",
                                 min_value=AUTOSTOP_MIN,
                                 max_value=AUTOSTOP_MAX,
                                 default_value=(rec.autostop_timeout if rec.autostop_timeout is not None
                                                else silencegate.DEFAULT_AUTOSTOP_TIMEOUT),
                                 format="%.1f s",
                                 width=slider_w,
                                 callback=self._on_autostop_slider,
                                 tag="audio_input_autostop_slider")  # tag

            dpg.add_spacer(height=6)
            dpg.add_slider_float(label="Meter peak hold",
                                 min_value=PEAK_HOLD_MIN,
                                 max_value=PEAK_HOLD_MAX,
                                 default_value=rec.vu_peak_hold,
                                 format="%.1f s",
                                 width=slider_w,
                                 callback=self._on_peak_hold_slider,
                                 tag="audio_input_peak_hold_slider")  # tag

            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Reset to configured defaults",
                               callback=lambda: self._reset_to_configured_defaults(),
                               tag="audio_input_reset_button")  # tag
                dpg.add_button(label="Close",
                               callback=lambda: self.close(),
                               tag="audio_input_close_button")  # tag
