"""A simple, minimalistic VU ("voltage units", audio level) meter for DPG."""

__all__ = ["DPGVUMeter"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import threading
from typing import Optional, Union

import dearpygui.dearpygui as dpg

from .. import numutils

class DPGVUMeter:
    def __init__(self,
                 width: int,
                 height: int,
                 border: int,
                 min_value: float,
                 max_value: float,
                 yellow_start: float,
                 red_start: float,
                 threshold_value: Optional[float] = None,
                 tooltip_text: Optional[str] = None,
                 parent: Optional[Union[int, str]] = None):
        """A simple, minimalistic VU ("voltage units", audio level) meter.

        `width`: Width of the widget, in pixels.
        `height`: Width of the widget, in pixels.
        `border`: Border thickness, in pixels. Can be zero.

        `min_value`: Signal level corresponding to the bottom edge of the meter.
                     Cannot be less than -90.0 (silence level of 16-bit audio).

        `max_value`: Signal level corresponding to the top edge of the meter.
                     For dBFS, use 0.0.

        `yellow_start`: Signal level above which the yellow area begins.
                        The yellow area continues up to `red_start`.

                        For dBFS, -24.0 may be a useful value.

        `red_start`: Signal level above which the red area begins.
                     The red area continues up to `max_value`.

                     For dBFS, -6.0 may be a useful value.

        `threshold_value`: A fixed signal level that will have a line drawn on it.
                           Useful e.g. for a silence autostop threshold in an audio recorder.

                           If `None`, no threshold is shown.

        `tooltip_text`: Text for tooltip if the user mouses over the VU meter.
                        Can be used for explaining what this meter is showing
                        (e.g. "Mic input level dBFS").

        `parent`: DPG ID or tag of the GUI parent widget, usually a group.

        To send data to the VU meter, call the `update` method.
        """
        self._width = width
        self._height = height
        self._border = border

        if min_value == max_value:
            raise ValueError(f"DPGVUMeter.__init__: min and max cannot be the same, got both = {min_value:0.6g}")
        if min_value > max_value:
            raise ValueError(f"DPGVUMeter.__init__: min must be <= max; got min = {min_value:0.6g}, max = {max_value:0.6g}")
        if yellow_start > red_start:
            raise ValueError(f"DPGVUMeter.__init__: `yellow_start` must be <= `red_start`; got `yellow_start` = {yellow_start:0.6g}, `red_start` = {red_start:0.6g}")
        if not (min_value <= yellow_start <= max_value):
            raise ValueError(f"DPGVUMeter.__init__: `yellow_start` must be between min and max; got min = {min_value:0.6g}, yellow_start = {yellow_start:0.6g}, max = {max_value:0.6g}")
        if not (min_value <= red_start <= max_value):
            raise ValueError(f"DPGVUMeter.__init__: `red_start` must be between min and max; got min = {min_value:0.6g}, red_start = {red_start:0.6g}, max = {max_value:0.6g}")
        if (threshold_value is not None) and (not (min_value <= threshold_value <= max_value)):
            raise ValueError(f"DPGVUMeter.__init__: `threshold_value`, when specified, must be between min and max; got min = {min_value:0.6g}, threshold_value = {threshold_value:0.6g}, max = {max_value:0.6g}")

        if min_value < -90.0:
            logger.warning(f"DPGVUMeter.__init__: `min_value` must be >= -90.0, got {min_value}. Clipping it to -90.0.")
            min_value = -90.0
        self._min = min_value
        self._max = max_value

        self._yellow_start = yellow_start
        self._red_start = red_start
        self._threshold = threshold_value

        self._render_lock = threading.RLock()
        self.parent = parent if (parent is not None) else dpg.top_container_stack()  # behave like `with dpg.*` by default, but allow explicitly specifying the parent (DPG "runtime adding" mode).

        self.group = dpg.add_group(parent=self.parent)
        self.drawlist = dpg.add_drawlist(width=self._width, height=self._height, parent=self.group)
        self._update_geometry()

        if tooltip_text is not None:
            self.tooltip = dpg.add_tooltip(self.group)
            self.tooltip_text = dpg.add_text(tooltip_text, parent=self.tooltip)

        self._instant = min_value
        self._peak = min_value
        self.render()

    def get_width(self) -> int:
        return self._width
    def set_width(self, width: int) -> None:
        if width != self.width:
            with self._render_lock:
                self._width = width
                self._update_geometry()
                self.render()
    width = property(fget=get_width, fset=set_width, doc="The width of the VU meter, in pixels. Read/write.")

    def get_height(self) -> int:
        return self._height
    def set_height(self, height: int) -> None:
        if height != self.height:
            with self._render_lock:
                self._height = height
                self._update_geometry()
                self.render()
    height = property(fget=get_height, fset=set_height, doc="The height of the VU meter, in pixels. Read/write.")

    def _value_to_pixels(self, value: float) -> int:
        """Data value -> pixels, measured from bottom of widget, accounting for border."""
        value = numutils.clamp(value, self._min, self._max)
        return int((self._height - 2 * self._border) * ((value - self._min) / (self._max - self._min)))

    def _update_geometry(self) -> None:
        with self._render_lock:
            self._yellow_start_pixels = self._value_to_pixels(self._yellow_start)
            self._red_start_pixels = self._value_to_pixels(self._red_start)
            if self._threshold is not None:
                self._threshold_pixels = self._value_to_pixels(self._threshold)
            else:
                self._threshold_pixels = None
            dpg.delete_item(self.drawlist, children_only=True)
            dpg.configure_item(self.drawlist, width=self._width, height=self._height)

    def update(self, instant: float, peak: Optional[float]) -> None:
        """Send new values to the widget and redraw it.

        `instant`: Instant signal level, e.g. the maximum dBFS in a single audio frame or playback buffer.

        `peak`: Peak signal level, or `None` to not display a peak level.

                Note that peak hold semantics are up to you. `DPGVUMeter` does not do any peak holding of its own,
                it simply displays the peak value you send.

        If you're recording audio, you may want to connect this method to the VU readout of a `raven.common.audio.recorder`.

        If you need to convert linear signal values to dBFS, see `raven.common.audio.utils`.
        """
        self._instant = instant
        self._peak = peak
        self.render()

    def render(self) -> None:
        """Redraw the widget.

        Usually there is no need to call this manually; just send new data with `update`.
        """
        bgcolor = (64, 64, 64)  # cf. DPG default gray: (45, 45, 48).
        threshold_color = (96, 96, 96)
        peak_color = (160, 160, 140)
        green_on = (180, 255, 180)
        green_off = (90, 128, 90)
        yellow_on = (255, 255, 128)
        yellow_off = (128, 128, 64)
        red_on = (255, 128, 128)
        red_off = (128, 64, 64)

        # Naming convention:
        #   raw value in (self._min, self._max)
        #   *_pixels -> corresponding height in pixels, positive up
        #   *_y -> corresponding position in widget coordinates, positive down

        instant_pixels = self._value_to_pixels(self._instant)
        instant_y = self._height - instant_pixels
        if self._peak is not None:
            peak_pixels = self._value_to_pixels(self._peak)
            peak_y = self.height - peak_pixels
        if self._threshold is not None:
            threshold_y = self._height - self._threshold_pixels
        yellow_start_y = self._height - self._yellow_start_pixels
        red_start_y = self._height - self._red_start_pixels

        try:
            with self._render_lock:
                dpg.delete_item(self.drawlist, children_only=True)  # clear old content
                dpg.draw_rectangle((0, 0), (self._width, self._height), color=bgcolor, fill=bgcolor, parent=self.drawlist)

                b = self._border
                w = self._width - b
                h = self._height - b

                # Green area
                if self._instant >= self._yellow_start:  # completely lit up?
                    dpg.draw_rectangle((b, yellow_start_y), (w, h),
                                       color=green_on, fill=green_on, thickness=0, rounding=0, parent=self.drawlist)
                elif self._min < self._instant < self._yellow_start:  # partly lit up?
                    dpg.draw_rectangle((b, yellow_start_y), (w, instant_y),
                                       color=green_off, fill=green_off, thickness=0, rounding=0, parent=self.drawlist)
                    dpg.draw_rectangle((b, instant_y), (w, h),
                                       color=green_on, fill=green_on, thickness=0, rounding=0, parent=self.drawlist)
                else:  # self._instant <= self._min:  # not lit up
                    dpg.draw_rectangle((b, yellow_start_y), (w, h),
                                       color=green_off, fill=green_off, thickness=0, rounding=0, parent=self.drawlist)

                # Yellow area
                if self._instant >= self._red_start:  # completely lit up?
                    dpg.draw_rectangle((b, red_start_y), (w, yellow_start_y),
                                       color=yellow_on, fill=yellow_on, thickness=0, rounding=0, parent=self.drawlist)
                elif self._yellow_start < self._instant < self._red_start:  # partly lit up?
                    dpg.draw_rectangle((b, red_start_y), (w, instant_y),
                                       color=yellow_off, fill=yellow_off, thickness=0, rounding=0, parent=self.drawlist)
                    dpg.draw_rectangle((b, instant_y), (w, yellow_start_y),
                                       color=yellow_on, fill=yellow_on, thickness=0, rounding=0, parent=self.drawlist)
                else:  # self._instant <= self._yellow_start:  # not lit up
                    dpg.draw_rectangle((b, red_start_y), (w, yellow_start_y),
                                       color=yellow_off, fill=yellow_off, thickness=0, rounding=0, parent=self.drawlist)

                # Red area
                if self._instant >= self._max:  # completely lit up?
                    dpg.draw_rectangle((b, b), (w, red_start_y),
                                       color=red_on, fill=red_on, thickness=0, rounding=0, parent=self.drawlist)
                elif self._red_start < self._instant < self._max:  # partly lit up?
                    dpg.draw_rectangle((b, b), (w, instant_y),
                                       color=red_off, fill=red_off, thickness=0, rounding=0, parent=self.drawlist)
                    dpg.draw_rectangle((b, instant_y), (w, red_start_y),
                                       color=red_on, fill=red_on, thickness=0, rounding=0, parent=self.drawlist)
                else:  # self._instant <= self._red_start:  # not lit up
                    dpg.draw_rectangle((b, b), (w, red_start_y),
                                       color=red_off, fill=red_off, thickness=0, rounding=0, parent=self.drawlist)

                # Threshold (if set)
                if self._threshold is not None:
                    dpg.draw_line((b, threshold_y), (w, threshold_y),
                                  color=threshold_color, thickness=1, parent=self.drawlist)

                # Peak value (if specified)
                if self._peak is not None:
                    dpg.draw_line((b, peak_y), (w, peak_y),
                                  color=peak_color, thickness=1, parent=self.drawlist)
        except SystemError:  # GUI widget no longer exists; may happen during app shutdown
            pass
