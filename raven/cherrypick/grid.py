"""Cherrypick's thumbnail grid: the shared grid widget, plus what triage adds to a tile.

Everything about laying out tiles, filling them with images, selecting and navigating them lives in
`raven.common.gui.thumbnailgrid`. What is here is what only Cherrypick means: a triage state per image
(cherry / lemon / neutral), the filter views over those states, and the marks drawn on a tile to say which
is which — plus compare mode's badges and the resize beacon.

The split follows the drawing order, which is why there are two hooks rather than one. Lemon dimming goes
*under* the tile's furniture (`draw_underlay`), so the triage border stays legible through it; icons and
badges go on top (`draw_overlay`).
"""

__all__ = ["FilterMode", "TriageGrid"]

import logging
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Optional

import dearpygui.dearpygui as dpg

from ..common.gui.thumbnailgrid import ThumbnailGrid
from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa
from . import config
from .triage import TriageState

logger = logging.getLogger(__name__)


class FilterMode(Enum):
    """Which tiles to show in the grid."""
    ALL = "all"
    CHERRY = "cherry"
    LEMON = "lemon"
    NEUTRAL = "neutral"


_FILTER_TO_STATE = {FilterMode.CHERRY: TriageState.CHERRY,
                    FilterMode.LEMON: TriageState.LEMON,
                    FilterMode.NEUTRAL: TriageState.NEUTRAL}


class TriageGrid(ThumbnailGrid):
    """A thumbnail grid whose tiles carry triage state, compare badges and a beacon."""

    def __init__(self, parent: str | int,
                 width: int, height: int,
                 tile_size: int = config.DEFAULT_TILE_SIZE,
                 icon_font=None,
                 on_current_changed=None,
                 on_selection_changed=None,
                 on_double_click=None,
                 debug: bool = False):
        self._triage_states: list[TriageState] = []
        self._filter = FilterMode.ALL
        self._compare_badges: dict[int, int] = {}   # image_idx -> badge number (1-9)
        self._compare_active_idx: int = -1          # currently cycling tile (-1 = none)
        self._compare_active_alpha: float = 0.0     # fade-out [0, 1]
        self._beacon_idx: int = -1                  # image index to flash (-1 = none)
        self._beacon_alpha: float = 0.0             # fade-out [0, 1]
        super().__init__(parent, width, height,
                         tile_size=tile_size,
                         icon_font=icon_font,
                         on_current_changed=on_current_changed,
                         on_selection_changed=on_selection_changed,
                         on_double_click=on_double_click,
                         font_size=config.FONT_SIZE,
                         frame_padding_y=config.DPG_FRAME_PADDING_Y,
                         item_spacing_y=config.DPG_ITEM_SPACING_Y,
                         scrollbar_size=config.DPG_SCROLLBAR_SIZE,
                         selection_tint=config.SELECTION_TINT,
                         current_color=config.CURRENT_COLOR,
                         border_color=config.NEUTRAL_BORDER_COLOR,
                         smooth_scrolling=config.SMOOTH_SCROLLING,
                         smooth_scrolling_step_parameter=config.SMOOTH_SCROLLING_STEP_PARAMETER,
                         scroll_end_flash_duration=config.SCROLL_ENDS_HERE_DURATION,
                         debug=debug)

    # ------------------------------------------------------------------
    # Entries and filtering
    # ------------------------------------------------------------------

    def set_entries(self, filenames: Sequence[str],
                    triage_states: Sequence[TriageState] | None = None) -> None:
        """Set the image list.  Call after opening a folder.

        *filenames* and *triage_states* must be parallel lists (same length, same ordering as the
        TriageManager).
        """
        with self._lock:
            self._triage_states = list(triage_states) if triage_states is not None else []
            super().set_entries(filenames)
            self._apply_filter()

    def set_filter(self, mode: FilterMode) -> None:
        """Set the active filter (which triage states to show)."""
        with self._lock:
            if mode == self._filter:
                return
            self._filter = mode
            self._apply_filter()

    def update_triage_state(self, idx: int, state: TriageState) -> None:
        """Notify the grid that a triage state changed (after file move)."""
        with self._lock:
            if idx < 0 or idx >= len(self._triage_states):
                return
            self._triage_states[idx] = state
            before = self.visible
            self._apply_filter()
            if self.visible == before:  # still shown in the same place; only its marks changed
                self.refresh_tile(idx)

    def _apply_filter(self) -> None:
        """Hand the grid the indices the current filter admits.

        "All" counts *entries*, not triage states: the two lists are parallel by contract, but a caller
        that supplies no states would otherwise be told nothing is visible rather than everything.
        """
        if self._filter is FilterMode.ALL:
            self.set_visible(range(self._n_entries))
        else:
            target = _FILTER_TO_STATE[self._filter]
            self.set_visible([i for i, state in enumerate(self._triage_states) if state is target])

    @property
    def filter_mode(self) -> FilterMode:
        with self._lock:
            return self._filter

    def _state_of(self, idx: int) -> TriageState:
        if 0 <= idx < len(self._triage_states):
            return self._triage_states[idx]
        return TriageState.NEUTRAL

    # ------------------------------------------------------------------
    # Compare mode overlays
    # ------------------------------------------------------------------

    def set_compare_badges(self, mapping: Mapping[int, int]) -> None:
        """Show number badges (1–9) on tiles.

        *mapping*: ``{image_idx: badge_number}``.
        """
        with self._lock:
            self._compare_badges = dict(mapping)
            for idx in self._compare_badges:
                self.refresh_tile(idx)

    def clear_compare_badges(self) -> None:
        """Remove all compare badges."""
        with self._lock:
            old = self._compare_badges
            self._compare_badges = {}
            for idx in old:
                self.refresh_tile(idx)

    def set_compare_active(self, idx: int, alpha: float) -> None:
        """Highlight the active compare tile.

        *idx*: image index of the currently cycling tile.
        *alpha*: fade-out intensity [0, 1] from ``CompareMode.fade_alpha()``.
        """
        with self._lock:
            prev = self._compare_active_idx
            self._compare_active_idx = idx
            self._compare_active_alpha = alpha
            if prev >= 0 and prev != idx:
                self.refresh_tile(prev)
            self.refresh_tile(idx)

    def clear_compare_active(self) -> None:
        """Remove the active-compare highlight."""
        with self._lock:
            prev = self._compare_active_idx
            self._compare_active_idx = -1
            self._compare_active_alpha = 0.0
            if prev >= 0:
                self.refresh_tile(prev)

    # --- Beacon (resize orientation flash) ---

    def set_beacon(self, idx: int, alpha: float) -> None:
        """Highlight a tile with a fade-out beacon overlay.

        *idx*: image index to flash.
        *alpha*: intensity [0, 1], driven from the app render loop.
        """
        with self._lock:
            prev = self._beacon_idx
            self._beacon_idx = idx
            self._beacon_alpha = alpha
            if prev >= 0 and prev != idx:
                self.refresh_tile(prev)
            self.refresh_tile(idx)

    def clear_beacon(self) -> None:
        """Remove the beacon highlight."""
        with self._lock:
            prev = self._beacon_idx
            self._beacon_idx = -1
            self._beacon_alpha = 0.0
            if prev >= 0:
                self.refresh_tile(prev)

    # ------------------------------------------------------------------
    # Navigation by triage state
    # ------------------------------------------------------------------

    def navigate_next_with_state(self, state: TriageState) -> Optional[int]:
        with self._lock:
            return self._navigate_to_state(state, 1)

    def navigate_prev_with_state(self, state: TriageState) -> Optional[int]:
        with self._lock:
            return self._navigate_to_state(state, -1)

    def _navigate_to_state(self, state: TriageState, direction: int) -> Optional[int]:
        """Jump to the next/prev image with *state* in the full list.

        Wraps around. Returns the new index, or ``None`` if none found.
        """
        n = len(self._triage_states)
        if n == 0:
            return None
        start = self._current if self._current >= 0 else 0
        for offset in range(1, n):
            candidate = (start + direction * offset) % n
            if self._triage_states[candidate] is state:
                self.set_current(candidate)
                return candidate
        return None

    # ------------------------------------------------------------------
    # Tile decoration
    # ------------------------------------------------------------------

    def border_color_for(self, idx: int) -> tuple:
        """Triage state, as the tile's border colour."""
        state = self._state_of(idx)
        if state is TriageState.CHERRY:
            return config.CHERRY_COLOR
        if state is TriageState.LEMON:
            return config.LEMON_COLOR
        return config.NEUTRAL_BORDER_COLOR

    def draw_underlay(self, idx: int, drawlist_tag: str) -> None:
        """Lemon dimming: rejects fade into the background, but keep their border readable."""
        if self._state_of(idx) is not TriageState.LEMON:
            return
        ts = self._tile_size
        dpg.draw_rectangle(pmin=(0, 0), pmax=(ts - 1, ts - 1),
                           fill=(0, 0, 0, 128),
                           parent=drawlist_tag)

    def draw_overlay(self, idx: int, drawlist_tag: str) -> None:
        """Triage icon, compare highlight, beacon and compare badge, in that stacking order."""
        ts = self._tile_size
        state = self._state_of(idx)

        # Triage icon (top-right corner).
        if self._icon_font is not None:
            icon_item = None
            if state is TriageState.CHERRY:
                icon_item = dpg.draw_text((ts - 18, 2), fa.ICON_STAR,
                                          color=config.CHERRY_COLOR, size=14,
                                          parent=drawlist_tag)
            elif state is TriageState.LEMON:
                icon_item = dpg.draw_text((ts - 18, 2), fa.ICON_LEMON,
                                          color=(180, 180, 180, 255), size=14,
                                          parent=drawlist_tag)
            if icon_item is not None:
                dpg.bind_item_font(icon_item, self._icon_font)

        # Compare mode: active tile highlight.
        if idx == self._compare_active_idx and self._compare_active_alpha > 0:
            r, g, b, a_max = config.COMPARE_FADE_COLOR
            a = int(a_max * self._compare_active_alpha)
            dpg.draw_rectangle(pmin=(0, 0), pmax=(ts - 1, ts - 1),
                               fill=(r, g, b, a),
                               parent=drawlist_tag)

        # Beacon overlay (resize orientation flash).
        if idx == self._beacon_idx and self._beacon_alpha > 0:
            r, g, b, a_max = config.BEACON_COLOR
            a = int(a_max * self._beacon_alpha)
            dpg.draw_rectangle(pmin=(0, 0), pmax=(ts - 1, ts - 1),
                               fill=(r, g, b, a),
                               parent=drawlist_tag)

        # Compare mode: number badge.
        if idx in self._compare_badges:
            badge_num = self._compare_badges[idx]
            badge_text = str(badge_num)
            font_size = max(14, ts // 6)
            # Box sized for a single digit. Offsets are empirical —
            # DPG draw_text positioning doesn't match simple glyph metrics.
            pad_x = max(2, font_size // 5)
            pad_y = max(1, font_size // 8)
            box_w = max(1, int(font_size * 0.55)) + 2 * pad_x
            box_h = max(1, int(font_size * 0.75)) + 2 * pad_y
            # Semi-transparent background for readability.
            bx = 2
            by = 2
            dpg.draw_rectangle(pmin=(bx, by),
                               pmax=(bx + box_w, by + box_h),
                               fill=(128, 128, 128, 160),
                               parent=drawlist_tag)
            # Digit position: nudged right and up within the box.
            tx = bx + pad_x + max(1, font_size // 10) - 1
            ty = by + pad_y - max(1, int(font_size * 0.2))
            dpg.draw_text((tx, ty), badge_text,
                          color=(255, 255, 255, 255), size=font_size,
                          parent=drawlist_tag)
