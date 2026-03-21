"""Thumbnail grid view for raven-cherrypick.

Displays image thumbnails in a scrollable grid with click-to-select,
triage state borders (cherry/lemon/neutral), and filter views.

Each tile is a small DPG drawlist (for full control over borders,
image, and icon overlays) plus a text label. Layout is computed
manually for O(1) hit detection from mouse position.
"""

__all__ = ["ThumbnailGrid", "FilterMode"]

import logging
import threading
from enum import Enum
from typing import List, Optional

import numpy as np
import dearpygui.dearpygui as dpg

from ..common.gui import utils as guiutils
from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa
from . import config
from .triage import TriageState

logger = logging.getLogger(__name__)

# Counter for unique DPG tags.
_tag_counter = 0
_tag_lock = threading.Lock()


def _next_tag(prefix: str) -> str:
    global _tag_counter
    with _tag_lock:
        _tag_counter += 1
        return f"grid_{prefix}_{_tag_counter}"


class FilterMode(Enum):
    """Which tiles to show in the grid."""
    ALL = "all"
    CHERRY = "cherry"
    LEMON = "lemon"
    NEUTRAL = "neutral"


# Spacing between tiles (pixels).
_TILE_SPACING = 4
# Height reserved for the filename label below each tile.
_LABEL_HEIGHT = 18


class ThumbnailGrid:
    """Scrollable thumbnail grid with triage state display and click selection.

    Create once, then call `set_entries` after opening a folder.  The render
    loop must call `update` every frame.

    Thumbnails arrive asynchronously via `set_thumbnail`.  Tiles show a
    VHS noise placeholder until their thumbnail is ready.

    Thread-safe: all public methods and mouse handlers are guarded by an
    `RLock` (reentrant, because public methods call each other internally).
    """

    def __init__(self, parent: str | int,
                 width: int, height: int,
                 tile_size: int = config.DEFAULT_TILE_SIZE,
                 icon_font=None,
                 on_current_changed=None,
                 on_selection_changed=None,
                 on_double_click=None,
                 debug: bool = False):
        """
        *parent*: DPG parent container.
        *width*, *height*: initial grid panel size in pixels.
        *tile_size*: thumbnail tile size (square, pixels).
        *icon_font*: DPG font ID for FontAwesome icons (optional).
        *on_current_changed*: callback ``f(idx)`` when the current image changes.
        *on_selection_changed*: callback ``f()`` when the multi-selection changes.
        *on_double_click*: callback ``f(idx)`` on double-click.
        *debug*: show click position logging.
        """
        self._lock = threading.RLock()

        self._parent = parent
        self._width = width
        self._height = height
        self._tile_size = tile_size
        self._icon_font = icon_font
        self._on_current_changed = on_current_changed
        self._on_selection_changed = on_selection_changed
        self._on_double_click = on_double_click
        self._debug = debug

        # Data.
        self._filenames: list[str] = []  # parallel to triage manager indices
        self._triage_states: list[TriageState] = []
        self._n_images: int = 0

        # View state.
        self._filter = FilterMode.ALL
        self._visible: list[int] = []  # indices visible under current filter
        self._current: int = -1  # current image index (in full list)
        self._selected: set[int] = set()  # multi-selected indices

        # DPG textures for thumbnails.  idx -> texture tag.
        self._textures: dict[int, str] = {}

        # VHS noise placeholder textures (shared pool).
        self._noise_textures: list[str] = []

        # Layout state.
        self._n_cols: int = 1
        self._col_width: float = 0.0
        self._row_height: float = 0.0

        # DPG items.
        self._child_window_tag = _next_tag("child")
        dpg.add_child_window(parent=parent, tag=self._child_window_tag,
                             width=width, height=height, border=False)

        # Per-tile drawlists.  Maps visible-list position -> drawlist tag.
        self._tile_drawlists: dict[int, str] = {}
        self._tile_labels: dict[int, str] = {}

        self._needs_rebuild = False

        # Mouse handlers.
        self._handler_tag = _next_tag("handlers")
        with dpg.handler_registry(tag=self._handler_tag):
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left,
                                        callback=self._on_click)
            dpg.add_mouse_double_click_handler(button=dpg.mvMouseButton_Left,
                                               callback=self._on_double_click_handler)

        self._last_click_idx: int = -1  # for shift+click range selection
        self.input_enabled: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_entries(self, filenames: list[str],
                    triage_states: list[TriageState]) -> None:
        """Set the image list.  Call after opening a folder.

        *filenames* and *triage_states* must be parallel lists (same length,
        same ordering as the TriageManager).
        """
        with self._lock:
            self._filenames = list(filenames)
            self._triage_states = list(triage_states)
            self._n_images = len(filenames)
            self._current = 0 if self._n_images > 0 else -1
            self._selected = set()  # no auto-select; user selects explicitly
            self._last_click_idx = -1
            self._clear_textures()
            self._recompute_visible()
            self._needs_rebuild = True

    def set_thumbnail(self, idx: int, flat_rgba) -> None:
        """Update the thumbnail for image *idx*.

        *flat_rgba*: flat float32 array (tile_size * tile_size * 4).
        Creates or updates the DPG texture, then redraws the tile if visible.
        Stale thumbnails from a previous tile size are silently discarded.
        """
        with self._lock:
            ts = self._tile_size
            expected = ts * ts * 4
            if len(flat_rgba) != expected:
                return  # stale thumbnail from previous tile size — discard
            if idx in self._textures:
                dpg.set_value(self._textures[idx], flat_rgba)
            else:
                tex_tag = _next_tag("thumb_tex")
                with dpg.texture_registry():
                    dpg.add_dynamic_texture(ts, ts,
                                            default_value=flat_rgba,
                                            tag=tex_tag)
                self._textures[idx] = tex_tag

            # Redraw tile if it's currently visible in the grid.
            if idx in self._visible:
                vis_pos = self._visible.index(idx)
                if vis_pos in self._tile_drawlists:
                    self._draw_tile(idx, self._tile_drawlists[vis_pos])

    def set_tile_size(self, size: int) -> None:
        """Change the tile size.  Clears all textures (caller must restart pipeline).

        Also clears noise pool textures — caller should call
        `set_noise_pool` with new tiles matching the new size.
        """
        with self._lock:
            self._tile_size = size
            self._clear_textures()
            self._clear_noise_pool()
            self._needs_rebuild = True

    def set_noise_pool(self, tiles: List[np.ndarray]) -> None:
        """Set VHS noise placeholder textures from DPG-flat float32 arrays.

        Each entry must be a flat array of ``tile_size * tile_size * 4`` floats.
        Old noise textures are deleted immediately.

        Generate tiles with `raven.common.video.postprocessor.vhs_noise_pool`.
        """
        with self._lock:
            self._clear_noise_pool()
            ts = self._tile_size
            for flat in tiles:
                tag = _next_tag("noise_tex")
                with dpg.texture_registry():
                    dpg.add_dynamic_texture(ts, ts, default_value=flat, tag=tag)
                self._noise_textures.append(tag)
            logger.info(f"ThumbnailGrid.set_noise_pool: instance 0x{id(self):x}: {len(tiles)} tiles loaded")

    def set_filter(self, mode: FilterMode) -> None:
        """Set the active filter (which triage states to show)."""
        with self._lock:
            if mode == self._filter:
                return
            self._filter = mode
            self._recompute_visible()
            self._needs_rebuild = True

    def set_size(self, width: int, height: int) -> None:
        """Resize the grid panel (call from viewport resize callback)."""
        with self._lock:
            self._width = width
            self._height = height
            dpg.configure_item(self._child_window_tag, width=width, height=height)
            self._needs_rebuild = True

    def set_current(self, idx: int) -> None:
        """Set the current image (shown in main view)."""
        with self._lock:
            if idx == self._current:
                return
            old = self._current
            self._current = idx
            # Redraw old and new tiles.
            self._redraw_tile_by_idx(old)
            self._redraw_tile_by_idx(idx)
            self._scroll_to_current()
            if self._on_current_changed is not None:
                self._on_current_changed(idx)

    def update_triage_state(self, idx: int, state: TriageState) -> None:
        """Notify the grid that a triage state changed (after file move)."""
        with self._lock:
            if idx < 0 or idx >= self._n_images:
                return
            self._triage_states[idx] = state
            old_visible = list(self._visible)
            self._recompute_visible()
            if self._visible != old_visible:
                self._needs_rebuild = True
            else:
                self._redraw_tile_by_idx(idx)

    @property
    def current(self) -> int:
        """Index of the current image, or -1."""
        with self._lock:
            return self._current

    @property
    def selected(self) -> set[int]:
        """Set of multi-selected indices (not including current)."""
        with self._lock:
            return set(self._selected)

    @property
    def filter_mode(self) -> FilterMode:
        with self._lock:
            return self._filter

    @property
    def visible_count(self) -> int:
        """Number of images visible under the current filter."""
        with self._lock:
            return len(self._visible)

    @property
    def visible(self) -> list[int]:
        """List of image indices visible under the current filter."""
        with self._lock:
            return list(self._visible)

    @property
    def n_cols(self) -> int:
        """Number of columns in the current grid layout."""
        with self._lock:
            return self._n_cols

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def navigate_next(self) -> Optional[int]:
        """Move to the next visible image.  Returns new index, or None."""
        with self._lock:
            return self._navigate_by(1)

    def navigate_prev(self) -> Optional[int]:
        """Move to the previous visible image.  Returns new index, or None."""
        with self._lock:
            return self._navigate_by(-1)

    def navigate_row_down(self) -> Optional[int]:
        """Move down by one row."""
        with self._lock:
            return self._navigate_by(self._n_cols)

    def navigate_row_up(self) -> Optional[int]:
        """Move up by one row."""
        with self._lock:
            return self._navigate_by(-self._n_cols)

    def navigate_page_down(self) -> Optional[int]:
        """Move down by a page (visible rows)."""
        with self._lock:
            visible_rows = max(1, int(self._height / self._row_height)) - 1
            return self._navigate_by(visible_rows * self._n_cols)

    def navigate_page_up(self) -> Optional[int]:
        """Move up by a page (visible rows)."""
        with self._lock:
            visible_rows = max(1, int(self._height / self._row_height)) - 1
            return self._navigate_by(-visible_rows * self._n_cols)

    def navigate_first(self) -> Optional[int]:
        """Jump to the first visible image."""
        with self._lock:
            if not self._visible:
                return None
            self.set_current(self._visible[0])
            return self._current

    def navigate_last(self) -> Optional[int]:
        """Jump to the last visible image."""
        with self._lock:
            if not self._visible:
                return None
            self.set_current(self._visible[-1])
            return self._current

    def navigate_next_with_state(self, state: TriageState) -> Optional[int]:
        """Jump forward to the next image with triage *state*.

        Searches the full image list (ignoring filter), wrapping around.
        Returns the new index, or ``None`` if no image has that state.
        """
        with self._lock:
            return self._navigate_to_state(state, direction=1)

    def navigate_prev_with_state(self, state: TriageState) -> Optional[int]:
        """Jump backward to the previous image with triage *state*.

        Searches the full image list (ignoring filter), wrapping around.
        Returns the new index, or ``None`` if no image has that state.
        """
        with self._lock:
            return self._navigate_to_state(state, direction=-1)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _notify_selection_changed(self) -> None:
        if self._on_selection_changed is not None:
            self._on_selection_changed()

    def select_all(self) -> None:
        """Select all visible images."""
        with self._lock:
            self._selected = set(self._visible)
            self._needs_rebuild = True  # many tiles changed
            self._notify_selection_changed()

    def deselect_all(self) -> None:
        """Clear multi-selection."""
        with self._lock:
            if not self._selected:
                return
            self._selected.clear()
            self._needs_rebuild = True
            self._notify_selection_changed()

    def invert_selection(self) -> None:
        """Invert selection among visible images."""
        with self._lock:
            visible_set = set(self._visible)
            self._selected = visible_set - self._selected
            self._needs_rebuild = True
            self._notify_selection_changed()

    def toggle_select(self, idx: int) -> None:
        """Toggle *idx* in/out of the multi-selection (Ctrl+click)."""
        with self._lock:
            if idx in self._selected:
                self._selected.discard(idx)
            else:
                self._selected.add(idx)
            self._redraw_tile_by_idx(idx)
            self._notify_selection_changed()

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Call from the render loop every frame."""
        with self._lock:
            if self._needs_rebuild:
                self._rebuild()
                self._needs_rebuild = False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Remove all DPG items.  Call on app shutdown."""
        with self._lock:
            self._clear_textures()
            self._clear_noise_pool()
            dpg.delete_item(self._handler_tag)
            dpg.delete_item(self._child_window_tag)

    # ------------------------------------------------------------------
    # Internal: texture management
    # ------------------------------------------------------------------

    def _clear_textures(self) -> None:
        """Delete all thumbnail DPG textures."""
        for tex_tag in self._textures.values():
            guiutils.maybe_delete_item(tex_tag)
        self._textures.clear()

    def _clear_noise_pool(self) -> None:
        """Delete all VHS noise placeholder textures."""
        for tex_tag in self._noise_textures:
            guiutils.maybe_delete_item(tex_tag)
        self._noise_textures.clear()

    # ------------------------------------------------------------------
    # Internal: layout
    # ------------------------------------------------------------------

    def _compute_layout(self) -> None:
        """Compute grid layout parameters from current width and tile size.

        Must match the actual DPG-rendered layout. DPG adds `item_spacing`
        (8px horizontal, 4px vertical) between sibling widgets automatically.
        """
        ts = self._tile_size
        dpg_item_spacing_x = 8
        dpg_item_spacing_y = config.DPG_ITEM_SPACING_Y
        self._col_width = ts + dpg_item_spacing_x
        # Row = drawlist + spacing + text (with frame padding) + spacing (between rows).
        text_h = config.FONT_SIZE + 2 * config.DPG_FRAME_PADDING_Y
        self._row_height = ts + dpg_item_spacing_y + text_h + dpg_item_spacing_y
        usable = self._width - config.DPG_SCROLLBAR_SIZE
        self._n_cols = max(1, int(usable / self._col_width))

    def _rebuild(self) -> None:
        """Tear down and re-create all tile DPG items."""
        dpg.delete_item(self._child_window_tag, children_only=True)
        self._tile_drawlists.clear()
        self._tile_labels.clear()
        self._compute_layout()

        ts = self._tile_size
        n_cols = self._n_cols

        for vis_pos, idx in enumerate(self._visible):
            col = vis_pos % n_cols
            if col == 0:
                row_tag = _next_tag("row")
                dpg.add_group(horizontal=True, parent=self._child_window_tag,
                              tag=row_tag)

            # Tile group.
            tile_tag = _next_tag("tile")
            dpg.add_group(parent=row_tag, tag=tile_tag)

            # Drawlist for image + borders + icons.
            dl_tag = _next_tag("tile_dl")
            dpg.add_drawlist(width=ts, height=ts,
                             parent=tile_tag, tag=dl_tag)
            self._tile_drawlists[vis_pos] = dl_tag

            # Filename label — truncate to fit tile width.
            # At font size 20, ~9px average character width (variable-width font).
            max_chars = max(4, ts // 9)
            name = self._filenames[idx]
            if len(name) > max_chars:
                name = name[:max_chars - 1] + "\u2026"
            label_tag = _next_tag("label")
            dpg.add_text(name, parent=tile_tag, tag=label_tag, wrap=ts)
            self._tile_labels[vis_pos] = label_tag

            # Add tooltip with full filename (on the tile group, not the drawlist).
            with dpg.tooltip(tile_tag):
                dpg.add_text(self._filenames[idx])

            # Draw tile contents.
            self._draw_tile(idx, dl_tag)

        self._scroll_to_current()

    def _draw_tile(self, idx: int, drawlist_tag: str) -> None:
        """Draw a single tile's contents on its drawlist."""
        dpg.delete_item(drawlist_tag, children_only=True)
        ts = self._tile_size

        # Thumbnail image (or VHS noise placeholder).
        if idx in self._textures:
            dpg.draw_image(self._textures[idx],
                           pmin=(0, 0), pmax=(ts, ts),
                           parent=drawlist_tag)
        elif self._noise_textures:
            dpg.draw_image(self._noise_textures[idx % len(self._noise_textures)],
                           pmin=(0, 0), pmax=(ts, ts),
                           parent=drawlist_tag)
        else:
            dpg.draw_rectangle(pmin=(0, 0), pmax=(ts, ts),
                               fill=(55, 55, 58, 255),
                               parent=drawlist_tag)

        # Lemon dimming overlay (rejects fade into background).
        state = self._triage_states[idx] if idx < len(self._triage_states) else TriageState.NEUTRAL
        if state is TriageState.LEMON:
            dpg.draw_rectangle(pmin=(0, 0), pmax=(ts - 1, ts - 1),
                               fill=(0, 0, 0, 128),
                               parent=drawlist_tag)

        # Selection tint.
        if idx in self._selected:
            dpg.draw_rectangle(pmin=(0, 0), pmax=(ts - 1, ts - 1),
                               fill=config.SELECTION_TINT,
                               parent=drawlist_tag)

        # Triage border.
        if state is TriageState.CHERRY:
            border_color = config.CHERRY_COLOR
        elif state is TriageState.LEMON:
            border_color = config.LEMON_COLOR
        else:
            border_color = config.NEUTRAL_BORDER_COLOR
        dpg.draw_rectangle(pmin=(0, 0), pmax=(ts - 1, ts - 1),
                           color=border_color, thickness=2,
                           parent=drawlist_tag)

        # Current image indicator (inner border).
        if idx == self._current:
            dpg.draw_rectangle(pmin=(3, 3), pmax=(ts - 4, ts - 4),
                               color=config.CURRENT_COLOR, thickness=2,
                               parent=drawlist_tag)

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

    # ------------------------------------------------------------------
    # Internal: filter and visibility
    # ------------------------------------------------------------------

    def _recompute_visible(self) -> None:
        """Rebuild the visible-index list based on the current filter."""
        if self._filter is FilterMode.ALL:
            self._visible = list(range(self._n_images))
        else:
            target = {FilterMode.CHERRY: TriageState.CHERRY,
                      FilterMode.LEMON: TriageState.LEMON,
                      FilterMode.NEUTRAL: TriageState.NEUTRAL}[self._filter]
            self._visible = [i for i in range(self._n_images)
                             if self._triage_states[i] is target]

    # ------------------------------------------------------------------
    # Internal: navigation helpers
    # ------------------------------------------------------------------

    def _navigate_by(self, delta: int) -> Optional[int]:
        """Move current by *delta* positions in the visible list.

        Respects filter: when the current image is hidden, navigation starts
        from its virtual position in the full list.
        """
        if not self._visible:
            return None

        # Find current position in visible list.
        if self._current in self._visible:
            vis_pos = self._visible.index(self._current)
        else:
            # Current is hidden — find nearest position.
            vis_pos = self._find_nearest_visible(self._current)

        new_vis_pos = max(0, min(len(self._visible) - 1, vis_pos + delta))
        new_idx = self._visible[new_vis_pos]
        self.set_current(new_idx)
        return new_idx

    def _navigate_to_state(self, state: TriageState, direction: int) -> Optional[int]:
        """Jump to the next/prev image with *state* in the full list.

        Wraps around. Returns the new index, or ``None`` if none found.
        """
        n = self._n_images
        if n == 0:
            return None
        start = self._current if self._current >= 0 else 0
        for offset in range(1, n):
            candidate = (start + direction * offset) % n
            if self._triage_states[candidate] is state:
                self.set_current(candidate)
                return candidate
        return None

    def _find_nearest_visible(self, idx: int) -> int:
        """Find the visible-list position nearest to *idx* in the full list.

        Used when the current image is hidden by a filter.
        """
        best_pos = 0
        best_dist = abs(self._visible[0] - idx) if self._visible else 0
        for pos, vis_idx in enumerate(self._visible):
            dist = abs(vis_idx - idx)
            if dist < best_dist:
                best_dist = dist
                best_pos = pos
        return best_pos

    def _scroll_to_current(self) -> None:
        """Scroll the grid to make the current tile visible."""
        if self._current < 0 or self._current not in self._visible:
            return
        vis_pos = self._visible.index(self._current)
        row = vis_pos // self._n_cols
        row_y = row * self._row_height

        # Scroll so the row is visible (with some margin).
        scroll_y = dpg.get_y_scroll(self._child_window_tag)
        max_scroll = dpg.get_y_scroll_max(self._child_window_tag)
        if row_y < scroll_y:
            dpg.set_y_scroll(self._child_window_tag, max(0, row_y - _TILE_SPACING))
        elif row_y + self._row_height > scroll_y + self._height:
            target = row_y + self._row_height - self._height + _TILE_SPACING
            dpg.set_y_scroll(self._child_window_tag, min(target, max_scroll))

    def _redraw_tile_by_idx(self, idx: int) -> None:
        """Redraw a single tile (if it's visible) after state change."""
        if idx < 0 or idx not in self._visible:
            return
        vis_pos = self._visible.index(idx)
        if vis_pos in self._tile_drawlists:
            self._draw_tile(idx, self._tile_drawlists[vis_pos])

    # ------------------------------------------------------------------
    # Internal: hit detection
    # ------------------------------------------------------------------

    def _hit_test(self) -> Optional[int]:
        """O(1) hit test: return image index under the mouse, or None."""
        if not guiutils.is_mouse_inside_widget(self._child_window_tag):
            return None

        local_x, local_y = guiutils.get_mouse_relative_pos(self._child_window_tag)
        content_y = local_y + dpg.get_y_scroll(self._child_window_tag)

        if self._col_width <= 0 or self._row_height <= 0:
            return None

        col = int(local_x / self._col_width)
        row = int(content_y / self._row_height)

        if col >= self._n_cols:
            return None

        vis_pos = row * self._n_cols + col
        if vis_pos < 0 or vis_pos >= len(self._visible):
            return None

        # Check that the click is on the tile, not the spacing.
        tile_x = local_x - col * self._col_width
        tile_y = content_y - row * self._row_height
        if tile_x > self._tile_size or tile_y > self._tile_size:
            return None

        return self._visible[vis_pos]

    # ------------------------------------------------------------------
    # Internal: mouse handlers
    # ------------------------------------------------------------------

    def _on_click(self, sender, app_data) -> None:
        """Handle single click on a tile."""
        with self._lock:
            if not self.input_enabled:
                return

            if self._debug and guiutils.is_mouse_inside_widget(self._child_window_tag):
                local_x, local_y = guiutils.get_mouse_relative_pos(self._child_window_tag)
                content_y = local_y + dpg.get_y_scroll(self._child_window_tag)
                logger.info(f"ThumbnailGrid._on_click: local=({local_x:.0f},{local_y:.0f}) "
                            f"content_y={content_y:.0f} row_h={self._row_height:.0f} "
                            f"col_w={self._col_width:.0f} "
                            f"row={int(content_y / self._row_height)} "
                            f"col={int(local_x / self._col_width)}")

            idx = self._hit_test()
            if idx is None:
                return

            ctrl = (dpg.is_key_down(dpg.mvKey_LControl)
                    or dpg.is_key_down(dpg.mvKey_RControl))
            shift = (dpg.is_key_down(dpg.mvKey_LShift)
                     or dpg.is_key_down(dpg.mvKey_RShift))

            if shift and self._last_click_idx >= 0 and self._last_click_idx in self._visible:
                # Range select from last click to this click (in visible order).
                a = self._visible.index(self._last_click_idx)
                b = self._visible.index(idx) if idx in self._visible else a
                lo, hi = min(a, b), max(a, b)
                self._selected = set(self._visible[lo:hi + 1])
                self._needs_rebuild = True  # many tiles changed
                self._notify_selection_changed()
            elif ctrl:
                self.toggle_select(idx)  # already notifies
            else:
                # Bare click: set current and replace selection with this one image.
                old_selected = self._selected
                self._selected = {idx}
                if old_selected != self._selected:
                    self._needs_rebuild = True
                    self._notify_selection_changed()
                self.set_current(idx)

            self._last_click_idx = idx

    def _on_double_click_handler(self, sender, app_data) -> None:
        """Handle double-click on a tile."""
        with self._lock:
            if not self.input_enabled:
                return
            idx = self._hit_test()
            if idx is None:
                return
            self._selected.clear()
            self.set_current(idx)
            if self._on_double_click is not None:
                self._on_double_click(idx)
