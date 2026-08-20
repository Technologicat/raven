"""Scrollable thumbnail grid: tiles with labels, click selection, and lazily filled images.

Each tile is a small DPG drawlist (for full control over borders, image and overlays) plus a text label.
Layout is computed manually rather than read back from DPG, which is what makes hit detection O(1) from a
mouse position — and what makes the arithmetic testable without a GUI.

Thumbnails arrive asynchronously via `set_thumbnail`; until one does, a tile shows a placeholder from the
pool set by `set_noise_pool`. Producing the images is not this widget's job — see
`raven.common.image.thumbnails.ThumbnailPipeline` for the decoder both current callers use. An entry whose
picture needs no decoding — a folder, a file type with an icon — takes `set_shared_image` instead, which
draws one texture on any number of tiles.

**Extending it.** Three hooks, all no-ops here, let an owner decorate tiles without this module learning what
the decoration means. Draw order is load-bearing, so the two drawing hooks are not interchangeable:

- `draw_underlay(idx, drawlist_tag)` — draw over the image but *under* the tile's tint, border and number,
  for decoration those must stay legible through.
- `draw_overlay(idx, drawlist_tag)` — draw on top of a finished tile. Cherrypick puts its triage icons,
  compare badges and beacon here.
- `border_color_for(idx)` — the tile's border colour. Cherrypick colours it by triage state.

Whatever those hooks draw is redrawn whenever the tile is, so an owner changing the state behind them calls
`refresh_tile(idx)` and needs to know nothing else.

Thread-safe: all public methods and mouse handlers are guarded by an `RLock` (reentrant, because public
methods call each other internally).
"""

__all__ = ["ThumbnailGrid"]

import logging
import threading
from collections.abc import Callable, Sequence
from typing import Optional, Union

import numpy as np
import dearpygui.dearpygui as dpg

from unpythonic import sym

from ...vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa
from . import animation as gui_animation
from . import keyboardmark
from . import utils as guiutils
from .gridnav import resolve_nav_target

logger = logging.getLogger(__name__)

# Counter for unique DPG tags.
_tag_counter = 0
_tag_lock = threading.Lock()


def _next_tag(prefix: str) -> str:
    global _tag_counter
    with _tag_lock:
        _tag_counter += 1
        return f"grid_{prefix}_{_tag_counter}"


# Spacing between tiles (pixels).
_TILE_SPACING = 4


class _CursorPulse(gui_animation.Animation):
    """Breathe whichever tile currently carries the cursor mark."""

    def __init__(self, grid: "ThumbnailGrid"):
        super().__init__(ambient=True)
        self._grid = grid

    def render_frame(self, t: int) -> sym:
        if (t - self.t0) / 10**9 > keyboardmark.PULSE_SECONDS:  # prevent loss of accuracy in long sessions
            self.reset()
        self._grid.paint_cursor(gui_animation.pulsating_alpha(self.t0, t, keyboardmark.PULSE_SECONDS))
        return gui_animation.action_continue


class ThumbnailGrid:
    """Scrollable thumbnail grid with click selection.

    Create once, then call `set_entries` after opening a folder. The render loop must call `update` every
    frame.

    The grid knows how many entries there are and which of them to show; it does not know what they *are*.
    The owner keeps the paths, decides the order, and supplies thumbnails as they become available.
    """

    def __init__(self, parent: str | int,
                 width: int, height: int,
                 tile_size: int = 128,
                 icon_font=None,
                 on_current_changed: Optional[Callable] = None,
                 on_selection_changed: Optional[Callable] = None,
                 on_double_click: Optional[Callable] = None,
                 label_height: int = 18,
                 font_size: int = 20,
                 frame_padding_y: int = 3,
                 item_spacing_x: int = 8,
                 item_spacing_y: int = 4,
                 scrollbar_size: int = 14,
                 selection_tint: tuple = (255, 255, 255, 40),
                 current_color: tuple = keyboardmark.COLOR,
                 border_color: tuple = (60, 60, 65, 255),
                 empty_tile_color: tuple = (55, 55, 58, 255),
                 show_position_numbers: bool = True,
                 allow_multi_select: bool = True,
                 smooth_scrolling: bool = True,
                 smooth_scrolling_step_parameter: float = 0.8,
                 scroll_end_flash_duration: float = 0.5,
                 scroll_end_flasher: Optional[gui_animation.ScrollEndFlasher] = None,
                 debug: bool = False):
        """
        *parent*: DPG parent container.
        *width*, *height*: initial grid panel size in pixels.
        *tile_size*: thumbnail tile size (square, pixels).
        *icon_font*: DPG font ID for icon glyphs an overlay may draw (optional).
        *on_current_changed*: callback ``f(idx)`` when the current entry changes.
        *on_selection_changed*: callback ``f()`` when the multi-selection changes.
        *on_double_click*: callback ``f(idx)`` on double-click.
        *label_height*: height reserved for the filename label below each tile.
        *font_size*, *frame_padding_y*, *item_spacing_x*, *item_spacing_y*, *scrollbar_size*: DPG's own
            metrics, which the layout arithmetic has to match because it is computed rather than measured.
            The defaults are Raven's standard theme; pass the app's values if it differs.
        *selection_tint*, *current_color*, *border_color*, *empty_tile_color*: tile colours.
        *show_position_numbers*: draw each tile's 1-based position in its corner.
        *allow_multi_select*: whether Ctrl-click and Shift-click extend the selection. `False` makes every
            click select exactly one entry, for an owner that can only act on one — a file dialog opened
            without multi-selection, say, where letting the user mark five and then honouring one would be
            worse than not letting them mark five.
        *smooth_scrolling*: glide to the current tile instead of jumping to it. A rebuild still repositions
            instantly — see `_scroll_to_current`.
        *smooth_scrolling_step_parameter*: nondimensional rate in (0, 1], independent of the render FPS.
            See `raven.common.gui.animation.SmoothScrolling`.
        *scroll_end_flash_duration*: fadeout seconds for the "there is no further this way" overlay. One is
            built automatically when *icon_font* is given, since the overlay targets a child window only
            this widget knows about. Pass 0 to do without.
        *scroll_end_flasher*: use this overlay instead of building one. For an app wanting different icons
            or a `custom_finish_pred`.
        *debug*: log click positions.
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

        self._label_height = label_height
        self._font_size = font_size
        self._frame_padding_y = frame_padding_y
        self._item_spacing_x = item_spacing_x
        self._item_spacing_y = item_spacing_y
        self._scrollbar_size = scrollbar_size
        self._selection_tint = selection_tint
        self._current_color = current_color
        self._border_color = border_color
        self._empty_tile_color = empty_tile_color
        self._show_position_numbers = show_position_numbers
        self._allow_multi_select = allow_multi_select
        self._smooth_scrolling = smooth_scrolling
        self._smooth_scrolling_step_parameter = smooth_scrolling_step_parameter
        self._scroll_end_flasher = scroll_end_flasher
        self._owns_scroll_end_flasher = False  # only tear down one we built ourselves

        # Data.
        self._labels: list[str] = []
        self._n_entries: int = 0

        # View state.
        self._visible: list[int] = []  # entry indices currently shown, in display order
        self._current: int = -1
        self._selected: set[int] = set()

        # The drawn rectangle that marks the cursor, and which tile it belongs to, so the pulsation has
        # something to recolour. Both are `None` whenever the mark is not on screen — a cursor scrolled out
        # of the built range has no rectangle, and neither does a grid nobody has put a cursor in yet.
        self._cursor_rect: Optional[Union[str, int]] = None
        self._cursor_rect_idx: Optional[int] = None
        self._cursor_pulse: Optional[gui_animation.Animation] = None

        # DPG textures for thumbnails.  idx -> texture tag.
        self._textures: dict[int, str] = {}

        # Textures belonging to somebody else, drawn in a tile's place.  idx -> texture tag.
        # Separate from `_textures` because these are *not* ours to delete: one icon texture typically
        # stands in for many entries, so deleting it with the first of them would blank the rest.
        self._shared_images: dict[int, str] = {}

        # Placeholder textures (shared pool), shown until a thumbnail arrives.
        self._noise_textures: list[str] = []

        # Layout state.
        self._n_cols: int = 1
        self._col_width: float = 0.0
        self._row_height: float = 0.0

        # DPG items.
        self._child_window_tag = _next_tag("child")
        dpg.add_child_window(parent=parent, tag=self._child_window_tag,
                             width=width, height=height, border=False)

        # Built here rather than by the owner: the overlay targets the child window above, which nothing
        # outside this class has a reason to know the tag of.
        if self._scroll_end_flasher is None and icon_font is not None and scroll_end_flash_duration > 0:
            self._scroll_end_flasher = gui_animation.ScrollEndFlasher(target=self._child_window_tag,  # tag
                                                                      tag=_next_tag("scroll_end_flasher"),
                                                                      duration=scroll_end_flash_duration,
                                                                      font=icon_font,
                                                                      text_top=fa.ICON_ARROWS_UP_TO_LINE,
                                                                      text_bottom=fa.ICON_ARROWS_DOWN_TO_LINE)
            self._owns_scroll_end_flasher = True

        # Per-tile drawlists.  Maps visible-list position -> drawlist tag.
        self._tile_drawlists: dict[int, str] = {}
        self._tile_labels: dict[int, str] = {}

        # The group currently holding the tiles. A rebuild builds its replacement hidden, shows it, and
        # retires this one for the next `update` to collect — see `_rebuild`.
        self._content_tag: Optional[str] = None
        self._retired_content: Optional[str] = None
        # Whether the textures the shown tiles draw from have been deleted since they were built. If they
        # have, those tiles are dangling references and must not be retired — see `_rebuild`.
        self._textures_cleared_since_rebuild = False

        self._needs_rebuild = False
        # Deferred scroll after rebuild: counts down frames to retry.
        # DPG needs at least one render frame after item creation before
        # get_y_scroll_max reflects the new content height; sometimes two.
        self._scroll_countdown: int = 0

        # Deferred callbacks — fired from update() outside the lock.
        # Prevents the lock from being held across expensive owner-side work, which would block the main
        # loop and deadlock with split_frame() waiters.
        self._pending_current_changed: Optional[int] = None
        self._pending_double_click: Optional[int] = None

        # Mouse handlers.
        self._handler_tag = _next_tag("handlers")
        with dpg.handler_registry(tag=self._handler_tag):
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left,
                                        callback=self._on_click)
            dpg.add_mouse_double_click_handler(button=dpg.mvMouseButton_Left,
                                               callback=self._on_double_click_handler)
            dpg.add_mouse_wheel_handler(callback=self._on_wheel)

        self._last_click_idx: int = -1  # for shift+click range selection
        self.input_enabled: bool = True

        # A grid that exists has a cursor to mark, so this starts here and stops in `destroy`. An owner that
        # keeps a grid it is not showing — a file dialog in table view — can stop it in the meantime.
        self.start_cursor_pulse()

    # ------------------------------------------------------------------
    # Hooks for owners
    # ------------------------------------------------------------------

    def draw_underlay(self, idx: int, drawlist_tag: str) -> None:
        """Draw over entry `idx`'s image but *under* its tint, border and number. No-op here.

        For decoration the tile's own furniture should stay legible through — a dimming wash over a
        de-emphasized entry, say, which must not also dim the border that says why.
        """

    def draw_overlay(self, idx: int, drawlist_tag: str) -> None:
        """Draw anything extra on top of entry `idx`'s finished tile. No-op here; override to decorate.

        Called at the end of every tile draw, with the tile's image, selection tint and border already in
        place, and the drawlist's origin at the tile's top-left corner (so coordinates run 0..`tile_size`).
        """

    def border_color_for(self, idx: int) -> tuple:
        """The border colour for entry `idx`. Override to colour tiles by state."""
        return self._border_color

    def is_selectable(self, idx: int) -> bool:
        """Whether entry `idx` may be part of the selection. Always, here; override to exclude some.

        The cursor still moves onto an excluded entry — it has to, or there would be no way to reach it
        with the keyboard, and nothing to double-click. What it cannot join is the *selection*, so a bulk
        action never sees it.

        For an owner whose list holds entries that are not candidates for whatever the selection feeds: a
        file dialog listing `..` and the directories you navigate through, none of which it can return.
        Showing them selected while ignoring them is the failure this prevents, and it reads as a bug
        rather than as a rule.
        """
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_entries(self, labels: Sequence[str]) -> None:
        """Set the entry labels, and show all of them. Call after opening a folder.

        Resets the current entry to the first, clears the selection and drops every thumbnail — the indices
        now mean something else, so a texture kept from before would be a picture of the wrong file.
        """
        with self._lock:
            self._labels = list(labels)
            self._n_entries = len(self._labels)
            self._current = 0 if self._n_entries > 0 else -1
            self._selected = set()  # no auto-select; the user selects explicitly
            self._last_click_idx = -1
            self._clear_textures()
            self._visible = list(range(self._n_entries))
            self._needs_rebuild = True

    def set_visible(self, indices: Sequence[int]) -> None:
        """Show only `indices`, in the given order — the grid's filtering and sorting surface.

        The grid does not know what any filter *means*; an owner computes the list and hands it over.
        """
        with self._lock:
            new_visible = list(indices)
            if new_visible == self._visible:
                return
            self._visible = new_visible
            self._needs_rebuild = True

    def set_thumbnail(self, idx: int, flat_rgba) -> None:
        """Update the thumbnail for entry *idx*.

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

    def set_shared_image(self, idx: int, texture_tag: Optional[str]) -> None:
        """Draw an existing texture as entry *idx*'s image, instead of a placeholder.

        For entries whose picture is known without decoding anything — a file-type icon, a folder — where
        one texture serves many entries and creating a copy per entry would be a texture per file.

        The texture stays the owner's: this widget draws it and never deletes it, and the owner is
        responsible for it still existing while any tile refers to it. It must be `tile_size` square, like
        a thumbnail. Pass `None` to go back to the placeholder.

        A thumbnail set later for the same entry takes precedence, so an owner may seed a tile with an icon
        and replace it once the real image arrives.
        """
        with self._lock:
            if texture_tag is None:
                self._shared_images.pop(idx, None)
            else:
                self._shared_images[idx] = texture_tag
            self._redraw_tile_by_idx(idx)

    def has_thumbnail(self, idx: int) -> bool:
        """Whether entry `idx` already has its image, so an owner can avoid asking for it twice."""
        with self._lock:
            return idx in self._textures

    def set_tile_size(self, size: int) -> None:
        """Change the tile size.  Clears all textures (caller must restart its thumbnail production).

        Also clears the placeholder pool — the caller should call `set_noise_pool` with new tiles matching
        the new size.
        """
        with self._lock:
            self._tile_size = size
            self._clear_textures()
            self._clear_noise_pool()
            self._needs_rebuild = True

    def set_noise_pool(self, tiles: list[np.ndarray]) -> None:
        """Set placeholder textures from DPG-flat float32 arrays, shown until a thumbnail arrives.

        Each entry must be a flat array of ``tile_size * tile_size * 4`` floats. Old placeholder textures
        are deleted immediately.

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

    def set_size(self, width: int, height: int) -> None:
        """Resize the grid panel (call from viewport resize callback)."""
        with self._lock:
            self._width = width
            self._height = height
            dpg.configure_item(self._child_window_tag, width=width, height=height)
            self._needs_rebuild = True

    def set_current(self, idx: int) -> None:
        """Set the current entry.

        The ``on_current_changed`` callback is deferred to ``update()`` (outside the lock) to avoid holding
        the lock across whatever the owner does in response.
        """
        with self._lock:
            if idx == self._current:
                return
            old = self._current
            self._current = idx
            self._redraw_tile_by_idx(old)
            self._redraw_tile_by_idx(idx)
            self._scroll_to_current()
            self._pending_current_changed = idx

    def refresh_tile(self, idx: int) -> None:
        """Redraw entry `idx`'s tile — how an owner reflects a change behind `draw_overlay`."""
        with self._lock:
            self._redraw_tile_by_idx(idx)

    @property
    def current(self) -> int:
        """Index of the current entry, or -1."""
        with self._lock:
            return self._current

    @property
    def selected(self) -> set[int]:
        """Set of multi-selected indices."""
        with self._lock:
            return set(self._selected)

    @property
    def visible_count(self) -> int:
        with self._lock:
            return len(self._visible)

    @property
    def visible(self) -> list[int]:
        with self._lock:
            return list(self._visible)

    def position_of(self, idx: int) -> Optional[int]:
        """Where *idx* sits among the visible items, counting from 1 — or `None` if it is filtered out.

        The number a tile shows and the number a caller would print as "*n* of *m*", so that both come
        from the same place rather than from each caller's own arithmetic over `visible`.
        """
        with self._lock:
            try:
                return self._visible.index(idx) + 1
            except ValueError:
                return None

    @property
    def n_cols(self) -> int:
        with self._lock:
            return self._n_cols

    @property
    def tile_size(self) -> int:
        with self._lock:
            return self._tile_size

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def navigate_next(self) -> Optional[int]:
        with self._lock:
            return self._navigate_by(1)

    def navigate_prev(self) -> Optional[int]:
        with self._lock:
            return self._navigate_by(-1)

    def navigate_row_down(self) -> Optional[int]:
        with self._lock:
            return self._navigate_by(self._n_cols)

    def navigate_row_up(self) -> Optional[int]:
        with self._lock:
            return self._navigate_by(-self._n_cols)

    def navigate_page_down(self) -> Optional[int]:
        with self._lock:
            return self._navigate_by(self._n_cols * self._rows_per_page())

    def navigate_page_up(self) -> Optional[int]:
        with self._lock:
            return self._navigate_by(-self._n_cols * self._rows_per_page())

    def navigate_first(self) -> Optional[int]:
        with self._lock:
            if not self._visible:
                return None
            if self._visible[0] == self._current:
                self._flash_scroll_end("top")
            self.set_current(self._visible[0])  # no-op when unchanged
            return self._visible[0]

    def navigate_last(self) -> Optional[int]:
        with self._lock:
            if not self._visible:
                return None
            if self._visible[-1] == self._current:
                self._flash_scroll_end("bottom")
            self.set_current(self._visible[-1])  # no-op when unchanged
            return self._visible[-1]

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _notify_selection_changed(self) -> None:
        if self._on_selection_changed is not None:
            self._on_selection_changed()

    def _apply_selection(self, new_selected: set[int]) -> None:
        """Replace the selection, redrawing only the tiles whose state actually changed.

        **Never a rebuild.** A selection change alters what is *drawn on* a tile, not which tiles exist or
        where they sit, so tearing the lattice down and re-creating it is work proportional to the whole
        directory in order to change the appearance of two tiles. It is also visible: on a few hundred
        entries the grid blanks and re-populates over a couple of frames, worst where the click was — near
        the end of a long listing, since the tiles are rebuilt from the top.

        The symmetric difference is what makes this exact: a click that moves the selection touches the
        tile gaining it and the one losing it, and a range selection touches only the ends that changed.
        """
        changed = self._selected ^ new_selected
        if not changed:
            return
        self._selected = new_selected
        for idx in changed:
            self._redraw_tile_by_idx(idx)
        self._notify_selection_changed()

    def _selectable_visible(self) -> set[int]:
        return {idx for idx in self._visible if self.is_selectable(idx)}

    def select_all(self) -> None:
        with self._lock:
            self._apply_selection(self._selectable_visible())

    def deselect_all(self) -> None:
        with self._lock:
            self._apply_selection(set())

    def invert_selection(self) -> None:
        with self._lock:
            self._apply_selection(self._selectable_visible() - self._selected)

    def toggle_select(self, idx: int) -> None:
        with self._lock:
            if not self.is_selectable(idx):
                return
            if idx in self._selected:
                self._selected.discard(idx)
            else:
                self._selected.add(idx)
            self._redraw_tile_by_idx(idx)
            self._notify_selection_changed()

    # ------------------------------------------------------------------
    # Render loop
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Call from the render loop every frame.

        Fires deferred ``on_current_changed`` / ``on_double_click`` callbacks *outside* the lock. This is
        critical: an owner's callback may do expensive work or wait for a frame, and holding the lock across
        that would block the main loop and deadlock with `split_frame` waiters.
        """
        pending_current = None
        pending_dblclick = None
        with self._lock:
            # Collect the content group a previous rebuild swapped out. A tick has passed since it was
            # hidden, so the replacement is on screen and destroying this cannot leave a gap.
            if self._retired_content is not None:
                guiutils.maybe_delete_item(self._retired_content)
                self._retired_content = None
            if self._needs_rebuild:
                self._rebuild()
                self._needs_rebuild = False
            elif self._scroll_countdown > 0:
                # Deferred scroll: retry for a few frames after rebuild,
                # giving DPG time to settle get_y_scroll_max. Instant, not smooth — see `_scroll_to_current`.
                self._scroll_to_current(smooth=False, flash_ends=False)
                self._scroll_countdown -= 1
            if self._pending_current_changed is not None:
                pending_current = self._pending_current_changed
                self._pending_current_changed = None
            if self._pending_double_click is not None:
                pending_dblclick = self._pending_double_click
                self._pending_double_click = None
        # Callbacks fire outside the lock.
        if pending_current is not None and self._on_current_changed is not None:
            self._on_current_changed(pending_current)
        if pending_dblclick is not None and self._on_double_click is not None:
            self._on_double_click(pending_dblclick)

    def visible_on_screen(self) -> list[int]:
        """Entry indices whose tiles were actually rendered in the last frame.

        What a lazy fill asks before deciding which thumbnails to produce: `visible` is everything the
        filter admits, which in a large folder is mostly off-screen and mostly not worth decoding.

        Ask a *tile*, never the scrolling container — see `dpg-notes.md`, "To find which rows are on
        screen". Needs at least one rendered frame; before that, everything reads as not visible.
        """
        with self._lock:
            on_screen = []
            for vis_pos, idx in enumerate(self._visible):
                tag = self._tile_drawlists.get(vis_pos)
                if tag is not None and dpg.does_item_exist(tag) and dpg.is_item_visible(tag):  # tag
                    on_screen.append(idx)
            return on_screen

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def start_cursor_pulse(self) -> None:
        """Set the cursor mark breathing. Idempotent."""
        # A drawn rectangle takes a colour, not a theme, so `PulsatingColor` cannot reach it and this
        # recolours the item itself once a frame. It shares `pulsating_alpha` with that animation, so a grid
        # cursor and a table cursor on screen together are the same shade at the same moment.
        #
        # Ambient: it says nothing is happening, so an app that throttles its idle frame rate should keep
        # throttling. It still gets drawn at that rate, which is what makes a two-second cycle survive it.
        if self._cursor_pulse is not None:
            return
        self._cursor_pulse = gui_animation.animator.add(_CursorPulse(self))

    def stop_cursor_pulse(self) -> None:
        """Stop the cursor breathing, and leave the mark at full strength. Idempotent."""
        if self._cursor_pulse is None:
            return
        gui_animation.animator.cancel(self._cursor_pulse)
        self._cursor_pulse = None
        self.paint_cursor(255)

    def paint_cursor(self, alpha: int) -> None:
        """Set the alpha of the cursor mark, if it is currently drawn anywhere."""
        if self._cursor_rect is None:
            return
        with guiutils.nonexistent_ok():
            dpg.configure_item(self._cursor_rect, color=(*self._current_color[:3], alpha))

    def destroy(self) -> None:
        """Remove all DPG items.  Call on app shutdown."""
        with self._lock:
            # Released in the reverse of the order they were acquired.
            self.stop_cursor_pulse()
            # Before the child window goes: a scroll still in flight would keep writing to a deleted item.
            gui_animation.SmoothScrolling.stop(self._child_window_tag)
            if self._owns_scroll_end_flasher:
                self._scroll_end_flasher.destroy()
            self._clear_textures()
            self._clear_noise_pool()
            guiutils.maybe_delete_item(self._handler_tag)
            guiutils.maybe_delete_item(self._child_window_tag)

    # ------------------------------------------------------------------
    # Internal: texture management
    # ------------------------------------------------------------------

    def _clear_textures(self) -> None:
        """Delete all thumbnail DPG textures, and forget any shared ones.

        The shared ones are dropped rather than deleted — they belong to the owner. Dropping them is right
        at every call site: the indices have been reassigned, or the tile size has changed, so a mapping
        kept from before would put the wrong picture on the tile or one of the wrong size.

        **Any tiles drawn from these textures go with them, or are marked not to outlive them.** Every image
        a tile draws is one of the textures being deleted here, so a tile group left alive afterwards is a
        set of dangling references — which DPG answers with a hard error rather than a blank tile. Two
        groups can be in that position, and they need different handling because they are at different
        stages:

        - A group already *retired* by an earlier rebuild is deleted here and now. It is hidden and its
          replacement is up, so nothing is lost by taking it a tick early.
        - The group currently *shown* cannot be taken here — it is what the user is looking at until the
          rebuild lands. Instead the rebuild is told not to retire it, and deletes it as soon as its
          replacement is shown. That is the ordering a folder change takes, and the one that actually bites:
          at the moment this runs there is usually no retired group at all.
        """
        self._textures_cleared_since_rebuild = True
        if self._retired_content is not None:
            guiutils.maybe_delete_item(self._retired_content)
            self._retired_content = None
        for tex_tag in self._textures.values():
            guiutils.maybe_delete_item(tex_tag)
        self._textures.clear()
        self._shared_images.clear()

    def _clear_noise_pool(self) -> None:
        """Delete all placeholder textures."""
        for tex_tag in self._noise_textures:
            guiutils.maybe_delete_item(tex_tag)
        self._noise_textures.clear()

    # ------------------------------------------------------------------
    # Internal: layout
    # ------------------------------------------------------------------

    def _compute_layout(self) -> None:
        """Compute grid layout parameters from current width and tile size.

        Must match the actual DPG-rendered layout: DPG adds `item_spacing` between sibling widgets
        automatically, and the arithmetic here is what hit detection and scrolling both trust.
        """
        ts = self._tile_size
        self._col_width = ts + self._item_spacing_x
        # Row = drawlist + spacing + text (with frame padding) + spacing (between rows).
        text_h = self._font_size + 2 * self._frame_padding_y
        self._row_height = ts + self._item_spacing_y + text_h + self._item_spacing_y
        usable = self._width - self._scrollbar_size
        self._n_cols = max(1, int(usable / self._col_width))

    def _rows_per_page(self) -> int:
        """How many whole rows fit in the panel — the step for page-wise navigation."""
        if self._row_height <= 0:
            return 1
        return max(1, int(self._height / self._row_height))

    def _rebuild(self) -> None:
        """Re-create all tile DPG items, and swap them in when they are ready.

        **Built into a fresh hidden group, then swapped** — the pattern Visualizer's info panel and
        annotation tooltip already use, for the reason they use it: tearing the old content down first
        leaves every frame until the new content exists rendering an empty panel, and on a few hundred tiles
        that is a visible blank-and-repopulate.

        The child window itself keeps its identity across the swap. That is load-bearing rather than
        incidental: `SmoothScrolling` instances are keyed by it and the scroll-end flasher targets it, so a
        swap that replaced the container would strand both.

        Every tag comes from the module-level monotonic counter, so the new group cannot collide with an old
        one that DPG has not collected yet — a duplicate ID takes the process down rather than raising.
        """
        self._compute_layout()

        old_content = self._content_tag
        new_content = _next_tag("content")
        dpg.add_group(parent=self._child_window_tag, tag=new_content, show=False)

        # Built into local maps and swapped in with the widgets: until the swap, what is on screen is still
        # the old content, and anything asking which tiles are visible should be told about *those*.
        tile_drawlists: dict[int, str] = {}
        tile_labels: dict[int, str] = {}

        ts = self._tile_size
        n_cols = self._n_cols
        row_tag = None

        for vis_pos, idx in enumerate(self._visible):
            col = vis_pos % n_cols
            if col == 0:
                row_tag = _next_tag("row")
                dpg.add_group(horizontal=True, parent=new_content,
                              tag=row_tag)

            # Tile group.
            tile_tag = _next_tag("tile")
            dpg.add_group(parent=row_tag, tag=tile_tag)

            # Drawlist for image + borders + overlays.
            dl_tag = _next_tag("tile_dl")
            dpg.add_drawlist(width=ts, height=ts,
                             parent=tile_tag, tag=dl_tag)
            tile_drawlists[vis_pos] = dl_tag

            # Label — truncate to fit tile width.
            # At font size 20, ~9px average character width (variable-width font).
            max_chars = max(4, ts // 9)
            name = self._labels[idx]
            if len(name) > max_chars:
                name = name[:max_chars - 1] + "…"
            label_tag = _next_tag("label")
            dpg.add_text(name, parent=tile_tag, tag=label_tag, wrap=ts)
            tile_labels[vis_pos] = label_tag

            # Tooltip with the full label (on the tile group, not the drawlist).
            with dpg.tooltip(tile_tag):
                dpg.add_text(self._labels[idx])

            # Draw tile contents.
            self._draw_tile(idx, dl_tag)

        # The swap. Old hidden *first*, then new shown — the order Visualizer's info panel and annotation
        # tooltip already use, and arrived at here the long way round.
        #
        # Showing the new one first looks safer on paper: a frame caught between the two calls renders
        # both, and since the old content comes first the viewport shows it "unchanged". That reasoning
        # holds only while the two contents are the same. When they are not — a filter change, a new
        # folder — "unchanged" means *stale*, and the user sees the previous listing for a frame after
        # clicking. Observed in Cherrypick, filter-switching a folder of a hundred images.
        #
        # This order can instead render neither for a frame. That is a blank flash rather than a wrong
        # one, and it is bounded by two adjacent calls rather than by the build, which is what the hidden
        # build already took care of. A frame of nothing beats a frame of the wrong thing.
        self._tile_drawlists = tile_drawlists
        self._tile_labels = tile_labels
        self._content_tag = new_content
        if old_content is not None and dpg.does_item_exist(old_content):  # tag
            dpg.configure_item(old_content, show=False)
        dpg.configure_item(new_content, show=True)
        # Retired rather than deleted here, and collected on the next `update`. Visualizer's version of this
        # waits for a frame and then deletes, which it can because it always runs off the render thread —
        # this widget cannot assume that, since Cherrypick drives `update` *from* the render loop, where
        # waiting for a frame can never succeed. Letting a tick pass costs one hidden group and needs no
        # such assumption; a hidden group renders nothing, so it is free until it goes.
        #
        # **It pins no textures.** A drawlist *references* a texture; the textures belong to `_textures`,
        # `_shared_images` and the placeholder pool. A re-filter or a re-sort does not touch those — that is
        # what makes it cheap, since the thumbnails already in hand are not decoded again — so the retired
        # tiles and the new ones point at the same textures, and what is duplicated for one tick is the DPG
        # item structures, on the CPU side.
        #
        # A *folder change* is the other case: it clears the textures before the rebuild, so the old tiles
        # reference deleted ones and must not outlive this call. Deleting them now costs nothing — the
        # replacement is already shown, which is what the tick of grace was protecting against in the first
        # place.
        if self._textures_cleared_since_rebuild:
            guiutils.maybe_delete_item(old_content)
            self._retired_content = None
        else:
            self._retired_content = old_content
        self._textures_cleared_since_rebuild = False

        # Defer scroll — DPG needs a render frame (sometimes two) after
        # item creation before get_y_scroll_max reflects the new content.
        # Retry for a few frames to be safe.
        self._scroll_countdown = 3

    def _draw_tile(self, idx: int, drawlist_tag: str) -> None:
        """Draw a single tile's contents on its drawlist."""
        dpg.delete_item(drawlist_tag, children_only=True)
        if idx == self._cursor_rect_idx:
            # The cursor mark was among the children that just went. Forgotten here rather than checked for
            # later: DPG reuses the ids of deleted items, so a stale one does not stay invalid, it starts
            # naming something else.
            self._cursor_rect = None
            self._cursor_rect_idx = None
        ts = self._tile_size

        # Thumbnail image (or owner-supplied icon, or placeholder).
        if idx in self._textures:
            dpg.draw_image(self._textures[idx],
                           pmin=(0, 0), pmax=(ts, ts),
                           parent=drawlist_tag)
        elif idx in self._shared_images:
            dpg.draw_image(self._shared_images[idx],
                           pmin=(0, 0), pmax=(ts, ts),
                           parent=drawlist_tag)
        elif self._noise_textures:
            dpg.draw_image(self._noise_textures[idx % len(self._noise_textures)],
                           pmin=(0, 0), pmax=(ts, ts),
                           parent=drawlist_tag)
        else:
            dpg.draw_rectangle(pmin=(0, 0), pmax=(ts, ts),
                               fill=self._empty_tile_color,
                               parent=drawlist_tag)

        self.draw_underlay(idx, drawlist_tag)

        # Selection tint.
        if idx in self._selected:
            dpg.draw_rectangle(pmin=(0, 0), pmax=(ts - 1, ts - 1),
                               fill=self._selection_tint,
                               parent=drawlist_tag)

        # Border.
        dpg.draw_rectangle(pmin=(0, 0), pmax=(ts - 1, ts - 1),
                           color=self.border_color_for(idx), thickness=2,
                           parent=drawlist_tag)

        # Current-entry indicator (inner border).
        if idx == self._current:
            self._cursor_rect = dpg.draw_rectangle(pmin=(3, 3), pmax=(ts - 4, ts - 4),
                                                   color=self._current_color, thickness=2,
                                                   parent=drawlist_tag)
            self._cursor_rect_idx = idx

        # Position number (lower-left corner).
        if self._show_position_numbers and idx in self._visible:
            vis_pos = self._visible.index(idx)
            num_text = str(vis_pos + 1)
            num_size = max(10, min(14, ts // 8))
            # Positioned inside the border (2px) with a small margin.
            nx = 4
            ny = ts - num_size - 4
            dpg.draw_text((nx, ny), num_text,
                          color=(255, 255, 255, 120), size=num_size,
                          parent=drawlist_tag)

        self.draw_overlay(idx, drawlist_tag)

    # ------------------------------------------------------------------
    # Internal: navigation helpers
    # ------------------------------------------------------------------

    def _navigate_by(self, delta: int) -> Optional[int]:
        """Move current by *delta* positions in the visible list.

        Respects the filter: when the current entry is hidden, navigation resolves from its gap position in
        the visible list. See `gridnav.resolve_nav_target`.
        """
        new_idx = resolve_nav_target(self._visible, self._current, delta)
        if new_idx is None:
            return None
        if new_idx == self._current:  # `resolve_nav_target` clamped: we are at that end already
            self._flash_scroll_end("bottom" if delta > 0 else "top")
        self.set_current(new_idx)  # no-op when unchanged
        return new_idx

    def _scroll_to_current(self, smooth: Optional[bool] = None, flash_ends: bool = True) -> None:
        """Scroll the grid to make the current tile visible.

        *smooth*: `None` uses the widget's configured setting; `False` forces an instant reposition.
        *flash_ends*: announce arrival at the top or bottom of the scroll, as the other scrolling views do.

        The rebuild path forces `smooth=False`: for a frame or two after items are created
        `get_y_scroll_max` is stale, so the target clamps to the wrong place and is corrected on a later
        retry. A jump lands wrong invisibly; a glide would animate *toward* wrong first. It also passes
        `flash_ends=False`, since a repositioning nobody asked for has arrived nowhere.
        """
        if self._current < 0 or self._current not in self._visible:
            return
        vis_pos = self._visible.index(self._current)
        row = vis_pos // self._n_cols
        row_y = row * self._row_height

        # Scroll so the row is visible (with some margin).
        scroll_y = dpg.get_y_scroll(self._child_window_tag)
        max_scroll = dpg.get_y_scroll_max(self._child_window_tag)
        if row_y < scroll_y:
            target = max(0, row_y - _TILE_SPACING)
        elif row_y + self._row_height > scroll_y + self._height:
            target = min(row_y + self._row_height - self._height + _TILE_SPACING, max_scroll)
        else:  # already fully on screen
            return

        gui_animation.SmoothScrolling.scroll(target_child_window=self._child_window_tag,
                                             target_y_scroll=int(target),
                                             smooth=(self._smooth_scrolling if smooth is None else smooth),
                                             smooth_step=self._smooth_scrolling_step_parameter,
                                             flasher=(self._scroll_end_flasher if flash_ends else None))

    def _flash_scroll_end(self, where: str) -> None:
        """Say "you asked to go further and there is no further", if the owner gave us a flasher.

        *where*: "top" or "bottom".

        Covers the case the scroll cannot: a cursor clamped at the last row requests no scroll, so there is
        nothing for `SmoothScrolling` to detect. *Arriving* at an end is announced by the scroll itself,
        which is where the other views announce it too — the flasher says "here is the wall" as you reach
        it, not only once you have walked into it.

        With less than a screenful there is nothing to scroll, and both ends are where you are — so both
        are flashed, whichever direction was refused. That is `show_by_position`'s own rule for this case,
        which is what the wheel path and Visualizer's info panel already do.
        """
        if self._scroll_end_flasher is None:
            return
        if dpg.get_y_scroll_max(self._child_window_tag) == 0:  # tag
            where = "both"
        self._scroll_end_flasher.show(where=where)

    def _redraw_tile_by_idx(self, idx: int) -> None:
        """Redraw a single tile (if it's visible) after a state change."""
        if idx < 0 or idx not in self._visible:
            return
        vis_pos = self._visible.index(idx)
        if vis_pos in self._tile_drawlists:
            self._draw_tile(idx, self._tile_drawlists[vis_pos])

    # ------------------------------------------------------------------
    # Internal: hit detection
    # ------------------------------------------------------------------

    def _hit_test(self) -> Optional[int]:
        """O(1) hit test: return the entry index under the mouse, or None."""
        if not guiutils.is_mouse_inside_widget(self._child_window_tag):
            return None

        local_x, local_y = guiutils.get_mouse_relative_pos(self._child_window_tag)
        content_y = local_y + dpg.get_y_scroll(self._child_window_tag)

        return self.hit_test_at(local_x, content_y)

    def hit_test_at(self, local_x: float, content_y: float) -> Optional[int]:
        """Which entry sits at panel-local `local_x` and content-space `content_y`, if any.

        Split out from the mouse handler so the arithmetic can be tested without a mouse or a rendered
        frame — this is the part that silently goes wrong when a layout constant drifts.
        """
        if self._col_width <= 0 or self._row_height <= 0:
            return None

        col = int(local_x / self._col_width)
        row = int(content_y / self._row_height)

        if col >= self._n_cols or col < 0 or row < 0:
            return None

        vis_pos = row * self._n_cols + col
        if vis_pos < 0 or vis_pos >= len(self._visible):
            return None

        # Check that the position is on the entry, not in the spacing around it.
        #
        # **The label counts as part of the entry.** Rejecting anything below the image made the filename a
        # dead strip 26 px tall, directly under the picture and visually part of it — so clicking a file
        # *by its name*, which is what a lifetime of file managers trains, did nothing at all. Nothing
        # errors and nothing moves, which is the worst shape a bug can take in a picker.
        #
        # The spacing between cells stays dead. That is a gap rather than a target, and a click landing
        # there is more plausibly a miss than a choice.
        tile_x = local_x - col * self._col_width
        tile_y = content_y - row * self._row_height
        text_h = self._font_size + 2 * self._frame_padding_y
        cell_height = self._tile_size + self._item_spacing_y + text_h
        if tile_x > self._tile_size or tile_y > cell_height:
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

            idx = self._hit_test()

            if self._debug:
                # Logged whatever the outcome, and with the outcome in it: a click that selects the wrong
                # tile and one that selects nothing have entirely different causes, and a line that only
                # appears on success cannot tell them apart.
                inside = guiutils.is_mouse_inside_widget(self._child_window_tag)
                local_x, local_y = guiutils.get_mouse_relative_pos(self._child_window_tag)
                content_y = local_y + dpg.get_y_scroll(self._child_window_tag)
                logger.info(f"ThumbnailGrid._on_click: inside={inside} local=({local_x:.0f},{local_y:.0f}) "
                            f"y_scroll={dpg.get_y_scroll(self._child_window_tag):.0f} "
                            f"content_y={content_y:.0f} row_h={self._row_height:.0f} "
                            f"col_w={self._col_width:.0f} "
                            f"row={int(content_y / self._row_height)} "
                            f"col={int(local_x / self._col_width)} "
                            f"-> idx={idx} (current={self._current})")
                # Where the panel is believed to be, against where the mouse actually was. An offset
                # between the two is invisible in the numbers above — every click simply misses — and it
                # is the failure this pair exists to make legible.
                logger.info(f"ThumbnailGrid._on_click: mouse={dpg.get_mouse_pos(local=False)} "
                            f"widget_pos={guiutils.get_widget_pos(self._child_window_tag)} "
                            f"widget_size={guiutils.get_widget_size(self._child_window_tag)}")

            if idx is None:
                return

            ctrl = self._allow_multi_select and (dpg.is_key_down(dpg.mvKey_LControl)
                                                 or dpg.is_key_down(dpg.mvKey_RControl))
            shift = self._allow_multi_select and (dpg.is_key_down(dpg.mvKey_LShift)
                                                  or dpg.is_key_down(dpg.mvKey_RShift))

            if shift and self._last_click_idx >= 0 and self._last_click_idx in self._visible:
                # Range select from last click to this click (in visible order).
                a = self._visible.index(self._last_click_idx)
                b = self._visible.index(idx) if idx in self._visible else a
                lo, hi = min(a, b), max(a, b)
                self._apply_selection({i for i in self._visible[lo:hi + 1] if self.is_selectable(i)})
            elif ctrl:
                self.toggle_select(idx)  # already notifies; a no-op on an unselectable entry
            else:
                # Bare click: set current, and replace the selection with this one entry — unless it is not
                # a candidate for selection, in which case the cursor moves and the selection is left as it
                # was. Clearing it would be the click doing something the user did not ask for.
                if self.is_selectable(idx):
                    self._apply_selection({idx})
                self.set_current(idx)

            self._last_click_idx = idx

    def _on_wheel(self, sender, app_data) -> None:
        """Flash the end when the wheel is turned against it.

        The wheel is a third way to move this view, and the only one nothing of ours sees: DPG scrolls the
        child window internally, so there is no animation to carry the flasher and no navigation to refuse.
        `note_wheel_scroll` handles the two-stage check that needs; see `ScrollEndFlasher`.
        """
        with self._lock:
            if not self.input_enabled or self._scroll_end_flasher is None:
                return
            if not guiutils.is_mouse_inside_widget(self._child_window_tag):  # tag
                return
            self._scroll_end_flasher.note_wheel_scroll()

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
            # Deferred — fires from update() outside the lock.
            self._pending_double_click = idx
