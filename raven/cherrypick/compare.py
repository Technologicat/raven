"""Compare mode for raven-cherrypick.

Cycles through a set of multi-selected images in the main view,
enabling rapid A/B/C comparison of variants. Up to 9 images,
numbered 1–9, at configurable FPS.

Compare mode is an overlay on normal operation — the underlying
state (triage, selection, etc.) is unchanged.
"""

__all__ = ["CompareMode"]

import logging
import time
from collections.abc import Sequence
from typing import Callable

from ..common.gui import animation as gui_animation
from . import config

logger = logging.getLogger(__name__)


class CompareMode:
    """Compare mode state machine.

    Create once at startup. The render loop calls `tick()` every frame.
    Hotkey dispatch calls `enter`, `exit`, `select_frame`, etc.

    Does not own DPG items directly — delegates to ImageView (number
    overlay) and ThumbnailGrid (badges, active highlight).
    """

    def __init__(self, *,
                 get_image_view,
                 get_grid,
                 get_preload,
                 get_triage,
                 load_image_fn: Callable[[int], None],
                 set_status_fn: Callable[[str], None],
                 update_status_fn: Callable[[], None]):
        """
        All ``get_*`` parameters are zero-argument callables returning
        the current component (or ``None``). This avoids holding stale
        references when a new folder is opened.

        *load_image_fn*: ``f(idx)`` — set grid.current and load image.
        *set_status_fn*: ``f(text)`` — set status bar text directly.
        *update_status_fn*: ``f()`` — rebuild normal status bar.
        """
        self._get_iv = get_image_view
        self._get_grid = get_grid
        self._get_preload = get_preload
        self._get_triage = get_triage
        self._load_image = load_image_fn
        self._set_status = set_status_fn
        self._update_status = update_status_fn

        # State.
        self.active: bool = False
        self.warming_up: bool = False
        self.paused: bool = False
        self.fps: float = config.COMPARE_DEFAULT_FPS
        self.frame_list: list[int] = []  # image indices, grid order
        self.frame_idx: int = 0
        self.saved_current: int = -1
        self._frame_start_ns: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enter(self, indices: Sequence[int]) -> bool:
        """Enter compare mode with the given image indices.

        *indices*: image indices to compare, capped to first 9 in
        grid-visible order.

        Returns ``True`` if compare mode was activated, ``False`` if
        there were fewer than 2 valid indices.
        """
        grid = self._get_grid()
        preload = self._get_preload()
        triage = self._get_triage()
        if grid is None or preload is None or triage is None:
            return False

        # Cap to first COMPARE_MAX_IMAGES in grid-visible order.
        indices_set = set(indices)
        ordered = [idx for idx in grid.visible if idx in indices_set]
        ordered = ordered[:config.COMPARE_MAX_IMAGES]
        if len(ordered) < 2:
            return False

        self.frame_list = ordered
        self.frame_idx = 0
        self.saved_current = grid.current
        self.fps = config.COMPARE_DEFAULT_FPS
        self.paused = False
        self.active = True
        self.warming_up = True

        # Pre-cache all compare images.
        preload.schedule_compare(ordered, triage)

        logger.info("CompareMode.enter: %d images, indices=%s",
                    len(ordered), ordered)
        return True

    def exit(self, restore: bool = True, redraw: bool = True) -> None:
        """Exit compare mode.

        *restore*: if ``True``, navigate back to the image that was
        current before entering compare mode.
        *redraw*: if ``False``, skip DPG draw operations (for shutdown,
        when the DPG item tree may already be torn down).

        Always unpins preload entries (finally-style cleanup).
        """
        if not self.active:
            return

        logger.info("CompareMode.exit: restore=%s", restore)

        self.active = False
        self.warming_up = False
        self.paused = False

        # Clean up overlays.
        iv = self._get_iv()
        grid = self._get_grid()
        preload = self._get_preload()

        if iv is not None:
            iv.clear_overlay()
        if grid is not None:
            if redraw:
                grid.clear_compare_badges()
                grid.clear_compare_active()
            else:
                # Shutdown path: clear state without redrawing
                # (DPG items may already be torn down).
                grid._compare_badges.clear()
                grid._compare_active_idx = -1
                grid._compare_active_alpha = 0.0

        # Always unpin — finally-style.
        if preload is not None:
            preload.unpin_all()

        # Restore or navigate.
        if restore and self.saved_current >= 0:
            self._load_image(self.saved_current)

        self._update_status()

    def select_frame(self, n: int) -> None:
        """Digit key handler: exit compare mode and navigate to frame *n*.

        *n*: 1-based frame number (key pressed). If out of range for
        the current frame list, does nothing.
        """
        idx_in_list = n - 1
        if idx_in_list < 0 or idx_in_list >= len(self.frame_list):
            return
        target = self.frame_list[idx_in_list]
        self.exit(restore=False)
        self._load_image(target)

    def tick(self) -> None:
        """Called every frame from the render loop."""
        if not self.active:
            return

        if self.warming_up:
            self._tick_warmup()
        else:
            self._tick_cycling()

    def toggle_pause(self) -> None:
        """Pause or resume cycling."""
        if not self.active or self.warming_up:
            return
        self.paused = not self.paused
        if not self.paused:
            # Reset timer to prevent immediate advance.
            self._frame_start_ns = time.monotonic_ns()
        self._update_compare_status()

    def adjust_fps(self, delta: float) -> None:
        """Adjust cycling FPS by *delta*. Clamps to config range."""
        self.fps = max(config.COMPARE_MIN_FPS,
                       min(config.COMPARE_MAX_FPS,
                           self.fps + delta))
        if self.active and not self.warming_up:
            self._update_compare_status()

    def reset_fps(self) -> None:
        """Reset FPS to default."""
        self.fps = config.COMPARE_DEFAULT_FPS
        if self.active and not self.warming_up:
            self._update_compare_status()

    def fade_alpha(self) -> float:
        """Current fade-out alpha [0, 1] for the active tile highlight.

        Uses `pulsation_envelope` — the cosine-squared 1→0→1 curve.

        When cycling: only the first half (1→0) plays per frame.
        When paused: full 1→0→1 cycle at COMPARE_PAUSED_PULSE_DURATION.
        """
        now = time.monotonic_ns()
        elapsed_s = (now - self._frame_start_ns) / 10**9

        if self.paused:
            cycle_dur = config.COMPARE_PAUSED_PULSE_DURATION
            t = (elapsed_s % cycle_dur) / cycle_dur
        else:
            # Half-cycle: 1→0 fade-out over one frame duration.
            frame_dur = 1.0 / self.fps
            half_cycle = frame_dur
            t = min(elapsed_s / half_cycle, 1.0) * 0.5  # map to [0, 0.5]
        return gui_animation.pulsation_envelope(t)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _tick_warmup(self) -> None:
        """Poll preload cache during warm-up phase."""
        preload = self._get_preload()
        if preload is None:
            self.exit(restore=True)
            return

        n_cached, n_total = preload.compare_progress(self.frame_list)
        self._set_status(
            f"Compare mode initializing [{n_cached}/{n_total}]...")

        if n_cached >= n_total:
            self._start_cycling()

    def _start_cycling(self) -> None:
        """Transition from warm-up to active cycling."""
        self.warming_up = False
        self.frame_idx = 0
        self._frame_start_ns = time.monotonic_ns()

        # Set up grid badges: {image_idx: badge_number}.
        grid = self._get_grid()
        badges = {idx: n + 1 for n, idx in enumerate(self.frame_list)}
        if grid is not None:
            grid.set_compare_badges(badges)

        # Show first frame.
        self._show_frame()

        logger.info("CompareMode._start_cycling: ready, %d frames at %.1f FPS",
                    len(self.frame_list), self.fps)

    def _tick_cycling(self) -> None:
        """Advance frame if enough time has elapsed."""
        if not self.paused:
            now = time.monotonic_ns()
            frame_ns = int(10**9 / self.fps)
            if now - self._frame_start_ns >= frame_ns:
                self.frame_idx = (self.frame_idx + 1) % len(self.frame_list)
                self._frame_start_ns = now
                self._show_frame()

        # Update grid active highlight (every frame for smooth fade).
        grid = self._get_grid()
        if grid is not None:
            current_img_idx = self.frame_list[self.frame_idx]
            grid.set_compare_active(current_img_idx, self.fade_alpha())

    def _show_frame(self) -> None:
        """Display the current compare frame in the image viewer."""
        img_idx = self.frame_list[self.frame_idx]
        iv = self._get_iv()
        preload = self._get_preload()
        if iv is None or preload is None:
            return

        # Load from preload cache.
        cached = preload.take(img_idx)
        if cached is not None:
            iv.set_preloaded_arrays(cached.mips, cached.img_w, cached.img_h)
            # Re-donate immediately so subsequent cycles still hit the cache.
            preload.donate(img_idx, cached.mips, cached.img_w, cached.img_h)
        else:
            # Shouldn't happen after warm-up, but handle gracefully.
            logger.warning("CompareMode._show_frame: cache miss for idx=%d",
                           img_idx)

        # Re-apply zoom-to-fit when switching between different-sized images.
        if iv._zoom_is_fit:
            iv.zoom_to_fit()

        # Update overlay number.
        frame_num = self.frame_idx + 1
        iv.set_overlay_number(frame_num)

        self._update_compare_status()

    def _update_compare_status(self) -> None:
        """Update status bar with compare mode information."""
        frame_num = self.frame_idx + 1
        n_total = len(self.frame_list)
        if self.paused:
            self._set_status(f"Compare [{frame_num}/{n_total}] | PAUSED")
        else:
            self._set_status(
                f"Compare [{frame_num}/{n_total}] | {self.fps:.1f} FPS")
