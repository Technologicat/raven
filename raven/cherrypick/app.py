"""raven-cherrypick — fast image triage tool.

Main application module: startup, GUI layout, render loop, hotkey dispatch.
"""

# WORKAROUND: Deleting a texture or image widget causes DPG to segfault on Nvidia/Linux.
# https://github.com/hoffstadt/DearPyGui/issues/554
# See raven/librarian/app.py:27-32 for the canonical example.
import platform
import os
if platform.system().upper() == "LINUX":
    os.environ["__GLVND_DISALLOW_PATCHING"] = "1"

import argparse
import logging
import math
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pathlib
import time
from typing import List

import numpy as np

from .. import __version__

logger.info(f"raven-cherrypick version {__version__} starting.")
logger.info("Loading libraries...")

import dearpygui.dearpygui as dpg
import torch
from ..common import deviceinfo
from ..common.gui import utils as guiutils
from ..common.gui import helpcard
from ..common.gui import animation as gui_animation
from ..vendor.file_dialog.fdialog import FileDialog
from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa
from ..vendor.tha3.util import torch_linear_to_srgb

from . import config
from .triage import TriageState, TriageManager
from .loader import ThumbnailPipeline
from .imageview import ImageView
from .grid import ThumbnailGrid, FilterMode
from .preload import PreloadCache
from ..common.image import utils as imageutils
from ..common.video import postprocessor

from unpythonic.env import env

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_app_state = {
    "image_view": None,
    "grid": None,
    "triage": None,
    "pipeline": None,
    "preload": None,
    "status_text": None,
    "themes_and_fonts": None,
}

_filedialog_open = None
_help_window = None
_debug = False
_preload_pending = False
_prev_current_idx = -1
_noise_pool_pending_size = None  # deferred noise pool regeneration (tile size)

# Validated at startup.
_device = None
_dtype = None
_preload_ram_budget_mb = config.PRELOAD_RAM_BUDGET_FALLBACK_MB


def _detect_preload_budget() -> int:
    """Auto-detect preload cache RAM budget.

    Returns 25% of available system RAM, clamped to [min, max] from config.
    Falls back to a fixed default if detection fails.
    """
    try:
        import psutil
        avail_mb = psutil.virtual_memory().available / (1024 * 1024)
        budget = int(avail_mb * config.PRELOAD_RAM_FRACTION)
        budget = max(config.PRELOAD_RAM_BUDGET_MIN_MB,
                     min(budget, config.PRELOAD_RAM_BUDGET_MAX_MB))
        logger.info("main: preload cache budget (system RAM) %d MB "
                     "(%.0f MB available, %.0f%% fraction)",
                     budget, avail_mb, config.PRELOAD_RAM_FRACTION * 100)
        return budget
    except Exception as exc:
        logger.warning("main: RAM detection failed (%s), "
                       "using fallback %d MB",
                       exc, config.PRELOAD_RAM_BUDGET_FALLBACK_MB)
        return config.PRELOAD_RAM_BUDGET_FALLBACK_MB


# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------

# Common aspect ratios (wider:narrower, as floats for matching).
_KNOWN_RATIOS = [
    (21, 9),
    (16, 9),
    (16, 10),
    (3, 2),
    (4, 3),
    (1, 1),
]
_RATIO_TOLERANCE = 0.02  # ±2%


def _approx_aspect_ratio(w: int, h: int) -> str:
    """Return a human-readable aspect ratio string for *w* × *h*.

    Matches against common ratios within tolerance.  Always returns the
    canonical form (wider:narrower), regardless of orientation.
    Falls back to GCD-reduced exact ratio for unusual sizes.
    """
    if w <= 0 or h <= 0:
        return ""
    long = max(w, h)
    short = min(w, h)
    actual = long / short
    for num, den in _KNOWN_RATIOS:
        if abs(actual - num / den) <= _RATIO_TOLERANCE:
            return f"{num}:{den}"
    # Fallback: GCD-reduced exact ratio.
    g = math.gcd(long, short)
    return f"{long // g}:{short // g}"


def _set_status(text: str) -> None:
    if _app_state["status_text"] is not None:
        dpg.set_value(_app_state["status_text"], text)


def _update_status() -> None:
    """Rebuild the status bar text from current state."""
    parts = []
    triage = _app_state["triage"]
    grid = _app_state["grid"]
    iv = _app_state["image_view"]

    if triage is not None and len(triage) > 0:
        # Current filename and position.
        idx = grid.current if grid is not None else -1
        if 0 <= idx < len(triage):
            name = triage[idx].filename
            if grid is not None and idx in grid.visible:
                pos = grid.visible.index(idx) + 1
                name += f"  [{pos} / {grid.visible_count}]"
            parts.append(name)

        # Image dimensions and aspect ratio.
        if iv is not None and iv.has_image:
            w, h = iv.image_size
            ratio = _approx_aspect_ratio(w, h)
            parts.append(f"{w}\u00d7{h} ({ratio})" if ratio else f"{w}\u00d7{h}")

            # Zoom level.
            parts.append(f"Zoom: {iv.zoom * 100:.0f}%")

        # Counts + filter indicator.
        n = len(triage)
        n_cherry = triage.count(TriageState.CHERRY)
        n_lemon = triage.count(TriageState.LEMON)
        if grid is not None and grid.filter_mode is not FilterMode.ALL:
            count_str = f"{grid.visible_count} / {n} images ({_FILTER_LABELS[grid.filter_mode]})"
        else:
            count_str = f"{n} images"
        if n_cherry:
            count_str += f", {n_cherry} cherries"
        if n_lemon:
            count_str += f", {n_lemon} lemons"
        parts.append(count_str)

        # Selection indicator.
        if grid is not None and grid.selected:
            parts.append(f"{len(grid.selected)} sel")

        # Focus indicator.
        if iv is not None and iv.focused:
            parts.append("IMAGE PANE FOCUSED")

    _set_status(" | ".join(parts) if parts else "Ready")


# ---------------------------------------------------------------------------
# Folder open
# ---------------------------------------------------------------------------

def _open_folder(folder_path: str) -> None:
    """Open a folder: scan images, start thumbnail pipeline, update GUI."""
    folder = pathlib.Path(folder_path).resolve()
    if not folder.is_dir():
        _set_status(f"Not a directory: {folder}")
        return

    logger.info("_open_folder: %s", folder)

    # Flush stale preload cache (indices from old folder must not be reused).
    preload = _app_state["preload"]
    if preload is not None:
        preload.clear()

    # Triage manager.
    triage = TriageManager(folder)
    _app_state["triage"] = triage

    if len(triage) == 0:
        _set_status(f"No images found in {folder}")
        return

    # Update grid.
    grid = _app_state["grid"]
    grid.set_entries([e.filename for e in triage.images],
                     [e.state for e in triage.images])

    # Start thumbnail pipeline.
    pipeline = _app_state["pipeline"]
    paths = [e.path for e in triage.images]
    pipeline.start(paths)

    # Load the first image in the main view.
    _load_current_image()

    # Update window title.
    dpg.set_viewport_title(f"raven-cherrypick {__version__} \u2014 {folder.name}")
    _update_status()


def _load_current_image() -> None:
    """Load the current grid image into the image viewer."""
    global _preload_pending
    grid = _app_state["grid"]
    triage = _app_state["triage"]
    iv = _app_state["image_view"]
    preload = _app_state["preload"]
    if grid is None or triage is None or iv is None:
        return
    idx = grid.current
    if idx < 0 or idx >= len(triage):
        iv.clear()
        return

    # Cancel preload tasks — free GPU for the current image.
    t_nav_start = time.perf_counter_ns()
    if preload is not None:
        preload.cancel_pending()
    t_cancel = time.perf_counter_ns()

    entry = triage[idx]
    try:
        old_size = iv.image_size

        # Try preload cache first.
        cached = preload.take(idx) if preload is not None else None
        t_take = time.perf_counter_ns()

        if cached is not None:
            t_cached_start = time.perf_counter_ns()

            # Donate the outgoing image's mip arrays to the preload cache
            # so navigating back is also instant. Don't donate if mips
            # are still loading — partial mip sets cause display bugs.
            if _prev_current_idx >= 0 and not iv.mip_loading:
                donated = iv.take_mip_arrays()
                if donated is not None and preload is not None:
                    arrays, dw, dh = donated
                    preload.donate(_prev_current_idx, arrays, dw, dh)

            t_donated = time.perf_counter_ns()

            new_size = (cached.img_w, cached.img_h)
            iv.set_preloaded_arrays(cached.mips,
                                    cached.img_w, cached.img_h)

            t_set_mips = time.perf_counter_ns()

            if _debug:
                logger.info(f"_load_current_image: {entry.filename} "
                            f"{cached.img_w}x{cached.img_h} "
                            f"PRELOADED cancel={(t_cancel - t_nav_start) / 1e6:.1f}ms "
                            f"take={(t_take - t_cancel) / 1e6:.1f}ms "
                            f"donate={(t_donated - t_cached_start) / 1e6:.1f}ms "
                            f"set_mips={(t_set_mips - t_donated) / 1e6:.1f}ms "
                            f"total={(t_set_mips - t_nav_start) / 1e6:.1f}ms "
                            f"cache_size={len(preload._cache)}")

            # Keep zoom/pan when switching between same-size images
            # (e.g. variants of the same shot). Otherwise zoom to fit.
            if new_size != old_size:
                iv.zoom_to_fit()

            # If the preloaded mips are capped (missing larger levels),
            # generate them in the background. Donated entries already
            # have the full chain, so check before submitting.
            has_fullres = any(s >= 1.0 for s, _w, _h, _f in cached.mips)
            if not has_fullres:
                iv.augment_mips(entry.path)
        else:
            # Cache miss — decode + mip generation on background thread.
            # zoom_to_fit handled inside load_from_file when dimensions
            # are known (after decode).
            iv.load_from_file(entry.path, old_size=old_size)

            if _debug:
                logger.info(f"_load_current_image: {entry.filename} MISS "
                            f"— queued background decode+mip "
                            f"(dimensions in _bg_file_mip_task log)")

    except Exception as exc:
        logger.warning("_load_current_image: failed to load %s: %s",
                       entry.filename, exc)
        iv.clear()

    _preload_pending = True
    _update_status()


# ---------------------------------------------------------------------------
# File dialog
# ---------------------------------------------------------------------------

def _open_file_dialog_callback(selected_files) -> None:
    """Callback for the folder dialog."""
    iv = _app_state["image_view"]
    grid = _app_state["grid"]
    if iv is not None:
        iv.input_enabled = True
    if grid is not None:
        grid.input_enabled = True
    if selected_files:
        _open_folder(str(selected_files[0]))


def _show_open_dialog(*_args) -> None:
    """Show the folder open dialog."""
    if _filedialog_open is not None:
        iv = _app_state["image_view"]
        grid = _app_state["grid"]
        if iv is not None:
            iv.input_enabled = False
        if grid is not None:
            grid.input_enabled = False
        _filedialog_open.show_file_dialog()


# ---------------------------------------------------------------------------
# Triage commands
# ---------------------------------------------------------------------------

def _mark_triage(state: TriageState, *, use_selection: bool = False) -> None:
    """Mark images with the given triage state.

    Without Ctrl (*use_selection=False*): operates on the current image only.
    With Ctrl (*use_selection=True*): operates on all selected images.
    """
    grid = _app_state["grid"]
    triage = _app_state["triage"]
    if grid is None or triage is None:
        return

    if use_selection:
        indices = list(grid.selected)
    else:
        indices = [grid.current] if grid.current >= 0 else []
    valid = [i for i in indices if 0 <= i < len(triage)]
    if len(valid) < len(indices):
        logger.warning("_mark_triage: %d out-of-range indices dropped (grid/triage desync?)",
                       len(indices) - len(valid))
    if not valid:
        return
    indices = valid

    errors = triage.set_state(indices, state)
    for err in errors:
        logger.warning("_mark_triage: %s", err)

    # Update grid tiles.
    for idx in indices:
        grid.update_triage_state(idx, triage[idx].state)

    _update_status()


# ---------------------------------------------------------------------------
# Navigation callbacks (for grid -> image view)
# ---------------------------------------------------------------------------

def _on_current_changed(idx: int) -> None:
    """Called by the grid when the current tile changes."""
    global _prev_current_idx
    _load_current_image()
    _prev_current_idx = idx


def _on_double_click(idx: int) -> None:
    """Called by the grid on double-click: load and zoom to fit."""
    _load_current_image()
    iv = _app_state["image_view"]
    if iv is not None:
        iv.zoom_to_fit()


# ---------------------------------------------------------------------------
# Zoom callbacks
# ---------------------------------------------------------------------------

def _on_zoom_changed(zoom: float) -> None:
    """Called by the image view when zoom changes."""
    _update_status()


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------

def _resize_gui(*_args) -> None:
    """Resize components to fill the main window."""
    w, h = guiutils.get_widget_size("cherrypick_main_window")
    iv = _app_state["image_view"]
    grid = _app_state["grid"]

    iv_w = int(w * config.IMAGE_PANE_RATIO) - config.IMAGE_VIEW_H_PADDING
    iv_h = h - config.IMAGE_VIEW_V_PADDING
    grid_w = w - int(w * config.IMAGE_PANE_RATIO) - config.IMAGE_VIEW_H_PADDING
    grid_h = iv_h

    if iv is not None and iv_w > 0 and iv_h > 0:
        iv.set_size(iv_w, iv_h)
    if grid is not None and grid_w > 0 and grid_h > 0:
        grid.set_size(grid_w, grid_h)


def _toggle_fullscreen(*_args) -> None:
    dpg.toggle_viewport_fullscreen()
    if guiutils.wait_for_resize("cherrypick_main_window"):
        _resize_gui()


def _set_zoom_fit_cap(value: bool) -> None:
    """Set the zoom-to-fit 100% cap and sync the checkbox."""
    iv = _app_state["image_view"]
    if iv is None:
        return
    iv.zoom_fit_cap = value
    dpg.set_value("cherrypick_zoom_fit_cap_cb", value)
    _update_status()


def _toggle_zoom_fit_cap() -> None:
    """Toggle the zoom-to-fit 100% cap."""
    iv = _app_state["image_view"]
    if iv is not None:
        _set_zoom_fit_cap(not iv.zoom_fit_cap)


# ---------------------------------------------------------------------------
# Hotkey handler
# ---------------------------------------------------------------------------

def _on_key(sender, app_data) -> None:
    """Global keyboard shortcut handler."""
    key = app_data
    iv = _app_state["image_view"]
    grid = _app_state["grid"]

    # Suppress input when modal dialogs are open.
    if iv is not None and not iv.input_enabled:
        return
    if _help_window is not None and _help_window.is_visible():
        return

    ctrl = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
    shift = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)

    # --- Ctrl+Shift: debug ---
    if ctrl and shift:
        if key == dpg.mvKey_M:
            dpg.show_metrics()
            # Toggle debug overlays.
            global _debug
            _debug = not _debug
            if iv is not None:
                iv._debug = _debug
                iv._needs_render = True
            if grid is not None:
                grid._debug = _debug
            logger.info(f"_on_key: debug mode {'ON' if _debug else 'OFF'}")
        elif key == dpg.mvKey_R:
            dpg.show_item_registry()
        elif key == dpg.mvKey_T:
            dpg.show_font_manager()
        elif key == dpg.mvKey_L:
            dpg.show_style_editor()
        return

    # --- Ctrl: app commands ---
    if ctrl:
        if key == dpg.mvKey_O:
            _show_open_dialog()
        # Ctrl+X/C/V: triage all selected images.
        elif key == dpg.mvKey_X:
            _mark_triage(TriageState.LEMON, use_selection=True)
        elif key == dpg.mvKey_C:
            _mark_triage(TriageState.CHERRY, use_selection=True)
        elif key == dpg.mvKey_V:
            _mark_triage(TriageState.NEUTRAL, use_selection=True)
        elif key == dpg.mvKey_A:
            if grid is not None:
                grid.select_all()
        elif key == dpg.mvKey_D:
            if grid is not None:
                grid.deselect_all()
        elif key == dpg.mvKey_I:
            if grid is not None:
                grid.invert_selection()
        elif key in (dpg.mvKey_1, dpg.mvKey_2, dpg.mvKey_3,
                     dpg.mvKey_4, dpg.mvKey_5):
            # Ctrl+1..5: tile size presets.
            idx = key - dpg.mvKey_1
            if idx < len(config.TILE_SIZES):
                _change_tile_size(config.TILE_SIZES[idx])
        return

    # --- Image pane focused: arrows pan ---
    if iv is not None and iv.focused:
        if key == dpg.mvKey_Escape:
            iv.focused = False
            _update_status()
            return
        if key == dpg.mvKey_Up:
            iv.pan_by(0, config.PAN_AMOUNT)
            return
        elif key == dpg.mvKey_Down:
            iv.pan_by(0, -config.PAN_AMOUNT)
            return
        elif key == dpg.mvKey_Left:
            iv.pan_by(config.PAN_AMOUNT, 0)
            return
        elif key == dpg.mvKey_Right:
            iv.pan_by(-config.PAN_AMOUNT, 0)
            return

    # --- Bare keys ---

    # Triage (current image only; Ctrl variants above operate on selection).
    if key == dpg.mvKey_X:
        _mark_triage(TriageState.LEMON)
    elif key == dpg.mvKey_C:
        _mark_triage(TriageState.CHERRY)
    elif key == dpg.mvKey_V:
        _mark_triage(TriageState.NEUTRAL)

    # Jump to next/prev by triage state (B/N/M mirror X/C/V, one column right).
    # Only in "All" view — when filtered, arrow keys already navigate that state only.
    elif key == dpg.mvKey_B:
        if grid is not None and grid.filter_mode is FilterMode.ALL:
            if shift:
                grid.navigate_prev_with_state(TriageState.LEMON)
            else:
                grid.navigate_next_with_state(TriageState.LEMON)
    elif key == dpg.mvKey_N:
        if grid is not None and grid.filter_mode is FilterMode.ALL:
            if shift:
                grid.navigate_prev_with_state(TriageState.CHERRY)
            else:
                grid.navigate_next_with_state(TriageState.CHERRY)
    elif key == dpg.mvKey_M:
        if grid is not None and grid.filter_mode is FilterMode.ALL:
            if shift:
                grid.navigate_prev_with_state(TriageState.NEUTRAL)
            else:
                grid.navigate_next_with_state(TriageState.NEUTRAL)

    # Filter.
    elif key == dpg.mvKey_G:
        if shift:
            _cycle_filter(-1)
        else:
            _cycle_filter(1)

    # Zoom.
    elif key in (dpg.mvKey_Plus, dpg.mvKey_Add):
        if iv is not None:
            iv.zoom_in()
    elif key in (dpg.mvKey_Minus, dpg.mvKey_Subtract):
        if iv is not None:
            iv.zoom_out()
    elif key == dpg.mvKey_F:
        if shift:
            _toggle_zoom_fit_cap()
        elif iv is not None:
            iv.zoom_to_fit()
    elif key == dpg.mvKey_1:
        if iv is not None:
            iv.zoom_to_actual()

    # Selection toggle for current image (keyboard equivalent of Ctrl+click).
    elif key == dpg.mvKey_Spacebar:
        if grid is not None and grid.current >= 0:
            grid.toggle_select(grid.current)

    # Focus toggle.
    elif key == dpg.mvKey_Tab:
        if iv is not None:
            iv.focused = not iv.focused
            _update_status()

    # Navigation.
    elif key == dpg.mvKey_Left:
        if grid is not None:
            grid.navigate_prev()
    elif key == dpg.mvKey_Right:
        if grid is not None:
            grid.navigate_next()
    elif key == dpg.mvKey_Up:
        if grid is not None:
            grid.navigate_row_up()
    elif key == dpg.mvKey_Down:
        if grid is not None:
            grid.navigate_row_down()
    elif key == dpg.mvKey_Home:
        if grid is not None:
            grid.navigate_first()
    elif key == dpg.mvKey_End:
        if grid is not None:
            grid.navigate_last()
    elif key in (dpg.mvKey_Prior, 517):  # Page Up (DPG 2.0 workaround)
        if grid is not None:
            grid.navigate_page_up()
    elif key in (dpg.mvKey_Next, 518):  # Page Down (DPG 2.0 workaround)
        if grid is not None:
            grid.navigate_page_down()

    # App.
    elif key == dpg.mvKey_F1:
        if _help_window is not None:
            _help_window.show()
    elif key == dpg.mvKey_F11:
        _toggle_fullscreen()


# ---------------------------------------------------------------------------
# Tile size change
# ---------------------------------------------------------------------------

def _on_tile_size_combo(sender, app_data) -> None:
    """Callback for the tile size combobox."""
    try:
        new_size = int(app_data)
    except (ValueError, TypeError):
        return
    _change_tile_size(new_size)


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

_FILTER_CYCLE = [FilterMode.ALL, FilterMode.CHERRY, FilterMode.LEMON, FilterMode.NEUTRAL]

_FILTER_LABELS = {
    FilterMode.ALL: "All",
    FilterMode.CHERRY: "Cherries",
    FilterMode.LEMON: "Lemons",
    FilterMode.NEUTRAL: "Neutral",
}


def _cycle_filter(direction: int = 1) -> None:
    """Cycle the grid filter forward (*direction=1*) or backward (*direction=-1*)."""
    grid = _app_state["grid"]
    if grid is None:
        return
    cur = grid.filter_mode
    idx = _FILTER_CYCLE.index(cur)
    new_idx = (idx + direction) % len(_FILTER_CYCLE)
    _set_filter(_FILTER_CYCLE[new_idx])


def _set_filter(mode: FilterMode) -> None:
    """Apply a filter mode, sync the combo, and update status."""
    grid = _app_state["grid"]
    if grid is None:
        return
    grid.set_filter(mode)
    dpg.set_value("cherrypick_filter_combo", _FILTER_LABELS[mode])
    # Reschedule preloading — visible list changed.
    global _preload_pending
    _preload_pending = True
    _update_status()


def _on_filter_combo(sender, app_data) -> None:
    """Callback for the filter combobox."""
    label_to_mode = {v: k for k, v in _FILTER_LABELS.items()}
    mode = label_to_mode.get(app_data)
    if mode is not None:
        _set_filter(mode)


# ---------------------------------------------------------------------------
# Tile size change
# ---------------------------------------------------------------------------

def _generate_noise_pool(tile_size: int) -> List[np.ndarray]:
    """Generate VHS noise placeholder tiles for the thumbnail grid."""
    n = config.PLACEHOLDER_POOL_SIZES[tile_size]  # KeyError → fail-fast for missing entry in config
    tensors = postprocessor.vhs_noise_pool(
        n,
        tile_size, tile_size,
        device=_device, dtype=_dtype,
        tint=config.PLACEHOLDER_TINT,
        brightness=config.PLACEHOLDER_BRIGHTNESS,
    )
    # Convert linear → sRGB for display (RGB channels only, not alpha).
    for t in tensors:
        t[:3] = torch_linear_to_srgb(t[:3])
    return [imageutils.tensor_to_dpg_flat(t.unsqueeze(0)) for t in tensors]


def _change_tile_size(new_size: int) -> None:
    """Change the grid tile size and restart the thumbnail pipeline."""
    grid = _app_state["grid"]
    triage = _app_state["triage"]
    pipeline = _app_state["pipeline"]
    if grid is None:
        return
    if grid._tile_size == new_size:
        return  # no-op

    grid.set_tile_size(new_size)
    # Defer noise pool generation to the main loop (outside render callback).
    global _noise_pool_pending_size
    _noise_pool_pending_size = new_size

    # Sync the combobox.
    dpg.set_value("cherrypick_tile_size_combo", str(new_size))

    # Restart pipeline at the new size.
    if pipeline is not None and triage is not None and len(triage) > 0:
        pipeline._tile_size = new_size
        paths = [e.path for e in triage.images]
        pipeline.start(paths)


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def _gui_cancel_tasks() -> None:
    """Exit callback: cancel background tasks without waiting.

    Called from inside ``render_dearpygui_frame``.  Must NOT wait — that
    would deadlock (the frame can't complete while we're waiting, and
    ``split_frame`` waiters need the frame to complete).

    The frame that triggers this callback unblocks ``split_frame`` waiters.
    Background threads then see ``cancelled`` and exit.  The actual wait
    and cleanup happen after the render loop exits (see end of ``main``).
    """
    preload = _app_state["preload"]
    if preload is not None:
        preload.cancel_pending()
    iv = _app_state["image_view"]
    if iv is not None:
        iv._mip_task_mgr.clear(wait=False)


def _gui_shutdown() -> None:
    """Full cleanup — call after the render loop has exited."""
    gui_animation.animator.clear()
    preload = _app_state["preload"]
    if preload is not None:
        preload.shutdown()
    pipeline = _app_state["pipeline"]
    if pipeline is not None:
        pipeline.shutdown()
    iv = _app_state["image_view"]
    if iv is not None:
        iv.destroy()
    grid = _app_state["grid"]
    if grid is not None:
        grid.destroy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Entry point for raven-cherrypick."""
    parser = argparse.ArgumentParser(
        description="raven-cherrypick \u2014 everyone's favorite mining tool",
        epilog="Fast image triage: sort a folder of images into "
               "cherries (keepers), lemons (rejects), and neutral.",
    )
    parser.add_argument("-v", "--version", action="version",
                        version=f"%(prog)s {__version__}")
    parser.add_argument("folder", nargs="?",
                        help="Folder of images to open")
    parser.add_argument("--width", type=int, default=config.DEFAULT_WIDTH,
                        help=f"Window width (default: {config.DEFAULT_WIDTH})")
    parser.add_argument("--height", type=int, default=config.DEFAULT_HEIGHT,
                        help=f"Window height (default: {config.DEFAULT_HEIGHT})")
    parser.add_argument("--tile-size", type=int, default=config.DEFAULT_TILE_SIZE,
                        help=f"Initial thumbnail size (default: {config.DEFAULT_TILE_SIZE})")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device override (e.g. cuda:1, cpu)")
    parser.add_argument("--debug", action="store_true",
                        help="Show debug overlays (pan/zoom coordinates, click positions)")
    args = parser.parse_args()

    global _debug
    _debug = args.debug

    # --- GPU config ---
    if args.device is not None:
        config.gpu_config["thumbnails"]["device_string"] = args.device
    deviceinfo.validate(config.gpu_config)
    global _device, _dtype
    _device = torch.device(config.gpu_config["thumbnails"]["device_string"])
    _dtype = config.gpu_config["thumbnails"]["dtype"]

    # --- Preload RAM budget ---
    global _preload_ram_budget_mb
    _preload_ram_budget_mb = _detect_preload_budget()

    # --- DPG bootup ---
    dpg.create_context()
    themes_and_fonts = guiutils.bootup(font_size=config.FONT_SIZE)
    _app_state["themes_and_fonts"] = themes_and_fonts

    if platform.system().upper() == "WINDOWS":
        icon_ext = "ico"
    else:
        icon_ext = "png"
    icons_dir = pathlib.Path(os.path.dirname(__file__), "..", "icons")
    dpg.create_viewport(title=f"raven-cherrypick {__version__}",
                        small_icon=str((icons_dir / f"app_128_notext.{icon_ext}").resolve()),
                        large_icon=str((icons_dir / f"app_256.{icon_ext}").resolve()),
                        width=args.width, height=args.height)
    dpg.setup_dearpygui()

    # --- File dialog ---
    global _filedialog_open
    _filedialog_open = FileDialog(
        title="Open image folder",
        tag="cherrypick_open_dialog",
        callback=_open_file_dialog_callback,
        modal=True,
        dirs_only=True,
        multi_selection=False,
        allow_drag=False,
        default_path=os.getcwd(),
    )

    # --- Thumbnail pipeline ---
    pipeline = ThumbnailPipeline(device=_device, dtype=_dtype,
                                 tile_size=args.tile_size,
                                 lanczos_order=config.THUMBNAIL_LANCZOS_ORDER)
    _app_state["pipeline"] = pipeline

    # --- Preload cache ---
    _app_state["preload"] = PreloadCache(
        device=_device,
        lanczos_order=config.THUMBNAIL_LANCZOS_ORDER,
        mip_min_size=config.MIP_MIN_SIZE,
        ram_budget_mb=_preload_ram_budget_mb,
        debug=_debug,
    )

    # --- Build GUI ---
    with dpg.window(tag="cherrypick_main_window"):
        # Toolbar.
        with dpg.group(horizontal=True):
            # Open folder.
            btn = dpg.add_button(label=fa.ICON_FOLDER_OPEN,
                                 tag="cherrypick_open_btn",
                                 callback=_show_open_dialog, width=30)
            dpg.bind_item_font(btn, themes_and_fonts.icon_font_solid)
            with dpg.tooltip(btn):
                dpg.add_text("Open folder [Ctrl+O]")

            # Zoom buttons.
            btn = dpg.add_button(label=fa.ICON_MAGNIFYING_GLASS_PLUS,
                                 tag="cherrypick_zoom_in_btn",
                                 callback=lambda: _app_state["image_view"].zoom_in()
                                 if _app_state["image_view"] else None,
                                 width=30)
            dpg.bind_item_font(btn, themes_and_fonts.icon_font_solid)
            with dpg.tooltip(btn):
                dpg.add_text("Zoom in [+]")

            btn = dpg.add_button(label=fa.ICON_MAGNIFYING_GLASS_MINUS,
                                 tag="cherrypick_zoom_out_btn",
                                 callback=lambda: _app_state["image_view"].zoom_out()
                                 if _app_state["image_view"] else None,
                                 width=30)
            dpg.bind_item_font(btn, themes_and_fonts.icon_font_solid)
            with dpg.tooltip(btn):
                dpg.add_text("Zoom out [-]")

            btn = dpg.add_button(label=fa.ICON_SQUARE,
                                 tag="cherrypick_zoom_fit_btn",
                                 callback=lambda: _app_state["image_view"].zoom_to_fit()
                                 if _app_state["image_view"] else None,
                                 width=30)
            dpg.bind_item_font(btn, themes_and_fonts.icon_font_regular)
            with dpg.tooltip(btn):
                dpg.add_text("Zoom to fit [F]")

            dpg.add_checkbox(label="Cap fit",
                             tag="cherrypick_zoom_fit_cap_cb",
                             default_value=config.ZOOM_FIT_CAP,
                             callback=lambda s, v: _set_zoom_fit_cap(v))
            with dpg.tooltip("cherrypick_zoom_fit_cap_cb"):
                dpg.add_text("Cap zoom-to-fit at 100% [Shift+F]\nPrevents upscaling small images")

            dpg.add_spacer(width=8)

            # Triage buttons. Click = current image; Ctrl+click = all selected.
            def _triage_btn_callback(sender, app_data, user_data):
                ctrl_held = (dpg.is_key_down(dpg.mvKey_LControl)
                             or dpg.is_key_down(dpg.mvKey_RControl))
                _mark_triage(user_data, use_selection=ctrl_held)

            btn = dpg.add_button(label=fa.ICON_STAR,
                                 tag="cherrypick_mark_cherry_btn",
                                 callback=_triage_btn_callback,
                                 user_data=TriageState.CHERRY,
                                 width=30)
            dpg.bind_item_font(btn, themes_and_fonts.icon_font_solid)
            with dpg.tooltip(btn):
                dpg.add_text("Mark cherry [C]\n    with Ctrl: all selected")

            btn = dpg.add_button(label=fa.ICON_LEMON,
                                 tag="cherrypick_mark_lemon_btn",
                                 callback=_triage_btn_callback,
                                 user_data=TriageState.LEMON,
                                 width=30)
            dpg.bind_item_font(btn, themes_and_fonts.icon_font_solid)
            with dpg.tooltip(btn):
                dpg.add_text("Mark lemon [X]\n    with Ctrl: all selected")

            btn = dpg.add_button(label=fa.ICON_XMARK,
                                 tag="cherrypick_clear_mark_btn",
                                 callback=_triage_btn_callback,
                                 user_data=TriageState.NEUTRAL,
                                 width=30)
            dpg.bind_item_font(btn, themes_and_fonts.icon_font_solid)
            with dpg.tooltip(btn):
                dpg.add_text("Clear mark (Neutral) [V]\n    with Ctrl: all selected")

            dpg.add_spacer(width=8)

            # Filter selector.
            filter_labels = [_FILTER_LABELS[m] for m in _FILTER_CYCLE]
            dpg.add_combo(items=filter_labels,
                          default_value=_FILTER_LABELS[FilterMode.ALL],
                          tag="cherrypick_filter_combo",
                          callback=_on_filter_combo,
                          width=90)
            with dpg.tooltip("cherrypick_filter_combo"):
                dpg.add_text("Filter view [G / Shift+G]")

            dpg.add_spacer(width=8)

            # Tile size selector.
            tile_size_labels = [str(s) for s in config.TILE_SIZES]
            dpg.add_combo(items=tile_size_labels,
                          default_value=str(args.tile_size),
                          tag="cherrypick_tile_size_combo",
                          callback=_on_tile_size_combo,
                          width=60)
            with dpg.tooltip("cherrypick_tile_size_combo"):
                dpg.add_text("Tile size [Ctrl+1..5]")

            dpg.add_spacer(width=8)

            # Fullscreen / help.
            btn = dpg.add_button(label=fa.ICON_EXPAND,
                                 tag="cherrypick_fullscreen_btn",
                                 callback=_toggle_fullscreen, width=30)
            dpg.bind_item_font(btn, themes_and_fonts.icon_font_solid)
            with dpg.tooltip(btn):
                dpg.add_text("Fullscreen [F11]")

            btn = dpg.add_button(label=fa.ICON_CIRCLE_QUESTION,
                                 tag="cherrypick_help_btn", width=30)
            dpg.bind_item_font(btn, themes_and_fonts.icon_font_regular)
            with dpg.tooltip(btn):
                dpg.add_text("Help [F1]")

        # Two-column layout.
        with dpg.group(horizontal=True, tag="cherrypick_columns"):
            iv_w = int(args.width * config.IMAGE_PANE_RATIO) - config.IMAGE_VIEW_H_PADDING
            iv_h = args.height - config.IMAGE_VIEW_V_PADDING
            grid_w = args.width - int(args.width * config.IMAGE_PANE_RATIO) - config.IMAGE_VIEW_H_PADDING
            grid_h = iv_h

            # Main image view.
            # NOTE: use explicit parent tag, not dpg.last_container(),
            # because component __init__ methods create handler registries
            # that pollute the DPG parent stack.
            image_view = ImageView(
                parent="cherrypick_columns",
                width=max(1, iv_w), height=max(1, iv_h),
                device=_device,
                lanczos_order=config.THUMBNAIL_LANCZOS_ORDER,
                mip_min_size=config.MIP_MIN_SIZE,
                debug=_debug,
                on_zoom_changed=_on_zoom_changed,
            )
            _app_state["image_view"] = image_view

            # Thumbnail grid.
            grid = ThumbnailGrid(
                parent="cherrypick_columns",
                width=max(1, grid_w), height=max(1, grid_h),
                tile_size=args.tile_size,
                icon_font=themes_and_fonts.icon_font_solid,
                on_current_changed=_on_current_changed,
                on_selection_changed=_update_status,
                on_double_click=_on_double_click,
                debug=_debug,
            )
            _app_state["grid"] = grid
            grid.set_noise_pool(_generate_noise_pool(args.tile_size))

        # Status bar.
        _app_state["status_text"] = dpg.add_text("Ready")

    # --- Hotkey handler ---
    with dpg.handler_registry():
        dpg.add_key_press_handler(callback=_on_key)

    # --- Help card ---
    hotkey_info = (
        env(key_indent=0, key="Ctrl+O", action_indent=0, action="Open folder", notes=""),
        env(key_indent=0, key="C", action_indent=0, action="Mark cherry", notes=""),
        env(key_indent=1, key="Ctrl+C", action_indent=1, action="Same, but all selected", notes=""),
        env(key_indent=0, key="X", action_indent=0, action="Mark lemon", notes=""),
        env(key_indent=1, key="Ctrl+X", action_indent=1, action="Same, but all selected", notes=""),
        env(key_indent=0, key="V", action_indent=0, action="Clear mark", notes=""),
        env(key_indent=1, key="Ctrl+V", action_indent=1, action="Same, but all selected", notes=""),
        helpcard.hotkey_blank_entry,
        env(key_indent=0, key="B / Shift+B", action_indent=0, action="Next / prev lemon", notes="All view only; wraps"),
        env(key_indent=0, key="N / Shift+N", action_indent=0, action="Next / prev cherry", notes="All view only; wraps"),
        env(key_indent=0, key="M / Shift+M", action_indent=0, action="Next / prev neutral", notes="All view only; wraps"),
        helpcard.hotkey_blank_entry,
        env(key_indent=0, key="G", action_indent=0, action="Cycle filter forward", notes="All/Cherries/Lemons/Neutral"),
        env(key_indent=0, key="Shift+G", action_indent=0, action="Cycle filter backward", notes=""),
        helpcard.hotkey_blank_entry,
        env(key_indent=0, key="Left / Right", action_indent=0, action="Prev / next image", notes="Navigate only"),
        env(key_indent=0, key="Up / Down", action_indent=0, action="Prev / next row", notes="Navigate only"),
        env(key_indent=0, key="Home / End", action_indent=0, action="First / last image", notes=""),
        env(key_indent=0, key="Page Up / Down", action_indent=0, action="Scroll by page", notes=""),
        helpcard.hotkey_blank_entry,
        env(key_indent=0, key="Click", action_indent=0, action="Navigate and select", notes=""),
        env(key_indent=0, key="Ctrl+Click", action_indent=0, action="Toggle in selection", notes=""),
        env(key_indent=0, key="Shift+Click", action_indent=0, action="Range select", notes=""),
        env(key_indent=0, key="Space", action_indent=0, action="Toggle select current", notes=""),
        env(key_indent=0, key="Ctrl+A", action_indent=0, action="Select all", notes=""),
        env(key_indent=0, key="Ctrl+D", action_indent=0, action="Deselect all", notes=""),
        env(key_indent=0, key="Ctrl+I", action_indent=0, action="Invert selection", notes=""),

        helpcard.hotkey_new_column,

        env(key_indent=0, key="+  / Numpad +", action_indent=0, action="Zoom in", notes=""),
        env(key_indent=0, key="-  / Numpad -", action_indent=0, action="Zoom out", notes=""),
        env(key_indent=0, key="F", action_indent=0, action="Zoom to fit", notes=""),
        env(key_indent=0, key="Shift+F", action_indent=0, action="Toggle fit cap at 100%", notes="No upscale"),
        env(key_indent=0, key="1", action_indent=0, action="Zoom to 1:1", notes=""),
        env(key_indent=0, key="Mouse wheel", action_indent=0, action="Zoom at cursor", notes=""),
        env(key_indent=0, key="Mouse drag", action_indent=0, action="Pan image", notes=""),
        helpcard.hotkey_blank_entry,
        env(key_indent=0, key="Tab", action_indent=0, action="Toggle image pane focus", notes=""),
        env(key_indent=1, key="Left / Right / Up / Down", action_indent=0, action="Pan (when focused)", notes=""),
        env(key_indent=1, key="Esc", action_indent=0, action="Unfocus", notes=""),
        helpcard.hotkey_blank_entry,
        env(key_indent=0, key="Ctrl+1..5", action_indent=0, action="Tile size preset", notes=""),
        env(key_indent=0, key="F1", action_indent=0, action="This help card", notes=""),
        env(key_indent=0, key="F11", action_indent=0, action="Fullscreen", notes=""),
    )

    def _help_on_show():
        if _app_state["image_view"] is not None:
            _app_state["image_view"].input_enabled = False
        if _app_state["grid"] is not None:
            _app_state["grid"].input_enabled = False

    def _help_on_hide():
        if _app_state["image_view"] is not None:
            _app_state["image_view"].input_enabled = True
        if _app_state["grid"] is not None:
            _app_state["grid"].input_enabled = True

    global _help_window
    _help_window = helpcard.HelpWindow(
        hotkey_info=hotkey_info,
        width=config.HELP_WINDOW_W,
        height=config.HELP_WINDOW_H,
        reference_window="cherrypick_main_window",
        themes_and_fonts=themes_and_fonts,
        on_show=_help_on_show,
        on_hide=_help_on_hide,
    )
    dpg.set_item_callback("cherrypick_help_btn", _help_window.show)

    # --- Start app ---
    dpg.set_primary_window("cherrypick_main_window", True)
    dpg.set_viewport_resize_callback(_resize_gui)
    dpg.set_exit_callback(_gui_cancel_tasks)
    dpg.set_viewport_vsync(True)
    dpg.show_viewport()

    # Open folder from CLI argument.
    if args.folder:
        _open_folder(args.folder)

    # Initial resize settling.
    def _initial_resize(*_args):
        _resize_gui()
        iv = _app_state["image_view"]
        if iv is not None and iv.has_image:
            iv.zoom_to_fit()
    dpg.set_frame_callback(10, _initial_resize)

    # --- Render loop ---
    try:
        while dpg.is_dearpygui_running():
            # Poll thumbnail pipeline.
            pipeline = _app_state["pipeline"]
            grid = _app_state["grid"]
            if pipeline is not None and grid is not None:
                for idx, flat_rgba in pipeline.poll():
                    grid.set_thumbnail(idx, flat_rgba)

            # Deferred noise pool generation (set by _change_tile_size callback).
            global _noise_pool_pending_size
            if _noise_pool_pending_size is not None and grid is not None:
                grid.set_noise_pool(_generate_noise_pool(_noise_pool_pending_size))
                _noise_pool_pending_size = None

            # Update components.  Grid first: its deferred callbacks
            # (on_current_changed, on_double_click) trigger image loading
            # which may set iv._needs_render.  Then iv.update() processes
            # the render in the same frame — no one-frame lag.
            if grid is not None:
                grid.update()
            iv = _app_state["image_view"]
            if iv is not None:
                iv.update()

            # Trigger preloading once the current image finishes loading.
            global _preload_pending
            preload = _app_state["preload"]
            triage = _app_state["triage"]
            if (_preload_pending and preload is not None
                    and iv is not None and not iv.mip_loading
                    and grid is not None and triage is not None):
                _preload_pending = False
                preload.schedule(grid.current, grid.visible,
                                 grid.n_cols, triage)

            gui_animation.animator.render_frame()
            dpg.render_dearpygui_frame()
    except KeyboardInterrupt:
        pass

    # The exit callback cancelled tasks without waiting.  The last
    # render_dearpygui_frame() unblocked any split_frame() waiters.
    # Now it's safe to wait for threads and clean up.
    _gui_shutdown()
    dpg.destroy_context()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
