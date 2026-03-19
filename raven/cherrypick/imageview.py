"""Main image viewer pane for raven-cherrypick.

Displays a single image on a DPG drawlist with pan/zoom interaction and
Lanczos mipmap selection for quality display at intermediate zoom-out levels.

Coordinate system:
  - **Image coordinates**: (0, 0) at top-left of image, in pixels.
  - **Screen coordinates**: (0, 0) at top-left of the drawlist widget.
  - Pan state ``(pan_cx, pan_cy)`` is the image coordinate at the center
    of the view.  Zoom is screen pixels per image pixel.

See `raven.common.gui.utils.content_to_screen` for the transform formula.

Mip textures are loaded in a background thread (smallest levels first).
The render loop shows the best available mip at each frame, so the image
appears quickly at reduced quality and sharpens as larger mips complete.
"""

__all__ = ["ImageView"]

import concurrent.futures
import logging
import pathlib
import threading
import time
from typing import Callable, Optional, Union

import dearpygui.dearpygui as dpg
import numpy as np
import torch

from unpythonic.env import env

from ..common.bgtask import TaskManager
from ..common.gui import utils as guiutils
from ..common.image import lanczos
from ..common.image import utils as imageutils
from . import config

logger = logging.getLogger(__name__)

# Counter for unique DPG tags (thread-safe).
_tag_counter = 0
_tag_lock = threading.Lock()


def _next_tag(prefix: str) -> str:
    global _tag_counter
    with _tag_lock:
        _tag_counter += 1
        return f"imageview_{prefix}_{_tag_counter}"


class ImageView:
    """Zoomable, pannable image viewer on a DPG drawlist.

    Create once, then call `set_image` to display an image.
    The parent DPG render loop must call `update` every frame.
    """

    def __init__(self, parent: str | int,
                 width: int, height: int,
                 device: torch.device,
                 lanczos_order: int = lanczos.DEFAULT_ORDER,
                 mip_min_size: int = config.MIP_MIN_SIZE,
                 on_zoom_changed: Optional[Callable[[float], None]] = None,
                 debug: bool = False):
        """
        *parent*: DPG parent container (tag or ID).
        *width*, *height*: initial drawlist size in pixels.
        *device*: torch device for mip generation.
        *on_zoom_changed*: optional callback ``f(zoom)`` fired when zoom changes.
        """
        self._device = device
        self._order = lanczos_order
        self._mip_min_size = mip_min_size
        self._on_zoom_changed = on_zoom_changed
        self._debug = debug

        self._view_w = width
        self._view_h = height

        # Pan/zoom state.
        self._zoom = 1.0
        self._pan_cx = 0.0  # image x at view center
        self._pan_cy = 0.0  # image y at view center
        self._zoom_is_fit = False  # True when last zoom was zoom_to_fit

        # Image state.
        self._img_w = 0
        self._img_h = 0
        self._mips: list[tuple[float, int | str]] = []  # (scale, dpg_texture_tag)
        self._mip_arrays: list[tuple[float, int, int, np.ndarray]] = []  # (scale, w, h, flat)
        self._old_mips: list[tuple[float, int | str]] = []  # kept until first new mip ready
        self._old_img_w = 0  # dimensions matching _old_mips
        self._old_img_h = 0
        self._old_zoom = 1.0  # zoom/pan matching _old_mips
        self._old_pan_cx = 0.0
        self._old_pan_cy = 0.0
        self._mip_loading = False  # True while background mip task is running
        self._mips_lock = threading.Lock()

        # Texture pool: reuse textures across same-size images via set_value
        # instead of creating/deleting (avoids DPG OpenGL glitches).
        self._tex_pool: dict[tuple[int, int], list[int | str]] = {}
        self._tex_sizes: dict[int | str, tuple[int, int]] = {}  # tag → (w, h)

        # Background mip loading.
        self._mip_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="cherrypick_mip")
        self._mip_task_mgr = TaskManager("mip_loader", "sequential",
                                         self._mip_executor)
        # Separate task group for augmenting a preloaded image with its
        # full-res mip. Shares the executor but has independent cancellation:
        # navigating away cancels the augment without disturbing set_image /
        # load_from_file, and vice versa.
        self._augment_task_mgr = TaskManager("mip_augment", "sequential",
                                             self._mip_executor)

        # Interaction state.
        self.input_enabled: bool = True
        self._focused = False
        self._dragging = False
        self._drag_cumulative = (0.0, 0.0)
        self._drag_pan_start = (0.0, 0.0)
        self._needs_render = True

        # Tracked draw items — _render creates new items first, then deletes
        # the old batch.  The drawlist is never empty, eliminating blank flashes.
        self._drawn_items: list[int | str] = []

        # Create DPG drawlist.
        self._drawlist_tag = _next_tag("drawlist")
        dpg.add_drawlist(width=width, height=height,
                         parent=parent, tag=self._drawlist_tag)

        # Register mouse handlers.
        self._handler_tag = _next_tag("handlers")
        with dpg.handler_registry(tag=self._handler_tag):
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left,
                                       callback=self._on_mouse_drag)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left,
                                          callback=self._on_mouse_release)
            dpg.add_mouse_wheel_handler(callback=self._on_mouse_wheel)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_image(self, rgba: np.ndarray) -> None:
        """Load a new image from an ``(H, W, 4)`` RGBA uint8 numpy array.

        Returns immediately. Mip chain generation and DPG texture creation
        happen in a background thread. The render loop shows the best
        available mip at each frame.
        """
        # Cancel any in-progress mip loading (both full pipeline and augment).
        self._augment_task_mgr.clear()
        self._mip_task_mgr.clear(wait=True)

        self._bridge_old_mips()

        self._img_h, self._img_w = rgba.shape[:2]
        self._mip_loading = True
        self._needs_render = True

        # Start background mip generation.
        task_env = env(rgba=rgba,
                       device=self._device,
                       order=self._order,
                       mip_min_size=self._mip_min_size,
                       debug=self._debug)
        self._mip_task_mgr.submit(self._bg_mip_task, task_env)

    def load_from_file(self, path: Union[pathlib.Path, str],
                       old_size: tuple[int, int] = (0, 0)) -> None:
        """Load an image from *path* entirely in a background thread.

        Decode, mip generation, and DPG texture creation all happen off
        the main thread. If the decoded image dimensions differ from
        *old_size*, ``zoom_to_fit`` is called automatically.

        The old image remains visible (via the texture bridge) until the
        first new mip is ready — no blank flash.
        """
        self._augment_task_mgr.clear()
        self._mip_task_mgr.clear(wait=True)

        self._bridge_old_mips()

        # Don't clear _img_w/_img_h — the old-texture bridge render path
        # uses _old_img_w/h, and these fields get overwritten after decode.
        self._mip_loading = True
        self._needs_render = True

        task_env = env(path=path,
                       old_size=old_size,
                       device=self._device,
                       order=self._order,
                       mip_min_size=self._mip_min_size,
                       debug=self._debug)
        self._mip_task_mgr.submit(self._bg_file_mip_task, task_env)

    def augment_mips(self, path: Union[pathlib.Path, str]) -> None:
        """Generate missing (larger) mip levels and insert them into the live set.

        Call after ``set_preloaded_arrays`` when the preload cache provided
        capped mips (e.g. 0.25× max). Decodes the image, generates a full
        mipchain, and inserts any levels not already present — typically the
        1.0× and 0.5× levels.

        The existing mips remain visible and usable throughout — no bridge,
        no blank flash. Does NOT set ``mip_loading`` — the image is already
        displayable. Donation (``take_mip_arrays``) works at any time,
        yielding whatever mips are available.
        """
        self._augment_task_mgr.clear()
        task_env = env(path=path,
                       device=self._device,
                       order=self._order,
                       mip_min_size=self._mip_min_size,
                       debug=self._debug)
        self._augment_task_mgr.submit(self._bg_augment_task, task_env)

    def set_preloaded_arrays(self, mip_arrays: list[tuple[float, int, int, np.ndarray]],
                             img_w: int, img_h: int) -> None:
        """Display pre-computed mip arrays instantly (from preload cache).

        *mip_arrays*: list of ``(scale, w, h, flat_array)``, largest-first.
        *img_w*, *img_h*: original image dimensions.

        Creates DPG textures via the texture pool (``set_value`` on pooled
        textures is fast — no ``split_frame`` needed). Skips background
        mip generation entirely.
        """
        t0 = time.perf_counter_ns()
        self._augment_task_mgr.clear()
        self._mip_task_mgr.clear(wait=True)
        t_clear = time.perf_counter_ns()

        # Create DPG textures from flat arrays via the pool.
        new_mips = []
        for scale, w, h, flat in mip_arrays:
            tex_tag = self._acquire_texture(w, h, flat)
            new_mips.append((scale, tex_tag))

        with self._mips_lock:
            # Release all old textures. Preloaded images are ready instantly,
            # so no need for the old-image-shown-until-new-ready bridge that
            # set_image/bg_mip_task uses.
            for _s, old_tag in self._old_mips:
                self._release_texture(old_tag)
            self._old_mips.clear()
            for _s, old_tag in self._mips:
                self._release_texture(old_tag)
            self._mips = new_mips
            self._mip_arrays = list(mip_arrays)
        t_swap = time.perf_counter_ns()

        self._img_w = img_w
        self._img_h = img_h
        self._mip_loading = False

        # Re-render immediately so draw items reference the new textures.
        # Without this, the old draw items (from the previous _render call)
        # may reference pooled textures whose data was just overwritten by
        # _acquire_texture's set_value — causing a one-frame glitch when
        # DPG renders the stale draw items with recycled texture data.
        self._render()
        self._needs_render = False
        t_done = time.perf_counter_ns()

        if self._debug:
            logger.info(f"ImageView.set_preloaded_arrays: "
                        f"clear={(t_clear - t0) / 1e6:.1f}ms "
                        f"tex+swap={(t_swap - t_clear) / 1e6:.1f}ms "
                        f"total={(t_done - t0) / 1e6:.1f}ms "
                        f"pool_sizes={[len(v) for v in self._tex_pool.values()]} "
                        f"tex_count={len(self._tex_sizes)}")

    def take_mip_arrays(self) -> Optional[tuple[list, int, int]]:
        """Return stored flat arrays for donation to preload cache.

        Returns ``(mip_arrays, img_w, img_h)`` or ``None`` if unavailable.
        *mip_arrays* is a list of ``(scale, w, h, flat_array)``.
        The DPG textures are released to the pool. No GPU readback.
        """
        if not self.has_image or not self._mip_arrays:
            return None
        with self._mips_lock:
            mip_arrays = list(self._mip_arrays)
            for _s, tex_tag in self._mips:
                self._release_texture(tex_tag)
            self._mips = []
            self._mip_arrays = []
        return mip_arrays, self._img_w, self._img_h

    def clear(self) -> None:
        """Remove the current image."""
        self._augment_task_mgr.clear()
        self._mip_task_mgr.clear(wait=True)
        self._clear_textures()
        self._mip_arrays = []
        self._img_w = 0
        self._img_h = 0
        self._mip_loading = False
        self._needs_render = True

    def set_size(self, width: int, height: int) -> None:
        """Resize the drawlist (call from viewport resize callback)."""
        self._view_w = width
        self._view_h = height
        dpg.configure_item(self._drawlist_tag, width=width, height=height)
        if self._zoom_is_fit:
            self.zoom_to_fit()
        else:
            self._needs_render = True

    @property
    def focused(self) -> bool:
        """Whether the image pane has keyboard focus (Tab mode)."""
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self._needs_render = True  # redraw focus indicator

    @property
    def zoom(self) -> float:
        return self._zoom

    @property
    def image_size(self) -> tuple[int, int]:
        """``(width, height)`` of the loaded image, or ``(0, 0)``."""
        return (self._img_w, self._img_h)

    @property
    def has_image(self) -> bool:
        return self._img_w > 0 and self._img_h > 0

    @property
    def mip_loading(self) -> bool:
        """True while background mip generation is in progress."""
        return self._mip_loading

    # ------------------------------------------------------------------
    # Zoom / pan commands
    # ------------------------------------------------------------------

    def zoom_to_fit(self) -> None:
        """Fit the entire image in the view, centered."""
        if not self.has_image:
            return
        z, cx, cy = guiutils.compute_zoom_to_fit(self._img_w, self._img_h,
                                                  self._view_w, self._view_h)
        self._zoom = z
        self._pan_cx = cx
        self._pan_cy = cy
        self._zoom_is_fit = True
        self._on_zoom_update()

    def zoom_to_actual(self) -> None:
        """Zoom to 1:1 (100%), keeping the current center."""
        self._zoom = 1.0
        self._zoom_is_fit = False
        self._on_zoom_update()

    def zoom_in(self) -> None:
        """Zoom in by the configured step factor, centered on view."""
        self._zoom_at_center(config.ZOOM_IN_FACTOR)

    def zoom_out(self) -> None:
        """Zoom out by the configured step factor, centered on view."""
        self._zoom_at_center(1.0 / config.ZOOM_OUT_FACTOR)

    def pan_by(self, dx: float, dy: float) -> None:
        """Pan by a screen-pixel offset.

        Positive *dx* moves the image content to the right (view pans left).
        """
        if self._zoom == 0:
            return
        self._pan_cx -= dx / self._zoom
        self._pan_cy -= dy / self._zoom
        self._clamp_pan()
        self._needs_render = True

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Call from the render loop every frame.

        Checks for newly available mip levels (from background thread)
        and redraws if needed.
        """
        # The background thread sets _needs_render when a new mip is ready.
        if self._needs_render:
            self._render()
            self._needs_render = False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Remove DPG items and handlers. Call on app shutdown."""
        self._augment_task_mgr.clear(wait=True)
        self._mip_task_mgr.clear(wait=True)
        self._mip_executor.shutdown(wait=True)
        self._clear_textures()
        dpg.delete_item(self._handler_tag)
        dpg.delete_item(self._drawlist_tag)

    # ------------------------------------------------------------------
    # Internal: old-texture bridge
    # ------------------------------------------------------------------

    def _bridge_old_mips(self) -> None:
        """Preserve current mips as *old* mips for display continuity.

        Called at the start of ``set_image`` and ``load_from_file``.
        The render loop shows old mips (with their original zoom/pan/size)
        until the first new mip is ready, eliminating blank flashes.

        Only replaces ``_old_mips`` when ``_mips`` is non-empty. If the
        previous background task was cancelled before producing any mip
        (rapid navigation), ``_mips`` is ``[]`` and overwriting would lose
        the last successfully displayed image.
        """
        with self._mips_lock:
            if self._mips:
                for _s, old_tag in self._old_mips:
                    self._release_texture(old_tag)
                self._old_mips = list(self._mips)
                self._old_img_w = self._img_w
                self._old_img_h = self._img_h
                self._old_zoom = self._zoom
                self._old_pan_cx = self._pan_cx
                self._old_pan_cy = self._pan_cy
            self._mips = []
            self._mip_arrays = []

    # ------------------------------------------------------------------
    # Internal: background mip loading
    # ------------------------------------------------------------------

    def _bg_file_mip_task(self, e: env) -> None:
        """Background thread: decode image from file, then generate mips.

        Wraps ``_bg_mip_task`` — decodes the image, sets dimensions,
        handles zoom-to-fit, then delegates to the standard mip pipeline.
        """
        t0 = time.perf_counter_ns()
        try:
            e.rgba = imageutils.decode_image(e.path)
        except Exception as exc:
            logger.warning("ImageView._bg_file_mip_task: decode failed for %s: %s",
                           e.path, exc)
            self._mip_loading = False
            self._needs_render = True
            return
        t_decode = time.perf_counter_ns() - t0

        if e.cancelled:
            return

        # Set image dimensions (needed before zoom_to_fit).
        self._img_h, self._img_w = e.rgba.shape[:2]
        new_size = (self._img_w, self._img_h)
        if new_size != e.old_size:
            self.zoom_to_fit()
        self._needs_render = True

        if e.debug:
            logger.info("ImageView._bg_file_mip_task: decode=%dms size=%dx%d %s",
                        t_decode / 1e6, self._img_w, self._img_h, e.path)

        # Delegate to the standard mip pipeline.
        self._bg_mip_task(e)

    def _bg_augment_task(self, e: env) -> None:
        """Background thread: generate missing (larger) mip levels.

        Decodes the image, generates a full mipchain, and inserts levels
        not already present in ``_mips``. Processes from smallest new level
        to largest (progressive sharpening). Safe to cancel at any point —
        already-inserted mips remain valid.
        """
        t0 = time.perf_counter_ns()
        try:
            rgba = imageutils.decode_image(e.path)
        except Exception as exc:
            logger.warning("ImageView._bg_augment_task: decode failed for %s: %s",
                           e.path, exc)
            return
        t_decode = time.perf_counter_ns() - t0

        if e.cancelled:
            return

        # Upload + mipchain on GPU.
        tensor = imageutils.np_to_tensor(rgba, e.device)
        del rgba
        mip_tensors = lanczos.lanczos_mipchain(tensor, min_size=e.mip_min_size,
                                                order=e.order)

        # Identify which scales are already present.
        with self._mips_lock:
            existing_scales = {s for s, _t in self._mips}

        # Build list of missing levels (scale, tensor), largest-first.
        scale = 1.0
        missing = []
        for mip_t in mip_tensors:
            if scale not in existing_scales:
                missing.append((scale, mip_t))
            scale *= 0.5

        del tensor, mip_tensors

        if not missing:
            return  # nothing to augment

        # Create textures from smallest missing level to largest
        # (progressive sharpening, same order as _bg_mip_task).
        n_inserted = 0
        for mip_scale, mip_t in reversed(missing):
            if e.cancelled:
                break
            _, _, mh, mw = mip_t.shape
            flat = imageutils.tensor_to_dpg_flat(mip_t)

            if e.cancelled:
                break
            tex_tag = self._acquire_texture(mw, mh, flat)
            dpg.split_frame()

            if e.cancelled:
                self._release_texture(tex_tag)
                break

            # Insert into _mips sorted (largest-first).
            with self._mips_lock:
                inserted = False
                for i, (s, _t) in enumerate(self._mips):
                    if mip_scale > s:
                        self._mips.insert(i, (mip_scale, tex_tag))
                        self._mip_arrays.insert(i, (mip_scale, mw, mh, flat))
                        inserted = True
                        break
                if not inserted:
                    self._mips.append((mip_scale, tex_tag))
                    self._mip_arrays.append((mip_scale, mw, mh, flat))
            self._needs_render = True
            n_inserted += 1

        if e.debug:
            t_total = (time.perf_counter_ns() - t0) / 1e6
            logger.info("ImageView._bg_augment_task: inserted %d levels, "
                        "decode=%dms total=%dms",
                        n_inserted, t_decode / 1e6, t_total)

    def _bg_mip_task(self, e: env) -> None:
        """Background thread: upload, generate mip chain, create DPG textures.

        Creates textures from smallest to largest, so the image appears
        quickly at reduced quality and sharpens progressively.
        """
        t_total_start = time.perf_counter_ns()

        # Upload to GPU.
        t0 = time.perf_counter_ns()
        tensor = imageutils.np_to_tensor(e.rgba, e.device)
        # No cuda.synchronize here — the mipchain naturally waits for the
        # upload (same stream), and per-mip .cpu() calls in tensor_to_dpg_flat
        # provide synchronization. Device-wide sync would stall on the
        # thumbnail pipeline's CUDA work, adding hundreds of ms.
        t_upload = time.perf_counter_ns() - t0

        if e.cancelled:
            return

        # Generate mip chain on GPU.
        t0 = time.perf_counter_ns()
        mip_tensors = lanczos.lanczos_mipchain(tensor,
                                               min_size=e.mip_min_size,
                                               order=e.order)
        t_mipgen = time.perf_counter_ns() - t0

        # Build (scale, tensor) list, largest-first.
        scale = 1.0
        mip_data = []
        for mip_t in mip_tensors:
            mip_data.append((scale, mip_t))
            scale *= 0.5

        # Create DPG textures from smallest to largest.
        # Per-mip split_frame is needed: without it, the render loop may
        # draw_image with a texture that DPG hasn't uploaded to OpenGL yet,
        # causing visual glitches.
        for mip_scale, mip_t in reversed(mip_data):
            if e.cancelled:
                break

            _, _, mh, mw = mip_t.shape
            t0 = time.perf_counter_ns()
            flat = imageutils.tensor_to_dpg_flat(mip_t)
            t_xfer = time.perf_counter_ns() - t0

            if e.cancelled:
                break
            t0 = time.perf_counter_ns()
            tex_tag = self._acquire_texture(mw, mh, flat)
            t_tex = time.perf_counter_ns() - t0

            if e.cancelled:
                break
            dpg.split_frame()

            # Insert into _mips sorted (largest-first).
            # Also store the flat array for zero-cost donation later.
            with self._mips_lock:
                inserted = False
                for i, (s, _t) in enumerate(self._mips):
                    if mip_scale > s:
                        self._mips.insert(i, (mip_scale, tex_tag))
                        self._mip_arrays.insert(i, (mip_scale, mw, mh, flat))
                        inserted = True
                        break
                if not inserted:
                    self._mips.append((mip_scale, tex_tag))
                    self._mip_arrays.append((mip_scale, mw, mh, flat))

            # Release old textures to pool once we have something new to show.
            if self._old_mips:
                with self._mips_lock:
                    for _s, old_tag in self._old_mips:
                        self._release_texture(old_tag)
                    self._old_mips.clear()

            # Trigger a render for every new mip (progressive sharpening).
            self._needs_render = True

            if e.debug:
                logger.info(f"ImageView._bg_mip_task: {mw}x{mh} "
                            f"scale={mip_scale:.3f} "
                            f"xfer={t_xfer / 1e6:.0f}ms "
                            f"tex={t_tex / 1e6:.0f}ms")

        t_total = (time.perf_counter_ns() - t_total_start) / 1e6
        if e.debug:
            sizes = [f"{mt.shape[3]}x{mt.shape[2]}" for mt in mip_tensors]
            logger.info(f"ImageView._bg_mip_task: done. upload={t_upload / 1e6:.0f}ms "
                        f"mipgen={t_mipgen / 1e6:.0f}ms "
                        f"levels={sizes} total={t_total:.0f}ms")

        del tensor, mip_tensors
        # Don't call empty_cache() here — it returns the allocator's cached
        # blocks to the driver, forcing cudaMalloc on the next load. Over
        # Thunderbolt, cudaMalloc requires PCIe round-trips (~25ms each),
        # turning a 1ms mipchain into a 500ms+ stall.

        if not e.cancelled:
            self._mip_loading = False
            self._needs_render = True  # clear loading indicator

    # ------------------------------------------------------------------
    # Internal: rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        """Redraw the drawlist.

        Creates new draw items first, then deletes the previous batch.
        The drawlist is never empty — eliminates blank-frame glitches
        that occur when ``delete_item(children_only=True)`` clears the
        drawlist before new items are added.
        """
        # Use new mips if available, otherwise show old image until ready.
        # When showing old mips, also use the old zoom/pan/dimensions so the
        # old image isn't distorted by the new image's aspect ratio.
        if self._mips:
            active_mips = self._mips
            img_w, img_h = self._img_w, self._img_h
            zoom = self._zoom
            pan_cx, pan_cy = self._pan_cx, self._pan_cy
        elif self._old_mips:
            active_mips = self._old_mips
            img_w, img_h = self._old_img_w, self._old_img_h
            zoom = self._old_zoom
            pan_cx, pan_cy = self._old_pan_cx, self._old_pan_cy
        else:
            return

        # Select the best mip level for current zoom.
        mip_scale, tex_tag = self._select_mip_from(active_mips)

        # Image corners in screen (drawlist) coordinates.
        pmin = guiutils.content_to_screen(0, 0,
                                          pan_cx, pan_cy,
                                          zoom, self._view_w, self._view_h)
        pmax = guiutils.content_to_screen(img_w, img_h,
                                          pan_cx, pan_cy,
                                          zoom, self._view_w, self._view_h)

        # At 1:1 zoom, snap to pixel grid to avoid subpixel blur.
        # Round pmin, then derive pmax exactly so the texture maps 1:1.
        if zoom == 1.0:
            pmin = (round(pmin[0]), round(pmin[1]))
            pmax = (pmin[0] + img_w, pmin[1] + img_h)

        # --- Create new draw items ---
        new_items = []

        item = dpg.draw_image(tex_tag, pmin=pmin, pmax=pmax,
                              parent=self._drawlist_tag)
        new_items.append(item)

        # Debug overlay: pan/zoom coordinates.
        if self._debug:
            mip_label = f"1/{1 / mip_scale:.0f}" if mip_scale > 0 else "?"
            item = dpg.draw_text((4, 4),
                                 f"pan=({self._pan_cx:.1f}, {self._pan_cy:.1f}) "
                                 f"zoom={self._zoom:.3f} "
                                 f"img={self._img_w}x{self._img_h} "
                                 f"fit={self._zoom_is_fit} "
                                 f"mip={mip_label} "
                                 f"mips={len(self._mips)}",
                                 color=(0, 255, 0, 200), size=config.FONT_SIZE,
                                 parent=self._drawlist_tag)
            new_items.append(item)

        # Loading indicator — shown while background mip generation is running.
        if self._mip_loading:
            label = "Enhancing..."
            tx = 8
            ty = self._view_h - config.FONT_SIZE - 8
            # Dark background pill for readability over any image content.
            pad_x, pad_y = 6, 2
            item = dpg.draw_rectangle(pmin=(tx - pad_x, ty - pad_y),
                                      pmax=(tx + len(label) * config.FONT_SIZE * 0.52 + pad_x,
                                            ty + config.FONT_SIZE + pad_y),
                                      color=(0, 0, 0, 0), fill=(0, 0, 0, 160),
                                      rounding=4,
                                      parent=self._drawlist_tag)
            new_items.append(item)
            item = dpg.draw_text((tx, ty), label,
                                 color=(200, 200, 200, 220), size=config.FONT_SIZE,
                                 parent=self._drawlist_tag)
            new_items.append(item)

        # Focus indicator.
        if self._focused:
            item = dpg.draw_rectangle(pmin=(1, 1),
                                      pmax=(self._view_w - 1, self._view_h - 1),
                                      color=config.CURRENT_COLOR,
                                      thickness=2,
                                      parent=self._drawlist_tag)
            new_items.append(item)

        # --- Delete previous draw items (new ones already cover them) ---
        for old_item in self._drawn_items:
            try:
                dpg.delete_item(old_item)
            except Exception:
                pass
        self._drawn_items = new_items

    def _select_mip_from(self, mips: list[tuple[float, int | str]]) -> tuple[float, int | str]:
        """Pick the mip level closest to (but not smaller than) the current zoom.

        At zoom `z`, each image pixel occupies `z` screen pixels. We want
        the mip whose scale is >= `z` so the GPU only downscales the
        texture (bilinear downscaling looks fine over a <= 2x range).
        Falls back to the best available if the ideal mip hasn't loaded yet.
        """
        # Mips are sorted largest-first: [(1.0, tex), (0.5, tex), (0.25, tex), ...]
        best = mips[0]
        for scale, tex_tag in mips:
            if scale >= self._zoom:
                best = (scale, tex_tag)
            else:
                break
        return best

    # ------------------------------------------------------------------
    # Internal: texture management
    # ------------------------------------------------------------------

    def _acquire_texture(self, w: int, h: int, flat) -> str:
        """Get a DPG texture of size ``(w, h)``, reusing from pool if possible.

        If a matching texture exists in the pool, updates its data via
        ``set_value`` (fast, no OpenGL texture creation).  Otherwise creates
        a new ``dynamic_texture``.
        """
        key = (w, h)
        pool = self._tex_pool.get(key)
        if pool:
            tex_tag = pool.pop()
            dpg.set_value(tex_tag, flat)
            return tex_tag
        tex_tag = _next_tag("mip")
        with dpg.texture_registry():
            dpg.add_dynamic_texture(w, h,
                                    default_value=flat,
                                    tag=tex_tag)
        self._tex_sizes[tex_tag] = key
        return tex_tag

    def _release_texture(self, tex_tag) -> None:
        """Return a texture to the pool for reuse."""
        size = self._tex_sizes.get(tex_tag)
        if size is not None:
            self._tex_pool.setdefault(size, []).append(tex_tag)

    def _clear_textures(self) -> None:
        """Delete all mip DPG textures (current, old, and pooled)."""
        with self._mips_lock:
            for mip_list in (self._mips, self._old_mips):
                for _scale, tex_tag in mip_list:
                    try:
                        dpg.delete_item(tex_tag)
                    except Exception:
                        pass
                    self._tex_sizes.pop(tex_tag, None)
                mip_list.clear()
        for pool_list in self._tex_pool.values():
            for tex_tag in pool_list:
                try:
                    dpg.delete_item(tex_tag)
                except Exception:
                    pass
                self._tex_sizes.pop(tex_tag, None)
        self._tex_pool.clear()

    # ------------------------------------------------------------------
    # Internal: zoom helpers
    # ------------------------------------------------------------------

    def _zoom_at_screen(self, factor: float, sx: float, sy: float) -> None:
        """Zoom by *factor*, keeping the image point under (sx, sy) stationary."""
        self._zoom_is_fit = False
        new_zoom = max(0.01, min(100.0, self._zoom * factor))
        self._pan_cx, self._pan_cy = guiutils.zoom_keep_point(
            self._zoom, new_zoom, sx, sy,
            self._pan_cx, self._pan_cy,
            self._view_w, self._view_h)
        self._zoom = new_zoom
        self._on_zoom_update()

    def _zoom_at_center(self, factor: float) -> None:
        """Zoom by *factor*, centered on the view."""
        self._zoom_is_fit = False
        new_zoom = self._zoom * factor
        new_zoom = max(0.01, min(100.0, new_zoom))
        self._zoom = new_zoom
        self._on_zoom_update()

    def _clamp_pan(self) -> None:
        """Keep the viewport center within the image bounds."""
        if not self.has_image:
            return
        self._pan_cx = max(0, min(self._img_w, self._pan_cx))
        self._pan_cy = max(0, min(self._img_h, self._pan_cy))

    def _on_zoom_update(self) -> None:
        """Common handling after any zoom change."""
        self._clamp_pan()
        self._needs_render = True
        if self._on_zoom_changed is not None:
            self._on_zoom_changed(self._zoom)

    # ------------------------------------------------------------------
    # Internal: mouse handlers
    # ------------------------------------------------------------------

    def _on_mouse_drag(self, sender, app_data) -> None:
        """Handle mouse drag for panning."""
        if not self.input_enabled:
            return
        button, dx, dy = app_data

        if not self._dragging:
            # Drag starts — check if we're over the drawlist.
            if not guiutils.is_mouse_inside_widget(self._drawlist_tag):
                return
            self._dragging = True
            self._drag_cumulative = (dx, dy)
            self._drag_pan_start = (self._pan_cx, self._pan_cy)
            return

        # Per-frame delta from cumulative.
        frame_dx = dx - self._drag_cumulative[0]
        frame_dy = dy - self._drag_cumulative[1]
        self._drag_cumulative = (dx, dy)

        self.pan_by(frame_dx, frame_dy)

    def _on_mouse_release(self, sender, app_data) -> None:
        """End drag."""
        self._dragging = False

    def _on_mouse_wheel(self, sender, app_data) -> None:
        """Handle mouse wheel for zooming."""
        if not self.input_enabled:
            return
        if not guiutils.is_mouse_inside_widget(self._drawlist_tag):
            return

        delta = app_data  # positive = scroll up = zoom in
        sx, sy = guiutils.get_mouse_relative_pos(self._drawlist_tag)

        factor = config.MOUSE_WHEEL_ZOOM_FACTOR if delta > 0 else 1.0 / config.MOUSE_WHEEL_ZOOM_FACTOR
        self._zoom_at_screen(factor, sx, sy)
