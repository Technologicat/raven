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
import threading
import time
from typing import Callable, Optional

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
        # Cancel any in-progress mip loading.
        self._mip_task_mgr.clear(wait=True)

        # Keep old textures visible until the first new mip is ready.
        # This eliminates the blank flash between images.
        #
        # Only replace _old_mips when _mips is non-empty. If the previous
        # background task was cancelled before producing any mip (rapid
        # navigation), _mips is [] and overwriting _old_mips would lose the
        # last successfully displayed image — causing a blank flash.
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

    def take_mips(self) -> Optional[tuple[list, dict, int, int]]:
        """Remove and return current mips, transferring ownership to caller.

        Returns ``(mips, tex_sizes, img_w, img_h)`` or ``None`` if no image.
        After this call, ImageView has no mips (caller owns the textures).
        """
        if not self.has_image or not self._mips:
            return None
        with self._mips_lock:
            mips = list(self._mips)
            tex_sizes = {tag: self._tex_sizes.pop(tag)
                         for _s, tag in mips if tag in self._tex_sizes}
            self._mips = []
        return mips, tex_sizes, self._img_w, self._img_h

    def set_preloaded_mips(self, mips: list[tuple[float, int | str]],
                           tex_sizes: dict[int | str, tuple[int, int]],
                           img_w: int, img_h: int) -> None:
        """Display pre-built mip textures instantly (from preload cache).

        *mips*: list of ``(scale, dpg_texture_tag)``, largest-first.
        *tex_sizes*: ``{tex_tag: (width, height)}`` for each texture.
        *img_w*, *img_h*: original image dimensions.

        Skips background mip generation entirely. The caller transfers
        texture ownership — ImageView will release these to its pool
        when done.
        """
        self._mip_task_mgr.clear(wait=True)

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
            self._mips = list(mips)

        # Adopt the textures into our size tracking (so _release_texture works).
        self._tex_sizes.update(tex_sizes)

        self._img_w = img_w
        self._img_h = img_h
        self._mip_loading = False
        self._needs_render = True

    def clear(self) -> None:
        """Remove the current image."""
        self._mip_task_mgr.clear(wait=True)
        self._clear_textures()
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
        self._mip_task_mgr.clear(wait=True)
        self._mip_executor.shutdown(wait=True)
        self._clear_textures()
        dpg.delete_item(self._handler_tag)
        dpg.delete_item(self._drawlist_tag)

    # ------------------------------------------------------------------
    # Internal: background mip loading
    # ------------------------------------------------------------------

    def _bg_mip_task(self, e: env) -> None:
        """Background thread: upload, generate mip chain, create DPG textures.

        Creates textures from smallest to largest, so the image appears
        quickly at reduced quality and sharpens progressively.
        """
        t_total_start = time.perf_counter_ns()

        # Upload to GPU.
        t0 = time.perf_counter_ns()
        tensor = imageutils.np_to_tensor(e.rgba, e.device)
        if e.device.type == "cuda":
            torch.cuda.synchronize(e.device)
        t_upload = time.perf_counter_ns() - t0

        if e.cancelled:
            return

        # Generate mip chain on GPU.
        t0 = time.perf_counter_ns()
        mip_tensors = lanczos.lanczos_mipchain(tensor,
                                               min_size=e.mip_min_size,
                                               order=e.order)
        if e.device.type == "cuda":
            torch.cuda.synchronize(e.device)
        t_mipgen = time.perf_counter_ns() - t0

        # Build (scale, tensor) list, largest-first.
        scale = 1.0
        mip_data = []
        for mip_t in mip_tensors:
            mip_data.append((scale, mip_t))
            scale *= 0.5

        # Create DPG textures from smallest to largest.
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

            # Wait for DPG to process the texture before the render loop
            # tries to use it. Eliminates flicker from half-uploaded textures.
            # (split_frame is safe from background threads; it waits for the
            # main thread's render loop to complete one frame.)
            if e.cancelled:
                break
            dpg.split_frame()

            # Insert into _mips sorted (largest-first).
            with self._mips_lock:
                inserted = False
                for i, (s, _t) in enumerate(self._mips):
                    if mip_scale > s:
                        self._mips.insert(i, (mip_scale, tex_tag))
                        inserted = True
                        break
                if not inserted:
                    self._mips.append((mip_scale, tex_tag))

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
        if e.device.type == "cuda":
            torch.cuda.empty_cache()

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
            item = dpg.draw_text((4, 4),
                                 f"pan=({self._pan_cx:.1f}, {self._pan_cy:.1f}) "
                                 f"zoom={self._zoom:.3f} "
                                 f"img={self._img_w}x{self._img_h} "
                                 f"fit={self._zoom_is_fit} "
                                 f"mip={mip_scale:.3f} "
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
