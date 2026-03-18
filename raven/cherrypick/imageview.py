"""Main image viewer pane for raven-cherrypick.

Displays a single image on a DPG drawlist with pan/zoom interaction and
Lanczos mipmap selection for quality display at intermediate zoom-out levels.

Coordinate system:
  - **Image coordinates**: (0, 0) at top-left of image, in pixels.
  - **Screen coordinates**: (0, 0) at top-left of the drawlist widget.
  - Pan state ``(pan_cx, pan_cy)`` is the image coordinate at the center
    of the view.  Zoom is screen pixels per image pixel.

See `raven.common.gui.utils.content_to_screen` for the transform formula.

Mip textures are loaded **progressively**: small levels are created
immediately (cheap), and one larger level is promoted per frame. This
avoids blocking the render loop when switching to large images.
"""

__all__ = ["ImageView"]

import logging
import time
from typing import Callable, Optional

import dearpygui.dearpygui as dpg
import numpy as np
import torch

from ..common.gui import utils as guiutils
from ..common.image import lanczos
from ..common.image import utils as imageutils
from . import config

logger = logging.getLogger(__name__)

# Counter for unique DPG tags.
_tag_counter = 0


def _next_tag(prefix: str) -> str:
    global _tag_counter
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

        # Progressive mip loading: GPU tensors waiting for DPG texture creation.
        # Sorted largest-first (level 0 = full res, created last).
        self._pending_mips: list[tuple[float, torch.Tensor]] = []

        # Interaction state.
        self.input_enabled: bool = True
        self._focused = False
        self._dragging = False
        self._drag_cumulative = (0.0, 0.0)
        self._drag_pan_start = (0.0, 0.0)
        self._needs_render = True

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

        Generates a Lanczos mip chain on the GPU. Creates DPG textures
        for the small mip levels immediately (cheap). Larger levels are
        promoted progressively, one per frame, via `update`.
        """
        self._clear_textures()
        self._pending_mips.clear()
        self._img_h, self._img_w = rgba.shape[:2]

        t0 = time.perf_counter_ns()

        # Generate mip chain on GPU.
        tensor = imageutils.np_to_tensor(rgba, self._device)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        t_upload = time.perf_counter_ns() - t0

        t1 = time.perf_counter_ns()
        mip_tensors = lanczos.lanczos_mipchain(tensor,
                                               min_size=self._mip_min_size,
                                               order=self._order)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        t_mipgen = time.perf_counter_ns() - t1

        # Determine which mip level is needed for the current zoom.
        # For zoom-to-fit, that's the smallest mip >= displayed size.
        fit_zoom = min(self._view_w / self._img_w,
                       self._view_h / self._img_h) if self._img_w > 0 and self._img_h > 0 else 1.0
        needed_zoom = self._zoom if not self._zoom_is_fit else fit_zoom

        # Create DPG textures from smallest upward. Stop once we've
        # created one level that covers the needed zoom (plus one extra
        # for safety). Queue the rest for progressive loading.
        scale = 1.0
        mip_data = []  # (scale, tensor) sorted largest-first
        for mip_t in mip_tensors:
            mip_data.append((scale, mip_t))
            scale *= 0.5

        # Process smallest-first (reverse order).
        created_count = 0
        sufficient = False
        for scale, mip_t in reversed(mip_data):
            if not sufficient or created_count < 2:
                # Create this level's DPG texture now.
                self._create_mip_texture(scale, mip_t)
                created_count += 1
                if scale >= needed_zoom:
                    sufficient = True
            else:
                # Queue for progressive loading (keep GPU tensor alive).
                self._pending_mips.append((scale, mip_t.clone()))

        # Pending mips are largest-first; we'll pop from the end (smallest first).
        # Actually we want to promote the next-largest, so reverse:
        # they're already largest-first from the loop above.

        if self._debug:
            mip_sizes = [f"{mt.shape[3]}x{mt.shape[2]}" for mt in mip_tensors]
            logger.info(f"ImageView.set_image: upload={t_upload / 1e6:.0f}ms "
                        f"mipgen={t_mipgen / 1e6:.0f}ms "
                        f"levels={mip_sizes} "
                        f"created={created_count} "
                        f"pending={len(self._pending_mips)}")

        del tensor, mip_tensors
        if not self._pending_mips and self._device.type == "cuda":
            torch.cuda.empty_cache()

        self._needs_render = True

    def clear(self) -> None:
        """Remove the current image."""
        self._clear_textures()
        self._pending_mips.clear()
        self._img_w = 0
        self._img_h = 0
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

        Promotes one pending mip level per frame (progressive loading),
        then redraws if needed.
        """
        if self._pending_mips:
            self._promote_one_mip()

        if self._needs_render:
            self._render()
            self._needs_render = False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Remove DPG items and handlers. Call on app shutdown."""
        self._clear_textures()
        self._pending_mips.clear()
        dpg.delete_item(self._handler_tag)
        dpg.delete_item(self._drawlist_tag)

    # ------------------------------------------------------------------
    # Internal: progressive mip loading
    # ------------------------------------------------------------------

    def _create_mip_texture(self, scale: float, mip_tensor: torch.Tensor) -> None:
        """Create a DPG texture for one mip level and insert into self._mips."""
        _, _, mh, mw = mip_tensor.shape
        flat = imageutils.tensor_to_dpg_flat(mip_tensor)
        tex_tag = _next_tag("mip")
        with dpg.texture_registry():
            dpg.add_dynamic_texture(mw, mh,
                                    default_value=flat,
                                    tag=tex_tag)
        # Insert sorted: _mips is largest-first.
        inserted = False
        for i, (s, _tag) in enumerate(self._mips):
            if scale > s:
                self._mips.insert(i, (scale, tex_tag))
                inserted = True
                break
        if not inserted:
            self._mips.append((scale, tex_tag))

    def _promote_one_mip(self) -> None:
        """Create one pending mip texture (called once per frame)."""
        if not self._pending_mips:
            return

        # Pop the last entry (smallest pending = next size up from what we have).
        scale, mip_tensor = self._pending_mips.pop()

        if self._debug:
            t0 = time.perf_counter_ns()

        self._create_mip_texture(scale, mip_tensor)

        if self._debug:
            elapsed = (time.perf_counter_ns() - t0) / 1e6
            _, _, mh, mw = mip_tensor.shape
            logger.info(f"ImageView._promote_one_mip: {mw}x{mh} "
                        f"scale={scale:.3f} took {elapsed:.0f}ms "
                        f"remaining={len(self._pending_mips)}")

        del mip_tensor
        if not self._pending_mips and self._device.type == "cuda":
            torch.cuda.empty_cache()

        self._needs_render = True

    # ------------------------------------------------------------------
    # Internal: rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        """Redraw the drawlist."""
        dpg.delete_item(self._drawlist_tag, children_only=True)

        if not self._mips:
            return

        # Select the best mip level for current zoom.
        mip_scale, tex_tag = self._select_mip()

        # Image corners in screen (drawlist) coordinates.
        pmin = guiutils.content_to_screen(0, 0,
                                          self._pan_cx, self._pan_cy,
                                          self._zoom, self._view_w, self._view_h)
        pmax = guiutils.content_to_screen(self._img_w, self._img_h,
                                          self._pan_cx, self._pan_cy,
                                          self._zoom, self._view_w, self._view_h)

        # At 1:1 zoom, snap to pixel grid to avoid subpixel blur.
        # Round pmin, then derive pmax exactly so the texture maps 1:1.
        if self._zoom == 1.0:
            pmin = (round(pmin[0]), round(pmin[1]))
            pmax = (pmin[0] + self._img_w, pmin[1] + self._img_h)

        dpg.draw_image(tex_tag, pmin=pmin, pmax=pmax,
                       parent=self._drawlist_tag)

        # Debug overlay: pan/zoom coordinates.
        if self._debug:
            dpg.draw_text((4, 4),
                          f"pan=({self._pan_cx:.1f}, {self._pan_cy:.1f}) "
                          f"zoom={self._zoom:.3f} "
                          f"img={self._img_w}x{self._img_h} "
                          f"fit={self._zoom_is_fit} "
                          f"mip={mip_scale:.3f} "
                          f"pending={len(self._pending_mips)}",
                          color=(0, 255, 0, 200), size=config.FONT_SIZE,
                          parent=self._drawlist_tag)

        # Focus indicator.
        if self._focused:
            dpg.draw_rectangle(pmin=(1, 1),
                               pmax=(self._view_w - 1, self._view_h - 1),
                               color=config.CURRENT_COLOR,
                               thickness=2,
                               parent=self._drawlist_tag)

    def _select_mip(self) -> tuple[float, int | str]:
        """Pick the mip level closest to (but not smaller than) the current zoom.

        At zoom `z`, each image pixel occupies `z` screen pixels. We want
        the mip whose scale is >= `z` so the GPU only downscales the
        texture (bilinear downscaling looks fine over a <= 2x range).
        Falls back to the best available if the ideal mip hasn't loaded yet.
        """
        # Mips are sorted largest-first: [(1.0, tex), (0.5, tex), (0.25, tex), ...]
        best = self._mips[0]
        for scale, tex_tag in self._mips:
            if scale >= self._zoom:
                best = (scale, tex_tag)
            else:
                break
        return best

    # ------------------------------------------------------------------
    # Internal: texture management
    # ------------------------------------------------------------------

    def _clear_textures(self) -> None:
        """Delete all mip DPG textures."""
        for _scale, tex_tag in self._mips:
            try:
                dpg.delete_item(tex_tag)
            except Exception:
                pass
        self._mips.clear()

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
