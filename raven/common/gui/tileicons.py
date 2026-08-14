"""Icon assets resampled to a thumbnail grid's tile size, as DPG textures.

A grid tile is square and much larger than a toolbar icon, so an icon shown in one has to be enlarged.
**DPG scales textures nearest-neighbor**, so handing it a 16×16 icon and asking for 128 gives visible
blocking — next to Lanczos-resampled photographs, which is where these icons appear, the difference reads
as a bug rather than as a small picture. So the resampling happens here, once per (icon, tile size), and
the result is a texture the grid can draw at 1:1.

One texture serves every tile showing that icon: a directory of a thousand files needs a handful of
textures rather than a thousand. `ThumbnailGrid.set_shared_image` is the other half of that arrangement.

Sources are given in the shape `dpg.load_image` returns — width, height, and flat RGBA floats in [0, 1] —
so an asset already loaded for a toolbar can be handed over without being read again.
"""

__all__ = ["TileIconCache"]

import logging
import threading
from collections.abc import Mapping, Sequence
from typing import Optional, Union

import numpy as np
import torch

import dearpygui.dearpygui as dpg

from ..image import lanczos
from ..image import utils as imageutils
from . import utils as guiutils

logger = logging.getLogger(__name__)

# Counter for unique DPG tags.
_tag_counter = 0
_tag_lock = threading.Lock()


def _next_tag() -> str:
    global _tag_counter
    with _tag_lock:
        _tag_counter += 1
        return f"tile_icon_tex_{_tag_counter}"


class TileIconCache:
    """Icon assets, resampled to one tile size at a time, as DPG textures.

    Register the sources once with `add`, then ask `texture` for the tag whenever a tile needs one.
    Textures are built on first request and kept until the tile size changes or the cache is destroyed.

    Thread-safe: DPG permits texture work from any thread, and the grid's owner is typically not on the
    render thread.
    """

    def __init__(self,
                 tile_size: int,
                 *,
                 device: Union[torch.device, str] = "cpu",
                 dtype: torch.dtype = torch.float32,
                 order: int = lanczos.DEFAULT_ORDER):
        """
        *tile_size*: edge of the square tile, in pixels.
        *device*, *dtype*: where the resampling runs. CPU is the sensible default — this is a few dozen
            small images, once per tile size, so a GPU transfer costs more than the work it saves.
        *order*: Lanczos kernel order, as in `raven.common.image.lanczos.resize`.
        """
        self._lock = threading.RLock()
        self._tile_size = tile_size
        self._device = torch.device(device)
        self._dtype = dtype
        self._order = order
        self._sources: dict[str, tuple[int, int, np.ndarray]] = {}
        self._textures: dict[str, str] = {}

    def add(self, name: str, width: int, height: int, rgba: Sequence[float]) -> None:
        """Register an icon source under *name*.

        *rgba*: flat RGBA floats in [0, 1], ``width * height * 4`` of them — what `dpg.load_image` returns
        as its fourth value.
        """
        arr = np.asarray(rgba, dtype=np.float32)
        expected = width * height * 4
        if arr.size != expected:
            raise ValueError(f"TileIconCache.add: '{name}': expected {expected} floats for {width}x{height} RGBA, got {arr.size}")
        with self._lock:
            self._sources[name] = (width, height, arr)
            self._drop_texture(name)

    def add_all(self, sources: Mapping[str, tuple[int, int, Sequence[float]]]) -> None:
        """Register several icons at once, as ``{name: (width, height, rgba)}``."""
        for name, (width, height, rgba) in sources.items():
            self.add(name, width, height, rgba)

    def texture(self, name: str) -> Optional[str]:
        """The DPG texture tag for icon *name* at the current tile size, or `None` if there is no such icon.

        Built on first request. An unknown name is not an error: a caller mapping file types to icons will
        have types it has no picture for, and answering `None` lets the tile fall back to whatever it shows
        when there is no image.
        """
        with self._lock:
            tag = self._textures.get(name)
            if tag is not None:
                return tag
            source = self._sources.get(name)
            if source is None:
                return None
            tag = self._build(name, source)
            self._textures[name] = tag
            return tag

    def set_tile_size(self, tile_size: int) -> None:
        """Change the tile size, discarding every texture built for the old one."""
        with self._lock:
            if tile_size == self._tile_size:
                return
            self._tile_size = tile_size
            self._drop_all_textures()

    def get_tile_size(self) -> int:
        with self._lock:
            return self._tile_size
    tile_size = property(fget=get_tile_size, doc="Edge of the square tile these icons are built for, in pixels.")

    def destroy(self) -> None:
        """Delete every texture. The registered sources are kept, so the cache can be used again."""
        with self._lock:
            self._drop_all_textures()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build(self, name: str, source: tuple[int, int, np.ndarray]) -> str:
        width, height, arr = source
        ts = self._tile_size
        tensor = (torch.from_numpy(arr.reshape(height, width, 4))
                  .permute(2, 0, 1)
                  .unsqueeze(0)
                  .to(device=self._device, dtype=self._dtype))
        # Transparent padding, not the letterbox default: an icon narrower than the tile should show the
        # panel behind it rather than a gray bar. Square sources — which every current icon is — pad by
        # nothing at all, so this only matters for the ones that are not.
        tile = imageutils.letterbox(tensor, ts, order=self._order,
                                    bg_value=(0.0, 0.0, 0.0, 0.0), allow_upscale=True)
        flat = imageutils.tensor_to_dpg_flat(tile)
        tag = _next_tag()
        with dpg.texture_registry():
            dpg.add_static_texture(ts, ts, default_value=flat, tag=tag)
        logger.debug(f"TileIconCache._build: instance 0x{id(self):x}: '{name}' {width}x{height} -> {ts}x{ts}")
        return tag

    def _drop_texture(self, name: str) -> None:
        tag = self._textures.pop(name, None)
        if tag is not None:
            guiutils.maybe_delete_item(tag)

    def _drop_all_textures(self) -> None:
        for tag in self._textures.values():
            guiutils.maybe_delete_item(tag)
        self._textures.clear()
