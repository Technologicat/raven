"""Thumbnail generation pipeline: decode a list of images and Lanczos-resize them to uniform tiles.

Triple-buffered with two background threads:

  - **Decode thread** (CPU): reads and decodes images to numpy RGBA arrays.
  - **GPU thread**: transfers decoded arrays to the GPU, Lanczos-resizes to
    tile size (with imageutils.letterboxing for non-square images), transfers back.

The caller polls for completed thumbnails and does whatever it wants with them — creating DPG textures, in
both current callers, but nothing here knows that: the output is flat float32 RGBA, and this module imports
no GUI toolkit.

The pipeline is managed via ``raven.common.bgtask.TaskManager`` for cooperative cancellation when the
caller moves on (a new folder, a new visible range).
"""

__all__ = ["placeholder_tiles", "ThumbnailPipeline"]

import concurrent.futures
import logging
import pathlib
import queue
from typing import Union

import numpy as np
import torch

from unpythonic.env import env

from .. import bgtask
from ..video import colorspace
from ..video import postprocessor
from . import codec as imagecodec
from . import lanczos
from . import utils as imageutils

logger = logging.getLogger(__name__)

# How long each thread waits on a queue before re-checking the cancellation flag.
_QUEUE_TIMEOUT_S = 0.5


def placeholder_tiles(n: int,
                      tile_size: int,
                      *,
                      device: Union[torch.device, str],
                      dtype: torch.dtype = torch.float32,
                      tint: tuple[float, float, float] = (0.92, 0.92, 1.0),
                      brightness: tuple[float, float] = (0.04, 0.40),
                      mode: str = "PAL") -> list[np.ndarray]:
    """Generate *n* VHS-noise tiles for a grid to show until the real thumbnails arrive.

    Returns flat float32 RGBA arrays of ``tile_size * tile_size * 4``, which is what
    `raven.common.gui.thumbnailgrid.ThumbnailGrid.set_noise_pool` takes.

    Here rather than in each app because the noise *is* the look: two grids in the same constellation
    showing different placeholders would read as two different pieces of software. The parameters are
    exposed anyway, since an app with a different palette may want to match it.

    `tint`, `brightness`, `mode`: see `raven.common.video.postprocessor.vhs_noise_pool`.
    """
    tiles = postprocessor.vhs_noise_pool(n, tile_size, tile_size,
                                         device=device, dtype=dtype,
                                         tint=tint, brightness=brightness, mode=mode)
    for tile in tiles:
        # Colour channels only — alpha is coverage, not light, and gamma-encoding it would make the tiles
        # translucent.
        tile[:3] = colorspace.linear_to_srgb(tile[:3])
    return [imageutils.tensor_to_dpg_flat(tile.unsqueeze(0)) for tile in tiles]


class ThumbnailPipeline:
    """Triple-buffered thumbnail generation pipeline.

    Decode and GPU resize run in separate background threads, overlapping
    CPU decode of image N+1 with GPU resize of image N.

    Usage::

        pipeline = ThumbnailPipeline(device, dtype, tile_size=128)
        pipeline.start(image_paths)

        # In the render loop:
        for idx, flat_rgba in pipeline.poll():
            dpg.set_value(texture_ids[idx], flat_rgba)

        # On folder change or shutdown:
        pipeline.shutdown()
    """

    def __init__(self,
                 device: torch.device,
                 dtype: torch.dtype,
                 tile_size: int = 128,
                 lanczos_order: int = lanczos.DEFAULT_ORDER):
        self._device = device
        self._dtype = dtype
        self._tile_size = tile_size
        self._order = lanczos_order

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="thumbnail")
        self._task_mgr = bgtask.TaskManager("thumbnails", "concurrent", self._executor)

        # Inter-thread queues.  A small decode queue keeps the GPU fed even when
        # individual image decode times vary.  At small tile sizes the GPU is much
        # faster than decode, so a deeper queue prevents GPU idle stalls.
        self._decode_queue: queue.Queue = queue.Queue(maxsize=4)
        self._result_queue: queue.Queue = queue.Queue()

        self._total: int = 0
        self._completed: int = 0

    def set_tile_size(self, tile_size: int) -> None:
        """Change the output tile size, cancelling any batch in progress.

        Cancelling is the point: whatever is in flight is being resized to the old size, and a thumbnail of
        the wrong size is not merely late — the consumer has to recognize and discard it. Restart the batch
        after this if the images are still wanted.
        """
        if tile_size == self._tile_size:
            return
        self.cancel()
        self._tile_size = tile_size

    @property
    def tile_size(self) -> int:
        """Edge of the square tiles this pipeline produces, in pixels."""
        return self._tile_size

    @property
    def total(self) -> int:
        """Total number of images in the current batch."""
        return self._total

    @property
    def completed(self) -> int:
        """Number of thumbnails completed so far."""
        return self._completed

    @property
    def in_progress(self) -> bool:
        return self._task_mgr.has_tasks()

    def start(self, paths: list[pathlib.Path]) -> None:
        """Start generating thumbnails for the given image paths.

        Cancels any in-progress generation first.
        """
        # Cancel previous run.
        self._task_mgr.clear(wait=True)

        # Fresh queues (old ones may have stale items).
        self._decode_queue = queue.Queue(maxsize=1)
        self._result_queue = queue.Queue()
        self._total = len(paths)
        self._completed = 0

        if not paths:
            return

        # Submit decode thread.
        decode_e = env(paths=paths,
                       decode_queue=self._decode_queue,
                       max_size=self._tile_size * 2)  # hint for scaled JPEG decode
        self._task_mgr.submit(self._decode_loop, decode_e)

        # Submit GPU thread.
        gpu_e = env(decode_queue=self._decode_queue,
                    result_queue=self._result_queue,
                    device=self._device,
                    tile_size=self._tile_size,
                    order=self._order)
        self._task_mgr.submit(self._gpu_loop, gpu_e)

    def poll(self) -> list[tuple[int, np.ndarray]]:
        """Non-blocking: return any newly completed thumbnails.

        Returns a list of ``(index, flat_rgba_float32)`` tuples.
        Call from the main thread each frame.
        """
        results = []
        while True:
            try:
                item = self._result_queue.get_nowait()
            except queue.Empty:
                break
            results.append(item)
            self._completed += 1
        return results

    def cancel(self) -> None:
        """Cancel in-progress generation."""
        self._task_mgr.clear()

    def shutdown(self) -> None:
        """Cancel all work and shut down the thread pool."""
        self._task_mgr.clear(wait=True)
        self._executor.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_loop(e: env) -> None:
        """Decode thread: read and decode images, feed the decode queue."""
        for i, path in enumerate(e.paths):
            if e.cancelled:
                break
            try:
                arr = imageutils.ensure_rgba(imagecodec.decode(path, max_size=e.max_size))
            except Exception as exc:
                logger.warning(f"ThumbnailPipeline._decode_loop: instance {e.task_name}: "
                               f"failed to decode {path.name}: {exc}")
                continue

            # Put with timeout so we can check cancellation periodically.
            while not e.cancelled:
                try:
                    e.decode_queue.put((i, arr), timeout=_QUEUE_TIMEOUT_S)
                    break
                except queue.Full:
                    continue

        # Sentinel: tell GPU thread we're done.
        if not e.cancelled:
            while not e.cancelled:
                try:
                    e.decode_queue.put(None, timeout=_QUEUE_TIMEOUT_S)
                    break
                except queue.Full:
                    continue

    @staticmethod
    def _gpu_loop(e: env) -> None:
        """GPU thread: consume decoded arrays, resize, produce DPG-ready results."""
        while not e.cancelled:
            try:
                item = e.decode_queue.get(timeout=_QUEUE_TIMEOUT_S)
            except queue.Empty:
                continue

            if item is None:  # sentinel from decode thread
                break

            idx, arr = item
            try:
                tensor = imageutils.np_to_tensor(arr, e.device)
                thumbnail = imageutils.letterbox(tensor, e.tile_size, e.order)
                flat = imageutils.tensor_to_dpg_flat(thumbnail)
                del tensor, thumbnail  # free GPU memory promptly
                e.result_queue.put((idx, flat))
            except Exception as exc:
                logger.warning(f"ThumbnailPipeline._gpu_loop: instance {e.task_name}: "
                               f"failed to resize index {idx}: {exc}")
