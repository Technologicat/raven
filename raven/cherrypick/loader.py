"""Thumbnail generation pipeline for raven-cherrypick.

Triple-buffered with two background threads:

  - **Decode thread** (CPU): reads and decodes images to numpy RGBA arrays.
  - **GPU thread**: transfers decoded arrays to the GPU, Lanczos-resizes to
    tile size (with imageutils.letterboxing for non-square images), transfers back.

The main thread polls for completed thumbnails and creates DPG textures.

The pipeline is managed via ``raven.common.bgtask.bgtask.TaskManager`` for
cooperative cancellation when the user opens a new folder.
"""

__all__ = ["ThumbnailPipeline"]

import concurrent.futures
import logging
import pathlib
import queue

import numpy as np
import torch

from unpythonic.env import env

from ..common import bgtask
from ..common.image import codec as imagecodec
from ..common.image import lanczos
from ..common.image import utils as imageutils

logger = logging.getLogger(__name__)

# How long each thread waits on a queue before re-checking the cancellation flag.
_QUEUE_TIMEOUT_S = 0.5


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
            max_workers=2, thread_name_prefix="cherrypick_thumb")
        self._task_mgr = bgtask.TaskManager("thumbnails", "concurrent", self._executor)

        # Inter-thread queues.  A small decode queue keeps the GPU fed even when
        # individual image decode times vary.  At small tile sizes the GPU is much
        # faster than decode, so a deeper queue prevents GPU idle stalls.
        self._decode_queue: queue.Queue = queue.Queue(maxsize=4)
        self._result_queue: queue.Queue = queue.Queue()

        self._total: int = 0
        self._completed: int = 0

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
