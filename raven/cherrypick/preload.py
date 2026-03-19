"""Preload cache for raven-cherrypick.

Pre-loads mip data for images near the current tile in the grid,
so that switching images is instant (no progressive mip loading visible).

Cached entries store pre-computed flat numpy arrays (ready for DPG texture
creation), NOT live DPG textures. This avoids per-frame overhead in DPG's
render loop, which scales O(n) with registered dynamic textures even when
they are not drawn. DPG textures are created only on ``take()``, by the
caller (ImageView), using its texture pool for fast ``set_value`` reuse.
"""

__all__ = ["PreloadCache"]

import concurrent.futures
import logging
import threading
import time

import torch

from unpythonic.env import env

from ..common.bgtask import TaskManager
from ..common.image import lanczos
from ..common.image import utils as imageutils
from . import config

logger = logging.getLogger(__name__)


class _CacheEntry:
    """One preloaded image's mip data (flat numpy arrays, no DPG textures).

    Storing flat arrays instead of DPG textures avoids per-frame overhead
    in ``render_dearpygui_frame()`` — DPG has O(n) cost per registered
    dynamic texture, even if they're not drawn.
    """
    __slots__ = ("idx", "img_w", "img_h", "mips", "ram_bytes")

    def __init__(self, idx, img_w, img_h, mips, ram_bytes):
        self.idx = idx
        self.img_w = img_w
        self.img_h = img_h
        self.mips = mips          # list[(scale, w, h, flat_array)], largest-first
        self.ram_bytes = ram_bytes  # sum of flat array nbytes


def _compute_targets(vis_pos, n_visible, n_cols, window):
    """Return visible-list positions in the ±window cross neighborhood.

    Plus-shaped: ±window along the row (horizontal) and ±window along
    the column (vertical).  Sorted by distance from center (nearest first).
    """
    row = vis_pos // n_cols
    col = vis_pos % n_cols
    targets = []
    # Horizontal: same row, ±window columns.
    for dc in range(-window, window + 1):
        if dc == 0:
            continue
        c = col + dc
        if 0 <= c < n_cols:
            p = row * n_cols + c
            if 0 <= p < n_visible:
                targets.append((abs(dc), p))
    # Vertical: same column, ±window rows.
    for dr in range(-window, window + 1):
        if dr == 0:
            continue
        p = (row + dr) * n_cols + col
        if 0 <= p < n_visible:
            targets.append((abs(dr), p))
    targets.sort()  # by distance
    return [p for _, p in targets]


class PreloadCache:
    """Mip cache for nearby images, stored as flat numpy arrays.

    Create once at startup. Call `schedule` after each navigation
    (once the current image's mips have finished loading) to preload
    neighbors. Call `take` in the image-load path to check for a
    cache hit before falling back to decode + mip generation.

    Cached entries hold pre-computed flat arrays (ready for
    ``dpg.set_value`` or ``dpg.add_dynamic_texture``), NOT live DPG
    textures. This avoids per-frame overhead in DPG's render loop
    (which scales O(n) with registered dynamic textures).
    """

    def __init__(self, device: torch.device,
                 lanczos_order: int = lanczos.DEFAULT_ORDER,
                 mip_min_size: int = config.MIP_MIN_SIZE,
                 ram_budget_mb: int = config.PRELOAD_VRAM_BUDGET_MB,
                 window: int = config.PRELOAD_WINDOW,
                 debug: bool = False):
        self._device = device
        self._order = lanczos_order
        self._mip_min_size = mip_min_size
        self._ram_budget = ram_budget_mb * 1024 * 1024  # bytes
        self._window = window
        self._debug = debug

        # Cache: idx → _CacheEntry. Protected by _lock.
        self._cache: dict[int, _CacheEntry] = {}
        self._loading: set[int] = set()  # indices with in-progress tasks
        self._lock = threading.Lock()
        self._ram_used = 0  # bytes

        # Background worker.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="cherrypick_preload")
        self._task_mgr = TaskManager("preload", "concurrent", self._executor)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def take(self, idx: int):
        """Remove and return a cache entry, or ``None``.

        The returned entry contains flat numpy arrays, not DPG textures.
        The caller is responsible for creating DPG textures from them.
        """
        with self._lock:
            entry = self._cache.pop(idx, None)
            if entry is not None:
                self._ram_used -= entry.ram_bytes
                if self._debug:
                    logger.info(f"PreloadCache.take: hit idx={idx} "
                                f"ram={entry.ram_bytes / 1e6:.0f}MB "
                                f"remaining={self._ram_used / 1e6:.0f}MB")
            return entry

    def donate(self, idx, mips, img_w, img_h) -> None:
        """Insert pre-computed mip arrays into the cache.

        *mips*: list of ``(scale, w, h, flat_array)``, largest-first.

        Called when the user navigates away from a fully-loaded image
        and the switch is instant (preload hit). The outgoing image's
        mip data is donated so navigating back is also instant.
        """
        ram_bytes = sum(flat.nbytes for _s, _w, _h, flat in mips)
        entry = _CacheEntry(idx=idx, img_w=img_w, img_h=img_h,
                            mips=mips, ram_bytes=ram_bytes)
        with self._lock:
            if idx in self._cache:
                self._evict(idx)
            self._cache[idx] = entry
            self._ram_used += ram_bytes
        if self._debug:
            logger.info(f"PreloadCache.donate: idx={idx} "
                        f"ram={ram_bytes / 1e6:.0f}MB "
                        f"total={self._ram_used / 1e6:.0f}MB")

    def schedule(self, current_idx, visible, n_cols, triage) -> None:
        """Recompute the preload window and start loading.

        *current_idx*: current image index (in the full list).
        *visible*: ``grid.visible`` — list of indices visible under the filter.
        *n_cols*: ``grid.n_cols`` — column count for 2D neighborhood.
        *triage*: ``TriageManager`` — for file paths.

        Evicts entries outside the new window, submits tasks for new targets.
        """
        if current_idx not in visible:
            return  # current is filtered out; don't preload

        vis_pos = visible.index(current_idx)
        target_positions = _compute_targets(vis_pos, len(visible), n_cols,
                                            self._window)
        target_indices = set(visible[p] for p in target_positions)

        # Cancel all pending tasks — window shifted.
        self._task_mgr.clear()

        with self._lock:
            # Clear stale loading markers.
            self._loading -= self._loading - target_indices

            # Make room for new targets if budget is tight.
            # Evict furthest-from-current first, but only entries
            # NOT in the new target set (keeps recently visited images
            # cached even if they've scrolled out of the cross neighborhood).
            evict_candidates = sorted(
                (idx for idx in self._cache if idx not in target_indices),
                key=lambda idx: abs(idx - current_idx),
                reverse=True,  # furthest first
            )

            # Submit tasks for targets not yet cached or loading.
            for idx in (visible[p] for p in target_positions):
                if idx in self._cache or idx in self._loading:
                    continue
                # Evict to make room if needed.
                while (self._ram_used >= self._ram_budget
                       and evict_candidates):
                    self._evict(evict_candidates.pop())
                if self._ram_used >= self._ram_budget:
                    break  # budget exhausted, nothing left to evict
                self._loading.add(idx)
                path = triage[idx].path
                task_env = env(idx=idx, path=path,
                               device=self._device,
                               order=self._order,
                               mip_min_size=self._mip_min_size,
                               debug=self._debug,
                               done_callback=self._on_task_done)
                self._task_mgr.submit(_preload_one, task_env)

        if self._debug:
            cached = set(self._cache.keys())
            logger.info(f"PreloadCache.schedule: current={current_idx} "
                        f"targets={sorted(target_indices)} "
                        f"cached={sorted(cached)} "
                        f"loading={sorted(self._loading)} "
                        f"ram={self._ram_used / 1e6:.0f}MB")

    def cancel_pending(self) -> None:
        """Cancel all in-progress preload tasks (free GPU for current image)."""
        self._task_mgr.clear()
        with self._lock:
            self._loading.clear()

    def shutdown(self) -> None:
        """Cancel all work, shut down executor."""
        self._task_mgr.clear(wait=True)
        self._executor.shutdown(wait=True)
        with self._lock:
            for idx in list(self._cache):
                self._evict(idx)
            self._loading.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_task_done(self, e: env) -> None:
        """Callback when a preload task completes (runs in background thread)."""
        with self._lock:
            self._loading.discard(e.idx)
            if e.cancelled:
                return  # no DPG textures to clean up
            if not hasattr(e, "result") or e.result is None:
                return  # task failed
            entry = e.result
            # Check budget — might have been exceeded while we were loading.
            if self._ram_used + entry.ram_bytes > self._ram_budget:
                if self._debug:
                    logger.info(f"PreloadCache._on_task_done: idx={entry.idx} "
                                f"dropped (budget exceeded)")
                return
            self._cache[entry.idx] = entry
            self._ram_used += entry.ram_bytes
            if self._debug:
                logger.info(f"PreloadCache._on_task_done: idx={entry.idx} "
                            f"cached, ram={self._ram_used / 1e6:.0f}MB")

    def _evict(self, idx: int) -> None:
        """Remove an entry. Caller holds _lock."""
        entry = self._cache.pop(idx, None)
        if entry is not None:
            self._ram_used -= entry.ram_bytes


def _preload_one(e: env) -> None:
    """Background task: decode, generate mip chain, store as flat arrays."""
    t0 = time.perf_counter_ns()

    try:
        rgba = imageutils.decode_image(e.path)
    except Exception as exc:
        logger.warning(f"PreloadCache._preload_one: failed to decode "
                       f"{e.path}: {exc}")
        return
    if e.cancelled:
        return

    img_h, img_w = rgba.shape[:2]

    # Upload to GPU.
    tensor = imageutils.np_to_tensor(rgba, e.device)
    del rgba
    if e.device.type == "cuda":
        torch.cuda.synchronize(e.device)
    if e.cancelled:
        return

    # Generate mip chain.
    mip_tensors = lanczos.lanczos_mipchain(tensor, min_size=e.mip_min_size,
                                            order=e.order)
    if e.device.type == "cuda":
        torch.cuda.synchronize(e.device)

    # Convert to flat DPG-format arrays (GPU → CPU).
    # Build (scale, w, h, flat_array) list, largest-first.
    scale = 1.0
    mips = []
    ram_bytes = 0
    for mip_t in mip_tensors:
        if e.cancelled:
            break
        _, _, mh, mw = mip_t.shape
        flat = imageutils.tensor_to_dpg_flat(mip_t)
        mips.append((scale, mw, mh, flat))
        ram_bytes += flat.nbytes
        scale *= 0.5

    del tensor, mip_tensors
    if e.device.type == "cuda":
        torch.cuda.empty_cache()

    if e.cancelled:
        return

    e.result = _CacheEntry(idx=e.idx, img_w=img_w, img_h=img_h,
                           mips=mips, ram_bytes=ram_bytes)

    t_total = (time.perf_counter_ns() - t0) / 1e6
    if e.debug:
        logger.info(f"PreloadCache._preload_one: idx={e.idx} "
                    f"{img_w}x{img_h} "
                    f"ram={ram_bytes / 1e6:.0f}MB "
                    f"total={t_total:.0f}ms")
