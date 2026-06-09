"""Preload cache for raven-cherrypick.

Pre-loads mip data for images near the current tile in the grid,
so that switching images is instant (no progressive mip loading visible).

Cached entries store pre-computed flat numpy arrays (ready for DPG texture
creation), NOT live DPG textures. This avoids per-frame overhead in DPG's
render loop, which scales O(n) with registered dynamic textures even when
they are not drawn. DPG textures are created only on ``take()``, by the
caller (ImageView), using its texture pool for fast ``set_value`` reuse.
"""

__all__ = ["PreloadCache", "mip_scale_for_zoom"]

import concurrent.futures
import logging
import math
import threading
import time
from collections.abc import Sequence

import torch

from unpythonic.env import env

from ..common.bgtask import TaskManager
from ..common.image import codec as imagecodec
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
    """Return visible-list positions in the ±window neighborhood, nearest-first.

    Two arms, each matching one pair of navigation keys:

      - Horizontal (Left/Right = ±1 in the visible list): linear positions
        ``vis_pos ± 1 .. ± window``.  Linear — *not* clipped to the current
        grid row — because Left/Right wrap across row boundaries: the image at
        the right edge of a row steps to the first image of the next row, and
        the left edge to the last of the previous row.  Clipping the arm to the
        row would leave those wrap targets uncached, so every edge-tile step
        would miss.
      - Vertical (Up/Down = ±n_cols): same column,
        ``vis_pos ± n_cols .. ± window*n_cols``.

    Deduplicated (the two arms coincide when ``n_cols == 1``) and sorted by
    distance from ``vis_pos`` (nearest first), so the caller loads the most
    likely next image soonest.
    """
    nearest: dict[int, int] = {}  # position -> smallest distance seen

    def offer(p, dist):
        if 0 <= p < n_visible and p != vis_pos and dist < nearest.get(p, dist + 1):
            nearest[p] = dist

    for d in range(1, window + 1):
        offer(vis_pos - d, d)              # horizontal (linear), Left
        offer(vis_pos + d, d)              # horizontal (linear), Right
        offer(vis_pos - d * n_cols, d)     # vertical (same column), Up
        offer(vis_pos + d * n_cols, d)     # vertical (same column), Down

    return [p for p, _d in sorted(nearest.items(), key=lambda kv: (kv[1], kv[0]))]


def mip_scale_for_zoom(zoom: float) -> float:
    """Smallest mip scale (1.0, 0.5, 0.25, …) that displays crisply at *zoom*.

    A mip chain halves from 1.0, and the mip engine picks the smallest mip whose
    scale is ``>= zoom`` so it downsamples a larger level rather than upsampling
    a smaller one (see ``ImageView._select_mip_from``). The matching preload cap
    is therefore ``2 ** ceil(log2(zoom))``, clamped to ``<= 1.0`` — no mip
    exceeds native, so past 1:1 (reachable with the zoom-fit 100% cap off) this
    returns 1.0 and the *view* magnifies the native level.

    This is the adaptive half of the preload cap: a 1 MP image at fit-zoom
    (~0.8) needs the 1.0 level, while a multi-MP photo at fit-zoom (~0.2) needs
    only 0.25 — so we never read back levels finer than the pane can show.
    Degenerate / unknown zooms (``<= 0``) fall back to full res.
    """
    if zoom >= 1.0 or zoom <= 0.0:
        return 1.0
    return min(1.0, 2.0 ** math.ceil(math.log2(zoom)))


class PreloadCache:
    """Mip cache for nearby images, stored as flat numpy arrays.

    Create once at startup. Call `schedule_neighbors` after each navigation
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
                 ram_budget_mb: int = config.PRELOAD_RAM_BUDGET_FALLBACK_MB,
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
        self._pinned: set[int] = set()  # compare mode: protect from eviction
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

    def schedule_neighbors(self, current_idx, visible, n_cols, triage,
                           display_scale: float = 1.0) -> None:
        """Recompute the preload window and start loading neighbors.

        *current_idx*: current image index (in the full list).
        *visible*: ``grid.visible`` — list of indices visible under the filter.
        *n_cols*: ``grid.n_cols`` — column count for 2D neighborhood.
        *triage*: ``TriageManager`` — for file paths.
        *display_scale*: current image's display zoom (``ImageView.zoom``).
        Neighbors are preloaded only up to the mip needed to display crisply at
        this zoom, so multi-MP photos browsed at a small fit-zoom don't read back
        full-res levels the pane can't show (GPU→host readback is the slow path).

        Evicts entries outside the new window (except pinned), submits
        tasks for new targets.
        """
        if current_idx not in visible:
            return  # current is filtered out; don't preload

        max_scale = mip_scale_for_zoom(display_scale)

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
            evict_candidates = self._evict_candidates(
                current_idx, exclude=target_indices)

            # Submit tasks for targets not yet cached or loading.
            for idx in (visible[p] for p in target_positions):
                if idx in self._cache or idx in self._loading:
                    continue
                # Evict to make room if needed.
                while self._ram_used >= self._ram_budget and evict_candidates:
                    self._evict(evict_candidates.pop())
                if self._ram_used >= self._ram_budget:
                    break  # budget exhausted, nothing left to evict
                self._loading.add(idx)
                path = triage[idx].path
                task_env = env(idx=idx, path=path,
                               device=self._device,
                               order=self._order,
                               mip_min_size=self._mip_min_size,
                               max_scale=max_scale,
                               debug=self._debug,
                               done_callback=self._on_task_done)
                self._task_mgr.submit(_preload_one, task_env)

        if self._debug:
            cached = set(self._cache.keys())
            logger.info(f"PreloadCache.schedule_neighbors: current={current_idx} "
                        f"display_scale={display_scale:.3f} max_scale={max_scale} "
                        f"targets={sorted(target_indices)} "
                        f"cached={sorted(cached)} "
                        f"loading={sorted(self._loading)} "
                        f"ram={self._ram_used / 1e6:.0f}MB")

    def clear(self) -> None:
        """Cancel all tasks and flush the cache.

        Call when opening a new folder — stale entries keyed by index
        from the old folder must not be served for the new one.
        """
        self._task_mgr.clear()
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        with self._lock:
            self._cache.clear()
            self._loading.clear()
            self._ram_used = 0

    def cancel_pending(self) -> None:
        """Cancel all in-progress preload tasks (free GPU for current image).

        Also flushes the CUDA pipeline — cancelled tasks may have queued
        GPU kernels that would otherwise contend with the display path.
        """
        self._task_mgr.clear()
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        with self._lock:
            self._loading.clear()

    # ------------------------------------------------------------------
    # Compare mode support
    # ------------------------------------------------------------------

    def schedule_compare(self, indices: Sequence[int], triage) -> None:
        """Pre-cache a specific set of image indices for compare mode.

        Cancels pending neighbor-preload tasks to free GPU bandwidth,
        pins the requested indices against eviction, and submits tasks
        for any indices not already cached.

        Uses full mip chain (no ``max_scale`` cap) — compare mode
        preserves whatever zoom level was active when it started.
        """
        self._task_mgr.clear()
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

        with self._lock:
            self._loading.clear()
            self._pinned = set(indices)

            for idx in indices:
                if idx in self._loading:
                    continue
                # Evict entries with capped mips — compare mode needs
                # the full chain for quality display at any zoom.
                if idx in self._cache:
                    entry = self._cache[idx]
                    has_fullres = any(s >= 1.0 for s, _w, _h, _f in entry.mips)
                    if has_fullres:
                        continue
                    self._evict(idx)
                self._loading.add(idx)
                path = triage[idx].path
                task_env = env(idx=idx, path=path,
                               device=self._device,
                               order=self._order,
                               mip_min_size=self._mip_min_size,
                               max_scale=1.0,  # full mip chain for compare
                               debug=self._debug,
                               done_callback=self._on_task_done)
                self._task_mgr.submit(_preload_one, task_env)

        if self._debug:
            cached = set(self._cache.keys())
            logger.info(f"PreloadCache.schedule_compare: "
                        f"indices={sorted(indices)} "
                        f"cached={sorted(cached & set(indices))} "
                        f"loading={sorted(self._loading & set(indices))}")

    def unpin_all(self) -> None:
        """Clear all pinned entries. Call when compare mode exits."""
        with self._lock:
            self._pinned.clear()

    def is_cached(self, idx: int) -> bool:
        """Check if an index is in the cache."""
        with self._lock:
            return idx in self._cache

    def compare_progress(self, indices: Sequence[int]) -> tuple[int, int]:
        """Return ``(n_cached, n_total)`` for compare mode warm-up display."""
        with self._lock:
            n_cached = sum(1 for idx in indices if idx in self._cache)
        return (n_cached, len(indices))

    # ------------------------------------------------------------------

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

    def _evict_candidates(self, center_idx: int,
                          exclude: set[int] | None = None) -> list[int]:
        """Build eviction candidate list, sorted nearest-first.

        Caller holds ``_lock``. Returns indices sorted by distance from
        *center_idx* (nearest first), so ``pop()`` yields the furthest.
        Pinned entries and *exclude* set are omitted.
        """
        skip = self._pinned
        if exclude:
            skip = skip | exclude
        return sorted(
            (idx for idx in self._cache
             if idx not in skip and idx != center_idx),
            key=lambda idx: abs(idx - center_idx),
        )

    def _evict_until_fits(self, needed: int, candidates: list[int]) -> None:
        """Evict from *candidates* (furthest first) until budget allows *needed* bytes.

        Caller holds ``_lock``.
        """
        while self._ram_used + needed > self._ram_budget and candidates:
            self._evict(candidates.pop())

    def _on_task_done(self, e: env) -> None:
        """Callback when a preload task completes (runs in background thread)."""
        with self._lock:
            self._loading.discard(e.idx)
            if e.cancelled:
                return  # nothing to clean up
            if not hasattr(e, "result") or e.result is None:
                return  # task failed
            entry = e.result
            # Evict distant entries to make room. A fresh neighbor is more
            # valuable than a stale entry from a previous browsing position.
            if self._ram_used + entry.ram_bytes > self._ram_budget:
                evict_candidates = self._evict_candidates(entry.idx)
                self._evict_until_fits(entry.ram_bytes, evict_candidates)
            if self._ram_used + entry.ram_bytes > self._ram_budget:
                if self._debug:
                    logger.info(f"PreloadCache._on_task_done: idx={entry.idx} "
                                f"dropped (budget exceeded, nothing to evict)")
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
        rgba = imageutils.ensure_rgba(imagecodec.decode(e.path))
    except Exception as exc:
        logger.warning(f"PreloadCache._preload_one: instance {e.task_name}: failed to decode "
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
    mip_tensors = lanczos.mipchain(tensor,
                                   min_size=e.mip_min_size,
                                   order=e.order)
    if e.device.type == "cuda":
        torch.cuda.synchronize(e.device)

    # Convert to flat DPG-format arrays (GPU → CPU).
    # Build (scale, w, h, flat_array) list, largest-first.
    # Skip mip levels above max_scale — full-res is generated on demand
    # when the image is actually displayed (see ImageView.augment_fullres).
    scale = 1.0
    mips = []
    ram_bytes = 0
    for mip_t in mip_tensors:
        if e.cancelled:
            break
        if scale > e.max_scale:
            scale *= 0.5
            continue
        _, _, mh, mw = mip_t.shape
        flat = imageutils.tensor_to_dpg_flat(mip_t)
        mips.append((scale, mw, mh, flat))
        ram_bytes += flat.nbytes
        scale *= 0.5

    del tensor, mip_tensors
    # Don't call empty_cache() — keeps the allocator's block cache warm
    # for the next image. See imageview.py comment for details.

    if e.cancelled:
        return

    e.result = _CacheEntry(idx=e.idx, img_w=img_w, img_h=img_h,
                           mips=mips, ram_bytes=ram_bytes)

    t_total = (time.perf_counter_ns() - t0) / 1e6
    if e.debug:
        logger.info(f"PreloadCache._preload_one: instance {e.task_name}: idx={e.idx} "
                    f"{img_w}x{img_h} "
                    f"ram={ram_bytes / 1e6:.0f}MB "
                    f"total={t_total:.0f}ms")
