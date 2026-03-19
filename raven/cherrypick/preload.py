"""Preload cache for raven-cherrypick.

Pre-loads mip textures for images near the current tile in the grid,
so that switching images is instant (no progressive mip loading visible).

The cache owns DPG textures for preloaded images. When the user navigates
to a cached image, `take` transfers texture ownership to `ImageView`.
Eviction deletes DPG textures to free VRAM.
"""

__all__ = ["PreloadCache"]

import concurrent.futures
import logging
import threading
import time

import dearpygui.dearpygui as dpg
import torch

from unpythonic.env import env

from ..common.bgtask import TaskManager
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
        return f"preload_{prefix}_{_tag_counter}"


class _CacheEntry:
    """One preloaded image's mip textures."""
    __slots__ = ("idx", "img_w", "img_h", "mips", "tex_sizes", "vram_bytes")

    def __init__(self, idx, img_w, img_h, mips, tex_sizes, vram_bytes):
        self.idx = idx
        self.img_w = img_w
        self.img_h = img_h
        self.mips = mips              # list[(scale, tex_tag)], largest-first
        self.tex_sizes = tex_sizes    # {tex_tag: (w, h)}
        self.vram_bytes = vram_bytes  # sum of w*h*16 for all mip textures


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
    """Mip-texture cache for nearby images.

    Create once at startup.  Call `schedule` after each navigation
    (once the current image's mips have finished loading) to preload
    neighbors.  Call `take` in the image-load path to check for a
    cache hit before falling back to decode + mip generation.
    """

    def __init__(self, device: torch.device,
                 lanczos_order: int = lanczos.DEFAULT_ORDER,
                 mip_min_size: int = config.MIP_MIN_SIZE,
                 vram_budget_mb: int = config.PRELOAD_VRAM_BUDGET_MB,
                 window: int = config.PRELOAD_WINDOW,
                 debug: bool = False):
        self._device = device
        self._order = lanczos_order
        self._mip_min_size = mip_min_size
        self._vram_budget = vram_budget_mb * 1024 * 1024  # bytes
        self._window = window
        self._debug = debug

        # Cache: idx → _CacheEntry.  Protected by _lock.
        self._cache: dict[int, _CacheEntry] = {}
        self._loading: set[int] = set()  # indices with in-progress tasks
        self._lock = threading.Lock()
        self._vram_used = 0  # bytes

        # Background worker.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="cherrypick_preload")
        self._task_mgr = TaskManager("preload", "concurrent", self._executor)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def take(self, idx: int):
        """Remove and return a cache entry, or ``None``.

        Transfers texture ownership to the caller (typically ImageView).
        """
        with self._lock:
            entry = self._cache.pop(idx, None)
            if entry is not None:
                self._vram_used -= entry.vram_bytes
                if self._debug:
                    logger.info(f"PreloadCache.take: hit idx={idx} "
                                f"vram={entry.vram_bytes / 1e6:.0f}MB "
                                f"remaining={self._vram_used / 1e6:.0f}MB")
            return entry

    def donate(self, idx, mips, tex_sizes, img_w, img_h) -> None:
        """Insert already-loaded mip textures into the cache.

        Called when the user navigates away from a directly-loaded image
        and the switch is instant (preload hit).  The outgoing image's
        textures are donated rather than released to ImageView's pool,
        so navigating back is also instant.
        """
        vram_bytes = sum(w * h * 16 for w, h in tex_sizes.values())
        entry = _CacheEntry(idx=idx, img_w=img_w, img_h=img_h,
                            mips=mips, tex_sizes=tex_sizes,
                            vram_bytes=vram_bytes)
        with self._lock:
            # If already cached (e.g. preloader finished while we were displaying),
            # delete the duplicate's textures.
            if idx in self._cache:
                self._evict(idx)
            self._cache[idx] = entry
            self._vram_used += vram_bytes
        if self._debug:
            logger.info(f"PreloadCache.donate: idx={idx} "
                        f"vram={vram_bytes / 1e6:.0f}MB "
                        f"total={self._vram_used / 1e6:.0f}MB")

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
                while (self._vram_used >= self._vram_budget
                       and evict_candidates):
                    self._evict(evict_candidates.pop())
                if self._vram_used >= self._vram_budget:
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
                        f"vram={self._vram_used / 1e6:.0f}MB")

    def cancel_pending(self) -> None:
        """Cancel all in-progress preload tasks (free GPU for current image)."""
        self._task_mgr.clear()
        with self._lock:
            self._loading.clear()

    def shutdown(self) -> None:
        """Cancel all work, delete all textures, shut down executor.

        Does not wait for tasks — they may be blocked on ``split_frame``
        with no render loop running.  DPG context destruction will clean
        up any leaked textures.
        """
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
                # Clean up any textures created before cancellation.
                if hasattr(e, "result") and e.result is not None:
                    self._delete_entry_textures(e.result)
                return
            if not hasattr(e, "result") or e.result is None:
                return  # task failed
            entry = e.result
            # Check budget — might have been exceeded while we were loading.
            if self._vram_used + entry.vram_bytes > self._vram_budget:
                self._delete_entry_textures(entry)
                if self._debug:
                    logger.info(f"PreloadCache._on_task_done: idx={entry.idx} "
                                f"dropped (budget exceeded)")
                return
            self._cache[entry.idx] = entry
            self._vram_used += entry.vram_bytes
            if self._debug:
                logger.info(f"PreloadCache._on_task_done: idx={entry.idx} "
                            f"cached, vram={self._vram_used / 1e6:.0f}MB")

    def _evict(self, idx: int) -> None:
        """Remove an entry and delete its DPG textures.  Caller holds _lock."""
        entry = self._cache.pop(idx, None)
        if entry is not None:
            self._delete_entry_textures(entry)
            self._vram_used -= entry.vram_bytes

    def _delete_entry_textures(self, entry: _CacheEntry) -> None:
        """Delete all DPG textures in a cache entry."""
        for _scale, tex_tag in entry.mips:
            try:
                dpg.delete_item(tex_tag)
            except Exception:
                pass


def _preload_one(e: env) -> None:
    """Background task: decode, mip-chain, create DPG textures for one image."""
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

    # Build (scale, tensor) list, largest-first.
    scale = 1.0
    mip_data = []
    for mip_t in mip_tensors:
        mip_data.append((scale, mip_t))
        scale *= 0.5

    # Create DPG textures, smallest first.
    mips = []
    tex_sizes = {}
    vram_bytes = 0

    for mip_scale, mip_t in reversed(mip_data):
        if e.cancelled:
            # Clean up partially created textures.
            for _s, tag in mips:
                try:
                    dpg.delete_item(tag)
                except Exception:
                    pass
            del tensor, mip_tensors
            return

        _, _, mh, mw = mip_t.shape
        flat = imageutils.tensor_to_dpg_flat(mip_t)

        if e.cancelled:
            break
        tex_tag = _next_tag("mip")
        with dpg.texture_registry():
            dpg.add_dynamic_texture(mw, mh,
                                    default_value=flat,
                                    tag=tex_tag)
        if e.cancelled:
            break
        dpg.split_frame()

        mips.append((mip_scale, tex_tag))
        tex_sizes[tex_tag] = (mw, mh)
        vram_bytes += mw * mh * 16  # RGBA float32

    # Sort largest-first (ImageView expects this order).
    mips.sort(key=lambda x: x[0], reverse=True)

    del tensor, mip_tensors
    if e.device.type == "cuda":
        torch.cuda.empty_cache()

    if not e.cancelled:
        e.result = _CacheEntry(idx=e.idx, img_w=img_w, img_h=img_h,
                               mips=mips, tex_sizes=tex_sizes,
                               vram_bytes=vram_bytes)

    t_total = (time.perf_counter_ns() - t0) / 1e6
    if e.debug:
        logger.info(f"PreloadCache._preload_one: idx={e.idx} "
                    f"{img_w}x{img_h} "
                    f"vram={vram_bytes / 1e6:.0f}MB "
                    f"total={t_total:.0f}ms")
