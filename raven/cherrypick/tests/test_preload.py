"""Tests for raven.cherrypick.preload — cache management and eviction logic.

Tests the computation layer only: target selection, RAM budget enforcement,
eviction ordering, pinning, take/donate, and cancellation. No DPG or GPU.
"""

import numpy as np
import torch

from unpythonic.env import env

from raven.cherrypick.preload import (
    PreloadCache,
    _CacheEntry,
    _compute_targets,
    mip_scale_for_zoom,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_entry(idx: int, ram_mb: float = 1.0) -> _CacheEntry:
    """Build a minimal cache entry with a known RAM footprint."""
    nbytes = int(ram_mb * 1024 * 1024)
    flat = np.zeros(nbytes // 4, dtype=np.float32)
    # Single mip at scale 1.0, dimensions don't matter for cache tests.
    mips = [(1.0, 64, 64, flat)]
    return _CacheEntry(idx=idx, img_w=64, img_h=64,
                       mips=mips, ram_bytes=flat.nbytes)


def _make_cache(ram_budget_mb: int = 100, **kwargs) -> PreloadCache:
    """Create a CPU-only PreloadCache for testing."""
    return PreloadCache(device=torch.device("cpu"),
                        ram_budget_mb=ram_budget_mb, **kwargs)


# ---------------------------------------------------------------------------
# _compute_targets
# ---------------------------------------------------------------------------

class TestComputeTargets:
    def test_center_of_single_row(self):
        # 10 items in a row with 10 columns → single row.
        # Center at position 5, window=2 → positions 3, 4, 6, 7.
        targets = _compute_targets(vis_pos=5, n_visible=10, n_cols=10, window=2)
        assert set(targets) == {3, 4, 6, 7}

    def test_sorted_by_distance(self):
        targets = _compute_targets(vis_pos=5, n_visible=10, n_cols=10, window=3)
        # Distance from center: 4→1, 6→1, 3→2, 7→2, 2→3, 8→3.
        distances = [abs(t - 5) for t in targets]
        assert distances == sorted(distances)

    def test_clamps_to_bounds(self):
        # Position 0, window=3 → only rightward/downward neighbors exist.
        targets = _compute_targets(vis_pos=0, n_visible=5, n_cols=5, window=3)
        assert all(0 <= t < 5 for t in targets)
        assert 0 not in targets  # center excluded

    def test_shape_with_grid(self):
        # 4×4 grid (16 items, 4 cols). Center at (1,1) = position 5, window=3.
        # Horizontal (linear, Left/Right): 5±1, 5±2, 5±3 → 2,3,4,6,7,8.
        # Vertical (same column, Up/Down): 5±4, 5±8(=−3 out, 13) → 1,9,13.
        targets = _compute_targets(vis_pos=5, n_visible=16, n_cols=4, window=3)
        assert set(targets) == {2, 3, 4, 6, 7, 8, 1, 9, 13}

    def test_includes_row_wrap_neighbors(self):
        # The linear horizontal arm must cover Left/Right's row-wrap targets:
        # otherwise stepping off a row edge always misses the cache.
        # 4×4 grid. Pos 3 is the right edge of row 0; Right steps to pos 4
        # (row 1, col 0). Pos 4 is the left edge of row 1; Left steps to pos 3.
        right_edge = _compute_targets(vis_pos=3, n_visible=16, n_cols=4, window=2)
        assert 4 in right_edge  # Right wraps to next row's first tile
        left_edge = _compute_targets(vis_pos=4, n_visible=16, n_cols=4, window=2)
        assert 3 in left_edge   # Left wraps to previous row's last tile

    def test_no_duplicates_single_column(self):
        # n_cols == 1: horizontal and vertical arms coincide. Must dedupe.
        targets = _compute_targets(vis_pos=5, n_visible=20, n_cols=1, window=3)
        assert len(targets) == len(set(targets))
        assert set(targets) == {2, 3, 4, 6, 7, 8}

    def test_empty_for_single_item(self):
        targets = _compute_targets(vis_pos=0, n_visible=1, n_cols=1, window=5)
        assert targets == []


# ---------------------------------------------------------------------------
# mip_scale_for_zoom
# ---------------------------------------------------------------------------

class TestMipScaleForZoom:
    def test_returns_smallest_mip_not_below_zoom(self):
        # Smallest mip scale (1.0, 0.5, 0.25, …) that is >= zoom.
        assert mip_scale_for_zoom(0.8) == 1.0   # needs the 1.0 level
        assert mip_scale_for_zoom(0.5) == 0.5   # exact boundary
        assert mip_scale_for_zoom(0.51) == 1.0  # just past 0.5 → 1.0
        assert mip_scale_for_zoom(0.25) == 0.25
        assert mip_scale_for_zoom(0.2) == 0.25  # multi-MP photo at small fit
        assert mip_scale_for_zoom(0.1) == 0.125

    def test_clamps_at_native(self):
        # No mip exceeds 1.0 — zoom past 1:1 still caps at native res.
        assert mip_scale_for_zoom(1.0) == 1.0
        assert mip_scale_for_zoom(2.5) == 1.0

    def test_degenerate_zoom_falls_back_to_full(self):
        assert mip_scale_for_zoom(0.0) == 1.0
        assert mip_scale_for_zoom(-1.0) == 1.0

    def test_never_below_zoom(self):
        # Below 1:1 the chosen mip is >= zoom, so the mip engine downsamples a
        # larger level rather than upsampling a smaller one. (Past 1:1 it clamps
        # to native and the *view* magnifies — covered by test_clamps_at_native.)
        for zoom in (0.01, 0.13, 0.27, 0.49, 0.5, 0.51, 0.99):
            assert mip_scale_for_zoom(zoom) >= zoom


# ---------------------------------------------------------------------------
# Take / Donate
# ---------------------------------------------------------------------------

class TestTakeDonate:
    def test_take_hit(self):
        cache = _make_cache()
        entry = _fake_entry(idx=3, ram_mb=2.0)
        cache.donate(3, entry.mips, entry.img_w, entry.img_h)
        assert cache.is_cached(3)

        taken = cache.take(3)
        assert taken is not None
        assert taken.idx == 3
        assert not cache.is_cached(3)

    def test_take_miss(self):
        cache = _make_cache()
        assert cache.take(99) is None

    def test_take_updates_ram(self):
        cache = _make_cache()
        entry = _fake_entry(idx=0, ram_mb=5.0)
        cache.donate(0, entry.mips, entry.img_w, entry.img_h)
        assert cache._ram_used == entry.ram_bytes

        cache.take(0)
        assert cache._ram_used == 0

    def test_donate_replaces_existing(self):
        cache = _make_cache()
        e1 = _fake_entry(idx=0, ram_mb=1.0)
        e2 = _fake_entry(idx=0, ram_mb=3.0)
        cache.donate(0, e1.mips, e1.img_w, e1.img_h)
        cache.donate(0, e2.mips, e2.img_w, e2.img_h)

        assert cache.is_cached(0)
        # RAM should reflect only the replacement entry.
        assert cache._ram_used == e2.ram_bytes


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------

class TestEviction:
    def test_evict_candidates_sorted_nearest_first(self):
        cache = _make_cache()
        for i in [10, 20, 30, 40, 50]:
            e = _fake_entry(idx=i, ram_mb=1.0)
            cache.donate(i, e.mips, e.img_w, e.img_h)

        # Candidates relative to center=30: nearest first.
        # pop() yields furthest → that's correct eviction order.
        with cache._lock:
            candidates = cache._evict_candidates(center_idx=30)
        distances = [abs(c - 30) for c in candidates]
        assert distances == sorted(distances)

    def test_evict_candidates_excludes_pinned(self):
        cache = _make_cache()
        for i in range(5):
            e = _fake_entry(idx=i, ram_mb=1.0)
            cache.donate(i, e.mips, e.img_w, e.img_h)

        cache._pinned = {0, 4}
        with cache._lock:
            candidates = cache._evict_candidates(center_idx=2)
        assert 0 not in candidates
        assert 4 not in candidates
        assert 2 not in candidates  # center excluded

    def test_evict_candidates_excludes_set(self):
        cache = _make_cache()
        for i in range(5):
            e = _fake_entry(idx=i, ram_mb=1.0)
            cache.donate(i, e.mips, e.img_w, e.img_h)

        with cache._lock:
            candidates = cache._evict_candidates(center_idx=2, exclude={1, 3})
        assert 1 not in candidates
        assert 3 not in candidates

    def test_evict_until_fits(self):
        cache = _make_cache(ram_budget_mb=10)
        # Fill with 5 × 2 MB = 10 MB (exactly at budget).
        for i in range(5):
            e = _fake_entry(idx=i, ram_mb=2.0)
            cache.donate(i, e.mips, e.img_w, e.img_h)

        needed = int(3.0 * 1024 * 1024)  # 3 MB
        with cache._lock:
            candidates = cache._evict_candidates(center_idx=2)
            cache._evict_until_fits(needed, candidates)

        # Must have evicted at least 3 MB (2 entries) to fit.
        assert cache._ram_used + needed <= cache._ram_budget
        # Center never evicted.
        assert cache.is_cached(2)


# ---------------------------------------------------------------------------
# _on_task_done (callback)
# ---------------------------------------------------------------------------

class TestOnTaskDone:
    def test_inserts_on_success(self):
        cache = _make_cache(ram_budget_mb=10)
        cache._loading.add(7)

        entry = _fake_entry(idx=7, ram_mb=1.0)
        e = env(idx=7, cancelled=False, task_name="test")
        e.result = entry
        cache._on_task_done(e)

        assert cache.is_cached(7)
        assert 7 not in cache._loading

    def test_skips_on_cancel(self):
        cache = _make_cache()
        cache._loading.add(7)

        e = env(idx=7, cancelled=True, task_name="test")
        cache._on_task_done(e)

        assert not cache.is_cached(7)
        assert 7 not in cache._loading

    def test_skips_on_no_result(self):
        cache = _make_cache()
        cache._loading.add(7)

        e = env(idx=7, cancelled=False, task_name="test")
        # No e.result attribute → task failed.
        cache._on_task_done(e)

        assert not cache.is_cached(7)

    def test_evicts_to_fit_budget(self):
        cache = _make_cache(ram_budget_mb=5)
        # Fill with 4 × 1 MB.
        for i in range(4):
            e = _fake_entry(idx=i, ram_mb=1.0)
            cache.donate(i, e.mips, e.img_w, e.img_h)

        # Completing a 3 MB entry should evict some existing entries.
        cache._loading.add(10)
        new_entry = _fake_entry(idx=10, ram_mb=3.0)
        task = env(idx=10, cancelled=False, task_name="test")
        task.result = new_entry

        cache._on_task_done(task)
        assert cache.is_cached(10)
        assert cache._ram_used <= cache._ram_budget

    def test_drops_when_nothing_to_evict(self):
        cache = _make_cache(ram_budget_mb=2)
        # Fill with a 2 MB pinned entry — can't evict.
        e = _fake_entry(idx=0, ram_mb=2.0)
        cache.donate(0, e.mips, e.img_w, e.img_h)
        cache._pinned = {0}

        # 1 MB task completes but budget is full and only entry is pinned.
        cache._loading.add(5)
        new_entry = _fake_entry(idx=5, ram_mb=1.0)
        task = env(idx=5, cancelled=False, task_name="test")
        task.result = new_entry

        cache._on_task_done(task)
        assert not cache.is_cached(5)  # dropped, not inserted
        assert cache._ram_used == e.ram_bytes  # unchanged


# ---------------------------------------------------------------------------
# Clear / Cancel
# ---------------------------------------------------------------------------

class TestClearCancel:
    def test_clear_flushes_everything(self):
        cache = _make_cache()
        for i in range(5):
            e = _fake_entry(idx=i, ram_mb=1.0)
            cache.donate(i, e.mips, e.img_w, e.img_h)
        cache._loading.add(99)

        cache.clear()
        assert cache._ram_used == 0
        assert len(cache._cache) == 0
        assert len(cache._loading) == 0

    def test_cancel_pending_clears_loading(self):
        cache = _make_cache()
        # Some cached entries should survive cancellation.
        e = _fake_entry(idx=0, ram_mb=1.0)
        cache.donate(0, e.mips, e.img_w, e.img_h)
        cache._loading.update({1, 2, 3})

        cache.cancel_pending()
        assert len(cache._loading) == 0
        assert cache.is_cached(0)  # cached entries preserved


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

class TestCompareMode:
    def test_pin_protects_from_eviction(self):
        cache = _make_cache(ram_budget_mb=5)
        # Insert 3 × 1 MB, pin entry 1.
        for i in range(3):
            e = _fake_entry(idx=i, ram_mb=1.0)
            cache.donate(i, e.mips, e.img_w, e.img_h)
        cache._pinned = {1}

        # Evict candidates from center=1 — pinned entry excluded.
        with cache._lock:
            candidates = cache._evict_candidates(center_idx=1)
        assert 1 not in candidates

    def test_unpin_all(self):
        cache = _make_cache()
        cache._pinned = {0, 1, 2}
        cache.unpin_all()
        assert len(cache._pinned) == 0

    def test_compare_progress(self):
        cache = _make_cache()
        for i in [0, 2, 4]:
            e = _fake_entry(idx=i, ram_mb=1.0)
            cache.donate(i, e.mips, e.img_w, e.img_h)

        n_cached, n_total = cache.compare_progress([0, 1, 2, 3, 4])
        assert n_cached == 3
        assert n_total == 5


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

class TestShutdown:
    def test_shutdown_drains_cache(self):
        cache = _make_cache()
        for i in range(3):
            e = _fake_entry(idx=i, ram_mb=1.0)
            cache.donate(i, e.mips, e.img_w, e.img_h)

        cache.shutdown()
        assert cache._ram_used == 0
        assert len(cache._cache) == 0
