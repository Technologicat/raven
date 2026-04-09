#!/usr/bin/env python
"""Benchmark the postprocessor filter chain.

Measures per-filter and total chain timing with the default avatar filter chain.
Run from the project root with the venv activated:

    python -m raven.common.video.tests.bench_postprocessor          # default chain + all filters
    python -m raven.common.video.tests.bench_postprocessor chain    # default chain only
    python -m raven.common.video.tests.bench_postprocessor all      # all filters only

Not collected by pytest (no `test_` prefix).
"""

import time

import torch

from raven.common.video.postprocessor import Postprocessor


DEVICE = torch.device("cuda")
DTYPE = torch.float16
N_WARMUP = 10
N_MEASURE = 200

# Default avatar filter chain from raven.server.config
DEFAULT_CHAIN = [
    ("bloom", {}),
    ("chromatic_aberration", {}),
    ("vignetting", {}),
    ("translucent_display", {}),
    ("banding", {}),
    ("scanlines", {}),
]

# All public filters with representative default parameters
ALL_FILTERS = [
    ("zoom", {"factor": 2.0, "quality": "low"}),
    ("bloom", {}),
    ("chromatic_aberration", {}),
    ("vignetting", {}),
    ("desaturate", {}),
    ("noise", {"strength": 0.2, "sigma": 2.0, "channel": "A"}),
    ("analog_lowres", {}),
    ("analog_rippling_hsync", {}),
    ("analog_runaway_hsync", {}),
    ("analog_vhs_noise", {}),
    ("analog_vhsglitches", {}),
    ("analog_vhs_headswitching", {}),
    ("analog_vhstracking", {}),
    ("digital_glitches", {}),
    ("translucent_display", {}),
    ("monochrome_display", {}),
    ("banding", {}),
    ("scanlines", {}),
]


def bench_full_chain(pp, image_template):
    """Benchmark the full chain as a unit."""
    pp.chain = DEFAULT_CHAIN

    # Warmup
    for _ in range(N_WARMUP):
        image = image_template.clone()
        with torch.inference_mode():
            pp.render_into(image)
        torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(N_MEASURE):
        image = image_template.clone()
        t0 = time.perf_counter()
        with torch.inference_mode():
            pp.render_into(image)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)
    print(f"  Full chain:  avg {avg * 1000:.3f}ms  min {mn * 1000:.3f}ms  max {mx * 1000:.3f}ms")
    return avg


def bench_per_filter(pp, image_template):
    """Benchmark each filter individually to find the hot spots."""
    print(f"\n  {'Filter':<30s} {'avg':>8s} {'min':>8s} {'max':>8s}  {'share':>6s}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8}  {'-' * 6}")

    filter_results = {}
    for filter_name, settings in DEFAULT_CHAIN:
        pp.chain = [(filter_name, settings)]

        # Warmup
        for _ in range(N_WARMUP):
            image = image_template.clone()
            with torch.inference_mode():
                pp.render_into(image)
            torch.cuda.synchronize()

        # Measure
        times = []
        for _ in range(N_MEASURE):
            image = image_template.clone()
            t0 = time.perf_counter()
            with torch.inference_mode():
                pp.render_into(image)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        mn = min(times)
        mx = max(times)
        filter_results[filter_name] = (avg, mn, mx)

    total = sum(avg for avg, _, _ in filter_results.values())
    for filter_name, (avg, mn, mx) in filter_results.items():
        share = avg / total * 100
        print(f"  {filter_name:<30s} {avg * 1000:>7.3f}ms {mn * 1000:>7.3f}ms {mx * 1000:>7.3f}ms  {share:>5.1f}%")

    print(f"  {'(sum of individual filters)':<30s} {total * 1000:>7.3f}ms")
    return filter_results


def bench_dispatch_overhead(pp, image_template):
    """Measure the overhead of render_into dispatch vs direct filter calls."""
    # Direct call to each filter method (bypass render_into dispatch)
    filters = [(filter_name, settings, getattr(pp, filter_name)) for filter_name, settings in DEFAULT_CHAIN]

    # Warmup
    for _ in range(N_WARMUP):
        image = image_template.clone()
        with torch.inference_mode():
            for _, settings, func in filters:
                func(image, **settings)
        torch.cuda.synchronize()

    # Measure direct calls
    times_direct = []
    for _ in range(N_MEASURE):
        image = image_template.clone()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _, settings, func in filters:
                func(image, **settings)
        torch.cuda.synchronize()
        times_direct.append(time.perf_counter() - t0)

    # Measure via render_into
    pp.chain = DEFAULT_CHAIN
    times_dispatch = []
    for _ in range(N_MEASURE):
        image = image_template.clone()
        t0 = time.perf_counter()
        with torch.inference_mode():
            pp.render_into(image)
        torch.cuda.synchronize()
        times_dispatch.append(time.perf_counter() - t0)

    avg_direct = sum(times_direct) / len(times_direct)
    avg_dispatch = sum(times_dispatch) / len(times_dispatch)
    overhead = (avg_dispatch - avg_direct) / avg_direct * 100

    print(f"\n  Direct calls:    avg {avg_direct * 1000:.3f}ms")
    print(f"  render_into:     avg {avg_dispatch * 1000:.3f}ms")
    print(f"  Dispatch overhead: {overhead:+.1f}%")


def bench_all_filters(pp, image_template):
    """Benchmark every filter individually."""
    print(f"\n  {'Filter':<30s} {'avg':>8s} {'min':>8s} {'max':>8s}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8}")

    results = {}
    for filter_name, settings in ALL_FILTERS:
        pp.chain = [(filter_name, settings)]

        # Warmup
        for _ in range(N_WARMUP):
            image = image_template.clone()
            with torch.inference_mode():
                pp.render_into(image)
            torch.cuda.synchronize()

        # Measure
        times = []
        for _ in range(N_MEASURE):
            image = image_template.clone()
            t0 = time.perf_counter()
            with torch.inference_mode():
                pp.render_into(image)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        mn = min(times)
        mx = max(times)
        results[filter_name] = (avg, mn, mx)
        print(f"  {filter_name:<30s} {avg * 1000:>7.3f}ms {mn * 1000:>7.3f}ms {mx * 1000:>7.3f}ms")

    # Sort by avg descending
    print("\n  Ranked by cost:")
    for name, (avg, _, _) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"    {name:<30s} {avg * 1000:>7.3f}ms")

    return results


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "default"
    sizes = [512, 768, 1024]

    pp = Postprocessor(device=DEVICE, dtype=DTYPE)

    for size in sizes:
        print(f"\n{'#' * 70}")
        print(f"# Image: 4×{size}×{size}, {N_MEASURE} iterations, {DEVICE}, {DTYPE}")
        print(f"{'#' * 70}")

        image_template = torch.rand(4, size, size, device=DEVICE, dtype=DTYPE)

        if mode in ("default", "chain"):
            print("\n" + "=" * 70)
            print("Full default chain timing:")
            bench_full_chain(pp, image_template)

            print("\n" + "=" * 70)
            print("Per-filter breakdown (default chain):")
            bench_per_filter(pp, image_template)

            if size == sizes[0]:
                print("\n" + "=" * 70)
                print("Dispatch overhead:")
                bench_dispatch_overhead(pp, image_template)

        if mode in ("default", "all"):
            print("\n" + "=" * 70)
            print("All filters (individual):")
            bench_all_filters(pp, image_template)

    print("\nDone.")
