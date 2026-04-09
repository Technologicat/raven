#!/usr/bin/env python
"""Benchmark: can THA3 and upscale+postproc overlap on the GPU?

Measures sequential vs concurrent execution of the two pipeline halves
on separate threads. If concurrent is faster, the 4-thread pipelining
approach will yield real throughput gains.

Run from the project root with the venv activated:
    python tools_bench_pipeline_overlap.py
"""

import os
import sys
import time
import threading

# THA3 vendor path setup
sys.path.append(os.path.join(os.path.dirname(__file__), "raven", "vendor"))

import torch

from raven.vendor.tha3.poser.modes.separable_half import create_poser
from raven.common.video.postprocessor import Postprocessor
from raven.common.video.upscaler import Upscaler

DEVICE = torch.device("cuda")
DTYPE = torch.float16
N_WARMUP = 5
N_MEASURE = 100

# Default postprocessor chain (from raven.server.config)
DEFAULT_CHAIN = [
    ("bloom", {}),
    ("chromatic_aberration", {}),
    ("vignetting", {}),
    ("translucent_display", {}),
    ("banding", {}),
    ("scanlines", {}),
]


def make_pose_workload():
    """Create the THA3 pose workload."""
    poser = create_poser(DEVICE, modelsdir="raven/vendor/tha3/models")
    poser.get_modules()
    image = torch.zeros(1, 4, 512, 512, device=DEVICE, dtype=DTYPE)
    pose = torch.zeros(1, 45, device=DEVICE, dtype=DTYPE)

    def run_pose():
        with torch.inference_mode():
            return poser.pose(image, pose)

    return run_pose


def make_postproc_workload():
    """Create the upscale+postproc workload (512→1024 + filter chain)."""
    upscaler = Upscaler(device=DEVICE, dtype=DTYPE,
                        upscaled_width=1024, upscaled_height=1024,
                        preset="C", quality="low")
    postprocessor = Postprocessor(device=DEVICE, dtype=DTYPE)
    postprocessor.chain = DEFAULT_CHAIN

    # Use a realistic input: 4-channel RGBA, range [0, 1], 512×512
    # (this is what THA3 outputs)
    input_image = torch.rand(4, 512, 512, device=DEVICE, dtype=DTYPE)

    def run_postproc():
        with torch.inference_mode():
            upscaled = upscaler.upscale(input_image)
            postprocessor.render_into(upscaled)

    return run_postproc


def measure_sequential(run_pose, run_postproc, n):
    """Run pose then postproc sequentially, n times."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        run_pose()
        run_postproc()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def measure_concurrent(run_pose, run_postproc, n):
    """Run pose and postproc on separate threads simultaneously, n times.

    Each thread does n iterations. We measure total wall time.
    This simulates the best case for pipelining: both stages running
    at full throughput simultaneously.
    """
    barrier = threading.Barrier(2)  # sync start

    def pose_worker():
        barrier.wait()
        for _ in range(n):
            run_pose()
        torch.cuda.synchronize()

    def postproc_worker():
        barrier.wait()
        for _ in range(n):
            run_postproc()
        torch.cuda.synchronize()

    t_pose = threading.Thread(target=pose_worker)
    t_post = threading.Thread(target=postproc_worker)

    t0 = time.perf_counter()
    t_pose.start()
    t_post.start()
    t_pose.join()
    t_post.join()
    total = time.perf_counter() - t0

    return total


def measure_individual(run_fn, name, n):
    """Measure a single workload in isolation."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        run_fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    print(f"  {name}: avg {avg * 1000:.2f}ms")
    return avg


if __name__ == "__main__":
    print(f"Pipeline overlap benchmark (device={DEVICE}, dtype={DTYPE})")
    print(f"THA3 separable_half, Anime4K C/low 512→1024, default postproc chain")
    print(f"{N_MEASURE} iterations per measurement, {N_WARMUP} warmup\n")

    print("Loading workloads...")
    run_pose = make_pose_workload()
    run_postproc = make_postproc_workload()

    # Warmup both
    print("Warming up...")
    for _ in range(N_WARMUP):
        run_pose()
        run_postproc()
    torch.cuda.synchronize()

    # Individual baselines
    print("\n--- Individual baselines ---")
    avg_pose = measure_individual(run_pose, "pose (THA3)", N_MEASURE)
    avg_postproc = measure_individual(run_postproc, "upscale+postproc", N_MEASURE)
    print(f"  sum: {(avg_pose + avg_postproc) * 1000:.2f}ms")

    # Sequential
    print(f"\n--- Sequential (pose then postproc, {N_MEASURE} frames) ---")
    seq_times = measure_sequential(run_pose, run_postproc, N_MEASURE)
    avg_seq = sum(seq_times) / len(seq_times)
    print(f"  avg per frame: {avg_seq * 1000:.2f}ms ({1.0 / avg_seq:.1f} FPS)")

    # Concurrent
    print(f"\n--- Concurrent (two threads, {N_MEASURE} iterations each) ---")
    # Re-warmup in concurrent mode (threads may need CUDA context init)
    for _ in range(N_WARMUP):
        measure_concurrent(run_pose, run_postproc, 1)

    total_concurrent = measure_concurrent(run_pose, run_postproc, N_MEASURE)
    avg_concurrent = total_concurrent / N_MEASURE
    print(f"  total: {total_concurrent * 1000:.1f}ms for {N_MEASURE} iterations")
    print(f"  avg per frame: {avg_concurrent * 1000:.2f}ms ({1.0 / avg_concurrent:.1f} FPS)")

    # CUDA streams (single thread, dual streams — no GIL contention)
    print(f"\n--- CUDA streams (single thread, two streams, {N_MEASURE} frames) ---")
    stream_pose = torch.cuda.Stream()
    stream_post = torch.cuda.Stream()

    # Make a realistic posed image for the postproc stage
    posed_image = torch.rand(4, 512, 512, device=DEVICE, dtype=DTYPE)
    postproc = Postprocessor(device=DEVICE, dtype=DTYPE)
    postproc.chain = DEFAULT_CHAIN
    upscaler_obj = Upscaler(device=DEVICE, dtype=DTYPE,
                            upscaled_width=1024, upscaled_height=1024,
                            preset="C", quality="low")

    poser = create_poser(DEVICE, modelsdir="raven/vendor/tha3/models")
    poser.get_modules()
    dummy_image = torch.zeros(1, 4, 512, 512, device=DEVICE, dtype=DTYPE)
    dummy_pose = torch.zeros(1, 45, device=DEVICE, dtype=DTYPE)

    # Warmup
    for _ in range(N_WARMUP):
        with torch.cuda.stream(stream_pose):
            with torch.inference_mode():
                poser.pose(dummy_image, dummy_pose)
        with torch.cuda.stream(stream_post):
            with torch.inference_mode():
                up = upscaler_obj.upscale(posed_image)
                postproc.render_into(up)
        torch.cuda.synchronize()

    # Measure: interleaved dispatch on two streams, simulating pipelined frames
    stream_times = []
    prev_posed = posed_image.clone()
    for _ in range(N_MEASURE):
        t0 = time.perf_counter()

        # Dispatch pose to stream_pose
        with torch.cuda.stream(stream_pose):
            with torch.inference_mode():
                poser.pose(dummy_image, dummy_pose)

        # Dispatch postproc (of previous frame) to stream_post
        with torch.cuda.stream(stream_post):
            with torch.inference_mode():
                up = upscaler_obj.upscale(prev_posed)
                postproc.render_into(up)

        # Wait for both to complete
        torch.cuda.synchronize()
        stream_times.append(time.perf_counter() - t0)

    avg_streams = sum(stream_times) / len(stream_times)
    print(f"  avg per frame: {avg_streams * 1000:.2f}ms ({1.0 / avg_streams:.1f} FPS)")

    # Analysis
    print("\n--- Analysis ---")
    theoretical_max = max(avg_pose, avg_postproc)
    print(f"  Individual sum:  {(avg_pose + avg_postproc) * 1000:.2f}ms")
    print(f"  Sequential:      {avg_seq * 1000:.2f}ms per frame")
    print(f"  Threads:         {avg_concurrent * 1000:.2f}ms per frame (GIL-limited)")
    print(f"  CUDA streams:    {avg_streams * 1000:.2f}ms per frame")
    print(f"  Theoretical:     {theoretical_max * 1000:.2f}ms (perfect overlap)")
    print()
    stream_speedup = avg_seq / avg_streams
    stream_efficiency = theoretical_max / avg_streams * 100
    print(f"  Streams speedup: {stream_speedup:.2f}×  ({stream_efficiency:.0f}% of theoretical)")
    if avg_streams < avg_seq * 0.85:
        print("  ✓ Significant overlap — CUDA stream pipelining is viable.")
    elif avg_streams < avg_seq * 0.95:
        print("  ~ Marginal overlap — may help slightly.")
    else:
        print("  ✗ No overlap — GPU fully contended.")

    print("\nDone.")
