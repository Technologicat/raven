#!/usr/bin/env python
"""Diagnostic script for torch.compile on THA3 modules.

Run from the project root with the venv activated:
    python tools_debug_torch_compile.py

Tests each THA3 module individually to find what causes torch.compile to hang.
"""

import os
import sys
import time

# THA3's internal imports expect `tha3` at the top level of sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "raven", "vendor"))

import torch

# Raven's vendored THA3
from raven.vendor.tha3.poser.modes.separable_half import (
    load_eyebrow_decomposer,
    load_eyebrow_morphing_combiner,
    load_face_morpher,
    load_two_algo_generator,
    load_editor,
)

DEVICE = torch.device("cuda")
MODELS_DIR = "raven/vendor/tha3/models/separable_half"


def timed(label, func, *args, **kwargs):
    """Run func, print elapsed time."""
    print(f"  {label}...", end=" ", flush=True)
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f"{dt:.2f}s")
    return result


def load_modules():
    """Load all 5 THA3 modules."""
    modules = {}
    modules["eyebrow_decomposer"] = timed(
        "eyebrow_decomposer",
        load_eyebrow_decomposer, f"{MODELS_DIR}/eyebrow_decomposer.pt")
    modules["eyebrow_morphing_combiner"] = timed(
        "eyebrow_morphing_combiner",
        load_eyebrow_morphing_combiner, f"{MODELS_DIR}/eyebrow_morphing_combiner.pt")
    modules["face_morpher"] = timed(
        "face_morpher",
        load_face_morpher, f"{MODELS_DIR}/face_morpher.pt")
    modules["two_algo_face_body_rotator"] = timed(
        "two_algo_generator",
        load_two_algo_generator, f"{MODELS_DIR}/two_algo_face_body_rotator.pt")
    modules["editor"] = timed(
        "editor",
        load_editor, f"{MODELS_DIR}/editor.pt")

    for m in modules.values():
        m.to(DEVICE)
        m.train(False)

    return modules


def make_inputs():
    """Create dummy inputs for each module."""
    dtype = torch.float16
    return {
        "eyebrow_decomposer": (
            torch.zeros(1, 4, 128, 128, device=DEVICE, dtype=dtype),
        ),
        "eyebrow_morphing_combiner": (
            torch.zeros(1, 4, 128, 128, device=DEVICE, dtype=dtype),  # background
            torch.zeros(1, 4, 128, 128, device=DEVICE, dtype=dtype),  # eyebrow
            torch.zeros(1, 12, device=DEVICE, dtype=dtype),           # pose
        ),
        "face_morpher": (
            torch.zeros(1, 4, 192, 192, device=DEVICE, dtype=dtype),  # image
            torch.zeros(1, 27, device=DEVICE, dtype=dtype),            # pose
        ),
        "two_algo_face_body_rotator": (
            torch.zeros(1, 4, 256, 256, device=DEVICE, dtype=dtype),  # image
            torch.zeros(1, 6, device=DEVICE, dtype=dtype),             # pose
        ),
        "editor": (
            torch.zeros(1, 4, 512, 512, device=DEVICE, dtype=dtype),  # original
            torch.zeros(1, 4, 512, 512, device=DEVICE, dtype=dtype),  # warped
            torch.zeros(1, 2, 512, 512, device=DEVICE, dtype=dtype),  # grid_change
            torch.zeros(1, 6, device=DEVICE, dtype=dtype),             # pose
        ),
    }


# ── Phase 1: dynamo.explain ────────────────────────────────────────

def phase1_explain(modules, inputs):
    print("\n" + "=" * 60)
    print("PHASE 1: torch._dynamo.explain()")
    print("Reports graph breaks without full compilation.")
    print("=" * 60)

    for name, module in modules.items():
        print(f"\n--- {name} ---")
        args = inputs[name]
        try:
            explanation = torch._dynamo.explain(module)(*args)
            print(f"  Graph breaks: {explanation.graph_break_count}")
            if explanation.graph_break_count > 0:
                print("  Break reasons:")
                for reason in explanation.break_reasons:
                    print(f"    - {reason}")
            print(f"  Graphs captured: {explanation.graph_count}")
            print(f"  Ops: {explanation.op_count}")
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")


# ── Phase 2: compile with different backends ───────────────────────

def phase2_backends(modules, inputs):
    print("\n" + "=" * 60)
    print("PHASE 2: torch.compile with different backends")
    print("Tests: eager → aot_eager → inductor (default)")
    print("=" * 60)

    backends = ["eager", "aot_eager", "inductor"]

    for name, module in modules.items():
        print(f"\n--- {name} ---")
        args = inputs[name]

        # Baseline: uncompiled forward pass
        with torch.inference_mode():
            timed("uncompiled forward", lambda: module(*args))

        for backend in backends:
            torch._dynamo.reset()  # clear compile caches between attempts

            try:
                compiled = torch.compile(module, backend=backend)
                # First call triggers compilation
                with torch.inference_mode():
                    timed(f"compile({backend}) first call",
                          lambda: compiled(*args))
                # Second call should be fast (cached)
                with torch.inference_mode():
                    timed(f"compile({backend}) second call",
                          lambda: compiled(*args))
            except Exception as exc:
                print(f"  compile({backend}) FAILED: {type(exc).__name__}: {exc}")

        torch._dynamo.reset()


# ── Phase 3: inductor with timeout ────────────────────────────────

def phase3_inductor_with_timeout(modules, inputs):
    """Try inductor on each module with a hard timeout via alarm signal."""
    import signal

    print("\n" + "=" * 60)
    print("PHASE 3: inductor with 60s timeout per module")
    print("Identifies which module hangs.")
    print("=" * 60)

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError("timed out")

    for name, module in modules.items():
        print(f"\n--- {name} ---")
        args = inputs[name]
        torch._dynamo.reset()

        compiled = torch.compile(module)
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(60)
        try:
            with torch.inference_mode():
                timed("inductor first call", lambda: compiled(*args))
            signal.alarm(0)
            print("  OK")
        except TimeoutError:
            print("  HUNG (>60s) ← this module is the problem")
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            signal.alarm(0)

        torch._dynamo.reset()


# ── Phase 4: full pipeline (mimics server warmup) ─────────────────

def phase4_full_pipeline(modules, inputs):
    """Compile all 5 modules and run them through the actual poser,
    exactly as the server does during warmup."""
    import signal

    print("\n" + "=" * 60)
    print("PHASE 4: full pipeline with torch.compile (mimics server)")
    print("Compiles all 5 modules, then runs poser.pose() — 180s timeout.")
    print("=" * 60)

    from raven.vendor.tha3.poser.modes.separable_half import create_poser

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError("timed out")

    # Create the poser (loads fresh modules).
    # get_modules() already wraps each module with torch.compile.
    torch._dynamo.reset()
    print("\nCreating poser...")
    poser = create_poser(DEVICE, modelsdir="raven/vendor/tha3/models")
    print("Eagerly loading modules (torch.compile wrapping happens here)...")
    poser.get_modules()

    # Now do exactly what the server warmup does
    dtype = torch.float16
    image = torch.zeros(1, 4, 512, 512, device=DEVICE, dtype=dtype)
    pose = torch.zeros(1, 45, device=DEVICE, dtype=dtype)

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(180)
    try:
        print("\nRunning poser.pose() (triggers compilation of all 5 modules)...")
        with torch.inference_mode():
            timed("poser.pose() first call", lambda: poser.pose(image, pose))
        signal.alarm(0)

        print("\nFirst call succeeded! Running second call...")
        with torch.inference_mode():
            timed("poser.pose() second call", lambda: poser.pose(image, pose))

        print("\nRunning 10 more calls for steady-state timing...")
        times = []
        for i in range(10):
            t0 = time.perf_counter()
            with torch.inference_mode():
                poser.pose(image, pose)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt)
        avg = sum(times) / len(times)
        print(f"  Steady-state average: {avg * 1000:.1f}ms ({1.0 / avg:.1f} FPS)")

    except TimeoutError:
        signal.alarm(0)
        print("\n  HUNG (>180s) — full pipeline doesn't work with torch.compile")
    except Exception as exc:
        signal.alarm(0)
        print(f"\n  FAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()

    torch._dynamo.reset()


# ── Phase 5: daemon thread test ───────────────────────────────────

def phase5_daemon_thread():
    """Test if torch.compile hangs when triggered from a daemon thread."""
    import threading

    print("\n" + "=" * 60)
    print("PHASE 5: torch.compile from daemon thread")
    print("Tests the threading theory — 120s timeout.")
    print("=" * 60)

    from raven.vendor.tha3.poser.modes.separable_half import create_poser

    result = {"status": None, "time": None, "error": None}

    def worker():
        try:
            torch._dynamo.reset()
            poser = create_poser(DEVICE, modelsdir="raven/vendor/tha3/models")
            poser.get_modules()

            dtype = torch.float16
            image = torch.zeros(1, 4, 512, 512, device=DEVICE, dtype=dtype)
            pose = torch.zeros(1, 45, device=DEVICE, dtype=dtype)

            print("  [daemon] Running poser.pose()...")
            t0 = time.perf_counter()
            with torch.inference_mode():
                poser.pose(image, pose)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            result["status"] = "ok"
            result["time"] = dt
            print(f"  [daemon] First call: {dt:.2f}s")

            # Steady-state
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                with torch.inference_mode():
                    poser.pose(image, pose)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            avg = sum(times) / len(times)
            print(f"  [daemon] Steady-state: {avg * 1000:.1f}ms ({1.0 / avg:.1f} FPS)")

        except Exception as exc:
            result["status"] = "error"
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  [daemon] FAILED: {result['error']}")

    t = threading.Thread(target=worker, daemon=True)
    print("Starting daemon thread...")
    t.start()
    t.join(timeout=120)

    if t.is_alive():
        print("  HUNG (>120s) on daemon thread ← confirms threading theory")
    elif result["status"] == "ok":
        print("  OK — daemon thread works fine (threading is NOT the issue)")
    else:
        print(f"  Failed: {result['error']}")


# ── Phase 6: compiled vs uncompiled baseline ──────────────────────

def phase6_benchmark(mode="both"):
    """Benchmark full pipeline. Run with separate processes for fair VRAM comparison.

    mode: "baseline" | "compiled" | "both" (both in same process, less fair)
    """
    print("\n" + "=" * 60)
    print(f"PHASE 6: full-pipeline benchmark (mode={mode})")
    print("=" * 60)

    from raven.vendor.tha3.poser.modes.separable_half import create_poser

    dtype = torch.float16
    image = torch.zeros(1, 4, 512, 512, device=DEVICE, dtype=dtype)
    pose = torch.zeros(1, 45, device=DEVICE, dtype=dtype)
    n_warmup = 5
    n_measure = 100

    def measure(poser, label):
        # Warmup
        for _ in range(n_warmup):
            with torch.inference_mode():
                poser.pose(image, pose)
            torch.cuda.synchronize()

        # Measure
        times = []
        for _ in range(n_measure):
            t0 = time.perf_counter()
            with torch.inference_mode():
                poser.pose(image, pose)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        avg = sum(times) / len(times)
        mn = min(times)
        mx = max(times)
        print(f"  {label}: avg {avg * 1000:.2f}ms  min {mn * 1000:.2f}ms  max {mx * 1000:.2f}ms  ({1.0 / avg:.1f} FPS)")
        return avg

    results = {}

    if mode in ("baseline", "both"):
        print("\nLoading uncompiled poser...")
        # Temporarily disable torch.compile in get_modules
        original_compile = torch.compile
        torch.compile = lambda m, **kw: m  # no-op
        torch._dynamo.reset()
        poser_baseline = create_poser(DEVICE, modelsdir="raven/vendor/tha3/models")
        poser_baseline.get_modules()
        torch.compile = original_compile  # restore

        print(f"Measuring uncompiled ({n_measure} iterations)...")
        results["baseline"] = measure(poser_baseline, "uncompiled")

        if mode == "both":
            del poser_baseline
            torch.cuda.empty_cache()

    if mode in ("compiled", "both"):
        print("\nLoading compiled poser...")
        torch._dynamo.reset()
        poser_compiled = create_poser(DEVICE, modelsdir="raven/vendor/tha3/models")
        poser_compiled.get_modules()

        print("Warming up compilation (first calls are slow)...")
        for i in range(5):
            with torch.inference_mode():
                poser_compiled.pose(image, pose)
            torch.cuda.synchronize()
            print(f"  compile warmup {i + 1}/5")

        print(f"Measuring compiled ({n_measure} iterations)...")
        results["compiled"] = measure(poser_compiled, "compiled  ")

    if "baseline" in results and "compiled" in results:
        speedup = results["baseline"] / results["compiled"]
        print(f"\n  Speedup: {speedup:.2f}×")

    torch._dynamo.reset()


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading THA3 modules...")
    modules = load_modules()
    inputs = make_inputs()

    phase = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    if phase == 0 or phase == 1:
        phase1_explain(modules, inputs)
    if phase == 0 or phase == 2:
        phase2_backends(modules, inputs)
    if phase == 0 or phase == 3:
        phase3_inductor_with_timeout(modules, inputs)
    if phase == 0 or phase == 4:
        phase4_full_pipeline(modules, inputs)
    if phase == 0 or phase == 5:
        phase5_daemon_thread()
    if phase == 6:
        # Run as: `python script.py 6 baseline` or `6 compiled` or `6 both`
        mode = sys.argv[2] if len(sys.argv) > 2 else "both"
        phase6_benchmark(mode)
    if phase == 0:
        phase6_benchmark("both")

    print("\nDone.")
