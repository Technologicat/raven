#!/usr/bin/env python3
"""Benchmark for the image loading and thumbnail pipeline.

Measures each stage independently, then the end-to-end pipeline, to identify
where the bottleneck is and how much pipelining helps.

Does nothing unless ``--dir`` is given.

Usage::

    python -m raven.cherrypick.tests.loader_bench --dir /path/to/images
    python -m raven.cherrypick.tests.loader_bench --dir /path/to/images --device cuda:1
    python -m raven.cherrypick.tests.loader_bench --dir /path/to/images --tile-size 256

Stages measured:

1. **Disk read**: raw ``open().read()`` — pure I/O, no decode.
2. **PIL decode**: ``Image.open()`` + ``.convert("RGBA")`` → numpy array.
3. **QOI decode**: ``qoi.decode()`` (for .qoi files).
4. **CPU→GPU transfer**: numpy → torch tensor on device.
5. **GPU resize (Lanczos)**: ``lanczos.resize`` on device.
6. **GPU→CPU transfer**: result tensor back to CPU numpy.
7. **End-to-end sequential**: decode + transfer + resize + transfer back, one at a time.
8. **End-to-end batched**: same but in batches of N.
"""

import argparse
import pathlib
import time
from typing import Optional

import numpy as np
import torch
from PIL import Image

from raven.common.image.codec import IMAGE_EXTENSIONS
from raven.common.image import lanczos


def _ns_to_s(ns: int) -> float:
    """Convert nanoseconds to seconds."""
    return ns / 1_000_000_000


def _collect_images(directory: pathlib.Path, limit: Optional[int] = None) -> list[pathlib.Path]:
    """Collect image files, sorted by name."""
    files = sorted(f for f in directory.iterdir()
                   if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
    if limit is not None:
        files = files[:limit]
    return files


def _decode_image(path: pathlib.Path) -> np.ndarray:
    """Decode an image file to RGBA uint8 numpy array."""
    if path.suffix.lower() == ".qoi":
        import qoi
        return qoi.decode(path.read_bytes())
    else:
        return np.array(Image.open(path).convert("RGBA"))


def _np_to_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    """Convert (H, W, 4) uint8 numpy to (1, 4, H, W) float32 tensor on device."""
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0


# ---------------------------------------------------------------------------
# Individual stage benchmarks
# ---------------------------------------------------------------------------

def bench_raw_read(files: list[pathlib.Path]) -> dict:
    """Stage 1: raw disk read (no decode)."""
    total_bytes = 0
    t0 = time.perf_counter_ns()
    for f in files:
        data = f.read_bytes()
        total_bytes += len(data)
    elapsed = _ns_to_s(time.perf_counter_ns() - t0)
    return {"stage": "raw_read", "files": len(files), "elapsed_s": elapsed,
            "total_mb": total_bytes / 1e6, "throughput_mbs": total_bytes / 1e6 / elapsed}


def bench_decode(files: list[pathlib.Path]) -> dict:
    """Stage 2: decode to numpy RGBA."""
    t0 = time.perf_counter_ns()
    arrays = []
    for f in files:
        arrays.append(_decode_image(f))
    elapsed = _ns_to_s(time.perf_counter_ns() - t0)
    total_pixels = sum(a.shape[0] * a.shape[1] for a in arrays)
    return {"stage": "decode", "files": len(files), "elapsed_s": elapsed,
            "total_mp": total_pixels / 1e6, "throughput_mps": total_pixels / 1e6 / elapsed,
            "per_image_ms": elapsed / len(files) * 1000}


def bench_transfer_to_gpu(arrays: list[np.ndarray], device: str) -> dict:
    """Stage 3: numpy → GPU tensor (one at a time, to avoid OOM)."""
    if device == "cpu":
        return {"stage": "cpu→device", "note": "skipped (device is cpu)"}
    # Warm up.
    _np_to_tensor(arrays[0], device)
    torch.cuda.synchronize(device)

    total_bytes = 0
    t0 = time.perf_counter_ns()
    for a in arrays:
        t = _np_to_tensor(a, device)
        total_bytes += a.nbytes
        del t
    torch.cuda.synchronize(device)
    elapsed = _ns_to_s(time.perf_counter_ns() - t0)
    return {"stage": "cpu→device", "files": len(arrays), "elapsed_s": elapsed,
            "total_mb": total_bytes / 1e6, "throughput_mbs": total_bytes / 1e6 / elapsed}


def bench_gpu_resize(arrays: list[np.ndarray], device: str,
                     tile_size: int, order: int) -> dict:
    """Stage 4: GPU Lanczos resize (transfer + resize + free, one at a time)."""
    # Warm up.
    t = _np_to_tensor(arrays[0], device)
    lanczos.resize(t, tile_size, tile_size, order=order)
    del t
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)

    total_pixels = 0
    t0 = time.perf_counter_ns()
    for a in arrays:
        t = _np_to_tensor(a, device)
        r = lanczos.resize(t, tile_size, tile_size, order=order)
        total_pixels += a.shape[0] * a.shape[1]
        del t, r
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed = _ns_to_s(time.perf_counter_ns() - t0)
    return {"stage": "gpu_resize", "files": len(arrays), "elapsed_s": elapsed,
            "total_mp": total_pixels / 1e6, "throughput_mps": total_pixels / 1e6 / elapsed,
            "per_image_ms": elapsed / len(arrays) * 1000, "order": order}


def bench_transfer_to_cpu(arrays: list[np.ndarray], device: str,
                          tile_size: int, order: int) -> dict:
    """Stage 5: GPU thumbnail tensor → CPU numpy (one at a time)."""
    if device == "cpu":
        return {"stage": "device→cpu", "note": "skipped (already cpu)"}
    # Generate thumbnails to transfer back.
    # Warm up.
    t = _np_to_tensor(arrays[0], device)
    r = lanczos.resize(t, tile_size, tile_size, order=order)
    _ = r.cpu().numpy()
    del t, r
    torch.cuda.synchronize(device)

    total_bytes = 0
    t0 = time.perf_counter_ns()
    for a in arrays:
        t = _np_to_tensor(a, device)
        r = lanczos.resize(t, tile_size, tile_size, order=order)
        out = r.cpu().numpy()
        total_bytes += out.nbytes
        del t, r
    torch.cuda.synchronize(device)
    elapsed = _ns_to_s(time.perf_counter_ns() - t0)
    return {"stage": "resize+device→cpu", "files": len(arrays), "elapsed_s": elapsed,
            "total_mb": total_bytes / 1e6, "throughput_mbs": total_bytes / 1e6 / elapsed}


# ---------------------------------------------------------------------------
# End-to-end benchmarks
# ---------------------------------------------------------------------------

def bench_e2e_sequential(files: list[pathlib.Path], device: str,
                         tile_size: int, order: int) -> dict:
    """End-to-end: decode → transfer → resize → transfer back, one image at a time."""
    # Warm up.
    arr = _decode_image(files[0])
    t = _np_to_tensor(arr, device)
    r = lanczos.resize(t, tile_size, tile_size, order=order)
    _ = r.cpu().numpy()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)

    t0 = time.perf_counter_ns()
    for f in files:
        arr = _decode_image(f)
        t = _np_to_tensor(arr, device)
        r = lanczos.resize(t, tile_size, tile_size, order=order)
        _ = r.cpu().numpy()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed = _ns_to_s(time.perf_counter_ns() - t0)
    return {"stage": "e2e_sequential", "files": len(files), "elapsed_s": elapsed,
            "per_image_ms": elapsed / len(files) * 1000,
            "images_per_s": len(files) / elapsed}


def bench_e2e_batched(files: list[pathlib.Path], device: str,
                      tile_size: int, order: int, batch_size: int) -> dict:
    """End-to-end batched: decode a batch, stack, resize, transfer back."""
    # Warm up.
    arr = _decode_image(files[0])
    t = _np_to_tensor(arr, device)
    lanczos.resize(t, tile_size, tile_size, order=order)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)

    t0 = time.perf_counter_ns()
    for batch_start in range(0, len(files), batch_size):
        batch_files = files[batch_start:batch_start + batch_size]
        # Decode batch.
        arrays = [_decode_image(f) for f in batch_files]
        # Resize individually (images may differ in size, can't stack).
        for arr in arrays:
            t = _np_to_tensor(arr, device)
            r = lanczos.resize(t, tile_size, tile_size, order=order)
            _ = r.cpu().numpy()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed = _ns_to_s(time.perf_counter_ns() - t0)
    return {"stage": f"e2e_batched(bs={batch_size})", "files": len(files),
            "elapsed_s": elapsed, "per_image_ms": elapsed / len(files) * 1000,
            "images_per_s": len(files) / elapsed}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_result(result: dict):
    """Pretty-print one benchmark result."""
    stage = result.pop("stage")
    note = result.pop("note", None)
    if note:
        print(f"  {stage:24s}  {note}")
        return

    parts = []
    if "elapsed_s" in result:
        parts.append(f"{result['elapsed_s']*1000:.1f} ms total")
    if "per_image_ms" in result:
        parts.append(f"{result['per_image_ms']:.2f} ms/image")
    if "images_per_s" in result:
        parts.append(f"{result['images_per_s']:.1f} images/s")
    if "throughput_mbs" in result:
        parts.append(f"{result['throughput_mbs']:.1f} MB/s")
    if "throughput_mps" in result:
        parts.append(f"{result['throughput_mps']:.1f} MP/s")
    if "files" in result:
        parts.append(f"n={result['files']}")

    print(f"  {stage:24s}  {' | '.join(parts)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark image loading and thumbnail pipeline. "
                    "Does nothing unless --dir is given.")
    parser.add_argument("--dir", required=True, help="Directory of images to benchmark")
    parser.add_argument("--device", default=None, help="Torch device (default: auto)")
    parser.add_argument("--tile-size", type=int, default=128, help="Thumbnail size (default: 128)")
    parser.add_argument("--order", type=int, default=4, help="Lanczos order (default: 4)")
    parser.add_argument("--limit", type=int, default=None, help="Max images to use")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for batched test")
    parser.add_argument("--drop-caches", action="store_true",
                        help="Hint: run 'sudo sh -c \"echo 3 > /proc/sys/vm/drop_caches\"' before "
                             "raw_read to measure cold-cache disk speed. This script can't do it "
                             "itself (needs root).")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    directory = pathlib.Path(args.dir)

    files = _collect_images(directory, limit=args.limit)
    if not files:
        print(f"No image files found in {directory}")
        return

    # Summary.
    total_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"Directory: {directory}")
    print(f"Files: {len(files)} images, {total_mb:.1f} MB on disk")
    print(f"Device: {device}", end="")
    if device.startswith("cuda"):
        print(f" ({torch.cuda.get_device_name(device)})", end="")
    print(f"\nTile size: {args.tile_size}, Lanczos order: {args.order}")

    # --- Stage benchmarks ---
    print(f"\n{'─'*70}")
    print("Stage benchmarks")
    print(f"{'─'*70}")

    print_result(bench_raw_read(files))

    print_result(bench_decode(files))
    # Keep decoded arrays for later stages.
    arrays = [_decode_image(f) for f in files]

    print_result(bench_transfer_to_gpu(arrays, device))
    print_result(bench_gpu_resize(arrays, device, args.tile_size, args.order))
    print_result(bench_transfer_to_cpu(arrays, device, args.tile_size, args.order))

    # --- End-to-end benchmarks ---
    print(f"\n{'─'*70}")
    print("End-to-end benchmarks")
    print(f"{'─'*70}")

    print_result(bench_e2e_sequential(files, device, args.tile_size, args.order))
    print_result(bench_e2e_batched(files, device, args.tile_size, args.order, args.batch_size))

    if args.drop_caches:
        print("\nNote: --drop-caches was given as a reminder. To measure cold-cache disk speed:")
        print("  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
        print("  Then re-run this benchmark.")


if __name__ == "__main__":
    main()
