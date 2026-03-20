#!/usr/bin/env python3
"""Visual test patterns for evaluating the Lanczos resize filter.

Generates synthetic test patterns at 512×512, then downscales each using our
Lanczos kernel to 256, 128, and 64 pixels.  Results are saved as PNGs.

Does nothing unless ``--outdir`` or ``--show`` is given, so regular test runs
don't spam the disk.

Usage::

    python -m raven.common.tests.lanczos_visual_test --outdir /tmp/lanczos_demo
    python -m raven.common.tests.lanczos_visual_test --show
    python -m raven.common.tests.lanczos_visual_test --show --orders 3,4,5
    python -m raven.common.tests.lanczos_visual_test --bench

What to look for:

- **freq_sweep_h**: clean rolloff at the Nyquist cutoff, no aliased beats
  past the cutoff.  Frequency appears to increase linearly (it's a chirp:
  instantaneous frequency ∝ x).
- **radial_sweep**: concentric rings.  High-frequency outer rings should
  fade to uniform gray; Moiré indicates insufficient stopband attenuation.
  Compare Lanczos-3 vs -4 vs -5 to see improvement.
- **spokes**: angular aliasing test.  Center should stay clean.
- **text_and_shapes**: sharpness preservation — text should remain readable.
- **checker_blockN**: ringing at black/white transitions (expected for Lanczos).
- **diag_lines**: 1px diagonal lines — tests sub-pixel accuracy.
- **color_gradient**: smooth color fidelity, no banding.
"""

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from raven.common.image import lanczos


def to_tensor(img, device="cuda"):
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0


def from_tensor(t):
    return (t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def make_patterns(H=512, W=512):
    """Generate a dict of {name: (H, W, 3) uint8 array} test patterns."""
    patterns = {}

    # 1. Checkerboards at various block sizes.
    for block in [4, 8, 16, 32]:
        y, x = np.mgrid[:H, :W]
        checker = ((y // block + x // block) % 2 * 255).astype(np.uint8)
        patterns[f"checker_block{block}"] = np.stack([checker] * 3, axis=-1)

    # 2. Horizontal frequency sweep (chirp: frequency increases left to right).
    xn = np.linspace(0, 1, W)
    freq = np.sin(2 * np.pi * xn**2 * 40)
    sweep = ((freq * 0.5 + 0.5) * 255).astype(np.uint8)
    patterns["freq_sweep_h"] = np.stack([np.tile(sweep[None, :], (H, 1))] * 3, axis=-1)

    # 3. Radial frequency sweep (concentric rings, frequency increases outward).
    y, x = np.mgrid[:H, :W]
    r = np.sqrt((y - H / 2)**2 + (x - W / 2)**2)
    rings = np.sin(r**1.3 * 0.15) * 0.5 + 0.5
    img = (rings * 255).clip(0, 255).astype(np.uint8)
    patterns["radial_sweep"] = np.stack([img] * 3, axis=-1)

    # 4. Spoke/star pattern (angular aliasing test).
    y, x = np.mgrid[:H, :W]
    theta = np.arctan2(y - H / 2, x - W / 2)
    spokes = np.sin(theta * 24) * 0.5 + 0.5
    patterns["spokes"] = np.stack([(spokes * 255).astype(np.uint8)] * 3, axis=-1)

    # 5. Diagonal lines (1px).
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(0, H + W, 4):
        for t in range(max(0, i - W + 1), min(H, i + 1)):
            c = i - t
            if 0 <= c < W:
                img[t, c] = 255
    patterns["diag_lines"] = img

    # 6. Color gradient (smooth).
    r_ch = np.linspace(0, 255, W).astype(np.uint8)
    g_ch = np.linspace(255, 0, W).astype(np.uint8)
    b_ch = np.full(W, 128, dtype=np.uint8)
    row = np.stack([r_ch, g_ch, b_ch], axis=-1)
    img = np.tile(row[None, :, :], (H, 1, 1))
    brightness = np.linspace(0.3, 1.0, H).astype(np.float32)
    img = (img.astype(np.float32) * brightness[:, None, None]).clip(0, 255).astype(np.uint8)
    patterns["color_gradient"] = img

    # 7. Text and geometric shapes.
    pil_img = Image.new("RGB", (W, H), (240, 240, 240))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
        small_font = font
    for i, t in enumerate(["The quick brown fox", "RAVEN-CHERRYPICK", "0123456789"]):
        draw.text((20, 20 + i * 60), t, fill=(0, 0, 0), font=font)
    for i in range(8):
        draw.text((20, 220 + i * 20), f"Small text line {i}: {chr(65 + i) * 30}",
                  fill=(80, 80, 80), font=small_font)
    draw.rectangle([350, 20, 490, 100], outline=(200, 0, 0), width=2)
    draw.ellipse([350, 120, 490, 220], outline=(0, 0, 200), width=2)
    draw.line([(350, 240), (490, 340)], fill=(0, 150, 0), width=1)
    draw.line([(350, 340), (490, 240)], fill=(0, 150, 0), width=1)
    patterns["text_and_shapes"] = np.array(pil_img)

    return patterns


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(device):
    """Benchmark Lanczos resize at different sizes and orders."""
    print(f"\n{'='*70}")
    print(f"Benchmark on {device}")
    if device.startswith("cuda"):
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print(f"{'='*70}")

    # Warm up.
    dummy = torch.randn(1, 3, 512, 512, device=device)
    lanczos.resize(dummy, 128, 128, order=3)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)

    configs = [
        # (input_h, input_w, target_h, target_w, label)
        (512, 512, 256, 256, "512→256 (2×)"),
        (512, 512, 128, 128, "512→128 (4×)"),
        (2048, 2048, 128, 128, "2048→128 (16×)"),
        (4032, 3024, 128, 128, "4032×3024→128 (photo)"),
        (512, 512, 1024, 1024, "512→1024 (2× up)"),
    ]
    orders = [3, 4, 5]

    print(f"\n{'Config':<30} {'Order':>5} {'Time (ms)':>10} {'Throughput':>14}")
    print("-" * 65)

    for in_h, in_w, out_h, out_w, label in configs:
        tensor = torch.randn(1, 3, in_h, in_w, device=device)
        for order in orders:
            # Warm up this specific config.
            lanczos.resize(tensor, out_h, out_w, order=order)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)

            n_iter = 20
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            for _ in range(n_iter):
                lanczos.resize(tensor, out_h, out_w, order=order)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - t0) / n_iter * 1000  # ms

            mpix = in_h * in_w / 1e6
            throughput = mpix / (elapsed / 1000)
            print(f"{label:<30} {order:>5} {elapsed:>9.2f}  {throughput:>10.1f} MP/s")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visual test for Lanczos resize filter. "
                    "Does nothing unless --outdir, --show, or --bench is given.")
    parser.add_argument("--outdir", default=None,
                        help="Output directory for generated PNGs")
    parser.add_argument("--show", action="store_true",
                        help="Generate, save to /tmp/lanczos_demo, and open in xviewer")
    parser.add_argument("--bench", action="store_true",
                        help="Run performance benchmarks")
    parser.add_argument("--sizes", default="256,128,64",
                        help="Comma-separated target sizes (default: 256,128,64)")
    parser.add_argument("--orders", default="3",
                        help="Comma-separated Lanczos orders to test (default: 3)")
    parser.add_argument("--device", default=None,
                        help="Torch device (default: cuda if available, else cpu)")
    args = parser.parse_args()

    if not args.outdir and not args.show and not args.bench:
        parser.print_help()
        print("\nNothing to do. Specify --outdir, --show, or --bench.")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.bench:
        run_benchmark(device)
        # Also benchmark internal GPU if eGPU is present.
        if torch.cuda.device_count() > 1 and args.device is None:
            for i in range(torch.cuda.device_count()):
                if i != torch.cuda.current_device():
                    run_benchmark(f"cuda:{i}")

    outdir = Path(args.outdir) if args.outdir else (Path("/tmp/lanczos_demo") if args.show else None)
    if outdir is None:
        return

    outdir.mkdir(exist_ok=True)
    sizes = [int(s) for s in args.sizes.split(",")]
    orders = [int(o) for o in args.orders.split(",")]

    print(f"\nDevice: {device}")
    print(f"Orders: {orders}")
    print(f"Sizes:  {sizes}")

    patterns = make_patterns()

    for name, img in sorted(patterns.items()):
        Image.fromarray(img).save(outdir / f"{name}.png")
        t = to_tensor(img, device)
        for order in orders:
            for s in sizes:
                result = lanczos.resize(t, s, s, order=order)
                downscaled = from_tensor(result)
                suffix = f"_L{order}_{s}" if len(orders) > 1 else f"_lanczos_{s}"
                Image.fromarray(downscaled).save(outdir / f"{name}{suffix}.png")
        print(f"  {name}: done")

    total = len(list(outdir.glob("*.png")))
    print(f"\n{total} images saved to {outdir}/")

    if args.show:
        # Show radial sweep comparison (most informative for filter evaluation).
        for name in ["radial_sweep", "freq_sweep_h", "spokes", "text_and_shapes"]:
            subprocess.Popen(["xviewer", str(outdir / f"{name}.png")])
            for order in orders:
                suffix = f"_L{order}_128" if len(orders) > 1 else "_lanczos_128"
                path = outdir / f"{name}{suffix}.png"
                if path.exists():
                    subprocess.Popen(["xviewer", str(path)])


if __name__ == "__main__":
    main()
