#!/usr/bin/env python
"""Render a still through a postprocessor filter at several settings, side by side, for eyeballing.

The postprocessor's tests are contract tests: they say the output has the right shape, stays in range,
and modulates what it claims to modulate. What they cannot say is whether it *looks* right, and the
gap between those two is where this module's real defects have lived. The `crt` filter passed every
contract test while emitting a washed-out, half-transparent character, because it applied its scanline
term to the colour and to alpha both, which squares it in a straight-alpha frame. A rendered still is
what showed that; nothing else would have.

Run from the project root with the venv activated:

    python -m raven.common.video.tests.preview_postprocessor crt          # a labelled contact sheet
    python -m raven.common.video.tests.preview_postprocessor crt --crop   # 1:1 crops of the head

Not collected by pytest (no `test_` prefix), and not shipped as a console script: this is a bench
instrument like `bench_postprocessor.py` beside it.

**Judge fine structure at 1:1, never on a contact sheet.** A raster at the pixel pitch survives neither
the sheet's tiling nor a viewer's downscaling: one bright row and one dark row average into a uniform
haze, and the filter gets blamed for what the resampling did. `--crop` exists for that reason.
"""

import argparse
import pathlib

import numpy as np
from PIL import Image, ImageDraw

import torch

from ..colorspace import linear_to_srgb, srgb_to_linear
from ..postprocessor import Postprocessor

# A character with a wide tonal range and a strongly coloured area (the hair), which is what makes a
# colour shift legible. Any RGBA image with a transparent background works.
DEFAULT_SOURCE = "raven/avatar/assets/characters/other/aria1.png"
DEFAULT_CROP = (150, 30, 370, 250)  # the head: skin and saturated hair side by side
BACKDROP = 0.08  # what the client composites behind the avatar, roughly

# Per filter, a list of (label, settings) to render. The first entry should switch the filter off, so
# there is an untouched reference in the sheet to compare against.
VARIANTS = {
    "atmospheric_dust": [
        ("off", dict(count=0)),
        ("defaults", dict()),
        ("sharp (aperture 0)", dict(aperture=0.0)),
        ("wide aperture 14", dict(aperture=14.0)),
        ("dense, count 800", dict(count=800)),
        ("soft shimmer, exp 6", dict(glint_exponent=6.0)),
        ("sharp flashes, exp 120", dict(glint_exponent=120.0)),
        ("focus racked to 0.35", dict(focal_plane=0.35)),
    ],
    "crt": [
        ("off", dict(scanline_strength=0.0, mask_type="none", corner_falloff=0.0,
                     beam_bleed=0.0, glow_strength=0.0, persistence_tau=0.0,
                     brightness_compensation=0.0)),
        ("defaults", dict()),
        ("no compensation", dict(brightness_compensation=0.0)),
        ("full compensation", dict(brightness_compensation=1.0)),
        ("period 4, weight 4", dict(scanline_period=4, scanline_weight=4.0)),
        ("slot mask, pitch 6", dict(mask_type="slot", mask_pitch=6)),
        ("shadow mask, pitch 6", dict(mask_type="shadow", mask_pitch=6)),
        ("warped + overscan", dict(warp_x=0.03, warp_y=0.05, overscan=1.05)),
    ],
}


def render(filter_name, label_and_settings, source, crop=None):
    """Run one filter at each of several settings, and return `[(label, HxWx3 uint8), ...]`.

    Each result is gamma-corrected and composited over a dark backdrop, as the client does, so what
    comes back is what a viewer would see rather than the linear-light RGBA the filter works in.
    """
    image = Image.open(source).convert("RGBA")
    w, h = image.size
    linear = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).permute(2, 0, 1)
    linear[:3] = srgb_to_linear(linear[:3])

    tiles = []
    for label, settings in label_and_settings:
        # A fresh Postprocessor per variant, so a stateful filter cannot carry a trail between them.
        pp = Postprocessor("cpu", torch.float32, chain=[])
        pp._setup_meshgrid(h, w)
        pp.frame_no = 0.0
        pp.last_frame_no = -1.0

        frame = linear.clone()
        getattr(pp, filter_name)(frame, **settings)

        rgb, alpha = frame[:3], frame[3:4]
        shown = linear_to_srgb(rgb) * alpha + BACKDROP * (1.0 - alpha)
        tile = (shown.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        if crop is not None:
            x0, y0, x1, y1 = crop
            tile = tile[y0:y1, x0:x1]
        tiles.append((label, tile))

        clipped = float((rgb >= 0.999).any(dim=0).float().mean())
        print(f"  {label:24s} light {float((rgb * alpha).mean()):.4f}  "
              f"alpha {float(alpha.mean()):.3f}  clipped {clipped:6.2%}")
    return tiles


def contact_sheet(tiles, columns=4):
    """Lay the tiles out in a grid, each labelled."""
    th, tw = tiles[0][1].shape[:2]
    pad = 18
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tw, rows * (th + pad)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, (label, tile) in enumerate(tiles):
        x, y = (i % columns) * tw, (i // columns) * (th + pad)
        draw.text((x + 4, y + 4), label, fill=(255, 255, 0))
        sheet.paste(Image.fromarray(tile), (x, y + pad))
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("filter", help=f"filter to preview; canned variants exist for: {', '.join(VARIANTS)}")
    parser.add_argument("-o", "--out", default="/tmp/postprocessor_preview.png", help="where to write the sheet")
    parser.add_argument("-s", "--source", default=DEFAULT_SOURCE, help="RGBA source image")
    parser.add_argument("--crop", action="store_true",
                        help="render 1:1 crops instead of whole frames. Use this to judge anything at "
                             "the pixel pitch — a downscaled sheet turns a raster into a flat haze.")
    args = parser.parse_args()

    if args.filter not in VARIANTS:
        parser.error(f"no canned variants for '{args.filter}'; add a list to VARIANTS in {pathlib.Path(__file__).name}")

    print(f"{args.filter}, from {args.source}:")
    tiles = render(args.filter, VARIANTS[args.filter], args.source,
                   crop=DEFAULT_CROP if args.crop else None)
    sheet = contact_sheet(tiles)
    sheet.save(args.out)
    print(f"wrote {args.out} {sheet.size}")


if __name__ == "__main__":
    main()
