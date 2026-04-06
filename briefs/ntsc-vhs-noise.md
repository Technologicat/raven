# Brief: NTSC VHS noise mode

## Goal

Add an NTSC noise mode alongside the existing (PAL) VHS noise in Raven's
postprocessor and in raven-cherrypick's placeholder tiles.

This is a second-order easter egg. The `noise` filter in the avatar
postprocessor chain gains `"VHS_PAL"` and `"VHS_NTSC"` channel options,
autodiscovered by the settings editor. Currently VHS noise only appears
internally in the `analog_vhsglitches` and `analog_vhstracking` filters;
this exposes full-image VHS noise as a user-facing option for the first time.
Cherrypick's unique VHS noise placeholder thumbnails gain a similar PAL/NTSC
mode switch.

**Scope:** noise texture generation and the `noise` filter only. The other
analog VHS filters (`analog_vhsglitches`, `analog_vhstracking`,
`analog_rippling_hsync`, `analog_lowres`) are out of scope for this task.


## Design rationale

The avatar settings editor currently uses a fixed filter chain (no
insert/remove/reorder), although it does autodiscover filter parameters.
This is why NTSC noise is a `channel` option on `noise`, not a separate
filter.


## What exists

All paths relative to repo root.

### `raven/common/video/postprocessor.py`

- **`vhs_noise(width, height, *, device, dtype)`** (module-level, line ~44):
  Single source of truth for VHS noise texture. Returns `[1, H, W]` tensor
  in [0, 1]. Gaussian blur kernel (5, 1), σ = 2.0 — horizontal runs, sharp
  vertical transitions. Monochrome only (PAL-correct).

- **`vhs_noise_pool(n, width, height, *, device, dtype, tint, brightness)`**
  (module-level, line ~64): Generates *n* tinted RGBA tiles for cherrypick
  placeholders. Takes the monochrome `vhs_noise` output, maps it through a
  brightness range, multiplies by a per-channel tint, adds alpha = 1.0.
  Returns list of `[4, H, W]` tensors.

- **`Postprocessor._vhs_noise(image, *, height)`** (line ~923): Thin wrapper
  calling `vhs_noise()`. Used by `analog_vhsglitches` and
  `analog_vhstracking`. Returns `[1, h, w]`.

- **`Postprocessor.noise(image, *, strength, sigma, channel, name)`**
  (line ~712): The general noise filter. `channel` accepts `"Y"` (luminance
  via RGB↔YUV round-trip) or `"A"` (alpha). Multiplicative:
  `out = in * ((1 - strength) + strength * noise)`. Isotropic Gaussian blur
  on noise (same σ in both axes). The metadata decorator at line ~709 declares
  `channel=["Y", "A"]` — this is what the settings editor GUI uses to build
  the dropdown.

### `raven/common/video/colorspace.py`

BT.709 (HDTV) YCbCr conversion. `rgb_to_yuv` / `yuv_to_rgb` operate on
`[C, H, W]` tensors. Y in [0, 1], Cb/Cr in [−0.5, +0.5]. Already used by
the `noise` filter in `"Y"` mode.

### `raven/cherrypick/app.py`

- **`_generate_noise_pool(tile_size)`** (line ~934): Calls `vhs_noise_pool`,
  converts linear→sRGB, flattens to DPG format. Called at startup and on
  tile-size change.


## What to add

### 1. Extend `vhs_noise` with a `mode` parameter

```python
def vhs_noise(width: int, height: int, *,
              device: torch.device,
              dtype: torch.dtype = torch.float32,
              mode: str = "PAL") -> torch.Tensor:
```

- **`mode="PAL"`** (default): unchanged behavior. Returns `[1, H, W]`.
- **`mode="NTSC"`**: returns `[3, H, W]` — three independent noise planes
  for Y, U, V respectively.

NTSC noise recipe (see §"How NTSC noise differs" below):

1. **Y channel** (index 0): identical to PAL — `torch.rand` + GaussianBlur
   kernel (5, 1), σ = 2.0.
2. **U channel** (index 1): independent `torch.rand`, blurred with a *wider*
   horizontal-only kernel to simulate the lower chroma bandwidth. Suggested:
   kernel (11, 1), σ = 4.0 (wider, smoother horizontal runs than Y). The
   anisotropy is horizontal for the same reason as Y — helical scan geometry.
   Chroma just has lower bandwidth, so the runs are coarser. Amplitude scaled
   down — maybe multiply by 0.5 before returning, so the caller's `strength`
   parameter still controls overall intensity.
3. **V channel** (index 2): same recipe as U, independently randomized.

The asymmetry — Y gets fine-grained horizontal runs, U/V get coarser blobs —
is the visual signature we're after.

### 2. Extend `vhs_noise_pool` with a `mode` parameter

```python
def vhs_noise_pool(n, width, height, *,
                   device, dtype,
                   tint=(0.92, 0.92, 1.0),
                   brightness=(0.04, 0.40),
                   mode="PAL") -> List[torch.Tensor]:
```

- **PAL**: unchanged — monochrome luma × tint.
- **NTSC**: use the 3-channel noise. The U/V noise planes add *color
  variation* to each tile instead of uniform tinting. Recipe:
  - Y plane → mapped through `brightness` range as before → this is the luma.
  - U, V planes → scaled to a small range (e.g. ±0.08) and added to zero-chroma.
  - Combine as YUV `[3, H, W]`, convert to RGB via `yuv_to_rgb`, apply `tint`,
    add alpha, clamp.

This gives each NTSC placeholder tile its own subtle color variation — some
slightly warm, some slightly cool — instead of being uniformly blue-gray.

### 3. Add `"VHS_PAL"` and `"VHS_NTSC"` channels to the `noise` filter

In `Postprocessor.noise`, extend the `channel` parameter:

```python
channel=["Y", "A", "VHS_PAL", "VHS_NTSC"],
```

Implementation for the new channels:

- **`"VHS_PAL"`**: Convert to YUV. Generate PAL-mode noise (1-channel)
  using the anisotropic blur from `vhs_noise` — kernel (5, 1), σ = 2.0.
  Apply multiplicatively to Y only. Note: the existing `"Y"` and `"A"`
  modes use *isotropic* Gaussian blur intentionally (they're general-purpose
  noise, not VHS-specific). The VHS modes are a new anisotropic variant.

- **`"VHS_NTSC"`**: Convert to YUV. Generate NTSC-mode noise (3-channel).
  Apply Y noise multiplicatively to the Y channel (same as PAL). Apply U/V
  noise *additively* to the Cb/Cr channels — chroma noise isn't a darkening
  effect, it's a random color shift. The `strength` parameter still controls
  overall magnitude. Convert back to RGB.

  The additive chroma formula:
  ```
  image_yuv[1] += strength * chroma_amplitude * noise_u
  image_yuv[1].clamp_(-0.5, 0.5)
  image_yuv[2] += strength * chroma_amplitude * noise_v
  image_yuv[2].clamp_(-0.5, 0.5)
  ```
  where `chroma_amplitude` is a constant (suggest 0.15–0.25, tune by eye).
  The Y channel still gets the multiplicative treatment. Clamping in YUV
  space avoids surprising color shifts from out-of-gamut values propagating
  through `yuv_to_rgb`.

### 4. Update `_vhs_noise` (optional, out of scope for `noise` filter)

For now, `_vhs_noise` stays PAL-only (monochrome). The `analog_vhsglitches`
and `analog_vhstracking` filters don't need NTSC chroma noise — their visual
effect is dominated by the spatial disruption, not the noise color. This can
be revisited later if desired.


## How NTSC noise differs from PAL — the technical picture

PAL's "Phase Alternating Line" system inverts the phase of one chroma
component on alternate scanlines. When the decoder averages adjacent lines,
phase errors cancel out. The result: PAL VHS noise is overwhelmingly in
luminance. Chroma is stable.

NTSC has no such cancellation. The chroma subcarrier phase encodes hue
directly, so any phase perturbation from tape speed variation, head switching,
or magnetic noise becomes a *hue error*. This is the "Never The Same Color"
phenomenon. Visually:

- **Chroma speckling**: random color flecks in dark/saturated areas. Looks
  like faint rainbow confetti overlaid on the image.
- **Hue wander**: slow, low-frequency tint drift. Per-frame or per-scanline.
  (We approximate this with the low-spatial-frequency U/V noise runs.)
- **Color bleed**: horizontal chroma smearing. VHS chroma bandwidth is ~500 kHz
  vs. ~3 MHz for luma. This is already partly covered by `analog_lowres` if
  applied; the noise filter adds the *random* component.

For our purposes, the coarse-run additive U/V noise captures the essential
visual character without needing to simulate the actual QAM demodulation
chain. It's an artistic approximation — same philosophy as the existing PAL
noise.

### Parameters to tune by eye

| Parameter | Suggested start | What it controls |
|---|---|---|
| Y blur kernel | (5, 1), σ = 2.0 | Horizontal run length (same as PAL) |
| U/V blur kernel | (11, 1), σ = 4.0 | Chroma horizontal run width (coarser = more realistic) |
| U/V amplitude scale | 0.5 | Raw noise amplitude before `strength` | 
| Chroma additive amplitude | 0.20 | How much color shift per unit `strength` |

These should be tuned against actual NTSC VHS footage screenshots. The
`ntsc-rs` project (https://github.com/valadaptive/ntsc-rs) and `ntscqt`
(https://github.com/JargeZ/ntscqt) are good visual references.


## What's NOT like the existing code

1. **3-channel noise return.** `vhs_noise` currently always returns
   `[1, H, W]`. In NTSC mode it returns `[3, H, W]`. Every call site that
   currently does `.squeeze(0)` to get `[H, W]` must be aware of this.
   Since we're only changing `vhs_noise_pool` and the `noise` filter (not
   `_vhs_noise`), the blast radius is contained — but document it clearly.

2. **Additive chroma noise.** The existing `noise` filter is purely
   multiplicative. The NTSC chroma component is *additive* in UV space.
   This is a different formula path inside the filter, not just a different
   blur kernel.

3. **Two different blur kernels in one noise texture.** The Y and U/V planes
   need different σ values. Generate them independently — don't try to blur
   a single noise field twice.

4. **The `vhs_noise_pool` NTSC path needs `colorspace.py`.** The PAL path
   never touches YUV — it's monochrome luma × tint. The NTSC path constructs
   a YUV image and converts to RGB. Import `yuv_to_rgb` from
   `raven.common.video.colorspace`.


## Integration points

### Avatar settings editor

The `noise` filter's `channel` dropdown is built from the `@with_metadata`
decorator. It will pick up `"VHS_PAL"` and `"VHS_NTSC"` automatically once
they're added to the list. No GUI code changes needed.

### Cherrypick

`_generate_noise_pool` in `raven/cherrypick/app.py` needs a way to select
the mode. Add a config entry in `raven/cherrypick/config.py`:

```python
# VHS noise standard for placeholder tiles.
# "PAL" (Phase Alternating Line) — stable color, luma noise only.
#     Standard in most of Europe, including Finland.
# "NTSC" (National Television System Committee) — adds chroma noise.
#     Standard in North America and Japan. Sometimes referred to by
#     field engineers as "Never The Same Color" due to the absence of
#     phase-alternation error correction in the chrominance channel.
PLACEHOLDER_VHS_MODE = "PAL"
```

Pass it through to `vhs_noise_pool(... mode=config.PLACEHOLDER_VHS_MODE)`
in `_generate_noise_pool`.


## Testing

- Visual comparison: render a few frames with `"VHS_PAL"` and `"VHS_NTSC"`
  side by side. PAL should look like the current output. NTSC should have
  visible color speckling — faint but noticeable rainbow-ish noise on top
  of the luma noise.
- Cherrypick: generate a grid of placeholder tiles in both modes. PAL tiles
  should be uniformly blue-gray (as now). NTSC tiles should each have
  subtly different color casts.
- Backward compatibility: existing configs that use `channel="Y"` or
  `channel="A"` must work unchanged. The default `mode="PAL"` in
  `vhs_noise` preserves the current behavior.
- Tensor shapes: verify that `vhs_noise(..., mode="PAL")` returns
  `[1, H, W]` and `mode="NTSC"` returns `[3, H, W]`.


## Out of scope

- Dot crawl / rainbow artifacts (composite video cross-luminance/cross-chroma).
  These are demodulation artifacts, not tape noise. Different filter entirely.
- Head switching noise at frame bottom (already handled by `analog_vhstracking`).
- Audio artifacts.
- Actual QAM simulation.
- SECAM (nobody asked, and "Système Essentially Contrary to the American Method"
  is a third-order easter egg we can save for later).
