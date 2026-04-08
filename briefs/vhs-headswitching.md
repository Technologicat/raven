# Brief: VHS head switching noise

## Goal

Add a head switching noise filter to Raven's avatar postprocessor. This is
the characteristic squiggly/displaced bottom edge visible on VHS playback —
the most immediately recognizable VHS artifact after tracking bars.

Head switching noise appears on **all** VHS tapes. CRT televisions hid it
behind overscan; it became visible when capturing VHS to digital at full
raster. It is the artifact that most immediately says "this is a VHS tape."


## Design rationale

The avatar settings editor currently uses a fixed filter chain (no
insert/remove/reorder), although it does autodiscover filter parameters.
This is a new filter, not an extension of an existing one — it is
a visually distinct artifact, not a variant of something we already have.

### How this relates to existing filters

Three existing filters are related but distinct:

1. **`analog_vhstracking`**: simulates *deck-to-deck tracking mismatch* —
   when a tape recorded on one VHS deck is played on another and the helical
   scan tracks don't align. The result is the image floating up/down and a
   band of **pure noise** at the bottom. The noise replaces the image
   entirely in the affected rows. This is a compatibility issue between
   decks.

2. **`analog_rippling_hsync`**: simulates *horizontal sync instability* —
   each scanline arrives at a slightly wrong time, producing a ripple/wave
   distortion across the full height of the image. Caused by signal
   degradation (e.g. a long cable). Affects the entire frame uniformly.

3. **`analog_runaway_hsync`**: a cousin of rippling hsync, confined to one
   edge (top or bottom) with a spatial decay profile. Simulates a more
   severe, localized sync failure.

The new filter simulates **head switching**: the physical moment when one
rotary video head disengages from the tape and the other engages, mid-frame.
The transition is not instantaneous, so the bottom few scanlines contain
*displaced image content* — you can still partly see what's supposed to be
there, but it's horizontally sheared and mangled. This is distinct from
tracking noise (which is pure static replacing the image) and from hsync
ripple (which affects the whole frame with small, smooth oscillations).


## What exists

All paths relative to repo root.

### Infrastructure in `raven/common/video/postprocessor.py`

- **`_meshx`, `_meshy`**: base coordinate grids for `grid_sample`-based
  geometric distortions, range [−1, +1] in both axes. Rebuilt when the
  image size changes (line ~472).

- **`grid_sample` pattern**: used by `analog_rippling_hsync`,
  `analog_runaway_hsync`, `analog_vhstracking`, and the zoom filter.
  The pattern is always:
  ```python
  grid = torch.stack((meshx, meshy), 2)
  grid = grid.unsqueeze(0)
  image_batch = image.unsqueeze(0)
  warped = torch.nn.functional.grid_sample(
      image_batch, grid, mode="bilinear",
      padding_mode="border", align_corners=False
  )
  warped = warped.squeeze(0)
  image[:, :, :] = warped
  ```

- **`_vhs_noise(image, *, height)`**: generates monochrome VHS noise band
  (PAL-style, `[1, h, w]`). Used by `analog_vhsglitches` and
  `analog_vhstracking`.

- **`vhs_noise(width, height, *, device, dtype, mode)`**: the module-level
  single source of truth for VHS noise texture. `mode="PAL"` returns
  `[1, H, W]`, `mode="NTSC"` returns `[3, H, W]`.

- **`@with_metadata` decorator**: declares parameter ranges for the settings
  editor GUI, and `_priority` for filter chain ordering.

- **`_kernel_size`**: the standard kernel size constant used across all
  Gaussian blur calls in the postprocessor.

### Priority ordering

Current filter priorities (lower = earlier in chain):
```
noise           4.0
analog_lowres   5.0
analog_rippling_hsync  6.0
analog_runaway_hsync   7.0
analog_vhsglitches     8.0
analog_vhstracking     9.0
digital_glitches       10.0
scanlines              (check current value)
```

Head switching should run near `analog_vhstracking` — both are VHS playback
artifacts. Suggest priority **8.5** (after glitches, before tracking), or
wherever fits best in the chain. The exact value doesn't matter much since
both head switching and tracking affect the bottom of the frame, but head
switching should probably come first so that tracking noise can overwrite
the head-switching region when both are active.


## What to add

### New filter: `analog_vhs_headswitching`

A per-scanline horizontal displacement applied to the bottom N rows of the
image, simulating the physical head switching interval.

#### Parameters

- **`height`**: height of the affected region, as a fraction of image
  height (consistent with other filter parameters — agnostic to output
  resolution, and matches how humans perceive the effect visually).
  Suggested default: ~0.01–0.02 (roughly 6–10 pixels at typical render
  sizes; on real VHS this is roughly 5–8 scanlines out of 576 PAL / 480
  NTSC).

- **`max_displacement`**: maximum horizontal shift, in the same units as
  `analog_rippling_hsync` (i.e. where image width = 2.0). The displacement
  should be large enough to be clearly visible — much larger than the ripple
  hsync amplitudes (those are ~0.001). Suggested default: 0.02–0.05.

- **`noise_blend`**: maximum fraction of VHS noise mixed into the displaced
  region at the very bottom. The blend ramps from 0 at the top of the
  affected band to `noise_blend` at the bottom — same envelope shape as
  the displacement. At the top: mostly displaced-but-recognizable image.
  At the bottom: mostly static. 0.0 = pure displaced image everywhere,
  1.0 = pure noise at the bottom edge. Suggested default: 0.3–0.5.

- **`speed`**: animation speed, same convention as other analog filters
  (cycle position updates by `speed / image_height` per frame). The head
  switching region doesn't move much in normal playback, but it can "bounce"
  slightly — the height and displacement vary slowly frame-to-frame.

#### Displacement profile

The key visual characteristic: **each scanline in the affected region is
shifted by a different horizontal amount**, and the displacement increases
toward the bottom of the frame. The profile should be:

1. **Monotonically increasing** from the top of the affected region (where
   it blends with the clean image) to the bottom (maximum displacement).
2. **Not smooth** — adjacent scanlines are correlated but not identical.
   This is what makes it look "squiggly" rather than "smoothly warped."

The exact shape is a tuning detail — get the engineering in place first,
then tweak the profile by eye. Two approaches to try:

**Option A (deterministic, preferred starting point):** superpose 2–3 sine
waves at different frequencies, with the input depending on both y position
and time (both linearly). Multiply by an increasing envelope (ramp from 0
to `max_displacement`). This gives correlated-looking wobble without needing
cached state or true randomness — same philosophy as
`analog_rippling_hsync`.

**Option B (stochastic):** generate white noise of length `height`, blur
with a small 1D Gaussian (σ ≈ 1.5–2.0 scanlines), multiply by the same
increasing envelope. Cache and re-randomize every N frames (like
`analog_vhsglitches`).

Option A is simpler and avoids caching. Start there; switch to B only if
the deterministic version looks too regular.

#### Implementation

Use the same `grid_sample` infrastructure as the other geometric filters.

- Copy `_meshx` and `_meshy`.
- For the bottom `height` rows of `meshx`, add the displacement profile
  (broadcast across the x dimension — the displacement is constant along
  each scanline, which is correct for horizontal shift).
- For rows above the affected region, leave `meshx` untouched.
- Apply `grid_sample` with `padding_mode="border"` (displaced pixels that
  read past the edge repeat the border, which looks reasonable).
- After warping, blend VHS noise into the affected region: use
  `vhs_noise()` directly (see cleanup note below) and lerp with
  the same increasing envelope used for displacement, scaled by
  `noise_blend`.

#### Animation

With Option A (sine-based), animation comes for free — the time dependence
in the sine arguments makes the wobble pattern drift naturally, same as
`analog_rippling_hsync`. The `speed` parameter controls how fast the pattern
evolves.

With Option B (stochastic), use a hold pattern like `analog_vhsglitches`:
re-randomize every N frames.

#### Interaction with `analog_vhstracking`

Both filters affect the bottom of the frame. When both are active:
- Head switching runs first (lower priority number), displacing the bottom
  rows.
- Tracking then shifts the entire image vertically and overwrites the very
  bottom with noise.

At typical settings the tracking noise band is larger than the head switching
region, so the tracking filter will mostly overwrite the head switching
effect. This is physically correct — on real VHS, both artifacts are present
but tracking noise dominates when tracking is bad. When tracking is good
(small `base_offset`), head switching becomes visible.


## What's NOT like the existing code

1. **Per-scanline displacement confined to a narrow band.** The existing
   hsync filters modify `meshx` for the entire frame. Head switching
   modifies it only for the bottom N rows. The implementation needs to
   construct a modified mesh that leaves the rest of the image untouched.

2. **Mixed warping + noise in the same region.** The existing filters
   either warp OR add noise — not both. This is coincidence, not a design
   constraint. Head switching needs both: displaced image content blended
   with static. The noise blend should happen after the `grid_sample` warp.


## Visual references

- **AV Artifact Atlas** (BAVC):
  http://www.avartifactatlas.com/artifacts/head_switching_noise.html
  — canonical reference with example clips.

- **ntsc-rs** (https://github.com/valadaptive/ntsc-rs): Rust-based NTSC
  simulator that includes head switching. Good for visual comparison.

- **ntscQT+** (https://github.com/rgm89git/ntscQTplus): Python/OpenCV
  NTSC simulator, also includes head switching.


## Testing

- Visual: with head switching enabled and tracking disabled/minimal, the
  bottom band should show horizontally displaced, wobbly image content
  partially mixed with noise. It should look like the bottom edge of the
  picture is being chewed.
- With both head switching and tracking active at typical settings, the
  tracking noise should mostly overwrite the head switching region. Turn
  tracking's `base_offset` down to near zero to reveal head switching.
- The displacement should be clearly larger than `analog_rippling_hsync`'s
  effect (which is subtle edge ripple).
- Animation: the wobbly pattern should drift smoothly over time (not flicker
  every frame, not be static).
- Performance: this is a `grid_sample` operation on the full image (same as
  the other geometric filters), so it should not be a bottleneck.


## Integration points

### Avatar settings editor

The `@with_metadata` decorator's `_priority` field may be sufficient to
add this filter to the editor's chain automatically — verify this. If not,
check how the existing analog filters are wired in and add this one at
priority 8.5 (between vhsglitches and vhstracking).

### Remove `_vhs_noise` private method

`_vhs_noise` is a one-line wrapper that redirects to the module-level
`vhs_noise` function. It's a leftover from before `vhs_noise` was
extracted as the single source of truth. Remove `_vhs_noise` and update
its call sites (`analog_vhsglitches`, `analog_vhstracking`, and this new
filter) to call `vhs_noise()` directly.


## Out of scope

- **Flagging / skew error**: a different artifact affecting the *top* of the
  frame, caused by tension problems. Visually it's a horizontal shearing of
  the top few lines. Could be a separate filter later.
- **NTSC vs. PAL differences in head switching**: the physical mechanism is
  the same on both systems. No mode switch needed.
- Interaction with the NTSC chroma noise — head switching noise is
  monochrome (it's reading between tracks, not valid chroma data).
