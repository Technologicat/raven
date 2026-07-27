# Brief: `crt` — raster projection simulation

Target file: `briefs/crt-display.md`

## 0. Prerequisite: give `_priority` a meaning

Currently `_priority` is an ordering hint with no stated semantics, which is why
adding a filter raises "what number?" as an open question every time. It turns
out the existing numbers are already *almost* a scheme — the section headings in
`postprocessor.py` line up with the numeric ranges. Codify it, and no existing
filter needs renumbering.

**Reference point: `0.0` is the moment of capture** — light entering the taking
lens. Everything before that exists in the world; everything after is signal.

| band | range | section heading | meaning | current members |
|---|---|---|---|---|
| **Scene** | < 0 | *Physical input signal* | what exists in front of the lens, and how the camera is aimed at it | `zoom` (−1.0) |
| **Capture** | 0 – 5 | *Video camera*, *Retouching / color grading*, *General use* | optics, sensor, grading | `bloom` (0.0), `chromatic_aberration` (1.0), `noise` (1.5), `vignetting` (2.0), `desaturate` (3.5) |
| **Signal** | 5 – 10 | *Lo-fi analog video* | everything between sensor and display | `analog_lowres` (5.0) … `digital_glitches` (10.0) |
| **Display** | ≥ 10 | *Display output* | the viewer's screen | `translucent_display` (10.5), `monochrome_display` (11.0), `banding` (12.0), `scanlines` (13.0) |

`bloom` at exactly 0.0 is then meaningful rather than incidental: it is the
first thing the taking lens does. And `zoom` at −1.0 stops reading as an
oversight — framing is the decision about what is in the scene at all, so it
belongs in the Scene band by construction, not by accident.

### What `_priority` actually governs

Worth stating in the same comment block, because it is easy to over-read.
`render_into` applies the chain **positionally** —
`for filter_name, settings in chain: getattr(self, filter_name)(image, **settings)`
(line 582). `_priority` is consumed only by `get_filters`, to sort the
settings-editor panels. So the bands are a *convention for the GUI's default
ordering*, not a constraint the engine enforces. A hand-written
`postprocessor_defaults` or `animator.json` can put any filter anywhere, and
can list the same filter twice — the `name` parameter on every caching filter
exists precisely so multiple instances key their caches apart.

The backend has therefore always supported multiple instances at arbitrary
positions. What is missing is the add/remove/reorder GUI, which is why
`strip_postprocessor_chain_for_gui` collapses the chain with
`dict(postprocessor_chain)` and keeps one entry per filter name. Until that GUI
exists, treat one-instance-per-filter as a GUI limitation being worked around,
not as a property of the postprocessor.

Land this as a comment block above the first filter definition, and add a
one-line reminder in the `with_metadata` docstring. It is a documentation
change plus a convention, no code motion. Split it into its own commit; it is
independently useful and should not be buried in a feature.

New assignments under the scheme:

| filter | priority | band |
|---|---|---|
| `backdrop` (future, see §5) | −10.0 | Scene — literally the furthest thing away |
| `crt` (this brief) | −3.0 | Scene |
| `atmospheric_dust` (companion brief) | −2.0 | Scene |

## 1. Two diegetic models

The avatar frame is RGBA with a transparent background; the backdrop image is
composited **client-side** in `raven/client/avatar_renderer.py`
(`raven/server/modules/avatar.py:925` — "The backdrop image is applied at the
client end"). The server-side postprocessor therefore only ever sees the
character plus alpha, never the backdrop.

That picks out one of two coherent models, and it is the one the system was
built for:

| | backdrop client-side (current) | backdrop server-side (v2) |
|---|---|---|
| model | avatar is a **hologram** projected into a physical scene | avatar is content on a **virtual display device** |
| raster applies to | the projected character only | the whole picture, backdrop included |
| scanline gaps | transparent — the room shows through | dark — the tube is off there |
| barrel warp | no glass, so ~none | meaningful, bends the whole raster |
| band | Scene (−3.0) | Display (14.0) |
| status | **ships for Researchers' Night** | wanted, §5 |

`translucent_display` (10.5) already sits in `postprocessor_defaults`, which is
the same hologram intuition showing up in the existing chain.

## 2. Placement: the hologram is *in the world*

Routing principle: in the world, or on the display? A hologram is an object in
the scene. The camera films it, so the raster passes through the capture optics
like everything else. Priority **−3.0**:

```
character → crt (raster) → atmospheric_dust (room air)
          → zoom (framing) → bloom / chromatic_aberration / vignetting (capture)
          → …signal… → …display…
```

Three consequences, each of which settles a question that was open before the
diegetic model was fixed:

1. **Dust does not get scanlined.** Room air is added after the projection.
2. **Emission is free.** The capture `bloom` downstream blooms the scanlines,
   because they are bright things in frame. No internal glow pass is needed to
   sell it — `glow_strength` defaults to 0.0 *for this reason*, not as a
   caution about muddiness.
3. **`scanlines` (13.0) is not obsoleted.** In the Display band it is the
   *viewer's* monitor, a different diegetic layer from the hologram's own
   raster. They can coexist. Leave it untouched; do not describe this filter
   as superseding it.

**Cost, and the fallback.** Placing the filter early means any downstream
resample softens the raster. In practice only `zoom` resamples, and `zoom` is
**not in `postprocessor_defaults`** — the default chain is unaffected. If a
zoomed configuration softens the scanlines objectionably, the fallback is
−0.5 (still Scene band, after `zoom`): crisp raster, still free bloom, at the
cost of scanlined dust. Do not implement a switch; pick −3.0 and note the
alternative in a comment.

## 3. Design

### Monolithic, not composable

Warp, scanline, and mask ship as **one filter** with many parameters, not three
chain entries. Two independent arguments:

1. **Technical.** Warp is a resample. As separate stages, the scanline and mask
   stages would sample the warped image on the output grid, and every
   subsequent resample softens the high-frequency structure that *is* the
   effect. Fusing means one `grid_sample`, with scanline and mask evaluated
   analytically at known coordinates.
2. **Ergonomic.** A raster display is one thing to configure. Three entries in
   the settings editor means three sections that must be kept mutually
   consistent, for no benefit — you never want "mask without the rest".

Monolithic ≠ opaque. Every sub-effect stays exposed as a parameter.

### One filter, locked position

**Correcting an earlier version of this brief**, which proposed a private
`_crt_impl` with two thin public wrappers (`crt_hologram` at −3.0,
`crt_display` at 14.0) on the reasoning that a fixed `_priority` prevents one
filter from occupying two bands. That reasoning was wrong: the engine dispatches
positionally and `_priority` only sorts the GUI (see §0). Nothing was ever
blocked.

Ship **one filter, `crt`, at `_priority = -3.0`**, and do not build a mode
system. Everything that distinguishes the two readings is either already a
parameter or is chain position:

| distinguishing feature | how it is expressed |
|---|---|
| gaps transparent vs. dark | `alpha_mode` |
| warp meaningful vs. vestigial | `warp_x`, `warp_y` |
| Scene band vs. Display band | chain position |

A `model="hologram"|"display"` enum would be redundant with the first two and
powerless over the third, so it earns nothing.

For the Display-band reading before the reorder GUI lands: hand-edit
`animator.json`. After it lands: drag it. Either way this brief does not need to
anticipate it.

Two wrappers would also have been actively bad for the GUI. This filter has one
of the most complex control panels in the module; two copies of it is a wall of
controls, and the settings editor has no add/remove/move affordance that would
make the choice legible anyway. One copy in a locked position is the right
shape for the GUI that exists.

### `name` parameter is required

Because persistence caches an accumulator (see below), `crt` is a caching
filter and needs `name: str = "crt0"` with `name=["!ignore"]` metadata,
following `zoom`, `chromatic_aberration`, `noise` and the rest. Key the
accumulator on it. This is also what makes a second hand-configured instance
work correctly the day someone wants one.

### Alpha: the scanline gaps are transparent

This is the parameter the hologram model flips relative to a naive CRT port.
Where the beam is not writing, a hologram emits *no light*, so the room shows
through. Modulate **alpha as well as luma**; this is the correct model, not a
compromise. `alpha_mode="both"` is the default; `"luma"` is kept for taste
testing against a bright backdrop, and is what you would set for a
Display-band configuration.

Alpha is modulated by the **scanline** term only, never the mask. The mask is a
chromatic structure — dimming one channel's emitters does not make that patch
of image transparent. Modulating alpha by the mask punches holes at the mask
pitch and reads as a screen door.

Interaction to watch: `translucent_display` (10.5) is already in the default
chain and also reduces alpha. Stacked at defaults the avatar may go too faint;
expect to pull one down.

**No filter in this brief sets `alpha = 1`.** An RGBA frame can always be
`over`'d onto a backdrop later; writing an opaque background is irrecoverable,
and it destroys two things at once — the client-side backdrop, and any alpha
work done by earlier filters in the chain (`translucent_display`, the `"A"`
channel modes of `noise` and `scanlines`, this filter's own `alpha_mode`).
An earlier version of this brief had a `flatten` / `tube_color` escape hatch;
it is **removed**. See §5 for where that capability correctly belongs.

### Where scanlines and mask live geometrically

- **Scanlines in warped (source) space.** The raster is laid down by the scan,
  so under any warp the lines curve with the image. Take the phase from the
  warped sample coordinate's `y`.
- **The mask in output (screen-pixel) space.** Physically the emitter structure
  is on the same surface and should warp too — but the mask pitch is comparable
  to the output pixel pitch, and warping it produces moiré that swamps the
  effect. Screen-aligned is the practical choice. This is a deliberate
  deviation from physical correctness; say so in the comment so nobody "fixes"
  it later.

### Warp is nearly vestigial here

Barrel distortion is only visible where there are straight lines to bend, and a
character silhouette on a transparent field has none. A hologram has no glass
either. Implement it — five lines, fuses into the same `grid_sample` — but
**default to 0.0** and do not tune the look around it. It earns its keep only in
a Display-band configuration with a server-side backdrop, where there is a full
raster with edges to bend.

### The Gaussian scanline is the single highest-value detail

Most amateur CRT implementations do `if y % 2 == 0: brighten else: darken`,
which gives uniform grey-and-dark stripes and reads as a cheap overlay. Lottes'
Gaussian falloff makes the bright rows look like *glowing emitters with falloff
into darkness*. If only one thing from the reference shader survives the port,
make it this.

### Free win from working in linear light

The postprocessor operates in **linear intensity space, before gamma** (module
docstring; `torch_linear_to_srgb` runs after the chain in
`raven/server/modules/avatar.py`). Lottes' shader does explicit
`ToLinear`/`ToSrgb` round-trips because it runs on an sRGB backbuffer. We get
that for free, and the Gaussian scanline and multiplicative mask are *more*
correct here than in the original. Do not add gamma handling.

### Brightness compensation

Scanlines and mask are both multiplicative and both < 1, so a naive
implementation dims the image substantially — on real hardware the beam is
driven correspondingly harder. Divide by the spatial mean of `w_scan * w_mask`,
a closed-form constant per parameter set, not a per-frame reduction: compute
once and cache alongside the mask tensor. Expose `brightness_compensation` in
[0, 1]; full compensation sometimes looks too hot going into the capture bloom.

## 4. Implementation

### Reference

Timothy Lottes' CRT shader (`CRT_Lottes.fxh`, as shipped with ReShade/vkBasalt
— already installed and configured on maia for Arcana Heart 3), ported in
spirit rather than transliterated.

### What exists

All paths relative to repo root, all in `raven/common/video/postprocessor.py`
unless noted.

- **`self._meshx`, `self._meshy`**: base coordinate grids in [−1, +1], shape
  `[h, w]`, rebuilt by `_setup_meshgrid` on size change. Input to the warp.
- **`grid_sample` pattern**: see `zoom`, `analog_rippling_hsync`,
  `analog_runaway_hsync`, `analog_vhstracking`. Use `padding_mode="zeros"` here
  so warped-away corners go to nothing rather than smearing edge pixels.
- **`scanlines`** (13.0): the existing hard-step implementation. Read it — its
  `double_size` flag generalizes to `scanline_period`, and its `dynamic` field
  alternation (VLC Phosphor-deinterlacer style) is worth carrying over. Do not
  modify or deprecate it.
- **`_blur_kernel_size(sigma)`** (~345) and the two-pass separable
  `torchvision.transforms.GaussianBlur` pattern in `bloom`: reuse if
  `glow_strength` is turned up.
- **`with_metadata(...)`**: every parameter with a default **must** have a
  metadata entry or `get_filters` raises `KeyError`
  (`ranges = {name: meth.metadata[name] for name in settings}`). `["!ignore"]`
  hides a parameter. The filter docstring is shipped to the client and rendered
  as the settings-editor tooltip (`avatar/settings_editor/app.py:800`, `:813`),
  so document every parameter there.
- **`self.frame_no`**: `CALIBRATION_FPS * seconds_since_stream_start`, i.e.
  wall time in disguise — 25 units per second regardless of real frame rate.
  Use directly for the field alternation.
- **Render loop** (`raven/server/modules/avatar.py:1630–1680`): crop → upscale
  (Anime4K) → postprocessor → gamma. The postprocessor already runs after the
  upscaler, so the raster is generated at output resolution rather than
  upscaled into existence.

### Shape

```
1. Warped sample grid from _meshx/_meshy:
       sx = mx * (1 + warp_x * my**2)      # Lottes-style separable barrel
       sy = my * (1 + warp_y * mx**2)
   (Lottes uses f2Warp = (48, 24), i.e. warp_x = 1/48, warp_y = 1/24. Use the
    direct-strength form — easier as a GUI slider than a radius.)
   Apply `overscan` as a uniform scale before sampling.
2. ONE grid_sample, padding_mode="zeros". SKIP entirely when warp_x, warp_y
   are 0 and overscan is 1.0 — that is the default path, and skipping avoids a
   pointless resample.
3. Scanline weight from the WARPED sy (source space):
       line_phase = frac(sy_pixels / scanline_period + field_offset)
       w_scan = exp(-scanline_weight * (2*line_phase - 1)**2)
4. Mask weight from OUTPUT pixel x (screen space), per mask_type.
5. w = w_scan * w_mask * corner_falloff, renormalized by brightness_compensation
   image[:3] *= w
   if alpha_mode == "both": image[3] *= w_scan   # scanline term ONLY
6. Optional beam bleed: 3-tap horizontal convolution.
7. Phosphor persistence (see below). AFTER modulation, so the trail carries the
   raster structure — the beam writes the pattern into the phosphor, and it is
   the pattern that decays.
8. Optional glow (default off): threshold + separable Gaussian + add.
9. Clamp to [0, 1].
```

### Phosphor persistence

In for v1. One extra `[c, h, w]` accumulator, keyed on `name`, in the existing
`defaultdict(lambda: None)` style:

```
acc = maximum(acc * decay, image)
image = maximum(image, acc)
```

**Max-with-decay, not additive.** Physically right for phosphor — emission
decays, it does not accumulate — and unconditionally stable, whereas an
additive feedback loop compounds any brightness error frame over frame.

Decay is per *second*, not per frame, via the FPS-correction machinery. This is
the one place in the CRT work where `self.last_frame_no` earns its keep:

```
dt = (self.frame_no - self.last_frame_no) / self.CALIBRATION_FPS   # seconds
decay = math.exp(-dt / persistence_tau)
```

Apply to **all four channels including alpha**. Under the hologram reading a
moving character should leave a trail of lingering light, and the light being
there is exactly what alpha encodes.

**Size-change hazard.** The crop can change mid-stream, and `render_into`
re-inits the meshgrid when it does. The accumulator must be dropped on any
`(h, w)` change or it is silently wrong — store the shape alongside it and
compare, do not rely on the meshgrid's own tracking.

**Tuning interaction worth knowing about.** With `dynamic_field=True`, the
previous field's lines persist into the current frame and partly fill the gaps,
reducing effective scanline contrast. This is what an interlaced CRT actually
does, so it is a feature — but it means `persistence_tau` and
`scanline_strength` are coupled, and a large tau washes the scanlines out. Tune
them together.

At 25 FPS the frame period is 40 ms, so `tau = 0.08 s` gives a per-frame decay
of `exp(-0.5) ≈ 0.61` — visible trailing without smearing. The default below sits
just under that; it is the first knob to touch.

**`persistence_tau = 0.0` means off, and must be bit-identical to the stateless
path** — not "approximately equal", identical. Short-circuit before touching the
accumulator, and assert it in the tests. This is what keeps the determinism the
rest of the test suite relies on, and it is the reason persistence is safe to
ship in an otherwise stateless filter.

### Mask types

`mask_type` ∈ `"aperture_grille"` | `"slot"` | `"shadow"` | `"none"`.

- `aperture_grille`: vertical RGB stripes, period `mask_pitch` output pixels.
  Cheapest, most legible at 1024², the Trinitron look. **Default.**
- `slot`: aperture grille with a vertical phase offset every `mask_pitch` rows —
  the consumer-TV look.
- `shadow`: triad dot mask. Most expensive, least legible at 1024². Implement
  last; shipping without it is acceptable.

`mask_pitch` matters more than it looks: at 1024² a 3-px triad is strongly
visible, at 4K it disappears. Same problem `scanlines` solved with
`double_size`; generalize to an integer pitch and let resolution decide.

### Parameters

| parameter | type | range | default | note |
|---|---|---|---|---|
| `warp_x` | float | [0.0, 0.10] | 0.0 | vestigial in hologram mode |
| `warp_y` | float | [0.0, 0.10] | 0.0 | ditto |
| `overscan` | float | [1.0, 1.15] | 1.0 | uniform scale |
| `scanline_period` | int | [1, 6] | 2 | output lines per raster line |
| `scanline_weight` | float | [0.0, 8.0] | 2.0 | Gaussian falloff exponent |
| `scanline_strength` | float | [0.0, 1.0] | 0.6 | overall depth |
| `dynamic_field` | bool | [False, True] | True | alternate field per frame |
| `field` | int | [0, 1] | 0 | which field starts dimmed |
| `mask_type` | str | see above | `"aperture_grille"` | |
| `mask_pitch` | int | [2, 12] | 3 | output px |
| `mask_strength` | float | [0.0, 1.0] | 0.35 | |
| `brightness_compensation` | float | [0.0, 1.0] | 0.85 | |
| `beam_bleed` | float | [0.0, 1.0] | 0.25 | horizontal 3-tap |
| `glow_sigma` | float | [0.3, 5.0] | 1.2 | only if glow_strength > 0 |
| `glow_strength` | float | [0.0, 1.0] | 0.0 | off: capture bloom does this |
| `corner_falloff` | float | [0.0, 1.0] | 0.10 | emitter-side, not lens |
| `alpha_mode` | str | `["both", "luma"]` | `"both"` | gaps transparent |
| `persistence_tau` | float | [0.0, 0.5] | 0.06 | seconds; 0.0 = off, stateless |
| `name` | str | `["!ignore"]` | `"crt0"` | cache key for the accumulator |

### Testing

`raven/common/video/tests/`, alongside the existing postprocessor tests. As
elsewhere in this module these are contract tests, not aesthetic ones.

1. **Contract**: shape/dtype/device preserved; in-place mutation; output in
   [0, 1].
2. **Identity**: strengths at 0, warps at 0, `overscan` 1.0, `mask_type="none"`
   → output *bitwise* equal to input. This exercises the step-2 resample skip,
   so a tolerance-based comparison would hide a regression there.
3. **Mean level**: with `brightness_compensation=1.0`, mean luma preserved to
   within a few percent across a parameter sweep. Catches a botched
   compensation constant.
4. **Mask periodicity**: for `aperture_grille`, per-column channel weights are
   periodic with `mask_pitch`.
5. **Field alternation**: `frame_no` N and N+1 give complementary phases; and
   output depends only on the *value* of `frame_no`, not on how many times the
   filter has been called.
6. **Alpha**: `"luma"` leaves alpha untouched; `"both"` modulates alpha at the
   scanline period but shows **no** mask-pitch periodicity in the alpha channel
   (the screen-door regression); no configuration produces `alpha == 1` where
   the input had `alpha == 0`.
7. **Resolution dependence of the *parameters***: mask pitch and scanline
   period are in output pixels by definition, so 512² and 1024² renders of the
   same content differ in apparent pitch. Assert it, so nobody later "fixes" it
   into normalized coordinates.
8. **Persistence off is stateless**: with `persistence_tau=0.0`, output is
   *bitwise* equal to a run with the accumulator code path removed, and
   repeated calls at a fixed `frame_no` are bitwise identical to each other.
   This is the test that protects every determinism assumption above.
9. **Persistence decay is per-second**: advancing `frame_no` by 1.0 twice must
   give the same accumulator state as advancing by 2.0 once. Catches a
   per-frame decay constant masquerading as a per-second one.
10. **Persistence drops on resize**: feed a frame, then a differently-sized
    frame; the second must not be contaminated by the first. Assert on the
    shape guard, not just on pixel values.

### Performance budget

Postproc currently runs ~11 ms of the frame budget
(`briefs/avatar-render-pipeline.md`). Target **≤ 2 ms at 1024²**: with warp at
its default there is no `grid_sample` at all, leaving a couple of elementwise
multiplies against cached weight tensors plus an optional 3-tap convolution.
Cache the mask weight tensor and the compensation constant, keyed on
`(h, w, mask_type, mask_pitch, mask_strength)` — they change only on chain swap
or resize.

Persistence adds one `[4, h, w]` resident tensor (~8 MB at fp32 / 4 MB at fp16
for 1024²) and two elementwise `maximum` ops per frame. Negligible in time;
the memory is the only thing worth noting, and it is freed when
`persistence_tau` is 0.0 because that path never allocates.

## 5. Future work (design recorded, not scheduled)

### Multi-instance filters with configurable priorities

The backend already supports this (§0): the chain is a positional list and every
caching filter carries a `name` to key its caches apart. The missing piece is
entirely GUI — add / remove / reorder, and per-instance priority — and it is
nontrivial UX work, which is why it was skipped when the settings editor was
built.

It is the thing that unlocks several items here at once: the Display-band
reading of this filter, a second dust layer with different tuning, and
`backdrop` coexisting cleanly with the rest. Until it exists, filters ship one
copy in a locked position and anything else is a hand-edit of `animator.json`.

### `backdrop` — server-side background

A Scene-band filter at **−10.0**, which composites a backdrop image under the
frame and is the **only** filter licensed to produce `alpha = 1`. That is the
right home for the capability removed from this brief: a filter whose entire job
is to be the background has business making the frame opaque; a display
simulation does not.

Placement: it must apply after upscaling (so the backdrop renders at full output
resolution rather than being upscaled) but before the postprocess filters proper.
Priority −10.0 satisfies this given that the whole chain already runs after the
upscaler, so it can be an ordinary filter and get settings-editor integration for
free, rather than a separate pipeline stage in `avatar.py`.

**Arbitration between client-side and server-side backdrops**: the client owns
the config, so the client arbitrates — if the client has a backdrop configured,
it does not enable the server-side one. This keeps the decision where the
knowledge is, and avoids the server having to guess what the client will do with
the alpha it is handed.

With `backdrop` present, the Display-band reading of `crt` needs no special
casing: the frame reaching it is already opaque. Without it, `crt` on a
transparent frame degrades gracefully to the hologram behaviour. Nothing needs
to force alpha anywhere — which is the check that the no-`alpha = 1` rule
composes.

## 6. Out of scope

- Bezel, glass reflections, curvature outside the image area.
- Real 50i/60i interlace — needs a two-field history and interacts badly with
  the 25 FPS target. The alternating-field trick is the whole interlace story
  here.
- Removing, deprecating or reworking `scanlines`.
- Renumbering any existing filter's `_priority`. §0 is descriptive of what is
  already there; if it turns out not to be, that is a finding to report, not a
  licence to renumber.
- Any GUI work. This brief adds a filter that the existing autodiscovery picks
  up; it does not touch the settings editor.
