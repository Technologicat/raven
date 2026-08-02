# Brief: `atmospheric_dust` — drifting in-air particles

Target file: `briefs/summer_2026_librarian_extension/atmospheric-dust.md`

## Goal

Add faint light-catching motes drifting in the air of the avatar's environment:
dust, pollen, or (with different tuning) something closer to snow or petals.
The effect is **diegetic** — it lives in the character's world, not on the
viewer's display, and not as a HUD overlay.

Purpose: a live avatar is real-time and interactive, and drifting particles are
the standard real-time idiom for signalling environmental dynamism. Anyone
reading the pop-culture visual language picks it up without noticing they did.
It is also cheap: we have a few GPU cycles spare.

Register to aim for: **anime-atmospheric** (light motes in a god-ray, dust in a
sunbeam), *not* game-HUD sparkle. The particles are a property of the air, not
a UI flourish.

## Design rationale

### Routing: in the world, or on the display?

That question sorts every stage of the postprocessor chain. Dust is **in the
world**, so it composites *early* — it must ride through the capture-stage
optics (`bloom`, `chromatic_aberration`, `vignetting`) exactly like the
character does. A late overlay would be the HUD case and would read wrong.

Concretely: priority **−2.0**, in the **Scene** band. See §0 of
`crt-display.md` (same folder) for the band scheme this assumes — briefly, `0.0` is
the moment of capture, negative priorities are things that exist in front of
the lens, positive ones are signal. Dust is air in the room, so it is Scene by
definition. That brief's §0 is a prerequisite for this one; if the two are
implemented separately, land the priority-semantics comment block first.

Being ahead of `zoom` (−1.0) also means camera framing moves the dust field
along with the character, which is what an in-world element should do.

### Ordering against `crt`

Under the shipping diegetic model the avatar is a **hologram projected into a
physical scene**. That makes the raster structure a property of the *projected
character*, not of the room — so dust drifting in the room must **not** carry
scanlines. `crt` is placed at −3.0, ahead of this filter, and the
ordering resolves itself:

```
character → crt (raster) → atmospheric_dust (room air)
          → zoom (framing) → bloom / chromatic_aberration / vignetting (capture)
```

Do not move this filter ahead of `crt`; scanlined dust is the failure
mode that ordering exists to prevent.

One caveat to test: `zoom` with `quality="high"`/`"ultra"` crops and re-runs
Anime4K on the region. Anime4K on 1–2 px bright dots may ring. Test with
`quality="low"` first; if ringing shows, the fallback is priority −0.5 (after
zoom, before bloom), which loses the parallax-under-zoom but keeps the bloom
coupling — which is the more important of the two.

### Why bloom coupling is the whole trick

`bloom` sits at priority 0.0 and does `image.add_(brights)` followed by the
tonemap `1 − exp(−x·exposure)`. Glints that clear `bloom.threshold` (default
0.560) therefore bleed a soft halo before being tonemapped. That halo is what
turns a white dot into a *mote catching the light*. Without it the effect reads
as dead pixels.

So: `atmospheric_dust` must **not** hard-clamp to 1.0 by default if bloom is in
the chain — see `max_intensity` below.

### Physical honesty

We are faking the *transport* (no airflow solver — explicitly not
`extrafeathers` Navier–Stokes), but not inventing the *phenomenon*: real air
carries motes, and the target look is strongly-lit, which is exactly the regime
where they become visible. So this sits closer to the optics-realism end than to
pure rule-of-cool, with a residual artistic-liberty margin whose acceptable size
is viewer-dependent. Tune by testing; "more subtle" is not automatically better.

### Depth of field lives inside this filter

The compositing pipeline is 2D-only: the character is a flat billboard and the
backdrop is a flat plane. Dust is the **only** element in the whole pipeline
with a populated depth range, so focal-length and aperture controls belong
inside this filter and nowhere else. This avoids making the composite
depth-aware, and it is barely a cheat: DoF is only meaningful where depth is
populated.

Corollary discovered while reading the code: the "glittery bits vs. bokeh blobs"
question is not an either/or. They are the same particle population at different
distances from the focal plane. In focus → a sharp glint. Out of focus → a soft
disc whose shape is the aperture's shape. One population, one focus parameter.

### One filter, not two chain entries

The natural design would be two chain entries (behind-character, in-front) the
way `chromatic_aberration` supports multiple instances via `name`. **This does
not work with the settings editor**: `strip_postprocessor_chain_for_gui` and
`generate_postprocessor_chain_from_gui` in
`raven/avatar/settings_editor/app.py` both do `dict(postprocessor_chain)`,
keeping only the last entry per filter name. The GUI is strictly
one-entry-per-filter.

So: **one filter, both layers internal.** A `character_depth` parameter splits
the particle set into behind and in-front, and the two sub-layers are composited
around the incoming image. This is simpler anyway and gives occlusion for free
(see below).

## What exists

All paths relative to repo root.

### `raven/common/video/postprocessor.py`

- **Image format**: `[c, h, w]`, `c = 4` (RGBA), **linear RGB**, straight
  (non-premultiplied) alpha, range nominally [0, 1]. Gamma correction happens
  *after* the whole postprocessor chain, in
  `raven/server/modules/avatar.py` (`torch_linear_to_srgb`).

- **Transparent background**: the avatar frame's background is transparent. The
  backdrop image is composited **client-side** (`raven/client/avatar_renderer.py`;
  see `avatar.py:925`, "The backdrop image is applied at the client end").
  This is a *feature* for this filter — see "Compositing" below.

- **`with_metadata(...)` decorator** (line ~358): stashes GUI hints on the
  function object. `get_filters` (line ~476) reads it to auto-populate the
  settings editor. **Every parameter with a default must have a metadata
  entry**, or `get_filters` raises `KeyError` — `ranges = {name: meth.metadata[name] for name in settings}`.
  Range conventions: `[min, max]` for numeric, explicit list for enums/bools,
  `["!ignore"]` to hide from the GUI. `_priority` sets chain order.

- **`self.frame_no`**: float, normalized frame number at
  `self.CALIBRATION_FPS = 25`, recomputed from wall-clock each frame (never
  accumulated — see the long comment in `render_into`). **Use this for time**,
  not a frame counter: `t = self.frame_no / self.CALIBRATION_FPS` gives seconds
  and is automatically FPS-corrected.

- **Per-filter caches**: `defaultdict(lambda: None)` keyed by the filter's
  `name` parameter (`zoom_data`, `ca_grid_cache`, …). We need only a small
  variant of this — see "State".

- **Statelessness intent**: "We intentionally keep very little state in this
  class, for a more FP/REST approach with less bugs." Honour it.

### `raven/common/video/compositor.py`

`render_celstack` contains a local closure `over(a, b)` implementing straight-alpha
"a over b". **Factor this out to a module-level function** and add it to
`__all__` — this filter needs it, and duplicating alpha compositing is exactly
the kind of thing that drifts out of sync. Keep `render_celstack` using the
extracted version.

## What to add

### New filter: `atmospheric_dust`, `_priority = -2.0`

Docstring tag: `[dynamic]`.

### State: none, beyond a memoized constants table

The particle trajectories are **closed-form functions of time** — no integrator,
no particle buffer, no resize invalidation, no warmup. This is worth insisting
on:

- exact FPS independence falls out for free (position is a function of `t`,
  not of how many frames elapsed);
- resolution independence too (work in normalized coordinates, scale at splat
  time);
- deterministic and reproducible from `seed`, which makes it testable;
- nothing to tear down, nothing to invalidate when the crop or upscale factor
  changes mid-stream.

The only state is a **memoized per-particle constants tensor**, keyed by
`(seed, count, device, dtype)` and regenerated only when one of those changes.
It holds, per particle, drawn once from a `torch.Generator` seeded with `seed`:

| symbol | meaning | distribution |
|---|---|---|
| `x0, y0` | initial normalized position | U[0, 1) |
| `z` | depth | U[`depth_near`, `depth_far`] |
| `vx, vy` | drift direction jitter | N(0, 1), scaled by params |
| `A, f, φ_sway` | sway amplitude / frequency / phase | U, scaled by params |
| `ω, φ_tumble` | tumble rate / phase | U around `tumble_rate` |
| `φ_light` | per-particle glint alignment offset | U[0, 2π) |
| `s` | size jitter | lognormal-ish, or U[0.6, 1.6] |

### Kinematics

With `t = self.frame_no / self.CALIBRATION_FPS` (seconds):

```
# Parallax: nearer particles sweep faster and look bigger. z in (0, 1],
# smaller = nearer. Screen velocity ∝ 1/z is the pinhole projection of a
# constant world-space velocity.
u = frac_wrap(x0 + (drift_x + vx*drift_jitter) * t / z
                 + A*sway_amplitude * sin(2π*(f*sway_frequency)*t + φ_sway))
v = frac_wrap(y0 + (drift_y + vy*drift_jitter) * t / z)
```

`frac_wrap` wraps into `[-margin, 1+margin]` rather than `[0, 1]`, where
`margin` covers the largest splat radius in normalized units. Wrapping at the
exact frame edge would pop a large bokeh disc into existence; wrapping outside
the visible area does not. This is the one place where being sloppy shows.

Screen position: `px = u * w`, `py = v * h`.

### Optics: one radius, two contributions

```
r_geo = size * s / z                              # projected particle size
r_coc = aperture * |z - focal_plane| / z          # circle of confusion, thin-lens
r     = sqrt(r_geo**2 + r_coc**2)                 # quadrature: 2nd moment of the
                                                  # convolution of the two discs
```

**Energy conservation matters here.** A defocused particle spreads the same
flux over a larger area, so peak intensity must go as `1/r²`. This is precisely
what makes an out-of-focus mote read as a soft bokeh disc rather than a big
bright blob. Normalize each splat kernel to unit total energy and scale by
intensity, don't normalize to unit peak.

### Brightness: thin disc, tumbling

Juha's instinct model, made concrete. A thin disc has a tumble phase
`φ(t) = φ_tumble + ω*tumble_rate*t`. Two terms:

```
projected_area = |cos φ|                                   # goes to 0 edge-on
glint          = |cos(φ - φ_light)| ** glint_exponent      # narrow specular lobe
I = brightness * (projected_area + glint_gain * glint) / r**2
```

The twinkle is *free* — a narrow lobe (`glint_exponent` ~ 40) means each disc
flashes only briefly as it rotates through alignment, which is exactly why real
tumbling flakes twinkle. No brightness animation needed; do not add one.

`glint_exponent` is the main expressive knob: low → soft continuous shimmer,
high → sharp sparse flashes. This is the parameter that decides
anime-atmospheric vs. game-sparkle. Expect to tune it at the console.

### Splatting

For `N` particles with per-particle radius `r_i`, let `K = 2*ceil(r_max) + 1`.

1. Build a `[N, K, K]` kernel batch by evaluating a soft-edged disc analytically
   on a local grid offset by each particle's **sub-pixel** fractional position:
   `k = smoothstep(r_i + softness, r_i - softness, dist)`, then normalize each
   `k[i]` to sum 1 and multiply by `I_i`.
   Evaluating at the sub-pixel offset (rather than snapping to integer pixels)
   is what stops the drift from looking like it's on a conveyor belt.
2. Scatter-add into a `[h, w]` accumulation buffer per layer with
   `index_put_(..., accumulate=True)` on flattened indices, masking
   out-of-bounds.

Cost sanity check: `N = 250`, `K = 63` → 250 × 3969 ≈ 1e6 float adds per layer.
Negligible. The `[N, K, K]` batch is ~4 MB at fp32, ~2 MB at fp16 — fine.

Guard: `K` is driven by `r_max`, so a large `aperture` with a near/far particle
can blow it up quadratically. Clamp `K` to a hard ceiling (say 129) and clamp
`r` accordingly; log a warning once if clamping engages. If perf ever bites,
the fallback is to bin particles into ~4 radius buckets and convolve each
bucket's point buffer with a shared disc kernel — but do not do this
pre-emptively, the direct path is exact and fast enough.

### Compositing

Split by `z` against `character_depth`:

- **Behind** (`z > character_depth`): build RGBA with
  `rgb = tint * I`, `alpha = clamp(I / alpha_reference, 0, 1)`, then
  `image ← over(image, dust_behind)`. Occlusion by the character is **free**:
  where the character's alpha is 1, the dust underneath is fully hidden.
  Where the frame is transparent, the dust raises alpha and the client's
  backdrop shows through around it. This is exactly right.
- **In front** (`z <= character_depth`): `image ← over(dust_front, image)`.

Note the transparent background works *for* us here: behind-dust in the empty
region becomes visible motes floating over the client's backdrop image, which is
the intended look, with no depth buffer and no server-side backdrop.

**This filter never writes `alpha = 1` into a previously transparent region.**
It raises alpha where a mote is, and nowhere else. General rule for the
postprocessor: an RGBA frame can always be `over`'d onto a backdrop later, but
writing an opaque background is irrecoverable, and destroys both the
client-side backdrop and any alpha work done by earlier filters in the chain. Only a future `backdrop` filter has business making the frame
opaque. Assert this in the tests.

Default `character_depth = focal_plane` puts the character in focus with dust
racking out of focus in both directions. Decoupling them lets you rack focus
onto the dust for a shot — keep both parameters.

### Parameters

All of these need `with_metadata` entries.

| parameter | type | range | default | note |
|---|---|---|---|---|
| `count` | int | [0, 2000] | 250 | 0 disables cheaply |
| `seed` | int | [0, 2**31−1] | 42 | |
| `size` | float | [0.2, 8.0] | 1.5 | px radius at `z = 1` |
| `depth_near` | float | [0.05, 1.0] | 0.25 | |
| `depth_far` | float | [0.05, 4.0] | 1.75 | |
| `focal_plane` | float | [0.05, 4.0] | 1.0 | |
| `character_depth` | float | [0.05, 4.0] | 1.0 | occlusion split |
| `aperture` | float | [0.0, 30.0] | 6.0 | px CoC per unit defocus |
| `drift_x` | float | [−0.2, 0.2] | 0.012 | normalized units/s |
| `drift_y` | float | [−0.2, 0.2] | 0.020 | +y = downward |
| `drift_jitter` | float | [0.0, 1.0] | 0.4 | |
| `sway_amplitude` | float | [0.0, 0.1] | 0.012 | |
| `sway_frequency` | float | [0.0, 1.0] | 0.15 | Hz |
| `tumble_rate` | float | [0.0, 8.0] | 1.2 | rad/s |
| `glint_exponent` | float | [1.0, 200.0] | 40.0 | the expressive knob |
| `glint_gain` | float | [0.0, 20.0] | 6.0 | |
| `brightness` | float | [0.0, 4.0] | 0.6 | |
| `tint` | list[float] | `["!ignore"]` | `[1.0, 0.97, 0.90]` | linear RGB, warm white |
| `alpha_reference` | float | [0.05, 2.0] | 0.35 | I at which a mote goes opaque |
| `max_intensity` | float | [1.0, 8.0] | 3.0 | HDR headroom, see below |
| `softness` | float | [0.3, 3.0] | 1.0 | splat edge, px |

**`max_intensity` is a real hazard, not a nicety.** The chain ends with
`255.0 * x` then `.byte()`; values above 1.0 that survive to that point will
wrap, not saturate. `bloom` clamps to [0, 1] at its end, so as long as `bloom`
is downstream, headroom above 1.0 is safe and is what makes glints bloom hard.
**If `bloom` is disabled, `max_intensity` must be 1.0.**

Put this warning in the **filter docstring**, not in a code comment.
`get_filters` ships `inspect.getdoc(func)` to the client, and the settings
editor renders it as the tooltip (`app.py:800`, `app.py:813`) — so the
docstring is the discoverability surface, and a warning that lives anywhere
else will not be seen by the person who needs it. Document each parameter
there too, in the style of the existing `scanlines` and `bloom` docstrings.

## Testing

Add to `raven/common/video/tests/` alongside the existing postprocessor tests.

1. **Contract**: output shape/dtype/device unchanged; input mutated in place;
   `count=0` is a no-op.
2. **Determinism**: same `seed`, same `frame_no` → bitwise-identical output.
3. **Path independence** (the one that actually matters). Note that this is
   *not* "render at two frame rates and compare" — `frame_no` is
   `CALIBRATION_FPS * seconds_since_stream_start`, i.e. it is wall-clock time
   in disguise, always 25 units per second no matter what the real frame rate
   is. Same `t` gives the same `frame_no` by construction, so that test would
   be vacuous.

   The property actually worth pinning down is that output depends on the
   *value* of `frame_no` and not on the sequence of calls that got there.
   Sample particle positions along a uniform sweep
   (`frame_no = 0, 1, 2, …, 10`) and along an irregular one
   (`frame_no = 0, 0.3, 2.7, 5.1, 10`); values at the shared points must match
   bitwise. Also: calling the filter ten times at a fixed `frame_no` must give
   the same result each time.

   This is what fails the moment someone replaces the closed form with an
   integrator, or reaches for `self.last_frame_no`. Which is the other half of
   the point: the FPS-correction machinery (`last_frame_no`) exists for
   rate-based effects that *must* accumulate. This filter needs none of it.
4. **Energy conservation**: total added flux (sum over the accumulation buffer)
   is invariant to `aperture` for fixed `count`, `brightness`, and tumble
   phases, to within splat-clipping at the borders.
5. **Wrapping**: no discontinuity in total flux as particles wrap — sample a
   sweep of `t` and check for spikes.
6. **Bounds**: with `max_intensity=1.0`, output stays in [0, 1].

## Performance budget

Current postproc stage is ~11 ms of the frame budget (see
`briefs/avatar-render-pipeline.md`). Target **≤ 1.5 ms at 1024²** for this
filter with default `count`. If it exceeds that, the first lever is `count`,
the second is the radius-binning fallback. Measure with the existing
`maybe_sync_cuda()` timing in the render loop.

## Out of scope

- Any airflow simulation. No advection, no curl noise field, no solver. The
  sinusoidal sway is the entire "physics".
- A real depth buffer or z-ordering in the compositor.
- Particle shape variants (petals, snowflakes, dandelion seeds). The thin-disc
  model is the deliverable; a `shape` parameter selecting a sprite is the
  obvious v2 and should be designed then, not anticipated now.
- Interaction with the character's motion (no wake, no displacement).
- Lighting from the scene. `φ_light` is a free parameter, not derived from
  anything.
