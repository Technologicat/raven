# Client-side Crop Support — Implementation Plan

**Status: implemented (2026-04-20).** All four phases landed end-to-end:
- Phase 0 `d2fcf5b` — reactive texture sizing from decoded frame dims.
- Phase 1 `9b6011d` — client honours the crop bbox; `X-Crop` + `X-Full-Size` per-frame headers; new `crop` dict format.
- Phase 2 `7d5bfed` — Crop GUI panel in the settings editor (sliders with gap clamping, debounced push via `bgtask.ManagedTask`, live overlay preview via viewport drawlist with panel clipping, FPS counter moved to a second viewport drawlist so it stays readable on top of the overlay, `X-Server-Stats` for per-phase server timings shown in the FPS counter).
- Phase 3 `81804a6` — server-side `crop → upscale → postprocess` reorder (was `upscale → crop → postprocess`). `Upscaler.reconfigure_output_size` is a cheap attribute update (no NN reinit needed — Anime4K handles variable input via `AutoDownscalePre`). `Animator.render_pipeline_lock` serializes the render thread against the settings handler.

The rest of this document is the plan that drove the implementation, kept for historical reference.

References:
- `raven/client/avatar_renderer.py:18` (standing TODO: "support non-square avatar video stream")
- `raven/server/modules/avatar.py:1415`, `1511–1521` (server-side crop)

## Background

The server's avatar pipeline supports cropping the output image via the `crop_left/right/top/bottom` settings in `animator.json` (server: `raven/server/modules/avatar.py`). Applied, this buys up to ~50% more throughput on the avatar rendering pipeline — proportional to how much is cropped away.

The client side currently ignores this: `raven/client/avatar_renderer.py:18` has a standing TODO confirming the omission is known. Today the client assumes the incoming frame fills the full unit bbox, so if the server crops, the client displays a squashed / mis-positioned image.

Fix: the client needs to receive the animator's crop settings (already sent by the client itself when configuring the server) and use them to align the cropped frame within the unit bbox that would have been occupied by the full non-cropped image. The math is straightforward — the crop is already in `[0, 1]` normalized coordinates; the client just places the received image at the appropriate offset and scale within its render target.

Typical target framing: THA3 natively renders the character at roughly a **cowboy shot** (head to mid-thigh / upper hips) in a square frame, with substantial empty space at the left and right of the standing figure. The crop feature trims those side margins, giving a taller-than-wide output with the character centred horizontally. Vertical crop is usually minimal — THA3's framing already fills the square top-to-bottom.

**Lower bound on horizontal crop**: the character animates into the side margins via head/body morphs (`neck_z`, `body_z`, `head_x`) — the sway animation drives these same morphs with controlled randomness on top of the base pose, so the silhouette drifts sideways continuously. Cropping too tight will clip the character mid-animation. Detecting the clip risk automatically is non-trivial (THA3 doesn't expose an extent box, and the excursion depends on the base pose as well as the sway range), so for now leave it to the user's judgment — the overlay + live preview in Phase 2 gives enough feedback to calibrate by eye.

## Overview

Goal: client honours the crop settings by blitting the received (smaller) image into the sub-region of the unit bbox where the full image *would have been*. This alone unlocks the latent server-side speedup (the postprocessor already benefits from the smaller region, since crop happens before postprocess in the existing pipeline — but nobody uses crop today because the client doesn't honour it). Then add a GUI for live adjustment. Finally, reorder crop-before-upscale on the server: this is what makes Anime4K tractable on lower-end hardware when a narrow crop is in effect.

## Semantics — new format

Break the existing `crop_left/right/top/bottom` format (nothing currently consumes it on the client, and Raven is the only avatar client). Replace with a single object carrying an `enabled` master switch plus the bbox edges in [0, 1]² unit coordinates:

```json
"crop": {"enabled": false, "left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0}   // disabled
"crop": {"enabled": true,  "left": 0.2, "top": 0.0, "right": 0.8, "bottom": 1.0}   // enabled, 20% trimmed each side
```

A dict rather than an array, to sidestep positional-ordering conventions entirely — no choosing between `[left, top, right, bottom]` (PIL/DOM/DPG reading-order) and `[left, right, top, bottom]` (matplotlib axis-range). Explicit names at every serialization boundary; typos and positional swaps are impossible. Extensible if ever needed.

The `enabled` field lets the user's edge-slider calibration persist across toggles: disable crop in the GUI, the four edges retain their values; re-enable later, the same framing comes back. Without `enabled`, the only way to represent "off" would be to zero the edges, losing the calibration.

Same representation server-side, client-side, in `animator.json`, and on the wire (see per-part header below). Server converts to pixel indices at render time; client uses directly for widget positioning. No conversions anywhere except at the bbox → pixel boundary on the server.

**Plain dicts throughout** — no wrapper class. The shape `{"left", "top", "right", "bottom"}` is the single shared contract, used identically in `animator.json`, in `self._settings` on the server, in the `X-Crop` header, and in all Python code on both sides. JSON-native, zero conversion at boundaries, consistent with the rest of `self.animator_settings` (which is dict-shaped throughout).

A module-level constant for the default, somewhere in the client (e.g. top of `avatar_renderer.py` — the primary consumer):
```python
from unpythonic import frozendict
NO_CROP = frozendict({"enabled": False, "left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0})
```
`frozendict` enforces the "don't mutate this" intent — a bare dict would rely on convention. `unpythonic` is already a Raven dep, so no new weight. Field access is `bbox["left"]` etc. Document the five keys in the docstring of any function that takes a bbox argument.

**Semantics of `enabled`**: the master switch. When `false`, server skips the crop tensor-slice (outputs the full frame) and client treats the effective bbox as full for widget positioning, regardless of the edge values. When `true`, both sides use the four edges. The header (see below) echoes the full five-field dict verbatim — one shape everywhere.

**Typo safety**: the server's `load_animator_settings` already has a `drop_unrecognized` pass (`raven/server/modules/avatar.py:763–767`) that warns on unknown top-level keys. Extend it with one extra check specifically for the nested `crop` dict: warn if its keys aren't exactly `{"enabled", "left", "top", "right", "bottom"}`. Mirrors the existing pattern; a typo anywhere in the crop dict surfaces as a server-log warning on load, same as any other settings typo.

Update the existing three settings files (`animator.json`, `glitchyholo.json`, `redsepia.json`) — all currently have `crop_* = 0`, so migration is mechanical: replace the four keys with the single `"crop"` dict.

## Phase 0 — Drive texture size from incoming frames

(Prerequisite for Phase 1, but valuable on its own — fixes the existing in-flight resolution-change issue.)

### Current problem

When the client changes upscale settings, it pre-emptively calls `configure_live_texture(new_size)` to resize the texture. The server is still producing frames at the old size for several frames afterwards. Those frames hit the size-mismatch branch in `update_live_texture` (`raven/client/avatar_renderer.py:560` and `:574`) and get software-rescaled via `PIL.Image.resize` — slow, and logged as a warning every time.

### Fix

Treat the incoming frame as the source of truth for texture size:

- In `update_live_texture`, after decoding, compute `(h_decoded, w_decoded)` from `image_rgba.shape`.
- If `(w_decoded, h_decoded)` differs from the current texture's size, call `configure_live_texture(w_decoded, h_decoded, bbox=<from header>)` to recreate the texture at the incoming size (seeded from `self.last_image_rgba` rescaled — the existing flicker-mitigation path still works).
- Delete the two `PIL.Image.resize` bridge paths inside `update_live_texture` (QOI branch at `avatar_renderer.py:561–568`, PIL branch at `:572–577`). They become dead code.
- **Leave the `PIL.Image.resize` in `configure_live_texture` (~line 301) alone** — it serves a completely different purpose (seeding the new texture from the last received frame to reduce flicker on size change). Not a bridge path; stays as-is.

This also means `configure_live_texture` no longer needs to be called pre-emptively from app code on upscale changes; it becomes purely reactive. External callers that currently call it (e.g. the settings editor on upscale slider change) can simply stop — the texture will catch up on its own once frames arrive. Leave the method callable (some apps may still want to pre-seed a size during initial setup) but document it as "optional; the renderer will adapt to incoming frames automatically."

### Signature change

`configure_live_texture(new_image_size: int)` → `configure_live_texture(new_w: int, new_h: int, crop_bbox: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0))`. The crop_bbox parameter is new in Phase 1; Phase 0 can land with just the separated w/h and default bbox.

### Interaction with Phase 1

With Phase 0 in place, Phase 1's crop handling is nearly free: the per-part `X-Crop` header tells the renderer where to position the image widget; the incoming frame size is already handled by Phase 0's reactive sizing. No separate "crop changed" machinery needed.

## Phase 1 — Client respects crop from `animator.json` (no GUI yet)

### Wire format: bbox in per-part header

When crop settings change, the client will receive some queued frames at the *old* crop before the new settings take effect. To let the client position each frame correctly regardless of in-flight settings, the server attaches the bbox as a per-part header on each frame — every frame self-identifies. Each multipart boundary already carries its own `Content-Type` / `Content-Length` (see `raven/server/modules/avatar.py:549–553` and the parser in `raven/common/netutil.py:71–79`); we add one more header.

Header format — JSON-encoded object, same shape as `animator.json`:
```
X-Crop: {"enabled":true,"left":0.2,"top":0.0,"right":0.8,"bottom":1.0}
```
Parse with `json.loads()` → plain dict. Single serialization format across every boundary (JSON in `animator.json`, JSON in the header, dict in Python).

Server side (`result_feed.generate`): stamp from a per-frame snapshot of the crop settings taken at render time (the values actually used for *this* frame), not from live `self._settings` — otherwise a settings update between render and send would misreport.

Client side:
- `raven.common.netutil.multipart_x_mixed_replace_payload_extractor`: generalise to yield `(mimetype, extra_headers, payload)` where `extra_headers` is a lowercase-keyed dict of every non-Content-* header parsed from the part. The existing `Content-Type` / `Content-Length` handling stays; everything else goes into `extra_headers` verbatim. Keeps the parser forward-compatible without special-casing crop.
- `raven.client.api.avatar_result_feed`: pass through the new tuple.
- `ResultFeedReader.get_frame` in `avatar_renderer.py`: parse the `x-crop` header via `json.loads()` and yield the dict alongside the payload. The header is mandatory — if it's missing, let it raise (it means a server/client version mismatch, which shouldn't happen in a shipped Raven release).

### JavaScript note

Per-part headers in `multipart/x-mixed-replace` are readable from JS, but **not** via the `<img src="…result_feed">` shortcut — that path delegates to the browser, which swallows the headers. A JS avatar client will have to `fetch()` the stream and parse the multipart body manually (a JS port of `multipart_x_mixed_replace_payload_extractor`, ~50 lines). This is already the right trade-off: the headers carry information the `<img>` path can't express anyway (per-frame crop, and later per-frame anything else we want to send), and `fetch() + ReadableStream` is the standard modern approach. Worth flagging in the JS-client design doc when the time comes.

### `DPGAvatarRenderer` changes

State added:
- `self.crop_bbox: Mapping[str, Any] = NO_CROP` — current bbox, updated per-frame from the header. Annotated as `Mapping` rather than `dict` because `NO_CROP` is a `frozendict`; per-frame values from the header are plain `dict` (freshly decoded from JSON, not aliased across frames, so mutation risk is nil). Value type is `Any` because the dict mixes `bool` (`enabled`) and `float` (edges).

`configure_live_texture(new_w, new_h, crop_bbox: Mapping[str, Any])` (signature already widened in Phase 0; `Any` because the dict mixes `bool` and `float` values — consistent with how other settings dicts are typed in Raven):
- Texture is created at `(new_w, new_h)` — the cropped size, matching what the server is actually sending.
- Effective bbox: if `crop_bbox["enabled"]` is `False`, positioning uses full bbox regardless of edge values; otherwise it uses the four edges.
- Widget is positioned so the effective bbox maps onto the region the full square would have occupied:
  ```
  left, top, right, bottom = (effective_bbox[k] for k in ("left", "top", "right", "bottom"))
  full_size = new_w / (right - left)   # = new_h / (bottom - top), sanity-check
  widget_x_left = avatar_x_center - full_size/2 + left * full_size
  widget_y_top  = avatar_y_bottom - full_size   + top  * full_size
  widget_w      = new_w
  widget_h      = new_h
  ```
  The previous "center the square within the parent" math is the `NO_CROP` special case of this.

Receiver loop (`update_live_texture`):
- Unpack the bbox from the per-frame header.
- If `(w_decoded, h_decoded)` differs from the current texture size OR `bbox` differs from `self.crop_bbox` → call `configure_live_texture(w_decoded, h_decoded, bbox)`. Both triggers go through the same reconfiguration path (already established in Phase 0).

### Disable crop when no explicit settings

When `animator.json` doesn't specify `crop`, server and client both default to `(0, 0, 1, 1)`. No special handling needed — the full-bbox case is just a value in the same codepath.

### Testing Phase 1

Manually edit `animator.json`, set `"crop": {"enabled": true, "left": 0.2, "top": 0.0, "right": 0.8, "bottom": 1.0}` (20% trimmed from each side, full vertical — cowboy-shot-ish framing). Start the avatar. Verify the character is correctly centered horizontally, not squashed, not offset. Toggle crop off by flipping `"enabled": false` (edges retained) and reloading — verify the image returns to full bbox while the edge values are preserved in the file.

## Phase 2 — GUI in the avatar settings editor

### Implementation note: no auto-discovery for Crop

The postprocessor panels (Zoom, Bloom, Noise, Vignetting, …) are not hand-coded section-by-section. They are generated by iterating over filter metadata retrieved via `api.avatar_get_available_filters()` — see `raven/avatar/settings_editor/app.py:690` and the GUI-building machinery that reads from `self.all_postprocessor_filters`.

Crop is **not** a postprocessor filter (it's implemented separately on the server, ordered differently in the render pipeline), so we can't just register it in the filter-discovery system. The Crop section must be hand-built using the same DPG primitives the auto-discovery uses for each filter's section — `[Reset]` section button, enable checkbox, per-setting `[X]` reset button, slider-with-value-label row.

Starting point for anyone picking up Phase 2: (a) find the function in `raven/avatar/settings_editor/app.py` that builds one filter's section from its metadata (near line 690), (b) read its DPG calls, (c) replicate them for the Crop section above the postprocessor panel. The screenshot in the chat history shows the visual target; if working in a fresh session, ask the user to re-send it.

### Panel placement

Above the postprocessor section, in the scrollable right column. Follows the same Reset / enable-checkbox / per-setting-reset pattern already used by Zoom, Bloom, etc. in the postprocessor panel:

```
[Reset] [ ] Crop
        [ ]        Show
        [X] [slider]  0.00   Left
        [X] [slider]  0.00   Top
        [X] [slider]  1.00   Right
        [X] [slider]  1.00   Bottom
```

where:
- `[Reset]` (section-level): resets all five controls (enable + four sliders) to defaults — enable off, sliders to full bbox.
- `[ ] Crop` (enable checkbox): directly bound to the `enabled` field in the crop dict. When off, the slider values are still persisted (in memory *and* in `animator.json`), so re-enabling restores the calibration. No separate "keep slider positions in memory" logic needed — `animator.json` is the memory.
- `[ ] Show`: toggles the crop-region overlay in the avatar panel — independent of Crop enable (you can hide the overlay even when crop is active).
- `[X]` (per-setting, one per slider): resets that one slider to its default — `0.0` for Left/Top, `1.0` for Right/Bottom.

Sliders operate directly on the four bbox keys (left/top/right/bottom, each in [0, 1]). No conversion anywhere — Phase 1's format change unified representation across the whole stack.

### Constraints

- `bbox["left"] ≤ bbox["right"]` and `bbox["top"] ≤ bbox["bottom"]`. When the user drags a slider past its partner, snap the partner along (standard "min can't exceed max" pattern). Or clamp the offending slider to the other. Pick one at implementation time based on DPG ergonomics.

### Crop region overlay

Superposed drawlist, z-ordered above the avatar image widget but **below the FPS counter** (which must remain readable on top). Draws:
- Four straight lines at the current bbox edges (in pixel coords = unit bbox × avatar_image_size, offset by the widget position).
- Maybe a semi-transparent mask over the cropped-away regions for clarity.

Lives in `DPGAvatarRenderer` so Librarian also gets it. Apps opt in by calling a new method:

```python
def configure_crop_overlay(self, show: Optional[bool]) -> None:
    """Show or hide the crop-region overlay. If `show is None`, toggle.

    Note: the overlay is only actually drawn when `show` is `True` AND the
    current `crop_bbox["enabled"]` is `True` — when crop is disabled, there
    is no active region to highlight.
    """
```

Mirrors the existing `configure_fps_counter(show: Optional[bool])` pattern.

Per-app defaults:
- **Settings editor**: calls `configure_crop_overlay(True)` at startup — the whole point of the app is calibration, and the overlay is the primary visual feedback. The `[ ] Show` checkbox (wired to this method) defaults to checked.
- **Librarian and other consumers**: don't call the method; overlay stays hidden.

### Four paths to handle symmetrically

Precedent: commit 71dd07d ("Fix settings editor crash on filters with !ignore parameters") is a direct template for this — it names the four places a settings field has to be touched whenever a new one is added. For auto-discovered postprocessor filters those four are `canonize`, `generate`, `GUI build`, and `reset`; two had a bug, two were correct. For Crop (hand-built, not auto-discovered), the conceptual counterparts are:

**Note on `Show` vs. data fields.** The five data fields (enable + four edges) persist into `animator.json`; the `Show` overlay toggle is a UI preference only — not saved, not loaded, set by the app at startup (`configure_crop_overlay(True)` in the settings editor). The four paths below refer to data-field handling; `Show` state is independent of all of them.

1. **Load** (`PostprocessorSettingsEditorGUI.load_animator_settings`): read `animator_settings["crop"]` from the JSON file → populate the enable checkbox and the four edge sliders. Do not touch the Show toggle.
2. **Save / send** (`PostprocessorSettingsEditorGUI.on_gui_settings_change`): read those same widgets → write to `self.animator_settings["crop"]` → push to server via `api.avatar_load_animator_settings(...)`. This is where the crop dict joins the rest of `custom_animator_settings`. Show is not part of the payload.
3. **GUI build**: create the Crop section widgets with initial data-field values taken from `self.animator_settings["crop"]` (or `NO_CROP` if not yet loaded). Show defaults to checked for the settings editor.
4. **Reset**: the section `[Reset]` button restores the five data controls (enable + 4 sliders) to defaults; each per-setting `[X]` restores one slider. Show is left alone — otherwise clicking Reset would unexpectedly hide the overlay the user was watching. Both variants must write back into the animator settings dict AND update the DPG widgets, otherwise the two go out of sync and the next save will re-write the old state.

Miss any one of these and the GUI will drift out of sync with the sent settings — exactly the class of bug 71dd07d fixed. Worth a quick four-point checklist at the end of Phase 2 implementation.

## Phase 3 — Server reorder: crop before upscale

### Rationale

Current order (`raven/server/modules/avatar.py`): `pose → animefx → upscale → crop → postprocess`. Postprocess already benefits from the smaller cropped region — that's the bulk of the ~50% speedup Phase 1 unlocks. The remaining inefficiency is that the upscaler processes blank space around the character.

With bicubic upscale, this is noise — it's nearly free either way. But Raven supports **Anime4K** as an upscaler too, and Anime4K is expensive enough that the blank-space processing eats into real-time budget on lower-end hardware. Reordering to crop-before-upscale is what makes Anime4K viable there with a narrow crop. Not an optional polish — it's a capability enabler.

### Complications

- The `Upscaler` is constructed with fixed input dimensions (preallocated tensors). Any crop change now forces an upscaler re-init — previously a cheap no-op.
- The postprocessor also cares about input resolution, but it re-inits itself lazily on the first `self.postprocessor.render_into(...)` call at the new size (`raven/common/video/postprocessor.py:512`). No explicit re-init needed in the handler — the render step handles it automatically.
- Mid-frame races: a client can update upscaler *or* crop settings while the render thread is mid-step. Currently there's nothing to lock because neither requires stateful re-init mid-render. After this change, it does.

### Fix

- A `threading.Lock` held across the crop → upscaler → postprocessor region of `render`, and across any re-init in the settings-update handler. Not across the whole render — keep it fine-grained to the stateful-resolution-sensitive section.
- Upscaler re-init path: when `crop` changes, recompute the cropped input size; if different from the current upscaler input size, release old upscaler and create a new one with the new size. Done in the settings-update handler under the lock.
- Postprocessor: nothing explicit. The next `render_into` call at the new resolution re-inits it. Because this happens inside the lock-protected region of `render`, a concurrent settings update waits its turn — no race.

### Testing

Drag the crop sliders during live render. Should not crash. Verify the total render time drops proportionally to the cropped area (measure with server metrics logging).

## Order of work

1. **Phase 0** — texture size reactive to incoming frames. Small and self-contained; fixes an existing issue; also deletes the software-rescale fallback.
2. **Phase 1** end-to-end — `animator.json` format change + server header + netutil parser generalisation + renderer bbox handling. Ship-ready on its own. **This is the phase that actually unlocks the ~50% speedup** (postprocess already benefits from crop server-side; the gain has been latent because the client was ignoring the setting).
3. **Phase 2** GUI. Makes the feature usable by humans without hand-editing JSON.
4. **Phase 3** server reorder + lock. What makes Anime4K viable on lower-end hardware with a narrow crop — nearly free with bicubic but a capability enabler with Anime4K.

Each phase is independently testable and doesn't strand the code in a half-working state if we stop after any of them.

## Decisions

- **Per-frame header carries crop only.** Client is the master copy for all other settings (it pushes updates to the server); nothing else needs echoing. The parser is nevertheless generalised (returns an `extra_headers` dict) so adding more per-frame data later is a one-line change on both ends.
- **Crop overlay lives in `DPGAvatarRenderer`.** Librarian also displays the avatar via this class and will want the same overlay for consistency. The settings editor just toggles its visibility via a new `DPGAvatarRenderer` method.
