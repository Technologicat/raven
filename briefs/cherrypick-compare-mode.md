# Compare Mode — Implementation Plan

Reference: SPEC.md §7 (Animation / Compare Mode).

## Overview

Compare mode is an overlay on normal operation that cycles through a set of
multi-selected images in the main view, enabling rapid A/B/C comparison of
variants. Up to 9 images, numbered 1–9, cycling at a configurable FPS.

## New file: `compare.py` (~200–250 lines)

### `CompareMode` class

**State:**
- `active: bool` — whether compare mode is running
- `warming_up: bool` — pre-caching in progress (not yet cycling)
- `paused: bool` — cycling paused (Space toggle)
- `fps: float` — current cycling speed
- `frame_list: list[int]` — image indices (first 9 selected, in grid-visible
  order), max `COMPARE_MAX_IMAGES`
- `frame_idx: int` — current position in `frame_list`
- `saved_current: int` — grid.current before entering, for Escape restore
- `frame_start_t: int` — `time.monotonic_ns()` when current frame started
  (for cycling timer and fade calculation)

**Constructor:**
- References to `image_view`, `grid`, `preload`, `triage` (from `_app_state`)
- No DPG items created in `__init__` — the overlay lives in ImageView,
  badges live in Grid

**Public methods:**

`enter(indices: Sequence[int]) -> None`
  - Validate: need ≥2 indices. Cap to first 9 in grid-visible order.
  - Save `grid.current` as `saved_current`.
  - Set `warming_up = True`, `active = True`.
  - Call `preload.schedule_compare(indices, triage)` to pre-cache.
  - Start a polling loop (in `tick()`) that checks cache readiness
    and updates status bar with progress.

`exit(restore: bool = True) -> None`
  - Set `active = False`, `warming_up = False`, `paused = False`.
  - Clear grid badges, clear ImageView number overlay.
  - Clear grid compare-active highlight.
  - **Always** call `preload.unpin_all()` — `finally`-style, regardless
    of exit reason (Escape, digit select, or any future exit path).
  - If `restore`: navigate to `saved_current`.
  - Leave the selection unchanged (user can act on it post-exit).
  - Restore toolbar button to ICON_PLAY state and original tooltip.
  - Update status bar (normal mode text).

`select_frame(n: int) -> None`
  - Digit key handler (1–9). Exit compare mode, navigate to the
    nth image in `frame_list` (0-indexed: key 1 → index 0).
  - Calls `exit(restore=False)` since we're explicitly navigating.

`tick() -> None`
  - Called every frame from the render loop.
  - **Warm-up phase**: check if all `frame_list` indices are in
    the preload cache (`preload.is_cached(idx)`). Update status bar
    with progress: `"Compare mode initializing [3/9]..."`. Once all
    ready: set `warming_up = False`, set grid badges, show ImageView
    number overlay, display first frame, set `frame_start_t`.
  - **Cycling phase** (not paused): if `monotonic_ns() - frame_start_t
    >= 10**9 / fps`, advance to next frame (wrapping). On advance:
    switch image in view (via the preload cache → ImageView path),
    update grid compare-active highlight, reset `frame_start_t`,
    update ImageView overlay number.
  - **Paused**: skip the time check. The active tile highlight keeps
    pulsating with the full 1→0→1 cycle at `COMPARE_PAUSED_PULSE_DURATION`
    (default 2.0s, matching other Raven pulsating indicators).

`toggle_pause() -> None`
  - Toggle `paused`. On unpause, reset `frame_start_t` to now
    (prevents immediate advance from accumulated time).

`adjust_fps(delta: float) -> None`
  - Clamp `fps` to [COMPARE_MIN_FPS, COMPARE_MAX_FPS].

`reset_fps() -> None`
  - Reset `fps` to `COMPARE_DEFAULT_FPS`.

`fade_alpha() -> float`
  - Returns [0, 1] for the grid's compare-active tile overlay.
  - **Cycling**: uses `pulsation_envelope(t)` from
    `raven.common.gui.animation` with effective cycle = `2 / fps`,
    so only the first half (1→0 fade-out) plays before the frame
    advances and resets.
  - **Paused**: uses `pulsation_envelope(t)` with the full 1→0→1
    cycle at `COMPARE_PAUSED_PULSE_DURATION`.

## Changes to `raven/common/gui/animation.py`

### New utility: `pulsation_envelope(t: float) -> float`

```python
def pulsation_envelope(t: float) -> float:
    """Cosine-squared envelope: 1 at *t*=0, 0 at *t*=0.5, 1 at *t*=1.

    Used by `PulsatingColor` and available for manual alpha calculations
    (e.g. compare mode tile fade).  *t* is the normalized cycle position
    [0, 1], where 0 and 1 are cycle boundaries.
    """
    return math.cos(t * math.pi) ** 2
```

Add to `__all__` (PEP 8).

Refactor `PulsatingColor.render_frame` to call this function instead
of inlining the formula. Single source of truth.

## Changes to `preload.py`

### Rename existing method

`schedule` → `schedule_neighbors` — makes it clear this is the
automatic cross-neighborhood preloader driven by navigation.

### New method: `schedule_compare(indices, triage) -> None`

Pre-cache a specific set of image indices for compare mode.

- Cancel any pending neighbor-preload tasks (free GPU bandwidth).
- Submit tasks for indices not already cached, respecting the
  existing `_preload_one` task function.
- Use full mip chain (no `max_scale` cap) — compare mode preserves
  whatever zoom level is active when it starts, so we need the larger
  mip levels for quality display at any zoom.
- Pin compare-mode entries: add `_pinned: set[int]` field.
  Eviction skips pinned entries. `schedule_neighbors` also skips
  evicting pinned entries.

### New method: `unpin_all() -> None`

Called from `CompareMode.exit()`. Clears `_pinned`. Because `exit()`
is the single exit path for all compare mode termination (Escape,
digit select, future paths), this provides `finally`-style cleanup.

### New method: `is_cached(idx: int) -> bool`

Check if an index is in the cache (for warm-up progress polling).

### New method: `compare_progress(indices) -> tuple[int, int]`

Returns `(n_cached, n_total)` for warm-up status bar display.

## Changes to `grid.py`

### `set_compare_badges(mapping: Mapping[int, int]) -> None`

Show number badges (1–9) on tiles. `mapping` is `{image_idx: badge_number}`.
Drawn as text on the tile drawlist (top-right corner, small font,
50% gray translucent background with white number for readability
against any image content).

### `clear_compare_badges() -> None`

Remove all badges.

### `set_compare_active(idx: int, alpha: float) -> None`

Draw a bright overlay on the active compare tile (the one currently
shown in the main view). `alpha` is the fade-out progress from
`CompareMode.fade_alpha()`. Color: bright blue tint, matching
`CURRENT_COLOR` but more prominent. We'll test visually and adjust.

### `clear_compare_active() -> None`

Remove the active-compare highlight.

## Changes to `imageview.py`

### Number overlay

A large semi-transparent digit (1–9) at the **top-right** of the image
pane — near the grid view, so the user can see both the large number
and the tile badges simultaneously without scanning across the full
window width.

Implementation: `dpg.draw_text()` on the drawlist. Uses the default
font loaded at a large size via `guiutils.load_extra_font()` — DPG
drawlist text is rendered from the font texture, so a small-size font
scaled up looks blurry. See raven-xdot-viewer for the extra-font
loading pattern.

Since we draw manually on the drawlist, there's no DPG widget `pos`
to manage — just redraw at the correct coordinates in `_render()`
when compare mode is active (accounting for current view size).

### `set_overlay_number(n: Optional[int]) -> None`

- `n is None`: hide the overlay.
- `n in 1..9`: show the digit at top-right.

### `clear_overlay() -> None`

Convenience alias for `set_overlay_number(None)`.

## Changes to `app.py`

### Toolbar button

Add a "Compare" button after the triage buttons. FontAwesome
`ICON_PLAY`, tooltip: `"Compare selected [Enter]"`.

Use `disablable_button_theme` (from `themes_and_fonts`) so the button
visually indicates enabled/disabled state. Disable when
`len(grid.selected) < 2`. The callback and hotkey handler both check
`dpg.is_item_enabled()` before acting (hotkey path bypasses the
button, so the check must be explicit).

When compare mode starts (including warm-up), the button changes to
`ICON_STOP` with tooltip `"Exit compare mode [Esc]"`. On exit, it
reverts to `ICON_PLAY` with tooltip `"Compare selected [Enter]"`.

Flash the button green on activation via `ButtonFlash` (Raven
convention for acknowledging a click or hotkey press).

### Hotkey routing

In `_on_key`, when `compare.active` (cycling or paused):
- `1`–`9`: `compare.select_frame(n)`.
- `Escape`: `compare.exit(restore=True)`.
- `,`: `compare.adjust_fps(-COMPARE_FPS_STEP)`.
- `.`: `compare.adjust_fps(+COMPARE_FPS_STEP)`.
- `M`: `compare.reset_fps()`.
- `Space`: `compare.toggle_pause()`.
- Zoom keys (`+`, `-`, `F`, mouse wheel): **tentatively available** —
  test whether zoom during cycling feels usable. May need to suppress
  at high FPS if it causes flicker. Decision deferred to manual testing.
  Note: `1` (zoom to 1:1) is NOT available — conflicts with digit keys.
- `F1`: help card (stays available).
- `F11`: fullscreen (stays available).
- All other keys: suppressed (no triage, no navigation).

When `compare.warming_up`:
- `Escape`: `compare.exit()` — cancel pre-caching.
- All others: suppressed.

### Render loop

Call `compare.tick()` each frame (before `animator.render_frame()`).

### Status bar

When compare mode is active:
- Warm-up: `"Compare mode initializing [3/9]..."`
- Cycling: `"Compare [2/5] | 3.0 FPS"`
- Paused: `"Compare [2/5] | PAUSED"`

When compare mode exits: restore normal status bar text
(via existing `_update_status()`).

## Changes to `config.py`

Add:
```python
COMPARE_MAX_IMAGES = 9
COMPARE_FPS_STEP = 0.5  # comma/period increment
COMPARE_FADE_COLOR = (80, 160, 255, 180)  # bright blue tint for active tile
COMPARE_PAUSED_PULSE_DURATION = 2.0  # seconds, full 1→0→1 cycle when paused
```

Existing values stay: `COMPARE_DEFAULT_FPS`, `COMPARE_MIN_FPS`,
`COMPARE_MAX_FPS`.

## Hotkey summary (compare mode active)

| Key         | Action                                    |
|-------------|-------------------------------------------|
| `1`–`9`     | Select that image, exit compare mode      |
| `Escape`    | Return to saved image, exit compare mode  |
| `,`         | Decrease FPS by step                      |
| `.`         | Increase FPS by step                      |
| `M`         | Reset FPS to default                      |
| `Space`     | Pause / resume cycling                    |
| `+` / `-`   | Zoom in / out (tentative — test needed)   |
| `F`         | Zoom to fit (tentative — test needed)     |
| Mouse wheel | Zoom at cursor (tentative — test needed)  |
| `F1`        | Help card                                 |
| `F11`       | Fullscreen                                |

All other keys suppressed.

## Implementation order

1. `animation.py` — extract `pulsation_envelope`, refactor PulsatingColor
2. `config.py` — add compare config constants
3. `preload.py` — rename `schedule` → `schedule_neighbors`, add
   `schedule_compare`, `unpin_all`, `is_cached`, `compare_progress`
4. `grid.py` — add `set_compare_badges`, `clear_compare_badges`,
   `set_compare_active`, `clear_compare_active`
5. `imageview.py` — add `set_overlay_number`, `clear_overlay`
6. `compare.py` — the `CompareMode` class
7. `app.py` — toolbar button, hotkey routing, render loop integration,
   status bar
8. Help card — add compare mode hotkeys (together with overflow fix)
9. Manual testing with real image folders
10. Update SPEC.md §7 to mark as implemented

## Typing convention note

Parameter types use abstract types from `collections.abc`
(`Mapping[K, V]`, `Sequence[T]`, `Iterable[T]`) for
widest-possible-accepted semantics. Return types use concrete
lowercase builtins (`tuple[int, int]`, `list[int]`, `dict[K, V]`) —
PEP 585, Python 3.9+. The capitalized `typing` forms (`Dict`,
`List`, `Tuple`) are deprecated aliases for the builtins and offer
no extra width — avoid them. See deferred TODO for auditing
existing code.
