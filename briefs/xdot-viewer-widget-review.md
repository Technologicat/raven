# XDot Widget Code Review Summary

Review conducted over two sessions (Feb 10–11, 2026) on the CC mega-refactor
that converted the monolithic wxPython xdot viewer into modular DPG components.

Reviewer: Juha (with Claude as review partner).

## Modules reviewed

- `__init__.py` — OK
- `constants.py` — new module, type aliases and color definitions collected here
- `graph.py` (~521 lines) — OK
- `highlight.py` (~225 lines) — OK with changes
- `hitdetect.py` — OK
- `parser.py` (~928 lines) — OK with changes
- `renderer.py` (~301 lines) — OK with changes
- `search.py` (~130 lines) — OK
- `viewport.py` (~316 lines) — OK with changes
- `widget.py` (~429 lines) — OK with changes

Also added: LICENSE files for xdottir/XDot provenance, and ColorBrewer color
definitions.

Not yet reviewed: tests (~478 lines), standalone app (~332 lines).

## Design decisions made

### Animation integration
- `XDotWidget` inherits from `raven.common.gui.animation.Animation` and is
  registered with the global `animator`.
- It acts as a **persistent updatable**: `render_frame` always returns
  `action_continue`. This follows the existing Visualizer convention.
- The widget's `update()` method drives both `Viewport.update()` and
  `HighlightState.update()` internally — the helpers are plain state objects,
  not `Animation` subclasses. Comments added to `highlight.py` and `viewport.py`
  to prevent accidental future "promotion" to `Animation`.

### Color mixing
- `mix_colors` in the xdot widget is a degenerate case of Porter-Duff "over"
  compositing (lerp with opaque background). Comment added noting this.
- Removed the 3-component (RGB) code path — `Color` is typealiased to RGBA,
  so the function should trust the type.

### Font handling
- `Pen.fontname` attribute removed. DPG's `draw_text` doesn't accept a font
  family parameter; fonts must be pre-registered via `dpg.add_font()` and bound
  with `dpg.bind_item_font()`. The Raven client apps handle font registration
  at startup.
- `draw_text` has a `size` parameter that scales the glyph texture quad, but
  rendering quality depends on the registered font size (ImGui texture atlas).

### Thread safety
- `HighlightState`: Added `RLock` for `_fading` dict (contention between
  `set_hover` and `update`) and `_programmatic` set.
- `SearchState`: No lock needed — `_results` is atomically replaced on search,
  and the graph is immutable after loading.

### Removed CC artifacts
- `_fracpart` subpixel tracking in `SmoothValue` — cargo-culted from the
  integer-valued `ScrollAnimation` during extraction. The viewport values are
  all floats, so no subpixel accumulation is needed. Removed entirely.

### Mouse handling in widget.py
- Global `dpg.handler_registry()` handlers with `_is_mouse_inside` guards —
  correct pattern for custom drawlist widgets. Item-level handlers wouldn't
  support hover-exit detection.
- `_on_mouse_move`: gated `_needs_render` on actual hover change (was
  unconditionally true on every mouse move inside the widget).
- Mouse position utilities extracted to `raven.common.gui.utils` (new:
  `get_mouse_relative_pos`; existing: `is_mouse_inside_widget`).

## TODOs flagged for CC

### Extract `SmoothValue` interpolation into `raven.common.gui.animation`
The FPS-corrected exponential decay algorithm now exists in three places:
1. `raven.avatar` — float-valued pose animation (canonical, fully documented)
2. `raven.common.gui.animation.SmoothScrolling` — int-valued with subpixel tracking
3. `xdotwidget.viewport.SmoothValue` — float-valued pan/zoom

Extract a single implementation with optional subpixel mode. The avatar version
has the authoritative derivation in its comments.

### Unify `HighlightState._programmatic` and `._programmatic_node_ids`
These are currently two independent highlight channels that only meet in the
query methods. `clear_programmatic` clears both, implying they're one feature.
Either unify (resolve node IDs to elements eagerly) or document clearly as
independent channels. Depends on whether highlights are ever set before the
graph is fully parsed.

### Filled Bézier curves
DPG has no native filled Bézier support. Current code approximates with a
polygon through the control points. For complex node shapes this will be
visually incorrect. Fix would be to tessellate Bézier segments into polyline
points first. Low priority — call graphs and chat trees (the main use cases)
are unlikely to produce filled Béziers.

### Text justification in renderer
`_render_text_shape`: CENTER and RIGHT justification both fall through to
left-aligned. Fixing requires measuring text width, which DPG doesn't
directly expose for `draw_text`. Note: Librarian's avatar subtitler has
text size measurement code, but it requires an off-screen render and
`dpg.split_frame()` (a one-frame delay). Calling `split_frame` from the
per-frame render path will deadlock DPG — it's safe from other threads,
but not from the animation update thread.

### Property-style accessors
Several classes use explicit getter/setter methods where `@property` would be
more Pythonic. Low priority — consistency within the subsystem matters more
than matching the broader codebase. If changed, Juha prefers the
`thing = property(fget=get_thing, ...)` pattern over decorator stacking.

### Parser token types as enum
Considered, decided against. The string constants are self-documenting, the
parser is self-contained, and the terseness aids readability in pattern matching.

### `Viewport.set_size` caller verification
Need to verify in the second pass (inter-module wiring) that `set_size` is
actually called when the DPG widget resizes. Widget.py has a `set_size` method
that forwards to the viewport, but need to confirm the resize event is wired.

## Notes for CC coding style

- Code density target is ~30% code, ~70% comments for complex algorithmic code.
  Explain *why*, not just *what*.
- Math-heavy comments with derivations are encouraged when the algorithm warrants it.
  See the avatar `interpolate` method for the gold standard.

## Files modified outside the xdot widget package

- `raven.common.gui.utils`: Added `get_mouse_relative_pos()`.

## Regex performance note
In `Lexer`: `r"\r\n?|\n"` — the `?` causes backtracking on bare `\r`. Not
a problem at dot-file scale, but flagged with a comment for awareness. (Cf.
tensor index shuffling performance cliffs in the avatar system.)
