# XDot Viewer App & Tests Review Summary

Review conducted Feb 11, 2026 on the standalone XDot viewer application
(`raven.xdot_viewer.app`) and the xdot widget unit tests
(`raven.common.gui.xdotwidget.tests`).

Continues from the widget code review (Feb 10–11).

Reviewer: Juha (with Claude as review partner).

## Modules reviewed

- `raven.xdot_viewer.app` (~290 lines) — standalone viewer application
- `raven.common.gui.xdotwidget.tests.__init__.py` — OK as-is
- `raven.common.gui.xdotwidget.tests.test_parser.py` (~140 lines) — rewritten
- `raven.common.gui.xdotwidget.tests.test_search.py` (~115 lines) — rewritten
- `raven.common.gui.xdotwidget.tests.test_viewport.py` (~110 lines) — rewritten

Reference modules consulted (not modified):
- `raven.common.gui.animation` — for animator integration pattern
- `raven.common.gui.utils` — for `bootup` call order contract
- `raven.librarian.app` — canonical DPG app structure reference
- `raven.common.gui.xdotwidget.widget` — for search/highlight wiring
- `raven.common.gui.xdotwidget.graph` — data model for test assertions
- `raven.common.gui.xdotwidget.parser` — parser internals for test assertions
- `raven.common.gui.xdotwidget.search` — search state for test coverage
- `raven.common.gui.xdotwidget.viewport` — viewport math for test verification
- `raven.common.gui.xdotwidget.constants` — `Color` type alias

## App fixes applied

### Structural (from Librarian reference)
- **DPG startup order corrected**: `create_context` → `bootup` → `create_viewport` → `setup_dearpygui`, matching the `bootup` docstring contract. Was previously calling `bootup` after `create_viewport`.
- **Animator integration**: Render loop now calls `animator.render_frame()` instead of manual `widget.update()`. Widget is driven by the global animator as a persistent `Animation` subclass.
- **Shutdown callback**: Added `dpg.set_exit_callback(_gui_shutdown)` with `animator.clear()`.
- **KeyboardInterrupt handling**: `try/except` around the render loop, matching Librarian pattern.
- **Import cleanup**: Removed unused imports, added `animator` import from `raven.common.gui.animation`.

### Bug fixes
- **Path resolution**: `_load_file` now uses `common_utils.absolutize_filename` so mtime comparisons remain valid if CWD changes.
- **mtime comparison**: Changed `!=` to `>` — only reload if file is newer, avoids float equality fragility.
- **DPG callback signatures**: All callbacks now accept `*_args` to handle DPG's `(sender, app_data, user_data)` arguments. Removed the `lambda: _do_search()` wrapper — `_do_search` is now passed directly.
- **Search/highlight wiring**: `_do_search` now calls `widget.set_highlighted_nodes()` after `widget.search()` — the two subsystems are independent in the widget.

### UX improvements
- **Keyboard handler restructured**: Focus-aware dispatch using `dpg.is_item_focused()`. Search input focus blocks bare-key shortcuts.
- **Changed N/Shift+N to F3/Shift+F3**: No conflict with text input, consistent with rest of Raven (and DOS tradition).
- **Added Esc handling**: Unfocuses search input (DPG auto-reverts content).
- **Added Enter handling**: Commits search and unfocuses.
- **Added arrow key panning**: New `widget.pan_by()` method wired to arrow keys.
- **FontAwesome toolbar icons**: Consistent with rest of Raven. Icons bound with `themes_and_fonts.icon_font_solid`/`icon_font_regular`.
- **Tooltips on toolbar buttons**: Show button function and keyboard shortcut (discoverable hotkeys).
- **Live search**: Removed `on_enter`, search updates incrementally on every keystroke.
- **Search field color feedback**: Green for matches, red for no matches, white when empty. Via DPG theme color.
- **Search field `width=-1`**: Moved to rightmost toolbar position, auto-sizes to fill available space.
- **Zoom-to-node on click**: Click handler now calls `widget.zoom_to_node()`.
- **Version display**: `--version` flag in argparse, version in viewport title.

### Code quality
- **Named layout constants**: `_WIDGET_H_PADDING`, `_WIDGET_V_PADDING` replacing magic numbers.
- **Logging**: Module-level logger with `logging.basicConfig`, `timer` for import timing, matching Librarian pattern. Error messages via `logger.error` instead of `print(..., file=sys.stderr)`.
- **`bootup` error handling**: Let it propagate (fail-fast). Fonts are part of the Raven install.

## Bug found in `graph.py`

**`filter_items_by_text` unconditionally lowercases both query and text.**
Intended behavior is Emacs-style smart-case: lowercase fragment → case-insensitive, fragment with uppercase → case-sensitive, checked per fragment independently. Fixed during the session using `common_utils.search_string_to_fragments`, consistent with Visualizer.

Also fixed: `_items_and_texts` was pre-computing lowercased text, breaking the case-sensitive path. Now stores original-case text.

## TODOs remaining in app

- **Viewport resize handling**: Widget sizes don't track viewport resizes. Need to wire a resize callback, extractable from Librarian's pattern. Flagged in code.
- **File dialog / Ctrl+O**: No interactive file opening yet. Recyclable from the avatar pose editor. Flagged in code.
- **Async loading for large graphs**: UI freezes during `subprocess.run(["dot", ...])` for large graphs. Replace with background thread + "Loading..." placeholder graph. Flagged in code.
- **Follow-edge-on-click**: Feature from xdottir that was lost in the refactor. Flagged in click handler.
- **Nordic keyboard layout**: `+`/`=` zoom shortcuts may not work on non-US layouts. Needs testing. Flagged in code.
- **GUI error dialogs**: Error messages currently go to logger only. TODO to additionally use `raven.common.gui.messagebox` for modal GUI error dialogs.
- **Search field width accessibility**: Green/red color feedback; may need blue/orange for colorblind users.

## Test improvements

### test_parser.py — rewritten

**New test classes:**
- `TestYTransform` — verifies X (preserved) and Y (flipped) for nodes, ellipse centers, and edge Bézier points. Documents the transform math.
- `TestShapeParsing` — verifies shape types and content: ellipse dimensions, fill state, text content, Bézier control point count and coordinates, arrowhead polygon vertices.
- `TestParseError` — malformed input raises `ParseError`, missing bb degrades gracefully, `ParseError.__str__` is informative.

**Strengthened existing tests:**
- `TestColorParsing` — now checks actual RGBA values. Uses `#1a4b7c` (all components nonzero and distinct) to catch channel swizzle. Added hex-with-alpha, HSV (same color via `colorsys` round-trip), named color, and fill color (`C` vs `c` opcode) tests.
- `TestGraphSearch` — smart-case tests: lowercase query matches case-insensitively, uppercase query is case-sensitive. Checks specific nodes in results, not just counts.

### test_search.py — rewritten

**New tests:**
- Smart-case section: lowercase, uppercase, mixed-case, and per-fragment smart-case.
- `test_search_returns_results` — verifies return value matches `get_results()`.
- `test_multi_fragment_order_independent` — "cat photo" == "photo cat".
- Navigation edge cases: no results, single result wraparound.
- `test_set_new_graph_clears_results` — now verifies query is retained.
- `test_set_new_graph_reruns_search` — verifies active query re-runs on new graph.

### test_viewport.py — rewritten

**`TestSmoothValue` additions:**
- Monotonicity tests (increasing and decreasing) — no overshoot.
- `test_higher_rate_converges_faster` — comparative convergence test.
- `test_is_not_animating_after_immediate`.
- Behavioral contract documented in class docstring: monotonic, reaches target, higher rate = faster. No assumption about specific interpolation.

**`TestViewport` additions:**
- `test_zoom_to_fit_zoom_level` — verifies `min(vp_w/g_w, vp_h/g_h)`.
- `test_zoom_to_fit_with_margin` — comparative: margin reduces zoom.
- `test_zoom_to_fit_degenerate_graph` — zero-size graph.
- `test_is_visible_partial_overlap` — box straddling viewport edge.
- `test_is_visible_fully_enclosing` — box larger than viewport.
- `test_pan_by_zoom_scaling` — pan delta inversely proportional to zoom.
- `test_set_size` — basic dimension update.

### General test improvements
- Shared `_approx` helper across test files.
- Tests assert behavior and content, not just "didn't crash" (upgraded from smoke tests).
- xdot fixture strings documented with expected transform results.

## Design decisions

### Smart-case search (Emacs HELM style)
Per-fragment case sensitivity: lowercase fragment is case-insensitive, fragment containing any uppercase letter is case-sensitive. Applied consistently in `Graph.filter_items_by_text` and tested in both `test_parser.py` and `test_search.py`. Uses `common_utils.search_string_to_fragments` (shared with Visualizer).

### Test philosophy
- Tests assert the **behavioral contract**, not the implementation.
- `SmoothValue` tests check monotonicity and convergence, not exponential decay math.
- `pan_by` tests check direction, not exact delta (implementation may change).
- Known-failing tests document intended behavior (smart-case was initially failing against the unconditional-lowercase implementation).

## Next steps

1. Boot up the viewer — "but does it work?" moment.
2. Generate test graphs (hand-written dot, or quick `ast`-based call graph).
3. Wire viewport resize handling (extract from Librarian).
4. Add file dialog (extract from avatar pose editor).
