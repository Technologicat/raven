# CC Briefing: GraphViz Filter Switching for raven-xdot-viewer

## Goal

Add a layout engine selector to `raven-xdot-viewer` so the user can re-render the currently loaded graph with different GraphViz layout engines (`dot`, `neato`, `fdp`, `sfdp`, `circo`, `twopi`). The view should preserve visual focus across re-layouts when possible.

This feature existed in the predecessor tool, XDottir, and was useful for exploring large call graphs from different perspectives.

## Context

Read these files before starting:

- `CLAUDE.md` (project root) — project overview, patterns, style
- `raven-style-guide.md` — coding conventions
- `raven/xdot_viewer/app.py` — the viewer app (all changes here + config)
- `raven/xdot_viewer/config.py` — constants
- `raven/common/gui/xdotwidget/widget.py` — `XDotWidget` public API
- `raven/common/gui/xdotwidget/viewport.py` — `Viewport` (pan/zoom, coordinate transforms)
- `raven/common/gui/xdotwidget/graph.py` — `Graph`, `Node`, `Edge` data model
- `raven/common/gui/xdotwidget/parser.py` — `DotScanner`, `DotLexer`, `DotParser`, `XDotParser`. The scanner and lexer are reused for attribute stripping (see §2).
- `raven/avatar/settings_editor/app.py` — reference for the combo keyboard navigation pattern (lines ~1256–1341). Read this for §6a.

## Design

### 1. Store the raw DOT source

Currently `_load_file` returns xdot code and `_open_file` passes it directly to `widget.set_xdotcode()`. We need to keep the *layout-stripped* DOT source around so we can re-run any engine on it.

Add to `_app_state`:

```python
"dot_source": None,       # layout-stripped DOT source (str)
"original_xdotcode": None,  # original file contents for .xdot fast path (str or None)
"current_filter": "[as-is]",  # active layout engine selection
```

### 2. Strip xdot layout attributes

Write a function `_strip_xdot_layout_attrs(xdotcode: str) -> str` that removes all layout-related attributes from an xdot string. This produces clean DOT that any layout engine can process from scratch.

**Attributes to strip** (at all levels — graph, subgraph, node, edge):

- `bb`
- `pos`
- `lp`
- `rects`
- `_background`
- `_draw_`
- `_ldraw_`
- `_hdraw_`
- `_tdraw_`
- `_hldraw_`
- `_tldraw_`

**Leave alone:** `width`, `height` on nodes — these may be user-specified, and layout engines treat them as minimum size hints.

**Always strip, unconditionally**, regardless of whether the input was `.dot` or `.xdot`. This is simpler than trying to detect which case we're in, and engines recompute everything anyway.

**Implementation approach — use the existing DOT scanner, not regex.** The codebase already has a proper DOT lexer in `raven/common/gui/xdotwidget/parser.py` (`DotScanner`) that correctly handles quoted strings (with escaped quotes), HTML labels (`<...>`), comments, and all the lexical edge cases. Use it.

`DotScanner.next(buf, pos)` returns `(token_type, text, end_pos)`. Since we know `start_pos = end_pos - len(text)`, we can track exact byte ranges in the original buffer. The approach:

1. Tokenize the input using `DotScanner`.
2. Track bracket depth (`[` / `]`) to know when we're inside attribute lists.
3. Inside `[...]`, watch for the pattern `name = value`. When `name` (case-insensitive) is in the layout attrs set, record the byte range of the entire `name = value` sequence (plus any trailing comma, semicolon, or whitespace separator) for deletion.
4. Build the output by copying everything from the original buffer *except* the recorded deletion ranges.
5. Clean up any resulting empty attribute lists (`[]`) or double separators — but be conservative; a few harmless empty brackets or extra commas won't affect GraphViz.

Working from the original buffer (not reconstructed token text) is important because the lexer's `_filter` method strips quotes from `STR_ID` and angle brackets from `HTML_ID`, so round-tripping through token text would lose those. By tracking positions and splicing the original string, we preserve everything we don't explicitly remove.

Note: `DotScanner` is case-insensitive (`.ignorecase = True`), and attribute names in DOT are case-insensitive, so this works naturally.

**Implementation note:** This requires a small state machine on top of the scanner's token stream — tracking bracket depth and whether the current tokens form a `name = value` pair to delete. Don't try to shortcut this; the states matter (e.g. an `=` outside brackets is a graph-level assignment, not an attribute). Keep the state machine explicit and easy to follow.

### 3. Modify file loading flow

In `_load_file`:

- For `.dot`/`.gv` files: read the raw source. Strip layout attrs → store as `dot_source`. Set `original_xdotcode = None`. Determine the effective engine: if the current filter is `[as-is]`, use `GRAPHVIZ_ENGINES[1]`; otherwise use the selected engine. Run GraphViz. Return xdot output.
- For `.xdot` files: read the raw source → store as `original_xdotcode`. Strip layout attrs → store as `dot_source`. Then decide what to return:
  - If the current filter is `[as-is]`: return the **original file contents** directly as xdot code (fast path — no GraphViz needed, uses the file's own pre-rendered layout).
  - Otherwise: run the selected engine on the stripped source. Return xdot output.

This preserves the existing behavior that `.xdot` files can be viewed without GraphViz installed, as long as the user doesn't switch away from `[as-is]`.

Factor out the "run a GraphViz engine" step:

```python
def _run_graphviz(dot_source: str, engine: str = "dot") -> Optional[str]:
    """Run a GraphViz layout engine on DOT source, return xdot code."""
    try:
        result = subprocess.run(
            [engine, "-Txdot"],
            input=dot_source,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except FileNotFoundError:
        logger.error(f"_run_graphviz: `{engine}` command not found. Please install GraphViz.")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"_run_graphviz: Error running `{engine}`: {e.stderr}")
        return None
```

Note: pass source via `input=` (stdin), not as a filename argument, since we're working with the stripped string.

### 4. Filter switching function

```python
def _apply_filter(engine: str) -> None:
    """Re-render the current graph with a different layout engine."""
```

This function:

1. Reads `_app_state["dot_source"]`. If None, return.
2. Determines the focus anchor (see §5 below).
3. Determines what xdot code to use:
   - If `engine == "[as-is]"` and `_app_state["original_xdotcode"]` is not None (i.e. the file was `.xdot`): use the original xdot code directly (fast path).
   - If `engine == "[as-is]"` and `original_xdotcode` is None (i.e. `.dot`/`.gv` file): run `_run_graphviz(dot_source, config.GRAPHVIZ_ENGINES[1])` (fall back to first real engine).
   - Otherwise: run `_run_graphviz(dot_source, engine)`.
4. If successful, calls `widget.set_xdotcode(xdotcode)`.
5. Restores focus (see §5).
6. Updates `_app_state["current_filter"]`.
7. Updates status bar.

### 5. Focus preservation across re-layout

Before re-layout:

1. Get the viewport center in graph coordinates and zoom level via a new public accessor on `XDotWidget` (see "Files to modify" below).
2. Get visible bounds: `widget.get_visible_bounds()` (delegates to `_viewport.get_visible_bounds()`) → `(vx1, vy1, vx2, vy2)`.
3. Compute the threshold radius as 25% of the half-extent of the shorter viewport axis:
   ```python
   threshold = 0.25 * min(vx2 - vx1, vy2 - vy1) / 2
   ```
4. For each node in the current graph, compute Euclidean distance from node center `(node.x, node.y)` to viewport center.
5. Sort by distance. Let `d1` = nearest distance, `d2` = second nearest.
6. **Anchor condition:** `d1 <= threshold` AND (`d2 > 1.5 * d1` OR only one node exists).
7. If anchored: save the node's `internal_name` and the current zoom level.

After re-layout:

1. If we have an anchor name, look it up in `widget.get_graph().nodes_by_name`.
2. If found: `widget.pan_to_node(anchor_name, animate=False)`, then restore zoom via `widget.set_zoom(saved_zoom)`. Request render.
3. Otherwise: `widget.zoom_to_fit(animate=False)`.

Edge case: if the graph has zero nodes (unlikely but defensive), fall back to zoom-to-fit.

### 6. UI: toolbar combo

Add a DPG combo (dropdown) in the toolbar, between the dark mode button and the help button. Available engines:

```python
GRAPHVIZ_ENGINES = ["[as-is]", "dot", "neato", "fdp", "sfdp", "circo", "twopi"]
```

Put this list in `config.py`.

The `[as-is]` entry means:

- For `.xdot` files: show the original pre-rendered layout (fast path — no GraphViz needed).
- For `.dot`/`.gv` files: fall back to `GRAPHVIZ_ENGINES[1]` (since there's no pre-rendered layout to show).

This is the default selection. All other entries always run the named engine.

```python
dpg.add_combo(
    items=config.GRAPHVIZ_ENGINES,
    default_value=config.GRAPHVIZ_ENGINES[0],
    tag="filter_combo",
    callback=_on_filter_changed,
    width=80,
)
with dpg.tooltip("filter_combo"):
    dpg.add_text("GraphViz layout engine\n(Ctrl+E; then Up, Down, Home, End to jump; Esc to return)")
```

The callback:

```python
def _on_filter_changed(sender, app_data, user_data):
    _apply_filter(app_data)
```

### 6a. Keyboard shortcut for engine selector

**Ctrl+E** focuses the combo. While the combo is focused, bare **Up/Down/Home/End** keys navigate through the engine list.

Follow the pattern from `raven/avatar/settings_editor/app.py` (lines ~1256–1341). Key elements:

1. In the `ctrl_pressed` branch of `_on_key`, add:
   ```python
   elif key == dpg.mvKey_E:
       dpg.focus_item("filter_combo")
   ```

2. Add a `_combobox_choice_map` that maps combo widget tags to `(choices_list, callback)` tuples:
   ```python
   _combobox_choice_map = {"filter_combo": (config.GRAPHVIZ_ENGINES, _on_filter_changed)}
   ```

3. In the bare-key branch of `_on_key`, when the focused item is in the choice map, use a `_browse_combo` helper to handle Up/Down/Home/End/Esc. The helper:
   - Gets the current value via `dpg.get_value(combo_widget)`.
   - For Up/Down/Home/End: computes new index, sets it via `dpg.set_value(combo_widget, new_choice)`, and **manually calls the callback** — `dpg.set_value` doesn't trigger it.
   - For Esc: re-focuses the graph widget via `dpg.focus_item(widget.get_dpg_widget_id())`, so the user can return to arrow-key panning without reaching for the mouse.

4. Use `dpg.get_item_alias(dpg.get_focused_item())` to get the string tag of the focused item for the map lookup, since the xdot viewer uses string tags throughout.

Note: the settings editor builds the map lazily on first use because it depends on an instance. Here we can build it at module level since all the pieces (`config.GRAPHVIZ_ENGINES`, `_on_filter_changed`) are available statically.

### 7. Auto-reload interaction

`_check_file_reload` currently calls `_open_file` when the file mtime changes. After this feature, `_open_file` should re-read the file, strip and store `dot_source`, update `original_xdotcode` (for `.xdot` files), and re-render using the *currently selected* filter (not always `dot`). So auto-reload respects the active engine choice. The `.xdot` fast path still applies during auto-reload if the filter is `[as-is]`.

### 8. Help card update

Add entries to the help card's hotkey table in the first column (search & file group), since these are UI focus actions like Ctrl+F:

```python
env(key_indent=0, key="Ctrl+E", action_indent=0, action="Focus layout engine selector", notes=""),
env(key_indent=1, key="Up / Down", action_indent=0, action="Previous / next engine", notes="While engine selector focused"),
env(key_indent=1, key="Home / End", action_indent=0, action="First / last engine", notes="While engine selector focused"),
env(key_indent=1, key="Esc", action_indent=0, action="Return to graph view", notes="While engine selector focused"),
```

### 9. Status bar

When switching filters, show:

- For the fast path (`[as-is]` on `.xdot`): `"Original layout: {filepath}"`.
- For engine runs: `"Rendered with {engine}: {filepath}"`.
- For `[as-is]` fallback on `.dot`/`.gv`: `"Rendered with {engine}: {filepath}"` (where `engine` is `GRAPHVIZ_ENGINES[1]`).

## Files to modify

- `raven/xdot_viewer/app.py` — main changes (strip function, load flow, filter switching, focus preservation, toolbar combo, auto-reload fix)
- `raven/xdot_viewer/config.py` — add `GRAPHVIZ_ENGINES` list
- `raven/common/gui/xdotwidget/widget.py` — add thin public accessors for viewport state:
  - `get_view_center() -> Tuple[float, float]` — returns `(pan_x.current, pan_y.current)` in graph coordinates.
  - `get_zoom() -> float` — returns `zoom.current`.
  - `set_zoom(zoom: float, animate: bool = True)` — sets zoom level.
  - `get_visible_bounds() -> Tuple[float, float, float, float]` — delegates to `_viewport.get_visible_bounds()`.

## Files NOT to modify

- `raven/common/gui/xdotwidget/parser.py` — import `DotScanner` and the token type constants for attribute stripping, but do not modify the module. Note: `DotScanner` and the token constants (`ID`, `STR_ID`, `HTML_ID`, `LSQUARE`, `RSQUARE`, `EQUAL`, `COMMA`, `SEMI`, `EOF`, `SKIP`) are not in `__all__` (they're internal to the parser module per PEP 8), but this level of coupling is acceptable here — the xdot viewer is tightly related to the xdotwidget package. Use explicit imports.
- Everything else in `raven/common/gui/xdotwidget/` besides `widget.py` — no changes needed.

## Testing

Manual testing:

1. Open a `.dot` file with default `[as-is]` → renders with `GRAPHVIZ_ENGINES[1]` (fallback).
2. Switch to `neato` → re-renders, focus preserved if centered on a node.
3. Switch to `fdp` → same.
4. Switch back to `[as-is]` on the `.dot` file → re-renders with `GRAPHVIZ_ENGINES[1]` (fallback).
5. Open a `.xdot` file with `[as-is]` → renders directly (fast path, no GraphViz needed).
6. Switch to `dot` on the `.xdot` file → re-renders via the `dot` engine (may differ from original layout if the `.xdot` was produced by a different engine).
7. Switch back to `[as-is]` on the `.xdot` file → shows the **original** pre-rendered layout (fast path again).
8. Open a `.xdot` file with filter set to `neato` → re-renders via `neato` (not fast path).
9. Modify the source file externally → auto-reload uses the active engine (or fast path for `.xdot` + `[as-is]`).
10. Open a file, zoom/pan to center on a specific node, switch engine → view stays centered on that node.
11. Pan to empty space between nodes, switch engine → zoom-to-fit.
12. Press Ctrl+E → combo is focused. Press Down arrow → engine changes, graph re-renders.
13. While combo focused, press Home → jumps to `[as-is]`. Press End → jumps to `twopi`.
14. While combo focused, press Esc → focus returns to graph, arrow keys pan again.

There are test graphs in `raven/xdot_viewer/testdata/`. Use those for quick iteration, but also test with a larger graph (e.g. a Pyan3 call graph output) to verify focus preservation behaves well.

## Scope boundary

This task is ONLY the filter switching feature. Do not:

- Refactor the existing toolbar code
- Change the xdotwidget internals beyond the specified public accessors in `widget.py`
- Touch any other Raven component
