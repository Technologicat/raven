# CC Briefing: Tests for `_strip_xdot_layout_attrs`

## Goal

Write unit tests for `_strip_xdot_layout_attrs` in `raven/xdot_viewer/app.py`.

This is the only new logic from the layout engine selector feature (commit `e95f13e`) that warrants dedicated tests. It's a nontrivial state machine that strips GraphViz layout attributes from dot/xdot source so a different engine can re-layout from clean DOT.

## The function under test

**Location:** `raven/xdot_viewer/app.py`, lines 93–193

**Signature:** `_strip_xdot_layout_attrs(xdotcode: str) -> str`

**What it does:** Uses `DotScanner` (from `raven/common/gui/xdotwidget/parser.py`) to tokenize the input, then runs a state machine that tracks `[...]` attribute lists and deletes `name = value` pairs where `name` is in `_XDOT_LAYOUT_ATTRS`. Also consumes trailing separators (`,`, `;`) and surrounding whitespace.

**Layout attrs stripped:** `bb`, `pos`, `lp`, `rects`, `_background`, `_draw_`, `_ldraw_`, `_hdraw_`, `_tdraw_`, `_hldraw_`, `_tldraw_`

**Dependencies:** `DotScanner` and token constants (`ID`, `STR_ID`, `HTML_ID`, `LSQUARE`, `RSQUARE`, `EQUAL`, `COMMA`, `SEMI`, `EOF`, `SKIP`) — all from `parser.py`, no DPG dependency.

## Prerequisite: extract to `dot_utils.py`

`app.py` imports DPG at module level. Testing `_strip_xdot_layout_attrs` through `app.py` would needlessly pull in the GUI stack.

**Before writing tests,** extract `_strip_xdot_layout_attrs`, `_XDOT_LAYOUT_ATTRS`, and `_dot_scanner` into `raven/xdot_viewer/dot_utils.py`. This module should only depend on `parser.py` (for `DotScanner` and token constants). Then update `app.py` to import from `dot_utils`.

## Test cases to cover

### Basic functionality
- **Strips known layout attrs:** Input with `pos="50,100"`, `bb="0,0,200,150"`, `_draw_="..."` → those attrs removed, other attrs preserved.
- **Preserves non-layout attrs:** `label`, `shape`, `style`, `fillcolor`, `fontname`, `color`, `penwidth` etc. must survive untouched.
- **No-op on clean DOT:** Input without any layout attrs → returned unchanged (identity).
- **Returns original buffer positions:** Output preserves original quoting, formatting, comments — the function works by deletion, not reconstruction.

### Edge cases in the state machine
- **Quoted attr names:** `"pos"="50,100"` — the name is `STR_ID`, function strips quotes for comparison. Should still match.
- **HTML attr values:** Attr with HTML value like `<...>` — the scanner handles this, but verify the value token is consumed correctly.
- **Mixed attrs in one bracket:** `[label="hello", pos="50,100", shape=box]` — only `pos` removed, comma handling correct, result is valid syntax.
- **All attrs stripped:** `[pos="50,100", bb="0,0,200,150"]` → empty or near-empty brackets `[]`. Verify no trailing comma/semicolon inside.
- **Trailing separator variants:** Test with comma, semicolon, and whitespace-only separators between attrs.
- **Multiple attribute lists:** Graph-level `[bb=...]`, node-level `[pos=..., label=...]`, edge-level `[pos=..., _draw_=...]` — all processed independently.
- **Nested brackets?** Not valid DOT, but verify the `bracket_depth` tracking doesn't break on unusual input.

### Attrs that should NOT be stripped
- **Layout attr names outside brackets:** A node named `pos` or `bb` (as a graph element name, not inside `[...]`) must not be touched.
- **Layout attr names as values:** `[label="pos"]` — `pos` appears as a value, not a name. Must not be stripped.
- **Partial name matches:** An attr named `position` or `_draw_extra` should not be stripped (the function uses exact match on the frozenset).

### Real-world inputs
- **testdata/test_graph.dot:** Run through `dot -Txdot`, then strip, then verify the result is valid DOT that `dot -Txdot` can re-process. Guard with `@pytest.mark.skipif(shutil.which("dot") is None, reason="GraphViz not installed")`.
- **testdata/test_callgraph.dot:** Same treatment — this one has subgraph clusters, `rankdir=LR`, and mixed node styles. Same skipif guard.

## Test location

`raven/xdot_viewer/tests/test_dot_utils.py` (create `tests/` dir + `__init__.py`). Import from `raven.xdot_viewer.dot_utils`.

## Style notes

Follow the existing test style in `raven/common/gui/xdotwidget/tests/` — pytest, descriptive fixture strings at module top, behavioral assertions (check what was preserved/removed, not just "didn't crash").
