# XDot Widget Implementation Summary

## Overview

This is a DearPyGUI widget for viewing graphs in xdot format, ported from xdottir (GTK3/Cairo, ~4300 lines). It supports:

- Pan and zoom with smooth animations
- Find/search with fragment matching
- Hover highlights with fade-out animation
- Programmatic highlighting
- Text compaction for zoomed-out views
- Viewport culling for performance

**Source:** `xdottir/xdot.py`
**Target:** `raven/common/gui/xdotwidget/`

---

## Module Structure

```
raven/common/gui/xdotwidget/
    __init__.py           # Public API: XDotWidget, parse_xdot
    parser.py             # XDot format parser (extracted from xdottir)
    graph.py              # Data model: Pen, Shape, Node, Edge, Graph
    renderer.py           # DPG drawing primitives
    viewport.py           # Pan/zoom transforms, SmoothValue animation
    hitdetect.py          # Mouse hit detection
    highlight.py          # Highlight state with fade animations
    search.py             # Find/search functionality
    widget.py             # Main XDotWidget class

    tests/
        __init__.py
        test_parser.py    # Parser and graph model tests
        test_viewport.py  # Coordinate transform tests
        test_search.py    # Search functionality tests
```

---

## Module Descriptions

### `__init__.py`
Public API exposing:
- `XDotWidget` - Main widget class
- `parse_xdot` - Parser function for xdot code

### `graph.py` (~17KB)
Data model extracted from xdottir:

- **`Pen`** - Drawing state (color, fillcolor, linewidth, fontsize, fontname, dash)
- **Shape classes:**
  - `Shape` (base class with bounding box)
  - `TextShape` - Text with font info
  - `EllipseShape` - Filled or outlined ellipse
  - `PolygonShape` - Filled or outlined polygon
  - `LineShape` - Polyline
  - `BezierShape` - Cubic bezier curves
  - `CompoundShape` - Container for grouped shapes
- **`Element`** - Base for graph elements with shapes
- **`Node`** - Graph node with hit detection
- **`Edge`** - Graph edge with endpoint hit detection
- **`Graph`** - Container with nodes, edges, and subgraphs

### `parser.py` (~33KB)
XDot parser extracted from xdottir:

- **`XDotAttrParser`** - Parses xdot drawing commands
- **`DotScanner`** - Tokenizer for DOT grammar
- **`DotLexer`** - Lexical analysis
- **`DotParser`** - DOT grammar parser
- **`XDotParser`** - Produces Graph objects from xdot code
- **`parse_xdot()`** - Convenience function

Includes complete X11 color lookup table (~150 colors) and Brewer color schemes.

### `viewport.py` (~11KB)
Coordinate transforms and animation:

- **`SmoothValue`** - Animated float using 1st-order ODE solution (Newton's law of cooling), with FPS correction from `animation.py`
- **`Viewport`** - Pan/zoom state with smooth animation
  - `graph_to_screen()` / `screen_to_graph()` - Coordinate transforms
  - `zoom_to_fit()` - Fit graph in view with margin
  - `pan_by()` / `zoom_by()` - Interactive manipulation
  - `is_visible()` - Bounding box visibility check for culling

Y-axis flip is handled in transforms (GraphViz uses bottom-left origin).

### `renderer.py` (~11KB)
DearPyGUI drawing:

- **`render_graph()`** - Main render function
- Shape-specific renderers for each shape type
- Bezier curve segment handling
- Text rendering with zoom-dependent sizing
- Viewport culling (skips elements outside visible area)

Shape mapping:
| xdottir (Cairo) | DPG |
|-----------------|-----|
| `cr.arc()` | `dpg.draw_ellipse()` |
| `cr.move_to/line_to` | `dpg.draw_polyline()` |
| `cr.curve_to` (cubic) | `dpg.draw_bezier_cubic()` |
| Polygon fill | `dpg.draw_polygon()` |
| Pango text | `dpg.draw_text()` |

### `hitdetect.py` (~3KB)
Mouse detection:

- **`hit_test_screen()`** - Find element at screen coordinates
- Checks nodes first (drawn on top), then edges
- Converts screen to graph coordinates via viewport

### `highlight.py` (~7KB)
Highlight state with animations:

- **`HighlightState`**
  - `set_hover()` / `get_hover()` - Hover highlight (instant on, fade off)
  - `set_highlighted()` / `get_highlighted()` - Programmatic highlights
  - `get_pen_for()` - Returns pen with highlight color mixed in
  - `update()` - Advance fade animations each frame

Uses exponential decay for fade-out.

### `search.py` (~4KB)
Find system:

- **`SearchState`**
  - `search()` - Fragment search (space-separated fragments, all must match)
  - `next_match()` / `prev_match()` - Navigate with wraparound
  - `get_result_ids()` - Get matching node IDs
  - `clear()` - Clear search state

Search is case-insensitive and matches substrings.

### `widget.py` (~14KB)
Main widget orchestrating all components:

- **`XDotWidget`**
  - Constructor takes parent, dimensions, callbacks
  - Creates DPG group + drawlist structure
  - Registers with animator for smooth updates
  - Handles mouse events via global handlers

Public API:
```python
def set_xdotcode(xdotcode: str)
def set_highlighted_nodes(node_ids: Set[str])
def zoom_to_fit(animate=True)
def zoom_to_node(node_id: str, animate=True)
def search(query: str) -> List[str]
def next_match() -> Optional[str]
def prev_match() -> Optional[str]
def clear_search()
def get_search_count() -> int
```

Callbacks:
- `on_hover(node_id: Optional[str])` - Called when hover changes
- `on_click(node_id: str, button: int)` - Called on element click
- `text_compaction_callback(text: str, width: float) -> str` - Custom text compaction

---

## Standalone Application

```
raven/xdot_viewer/
    __init__.py
    app.py               # Main application (~9KB)
```

CLI entry point in `pyproject.toml`:
```toml
raven-xdot-viewer = "raven.xdot_viewer.app:main"
```

Features:
- Opens `.xdot` files directly
- Opens `.dot` / `.gv` files via GraphViz filter
- Toolbar with zoom controls and search
- Keyboard shortcuts:
  - `Ctrl+F` - Focus search
  - `N` / `Shift+N` - Next/previous match
  - `+` / `-` - Zoom in/out
  - `0` - Zoom to fit
- Auto-reload on file change (2-second poll)
- Status bar showing hover info and search results

---

## Key Design Decisions

1. **Static color lookup** - Replaced `Gdk.color_parse()` with X11 color table (~150 colors) plus Brewer schemes

2. **FPS-corrected animation** - `SmoothValue` uses the same formula as `SmoothScrolling` in `animation.py` for consistent animation speed regardless of frame rate

3. **Viewport culling** - Elements outside visible area are skipped during render (DPG's built-in culling only works with triangles)

4. **Thread safety** - Uses `threading.RLock` following the VU meter pattern

5. **Y-axis flip** - Handled in `Viewport.graph_to_screen()` since GraphViz uses bottom-left origin

6. **Fragment search** - Space-separated fragments that all must match (like Emacs HELM)

---

## Running Tests

```bash
cd /home/jje/Documents/JAMK/gH2ADDVA_ja_AI/raven
python -m pytest raven/common/gui/xdotwidget/tests/ -v
```

---

## Usage Example

```python
import dearpygui.dearpygui as dpg
from raven.common.gui.xdotwidget import XDotWidget, parse_xdot

dpg.create_context()
dpg.create_viewport(title="Graph Viewer", width=800, height=600)

with dpg.window(tag="main"):
    widget = XDotWidget(
        parent="main",
        width=780,
        height=560,
        on_hover=lambda node_id: print(f"Hover: {node_id}"),
        on_click=lambda node_id, btn: print(f"Click: {node_id}")
    )

# Load a graph
xdotcode = open("my_graph.xdot").read()
widget.set_xdotcode(xdotcode)
widget.zoom_to_fit()

# Search
results = widget.search("keyword")
widget.next_match()  # Jump to first result

# Programmatic highlight
widget.set_highlighted_nodes({"node1", "node2"})

dpg.set_primary_window("main", True)
dpg.setup_dearpygui()
dpg.show_viewport()

while dpg.is_dearpygui_running():
    widget.update()
    dpg.render_dearpygui_frame()

dpg.destroy_context()
```

---

## File Sizes

| File | Size |
|------|------|
| `graph.py` | ~17KB |
| `parser.py` | ~33KB |
| `viewport.py` | ~11KB |
| `renderer.py` | ~11KB |
| `widget.py` | ~14KB |
| `highlight.py` | ~7KB |
| `search.py` | ~4KB |
| `hitdetect.py` | ~3KB |
| `__init__.py` | ~1KB |
| **Total widget** | **~101KB** |
| `xdot_viewer/app.py` | ~9KB |

---

## Dependencies

- `dearpygui>=2.0.0` (already in project)
- Uses `raven.common.gui.animation` for animator integration
- Uses `raven.common.gui.utils` for mouse detection helpers

---

## Raw report from Claude Code at job completion

All tasks completed. Let me provide a summary of what was implemented:

### Implementation Summary

I've implemented the XDot Widget for DearPyGUI as specified in the plan. Here's what was created:

**Module Structure**

`raven/common/gui/xdotwidget/`:

  1. `__init__.py` - Public API exposing `XDotWidget` and `parse_xdot`
  2. `graph.py` - Data model (~300 lines)
    - Pen class for drawing state (color, fillcolor, linewidth, fontsize, fontname, dash)
    - Shape classes: `TextShape`, `EllipseShape`, `PolygonShape`, `LineShape`, `BezierShape`, `CompoundShape`
    - `Element`, `Node`, `Edge` classes with bounding boxes and hit detection
    - Graph container with filter_items_by_text() for search
    - Color mixing utilities
  3. `parser.py` - XDot format parser (~700 lines)
    - `DotScanner`, `DotLexer`, `DotParser` - DOT grammar implementation
    - `XDotAttrParser` - parses xdot drawing commands (c, C, S, F, T, E, e, L, B, b, P, p, I)
    - `XDotParser` - produces Graph objects from xdot code
    - X11/CSS color lookup table
    - ColorBrewer palette support
  4. `viewport.py` - Pan/zoom transforms (~220 lines)
    - `SmoothValue` - animated float using 1st-order ODE solution (FPS-corrected)
    - `Viewport` - coordinate transforms with smooth animation
    - graph_to_screen() / screen_to_graph() transforms
    - zoom_to_fit(), pan_to_point(), zoom_by(), pan_by()
    - Viewport culling with is_visible()
  5. `renderer.py` - DPG drawing (~250 lines)
    - `render_graph()` - renders complete graph to drawlist
    - Shape rendering: ellipse, polygon, line, bezier, text
    - Highlight color blending
    - Viewport culling for performance
  6. `hitdetect.py` - Mouse detection (~60 lines)
    - `hit_test()` / `hit_test_screen()` - find element under cursor
    - Nodes use AABB collision, edges use endpoint radius
  7. `highlight.py` - Highlight state (~180 lines)
    - HighlightState - manages hover, fade animations, programmatic highlights
    - Smooth fade-out using non-analytic smooth transition
    - Multiple simultaneous fades supported
  8. `search.py` - Search functionality (~100 lines)
    - `SearchState` - fragment search with navigation
    - Case-insensitive multi-fragment AND search
    - `next_match()` / `prev_match()` with wraparound
  9. `widget.py` - Main widget (~350 lines)
    - `XDotWidget` class orchestrating all components
    - DPG drawlist-based rendering
    - Mouse handlers for pan/zoom/hover/click
    - Public API: `set_xdotcode()`, `zoom_to_fit()`, `set_highlighted_nodes()`, `search()`, etc.

**Unit Tests**

`raven/common/gui/xdotwidget/tests/`:

  - `test_parser.py` - Tests for parser and graph model
  - `test_viewport.py` - Tests for coordinate transforms and animation
  - `test_search.py` - Tests for search functionality

**Standalone Application**

`raven/xdot_viewer/`:

  - `app.py` - Full-featured viewer application
    - Opens `.xdot` and `.dot` files (dot filtered through GraphViz)
    - Toolbar with zoom/search controls
    - Keyboard shortcuts (Ctrl+F, N/Shift+N, +/-, 0)
    - File watching for auto-reload
    - Status bar with hover/click feedback

**CLI Entry Point**

Added to `pyproject.toml`:

```
  raven-xdot-viewer = "raven.xdot_viewer.app:main"
```

### Key Design Decisions

  1. No GTK/Cairo dependencies - All rendering uses DPG primitives
  2. Color lookup table - Replaced Gdk.color_parse with static X11 color table
  3. FPS-corrected animation - SmoothValue uses the same math as Raven's SmoothScrolling
  4. Viewport culling - Elements outside visible area are not rendered
  5. Thread-safe rendering - Uses threading.RLock like VU meter pattern
