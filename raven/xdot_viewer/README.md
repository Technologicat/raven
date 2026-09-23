<p align="center">
<img src="../../img/logo.png" alt="Logo of Raven" height="200"/> <br/>
</p>

-----

<p align="center">
<img src="../../img/xdot-viewer.png" alt="Screenshot of Raven-xdot-viewer" width="800"/> <br/>
<i>Raven-xdot-viewer displays a GraphViz graph, and lets you walk it.</i>
</p>

**Table of Contents**

- [Introduction](#introduction)
- [Opening a graph](#opening-a-graph)
    - [Choosing a layout engine](#choosing-a-layout-engine)
    - [Auto-reload](#auto-reload)
- [Walking the graph](#walking-the-graph)
- [Searching](#searching)
- [Dark mode](#dark-mode)
- [Keyboard reference](#keyboard-reference)
- [Command-line options](#command-line-options)
- [Configuration](#configuration)
- [Limitations](#limitations)


# Introduction

*Raven-xdot-viewer* displays a [GraphViz](https://graphviz.org/) graph and lets you move around inside it:
zoom, pan, search for a node by name, and follow an edge to the node at its other end.

It exists because a large graph rendered to a PNG is unreadable — the interesting question about a call
graph or a dependency graph is usually *what is connected to this one thing*, and that is a question you
answer by moving around rather than by looking at the whole picture at once.

```bash
raven-xdot-viewer mygraph.dot
```

It reads `.dot` and `.gv` sources, and `.xdot` — GraphViz's own output format, which carries the computed
layout. A source file is laid out for you; see below.

The viewer is built on the reusable `XDotWidget` in [`raven.common.gui.xdotwidget`](../common/gui/xdotwidget/),
so the same canvas can be embedded in another Raven app.


# Opening a graph

Pass a file on the command line, use **Ctrl+O**, or drag a `.dot`, `.xdot` or `.gv` file onto the window
from your file manager.

## Choosing a layout engine

A `.dot` source says what is connected to what, and *not* where anything sits. The selector in the toolbar
picks which GraphViz engine works that out — `dot`, `neato`, `fdp`, `sfdp`, `circo` or `twopi` — and the
graph is re-laid out when you change it. Each engine has its own idea of a good picture, and which one
suits a given graph is a matter of trying them:

- **`dot`** — layered, for anything with a direction to it (call graphs, dependency graphs, hierarchies).
- **`neato`**, **`fdp`**, **`sfdp`** — spring models, for graphs with no natural direction. `sfdp` is the
  one that scales to large graphs.
- **`circo`**, **`twopi`** — circular and radial, which suit graphs with a centre.

**`[as-is]`** is the first entry and means *do not lay this out*: use the positions the file already
carries. That is what you want for an `.xdot`, which has been through an engine already.

**Ctrl+E** puts the keyboard on the selector; then `Up` / `Down` step through the engines, `Home` / `End`
go to the ends, and `Esc` hands the keyboard back to the graph. The list carries a blue border while it has
the arrow keys, since the toolkit draws nothing of its own to say so.

## Auto-reload

The open file is polled for changes and reloaded when it changes, so a graph you are regenerating from a
script updates in place while you watch. The poll interval is `FILE_RELOAD_POLL_INTERVAL` in
[`raven.xdot_viewer.config`](config.py).


# Walking the graph

- **Click** a node or an edge to focus the view on it. Clicking an edge cycles: zoom-to-fit → the source
  node → the destination node → zoom-to-fit, so one edge can be followed in either direction by clicking it
  repeatedly.
- **Right-click** a node to open its URL in your browser, if the graph gave it one.
- **Shift+hover** a node highlights its *outgoing* connections; **Ctrl+hover** highlights its *incoming*
  ones. This is the quickest way to answer "what does this call?" and "what calls this?" without moving.
- **Hover near an edge endpoint** and a follow indicator appears; **click** it to jump to the node at the
  other end of that edge. This is what makes a graph too large to see navigable at all — you can travel
  along an edge whose far end is off screen.

Zoom with `Numpad +` / `Numpad -` or the mouse wheel (which zooms at the cursor), `1` for actual size and
**F** to fit the whole graph. Pan with the arrow keys or by dragging.

Highlights fade rather than snapping off, over `HIGHLIGHT_FADE_DURATION` seconds.


# Searching

**Ctrl+F** puts the caret in the search field. Results update as you type, and **F3** / **Shift+F3** jump to
the next and previous match.

The rules are the constellation's, the same ones *Raven-visualizer* and *Raven-librarian* use:

- Each space-separated term is a **fragment**, and **all** of them must match. Order does not matter.
- A **lowercase** fragment matches case-insensitively: `cat photo` matches *photocatalytic*.
- A fragment with **at least one uppercase letter** matches case-sensitively: `TiO` matches titanium oxide
  and not *bastion*.

`Enter` in the field accepts the search and jumps to the first match; `Esc` cancels the edit and hands the
keyboard back to the graph.


# Dark mode

**F12** toggles it. The graph's own colours are remapped rather than inverted — lightness is remapped and
clamped, so a graph authored for a white background stays legible on a dark one without its hues turning
into their opposites. `DARK_MODE` in [`raven.xdot_viewer.config`](config.py) sets which mode it starts in.


# Keyboard reference

The same table is on the app's own **F1** card, which is the copy to trust if these ever disagree.

| Key | Action | |
|---|---|---|
| `Ctrl+O` | Open a file | |
| `Ctrl+F` | Focus the search field | |
| `Enter` | Accept and jump to the first match | when focused |
| `Esc` | Cancel the edit and unfocus | when focused |
| `F3` | Jump to the next match | |
| `Shift+F3` | Jump to the previous match | |
| `Ctrl+E` | Focus the layout engine selector | |
| `Up` / `Down` | Previous / next engine | while focused |
| `Home` / `End` | First / last engine | while focused |
| `Esc` | Focus the graph view | while engine selector focused |
| `Numpad +` | Zoom in | |
| `Numpad -` | Zoom out | |
| `Mouse wheel` | ...the same, at the cursor | |
| `1` / `Numpad 1` | Zoom to actual size (1:1) | |
| `F` | Zoom to fit | |
| `Arrow keys` | Pan the view | |
| `Mouse drag` | ...the same, with the mouse | |
| `F1` | Open this help card | |
| `F11` | Toggle fullscreen | |
| `F12` | Toggle dark mode | |


# Command-line options

```bash
raven-xdot-viewer [file] [--width N] [--height N]
```

- **`file`** is the graph to open — `.dot`, `.gv` or `.xdot`. Omit it and the viewer starts empty.
- **`--width`** and **`--height`** set the window size, defaulting to
  [`raven.xdot_viewer.config`](config.py).

`--log`, `--log-level`, `--repl`, `--version` and `--help` work as they do across the constellation; see
[*Options every app takes*](../../README.md#options-every-app-takes) in the main README.


# Configuration

[`raven.xdot_viewer.config`](config.py), a Python module that exists to be edited — or, better, overridden
from `~/.config/raven/overrides.json`, which keeps your settings out of the repository. See
[*Settings that belong to your machine, not to Raven*](../../README.md#settings-that-belong-to-your-machine-not-to-raven).

Worth knowing about:

- `GRAPHVIZ_ENGINES` — which engines the selector offers, `[as-is]` first.
- `DARK_MODE`, `DARK_MODE_BACKGROUND`, `LIGHT_MODE_BACKGROUND` — the mode at startup, and the two backdrops.
- `FILE_RELOAD_POLL_INTERVAL` — how often the open file is checked for changes.
- `PAN_AMOUNT`, `ZOOM_IN_FACTOR`, `ZOOM_OUT_FACTOR`, `MOUSE_WHEEL_ZOOM_FACTOR` — how far one keypress or
  one wheel notch moves you.
- `HIGHLIGHT_FADE_DURATION` — how long a connection highlight takes to fade.


# Limitations

**GraphViz's `--concentrate` leaves small gaps at high zoom.** Where edges are merged and split again, the
endpoints it writes into the xdot are a hundredth or so of a graph unit apart, which is invisible at normal
zoom and shows as a small gap when you are close in. That is a precision limit in the data the viewer is
given, not something it can draw its way out of.
