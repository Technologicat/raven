# Brief: image shapes in the xdot widget, and mip selection for both consumers

Written 2026-09-07, on the back of brief 16's items 4 and 5 (`ImageShape` and the chat graph's attachment
thumbnails). Not Researchers' Night work — this is the follow-on the chat graph paid for and did not use.

**Two deliverables, and the second is a defect in shipped code**:

1. **`XDotWidget` honours xdot's `I` operation**, so a GraphViz graph with `image=` on a node draws it, and
   `raven-xdot-viewer` gets that for free.
2. **A drawn image stops going blocky when the reader zooms in.** The chat graph's thumbnails have this
   problem today: they are prepared once at 128 px and DPG samples nearest-neighbour, so past that size on
   screen they are visibly stepped. Juha, 2026-09-07: *"thumbnails shouldn't get blurry if the user zooms
   in."*

They are one brief because they are one mechanism. The parser cannot honour `I` without solving the second
problem — an `image=` file has no size chosen for the display, so there is no single size to prepare it at.

## What already exists

Most of the machinery landed for the chat graph and is directly reusable.

- **`ImageShape`** (`raven/common/gui/xdotwidget/graph.py`) — a texture and a rectangle in graph
  coordinates, with `max_screen_size` capping how large it is drawn. A `None` texture draws nothing and
  keeps its rectangle, which is the placeholder case.
- **The renderer draws it** (`_render_image_shape`), with the cap applied about the rectangle's centre.
- **The resampler**: `raven.common.image.lanczos.mipchain` builds the levels, and
  `lanczos.mip_scale_for_zoom` picks one for a given zoom. Both are in the shared layer as of 2026-09-07,
  the second having moved out of `raven.cherrypick.preload` for exactly this reason.
- **The non-blocking provider pattern**, from `DPGChatController.get_graph_thumbnail_texture`: a miss
  queues a background task and answers `None`, because a rebuild runs on the render thread where
  `split_frame` deadlocks. The consumer then has to notice the answer changing — a texture landing alters
  nothing else the view polls.
- **The parser's `I` branch** already reads the operation's arguments in a comment, and
  `read_point` / `read_number` / `read_text` all exist.

## The three pieces

### 1. An image store on the widget

The bulk of the work, and the part with no precedent in this package: **no shape owns a texture today.**
`ImageShape` holds a tag somebody else owns, and for the chat graph that somebody is the chat controller,
which outlives every graph it draws. A parsed graph has no such owner.

So `XDotWidget` grows one: `(path, level) -> texture`, cleared on `set_graph`, on `set_xdotcode` and on
`destroy`. Preparation goes on a background task with the shape drawing its placeholder meanwhile, exactly
as the chat graph does.

`raven-xdot-viewer` has no task manager today, so it needs one (`bgtask.TaskManager`, small).

### 2. Parser support for `I`

Small once the store exists. Read `I x y w h "path"`, `transform()` the coordinates, and emit an
`ImageShape` the store resolves.

**Two decisions this needs and the brief does not make:**

- **How a path is resolved.** GraphViz writes whatever the `image=` attribute said, which may be relative
  — to the `.dot` file's directory, or to GraphViz's own `imagepath`. Resolving relative to the file the
  xdot came from is the obvious first cut; `set_xdotcode` does not currently know that path, so it would
  have to be told.
- **What a missing or undecodable file draws.** The honest answer is probably the rectangle with a broken
  outline, which is a shape this package already knows how to draw — an unfilled `PolygonShape` with a
  dash pattern — rather than nothing at all, which reads as a graph that has no image in it.

*Not a decision:* the viewer will read arbitrary paths named by a `.dot` file it was given. That is what a
local viewer does with a local file, and it is worth having said out loud rather than discovered.

### 3. Mip selection, which is the half that applies to both

**The rule, and it is not the obvious one**: what decides the level is the image's size **in screen
pixels**, which is its size in graph units times the current zoom. A graph at 1:1 is 1:1 in *graph
coordinates*, and a card 55 graph units wide is 55 pixels there — so "the graph is at 1:1" says nothing on
its own about which level to draw.

**Juha's shape for it** (2026-09-07): prepare each image once at the largest size it could reasonably be
wanted at — for the chat graph, the same size the chat log's inline images use — build a mip chain from
that, and draw the level that suits the current screen size. The graph at 1:1 then shows a *lower* mip of a
texture prepared much larger, and zooming in walks up the chain it already has instead of going back to the
file.

That also collapses a cache split: the chat log and the chat graph would want the same prepared size, where
today they hold the same sidecar twice under a `(filename, size)` key.

**Where the selection lives.** `ImageShape` holds one texture, so something has to change:

- **(a) The shape holds the chain** — a sequence of `(scale, texture)` — and the renderer picks, since it
  is the one place that knows the zoom. Keeps the shape pure data and needs no watcher. **Recommended.**
- **(b) The shape holds a callable** `texture_for(screen_size)`. More flexible, and it puts behaviour into
  what is otherwise a plain data model, which the rest of this module is not.
- **(c) The consumer swaps the texture when the zoom changes materially.** Cheapest to write and wrong for
  the chat graph, which does not rebuild on zoom — it would need a zoom watcher that exists for nothing
  else.

`max_screen_size` stays either way: it is the cap that stops the *top* level being upsampled, which is a
different question from which level to draw.

## Sizing

**M — about a day, three commits**, in the order above. The store is roughly half of it, the mip selection
about a third, the parser the remainder.

**The mip work is separable and is the part with independent value**: done alone, it fixes the chat graph's
blockiness without any `I` support at all. If the day has to be split, that is where to cut it.

## What closes

- `raven/common/gui/xdotwidget/parser.py`'s `I` branch and its TODO.
- The blockiness noted above, which has no entry anywhere else — it was reported live on 2026-09-07 and is
  recorded here rather than in `TODO_DEFERRED.md`, this brief being where it will be acted on.
