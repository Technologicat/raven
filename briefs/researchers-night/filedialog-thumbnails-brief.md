# FileDialog: image thumbnail previews

**Status: foundations built and shipped, integration designed and unbuilt.** The shared grid widget, its
decoder and the extension hooks landed on 2026-08-13, with Cherrypick ported onto them as the proving
ground. What remains is the file dialog's own view. Moved out of `TODO_DEFERRED.md` on 2026-08-13.

One of the two final FileDialog pieces for Researchers' Night, with `filedialog-keyboard-brief.md`. They
touch the same widget and should be built with each other in view: the grid needs the cursor and selection
machinery that brief defines, and its type-filter hotkeys are what switch this view on.

## Why

The dialog lists files by name only. For picking *image* files that is close to useless: photographs and
generated images have non-descriptive filenames — hashes, timestamps, auto-names — so the image data is the
only reliable way to identify the right file. You pick by looking, or you guess.

The original framing said the picker was the *only* way to bring an image in, since DPG apps had no OS
drag-and-drop. That stopped being true on 2026-08-10 when `raven.common.gui.filedrop` shipped into all six
GUI apps. The brief stands on its own merits — a drop only helps when the file manager is already open on the
right folder — but it is one justification lighter than it was.

## What already exists

| Piece | Where |
|---|---|
| The grid widget — layout, tiles, placeholder pool, textures, selection, hit testing, navigation | `raven.common.gui.thumbnailgrid.ThumbnailGrid` |
| Decode + Lanczos resize on two background threads | `raven.common.image.thumbnails.ThumbnailPipeline` |
| Navigation arithmetic | `raven.common.gui.gridnav` |
| A worked example of extending the grid | `raven.cherrypick.grid.TriageGrid` |
| A folder of demo images for hand-testing | `raven.common.tests.write_demo_image_folder` |

Extension is by subclass, through three hooks: `draw_underlay(idx, drawlist)` for decoration that the tile's
own furniture must stay readable through, `draw_overlay(idx, drawlist)` for decoration on top, and
`border_color_for(idx)`. Filtering is the owner's job — the grid takes a list of visible indices and knows
nothing about what admitted them, which is what lets a *file* dialog drive it.

## The design for the dialog's view

Settled 2026-08-13 (Juha and Claude).

- **A grid view mode, toggled**, replacing the table in the same slot rather than growing the table's rows.
  Taller uniform rows were the alternative and were rejected — they cost every listing vertical space, and a
  clipped table needs *uniform* row height, so "tall rows only where there is an image" is not available. A
  preview pane was also rejected: it shows one image, and the point is picking by looking across many.
- **Auto-on whenever the selected type filter is image-typed, with a manual toggle that overrides in either
  direction.** Grouped filters landed 2026-08-13, so "the filter is image-typed" is a real predicate.
- **The grid must list directories too**, as folder tiles before the image tiles, mirroring the table's
  order. Otherwise switching to grid mode removes the only way to navigate — which is the obvious version of
  this feature, and wrong.
- **The dialog owns the ticker.** The grid needs `update()` every frame and the pipeline needs polling, but
  `FileDialog` is a widget inside apps that own their render loops. Rather than requiring every host app to
  call something, the dialog runs its own tick while the grid is visible. DPG permits widget work from any
  thread, and `visible_on_screen()` reads last-frame state, which is safe to read from one.
- **Lazy decode is the whole point.** Ask `grid.visible_on_screen()` for the tiles actually on screen and
  decode only those, restarting the pipeline when that set changes materially rather than every frame.

**Which device it decodes on: default to the literal `"gpu"`, with an optional override.** Settled
2026-08-13. `raven.common.deviceinfo` already resolves that string — it autodetects the single available GPU
backend, falls back to CPU with an info log when there is none, and raises only if two distinct GPU vendors
are active at once, which is rare enough that forcing an explicit pick is right.

That matters because of the failure it avoids: **an app that innocently wants a file-open dialog must not
have to know about torch devices.** A required per-app setting is a landmine — the app that forgets it is
the app that gets a crash or a silent CPU fallback nobody chose. Defaulting to `"gpu"` means configuring
nothing works everywhere, CPU-only machines included. The optional parameter is for apps that *do* care:
pinning thumbnails to the same device as their other work, or deliberately keeping them off a GPU already
busy with inference.

For reference, Cherrypick pins `cuda:0` in `config.gpu_config["thumbnails"]` with a `--device` override, and
still passes it through `deviceinfo.validate`, which checks availability and falls back to CPU — so its
hardcoded string is not the portability hazard it looks like.

## Sorting, and the refactor it forces

Settled 2026-08-14, on finding that the grid cannot inherit the table's sorting.

**The table's sort exists only as widget order.** `table_sort_callback` reads its keys back *out of DPG* —
walking `get_item_children` into each row, unpacking the group, pulling `user_data` off the selectable —
builds a list of row IDs, and calls `dpg.reorder_items`. There is no sorted list of entries anywhere. A
grid has no table, no rows and nothing to reorder, so none of it transfers, and a grid without sorting
would be a serious step down from the listing it replaces.

So:

- **Extract "produce the ordered list of entries" from "make table rows out of them"**, and let both views
  consume the same list. This is the enabling change, and it is worth doing on its own merits: it also
  makes the listing testable, which it is not while the entries exist only as widgets.
- **The sort criterion becomes shared state rather than DPG state**, so switching views keeps the order.
  Watch for the table's own header arrows disagreeing with it once the data is authoritative — DPG draws
  those itself, and they are driven by the header clicks rather than by us.
- **`..` and the directories travel as data.** `..` is currently special-cased in three places: built
  inline in `reset_dir` rather than through `_makedir`, skipped by `rows[1:]` in the sort, and
  re-prepended as `new_order[0]`. The grid needs it as much as the table does, so the shared list has to
  carry it rather than each view inventing it.

**The grid gets a row of sort buttons above it**, one per criterion the table offers — Name, Date, Type,
Size — with the table's semantics: click to sort ascending, click the same button again for descending, and
a triangle on the active button showing which way. The point is that a user who has learned the table
header has learned this too.

## The folder tile needs a large icon, and it must be resampled rather than scaled

For prototyping, reuse the dialog's own `folder.png` (`self.img_folder`, 94×94 RGBA) — the one it already
uses for drag payloads. It is *not* the 16×16 `mini_folder.png` the table rows carry, which would be
unusable at tile size.

**DPG scales textures nearest-neighbor**, so handing it a 94×94 icon and asking for 256 gives visible
blocking. Any image drawn at tile size therefore has to be resampled by us, with
`raven.common.image.lanczos.resize` — **at load, and again whenever the tile size changes.** The result
caches per tile size; the sizes are a small fixed set. This is the same treatment the photo thumbnails
already get, so a folder tile and an image tile end up looking equally deliberate rather than the folder
looking like a mistake.

**Generate the final icon at 512×512**, the largest tile size. Then every tile size is a *downscale*, which
is what Lanczos is good at; the prototype's 94 px is a 5.4× enlargement at that size, and no resampler
invents the detail that is not there.

The same rule covers anything else that ends up drawn at tile size — a per-filetype icon for non-image
files, say, if the grid ever grows one.

## The budget, which is what forces the design

Measured live on a real 1625-entry directory (`investigations/filedialog-performance/`): a full open costs
0.32 s, of which 0.26 s is building the rows — about 60 µs each. Decoding a thumbnail per row at build time
would add *milliseconds* each, three orders of magnitude over that, so a naive version turns a third of a
second into minutes.

The laziness that avoids it — build the tiles first, fill textures from a background task, and only for tiles
actually on screen — is not an optimization to add later. It is the only version of this feature that can
exist.

The dialog logs its phases at DEBUG (`list / delete / build / sort`, plus `show_file_dialog`'s frame wait),
so a before-and-after is one `--log-level DEBUG` run away.

## Feel

**Follow Cherrypick's pattern deliberately** (Juha, 2026-08-13): thumbnails should feel the same across the
two apps. That means the VHS-noise placeholder standing in for "not loaded yet"
(`raven.common.video.postprocessor.vhs_noise_pool`, tuned by `PLACEHOLDER_*` in `raven/cherrypick/config.py`),
and the texture-upload glitch when the real thumbnail arrives from the background job — which started as an
artifact and is now intentional.

## An open question for after the feature works: is one size enough?

Raised by Juha, 2026-08-14, explicitly as something to prototype *after* the rest of this lands rather than
to design now.

A grid tile has two jobs in tension — small enough to navigate a directory at a glance, large enough to
actually see what is in the image. A **separate preview pane** showing the selected image larger is the
obvious answer, and may not be the right one: there may be a tile size that serves both well enough, in
which case a pane costs horizontal space for nothing. Cherrypick is the evidence *for* a pane (it has a
full image view beside its grid) and also the evidence against reading too much into that, since triage
there is about judging one image at a time, where a picker is about finding one among many.

So: settle it by trying tile sizes first, and treat the pane as the fallback if none of them work. Nothing
in the design above forecloses either — `set_tile_size` already exists, and a pane would be the dialog's
own widget beside the grid rather than anything the grid needs to know about.

## Two things the Cherrypick pattern does not solve

- **Cherrypick itself slows down in huge directories.** Its grid materializes a tile for every entry passing
  the filter, "visible" there meaning "not filtered out" rather than "on screen", and it holds a texture per
  thumbnail. Both apps now share that grid, so **one fix serves both** — which is the reason to fix it there
  rather than working around it here. The table's escape, `clipper=True`, is a table feature with no
  counterpart for a grid of drawlists, so real windowing is what it would take. Not on this brief's path:
  thumbnail mode is auto-on only for image filters, so the case that bites is a folder of many thousands of
  images.
  - Do **not** reach for one drawlist sized to the whole scroll extent. Measured 2026-08-13: it renders the
    X session unusable, recoverable only from a text terminal. See `dpg-notes.md`, "Never size a drawlist to
    a scroll extent".
- **A file dialog needs the *last few* folders, not just the current one.** Navigating up and back down is
  the normal way to use a picker, so re-decoding every thumbnail on the way back is the case that will
  actually be felt. Hold thumbnails for a couple of recent directories and evict beyond that.

## Testing

`raven.common.tests.write_demo_image_folder` generates a folder whose images carry their own index and hue,
with orientation alternating so letterboxing is visible. For realistic filenames — the case this brief exists
for — use a real generated-image folder; there is one on the work machine, noted in machine-local memory
rather than here.
