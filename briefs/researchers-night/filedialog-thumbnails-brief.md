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

Left open: which torch device the dialog decodes on. Cherrypick resolves one at app level; the dialog has no
such config, so it probably wants an optional constructor parameter with a lazy default.

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
