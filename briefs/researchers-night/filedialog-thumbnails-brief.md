# FileDialog: image thumbnail previews

**Status: built, 2026-08-14, and not yet live-tested.** What is left is judging it by looking — tile size,
how the lazy decode feels while scrolling, and whether a recent-directory cache is needed — plus the real
tileset, which is Juha's to generate.

- **The shared grid widget**, its decoder and its extension hooks landed 2026-08-13, with Cherrypick ported
  onto them as the proving ground. Smooth scrolling and the scroll-end flasher followed on 2026-08-14.
- **The listing refactor this brief calls for is done** (2026-08-14). `raven.common.filelisting` produces
  the ordered entries as data — `FileEntry` objects, `..` and the directories among them — and both views
  build from that same list. The sort no longer walks the widget tree, and the sort criterion is state that
  a rebuild reproduces. That was the enabling change. Live-tested in Visualizer and Librarian.
- **The view itself** landed the same day: `raven.common.gui.filegrid.FileGrid` joins a listing to a grid,
  `raven.common.gui.tileicons.TileIconCache` resamples the icons to tile size, and the dialog carries the
  toggle, the shared sort-button row and its own tick thread.

Two things to look at first when testing, because they are the ones no test can answer: whether the tiles
appear fast enough while scrolling, and whether the column *resize* gesture still works now that the header
no longer sorts (`resizable` and `sortable` are separate flags, but that is read from the API rather than
measured).

Moved out of `TODO_DEFERRED.md` on 2026-08-13.

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

**One row of sort buttons, above the listing, serving both views** — Name, Date, Type, Size, with the table
header's own semantics: click to sort ascending, click again for descending, a triangle on the active
button showing which way.

Not "sort buttons for the grid, header clicks for the table", which was the first shape and is wrong. Two
controls over one order means two things that can disagree, and they will: ImGui draws the header's sort
arrow from its *own* state, so a sort chosen in grid mode leaves the table header asserting an order the
data no longer has. Set `no_sort=True` on the columns — measured 2026-08-14 as settable on a live column,
see `dpg-notes.md` — and the second source of truth is gone by construction rather than by keeping two
things in step.

**The requirement this serves, stated by Juha 2026-08-14: switching views must not change anything.** The
sort order carries over, and the cursor stays on the same file. Both fall out of the refactor rather than
needing work here — the sort criterion becomes app state that a view switch does not touch, and the cursor
is already specified as re-anchored *by path* after every rebuild (`filedialog-keyboard-brief.md`), because
typing in the find field rebuilds the listing constantly. A view switch is one more rebuild.

**A fake header styled to look like the real one was considered and rejected**, having been the natural
answer: replace `header_row` with our own buttons, and the familiar gesture survives. It does not survive
the column widths. `resizable`, `reorderable` and `hideable` are all header-drag gestures, so removing the
header removes all three — and while reordering and hiding are dispensable here, **resizing is not**:
filename lengths vary enormously between users and directories, which is exactly when a fixed Name column
hurts. `no_sort` is a per-column *sorting* flag and leaves resizing alone, so keeping the real header and
moving only the sort out of it is what preserves the gesture that matters.

It costs the familiar click-the-header gesture, and buys three things:

- **The disagreement cannot happen.** A guarantee, where synchronizing would be a hope. Whether reconfiguring
  `default_sort` actually moves the drawn arrow is *unknown* and needs a rendered frame to find out, so the
  synchronizing design rests on an unverified assumption; this one does not.
- **Sorting becomes keyboard-operable**, which `filedialog-keyboard-brief.md` needs and would otherwise have
  to solve on its own. ImGui's header sorting has no keyboard path at all — the same hole as its combos,
  which that brief already works around with a focus-then-arrows idiom. Buttons are focusable; a header is
  not.
- **One place to learn.** The control does not move or change when the view does.

**Switch `reorderable` and `hideable` off while here** (Juha, 2026-08-14). Both are on today and neither
earns its place in a file dialog with four fixed columns — and both are gestures on a header that no longer
sorts, so leaving them is a header that responds to three drags and ignores the click everyone tries first.
`resizable` stays on.

## Where each piece lives

Settled 2026-08-14, before writing any of it, because this is cheap to decide now and expensive once the
code exists. The test applied throughout: has sibling code already committed to this problem class?

**`raven/common/filelisting.py` — new, and DPG-free.** The listing logic: enumerate a directory, apply the
hidden-file policy, the name filter and the type filter, sort by name / date / type / size in either
direction, and yield the entries — `..` and the directories among them, as *data*. Nothing here imports
`dearpygui`.

Two reasons, and the second is the one that decides it:

- The semantics are generic file-browser semantics, not dialog semantics. Nothing in the above is about
  being a dialog.
- **`fdialog.py` imports `dearpygui` at module level**, so logic living there cannot be tested without a
  GUI context — which is why the current listing has no tests and why the sort callback has to walk the
  widget tree to recover data it created itself. This is the `cleanup.py` / `cleanup_dialog.py` split the
  project already names as its worked example: separating the operation from its dialog is what makes the
  operation testable at all.

A third reason applies to the location specifically: `raven/vendor/file_dialog/` is *adopted* code, so
editing it in place is fine, but new first-party logic is better outside the vendor tree than inside it.

**Tile-sized icon resampling and caching — with `ThumbnailGrid`, in `raven.common.gui`.** "Resample an
image asset to the current tile size, cache per size, redo it when the tile size changes" is a property of
drawing into the grid, not of being a file dialog, and the grid already owns the tile-size concept. Any
later grid consumer that shows non-thumbnail tiles needs exactly this.

**The sort button row — inline in the dialog, but written so it can leave.** It takes a list of criteria
and a callback rather than reaching into dialog state, so promoting it later is a move rather than a
rewrite. Inline for now because nothing else sorts today — but **Cherrypick is a likely second consumer**
(Juha, 2026-08-14): its grid is unsorted only because his ComfyUI workflows embed the datetime in the
filename and conference photos arrive in timestamp order, so the files happen to arrive pre-sorted the way
he wants them. That is a property of his current inputs, not a decision, and it will stop holding. Written
movable for that reason, not on speculation.

**The view toggle — inline.** Dialog-specific by nature.

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

**The grid shows every entry, not only the images** — decided 2026-08-14. A directory gets the folder tile,
an image gets its thumbnail, and anything else gets an icon for its file type. The alternative, showing only
what has a thumbnail, silently hides files that are *there*, and a picker that lies about the contents of a
directory is worse than one that is plain. It also matters for the case Librarian actually presents: "images
and documents" is one filter, and a grid with the documents missing from it would be the common view rather
than an edge case.

So this wants **a full tileset, one icon per file type the dialog distinguishes** — Juha to generate,
512×512 as above. The dialog already has the type→icon mapping to follow: `_makefile` picks from a table
keyed by extension groups (`.dll`/`.so` → gears, `.png`/`.jpg` → picture, `.iso` → disc, archives, Python,
and so on), so the tileset's contents are that list rather than a fresh decision. Prototype with those same
16×16 and 94×94 assets Lanczos'd up — ugly at large tile sizes, which is the point of calling it a
prototype.

That in turn loosens the auto-on rule above: a grid that shows documents legibly is useful for more than
image-typed filters, so **which filters turn it on automatically is worth re-deciding once the tileset
exists** and it can be judged by looking rather than argued.

## The budget, and a correction to it

Measured live on a real 1625-entry directory (`investigations/filedialog-performance/`): a full open costs
0.32 s, of which 0.26 s is building the rows — about 60 µs each. Decoding a thumbnail per row at build time
would add *milliseconds* each, three orders of magnitude over that.

**That arithmetic is right and the conclusion drawn from it was overstated** (Juha, 2026-08-14). Image
directories typically hold *hundreds* of images, not thousands, and Cherrypick decodes a whole folder at
once with a perfectly good user experience — which is the empirical answer to the question the arithmetic
was being used to settle. So lazy decode is worth having and is not the only version that could exist.

It is built, so the thing to do is measure it rather than re-argue it. What would send it back:

- It is **more machinery**, and the settle-then-restart scheduler has three separate refusals it has to get
  right (see `FileGrid`). Whole-folder decoding has none of that.
- A grid that fills only what you look at **shows its seams when you scroll fast**, where a folder decoded
  up front simply is done.

What keeps it: it is the only version that degrades gracefully into the thousands, and the visible-set
machinery is what a recent-directory cache would sit on top of anyway.

The dialog logs its phases at DEBUG (`list / delete / build / sort`, plus `show_file_dialog`'s frame wait),
so a before-and-after is one `--log-level DEBUG` run away.

## Feel

**Follow Cherrypick's pattern deliberately** (Juha, 2026-08-13): thumbnails should feel the same across the
two apps. That means the VHS-noise placeholder standing in for "not loaded yet"
(`raven.common.video.postprocessor.vhs_noise_pool`, tuned by `PLACEHOLDER_*` in `raven/cherrypick/config.py`),
and the texture-upload glitch when the real thumbnail arrives from the background job — which started as an
artifact and is now intentional.

## A refinement worth having: folder tiles that preview what is inside

Raised 2026-08-14. A file manager shows a folder holding pictures as a folder *with pictures in it*, and
that is a better tile than a generic icon in exactly the case this feature is for — hunting through a tree
of image directories, where the folder names are as uninformative as the filenames.

It also settles a question the build had to answer in the meantime: **the grid view is currently switched
off entirely for a directory picker** (`dirs_only`, which is how `raven-cherrypick` opens the dialog),
because every tile would be the same folder icon and the grid would cost space and legibility for nothing.
A previewing folder tile overturns that, which is why it is a predicate in the code (`_grid_is_available`)
rather than an assumption baked into the layout.

Cost: read the first few image entries of each listed directory and compose them into one tile. Cheap per
folder, but it is a directory read per tile, so it wants the same visible-set laziness the images already
get, and probably a cap on how deep it looks.

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
  - **The two apps are used differently, and this is where it shows** (Juha, 2026-08-14). In Cherrypick the
    user opens one folder and stays in it for a long time; in the dialog they go back and forth between
    folders looking for the right one. So the cost Cherrypick never pays — decoding the same directory
    again — is the dialog's ordinary case, and it is the one thing here that the shared grid does not
    already solve.
  - Whether it is *needed* depends on how fast the built version turns out to be, so it waits on the live
    test rather than on an argument.

## Testing

`raven.common.tests.write_demo_image_folder` generates a folder whose images carry their own index and hue,
with orientation alternating so letterboxing is visible. For realistic filenames — the case this brief exists
for — use a real generated-image folder; there is one on the work machine, noted in machine-local memory
rather than here.
