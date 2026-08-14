"""A thumbnail grid over a directory listing: image previews for the images, icons for everything else.

`raven.common.filelisting` says what is in a directory; `ThumbnailGrid` draws a lattice of tiles. This is
the join between them, and the part neither can hold: which entries get a decoded preview, which get an
icon, and — the whole reason the feature is affordable — decoding only the tiles that are actually on
screen.

**Lazy decode is not an optimization here, it is the only version that can exist.** Decoding one image
costs milliseconds where building one tile costs tens of microseconds, so a directory of a couple of
thousand images would take minutes to open if every tile were filled at build time. What this does instead:
ask the grid which tiles the last frame rendered, wait for that set to stop changing (the user has stopped
scrolling), and hand *those* to the decoder.

The grid shows **every** entry, not only the ones with previews. A picker that silently omits the files it
cannot draw is lying about the contents of the directory, and "images and documents" is one filter rather
than two — a grid with the documents missing from it would be the common view rather than an edge case.

Not a widget for one app: the owner supplies the entries and the icon vocabulary, and gets back a grid.
What makes it a *file* grid rather than a thumbnail grid is only that it speaks `FileEntry`.

**Why the thumbnail cache lives here and not in `ThumbnailGrid`.** The grid deliberately knows nothing about
what an entry *is* — labels and indices, and no notion of identity — while a cache needs a stable key, which
here is the path. Pushing it down would mean widening the shared API with a "key for this index" hook for a
single consumer.

And the need is not shared either. Cherrypick sets its entries once per folder and filters with
`set_visible`, so its indices never move and it never re-decodes; what bites it is *unbounded* texture
growth in a huge folder, which an evicting cache would make worse rather than better. The reusable half is
already shared: `ThumbnailGrid.set_shared_image`, the hook that lets an owner hold textures of its own
across a rebuild.
"""

__all__ = ["FileGrid"]

import logging
import pathlib
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Optional, Union

import torch

import dearpygui.dearpygui as dpg

from .. import deviceinfo
from ..filelisting import FileEntry
from ..image import lanczos
from ..image import thumbnails
from . import utils as guiutils
from .thumbnailgrid import ThumbnailGrid
from .tileicons import TileIconCache

logger = logging.getLogger(__name__)

# Counter for unique DPG tags.
_tag_counter = 0
_tag_lock = threading.Lock()


def _next_tag() -> str:
    global _tag_counter
    with _tag_lock:
        _tag_counter += 1
        return f"filegrid_thumb_tex_{_tag_counter}"

# Roughly how much placeholder texture memory to keep, as tile_size² × count. Constant-ish by design: small
# tiles need many distinct noise tiles before the repetition shows, large ones need few, and this is what
# keeps the bill the same either way.
_PLACEHOLDER_AREA_BUDGET = 768 * 1024


def _placeholder_count(tile_size: int) -> int:
    return max(4, min(256, _PLACEHOLDER_AREA_BUDGET // (tile_size * tile_size)))


class FileGrid(ThumbnailGrid):
    """A `ThumbnailGrid` showing a `filelisting` directory listing.

    Give it entries with `set_listing`, and call `tick` once per frame — from wherever the owner's frames
    are driven, which need not be the render thread.
    """

    def __init__(self, parent: Union[str, int],
                 width: int, height: int,
                 *,
                 icon_assets: Mapping[str, tuple[int, int, Sequence[float]]],
                 icon_name_for: Callable[[FileEntry], Optional[str]],
                 selectable_for: Optional[Callable[[FileEntry], bool]] = None,
                 tile_size: int = 128,
                 thumbnail_device: str = "gpu",
                 thumbnail_dtype: torch.dtype = torch.float32,
                 lanczos_order: int = lanczos.DEFAULT_ORDER,
                 placeholder_count: Optional[int] = None,
                 thumbnail_cache_size: int = 512,
                 settle_time: float = 0.15,
                 on_current_entry_changed: Optional[Callable[[Optional[FileEntry]], None]] = None,
                 on_selection_changed_entries: Optional[Callable[[list[FileEntry]], None]] = None,
                 on_activate: Optional[Callable[[FileEntry], None]] = None,
                 **grid_kwargs):
        """
        *icon_assets*: ``{name: (width, height, flat RGBA floats)}`` — the shape `dpg.load_image` returns,
            so an asset already loaded for a toolbar can be handed over as it stands.
        *icon_name_for*: which icon an entry gets, by name. Returning `None` means "this one has a picture
            worth decoding", and is what puts an entry in the thumbnail queue. A name with no asset behind
            it draws the placeholder, which is a tolerable answer for a type nobody has drawn an icon for.
        *selectable_for*: which entries may join the selection. `None` admits all of them. The cursor still
            reaches an excluded entry — it must, or a directory could be neither reached by keyboard nor
            double-clicked — but a bulk action never sees one, and neither does the selection tint. For a
            picker the exclusions are `..` and the directories: it cannot return them, so showing them
            selected would read as a bug rather than as a rule.
        *thumbnail_device*: where decoded images are resized. The literal `"gpu"` is `deviceinfo`'s
            autodetect — the single available GPU backend, or CPU when there is none — and is the default
            *because an app that innocently wants a file picker must not have to know about torch devices*.
            Name one explicitly to pin thumbnails to a particular device, or to keep them off one already
            busy with inference.
        *placeholder_count*: how many distinct noise tiles stand in for the not-yet-decoded images. `None`
            scales it to the tile size so the memory stays about constant; 0 does without them, leaving a
            flat colour.
        *thumbnail_cache_size*: how many decoded thumbnails to keep, by path, across re-listings. Costs
            `tile_size² × 4` floats each — a quarter of a megabyte at 128 px — and buys a free re-filter, a
            free re-sort, and a free return to a folder visited a moment ago. Entries currently on the
            listing are never evicted, so a directory larger than this still shows every tile; the limit
            bounds what is kept for folders no longer on screen.
        *settle_time*: seconds the on-screen set must hold still before decoding starts. What stops a scroll
            from cancelling and restarting the decoder on the way past every row.
        *on_current_entry_changed*, *on_selection_changed_entries*, *on_activate*: as `ThumbnailGrid`'s
            `on_current_changed`, `on_selection_changed` and `on_double_click`, but handed `FileEntry`
            objects rather than indices. `on_activate` is what a double-click means here: descend into the
            directory, or choose the file.

        Remaining keyword arguments go to `ThumbnailGrid`.
        """
        self._entries: list[FileEntry] = []
        self._icon_name_for = icon_name_for
        self._selectable_for = selectable_for
        self._decodable: set[int] = set()
        self._on_current_entry_changed = on_current_entry_changed
        self._on_selection_changed_entries = on_selection_changed_entries
        self._on_activate = on_activate

        device_string, dtype = deviceinfo.get_device_and_dtype({"device_string": thumbnail_device,
                                                                "dtype": thumbnail_dtype})
        self._device = torch.device(device_string)
        self._dtype = dtype
        self._lanczos_order = lanczos_order

        # A file listing has no ordinal identity worth drawing. Cherrypick's does — triage is a pass over a
        # numbered sequence, and "image 47" is a thing its user says — but a folder is a set of *names*, and
        # a number in the corner of every tile reads as a joke to anyone who has seen the convention where
        # it belongs. Overridable like any other grid setting.
        grid_kwargs.setdefault("show_position_numbers", False)
        super().__init__(parent, width, height,
                         tile_size=tile_size,
                         on_current_changed=self._current_changed,
                         on_selection_changed=self._selection_changed,
                         on_double_click=self._double_clicked,
                         **grid_kwargs)

        self._icons = TileIconCache(tile_size, order=lanczos_order)
        self._icons.add_all(icon_assets)
        self._pipeline = thumbnails.ThumbnailPipeline(device=self._device, dtype=self._dtype,
                                                      tile_size=tile_size,
                                                      lanczos_order=lanczos_order)

        # Lazy-decode bookkeeping. `_batch` is the index mapping for whatever the pipeline is working on or
        # last worked on: it translates the pipeline's positions back to entry indices, and so must outlive
        # the batch itself — a result can arrive a tick after the batch is done.
        self._batch: list[int] = []
        self._batch_finished = True
        self._attempted: set[int] = set()  # in a batch that ran to completion; not asked for again
        self._wanted: list[int] = []
        self._wanted_since: float = 0.0
        self._settle_time = settle_time
        self._placeholder_count = placeholder_count

        # Decoded thumbnails, by path. Insertion-ordered, so the oldest is the first key.
        self._thumbnail_cache: dict[str, str] = {}
        self._thumbnail_cache_limit = thumbnail_cache_size

        self._refresh_placeholders()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_listing(self, entries: Sequence[FileEntry]) -> None:
        """Show `entries`, in the order given.

        The cursor is re-anchored **by path**, so a listing rebuilt under a changed filter or a changed sort
        leaves it on the same file. It falls to the first entry when that file is no longer listed, which is
        the only honest answer.

        **Thumbnails already decoded are re-attached, not decoded again.** A file dialog re-lists constantly
        — every keystroke in the find field is a new listing of the same directory — and the entries a file
        occupies move under it each time, so a texture remembered by index would be a picture of the wrong
        file. Remembering by path costs a dictionary and makes a re-filter, a re-sort and a return to a
        recently visited folder all free.
        """
        with self._lock:
            previous_path = self._current_path()
            self._entries = list(entries)
            self._pipeline.cancel()
            self._batch = []
            self._batch_finished = True
            self._attempted = set()
            self._wanted = []

            super().set_entries([entry.name for entry in self._entries])

            self._decodable = set()
            for idx, entry in enumerate(self._entries):
                name = self._icon_name_for(entry)
                if name is None:
                    self._decodable.add(idx)
                    cached = self._thumbnail_cache.get(entry.path)
                    if cached is not None:
                        self.set_shared_image(idx, cached)
                    continue
                texture = self._icons.texture(name)
                if texture is not None:
                    self.set_shared_image(idx, texture)

            self._evict_thumbnails()

            if previous_path is not None:
                for idx, entry in enumerate(self._entries):
                    if entry.path == previous_path:
                        self.set_current(idx)
                        break

    def get_entries(self) -> list[FileEntry]:
        with self._lock:
            return list(self._entries)
    entries = property(fget=get_entries, doc="The listing being shown, in display order.")

    def get_current_entry(self) -> Optional[FileEntry]:
        with self._lock:
            if 0 <= self._current < len(self._entries):
                return self._entries[self._current]
            return None
    current_entry = property(fget=get_current_entry, doc="The entry the cursor is on, or `None`.")

    def get_selected_entries(self) -> list[FileEntry]:
        with self._lock:
            return [self._entries[idx] for idx in sorted(self._selected) if idx < len(self._entries)]
    selected_entries = property(fget=get_selected_entries,
                                doc="The multi-selected entries, in display order.")

    def tick(self) -> None:
        """Do one frame's worth of work: collect finished thumbnails, update the grid, feed the decoder.

        Call once per frame. Safe off the render thread, which is where a widget inside somebody else's
        render loop generally has to live: DPG permits item work from any thread, and `visible_on_screen`
        reads what the last frame drew.
        """
        for position, flat_rgba in self._pipeline.poll():
            if 0 <= position < len(self._batch):
                self._store_thumbnail(self._batch[position], flat_rgba)
        self.update()
        self._pump_decoding()

    def set_tile_size(self, size: int) -> None:
        """Change the tile size, re-rendering the icons and placeholders and re-queueing the thumbnails."""
        with self._lock:
            if size == self._tile_size:
                return
            super().set_tile_size(size)  # drops every texture, ours and the icons' mapping alike
            self._icons.set_tile_size(size)
            self._pipeline.set_tile_size(size)
            self._clear_thumbnail_cache()  # every cached tile is now the wrong size
            self._batch = []
            self._batch_finished = True
            self._attempted = set()
            self._wanted = []
            self._refresh_placeholders()
            for idx, entry in enumerate(self._entries):
                if idx in self._decodable:
                    continue
                texture = self._icons.texture(self._icon_name_for(entry))
                if texture is not None:
                    self.set_shared_image(idx, texture)

    def destroy(self) -> None:
        """Stop the decoder and remove every DPG item. Call on shutdown."""
        with self._lock:
            self._pipeline.shutdown()
            self._icons.destroy()
            self._clear_thumbnail_cache()
            super().destroy()

    # ------------------------------------------------------------------
    # Internal: the thumbnail cache
    # ------------------------------------------------------------------

    def _store_thumbnail(self, idx: int, flat_rgba) -> None:
        """Take a decoded thumbnail from the pipeline: cache it by path, and show it.

        A **static** texture, and one the cache owns.

        Static costs nothing here: it cannot be updated after creation, and a cached thumbnail is written
        once and never changes. (`ThumbnailGrid.set_thumbnail` uses a dynamic one because it genuinely does
        update, when the same index is decoded again.) The reason to prefer it is that DPG carries a
        per-frame cost for every registered *dynamic* texture whether or not it is drawn — measured for this
        project in `raven.cherrypick.preload`, which keeps flat arrays rather than textures for that reason —
        and a cache exists precisely to hold textures nothing is currently drawing. That static textures
        escape that cost is inferred from the same note rather than separately measured; if they do not, this
        is merely no worse.

        Owned by the cache because `ThumbnailGrid` discards its own textures whenever the entries change,
        which here is every keystroke.
        """
        with self._lock:
            if not (0 <= idx < len(self._entries)):
                return
            path = self._entries[idx].path
            ts = self._tile_size
            if len(flat_rgba) != ts * ts * 4:
                return  # stale: the tile size changed while this was in flight
            tag = _next_tag()
            with dpg.texture_registry():
                dpg.add_static_texture(ts, ts, default_value=flat_rgba, tag=tag)
            # A path decoded twice would otherwise strand the first texture: nothing else refers to it, and
            # the cache is the only thing that would have deleted it.
            previous = self._thumbnail_cache.get(path)
            if previous is not None and previous != tag:
                guiutils.maybe_delete_item(previous)
            self._thumbnail_cache[path] = tag
            self.set_shared_image(idx, tag)

    def _evict_thumbnails(self) -> None:
        """Drop the oldest cached thumbnails, down to the limit.

        **Never one the current listing shows**, however old: evicting a tile that is on screen would blank
        it and then decode it again, which is the opposite of the point. So a directory larger than the
        limit still displays in full — what the limit bounds is how much is kept for folders that are not on
        screen any more, which is what makes going back to one free.
        """
        excess = len(self._thumbnail_cache) - self._thumbnail_cache_limit
        if excess <= 0:
            return
        in_use = {entry.path for entry in self._entries}
        for path in list(self._thumbnail_cache):  # insertion order: oldest first
            if excess <= 0:
                break
            if path in in_use:
                continue
            guiutils.maybe_delete_item(self._thumbnail_cache.pop(path))
            excess -= 1

    def _clear_thumbnail_cache(self) -> None:
        for tag in self._thumbnail_cache.values():
            guiutils.maybe_delete_item(tag)
        self._thumbnail_cache.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _current_path(self) -> Optional[str]:
        if 0 <= self._current < len(self._entries):
            return self._entries[self._current].path
        return None

    def _current_changed(self, idx: int) -> None:
        if self._on_current_entry_changed is not None:
            self._on_current_entry_changed(self.current_entry)

    def _selection_changed(self) -> None:
        if self._on_selection_changed_entries is not None:
            self._on_selection_changed_entries(self.selected_entries)

    def is_selectable(self, idx: int) -> bool:
        """Whether entry `idx` may join the selection, per the owner's `selectable_for`."""
        if self._selectable_for is None:
            return True
        if not (0 <= idx < len(self._entries)):
            return False
        return self._selectable_for(self._entries[idx])

    def _double_clicked(self, idx: int) -> None:
        if self._on_activate is None:
            return
        with self._lock:
            entry = self._entries[idx] if 0 <= idx < len(self._entries) else None
        if entry is not None:
            self._on_activate(entry)

    def _refresh_placeholders(self) -> None:
        """Fill the placeholder pool for the current tile size."""
        n = (_placeholder_count(self._tile_size) if self._placeholder_count is None
             else self._placeholder_count)
        if n <= 0:
            self.set_noise_pool([])
            return
        try:
            tiles = thumbnails.placeholder_tiles(n, self._tile_size,
                                                 device=self._device, dtype=self._dtype)
        except Exception as exc:
            # Not fatal: without a pool the tiles draw a flat colour, which is a duller wait rather than a
            # broken listing. Worth a warning, since it means the GPU path is unhappy.
            logger.warning(f"FileGrid._refresh_placeholders: instance 0x{id(self):x}: {type(exc)}: {exc}")
            return
        self.set_noise_pool(tiles)

    def _pump_decoding(self) -> None:
        """Keep the decoder pointed at whatever is on screen and still blank.

        Three things have to be true before a batch starts, and each of them is a bug if left out: the set
        has to have stopped changing (or a scroll restarts the decoder at every row it passes), it has to
        contain something not already in flight (or a completing thumbnail restarts the batch that is
        producing it), and an entry that a finished batch failed to decode must not come round again (or a
        file the decoder cannot read is retried for as long as it is on screen).
        """
        with self._lock:
            if self._batch and not self._batch_finished and not self._pipeline.in_progress:
                # The batch is done. Whatever it did not produce, it is not going to.
                self._batch_finished = True
                self._attempted.update(self._batch)

            wanted = [idx for idx in self.visible_on_screen()
                      # `_shared_images` is where a decoded thumbnail lands, the cache owning the texture —
                      # so an entry already in it has its picture, whether from this listing or a previous
                      # one that put it in the cache.
                      if idx in self._decodable and idx not in self._shared_images and idx not in self._attempted]
            now = time.monotonic()
            if wanted != self._wanted:
                self._wanted = wanted
                self._wanted_since = now
                return
            if not wanted or (now - self._wanted_since) < self._settle_time:
                return
            if not self._batch_finished and all(idx in self._batch for idx in wanted):
                return  # already being worked on

            self._batch = wanted
            self._batch_finished = False
            paths = [pathlib.Path(self._entries[idx].path) for idx in wanted]

        # Outside the lock: `start` waits for the previous batch's threads to notice the cancellation, and
        # holding the grid's lock across that would block every mouse handler for as long as it takes.
        logger.debug(f"FileGrid._pump_decoding: instance 0x{id(self):x}: decoding {len(paths)} thumbnails")
        self._pipeline.start(paths)
