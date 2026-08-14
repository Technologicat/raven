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
"""

__all__ = ["FileGrid"]

import logging
import pathlib
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Optional, Union

import torch

from .. import deviceinfo
from ..filelisting import FileEntry
from ..image import lanczos
from ..image import thumbnails
from .thumbnailgrid import ThumbnailGrid
from .tileicons import TileIconCache

logger = logging.getLogger(__name__)

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
                 tile_size: int = 128,
                 thumbnail_device: str = "gpu",
                 thumbnail_dtype: torch.dtype = torch.float32,
                 lanczos_order: int = lanczos.DEFAULT_ORDER,
                 placeholder_count: Optional[int] = None,
                 settle_time: float = 0.15,
                 on_current_entry_changed: Optional[Callable[[Optional[FileEntry]], None]] = None,
                 on_activate: Optional[Callable[[FileEntry], None]] = None,
                 **grid_kwargs):
        """
        *icon_assets*: ``{name: (width, height, flat RGBA floats)}`` — the shape `dpg.load_image` returns,
            so an asset already loaded for a toolbar can be handed over as it stands.
        *icon_name_for*: which icon an entry gets, by name. Returning `None` means "this one has a picture
            worth decoding", and is what puts an entry in the thumbnail queue. A name with no asset behind
            it draws the placeholder, which is a tolerable answer for a type nobody has drawn an icon for.
        *thumbnail_device*: where decoded images are resized. The literal `"gpu"` is `deviceinfo`'s
            autodetect — the single available GPU backend, or CPU when there is none — and is the default
            *because an app that innocently wants a file picker must not have to know about torch devices*.
            Name one explicitly to pin thumbnails to a particular device, or to keep them off one already
            busy with inference.
        *placeholder_count*: how many distinct noise tiles stand in for the not-yet-decoded images. `None`
            scales it to the tile size so the memory stays about constant; 0 does without them, leaving a
            flat colour.
        *settle_time*: seconds the on-screen set must hold still before decoding starts. What stops a scroll
            from cancelling and restarting the decoder on the way past every row.
        *on_current_entry_changed*, *on_activate*: as `ThumbnailGrid`'s `on_current_changed` and
            `on_double_click`, but handed the `FileEntry` rather than an index. `on_activate` is what a
            double-click means here: descend into the directory, or choose the file.

        Remaining keyword arguments go to `ThumbnailGrid`.
        """
        self._entries: list[FileEntry] = []
        self._icon_name_for = icon_name_for
        self._decodable: set[int] = set()
        self._on_current_entry_changed = on_current_entry_changed
        self._on_activate = on_activate

        device_string, dtype = deviceinfo.get_device_and_dtype({"device_string": thumbnail_device,
                                                                "dtype": thumbnail_dtype})
        self._device = torch.device(device_string)
        self._dtype = dtype
        self._lanczos_order = lanczos_order

        super().__init__(parent, width, height,
                         tile_size=tile_size,
                         on_current_changed=self._current_changed,
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

        self._refresh_placeholders()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_listing(self, entries: Sequence[FileEntry]) -> None:
        """Show `entries`, in the order given.

        The cursor is re-anchored **by path**, so a listing rebuilt under a changed filter or a changed sort
        leaves it on the same file. It falls to the first entry when that file is no longer listed, which is
        the only honest answer.
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
                    continue
                texture = self._icons.texture(name)
                if texture is not None:
                    self.set_shared_image(idx, texture)

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
                self.set_thumbnail(self._batch[position], flat_rgba)
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
            super().destroy()

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
                      if idx in self._decodable and idx not in self._textures and idx not in self._attempted]
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
