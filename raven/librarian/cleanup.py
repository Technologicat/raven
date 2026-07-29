"""Datastore maintenance for the Librarian: the manual "Clean up & save" operation, and its preview dialog.

Two things live here, in that order — the operation, then the GUI that drives it. The operation half is pure
Python (no DPG) and testable on its own; only `DPGCleanupDialog` at the bottom touches the GUI.

**The operation** is a pair of prunes plus a save, always run together:

    prune_unreachable_nodes -> prune_dead_links -> prune_unreferenced_sidecars -> save

The order matters. Deleting a chat branch pops its nodes but leaves their attachments on disk, because
attachment storage is content-addressed and shared: the same image referenced from two branches is one file,
so no single node's deletion can decide the file's fate. Only a mark-and-sweep over every surviving payload
can, and it must run *after* the node prune, or attachments held solely by doomed nodes still look live.

**The preview** exists because the sweep is the one destructive step a user cannot undo and cannot inspect
beforehand: an orphaned sidecar is named by its content hash, so the folder it lives in is unreadable by eye.
So the dialog reports what would go, renders images as thumbnails and documents as a named list, and offers to
copy anything worth keeping to a staging directory before committing. Copy, not move — the sidecar stays until
the user commits, so changing their mind again costs nothing.

Names come from the description file written beside each sidecar at store time (`get_sidecar_metadata`), which
outlives the node that referenced it. A sidecar stored before descriptions existed falls back to its hash.
"""

__all__ = ["SidecarEntry",
           "CleanupPreview",
           "describe_sidecar",
           "preview_cleanup",
           "commit_cleanup",
           "rescue_to_staging",
           "format_size",

           "DPGCleanupDialog"]

import logging
logger = logging.getLogger(__name__)

import concurrent.futures
import dataclasses
import pathlib
import shutil
import urllib.parse
from typing import Any, Callable, Optional, Union

import dearpygui.dearpygui as dpg

from unpythonic.env import env

from ..common import bgtask
from ..common import utils as common_utils
from ..common.gui import utils as guiutils

from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa

from . import chattree
from . import sidecarstore
from . import config as librarian_config

gui_config = librarian_config.gui_config  # shorthand, as in `chat_controller`


# Kind dispatch for a sidecar with no usable description. Content type is the better signal (it is what the
# attaching store recorded), but a pre-description sidecar has none, and then the extension is all we have —
# it is the store's own choice of container, so it is reliable as far as it goes.
_IMAGE_EXTENSIONS = frozenset([".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"])


@dataclasses.dataclass
class SidecarEntry:
    """One unreferenced attachment, as the cleanup preview needs to show it.

    `filename`: the sidecar's on-disk name — a content hash plus extension. Unique, and the handle for every
                operation on it (read, rescue, delete).
    `display_name`: the best human-readable name recoverable for it, or `filename` when nothing better exists.
    `size_bytes`: size on disk.
    `is_image`: which of the preview's two sections this belongs in (see `describe_sidecar` for how it is decided).
    `metadata`: the stored description, verbatim; `{}` for a sidecar predating descriptions. Read for the
                tooltip, and kept whole so a caller can look at fields this dataclass does not promote.
    `companion_filenames` / `companion_bytes`: further sidecar files that live and die with this one, and the
                disk they occupy. An image stored over the downsampling cap keeps its untouched original as a
                second sidecar; that pair is one attachment to the user, and is presented as one — see
                `preview_cleanup`.
    """
    filename: str
    display_name: str
    size_bytes: int
    is_image: bool
    metadata: dict[str, Any]
    companion_filenames: list[str] = dataclasses.field(default_factory=list)
    companion_bytes: int = 0

    def get_archival_filename(self) -> str:
        """The best stored copy of this attachment — the preserved original if there is one, else the file itself.

        The same resolution the chat log's "show full-size image" action makes, and for the same reason: what
        the user wants to look at (or keep) is the untouched original, not the downsample made for the model.
        A document is never transformed, so for one of those this is always `filename`.
        """
        return self.metadata.get("original_sidecar") or self.filename

    archival_filename = property(fget=get_archival_filename,
                                 doc="The best stored copy: preserved original if any, else the file itself. See `get_archival_filename`.")

    def get_total_bytes(self) -> int:
        """Disk this attachment occupies in total — itself plus any companion files deleted alongside it."""
        return self.size_bytes + self.companion_bytes

    total_bytes = property(fget=get_total_bytes,
                           doc="Disk this attachment occupies in total, companions included. See `get_total_bytes`.")

    def get_sort_key(self) -> tuple[str, str]:
        """Case-insensitive by display name, with the filename breaking ties (two attachments can share a name).

        Alphabetical is a weak ordering for a set of forgotten files, but it is the only one available while
        they are a flat unordered set: semantic grouping — "everything from that one paper", "all the diagrams"
        — is the ordering that would actually help, and it waits on embeddings the datastore does not have yet.
        """
        return (self.display_name.casefold(), self.filename)

    sort_key = property(fget=get_sort_key,
                        doc="Case-insensitive display name, then filename. See `get_sort_key`.")


@dataclasses.dataclass
class CleanupPreview:
    """What a cleanup would delete, as computed by `preview_cleanup` — a dry run, nothing has been touched yet.

    `node_ids`: chat nodes unreachable from the roots. Usually invisible to the user already (an unreachable
                node is not on any branch they can navigate to), so this is reported as a count, not a list.
    `images` / `documents`: the unreferenced attachments, split by kind and sorted by `SidecarEntry.sort_key`.
                            Two lists rather than one because they are shown differently: an image is
                            recognized by looking at it, a document by reading its name.
    """
    node_ids: list[str]
    images: list[SidecarEntry]
    documents: list[SidecarEntry]

    def get_sidecars(self) -> list[SidecarEntry]:
        """All unreferenced attachments, both kinds, images first."""
        return self.images + self.documents

    sidecars = property(fget=get_sidecars,
                        doc="All unreferenced attachments, both kinds, images first. See `get_sidecars`.")

    def get_total_bytes(self) -> int:
        """Disk space the sidecar sweep would reclaim, over both kinds, companion files included."""
        return sum(entry.total_bytes for entry in self.get_sidecars())

    total_bytes = property(fget=get_total_bytes,
                           doc="Disk space the sidecar sweep would reclaim. See `get_total_bytes`.")

    def get_is_empty(self) -> bool:
        """Whether there is nothing to do — no unreachable nodes and no unreferenced attachments."""
        return not self.node_ids and not self.images and not self.documents

    is_empty = property(fget=get_is_empty,
                        doc="Whether a cleanup would do nothing at all. See `get_is_empty`.")


def describe_sidecar(datastore: chattree.PersistentForest, filename: str) -> SidecarEntry:
    """Look up everything the preview needs about one sidecar file, from its description and the file itself.

    Kind is decided by the recorded `content_type` when there is one, and by file extension otherwise. The
    display name comes from the document `name` field if present, else the basename of the provenance URL (how
    an image carries its original name — see `sidecarstore.provenance_filename_from_url`), else the sidecar
    filename itself, which is a hash and tells the user nothing but is at least unique.

    Never raises for a sidecar that has vanished or has no description: a preview of a directory that is being
    written to concurrently should degrade to less information, not to an error.
    """
    metadata = datastore.get_sidecar_metadata(filename) or {}
    content_type = metadata.get("content_type") or ""
    if content_type:
        is_image = content_type.startswith("image/")
    else:
        is_image = pathlib.Path(filename).suffix.lower() in _IMAGE_EXTENSIONS
    display_name = (metadata.get("name") or
                    sidecarstore.provenance_filename_from_url(metadata.get("url")) or
                    filename)
    try:
        size_bytes = datastore.sidecar_path(filename).stat().st_size
    except OSError:  # vanished under us, or unreadable; report it as present-but-unknown rather than failing
        size_bytes = 0
    return SidecarEntry(filename=filename,
                        display_name=display_name,
                        size_bytes=size_bytes,
                        is_image=is_image,
                        metadata=metadata)


def preview_cleanup(datastore: chattree.PersistentForest, *roots: str) -> CleanupPreview:
    """Compute what `commit_cleanup(datastore, *roots)` would delete, without deleting anything.

    The two steps are asked in the same order they would run, and the second is told the answer to the first:
    the sidecar dry run discounts references held by the nodes the node prune is about to take. Asking them
    independently would under-report by exactly the attachments a cleanup exists to reclaim — the ones whose
    only remaining referent is a branch the user already deleted.

    Preserved originals are folded into the images they belong to rather than listed in their own right. An
    image stored over the downsampling cap occupies two sidecar files, both referenced from the same payload,
    so both fall unreferenced together — and a flat listing would show the same picture twice, under the same
    name, which reads as a bug and makes the count wrong. The chat log resolves the pair the same way (one
    thumbnail, the original reachable through it); a preview that disagreed with it about what "an attachment"
    is would be teaching the user two different data models for the same files.
    """
    node_ids = datastore.list_unreachable_nodes(*roots)
    filenames = datastore.list_unreferenced_sidecars(excluding_nodes=node_ids)
    entries = {filename: describe_sidecar(datastore, filename) for filename in filenames}

    for entry in entries.values():
        companion = entry.metadata.get("original_sidecar")
        if companion is not None and companion in entries and companion != entry.filename:
            entry.companion_filenames.append(companion)
            entry.companion_bytes += entries[companion].size_bytes
    subsumed = {companion for entry in entries.values() for companion in entry.companion_filenames}
    listed = [entry for filename, entry in entries.items() if filename not in subsumed]

    images = sorted((entry for entry in listed if entry.is_image), key=SidecarEntry.get_sort_key)
    documents = sorted((entry for entry in listed if not entry.is_image), key=SidecarEntry.get_sort_key)
    return CleanupPreview(node_ids=node_ids, images=images, documents=documents)


def commit_cleanup(datastore: chattree.PersistentForest, *roots: str) -> env:
    """Prune the datastore and persist it: unreachable nodes, then dead links, then unreferenced sidecars, then save.

    This is the whole "Clean up & save" operation. Saving is part of it rather than left to the caller, because
    the prunes have already mutated the in-memory forest by the time this returns — leaving the file unwritten
    would put disk and memory into states that disagree about what exists, which is worse than either.

    Returns an `env` with `deleted_node_ids` and `deleted_sidecars` (both lists), so the caller can report what
    actually happened rather than what the preview predicted. The two can differ: the datastore is live, and a
    chat turn between the preview and the commit may have referenced or created files.
    """
    with datastore.lock:
        deleted_node_ids = datastore.list_unreachable_nodes(*roots)
        datastore.prune_unreachable_nodes(*roots)
        datastore.prune_dead_links(*roots)
        deleted_sidecars = datastore.prune_unreferenced_sidecars()
        datastore.save()
    logger.info(f"commit_cleanup: deleted {len(deleted_node_ids)} unreachable node(s) and "
                f"{len(deleted_sidecars)} unreferenced sidecar file(s); datastore saved.")
    return env(deleted_node_ids=deleted_node_ids, deleted_sidecars=deleted_sidecars)


def rescue_to_staging(datastore: chattree.PersistentForest,
                      entry: SidecarEntry,
                      staging_dir: Optional[Union[str, pathlib.Path]] = None) -> pathlib.Path:
    """Copy one about-to-be-deleted sidecar out to the staging directory, under its human-readable name.

    `staging_dir` defaults to `librarian_config.attachment_staging_dir`, and is created if missing.

    What gets copied is `entry.archival_filename` — the preserved original when the stored image was
    downsampled, the file itself otherwise. Rescuing the downsample instead would hand the user a copy that is
    strictly worse than one sitting right next to it, and the downsample is derivable from the original anyway.

    Copy rather than move, so the sidecar is still there if the user cancels the cleanup after all. Returns the
    path written. Raises like the underlying copy on an unwritable destination, so a GUI caller can flash a
    failure rather than silently claiming a rescue that did not happen.

    Name collisions are resolved in favour of not losing data: an existing file with identical content is
    treated as the same rescue and its path returned unchanged (rescuing twice is idempotent), and an existing
    file with *different* content gets a ` (2)`, ` (3)`, ... suffix rather than being overwritten. Two distinct
    attachments really can share a display name — the names come from wherever the user got the files.
    """
    directory = pathlib.Path(staging_dir if staging_dir is not None
                             else librarian_config.attachment_staging_dir).expanduser().resolve()
    common_utils.create_directory(directory)
    source = datastore.sidecar_path(entry.archival_filename)

    stem = pathlib.Path(entry.display_name).stem or entry.filename
    suffix = pathlib.Path(entry.display_name).suffix or pathlib.Path(entry.archival_filename).suffix
    candidate = directory / f"{stem}{suffix}"
    counter = 2
    while candidate.exists():
        if candidate.stat().st_size == source.stat().st_size and candidate.read_bytes() == source.read_bytes():
            logger.info(f"rescue_to_staging: '{entry.filename}' is already staged at '{candidate}'.")
            return candidate
        candidate = directory / f"{stem} ({counter}){suffix}"
        counter += 1

    shutil.copyfile(source, candidate)
    logger.info(f"rescue_to_staging: copied sidecar '{entry.filename}' to '{candidate}'.")
    return candidate


def format_size(size_bytes: int) -> str:
    """Render a byte count for the preview: `"812 kB"`, `"3.4 MB"`. Decimal units, as disk sizes are quoted."""
    if size_bytes < 1000:
        return f"{size_bytes} B"
    for unit, scale in (("kB", 10**3), ("MB", 10**6), ("GB", 10**9)):
        if size_bytes < 1000 * scale:
            value = size_bytes / scale
            return f"{value:.0f} {unit}" if value >= 100 else f"{value:.1f} {unit}"
    return f"{size_bytes / 10**12:.1f} TB"


def _plural(count: int, singular: str, plural: Optional[str] = None) -> str:
    """`"1 node"`, `"3 nodes"` — the count and its noun, agreeing. `plural` defaults to `singular + "s"`."""
    return f"{count} {singular if count == 1 else (plural if plural is not None else singular + 's')}"


def _ellipsize(text: str, max_chars: int) -> str:
    """Shorten `text` to `max_chars` by eliding the *middle*, e.g. `"quarterly_re…port_2026.pdf"`.

    Middle rather than end, because the informative parts of an attachment's name sit at both ends: the topic
    at the front, the file type at the back. Chopping the tail throws away the extension, which is precisely
    what tells the reader whether the thing about to be deleted is a paper or a slide deck.
    """
    if len(text) <= max_chars:
        return text
    keep = max_chars - 1  # the ellipsis costs one character
    head = keep - keep // 2
    return f"{text[:head]}…{text[len(text) - keep // 2:]}"


class DPGCleanupDialog:
    """The "Clean up & save" dialog: shows what a cleanup would delete, lets the user rescue any of it, commits.

    One instance is built with the GUI and reused; `open` recomputes the preview and rebuilds the window every
    time, because the answer changes with every chat turn. Nothing is deleted until the user clicks the commit
    button, so opening the dialog is always safe.

    `datastore`: the chat datastore to clean.
    `get_roots`: called at open time (not at construction) for the root node IDs everything must stay reachable
                 from — normally a one-tuple of the system prompt node. Reading it fresh matters: a factory
                 reset replaces that node, and a dialog holding the old ID would declare the entire chat
                 history unreachable.
    `executor`: `ThreadPoolExecutor` for the thumbnail decoding, which is too slow to do on the GUI thread.
    `themes_and_fonts`: the app's `guiutils.bootup` result, for the icon font the buttons are drawn in.
    `save_app_state`: optional zero-argument callable, run after a successful commit. The button says "save",
                      and a user reading that expects their toggles saved too, not just the chat tree.
    `on_committed`: optional callback, receives the `env` from `commit_cleanup`. For acknowledging the action
                    on whatever surface opened the dialog.
    `centering_reference_window`: DPG tag or ID to center on; the main window, normally.
    """

    # A grid this wide fits four thumbnails plus their names in a dialog that still leaves the chat visible
    # behind it. Fixed rather than computed from the window width: the window is a fixed size too, and a
    # reflowing grid buys nothing when the container cannot change shape.
    grid_columns = 4

    def __init__(self,
                 *,
                 datastore: chattree.PersistentForest,
                 get_roots: Callable[[], tuple],
                 executor: concurrent.futures.Executor,
                 themes_and_fonts: env,
                 save_app_state: Optional[Callable[[], None]] = None,
                 on_committed: Optional[Callable[[env], None]] = None,
                 centering_reference_window: Optional[Union[str, int]] = None):
        self.datastore = datastore
        self.get_roots = get_roots
        self.themes_and_fonts = themes_and_fonts
        self.save_app_state = save_app_state
        self.on_committed = on_committed
        self.centering_reference_window = centering_reference_window

        self.task_manager = bgtask.TaskManager(name="librarian_cleanup_dialog",
                                               mode="sequential",  # a re-open cancels a still-running thumbnail load
                                               executor=executor)
        self.texture_registry = dpg.add_texture_registry(tag="librarian_cleanup_texture_registry")  # tag

        self.is_open = False
        self.preview = None
        self.window_id = None
        # Rebuild counter, part of every texture tag this dialog creates. DPG frees deleted items lazily, so a
        # texture from the previous build may still hold its tag when the next build asks for the same one —
        # and a duplicate tag takes the process down rather than raising. See CLAUDE.md, DPG pitfall #5.
        self.build_number = 0
        self._image_slots = []    # env(entry, cell_group_id): where the background loader puts each thumbnail
        self._rescue_widgets = {}  # sidecar filename -> env(entry, button_id, tooltip_text_id)

    def open(self) -> None:
        """Compute a fresh preview and show the dialog. Deletes nothing. Safe to call when already open (no-op)."""
        if self.is_open:
            return
        self.preview = preview_cleanup(self.datastore, *self.get_roots())
        self.build_number += 1
        self._build_window()
        self.is_open = True

        dpg.split_frame()  # let any modal that is closing finish first, or ours may not appear (as in `messagebox`)
        if self.centering_reference_window is not None:
            guiutils.recenter_window(self.window_id, reference_window=self.centering_reference_window)
        else:
            dpg.show_item(self.window_id)

        if self.preview.images:
            self.task_manager.submit(self._load_thumbnails_task, env())

    def close(self) -> None:
        """Close the dialog and release its widgets and textures. Deletes nothing from the datastore."""
        self.is_open = False
        self.task_manager.clear(wait=False)  # a thumbnail load still in flight has nowhere to draw now
        if self.window_id is not None:
            with guiutils.nonexistent_ok():
                dpg.delete_item(self.window_id)
        self.window_id = None
        self._image_slots.clear()
        self._rescue_widgets.clear()
        # Textures go after the widgets that draw them, never before. The build counter in their tags means a
        # not-yet-collected texture cannot collide with the next build's, so this is a disk/VRAM courtesy
        # rather than a correctness requirement.
        with guiutils.nonexistent_ok():
            dpg.delete_item(self.texture_registry, children_only=True)

    def _build_window(self) -> None:
        """Build the dialog window for the current `self.preview`, hidden. Replaces any previous build."""
        if self.window_id is not None:  # by ID, never by alias — see CLAUDE.md, DPG pitfall #6
            with guiutils.nonexistent_ok():
                dpg.delete_item(self.window_id)
        self._image_slots.clear()
        self._rescue_widgets.clear()

        preview = self.preview
        with dpg.window(label="Clean up chat data",
                        modal=True,
                        show=False,
                        no_collapse=True,
                        width=900,
                        height=620,
                        on_close=lambda: self.close()) as window_id:
            self.window_id = window_id

            if preview.is_empty:
                dpg.add_text("Nothing to clean up — no unreachable chat nodes, no unreferenced attachments.")
                dpg.add_separator()
                dpg.add_button(label="Close", width=self.button_w, callback=lambda: self.close())
                return

            dpg.add_text(self._summary_text())
            if preview.sidecars:
                # Same triangle and same orange as the main window's AI disclosure, so the two read as one
                # vocabulary of warning rather than as two unrelated colored texts. No spacer above the icon,
                # unlike there: this warning is a single line, so the glyph already sits on the text's line.
                with dpg.group(horizontal=True):
                    icon_id = dpg.add_text(fa.ICON_TRIANGLE_EXCLAMATION, color=(255, 180, 120))  # orange
                    dpg.bind_item_font(icon_id, self.themes_and_fonts.icon_font_solid)
                    dpg.add_text("Attachments are deleted for good. Save a copy of anything you want to keep first.",
                                 color=(255, 180, 120))  # orange
            dpg.add_separator()

            with dpg.child_window(height=-72):  # leave room for the two button rows below
                if preview.images:
                    self._build_image_section()
                if preview.documents:
                    self._build_document_section()

            dpg.add_separator()
            self._build_action_buttons()

    def _summary_text(self) -> str:
        """The one-line "would delete ..." headline, naming only the categories that actually have something in them."""
        preview = self.preview
        parts = []
        if preview.node_ids:
            parts.append(_plural(len(preview.node_ids), "unreachable chat node"))
        if preview.sidecars:
            parts.append(f"{_plural(len(preview.sidecars), 'unreferenced attachment')} "
                         f"({format_size(preview.total_bytes)})")
        return "Would delete " + " and ".join(parts) + "."

    def _build_image_section(self) -> None:
        """The image grid: a thumbnail per orphaned image, with its name and a rescue button.

        The cells are built empty and filled in later by `_load_thumbnails_task` — decoding is far too slow to
        do while the user waits for the dialog to appear. The section is collapsed by default, so in the common
        case the thumbnails are ready before anyone looks at them.
        """
        images = self.preview.images
        size = format_size(sum(entry.size_bytes for entry in images))
        with dpg.collapsing_header(label=f"Images ({len(images)}, {size})", default_open=False):
            with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp):
                for _ in range(self.grid_columns):
                    dpg.add_table_column()
                for row_start in range(0, len(images), self.grid_columns):
                    with dpg.table_row():
                        for entry in images[row_start:row_start + self.grid_columns]:
                            with dpg.table_cell():
                                cell = dpg.add_group()
                                self._image_slots.append(env(entry=entry, cell_group_id=cell))
                                dpg.add_text(_ellipsize(entry.display_name, 22))
                                with dpg.group(horizontal=True):
                                    dpg.add_text(format_size(entry.total_bytes), color=(140, 140, 140))
                                    self._add_open_button(entry)
                                    self._add_rescue_button(entry)

    def _build_document_section(self) -> None:
        """The document list: one named row per orphaned document, with its size and a rescue button.

        A list rather than a grid because a document has no thumbnail — a file-type icon in a picture grid
        would be a tile pretending to be a picture. What identifies a document is its name.
        """
        documents = self.preview.documents
        size = format_size(sum(entry.size_bytes for entry in documents))
        with dpg.collapsing_header(label=f"Documents ({len(documents)}, {size})", default_open=False):
            for entry in documents:
                with dpg.group(horizontal=True):
                    self._add_open_button(entry)
                    self._add_rescue_button(entry)
                    icon_id = dpg.add_text(fa.ICON_FILE_LINES)
                    dpg.bind_item_font(icon_id, self.themes_and_fonts.icon_font_solid)
                    dpg.add_text(entry.display_name)
                    dpg.add_text(f"({format_size(entry.total_bytes)})", color=(140, 140, 140))

    def _add_open_button(self, entry: SidecarEntry) -> None:
        """Add this attachment's "open it in the default application" button, into the current DPG container.

        The same action the chat log offers on an inline attachment, and it opens the same copy — the preserved
        original where there is one. A thumbnail at 140 px is enough to remember an image by but not always
        enough to judge it, and a document has no preview at all, so this is what makes "do I still want this?"
        answerable before the deletion rather than after.
        """
        button_id = dpg.add_button(label=fa.ICON_IMAGE if entry.is_image else fa.ICON_FILE_LINES,
                                   width=gui_config.toolbutton_w,
                                   callback=lambda: self._open_one(entry))
        dpg.bind_item_font(button_id, self.themes_and_fonts.icon_font_solid)
        with dpg.tooltip(button_id):
            dpg.add_text(f"Open '{entry.display_name}'\nin its default application")

    def _open_one(self, entry: SidecarEntry) -> None:
        """Open one attachment's stored copy in the OS default application. Logs a failure rather than raising."""
        try:
            common_utils.open_file(self.datastore.sidecar_path(entry.archival_filename))
        except Exception as exc:  # noqa: BLE001 -- a failed open must leave the dialog usable
            logger.error(f"DPGCleanupDialog._open_one: could not open '{entry.archival_filename}': "
                         f"{type(exc)}: {exc}")

    def _add_rescue_button(self, entry: SidecarEntry) -> None:
        """Add this attachment's "Save a copy to staging" button, into the current DPG container.

        On success the button turns into a check mark and goes inert, and its tooltip becomes the path written
        — so the grid doubles as a record of what has already been rescued in this session, which matters when
        the user is working through a few dozen thumbnails deciding one at a time.
        """
        button_id = dpg.add_button(label=fa.ICON_DOWNLOAD,
                                   width=gui_config.toolbutton_w,
                                   callback=lambda: self._rescue_one(entry))
        dpg.bind_item_font(button_id, self.themes_and_fonts.icon_font_solid)
        dpg.bind_item_theme(button_id, "disablable_widget_theme")  # tag  # it goes inert once rescued
        with dpg.tooltip(button_id):
            tooltip_text_id = dpg.add_text(f"Save a copy of '{entry.display_name}'\n"
                                           f"to the staging folder, before it is deleted")
        self._rescue_widgets[entry.filename] = env(entry=entry,
                                                   button_id=button_id,
                                                   tooltip_text_id=tooltip_text_id)

    def _rescue_one(self, entry: SidecarEntry) -> None:
        """Copy one attachment to staging and mark its button done. Reports failure on the button, not by raising."""
        try:
            staged_path = rescue_to_staging(self.datastore, entry)
        except Exception as exc:  # noqa: BLE001 -- a failed rescue must leave the dialog usable for the rest
            logger.error(f"DPGCleanupDialog._rescue_one: could not stage '{entry.filename}': {type(exc)}: {exc}")
            self._mark_rescued(entry, message=f"Could not save '{entry.display_name}'\n{type(exc).__name__}: {exc}",
                               ok=False)
        else:
            self._mark_rescued(entry, message=f"Saved to\n{staged_path}", ok=True)

    def _mark_rescued(self, entry: SidecarEntry, *, message: str, ok: bool) -> None:
        """Update one rescue button to its post-click state: a check (done, inert) or a warning (still clickable)."""
        widgets = self._rescue_widgets.get(entry.filename)
        if widgets is None:
            return
        with guiutils.nonexistent_ok():  # the dialog may have been closed under a slow copy
            dpg.configure_item(widgets.button_id,
                               label=fa.ICON_CHECK if ok else fa.ICON_TRIANGLE_EXCLAMATION,
                               enabled=not ok)  # a failed rescue stays clickable: the cause may be fixable
            dpg.set_value(widgets.tooltip_text_id, message)

    def _rescue_all(self) -> None:
        """Copy every listed attachment to staging. Already-rescued ones are re-copied harmlessly (same bytes, same path)."""
        for entry in self.preview.sidecars:
            self._rescue_one(entry)

    # One width for every button in the dialog, so the two rows line up into a grid instead of a ragged edge.
    # Sized for the longest label ("Open staging folder").
    button_w = 180

    def _build_action_buttons(self) -> None:
        """The two button rows: rescue actions, then the commit/cancel pair, into the current DPG container."""
        preview = self.preview
        with dpg.group(horizontal=True):
            save_all_id = dpg.add_button(label="Save all to staging", width=self.button_w,
                                         callback=lambda: self._rescue_all(),
                                         enabled=bool(preview.sidecars))
            dpg.bind_item_theme(save_all_id, "disablable_widget_theme")  # tag  # nothing to save -> inert
            with dpg.tooltip(save_all_id):
                dpg.add_text(f"Copy all {len(preview.sidecars)} attachment(s) to\n"
                             f"{librarian_config.attachment_staging_dir}\n"
                             f"before deleting them")
            open_staging_id = dpg.add_button(label="Open staging folder", width=self.button_w,
                                             callback=lambda: self._open_staging_dir())
            with dpg.tooltip(open_staging_id):
                dpg.add_text(f"Open {librarian_config.attachment_staging_dir}\nin the file manager")

        with dpg.group(horizontal=True):
            commit_id = dpg.add_button(label="Clean up & save", width=self.button_w, callback=lambda: self._commit())
            with dpg.tooltip(commit_id):
                dpg.add_text("Delete the items listed above, then save the chat data.\nThis cannot be undone.")
            dpg.add_button(label="Cancel", width=self.button_w, callback=lambda: self.close())

    def _open_staging_dir(self) -> None:
        """Reveal the staging directory, creating it first — it may not exist yet if nothing has been rescued."""
        try:
            common_utils.create_directory(librarian_config.attachment_staging_dir)
            common_utils.open_in_file_manager(librarian_config.attachment_staging_dir)
        except Exception as exc:  # noqa: BLE001 -- opening a folder must never take the dialog down with it
            logger.error(f"DPGCleanupDialog._open_staging_dir: {type(exc)}: {exc}")

    def _commit(self) -> None:
        """Run the cleanup for real, then close. The preview is recomputed inside `commit_cleanup`, not trusted from here."""
        try:
            result = commit_cleanup(self.datastore, *self.get_roots())
            if self.save_app_state is not None:
                self.save_app_state()
        except Exception as exc:  # noqa: BLE001 -- report a failed cleanup; never leave the dialog stuck open
            logger.error(f"DPGCleanupDialog._commit: {type(exc)}: {exc}")
            self.close()
            return
        self.close()
        if self.on_committed is not None:
            self.on_committed(result)

    def _load_thumbnails_task(self, task_env: env) -> None:
        """Background task: decode each orphaned image and drop a thumbnail into its waiting grid cell.

        Runs off the GUI thread because Lanczos-resampling a few dozen images is not something to do between
        two frames. Textures are created from here, which DPG permits from any thread; the two `split_frame`s
        after each upload are what make the texture drawable before the image widget referencing it appears
        (one wait empirically isn't enough — see dpg-notes.md, "Texture upload ordering").

        Thumbnails are letterboxed into uniform squares rather than fitted to their own aspect ratio, which is
        what the chat log does with the same images. A grid wants a regular lattice — cells of assorted heights
        read as damage — and the padding is cheaper than the alternative of cropping to square, which would
        discard the part of a forgotten image that identifies it.
        """
        from ..common.image import codec  # deferred: pulls torch / Pillow only when there are images to show
        from ..common.image import utils as image_utils

        build_number = self.build_number
        for slot in self._image_slots:
            if task_env.cancelled or build_number != self.build_number:  # dialog closed, or reopened under us
                return
            entry = slot.entry
            try:
                tile = librarian_config.cleanup_thumbnail_size
                raw = self.datastore.read_sidecar(entry.filename)
                arr = image_utils.ensure_rgba(codec.decode(raw))  # (H, W, 4) uint8
                tensor = image_utils.np_to_tensor(arr, device="cpu")  # (1, 4, H, W) float32
                flat = image_utils.tensor_to_dpg_flat(image_utils.letterbox(tensor, tile))
                disp_w = disp_h = tile
                texture_tag = f"librarian_cleanup_thumbnail_{build_number}_{entry.filename}"  # tag
                dpg.add_static_texture(disp_w, disp_h, flat,
                                       tag=texture_tag,  # tag
                                       parent=self.texture_registry)
                dpg.split_frame()  # trigger the deferred OpenGL upload...
                dpg.split_frame()  # ...and ensure it completed before the image widget draws it
                if task_env.cancelled or build_number != self.build_number:
                    return
                with guiutils.nonexistent_ok():  # the cell is gone if the dialog closed during the two waits
                    image_id = dpg.add_image(texture_tag, width=disp_w, height=disp_h,  # tag
                                             parent=slot.cell_group_id)
                    with dpg.tooltip(image_id):
                        dpg.add_text(self._thumbnail_tooltip_text(entry))
            except Exception as exc:  # noqa: BLE001 -- one undecodable orphan must not stop the rest of the grid
                logger.error(f"DPGCleanupDialog._load_thumbnails_task: could not render '{entry.filename}': "
                             f"{type(exc)}: {exc}")
                with guiutils.nonexistent_ok():
                    dpg.add_text("[unreadable]", color=(180, 120, 120), parent=slot.cell_group_id)

    def _thumbnail_tooltip_text(self, entry: SidecarEntry) -> str:
        """Everything known about one orphaned image, for its tooltip: name, size, when it was stored, where from.

        Fuller than the grid cell can show, and the point of the exercise — recognizing a forgotten attachment
        takes the picture *plus* the story of where it came from.
        """
        lines = [entry.display_name, format_size(entry.total_bytes)]
        if entry.companion_filenames:
            # Otherwise the size reads as wrong: the tile shows one picture, but two files go with it.
            lines.append("Downsampled copy + preserved original")
        stored_at = entry.metadata.get("fetched_at")
        if stored_at:
            lines.append(f"Attached {stored_at}")
        source_url = entry.metadata.get("url")
        if source_url and not source_url.startswith("data:"):
            lines.append(urllib.parse.unquote(source_url))
        if entry.display_name != entry.filename:  # otherwise the next line would just repeat the name
            lines.append(f"Stored as {entry.filename}")
        return "\n".join(lines)
