"""Datastore maintenance for the Librarian: the manual "Clean up & save" operation.

Pure Python, deliberately — no `dearpygui` anywhere in this module's import graph. The dialog that drives it
lives in `cleanup_dialog`. The separation is not tidy-mindedness: the test environment installs no GUI toolkit
(headless, on three operating systems), so a `dpg` import here would make this half uncollectable and leave
the operation untested. Presentation logic that does not touch `dpg` — name shortening, size formatting —
still belongs here.

**The operation** is a pair of prunes plus a save, always run together:

    prune_unreachable_nodes -> prune_dead_links -> prune_unreferenced_sidecars -> save

The order matters. Deleting a chat branch pops its nodes but leaves their attachments on disk, because
attachment storage is content-addressed and shared: the same image referenced from two branches is one file,
so no single node's deletion can decide the file's fate. Only a mark-and-sweep over every surviving payload
can, and it must run *after* the node prune, or attachments held solely by doomed nodes still look live.

**The preview** (`preview_cleanup`) exists because the sweep is the one destructive step a user cannot undo
and cannot inspect beforehand: an orphaned sidecar is named by its content hash, so the folder it lives in is
unreadable by eye. `rescue_to_staging` is the escape hatch — copy, not move, so the sidecar stays until the
user commits and changing their mind again costs nothing.

Names come from the description file written beside each sidecar at store time (`get_sidecar_metadata`), which
outlives the node that referenced it. A sidecar stored before descriptions existed falls back to its hash.
"""

__all__ = ["SidecarEntry",
           "CleanupPreview",
           "describe_sidecar",
           "preview_cleanup",
           "commit_cleanup",
           "rescue_to_staging",
           "format_size"]

import logging
logger = logging.getLogger(__name__)

import dataclasses
import pathlib
from typing import Any, Optional, Union

from unpythonic.env import env

from ..common import utils as common_utils

from . import chattree
from . import sidecarstore
from . import config as librarian_config


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
        size_bytes = datastore.sidecar_size(filename)
    except (OSError, KeyError):  # vanished under us, or unreadable; report it as present-but-unknown rather than failing
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
    # By bytes rather than by path: the sidecar may be held in memory, and the comparison below reads the
    # whole of both files anyway, so nothing is spent that was not already being spent.
    source_bytes = datastore.read_sidecar(entry.archival_filename)

    stem = pathlib.Path(entry.display_name).stem or entry.filename
    suffix = pathlib.Path(entry.display_name).suffix or pathlib.Path(entry.archival_filename).suffix
    candidate = directory / f"{stem}{suffix}"
    counter = 2
    while candidate.exists():
        if candidate.stat().st_size == len(source_bytes) and candidate.read_bytes() == source_bytes:
            logger.info(f"rescue_to_staging: '{entry.filename}' is already staged at '{candidate}'.")
            return candidate
        candidate = directory / f"{stem} ({counter}){suffix}"
        counter += 1

    candidate.write_bytes(source_bytes)
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
