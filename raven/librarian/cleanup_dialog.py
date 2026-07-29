"""The Librarian's "Clean up & save" dialog: shows what a cleanup would delete, and lets the user act on it.

The GUI half of datastore maintenance. The operation it drives — the prunes, the dry run, the rescue-to-staging
— is in `cleanup`, which has no DPG dependency and is unit-tested on its own; this module is the part that
needs a render loop, and is verified by hand.

The split is load-bearing rather than tidy-minded: the test environment installs no GUI toolkit (headless, on
three operating systems), so a `dearpygui` import anywhere in the operation's import graph makes the pure
half uncollectable. Keep it that way — presentation logic that does not touch `dpg` (name shortening, size
formatting) belongs in `cleanup`, not here.
"""

__all__ = ["DPGCleanupDialog"]

import logging
logger = logging.getLogger(__name__)

import concurrent.futures
import urllib.parse
from typing import Callable, Optional, Union

import dearpygui.dearpygui as dpg

from unpythonic.env import env

from ..common import bgtask
from ..common import utils as common_utils
from ..common.gui import utils as guiutils

from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa

from . import chattree
from . import cleanup
from . import config as librarian_config

gui_config = librarian_config.gui_config  # shorthand, as in `chat_controller`


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
        self.preview = cleanup.preview_cleanup(self.datastore, *self.get_roots())
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
            parts.append(cleanup._plural(len(preview.node_ids), "unreachable chat node"))
        if preview.sidecars:
            parts.append(f"{cleanup._plural(len(preview.sidecars), 'unreferenced attachment')} "
                         f"({cleanup.format_size(preview.total_bytes)})")
        return "Would delete " + " and ".join(parts) + "."

    def _build_image_section(self) -> None:
        """The image grid: a thumbnail per orphaned image, with its name and a rescue button.

        The cells are built empty and filled in later by `_load_thumbnails_task` — decoding is far too slow to
        do while the user waits for the dialog to appear. The section is collapsed by default, so in the common
        case the thumbnails are ready before anyone looks at them.
        """
        images = self.preview.images
        size = cleanup.format_size(sum(entry.size_bytes for entry in images))
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
                                dpg.add_text(cleanup._ellipsize(entry.display_name, 22))
                                with dpg.group(horizontal=True):
                                    dpg.add_text(cleanup.format_size(entry.total_bytes), color=(140, 140, 140))
                                    self._add_open_button(entry)
                                    self._add_rescue_button(entry)

    def _build_document_section(self) -> None:
        """The document list: one named row per orphaned document, with its size and a rescue button.

        A list rather than a grid because a document has no thumbnail — a file-type icon in a picture grid
        would be a tile pretending to be a picture. What identifies a document is its name.
        """
        documents = self.preview.documents
        size = cleanup.format_size(sum(entry.size_bytes for entry in documents))
        with dpg.collapsing_header(label=f"Documents ({len(documents)}, {size})", default_open=False):
            for entry in documents:
                with dpg.group(horizontal=True):
                    self._add_open_button(entry)
                    self._add_rescue_button(entry)
                    icon_id = dpg.add_text(fa.ICON_FILE_LINES)
                    dpg.bind_item_font(icon_id, self.themes_and_fonts.icon_font_solid)
                    dpg.add_text(entry.display_name)
                    dpg.add_text(f"({cleanup.format_size(entry.total_bytes)})", color=(140, 140, 140))

    def _add_open_button(self, entry: cleanup.SidecarEntry) -> None:
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

    def _open_one(self, entry: cleanup.SidecarEntry) -> None:
        """Open one attachment's stored copy in the OS default application. Logs a failure rather than raising."""
        try:
            common_utils.open_file(self.datastore.sidecar_path(entry.archival_filename))
        except Exception as exc:  # noqa: BLE001 -- a failed open must leave the dialog usable
            logger.error(f"DPGCleanupDialog._open_one: could not open '{entry.archival_filename}': "
                         f"{type(exc)}: {exc}")

    def _add_rescue_button(self, entry: cleanup.SidecarEntry) -> None:
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

    def _rescue_one(self, entry: cleanup.SidecarEntry) -> None:
        """Copy one attachment to staging and mark its button done. Reports failure on the button, not by raising."""
        try:
            staged_path = cleanup.rescue_to_staging(self.datastore, entry)
        except Exception as exc:  # noqa: BLE001 -- a failed rescue must leave the dialog usable for the rest
            logger.error(f"DPGCleanupDialog._rescue_one: could not stage '{entry.filename}': {type(exc)}: {exc}")
            self._mark_rescued(entry, message=f"Could not save '{entry.display_name}'\n{type(exc).__name__}: {exc}",
                               ok=False)
        else:
            self._mark_rescued(entry, message=f"Saved to\n{staged_path}", ok=True)

    def _mark_rescued(self, entry: cleanup.SidecarEntry, *, message: str, ok: bool) -> None:
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
            result = cleanup.commit_cleanup(self.datastore, *self.get_roots())
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

    def _thumbnail_tooltip_text(self, entry: cleanup.SidecarEntry) -> str:
        """Everything known about one orphaned image, for its tooltip: name, size, when it was stored, where from.

        Fuller than the grid cell can show, and the point of the exercise — recognizing a forgotten attachment
        takes the picture *plus* the story of where it came from.
        """
        lines = [entry.display_name, cleanup.format_size(entry.total_bytes)]
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
