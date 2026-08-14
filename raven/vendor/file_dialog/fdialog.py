# file_dialog 3.1
# MIT licensed

__all__ = ["FileDialog"]

import logging
logger = logging.getLogger(__name__)

import glob
import os
import platform
import psutil
import textwrap
import threading
import time
from typing import Iterable, Optional, Union

import dearpygui.dearpygui as dpg

from unpythonic import timer

from ...common import filelisting
from ...common import utils as common_utils
from ...common.gui import animation as gui_animation
from ...common.gui import utils as guiutils


# The icon assets, by name. `ico_<name>` holds the loaded pixels and `img_<name>` the texture, both set on
# the class; the grid view resamples the former to tile size and the table draws the latter.
_ICON_NAMES = [
    "document", "home", "add_folder", "add_file", "mini_folder", "folder", "mini_document",
    "mini_error", "refresh", "hard_disk", "picture", "big_picture", "picture_folder",
    "desktop", "videos", "music_folder", "downloads", "document_folder", "search", "back",
    "c", "gears", "music_note", "note", "object", "python", "script", "video", "link",
    "url", "vector", "zip", "app", "iso"
]

# Extensions this dialog can show a *picture* of rather than an icon for — everything
# `raven.common.image.codec` decodes. Also what makes a file type filter "image-typed", which is what turns
# the grid view on by itself.
_DECODABLE_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff",
                               ".avif", ".qoi")

# Extension -> icon name, shared by both views: the table draws the 16px asset of that name, the grid the
# tile-sized one. Module level rather than per instance, since it is twenty tuples that never vary.
_EXTENSION_ICONS = {
    # Binary blobs: shared libraries, and the model-weight formats, which are the same kind of thing to a
    # file picker — something opaque that a program loads.
    (".dll", ".a", ".o", ".so", ".ko",
     ".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".onnx"): "gears",
    _DECODABLE_IMAGE_EXTENSIONS: "picture",
    (".msi", ".exe", ".bat", ".bin", ".elf", ".appimage", ".desktop"): "app",
    (".iso",): "iso",
    (".zip", ".deb", ".rpm", ".tar.gz", ".tgz", ".tar", ".gz", ".xz", ".bz2", ".zst",
     ".lzo", ".lz4", ".7z", ".rar", ".whl", ".ppack"): "zip",
    (".py", ".pyo", ".pyw", ".pyi", ".pyc", ".pyz", ".pyd", ".pyx", ".pxd"): "python",
    (".c",): "c",
    (".js", ".json", ".cs", ".cpp", ".h", ".hpp", ".sh", ".pyl", ".rs", ".vbs", ".cmd",
     ".ts", ".go", ".rb", ".lua", ".jl", ".java", ".yaml", ".yml", ".toml", ".ini", ".cfg",
     ".xml", ".html", ".htm", ".css"): "script",
    (".url",): "url",
    (".lnk",): "link",
    # Prose, in whatever container. `.bib` and `.tex` earn their place because this dialog is how a
    # bibliography or a paper source gets picked; `.docx` / `.odt` / `.org` because Raven's document
    # database reads them (`llm_docs_exts`), so they turn up here as things to open.
    #
    # Deliberately absent: `.pdf`, `.pptx`, `.odp`. There is no icon for a presentation, and the fallback —
    # the generic document — is already the right picture for all three.
    (".txt", ".md", ".rst", ".org", ".bib", ".tex", ".docx", ".odt",
     ".log", ".csv", ".tsv"): "note",
    (".mp3", ".ogg", ".wav", ".flac", ".m4a", ".opus", ".aac"): "music_note",
    (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".wmv"): "video",
    (".obj", ".fbx", ".blend"): "object",
    (".svg",): "vector",
}

# How often the grid's own tick runs while it is on screen. `FileDialog` is a widget inside apps that own
# their render loops, so requiring every host app to call something would be a landmine — the app that
# forgets is the one whose thumbnails never appear.
_GRID_TICK_INTERVAL = 1.0 / 60


def _normalize_filter(entry: Union[str, tuple[str, Iterable[str]]]) -> tuple[str, Optional[tuple[str, ...]]]:
    """Normalize one `FileDialog` `filter_list` entry to a `(label, extensions)` pair.

    `extensions` is a tuple of lowercase suffixes, or `None` for the ".*" catch-all. A bare string is its
    own label and matches that one suffix, which is the original single-extension form.
    """
    if isinstance(entry, str):
        if entry == ".*":
            return (entry, None)
        return (entry, (entry.lower(),))
    label, extensions = entry
    return (label, tuple(sorted({ext.lower() for ext in extensions})))


# Hotkey support
visible_dialog_instance = None  # fdialog is modal so There Can Be Only One (TM). If needed, could use a list, and check which one has keyboard focus, but that might not always work.
def fdialog_hotkeys_callback(sender, app_data):
    if visible_dialog_instance is None:
        return

    key = app_data  # for documentation only
    # shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
    ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)

    # TODO: Add hotkeys to navigate up/down in the table, descend into folder, ...
    if key == dpg.mvKey_Return:
        visible_dialog_instance.ok()
    elif key == dpg.mvKey_Escape:
        visible_dialog_instance.cancel()
    elif key == dpg.mvKey_F5:
        visible_dialog_instance.refresh()
    elif ctrl_pressed and key == dpg.mvKey_Home:
        visible_dialog_instance.back_to_default_path()
    elif ctrl_pressed and key == dpg.mvKey_F:
        dpg.focus_item(visible_dialog_instance.search_field)


class FileDialog:
    _class_init_lock = threading.Lock()  # thread-safe asset loading
    _class_initialized = False

    @classmethod
    def _initialize_class(cls):
        with cls._class_init_lock:
            # Everything cached here — icon textures, themes, the hotkey handler registry — belongs to the DPG
            # context that created it, and `dpg.destroy_context` takes it along while leaving this flag set. An
            # app holds one context for its whole life and so never meets this; a test suite that builds a
            # second one gets `SystemError: Texture not found` out of the first `add_image` in the constructor.
            # So ask the context whether the cached items are still there, rather than trusting the flag alone.
            if cls._class_initialized and dpg.does_item_exist("ico_home"):  # tag
                return
            cls._class_initialized = True

            # register our hotkey handler
            with dpg.handler_registry(tag="fdialog_handler_registry"):  # global (whole viewport)
                dpg.add_key_press_handler(tag="fdialog_hotkeys_handler", callback=fdialog_hotkeys_callback)

            cls.fd_img_path = os.path.join(os.path.dirname(__file__), "images")

            # file dialog theme
            with dpg.theme() as cls.selec_alignt:
                with dpg.theme_component(dpg.mvThemeCat_Core):
                    dpg.add_theme_style(dpg.mvStyleVar_SelectableTextAlign, x=0, y=.5)

            with dpg.theme() as cls.size_alignt:
                with dpg.theme_component(dpg.mvThemeCat_Core):
                    dpg.add_theme_style(dpg.mvStyleVar_SelectableTextAlign, x=1, y=.5)

            # texture loading
            for img in _ICON_NAMES:
                width, height, _, data = dpg.load_image(os.path.join(cls.fd_img_path, f"{img}.png"))
                setattr(cls, f"ico_{img}", [width, height, data])

            with dpg.texture_registry():
                for img in _ICON_NAMES:
                    width, height, data = getattr(cls, f"ico_{img}")
                    dpg.add_static_texture(width=width, height=height, default_value=data, tag=f"ico_{img}")
                    setattr(cls, f"img_{img}", f"ico_{img}")

    def __init__(
        self,
        title="File dialog",
        tag="file_dialog",
        width=1150,
        height=650,
        min_size=(460, 320),
        dirs_only=False,
        save_mode=False,
        default_file_extension=None,
        default_path=os.getcwd(),
        filter_list=[".*", ".exe", ".bat", ".sh", ".msi", ".apk", ".bin", ".cmd", ".com", ".jar", ".out", ".py", ".pyl", ".phs", ".js", ".json", ".java", ".c", ".cpp", ".cs", ".h", ".rs", ".vbs", ".php", ".pl", ".rb", ".go", ".swift", ".ts", ".asm", ".lua", ".sh", ".bat", ".r", ".dart", ".ps1", ".html", ".htm", ".xml", ".css", ".ini", ".yaml", ".yml", ".config", ".md", ".rst", ".txt", ".rtf", ".doc", ".docx", ".pdf", ".odt", ".tex", ".log", ".csv", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".svg", ".webp", ".ico", ".psd", ".ai", ".eps", ".tga", ".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".aiff", ".mid", ".midi", ".opus", ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".mpeg", ".mpg", ".3gp", ".m4v", ".blend", ".fbx", ".obj", ".stl", ".3ds", ".dae", ".ply", ".glb", ".gltf", ".csv", ".sql", ".db", ".dbf", ".mdb", ".accdb", ".sqlite", ".xml", ".json", ".zip", ".rar", ".7z", ".tar", ".gz", ".iso", ".bz2", ".xz", ".tgz", ".cab", ".vdi", ".vmdk", ".vhd", ".vhdx", ".ova", ".ovf", ".qcow2", ".dockerfile", ".bak", ".old", ".sav", ".tmp", ".bk", ".ppack", ".mlt", ".torrent", ".ics"],
        file_filter=None,
        callback=None,
        show_dir_size=False,
        allow_drag=False,
        multi_selection=False,
        show_shortcuts_menu=True,
        no_resize=True,
        modal=True,
        show_hidden_files=False,
        show_thumbnails=None,
        thumbnail_size=128,
        thumbnail_device="gpu",
        user_style=0
    ):
        """
        Arguments:
            title:                  str, File dialog window title.
            tag:                    str, File dialog window DPG tag.
            width:                  int, File dialog window width (pixels).
            height:                 int, File dialog window height (pixels).
            min_size:               (int, int), File dialog minimum size.
            dirs_only:              When True, only directories will be listed.
            save_mode:              When True, asks for a filename to save as, instead of selecting file(s) to open.
                                    In the GUI, the "Search files" field becomes the filename field. (Searching is still enabled, to help avoid accidental overwriting.)
            default_file_extension: Only used when save_mode is True. The extension (e.g. ".png") automatically added
                                    when the user names a "save as" file without one.

                                    `None` (the default) derives it from the selected file type filter, when that
                                    filter names exactly one extension — which is the usual shape of a save dialog,
                                    and saves every such caller from repeating the extension a third time. A filter
                                    naming several (or none) leaves this at "add nothing", there being no principled
                                    choice among them; say which one explicitly if you want one.

                                    Pass "" to add nothing regardless.
            default_path:           str, The default path when file_dialog starts, if it's the string 'cwd', the default path will be the current working directory.
            filter_list:            The items offered in the file type filter. Each item is either:

                                        - `str`: a single file extension, which is also its own label, e.g. ".png".
                                          The special value ".*" matches every file.
                                        - `(label, extensions)`: a label plus any number of extensions it matches,
                                          e.g. ("Images", [".png", ".jpg", ".webp"]). The listing shows only the
                                          label; a tooltip on the filter spells out the extensions.

                                    The pair form exists because a useful extension set is often far too long to
                                    read as a label — "every image Pillow can open" is 67 extensions — so the
                                    label has to be written by hand rather than derived.

                                    Matching is by case-insensitive suffix, so an entry like ".tar.gz" works, and
                                    ".png" matches "PHOTO.PNG".
            file_filter:            str, The file type filter selected when the dialog is opened. This is an item's
                                    *label*, e.g. ".py" for a bare-string item or "Images" for a pair.
                                    `None` (the default) selects the first item of `filter_list`, which is what a
                                    caller listing the types it wants almost always means.
            callback:               callable, When the OK or Cancel button is pressed, the file dialog will call this, sending the list of selected files. Upon cancel, the list is empty.

                                    The argument is a `list` of `str`, each an absolute path. (Not `pathlib.Path`.)

                                    Consume it before returning; do not retain it. The list handed to the callback is the
                                    dialog's own `selected_files`, which is cleared as soon as the callback returns, so a
                                    reference stashed for later reads as empty. Copy it if you need to keep it.
            show_dir_size:          If True, directories will be listed with the size of the directory and its sub-directories and files. Not recommended.
            allow_drag:             If True, the files and folders in the dialog act as a DPG drag source, so you can set up a drop target to accept them as drag'n'drops in your app. See source code for details.
                                    Off by default: it costs a drag payload and an icon widget per row, which is
                                    worth paying only where something is actually listening for the drop.
            multi_selection:        If True, the user can select multiple files and folders by holding down Ctrl and clicking. If False, only one file/folder can be selected, and Ctrl does nothing.
                                    Ignored when save_mode is True.
                                    Off by default, so that a picker multi-selects only where the caller asked for it.
            show_shortcuts_menu:    if True, show a child window (side panel) containing different shortcuts (like desktop and downloads), and the external and internal drives.
            no_resize:              If True, the window will not be resizable.
            modal:                  If True, use DPG modal mode; a sort of popup effect. Can cause problems if the file dialog is opened by a modal window.
            show_hidden_files:      If True, the dialog shows also hidden files and folders.
            show_thumbnails:        Whether to open in the thumbnail grid view instead of the table.

                                    `None` (the default) decides per file type filter: the grid comes up
                                    when the selected filter names image formats and nothing else, which is
                                    when picking by name is close to useless — generated and photographed
                                    images have hashes and timestamps for filenames.

                                    The checkbox in the dialog overrides this in either direction, **for as
                                    long as that opening lasts**. The next opening resets to this argument,
                                    so an app that asked for a view gets it every time, and a `None` lets
                                    the automatic rule decide afresh. The override cannot outlive an opening
                                    because the dialog cannot: one instance serves the whole app run, so a
                                    stickier override would be a one-way door out of the automatic mode.

                                    Both views list the same entries, directories included, and share one
                                    sort order and one cursor; switching between them changes nothing else.
            thumbnail_size:         int, edge of a grid tile in pixels. Larger tiles show more of each
                                    image and fewer of them.
            thumbnail_device:       str, where thumbnails are decoded and resized. The literal "gpu" (the
                                    default) is `raven.common.deviceinfo`'s autodetect: whichever GPU
                                    backend this machine has, or CPU when it has none. Name one explicitly
                                    ("cuda:0", "cpu") to pin thumbnails to a particular device, or to keep
                                    them off one already busy with inference.
            user_style:             int, different graphical styles for file_dialog. Currently available values: 0 (full), 1 (compact).
        Returns:
            None
        """

        # args
        self.title = title
        self.tag = tag
        self.width = width
        self.height = height
        self.min_size = min_size
        self.dirs_only = dirs_only
        self.save_mode = save_mode
        self.default_file_extension = default_file_extension
        self.default_path = os.getcwd() if default_path == "cwd" else default_path
        self.filter_list = filter_list
        self.file_filter = file_filter
        self.callback = callback
        self.show_dir_size = show_dir_size
        self.allow_drag = allow_drag
        self.multi_selection = (not save_mode) and multi_selection
        self.show_shortcuts_menu = show_shortcuts_menu
        self.no_resize = no_resize
        self.modal = modal
        self.show_hidden_files = show_hidden_files
        self.thumbnail_size = thumbnail_size
        self.thumbnail_device = thumbnail_device
        self.user_style = user_style

        # Grid view. Built on first use — see `_the_grid` — so a dialog only ever used as a list never
        # loads the thumbnail decoder.
        self._grid = None
        self._show_thumbnails_default = show_thumbnails  # what each opening resets to; `None` = decide automatically
        self._grid_mode = bool(show_thumbnails) and not dirs_only
        self._grid_mode_chosen_by_user = (show_thumbnails is not None)
        self._grid_size = (400, 300)  # replaced by a measurement as soon as the dialog has rendered
        self._ticker = None
        self._ticker_stop = threading.Event()

        self.instance_tag = f"0x{id(self):x}"  # for making unique DPG tags
        self.last_path = default_path  # for returning to last used directory when the dialog is closed and later re-opened

        self.PAYLOAD_TYPE = 'ws_' + self.tag
        self.selected_files = []
        self.shown_items = []  # for selection by search filter upon pressing ok
        self.selec_height = 16
        # The listing's order, held as data rather than as the table's row order. A rebuild reproduces it,
        # which is what lets the listing be re-rendered — re-filtered, or shown a different way — without
        # the sort having to be recovered from the widgets it produced.
        self._sort_key = filelisting.SortKey.NAME
        self._sort_descending = False
        self.image_transparency = 100
        self.last_click_time = 0
        self.last_ok_time = 0
        self.double_click_threshold = 0.25  # seconds; adjust the time as needed.  # TODO: should really get this from OS if possible in a cross-platform way

        self._initialize_class()

        # File type filter.
        def _set_type_filter(label: str) -> None:
            self.file_filter = label
            if label in self._filter_extensions:
                self._active_extensions = self._filter_extensions[label]
            else:  # not one of the offered items; read it as a literal extension, as the single-extension form did
                self._active_extensions = None if label == ".*" else (label.lower(),)

        # Whether the caller named a save extension. If not, it is derived from the filter, and so has to be
        # re-derived whenever the offered filters change.
        self._default_file_extension_was_given = (default_file_extension is not None)

        def _install_filters(filter_list, file_filter=None) -> None:
            """Recompute the offered file type filters. Touches no widgets; callers refresh the GUI."""
            self.filter_list = list(filter_list)
            self._filters = [_normalize_filter(entry) for entry in self.filter_list]
            self._filter_labels = [label for label, _extensions in self._filters]
            self._filter_extensions = dict(self._filters)
            # Every extension any filter knows of. Save mode uses this to decide whether a typed filename already
            # carries an extension, so it accepts any offered one rather than only the active filter's.
            self._all_extensions = tuple(sorted({ext
                                                 for _label, extensions in self._filters if extensions is not None
                                                 for ext in extensions}))
            # `None` means "the first item", which is what a caller listing the types it wants almost always means.
            _set_type_filter(self._filter_labels[0] if file_filter is None else file_filter)

            # A save dialog's default extension is nearly always the one extension its filter names, so derive
            # it rather than making the caller write it a third time. Only for a filter naming exactly one:
            # among several there is no principled choice, and silently picking the first would be a rule
            # nobody could predict from a call site.
            if not self._default_file_extension_was_given:
                if self._active_extensions is not None and len(self._active_extensions) == 1:
                    self.default_file_extension = self._active_extensions[0]
                else:
                    self.default_file_extension = None
        _install_filters(self.filter_list, self.file_filter)

        def _matches_type_filter(file_name: str) -> bool:
            if self._active_extensions is None:  # ".*"
                return True
            file_name = file_name.lower()
            return any(file_name.endswith(ext) for ext in self._active_extensions)

        def _describe_type_filter(label: str) -> str:
            extensions = self._filter_extensions.get(label)
            if extensions is None:
                return "Every file, whatever its extension."
            return textwrap.fill(" ".join(extensions), width=72,
                                 initial_indent="Matches: ", subsequent_indent="         ")

        # low-level functions
        def _get_all_drives():
            """Mount points to offer in the shortcuts panel, one menu item each.

            Mount points, specifically: every entry has to be somewhere `chdir` can go, because that is the
            only thing clicking one does.

            A POSIX-only branch here used to also scan /dev for names starting with "sd" or "nvme" and append
            the raw device paths. It was skipped on Windows (`os.name == 'posix'`), which is why the panel
            looked right there and only there. On this machine it added four entries — the two partitions
            already listed by their mount points, plus the whole disk and the controller — and a block device
            is not a directory, so each could only raise `NotADirectoryError` into the message box. Its
            dedup test could never fire either: it compared /dev paths against a list of mount points.
            """
            return [drive.mountpoint for drive in psutil.disk_partitions() if drive.mountpoint]

        def delete_table():
            for child in dpg.get_item_children(f"explorer_{self.instance_tag}", 1):
                dpg.delete_item(child)

        def on_path_enter():
            try:
                chdir(dpg.get_value(f"ex_path_input_{self.instance_tag}"))
            except FileNotFoundError:
                message_box("Invalid path", "No such file or directory")

        def message_box(title, message):
            if not self.modal:
                with dpg.mutex():
                    viewport_width = dpg.get_viewport_client_width()
                    viewport_height = dpg.get_viewport_client_height()
                    with dpg.window(label=title, no_close=True, modal=True) as modal_id:
                        dpg.add_text(message)
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Ok", width=-1, user_data=(modal_id, True), callback=lambda: dpg.delete_item(modal_id))

                # Waited for so the box can be centered on geometry that exists. Not required: an
                # off-center message box beats a hang with no traceback, which is what a bare
                # `dpg.split_frame` gives when there is no render loop to wait for.
                guiutils.split_frame(operation="file dialog: centering a message box", required=False)
                width = dpg.get_item_width(modal_id)
                height = dpg.get_item_height(modal_id)
                dpg.set_item_pos(modal_id, [viewport_width // 2 - width // 2, viewport_height // 2 - height // 2])
            else:
                # TODO: We really need a message box that works while the file dialog is modal.
                logger.warning(f"message_box: Cannot display message box while file_dialog is in modal. Message follows:\n{title}:\t{message}\n")

        def open_drive(sender, app_data, user_data):
            chdir(user_data)

        def _deselect_recursive(root):
            """Deselect all selectables inside DPG widget `root`, including `root` itself."""
            if dpg.get_item_type(root) == "mvAppItemType::mvSelectable":
                dpg.set_value(root, False)
            for item in dpg.get_item_children(root, slot=1):
                _deselect_recursive(item)

        def open_file(sender, app_data, user_data):  # `user_data`: [name, fullpath, timestamp, size]
            ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)

            # Detect double-click.
            # double_clicked = dpg.is_mouse_button_double_clicked(dpg.mvMouseButton_Left)  # TODO: doesn't work, why?
            current_time = time.time()
            double_clicked = (current_time - self.last_click_time < self.double_click_threshold)
            self.last_click_time = current_time

            logger.debug(f"open_file: instance '{self.tag}' ({self.instance_tag}), sender is {sender} (tag '{dpg.get_item_alias(sender)}', type {dpg.get_item_type(sender)}, value = {dpg.get_value(sender)}), app_data = {app_data}, user_data = {user_data}, ctrl = {ctrl_pressed}, doubleclick = {double_clicked}")

            # Multi selection
            if self.multi_selection and ctrl_pressed:
                if dpg.get_value(sender) is True:
                    self.selected_files.append(user_data[1])
                elif user_data[1] in self.selected_files:
                    self.selected_files.remove(user_data[1])
            # Single selection
            else:
                dpg.set_value(sender, False)  # unselect this item  (TODO: why? double-click handling?)

                if double_clicked:
                    if user_data is not None and user_data[1] is not None:
                        if os.path.isdir(user_data[1]):
                            logger.debug(f"open_file: instance '{self.tag}' ({self.instance_tag}), Content: {dpg.get_item_label(sender)}, files: {user_data}")
                            chdir(user_data[1])
                            dpg.set_value(f"ex_search_{self.instance_tag}", "")
                        elif os.path.isfile(user_data[1]):
                            if len(self.selected_files) < 1:
                                self.selected_files.append(user_data[1])
                            self.ok()
                            return user_data[1]
                else:
                    if os.path.isfile(user_data[1]) or (self.dirs_only and os.path.isdir(user_data[1])):
                        _deselect_recursive(f"explorer_{self.instance_tag}")  # unselect others
                        dpg.set_value(sender, True)  # and select this item
                        # Save mode: populate file name field from clicked file, without file extension
                        if self.save_mode:
                            basename, ext = os.path.splitext(user_data[0])
                            dpg.set_value(f"ex_search_{self.instance_tag}", basename)
                            self._update_search()
                        self.selected_files.clear()
                        self.selected_files.append(user_data[1])

        def get_directory_path(directory_name):
            try:
                # Check for Linux or MacOS
                if platform.system() in ["Linux", "Darwin"] and directory_name.lower() == "home":
                    directory_path = os.path.expanduser("~")
                # Check for Windows
                elif platform.system() == "Windows" and directory_name.lower() == "home":
                    directory_path = os.path.expanduser("~")
                else:
                    # Attempt to join the home directory with the specified directory name
                    directory_path = os.path.join(os.path.expanduser("~"), directory_name)

                # Verify if the directory exists
                os.listdir(directory_path)  # Test access
            except FileNotFoundError:
                # Search for the directory in the user's home folder
                search_path = os.path.expanduser("~/*/" + directory_name)
                directory_path = glob.glob(search_path)
                if directory_path:
                    try:
                        os.listdir(directory_path[0])  # Test access to the found path
                        directory_path = directory_path[0]  # Use the found path
                    except FileNotFoundError:
                        message_box("File dialog - Error", "Could not find the selected directory")
                        return "."
                else:
                    message_box("File dialog - Error", "Could not find the selected directory")
                    return "."

            return directory_path

        def _icon_name_for_extension(file_name: str) -> Optional[str]:
            """Which icon `file_name`'s extension asks for, or `None` for a type with no icon of its own.

            Matched case-insensitively, so `PHOTO.JPG` gets the picture icon that `photo.jpg` does. The
            answer is an icon *name*, not a texture: the table draws it small and the grid draws it at tile
            size, so the two views share this table and pick their own assets from it.
            """
            file_name = file_name.lower()
            for extensions, icon_name in _EXTENSION_ICONS.items():
                if file_name.endswith(extensions):  # `str.endswith` takes a tuple
                    return icon_name
            return None

        def _icon_for(entry) -> Union[str, int]:
            """The small icon shown at the left of `entry`'s row."""
            if entry.is_dir:
                return self.img_mini_folder
            if entry.kind == filelisting.KIND_BROKEN_LINK:
                return self.img_mini_error
            icon_name = _icon_name_for_extension(entry.name)
            if icon_name is None:
                return self.img_mini_document
            return getattr(self, f"img_{icon_name}")

        def _tile_icon_for(entry) -> Optional[str]:
            """Which icon `entry`'s *tile* gets in grid view, or `None` to decode the image itself.

            `None` is what puts an entry in the thumbnail queue, so it is the answer for exactly the files
            worth looking at — which is the whole reason the grid view exists. Everything else gets a
            picture of its type, because a picker that shows only what it can preview is lying about the
            contents of the directory.
            """
            if entry.is_dir:
                return "folder"  # the large one; `mini_folder` is 16px and unusable at tile size
            if entry.kind == filelisting.KIND_BROKEN_LINK:
                return "mini_error"
            if entry.name.lower().endswith(_DECODABLE_IMAGE_EXTENSIONS):
                return None
            return _icon_name_for_extension(entry.name) or "document"

        def _make_row(entry, callback, parent=f"explorer_{self.instance_tag}", selected_paths=()):
            """Build one table row from a `filelisting.FileEntry`.

            The entry carries everything the row needs, so nothing here consults the filesystem and nothing
            has to be read back off the widgets later.

            `selected_paths`: which paths were selected before this rebuild. A row for one of them comes up
            already selected, so a selection survives a re-listing instead of quietly evaporating.
            """
            # `..` is the way out of the directory rather than something in it: one spanning cell, and no
            # date/type/size.
            if entry.is_parent:
                with dpg.table_row(parent=parent):
                    with dpg.group(horizontal=True):
                        dpg.add_image(self.img_mini_folder, tint_color=[255, 255, 255, 255], user_data=entry.kind)
                        dpg.add_selectable(label=entry.name, callback=_go_up_one_level,
                                           span_columns=True, height=self.selec_height)
                return

            # `user_data` shape is `open_file`'s contract: [name, full path, mtime, size].
            kwargs_cell = {'callback': callback, 'span_columns': True, 'height': self.selec_height,
                           'user_data': [entry.name, entry.path, entry.mtime, entry.size or 0]}
            alpha = self.image_transparency if entry.is_hidden else 255
            kwargs_image = {'tint_color': [255, 255, 255, alpha], 'user_data': entry.kind}

            with dpg.table_row(parent=parent):
                with dpg.group(horizontal=True):
                    dpg.add_image(_icon_for(entry), **kwargs_image)
                    cell_name = dpg.add_selectable(label=entry.name, **kwargs_cell)
                cell_time = dpg.add_selectable(label=filelisting.format_mtime(entry.mtime), **kwargs_cell)
                cell_type = dpg.add_selectable(label=filelisting.format_kind(entry), **kwargs_cell)
                cell_size = dpg.add_selectable(label=filelisting.format_size(entry.size), **kwargs_cell)

                # Restore the selection: only the name cell is set, since every cell spans the columns and
                # setting all four would stack the tint on itself.
                if entry.path in selected_paths and _is_choosable(entry):
                    dpg.set_value(cell_name, True)
                    self.selected_files.append(entry.path)

                if self.allow_drag:
                    drag_payload = dpg.add_drag_payload(parent=cell_name, payload_type=self.PAYLOAD_TYPE)
                dpg.bind_item_theme(cell_name, self.selec_alignt)
                dpg.bind_item_theme(cell_time, self.selec_alignt)
                dpg.bind_item_theme(cell_type, self.selec_alignt)
                dpg.bind_item_theme(cell_size, self.size_alignt)
                if self.allow_drag:
                    if entry.name.lower().endswith((".png", ".jpg")):
                        dpg.add_image(self.img_big_picture, parent=drag_payload)
                    elif entry.is_dir:
                        dpg.add_image(self.img_folder, parent=drag_payload)
                    else:
                        dpg.add_image(self.img_document, parent=drag_payload)

        def _go_up_one_level(sender, app_data, user_data):
            """GUI callback: if this item double-clicked, go up one level."""
            ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
            current_time = time.time()
            double_clicked = (current_time - self.last_click_time < self.double_click_threshold)
            self.last_click_time = current_time

            dpg.set_value(sender, False)  # unselect the ".." entry

            if ctrl_pressed:
                return
            if double_clicked:
                dpg.set_value(f"ex_search_{self.instance_tag}", "")
                chdir("..")

        def set_type_filter(label):
            """Select the file type filter by its label, exactly as picking it from the combo would.

            `label` is one of the labels derived from `filter_list` — a bare extension for a string entry,
            or the given label for a `(label, extensions)` pair.
            """
            _set_type_filter(label)
            dpg.set_value(self.combo_file_filter, self.file_filter)  # keep the GUI in sync when called programmatically
            dpg.set_value(self.text_file_filter_extensions, _describe_type_filter(self.file_filter))
            _apply_automatic_grid_mode(rebuild=False)  # the listing is about to be rebuilt anyway
            reset_dir(default_path=os.getcwd())
        self.set_type_filter = set_type_filter  # needs to be accessible from the outside; uses closure data from this scope, so shouldn't be injected as an instance method (on the class); inject as a regular function *on the instance*.

        def set_filter_list(filter_list, file_filter=None):
            """Replace the offered file type filters, as `filter_list` in the constructor.

            For an app whose acceptable types depend on state that can change while it runs — a Librarian
            that offers image formats only while a vision model is loaded. Call it before
            `show_file_dialog`, so what is offered reflects the answer at the moment of opening rather than
            at construction.

            `file_filter` selects one of the new labels; `None` takes the first, as in the constructor.

            The listing is rebuilt only if the dialog is already open. Called just before `show_file_dialog`,
            which is the intended use, rebuilding here would be work thrown away: that call rebuilds anyway,
            and on a directory of thousands each rebuild is the couple of seconds the dialog is already slow by.
            """
            _install_filters(filter_list, file_filter)
            dpg.configure_item(self.combo_file_filter, items=self._filter_labels)
            dpg.set_value(self.combo_file_filter, self.file_filter)
            dpg.set_value(self.text_file_filter_extensions, _describe_type_filter(self.file_filter))
            _apply_automatic_grid_mode(rebuild=False)
            # The *configured* show flag, not `is_visible`: the latter answers "did the user see it in the last
            # rendered frame", which is False for a window shown microseconds ago and False always with no
            # render loop. The question here is whether a listing exists to be brought up to date.
            if dpg.get_item_configuration(self.tag)["show"]:  # tag
                reset_dir(default_path=os.getcwd())
        self.set_filter_list = set_filter_list  # instance-injected for the same reason as `set_type_filter` above.

        def filter_combo_selector(sender, app_data):
            set_type_filter(dpg.get_value(sender))

        def chdir(path):
            try:
                os.chdir(path)
                cwd = os.getcwd()
                reset_dir(default_path=cwd)
            except PermissionError as e:
                message_box("File dialog - PerimssionError", f"Cannot open the folder because is a system folder or the access is denied\n\nMore info:\n{e}")
            except NotADirectoryError as e:
                message_box("File dialog - not a directory", f"The selected item is not a directory, but a file.\n\nMore info:\n{e}")
        self.chdir = chdir  # needs to be accessible from the outside; uses closure data from this scope, so shouldn't be injected as an instance method (on the class); inject as a regular function *on the instance*.

        def reset_dir(file_name_filter=None, default_path=self.default_path):
            logger.debug(f"reset_dir: instance '{self.tag}' ({self.instance_tag}), called with file_name_filter = {file_name_filter}, default_path = '{str(default_path)}'")
            # Phase timings, so a slow open says *which* phase is slow rather than only that it was. Reading
            # the directory, deleting the old rows and creating the new ones have entirely different fixes,
            # and a report of "a couple of seconds" does not distinguish them.
            # What was selected, so it can be restored against the new listing. A rebuild happens on every
            # keystroke in the find field and on every view switch, and until 2026-08-14 each of those
            # silently dropped the selection: the *cursor* was re-anchored by path and the selection was
            # not, so switching to the grid and back left the file chosen but no longer shown as chosen.
            previously_selected = set(self.selected_files)
            self.selected_files.clear()
            self.shown_items.clear()
            try:
                dpg.configure_item(f"ex_path_input_{self.instance_tag}", default_value=os.getcwd())
                # Compiled once per rebuild rather than per entry: on a directory of thousands, the split is
                # the part worth hoisting out of the loop.
                matches_name_filter = common_utils.make_search_matcher(file_name_filter or "")

                # Enumerating, filtering and sorting all happen here, on data, before a widget is touched.
                with timer() as tim_list:
                    entries = filelisting.list_directory(default_path,
                                                         show_hidden=self.show_hidden_files,
                                                         dirs_only=self.dirs_only,
                                                         name_filter=matches_name_filter,
                                                         type_filter=_matches_type_filter,
                                                         sort_key=self._sort_key,
                                                         descending=self._sort_descending)
                # `..` stays out: it is the way out of the directory rather than a candidate for the
                # unique-match shortcut in `ok`.
                self.shown_items.extend(entry.path for entry in entries if not entry.is_parent)

                # Only the view on screen is built. The other one is emptied rather than left holding a
                # stale listing, which would be both the memory and, on a switch back, the wrong answer.
                with timer() as tim_delete:
                    delete_table()

                with timer() as tim_build:
                    if self._grid_mode:
                        grid = _the_grid()
                        grid.set_listing(entries)
                        # Re-made against the new order, and the grid's own callback puts the survivors
                        # back into `selected_files`.
                        grid.set_selected_paths(previously_selected)
                    else:
                        for entry in entries:
                            _make_row(entry, open_file, selected_paths=previously_selected)

                logger.debug(f"reset_dir: instance '{self.tag}' ({self.instance_tag}), {len(self.shown_items)} entries "
                             f"as {'tiles' if self._grid_mode else 'rows'}: "
                             f"list {tim_list.dt:.3f}s, delete {tim_delete.dt:.3f}s, build {tim_build.dt:.3f}s")

            # exceptions
            except FileNotFoundError:
                logger.error(f"reset_dir: instance '{self.tag}' ({self.instance_tag}), invalid path: '{str(default_path)}'")
            except Exception as e:
                message_box("File dialog - Error", f"An unknown error has occured when listing the items, More info:\n{e}")
        self.reset_dir = reset_dir  # needs to be accessible from the outside; uses closure data from this scope, so shouldn't be injected as an instance method (on the class); inject as a regular function *on the instance*.

        # --------------------------------------------------------------------------------
        # Sorting.
        #
        # One row of buttons above the listing, serving both views, with the table header's own semantics:
        # click to sort ascending, click again for descending.
        #
        # Not "buttons for the grid, header clicks for the table", which was the first shape and is wrong.
        # ImGui draws the header's sort arrow from its *own* state, so a sort chosen in grid view would
        # leave the header asserting an order the data no longer has. Turning the header's sorting off
        # removes the second source of truth by construction rather than by keeping two things in step.
        #
        # It costs the familiar click-the-header gesture and buys two things beyond that guarantee: sorting
        # becomes keyboard-operable, which ImGui's header sorting is not at all, and the control does not
        # move when the view does.
        #
        # The header itself stays, because `resizable` is a header-drag gesture and filename lengths vary
        # enormously between users and directories — which is exactly when a fixed Name column hurts.
        _SORT_CRITERIA = [(filelisting.SortKey.NAME, "Name"),
                          (filelisting.SortKey.DATE, "Date"),
                          (filelisting.SortKey.KIND, "Type"),
                          (filelisting.SortKey.SIZE, "Size")]
        _sort_indicators = {}  # SortKey -> drawlist tag

        def _draw_sort_indicators():
            """Redraw the triangle marking which criterion is active, and which way it points.

            Drawn rather than written: Raven's UI font is OpenSans, which has no triangle or arrow glyphs
            at all, so a text label would render a missing-glyph box. Ten lines of drawlist is cheaper than
            binding the icon font to a button, which would apply to its whole label.
            """
            for sort_key, drawlist in _sort_indicators.items():
                dpg.delete_item(drawlist, children_only=True)
                if sort_key is not self._sort_key:
                    continue
                color = (210, 210, 210, 255)
                if self._sort_descending:
                    points = [(2, 10), (12, 10), (7, 18)]
                else:
                    points = [(2, 18), (12, 18), (7, 10)]
                dpg.draw_triangle(*points, color=color, fill=color, parent=drawlist)

        def sort_by(sort_key, descending=None):
            """Order the listing by `sort_key`, and rebuild it.

            `descending`: `None` (the default) is the click semantics — asking for the criterion already in
            force reverses it, and any other criterion starts ascending. Pass a bool to say which way
            outright, which is what restoring a remembered order wants.
            """
            if descending is None:
                descending = (not self._sort_descending) if sort_key is self._sort_key else False
            self._sort_key = sort_key
            self._sort_descending = descending
            _draw_sort_indicators()
            self._update_search()  # re-lists the current directory under the current find query
        self.sort_by = sort_by  # instance-injected for the same reason as `set_type_filter` above.

        def _make_sort_row():
            """The sort buttons, plus the view toggle, on one row above the listing.

            The toggle sits next to them rather than off at the right edge, because the case it exists for
            is a filter that selects images *and* something else — Librarian's "Documents and images", say —
            where the automatic rule deliberately does not fire and the user has to find this. Next to the
            controls they are already using is where they will.
            """
            with dpg.group(horizontal=True):
                dpg.add_text("Sort by")
                for sort_key, label in _SORT_CRITERIA:
                    with dpg.group(horizontal=True):
                        dpg.add_button(label=label, width=70,
                                       user_data=sort_key,
                                       callback=lambda s, a, u: sort_by(u))
                        drawlist = dpg.add_drawlist(width=14, height=self.selec_height + 10)
                        _sort_indicators[sort_key] = drawlist
                self.spacer_view_toggle = dpg.add_spacer(width=16)
                self.checkbox_thumbnails = dpg.add_checkbox(label="Thumbnails",
                                                            default_value=self._grid_mode,
                                                            show=_grid_is_available(),
                                                            callback=_thumbnails_checkbox_callback)
                with dpg.tooltip(self.checkbox_thumbnails):
                    dpg.add_text("Show the listing as image thumbnails instead of a table.\n"
                                 "Turns itself on when the file type filter selects images;\n"
                                 "setting it by hand overrides that until you close the dialog.")
            _draw_sort_indicators()

        # --------------------------------------------------------------------------------
        # Grid view.

        def _grid_is_available() -> bool:
            """Whether this dialog offers the grid view at all.

            A directory picker does not, *as things stand*: with no files listed, every tile would be the
            same folder icon, so the grid would cost space and legibility and show nothing the table does
            not. `raven-cherrypick` opens the dialog this way, which is what makes this worth stating
            rather than leaving to the automatic rule below.

            What would overturn it is a folder tile that previews what is *inside* the folder, which is
            worth having and is the reason this is a predicate rather than an assertion baked into the
            layout.
            """
            return not self.dirs_only

        def _filter_is_image_typed(label) -> bool:
            """Whether the named file type filter selects images and nothing else.

            The catch-all does not count: ".*" selects images *among* everything, and a directory of source
            code shown as thumbnails would be a wall of identical icons.
            """
            extensions = self._filter_extensions.get(label)
            if not extensions:
                return False
            return all(ext in _DECODABLE_IMAGE_EXTENSIONS for ext in extensions)

        def _the_grid():
            """The grid view, built on first use.

            Deferred because building it costs the thumbnail decoder and its device — several seconds of
            torch import for an app that may never switch views. A dialog used only as a list pays none of
            it.
            """
            if self._grid is None:
                # Imported here for the same reason: `raven.common.gui.filegrid` reaches torch, and every
                # app with a file dialog would otherwise pay that at startup.
                from ...common.gui import filegrid

                icon_assets = {name: tuple(getattr(self, f"ico_{name}")) for name in _ICON_NAMES}
                self._grid = filegrid.FileGrid(parent=f"grid_host_{self.instance_tag}",  # tag
                                               width=self._grid_size[0], height=self._grid_size[1],
                                               icon_assets=icon_assets,
                                               icon_name_for=_tile_icon_for,
                                               selectable_for=_is_choosable,
                                               tile_size=self.thumbnail_size,
                                               thumbnail_device=self.thumbnail_device,
                                               allow_multi_select=self.multi_selection,
                                               on_current_entry_changed=_grid_current_changed,
                                               on_selection_changed_entries=_grid_selection_changed,
                                               on_activate=_grid_activate)
            return self._grid

        def _is_choosable(entry) -> bool:
            """Whether `entry` is a thing this dialog can return.

            A directory is navigated into rather than chosen, except in a directory picker where it is the
            only thing there is to choose. `..` is never a choice, and a broken link leads nowhere.
            """
            if entry is None or entry.is_parent:
                return False
            if entry.kind == filelisting.KIND_BROKEN_LINK:
                return False
            return self.dirs_only if entry.is_dir else True

        def _grid_selection_changed(entries):
            """The grid's selection is the dialog's, filtered to what can actually be returned.

            This is what makes Ctrl+click mean in the grid what it means in the table. Without it the
            dialog knew only about the *cursor*, so a user could mark five images, press OK, and get one —
            which is worse than not offering the gesture, and is what `allow_multi_select` denies outright
            when the dialog was not opened for multi-selection.
            """
            self.selected_files.clear()
            self.selected_files.extend(entry.path for entry in entries if _is_choosable(entry))

        def _grid_current_changed(entry):
            """Single click in the grid: select, exactly as clicking a row does.

            The selection callback has already recorded the click; what is left is save mode's habit of
            offering the clicked name as the name to save as.
            """
            if not _is_choosable(entry):
                return
            if self.save_mode:
                basename, _ext = os.path.splitext(entry.name)
                dpg.set_value(f"ex_search_{self.instance_tag}", basename)
                self._update_search()

        def _grid_activate(entry):
            """Double click in the grid: descend into the directory, or accept the file."""
            if entry.is_dir:
                dpg.set_value(f"ex_search_{self.instance_tag}", "")
                chdir(entry.path)
                return
            if not _is_choosable(entry):
                return  # a broken link leads nowhere, and `..` was handled above
            self.selected_files.clear()
            self.selected_files.append(entry.path)
            self.ok()

        def set_grid_mode(enabled, remember=True, rebuild=True):
            """Switch between the table and the thumbnail grid.

            `remember`: whether this counts as the user's own choice, which then overrides the automatic
            switching until they choose again. The automatic path passes `False`.
            `rebuild`: whether to re-list into the new view. `False` where the caller is about to re-list
            anyway, so a filter change does not build the listing twice.

            **Switching views changes nothing else**: the sort order is app state that a view switch does
            not touch, and the cursor is re-anchored by path on every rebuild, a view switch included.
            """
            enabled = bool(enabled) and _grid_is_available()
            if remember:
                self._grid_mode_chosen_by_user = True
            if enabled == self._grid_mode:
                dpg.set_value(self.checkbox_thumbnails, enabled)  # in case a refused request left it on
                return
            self._grid_mode = enabled
            dpg.set_value(self.checkbox_thumbnails, enabled)
            dpg.configure_item(f"grid_host_{self.instance_tag}", show=enabled)  # tag
            dpg.configure_item(f"explorer_{self.instance_tag}", show=not enabled)  # tag
            if enabled:
                _resize_grid()
                if self.is_visible():
                    _start_grid_ticker()
            if rebuild:
                self._update_search()  # rebuild into the view that is now on screen
        self.set_grid_mode = set_grid_mode  # instance-injected for the same reason as `set_type_filter` above.

        def _thumbnails_checkbox_callback(sender, app_data):
            set_grid_mode(app_data)

        def _apply_automatic_grid_mode(rebuild=True):
            """Turn the grid on for an image-typed filter, unless the user has said otherwise."""
            if self._grid_mode_chosen_by_user:
                return
            set_grid_mode(_filter_is_image_typed(self.file_filter), remember=False, rebuild=rebuild)

        def _reset_grid_mode_for_opening():
            """Forget a hand-set view, so the automatic rule gets to decide again. Called on each opening.

            **The checkbox is per-opening**, and it has to be: a `FileDialog` is built once and lives as long
            as the app, so an override that outlived one opening would outlive the whole session — tick the
            box once and the automatic switching is gone until the app restarts. That is not an override,
            it is a one-way door.

            Within an opening the choice does hold, filter changes included: having said "not this time",
            the user should not have to say it again for every filter they try.

            What it resets *to* is the caller's `show_thumbnails`, not "no preference": an app that asked
            for a particular view asked for it every time, not only the first.
            """
            self._grid_mode_chosen_by_user = (self._show_thumbnails_default is not None)
            if self._show_thumbnails_default is not None:
                set_grid_mode(self._show_thumbnails_default, remember=True, rebuild=False)
            else:
                _apply_automatic_grid_mode(rebuild=False)
        self._reset_grid_mode_for_opening = _reset_grid_mode_for_opening  # instance-injected, as `set_type_filter` above.

        def _resize_grid():
            """Match the grid to the area the table would have filled.

            Measured rather than computed: the shortcuts panel is resizable, so the listing's width is not
            known at construction and changes while the dialog is open.
            """
            if self._grid is None:
                return
            width, height = guiutils.get_widget_size(f"listing_area_{self.instance_tag}")  # tag
            size = (max(64, int(width) - 4), max(64, int(height) - 4))
            if size != self._grid_size:
                self._grid_size = size
                self._grid.set_size(*size)

        def _start_grid_ticker():
            """Run the grid's per-frame work on a thread of the dialog's own, for as long as it is on screen.

            The grid needs `update()` every frame and the decoder needs polling, and `FileDialog` is a
            widget inside apps that own their render loops — so requiring every host app to call something
            would be a landmine: the app that forgets is the one whose thumbnails never appear. DPG permits
            item work from any thread, and `visible_on_screen` reads what the last frame drew.

            **It exists only while the dialog is up, and closing it joins the thread**, which is the part
            that is not merely tidy. A thread that calls DPG cannot outlive the DPG context: after
            `destroy_context` every call into the library is into freed memory, and the failure is a
            segfault rather than an exception — the guard would have to be a DPG call itself. Tying the
            thread's life to the dialog being visible keeps it inside a window where the context provably
            exists. (Found by the test suite, which builds and tears down contexts for a living.)
            """
            if self._ticker is not None and self._ticker.is_alive():
                return

            def tick_loop():
                while not self._ticker_stop.wait(_GRID_TICK_INTERVAL):
                    try:
                        # The app closing with the picker still open is the one exit this thread is not
                        # told about, and the one that races `destroy_context`. The render loop stops
                        # first, so this reads False well before the context goes — and it is only ever
                        # consulted from a thread that started while a *visible* dialog was rendering,
                        # which is what makes False mean "stopped" here rather than "not started yet".
                        if not dpg.is_dearpygui_running():
                            return
                        if self._grid is None or not self._grid_mode or not self.is_visible():
                            continue
                        _resize_grid()
                        self._grid.tick()
                    except Exception as exc:
                        logger.error(f"tick_loop: instance '{self.tag}' ({self.instance_tag}): {type(exc)}: {exc}")

            self._ticker_stop.clear()
            self._ticker = threading.Thread(target=tick_loop, daemon=True,
                                            name=f"fdialog_grid_tick_{self.instance_tag}")
            self._ticker.start()
        self._start_grid_ticker = _start_grid_ticker  # instance-injected for the same reason as `set_type_filter` above.

        # main file dialog header
        with dpg.window(label=self.title, tag=self.tag, on_close=self.cancel, no_resize=self.no_resize, show=False, modal=self.modal, width=self.width, height=self.height, min_size=self.min_size, no_collapse=True, pos=(50, 50)):
            info_px = 90

            # horizontal group (shot_menu + dir_list)
            with dpg.group(horizontal=True):
                # shortcut menu
                if (self.user_style == 0):
                    with dpg.child_window(tag=f"shortcut_menu_{self.instance_tag}", width=200, resizable_x=True, show=self.show_shortcuts_menu, height=-info_px):
                        home = get_directory_path("Home")
                        desktop = get_directory_path("Desktop")
                        downloads = get_directory_path("Downloads")
                        images = get_directory_path("Pictures")
                        documents = get_directory_path("Documents")
                        musics = get_directory_path("Music")
                        videos = get_directory_path("Videos")

                        with dpg.group(horizontal=True):
                            dpg.add_image(self.img_home)
                            dpg.add_menu_item(label="Home", callback=lambda: chdir(home))
                        with dpg.group(horizontal=True):
                            dpg.add_image(self.img_desktop)
                            dpg.add_menu_item(label="Desktop", callback=lambda: chdir(desktop))
                        with dpg.group(horizontal=True):
                            dpg.add_image(self.img_downloads)
                            dpg.add_menu_item(label="Downloads", callback=lambda: chdir(downloads))
                        with dpg.group(horizontal=True):
                            dpg.add_image(self.img_picture_folder)
                            dpg.add_menu_item(label="Images", callback=lambda: chdir(images))
                        with dpg.group(horizontal=True):
                            dpg.add_image(self.img_document_folder)
                            dpg.add_menu_item(label="Documents", callback=lambda: chdir(documents))
                        with dpg.group(horizontal=True):
                            dpg.add_image(self.img_music_folder)
                            dpg.add_menu_item(label="Music", callback=lambda: chdir(musics))
                        with dpg.group(horizontal=True):
                            dpg.add_image(self.img_videos)
                            dpg.add_menu_item(label="Videos", callback=lambda: chdir(videos))

                        dpg.add_separator()

                        # i/e drives list
                        with dpg.group():
                            drives = _get_all_drives()
                            for drive in drives:
                                with dpg.group(horizontal=True):
                                    dpg.add_image(self.img_hard_disk)
                                    dpg.add_menu_item(label=drive, user_data=drive, callback=open_drive)

                elif (self.user_style == 1):
                    with dpg.child_window(tag=f"shortcut_menu_{self.instance_tag}", width=40, show=self.show_shortcuts_menu, height=-info_px):
                        home = get_directory_path("Home")
                        desktop = get_directory_path("Desktop")
                        downloads = get_directory_path("Downloads")
                        images = get_directory_path("Pictures")
                        documents = get_directory_path("Documents")
                        musics = get_directory_path("Music")
                        videos = get_directory_path("Videos")

                        dpg.add_image_button(self.img_home, callback=lambda: chdir(home))
                        dpg.add_image_button(self.img_desktop, callback=lambda: chdir(desktop))
                        dpg.add_image_button(self.img_downloads, callback=lambda: chdir(downloads))
                        dpg.add_image_button(self.img_picture_folder, callback=lambda: chdir(images))
                        dpg.add_image_button(self.img_document_folder, callback=lambda: chdir(documents))
                        dpg.add_image_button(self.img_music_folder, callback=lambda: chdir(musics))
                        dpg.add_image_button(self.img_videos, callback=lambda: chdir(videos))

                        dpg.add_separator()

                        with dpg.group():
                            drives = _get_all_drives()
                            for drive in drives:
                                dpg.add_image_button(texture_tag=self.img_hard_disk, label=drive, user_data=drive, callback=open_drive)

                with dpg.child_window(height=-info_px):
                    # main explorer header
                    with dpg.group():
                        with dpg.group(horizontal=True):
                            self.button_refresh = dpg.add_image_button(self.img_refresh, tag=f"button_refresh_{self.instance_tag}")
                            with dpg.tooltip(self.button_refresh):
                                dpg.add_text("Refresh the current folder listing [F5]")  # TODO: move the hotkey handler for this dialog here
                            self.button_back_to_default_path = dpg.add_image_button(self.img_back, tag=f"button_back_to_default_path_{self.instance_tag}")
                            with dpg.tooltip(self.button_back_to_default_path):
                                dpg.add_text("Go back to the default path [Ctrl+Home]")  # TODO: move the hotkey handler for this dialog here
                            dpg.set_item_callback(self.button_refresh, self.refresh)
                            dpg.set_item_callback(self.button_back_to_default_path, self.back_to_default_path)

                            dpg.add_input_text(hint="Path", on_enter=True, callback=on_path_enter, default_value=os.getcwd(), width=-1, tag=f"ex_path_input_{self.instance_tag}")

                        with dpg.group(horizontal=True):
                            search_hint = "Search files [Ctrl+F]" if not save_mode else "Filename to save as [Ctrl+F]"  # TODO: move the hotkey handler for this dialog here
                            self.search_field = dpg.add_input_text(hint=search_hint, callback=self._update_search, tag=f"ex_search_{self.instance_tag}", width=-1)

                        _make_sort_row()

                        # Both views live here, one shown at a time. A container of their own, so the grid
                        # can be sized to the area the table would have filled — which is not known at
                        # construction, the shortcuts panel being resizable.
                        with dpg.child_window(tag=f"listing_area_{self.instance_tag}",  # tag
                                              width=-1, height=-1, border=False, no_scrollbar=True):
                            dpg.add_group(tag=f"grid_host_{self.instance_tag}", show=self._grid_mode)  # tag

                            # main explorer table header
                            with dpg.table(
                                tag=f'explorer_{self.instance_tag}',
                                show=not self._grid_mode,
                                height=-1,
                                width=-1,
                                resizable=True,
                                policy=dpg.mvTable_SizingStretchProp,
                                borders_innerV=True,
                                # Reordering and hiding are header-drag gestures that earn nothing in a
                                # picker with four fixed columns; sorting has moved to the button row above.
                                # `resizable` stays, and is why the header itself does: filename lengths
                                # vary enormously between users and directories.
                                reorderable=False,
                                hideable=False,
                                sortable=False,
                                scrollX=True,
                                scrollY=True,
                                # ImGui submits every row of a table each frame unless the table clips to
                                # the visible range. Measured on a 2500-row listing: 3.76 ms per frame
                                # without, 0.68 ms with — the latter being what an empty listing costs,
                                # i.e. the row count stops mattering. The clipper requires uniform row
                                # height, which holds here because every cell is created with
                                # `height=self.selec_height`.
                                clipper=True,
                            ):
                                # Proportional weights (the table's policy is `mvTable_SizingStretchProp`),
                                # so what matters is the ratios rather than the numbers. Tuned by looking,
                                # on a directory of papers with long filenames.
                                #
                                # `Size` and `Type` are sized to their *widest* value and no wider, since
                                # every pixel here comes out of the filename: `239.5 KiB` for size — IEC
                                # prefixes are a character wider than the old `240 KB` and carry a decimal
                                # — and `Link»File` for type, which is why that column cannot go back to
                                # what it was. `Date` gave up a little as well; it had slack the eye does
                                # not miss.
                                iwow_name = 100
                                iwow_date = 40
                                iwow_type = 15
                                iwow_size = 17
                                dpg.add_table_column(label='Name', init_width_or_weight=iwow_name, tag=f"ex_name_{self.instance_tag}")
                                dpg.add_table_column(label='Date', init_width_or_weight=iwow_date, tag=f"ex_date_{self.instance_tag}")
                                dpg.add_table_column(label='Type', init_width_or_weight=iwow_type, tag=f"ex_type_{self.instance_tag}")
                                dpg.add_table_column(label='Size', init_width_or_weight=iwow_size, width=10, tag=f"ex_size_{self.instance_tag}")

            with dpg.group(horizontal=True):
                # The combo's right edge needs no tuning: `width=-1` takes it to the container's edge, which
                # is the table's. What the spacer sets is where the *label* starts, and therefore how wide
                # the combo ends up — so it was retuned when the label was shortened, to keep the combo the
                # width the filter names actually need rather than letting it sprawl.
                dpg.add_spacer(width=610)
                dpg.add_text('Show')
                self.combo_file_filter = dpg.add_combo(items=self._filter_labels,
                                                       callback=filter_combo_selector, default_value=self.file_filter, width=-1)
                with dpg.tooltip(self.combo_file_filter):
                    self.text_file_filter_extensions = dpg.add_text(_describe_type_filter(self.file_filter))

            with dpg.group(horizontal=True):
                self.spacer_notification = dpg.add_spacer(width=int(self.width * 0.5))
                self.text_notification = dpg.add_text("")

            with dpg.group(horizontal=True):
                self.spacer_okcancel = dpg.add_spacer(width=int(self.width * 0.5))
                self.btn_ok = dpg.add_button(label="OK", width=100, tag=self.tag + "_return", callback=self.ok)
                self.btn_cancel = dpg.add_button(label="Cancel", width=100, callback=self.cancel)

            # After the widgets exist, since it may flip the view: an image-typed filter comes up as a grid
            # unless the caller said otherwise. `rebuild=False` because `chdir` below lists the directory.
            _apply_automatic_grid_mode(rebuild=False)
            chdir(self.default_path)

    # high-level functions
    def show_file_dialog(self):
        # Timed alongside `reset_dir`'s own phases, because "the dialog takes a moment to appear" can mean
        # the listing, or the frame this waits for, and the two have nothing to do with each other. The
        # entry line also timestamps the moment this callback got to run, which is the other candidate: DPG
        # runs callbacks one at a time, so a click can be waiting behind whatever ran before it.
        logger.debug(f"show_file_dialog: instance '{self.tag}' ({self.instance_tag}), entered")
        # Before the listing is built, so it is built into the view this opening will actually show.
        self._reset_grid_mode_for_opening()
        with timer() as tim_listing:
            self.chdir(self.last_path)
        dpg.show_item(self.tag)

        global visible_dialog_instance
        visible_dialog_instance = self

        # Align the OK/Cancel buttons to the right. The wait is for the geometry to exist, and is not
        # required: buttons a few pixels off beat an app that hangs with no traceback, which is what a bare
        # `dpg.split_frame` does when nothing will render a frame — a test suite, or startup before the loop.
        with timer() as tim_frame:
            guiutils.split_frame(operation="file dialog: aligning the OK/Cancel buttons", required=False)
        logger.debug(f"show_file_dialog: instance '{self.tag}' ({self.instance_tag}), "
                     f"listing {tim_listing.dt:.3f}s, waited {tim_frame.dt:.3f}s for a frame")
        old_width = dpg.get_item_width(self.spacer_okcancel)
        new_width = self.width - (dpg.get_item_width(self.btn_ok) +
                                  dpg.get_item_width(self.btn_cancel) +
                                  33)  # 33: magical constant matching the default theme, to align the buttons to the right edge of the file type picker. 3 * (8 (outer padding) + 3 (inner padding))?
        logger.debug(f"show_file_dialog: instance '{self.tag}' ({self.instance_tag}), window width = {self.width}, spacer old width = {old_width}, new width = {new_width}")
        dpg.set_item_width(self.spacer_okcancel, new_width)
        dpg.set_item_width(self.spacer_notification, new_width)

        if self._grid_mode:
            self._start_grid_ticker()

        dpg.focus_item(self.search_field)

    def _stop_grid_ticker(self):
        """Stop the grid's tick thread and wait for it to notice.

        Waiting matters: the thread calls DPG, and a DPG call after the context is destroyed is a segfault
        rather than an exception. The timeout is a few tick intervals, so a wedged thread does not hold the
        GUI — at which point it is a daemon and the process can still exit.

        **Except when the tick thread is the one closing the dialog**, which is the ordinary way to choose a
        file in grid view: a double-click is dispatched from the grid's own `update`, so `ok` runs *on* the
        tick thread and joining it would be joining the caller. Python raises `RuntimeError` for that, and
        the exception landed mid-`ok` — after the file had been handed to the app, before the selection was
        cleared — leaving state behind for the next `ok` to act on. Setting the flag is enough here: the
        loop is already on its way out and will see it on the next pass.
        """
        self._ticker_stop.set()
        ticker, self._ticker = self._ticker, None
        if ticker is None or not ticker.is_alive() or ticker is threading.current_thread():
            return
        ticker.join(timeout=1.0)
        if ticker.is_alive():
            logger.warning(f"_stop_grid_ticker: instance '{self.tag}' ({self.instance_tag}), tick thread did not stop within the timeout")

    def destroy(self):
        """Release what the dialog holds outside its widget tree. Call before destroying the DPG context.

        Only the grid view holds anything: a thumbnail decoder with its own threads, and the tick thread
        that feeds it. A dialog that was never switched to the grid has nothing to do here.
        """
        self._stop_grid_ticker()
        if self._grid is not None:
            self._grid.destroy()
            self._grid = None

    def is_visible(self):
        """Return whether the dialog is currently on screen.

        Apps ask this to suppress hotkeys and drops while a modal picker is up. Having it here is what keeps
        `tag` from having to be known outside the constructor that set it.
        """
        return dpg.is_item_visible(self.tag)  # tag

    def _forget_listing(self):
        """Drop what the closed dialog knew about its listing, without touching the widgets.

        Closing used to rebuild the listing instead — `ok` did it twice, `cancel` once — which is work
        thrown away twice over: the rows are hidden, and the next `show_file_dialog` rebuilds them anyway.
        Measured on a 2520-entry directory, a rebuild is ~0.19 s, so a close cost up to ~0.4 s.

        That cost was not merely slow, it was the second symptom filed against this dialog: the button
        that opens it appeared dead for a moment afterwards, its own acknowledgement flash included.
        Callbacks run one at a time on DPG's single callback thread, so the opener's callback was waiting
        behind this one rather than being lost.

        The rows themselves stay, and cost nothing while the window is hidden: a hidden window renders
        nothing, and `reset_dir` starts by deleting them on the next open.

        The grid's tick thread is the exception, and does stop: it costs whether or not anyone is looking,
        and it must not be running when the app tears the DPG context down.
        """
        self.selected_files.clear()
        self.shown_items.clear()
        self._stop_grid_ticker()

    def refresh(self):
        cwd = os.getcwd()
        logger.debug(f"refresh: instance '{self.tag}' ({self.instance_tag}), refreshing at cwd = '{cwd}'")
        self.reset_dir(default_path=cwd)
        # Raven: Acknowledge the action in the GUI.
        gui_animation.animator.add(gui_animation.WidgetFlash(message="",
                                                             target=self.button_refresh,
                                                             target_tooltip=None,
                                                             target_text=None,
                                                             duration=1.0))

    def back_to_default_path(self):
        logger.debug(f"back_to_default_path: instance '{self.tag}' ({self.instance_tag}), going back to '{self.default_path}'")
        self.chdir(self.default_path)
        # Raven: Acknowledge the action in the GUI.
        gui_animation.animator.add(gui_animation.WidgetFlash(message="",
                                                             target=self.button_back_to_default_path,
                                                             target_tooltip=None,
                                                             target_text=None,
                                                             duration=1.0))

    def _update_search(self):
        res = dpg.get_value(f"ex_search_{self.instance_tag}")
        self.reset_dir(default_path=os.getcwd(), file_name_filter=res)

    def ok(self):
        """Close dialog and accept currently selected files.

        The list of selected files is sent to `callback`.
        """
        if not self.selected_files:
            logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), no file(s) selected from the GUI table; figuring out what to do.")

            if self.save_mode:
                logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), this dialog is in save mode; using content of search field as the 'save as' filename.")
                save_as_file_name = dpg.get_value(f"ex_search_{self.instance_tag}")
                if not save_as_file_name:
                    logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), search field is empty, cannot save with empty filename; rejecting the ok.")
                    gui_animation.animator.add(gui_animation.WidgetFlash(message="Please enter a filename",
                                                                         target=self.btn_ok,
                                                                         target_tooltip=None,
                                                                         target_text=self.text_notification,
                                                                         flash_color=(255, 32, 32),  # orange for warning
                                                                         text_color=(255, 255, 255),
                                                                         duration=1.0))
                    return
                full_path = os.path.join(os.getcwd(), save_as_file_name)
                self.selected_files.append(full_path)
            else:  # "open file" (or directory) mode
                logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), this dialog is in 'open file' mode; checking if we can select all item(s) shown.")
                if len(self.shown_items) == 1:  # This allows typing into search until there is a unique match, and then pressing ok to open that item.
                    logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), exactly one item is shown; selecting that item.")
                    self.selected_files.append(self.shown_items[0])
                elif len(self.shown_items) > 1:  # ...and the same for multiple items in `multi_selection` mode.
                    if self.multi_selection:
                        logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), multiple items are shown, multi_selection is enabled; selecting all of them.")
                        self.selected_files.extend(self.shown_items)
                    else:
                        logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), multiple items are shown, multi_selection is disabled; rejecting the ok.")
                        gui_animation.animator.add(gui_animation.WidgetFlash(message="Please select an item",
                                                                             target=self.btn_ok,
                                                                             target_tooltip=None,
                                                                             target_text=self.text_notification,
                                                                             flash_color=(255, 32, 32),  # orange for warning
                                                                             text_color=(255, 255, 255),
                                                                             duration=1.0))
                        return
                else:
                    logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), no items shown (maybe nothing matches the search?); rejecting the ok.")
                    if self.multi_selection:
                        msg = "Please select at least one item"
                    else:
                        msg = "Please select an item"
                    gui_animation.animator.add(gui_animation.WidgetFlash(message=msg,
                                                                         target=self.btn_ok,
                                                                         target_tooltip=None,
                                                                         target_text=self.text_notification,
                                                                         flash_color=(255, 32, 32),  # orange for warning
                                                                         text_color=(255, 255, 255),
                                                                         duration=1.0))
                    return
        assert len(self.selected_files)  # at least one file selected if we get here

        # Save mode: Ensure presence of file extension.
        if self.save_mode and self.default_file_extension is not None:
            def ensure_ext(path):
                path_lower = path.lower()
                if not any(path_lower.endswith(ext) for ext in self._all_extensions):  # any valid ext is fine, but if none match, add the default ext.
                    logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), automatically adding default file extension '{self.default_file_extension}' to '{path}'.")
                    return path + self.default_file_extension
                return path
            new_selected_files = [ensure_ext(path) for path in self.selected_files]
            self.selected_files.clear()
            self.selected_files.extend(new_selected_files)

        # Save mode: Require another click of OK (within a short time) (or a triple-click, or two double-clicks, of the filename in the list) to confirm overwrite.
        # This is a non-intrusive UI that doesn't need another modal dialog.
        confirm_duration = 2.0
        current_time = time.time()
        double_okd = (current_time - self.last_ok_time < confirm_duration)
        self.last_ok_time = current_time
        if self.save_mode and os.path.exists(self.selected_files[0]) and not double_okd:
            # Raven: Acknowledge the action in the GUI.
            gui_animation.animator.add(gui_animation.WidgetFlash(message="Press again to overwrite file",
                                                                 target=self.btn_ok,
                                                                 target_tooltip=None,
                                                                 target_text=self.text_notification,
                                                                 flash_color=(255, 32, 32),  # orange for warning
                                                                 text_color=(255, 255, 255),
                                                                 duration=confirm_duration))
            return

        logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), hiding dialog and returning {self.selected_files}.")
        dpg.hide_item(self.tag)
        global visible_dialog_instance
        visible_dialog_instance = None
        if self.callback is not None:
            self.callback(self.selected_files)
        dpg.set_value(f"ex_search_{self.instance_tag}", "")  # clear the search when exiting
        self.last_path = os.getcwd()  # update remembered path when the dialog is closed with OK
        self._forget_listing()  # after the callback, which was handed `selected_files`

    def cancel(self):
        """Close dialog without selecting any files.

        An empty list is sent to `callback`, so that your app can trigger any cleanup actions needed
        (e.g. re-enabling certain GUI elements or animations after a modal dialog exits).
        """
        logger.debug(f"cancel: instance '{self.tag}' ({self.instance_tag}), hiding dialog and returning empty list.")
        dpg.hide_item(self.tag)
        global visible_dialog_instance
        visible_dialog_instance = None
        if self.callback is not None:
            self.callback([])
        dpg.set_value(f"ex_search_{self.instance_tag}", "")  # clear the search when exiting
        self._forget_listing()

    def change_callback(self, callback):
        self.callback = callback
        dpg.configure_item(self.tag + "_return", callback=self.callback)
