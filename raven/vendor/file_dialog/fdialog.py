# file_dialog 3.1
# MIT licensed

__all__ = ["FileDialog"]

import logging
logger = logging.getLogger(__name__)

import os
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
from ...common.gui import thumbnailgrid
from ...common.gui import utils as guiutils
from ...common.gui.tablecursor import TableCursor


# Page Up and Page Down as a key handler actually receives them. `dpg.mvKey_Prior` and `dpg.mvKey_Next`
# are stale DPG-1.x values (266, 267) that no longer match anything delivered, so comparing against the
# constants silently never fires. Confirmed against the live enum: Tab=512, Up=515, Down=516, **517**,
# **518**, Home=519, End=520 — the pair sits exactly where the sequence says it should.
_KEY_PAGE_UP = 517
_KEY_PAGE_DOWN = 518

# How many rendered frames to wait for the find field to release the caret before giving up on writing to
# it. A queued focus change lands within a frame or two on an idle app, but that is one sample and not a
# number to build on — what it actually costs depends on what else is in flight. Generous, therefore, and
# a bound rather than an estimate: the point is to fail with a log line instead of looping forever.
_FIELD_DEACTIVATION_FRAMES = 30


# The shortcuts the places panel offers, in the order they are shown: the label, which is also the folder
# name looked for under the home directory, and the icon.
#
# The two being the same is the point. `get_directory_path` resolves a place by joining `~` with this name
# on every platform, so a label that differs from the folder names a directory the panel will not open.
# `Pictures` was labelled "Images" and pointed at `~/Pictures`, which is what Linux, macOS and Windows all
# call it.
# Home leads, being where the others live; the rest are alphabetical, case-insensitively. A list a reader
# cannot predict is one they have to scan every time. `test_the_places_are_ordered_predictably` holds this,
# so an addition cannot quietly land in the wrong row.
_PLACES = [("Home", "img_home"),
           ("Desktop", "img_desktop"),
           ("Documents", "img_document_folder"),
           ("Downloads", "img_downloads"),
           ("Music", "img_music_folder"),
           ("Pictures", "img_picture_folder"),
           ("Videos", "img_videos")]


# The sort criteria a dialog offers, in the order its buttons appear — which is also the order Ctrl+Shift+N
# indexes, so the two rows read off one list and cannot drift apart.
_SORT_CRITERIA = [(filelisting.SortKey.NAME, "Name"),
                  (filelisting.SortKey.DATE, "Date"),
                  (filelisting.SortKey.KIND, "Type"),
                  (filelisting.SortKey.SIZE, "Size")]


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


def _icon_name_for_extension(file_name: str) -> Optional[str]:
    """Which icon `file_name`'s extension asks for, or `None` for a type with no icon of its own.

    Matched case-insensitively, so `PHOTO.JPG` gets the picture icon that `photo.jpg` does. The answer is
    an icon *name*, not a texture: the table draws it small and the grid draws it at tile size, so the two
    views share this table and pick their own assets from it.
    """
    file_name = file_name.lower()
    for extensions, icon_name in _EXTENSION_ICONS.items():
        if file_name.endswith(extensions):  # `str.endswith` takes a tuple
            return icon_name
    return None


def _get_all_drives():
    """Mount points to offer in the shortcuts panel, one menu item each.

    Mount points, specifically: every entry has to be somewhere `chdir` can go, because that is the only
    thing clicking one does.

    A POSIX-only branch here used to also scan /dev for names starting with "sd" or "nvme" and append
    the raw device paths. It was skipped on Windows (`os.name == 'posix'`), which is why the panel
    looked right there and only there. On this machine it added four entries — the two partitions
    already listed by their mount points, plus the whole disk and the controller — and a block device
    is not a directory, so each could only raise `NotADirectoryError` into the message box. Its
    dedup test could never fire either: it compared /dev paths against a list of mount points.
    """
    return [drive.mountpoint for drive in psutil.disk_partitions() if drive.mountpoint]


# How often the grid's own tick runs while it is on screen. `FileDialog` is a widget inside apps that own
# their render loops, so requiring every host app to call something would be a landmine — the app that
# forgets is the one whose thumbnails never appear.
_GRID_TICK_INTERVAL = 1.0 / 60


def _complete_from(text: str, candidates: Iterable[str]) -> Optional[str]:
    """What `text` becomes when Tab completes it against `candidates`. `None` when there is nothing to add.

    The answer is the candidates' longest common prefix — Tab asks *"what do the things I am looking at
    have in common?"*, and `candidates` is what the listing is showing.

    Which is the whole of the rule, because the listing has already been narrowed by the fragment search.
    Typing `re` leaves only the entries containing `re` on screen, so their common prefix is a real answer
    about them; there is no need to prefer the ones that happen to *start* with what was typed, and
    preferring them is actively wrong. Type `data` against `rawdata`, `datasets` and `tempdatasets` — all
    three shown, all three legitimate — and a prefix preference answers `datasets`, which then filters
    `rawdata` off the screen. Better to complete nothing than to discard a match the user can see.

    Matching is smart-case, as everywhere else in this dialog: `text` in all lowercase compares
    case-insensitively, and one carrying an uppercase letter compares exactly.

    Which decides the casing of the answer, and it is the same principle again. In case-sensitive mode all
    the candidates agree on the shared prefix letter for letter, so it can be returned as it stands. In
    case-insensitive mode they need not — `README` and `readme.txt` share `readme` only when folded — and
    returning either spelling would make the field case-*sensitive*, dropping the other from the listing.
    So the folded form is returned there: it still matches everything that matched before.
    """
    candidates = list(candidates)
    if not candidates:
        return None
    case_insensitive = text.islower() or not any(c.isalpha() for c in text)
    fold = str.lower if case_insensitive else (lambda s: s)
    common = os.path.commonprefix([fold(c) for c in candidates])
    if len(common) <= len(text):
        return None
    return common if case_insensitive else candidates[0][:len(common)]


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
    """Route a key press to whichever dialog is on screen.

    Registry-level concerns only — there is one global handler for every dialog, so its job is to decide
    *who* is listening. What each key does belongs to the instance, where the closures a hotkey needs to
    reach actually live.
    """
    if visible_dialog_instance is None:
        return
    visible_dialog_instance._handle_key(app_data)


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

            # `..` is never something the dialog returns, so it should read like the other entries it will
            # not return — but it is also the way out of the directory, so it cannot be a *disabled* widget
            # the way those are. Same grey, reached for directly instead of through the widget state.
            with dpg.theme() as cls.unreturnable_text_theme:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvThemeCol_Text, guiutils.DISABLED_TEXT_COLOR, category=dpg.mvThemeCat_Core)

            # A cursor twin for every theme a row cell can wear. Only one theme binds per item, so the
            # cursor's colour cannot be *added* to a cell that already carries an alignment — each base
            # theme needs a variant saying the same thing plus "the cursor is here".
            #
            # That is a product, and products grow: it stays affordable only because both axes are fixed
            # and small — three alignments, cursor or not. A third axis would be six more themes, and would
            # be the moment to stop binding whole themes per cell and find another way.
            #
            # Both enabled states, because a theme component covers one of them. A file in a folder picker
            # is a *disabled* selectable, and the cursor must stay visible as it travels over one —
            # otherwise it would vanish over exactly the rows such a picker is mostly made of.
            def _cursor_variant(align_x):
                with dpg.theme() as theme:
                    for enabled in (True, False):
                        with dpg.theme_component(dpg.mvAll, enabled_state=enabled):
                            dpg.add_theme_color(dpg.mvThemeCol_Text, thumbnailgrid.CURSOR_COLOR,
                                                category=dpg.mvThemeCat_Core)
                            if align_x is not None:
                                dpg.add_theme_style(dpg.mvStyleVar_SelectableTextAlign, x=align_x, y=.5)
                return theme
            cls.selec_alignt_cursor = _cursor_variant(0)
            cls.size_alignt_cursor = _cursor_variant(1)
            cls.unreturnable_text_theme_cursor = _cursor_variant(None)

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
        width=1400,
        height=820,
        min_size=(960, 400),
        pick="file",
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
        no_resize=False,
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
            width:                  int, File dialog window width (pixels). The default is chosen to show a
                                    useful number of files without the user reaching for the border, while
                                    still fitting at the window's (50, 50) position inside the smallest
                                    viewport in the constellation.
            height:                 int, File dialog window height (pixels).
            min_size:               (int, int), File dialog minimum size. The floor exists because the sort
                                    row is fixed-width buttons that cannot reflow: below 945 px of window
                                    width the rightmost checkbox is clipped off the edge. Measured at
                                    `font_size=20`, which every app in the constellation uses; raise it if
                                    yours does not, and `test_the_sort_row_fits_the_minimum_width` re-measures
                                    it whenever the row grows another control.
            pick:                   What this dialog returns, and what it lists on the way there. Two axes
                                    that used to be one flag, because a folder picker that lets you *look*
                                    at a folder before choosing it needs them apart.

                                    "file" (the default): returns file(s). Directories are listed and are
                                    navigated into rather than chosen.

                                    "dir": returns a directory. Only directories are listed, so there is
                                    nothing on screen but the choices. The right mode for picking a place to
                                    write to.

                                    "dir-with-contents": returns a directory, but lists its files too, shown
                                    dimmed and not selectable. For picking a folder *by what is in it* — an
                                    image tool wants the thumbnail grid here, so `show_thumbnails` becomes
                                    available where "dir" refuses it.

                                    In both directory modes, OK with nothing selected returns the directory
                                    currently being shown, so descending into a folder and accepting is a
                                    way of choosing it. Clicking a folder and pressing OK still returns that
                                    folder. The notification line names whichever one OK would return.
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
            no_resize:              If True, the window will not be resizable. Defaults to False, so the user
                                    can trade screen space for rows when a directory warrants it; `min_size`
                                    is what stops the layout being shrunk past the point where its controls
                                    fit.
            modal:                  If True, use DPG modal mode; a sort of popup effect. Can cause problems if the file dialog is opened by a modal window.
            show_hidden_files:      If True, the dialog shows also hidden files and folders. This is what
                                    it opens with; the user can toggle it from the Hidden checkbox or
                                    with Ctrl+H, and their choice holds for the rest of the session.
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
        if pick not in ("file", "dir", "dir-with-contents"):
            raise ValueError(f"FileDialog: unknown `pick` mode '{pick}'; expected 'file', 'dir' or 'dir-with-contents'.")
        self.pick = pick
        # The two axes `pick` separates, named for what each one decides, so no call site has to re-derive
        # them from the mode string.
        self.returns_dir = (pick != "file")
        self.lists_files = (pick != "dir")
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
        self._grid_mode = bool(show_thumbnails) and self.lists_files
        self._grid_mode_chosen_by_user = (show_thumbnails is not None)
        self._grid_size = (400, 300)  # replaced by a measurement as soon as the dialog has rendered
        self._ticker = None
        self._ticker_stop = threading.Event()

        self.instance_tag = f"0x{id(self):x}"  # for making unique DPG tags
        self.last_path = default_path  # for returning to last used directory when the dialog is closed and later re-opened

        self.PAYLOAD_TYPE = 'ws_' + self.tag
        self.selected_files = []
        self.shown_items = []  # for selection by search filter upon pressing ok

        # The rows the table is currently showing, in display order, and the cells of each with the themes
        # they wear normally and as the cursor. Both include `..`, which `shown_items` deliberately does
        # not: the cursor has to be able to reach it — arrowing up to `..` and pressing Enter is how you
        # leave a directory from the keyboard — while `ok`'s unique-match shortcut must never select it.
        self._row_entries = []
        self._row_themes = []
        # `(origin, pitch)` once two rows have been seen laid out; see `_row_metrics`. Not cleared on
        # rebuild: both are constants of the table's styling rather than of its contents, and the rows that
        # could re-measure them are exactly the ones a clipping table may not have drawn.
        self._row_metrics_cache = None
        self._sort_indicators = {}  # SortKey -> drawlist tag, one per sort button
        # Which of the dialog's two keyboard modes is up: the caret in the find field, or in the listing.
        # Tab swaps them. Held as a flag rather than derived from `is_item_active` on the field, because
        # the two are not the same question — the field is inactive whenever anything else has been
        # clicked, and that must not silently rebind the arrow keys.
        self._caret_in_listing = False
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

        # Whether the caller named a save extension. If not, it is derived from the filter, and so has to be
        # re-derived whenever the offered filters change.
        self._default_file_extension_was_given = (default_file_extension is not None)

        self._install_filters(self.filter_list, self.file_filter)



        # low-level functions

















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



        # --------------------------------------------------------------------------------
        # Grid view.














        # --------------------------------------------------------------------------------
        # The table's keyboard cursor. The grid brings its own; this gives the other view one under the
        # same method names, so `_handle_key` picks a navigator and stops caring which view is up.







        self._table_cursor = TableCursor(on_paint=self._paint_row,
                                         on_scroll_into_view=self._scroll_row_into_view,
                                         page_size=self._rows_per_page,
                                         # The promised target follows the cursor now, so it has to be
                                         # rewritten whenever the cursor moves — including by a rebuild.
                                         on_current_changed=lambda _idx: self._refresh_target_notification())




        # --------------------------------------------------------------------------------
        # Hotkeys.


        # main file dialog header
        with dpg.window(label=self.title, tag=self.tag, on_close=self.cancel, no_resize=self.no_resize, show=False, modal=self.modal, width=self.width, height=self.height, min_size=self.min_size, no_collapse=True, pos=(50, 50)):
            info_px = 90

            # The places, resolved once. Held as data rather than as seven locals per branch: the two
            # `user_style` layouts were each spelling out the same seven lookups and then seven near-identical
            # rows, and a keyboard cursor over this panel needs the list to be something it can index anyway.
            self._places = {label: path for label, _icon in _PLACES
                            if (path := self.get_directory_path(label)) is not None}

            # horizontal group (shot_menu + dir_list)
            with dpg.group(horizontal=True):
                # shortcut menu
                if (self.user_style == 0):
                    with dpg.child_window(tag=f"shortcut_menu_{self.instance_tag}", width=200, resizable_x=True, show=self.show_shortcuts_menu, height=-info_px):
                        for label, icon in _PLACES:
                            if label not in self._places:  # this user has no such directory
                                continue
                            with dpg.group(horizontal=True):
                                dpg.add_image(getattr(self, icon))
                                # `label=label` binds this row's label at definition time; a bare closure over
                                # the loop variable would leave every entry pointing at Videos.
                                dpg.add_menu_item(label=label, callback=lambda label=label: self.chdir(self._places[label]))

                        dpg.add_separator()

                        # i/e drives list
                        with dpg.group():
                            drives = _get_all_drives()
                            for drive in drives:
                                with dpg.group(horizontal=True):
                                    dpg.add_image(self.img_hard_disk)
                                    dpg.add_menu_item(label=drive, user_data=drive, callback=self.open_drive)

                elif (self.user_style == 1):
                    with dpg.child_window(tag=f"shortcut_menu_{self.instance_tag}", width=40, show=self.show_shortcuts_menu, height=-info_px):
                        for label, icon in _PLACES:
                            if label not in self._places:  # this user has no such directory
                                continue
                            dpg.add_image_button(getattr(self, icon), callback=lambda label=label: self.chdir(self._places[label]))

                        dpg.add_separator()

                        with dpg.group():
                            drives = _get_all_drives()
                            for drive in drives:
                                dpg.add_image_button(texture_tag=self.img_hard_disk, label=drive, user_data=drive, callback=self.open_drive)

                with dpg.child_window(height=-info_px):
                    # main explorer header
                    with dpg.group():
                        with dpg.group(horizontal=True):
                            self.button_refresh = dpg.add_image_button(self.img_refresh, tag=f"button_refresh_{self.instance_tag}")
                            with dpg.tooltip(self.button_refresh):
                                dpg.add_text("Refresh the current folder listing [F5]")
                            self.button_back_to_default_path = dpg.add_image_button(self.img_back, tag=f"button_back_to_default_path_{self.instance_tag}")
                            with dpg.tooltip(self.button_back_to_default_path):
                                dpg.add_text("Go back to the default path [Ctrl+Home]")
                            dpg.set_item_callback(self.button_refresh, self.refresh)
                            dpg.set_item_callback(self.button_back_to_default_path, self.back_to_default_path)

                            dpg.add_input_text(hint="Path", on_enter=True, callback=self.on_path_enter, default_value=os.getcwd(), width=-1, tag=f"ex_path_input_{self.instance_tag}")

                        with dpg.group(horizontal=True):
                            search_hint = "Search files [Ctrl+F]" if not save_mode else "Filename to save as [Ctrl+F]"
                            self.search_field = dpg.add_input_text(hint=search_hint, callback=self._update_search, tag=f"ex_search_{self.instance_tag}", width=-1)

                        self._make_sort_row()

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
                #
                # A borderless child window rather than a spacer, because this gap is the only wide, empty,
                # already-existing place to say which folder OK would return. A child *clips* what it holds,
                # so a path longer than the gap is cut off inside it instead of running under the combo or
                # off the window — which is what a bare `add_text` here would do, at whatever depth the user
                # happened to navigate to. Adding a row of its own was the other option, and it pushed the
                # buttons off the bottom: the listing is `height=-1`, so anything after it overflows.
                #
                # Sized here for the construction width and re-sized by `_relayout` for whatever the window
                # measures later. Derived rather than written out, so it cannot go stale against the
                # construction width the way the literal it replaced did.
                with dpg.child_window(tag=f"target_area_{self.instance_tag}",  # tag
                                      width=max(0, self.width - self._TYPE_FILTER_ROW_TAIL),
                                      height=self.selec_height + 8,
                                      border=False, no_scrollbar=True, no_scroll_with_mouse=True):
                    self.text_target = dpg.add_text("", show=self.returns_dir)
                dpg.add_text('Show')
                self.combo_file_filter = dpg.add_combo(items=self._filter_labels,
                                                       callback=self.filter_combo_selector, default_value=self.file_filter, width=-1)
                with dpg.tooltip(self.combo_file_filter):
                    self.text_file_filter_extensions = dpg.add_text(self._describe_type_filter(self.file_filter))

            with dpg.group(horizontal=True):
                self.spacer_notification = dpg.add_spacer(width=int(self.width * 0.5))
                self.text_notification = dpg.add_text("")

            with dpg.group(horizontal=True):
                self.spacer_okcancel = dpg.add_spacer(width=int(self.width * 0.5))
                # "Pick folder" rather than "OK" where OK does not need a selection: the label is the
                # shortest place to say that pressing it now, with nothing selected, is a complete action.
                # *Pick* rather than *use*, to match what these dialogs are called — a file picker.
                self.btn_ok = dpg.add_button(label="Pick folder" if (self.returns_dir and not save_mode) else "OK",
                                             width=100, tag=self.tag + "_return", callback=self.ok)
                # Worth a tooltip precisely *because* this one changed: bare Enter used to press this
                # button, and now it acts on the cursor instead — descending into a folder rather than
                # accepting one. Anybody who learned the old behaviour learned it silently, and will
                # un-learn it the same way unless the button says so.
                with dpg.tooltip(self.btn_ok):
                    if self.returns_dir:
                        # Same first line whether or not this is a save: the dialog picks a folder either
                        # way, and what the caller then does with it is not the dialog's to describe.
                        second = ("Enter descends into the folder under the cursor." if save_mode else
                                  "Enter descends into the folder under the cursor instead of picking it.")
                        dpg.add_text(f"Pick the folder named above. [Ctrl+Enter]\n{second}")
                    else:
                        dpg.add_text("Accept the selection. [Ctrl+Enter]\n"
                                     "Enter descends into the folder under the cursor, or picks a file.")
                self.btn_cancel = dpg.add_button(label="Cancel", width=100, callback=self.cancel)
                with dpg.tooltip(self.btn_cancel):
                    dpg.add_text("Close without choosing anything. [Esc]")

            # After the widgets exist, since it may flip the view: an image-typed filter comes up as a grid
            # unless the caller said otherwise. `rebuild=False` because `chdir` below lists the directory.
            self._apply_automatic_grid_mode(rebuild=False)
            self.chdir(self.default_path)

        # Outside the window's `with`, so the registry is not parented into it. Held by ID rather than by
        # tag: nothing looks it up, and it lives as long as the window it is bound to.
        with dpg.item_handler_registry() as resize_registry:
            dpg.add_item_resize_handler(callback=lambda *_: self._relayout())
        dpg.bind_item_handler_registry(self.tag, resize_registry)  # tag

    # Widths reserved at the right end of the two bottom rows, so both can be re-aligned against a window
    # width that is not the one they were built at.
    #
    # `_TYPE_FILTER_ROW_TAIL` preserves what the literal spacer encodes: at the construction default of
    # 1150 px, a 610 px spacer leaves 540 for the `Show` label and its combo. Holding that constant is the
    # point — the combo is meant to be as wide as the filter names need, not as wide as the window.
    _TYPE_FILTER_ROW_TAIL = 540
    _OKCANCEL_ROW_PADDING = 33  # matches the default theme: 3 * (8 outer + 3 inner)?

    def _effective_target(self) -> Optional[str]:
        """What OK would return right now in a directory-picking mode. `None` in a file picker, which has
        no such notion — there, OK with nothing selected is a question rather than an answer.

        Four ways to name a folder, in the order a user's intent narrows:

        1. One they clicked. An explicit choice outranks everything.
        2. The one under the cursor, if it is a folder. A keyboard user arrows to a folder and expects to
           get *that* folder; without this they would get whatever directory the listing happens to be
           showing, which is the one thing they just navigated away from choosing.
        3. The only choosable thing left on screen — typing into the find field until a single folder
           survives is a way of picking it, and predates this method.
        4. The directory being shown. Descending into a folder and accepting is how you choose a folder
           you wanted to look inside first, which is the whole point of `"dir-with-contents"`.

        Exists so that the notification line and `ok` cannot disagree: the line promises what this returns,
        and `ok` returns it. Two copies of this reasoning would drift the first time one of them grew a
        case.
        """
        if not self.returns_dir:
            return None
        if self.save_mode:
            # Save mode answers from the name field, not from the listing — `ok` joins what was typed onto
            # the current directory, and creating a folder that does not exist yet is the usual reason to
            # be here. Nothing typed means nothing promised.
            typed = dpg.get_value(f"ex_search_{self.instance_tag}")  # tag
            return os.path.join(os.getcwd(), typed) if typed else None
        if self.selected_files:
            return self.selected_files[0]
        entry = self._cursor_entry()
        if entry is not None and entry.is_dir and not entry.is_parent:
            return entry.path
        # Only while something is typed. The shortcut is for *narrowing* — type until one folder survives,
        # then accept it — and without that guard it fires on any directory that merely happens to contain
        # one subfolder, promising the child while the cursor rests on `..` meaning the parent. Browsing
        # `~/Pictures` with a single album in it is enough to hit that.
        if dpg.get_value(f"ex_search_{self.instance_tag}"):  # tag
            choosable = [path for path in self.shown_items if os.path.isdir(path)]
            if len(choosable) == 1:
                return choosable[0]
        return os.getcwd()

    def _refresh_target_notification(self) -> None:
        """Keep the notification line naming the folder OK would return.

        The affordance is otherwise invisible — nothing on screen says that OK with no selection means
        "this one" — and an invisible affordance is one nobody uses. Naming the folder also doubles as
        confirmation of what is about to be handed back, which is worth having even once you know the rule.
        """
        if not self.returns_dir:
            return
        target = self._effective_target()
        with guiutils.nonexistent_ok():
            # "Pick" rather than "open" or "save": the dialog hands a path back and has no idea what the
            # caller does with it. `raven-cherrypick` opens the folder, the pose editor batch-writes into
            # it — same dialog, and neither verb is the dialog's to claim.
            dpg.set_value(self.text_target, f"Will pick: {target}" if target else "")

    def _relayout(self) -> None:
        """Re-align the bottom rows against the window's *current* width.

        Three spacers push the type-filter combo and the OK/Cancel buttons to the right edge, and all
        three used to be sized from the width the dialog was constructed at — which is correct exactly
        once. Everything else in the layout is elastic already (`width=-1`), so these were the only
        things standing between the dialog and a resizable window: below the construction width they
        pushed the OK and Cancel buttons off the edge entirely, leaving no way to accept or dismiss it.

        Safe to call before any geometry exists; it does nothing, and the first `show_file_dialog` runs
        it again once there is a window to measure.
        """
        width = dpg.get_item_width(self.tag)  # tag
        if not width:
            return
        okcancel_width = max(0, width - (dpg.get_item_width(self.btn_ok) +
                                         dpg.get_item_width(self.btn_cancel) +
                                         self._OKCANCEL_ROW_PADDING))
        dpg.set_item_width(self.spacer_okcancel, okcancel_width)
        dpg.set_item_width(self.spacer_notification, okcancel_width)
        dpg.set_item_width(f"target_area_{self.instance_tag}", max(0, width - self._TYPE_FILTER_ROW_TAIL))  # tag

    # high-level functions

    def _cursor_entry(self):
        """The listing entry the cursor is on, in whichever view is showing. `None` if there is none."""
        if self._grid_mode and self._grid is not None:
            return self._grid.current_entry
        idx = self._table_cursor.current
        if 0 <= idx < len(self._row_entries):
            return self._row_entries[idx]
        return None

    def _paint_row(self, idx, is_cursor):
        """Draw row `idx` as the cursor row, or as an ordinary one.

        Rebinding themes rather than rebuilding the row: a listing can be thousands of rows deep and a
        cursor move touches two of them.
        """
        if not (0 <= idx < len(self._row_themes)):
            return
        with guiutils.nonexistent_ok():
            for cell, base_theme, cursor_theme in self._row_themes[idx]:
                dpg.bind_item_theme(cell, cursor_theme if is_cursor else base_theme)

    def _row_metrics(self):
        """`(origin, pitch)` for the listing's rows — where row 0 starts, and how far apart rows sit.

        Measured, because neither number is the one that was asked for: cells created at
        `selec_height` = 16 come out 18 px tall at a 22 px pitch, below a header contributing an origin
        of its own.

        Measured *once*, because the row being scrolled *to* can never be measured. The table clips —
        `clipper=True`, which is what keeps a thousand-row listing cheap — so ImGui never submits a row
        outside the visible range, and its position reads back as 0. That is exactly the row a scroll
        is aimed at, so asking it where it is returns zero and the view never moves. Both numbers are
        constants for the dialog's life (uniform row height is the clipper's own requirement), so two
        adjacent rows measured while they happen to be on screen answer for every row afterwards.
        """
        if self._row_metrics_cache is not None:
            return self._row_metrics_cache
        if len(self._row_themes) < 2:
            return None
        with guiutils.nonexistent_ok():
            _, first = dpg.get_item_pos(self._row_themes[0][0][0])
            _, second = dpg.get_item_pos(self._row_themes[1][0][0])
            pitch = second - first
            if pitch > 0:  # both were laid out; zeros mean "not rendered yet" or "clipped away"
                self._row_metrics_cache = (first, pitch)
                logger.debug(f"_row_metrics: instance '{self.tag}' ({self.instance_tag}), "
                             f"origin={first}, pitch={pitch}")
                return self._row_metrics_cache
        return None

    def _view_height(self):
        """The visible height of the listing, measured on the container that reports one.

        Not the table: a DPG table has no `rect_size` in its state at all, so `get_widget_size` falls
        through to its *configuration* and answers with the `-1` it was created with. The enclosing
        child window reports the real number, and the table is what scrolls inside it.
        """
        with guiutils.nonexistent_ok():
            _, height = guiutils.get_widget_size(f"listing_area_{self.instance_tag}")  # tag
            return height if height > 0 else 0
        return 0

    def delete_table(self):
        for child in dpg.get_item_children(f"explorer_{self.instance_tag}", 1):
            dpg.delete_item(child)

    def _describe_type_filter(self, label: str) -> str:
        extensions = self._filter_extensions.get(label)
        if extensions is None:
            return "Every file, whatever its extension."
        return textwrap.fill(" ".join(extensions), width=72,
                             initial_indent="Matches: ", subsequent_indent="         ")

    def _draw_sort_indicators(self):
        """Redraw the triangle marking which criterion is active, and which way it points.

        Drawn rather than written: Raven's UI font is OpenSans, which has no triangle or arrow glyphs
        at all, so a text label would render a missing-glyph box. Ten lines of drawlist is cheaper than
        binding the icon font to a button, which would apply to its whole label.
        """
        for sort_key, drawlist in self._sort_indicators.items():
            dpg.delete_item(drawlist, children_only=True)
            if sort_key is not self._sort_key:
                continue
            color = (210, 210, 210, 255)
            if self._sort_descending:
                points = [(2, 10), (12, 10), (7, 18)]
            else:
                points = [(2, 18), (12, 18), (7, 10)]
            dpg.draw_triangle(*points, color=color, fill=color, parent=drawlist)

    def _filter_is_image_typed(self, label) -> bool:
        """Whether the named file type filter selects images and nothing else.

        The catch-all does not count: ".*" selects images *among* everything, and a directory of source
        code shown as thumbnails would be a wall of identical icons.
        """
        extensions = self._filter_extensions.get(label)
        if not extensions:
            return False
        return all(ext in _DECODABLE_IMAGE_EXTENSIONS for ext in extensions)

    def _grid_is_available(self) -> bool:
        """Whether this dialog offers the grid view at all.

        The question is whether there are files to show, not whether files can be *chosen*. A `"dir"`
        picker lists none, so every tile would be the same folder icon and the grid would cost space
        and legibility to show nothing the table does not. `"dir-with-contents"` lists them precisely
        so they can be looked at, which is the whole reason that mode exists — so it gets the grid even
        though what it returns is a directory.
        """
        return self.lists_files

    def _is_choosable(self, entry) -> bool:
        """Whether `entry` is a thing this dialog can return.

        One kind is returnable and the other is scenery, and which is which is `pick`'s whole job: a
        file picker navigates into directories rather than choosing them, and a directory picker shows
        files (in `"dir-with-contents"`) so the folder can be judged by them, without either becoming
        an answer. `..` is never a choice, and a broken link leads nowhere.
        """
        if entry is None or entry.is_parent:
            return False
        if entry.kind == filelisting.KIND_BROKEN_LINK:
            return False
        return self.returns_dir if entry.is_dir else not self.returns_dir

    def _matches_type_filter(self, file_name: str) -> bool:
        if self._active_extensions is None:  # ".*"
            return True
        file_name = file_name.lower()
        return any(file_name.endswith(ext) for ext in self._active_extensions)

    def _resize_grid(self):
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

    def _set_type_filter(self, label: str) -> None:
        self.file_filter = label
        if label in self._filter_extensions:
            self._active_extensions = self._filter_extensions[label]
        else:  # not one of the offered items; read it as a literal extension, as the single-extension form did
            self._active_extensions = None if label == ".*" else (label.lower(),)

    def get_directory_path(self, directory_name):
        """Where the shortcut named `directory_name` should go, or `None` if this user has no such place.

        `None` rather than a fallback path, because a shortcut that silently goes somewhere else is
        worse than one that is not offered: the panel omits the row instead.
        """
        # `common_utils.user_directory` is what knows that these directories are renamed on disk on
        # Linux — `~/Pictures` is `~/Kuvat` on a Finnish desktop — and reads the XDG definitions.
        # Joining `~` with an English name, which is what this did, finds nothing there.
        directory_path = common_utils.user_directory(directory_name)
        try:
            os.listdir(directory_path)  # test access, not just existence
        except OSError:
            logger.debug(f"get_directory_path: instance '{self.tag}' ({self.instance_tag}): "
                         f"no usable '{directory_name}' at '{directory_path}', omitting the shortcut")
            return None
        return str(directory_path)

    def message_box(self, title, message):
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

    def _grid_current_changed(self, entry):
        """Single click in the grid: select, exactly as clicking a row does.

        The selection callback has already recorded the click; what is left is save mode's habit of
        offering the clicked name as the name to save as.
        """
        # Before the choosability test: the promised target follows the cursor wherever it lands, and
        # a cursor sitting on something unreturnable is exactly when the line has to say so.
        self._refresh_target_notification()
        if not self._is_choosable(entry):
            return
        if self.save_mode:
            basename, _ext = os.path.splitext(entry.name)
            dpg.set_value(f"ex_search_{self.instance_tag}", basename)
            self._update_search()

    def _grid_selection_changed(self, entries):
        """The grid's selection is the dialog's, filtered to what can actually be returned.

        This is what makes Ctrl+click mean in the grid what it means in the table. Without it the
        dialog knew only about the *cursor*, so a user could mark five images, press OK, and get one —
        which is worse than not offering the gesture, and is what `allow_multi_select` denies outright
        when the dialog was not opened for multi-selection.
        """
        self.selected_files.clear()
        self.selected_files.extend(entry.path for entry in entries if self._is_choosable(entry))
        self._refresh_target_notification()

    def _icon_for(self, entry) -> Union[str, int]:
        """The small icon shown at the left of `entry`'s row."""
        if entry.is_dir:
            return self.img_mini_folder
        if entry.kind == filelisting.KIND_BROKEN_LINK:
            return self.img_mini_error
        icon_name = _icon_name_for_extension(entry.name)
        if icon_name is None:
            return self.img_mini_document
        return getattr(self, f"img_{icon_name}")

    def _install_filters(self, filter_list, file_filter=None) -> None:
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
        self._set_type_filter(self._filter_labels[0] if file_filter is None else file_filter)

        # A save dialog's default extension is nearly always the one extension its filter names, so derive
        # it rather than making the caller write it a third time. Only for a filter naming exactly one:
        # among several there is no principled choice, and silently picking the first would be a rule
        # nobody could predict from a call site.
        if not self._default_file_extension_was_given:
            if self._active_extensions is not None and len(self._active_extensions) == 1:
                self.default_file_extension = self._active_extensions[0]
            else:
                self.default_file_extension = None

    def _row_extent(self, idx):
        """Where row `idx` sits inside the table's scrollable content, as `(top, height)`."""
        metrics = self._row_metrics()
        if metrics is None or not (0 <= idx < len(self._row_themes)):
            return None
        origin, pitch = metrics
        return origin + idx * pitch, pitch

    def _rows_per_page(self):
        """Most of a screenful, keeping one row of context to read the new position against."""
        height = self._view_height()
        metrics = self._row_metrics()
        if not height or metrics is None:
            return 1
        return max(1, int(height / metrics[1]) - 1)

    def _start_grid_ticker(self):
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
                    self._resize_grid()
                    self._grid.tick()
                except Exception as exc:
                    logger.error(f"tick_loop: instance '{self.tag}' ({self.instance_tag}): {type(exc)}: {exc}")

        self._ticker_stop.clear()
        self._ticker = threading.Thread(target=tick_loop, daemon=True,
                                        name=f"fdialog_grid_tick_{self.instance_tag}")
        self._ticker.start()

    def _tile_icon_for(self, entry) -> Optional[str]:
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

    def sort_by(self, sort_key, descending=None):
        """Order the listing by `sort_key`, and rebuild it.

        `descending`: `None` (the default) is the click semantics — asking for the criterion already in
        force reverses it, and any other criterion starts ascending. Pass a bool to say which way
        outright, which is what restoring a remembered order wants.
        """
        if descending is None:
            descending = (not self._sort_descending) if sort_key is self._sort_key else False
        self._sort_key = sort_key
        self._sort_descending = descending
        self._draw_sort_indicators()
        self._update_search()  # re-lists the current directory under the current find query

    def _scroll_row_into_view(self, idx):
        """Move the least that puts row `idx` on screen, and nothing at all when it already is.

        Scrolling only when the row is outside the visible band is what keeps arrow navigation from
        yanking the listing on every keypress.
        """
        extent = self._row_extent(idx)
        height = self._view_height()
        table = f"explorer_{self.instance_tag}"  # tag
        if extent is None or not height:
            logger.debug(f"_scroll_row_into_view: instance '{self.tag}' ({self.instance_tag}), "
                         f"row {idx}: no geometry (extent={extent}, view height={height})")
            return
        row_top, row_height = extent
        with guiutils.nonexistent_ok():
            view_top = dpg.get_y_scroll(table)
            if row_top < view_top:
                new_top = row_top
            elif row_top + row_height > view_top + height:
                new_top = row_top + row_height - height
            else:
                logger.debug(f"_scroll_row_into_view: instance '{self.tag}' ({self.instance_tag}), "
                             f"row {idx} at {row_top}+{row_height} already inside "
                             f"{view_top}..{view_top + height}, not scrolling")
                return
            logger.debug(f"_scroll_row_into_view: instance '{self.tag}' ({self.instance_tag}), "
                         f"row {idx} at {row_top}+{row_height}, view {view_top}..{view_top + height}, "
                         f"scrolling to {max(0.0, float(new_top))}")
            dpg.set_y_scroll(table, max(0.0, float(new_top)))

    def set_grid_mode(self, enabled, remember=True, rebuild=True):
        """Switch between the table and the thumbnail grid.

        `remember`: whether this counts as the user's own choice, which then overrides the automatic
        switching until they choose again. The automatic path passes `False`.
        `rebuild`: whether to re-list into the new view. `False` where the caller is about to re-list
        anyway, so a filter change does not build the listing twice.

        **Switching views changes nothing else**: the sort order is app state that a view switch does
        not touch, and the cursor is re-anchored by path on every rebuild, a view switch included.
        """
        enabled = bool(enabled) and self._grid_is_available()
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
            self._resize_grid()
            if self.is_visible():
                self._start_grid_ticker()
        if rebuild:
            self._update_search()  # rebuild into the view that is now on screen

    def _apply_automatic_grid_mode(self, rebuild=True):
        """Turn the grid on for an image-typed filter, unless the user has said otherwise."""
        if self._grid_mode_chosen_by_user:
            return
        self.set_grid_mode(self._filter_is_image_typed(self.file_filter), remember=False, rebuild=rebuild)

    def _thumbnails_checkbox_callback(self, sender, app_data):
        self.set_grid_mode(app_data)

    def set_show_hidden_files(self, enabled):
        """Show or hide dotfiles and their platform equivalents, and re-list under the current query.

        The listing goes through the same rebuild a sort or a filter change does, so the cursor is
        re-anchored by path and clamped like any other — if it was sitting on a hidden entry when this
        turns them off, the clamp catches it.
        """
        enabled = bool(enabled)
        dpg.set_value(self.checkbox_hidden_files, enabled)  # which a keyboard route has not already done
        if enabled == self.show_hidden_files:
            return
        self.show_hidden_files = enabled
        self._update_search()  # re-lists the current directory under the current find query

    def _hidden_files_checkbox_callback(self, sender, app_data):
        self.set_show_hidden_files(app_data)

    def _make_sort_row(self):
        """The sort buttons, plus the two view toggles, on one row above the listing.

        The toggles sit next to them rather than off at the right edge, because the case the thumbnail
        one exists for is a filter that selects images *and* something else — Librarian's "Documents and
        images", say — where the automatic rule deliberately does not fire and the user has to find this.
        Next to the controls they are already using is where they will.
        """
        with dpg.group(horizontal=True):
            dpg.add_text("Sort by")
            for n, (sort_key, label) in enumerate(_SORT_CRITERIA, start=1):
                with dpg.group(horizontal=True):
                    button = dpg.add_button(label=label, width=70,
                                            user_data=sort_key,
                                            callback=lambda s, a, u: self.sort_by(u))
                    # The keys are numbered by position, so the tooltip is the only place a user finds
                    # out *which* number this button is without counting the row.
                    with dpg.tooltip(button):
                        dpg.add_text(f"Sort by {label.lower()} [Ctrl+Shift+{n}]\n"
                                     "Again to reverse the order.")
                    drawlist = dpg.add_drawlist(width=14, height=self.selec_height + 10)
                    self._sort_indicators[sort_key] = drawlist
            self.spacer_view_toggle = dpg.add_spacer(width=16)
            self.checkbox_thumbnails = dpg.add_checkbox(label="Thumbnails",
                                                        default_value=self._grid_mode,
                                                        show=self._grid_is_available(),
                                                        callback=self._thumbnails_checkbox_callback)
            with dpg.tooltip(self.checkbox_thumbnails):
                dpg.add_text("Show the listing as image thumbnails instead of a table. [Ctrl+T]\n"
                             "Turns itself on when the file type filter selects images;\n"
                             "setting it by hand overrides that until you close the dialog.")
            dpg.add_spacer(width=8)
            # Always offered, unlike the Thumbnails box: a directory picker lists no files to make tiles
            # of, but it lists hidden *folders* — which is the case where a config directory is what you
            # came for and nothing else reaches it.
            self.checkbox_hidden_files = dpg.add_checkbox(label="Hidden",
                                                          default_value=self.show_hidden_files,
                                                          callback=self._hidden_files_checkbox_callback)
            with dpg.tooltip(self.checkbox_hidden_files):
                dpg.add_text("Show hidden files and folders — names beginning with a dot,\n"
                             "and on Windows the ones the filesystem marks hidden. [Ctrl+H]")
        self._draw_sort_indicators()

    def _reset_grid_mode_for_opening(self):
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
            self.set_grid_mode(self._show_thumbnails_default, remember=True, rebuild=False)
        else:
            self._apply_automatic_grid_mode(rebuild=False)

    def on_path_enter(self):
        try:
            self.chdir(dpg.get_value(f"ex_path_input_{self.instance_tag}"))
        except FileNotFoundError:
            self.message_box("Invalid path", "No such file or directory")

    def open_drive(self, sender, app_data, user_data):
        self.chdir(user_data)

    def _deselect_recursive(self, root):
        """Deselect all selectables inside DPG widget `root`, including `root` itself."""
        if dpg.get_item_type(root) == "mvAppItemType::mvSelectable":
            dpg.set_value(root, False)
        for item in dpg.get_item_children(root, slot=1):
            self._deselect_recursive(item)

    def open_file(self, sender, app_data, user_data):  # `user_data`: [name, fullpath, timestamp, size]
        ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)

        # Detect double-click.
        # double_clicked = dpg.is_mouse_button_double_clicked(dpg.mvMouseButton_Left)  # TODO: doesn't work, why?
        current_time = time.time()
        double_clicked = (current_time - self.last_click_time < self.double_click_threshold)
        self.last_click_time = current_time

        logger.debug(f"open_file: instance '{self.tag}' ({self.instance_tag}), sender is {sender} (tag '{dpg.get_item_alias(sender)}', type {dpg.get_item_type(sender)}, value = {dpg.get_value(sender)}), app_data = {app_data}, user_data = {user_data}, ctrl = {ctrl_pressed}, doubleclick = {double_clicked}")

        # Multi selection. DPG has already flipped the selectable by the time this runs, so the widget is
        # the record of what the user just asked for and the bookkeeping follows it.
        if self.multi_selection and ctrl_pressed:
            self._mark_selected(user_data[1], dpg.get_value(sender) is True)
        # Single selection
        else:
            dpg.set_value(sender, False)  # unselect this item  (TODO: why? double-click handling?)

            if double_clicked:
                if user_data is not None and user_data[1] is not None:
                    if os.path.isdir(user_data[1]):
                        logger.debug(f"open_file: instance '{self.tag}' ({self.instance_tag}), Content: {dpg.get_item_label(sender)}, files: {user_data}")
                        self.chdir(user_data[1])
                        dpg.set_value(f"ex_search_{self.instance_tag}", "")
                    elif os.path.isfile(user_data[1]):
                        if len(self.selected_files) < 1:
                            self.selected_files.append(user_data[1])
                        self.ok()
                        return user_data[1]
            else:
                # Only what this dialog can return responds to a click. In "dir-with-contents" the file
                # rows are built disabled, so this is a second line of defence rather than the only one.
                if os.path.isdir(user_data[1]) if self.returns_dir else os.path.isfile(user_data[1]):
                    self._deselect_recursive(f"explorer_{self.instance_tag}")  # unselect others
                    dpg.set_value(sender, True)  # and select this item
                    # Save mode: populate file name field from clicked file, without file extension
                    if self.save_mode:
                        basename, ext = os.path.splitext(user_data[0])
                        dpg.set_value(f"ex_search_{self.instance_tag}", basename)
                        self._update_search()
                    self.selected_files.clear()
                    self.selected_files.append(user_data[1])
                    self._refresh_target_notification()

    def _mark_selected(self, path, selected) -> None:
        """Record `path` as selected or not, leaving the widget that shows it alone.

        The bookkeeping half of a multi-selection change, shared by Ctrl+click and Ctrl+Space so that the
        two cannot drift — and so that the promised-target line is refreshed by both. In a directory
        picker an explicit selection outranks the cursor, so a click that did not refresh the line left
        it naming the folder the user had just stopped choosing.
        """
        if selected:
            if path not in self.selected_files:
                self.selected_files.append(path)
        elif path in self.selected_files:
            self.selected_files.remove(path)
        self._refresh_target_notification()

    def _toggle_cursor_selection(self) -> None:
        """Ctrl+Space: mark or unmark the cursor's entry — what Ctrl+click does, without the mouse.

        Silently nothing outside multi-selection mode, where there is no such gesture to mirror, and on
        an entry the dialog would not return anyway.
        """
        if not self.multi_selection:
            return
        entry = self._cursor_entry()
        if entry is None or not self._is_choosable(entry):
            return
        if self._grid_mode:
            # The grid owns its selection and tells the dialog about it through `_grid_selection_changed`,
            # so asking the grid is the whole of it here.
            grid = self._the_grid()
            grid.toggle_select(grid.current)
            return
        # The name cell is the one the row's selection is kept on: every cell spans the columns, so
        # setting all four would stack the tint on itself.
        cell = self._row_themes[self._table_cursor.current][0][0]
        now_selected = not dpg.get_value(cell)
        dpg.set_value(cell, now_selected)
        self._mark_selected(entry.path, now_selected)

    def _make_row(self, entry, callback, parent=None, selected_paths=()):
        """Build one table row from a `filelisting.FileEntry`.

        The entry carries everything the row needs, so nothing here consults the filesystem and nothing
        has to be read back off the widgets later.

        `selected_paths`: which paths were selected before this rebuild. A row for one of them comes up
        already selected, so a selection survives a re-listing instead of quietly evaporating.

        `parent`: the table to build into. Defaults to this dialog's own listing table.
        """
        # Resolved here rather than in the signature: a default argument is evaluated once, when the
        # `def` runs, and for a method that is at class-definition time — where there is no instance to
        # ask. The value is the same for the life of a dialog either way, `instance_tag` being fixed.
        if parent is None:
            parent = f"explorer_{self.instance_tag}"  # tag

        # `..` is the way out of the directory rather than something in it: one spanning cell, and no
        # date/type/size.
        if entry.is_parent:
            with dpg.table_row(parent=parent):
                with dpg.group(horizontal=True):
                    # Dimmed like everything else the dialog will not return, which is what the grid
                    # does with it too: `..` is not a choice in any mode, so showing it at full strength
                    # among entries that *are* choices says the opposite of what is true.
                    dpg.add_image(self.img_mini_folder, user_data=entry.kind,
                                  tint_color=[255, 255, 255, self.image_transparency])
                    parent_cell = dpg.add_selectable(label=entry.name, callback=self._go_up_one_level,
                                                     span_columns=True, height=self.selec_height)
                    dpg.bind_item_theme(parent_cell, self.unreturnable_text_theme)
                    self._row_themes.append([(parent_cell,
                                              self.unreturnable_text_theme,
                                              self.unreturnable_text_theme_cursor)])
            return

        # Shown, but nothing can be done with it: not an answer this dialog returns, and not somewhere
        # to go either. Directories stay live in every mode — a file picker cannot *choose* one but must
        # let you walk into it — so the rule is choosability plus navigability, not choosability alone.
        # Files in a directory picker are the case this exists for; a broken link falls out of it too,
        # having been inert in practice all along without looking it.
        inert = not self._is_choosable(entry) and not entry.is_dir

        # `user_data` shape is `open_file`'s contract: [name, full path, mtime, size].
        kwargs_cell = {'callback': callback, 'span_columns': True, 'height': self.selec_height,
                       'enabled': not inert,
                       'user_data': [entry.name, entry.path, entry.mtime, entry.size or 0]}
        # Two independent reasons to recede, so they compound rather than saturate: a hidden file you
        # also cannot choose is less relevant than either alone, and one fixed alpha for "some reason
        # applies" would flatten that back out.
        dimming = 1.0
        if entry.is_hidden:
            dimming *= self.image_transparency / 255
        if inert:
            dimming *= self.image_transparency / 255
        alpha = round(255 * dimming)
        kwargs_image = {'tint_color': [255, 255, 255, alpha], 'user_data': entry.kind}

        with dpg.table_row(parent=parent):
            with dpg.group(horizontal=True):
                dpg.add_image(self._icon_for(entry), **kwargs_image)
                cell_name = dpg.add_selectable(label=entry.name, **kwargs_cell)
            cell_time = dpg.add_selectable(label=filelisting.format_mtime(entry.mtime), **kwargs_cell)
            cell_type = dpg.add_selectable(label=filelisting.format_kind(entry), **kwargs_cell)
            cell_size = dpg.add_selectable(label=filelisting.format_size(entry.size), **kwargs_cell)

            # Restore the selection: only the name cell is set, since every cell spans the columns and
            # setting all four would stack the tint on itself.
            if entry.path in selected_paths and self._is_choosable(entry):
                dpg.set_value(cell_name, True)
                self.selected_files.append(entry.path)

            if self.allow_drag:
                drag_payload = dpg.add_drag_payload(parent=cell_name, payload_type=self.PAYLOAD_TYPE)
            dpg.bind_item_theme(cell_name, self.selec_alignt)
            dpg.bind_item_theme(cell_time, self.selec_alignt)
            dpg.bind_item_theme(cell_type, self.selec_alignt)
            dpg.bind_item_theme(cell_size, self.size_alignt)

            # Each cell with the theme it wears normally and the one it wears as the cursor, so moving
            # the cursor is a rebind of four cells rather than a rebuild of the listing. Kept in row
            # order, matching `shown_items`, which is what the cursor indexes.
            self._row_themes.append([(cell_name, self.selec_alignt, self.selec_alignt_cursor),
                                     (cell_time, self.selec_alignt, self.selec_alignt_cursor),
                                     (cell_type, self.selec_alignt, self.selec_alignt_cursor),
                                     (cell_size, self.size_alignt, self.size_alignt_cursor)])
            if self.allow_drag:
                if entry.name.lower().endswith((".png", ".jpg")):
                    dpg.add_image(self.img_big_picture, parent=drag_payload)
                elif entry.is_dir:
                    dpg.add_image(self.img_folder, parent=drag_payload)
                else:
                    dpg.add_image(self.img_document, parent=drag_payload)

    def _go_up_one_level(self, sender, app_data, user_data):
        """GUI callback: if this item double-clicked, go up one level."""
        ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
        current_time = time.time()
        double_clicked = (current_time - self.last_click_time < self.double_click_threshold)
        self.last_click_time = current_time

        dpg.set_value(sender, False)  # unselect the ".." entry

        if ctrl_pressed:
            return
        if double_clicked:
            self.chdir("..")  # which clears the find field, as every route into it does

    def set_type_filter(self, label):
        """Select the file type filter by its label, exactly as picking it from the combo would.

        `label` is one of the labels derived from `filter_list` — a bare extension for a string entry,
        or the given label for a `(label, extensions)` pair.
        """
        self._set_type_filter(label)
        dpg.set_value(self.combo_file_filter, self.file_filter)  # keep the GUI in sync when called programmatically
        dpg.set_value(self.text_file_filter_extensions, self._describe_type_filter(self.file_filter))
        self._apply_automatic_grid_mode(rebuild=False)  # the listing is about to be rebuilt anyway
        self.reset_dir()

    def set_filter_list(self, filter_list, file_filter=None):
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
        self._install_filters(filter_list, file_filter)
        dpg.configure_item(self.combo_file_filter, items=self._filter_labels)
        dpg.set_value(self.combo_file_filter, self.file_filter)
        dpg.set_value(self.text_file_filter_extensions, self._describe_type_filter(self.file_filter))
        self._apply_automatic_grid_mode(rebuild=False)
        # The *configured* show flag, not `is_visible`: the latter answers "did the user see it in the last
        # rendered frame", which is False for a window shown microseconds ago and False always with no
        # render loop. The question here is whether a listing exists to be brought up to date.
        if dpg.get_item_configuration(self.tag)["show"]:  # tag
            self.reset_dir()

    def filter_combo_selector(self, sender, app_data):
        self.set_type_filter(dpg.get_value(sender))

    def chdir(self, path):
        """Go to `path` and list it. The one place this dialog navigates; every route ends up here."""
        try:
            os.chdir(path)
            # A query belongs to the directory it was typed in. `reset_dir` lists the new one unfiltered,
            # so a query left in the field would describe nothing on screen — the field claiming to narrow
            # while the listing shows everything. Cleared here rather than at the call sites because there
            # are several of them (a row's double-click, Enter on the cursor, the places panel, a drive,
            # the path field) and only this one is common to all.
            dpg.set_value(self.search_field, "")
            self.reset_dir()

            # Arriving somewhere new, the next thing a user does is usually look for something in it, so
            # the caret goes back to the find field ready to be typed into. Enter takes it away on the way
            # in — committing a single-line `InputText` deactivates it — so without this the first
            # keystroke after descending would land nowhere.
            #
            # Unless the listing had the caret, in which case it keeps it: arriving is not a reason to
            # change modes, and in grid view it would cost the arrow keys that Tab was needed to free.
            if not self._caret_in_listing:
                self._focus_field()
        except PermissionError as e:
            self.message_box("File dialog - PerimssionError", f"Cannot open the folder because is a system folder or the access is denied\n\nMore info:\n{e}")
        except NotADirectoryError as e:
            self.message_box("File dialog - not a directory", f"The selected item is not a directory, but a file.\n\nMore info:\n{e}")

    def reset_dir(self, file_name_filter=None):
        """Rebuild the listing of the working directory, optionally narrowed to `file_name_filter`.

        This *lists*; it does not navigate. Going somewhere is `chdir`, which moves the process and
        then calls this.
        """
        # Read from the process rather than accepted as an argument, so the two cannot disagree: `ok`
        # and the target notification both answer from `os.getcwd()`, and a listing built from
        # anywhere else would be a dialog that shows you A and hands back B.
        default_path = os.getcwd()
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
        self._row_entries = []
        self._row_themes = []

        # What this is a listing *of*. A rebuild of the same directory and a move to a different one
        # look identical from here, and the cursor wants opposite things from them — hold its place
        # across a re-filter, start at the top of somewhere new — so the views are told which directory
        # this is and work it out themselves.
        listed_dir = os.path.abspath(str(default_path))
        try:
            # Only when it would actually change. A rebuild happens on every keystroke in the find field
            # and on every Tab, and the directory is the same for all of them — reconfiguring the widget
            # each time is churn nobody asked for, in the one field the user is not interacting with.
            path_field = f"ex_path_input_{self.instance_tag}"  # tag
            if dpg.get_value(path_field) != default_path:
                dpg.configure_item(path_field, default_value=default_path)
            # Compiled once per rebuild rather than per entry: on a directory of thousands, the split is
            # the part worth hoisting out of the loop.
            matches_name_filter = common_utils.make_search_matcher(file_name_filter or "")

            # Enumerating, filtering and sorting all happen here, on data, before a widget is touched.
            with timer() as tim_list:
                entries = filelisting.list_directory(default_path,
                                                     show_hidden=self.show_hidden_files,
                                                     dirs_only=not self.lists_files,
                                                     name_filter=matches_name_filter,
                                                     type_filter=self._matches_type_filter,
                                                     sort_key=self._sort_key,
                                                     descending=self._sort_descending)
            # `..` stays out: it is the way out of the directory rather than a candidate for the
            # unique-match shortcut in `ok`.
            self.shown_items.extend(entry.path for entry in entries if not entry.is_parent)

            # Only the view on screen is built. The other one is emptied rather than left holding a
            # stale listing, which would be both the memory and, on a switch back, the wrong answer.
            with timer() as tim_delete:
                self.delete_table()

            with timer() as tim_build:
                if self._grid_mode:
                    grid = self._the_grid()
                    grid.set_listing(entries, listing_key=listed_dir)
                    # Re-made against the new order, and the grid's own callback puts the survivors
                    # back into `selected_files`.
                    grid.set_selected_paths(previously_selected)
                else:
                    for entry in entries:
                        self._make_row(entry, self.open_file, selected_paths=previously_selected)
                    # After the rows exist, since the cursor paints itself onto one of them.
                    self._row_entries = list(entries)
                    self._table_cursor.set_listing([entry.path for entry in entries],
                                                   listing_key=listed_dir)

                # `..` is where the cursor rests while nothing has been typed, and that is load-bearing
                # rather than incidental: the cursor is what Ctrl+Enter returns, so a cursor parked on the
                # first subfolder would hand back a *child* of the directory you navigated to in order to
                # choose it.
                #
                # Typing changes the question. A filter is a search, and a search puts the cursor on its
                # first hit — otherwise "type a few characters, press Enter" leaves the directory instead
                # of opening the match, `..` being what Enter would act on.
                #
                # `anchor=False` because the user chose a *query*, not an entry. The anchor is where the
                # cursor tries to return when the listing changes again, so anchoring a landing nobody
                # picked means erasing the query returns the cursor to whatever happened to match first
                # rather than to `..`.
                navigator = self._navigator()
                if entries:
                    if file_name_filter:
                        # A search shows its first hit. Always — typing a query is a fresh intent, so it
                        # overrides wherever the cursor had got to, including an entry arrowed to earlier.
                        #
                        # `..` is one of the things it can hit. The listing keeps it whatever is typed, so
                        # it is always there to escape through, but it answers the query like any other
                        # name: type `..` and the cursor lands on it, which is how "go up" is reachable by
                        # search rather than by knowing a separate key. When the query matches nothing at
                        # all the cursor stays there too, that being the one row left to act on.
                        hits = (i for i, entry in enumerate(entries) if matches_name_filter(entry.name))
                        navigator.set_current(next(hits, 0), anchor=False)  # 0, the way out, if none match
                    elif not navigator.is_anchored:
                        # No query, and nobody moved the cursor: it belongs at `..`, the resting place.
                        # A cursor that *was* moved is left alone — `set_listing` has already returned it
                        # to the entry it was moved to, which is where erasing a query should land you.
                        navigator.set_current(0, anchor=False)
                    # Neither placement anchors: the user chose a query, or nothing, but not an entry.

            logger.debug(f"reset_dir: instance '{self.tag}' ({self.instance_tag}), {len(self.shown_items)} entries "
                         f"as {'tiles' if self._grid_mode else 'rows'}: "
                         f"list {tim_list.dt:.3f}s, delete {tim_delete.dt:.3f}s, build {tim_build.dt:.3f}s")

        # exceptions
        except FileNotFoundError:
            logger.error(f"reset_dir: instance '{self.tag}' ({self.instance_tag}), invalid path: '{str(default_path)}'")
        except Exception as exc:
            # Logged with its traceback *before* the message box, which shows only `str(exc)`. A listing
            # error is otherwise reduced to one line with no stack — and where the dialog is modal the
            # box cannot even be shown, so the line goes to the log stripped of everything that would
            # locate it. Cost a CI round on a Windows-only failure that said "negative dimensions are
            # not allowed" and nothing about where.
            logger.exception(f"reset_dir: instance '{self.tag}' ({self.instance_tag}), failed to list '{str(default_path)}'")
            self.message_box("File dialog - Error", f"An unknown error has occured when listing the items, More info:\n{exc}")

        # Every path into here changes something the promised target depends on — which directory is
        # shown, and what survives the find field — so this is the one place that has to refresh it.
        self._refresh_target_notification()

    def _the_grid(self):
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
                                           icon_name_for=self._tile_icon_for,
                                           selectable_for=self._is_choosable,
                                           tile_size=self.thumbnail_size,
                                           thumbnail_device=self.thumbnail_device,
                                           allow_multi_select=self.multi_selection,
                                           on_current_entry_changed=self._grid_current_changed,
                                           on_selection_changed_entries=self._grid_selection_changed,
                                           on_activate=self._grid_activate)
        return self._grid

    def _grid_activate(self, entry):
        """Double click in the grid: descend into the directory, or accept the file."""
        if entry.is_dir:
            dpg.set_value(f"ex_search_{self.instance_tag}", "")
            self.chdir(entry.path)
            return
        if not self._is_choosable(entry):
            return  # a broken link leads nowhere, and `..` was handled above
        self.selected_files.clear()
        self.selected_files.append(entry.path)
        self.ok()

    def _activate_cursor_entry(self):
        """Enter: go as deep as this entry allows.

        One sentence covers every mode, which is why the rule reads as a rule rather than a table: `..`
        and directories are somewhere to *go*, and a file in a picker that returns files is the bottom,
        so accepting it is the deepest move available. A file in a folder picker is scenery — it is
        shown so the folder can be judged by it — and Enter on scenery does nothing.

        Ctrl+Enter is the counterpart that declines to descend, and it is `ok` unchanged.
        """
        entry = self._cursor_entry()
        if entry is None:
            self.ok()  # nothing under the cursor: fall back to what the OK button would do
            return
        if entry.is_parent or entry.is_dir:
            self.chdir(entry.path)
            return
        if not self._is_choosable(entry):
            return
        self.selected_files.clear()
        self.selected_files.append(entry.path)
        self.ok()

    def _handle_key(self, key: int) -> None:
        """Handle one key press for this dialog. Called by the module-level handler, which owns the
        registry and decides *which* dialog is listening; this decides what the key does.
        """
        ctrl = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)

        # TODO (briefs/researchers-night/filedialog-keyboard-brief.md): the rest of the keyboard —
        # TODO: the focus-parking chords.
        shift = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
        alt = dpg.is_key_down(dpg.mvKey_LAlt) or dpg.is_key_down(dpg.mvKey_RAlt)

        # Tab swaps the caret's two homes. ImGui does not spend Tab on an `InputText` — it neither
        # moves focus nor inserts anything — so the key is ours to define, and this is the only way
        # to reach the state where the find field is inactive.
        if key == dpg.mvKey_Tab:
            if self._caret_in_listing:
                # Written before the caret returns, the field being writable only while it does not have
                # it — the mirror of the outbound order below.
                self._fill_field_from_cursor()
                self._focus_field()
            else:
                # Order is load-bearing: the completion is a write to the find field, and a field with the
                # caret in it reverts one. Leaving is what makes the write possible, so the caret goes
                # first and the completion follows it out.
                self._focus_listing()
                self._complete_find_field()
            return

        # Ctrl and a number picks the Nth type filter; add Shift and it picks the Nth sort criterion.
        # One indexed rule for two labelled rows, which is what makes them worth remembering — and what
        # keeps the criteria off letters they cannot have, `Ctrl+Shift+S` reading as a Save variant in a
        # dialog that saves and `Ctrl+Shift+N` being *new folder* in every file manager.
        if ctrl and dpg.mvKey_1 <= key <= dpg.mvKey_9:
            n = key - dpg.mvKey_1
            if shift:
                if n < len(_SORT_CRITERIA):
                    self.sort_by(_SORT_CRITERIA[n][0])
            elif n < len(self._filter_labels):
                self.set_type_filter(self._filter_labels[n])
            return

        # Up one level, by either of two chords, before the bare key gets to mean "one row up".
        #
        # Alt+Up is what a file manager binds, and Ctrl+Up is a one-handed alias for it: on a Nordic
        # layout Alt sits only to the left of space — the right-hand key is AltGr, a different key —
        # so Alt+Up needs two hands, while Ctrl is mirrored and right Ctrl and the arrow cluster are
        # both under the right hand. It was the one two-handed chord in the set.
        if key == dpg.mvKey_Up and (alt or ctrl):
            self.chdir("..")
            return

        nav = self._navigator()
        if key == dpg.mvKey_Up:
            nav.navigate_row_up()
        elif key == dpg.mvKey_Down:
            nav.navigate_row_down()
        elif key == _KEY_PAGE_UP:
            nav.navigate_page_up()
        elif key == _KEY_PAGE_DOWN:
            nav.navigate_page_down()
        elif key == dpg.mvKey_Left and self._caret_in_listing:
            # Left and Right are not unwanted while the caret is in the find field, they are
            # *occupied* — a single-line entry spends them on the text caret. Tab is what frees them,
            # which is why the grid is only now completely reachable: its rows hold several tiles, so
            # without a horizontal step every column but the first was unvisitable from the keyboard.
            nav.navigate_prev()
        elif key == dpg.mvKey_Right and self._caret_in_listing:
            nav.navigate_next()
        elif key == dpg.mvKey_Home and not ctrl:
            nav.navigate_first()
        elif key == dpg.mvKey_End:
            nav.navigate_last()
        elif key == dpg.mvKey_Return and ctrl:
            self.ok()  # commit here, without descending
        elif key == dpg.mvKey_Return:
            self._activate_cursor_entry()
        elif key == dpg.mvKey_Escape:
            self.cancel()
        elif key == dpg.mvKey_F5:
            self.refresh()
        elif ctrl and key == dpg.mvKey_Home:
            self.back_to_default_path()
        elif ctrl and key == dpg.mvKey_F:
            self._focus_field()
        elif ctrl and key == dpg.mvKey_Spacebar:
            self._toggle_cursor_selection()
        elif ctrl and key == dpg.mvKey_H:
            # What every GTK and GNOME file chooser binds this to, which is the whole argument for it.
            self.set_show_hidden_files(not self.show_hidden_files)
        elif ctrl and key == dpg.mvKey_T:
            # `T` for Thumbnails, and free to mean it: the type filter gave the letter up when its label
            # became `Show`, which left the dialog's one unlabelled-by-key control holding the one
            # mnemonic that fits it. Silently ignored where the grid is not on offer — a directory
            # picker listing no files has nothing to make tiles of — since the checkbox is hidden there
            # too and a key that acts where its control is invisible is worse than one that does not.
            if self._grid_is_available():
                self.set_grid_mode(not self._grid_mode)

    def _navigator(self):
        """Whichever view is on screen, as the thing that answers to `navigate_*`.

        The two are interchangeable by construction rather than by adaptor — `TableCursor` was written to
        the interface `ThumbnailGrid` already had — so a key handler names a movement once and both views
        do the right thing with it.
        """
        return self._grid if (self._grid_mode and self._grid is not None) else self._table_cursor

    def _write_find_field(self, text: str) -> bool:
        """Put `text` in the find field and re-filter the listing. Returns whether the write landed.

        Only safe once the caret has left the field, which is the caller's job to have arranged.
        """
        # ImGui's edit buffer owns an *active* `InputText`: `set_value` appears to work — `get_value`
        # immediately after reports the new string — and the next frame writes the old buffer back, firing
        # the edit callback as it goes. So the write has to wait for the field to go inactive, which a
        # queued focus change does not achieve on the calling frame. How many frames it takes depends on
        # what else is in flight, so this polls rather than counting frames.
        for _ in range(_FIELD_DEACTIVATION_FRAMES):
            if not dpg.is_item_active(self.search_field):
                break
            if not guiutils.split_frame(operation="file dialog: waiting for the find field to go inactive",
                                        required=False):
                return False  # no render loop to wait for; nothing would land anyway
        else:
            logger.warning(f"_write_find_field: instance '{self.tag}' ({self.instance_tag}): the find field "
                           f"is still active after {_FIELD_DEACTIVATION_FRAMES} frames; not writing '{text}'")
            return False

        dpg.set_value(self.search_field, text)
        self._update_search()  # `set_value` fires no callback, so the filter is re-run by hand
        return True

    def _fill_field_from_cursor(self) -> None:
        """Put the cursor entry's name in the find field. Tab's inbound half.

        Does nothing when the cursor is on `..`, which is a way out of the directory rather than a name.
        """
        # Unconditional otherwise, because the only reason to press Tab in the first place is to go and
        # navigate: coming back means "give me the one I navigated to". A version that filled only when
        # the cursor had been *arrowed* to would guard a path nobody walks — refining a query needs no Tab,
        # the caret being in the field already.
        #
        # The query is not preserved, and that is the trade. Returning arms ImGui's select-all, so it was
        # a keystroke from gone regardless; a name you can edit beats text you were about to overwrite. It
        # is what makes "save a variant of this file" reachable from the keyboard — Tab onto the existing
        # name, then amend it — and in open mode it collapses the listing to the entry you picked.
        entry = self._cursor_entry()
        if entry is None or entry.is_parent:
            return
        self._write_find_field(os.path.basename(entry.path))

    def _complete_find_field(self) -> None:
        """Extend the find field to what the entries on screen have in common.

        The other half of Tab, applied on the way out of the field. Does nothing when there is nothing to
        add, which includes the case where what is shown has no common prefix at all.
        """
        text = dpg.get_value(self.search_field)
        completed = _complete_from(text, [os.path.basename(path) for path in self.shown_items])
        if completed is None:
            return
        self._write_find_field(completed)

    def _focus_field(self) -> None:
        """Put the caret back in the find field, where typing filters the listing."""
        self._caret_in_listing = False
        dpg.focus_item(self.search_field)

    def _focus_listing(self) -> None:
        """Take the caret out of the find field and give the listing the arrow keys.

        Leaves the find field inactive, which is what lets it be written programmatically: an active
        `InputText` owns ImGui's edit buffer and reverts writes on the next frame.
        """
        # Both views park on the same target, which keeps them answering to the same code. A table row is
        # a selectable and could hold focus itself, but then focus would have to chase the cursor on every
        # move, and grid view has nothing to chase — a drawlist has no focusable items.
        #
        # *Which* target is not free to choose, and not for a visual reason. `focus_item` cannot move
        # focus from outside a child window to inside one; measured across every source/target pair, that
        # is the only refused direction. The find field lives in the listing's child window, so parking
        # below it — on the OK button, say — makes every later Ctrl+F and Tab-back an outside-to-inside
        # request, which is ignored in silence: the caret never returns and typing goes nowhere. The
        # refresh button shares the child window with the field, so the return trip stays inside it.
        #
        # A button is also safe to park on: DPG leaves ImGui's keyboard-nav activation off, so a focused
        # button ignores Space and Enter instead of pressing itself. Pinned by
        # `test_a_focused_button_ignores_the_keys_that_would_press_it`, since nothing in the API reports it.
        self._caret_in_listing = True
        dpg.focus_item(self.button_refresh)

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
        self._relayout()

        if self._grid_mode:
            self._start_grid_ticker()

        # A dialog always opens ready to be typed into, whichever mode the previous one closed in.
        self._focus_field()

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
        self.reset_dir()
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
        self.reset_dir(file_name_filter=res)

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
            elif self.returns_dir:
                # A directory picker always has an answer: the folder being shown, if nothing narrower was
                # said. So it never rejects the OK, and the notification line has been promising this exact
                # path the whole time.
                target = self._effective_target()
                logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), directory picker; returning {target}.")
                self.selected_files.append(target)
            else:  # "open file" mode
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
            # A copy, because `_forget_listing` below clears `selected_files`: a callback that stores what
            # it is handed, rather than reading it immediately, would otherwise find the list empty by the
            # time it looks. The receiver owns what it is given.
            self.callback(list(self.selected_files))
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
