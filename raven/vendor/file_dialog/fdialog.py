# file_dialog 3.1
# MIT licensed

__all__ = ["FileDialog"]

import enum
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
from unpythonic.env import env

from ...common import filelisting
from ...common import utils as common_utils
from ...common.gui import animation as gui_animation
from ...common.gui import helpcard
from ...common.gui import keyboardmark
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

# The help card's size, in pixels. `HelpWindow` builds a fixed-size window with no scrollbar, so content
# that does not fit is clipped away silently — a row added to the table past this point needs the height
# raised with it, and a column needs the width.
_HELP_CARD_SIZE = (1250, 640)


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


class CaretHome(enum.Enum):
    """Where this dialog is taking keys — the caret's home, named rather than counted.

    Two rules read it, and both are written against *whichever* home it is rather than against a particular
    one, which is what lets a new one be an entry here plus a branch in the dispatch:

      - Bare Up / Down / Home / End mean whatever the home they arrive in says they mean.
      - Escape hands the caret back to the find field from wherever it was parked, and cancels the dialog
        only once it is already there.

    **The homes are two tiers, and Escape means "up one tier".** `FIELD` and `LISTING` are *the main thing*,
    and they are one tier because they are sister widgets: you type in one and watch the other, and Tab and
    Ctrl+F move within the pair. Everything else is an auxiliary control reached by a chord, so Escape from
    one abandons what was being done there and returns to the main thing — and from the main thing there is
    nowhere left to go but out of the dialog.

    So the listing answering Escape differently is not an exception to be tidied away: it is the rule, seen
    from the tier that has no tier above it. A new home is classified by asking which tier it belongs to,
    which is how the places panel is already answered.

    It is held rather than derived from `dpg.is_item_active` on the find field, because the two are not the
    same question: the field goes inactive whenever anything at all is clicked, and that must not silently
    rebind the arrow keys.
    """
    FIELD = "field"  # the find field — which is the filename field, in save mode
    LISTING = "listing"  # the table or the thumbnail grid, whichever view is up
    FILTER = "filter"  # the file type combo
    PATH = "path"  # the path field, where a whole folder is pasted or a short root typed
    PLACES = "places"  # the side panel of folder shortcuts and drives


# How a caller writes "every file" in a `filter_list`, and what the combo calls it. The two differ because
# `.*` is glob syntax, which belongs in the call site rather than in front of a user.
#
# The label says *files* and not "files and folders", which reads as the friendlier phrasing and would be
# wrong: a type filter is applied to files only — `filelisting.list_directory` says so, and must, since a
# filter that hid directories would hide the way to the files it selects. Folders are listed under every
# filter, so naming them here would imply the others exclude them.
_CATCH_ALL = ".*"
_CATCH_ALL_LABEL = "All files"


# What the dialog's text says by its color. The same three `raven-visualizer` paints its search field with,
# so an answer looks the same wherever a Raven app gives one.
#
# Two places use them, and they are the same three answers in both: the find field says whether what was
# typed finds anything, and the target line says whether what was asked for could be done.
#
# The neutral one is also DPG's own text color, which is what makes it the one to give a widget that has to
# fade back to something.
_TEXT_NEUTRAL = (255, 255, 255)  # nothing to report
_TEXT_GOOD = (180, 255, 180)
_TEXT_BAD = (255, 128, 128)

# Louder, and deliberately so. `_TEXT_BAD` is soft because it reports a *state* — that nothing here matches
# what you typed, which is an ordinary thing for a search to say and no reason to raise a voice. This one
# marks something that just went wrong, or a question that has to be answered before anything proceeds, and
# it is the same red the OK button flashes to ask one. Overrides the green `WidgetFlash` flashes by default.
_ALARM_RED = (255, 32, 32)

# Reporting a problem, in two times rather than one, because the two do different jobs. The fade from
# `_ALARM_RED` back to `_TEXT_NEUTRAL` is what catches the eye, and a slow one catches nothing — it has to
# read as a flash. The message then has to be read, which takes longer than that.
_REPORT_FLASH_SECONDS = 1.0
_REPORT_TEXT_SECONDS = 3.0


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
        if entry == _CATCH_ALL:
            return (_CATCH_ALL_LABEL, None)
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


def fdialog_key_release_callback(sender, app_data):
    """Route a key release to whichever dialog is on screen. Same division of labour as the press handler."""
    if visible_dialog_instance is None:
        return
    visible_dialog_instance._handle_key_release(app_data)


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
                dpg.add_key_release_handler(tag="fdialog_key_release_handler", callback=fdialog_key_release_callback)

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

            # Every colour the cursor is drawn in, across all of the variants below, so one animation can
            # pulsate the lot. They are one cursor however many themes it takes to draw it, and separate
            # animations over them would be free to drift out of phase.
            cls.cursor_color_widgets = []

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
                            cls.cursor_color_widgets.append(
                                dpg.add_theme_color(dpg.mvThemeCol_Text, keyboardmark.COLOR,
                                                    category=dpg.mvThemeCat_Core))
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
        filter_list=None,
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

                                    `None` (the default) offers only ".*", every file. A dialog that wants
                                    type filters knows which types it is for; one that does not say is
                                    better off not offering the choice at all.
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
        # A caller that named no types gets the one filter that means "no filtering". The old default was a
        # list of some 170 extensions, which reads in the combo as a junk drawer of formats the app has
        # nothing to do with — `.vhd`, `.qcow2`, `.msi` — and is at its most absurd in a folder picker,
        # where the listing has no files in it to be filtered.
        self.filter_list = [".*"] if filter_list is None else filter_list
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

        # The places panel's rows, in display order, and the cell of each with the themes it wears
        # normally and as the cursor. A drive is a place — both are a name standing for somewhere to go,
        # which is all a row of this panel means — so the two kinds share one list and one cursor runs
        # over both, separator and all.
        self._place_entries = []  # (label, path)
        self._place_themes = []  # (cell, base theme, cursor theme)
        # `(origin, pitch)` for the panel's rows once two have been seen laid out; see `_place_metrics`.
        self._place_metrics_cache = None

        # Where the dialog is taking keys. Assigned directly here, ahead of the widgets: the property
        # behind it repaints the places cursor, which needs the panel to exist. See `CaretHome`.
        self._caret_home_now = CaretHome.FIELD
        # What the parent directory of a half-typed path holds, as `(parent, subdirectory names)`. One slot
        # is enough: a path is typed one component at a time, so every keystroke between two separators asks
        # about the same parent, and crossing a separator is the moment the old answer stops being wanted.
        self._path_prefix_cache = None
        # The help card, built on first F1 — see `_the_help_card` — and whether it is currently up. The
        # flag is what the dialog is hidden behind: DPG will not stack a modal over a modal, so showing the
        # card means taking the dialog off the screen, and `is_visible` has to keep saying yes across that
        # gap or the app underneath re-enables its own hotkeys while the card is showing.
        self._help_window = None
        self._help_card_up = False
        # Set while the card is gone and the dialog is not back yet, waiting for Escape to be released.
        self._restore_pending = False
        # The cursor's pulsation, which runs only while this dialog is on screen. See `_start_cursor_pulse`.
        self._cursor_pulse = None
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

        # The table's keyboard cursor. The grid brings its own; this gives the other view one under the
        # same method names, so `_handle_key` picks a navigator and stops caring which view is up.
        self._table_cursor = TableCursor(on_paint=self._paint_row,
                                         on_scroll_into_view=self._scroll_row_into_view,
                                         page_size=self._rows_per_page,
                                         # The promised target follows the cursor now, so it has to be
                                         # rewritten whenever the cursor moves — including by a rebuild.
                                         on_current_changed=lambda _idx: self._refresh_target_notification())

        # The places panel's cursor, the same class over a much shorter list — which is the point of the
        # class having been written against callbacks rather than against the table. Nothing here reports
        # a promised target: a place is somewhere to go, and the dialog never returns one.
        self._places_cursor = TableCursor(on_paint=self._paint_place,
                                          on_scroll_into_view=self._scroll_place_into_view,
                                          page_size=self._places_per_page)

        # main file dialog header
        with dpg.window(label=self.title, tag=self.tag, on_close=self.cancel, no_resize=self.no_resize, show=False, modal=self.modal, width=self.width, height=self.height, min_size=self.min_size, no_collapse=True, pos=(50, 50)):
            # What the listing gives up to the rows under it — it is `height=-info_px`, so this is the
            # reservation for the target/type-filter row, the notification line, the buttons and the
            # spacing between them. Under-count it and the window's content overflows: DPG answers that
            # with a scrollbar, which takes its width off the right edge and clips the dialog there.
            #
            # The button row is named rather than folded in, because it is the term that moves. 66 is what
            # the measured 90 leaves once a 24 px button row is taken out of it.
            info_px = 66 + self._OKCANCEL_ROW_HEIGHT

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
                            self._add_place_row(label, self._places[label], getattr(self, icon))

                        dpg.add_separator()

                        # i/e drives list
                        for drive in _get_all_drives():
                            self._add_place_row(drive, drive, self.img_hard_disk)

                    self._places_cursor.set_listing([path for _label, path in self._place_entries],
                                                    listing_key="places")

                elif (self.user_style == 1):
                    with dpg.child_window(tag=f"shortcut_menu_{self.instance_tag}", width=40, show=self.show_shortcuts_menu, height=-info_px):
                        for label, icon in _PLACES:
                            if label not in self._places:  # this user has no such directory
                                continue
                            dpg.add_image_button(getattr(self, icon), user_data=self._places[label], callback=self.open_place)

                        dpg.add_separator()

                        with dpg.group():
                            drives = _get_all_drives()
                            for drive in drives:
                                dpg.add_image_button(texture_tag=self.img_hard_disk, label=drive, user_data=drive, callback=self.open_place)

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

                            self.path_field = dpg.add_input_text(hint="Path", on_enter=True, callback=self.on_path_enter,
                                                                 default_value=os.getcwd(), width=-1,
                                                                 tag=f"ex_path_input_{self.instance_tag}")
                            with dpg.tooltip(self.path_field):
                                dpg.add_text("Folder to browse — paste or type a path, then Enter [Ctrl+L]")

                            # Same three states as the find field below, and the same one-theme mechanism.
                            # What they *mean* differs: this one says whether Enter would go anywhere.
                            with dpg.theme() as path_field_theme:
                                with dpg.theme_component(dpg.mvInputText):
                                    self._path_field_color = dpg.add_theme_color(dpg.mvThemeCol_Text, _TEXT_NEUTRAL)
                            dpg.bind_item_theme(self.path_field, path_field_theme)

                            # `on_enter=True` buys Enter and costs every other keystroke: the field's own
                            # callback then fires only on the commit, so the color has nowhere to hook. An
                            # edited handler fires on each keystroke regardless of that flag (measured on
                            # DPG 2.3.1, one frame after the key, carrying the new value), which is what
                            # lets the field be a readout as it is typed into.
                            #
                            # The other two are the caret's, and they are what make *clicking* this field
                            # mean what Ctrl+L means. Were they missing, a click would leave `_caret_home`
                            # naming somewhere else and Enter would go two places at once: the global key
                            # handler runs first and would descend into whatever the listing cursor is on,
                            # with `on_path_enter` then navigating from there. An absolute path would still
                            # arrive; a relative one would be read against a directory nobody chose.
                            with dpg.item_handler_registry() as path_field_registry:
                                dpg.add_item_edited_handler(callback=self._recolor_path_field)
                                dpg.add_item_activated_handler(callback=self._on_path_field_activated)
                                dpg.add_item_deactivated_handler(callback=self._on_path_field_deactivated)
                            dpg.bind_item_handler_registry(self.path_field, path_field_registry)

                        with dpg.group(horizontal=True):
                            search_hint = "Search files [Ctrl+F]" if not save_mode else "Filename to save as [Ctrl+F]"
                            self.search_field = dpg.add_input_text(hint=search_hint, callback=self._update_search, tag=f"ex_search_{self.instance_tag}", width=-1)

                            # One theme, bound once, whose color is then moved by `set_value` — rebinding a
                            # fresh theme per keystroke would leak one per character typed.
                            with dpg.theme() as search_field_theme:
                                with dpg.theme_component(dpg.mvInputText):
                                    self._search_field_color = dpg.add_theme_color(dpg.mvThemeCol_Text, _TEXT_NEUTRAL)
                            dpg.bind_item_theme(self.search_field, search_field_theme)

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
                                      # The whole row where there is no combo beside it to leave room for,
                                      # which is also where the longest paths are read: a folder picker.
                                      width=(max(0, self.width - self._TYPE_FILTER_ROW_TAIL) if self._type_filter_is_available() else -1),
                                      height=self.selec_height + 8,
                                      border=False, no_scrollbar=True, no_scroll_with_mouse=True):
                    # Shown in every mode, though only a directory picker writes a target into it: it is
                    # also the widest line the dialog has, and so where anything that has to be *read*
                    # goes. A color of its own because a flash fades back to one — see `WidgetFlash`.
                    self.text_target = dpg.add_text("", color=_TEXT_NEUTRAL)
                # The label and the combo are one control, and this is what says so. It also decides
                # whether the caret can ever come back from here: `focus_item` is refused when focus sits
                # on an item at *window level* and the target is inside a child window, so a click on a
                # combo sitting directly in the dialog window left the find field unreachable — Ctrl+F,
                # Tab-back and Escape all fired and none of them arrived. Inside a child window the same
                # click is a child→child move away from the field, which works.
                #
                # Borderless, unpadded and background-free, so the grouping costs nothing on screen.
                with dpg.child_window(tag=f"type_filter_area_{self.instance_tag}",  # tag
                                      width=-1, height=self.selec_height + 8,
                                      show=self._type_filter_is_available(),
                                      border=False, no_scrollbar=True, no_scroll_with_mouse=True):
                    with dpg.group(horizontal=True):
                        dpg.add_text('Show')
                        self.combo_file_filter = dpg.add_combo(items=self._filter_labels,
                                                               callback=self.filter_combo_selector, default_value=self.file_filter, width=-1)
                with dpg.tooltip(self.combo_file_filter):
                    dpg.add_text("Show only files of this type [Ctrl+1 ... Ctrl+9]")
                    dpg.add_text("Browse the types with Up / Down / Home / End [Ctrl+Shift+F]")
                    self.text_file_filter_extensions = dpg.add_text(self._describe_type_filter(self.file_filter))

            with dpg.group(horizontal=True):
                self.spacer_notification = dpg.add_spacer(width=int(self.width * 0.5))
                self.text_notification = dpg.add_text("")

            # In a child window for the same reason the type filter is, and this pair is the subtler case:
            # they are usually harmless *because* clicking one closes the dialog, so there is no "rest of
            # the dialog's life" left for a stranded caret to matter in. The exception is the overwrite
            # confirmation, where the first click on OK deliberately leaves the dialog open — and from that
            # click on, focus sits at window level and the filename field can no longer be reached by
            # Ctrl+F, by Tab, or by any other key. Which is precisely the moment a user reaches for it, to
            # change the name instead of overwriting.
            with dpg.child_window(tag=f"okcancel_area_{self.instance_tag}",  # tag
                                  width=-1, height=self._OKCANCEL_ROW_HEIGHT,
                                  border=False, no_scrollbar=True, no_scroll_with_mouse=True):
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

    # A button is taller than a line of text or a combo, and this row holds buttons. ImGui sizes one as
    # `font_size + 2 * FramePadding.y` — 26 px at the font size every app in the constellation uses, which
    # is what `raven.cherrypick.config` and `raven.xdot_viewer.config` both call `TOOLBAR_H` — against the
    # 24 px the rows above take. A child window clips what it holds, so those two pixels came off the
    # bottom of the buttons, where `FrameRounding` is 6 and the curve is.
    _OKCANCEL_ROW_HEIGHT = 32

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
            #
            # Written through the flash, because this line is also where a problem is reported: while one
            # is on screen the message keeps the line, and this becomes what appears once it has been read.
            # A plain write would wipe the message mid-sentence, and a flash left to restore what it found
            # would put back a target from before the folders the user has visited since.
            gui_animation.set_text_under_flash(self.text_target, f"Will pick: {target}" if target else "")

    def _recolor_find_field(self) -> None:
        """Repaint the find field to say whether what is typed there finds anything here."""
        # In save mode the field names the file to be written, so the find-mode meaning does not apply: a
        # name nothing matches is the ordinary case there rather than a miss. Neither does the reading that
        # suggests itself instead — coloring by whether the name is taken, as an early warning before the
        # overwrite question is asked. Both directions of it say something false. Red would read as "there
        # is something wrong with this name", when the name is fine and it is the *write* that wants a
        # second look; green would read as approval of the one outcome the dialog stops to ask about twice.
        if self.save_mode:
            return
        query = dpg.get_value(self.search_field)
        if not query:
            color = _TEXT_NEUTRAL
        else:
            # `..` counts, and is not in `shown_items`. The listing keeps it whatever is typed, but it
            # answers a query like any other name — typing `..` puts the cursor on it, which is how going
            # up is reachable by search. Red there would deny a row that is on screen and about to work.
            found = bool(self.shown_items) or common_utils.make_search_matcher(query)(os.pardir)
            color = _TEXT_GOOD if found else _TEXT_BAD
        dpg.set_value(self._search_field_color, color)

    def _recolor_path_field(self) -> None:
        """Repaint the path field to say what Enter would do with what is typed there.

        Three states, and the middle one is why there are three: green names an existing directory, so
        Enter goes there; neutral is a path still being typed towards one; red cannot lead anywhere.
        """
        # What red predicts is the message the dialog would otherwise answer Enter with, moved to before
        # the commit and made free. The message stays as the backstop — a directory can go away between
        # the typing and the Enter — but it should be a surprise rather than the ordinary way to find out.
        typed = dpg.get_value(self.path_field)
        if not typed:
            dpg.set_value(self._path_field_color, _TEXT_NEUTRAL)
            return

        # `~` is expanded before anything is asked, because the backend expands it too: green promises that
        # Enter goes *there*, and it can only promise that about the path that will actually be opened.
        # The field keeps showing what was typed — expanding under the caret would replace a character the
        # user entered with eight of somewhere else, mid-edit.
        expanded = os.path.expanduser(typed)
        if os.path.isdir(expanded):
            # Which also answers the trailing-separator case without a rule of its own: `/some/dir/` names
            # an existing directory, so it is green, and Enter takes it whether or not anything is inside.
            dpg.set_value(self._path_field_color, _TEXT_GOOD)
            return

        # Not a directory yet, so the question becomes whether it could still become one. Splitting on the
        # last separator gives the two halves that decide it — `os.path.split` knows the platform's
        # separators, including Windows accepting both.
        parent, fragment = os.path.split(expanded)
        # A bare name is relative to the working directory, which is what `os.chdir` would do with it.
        parent = parent or os.curdir
        color = _TEXT_BAD
        if os.path.isdir(parent):
            names = self._subdirectory_names(parent)
            # Exact case, deliberately, where the find field one line below is smart-case. That field
            # *searches*, and being generous there costs nothing; this one *addresses*, and on a
            # case-sensitive filesystem a generous match would show neutral for a path that cannot be
            # completed — the one thing the color must never say.
            if any(name.startswith(fragment) for name in names):
                color = _TEXT_NEUTRAL
        dpg.set_value(self._path_field_color, color)

    def _subdirectory_names(self, parent: str) -> tuple[str, ...]:
        """The names of `parent`'s subdirectories, from a one-slot cache. Used to color the path field."""
        # Hidden directories are in here whatever the Hidden checkbox says. A dot typed into a path field is
        # an intention rather than a browsing preference, so `.conf` must not go red because a toggle
        # elsewhere is off.
        if self._path_prefix_cache is not None and self._path_prefix_cache[0] == parent:
            return self._path_prefix_cache[1]
        try:
            with os.scandir(parent) as entries:
                names = tuple(entry.name for entry in entries if entry.is_dir())
        except OSError:  # unreadable, gone, or not a directory after all
            names = ()
        self._path_prefix_cache = (parent, names)
        return names

    def _on_path_field_activated(self) -> None:
        """The path field just took the caret, by click or by Ctrl+L: record where the keys go."""
        # A click has to mean what the key means, or Enter is read twice — see the handler registry this is
        # bound from. `_focus_path_field` sets this too, and arrives here as well, which costs nothing.
        self._caret_home = CaretHome.PATH

    def _on_path_field_deactivated(self) -> None:
        """The path field lost the caret: the keys go back to the find field, unless they went elsewhere."""
        # Guarded rather than unconditional, because deactivation is also what *leaving deliberately* looks
        # like. Tab out of here sets the listing as the home and the deactivation follows a frame later, so
        # an unconditional write would silently undo it.
        if self._caret_home is CaretHome.PATH:
            self._caret_home = CaretHome.FIELD

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
        # Only where a combo shares the row. Without one the area is `width=-1` and follows the window on
        # its own, and writing a number over that would pin it to whatever width it was last resized to.
        if self._type_filter_is_available():
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

    def _type_filter_is_available(self) -> bool:
        """Whether this dialog offers the file type filter at all.

        The question is whether a file is what comes back. A dialog returning a directory chooses among
        folders, and a type filter cannot narrow those — `list_directory` applies one to files only, and
        must, since a filter that hid directories would hide the way to the files it selects. So in
        `"dir"`, which lists no files, the combo narrows an empty set; in `"dir-with-contents"` it narrows
        the scenery. Either way it is a control that cannot affect the answer.
        """
        return not self.returns_dir

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
        # `.*` is how a caller asks for the catch-all and `All files` is what it is called on screen, so a
        # request written the first way has to find the item offered under the second.
        if label == _CATCH_ALL and _CATCH_ALL_LABEL in self._filter_extensions:
            label = _CATCH_ALL_LABEL
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
            # DPG stacks no modal over a modal, so no box can be drawn while this dialog is up — which is
            # every dialog Raven opens. The notification line above the buttons says it instead: it is in
            # the window the user is already looking at, and needs nothing stacked over anything.
            #
            # Only the first line of `message` goes there. These messages are written as a sentence
            # followed by the exception text, and a status line has room for the sentence; the whole of it
            # still reaches the log, which is where an exception is of use anyway.
            logger.warning(f"message_box: shown on the target line, this dialog being modal:\n{title}:\t{message}\n")
            gui_animation.animator.add(gui_animation.WidgetFlash(target=self.text_target,
                                                                 duration=_REPORT_FLASH_SECONDS,
                                                                 message=message.split("\n", maxsplit=1)[0],
                                                                 message_duration=_REPORT_TEXT_SECONDS,
                                                                 text_color=_ALARM_RED))

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
        # Buttons rather than the table header's own sorting, which is turned off. ImGui draws the header's
        # sort arrow from its *own* state, so a sort chosen in grid view would leave the header asserting an
        # order the data no longer has — and "buttons for the grid, header clicks for the table" was the
        # first shape here for exactly that reason. Turning the header's sorting off removes the second
        # source of truth by construction rather than by keeping two of them in step.
        #
        # It costs the familiar click-the-header gesture and buys two things beyond that guarantee: sorting
        # becomes keyboard-operable, which ImGui's header sorting is not at all, and the control does not
        # move when the view does. The header's own semantics are kept — click to sort ascending, click
        # again for descending.
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
        typed = dpg.get_value(self.path_field)
        try:
            self.chdir(typed)
        except FileNotFoundError:
            self.message_box("Invalid path", f"No such file or directory: {typed}")

    def open_place(self, sender, app_data, user_data):
        """DPG GUI event handler: list the place this row stands for.

        `user_data`: The directory the row opens — one of the user's directories, or a drive's root. A
                     drive is a place: both are a name in the panel standing for somewhere to go, which
                     is all a row of it means.
        """
        place_path = user_data
        # A place is somewhere to go, not something to select, and a selectable is a toggle — so the row
        # is cleared the moment it is used, the way the listing's rows are. Leaving it lit would assert a
        # selection this panel has no concept of.
        with guiutils.nonexistent_ok():
            dpg.set_value(sender, False)
        # Mouse and keyboard agree about where the cursor is, so Ctrl+B after a click resumes from the row
        # that was clicked rather than from wherever the arrows last were.
        for idx, (cell, _base_theme, _cursor_theme) in enumerate(self._place_themes):
            if cell == sender:
                self._places_cursor.set_current(idx)
                break
        if self._caret_home is CaretHome.LISTING:
            # `chdir` hands the caret to the find field from every other home, and deliberately not from
            # this one — arriving somewhere is no reason to change modes. Which leaves DPG's focus on the
            # clicked row, outside the child window the listing's homes park in.
            self._park_focus()
        self.chdir(place_path)

    # ------------------------------------------------------------------
    # The places panel — the side list of folder shortcuts and drives
    # ------------------------------------------------------------------

    def _add_place_row(self, label: str, path: str, icon) -> None:
        """Build one row of the places panel: an icon, and a selectable that goes where the row says.

        `label`: what the row reads — a directory name for one of the user's places, the mount point for
                 a drive.
        `path`: where it goes.
        `icon`: the texture to show beside it.
        """
        # A selectable rather than a menu item, and for the keyboard rather than for the look: a
        # `menu_item` cannot hold focus and cannot be asked to — `get_item_state` on one has no "focused"
        # key at all — so a panel built from them is reachable by mouse and by nothing else. A selectable
        # is what the listing's rows are made of, which also makes this the listing's cursor rather than a
        # second kind of one.
        with dpg.group(horizontal=True):
            dpg.add_image(icon)
            cell = dpg.add_selectable(label=label, user_data=path, callback=self.open_place,
                                      height=self.selec_height)
        dpg.bind_item_theme(cell, self.selec_alignt)
        self._place_entries.append((label, path))
        self._place_themes.append((cell, self.selec_alignt, self.selec_alignt_cursor))

    def _places_are_navigable(self) -> bool:
        """Whether the places panel is on screen and has rows for a cursor to move over.

        False in the compact style, which is a strip of image buttons with no text to paint a cursor on,
        and false where the caller asked for no panel at all.
        """
        return bool(self.show_shortcuts_menu and self._place_entries)

    def _paint_place(self, idx: int, is_cursor: bool) -> None:
        """Draw place row `idx` as the cursor row, or as an ordinary one.

        The mark shows only while the panel has the keys, which is where this cursor differs from the
        listing's. That one is what Enter acts on from every home, so it is true wherever the caret is;
        this one is reachable only from inside the panel, so a blue row left behind after Escape would
        promise something Enter would not do.
        """
        if not (0 <= idx < len(self._place_themes)):
            return
        cell, base_theme, cursor_theme = self._place_themes[idx]
        lit = is_cursor and self._caret_home is CaretHome.PLACES
        with guiutils.nonexistent_ok():
            dpg.bind_item_theme(cell, cursor_theme if lit else base_theme)

    def _focus_places(self) -> None:
        """Hand the bare arrow keys to the places panel, without handing it DPG's focus.

        Esc, or Tab, or going somewhere, takes them away again.
        """
        # Focus is parked exactly where the type filter parks it, and for the same reason: which home has
        # the keys is `_caret_home`, and DPG's focus has one job here, which is to be somewhere that the
        # find field can be reached from afterwards. A places row could hold focus — that is the whole
        # point of the migration off menu items — but then focus would have to chase the cursor on every
        # move, buying nothing the cursor's own colour does not already say.
        self._caret_home = CaretHome.PLACES
        self._park_focus()

    def _open_cursor_place(self) -> None:
        """Go to the place the panel's cursor is on. Enter's meaning inside the panel."""
        # No special case for the return to the find field: `chdir` does that from every home but the
        # listing, so Enter and a click leave by the same door. And where it fails — an unreadable mount
        # is the usual way — it reports and does not move the caret, which leaves the panel holding the
        # keys with its cursor still lit, ready for another try.
        path = self._places_cursor.current_key
        if path is None:
            return
        self.chdir(path)

    def _places_height(self) -> float:
        """The visible height of the places panel. The child window is what scrolls, here and in the notes."""
        with guiutils.nonexistent_ok():
            _, height = guiutils.get_widget_size(f"shortcut_menu_{self.instance_tag}")  # tag
            return height if height > 0 else 0
        return 0

    def _place_extent(self, idx: int):
        """Where place row `idx` sits inside the panel's scrollable content, as `(top, height)`.

        Measured per row, where the listing extrapolates one pitch across all of its rows: the separator
        between the user's directories and the drives makes these rows unevenly spaced, so there is no
        single pitch that would answer for all of them. Affordable because the panel does not clip — it
        is a handful of rows, all of them submitted every frame, so each one can be asked where it is.
        """
        if not (0 <= idx < len(self._place_themes)):
            return None
        cell = self._place_themes[idx][0]
        with guiutils.nonexistent_ok():
            # `get_item_pos` answers relative to the visible area, so a panel scrolled down reports its
            # rows too high by exactly the scroll. Adding it back is what makes these content coordinates,
            # which is what the scroll position is measured in.
            _, local_top = dpg.get_item_pos(cell)
            _, height = guiutils.get_widget_size(cell)
            if height <= 0:  # not laid out yet
                return None
            return local_top + dpg.get_y_scroll(f"shortcut_menu_{self.instance_tag}"), height  # tag
        return None

    def _places_per_page(self) -> int:
        """Most of a screenful of places, keeping one row of context to read the new position against."""
        height = self._places_height()
        extent = self._place_extent(0)
        if not height or extent is None or extent[1] <= 0:
            return 1
        return max(1, int(height / extent[1]) - 1)

    def _scroll_place_into_view(self, idx: int) -> None:
        """Move the least that puts place row `idx` on screen, and nothing at all when it already is."""
        extent = self._place_extent(idx)
        height = self._places_height()
        panel = f"shortcut_menu_{self.instance_tag}"  # tag
        if extent is None or not height:
            logger.debug(f"_scroll_place_into_view: instance '{self.tag}' ({self.instance_tag}), "
                         f"row {idx}: no geometry (extent={extent}, panel height={height})")
            return
        row_top, row_height = extent
        with guiutils.nonexistent_ok():
            view_top = dpg.get_y_scroll(panel)
            if row_top < view_top:
                new_top = row_top
            elif row_top + row_height > view_top + height:
                new_top = row_top + row_height - height
            else:
                return
            dpg.set_y_scroll(panel, max(0.0, float(new_top)))

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
            if self._caret_home is not CaretHome.LISTING:
                self._focus_field()
        # Each of these names what it failed on. The report outlives the moment — it stays for a few
        # seconds, and the next thing the user does usually succeeds — so a message saying only "that
        # folder" ends up sitting over a listing of somewhere else entirely, describing nothing on screen.
        except PermissionError as e:
            self.message_box("File dialog - access denied", f"Cannot open {path} — access denied.\n\nMore info:\n{e}")
        except NotADirectoryError as e:
            self.message_box("File dialog - not a directory", f"{path} is a file, not a folder.\n\nMore info:\n{e}")

    def reset_dir(self, file_name_filter=None):
        """Rebuild the listing of the working directory, optionally narrowed to `file_name_filter`.

        This *lists*; it does not navigate. Going somewhere is `chdir`, which moves the process and
        then calls this.
        """
        # What this is a listing *of*, and the one thing here that is not `self.default_path` — that is the
        # folder the caller wants each opening to start from, which is a different question and one this
        # never asks.
        #
        # Read from the process rather than accepted as an argument, so the two cannot disagree: `ok`
        # and the target notification both answer from `os.getcwd()`, and a listing built from
        # anywhere else would be a dialog that shows you A and hands back B. Already absolute, that being
        # what `getcwd` returns.
        listed_dir = os.getcwd()
        logger.debug(f"reset_dir: instance '{self.tag}' ({self.instance_tag}), called with file_name_filter = {file_name_filter}, listing '{listed_dir}'")
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

        # The views are told which directory this is, and work out the rest themselves: a rebuild of the
        # same directory and a move to a different one look identical from here, and the cursor wants
        # opposite things from them — hold its place across a re-filter, start at the top of somewhere new.
        try:
            # Only when it would actually change. A rebuild happens on every keystroke in the find field
            # and on every Tab, and the directory is the same for all of them — reconfiguring the widget
            # each time is churn nobody asked for, in the one field the user is not interacting with.
            if dpg.get_value(self.path_field) != listed_dir:
                dpg.configure_item(self.path_field, default_value=listed_dir)
            # Back to saying nothing, whatever a draft left it saying. The three states answer what the
            # *user* asked for, and this is the dialog showing where they now are — which is not a question,
            # so it gets no answer. Green at rest would leave the field permanently green in ordinary use,
            # and a color that is always on reports nothing when it matters.
            #
            # Outside the rewrite guard above, and that is the whole difference: typing a path in full and
            # pressing Enter arrives somewhere the field *already* names, so there is nothing to rewrite —
            # and the green from the typing would be left standing over a navigation that has finished.
            dpg.set_value(self._path_field_color, _TEXT_NEUTRAL)
            # A listing was just read, so anything cached about a half-typed path in this directory is from
            # before it. Cheap to drop and re-read on the next keystroke; wrong to keep across an F5, which
            # is what a user presses when they believe the folder has changed under them.
            self._path_prefix_cache = None
            # Compiled once per rebuild rather than per entry: on a directory of thousands, the split is
            # the part worth hoisting out of the loop.
            matches_name_filter = common_utils.make_search_matcher(file_name_filter or "")

            # Enumerating, filtering and sorting all happen here, on data, before a widget is touched.
            with timer() as tim_list:
                entries = filelisting.list_directory(listed_dir,
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
            logger.error(f"reset_dir: instance '{self.tag}' ({self.instance_tag}), invalid path: '{listed_dir}'")
        except Exception as exc:
            # Logged with its traceback *before* the message box, which shows only `str(exc)`. A listing
            # error is otherwise reduced to one line with no stack — and where the dialog is modal the
            # box cannot even be shown, so the line goes to the log stripped of everything that would
            # locate it. Cost a CI round on a Windows-only failure that said "negative dimensions are
            # not allowed" and nothing about where.
            logger.exception(f"reset_dir: instance '{self.tag}' ({self.instance_tag}), failed to list '{listed_dir}'")
            self.message_box("File dialog - listing failed", f"Could not list {listed_dir}.\n\nMore info:\n{exc}")

        # Every path into here changes something the promised target depends on — which directory is
        # shown, and what survives the find field — so this is the one place that has to refresh it.
        # The find field's color reads off the same two, and follows it for the same reason.
        self._refresh_target_notification()
        self._recolor_find_field()

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

    def _help_hotkey_info(self):
        """The keys this dialog answers to, as `helpcard` entries, in reading order.

        Built per instance rather than as a constant, because several keys exist only in some dialogs —
        marking a selection needs a dialog that takes more than one file, the grid needs one that lists
        files at all, and the two type-filter keys need one that returns a file. A card offering a key that
        does nothing here is worse than one that stays quiet about it, and the dialog's shape is fixed at
        construction, so the question can be answered once.

        Keys are spelled out as words. The UI font is OpenSans, which has no arrow glyphs at all, so "↑" in
        a label renders as a missing-glyph box.
        """
        typing_finds = ("Name the file to save as" if self.save_mode else "Find in this folder")
        return [entry for entry in (
            # --- Column 1: moving through the listing, and acting on what the cursor is on ---
            env(key_indent=0, key="Up / Down", action_indent=0, action="Move the cursor one row", notes=""),
            env(key_indent=0, key="Page Up / Page Down", action_indent=0, action="Move about a screenful", notes=""),
            env(key_indent=0, key="Home / End", action_indent=0, action="First / last entry", notes=""),
            env(key_indent=0, key="Left / Right", action_indent=0, action="Previous / next entry", notes="Once Tab is in the listing"),
            helpcard.hotkey_blank_entry,
            env(key_indent=0, key="Enter", action_indent=0, action="Go as deep as this entry allows", notes="Into a folder, or accept a file"),
            env(key_indent=0, key="Ctrl+Enter", action_indent=0, action="Accept without going deeper", notes="The OK button"),
            # Two homes answer Escape by handing the caret back rather than cancelling, and a bare "Cancel"
            # promises something else for both. It goes in the notes rather than a row of its own, and it is
            # kept short enough not to wrap, for the same reason: column one is already the height the card
            # was measured at, `HelpWindow` gives it no scrollbar, and either a fifteenth row or a second
            # line here spends the margin the measurement left for the row a multi-selection dialog adds.
            env(key_indent=0, key="Esc", action_indent=0, action="Cancel", notes="Or out of a side control"),
            (env(key_indent=0, key="Ctrl+Space", action_indent=0, action="Mark or unmark this entry", notes="Ctrl+click, without the mouse")
             if self.multi_selection else None),
            helpcard.hotkey_blank_entry,
            env(key_indent=0, key="Alt+Up", action_indent=0, action="Up one level", notes=""),
            env(key_indent=1, key="Ctrl+Up", action_indent=1, action="...the same, one-handed", notes=""),
            env(key_indent=0, key="Ctrl+Home", action_indent=0, action="Back to the starting folder", notes=""),
            env(key_indent=0, key="F5", action_indent=0, action="Re-read this folder", notes=""),

            helpcard.hotkey_new_column,

            # --- Column 2: the text field, and what the listing shows ---
            env(key_indent=0, key="Type anything", action_indent=0, action=typing_finds, notes="Fragments, in any order"),
            env(key_indent=0, key="Tab", action_indent=0, action="Caret to the listing", notes="Completing what you typed"),
            env(key_indent=1, key="Tab", action_indent=1, action="...and back, carrying the name", notes="Of whatever the cursor is on"),
            env(key_indent=0, key="Ctrl+F", action_indent=0, action="Caret back to the field", notes="Keeping what you typed"),
            env(key_indent=0, key="Ctrl+L", action_indent=0, action="Caret to the path field", notes="Paste a folder, Enter to go"),
            # Column two now matches column one's fourteen rows at its longest, which is the height the
            # card was measured at — so this is the last row either column has room for.
            (env(key_indent=0, key="Ctrl+B", action_indent=0, action="Caret to the shortcuts panel", notes="Arrow to one, Enter to go")
             if self._places_are_navigable() else None),
            helpcard.hotkey_blank_entry,
            (env(key_indent=0, key="Ctrl+1 ... Ctrl+9", action_indent=0, action="Show the Nth file type", notes="")
             if self._type_filter_is_available() else None),
            (env(key_indent=1, key="Ctrl+Shift+F", action_indent=1, action="...or browse them", notes="Up / Down / Home / End")
             if self._type_filter_is_available() else None),
            env(key_indent=0, key="Ctrl+Shift+1 ... Ctrl+Shift+4", action_indent=0, action="Sort by name / date / type / size", notes="Again to reverse"),
            env(key_indent=0, key="Ctrl+H", action_indent=0, action="Show or hide hidden files", notes=""),
            (env(key_indent=0, key="Ctrl+T", action_indent=0, action="Thumbnails, or the list", notes="")
             if self._grid_is_available() else None),
            helpcard.hotkey_blank_entry,
            env(key_indent=0, key="F1", action_indent=0, action="This card", notes=""),
        ) if entry is not None]

    def _the_help_card(self):
        """The card listing this dialog's keys, built on first use.

        Deferred because most dialogs are never asked for it, and building one costs a window and a table
        of some two dozen rows.
        """
        if self._help_window is None:
            self._help_window = helpcard.HelpWindow(hotkey_info=self._help_hotkey_info(),
                                                    width=_HELP_CARD_SIZE[0],
                                                    height=_HELP_CARD_SIZE[1],
                                                    # Centering is on the reference window's *size*, taken
                                                    # from the origin, so this lands the card on the middle
                                                    # of the dialog less the dialog's own position — 50 px
                                                    # up and left of where the dialog sits. Near enough that
                                                    # the card appears where the dialog was, which is where
                                                    # the reader is looking.
                                                    reference_window=self.tag,  # tag
                                                    # All `HelpWindow` wants of this is a spacer height. 20
                                                    # is the constellation's GUI font size, the figure
                                                    # `min_size` is measured at and the one the grid view
                                                    # takes by default.
                                                    themes_and_fonts=env(font_size=20),
                                                    # The dialog is hidden while the card is up, so the
                                                    # title is the only thing left saying whose keys these
                                                    # are.
                                                    label=f"{self.title} — keyboard",
                                                    handle_own_hotkeys=False,
                                                    on_hide=self._on_help_card_hidden)
        return self._help_window

    def _show_help_card(self) -> None:
        """F1: put the dialog away and show the card of its keys.

        Away rather than behind, because DPG will not stack a modal over a modal: shown while the dialog is
        up, the card would never appear, and no error would say so.
        """
        self._help_card_up = True  # set first, so `is_visible` never dips while the swap is in progress
        dpg.hide_item(self.tag)  # tag
        # Hiding is not immediate, and the wait is load-bearing rather than cosmetic. A window leaves
        # ImGui's popup stack only once a frame has drawn without it, and a modal opened while another is
        # still on that stack never appears — DPG then treats the card as closed and fires its close
        # handler, so it undoes itself some 80 ms after F1 with nothing in the log to say why.
        guiutils.split_frame(operation="file dialog: making room for the help card", required=True)
        if not self._the_help_card().show():
            # The card declines to exist during an app's first few frames. Nothing is on screen at that
            # point that could have pressed F1, but a dialog left hidden with no card over it would be a
            # dead app, so put it back rather than reason about who could reach this.
            logger.info(f"_show_help_card: instance '{self.tag}' ({self.instance_tag}), the card was not built; restoring the dialog.")
            self._help_card_up = False
            dpg.show_item(self.tag)  # tag

    def _on_help_card_hidden(self) -> None:
        """The card has closed — by Esc, or by its own close button. Bring the dialog back."""
        self._help_card_up = False
        # Not while Escape is still down. ImGui dismisses the topmost modal popup on Escape by itself, and
        # this dialog's close handler is `cancel` — so a dialog put back under a held key is dismissed the
        # frame it appears, and the picker returns nothing having been open a moment ago. A *tap* is over
        # before the dialog draws, which is why this shows up under a real press and not under a driven one.
        if dpg.is_key_down(dpg.mvKey_Escape):
            self._restore_pending = True
            return
        self._restore_dialog_window()

    def _restore_dialog_window(self) -> None:
        """Put the dialog back on the screen, the card that replaced it having gone."""
        self._restore_pending = False
        dpg.show_item(self.tag)  # tag
        # The caret went nowhere while the dialog was hidden, but focus did: the card took it. Return it to
        # whichever home the dialog was in, so F1 costs nothing but the reading.
        if self._caret_home is CaretHome.LISTING:
            self._focus_listing()
        else:
            self._focus_field()

    def _handle_key_release(self, key: int) -> None:
        """Handle one key release for this dialog. Only one key is waited on, and only sometimes.

        Escape leaving the keyboard is what lets a dialog come back after its help card: see
        `_on_help_card_hidden` for why it may not come back before then.
        """
        if key == dpg.mvKey_Escape and self._restore_pending:
            self._restore_dialog_window()

    def _handle_key(self, key: int) -> None:
        """Handle one key press for this dialog. Called by the module-level handler, which owns the
        registry and decides *which* dialog is listening; this decides what the key does.
        """
        ctrl = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)

        # While the card is up it is the only thing on screen, so every key belongs to it — and this is the
        # only handler that will act on one, the card being built with `handle_own_hotkeys=False`. That is
        # what keeps Escape from being read twice: closing the card here means it cannot also reach the
        # branch below that cancels the dialog.
        if self._help_card_up:
            if key == dpg.mvKey_Escape:
                self._help_window.hide()
            return
        if self._restore_pending:
            return  # between the card and the dialog, with neither on the screen to act on

        # TODO (briefs/researchers-night/filedialog-keyboard-brief.md): the navigation history —
        # TODO: Alt+Left / Alt+Right, Ctrl+Left / Ctrl+Right in the listing, and the mouse's own back
        # TODO: and forward buttons.
        shift = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
        alt = dpg.is_key_down(dpg.mvKey_LAlt) or dpg.is_key_down(dpg.mvKey_RAlt)

        # Tab swaps the caret's two homes. ImGui does not spend Tab on an `InputText` — it neither
        # moves focus nor inserts anything — so the key is ours to define, and this is the only way
        # to reach the state where the find field is inactive.
        if key == dpg.mvKey_Tab:
            if self._caret_home is CaretHome.LISTING:
                # Written before the caret returns, the field being writable only while it does not have
                # it — the mirror of the outbound order below.
                self._fill_field_from_cursor()
                self._focus_field()
            elif self._caret_home is CaretHome.FIELD:
                # Order is load-bearing: the completion is a write to the find field, and a field with the
                # caret in it reverts one. Leaving is what makes the write possible, so the caret goes
                # first and the completion follows it out.
                self._focus_listing()
                self._complete_find_field()
            else:
                # Parked on a control. Tab still means the listing, but there is no completion to apply:
                # completing is what leaving the *find field* does, and that is not where the caret is.
                self._focus_listing()
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
            elif self._type_filter_is_available() and n < len(self._filter_labels):
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

        # Bare keys belong to whichever home the caret is in, and the type filter's are the four that
        # browse a combo plus the one that gives the caret back. Modified keys are not the home's business:
        # Ctrl+H and F5 mean the same wherever the caret is parked.
        if self._caret_home is CaretHome.FILTER and not (ctrl or alt or shift):
            if key in (dpg.mvKey_Up, dpg.mvKey_Down, dpg.mvKey_Home, dpg.mvKey_End):
                self._browse_type_filter(key)
                return
            if key == dpg.mvKey_Escape:
                # Escape hands the caret back to the find field from wherever it was parked, and cancels
                # the dialog only once it is already there. Nothing is restored on the way out, unlike a
                # draft in the path field: the combo applied every step as it was made, and reverting would
                # undo a change the user watched happen and kept going past.
                self._focus_field()
                return

        # The path field is the one home that is a text field with the caret really in it, so most of the
        # bare keys are ImGui's and this dialog's job is to keep its hands off them: Home and End move
        # within the text, and Enter commits through the field's own `on_enter` callback. Enter is the one
        # that must not fall through — measured on DPG 2.3.1, the global handler runs *before* that
        # callback, so a press reaching both would descend into whatever the listing cursor is on and
        # navigate to the typed path from there, opening whatever it passed through on the way.
        if self._caret_home is CaretHome.PATH and not (ctrl or alt or shift):
            if key == dpg.mvKey_Escape:
                self._abandon_path_draft()
                return
            if key in (dpg.mvKey_Up, dpg.mvKey_Down, _KEY_PAGE_UP, _KEY_PAGE_DOWN,
                       dpg.mvKey_Home, dpg.mvKey_End, dpg.mvKey_Return):
                return

        # The places panel needs no branch for the six movement keys — it is the same cursor class over
        # different rows, so picking it below is enough. Only the two that *leave* differ, and both are
        # the universal rules rather than anything this panel invents: Enter goes as deep as it can, and
        # from in here the deepest thing there is is the place under the cursor; Escape gives the caret
        # back to the main thing.
        if self._caret_home is CaretHome.PLACES and not (ctrl or alt or shift):
            if key == dpg.mvKey_Return:
                self._open_cursor_place()
                return
            if key == dpg.mvKey_Escape:
                # Nothing to restore on the way out, as with the type filter: moving this cursor changes
                # only where the cursor is, so there is no draft left standing to abandon.
                self._focus_field()
                return

        # Which cursor the movement keys drive. The listing's view, unless the caret is parked on the
        # places panel — not `_navigator`, which answers *which view of the listing is up* and is asked
        # that question by the rebuild too, where the answer must stay the listing's whatever has the keys.
        nav = self._places_cursor if self._caret_home is CaretHome.PLACES else self._navigator()
        if key == dpg.mvKey_Up:
            nav.navigate_row_up()
        elif key == dpg.mvKey_Down:
            nav.navigate_row_down()
        elif key == _KEY_PAGE_UP:
            nav.navigate_page_up()
        elif key == _KEY_PAGE_DOWN:
            nav.navigate_page_down()
        elif key == dpg.mvKey_Left and self._caret_home is CaretHome.LISTING:
            # Left and Right are not unwanted while the caret is in the find field, they are
            # *occupied* — a single-line entry spends them on the text caret. Tab is what frees them,
            # which is why the grid is only now completely reachable: its rows hold several tiles, so
            # without a horizontal step every column but the first was unvisitable from the keyboard.
            nav.navigate_prev()
        elif key == dpg.mvKey_Right and self._caret_home is CaretHome.LISTING:
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
        elif key == dpg.mvKey_F1:
            self._show_help_card()
        elif ctrl and key == dpg.mvKey_Home:
            self.back_to_default_path()
        elif ctrl and key == dpg.mvKey_F:
            # Ctrl+F narrows the listing by name fragment; Ctrl+Shift+F narrows it by type. Same listing,
            # two filters, and the mnemonic is carried by what the key *does* rather than by a label — the
            # combo is labelled `Show`, so a key named after the label would name the wrong control, and
            # nothing has to be re-learned if the label changes again.
            # The shifted chord is a toggle, and that is what makes the pair usable rather than a nicety.
            # The two share a letter, so the way to reach for Ctrl+F right after Ctrl+Shift+F is to keep the
            # modifiers down and press F again — which is the shifted chord a second time. Modifier state is
            # read when the handler runs, so a Shift still held from a moment ago is indistinguishable from
            # one meant, and the caret would silently stay on the filter. Toggling means the second press
            # lands in the field whether or not the finger left Shift.
            #
            # Where there is no filter to browse, the shifted chord falls through to the plain one rather
            # than doing nothing: Ctrl+Shift+F is then a Ctrl+F pressed with a finger still on Shift.
            if shift and self._caret_home is not CaretHome.FILTER and self._type_filter_is_available():
                self._focus_type_filter()
            else:
                self._focus_field()
        elif ctrl and key == dpg.mvKey_L:
            # What every browser and file manager binds to the address bar, and it means the same here.
            # Deliberately without completion: writing one is the easy half and handing the field *back*
            # has no answer — refocusing an `InputText` arms ImGui's select-all, which DPG exposes no way
            # to clear, so the next character would replace the whole path. The find field accepts that
            # trade because a lost query is a few characters retyped; a completed path is the entire thing
            # the user was building.
            #
            # Which costs nothing, because this field is not for typing paths into. The dialog already
            # completes them, better, through the find field — a fragment, Enter, repeat, and being
            # fragment-based and smart-case it beats prefix completion at its own job. What only this can
            # do is take a whole path from somewhere else (Ctrl+L, Ctrl+V, Enter) and reach a short root
            # like `/mnt` that is nowhere near here and in nobody's places panel.
            self._focus_path_field()
        elif ctrl and key == dpg.mvKey_B:
            # `B` for bookmarks, which is what a browser and GTK's own file chooser call this panel.
            #
            # It is the counterpart of Ctrl+L rather than a duplicate of it: that one reaches anywhere at
            # the cost of typing the whole path, this one reaches the handful of somewheres worth a row of
            # their own, in two keys and an arrow. Between them they cover *starting over from elsewhere*,
            # which the find field cannot do at all — it narrows what is here.
            #
            # Silently ignored where the panel is not on offer, as Ctrl+T is where the grid is not: a key
            # that acts on a control nobody can see is worse than one that does nothing.
            if self._places_are_navigable():
                self._focus_places()
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

    def _write_field(self, widget: Union[str, int], name: str, text: str) -> bool:
        """Put `text` into text field `widget`, called `name` in the log. Returns whether the write landed.

        Only safe once the caret has left that field, which is the caller's job to have arranged.
        """
        # ImGui's edit buffer owns an *active* `InputText`: `set_value` appears to work — `get_value`
        # immediately after reports the new string — and the next frame writes the old buffer back, firing
        # the edit callback as it goes. So the write has to wait for the field to go inactive, which a
        # queued focus change does not achieve on the calling frame. How many frames it takes depends on
        # what else is in flight, so this polls rather than counting frames.
        for _ in range(_FIELD_DEACTIVATION_FRAMES):
            if not dpg.is_item_active(widget):
                break
            if not guiutils.split_frame(operation=f"file dialog: waiting for the {name} to go inactive",
                                        required=False):
                return False  # no render loop to wait for; nothing would land anyway
        else:
            logger.warning(f"_write_field: instance '{self.tag}' ({self.instance_tag}): the {name} "
                           f"is still active after {_FIELD_DEACTIVATION_FRAMES} frames; not writing '{text}'")
            return False

        dpg.set_value(widget, text)
        return True

    def _write_find_field(self, text: str) -> bool:
        """Put `text` in the find field and re-filter the listing. Returns whether the write landed."""
        if not self._write_field(self.search_field, "find field", text):
            return False
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

    def _get_caret_home(self) -> CaretHome:
        """Which home currently has the keys."""
        return self._caret_home_now

    def _set_caret_home(self, home: CaretHome) -> None:
        """Move the caret's home, repainting whatever marks depend on which one it is."""
        # A property, where every other piece of this state is a plain attribute, because the places
        # cursor is drawn only while that panel has the keys — and the panel is left by nine different
        # routes (Escape, Enter, a click, Tab, Ctrl+F, Ctrl+L, Ctrl+Shift+F, a numbered filter, Ctrl+Home).
        # Repainting at each of them is nine chances to forget one and leave a blue row behind claiming
        # Enter would go there; repainting *here* is the one place all nine already pass through.
        if home is self._caret_home_now:
            return
        was_places = self._caret_home_now is CaretHome.PLACES
        self._caret_home_now = home
        if was_places or home is CaretHome.PLACES:
            self._paint_place(self._places_cursor.current, True)

    _caret_home = property(fget=_get_caret_home, fset=_set_caret_home,
                           doc="Where this dialog is taking keys. See `CaretHome`.")

    def _focus_field(self) -> None:
        """Put the caret back in the find field, where typing filters the listing."""
        self._caret_home = CaretHome.FIELD
        dpg.focus_item(self.search_field)

    def _focus_path_field(self) -> None:
        """Put the caret in the path field, where a folder is pasted or a short root typed.

        Esc, or Tab, takes it away again.
        """
        # This home does get DPG's focus, unlike the type filter: it is a real text field, and the caret has
        # to be in it for anything to be typed or pasted. Safe to focus for the reason the find field is —
        # both live in the listing's child window, which is the side `focus_item` can reach from.
        self._caret_home = CaretHome.PATH
        dpg.focus_item(self.path_field)

    def _abandon_path_draft(self) -> None:
        """Escape from the path field: put back where we actually are, and hand the caret to the find field."""
        # Restored, where escaping the type filter restores nothing — and the two are consistent rather
        # than divergent. The combo applied every step as it was made, so there is nothing left to abandon;
        # this field is a *draft* until Enter, and a half-typed path left standing over a listing of
        # somewhere else is a field that lies about where you are.
        #
        # Written after the caret has been asked to leave, an active `InputText` owning ImGui's edit buffer
        # and reverting a write on the next frame. `_write_field` waits for it.
        self._focus_field()
        self._write_field(self.path_field, "path field", os.getcwd())
        # Neutral, not green, for the same reason arriving somewhere is: the three states answer what was
        # asked for, and abandoning a draft asks nothing. `set_value` fires no edit handler, so this is the
        # only thing that would repaint it.
        dpg.set_value(self._path_field_color, _TEXT_NEUTRAL)

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
        self._caret_home = CaretHome.LISTING
        self._park_focus()

    def _start_cursor_pulse(self) -> None:
        """Set the table cursor breathing, for as long as this dialog is on screen."""
        # Only while it is open, rather than for the app's lifetime. The animation is ambient — it never
        # ends on its own and says nothing is happening — but it still has to be *drawn*, so leaving one
        # registered after the dialog closes would keep the app rendering a mark nobody can see.
        #
        # The grid's cursor is a drawn border, which no theme reaches, so `ThumbnailGrid` breathes its own
        # and every app showing a grid gets it. Started and stopped alongside this one so that a dialog that
        # has been in grid view leaves nothing running behind it either.
        if self._grid is not None:
            self._grid.start_cursor_pulse()
        if self._cursor_pulse is not None:
            return
        self._cursor_pulse = gui_animation.animator.add(
            gui_animation.PulsatingColor(cycle_duration=keyboardmark.PULSE_SECONDS,
                                         theme_color_widget=self.cursor_color_widgets))

    def _stop_cursor_pulse(self) -> None:
        """Stop the cursor breathing, and leave it at full strength."""
        if self._grid is not None:
            self._grid.stop_cursor_pulse()
        if self._cursor_pulse is None:
            return
        gui_animation.animator.cancel(self._cursor_pulse)
        self._cursor_pulse = None
        # The animation leaves behind whatever alpha it wrote last, and the themes belong to the class
        # rather than to this dialog — so a half-faded cursor would be what the next dialog to use them
        # starts from.
        for theme_color_widget in self.cursor_color_widgets:
            dpg.set_value(theme_color_widget, keyboardmark.COLOR)

    def _park_focus(self) -> None:
        """Take the caret out of the find field, leaving DPG's focus somewhere harmless."""
        # The same target serves every home that is not the field, for the reason spelled out in
        # `_focus_listing`: it has to be inside the listing's child window or the caret can never be
        # brought back. Which home has the keys is decided by `_caret_home`, not by what DPG considers
        # focused, so there is nothing for the focus itself to say.
        dpg.focus_item(self.button_refresh)

    def _focus_type_filter(self) -> None:
        """Hand the bare arrow keys to the file type combo, without handing it DPG's focus.

        Esc, or Tab, takes them away again.
        """
        # The combo does not get DPG's focus, and every home here is the same way: which one has the keys
        # is `_caret_home`, and DPG's focus is parked somewhere harmless and inside the listing's child
        # window. One answer to "where do the arrows go", rather than that answer plus whatever
        # `get_focused_item` happens to say — which is also why the flag is not derived from it.
        #
        # DPG combos have no keyboard operation of their own, so the arrows are this dialog's to route
        # either way. What focus would buy is a highlight saying where the keys went, and that mark is owed
        # to every home equally, so it is one job rather than a freebie for this one.
        self._caret_home = CaretHome.FILTER
        self._park_focus()

    def _browse_type_filter(self, key: int) -> None:
        """Move the type filter by one of Up / Down / Home / End, and apply the result immediately."""
        # The Raven combo idiom, copied rather than invented: a hotkey hands a combo the arrows, and the
        # choices are stepped through with them. `raven-avatar-settings-editor` is the reference.
        #
        # Applying on every press is the point rather than a shortcut — watching the listing re-filter as
        # you go is how the right filter gets found — and it is also why Escape restores nothing here.
        labels = self._filter_labels
        if not labels:
            return
        index = labels.index(self.file_filter) if self.file_filter in labels else 0
        if key == dpg.mvKey_Down:
            index = min(index + 1, len(labels) - 1)
        elif key == dpg.mvKey_Up:
            index = max(index - 1, 0)
        elif key == dpg.mvKey_Home:
            index = 0
        else:  # End
            index = len(labels) - 1
        self.set_type_filter(labels[index])

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
        self._start_cursor_pulse()

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

        Two things: the cursor's pulsation, which is registered with the process-wide animator and would
        outlive both this dialog and the DPG context its theme colours belong to, and — if the grid view was
        ever switched to — a thumbnail decoder with its own threads and the tick thread that feeds it.
        """
        self._stop_cursor_pulse()
        self._stop_grid_ticker()
        if self._grid is not None:
            self._grid.destroy()
            self._grid = None

    def is_visible(self):
        """Return whether the dialog is currently on screen — its help card included.

        Apps ask this to suppress hotkeys and drops while a modal picker is up. Having it here is what keeps
        `tag` from having to be known outside the constructor that set it.

        F1 swaps the dialog's window for the card's, the two being modal and DPG stacking no such pair. The
        answer must not change across that swap: an app that saw "no picker up" would re-enable its own
        hotkeys and file drops with the card sitting on the screen.
        """
        return self._help_card_up or self._restore_pending or dpg.is_item_visible(self.tag)  # tag

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
        and it must not be running when the app tears the DPG context down. So does the cursor pulsation,
        for the same reason at a smaller scale — it is a frame's worth of work per frame, spent drawing a
        mark on a window nobody is looking at.
        """
        self.selected_files.clear()
        self.shown_items.clear()
        # Released in the reverse of the order `show_file_dialog` acquires them.
        self._stop_cursor_pulse()
        self._stop_grid_ticker()

    def refresh(self):
        cwd = os.getcwd()
        logger.debug(f"refresh: instance '{self.tag}' ({self.instance_tag}), refreshing at cwd = '{cwd}'")
        self.reset_dir()
        # Raven: Acknowledge the action in the GUI.
        gui_animation.flash_button(button=self.button_refresh,
                                   duration=1.0)

    def back_to_default_path(self):
        logger.debug(f"back_to_default_path: instance '{self.tag}' ({self.instance_tag}), going back to '{self.default_path}'")
        self.chdir(self.default_path)
        # Raven: Acknowledge the action in the GUI.
        gui_animation.flash_button(button=self.button_back_to_default_path,
                                   duration=1.0)

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
                    gui_animation.animator.add(gui_animation.WidgetFlash(target=self.btn_ok,
                                                                         duration=1.0,
                                                                         message="Please enter a filename",
                                                                         message_target=self.text_notification,
                                                                         flash_color=_ALARM_RED,
                                                                         text_color=(255, 255, 255)))
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
                        gui_animation.animator.add(gui_animation.WidgetFlash(target=self.btn_ok,
                                                                             duration=1.0,
                                                                             message="Please select an item",
                                                                             message_target=self.text_notification,
                                                                             flash_color=_ALARM_RED,
                                                                             text_color=(255, 255, 255)))
                        return
                else:
                    logger.debug(f"ok: instance '{self.tag}' ({self.instance_tag}), no items shown (maybe nothing matches the search?); rejecting the ok.")
                    if self.multi_selection:
                        msg = "Please select at least one item"
                    else:
                        msg = "Please select an item"
                    gui_animation.animator.add(gui_animation.WidgetFlash(target=self.btn_ok,
                                                                         duration=1.0,
                                                                         message=msg,
                                                                         message_target=self.text_notification,
                                                                         flash_color=_ALARM_RED,
                                                                         text_color=(255, 255, 255)))
                    return
        assert len(self.selected_files)  # at least one file selected if we get here

        # Save mode: Ensure presence of file extension.
        #
        # Before the overwrite check below, and that is the load-bearing part rather than an accident of
        # where it was written. What gets written is the completed name, so that is what the guard has to
        # ask about: reversed, a user who types `portrait` where `portrait.png` exists is asked about
        # `portrait`, which does not exist, and the confirmation is skipped in exactly the case a user who
        # lets the dialog name the file for them will be in.
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
            # A folder is not overwritten — it is written into, and whether that merges with what is
            # already there or replaces it is the calling app's decision. The dialog hands back a path and
            # nothing more, so the folder wording states what exists and stops there.
            already_there = ("Folder exists — press again to confirm" if os.path.isdir(self.selected_files[0]) else
                             "Press again to overwrite file")
            # Raven: Acknowledge the action in the GUI.
            gui_animation.animator.add(gui_animation.WidgetFlash(target=self.btn_ok,
                                                                 duration=confirm_duration,
                                                                 message=already_there,
                                                                 message_target=self.text_notification,
                                                                 flash_color=_ALARM_RED,
                                                                 text_color=(255, 255, 255)))
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
