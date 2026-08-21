"""Generic single-screen GUI help window for DPG apps, intended mainly as a hotkey reference."""

__all__ = ["hotkey_new_column", "hotkey_blank_entry",
           "HelpWindow"]

import logging
logger = logging.getLogger(__name__)

import itertools
import threading
from typing import Callable, List, Optional, Tuple, Union
import uuid

from unpythonic import sym
from unpythonic.env import env

import dearpygui.dearpygui as dpg

from ...vendor import DearPyGui_Markdown as dpg_markdown  # https://github.com/IvanNazaruk/DearPyGui-Markdown

from . import utils as guiutils

# --------------------------------------------------------------------------------

hotkey_new_column = sym("next_column")
hotkey_blank_entry = env(key_indent=0, key="", action_indent=0, action="", notes="")

# Hotkey support
visible_help_window_instance = None  # fdialog is modal so There Can Be Only One (TM). If needed, could use a list, and check which one has keyboard focus, but that might not always work.
def helpcard_hotkeys_callback(sender, app_data):
    if visible_help_window_instance is None:
        return
    if not visible_help_window_instance.handle_own_hotkeys:
        return  # this card's owner routes keys to it, so Escape reaches one handler rather than two

    key = app_data  # for documentation only
    # shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
    # ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)

    if key == dpg.mvKey_Escape:
        visible_help_window_instance.hide()
    return

class HelpWindow:
    _class_init_lock = threading.Lock()  # thread-safe global setup
    _class_initialized = False

    @classmethod
    def _initialize_class(cls) -> None:
        with cls._class_init_lock:
            # The registry belongs to the DPG context that created it, and `dpg.destroy_context` takes it
            # along while leaving this flag set — so ask the context whether it is still there rather than
            # trusting the flag alone. An app holds one context for its whole life and never meets this; a
            # test suite meets it on the second context it builds, and the symptom is a card that quietly
            # stops answering Esc.
            if cls._class_initialized and dpg.does_item_exist("helpcard_handler_registry"):  # tag
                return
            cls._class_initialized = True

            # register our hotkey handler
            with dpg.handler_registry(tag="helpcard_handler_registry"):  # global (whole viewport)
                dpg.add_key_press_handler(tag="helpcard_hotkeys_handler", callback=helpcard_hotkeys_callback)

    def __init__(self,
                 hotkey_info: List[env],
                 width: int,
                 height: int,
                 reference_window: Union[str, int],
                 themes_and_fonts: env,
                 highlight_color: Tuple[int] = (255, 0, 0, 255),
                 heading_color: Tuple[int] = (255, 255, 255, 255),
                 text_color: Tuple[int] = (180, 180, 180, 255),
                 dimmed_color: Tuple[int] = (140, 140, 140, 255),
                 gui_font: Optional[int] = None,
                 on_render_extras: Optional[Callable] = None,
                 on_show: Optional[Callable] = None,
                 on_hide: Optional[Callable] = None,
                 label: str = "Help",
                 handle_own_hotkeys: bool = True):
        """Set up the help window. You only need one instance per app (or per main view, which has different hotkeys).

        `hotkey_info`: The main part of the help window is a human-readable hotkey table, created from this.

            - Entries are listed in human reading order, column first.

            - Each entry is an `unpythonic.env.env` (a fancy namespace). The format for one entry is::

                  env(key_indent=0, key="Ctrl+I", action_indent=0, action="Import BibTeX files", notes="Use this to create a dataset")

              You can group hotkeys using the `indent` parameters::

                  env(key_indent=0, key="Ctrl+F", action_indent=0, action="Focus search field", notes=""),
                  env(key_indent=1, key="Enter", action_indent=0, action="Select search matches, and unfocus", notes="When search field focused"),
                  env(key_indent=2, key="Shift+Enter", action_indent=1, action="Same, but add to selection", notes="When search field focused"),
                  env(key_indent=2, key="Ctrl+Enter", action_indent=1, action="Same, but subtract from selection", notes="When search field focused"),
                  env(key_indent=2, key="Ctrl+Shift+Enter", action_indent=1, action="Same, but intersect with selection", notes="When search field focused"),

            - To start a new column (other than the first one), use the constant `helpcard.hotkey_new_column` as an entry.

            - To leave an empty row in the current column, use the constant `helpcard.hotkey_blank_entry` as an entry.
              This is useful to visually separate hotkey groups.

            - The help columns don't have to be the same length.

        `width`: Width of help window, in pixels.
        `height`: Height of help window, in pixels.

        `reference_window`: The window on which the help card will be centered when shown. Usually this is the DPG primary window,
                            so that the help will be centered on the whole viewport.

                            To manually recenter (e.g. in a DPG viewport resize handler), you can call the `reposition` method.

        `themes_and_fonts`: Obtain by calling `raven.common.gui.utils.bootup` at app start time.
                            Apps that skip `bootup` can pass a minimal ``env(font_size=N)`` instead.

        `gui_font`: Optional DPG font ID to bind to the help window. If ``None`` (the default),
                    the help card inherits the DPG default font, which is correct for apps that
                    call `bootup`. Apps that use a non-standard default font (e.g. a large countdown
                    font) should pass a GUI-sized font here.

        `highlight_color`: RGB or RGBA tuple, range [0, 255]. Text color for a highlighted segment. Meant for use by `on_render_extras`.
        `heading_color`: RGB or RGBA tuple, range [0, 255]. Text color for help headings.
        `text_color`: RGB or RGBA tuple, range [0, 255]. Text color for regular help text.
        `dimmed_color`: RGB or RGBA tuple, range [0, 255]. Text color for dimmed help text.

        `on_render_extras`: User extras renderer. Optional 2-argument callable, signature is::

                                (self: HelpWindow, gui_parent: Union[str, int]) -> None

                            The return value is ignored.

                            If provided, this is called once, when the help card is first rendered. The callback can generate
                            arbitrary DPG widgets. The `gui_parent` argument is the DPG tag of the group the widgets should be
                            rendered in (so as not to depend on the DPG stack state; set the parent using `parent=...`).

                            If you have several help windows in your app, this instance's unique identifier, meant for use
                            in DPG tags, is available in `self.gui_uuid`.

                            For use as the `color` argument of `dpg.add_text`, the configured colors are available as RGB or RGBA tuples
                            in the attributes `highlight_color`, `heading_color`, `text_color`, and `dimmed_color`.

                            For `dpg_markdown` color formatting, HTML tag variants of the colors are available in the attributes
                            `c_hig`, `c_hed`, `c_txt`, and `c_dim`. The HTML tag to end a colored segment is available as the attribute `c_end`.

                            Note that `dpg_markdown` does not support nesting color tags.

        `on_show`: Triggered when the help window opens. 0-argument callable. Return value is ignored.
        `on_hide`: Triggered when the help window closes. 0-argument callable. Return value is ignored.

                   These can be useful e.g. if the app needs to enter a modal mode (disable some UI animations etc.)
                   while a modal dialog (such as the help window) is on the screen.

        `label`: The window title. Worth setting when the card belongs to something other than the app as a
                 whole — a dialog's card appears with that dialog hidden behind it (see `handle_own_hotkeys`),
                 so the title is what says whose keys are being listed.

        `handle_own_hotkeys`: Whether this card closes itself on Esc, via the module-level key handler that
                              every `HelpWindow` shares. True (the default) is right for an app's own card.

                              Pass False when the card belongs to another modal window that already has a key
                              handler of its own — a file dialog offering a card of its keys. That owner then
                              closes the card by calling `hide`, and is free to bind whatever else it likes
                              while the card is up.
        """
        self.gui_uuid = str(uuid.uuid4())  # used in GUI widget tags
        self.hotkey_info = hotkey_info
        self._width = width
        self._height = height
        self.reference_window = reference_window

        self.themes_and_fonts = themes_and_fonts

        self.highlight_color = highlight_color
        self.heading_color = heading_color
        self.text_color = text_color
        self.dimmed_color = dimmed_color

        self.help_indent_pixels = 20  # per indent level

        # Shorthand for color control sequences for MD renderer
        self.c_hig = f'<font color="{self.highlight_color}">'
        self.c_hed = f'<font color="{self.heading_color}">'
        self.c_txt = f'<font color="{self.text_color}">'
        self.c_dim = f'<font color="{self.dimmed_color}">'
        self.c_end = '</font>'

        self.gui_font = gui_font
        self.on_render_extras = on_render_extras
        self.on_show = on_show
        self.on_hide = on_hide
        self.label = label
        self.handle_own_hotkeys = handle_own_hotkeys

        self._window = None
        self._content_group = None

        self._initialize_class()

    def _apply_size(self) -> None:
        """Push the current size onto the window, if there is one yet.

        Recentering is part of this rather than left to the caller. `show` and `reposition` both keep the
        card centered on its reference window, so a resize that quietly stopped doing so would leave the
        caller repairing an invariant it had no way to know it had disturbed — and no public way to repair,
        since centering on a size DPG has not laid out yet is exactly what `reposition` spends a rendered
        frame on. Only while the card is on screen: recentering shows the window as a side effect of
        measuring it, which would pop a hidden card onto the screen.
        """
        if self._window is None:  # not built yet; `_render` reads the size when it builds
            return
        dpg.configure_item(self._window, width=self._width, height=self._height)
        if self.is_visible():
            self.reposition(_force=True)

    def _get_width(self) -> int:
        return self._width
    def _set_width(self, width: int) -> None:
        if width != self._width:
            self._width = width
            self._apply_size()
    width = property(fget=_get_width, fset=_set_width,
                     doc="Width of the help window, in pixels. Read/write; setting it recenters the window.")

    def _get_height(self) -> int:
        return self._height
    def _set_height(self, height: int) -> None:
        if height != self._height:
            self._height = height
            self._apply_size()
    height = property(fget=_get_height, fset=_set_height,
                      doc="Height of the help window, in pixels. Read/write; setting it recenters the window.")

    def _render(self) -> None:
        """Construct the GUI. Called automatically when the window is shown for the first time."""
        if self._window is not None:  # already rendered
            logger.info("HelpWindow._render: Done, GUI already rendered.")
            return
        if dpg.get_frame_count() < 10:
            logger.info("HelpWindow._render: Too early, ignoring. (Fewer than 10 DPG frames elapsed since app start.)")
            return
        logger.info("HelpWindow._render: Rendering GUI.")

        # Extract columns from the human-readable representation
        columns = []
        current_column = []
        for help_entry in self.hotkey_info:
            if help_entry is hotkey_new_column:
                columns.append(current_column)
                current_column = []
            else:
                current_column.append(help_entry)
        if len(current_column):  # loop-and-a-half, kind of
            columns.append(current_column)
        ncols = len(columns)

        # Convert to rows (format actually used by DPG for constructing tables)
        rows = list(itertools.zip_longest(*columns, fillvalue=hotkey_blank_entry))

        # --------------------------------------------------------------------------------

        help_window = dpg.add_window(show=False, label=self.label, tag=f"help_window_{self.gui_uuid}",
                                     modal=True,
                                     on_close=self.hide,
                                     no_collapse=True,
                                     no_resize=True,
                                     no_scrollbar=True,
                                     no_scroll_with_mouse=True,
                                     width=self._width,
                                     height=self._height)

        if self.gui_font is not None:
            dpg.bind_item_font(help_window, self.gui_font)

        help_group = dpg.add_group(tag=f"help_group_{self.gui_uuid}",
                                   parent=help_window)
        # Header
        dpg_markdown.add_text(f"{self.c_dim}[Press Esc to close. For a handy reference, screenshot this!]{self.c_end}",
                              parent=help_group)
        dpg.add_spacer(width=1,
                       height=self.themes_and_fonts.font_size // 2,
                       parent=help_group)

        # Table of hotkeys.
        hotkeys_table = dpg.add_table(header_row=True,
                                      borders_innerV=True,
                                      sortable=False,
                                      parent=help_group)
        for _ in range(ncols):
            dpg.add_table_column(label="Key or combination",  # key
                                 parent=hotkeys_table)
            dpg.add_table_column(label="Action",  # action
                                 parent=hotkeys_table)
            dpg.add_table_column(label="Notes",  # notes
                                 parent=hotkeys_table)
        for row in rows:
            table_row = dpg.add_table_row(parent=hotkeys_table)
            for help_entry in row:
                if help_entry.key_indent > 0:
                    g = dpg.add_group(horizontal=True,
                                      parent=table_row)
                    dpg.add_spacer(width=help_entry.key_indent * self.help_indent_pixels,
                                   parent=g)
                    dpg.add_text(help_entry.key, wrap=0, color=self.heading_color,
                                 parent=g)
                else:
                    dpg.add_text(help_entry.key, wrap=0, color=self.heading_color,
                                 parent=table_row)

                if help_entry.action_indent > 0:
                    g = dpg.add_group(horizontal=True,
                                      parent=table_row)
                    dpg.add_spacer(width=help_entry.action_indent * self.help_indent_pixels,
                                   parent=g)
                    dpg.add_text(help_entry.action, wrap=0, color=self.dimmed_color,
                                 parent=g)
                else:
                    dpg.add_text(help_entry.action, wrap=0, color=self.dimmed_color,
                                 parent=table_row)

                dpg.add_text(help_entry.notes, wrap=0, color=self.dimmed_color,
                             parent=table_row)

        # End spacer for table of hotkeys
        dpg.add_spacer(width=1, height=self.themes_and_fonts.font_size,
                       parent=help_group)

        # Optional user extras
        if self.on_render_extras is not None:
            logger.info("HelpWindow._render: Rendering user extras.")
            self.on_render_extras(self, help_group)
        else:
            logger.info("HelpWindow._render: No user extras renderer specified, skipping.")

        self._window = help_window
        self._content_group = help_group
        logger.info("HelpWindow._render: Done.")

    def show(self) -> bool:
        """Show the help window. Returns whether it is now up.

        This also auto-centers the help window on the reference window.

        The `on_show` handler, if set, will be called.

        `False` means the GUI has not run long enough to build the card yet (the first few frames after app
        start), and nothing was done. Callers that hide something *behind* the card need this answer; a
        caller that merely opens a card can ignore it.
        """
        global visible_help_window_instance
        logger.info("HelpWindow.show: Showing window.")
        self._render()
        # `_render` declines to build during the first few frames, and a window that does not exist cannot
        # be positioned or shown. Reported rather than raised: this is a timing condition, not a mistake at
        # the call site.
        if self._window is None:
            logger.info("HelpWindow.show: Window was not built, nothing to show.")
            return False
        self.reposition(_force=True)
        dpg.show_item(self._window)  # For some reason, we need to do this *after* `set_item_pos` for a modal window, or this works only every other time (1, 3, 5, ...). Maybe a modal must be inside the viewport to successfully show it?
        visible_help_window_instance = self
        if self.on_show is not None:
            self.on_show()
        dpg.focus_item(self._window)
        logger.info("HelpWindow.show: Done.")
        return True

    def hide(self) -> None:
        """Close the help window, if it is open.

        If the window was open, and is being closed, the `on_hide` handler, if set, will be called.
        """
        global visible_help_window_instance
        if self._window is None:
            logger.info("HelpWindow.hide: Window does not exist. Nothing needs to be done.")
            return
        logger.info("HelpWindow.hide: Hiding window.")
        visible_help_window_instance = None
        dpg.hide_item(self._window)
        if self.on_hide is not None:
            self.on_hide()
        logger.info("HelpWindow.hide: Done.")

    def is_visible(self) -> bool:
        """Return whether the help window is open.

        We have this abstraction (not just `dpg.is_item_visible`) because the window might not exist, if it has not been opened yet.
        """
        if self._window is None:
            return False
        return dpg.is_item_visible(self._window)

    def measure_content_height(self) -> int | None:
        """The window height that would exactly fit this card's content, in pixels.

        Measured from the window's top edge, so the title bar is included and the answer is directly
        comparable with `height`. The content ends in a spacer, which is what stands in for bottom padding.

        `None` while there is nothing to measure: the card has not been built, or has not yet been drawn.
        Layout is what produces geometry. `show` renders the card for a frame before it places it, so an
        `on_show` handler is the earliest point at which there is an answer.

        Nothing here acts on the number — the window keeps the size it was given. This is for a caller
        whose card holds a different set of rows in different instances, and who would otherwise carry one
        height tuned for the tallest of them.
        """
        if self._window is None or self._content_group is None:
            return None
        content_height = dpg.get_item_rect_size(self._content_group)[1]
        if not content_height:  # a group that has never been laid out reports a zero rect
            return None
        # The group's own position, which is relative to the enclosing window, is the title bar plus the
        # window's top padding — measured rather than assumed, both being font-size dependent. Deliberately
        # not the difference of two viewport coordinates: `rect_min` is where the group was drawn on the
        # last frame, so a window that has been moved since (which is exactly what `show` does before it
        # calls `on_show`) would have the two halves of that subtraction a frame apart. A parent-relative
        # position does not move when the window does.
        content_top = dpg.get_item_pos(self._content_group)[1]
        return int(content_top + content_height)

    def reposition(self, _force: bool = False) -> None:
        """Recenter the help window on its reference window.

        `_force`: Center the window even if it is not visible. (This will make it visible.)
                  For internal use by `show`.
        """
        if self._window is None:
            logger.info("HelpWindow.reposition: Window does not exist. Nothing needs to be done.")
            return
        if _force or self.is_visible():
            logger.info("HelpWindow.reposition: Recentering window.")
            guiutils.recenter_window(self._window, reference_window=self.reference_window, update_window_size=_force)
            logger.info("HelpWindow.reposition: Done.")
        else:
            logger.info("HelpWindow.reposition: Window is not visible. Nothing needs to be done.")
