"""Generic GUI help window for DPG apps, intended mainly as a hotkey reference. One screen, or several."""

__all__ = ["hotkey_new_column", "hotkey_blank_entry",
           "page", "section",
           "helpcard_hotkeys_callback", "HelpWindow"]

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
from ...vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa  # https://github.com/juliettef/IconFontCppHeaders

from . import utils as guiutils

# --------------------------------------------------------------------------------

#: How many frames a page is given to settle before its height is believed. Matches the file dialog's own
#: fitting pass; a table's column widths, and so which cells wrap, take more than one frame to stop moving.
_PAGE_FIT_PASSES = 4

#: What the card says about itself. On a paged card it rides at the right end of the toolbar, where it
#: costs no line of its own; otherwise it has one, above the content.
_ESC_HINT = "[Press Esc to close. For a handy reference, screenshot this!]"

#: White space kept to the right of each column of prose - between the two columns, and between the second
#: one and the card's edge. This is on top of the item spacing DPG already puts between two groups: with
#: that alone, a line long enough to fill its wrap ends one space short of the next column, and the break
#: stops reading as a break.
_COLUMN_GUTTER = 8

hotkey_new_column = sym("next_column")
hotkey_blank_entry = env(key_indent=0, key="", action_indent=0, action="", notes="")


def page(name: str,
         hotkey_info: Optional[List[env]] = None,
         on_render_extras: Optional[Callable] = None) -> env:
    """One page of a multi-page help card, for `HelpWindow`'s `pages`.

    `name`: What the toolbar calls this page. Required.
    `hotkey_info`: A hotkey table for this page, in the format `HelpWindow` documents. Optional.
    `on_render_extras`: An extras renderer for this page, with `HelpWindow`'s signature. Optional.

    At least one of the two content arguments is needed; passing both puts the table above the extras, as
    an unpaged card does.

    A constructor rather than a bare `env` so that the name cannot be left out. A card that pages has a
    toolbar, and a toolbar with nothing to say about where you are is most of the way to no toolbar at all
    — so the requirement is enforced here, where it is one line, rather than being a convention each
    caller has to have read.
    """
    if not name:
        raise ValueError("helpcard.page: a page needs a name; it is what the toolbar shows.")
    if hotkey_info is None and on_render_extras is None:
        raise ValueError(f"helpcard.page: page '{name}' would be blank; pass `hotkey_info`, `on_render_extras`, or both.")
    return env(name=name, hotkey_info=hotkey_info, on_render_extras=on_render_extras)


def section(heading: Optional[str], *paragraphs: str) -> env:
    """One headed block of prose, for `HelpWindow.prose_columns`.

    `heading`: The section's title, as Markdown, or `None` for a block that has none. Drawn in the card's
               heading colour; a run that should not be — a parenthetical gloss, say — can say so with a
               `c_txt` span.
    `paragraphs`: The body, top to bottom, as Markdown.

                  A paragraph may be a block of several lines — a bullet list, say — in which case
                  `textwrap.dedent(...).strip()` is how to write one legibly in indented source. Note the
                  order: `strip` first would take the indentation off the opening line only, leaving
                  `dedent` no common prefix to find.
    """
    if heading is None and not paragraphs:
        raise ValueError("helpcard.section: a section with neither heading nor paragraphs would be blank.")
    return env(heading=heading, paragraphs=list(paragraphs))

# Hotkey support
visible_help_window_instance = None  # fdialog is modal so There Can Be Only One (TM). If needed, could use a list, and check which one has keyboard focus, but that might not always work.
def helpcard_hotkeys_callback(sender, app_data):
    card = visible_help_window_instance
    if card is None:
        return

    key = app_data  # for documentation only
    # shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
    # ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)

    # Turning a page is the card's own whoever handles Escape. `handle_own_hotkeys` is about that one key,
    # which an owning modal wants for itself; these four collide with nothing, since a card is always up
    # with its owner hidden behind it, and the card is modal so no other window is listening either.
    if card.page_count > 1:
        turn = {dpg.mvKey_Left: card.previous_page,
                dpg.mvKey_Right: card.next_page,
                dpg.mvKey_Home: card.first_page,
                dpg.mvKey_End: card.last_page}.get(key)
        if turn is not None:
            turn()
            return

    if not card.handle_own_hotkeys:
        return  # this card's owner routes keys to it, so Escape reaches one handler rather than two
    if key == dpg.mvKey_Escape:
        card.hide()
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

            # Register our hotkey handler. Explicit parent, no `with`: DPG's container stack is one
            # process-wide global. See `dpg-notes.md`, "DPG parent management".
            registry = dpg.add_handler_registry(tag="helpcard_handler_registry")  # tag  # global (whole viewport)
            dpg.add_key_press_handler(parent=registry, tag="helpcard_hotkeys_handler", callback=helpcard_hotkeys_callback)  # tag

    def __init__(self,
                 width: int,
                 height: int,
                 reference_window: Union[str, int],
                 themes_and_fonts: env,
                 hotkey_info: Optional[List[env]] = None,
                 pages: Optional[List[env]] = None,
                 highlight_color: Tuple[int] = (255, 0, 0, 255),
                 heading_color: Tuple[int] = (255, 255, 255, 255),
                 text_color: Tuple[int] = (180, 180, 180, 255),
                 dimmed_color: Tuple[int] = (140, 140, 140, 255),
                 gui_font: Optional[int] = None,
                 on_render_extras: Optional[Callable] = None,
                 on_parked: Optional[Callable] = None,
                 on_show: Optional[Callable] = None,
                 on_hide: Optional[Callable] = None,
                 label: str = "Help",
                 handle_own_hotkeys: bool = True):
        """Set up the help window. You only need one instance per app (or per main view, which has different hotkeys).

        The card holds either one screen or several. One screen is `hotkey_info` and `on_render_extras`,
        which is what most apps want and what every app had before pages existed. Several is `pages`, and
        then the card grows a toolbar for turning them. Pass one form or the other, not both.

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

        `pages`: Several screens instead of one, each built by `helpcard.page`, in the order they are
                 turned. Mutually exclusive with `hotkey_info` and `on_render_extras`, which describe the
                 single-screen form.

                 With two or more, the card grows a toolbar: first / previous / next / last, the page's
                 name, and an `N / M` counter. `Left`, `Right`, `Home` and `End` do the same from the
                 keyboard — free, since the card is modal and owns the keyboard while it is up.

                 **Reach for this when one screen has stopped being the right format**, not merely when
                 the content is tight. The pull is to page the hotkey table, and that is rarely the first
                 cut: a table split across pages is harder to scan than a full one, while prose moved off
                 the table's page leaves a reference card a reader can screenshot and keep.

                 The card sizes itself to the tallest page and keeps that height, so turning a page does
                 not resize the window under the reader — the cost is some empty space on the shorter
                 pages, which is the cheaper of the two.

        `width`: Width of help window, in pixels.
        `height`: Height of help window, in pixels. With `pages`, a starting value only: the card measures
                  its pages when first shown and grows to fit the tallest.

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

        `on_render_extras`: User extras renderer, for the single-screen form. Optional 2-argument callable,
                            signature is::

                                (self: HelpWindow, gui_parent: Union[str, int]) -> None

                            The return value is ignored. A paged card passes one of these per page, to
                            `helpcard.page`, with the same signature.

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

        `on_parked`: Triggered once per `show`, while the card is drawn but parked outside the viewport,
                     before it has been placed. 0-argument callable. Return value is ignored.

                     For a card that has to be *measured* before it is placed — `measure_content_height`
                     needs it laid out, and parking is how a window is laid out unseen. Resizing here is
                     free to look at; resizing from `on_show` instead would place the card twice and show
                     the reader the wrong one.

                     A handler that spends frames must park on each of them (`settle_offscreen` does the
                     pair), since a park holds for one frame only.

        `on_show`: Triggered when the help window opens. 0-argument callable. Return value is ignored.
        `on_hide`: Triggered when the help window closes. 0-argument callable. Return value is ignored.

                   These can be useful e.g. if the app needs to enter a modal mode (disable some UI animations etc.)
                   while a modal dialog (such as the help window) is on the screen.

                   `on_show` fires with the card **placed and on screen**, which is what a handler doing
                   that bookkeeping needs: `raven.visualizer`'s `enter_modal_mode` asks the GUI what is
                   visible, and spends a frame doing it. `on_parked` is the hook for anything that has to
                   happen earlier.

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

        # One internal shape for both forms, so that nothing below has to ask which one it was given. The
        # single-screen card is one nameless page: it grows no toolbar, so it has nowhere to show a name
        # and nothing to ask for one.
        if pages is not None:
            if hotkey_info is not None or on_render_extras is not None:
                raise ValueError("HelpWindow: pass either `pages` or the single-screen `hotkey_info` / "
                                 "`on_render_extras`, not both.")
            if not pages:
                raise ValueError("HelpWindow: `pages` is empty; a card needs something to show.")
            self._pages = list(pages)
        else:
            if hotkey_info is None and on_render_extras is None:
                raise ValueError("HelpWindow: nothing to show; pass `hotkey_info`, `on_render_extras`, "
                                 "or `pages`.")
            self._pages = [env(name=None, hotkey_info=hotkey_info, on_render_extras=on_render_extras)]
        self._page_index = 0
        self._page_groups = []  # populated by `_render`, one per page, in the same order
        self._toolbar = None  # populated by `_render` when there is more than one page
        self._height_fitted = False  # whether the tallest page has been measured; see `_fit_height_to_pages`

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
        self.on_parked = on_parked
        self.on_show = on_show
        self.on_hide = on_hide
        self.label = label
        self.handle_own_hotkeys = handle_own_hotkeys

        self._window = None
        self._content_group = None

        self._initialize_class()

    def _apply_size(self) -> None:
        """Push the current size onto the window, if there is one yet.

        Placement is deliberately *not* part of this, and `show` is why: it resizes between parking the
        card and placing it, so a setter that placed the card would place it twice — once for a size it is
        about to stop having, in view of the reader. Resizing a card that is already up therefore leaves it
        where it was, and `reposition` is what re-centers it.
        """
        if self._window is None:  # not built yet; `_render` reads the size when it builds
            return
        dpg.configure_item(self._window, width=self._width, height=self._height)

    def _get_width(self) -> int:
        return self._width
    def _set_width(self, width: int) -> None:
        if width != self._width:
            self._width = width
            self._apply_size()
    width = property(fget=_get_width, fset=_set_width,
                     doc="Width of the help window, in pixels. Read/write. Setting it while the card is up "
                         "leaves the card where it is; call `reposition` to re-center it.")

    def _get_height(self) -> int:
        return self._height
    def _set_height(self, height: int) -> None:
        if height != self._height:
            self._height = height
            self._apply_size()
    height = property(fget=_get_height, fset=_set_height,
                      doc="Height of the help window, in pixels. Read/write. Setting it while the card is up "
                          "leaves the card where it is; call `reposition` to re-center it.")

    def _get_content_width(self) -> int:
        # Both sides. The card has no scrollbar, so `DPG_SCROLLBAR_SIZE` does not come off as well.
        return self._width - 2 * guiutils.DPG_WINDOW_PADDING
    content_width = property(fget=_get_content_width,
                             doc="How wide text may be before it runs off the card, in pixels. Read-only.\n\n"
                                 "Pass this as `wrap` when rendering into the card - `dpg_markdown.add_text` "
                                 "and `dpg.add_text` both leave text unwrapped unless told a width, and the "
                                 "card has no scrollbar, so an unwrapped line is simply cut off at the edge.")

    def _get_column_width(self) -> int:
        # Across the content width sit: column, gutter, the item spacing a horizontal group puts between
        # its two children, column, gutter. Everything but the columns comes off before the halving —
        # a horizontal group *adds* its spacing to whatever its children asked for, so halving the whole
        # width overflows the card by exactly that. Rounding down leaves the odd pixel unclaimed, which is
        # the harmless direction.
        return (self.content_width - 2 * _COLUMN_GUTTER - guiutils.DPG_ITEM_SPACING_X) // 2
    column_width = property(fget=_get_column_width,
                            doc="How wide text may be in one of two side-by-side columns, in pixels. Read-only.\n\n"
                                "`content_width` for prose that runs the width of the card; this for prose "
                                "split into two columns, which a card this wide usually wants - a single "
                                "column gives lines too long to track back to the start of. Pass it as `wrap`, "
                                "and put the two column groups in one horizontal group - or let "
                                "`prose_columns` do both. The gap between them "
                                "is already accounted for here.")

    def prose_columns(self,
                      gui_parent: Union[str, int],
                      left: List[env],
                      right: List[env],
                      paragraph_gap: Optional[int] = None,
                      section_gap: Optional[int] = None,
                      color: Optional[Tuple[int]] = None) -> Union[str, int]:
        """Render a page of prose as two newspaper columns. Returns the group holding them.

        `gui_parent`: Where to put them - the parent an extras renderer is handed.
        `left`, `right`: The sections of each column, top to bottom, each built by `helpcard.section`.
                         Either may be empty.

                         **Newspaper columns, not a pair per section**: a reader scans one column to its
                         end before crossing to the other, so a section belongs wholly to one of them.
                         Which sections go where is the caller's to balance — a column cannot be measured
                         before it is drawn, so nothing here can do it.
        `paragraph_gap`: Vertical space between two paragraphs of the same section, in pixels. `None`, the
                         default, is half a line.
        `section_gap`: Vertical space between two sections of the same column, in pixels. `None`, the
                       default, is a full line.
        `color`: Colour for text the Markdown does not colour itself, as `dpg_markdown.add_text` takes it.
                 `None`, the default, is the card's `text_color`.

                 Prefer this to opening a `c_txt` span at the head of a paragraph: an open `<font>` tag on
                 the same line as the content makes the whole string one CommonMark paragraph, so a list
                 inside it is read as literal text. The `c_hig` / `c_hed` / `c_dim` shorthands remain the
                 way to colour a run *within* a paragraph.

        Each column is pinned to `column_width` plus a gutter whatever it holds, so the divide between
        them falls at the same place all the way down the page.
        """
        if paragraph_gap is None:
            paragraph_gap = self.themes_and_fonts.font_size // 2
        if section_gap is None:
            section_gap = self.themes_and_fonts.font_size
        if color is None:
            color = self.text_color
        columns_group = dpg.add_group(horizontal=True, parent=gui_parent)
        for sections in (left, right):
            column_group = dpg.add_group(horizontal=False, parent=columns_group)
            # A group comes out as wide as its widest *rendered* line rather than as wide as the `wrap` its
            # text was given, so a column holding one short paragraph is narrower than one holding three
            # and the divide wanders down the page. An explicitly sized spacer states the width instead,
            # which is how Raven pins a container's width elsewhere.
            dpg.add_spacer(width=self.column_width + _COLUMN_GUTTER, parent=column_group)
            for section_index, one_section in enumerate(sections):
                if section_index:
                    dpg.add_spacer(height=section_gap, parent=column_group)
                if one_section.heading is not None:
                    dpg_markdown.add_text(one_section.heading, parent=column_group,
                                          wrap=self.column_width, color=self.heading_color)
                for index, paragraph in enumerate(one_section.paragraphs):
                    if index:
                        dpg.add_spacer(height=paragraph_gap, parent=column_group)
                    dpg_markdown.add_text(paragraph, parent=column_group, wrap=self.column_width, color=color)
        return columns_group

    def _get_page_count(self) -> int:
        """Return how many pages this card holds. One for a single-screen card."""
        return len(self._pages)
    page_count = property(fget=_get_page_count,
                          doc="How many pages this card holds; 1 for a single-screen card. Read-only.")

    def _get_page_index(self) -> int:
        """Return the 0-based index of the page currently shown."""
        return self._page_index
    def _set_page_index(self, index: int) -> None:
        """Turn to page `index`, clamping to the ends rather than wrapping."""
        index = max(0, min(index, len(self._pages) - 1))
        if index == self._page_index:
            return
        self._page_index = index
        self._apply_page()
    page_index = property(fget=_get_page_index, fset=_set_page_index,
                          doc="0-based index of the page on screen. Read/write; setting it turns the page.\n\n"
                              "Clamped to the ends rather than wrapped, which is what the buttons and keys "
                              "do too: asking to go past the last page means staying on it.")

    def first_page(self) -> None:
        """Turn to the first page. Does nothing on a single-screen card."""
        self.page_index = 0

    def previous_page(self) -> None:
        """Turn back one page, stopping at the first."""
        self.page_index = self._page_index - 1

    def next_page(self) -> None:
        """Turn on one page, stopping at the last."""
        self.page_index = self._page_index + 1

    def last_page(self) -> None:
        """Turn to the last page. Does nothing on a single-screen card."""
        self.page_index = len(self._pages) - 1

    def _apply_page(self) -> None:
        """Show the current page's widgets, hide the rest, and bring the toolbar up to date."""
        if not self._page_groups:  # not rendered yet; `_render` builds with the current page already showing
            return
        for index, group in enumerate(self._page_groups):
            with guiutils.nonexistent_ok():
                dpg.configure_item(group, show=(index == self._page_index))
        self._update_toolbar()

    def _update_toolbar(self) -> None:
        """Refresh the toolbar's name, counter and button states for the page now showing."""
        if self._toolbar is None:
            return
        at_first = (self._page_index == 0)
        at_last = (self._page_index == len(self._pages) - 1)
        with guiutils.nonexistent_ok():
            dpg.set_value(self._toolbar.name_widget, self._pages[self._page_index].name)
            dpg.set_value(self._toolbar.counter_widget, f"{self._page_index + 1} / {len(self._pages)}")
            # A button is enabled exactly when pressing it would do something, which is the Raven way and
            # what the chat graph, the chat log and the Visualizer all do with their own step buttons.
            for widget in (self._toolbar.first_button, self._toolbar.previous_button):
                dpg.configure_item(widget, enabled=not at_first)
            for widget in (self._toolbar.next_button, self._toolbar.last_button):
                dpg.configure_item(widget, enabled=not at_last)
            self._align_hint_right()  # the name just changed width, and the hint sits after it

    def _render(self) -> None:
        """Construct the GUI. Called automatically when the window is shown for the first time."""
        if self._window is not None:  # already rendered
            logger.info("HelpWindow._render: Done, GUI already rendered.")
            return
        if dpg.get_frame_count() < 10:
            logger.info("HelpWindow._render: Too early, ignoring. (Fewer than 10 DPG frames elapsed since app start.)")
            return
        logger.info("HelpWindow._render: Rendering GUI.")

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

        # The page controls sit above everything: they are the card's own chrome, and a reader looking for
        # "how do I get to the other page" looks at the top edge. The Esc hint rides along at the right end
        # of that row, so a paged card spends no line on either of them.
        if len(self._pages) > 1:
            self._build_toolbar(help_group)
        else:
            dpg_markdown.add_text(f"{self.c_dim}{_ESC_HINT}{self.c_end}",
                                  parent=help_group)
            dpg.add_spacer(width=1,
                           height=self.themes_and_fonts.font_size // 2,
                           parent=help_group)

        # One group per page, all built now and all but the current one hidden. Built once rather than on
        # demand because the extras renderers are documented to run once, and because a page that appears
        # only when first turned to would have to be measured then too — which is a resize under the
        # reader's eye, the one thing the fixed height exists to prevent.
        self._page_groups = []
        for index, one_page in enumerate(self._pages):
            page_group = dpg.add_group(tag=f"help_page_{index}_{self.gui_uuid}",
                                       show=(index == self._page_index),
                                       parent=help_group)
            if one_page.hotkey_info is not None:
                self._render_hotkey_table(one_page.hotkey_info, page_group)
            if one_page.on_render_extras is not None:
                logger.info(f"HelpWindow._render: Rendering user extras for page {index}.")
                one_page.on_render_extras(self, page_group)
            self._page_groups.append(page_group)

        self._window = help_window
        self._content_group = help_group
        self._update_toolbar()
        logger.info("HelpWindow._render: Done.")

    def _render_hotkey_table(self, hotkey_info: List[env], gui_parent: Union[str, int]) -> None:
        """Draw one hotkey table, from the human-readable entry list, into `gui_parent`."""
        # Extract columns from the human-readable representation
        columns = []
        current_column = []
        for help_entry in hotkey_info:
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

        hotkeys_table = dpg.add_table(header_row=True,
                                      borders_innerV=True,
                                      sortable=False,
                                      parent=gui_parent)
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
                       parent=gui_parent)

    def _build_toolbar(self, gui_parent: Union[str, int]) -> None:
        """Draw the page controls: first / previous / next / last, the page's name, and an `N / M` counter.

        The glyphs and their order are the chat graph's sibling-navigation ones, minus its ±10 pair, which
        has no counterpart on a card of two or three pages. Same verbs on the same shape of run, so a
        reader should not have to recognise them twice.

        The first and last buttons earn their place as much by *signage* as by what they do: they are what
        `Home` and `End` hang their tooltips on. A help card that needed a help card to explain its own
        navigation would be a delicious recursion and a bad card.
        """
        toolbar_group = dpg.add_group(horizontal=True, parent=gui_parent)
        button_w = 2 * self.themes_and_fonts.font_size

        def add_button(icon: str, callback: Callable, caption: str) -> Union[str, int]:
            button = dpg.add_button(label=icon, callback=lambda: callback(),
                                    width=button_w,
                                    parent=toolbar_group)
            # An app that skipped `bootup` passes a minimal `env(font_size=N)`, which carries neither of
            # these. The buttons then draw in the default font, as the literal glyph names rather than
            # icons - ugly and still usable, which is the right failure for a help card.
            icon_font = getattr(self.themes_and_fonts, "icon_font_solid", None)
            if icon_font is not None:
                dpg.bind_item_font(button, icon_font)
            disablable_theme = getattr(self.themes_and_fonts, "disablable_widget_theme", None)
            if disablable_theme is not None:
                dpg.bind_item_theme(button, disablable_theme)
            dpg.add_text(caption, parent=dpg.add_tooltip(button))
            return button

        first_button = add_button(fa.ICON_BACKWARD_FAST, self.first_page, "First page [Home]")
        previous_button = add_button(fa.ICON_CARET_LEFT, self.previous_page, "Previous page [Left]")
        next_button = add_button(fa.ICON_CARET_RIGHT, self.next_page, "Next page [Right]")
        last_button = add_button(fa.ICON_FORWARD_FAST, self.last_page, "Last page [End]")

        fixed_gap = self.themes_and_fonts.font_size
        dpg.add_spacer(width=fixed_gap, parent=toolbar_group)
        name_widget = dpg.add_text("", color=self.heading_color, parent=toolbar_group)
        dpg.add_spacer(width=fixed_gap, parent=toolbar_group)
        counter_widget = dpg.add_text("", color=self.dimmed_color, parent=toolbar_group)

        # The card's own line about itself, at the right end rather than on a line of its own — which is a
        # line back for the content, and puts all of the card's chrome in one row.
        hint_spacer = dpg.add_spacer(width=1, parent=toolbar_group)
        dpg.add_text(_ESC_HINT, color=self.dimmed_color, parent=toolbar_group)

        self._toolbar = env(first_button=first_button, previous_button=previous_button,
                            next_button=next_button, last_button=last_button,
                            name_widget=name_widget, counter_widget=counter_widget,
                            hint_spacer=hint_spacer,
                            button_w=button_w, fixed_gap=fixed_gap)

        dpg.add_spacer(width=1,
                       height=self.themes_and_fonts.font_size // 2,
                       parent=gui_parent)

    def _align_hint_right(self) -> None:
        """Push the Esc hint to the card's right edge, by sizing the spacer in front of it.

        DPG has no flexible spacer and a horizontal group lays its items out left to right, so where the
        hint lands has to be computed. The text widths are measured, depending as they do on the font; the
        rest is known — four buttons, two fixed gaps, and the theme's spacing between each pair of items.

        Recomputed on every page turn, the page's name being the one piece whose width changes.
        """
        toolbar = self._toolbar
        font = self.gui_font if self.gui_font is not None else 0  # 0 asks for the font the card draws in

        def text_width(text: str) -> Optional[float]:
            if not text:
                return 0.0
            measured = dpg.get_text_size(text, font=font)
            return measured[0] if measured is not None else None

        texts = [dpg.get_value(toolbar.name_widget), dpg.get_value(toolbar.counter_widget), _ESC_HINT]
        widths = [text_width(text) for text in texts]
        if any(width is None for width in widths):
            # DPG cannot measure text until the font atlas is up, and answers `None` rather than raising.
            # A running app never sees this — the card is built long after startup — but a headless test
            # does. Leave the hint where the layout put it rather than placing it from an invented width.
            return

        # Nine items in the row, so eight gaps between them — the spacer being sized is one of the nine.
        # Plus one more gap's worth at the far end: flush against the window edge, the hint reads as
        # having been cut off rather than as ending.
        used = (4 * toolbar.button_w +
                2 * toolbar.fixed_gap +
                sum(widths) +
                9 * guiutils.DPG_ITEM_SPACING_X)
        # Never less than an ordinary gap: on a card too narrow to hold the row, the hint runs off the
        # right edge, and crowding it against the counter would only make that harder to read.
        dpg.configure_item(toolbar.hint_spacer,
                           width=max(guiutils.DPG_ITEM_SPACING_X, int(self.content_width - used)))

    def show(self) -> bool:
        """Show the help window. Returns whether it is now up.

        This also auto-centers the help window on the reference window.

        The `on_parked` handler, if set, is called while the card is drawn but parked out of sight, so a
        handler that resizes it (to fit its content, say) does so before the reader sees anything. The
        `on_show` handler, if set, is called once the card is placed and up.

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
        # Draw the card before deciding where it goes. A hidden item is not laid out, so a card that has
        # never been drawn has no geometry — and `on_parked` is where a caller measures it. Placing it
        # first and letting the handler resize it afterwards costs two placements, of which the reader sees
        # the wrong one: `fdialog`'s card appeared centered at its built height and then jumped to its
        # fitted one, over four frames.
        self.settle_offscreen()
        self._fit_height_to_pages()
        if self.on_parked is not None:
            self.on_parked()
        self.reposition(_force=True)
        dpg.show_item(self._window)  # For some reason, we need to do this *after* `set_item_pos` for a modal window, or this works only every other time (1, 3, 5, ...). Maybe a modal must be inside the viewport to successfully show it?
        visible_help_window_instance = self
        # Placed and up before `on_show` runs, deliberately. A handler here may ask the GUI what is visible,
        # and may spend a frame — and a frame spent while the card is still parked draws it back inside the
        # viewport, ImGui clamping any window whose position did not come through the API that frame.
        # `raven.visualizer`'s `enter_modal_mode` does both, and showed the card at the bottom right for a
        # frame or two when this ran earlier.
        if self.on_show is not None:
            self.on_show()
        dpg.focus_item(self._window)
        logger.info("HelpWindow.show: Done.")
        return True

    def _fit_height_to_pages(self) -> None:
        """Grow a paged card to its tallest page, once per session.

        The height is the tallest page's rather than each page's own, so that turning a page does not
        resize the window under the reader — which is unpleasant to read and costs the card its placement.
        What it costs instead is some empty space on the shorter pages, which is the cheaper of the two.

        Once, because the content is built once and never changes; a later show would spend these frames
        confirming a height the card already has. Each page is measured while the card is parked out of
        sight, so nothing here is drawn where the reader can see it — the caller has already parked it,
        and `settle_offscreen` renews the park per frame.

        A single-screen card is left alone entirely: its owner may be fitting it through `on_parked`
        (`fdialog` does), and two things sizing one window would fight.
        """
        if self._height_fitted or len(self._pages) < 2:
            return
        self._height_fitted = True

        was_showing = self._page_index
        tallest = self._height
        tallest_page = None
        for index in range(len(self._pages)):
            self.page_index = index
            # Several frames per page, as `fdialog._fit_help_card_to_content` spends: a table's column
            # widths settle over more frames than one, and which cells wrap to two lines follows from
            # those widths — so the first answer describes a layout still on its way somewhere. Stop as
            # soon as it stops moving.
            previous = None
            for _ in range(_PAGE_FIT_PASSES):
                self.settle_offscreen()
                measured = self.measure_content_height()
                if measured is None or measured == previous:
                    break
                previous = measured
            if previous is not None and previous > tallest:
                tallest, tallest_page = previous, self._pages[index].name
        self.page_index = was_showing

        # Clamped so that an oversized card stays reachable — a modal taller than the viewport puts its
        # own title bar out of reach. It should never fire: a page that does not fit the screen is a page
        # that wants splitting, which is what the log line is for.
        ceiling = dpg.get_viewport_client_height()
        if tallest > ceiling:
            logger.warning(f"HelpWindow._fit_height_to_pages: page '{tallest_page}' wants {tallest} px "
                           f"but the viewport is {ceiling} px tall; clamping, so that page is cut off. "
                           "Split it, or give the card less to say.")
            tallest = ceiling
        if tallest != self._height:
            logger.info(f"HelpWindow._fit_height_to_pages: fitting to the tallest page ('{tallest_page}'): "
                        f"{self._height} -> {tallest} px.")
            self.height = tallest
        # Whatever the size ends up being, the pages were shown and hidden to measure them, which leaves
        # the last one laid out and the rest not. One more park settles the page actually being shown.
        self.settle_offscreen()

    def settle_offscreen(self) -> None:
        """Draw the card one frame, outside the viewport, so that it has geometry nobody had to see.

        For `measure_content_height`, which needs the card laid out — a hidden item is not laid out at all
        — and for anything else that has to look at the card before it is placed. `show` spends one of
        these before calling `on_show`; a caller settling a layout that takes several frames calls this
        once per frame, which is also what keeps the card out of sight:

        **A park lasts exactly one frame**, which is why the position is re-set on every call rather than
        once at the start of a settle: ImGui pulls a window back inside the viewport on any frame whose
        position did not come through the API, and a modal is pulled *fully* into view. See
        `guiutils.park_offscreen`.

        The wait is not required: without it the card is merely unmeasurable, which costs a caller its
        chance to resize and nothing else.
        """
        guiutils.park_offscreen(self._window)
        dpg.show_item(self._window)
        guiutils.split_frame(operation="help card: laying the card out where it cannot be seen", required=False)

    def hide(self) -> None:
        """Take the help window off the screen, if it is up.

        If the window was open, and is being closed, the `on_hide` handler, if set, will be called.

        **`hide` and not `close`, where Raven's other windows close.** The distinction is real and worth
        keeping: this card is only made invisible — its widgets, its fitted height and the page it was
        left on all survive, and `show` re-renders nothing. A window whose closing *settles* something (the
        audio input panel stops metering and saves) or *releases* something (the cleanup dialog deletes its
        widgets and textures) is the one that closes, because reopening has to undo it. Here there is
        nothing to undo, so there is nothing for a close to mean.
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
