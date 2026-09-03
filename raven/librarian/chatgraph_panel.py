"""The chat graph view: an `XDotWidget` showing the chat forest, and what its clicks mean.

The picture itself is `chatgraph`'s, which is pure and knows nothing about DearPyGui. This is the half that
does: it owns the widget, keeps the view in step with a forest another thread is writing to, and turns a
click on a box into either a preview or a move.

**Preview and commit are separate acts, and that is the whole interaction model.** Clicking a box shows what
is there — scrolling the chat log to it if it is on the branch you are on, re-laying the graph out around it
if it is not. Nothing has changed at that point: HEAD is where it was, and the picture says so by keeping
the branch you are actually on green while the branch you are looking at is not. Only a *second* deliberate
act moves HEAD. So a visitor can explore the whole tree without leaving the session somewhere the next
visitor inherits.

The panel does not know how to scroll a chat log or how to move HEAD; it is handed a callback for each.
That is what keeps `chat_controller` out of its import graph, and what lets its behaviour be tested without
building one.
"""

__all__ = ["DPGChatGraphPanel"]

import dataclasses
import logging
import threading
import uuid
from typing import Callable, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

import dearpygui.dearpygui as dpg

from unpythonic import env, sym

from ..common import navhistory
from ..common.gui import animation as gui_animation
from ..common.gui import keyboardmark
from ..common.gui import utils as guiutils
from ..common.gui.xdotwidget import graph as xdotgraph
from ..common.gui.xdotwidget.widget import XDotWidget

from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa

from . import chatgraph
from . import chattree
from . import config as librarian_config

gui_config = librarian_config.gui_config

# How far the depth budget grows each time the reader clicks the "...N more" standing for elided ancestors.
# A factor rather than a step, because what is being escaped is a chat long enough that a step would have to
# be pressed many times over.
_DEPTH_EXPANSION_FACTOR = 2

# How far one arrow press moves the view, in screen pixels. `raven-xdot-viewer`'s value, because
# these are its keys and a reader who has learnt them there should find them behaving the same.
_PAN_AMOUNT = 10

_TOOLBAR_H = 34  # pixels, the row of view controls above the graph

# Font atlas sizes for graph labels. The renderer picks whichever is closest to the size it is drawing at,
# so this is a ladder rather than a choice: a label is legible at one zoom and unreadable at the next, and
# scaling one atlas across that range is what makes it look smeared. Same ladder `raven-xdot-viewer` uses.
_GRAPH_TEXT_FONT_SIZES = (4, 8, 16, 32, 64)


class DPGChatGraphPanel(gui_animation.Animation):
    """The chat graph view, as a self-contained panel.

    Create one inside whichever container is to hold it, then `set_size` it when that container resizes. It
    builds its own child window, so `show` and `hide` are unambiguous and the caller never has to know
    which widgets belong to the graph.

    It registers itself with Raven's animator, which is how it notices that the forest changed: it reads
    the datastore's change counter once per frame and rebuilds when that moves. Polling rather than
    notification, so the dozen-odd places that write to the tree do not each have to know a view exists —
    and honest polling, `chattree.Forest.generation` being a counter that only a mutation advances.
    """

    def __init__(self,
                 gui_parent: Union[int, str],
                 datastore: chattree.Forest,
                 app_state: dict,
                 themes_and_fonts: env,
                 width: int,
                 height: int,
                 on_preview: Optional[Callable[[str], None]] = None,
                 on_commit: Optional[Callable[[str], None]] = None,
                 input_blocked: Optional[Callable[[], bool]] = None,
                 graph_text_fonts: Optional[Sequence[Tuple[float, Union[int, str]]]] = None,
                 dark_mode: bool = True,
                 show: bool = False):
        """Build the panel.

        `gui_parent`: DPG container to build into.
        `datastore`: The chat forest to draw.
        `app_state`: Librarian's app state, read for `"HEAD"` and `"new_chat_HEAD"`. Read, never written:
                     moving HEAD is the caller's business, through `on_commit`.
        `themes_and_fonts`: The app's `guiutils.bootup` result, for the icon font and the disabled-widget
                            theme.
        `width`, `height`: Panel size in pixels.
        `on_preview`: Called with a chat node ID when a node *on the current branch* is previewed, so the
                      caller can scroll its chat log to that message. Not called for a node on another
                      branch: there is no message in the log for one, which is why the graph redraws around
                      it instead.
        `on_commit`: Called with a chat node ID when the reader deliberately moves HEAD there. The caller
                     does the moving; the panel sees the result the way it sees any other change.
        `input_blocked`: Predicate answering "is something on top of me?", passed to the widget. Its mouse
                         handlers are global, so without this the click that dismisses a dialog also lands
                         on the graph behind it.
        `graph_text_fonts`: `(size, font_id)` pairs for graph labels; the renderer picks the closest to the
                            size it is drawing at. `None` (the default) loads a ladder of sizes into
                            `themes_and_fonts` and uses that, since every caller wants the same one.
        `dark_mode`: Whether to invert the graph's lightness for a dark background. Raven's interface is
                     dark, so this defaults on.
        `show`: Whether the panel starts visible.
        """
        super().__init__(ambient=True)
        self.gui_uuid = str(uuid.uuid4())  # used in GUI widget tags

        self.datastore = datastore
        self.app_state = app_state
        self.themes_and_fonts = themes_and_fonts
        self._on_preview = on_preview
        self._on_commit = on_commit

        self._lock = threading.RLock()
        self._chat_graph: Optional[chatgraph.ChatGraph] = None
        self._view_state = chatgraph.ViewState(head_node_id=app_state["HEAD"],
                                               new_chat_node_id=app_state.get("new_chat_HEAD"))
        self._layout = chatgraph.LayoutConfig()
        # The cursor: which box a click or `Enter` acts on. A *graph* node name, so it can rest on a gap
        # box as well as on a message — see `chatgraph.ViewState.cursor_name`. Mirrored into the view
        # state, which is what draws the ring; kept here too so the panel can answer without a rebuild.
        self._cursor_name: Optional[str] = None
        self._is_shown = bool(show)
        # Whether the view has been framed yet. The first build fits the branch; later ones must not, or
        # the picture would re-frame itself under a reader once per turn of the conversation.
        self._framed = False
        # What the last rebuild was made from. Compared once per frame; a mismatch is the whole
        # "did anything change?" test.
        self._seen_generation: Optional[int] = None
        self._seen_head: Optional[str] = None

        self._commit_button_tag = f"chat_graph_commit_button_{self.gui_uuid}"  # tag
        self._dark_mode_button_tag = f"chat_graph_dark_mode_button_{self.gui_uuid}"  # tag
        self._back_button_tag = f"chat_graph_back_button_{self.gui_uuid}"  # tag
        self._forward_button_tag = f"chat_graph_forward_button_{self.gui_uuid}"  # tag

        # Where the reader has been. Panel-local: two panels would be two readers, and a shared history
        # would hand each of them the other's places.
        self._history = navhistory.NavigationHistory(is_valid=self._snapshot_is_reachable)
        self._has_keyboard = False  # set by the app, which owns the Tab cycle
        self._container = dpg.add_child_window(parent=gui_parent,
                                               width=width, height=height,
                                               no_scrollbar=True, no_scroll_with_mouse=True,
                                               show=self._is_shown,
                                               tag=f"chat_graph_panel_{self.gui_uuid}")  # tag
        self._build_toolbar(dark_mode=dark_mode)
        if graph_text_fonts is None:
            graph_text_fonts = [(size, guiutils.load_extra_font(themes_and_fonts, size,
                                                                "OpenSans", "Regular")[1])
                                for size in _GRAPH_TEXT_FONT_SIZES]
        self._graph_text_fonts = list(graph_text_fonts)
        # An inner window holding nothing but the graph, so the keyboard mark has something to frame that
        # is *not* the toolbar. A `Mark` binds a theme to its target, and DPG composes a theme down the
        # whole parent chain — so a mark on the outer panel reaches the toolbar's tooltips, which came out
        # blue-edged with their text squeezed against the border. Narrowing the theme by item type does
        # not stop it; putting the toolbar outside the marked subtree does, and the mark belongs on the
        # picture rather than on the buttons above it either way.
        #
        # `border=True` only so there is an edge to recolour. It costs nothing visually: an unlit mark is
        # transparent, and `padding=(0, 0)` undoes the window padding ImGui switches on with the border,
        # restoring the borderless layout to the pixel.
        self._canvas = dpg.add_child_window(parent=self._container,
                                            width=self._graph_w(width), height=self._graph_h(height),
                                            no_scrollbar=True, no_scroll_with_mouse=True,
                                            border=True,
                                            tag=f"chat_graph_canvas_{self.gui_uuid}")  # tag
        self._keyboard_mark = keyboardmark.Mark(self._canvas,
                                                kind=keyboardmark.MarkKind.PANEL,
                                                padding=(0, 0))

        self._widget = XDotWidget(parent=self._canvas,
                                  width=self._graph_w(width),
                                  height=self._graph_h(height),
                                  on_click=self._on_click,
                                  input_blocked=input_blocked,
                                  graph_text_fonts=graph_text_fonts,
                                  # The graph is the whole of what this panel is for, so space beyond it
                                  # is only distance to pan back across. Without the clamp, centring a
                                  # node near the bottom of the tree spends the lower half of the panel
                                  # on nothing -- and so does returning to a HEAD that has no replies
                                  # under it, which is the ordinary case.
                                  clamp_pan_to_graph=True,
                                  dark_mode=dark_mode,
                                  tag=f"chat_graph_widget_{self.gui_uuid}")  # tag

        gui_animation.animator.add(self)

    # ------------------------------------------------------------------
    # Construction

    @staticmethod
    def _graph_w(panel_w: int) -> int:
        """Return the graph widget's width inside a panel of width `panel_w`."""
        return max(1, panel_w - 2 * gui_config.margin)

    @staticmethod
    def _graph_h(panel_h: int) -> int:
        """Return the graph widget's height inside a panel of height `panel_h`."""
        return max(1, panel_h - _TOOLBAR_H - 2 * gui_config.margin)

    def _build_toolbar(self, dark_mode: bool) -> None:
        """Build the row of view controls above the graph.

        `dark_mode`: the widget's starting mode. An argument rather than something read off the widget,
        which does not exist yet — the toolbar is built first so that DPG lays it out above the graph.

        The view controls use the same glyphs, in the same order, as `raven-xdot-viewer`'s toolbar. The two
        show the same widget, so a reader who has used one should not have to relearn the other; where they
        differ it should be because the *job* differs, which is what the two buttons after the separator
        are.
        """
        def add_button(icon: str, callback: Callable, caption: str, tag: str,
                       enabled: bool = True, solid: bool = True) -> None:
            dpg.add_button(label=icon, callback=callback, width=gui_config.toolbutton_w,
                           enabled=enabled, tag=tag)
            font = (self.themes_and_fonts.icon_font_solid if solid
                    else self.themes_and_fonts.icon_font_regular)
            dpg.bind_item_font(tag, font)  # tag
            dpg.bind_item_theme(tag, self.themes_and_fonts.disablable_widget_theme)  # tag
            # A caption that never changes, so a plain DPG tooltip rather than the self-sizing one.
            with dpg.tooltip(tag):  # tag
                dpg.add_text(caption)

        with dpg.group(horizontal=True, parent=self._container):
            add_button(fa.ICON_SQUARE, lambda: self._widget.zoom_to_fit(),
                       "Zoom to fit [F]", f"chat_graph_fit_button_{self.gui_uuid}",  # tag
                       solid=False)
            add_button(fa.ICON_MAGNIFYING_GLASS, self.zoom_1_to_1,
                       "Actual size (1:1) [1 / numpad 1]", f"chat_graph_actual_size_button_{self.gui_uuid}")  # tag
            add_button(fa.ICON_MAGNIFYING_GLASS_PLUS, lambda: self._widget.zoom_in(),
                       "Zoom in [numpad +]", f"chat_graph_zoom_in_button_{self.gui_uuid}")  # tag
            add_button(fa.ICON_MAGNIFYING_GLASS_MINUS, lambda: self._widget.zoom_out(),
                       "Zoom out [numpad -]", f"chat_graph_zoom_out_button_{self.gui_uuid}")  # tag
            add_button(fa.ICON_SUN if dark_mode else fa.ICON_MOON, self.toggle_dark_mode,
                       "Switch to light mode" if dark_mode else "Switch to dark mode",
                       self._dark_mode_button_tag)

            guiutils.add_toolbar_separator(horizontal=True, toolbar_extent=_TOOLBAR_H,
                                           size=gui_config.toolbar_separator_w, line=False)

            # Back and forward through the views this panel has shown. Disabled until there is somewhere
            # to go, which is the convention Visualizer set: a button is enabled exactly when pressing it
            # would do something.
            add_button(fa.ICON_ARROW_LEFT, self.go_back,
                       "Back to the previous view [Alt+Left]", self._back_button_tag, enabled=False)
            add_button(fa.ICON_ARROW_RIGHT, self.go_forward,
                       "Forward again [Alt+Right]", self._forward_button_tag, enabled=False)

            # After the separator because a *branch* is a chat idea, where the plain fit beside the zoom
            # controls is the widget's own. The two answer different questions: fitting the picture shows
            # how wide the tree is, and fitting the branch shows the conversation.
            add_button(fa.ICON_ARROWS_UP_DOWN, self.fit_branch,
                       "Fit the current branch [B]", f"chat_graph_fit_branch_button_{self.gui_uuid}")  # tag
            add_button(fa.ICON_LOCATION_CROSSHAIRS, self.go_to_head,
                       "Back to where you are [Home]", f"chat_graph_home_button_{self.gui_uuid}")  # tag
            # The discoverable half of the commit gesture. Its caption names the fluent half, which is
            # otherwise unfindable: a second click on a box already clicked once.
            add_button(fa.ICON_CODE_BRANCH, self._commit_cursor,
                       "Switch to the previewed branch\n(or click its box a second time)",
                       self._commit_button_tag, enabled=False)

    # ------------------------------------------------------------------
    # Where the reader has been

    def _view_snapshot(self) -> Optional[Tuple]:
        """Return what the view is showing now, as something the history can hold and compare.

        **Identified by the picture, not by the state that produced it.** The branch is named by the tip
        it was drawn to, because that is what a reader means by "where I was looking" — and because the
        obvious alternative, the focus, does not survive being remembered. A focus of `None` means *follow
        HEAD*, so a snapshot holding one resolves to wherever HEAD is when it is restored rather than
        where it was when it was taken. Commit a branch switch and every such entry silently becomes the
        branch you just moved to, so Back walks through views that are all the same one.

        Naming the branch is also what makes the granularity right. Previewing a node moves the picture
        and is a step; clicking the same box again commits the switch, which moves HEAD and draws the very
        same branch — no step, because the reader did not go anywhere.

        A tuple rather than the `ViewState`: immutable, so a later edit cannot reach back and change what
        was remembered, and comparable by value, which is what a commit needs.

        The preview ring is deliberately left out. It is a question waiting for a second click, and
        arriving somewhere with someone else's question already posed is not what Back means.

        Returns `None` before the first picture exists, there being no view to record yet.

        The caller holds the lock.
        """
        if self._chat_graph is None or not self._chat_graph.spine:
            return None
        state = self._view_state
        return (self._chat_graph.spine[-1],
                tuple(sorted(state.sibling_focus.items())),
                frozenset(state.expanded_tool_turns))

    def _snapshot_is_reachable(self, snapshot: Tuple) -> bool:
        """Return whether a remembered view can still be gone back to.

        Only the branch tip is checked. The rest of a snapshot degrades on its own — a sibling window
        centred on a node that has gone falls back to the branch, which is what it does for a level nobody
        has touched — where a tip that has gone is the one thing `chatgraph.build` cannot draw around.

        The chat forest is written by others: a reply in progress adds nodes, and the cleanup dialog
        removes them. So this is asked again on every step rather than once, and a node that is gone is
        stepped over rather than stopping the walk.
        """
        with self.datastore.lock:
            return snapshot[0] in self.datastore.nodes

    def _restore_snapshot(self, snapshot: Tuple) -> bool:
        """Put the view back to a remembered snapshot. Returns whether it was applied.

        The tip is handed back as the focus, which draws the branch it was the tip of. HEAD is not
        touched: this restores what the reader was *looking at*, never where they are.
        """
        branch_tip, sibling_focus, expanded = snapshot
        with self._lock:
            self._view_state.focus_node_id = branch_tip
            self._view_state.sibling_focus = dict(sibling_focus)
            self._view_state.expanded_tool_turns = set(expanded)
        self._set_cursor(None)  # which redraws; arriving with someone else's question posed is not Back
        return True

    def _remember_view(self) -> None:
        """Record where the view is now, if it has moved. Cheap to call after any gesture.

        A commit that changes nothing is a no-op, so this does not need to know which gestures move the
        view — which is what keeps the knowledge in one place instead of at each of the four call sites.
        """
        with self._lock:
            snapshot = self._view_snapshot()
        if snapshot is None:
            return
        self._history.commit(snapshot)
        self._update_history_buttons()

    def _update_history_buttons(self) -> None:
        """Enable each history button exactly when there is a view to go to in that direction."""
        for tag, enabled in ((self._back_button_tag, self._history.can_go_back),
                             (self._forward_button_tag, self._history.can_go_forward)):
            if dpg.does_item_exist(tag):  # tag
                (dpg.enable_item if enabled else dpg.disable_item)(tag)  # tag

    def go_back(self) -> None:
        """Return to the previous view. Does not move HEAD."""
        self._history.back(apply=self._restore_snapshot)
        self._update_history_buttons()

    def go_forward(self) -> None:
        """Undo a `go_back`. Does not move HEAD."""
        self._history.forward(apply=self._restore_snapshot)
        self._update_history_buttons()

    # ------------------------------------------------------------------
    # The keyboard

    def _get_has_keyboard(self) -> bool:
        """Return whether this panel is the one the keys are going to."""
        return self._has_keyboard

    def _set_has_keyboard(self, value: bool) -> None:
        """Take or release the keyboard, lighting or darkening the panel's border to say which.

        A hidden panel refuses to take it, rather than taking it and hiding the fact. The two are
        different states and only one of them is coherent: a panel that answered `True` while showing no
        border would be claiming keys the reader has no way to see it holding.
        """
        self._has_keyboard = bool(value) and self._is_shown
        self._keyboard_mark.lit = self._has_keyboard

    has_keyboard = property(fget=_get_has_keyboard, fset=_set_has_keyboard,
                            doc="Whether the keys are going to this panel. Set by the app, which owns the "
                                "cycle; the panel's job is to say so, which it does in the blue every "
                                "other keyboard home in the constellation wears. A hidden panel cannot "
                                "hold it — assigning `True` to one leaves this `False`.")

    def handle_key(self, key: int, ctrl: bool = False, shift: bool = False, alt: bool = False) -> bool:
        """Act on a key that belongs to this panel. Returns whether it was one.

        The view controls are `raven-xdot-viewer`'s, key for key, for the reason the toolbar's glyphs are:
        the two show the same widget, so a reader who has learnt one should not have to learn the other.
        What is added here is what this panel has and that viewer does not — a cursor to move over the
        boxes, a history, a branch to frame, and a HEAD to come back to.

        The arrows move the cursor and `Enter` acts on it, which is the pair that makes this view usable
        without a pointer at all. Panning falls to Shift+arrows: it is the secondary gesture here, the
        mouse having a drag and the wheel a zoom, while stepping between boxes has no other spelling.

        Returns `False` for anything it does not claim, so the caller can go on looking.
        """
        # Alt+Left / Alt+Right for history, which is what a browser and every file manager bind, and what
        # the file dialog's own navigation history is specified to use.
        if alt:
            if key == dpg.mvKey_Left:
                self.go_back()
                return True
            if key == dpg.mvKey_Right:
                self.go_forward()
                return True
            return False
        # Ctrl+arrows step along the siblings, Ctrl+Shift+arrows by ten, Ctrl+Home and Ctrl+End to the
        # ends. The chat log's keys for the chat log's verbs, on the same tree.
        #
        # Before the Shift branch below, because Ctrl+Shift is a Ctrl gesture here and testing Shift
        # first would pan on it instead of stepping ten.
        #
        # Claiming these also closes a hole. They fell through to the app, whose own Ctrl+arrows step the
        # siblings of the *marked message in the chat log* -- and those move HEAD. So a browsing gesture
        # aimed at the graph switched branch, silently, from a pane that was not showing the message it
        # acted on.
        if ctrl:
            if key == dpg.mvKey_Left:
                self._step_sibling("prev", step=10 if shift else 1)
            elif key == dpg.mvKey_Right:
                self._step_sibling("next", step=10 if shift else 1)
            elif key == dpg.mvKey_Home:
                self._step_sibling("prev", step=None)
            elif key == dpg.mvKey_End:
                self._step_sibling("next", step=None)
            else:
                return False
            return True
        # Shift+arrows pan. Panning is the secondary gesture here -- the mouse drags, the wheel zooms, and
        # what the *keyboard* wants from a graph it can commit from is to move between boxes, which is
        # what the bare arrows do.
        if shift:
            if key == dpg.mvKey_Up:
                self._widget.pan_by(dx=0, dy=+_PAN_AMOUNT)
            elif key == dpg.mvKey_Down:
                self._widget.pan_by(dx=0, dy=-_PAN_AMOUNT)
            elif key == dpg.mvKey_Left:
                self._widget.pan_by(dx=+_PAN_AMOUNT, dy=0)
            elif key == dpg.mvKey_Right:
                self._widget.pan_by(dx=-_PAN_AMOUNT, dy=0)
            else:
                return False
            return True

        if key == dpg.mvKey_Up:
            self._move_cursor("up")
        elif key == dpg.mvKey_Down:
            self._move_cursor("down")
        elif key == dpg.mvKey_Left:
            self._move_cursor("left")
        elif key == dpg.mvKey_Right:
            self._move_cursor("right")
        # Enter does to the box under the cursor what a click on it does, which is the whole of what it
        # means: on a message, switch to that branch; on a gap, open it. One verb, so that a reader who
        # has learnt what a box does by clicking it already knows what Enter will do.
        elif key == dpg.mvKey_Return:
            self._activate_cursor()
        # Backspace folds an opened tool round back up, from anywhere inside it. Its own key because both
        # of the obvious ones are spoken for: Esc puts the cursor away, and Enter on a drawn tool result
        # has to commit, that being the capability opening the round exists to restore.
        elif key == dpg.mvKey_Back:
            self._collapse_round()
        # Esc puts the cursor away. Nothing is undone by it -- the ring commits nothing on its own, being
        # a place to stand rather than a change -- so this is "I am done pointing at things", and the next
        # arrow starts again from HEAD.
        elif key == dpg.mvKey_Escape:
            self._set_cursor(None)
        # Numpad only, and the main row deliberately left out. `mvKey_Plus` is a pre-2.0 constant (61)
        # against a key that arrives as 602 -- measured, in `briefs/reference/dpg-keycodes.md` -- so a
        # branch on it never runs, and the main-row pair is separately known to misbehave on a Nordic
        # layout (`TODO_DEFERRED.md`, "Main-row `+` and `-` both zoom out"). Binding it here would add a
        # fourth broken instance of a filed bug rather than a working key; the wheel and the toolbar cover
        # zoom meanwhile.
        elif key == dpg.mvKey_Add:
            self._widget.zoom_in()
        elif key == dpg.mvKey_Subtract:
            self._widget.zoom_out()
        elif key == dpg.mvKey_F:
            self._widget.zoom_to_fit()
        # "1" for 1:1, on the main row and the numpad both. The main row is where the key is on a US,
        # Nordic or German layout; on a French one the digits are shifted, so the physical key sends
        # this code while its cap reads "&" and the key labelled 1 sends nothing we bind. The numpad is
        # the layout-stable path, exactly as it is for zoom, and binding both costs a line.
        elif key in (dpg.mvKey_1, dpg.mvKey_NumPad1):
            self.zoom_1_to_1()
        elif key == dpg.mvKey_B:
            self.fit_branch()
        elif key == dpg.mvKey_Home:
            self.go_to_head()
        else:
            return False
        return True

    def zoom_1_to_1(self) -> None:
        """Set the zoom to 1:1, leaving the pan where it is."""
        self._widget.set_zoom(1.0, animate=True)

    def fit_branch(self) -> None:
        """Frame the branch on screen — the spine, and nothing beside it.

        A level of this picture can run to thousands of graph units where the branch is one column wide,
        so fitting the whole thing is width-limited and lands at a zoom where the renderer stops drawing
        text at all. Fitting the branch is height-limited instead: the labels stay legible and the width
        is left to overflow into a pan.
        """
        with self._lock:
            chat_graph = self._chat_graph
        if chat_graph is not None:
            self._widget.zoom_to_bbox(*chat_graph.spine_bbox, animate=True)

    def toggle_dark_mode(self) -> None:
        """Flip the graph between the dark and light palettes, and relabel the button."""
        self._widget.dark_mode = not self._widget.dark_mode
        dark = self._widget.dark_mode
        with guiutils.nonexistent_ok():
            dpg.set_item_label(self._dark_mode_button_tag, fa.ICON_SUN if dark else fa.ICON_MOON)

    # ------------------------------------------------------------------
    # Public API

    def _get_is_shown(self) -> bool:
        """Return whether the panel is currently shown."""
        return self._is_shown

    is_shown = property(fget=_get_is_shown,
                        doc="Whether the panel is shown. Change it with `show`, `hide` or `toggle`.")

    def show(self) -> None:
        """Show the panel, refreshing the picture first — it may have gone stale while nobody was looking."""
        self._is_shown = True
        self.refresh()
        with guiutils.nonexistent_ok():
            dpg.show_item(self._container)

    def hide(self) -> None:
        """Hide the panel, and give up the keyboard if it had it."""
        self._is_shown = False
        # A hidden panel cannot be the keyboard home: its border is not on screen to say so, and the
        # keys would go somewhere the reader cannot see.
        self.has_keyboard = False
        with guiutils.nonexistent_ok():
            dpg.hide_item(self._container)

    def toggle(self) -> bool:
        """Show the panel if it is hidden, hide it if it is shown. Returns the new state."""
        if self._is_shown:
            self.hide()
        else:
            self.show()
        return self._is_shown

    def set_size(self, width: int, height: int) -> None:
        """Resize the panel and the graph inside it."""
        with guiutils.nonexistent_ok():
            dpg.set_item_width(self._container, width)
            dpg.set_item_height(self._container, height)
            dpg.set_item_width(self._canvas, self._graph_w(width))
            dpg.set_item_height(self._canvas, self._graph_h(height))
        self._widget.set_size(self._graph_w(width), self._graph_h(height))

    def refresh(self) -> None:
        """Rebuild the picture from the datastore, keeping the reader where they were looking.

        Called for itself whenever the forest or HEAD changes; call it directly after doing something the
        change counter cannot see.
        """
        with self._lock:
            self._view_state.head_node_id = self.app_state["HEAD"]
            self._view_state.new_chat_node_id = self.app_state.get("new_chat_HEAD")
            generation = self.datastore.generation

            chat_graph = self._try_build()
            if chat_graph is None:  # the node the picture was drawn around is gone -- fall back to HEAD
                logger.info("DPGChatGraphPanel.refresh: the focused node is gone; falling back to HEAD")
                self._view_state.focus_node_id = None
                # The cursor is deliberately *not* cleared here. Its box is very likely gone too — the
                # focus is usually the node the cursor is on, a click having set both — but that is what
                # the landing policy below is for, and clearing it first would throw away the one thing
                # that says where the reader was.
                chat_graph = self._try_build()
            if chat_graph is None:
                logger.warning("DPGChatGraphPanel.refresh: HEAD is gone too; leaving the picture as it is")
                return

            cursor_before = self._view_state.cursor_name
            self._reland_cursor(chat_graph)
            if self._view_state.cursor_name != cursor_before:
                # The ring is part of the picture, so the build above put it on the box the cursor has
                # this instant left — a box that build no longer contains, which is to say nowhere. The
                # cursor is then invisible while still answering the arrow keys, and the reader is moving
                # a mark they cannot see.
                #
                # Build once more, now that it has landed. Only when it moved, which is only when its box
                # was destroyed: an ordinary rebuild pays nothing, and the paths that move the cursor
                # themselves set it before refreshing, so they pay nothing either.
                rebuilt = self._try_build()
                if rebuilt is not None:
                    chat_graph = rebuilt
            self._chat_graph = chat_graph
            self._seen_generation = generation
            self._seen_head = self._view_state.head_node_id
            self._widget.set_graph(chat_graph.graph)
            anchor = (self._cursor_name
                      or self._view_state.focus_node_id
                      or self._view_state.head_node_id)

        # The first picture is framed; every one after it only follows the anchor.
        #
        # Framing means 1:1 on HEAD -- what the crosshair does, and where every other path leaves the
        # zoom. A fitted zoom would be a computed one, so the view would open at a different size for
        # every chat and at a size the reader cannot get back to by any other means. 1:1 is also what the
        # node font is sized for.
        #
        # And only the first, because `set_graph` leaves pan and zoom alone: an anchor that kept its place
        # then needs nothing done, and one that moved is followed smoothly. That is what keeps the picture
        # still while a reply is arriving and the tree gains a node per round -- a re-frame on every
        # rebuild would make it lurch once per turn.
        if not self._framed:
            self._framed = True
            self._frame_on_head(chat_graph, animate=False)
            self._remember_view()  # the view opened on is the one Back should eventually reach
        elif chat_graph.graph.get_node_by_name(anchor) is not None:
            self._widget.pan_to_node(anchor, animate=True)
        else:
            self._widget.zoom_to_fit(animate=True)

    def _try_build(self) -> Optional[chatgraph.ChatGraph]:
        """Build the picture, or return `None` if the node it would be drawn around is gone."""
        try:
            return chatgraph.build(self.datastore, self._view_state, self._layout,
                                   measure_text=self._measure_text)
        except KeyError:
            return None

    def _measure_text(self, text: str, font_size: float) -> Optional[float]:
        """Return how wide `text` is at `font_size`, in graph units, or `None` if DPG cannot say yet.

        This is what `chatgraph` cannot do for itself: it holds no DPG, and an average glyph advance is
        enough to size a box but not to centre text inside one — the renderer starts centred text at
        `centre - w/2`, so an error in the width displaces the glyphs by half of it.

        Measured against whichever font atlas is nearest the requested size and scaled, the atlases being
        a ladder rather than a continuum.
        """
        atlas_size, font_id = min(self._graph_text_fonts, key=lambda pair: abs(pair[0] - font_size))
        measured = dpg.get_text_size(text, font=font_id)
        if not measured:  # no atlas until a frame has been rendered; an ordinary state, not a fault
            return None
        return measured[0] * (font_size / atlas_size)

    def _frame_on_head(self, chat_graph: chatgraph.ChatGraph, animate: bool) -> bool:
        """Put HEAD in the lower third of the panel at 1:1. Returns whether HEAD was there to go to.

        HEAD is placed low rather than centred because what a reader wants around it is what came before —
        below HEAD there is at most one row of replies, so centring it spends half the panel on nothing.

        The one framing the panel has, used both on opening and by the crosshair. Two would mean a view
        the reader arrives at on startup and cannot get back to, since only one of them has a button.
        """
        head_node = chat_graph.graph.get_node_by_name(self._view_state.head_node_id)
        if head_node is None:
            return False
        self._widget.set_zoom(1.0, animate=animate)
        # A sixth of the panel's height above centre puts HEAD two-thirds of the way down. In graph units
        # at 1:1, which is what the zoom above is settling to.
        _, panel_h = self._widget.get_size()
        self._widget.pan_to_point(head_node.x, head_node.y - panel_h / 6.0, animate=animate)
        return True

    def go_to_head(self) -> None:  # noqa: D401 -- the docstring below is a description, not an imperative
        """Abandon any preview and put the reader back at HEAD, at 1:1."""
        with self._lock:
            self._view_state.focus_node_id = None
        self._framed = True  # this is a framing of its own; the refresh below must not override it
        self._set_cursor(None)  # which redraws

        self._remember_view()  # returning to HEAD is a place too, and one worth being able to leave again

        with self._lock:
            chat_graph = self._chat_graph
        if chat_graph is None or not self._frame_on_head(chat_graph, animate=True):
            return
        # Flash it. The view slides and the zoom changes at the same time, so "the box you were brought
        # back to" is not obvious from the motion alone -- and HEAD is deliberately off-centre here, which
        # removes the other cue. Not done on opening, where nothing moved and there is nothing to explain.
        self._widget.flash_nodes({self._view_state.head_node_id})

    def destroy(self) -> None:
        """Tear the panel down. Reverse of the order things were set up in."""
        gui_animation.animator.cancel(self)
        self._keyboard_mark.detach()
        self._widget.destroy()
        with guiutils.nonexistent_ok():
            dpg.delete_item(self._container)

    # ------------------------------------------------------------------
    # Keeping up with the forest

    def render_frame(self, t: int) -> sym:
        """Adapter; hook for Raven's GUI animation system.

        See `raven.common.gui.animation.Animation`.
        """
        if self._is_shown and self._is_stale():
            self.refresh()
        return gui_animation.action_continue

    def _is_stale(self) -> bool:
        """Return whether anything the picture depends on has changed since it was built.

        Two things, and neither implies the other: the forest gains and loses nodes, and HEAD moves among
        nodes that were already there. A branch switch changes only the second.
        """
        return (self.datastore.generation != self._seen_generation
                or self.app_state["HEAD"] != self._seen_head)

    # ------------------------------------------------------------------
    # Clicks

    def _on_click(self, element, button: int) -> None:
        """Handle a click on the graph.

        `element`: The `Node` or `Edge` the widget hit. Edges are ignored: every edge here is a parent
                   link, so clicking one asks nothing the two boxes at its ends do not already answer.
        """
        if button != dpg.mvMouseButton_Left:
            return
        if not isinstance(element, xdotgraph.Node):
            return
        with self._lock:
            chat_graph = self._chat_graph
        if chat_graph is None:
            return
        self._activate(element.internal_name)

    def _activate(self, name: str) -> None:
        """Do to the box called `name` whatever acting on it means.

        The one place that answers it, because a click and `Enter` must not be able to disagree: the
        keyboard's whole claim on this view is that it reaches the same things the pointer does, and two
        dispatches would drift the first time a gap kind gained a verb.
        """
        with self._lock:
            chat_graph = self._chat_graph
        ref = chat_graph.ref_for(name) if chat_graph is not None else None
        if ref is None:
            logger.warning(f"DPGChatGraphPanel._activate: no ref for graph node '{name}'; ignoring")
            return

        if isinstance(ref, chatgraph.ChatNodeRef):
            self._click_chat_node(ref)
        elif isinstance(ref, chatgraph.SiblingGapRef):
            self._move_sibling_window(ref)
        elif isinstance(ref, chatgraph.DepthGapRef):
            self._widen_depth_window()
        elif isinstance(ref, chatgraph.ToolRoundGapRef):
            self._set_round_expanded(ref.owner_node_id, expanded=True)
        elif isinstance(ref, chatgraph.SubtreeGapRef):
            self._look_inside(ref)
        elif isinstance(ref, chatgraph.RootGapRef):
            # Inert in v1, deliberately: switching to a chat written under an older character card would
            # leave the configured avatar and voice running against a different system prompt.
            logger.info(f"DPGChatGraphPanel._activate: {ref.hidden_count} chat(s) under other cards; "
                        "reaching them is not implemented")

        # Once, here, rather than in each of the four branches above. A commit that changes nothing is a
        # no-op, so the ones that only scrolled the chat log cost nothing and need no special case.
        self._remember_view()

    def _click_chat_node(self, ref: chatgraph.ChatNodeRef) -> None:
        """Preview a chat node — or commit to it, if the cursor was already on it."""
        with self._lock:
            already_previewed = (ref.node_id == self._cursor_name)
            chat_graph = self._chat_graph
        if already_previewed:
            self._commit(ref.node_id)
            return

        # Two independent questions, and answering the second with the first is a bug this had:
        #
        #   - Is the node *drawn as part of the branch on screen*? That decides whether the picture has
        #     to move. Off the drawn branch, it is shown only as somebody's sibling, and its own
        #     continuation is a "...N more" -- so the graph redraws around it.
        #   - Is there a *message in the chat log* for it? That is a question about HEAD's branch, since
        #     the log shows that one and no other.
        #
        # They agree until a preview puts a different branch on screen. Then a node on HEAD's branch can
        # be off the drawn one, and treating "there is a message for it" as "the picture already shows
        # it" leaves the reader clicking a box whose conversation stays collapsed behind it.
        on_drawn_branch = ref.node_id in set(chat_graph.spine) if chat_graph is not None else False

        if not on_drawn_branch:
            with self._lock:
                self._view_state.focus_node_id = ref.node_id
        self._set_cursor(ref.node_id)  # redraws, so the branch change above lands with it
        self._widget.pan_to_node(ref.node_id, animate=True)

        if ref.on_current_branch and self._on_preview is not None:
            self._on_preview(ref.node_id)

    def _move_sibling_window(self, ref: chatgraph.SiblingGapRef) -> None:
        """Re-centre one level's window on the middle of the run this gap hides."""
        with self._lock:
            self._view_state.sibling_focus[ref.parent_node_id] = ref.recenter_on
        # The cursor lands on the node the window recentred on, which is also the one the view pans to.
        # Clearing it instead — which this did while the ring was placed only by clicking — leaves a
        # keyboard reader with nothing to move from, at the one gesture whose whole purpose is to carry
        # them across a level too wide to walk. It lands where any other cursor move lands it — one act on
        # it commits — because the alternative needs a second, weaker cursor state that nothing on screen
        # distinguishes, and a ring that sometimes acts and sometimes only arms is a ring that says less.
        self._set_cursor(ref.recenter_on)
        self._widget.pan_to_node(ref.recenter_on, animate=True)

    def _widen_depth_window(self) -> None:
        """Show more of the elided middle of the branch."""
        with self._lock:
            self._layout = dataclasses.replace(
                self._layout,
                max_visible_depth=self._layout.max_visible_depth * _DEPTH_EXPANSION_FACTOR)
        self.refresh()

    def _look_inside(self, ref: chatgraph.SubtreeGapRef) -> None:
        """Draw the picture around an off-branch node, so that what continues below it becomes visible."""
        with self._lock:
            self._view_state.focus_node_id = ref.node_id
        self._set_cursor(ref.node_id)

    def _set_round_expanded(self, owner_node_id: str, expanded: bool) -> None:
        """Draw one tool round's results as boxes of their own, or fold them back behind a gap box.

        `owner_node_id`: The message that asked for the tools.

        Folded, a round's results have no box, so no cursor reaches them and none can be made HEAD — which
        the chat log allows and this view otherwise would not. Opening the gap is what restores that, and
        it is the gap's ordinary verb rather than a gesture of its own.
        """
        with self._lock:
            if expanded:
                self._view_state.expanded_tool_turns.add(owner_node_id)
            else:
                self._view_state.expanded_tool_turns.discard(owner_node_id)
        # The cursor's own box is the one thing this is certain to destroy — the gap when opening it, a
        # result when folding one away — so the landing policy inside the rebuild is what moves it, onto
        # whichever box now stands for what the reader was pointing at. Which is the first result on the
        # way in and the gap itself on the way out, both without being told.
        self.refresh()
        self._enable_commit(self._cursor_chat_node_id() is not None)

    def _collapse_round(self) -> None:
        """Fold the tool round the cursor is inside back into a gap box, if it is inside one."""
        owner_node_id = self._round_at_cursor()
        if owner_node_id is None:
            return
        self._set_round_expanded(owner_node_id, expanded=False)
        self._remember_view()

    def _round_at_cursor(self) -> Optional[str]:
        """Return the owner of the expanded tool round the cursor is inside, or `None` if it is in none.

        Inside means on the message that asked for the tools or on one of the results it got back — the
        boxes a reader would point at to say *this round*. A round the picture draws open because it is
        too small to be worth folding is not one of these: closing it would hide nothing.
        """
        with self._lock:
            chat_graph = self._chat_graph
        node_id = self._cursor_chat_node_id()
        if node_id is None or chat_graph is None:
            return None
        for owner_node_id, results in chat_graph.expanded_rounds.items():
            if node_id == owner_node_id or node_id in results:
                return owner_node_id
        return None

    # ------------------------------------------------------------------
    # Committing

    def _commit_cursor(self) -> None:
        """Toolbar button: move HEAD to the message the cursor is on, if it is on one."""
        node_id = self._cursor_chat_node_id()
        if node_id is not None:
            self._commit(node_id)

    def _commit(self, node_id: str) -> None:
        """Move HEAD to `node_id`, through the caller."""
        with self._lock:
            self._view_state.focus_node_id = None
        # Nothing left armed. The reader has arrived where they were going, and the first movement key
        # after this starts from HEAD — which is now the node just committed to, so the cursor picks up
        # exactly where the gesture left them.
        self._set_cursor(None)
        if self._on_commit is not None:
            self._on_commit(node_id)
        else:
            logger.warning("DPGChatGraphPanel._commit: no `on_commit` was given; HEAD not moved")

        # Adopt the move now rather than waiting for the poll to notice it next frame. Waiting draws one
        # frame of the branch just left -- the focus has been cleared, so the picture falls back to a HEAD
        # that has not caught up yet -- and, worse, anything reading the view in between sees that stale
        # branch as though it were where the reader is. The history is the thing that reads it.
        with self._lock:
            self._view_state.head_node_id = self.app_state["HEAD"]
        self.refresh()

    def _enable_commit(self, enabled: bool) -> None:
        """Enable or disable the toolbar's commit button."""
        with guiutils.nonexistent_ok():
            if enabled:
                dpg.enable_item(self._commit_button_tag)
            else:
                dpg.disable_item(self._commit_button_tag)

    def _set_cursor(self, name: Optional[str]) -> None:
        """Move the cursor to a box, or clear it, and redraw so the ring moves with it.

        `name`: A graph node name, as `chatgraph.ViewState.cursor_name` takes — so a gap box is as
                addressable as a message.

        The mark lives in the picture rather than in the widget's highlight state. That state is shared
        with hover and has one pair of colours, so a cursor drawn through it is indistinguishable from a
        hover — and worse, a node lit and left lit reads as *the important one*, which is HEAD's job.
        Keeping it in the `Graph` also means it survives a rebuild without being re-applied.

        Redraws unconditionally, including when the cursor was already where it is being put.
        """
        # Unconditionally, because skipping the rebuild when the cursor has not moved would be free only
        # if the cursor were the sole reason to redraw, and it is not: every caller has just changed
        # something else as well — a level's window, the focused branch — and one that happened to leave
        # the cursor where it was would silently lose its own change. A rebuild costs about a millisecond
        # and none of these paths is hot.
        self._set_cursor_fields(name)
        # The rebuild first, then the button: whether there is anything to commit to is a question about
        # the picture that comes *out* of it. A cursor moved to a box the rebuild then drops has nothing
        # to commit to, and asking before the rebuild would have said otherwise.
        self.refresh()
        self._enable_commit(self._cursor_chat_node_id() is not None)

    def _move_cursor(self, direction: str) -> None:
        """Step the cursor one box `direction`, planting it first if it is nowhere.

        `direction`: As `chatgraph.neighbor_of` takes it.

        The first press after the graph takes the keyboard plants the cursor without moving it. HEAD is
        already the loudest thing in the picture, so the ring appearing on it teaches what the ring means
        on a box the reader can already find — where stepping away from an unseen resting place would
        show them a mark somewhere they never saw it leave.
        """
        with self._lock:
            chat_graph = self._chat_graph
            at = self._cursor_name
        if chat_graph is None:
            return
        if at is None:
            self._set_cursor(self._cursor_home(chat_graph))
            return
        stepped = chatgraph.neighbor_of(chat_graph.graph, at, direction)
        if stepped is None:  # the edge of the picture; staying put is the answer
            return
        self._activate_or_move(stepped)

    def _step_sibling(self, direction: str, step: Optional[int]) -> None:
        """Move the cursor along the siblings at its own level, sliding that level's window to suit.

        `direction`: `"prev"` or `"next"`.
        `step`: How many siblings to move, or `None` for as far as they go.

        The verbs and the keys are the chat log's — Ctrl+arrows by one, Ctrl+Shift+arrows by ten,
        Ctrl+Home and Ctrl+End to the ends — because they are the same verbs, and a reader should not have
        to learn them twice for the two views of one tree. What differs is what they *do*: in the log they
        move HEAD, having no other way to show you a sibling, while here they only move the cursor. That
        is the whole distinction between the two views, not an inconsistency between them.

        This is what the bare arrows cannot do. They step from box to box, and a level's window shows a
        handful of a fan that can run to hundreds; going further means moving the window, which is the
        gesture a sibling gap offers to the pointer and this offers to the keyboard.
        """
        node_id = self._cursor_sibling_anchor()
        if node_id is None:
            return
        with self.datastore.lock:
            siblings, index = self.datastore.get_siblings(node_id)
            parent_node_id = self.datastore.get_parent(node_id)
        if not siblings or index is None:
            return

        if step is None:
            target_index = 0 if direction == "prev" else len(siblings) - 1
        else:
            moved = index - step if direction == "prev" else index + step
            # Clamped rather than wrapped, and clamped rather than refused: a step of ten near an end
            # should land on the end, which is what a reader asking to move ten in that direction means.
            target_index = max(0, min(len(siblings) - 1, moved))
        target = siblings[target_index]
        if target == node_id:
            return

        if parent_node_id is not None:
            # A root has no level to slide — its siblings are the other roots, drawn as the inert roots
            # gap — so there is nothing to record for it, and the cursor move below is all that happens.
            with self._lock:
                self._view_state.sibling_focus[parent_node_id] = target
        self._set_cursor(target)
        self._widget.pan_to_node(target, animate=True)
        self._remember_view()

    def _cursor_sibling_anchor(self) -> Optional[str]:
        """Return the chat node whose siblings a sibling step should move along, or `None` if there is none.

        The cursor's own node where it is on a message. Where it is on a *sibling* gap, the first node
        that gap hides — the gap stands for a run of siblings at that very level, so stepping from it
        continues along the level the reader is looking at rather than refusing because the box under the
        ring is not a message.
        """
        with self._lock:
            name = self._cursor_name
            chat_graph = self._chat_graph
        ref = chat_graph.ref_for(name) if (name is not None and chat_graph is not None) else None
        if isinstance(ref, chatgraph.ChatNodeRef):
            return ref.node_id
        if isinstance(ref, chatgraph.SiblingGapRef) and ref.hidden_node_ids:
            return ref.hidden_node_ids[0]
        return None

    def _activate_cursor(self) -> None:
        """Act on the box the cursor is on, if it is on one."""
        with self._lock:
            name = self._cursor_name
        if name is not None:
            self._activate(name)

    def _activate_or_move(self, name: str) -> None:
        """Put the cursor on `name`, taking the route a click takes so the two cannot behave differently.

        A message goes through the click path — which scrolls the chat log to it, or redraws the picture
        around its branch — because arriving at a box by stepping and by clicking should show the same
        thing. A gap is only stepped onto; acting on it is `Enter`'s business, not the arrow's.
        """
        with self._lock:
            chat_graph = self._chat_graph
        ref = chat_graph.ref_for(name) if chat_graph is not None else None
        if isinstance(ref, chatgraph.ChatNodeRef):
            self._click_chat_node(ref)
        else:
            self._set_cursor(name)
            self._widget.pan_to_node(name, animate=True)

    def _cursor_home(self, chat_graph: chatgraph.ChatGraph) -> Optional[str]:
        """Return where a cursor that is nowhere should appear.

        HEAD, which is where the reader is. Failing that — a Back step can restore a branch HEAD is not
        on — whatever the picture is drawn around, and failing that the branch's own top, so that the
        answer is never "nowhere" while there is a picture to stand in.
        """
        with self._lock:
            for candidate in (self._view_state.head_node_id, self._view_state.focus_node_id):
                if candidate is not None and candidate in chat_graph.refs:
                    return candidate
        for node_id in chat_graph.spine:
            if node_id in chat_graph.refs:
                return node_id
        return None

    def _set_cursor_fields(self, name: Optional[str]) -> None:
        """Put the cursor at `name` without redrawing. For callers that are inside a rebuild already."""
        with self._lock:
            self._cursor_name = name
            self._view_state.cursor_name = name

    def _reland_cursor(self, chat_graph: chatgraph.ChatGraph) -> None:
        """Move the cursor onto a box that exists in `chat_graph`, if the one it was on does not.

        A rebuild can destroy the box the cursor is on, and several ordinary things cause one: a reply
        arrives, HEAD moves, a level's window slides, and — by design — opening a gap, whose whole
        purpose is to replace itself with what it was hiding. The mouse needed no policy for this because
        the ring was only ever placed by a click and cleared by code; a cursor is a place the reader
        expects to still be standing afterwards.

        So: keep the node the cursor stood for, and follow it to whatever box represents it now — its own
        if it is drawn, else whatever stands for it, else the nearest drawn ancestor. HEAD is the last
        resort, for a node that has left the datastore entirely: jumping there is a jump to who knows
        where, possibly off the picture, and an ancestor of where the reader was is nearer to where they
        were than the tip of a branch they may not even be looking at (Juha, 2026-09-03).

        The caller holds the lock, and this must not redraw: it runs inside the rebuild whose result it is
        landing on.
        """
        name = self._cursor_name
        if name is None or name in chat_graph.refs:
            return

        # Which chat node the cursor *meant*. A message box is named for its node; a gap box is named for
        # the first node it hides, and that node is the best answer to "what was I pointing at" once the
        # gap itself is gone.
        was = self._chat_graph.ref_for(name) if self._chat_graph is not None else None
        if isinstance(was, chatgraph.ChatNodeRef):
            stood_for: Optional[str] = was.node_id
        elif was is not None and was.hidden_node_ids:
            stood_for = was.hidden_node_ids[0]
        else:
            stood_for = None

        landed = (chat_graph.representative_of(stood_for, datastore=self.datastore)
                  if stood_for is not None else None)
        if landed is None:
            landed = self._what_was_above(name, chat_graph)
        if landed is None and self._view_state.head_node_id in chat_graph.refs:
            landed = self._view_state.head_node_id
        self._set_cursor_fields(landed)

    def _what_was_above(self, name: str, chat_graph: chatgraph.ChatGraph) -> Optional[str]:
        """Return the box that stood above `name` in the *previous* picture, if it is still in this one.

        For the one case the forest cannot answer: the node the cursor was on has been deleted, so there
        is no lineage left to walk up. The old picture still knows — its edges are the parent links, and
        the box above is where that message hung from.

        The caller holds the lock.
        """
        previous = self._chat_graph
        if previous is None:
            return None
        above = chatgraph.neighbor_of(previous.graph, name, "up")
        if above is None:
            return None
        if above in chat_graph.refs:  # still drawn, and under the same name
            return above
        # It is not, so ask what stands for it -- a deletion can take a whole run of messages with it, and
        # what was above may itself have gone behind a gap. Only a message can be asked that, `refs` being
        # keyed by *graph* name while `representative_of` takes a chat node; for a message the two are the
        # same string, and for a gap that did not survive there is nothing to look up anyway.
        was_above = previous.ref_for(above)
        if not isinstance(was_above, chatgraph.ChatNodeRef):
            return None
        return chat_graph.representative_of(was_above.node_id, datastore=self.datastore)

    def _cursor_chat_node_id(self) -> Optional[str]:
        """Return the chat node the cursor is on — `None` if it is on a gap box, or nowhere.

        What separates the two is whether there is anywhere to commit to. A gap stands for messages
        without being one, so acting on it opens it, where acting on a message switches branch.
        """
        with self._lock:
            name = self._cursor_name
            chat_graph = self._chat_graph
        if name is None or chat_graph is None:
            return None
        ref = chat_graph.ref_for(name)
        return ref.node_id if isinstance(ref, chatgraph.ChatNodeRef) else None
