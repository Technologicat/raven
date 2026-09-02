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
        self._previewed_node_id: Optional[str] = None
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
        self._widget = XDotWidget(parent=self._container,
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
                       "Zoom to fit", f"chat_graph_fit_button_{self.gui_uuid}",  # tag
                       solid=False)
            add_button(fa.ICON_MAGNIFYING_GLASS, self.zoom_1_to_1,
                       "Actual size (1:1)", f"chat_graph_actual_size_button_{self.gui_uuid}")  # tag
            add_button(fa.ICON_MAGNIFYING_GLASS_PLUS, lambda: self._widget.zoom_in(),
                       "Zoom in", f"chat_graph_zoom_in_button_{self.gui_uuid}")  # tag
            add_button(fa.ICON_MAGNIFYING_GLASS_MINUS, lambda: self._widget.zoom_out(),
                       "Zoom out", f"chat_graph_zoom_out_button_{self.gui_uuid}")  # tag
            add_button(fa.ICON_SUN if dark_mode else fa.ICON_MOON, self.toggle_dark_mode,
                       "Switch to light mode" if dark_mode else "Switch to dark mode",
                       self._dark_mode_button_tag)

            guiutils.add_toolbar_separator(horizontal=True, toolbar_extent=_TOOLBAR_H,
                                           size=gui_config.toolbar_separator_w, line=False)

            # Back and forward through the views this panel has shown. Disabled until there is somewhere
            # to go, which is the convention Visualizer set: a button is enabled exactly when pressing it
            # would do something.
            add_button(fa.ICON_ARROW_LEFT, self.go_back,
                       "Back to the previous view", self._back_button_tag, enabled=False)
            add_button(fa.ICON_ARROW_RIGHT, self.go_forward,
                       "Forward again", self._forward_button_tag, enabled=False)

            # After the separator because a *branch* is a chat idea, where the plain fit beside the zoom
            # controls is the widget's own. The two answer different questions: fitting the picture shows
            # how wide the tree is, and fitting the branch shows the conversation.
            add_button(fa.ICON_ARROWS_UP_DOWN, self.fit_branch,
                       "Fit the current branch", f"chat_graph_fit_branch_button_{self.gui_uuid}")  # tag
            add_button(fa.ICON_LOCATION_CROSSHAIRS, self.go_to_head,
                       "Back to where you are", f"chat_graph_home_button_{self.gui_uuid}")  # tag
            # The discoverable half of the commit gesture. Its caption names the fluent half, which is
            # otherwise unfindable: a second click on a box already clicked once.
            add_button(fa.ICON_CODE_BRANCH, self._commit_previewed,
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
        self._set_preview(None)
        self.refresh()
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
        """Hide the panel."""
        self._is_shown = False
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
                self._previewed_node_id = None
                chat_graph = self._try_build()
            if chat_graph is None:
                logger.warning("DPGChatGraphPanel.refresh: HEAD is gone too; leaving the picture as it is")
                return

            self._chat_graph = chat_graph
            self._seen_generation = generation
            self._seen_head = self._view_state.head_node_id
            self._widget.set_graph(chat_graph.graph)
            anchor = (self._previewed_node_id
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
        self._set_preview(None)
        self.refresh()

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
        ref = chat_graph.ref_for(element.internal_name)
        if ref is None:
            logger.warning("DPGChatGraphPanel._on_click: no ref for graph node "
                           f"'{element.internal_name}'; ignoring")
            return

        if isinstance(ref, chatgraph.ChatNodeRef):
            self._click_chat_node(ref)
        elif isinstance(ref, chatgraph.SiblingGapRef):
            self._move_sibling_window(ref)
        elif isinstance(ref, chatgraph.DepthGapRef):
            self._widen_depth_window()
        elif isinstance(ref, chatgraph.SubtreeGapRef):
            self._look_inside(ref)
        elif isinstance(ref, chatgraph.RootGapRef):
            # Inert in v1, deliberately: switching to a chat written under an older character card would
            # leave the configured avatar and voice running against a different system prompt.
            logger.info(f"DPGChatGraphPanel._on_click: {ref.hidden_count} chat(s) under other cards; "
                        "reaching them is not implemented")

        # Once, here, rather than in each of the four branches above. A commit that changes nothing is a
        # no-op, so the ones that only scrolled the chat log cost nothing and need no special case.
        self._remember_view()

    def _click_chat_node(self, ref: chatgraph.ChatNodeRef) -> None:
        """Preview a chat node — or commit to it, if it was already the previewed one."""
        with self._lock:
            already_previewed = (ref.node_id == self._previewed_node_id)
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
        self._set_preview(ref.node_id)  # redraws, so the branch change above lands with it
        self._widget.pan_to_node(ref.node_id, animate=True)

        if ref.on_current_branch and self._on_preview is not None:
            self._on_preview(ref.node_id)

    def _move_sibling_window(self, ref: chatgraph.SiblingGapRef) -> None:
        """Re-centre one level's window on the middle of the run this gap hides."""
        with self._lock:
            self._view_state.sibling_focus[ref.parent_node_id] = ref.recenter_on
        self._set_preview(None)  # navigation, not a choice of node, so nothing is left armed
        self.refresh()
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
        self._set_preview(ref.node_id)

    # ------------------------------------------------------------------
    # Committing

    def _commit_previewed(self) -> None:
        """Toolbar button: move HEAD to the previewed node, if there is one."""
        with self._lock:
            node_id = self._previewed_node_id
        if node_id is not None:
            self._commit(node_id)

    def _commit(self, node_id: str) -> None:
        """Move HEAD to `node_id`, through the caller."""
        with self._lock:
            self._view_state.focus_node_id = None
        self._set_preview(None)
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

    def _set_preview(self, node_id: Optional[str]) -> None:
        """Select a box, or clear the selection, and redraw so the ring moves with it.

        The mark lives in the picture rather than in the widget's highlight state. That state is shared
        with hover and has one pair of colours, so a preview drawn through it is indistinguishable from a
        hover — and worse, a node lit and left lit reads as *the important one*, which is HEAD's job.
        Keeping it in the `Graph` also means it survives a rebuild without being re-applied.
        """
        with self._lock:
            if self._previewed_node_id == node_id:
                return
            self._previewed_node_id = node_id
            self._view_state.previewed_node_id = node_id
        self._enable_commit(node_id is not None)
        self.refresh()
