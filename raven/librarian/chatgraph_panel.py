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

from ..common.gui import animation as gui_animation
from ..common.gui import utils as guiutils
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
        # What the last rebuild was made from. Compared once per frame; a mismatch is the whole
        # "did anything change?" test.
        self._seen_generation: Optional[int] = None
        self._seen_head: Optional[str] = None

        self._commit_button_tag = f"chat_graph_commit_button_{self.gui_uuid}"  # tag
        self._container = dpg.add_child_window(parent=gui_parent,
                                               width=width, height=height,
                                               no_scrollbar=True, no_scroll_with_mouse=True,
                                               show=self._is_shown,
                                               tag=f"chat_graph_panel_{self.gui_uuid}")  # tag
        self._build_toolbar()
        if graph_text_fonts is None:
            graph_text_fonts = [(size, guiutils.load_extra_font(themes_and_fonts, size,
                                                                "OpenSans", "Regular")[1])
                                for size in _GRAPH_TEXT_FONT_SIZES]
        self._widget = XDotWidget(parent=self._container,
                                  width=self._graph_w(width),
                                  height=self._graph_h(height),
                                  on_click=self._on_click,
                                  input_blocked=input_blocked,
                                  graph_text_fonts=graph_text_fonts,
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

    def _build_toolbar(self) -> None:
        """Build the row of view controls above the graph."""
        def add_button(icon: str, callback: Callable, caption: str, tag: str,
                       enabled: bool = True) -> None:
            dpg.add_button(label=icon, callback=callback, width=gui_config.toolbutton_w,
                           enabled=enabled, tag=tag)
            dpg.bind_item_font(tag, self.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(tag, self.themes_and_fonts.disablable_widget_theme)  # tag
            # A caption that never changes, so a plain DPG tooltip rather than the self-sizing one.
            with dpg.tooltip(tag):  # tag
                dpg.add_text(caption)

        with dpg.group(horizontal=True, parent=self._container):
            add_button(fa.ICON_EXPAND, lambda: self._widget.zoom_to_fit(),
                       "Fit the whole picture", f"chat_graph_fit_button_{self.gui_uuid}")  # tag
            add_button(fa.ICON_MAGNIFYING_GLASS_PLUS, lambda: self._widget.zoom_in(),
                       "Zoom in", f"chat_graph_zoom_in_button_{self.gui_uuid}")  # tag
            add_button(fa.ICON_MAGNIFYING_GLASS_MINUS, lambda: self._widget.zoom_out(),
                       "Zoom out", f"chat_graph_zoom_out_button_{self.gui_uuid}")  # tag

            guiutils.add_toolbar_separator(horizontal=True, toolbar_extent=_TOOLBAR_H,
                                           size=gui_config.toolbar_separator_w, line=False)

            add_button(fa.ICON_LOCATION_CROSSHAIRS, self.go_to_head,
                       "Back to where you are", f"chat_graph_home_button_{self.gui_uuid}")  # tag
            # The discoverable half of the commit gesture. Its caption names the fluent half, which is
            # otherwise unfindable: a second click on a box already clicked once.
            add_button(fa.ICON_CODE_BRANCH, self._commit_previewed,
                       "Switch to the previewed branch\n(or click its box a second time)",
                       self._commit_button_tag, enabled=False)

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
            self._apply_preview_highlight()
            anchor = (self._previewed_node_id
                      or self._view_state.focus_node_id
                      or self._view_state.head_node_id)

        # `set_graph` leaves pan and zoom alone, so an anchor that kept its place needs nothing done; one
        # that moved is followed smoothly. That is what keeps the picture from jumping under a reader while
        # a reply is arriving and the tree is gaining a node per round.
        if chat_graph.graph.get_node_by_name(anchor) is not None:
            self._widget.pan_to_node(anchor, animate=True)
        else:
            self._widget.zoom_to_fit(animate=True)

    def _try_build(self) -> Optional[chatgraph.ChatGraph]:
        """Build the picture, or return `None` if the node it would be drawn around is gone."""
        try:
            return chatgraph.build(self.datastore, self._view_state, self._layout)
        except KeyError:
            return None

    def go_to_head(self) -> None:
        """Abandon any preview and draw the picture around HEAD again."""
        with self._lock:
            self._view_state.focus_node_id = None
            self._previewed_node_id = None
        self._enable_commit(False)
        self.refresh()

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

    def _on_click(self, node_id: str, button: int) -> None:
        """Handle a click on a box. `node_id` is the graph node's name, not necessarily a chat node ID."""
        if button != dpg.mvMouseButton_Left:
            return
        with self._lock:
            chat_graph = self._chat_graph
        if chat_graph is None:
            return
        ref = chat_graph.ref_for(node_id)
        if ref is None:
            logger.warning(f"DPGChatGraphPanel._on_click: no ref for graph node '{node_id}'; ignoring")
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

    def _click_chat_node(self, ref: chatgraph.ChatNodeRef) -> None:
        """Preview a chat node — or commit to it, if it was already the previewed one."""
        with self._lock:
            already_previewed = (ref.node_id == self._previewed_node_id)
        if already_previewed:
            self._commit(ref.node_id)
            return

        with self._lock:
            self._previewed_node_id = ref.node_id
        self._enable_commit(True)

        if ref.on_current_branch:
            # There is a message for this node in the chat log, so the useful thing is to show it there.
            # The picture does not move: the reader is already looking at the branch they are on.
            self._apply_preview_highlight()
            self._widget.pan_to_node(ref.node_id, animate=True)
            if self._on_preview is not None:
                self._on_preview(ref.node_id)
        else:
            # Nothing in the chat log corresponds to this node, so the graph is the only place it can be
            # shown. Redraw around it, which also brings its siblings and its children into view -- those
            # being what the reader is now choosing between.
            with self._lock:
                self._view_state.focus_node_id = ref.node_id
            self.refresh()

    def _move_sibling_window(self, ref: chatgraph.SiblingGapRef) -> None:
        """Re-centre one level's window on the middle of the run this gap hides."""
        with self._lock:
            self._view_state.sibling_focus[ref.parent_node_id] = ref.recenter_on
            self._previewed_node_id = None
        self._enable_commit(False)
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
            self._previewed_node_id = ref.node_id
        self._enable_commit(True)
        self.refresh()

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
            self._previewed_node_id = None
            self._view_state.focus_node_id = None
        self._enable_commit(False)
        if self._on_commit is not None:
            self._on_commit(node_id)
        else:
            logger.warning("DPGChatGraphPanel._commit: no `on_commit` was given; HEAD not moved")
        # The move arrives through the change poll like any other, so there is nothing to rebuild here.

    def _enable_commit(self, enabled: bool) -> None:
        """Enable or disable the toolbar's commit button."""
        with guiutils.nonexistent_ok():
            if enabled:
                dpg.enable_item(self._commit_button_tag)
            else:
                dpg.disable_item(self._commit_button_tag)

    def _apply_preview_highlight(self) -> None:
        """Mark the previewed box, so what a second click would act on is on screen before it happens."""
        with self._lock:
            node_id = self._previewed_node_id
        self._widget.set_highlighted_nodes({node_id} if node_id is not None else set())
