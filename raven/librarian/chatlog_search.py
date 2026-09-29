"""The chat log's search: which messages of the branch on screen match, and where the reader is among them.

`DPGChatLogSearch` is the chat log's side of the search row. The chat graph has the other side in
`chatgraph_panel.DPGChatGraphPanel`, which searches the whole forest; the two expose the same interface
(`set_search`, `step_search`, `search_matches`, `search_position`, `search_can_go_back`/`_forward`), so the
app drives both alike. What a match *is* lives in `chatsearch`, which both use.
"""

__all__ = ["DPGChatLogSearch"]

import logging
logger = logging.getLogger(__name__)

import threading
from typing import Callable, TYPE_CHECKING

import dearpygui.dearpygui as dpg

from unpythonic.env import env

from ..common import bgtask
from ..common.gui import animation as gui_animation
from ..common.gui import utils as guiutils
from ..common.gui import widgetfinder

from . import chatsearch
from . import chattree
from . import config as librarian_config

if TYPE_CHECKING:  # the view's module imports this one
    from .chat_controller import DPGLinearizedChatView

gui_config = librarian_config.gui_config


class DPGChatLogSearch:
    """The running search over the chat log, and the reader's position among its matches."""

    def __init__(self, *,
                 datastore: chattree.Forest,
                 view: "DPGLinearizedChatView",
                 history: list,
                 history_lock: threading.RLock,
                 task_manager: bgtask.TaskManager,
                 gui_updates_safe: Callable[[], bool],
                 on_search_results_changed: Callable[[], None] | None = None):
        """`datastore`: The datastore the chat log shows a branch of.

        `view`: The `chat_controller.DPGLinearizedChatView` being searched, for its panel and its messages.

        `history`, `history_lock`: The messages on screen, in branch order, and the lock that guards the list.
                                   The same list object throughout: the view refills it in place.

        `task_manager`: Where re-highlighting runs. Sequential, so a new search cancels the previous one's.

        `gui_updates_safe`: Answers whether the GUI may still be touched, which stops being true at shutdown.

        `on_search_results_changed`: Called with no arguments when the match count or the current match's
                                     position changes, so the app can update its search row.
        """
        self.datastore = datastore
        self.view = view
        self._history = history
        self._history_lock = history_lock
        self.task_manager = task_manager
        self._gui_updates_safe = gui_updates_safe
        self.on_search_results_changed = on_search_results_changed

        # `_query` is what paragraphs are highlighted with as they render, so it is rebound whole and read
        # without a lock: a render sees either the old query or the new one, and the re-highlight that follows
        # a change catches up whatever rendered in between. `_matches` is `chatsearch.find_matches`' answer for
        # the branch on screen, likewise rebound whole. The rest is where the view is among them, kept by
        # `update_position`: the current match's index, if one is on screen, and whether previous and next
        # have anywhere to go.
        self._query = None
        self._matches = []
        self._match_index = None
        self._can_go_back = False
        self._can_go_forward = False
        self._position_y_scroll = None
        self._position_stale = True
        self._jump = None  # `(index, target_y_scroll)`; see `_jump_holds`
        # The one node whose thinking trace is owed to the reader once its message exists; see
        # `open_thinking_trace_when_it_matches`. `None` whenever nothing is awaited, which is nearly always.
        self._node_awaiting_trace_open = None

    # ------------------------------------------------------------------
    # The interface shared with the chat graph's search

    def set_search(self, maybe_query: chatsearch.SearchQuery | None) -> None:
        """Search the branch on screen with `maybe_query`, and re-highlight the chat log to match. `None` ends the search.

        Callable from any thread. Returns at once. Finding the matches and re-highlighting run in the background,
        messages nearest the view first, and a newer search cancels both — so a caller on DPG's callback thread,
        a keystroke's, is not held up by a long branch.
        """
        self._query = maybe_query
        if self._gui_updates_safe():
            self.task_manager.submit(self._search_task, env())

    def _get_query(self) -> chatsearch.SearchQuery | None:
        """Return the running search, or `None` for none."""
        return self._query
    query = property(fget=_get_query,
                     doc="The running search, or `None` for none. What a message highlights with as it renders.")

    def _get_search_matches(self) -> list[tuple[str, chatsearch.MatchCounts]]:
        """Return the running search's matches over the branch on screen, in branch order."""
        return self._matches
    search_matches = property(fget=_get_search_matches,
                              doc="The running search's matches over the branch on screen, in branch order. Empty for no search.")

    def search_position(self) -> tuple[int | None, int]:
        """Return `(which match the reader is at or None, how many there are)`, for a counter to draw.

        As of the last `update_position`, which the app calls once a frame.
        """
        return self._match_index, len(self._matches)

    def _get_search_can_go_back(self) -> bool:
        """Return whether a step back would go anywhere."""
        return self._can_go_back

    def _get_search_can_go_forward(self) -> bool:
        """Return whether a step forward would go anywhere."""
        return self._can_go_forward

    search_can_go_back = property(fget=_get_search_can_go_back,
                                  doc="Whether a step back would go anywhere, as of the last `update_position`.")
    search_can_go_forward = property(fget=_get_search_can_go_forward,
                                     doc="Whether a step forward would go anywhere, as of the last `update_position`.")

    # Where the reader is, in the matches, is read off the view rather than remembered: the current match is the
    # topmost one at or below the top of the view, and next and previous are the nearest matches more than a line
    # below and above it. So scrolling by hand moves the counter, and a jump goes from what is on screen rather
    # than from wherever the last jump went. The Visualizer's info panel works the same way, and the two should
    # stay alike.

    def step_search(self, direction: int) -> bool:
        """Jump to the next (`direction=+1`) or previous (`-1`) matching message, relative to the top of the view.

        Stops at either end rather than wrapping around. A message whose thinking trace matched has its trace
        opened, whether or not its text matched too, so that every match the search counted is on screen.

        Returns whether it went anywhere, so a caller can follow the jump with something — sending the
        keyboard after it — and do nothing at all where there was nothing to jump to.
        """
        maybe_jump = self._jump  # one read: the render thread may clear it meanwhile
        if self._jump_holds(maybe_jump):
            maybe_index = maybe_jump[0] + (1 if direction > 0 else -1)
            if not 0 <= maybe_index < len(self._matches):
                return False
        else:
            maybe_index = self._find_match(forward=(direction > 0), beyond_a_line=True)
            if maybe_index is None:
                return False
        node_id, counts = self._matches[maybe_index]
        if counts.thinking and (message := self.view.find_message(node_id)) is not None:
            message.show_thinking_trace()
        maybe_y_scroll = self.view.jump_to_node(node_id)
        # Recorded once the scroll has started, so that `_jump_holds` finds it gliding rather than finding the
        # view not yet where it is going.
        self._jump = (maybe_index, maybe_y_scroll) if maybe_y_scroll is not None else None
        self._position_stale = True
        return True

    # ------------------------------------------------------------------
    # Keeping the matches in step with the view

    def clear_matches(self) -> None:
        """Forget the matches, for a view about to be rebuilt; `add_matches_for` refills them as it is."""
        self._matches = []
        self._matches_changed()

    def add_matches_for(self, node_id: str) -> None:
        """Test one message just built into the view against the search, and count it if it matches.

        Called once per message by whatever built it, which is a whole branch's worth during a rebuild and a
        single message when a turn writes one. Per message rather than per build, so that a view assembled
        message by message tests each exactly once.

        A reply still streaming is not tested, its text being still in motion; it is counted when it
        finalizes and is rebuilt as a stored message.

        Also where a jump that had to wait for this message gets to open its thinking trace; see
        `open_thinking_trace_when_it_matches`.
        """
        # `find_matches` answers `[]` for no search, so the no-search case needs no branch of its own here —
        # and must not take an early return, because an awaited trace is still awaited when the reader has
        # cleared the search in the frames since they jumped.
        new_matches = chatsearch.find_matches(self.datastore, [node_id], self._query)
        if new_matches:
            self._matches = self._matches + new_matches  # rebound whole, never mutated: readers take no lock
            self._matches_changed()
        self._open_awaited_thinking_trace(node_id, new_matches[0][1] if new_matches else None)

    def refresh_matches(self) -> None:
        """Recompute which messages of the branch on screen match the current search, all of them."""
        maybe_query = self._query
        with self._history_lock:
            node_ids = [message.node_id for message in self._history if message.node_id is not None]
        self._matches = chatsearch.find_matches(self.datastore, node_ids, maybe_query)
        self._matches_changed()

    def _matches_changed(self) -> None:
        self._jump = None  # an index into the old matches
        self._position_stale = True
        if self.on_search_results_changed is not None:  # the count, at once; the position follows on the next frame
            self.on_search_results_changed()

    # ------------------------------------------------------------------
    # A thinking trace a jump has to wait for

    def open_thinking_trace_when_it_matches(self, node_id: str) -> None:
        """Ask that `node_id`'s thinking trace be opened as soon as that trace actually matches the search.

        For a caller that cannot open it itself, because the message is not there to open. Moving HEAD
        rebuilds the view on another thread, so at the moment of the request `view.find_message` answers
        `None`; and a reply still being generated may not yet have written the words that make its trace
        match. Both are answered by asking later rather than now — `add_matches_for` when the message
        arrives, `recheck_awaited_thinking_trace` while it is still being written.

        The chat graph's commit gesture is the caller today, having a box whose count says the trace matched
        and no trace of its own to open. Nothing here is particular to it.

        Only one node is remembered. A second request arriving before the first is answered is the reader
        changing their mind, and the trace they no longer want opened is the one they left.
        """
        self._node_awaiting_trace_open = node_id

    def recheck_awaited_thinking_trace(self, node_id: str) -> None:
        """A message still being written has new words; open its trace if that is what was asked for.

        The rule is the one `add_matches_for` applies to a message that has arrived — open the trace when
        the trace is what matched — asked repeatedly rather than once, because here the text is still
        moving and the answer can change from no to yes.

        **It does not end the wait when the answer is still no**, and that is the whole difference from the
        arrival case: a stored message's answer is final, so the request is spent on it either way, while a
        reply in progress may yet write the words being waited for. The wait ends when the reply finalizes
        and is rebuilt as a stored message, which goes through `add_matches_for`.
        """
        if node_id != self._node_awaiting_trace_open:
            return
        matches = chatsearch.find_matches(self.datastore, [node_id], self._query)
        if matches and matches[0][1].thinking:
            self._open_awaited_thinking_trace(node_id, matches[0][1])

    def _open_awaited_thinking_trace(self, node_id: str, maybe_counts: chatsearch.MatchCounts | None) -> None:
        """Open the trace of `node_id`, if it is the message a jump was waiting for and its trace is what matched."""
        if node_id != self._node_awaiting_trace_open:
            return
        # Arrived, so the wait is over whether or not it ends in an open one: leaving it set would spend the
        # request on whichever later rebuild happened to pass this node next.
        self._node_awaiting_trace_open = None
        # The rule `step_search` follows: a trace that matched is opened whether or not the message text
        # matched too, so that every match the search counted is on screen.
        if maybe_counts is None or not maybe_counts.thinking:
            return
        if (message := self.view.find_message(node_id)) is not None:
            message.show_thinking_trace()

    # ------------------------------------------------------------------
    # Where the reader is

    # Position alone cannot say where the reader is after a jump near the end of the chat: the last few messages
    # cannot be scrolled up to the top of the view, there being nothing below them to scroll into, so the topmost
    # match on screen is still an earlier one. So the match a jump went to is current for as long as the view is
    # where the jump sent it — gliding there, or resting there — and the position rules take over again the moment
    # the reader scrolls anywhere else, by wheel or by key.

    def _jump_holds(self, maybe_jump: tuple[int, int] | None) -> bool:
        """Whether `maybe_jump`, a value of `_jump`, still says where the reader is. Lock-free, for the render thread."""
        if maybe_jump is None:
            return False
        index, target_y_scroll = maybe_jump
        if index >= len(self._matches):
            return False
        maybe_animation = gui_animation.SmoothScrolling.instances.get(self.view.gui_parent)
        if maybe_animation is not None and maybe_animation.target_y_scroll == target_y_scroll:
            return True
        with guiutils.nonexistent_ok() as nok:
            y_scroll = dpg.get_y_scroll(self.view.gui_parent)
        return not nok.errored and abs(y_scroll - target_y_scroll) <= 1

    def update_position(self) -> None:
        """Recompute which match is current and whether next and previous have anywhere to go. Call once per frame.

        Does nothing unless the view has scrolled or the matches have changed since the last call, and tells
        `on_search_results_changed` only when the answer differs.
        """
        with guiutils.nonexistent_ok() as nok:
            y_scroll = dpg.get_y_scroll(self.view.gui_parent)
        if nok.errored or (y_scroll == self._position_y_scroll and not self._position_stale):
            return
        self._position_y_scroll = y_scroll
        self._position_stale = False
        maybe_jump = self._jump  # one read: a navigation handler may replace it meanwhile
        if self._jump_holds(maybe_jump):
            jumped = maybe_jump[0]
            self._set_position(jumped, jumped > 0, jumped < len(self._matches) - 1)
            return
        self._jump = None
        maybe_index = self._find_match(forward=True, beyond_a_line=False)
        if maybe_index is not None:
            with guiutils.nonexistent_ok() as nok:
                view_bottom = guiutils.get_widget_pos(self.view.gui_parent)[1] + guiutils.get_widget_size(self.view.gui_parent)[1]
                indices, containers = self._match_containers()  # not `view.find_message`, which takes the lock
                if maybe_index not in indices or guiutils.get_widget_pos(containers[indices.index(maybe_index)])[1] >= view_bottom:
                    maybe_index = None  # the topmost match below the top is not on screen, so none is current
            if nok.errored:
                self._position_stale = True  # the view changed under the read; try again next frame
                return
        self._set_position(maybe_index,
                           self._find_match(forward=False, beyond_a_line=True) is not None,
                           self._find_match(forward=True, beyond_a_line=True) is not None)

    def _set_position(self, maybe_index: int | None, can_go_back: bool, can_go_forward: bool) -> None:
        position = (maybe_index, can_go_back, can_go_forward)
        if position != (self._match_index, self._can_go_back, self._can_go_forward):
            self._match_index, self._can_go_back, self._can_go_forward = position
            if self.on_search_results_changed is not None:
                self.on_search_results_changed()

    def _match_containers(self) -> tuple[list[int], list]:
        """`(indices, containers)`: each matching message on screen, as its index into the matches and its container widget.

        In branch order. A match with no widget — the view mid-rebuild — is left out of both lists together.

        Read without the history lock, since this runs on the render thread every frame the view scrolls, and the
        view's `build` holds that lock from another thread across frames. `tuple` copies the list in one step; a
        widget that has gone by the time it is read raises, and callers treat that as "try again next frame".
        """
        containers_by_node_id = {message.node_id: message.gui_container_group
                                 for message in tuple(self._history)}
        indices, containers = [], []
        for index, (node_id, _counts) in enumerate(self._matches):
            if node_id in containers_by_node_id:
                indices.append(index)
                containers.append(containers_by_node_id[node_id])
        return indices, containers

    def _find_match(self, *, forward: bool, beyond_a_line: bool) -> int | None:
        """Index into the matches of the nearest match below (`forward`) or above the top of the view, or `None`.

        `beyond_a_line`: whether the match must be more than a line of text past the top — true for next and
                         previous, so that the match already at the top is neither. Without it, forward finds the
                         topmost match at or below the top, which is the current one.
        """
        indices, containers = self._match_containers()
        if not containers:
            return None
        with guiutils.nonexistent_ok():
            view_top = guiutils.get_widget_pos(self.view.gui_parent)[1]
            line = gui_config.font_size if beyond_a_line else 0
            target_y = view_top + (line if forward else -line)

            def is_completely_below(widget):
                return widgetfinder.is_completely_below_target_y(widget, target_y=target_y)
            maybe_widget = widgetfinder.binary_search_widget(widgets=containers, accept=is_completely_below,
                                                             consider=None, direction=("right" if forward else "left"))
            return indices[containers.index(maybe_widget)] if maybe_widget is not None else None
        return None

    # ------------------------------------------------------------------
    # Re-highlighting

    def _search_task(self, task_env: env) -> None:
        self.refresh_matches()
        if task_env.cancelled:
            return
        with self._history_lock:
            messages = list(self._history)
        for message in self._nearest_the_view_first(messages):
            if task_env.cancelled or not self._gui_updates_safe():
                return
            message.rehighlight(task_env)

    def _nearest_the_view_first(self, messages: list) -> list:
        """`messages`, ordered by how far each is from the view: those on screen first, then outwards.

        Left in their order if the view cannot be measured, which only costs the order.
        """
        with guiutils.nonexistent_ok():
            view_top = guiutils.get_widget_pos(self.view.gui_parent)[1]
            view_bottom = view_top + guiutils.get_widget_size(self.view.gui_parent)[1]

            def distance(message) -> int:
                top = guiutils.get_widget_pos(message.gui_container_group)[1]
                bottom = top + guiutils.get_widget_size(message.gui_container_group)[1]
                return max(0, view_top - bottom, top - view_bottom)
            return sorted(messages, key=distance)
        return messages
