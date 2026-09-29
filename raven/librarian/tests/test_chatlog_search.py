"""Unit tests for raven.librarian.chatlog_search."""

import threading
import types

import pytest

pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from raven.librarian import chatlog_search, chatsearch, chattree  # noqa: E402 -- after importorskip by design


def make_search(view, datastore=None):
    """A real `DPGChatLogSearch` over a stand-in view, with the GUI reported gone.

    Gone, so that `set_search` records the query and submits no re-highlight: there are no messages on
    screen to re-highlight, and no task manager to run it on.
    """
    return chatlog_search.DPGChatLogSearch(datastore=datastore,
                                           view=view,
                                           history=[],
                                           history_lock=threading.RLock(),
                                           task_manager=None,
                                           gui_updates_safe=lambda: False)


class TestSteppingTheSearch:
    """What `step_search` reports back, which is what lets a caller follow a jump that happened.

    The app sends the keyboard to the chat log after one, so that the arrows act on the pane the key just
    took the reader to — and must not when the key did nothing, which is the whole reason this answers at
    all rather than returning `None` as it used to.

    Where the view stands is patched rather than rendered: `_find_match` reads widget positions, which a
    view needs rendered frames to have.
    """

    @staticmethod
    def _search(matches, jumped_to=None):
        search = make_search(types.SimpleNamespace(find_message=lambda node_id: None,
                                                   jump_to_node=lambda node_id: jumped_to))
        search._matches = matches  # as `add_matches_for` would leave them, without a view to build
        search._position_stale = False
        return search

    def test_it_says_no_when_there_is_nowhere_to_go(self, monkeypatch):
        search = self._search(matches=[])
        monkeypatch.setattr(type(search), "_find_match",
                            lambda self, forward, beyond_a_line: None)
        assert search.step_search(+1) is False

    def test_it_says_yes_when_it_jumped(self, monkeypatch):
        counts = chatsearch.MatchCounts(content=1, thinking=0)
        search = self._search(matches=[("n1", counts)], jumped_to=120)
        monkeypatch.setattr(type(search), "_find_match",
                            lambda self, forward, beyond_a_line: 0)
        assert search.step_search(+1) is True
        assert search._position_stale, \
            "nothing was recorded, so this fixture did not reach the body it claims to have run"


class TestOpeningAnAwaitedThinkingTrace:
    """A jump that moves HEAD cannot open the trace it lands on, so it asks for the trace and the rebuild obliges.

    The chat graph's commits are where this comes from: a box wearing a count that was a thinking-trace hit
    has no trace of its own to open, and acting on it moves HEAD — at which moment the message the count was
    about does not exist yet, the view rebuilding on another thread. `open_thinking_trace_when_it_matches` names
    the node, and `add_matches_for` — which the rebuild already calls once per message — collects.
    """

    @staticmethod
    def _payload(reasoning):
        return {"message": {"role": "assistant",
                            "content": [{"type": "text", "text": "It speeds up the reaction using light."}],
                            "reasoning_content": reasoning},
                "general_metadata": {"persona": None}}

    @classmethod
    def _branch(cls, reasoning="Summarize the tool result for the reader."):
        """One reply whose reasoning says something its answer does not, which is the whole case here."""
        forest = chattree.Forest()
        return forest, forest.create_node(payload=cls._payload(reasoning), parent_id=None)

    @staticmethod
    def _search(forest, search_string, opened):
        search = make_search(types.SimpleNamespace(
                                 find_message=lambda node_id: types.SimpleNamespace(
                                     show_thinking_trace=lambda: opened.append(node_id))),
                             datastore=forest)
        search.set_search(chatsearch.make_query(search_string, include_thinking=True))
        return search

    def test_the_awaited_message_opens_its_trace_when_it_arrives(self):
        forest, node_id = self._branch()
        opened = []
        search = self._search(forest, "summarize", opened)
        search.open_thinking_trace_when_it_matches(node_id)
        search.add_matches_for(node_id)
        assert opened == [node_id]

    def test_a_message_that_matched_only_in_its_text_keeps_its_trace_closed(self):
        forest, node_id = self._branch()
        opened = []
        search = self._search(forest, "light", opened)
        search.open_thinking_trace_when_it_matches(node_id)
        search.add_matches_for(node_id)
        assert search.search_matches, \
            "nothing matched at all, so this fixture cannot tell a trace left closed from a message not found"
        assert opened == []

    def test_a_message_nobody_awaited_keeps_its_trace_closed(self):
        forest, node_id = self._branch()
        opened = []
        search = self._search(forest, "summarize", opened)
        search.add_matches_for(node_id)  # no request made
        assert search.search_matches[0][1].thinking, \
            "the trace did not match, so this fixture cannot tell a request being honoured from one never made"
        assert opened == []

    def test_the_request_is_spent_on_the_message_it_named(self):
        forest, node_id = self._branch()
        opened = []
        search = self._search(forest, "summarize", opened)
        search.open_thinking_trace_when_it_matches(node_id)
        search.add_matches_for(node_id)
        search.add_matches_for(node_id)  # a later rebuild walking past the same message
        assert opened == [node_id]

    def test_clearing_the_search_before_the_message_arrives_still_ends_the_wait(self):
        forest, node_id = self._branch()
        opened = []
        search = self._search(forest, "summarize", opened)
        search.open_thinking_trace_when_it_matches(node_id)
        search.set_search(None)  # the reader cleared it in the frames since the jump
        search.add_matches_for(node_id)
        assert opened == []
        assert search._node_awaiting_trace_open is None, \
            "the request outlived the message it named, and would be spent on some later rebuild"

    def test_a_reply_still_writing_its_trace_keeps_the_jump_waiting(self):
        forest, node_id = self._branch(reasoning="Let me think about this for a moment.")
        opened = []
        search = self._search(forest, "summarize", opened)
        search.open_thinking_trace_when_it_matches(node_id)
        search.recheck_awaited_thinking_trace(node_id)
        assert opened == []
        assert search._node_awaiting_trace_open == node_id, \
            "the wait ended on a trace that had not yet written the words it was waiting for"

    def test_the_trace_opens_as_soon_as_the_words_arrive(self):
        forest, node_id = self._branch(reasoning="Let me think about this for a moment.")
        opened = []
        search = self._search(forest, "summarize", opened)
        search.open_thinking_trace_when_it_matches(node_id)
        search.recheck_awaited_thinking_trace(node_id)
        assert opened == [], \
            "the trace matched before the words arrived, so this fixture cannot show them arriving"
        forest.add_revision(node_id, self._payload("Summarize the tool result for the reader."))
        search.recheck_awaited_thinking_trace(node_id)
        assert opened == [node_id]
        assert search._node_awaiting_trace_open is None, \
            "the trace opened but the request was not spent, so a later message could collect it too"
