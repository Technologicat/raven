"""Unit tests for raven.librarian.chatsearch: which messages of a branch a search finds, and where."""

import pytest

from raven.librarian import chatsearch
from raven.librarian.chattree import Forest


CONTENT = frozenset({"content"})
THINKING = frozenset({"thinking"})
BOTH = frozenset({"content", "thinking"})


def payload(role, text, reasoning=None):
    message = {"role": role, "content": [{"type": "text", "text": text}]}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {"message": message, "general_metadata": {"persona": None}}


@pytest.fixture
def branch():
    """A short conversation: a question, a tool result, and a reply that thought first."""
    forest = Forest()
    question = forest.create_node(payload=payload("user", "What is photocatalysis?"), parent_id=None)
    tool = forest.create_node(payload=payload("tool", "Wikipedia: Photocatalysis is the acceleration of a photoreaction."),
                              parent_id=question)
    reply = forest.create_node(payload=payload("assistant", "It speeds up the reaction using light.",
                                               reasoning="The user asks about photocatalysis; summarize the tool result."),
                               parent_id=tool)
    return forest, [question, tool, reply]


def find(branch, search_string, **options):
    forest, node_ids = branch
    return chatsearch.find_matches(forest, node_ids, chatsearch.make_query(search_string, **options))


class TestMakeQuery:
    @pytest.mark.parametrize("search_string", ["", "   "])
    def test_an_empty_search_is_no_query(self, search_string):
        assert chatsearch.make_query(search_string) is None

    def test_the_highlighters_follow_the_case_rule(self):
        query = chatsearch.make_query("Light photo")
        maybe_case_sensitive, maybe_case_insensitive = query.highlight
        assert maybe_case_sensitive.pattern and maybe_case_insensitive.pattern


class TestFindMatches:
    def test_no_query_matches_nothing(self, branch):
        forest, node_ids = branch
        assert chatsearch.find_matches(forest, node_ids, None) == []

    def test_every_matching_message_is_found_in_branch_order(self, branch):
        _, (question, tool, reply) = branch
        assert find(branch, "photocatalysis") == [(question, CONTENT), (tool, CONTENT), (reply, THINKING)]

    def test_all_fragments_must_occur_in_one_text(self, branch):
        _, (question, tool, reply) = branch
        assert find(branch, "light reaction") == [(reply, CONTENT)]
        assert find(branch, "light photo") == [], "the fragments are split between the content and the trace"

    def test_a_fragment_with_a_capital_matches_case_sensitively(self, branch):
        _, (question, tool, reply) = branch
        assert find(branch, "Photo") == [(tool, CONTENT)]

    def test_a_message_matching_in_both_texts_says_so(self, branch):
        """Distinct from either alone: a caller opens a collapsed trace only when nothing visible matched."""
        _, (question, tool, reply) = branch
        assert (reply, BOTH) in find(branch, "the")
        assert (reply, THINKING) in find(branch, "summarize"), "the control: a trace-only match is not reported as both"

    def test_thinking_traces_can_be_left_out(self, branch):
        _, (question, tool, reply) = branch
        assert (reply, THINKING) in find(branch, "summarize"), "the control: the trace is searched by default"
        assert find(branch, "summarize", include_thinking=False) == []
        assert (reply, CONTENT) in find(branch, "the", include_thinking=False), "left out, a trace does not add to where"

    def test_tool_messages_can_be_left_out(self, branch):
        _, (question, tool, reply) = branch
        assert (tool, CONTENT) in find(branch, "wikipedia"), "the control: tool messages are searched by default"
        assert find(branch, "wikipedia", include_tools=False) == []

    def test_the_text_is_normalized_as_the_query_is(self):
        forest = Forest()
        node = forest.create_node(payload=payload("assistant", "Oxygen is O₂."), parent_id=None)
        assert chatsearch.find_matches(forest, [node], chatsearch.make_query("o2")) == [(node, CONTENT)]
