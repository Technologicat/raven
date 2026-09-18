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


def find_counted(branch, search_string, **options):
    """Search `branch` with thinking traces included unless told otherwise, since most of these tests are about them."""
    options.setdefault("include_thinking", True)
    forest, node_ids = branch
    return chatsearch.find_matches(forest, node_ids, chatsearch.make_query(search_string, **options))


def where_of(counts):
    """The texts a `MatchCounts` reports a match in, as a set: a count is nonzero exactly where the query matched."""
    return frozenset(name for name in ("content", "thinking") if getattr(counts, name))


def find(branch, search_string, **options):
    """`find_counted`, reduced to *where* each message matched — which is what most of these tests are about."""
    return [(node_id, where_of(counts)) for node_id, counts in find_counted(branch, search_string, **options)]


class TestMakeQuery:
    @pytest.mark.parametrize("search_string", ["", "   "])
    def test_an_empty_search_is_no_query(self, search_string):
        assert chatsearch.make_query(search_string) is None

    def test_the_highlighters_follow_the_case_rule(self):
        query = chatsearch.make_query("Light photo")  # one fragment of each case kind
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
        """Distinct from either alone, so a caller can tell a trace match from a visible one whichever else matched."""
        _, (question, tool, reply) = branch
        assert (reply, BOTH) in find(branch, "the")
        assert (reply, THINKING) in find(branch, "summarize"), "the control: a trace-only match is not reported as both"

    def test_thinking_traces_can_be_left_out(self, branch):
        _, (question, tool, reply) = branch
        assert (reply, THINKING) in find(branch, "summarize"), "the control: the trace is searched when asked"
        assert find(branch, "summarize", include_thinking=False) == []
        assert (reply, CONTENT) in find(branch, "the", include_thinking=False), "left out, a trace does not add to where"

    def test_by_default_traces_are_left_out_and_tool_messages_searched(self, branch):
        forest, node_ids = branch
        _, tool, _ = node_ids
        query = chatsearch.make_query("summarize")
        assert (query.include_thinking, query.include_tools) == (False, True)
        assert chatsearch.find_matches(forest, node_ids, query) == [], "the trace-only match is left out"
        found = chatsearch.find_matches(forest, node_ids, chatsearch.make_query("wikipedia"))
        assert [(node_id, where_of(counts)) for node_id, counts in found] == [(tool, CONTENT)]

    def test_tool_messages_can_be_left_out(self, branch):
        _, (question, tool, reply) = branch
        assert (tool, CONTENT) in find(branch, "wikipedia"), "the control: tool messages are searched when asked"
        assert find(branch, "wikipedia", include_tools=False) == []

    def test_the_text_is_normalized_as_the_query_is(self):
        """Also the one place asserting the raw shape, so a change to it cannot slip past the reducing helper above."""
        forest = Forest()
        node = forest.create_node(payload=payload("assistant", "Oxygen is O₂."), parent_id=None)  # content only, so defaults suffice
        assert chatsearch.find_matches(forest, [node], chatsearch.make_query("o2")) == \
            [(node, chatsearch.MatchCounts(content=1, thinking=0))]


class TestMatchCounts:
    """How many hits a match holds, and which of a message's texts they are in.

    A chat graph box reports a count rather than a yes-or-no, so the number has to mean something exact:
    occurrences of the query's fragments, in the texts the query matched.
    """

    def test_a_fragment_occurring_twice_is_counted_twice(self, branch):
        _, (_question, tool, _reply) = branch
        counted = dict(find_counted(branch, "photo"))
        assert counted[tool].content == 2, "'Photocatalysis' and 'photoreaction' are two occurrences of the one fragment"

    def test_the_texts_are_counted_apart_and_the_total_is_their_sum(self, branch):
        _, (_question, _tool, reply) = branch
        counts = dict(find_counted(branch, "the"))[reply]
        assert (counts.content, counts.thinking) == (1, 2)
        assert counts.total == 3

    def test_a_text_that_did_not_match_contributes_nothing(self, branch):
        """A lone fragment sitting in a non-matching text is not a hit, though the renderer would colour it."""
        _, (_question, _tool, reply) = branch
        counts = dict(find_counted(branch, "the reaction"))[reply]
        assert dict(find_counted(branch, "the"))[reply].thinking == 2, \
            "the control: that trace holds two occurrences of 'the', so a count over every text would find them"
        assert counts.thinking == 0, "the trace lacks the other fragment, so it did not match and holds no hits"
        assert counts.content == 2

    def test_a_trace_left_out_of_the_search_counts_nothing(self, branch):
        _, (_question, _tool, reply) = branch
        assert dict(find_counted(branch, "the", include_thinking=False))[reply].thinking == 0

    @pytest.mark.parametrize("search_string", ["photocatalysis", "the", "photo", "the reaction"])
    def test_a_matching_message_always_counts_at_least_one(self, branch, search_string):
        """A match means every fragment occurs somewhere in one text, so a count of zero would be the two disagreeing."""
        found = find_counted(branch, search_string)
        assert found, f"{search_string!r} matches nothing here, so this parameter checks nothing"
        for _node_id, counts in found:
            assert counts.total >= 1
