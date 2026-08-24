"""Unit tests for raven.librarian.chat_controller.

Only the datastore-side helpers so far — the ones that decide which of a message's buttons are live, and the
descent that picks where a rebuilt view lands. Those are pure functions over a `Forest` and need no GUI.

The module they live in needs rather more, though: importing it reaches DearPyGui *and*, through the avatar
client, the full ML stack down to spaCy. So the skip names the module under test rather than any one of its
dependencies — the same thing `test_scaffold.py` did until `api.initialize` was made lazy, and the reason
these tests do not run in the minimal-dependency CI job even though they would pass there.
"""

import pytest

pytest.importorskip("raven.librarian.chat_controller")  # noqa: E402 -- reaches DearPyGui and the ML stack

from raven.librarian import chat_controller, chattree  # noqa: E402


def _payload(role, text, timestamp=0):
    """A chat node payload with the two fields these helpers read: the role, and the timestamp."""
    return {"message": {"role": role, "content": [{"type": "text", "text": text}], "tool_calls": []},
            "general_metadata": {"persona": None, "timestamp": timestamp}}


@pytest.fixture
def two_card_forest():
    """Two system prompts, each with its own greeting, and one message under the first.

    The shape `appstate` produces once the datastore has seen more than one system prompt: every root is a
    system prompt node, and its children are the greetings recorded under it.
    """
    f = chattree.Forest()
    card1 = f.create_node(_payload("system", "system prompt 1"), parent_id=None)
    card2 = f.create_node(_payload("system", "system prompt 2"), parent_id=None)
    greeting1 = f.create_node(_payload("assistant", "greeting under card 1", 1), parent_id=card1)
    greeting2 = f.create_node(_payload("assistant", "greeting under card 2", 1), parent_id=card2)
    message = f.create_node(_payload("user", "a user message", 2), parent_id=greeting1)
    return f, card1, card2, greeting1, greeting2, message


class TestGreetingNodeIds:
    def test_every_greeting_under_every_card_is_listed(self, two_card_forest):
        f, _card1, _card2, greeting1, greeting2, message = two_card_forest
        greeting_node_ids = chat_controller._get_all_greeting_node_ids(datastore=f)
        assert set(greeting_node_ids) == {greeting1, greeting2}
        assert message not in greeting_node_ids

    def test_the_same_question_gets_the_same_answer_every_time(self, two_card_forest):
        # The four button gates on one message -- reroll, continue, branch, delete -- each ask this list
        # whether the message is a greeting, and they ask the object they were handed once. A lazily
        # evaluated answer is consumed by the first question, so the rest are answered from the leftovers,
        # which reads as "no": a greeting's delete button comes up live and deletes the chat under it.
        f, _card1, _card2, greeting1, _greeting2, _message = two_card_forest
        greeting_node_ids = chat_controller._get_all_greeting_node_ids(datastore=f)
        assert [greeting1 in greeting_node_ids for _ in range(4)] == [True] * 4

    def test_a_deleted_card_takes_its_greetings_out_of_the_list(self, two_card_forest):
        # The root scan underneath is memoized, so this is really asking whether the memo is filtered
        # against the live nodes. It has to be: `get_children` raises on a node that is gone.
        f, _card1, card2, greeting1, greeting2, _message = two_card_forest
        f.delete_subtree(card2)
        greeting_node_ids = chat_controller._get_all_greeting_node_ids(datastore=f)
        assert greeting1 in greeting_node_ids
        assert greeting2 not in greeting_node_ids

    def test_a_message_the_user_sent_is_not_a_greeting_however_it_sits(self, two_card_forest):
        # HEAD can rest on a system prompt node — deleting another card lands there — and a message sent
        # from there attaches beside the greetings. Going by position alone would call it one, and a
        # greeting has its reroll, continue, branch and delete buttons taken away: the user would be left
        # with a message of their own that they cannot remove.
        f, card1, _card2, _greeting1, _greeting2, _message = two_card_forest
        typed_at_the_root = f.create_node(_payload("user", "sent with HEAD on the card", 3), parent_id=card1)
        assert typed_at_the_root not in chat_controller._get_all_greeting_node_ids(datastore=f)


class TestDescendToLatest:
    def test_recursive_walks_to_the_tip(self, two_card_forest):
        # What the sibling arrows and the "show chat continuation" button mean by descending.
        f, card1, _card2, _greeting1, _greeting2, message = two_card_forest
        assert chat_controller._descend_to_latest(f, card1) == message

    def test_one_step_stops_at_the_greeting(self, two_card_forest):
        # What deleting a system prompt wants: the start of the chat under the card we landed on, not the
        # middle of whatever conversation was last held there.
        f, card1, _card2, greeting1, _greeting2, _message = two_card_forest
        assert chat_controller._descend_to_latest(f, card1, recursive=False) == greeting1

    def test_the_newest_child_is_the_one_taken(self, two_card_forest):
        f, card1, _card2, _greeting1, _greeting2, _message = two_card_forest
        newer_greeting = f.create_node(_payload("assistant", "a later greeting", 99), parent_id=card1)
        assert chat_controller._descend_to_latest(f, card1, recursive=False) == newer_greeting

    def test_a_node_with_no_children_is_its_own_answer(self, two_card_forest):
        # Which is why neither caller needs a special case for a card whose chat has not started.
        f, _card1, _card2, _greeting1, greeting2, _message = two_card_forest
        assert chat_controller._descend_to_latest(f, greeting2) == greeting2
        assert chat_controller._descend_to_latest(f, greeting2, recursive=False) == greeting2


class TestSystemPromptNodeIds:
    def test_every_root_is_a_system_prompt_node(self, two_card_forest):
        f, card1, card2, _greeting1, _greeting2, _message = two_card_forest
        assert set(chat_controller._get_all_system_prompt_node_ids(datastore=f)) == {card1, card2}

    def test_a_deleted_card_drops_out(self, two_card_forest):
        f, card1, card2, _greeting1, _greeting2, _message = two_card_forest
        f.delete_subtree(card2)
        assert chat_controller._get_all_system_prompt_node_ids(datastore=f) == [card1]
