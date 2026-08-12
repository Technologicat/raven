"""Unit tests for raven.librarian.chat_controller.

Only the datastore-side helpers so far — the ones that decide which of a message's buttons are live. Those
are pure functions over a `Forest`, so they need no GUI, but the module imports DearPyGui at import time,
hence the skip below.
"""

import pytest

pytest.importorskip("dearpygui")  # noqa: E402 -- the module under test imports it at module level

from raven.librarian import chat_controller, chattree  # noqa: E402


@pytest.fixture
def two_card_forest():
    """Two system prompts, each with its own greeting, and one message under the first.

    The shape `appstate` produces once the datastore has seen more than one system prompt: every root is a
    system prompt node, and its children are the greetings recorded under it.
    """
    f = chattree.Forest()
    card1 = f.create_node("system prompt 1", parent_id=None)
    card2 = f.create_node("system prompt 2", parent_id=None)
    greeting1 = f.create_node("greeting under card 1", parent_id=card1)
    greeting2 = f.create_node("greeting under card 2", parent_id=card2)
    message = f.create_node("a user message", parent_id=greeting1)
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


class TestSystemPromptNodeIds:
    def test_every_root_is_a_system_prompt_node(self, two_card_forest):
        f, card1, card2, _greeting1, _greeting2, _message = two_card_forest
        assert set(chat_controller._get_all_system_prompt_node_ids(datastore=f)) == {card1, card2}

    def test_a_deleted_card_drops_out(self, two_card_forest):
        f, card1, card2, _greeting1, _greeting2, _message = two_card_forest
        f.delete_subtree(card2)
        assert chat_controller._get_all_system_prompt_node_ids(datastore=f) == [card1]
