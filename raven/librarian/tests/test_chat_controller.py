"""Unit tests for raven.librarian.chat_controller.

Only the datastore-side helpers so far — the ones that decide which of a message's buttons are live. Those
are pure functions over a `Forest` and need no GUI. (The descent they used to sit beside now lives in
`chatutil.descend_to_latest`, and is tested there.)

The module they live in needs rather more, though, so the skip names the module under test rather than any
one of its dependencies. Two of the paths that used to force this are gone — the avatar controller is a
`TYPE_CHECKING` import now, and `raven.client.api` is reached through `chat_controller._client_api()` — but
several remain, and `python scripts/check_ci_imports.py` names them: `hybridir` (bm25s, chromadb, watchdog),
`scaffold`'s own route to `raven.client.api` (spaCy), the audio player (pygame), the codec (av). Clearing
those is a dependency-hygiene sweep across several modules rather than a change to this one, and until it
happens these tests do not run in the minimal-dependency CI job even though they would pass there.
"""

import pytest

pytest.importorskip("raven.librarian.chat_controller")  # noqa: E402 -- still reaches the ML stack; see above

from raven.librarian import chat_controller  # noqa: E402


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

    def test_a_message_the_user_sent_is_not_a_greeting_however_it_sits(self, two_card_forest, chat_payload):
        # HEAD can rest on a system prompt node — deleting another card lands there — and a message sent
        # from there attaches beside the greetings. Going by position alone would call it one, and a
        # greeting has its reroll, continue, branch and delete buttons taken away: the user would be left
        # with a message of their own that they cannot remove.
        f, card1, _card2, _greeting1, _greeting2, _message = two_card_forest
        typed_at_the_root = f.create_node(chat_payload("user", "sent with HEAD on the card", 3), parent_id=card1)
        assert typed_at_the_root not in chat_controller._get_all_greeting_node_ids(datastore=f)


class TestSystemPromptNodeIds:
    def test_every_root_is_a_system_prompt_node(self, two_card_forest):
        f, card1, card2, _greeting1, _greeting2, _message = two_card_forest
        assert set(chat_controller._get_all_system_prompt_node_ids(datastore=f)) == {card1, card2}

    def test_a_deleted_card_drops_out(self, two_card_forest):
        f, card1, card2, _greeting1, _greeting2, _message = two_card_forest
        f.delete_subtree(card2)
        assert chat_controller._get_all_system_prompt_node_ids(datastore=f) == [card1]
