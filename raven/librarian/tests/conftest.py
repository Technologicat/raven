"""Shared pytest fixtures for raven.librarian tests."""

import pytest

from unpythonic.env import env

from raven.librarian import chattree, llmclient


def chat_node_payload(role, text, timestamp=0):
    """A chat node payload carrying the two fields the tree-shape tests read: the role, and the timestamp."""
    return {"message": {"role": role, "content": [{"type": "text", "text": text}], "tool_calls": []},
            "general_metadata": {"persona": None, "timestamp": timestamp}}


@pytest.fixture
def chat_payload():
    """`chat_node_payload`, for a test that adds nodes of its own to a fixture forest."""
    return chat_node_payload


@pytest.fixture
def in_memory_forest():
    """A bare `Forest`, for tests about attachments rather than about tree shape.

    An in-memory forest carries sidecars like a persistent one — the policy is `Forest`'s and only the
    bytes' home differs — so a document can be stored and read back with no temporary directory.
    """
    return chattree.Forest()


@pytest.fixture
def two_card_forest():
    """Two system prompts, each with its own greeting, and one message under the first.

    The shape `appstate` produces once the datastore has seen more than one system prompt: every root is a
    system prompt node, and its children are the greetings recorded under it. Shared, because two modules
    ask questions of it — `chatutil.descend_to_latest` about where a descent lands, and `chat_controller`
    about which nodes are greetings.
    """
    f = chattree.Forest()
    card1 = f.create_node(chat_node_payload("system", "system prompt 1"), parent_id=None)
    card2 = f.create_node(chat_node_payload("system", "system prompt 2"), parent_id=None)
    greeting1 = f.create_node(chat_node_payload("assistant", "greeting under card 1", 1), parent_id=card1)
    greeting2 = f.create_node(chat_node_payload("assistant", "greeting under card 2", 1), parent_id=card2)
    message = f.create_node(chat_node_payload("user", "a user message", 2), parent_id=greeting1)
    return f, card1, card2, greeting1, greeting2, message


@pytest.fixture
def llm_settings(monkeypatch):
    """The real `llm_settings` env, built without contacting a backend.

    `llmclient.configure` is `setup` minus the two network queries that discover its arguments, so a caller
    holding the facts gets the genuine object — the same system prompt, character card, tool tables, sampler
    settings and personas a running app has. Forging one instead means tests assert against a replica that
    can drift from what the app actually uses, and the drift is invisible until something depends on a field
    the replica got wrong.

    That was not available until `llmclient` became importable without the client stack: it used to reach
    `spacy` and `transformers` through `raven.client.api`, and an import failure in a *conftest* takes the
    whole package's collection with it rather than skipping the few tests that need a backend. `api` is now
    imported on first use by the two tool wrappers that need it, so this import is safe — including in CI,
    which installs neither.

    Two things are pinned rather than taken from the local configuration, so that the fixture describes the
    same model on every machine:

      - `model_info`, which a running app discovers from the backend. Stated here.
      - `llm_tokenizer_path`, which selects exact token counting when set. Left unset, so `count_tokens`
        takes the estimate path, which needs no model files — the exactness of the count is not what any of
        these tests are about, and a configured tokenizer would otherwise be fetched.
    """
    monkeypatch.setattr(llmclient.librarian_config, "llm_tokenizer_path", None)
    return llmclient.configure(model_info=env(label="test-model",
                                              model_id="test-model",
                                              context_length=32768,
                                              is_vlm=None,  # "cannot tell", as a backend that does not report it gives
                                              loaded=True),  # a test that gets as far as a turn is testing one against a backend that could answer
                               backend_flavor="lmstudio",
                               backend_url="http://test-backend",
                               quiet=True)
