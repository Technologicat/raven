"""Shared pytest fixtures for raven.librarian tests."""

import pytest

from unpythonic.env import env


@pytest.fixture
def llm_settings():
    """Minimal `llm_settings` env for tests that don't talk to a real LLM backend.

    Covers the fields read by `chatutil`, `scaffold`, and `appstate`: persona names,
    system prompt and character card (used by `create_initial_system_message`),
    greeting (used by `appstate._refresh_greeting`), and the `personas` map
    (used by `create_chat_message` to prefix messages with the speaker's name).
    """
    return env(user="User",
               char="Aria",
               model="test-model",
               system_prompt="You are a helpful assistant.",
               character_card="Name: Aria",
               greeting="How can I help you today?",
               personas={"user": "User",
                         "assistant": "Aria",
                         "system": None,
                         "tool": None},
               # Token accounting, as `llmclient.setup` builds it. `tokenizer=None` selects the estimate
               # path (`count_tokens` tier 3), which needs no model files - the exactness of the count is
               # not what any of these tests are about.
               tokenizer=None,
               tokens_per_character=0.27,
               context_length=32768,
               backend_flavor="lmstudio",
               # A copy of the tool registry, and it has to be one: importing `llmclient` to get the real
               # thing pulls `spacy` and `transformers` through `raven.client.api`, which CI deliberately
               # does not install — so a module-level import here fails *collection* for this whole
               # package, and every librarian test disappears rather than the few that need a backend.
               # `test_llmclient` `importorskip`s for the same reason; this file cannot, because it is the
               # conftest and everything else in the directory depends on it.
               #
               # The copy is guarded rather than trusted: `TestToolRegistry.test_the_fixture_matches_the_real_registry`
               # compares it against `llmclient.TOOL_ENTRYPOINTS` and fails if they diverge. That test skips
               # in CI along with the rest of `test_llmclient`, which is the right place for the cost to
               # land — drift is a thing a developer introduces while adding a tool, and they run the full
               # suite locally, where the import works.
               #
               # The entrypoints are never called (tests fake `perform_tool_calls`), so the names are what
               # matter; `None` stands in for each function.
               tool_entrypoints={"websearch": None,
                                 "webfetch": None,
                                 "get_current_time": None,
                                 "search_documents": None,
                                 "fetch_document": None,
                                 "list_consulted_documents": None},
               document_tool_names=frozenset({"search_documents", "fetch_document", "list_consulted_documents"}))
