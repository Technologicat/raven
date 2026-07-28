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
               # Tool registry, as `llmclient.setup` builds it: every tool is registered for the session,
               # and `ai_turn` picks the subset to offer on each turn. The entrypoints are never called
               # here (tests fake `perform_tool_calls`), so the names are what matter.
               tool_entrypoints={"websearch": None,
                                 "webfetch": None,
                                 "search_documents": None,
                                 "fetch_document": None},
               document_tool_names=frozenset({"search_documents", "fetch_document"}))
