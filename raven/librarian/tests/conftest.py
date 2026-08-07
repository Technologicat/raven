"""Shared pytest fixtures for raven.librarian tests."""

import pytest

from unpythonic.env import env

from raven.librarian import llmclient


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
               # The real tool registry, not a copy of it. `setup` cannot run here — it needs a live backend
               # to ask for the model name, the tokenizer and the sampler defaults — but the registry is
               # module-level precisely so that this does not force the tests to retype it. A hand-copy
               # would be free to drift, and a test that asserts something about tools the product does not
               # have is worse than no test.
               #
               # The entrypoints are never called (tests fake `perform_tool_calls`), so what matters is the
               # names; the real dict supplies them and cannot disagree with itself.
               tool_entrypoints=llmclient.TOOL_ENTRYPOINTS,
               document_tool_names=llmclient.DOCUMENT_TOOL_NAMES)
