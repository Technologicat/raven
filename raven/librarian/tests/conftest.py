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
                         "tool": None})
