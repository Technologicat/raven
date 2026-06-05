"""Unit tests for raven.librarian.chatutil."""

import copy
import datetime
import json
import re
import time

import pytest

from unpythonic.env import env

from raven.librarian import chattree, chatutil


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def llm_settings():
    """A lightweight mock of the llm_settings env returned by llmclient.setup."""
    return env(personas={"user": "User", "assistant": "Aria"},
               system_prompt="You are a helpful assistant.",
               character_card="Name: Aria",
               greeting="How can I help you today?")


@pytest.fixture
def forest():
    """An empty in-memory chat tree."""
    return chattree.Forest()


@pytest.fixture
def populated_forest(llm_settings):
    """A forest initialized via factory_reset_datastore (system prompt + greeting)."""
    f = chattree.Forest()
    greeting_id = chatutil.factory_reset_datastore(f, llm_settings)
    return f, greeting_id


# ---------------------------------------------------------------------------
# Display formatting: format_message_number
# ---------------------------------------------------------------------------

class TestFormatMessageNumber:
    def test_none_returns_empty(self):
        assert chatutil.format_message_number(None, None) == ""

    def test_integer_no_markup(self):
        assert chatutil.format_message_number(42, None) == "[#42]"

    def test_zero(self):
        assert chatutil.format_message_number(0, None) == "[#0]"

    def test_markdown(self):
        assert chatutil.format_message_number(1, "markdown") == "*[#1]*"

    def test_ansi(self):
        result = chatutil.format_message_number(1, "ansi")
        # Should contain ANSI escape codes and the number
        assert "[#1]" in result
        assert "\x1b[" in result

    def test_invalid_markup_raises(self):
        with pytest.raises(ValueError, match="unknown markup kind"):
            chatutil.format_message_number(1, "html")


# ---------------------------------------------------------------------------
# Display formatting: format_persona
# ---------------------------------------------------------------------------

class TestFormatPersona:
    def test_named_persona_no_markup(self):
        assert chatutil.format_persona("assistant", "Aria", None) == "Aria"

    def test_named_persona_markdown(self):
        assert chatutil.format_persona("assistant", "Aria", "markdown") == "**Aria**"

    def test_named_persona_ansi(self):
        result = chatutil.format_persona("assistant", "Aria", "ansi")
        assert "Aria" in result
        assert "\x1b[" in result

    def test_none_persona_no_markup(self):
        assert chatutil.format_persona("system", None, None) == "<<system>>"

    def test_none_persona_markdown(self):
        assert chatutil.format_persona("system", None, "markdown") == "`<<system>>`"

    def test_none_persona_ansi(self):
        result = chatutil.format_persona("system", None, "ansi")
        assert "<<system>>" in result

    def test_invalid_markup_raises(self):
        with pytest.raises(ValueError):
            chatutil.format_persona("user", "User", "xml")


# ---------------------------------------------------------------------------
# Display formatting: format_message_heading
# ---------------------------------------------------------------------------

class TestFormatMessageHeading:
    def test_with_number_and_persona(self):
        result = chatutil.format_message_heading(1, "assistant", "Aria", None)
        assert result == "[#1] Aria: "

    def test_without_number(self):
        result = chatutil.format_message_heading(None, "assistant", "Aria", None)
        assert result == "Aria: "

    def test_system_role_no_persona(self):
        result = chatutil.format_message_heading(1, "system", None, None)
        assert result == "[#1] <<system>>: "

    def test_markdown(self):
        result = chatutil.format_message_heading(1, "assistant", "Aria", "markdown")
        assert "**Aria**" in result
        assert "*[#1]*" in result


# ---------------------------------------------------------------------------
# Timestamp utilities
# ---------------------------------------------------------------------------

class TestMakeTimestamp:
    def test_explicit_timestamp(self):
        # 2025-01-15 12:00:00 UTC in nanoseconds
        ts_ns = 1736942400_000_000_000
        ts, weekday, isodate, isotime = chatutil.make_timestamp(ts_ns)
        assert ts == ts_ns
        assert isinstance(weekday, str)
        assert isinstance(isodate, str)
        assert isinstance(isotime, str)
        # Verify the date parses correctly
        assert re.match(r"\d{4}-\d{2}-\d{2}", isodate)
        assert re.match(r"\d{2}:\d{2}:\d{2}", isotime)

    def test_auto_timestamp(self):
        before = time.time_ns()
        ts, weekday, isodate, isotime = chatutil.make_timestamp()
        after = time.time_ns()
        assert before <= ts <= after

    def test_weekday_matches_date(self):
        ts_ns = 1736942400_000_000_000  # known date
        ts, weekday, isodate, isotime = chatutil.make_timestamp(ts_ns)
        parsed_date = datetime.date.fromisoformat(isodate)
        expected_weekday = ["Monday", "Tuesday", "Wednesday", "Thursday",
                            "Friday", "Saturday", "Sunday"][parsed_date.weekday()]
        assert weekday == expected_weekday

    def test_returns_four_tuple(self):
        result = chatutil.make_timestamp()
        assert len(result) == 4
        assert isinstance(result[0], int)


# ---------------------------------------------------------------------------
# Datetime formatting
# ---------------------------------------------------------------------------

class TestFormatDatetime:
    def test_chat_datetime_now_format(self):
        result = chatutil.format_chat_datetime_now()
        assert result.startswith("[System information:")
        assert result.endswith("]")

    def test_chatlog_datetime_now_has_weekday_date_time(self):
        result = chatutil.format_chatlog_datetime_now()
        parts = result.split()
        assert len(parts) == 3  # weekday, date, time

    def test_chatlog_date_now_has_weekday_and_date(self):
        result = chatutil.format_chatlog_date_now()
        parts = result.split()
        assert len(parts) == 2  # weekday, date


class TestFormatReminders:
    def test_focus_reminder_is_nonempty_string(self):
        result = chatutil.format_reminder_to_focus_on_latest_input()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "[System information:" in result

    def test_context_only_reminder_is_nonempty_string(self):
        result = chatutil.format_reminder_to_use_information_from_context_only()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "[System information:" in result


# ---------------------------------------------------------------------------
# Persona removal
# ---------------------------------------------------------------------------

class TestRemovePersona:
    def test_basic_removal(self):
        assert chatutil.remove_persona_from_start_of_line("Aria", "Aria: hello") == "hello"

    def test_multiple_lines(self):
        text = "Aria: line one\nAria: line two"
        result = chatutil.remove_persona_from_start_of_line("Aria", text)
        assert result == "line one\nline two"

    def test_none_persona_unchanged(self):
        text = "Aria: hello"
        assert chatutil.remove_persona_from_start_of_line(None, text) == text

    def test_no_match_unchanged(self):
        text = "Bob: hello"
        assert chatutil.remove_persona_from_start_of_line("Aria", text) == text

    def test_partial_name_not_removed(self):
        # "Ariadne" should NOT match "Aria" — different prefix before ":"
        text = "Ariadne: hello"
        result = chatutil.remove_persona_from_start_of_line("Aria", text)
        assert result == "Ariadne: hello"

    def test_mid_line_not_removed(self):
        text = "said Aria: hello"
        result = chatutil.remove_persona_from_start_of_line("Aria", text)
        assert result == "said Aria: hello"

    def test_persona_in_middle_of_text_unchanged(self):
        text = "hello Aria: world"
        result = chatutil.remove_persona_from_start_of_line("Aria", text)
        assert result == "hello Aria: world"


# ---------------------------------------------------------------------------
# scrub — the main complex function
# ---------------------------------------------------------------------------

class TestScrub:
    # -- Complete thought blocks --

    def test_complete_thought_discard(self):
        text = "<think>some thoughts</think>the answer"
        result = chatutil.scrub(persona=None, text=text,
                                thoughts_mode="discard", markup=None,
                                add_persona=False)
        assert result == "the answer"

    def test_complete_thought_keep(self):
        text = "<think>some thoughts</think>the answer"
        result = chatutil.scrub(persona=None, text=text,
                                thoughts_mode="keep", markup=None,
                                add_persona=False)
        assert "<think>" in result
        assert "some thoughts" in result
        assert "the answer" in result

    def test_complete_thought_markup_markdown(self):
        text = "<think>some thoughts</think>the answer"
        result = chatutil.scrub(persona=None, text=text,
                                thoughts_mode="markup", markup="markdown",
                                add_persona=False)
        assert "some thoughts" in result
        assert "the answer" in result
        assert "<think>" not in result  # raw tags should be gone
        assert "Thought" in result  # decorated header

    def test_complete_thought_markup_ansi(self):
        text = "<think>some thoughts</think>the answer"
        result = chatutil.scrub(persona=None, text=text,
                                thoughts_mode="markup", markup="ansi",
                                add_persona=False)
        assert "some thoughts" in result
        assert "the answer" in result
        assert "\x1b[" in result  # ANSI codes present

    def test_complete_thought_markup_none(self):
        # markup=None with thoughts_mode="markup" → same as "keep"
        text = "<think>some thoughts</think>the answer"
        result = chatutil.scrub(persona=None, text=text,
                                thoughts_mode="markup", markup=None,
                                add_persona=False)
        assert "<think>" in result or "some thoughts" in result

    # -- Incomplete thought blocks --

    def test_incomplete_thought_discard(self):
        text = "<think>partial thoughts without closing tag"
        result = chatutil.scrub(persona=None, text=text,
                                thoughts_mode="discard", markup=None,
                                add_persona=False)
        assert "partial thoughts" not in result

    def test_incomplete_thought_keep(self):
        text = "<think>partial thoughts without closing tag"
        result = chatutil.scrub(persona=None, text=text,
                                thoughts_mode="keep", markup=None,
                                add_persona=False)
        assert "partial thoughts" in result

    # -- Missing opening tag (QwQ-32B quirk) --

    def test_missing_opening_tag_discard(self):
        text = "thoughts here</think>the answer"
        result = chatutil.scrub(persona=None, text=text,
                                thoughts_mode="discard", markup=None,
                                add_persona=False)
        assert result == "the answer"

    def test_missing_opening_tag_keep(self):
        text = "thoughts here</think>the answer"
        result = chatutil.scrub(persona=None, text=text,
                                thoughts_mode="keep", markup=None,
                                add_persona=False)
        assert "thoughts here" in result
        assert "the answer" in result

    # -- Doubled think tag --

    def test_doubled_think_tag(self):
        text = "<think>\n<think>\nsome thoughts</think>the answer"
        result = chatutil.scrub(persona=None, text=text,
                                thoughts_mode="discard", markup=None,
                                add_persona=False)
        assert result == "the answer"

    # -- NaN thought block --

    def test_nan_thought_block(self):
        text = "<think>\nNaN\n</think>\nthe answer"
        result = chatutil.scrub(persona=None, text=text,
                                thoughts_mode="discard", markup=None,
                                add_persona=False)
        assert result == "the answer"

    # -- Square bracket variants --

    def test_square_bracket_think(self):
        text = "[think]some thoughts[/think]the answer"
        result = chatutil.scrub(persona=None, text=text,
                                thoughts_mode="discard", markup=None,
                                add_persona=False)
        assert result == "the answer"

    def test_thinking_tag(self):
        text = "<thinking>some thoughts</thinking>the answer"
        result = chatutil.scrub(persona=None, text=text,
                                thoughts_mode="discard", markup=None,
                                add_persona=False)
        assert result == "the answer"

    # -- Persona handling in scrub --

    def test_persona_removed_from_text(self):
        text = "Aria: <think>thoughts</think>the answer"
        result = chatutil.scrub(persona="Aria", text=text,
                                thoughts_mode="discard", markup=None,
                                add_persona=False)
        assert result == "the answer"

    def test_add_persona_true(self):
        text = "the answer"
        result = chatutil.scrub(persona="Aria", text=text,
                                thoughts_mode="discard", markup=None,
                                add_persona=True)
        assert result == "Aria: the answer"

    def test_add_persona_false(self):
        text = "the answer"
        result = chatutil.scrub(persona="Aria", text=text,
                                thoughts_mode="discard", markup=None,
                                add_persona=False)
        assert result == "the answer"

    def test_add_persona_none_persona(self):
        text = "the answer"
        result = chatutil.scrub(persona=None, text=text,
                                thoughts_mode="discard", markup=None,
                                add_persona=True)
        # add_persona with None persona should not add anything
        assert result == "the answer"

    # -- Edge cases --

    def test_no_thought_block(self):
        text = "just plain text"
        result = chatutil.scrub(persona=None, text=text,
                                thoughts_mode="discard", markup=None,
                                add_persona=False)
        assert result == "just plain text"

    def test_whitespace_stripping(self):
        text = "  \n  the answer  \n  "
        result = chatutil.scrub(persona=None, text=text,
                                thoughts_mode="discard", markup=None,
                                add_persona=False)
        assert result == "the answer"

    def test_empty_after_scrub(self):
        text = "<think>only thoughts</think>"
        result = chatutil.scrub(persona=None, text=text,
                                thoughts_mode="discard", markup=None,
                                add_persona=False)
        assert result == ""

    def test_invalid_thoughts_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown thoughts_mode"):
            chatutil.scrub(persona=None, text="text",
                           thoughts_mode="invalid", markup=None,
                           add_persona=False)

    def test_invalid_markup_raises(self):
        with pytest.raises(ValueError, match="unknown markup kind"):
            chatutil.scrub(persona=None, text="text",
                           thoughts_mode="discard", markup="html",
                           add_persona=False)

    def test_multiline_thought_block(self):
        text = "<think>\nline 1\nline 2\nline 3\n</think>\nthe answer"
        result = chatutil.scrub(persona=None, text=text,
                                thoughts_mode="discard", markup=None,
                                add_persona=False)
        assert result == "the answer"


# ---------------------------------------------------------------------------
# Chat message creation
# ---------------------------------------------------------------------------

class TestContentParts:
    """Content-parts helpers (brief 03): `text_content_part` / `normalize_content` / `content_to_text`."""

    def test_text_content_part_shape(self):
        assert chatutil.text_content_part("hi") == {"type": "text", "text": "hi"}

    def test_normalize_content_wraps_string(self):
        assert chatutil.normalize_content("hello") == [{"type": "text", "text": "hello"}]

    def test_normalize_content_passes_list_through_unchanged(self):
        parts = [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "sidecar:x.png"}}]
        assert chatutil.normalize_content(parts) is parts  # idempotent, no copy

    def test_normalize_content_rejects_non_str_non_list(self):
        with pytest.raises(TypeError):
            chatutil.normalize_content(42)

    def test_content_to_text_joins_text_parts_and_skips_images(self):
        parts = [{"type": "text", "text": "Hello "},
                 {"type": "image_url", "image_url": {"url": "sidecar:x.png"}},
                 {"type": "text", "text": "world"}]
        assert chatutil.content_to_text(parts) == "Hello world"

    def test_content_to_text_none_and_empty_yield_empty_string(self):
        assert chatutil.content_to_text(None) == ""
        assert chatutil.content_to_text([]) == ""


class TestCreateChatMessage:
    def test_user_message_with_persona(self, llm_settings):
        msg = chatutil.create_chat_message(llm_settings, "user", "hello")
        assert msg["role"] == "user"
        assert msg["content"] == [{"type": "text", "text": "User: hello"}]  # string wrapped as a single text part
        assert msg["tool_calls"] == []

    def test_assistant_message(self, llm_settings):
        msg = chatutil.create_chat_message(llm_settings, "assistant", "hi there")
        assert chatutil.content_to_text(msg["content"]) == "Aria: hi there"

    def test_system_message_no_persona(self, llm_settings):
        msg = chatutil.create_chat_message(llm_settings, "system", "prompt text")
        # system has no persona in our fixture
        assert chatutil.content_to_text(msg["content"]) == "prompt text"

    def test_add_persona_false(self, llm_settings):
        msg = chatutil.create_chat_message(llm_settings, "user", "hello", add_persona=False)
        assert chatutil.content_to_text(msg["content"]) == "hello"

    def test_persona_override(self, llm_settings):
        msg = chatutil.create_chat_message(llm_settings, "assistant", "hi",
                                           persona="CustomName")
        assert chatutil.content_to_text(msg["content"]) == "CustomName: hi"

    def test_tool_calls_passed_through(self, llm_settings):
        calls = ['{"name": "search", "args": {}}']
        msg = chatutil.create_chat_message(llm_settings, "assistant", "let me search",
                                           tool_calls=calls)
        assert msg["tool_calls"] == calls

    def test_invalid_role_raises(self, llm_settings):
        with pytest.raises(ValueError, match="Unknown role"):
            chatutil.create_chat_message(llm_settings, "invalid_role", "text")

    def test_tool_role(self, llm_settings):
        msg = chatutil.create_chat_message(llm_settings, "tool", "tool result")
        assert msg["role"] == "tool"
        # tool has no persona in our fixture
        assert chatutil.content_to_text(msg["content"]) == "tool result"


class TestCreateInitialSystemMessage:
    def test_has_system_prompt_and_character_card(self, llm_settings):
        msg = chatutil.create_initial_system_message(llm_settings)
        assert msg["role"] == "system"
        content = chatutil.content_to_text(msg["content"])
        assert "helpful assistant" in content
        assert "Aria" in content
        assert "-----" in content

    def test_system_prompt_only(self):
        settings = env(personas={},
                       system_prompt="Be helpful.",
                       character_card="",
                       greeting="Hello!")
        msg = chatutil.create_initial_system_message(settings)
        content = chatutil.content_to_text(msg["content"])
        assert "Be helpful." in content
        assert "-----" in content

    def test_character_card_only(self):
        settings = env(personas={},
                       system_prompt="",
                       character_card="Name: Bot",
                       greeting="Hello!")
        msg = chatutil.create_initial_system_message(settings)
        assert "Name: Bot" in chatutil.content_to_text(msg["content"])

    def test_neither_raises(self):
        settings = env(personas={},
                       system_prompt="",
                       character_card="",
                       greeting="Hello!")
        with pytest.raises(ValueError, match="Need at least"):
            chatutil.create_initial_system_message(settings)


# ---------------------------------------------------------------------------
# Payload creation
# ---------------------------------------------------------------------------

class TestCreatePayload:
    def test_structure(self, llm_settings):
        msg = chatutil.create_chat_message(llm_settings, "user", "hello")
        payload = chatutil.create_payload(llm_settings, msg)
        assert "message" in payload
        assert "general_metadata" in payload

    def test_general_metadata_fields(self, llm_settings):
        msg = chatutil.create_chat_message(llm_settings, "user", "hello")
        payload = chatutil.create_payload(llm_settings, msg)
        meta = payload["general_metadata"]
        assert "timestamp" in meta
        assert "datetime" in meta
        assert "persona" in meta
        assert isinstance(meta["timestamp"], int)
        assert meta["persona"] == "User"

    def test_system_role_persona_is_none(self, llm_settings):
        msg = chatutil.create_chat_message(llm_settings, "system", "prompt")
        payload = chatutil.create_payload(llm_settings, msg)
        assert payload["general_metadata"]["persona"] is None

    def test_persona_override(self, llm_settings):
        msg = chatutil.create_chat_message(llm_settings, "assistant", "hi")
        payload = chatutil.create_payload(llm_settings, msg, persona="OtherAI")
        assert payload["general_metadata"]["persona"] == "OtherAI"

    def test_message_preserved(self, llm_settings):
        msg = chatutil.create_chat_message(llm_settings, "user", "hello")
        payload = chatutil.create_payload(llm_settings, msg)
        assert payload["message"] is msg


# ---------------------------------------------------------------------------
# Chat datastore utilities
# ---------------------------------------------------------------------------

class TestLinearizeChat:
    def test_linear_chain(self, llm_settings, forest):
        msg_a = chatutil.create_chat_message(llm_settings, "system", "system prompt")
        msg_b = chatutil.create_chat_message(llm_settings, "user", "hello")
        msg_c = chatutil.create_chat_message(llm_settings, "assistant", "hi there")

        id_a = forest.create_node(chatutil.create_payload(llm_settings, msg_a), parent_id=None)
        id_b = forest.create_node(chatutil.create_payload(llm_settings, msg_b), parent_id=id_a)
        id_c = forest.create_node(chatutil.create_payload(llm_settings, msg_c), parent_id=id_b)

        history = chatutil.linearize_chat(forest, id_c)
        assert len(history) == 3
        assert history[0]["role"] == "system"
        assert history[1]["role"] == "user"
        assert history[2]["role"] == "assistant"

    def test_linearize_from_middle(self, llm_settings, forest):
        msg_a = chatutil.create_chat_message(llm_settings, "system", "system prompt")
        msg_b = chatutil.create_chat_message(llm_settings, "user", "hello")
        msg_c = chatutil.create_chat_message(llm_settings, "assistant", "hi")

        id_a = forest.create_node(chatutil.create_payload(llm_settings, msg_a), parent_id=None)
        id_b = forest.create_node(chatutil.create_payload(llm_settings, msg_b), parent_id=id_a)
        forest.create_node(chatutil.create_payload(llm_settings, msg_c), parent_id=id_b)

        # Linearize from B, not C
        history = chatutil.linearize_chat(forest, id_b)
        assert len(history) == 2

    def test_single_node(self, llm_settings, forest):
        msg = chatutil.create_chat_message(llm_settings, "system", "prompt")
        id_a = forest.create_node(chatutil.create_payload(llm_settings, msg), parent_id=None)

        history = chatutil.linearize_chat(forest, id_a)
        assert len(history) == 1


class TestFactoryResetDatastore:
    def test_returns_node_id(self, llm_settings, forest):
        greeting_id = chatutil.factory_reset_datastore(forest, llm_settings)
        assert isinstance(greeting_id, str)
        assert greeting_id in forest.nodes

    def test_creates_two_nodes(self, llm_settings, forest):
        chatutil.factory_reset_datastore(forest, llm_settings)
        assert len(forest.nodes) == 2

    def test_greeting_parent_is_system_prompt(self, llm_settings, forest):
        greeting_id = chatutil.factory_reset_datastore(forest, llm_settings)
        parent_id = forest.get_parent(greeting_id)
        assert parent_id is not None
        # The parent should be a root node
        assert forest.get_parent(parent_id) is None

    def test_greeting_has_correct_content(self, llm_settings, forest):
        greeting_id = chatutil.factory_reset_datastore(forest, llm_settings)
        payload = forest.get_payload(greeting_id)
        assert payload["message"]["role"] == "assistant"
        assert "How can I help you today?" in chatutil.content_to_text(payload["message"]["content"])

    def test_purges_existing_data(self, llm_settings, forest):
        # Create some data first
        forest.create_node("junk", parent_id=None)
        forest.create_node("more junk", parent_id=None)
        assert len(forest.nodes) == 2

        # Factory reset should purge and recreate
        chatutil.factory_reset_datastore(forest, llm_settings)
        assert len(forest.nodes) == 2  # only system prompt + greeting


# ---------------------------------------------------------------------------
# compute_auto_allowed_hosts
# ---------------------------------------------------------------------------

def _build_chain(forest, messages):
    """Create a linear chain of nodes from `messages` and return the head (last) node id.

    `messages`: list of `(role, content, generation_metadata_or_None)`.
    """
    parent = None
    for role, content, gen_meta in messages:
        # Live (post-migration) format: content is a parts list. `compute_auto_allowed_hosts` reads it via
        # `content_to_text`, so build the chain the way the datastore holds it after `upgrade_datastore`.
        payload = {"message": {"role": role, "content": [chatutil.text_content_part(content)], "tool_calls": []}}
        if gen_meta is not None:
            payload["generation_metadata"] = gen_meta
        parent = forest.create_node(payload=payload, parent_id=parent)
    return parent


class TestComputeAutoAllowedHosts:
    def test_no_user_message_empty(self, forest):
        head = _build_chain(forest, [("system", "you are an AI", None)])
        assert chatutil.compute_auto_allowed_hosts(forest, head) == frozenset()

    def test_user_typed_urls_allowed(self, forest):
        head = _build_chain(forest, [
            ("user", "look at https://arxiv.org/abs/2301.1 and https://example.com/x please", None),
        ])
        assert chatutil.compute_auto_allowed_hosts(forest, head) == frozenset({"arxiv.org", "example.com"})

    def test_persona_prefixed_user_message(self, forest):
        # create_chat_message prepends "User: " for the user role; URL extraction must still work.
        head = _build_chain(forest, [("user", "User: see https://foo.org/y", None)])
        assert chatutil.compute_auto_allowed_hosts(forest, head) == frozenset({"foo.org"})

    def test_user_message_without_urls(self, forest):
        head = _build_chain(forest, [("user", "what is the capital of France?", None)])
        assert chatutil.compute_auto_allowed_hosts(forest, head) == frozenset()

    def test_only_current_turn_user_message_counts(self, forest):
        # A URL pasted in an earlier turn must NOT keep widening the trust surface.
        head = _build_chain(forest, [
            ("user", "earlier https://old-turn.com/x", None),
            ("assistant", "ok", None),
            ("user", "now a question with no link", None),
        ])
        assert chatutil.compute_auto_allowed_hosts(forest, head) == frozenset()

    def test_search_results_excluded_by_default(self, forest):
        head = _build_chain(forest, [
            ("user", "find cosmology news", None),
            ("assistant", "", None),
            ("tool", "result: https://news-site.com/article", {"function_name": "websearch"}),
        ])
        assert chatutil.compute_auto_allowed_hosts(forest, head) == frozenset()

    def test_search_results_included_when_trusted(self, forest):
        head = _build_chain(forest, [
            ("user", "find cosmology news", None),
            ("assistant", "", None),
            ("tool", "result: https://news-site.com/article", {"function_name": "websearch"}),
        ])
        result = chatutil.compute_auto_allowed_hosts(forest, head, trust_search_results=True)
        assert result == frozenset({"news-site.com"})

    def test_webfetch_output_never_auto_allowed(self, forest):
        # One-hop trust: a URL the model found *inside* a fetched page is not user intent,
        # so it must not gain auto-allow even with trust_search_results on.
        head = _build_chain(forest, [
            ("user", "fetch the news", None),
            ("assistant", "", None),
            ("tool", "page body links to https://attacker.com/inject", {"function_name": "webfetch"}),
        ])
        result = chatutil.compute_auto_allowed_hosts(forest, head, trust_search_results=True)
        assert result == frozenset()

    def test_user_url_plus_trusted_search_url(self, forest):
        head = _build_chain(forest, [
            ("user", "compare https://user-typed.com with search results", None),
            ("assistant", "", None),
            ("tool", "found https://searched.com/p", {"function_name": "websearch"}),
        ])
        result = chatutil.compute_auto_allowed_hosts(forest, head, trust_search_results=True)
        assert result == frozenset({"user-typed.com", "searched.com"})


# ---------------------------------------------------------------------------
# Datastore migration (brief 02 §11): inline reasoning / tool-call normalization
# ---------------------------------------------------------------------------

class TestMigrateInlineReasoning:
    def test_extracts_think_block(self):
        message = {"role": "assistant", "content": "Aria: <think>hmm</think>\nBla...", "tool_calls": []}
        chatutil._migrate_inline_reasoning(message)
        assert message["reasoning_content"] == "hmm"
        assert message["content"] == "Aria: Bla..."  # persona prefix preserved, tag+whitespace collapsed

    def test_inline_think_after_some_text(self):
        message = {"role": "assistant", "content": "Aria: Let me think. <think>pondering</think>\nMy answer"}
        chatutil._migrate_inline_reasoning(message)
        assert message["reasoning_content"] == "pondering"
        assert message["content"] == "Aria: Let me think. My answer"

    def test_only_thinking_yields_empty_content(self):
        message = {"role": "assistant", "content": "<think>only thinking</think>"}
        chatutil._migrate_inline_reasoning(message)
        assert message["reasoning_content"] == "only thinking"
        assert message["content"] == ""

    def test_multiple_blocks_concatenated(self):
        message = {"role": "assistant", "content": "<think>one</think>middle<think>two</think>end"}
        chatutil._migrate_inline_reasoning(message)
        assert message["reasoning_content"] == "one\n\ntwo"
        assert message["content"] == "middle end"

    def test_empty_ghost_block_dropped(self):
        # Gemma's `<think></think>` ghost: tag stripped, no reasoning_content field added.
        message = {"role": "assistant", "content": "<think></think>Just the answer."}
        chatutil._migrate_inline_reasoning(message)
        assert message["content"] == "Just the answer."
        assert "reasoning_content" not in message

    def test_no_think_is_noop(self):
        message = {"role": "assistant", "content": "plain answer"}
        chatutil._migrate_inline_reasoning(message)
        assert message == {"role": "assistant", "content": "plain answer"}

    def test_idempotent(self):
        message = {"role": "assistant", "content": "<think>x</think>y"}
        chatutil._migrate_inline_reasoning(message)
        once = dict(message)
        chatutil._migrate_inline_reasoning(message)
        assert message == once


class TestMigrateInlineToolCalls:
    def test_extracts_and_strips(self):
        message = {"role": "assistant",
                   "content": '<tool_call>{"name": "websearch", "arguments": {"query": "ravens"}}</tool_call>',
                   "tool_calls": []}
        chatutil._migrate_inline_tool_calls(message)
        assert message["content"] == ""
        assert len(message["tool_calls"]) == 1
        assert message["tool_calls"][0]["function"]["name"] == "websearch"
        assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"query": "ravens"}

    def test_dedup_against_existing_structured_call(self):
        # Already in tool_calls (structured) AND inline in content: strip the inline copy, don't double-add.
        existing = [{"type": "function", "id": "call_1",
                     "function": {"name": "websearch", "arguments": '{"query": "ravens"}'}}]
        message = {"role": "assistant",
                   "content": '<tool_call>{"name": "websearch", "arguments": {"query": "ravens"}}</tool_call>',
                   "tool_calls": list(existing)}
        chatutil._migrate_inline_tool_calls(message)
        assert message["content"] == ""
        assert len(message["tool_calls"]) == 1  # no duplicate added

    def test_malformed_json_leaves_content_unchanged(self):
        # §11.4 lossless fallback: broken inline JSON -> don't strip, don't extract.
        original = '<tool_call>{"name": "websearch", arguments oops}</tool_call>'
        message = {"role": "assistant", "content": original, "tool_calls": []}
        chatutil._migrate_inline_tool_calls(message)
        assert message["content"] == original
        assert message["tool_calls"] == []

    def test_no_tool_call_is_noop(self):
        message = {"role": "assistant", "content": "no calls here", "tool_calls": []}
        chatutil._migrate_inline_tool_calls(message)
        assert message["content"] == "no calls here"

    def test_idempotent(self):
        message = {"role": "assistant",
                   "content": 'pre <tool_call>{"name": "f", "arguments": {}}</tool_call> post',
                   "tool_calls": []}
        chatutil._migrate_inline_tool_calls(message)
        once = {"content": message["content"], "tool_calls": list(message["tool_calls"])}
        chatutil._migrate_inline_tool_calls(message)
        assert message["content"] == once["content"]
        assert len(message["tool_calls"]) == len(once["tool_calls"])


class TestMigrateToolCallId:
    def test_moves_to_message(self):
        payload = {"message": {"role": "tool", "content": "result"},
                   "generation_metadata": {"status": "success", "toolcall_id": "call_7", "function_name": "websearch"}}
        chatutil._migrate_tool_call_id(payload)
        assert payload["message"]["tool_call_id"] == "call_7"
        assert "toolcall_id" not in payload["generation_metadata"]
        assert payload["generation_metadata"]["function_name"] == "websearch"  # other fields stay

    def test_non_tool_role_untouched(self):
        payload = {"message": {"role": "assistant", "content": "hi"},
                   "generation_metadata": {"toolcall_id": "call_x"}}
        chatutil._migrate_tool_call_id(payload)
        assert "tool_call_id" not in payload["message"]
        assert payload["generation_metadata"]["toolcall_id"] == "call_x"

    def test_idempotent(self):
        payload = {"message": {"role": "tool", "content": "r"},
                   "generation_metadata": {"status": "success", "toolcall_id": "call_7"}}
        chatutil._migrate_tool_call_id(payload)
        chatutil._migrate_tool_call_id(payload)  # second pass: nothing left to move
        assert payload["message"]["tool_call_id"] == "call_7"
        assert "toolcall_id" not in payload["generation_metadata"]


class TestUpgradeDatastoreReasoningAndToolCallId:
    def _old_format_forest(self, llm_settings):
        f = chattree.Forest()
        system_id = f.create_node(payload={"message": {"role": "system", "content": "sys", "tool_calls": []}},
                                  parent_id=None)
        assistant_id = f.create_node(
            payload={"message": {"role": "assistant",
                                 "content": 'Aria: <think>let me search</think>\n<tool_call>{"name": "websearch", "arguments": {"query": "x"}}</tool_call>',
                                 "tool_calls": []}},
            parent_id=system_id)
        tool_id = f.create_node(
            payload={"message": {"role": "tool", "content": "search result"},
                     "generation_metadata": {"status": "success", "toolcall_id": "call_9", "function_name": "websearch"}},
            parent_id=assistant_id)
        return f, system_id, assistant_id, tool_id

    def test_end_to_end_migration(self, llm_settings):
        f, system_id, assistant_id, tool_id = self._old_format_forest(llm_settings)
        chatutil.upgrade_datastore(llm_settings, f, system_id)

        assistant_msg = f.get_payload(assistant_id)["message"]
        assert assistant_msg["reasoning_content"] == "let me search"
        assert "<think>" not in chatutil.content_to_text(assistant_msg["content"])
        assert "<tool_call>" not in chatutil.content_to_text(assistant_msg["content"])
        assert len(assistant_msg["tool_calls"]) == 1
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "websearch"

        tool_msg = f.get_payload(tool_id)["message"]
        assert tool_msg["tool_call_id"] == "call_9"
        assert "toolcall_id" not in f.get_payload(tool_id)["generation_metadata"]

    def test_migration_is_idempotent(self, llm_settings):
        # The brief's repeat-safety check: a second migration over migrated data produces no changes.
        f, system_id, assistant_id, tool_id = self._old_format_forest(llm_settings)
        chatutil.upgrade_datastore(llm_settings, f, system_id)
        snapshot = {nid: copy.deepcopy(f.get_payload(nid)) for nid in (system_id, assistant_id, tool_id)}
        chatutil.upgrade_datastore(llm_settings, f, system_id)
        for nid, before in snapshot.items():
            assert f.get_payload(nid) == before

    def test_no_think_or_tool_call_substrings_remain(self, llm_settings):
        # Idempotency invariant from the brief: no message content retains the inline tags after migration.
        f, system_id, assistant_id, tool_id = self._old_format_forest(llm_settings)
        chatutil.upgrade_datastore(llm_settings, f, system_id)
        for nid in (system_id, assistant_id, tool_id):
            content = chatutil.content_to_text(f.get_payload(nid)["message"]["content"])
            assert "<think>" not in content
            assert "<tool_call>" not in content

    def test_string_content_wrapped_to_text_part(self, llm_settings):
        # brief 03 §3b: legacy string `content` becomes a single text part, with the persona prefix preserved
        # verbatim. The wrap runs after the §11 stanzas, so the wrapped text is already think/tool_call-free.
        f, system_id, assistant_id, tool_id = self._old_format_forest(llm_settings)
        chatutil.upgrade_datastore(llm_settings, f, system_id)
        for nid in (system_id, assistant_id, tool_id):
            content = f.get_payload(nid)["message"]["content"]
            assert isinstance(content, list)
            assert all(part["type"] == "text" for part in content)
        # Messages with nothing to extract are wrapped verbatim (single text part holding the original string).
        assert f.get_payload(system_id)["message"]["content"] == [{"type": "text", "text": "sys"}]
        assert f.get_payload(tool_id)["message"]["content"] == [{"type": "text", "text": "search result"}]
