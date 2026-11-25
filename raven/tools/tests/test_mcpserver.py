"""Tests for raven.tools.mcpserver — which tools `raven-mcp` offers, and what a call answers.

Raven-server is mocked out: `llmtools.websearch` and `llmtools.webfetch` are replaced, so nothing here opens a
connection. The MCP layer itself is real, the client connecting to the server object in memory.
"""

import asyncio
from unittest.mock import patch

import pytest

from mcp import Client

from raven.librarian import config as librarian_config
from raven.librarian import llmtools
from raven.librarian import lorebookcore
from raven.tools import mcpserver


@pytest.fixture
def lorebook_dir(tmp_path, monkeypatch):
    """A lorebook with one entry, configured as the lorebook to read."""
    (tmp_path / "glossary.md").write_text("# Hydra tax\n\nThe overhead of closing an item that opens two more.\n")
    monkeypatch.setattr(librarian_config, "lorebook_dir", tmp_path)
    return tmp_path


@pytest.fixture
def no_lorebook(tmp_path, monkeypatch):
    """No lorebook: the configured folder does not exist."""
    monkeypatch.setattr(librarian_config, "lorebook_dir", tmp_path / "absent")


def _names(specs):
    return [spec["function"]["name"] for spec in specs]


class TestToolSpecs:
    """What is offered, and that it is what Raven-librarian's LLM is shown."""

    def test_the_web_tools_are_the_librarians_own_specifications(self, no_lorebook):
        librarian_specs = {spec["function"]["name"]: spec for spec in llmtools.TOOLS}
        specs = mcpserver.tool_specs()
        assert _names(specs) == ["websearch", "webfetch"]
        for spec in specs:
            assert spec is librarian_specs[spec["function"]["name"]]  # the same object: nothing to drift

    def test_the_lorebook_lookup_is_offered_when_the_lorebook_has_entries(self, lorebook_dir):
        assert _names(mcpserver.tool_specs()) == ["websearch", "webfetch", lorebookcore.TOOL_NAME]

    def test_the_lorebook_lookup_is_not_offered_without_entries(self, no_lorebook):
        assert lorebookcore.TOOL_NAME not in _names(mcpserver.tool_specs())

    def test_the_lorebook_lookup_can_be_switched_off(self, lorebook_dir):
        assert lorebookcore.TOOL_NAME not in _names(mcpserver.tool_specs(lorebook=False))


class TestCallTool:
    """Dispatch, and the answer as a list of texts."""

    def test_a_search_answers_with_one_text_per_result(self):
        parts = [{"type": "text", "text": "first result"}, {"type": "text", "text": "second result"}]
        with patch.object(llmtools, "websearch", return_value=(parts, {"grounding": True})) as websearch:
            assert mcpserver.call_tool("websearch", {"query": "hydrogen"}) == ["first result", "second result"]
        websearch.assert_called_once_with("hydrogen")

    def test_a_canned_answer_is_passed_on_as_the_model_would_read_it(self):
        with patch.object(llmtools, "websearch", return_value=(llmtools.CANONICAL_NO_WEB_RESULTS, {"grounding": False})):
            assert mcpserver.call_tool("websearch", {"query": "hydrogen"}) == [llmtools.CANONICAL_NO_WEB_RESULTS]

    def test_a_fetch_goes_to_the_librarians_webfetch(self):
        with patch.object(llmtools, "webfetch", return_value=("the page", {"grounding": True})) as webfetch:
            assert mcpserver.call_tool("webfetch", {"url": "https://example.com"}) == ["the page"]
        webfetch.assert_called_once_with("https://example.com", format="markdown")

    def test_a_raw_fetch_passes_the_format_on(self):
        with patch.object(llmtools, "webfetch", return_value=("<html/>", {"grounding": True})) as webfetch:
            mcpserver.call_tool("webfetch", {"url": "https://example.com", "format": "raw"})
        webfetch.assert_called_once_with("https://example.com", format="raw")

    def test_a_long_page_is_cut_off_and_says_so(self):
        with patch.object(llmtools, "webfetch", return_value=("x" * 250, {"grounding": True})):
            [text] = mcpserver.call_tool("webfetch", {"url": "https://example.com"}, max_chars=100)
        assert text.startswith("x" * 100)
        assert "x" * 101 not in text
        assert "the first 100 of 250 characters" in text

    def test_a_page_within_the_cap_is_whole(self):
        with patch.object(llmtools, "webfetch", return_value=("x" * 100, {"grounding": True})):
            assert mcpserver.call_tool("webfetch", {"url": "https://example.com"}, max_chars=100) == ["x" * 100]

    def test_a_lookup_reads_the_lorebook(self, lorebook_dir):
        [text] = mcpserver.call_tool(lorebookcore.TOOL_NAME, {"term": "hydra tax"})
        assert "closing an item that opens two more" in text

    def test_a_lookup_sees_an_edit_without_a_restart(self, lorebook_dir):
        (lorebook_dir / "glossary.md").write_text("# Hydra tax\n\nRewritten since the server started.\n")
        [text] = mcpserver.call_tool(lorebookcore.TOOL_NAME, {"term": "hydra tax"})
        assert "Rewritten since the server started" in text

    def test_a_switched_off_lookup_is_not_a_tool(self, lorebook_dir):
        with pytest.raises(ValueError, match="no tool named"):
            mcpserver.call_tool(lorebookcore.TOOL_NAME, {"term": "hydra tax"}, lorebook=False)

    def test_an_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="no tool named"):
            mcpserver.call_tool("calculate", {"expression": "2 + 2"})


class TestOverMCP:
    """The same, through an MCP client connected to the server in memory."""

    @staticmethod
    def _run(coroutine_function):
        return asyncio.run(coroutine_function())

    def test_the_client_sees_the_tools_with_their_schemas(self, lorebook_dir):
        async def listing():
            async with Client(mcpserver._make_server(lorebook=True)) as client:
                return (await client.list_tools()).tools
        tools = self._run(listing)
        assert [tool.name for tool in tools] == ["websearch", "webfetch", lorebookcore.TOOL_NAME]
        websearch_spec = next(spec for spec in llmtools.TOOLS if spec["function"]["name"] == "websearch")
        assert tools[0].description == websearch_spec["function"]["description"]
        assert tools[0].input_schema == websearch_spec["function"]["parameters"]

    def test_a_call_answers_with_text_content(self, no_lorebook):
        async def calling():
            async with Client(mcpserver._make_server(lorebook=True)) as client:
                return await client.call_tool("webfetch", {"url": "https://example.com"})
        with patch.object(llmtools, "webfetch", return_value=("the page", {"grounding": True})):
            result = self._run(calling)
        assert not result.is_error
        assert [content.text for content in result.content] == ["the page"]

    def test_a_tool_that_fails_answers_with_an_error_result(self, no_lorebook):
        async def calling():
            async with Client(mcpserver._make_server(lorebook=True)) as client:
                return await client.call_tool("websearch", {"query": "hydrogen"})
        with patch.object(llmtools, "websearch", side_effect=RuntimeError("server on fire")):
            result = self._run(calling)
        assert result.is_error
        assert "server on fire" in result.content[0].text
