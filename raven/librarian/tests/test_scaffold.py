"""Unit tests for raven.librarian.scaffold (user_turn, ai_turn)."""

import pytest

# `scaffold` transitively imports `llmclient` → `raven.client.api`, which pulls
# the heavy dep stack (qoi, spaCy, the TTS stack). The CI test job runs a
# hand-picked minimal subset of dependencies; when the heavy stack is missing,
# skip this whole module. Mirrors the pattern in `test_api.py` (importorskip on
# qoi) and `test_hybridir.py` (importorskip on chromadb/bm25s).
pytest.importorskip("raven.librarian.scaffold",
                    reason="scaffold transitively needs the full raven-client dep stack")

from unpythonic.env import env  # noqa: E402 -- after importorskip by design

from raven.librarian import chattree, chatutil, scaffold  # noqa: E402 -- after importorskip by design


# ---------------------------------------------------------------------------
# Fakes for the external seams scaffold calls into
# ---------------------------------------------------------------------------

def make_invoke_result(content="Hello from the LLM.",
                       tool_calls=None,
                       n_tokens=5,
                       dt=0.1,
                       model="test-model",
                       interrupted=False):
    """Build a fake return value for `llmclient.invoke` — the shape scaffold reads."""
    return env(data={"role": "assistant",
                     "content": content,
                     "tool_calls": tool_calls},
               n_tokens=n_tokens,
               dt=dt,
               model=model,
               interrupted=interrupted)


def make_tool_response(content="tool result",
                       toolcall_id="call_0",
                       function_name="websearch",
                       status="success",
                       dt=0.01):
    """Build a fake tool response record — the shape scaffold reads from `perform_tool_calls`."""
    return env(data={"role": "tool",
                     "content": content,
                     "tool_calls": None},
               status=status,
               toolcall_id=toolcall_id,
               function_name=function_name,
               dt=dt)


class FakeRetriever:
    """Minimal stand-in for `raven.librarian.hybridir.HybridIR`.

    Only implements `.query(q, k, return_extra_info=False)`, which is the single method
    that scaffold calls on the retriever.
    """
    def __init__(self, results=None):
        self.results = list(results) if results is not None else []
        self.calls = []

    def query(self, q, k=10, return_extra_info=False):
        self.calls.append({"q": q, "k": k, "return_extra_info": return_extra_info})
        return list(self.results)


def sample_rag_match(document_id="abstract.txt", text="Sample matched content.", score=0.9, offset=0):
    return {"document_id": document_id,
            "text": text,
            "score": score,
            "offset": offset}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def populated_forest(llm_settings):
    """Forest pre-populated with a system prompt root + greeting child, via `factory_reset_datastore`.

    Returns `(forest, greeting_node_id)`; the greeting node is a valid starting HEAD for a new chat.
    """
    forest = chattree.Forest()
    greeting = chatutil.factory_reset_datastore(forest, llm_settings)
    return forest, greeting


# All callbacks that `ai_turn` accepts, in order.
_AI_TURN_CALLBACKS = ("on_docs_start", "on_docs_done",
                      "on_prompt_ready",
                      "on_llm_start", "on_llm_progress", "on_llm_done",
                      "on_nomatch_done",
                      "on_tools_start",
                      "on_call_lowlevel_start", "on_call_lowlevel_done",
                      "on_tool_done", "on_tools_done")


def run_ai_turn(forest, llm_settings, head, *,
                retriever=None,
                tools_enabled=True,
                continue_=False,
                docs_query=None,
                docs_num_results=None,
                speculate=True,
                markup=None,
                **callbacks):
    """Call `scaffold.ai_turn` with `None` defaults for unspecified callbacks."""
    cb_kwargs = {name: callbacks.get(name, None) for name in _AI_TURN_CALLBACKS}
    return scaffold.ai_turn(llm_settings=llm_settings,
                            datastore=forest,
                            retriever=retriever,
                            head_node_id=head,
                            tools_enabled=tools_enabled,
                            continue_=continue_,
                            docs_query=docs_query,
                            docs_num_results=docs_num_results,
                            speculate=speculate,
                            markup=markup,
                            **cb_kwargs)


# ---------------------------------------------------------------------------
# user_turn
# ---------------------------------------------------------------------------

class TestUserTurn:
    def test_adds_node_with_user_role(self, llm_settings, populated_forest):
        forest, head = populated_forest
        new_head = scaffold.user_turn(llm_settings=llm_settings,
                                      datastore=forest,
                                      head_node_id=head,
                                      user_message_text="Hi there")
        payload = forest.get_payload(new_head)
        assert payload["message"]["role"] == "user"

    def test_new_node_parent_is_head(self, llm_settings, populated_forest):
        forest, head = populated_forest
        new_head = scaffold.user_turn(llm_settings=llm_settings,
                                      datastore=forest,
                                      head_node_id=head,
                                      user_message_text="Hi there")
        assert forest.get_parent(new_head) == head

    def test_content_preserved(self, llm_settings, populated_forest):
        forest, head = populated_forest
        new_head = scaffold.user_turn(llm_settings=llm_settings,
                                      datastore=forest,
                                      head_node_id=head,
                                      user_message_text="Hi there")
        content = forest.get_payload(new_head)["message"]["content"]
        assert "Hi there" in content


# ---------------------------------------------------------------------------
# ai_turn — simple case (no RAG, no tools)
# ---------------------------------------------------------------------------

class TestAITurnSimple:
    def test_creates_assistant_node_as_child_of_user(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings,
                                       datastore=forest,
                                       head_node_id=head,
                                       user_message_text="Hello")
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: make_invoke_result(content="Hi!"))

        final_head = run_ai_turn(forest, llm_settings, user_head)
        assert forest.get_parent(final_head) == user_head
        payload = forest.get_payload(final_head)
        assert payload["message"]["role"] == "assistant"
        assert "Hi!" in payload["message"]["content"]

    def test_generation_metadata_stored(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings,
                                       datastore=forest,
                                       head_node_id=head,
                                       user_message_text="Hello")
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: make_invoke_result(content="Hi!", n_tokens=42, dt=1.5, model="my-model"))

        final_head = run_ai_turn(forest, llm_settings, user_head)
        meta = forest.get_payload(final_head)["generation_metadata"]
        assert meta["model"] == "my-model"
        assert meta["n_tokens"] == 42
        assert meta["dt"] == 1.5

    def test_llm_callbacks_fire_once_each(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings,
                                       datastore=forest,
                                       head_node_id=head,
                                       user_message_text="Hello")
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: make_invoke_result(content="Hi!"))

        calls = []
        run_ai_turn(forest, llm_settings, user_head,
                    on_llm_start=lambda: calls.append("start"),
                    on_llm_done=lambda nid: calls.append(("done", nid)))
        assert calls[0] == "start"
        assert calls[1][0] == "done"
        assert len(calls) == 2

    def test_no_rag_callbacks_when_docs_query_is_none(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings,
                                       datastore=forest,
                                       head_node_id=head,
                                       user_message_text="Hello")
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: make_invoke_result(content="Hi!"))

        docs_calls = []
        run_ai_turn(forest, llm_settings, user_head,
                    retriever=FakeRetriever(),
                    docs_query=None,
                    on_docs_start=lambda: docs_calls.append("start"),
                    on_docs_done=lambda matches: docs_calls.append(("done", matches)))
        assert docs_calls == []


# ---------------------------------------------------------------------------
# ai_turn — continue mode
# ---------------------------------------------------------------------------

class TestAITurnContinue:
    def test_continue_adds_revision_to_existing_node(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings,
                                       datastore=forest,
                                       head_node_id=head,
                                       user_message_text="Hello")
        # First generation.
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: make_invoke_result(content="Partial response"))
        ai_head = run_ai_turn(forest, llm_settings, user_head)
        initial_revs = forest.get_revisions(ai_head)
        assert len(initial_revs) == 1

        # Continue that same message. Scaffold should add a revision, not a new node.
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: make_invoke_result(content="Partial response continued"))
        continued_head = run_ai_turn(forest, llm_settings, ai_head, continue_=True)
        assert continued_head == ai_head
        revs = forest.get_revisions(ai_head)
        assert len(revs) == 2

    def test_continue_on_non_assistant_raises(self, llm_settings, populated_forest):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings,
                                       datastore=forest,
                                       head_node_id=head,
                                       user_message_text="Hello")
        with pytest.raises(ValueError):
            run_ai_turn(forest, llm_settings, user_head, continue_=True)


# ---------------------------------------------------------------------------
# ai_turn — RAG branches
# ---------------------------------------------------------------------------

class TestAITurnRAG:
    def test_rag_match_invokes_llm_and_stores_retrieval(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings,
                                       datastore=forest,
                                       head_node_id=head,
                                       user_message_text="What is X?")

        retriever = FakeRetriever(results=[sample_rag_match(text="X is foo.", score=0.95)])
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: make_invoke_result(content="X is foo."))

        final_head = run_ai_turn(forest, llm_settings, user_head,
                                 retriever=retriever,
                                 docs_query="What is X?")
        payload = forest.get_payload(final_head)
        assert payload["message"]["role"] == "assistant"
        assert payload["retrieval"]["query"] == "What is X?"
        assert len(payload["retrieval"]["results"]) == 1
        assert retriever.calls[0]["q"] == "What is X?"

    def test_rag_match_appears_in_prompt_seen_by_on_prompt_ready(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings,
                                       datastore=forest,
                                       head_node_id=head,
                                       user_message_text="What is X?")

        retriever = FakeRetriever(results=[sample_rag_match(text="SECRET-MARKER-CONTENT-42")])
        prompt_snapshot = []

        def fake_invoke(**kw):
            if kw.get("on_prompt_ready") is not None:
                kw["on_prompt_ready"](kw["history"])
            return make_invoke_result(content="OK")

        monkeypatch.setattr("raven.librarian.llmclient.invoke", fake_invoke)

        run_ai_turn(forest, llm_settings, user_head,
                    retriever=retriever,
                    docs_query="What is X?",
                    on_prompt_ready=lambda history: prompt_snapshot.append(history))

        assert len(prompt_snapshot) == 1
        all_content = "\n".join(msg["content"] for msg in prompt_snapshot[0])
        assert "SECRET-MARKER-CONTENT-42" in all_content

    def test_rag_no_match_bypass_creates_nomatch_node(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings,
                                       datastore=forest,
                                       head_node_id=head,
                                       user_message_text="What is X?")

        retriever = FakeRetriever(results=[])
        invoke_called = []
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: invoke_called.append(kw) or make_invoke_result())

        nomatch_done_calls = []
        llm_done_calls = []
        final_head = run_ai_turn(forest, llm_settings, user_head,
                                 retriever=retriever,
                                 docs_query="What is X?",
                                 speculate=False,
                                 on_nomatch_done=lambda nid: nomatch_done_calls.append(nid),
                                 on_llm_done=lambda nid: llm_done_calls.append(nid))

        # The LLM must not be called in the no-match bypass path.
        assert invoke_called == []
        # A new assistant "no match" node should be created as a child of user_head.
        assert forest.get_parent(final_head) == user_head
        payload = forest.get_payload(final_head)
        assert payload["message"]["role"] == "assistant"
        # Retrieval metadata recorded with empty results.
        assert payload["retrieval"]["results"] == []
        # on_nomatch_done fires exactly once with the new node id; on_llm_done doesn't fire.
        assert nomatch_done_calls == [final_head]
        assert llm_done_calls == []

    def test_rag_no_match_with_speculate_invokes_llm(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings,
                                       datastore=forest,
                                       head_node_id=head,
                                       user_message_text="What is X?")

        retriever = FakeRetriever(results=[])
        invoke_called = []

        def fake_invoke(**kw):
            invoke_called.append(kw)
            return make_invoke_result(content="I'll speculate here.")

        monkeypatch.setattr("raven.librarian.llmclient.invoke", fake_invoke)

        nomatch_done_calls = []
        final_head = run_ai_turn(forest, llm_settings, user_head,
                                 retriever=retriever,
                                 docs_query="What is X?",
                                 speculate=True,
                                 on_nomatch_done=lambda nid: nomatch_done_calls.append(nid))
        assert len(invoke_called) == 1
        assert nomatch_done_calls == []
        assert "speculate" in forest.get_payload(final_head)["message"]["content"]

    def test_docs_query_without_retriever_logs_warning(self, monkeypatch, caplog, llm_settings, populated_forest):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings,
                                       datastore=forest,
                                       head_node_id=head,
                                       user_message_text="Hello")
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: make_invoke_result(content="Hi!"))

        import logging
        caplog.set_level(logging.WARNING, logger="raven.librarian.scaffold")

        run_ai_turn(forest, llm_settings, user_head,
                    retriever=None,
                    docs_query="something")
        assert any("docs_query" in rec.message or "retriever" in rec.message
                   for rec in caplog.records)


# ---------------------------------------------------------------------------
# ai_turn — tool call loop
# ---------------------------------------------------------------------------

class TestAITurnToolCalls:
    def test_tool_call_loop_creates_three_nodes(self, monkeypatch, llm_settings, populated_forest):
        """First LLM response has tool_calls, tool runs, second LLM response has none."""
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings,
                                       datastore=forest,
                                       head_node_id=head,
                                       user_message_text="Search for raven")

        # First call returns tool_calls; second call returns plain text.
        responses = iter([
            make_invoke_result(content="",
                               tool_calls=[{"type": "function",
                                            "function": {"name": "websearch",
                                                         "arguments": '{"query": "raven"}'},
                                            "id": "call_m357947b",
                                            "index": "0"}]),
            make_invoke_result(content="Here is what I found.")
        ])
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: next(responses))
        monkeypatch.setattr("raven.librarian.llmclient.perform_tool_calls",
                            lambda settings, message, on_call_start, on_call_done:
                                [make_tool_response(content="Search result: raven is a bird.")])

        counters = {"tools_start": 0, "tool_done": 0, "tools_done": 0, "llm_done": 0}
        final_head = run_ai_turn(forest, llm_settings, user_head,
                                 on_tools_start=lambda tcs: counters.update(tools_start=counters["tools_start"] + 1),
                                 on_tool_done=lambda nid: counters.update(tool_done=counters["tool_done"] + 1),
                                 on_tools_done=lambda: counters.update(tools_done=counters["tools_done"] + 1),
                                 on_llm_done=lambda nid: counters.update(llm_done=counters["llm_done"] + 1))

        # Walk back from final_head: final assistant -> tool -> first assistant -> user.
        walk = []
        nid = final_head
        while nid is not None:
            walk.append(forest.get_payload(nid)["message"]["role"])
            nid = forest.get_parent(nid)
        assert walk[0] == "assistant"
        assert walk[1] == "tool"
        assert walk[2] == "assistant"
        assert walk[3] == "user"

        # Callback counts: tools_start fires once (first response had tool_calls),
        # tool_done fires once (one tool response), tools_done fires once,
        # llm_done fires twice (once per LLM response).
        assert counters == {"tools_start": 1, "tool_done": 1, "tools_done": 1, "llm_done": 2}
