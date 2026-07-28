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

from raven.librarian import chattree, chatutil, textfilestore, imagestore, scaffold, sidecarstore  # noqa: E402 -- after importorskip by design


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
                     "content": chatutil.normalize_content(content),  # content-parts list, as invoke returns
                     "tool_calls": tool_calls},
               n_tokens=n_tokens,
               dt=dt,
               model=model,
               interrupted=interrupted)


def make_tool_response(content="tool result",
                       tool_call_id="call_0",
                       function_name="websearch",
                       status="success",
                       dt=0.01):
    """Build a fake tool response record — the shape scaffold reads from `perform_tool_calls`."""
    return env(data={"role": "tool",
                     "content": chatutil.normalize_content(content),  # content-parts list
                     "tool_calls": None},
               status=status,
               tool_call_id=tool_call_id,
               function_name=function_name,
               dt=dt)


def make_denial_response(host="blocked.com", tool_call_id="call_0"):
    """A faked webfetch denial record: carries `tool_metadata={'webfetch_denied_host': host}`,
    the structured marker the GUI override reads to offer "approve this host & retry"."""
    return env(data={"role": "tool",
                     "content": chatutil.normalize_content(f"The host {host} is not on the configured allowlist."),
                     "tool_calls": None},
               status="success",
               tool_call_id=tool_call_id,
               function_name="webfetch",
               dt=0.01,
               tool_metadata={"webfetch_denied_host": host})


def tool_call(name, call_id, index="0"):
    """An OpenAI-format tool call request, as it appears in an assistant message's `tool_calls`."""
    return {"type": "function",
            "function": {"name": name, "arguments": "{}"},
            "id": call_id,
            "index": index}


def roles_up(forest, node_id):
    """List of message roles walking from `node_id` up to the root (node first)."""
    out = []
    while node_id is not None:
        out.append(forest.get_payload(node_id)["message"]["role"])
        node_id = forest.get_parent(node_id)
    return out


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
                docs_enabled=True,
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
                            docs_enabled=docs_enabled,
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
        content = chatutil.content_to_text(forest.get_payload(new_head)["message"]["content"])
        assert "Hi there" in content


class TestUserTurnStagedImages:
    """user_turn with attached images: sidecars stored, image parts appended, provenance recorded."""

    @staticmethod
    def _png_bytes(width, height, color=(30, 160, 90)):
        import io
        from PIL import Image  # deferred; the heavy stack is present (module-level importorskip)
        buffer = io.BytesIO()
        Image.new("RGB", (width, height), color).save(buffer, format="PNG")
        return buffer.getvalue()

    def _forest(self, tmp_path, llm_settings):
        forest = chattree.PersistentForest(tmp_path / "chat.json", autosave=False,
                                           sidecar_extractor=imagestore.sidecar_refs_in_payload)
        greeting = chatutil.factory_reset_datastore(forest, llm_settings)
        return forest, greeting

    def test_image_part_appended_and_sidecar_recorded(self, tmp_path, llm_settings):
        forest, head = self._forest(tmp_path, llm_settings)
        staged = env(raw=self._png_bytes(48, 32),
                     provenance_url="file:///tmp/pic.png",
                     provenance_source="user_attachment")
        new_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                      head_node_id=head, user_message_text="look at this",
                                      staged_images=[staged])
        payload = forest.get_payload(new_head)
        content = payload["message"]["content"]
        assert "look at this" in chatutil.content_to_text(content)  # text part preserved

        image_parts = [part for part in content if part.get("type") == "image_url"]
        assert len(image_parts) == 1  # image part appended after the text part
        url = image_parts[0]["image_url"]["url"]
        assert url.startswith(sidecarstore.SIDECAR_SCHEME)
        filename = url[len(sidecarstore.SIDECAR_SCHEME):]

        assert len(forest.read_sidecar(filename)) > 0  # sidecar file was written
        sidecars = payload["general_metadata"]["sidecars"]
        assert sidecars[filename]["url"] == "file:///tmp/pic.png"  # provenance recorded
        assert sidecars[filename]["source"] == "user_attachment"

    def test_text_only_adds_no_sidecars_key(self, tmp_path, llm_settings):
        forest, head = self._forest(tmp_path, llm_settings)
        new_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                      head_node_id=head, user_message_text="plain text",
                                      staged_images=None)
        payload = forest.get_payload(new_head)
        assert "sidecars" not in payload["general_metadata"]
        assert all(part.get("type") == "text" for part in payload["message"]["content"])


class TestUserTurnStagedFiles:
    """user_turn with attached documents: sidecars stored, text_file parts appended, provenance recorded."""

    def _forest(self, tmp_path, llm_settings):
        forest = chattree.PersistentForest(tmp_path / "chat.json", autosave=False,
                                           sidecar_extractor=textfilestore.sidecar_refs_in_payload)
        greeting = chatutil.factory_reset_datastore(forest, llm_settings)
        return forest, greeting

    def test_file_part_appended_and_sidecar_recorded(self, tmp_path, llm_settings):
        forest, head = self._forest(tmp_path, llm_settings)
        staged = env(raw=b"the spec body text",
                     name="spec.txt",
                     provenance_url="file:///tmp/spec.txt",
                     provenance_source="user_attachment")
        new_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                      head_node_id=head, user_message_text="review the spec",
                                      staged_files=[staged])
        payload = forest.get_payload(new_head)
        content = payload["message"]["content"]
        assert "review the spec" in chatutil.content_to_text(content)  # text part preserved
        assert "the spec body text" not in chatutil.content_to_text(content)  # document not in the message's own text

        file_parts = [part for part in content if part.get("type") == "text_file"]
        assert len(file_parts) == 1  # text_file part appended after the text part
        assert file_parts[0]["text_file"]["name"] == "spec.txt"
        url = file_parts[0]["text_file"]["url"]
        assert url.startswith(sidecarstore.SIDECAR_SCHEME)
        filename = url[len(sidecarstore.SIDECAR_SCHEME):]

        assert forest.read_sidecar(filename) == b"the spec body text"  # sidecar stored verbatim
        sidecars = payload["general_metadata"]["sidecars"]
        assert sidecars[filename]["url"] == "file:///tmp/spec.txt"  # provenance recorded
        assert sidecars[filename]["source"] == "user_attachment"
        assert sidecars[filename]["name"] == "spec.txt"

    def test_images_and_files_share_the_sidecars_metadata(self, tmp_path, llm_settings):
        # A message carrying both an image and a document records both under general_metadata["sidecars"].
        forest = chattree.PersistentForest(
            tmp_path / "chat.json", autosave=False,
            sidecar_extractor=lambda p: imagestore.sidecar_refs_in_payload(p) | textfilestore.sidecar_refs_in_payload(p))
        head = chatutil.factory_reset_datastore(forest, llm_settings)
        img = env(raw=TestUserTurnStagedImages._png_bytes(16, 16),
                  provenance_url="file:///tmp/pic.png", provenance_source="user_attachment")
        doc = env(raw=b"doc body", name="d.txt",
                  provenance_url="file:///tmp/d.txt", provenance_source="user_attachment")
        new_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                      head_node_id=head, user_message_text="both",
                                      staged_images=[img], staged_files=[doc])
        payload = forest.get_payload(new_head)
        content = payload["message"]["content"]
        assert sum(1 for p in content if p.get("type") == "image_url") == 1
        assert sum(1 for p in content if p.get("type") == "text_file") == 1
        assert len(payload["general_metadata"]["sidecars"]) == 2  # both recorded


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
        assert "Hi!" in chatutil.content_to_text(payload["message"]["content"])

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
        all_content = "\n".join(chatutil.content_to_text(msg["content"]) for msg in prompt_snapshot[0])
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
        assert "speculate" in chatutil.content_to_text(forest.get_payload(final_head)["message"]["content"])

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

    def test_tool_context_bound_with_user_typed_url(self, monkeypatch, llm_settings, populated_forest):
        """The agent loop binds `dyn.tool_context` so a tool entrypoint can see the hosts the user
        auto-allowed by typing a URL this turn. Asserts the wiring between `user_turn` (the typed URL),
        `compute_auto_allowed_hosts`, and the `dyn.let` around the tool dispatch.
        """
        from unpythonic import dyn  # noqa: PLC0415 -- local to this test

        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings,
                                       datastore=forest,
                                       head_node_id=head,
                                       user_message_text="please look at https://example.com/article")

        responses = iter([
            make_invoke_result(content="",
                               tool_calls=[{"type": "function",
                                            "function": {"name": "webfetch",
                                                         "arguments": '{"url": "https://example.com/article"}'},
                                            "id": "call_abc",
                                            "index": "0"}]),
            make_invoke_result(content="Done."),
        ])
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: next(responses))

        captured = {}
        def capture_perform(settings, message, on_call_start, on_call_done):
            # The binding under test is live here; read what an entrypoint would read.
            captured["hosts"] = getattr(dyn.tool_context, "webfetch_allowed_hosts", None)
            return [make_tool_response(content="fetched content")]
        monkeypatch.setattr("raven.librarian.llmclient.perform_tool_calls", capture_perform)

        run_ai_turn(forest, llm_settings, user_head)

        assert captured["hosts"] == frozenset({"example.com"})


# ---------------------------------------------------------------------------
# retry_tool_calls — the "approve denied host & retry" override
# ---------------------------------------------------------------------------

def run_retry(forest, llm_settings, tool_node_id, *,
              retriever=None,
              tools_enabled=True,
              docs_enabled=True,
              speculate=True,
              markup=None,
              docs_num_results=None,
              **callbacks):
    """Call `scaffold.retry_tool_calls` with `None` defaults for unspecified callbacks."""
    cb_kwargs = {name: callbacks.get(name, None) for name in _AI_TURN_CALLBACKS}
    return scaffold.retry_tool_calls(llm_settings=llm_settings,
                                     datastore=forest,
                                     retriever=retriever,
                                     tool_node_id=tool_node_id,
                                     tools_enabled=tools_enabled,
                                     docs_enabled=docs_enabled,
                                     speculate=speculate,
                                     markup=markup,
                                     docs_num_results=docs_num_results,
                                     **cb_kwargs)


class TestRetryToolCalls:
    def _make_denied_state(self, monkeypatch, llm_settings, forest, head, *, tool_calls, records):
        """Drive one `ai_turn` that issues `tool_calls`, runs `records`, then a giving-up reply.

        Returns `(first_head, tool_done_nodes)` — the giving-up assistant and the tool-result node ids
        (in creation order), so a test can pick the denied one.
        """
        user_head = scaffold.user_turn(llm_settings=llm_settings,
                                       datastore=forest,
                                       head_node_id=head,
                                       user_message_text="please fetch some pages")
        responses = iter([make_invoke_result(content="", tool_calls=tool_calls),
                          make_invoke_result(content="Sorry, I could not reach that.")])
        monkeypatch.setattr("raven.librarian.llmclient.invoke", lambda **kw: next(responses))
        monkeypatch.setattr("raven.librarian.llmclient.perform_tool_calls",
                            lambda settings, message, on_call_start, on_call_done: list(records))
        tool_done_nodes = []
        first_head = run_ai_turn(forest, llm_settings, user_head,
                                 on_tool_done=lambda nid: tool_done_nodes.append(nid))
        return first_head, tool_done_nodes

    def test_single_denied_fetch_reruns_only_that_call_on_a_branch(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        _first, tool_nodes = self._make_denied_state(
            monkeypatch, llm_settings, forest, head,
            tool_calls=[tool_call("webfetch", "call_0")],
            records=[make_denial_response(host="blocked.com", tool_call_id="call_0")])
        denied_node = tool_nodes[0]
        assert forest.get_payload(denied_node)["generation_metadata"]["webfetch_denied_host"] == "blocked.com"

        # Approve + retry: the re-run now succeeds. Capture what perform_tool_calls is asked to run.
        rerun_messages = []
        def capture_perform(settings, message, on_call_start, on_call_done):
            rerun_messages.append(message)
            return [make_tool_response(content="FETCHED OK", tool_call_id="call_0", function_name="webfetch")]
        monkeypatch.setattr("raven.librarian.llmclient.perform_tool_calls", capture_perform)
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: make_invoke_result(content="Here is the page content."))

        new_head = run_retry(forest, llm_settings, denied_node)

        # Exactly the denied call was re-run (one call in the synthetic message).
        assert len(rerun_messages) == 1
        assert [tc["id"] for tc in rerun_messages[0]["tool_calls"]] == ["call_0"]

        # New branch: continuation assistant -> tool(webfetch success) -> first (tool-calling) assistant.
        assert roles_up(forest, new_head)[:3] == ["assistant", "tool", "assistant"]
        new_tool_node = forest.get_parent(new_head)
        assert "FETCHED OK" in chatutil.content_to_text(forest.get_payload(new_tool_node)["message"]["content"])
        assert "webfetch_denied_host" not in forest.get_payload(new_tool_node)["generation_metadata"]

        # It is a real branch: the new tool node and the old denied node share a parent (the assistant)...
        assert forest.get_parent(new_tool_node) == forest.get_parent(denied_node)
        assert new_tool_node != denied_node
        # ...and the old denial is preserved untouched.
        assert forest.get_payload(denied_node)["generation_metadata"]["webfetch_denied_host"] == "blocked.com"

    def test_websearch_prefix_is_shared_not_rerun(self, monkeypatch, llm_settings, populated_forest):
        """Assistant issues [websearch, webfetch] in one message; webfetch is denied. The retry must
        re-run ONLY webfetch and reuse the existing websearch node (no re-query — reboot-safe)."""
        forest, head = populated_forest
        _first, tool_nodes = self._make_denied_state(
            monkeypatch, llm_settings, forest, head,
            tool_calls=[tool_call("websearch", "call_0"), tool_call("webfetch", "call_1")],
            records=[make_tool_response(content="websearch result text", tool_call_id="call_0", function_name="websearch"),
                     make_denial_response(host="blocked.com", tool_call_id="call_1")])
        websearch_node, denied_node = tool_nodes  # creation order: websearch, then denied webfetch
        assert forest.get_parent(denied_node) == websearch_node  # chained

        rerun_messages = []
        def capture_perform(settings, message, on_call_start, on_call_done):
            rerun_messages.append(message)
            return [make_tool_response(content="FETCHED OK", tool_call_id="call_1", function_name="webfetch")]
        monkeypatch.setattr("raven.librarian.llmclient.perform_tool_calls", capture_perform)
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: make_invoke_result(content="Combined answer."))

        new_head = run_retry(forest, llm_settings, denied_node)

        # Only webfetch (call_1) re-run; websearch (call_0) was NOT.
        assert len(rerun_messages) == 1
        assert [tc["id"] for tc in rerun_messages[0]["tool_calls"]] == ["call_1"]

        # The new webfetch node branches off the SAME (shared) websearch node — not a copy.
        new_tool_node = forest.get_parent(new_head)
        assert forest.get_parent(new_tool_node) == websearch_node
        assert "FETCHED OK" in chatutil.content_to_text(forest.get_payload(new_tool_node)["message"]["content"])

    def test_suffix_tool_results_are_copied_verbatim(self, monkeypatch, llm_settings, populated_forest):
        """Assistant issues [webfetch, websearch] in one message; webfetch (first) is denied. The retry
        re-runs webfetch and COPIES the trailing websearch result verbatim (not re-run) onto the branch."""
        forest, head = populated_forest
        _first, tool_nodes = self._make_denied_state(
            monkeypatch, llm_settings, forest, head,
            tool_calls=[tool_call("webfetch", "call_0"), tool_call("websearch", "call_1")],
            records=[make_denial_response(host="blocked.com", tool_call_id="call_0"),
                     make_tool_response(content="ORIGINAL websearch result", tool_call_id="call_1", function_name="websearch")])
        denied_node, websearch_node = tool_nodes  # creation order: denied webfetch, then websearch

        rerun_messages = []
        def capture_perform(settings, message, on_call_start, on_call_done):
            rerun_messages.append(message)
            return [make_tool_response(content="FETCHED OK", tool_call_id="call_0", function_name="webfetch")]
        monkeypatch.setattr("raven.librarian.llmclient.perform_tool_calls", capture_perform)
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: make_invoke_result(content="Combined answer."))

        new_head = run_retry(forest, llm_settings, denied_node)

        # Only the denied webfetch (call_0) re-run; the websearch was copied, never re-run.
        assert len(rerun_messages) == 1
        assert [tc["id"] for tc in rerun_messages[0]["tool_calls"]] == ["call_0"]

        # New branch: continuation -> tool(websearch copy) -> tool(webfetch success) -> assistant.
        assert roles_up(forest, new_head)[:4] == ["assistant", "tool", "tool", "assistant"]
        websearch_copy = forest.get_parent(new_head)
        assert forest.get_payload(websearch_copy)["generation_metadata"]["function_name"] == "websearch"
        assert chatutil.content_to_text(forest.get_payload(websearch_copy)["message"]["content"]) == "ORIGINAL websearch result"
        assert websearch_copy != websearch_node  # a copy, distinct node id (not a reparent)

    def test_non_tool_node_raises(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                       head_node_id=head, user_message_text="Hello")
        with pytest.raises(ValueError):
            run_retry(forest, llm_settings, user_head)  # a user node, not a tool node


# ---------------------------------------------------------------------------
# Temporary context injects
# ---------------------------------------------------------------------------

def assert_at_most_one_leading_system_message(history):
    """Assert the strictest chat templates would accept `history`'s role sequence.

    Qwen3.5's guard is `{%- if message.role == "system" %}{%- if not loop.first %}` -> `raise_exception`,
    i.e. exactly one system message, at index 0. Note this is a *count* rule wearing the clothes of a
    position rule: its error text says "System message must be at the beginning", but a second system
    message fails even when it sits ahead of the conversation.
    """
    roles = [message["role"] for message in history]
    assert "system" not in roles[1:], roles


def make_conversation(llm_settings):
    """A minimal system -> assistant -> user history, the shape a chat has when the AI's turn starts."""
    return [chatutil.create_chat_message(llm_settings=llm_settings, role="system", text="You are a helpful assistant."),
            chatutil.create_chat_message(llm_settings=llm_settings, role="assistant", text="How can I help you today?"),
            chatutil.create_chat_message(llm_settings=llm_settings, role="user", text="What is X?")]


class TestPerformInjects:
    """`_perform_injects` builds the temporary history handed to the LLM.

    Two families of invariant are pinned here.

    The first is a chat-template contract, not a Raven preference: several templates require every
    "system" message to precede the first user/assistant turn, and enforce it with a hard
    `raise_exception` rather than by ignoring the stray message. Qwen3.5's template does; Qwen3.6's
    dropped the guard. A violation therefore fails the whole request, and does so only on the strict
    models — invisible while developing against a permissive one.

    The second is the shape and placement of the injected material, each item of which was chosen by
    measurement rather than by taste (`briefs/context-inject-shape-measurements.md`). Nothing about
    these shapes is self-evidently right, so they are worth holding still: a plausible-looking
    simplification here costs an hour of live probing to catch.
    """

    def test_injects_add_no_system_message(self, llm_settings):
        history = make_conversation(llm_settings)
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  speculate=False, docs_query=None, docs_matches=[])
        assert_at_most_one_leading_system_message(history)

    def test_rag_matches_add_no_system_message(self, llm_settings):
        # Each match used to go in as its own system message at index 1, which failed every AI turn on
        # Qwen3.5 — several system messages are rejected even though all of them precede the conversation.
        history = make_conversation(llm_settings)
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  speculate=False, docs_query="what is X?",
                                  docs_matches=[sample_rag_match(document_id="a.txt"),
                                                sample_rag_match(document_id="b.txt")])
        assert_at_most_one_leading_system_message(history)

    def test_rag_matches_ride_in_one_tool_message_in_corpus_order(self, llm_settings):
        # A `tool` message answers exactly one `tool_call_id`, so all matches share one message rather
        # than getting one each — the one-per-match form shares an id across messages, which Gemma4-E4B
        # reads as nothing at all.
        history = make_conversation(llm_settings)
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  speculate=False, docs_query="what is X?",
                                  docs_matches=[sample_rag_match(document_id="a.txt"),
                                                sample_rag_match(document_id="b.txt")])
        docs_messages = [message for message in history
                         if message["role"] == "tool" and "Knowledge-base match" in chatutil.content_to_text(message["content"])]
        assert len(docs_messages) == 1
        text = chatutil.content_to_text(docs_messages[0]["content"])
        assert text.index("a.txt") < text.index("b.txt")  # corpus order preserved

    def test_injected_tool_results_answer_a_tool_call(self, llm_settings):
        # The synthetic assistant call is load-bearing, not decoration: handed a bare `tool` message with
        # no call to answer, Gemma 4 ignores the material and confabulates a confident wrong answer.
        history = make_conversation(llm_settings)
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  speculate=False, docs_query="what is X?",
                                  docs_matches=[sample_rag_match()])
        requested_call_ids = {call["id"]
                              for message in history
                              for call in message.get("tool_calls", [])}
        answered_call_ids = {message["tool_call_id"] for message in history if message["role"] == "tool"}
        assert answered_call_ids  # guard: the rest of this test is vacuous if nothing was injected
        assert answered_call_ids == requested_call_ids

    def test_injects_precede_the_users_latest_message(self, llm_settings):
        # With a tool result as the last message, the history reads as a paused agent loop, and Qwen 3.6
        # sometimes replies by requesting *another* search instead of answering. Leaving the user's
        # question last is what keeps the model talking to the user.
        history = make_conversation(llm_settings)
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  speculate=False, docs_query="what is X?",
                                  docs_matches=[sample_rag_match()])
        assert history[-1]["role"] == "user"
        assert chatutil.content_to_text(history[-1]["content"]).endswith("What is X?")

    def test_injects_do_not_disturb_a_message_being_continued(self, llm_settings):
        # When continuing, the history must look as it did when generation was interrupted: the AI's
        # incomplete message stays last, with the injects ahead of the user turn it is answering.
        history = make_conversation(llm_settings)
        history.append(chatutil.create_chat_message(llm_settings=llm_settings, role="assistant", text="X is"))
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  speculate=False, docs_query=None, docs_matches=[])
        assert_at_most_one_leading_system_message(history)
        assert history[-1]["role"] == "assistant"
        assert history[-2]["role"] == "user"

    def test_reminders_and_date_go_into_the_system_message(self, llm_settings):
        # Instruction-like injects want the leading system block: measured the cheapest placement in
        # deliberation tokens, and the only one that never provoked the model into remarking on them.
        history = make_conversation(llm_settings)
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  speculate=False, docs_query="what is X?",
                                  docs_matches=[sample_rag_match()])
        system_text = chatutil.content_to_text(history[0]["content"])
        assert "You are a helpful assistant." in system_text  # the original system prompt survives
        assert "Today is" in system_text
        assert "structured report" in system_text
        assert "Base claims about the provided documents" in system_text

    def test_system_message_is_not_mutated_in_place(self, llm_settings):
        # `chatutil.linearize_chat` hands out the datastore's own message dicts. Editing one here would
        # write the injects into the stored system prompt, permanently, once per turn.
        history = make_conversation(llm_settings)
        stored_system_message = history[0]
        stored_text = chatutil.content_to_text(stored_system_message["content"])
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  speculate=False, docs_query=None, docs_matches=[])
        assert chatutil.content_to_text(stored_system_message["content"]) == stored_text

    def test_context_only_reminder_is_skipped_without_context(self, llm_settings):
        # Asking a model to stick to documents that were never provided is a contradiction it will
        # dutifully try to resolve — up to 37x the deliberation, and on one model, never terminating.
        history = make_conversation(llm_settings)
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  speculate=False, docs_query=None, docs_matches=[])
        assert "Base claims about the provided documents" not in chatutil.content_to_text(history[0]["content"])

    def test_context_only_reminder_counts_an_attachment_as_context(self, llm_settings):
        # "Context" is broader than docs matches: an attached document or image is material to ground in,
        # even on a turn where the document database returned nothing.
        history = make_conversation(llm_settings)
        history[-1]["content"].append(chatutil.text_file_content_part(url="sidecar:deadbeef.pdf", name="paper.pdf"))
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  speculate=False, docs_query=None, docs_matches=[])
        assert "Base claims about the provided documents" in chatutil.content_to_text(history[0]["content"])

    def test_context_only_reminder_ignores_a_stale_tool_result(self, llm_settings):
        # A tool result answers one specific earlier question. Counting it as context forever would leave
        # the reminder switched on for the rest of the conversation, however unrelated the later turns get.
        history = [chatutil.create_chat_message(llm_settings=llm_settings, role="system", text="You are a helpful assistant."),
                   chatutil.create_chat_message(llm_settings=llm_settings, role="user", text="What is the weather in Tampere?"),
                   chatutil.create_chat_message(llm_settings=llm_settings, role="tool", text="Tampere: 17 C, cloudy."),
                   chatutil.create_chat_message(llm_settings=llm_settings, role="assistant", text="It is 17 C and cloudy."),
                   chatutil.create_chat_message(llm_settings=llm_settings, role="user", text="What is the baseline drift of the Kelvin-7 microarray?")]
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  speculate=False, docs_query=None, docs_matches=[])
        assert "Base claims about the provided documents" not in chatutil.content_to_text(history[0]["content"])

    def test_context_only_reminder_counts_this_turns_tool_result(self, llm_settings):
        # Mid-agent-loop, on the other hand, the search we just ran is exactly the material to ground in.
        history = make_conversation(llm_settings)
        history.append(chatutil.create_chat_message(llm_settings=llm_settings, role="tool", text="Search result: X is a variable."))
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  speculate=False, docs_query=None, docs_matches=[])
        assert "Base claims about the provided documents" in chatutil.content_to_text(history[0]["content"])

    def test_speculation_on_sends_no_context_only_reminder(self, llm_settings):
        history = make_conversation(llm_settings)
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  speculate=True, docs_query="what is X?",
                                  docs_matches=[sample_rag_match()])
        assert "Base claims about the provided documents" not in chatutil.content_to_text(history[0]["content"])

    def test_injects_carry_no_persona_prefix(self, llm_settings):
        # The inject text is bracketed and self-labelling; prefixing the speaker's persona to it
        # ("User: [System information: ...]") would read as the user narrating a system notice.
        history = make_conversation(llm_settings)
        before = len(history)
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  speculate=False, docs_query=None, docs_matches=[])

        injected = [message for message in history[:before + 2] if message["role"] == "tool"]
        assert injected  # guard: the rest of this test is vacuous if nothing was injected
        for message in injected:
            assert chatutil.content_to_text(message["content"]).startswith("[System information:")


# ---------------------------------------------------------------------------
# Grounding accumulation (the per-turn tool context)
# ---------------------------------------------------------------------------

def make_tool_record(text="Some result.", status="success", tool_metadata=None):
    record = env(data=chatutil.create_message_from_parts("tool", [chatutil.text_content_part(text)]),
                 status=status)
    if tool_metadata is not None:
        record.tool_metadata = tool_metadata
    return record


class TestGroundingAccumulation:
    """Whether a turn has material to answer from is declared by tools, not inferred from message shape.

    The reminder to base claims on the provided context is only sound when there *is* context; sent with
    nothing to ground in, it measured 5-37x the deliberation of sending nothing (brief 08, Q4). So a tool
    result that is textually present but materially empty must not read as grounding.
    """

    def test_successful_nonempty_result_grounds_by_default(self):
        tool_context = scaffold._make_tool_context(retriever=None)
        scaffold._record_grounding(tool_context, make_tool_record("Kelvin-7 drifts 0.3 K/h."))
        assert tool_context.grounded

    def test_failed_call_does_not_ground(self):
        tool_context = scaffold._make_tool_context(retriever=None)
        scaffold._record_grounding(tool_context, make_tool_record("Tool call failed.", status="error"))
        assert not tool_context.grounded

    def test_empty_result_does_not_ground(self):
        # A search that found nothing is the case the whole mechanism exists for: it is a perfectly
        # well-formed tool message carrying no material at all.
        tool_context = scaffold._make_tool_context(retriever=None)
        scaffold._record_grounding(tool_context, make_tool_record("   \n  "))
        assert not tool_context.grounded

    def test_declaration_overrides_the_default(self):
        # webfetch's allowlist refusal is the live example: non-empty, successful, grounds nothing.
        tool_context = scaffold._make_tool_context(retriever=None)
        scaffold._record_grounding(tool_context,
                                   make_tool_record("The host example.com is not on the configured allowlist.",
                                                    tool_metadata={"grounding": False}))
        assert not tool_context.grounded

    def test_declaration_can_ground_an_empty_looking_result(self):
        tool_context = scaffold._make_tool_context(retriever=None)
        scaffold._record_grounding(tool_context, make_tool_record("", tool_metadata={"grounding": True}))
        assert tool_context.grounded

    def test_grounding_is_monotonic_within_a_turn(self):
        # A tool call in round 1 must still count in round 3 - which is why the context is per-turn and
        # not per-round. A later empty search cannot un-ground what an earlier one found.
        tool_context = scaffold._make_tool_context(retriever=None)
        scaffold._record_grounding(tool_context, make_tool_record("Found it."))
        scaffold._record_grounding(tool_context, make_tool_record(""))
        scaffold._record_grounding(tool_context, make_tool_record("failed", status="error"))
        assert tool_context.grounded


# ---------------------------------------------------------------------------
# Per-turn tool availability (the document tools follow the document database)
# ---------------------------------------------------------------------------

class TestDocumentToolGating:
    """`docs_enabled` gates the document *tools*; `docs_query` gates the automatic *search*.

    They collapsed into one switch while the automatic search was the only way to reach the documents.
    They cannot stay collapsed: a continuation turn runs no automatic search but must keep offering the
    tools, because a tool that appears and vanishes between rounds of one agent loop is a shape models
    read as noise.
    """

    def _capture_tool_names(self, monkeypatch):
        seen = {}

        def fake_invoke(**kw):
            seen["tool_names"] = kw.get("tool_names")
            return make_invoke_result(content="OK")

        monkeypatch.setattr("raven.librarian.llmclient.invoke", fake_invoke)
        return seen

    def test_documents_in_play_offers_every_tool(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                       head_node_id=head, user_message_text="What is X?")
        seen = self._capture_tool_names(monkeypatch)
        run_ai_turn(forest, llm_settings, user_head,
                    retriever=FakeRetriever(results=[sample_rag_match()]),
                    docs_enabled=True, docs_query="What is X?")
        assert seen["tool_names"] is None  # `None` is the permissive value: offer all registered tools

    def test_docs_disabled_withdraws_only_the_document_tools(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                       head_node_id=head, user_message_text="What is 2+2?")
        seen = self._capture_tool_names(monkeypatch)
        run_ai_turn(forest, llm_settings, user_head,
                    retriever=FakeRetriever(results=[sample_rag_match()]),
                    docs_enabled=False, docs_query="What is 2+2?")
        assert "search_documents" not in seen["tool_names"]
        assert "websearch" in seen["tool_names"]  # unrelated tools are untouched

    def test_no_retriever_withdraws_the_document_tools(self, monkeypatch, llm_settings, populated_forest):
        # An app with no document database at all must not advertise tools that search one.
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                       head_node_id=head, user_message_text="Hi")
        seen = self._capture_tool_names(monkeypatch)
        run_ai_turn(forest, llm_settings, user_head, retriever=None, docs_enabled=True)
        assert "search_documents" not in seen["tool_names"]

    def test_docs_disabled_runs_no_automatic_search(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                       head_node_id=head, user_message_text="What is X?")
        retriever = FakeRetriever(results=[sample_rag_match()])
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: make_invoke_result(content="OK"))
        run_ai_turn(forest, llm_settings, user_head,
                    retriever=retriever, docs_enabled=False, docs_query="What is X?")
        assert retriever.calls == []

    def test_tools_offered_without_an_automatic_search(self, monkeypatch, llm_settings, populated_forest):
        # The continuation shape: no `docs_query`, but the documents are still in play, so the model may
        # still search for itself.
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                       head_node_id=head, user_message_text="Go on")
        seen = self._capture_tool_names(monkeypatch)
        run_ai_turn(forest, llm_settings, user_head,
                    retriever=FakeRetriever(), docs_enabled=True, docs_query=None)
        assert seen["tool_names"] is None
