"""Unit tests for raven.librarian.scaffold (user_turn, ai_turn)."""

import threading

import pytest

# `scaffold` transitively imports `llmclient` → `raven.client.api`, whose module-level imports include
# `spacy` and (via the vendored Kokoro streaming writer) `av`. The CI test job installs a hand-picked
# dependency subset that has neither, so the import fails there and this whole module is skipped.
# Mirrors `test_hybridir.py` (importorskip on chromadb/bm25s).
#
# Name those two specifically rather than "the heavy stack": CI *does* install qoi, and torch (the CPU
# wheel, which imports fine — only CUDA is absent). Listing them as reasons sent at least one reader
# chasing dependencies that were already there.
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


def payloads_up(forest, node_id):
    """List of node payloads walking from `node_id` up to the root (node first)."""
    out = []
    while node_id is not None:
        out.append(forest.get_payload(node_id))
        node_id = forest.get_parent(node_id)
    return out


def roles_up(forest, node_id):
    """List of message roles walking from `node_id` up to the root (node first)."""
    return [payload["message"]["role"] for payload in payloads_up(forest, node_id)]


class FakeRetriever:
    """Minimal stand-in for `raven.librarian.hybridir.HybridIR`.

    Implements `.query(q, k, max_span_length=None, return_extra_info=False)`, plus the `documents` mapping
    and the `datastore_lock` guarding it — between them, the whole retriever surface the librarian touches.

    The query signature takes `**kwargs` for the tuning parameters rather than naming each: a double that
    names them has to be edited every time one is added, and the failure is a `TypeError` in a dozen
    unrelated tests that says nothing about what the change was. The ones this file asserts on are named
    explicitly; the rest are recorded and ignored.
    """
    def __init__(self, results=None, documents=None):
        self.results = list(results) if results is not None else []
        self.documents = {document_id: {"text": text} for document_id, text in (documents or {}).items()}
        self.datastore_lock = threading.RLock()
        self.calls = []

    def query(self, q, k=10, return_extra_info=False, **kwargs):
        self.calls.append({"q": q, "k": k, "return_extra_info": return_extra_info, **kwargs})
        return list(self.results)



def grounding_context(grounded=False):
    """A tool context for direct `_perform_injects` tests. `grounded` is what a tool or the auto-search
    would have declared during the turn; see `scaffold._record_grounding`."""
    tool_context = scaffold._make_tool_context(llm_settings=None, retriever=None)
    tool_context.grounded = grounded
    return tool_context


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
                      "on_tools_start",
                      "on_call_lowlevel_start", "on_call_lowlevel_done",
                      "on_tool_done", "on_tools_done")


def run_ai_turn(forest, llm_settings, head, *,
                retriever=None,
                internet_enabled=True,
                continue_=False,
                docs_enabled=True,
                docs_query=None,
                docs_num_results=None,
                markup=None,
                **callbacks):
    """Call `scaffold.ai_turn` with `None` defaults for unspecified callbacks."""
    cb_kwargs = {name: callbacks.get(name, None) for name in _AI_TURN_CALLBACKS}
    return scaffold.ai_turn(llm_settings=llm_settings,
                            datastore=forest,
                            retriever=retriever,
                            head_node_id=head,
                            internet_enabled=internet_enabled,
                            continue_=continue_,
                            docs_enabled=docs_enabled,
                            docs_query=docs_query,
                            docs_num_results=docs_num_results,
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

    def test_rag_no_match_still_lets_the_model_answer(self, monkeypatch, llm_settings, populated_forest):
        # An empty search used to end the turn before the LLM ran. It could not tell a question about the
        # documents from a general-knowledge aside, so in the default configuration (docs on, speculation
        # off) it answered "what is 2+2?" with "No matches in document database." The reply now goes
        # through, carrying a record of what it did *not* have to stand on.
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings,
                                       datastore=forest,
                                       head_node_id=head,
                                       user_message_text="What is 2+2?")

        invoke_called = []
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: invoke_called.append(kw) or make_invoke_result(content="Four."))

        llm_done_calls = []
        final_head = run_ai_turn(forest, llm_settings, user_head,
                                 retriever=FakeRetriever(results=[]),
                                 docs_query="What is 2+2?",
                                 on_llm_done=lambda nid: llm_done_calls.append(nid))

        assert len(invoke_called) == 1
        payload = forest.get_payload(final_head)
        assert payload["message"]["role"] == "assistant"
        assert chatutil.content_to_text(payload["message"]["content"]).endswith("Four.")
        assert payload["generation_metadata"]["grounded"] is False
        assert payload["retrieval"]["results"] == []  # still recorded, for the citation mechanism
        assert llm_done_calls == [final_head]

    def test_a_grounded_reply_is_recorded_as_grounded(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings,
                                       datastore=forest,
                                       head_node_id=head,
                                       user_message_text="What is X?")
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: make_invoke_result(content="X is foo."))
        final_head = run_ai_turn(forest, llm_settings, user_head,
                                 retriever=FakeRetriever(results=[sample_rag_match()]),
                                 docs_query="What is X?")
        assert forest.get_payload(final_head)["generation_metadata"]["grounded"] is True

    def test_documents_off_records_no_grounding_verdict(self, monkeypatch, llm_settings, populated_forest):
        """With documents off there is nothing worth saying: "no sources retrieved" would only report the
        switch the user just set, and would be indistinguishable from the case that *is* worth reporting -
        documents on, nothing came back. Absent beats a third state."""
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings,
                                       datastore=forest,
                                       head_node_id=head,
                                       user_message_text="What is X?")
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: make_invoke_result(content="X is foo."))
        final_head = run_ai_turn(forest, llm_settings, user_head,
                                 retriever=FakeRetriever(results=[]),
                                 docs_enabled=False)
        assert "grounded" not in forest.get_payload(final_head)["generation_metadata"]

    def test_an_attachment_grounds_a_reply_even_with_documents_off(self, monkeypatch, llm_settings, tmp_path):
        """The documents switch governs the document database, not the whole notion of having something to
        stand on. A user who attached a document and turned the database off has still supplied material,
        so the verdict is recorded — and it is `True`.

        Attaches the way the apps do, through `staged_files`, which needs a `PersistentForest` to hold the
        sidecar bytes. Building the node bare and adding the attachment as a second revision would test the
        same predicate while documenting a shape that never occurs: an attachment arrives with the message
        it was attached to, and revisions are for edits.
        """
        forest = chattree.PersistentForest(
            tmp_path / "chat.json", autosave=False,
            sidecar_extractor=lambda p: imagestore.sidecar_refs_in_payload(p) | textfilestore.sidecar_refs_in_payload(p))
        head = chatutil.factory_reset_datastore(forest, llm_settings)
        doc = env(raw=b"the body of the attached document", name="paper.txt",
                  provenance_url="file:///tmp/paper.txt", provenance_source="user_attachment")
        user_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                       head_node_id=head, user_message_text="What does this say?",
                                       staged_files=[doc])
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: make_invoke_result(content="It says foo."))
        final_head = run_ai_turn(forest, llm_settings, user_head,
                                 retriever=FakeRetriever(results=[]),
                                 docs_enabled=False)
        assert forest.get_payload(final_head)["generation_metadata"]["grounded"] is True

    def test_rag_no_match_still_invokes_the_llm(self, monkeypatch, llm_settings, populated_forest):
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

        final_head = run_ai_turn(forest, llm_settings, user_head,
                                 retriever=retriever,
                                 docs_query="What is X?")
        assert len(invoke_called) == 1
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
                            lambda settings, message, on_call_start, on_call_done, **kw:
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
        def capture_perform(settings, message, on_call_start, on_call_done, **kw):
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
              internet_enabled=True,
              docs_enabled=True,
              markup=None,
              docs_num_results=None,
              **callbacks):
    """Call `scaffold.retry_tool_calls` with `None` defaults for unspecified callbacks."""
    cb_kwargs = {name: callbacks.get(name, None) for name in _AI_TURN_CALLBACKS}
    return scaffold.retry_tool_calls(llm_settings=llm_settings,
                                     datastore=forest,
                                     retriever=retriever,
                                     tool_node_id=tool_node_id,
                                     internet_enabled=internet_enabled,
                                     docs_enabled=docs_enabled,
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
                            lambda settings, message, on_call_start, on_call_done, **kw: list(records))
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
        def capture_perform(settings, message, on_call_start, on_call_done, **kw):
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
        def capture_perform(settings, message, on_call_start, on_call_done, **kw):
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
        def capture_perform(settings, message, on_call_start, on_call_done, **kw):
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
    measurement rather than by taste (`investigations/context-injects/context-inject-shape-measurements.md`). Nothing about
    these shapes is self-evidently right, so they are worth holding still: a plausible-looking
    simplification here costs an hour of live probing to catch.
    """

    def test_injects_add_no_system_message(self, llm_settings):
        history = make_conversation(llm_settings)
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  docs_query=None, docs_matches=[],
                                  tool_context=grounding_context())
        assert_at_most_one_leading_system_message(history)

    def test_rag_matches_add_no_system_message(self, llm_settings):
        # Each match used to go in as its own system message at index 1, which failed every AI turn on
        # Qwen3.5 — several system messages are rejected even though all of them precede the conversation.
        history = make_conversation(llm_settings)
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  docs_query="what is X?",
                                  docs_matches=[sample_rag_match(document_id="a.txt"),
                                                sample_rag_match(document_id="b.txt")],
                                  tool_context=grounding_context())
        assert_at_most_one_leading_system_message(history)

    def test_rag_matches_ride_in_one_tool_message_in_corpus_order(self, llm_settings):
        # A `tool` message answers exactly one `tool_call_id`, so all matches share one message rather
        # than getting one each — the one-per-match form shares an id across messages, which Gemma4-E4B
        # reads as nothing at all.
        history = make_conversation(llm_settings)
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  docs_query="what is X?",
                                  docs_matches=[sample_rag_match(document_id="a.txt"),
                                                sample_rag_match(document_id="b.txt")],
                                  tool_context=grounding_context())
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
                                  docs_query="what is X?",
                                  docs_matches=[sample_rag_match()],
                                  tool_context=grounding_context())
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
                                  docs_query="what is X?",
                                  docs_matches=[sample_rag_match()],
                                  tool_context=grounding_context())
        assert history[-1]["role"] == "user"
        assert chatutil.content_to_text(history[-1]["content"]).endswith("What is X?")

    def test_injects_do_not_disturb_a_message_being_continued(self, llm_settings):
        # When continuing, the history must look as it did when generation was interrupted: the AI's
        # incomplete message stays last, with the injects ahead of the user turn it is answering.
        history = make_conversation(llm_settings)
        history.append(chatutil.create_chat_message(llm_settings=llm_settings, role="assistant", text="X is"))
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  docs_query=None, docs_matches=[],
                                  tool_context=grounding_context())
        assert_at_most_one_leading_system_message(history)
        assert history[-1]["role"] == "assistant"
        assert history[-2]["role"] == "user"

    def test_reminders_and_date_go_into_the_system_message(self, llm_settings):
        # Instruction-like injects want the leading system block: measured the cheapest placement in
        # deliberation tokens, and the only one that never provoked the model into remarking on them.
        history = make_conversation(llm_settings)
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  docs_query="what is X?",
                                  docs_matches=[sample_rag_match()],
                                  tool_context=grounding_context(grounded=True))
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
                                  docs_query=None, docs_matches=[],
                                  tool_context=grounding_context())
        assert chatutil.content_to_text(stored_system_message["content"]) == stored_text

    def test_context_only_reminder_is_skipped_without_context(self, llm_settings):
        # Asking a model to stick to documents that were never provided is a contradiction it will
        # dutifully try to resolve — up to 37x the deliberation, and on one model, never terminating.
        history = make_conversation(llm_settings)
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  docs_query=None, docs_matches=[],
                                  tool_context=grounding_context())
        assert "Base claims about the provided documents" not in chatutil.content_to_text(history[0]["content"])

    def test_the_injects_only_ever_name_registered_tools(self, llm_settings):
        """A synthetic tool exchange whose function is not a real tool is a fiction the model can act on.

        This is the invariant the clock inject broke: it called `get_current_time`, which was in no
        registry, for as long as it existed. The same shape cost a measured failure once already — before
        `search_documents` was a real tool, the document-matches inject named it anyway and the model wrote
        the call out as literal text instead of answering, roughly one turn in three.
        """
        history = make_conversation(llm_settings)
        tool_context = grounding_context(grounded=True)
        tool_context.consulted_documents = [{"document_id": "d.bib"}]  # so the consulted-list inject fires too
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  docs_query="what is X?", docs_matches=[sample_rag_match()],
                                  tool_context=tool_context)
        named = {call["function"]["name"]
                 for message in history
                 for call in (message.get("tool_calls") or [])}
        assert named, "no synthetic tool calls were injected; this test would pass vacuously"
        assert named <= set(llm_settings.tool_entrypoints), (
            f"injects name tools that do not exist: {sorted(named - set(llm_settings.tool_entrypoints))}")

    def test_context_only_reminder_counts_an_attachment_as_context(self, llm_settings):
        # "Context" is broader than docs matches: an attached document or image is material to ground in,
        # even on a turn where the document database returned nothing.
        history = make_conversation(llm_settings)
        history[-1]["content"].append(chatutil.text_file_content_part(url="sidecar:deadbeef.pdf", name="paper.pdf",
                                                                      source="user_attachment"))
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  docs_query=None, docs_matches=[],
                                  tool_context=grounding_context())
        assert "Base claims about the provided documents" in chatutil.content_to_text(history[0]["content"])

    def test_context_only_reminder_ignores_the_shape_of_the_history(self, llm_settings):
        # `_perform_injects` asks the turn's state, never the history's shape. Here the branch is full of
        # `role="tool"` messages and the answer is still "nothing grounds this", because that is what was
        # declared. Reintroducing an inference from message shape — the mechanism this replaced — would
        # switch the reminder back on, which is the regression this guards.
        #
        # Whether a *stored* tool result like this one grounds a later turn is a separate question, decided
        # by `_branch_grounding_is_present` from the node's recorded declaration, and tested there.
        history = [chatutil.create_chat_message(llm_settings=llm_settings, role="system", text="You are a helpful assistant."),
                   chatutil.create_chat_message(llm_settings=llm_settings, role="user", text="What is the weather in Tampere?"),
                   chatutil.create_chat_message(llm_settings=llm_settings, role="tool", text="Tampere: 17 C, cloudy."),
                   chatutil.create_chat_message(llm_settings=llm_settings, role="assistant", text="It is 17 C and cloudy."),
                   chatutil.create_chat_message(llm_settings=llm_settings, role="user", text="What is the baseline drift of the Kelvin-7 microarray?")]
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  docs_query=None, docs_matches=[],
                                  tool_context=grounding_context())
        assert "Base claims about the provided documents" not in chatutil.content_to_text(history[0]["content"])

    def test_context_only_reminder_follows_the_declaration_not_the_history(self, llm_settings):
        # Mid-agent-loop, a search that found something is exactly the material to ground in - but whether
        # it found something is what the tool declared, not what the history looks like. A `role="tool"`
        # message is present either way, which is why the shape cannot answer the question: an empty search
        # result is a perfectly well-formed tool message carrying nothing.
        history = make_conversation(llm_settings)
        history.append(chatutil.create_chat_message(llm_settings=llm_settings, role="tool", text="Search result: X is a variable."))
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  docs_query=None, docs_matches=[],
                                  tool_context=grounding_context(grounded=True))
        assert "Base claims about the provided documents" in chatutil.content_to_text(history[0]["content"])

    def test_context_only_reminder_is_skipped_when_the_tool_found_nothing(self, llm_settings):
        # The same history shape, with the opposite declaration. This is the case the whole mechanism
        # exists for: sent with nothing to ground in, the reminder is a self-contradiction that measured
        # 5-37x the deliberation of sending nothing, and on one model never terminated.
        history = make_conversation(llm_settings)
        history.append(chatutil.create_chat_message(llm_settings=llm_settings, role="tool", text="No matches."))
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  docs_query=None, docs_matches=[],
                                  tool_context=grounding_context(grounded=False))
        assert "Base claims about the provided documents" not in chatutil.content_to_text(history[0]["content"])

    def test_speculation_on_sends_no_context_only_reminder(self, llm_settings):
        history = make_conversation(llm_settings)
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  docs_query="what is X?",
                                  docs_matches=[sample_rag_match()],
                                  tool_context=grounding_context())
        assert "Base claims about the provided documents" not in chatutil.content_to_text(history[0]["content"])

    def test_injects_carry_no_persona_prefix(self, llm_settings):
        # The inject text is bracketed and self-labelling; prefixing the speaker's persona to it
        # ("User: [System information: ...]") would read as the user narrating a system notice.
        history = make_conversation(llm_settings)
        before = len(history)
        scaffold._perform_injects(llm_settings=llm_settings, history=history,
                                  docs_query=None, docs_matches=[],
                                  tool_context=grounding_context())

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
        tool_context = scaffold._make_tool_context(llm_settings=None, retriever=None)
        scaffold._record_grounding(tool_context, make_tool_record("Kelvin-7 drifts 0.3 K/h."))
        assert tool_context.grounded

    def test_failed_call_does_not_ground(self):
        tool_context = scaffold._make_tool_context(llm_settings=None, retriever=None)
        scaffold._record_grounding(tool_context, make_tool_record("Tool call failed.", status="error"))
        assert not tool_context.grounded

    def test_empty_result_does_not_ground(self):
        # A search that found nothing is the case the whole mechanism exists for: it is a perfectly
        # well-formed tool message carrying no material at all.
        tool_context = scaffold._make_tool_context(llm_settings=None, retriever=None)
        scaffold._record_grounding(tool_context, make_tool_record("   \n  "))
        assert not tool_context.grounded

    def test_declaration_overrides_the_default(self):
        # webfetch's allowlist refusal is the live example: non-empty, successful, grounds nothing.
        tool_context = scaffold._make_tool_context(llm_settings=None, retriever=None)
        scaffold._record_grounding(tool_context,
                                   make_tool_record("The host example.com is not on the configured allowlist.",
                                                    tool_metadata={"grounding": False}))
        assert not tool_context.grounded

    def test_declaration_can_ground_an_empty_looking_result(self):
        tool_context = scaffold._make_tool_context(llm_settings=None, retriever=None)
        scaffold._record_grounding(tool_context, make_tool_record("", tool_metadata={"grounding": True}))
        assert tool_context.grounded

    def test_grounding_is_monotonic_within_a_turn(self):
        # A tool call in round 1 must still count in round 3 - which is why the context is per-turn and
        # not per-round. A later empty search cannot un-ground what an earlier one found.
        tool_context = scaffold._make_tool_context(llm_settings=None, retriever=None)
        scaffold._record_grounding(tool_context, make_tool_record("Found it."))
        scaffold._record_grounding(tool_context, make_tool_record(""))
        scaffold._record_grounding(tool_context, make_tool_record("failed", status="error"))
        assert tool_context.grounded


# ---------------------------------------------------------------------------
# Per-turn tool availability (the document tools follow the document database)
# ---------------------------------------------------------------------------

def capture_tool_names(monkeypatch):
    """Stub `llmclient.invoke` and capture the `tool_names` it was handed. Returns the dict it writes into."""
    seen = {}

    def fake_invoke(**kw):
        seen["tool_names"] = kw.get("tool_names")
        return make_invoke_result(content="OK")

    monkeypatch.setattr("raven.librarian.llmclient.invoke", fake_invoke)
    return seen


class TestDocumentToolGating:
    """`docs_enabled` gates the document *tools*; `docs_query` gates the automatic *search*.

    They collapsed into one switch while the automatic search was the only way to reach the documents.
    They cannot stay collapsed: a continuation turn runs no automatic search but must keep offering the
    tools, because a tool that appears and vanishes between rounds of one agent loop is a shape models
    read as noise.
    """

    def test_documents_in_play_offers_every_tool(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                       head_node_id=head, user_message_text="What is X?")
        seen = capture_tool_names(monkeypatch)
        run_ai_turn(forest, llm_settings, user_head,
                    retriever=FakeRetriever(results=[sample_rag_match()]),
                    docs_enabled=True, docs_query="What is X?")
        assert seen["tool_names"] is None  # `None` is the permissive value: offer all registered tools

    def test_docs_disabled_withdraws_only_the_document_tools(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                       head_node_id=head, user_message_text="What is 2+2?")
        seen = capture_tool_names(monkeypatch)
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
        seen = capture_tool_names(monkeypatch)
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
        seen = capture_tool_names(monkeypatch)
        run_ai_turn(forest, llm_settings, user_head,
                    retriever=FakeRetriever(), docs_enabled=True, docs_query=None)
        assert seen["tool_names"] is None


class TestTheTwoSwitchesAreIndependent:
    """Each switch owns one group of tools outright, so all four combinations mean something.

    The arrangement they replaced did not: a blanket *Tools* switch sat above *Documents*, so with tools
    off and documents on the user had switched documents on and the model still could not search them.

    A tool answering to neither switch is always offered. That is `get_current_time`, and it has to be:
    the clock inject is delivered on every turn regardless of both switches, as a synthetic call to that
    very function, so withholding the spec would leave the model reading a call it cannot resolve.
    """

    def _offered(self, monkeypatch, llm_settings, populated_forest, *, internet, docs):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                       head_node_id=head, user_message_text="What is X?")
        seen = capture_tool_names(monkeypatch)
        run_ai_turn(forest, llm_settings, user_head,
                    retriever=FakeRetriever(results=[sample_rag_match()]),
                    internet_enabled=internet, docs_enabled=docs, docs_query=None)
        names = seen["tool_names"]
        if names is None:  # the permissive value: every registered tool
            return set(llm_settings.tool_entrypoints)
        return set(names)

    def test_both_on_offers_everything(self, monkeypatch, llm_settings, populated_forest):
        offered = self._offered(monkeypatch, llm_settings, populated_forest, internet=True, docs=True)
        assert offered == set(llm_settings.tool_entrypoints)

    def test_documents_on_internet_off_still_lets_the_model_search_the_documents(self, monkeypatch, llm_settings, populated_forest):
        """The combination the old arrangement got wrong, and the reason for this change."""
        offered = self._offered(monkeypatch, llm_settings, populated_forest, internet=False, docs=True)
        assert llm_settings.document_tool_names <= offered
        assert not (llm_settings.network_tool_names & offered)

    def test_internet_on_documents_off(self, monkeypatch, llm_settings, populated_forest):
        offered = self._offered(monkeypatch, llm_settings, populated_forest, internet=True, docs=False)
        assert llm_settings.network_tool_names <= offered
        assert not (llm_settings.document_tool_names & offered)

    def test_both_off_leaves_only_the_ungated_tools(self, monkeypatch, llm_settings, populated_forest):
        offered = self._offered(monkeypatch, llm_settings, populated_forest, internet=False, docs=False)
        ungated = set(llm_settings.tool_entrypoints) - set(llm_settings.document_tool_names) - set(llm_settings.network_tool_names)
        assert offered == ungated

    def test_the_clock_tool_answers_to_neither_switch(self, monkeypatch, llm_settings, populated_forest):
        """Named explicitly, not derived: the general rule above would still pass if the clock tool were
        quietly moved into a gated group, and that would break the inject that names it."""
        for internet, docs in ((True, True), (True, False), (False, True), (False, False)):
            offered = self._offered(monkeypatch, llm_settings, populated_forest, internet=internet, docs=docs)
            assert "get_current_time" in offered, f"withheld with internet={internet}, docs={docs}"


# ---------------------------------------------------------------------------
# Agent-loop round cap
# ---------------------------------------------------------------------------

class TestToolCallRoundCap:
    """A model that keeps rephrasing a failing search must still end up answering.

    Past the cap the tools stay in the schema and calls are refused, so the loadout does not change under
    the model mid-turn. Withdrawing them is the terminator of last resort, and it is an invocation with no
    tools rather than a `break`: breaking would leave the turn's last message a tool result, which reads as
    a paused agent loop and draws yet another call instead of a reply.
    """

    def _always_calls_tools(self, monkeypatch, counter, refusals=None):
        """A model that asks for a tool whenever it is offered one, and only answers when it is not.

        `refusals` collects the `maybe_refusal_text` each round was dispatched with (`None` = really ran).
        """
        def fake_invoke(**kw):
            counter.append(kw.get("tools_enabled"))
            if not kw.get("tools_enabled"):  # no tools on offer -> the model has to answer
                return make_invoke_result(content="Fine, here is my answer.")
            return make_invoke_result(content="",
                                      tool_calls=[tool_call("search_documents", f"call_{len(counter)}")])

        def fake_perform_tool_calls(*a, **kw):
            if refusals is not None:
                refusals.append(kw.get("maybe_refusal_text"))
            if kw.get("maybe_refusal_text") is not None:
                return [make_tool_response(content=kw["maybe_refusal_text"],
                                           function_name="search_documents", status="error")]
            return [make_tool_response(content="No matches.", function_name="search_documents")]

        monkeypatch.setattr("raven.librarian.llmclient.invoke", fake_invoke)
        monkeypatch.setattr("raven.librarian.llmclient.perform_tool_calls", fake_perform_tool_calls)

    def test_the_cap_refuses_before_it_withdraws(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                       head_node_id=head, user_message_text="Find X")
        monkeypatch.setattr("raven.librarian.config.max_tool_call_rounds", 3)
        monkeypatch.setattr("raven.librarian.config.max_tool_call_refusal_rounds", 1)
        offered = []
        refusals = []
        self._always_calls_tools(monkeypatch, offered, refusals)

        final_head = run_ai_turn(forest, llm_settings, user_head, retriever=FakeRetriever())

        # 3 rounds that run, 1 that is refused with the tools still on offer, then the withdrawal.
        assert offered == [True, True, True, True, False]
        assert [text is None for text in refusals] == [True, True, True, False]
        assert forest.get_payload(final_head)["message"]["role"] == "assistant"

    def test_a_refusal_round_calls_no_tool(self, monkeypatch, llm_settings, populated_forest):
        # The whole point: the round past the cap answers the model without doing any work.
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                       head_node_id=head, user_message_text="Find X")
        monkeypatch.setattr("raven.librarian.config.max_tool_call_rounds", 1)
        monkeypatch.setattr("raven.librarian.config.max_tool_call_refusal_rounds", 1)
        refusals = []
        self._always_calls_tools(monkeypatch, [], refusals)

        run_ai_turn(forest, llm_settings, user_head, retriever=FakeRetriever())
        assert refusals == [None, chatutil.format_error_that_tools_are_spent()]

    def test_zero_refusal_rounds_withdraws_at_the_cap(self, monkeypatch, llm_settings, populated_forest):
        # The escape hatch back to withdrawing outright, for a model the refusal does not reach.
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                       head_node_id=head, user_message_text="Find X")
        monkeypatch.setattr("raven.librarian.config.max_tool_call_rounds", 2)
        monkeypatch.setattr("raven.librarian.config.max_tool_call_refusal_rounds", 0)
        offered = []
        self._always_calls_tools(monkeypatch, offered)

        run_ai_turn(forest, llm_settings, user_head, retriever=FakeRetriever())
        assert offered == [True, True, False]

    def test_the_final_reply_is_an_answer_not_a_tool_result(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                       head_node_id=head, user_message_text="Find X")
        monkeypatch.setattr("raven.librarian.config.max_tool_call_rounds", 2)
        self._always_calls_tools(monkeypatch, [])

        final_head = run_ai_turn(forest, llm_settings, user_head, retriever=FakeRetriever())
        payload = forest.get_payload(final_head)
        assert payload["message"]["role"] == "assistant"
        assert not payload["message"]["tool_calls"]

    def test_the_refusal_is_stored_as_a_failed_tool_result(self, monkeypatch, llm_settings, populated_forest):
        # The model sees it through the tool channel, so it has to be a tool node like any other - and an
        # errored one, so that nothing downstream mistakes it for material to ground an answer on.
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                       head_node_id=head, user_message_text="Find X")
        monkeypatch.setattr("raven.librarian.config.max_tool_call_rounds", 1)
        monkeypatch.setattr("raven.librarian.config.max_tool_call_refusal_rounds", 1)
        self._always_calls_tools(monkeypatch, [])

        final_head = run_ai_turn(forest, llm_settings, user_head, retriever=FakeRetriever())
        # `payloads_up` walks from the leaf, so the newest tool node comes first.
        refusal = next(payload for payload in payloads_up(forest, final_head)
                       if payload["message"]["role"] == "tool")
        assert refusal["generation_metadata"]["status"] == "error"
        assert "budget for this reply is spent" in chatutil.content_to_text(refusal["message"]["content"])

    def test_an_ordinary_turn_never_reaches_the_cap(self, monkeypatch, llm_settings, populated_forest):
        # A model that gets what it needs stops on its own, well under the cap - the cap is a backstop,
        # not a normal limit, and must not perturb the common path.
        forest, head = populated_forest
        user_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                       head_node_id=head, user_message_text="Find X")
        offered = []

        responses = iter([make_invoke_result(content="", tool_calls=[tool_call("search_documents", "call_0")]),
                          make_invoke_result(content="X is foo.")])

        def fake_invoke(**kw):
            offered.append(kw.get("tools_enabled"))
            return next(responses)

        monkeypatch.setattr("raven.librarian.llmclient.invoke", fake_invoke)
        monkeypatch.setattr("raven.librarian.llmclient.perform_tool_calls",
                            lambda *a, **kw: [make_tool_response(content="X is foo.",
                                                                 function_name="search_documents")])

        run_ai_turn(forest, llm_settings, user_head, retriever=FakeRetriever())
        assert offered == [True, True]  # tools stayed on offer throughout; the cap never intervened


# ---------------------------------------------------------------------------
# Grounding carried across turns (persisted material stays material)
# ---------------------------------------------------------------------------

class TestBranchGrounding:
    """A tool result is a persisted node, so it grounds for as long as it is in the window.

    The rule is mechanical — *is this material still in the context* — rather than semantic, because
    "has it gone stale" is a judgment none of this code can make. The automatic pre-turn search is the one
    thing scoped to its own turn, and that is a fact about the data (it is never persisted) rather than a
    policy about lifetimes.
    """

    def _forest_with_tool_node(self, llm_settings, text, generation_metadata):
        forest = chattree.Forest()
        head = chatutil.factory_reset_datastore(forest, llm_settings)
        head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                  head_node_id=head, user_message_text="What is X?")
        payload = chatutil.create_payload(llm_settings=llm_settings,
                                          message=chatutil.create_chat_message(llm_settings=llm_settings,
                                                                               role="tool", text=text))
        payload["generation_metadata"] = generation_metadata
        head = forest.create_node(payload=payload, parent_id=head)
        head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                  head_node_id=head, user_message_text="And the follow-up?")
        return forest, head

    def test_an_earlier_declared_result_still_grounds(self, llm_settings):
        forest, head = self._forest_with_tool_node(llm_settings, "X is foo.",
                                                   {"status": "success", "function_name": "search_documents",
                                                    "grounding": True})
        assert scaffold._branch_grounding_is_present(forest, head)

    def test_an_earlier_empty_search_does_not_ground(self, llm_settings):
        # The declaration is what carries, not the presence of a tool node: "no matches" is a well-formed
        # tool message holding nothing.
        forest, head = self._forest_with_tool_node(llm_settings, "The document database contains no matches.",
                                                   {"status": "success", "function_name": "search_documents",
                                                    "grounding": False})
        assert not scaffold._branch_grounding_is_present(forest, head)

    def test_an_undeclared_nonempty_result_grounds(self, llm_settings):
        # Tools that predate the declaration convention, and any future MCP tool, fall back to
        # "it returned something".
        forest, head = self._forest_with_tool_node(llm_settings, "Some retrieved material.",
                                                   {"status": "success", "function_name": "mystery_tool"})
        assert scaffold._branch_grounding_is_present(forest, head)

    def test_a_failed_earlier_call_does_not_ground(self, llm_settings):
        forest, head = self._forest_with_tool_node(llm_settings, "Tool call failed.",
                                                   {"status": "error", "function_name": "websearch"})
        assert not scaffold._branch_grounding_is_present(forest, head)

    def test_a_conversation_with_no_tool_results_does_not_ground(self, llm_settings):
        # The conversation itself is never grounding: a model summarizing its own earlier reply is exactly
        # the ungrounded answer this guards against.
        forest = chattree.Forest()
        head = chatutil.factory_reset_datastore(forest, llm_settings)
        head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                  head_node_id=head, user_message_text="What is X?")
        assert not scaffold._branch_grounding_is_present(forest, head)

    def test_ai_turn_seeds_the_flag_from_the_branch(self, monkeypatch, llm_settings):
        # The end-to-end shape of the follow-up problem: turn 1's search grounded, turn 2 searches nothing,
        # and the context-only reminder must still be sent because turn 1's material is still in the window.
        forest, head = self._forest_with_tool_node(llm_settings, "X is foo.",
                                                   {"status": "success", "function_name": "search_documents",
                                                    "grounding": True})
        seen = {}

        def fake_invoke(**kw):
            if kw.get("on_prompt_ready") is not None:
                kw["on_prompt_ready"](kw["history"])
            seen["history"] = kw["history"]
            return make_invoke_result(content="Still foo.")

        monkeypatch.setattr("raven.librarian.llmclient.invoke", fake_invoke)
        run_ai_turn(forest, llm_settings, head, retriever=FakeRetriever())
        system_text = chatutil.content_to_text(seen["history"][0]["content"])
        assert "Base claims about the provided documents" in system_text


# ---------------------------------------------------------------------------
# The provenance list: what this branch has already read
# ---------------------------------------------------------------------------

class TestConsultedDocuments:
    """The automatic search's matches are injected once and never persisted, so a follow-up question
    arrives with the reply in view and the material behind it gone. Only the IDs survive."""

    def _forest_with_retrieval(self, llm_settings, *, query, document_ids):
        forest = chattree.Forest()
        root = forest.create_node(payload=chatutil.create_payload(
            llm_settings=llm_settings,
            message=chatutil.create_chat_message(llm_settings=llm_settings, role="user", text="Tell me about X.")),
            parent_id=None)
        payload = chatutil.create_payload(
            llm_settings=llm_settings,
            message=chatutil.create_chat_message(llm_settings=llm_settings, role="assistant", text="X is foo."))
        payload["retrieval"] = {"query": query,
                                "results": [sample_rag_match(document_id=document_id) for document_id in document_ids]}
        return forest, forest.create_node(payload=payload, parent_id=root)

    def test_an_earlier_turns_auto_search_is_remembered(self, llm_settings):
        forest, head = self._forest_with_retrieval(llm_settings, query="about X", document_ids=["a.txt", "b.txt"])
        entries = scaffold._collect_consulted_documents(forest, head, exclude_document_ids=[])
        assert [entry["document_id"] for entry in entries] == ["a.txt", "b.txt"]
        assert all(entry["query"] == "about X" for entry in entries)

    def test_this_turns_matches_are_left_out(self, llm_settings):
        # Their full text is sitting right beside the list; naming them again is redundancy in the one
        # place that has to stay compact.
        forest, head = self._forest_with_retrieval(llm_settings, query="q", document_ids=["a.txt", "b.txt"])
        entries = scaffold._collect_consulted_documents(forest, head, exclude_document_ids=["a.txt"])
        assert [entry["document_id"] for entry in entries] == ["b.txt"]

    def test_a_document_appearing_twice_is_listed_once(self, llm_settings):
        forest, head = self._forest_with_retrieval(llm_settings, query="q1", document_ids=["a.txt"])
        payload = chatutil.create_payload(
            llm_settings=llm_settings,
            message=chatutil.create_chat_message(llm_settings=llm_settings, role="assistant", text="More."))
        payload["retrieval"] = {"query": "q2", "results": [sample_rag_match(document_id="a.txt")]}
        head = forest.create_node(payload=payload, parent_id=head)
        entries = scaffold._collect_consulted_documents(forest, head, exclude_document_ids=[])
        assert [entry["document_id"] for entry in entries] == ["a.txt"]
        assert entries[0]["query"] == "q2"  # newest first, so the most recent query is the one shown

    def test_what_the_model_fetched_counts_too(self, llm_settings):
        # "Consulted" is silent about who consulted: a tool node's declared metadata is read alongside the
        # automatic search's stored payload.
        forest, head = self._forest_with_retrieval(llm_settings, query="q", document_ids=["a.txt"])
        payload = chatutil.create_payload(
            llm_settings=llm_settings,
            message=chatutil.create_chat_message(llm_settings=llm_settings, role="tool", text="(document text)"))
        payload["generation_metadata"] = {"status": "success", "document_ids": ["fetched.bib"]}
        head = forest.create_node(payload=payload, parent_id=head)
        entries = scaffold._collect_consulted_documents(forest, head, exclude_document_ids=[])
        assert {entry["document_id"] for entry in entries} == {"a.txt", "fetched.bib"}

    def test_the_list_is_capped_newest_first(self, monkeypatch, llm_settings):
        monkeypatch.setattr("raven.librarian.config.max_consulted_documents_listed", 2)
        forest, head = self._forest_with_retrieval(llm_settings, query="q",
                                                   document_ids=["a.txt", "b.txt", "c.txt"])
        entries = scaffold._collect_consulted_documents(forest, head, exclude_document_ids=[])
        assert len(entries) == 2

    def test_the_list_is_injected_so_the_model_need_not_ask(self, monkeypatch, llm_settings):
        # Pushed rather than only offered as a tool: at a follow-up question the model's own transcript
        # shows it answering from documents, so nothing signals that the material was dropped.
        forest, head = self._forest_with_retrieval(llm_settings, query="about X", document_ids=["a.txt"])
        seen = {}

        def fake_invoke(**kw):
            seen["history"] = kw["history"]
            return make_invoke_result(content="Still foo.")

        monkeypatch.setattr("raven.librarian.llmclient.invoke", fake_invoke)
        run_ai_turn(forest, llm_settings, head,
                    retriever=FakeRetriever(documents={"a.txt": "The Title Of The Document\n\nBody."}),
                    docs_enabled=True)
        wire_text = "\n".join(chatutil.content_to_text(message["content"]) for message in seen["history"])
        assert "a.txt" in wire_text
        assert "The Title Of The Document" in wire_text  # labelled, so the model can decide without fetching

    def test_nothing_consulted_means_no_inject(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        seen = {}

        def fake_invoke(**kw):
            seen["history"] = kw["history"]
            return make_invoke_result(content="Hi.")

        monkeypatch.setattr("raven.librarian.llmclient.invoke", fake_invoke)
        run_ai_turn(forest, llm_settings, head, retriever=FakeRetriever(), docs_enabled=True)
        wire_text = "\n".join(chatutil.content_to_text(message["content"]) for message in seen["history"])
        assert "list_consulted_documents" not in wire_text


class TestSpentToolsNotice:
    """Withdrawing the tools at the cap is not by itself enough to make the model answer.

    Measured live: given a list of documents to work through, the model spends its rounds fetching them one
    at a time, and on the invocation where the tools are gone it announces the *next* fetch and stops, with
    no reply written at all. Five of six sampled turns that reached the cap ended empty.
    """

    def _system_text_of(self, monkeypatch, llm_settings, forest, head, *, rounds_before_reply):
        """Run an AI turn whose model asks for a tool `rounds_before_reply` times, and return the last
        system message it saw."""
        seen = []
        calls = {"n": 0}

        def fake_invoke(**kw):
            seen.append(chatutil.content_to_text(kw["history"][0]["content"]))
            calls["n"] += 1
            if calls["n"] <= rounds_before_reply:
                return make_invoke_result(content="", tool_calls=[{"type": "function", "id": f"c{calls['n']}",
                                                                   "index": "0",
                                                                   "function": {"name": "websearch",
                                                                                "arguments": '{"query": "x"}'}}])
            return make_invoke_result(content="Here is the answer.")

        monkeypatch.setattr("raven.librarian.llmclient.invoke", fake_invoke)
        monkeypatch.setattr("raven.librarian.llmclient.perform_tool_calls",
                            lambda settings, message, on_call_start, on_call_done, **kw: [
                                env(data=chatutil.create_chat_message(llm_settings=llm_settings, role="tool",
                                                                      text="(a result)"),
                                    status="success", function_name="websearch", tool_call_id="c1")])
        run_ai_turn(forest, llm_settings, head, retriever=None, docs_enabled=False)
        return seen[-1]

    def test_the_notice_is_absent_below_the_cap(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        monkeypatch.setattr("raven.librarian.config.max_tool_call_rounds", 5)
        system_text = self._system_text_of(monkeypatch, llm_settings, forest, head, rounds_before_reply=1)
        assert "No further tool calls" not in system_text

    def test_the_notice_arrives_when_the_budget_is_spent(self, monkeypatch, llm_settings, populated_forest):
        forest, head = populated_forest
        monkeypatch.setattr("raven.librarian.config.max_tool_call_rounds", 2)
        system_text = self._system_text_of(monkeypatch, llm_settings, forest, head, rounds_before_reply=99)
        assert "No further tool calls" in system_text
        assert "Write the answer now" in system_text

    def test_the_notice_permits_an_incomplete_answer(self, monkeypatch, llm_settings, populated_forest):
        # A model whose gathering was cut short mid-task otherwise has a reason to keep trying rather than
        # to report what it has.
        forest, head = populated_forest
        monkeypatch.setattr("raven.librarian.config.max_tool_call_rounds", 1)
        system_text = self._system_text_of(monkeypatch, llm_settings, forest, head, rounds_before_reply=99)
        assert "say so in the answer" in system_text


# ---------------------------------------------------------------------------
# Large tool results become attachments
# ---------------------------------------------------------------------------

def make_fetch_response(text, *, url="https://example.com/paper", title="A Paper", tool_call_id="call_0"):
    """A faked successful `webfetch` record: declares `fetched_document`, as `llmclient.webfetch_wrapper` does."""
    return env(data={"role": "tool",
                     "content": chatutil.normalize_content(text),
                     "tool_calls": None},
               status="success",
               tool_call_id=tool_call_id,
               function_name="webfetch",
               dt=0.01,
               tool_metadata={"fetched_document": {"url": url, "name": title}})


class TestDocumentDisplayName:
    """The name a fetched document is filed under — for humans, not for uniqueness (the hash has that)."""

    @staticmethod
    def _name(url, title):
        return scaffold._document_display_name({"url": url, "name": title})

    def test_host_leads_the_title(self):
        # The host is what still means something in a folder of rescued attachments months later.
        assert self._name("https://arxiv.org/abs/1706.03762", "Attention Is All You Need") == "arxiv.org - Attention Is All You Need"

    def test_a_titleless_page_is_named_by_its_host(self):
        # `webfetch_wrapper` falls back to the URL as the title when a page has none; repeating the whole
        # URL after the host would be noise.
        url = "https://example.com/some/deep/path"
        assert self._name(url, url) == "example.com"

    def test_path_separators_are_not_left_in_the_name(self):
        # A page title is arbitrary text from the open web, and this name reaches the filesystem via
        # `cleanup.rescue_to_staging`.
        assert self._name("https://example.com/x", "AC/DC: Back in Black") == "example.com - AC-DC- Back in Black"

    def test_two_pages_sharing_a_title_are_told_apart_by_host(self):
        assert self._name("https://a.example/x", "Overview") != self._name("https://b.example/y", "Overview")

    def test_never_empty(self):
        # Something has to go on the chip even when the fetch tells us nothing useful.
        assert self._name("", "") == "fetched document"


class TestToolResultAttachments:
    """A long fetched document goes to a sidecar; the chat log keeps an excerpt and a chip."""

    # Long enough to cross the default threshold, and shaped like a fetched page: a source header, then prose.
    LONG_DOCUMENT = ("**Webfetch result from** [https://example.com/paper](https://example.com/paper):\n\n"
                     "**A Paper**\n\n-----\n\n" + "The body of the paper. " * 500)

    def _forest(self, tmp_path, llm_settings):
        forest = chattree.PersistentForest(tmp_path / "chat.json", autosave=False,
                                           sidecar_extractor=textfilestore.sidecar_refs_in_payload)
        greeting = chatutil.factory_reset_datastore(forest, llm_settings)
        return forest, greeting

    def _run_one_fetch(self, monkeypatch, llm_settings, forest, head, record):
        """Drive one `ai_turn` whose single tool call returns `record`. Returns the tool node's payload."""
        user_head = scaffold.user_turn(llm_settings=llm_settings, datastore=forest,
                                       head_node_id=head, user_message_text="read this page")
        responses = iter([make_invoke_result(content="", tool_calls=[tool_call("webfetch", "call_0")]),
                          make_invoke_result(content="Here is what it says.")])
        monkeypatch.setattr("raven.librarian.llmclient.invoke", lambda **kw: next(responses))
        monkeypatch.setattr("raven.librarian.llmclient.perform_tool_calls",
                            lambda settings, message, on_call_start, on_call_done, **kw: [record])
        tool_nodes = []
        run_ai_turn(forest, llm_settings, user_head, on_tool_done=lambda nid: tool_nodes.append(nid))
        return forest.get_payload(tool_nodes[0])

    def test_long_fetch_becomes_an_excerpt_plus_a_chip(self, monkeypatch, tmp_path, llm_settings):
        forest, head = self._forest(tmp_path, llm_settings)
        payload = self._run_one_fetch(monkeypatch, llm_settings, forest, head,
                                      make_fetch_response(self.LONG_DOCUMENT))
        content = payload["message"]["content"]
        assert [part["type"] for part in content] == ["text", "text_file"]

        shown = chatutil.content_to_text(content)  # what the chat log renders as the message's own text
        assert shown.startswith("**Webfetch result from**")  # the excerpt opens where the document does
        assert len(shown) < len(self.LONG_DOCUMENT) / 10  # ...and is a small fraction of it

    def test_the_full_document_is_stored_verbatim_as_a_sidecar(self, monkeypatch, tmp_path, llm_settings):
        forest, head = self._forest(tmp_path, llm_settings)
        payload = self._run_one_fetch(monkeypatch, llm_settings, forest, head,
                                      make_fetch_response(self.LONG_DOCUMENT))
        part = payload["message"]["content"][1]
        url = part["text_file"]["url"]
        assert url.startswith(sidecarstore.SIDECAR_SCHEME)
        filename = url[len(sidecarstore.SIDECAR_SCHEME):]
        assert forest.read_sidecar(filename).decode("utf-8") == self.LONG_DOCUMENT
        assert filename.endswith(".md")  # so `docextract` reads it back as markdown, not as an unknown type

    def test_provenance_records_the_fetch_url_and_the_pathway(self, monkeypatch, tmp_path, llm_settings):
        forest, head = self._forest(tmp_path, llm_settings)
        payload = self._run_one_fetch(monkeypatch, llm_settings, forest, head,
                                      make_fetch_response(self.LONG_DOCUMENT,
                                                          url="https://arxiv.org/abs/1706.03762",
                                                          title="Attention Is All You Need"))
        sidecars = payload["general_metadata"]["sidecars"]
        assert len(sidecars) == 1
        entry = next(iter(sidecars.values()))
        assert entry["url"] == "https://arxiv.org/abs/1706.03762"  # what "Open source" opens
        assert entry["source"] == "tool_result"
        assert entry["name"] == "arxiv.org - Attention Is All You Need.md"

    def test_the_document_declaration_is_consumed_not_stored(self, monkeypatch, tmp_path, llm_settings):
        # `fetched_document` is an instruction to this code, not a fact about the tool call worth keeping.
        forest, head = self._forest(tmp_path, llm_settings)
        payload = self._run_one_fetch(monkeypatch, llm_settings, forest, head,
                                      make_fetch_response(self.LONG_DOCUMENT))
        assert "fetched_document" not in payload["generation_metadata"]
        assert payload["generation_metadata"]["function_name"] == "webfetch"  # the rest still lands

    def test_a_short_fetch_stays_inline(self, monkeypatch, tmp_path, llm_settings):
        # Hiding three paragraphs behind a chip is worse than showing them.
        forest, head = self._forest(tmp_path, llm_settings)
        payload = self._run_one_fetch(monkeypatch, llm_settings, forest, head,
                                      make_fetch_response("a short page"))
        assert [part["type"] for part in payload["message"]["content"]] == ["text"]
        assert chatutil.content_to_text(payload["message"]["content"]) == "a short page"
        assert "sidecars" not in payload["general_metadata"]

    def test_an_undeclared_tool_result_stays_inline_however_long(self, monkeypatch, tmp_path, llm_settings):
        # `websearch` returns a list of links the user wants to click; a chip over them is a regression.
        forest, head = self._forest(tmp_path, llm_settings)
        payload = self._run_one_fetch(monkeypatch, llm_settings, forest, head,
                                      make_tool_response(content="link. " * 2000, tool_call_id="call_0"))
        assert [part["type"] for part in payload["message"]["content"]] == ["text"]
        assert "sidecars" not in payload["general_metadata"]

    def test_a_failed_store_leaves_the_result_inline(self, monkeypatch, tmp_path, llm_settings):
        # Losing the tool result would be far worse than a long one in the log.
        forest, head = self._forest(tmp_path, llm_settings)
        def explode(**kwargs):
            raise OSError("disk full")
        monkeypatch.setattr("raven.librarian.textfilestore.store_file_as_sidecar", explode)
        payload = self._run_one_fetch(monkeypatch, llm_settings, forest, head,
                                      make_fetch_response(self.LONG_DOCUMENT))
        assert [part["type"] for part in payload["message"]["content"]] == ["text"]
        assert chatutil.content_to_text(payload["message"]["content"]) == self.LONG_DOCUMENT

    def test_the_model_still_reads_the_whole_document(self, monkeypatch, tmp_path, llm_settings):
        # The property that makes this safe to do behind the user's back: what changes is the chat log and
        # the datastore JSON, not the conversation. The wire build folds the sidecar's text back in.
        from raven.librarian import llmclient
        forest, head = self._forest(tmp_path, llm_settings)
        payload = self._run_one_fetch(monkeypatch, llm_settings, forest, head,
                                      make_fetch_response(self.LONG_DOCUMENT))
        wire = llmclient._serialize_history_for_wire(llm_settings, [payload["message"]],
                                                     continue_=False, datastore=forest)
        wire_text = chatutil.content_to_text(wire[0]["content"])
        assert "[Attached file: example.com - A Paper.md]" in wire_text
        body = self.LONG_DOCUMENT.split("-----\n\n")[1].strip()  # extraction strips trailing whitespace
        assert body in wire_text  # the body, in full — nothing was lost by storing it out of line
