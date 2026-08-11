"""Unit tests for raven.librarian.agent (the scripting surface over the agent loop).

No backend anywhere: `llmclient.invoke` and `llmclient.perform_tool_calls` are faked with the same helpers
`test_scaffold` uses, so what is exercised is the wiring between `agent` and the real `scaffold.ai_turn`
rather than a replica of the loop.
"""

import dataclasses
import logging

import pytest

from raven.librarian import agent, chattree, chatutil, llmclient

from .test_scaffold import FakeRetriever, make_invoke_result, make_tool_response, tool_call


# ---------------------------------------------------------------------------
# Building a branch by hand, for the `describe_turn` tests
# ---------------------------------------------------------------------------

def add(forest, llm_settings, parent_node_id, role, text="", **message_fields):
    """Append one chat node to `parent_node_id`, returning its id.

    `message_fields` go into the message (`tool_calls`, `reasoning_content`); a `generation_metadata` key
    goes onto the payload instead, which is where the tool name and the grounding marker live.
    """
    generation_metadata = message_fields.pop("generation_metadata", None)
    message = chatutil.create_chat_message(llm_settings=llm_settings, role=role, text=text)
    message.update(message_fields)
    payload = chatutil.create_payload(llm_settings=llm_settings, message=message)
    if generation_metadata is not None:
        payload["generation_metadata"] = generation_metadata
    return forest.create_node(payload=payload, parent_id=parent_node_id)


class TestDescribeTurn:
    """The branch walk every probe used to write out longhand, and the distinction they got wrong."""

    def test_one_assistant_message_asking_for_three_tools_is_one_round(self, llm_settings):
        # The vocabulary this record exists to fix in one place: a *round* is one assistant message asking
        # for tools, however many calls it asks for. Counting the tool nodes instead reports three.
        forest = chattree.Forest()
        head = add(forest, llm_settings, None, "user", "Find me three things.")
        head = add(forest, llm_settings, head, "assistant",
                   tool_calls=[tool_call("search_documents", "call_0"),
                               tool_call("search_documents", "call_1"),
                               tool_call("websearch", "call_2")])
        for name, call_id in (("search_documents", "call_0"), ("search_documents", "call_1"), ("websearch", "call_2")):
            head = add(forest, llm_settings, head, "tool", f"result of {call_id}",
                       generation_metadata={"function_name": name})
        head = add(forest, llm_settings, head, "assistant", "Here they are.")

        record = agent.describe_turn(forest, head)
        assert record.rounds == 1
        assert record.tool_calls == {"search_documents": 2, "websearch": 1}
        assert sum(record.tool_calls.values()) == 3

    def test_the_span_stops_at_the_node_it_was_given(self, llm_settings):
        # Walking to the root totals every turn on the branch, which answers a different question than
        # "what did this turn do" -- and is what a hand-rolled walk silently does.
        forest = chattree.Forest()
        head = add(forest, llm_settings, None, "user", "First.")
        head = add(forest, llm_settings, head, "assistant", tool_calls=[tool_call("websearch", "call_0")])
        head = add(forest, llm_settings, head, "tool", "result", generation_metadata={"function_name": "websearch"})
        first_turn_head = add(forest, llm_settings, head, "assistant", "Answer to the first.")

        head = add(forest, llm_settings, first_turn_head, "user", "Second.")
        started_from = head
        head = add(forest, llm_settings, head, "assistant", tool_calls=[tool_call("get_current_time", "call_1")])
        head = add(forest, llm_settings, head, "tool", "12:00", generation_metadata={"function_name": "get_current_time"})
        head = add(forest, llm_settings, head, "assistant", "Answer to the second.")

        second = agent.describe_turn(forest, head, since_node_id=started_from)
        assert second.rounds == 1
        assert second.tool_calls == {"get_current_time": 1}
        assert second.reply == "Answer to the second."

        whole_branch = agent.describe_turn(forest, head)
        assert whole_branch.rounds == 2
        assert whole_branch.tool_calls == {"websearch": 1, "get_current_time": 1}

    def test_the_span_is_chronological_and_ends_at_the_head(self, llm_settings):
        forest = chattree.Forest()
        head = add(forest, llm_settings, None, "user", "Hello?")
        started_from = head
        head = add(forest, llm_settings, head, "assistant", tool_calls=[tool_call("websearch", "call_0")])
        head = add(forest, llm_settings, head, "tool", "result", generation_metadata={"function_name": "websearch"})
        head = add(forest, llm_settings, head, "assistant", "Hi.")

        record = agent.describe_turn(forest, head, since_node_id=started_from)
        assert [message["role"] for message in record.messages] == ["assistant", "tool", "assistant"]
        assert record.node_ids[-1] == head == record.head_node_id
        assert len(record.node_ids) == len(record.messages) == 3

    def test_a_span_ending_on_a_tool_node_has_no_reply(self, llm_settings):
        # An empty reply is a real diagnostic signal, and this is one of the three things it can mean.
        # `messages` is what tells them apart, which is why the record carries it.
        forest = chattree.Forest()
        head = add(forest, llm_settings, None, "user", "Hello?")
        started_from = head
        head = add(forest, llm_settings, head, "assistant", tool_calls=[tool_call("websearch", "call_0")])
        head = add(forest, llm_settings, head, "tool", "result", generation_metadata={"function_name": "websearch"})

        record = agent.describe_turn(forest, head, since_node_id=started_from)
        assert record.reply == ""
        assert record.messages[-1]["role"] == "tool"

    def test_reasoning_is_collected_in_order_and_only_where_there_was_some(self, llm_settings):
        forest = chattree.Forest()
        head = add(forest, llm_settings, None, "user", "Hello?")
        started_from = head
        head = add(forest, llm_settings, head, "assistant", reasoning_content="First I should search.",
                   tool_calls=[tool_call("websearch", "call_0")])
        head = add(forest, llm_settings, head, "tool", "result", generation_metadata={"function_name": "websearch"})
        head = add(forest, llm_settings, head, "assistant", "Hi.", reasoning_content="Now I can answer.")

        record = agent.describe_turn(forest, head, since_node_id=started_from)
        assert record.reasoning == ("First I should search.", "Now I can answer.")

    def test_grounded_is_none_when_the_reply_did_not_record_it(self, llm_settings):
        # `ai_turn` records the marker only when the documents were in play or an attachment was present,
        # so absent means "nothing to say" -- distinct from a recorded `False`.
        forest = chattree.Forest()
        head = add(forest, llm_settings, None, "user", "What is 2+2?")
        started_from = head
        unmarked = add(forest, llm_settings, head, "assistant", "Four.")
        assert agent.describe_turn(forest, unmarked, since_node_id=started_from).grounded is None

        marked = add(forest, llm_settings, head, "assistant", "Four.",
                     generation_metadata={"grounded": False})
        assert agent.describe_turn(forest, marked, since_node_id=started_from).grounded is False

    def test_a_record_built_from_a_stored_branch_carries_no_prompts(self, llm_settings):
        # Nothing on disk records what went on the wire, so the field is empty rather than reconstructed.
        forest = chattree.Forest()
        head = add(forest, llm_settings, None, "user", "Hello?")
        head = add(forest, llm_settings, head, "assistant", "Hi.")
        assert agent.describe_turn(forest, head).prompts == ()


# ---------------------------------------------------------------------------
# The turn itself
# ---------------------------------------------------------------------------

class TestTurn:
    """`turn` runs the real `scaffold.ai_turn`; only the backend seam is faked."""

    def test_the_one_liner_starts_a_conversation_and_returns_the_reply(self, monkeypatch, llm_settings):
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: make_invoke_result(content="Hello from the LLM."))

        record = agent.turn(llm_settings, "Hello?")

        assert record.reply == "Hello from the LLM."
        assert record.rounds == 0
        assert record.tool_calls == {}
        # The span is the assistant's turn: the user's message is what prompted it, not part of it.
        assert [message["role"] for message in record.messages] == ["assistant"]

    def test_a_datastore_with_nodes_in_it_will_not_be_reset_by_accident(self, monkeypatch, llm_settings):
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: make_invoke_result(content="Hi."))
        datastore = chattree.Forest()
        first = agent.turn(llm_settings, "Hello?", datastore=datastore)

        with pytest.raises(ValueError):
            agent.turn(llm_settings, "Again?", datastore=datastore)
        # ...and the conversation is still there, which is the point of refusing.
        assert datastore.get_payload(first.head_node_id)["message"]["role"] == "assistant"

    def test_an_attached_document_reaches_the_wire(self, monkeypatch, llm_settings, tmp_path):
        # The scripted counterpart of dragging a file into the chat. A document has no native wire form:
        # it is stored verbatim as a sidecar, and its text is folded into the message at wire-build time --
        # which is what lets any model use one, vision capability or not. So the assertion that matters is
        # about the prompt, not about the node.
        from unpythonic.env import env  # noqa: PLC0415 -- only the attachment tests build a staged entry

        def fake_invoke(**kw):
            kw["on_prompt_ready"](llmclient.serialize_history_for_wire(kw["settings"], kw["history"],
                                                                       continue_=kw["continue_"],
                                                                       datastore=kw["datastore"]))
            return make_invoke_result(content="It says the stack draws 47.2 kWh/kg.")
        monkeypatch.setattr("raven.librarian.llmclient.invoke", fake_invoke)

        staged = env(raw=b"The Kelvin-7 stack recorded 47.2 kWh/kg at nominal load.",
                     name="kelvin7.txt", provenance_url=None, provenance_source="user_attachment")
        datastore = chattree.PersistentForest(tmp_path / "chat.json")

        record = agent.turn(llm_settings, "What does this say?", staged_files=[staged],
                            datastore=datastore)

        wire = "\n".join(chatutil.content_to_text(message.get("content")) for message in record.prompts[-1])
        assert "47.2 kWh/kg" in wire
        # ...and it is a stored attachment rather than pasted text: the sidecar is on disk, content-addressed.
        assert datastore.list_sidecar_files()

    def test_attaching_to_an_in_memory_datastore_is_refused_up_front(self, llm_settings):
        # The sidecar store is a directory beside the datastore file, so `chattree.Forest` has none — and
        # the default datastore *is* a `Forest`. Without the guard this fails inside `imagestore`, on a
        # missing method, after the turn has already begun.
        from unpythonic.env import env  # noqa: PLC0415 -- only this test builds a staged attachment

        staged = env(raw=b"not really a png", provenance_url=None, provenance_source="user_attachment")
        with pytest.raises(ValueError):
            agent.turn(llm_settings, "What is in this image?", staged_images=[staged])

    def test_images_are_refused_on_a_model_that_cannot_see_them(self, llm_settings, tmp_path):
        # A batch feeding page images to a text-only model would pay for every call and get an answer about
        # nothing. Librarian's attach button already refuses this; a script had nothing.
        from unpythonic.env import env  # noqa: PLC0415 -- only the attachment tests build a staged entry

        staged = env(raw=b"not really a png", provenance_url=None, provenance_source="user_attachment")
        datastore = chattree.PersistentForest(tmp_path / "chat.json")

        llm_settings.model_is_vlm = False
        with pytest.raises(ValueError):
            agent.turn(llm_settings, "What is in this image?", staged_images=[staged], datastore=datastore)

        # `None` is "the backend did not say", not "no" -- refusing on it would block every backend that
        # reports nothing, so it must get past the guard. (It fails later, on the fake image bytes.)
        llm_settings.model_is_vlm = None
        with pytest.raises(Exception) as excinfo:
            agent.turn(llm_settings, "What is in this image?", staged_images=[staged], datastore=datastore)
        assert "image input" not in str(excinfo.value)

    def test_a_second_turn_continues_from_the_first_record(self, monkeypatch, llm_settings):
        # The record carries the datastore as well as the head, so a script that looks at the answer before
        # deciding what to ask next can continue -- including from the one-liner form, whose datastore it
        # never named and would otherwise have no way back to.
        replies = iter([make_invoke_result(content="First answer."),
                        make_invoke_result(content="Second answer.")])
        monkeypatch.setattr("raven.librarian.llmclient.invoke", lambda **kw: next(replies))

        first = agent.turn(llm_settings, "First question?")
        second = agent.turn(llm_settings, "Second question?", datastore=first.datastore,
                            head_node_id=first.head_node_id)
        datastore = first.datastore

        assert first.reply == "First answer."
        assert second.reply == "Second answer."
        # Each record describes its own turn, so the second does not re-count the first.
        assert len(second.messages) == 1
        # ...and they are one conversation, not two: system, greeting, Q, A, Q, A.
        assert [message["role"] for message in chatutil.linearize_chat(datastore=datastore,
                                                                      node_id=second.head_node_id)] == \
            ["system", "assistant", "user", "assistant", "user", "assistant"]

    def test_running_the_same_head_again_branches_rather_than_appends(self, monkeypatch, llm_settings):
        # What the GUI calls reroll, and what a script sampling one turn several times needs: no user
        # message, and a head that already has a reply under it. The chat is a tree, so the second answer
        # is a sibling of the first and both survive.
        replies = iter([make_invoke_result(content="One answer."),
                        make_invoke_result(content="Another answer.")])
        monkeypatch.setattr("raven.librarian.llmclient.invoke", lambda **kw: next(replies))

        first = agent.turn(llm_settings, "Question?")
        asked_at = first.datastore.get_parent(first.node_ids[0])  # the user's message
        again = agent.turn(llm_settings, datastore=first.datastore, head_node_id=asked_at)

        assert first.reply == "One answer."
        assert again.reply == "Another answer."
        assert set(first.datastore.get_children(asked_at)) == {first.head_node_id, again.head_node_id}

    def test_a_tool_round_is_one_round_and_its_calls_are_counted_by_name(self, monkeypatch, llm_settings):
        responses = iter([make_invoke_result(content="",
                                             tool_calls=[tool_call("websearch", "call_0"),
                                                         tool_call("websearch", "call_1")]),
                          make_invoke_result(content="Here is what I found.")])
        monkeypatch.setattr("raven.librarian.llmclient.invoke", lambda **kw: next(responses))
        monkeypatch.setattr("raven.librarian.llmclient.perform_tool_calls",
                            lambda settings, message, on_call_start, on_call_done, **kw:
                                [make_tool_response(content="a raven is a bird", tool_call_id="call_0"),
                                 make_tool_response(content="a raven is also an app", tool_call_id="call_1")])

        record = agent.turn(llm_settings, "What is a raven?", internet_enabled=True)

        assert record.rounds == 1
        assert record.tool_calls == {"websearch": 2}
        assert record.reply == "Here is what I found."
        assert [message["role"] for message in record.messages] == ["assistant", "tool", "tool", "assistant"]

    def test_the_record_carries_the_prompt_of_every_model_call(self, monkeypatch, llm_settings):
        responses = iter([make_invoke_result(content="", tool_calls=[tool_call("websearch", "call_0")]),
                          make_invoke_result(content="Here is what I found.")])

        def fake_invoke(**kw):
            # `on_prompt_ready` fires inside `invoke`, with the history in its wire form — which is where
            # the record's `prompts` come from, so a fake that skips it would leave nothing to assert on.
            # The serializer is the real one; it touches no network.
            kw["on_prompt_ready"](llmclient.serialize_history_for_wire(kw["settings"], kw["history"],
                                                                       continue_=kw["continue_"]))
            return next(responses)
        monkeypatch.setattr("raven.librarian.llmclient.invoke", fake_invoke)
        monkeypatch.setattr("raven.librarian.llmclient.perform_tool_calls",
                            lambda settings, message, on_call_start, on_call_done, **kw:
                                [make_tool_response(content="a raven is a bird")])

        record = agent.turn(llm_settings, "What is a raven?", internet_enabled=True)

        # One per model call, in order, so `prompts[-1]` is the one that produced the reply. This is what a
        # script asserting on "what was actually sent" reads, in place of catching `on_prompt_ready`.
        assert len(record.prompts) == 2
        assert record.prompts[-1][0]["role"] == "system"
        assert any("What is a raven?" in str(message.get("content")) for message in record.prompts[-1])
        # The second call carries the tool exchange the first one did not.
        assert len(record.prompts[-1]) > len(record.prompts[0])

    def test_the_network_tools_are_withheld_unless_asked_for(self, monkeypatch, llm_settings):
        offered = []

        def fake_invoke(**kw):
            offered.append(kw["tool_names"])
            return make_invoke_result(content="Hi.")
        monkeypatch.setattr("raven.librarian.llmclient.invoke", fake_invoke)

        agent.turn(llm_settings, "Hello?")
        agent.turn(llm_settings, "Hello?", internet_enabled=True)

        # A run with the tools enabled performs *real* calls, so the surface's default has to be the
        # cautious one -- unlike the apps, whose user is watching.
        default_offer, explicit_offer = offered
        assert set(llm_settings.network_tool_names).isdisjoint(default_offer)
        assert set(llm_settings.network_tool_names).issubset(explicit_offer)

    def test_the_automatic_search_uses_the_users_own_words_by_default(self, monkeypatch, llm_settings):
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: make_invoke_result(content="Hi."))
        retriever = FakeRetriever(results=[])

        agent.turn(llm_settings, "What is a raven?", retriever=retriever)

        assert [call["q"] for call in retriever.calls] == ["What is a raven?"]

    def test_no_retriever_means_no_query_to_ignore(self, monkeypatch, llm_settings, caplog):
        # The default query is the user's own words, so without a corpus it would reach `ai_turn` as a
        # query that cannot be run -- one warning per call of the form this surface exists to make easy.
        monkeypatch.setattr("raven.librarian.llmclient.invoke",
                            lambda **kw: make_invoke_result(content="Hi."))

        with caplog.at_level(logging.WARNING):
            agent.turn(llm_settings, "Hello?")

        assert "docs_query" not in caplog.text

    def test_docs_query_none_searches_nothing_but_keeps_the_document_tools(self, monkeypatch, llm_settings):
        offered = []

        def fake_invoke(**kw):
            offered.append(kw["tool_names"])
            return make_invoke_result(content="Hi.")
        monkeypatch.setattr("raven.librarian.llmclient.invoke", fake_invoke)
        retriever = FakeRetriever(results=[])

        agent.turn(llm_settings, "What is a raven?", retriever=retriever, docs_query=None)

        assert retriever.calls == []
        # The control arm for "what is the automatic search worth?" must still let the model search itself,
        # or it measures the tools' absence instead.
        assert set(llm_settings.document_tool_names).issubset(offered[0])


class TestTheSurfaceIsDeclared:
    """A scripting surface that is only nominally public breaks its callers silently.

    Re-privatizing a name or dropping a record field is not a signature change, so nothing else in the
    suite would notice; a probe written against it would fail at run time, months later, on a machine
    without the context to diagnose it.
    """

    def test_the_names_a_script_uses_are_exported(self):
        for name in ("TurnRecord", "describe_turn", "turn"):
            assert name in agent.__all__

    def test_the_record_names_what_a_probe_asserts_on(self):
        fields = {field.name for field in dataclasses.fields(agent.TurnRecord)}
        assert fields == {"datastore", "head_node_id", "node_ids", "messages", "reply", "reasoning",
                          "rounds", "tool_calls", "grounded", "prompts"}

    def test_the_record_is_frozen(self, llm_settings):
        forest = chattree.Forest()
        head = add(forest, llm_settings, None, "assistant", "Hi.")
        record = agent.describe_turn(forest, head)
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.rounds = 99
