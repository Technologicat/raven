"""Unit tests for raven.librarian.chat_controller.

Only the datastore-side helpers so far — the ones that decide which of a message's buttons are live. Those
are pure functions over a `Forest` and need no GUI. (The descent they used to sit beside now lives in
`chatutil.descend_to_latest`, and is tested there.)

The module they live in needs rather more, though, so the skip names the module under test rather than any
one of its dependencies. Two of the paths that used to force this are gone — the avatar controller is a
`TYPE_CHECKING` import now, and `raven.client.api` is reached through `chat_controller._client_api()` — but
several remain, and `python scripts/check_ci_imports.py` names them: `hybridir` (bm25s, chromadb, watchdog),
`scaffold`'s own route to `raven.client.api` (spaCy), the audio player (pygame), the codec (av). Clearing
those is a dependency-hygiene sweep across several modules rather than a change to this one, and until it
happens these tests do not run in the minimal-dependency CI job even though they would pass there.
"""

import threading
import types

import pytest

pytest.importorskip("raven.librarian.chat_controller")  # noqa: E402 -- still reaches the ML stack; see above

from raven.librarian import chat_controller  # noqa: E402


def message_over(forest):
    """A real `DPGChatMessage` with no widgets, for the methods that only read the datastore.

    Built with `__new__` rather than a stand-in object so the methods under test find each other through
    the real class — `_clipboard_text` calls `_document_body`, and a duck-typed namespace would have to
    supply the second by hand, which is the collaboration worth checking. `__init__` is skipped because it
    builds DPG widgets; the only attribute these paths reach for is the datastore, through `parent_view`.
    """
    message = chat_controller.DPGChatMessage.__new__(chat_controller.DPGChatMessage)
    message.parent_view = types.SimpleNamespace(
        chat_controller=types.SimpleNamespace(datastore=forest))
    return message


class TestGreetingNodeIds:
    def test_every_greeting_under_every_card_is_listed(self, two_card_forest):
        f, _card1, _card2, greeting1, greeting2, message = two_card_forest
        greeting_node_ids = chat_controller._get_all_greeting_node_ids(datastore=f)
        assert set(greeting_node_ids) == {greeting1, greeting2}
        assert message not in greeting_node_ids

    def test_the_same_question_gets_the_same_answer_every_time(self, two_card_forest):
        # The four button gates on one message -- reroll, continue, branch, delete -- each ask this list
        # whether the message is a greeting, and they ask the object they were handed once. A lazily
        # evaluated answer is consumed by the first question, so the rest are answered from the leftovers,
        # which reads as "no": a greeting's delete button comes up live and deletes the chat under it.
        f, _card1, _card2, greeting1, _greeting2, _message = two_card_forest
        greeting_node_ids = chat_controller._get_all_greeting_node_ids(datastore=f)
        assert [greeting1 in greeting_node_ids for _ in range(4)] == [True] * 4

    def test_a_deleted_card_takes_its_greetings_out_of_the_list(self, two_card_forest):
        # The root scan underneath is memoized, so this is really asking whether the memo is filtered
        # against the live nodes. It has to be: `get_children` raises on a node that is gone.
        f, _card1, card2, greeting1, greeting2, _message = two_card_forest
        f.delete_subtree(card2)
        greeting_node_ids = chat_controller._get_all_greeting_node_ids(datastore=f)
        assert greeting1 in greeting_node_ids
        assert greeting2 not in greeting_node_ids

    def test_a_message_the_user_sent_is_not_a_greeting_however_it_sits(self, two_card_forest, chat_payload):
        # HEAD can rest on a system prompt node — deleting another card lands there — and a message sent
        # from there attaches beside the greetings. Going by position alone would call it one, and a
        # greeting has its reroll, continue, branch and delete buttons taken away: the user would be left
        # with a message of their own that they cannot remove.
        f, card1, _card2, _greeting1, _greeting2, _message = two_card_forest
        typed_at_the_root = f.create_node(chat_payload("user", "sent with HEAD on the card", 3), parent_id=card1)
        assert typed_at_the_root not in chat_controller._get_all_greeting_node_ids(datastore=f)


class TestSystemPromptNodeIds:
    def test_every_root_is_a_system_prompt_node(self, two_card_forest):
        f, card1, card2, _greeting1, _greeting2, _message = two_card_forest
        assert set(chat_controller._get_all_system_prompt_node_ids(datastore=f)) == {card1, card2}

    def test_a_deleted_card_drops_out(self, two_card_forest):
        f, card1, card2, _greeting1, _greeting2, _message = two_card_forest
        f.delete_subtree(card2)
        assert chat_controller._get_all_system_prompt_node_ids(datastore=f) == [card1]


class TestFormatGenerationStats:
    """The chat log's `[900t, 22.0s, 40.9t/s]` line, which two different readouts share."""

    def test_an_exact_count_carries_no_tilde(self):
        assert chat_controller.format_generation_stats(n_tokens=900, dt=22.0) == "[900t, 22.00s, 40.91t/s]"

    def test_an_estimate_says_so_on_both_derived_figures(self):
        # The speed is only as exact as the count it comes from, so one `~` on the tokens would understate
        # how much of the line is a guess.
        out = chat_controller.format_generation_stats(n_tokens=900, dt=22.0, exact=False)
        assert out == "[~900t, 22.00s, ~40.91t/s]"

    def test_no_speed_where_there_is_no_time_to_divide_by(self):
        # A turn that thought and then asked for a tool has no answer phase; its leftover tokens are real
        # and its duration is not. `0.00t/s` would state a measurement nobody made.
        out = chat_controller.format_generation_stats(n_tokens=26, dt=0.0)
        assert out == "[26t, 0.00s]"
        assert "t/s" not in out

    def test_the_label_goes_inside_the_brackets(self):
        out = chat_controller.format_generation_stats(n_tokens=759, dt=8.79, label="Thought for")
        assert out.startswith("[Thought for 759t,")


class TestFormatMessageMetadataLine:
    """The small grey line above a message, and the one thing it says that the icons cannot.

    A tool result's cogs badge reports that *a* tool ran. A turn calling three of them is then three
    messages with identical badges, and the name is the whole of what separates them — so it is on the
    line, spelled as the chat graph spells it in a box's speaker line.
    """

    def _payload(self, generation_metadata=None):
        payload = {"general_metadata": {"datetime": "2026-09-04 07:52:48"}}
        if generation_metadata is not None:
            payload["generation_metadata"] = generation_metadata
        return payload

    def test_an_ordinary_message_says_when_and_which_revision(self):
        line = chat_controller.format_message_metadata_line(self._payload(), "assistant", 1)
        assert line == "2026-09-04 07:52:48 R1"

    def test_a_tool_result_names_the_tool(self):
        line = chat_controller.format_message_metadata_line(
            self._payload({"function_name": "websearch"}), "tool", 1)
        assert line == "2026-09-04 07:52:48 R1 [websearch]"

    def test_a_tool_result_that_recorded_no_tool_says_only_when(self):
        # A call that failed before it had a function to name records none, and so does anything written
        # before the field existed. "[None]" would be worse than the bare line.
        line = chat_controller.format_message_metadata_line(self._payload(), "tool", 1)
        assert line == "2026-09-04 07:52:48 R1"

    def test_only_a_tool_result_is_named(self):
        # The control, and it needs a payload that *has* the field: every other role reaches this with no
        # `function_name` to find, so a check that forgot to test the role would pass on them anyway. An
        # assistant message carries `generation_metadata` of its own, which is where one could come from.
        line = chat_controller.format_message_metadata_line(
            self._payload({"function_name": "websearch"}), "assistant", 1)
        assert line == "2026-09-04 07:52:48 R1", "a non-tool message was captioned with a tool name"


class TestDocumentBody:
    """What a tool result *actually* says, as against the excerpt of it the log has room for.

    A fetched page over `tool_result_attachment_threshold` is moved to a sidecar and the stored message
    content is replaced by an 800-character excerpt plus a chip. So the payload is lossy, and anything
    handing the reader "this message" — the expand toggle, the clipboard — has to go to the sidecar for
    the rest of it. On the base class rather than on `DPGCompleteChatMessage` because the copy button is,
    and a streaming message would otherwise raise on its own copy callback.

    Driven through a duck-typed `self`: the method reaches for nothing but the datastore, which is what
    makes it checkable with no DPG context and no widget.
    """

    def _payload_with_document(self, forest, role, body):
        from raven.librarian import chatutil, textfilestore
        stored = textfilestore.store_file_as_sidecar(datastore=forest,
                                                     file_source=body.encode("utf-8"),
                                                     name="a page.md",
                                                     provenance_url="https://example.invalid/page",
                                                     provenance_source="tool_result",
                                                     content_type="text/markdown")
        return {"message": {"role": role,
                            "content": [chatutil.text_content_part("the opening of it…"), stored.part]}}

    def test_a_tool_result_reports_the_whole_document(self, in_memory_forest):
        body = "\n".join(f"line {k} of the fetched page" for k in range(500))
        payload = self._payload_with_document(in_memory_forest, "tool", body)
        got = message_over(in_memory_forest)._document_body(payload)
        assert got == body
        assert got != "the opening of it…", "the excerpt came back, so the sidecar was not consulted"

    def test_a_user_attachment_is_not_the_message(self, in_memory_forest):
        # Load-bearing, not tidiness: a user message with an attached document has a `text_file` part too,
        # and its text part is what the person wrote. Reporting the attachment as the body would replace
        # their words with an excerpt of the file — and, through the copy button, put the file on the
        # clipboard instead of the question.
        payload = self._payload_with_document(in_memory_forest, "user", "a paper they attached")
        assert message_over(in_memory_forest)._document_body(payload) is None

    def test_a_message_with_no_document_reports_none(self):
        from raven.librarian import chatutil
        payload = {"message": {"role": "tool", "content": [chatutil.text_content_part("12:00")]}}
        assert message_over(None)._document_body(payload) is None

    def test_an_unreadable_sidecar_degrades_rather_than_raising(self, in_memory_forest):
        # The stored excerpt is then what renders, and the copy carries it. Less than we wanted; never a
        # message that shows nothing, and never an exception out of a button callback.
        payload = {"message": {"role": "tool",
                               "content": [{"type": "text_file",
                                            "text_file": {"url": "sidecar:nothing-is-here.md",
                                                          "name": "gone.md"}}]}}
        assert message_over(in_memory_forest)._document_body(payload) is None


class TestClipboardText:
    """What the copy button puts on the clipboard, which is not always what the log has room to show."""

    def test_a_truncated_tool_result_copies_whole(self, in_memory_forest):
        # The reported bug: the log shows an excerpt of a fetched page, and copying handed back exactly
        # that excerpt. The document is right there in the sidecar, and it is what the reader asked for.
        from raven.librarian import chatutil, textfilestore
        body = "\n".join(f"line {k} of the fetched page" for k in range(500))
        stored = textfilestore.store_file_as_sidecar(datastore=in_memory_forest,
                                                     file_source=body.encode("utf-8"),
                                                     name="a page.md",
                                                     provenance_url="https://example.invalid/page",
                                                     provenance_source="tool_result",
                                                     content_type="text/markdown")
        payload = {"message": {"role": "tool",
                               "content": [chatutil.text_content_part("the opening of it…"), stored.part]}}
        got = message_over(in_memory_forest)._clipboard_text(payload)
        assert got == body
        assert "the opening of it…" not in got, "the excerpt came along, so the message was not replaced"

    def test_an_ordinary_message_copies_what_is_stored(self, in_memory_forest):
        # The control. Only the attachmentified case reads a sidecar; everything else must keep taking the
        # route it always took, or a copy of a question would start returning something else entirely.
        from raven.librarian import chatutil
        payload = {"message": {"role": "user",
                               "content": [chatutil.text_content_part("what is the square root of 10?")]}}
        got = message_over(in_memory_forest)._clipboard_text(payload)
        assert got == "what is the square root of 10?"


class TestDemolishLeavesNoWidgetReference:
    """`demolish` must clear every widget reference `build` made, because the instance may be rebuilt.

    Only reachable through demolish-then-rebuild of the same instance. A reference that survives is not
    inert: `_thought_bubble` reads a non-`None` `gui_thought_group` as "already built" and hands the
    deleted id back to the renderer as the parent to draw into, which cannot work.

    Deliberately structural rather than a rendered-widget test: it needs no DPG context, and it keeps
    holding when a future `build` adds a sixth widget attribute, which is the case a screenshot test
    would silently stop covering.
    """

    # What `build` populates, per the declarations in `DPGChatMessage.__init__`.
    BUILT_BY_BUILD = ("gui_text_group", "gui_thought_button", "gui_thought_group", "gui_thought_stats",
                      "gui_keyboard_mark_widget", "gui_buttons_group")

    @staticmethod
    def _demolished_message(monkeypatch):
        """A bare `DPGChatMessage` with every widget reference set, put through `demolish`."""
        monkeypatch.setattr(chat_controller.dpg, "delete_item", lambda *args, **kwargs: None)

        message = object.__new__(chat_controller.DPGChatMessage)
        message.paragraphs_lock = threading.RLock()
        message.paragraphs = [{"text": "hi", "is_thought": False, "rendered": True, "widget": 11}]
        message.owned_handler_registries = []
        message.owned_tooltips = []
        message.gui_parent = "some_container"
        message.gui_container_group = 1000
        message.role = "assistant"
        message.persona = "Aria"
        message.gui_button_callbacks = {"reroll": lambda: None}
        for n, name in enumerate(TestDemolishLeavesNoWidgetReference.BUILT_BY_BUILD):
            setattr(message, name, 2000 + n)

        message.demolish()
        return message

    def test_every_widget_reference_build_made_is_cleared(self, monkeypatch):
        message = self._demolished_message(monkeypatch)
        left_behind = [name for name in self.BUILT_BY_BUILD if getattr(message, name) is not None]
        assert not left_behind, f"demolish left dangling widget references: {left_behind}"

    def test_the_container_it_renders_into_survives(self, monkeypatch):
        """The negative control: `demolish` empties the container, it does not forget where to render.

        Without this, "clear everything named `gui_*`" would pass the test above while breaking every
        rebuild — so the assertion that some references survive is what gives the one above its meaning.
        """
        message = self._demolished_message(monkeypatch)
        assert message.gui_container_group is not None
        assert message.gui_parent is not None


class TestIncompletenessNote:
    """A reply that stopped early has to say so, because the text cannot.

    What is kept when the user presses Stop is a reply ending mid-sentence — which is also what a model
    rambling to a halt looks like, and what a reply that simply finished tersely looks like. By the next
    session nobody remembers pressing the button.
    """

    def test_a_finished_reply_says_nothing(self):
        assert chat_controller._incompleteness_note({"model": "m", "n_tokens": 100, "dt": 2.0}) is None

    def test_a_node_with_no_metadata_at_all_says_nothing(self):
        """A message Raven authored — a backend-error report — carries no `generation_metadata`."""
        assert chat_controller._incompleteness_note({}) is None

    def test_a_stopped_reply_says_it_was_stopped(self):
        note = chat_controller._incompleteness_note({"n_tokens": 30, "dt": 1.0, "interrupted": True})
        assert note is not None
        assert "Interrupted" in note

    def test_a_reply_cut_off_by_the_app_going_away_says_so_instead(self):
        """Different event, different words: `status: incomplete` is only reachable from disk."""
        note = chat_controller._incompleteness_note({"status": "incomplete"})
        assert note is not None
        assert "Interrupted" not in note, "the two cases must not collapse into one message"
        assert "Raven exited" in note

    def test_being_stopped_wins_over_the_leftover_marker(self):
        """Belt and braces: a payload carrying both is describing a stopped reply, which is the specific one."""
        note = chat_controller._incompleteness_note({"interrupted": True, "status": "incomplete"})
        assert "Interrupted" in note


class TestRemovingAMessageByNode:
    """A turn tidying up after its own round must name the *live* message, not merely the node.

    One node is rendered by two different widgets over its lifetime: the live message while the reply
    streams, and the stored one `on_done` swaps in when it finishes. The cleanup that follows a round runs
    either way — a round can end without `on_done`, aborted or failed — so it cannot assume the live widget
    is still what renders that node.

    The failure is silent and reads as the reply never arriving: the finished message is created, taken off
    screen a moment later by the turn's own epilogue, and nothing is logged because removing a message that
    is there is not an error.

    Structural rather than rendered, so it needs no DPG context: what is under test is which message the
    search picks, and `demolish` is exactly the boundary where the view stops and the toolkit begins.
    """

    @staticmethod
    def _message(cls, node_id: str):
        message = object.__new__(cls)
        message.node_id = node_id
        return message

    @staticmethod
    def _view_showing(monkeypatch, *messages):
        """A bare view whose chat history is `messages`, plus the list `demolish` records into."""
        demolished = []
        for cls in (chat_controller.DPGStreamingChatMessage, chat_controller.DPGCompleteChatMessage):
            monkeypatch.setattr(cls, "demolish", lambda self: demolished.append(self))

        view = object.__new__(chat_controller.DPGLinearizedChatView)
        view.chat_controller = types.SimpleNamespace(current_chat_history=list(messages),
                                                     current_chat_history_lock=threading.RLock())
        return view, demolished

    def test_the_live_message_goes(self, monkeypatch):
        live = self._message(chat_controller.DPGStreamingChatMessage, "ai")
        view, demolished = self._view_showing(monkeypatch, live)

        view.remove_streaming_message_for("ai")

        assert view.chat_controller.current_chat_history == []
        assert demolished == [live]

    def test_the_stored_message_that_replaced_it_stays(self, monkeypatch):
        stored = self._message(chat_controller.DPGCompleteChatMessage, "ai")
        view, demolished = self._view_showing(monkeypatch, stored)

        view.remove_streaming_message_for("ai")

        assert view.chat_controller.current_chat_history == [stored], "the finished reply was taken off screen"
        assert demolished == []

    def test_removing_by_node_alone_reaches_the_stored_message(self, monkeypatch):
        """The negative control: by node alone, that same call *does* take the stored message.

        Without this, a fixture in which nothing could be removed at all would satisfy the assertion above
        for the wrong reason — and the distinction the two calls draw is the entire point of having both.
        """
        stored = self._message(chat_controller.DPGCompleteChatMessage, "ai")
        view, demolished = self._view_showing(monkeypatch, stored)

        view.remove_message_for("ai")

        assert view.chat_controller.current_chat_history == []
        assert demolished == [stored]
