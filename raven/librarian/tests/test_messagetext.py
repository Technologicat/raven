"""Unit tests for raven.librarian.messagetext.

How a stored chat message reads as text: the grey line, the generation statistics, the incompleteness note,
and what a copy of one message or an export of the whole log carries.
"""

from raven.librarian import messagetext


class TestFormatGenerationStats:
    """The chat log's `[900t, 22.0s, 40.9t/s]` line, which two different readouts share."""

    def test_an_exact_count_carries_no_tilde(self):
        assert messagetext.format_generation_stats(n_tokens=900, dt=22.0) == "[900t, 22.00s, 40.91t/s]"

    def test_an_estimate_says_so_on_both_derived_figures(self):
        # The speed is only as exact as the count it comes from, so one `~` on the tokens would understate
        # how much of the line is a guess.
        out = messagetext.format_generation_stats(n_tokens=900, dt=22.0, exact=False)
        assert out == "[~900t, 22.00s, ~40.91t/s]"

    def test_no_speed_where_there_is_no_time_to_divide_by(self):
        # A turn that thought and then asked for a tool has no answer phase; its leftover tokens are real
        # and its duration is not. `0.00t/s` would state a measurement nobody made.
        out = messagetext.format_generation_stats(n_tokens=26, dt=0.0)
        assert out == "[26t, 0.00s]"
        assert "t/s" not in out

    def test_the_label_goes_inside_the_brackets(self):
        out = messagetext.format_generation_stats(n_tokens=759, dt=8.79, label="Thought for")
        assert out.startswith("[Thought for 759t,")


class TestFormatMessageMetadataLine:
    """The small grey line above a message, and the thing it says that the icons cannot: what produced it.

    A tool result's cogs badge reports that *a* tool ran, so a turn calling three of them is three
    messages with identical badges. A reply's icon says the AI wrote it and not which model — recorded per
    message, and until now readable only by hovering the generation-stats line, though it varies across a
    tree and along one branch alike: two rerolls from different models, or the loaded model swapped
    mid-conversation.

    One bracket for both, spelled as the chat graph spells it in a box's speaker line.
    """

    def _payload(self, generation_metadata=None):
        payload = {"general_metadata": {"datetime": "2026-09-04 07:52:48"}}
        if generation_metadata is not None:
            payload["generation_metadata"] = generation_metadata
        return payload

    def test_an_ordinary_message_says_when_and_which_revision(self):
        line = messagetext.format_message_metadata_line(self._payload(), "assistant", 1)
        assert line == "2026-09-04 07:52:48 R1"

    def test_a_tool_result_names_the_tool(self):
        line = messagetext.format_message_metadata_line(
            self._payload({"function_name": "websearch"}), "tool", 1)
        assert line == "2026-09-04 07:52:48 R1 [websearch]"

    def test_a_tool_result_that_recorded_no_tool_says_only_when(self):
        # A call that failed before it had a function to name records none, and so does anything written
        # before the field existed. "[None]" would be worse than the bare line.
        line = messagetext.format_message_metadata_line(self._payload(), "tool", 1)
        assert line == "2026-09-04 07:52:48 R1"

    def test_only_a_tool_result_is_named(self):
        # The control, and it needs a payload that *has* the field: every other role reaches this with no
        # `function_name` to find, so a check that forgot to test the role would pass on them anyway. An
        # assistant message carries `generation_metadata` of its own, which is where one could come from.
        line = messagetext.format_message_metadata_line(
            self._payload({"function_name": "websearch"}), "assistant", 1)
        assert line == "2026-09-04 07:52:48 R1", "a non-tool message was captioned with a tool name"

    def test_a_reply_names_the_model_that_wrote_it(self):
        # Recorded per message and, until now, readable only by hovering the generation-stats line. It
        # varies across a tree and along a branch alike — two rerolls from different models, or the loaded
        # model swapped mid-conversation — which is exactly what a per-app readout cannot say.
        line = messagetext.format_message_metadata_line(
            self._payload({"model": "qwen3.5-4b, Q4_K_XL, 128 Ki context"}), "assistant", 1)
        assert line == "2026-09-04 07:52:48 R1 [qwen3.5-4b, Q4_K_XL, 128 Ki context]"

    def test_a_reply_that_recorded_no_model_says_only_when(self):
        line = messagetext.format_message_metadata_line(self._payload(), "assistant", 1)
        assert line == "2026-09-04 07:52:48 R1"

    def test_a_users_own_message_is_not_credited_to_anything(self):
        # The control for the pair. Both branches are gated on the role, and a payload carrying both
        # fields is what would expose a gate that had stopped checking it.
        line = messagetext.format_message_metadata_line(
            self._payload({"model": "qwen3.5-4b", "function_name": "websearch"}), "user", 1)
        assert line == "2026-09-04 07:52:48 R1", "a message the user typed was credited to a producer"


class TestDocumentBody:
    """What a tool result *actually* says, as against the excerpt of it the log has room for.

    A fetched page over `tool_result_attachment_threshold` is moved to a sidecar and the stored message
    content is replaced by an 800-character excerpt plus a chip. So the payload is lossy, and anything
    handing the reader "this message" — the expand toggle, the clipboard — has to go to the sidecar for
    the rest of it.
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
        got = messagetext.document_body(in_memory_forest, payload)
        assert got == body
        assert got != "the opening of it…", "the excerpt came back, so the sidecar was not consulted"

    def test_a_user_attachment_is_not_the_message(self, in_memory_forest):
        # Load-bearing, not tidiness: a user message with an attached document has a `text_file` part too,
        # and its text part is what the person wrote. Reporting the attachment as the body would replace
        # their words with an excerpt of the file — and, through the copy button, put the file on the
        # clipboard instead of the question.
        payload = self._payload_with_document(in_memory_forest, "user", "a paper they attached")
        assert messagetext.document_body(in_memory_forest, payload) is None

    def test_a_message_with_no_document_reports_none(self):
        from raven.librarian import chatutil
        payload = {"message": {"role": "tool", "content": [chatutil.text_content_part("12:00")]}}
        assert messagetext.document_body(None, payload) is None

    def test_an_unreadable_sidecar_degrades_rather_than_raising(self, in_memory_forest):
        # The stored excerpt is then what renders, and the copy carries it. Less than we wanted; never a
        # message that shows nothing, and never an exception out of a button callback.
        payload = {"message": {"role": "tool",
                               "content": [{"type": "text_file",
                                            "text_file": {"url": "sidecar:nothing-is-here.md",
                                                          "name": "gone.md"}}]}}
        assert messagetext.document_body(in_memory_forest, payload) is None


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
        got = messagetext.clipboard_text(in_memory_forest, payload)
        assert got == body
        assert "the opening of it…" not in got, "the excerpt came along, so the message was not replaced"

    def test_an_ordinary_message_copies_what_is_stored(self, in_memory_forest):
        # The control. Only the attachmentified case reads a sidecar; everything else must keep taking the
        # route it always took, or a copy of a question would start returning something else entirely.
        from raven.librarian import chatutil
        payload = {"message": {"role": "user",
                               "content": [chatutil.text_content_part("what is the square root of 10?")]}}
        got = messagetext.clipboard_text(in_memory_forest, payload)
        assert got == "what is the square root of 10?"


class TestFormatForClipboard:
    """What a single-message copy produces, with and without Shift.

    Checkable at all because `format_message_for_clipboard` takes `include_node_id` rather than reading the
    keyboard: the button and the hotkey read the modifier and hand the answer down, which is the same
    split the Visualizer's report copy uses.
    """

    @staticmethod
    def _copy(forest, *, role, text, persona=None, include_node_id=False):
        """Store a one-message chat, copy it, and return `(node_id, clipboard text)`."""
        from raven.librarian import chatutil
        payload = {"message": {"role": role, "content": [chatutil.text_content_part(text)], "tool_calls": []},
                   "general_metadata": {"persona": persona, "timestamp": 0, "datetime": "2026-09-10 12:00:00"}}
        node_id = forest.create_node(payload, parent_id=None)
        return node_id, messagetext.format_message_for_clipboard(forest, node_id, role=role, persona=persona,
                                                                 include_node_id=include_node_id)

    def test_a_plain_copy_carries_the_text_and_no_node_id(self, in_memory_forest):
        _, got = self._copy(in_memory_forest, role="user", text="what is the square root of 10?")

        assert "what is the square root of 10?" in got
        assert "Node ID" not in got

    def test_shift_adds_the_node_id_and_the_revision(self, in_memory_forest):
        node_id, got = self._copy(in_memory_forest, role="user", text="what is the square root of 10?",
                                  include_node_id=True)

        assert node_id in got, "the node ID is what Shift was held for"
        assert "2026-09-10 12:00:00" in got
        assert "R1" in got, "the active revision number is missing"  # revisions are numbered from one

    def test_a_question_of_your_own_carries_no_disclosure_manifest(self, in_memory_forest):
        # There is no AI generation to disclose on a human turn, and a YAML block on a copied question
        # would only be something to delete before pasting it back into the composer.
        _, user = self._copy(in_memory_forest, role="user", text="what is the square root of 10?")
        _, ai = self._copy(in_memory_forest, role="assistant", text="About 3.1623.")

        # The control is the AI message: without it, a `format_message_for_clipboard` that never emitted a
        # manifest at all would satisfy the assertion below.
        assert "ai_generated" in ai
        assert "ai_generated" not in user


class TestExportText:
    """What a *whole-log* copy carries, which is deliberately not what a single-message copy carries.

    One message copied is a request for that message's data as it stands, so it takes the document. A log
    is a document itself, read and passed on: several fetched pages inlined at full length make it
    unreadable, and a shared log carrying complete copies of pages that were only quoted invites a
    complaint an excerpt would not. So the log keeps the excerpt — and says that it is one.
    """

    def _attachmentified(self, forest, body, name="a page.md"):
        from raven.librarian import chatutil, textfilestore
        stored = textfilestore.store_file_as_sidecar(datastore=forest,
                                                     file_source=body.encode("utf-8"),
                                                     name=name,
                                                     provenance_url="https://example.invalid/page",
                                                     provenance_source="tool_result",
                                                     content_type="text/markdown")
        return {"message": {"role": "tool",
                            "content": [chatutil.text_content_part("the opening of it…"), stored.part]}}

    def test_the_log_keeps_the_excerpt(self, in_memory_forest):
        body = "\n".join(f"line {k} of the fetched page" for k in range(500))
        payload = self._attachmentified(in_memory_forest, body)
        got = messagetext.export_text(in_memory_forest, payload)
        assert got.startswith("the opening of it…")
        assert "line 400 of the fetched page" not in got, "the whole document was inlined into the log"

    def test_the_log_says_the_excerpt_is_one(self, in_memory_forest):
        # The half that makes keeping the excerpt honest rather than merely smaller. Without it the log is
        # exactly the silent truncation the single-message copy was fixed for.
        body = "\n".join(f"line {k} of the fetched page" for k in range(500))
        payload = self._attachmentified(in_memory_forest, body, name="Example Page.md")
        got = messagetext.export_text(in_memory_forest, payload)
        assert "Excerpt." in got
        assert "Example Page.md" in got, "the note does not say which attachment it is about"
        assert f"{len(body):,}" in got, "the note does not say how much was left out"

    def test_a_message_with_no_document_is_untouched(self, in_memory_forest):
        # The control. Every ordinary message goes through this too, and a note appended to a question the
        # user typed would be a claim about an attachment that does not exist.
        from raven.librarian import chatutil
        payload = {"message": {"role": "user",
                               "content": [chatutil.text_content_part("what is the square root of 10?")]}}
        got = messagetext.export_text(in_memory_forest, payload)
        assert got == "what is the square root of 10?"

    def test_the_two_destinations_disagree_on_purpose(self, in_memory_forest):
        # Stated as one assertion because it is one decision. If these ever come out equal, either the log
        # has started inlining documents or the message copy has stopped.
        body = "\n".join(f"line {k} of the fetched page" for k in range(500))
        payload = self._attachmentified(in_memory_forest, body)
        assert messagetext.clipboard_text(in_memory_forest, payload) == body
        assert messagetext.export_text(in_memory_forest, payload) != body


class TestFormatExcerptNotice:
    def test_it_names_the_document(self):
        notice = messagetext.format_excerpt_notice(
            {"message": {"content": [{"type": "text_file",
                                      "text_file": {"url": "sidecar:x.md", "name": "Example Page.md"}}]}},
            full_length=45678)
        assert '"Example Page.md"' in notice
        assert "45,678 characters" in notice

    def test_an_unnamed_attachment_still_gets_a_notice(self):
        # A message can carry a `text_file` part with no name. Saying "the attached document" is worth more
        # than saying nothing, the useful half being that something was left out at all.
        notice = messagetext.format_excerpt_notice(
            {"message": {"content": [{"type": "text_file", "text_file": {"url": "sidecar:x.md"}}]}},
            full_length=100)
        assert "the attached document" in notice
        assert "None" not in notice, "the missing name was formatted into the notice"


class TestIncompletenessNote:
    """A reply that stopped early has to say so, because the text cannot.

    What is kept when the user presses Stop is a reply ending mid-sentence — which is also what a model
    rambling to a halt looks like, and what a reply that simply finished tersely looks like. By the next
    session nobody remembers pressing the button.
    """

    def test_a_finished_reply_says_nothing(self):
        assert messagetext.incompleteness_note({"model": "m", "n_tokens": 100, "dt": 2.0}) is None

    def test_a_node_with_no_metadata_at_all_says_nothing(self):
        """A message Raven authored — a backend-error report — carries no `generation_metadata`."""
        assert messagetext.incompleteness_note({}) is None

    def test_a_stopped_reply_says_it_was_stopped(self):
        note = messagetext.incompleteness_note({"n_tokens": 30, "dt": 1.0, "interrupted": True})
        assert note is not None
        assert "Interrupted" in note

    def test_a_reply_cut_off_by_the_app_going_away_says_so_instead(self):
        """Different event, different words: `status: incomplete` is only reachable from disk."""
        note = messagetext.incompleteness_note({"status": "incomplete"})
        assert note is not None
        assert "Interrupted" not in note, "the two cases must not collapse into one message"
        assert "Raven exited" in note

    def test_being_stopped_wins_over_the_leftover_marker(self):
        """Belt and braces: a payload carrying both is describing a stopped reply, which is the specific one."""
        note = messagetext.incompleteness_note({"interrupted": True, "status": "incomplete"})
        assert "Interrupted" in note
