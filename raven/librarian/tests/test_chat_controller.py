"""Unit tests for raven.librarian.chat_controller.

The parts reachable without building widgets: a message's teardown, and the controller's bookkeeping driven
through stand-ins. How a message reads as text is `messagetext`'s, and the pure datastore rules the buttons
are gated on are `chatutil`'s; both are tested there.

The module under test needs rather more than these tests do, so the skip names that module rather than any
one of its dependencies. Two of the paths that used to force this are gone — the avatar controller is a
`TYPE_CHECKING` import now, and `raven.client.api` is reached through `chat_controller._client_api()` — but
several remain, and `python scripts/check_ci_imports.py` names them: `hybridir` (bm25s, chromadb, watchdog),
`scaffold`'s own route to `raven.client.api` (spaCy), the audio player (pygame), the codec (av). Clearing
those is a dependency-hygiene sweep across several modules rather than a change to this one, and until it
happens these tests do not run in the minimal-dependency CI job even though they would pass there.
"""

import concurrent.futures
import threading
import time
import types

import pytest

pytest.importorskip("raven.librarian.chat_controller")  # noqa: E402 -- still reaches the ML stack; see above

from raven.common import bgtask  # noqa: E402
from raven.librarian import chat_controller, chatmessage, chatutil  # noqa: E402


class TestWhenTheContextFillReadoutIsExact:
    """The backend counts everything up to the last user message; the readout adds the rest from a local count."""

    reply = [{"role": "assistant", "content": [chatutil.text_content_part("a long reply")]}]
    with_image = [{"role": "assistant", "content": [chatutil.text_content_part("look"),
                                                    chatutil.image_content_part("sidecar:abc.png")]}]

    def test_a_negligible_tail_is_exact_however_it_was_counted(self):
        assert chat_controller.readout_is_exact(100, 10000, self.reply, tokenizer_loaded=False)

    def test_a_large_estimated_tail_is_not(self):
        # The negative control for the case below: the same tail, estimated through the ratio.
        assert not chat_controller.readout_is_exact(3000, 10000, self.reply, tokenizer_loaded=False)

    def test_a_large_counted_tail_is_exact(self):
        # A short conversation's last reply is easily over 2% of it; with a tokenizer it is counted, not guessed.
        assert chat_controller.readout_is_exact(3000, 10000, self.reply, tokenizer_loaded=True)

    def test_an_image_in_the_tail_keeps_it_an_estimate(self):
        assert not chat_controller.readout_is_exact(3000, 10000, self.with_image, tokenizer_loaded=True)


class TestWhichSearchMatchesShowInFull:
    """A collapsed document search shows a match whole when the reader opened it, or opened them all."""

    @staticmethod
    def _message(show_full_text=False, expanded_parts=()):
        message = chatmessage.DPGCompleteChatMessage.__new__(chatmessage.DPGCompleteChatMessage)
        message.show_full_text = show_full_text
        message.expanded_parts = set(expanded_parts)
        return message

    def test_one_opened_match_and_not_its_neighbours(self):
        message = self._message(expanded_parts={2})
        assert message._part_shown_in_full(2, "per_part")
        assert not message._part_shown_in_full(1, "per_part"), "opening one match opened another"

    def test_the_state_is_the_matches_own(self):
        # The message's toggle opens them all by filling the set, so it keeps no flag that could disagree.
        message = self._message(show_full_text=True)
        assert not message._part_shown_in_full(1, "per_part"), "a per-match result read the whole-message flag"
        message.expanded_parts = set(range(5))
        assert all(message._part_shown_in_full(index, "per_part") for index in range(5))

    def test_other_results_are_not_per_match_at_all(self):
        # The control: a result that is not collapsed per match shows its parts whole whatever the state says.
        assert self._message()._part_shown_in_full(3, None)


class TestOnlyALiveReplyRechecksAnAwaitedTrace:
    """A stored message being built must not spend a jump's request to open its thinking trace.

    It adds its paragraphs while it is being built, before the view holds it, so the request would be spent
    on a message not yet there to open — which is how committing a chat graph box whose count was a
    thinking-trace hit came to open nothing. A reply still being written is what the recheck is for.
    """

    @staticmethod
    def _rechecks(message_class):
        asked = []
        message = message_class.__new__(message_class)
        message.node_id = "n1"
        message.parent_view = types.SimpleNamespace(chat_controller=types.SimpleNamespace(
            search=types.SimpleNamespace(recheck_awaited_thinking_trace=asked.append)))
        message._recheck_awaited_thinking_trace()
        return asked

    def test_a_stored_message_does_not_recheck(self):
        assert self._rechecks(chatmessage.DPGCompleteChatMessage) == []

    def test_a_live_reply_does(self):
        # The control: without it, a recheck that had stopped working altogether would pass the test above.
        assert self._rechecks(chatmessage.DPGStreamingChatMessage) == ["n1"]


class TestDemolishIsATeardown:
    """`demolish` deletes every widget the message owns, its container included, and forgets them all.

    The references matter because another thread may still hold the instance: a render on its way in must
    find nothing to draw into. A reference that survives is not inert — `_thought_bubble` reads a non-`None`
    `gui_thought_group` as "already built" and hands the deleted id back to the renderer as a parent.

    The container matters because an empty group still takes a line's item spacing in the view, so one left
    standing is a gap in the chat log — which is what rerolling a tool-calling reply used to leave, one per
    message it rewound.

    Deliberately structural rather than a rendered-widget test: it needs no DPG context, and it keeps
    holding when a future `build` adds a widget attribute, which is the case a screenshot test would
    silently stop covering.
    """

    # What `build` populates, per the declarations in `DPGChatMessage.__init__`.
    BUILT_BY_BUILD = ("gui_text_group", "gui_thought_button", "gui_thought_group", "gui_thought_stats",
                      "gui_keyboard_mark_widget", "gui_buttons_group")

    @staticmethod
    def _demolished_message(monkeypatch, deleted=None):
        """A bare `DPGChatMessage` with every widget reference set, put through `demolish`.

        `deleted`: optional list, receiving each `dpg.delete_item` call as `(args, kwargs)`.
        """
        def fake_delete_item(*args, **kwargs):
            if deleted is not None:
                deleted.append((args, kwargs))
        monkeypatch.setattr(chat_controller.dpg, "delete_item", fake_delete_item)

        message = object.__new__(chatmessage.DPGChatMessage)
        message.paragraphs_lock = threading.RLock()
        message.paragraphs = [{"text": "hi", "is_thought": False, "rendered": True, "widget": 11}]
        message.owned_handler_registries = []
        message.owned_tooltips = []
        message.gui_parent = "some_container"
        message.gui_container_group = 1000
        message.role = "assistant"
        message.persona = "Aria"
        message.gui_button_callbacks = {"reroll": lambda: None}
        for n, name in enumerate(TestDemolishIsATeardown.BUILT_BY_BUILD):
            setattr(message, name, 2000 + n)

        message.demolish()
        return message

    def test_every_widget_reference_build_made_is_cleared(self, monkeypatch):
        message = self._demolished_message(monkeypatch)
        left_behind = [name for name in self.BUILT_BY_BUILD if getattr(message, name) is not None]
        assert not left_behind, f"demolish left dangling widget references: {left_behind}"

    def test_the_container_itself_is_deleted(self, monkeypatch):
        deleted = []
        message = self._demolished_message(monkeypatch, deleted)
        container_deletes = [kwargs for args, kwargs in deleted if args == (1000,)]
        assert container_deletes, "demolish never deleted the container group"
        assert not any(kwargs.get("children_only") for kwargs in container_deletes), \
            "demolish emptied the container and left it standing, which is a gap in the chat log"
        assert message.gui_container_group is None

    def test_a_demolished_message_refuses_to_build(self, monkeypatch):
        """A rebuild would need a new container, which could only go at the end of the view — the wrong place."""
        message = self._demolished_message(monkeypatch)
        with pytest.raises(RuntimeError, match="rebuild_in_place"):
            message.build(role="assistant", persona=None, node_id=None)


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
        for cls in (chatmessage.DPGStreamingChatMessage, chatmessage.DPGCompleteChatMessage):
            monkeypatch.setattr(cls, "demolish", lambda self: demolished.append(self))

        view = object.__new__(chat_controller.DPGLinearizedChatView)
        view.chat_controller = types.SimpleNamespace(current_chat_history=list(messages),
                                                     current_chat_history_lock=threading.RLock())
        return view, demolished

    def test_the_live_message_goes(self, monkeypatch):
        live = self._message(chatmessage.DPGStreamingChatMessage, "ai")
        view, demolished = self._view_showing(monkeypatch, live)

        view.remove_streaming_message_for("ai")

        assert view.chat_controller.current_chat_history == []
        assert demolished == [live]

    def test_the_stored_message_that_replaced_it_stays(self, monkeypatch):
        stored = self._message(chatmessage.DPGCompleteChatMessage, "ai")
        view, demolished = self._view_showing(monkeypatch, stored)

        view.remove_streaming_message_for("ai")

        assert view.chat_controller.current_chat_history == [stored], "the finished reply was taken off screen"
        assert demolished == []

    def test_removing_by_node_alone_reaches_the_stored_message(self, monkeypatch):
        """The negative control: by node alone, that same call *does* take the stored message.

        Without this, a fixture in which nothing could be removed at all would satisfy the assertion above
        for the wrong reason — and the distinction the two calls draw is the entire point of having both.
        """
        stored = self._message(chatmessage.DPGCompleteChatMessage, "ai")
        view, demolished = self._view_showing(monkeypatch, stored)

        view.remove_message_for("ai")

        assert view.chat_controller.current_chat_history == []
        assert demolished == [stored]


class TestChatExchange:
    """One send, one AI turn: the gate has to see a send from the moment it is accepted.

    A send whose AI turn has not started yet used to read as idle, so a second send arriving in between
    passed the gate and started a second turn on top of the first.
    """

    @staticmethod
    def _controller():
        """A controller with just what `chat_exchange` and `is_generating` touch, and a user turn that waits."""
        executor = concurrent.futures.ThreadPoolExecutor()
        controller = chat_controller.DPGChatController.__new__(chat_controller.DPGChatController)
        controller.gui_updates_safe = False
        controller.app_state = {"HEAD": "head"}
        controller.chat_exchange_task_manager = bgtask.TaskManager(name="test_exchange", mode="concurrent", executor=executor)
        controller.ai_turn_task_manager = bgtask.TaskManager(name="test_ai_turn", mode="concurrent", executor=executor)
        controller.user_turn_may_finish = threading.Event()
        controller.ai_turns = []
        controller.user_turn = lambda text, staged_images=None, staged_files=None: controller.user_turn_may_finish.wait(5.0)
        controller.ai_turn = lambda docs_query, continue_: controller.ai_turns.append(docs_query)
        return controller

    @staticmethod
    def _wait_for(predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def test_a_send_is_in_flight_before_its_turn_has_started(self):
        controller = self._controller()
        assert not controller.is_generating(), "a fresh controller already reads as busy, so this cannot tell anything"
        controller.chat_exchange("Hi!")
        assert controller.is_generating(), "a send still in its user turn read as idle, which lets a second send through"
        controller.user_turn_may_finish.set()
        assert self._wait_for(lambda: controller.ai_turns == ["Hi!"])

    def test_stopping_before_the_turn_starts_prevents_it(self):
        controller = self._controller()
        controller.chat_exchange("Hi!")
        controller.stop_ai_turn()
        controller.user_turn_may_finish.set()
        # On the exchange's own task, not on `is_generating`: that would read idle as soon as the gate forgot the
        # send, and the assertion below would then run before the exchange had had its chance to go wrong.
        assert self._wait_for(lambda: not controller.chat_exchange_task_manager.has_tasks())
        assert controller.ai_turns == [], "a stopped send still started its AI turn"

    def test_an_empty_send_does_nothing_unless_allowed(self, monkeypatch):
        # On an AI reply, which is the case the setting governs; on a user message it is always allowed
        # (`chatutil.empty_send_allowed`, tested there).
        monkeypatch.setattr(chat_controller.chatutil, "latest_user_message_text", lambda datastore, head: "earlier question")
        controller = self._controller()
        controller.datastore = chat_controller.chattree.Forest()
        controller.app_state["HEAD"] = controller.datastore.create_node(
            {"message": {"role": "assistant", "content": [chatutil.text_content_part("an answer")]},
             "general_metadata": {"persona": None}}, parent_id=None)
        controller.user_turn_may_finish.set()

        monkeypatch.setattr(chat_controller.librarian_config, "llm_allow_empty_send", False)
        controller.chat_exchange("")
        assert not controller.is_generating(), "an empty send was accepted with `llm_allow_empty_send` off"

        monkeypatch.setattr(chat_controller.librarian_config, "llm_allow_empty_send", True)
        controller.chat_exchange("")
        assert self._wait_for(lambda: controller.ai_turns == ["earlier question"]), \
            "an empty send did nothing with the setting on, so the refusal above proves nothing"


class TestDeletingASubtree:
    """The one route to a delete, which both the chat log and the chat graph take."""

    @staticmethod
    def _controller(forest, head, generating=False):
        """A controller with just what `delete_subtree` touches, counting the chat log's rebuilds."""
        controller = chat_controller.DPGChatController.__new__(chat_controller.DPGChatController)
        controller.datastore = forest
        controller.app_state = {"HEAD": head}
        controller.is_generating = lambda: generating
        controller.builds = 0
        def build():
            controller.builds += 1
        controller.view = types.SimpleNamespace(build=build)
        return controller

    def test_refused_while_a_turn_is_in_flight(self, two_card_forest, chat_payload):
        # The turn is writing into the tree, so nothing is deleted from under it.
        f, _card1, _card2, greeting1, _greeting2, message = two_card_forest
        controller = self._controller(f, message, generating=True)
        maybe_refusal = controller.delete_subtree(message)
        assert isinstance(maybe_refusal, str) and maybe_refusal
        assert controller.delete_refusal() == maybe_refusal, "a button asking first would be told something else"
        assert message in f.nodes and controller.app_state["HEAD"] == message and controller.builds == 0
        # The control: the same delete, with no turn, goes through.
        controller = self._controller(f, message)
        assert controller.delete_refusal() is None
        assert controller.delete_subtree(message) is None
        assert message not in f.nodes and controller.app_state["HEAD"] == greeting1 and controller.builds == 1

    def test_the_chat_log_is_rebuilt_only_when_its_branch_changed(self, two_card_forest, chat_payload):
        f, _card1, _card2, _greeting1, greeting2, message = two_card_forest
        elsewhere = f.create_node(chat_payload("user", "under the other card", 5), parent_id=greeting2)
        controller = self._controller(f, message)
        assert controller.delete_subtree(elsewhere) is None
        assert elsewhere not in f.nodes and controller.app_state["HEAD"] == message
        assert controller.builds == 0, "a delete nowhere near the branch on screen rebuilt the chat log"


class TestRevisingAMessage:
    """The one route to an edit: a new revision holding the new text, refused when it must be."""

    @staticmethod
    def _controller(forest, generating=False):
        """A controller with just what `revise_message` touches."""
        controller = chat_controller.DPGChatController.__new__(chat_controller.DPGChatController)
        controller.datastore = forest
        controller.is_generating = lambda: generating
        controller.update_context_fill_indicator = lambda: None
        return controller

    def test_an_edit_adds_a_revision_and_makes_it_active(self, two_card_forest):
        f, _card1, _card2, _greeting1, _greeting2, message = two_card_forest
        old_revision = f.get_revision(message)
        assert self._controller(f).revise_message(message, "an edited user message") is None
        assert f.get_revision(message) != old_revision
        assert old_revision in f.get_revisions(message), "the old revision is kept"
        assert chatutil.content_to_text(f.get_payload(message)["message"]["content"]) == "an edited user message"

    def test_refused_while_a_turn_is_in_flight(self, two_card_forest):
        # A turn finishes its node by replacing the active revision, which an edit there would then be.
        f, _card1, _card2, _greeting1, _greeting2, message = two_card_forest
        controller = self._controller(f, generating=True)
        maybe_refusal = controller.revise_message(message, "edited mid-turn")
        assert isinstance(maybe_refusal, str) and maybe_refusal
        assert controller.edit_refusal() == maybe_refusal, "a button asking first would be told something else"
        assert len(f.get_revisions(message)) == 1
        assert self._controller(f).edit_refusal() is None  # the control: no turn, no refusal

    def test_emptying_a_message_with_nothing_else_in_it_is_refused(self, two_card_forest):
        f, _card1, _card2, _greeting1, _greeting2, message = two_card_forest
        assert isinstance(self._controller(f).revise_message(message, "  \n"), str)
        assert len(f.get_revisions(message)) == 1
        # The control: with an attachment left, the message is not empty, and the edit goes through.
        payload = f.get_payload(message)
        payload["message"]["content"].append(chatutil.image_content_part("sidecar:a.png"))
        assert self._controller(f).revise_message(message, "") is None
        assert f.get_payload(message)["message"]["content"] == [chatutil.image_content_part("sidecar:a.png")]


class TestBranchingAnEditedMessage:
    """The other route to an edit: a sibling holding the new text, with HEAD moved to it."""

    @staticmethod
    def _controller(forest, head, generating=False):
        """A controller with just what `branch_message` touches."""
        controller = TestRevisingAMessage._controller(forest, generating=generating)
        controller.app_state = {"HEAD": head}
        return controller

    def test_the_edit_becomes_a_sibling_and_the_original_keeps_its_replies(self, two_card_forest, chat_payload):
        f, _card1, _card2, _greeting1, _greeting2, message = two_card_forest
        reply = f.create_node(chat_payload("assistant", "the answer to the original", 5), parent_id=message)
        controller = self._controller(f, reply)
        assert controller.branch_message(message, "a different question") is None
        new_node = controller.app_state["HEAD"]
        assert new_node != reply, "HEAD did not move"
        assert f.get_parent(new_node) == f.get_parent(message), "the edit is not a sibling of the original"
        assert f.get_children(new_node) == [], "the replies came along to the new branch"
        assert chatutil.content_to_text(f.get_payload(new_node)["message"]["content"]) == "a different question"
        assert f.get_children(message) == [reply], "the original lost its replies"
        assert len(f.get_revisions(message)) == 1, "the original was revised rather than branched from"

    def test_refused_as_a_revision_is(self, two_card_forest):
        f, _card1, _card2, _greeting1, _greeting2, message = two_card_forest
        parent = f.get_parent(message)
        n_siblings = len(f.get_children(parent))
        for controller, text in ((self._controller(f, message, generating=True), "edited mid-turn"),
                                 (self._controller(f, message), "  \n")):
            assert isinstance(controller.branch_message(message, text), str)
            assert controller.app_state["HEAD"] == message
        assert len(f.get_children(parent)) == n_siblings
        # The control: neither condition, and the branch goes through.
        assert self._controller(f, message).branch_message(message, "fine") is None
        assert len(f.get_children(parent)) == n_siblings + 1
