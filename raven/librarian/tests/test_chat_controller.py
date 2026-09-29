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

from unpythonic.env import env  # noqa: E402 -- the class; `from unpythonic import env` gets the submodule

from raven.common import bgtask  # noqa: E402
from raven.librarian import chat_controller, chatutil  # noqa: E402


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


class TestTheSpeakerGlyphFollowsTheStoredCharacter:
    """`icon_texture_for`, which both the chat log and the chat graph ask.

    A chat holds turns by whichever characters wrote them, and the character configured *now* is not who
    wrote the older ones. Keyed by role alone — which is how this worked until 0.2.9 — there is exactly
    one slot for the AI's face, so a stored "Juha" message was drawn wearing Aria's.

    Nothing here needs widgets: the textures are opaque handles and the method only chooses between them.
    """

    @staticmethod
    def _controller(monkeypatch, configured="Aria", character_icon=None,
                    configured_user="Juha", user_icon=None):
        """A controller with just the fields the resolver reads.

        `character_icon`, `user_icon`: what `_load_instance_textures` would have set for a character or a
                                       user that ships an icon of its own, as an *instance* attribute
                                       shadowing the class's generic one. `None` leaves the generic
                                       showing, which is the case for somebody without one.
        """
        controller = chat_controller.DPGChatController.__new__(chat_controller.DPGChatController)
        # On the class, because that is where `_load_class_textures` puts them and where the fallback
        # reads them from. `monkeypatch` puts them back, so no other test inherits them.
        for attribute, value in (("icon_ai_texture", "tex_generic_ai"),
                                 ("icon_user_texture", "tex_generic_user")):
            monkeypatch.setattr(chat_controller.DPGChatController, attribute, value, raising=False)
        # Only the two roles that have no speaker; the other two are resolved per persona.
        controller._role_icon_textures = {"system": "tex_system", "tool": "tex_tool"}
        controller.llm_settings = env(personas={"assistant": configured, "user": configured_user,
                                                "system": None, "tool": None})
        if character_icon is not None:
            controller.icon_ai_texture = character_icon
        if user_icon is not None:
            controller.icon_user_texture = user_icon
        return controller

    def test_the_configured_character_wears_its_own_face(self, monkeypatch):
        controller = self._controller(monkeypatch, configured="Aria", character_icon="tex_aria")
        assert controller.icon_texture_for("assistant", "Aria") == "tex_aria"

    def test_another_character_does_not_wear_it(self, monkeypatch):
        """The defect this exists to fix, with its own control beside it.

        The first assertion is the control: a resolver that answered the generic glyph for *everything*
        would satisfy the second one while fixing nothing, and would look exactly like a pass.
        """
        controller = self._controller(monkeypatch, configured="Aria", character_icon="tex_aria")
        assert controller.icon_texture_for("assistant", "Aria") == "tex_aria", \
            "the configured character has no icon of its own here, so borrowing it cannot be detected"
        assert controller.icon_texture_for("assistant", "Juha") == "tex_generic_ai"

    def test_an_assistant_message_with_no_recorded_character_gets_the_generic_glyph(self, monkeypatch):
        # The defensive branch rather than a case anyone meets: every payload gets a persona written with
        # it. Pinned because what it must not do is guess — drawing the configured character's face here
        # would assert something nothing recorded, which is the defect this whole method exists to remove.
        controller = self._controller(monkeypatch, configured="Aria", character_icon="tex_aria")
        assert controller.icon_texture_for("assistant", None) == "tex_generic_ai"

    def test_a_character_without_an_icon_of_its_own_gets_the_generic_one(self, monkeypatch):
        """Which is what happens today for such a character when it is the configured one."""
        controller = self._controller(monkeypatch, configured="Aria", character_icon=None)
        assert controller.icon_texture_for("assistant", "Aria") == "tex_generic_ai"

    def test_the_configured_user_wears_their_own_face(self, monkeypatch):
        """The user side, resolved exactly as the character side is — a conversation has two participants.

        Before 0.2.9 the user always got the generic glyph, there being nowhere to declare another; a
        profile can now carry a `_icon.png` the way a character does.
        """
        controller = self._controller(monkeypatch, configured_user="Juha", user_icon="tex_juha")
        assert controller.icon_texture_for("user", "Juha") == "tex_juha"

    def test_another_user_name_does_not_wear_it(self, monkeypatch):
        # Same reasoning as for a character, and the same control beside it: the configured user's turns
        # are theirs, and a turn stored under another name is somebody else's.
        controller = self._controller(monkeypatch, configured_user="Juha", user_icon="tex_juha")
        assert controller.icon_texture_for("user", "Juha") == "tex_juha", \
            "the configured user has no icon of their own here, so borrowing it cannot be detected"
        assert controller.icon_texture_for("user", "somebody else entirely") == "tex_generic_user"

    def test_a_user_without_a_profile_icon_gets_the_generic_one(self, monkeypatch):
        """Which is what every user got before 0.2.9, and what one without a profile still gets."""
        controller = self._controller(monkeypatch, configured_user="Juha", user_icon=None)
        assert controller.icon_texture_for("user", "Juha") == "tex_generic_user"

    def test_the_roles_with_no_speaker_answer_from_the_role_alone(self, monkeypatch):
        """A system prompt and a tool result are nobody's, so one glyph each is the whole answer.

        Two of the four roles have a speaker and two do not, and this is the half that does not: a persona
        must not change what is drawn for them.
        """
        controller = self._controller(monkeypatch, character_icon="tex_aria", user_icon="tex_juha")
        for role, expected in (("system", "tex_system"), ("tool", "tex_tool")):
            assert controller.icon_texture_for(role, None) == expected
            assert controller.icon_texture_for(role, "somebody else entirely") == expected, \
                f"a persona changed the glyph for role '{role}'"

    def test_an_unknown_role_draws_nothing(self, monkeypatch):
        controller = self._controller(monkeypatch)
        assert controller.icon_texture_for("narrator", None) is None


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
        monkeypatch.setattr(chat_controller.chatutil, "latest_user_message_text", lambda datastore, head: "earlier question")
        controller = self._controller()
        controller.datastore = None
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


class TestSteppingTheSearch:
    """What `step_search` reports back, which is what lets a caller follow a jump that happened.

    The app sends the keyboard to the chat log after one, so that the arrows act on the pane the key just
    took the reader to — and must not when the key did nothing, which is the whole reason this answers at
    all rather than returning `None` as it used to.

    Built with `__new__`, as the datastore-side tests above are: the paths here touch four attributes and
    the view, and `__init__` would build a GUI to reach them.
    """

    @staticmethod
    def _controller(matches, jumped_to=None):
        controller = chat_controller.DPGChatController.__new__(chat_controller.DPGChatController)
        controller._search_jump = None
        controller._search_position_stale = False
        controller.search_matches = matches
        controller.view = types.SimpleNamespace(find_message=lambda node_id: None,
                                                jump_to_node=lambda node_id: jumped_to)
        return controller

    def test_it_says_no_when_there_is_nowhere_to_go(self, monkeypatch):
        controller = self._controller(matches=[])
        monkeypatch.setattr(type(controller), "_find_search_match",
                            lambda self, forward, beyond_a_line: None)
        assert controller.step_search(+1) is False

    def test_it_says_yes_when_it_jumped(self, monkeypatch):
        counts = chat_controller.chatsearch.MatchCounts(content=1, thinking=0)
        controller = self._controller(matches=[("n1", counts)], jumped_to=120)
        monkeypatch.setattr(type(controller), "_find_search_match",
                            lambda self, forward, beyond_a_line: 0)
        assert controller.step_search(+1) is True
        assert controller._search_position_stale, \
            "nothing was recorded, so this fixture did not reach the body it claims to have run"


class TestOpeningAnAwaitedThinkingTrace:
    """A jump that moves HEAD cannot open the trace it lands on, so it asks for the trace and the rebuild obliges.

    The chat graph's commits are where this comes from: a box wearing a count that was a thinking-trace hit
    has no trace of its own to open, and acting on it moves HEAD — at which moment the message the count was
    about does not exist yet, the view rebuilding on another thread. `open_thinking_trace_when_it_matches` names
    the node, and `add_search_matches_for` — which the rebuild already calls once per message — collects.

    Built with `__new__`, as `TestSteppingTheSearch` above is, and for the same reason.
    """

    @staticmethod
    def _payload(reasoning):
        return {"message": {"role": "assistant",
                            "content": [{"type": "text", "text": "It speeds up the reaction using light."}],
                            "reasoning_content": reasoning},
                "general_metadata": {"persona": None}}

    @classmethod
    def _branch(cls, reasoning="Summarize the tool result for the reader."):
        """One reply whose reasoning says something its answer does not, which is the whole case here."""
        forest = chat_controller.chattree.Forest()
        return forest, forest.create_node(payload=cls._payload(reasoning), parent_id=None)

    @staticmethod
    def _controller(forest, search_string, opened):
        controller = chat_controller.DPGChatController.__new__(chat_controller.DPGChatController)
        controller.datastore = forest
        controller.search_query = chat_controller.chatsearch.make_query(search_string, include_thinking=True)
        controller.search_matches = []
        controller._search_jump = None
        controller._search_position_stale = False
        controller.on_search_results_changed = None
        controller._node_awaiting_trace_open = None
        controller.view = types.SimpleNamespace(
            find_message=lambda node_id: types.SimpleNamespace(
                show_thinking_trace=lambda: opened.append(node_id)))
        return controller

    def test_the_awaited_message_opens_its_trace_when_it_arrives(self):
        forest, node_id = self._branch()
        opened = []
        controller = self._controller(forest, "summarize", opened)
        controller.open_thinking_trace_when_it_matches(node_id)
        controller.add_search_matches_for(node_id)
        assert opened == [node_id]

    def test_a_message_that_matched_only_in_its_text_keeps_its_trace_closed(self):
        forest, node_id = self._branch()
        opened = []
        controller = self._controller(forest, "light", opened)
        controller.open_thinking_trace_when_it_matches(node_id)
        controller.add_search_matches_for(node_id)
        assert controller.search_matches, \
            "nothing matched at all, so this fixture cannot tell a trace left closed from a message not found"
        assert opened == []

    def test_a_message_nobody_awaited_keeps_its_trace_closed(self):
        forest, node_id = self._branch()
        opened = []
        controller = self._controller(forest, "summarize", opened)
        controller.add_search_matches_for(node_id)  # no request made
        assert controller.search_matches[0][1].thinking, \
            "the trace did not match, so this fixture cannot tell a request being honoured from one never made"
        assert opened == []

    def test_the_request_is_spent_on_the_message_it_named(self):
        forest, node_id = self._branch()
        opened = []
        controller = self._controller(forest, "summarize", opened)
        controller.open_thinking_trace_when_it_matches(node_id)
        controller.add_search_matches_for(node_id)
        controller.add_search_matches_for(node_id)  # a later rebuild walking past the same message
        assert opened == [node_id]

    def test_clearing_the_search_before_the_message_arrives_still_ends_the_wait(self):
        forest, node_id = self._branch()
        opened = []
        controller = self._controller(forest, "summarize", opened)
        controller.open_thinking_trace_when_it_matches(node_id)
        controller.search_query = None  # the reader cleared it in the frames since the jump
        controller.add_search_matches_for(node_id)
        assert opened == []
        assert controller._node_awaiting_trace_open is None, \
            "the request outlived the message it named, and would be spent on some later rebuild"

    def test_a_reply_still_writing_its_trace_keeps_the_jump_waiting(self):
        forest, node_id = self._branch(reasoning="Let me think about this for a moment.")
        opened = []
        controller = self._controller(forest, "summarize", opened)
        controller.open_thinking_trace_when_it_matches(node_id)
        controller.recheck_awaited_thinking_trace(node_id)
        assert opened == []
        assert controller._node_awaiting_trace_open == node_id, \
            "the wait ended on a trace that had not yet written the words it was waiting for"

    def test_the_trace_opens_as_soon_as_the_words_arrive(self):
        forest, node_id = self._branch(reasoning="Let me think about this for a moment.")
        opened = []
        controller = self._controller(forest, "summarize", opened)
        controller.open_thinking_trace_when_it_matches(node_id)
        controller.recheck_awaited_thinking_trace(node_id)
        assert opened == [], \
            "the trace matched before the words arrived, so this fixture cannot show them arriving"
        forest.add_revision(node_id, self._payload("Summarize the tool result for the reader."))
        controller.recheck_awaited_thinking_trace(node_id)
        assert opened == [node_id]
        assert controller._node_awaiting_trace_open is None, \
            "the trace opened but the request was not spent, so a later message could collect it too"
