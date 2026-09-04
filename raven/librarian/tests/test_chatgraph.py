"""Unit tests for raven.librarian.chatgraph — the chat forest as a renderable graph.

What is worth pinning here is *geometry and selection*, because nothing else can check them. A rendered
widget cannot be asked whether two boxes overlap or whether the spine came out straight, and the eye that
could is not present in CI. So these tests assert positions.

The module needs no DearPyGui: the xdot package imports its widget lazily, and everything used here --
`Graph`, `Node`, the shapes -- is plain data.
"""

import math

import pytest

from raven.common.gui.xdotwidget import graph as xdotgraph

from raven.librarian import chatgraph
from raven.librarian import chatutil
from raven.librarian.chattree import Forest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def payload(role: str, text: str, tool_calls=None) -> dict:
    """A chat node payload of the shape `chatutil` writes, with only the fields this module reads.

    The timestamp is real data rather than filler: `chatutil.descend_to_latest` orders siblings by it, so
    a fixture without one cannot say which branch a focus continues into. It counts up per call, in
    creation order, which is what wall-clock time does to nodes written one after another.
    """
    global _payload_serial
    _payload_serial += 1
    message = {"role": role, "content": [{"type": "text", "text": text}]}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"message": message,
            "general_metadata": {"persona": None, "timestamp": _payload_serial}}


_payload_serial = 0


def chain(forest: Forest, length: int, parent_id=None) -> list:
    """Create a linear run of `length` alternating user/assistant nodes. Returns their IDs, root first."""
    ids = []
    for k in range(length):
        role = "user" if k % 2 == 0 else "assistant"
        parent_id = forest.create_node(payload(role, f"message {k}"), parent_id=parent_id)
        ids.append(parent_id)
    return ids


@pytest.fixture
def conversation():
    """A system prompt, a greeting, and one exchange under it.

        system -> greeting -> user -> assistant
    """
    forest = Forest()
    system = forest.create_node(payload("system", "you are a helpful assistant"), parent_id=None)
    greeting = forest.create_node(payload("assistant", "hello!"), parent_id=system)
    user = forest.create_node(payload("user", "what is a multiverse"), parent_id=greeting)
    reply = forest.create_node(payload("assistant", "many worlds, briefly"), parent_id=user)
    return forest, system, greeting, user, reply


def boxes_of(chat_graph: chatgraph.ChatGraph) -> dict:
    """Return graph node name -> its bounding box, for the geometry assertions."""
    return {node.internal_name: node.get_bounding_box() for node in chat_graph.graph.nodes}


def overlapping_pairs(chat_graph: chatgraph.ChatGraph) -> list:
    """Return every pair of graph nodes whose boxes overlap by more than a rounding error."""
    epsilon = 1e-6
    named = list(boxes_of(chat_graph).items())
    found = []
    for i, (name_a, a) in enumerate(named):
        for name_b, b in named[i + 1:]:
            if (a[0] < b[2] - epsilon and b[0] < a[2] - epsilon
                    and a[1] < b[3] - epsilon and b[1] < a[3] - epsilon):
                found.append((name_a, name_b))
    return found


def refs_of_type(chat_graph: chatgraph.ChatGraph, ref_type) -> list:
    """Return every ref of the given kind, in no particular order."""
    return [ref for ref in chat_graph.refs.values() if isinstance(ref, ref_type)]


def only_ref_of_type(chat_graph: chatgraph.ChatGraph, ref_type):
    """Return the one ref of the given kind, failing if there is not exactly one.

    Fails rather than returning the first, because a fixture that grew a second gap of the same kind would
    otherwise go on passing while testing whichever one the dictionary happened to yield first.
    """
    found = refs_of_type(chat_graph, ref_type)
    assert len(found) == 1, f"expected one {ref_type.__name__}, got {len(found)}"
    return found[0]


def only_depth_gap(chat_graph: chatgraph.ChatGraph):
    """Return the graph node of the one depth gap, failing if there is not exactly one.

    Looked up by ref rather than by name: a depth gap is named for the first node it hides, so that the
    name survives the window moving, and a test that spelled the name out would be asserting the naming
    scheme rather than the layout.
    """
    gaps = refs_of_type(chat_graph, chatgraph.DepthGapRef)
    assert len(gaps) == 1, f"expected one depth gap, got {len(gaps)}"
    return chat_graph.graph.get_node_by_name(gaps[0].name)


def texts_on(chat_graph: chatgraph.ChatGraph, node_name: str) -> list:
    """Return every string drawn on one graph node, its pill labels included.

    Gap boxes are why this reads the drawing rather than a ref: a `ChatNodeRef` carries its `pills`, and
    the three gap refs have no such field -- what a gap says about HEAD exists only on screen.
    """
    node = chat_graph.graph.get_node_by_name(node_name)
    return [shape.t for shape in node.shapes if isinstance(shape, xdotgraph.TextShape)]


def _width_of(shape) -> float:
    """Return the width of a shape's bounding box."""
    box = shape.get_bounding_box()
    return box[2] - box[0]


# ---------------------------------------------------------------------------
# The spine
# ---------------------------------------------------------------------------

class TestSpine:
    def test_a_chain_becomes_one_box_per_message(self, conversation):
        forest, system, greeting, user, reply = conversation
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        for node_id in (system, greeting, user, reply):
            assert node_id in built.refs

    def test_the_spine_is_a_straight_vertical_line(self):
        # The property `TODO.md` asks for by name: a stable picture. Aligning every row on the node the
        # branch passes through is what keeps a new sibling from moving anything but its own row.
        #
        # The fixture needs rows of *differing* width. In a plain chain every row holds one box and starts
        # at the same offset, so the spine comes out straight whether the rows are aligned on it or merely
        # all left-aligned by accident -- and the test then passes against a layout with no alignment step
        # at all.
        forest = Forest()
        system = forest.create_node(payload("system", "you are helpful"), parent_id=None)
        greeting = forest.create_node(payload("assistant", "hello!"), parent_id=system)
        chats = [forest.create_node(payload("user", f"chat {k}"), parent_id=greeting) for k in range(12)]
        head = chats[6]
        for k in range(3):
            forest.create_node(payload("assistant", f"reroll {k}"), parent_id=head)

        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=head))

        widths_per_row = {}
        for node in built.graph.nodes:
            widths_per_row[node.y] = widths_per_row.get(node.y, 0) + 1
        assert len(set(widths_per_row.values())) > 1, \
            "every row here holds the same number of boxes, so this fixture cannot detect a missing alignment"

        xs = {built.graph.get_node_by_name(node_id).x for node_id in (system, greeting, head)}
        assert len(xs) == 1, f"the branch to HEAD is not on one vertical line: {sorted(xs)}"

    def test_depth_increases_downward(self, conversation):
        forest, system, greeting, user, reply = conversation
        ys = [built_y for built_y in
              (chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
               .graph.get_node_by_name(node_id).y
               for node_id in (system, greeting, user, reply))]
        assert ys == sorted(ys) and len(set(ys)) == len(ys)

    def test_a_graph_node_is_named_by_its_chat_node_id(self, conversation):
        # Not cosmetic: it is what lets the panel call `XDotWidget.pan_to_node(head_node_id)` without
        # keeping a translation table of its own.
        forest, system, greeting, user, reply = conversation
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        assert built.graph.get_node_by_name(reply) is not None

    def test_every_graph_node_has_a_ref(self, conversation):
        # A click arrives as a name and nothing else, so a name without a ref is a click that cannot be
        # acted on.
        forest, system, greeting, user, reply = conversation
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        assert {node.internal_name for node in built.graph.nodes} == set(built.refs)

    def test_head_children_are_shown(self, conversation):
        forest, system, greeting, user, reply = conversation
        follow_up = forest.create_node(payload("user", "go on"), parent_id=reply)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        assert follow_up in built.refs, "a branch that continues below HEAD must not look like it ends there"

    def test_building_does_not_change_the_forest(self, conversation):
        forest, system, greeting, user, reply = conversation
        before = forest.generation
        chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        assert forest.generation == before


# ---------------------------------------------------------------------------
# Previewing another branch
# ---------------------------------------------------------------------------

class TestPreview:
    """Browsing the multiverse must change nothing; only a deliberate second act does.

    So the picture is drawn around a previewed node while HEAD stays put, and the two questions -- what is
    on screen, and where you actually are -- are answered separately.
    """

    def _fork(self):
        """A shared prefix, then two ways on.

            system -> user -> taken -> taken_tip
                           -> not_taken -> not_taken_tip
        """
        forest = Forest()
        system = forest.create_node(payload("system", "you are helpful"), parent_id=None)
        user = forest.create_node(payload("user", "which way"), parent_id=system)
        taken = forest.create_node(payload("assistant", "this way"), parent_id=user)
        taken_tip = forest.create_node(payload("user", "onwards"), parent_id=taken)
        not_taken = forest.create_node(payload("assistant", "or that way"), parent_id=user)
        not_taken_tip = forest.create_node(payload("user", "elsewhere"), parent_id=not_taken)
        return forest, system, user, taken, taken_tip, not_taken, not_taken_tip

    def test_the_previewed_branch_is_the_one_drawn(self):
        forest, system, user, taken, taken_tip, not_taken, not_taken_tip = self._fork()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=taken_tip,
                                                            focus_node_id=not_taken_tip))
        assert not_taken_tip in built.refs, "the previewed branch was not drawn"
        assert built.spine == tuple(forest.linearize_up(not_taken_tip))

    def test_the_colour_still_says_where_head_is(self):
        # The half that makes a preview readable: the shared prefix is coloured and the divergence is not,
        # so the picture shows where you would be going against where you are.
        forest, system, user, taken, taken_tip, not_taken, not_taken_tip = self._fork()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=taken_tip,
                                                            focus_node_id=not_taken_tip))
        assert built.refs[system].on_current_branch and built.refs[user].on_current_branch
        assert not built.refs[not_taken].on_current_branch
        assert not built.refs[not_taken_tip].on_current_branch

    def test_previewing_leaves_the_head_pill_where_head_is(self):
        forest, system, user, taken, taken_tip, not_taken, not_taken_tip = self._fork()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=taken_tip,
                                                            focus_node_id=not_taken_tip))
        assert "HEAD" not in built.refs[not_taken_tip].pills, \
            "the previewed node is wearing the pointer it has not been given"
        assert taken in built.refs, "the fixture cannot say where HEAD's pill went if that row is off-screen"
        assert built.refs[taken].pills == ()

    def test_no_focus_means_the_picture_is_heads(self):
        # The control: without it, a build that ignored `focus_node_id` entirely would satisfy nothing
        # above, but a build that ignored `head_node_id` would satisfy all of it.
        forest, system, user, taken, taken_tip, not_taken, not_taken_tip = self._fork()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=taken_tip))
        assert built.spine == tuple(forest.linearize_up(taken_tip))
        assert built.refs[taken_tip].pills == ("HEAD",)
        assert built.refs[taken].on_current_branch

    def test_a_deleted_head_does_not_take_the_picture_with_it(self):
        # HEAD can vanish under a running view -- a cleanup, a deleted subtree. The colour is then unknown,
        # which is worth strictly less than the picture.
        forest, system, user, taken, taken_tip, not_taken, not_taken_tip = self._fork()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id="no-such-node",
                                                            focus_node_id=not_taken_tip))
        assert not_taken_tip in built.refs
        assert not any(ref.on_current_branch for ref in built.refs.values()
                       if isinstance(ref, chatgraph.ChatNodeRef))


# ---------------------------------------------------------------------------
# Who said it
# ---------------------------------------------------------------------------

class TestAMessageWithNoText:
    """Three ways a message ends up with no prose, and they are three different things.

    Drawn as one `[empty]` box they read as a tree full of replies that never happened — and the commonest
    of them, a turn that asked for a tool, is the ordinary shape of an agent doing its job. Counted over a
    real datastore on 2026-09-03: of 217 assistant messages, 45 carried no text, and 33 of those were tool
    calls, 7 were an interrupted thinking model, 5 were genuinely nothing.
    """

    def _texts(self, built, node_id):
        node = built.graph.get_node_by_name(node_id)
        return [s.t for s in node.shapes if isinstance(s, xdotgraph.TextShape)]

    def _forest(self, message_extras):
        forest = Forest()
        system = forest.create_node(payload("system", "the card"), parent_id=None)
        asked = forest.create_node(payload("user", "go on then"), parent_id=system)
        silent = forest.create_node(payload("assistant", ""), parent_id=asked)
        forest.get_payload(silent)["message"].update(message_extras)
        return forest, silent

    def _round(self, forest, parent, calls):
        """Build one agent round: the assistant message that asked, then a `tool` node per call.

        The loop chains the results rather than fanning them, so three calls are three nodes deep. Built
        by hand because no model we have will emit several calls in one message, and the shape is legal
        OpenAI — so the only way this case gets exercised is if the fixture makes it.
        """
        asking = forest.create_node(payload("assistant", "", tool_calls=calls), parent_id=parent)
        node = asking
        for call in calls:
            node = forest.create_node(payload("tool", f"result of {call['function']['name']}"),
                                      parent_id=node)
        return asking, node

    def test_the_speaker_line_says_it_is_a_tool_call(self):
        # Without it the box reads as the character saying "calculate(...)" to the user, which is not what
        # happened. (Juha, 2026-09-03, from the live picture.)
        calls = [{"id": "c1", "function": {"name": "calculate", "arguments": '{"expression": "sqrt(10)"}'}}]
        forest, silent = self._forest({"tool_calls": calls})
        forest.get_payload(silent)["general_metadata"]["persona"] = "Aria"
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=silent))
        assert "Aria [tool call]" in self._texts(built, silent)

    def test_an_ordinary_reply_says_only_who_spoke(self):
        # The control: tagging every box would satisfy the test above and be wrong on all the others.
        forest = Forest()
        system = forest.create_node(payload("system", "the card"), parent_id=None)
        spoke = forest.create_node(payload("assistant", "hello!"), parent_id=system)
        forest.get_payload(spoke)["general_metadata"]["persona"] = "Aria"
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=spoke))
        texts = self._texts(built, spoke)
        assert "Aria" in texts
        assert not any("tool call" in text for text in texts)

    def test_a_tool_call_says_what_it_called(self):
        calls = [{"id": "c1", "function": {"name": "websearch",
                                           "arguments": '{"query": "cosmology news 2026"}'}}]
        forest, silent = self._forest({"tool_calls": calls})
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=silent))
        drawn = " ".join(self._texts(built, silent))
        assert "websearch" in drawn
        assert "[empty]" not in drawn

    def test_it_says_it_the_way_the_chat_log_does(self):
        # One spelling for one call. Two would read as two different calls to a reader moving between the
        # views, which is exactly what the graph is for.
        calls = [{"id": "c1", "function": {"name": "websearch",
                                           "arguments": '{"query": "cosmology news 2026"}'}}]
        forest, silent = self._forest({"tool_calls": calls})
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=silent))
        expected = chatutil.format_tool_call("websearch", '{"query": "cosmology news 2026"}')
        drawn = " ".join(self._texts(built, silent))
        # Wrapped and possibly truncated into the box, so the head of it is what can be asserted.
        assert drawn.startswith(expected[:20]) or expected[:20] in drawn

    def test_several_calls_are_counted(self):
        calls = [{"id": f"c{k}", "function": {"name": f"tool{k}", "arguments": "{}"}} for k in range(3)]
        forest, silent = self._forest({"tool_calls": calls})
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=silent))
        assert "3 tool calls" in self._texts(built, silent)

    def test_one_call_is_not_counted(self):
        # The control: a count on every tool-calling box would satisfy the test above and spend a line
        # saying "1 tool calls" on the overwhelmingly common case.
        calls = [{"id": "c1", "function": {"name": "websearch", "arguments": "{}"}}]
        forest, silent = self._forest({"tool_calls": calls})
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=silent))
        assert not any("tool calls" in text for text in self._texts(built, silent))

    def test_a_counted_box_keeps_its_text_inside_it(self):
        # The count costs a label line rather than being added below them: a message box carries a speaker
        # line that the gap boxes `sub_label` was written for do not, and the full-height version puts the
        # last baseline two units inside an 84-unit box -- inside by the baseline, outside by the
        # descenders.
        calls = [{"id": f"c{k}", "function": {"name": f"tool{k}", "arguments": "{}"}} for k in range(3)]
        forest, silent = self._forest({"tool_calls": calls})
        config = chatgraph.LayoutConfig()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=silent), config)
        node = built.graph.get_node_by_name(silent)
        lowest = max(s.y for s in node.shapes if isinstance(s, xdotgraph.TextShape))
        assert lowest <= node.y2 - config.role_font_size / 2, \
            "the last line's descenders fall outside the box"

    def test_an_interrupted_thinking_model_says_so(self):
        forest, silent = self._forest({"reasoning_content": "let me work through this..."})
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=silent))
        drawn = self._texts(built, silent)
        assert "[thinking only]" in drawn
        assert "[empty]" not in drawn

    def test_the_answer_after_a_tool_round_is_still_attached(self):
        # The row after a folded round hangs, in the datastore, from a `tool` node the picture does not
        # draw — so the edge had nowhere to start and the whole row lost every edge it had, siblings
        # included. The branch then appeared to stop at the tool call with its answer floating unattached
        # below it. (Juha, 2026-09-03, from the live picture: "the graph breaks there on both sides".)
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        asked = forest.create_node(payload("user", "what is sqrt(10)?"), parent_id=root)
        calls = [{"id": f"c{k}", "function": {"name": f"tool{k}", "arguments": "{}"}} for k in range(3)]
        asking, last_result = self._round(forest, asked, calls)
        answer = forest.create_node(payload("assistant", "about 3.1623"), parent_id=last_result)

        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=answer))
        assert last_result not in built.refs, "the round is drawn expanded, so nothing was folded"
        gap = only_ref_of_type(built, chatgraph.ToolRoundGapRef)

        attached = {(edge.src.internal_name, edge.dst.internal_name) for edge in built.graph.edges}
        assert (asking, gap.name) in attached and (gap.name, answer) in attached, \
            "the answer hangs off nothing; the branch is drawn in two disconnected pieces"
        assert (asking, answer) not in attached, \
            "the call is joined straight to its answer, which says no node sits between them"

    def test_a_wide_round_keeps_its_siblings_attached_too(self):
        # The half that made this visible on screen: it is not one edge that goes missing but the row's,
        # so a sibling of the answer is orphaned as well. Needs a fixture with siblings on that row, which
        # the single-branch case above does not have.
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        asked = forest.create_node(payload("user", "what is sqrt(10)?"), parent_id=root)
        calls = [{"id": f"c{k}", "function": {"name": f"tool{k}", "arguments": "{}"}} for k in range(3)]
        asking, last_result = self._round(forest, asked, calls)
        answers = [forest.create_node(payload("assistant", f"answer {k}"), parent_id=last_result)
                   for k in range(3)]

        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=answers[1]))
        gap = only_ref_of_type(built, chatgraph.ToolRoundGapRef)
        attached = {(edge.src.internal_name, edge.dst.internal_name) for edge in built.graph.edges}
        drawn_answers = [node_id for node_id in answers if node_id in built.refs]
        assert len(drawn_answers) == 3, "the fixture did not draw the siblings, so it checks one edge"
        for node_id in drawn_answers:
            assert (gap.name, node_id) in attached, f"sibling {node_id} was left unattached"

    def test_a_message_with_nothing_at_all_still_says_empty(self):
        # The one case `[empty]` is the honest answer for, and the control for the two above: a box that
        # never said it would be indistinguishable from a tool call that went unlabelled.
        forest, silent = self._forest({})
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=silent))
        assert "[empty]" in self._texts(built, silent)


class TestSpeaker:
    """Role was the one thing a node carried no channel for: colour is branch membership, and the label is
    the message. The plan was glyphs, which wait on an `ImageShape` the widget does not have; text does not.
    """

    def _texts(self, built, node_id):
        node = built.graph.get_node_by_name(node_id)
        return [s.t for s in node.shapes if isinstance(s, xdotgraph.TextShape)]

    def test_a_node_says_who_spoke(self, conversation):
        forest, system, greeting, user, reply = conversation
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        assert "SYSTEM" in self._texts(built, system)
        assert "USER" in self._texts(built, user)
        assert "AI" in self._texts(built, reply)

    def test_a_persona_is_preferred_to_the_role(self):
        # The chat log shows the character's name, so the graph shows the same one; "AI" is the fallback
        # for a message stored without a persona, not the normal case.
        forest = Forest()
        system = forest.create_node(payload("system", "you are helpful"), parent_id=None)
        spoken = forest.create_node(payload("assistant", "hello!"), parent_id=system)
        forest.get_payload(spoken)["general_metadata"]["persona"] = "Aria"

        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=spoken))
        texts = self._texts(built, spoken)
        assert "Aria" in texts
        assert "AI" not in texts, "the persona is there but the role caption is too, so both are drawn"

    def test_a_gap_says_nobody(self):
        # The control for the above: a speaker line on every box would satisfy them and be wrong here --
        # a gap is not a message and has no speaker.
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        greeting = forest.create_node(payload("assistant", "hello!"), parent_id=root)
        chats = [forest.create_node(payload("user", f"chat {k}"), parent_id=greeting) for k in range(30)]
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=chats[0]))

        gaps = refs_of_type(built, chatgraph.SiblingGapRef)
        assert gaps, "no gap in this fixture, so nothing is being checked"
        for gap in gaps:
            texts = self._texts(built, gap.name)
            assert len(texts) == 1, f"a gap box drew {len(texts)} texts; it should draw only its count"

    def _reply(self, model):
        """A drawn AI reply with a persona, and `model` as its recorded identity."""
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        reply = forest.create_node(payload("assistant", "hello there"), parent_id=root)
        forest.get_payload(reply)["general_metadata"]["persona"] = "Aria"
        if model is not None:
            forest.get_payload(reply)["generation_metadata"] = {"model": model}
        return forest, reply

    def test_a_reply_says_which_model_wrote_it(self):
        # The short name, not the whole recorded identity: the chat log's metadata line has room for
        # "qwen3.5-4b, Q4_K_XL, 128 Ki context" and a box does not.
        forest, reply = self._reply("qwen3.5-4b, Q4_K_XL, 128 Ki context")
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        assert "Aria [qwen3.5-4b]" in self._texts(built, reply)

    def test_a_reply_with_no_recorded_model_says_only_the_speaker(self):
        # The control. Every reply reaches this branch, and one written before the field existed — or
        # interrupted before there was anything to record — must not gain an empty bracket.
        forest, reply = self._reply(None)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        texts = self._texts(built, reply)
        assert "Aria" in texts
        assert not any(text.startswith("Aria [") for text in texts), \
            "something was bracketed onto the speaker with no model to put there"

    def test_a_model_too_long_for_the_box_is_cut_rather_than_drawn_past_the_edge(self):
        # Nothing downstream would cut it: the renderer draws a `TextShape` whole, using its width only to
        # offset justification, and its compaction callback fires only once the text is too small to read.
        # So an over-long speaker line spills out of the box and across whatever sits beside it.
        config = chatgraph.LayoutConfig()
        forest, reply = self._reply("a-very-long-model-identifier-that-goes-on-and-on-forever")
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply), config)
        speaker = next(text for text in self._texts(built, reply) if text.startswith("Aria"))
        assert len(speaker) <= config._get_effective_speaker_chars()
        assert speaker.endswith("…]"), "the line was cut without saying that it was"

    def test_a_speaker_with_no_room_left_keeps_its_name_and_drops_the_model(self):
        # Which of the two gives, when they cannot both fit. A box whose author's name was eaten to make
        # room for a model number has lost the more important of the two.
        #
        # The width is chosen against the threshold rather than guessed at: a box narrow enough to cut the
        # model to fewer than `_MIN_BRACKETED_CHARS` is where the bracket is abandoned, and a wider one
        # would merely truncate — which is the *other* behaviour, and would pass this assertion trivially.
        narrow = chatgraph.LayoutConfig(node_w=90.0)
        assert narrow._get_effective_speaker_chars() - len("Aria") - len(" []") < chatgraph._MIN_BRACKETED_CHARS, \
            "this box is wide enough to truncate into, so the drop is not what is being tested"
        forest, reply = self._reply("qwen3.5-4b, Q4_K_XL, 128 Ki context")
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply), narrow)
        assert "Aria" in self._texts(built, reply), "the speaker's own name did not survive"

    def test_a_calling_turn_leaves_room_for_its_own_marker(self):
        # The bracket that slipped: a turn that asked for tools ends its speaker line with `[tool call]`,
        # and the model name was fitted against the whole line before that was appended. 52 characters
        # against a budget of 40, drawn straight out of the box. (Spotted in a screenshot of the live
        # app, where the model happened to be short enough to fit anyway.)
        config = chatgraph.LayoutConfig()
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        calls = [{"id": "c1", "function": {"name": "calculate", "arguments": "{}"}}]
        asking = forest.create_node(payload("assistant", "", tool_calls=calls), parent_id=root)
        forest.get_payload(asking)["general_metadata"]["persona"] = "Aria"
        forest.get_payload(asking)["generation_metadata"] = {
            "model": "unsloth/Qwen3-30B-A3B-Instruct-2507-abliterated-v2, Q4_K_M"}

        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=asking))
        speaker = next(text for text in self._texts(built, asking) if text.startswith("Aria"))
        assert speaker.endswith("[tool call]"), "the fixture drew no marker, so it reserves nothing"
        assert len(speaker) <= config._get_effective_speaker_chars(), \
            f"the marker was appended past the end of the line: {speaker!r}"

    def test_a_tool_result_with_a_long_name_is_cut_the_same_way(self):
        # The tool branch shares the fitter, so it inherits the cut. Asserted because it did *not* before:
        # the tool name was concatenated straight on, and a long one would have overflowed unremarked.
        config = chatgraph.LayoutConfig()
        forest, result, reply = self._tool_result("a_tool_with_an_unreasonably_long_function_name_indeed")
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        speaker = next(text for text in self._texts(built, result) if text.startswith("TOOL"))
        assert len(speaker) <= config._get_effective_speaker_chars()

    def _tool_result(self, function_name):
        """A drawn tool result, named or not. Returns `(forest, the result node, HEAD)`.

        One call, so the round is below the folding threshold and its result is drawn — a folded one has
        no box and therefore no speaker line to check.
        """
        forest = Forest()
        system = forest.create_node(payload("system", "the card"), parent_id=None)
        asked = forest.create_node(payload("user", "what is the time"), parent_id=system)
        calls = [{"id": "c1", "function": {"name": "get_current_time", "arguments": "{}"}}]
        asking = forest.create_node(payload("assistant", "let me look", tool_calls=calls), parent_id=asked)
        result = forest.create_node(payload("tool", "12:00"), parent_id=asking)
        if function_name is not None:
            forest.get_payload(result)["generation_metadata"] = {"function_name": function_name}
        reply = forest.create_node(payload("assistant", "it is noon"), parent_id=result)
        return forest, result, reply

    def test_a_tool_result_says_which_tool_answered(self):
        # The cogs say one ran; the name says which. A turn that called three tools is otherwise three
        # boxes captioned identically, and telling them apart is the whole reason to draw them.
        forest, result, reply = self._tool_result("websearch")
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        assert result in built.refs, "the round folded, so there is no result box to caption"
        assert "TOOL [websearch]" in self._texts(built, result)

    def test_a_tool_result_with_no_recorded_tool_says_only_TOOL(self):
        # The control, and a real case rather than a hypothetical: a call that failed before it had a
        # function to name records none, and so does anything written before the field existed. A caption
        # reading "TOOL [None]" would be worse than the bare one.
        forest, result, reply = self._tool_result(None)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        texts = self._texts(built, result)
        assert "TOOL" in texts
        assert not any(text.startswith("TOOL [") for text in texts), \
            "something was bracketed onto the caption with no tool name to put there"

    def test_the_label_cut_follows_the_node_width_and_the_font(self):
        # Three numbers that have to agree. Derived rather than written down, so that changing any one of
        # them cannot leave a label that overflows its box or stops short of it.
        narrow = chatgraph.LayoutConfig(node_w=120.0)
        wide = chatgraph.LayoutConfig(node_w=600.0)
        assert wide._get_effective_label_chars() > narrow._get_effective_label_chars()

        big_font = chatgraph.LayoutConfig(font_size=40.0)
        assert big_font._get_effective_label_chars() < chatgraph.LayoutConfig()._get_effective_label_chars()

        assert chatgraph.LayoutConfig(label_chars=7)._get_effective_label_chars() == 7, \
            "an explicit setting must still win"


# ---------------------------------------------------------------------------
# Which box is which
# ---------------------------------------------------------------------------

class TestEmphasis:
    """Three states a box can be in, and none of them may be mistaken for another.

    The hover highlight is the widget's and belongs to the pointer. HEAD is where the reader *is*. A
    preview is where a second click *would* take them. Drawing any two of these the same way is what makes
    a lit box read as "this is the current one" when it is nothing of the sort.
    """

    def _outline_widths(self, built, node_id):
        node = built.graph.get_node_by_name(node_id)
        return [s.pen.linewidth for s in node.shapes
                if isinstance(s, xdotgraph.PolygonShape) and not s.filled]

    def _rings(self, built, node_id):
        node = built.graph.get_node_by_name(node_id)
        return [s for s in node.shapes
                if isinstance(s, xdotgraph.PolygonShape) and s.pen.color == chatgraph.PREVIEW_COLOR]

    def test_head_is_drawn_heavier_than_the_rest(self, conversation):
        forest, system, greeting, user, reply = conversation
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        config = chatgraph.LayoutConfig()
        assert config.head_line_width > config.line_width, \
            "the two weights are equal, so this fixture cannot tell them apart"
        assert max(self._outline_widths(built, reply)) == pytest.approx(config.head_line_width)
        assert max(self._outline_widths(built, user)) == pytest.approx(config.line_width)

    def test_a_previewed_box_gets_a_ring_of_its_own(self, conversation):
        forest, system, greeting, user, reply = conversation
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply,
                                                            cursor_name=user))
        assert len(self._rings(built, user)) == 1
        assert self._rings(built, reply) == [], "the ring is on every box, so it marks nothing"

    def test_the_ring_sits_outside_the_box_and_is_dotted(self, conversation):
        # Outside, because the box's own outline is already saying something -- solid or dashed, heavy for
        # HEAD -- and a selection has to be legible over every combination of those. Dotted, because the
        # selection is tentative until a second click.
        forest, system, greeting, user, reply = conversation
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply,
                                                            cursor_name=user))
        node = built.graph.get_node_by_name(user)
        ring = self._rings(built, user)[0]
        assert ring.pen.dash, "the ring is solid, so nothing says the selection is provisional"
        assert ring.get_bounding_box()[0] < node.get_bounding_box()[0]

    def test_a_dash_mark_is_longer_than_the_stroke_is_thick(self):
        # Otherwise each mark is wider than it is long, and the rounding of its endpoints -- plus the join
        # where one spans two segments of a rounded corner -- changes its apparent *weight* rather than its
        # length. A dashed line of visibly uneven thickness is the result, and it looks like a rendering
        # fault rather than a style.
        config = chatgraph.LayoutConfig()
        assert chatgraph._PREVIEW_DOTS[0] >= 2.0 * config.preview_line_width
        assert chatgraph._GAP_DASH[0] >= 2.0 * config.line_width

    def test_head_can_also_be_the_previewed_box(self, conversation):
        # The two marks are independent, and a reader who clicks the box they are already on should get
        # both rather than either winning.
        forest, system, greeting, user, reply = conversation
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply, cursor_name=reply))
        config = chatgraph.LayoutConfig()
        assert len(self._rings(built, reply)) == 1
        assert max(self._outline_widths(built, reply)) == pytest.approx(config.head_line_width)


# ---------------------------------------------------------------------------
# Pointer pills
# ---------------------------------------------------------------------------

class TestPills:
    def test_the_three_pointers_land_on_their_nodes(self, conversation):
        forest, system, greeting, user, reply = conversation
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply,
                                                            new_chat_node_id=greeting))
        assert built.refs[system].pills == ("SYS",)
        assert built.refs[greeting].pills == ("NEW",)
        assert built.refs[reply].pills == ("HEAD",)

    def test_a_pill_outline_has_no_repeated_vertex(self):
        """A repeated vertex is a zero-length segment, which a stroked polyline renders as a spur.

        Pills are where this bites: their radius clamps to half their height, so the box is a stadium, the
        two arcs on each side share a centre, and each arc ends exactly where the next begins. Visible as a
        horizontal flick at the pill's left and right extremes -- which is where those joins are.
        """
        forest = Forest()
        system = forest.create_node(payload("system", "you are helpful"), parent_id=None)
        user = forest.create_node(payload("user", "hello"), parent_id=system)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=user))

        config = chatgraph.LayoutConfig()
        stadium = chatgraph._rounded_rect_points(0.0, 0.0, 40.0, config.pill_h, 0.5 * config.pill_h)
        assert len(stadium) < 4 * (chatgraph._ROUNDED_CORNER_SEGMENTS + 1), \
            ("this box is not degenerate enough to have coincident arc centres, so it cannot show the "
             "duplicate-vertex problem at all")
        # `range(len(...))` rather than `range(1, ...)`, so that k=0 compares the first vertex with the
        # last. That wrap-around *is* the closing segment, and checking only interior joins is how the
        # first version of this test agreed with a spur that was still there: the two arcs meeting at the
        # seam are the ones the walk starts and ends on, which for a stadium is its leftmost point.
        repeats = [k for k in range(len(stadium))
                   if math.dist(stadium[k], stadium[k - 1]) <= chatgraph._COINCIDENT_POINT_TOLERANCE]
        assert repeats == [], f"repeated vertices at {repeats}"

        # And the same for what actually gets drawn, since the pill above is only the helper's output.
        for shape in built.graph.get_node_by_name(system).shapes:
            if isinstance(shape, xdotgraph.PolygonShape):
                doubled = [k for k in range(len(shape.points))
                           if math.dist(shape.points[k],
                                        shape.points[k - 1]) <= chatgraph._COINCIDENT_POINT_TOLERANCE]
                assert doubled == [], f"repeated vertices at {doubled} in a drawn outline"

    def test_a_pill_label_is_measured_rather_than_given_the_box_width(self):
        """The renderer centres text by starting it at `centre - w/2` and drawing left-aligned.

        So `w` has to be what the text measures. Passing the box width instead begins a short label half
        the difference too far left, which is what put "SYS" inside its own rounded cap.
        """
        # HEAD is put below the root so the root wears exactly one pill; with two, "the pill box" below is
        # ambiguous and the test would be asserting about whichever came first.
        forest = Forest()
        system = forest.create_node(payload("system", "you are helpful"), parent_id=None)
        user = forest.create_node(payload("user", "hello"), parent_id=system)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=user))
        assert built.refs[system].pills == ("SYS",)

        node = built.graph.get_node_by_name(system)
        pill_texts = [s for s in node.shapes
                      if isinstance(s, xdotgraph.TextShape) and s.t == "SYS"]
        assert len(pill_texts) == 1
        pill_boxes = [s for s in node.shapes
                      if isinstance(s, xdotgraph.PolygonShape) and _width_of(s) < chatgraph.LayoutConfig().node_w]
        assert len(pill_boxes) == 1, "expected exactly one box smaller than the node: the pill"

        text_w = pill_texts[0].w
        box_w = _width_of(pill_boxes[0])
        assert text_w < box_w, "the label claims the whole box, so it will be drawn off to the left"
        # The text, centred, must clear both rounded caps. Equality is the intent -- the box is built as
        # the text plus one cap's width at each end -- so this compares with a tolerance rather than
        # failing on the last bit of a float.
        assert 0.5 * (box_w - text_w) == pytest.approx(0.5 * chatgraph.LayoutConfig().pill_h), \
            "the label does not clear the rounded caps by exactly one radius each side"

    def test_sys_and_new_coincide_when_there_is_no_greeting(self):
        # The AI greeting is becoming optional, and optional per chat rather than globally -- so one
        # datastore will hold both shapes and a chat can start at the system prompt itself. A single-valued
        # pill would have to pick one of the two pointers to lose.
        forest = Forest()
        system = forest.create_node(payload("system", "you are helpful"), parent_id=None)
        user = forest.create_node(payload("user", "hello"), parent_id=system)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=user,
                                                            new_chat_node_id=system))
        assert built.refs[system].pills == ("SYS", "NEW")


# ---------------------------------------------------------------------------
# Sibling windowing
# ---------------------------------------------------------------------------

class TestSiblingWindow:
    def _fan(self, width: int, spine_index: int = 0):
        """A system prompt whose greeting has `width` children; the branch goes through one of them."""
        forest = Forest()
        system = forest.create_node(payload("system", "you are helpful"), parent_id=None)
        greeting = forest.create_node(payload("assistant", "hello!"), parent_id=system)
        chats = [forest.create_node(payload("user", f"chat {k}"), parent_id=greeting)
                 for k in range(width)]
        return forest, system, greeting, chats, chats[spine_index]

    def test_a_narrow_level_is_shown_whole(self):
        # The negative control for everything below: with three siblings there is nothing to hide, so a
        # windowing bug that produced gaps unconditionally would show up here rather than passing quietly.
        forest, system, greeting, chats, head = self._fan(width=3)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=head))
        assert all(chat in built.refs for chat in chats)
        assert refs_of_type(built, chatgraph.SiblingGapRef) == []

    def test_a_wide_level_is_windowed_and_the_ends_stay_visible(self):
        forest, system, greeting, chats, head = self._fan(width=20, spine_index=9)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=head))

        shown = [chat for chat in chats if chat in built.refs]
        assert len(shown) < len(chats), "a fan of twenty was not windowed at all"
        assert chats[0] in built.refs and chats[-1] in built.refs, "the ends of the fan are its anchors"
        assert head in built.refs

        gaps = refs_of_type(built, chatgraph.SiblingGapRef)
        hidden = [node_id for gap in gaps for node_id in gap.hidden_node_ids]
        assert sorted(hidden + shown) == sorted(chats), "every sibling is either shown or inside a gap"

    def test_a_gap_recenters_on_the_middle_of_what_it_hides(self):
        # Which is what makes repeated clicks bisect a wide fan instead of walking it at a fixed stride --
        # and is why this view wants no plus-or-minus-ten buttons of its own.
        forest, system, greeting, chats, head = self._fan(width=20, spine_index=0)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=head))
        gap = refs_of_type(built, chatgraph.SiblingGapRef)[0]
        assert gap.recenter_on == gap.hidden_node_ids[len(gap.hidden_node_ids) // 2]

    def test_the_branch_stays_visible_when_the_window_is_moved_away_from_it(self):
        # Moving the window is a preview gesture; it must not disconnect the picture from HEAD.
        forest, system, greeting, chats, head = self._fan(width=20, spine_index=0)
        far_away = chats[15]
        built = chatgraph.build(forest,
                                chatgraph.ViewState(head_node_id=head,
                                                    sibling_focus={greeting: far_away}))
        assert far_away in built.refs, "the window did not move"    # the fixture discriminates
        assert head in built.refs, "the branch to HEAD was windowed out of its own picture"


# ---------------------------------------------------------------------------
# Tool rounds
# ---------------------------------------------------------------------------

class TestToolRounds:
    """A round's results are folded away when there are enough of them to be worth a box, and are then
    reachable through that box.

    The threshold is the sibling and depth gaps' own, and it is what keeps the commonest round — one call,
    one result — free of the whole mechanism: a gap there spends a box to hide a box, and charges the
    reader a gesture to get the message back.
    """

    def _turn_with_tools(self, call_count: int = 3):
        """user -> assistant(N calls) -> N chained tool nodes -> assistant(reply)."""
        forest = Forest()
        system = forest.create_node(payload("system", "you are helpful"), parent_id=None)
        user = forest.create_node(payload("user", "what is the time"), parent_id=system)
        asking = forest.create_node(payload("assistant", "let me look",
                                            tool_calls=[{"id": str(k)} for k in range(call_count)]),
                                    parent_id=user)
        results = []
        parent = asking
        for k in range(call_count):
            parent = forest.create_node(payload("tool", f"result {k}"), parent_id=parent)
            results.append(parent)
        reply = forest.create_node(payload("assistant", "it is noon on Tuesday"), parent_id=parent)
        return forest, asking, tuple(results), reply

    def test_tool_results_fold_into_a_gap_below_the_round_that_asked_for_them(self):
        forest, asking, tool_nodes, reply = self._turn_with_tools()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        assert all(node_id not in built.refs for node_id in tool_nodes)
        assert built.refs[asking].tool_call_count == 3
        assert asking in built.refs and reply in built.refs, "both messages keep their own node"
        gap = only_ref_of_type(built, chatgraph.ToolRoundGapRef)
        assert gap.owner_node_id == asking
        assert gap.hidden_node_ids == tool_nodes, "the gap does not stand for what it folded away"

    def test_a_round_too_small_to_fold_draws_its_results(self):
        # The commonest round by a distance, and the case the threshold exists for. Two results are below
        # it, so they are drawn as ordinary boxes: reachable, committable, and costing no gesture.
        forest, asking, tool_nodes, reply = self._turn_with_tools(call_count=2)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        assert all(node_id in built.refs for node_id in tool_nodes), \
            "a round of two folded, so the threshold is not being applied"
        assert not refs_of_type(built, chatgraph.ToolRoundGapRef), \
            "a gap box was drawn for a round that hides nothing"

    def test_the_gap_sits_between_the_call_and_its_answer(self):
        # Which is the whole of what the box says. A round drawn anywhere else — beside the call, or below
        # the answer — would put the results somewhere the datastore does not have them.
        forest, asking, _tool_nodes, reply = self._turn_with_tools()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        gap = only_ref_of_type(built, chatgraph.ToolRoundGapRef)
        boxes = boxes_of(built)
        assert boxes[asking][3] <= boxes[gap.name][1], "the gap overlaps the message that asked"
        assert boxes[gap.name][3] <= boxes[reply][1], "the gap overlaps the answer"

    def test_the_gap_stands_in_the_spine_s_own_column(self):
        # The branch is a straight vertical line, and a box belonging to it has to be on that line: a gap
        # off to one side would read as a subtree hanging off the conversation rather than as part of it.
        forest, asking, _tool_nodes, reply = self._turn_with_tools()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        gap = only_ref_of_type(built, chatgraph.ToolRoundGapRef)
        boxes = boxes_of(built)

        def centre_x(name):
            return 0.5 * (boxes[name][0] + boxes[name][2])

        assert centre_x(gap.name) == pytest.approx(centre_x(asking))

    def test_expanding_a_round_shows_its_results(self):
        forest, asking, tool_nodes, reply = self._turn_with_tools()
        built = chatgraph.build(forest,
                                chatgraph.ViewState(head_node_id=reply,
                                                    expanded_tool_turns={asking}))
        assert all(node_id in built.refs for node_id in tool_nodes)
        assert not refs_of_type(built, chatgraph.ToolRoundGapRef), \
            "the gap is still there beside the results it was hiding"

    def test_an_opened_round_says_it_can_be_closed_again(self):
        # What `Backspace` reads. Without this table a caller has to walk the forest to find out which
        # round the box under the cursor belongs to, and the picture already knows.
        forest, asking, tool_nodes, reply = self._turn_with_tools()
        built = chatgraph.build(forest,
                                chatgraph.ViewState(head_node_id=reply,
                                                    expanded_tool_turns={asking}))
        assert built.expanded_rounds == {asking: tool_nodes}

    def test_a_round_too_small_to_fold_cannot_be_closed(self):
        # The negative control for the table above, and not a quibble: a round of one is drawn open
        # whatever the reader asks, so offering to close it would be a key that appears to do nothing.
        forest, asking, _tool_nodes, reply = self._turn_with_tools(call_count=1)
        built = chatgraph.build(forest,
                                chatgraph.ViewState(head_node_id=reply,
                                                    expanded_tool_turns={asking}))
        assert built.expanded_rounds == {}

    def test_the_badge_counts_calls_rather_than_results(self):
        # A turn stopped mid-round has fewer results than calls, and the badge should say what was asked
        # for. Counting the nodes below would report the interruption as a smaller request.
        forest = Forest()
        user = forest.create_node(payload("user", "search for three things"), parent_id=None)
        asking = forest.create_node(payload("assistant", "on it",
                                            tool_calls=[{"id": "1"}, {"id": "2"}, {"id": "3"}]),
                                    parent_id=user)
        forest.create_node(payload("tool", "only one came back"), parent_id=asking)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=asking))
        assert built.refs[asking].tool_call_count == 3


# ---------------------------------------------------------------------------
# Truncation, and the rule that it must always show itself
# ---------------------------------------------------------------------------

class TestTruncation:
    def test_a_short_branch_has_no_depth_gap(self):
        # The control for the case below.
        forest = Forest()
        ids = chain(forest, length=4)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=ids[-1]))
        assert refs_of_type(built, chatgraph.DepthGapRef) == []

    def test_a_long_branch_keeps_its_root_and_gains_one_gap(self):
        forest = Forest()
        ids = chain(forest, length=30)
        config = chatgraph.LayoutConfig(max_visible_depth=8)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=ids[-1]), config)

        gaps = refs_of_type(built, chatgraph.DepthGapRef)
        assert len(gaps) == 1
        assert ids[0] in built.refs, "the root carries SYS and names the card; it is never the part elided"
        assert ids[-1] in built.refs
        shown_spine = [node_id for node_id in ids if node_id in built.refs]
        assert sorted(shown_spine + list(gaps[0].hidden_node_ids)) == sorted(ids)

    def test_the_depth_gap_stands_between_the_root_and_what_follows(self):
        # The truncation rule in its strict form: a node with no visible links has to mean the graph really
        # ends there. A root wired straight to a distant descendant would instead be a lie about adjacency.
        forest = Forest()
        ids = chain(forest, length=30)
        config = chatgraph.LayoutConfig(max_visible_depth=8)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=ids[-1]), config)

        gap_node = only_depth_gap(built)
        root_node = built.graph.get_node_by_name(ids[0])
        assert any(edge.src is root_node and edge.dst is gap_node for edge in built.graph.edges)
        assert any(edge.src is gap_node for edge in built.graph.edges)
        assert not any(edge.src is root_node and edge.dst is not gap_node for edge in built.graph.edges)

    def _long_chat(self, n_chats=8, depth=25):
        """A card, a greeting, a fan of chats under it, and one of them carried on well past the budget."""
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        greeting = forest.create_node(payload("assistant", "hello!"), parent_id=root)
        sessions = [forest.create_node(payload("user", f"chat {k} opens"), parent_id=greeting)
                    for k in range(n_chats)]
        node = sessions[3]
        for d in range(depth):
            node = forest.create_node(payload("assistant" if d % 2 == 0 else "user", f"message {d}"),
                                      parent_id=node)
        return forest, root, greeting, sessions, node

    def test_the_way_out_of_a_long_chat_stays_on_screen(self):
        # The session level -- the children of `new_chat_HEAD` -- doubles as the list of recent chats, so
        # eliding it strands the reader in the conversation they are trying to leave. It is pinned against
        # the depth window the way the root is.
        forest, root, greeting, sessions, head = self._long_chat()
        config = chatgraph.LayoutConfig(max_visible_depth=8)
        built = chatgraph.build(forest,
                                chatgraph.ViewState(head_node_id=head, new_chat_node_id=greeting), config)

        assert refs_of_type(built, chatgraph.DepthGapRef), \
            "this branch is not long enough to be truncated, so the pinning is not being tested"
        assert root in built.refs and greeting in built.refs
        assert sessions[3] in built.refs, "the chat this branch belongs to was elided"
        assert any(other in built.refs for other in sessions if other != sessions[3]), \
            "the session level is there but its siblings are not, so there is still no way to another chat"

    def test_without_the_pointer_the_session_level_goes(self):
        # The control for the case above, and it is the behaviour that prompted the change: knowing where
        # the session level *is* takes `new_chat_HEAD`, and without it the window can only keep the root.
        forest, root, greeting, sessions, head = self._long_chat()
        config = chatgraph.LayoutConfig(max_visible_depth=8)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=head), config)
        assert root in built.refs
        assert sessions[3] not in built.refs

    def test_the_gap_sits_below_the_pinned_prefix(self):
        forest, root, greeting, sessions, head = self._long_chat()
        config = chatgraph.LayoutConfig(max_visible_depth=8)
        built = chatgraph.build(forest,
                                chatgraph.ViewState(head_node_id=head, new_chat_node_id=greeting), config)

        gap_node = only_depth_gap(built)
        session_node = built.graph.get_node_by_name(sessions[3])
        assert any(edge.src is session_node and edge.dst is gap_node for edge in built.graph.edges), \
            "the elision is between the session level and HEAD, so that is where the gap belongs"
        assert gap_node.y > built.graph.get_node_by_name(root).y

    def test_a_prefix_that_would_crowd_out_head_is_not_pinned(self):
        # `new_chat_HEAD` is normally one node under the root, but nothing guarantees it. A prefix reaching
        # halfway down the budget would answer "where am I" by dropping "what is happening".
        forest = Forest()
        ids = chain(forest, length=40)
        config = chatgraph.LayoutConfig(max_visible_depth=8)
        built = chatgraph.build(forest,
                                chatgraph.ViewState(head_node_id=ids[-1], new_chat_node_id=ids[20]),
                                config)
        assert ids[20] not in built.refs
        assert ids[0] in built.refs and ids[-1] in built.refs

    def test_an_off_spine_sibling_with_children_says_so(self):
        # Two children, because one is inlined instead: see `TestBranchExtent`. A gap is what stands in
        # for a branch too wide to draw here.
        forest = Forest()
        system = forest.create_node(payload("system", "you are helpful"), parent_id=None)
        taken = forest.create_node(payload("user", "the branch we are on"), parent_id=system)
        not_taken = forest.create_node(payload("user", "the one we are not"), parent_id=system)
        for k in range(2):
            forest.create_node(payload("assistant", f"a whole conversation below here {k}"),
                               parent_id=not_taken)

        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=taken))
        gaps = refs_of_type(built, chatgraph.SubtreeGapRef)
        assert [gap.node_id for gap in gaps] == [not_taken]
        assert gaps[0].child_count == 2

    def test_a_childless_off_spine_sibling_gets_no_gap(self):
        # The control for the case above: a gap under every sibling would satisfy it just as well.
        forest = Forest()
        system = forest.create_node(payload("system", "you are helpful"), parent_id=None)
        taken = forest.create_node(payload("user", "the branch we are on"), parent_id=system)
        forest.create_node(payload("user", "a chat that never went anywhere"), parent_id=system)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=taken))
        assert refs_of_type(built, chatgraph.SubtreeGapRef) == []

    def test_other_roots_are_declared_even_though_v1_cannot_visit_them(self):
        forest = Forest()
        current = forest.create_node(payload("system", "the card in use"), parent_id=None)
        head = forest.create_node(payload("user", "hello"), parent_id=current)
        older = forest.create_node(payload("system", "an older version of the card"), parent_id=None)

        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=head))
        gaps = refs_of_type(built, chatgraph.RootGapRef)
        assert len(gaps) == 1 and gaps[0].hidden_node_ids == (older,)

    def test_a_lone_root_declares_nothing(self, conversation):
        forest, system, greeting, user, reply = conversation
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        assert refs_of_type(built, chatgraph.RootGapRef) == []


# ---------------------------------------------------------------------------
# Geometry the eye would catch and no other test would
# ---------------------------------------------------------------------------

class TestGeometry:
    def _crowded(self):
        """A tree exercising every box kind at once: a fan, off-spine subtrees, and a wide row below HEAD.

        HEAD needs *several* children, not one. The collision a reserved band exists to prevent is between
        a subtree gap and the row under it, and a single child sits on the spine's own line -- directly
        below HEAD, where no off-spine sibling's gap ever is. A one-child fixture therefore cannot fail
        however the bands are computed.
        """
        forest = Forest()
        system = forest.create_node(payload("system", "you are helpful"), parent_id=None)
        greeting = forest.create_node(payload("assistant", "hello!"), parent_id=system)
        # Thirty, so that the fan is wider than any plausible `siblings_each_side` and the row still
        # has gaps in it. Sized to the shipped default once, it went vacuous the day that default was
        # raised -- the window swallowed the whole fan, the gap assertions iterated over nothing, and the
        # tests went on passing while they stopped testing.
        chats = [forest.create_node(payload("user", f"chat {k}"), parent_id=greeting) for k in range(30)]
        # Every off-spine sibling has a continuation, so every one puts something in the band -- and the
        # band holds two different widths, which is what makes this fixture worth its size. An only child
        # is inlined at full node width; three children stay a gap box, which is narrower. The wide one is
        # the one that can reach its neighbour's column, so a fixture with only gaps in the band would
        # miss exactly the collision worth checking.
        for k, chat in enumerate(chats):
            for j in range(1 if k % 2 else 3):
                forest.create_node(payload("assistant", f"and so on {j}"), parent_id=chat)
        head = chats[15]
        for k in range(4):
            forest.create_node(payload("assistant", f"reroll {k}"), parent_id=head)
        return forest, head

    def test_no_two_boxes_overlap(self):
        forest, head = self._crowded()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=head))
        boxes = boxes_of(built)

        # The control, and it has to be this specific. What makes the band load-bearing is a subtree gap
        # with something drawn *below its sibling and in its column*; a fixture whose lower row is
        # narrower than the fan above it passes with or without the band, while looking like a thorough
        # test of both. Phrased against the sibling rather than against the gap on purpose -- measuring
        # from the gap would make the control's own answer depend on the placement it is there to check.
        gap_refs = refs_of_type(built, chatgraph.SubtreeGapRef)
        assert gap_refs, "no subtree gaps here, so this fixture says nothing about the band"
        assert any(boxes[ref.node_id][3] < box[1]
                   and boxes[ref.name][0] < box[2] and box[0] < boxes[ref.name][2]
                   for ref in gap_refs for box in boxes.values()), \
            ("nothing is drawn below a sibling that has a subtree gap, and within that gap's column, so "
             "this fixture cannot tell a reserved band from an unreserved one")

        # The second control, for the other thing a band can hold. An inlined child is a full-width box
        # in a column sized for one, so it is the case that can reach a neighbour -- and a fixture whose
        # bands held only the narrower gap boxes could not collide however wrong the placement was.
        band_ys = {boxes[ref.name][1] for ref in gap_refs}
        assert any(box[1] in band_ys and (box[2] - box[0]) > chatgraph.LayoutConfig().gap_node_w
                   for box in boxes.values()), \
            "nothing full-width is drawn in a band, so this fixture cannot detect a column overrun"

        assert overlapping_pairs(built) == []

    def test_the_graph_box_contains_everything_drawn(self):
        # Pills are drawn in the space above their node, so they fall outside every node box there is, and
        # a fit computed from node boxes alone clips the one label that says where HEAD is.
        #
        # Zero margin on purpose. At the default margin the pills happen to be exactly as tall as the
        # margin is wide, so a node-box fit lands them at y = 0 and passes -- an accident of two unrelated
        # constants, which would go on satisfying this test until somebody changed either one.
        forest, head = self._crowded()
        config = chatgraph.LayoutConfig(margin=0.0)
        built = chatgraph.build(forest,
                                chatgraph.ViewState(head_node_id=head, new_chat_node_id=head), config)

        shape_boxes = [box for node in built.graph.nodes
                       for box in (shape.get_bounding_box() for shape in node.shapes)
                       if box is not None]
        top_of_boxes = min(node.get_bounding_box()[1] for node in built.graph.nodes)
        assert min(box[1] for box in shape_boxes) < top_of_boxes, \
            "nothing is drawn above the topmost node box, so this fixture cannot detect clipping"

        assert min(box[0] for box in shape_boxes) >= 0.0
        assert min(box[1] for box in shape_boxes) >= 0.0
        assert max(box[2] for box in shape_boxes) <= built.graph.width
        assert max(box[3] for box in shape_boxes) <= built.graph.height

    def test_a_gap_is_drawn_at_the_width_the_row_reserved_for_it(self):
        # Boxes are placed by width, so a box drawn wider than its slot slides under its neighbour --
        # invisible in any assertion about which nodes exist, and obvious on screen.
        forest, head = self._crowded()
        config = chatgraph.LayoutConfig()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=head), config)
        gap_refs = refs_of_type(built, chatgraph.SiblingGapRef)
        assert gap_refs, "no sibling gaps in this fixture, so the loop below asserts nothing"
        for ref in gap_refs:
            box = built.graph.get_node_by_name(ref.name).get_bounding_box()
            assert box[2] - box[0] == pytest.approx(config.gap_node_w)

    def test_the_branchs_frame_contains_what_is_drawn_on_the_branch(self):
        """What a view fits on opening. A pill hangs above its node, so the topmost box of the branch has
        part of itself outside its own rectangle — and a frame built from rectangles clips the SYS pill off
        the top, which is the one label saying which card this conversation belongs to."""
        forest = Forest()
        system = forest.create_node(payload("system", "you are helpful"), parent_id=None)
        user = forest.create_node(payload("user", "hello"), parent_id=system)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=user, new_chat_node_id=system))

        top_node = built.graph.get_node_by_name(system)
        assert built.refs[system].pills, "the top of this branch wears no pill, so nothing can be clipped"
        shape_tops = [box[1] for box in (s.get_bounding_box() for s in top_node.shapes) if box is not None]
        assert min(shape_tops) < top_node.get_bounding_box()[1], \
            "nothing is drawn above the top node's box, so this fixture cannot detect the clipping"

        assert built.spine_bbox[1] <= min(shape_tops)

    def test_the_picture_starts_at_the_origin(self):
        # `Viewport.zoom_to_fit` fits the box (0, 0)-(width, height) and nothing else, so content placed
        # outside it is simply not framed.
        forest, head = self._crowded()
        config = chatgraph.LayoutConfig()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=head), config)
        boxes = list(boxes_of(built).values())
        assert min(box[0] for box in boxes) >= 0.0
        assert min(box[1] for box in boxes) >= 0.0
        assert built.graph.width > 0.0 and built.graph.height > 0.0


# ---------------------------------------------------------------------------
# Reading a forest that is being written to
# ---------------------------------------------------------------------------

class TestBranchExtent:
    """The four rules about how much of a branch to draw, settled 2026-09-01.

    They exist because clicking back onto the current branch used to leave the message below the click
    collapsed as a bare "…1 more" -- one hidden box, announced by a box.
    """

    def _branch(self, before=4, after=6):
        """A card, a greeting, then a chain; returns the node `before` steps in, and the whole chain."""
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        ids = chain(forest, length=before + after, parent_id=root)
        return forest, root, ids, ids[before - 1]

    # -- Rule 2: the drawn branch is a whole branch, not a stump -------------

    def test_focusing_mid_branch_still_draws_it_to_the_tip(self):
        forest, root, ids, focus = self._branch()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=ids[-1], focus_node_id=focus))
        assert ids[-1] in built.refs, "the branch was truncated at the focus instead of running to its tip"
        assert all(node_id in built.refs for node_id in ids), \
            "some of the branch below the focus went missing"

    def test_the_continuation_is_branch_rather_than_decoration(self):
        # Being *on screen* is not the claim; being on the spine is. Truncating at the focus still draws
        # the focus's children, as a row hanging below it and then as inlined boxes in bands -- so a test
        # that only asked whether those nodes were drawn would pass against the stump.
        forest, root, ids, focus = self._branch()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=ids[-1], focus_node_id=focus))
        continuation = ids[ids.index(focus):]
        assert all(node_id in built.spine for node_id in continuation), \
            "the branch below the focus is drawn as something hanging off it rather than as more branch"

    def test_a_focus_on_a_leaf_ends_the_branch_there(self):
        # The control for the two above: descending from a leaf must not invent anything, or "draw to the
        # tip" would be indistinguishable from "draw one extra row whatever happens".
        forest, root, ids, _focus = self._branch()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=ids[-1], focus_node_id=ids[-1]))
        drawn = [node_id for node_id in ids if node_id in built.refs]
        assert drawn == ids
        assert refs_of_type(built, chatgraph.SubtreeGapRef) == []

    def test_the_branch_followed_is_the_most_recent_one(self):
        # Two continuations under the focus, and the descent has to pick the later-written one -- the same
        # branch the chat itself would resume into.
        forest, root, ids, focus = self._branch(before=3, after=0)
        older = forest.create_node(payload("user", "the road not taken"), parent_id=focus)
        newer = forest.create_node(payload("user", "the one we were on"), parent_id=focus)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=newer, focus_node_id=focus))
        assert built.spine[-1] == newer, f"the spine ran to {built.spine[-1]!r}, not to the latest child"
        assert older in built.refs, "the sibling is narrow enough to draw, so it should still be visible"

    # -- Rule 1: a gap that hides less than it costs is not drawn ------------

    def test_two_omitted_siblings_are_drawn_rather_than_hidden(self):
        # A gap occupies a slot, so hiding two trades two nodes for one box that names neither.
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        greeting = forest.create_node(payload("assistant", "hello!"), parent_id=root)
        # `siblings_each_side=1` keeps indices 0, 1, 2 around a focus at 1, plus the last one. At six
        # that omits exactly two, which is under the threshold.
        chats = [forest.create_node(payload("user", f"chat {k}"), parent_id=greeting) for k in range(6)]
        config = chatgraph.LayoutConfig(siblings_each_side=1)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=chats[1]), config)

        assert all(chat in built.refs for chat in chats), \
            "a run of two was hidden behind a gap that costs a slot to say so"
        assert refs_of_type(built, chatgraph.SiblingGapRef) == []

    def test_three_omitted_siblings_do_get_a_gap(self):
        # The other side of the threshold, and the control for the test above: without it, a windowing
        # bug that never hid anything would satisfy that assertion perfectly. One more sibling than the
        # fixture above, which is the whole difference between inlining and a gap.
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        greeting = forest.create_node(payload("assistant", "hello!"), parent_id=root)
        chats = [forest.create_node(payload("user", f"chat {k}"), parent_id=greeting) for k in range(7)]
        config = chatgraph.LayoutConfig(siblings_each_side=1)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=chats[1]), config)

        gaps = refs_of_type(built, chatgraph.SiblingGapRef)
        assert len(gaps) == 1
        assert len(gaps[0].hidden_node_ids) == 3

    def _off_spine_with(self, n_children: int):
        """A branch plus one sibling that was abandoned after `n_children` continuations."""
        forest = Forest()
        system = forest.create_node(payload("system", "you are helpful"), parent_id=None)
        taken = forest.create_node(payload("user", "the branch we are on"), parent_id=system)
        not_taken = forest.create_node(payload("user", "the one we are not"), parent_id=system)
        kids = [forest.create_node(payload("assistant", f"reply {k}"), parent_id=not_taken)
                for k in range(n_children)]
        return forest, taken, not_taken, kids

    def test_an_only_child_is_drawn_instead_of_counted(self):
        # A gap box saying "…1 more" spends the slot the message would have taken to report the number
        # one. The child is drawn there instead, as itself and clickable as itself.
        forest, taken, not_taken, kids = self._off_spine_with(1)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=taken))
        assert [ref for ref in refs_of_type(built, chatgraph.SubtreeGapRef)
                if ref.node_id == not_taken] == []
        assert kids[0] in built.refs, "the only child was neither drawn nor announced"
        assert isinstance(built.refs[kids[0]], chatgraph.ChatNodeRef), \
            "the inlined child is not a chat node, so clicking it cannot preview the message"

    def test_two_children_stay_a_gap(self):
        # The control, and the boundary: two node-width boxes do not fit one column, so unlike the
        # sibling and depth thresholds this one cuts at one.
        forest, taken, not_taken, kids = self._off_spine_with(2)
        gaps = refs_of_type(chatgraph.build(forest, chatgraph.ViewState(head_node_id=taken)),
                            chatgraph.SubtreeGapRef)
        assert [gap.node_id for gap in gaps] == [not_taken]

    def test_an_inlined_child_with_children_of_its_own_gets_a_gap_of_its_own(self):
        # Otherwise it is a node with no visible links, which in this picture means the graph ends here.
        # It gets the same marker as anything else whose continuation is not drawn: one vocabulary, so a
        # reader who has learnt what one dashed box means has learnt them all.
        forest, taken, not_taken, kids = self._off_spine_with(1)
        forest.create_node(payload("user", "and it went on"), parent_id=kids[0])
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=taken))

        assert kids[0] in built.refs, "the only child was not inlined, so there is nothing below it to mark"
        gaps = [ref for ref in refs_of_type(built, chatgraph.SubtreeGapRef) if ref.node_id == kids[0]]
        assert len(gaps) == 1, "the inlined child's own continuation is not announced anywhere"

    def test_an_inlined_leaf_says_nothing_of_the_sort(self):
        # The control: a gap under every inlined child would satisfy the test above and would be a lie
        # here, where the branch really does end.
        forest, taken, not_taken, kids = self._off_spine_with(1)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=taken))
        assert refs_of_type(built, chatgraph.SubtreeGapRef) == [], \
            "a leaf claims the branch continues below it"

    def test_a_node_whose_children_are_drawn_needs_no_gap(self):
        # The other control: a node on the spine has children and they are the row below it, which needs
        # no saying. A marker there would be noise on every box in the picture.
        forest, taken, not_taken, kids = self._off_spine_with(0)
        forest.create_node(payload("assistant", "the reply"), parent_id=taken)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=taken))
        assert [ref for ref in refs_of_type(built, chatgraph.SubtreeGapRef)
                if ref.node_id == taken] == [], "a node whose children are drawn still claims more"

    def test_an_inlined_child_is_drawn_at_the_depth_it_has(self):
        # It is a real message, so it has a real depth, and the spine's own next node has the same one.
        # Putting them on two rows makes the spine appear to skip a level while its neighbours have
        # content at that level -- the picture lying about the thing it exists to show.
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        taken = forest.create_node(payload("user", "the branch we are on"), parent_id=root)
        deeper = forest.create_node(payload("assistant", "the reply, one level down"), parent_id=taken)
        not_taken = forest.create_node(payload("user", "the one we are not"), parent_id=root)
        inlined = forest.create_node(payload("assistant", "also one level down"), parent_id=not_taken)

        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=deeper))
        assert inlined in built.refs, "the only child was not inlined, so there is no placement to check"
        assert (built.graph.get_node_by_name(inlined).y
                == built.graph.get_node_by_name(deeper).y), \
            "the inlined child and the spine's next node are the same depth and were drawn on two rows"

    def test_a_subtree_gap_is_drawn_at_the_depth_it_stands_for(self):
        # It stands for messages one level below the node it hangs from, so it has a depth like anything
        # else. Putting it in a band cost a whole row of height for the *entire* level -- and the level
        # is as wide as the widest fan, so one gap far off to the side stretched the part the reader was
        # actually looking at. Seen on 2026-09-02: a box at x=170 pushed everything from x=638 rightward
        # down a row, and the middle of the picture went empty.
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        taken = forest.create_node(payload("user", "the branch we are on"), parent_id=root)
        deeper = forest.create_node(payload("assistant", "the reply, one level down"), parent_id=taken)
        not_taken = forest.create_node(payload("user", "the one we are not"), parent_id=root)
        for k in range(2):  # two children, so this stays a gap rather than being inlined
            forest.create_node(payload("assistant", f"reroll {k}"), parent_id=not_taken)

        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=deeper))
        gap = self._gap_under(built, not_taken)
        assert (built.graph.get_node_by_name(gap).y
                == built.graph.get_node_by_name(deeper).y), \
            "the gap stands for content one level down and was drawn on a row of its own"
        assert len({node.y for node in built.graph.nodes}) == 3, \
            "the tree is three levels deep and the picture uses more, so a band was reserved after all"

    def test_a_subtree_gap_with_nowhere_to_go_takes_a_band(self):
        # The control for the above, and what the band is still for: where the level below has no free
        # column, adjacent beats overlapping.
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        taken = forest.create_node(payload("user", "the branch we are on"), parent_id=root)
        kids = [forest.create_node(payload("assistant", f"reply {k}"), parent_id=taken) for k in range(9)]
        not_taken = forest.create_node(payload("user", "the one we are not"), parent_id=root)
        for k in range(2):
            forest.create_node(payload("assistant", f"reroll {k}"), parent_id=not_taken)

        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=kids[4]))
        row_ys = {built.graph.get_node_by_name(k).y for k in kids if k in built.refs}
        assert len(row_ys) == 1, "the fan is not one row, so this fixture cannot say a column was taken"
        assert built.graph.get_node_by_name(self._gap_under(built, not_taken)).y not in row_ys, \
            "the gap was placed into an occupied column instead of falling back to a band"
        assert overlapping_pairs(built) == []

    def test_a_child_with_nowhere_to_go_is_counted_rather_than_misplaced(self):
        # Rows pack independently, so the level below does not reserve a column under every parent. Where
        # it has none free, inlining is not worth it: the gap box says truthfully that something is not
        # drawn, while a message drawn a level away from its own depth says something false about where
        # it sits, which is the fault the depth placement exists to remove.
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        taken = forest.create_node(payload("user", "the branch we are on"), parent_id=root)
        # A wide fan one level down, so every column at that depth is occupied.
        kids = [forest.create_node(payload("assistant", f"reply {k}"), parent_id=taken) for k in range(9)]
        not_taken = forest.create_node(payload("user", "the one we are not"), parent_id=root)
        crowded_out = forest.create_node(payload("assistant", "nowhere to go"), parent_id=not_taken)

        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=kids[4]))
        row_ys = {built.graph.get_node_by_name(k).y for k in kids if k in built.refs}
        assert len(row_ys) == 1, "the fan is not one row, so this fixture cannot say a column was taken"
        assert crowded_out not in built.refs, \
            "the child was drawn even though its own level had no room for it"
        assert [ref.node_id for ref in refs_of_type(built, chatgraph.SubtreeGapRef)] == [not_taken]
        assert overlapping_pairs(built) == []

    def _gap_under(self, built, node_id):
        """Return the graph node of the subtree gap hanging from `node_id`."""
        ref = next(r for r in refs_of_type(built, chatgraph.SubtreeGapRef) if r.node_id == node_id)
        return ref.name

    def test_a_downward_gap_says_both_how_many_and_how_deep(self):
        # One number cannot describe a subtree. A fan of five leaves and a chain of five messages are
        # both "five", and they are not the same thing to a reader deciding whether to click -- so the
        # label counts the messages and a quieter second line says how far down they go.
        forest, taken, not_taken, kids = self._off_spine_with(1)
        node = kids[0]
        for _ in range(3):
            node = forest.create_node(payload("user", "on and on"), parent_id=node)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=taken))
        texts = texts_on(built, self._gap_under(built, kids[0]))
        assert "…3 more" in texts and "3 levels" in texts, \
            f"the gap says {texts}, which does not give both the count and the depth"

    def test_a_flat_fan_is_not_reported_as_one(self):
        # The control, and the case that decided the unit: five leaves under one node are five messages
        # one level down. Counting levels alone called that "…1 more" on the live datastore.
        forest, taken, not_taken, kids = self._off_spine_with(1)
        for k in range(5):
            forest.create_node(payload("user", f"a reroll {k}"), parent_id=kids[0])
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=taken))
        texts = texts_on(built, self._gap_under(built, kids[0]))
        assert "…5 more" in texts, f"the gap says {texts}, understating five messages"
        assert "1 level" in texts, f"the gap says {texts}; the fan is one level deep and should say so"

    def test_a_forked_continuation_gets_a_range(self):
        # A branch that forks has no single answer for its depth, and the short arm is as much use as
        # the long one.
        forest, taken, not_taken, kids = self._off_spine_with(1)
        forest.create_node(payload("user", "the short arm"), parent_id=kids[0])
        node = forest.create_node(payload("user", "the long arm"), parent_id=kids[0])
        for _ in range(3):
            node = forest.create_node(payload("user", "on and on"), parent_id=node)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=taken))
        texts = texts_on(built, self._gap_under(built, kids[0]))
        assert "…5 more" in texts and "1–4 levels" in texts, \
            f"the gap says {texts}, which does not report both arms"

    def test_every_gap_wears_the_ellipsis(self):
        # One phrasing across all of them: the leading ellipsis is what marks a box as standing for
        # content rather than holding any, and a reader who has seen one has seen them all.
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        greeting = forest.create_node(payload("assistant", "hello!"), parent_id=root)
        chats = [forest.create_node(payload("user", f"chat {k}"), parent_id=greeting) for k in range(30)]
        head = chain(forest, length=15, parent_id=chats[3])[-1]        # long enough for a depth gap
        for k in range(2):                                             # two children: a subtree gap
            forest.create_node(payload("assistant", f"reroll {k}"), parent_id=chats[7])
        forest.create_node(payload("system", "an older card"), parent_id=None)  # a root gap

        config = chatgraph.LayoutConfig(max_visible_depth=8)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=head,
                                                            new_chat_node_id=greeting), config)

        kinds = {chatgraph.SiblingGapRef, chatgraph.DepthGapRef,
                 chatgraph.SubtreeGapRef, chatgraph.RootGapRef}
        seen = {type(ref) for ref in built.refs.values()} & kinds
        assert seen == kinds, f"only {sorted(k.__name__ for k in seen)} here; not all four are checked"
        for ref in built.refs.values():
            if type(ref) in kinds:
                texts = texts_on(built, ref.name)
                assert any(text.startswith("…") for text in texts), \
                    f"{type(ref).__name__} says {texts}, without the ellipsis every other gap wears"

    def test_nothing_hanging_off_a_box_reaches_another_one(self):
        # The overlap test compares *node* boxes, and a pill is drawn in the space above the box it
        # belongs to -- so it is visible to neither. That blind spot hid a marker overlapping the row
        # above it by 7.2 units of a 44-unit gap.
        forest, old_chat, head, new_chat = self._head_buried(branches=1)
        for k in range(4):
            forest.create_node(payload("user", f"another chat {k}"), parent_id=new_chat)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=head, focus_node_id=new_chat))

        boxes = boxes_of(built)
        overhanging = [n for n in built.graph.nodes
                       if any(s.get_bounding_box() is not None
                              and (s.get_bounding_box()[1] < boxes[n.internal_name][1] - 0.5
                                   or s.get_bounding_box()[3] > boxes[n.internal_name][3] + 0.5)
                              for s in n.shapes)]
        assert overhanging, "nothing is drawn outside its own box here, so this fixture checks nothing"
        for node in overhanging:
            drawn = [s.get_bounding_box() for s in node.shapes if s.get_bounding_box() is not None]
            top, bottom = min(b[1] for b in drawn), max(b[3] for b in drawn)
            for name, box in boxes.items():
                if name == node.internal_name or not (box[0] < node.x < box[2]):
                    continue
                assert box[3] <= top or box[1] >= bottom, \
                    f"what hangs off {node.internal_name[-8:]} reaches into {name[-8:]}"

    def test_a_depth_overrun_of_two_is_drawn_rather_than_gapped(self):
        # The same threshold, and deliberately the same number: two rules here would disagree in front of
        # the reader, who sees only that one row inlined its leftovers and another did not.
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        ids = chain(forest, length=9, parent_id=root)
        config = chatgraph.LayoutConfig(max_visible_depth=8)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=ids[-1]), config)

        assert refs_of_type(built, chatgraph.DepthGapRef) == [], \
            "a two-node overrun bought a gap box, which is a slot spent to hide two slots"
        assert all(node_id in built.refs for node_id in ids)

    # -- Rule 3: the depth window follows the focus --------------------------

    def _long(self, depth=40):
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        ids = chain(forest, length=depth, parent_id=root)
        return forest, root, ids

    def test_the_window_follows_the_focus_rather_than_the_tip(self):
        forest, root, ids = self._long()
        config = chatgraph.LayoutConfig(max_visible_depth=10)
        focus = ids[12]
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=ids[-1], focus_node_id=focus),
                                config)
        assert focus in built.refs
        assert ids[13] in built.refs, "nothing of the focus's own neighbourhood survived the window"

    def test_the_window_reaches_both_ways_from_the_focus(self):
        forest, root, ids = self._long()
        config = chatgraph.LayoutConfig(max_visible_depth=10)
        focus = ids[20]
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=ids[-1], focus_node_id=focus),
                                config)
        assert ids[19] in built.refs, "nothing above the focus"
        assert ids[21] in built.refs, "nothing below the focus"

    def test_the_floors_hold_when_the_budget_is_already_spent(self):
        # The floors are what make the focus navigable, and they only bite when there is nothing left to
        # spend: one node above is the step-up handle, and a single node below is too few to say where
        # the conversation went. A roomy budget reaches both ways on its own, so a fixture with one
        # cannot tell a floor from an accident -- hence a budget the pins have all but used up.
        forest, root, ids = self._long()
        config = chatgraph.LayoutConfig(max_visible_depth=4)
        focus = ids[20]
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=ids[-1], focus_node_id=focus),
                                config)
        assert ids[19] in built.refs, "no node above the focus, so stepping one level up needs a gap click"
        below = [node_id for node_id in ids[21:24] if node_id in built.refs]
        assert len(below) == 3, f"only {len(below)} of the three nodes below the focus were drawn"

    def test_the_tip_is_kept_however_far_the_focus_is_from_it(self):
        # Without it a long branch fades out mid-conversation and the reader cannot tell whether they are
        # near the end of it.
        #
        # HEAD sits *beside the focus* rather than at the tip, which is load-bearing: HEAD is pinned too,
        # so a fixture with HEAD at the tip keeps the tip either way and cannot tell the two pins apart.
        # It is the ordinary arrangement, at that -- HEAD may point at an internal node, and previewing
        # from one is exactly when this question comes up.
        forest, root, ids = self._long()
        config = chatgraph.LayoutConfig(max_visible_depth=10)
        built = chatgraph.build(forest,
                                chatgraph.ViewState(head_node_id=ids[6], focus_node_id=ids[5]), config)
        assert ids[-1] not in (ids[5], ids[6]), "HEAD or the focus is the tip, so nothing here needs pinning"
        assert refs_of_type(built, chatgraph.DepthGapRef), \
            "nothing was elided, so the tip is on screen for want of a window rather than because it is pinned"
        assert ids[-1] in built.refs, "the end of the branch is not on screen"

    def test_a_focus_in_the_middle_gets_a_gap_on_each_side(self):
        # The structural consequence: prefix, gap, window, gap, tip. A window that stayed at the end of
        # the branch would leave one run and therefore one gap.
        #
        # HEAD is at the tip here so that it pins nothing of its own -- put it beside the focus instead
        # and it becomes a third kept island, which yields two gaps whether the window followed the focus
        # or not. Pinning the *tip* is left to the test above for the same reason; one confound per
        # fixture, and each one somewhere it cannot hide.
        forest, root, ids = self._long()
        config = chatgraph.LayoutConfig(max_visible_depth=10)
        focus = ids[20]
        assert len(ids) - 1 - 20 > config.max_visible_depth, \
            "the focus is close enough to the tip that a window at either would look the same"
        built = chatgraph.build(forest,
                                chatgraph.ViewState(head_node_id=ids[-1], focus_node_id=focus), config)
        gaps = refs_of_type(built, chatgraph.DepthGapRef)
        assert len(gaps) == 2, f"expected a gap above the window and one below it, got {len(gaps)}"

    # -- Rule 4: the focus and HEAD both want to be on screen ----------------

    def test_head_survives_a_depth_window_centred_elsewhere(self):
        forest, root, ids = self._long()
        config = chatgraph.LayoutConfig(max_visible_depth=8)
        head = ids[30]
        built = chatgraph.build(forest,
                                chatgraph.ViewState(head_node_id=head, focus_node_id=ids[5]), config)
        assert refs_of_type(built, chatgraph.DepthGapRef), \
            "nothing was elided, so this fixture cannot tell a pinned HEAD from a lucky one"
        assert head in built.refs, "HEAD left the picture the preview exists to compare against"

    def test_head_survives_a_sibling_window_moved_off_it(self):
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        greeting = forest.create_node(payload("assistant", "hello!"), parent_id=root)
        chats = [forest.create_node(payload("user", f"chat {k}"), parent_id=greeting) for k in range(30)]
        config = chatgraph.LayoutConfig(siblings_each_side=2)
        built = chatgraph.build(forest,
                                chatgraph.ViewState(head_node_id=chats[3], focus_node_id=chats[25]),
                                config)
        assert refs_of_type(built, chatgraph.SiblingGapRef), \
            "the fan was drawn whole, so nothing here is being windowed out to begin with"
        assert chats[3] in built.refs, "HEAD was windowed out in favour of the previewed branch"

    def _head_buried(self, branches: int):
        """A card whose old chat has `branches` continuations and holds HEAD, plus a fresh chat to focus.

        With two or more the old chat's subtree is a gap; with one it is an inlined child. Both hide HEAD,
        and both have to say so.
        """
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        greeting = forest.create_node(payload("assistant", "hello!"), parent_id=root)
        old_chat = forest.create_node(payload("user", "the long one"), parent_id=greeting)
        head = chain(forest, length=20, parent_id=old_chat)[-1]
        for k in range(branches - 1):
            forest.create_node(payload("user", f"an abandoned reroll {k}"), parent_id=old_chat)
        new_chat = forest.create_node(payload("user", "a fresh start"), parent_id=greeting)
        return forest, old_chat, head, new_chat

    def test_a_gap_that_swallowed_head_says_so(self):
        # When HEAD is genuinely unreachable -- previewing near the top of a long chat buries it deep in
        # somebody's subtree -- the gap it went into wears the HEAD pill. On a dashed box that reads as
        # "this way", which is the whole of what can still be said about it.
        forest, old_chat, head, new_chat = self._head_buried(branches=2)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=head, focus_node_id=new_chat))
        assert head not in built.refs, \
            "HEAD is on screen after all, so this fixture cannot check what happens when it is not"
        gap = next(ref for ref in refs_of_type(built, chatgraph.SubtreeGapRef)
                   if ref.node_id == old_chat)
        pills = texts_on(built, gap.name)
        assert "HEAD" in pills, f"the gap hiding HEAD carries {pills}, so HEAD vanished without comment"

    def test_the_obligation_follows_the_marker_one_level_down(self):
        # Inlining moves the question rather than answering it: the child is drawn, so it is the *child's*
        # gap that now hides HEAD, and that is where the pointer has to be. A pill on the child's own box
        # would say "this one is HEAD", which is a lie about a box that is merely on the way.
        forest, old_chat, head, new_chat = self._head_buried(branches=1)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=head, focus_node_id=new_chat))
        assert head not in built.refs, \
            "HEAD is on screen after all, so this fixture cannot check what happens when it is not"
        inlined = forest.get_children(old_chat)[0]
        assert inlined in built.refs, "the only child was not inlined"
        assert "HEAD" not in texts_on(built, inlined), \
            "the inlined child claims to be HEAD; it is only on the way to it"
        assert "HEAD" in texts_on(built, self._gap_under(built, inlined)), \
            "HEAD vanished under an inlined child without comment"

    def test_an_ordinary_gap_wears_no_head_pill(self):
        # The control for the above: a pill drawn on every gap would satisfy it and mean nothing.
        forest, root, ids = self._long()
        config = chatgraph.LayoutConfig(max_visible_depth=10)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=ids[-1]), config)
        gaps = refs_of_type(built, chatgraph.DepthGapRef)
        assert gaps, "nothing was elided, so no gap is being checked"
        for gap in gaps:
            assert "HEAD" not in texts_on(built, gap.name), \
                "a gap hiding nothing of HEAD's still claims to point at it"


class TestGapIdentity:
    """Every gap is named for what it hides, so two builds can be matched against each other.

    Which is what an animated transition between layouts needs: pair the boxes by name, tween the ones in
    both, and look up where a node that is not drawn would have been. A name that encodes *where the box
    was in the build order* instead pairs the wrong boxes precisely when something moved.
    """

    def test_a_sibling_gap_keeps_its_name_when_another_level_gains_one(self):
        # The case a serial gets wrong, and it needs a fixture that isolates it: a run that is hidden
        # *identically* in both builds, with a new gap appearing at a level emitted before it. A counter
        # counts the gaps ahead of this one, so the untouched run is renumbered by something that happened
        # somewhere else — and anything matching two builds by name then pairs the wrong boxes.
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        opening = forest.create_node(payload("assistant", "hello!"), parent_id=root)
        chats = [forest.create_node(payload("user", f"chat {k}"), parent_id=opening) for k in range(30)]

        def gaps_by_hidden():
            built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=chats[0]))
            return {ref.hidden_node_ids: ref.name
                    for ref in refs_of_type(built, chatgraph.SiblingGapRef)}

        before = gaps_by_hidden()
        assert before, "no sibling gap in the first picture, so there is nothing to rename"

        # A fan at the level *above*, which gaps too and is emitted first. The chats' own window is
        # untouched: they are `opening`'s children, and this adds `opening`'s siblings.
        for k in range(30):
            forest.create_node(payload("assistant", f"another opening {k}"), parent_id=root)
        after = gaps_by_hidden()
        assert len(after) > len(before), "no gap was added ahead of it, so nothing would renumber"

        shared = set(before) & set(after)
        assert shared, "the run this is about is not hidden in both pictures"
        for hidden in shared:
            assert before[hidden] == after[hidden], \
                "the same hidden run got two different names, so the two builds cannot be matched"

    def test_every_gap_says_what_it_stands_for(self):
        # Uniform across the kinds, because whatever looks a node up has to do it the same way for all of
        # them. `SubtreeGapRef` was the odd one out.
        forest, ids = _forest_with_every_gap_kind()
        config = chatgraph.LayoutConfig(max_visible_depth=8)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=ids["head"],
                                                            new_chat_node_id=ids["greeting"]), config)

        kinds = (chatgraph.SiblingGapRef, chatgraph.DepthGapRef, chatgraph.SubtreeGapRef,
                 chatgraph.ToolRoundGapRef)
        for kind in kinds:
            found = refs_of_type(built, kind)
            assert found, f"no {kind.__name__} here, so it is not being checked"
            for ref in found:
                assert ref.hidden_node_ids, f"{kind.__name__} says nothing about what it stands for"

    def test_a_subtree_gap_names_its_whole_subtree_not_just_its_children(self):
        # The lookup has to answer for a node at any depth behind the gap, not only the level below it.
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        taken = forest.create_node(payload("user", "the branch we are on"), parent_id=root)
        not_taken = forest.create_node(payload("user", "the one we are not"), parent_id=root)
        buried = chain(forest, length=4, parent_id=not_taken)
        for k in range(2):  # two children, so this stays a gap rather than being inlined
            forest.create_node(payload("assistant", f"reroll {k}"), parent_id=not_taken)

        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=taken))
        gap = next(r for r in refs_of_type(built, chatgraph.SubtreeGapRef) if r.node_id == not_taken)
        assert gap.child_count == 3, "the fixture changed shape; the counts below assume three children"
        assert buried[-1] in gap.hidden_node_ids, \
            "a node four levels down is behind this gap and the gap does not know it"
        assert not_taken not in gap.hidden_node_ids, "the owner is drawn; it is not behind its own gap"

    def test_a_drawn_node_represents_itself(self):
        forest, ids = _forest_with_every_gap_kind()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=ids["head"],
                                                            new_chat_node_id=ids["greeting"]),
                                chatgraph.LayoutConfig(max_visible_depth=8))
        assert built.representative_of(ids["head"]) == ids["head"]

    @pytest.mark.parametrize("kind", [chatgraph.SiblingGapRef,
                                      chatgraph.DepthGapRef,
                                      chatgraph.SubtreeGapRef,
                                      chatgraph.ToolRoundGapRef])
    def test_a_node_behind_a_gap_is_represented_by_that_gap(self, kind):
        # One picture, asked about a node behind each kind of gap in turn. Uniformity is the whole point:
        # a caller that has lost the box it was standing on asks one question, whatever swallowed it.
        forest, ids = _forest_with_every_gap_kind()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=ids["head"],
                                                            new_chat_node_id=ids["greeting"]),
                                chatgraph.LayoutConfig(max_visible_depth=8))
        gaps = refs_of_type(built, kind)
        assert gaps, f"no {kind.__name__} in this picture, so this parameter checks nothing"
        gap = gaps[0]
        hidden = gap.hidden_node_ids[0]
        assert hidden not in built.refs, \
            "the fixture draws the node this gap claims to hide, so the lookup cannot be tested on it"
        assert built.representative_of(hidden) == gap.name

    def _folding_round(self):
        """A forest whose one tool round is big enough to fold. Returns `(forest, asking, results, reply)`."""
        forest = Forest()
        system = forest.create_node(payload("system", "the card"), parent_id=None)
        asked = forest.create_node(payload("user", "what is the time"), parent_id=system)
        calls = [{"id": f"c{k}", "function": {"name": f"tool{k}", "arguments": "{}"}} for k in range(3)]
        answering = forest.create_node(payload("assistant", "let me look", tool_calls=calls),
                                       parent_id=asked)
        parent = answering
        results = []
        for k in range(3):
            parent = forest.create_node(payload("tool", f"result {k}"), parent_id=parent)
            results.append(parent)
        reply = forest.create_node(payload("assistant", "it is noon"), parent_id=parent)
        return forest, answering, tuple(results), reply

    def test_a_folded_tool_node_is_represented_by_its_round_s_gap(self):
        # A round's results leave the drawn spine, so a lookup that knew only about the other gap kinds
        # would answer "nowhere" for a tool node — and be believed, the answer being indistinguishable
        # from the honest one. (Raised by Juha, 2026-09-03, from the case where a Back step lands the
        # cursor on one.)
        forest, answering, results, reply = self._folding_round()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        assert results[1] not in built.refs, "the round is drawn open, so nothing was folded"
        gap = only_ref_of_type(built, chatgraph.ToolRoundGapRef)
        assert built.representative_of(results[1]) == gap.name

    def test_an_opened_tool_node_represents_itself(self):
        # The negative control: the same fixture with the round opened draws the tool node, so the answer
        # must come from its own box rather than from the one that would otherwise hide it. A
        # `representative_of` that always named a container would pass the test above and fail this.
        forest, answering, results, reply = self._folding_round()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply,
                                                            expanded_tool_turns={answering}))
        assert results[1] in built.refs, "the round did not open, so this is the other case again"
        assert built.representative_of(results[1]) == results[1]

    def test_a_node_the_boxes_do_not_name_is_found_by_walking_up(self):
        # Where the "everything absent is behind some gap" construction stops short: the roots gap stands
        # for other *roots*, so a message written under one of them is named by no box at all. Its own
        # root is drawn, though, so the ancestor walk reaches it — and the nearest drawn ancestor is the
        # closest thing to "where it would be" the picture can offer.
        forest, ids = _forest_with_every_gap_kind()
        other_card = forest.create_node(payload("system", "another card"), parent_id=None)
        stranger = forest.create_node(payload("user", "a chat under it"), parent_id=other_card)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=ids["head"],
                                                            new_chat_node_id=ids["greeting"]),
                                chatgraph.LayoutConfig(max_visible_depth=8))
        assert built.representative_of(stranger) is None, \
            "some box claims it, so the walk is not what is being tested"
        assert built.representative_of(stranger, datastore=forest) == "gap:roots", \
            "walking up did not reach the box standing for the card this chat is under"

    def test_a_node_that_has_left_the_forest_has_no_representative(self):
        # The honest answer where there genuinely is none, and the reason the caller still needs a last
        # resort of its own.
        forest, ids = _forest_with_every_gap_kind()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=ids["head"],
                                                            new_chat_node_id=ids["greeting"]),
                                chatgraph.LayoutConfig(max_visible_depth=8))
        assert built.representative_of("no-such-node", datastore=forest) is None


def _forest_with_every_gap_kind():
    """A forest whose picture gaps in all four ways at once. Returns `(forest, ids)`.

    A wide level of chats (sibling gaps), a long branch below one of them (a depth gap once the window is
    narrowed), a fan under an off-spine chat (a subtree gap), and a three-call tool round at the end of
    the branch (a tool-round gap).
    """
    forest = Forest()
    root = forest.create_node(payload("system", "the card"), parent_id=None)
    greeting = forest.create_node(payload("assistant", "hello!"), parent_id=root)
    chats = [forest.create_node(payload("user", f"chat {k}"), parent_id=greeting) for k in range(30)]
    tip = chain(forest, length=15, parent_id=chats[3])[-1]
    for k in range(2):
        forest.create_node(payload("assistant", f"reroll {k}"), parent_id=chats[2])

    calls = [{"id": f"c{k}", "function": {"name": f"tool{k}", "arguments": "{}"}} for k in range(3)]
    asking = forest.create_node(payload("assistant", "let me look", tool_calls=calls), parent_id=tip)
    parent = asking
    for k in range(3):
        parent = forest.create_node(payload("tool", f"result {k}"), parent_id=parent)
    head = forest.create_node(payload("assistant", "here you are"), parent_id=parent)
    return forest, {"root": root, "greeting": greeting, "chats": chats, "asking": asking, "head": head}


class TestCursorMovement:
    """Where the arrow keys take the cursor, which is a question about the drawn picture.

    Two rules, deliberately different, because the picture answers "what continues from this" and "what is
    beside this" in two different ways: vertically there are edges to walk, and sideways there are none —
    the boxes on one level are siblings, which is a fact about their parent.
    """

    def test_down_follows_the_branch(self, conversation):
        forest, system, greeting, user, reply = conversation
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        assert chatgraph.neighbor_of(built.graph, system, "down") == greeting
        assert chatgraph.neighbor_of(built.graph, greeting, "down") == user

    def test_up_follows_it_back(self, conversation):
        forest, system, greeting, user, reply = conversation
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        assert chatgraph.neighbor_of(built.graph, user, "up") == greeting

    def test_the_ends_of_the_branch_go_nowhere(self, conversation):
        # The clamp. Wrapping around a tree would put the reader at the far end of the conversation for
        # one keypress too many, with nothing having said they had reached an end.
        forest, system, greeting, user, reply = conversation
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        assert chatgraph.neighbor_of(built.graph, system, "up") is None
        assert chatgraph.neighbor_of(built.graph, reply, "down") is None

    def test_sideways_steps_along_the_level(self):
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        greeting = forest.create_node(payload("assistant", "hello!"), parent_id=root)
        chats = [forest.create_node(payload("user", f"chat {k}"), parent_id=greeting) for k in range(5)]
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=chats[2]))

        drawn = [ref.node_id for ref in refs_of_type(built, chatgraph.ChatNodeRef)
                 if ref.node_id in chats]
        drawn.sort(key=lambda node_id: built.graph.get_node_by_name(node_id).x)
        assert len(drawn) >= 3, "this level is too narrow to step along"

        for left, right in zip(drawn, drawn[1:]):
            assert chatgraph.neighbor_of(built.graph, left, "right") == right
            assert chatgraph.neighbor_of(built.graph, right, "left") == left
        assert chatgraph.neighbor_of(built.graph, drawn[-1], "right") is None

    def test_sideways_never_leaves_the_level(self):
        # The negative control for the rule above, and it has to be stated over a whole picture rather
        # than over one pair. A three-box fixture cannot tell the two rules apart: a child sits at its
        # parent's own x, so "the nearest box in this direction" excludes it for being neither left nor
        # right, and a rule with no notion of a level passes anyway. What discriminates is a picture with
        # gaps in it, whose rows are spaced differently because a gap box is narrower than a message.
        forest, ids = _forest_with_every_gap_kind()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=ids["head"],
                                                            new_chat_node_id=ids["greeting"]),
                                chatgraph.LayoutConfig(max_visible_depth=8))

        checked = 0
        for node in built.graph.nodes:
            for direction in ("left", "right"):
                stepped = chatgraph.neighbor_of(built.graph, node.internal_name, direction)
                if stepped is None:
                    continue
                checked += 1
                other = built.graph.get_node_by_name(stepped)
                assert abs(other.y - node.y) <= 0.5 * (node.y2 - node.y1), \
                    f"stepping {direction} from '{node.internal_name}' left the level it was on"
        assert checked > 4, "almost nothing was stepped; this picture is too thin to be a control"

    def test_the_cursor_can_reach_a_gap(self):
        # The whole reason gaps are destinations: a keyboard that stepped over them could not reach what
        # they hide, and on a wide level that is nearly everything.
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        greeting = forest.create_node(payload("assistant", "hello!"), parent_id=root)
        chats = [forest.create_node(payload("user", f"chat {k}"), parent_id=greeting) for k in range(30)]
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=chats[15]))

        gaps = refs_of_type(built, chatgraph.SiblingGapRef)
        assert gaps, "no sibling gap in this picture, so there is nothing to step onto"
        reached = set()
        for direction in ("left", "right"):
            at = chats[15]
            while at is not None:
                at = chatgraph.neighbor_of(built.graph, at, direction)
                if at is not None:
                    reached.add(at)
        assert {gap.name for gap in gaps} <= reached, \
            "stepping along the level from HEAD never lands on a gap box"

    def test_an_unknown_direction_is_an_error(self, conversation):
        forest, system, greeting, user, reply = conversation
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        with pytest.raises(ValueError):
            chatgraph.neighbor_of(built.graph, reply, "sideways")

    def test_a_box_that_is_not_in_the_picture_goes_nowhere(self, conversation):
        # A cursor whose box a rebuild has just destroyed asks this, and the answer has to be an answer
        # rather than a crash — the panel's landing policy is what decides where it goes instead.
        forest, system, greeting, user, reply = conversation
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        assert chatgraph.neighbor_of(built.graph, "no-such-box", "down") is None


class TestTolerance:
    def test_a_node_with_no_payload_still_gets_a_box(self):
        # The builder runs against a live forest, so a node can lose its payload between the lineage walk
        # and the label lookup. That should cost one label, not the frame.
        forest = Forest()
        system = forest.create_node(payload("system", "you are helpful"), parent_id=None)
        broken = forest.create_node({}, parent_id=system)
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=broken))
        assert broken in built.refs
        assert built.refs[broken].role == ""

    def test_an_unknown_head_is_an_error_rather_than_an_empty_picture(self):
        forest = Forest()
        forest.create_node(payload("system", "you are helpful"), parent_id=None)
        with pytest.raises(KeyError):
            chatgraph.build(forest, chatgraph.ViewState(head_node_id="no-such-node"))
