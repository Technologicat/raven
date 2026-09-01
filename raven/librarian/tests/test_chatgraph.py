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
from raven.librarian.chattree import Forest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def payload(role: str, text: str, tool_calls=None) -> dict:
    """A chat node payload of the shape `chatutil` writes, with only the fields this module reads."""
    message = {"role": role, "content": [{"type": "text", "text": text}]}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"message": message, "general_metadata": {"persona": None}}


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
    def _turn_with_tools(self):
        """user -> assistant(2 calls) -> tool -> tool -> assistant(reply)."""
        forest = Forest()
        system = forest.create_node(payload("system", "you are helpful"), parent_id=None)
        user = forest.create_node(payload("user", "what is the time"), parent_id=system)
        asking = forest.create_node(payload("assistant", "let me look",
                                            tool_calls=[{"id": "1"}, {"id": "2"}]), parent_id=user)
        first = forest.create_node(payload("tool", "12:00"), parent_id=asking)
        second = forest.create_node(payload("tool", "Tuesday"), parent_id=first)
        reply = forest.create_node(payload("assistant", "it is noon on Tuesday"), parent_id=second)
        return forest, asking, (first, second), reply

    def test_tool_results_collapse_onto_the_round_that_asked_for_them(self):
        forest, asking, tool_nodes, reply = self._turn_with_tools()
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply))
        assert all(node_id not in built.refs for node_id in tool_nodes)
        assert built.refs[asking].tool_call_count == 2
        assert asking in built.refs and reply in built.refs, "both rounds keep their own node"

    def test_expanding_a_round_shows_its_results(self):
        forest, asking, tool_nodes, reply = self._turn_with_tools()
        built = chatgraph.build(forest,
                                chatgraph.ViewState(head_node_id=reply,
                                                    expanded_tool_turns={asking}))
        assert all(node_id in built.refs for node_id in tool_nodes)

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

        gap_node = built.graph.get_node_by_name("gap:depth")
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

        gap_node = built.graph.get_node_by_name("gap:depth")
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
        forest = Forest()
        system = forest.create_node(payload("system", "you are helpful"), parent_id=None)
        taken = forest.create_node(payload("user", "the branch we are on"), parent_id=system)
        not_taken = forest.create_node(payload("user", "the one we are not"), parent_id=system)
        forest.create_node(payload("assistant", "a whole conversation below here"), parent_id=not_taken)

        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=taken))
        gaps = refs_of_type(built, chatgraph.SubtreeGapRef)
        assert [gap.node_id for gap in gaps] == [not_taken]
        assert gaps[0].child_count == 1

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
        for chat in chats:  # every off-spine sibling has a continuation, so every one wants a subtree gap
            forest.create_node(payload("assistant", "and so on"), parent_id=chat)
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
