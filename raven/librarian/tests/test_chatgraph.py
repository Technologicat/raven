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
                                                            previewed_node_id=user))
        assert len(self._rings(built, user)) == 1
        assert self._rings(built, reply) == [], "the ring is on every box, so it marks nothing"

    def test_the_ring_sits_outside_the_box_and_is_dotted(self, conversation):
        # Outside, because the box's own outline is already saying something -- solid or dashed, heavy for
        # HEAD -- and a selection has to be legible over every combination of those. Dotted, because the
        # selection is tentative until a second click.
        forest, system, greeting, user, reply = conversation
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply,
                                                            previewed_node_id=user))
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
        built = chatgraph.build(forest, chatgraph.ViewState(head_node_id=reply, previewed_node_id=reply))
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
