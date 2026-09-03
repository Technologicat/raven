"""Tests for `chatgraph_panel` — what a click on the chat graph means, and when the picture rebuilds.

The panel's whole job is a state machine over a real widget: preview, preview again, commit; move a
window; notice that the tree changed. What is worth pinning is that **preview changes nothing** — the
non-destructive half of the interaction is the half a visitor depends on, and it is the half that would
break silently, since a commit that happens too eagerly looks exactly like one the reader asked for.

A real DPG context is needed (the panel builds widgets) but no rendered frame and no mapped window: nothing
here asks about layout, and a mapped window would steal the keyboard from whoever is running the suite.
"""

import time

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from raven.common.gui import animation as gui_animation  # noqa: E402 -- after importorskip by design
from raven.common.gui import utils as guiutils  # noqa: E402 -- after importorskip by design

from raven.librarian import chatgraph  # noqa: E402 -- after importorskip by design
from raven.librarian import chatgraph_panel  # noqa: E402 -- after importorskip by design
from raven.librarian.chattree import Forest  # noqa: E402 -- after importorskip by design


def payload(role: str, text: str) -> dict:
    """A chat node payload of the shape `chatutil` writes.

    The timestamp is not filler. `chatutil.descend_to_latest` orders siblings by it, and `chatgraph.build`
    uses that to run the drawn branch to its tip; without one it cannot say which child is latest, logs a
    warning, and draws the branch only as far as the focus. A fixture missing it therefore exercises a
    degraded builder while looking like it exercises the real one.
    """
    global _payload_serial
    _payload_serial += 1
    return {"message": {"role": role, "content": [{"type": "text", "text": text}]},
            "general_metadata": {"persona": None, "timestamp": _payload_serial}}


_payload_serial = 0


@pytest.fixture(scope="module")
def dpg_context():
    """One DPG context for the whole module, with an unmapped viewport.

    Module-scoped, which is the house pattern for DPG tests here: a context is not cheap, DPG keeps global
    state, and recreating one per test is what took the `--run-gui` group down for eleven days.
    """
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    themes_and_fonts = guiutils.bootup(font_size=14)
    yield themes_and_fonts
    dpg.destroy_context()


class Calls:
    """Records what the panel asked its caller to do, and does the app's half of it.

    Recording alone is not enough for the commit. The panel does not move HEAD itself — it asks, and the
    app moves it — so a callback that only appends leaves HEAD where it was and the panel goes on drawing
    the branch it was already on. Every test of what happens *after* a commit would then be testing a
    commit that did not happen.
    """
    def __init__(self, app_state=None):
        self.previewed = []
        self.committed = []
        self._app_state = app_state

    def commit(self, node_id):
        """Record a commit, and move HEAD as `app.switch_to_chat_node` does."""
        self.committed.append(node_id)
        if self._app_state is not None:
            self._app_state["HEAD"] = node_id


@pytest.fixture
def panel(dpg_context):
    """A panel over a small forked forest, plus the handles a test needs to drive it.

        system -> user -> taken -> taken_tip        <- HEAD is here
                       -> not_taken -> not_taken_tip
    """
    themes_and_fonts = dpg_context
    forest = Forest()
    ids = {}
    ids["system"] = forest.create_node(payload("system", "you are helpful"), parent_id=None)
    ids["user"] = forest.create_node(payload("user", "which way"), parent_id=ids["system"])
    ids["taken"] = forest.create_node(payload("assistant", "this way"), parent_id=ids["user"])
    ids["taken_tip"] = forest.create_node(payload("user", "onwards"), parent_id=ids["taken"])
    ids["not_taken"] = forest.create_node(payload("assistant", "or that way"), parent_id=ids["user"])
    ids["not_taken_tip"] = forest.create_node(payload("user", "elsewhere"), parent_id=ids["not_taken"])

    app_state = {"HEAD": ids["taken_tip"], "new_chat_HEAD": ids["user"]}
    calls = Calls(app_state)

    with dpg.window() as holder:
        built = chatgraph_panel.DPGChatGraphPanel(
            gui_parent=holder,
            datastore=forest,
            app_state=app_state,
            themes_and_fonts=themes_and_fonts,
            width=400, height=300,
            on_preview=calls.previewed.append,
            on_commit=calls.commit,
            show=True)
    built.refresh()

    yield built, forest, app_state, ids, calls

    built.destroy()
    dpg.delete_item(holder)


def click(panel_obj, node_name: str) -> None:
    """Deliver a left click on the named graph box, the way the widget would.

    The widget hands over the `Node` it hit, not its name, so the lookup here is part of what is being
    tested: a panel reading the wrong field off it finds no ref and does nothing.
    """
    node = panel_obj._chat_graph.graph.get_node_by_name(node_name)
    assert node is not None, f"no box named '{node_name}' in the current picture"
    panel_obj._on_click(node, dpg.mvMouseButton_Left)


# ---------------------------------------------------------------------------
# Preview changes nothing
# ---------------------------------------------------------------------------

class TestPreview:
    def test_clicking_a_node_on_the_branch_scrolls_the_chat_log(self, panel):
        built, forest, app_state, ids, calls = panel
        click(built, ids["taken"])
        assert calls.previewed == [ids["taken"]]
        assert calls.committed == []

    def test_clicking_a_node_off_the_branch_moves_the_picture_instead(self, panel):
        # There is no message in the chat log for a node on another branch, so there is nothing to scroll
        # to; the graph redraws around it instead, bringing its siblings and children into view.
        built, forest, app_state, ids, calls = panel
        click(built, ids["not_taken"])
        assert calls.previewed == [], "a node off the branch has no message in the log to scroll to"
        assert calls.committed == []
        # To the *tip* of that branch, not to the node clicked: the focus picks a branch and the branch is
        # drawn whole. Stopping at the click would make everything below it a gap hanging off the box just
        # pressed.
        assert built._chat_graph.spine == tuple(forest.linearize_up(ids["not_taken_tip"]))

    def test_no_preview_moves_head(self, panel):
        # The property a visitor depends on, stated once for every kind of box there is: browsing leaves
        # the session where it was found.
        built, forest, app_state, ids, calls = panel
        head_before = app_state["HEAD"]
        for name in (ids["taken"], ids["not_taken"], ids["system"]):
            click(built, name)
        assert calls.committed == []
        assert app_state["HEAD"] == head_before

    def test_the_previewed_box_is_marked(self, panel):
        # What a second click will act on has to be visible before it happens, or commit-on-second-click is
        # a trap rather than a gesture. The mark goes into the picture; that it is *drawn* is asserted in
        # the builder's own tests, and what this one pins is that the panel asks for it.
        built, forest, app_state, ids, calls = panel
        click(built, ids["taken"])
        assert built._view_state.cursor_name == ids["taken"]

    def test_previewing_does_not_touch_the_hover_highlight(self, panel):
        # That channel is shared with hover and has one pair of colours, so a preview drawn through it is
        # indistinguishable from a hover -- and a node left lit reads as *the important one*, which is
        # HEAD's job.
        built, forest, app_state, ids, calls = panel
        click(built, ids["taken"])
        assert built._widget.get_highlighted_nodes() == set()


# ---------------------------------------------------------------------------
# Commit is the second act
# ---------------------------------------------------------------------------

class TestCommit:
    def test_a_second_click_on_the_same_box_commits(self, panel):
        built, forest, app_state, ids, calls = panel
        click(built, ids["taken"])
        click(built, ids["taken"])
        assert calls.committed == [ids["taken"]]

    def test_a_second_click_on_a_different_box_does_not(self, panel):
        # The control that keeps the test above honest: "clicked twice" must mean the same box twice, not
        # any two clicks. Without this, a panel that committed on every second click would pass it.
        built, forest, app_state, ids, calls = panel
        click(built, ids["taken"])
        click(built, ids["system"])
        assert calls.committed == []
        assert calls.previewed == [ids["taken"], ids["system"]]

    def test_the_toolbar_button_commits_the_previewed_node(self, panel):
        built, forest, app_state, ids, calls = panel
        click(built, ids["not_taken"])
        built._commit_cursor()
        assert calls.committed == [ids["not_taken"]]

    def test_the_toolbar_button_does_nothing_with_nothing_previewed(self, panel):
        built, forest, app_state, ids, calls = panel
        built._commit_cursor()
        assert calls.committed == []

    def test_committing_clears_the_preview(self, panel):
        # Otherwise the next click on that same box would read as a second click and commit again, which is
        # a move the reader did not ask for.
        built, forest, app_state, ids, calls = panel
        click(built, ids["taken"])
        click(built, ids["taken"])
        click(built, ids["taken"])
        assert calls.committed == [ids["taken"]], "the third click committed again"

    def test_going_home_abandons_the_preview(self, panel):
        built, forest, app_state, ids, calls = panel
        click(built, ids["not_taken"])
        assert built._view_state.cursor_name, "nothing was previewed, so clearing it proves nothing"

        built.go_to_head()
        assert built._chat_graph.spine == tuple(forest.linearize_up(app_state["HEAD"]))
        assert built._cursor_name is None, "the preview survived, so a click would commit it"
        assert built._view_state.cursor_name is None, "the ring is still drawn"

    def test_going_home_flashes_where_it_landed(self, panel):
        # The view slides and the zoom changes together, and HEAD is parked off-centre on purpose, so
        # nothing about the motion says which box was the destination.
        built, forest, app_state, ids, calls = panel
        built.go_to_head()
        highlight = built._widget._highlight
        flashed = {element.internal_name for element in highlight._fading}
        assert flashed == {app_state["HEAD"]}

    def test_the_flash_fades_rather_than_switching_off(self, panel):
        # Through the same fade-out a hover uses. Lighting a box and then cutting it reads as a glitch
        # beside a hover that fades, which is what the first version of this did.
        built, forest, app_state, ids, calls = panel
        built.go_to_head()
        highlight = built._widget._highlight
        assert highlight.is_animating(), "the flash is not fading, so it can only be switching off"

        # Wind its start time past the fade rather than sleeping through it.
        element = next(iter(highlight._fading))
        highlight._fading[element] = (time.monotonic_ns() - int(2e9 * highlight.fade_duration), 1.0)
        highlight.update()
        assert not highlight.is_animating()
        assert built._widget.get_highlighted_nodes() == set(), "the fade left a permanent mark behind"


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------

class TestFraming:
    def test_the_first_build_opens_at_full_size(self, panel):
        built, forest, app_state, ids, calls = panel  # the fixture already refreshed once
        assert built._framed
        assert built._widget._viewport.zoom.target == pytest.approx(1.0), \
            "the view opens at a computed zoom, which is a size no button can return the reader to"

    def test_the_view_holds_no_empty_space_below_the_graph(self, panel):
        # HEAD is a leaf in this fixture, which is the ordinary case — a conversation is at its own end.
        # Putting it two-thirds down then asks for a view whose lower third is past the bottom of the
        # tree, and without the clamp that third is spent on nothing.
        built, forest, app_state, ids, calls = panel
        built.go_to_head()
        viewport = built._widget._viewport
        graph_bottom = built._chat_graph.graph.height
        assert graph_bottom > viewport.height, \
            "the graph is shorter than the panel, so it is centred and there is nothing to clamp"
        bottom_of_view = viewport.pan_y.target + 0.5 * viewport.height / viewport.zoom.target
        assert viewport.zoom.target == pytest.approx(1.0)
        assert bottom_of_view <= graph_bottom + 1e-6, \
            f"the view reaches {bottom_of_view:.1f} where the graph ends at {graph_bottom:.1f}"

    def test_opening_and_the_crosshair_frame_alike(self, panel):
        # One framing, not two. A view the reader arrives at on startup and cannot get back to is a view
        # they lose the moment they touch anything -- and only the crosshair has a button.
        built, forest, app_state, ids, calls = panel
        viewport = built._widget._viewport
        opened = (viewport.zoom.target, viewport.pan_x.target, viewport.pan_y.target)

        built._widget.set_zoom(0.3, animate=False)
        built._widget.pan_to_point(9999.0, 9999.0, animate=False)
        assert (viewport.zoom.target, viewport.pan_x.target) != opened[:2], \
            "the view did not move, so this fixture cannot tell a re-framing from doing nothing"

        built.go_to_head()
        assert (viewport.zoom.target, viewport.pan_x.target,
                viewport.pan_y.target) == pytest.approx(opened), \
            "the crosshair puts the reader somewhere other than where the panel opened"

    def test_later_rebuilds_do_not_reframe(self, panel):
        # The picture must stay still while a reply arrives. The tree gains a node per round, so a re-fit
        # per rebuild is a lurch per turn of the conversation.
        built, forest, app_state, ids, calls = panel
        before = built._widget.get_zoom()
        forest.create_node(payload("assistant", "a fresh reroll"), parent_id=ids["taken"])
        built.refresh()
        assert built._widget.get_zoom() == before

    def test_going_home_reframes(self, panel):
        # The one deliberate exception: asking to be returned to HEAD is asking for a framing.
        built, forest, app_state, ids, calls = panel
        built._widget.set_zoom(0.3, animate=False)
        built.go_to_head()
        assert built._widget._viewport.zoom.target == pytest.approx(1.0), \
            "the crosshair should return the reader to full size"


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------

class TestGaps:
    def _wide(self, dpg_context, width=30, on_commit=None):
        themes_and_fonts = dpg_context
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        greeting = forest.create_node(payload("assistant", "hello!"), parent_id=root)
        chats = [forest.create_node(payload("user", f"chat {k}"), parent_id=greeting)
                 for k in range(width)]
        app_state = {"HEAD": chats[0], "new_chat_HEAD": greeting}
        with dpg.window() as holder:
            built = chatgraph_panel.DPGChatGraphPanel(
                gui_parent=holder, datastore=forest, app_state=app_state,
                themes_and_fonts=themes_and_fonts, width=400, height=300, show=True,
                on_commit=on_commit)
        built.refresh()
        return built, forest, chats, holder

    def test_ctrl_arrows_step_the_cursor_along_the_siblings(self, dpg_context):
        built, forest, chats, holder = self._wide(dpg_context)
        try:
            built.handle_key(dpg.mvKey_Down)  # plant the cursor on HEAD, which is chats[0]
            assert built._cursor_name == chats[0]
            assert built.handle_key(dpg.mvKey_Right, ctrl=True) is True
            assert built._cursor_name == chats[1]
            assert built.handle_key(dpg.mvKey_Left, ctrl=True) is True
            assert built._cursor_name == chats[0]
        finally:
            built.destroy()
            dpg.delete_item(holder)

    def test_a_sibling_step_reaches_what_is_not_drawn(self, dpg_context):
        # The whole point of these keys, and what the bare arrows cannot do: a level's window shows a
        # handful of a fan that runs to dozens, so going further means sliding the window. Ten at a time,
        # which is the chat log's stride for the same verb.
        built, forest, chats, holder = self._wide(dpg_context)
        try:
            built.handle_key(dpg.mvKey_Down)
            assert chats[10] not in built._chat_graph.refs, \
                "the fixture already draws the target, so sliding the window would prove nothing"
            assert built.handle_key(dpg.mvKey_Right, ctrl=True, shift=True) is True
            assert built._cursor_name == chats[10]
            assert chats[10] in built._chat_graph.refs, "the window did not slide onto it"
        finally:
            built.destroy()
            dpg.delete_item(holder)

    def test_a_sibling_step_stops_at_the_ends(self, dpg_context):
        # Clamped rather than wrapped: asking to move ten near an end means landing on the end, and
        # wrapping would put the reader at the far side of a fan with nothing having said they arrived.
        built, forest, chats, holder = self._wide(dpg_context)
        try:
            built.handle_key(dpg.mvKey_Down)
            built.handle_key(dpg.mvKey_End, ctrl=True)
            assert built._cursor_name == chats[-1], "Ctrl+End did not reach the last sibling"
            built.handle_key(dpg.mvKey_Right, ctrl=True, shift=True)
            assert built._cursor_name == chats[-1], "stepping past the last sibling went somewhere"
            built.handle_key(dpg.mvKey_Home, ctrl=True)
            assert built._cursor_name == chats[0], "Ctrl+Home did not reach the first sibling"
        finally:
            built.destroy()
            dpg.delete_item(holder)

    def test_a_sibling_step_does_not_move_head(self, dpg_context):
        # The one way these differ from the chat log's identical keys, and the reason they are claimed
        # here at all: in the log they switch branch, and while the graph holds the keyboard they used to
        # fall through to it. Browsing the multiverse changes nothing.
        #
        # The claim is what has to be asserted, not the absence of a commit. Calling `handle_key` runs
        # nothing downstream of it either way, so "HEAD did not move" is true whether the key was claimed
        # or dropped — it is the `is True` that says the app never sees it.
        committed = []
        built, forest, chats, holder = self._wide(dpg_context, on_commit=committed.append)
        head_before = built.app_state["HEAD"]
        try:
            built.handle_key(dpg.mvKey_Down)
            for _ in range(3):
                assert built.handle_key(dpg.mvKey_Right, ctrl=True) is True, \
                    "the graph let a Ctrl+arrow through to the chat log, which would move HEAD"
            assert built._cursor_name == chats[3], "the fixture did not step, so nothing was exercised"
            assert committed == []
            assert built.app_state["HEAD"] == head_before
        finally:
            built.destroy()
            dpg.delete_item(holder)

    def test_a_sibling_gap_moves_the_window_onto_what_it_hid(self, dpg_context):
        built, forest, chats, holder = self._wide(dpg_context)
        try:
            gaps = [ref for ref in built._chat_graph.refs.values()
                    if isinstance(ref, chatgraph.SiblingGapRef)]
            assert gaps, "no sibling gap in this fixture, so there is nothing to click"
            target = gaps[0].recenter_on
            assert target not in built._chat_graph.refs, \
                "the gap's target is already on screen, so moving the window would prove nothing"

            click(built, gaps[0].name)
            assert target in built._chat_graph.refs, "the window did not move onto the hidden sibling"
        finally:
            built.destroy()
            dpg.delete_item(holder)

    def test_moving_the_window_lands_the_cursor_on_what_it_moved_to(self, dpg_context):
        # The cursor has to come out of this gesture somewhere, because it is the one that carries a
        # reader across a level too wide to walk: a keyboard left with nothing to step from would be back
        # where it started. It lands on the node the window recentred on, which is also where the view
        # goes, so the picture and the keyboard agree about where the reader now is.
        built, forest, chats, holder = self._wide(dpg_context)
        try:
            gaps = [ref for ref in built._chat_graph.refs.values()
                    if isinstance(ref, chatgraph.SiblingGapRef)]
            gap = gaps[0]
            assert gap.recenter_on not in built._chat_graph.refs, \
                "the gap's target is already drawn, so landing on it would prove nothing"
            click(built, gap.name)
            assert built._cursor_name == gap.recenter_on, "the cursor did not follow the window"
            assert built._widget.get_highlighted_nodes() == set()
        finally:
            built.destroy()
            dpg.delete_item(holder)

    def test_the_box_the_window_landed_on_acts_like_any_other(self, dpg_context):
        # There is one cursor state, not two: the ring means "an act here acts on this box", wherever the
        # cursor came from. So the box a window move landed on commits on the next act, exactly as a box
        # stepped onto does — the reader has no way to see a difference, so there must not be one.
        committed = []
        built, forest, chats, holder = self._wide(dpg_context, on_commit=committed.append)
        try:
            gaps = [ref for ref in built._chat_graph.refs.values()
                    if isinstance(ref, chatgraph.SiblingGapRef)]
            landed = gaps[0].recenter_on
            click(built, gaps[0].name)
            assert committed == [], "the window move itself switched branch"

            click(built, landed)
            assert committed == [landed], "one act on the landed box did not commit"
        finally:
            built.destroy()
            dpg.delete_item(holder)

    def test_a_depth_gap_shows_more_of_the_branch(self, dpg_context):
        themes_and_fonts = dpg_context
        forest = Forest()
        node = None
        ids = []
        for k in range(40):
            node = forest.create_node(payload("user" if k % 2 == 0 else "assistant", f"m{k}"),
                                      parent_id=node)
            ids.append(node)
        app_state = {"HEAD": ids[-1], "new_chat_HEAD": None}
        with dpg.window() as holder:
            built = chatgraph_panel.DPGChatGraphPanel(
                gui_parent=holder, datastore=forest, app_state=app_state,
                themes_and_fonts=themes_and_fonts, width=400, height=300, show=True)
        try:
            built.refresh()
            before = sum(1 for ref in built._chat_graph.refs.values()
                         if isinstance(ref, chatgraph.ChatNodeRef))
            gaps = [ref for ref in built._chat_graph.refs.values()
                    if isinstance(ref, chatgraph.DepthGapRef)]
            assert gaps, "this branch is not long enough to be truncated"

            click(built, gaps[0].name)
            after = sum(1 for ref in built._chat_graph.refs.values()
                        if isinstance(ref, chatgraph.ChatNodeRef))
            assert after > before, f"clicking the depth gap showed no more of the branch ({before} -> {after})"
        finally:
            built.destroy()
            dpg.delete_item(holder)


# ---------------------------------------------------------------------------
# Tool rounds
# ---------------------------------------------------------------------------

@pytest.fixture
def rounds(dpg_context):
    """A panel over a chat with one foldable tool round and one too small to fold.

        system -> user -> asking(3 calls) -> 3 results -> small(1 call) -> 1 result -> reply   <- HEAD
    """
    themes_and_fonts = dpg_context
    forest = Forest()
    ids = {}
    ids["system"] = forest.create_node(payload("system", "the card"), parent_id=None)
    ids["user"] = forest.create_node(payload("user", "what is the time"), parent_id=ids["system"])
    ids["asking"] = forest.create_node(payload("assistant", "let me look"), parent_id=ids["user"])
    parent = ids["asking"]
    ids["results"] = []
    for k in range(3):
        parent = forest.create_node(payload("tool", f"result {k}"), parent_id=parent)
        ids["results"].append(parent)
    ids["small"] = forest.create_node(payload("assistant", "one more thing"), parent_id=parent)
    ids["small_result"] = forest.create_node(payload("tool", "and its answer"), parent_id=ids["small"])
    ids["reply"] = forest.create_node(payload("assistant", "it is noon"), parent_id=ids["small_result"])

    app_state = {"HEAD": ids["reply"], "new_chat_HEAD": ids["user"]}
    calls = Calls(app_state)
    with dpg.window() as holder:
        built = chatgraph_panel.DPGChatGraphPanel(
            gui_parent=holder, datastore=forest, app_state=app_state,
            themes_and_fonts=themes_and_fonts, width=400, height=300, show=True,
            on_preview=calls.previewed.append, on_commit=calls.commit)
    built.refresh()

    yield built, forest, app_state, ids, calls

    built.destroy()
    dpg.delete_item(holder)


def the_round_gap(panel_obj) -> str:
    """Return the name of the one tool-round gap box in the current picture."""
    gaps = [ref for ref in panel_obj._chat_graph.refs.values()
            if isinstance(ref, chatgraph.ToolRoundGapRef)]
    assert len(gaps) == 1, f"expected one tool-round gap in the picture, got {len(gaps)}"
    return gaps[0].name


class TestToolRounds:
    """Opening a folded round and folding it up again — the pair of gestures that make a round's results
    reachable at all.

    Reachable is the word that matters. Folded, a result has no box, so no cursor lands on it and no
    branch switch can reach it, which the chat log allows and this view did not.
    """

    def test_the_results_are_behind_a_gap_to_begin_with(self, rounds):
        built, forest, app_state, ids, calls = rounds
        assert all(node_id not in built._chat_graph.refs for node_id in ids["results"]), \
            "the round is drawn open, so every test below starts from the wrong state"
        assert ids["small_result"] in built._chat_graph.refs, \
            "the one-call round folded, so the threshold is not being applied"

    def test_opening_the_gap_draws_the_results(self, rounds):
        built, forest, app_state, ids, calls = rounds
        click(built, the_round_gap(built))
        assert all(node_id in built._chat_graph.refs for node_id in ids["results"])

    def test_opening_the_gap_lands_the_cursor_inside_the_round(self, rounds):
        # The gap the cursor was on is exactly what opening it destroys, so the landing policy is what
        # keeps the keyboard usable across the gesture: it follows the first node the gap stood for.
        built, forest, app_state, ids, calls = rounds
        gap_name = the_round_gap(built)
        built._set_cursor(gap_name)
        built.handle_key(dpg.mvKey_Return)
        assert built._cursor_name == ids["results"][0]

    def test_an_opened_result_can_become_head(self, rounds):
        # The capability the whole item exists to restore. Two acts on the box, as anywhere else: the
        # first previews it, the second commits.
        built, forest, app_state, ids, calls = rounds
        click(built, the_round_gap(built))
        click(built, ids["results"][1])
        click(built, ids["results"][1])
        assert calls.committed == [ids["results"][1]]

    def test_opening_a_round_does_not_move_head(self, rounds):
        # Browsing changes nothing, the same as every other gap: opening one is a question about the
        # picture, not about where the reader is.
        built, forest, app_state, ids, calls = rounds
        head_before = app_state["HEAD"]
        click(built, the_round_gap(built))
        assert calls.committed == []
        assert app_state["HEAD"] == head_before

    def test_backspace_folds_the_round_again(self, rounds):
        built, forest, app_state, ids, calls = rounds
        click(built, the_round_gap(built))
        built._set_cursor(ids["results"][1])
        assert built.handle_key(dpg.mvKey_Back) is True
        assert all(node_id not in built._chat_graph.refs for node_id in ids["results"]), \
            "the results are still drawn, so nothing folded"
        assert built._cursor_name == the_round_gap(built), \
            "the cursor did not follow what it was standing on to the box that now stands for it"

    def test_backspace_works_from_the_message_that_asked(self, rounds):
        # The other half of "anywhere inside the round". A reader who opened it from the call and read no
        # further is standing on the call, and that is where they will press the key.
        built, forest, app_state, ids, calls = rounds
        click(built, the_round_gap(built))
        built._set_cursor(ids["asking"])
        built.handle_key(dpg.mvKey_Back)
        assert all(node_id not in built._chat_graph.refs for node_id in ids["results"])

    def test_backspace_outside_a_round_folds_nothing(self, rounds):
        # The negative control, and it needs an *opened* round in the picture: with everything already
        # folded, "nothing folded" is true however wrong the key is.
        built, forest, app_state, ids, calls = rounds
        click(built, the_round_gap(built))
        built._set_cursor(ids["user"])
        built.handle_key(dpg.mvKey_Back)
        assert all(node_id in built._chat_graph.refs for node_id in ids["results"]), \
            "a round the cursor was nowhere near folded up"

    def test_backspace_on_a_round_too_small_to_fold_folds_nothing(self, rounds):
        # The near miss the control above cannot catch: the cursor is on a tool result, which is what
        # `Backspace` acts on — but one belonging to a round nobody opened, because it was never folded.
        # A check that asked "is this a tool node?" rather than "which opened round is it in?" would fold
        # the other round from here.
        built, forest, app_state, ids, calls = rounds
        click(built, the_round_gap(built))
        built._set_cursor(ids["small_result"])
        built.handle_key(dpg.mvKey_Back)
        assert all(node_id in built._chat_graph.refs for node_id in ids["results"]), \
            "a round the cursor was not inside folded up"
        assert built._cursor_name == ids["small_result"], "the cursor moved off a box that is still there"


# ---------------------------------------------------------------------------
# The keyboard
# ---------------------------------------------------------------------------

class TestKeyboard:
    def test_a_hidden_panel_gives_up_the_keyboard(self, panel):
        # Its border is not on screen to say it has the keys, so a reader pressing them would be aiming
        # at something they cannot see.
        built, forest, app_state, ids, calls = panel
        built.has_keyboard = True
        assert built.has_keyboard, "the panel is shown, so it should have been able to take the keyboard"
        built.hide()
        assert not built.has_keyboard

    def test_a_hidden_panel_cannot_be_given_the_keyboard(self, panel):
        # The other half, and the one that keeps the state coherent rather than merely tidy: answering
        # True while showing no border would be claiming keys the reader cannot see it holding.
        built, forest, app_state, ids, calls = panel
        built.hide()
        built.has_keyboard = True
        assert not built.has_keyboard

    def test_keys_it_does_not_claim_are_passed_on(self, panel):
        # The whole reason it can use bare letters: F1 and the rest still reach the app from inside it.
        built, forest, app_state, ids, calls = panel
        assert built.handle_key(dpg.mvKey_F1) is False
        assert built.handle_key(dpg.mvKey_Z) is False

    def test_a_modified_key_is_left_to_the_app(self, panel):
        # Ctrl+N starts a new chat wherever the keyboard is; claiming bare letters must not claim chords
        # built on them.
        built, forest, app_state, ids, calls = panel
        assert built.handle_key(dpg.mvKey_F, ctrl=True) is False
        assert built.handle_key(dpg.mvKey_Home, shift=True) is False

    def test_the_first_arrow_plants_the_cursor_without_moving_it(self, panel):
        # HEAD is already the loudest box in the picture, so the ring appearing there says what the ring
        # means on a box the reader can find. Stepping away from a resting place they never saw would
        # show them the mark somewhere it had not visibly been.
        built, forest, app_state, ids, calls = panel
        assert built._cursor_name is None, "something already placed the cursor; this proves nothing"
        assert built.handle_key(dpg.mvKey_Up) is True
        assert built._cursor_name == app_state["HEAD"]

    def test_the_arrows_move_the_cursor(self, panel):
        built, forest, app_state, ids, calls = panel
        built.handle_key(dpg.mvKey_Up)  # plant it
        planted = built._cursor_name
        assert built.handle_key(dpg.mvKey_Up) is True
        assert built._cursor_name != planted, "the second press did not step"
        assert built._cursor_name == forest.get_parent(planted), \
            "up went somewhere other than the message before this one"

    def test_a_deleted_box_lands_the_cursor_on_what_was_above_it(self, panel):
        # The forest is written by others, and a cleanup pass can remove the very node the cursor is
        # resting on. Falling back to HEAD would be a jump to who knows where — possibly off the visible
        # picture — where the box that message hung from is a step away and is still on screen.
        #
        # The forest cannot answer this one: the node is gone, so there is no lineage left to walk. What
        # answers is the *previous* picture, whose edges are the parent links.
        built, forest, app_state, ids, calls = panel
        click(built, ids["not_taken"])
        assert built._cursor_name == ids["not_taken"]
        assert ids["user"] != app_state["HEAD"], \
            "the parent is HEAD in this fixture, so landing there proves nothing"

        forest.delete_subtree(ids["not_taken"])
        built.refresh()
        assert built._cursor_name == ids["user"], \
            "the cursor did not land on the box the deleted message hung from"

    def test_a_deleted_box_falls_back_to_head_when_nothing_was_above(self, panel):
        # The control for the rule above, and for the last resort still being reachable: with the whole
        # branch gone there is no previous box to inherit, and HEAD is then the only answer left.
        built, forest, app_state, ids, calls = panel
        click(built, ids["not_taken"])
        forest.delete_subtree(ids["system"])
        new_root = forest.create_node(payload("system", "a fresh card"), parent_id=None)
        app_state["HEAD"] = new_root
        built.refresh()
        assert built._cursor_name == new_root

    def test_escape_puts_the_cursor_away(self, panel):
        # Nothing is undone by it — the ring commits nothing on its own — so this only stops pointing.
        built, forest, app_state, ids, calls = panel
        click(built, ids["taken"])
        assert built._cursor_name is not None, "nothing was pointed at, so clearing it proves nothing"
        assert built.handle_key(dpg.mvKey_Escape) is True
        assert built._cursor_name is None
        assert calls.committed == [], "Escape moved HEAD"

    def test_shift_arrows_pan(self, panel):
        built, forest, app_state, ids, calls = panel
        before = built._widget._viewport.pan_x.target
        assert built.handle_key(dpg.mvKey_Right, shift=True) is True
        assert built._widget._viewport.pan_x.target != before

    def test_enter_acts_on_the_box_under_the_cursor(self, panel):
        built, forest, app_state, ids, calls = panel
        click(built, ids["not_taken"])
        assert built._cursor_name == ids["not_taken"]
        assert built.handle_key(dpg.mvKey_Return) is True
        assert calls.committed == [ids["not_taken"]]

    def test_enter_with_no_cursor_does_nothing(self, panel):
        # The control for the above: an Enter that committed something regardless of where the cursor is
        # would pass it while moving HEAD on a keypress the reader had not aimed anywhere.
        built, forest, app_state, ids, calls = panel
        assert built._cursor_name is None
        built.handle_key(dpg.mvKey_Return)
        assert calls.committed == []

    def test_alt_arrows_walk_the_history(self, panel):
        built, forest, app_state, ids, calls = panel
        click(built, ids["not_taken"])
        moved_to = built._chat_graph.spine
        assert built.handle_key(dpg.mvKey_Left, alt=True) is True
        assert built._chat_graph.spine != moved_to, "Alt+Left did not go back"
        assert built.handle_key(dpg.mvKey_Right, alt=True) is True
        assert built._chat_graph.spine == moved_to, "Alt+Right did not return"

    def test_a_bare_arrow_is_not_a_history_step(self, panel):
        # The control for the above: an Alt that was ignored would make the two indistinguishable. The
        # spine cannot say so any more — a bare arrow moves the cursor, a cursor off the drawn branch
        # redraws the picture around it, and stepping sideways from one sibling onto the other restores
        # exactly the spine a Back step would have. What still separates them is the cursor: a step
        # leaves it standing on a box, where going back abandons it, arriving somewhere with nobody
        # else's question posed.
        built, forest, app_state, ids, calls = panel
        click(built, ids["not_taken"])
        built.handle_key(dpg.mvKey_Left)
        assert built._cursor_name is not None, "a bare arrow walked the history instead of stepping"

        built.handle_key(dpg.mvKey_Left, alt=True)
        assert built._cursor_name is None, "Alt+Left did not go back, so the two are not being told apart"


# ---------------------------------------------------------------------------
# Where the reader has been
# ---------------------------------------------------------------------------

class TestHistory:
    """Going back to where you were looking, which is not the same as undoing where you are."""

    def test_the_view_opens_with_nowhere_to_go(self, panel):
        built, forest, app_state, ids, calls = panel
        assert not built._history.can_go_back and not built._history.can_go_forward

    def test_a_branch_switch_leaves_a_way_back(self, panel):
        # The reported case: switch to another branch, and the position you were reading from is gone
        # with no convenient way back. Panning to it across a wide level is a search, not a way back.
        built, forest, app_state, ids, calls = panel
        click(built, ids["not_taken"])
        assert built._history.can_go_back
        built.go_back()
        assert built._chat_graph.spine[-1] == ids["taken_tip"], \
            "back did not restore the branch we came from"

    def test_previewing_then_committing_is_one_step_back(self, panel):
        # Found by driving, 2026-09-02: it took three presses of Back to leave a branch switched to with
        # two clicks, and the first two appeared to do nothing.
        #
        # Two faults, one fix. The second click moves HEAD but draws the same branch, so it is not
        # somewhere the reader went — and the entries were identified by the *focus*, where `None` means
        # "follow HEAD" and therefore resolved to the new branch once HEAD had moved. Naming the branch
        # that was drawn fixes both: the commit dedupes, and what is remembered stays put.
        built, forest, app_state, ids, calls = panel
        started_on = built._chat_graph.spine[-1]

        click(built, ids["not_taken"])   # preview: the picture moves
        click(built, ids["not_taken"])   # commit: HEAD moves, the picture does not
        assert calls.committed, "the second click did not commit, so this is not the sequence reported"

        assert built.go_back() is None and built._chat_graph.spine[-1] == started_on, \
            "one Back should be enough to leave a branch that was entered with two clicks"

    def test_committing_alone_records_nothing(self, panel):
        # The control for the above, and the narrower claim: a commit moves HEAD and the picture stays,
        # so there is no new place to remember. Without this, a history that simply never grew would
        # satisfy the test above.
        built, forest, app_state, ids, calls = panel
        click(built, ids["not_taken"])
        depth = len(built._history)
        click(built, ids["not_taken"])
        assert calls.committed
        assert len(built._history) == depth, "committing a branch switch filled the history"

    def test_going_back_does_not_move_head(self, panel):
        # The mistake this design exists to avoid. Moving HEAD is a deliberate act and stays reversible by
        # navigating; an undo that silently put it back would break the one promise the view makes.
        built, forest, app_state, ids, calls = panel
        click(built, ids["not_taken"])
        click(built, ids["not_taken"])   # the second click commits the branch switch
        assert calls.committed, "nothing was committed, so this fixture cannot check that back leaves it"
        head_after_commit = built._view_state.head_node_id
        built.go_back()
        assert built._view_state.head_node_id == head_after_commit, "back un-committed a branch switch"

    def test_forward_returns_to_where_back_was_pressed(self, panel):
        # Compared by the branch drawn rather than by the focus: a restored view names its branch by the
        # tip, so the focus that arrives is not the node that was clicked to get there — and the picture,
        # which is what "where I was" means, is the same either way.
        built, forest, app_state, ids, calls = panel
        click(built, ids["not_taken"])
        was_showing = built._chat_graph.spine
        built.go_back()
        assert built._chat_graph.spine != was_showing, \
            "back did not move the picture, so forward has nothing to undo"
        built.go_forward()
        assert built._chat_graph.spine == was_showing

    def test_a_click_that_moves_nothing_leaves_no_step(self, panel):
        # A first click on a node already on the drawn branch scrolls the chat log and previews it; the
        # picture does not move. That is not a place the reader went, and Back should not have to walk
        # through it. (A *second* click on the same box commits a branch switch, which is a move.)
        built, forest, app_state, ids, calls = panel
        depth = len(built._history)
        click(built, ids["taken"])
        assert built._cursor_name == ids["taken"], \
            "the click did nothing at all, so this fixture cannot tell a no-op commit from a no-op click"
        assert built._view_state.focus_node_id is None, "the view moved, so there is a step to record"
        assert len(built._history) == depth, "a click that moved no view filled the history"

    def _three_branches(self, dpg_context):
        """A card, a greeting, and three chats under it — three views a click apart."""
        themes_and_fonts = dpg_context
        forest = Forest()
        root = forest.create_node(payload("system", "the card"), parent_id=None)
        greeting = forest.create_node(payload("assistant", "hello!"), parent_id=root)
        chats = [forest.create_node(payload("user", f"chat {k}"), parent_id=greeting) for k in range(3)]
        app_state = {"HEAD": chats[0], "new_chat_HEAD": greeting}
        with dpg.window() as holder:
            built = chatgraph_panel.DPGChatGraphPanel(
                gui_parent=holder, datastore=forest, app_state=app_state,
                themes_and_fonts=themes_and_fonts, width=400, height=300, show=True)
        built.refresh()
        return built, forest, chats, holder

    def test_back_steps_over_a_view_whose_focus_was_deleted(self, dpg_context):
        # The forest is written by others: a cleanup pass removes nodes. A remembered view that named one
        # cannot be returned to, and stopping there would make Back appear to do nothing.
        built, forest, chats, holder = self._three_branches(dpg_context)
        try:
            click(built, chats[1])
            click(built, chats[2])
            assert built._history.states[-2][0] == chats[1], \
                "chat 1 is not the step behind, so deleting it does not test the skip"

            forest.delete_subtree(chats[1])

            assert built._history.can_go_back, "nothing to go back to, so the walk below checks nothing"
            built.go_back()
            # Asserted on the cursor rather than on the picture, because the picture cannot tell the two
            # apart: `refresh` falls back to HEAD when it cannot draw around a focus, so a Back that
            # walked *into* the dead entry produces the same view as one that stepped over it. What
            # differs is where the history now thinks the reader is — and a cursor resting on a phantom
            # makes the next Back, and every Forward, wrong.
            assert built._history.current[0] != chats[1], \
                "the history came to rest on a view whose branch no longer exists"
        finally:
            built.destroy()
            dpg.delete_item(holder)

    def test_back_lands_on_the_deleted_view_s_neighbour_when_it_is_alive(self, dpg_context):
        # The control: a back that always skipped, or that always refused, would satisfy the test above.
        built, forest, chats, holder = self._three_branches(dpg_context)
        try:
            click(built, chats[1])
            click(built, chats[2])
            built.go_back()
            assert built._view_state.focus_node_id == chats[1], \
                "chat 1 is alive and one step behind; back should land exactly there"
        finally:
            built.destroy()
            dpg.delete_item(holder)


# ---------------------------------------------------------------------------
# Keeping up with the forest
# ---------------------------------------------------------------------------

class TestStaleness:
    def test_a_new_node_makes_the_picture_stale(self, panel):
        built, forest, app_state, ids, calls = panel
        assert not built._is_stale(), "the fixture starts stale, so this test cannot tell staleness apart"
        forest.create_node(payload("assistant", "a fresh reroll"), parent_id=ids["taken"])
        assert built._is_stale()

    def test_a_moved_head_makes_the_picture_stale(self, panel):
        # The other half, and neither implies the other: a branch switch adds no nodes, so a check that
        # watched only the change counter would sit on the wrong branch indefinitely.
        built, forest, app_state, ids, calls = panel
        generation_before = forest.generation
        app_state["HEAD"] = ids["not_taken_tip"]
        assert forest.generation == generation_before, \
            "moving HEAD touched the datastore, so this fixture cannot isolate the two signals"
        assert built._is_stale()

    def test_a_refresh_settles_it(self, panel):
        built, forest, app_state, ids, calls = panel
        forest.create_node(payload("assistant", "a fresh reroll"), parent_id=ids["taken"])
        built.refresh()
        assert not built._is_stale()

    def test_the_frame_hook_rebuilds_a_stale_picture(self, panel):
        built, forest, app_state, ids, calls = panel
        new_node = forest.create_node(payload("assistant", "a fresh reroll"), parent_id=ids["taken"])
        assert new_node not in built._chat_graph.refs
        built.render_frame(0)
        assert new_node in built._chat_graph.refs

    def test_a_hidden_panel_does_not_rebuild(self, panel):
        # It refreshes on `show` instead. A panel nobody is looking at should not be paying for a rebuild
        # on every turn of a conversation.
        built, forest, app_state, ids, calls = panel
        built.hide()
        forest.create_node(payload("assistant", "a fresh reroll"), parent_id=ids["taken"])
        built.render_frame(0)
        assert built._is_stale(), "the hidden panel rebuilt anyway"
        built.show()
        assert not built._is_stale()

    def test_a_vanished_head_does_not_take_the_panel_with_it(self, panel):
        # A cleanup can delete the node HEAD names while the view is up.
        built, forest, app_state, ids, calls = panel
        app_state["HEAD"] = "no-such-node"
        built.refresh()  # must not raise
        assert built._chat_graph is not None


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_two_panels_can_coexist(self, dpg_context):
        # Per-instance tags, checked because the widget beneath had exactly this defect: fixed tags, and
        # the second instance died on "Alias already exists".
        themes_and_fonts = dpg_context
        forest = Forest()
        root = forest.create_node(payload("system", "hi"), parent_id=None)
        app_state = {"HEAD": root}
        with dpg.window() as holder:
            first = chatgraph_panel.DPGChatGraphPanel(
                gui_parent=holder, datastore=forest, app_state=app_state,
                themes_and_fonts=themes_and_fonts, width=200, height=200)
            second = chatgraph_panel.DPGChatGraphPanel(
                gui_parent=holder, datastore=forest, app_state=app_state,
                themes_and_fonts=themes_and_fonts, width=200, height=200)
        assert first.gui_uuid != second.gui_uuid
        first.destroy()
        second.destroy()
        dpg.delete_item(holder)

    def test_destroy_stops_the_animator_calling_back(self, panel):
        # A registered animation whose widgets are gone calls DPG on a freed item every frame, which is how
        # a teardown race becomes a segfault rather than an exception.
        built, forest, app_state, ids, calls = panel
        assert built in gui_animation.animator._animations, \
            "the panel never registered, so this test cannot tell an unregister from a no-op"
        built.destroy()
        assert built not in gui_animation.animator._animations
        built.destroy = lambda: None  # the fixture tears down too; once is enough
