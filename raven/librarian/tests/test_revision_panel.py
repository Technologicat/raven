"""Tests for `raven.librarian.revision_panel`.

Headless: an unmapped viewport renders no frames, so where DPG's focus is cannot be asked here, and
`has_keyboard` is left to live testing. What the panel *does* with a key is independent of that, and is
what these pin — which revision a key or a click acts on, the two-press delete, and following the datastore.
"""

import threading

import pytest

dpg =pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed (GUI toolkit absent in CI)")

from raven.common.gui import animation as gui_animation  # noqa: E402 -- after importorskip by design
from raven.common.gui import utils as guiutils  # noqa: E402 -- ditto
from raven.librarian import chattree, chatutil  # noqa: E402 -- ditto
from raven.librarian import revision_panel  # noqa: E402 -- ditto


def _payload(text, datetime_str):
    return {"message": {"role": "user", "content": [chatutil.text_content_part(text)], "tool_calls": []},
            "general_metadata": {"persona": None, "timestamp": 0, "datetime": datetime_str}}


@pytest.fixture(scope="module")
def dpg_context():
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: these tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


@pytest.fixture(scope="module")
def themes_and_fonts(dpg_context):
    yield guiutils.bootup(font_size=20)
    guiutils.teardown()


@pytest.fixture
def forest_and_node():
    """A message with three revisions, R2 on screen."""
    f = chattree.Forest()
    node_id = f.create_node(_payload("first version", "2026-09-28 10:00:00"), parent_id=None)
    f.add_revision(node_id, _payload("second version", "2026-09-28 11:00:00"))
    f.add_revision(node_id, _payload("third version", "2026-09-28 12:00:00"))
    f.set_revision(node_id, 2)
    return f, node_id


@pytest.fixture
def panel(dpg_context, themes_and_fonts, forest_and_node):
    """The panel, open on the message, with the controller's two operations done directly on the forest."""
    f, node_id = forest_and_node
    calls = []
    refusal = {"show": None, "delete": None}

    def show_revision(nid, revision_id):
        calls.append(("show", nid, revision_id))
        if refusal["show"] is None:
            f.set_revision(nid, revision_id)
        return refusal["show"]

    def delete_revision(nid, revision_id):
        calls.append(("delete", nid, revision_id))
        if refusal["delete"] is None:
            f.delete_revision(nid, revision_id)
        return refusal["delete"]

    p = revision_panel.DPGRevisionPanel(f, themes_and_fonts,
                                        show_revision=show_revision, delete_revision=delete_revision)
    p.calls = calls
    p.refusal = refusal
    p.open(node_id)
    yield p
    p.destroy()
    # The delete presses started flashes, and the flash tooltip its updater; nothing here renders the frames
    # that would end them. With their widgets gone, each ends on the first frame it is given.
    for _ in range(3):
        gui_animation.animator.render_frame()


def _listed(p):
    return [revision_id for revision_id, *_ in p._rows]


class TestDescribeRevisions:
    """What the list says about each revision."""

    def test_every_revision_is_described_oldest_first(self, forest_and_node):
        f, node_id = forest_and_node
        descriptions = chatutil.describe_revisions(f, node_id)
        assert [d["revision"] for d in descriptions] == [1, 2, 3]
        assert [d["opening"] for d in descriptions] == ["first version", "second version", "third version"]
        assert [d["active"] for d in descriptions] == [False, True, False]
        assert descriptions[0]["datetime"] == "2026-09-28 10:00:00"

    def test_the_opening_is_the_first_line_cut_to_length(self):
        f = chattree.Forest()
        node_id = f.create_node(_payload("\n\n" + "x" * 100 + "\nsecond line", "2026-09-28 10:00:00"), parent_id=None)
        opening = chatutil.describe_revisions(f, node_id, opening_chars=10)[0]["opening"]
        assert opening == "x" * 10 + "…"


class TestTheList:
    """Which revision a key or a click acts on."""

    def test_opening_lists_every_revision_with_the_cursor_on_the_one_shown(self, panel):
        assert _listed(panel) == [1, 2, 3]
        assert _listed(panel)[panel._cursor.current] == 2

    def test_enter_shows_the_revision_under_the_cursor(self, panel, forest_and_node):
        f, node_id = forest_and_node
        assert panel.handle_key(dpg.mvKey_Down)
        assert panel.handle_key(dpg.mvKey_Return)
        assert panel.calls == [("show", node_id, 3)]
        assert f.get_revision(node_id) == 3

    def test_showing_the_revision_already_shown_closes_the_panel(self, panel, forest_and_node):
        # Enter again on the same row, or a double-click, whose first click showed it.
        f, node_id = forest_and_node
        panel.handle_key(dpg.mvKey_Down)
        panel.handle_key(dpg.mvKey_Return)
        assert panel.is_open, "showing another revision closed the panel, so the next check cannot tell"
        panel.handle_key(dpg.mvKey_Return)
        assert not panel.is_open
        assert panel.calls == [("show", node_id, 3)], "the second Enter asked to show it again"

    def test_a_refused_show_says_why(self, panel, forest_and_node):
        f, node_id = forest_and_node
        panel.refusal["show"] = "Not while a reply is being written."
        panel.handle_key(dpg.mvKey_Home)
        panel.handle_key(dpg.mvKey_Return)
        assert f.get_revision(node_id) == 2
        assert dpg.get_value(panel._status_text) == "Not while a reply is being written."

    def test_a_modified_key_is_left_for_the_app(self, panel):
        # Ctrl+Shift+E closes the panel from inside it, which is the app's chord and must reach the app.
        assert not panel.handle_key(dpg.mvKey_E, ctrl=True, shift=True)
        assert not panel.handle_key(dpg.mvKey_Down, ctrl=True)


class TestDeleting:
    """Delete, pressed twice, deletes the revision under the cursor."""

    def test_one_press_deletes_nothing_and_a_second_deletes(self, panel, forest_and_node):
        f, node_id = forest_and_node
        panel.handle_key(dpg.mvKey_Home)
        panel.handle_key(dpg.mvKey_Delete)
        assert f.get_revisions(node_id) == [1, 2, 3], "one press deleted"
        panel.handle_key(dpg.mvKey_Delete)
        assert f.get_revisions(node_id) == [2, 3]
        assert _listed(panel) == [2, 3]

    def test_the_second_press_must_be_on_the_same_revision(self, panel, forest_and_node):
        f, node_id = forest_and_node
        panel.handle_key(dpg.mvKey_Home)
        panel.handle_key(dpg.mvKey_Delete)
        panel.handle_key(dpg.mvKey_Down)
        panel.handle_key(dpg.mvKey_Delete)
        assert f.get_revisions(node_id) == [1, 2, 3], "presses on two revisions confirmed a delete"

    def test_a_refused_delete_says_why_and_keeps_the_row(self, panel, forest_and_node):
        f, node_id = forest_and_node
        panel.refusal["delete"] = "The only revision. To remove the message, delete it."
        panel.handle_key(dpg.mvKey_Delete)
        panel.handle_key(dpg.mvKey_Delete)
        assert f.get_revisions(node_id) == [1, 2, 3]
        assert dpg.get_value(panel._status_text).startswith("The only revision")


class TestFollowingTheDatastore:
    """The list follows changes made elsewhere — an edit, a Continue, a delete."""

    def test_a_revision_added_elsewhere_appears_at_the_next_poll(self, panel, forest_and_node):
        f, node_id = forest_and_node
        f.add_revision(node_id, _payload("an edit", "2026-09-28 13:00:00"))
        assert _listed(panel) == [1, 2, 3], "the list changed with no poll, so the poll below proves nothing"
        panel.poll()
        assert _listed(panel) == [1, 2, 3, 4]

    def test_a_change_elsewhere_in_the_tree_does_not_rebuild_the_list(self, panel, forest_and_node):
        # A streaming reply moves the counter chunk by chunk, and a rebuild on each took the focus every frame.
        f, node_id = forest_and_node
        builds, generation = panel._build_count, f.generation
        f.create_node(_payload("a reply being written", "2026-09-28 14:00:00"), parent_id=node_id)
        assert f.generation != generation, "the counter did not move, so this cannot tell a skip from a miss"
        panel.poll()
        assert panel._build_count == builds, "a change to another node rebuilt the list"
        assert _listed(panel) == [1, 2, 3]

    def test_the_poll_neither_waits_for_nor_joins_a_rebuild_in_progress(self, panel, forest_and_node):
        # A click rebuilds the rows on the callback thread while the render thread polls. Two rebuilds at once
        # delete each other's new rows mid-build, which took the render loop down; waiting instead would put
        # the render thread behind a lock.
        f, node_id = forest_and_node
        f.add_revision(node_id, _payload("an edit", "2026-09-28 13:00:00"))
        with panel._rows_lock:  # a rebuild in progress, on this thread
            poller = threading.Thread(target=panel.poll)
            poller.start()
            poller.join(timeout=2.0)
            assert not poller.is_alive(), "the poll waited for the rebuild"
            assert _listed(panel) == [1, 2, 3], "the poll rebuilt alongside the rebuild in progress"
        panel.poll()  # the control: with the rebuild done, the next poll catches up
        assert _listed(panel) == [1, 2, 3, 4]

    def test_the_panel_closes_when_its_message_is_deleted(self, panel, forest_and_node):
        f, node_id = forest_and_node
        f.delete_subtree(node_id)
        panel.poll()
        assert not panel.is_open
