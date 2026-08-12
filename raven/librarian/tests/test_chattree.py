"""Unit tests for raven.librarian.chattree (Forest and PersistentForest)."""

import json
import logging
import pathlib

import pytest

from raven.librarian import chattree
from raven.librarian.chattree import Forest, PersistentForest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def forest():
    return Forest()


@pytest.fixture
def chain(forest):
    """A -> B -> C linear chain, for testing traversal and structural ops."""
    a = forest.create_node("payload_a", parent_id=None)
    b = forest.create_node("payload_b", parent_id=a)
    c = forest.create_node("payload_c", parent_id=b)
    return forest, a, b, c


@pytest.fixture
def branching(forest):
    """Root with two children (a branch point), for testing siblings/copy/delete.

        root
       /    \\
    left   right
    """
    root = forest.create_node("root", parent_id=None)
    left = forest.create_node("left", parent_id=root)
    right = forest.create_node("right", parent_id=root)
    return forest, root, left, right


# ---------------------------------------------------------------------------
# Node creation and basic structure
# ---------------------------------------------------------------------------

class TestCreateNode:
    def test_root_node(self, forest):
        nid = forest.create_node("hello", parent_id=None)
        assert nid in forest.nodes
        assert forest.get_parent(nid) is None
        assert forest.get_payload(nid) == "hello"

    def test_child_node_links_both_directions(self, forest):
        parent = forest.create_node("parent", parent_id=None)
        child = forest.create_node("child", parent_id=parent)
        assert forest.get_parent(child) == parent
        assert child in forest.get_children(parent)

    def test_multiple_roots(self, forest):
        r1 = forest.create_node("a", parent_id=None)
        r2 = forest.create_node("b", parent_id=None)
        roots = forest.get_all_root_nodes()
        assert set(roots) == {r1, r2}

    def test_child_of_nonexistent_parent_raises(self, forest):
        with pytest.raises(KeyError):
            forest.create_node("orphan", parent_id="no-such-node")

    def test_node_has_timestamp(self, forest):
        nid = forest.create_node("t", parent_id=None)
        with forest.node(nid) as n:
            assert isinstance(n["timestamp"], int)
            assert n["timestamp"] > 0


# ---------------------------------------------------------------------------
# Payload and revisions
# ---------------------------------------------------------------------------

class TestPayload:
    def test_get_initial_payload(self, forest):
        nid = forest.create_node({"msg": "hi"}, parent_id=None)
        assert forest.get_payload(nid) == {"msg": "hi"}

    def test_get_payload_nonexistent_node_raises(self, forest):
        with pytest.raises(KeyError):
            forest.get_payload("bogus")

    def test_payload_returns_reference_not_copy(self, forest):
        nid = forest.create_node({"x": 1}, parent_id=None)
        p = forest.get_payload(nid)
        p["x"] = 999
        assert forest.get_payload(nid)["x"] == 999

    def test_get_payload_specific_revision(self, forest):
        nid = forest.create_node("v1", parent_id=None)
        r2 = forest.add_revision(nid, "v2")
        assert forest.get_payload(nid, revision_id=1) == "v1"
        assert forest.get_payload(nid, revision_id=r2) == "v2"

    def test_get_payload_nonexistent_revision_raises(self, forest):
        nid = forest.create_node("v1", parent_id=None)
        with pytest.raises(KeyError):
            forest.get_payload(nid, revision_id=42)


class TestRevisions:
    def test_initial_revision_is_1(self, forest):
        nid = forest.create_node("data", parent_id=None)
        assert forest.get_revision(nid) == 1
        assert forest.get_revisions(nid) == [1]

    def test_add_revision_increments(self, forest):
        nid = forest.create_node("v1", parent_id=None)
        r2 = forest.add_revision(nid, "v2")
        r3 = forest.add_revision(nid, "v3")
        assert r2 == 2
        assert r3 == 3
        assert forest.get_revisions(nid) == [1, 2, 3]

    def test_add_revision_activates_new(self, forest):
        nid = forest.create_node("v1", parent_id=None)
        r2 = forest.add_revision(nid, "v2")
        assert forest.get_revision(nid) == r2
        assert forest.get_payload(nid) == "v2"

    def test_add_revision_with_name(self, forest):
        nid = forest.create_node("v1", parent_id=None)
        r2 = forest.add_revision(nid, "v2", revision_name="fixed typo")
        assert forest.get_revision_name(nid, r2) == "fixed typo"

    def test_add_revision_nonexistent_node_raises(self, forest):
        with pytest.raises(KeyError):
            forest.add_revision("nope", "data")

    def test_set_revision(self, forest):
        nid = forest.create_node("v1", parent_id=None)
        forest.add_revision(nid, "v2")
        forest.set_revision(nid, 1)
        assert forest.get_revision(nid) == 1
        assert forest.get_payload(nid) == "v1"

    def test_set_revision_nonexistent_revision_raises(self, forest):
        nid = forest.create_node("v1", parent_id=None)
        with pytest.raises(KeyError):
            forest.set_revision(nid, 99)

    def test_set_revision_nonexistent_node_raises(self, forest):
        with pytest.raises(KeyError):
            forest.set_revision("nope", 1)

    def test_delete_revision(self, forest):
        nid = forest.create_node("v1", parent_id=None)
        r2 = forest.add_revision(nid, "v2")
        forest.delete_revision(nid, 1)
        assert forest.get_revisions(nid) == [r2]

    def test_delete_active_revision_selects_next_newer(self, forest):
        nid = forest.create_node("v1", parent_id=None)
        r2 = forest.add_revision(nid, "v2")
        forest.add_revision(nid, "v3")
        # Active is now 3. Set it to 2 so deleting 2 tests the "select next newer" path.
        forest.set_revision(nid, r2)
        forest.delete_revision(nid, r2)
        assert forest.get_revision(nid) == 3

    def test_delete_active_last_revision_selects_previous(self, forest):
        nid = forest.create_node("v1", parent_id=None)
        forest.add_revision(nid, "v2")
        r3 = forest.add_revision(nid, "v3")
        # Active is 3 (the last). Deleting it should select 2 (the most recent remaining).
        forest.delete_revision(nid, r3)
        assert forest.get_revision(nid) == 2

    def test_delete_only_revision_raises(self, forest):
        nid = forest.create_node("only", parent_id=None)
        with pytest.raises(ValueError):
            forest.delete_revision(nid, 1)

    def test_delete_revision_nonexistent_node_raises(self, forest):
        with pytest.raises(KeyError):
            forest.delete_revision("nope", 1)

    def test_delete_revision_nonexistent_revision_raises(self, forest):
        nid = forest.create_node("v1", parent_id=None)
        with pytest.raises(KeyError):
            forest.delete_revision(nid, 42)

    def test_deleted_revision_id_not_reused(self, forest):
        nid = forest.create_node("v1", parent_id=None)
        r2 = forest.add_revision(nid, "v2")
        forest.delete_revision(nid, r2)
        r3 = forest.add_revision(nid, "v3")
        assert r3 == 3  # not 2 again

    def test_revision_name_crud(self, forest):
        nid = forest.create_node("v1", parent_id=None)
        # Initially unnamed.
        assert forest.get_revision_name(nid, 1) is None
        # Set a name.
        forest.set_revision_name(nid, 1, "initial")
        assert forest.get_revision_name(nid, 1) == "initial"
        # Overwrite.
        forest.set_revision_name(nid, 1, "renamed")
        assert forest.get_revision_name(nid, 1) == "renamed"

    def test_set_revision_name_nonexistent_node_raises(self, forest):
        with pytest.raises(KeyError):
            forest.set_revision_name("nope", 1, "name")

    def test_set_revision_name_nonexistent_revision_raises(self, forest):
        nid = forest.create_node("v1", parent_id=None)
        with pytest.raises(KeyError):
            forest.set_revision_name(nid, 99, "name")

    def test_delete_revision_also_deletes_name(self, forest):
        nid = forest.create_node("v1", parent_id=None)
        r2 = forest.add_revision(nid, "v2", revision_name="named")
        assert forest.get_revision_name(nid, r2) == "named"
        forest.delete_revision(nid, r2)
        # The revision is gone, so asking for its name should raise.
        # (The name dict entry was cleaned up too.)
        with forest.node(nid) as n:
            assert str(r2) not in n["revision_names"]

    def test_get_revisions_nonexistent_node_raises(self, forest):
        with pytest.raises(KeyError):
            forest.get_revisions("nope")

    def test_get_revision_nonexistent_node_raises(self, forest):
        with pytest.raises(KeyError):
            forest.get_revision("nope")

    def test_get_revision_name_nonexistent_node_raises(self, forest):
        with pytest.raises(KeyError):
            forest.get_revision_name("nope", 1)


# ---------------------------------------------------------------------------
# Node context manager
# ---------------------------------------------------------------------------

class TestNodeContextManager:
    def test_yields_node_dict(self, forest):
        nid = forest.create_node("cm", parent_id=None)
        with forest.node(nid) as n:
            assert n["id"] == nid
            assert n["data"][str(1)] == "cm"

    def test_nonexistent_raises(self, forest):
        with pytest.raises(KeyError):
            with forest.node("no-such"):
                pass  # pragma: no cover


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

class TestNavigation:
    def test_get_parent(self, chain):
        f, a, b, c = chain
        assert f.get_parent(b) == a
        assert f.get_parent(c) == b

    def test_get_parent_of_root_is_none(self, chain):
        f, a, _b, _c = chain
        assert f.get_parent(a) is None

    def test_get_parent_nonexistent_raises(self, forest):
        with pytest.raises(KeyError):
            forest.get_parent("nope")

    def test_get_children(self, branching):
        f, root, left, right = branching
        assert f.get_children(root) == [left, right]

    def test_get_children_of_leaf_is_empty(self, branching):
        f, _root, left, _right = branching
        assert f.get_children(left) == []

    def test_get_children_nonexistent_raises(self, forest):
        with pytest.raises(KeyError):
            forest.get_children("nope")

    def test_get_siblings(self, branching):
        f, root, left, right = branching
        siblings, idx = f.get_siblings(left)
        assert siblings == [left, right]
        assert idx == 0
        siblings2, idx2 = f.get_siblings(right)
        assert siblings2 == [left, right]
        assert idx2 == 1

    def test_a_lone_root_is_its_own_only_sibling(self, chain):
        f, a, _b, _c = chain
        siblings, idx = f.get_siblings(a)
        assert siblings == [a]
        assert idx == 0

    def test_the_siblings_of_a_root_are_the_other_roots(self, chain):
        # A forest's trees stand beside each other exactly as two replies to one message do. Librarian walks
        # this to reach a chat held under a different system prompt: each distinct card is its own root, so
        # without it those chats are stored and unreachable.
        f, a, _b, _c = chain
        second = f.create_node("another root", parent_id=None)
        third = f.create_node("a third root", parent_id=None)

        siblings, idx = f.get_siblings(a)
        assert siblings == [a, second, third]
        assert idx == 0

        siblings, idx = f.get_siblings(third)
        assert siblings == [a, second, third]
        assert idx == 2

    def test_a_root_and_a_child_do_not_see_each_other_as_siblings(self, chain):
        # The two cases stay separate: roots answer with roots, children with their parent's children.
        f, a, b, _c = chain
        f.create_node("another root", parent_id=None)
        root_siblings, _idx = f.get_siblings(a)
        child_siblings, _idx = f.get_siblings(b)
        assert b not in root_siblings
        assert a not in child_siblings

    def test_get_siblings_nonexistent_raises(self, forest):
        with pytest.raises(KeyError):
            forest.get_siblings("nope")


# ---------------------------------------------------------------------------
# Walking
# ---------------------------------------------------------------------------

class TestWalk:
    def test_walk_up_to_root(self, chain):
        f, a, b, c = chain
        visited = []
        f.walk_up(c, callback=lambda n: visited.append(n["id"]))
        assert visited == [c, b, a]

    def test_walk_up_returns_root_id(self, chain):
        f, a, _b, c = chain
        root_id = f.walk_up(c)
        assert root_id == a

    def test_walk_up_stop_iteration(self, chain):
        f, _a, b, c = chain
        visited = []
        def stop_at_b(n):
            visited.append(n["id"])
            if n["id"] == b:
                raise StopIteration
        stopped_at = f.walk_up(c, callback=stop_at_b)
        assert visited == [c, b]
        assert stopped_at == b

    def test_walk_up_single_node(self, forest):
        nid = forest.create_node("solo", parent_id=None)
        root_id = forest.walk_up(nid)
        assert root_id == nid

    def test_linearize_up(self, chain):
        f, a, b, c = chain
        assert f.linearize_up(c) == [a, b, c]

    def test_linearize_up_from_middle(self, chain):
        f, a, b, _c = chain
        assert f.linearize_up(b) == [a, b]

    def test_linearize_up_from_root(self, chain):
        f, a, _b, _c = chain
        assert f.linearize_up(a) == [a]


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------

class TestCopy:
    def test_copy_node_creates_new_id(self, forest):
        orig = forest.create_node("data", parent_id=None)
        copied = forest.copy_node(orig, new_parent_id=None)
        assert copied != orig
        assert forest.get_payload(copied) == "data"

    def test_copy_node_deep_copies_payload(self, forest):
        orig = forest.create_node({"x": [1, 2, 3]}, parent_id=None)
        copied = forest.copy_node(orig, new_parent_id=None)
        # Mutating the copy should not affect the original.
        forest.get_payload(copied)["x"].append(4)
        assert forest.get_payload(orig)["x"] == [1, 2, 3]

    def test_copy_node_preserves_revisions(self, forest):
        orig = forest.create_node("v1", parent_id=None)
        forest.add_revision(orig, "v2", revision_name="second")
        copied = forest.copy_node(orig, new_parent_id=None)
        assert forest.get_revisions(copied) == [1, 2]
        assert forest.get_revision(copied) == 2
        assert forest.get_payload(copied, revision_id=1) == "v1"
        assert forest.get_payload(copied, revision_id=2) == "v2"
        assert forest.get_revision_name(copied, 2) == "second"

    def test_copy_node_does_not_copy_children(self, branching):
        f, root, _left, _right = branching
        copied = f.copy_node(root, new_parent_id=None)
        assert f.get_children(copied) == []

    def test_copy_node_links_to_new_parent(self, forest):
        parent = forest.create_node("p", parent_id=None)
        orig = forest.create_node("o", parent_id=None)
        copied = forest.copy_node(orig, new_parent_id=parent)
        assert forest.get_parent(copied) == parent
        assert copied in forest.get_children(parent)

    def test_copy_subtree(self, branching):
        f, root, left, right = branching
        # Add a grandchild to test recursion.
        gc = f.create_node("grandchild", parent_id=left)  # noqa: F841 -- side effect: creates the node

        new_root = f.copy_subtree(root, new_parent_id=None)
        assert new_root != root
        new_children = f.get_children(new_root)
        assert len(new_children) == 2
        # The copied children should have the same payloads.
        child_payloads = {f.get_payload(c) for c in new_children}
        assert child_payloads == {"left", "right"}
        # Find the copy of "left" and check its grandchild was copied.
        for c in new_children:
            if f.get_payload(c) == "left":
                gc_copies = f.get_children(c)
                assert len(gc_copies) == 1
                assert f.get_payload(gc_copies[0]) == "grandchild"

    def test_copy_subtree_independence(self, branching):
        f, root, _left, _right = branching
        new_root = f.copy_subtree(root, new_parent_id=None)
        # Deleting the copy should not affect the original.
        f.delete_subtree(new_root)
        assert root in f.nodes
        assert f.get_payload(root) == "root"
        assert len(f.get_children(root)) == 2


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_leaf_node(self, chain):
        f, a, b, c = chain
        f.delete_node(c)
        assert c not in f.nodes
        assert f.get_children(b) == []

    def test_delete_middle_node_children_become_roots(self, chain):
        f, a, b, c = chain
        f.delete_node(b)
        assert b not in f.nodes
        assert c not in f.get_children(a)  # a lost its child
        assert f.get_parent(c) is None  # c is now a root

    def test_delete_subtree(self, branching):
        f, root, left, right = branching
        gc = f.create_node("gc", parent_id=left)
        f.delete_subtree(root)
        for nid in (root, left, right, gc):
            assert nid not in f.nodes

    def test_delete_subtree_leaf(self, chain):
        f, a, b, c = chain
        f.delete_subtree(c)
        assert c not in f.nodes
        assert f.get_children(b) == []

    def test_delete_node_nonexistent_raises(self, forest):
        with pytest.raises(KeyError):
            forest.delete_node("nope")

    def test_delete_subtree_nonexistent_raises(self, forest):
        with pytest.raises(KeyError):
            forest.delete_subtree("nope")


# ---------------------------------------------------------------------------
# Detach and reparent
# ---------------------------------------------------------------------------

class TestDetachReparent:
    def test_detach_subtree_makes_root(self, chain):
        f, a, b, c = chain
        f.detach_subtree(b)
        assert f.get_parent(b) is None
        assert b not in f.get_children(a)
        # b -> c link should still be intact.
        assert f.get_parent(c) == b

    def test_detach_subtree_returns_node_id(self, chain):
        f, _a, b, _c = chain
        assert f.detach_subtree(b) == b

    def test_detach_subtree_root_is_noop(self, chain):
        f, a, b, _c = chain
        f.detach_subtree(a)
        assert f.get_parent(a) is None
        # Children should be unaffected.
        assert b in f.get_children(a)

    def test_detach_children(self, branching):
        f, root, left, right = branching
        f.detach_children(root)
        assert f.get_children(root) == []
        assert f.get_parent(left) is None
        assert f.get_parent(right) is None

    def test_detach_children_returns_node_id(self, branching):
        f, root, _left, _right = branching
        assert f.detach_children(root) == root

    def test_reparent_subtree(self, forest):
        old_parent = forest.create_node("old", parent_id=None)
        child = forest.create_node("child", parent_id=old_parent)
        new_parent = forest.create_node("new", parent_id=None)

        forest.reparent_subtree(child, new_parent)
        assert forest.get_parent(child) == new_parent
        assert child in forest.get_children(new_parent)
        assert child not in forest.get_children(old_parent)

    def test_reparent_subtree_to_none(self, chain):
        f, _a, b, c = chain
        f.reparent_subtree(b, None)
        assert f.get_parent(b) is None
        # b -> c link survives.
        assert f.get_parent(c) == b

    def test_reparent_children(self, branching):
        f, root, left, right = branching
        new_parent = f.create_node("new", parent_id=None)
        f.reparent_children(root, new_parent)
        assert f.get_children(root) == []
        assert set(f.get_children(new_parent)) == {left, right}
        assert f.get_parent(left) == new_parent
        assert f.get_parent(right) == new_parent

    def test_reparent_children_to_none(self, branching):
        f, root, left, right = branching
        f.reparent_children(root, None)
        assert f.get_children(root) == []
        assert f.get_parent(left) is None
        assert f.get_parent(right) is None

    def test_reparent_children_appends_to_existing(self, forest):
        p1 = forest.create_node("p1", parent_id=None)
        c1 = forest.create_node("c1", parent_id=p1)
        p2 = forest.create_node("p2", parent_id=None)
        c2 = forest.create_node("c2", parent_id=p2)
        # Move c1 under p2 (which already has c2).
        forest.reparent_children(p1, p2)
        assert forest.get_children(p2) == [c2, c1]


# ---------------------------------------------------------------------------
# Maintenance utilities
# ---------------------------------------------------------------------------

class TestMaintenance:
    def test_get_all_root_nodes(self, chain):
        f, a, _b, _c = chain
        assert f.get_all_root_nodes() == [a]

    def test_get_all_root_nodes_after_detach(self, chain):
        f, a, b, _c = chain
        f.detach_subtree(b)
        roots = set(f.get_all_root_nodes())
        assert roots == {a, b}

    def test_prune_unreachable_nodes(self, forest):
        r1 = forest.create_node("keep", parent_id=None)
        forest.create_node("keep_child", parent_id=r1)
        orphan = forest.create_node("orphan", parent_id=None)
        forest.prune_unreachable_nodes(r1)
        assert orphan not in forest.nodes
        assert r1 in forest.nodes

    def test_list_unreachable_nodes_is_a_dry_run(self, forest):
        """The dry run names exactly what the prune would delete, and deletes nothing itself."""
        r1 = forest.create_node("keep", parent_id=None)
        kept_child = forest.create_node("keep_child", parent_id=r1)
        orphan = forest.create_node("orphan", parent_id=None)
        orphan_child = forest.create_node("orphan_child", parent_id=orphan)

        doomed = forest.list_unreachable_nodes(r1)
        assert set(doomed) == {orphan, orphan_child}  # the whole unreachable subtree, not just its root
        assert set(forest.nodes) == {r1, kept_child, orphan, orphan_child}  # nothing deleted

        forest.prune_unreachable_nodes(r1)
        assert set(forest.nodes) == {r1, kept_child}  # ...and the prune agrees with what the dry run said

    def test_prune_unreachable_with_multiple_roots(self, forest):
        r1 = forest.create_node("r1", parent_id=None)
        r2 = forest.create_node("r2", parent_id=None)
        orphan = forest.create_node("orphan", parent_id=None)
        forest.prune_unreachable_nodes(r1, r2)
        assert r1 in forest.nodes
        assert r2 in forest.nodes
        assert orphan not in forest.nodes

    def test_naming_one_root_condemns_the_other_trees(self, forest):
        # Characterization, not a wish: a prune keeps what is reachable from the roots it was *given*, so a
        # caller that names one root in a forest of several deletes everything under the rest. Librarian
        # keeps one root per distinct system prompt, which makes every one of them a tree of real chats —
        # so its callers pass `get_all_root_nodes()`, and this is what the shortcut would cost.
        r1 = forest.create_node("r1", parent_id=None)
        r2 = forest.create_node("r2", parent_id=None)
        chat_under_r2 = forest.create_node("a chat held under the other card", parent_id=r2)

        assert set(forest.list_unreachable_nodes(r1)) == {r2, chat_under_r2}
        assert forest.list_unreachable_nodes(*forest.get_all_root_nodes()) == []

    def test_prune_dead_links_parent(self, forest):
        root = forest.create_node("root", parent_id=None)
        child = forest.create_node("child", parent_id=root)
        # Manually corrupt: point child's parent to a nonexistent node.
        forest.nodes[child]["parent"] = "ghost"
        forest.prune_dead_links(root)
        # child should now be a root (dead parent link removed).
        assert forest.get_parent(child) is None

    def test_prune_dead_links_children(self, forest):
        root = forest.create_node("root", parent_id=None)
        child = forest.create_node("child", parent_id=root)
        # Manually corrupt: add a ghost child to root.
        forest.nodes[root]["children"].append("ghost")
        forest.prune_dead_links(root)
        # Ghost should be gone, real child should remain.
        assert forest.get_children(root) == [child]

    def test_purge(self, chain):
        f, _a, _b, _c = chain
        assert len(f.nodes) == 3
        f.purge()
        assert len(f.nodes) == 0

    def test_str_contains_node_ids(self, chain):
        f, a, b, c = chain
        s = str(f)
        assert a in s
        assert b in s
        assert c in s


# ---------------------------------------------------------------------------
# PersistentForest: JSON roundtrip
# ---------------------------------------------------------------------------

class TestPersistentForestRoundtrip:
    def test_save_and_load(self, tmp_path):
        filepath = tmp_path / "forest.json"

        # --- Create and populate ---
        pf1 = PersistentForest(datastore_file=pathlib.Path(filepath))
        root = pf1.create_node({"role": "system", "content": "hello"}, parent_id=None)
        child = pf1.create_node({"role": "user", "content": "world"}, parent_id=root)
        r2 = pf1.add_revision(child, {"role": "user", "content": "world (edited)"}, revision_name="typo fix")
        pf1.save()

        # --- Load into a fresh instance ---
        pf2 = PersistentForest(datastore_file=pathlib.Path(filepath))

        # Structure preserved.
        assert pf2.get_parent(root) is None
        assert pf2.get_parent(child) == root
        assert pf2.get_children(root) == [child]

        # Payloads preserved.
        assert pf2.get_payload(root) == {"role": "system", "content": "hello"}
        assert pf2.get_payload(child) == {"role": "user", "content": "world (edited)"}
        assert pf2.get_payload(child, revision_id=1) == {"role": "user", "content": "world"}

        # Revision metadata preserved.
        assert pf2.get_revision(child) == r2
        assert pf2.get_revisions(child) == [1, r2]
        assert pf2.get_revision_name(child, r2) == "typo fix"

    def test_load_nonexistent_file_creates_empty_forest(self, tmp_path):
        filepath = tmp_path / "does_not_exist.json"
        pf = PersistentForest(datastore_file=pathlib.Path(filepath))
        assert len(pf.nodes) == 0


# ---------------------------------------------------------------------------
# PersistentForest: image sidecar storage + mark-and-sweep GC
# ---------------------------------------------------------------------------

def _refs_from_sidecars_list(payload):
    """Trivial sidecar extractor for tests: payloads declare their refs under a "sidecars" key."""
    return set(payload.get("sidecars", []))


def _write_datastore(path, nodes=None):
    """Write a minimal valid datastore file, so a migration has something real to move."""
    path.write_text(json.dumps(nodes if nodes is not None else {}), encoding="utf-8")
    return path


class TestLegacySidecarDirMigration:
    """`<datastore>.images/` became `<datastore>.sidecars/`; the old name predates documents being storable.

    The rename happens on load, in place. What makes it worth pinning is that getting it wrong is silent:
    payloads reference sidecars by *filename*, so a directory left behind under the old name produces
    attachments that resolve to a name and not to a file — visible only when someone opens an old chat.
    """

    def test_an_old_directory_is_renamed_on_load(self, tmp_path):
        datastore_file = _write_datastore(tmp_path / "chat.json")
        old_dir = tmp_path / ("chat" + chattree.LEGACY_SIDECAR_SUFFIX)
        old_dir.mkdir()
        (old_dir / "abc.png").write_bytes(b"payload")

        pf = PersistentForest(datastore_file, autosave=False)
        assert not old_dir.exists()
        assert pf.sidecar_dir.name == "chat" + chattree.SIDECAR_SUFFIX
        assert pf.read_sidecar("abc.png") == b"payload"  # the file came across, not just the directory

    def test_a_current_directory_is_left_alone(self, tmp_path):
        datastore_file = _write_datastore(tmp_path / "chat.json")
        (tmp_path / ("chat" + chattree.SIDECAR_SUFFIX)).mkdir()
        (tmp_path / ("chat" + chattree.LEGACY_SIDECAR_SUFFIX)).mkdir()

        PersistentForest(datastore_file, autosave=False)
        # Both present means someone has two directories and only one of them is ours; merging them is not
        # this migration's business, so the current one wins and the old one is left for a human to look at.
        assert (tmp_path / ("chat" + chattree.SIDECAR_SUFFIX)).is_dir()
        assert (tmp_path / ("chat" + chattree.LEGACY_SIDECAR_SUFFIX)).is_dir()

    def test_no_directory_at_all_is_not_an_error(self, tmp_path):
        # The overwhelmingly common case: a fresh datastore with nothing attached yet.
        pf = PersistentForest(_write_datastore(tmp_path / "chat.json"), autosave=False)
        assert not pf.sidecar_dir.exists()


class TestRenameDatastore:
    """Renaming a datastore has to take its sidecar directory with it, or the attachments are stranded."""

    def test_the_pair_moves_together(self, tmp_path):
        old_file = _write_datastore(tmp_path / "data.json")
        old_dir = tmp_path / ("data" + chattree.SIDECAR_SUFFIX)
        old_dir.mkdir()
        (old_dir / "abc.png").write_bytes(b"payload")

        assert chattree.rename_datastore(old_file, tmp_path / "chat.json") is True
        assert not old_file.exists() and not old_dir.exists()
        assert (tmp_path / "chat.json").is_file()
        assert (tmp_path / ("chat" + chattree.SIDECAR_SUFFIX) / "abc.png").read_bytes() == b"payload"

    def test_a_legacy_sidecar_directory_moves_too(self, tmp_path):
        # A datastore that predates *both* renames: it must survive being moved and then migrated.
        old_file = _write_datastore(tmp_path / "data.json")
        (tmp_path / ("data" + chattree.LEGACY_SIDECAR_SUFFIX)).mkdir()
        (tmp_path / ("data" + chattree.LEGACY_SIDECAR_SUFFIX) / "abc.png").write_bytes(b"payload")

        chattree.rename_datastore(old_file, tmp_path / "chat.json")
        assert (tmp_path / ("chat" + chattree.LEGACY_SIDECAR_SUFFIX)).is_dir()
        # ...and the suffix migration then completes the job on load.
        pf = PersistentForest(tmp_path / "chat.json", autosave=False)
        assert pf.read_sidecar("abc.png") == b"payload"

    def test_an_existing_target_is_never_overwritten(self, tmp_path):
        old_file = _write_datastore(tmp_path / "data.json", {"old": {}})
        new_file = _write_datastore(tmp_path / "chat.json", {"new": {}})
        assert chattree.rename_datastore(old_file, new_file) is False
        assert json.loads(new_file.read_text(encoding="utf-8")) == {"new": {}}
        assert old_file.is_file()

    def test_a_missing_source_is_not_an_error(self, tmp_path):
        assert chattree.rename_datastore(tmp_path / "nope.json", tmp_path / "chat.json") is False

    def test_renaming_onto_itself_does_nothing(self, tmp_path):
        same = _write_datastore(tmp_path / "chat.json")
        assert chattree.rename_datastore(same, same) is False
        assert same.is_file()

    def test_a_failure_moving_the_file_puts_the_directory_back(self, monkeypatch, tmp_path):
        """The halfway state is the one that silently strands attachments, so it must not survive.

        A sidecar directory is found by *deriving* its name from the datastore's, with nothing recording
        where it actually went — so a datastore left under its old name beside a directory moved to the new
        one looks, to every later reader, like a chat whose attachments were deleted.
        """
        old_file = _write_datastore(tmp_path / "data.json")
        old_dir = tmp_path / ("data" + chattree.SIDECAR_SUFFIX)
        old_dir.mkdir()
        (old_dir / "abc.png").write_bytes(b"payload")

        real_rename = pathlib.Path.rename

        def rename_but_not_the_json(self, target):
            if self.suffix == ".json":
                raise OSError("nope")
            return real_rename(self, target)

        monkeypatch.setattr(pathlib.Path, "rename", rename_but_not_the_json)

        with pytest.raises(OSError):
            chattree.rename_datastore(old_file, tmp_path / "chat.json")

        assert old_file.is_file()
        assert (old_dir / "abc.png").read_bytes() == b"payload"
        assert not (tmp_path / ("chat" + chattree.SIDECAR_SUFFIX)).exists()


class TestSidecarStorage:
    def test_store_read_roundtrip(self, tmp_path):
        pf = PersistentForest(tmp_path / "chat.json", autosave=False)
        name = pf.store_sidecar(b"hello-bytes", "png")
        assert name.endswith(".png")
        assert pf.read_sidecar(name) == b"hello-bytes"
        assert pf.sidecar_dir.name == "chat" + chattree.SIDECAR_SUFFIX

    def test_store_is_content_addressed_dedup(self, tmp_path):
        pf = PersistentForest(tmp_path / "chat.json", autosave=False)
        n1 = pf.store_sidecar(b"same", "png")
        n2 = pf.store_sidecar(b"same", "png")
        assert n1 == n2
        assert pf.list_sidecar_files() == [n1]

    def test_sidecar_path_rejects_traversal(self, tmp_path):
        pf = PersistentForest(tmp_path / "chat.json", autosave=False)
        for bad in ("../escape.png", "sub/dir.png", "/abs.png"):
            with pytest.raises(ValueError):
                pf.sidecar_path(bad)

    def test_list_empty_when_no_dir(self, tmp_path):
        pf = PersistentForest(tmp_path / "chat.json", autosave=False)
        assert pf.list_sidecar_files() == []

    def test_prune_sweeps_orphans_keeps_referenced(self, tmp_path):
        pf = PersistentForest(tmp_path / "chat.json", autosave=False,
                              sidecar_extractor=_refs_from_sidecars_list)
        kept = pf.store_sidecar(b"kept", "png")
        orphan = pf.store_sidecar(b"orphan", "png")
        pf.create_node({"sidecars": [kept]}, parent_id=None)

        assert pf.list_unreferenced_sidecars() == [orphan]          # dry-run reports, deletes nothing
        assert set(pf.list_sidecar_files()) == {kept, orphan}
        assert pf.prune_unreferenced_sidecars() == [orphan]    # sweep
        assert pf.list_sidecar_files() == [kept]

    def test_prune_scans_all_revisions(self, tmp_path):
        """An old revision's sidecar stays referenced even after the node is edited to drop it."""
        pf = PersistentForest(tmp_path / "chat.json", autosave=False,
                              sidecar_extractor=_refs_from_sidecars_list)
        old_img = pf.store_sidecar(b"old", "png")
        node = pf.create_node({"sidecars": [old_img]}, parent_id=None)
        pf.add_revision(node, {"sidecars": []})  # edit removes the image; old revision still has it
        assert pf.prune_unreferenced_sidecars() == []          # old revision still references it -> kept
        assert pf.list_sidecar_files() == [old_img]

    def test_dry_run_can_discount_doomed_nodes(self, tmp_path):
        """`excluding_nodes` is what lets a preview describe the state *after* the node prune, not before.

        Without it, an attachment referenced only by an unreachable node still counts as live, and the preview
        under-reports exactly the files the cleanup exists to reclaim.
        """
        pf = PersistentForest(tmp_path / "chat.json", autosave=False,
                              sidecar_extractor=_refs_from_sidecars_list)
        kept = pf.store_sidecar(b"kept", "png")
        doomed_file = pf.store_sidecar(b"doomed", "png")
        root = pf.create_node({"sidecars": [kept]}, parent_id=None)
        unreachable = pf.create_node({"sidecars": [doomed_file]}, parent_id=None)

        assert pf.list_unreferenced_sidecars() == []  # both are referenced, as things stand
        doomed_nodes = pf.list_unreachable_nodes(root)
        assert doomed_nodes == [unreachable]
        assert pf.list_unreferenced_sidecars(excluding_nodes=doomed_nodes) == [doomed_file]

        # ...and running the real thing in that order produces what the preview promised.
        pf.prune_unreachable_nodes(root)
        assert pf.prune_unreferenced_sidecars() == [doomed_file]
        assert pf.list_sidecar_files() == [kept]

    def test_prune_without_extractor_is_noop(self, tmp_path):
        pf = PersistentForest(tmp_path / "chat.json", autosave=False)  # no extractor
        pf.store_sidecar(b"data", "png")
        assert pf.prune_unreferenced_sidecars() == []
        assert len(pf.list_sidecar_files()) == 1
        assert pf.list_unreferenced_sidecars() == []


class TestInMemorySidecarStorage:
    """`Forest` holds attachments too, in memory instead of in a directory.

    The policy above the storage — content addressing, first-write-wins descriptions, mark-and-sweep GC —
    is written once and shared, so these do not re-assert it in detail. What they pin is that the two
    backends answer the same questions the same way, and that the two members which can only mean something
    on disk say so rather than returning something plausible.
    """

    def test_store_read_roundtrip(self):
        forest = chattree.Forest()
        name = forest.store_sidecar(b"hello-bytes", "png")
        assert name.endswith(".png")
        assert forest.read_sidecar(name) == b"hello-bytes"
        assert forest.sidecar_size(name) == len(b"hello-bytes")
        assert forest.has_sidecar(name)

    def test_the_same_bytes_get_the_same_name_as_on_disk(self, tmp_path):
        # Content addressing is the shared half, so the name must not depend on where it is stored: a chat
        # moved between the two must keep resolving its own `sidecar:` URLs.
        forest = chattree.Forest()
        pf = PersistentForest(tmp_path / "chat.json", autosave=False)
        assert forest.store_sidecar(b"same bytes", "png") == pf.store_sidecar(b"same bytes", "png")

    def test_a_path_is_an_error_rather_than_a_plausible_answer(self):
        forest = chattree.Forest()
        name = forest.store_sidecar(b"bytes", "png")
        with pytest.raises(NotImplementedError):
            forest.sidecar_path(name)
        with pytest.raises(NotImplementedError):
            forest.sidecar_dir

    def test_unsafe_filenames_are_refused_here_too(self):
        # The check belongs to the data, not to the filesystem: a name out of a corrupt payload is no more
        # trustworthy for being held in memory.
        forest = chattree.Forest()
        for bad in ("../escape.png", "sub/dir.png", "/abs.png"):
            with pytest.raises(ValueError):
                forest.has_sidecar(bad)

    def test_descriptions_are_first_write_wins(self):
        forest = chattree.Forest()
        name = forest.store_sidecar(b"bytes", "png", metadata={"name": "first.png"})
        assert forest.maybe_set_sidecar_metadata(name, {"name": "second.png"}) is False
        assert forest.get_sidecar_metadata(name) == {"name": "first.png"}

    def test_gc_sweeps_what_no_payload_references(self):
        forest = chattree.Forest(sidecar_extractor=lambda payload: set(payload.get("sidecars", [])))
        kept = forest.store_sidecar(b"referenced", "png")
        dropped = forest.store_sidecar(b"orphaned", "png", metadata={"name": "orphan.png"})
        forest.create_node(payload={"sidecars": [kept]}, parent_id=None)

        assert forest.list_unreferenced_sidecars() == [dropped]
        assert forest.prune_unreferenced_sidecars() == [dropped]
        assert forest.list_sidecar_files() == [kept]
        # The description goes with what it describes, or it becomes its own slow leak.
        assert forest.get_sidecar_metadata(dropped) is None

    def test_gc_without_an_extractor_deletes_nothing(self, caplog):
        # An in-memory sidecar occupies RAM for the life of the process and nothing else reclaims it, so an
        # unconfigured store is worth warning about -- but never at the price of deleting a live attachment.
        forest = chattree.Forest()
        name = forest.store_sidecar(b"bytes", "png")
        with caplog.at_level(logging.WARNING):
            assert forest.prune_unreferenced_sidecars() == []
        assert forest.list_sidecar_files() == [name]
        assert "sidecar_extractor" in caplog.text
