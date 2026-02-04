"""Unit tests for raven.librarian.chattree (Forest and PersistentForest)."""

import pathlib

import pytest

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

    def test_get_siblings_of_root_returns_none(self, chain):
        f, a, _b, _c = chain
        siblings, idx = f.get_siblings(a)
        assert siblings is None
        assert idx is None

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
        gc = f.create_node("grandchild", parent_id=left)

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

    def test_prune_unreachable_with_multiple_roots(self, forest):
        r1 = forest.create_node("r1", parent_id=None)
        r2 = forest.create_node("r2", parent_id=None)
        orphan = forest.create_node("orphan", parent_id=None)
        forest.prune_unreachable_nodes(r1, r2)
        assert r1 in forest.nodes
        assert r2 in forest.nodes
        assert orphan not in forest.nodes

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
        pf1._save()

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
