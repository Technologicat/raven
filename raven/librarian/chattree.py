"""Forest data structure, with optional persistence (as JSON).

Used as branching chat history for Raven's LLM client.
"""

__all__ = ["Forest", "PersistentForest"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import atexit
import collections
import contextlib
import copy
import io  # we occasionally need one of Jupiter's moons
import json
import pathlib
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from unpythonic import gensym, partition

from ..common import utils as common_utils

class Forest:
    def __init__(self):
        """Forest datastore with data revisioning.

        Each node has at most one parent, but may have many children, making a forest structure.
        Starting from any node, it is easy to produce a linearized branch up to that point,
        by walking up the parent chain.

        NOTE: It is the caller's responsibility to keep a copy of important node IDs (such as root nodes);
        this class only provides the forest structure itself.

        NOTE: This class provides various methods for creating, reading, updating and deleting the nodes in the tree.
        If you need to do something that is not covered by the existing methods, the raw node storage can be accessed
        via `datastore.nodes`, where `datastore` is your `Forest` instance. See storage format and thread-safety notes below.

        The lock is a `threading.RLock`, so other functions from the same thread can still access the datastore
        while it is already locked.

        For a persistent version, see `PersistentForest`.

        For easy JSON-ability, we store the nodes in a dictionary, as a doubly-linked forest:

        {"node_unique_id": {"id": "node_unique_id",                  # so that each node knows its own ID
                            "timestamp": int,                        # as nanoseconds since epoch
                            "data": {revision_id: payload,           # payload is any JSON serializable data, the node content
                                     ...},
                            "active_revision": int,                  # current default revision of data
                            "next_free_revision": int,               # next revision ID that has never been used for this node
                            "revision_names": {revision_id0: name0,  # revisions can be named (optional)
                                               ...}
                            "parent": Optional[str],                 # unique_id_of_parent_node; or for a root node, `None`
                            "children": List[str]                    # [unique_id_of_child0, ...]
                           }
        }


        **Thread safety**

        If you access nodes manually, in order to be thread-safe, you should `with datastore.lock` the dynamic extent
        where you do so, at least if you expect that relevant things might be changed by another thread.

        For locking and grabbing a single node, there is a convenient context manager::

            with datastore.node(node_id) as my_node:
                ...

        This gives you the requested node, while also locking the datastore for the dynamic extent of the `with`.

        If you need lock-free manual access, EAFP to avoid TOCTTOU::

            try:
                my_node = datastore.nodes[node_id]
            except KeyError:  # wasn't there
                ...
            else:  # you have the node now
                ...

        That is, atomize the check-and-get by just trying to grab a reference, instead of checking for presence separately.

        Lock-free access is usually fine for a single node - though then there aren't any guarantees whether that node is
        still in the datastore by the time you're done with it (vs. having been deleted in another thread).

        If you want to walk links, it is advisable to lock the datastore first, just to be safe against any creations or deletions
        that might affect the vicinity you are looking at.
        """
        self.nodes = {}
        self.lock = threading.RLock()

    def create_node(self, payload: Any, parent_id: Optional[str]) -> str:
        """Create a node containing `payload`, and store it in the forest.

        Link it to the parent node with unique id `parent_id`, if given. Linking is done in both directions:
          - The new node gets a parent node, and
          - The parent node gets a new child node (added to the end of the list of children).

        If `parent_id is None`, the new node becomes a root node.

        There is no limitation on how many root nodes the forest may have.

        Raven v0.2.3+: When a node is created, it gets its "timestamp" field set to `time.time_ns()`.
                       Note this only concerns newly created nodes; any copied nodes retain their
                       original timestamps.

                       The node now supports payload revisioning:

                           "data": {revision_id0: payload,
                                    ...},
                           "active_revision": revision_id0,
                           "next_free_revision": revision_id1,
                           "revision_names": {revision_id0: name0,
                                              ...}

                       When the node is created, the initial payload is stored as revision 1.

                       The revision ID is a 1-based integer.

                       The "next_free_revision" counter tracks the first nonnegative integer that has
                       not yet been used for a revision of this node. If you delete a revision later,
                       its ID is never reused - so that for any given node, any given revision number
                       is guaranteed to only ever point at one specific revision (if it points to anything).

        **IMPORTANT**: A node is specific to the place it appears in the forest. Do NOT attempt to use the
                       same instance in unrelated places. Doing so will mess up the parent/children links.

                       If you need to link a copy of a node to a new place in the forest, use `copy_node`.
                       It copies the content, too, to avoid unintended edits.

        Returns the unique ID of the new node.
        """
        node_id = str(gensym("forest-node"))  # string form for easy JSON-ability
        node = {"id": node_id,
                "timestamp": time.time_ns(),
                "active_revision": 1,
                "next_free_revision": 2,
                "revision_names": {},  # str(int) -> str, to allow a client app to give a human-readable name (entered by the user) to zero or more revisions
                "data": {str(1): payload},  # use str key for JSON compatibility (we abstract this detail away; API takes/returns revision IDs as int)
                "parent": parent_id,  # link to parent
                "children": []}
        with self.lock:
            if parent_id is not None:  # link parent to this node (do this first before saving the new node, in case `parent_id` is not found)
                self.nodes[parent_id]["children"].append(node_id)
            self.nodes[node_id] = node
        return node_id

    def add_revision(self, node_id: str, payload: Any, revision_name: Optional[str] = None) -> int:
        """Add a new payload revision to node `node_id`, and make the new revision active.

        `payload`: The payload. Semantics depend on your app.

                   For `PersistentForest`, needs to be JSON-able to facilitate saving/loading.

                   For example, in Raven-librarian, the payload is a `dict`, which contains
                   the chat message (see `chatutil.create_chat_message`) and its metadata,
                   such as the revision's creation timestamp, as well as AI generation
                   metadata when applicable. See `chatutil.create_payload`.

        `revision_name`: Optional human-readable name for the revision.

                         Most often, revisions are not named, but sometimes it can be
                         helpful if the user can set a label to help them remember
                         what the revision was about.

                         This parameter a convenience feature to be able to name the new revision
                         right away, if a name is known. You can also set/change the name later,
                         with `set_revision_name`.

        Returns the revision ID of the new revision.

        The main use case of revisioning is to facilitate a chat client to allow the user
        to fix typos and/or perform editorial changes after the fact (e.g. for sharing a
        polished chat log online).

        For restarting the conversation from a given point and taking it in a completely
        different direction, then it is better to branch the chat, by creating a new sibling
        node (i.e. get the original node's parent node, and add a new child node to that).
        """
        with self.lock:
            if node_id not in self.nodes:
                raise KeyError(f"Forest.add_revision: no such node '{node_id}'")
            node = self.nodes[node_id]
            revision_id = node["next_free_revision"]
            node["data"][str(revision_id)] = payload
            if revision_name is not None:
                node["revision_names"][str(revision_id)] = revision_name
            node["active_revision"] = revision_id
            node["next_free_revision"] += 1
        return revision_id

    def delete_revision(self, node_id: str, revision_id: int) -> None:
        """Delete an existing payload revision from node `node_id`."""
        with self.lock:
            if node_id not in self.nodes:
                raise KeyError(f"Forest.delete_revision: no such node '{node_id}'")
            node = self.nodes[node_id]
            if str(revision_id) not in node["data"]:
                raise KeyError(f"Forest.delete_revision: node '{node_id}' has no revision '{revision_id}'")
            node["data"].pop(str(revision_id))
            if str(revision_id) in node["revision_names"]:  # when deleting a revision, delete its name too (if any)
                node["revision_names"].pop(str(revision_id))

    def get_revisions(self, node_id: str) -> List[int]:
        """Return a list of all revision IDs of the payload revisions of node `node_id`, in numerical order."""
        with self.lock:
            if node_id not in self.nodes:
                raise KeyError(f"Forest.get_revisions: no such node '{node_id}'")
            node = self.nodes[node_id]
        return [int(revision_id) for revision_id in node["data"].keys()]  # already sorted because we add revisions in numerical order

    def get_revision(self, node_id: str) -> int:
        """Return the revision ID of the active payload revision of node `node_id`."""
        with self.lock:
            if node_id not in self.nodes:
                raise KeyError(f"Forest.get_revision: no such node '{node_id}'")
            node = self.nodes[node_id]
        return node["active_revision"]

    def get_revision_name(self, node_id: str, revision_id: int) -> Optional[str]:
        """Return the human-readable name of payload revision `revision_id` of node `node_id`, if it is named.

        If not named, return `None`.

        To get a list of all revision names::

            revision_names = [datastore.get_revision_name(node_id, revision_id) for revision_id in datastore.get_revisions(node_id)]
        """
        with self.lock:
            if node_id not in self.nodes:
                raise KeyError(f"Forest.get_revision_name: no such node '{node_id}'")
            node = self.nodes[node_id]
            if str(revision_id) in node["revision_names"]:
                return node["revision_names"][str(revision_id)]
            return None

    def set_revision(self, node_id: str, revision_id: int) -> None:
        """Set the active payload revision of node `node_id`.

        This causes `get_payload` to return that revision as the default.
        """
        with self.lock:
            if node_id not in self.nodes:
                raise KeyError(f"Forest.set_revision: no such node '{node_id}'")
            node = self.nodes[node_id]
            if str(revision_id) not in node["data"]:
                raise KeyError(f"Forest.set_revision: node '{node_id}' has no revision '{revision_id}'")
        node["active_revision"] = revision_id

    def set_revision_name(self, node_id: str, revision_id: int, revision_name: str) -> str:
        """Set the human-readable name of payload revision `revision_id` of node `node_id`.

        The revision must exist.

        The existing name of the revision, if any, is overwritten.

        Returns `revision_name`, for convenience.
        """
        with self.lock:
            if node_id not in self.nodes:
                raise KeyError(f"Forest.set_revision_name: no such node '{node_id}'")
            node = self.nodes[node_id]
            if str(revision_id) not in node["data"]:  # the revision being named must exist in the payloads
                raise KeyError(f"Forest.set_revision_name: node '{node_id}' has no revision '{revision_id}'")
            assert str(revision_id) in node["data"]
            node["revision_names"][str(revision_id)] = revision_name
            return revision_name

    def get_payload(self, node_id: str, revision_id: Optional[int] = None) -> Any:
        """Return the payload of node `node_id`.

        `revision_id`: optionally, specify which payload revision to return.

                       If `revision_id is None` (default), return the currently active revision.

        See `get_revisions` (get list of available revisions) and `set_revision` (choose active revision).

        NOTE: This returns a reference to the original payload as-is (not a copy).
        """
        with self.lock:
            if node_id not in self.nodes:
                raise KeyError(f"Forest.get_payload: no such node '{node_id}'")
            node = self.nodes[node_id]
            if revision_id is None:
                revision_id = node["active_revision"]
            else:
                if str(revision_id) not in node["data"]:
                    raise KeyError(f"Forest.get_payload: node '{node_id}' has no revision '{revision_id}'")
            assert str(revision_id) in node["data"]
            return node["data"][str(revision_id)]

    # Return type: https://stackoverflow.com/questions/49733699/python-type-hints-and-context-managers
    @contextlib.contextmanager
    def node(self, node_id: str) -> Iterator[Dict]:
        """Context manager: get the node `node_id` in a thread-safe manner, for direct access.

        The datastore is locked for the dynamic extent of the context so that e.g. the payload revision
        and any links to children are guaranteed to stay the same.
        """
        with self.lock:
            if node_id not in self.nodes:
                raise KeyError(f"Forest.node: no such node '{node_id}'")
            yield self.nodes[node_id]

    def get_parent(self, node_id: str) -> Optional[str]:
        """Return the parent of `node_id`.

        It may be `None` if `node_id` is a root node.
        """
        with self.lock:
            if node_id not in self.nodes:
                raise KeyError(f"Forest.get_parent: no such node '{node_id}'")
            node = self.nodes[node_id]
            parent = node["parent"]
            return parent

    def get_children(self, node_id: str) -> List[str]:
        """Return a list of children of `node_id`.

        That list may be empty, if `node_id` is a leaf node.
        """
        with self.lock:
            if node_id not in self.nodes:
                raise KeyError(f"Forest.get_children: no such node '{node_id}'")
            node = self.nodes[node_id]
            children = node["children"]
            return children

    def get_siblings(self, node_id: str) -> Tuple[Optional[List[str]], Optional[int]]:
        """Return a list of siblings of `node_id`, including that node itself.

        Returns the tuple `(siblings, node_index)`, where:
            `siblings` is a list of node IDs,
            `node_index` is the (0-based) index of `node_id` itself in the `siblings` list.

        The sibling scan is performed via the parent node of `node_id`. If the parent is not found,
        the return value is `(None, None)`. The return value is always arity-2 to support the pattern
        `children, idx = datastore.get_siblings(node_id)` and then checking for `idx is None`.

        A root node is defined as having no siblings. If you want all roots, use `get_all_root_nodes`.
        """
        with self.lock:
            if node_id not in self.nodes:
                raise KeyError(f"Forest.get_siblings: no such node '{node_id}'")
            node = self.nodes[node_id]

            parent_node_id = node["parent"]
            if parent_node_id is None:  # root node?
                return None, None

            if parent_node_id not in self.nodes:
                raise KeyError(f"Forest.get_siblings: node '{node_id}': its parent node '{parent_node_id}' does not exist.")
            parent_node = self.nodes[parent_node_id]

            siblings = parent_node["children"]  # including the node itself so we can get its index
            try:
                node_index = siblings.index(node_id)
            except ValueError:
                raise ValueError(f"Forest.get_siblings: node '{node_id}' is not in the children of its parent")
            return siblings, node_index

    def copy_node(self, node_id: str, new_parent_id: Optional[str]) -> str:
        """Copy node `node_id`, copying also its contents.

        Optionally, link the new node to a given parent node (linking is performed in both directions).
        If not linked, the new node becomes another root node in the forest.

        The contents are copied via `copy.deepcopy`. Payload revisions and their IDs are preserved;
        the new node gets a full deep copy of the revision history of the original node `node_id`.

        Returns the node ID of the new node.

        Child nodes of the original node are NOT copied, and are NOT linked to the new node.
        If you want a recursive copy, use `copy_subtree` instead.
        """
        with self.lock:
            original_node = self.nodes[node_id]
            new_node_id = self.create_node(payload="__dummy_content__",  # we will replace the dummy content almost immediately...
                                           parent_id=new_parent_id)
            # ...by deep-copying the payload revision history from the original node
            new_node = self.nodes[new_node_id]
            new_node["data"] = copy.deepcopy(original_node["data"])
            new_node["active_revision"] = original_node["active_revision"]
            new_node["next_free_revision"] = original_node["next_free_revision"]
            new_node["revision_names"] = copy.deepcopy(original_node["revision_names"])
            return new_node_id

    def copy_subtree(self, node_id: str, new_parent_id: Optional[str]) -> str:
        """Copy the subtree starting from `node_id`, copying also the node's contents, and recursively, all child nodes.

        The contents are copied via `copy.deepcopy`.

        Optionally, link the new node to a given parent node (linking is performed in both directions).

        Returns the node ID of the new node that is the copy of node `node_id`.
        """
        original_node = self.nodes[node_id]  # look up first to raise KeyError if needed, before we create any nodes
        new_node_id = self.copy_node(node_id, new_parent_id)
        new_node = self.nodes[new_node_id]
        for original_child_node_id in original_node["children"]:
            try:
                new_child_node_id = self.copy_subtree(node_id=original_child_node_id,
                                                      new_parent_id=new_node_id)
                # The parent of each child node is set during node creation, so we only need to update the links on the parent side.
                new_node["children"].append(new_child_node_id)
            except KeyError:
                logger.warning(f"Forest.copy_subtree: while recursively copying node '{node_id}': one of the child nodes, '{original_child_node_id}', does not exist. Ignoring error.")
        return new_node_id

    def delete_node(self, node_id: str) -> None:
        """Delete a node from the forest.

        Links to this node from both directions are severed, but all those other nodes remain in the storage.
        Each of the child nodes of this node becomes a new root node.

        There is no limitation on how many root nodes the forest may have.

        If you want to delete the whole subtree instead, use `delete_subtree`.
        """
        with self.lock:
            self.detach_subtree(node_id)  # this will also raise KeyError if the node is not found
            self.detach_children(node_id)
            self.nodes.pop(node_id)  # the datastore has the only reference to the actual node data, so the node becomes eligible for GC

    def delete_subtree(self, node_id: str) -> None:
        """Delete the subtree starting from `node_id`. All child nodes are deleted, recursively."""
        with self.lock:
            self.detach_subtree(node_id)  # sever link from parent, if any (this will also raise KeyError if the node is not found)

            # At this step broken links no longer matter, since the whole subtree (which is now detached) is being deleted
            def recursive_delete(node_id):
                node = self.nodes[node_id]
                for child_node_id in node["children"]:
                    try:
                        recursive_delete(child_node_id)
                    except KeyError:
                        logger.warning(f"Forest.delete_subtree: while deleting children of '{node_id}': one of its child nodes '{child_node_id}' does not exist. Ignoring error.")
                self.nodes.pop(node_id)
            recursive_delete(node_id)

    def detach_subtree(self, node_id: str) -> str:
        """Detach the subtree starting from `node_id`, so that `node_id` becomes a new root node.

        In other words, this function severs the parent link of `node_id`, on both sides.

        There is no limitation on how many root nodes the forest may have.

        This is mostly a utility for the delete functions, but is also part of the public API.

        For convenience, returns `node_id`.
        """
        node = self.nodes[node_id]
        parent_node_id = node["parent"]
        if parent_node_id is not None:  # a root node has no parent
            try:
                parent_node = self.nodes[parent_node_id]
            except KeyError:
                logger.warning(f"Forest.detach_subtree: while detaching node '{node_id}' from its parent: its parent node '{parent_node_id}' does not exist. Ignoring error.")
            else:
                try:
                    parent_node["children"].remove(node_id)
                except ValueError:
                    logger.warning(f"Forest.detach_subtree: while detaching node '{node_id}' from its parent: this node was not listed in the children of its parent node '{parent_node_id}'. Ignoring error.")
        node["parent"] = None
        return node_id

    # TODO: do we need `reparent_children`, for symmetry with `reparent_subtree`?
    def detach_children(self, node_id: str) -> str:
        """Detach all children of `node_id`, so that each of them becomes a new root node.

        In other words, this function severs the child links of `node_id`, on both sides.

        There is no limitation on how many root nodes the forest may have.

        This is mostly a utility for the delete functions, but is also part of the public API.

        For convenience, returns `node_id`.
        """
        with self.lock:
            node = self.nodes[node_id]  # this will raise KeyError if `node_id` is not found
            for child_node_id in node["children"]:
                try:
                    child_node = self.nodes[child_node_id]
                except KeyError:
                    logger.warning(f"Forest.detach_children: while detaching node '{node_id}' from its children: one of the child nodes, '{child_node_id}', does not exist. Ignoring error.")
                else:
                    child_node["parent"] = None
            node["children"].clear()
        return node_id

    def reparent_subtree(self, node_id: str, new_parent_id: Optional[str]) -> str:  # Not sure if this operation is needed, ever.
        """Reparent (reattach to a different parent node) the subtree starting from `node_id`.

        If `new_parent_id is None`, just detach that subtree (equivalent to `detach_subtree`).

        For convenience, returns `node_id`.
        """
        with self.lock:
            # look up first to raise KeyError if needed, before we edit any nodes
            node = self.nodes[node_id]
            if new_parent_id is not None:
                new_parent_node = self.nodes[new_parent_id]

            self.detach_subtree(node_id)

            # have a new parent to set?
            if new_parent_id is not None:
                node["parent"] = new_parent_id
                new_parent_node["children"].append(node_id)
        return node_id

    def reparent_children(self, node_id: str, new_parent_id: Optional[str]) -> str:
        """Reparent (reattach to a different parent node) all children of `node_id`.

        The children are appended to the children list of `new_parent_id`; it does not matter whether it already has child nodes.

        If `new_parent_id is None`, just detach the children (equivalent to `detach_children`).

        For convenience, returns `node_id`.
        """
        with self.lock:
            # look up first to raise KeyError if needed, before we edit any nodes
            node = self.nodes[node_id]
            if new_parent_id is not None:
                new_parent_node = self.nodes[new_parent_id]

            detached_children = copy.copy(node["children"])  # copy because detach clears the "children" field
            self.detach_children(node_id)

            # have a new parent to set?
            if new_parent_id is not None:
                for child_node_id in detached_children:
                    try:
                        child_node = self.nodes[child_node_id]
                    except KeyError:
                        logger.warning(f"Forest.reparent_children: while reparenting children of node '{node_id}' (to '{new_parent_id}'): one of the child nodes, '{child_node_id}', does not exist. Ignoring error.")
                    else:
                        child_node["parent"] = new_parent_id
                        new_parent_node["children"].append(child_node_id)
        return node_id

    def walk_up(self, node_id: str, callback: Optional[Callable] = None) -> str:
        """Starting from `node_id`, walk up the parent chain until a root node is reached.

        `callback`: Optional. This can be used e.g. to gather data from the parent chain.

                    For each node encountered, including `node_id` itself, `callback` (if provided) is called
                    with one argument, the actual node data record. The return value of `callback` is ignored.

                    `callback` may raise `StopIteration` to terminate the walk at that node.
                    This is useful when looking for a specific node further up the chain, but not quite at the root.

        Returns the unique ID of the root node that was found, or the unique ID of the node where the walk was terminated
        (if told to stop by `callback`).
        """
        with self.lock:
            node = self.nodes[node_id]
            while True:
                if callback is not None:
                    try:
                        callback(node)
                    except StopIteration:
                        break
                parent_node_id = node["parent"]
                if parent_node_id is None:
                    break
                node = self.nodes[parent_node_id]
            return node["id"]

    def linearize_up(self, node_id: str) -> List[str]:
        """Walking up from `node_id` up to and including a root node, return a linearized representation of that branch.

        This collects the node ID of each node, and puts those into a list, in depth order (root node first).

        Note that the starting `node_id` doesn't need to be a leaf node; but it will be the last node of the linearized
        representation; children are not scanned.
        """
        linearized_history = collections.deque()
        def prepend_to_history(node):
            linearized_history.appendleft(node["id"])
        self.walk_up(node_id, callback=prepend_to_history)
        return list(linearized_history)

    def get_all_root_nodes(self) -> List[str]:
        """Return the IDs of all root nodes (i.e. nodes whose parent is `None`) currently in the forest.

        We don't keep track of these separately; this is done by an O(n) linear scan over the whole forest.
        """
        return [node["id"] for node in self.nodes.values() if node["parent"] is None]

    def prune_unreachable_nodes(self, *roots: str) -> None:
        """Delete any nodes that are not reachable from any of the `roots` (list of root node unique IDs).

        Note this walks only down (children), not up (parent chain).

        Convenient for purging unreachable nodes before saving the forest to disk.
        """
        with self.lock:
            reachable_node_ids = set()
            def find_nodes_reachable_from(node_id):
                if node_id not in self.nodes:
                    logger.warning(f"Forest.prune_unreachable_nodes: trying to scan non-existent node '{node_id}'. Ignoring error.")
                    return
                reachable_node_ids.add(node_id)
                node = self.nodes[node_id]
                for child_node_id in node["children"]:
                    find_nodes_reachable_from(child_node_id)

            for root_node_id in roots:
                find_nodes_reachable_from(root_node_id)
            all_node_ids = set(self.nodes.keys())
            unreachable_node_ids = all_node_ids.difference(reachable_node_ids)

            if unreachable_node_ids:
                plural_s = "s" if len(unreachable_node_ids) != 1 else ""
                logger.info(f"Forest.prune_unreachable_nodes: found {len(unreachable_node_ids)} unreachable node{plural_s}. Deleting.")

            for unreachable_node_id in unreachable_node_ids:
                self.delete_node(unreachable_node_id)  # this ensures any links to them get removed too

    def prune_dead_links(self, *roots: str) -> None:
        """Delete any links (parent or child) that point to a nonexistent node.

        This is a depth-first tree scan that starts at each of the `roots` (list of root node unique IDs).

        Note this walks only down (children), not up (parent chain).

        If a node's parent does not exist, that node becomes a root node.

        If a node's child does not exist, that child is removed from the list of children.

        Dead links should never occur; we provide this utility just in case.
        """
        with self.lock:
            def walk(node_id):
                node = self.nodes[node_id]

                parent_node_id = node["parent"]
                if parent_node_id is not None and parent_node_id not in self.nodes:  # dead link?
                    logger.warning(f"Forest.prune_dead_links: Node '{node_id}' links to nonexistent parent '{parent_node_id}'; removing the link.")
                    node["parent"] = None

                nonexistent_children, valid_children = partition(pred=lambda node_id: node_id in self.nodes,
                                                                 iterable=node["children"])
                nonexistent_children = list(nonexistent_children)
                valid_children = list(valid_children)

                if nonexistent_children:  # any dead links?
                    logger.warning(f"Forest.prune_dead_links: Node '{node_id}' links to one or more nonexistent children, {nonexistent_children}; removing the links.")
                    node["children"].clear()
                    node["children"].extend(valid_children)

                for child_node_id in node["children"]:  # walk the remaining (valid) ones
                    walk(child_node_id)

            for root_node_id in roots:
                walk(root_node_id)

    def __str__(self) -> str:
        """Return a human-readable, multiline string listing all nodes currently in the datastore. Mainly for debugging."""
        output = io.StringIO()
        with self.lock:
            for node_id, node in self.nodes.items():
                output.write(f"{node_id}\n")  # on its own line for easy copy'n'pasting
                for key, value in node.items():
                    output.write(f"    {key}: {value}\n")
                output.write("\n")
        return output.getvalue()

    def purge(self) -> None:
        """Delete all data in the forest.

        Affects in-memory first; if this is a `PersistentForest` instance, persisted at app shutdown.
        """
        with self.lock:
            self.nodes.clear()

class PersistentForest(Forest):
    def __init__(self,
                 datastore_file: Union[str, pathlib.Path]):
        """Exactly like `Forest`, but with persistent storage as JSON.

        `datastore_file`: Where to store the data (for the specific collection you're creating/loading).
        """
        super().__init__()

        self.datastore_file = datastore_file

        # Load persisted state, if any.
        self._load()

        # Persist at shutdown.
        #
        # In `Forest`, we are extra careful in any operations that edit the data, to check and raise errors first,
        # before making any edits. Hence whatever the state is at shutdown, it is the latest valid state, and
        # it is always safe to persist it.
        atexit.register(self._save)

    def _save(self) -> None:
        """Save the forest to a file, so that it can be reloaded later with `_load`.

        This is called automatically at app exit time.
        """
        with self.lock:
            absolute_path = self.datastore_file.expanduser().resolve()
            logger.info(f"PersistentForest._save: Saving datastore to '{str(self.datastore_file)}' (resolved to '{str(absolute_path)}').")

            directory = self.datastore_file.parent
            common_utils.create_directory(directory)

            with open(absolute_path, "w", encoding="utf-8") as json_file:
                json.dump(self.nodes, json_file, indent=2)

            logger.info("PersistentForest._save: All done.")

    def _load(self) -> None:
        """Load the forest from a file.

        Loading replaces the current in-memory forest.

        This is called automatically at instantiation time.
        """
        with self.lock:
            absolute_path = self.datastore_file.expanduser().resolve()
            logger.info(f"PersistentForest._load: Loading datastore from '{str(self.datastore_file)}' (resolved to '{str(absolute_path)}').")

            try:
                with open(absolute_path, "r", encoding="utf-8") as json_file:
                    data = json.load(json_file)
            except Exception as exc:
                logger.warning(f"PersistentForest._load: While loading datastore from '{str(absolute_path)}': {type(exc)}: {exc}")
                logger.info(f"PersistentForest._load: Will create new datastore at '{str(absolute_path)}', at app shutdown.")
            else:
                self._upgrade(data)
                self.nodes.clear()
                self.nodes.update(data)
                plural_s = "s" if len(data) != 1 else ""
                logger.info(f"PersistentForest._load: PersistentForest loaded successfully ({len(data)} node{plural_s}).")

    def _upgrade(self, nodes: Dict[str, Dict[str, Any]]) -> None:
        """Migrate `nodes` (loaded from a saved datastore) to the latest format.

        Called automatically by `_load`.

        NOTE: There are two upgrade functions. This updates the forest itself
              to support revisioned data.

              See also `llmclient.upgrade`, which upgrades the payload format
              inside each revision of the data.
        """
        upgrade_time = time.time_ns()
        for node_id, node in nodes.items():
            # v0.2.3+: chat node timestamps
            if "timestamp" not in node:
                node["timestamp"] = upgrade_time

            # v0.2.3+: revision history
            if "active_revision" not in node:
                node["active_revision"] = 1
                node["next_free_revision"] = 2
                node["data"] = {str(1): node["data"]}  # up to v0.2.2, the "data" field (payload) has no revisions container
            if "revision_names" not in node:  # separate check, because I didn't think of needing this feature later, until I had committed and uploaded the code
                node["revision_names"] = {}
