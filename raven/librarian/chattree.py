"""Forest data structure, with optional persistence (as JSON).

Used as branching chat history for Raven's LLM client.
"""

__all__ = ["SIDECAR_SUFFIX", "LEGACY_SIDECAR_SUFFIX",
           "rename_datastore",

           "Forest", "PersistentForest"]

import logging
logger = logging.getLogger(__name__)

import atexit
import collections
import contextlib
import copy
import hashlib
import io  # we occasionally need one of Jupiter's moons
import json
import pathlib
import threading
import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from unpythonic import gensym, partition

from ..common import utils as common_utils

class Forest:
    def __init__(self, sidecar_extractor: Callable[[Any], set[str]] | None = None):
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

        **Attachment sidecars**

        A node payload may reference attached bytes — an image, a document — held beside the tree rather than
        inside it, content-addressed by hash. Here they live in memory and last as long as this object;
        `PersistentForest` keeps them in a directory next to its JSON. Everything above the storage is the
        same either way, so a caller stores and reads attachments without knowing which it holds. The two
        members that can only mean something on disk — `sidecar_path` and `sidecar_dir` — say so when asked.

        There is deliberately no way to write these out while the tree stays in memory. A sidecar is
        content-addressed bytes whose meaning lives in the payload that references it, so sidecars saved
        without their tree are hash-named orphans — the state the GC exists to reclaim. Use a
        `PersistentForest` when the chat is worth keeping; it keeps the attachments with it. What is stored
        here is a copy of bytes the caller already had, which makes it a cache rather than an archive.

        `sidecar_extractor`: How to read the sidecar references out of one (otherwise opaque) node payload —
                    a callable `payload -> set[str]` returning the sidecar filenames that payload references.
                    Configured once here by the layer that owns the payload format (for Librarian chats,
                    `raven.librarian.appstate.sidecar_refs_in_payload`), because payloads are opaque to
                    `chattree` by design and only the format owner can read a `sidecar:` reference out of one.
                    `chattree` drives the revision traversal itself and calls this per revision at GC time; it is
                    never invoked during load, so it only needs to understand the *current* payload format.
                    `None` (default) means this datastore does no sidecar GC — `prune_unreferenced_sidecars`
                    becomes a safe no-op (it will not delete anything it can't prove is unreferenced).

                    Worth configuring even in memory, and arguably more so: an in-memory sidecar occupies RAM
                    for the life of the process, and no filesystem cleanup will ever come along and reclaim it.
        """
        self.nodes = {}
        self.lock = threading.RLock()
        self._sidecar_extractor = sidecar_extractor
        self._sidecar_bytes: dict[str, bytes] = {}
        self._sidecar_descriptions: dict[str, dict[str, Any]] = {}

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
            revision_ids = self.get_revisions(node_id)
            if len(revision_ids) == 1:
                raise ValueError(f"Forest.delete_revision: cannot delete the only revision ('{revision_id}') of node '{node_id}'; if you want to delete the node, use `delete_node` or `delete_subtree` instead.")
            assert len(revision_ids) >= 2  # before deletion

            # If deleting the active revision, select another revision to set active after deletion.
            active_revision_before_deletion = self.get_revision(node_id)
            deleting_active_revision = (revision_id == active_revision_before_deletion)
            if deleting_active_revision:
                old_idx = revision_ids.index(revision_id)
                if old_idx == len(revision_ids) - 1:
                    revision_id_to_activate = revision_ids[-2]  # last one deleted -> select the most recent remaining one
                else:
                    revision_id_to_activate = revision_ids[old_idx + 1]  # else select the next newer one

            # Delete.
            node["data"].pop(str(revision_id))
            if str(revision_id) in node["revision_names"]:  # when deleting a revision, delete its name too (if any)
                node["revision_names"].pop(str(revision_id))

            # Set new active revision if needed.
            if deleting_active_revision:
                node["active_revision"] = revision_id_to_activate

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

        The sibling scan is performed via the parent node of `node_id`. The return value is always arity-2
        to support the pattern `children, idx = datastore.get_siblings(node_id)` and then checking for
        `idx is None`.

        **A root node's siblings are the other roots.** A forest holds several trees, and they stand beside
        each other exactly as two replies to one message do — so a caller walking siblings walks between
        trees at the top, which is how a chat under one system prompt is reached from a chat under another.
        `get_all_root_nodes` gets the same list without needing a node to ask from.
        """
        with self.lock:
            if node_id not in self.nodes:
                raise KeyError(f"Forest.get_siblings: no such node '{node_id}'")
            node = self.nodes[node_id]

            parent_node_id = node["parent"]
            if parent_node_id is None:  # root node -> its siblings are the forest's other roots
                siblings = self.get_all_root_nodes()
                return siblings, siblings.index(node_id)

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
        for original_child_node_id in original_node["children"]:
            try:
                self.copy_subtree(node_id=original_child_node_id,
                                  new_parent_id=new_node_id)
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

    def list_unreachable_nodes(self, *roots: str) -> List[str]:
        """Return the IDs of nodes not reachable from any of the `roots` (list of root node unique IDs).

        Note this walks only down (children), not up (parent chain).

        The dry run for `prune_unreachable_nodes`: the same computation, without deleting anything. Useful for
        a pre-commit preview ("would delete N nodes"), and for asking what *else* a prune would take with it —
        e.g. which attachment sidecars only these nodes still reference (see `list_unreferenced_sidecars`).

        The result is sorted, so anything built on it is deterministic (the underlying set iteration is not).
        """
        with self.lock:
            reachable_node_ids = set()
            def find_nodes_reachable_from(node_id):
                if node_id not in self.nodes:
                    logger.warning(f"Forest.list_unreachable_nodes: trying to scan non-existent node '{node_id}'. Ignoring error.")
                    return
                reachable_node_ids.add(node_id)
                node = self.nodes[node_id]
                for child_node_id in node["children"]:
                    find_nodes_reachable_from(child_node_id)

            for root_node_id in roots:
                find_nodes_reachable_from(root_node_id)
            return sorted(set(self.nodes.keys()).difference(reachable_node_ids))

    def prune_unreachable_nodes(self, *roots: str) -> None:
        """Delete any nodes that are not reachable from any of the `roots` (list of root node unique IDs).

        Note this walks only down (children), not up (parent chain).

        Convenient for purging unreachable nodes before saving the forest to disk.
        """
        with self.lock:
            unreachable_node_ids = self.list_unreachable_nodes(*roots)

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

    # ------------------------------------------------------------------
    # Attachment sidecars
    #
    # Split in two on purpose. Everything down to `_sweep_orphaned_descriptions` is *storage*: the handful of
    # operations that differ between holding bytes in a dict and holding them in a directory, and the only
    # thing `PersistentForest` overrides. Everything after it is policy — content-addressing, first-write-wins
    # descriptions, mark-and-sweep GC — and is written once, here, because it is the same reasoning either
    # way. A bug fixed in the GC is fixed for both backends by construction rather than by remembering.

    def _sidecar_exists(self, filename: str) -> bool:
        return filename in self._sidecar_bytes

    def _write_sidecar(self, filename: str, data: bytes) -> None:
        self._sidecar_bytes[filename] = data

    def _delete_sidecar(self, filename: str) -> None:
        """Remove the sidecar and its description. Called only for something `list_sidecar_files` reported."""
        self._sidecar_bytes.pop(filename, None)
        self._sidecar_descriptions.pop(filename, None)

    def _read_sidecar_description(self, filename: str) -> dict[str, Any] | None:
        return self._sidecar_descriptions.get(filename)

    def _write_sidecar_description(self, filename: str, metadata: dict[str, Any]) -> bool:
        self._sidecar_descriptions[filename] = metadata
        return True

    def _sweep_orphaned_descriptions(self) -> None:
        """Drop descriptions whose sidecar is gone.

        Nothing to do in memory — a description is dropped with its sidecar, and no outside hand can remove
        one without the other. `PersistentForest` overrides it, because a directory can be edited by anyone.
        """

    def has_sidecar(self, filename: str) -> bool:
        """Whether sidecar `filename` is present. Raises `ValueError` if the name is not a bare basename.

        The question a caller asks when it holds a name out of a payload and does not yet know whether the
        bytes are still there — a backfill pass, say. Asking it here rather than by probing a path is what
        lets such a caller work against either backend.
        """
        self._validate_sidecar_filename(filename)
        with self.lock:
            return self._sidecar_exists(filename)

    def read_sidecar(self, filename: str) -> bytes:
        """Return the raw bytes of sidecar `filename`. Raises `KeyError` if there is no such sidecar."""
        self._validate_sidecar_filename(filename)
        return self._sidecar_bytes[filename]

    def sidecar_size(self, filename: str) -> int:
        """Return the size of sidecar `filename` in bytes.

        Asked as its own question rather than derived from `sidecar_path(...).stat()`, so that a caller
        reporting how much a cleanup would reclaim works against either backend. Size is a property of the
        bytes; needing a filesystem to learn it was an accident of where the store used to live.
        """
        self._validate_sidecar_filename(filename)
        return len(self._sidecar_bytes[filename])

    def list_sidecar_files(self) -> list[str]:
        """List the sidecar filenames held, sorted. Descriptions are not sidecars and are not listed.

        Sorted so that everything built on it — the GC sweep, its log line, the dry-run preview — is
        deterministic. The exclusion of descriptions is load-bearing rather than cosmetic: the sweep deletes
        every listed name no payload references, and no payload ever references a description.
        """
        with self.lock:
            return sorted(self._sidecar_bytes)

    def _get_sidecar_dir(self) -> pathlib.Path:
        raise NotImplementedError("Forest.sidecar_dir: this datastore holds its attachments in memory, so they "
                                  "have no directory. Use a `PersistentForest` if you need them on disk.")
    sidecar_dir = property(fget=_get_sidecar_dir,
                           doc="Directory holding this datastore's attachment sidecars. In-memory datastores have none; see `PersistentForest`.")

    def sidecar_path(self, filename: str) -> pathlib.Path:
        """Absolute path of sidecar `filename`. In-memory datastores have none; see `PersistentForest`."""
        raise NotImplementedError("Forest.sidecar_path: this datastore holds its attachments in memory, so they "
                                  "have no path. Read the bytes with `read_sidecar`, or use a "
                                  "`PersistentForest` if you need a file.")

    def _validate_sidecar_filename(self, filename: str) -> None:
        # Filenames reaching here come from stored `sidecar:` URLs, i.e. from datastore data. Refuse anything
        # that isn't a bare basename, so a crafted/corrupt datastore can't escape the sidecar directory. The
        # check lives here rather than on the persistent subclass because the *data* is what is untrusted, and
        # an in-memory store built from a loaded payload is no more trustworthy than a file-backed one.
        if pathlib.Path(filename).name != filename:
            raise ValueError(f"{type(self).__name__}: unsafe sidecar filename '{filename}' (must be a bare basename, no path separators).")

    def store_sidecar(self, data: bytes, ext: str, metadata: dict[str, Any] | None = None) -> str:
        """Store `data` as a sidecar; return its content-hash filename `<sha256>.<ext>`.

        Content-addressed: storing identical bytes twice keeps one copy and returns the same name (natural
        dedup). `ext` is the extension without a leading dot (e.g. "png", "jpeg"). The caller decides *what*
        bytes to store — the verbatim original, or a re-encoded downsample (see
        `raven.librarian.imagestore.store_image_as_sidecar`).

        `metadata`, if given, describes this sidecar — see `get_sidecar_metadata` for what it is for and why it
        is kept beside the bytes rather than only in the referencing payload. It is stored as an opaque JSON
        dict; `chattree` neither reads nor interprets its contents, exactly as with node payloads. Must be
        JSON-serializable.

        Deduplication applies to the metadata too, and **first write wins**: attaching identical bytes a second
        time under a different name does not overwrite the record made the first time. Overwriting would let a
        later and possibly worse name — a temp file, a copy with a mangled name — displace a good one, and there
        is no way to tell from here which of two names is the better description of the same bytes.
        """
        filename = f"{hashlib.sha256(data).hexdigest()}.{ext.lstrip('.')}"
        with self.lock:
            if not self._sidecar_exists(filename):  # content-addressed: identical bytes -> identical name
                self._write_sidecar(filename, data)
            if metadata is not None:
                self.maybe_set_sidecar_metadata(filename, metadata)
        return filename

    def maybe_set_sidecar_metadata(self, filename: str, metadata: dict[str, Any]) -> bool:
        """Attach a description to an already-stored sidecar, if it does not have one. Return whether it was written.

        The *maybe* is the first-write-wins rule: returns `False` without touching anything if `filename`
        already has metadata, and `False` again if the write fails. A caller that needs to know whether the
        stored description is now the one it passed has to read the return value; nothing here reports the two
        cases apart, because no caller so far cares which way it declined. See
        `store_sidecar` for why a later name must not displace an earlier one, and `get_sidecar_metadata` for what
        the description is for.

        Also the backfill entry point for datastores predating sidecar metadata. Those still hold the provenance
        in the payloads that reference each sidecar, so it can be recovered for everything currently referenced
        — an orphan from a deletion that already happened is past saving, since the payload that named it went
        with the node.

        Never raises on a write failure: the sidecar is stored and usable either way, and a description that
        could not be written is a display problem rather than a reason to fail the caller's operation.
        """
        with self.lock:
            if self._read_sidecar_description(filename) is not None:
                return False
            return self._write_sidecar_description(filename, metadata)

    def get_sidecar_metadata(self, filename: str) -> dict[str, Any] | None:
        """Return the stored description of sidecar `filename`, or `None` if it has none.

        A sidecar is named by content hash, so the bytes themselves say nothing about where they came from. The
        human-readable name lives in the payload that references it — which is exactly what an *orphaned*
        sidecar no longer has, and orphans are precisely the ones a cleanup preview needs to name. Hence a
        description stored beside the bytes at store time, surviving independently of the tree.

        The cleanup preview is what motivated it, but the larger effect is that a file-backed sidecar directory
        becomes **self-describing**. Without these, it is a pile of hash-named files that can only be
        interpreted by loading the datastore and cross-referencing every payload. With them, anything that can
        read a directory — a person, a shell script, an agent asked to tidy up — can tell what each file is,
        where it came from and when it arrived, without Raven's help and without the datastore being present.

        Returns `None` for a sidecar stored before this existed, or if the description is unreadable or
        corrupt: a missing description is a display problem, never a reason to fail an operation on the file.
        """
        return self._read_sidecar_description(filename)

    def _referenced_sidecars(self, *, excluding_nodes: Iterable[str] = ()) -> set[str]:
        """Union of sidecar filenames referenced by every revision of every node — the GC "mark" phase.

        `chattree` owns this traversal over its own revision model; the per-payload interpretation is delegated
        to the `sidecar_extractor` configured at construction, because payloads are opaque to `chattree` by design.
        Returns an empty set if no extractor is configured (callers guard against acting on that).

        `excluding_nodes` names node IDs to treat as already gone — see `list_unreferenced_sidecars`.
        """
        if self._sidecar_extractor is None:
            return set()
        skip = set(excluding_nodes)
        referenced = set()
        with self.lock:
            for node_id, node in self.nodes.items():
                if node_id in skip:
                    continue
                for payload in node.get("data", {}).values():  # every revision of every node
                    referenced |= set(self._sidecar_extractor(payload))
        return referenced

    def list_unreferenced_sidecars(self, *, excluding_nodes: Iterable[str] = ()) -> list[str]:
        """Sidecar filenames stored but referenced by no revision — the GC dry-run.

        The same computation as `prune_unreferenced_sidecars` without deleting, for a pre-commit preview
        ("would delete N files, X MB"). Returns `[]` (deleting nothing) if no `sidecar_extractor` is configured
        — references can't be determined, so nothing is reported as safe to delete.

        `excluding_nodes` names node IDs to treat as though they had already been deleted, so that the answer
        describes a *future* state rather than the current one. This is what makes an honest preview of the
        full cleanup possible: the two prune steps run as a pair (`prune_unreachable_nodes` first, then
        `prune_unreferenced_sidecars`), so a dry run taken before either has run must discount the references
        held by nodes the first step is about to delete — otherwise it under-reports exactly the attachments
        the cleanup is there to reclaim. Pass `list_unreachable_nodes(*roots)` to preview that pair.
        """
        with self.lock:
            if self._sidecar_extractor is None:
                if self.list_sidecar_files():
                    logger.warning(f"{type(self).__name__}.list_unreferenced_sidecars: no sidecar_extractor configured; cannot determine references.")
                return []
            referenced = self._referenced_sidecars(excluding_nodes=excluding_nodes)
            return [filename for filename in self.list_sidecar_files() if filename not in referenced]

    def prune_unreferenced_sidecars(self) -> list[str]:
        """Delete sidecars referenced by no revision of any node; return the filenames deleted.

        Mark-and-sweep GC. The mark phase (`_referenced_sidecars`) delegates per-payload reading to the
        `sidecar_extractor` configured at construction; the sweep deletes everything else in the store. Pairs
        with `prune_unreachable_nodes`: run that first, so attachments referenced only by
        now-unreachable nodes become unreferenced here and get swept. If no `sidecar_extractor` is configured
        this is a safe no-op — it will not delete anything it cannot prove is unreferenced (returns `[]`, and
        warns if any sidecars exist).
        """
        with self.lock:
            if self._sidecar_extractor is None:
                if self.list_sidecar_files():
                    logger.warning(f"{type(self).__name__}.prune_unreferenced_sidecars: no sidecar_extractor configured; skipping sidecar GC to avoid deleting referenced files.")
                return []
            referenced = self._referenced_sidecars()
            deleted = []
            for filename in self.list_sidecar_files():
                if filename not in referenced:
                    # The description goes with what it describes; otherwise it accumulates as its own slow
                    # leak, which is the very thing this sweep exists to stop.
                    self._delete_sidecar(filename)
                    deleted.append(filename)
            self._sweep_orphaned_descriptions()
            if deleted:
                plural_s = "s" if len(deleted) != 1 else ""
                logger.info(f"{type(self).__name__}.prune_unreferenced_sidecars: deleted {len(deleted)} unreferenced sidecar{plural_s}.")
            return deleted


# Suffix of the directory holding a datastore's attachment sidecars, derived from the datastore's own
# filename so that two datastores in one directory cannot share a sidecar store — which is what keeps the
# GC correct, since a prune against one must not delete files the other still references.
SIDECAR_SUFFIX = ".sidecars"

# What that suffix was before 0.2.9, from when images were the only kind of attachment. It holds documents
# too, so the old name was inaccurate rather than merely terse. `rename_datastore` and `PersistentForest`
# migrate it in place on load; this stays until no datastore in the wild still uses it.
LEGACY_SIDECAR_SUFFIX = ".images"


def rename_datastore(old_file: Union[str, pathlib.Path],
                     new_file: Union[str, pathlib.Path]) -> bool:
    """Rename a datastore file, taking its sidecar directory with it. Returns whether anything moved.

    The pairing is the point. A sidecar directory is named after the datastore file it belongs to, so
    moving the JSON on its own would leave every attachment behind, referenced by URLs that still resolve
    to a filename but no longer to a file.

    Does nothing (returning `False`) if `old_file` does not exist, or if `new_file` already does — this
    never overwrites a datastore, so the worst case of calling it wrongly is that nothing happens.

    Both sidecar suffixes are looked for, so a datastore that predates the `.sidecars` rename can still be
    moved as a unit; the directory keeps whichever suffix it had, and `PersistentForest` migrates that on
    load.

    **All or nothing.** Raises whatever `pathlib.Path.rename` raises, having first put back whatever it had
    already moved — so a caller that treats the failure as non-fatal is looking at the layout it started
    with, not at half of one. That matters here more than it usually would: the halfway state splits the
    datastore from its sidecars, and nothing downstream can detect that, because a sidecar directory is
    identified by *derivation* from the datastore's name rather than by any record of where it went.

    A best-effort rollback, since the undo can fail too — if it does, the exception carries the original
    failure (the useful one) and the log carries the rest.
    """
    old_file = pathlib.Path(old_file).expanduser().resolve()
    new_file = pathlib.Path(new_file).expanduser().resolve()
    if old_file == new_file or not old_file.is_file() or new_file.exists():
        return False

    # Directories first, then the file. Either order can fail halfway, so the ordering is not what makes
    # this safe — the rollback is; this order merely keeps the window short.
    moved_dirs = []
    try:
        for suffix in (SIDECAR_SUFFIX, LEGACY_SIDECAR_SUFFIX):
            old_dir = old_file.with_suffix(suffix)
            new_dir = new_file.with_suffix(suffix)
            if old_dir.is_dir() and not new_dir.exists():
                logger.info(f"rename_datastore: Moving sidecar directory '{old_dir}' -> '{new_dir}'.")
                old_dir.rename(new_dir)
                moved_dirs.append((old_dir, new_dir))

        logger.info(f"rename_datastore: Moving datastore '{old_file}' -> '{new_file}'.")
        old_file.rename(new_file)
    except OSError:
        for old_dir, new_dir in reversed(moved_dirs):
            try:
                new_dir.rename(old_dir)
            except OSError as undo_exc:
                logger.error(f"rename_datastore: Rolling back after a failed rename, could not move "
                             f"'{new_dir}' back to '{old_dir}'. That datastore's attachments are now under a "
                             f"name it does not look for; moving the directory by hand fixes this. "
                             f"Reason {type(undo_exc)}: {undo_exc}")
            else:
                logger.info(f"rename_datastore: Rolled back sidecar directory '{new_dir}' -> '{old_dir}'.")
        raise
    return True


class PersistentForest(Forest):
    def __init__(self,
                 datastore_file: Union[str, pathlib.Path],
                 autosave: bool = True,
                 sidecar_extractor: Callable[[Any], set[str]] | None = None):
        """Exactly like `Forest`, but with persistent storage as JSON.

        `datastore_file`: Where to store the data (for the specific collection you're creating/loading).

        `autosave`: If `True` (default), register `self.save` to be called at interpreter exit via `atexit`.
                    This is the right behaviour for app lifecycle use — whatever state the datastore is in
                    when the app exits, that's what gets persisted.

                    If `False`, skip the `atexit` registration. In-memory mutations still work; the caller
                    is responsible for calling `self.save()` explicitly when (or if) persistence is wanted.
                    This is primarily useful for tests and for ad-hoc inspection of a real datastore file
                    from a Python session, where unconditional autosave would silently rewrite the file
                    at interpreter exit.

        `sidecar_extractor`: How to read the sidecar references out of one (otherwise opaque) node payload —
                    a callable `payload -> set[str]` returning the sidecar filenames that payload references.
                    Configured once here by the layer that owns the payload format (for Librarian chats,
                    `raven.librarian.imagestore.sidecar_refs_in_payload`), because payloads are opaque to
                    `chattree` by design and only the format owner can read a `sidecar:` reference out of one.
                    `chattree` drives the revision traversal itself and calls this per revision at GC time; it is
                    never invoked during load, so it only needs to understand the *current* payload format.
                    `None` (default) means this datastore does no sidecar GC — `prune_unreferenced_sidecars`
                    becomes a safe no-op (it will not delete files it can't prove are unreferenced).
        """
        super().__init__(sidecar_extractor=sidecar_extractor)

        self.datastore_file = datastore_file
        self._autosave = autosave

        # Filesystem-level migration, so it cannot live in `_upgrade` — that one migrates the loaded nodes
        # dict and knows nothing about paths.
        self._migrate_legacy_sidecar_dir()

        # Load persisted state, if any.
        self._load()

        # Persist at shutdown.
        #
        # In `Forest`, we are extra careful in any operations that edit the data, to check and raise errors first,
        # before making any edits. Hence whatever the state is at shutdown, it is the latest valid state, and
        # it is always safe to persist it.
        if autosave:
            atexit.register(self.save)

    def _migrate_legacy_sidecar_dir(self) -> None:
        """Rename this datastore's `<datastore>.images/` to `<datastore>.sidecars/`, if that is what it has.

        A failure here is logged and swallowed rather than raised. The attachments are then unreachable —
        the payloads name files the app looks for under the new directory — but the app still opens, the
        chat text is intact, and the fix is a `mv` the user can perform. Refusing to start over a directory
        rename would be the worse trade.
        """
        new_dir = self.sidecar_dir
        old_dir = new_dir.with_suffix(LEGACY_SIDECAR_SUFFIX)
        if new_dir.exists() or not old_dir.is_dir():
            return
        try:
            old_dir.rename(new_dir)
        except OSError as exc:
            logger.error(f"PersistentForest._migrate_legacy_sidecar_dir: Could not rename '{old_dir}' -> '{new_dir}', "
                         f"so this datastore's attachments will not be found. Renaming it by hand fixes this. "
                         f"Reason {type(exc)}: {exc}")
        else:
            logger.info(f"PersistentForest._migrate_legacy_sidecar_dir: Renamed '{old_dir}' -> '{new_dir}'.")

    def _get_autosave(self) -> bool:
        return self._autosave
    autosave = property(fget=_get_autosave,
                        doc="Whether this instance auto-persists at interpreter exit via `atexit`. Read-only; determined at construction time (changing it post-construction would not affect the already-registered hook, so we prevent the trap by making it immutable).")

    def save(self) -> None:
        """Save the forest to its `datastore_file`, as JSON.

        With `autosave=True` (the default), this is called automatically at interpreter exit via `atexit`.
        With `autosave=False`, the caller must invoke this explicitly if persistence is wanted.
        """
        with self.lock:
            absolute_path = self.datastore_file.expanduser().resolve()
            logger.info(f"PersistentForest.save: Saving datastore to '{str(self.datastore_file)}' (resolved to '{str(absolute_path)}').")

            directory = self.datastore_file.parent
            common_utils.create_directory(directory)

            with open(absolute_path, "w", encoding="utf-8") as json_file:
                json.dump(self.nodes, json_file, indent=2)

            logger.info("PersistentForest.save: All done.")

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
            except Exception:
                logger.warning(f"PersistentForest._load: Caught exception while loading datastore from '{str(absolute_path)}'", exc_info=True)
                logger.info(f"PersistentForest._load: Will create new datastore at '{str(absolute_path)}', at app shutdown.")
            else:
                self._upgrade(data)
                self.nodes.clear()
                self.nodes.update(data)
                plural_s = "s" if len(data) != 1 else ""
                logger.info(f"PersistentForest._load: PersistentForest loaded successfully ({len(data)} node{plural_s}).")

    # --------------------------------------------------------------------------------
    # Attachment sidecar storage
    #
    # Attachments to messages — images, and documents such as PDFs and office files — are stored as files next
    # to the datastore JSON, referenced from messages by `sidecar:<filename>` URLs. Files are named by content
    # hash (`<sha256>.<ext>`), so attaching the same file twice costs one file. These methods own the sidecar
    # *files*; deciding what to store and which files are still referenced lives one layer up (see
    # `raven.librarian.imagestore` and `raven.librarian.textfilestore`), keeping this storage layer free of
    # chat-message-schema knowledge.
    #
    # The directory is `<datastore>` + `SIDECAR_SUFFIX`, derived from the datastore's filename rather than
    # fixed, so two datastores in one directory keep their sidecars apart — which is what keeps the GC
    # correct, since a prune against one must not delete files the other still references.

    # Suffix marking a sidecar's metadata sibling (`<sha256>.<ext>.meta.json`). A sidecar is named by content
    # hash, so the file carries no trace of what it is; this is where its human-readable name and stored-at
    # timestamp live. A suffix on the full sidecar filename rather than a replaced extension, so that two
    # sidecars differing only in extension cannot collide on one metadata file.
    _SIDECAR_METADATA_SUFFIX = ".meta.json"

    def _get_sidecar_dir(self) -> pathlib.Path:
        return pathlib.Path(self.datastore_file).expanduser().resolve().with_suffix(SIDECAR_SUFFIX)
    sidecar_dir = property(fget=_get_sidecar_dir,
                           doc=f"Directory holding this datastore's attachment sidecar files: `<datastore>{SIDECAR_SUFFIX}/`, alongside the JSON. Derived from `datastore_file`; created lazily on the first `store_sidecar`.")

    # The storage half of the sidecar store, as files in `sidecar_dir`. Everything above it — content
    # addressing, first-write-wins descriptions, the mark-and-sweep GC — is `Forest`'s and is not repeated
    # here, so the two backends cannot drift on the policy that is the same for both.

    def _sidecar_exists(self, filename: str) -> bool:
        return self.sidecar_path(filename).exists()

    def _write_sidecar(self, filename: str, data: bytes) -> None:
        directory = self.sidecar_dir
        common_utils.create_directory(directory)
        with open(directory / filename, "wb") as sidecar_file:
            sidecar_file.write(data)

    def _delete_sidecar(self, filename: str) -> None:
        (self.sidecar_dir / filename).unlink()
        self._sidecar_metadata_path(filename).unlink(missing_ok=True)

    def _read_sidecar_description(self, filename: str) -> dict[str, Any] | None:
        metadata_path = self._sidecar_metadata_path(filename)
        try:
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                return json.load(metadata_file)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"PersistentForest._read_sidecar_description: could not read metadata for '{filename}': "
                           f"{type(exc)}: {exc}")
            return None

    def _write_sidecar_description(self, filename: str, metadata: dict[str, Any]) -> bool:
        # Never raises: the sidecar is on disk and usable either way, and a description that could not be
        # written is a display problem rather than a reason to fail the caller's operation.
        metadata_path = self._sidecar_metadata_path(filename)
        try:
            common_utils.create_directory(metadata_path.parent)
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump(metadata, metadata_file, indent=2)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"PersistentForest._write_sidecar_description: could not write metadata for "
                           f"'{filename}': {type(exc)}: {exc}")
            return False
        return True

    def _sidecar_metadata_path(self, filename: str) -> pathlib.Path:
        """Path of the metadata sibling describing sidecar `filename`. Validates `filename`; existence unchecked."""
        path = self.sidecar_path(filename)
        return path.with_name(f"{path.name}{self._SIDECAR_METADATA_SUFFIX}")

    def _sweep_orphaned_descriptions(self) -> None:
        """Delete metadata files whose sidecar is gone.

        The sweep removes each description along with its file, so this only finds strays — a sidecar deleted by
        hand, or by a version that did not know about metadata. Since a metadata file is never referenced by
        anything, nothing else would ever collect it. The in-memory store overrides this to nothing: there,
        the two cannot come apart.
        """
        directory = self.sidecar_dir
        if not directory.is_dir():
            return
        for entry in directory.iterdir():
            if entry.is_file() and entry.name.endswith(self._SIDECAR_METADATA_SUFFIX):
                described = entry.name[:-len(self._SIDECAR_METADATA_SUFFIX)]
                if not (directory / described).exists():
                    entry.unlink(missing_ok=True)

    def sidecar_path(self, filename: str) -> pathlib.Path:
        """Absolute path to sidecar file `filename` within `sidecar_dir`. Does not check existence."""
        self._validate_sidecar_filename(filename)
        return self.sidecar_dir / filename

    def read_sidecar(self, filename: str) -> bytes:
        """Read and return the raw bytes of sidecar file `filename`."""
        with open(self.sidecar_path(filename), "rb") as sidecar_file:
            return sidecar_file.read()

    def sidecar_size(self, filename: str) -> int:
        """Return the size of sidecar file `filename` in bytes, without reading it."""
        return self.sidecar_path(filename).stat().st_size

    def list_sidecar_files(self) -> list[str]:
        """List the sidecar filenames present in `sidecar_dir` (bare names, not paths), sorted. Empty if the directory doesn't exist yet.

        Sorted rather than in raw `iterdir` order so that everything built on it — the GC sweep, its log line,
        and the dry-run preview — is deterministic and platform-independent (filesystem iteration order is not).
        A UI is free to re-sort by a more meaningful key (size, provenance URL) on top.

        Metadata siblings are *not* listed: they live in the same directory but describe sidecars rather than
        being ones. The exclusion is load-bearing rather than cosmetic — the GC sweep deletes every listed file
        no payload references, and no payload ever references a metadata file, so listing them here would delete
        the whole set on the first cleanup.
        """
        directory = self.sidecar_dir
        if not directory.is_dir():
            return []
        return sorted(entry.name for entry in directory.iterdir()
                      if entry.is_file() and not entry.name.endswith(self._SIDECAR_METADATA_SUFFIX))

    def _upgrade(self, nodes: Dict[str, Dict[str, Any]]) -> None:
        """Migrate `nodes` (loaded from a saved datastore) to the latest format.

        Called automatically by `_load`.

        NOTE: There are two upgrade functions. This updates the forest itself
              to support revisioned data.

              See also `chatutil.upgrade_datastore`, which upgrades the
              payload format inside each revision of the data.
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
