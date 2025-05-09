"""Branching chat history for the LLM client."""

__all__ = ["create_node", "delete_node", "copy_node",
           "copy_subtree", "delete_subtree",
           "detach_subtree", "detach_children",
           "reparent_subtree", "reparent_children",
           "walk_up",
           "linearize_branch",
           "get_all_root_nodes", "prune_unreachable_nodes", "prune_dead_links",
           "print_datastore", "clear_datastore", "save_datastore", "load_datastore"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import collections
import copy
import json
import pathlib
import threading
from typing import Any, Callable, List, Optional, Union

from unpythonic import gensym, partition

from . import utils

# TODO: Make this into a class (like `HybridIR` is), so that we can have per-instance independent datastores.

# For easy JSON-ability, we store the chat nodes in a global dictionary, as a doubly-linked forest:
#
# {"node_unique_id": {"id": "node_unique_id",   # so that each node knows its own ID
#                     "data": Any,              # the API is generic by design; in practice, a chat message in the form {"role": ..., "content": ...}
#                     "parent": Optional[str],  # unique_id_of_parent_node; or for a root node, `None`
#                     "children": List[str]     # [unique_id_of_child0, ...]
#                    }
# }
#
# Since each node has at most one parent, but may have many children, this makes a forest structure.
# We simplify that to a tree by having just one root - the system prompt node.
#
# Starting from any node, it is easy to produce a linearized history up to that point, by walking up the parent chain.
#
storage = {}  # global storage for chat node records (JSON-compatible)
storage_lock = threading.RLock()

def create_node(message: Any, parent_id: Optional[str]) -> str:
    """Create a chat node from `message` and store it in the global chat node storage.

    Link it to the parent node with unique id `parent_id`, if given. Linking is done in both directions:
      - The new node gets a parent node, and
      - The parent node gets a new child node (added to the end of the list).

    If `parent_id is None`, the new node becomes a root node.

    There is no limitation on how many root nodes the forest may have.

    **IMPORTANT**: A chat node is specific to the place it appears in the forest. Do NOT attempt to use the
                   same instance in unrelated places. Doing so will mess up the parent/children links.

                   If you need to link a copy of a node to a new place in the forest, use `copy_node`.
                   It copies the message data, too, to avoid unintended edits.

    Returns the unique ID of the new node.
    """
    node_id = str(gensym("chat-node"))  # string form for easy JSON-ability
    node = {"id": node_id,
            "data": message,
            "parent": parent_id,  # link to parent
            "children": []}
    with storage_lock:
        if parent_id is not None:  # link parent to this node (do this first before saving the new node, in case `parent_id` is not found)
            storage[parent_id]["children"].append(node_id)
        storage[node_id] = node
    return node_id

def copy_node(node_id: str, new_parent_id: Optional[str]) -> str:
    """Copy chat node `node_id`, copying also its contents.

    Optionally, link the new node to a given parent node (linking is performed in both directions).

    The contents are copied via `copy.deepcopy`.

    Child nodes of the original node are NOT copied, and are NOT linked to the new node.
    If you want a recursive copy, use `copy_subtree` instead.
    """
    original_node = storage[node_id]
    copied_message = copy.deepcopy(original_node["data"])  # just {"role": ..., "content": ...}
    new_node_id = create_node(copied_message, new_parent_id)
    return new_node_id

def copy_subtree(node_id: str, new_parent_id: Optional[str]) -> str:
    """Copy the chat subtree starting from `node_id`, copying also the node's contents, and recursively, all child nodes.

    The contents are copied via `copy.deepcopy`.

    Optionally, link the new node to a given parent node (linking is performed in both directions).
    """
    original_node = storage[node_id]  # look up first to raise KeyError if needed, before we create any nodes
    new_node_id = copy_node(node_id, new_parent_id)
    new_node = storage[new_node_id]
    for original_child_node_id in original_node["children"]:
        try:
            new_child_node_id = copy_subtree(node_id=original_child_node_id,
                                             new_parent_id=new_node_id)
            # The parent of each child node is set during node creation, so we only need to update the links on the parent side.
            new_node["children"].append(new_child_node_id)
        except KeyError:
            logger.warning(f"copy_subtree: while recursively copying node '{node_id}': one of the child nodes, '{original_child_node_id}', does not exist. Ignoring error.")
    return new_node_id

def delete_node(node_id: str) -> None:
    """Delete a chat node from the global chat node storage.

    Links to this node from both directions are deleted, but all those other nodes remain in the storage.
    Each of the child nodes of this node becomes a new root node.

    There is no limitation on how many root nodes the forest may have.

    If you want to delete the whole subtree instead, use `delete_subtree`.
    """
    with storage_lock:
        detach_subtree(node_id)  # this will also raise KeyError if the node is not found
        detach_children(node_id)
        storage.pop(node_id)  # the global storage has the only reference to the actual node data, so the node becomes eligible for GC

def delete_subtree(node_id: str) -> None:
    """Delete the chat subtree starting from `node_id`. All child nodes are deleted, recursively."""
    with storage_lock:
        detach_subtree(node_id)  # sever link from parent, if any (this will also raise KeyError if the node is not found)

        # At this step broken links no longer matter, since the whole subtree (which is now detached) is being deleted
        def recursive_delete(node_id):
            node = storage[node_id]
            for child_node_id in node["children"]:
                try:
                    recursive_delete(child_node_id)
                except KeyError:
                    logger.warning(f"delete_subtree: while deleting children of '{node_id}': one of its child nodes '{child_node_id}' does not exist. Ignoring error.")
            storage.pop(node_id)
        recursive_delete(node_id)

def detach_subtree(node_id: str) -> str:
    """Detach the chat subtree starting from `node_id`, so that `node_id` becomes a new root node.

    In other words, this function severs the parent link of `node_id`, on both sides.

    There is no limitation on how many root nodes the forest may have.

    This is mostly a utility for the delete functions, but is also part of the public API.

    For convenience, returns `node_id`.
    """
    node = storage[node_id]
    parent_node_id = node["parent"]
    if parent_node_id is not None:  # a root node has no parent
        try:
            parent_node = storage[parent_node_id]
        except KeyError:
            logger.warning(f"detach_subtree: while detaching node '{node_id}' from its parent: its parent node '{parent_node_id}' does not exist. Ignoring error.")
        else:
            try:
                parent_node["children"].remove(node_id)
            except ValueError:
                logger.warning(f"detach_subtree: while detaching node '{node_id}' from its parent: this node was not listed in the children of its parent node '{parent_node_id}'. Ignoring error.")
    node["parent"] = None
    return node_id

# TODO: do we need `reparent_children`, for symmetry with `reparent_subtree`?
def detach_children(node_id: str) -> str:
    """Detach all children of `node_id`, so that each of them becomes a new root node.

    In other words, this function severs the child links of `node_id`, on both sides.

    There is no limitation on how many root nodes the forest may have.

    This is mostly a utility for the delete functions, but is also part of the public API.

    For convenience, returns `node_id`.
    """
    with storage_lock:
        node = storage[node_id]  # this will raise KeyError if `node_id` is not found
        for child_node_id in node["children"]:
            try:
                child_node = storage[child_node_id]
            except KeyError:
                logger.warning(f"detach_children: while detaching node '{node_id}' from its children: one of the child nodes, '{child_node_id}', does not exist. Ignoring error.")
            else:
                child_node["parent"] = None
        node["children"].clear()
    return node_id

def reparent_subtree(node_id: str, new_parent_id: Optional[str]) -> str:  # Not sure if this operation is needed, ever.
    """Reparent (reattach to a different parent node) the chat subtree starting from `node_id`.

    If `new_parent_id is None`, just detach that subtree (equivalent to `detach_subtree`).
    """
    with storage_lock:
        # look up first to raise KeyError if needed, before we edit any nodes
        node = storage[node_id]
        if new_parent_id is not None:
            new_parent_node = storage[new_parent_id]

        detach_subtree(node_id)

        # have a new parent to set?
        if new_parent_id is not None:
            node["parent"] = new_parent_id
            new_parent_node["children"].append(node_id)

def reparent_children(node_id: str, new_parent_id: Optional[str]) -> str:
    """Reparent (reattach to a different parent node) all children of `node_id`.

    The children are appended to the children list of `new_parent_id`; it does not matter whether it already has child nodes.

    If `new_parent_id is None`, just detach the children (equivalent to `detach_children`).
    """
    with storage_lock:
        # look up first to raise KeyError if needed, before we edit any nodes
        node = storage[node_id]
        if new_parent_id is not None:
            new_parent_node = storage[new_parent_id]

        detached_children = copy.copy(node["children"])  # copy because detach clears the "children" field
        detach_children(node_id)

        # have a new parent to set?
        if new_parent_id is not None:
            for child_node_id in detached_children:
                try:
                    child_node = storage[child_node_id]
                except KeyError:
                    logger.warning(f"delete_node: while reparenting children of node '{node_id}' (to '{new_parent_id}'): one of the child nodes, '{child_node_id}', does not exist. Ignoring error.")
                else:
                    child_node["parent"] = new_parent_id
                    new_parent_node["children"].append(child_node_id)

def walk_up(node_id: str, callback: Optional[Callable] = None) -> str:
    """Given a chat node with `node_id`, walk up the parent chain until a root node is reached.

    `callback`: Optional. This can be used e.g. to gather data from the parent chain.

                For each node encountered, including `node_id` itself, `callback` (if provided) is called
                with one argument, the actual node data record. The return value of `callback` is ignored.

                `callback` may raise `StopIteration` to terminate the walk at that node.
                This is useful when looking for a specific node further up the chain, but not quite at the root.

    Returns the unique ID of the root node that was found, or the unique ID of the node where the walk was terminated
    (if told to stop by `callback`).
    """
    with storage_lock:
        node = storage[node_id]
        while True:
            if callback is not None:
                try:
                    callback(node)
                except StopIteration:
                    break
            parent_node_id = node["parent"]
            if parent_node_id is None:
                break
            node = storage[parent_node_id]
        return node["id"]

def linearize_branch(node_id: str) -> List[Any]:
    """Walking up from `node_id` until we reach a root node, return a linearized chat history for that branch.

    Note `node_id` doesn't need to be a leaf of the chat tree; but it will be the last node of the linearized history.

    This collects the "data" field from each node and puts those into a list, in chronological order (root node first).

    The linearized history can be sent to the LLM to build the context.
    """
    linearized_history = collections.deque()
    def prepend_to_history(node):
        linearized_history.appendleft(node["data"])
    walk_up(node_id, callback=prepend_to_history)
    return list(linearized_history)

def get_all_root_nodes() -> List[str]:
    """Return the IDs of all root nodes (i.e. nodes whose parent is `None`) currently in storage.

    We don't keep track of these separately; this is done by an O(n) linear scan over the whole global chat node storage.
    """
    return [node["id"] for node in storage.values() if node["parent"] is None]

def prune_unreachable_nodes(*roots: str) -> None:
    """Delete any chat nodes in the global storage that are not reachable from any of the `roots` (list of root node unique IDs).

    Note this walks only down (children), not up (parent chain).

    Convenient to purge unreachable data before saving the chat datastore to disk.
    """
    with storage_lock:
        reachable_node_ids = set()
        def find_nodes_reachable_from(node_id):
            if node_id not in storage:
                logger.warning(f"prune_unreachable_nodes: trying to scan non-existent chat node '{node_id}'. Ignoring error.")
                return
            reachable_node_ids.add(node_id)
            node = storage[node_id]
            for child_node_id in node["children"]:
                find_nodes_reachable_from(child_node_id)

        for root_node_id in roots:
            find_nodes_reachable_from(root_node_id)
        all_node_ids = set(storage.keys())
        unreachable_node_ids = all_node_ids.difference(reachable_node_ids)

        if unreachable_node_ids:
            plural_s = "s" if len(unreachable_node_ids) != 1 else ""
            logger.info(f"prune_unreachable_nodes: found {len(unreachable_node_ids)} unreachable node{plural_s}. Deleting.")

        for unreachable_node_id in unreachable_node_ids:
            delete_node(unreachable_node_id)  # this ensures any links to them get removed too

def prune_dead_links(*roots: str) -> None:
    """Delete any links (parent or child) in the chat tree that point to a nonexistent node.

    This is a depth-first tree scan that starts at each of the `roots` (list of root node unique IDs).

    If a node's parent does not exist, that node becomes a root node.

    If a node's child does not exist, that child is removed from the list of children.

    Dead links should never occur; we provide this utility just in case.
    """
    with storage_lock:
        def walk(node_id):
            node = storage[node_id]

            parent_node_id = node["parent"]
            if parent_node_id is not None and parent_node_id not in storage:  # dead link?
                logger.warning(f"Node '{node_id}' links to nonexistent parent '{parent_node_id}'; removing the link.")
                node["parent"] = None

            nonexistent_children, valid_children = partition(pred=lambda node_id: node_id in storage,
                                                             iterable=node["children"])
            nonexistent_children = list(nonexistent_children)
            valid_children = list(valid_children)

            if nonexistent_children:  # any dead links?
                logger.warning(f"Node '{node_id}' links to one or more nonexistent children {nonexistent_children}; removing the links.")
                node["children"].clear()
                node["children"].extend(valid_children)

            for child_node_id in node["children"]:  # walk the remaining (valid) ones
                walk(child_node_id)

        for root_node_id in roots:
            walk(root_node_id)

def print_datastore() -> None:
    """Show the raw contents of the global storage for chat nodes. For debugging."""
    with storage_lock:
        for node_id, node in storage.items():
            print(f"{node_id}")  # on its own line for easy copy'n'pasting
            for key, value in node.items():
                print(f"    {key}: {value}")
            print()

def clear_datastore() -> None:
    """Delete all data in the global chat node storage.

    In-memory operation, does not affect saved datastores.
    """
    with storage_lock:
        storage.clear()

def save_datastore(path: Union[str, pathlib.Path]) -> None:
    """Save the global chat node storage to a file, so that it can be reloaded later with `load_datastore`."""
    with storage_lock:
        absolute_path = path.expanduser().resolve()
        logger.info(f"save_datastore: Saving chat node datastore to '{str(path)}' (resolved to '{str(absolute_path)}').")

        directory = path.parent
        logger.info(f"save_datastore: Creating directory '{str(directory)}'.")
        utils.create_directory(directory)

        logger.info("save_datastore: Saving data.")
        with open(absolute_path, "w") as json_file:
            json.dump(storage, json_file, indent=2)

        logger.info("save_datastore: All done.")

def load_datastore(path: Union[str, pathlib.Path]) -> None:
    """Load the global chat node storage to a file.

    Loading replaces the current in-memory datastore.
    """
    with storage_lock:
        absolute_path = path.expanduser().resolve()
        logger.info(f"load_datastore: Loading chat node datastore from '{str(path)}' (resolved to '{str(absolute_path)}').")

        try:
            with open(absolute_path, "r") as json_file:
                data = json.load(json_file)
        except Exception as exc:
            logger.warning(f"load_datastore: While loading datastore from '{str(absolute_path)}': {type(exc)}: {exc}")
        else:
            storage.clear()
            storage.update(data)
            logger.info("load_datastore: Datastore loaded successfully.")
