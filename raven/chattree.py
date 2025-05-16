"""Forest data structure, with optional persistence (as JSON).

Used as branching chat history for Raven's LLM client.
"""

__all__ = ["Forest", "PersistentForest"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import atexit
import collections
import copy
import io  # we occasionally need one of Jupiter's moons
import json
import pathlib
import threading
from typing import Any, Callable, List, Optional, Union

from unpythonic import gensym, partition

from . import utils

class Forest:
    def __init__(self):
        """Forest datastore.

        Each node has at most one parent, but may have many children, making a forest structure.
        Starting from any node, it is easy to produce a linearized branch up to that point,
        by walking up the parent chain.

        NOTE: it is the caller's responsibility to keep a copy of important node IDs (such as root nodes);
        this class only provides the forest structure itself.

        For a persistent version, see `PersistentForest`.

        For easy JSON-ability, we store the nodes in a dictionary, as a doubly-linked forest:

        {"node_unique_id": {"id": "node_unique_id",   # so that each node knows its own ID
                            "data": Any,              # any JSON serializable data, the node content
                            "parent": Optional[str],  # unique_id_of_parent_node; or for a root node, `None`
                            "children": List[str]     # [unique_id_of_child0, ...]
                           }
        }
        """
        self.nodes = {}
        self.lock = threading.RLock()

    def create_node(self, data: Any, parent_id: Optional[str]) -> str:
        """Create a node conataining `data`, and store it in the forest.

        Link it to the parent node with unique id `parent_id`, if given. Linking is done in both directions:
          - The new node gets a parent node, and
          - The parent node gets a new child node (added to the end of the list of children).

        If `parent_id is None`, the new node becomes a root node.

        There is no limitation on how many root nodes the forest may have.

        **IMPORTANT**: A node is specific to the place it appears in the forest. Do NOT attempt to use the
                       same instance in unrelated places. Doing so will mess up the parent/children links.

                       If you need to link a copy of a node to a new place in the forest, use `copy_node`.
                       It copies the content, too, to avoid unintended edits.

        Returns the unique ID of the new node.
        """
        node_id = str(gensym("forest-node"))  # string form for easy JSON-ability
        node = {"id": node_id,
                "data": data,
                "parent": parent_id,  # link to parent
                "children": []}
        with self.lock:
            if parent_id is not None:  # link parent to this node (do this first before saving the new node, in case `parent_id` is not found)
                self.nodes[parent_id]["children"].append(node_id)
            self.nodes[node_id] = node
        return node_id

    def copy_node(self, node_id: str, new_parent_id: Optional[str]) -> str:
        """Copy node `node_id`, copying also its contents.

        Optionally, link the new node to a given parent node (linking is performed in both directions).

        The contents are copied via `copy.deepcopy`.

        Child nodes of the original node are NOT copied, and are NOT linked to the new node.
        If you want a recursive copy, use `copy_subtree` instead.
        """
        original_node = self.nodes[node_id]
        copied_message = copy.deepcopy(original_node["data"])
        new_node_id = self.create_node(copied_message, new_parent_id)
        return new_node_id

    def copy_subtree(self, node_id: str, new_parent_id: Optional[str]) -> str:
        """Copy the subtree starting from `node_id`, copying also the node's contents, and recursively, all child nodes.

        The contents are copied via `copy.deepcopy`.

        Optionally, link the new node to a given parent node (linking is performed in both directions).
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
            self.nodes.pop(node_id)  # the actual datastore has the only reference to the actual node data, so the node becomes eligible for GC

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

    def reparent_children(self, node_id: str, new_parent_id: Optional[str]) -> str:
        """Reparent (reattach to a different parent node) all children of `node_id`.

        The children are appended to the children list of `new_parent_id`; it does not matter whether it already has child nodes.

        If `new_parent_id is None`, just detach the children (equivalent to `detach_children`).
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

    def linearize_up(self, node_id: str) -> List[Any]:
        """Walking up from `node_id` up to and including a root node, return a linearized representation of that branch.

        This collects the "data" field from each node and puts those into a list, in depth order (root node first).

        Note `node_id` doesn't need to be a leaf node; but it will be the last node of the linearized representation;
        children are not scanned.
        """
        linearized_history = collections.deque()
        def prepend_to_history(node):
            linearized_history.appendleft(node["data"])
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

        Convenient for purging unreachable data before saving the forest to disk.
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

        Affects in-memory first; persisted at app shutdown.
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
        # it's always safe to persist it.
        atexit.register(self._save)

    def _save(self) -> None:
        """Save the forest to a file, so that it can be reloaded later with `_load`.

        This is called automatically at app exit time.
        """
        with self.lock:
            absolute_path = self.datastore_file.expanduser().resolve()
            logger.info(f"PersistentForest._save: Saving datastore to '{str(self.datastore_file)}' (resolved to '{str(absolute_path)}').")

            directory = self.datastore_file.parent
            utils.create_directory(directory)

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
                self.nodes.clear()
                self.nodes.update(data)
                plural_s = "s" if len(data) != 1 else ""
                logger.info(f"PersistentForest._load: PersistentForest loaded successfully ({len(data)} node{plural_s}).")
