"""Chat app state saving/loading.

This module is shared between `minichat` (command-line app) and `app` (GUI app).
"""

__all__ = ["load", "save"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import atexit
import functools
import json
import pathlib
from typing import Dict, Union

from unpythonic.env import env

from . import chattree
from . import chatutil

# --------------------------------------------------------------------------------
# Helper functions

def _reset_datastore_and_update_state(settings: env,
                                      datastore: chattree.Forest,
                                      state: Dict) -> None:
    """Factory-reset `datastore`.

    Its fresh 'new_chat_HEAD' will be written to `state`, and the 'HEAD' of `state` will be set to the 'new_chat_HEAD'.
    """
    # Factory-reset first. This creates the first two nodes (system prompt with character card, and the AI's initial greeting).
    state["new_chat_HEAD"] = chatutil.factory_reset_datastore(datastore, settings)
    state["HEAD"] = state["new_chat_HEAD"]  # current last node in chat; like HEAD pointer in git

def _scan_for_new_chat_head(datastore: chattree.Forest) -> str:
    """Scan for the AI greeting node that starts a new chat, and return its node ID.

    This is needed if the app state file is missing or corrupted.
    """
    with datastore.lock:
        root_node_ids = datastore.get_all_root_nodes()
        if not root_node_ids:
            raise ValueError("No system prompt nodes found in chat database, cannot proceed.")
        if len(root_node_ids) > 1:
            logger.warning("_scan_for_new_chat_head: Found more than one system prompt node in chat database, picking the first one.")
        system_prompt_node_id = root_node_ids[0]
        system_prompt_node = datastore.nodes[system_prompt_node_id]
        n_new_chat_heads = len(system_prompt_node["children"])
        if not n_new_chat_heads:
            raise ValueError(f"System prompt node '{system_prompt_node_id}' has no AI greeting nodes attached to it, cannot proceed.")
        if n_new_chat_heads > 1:
            logger.warning(f"_scan_for_new_chat_head: Found more than one AI greeting node attached to system prompt node '{system_prompt_node_id}'; picking the first one, '{system_prompt_node['children'][0]}'.")
        new_chat_node_id = system_prompt_node["children"][0]
        return new_chat_node_id

# --------------------------------------------------------------------------------
# API

def load(llm_settings: env,
         datastore_file: Union[str, pathlib.Path],
         state_file: Union[str, pathlib.Path]) -> Dict:
    """Load chat app state.

    `llm_settings`: LLM client settings; this is the return value of `llmclient.setup`.

    `datastore_file`: Path to the JSON file to load the persistent chat forest from.
                      Will be auto-persisted to the same path at app exit.

    `state_file`: Path to the app state JSON file to load things such as the
                  new-chat HEAD, the current chat HEAD, and various settings.
                  Will be auto-persisted to the same path at app exit.

    Return value is the tuple `(datastore, state)`, where:
        `datastore` is a `chattree.PersistentForest` containing the chat database,
        `state` is a `dict` containing the settings.

    NOTE: Object identity is important - for the state auto-persist (at app exit)
          to work correctly, you should modify the original `state` dict in-place;
          that object is what gets auto-persisted at exit.

    NOTE: The state file is important; without it, it is impossible to know
          which chat node to use as the HEAD of a new chat session.

          Hence if the state file is missing or does not contain "new_chat_HEAD",
          this will **FACTORY-RESET** the chat datastore, thus deleting all chats.

          Otherwise, the chat datastore is loaded normally.

          If "HEAD" is missing in the state file, it will be set to "new_chat_HEAD",
          so that when the app opens, it opens into a new chat session.

          If any settings are missing in the state file, they are initialized to
          their default values (which are defined in the source code of this function).
    """
    # Resolve paths
    orig_datastore_file = datastore_file
    orig_state_file = state_file
    datastore_file = pathlib.Path(datastore_file).expanduser().resolve()
    state_file = pathlib.Path(state_file).expanduser().resolve()

    # Ensure directories exist
    datastore_dir = datastore_file.parent
    datastore_dir.mkdir(parents=True, exist_ok=True)
    state_dir = state_file.parent
    state_dir.mkdir(parents=True, exist_ok=True)

    # Load app state
    try:
        with open(state_file, "r", encoding="utf-8") as json_file:
            state = json.load(json_file)
    except FileNotFoundError:
        logger.info(f"load: App state file '{orig_state_file}' (resolved to '{state_file}') does not exist.")
        state = {}
    else:
        logger.info(f"load: Loaded app state from '{orig_state_file}' (resolved to '{state_file}').")

    # Load datastore
    datastore = chattree.PersistentForest(datastore_file)  # This autoloads and auto-persists.
    with datastore.lock:
        if datastore.nodes:
            logger.info(f"load: Loaded chat datastore from '{orig_datastore_file}' (resolved to '{datastore_file}'). Found {len(datastore.nodes)} chat nodes in datastore.")
        else:
            logger.info("load: No chat nodes in datastore at '{orig_datastore_file}' (resolved to '{datastore_file}'). Creating new datastore, will be saved at app exit.")
            _reset_datastore_and_update_state(llm_settings, datastore, state)

    # Set any missing app state to defaults
    #
    if "new_chat_HEAD" not in state:  # New-chat start node ID missing -> reset datastore
        logger.info(f"load: Missing key 'new_chat_HEAD' in '{orig_state_file}' (resolved to '{state_file}'). Scanning chat datastore for 'new_chat_HEAD'.")
        state["new_chat_HEAD"] = _scan_for_new_chat_head(datastore)
        logger.info(f"load: Scan found 'new_chat_HEAD', it is now '{state['new_chat_HEAD']}'.")

    if "HEAD" not in state:  # Current chat node ID missing -> start at new chat
        state["HEAD"] = state["new_chat_HEAD"]
        logger.info(f"load: Missing key 'HEAD' in '{orig_state_file}' (resolved to '{state_file}'), resetting it to 'new_chat_HEAD'")

    if "docs_enabled" not in state:
        state["docs_enabled"] = True
        logger.info(f"load: Missing key 'docs_enabled' in '{orig_state_file}' (resolved to '{state_file}'), using default '{state['docs_enabled']}'")

    if "speculate_enabled" not in state:
        state["speculate_enabled"] = False
        logger.info(f"load: Missing key 'speculate_enabled' in '{orig_state_file}' (resolved to '{state_file}'), using default '{state['speculate_enabled']}'")

    if "avatar_subtitles" not in state:
        state["avatar_subtitles"] = True
        logger.info(f"load: Missing key 'avatar_subtitles' in '{orig_state_file}' (resolved to '{state_file}'), using default '{state['avatar_subtitles']}'")

    # Refresh the system prompt in the datastore (to the one currently produced by `llmclient`)
    new_chat_node_id = state["new_chat_HEAD"]
    system_prompt_node_id = datastore.get_parent(new_chat_node_id)
    state["system_prompt_node_id"] = system_prompt_node_id  # remember it, the GUI chat client needs it
    old_system_prompt_revision_id = datastore.get_revision(node_id=system_prompt_node_id)
    datastore.add_revision(node_id=system_prompt_node_id,
                           payload={"message": chatutil.create_initial_system_message(llm_settings)})
    datastore.delete_revision(node_id=system_prompt_node_id,
                              revision_id=old_system_prompt_revision_id)

    # Migrate datastore (this updates only if needed)
    chatutil.upgrade_datastore(datastore, system_prompt_node_id)  # v0.2.3+: data format change

    # Set up auto-persist for app state
    atexit.register(functools.partial(save,
                                      state_file=orig_state_file,
                                      state=state))

    return datastore, state

def save(state_file: Union[str, pathlib.Path],
         state: Dict) -> None:
    """Save chat app state.

    `state_file`: Path to the app state JSON file to save in.

    `state`: The state dictionary that was returned by `load`.

    NOTE: `load` automatically registers this function to be called at app exit,
          using the original `state_file` and `state` arguments.
    """
    # validate
    required_keys = ("new_chat_HEAD",
                     "HEAD",
                     "docs_enabled",
                     "speculate_enabled")
    if any(key not in state for key in required_keys):
        raise KeyError(f"at least one required setting missing from `state`; required keys = {list(sorted(required_keys))}; got existing keys = {list(sorted(state.keys()))}")

    orig_state_file = state_file
    state_file = pathlib.Path(state_file).expanduser().resolve()

    with open(state_file, "w", encoding="utf-8") as json_file:
        json.dump(state, json_file, indent=2)

    logger.info(f"save: Saved app state to '{orig_state_file}' (resolved to '{state_file}').")
