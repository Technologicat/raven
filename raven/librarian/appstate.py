"""Persistent chat app state.

Importantly, contains the HEAD node ID of the current chat, as well as some persistent option flags.

This module is shared between `minichat` (command-line app) and `app` (Raven-librarian GUI app).
"""

__all__ = ["load", "save"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import atexit
import functools
import json
import pathlib
from typing import Dict, Tuple, Union

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

def _get_system_prompt_node_id(datastore: chattree.Forest) -> str:
    """Return the chat node ID of the system prompt.

    `datastore`: `chattree.PersistentForest` containing the chat database.
    """
    root_node_ids = datastore.get_all_root_nodes()
    if not root_node_ids:
        logger.error("_get_system_prompt_node_id: No system prompt nodes found in datastore, cannot proceed.")
        raise ValueError("No system prompt nodes found in datastore, cannot proceed.")
    if len(root_node_ids) > 1:
        logger.info(f"_get_system_prompt_node_id: There is more than one system prompt node in datastore, picking the first one. IDs of all system prompt nodes: {root_node_ids}")
    logger.info(f"_get_system_prompt_node_id: System prompt node ID is '{root_node_ids[0]}'.")
    return root_node_ids[0]

def _refresh_system_prompt(llm_settings: env,
                           datastore: chattree.Forest,
                           state: Dict) -> None:
    """Refresh the system prompt in the datastore (to the one currently produced by `llmclient`).

    A new revision is created on the system prompt node, and the previous revision is deleted.

    NOTE: This is an evil mutating function that writes to `datastore`. The write happens in-memory;
    if `datastore` is a `PersistentForest`, it persists the changes at app exit.

    NOTE: This also writes to `state["system_prompt_node_id"]`, as the Raven-librarian GUI needs it too.

    `llm_settings`: LLM client settings; this is the return value of `llmclient.setup`.

    `datastore`: `chattree.PersistentForest` containing the chat database.

    `state`: `dict` containing the app state (HEAD node, various persistent settings).
    """
    system_prompt_node_id = _get_system_prompt_node_id(datastore)
    state["system_prompt_node_id"] = system_prompt_node_id  # remember it, the GUI chat client needs it
    old_system_prompt_revision_id = datastore.get_revision(node_id=system_prompt_node_id)
    datastore.add_revision(node_id=system_prompt_node_id,
                           payload=chatutil.create_payload(llm_settings=llm_settings,
                                                           message=chatutil.create_initial_system_message(llm_settings)))
    datastore.delete_revision(node_id=system_prompt_node_id,
                              revision_id=old_system_prompt_revision_id)

def _refresh_greeting(llm_settings: env,
                      datastore: chattree.Forest,
                      state: Dict) -> None:
    """Refresh "new_chat_HEAD" so that it points to `llm_settings.greeting`.

    If the current greeting is found under the system prompt node, this simply sets the "new_chat_HEAD" pointer.

    Otherwise, a new node is created (under the system prompt node), the current greeting is written there,
    and the "new_chat_HEAD" pointer is set to that new node.

    NOTE: This is an evil mutating function that updates `state` (and possibly writes to `datastore`).

    NOTE: This uses `state["system_prompt_node_id"]`, so that needs to be up to date first.
          The app state loader calls `_refresh_system_prompt` first, ensuring proper initialization.

    `llm_settings`: LLM client settings; this is the return value of `llmclient.setup`.

    `datastore`: `chattree.PersistentForest` containing the chat database.

    `state`: `dict` containing the app state (HEAD node, various persistent settings).
    """
    with datastore.lock:
        # Scan AI greeting nodes under the system prompt node. Look for one that matches the currently configured greeting.
        #
        # The greeting must be under the system prompt node we actually use (in case there are several),
        # so it can be used with that system prompt node, to preserve the forest structure
        # (cannot link to an AI greeting node under a different system prompt node).
        system_prompt_node_id = state["system_prompt_node_id"]
        greeting_node_ids = datastore.get_children(system_prompt_node_id)

        # Due to the OAI-compatible chatlog format, the actual stored message content begins with the AI character's name,
        # e.g. "Aria: How can I help you today?".
        #
        # So format the greeting as a chat message for the currently configured AI character,
        # so that we can detect whether the datastore has this greeting for this character.
        greeting_message = chatutil.create_chat_message(llm_settings=llm_settings,
                                                        role="assistant",
                                                        text=llm_settings.greeting.strip())
        greeting_message_content = greeting_message["content"]

        for greeting_node_id in greeting_node_ids:
            payload = datastore.get_payload(greeting_node_id)  # get currently active revision
            message = payload["message"]
            message_role = message["role"]
            message_text = message["content"]
            if message_role != "assistant":  # skip non-AI messages (should not happen, but let's be robust)
                logger.warning(f"_refresh_greeting: Detected non-AI message node (role = '{message_role}') '{greeting_node_id}' under system prompt node '{system_prompt_node_id}'. Skipping.")
                continue
            if message_text.strip() == greeting_message_content:  # found it?
                logger.info(f"_refresh_greeting: Found currently configured AI greeting for current AI character '{llm_settings.char}' at AI message node '{greeting_node_id}' under system prompt node '{system_prompt_node_id}'.")
                break
        else:  # Currently configured greeting not found under the system prompt node -> create new node for it
            logger.info(f"_refresh_greeting: Currently configured AI greeting text (see `raven.llmclient.config`) for current AI character '{llm_settings.char}' not found under system prompt node '{system_prompt_node_id}'. Creating new AI greeting node for it.")
            greeting_node_id = datastore.create_node(payload=chatutil.create_payload(llm_settings=llm_settings,
                                                                                     message=greeting_message),
                                                     parent_id=system_prompt_node_id)
            logger.info(f"_refresh_greeting: Created new AI greeting node '{greeting_node_id}' for current AI character '{llm_settings.char}' under system prompt node '{system_prompt_node_id}'.")
        logger.info(f"_refresh_greeting: Setting 'new_chat_HEAD' to {llm_settings.char}'s AI greeting node '{greeting_node_id}'.")
        state["new_chat_HEAD"] = greeting_node_id

# --------------------------------------------------------------------------------
# API

def load(llm_settings: env,
         datastore_file: Union[str, pathlib.Path],
         state_file: Union[str, pathlib.Path]) -> Tuple[chattree.Forest, Dict]:
    """Load chat app state.

    `llm_settings`: LLM client settings; this is the return value of `llmclient.setup`.

    `datastore_file`: Path to the JSON file to load the persistent chat forest from.
                      Will be auto-persisted to the same path at app exit.

    `state_file`: Path to the app state JSON file to load things such as the
                  new-chat HEAD, the current chat HEAD, and various settings.
                  Will be auto-persisted to the same path at app exit.

    Return value is the tuple `(datastore, state)`, where:
        `datastore`: `chattree.PersistentForest` containing the chat database,
        `state`: `dict` containing the app state (HEAD node, various persistent settings).

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
    mayberel_datastore_file = datastore_file
    mayberel_state_file = state_file
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
        logger.info(f"load: App state file '{mayberel_state_file}' (resolved to '{state_file}') does not exist.")
        state = {}
    else:
        logger.info(f"load: Loaded app state from '{mayberel_state_file}' (resolved to '{state_file}').")

    # Load datastore
    datastore = chattree.PersistentForest(datastore_file)  # This autoloads and auto-persists.
    with datastore.lock:
        if datastore.nodes:
            logger.info(f"load: Loaded chat datastore from '{mayberel_datastore_file}' (resolved to '{datastore_file}'). Found {len(datastore.nodes)} chat nodes in datastore.")
        else:
            logger.info("load: No chat nodes in datastore at '{mayberel_datastore_file}' (resolved to '{datastore_file}'). Creating new datastore, will be saved at app exit.")
            _reset_datastore_and_update_state(llm_settings, datastore, state)

    # Set any missing app state to defaults
    #
    if "tools_enabled" not in state:
        state["tools_enabled"] = True
        logger.info(f"load: Missing key 'tools_enabled' in '{mayberel_state_file}' (resolved to '{state_file}'), using default '{state['tools_enabled']}'")

    if "docs_enabled" not in state:
        state["docs_enabled"] = True
        logger.info(f"load: Missing key 'docs_enabled' in '{mayberel_state_file}' (resolved to '{state_file}'), using default '{state['docs_enabled']}'")

    if "speculate_enabled" not in state:
        state["speculate_enabled"] = False
        logger.info(f"load: Missing key 'speculate_enabled' in '{mayberel_state_file}' (resolved to '{state_file}'), using default '{state['speculate_enabled']}'")

    if "avatar_speech_enabled" not in state:
        state["avatar_speech_enabled"] = True
        logger.info(f"load: Missing key 'avatar_speech_enabled' in '{mayberel_state_file}' (resolved to '{state_file}'), using default '{state['avatar_speech_enabled']}'")

    if "avatar_subtitles_enabled" not in state:
        state["avatar_subtitles_enabled"] = True
        logger.info(f"load: Missing key 'avatar_subtitles' in '{mayberel_state_file}' (resolved to '{state_file}'), using default '{state['avatar_subtitles']}'")

    # Refresh the system prompt and AI greeting to the ones configured in `raven.librarian.config`.
    #
    #   - The system prompt node payload is overwritten by the new version.
    #     - A new revision of the payload is created, and the old revision is deleted.
    #     - Refreshing the system prompt also ensures that `state["system_prompt_node_id"]` is correct (and adds it to `state`, if missing).
    #   - The AI greeting either uses an existing node, or creates a new node.
    #     - If there is an existing AI greeting node (under the system prompt node) that matches the configured AI greeting text *for the current AI character*, that node is selected.
    #     - Otherwise a new node is created with the configured AI greeting text (and due to OAI-compatible chatlog format, starting the message content with the AI character name)
    #     - Refreshing the AI greeting sets `state["new_chat_HEAD"]` (always to a valid node, so we don't need to validate it here).
    #
    _refresh_system_prompt(llm_settings,
                           datastore,
                           state)
    _refresh_greeting(llm_settings,
                      datastore,
                      state)

    if "HEAD" not in state:  # Current chat node ID missing -> start at new chat
        state["HEAD"] = state["new_chat_HEAD"]
        logger.info(f"load: Missing key 'HEAD' in '{mayberel_state_file}' (resolved to '{state_file}'), resetting it to 'new_chat_HEAD'")

    if state["HEAD"] not in datastore.nodes:
        logger.info(f"load: Key 'HEAD' in '{mayberel_state_file}' (resolved to '{state_file}') points to nonexistent chat node '{state['HEAD']}', resetting it to 'new_chat_HEAD'")
        state["HEAD"] = state["new_chat_HEAD"]

    # Migrate datastore (this updates only if needed)
    # v0.2.3+: data format change
    chatutil.upgrade_datastore(llm_settings,
                               datastore,
                               system_prompt_node_id=state["system_prompt_node_id"])

    # Set up auto-persist for app state
    atexit.register(functools.partial(save,
                                      state_file=mayberel_state_file,
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
    required_keys = ("new_chat_HEAD",  # HEAD node for starting a new chat
                     "HEAD",  # current HEAD node (last message of currently open chat)
                     # various Raven-librarian GUI state
                     "tools_enabled",
                     "docs_enabled",
                     "speculate_enabled",
                     "avatar_speech_enabled",
                     "avatar_subtitles_enabled")
    if any(key not in state for key in required_keys):
        raise KeyError(f"At least one required setting is missing from `state`; required keys = {list(sorted(required_keys))}; got existing keys = {list(sorted(state.keys()))}")

    mayberel_state_file = state_file
    state_file = pathlib.Path(state_file).expanduser().resolve()

    with open(state_file, "w", encoding="utf-8") as json_file:
        json.dump(state, json_file, indent=2)

    logger.info(f"save: Saved app state to '{mayberel_state_file}' (resolved to '{state_file}').")
