"""Persistent chat app state.

Importantly, contains the HEAD node ID of the current chat, as well as some persistent option flags.

This module is shared between `minichat` (command-line app) and `app` (Raven-librarian GUI app).
"""

__all__ = ["sidecar_refs_in_payload",
           "load", "save",
           "refresh_system_prompt",
           "backfill_sidecar_metadata"]

import logging
logger = logging.getLogger(__name__)

import atexit
import functools
import json
import pathlib
from typing import Dict, Tuple, Union

from unpythonic.env import env

from . import chattree
from . import chatutil
from . import imagestore
from . import sidecarstore
from . import textfilestore

# Default values for the persistent per-app state flags (the toggles that the Librarian apps
# expose to the user). `load` uses these to fill any missing keys from an on-disk state file;
# `save` uses the keys to validate that the state dict has all required flags. Adding or removing
# a flag means touching this one mapping — `load`, `save`, and the tests derive from it.
_DEFAULT_FLAGS = {"internet_enabled": True,
                  "docs_enabled": True,
                  "avatar_speech_enabled": True,
                  "avatar_subtitles_enabled": True}

# Flags that used to exist, dropped from a state file on load so they do not sit there forever confusing
# whoever reads it next. Removable once no state file in the wild still carries them.
_RETIRED_FLAGS = ("speculate_enabled",)

# Flags that were renamed, old name -> new name. The stored *value* carries across, which is the whole
# difference from retiring one: the user's setting survives the rename. `tools_enabled` governed every tool
# wholesale; `internet_enabled` governs the network-reaching ones, so a user who had tools off gets the
# network off, which is the reading that keeps their intent.
#
# Applied before the defaults are filled in — the other order would see the new name missing, write the
# default over it, and silently discard the setting being migrated.
_RENAMED_FLAGS = {"tools_enabled": "internet_enabled"}

# What the chat datastore was called before 0.2.9, when the default became `chat.json`. `load` adopts a file
# by this name if the configured one is absent, so an existing chat history is not left behind by an upgrade.
_LEGACY_DATASTORE_FILENAME = "data.json"


def _looks_like_a_chat_datastore(path: pathlib.Path) -> bool:
    """Whether `path` holds a `chattree` forest, judged by reading it rather than by its name.

    The adoption in `load` is the reason this exists, and the name is why it cannot go by the name.
    `data.json` is generic enough to belong to anything, and the file is looked for beside the *configured*
    datastore — which a user may have pointed at a directory of their own. Renaming a stranger's `data.json`
    into Raven's chat history would take a file away from whatever actually owned it, and the symptom would
    be that other program's, not Raven's.

    A forest on disk is a flat mapping of node ID to node, so this checks the shape: an object whose every
    value is an object carrying a node's own bookkeeping keys. An empty object qualifies — that is an empty
    forest, which is exactly what a fresh datastore holds.
    """
    try:
        with open(path, "r", encoding="utf-8") as json_file:
            content = json.load(json_file)
    except (OSError, ValueError):  # unreadable, or not JSON at all
        return False
    if not isinstance(content, dict):
        return False
    return all(isinstance(node, dict) and "id" in node and "data" in node
               for node in content.values())

# --------------------------------------------------------------------------------
# Sidecar GC configuration

def sidecar_refs_in_payload(payload: dict) -> set[str]:
    """Return every sidecar filename referenced by one node `payload` — the GC mark interpreter for the apps.

    The union of the two per-kind interpreters (`imagestore` for attached images, `textfilestore` for attached
    documents), which is what a datastore holding both kinds needs. `chattree` drives the revision traversal and
    calls this to read the references out of each payload, because payloads are opaque to it by design.

    The union matters more than it looks. The mark phase deletes whatever it does not mark, so a datastore
    configured with only one of the two interpreters would sweep away every sidecar of the other kind on the
    first cleanup — silently, since the files are content-addressed and nothing else names them. Having one
    function that both apps get by default is what keeps that from being an easy mistake to make.
    """
    return imagestore.sidecar_refs_in_payload(payload) | textfilestore.sidecar_refs_in_payload(payload)


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

def refresh_system_prompt(llm_settings: env,
                          datastore: chattree.Forest,
                          state: Dict) -> None:
    """Refresh the system prompt in the datastore (to the one currently produced by `llmclient`).

    A new revision is created on the system prompt node, and the previous revision is deleted.

    `load` calls this, and so does a caller that has just changed what `llm_settings` says — notably
    `llmclient.reconnect`, after which the stored card names a model that is no longer the one loaded, and
    states a context length that was a default standing in for one.

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
          The app state loader calls `refresh_system_prompt` first, ensuring proper initialization.

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
        greeting_message_content = chatutil.content_to_text(greeting_message["content"])

        for greeting_node_id in greeting_node_ids:
            payload = datastore.get_payload(greeting_node_id)  # get currently active revision
            message = payload["message"]
            message_role = message["role"]
            message_text = chatutil.content_to_text(message["content"])
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
         state_file: Union[str, pathlib.Path],
         autosave: bool = True) -> Tuple[chattree.Forest, Dict]:
    """Load chat app state.

    `llm_settings`: LLM client settings; this is the return value of `llmclient.setup`.

    `datastore_file`: Path to the JSON file to load the persistent chat forest from.
                      Will be auto-persisted to the same path at app exit.

    `state_file`: Path to the app state JSON file to load things such as the
                  new-chat HEAD, the current chat HEAD, and various settings.
                  Will be auto-persisted to the same path at app exit.

    `autosave`: If `True` (default), register automatic `atexit` persistence for both the datastore
                and the app state file. This is the right behaviour for app lifecycle use.

                If `False`, skip both `atexit` registrations. The caller is responsible for calling
                `save(state_file, state)` and `datastore.save()` explicitly if persistence is wanted.
                Primarily useful for tests and ad-hoc inspection.

    Return value is the tuple `(datastore, state)`, where:
        `datastore`: `chattree.PersistentForest` containing the chat database,
        `state`: `dict` containing the app state (HEAD node, various persistent settings).

    NOTE: Object identity is important - for the state auto-persist (at app exit)
          to work correctly, you should modify the original `state` dict in-place;
          that object is what gets auto-persisted at exit.

    NOTE: Recovery procedure for corrupted state:

      - Empty datastore (no nodes) -> factory reset.
      - Missing state file -> empty dict, but datastore is preserved. Missing keys get defaults.
      - Dangling HEAD (points to nonexistent node) -> reset to new_chat_HEAD, so that when the
                                                      app opens, it opens into a new chat session.
      - Missing HEAD -> set to new_chat_HEAD.

    new_chat_HEAD is always computed at startup.

    If any settings are missing in the state file (e.g. state file from an older version
    of Librarian that doesn't yet have a particular setting), they are initialized to their
    default values (which are defined in the source code of this function).
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

    # Adopt a pre-0.2.9 datastore, which was called `data.json` before the default was renamed to say what
    # is in it. Only when the configured file is absent, so this never overwrites a datastore, and it takes
    # the sidecar directory along — moving the JSON alone would leave every attachment behind, still named
    # by payloads that could no longer find it.
    if not datastore_file.exists():
        legacy_datastore_file = datastore_file.parent / _LEGACY_DATASTORE_FILENAME
        if legacy_datastore_file.is_file() and not _looks_like_a_chat_datastore(legacy_datastore_file):
            logger.info(f"load: '{legacy_datastore_file}' is not a chat datastore, leaving it alone.")
            legacy_datastore_file = datastore_file  # `rename_datastore` no-ops on this, so nothing moves
        try:
            if chattree.rename_datastore(legacy_datastore_file, datastore_file):
                logger.info(f"load: Adopted pre-0.2.9 datastore '{legacy_datastore_file}' as '{datastore_file}'.")
        except OSError as exc:
            # Not fatal: `rename_datastore` rolls back before raising, so what is on disk is the layout we
            # started with, and the app opening on an empty chat leaves it that way. Crashing here would
            # not improve on that, and it would make a failed *rename* look like lost history.
            logger.error(f"load: Could not adopt pre-0.2.9 datastore '{legacy_datastore_file}' as '{datastore_file}'; "
                         f"starting from an empty chat instead. Renaming it by hand fixes this — take its sidecar "
                         f"directory along. Reason {type(exc)}: {exc}")

    # Load datastore
    datastore = chattree.PersistentForest(datastore_file, autosave=autosave,  # This autoloads; auto-persists iff autosave.
                                          sidecar_extractor=sidecar_refs_in_payload)
    with datastore.lock:
        if datastore.nodes:
            logger.info(f"load: Loaded chat datastore from '{mayberel_datastore_file}' (resolved to '{datastore_file}'). Found {len(datastore.nodes)} chat nodes in datastore.")
        else:
            logger.info("load: No chat nodes in datastore at '{mayberel_datastore_file}' (resolved to '{datastore_file}'). Creating new datastore, will be saved at app exit.")
            _reset_datastore_and_update_state(llm_settings, datastore, state)

    # Carry renamed flags over before the defaults are filled in (see `_RENAMED_FLAGS` for why the order
    # matters). An old name whose new name is already present is dropped rather than applied: the new one
    # is what the app has been writing, so it is the current setting.
    for old_key, new_key in _RENAMED_FLAGS.items():
        if old_key in state:
            old_value = state.pop(old_key)
            if new_key in state:
                logger.info(f"load: Dropping renamed key '{old_key}' from '{mayberel_state_file}' (resolved to '{state_file}'); '{new_key}' is already present")
            else:
                state[new_key] = old_value
                logger.info(f"load: Renaming key '{old_key}' -> '{new_key}' in '{mayberel_state_file}' (resolved to '{state_file}'), keeping value '{old_value}'")

    # Set any missing app state flags to their defaults.
    for key, default in _DEFAULT_FLAGS.items():
        if key not in state:
            state[key] = default
            logger.info(f"load: Missing key '{key}' in '{mayberel_state_file}' (resolved to '{state_file}'), using default '{default}'")

    for key in _RETIRED_FLAGS:
        if state.pop(key, None) is not None:
            logger.info(f"load: Dropping retired key '{key}' from '{mayberel_state_file}' (resolved to '{state_file}')")

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
    refresh_system_prompt(llm_settings,
                          datastore,
                          state)

    # Migrate the datastore to the current format BEFORE anything reads message content. `refresh_system_prompt`
    # above only overwrites the system prompt node (it does not read content), and it sets
    # `state["system_prompt_node_id"]`, which `upgrade_datastore` needs. `_refresh_greeting` below, however,
    # compares stored greeting content via `chatutil.content_to_text`, which assumes the content-parts format;
    # a legacy datastore stores `content` as a bare string, so the migration must run first or that comparison
    # crashes on the un-migrated data. (This updates only if needed.)
    # v0.2.3+: data format change
    chatutil.upgrade_datastore(llm_settings,
                               datastore,
                               system_prompt_node_id=state["system_prompt_node_id"])

    _refresh_greeting(llm_settings,
                      datastore,
                      state)

    if "HEAD" not in state:  # Current chat node ID missing -> start at new chat
        state["HEAD"] = state["new_chat_HEAD"]
        logger.info(f"load: Missing key 'HEAD' in '{mayberel_state_file}' (resolved to '{state_file}'), resetting it to 'new_chat_HEAD'")

    if state["HEAD"] not in datastore.nodes:
        logger.info(f"load: Key 'HEAD' in '{mayberel_state_file}' (resolved to '{state_file}') points to nonexistent chat node '{state['HEAD']}', resetting it to 'new_chat_HEAD'")
        state["HEAD"] = state["new_chat_HEAD"]

    # Recover descriptions for sidecars stored before they were written beside the file. Runs at load rather
    # than at cleanup time because the payloads are the only place the names exist, and a node deletion takes
    # them with it — by the time a cleanup runs, anything already orphaned is past recovering.
    backfill_sidecar_metadata(datastore)

    # Set up auto-persist for app state
    if autosave:
        atexit.register(functools.partial(save,
                                          state_file=mayberel_state_file,
                                          state=state))

    return datastore, state

def backfill_sidecar_metadata(datastore: chattree.PersistentForest) -> int:
    """Write a description beside every referenced sidecar that lacks one. Return how many were written.

    A migration for datastores predating sidecar metadata. Those still carry the provenance in the payloads
    that reference each sidecar, so the name, source and timestamp can be copied out to where they survive the
    referencing node — which is what a cleanup preview needs, since an orphan has no payload left to ask.

    Only reaches sidecars that are still referenced. Anything orphaned by a deletion that already happened
    stays anonymous: the payload naming it went with the node, and nothing else ever recorded it.

    Idempotent, and safe to run at every load — `maybe_set_sidecar_metadata` is first-write-wins, so an existing
    description is never overwritten by a later attachment of the same bytes under a different name.
    """
    written = 0
    with datastore.lock:
        for node in datastore.nodes.values():
            for payload in node.get("data", {}).values():  # every revision: an older one may name a sidecar the current one dropped
                for filename, provenance in sidecarstore.provenance_entries_in_payload(payload).items():
                    try:
                        if datastore.has_sidecar(filename) and datastore.maybe_set_sidecar_metadata(filename, provenance):
                            written += 1
                    except ValueError:  # unsafe filename from a corrupt datastore; `has_sidecar` refuses it
                        logger.warning(f"backfill_sidecar_metadata: skipping unsafe sidecar filename '{filename}'.")
    if written:
        plural_s = "s" if written != 1 else ""
        logger.info(f"backfill_sidecar_metadata: recovered description{plural_s} for {written} sidecar{plural_s}.")
    return written

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
                     "HEAD") + tuple(_DEFAULT_FLAGS.keys())  # current HEAD + per-app GUI flags
    if any(key not in state for key in required_keys):
        raise KeyError(f"At least one required setting is missing from `state`; required keys = {list(sorted(required_keys))}; got existing keys = {list(sorted(state.keys()))}")

    mayberel_state_file = state_file
    state_file = pathlib.Path(state_file).expanduser().resolve()

    with open(state_file, "w", encoding="utf-8") as json_file:
        json.dump(state, json_file, indent=2)

    logger.info(f"save: Saved app state to '{mayberel_state_file}' (resolved to '{state_file}').")
