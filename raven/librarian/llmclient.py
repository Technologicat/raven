"""LLM client low-level library functions for Raven.

See `raven.librarian.scaffold` for the higher-level scaffolding that goes on top of this,
e.g. automatically applying tool-calls.

For an example chat client built using these, see `raven.librarian.minichat`.

NOTE for oobabooga/text-generation-webui users:

If you want to see the final prompt in instruct or chat mode, start your server in `--verbose` mode.
"""

__all__ = ["TOOLS", "TOOL_ENTRYPOINTS", "DOCUMENT_TOOL_NAMES", "NETWORK_TOOL_NAMES",

           "list_models",
           "test_connection",
           "detect_backend_flavor",
           "setup",
           "configure",

           # For frontends that open a window whether or not a backend answers
           "connect", "reconnect", "backend_status", "describe_backend_status",
           "backend_unreachable", "backend_has_no_model", "backend_ready",

           "count_tokens",
           "image_token_cost",
           "count_branch_tokens", "prompt_size_report_looks_whole",

           "budget_for_fetched_text",
           "truncate_middle",
           "fit_text_to_token_budget",
           "fit_attachments_to_context",

           "StreamParser",
           # For scripting: the wire form of what a turn would send
           "serialize_history_for_wire",
           "invoke", "prefill", "action_ack", "action_stop",
           "make_console_progress_handler",
           "perform_tool_calls",
           "approve_host_for_session"]

import logging
logger = logging.getLogger(__name__)

import collections
import copy
import io
import json
import os
import pathlib
import requests
import sys
import threading
import time
from typing import Any, Callable, Collection, Dict, List, Optional, Tuple

import sseclient  # pip install sseclient-py

from mcpyrate import colorizer

from unpythonic import si_prefix, sym, timer
from unpythonic.env import env


from . import chattree
from . import chatutil
from . import config as librarian_config
# The tool subsystem lives in its own module; re-exported here because `llmclient.TOOLS` and
# `llmclient.perform_tool_calls` are what `scaffold`, `agent`, `chat_controller` and the tests have
# always called, and where the tools live is not their business.
#
# Listed rather than star-imported: this is an ordinary module, not a package `__init__`, so a `*` here
# buys nothing that a list does not and costs every IDE and static checker the ability to resolve these
# names. The cost of the list is that a new public name in `llmtools` has to be added here too, which is
# the same discipline `__all__` already asks for.
from .llmtools import (TOOLS, TOOL_ENTRYPOINTS,  # noqa: F401 -- re-export
                       DOCUMENT_TOOL_NAMES, NETWORK_TOOL_NAMES, EXTERNAL_SOURCE_TOOL_NAMES,
                       maybe_tool_names_for_turn,
                       perform_tool_calls, approve_host_for_session,

                       CANONICAL_NOT_ON_ALLOWLIST,
                       CANONICAL_NO_DOCUMENT_DATABASE, CANONICAL_NO_DOCUMENT_MATCHES,
                       CANONICAL_NO_SUCH_DOCUMENT, CANONICAL_NOTHING_CONSULTED,
                       CANONICAL_NO_ROOM_TO_FETCH, CANONICAL_BAD_EXPRESSION,

                       document_text, document_path, label_documents,

                       websearch, webfetch, search_documents, fetch_document,
                       get_current_time, calculate, list_consulted_documents)
from . import gguftokenizer
from . import textfilestore
from . import imagestore
from . import sidecarstore

action_ack = sym("ack")  # acknowledge LLM progress, keep generating
action_stop = sym("stop")  # interrupt the LLM, stop generating now

# Canonical identity string injected into the character card when the loaded model can't be determined.
# The card asserts the model's identity as a fact, so saying "unknown" is correct; guessing would make the
# assistant broadcast something false if a user asks "which model are you?".
NO_MODEL_INFO = "No model information is available"

# --------------------------------------------------------------------------------
# Talking to raven-server
#
# `raven.client.api` is imported where it is used rather than at module top, and there are exactly two such
# places: the `websearch` and `webfetch` tool wrappers below. Importing it pulls `spacy` and — via the
# vendored Kokoro streaming writer — `av`, so a module-top import made *importing* `llmclient`, and
# therefore `scaffold`, and therefore the whole agent layer, require the full client dependency stack. A
# probe that must boot a TTS stack to test tool-calling is a probe nobody writes.
#
# Note what is deliberately *not* on this list: `setup`. It reaches the LLM backend over plain `requests`
# and never touches raven-server, so a headless run that calls `setup` and then generates without invoking
# a network tool never loads any of this. That is the case the scripting surface is for, so it is the case
# worth keeping cheap.


# ----------------------------------------
# LLM communication setup

# HTTP headers for LLM requests
headers = {
    "Content-Type": "application/json"
}

# Read API key for cloud LLM support
if os.path.exists(librarian_config.llm_api_key_file):  # TODO: test this (implemented according to spec)
    with open(librarian_config.llm_api_key_file, "r", encoding="utf-8") as f:
        api_key = f.read().replace('\n', '')
    # "Authorization": "Bearer yourPassword123"
    # https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API
    headers["Authorization"] = api_key.strip()

# --------------------------------------------------------------------------------
# Websearch integration (requires `raven.server` to be running)


# --------------------------------------------------------------------------------
# Webfetch integration (requires `raven.server` to be running)


# --------------------------------------------------------------------------------
# Document database integration (the local knowledge base; no server needed)


# --------------------------------------------------------------------------------
# Utilities

def list_models(backend_url: str) -> List[str]:
    """List the model ids available at `backend_url`, via the standard OpenAI `/v1/models` endpoint.

    Used for a model picker and the connection probe (order irrelevant). For the *loaded* model's identity,
    see `_resolve_model_info` instead — this list can't tell you which model is actually loaded on LM Studio
    under just-in-time loading.
    """
    response = requests.get(f"{backend_url}/v1/models",
                            headers=headers,
                            verify=False,
                            timeout=librarian_config.llm_network_timeout)
    payload = response.json()
    ids = [model["id"] for model in payload.get("data", []) if model.get("id")]
    return sorted(ids, key=lambda s: s.lower())

def test_connection(backend_url: str,
                    quiet: bool = False) -> bool:
    """Test the connection to the LLM backend.

    Return `True` if test successful, `False` if not (e.g. server not running or unreachable).

    `quiet`: If `False` (default), print test result to stdout.
             If `True`, don't print anything (like `-q` command-line option of many *nix tools).
    """
    try:
        list_models(backend_url)  # just do something, to try to connect
    except requests.exceptions.ConnectionError as exc:
        if not quiet:
            print(colorizer.colorize(f"Cannot connect to LLM backend at {backend_url}.",
                                     colorizer.Style.BRIGHT, colorizer.Fore.RED) + " Is the LLM server running?")
        msg = f"Failed to connect to LLM backend at {backend_url}, reason {type(exc)}: {exc}"
        logger.error(msg)
        return False
    else:
        if not quiet:
            print(colorizer.colorize(f"Connected to LLM backend at {backend_url}", colorizer.Style.BRIGHT, colorizer.Fore.GREEN))
        return True

def detect_backend_flavor(backend_url: str) -> str:
    """Probe `backend_url` to determine which OpenAI-compatible backend it is.

    Returns "lmstudio", "oobabooga", or "generic". Detection is by *payload shape*, not HTTP status: LM
    Studio answers unknown endpoints with HTTP 200 and an `{"error": ...}` body, so a status check would
    misfire. The probe *order* is load-bearing — the LM-Studio-native endpoint is tried first, because the
    ooba-private endpoint is not a clean discriminator (LM Studio returns 200 for it too, just without the
    expected field).
    """
    # LM Studio: the native `/api/v0/models` returns {"data": [{id, state, arch, loaded_context_length, ...}]}.
    # No other backend serves this namespace.
    try:
        models = requests.get(f"{backend_url}/api/v0/models", headers=headers, verify=False, timeout=librarian_config.llm_network_timeout).json().get("data")
        if isinstance(models, list) and models and "state" in models[0]:
            return "lmstudio"
    except (requests.RequestException, ValueError, AttributeError):  # connection / non-JSON / unexpected shape -> not LM Studio
        pass
    # oobabooga: the private `/v1/internal/model/info` returns {"model_name": ...}. Check the field, not the
    # status — LM Studio returns 200 here too, but with {"error": ...} and no `model_name`.
    try:
        if "model_name" in requests.get(f"{backend_url}/v1/internal/model/info", headers=headers, verify=False, timeout=librarian_config.llm_network_timeout).json():
            return "oobabooga"
    except (requests.RequestException, ValueError, AttributeError):
        pass
    return "generic"

def _format_lmstudio_model_label(model_record: Dict) -> str:
    """Assemble a rich identity line from an LM Studio `/api/v0/models` record.

    E.g. `qwen3.5-4b, Q4_K_XL, 128 Ki context` — accurate, structured, better than a bare GGUF filename.
    The context length uses an IEC binary prefix (`si_prefix` with `binary=True`), since model context
    windows are powers of two (131072 -> "128 Ki", exactly).
    """
    parts = [model_record["id"]]
    if model_record.get("quantization"):
        parts.append(model_record["quantization"])
    ctx = model_record.get("loaded_context_length")
    if ctx:
        parts.append(f"{si_prefix(ctx, precision=0, binary=True)} context")
    return ", ".join(parts)

def _resolve_model_info(backend_url: str, flavor: str) -> env:
    """Resolve the loaded model's identity and context window for `flavor`.

    Returns an `env` with:
      `label`: human-facing model identity for the character card. The card asserts this as a *fact* about
               the model's own identity, so a wrong value is worse than none — when a generic backend can't
               disambiguate the loaded model, this is the literal string "No model information is available"
               rather than a guess.
      `model_id`: the model id to send in requests (relevant for LM Studio JIT), or `None`.
      `context_length`: the loaded context window in tokens, or `None` if the backend doesn't report it.
      `is_vlm`: whether the loaded model accepts image input, as a tri-state — `True` / `False` when the backend
                reports it (LM Studio flags this via the model record's `type == "vlm"`), or `None` when it
                can't be determined (ooba / generic expose no capability field). Gates the image-attach UI: a
                definite `False` hard-refuses attachment; `None` allows it and lets the backend reject.
      `loaded`: whether the backend currently has a model resident and ready to answer, as the same tri-state.
                LM Studio says so per model (`state`); ooba reports the string "None" as its model name when
                it has nothing. Both are therefore `True`/`False`. A generic backend lists models it *has*,
                which says nothing about what is resident, so there it is `None` — "cannot tell", which must
                not be shown to the user as a fault. Reachable-but-empty is otherwise invisible until the
                first message of a session fails, which is the worst moment to discover it.
    """
    if flavor == "oobabooga":
        # ooba reports the GGUF filename (fine — the model can interpret `name-size-quant.gguf` itself) but
        # not the active context length here; the latter falls through to the default in `setup`. It exposes no
        # VLM-capability flag either, so `is_vlm` is unknown (`None`).
        model_name = requests.get(f"{backend_url}/v1/internal/model/info", headers=headers, verify=False, timeout=librarian_config.llm_network_timeout).json().get("model_name")
        # ooba reports "nothing loaded" as the *string* "None" in this field, which is the check its own
        # `list_models_openai_format` makes before deciding whether it has a model to list at all. So the
        # test below is ooba's test, not a guess at one - but it is read from its source rather than
        # observed, since there is no ooba instance here to try it against.
        model_is_loaded = bool(model_name) and model_name != "None"
        return env(label=model_name if model_is_loaded else NO_MODEL_INFO,
                   model_id=model_name if model_is_loaded else None,
                   context_length=None,
                   is_vlm=None,
                   loaded=model_is_loaded)
    if flavor == "lmstudio":
        # `/api/v0/models` lists all downloaded models; exactly the `state == "loaded"` one is resident under
        # JIT, and only that record carries `loaded_context_length`. The record's `type` field is `"vlm"` for
        # vision models (vs `"llm"` / `"embeddings"`); vision is signaled there, not in `capabilities`.
        models = requests.get(f"{backend_url}/api/v0/models", headers=headers, verify=False, timeout=librarian_config.llm_network_timeout).json().get("data", [])
        loaded = [m for m in models if m.get("state") == "loaded"]
        if loaded:
            record = loaded[0]
            # A record with no `type` at all is "cannot tell", not "cannot see": a bare `== "vlm"` would
            # hard-refuse image attachment on the strength of a field that was never there. Every record
            # this LM Studio returns carries one, so this is the shape of the answer rather than a
            # workaround for an observed gap.
            maybe_model_type = record.get("type")
            return env(label=_format_lmstudio_model_label(record),
                       model_id=record.get("id"),
                       context_length=record.get("loaded_context_length"),
                       is_vlm=(maybe_model_type == "vlm") if maybe_model_type is not None else None,
                       loaded=True)
        # JIT idle: nothing resident right now. If the user named a model, trust that; else say so honestly.
        # Nothing loaded means no capability record to read, so `is_vlm` is unknown either way.
        #
        # `loaded=False` even when the user named a model: naming one says which model a request would ask
        # for, not that the backend can answer right now. JIT does load on demand, so this state often
        # resolves itself - but it also fails outright when the named model does not fit in what is free,
        # and telling the user which of those they are in is the whole point of reporting it.
        if librarian_config.llm_model:
            return env(label=librarian_config.llm_model, model_id=librarian_config.llm_model, context_length=None, is_vlm=None, loaded=False)
        return env(label=NO_MODEL_INFO, model_id=None, context_length=None, is_vlm=None, loaded=False)
    # generic: best-effort from the standard list; never guess identity, and no capability field to read.
    ids = [m.get("id") for m in requests.get(f"{backend_url}/v1/models", headers=headers, verify=False, timeout=librarian_config.llm_network_timeout).json().get("data", [])]
    ids = [model_id for model_id in ids if model_id]
    if len(ids) == 1:
        return env(label=ids[0], model_id=ids[0], context_length=None, is_vlm=None, loaded=None)
    return env(label=NO_MODEL_INFO, model_id=librarian_config.llm_model, context_length=None, is_vlm=None, loaded=None)

# --------------------------------------------------------------------------------
# The tool registry
#
# Module level rather than built inside `setup`, because none of it depends on anything `setup`
# fetches - it is three literals - while `setup` itself cannot run without a live backend to ask
# for the model name, the tokenizer and the sampler defaults. Trapped inside it, the registry was
# unavailable to anything that could not open a connection, so the tests kept a hand-copy and the
# two were free to drift.
#
# Read, never mutated, so one shared instance across sessions is safe.
# --------------------------------------------------------------------------------


def setup(backend_url: str,
          quiet: bool = False) -> env:
    """Connect to LLM at `backend_url`.

    `quiet`: If `False` (default), print authentication reminder to stdout.
             If `True`, don't print anything (like `-q` command-line option of many *nix tools).

    Return an `unpythonic.env.env` object (a fancy namespace) populated with the following fields:

        `user: str`: User persona (name of user's character).

        `char: str`: AI persona name (name of the AI's character).

        `model: str`: Human-facing identity of the loaded model, for the character card — a rich line on
                      LM Studio (id, quant, context), the GGUF filename on ooba, or "No model information is
                      available" when a generic backend can't disambiguate (never a guess). See `_resolve_model_info`.

        `model_id: Optional[str]`: The model id sent in each request's `model` field (LM Studio JIT loads it on
                                   demand), or `None`. Distinct from `model`, which is the display identity.

        `backend_flavor: str`: Which OpenAI-compatible backend this is — "oobabooga", "lmstudio", or "generic".
                               Autodetected (or forced via `config.llm_backend_flavor`); gates a few request details.

        `context_length: int`: The loaded context window in tokens — backend-reported where available, else a
                               conservative 64k default (a warning is logged when defaulted).

        `model_is_vlm: Optional[bool]`: Whether the loaded model accepts image input, as a tri-state — `True` /
                                        `False` when the backend reports it (LM Studio, via the model record's
                                        `type == "vlm"`), or `None` when it can't be determined (ooba / generic).
                                        The image-attach UI gates on this: a definite `False` refuses attachment
                                        with a clear message; `None` allows it and lets the backend reject.

        `model_is_loaded: Optional[bool]`: Whether the backend has a model resident and ready to answer, as the
                                           same tri-state — `True` / `False` on LM Studio and ooba, `None` on a
                                           generic backend, whose model list says what it has rather than what
                                           is running. The frontends show a definite `False`; `None` is not a
                                           fault and is not shown.

        `backend_is_reachable: bool`: Whether anything answered at `backend_url` when these settings were
                                      built. Always `True` from `setup`, which cannot return otherwise;
                                      `False` from `connect` when the backend was down. Read it through
                                      `backend_status`, which folds it together with `model_is_loaded`.

        `backend_supports_continue: bool`: Whether the backend supports continuing an existing assistant message
                                           (ooba does, via an explicit flag; lmstudio/generic don't).

        `system_prompt: str`: Currently empty. Used to be a generic system prompt for the LLM (the LLaMA 3 preset from SillyTavern), to make it follow the character card.

        `character_card: str`: Who the AI is: the assistant character's identity and manner. It shapes what
                               the model does as well as how it sounds — a character card elicits a persona,
                               propensities included — so it is part of the answer rather than a coat of
                               paint on one. That is why withholding it is offered as a choice
                               (`chatutil.create_initial_system_message`) rather than assumed either way.

        `user_card: str`: Who the user is and how they prefer to be communicated with. Empty unless the
                          deployment fills it in. Travels with the character card — both describe the two
                          ends of a conversation, so a call made without the character is made without this
                          too; see `chatutil.create_initial_system_message`.

        `stopping_strings: List[str]`: List of strings that automatically interrupt the AI in `invoke`.
                                       The default is `[f"\n{user}:"]`, which prevents old models' habit of speaking on the user's behalf.

                                       NOTE: Tool calls will not be processed if a stopping string is hit.

        `greeting: str`: The AI's first message, used for starting a new chat.

        `tools: List[Dict[str, Any]]`: JSON specifications of available tools (for LLMs capable of tool-calling).

        `tool_entrypoints: Dict[str, Callable]`: The Python functions that implement the tools.

        `document_tool_names: FrozenSet[str]`: Those tool names that search or read the document database, and so
                                               should only be offered on turns where it is in play. Callers pass a
                                               filtered name set to `invoke`'s `tool_names`;
                                               `maybe_tool_names_for_turn` builds it. Every tool is *registered*
                                               regardless — availability is a per-turn question, not a
                                               per-session one.

        `network_tool_names: FrozenSet[str]`: The same, for the tools that reach the network, gated on the
                                              user's "Internet" switch. A tool in neither set answers to no
                                              switch and is always offered.

        `backend_url: str`: The `backend_url` argument, as-is.

        `request_data: Dict[str, Any]`: Generation settings for the LLM backend.

        `personas: Dict[str, Optional[str]]`: Persona (character name) for each of the roles (dict keys) "user", "assistant", "system", and "tool".
                                              Used for constructing chat messages (see `raven.librarian.chatutil.create_chat_message`).

                                              The "system" and "tool" roles typically have no persona; for them, the persona is stored as `None`.
    """
    # Identify the backend, then resolve the loaded model's identity and context window for the character card.
    # A few request/response details differ between backends (see `detect_backend_flavor`, `_resolve_model_info`).
    #
    # These two calls are the whole of what `setup` needs a live backend for; everything after them is
    # `configure`, which is pure. That is the split, and it is why it is drawn here.
    backend_flavor = librarian_config.llm_backend_flavor or detect_backend_flavor(backend_url)
    model_info = _resolve_model_info(backend_url, backend_flavor)
    return configure(model_info=model_info,
                     backend_flavor=backend_flavor,
                     backend_url=backend_url,
                     quiet=quiet)


# What a backend that cannot be reached has told us about itself: nothing.
_UNREACHABLE_MODEL_INFO = env(label=NO_MODEL_INFO, model_id=None, context_length=None, is_vlm=None, loaded=False)

backend_unreachable = sym("backend_unreachable")
backend_has_no_model = sym("backend_has_no_model")
backend_ready = sym("backend_ready")

def connect(backend_url: str, quiet: bool = False) -> env:
    """`setup`, but a backend that cannot be reached yields settings instead of an exception.

    For the interactive frontends, which open a window either way: past chats, the cleanup dialog and the
    settings are all useful with no model in sight, and a user who started the LLM server second is one
    click from fixing it. Batch tools want the opposite and should keep calling `setup` — failing at
    document 1 with a precise diagnosis beats discovering it at document 2400.

    The settings come back fully formed and usable; what they cannot contain is anything only the backend
    knows, so the character card names no model and states the default context length. `backend_status`
    reports which of the three states this is, and `reconnect` replaces the placeholders once there is
    something to ask.

    Unless `quiet`, the verdict is reported for all three states — to the console in the same words and with
    the same advice a frontend's status readout gives, so a user who has a terminal in view and a user who
    has not are told the same thing, and to the log, so that a session can be diagnosed after the fact from
    a `--log` file that has no console attached.

    `quiet` silences **both** channels, which is a wider meaning than "don't print" and is what its one user
    needs: the only caller that passes it is `reconnect` under a frontend's poll, which asks this question
    every few seconds for as long as the answer is bad. Logging the verdict there would write the same line
    a few hundred times an hour, all of it describing one condition. A frontend that polls logs the
    *transitions* instead.
    """
    try:
        settings = setup(backend_url, quiet=quiet)
    except requests.exceptions.RequestException as exc:
        if not quiet:
            headline, advice = describe_backend_status(backend_unreachable, backend_url)
            logger.warning(f"connect: {headline} Continuing without one. Reason {type(exc)}: {exc}")
            print(colorizer.colorize(headline, colorizer.Style.BRIGHT, colorizer.Fore.RED) + f" {advice}")
        return configure(model_info=_UNREACHABLE_MODEL_INFO,
                         backend_flavor=librarian_config.llm_backend_flavor or "generic",
                         backend_url=backend_url,
                         quiet=quiet,
                         backend_is_reachable=False)
    if not quiet:
        status = backend_status(settings)
        headline, advice = describe_backend_status(status, backend_url)
        if status is backend_has_no_model:
            logger.warning(f"connect: {headline}")
            print(colorizer.colorize(headline, colorizer.Style.BRIGHT, colorizer.Fore.YELLOW) + f" {advice}")
        else:
            logger.info(f"connect: {headline} Model is '{settings.model}'.")
            print(colorizer.colorize(headline, colorizer.Style.BRIGHT, colorizer.Fore.GREEN))
    return settings

def describe_backend_status(status: sym, backend_url: str) -> Tuple[str, str]:
    """Return `(headline, advice)` for `status` — what is true, and what the user can do about it.

    One source for the wording every frontend needs: the console verdict `connect` prints, the message a
    batch tool dies with, and a GUI's tooltip. A message like this goes stale without looking stale — a copy
    still naming an address the app no longer reads reads exactly like a correct one — so it is written once.

    Split rather than joined into one sentence because the two halves are wanted in different places: a
    narrow GUI row can show a short label of its own and put both of these in the tooltip, while a console
    prints them together. `advice` is empty where there is nothing to do.

    Not `str`-formatted with color: whether to colorize, and in which color, belongs to the frontend.
    """
    if status is backend_unreachable:
        return (f"Cannot connect to the LLM backend at {backend_url}.",
                "Is the LLM server running, and is that the right address?")
    if status is backend_has_no_model:
        return (f"The LLM backend at {backend_url} has no model loaded.",
                "Load one in your LLM server.")
    return (f"Connected to the LLM backend at {backend_url}.", "")

def backend_status(settings: env) -> sym:
    """Return which of the three states `settings` describes.

    `backend_unreachable`: nothing answered at the URL. Is the server running, is the URL right?

    `backend_has_no_model`: the backend answered, and has nothing resident to answer *with*. Load a model.

    `backend_ready`: as far as can be told, a turn would work. Includes the backends that do not report
                     whether a model is loaded, since "cannot tell" is not a fault to report — see
                     `_resolve_model_info`.

    The three are worth distinguishing rather than collapsing into "not working": the user meets all of them
    at the same moment, having done nothing wrong, and what they should do about it differs.
    """
    if not settings.backend_is_reachable:
        return backend_unreachable
    if settings.model_is_loaded is False:
        return backend_has_no_model
    return backend_ready

def reconnect(settings: env, quiet: bool = True) -> sym:
    """Re-probe the backend and bring `settings` up to date, returning the new `backend_status`.

    Mutating rather than returning a new object, because every consumer — the chat controller, the app
    state, whatever a script is holding — already has this one, and handing back a replacement would leave
    them all on the old.

    What changes is everything the backend has a say in: which model is loaded, its context window, and
    whether it can see images. Nothing stored has to be repaired afterwards, because none of that is written
    into the stored system prompt — a card's text comes from the configuration alone, and the model's
    identity and the context length are stated per turn as injects. That is what makes reconnecting a
    change of settings and nothing else.

    The token-per-character calibration `invoke` accumulates is reset along with the rest, which is what a
    changed model wants: the old figure describes a tokenizer that is no longer in the picture.
    """
    fresh = connect(settings.backend_url, quiet=quiet)
    for name in fresh:
        settings[name] = fresh[name]
    return backend_status(settings)


def configure(model_info: env,
              backend_flavor: str,
              backend_url: str,
              quiet: bool = False,
              backend_is_reachable: bool = True) -> env:
    """Build the settings `env` from facts about a backend, without contacting one.

    `setup` is this function plus the two network queries that discover its arguments. Everything a turn
    needs is built here — system prompt, character card, tool tables, sampler settings, tokenizer, personas
    — so a caller holding the facts can obtain the *real* settings object with no backend in the picture.

    That is what makes it possible to build the prompt Raven would send and measure it elsewhere, which is
    otherwise done by hand-assembling an `env` with a few plausible fields. Such a replica carries about a
    third of the real object and a stand-in system prompt, so a probe claiming to measure "what Raven sends"
    measures something else — the failure this split exists to remove.

    `model_info`: What `_resolve_model_info` returns: an `env` with `label`, `model_id`, `context_length`,
                  `is_vlm` and `loaded`. Synthesize one to configure against a hypothetical model;
                  `context_length` may be `None`, which defaults as it does for a backend that does not
                  report one, and `is_vlm`/`loaded` may be `None`, which is how a backend that reports
                  neither is represented.

    `backend_flavor`: "oobabooga", "lmstudio" or "generic". Gates a few request/response details.

    `backend_url`: Recorded in the settings and used by `invoke`. No request is made from here, so a
                   configure-only caller may pass the URL it intends to use later, or a placeholder.

    `quiet`: As `setup`.

    `backend_is_reachable`: Whether anything answered at `backend_url`. `True` for every caller that has
                            actually spoken to a backend, which is why it defaults that way; `connect`
                            passes `False` when it could not, so that the placeholder settings it returns
                            say what they are rather than looking like a backend reporting nothing.

    Returns the same `env` as `setup`; see its docstring for the fields.
    """
    model = model_info.label  # human-facing identity for the character card (never a guess)
    request_model = librarian_config.llm_model or model_info.model_id  # id sent in requests (LM Studio JIT), or None

    # Context window: report the *loaded* length, never the model's theoretical max. When the backend doesn't
    # expose it (ooba doesn't here; a generic backend can't), default conservatively to 64k and warn — smaller
    # than that isn't useful for discussing a scientific fulltext, so we can assume at least that much.
    context_length = model_info.context_length
    if context_length is None:
        context_length = 64 * 1024
        if not quiet:
            logger.warning(f"configure: backend '{backend_flavor}' at {backend_url} did not report a loaded context length; defaulting to {context_length} tokens.")

    user = librarian_config.llm_user_name
    char = librarian_config.llm_char_name

    # SillyTavern would call these "macros".
    #
    # No date here, deliberately: this text is built once, at app start, so a date written into it goes wrong
    # at the first midnight the session survives. Raven states the current date - weekday included, so that the
    # model never has to do calendar arithmetic - in the system message on every turn instead; see
    # `scaffold.build_turn_prompt`.
    #
    # `model` and `context_length` are here but are unused by what Raven ships, for the same reason one step
    # weaker: they are stable within a session rather than within a day, and not even that if the user loads
    # a different model or the app reconnects to a backend that was down. They are stated per turn as well.
    # The config file says so at the slots that receive them, which is where someone writing prose looks.
    template_vars = env(user=user,
                        char=char,
                        model=model,
                        context_length=context_length)
    system_prompt = librarian_config.setup_system_prompt(template_vars)
    character_card = librarian_config.setup_character_card(template_vars)
    user_card = librarian_config.setup_user_card(template_vars)
    greeting = librarian_config.llm_greeting

    # Set up the chat completion request metadata template. Tool-calling instructions are NOT injected
    # client-side: every tool-capable model new enough to matter carries them in its own chat template, and the
    # backend builds them from the `tools` field below. `invoke` provides or strips `tools` per invocation.
    request_data = {
        "stream": True,  # stream each token to the client as it is generated, for live UI updates
        "messages": [],  # chat transcript including system messages; populated per-call by `invoke`
        "tools": TOOLS,  # tools available for tool-calling, for models that support it
    }
    if request_model is not None:
        request_data["model"] = request_model  # names the model (LM Studio JIT loads it on demand); harmless elsewhere
    if backend_flavor == "oobabooga":
        # ooba's API default mode is already "instruct" (verified), but other installs/versions can default to
        # "chat-instruct" (which adds roleplay framing), so send it explicitly. lmstudio/generic have no `mode`
        # field — there, messages -> the baked-in chat template is the only behaviour.
        request_data["mode"] = "instruct"
    # Merge the sampler settings. A `None` value drops the field — the Pythonic "use the backend default" signal,
    # rather than literally sending `null` (which some backends reject).
    request_data.update({key: value for key, value in librarian_config.llm_sampler_config.items() if value is not None})
    # Per-turn output cap. `None` (or an absent key) in the sampler config means "no cap": let the model generate
    # up to the full context window, the backend clamping to whatever the prompt leaves free. We send this as an
    # explicit ceiling rather than omitting the field, because omission is NOT backend-uniform — LM Studio treats
    # an absent `max_tokens` as unbounded, but ooba's OpenAI layer falls back to its own small default. `prefill`
    # overrides this per-call (see `invoke`'s `max_tokens`), so it doesn't affect token counting.
    if request_data.get("max_tokens") is None:
        request_data["max_tokens"] = context_length

    # See `raven.librarian.chatutil.create_chat_message`.
    personas = {"user": user,
                "assistant": char,
                "system": None,
                "tool": None}

    # List of strings after which to interrupt the LLM.
    # Useful mainly with older models that tend to speak on behalf of the user.
    stopping_strings = [f"\n{user}:"]

    # Token counting: find the optional local tokenizer for exact counts, and seed the tokens-per-character ratio used
    # by the estimate path (`count_tokens` tier 3) until real `usage` refines it (see `invoke`).
    tokenizer_source = None
    if librarian_config.llm_tokenizer_path:
        tokenizer_source = _resolve_tokenizer_source(librarian_config.llm_tokenizer_path,
                                                     [model, request_model])

    settings = env(user=user, char=char, model=model,
                   model_id=request_model,  # model id sent in requests (LM Studio JIT), or None
                   backend_flavor=backend_flavor,
                   context_length=context_length,  # loaded context window in tokens (backend-reported, or the 64k default)
                   model_is_vlm=model_info.is_vlm,  # whether the loaded model accepts image input: True/False, or None if unknown (gates image attach)
                   model_is_loaded=model_info.loaded,  # whether the backend has a model resident: True/False, or None if unknown (drives the backend-status readout)
                   backend_is_reachable=backend_is_reachable,  # whether anything answered at `backend_url`; see `backend_status`
                   backend_supports_continue=(backend_flavor == "oobabooga"),  # ooba has an explicit continue flag; others don't
                   tokenizer=None,  # local tokenizer for exact counts, or None (see `count_tokens`); loaded in the background below
                   tokens_per_character=_DEFAULT_TOKENS_PER_CHARACTER,  # estimate-path calibration; refined from usage in `invoke`
                   system_prompt=system_prompt,
                   character_card=character_card,
                   user_card=user_card,
                   stopping_strings=stopping_strings,
                   greeting=greeting,
                   tools=TOOLS,  # for inspection
                   tool_entrypoints=TOOL_ENTRYPOINTS,  # for our implementation to be able to call them
                   document_tool_names=DOCUMENT_TOOL_NAMES,  # subset of `TOOLS` gated on the document database
                   network_tool_names=NETWORK_TOOL_NAMES,  # subset of `TOOLS` gated on the "Internet" switch
                   backend_url=backend_url,
                   request_data=request_data,
                   personas=personas,
                   formatters=chatutil.default_formatters())  # per-run overridable; see its docstring

    if not quiet:
        # API key already loaded during module bootup; here, we just inform the user.
        if "Authorization" in headers:
            print(f"{colorizer.Fore.GREEN}{colorizer.Style.BRIGHT}Loaded LLM API key from '{str(librarian_config.llm_api_key_file)}'.{colorizer.Style.RESET_ALL}")
            print()
        else:
            print(f"{colorizer.Fore.YELLOW}{colorizer.Style.BRIGHT}No LLM API key configured.{colorizer.Style.RESET_ALL} If your LLM needs an API key to connect, put it into '{str(librarian_config.llm_api_key_file)}'.")
            print("This can be any plain-text data your LLM's API accepts in the 'Authorization' field of the HTTP headers.")
            print("For username/password, the format is 'user pass'. Do NOT use a plaintext password over an unencrypted http:// connection!")
            print()

    _start_tokenizer_load(settings, tokenizer_source)
    return settings


def _start_tokenizer_load(settings: env, tokenizer_source: Optional[str]) -> None:
    """Report which tier `count_tokens` will use, and start loading a local tokenizer if there is one.

    Loading happens on a daemon thread and assigns `settings.tokenizer` when it finishes: reading a GGUF
    takes several seconds, and the counting callers include the GUI's context-fill readout, which runs on the
    thread that also delivers keystrokes. Counts asked for before then use the tier below, so the readout
    starts out approximate and sharpens — the same shape as the backend-figure upgrade it replaces.
    """
    if tokenizer_source is None:
        if settings.backend_flavor == "oobabooga":
            logger.info("_start_tokenizer_load: no local tokenizer configured; token counts will be exact, from the backend's token-count endpoint.")
        else:
            logger.info("_start_tokenizer_load: no local tokenizer configured; token counts will be estimated from a character ratio, "
                        f"and upgraded to the backend's own figure where that looks like the whole prompt. Set '{librarian_config.__name__}.llm_tokenizer_path' "
                        "to a model archive, a .gguf, or a HuggingFace tokenizer directory for exact offline counts.")
        return

    logger.info(f"_start_tokenizer_load: token counts will be exact and offline, from '{tokenizer_source}' (loading in the background).")

    def load_it() -> None:
        # The backend is asked to confirm the tokenizer, which is two small requests — one more reason this
        # belongs on a thread of its own rather than in the startup path.
        settings.tokenizer = _load_local_tokenizer(tokenizer_source, _make_backend_token_counter(settings))

    threading.Thread(target=load_it, name="llmclient tokenizer load", daemon=True).start()

# # neutralize other samplers (copied from what SillyTavern sends)
# "top_p": 1,
# "typical_p": 1,
# "typical": 1,
# "top_k": 0,
# "add_bos_token": True,
# "sampler_priority": [
#     'quadratic_sampling',
#     'top_k',
#     'top_p',
#     'typical_p',
#     'epsilon_cutoff',
#     'eta_cutoff',
#     'tfs',
#     'top_a',
#     'min_p',
#     'mirostat',
#     'temperature',
#     'dynamic_temperature'
# ],
# "truncation_length": 24576,
# "ban_eos_token": False,
# "skip_special_tokens": True,
# "top_a": 0,
# "tfs": 1,
# "epsilon_cutoff": 0,
# "eta_cutoff": 0,
# "mirostat_mode": 0,
# "mirostat_tau": 5,
# "mirostat_eta": 0.1,
# "rep_pen": 1,
# "rep_pen_range": 0,
# "repetition_penalty_range": 0,
# "encoder_repetition_penalty": 1,
# "no_repeat_ngram_size": 0,
# "penalty_alpha": 0,
# "temperature_last": True,
# "do_sample": True,
# "repeat_penalty": 1,
# "tfs_z": 1,
# "repeat_last_n": 0,
# "n_predict": 800,
# "num_predict": 800,
# "num_ctx": 65536,

_DEFAULT_TOKENS_PER_CHARACTER = 0.27  # tokens per character; rough English/markup default, refined from real usage
_tokenizer_cache = {}  # path -> loaded tokenizer (or None if loading failed); avoids reloading the same tokenizer

def _resolve_tokenizer_source(path: str, model_names: Collection[str]) -> Optional[str]:
    """Decide what `path` is pointing at, and return the thing to load from. `None` if there is nothing.

    Three shapes are accepted, because a user has whichever one their setup produced:

      - **A HuggingFace tokenizer directory** (it contains `tokenizer.json`), or anything that is not a
        directory at all, which includes a repo id — passed through to `transformers` unchanged.
      - **A single `.gguf` file** — the model served by a llama.cpp-family backend.
      - **A directory to search**, which is the useful one when several models are in rotation: the `.gguf`
        matching `model_names` is picked out of it (`gguftokenizer.find_for_model`). Point this at the model
        archive rather than at any one backend's directory, so the answer does not depend on which backend is
        serving today. The backend may be on another machine; the archive must be reachable from this one by
        a file path, mounted or local.

    Deciding by *content* rather than by a config flag: the user has one path to give, and which kind it is
    is visible from the path itself.
    """
    source = pathlib.Path(path).expanduser()
    if source.is_dir() and not (source / "tokenizer.json").exists():
        found = gguftokenizer.find_for_model(source, model_names)
        return str(found) if found is not None else None
    return str(source)


def _make_backend_token_counter(settings: env) -> Callable[[str], Optional[int]]:
    """Return `text -> the backend's token count for it`, for checking a local tokenizer against the model in use.

    The count includes whatever framing the backend's chat template adds, and is not required to be free of
    it: `gguftokenizer` compares two probes and takes the difference, where a fixed overhead cancels.
    """
    def count(text: str) -> Optional[int]:
        if settings.backend_flavor == "oobabooga":
            return _ooba_token_count(settings.backend_url, text)  # counts the raw text, no template around it
        probe = [{"role": "user", "content": [chatutil.text_content_part(text)]}]
        # `calibrate=False` is load-bearing: these probes are short and mostly chat-template framing, so the
        # tokens-per-character ratio they imply is far above what ordinary text costs. Letting them calibrate
        # shrinks the character budget `fit_attachments_to_context` computes from that ratio, and the
        # attachments of the branch being measured are then truncated harder than they will be when sent.
        out = prefill(settings, probe, tools_enabled=False, calibrate=False)
        return (out.usage or {}).get("prompt_tokens") if out is not None else None
    return count


def _load_local_tokenizer(path: str, backend_counter: Optional[Callable[[str], Optional[int]]] = None):
    """Load (and cache) a local tokenizer for exact token counting, or return `None` on failure.

    `path` is a `.gguf` file, a directory with `tokenizer.json` + `tokenizer_config.json`, or a HF repo id.
    Failures (missing files, network, version skew, a GGUF whose tokenizer this Raven has not been verified
    against) are logged and degrade to the calibrated estimate rather than raising.

    **Slow** — several seconds for a GGUF. Call it off any thread that has to stay responsive; `configure`
    does that for the app's own tokenizer.
    """
    if path in _tokenizer_cache:
        return _tokenizer_cache[path]
    if path.lower().endswith(".gguf"):
        tokenizer = gguftokenizer.load(pathlib.Path(path), backend_counter)
    else:
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415 -- heavy import, deferred to first use
            tokenizer = AutoTokenizer.from_pretrained(path)
        except Exception as exc:  # noqa: BLE001 -- any load failure just means "no local tokenizer; use the estimate"
            logger.warning(f"_load_local_tokenizer: could not load tokenizer from '{path}': {type(exc)}: {exc}. Falling back to usage-calibrated token estimates.")
            tokenizer = None
    _tokenizer_cache[path] = tokenizer
    return tokenizer

def _ooba_token_count(backend_url: str, text: str) -> int:
    """Exact token count from oobabooga's `/v1/internal/token-count` endpoint."""
    # ooba's undocumented web API endpoints are listed in `text-generation-webui/extensions/openai/script.py`.
    response = requests.post(f"{backend_url}/v1/internal/token-count", headers=headers, json={"text": text}, timeout=librarian_config.llm_network_timeout)
    return response.json()["length"]

def count_tokens(settings: env, text: str) -> Tuple[int, bool]:
    """Count tokens in `text` for the loaded model. Returns `(count, is_exact)`.

    Useful for checking prompt length after injecting RAG context etc. Tiers, in order of preference:
      1. A configured local tokenizer (`config.llm_tokenizer_path`) — exact, offline, works on any backend.
      2. oobabooga's `/v1/internal/token-count` endpoint — exact.
      3. A calibrated tokens-per-character ratio (refined from each call's real `usage`; see `invoke`) — an *estimate*.
    The `is_exact` flag drives the GUI context-fill indicator's `X%` (exact) vs `~X%` (estimate) typography.
    Callers that only want the number use `count_tokens(...)[0]`.
    """
    if settings.tokenizer is not None:
        return len(settings.tokenizer.encode(text)), True
    if settings.backend_flavor == "oobabooga":
        return _ooba_token_count(settings.backend_url, text), True
    return round(len(text) * settings.tokens_per_character), False

def image_token_cost(settings: env, height: int, width: int) -> int:
    """Estimated token cost of one attached image for the loaded model — for the context-fill budget.

    A VLM image consumes a chunk of context that the text-only tokens-per-character ratio (`count_tokens` tier 3) can't
    see, so the pre-send indicator has to add it explicitly. The per-family costs live in
    `config.llm_image_token_cost`, keyed by a lowercase substring matched against the loaded model's id/family
    (first match wins; the `None` key is the fallback for unknown families). Each entry is a flat token count
    or a callable `(height, width) -> int` for models whose cost scales with resolution.

    Necessarily an *estimate*: it is a conservative published-scheme figure, refined away entirely once the
    backend reports the real `usage.prompt_tokens` for an image-bearing call (same self-correction path as the
    tokens-per-character ratio). `height`/`width` are the stored (wire) dimensions; they only matter for the
    resolution-scaling families.
    """
    table = librarian_config.llm_image_token_cost
    haystack = " ".join(str(part) for part in (settings.model, settings.model_id) if part).lower()
    chosen = table.get(None)  # fallback for unknown families
    for key, value in table.items():
        if key is not None and key in haystack:
            chosen = value
            break
    return int(chosen(height, width)) if callable(chosen) else int(chosen)

def count_branch_tokens(settings: env,
                        datastore: chattree.Forest,
                        head_node_id: str,
                        extract_attachments: bool = True) -> Tuple[int, bool]:
    """Estimate the token size of the conversation ending at `head_node_id`. Returns `(count, is_exact)`.

    Counts the *visible conversation content* — every message from the root down to `head_node_id`. It
    therefore under-reports the real prompt slightly: the system prompt's framing, the per-turn injects and
    the tool definitions all add tokens that are not in the stored messages. For the backend's own exact
    figure, submit the prompt and read `usage["prompt_tokens"]`; that is what `prefill` is for.

    Attachments are counted in the two different ways they cost context. An image consumes tokens the
    tokens-per-character ratio cannot see, so it is added as a per-family estimate (`image_token_cost`) and forces
    `is_exact` to `False`. An attached document rides the wire as text (folded into the message at
    wire-build), so its extracted text is counted alongside everything else and costs no accuracy - but it
    is counted at the size that will actually be *sent*, after `fit_attachments_to_context` has had its say.
    Counting the full text would read past 100% for a conversation whose prompt comfortably fits, since
    what overflows never leaves the machine.

    Two callers, wanting the same number for different reasons: the GUI's context-fill readout, and the
    budget that decides how much of a document `fetch_document` may return. Shared so they cannot disagree
    about how full the context is.

    `extract_attachments`: whether an attached document whose text is not extracted yet may be extracted
                           here. `False` skips such attachments and forces `is_exact` to `False`; text
                           already extracted is counted either way, so the answer is the same once a chat
                           has been used.

                           For a caller that must not *wait*, which is the GUI readout: extraction runs
                           pypdf, and that readout is refreshed on every HEAD change from a DPG callback —
                           where seconds of work freeze the keyboard, DPG running callbacks one at a time.
                           The undercount it accepts is temporary by construction, the same readout being
                           two-stage: a debounced background prefill replaces the figure with the backend's
                           exact one a moment later.

                           The budget caller passes the default, and must: an under-reported context would
                           let `fetch_document` return more than actually fits.
    """
    text_segments = []
    attachments = []  # (extracted text, budget kind)
    image_tokens = 0
    counted_every_attachment = True  # cleared when `extract_attachments=False` meets one that is not extracted yet
    for node_id in datastore.linearize_up(head_node_id):
        payload = datastore.get_payload(node_id)
        message = payload["message"]
        text_segments.append(chatutil.content_to_text(message.get("content")))
        sidecars_meta = payload.get("general_metadata", {}).get("sidecars", {})
        for part in message.get("content") or []:
            part_type = part.get("type")
            if part_type == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                filename = url[len(sidecarstore.SIDECAR_SCHEME):] if url.startswith(sidecarstore.SIDECAR_SCHEME) else None
                dims = (sidecars_meta.get(filename) or {}).get("stored_dimensions") if filename else None
                image_h, image_w = dims if dims else (1024, 1024)  # fallback for pre-stored-dims data; only matters for resolution-scaling families
                image_tokens += image_token_cost(settings, image_h, image_w)
            elif part_type == "text_file":
                file_url = (part.get("text_file") or {}).get("url", "")
                if file_url.startswith(sidecarstore.SIDECAR_SCHEME):
                    if extract_attachments:
                        text = textfilestore.sidecar_to_text(datastore, file_url)
                    else:
                        text = textfilestore.sidecar_text_if_extracted(file_url)
                        if text is None:  # not extracted yet, and this caller will not wait for it
                            counted_every_attachment = False
                            continue
                    attachments.append((text, attachment_budget_kind(part)))

    conversation_characters = sum(len(segment) for segment in text_segments)
    fitted_attachments = fit_attachments_to_context(settings, conversation_characters, attachments)
    count, is_exact = count_tokens(settings, "".join(text_segments + fitted_attachments))
    if image_tokens:
        count += image_tokens
        is_exact = False  # per-image token cost is an estimate, so the whole figure is now approximate
    if not counted_every_attachment:
        is_exact = False  # content was left out entirely, which is a stronger caveat than the readout's `~` can say
    return count, is_exact


def _clamped_fraction(value: float,
                      setting_name: str) -> float:
    """Clamp a configured fraction to `[0, 1]`, warning if that was necessary.

    These are hand-edited in a `.py` config, so a typo is a plain possibility rather than a hypothetical -
    and both plausible slips are bad in different ways. A negative per-fetch ceiling would refuse every
    fetch (looking like a broken tool), and a reserve above 1 would do the same for a different reason.
    Clamping keeps a slip degrading gracefully instead of disabling a feature silently.

    But clamping *quietly* would trade one silent failure for another: the feature would work, in a way the
    config plainly does not describe, and the config would go on saying something untrue indefinitely. So
    the log names the setting and both values. A fetch is a rare enough event that this cannot become spam,
    and a misconfiguration that survives because nobody was told is the outcome worth avoiding here.
    """
    clamped = min(1.0, max(0.0, float(value)))
    if clamped != value:
        logger.warning(f"_clamped_fraction: config setting '{setting_name}' is {value}, which is outside "
                       f"[0, 1]; using {clamped}. Fractions of the context window cannot be negative or "
                       f"exceed the whole window.")
    return clamped

def budget_for_fetched_text(settings: env,
                            used_tokens: int) -> int:
    """How many tokens of fetched text this conversation can still afford. `<= 0` means "refuse the fetch".

    Two limits, doing two different jobs, and the smaller one wins:

      - A *per-fetch* ceiling (`config.docs_fetch_max_fraction_of_context`), so that one document cannot
        crowd out the conversation it is supposed to inform. This is the one that normally binds, and what
        oversized text is truncated *to*.
      - A *floor* on what is left for the discussion (`config.context_reserve_fraction`), which binds only
        once the conversation has already grown large. When it does bind, the answer is to refuse rather
        than to hand back a sliver: at that point the useful move is a new chat.

    `used_tokens`: how much of the window the conversation already occupies, from `count_branch_tokens`.
                   Recompute it per tool round rather than reusing a turn-start figure - the model's own
                   reasoning and any earlier tool results in the same turn have already been added by then.
    """
    context_length = settings.context_length
    per_fetch_ceiling = _clamped_fraction(librarian_config.docs_fetch_max_fraction_of_context,
                                          "docs_fetch_max_fraction_of_context") * context_length
    reserve = _clamped_fraction(librarian_config.context_reserve_fraction,
                                "context_reserve_fraction") * context_length
    return int(min(per_fetch_ceiling, context_length - used_tokens - reserve))

def truncate_middle(text: str,
                    max_characters: int) -> str:
    """Return `text` shortened to at most `max_characters`, dropping from the middle, omission marked.

    The middle is what goes because the ends are what carry: for a paper that keeps the abstract and
    introduction at one end and the conclusions at the other, and spends the omission on the methods.

    The marker is not decoration. Handed silently truncated text, a model has no way to tell a document that
    stops mid-sentence from one that ends there, and will summarize the fragment as though it were the whole
    - so the omission is stated, in characters, exactly where it happens.
    """
    if len(text) <= max_characters:
        return text
    marker_template = "\n\n[... {} characters omitted ...]\n\n"
    # Reserve room for the marker itself, so the result really does fit. The marker's length depends on the
    # number printed in it, which depends on how much is cut - so size it against the worst case (the full
    # length), which errs towards keeping slightly less text rather than overshooting the budget.
    keepable = max_characters - len(marker_template.format(len(text)))
    if keepable <= 0:  # budget too small to say anything useful in
        return ""
    head_length = (keepable + 1) // 2  # odd character goes to the head: an opening is worth more than a tail
    tail_length = keepable - head_length
    head = text[:head_length]
    tail = text[len(text) - tail_length:] if tail_length else ""
    return head + marker_template.format(len(text) - head_length - tail_length) + tail

def fit_text_to_token_budget(settings: env,
                             text: str,
                             budget_tokens: int) -> str:
    """Return `text` cut down to roughly `budget_tokens`, or `""` if the budget cannot hold anything.

    The token-facing front for `truncate_middle`. It exists so that callers never do the unit conversion
    themselves: the budget is in tokens, the truncation is in characters, and getting that backwards
    produces a limit wrong by a factor of about four in whichever direction hurts.

    "Roughly" is honest rather than hedging. The conversion uses `settings.tokens_per_character`, the same
    calibrated estimate `count_tokens` falls back on, which drifts with the text: dense markup and long
    identifiers tokenize worse than prose. Exactness is not needed here - the reserve that
    `budget_for_fetched_text` keeps free is far larger than the error.
    """
    if budget_tokens <= 0:
        return ""
    tokens_per_character = settings.tokens_per_character or _DEFAULT_TOKENS_PER_CHARACTER
    return truncate_middle(text, int(budget_tokens / tokens_per_character))  # tokens / (tokens/character) = characters

# What stands in for an attachment the window has no room for at all, in the manner of
# `CANONICAL_NO_ROOM_TO_FETCH`. The file is still named: a silently vanished attachment leaves the model
# reading a message that refers to a document it cannot see, which it will resolve by guessing.
CANONICAL_ATTACHMENT_OMITTED = "[Attached file: {name} - not shown, because there is no room left for it in the context window.]"

# How coarsely the shared attachment budget is rounded down, in characters (~2200 tokens at the default
# ratio). Purely a stability measure; see `fit_attachments_to_context`.
_ATTACHMENT_BUDGET_QUANTUM = 8192

def _share_characters(wanted: List[int],
                      budget: int) -> List[int]:
    """Split `budget` characters over items wanting `wanted` characters each. Returns the allowances.

    Max-min fair (the classic water-filling allocation): raise a common level until the budget runs out,
    and let anything that wanted less than the level through untouched. So a short attachment alongside a
    book is not cut at all - it never reached the level - and the book absorbs the whole shortfall. Equal
    shares would instead cut the short one to half the budget for no gain, since the characters freed that
    way are ones nobody was asking for.

    Order-independent by construction: the allocation is a property of the multiset of demands, so permuting
    the inputs permutes the outputs and changes nothing else. That is the right shape for an allocation
    anyway — an attachment's share should not depend on where it happens to sit in the list — and it is what
    lets two callers walk the same attachments in opposite directions (`count_branch_tokens` from the head
    up, `serialize_history_for_wire` from the root down) and arrive at the same numbers.
    """
    allowances = [0] * len(wanted)
    if budget <= 0:
        return allowances
    remaining = budget
    unsettled = list(range(len(wanted)))
    while unsettled:
        level = remaining // len(unsettled)
        modest = [i for i in unsettled if wanted[i] <= level]
        if not modest:  # everyone left wants more than an equal share of what is left: split it evenly
            for i in unsettled:
                allowances[i] = level
            break
        for i in modest:  # served in full, and their leftovers raise the level for the rest
            allowances[i] = wanted[i]
            remaining -= wanted[i]
        unsettled = [i for i in unsettled if wanted[i] > level]
    return allowances

# The two ways a document can end up attached to a message, as far as the context budget is concerned. The
# distinction is one of *intent*, which is why it cannot be read off the document itself: the same PDF is
# governed differently depending on who put it there.
ATTACHMENT_REQUESTED = "requested"  # the user handed it over: read this
ATTACHMENT_SPECULATIVE = "speculative"  # the model reached for it, having seen a search result

def attachment_budget_kind(part: Dict[str, Any]) -> str:
    """Classify one `text_file` content-part as `ATTACHMENT_REQUESTED` or `ATTACHMENT_SPECULATIVE`.

    The single place that maps a provenance `source` onto a budget policy, so that the two readers of the
    attachment budget cannot disagree about it. `count_branch_tokens` (the GUI's context-fill readout) and
    `serialize_history_for_wire` (what is actually sent) walk the same attachments from opposite ends, and
    a divergence here would show up as a readout that drifts away from the request it claims to describe.

    A classification rather than an equality test, because the vocabulary is open: `sidecarstore.base_provenance`
    reserves `"paste_url"` and `"mcp:<server>"` for pathways that do not exist yet, and both already have an
    answer here - `"paste_url"` is a URL the user typed, so it is requested, and `"mcp:<server>"` is a tool
    result whatever server produced it.

    Two axes are visible in those values - *where the bytes came from* (a local file, the network) and *who
    asked for them* (the user, a tool) - and only the second is a budget question. They stay **projections of
    the one stored `source`** rather than becoming two stored fields. A pathway is a sum type whose cases are
    the ones that actually occur, where two independent fields would also spell combinations that cannot happen;
    `source` is written into the sidecar's `.meta.json` on disk as well as onto the part, so splitting it later
    costs a migration of both; and `"mcp:<server>"` carries a third thing (which server) that neither axis
    captures. If a second reader ever wants the other axis, give it a predicate over `source`, not a new field.

    An unrecognized source is treated as requested. That is the conservative direction: it sends the document
    whole, which risks a large prompt, rather than silently truncating something the user asked for.
    """
    source = (part.get("text_file") or {}).get("source", "")
    if source == "tool_result" or source.startswith("mcp:"):
        return ATTACHMENT_SPECULATIVE
    return ATTACHMENT_REQUESTED

def fit_attachments_to_context(settings: env,
                               conversation_characters: int,
                               attachments: List[Tuple[str, str]]) -> List[str]:
    """Cut attached-document texts down to what the context window can carry. Returns them in the same order.

    `attachments`: `(text, kind)` pairs, `kind` being one of `ATTACHMENT_REQUESTED` / `ATTACHMENT_SPECULATIVE`
                   (see `attachment_budget_kind`). Callers pass the classification rather than deciding
                   anything themselves - that is what keeps the two of them agreeing.

    Two limits, and which ones apply depends on how the document got here, because the difference is one of
    intent rather than of content:

      - **Both kinds** are bounded by `config.context_reserve_fraction`, the floor under the discussion
        itself, and share whatever that leaves among themselves.
      - **A speculative one additionally gets the per-document ceiling**
        (`config.docs_fetch_max_fraction_of_context`, the same one `budget_for_fetched_text` applies to a
        `fetch_document` call). The model saw a search result and reached for the page; a hunch should not be
        able to crowd out the conversation it was meant to inform.

    A requested attachment gets no per-document ceiling on purpose. It is the user handing over a paper and
    saying read this, and a ceiling of a tenth of the window would answer that by showing four pages.

    Nothing is cut while everything fits, which is the overwhelmingly common case and returns the texts
    unchanged. The budget only binds where the alternative is not "a slightly shorter paper" but a request
    that overflows the window outright.

    The budget is quantized (`_ATTACHMENT_BUDGET_QUANTUM`) once it binds, and that is worth a word because
    it looks like sloppiness. Folded attachment text is part of the prompt *prefix*, so a budget that
    drifted by a few characters per turn - which it would, since the conversation grows under it - would
    rewrite that prefix every turn and force the backend to reprocess the whole prompt each time, in
    precisely the situation where the prompt is already enormous. Rounding down to a coarse step keeps the
    fold byte-identical across a run of turns and costs at most one step of unused budget.

    `conversation_characters`: how many characters of everything *else* the request carries - the messages,
                              minus the attachment text being sized here. Characters rather than tokens
                              throughout: this runs on the hot path, once per request, and the truncation
                              it feeds is in characters anyway, so a token count would be converted back.
    """
    if not attachments:
        return []
    tokens_per_character = settings.tokens_per_character or _DEFAULT_TOKENS_PER_CHARACTER
    reserve = _clamped_fraction(librarian_config.context_reserve_fraction, "context_reserve_fraction")
    window_characters = settings.context_length / tokens_per_character  # tokens / (tokens/character) = characters
    budget = int(window_characters * (1.0 - reserve)) - conversation_characters
    # The per-document ceiling is applied to what a speculative attachment *asks for*, before the fair split
    # rather than after it. Clamping the demand instead of the allowance means the characters a ceilinged
    # document does not get are released to the others, which is what the water-filling is for - clamping
    # afterwards would leave them unused.
    ceiling = int(_clamped_fraction(librarian_config.docs_fetch_max_fraction_of_context,
                                    "docs_fetch_max_fraction_of_context") * window_characters)
    wanted = [min(len(text), ceiling) if kind == ATTACHMENT_SPECULATIVE else len(text)
              for text, kind in attachments]
    if sum(wanted) <= budget:
        allowances = wanted  # everything fits; a ceilinged document is still cut to its ceiling
    else:
        budget -= budget % _ATTACHMENT_BUDGET_QUANTUM
        allowances = _share_characters(wanted, budget)
    # `truncate_middle` is a no-op when the text already fits its allowance, which is the ordinary case.
    return [truncate_middle(text, allowance) for (text, _kind), allowance in zip(attachments, allowances)]

# --------------------------------------------------------------------------------
# Streaming tool-call accumulation (shared by `invoke`)

def _accumulate_tool_call_delta(accumulator: Dict[int, Dict[str, str]],
                                tool_call_fragments: List[Dict]) -> None:
    """Fold one streamed delta's `tool_calls` fragments into `accumulator` (keyed by `index`), in place.

    Unifies the two backend behaviours behind one accumulator:
      - oobabooga delivers a complete tool-call object in a single delta (id + name + full arguments at once);
      - LM Studio / OpenAI stream incrementally — the first fragment carries id/type/function.name with empty
        arguments, later fragments carry only `function.arguments` string pieces to concatenate.
    Setting id/type/name when present and *appending* arguments handles both; parallel calls (distinct
    indices, e.g. a model requesting two cities' weather at once) accumulate into separate slots.
    """
    for fragment in tool_call_fragments:
        idx = fragment.get("index", 0)
        slot = accumulator.setdefault(idx, {"id": "", "type": "function", "name": "", "arguments": ""})
        if fragment.get("id"):
            slot["id"] = fragment["id"]
        if fragment.get("type"):
            slot["type"] = fragment["type"]
        function_fragment = fragment.get("function") or {}
        if function_fragment.get("name"):
            slot["name"] = function_fragment["name"]
        if function_fragment.get("arguments"):
            slot["arguments"] += function_fragment["arguments"]

def _materialize_tool_calls(accumulator: Dict[int, Dict[str, str]]) -> Optional[List[Dict]]:
    """Turn the streaming accumulator into a `tool_calls` list in index order, or `None` if empty.

    Output shape matches what `perform_tool_calls` consumes: `{type, function: {name, arguments}, id, index}`.
    """
    if not accumulator:
        return None
    return [{"type": slot["type"],
             "function": {"name": slot["name"], "arguments": slot["arguments"]},
             "id": slot["id"],
             "index": str(idx)}
            for idx, slot in sorted(accumulator.items())]

# --------------------------------------------------------------------------------
# Streaming parser: raw deltas -> typed events (`invoke`'s single source of truth)

# Inline-tag tokens that some models/backends emit in the *content* stream. `invoke` parses them out
# and re-routes them into typed events, so the chat client never has to regex-sniff the text.
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

# The generic / Qwen-style inline tool-call spelling: a JSON object between `<tool_call>` and `</tool_call>`.
# This is the only inline tool-call form we parse. Gemma's spelling is different — `<|tool_call>call:` NAME
# `{...}<tool_call|>` (inner pipes, a `call:` prefix, and a bespoke non-JSON argument body) — and we don't parse
# it. On LM Studio (the live-verified Gemma backend) tool calls arrive structured in the OpenAI `tool_calls`
# field, so there's nothing to parse inline. Whether a raw-passthrough backend (oobabooga / generic) serving
# Gemma emits this form inline in `content` instead — the way it does for the reasoning channel below — is
# unverified; if one does, Gemma tool-calling there would need a dedicated parser for the `call:...` syntax.
_TOOLCALL_OPEN = "<tool_call>"
_TOOLCALL_CLOSE = "</tool_call>"

# Gemma 3/4 spell the reasoning channel differently from the `<think>` convention: an asymmetric
# `<|channel>thought` ... `<channel|>` pair (Gemma emits the channel name `thought` right after the opening
# marker; see the model's chat template). A backend that passes the raw stream through (oobabooga, generic
# OpenAI-compat) delivers this inline in `content`; llama.cpp / LM Studio split it out into the native
# `reasoning_content` delta channel instead (handled directly in `StreamParser.feed`). We match the opening
# marker without a trailing newline so a stray whitespace variation can't hide it — the model's `\n` after
# `thought` just rides along into the reasoning text, same as the blank line Qwen emits after `<think>`.
_GEMMA_THINK_OPEN = "<|channel>thought"
_GEMMA_THINK_CLOSE = "<channel|>"

# Every reasoning-open tag mapped to the close that ends it. `_PS_TEXT` scans for any open; on a match the
# parser remembers the corresponding close to scan for while in `_PS_THINK` (the `<think>` and Gemma pairs
# are not interchangeable — `<think>` closes with `</think>`, `<|channel>thought` closes with `<channel|>`).
_THINK_OPEN_TO_CLOSE = {_THINK_OPEN: _THINK_CLOSE,
                        _GEMMA_THINK_OPEN: _GEMMA_THINK_CLOSE}
_THINK_OPEN_TAGS = tuple(_THINK_OPEN_TO_CLOSE.keys())  # just the opens; `.keys()` spelled out for clarity

# Parser states.
_PS_TEXT = "text"          # outside any special block
_PS_THINK = "think"        # inside an inline reasoning block (<think>...</think> or Gemma's channel form)
_PS_TOOLCALL = "toolcall"  # inside an inline <tool_call>...</tool_call> block

def _longest_partial_tag_suffix(buf: str, tags: Tuple[str, ...]) -> int:
    """Length of the longest suffix of `buf` that is a *proper* prefix of some tag in `tags`.

    This is the look-ahead the streaming parser holds back at a chunk boundary: a tag may arrive split
    across two stream chunks (`</thi` then `nk>`), so the trailing bytes that could begin a tag must wait
    for the next chunk before being emitted as plain text. Returns 0 when nothing needs holding back.
    """
    best = 0
    for tag in tags:
        maxk = min(len(buf), len(tag) - 1)  # a *proper* prefix is shorter than the whole tag
        for k in range(maxk, best, -1):
            if buf.endswith(tag[:k]):
                best = k
                break
    return best

def _scan_for_tags(buf: str, tags: Tuple[str, ...]) -> Tuple[str, Optional[str], str]:
    """Scan `buf` for the earliest complete tag from `tags`.

    Returns `(emit, tag, rest)`:
      - complete tag found: `emit` is the text before it, `tag` is the matched tag, `rest` is the text after.
      - no complete tag: `tag` is `None`, `emit` is the text safe to emit now, and `rest` is a held-back
        trailing partial (a possible tag split across the chunk boundary), to be reconsidered next chunk.
        `rest` may be empty.
    """
    best_pos = None
    best_tag = None
    for tag in tags:
        pos = buf.find(tag)
        if pos != -1 and (best_pos is None or pos < best_pos):
            best_pos = pos
            best_tag = tag
    if best_tag is not None:
        return buf[:best_pos], best_tag, buf[best_pos + len(best_tag):]
    hold = _longest_partial_tag_suffix(buf, tags)
    if hold:
        return buf[:-hold], None, buf[-hold:]
    return buf, None, ""

def _tool_call_dedup_key(name: str, arguments: str) -> Tuple[str, str]:
    """Stable identity for dedup: `(name, normalized-JSON arguments)`.

    Used to suppress double-emitted tool calls — some backends emit a call both as an inline `<tool_call>`
    tag in the content stream *and* in the structured `tool_calls` field at EOS. Normalizing the arguments
    JSON (sorted keys) makes the two representations compare equal despite whitespace/key-order differences.
    """
    try:
        normalized = json.dumps(json.loads(arguments), sort_keys=True)
    except (json.JSONDecodeError, ValueError, TypeError):
        normalized = (arguments or "").strip()
    return (name, normalized)

class StreamParser:
    """Turn raw streamed deltas into typed events; `invoke`'s single source of truth for the response stream.

    Feed each delta's `content` and `reasoning_content` (either may be empty) via `feed`. The parser:

      - routes `reasoning_content` deltas straight to `reasoning` events (the native separate channel that
        llama.cpp / LM Studio use for Qwen / Gemma / GPT-OSS);
      - parses inline reasoning out of the `content` stream into `reasoning` events — both the `<think>`
        convention (Qwen and most others) and Gemma's `<|channel>thought` ... `<channel|>` form, for backends
        (oobabooga, generic OpenAI-compat) that pass the model's raw stream through instead of splitting the
        reasoning into the native channel above;
      - parses inline `<tool_call>...</tool_call>` out of the `content` stream into `tool_call` events;
      - emits everything else as `content` events;

    stripping the inline tags from the content stream as it goes. A small look-ahead buffer
    (see `_scan_for_tags`) handles tags split across chunk boundaries.

    Events are dicts:

        {"type": "content",   "text": str}
        {"type": "reasoning", "text": str}
        {"type": "tool_call", "id": str, "name": str, "arguments": str}   # `arguments` is a JSON string

    At stream end, call `finalize(native_tool_calls)` to flush any buffered text and emit native (OpenAI
    `tool_calls` field) calls that weren't already seen inline — deduped against inline-parsed calls by
    `(name, normalized arguments)`, so a backend that double-emits the same call (inline tag AND structured
    field, as some ooba builds do) yields exactly one `tool_call` event.
    """
    def __init__(self):
        self._state = _PS_TEXT
        self._buf = ""                   # content look-ahead buffer (may hold a split tag at a chunk boundary)
        self._think_close = _THINK_CLOSE  # the close tag that ends the current _PS_THINK block (set on open)
        self._toolcall_json = ""         # accumulates the raw JSON inside an inline <tool_call> block
        self._inline_call_keys = set()   # (name, normalized args) of inline-emitted calls, for native dedup
        self._synthetic_id_counter = 0   # inline tool calls carry no id; assign a synthetic one

    def feed(self, content: str, reasoning: str) -> List[Dict]:
        """Feed one delta's content and reasoning_content (either may be empty). Returns the typed events produced."""
        events: List[Dict] = []
        if reasoning:  # native reasoning channel: never contains inline tags
            events.append({"type": "reasoning", "text": reasoning})
        if content:
            self._buf += content
            events.extend(self._drain())
        return events

    def _drain(self) -> List[Dict]:
        events: List[Dict] = []
        progressing = True
        while progressing and self._buf:
            progressing = False
            if self._state == _PS_TEXT:
                emit, tag, rest = _scan_for_tags(self._buf, _THINK_OPEN_TAGS + (_TOOLCALL_OPEN,))
                if emit:
                    events.append({"type": "content", "text": emit})
                self._buf = rest
                if tag in _THINK_OPEN_TO_CLOSE:
                    self._state = _PS_THINK
                    self._think_close = _THINK_OPEN_TO_CLOSE[tag]  # the matching close (<think> and Gemma differ)
                    progressing = True
                elif tag == _TOOLCALL_OPEN:
                    self._state = _PS_TOOLCALL
                    self._toolcall_json = ""
                    progressing = True
            elif self._state == _PS_THINK:
                emit, tag, rest = _scan_for_tags(self._buf, (self._think_close,))
                if emit:
                    events.append({"type": "reasoning", "text": emit})
                self._buf = rest
                if tag == self._think_close:
                    self._state = _PS_TEXT
                    progressing = True
            else:  # _PS_TOOLCALL: accumulate raw JSON until the closing tag
                idx = self._buf.find(_TOOLCALL_CLOSE)
                if idx != -1:
                    self._toolcall_json += self._buf[:idx]
                    self._buf = self._buf[idx + len(_TOOLCALL_CLOSE):]
                    maybe_event = self._inline_tool_call_event(self._toolcall_json)
                    if maybe_event is not None:
                        events.append(maybe_event)
                    self._toolcall_json = ""
                    self._state = _PS_TEXT
                    progressing = True
                else:  # no closing tag yet — accumulate, but hold back a possible split closing tag at the end
                    hold = _longest_partial_tag_suffix(self._buf, (_TOOLCALL_CLOSE,))
                    cut = len(self._buf) - hold
                    self._toolcall_json += self._buf[:cut]
                    self._buf = self._buf[cut:]
        return events

    def _inline_tool_call_event(self, raw_json: str) -> Optional[Dict]:
        raw = raw_json.strip()
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"StreamParser: failed to parse inline <tool_call> JSON; dropping. Raw: {raw!r}")
            return None
        name = parsed.get("name", "")
        arguments = parsed.get("arguments", {})
        if not isinstance(arguments, str):  # OAI convention stores `arguments` as a JSON *string*
            arguments = json.dumps(arguments)
        self._synthetic_id_counter += 1
        self._inline_call_keys.add(_tool_call_dedup_key(name, arguments))
        return {"type": "tool_call", "id": f"inline_{self._synthetic_id_counter}", "name": name, "arguments": arguments}

    def finalize(self, native_tool_calls: Optional[List[Dict]]) -> List[Dict]:
        """Flush buffered text and emit native tool calls not already seen inline. Returns the trailing events."""
        events: List[Dict] = []
        if self._buf:  # an unterminated block at stream end — emit what we have so nothing is silently lost
            if self._state == _PS_THINK:
                events.append({"type": "reasoning", "text": self._buf})
            elif self._state == _PS_TOOLCALL:
                logger.warning("StreamParser.finalize: stream ended inside an unterminated <tool_call> block; dropping partial JSON.")
            else:
                events.append({"type": "content", "text": self._buf})
            self._buf = ""
        for call in native_tool_calls or []:
            function = call.get("function") or {}
            name = function.get("name", "")
            arguments = function.get("arguments", "")
            if _tool_call_dedup_key(name, arguments) in self._inline_call_keys:
                continue  # double-emission: this call already surfaced inline; suppress the structured copy
            events.append({"type": "tool_call", "id": call.get("id", ""), "name": name, "arguments": arguments})
        return events

# --------------------------------------------------------------------------------
# The most important function - call LLM, parse result

def serialize_history_for_wire(settings: env,
                               history: List[Dict],
                               *,
                               continue_: bool,
                               datastore: Optional[chattree.PersistentForest] = None) -> List[Dict]:
    """Return a wire-ready deep copy of `history`: text scrubbed, image parts preserved and sidecar-resolved.

    Per-message transform, applied to every message (or all but the last when `continue_`):

      - **Text.** All text parts are joined and scrubbed (`scrub(thoughts_mode="discard")`) into a single text
        part. Reasoning (thinking) rides out-of-band in the `reasoning_content` sibling field, untouched here —
        the supported families' chat templates (Qwen 3, Gemma 4) read it on input and apply their own
        strip-prior / preserve-current-turn policy via the last-user-message boundary, so Raven doesn't
        second-guess them. The scrub is mostly a legacy safety net now: it strips any inline `<think>` blocks
        still embedded in OLD content (pre-migration data) and normalizes the persona prefix; on new-parser
        content it's a no-op on the text.

      - **Images.** `image_url` parts are preserved (not collapsed away) and appended after the text part in
        their original order. A `sidecar:<filename>` URL is resolved to a real `data:<mime>;base64,...` URL by
        reading the sidecar bytes (`imagestore.sidecar_url_to_data_url`), so the model receives the image while
        the stored message keeps its `sidecar:` reference. Resolution needs `datastore` (the chat's
        `PersistentForest`); without it, sidecar URLs pass through unchanged — harmless for image-free callers
        (throwaway tasks / prefill on text-only chats), which carry no sidecar parts anyway.

      - **Documents.** `text_file` parts (attached plain-text / PDF documents) have no native wire form, so each
        is *folded into the message text*: its plaintext is extracted on demand from the sidecar
        (`textfilestore.sidecar_to_text`) and appended after the user's text under an `[Attached file: ...]`
        header. Any model can therefore use an attached document — no vision capability required. Like image
        resolution this needs `datastore`; without it there are no `text_file` parts to fold.

        The attachments of the *whole* request are sized together, against what the context window has left
        once the conversation and the reserve are accounted for (`fit_attachments_to_context`). That is why
        the fold takes two passes over the history rather than one: the budget one attachment gets depends
        on how much the others are asking for, and on how long the conversation has become — neither of
        which is known while looking at a single message. When everything fits, which is the ordinary case,
        the result is the same text as an unbudgeted fold would produce.

    `continue_`: when `True`, the last message (the AI message being continued) is left exactly as-is — neither
                 scrubbed nor image/document-resolved (assistant continuations carry no attachments).

    `datastore`: the chat's `PersistentForest`, needed only to resolve the `sidecar:` URLs of attachments.
                 A history with no attachments — which is what a script assembling a prompt from
                 `scaffold.build_turn_prompt` usually has — needs none, and the default is therefore `None`.
    """
    history = copy.deepcopy(history)
    end_idx = -1 if continue_ else None  # Don't touch the current AI message when continuing; else process all.
    messages = history[:end_idx]  # aliases the same dicts, so mutating a message below mutates `history`

    # Pass 1: scrub the text, and collect the attached documents' plaintext. A `text_file` part has no native
    # wire form, so its text (extracted on demand from the sidecar) has to ride as message text — which is why
    # any model can use an attached document, no vision capability required. Extraction needs `datastore`;
    # without it (throwaway tasks / prefill on attachment-free chats) there are no `text_file` parts anyway.
    scrubbed_texts = []
    attachments = []  # (message index, display name, extracted text, budget kind)
    for message_index, message in enumerate(messages):
        scrubbed_texts.append(chatutil.scrub(persona=settings.personas.get(message["role"], None),
                                             text=chatutil.content_to_text(message["content"]),
                                             thoughts_mode="discard",
                                             markup=None,
                                             add_persona=True))
        if datastore is None:
            continue
        for part in message["content"]:
            if isinstance(part, dict) and part.get("type") == "text_file":
                url = part.get("text_file", {}).get("url", "")
                name = part.get("text_file", {}).get("name") or "attached file"
                if url.startswith(sidecarstore.SIDECAR_SCHEME):
                    attachments.append((message_index, name, textfilestore.sidecar_to_text(datastore, url),
                                        attachment_budget_kind(part)))

    # Size all the attachments against one budget, then hand each message back its own share.
    fitted_texts = fit_attachments_to_context(settings,
                                              conversation_characters=sum(len(text) for text in scrubbed_texts),
                                              attachments=[(text, kind) for _, _, text, kind in attachments])
    file_blocks = collections.defaultdict(list)
    for (message_index, name, _, _kind), fitted_text in zip(attachments, fitted_texts):
        if fitted_text:
            file_blocks[message_index].append(f"[Attached file: {name}]\n{fitted_text}\n[End of attached file: {name}]")
        else:  # nothing left to give it - say so rather than let the document silently disappear
            file_blocks[message_index].append(CANONICAL_ATTACHMENT_OMITTED.format(name=name))

    # Pass 2: rebuild each message's content from the scrubbed text, its attachment blocks, and its images.
    for message_index, message in enumerate(messages):
        scrubbed_text = "\n\n".join([text for text in (scrubbed_texts[message_index], *file_blocks[message_index]) if text])

        new_content = [chatutil.text_content_part(scrubbed_text)]
        for part in message["content"]:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if datastore is not None and url.startswith(sidecarstore.SIDECAR_SCHEME):
                    part = chatutil.image_content_part(imagestore.sidecar_url_to_data_url(datastore, url))
                new_content.append(part)
        message["content"] = new_content
    return history

def _describe_strict_template_violations(history: List[Dict]) -> List[str]:
    """Describe any message shapes in `history` that strict chat templates reject. Empty list if none.

    Some chat templates enforce their message-ordering contract with a hard `raise_exception` rather
    than by ignoring the offending message, so a violation fails the *whole* request. The backend
    reports that as a template-parser failure ("Unable to generate parser for this template.
    Automatic parser generation failed"), which reads as a backend bug — the conversation we sent is
    nowhere in the message. Naming the offending shape, while we still know what we built, is what
    turns that into a quick diagnosis instead of a hunt through the backend.

    Describe, don't log: the caller checks the shape at the point of send, holds the result, and
    emits it only if the request is actually refused. Most backends are permissive, and a shape they
    accept is nothing to report — a warning that fires on every request (Raven's own idle context
    prefill sends `[system, greeting]`, which has no user message) teaches the reader to skip the
    line that would one day matter.

    Qwen3.5's template is the strict reference. It requires at least one user message, and permits
    exactly **one** system message, which must be the very first message — the guard is
    `{%- if message.role == "system" %}{%- if not loop.first %}`, so the second system message trips
    it no matter how early it appears. Read its error text with care: "System message must be at the
    beginning" means *be* the beginning, not merely *precede the conversation*. Several system
    messages in a row at the front are rejected just as firmly as one placed after a user turn.

    Qwen3.6's template dropped both guards, so a history that works on one model of a family can
    hard-fail on another.

    Describe, don't raise: which shapes a template accepts is the template's business. A refused
    request surfaces on its own; these descriptions only make the reason legible.
    """
    roles = [message["role"] for message in history]
    if not roles:
        return []
    role_sequence = ", ".join(roles)

    violations = []
    if "user" not in roles:
        violations.append(f"history has no user message; roles are [{role_sequence}]. Strict chat templates reject this.")
    if "system" in roles[1:]:
        violations.append(f"history has a system message that is not the first message; roles are [{role_sequence}]. Strict chat templates allow only one system message, as the very first one.")
    return violations

# --------------------------------------------------------------------------------
# Where an invocation's wall time went

def thinking_token_count(*,
                         reasoning_content: str,
                         n_tokens: int,
                         n_chunks: int,
                         n_chunks_at_first_content: int | None,
                         tokenizer: Any | None,
                         usage: dict | None) -> tuple[int | None, bool]:
    """How many of an invocation's `n_tokens` went into the thinking trace, and whether that is exact.

    `reasoning_content`: the accumulated thinking trace. Empty means the model did not think, which is
                         reported as `(None, False)` — the one case where the first element is `None`.
    `n_tokens`: the invocation's total completion token count.
    `n_chunks`: how many text-bearing deltas arrived in total.
    `n_chunks_at_first_content`: how many had arrived when the visible answer began, or `None` if it never
                                 began — a round that thought and then asked for a tool.
    `tokenizer`: the configured local tokenizer, or `None`.
    `usage`: the backend's token usage report for this call, or `None`.

    Returns `(tokens_of_thinking, is_exact)`.
    """
    if not reasoning_content:
        return None, False

    # The backend's own split, where it reports one — OpenAI's o-series spelling. Tried rather than relied
    # on: a local backend may leave it out, and no local one is known to fill it in.
    details = (usage or {}).get("completion_tokens_details") or {}
    if details.get("reasoning_tokens") is not None:
        return details["reasoning_tokens"], True

    # Failing that, count the trace with the model's own vocabulary.
    if tokenizer is not None:
        return len(tokenizer.encode(reasoning_content)), True

    # Failing both, apportion the total by where the visible answer began. A streaming backend emits one
    # text-bearing delta per token, so the ratio is close — but it is a ratio, and the readout says so.
    if n_chunks_at_first_content is None:  # the answer never began, so everything generated was thinking
        return n_tokens, False  # inexact even so: a tool call's own tokens are in `n_tokens` and not in the trace
    return round(n_tokens * n_chunks_at_first_content / n_chunks), False

def phase_report(*,
                 dt: float,
                 t0: float,
                 t_first_token: float | None,
                 t_first_content: float | None,
                 maybe_thinking_tokens: int | None,
                 thinking_tokens_exact: bool) -> dict | None:
    """Split an invocation's wall time into prompt processing and thinking, in the shape that is stored.

    `dt`: the whole invocation's wall time.
    `t0`: `perf_counter` at its start — `timer.t0`, so that the phases and the `dt` they are stored beside
          are measured off one clock and compose exactly.
    `t_first_token`: when the first generated text arrived on any channel, or `None` if none did.
    `t_first_content`: when the visible answer began, or `None` if it never did.
    `maybe_thinking_tokens`: the thinking trace's token count, or `None` if the model did not think.
    `thinking_tokens_exact`: whether that count is a count rather than an estimate.

    Returns `{"prefill": {"dt": ...}, "thinking": {"dt": ..., "n_tokens": ..., "tokens_exact": ...}}`, with
    `thinking` absent when the model did not think, or `None` when there is nothing to report at all.

    The answer phase is deliberately not among them: it is whatever remains of `dt`, so there is no third
    number that could disagree with the other two. Prompt processing is a phase of its own rather than part
    of thinking because nothing is being generated during it — how long it takes says how much of the prompt
    the backend's cache did not already hold, which is a different fact about the turn.
    """
    if t_first_token is None:  # nothing was generated as text: a round that asked for a tool and said nothing
        return None

    # An event flushed out of the parser at stream end is timestamped after the timer has already stopped,
    # so the samples are pulled back into the interval before any subtraction rather than each duration
    # being repaired afterwards. That way the phases cannot sum past `dt`.
    end = t0 + dt
    def clamp(t: float) -> float:
        return min(max(t, t0), end)

    phases = {"prefill": {"dt": clamp(t_first_token) - t0}}
    if maybe_thinking_tokens is not None:
        thinking_ended = clamp(t_first_content) if t_first_content is not None else end
        phases["thinking"] = {"dt": thinking_ended - clamp(t_first_token),
                              "n_tokens": maybe_thinking_tokens,
                              "tokens_exact": thinking_tokens_exact}
    return phases

def invoke(settings: env,
           history: List[Dict],
           on_progress: Optional[Callable] = None,
           on_prompt_ready: Optional[Callable] = None,
           tools_enabled: bool = True,
           tool_names: Optional[Collection[str]] = None,
           continue_: bool = False,
           max_tokens: Optional[int] = None,
           datastore: Optional[chattree.PersistentForest] = None,
           calibrate: bool = True) -> env:
    """Invoke the LLM with the given chat history.

    This is typically done after adding the user's message to the chat history, to ask the LLM to generate a reply.

    This is mainly meant as a low-level building block.

    If you just need to script the LLM (perform a throwaway task without storing the chat history),
    see `raven.librarian.agent.turn`, which runs the agent loop and hands back a record of what it did.

    `settings`: Obtain this by calling `setup()` at app start time.

    `history`: List of chat messages, where each message is in OpenAI format (with "role" and "content" fields,
               and an optional "tool_calls" field). See `raven.librarian.chatutil.create_chat_message`.

    `on_prompt_ready`: 1-argument callable, with argument `history: List[Dict]`. Debug/info hook.
                       The return value is ignored.

                       Called after the LLM context has been completely prepared, before sending it to the LLM.

                       This is the modified history, after scrubbing thought blocks.

                       Each element of the list is a chat message in the format accepted by the LLM backend,
                       with "role" and "content" fields.

    `on_progress`: 1-argument callable with argument `event: Dict`, a typed event from the parsed response
                   stream (`invoke` is the single parser; consumers dispatch on `event["type"]`
                   and never sniff raw text). Called while streaming, typically once per generated token. The
                   event is one of:

           `{"type": "content",   "text": str, "n_chunks": int}`: a piece of the visible answer.
           `{"type": "reasoning", "text": str, "n_chunks": int}`: a piece of the thinking trace — whether it
                            arrived via the native `reasoning_content` channel or as an inline `<think>` block
                            (both unified here). Render as a thought bubble, not as the answer.
           `{"type": "tool_call", "id": str, "name": str, "arguments": str}`: one completed tool call (emitted
                            once, deduped across inline-tag and native channels). `arguments` is a JSON string.

           `n_chunks` (on content / reasoning events) is how many chunks have been generated so far this
           invocation — useful for live UI throttling.

           Return value: `action_ack` to let the LLM keep generating, `action_stop` to interrupt and finish
           forcibly (meaningful on content / reasoning events; ignored on tool-call events).

           If you interrupt the LLM by returning `action_stop`, normal finalization still takes place, and you'll get
           a chat message populated with the content received so far. It is up to the caller what to do with that data.

    `tools_enabled`: Whether the LLM is allowed to use the tools available in `llmclient.setup`.
                     This can be disabled e.g. to temporarily turn off websearch.

    `tool_names`: Which of those tools to offer, by name. `None` (default) offers all of them; a collection
                  of names offers only those. Ignored entirely when `tools_enabled` is `False`.

                  The available names are the keys of `settings.tool_entrypoints` (equivalently, the
                  `function.name` of each entry in `settings.tools`) — both built by `llmclient.setup`.
                  A name that matches no tool is logged as a warning and otherwise ignored, since the
                  alternative is a typo silently switching a tool off.

                  Needed because tool availability is not one switch: the document tools are gated on the
                  document database being enabled, while websearch is not, so the advertised list varies
                  per turn rather than being a property of the session. Keep the list *stable within a
                  turn* — tools appearing or vanishing between rounds of one agent loop is a shape models
                  read as noise.

                  A restriction that selects nothing removes the `tools` field entirely rather than sending
                  an empty list, which not every backend accepts.

    `continue_`: If `False` (default), generate a new AI message. Most of the time, this is what you want.
                 The new message is returned.

                 If `True`, continue an incomplete AI message. The last message in `history` should be the AI message
                 that you want the AI to continue. The updated (continued) message is returned.

    `max_tokens`: If given, override the configured generation length cap (`config.llm_sampler_config["max_tokens"]`)
                  for this one call. The main use is `prefill`, which sets it to a minimal value to measure the
                  prompt size and warm the backend KV cache without producing a real reply. `None` (default) keeps
                  the configured cap.

    `datastore`: The chat's `chattree.PersistentForest`, needed only when messages carry attachments: image
                 and document parts are stored as `sidecar:<filename>` references, and the wire copy resolves
                 them by reading the sidecar files (see `serialize_history_for_wire`). `None` (default) is
                 correct exactly when the history carries no such reference — which is the case for a
                 history built entirely out of plain strings, with nowhere to put one.
                 Pass the datastore whenever the history might carry attachments: an unresolved reference
                 travels verbatim, so the model receives the literal text `sidecar:<filename>` in place of
                 the attachment, and nothing reports that it happened.

    Returns an `unpythonic.env.env` WITHOUT adding the LLM's reply to `history`.

    The returned `env` has the following attributes:

        `data: dict`: The new message generated by the LLM (for the format, see `raven.librarian.chatutil.create_chat_message`).
                      If the text content begins with the assistant character's name (e.g. "AI: ..."), this is automatically stripped.
        `n_tokens: int`: Number of tokens emitted by the LLM (from the backend's `usage` when available,
                         else estimated from the streamed chunk count).
        `usage: Optional[dict]`: The backend's token `usage` stats for this call (`prompt_tokens`,
                                 `completion_tokens`, `total_tokens`), or `None` if the backend didn't report
                                 them (e.g. interrupted before the final chunk). `prompt_tokens` is the exact
                                 size of the whole prompt this turn — useful for the context-fill indicator.
        `dt: float`: Wall time elapsed for this invocation, in seconds.
        `phases: Optional[dict]`: Where that wall time went — see `phase_report`, whose return value this is.
                                  `None` when the model generated no text at all (a round that only asked
                                  for a tool).
        `interrupted: bool`: Whether the invocation was interrupted by the `on_progress` callback.
                             This is provided for convenience.
    """
    data = copy.deepcopy(settings.request_data)

    # Normalize message content for resend (see `serialize_history_for_wire`).
    history = serialize_history_for_wire(settings, history, continue_=continue_, datastore=datastore)

    # Held, not logged: on a refusal this is the diagnosis (see `_describe_strict_template_violations`).
    template_violations = _describe_strict_template_violations(history)

    def report_template_violations() -> None:
        """Emit the held diagnosis. Call from every path that reports a refused request, and only those.

        There are two such paths, because backends disagree on how to refuse: an HTTP error status, and —
        LM Studio's way, which is what a template rejection actually looks like here — HTTP 200 followed by
        an SSE error event mid-stream.
        """
        for violation in template_violations:  # a candidate cause, if the backend's template is a strict one
            logger.error(f"llmclient.invoke: {violation}")

    # Not mentioned in the oobabooga docs, but see:
    #  `text-generation-webui/extensions/openai/script.py`, function `openai_chat_completions`
    #  `text-generation-webui/extensions/openai/typing.py`, classes `ChatCompletionRequest` and `ChatCompletionRequestParams`
    #  `text-generation-webui/extensions/openai/completions.py`, function `chat_completions_common`
    data["continue_"] = continue_

    if max_tokens is not None:
        data["max_tokens"] = max_tokens  # override the configured generation cap (used by `prefill`)

    data["messages"] = history

    # Ask for token usage stats to be included in the stream. LM Studio / OpenAI require this opt-in (and send
    # usage in a final, otherwise-empty chunk); ooba sends usage unconditionally and ignores the field.
    data["stream_options"] = {"include_usage": True}

    if on_prompt_ready is not None:
        on_prompt_ready(history)

    if not tools_enabled:
        logger.info("llmclient.invoke: Tool calling is disabled. Stripping tool specifications from request.")
        data.pop("tools")  # Tools? What tools? (Pretend to LLM backend we don't have any -> no tool-calls.)
    elif tool_names is None:
        logger.info("llmclient.invoke: Tool calling is enabled. Providing tool specifications in request.")
        # The `tools` field is already in `settings.request_data`, so there's nothing to do. The backend builds
        # the tool-calling instructions from it, using the model's own chat template.
    else:
        unknown_names = set(tool_names) - {tool["function"]["name"] for tool in data["tools"]}
        if unknown_names:  # a typo here would silently switch a tool off, so say so rather than filter quietly
            logger.warning(f"llmclient.invoke: Ignoring unknown tool name(s) {sorted(unknown_names)} in `tool_names`; "
                           f"available: {sorted(settings.tool_entrypoints)}.")
        data["tools"] = [tool for tool in data["tools"] if tool["function"]["name"] in tool_names]
        logger.info(f"llmclient.invoke: Tool calling is enabled, restricted to {sorted(tool_names)}. "
                    f"Providing {len(data['tools'])} tool specification(s) in request.")
        if not data["tools"]:  # an empty `tools` list is not the same thing as no tools; some backends reject it
            data.pop("tools")

    stream_response = requests.post(f"{settings.backend_url}/v1/chat/completions", headers=headers, json=data, verify=False, stream=True, timeout=librarian_config.llm_network_timeout_streaming)

    if stream_response.status_code != 200:  # not "200 OK"?
        logger.error(f"LLM server returned error: {stream_response.status_code} {stream_response.reason}. Content of error response follows.")
        logger.error(stream_response.text)
        report_template_violations()
        raise RuntimeError(f"While calling LLM: HTTP {stream_response.status_code} {stream_response.reason}")

    client = sseclient.SSEClient(stream_response)
    def stop_generating():
        # The local LLM is OpenAI-compatible, so the same trick works - to tell the server to stop, just close the stream.
        # https://community.openai.com/t/interrupting-completion-stream-in-python/30628/7
        # Alternatively, in oobabooga, we could call the undocumented "/v1/internal/stop-generation" endpoint.
        client.close()

    # `invoke` is the single parser of the response stream: the `StreamParser` turns raw deltas — content,
    # the native `reasoning_content` channel, and inline `<think>` / `<tool_call>` tags alike — into typed
    # events. Consumers (`on_progress`) dispatch on event type; they never regex-sniff the text.
    parser = StreamParser()
    llm_output_text = io.StringIO()       # accumulates `content` events -> message["content"]
    reasoning_output_text = io.StringIO()  # accumulates `reasoning` events -> message["reasoning_content"]
    collected_tool_calls: List[Dict] = []  # `tool_call` events in arrival order -> message["tool_calls"]
    last_few_chunks = collections.deque([""] * 10)  # ring buffer over recent *content* for stopping-string checks; prepopulate with empties since `popleft` needs an element
    n_chunks = 0
    stopped = False  # whether one of the stop strings triggered
    interrupted = False  # whether the progress callback interrupted generation
    usage = None  # token usage stats, once the backend reports them (final chunk)
    stop = []  # which stopping strings matched at the break point (assigned inside the loop)

    # Phase boundaries within this call, for `phase_report` below. Sampled off `perf_counter` because that
    # is the clock `timer` uses, so the phases and the `dt` they are reported beside compose exactly.
    t_first_token = None            # first generated text on any channel: prompt processing ended here
    t_first_content = None          # first text of the visible answer: thinking ended here
    n_chunks_at_first_content = None  # ...and this is how many text-bearing deltas it took to get there

    # Streaming tool-call accumulator, keyed by `tool_calls[i].index`. Unifies ooba's whole-object-in-one-delta
    # with LM Studio's / OpenAI's incremental fragments (see `_accumulate_tool_call_delta`).
    tool_call_acc: Dict[int, Dict[str, str]] = {}

    def handle_event(parsed_event: Dict) -> sym:
        """Accumulate one typed event into the response, notify `on_progress`, return its action (default ack)."""
        nonlocal t_first_token, t_first_content, n_chunks_at_first_content
        etype = parsed_event["type"]
        # A tool call is not text and does not end prompt processing: its deltas arrive through the
        # structured accumulator, which is also why they are not counted as chunks.
        if etype in ("content", "reasoning") and t_first_token is None:
            t_first_token = time.perf_counter()
        if etype == "content":
            llm_output_text.write(parsed_event["text"])
            if t_first_content is None:
                t_first_content = time.perf_counter()
                n_chunks_at_first_content = n_chunks
        elif etype == "reasoning":
            reasoning_output_text.write(parsed_event["text"])
        elif etype == "tool_call":
            collected_tool_calls.append(parsed_event)
        if on_progress is not None:
            return on_progress({**parsed_event, "n_chunks": n_chunks})
        return action_ack

    try:
        with timer() as tim:
            for event in client.events():
                raw = event.data.strip()
                # LM Studio / OpenAI terminate the stream with a literal `data: [DONE]` sentinel (ooba doesn't).
                # It is not JSON, so skip it before `json.loads`.
                if raw == "[DONE]":
                    break
                payload = json.loads(raw)

                # LM Studio reports backend errors as HTTP 200 + an SSE `event: error` whose data is
                # `{"error": {"message": ...}}` with no `choices` (e.g. a model whose chat template fails to
                # render). Surface it instead of `KeyError`-ing on the missing `choices`. A usage-only final
                # chunk (from `stream_options.include_usage`) also has empty `choices`, but no error.
                if not payload.get("choices"):
                    if "error" in payload:
                        err = payload["error"]
                        error_text = err.get("message") if isinstance(err, dict) else str(err)
                        report_template_violations()
                        raise RuntimeError(f"LLM backend error: {error_text}")
                    if payload.get("usage"):
                        usage = payload["usage"]
                    continue
                if payload.get("usage"):
                    usage = payload["usage"]

                delta = payload["choices"][0]["delta"]
                # `or ""` coerces {absent, null, ""} all to "": standard OpenAI streaming sends `content: null`
                # (and `reasoning_content: null`) on the role-priming first delta and on tool-call deltas, and a
                # plain `.get(..., "")` returns `None` for the present-but-null case, which would crash the parser.
                content_chunk = delta.get("content") or ""
                reasoning_chunk = delta.get("reasoning_content") or ""  # native reasoning channel (llama.cpp / LM Studio)

                if delta.get("tool_calls"):
                    _accumulate_tool_call_delta(tool_call_acc, delta["tool_calls"])

                # Count a delta as a chunk when it carried any generated text (content or reasoning): keeps the
                # `n_chunks - 1` fallback token count meaningful, and feeds the GUI's chunk-rate throttle.
                if content_chunk or reasoning_chunk:
                    n_chunks += 1

                action = action_ack
                for parsed_event in parser.feed(content_chunk, reasoning_chunk):
                    if handle_event(parsed_event) is action_stop:
                        action = action_stop
                    # Stopping strings guard the *visible* answer (model talking as the user) — check on content only.
                    if parsed_event["type"] == "content":
                        last_few_chunks.append(parsed_event["text"])
                        last_few_chunks.popleft()

                recent_text = "".join(last_few_chunks)  # Note start-of-word LLM tokens begin with a space.
                stop = [stopping_string in recent_text for stopping_string in settings.stopping_strings]  # check which stopping strings match (if any)

                if any(stop):  # should stop due to a stopping string?
                    stop_generating()
                    stopped = True
                    break
                if action is action_stop:  # did the callback tell us to interrupt the LLM generation?
                    stop_generating()
                    interrupted = True
                    break
    except KeyboardInterrupt:  # on Ctrl+C, stop generating, and let the exception propagate
        stop_generating()
        raise
    except requests.exceptions.ChunkedEncodingError:
        logger.exception(f"invoke: Connection lost. Please check if your LLM backend is still alive (was at {settings.backend_url}). Original error message follows.")
        raise

    # Flush the parser's buffers (any unterminated trailing block) and emit native `tool_calls`-field calls not
    # already seen inline. Materialize the native accumulator only on a clean finish, matching the prior behaviour
    # of not attributing tool calls to a stopping-string-interrupted turn.
    native_tool_calls = None if stopped else _materialize_tool_calls(tool_call_acc)
    for parsed_event in parser.finalize(native_tool_calls):
        handle_event(parsed_event)

    llm_output_text = llm_output_text.getvalue()
    reasoning_content = reasoning_output_text.getvalue()

    if stopped:  # due to a stopping string
        # From the final LLM output, remove the longest suffix that is in the stopping strings
        matched_stopping_strings = [stopping_string for is_match, stopping_string in zip(stop, settings.stopping_strings) if is_match]
        assert matched_stopping_strings  # we only get here if at least one stopping string matches
        stopping_string_start_positions = [llm_output_text.rfind(match) for match in matched_stopping_strings]
        assert not any(start_position == -1 for start_position in stopping_string_start_positions)  # we only checked matching strings
        chop_position = min(stopping_string_start_positions)
        llm_output_text = llm_output_text[:chop_position]

    # Materialize the collected `tool_call` events (inline-parsed + deduped native) into OAI tool-call dicts.
    tool_calls = None
    if collected_tool_calls:
        tool_calls = [{"type": "function",
                       "function": {"name": ev["name"], "arguments": ev["arguments"]},
                       "id": ev["id"],
                       "index": str(idx)}
                      for idx, ev in enumerate(collected_tool_calls)]

    # Completion token count: prefer the backend's real `usage` (exact, server-side). With
    # `stream_options.include_usage` requested above, a normal completion reports it on both ooba and LM Studio,
    # so the fallbacks are reached only when an interrupt (stopping string / callback / Ctrl-C) closed the stream
    # before the final usage chunk, or a generic backend ignores the opt-in. Then: count the generated text with
    # a local tokenizer if one is configured (exact), else use the streamed delta count — `n_chunks` already
    # counts only text-bearing deltas (one ≈ one token), so the empty role-priming delta is excluded for free.
    #
    # Reasoning counts as generated text: the model spent tokens on it and `dt` covers the time it took, so
    # leaving it out understates the speed by however much of the turn was spent thinking — which on a
    # thinking model is most of it. The two channels are encoded separately rather than as one string: they
    # arrive as separate generations, and no token spans the boundary between them.
    if usage is not None and usage.get("completion_tokens") is not None:
        n_tokens = usage["completion_tokens"]
    elif settings.tokenizer is not None:
        n_tokens = len(settings.tokenizer.encode(llm_output_text)) + len(settings.tokenizer.encode(reasoning_content))
    else:
        n_tokens = n_chunks

    # Refine the tokens-per-character calibration from this call's real prompt usage (the estimate path in
    # `count_tokens`), and cross-check a configured local tokenizer against the backend: if the tokenizer counts
    # MORE tokens for the message content alone than the backend reported for the whole templated prompt, it
    # almost certainly doesn't match the served model.
    if usage is not None and usage.get("prompt_tokens"):
        prompt_content = "".join(chatutil.content_to_text(message.get("content")) for message in history)
        if prompt_content:
            if calibrate:
                settings.tokens_per_character = usage["prompt_tokens"] / len(prompt_content)
            if settings.tokenizer is not None:
                tokenizer_count = len(settings.tokenizer.encode(prompt_content))
                if tokenizer_count > usage["prompt_tokens"] * 1.1:
                    logger.warning(f"invoke: local tokenizer counted {tokenizer_count} tokens for the prompt content, exceeding the backend's reported {usage['prompt_tokens']} for the full templated prompt — the configured tokenizer likely does not match the served model; token counts may be wrong.")

    maybe_thinking_tokens, thinking_tokens_exact = thinking_token_count(reasoning_content=reasoning_content,
                                                                        n_tokens=n_tokens,
                                                                        n_chunks=n_chunks,
                                                                        n_chunks_at_first_content=n_chunks_at_first_content,
                                                                        tokenizer=settings.tokenizer,
                                                                        usage=usage)
    phases = phase_report(dt=tim.dt,
                          t0=tim.t0,
                          t_first_token=t_first_token,
                          t_first_content=t_first_content,
                          maybe_thinking_tokens=maybe_thinking_tokens,
                          thinking_tokens_exact=thinking_tokens_exact)

    message = chatutil.create_chat_message(llm_settings=settings,
                                           role="assistant",
                                           text=llm_output_text,
                                           add_persona=False,
                                           tool_calls=tool_calls,
                                           reasoning_content=(reasoning_content or None))
    return env(data=message,
               model=settings.model,
               n_tokens=n_tokens,
               usage=usage,
               dt=tim.dt,
               phases=phases,
               interrupted=interrupted)

# How far below the local estimate a backend's `prompt_tokens` may sit and still be believable as a count of
# the *whole* prompt. Measured 2026-08-24 (`investigations/prompt-size-cache-relative/`): on a prompt whose
# true size is ~88500 tokens — established offline from the served model's own tokenizer — the same backend
# reported 88524 on one day and 8745 on another, while the character-ratio estimate read 81158, about 8%
# low. So the believable and the absurd are an order of magnitude apart, and the estimate sits near the
# truth; half is the midpoint of that gap, and the log line in `prompt_size_report_looks_whole` is what
# would show it being wrong.
_WHOLE_PROMPT_MIN_FRACTION_OF_ESTIMATE = 0.5


def prompt_size_report_looks_whole(reported: int, estimate: int) -> bool:
    """Whether a backend's reported `prompt_tokens` is plausibly the size of the *whole* prompt.

    **It is not always.** Measured against LM Studio: a branch whose true prompt is ~88500 tokens reported
    8745 on one day and 88524 on another, for byte-identical content — an order of magnitude short in the
    first case, with nothing in the response saying so (no `prompt_tokens_details.cached_tokens`, just a
    smaller number).

    **Why it does that is not established**, which is why this only refuses a figure rather than trying to
    repair one. The obvious reading — that it counts what it had to process, the rest being cached — is
    contradicted twice over: a byte-identical prompt sent again reports the same figure, not a smaller one,
    and appending to a prompt moves the figure as a straight count would. See
    `investigations/prompt-size-cache-relative/`.

    Believed as-is, a short figure reads to a user as the conversation having shrunk — a chat with three
    attached papers showing 7% of the window instead of 68%.

    Where a local tokenizer is configured (`count_tokens` tier 1) none of this arises, since the backend is
    never asked; that is unavailable when the backend is on another machine, which is why this stays.

    `estimate`: the local character-ratio count for the same branch, from `count_branch_tokens`. Crude, and
                that is fine here: the two cases are an order of magnitude apart, so this only has to tell
                *far below* from *near*.
    """
    if estimate <= 0:  # nothing to compare against; take the backend at its word
        return True
    if reported >= _WHOLE_PROMPT_MIN_FRACTION_OF_ESTIMATE * estimate:
        return True
    logger.info(f"prompt_size_report_looks_whole: backend reported {reported} prompt tokens against a local estimate of {estimate}; "
                f"too far below to be a count of the whole prompt, so keeping the estimate.")
    return False


def prefill(settings: env,
            history: List[Dict],
            tools_enabled: bool = True,
            tool_names: Optional[Collection[str]] = None,
            datastore: Optional[chattree.PersistentForest] = None,
            calibrate: bool = True) -> Optional[env]:
    """Send `history` to the backend generating essentially no output. Returns the `invoke` env, or `None` on failure.

    Two purposes, both side effects of submitting the real prompt:

      1. **Prompt size.** The returned env's `usage["prompt_tokens"]` is the backend's own count of the whole
         templated prompt (system prompt + character card + history + tool definitions), which is the only figure
         available on backends with no offline token-count endpoint (LM Studio / generic). It upgrades the GUI
         context-fill indicator from a calibrated estimate (`~X%`) to the reported figure (`X%`) — after
         `prompt_size_report_looks_whole` has checked it, since it is not always about the whole prompt.

      2. **KV-cache warm-up.** The backend processes (prefills) the prompt, so when the user's next turn sends the
         same prefix, the expensive prompt-processing pass is already cached and generation starts sooner.

    `tools_enabled` and `tool_names` should both match the next turn's settings, so the tool definitions are counted
    (and cached) identically. A different tool list is a different prompt prefix: get it wrong and the warm-up warms
    a prefix the real turn never sends, which costs the reprocessing it was supposed to save.

    `datastore`: passed through to `invoke` so image attachments in `history` are resolved and counted in the
                 prompt size (see `invoke`). `None` for text-only chats.

    `calibrate`: passed through to `invoke`. Pass `False` when `history` is a probe rather than a real
                 conversation: a short prompt's `prompt_tokens` is mostly chat-template framing, so the ratio
                 it would imply is far too high for ordinary text.

    We cap generation at one token rather than zero: a single token is negligible compute, while `max_tokens == 0` is
    below the OpenAI-documented minimum and some backends reject it. The prompt-processing pass — the part that matters
    for both the count and the cache — happens regardless of the cap.

    Failures (backend down, template render error surfaced as an SSE error, ...) are logged and return `None`; callers
    keep showing the estimate. This is a best-effort enhancement, never load-bearing.
    """
    try:
        return invoke(settings,
                      history,
                      on_progress=None,
                      tools_enabled=tools_enabled,
                      tool_names=tool_names,
                      max_tokens=1,
                      datastore=datastore,
                      calibrate=calibrate)
    except Exception as exc:  # noqa: BLE001 -- best-effort; any failure just leaves the estimate in place
        logger.warning(f"prefill: backend prefill failed; keeping the token estimate. Reason {type(exc)}: {exc}")
        return None

# --------------------------------------------------------------------------------
# Console progress indicator

def make_console_progress_handler(progress_symbol: str) -> Callable:
    """Make an `on_progress` function that prints `progress_symbol` to `sys.stderr` every 10 chunks.

    The returned function works as an `on_progress` event handler in `invoke` and in
    `raven.librarian.agent.turn`, which see.

    Note that this is a convenience function for a common use case with command-line apps,
    where it can be important to show that the LLM is writing (i.e. that the backend has
    not crashed or errored out, when answering the user's request takes a long time).

    This progress function will never cancel the generation; it always returns `action_ack`.
    If you need something more customized, you'll need to supply a custom `on_progress` handler.
    """
    def console_progress(event: Dict) -> sym:
        """Progress indicator while the LLM is processing. Callback for `llmclient.invoke`."""
        n_chunks = event.get("n_chunks", 0)  # tool-call events carry no chunk count
        if (n_chunks == 1 or n_chunks % 10 == 0):  # in any message being written by the AI, print a progress symbol for the first chunk, and then again every 10 chunks.
            print(progress_symbol, end="", file=sys.stderr)
            sys.stderr.flush()
        return action_ack  # let the LLM continue generating if it wants
    return console_progress

# --------------------------------------------------------------------------------
# For tool-using LLMs: tool-calling
