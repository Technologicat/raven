"""LLM client low-level library functions for Raven.

See `raven.librarian.scaffold` for the higher-level scaffolding that goes on top of this,
e.g. automatically applying tool-calls.

For an example chat client built using these, see `raven.librarian.minichat`.

NOTE for oobabooga/text-generation-webui users:

If you want to see the final prompt in instruct or chat mode, start your server in `--verbose` mode.
"""

__all__ = ["list_models",
           "test_connection",
           "detect_backend_flavor",
           "setup",
           "count_tokens",
           "image_token_cost",
           "StreamParser",
           "invoke", "prefill", "action_ack", "action_stop",
           "perform_throwaway_task", "make_console_progress_handler",
           "perform_tool_calls",
           "approve_host_for_session"]

import logging
logger = logging.getLogger(__name__)

import collections
import copy
import io
import json
import os
import requests
import sys
from typing import Callable, Dict, List, Optional, Tuple, Union

import sseclient  # pip install sseclient-py

from mcpyrate import colorizer

from unpythonic import dyn, make_dynvar, si_prefix, sym, timer
from unpythonic.env import env

from ..client import api
from ..client import config as client_config
from ..common import netutil
from ..common import text as common_text

from . import chattree
from . import chatutil
from . import config as librarian_config
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
# Module bootup

api.initialize(raven_server_url=client_config.raven_server_url,
               raven_api_key_file=client_config.raven_api_key_file)  # let it create a default executor

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

def websearch_wrapper(query: str,
                      engine: Optional[str] = None) -> List[Dict[str, str]]:
    """Perform a websearch via Raven-server; return the results as content parts, one text part per result.

    `engine`: search backend, "duckduckgo" or "google". `None` (the default) uses the configured
              `librarian_config.websearch_engine`. The LLM's websearch tool doesn't pass this — engine choice
              is host configuration, not a model decision.

    Each result becomes a single markdown text part — a `[title](link)` heading followed by the snippet. The
    GUI renders one part per result (clickable markdown links); the model reads the same markdown text on the
    wire.

    Every text-bearing field (`text`, `title`, `link`) is run through `raven.common.text.normalize`: SERP
    snippets are scraped HTML from the search engine — external untrusted content, the same hostile-input class
    that motivated the normalizer (it strips invisible-injection glyphs and control characters). Normalizing
    the link too is deliberate: a URL carrying zero-width characters is exactly what we want cleaned.
    """
    if engine is None:
        engine = librarian_config.websearch_engine
    websearch_results = api.websearch_search(query,
                                             engine,
                                             librarian_config.web_num_results)  # -> {"results": preformatted_text, "data": structured_results}
    structured_results = websearch_results["data"]

    def format_result_part(result: Dict[str, str]) -> Dict[str, str]:
        text = common_text.normalize(result.get("text", ""))
        title = common_text.normalize(result.get("title", ""))
        link = common_text.normalize(result.get("link", ""))
        if title and link:
            heading = f"[{title}]({link})"
        elif title:
            heading = title
        elif link:
            heading = f"<{link}>"  # bare-URL autolink (markdown)
        else:
            heading = None
        body = f"{heading}\n\n{text}\n" if heading else f"{text}\n"
        return chatutil.text_content_part(body)

    return [format_result_part(result) for result in structured_results]

# --------------------------------------------------------------------------------
# Webfetch integration (requires `raven.server` to be running)

# Per-turn "request context" passed to tool entrypoints, in the manner of Racket's `parameterize`
# (https://docs.racket-lang.org/reference/parameters.html) or Flask's request-global `g`: state
# that comes from the harness, not the model, scoped to one agent turn. `raven.librarian.scaffold`
# binds it (via `dyn.let`) around the agent loop's tool dispatch; an entrypoint that needs
# harness-supplied (NOT model-supplied) context reads it here. The model never sees or sets it —
# that separation is the point: a host the user auto-allowed must not be something the LLM can
# forge through its tool-call arguments.
#
# Keep this to a single `dyn.tool_context` env that grows fields over time — one request-context
# object, never a scatter of dyn vars.
#
# Fields currently carried (and the entrypoint that reads each):
#   webfetch_allowed_hosts : frozenset[str]  — hosts auto-allowed for this turn (URLs the user typed,
#                                              plus, if `webfetch_trust_search_results`, this turn's
#                                              websearch-result hosts). Read by `webfetch_wrapper`.
#                                              Absent -> treated as empty (fail closed: no auto-allow).
#
# The process-wide default (an empty env) means a thread that never entered a `dyn.let` — e.g. a
# direct unit-test call of an entrypoint — still reads a valid, empty context instead of erroring.
make_dynvar(tool_context=env())

# Canonical user-facing string for an allowlist refusal — the client-side counterpart to the
# server-side SSRF / scheme / SPA strings in `raven.server.modules.webfetch`. Pre-templated so the
# model copies it verbatim instead of improvising an explanation.
CANONICAL_NOT_ON_ALLOWLIST = ("The host {host} is not on the configured allowlist. The user can add it to the "
                              "webfetch_allowlist setting if you should be able to access this site.")

# Hosts the user has explicitly approved during this session (in-memory; NOT persisted). Populated by
# the GUI "allow this fetch" override when the user approves a host that `webfetch` denied. Consulted
# by `webfetch_wrapper`'s gate alongside the configured allowlist and the per-turn auto-allow set.
# Session-scoped by design: persisting approvals is deferred to a future JSON-config migration — we do
# NOT programmatically rewrite the `.py` config files (that reads as dangerous and is fragile).
_session_approved_hosts: set[str] = set()

def approve_host_for_session(host: str) -> None:
    """Approve `host` for `webfetch` for the rest of this session (in-memory, not persisted).

    Used by the GUI override when the user allows a host the allowlist denied. Afterward,
    `webfetch_wrapper` fetches from `host` even if it is not on `librarian_config.webfetch_allowlist`.
    """
    _session_approved_hosts.add(host.lower())

# !!! DO NOT memoize `webfetch_wrapper` (or anything that wraps it). !!!
#
# It is deliberately IMPURE: its result depends on two pieces of hidden state that are NOT in its
# argument list — `dyn.tool_context.webfetch_allowed_hosts` (per-turn, set by the harness) and the
# `_session_approved_hosts` module global (mutated by `approve_host_for_session`). A `@memoize` keys
# on `url` alone, so it would:
#   - cache a denial forever, so the GUI "approve host & retry" override would re-serve the stale
#     refusal even after the user approved the host (the whole override mechanism would silently break); and
#   - cache a per-turn auto-allow, leaking a one-turn permission into later turns.
# The gate is a security boundary; memoizing it turns a transient decision into a permanent one.
#
# This composes safely with the @memoize that DOES exist (server-side `websearch`,
# `raven.server.modules.websearch`) precisely because the two never touch: the memoized function
# (websearch) does not read the allowlist, and the allowlist-reading function (this one) is not
# memoized. Keep it that way.
def webfetch_wrapper(url: str) -> str | tuple[str, dict]:
    """Fetch a web page's main content, gated by the client-side domain allowlist.

    Tool entrypoint for the LLM's `webfetch` tool. Enforces the allowlist policy (which constrains
    the AI's *initiative*), then delegates the actual fetch to Raven-server, which enforces the
    network-level safety (SSRF / scheme blocking) and does the two-tier extraction.

    Reads `dyn.tool_context.webfetch_allowed_hosts` — the per-turn set of hosts the user auto-allowed
    by typing their URLs this turn (and, with `librarian_config.webfetch_trust_search_results`, this
    turn's websearch-result hosts). `raven.librarian.scaffold` binds `tool_context` around the agent
    loop's tool dispatch; the set itself is computed by `chatutil.compute_auto_allowed_hosts`.
    """
    host = netutil.url_host(url)

    # Allowlist gate. `None` means unrestricted (subject only to the server-side network checks); when
    # a list is configured, the host must be on it, auto-allowed by the user this turn, or approved by
    # the user earlier this session (via the GUI override).
    allowlist = librarian_config.webfetch_allowlist
    if allowlist is not None:
        auto_allowed_hosts = getattr(dyn.tool_context, "webfetch_allowed_hosts", frozenset())
        if not (netutil.host_matches_allowlist(host, allowlist) or host in auto_allowed_hosts or host in _session_approved_hosts):
            logger.info(f"webfetch_wrapper: refusing '{url}': host '{host}' not on allowlist, not user-allowed this turn, not session-approved.")
            # Structured return: the canonical refusal for the model, plus metadata the GUI override reads
            # (on the resulting tool node) to offer "approve this host" and re-run with the fetch allowed.
            return (CANONICAL_NOT_ON_ALLOWLIST.format(host=(host or "(none)")),
                    {"webfetch_denied_host": host})

    result = api.webfetch_fetch(url)  # server enforces SSRF/scheme, fetches, returns {"content", "url", "spaSuspected"}
    if result.get("spaSuspected"):
        logger.info(f"webfetch_wrapper: '{result.get('url', url)}' flagged spaSuspected (neither fetch tier extracted usable content).")
    return result["content"]

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
        print(colorizer.colorize(f"Cannot connect to LLM backend at {backend_url}.",
                                 colorizer.Style.BRIGHT, colorizer.Fore.RED) + " Is the LLM server running?")
        msg = f"Failed to connect to LLM backend at {backend_url}, reason {type(exc)}: {exc}"
        logger.error(msg)
        return False
    else:
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
    """
    if flavor == "oobabooga":
        # ooba reports the GGUF filename (fine — the model can interpret `name-size-quant.gguf` itself) but
        # not the active context length here; the latter falls through to the default in `setup`. It exposes no
        # VLM-capability flag either, so `is_vlm` is unknown (`None`).
        model_name = requests.get(f"{backend_url}/v1/internal/model/info", headers=headers, verify=False, timeout=librarian_config.llm_network_timeout).json().get("model_name")
        return env(label=model_name or NO_MODEL_INFO,
                   model_id=model_name,
                   context_length=None,
                   is_vlm=None)
    if flavor == "lmstudio":
        # `/api/v0/models` lists all downloaded models; exactly the `state == "loaded"` one is resident under
        # JIT, and only that record carries `loaded_context_length`. The record's `type` field is `"vlm"` for
        # vision models (vs `"llm"` / `"embeddings"`); vision is signaled there, not in `capabilities`.
        models = requests.get(f"{backend_url}/api/v0/models", headers=headers, verify=False, timeout=librarian_config.llm_network_timeout).json().get("data", [])
        loaded = [m for m in models if m.get("state") == "loaded"]
        if loaded:
            record = loaded[0]
            return env(label=_format_lmstudio_model_label(record),
                       model_id=record.get("id"),
                       context_length=record.get("loaded_context_length"),
                       is_vlm=(record.get("type") == "vlm"))
        # JIT idle: nothing resident right now. If the user named a model, trust that; else say so honestly.
        # Nothing loaded means no capability record to read, so `is_vlm` is unknown either way.
        if librarian_config.llm_model:
            return env(label=librarian_config.llm_model, model_id=librarian_config.llm_model, context_length=None, is_vlm=None)
        return env(label=NO_MODEL_INFO, model_id=None, context_length=None, is_vlm=None)
    # generic: best-effort from the standard list; never guess identity, and no capability field to read.
    ids = [m.get("id") for m in requests.get(f"{backend_url}/v1/models", headers=headers, verify=False, timeout=librarian_config.llm_network_timeout).json().get("data", [])]
    ids = [model_id for model_id in ids if model_id]
    if len(ids) == 1:
        return env(label=ids[0], model_id=ids[0], context_length=None, is_vlm=None)
    return env(label=NO_MODEL_INFO, model_id=librarian_config.llm_model, context_length=None, is_vlm=None)

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

        `backend_supports_continue: bool`: Whether the backend supports continuing an existing assistant message
                                           (ooba does, via an explicit flag; lmstudio/generic don't).

        `system_prompt: str`: Currently empty. Used to be a generic system prompt for the LLM (the LLaMA 3 preset from SillyTavern), to make it follow the character card.

        `character_card: str`: Character card that configures the AI to improve the model's performance.

        `stopping_strings: List[str]`: List of strings that automatically interrupt the AI in `invoke`.
                                       The default is `[f"\n{user}:"]`, which prevents old models' habit of speaking on the user's behalf.

                                       NOTE: Tool calls will not be processed if a stopping string is hit.

        `greeting: str`: The AI's first message, used for starting a new chat.

        `tools: List[Dict[str, Any]]`: JSON specifications of available tools (for LLMs capable of tool-calling).

        `tool_entrypoints: Dict[str, Callable]`: The Python functions that implement the tools.

        `backend_url: str`: The `backend_url` argument, as-is.

        `request_data: Dict[str, Any]`: Generation settings for the LLM backend.

        `personas: Dict[str, Optional[str]]`: Persona (character name) for each of the roles (dict keys) "user", "assistant", "system", and "tool".
                                              Used for constructing chat messages (see `raven.librarian.chatutil.create_chat_message`).

                                              The "system" and "tool" roles typically have no persona; for them, the persona is stored as `None`.
    """
    # Identify the backend, then resolve the loaded model's identity and context window for the character card.
    # A few request/response details differ between backends (see `detect_backend_flavor`, `_resolve_model_info`).
    backend_flavor = librarian_config.llm_backend_flavor or detect_backend_flavor(backend_url)
    model_info = _resolve_model_info(backend_url, backend_flavor)
    model = model_info.label  # human-facing identity for the character card (never a guess)
    request_model = librarian_config.llm_model or model_info.model_id  # id sent in requests (LM Studio JIT), or None

    # Context window: report the *loaded* length, never the model's theoretical max. When the backend doesn't
    # expose it (ooba doesn't here; a generic backend can't), default conservatively to 64k and warn — smaller
    # than that isn't useful for discussing a scientific fulltext, so we can assume at least that much.
    context_length = model_info.context_length
    if context_length is None:
        context_length = 64 * 1024
        logger.warning(f"setup: backend '{backend_flavor}' at {backend_url} did not report a loaded context length; defaulting to {context_length} tokens.")

    user = librarian_config.llm_user_name
    char = librarian_config.llm_char_name
    weekday_and_date = chatutil.format_chatlog_date_now()

    # SillyTavern would call these "macros".
    template_vars = env(user=user,
                        char=char,
                        model=model,
                        context_length=context_length,  # loaded context window, for the card to tell the model its real size
                        weekday_and_date=weekday_and_date)
    system_prompt = librarian_config.setup_system_prompt(template_vars)
    character_card = librarian_config.setup_character_card(template_vars)
    greeting = librarian_config.llm_greeting

    # Tools (functions) to make available to the AI for tool-calling (for models that support that - as of May 2025, at least Qwen 2 or later do).
    # These tools can be called by the LLM; see function `ai_turn` in `raven.librarian.scaffold`.
    #
    # For now, these are hardcoded, because Raven must provide the backends (at least an adaptor) for any tools it makes available to the LLM.
    tools = [
        {"type": "function",
         "function": {"name": "websearch",
                      "description": "Perform a web search.",
                      "parameters": {"type": "object",
                                     "required": ["query"],
                                     "properties": {"query": {"type": "string",
                                                              "description": "The search query."}}}}},
        {"type": "function",
         "function": {"name": "webfetch",
                      "description": "Retrieve a web page's main content as clean text.",
                      "parameters": {"type": "object",
                                     "additionalProperties": False,
                                     "required": ["url"],
                                     "properties": {"url": {"type": "string",
                                                            "description": "The URL to fetch."}}}}}
    ]
    tool_entrypoints = {"websearch": websearch_wrapper,
                        "webfetch": webfetch_wrapper}

    # Set up the chat completion request metadata template. Tool-calling instructions are NOT injected
    # client-side: every tool-capable model new enough to matter carries them in its own chat template, and the
    # backend builds them from the `tools` field below. `invoke` provides or strips `tools` per invocation.
    request_data = {
        "stream": True,  # stream each token to the client as it is generated, for live UI updates
        "messages": [],  # chat transcript including system messages; populated per-call by `invoke`
        "tools": tools,  # tools available for tool-calling, for models that support it
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

    # Token counting: load the optional local tokenizer for exact counts, and seed the char->token ratio used
    # by the estimate path (`count_tokens` tier 3) until real `usage` refines it (see `invoke`).
    tokenizer = _load_local_tokenizer(librarian_config.llm_tokenizer_path) if librarian_config.llm_tokenizer_path else None

    settings = env(user=user, char=char, model=model,
                   model_id=request_model,  # model id sent in requests (LM Studio JIT), or None
                   backend_flavor=backend_flavor,
                   context_length=context_length,  # loaded context window in tokens (backend-reported, or the 64k default)
                   model_is_vlm=model_info.is_vlm,  # whether the loaded model accepts image input: True/False, or None if unknown (gates image attach)
                   backend_supports_continue=(backend_flavor == "oobabooga"),  # ooba has an explicit continue flag; others don't
                   tokenizer=tokenizer,  # local HF tokenizer for exact counts, or None (see `count_tokens`)
                   char_to_token_ratio=_DEFAULT_CHAR_TO_TOKEN_RATIO,  # estimate-path calibration; refined from usage in `invoke`
                   system_prompt=system_prompt,
                   character_card=character_card,
                   stopping_strings=stopping_strings,
                   greeting=greeting,
                   tools=tools,  # for inspection
                   tool_entrypoints=tool_entrypoints,  # for our implementation to be able to call them
                   backend_url=backend_url,
                   request_data=request_data,
                   personas=personas)

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

    return settings

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

_DEFAULT_CHAR_TO_TOKEN_RATIO = 0.27  # tokens per character; rough English/markup default, refined from real usage
_tokenizer_cache = {}  # path -> loaded tokenizer (or None if loading failed); avoids reloading the same tokenizer

def _load_local_tokenizer(path: str):
    """Load (and cache) a local HuggingFace tokenizer for exact token counting, or return `None` on failure.

    `path` is a directory with `tokenizer.json` + `tokenizer_config.json`, or a HF repo id. Failures (missing
    files, network, version skew) are logged and degrade to the calibrated estimate rather than raising.
    """
    if path in _tokenizer_cache:
        return _tokenizer_cache[path]
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
      3. A calibrated char->token ratio (refined from each call's real `usage`; see `invoke`) — an *estimate*.
    The `is_exact` flag drives the GUI context-fill indicator's `X%` (exact) vs `~X%` (estimate) typography.
    Callers that only want the number use `count_tokens(...)[0]`.
    """
    if settings.tokenizer is not None:
        return len(settings.tokenizer.encode(text)), True
    if settings.backend_flavor == "oobabooga":
        return _ooba_token_count(settings.backend_url, text), True
    return round(len(text) * settings.char_to_token_ratio), False

def image_token_cost(settings: env, height: int, width: int) -> int:
    """Estimated token cost of one attached image for the loaded model — for the context-fill budget.

    A VLM image consumes a chunk of context that the text-only char->token ratio (`count_tokens` tier 3) can't
    see, so the pre-send indicator has to add it explicitly. The per-family costs live in
    `config.llm_image_token_cost`, keyed by a lowercase substring matched against the loaded model's id/family
    (first match wins; the `None` key is the fallback for unknown families). Each entry is a flat token count
    or a callable `(height, width) -> int` for models whose cost scales with resolution.

    Necessarily an *estimate*: it is a conservative published-scheme figure, refined away entirely once the
    backend reports the real `usage.prompt_tokens` for an image-bearing call (same self-correction path as the
    char->token ratio). `height`/`width` are the stored (wire) dimensions; they only matter for the
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

def _serialize_history_for_wire(settings: env,
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

    `continue_`: when `True`, the last message (the AI message being continued) is left exactly as-is — neither
                 scrubbed nor image/document-resolved (assistant continuations carry no attachments).
    """
    history = copy.deepcopy(history)
    end_idx = -1 if continue_ else None  # Don't touch the current AI message when continuing; else process all.
    for message in history[:end_idx]:
        scrubbed_text = chatutil.scrub(persona=settings.personas.get(message["role"], None),
                                       text=chatutil.content_to_text(message["content"]),
                                       thoughts_mode="discard",
                                       markup=None,
                                       add_persona=True)

        # Fold any attached documents into the message text. A `text_file` part has no native wire form, so its
        # plaintext (extracted on demand from the sidecar) rides as text under a clear header — which is why any
        # model can use an attached document, no vision capability required. Resolution needs `datastore`;
        # without it (throwaway tasks / prefill on attachment-free chats) there are no `text_file` parts anyway.
        if datastore is not None:
            file_blocks = []
            for part in message["content"]:
                if isinstance(part, dict) and part.get("type") == "text_file":
                    url = part.get("text_file", {}).get("url", "")
                    name = part.get("text_file", {}).get("name") or "attached file"
                    if url.startswith(sidecarstore.SIDECAR_SCHEME):
                        doc_text = textfilestore.sidecar_to_text(datastore, url)
                        file_blocks.append(f"[Attached file: {name}]\n{doc_text}\n[End of attached file: {name}]")
            if file_blocks:
                scrubbed_text = "\n\n".join([scrubbed_text, *file_blocks]) if scrubbed_text else "\n\n".join(file_blocks)

        new_content = [chatutil.text_content_part(scrubbed_text)]
        for part in message["content"]:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if datastore is not None and url.startswith(sidecarstore.SIDECAR_SCHEME):
                    part = chatutil.image_content_part(imagestore.sidecar_url_to_data_url(datastore, url))
                new_content.append(part)
        message["content"] = new_content
    return history

def invoke(settings: env,
           history: List[Dict],
           on_progress: Optional[Callable] = None,
           on_prompt_ready: Optional[Callable] = None,
           tools_enabled: bool = True,
           continue_: bool = False,
           max_tokens: Optional[int] = None,
           datastore: Optional[chattree.PersistentForest] = None) -> env:
    """Invoke the LLM with the given chat history.

    This is typically done after adding the user's message to the chat history, to ask the LLM to generate a reply.

    This is mainly meant as a low-level building block.

    If you just need to script the LLM (perform a throwaway task without storing the chat history),
    see `chatutil.oneshot_llm_task`.

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

    `continue_`: If `False` (default), generate a new AI message. Most of the time, this is what you want.
                 The new message is returned.

                 If `True`, continue an incomplete AI message. The last message in `history` should be the AI message
                 that you want the AI to continue. The updated (continued) message is returned.

    `max_tokens`: If given, override the configured generation length cap (`config.llm_sampler_config["max_tokens"]`)
                  for this one call. The main use is `prefill`, which sets it to a minimal value to measure the
                  prompt size and warm the backend KV cache without producing a real reply. `None` (default) keeps
                  the configured cap.

    `datastore`: The chat's `chattree.PersistentForest`, needed only when messages carry image attachments:
                 image parts are stored as `sidecar:<filename>` references, and the wire copy resolves them to
                 `data:` URLs by reading the sidecar files (see `_serialize_history_for_wire`). `None` (default)
                 is correct for text-only callers (throwaway tasks); sidecar URLs then pass through unresolved,
                 which is harmless because such callers carry no image parts.

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
        `interrupted: bool`: Whether the invocation was interrupted by the `on_progress` callback.
                             This is provided for convenience.
    """
    data = copy.deepcopy(settings.request_data)

    # Normalize message content for resend (see `_serialize_history_for_wire`).
    history = _serialize_history_for_wire(settings, history, continue_=continue_, datastore=datastore)

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

    if tools_enabled:
        logger.info("llmclient.invoke: Tool calling is enabled. Providing tool specifications in request.")
        # The `tools` field is already in `settings.request_data`, so there's nothing to do. The backend builds
        # the tool-calling instructions from it, using the model's own chat template.
    else:
        logger.info("llmclient.invoke: Tool calling is disabled. Stripping tool specifications from request.")
        data.pop("tools")  # Tools? What tools? (Pretend to LLM backend we don't have any -> no tool-calls.)

    stream_response = requests.post(f"{settings.backend_url}/v1/chat/completions", headers=headers, json=data, verify=False, stream=True, timeout=librarian_config.llm_network_timeout_streaming)

    if stream_response.status_code != 200:  # not "200 OK"?
        logger.error(f"LLM server returned error: {stream_response.status_code} {stream_response.reason}. Content of error response follows.")
        logger.error(stream_response.text)
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

    # Streaming tool-call accumulator, keyed by `tool_calls[i].index`. Unifies ooba's whole-object-in-one-delta
    # with LM Studio's / OpenAI's incremental fragments (see `_accumulate_tool_call_delta`).
    tool_call_acc: Dict[int, Dict[str, str]] = {}

    def handle_event(parsed_event: Dict) -> sym:
        """Accumulate one typed event into the response, notify `on_progress`, return its action (default ack)."""
        etype = parsed_event["type"]
        if etype == "content":
            llm_output_text.write(parsed_event["text"])
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
    if usage is not None and usage.get("completion_tokens") is not None:
        n_tokens = usage["completion_tokens"]
    elif settings.tokenizer is not None:
        n_tokens = len(settings.tokenizer.encode(llm_output_text))
    else:
        n_tokens = n_chunks

    # Refine the char->token calibration from this call's real prompt usage (the estimate path in
    # `count_tokens`), and cross-check a configured local tokenizer against the backend: if the tokenizer counts
    # MORE tokens for the message content alone than the backend reported for the whole templated prompt, it
    # almost certainly doesn't match the served model.
    if usage is not None and usage.get("prompt_tokens"):
        prompt_content = "".join(chatutil.content_to_text(message.get("content")) for message in history)
        if prompt_content:
            settings.char_to_token_ratio = usage["prompt_tokens"] / len(prompt_content)
            if settings.tokenizer is not None:
                tokenizer_count = len(settings.tokenizer.encode(prompt_content))
                if tokenizer_count > usage["prompt_tokens"] * 1.1:
                    logger.warning(f"invoke: local tokenizer counted {tokenizer_count} tokens for the prompt content, exceeding the backend's reported {usage['prompt_tokens']} for the full templated prompt — the configured tokenizer likely does not match the served model; token counts may be wrong.")

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
               interrupted=interrupted)

def prefill(settings: env,
            history: List[Dict],
            tools_enabled: bool = True,
            datastore: Optional[chattree.PersistentForest] = None) -> Optional[env]:
    """Send `history` to the backend generating essentially no output. Returns the `invoke` env, or `None` on failure.

    Two purposes, both side effects of submitting the real prompt:

      1. **Exact prompt size.** The returned env's `usage["prompt_tokens"]` is the backend's own count of the whole
         templated prompt (system prompt + character card + history + tool definitions) — exact on every backend,
         including LM Studio / generic, which have no offline token-count endpoint. This upgrades the GUI context-fill
         indicator from a calibrated estimate (`~X%`) to the real figure (`X%`).

      2. **KV-cache warm-up.** The backend processes (prefills) the prompt, so when the user's next turn sends the
         same prefix, the expensive prompt-processing pass is already cached and generation starts sooner.

    `tools_enabled` should match the next turn's setting, so the tool definitions are counted (and cached) identically.

    `datastore`: passed through to `invoke` so image attachments in `history` are resolved and counted in the
                 prompt size (see `invoke`). `None` for text-only chats.

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
                      max_tokens=1,
                      datastore=datastore)
    except Exception as exc:  # noqa: BLE001 -- best-effort; any failure just leaves the estimate in place
        logger.warning(f"prefill: backend prefill failed; keeping the token estimate. Reason {type(exc)}: {exc}")
        return None

# --------------------------------------------------------------------------------
# Agentic workflow utility

def perform_throwaway_task(llm_settings: env,
                           instruction: str,
                           on_progress: Callable = None) -> Tuple[str, str]:
    """Perform a throwaway task on the LLM.

    That is, behave as if the user sent `instruction` in chat mode as the first and only message
    (after the system prompt and the AI's initial greeting).

    This facilitates old-style agentic workflows, where the tool-running loop is a hardcoded script.
    This way of interacting with an LLM is also known as the "workflow" LLM agent pattern.

    (Contrast modern LLM agent style, which lets the LLM decide which tools to run,
     as well as when to finish.)

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.

                    The task will use the system prompt and AI character as configured
                    in `llm_settings`.

    `instruction`: The user prompt. Task specification and input data for the LLM.
                   This is what the user would type in as a message to an LLM chat app.

    `on_progress`: Passed to `invoke`, which see.

    Returns the tuple `(raw_output_text, scrubbed_output_text)`, where:

         `scrubbed_output_text` is the LLM's final response to the task,
                                ready for feeding into the rest of your
                                text processing pipeline.

         `raw_output_text` contains the thinking trace, too (if running on
                           a thinking model). Useful for debugging/logging.
    """
    # Start with an empty chat history (non-persistent) with just the system prompt,
    # and the AI's initial greeting, as currently configured in `llm_settings`.
    datastore = chattree.Forest()
    root_node_id = chatutil.factory_reset_datastore(datastore, llm_settings)

    # Add `instruction` as the user's first message.
    request_node_id = datastore.create_node(payload=chatutil.create_payload(llm_settings=llm_settings,
                                                                            message=chatutil.create_chat_message(llm_settings=llm_settings,
                                                                                                                 role="user",
                                                                                                                 text=instruction)),
                                            parent_id=root_node_id)

    # Linearize and run.
    history = chatutil.linearize_chat(datastore, request_node_id)
    out = invoke(llm_settings,
                 history,
                 on_progress=on_progress,
                 tools_enabled=False)

    # Postprocess the AI's response.
    #
    # `invoke` now returns think-free `content` with the thinking trace separated into `reasoning_content`.
    # Reassemble the inline `<think>...</think>` form for `raw_output_text` to keep the
    # debugging/logging contract; the scrubbed final answer comes from the already-think-free content.
    content = chatutil.content_to_text(out.data["content"])
    reasoning = out.data.get("reasoning_content") or ""
    raw_output_text = f"<think>{reasoning}</think>\n{content}" if reasoning else content
    scrubbed_output_text = chatutil.scrub(persona=llm_settings.personas.get("assistant", None),
                                          text=content,
                                          thoughts_mode="discard",
                                          markup=None,
                                          add_persona=False)
    return raw_output_text, scrubbed_output_text

def make_console_progress_handler(progress_symbol: str) -> Callable:
    """Make an `on_progress` function that prints `progress_symbol` to `sys.stderr` every 10 chunks.

    The returned function works as an `on_progress` event handler in `invoke` and in
    `perform_throwaway_task`, which see.

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

def perform_tool_calls(settings: env,
                       message: Dict,
                       on_call_start: Optional[Callable],
                       on_call_done: Optional[Callable]) -> List[env]:
    """Perform tool calls as requested in `message["tool_calls"]`.

    Returns a list of chat payloads (where each message's `role="tool"`) containing the tool outputs,
    one for each tool call.

    If the "tool_calls" field of `message` is missing or if it is empty, return the empty list.

    `on_call_start`: 3-argument callable: `(tool_call_id: str, function_name: str, arguments: Dict[str, Any])`.

                     The return value of the event is ignored.

                     Called just before a tool call starts.

                     Only called if the request record was valid and it was possible to determine
                     the tool name and the arguments.

    `on_call_done`: 4-argument callable: `(tool_call_id: str, function_name: str, status: str, text: str)`.

                    `status` is "success" or "error".

                    `text` is the tool output (upon success), or the error message (upon error).

                    The return value of the event is ignored.

                    Called just after a tool call has completed.

                    In error cases that never got so far as to call the tool, `on_call_done`
                    may be called with no corresponding `on_call_start`, to report the error.

    Each returned `env` has the following attributes:

        `data`: dict, The new message containing the tool response (for the format, see `raven.librarian.chatutil.create_chat_message`).

        `status`: str, one of "success" or "error".

            When an error occurs, the text of the output message will describe the error instead,
            and the full error message is posted to the server's log at warning level.

            Even if a tool call errors out, processing continues with the remaining tool calls, if any.

        `tool_call_id`: str. The ID of the tool call, copied from the input `message`.
                       Missing if no ID was provided.

        `dt`: float, Wall time elapsed for the call, in seconds.
              Missing if something went wrong before the tool was called (usually, bad input).

    Usually the input `message` looks something like this::

        message = {'role': 'assistant',
                   'content': '',
                   'tool_calls': [{'type': 'function',
                                   'function': {'name': 'websearch',
                                                'arguments': '{"query": "Sharon Apple"}'},
                                   'id': 'call_m357947b',
                                   'index': '0'}],
                  }
    """
    if "tool_calls" not in message:
        logger.debug(f"perform_tool_calls: `tool_calls` field missing from message record. Data: {message}")
        return []

    tool_calls = message["tool_calls"]
    if not tool_calls:
        logger.debug("perform_tool_calls: No tool calls requested by the LLM.")
        return []
    plural_s = "s" if len(tool_calls) != 1 else ""
    logger.info(f"perform_tool_calls: The LLM requested {len(tool_calls)} tool call{plural_s}.")

    tool_response_records = []
    def add_tool_response_record(output: Union[str, List[Dict]], *,
                                 status: str,
                                 tool_call_id: Optional[str],
                                 function_name: Optional[str],
                                 dt: Optional[float],
                                 tool_metadata: Optional[Dict] = None) -> None:
        """Add a tool response record to `tool_response_records`.

        `output` is the tool result: either a plain string (wrapped as a single text content-part) or an
        already-built content-parts list — e.g. `websearch_wrapper`'s one-text-part-per-result output.
        Error reports are passed as plain strings.

        The record is an `unpythonic.env.env` with the following attributes:

            `data`: dict: chat message object, with `role="tool"` and `content` the content-parts list.

            `status`: str: Values "success" or "error" are recommended.

            `tool_call_id`: Optional[str]: ID of this tool call (can be matched against the `id` in the
                           `tool_calls` list of the AI chat message that spawned this call).

                           The ID should be included whenever it was present in the tool call request record.

            `function_name`: Optional[str]: Which tool was called (or at least attempted),
                             if the call got that far. If it didn't, this is `None`.

            `dt`: Optional[float]: Duration of this tool call, in seconds. Recommended to be included whenever
                                   the request was valid enough to actually proceed to call the function
                                   (so that the call timing can be measured).

            `tool_metadata`: Optional[Dict]: Structured metadata the entrypoint attached to this result
                             (by returning `(output, metadata)` instead of a bare `output`). The caller
                             (`scaffold`) merges it into the tool node's `generation_metadata`. Used e.g.
                             by `webfetch_wrapper` to record `webfetch_denied_host` for the GUI override.
        """
        content = chatutil.normalize_content(output)  # str -> single text part; parts list -> used verbatim
        tool_response_message = chatutil.create_message_from_parts("tool", content)
        record = env(data=tool_response_message,
                     status=status)
        if tool_call_id is not None:
            record.tool_call_id = tool_call_id
        if function_name is not None:
            record.function_name = function_name
        if dt is not None:
            record.dt = dt
        if tool_metadata is not None:
            record.tool_metadata = tool_metadata
        tool_response_records.append(record)
        if on_call_done is not None:
            try:
                on_call_done(tool_call_id, function_name, status, chatutil.content_to_text(content))
            except Exception:
                logger.warning(f"perform_tool_calls: {tool_call_id}: function '{function_name}': ignoring exception from event handler `on_call_done`", exc_info=True)

    for request_record in tool_calls:
        tool_call_id = request_record.get("id", None)

        if "type" not in request_record:
            # The response message is intended for the LLM, whereas the log message (with all technical details) goes into the log.
            logger.warning(f"perform_tool_calls: {tool_call_id}: missing 'type' field in request. Data: {request_record}")
            add_tool_response_record("Tool call failed. The request is missing the 'type' field.", status="error", tool_call_id=tool_call_id)
            continue
        if request_record["type"] != "function":
            logger.warning(f"perform_tool_calls: {tool_call_id}: unknown type '{request_record['type']}' in request, expected 'function'. Data: {request_record}")
            add_tool_response_record(f"Tool call failed. Unknown request type '{request_record['type']}'; expected 'function'.", status="error", tool_call_id=tool_call_id)
            continue
        if "function" not in request_record:
            logger.warning(f"perform_tool_calls: {tool_call_id}: missing 'function' field. Data: {request_record}")
            add_tool_response_record("Tool call failed. The request is missing the 'function' field.", status="error", tool_call_id=tool_call_id)
            continue

        function_record = request_record["function"]
        if "name" not in function_record:
            logger.warning(f"perform_tool_calls: {tool_call_id}: missing 'function.name' field in request. Data: {request_record}")
            add_tool_response_record("Tool call failed. The request's function record is missing the 'name' field.", status="error", tool_call_id=tool_call_id)
            continue

        function_name = function_record["name"]
        try:
            function = settings.tool_entrypoints[function_name]
        except KeyError:
            logger.warning(f"perform_tool_calls: {tool_call_id}: unknown function '{function_name}'.")
            add_tool_response_record(f"Tool call failed. Function not found: '{function_name}'.", status="error", tool_call_id=tool_call_id, function_name=function_name)
            continue

        if "arguments" in function_record:
            try:
                kwargs = json.loads(function_record["arguments"])
            except Exception:
                logger.warning(f"perform_tool_calls: {tool_call_id}: function '{function_name}': failed to parse JSON for arguments", exc_info=True)
                add_tool_response_record(f"Tool call failed. When calling '{function_name}', failed to parse the request's JSON for the function arguments.", status="error", tool_call_id=tool_call_id, function_name=function_name)
                continue
            else:
                logger.debug(f"perform_tool_calls: {tool_call_id}: calling '{function_name}' with arguments {kwargs}.")
        else:
            logger.debug(f"perform_tool_calls: {tool_call_id}: for function '{function_name}: The request's function record is missing the 'arguments' field. Calling without arguments.")
            kwargs = {}

        # TODO: websearch return format: for the chat history, need only the preformatted text, but for the eventual GUI, would be nice to have the links separately. Could use a new metadata field in the chat datastore for this.
        try:
            if on_call_start is not None:
                on_call_start(tool_call_id, function_name, kwargs)
        except Exception:
            logger.warning(f"perform_tool_calls: {tool_call_id}: function '{function_name}': ignoring exception from event handler `on_call_start`", exc_info=True)
        try:
            with timer() as tim:
                tool_output = function(**kwargs)
        except Exception as exc:
            logger.warning(f"perform_tool_calls: {tool_call_id}: function '{function_name}': exited with exception", exc_info=True)
            add_tool_response_record(f"Tool call failed. Function '{function_name}' exited with exception {type(exc)}: {exc}", status="error", tool_call_id=tool_call_id, function_name=function_name, dt=tim.dt)
        else:  # success!
            logger.debug(f"perform_tool_calls: {tool_call_id}: Function '{function_name}' returned successfully.")
            # An entrypoint returns its output as either a plain string (wrapped downstream as a single text
            # content-part) or a content-parts list (e.g. websearch's one-part-per-result output),
            # optionally wrapped in an `(output, metadata_dict)` tuple to attach structured metadata to the
            # tool-response node (e.g. webfetch records a denied host for the GUI override). `add_tool_response_record`
            # normalizes the output to a parts list either way.
            if isinstance(tool_output, tuple):
                tool_output_value, tool_metadata = tool_output
            else:
                tool_output_value, tool_metadata = tool_output, None
            add_tool_response_record(tool_output_value, status="success", tool_call_id=tool_call_id, function_name=function_name, dt=tim.dt, tool_metadata=tool_metadata)

    return tool_response_records
