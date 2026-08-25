"""The tools the LLM may call, and the registry that offers them.

A tool is three things that have to agree: a JSON schema the model is shown (`TOOLS`), a Python function
that runs when it asks (`TOOL_ENTRYPOINTS`), and a decision about whether to offer it this turn at all
(`maybe_tool_names_for_turn`, over `DOCUMENT_TOOL_NAMES` and `NETWORK_TOOL_NAMES`). Keeping the three in one
module is the point: a schema that names a function which is not registered, or a name gated in neither
group, is the kind of mismatch that shows up as a model calling something that does not exist.

`perform_tool_calls` is the other half — it takes what the model asked for, validates it against the
registry, dispatches, and packages the results as `role="tool"` messages.

**The tools decide what leaves this machine**, which is why the allowlist gating lives here rather than at
the call site: `webfetch` is the one tool that visits an address the *model* chose, so its refusal path
(`CANONICAL_NOT_ON_ALLOWLIST`), its per-session approvals (`approve_host_for_session`) and the standing
warning against memoizing it are all in one place, where a reader changing one of them sees the others.

Every canned string a tool can return is a `CANONICAL_*` constant here. They are what the *model* reads
when something is unavailable, so they are written to be acted on rather than merely reported.

Imported by `llmclient`, and not the reverse: this module reaches back into it exactly once, deferred, for
the token budget `fetch_document` has to fit its answer into.
"""

__all__ = ["TOOLS", "TOOL_ENTRYPOINTS", "DOCUMENT_TOOL_NAMES", "NETWORK_TOOL_NAMES",
           "EXTERNAL_SOURCE_TOOL_NAMES",
           "maybe_tool_names_for_turn",
           "perform_tool_calls",
           "approve_host_for_session",

           # What a tool says when it cannot do the thing. Public because the frontends and the tests
           # recognize them: they are matched, not just displayed.
           "CANONICAL_NOT_ON_ALLOWLIST",
           "CANONICAL_NO_DOCUMENT_DATABASE", "CANONICAL_NO_DOCUMENT_MATCHES",
           "CANONICAL_NO_SUCH_DOCUMENT", "CANONICAL_NOTHING_CONSULTED",
           "CANONICAL_NO_ROOM_TO_FETCH", "CANONICAL_BAD_EXPRESSION",

           # The document helpers, which read the retriever directly rather than through a tool call.
           "document_text", "document_path", "label_documents",

           # The entrypoints themselves. `TOOL_ENTRYPOINTS` is how the agent loop reaches them, but they are
           # named here too: the tests call them directly, and so does anything driving one tool on purpose.
           "websearch", "webfetch", "search_documents", "fetch_document",
           "get_current_time", "calculate", "list_consulted_documents"]

import logging
logger = logging.getLogger(__name__)

import ast
import json
import math
from typing import Any, Callable, TYPE_CHECKING

import simpleeval

from unpythonic import dyn, make_dynvar, timer, uniqify
from unpythonic.env import env

from ..common import netutil
from ..common import text as common_text

from . import chatutil
from . import config as librarian_config

# The retriever is duck-typed at every use site, so that this module does not acquire a runtime dependency
# on `hybridir`'s chromadb / bm25s / watchdog stack. Same arrangement as `llmclient` and `scaffold`.
if TYPE_CHECKING:
    from . import hybridir


# ------------------------------------------------------------------------------------------------
# What the model is offered
#
# The schemas the model sees, and which switch each tool answers to. Data only, so it leads: a reader
# arriving here should meet the catalogue before any of the implementations. `TOOL_ENTRYPOINTS`, the
# third member of this set, cannot join it — it maps each name to the *function*, so it has to wait
# until those exist, and sits directly below them.
#
# It could be built by name instead — resolved from this module's namespace, or accumulated by a
# decorator on each tool — which would let the whole catalogue lead, and would close the gap this
# module's docstring warns about, since registering and declaring would become one act. Considered
# 2026-08-25 and not done: it trades a dependency you can see for one you cannot, and the mismatch it
# would prevent has not actually happened. Worth revisiting if it ever does.
# ------------------------------------------------------------------------------------------------


TOOLS = [
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
                                                        "description": "The URL to fetch."}}}}},
    # The per-turn clock inject presents itself as a call to this tool, so it has to be a real one: a
    # synthetic call naming a function the model was never offered is a fiction the model can act on.
    # That is not a guess — it is the situation the document-matches inject was in before
    # `search_documents` existed, where the model wrote the call out as literal text and the user got
    # that instead of an answer, roughly one turn in three on Qwen3.6-27B.
    #
    # Registering it is correct on those grounds alone, and deliberately *not* claimed to fix the thing
    # that prompted the look: a 2026-08-07 trace where the model called the clock call "erroneous" on a
    # turn about arithmetic. Read again, that complaint is about *relevance* — "get_current_time is
    # useless for math" — not about the function being absent. The inject still arrives on every turn
    # whether or not the turn is about time, so a model inclined to remark on that will still remark.
    {"type": "function",
     "function": {"name": "get_current_time",
                  "description": "Get the current local time.",
                  "parameters": {"type": "object",
                                 "additionalProperties": False,
                                 "required": [],
                                 "properties": {}}}},
    {"type": "function",
     "function": {"name": "calculate",
                  "description": ("Evaluate an arithmetic expression exactly. Use this whenever the answer "
                                  "depends on a calculation being right, rather than working it out in your "
                                  "head."),
                  "parameters": {"type": "object",
                                 "additionalProperties": False,
                                 "required": ["expression"],
                                 "properties": {"expression": {"type": "string",
                                                               "description": ("The expression, in Python "
                                                                               "syntax; e.g. '2 + 2', "
                                                                               "'sqrt(2)', '(1 + 5) / 7'. "
                                                                               "Functions from the math "
                                                                               "module are available, as are "
                                                                               "pi and e.")}}}}},
    {"type": "function",
     "function": {"name": "search_documents",
                  "description": ("Search the user's local document database. Use this to look for material "
                                  "the conversation does not already contain, or to search again with a "
                                  "better query once you have seen what a first search returned."),
                  "parameters": {"type": "object",
                                 "additionalProperties": False,
                                 "required": ["query"],
                                 # No `k`: how many results come back is host configuration, not a model
                                 # decision. Keeping the surface to one required string also leaves the
                                 # fewest ways to emit a malformed call.
                                 "properties": {"query": {"type": "string",
                                                          "description": "Keywords or a natural-language question."}}}}},
    {"type": "function",
     "function": {"name": "fetch_document",
                  "description": ("Read a document from the user's local document database, by the ID "
                                  "reported in a search result. Use this when a search match looks "
                                  "relevant but you need more of the document around it."),
                  "parameters": {"type": "object",
                                 "additionalProperties": False,
                                 "required": ["document_id"],
                                 "properties": {"document_id": {"type": "string",
                                                                "description": "The document ID, as given in a search result."},
                                                # Spans are in characters because that is the unit the search
                                                # results report; a model left to guess would assume tokens.
                                                "offset": {"type": "integer",
                                                           "description": "Character offset to start reading from. Omit to start at the beginning."},
                                                "length": {"type": "integer",
                                                           "description": "How many characters to read. Omit to read to the end."}}}}},
    {"type": "function",
     "function": {"name": "list_consulted_documents",
                  "description": ("List the documents from the user's local document database that this "
                                  "conversation has already looked at. Use this when the discussion "
                                  "refers back to material that is no longer written out above; the "
                                  "list gives the IDs to read again with fetch_document."),
                  # No parameters at all: the list is a property of the conversation, and nothing about
                  # it is the model's to choose.
                  "parameters": {"type": "object",
                                 "additionalProperties": False,
                                 "properties": {}}}}
]

# The two gated groups, each answering to one user-facing toggle: "Documents" and "Internet". Callers gate
# with `invoke`'s `tool_names`; `maybe_tool_names_for_turn` assembles the per-turn list, and
# `raven.librarian.scaffold.ai_turn` calls it. Named here, next to the specs, so the two cannot drift.
#
# A tool in neither group is *ungated* — always offered, because no switch claims to govern it. That is
# `get_current_time` today, and it has to be: the clock inject is delivered on every turn regardless of
# either toggle, as a synthetic call to this very function. Withholding the spec while still sending the
# call would put the model back to reading a call to a function it cannot see, which is the defect
# registering the tool exists to fix.
DOCUMENT_TOOL_NAMES = frozenset({"search_documents", "fetch_document", "list_consulted_documents"})

NETWORK_TOOL_NAMES = frozenset({"websearch", "webfetch"})

# The tools that reach outside this conversation for material. A frontend can use this to say so - Librarian
# lights the avatar's "data eyes" while one runs.
#
# **The test for membership is "would a scifi AI system show this effect when doing this?"** (Juha,
# 2026-08-25), which is the right register for a signal whose job is to look like something is happening.
# It settles the cases quickly: consulting a database or the net, yes; adding two numbers or reading a
# clock, no - nobody's eyes flicker over 2+2, and a signal that fires for that stops meaning anything.
#
# So this is a list to extend deliberately, one tool at a time, rather than a rule to derive from what a
# tool technically touches.
EXTERNAL_SOURCE_TOOL_NAMES = NETWORK_TOOL_NAMES | DOCUMENT_TOOL_NAMES


# ------------------------------------------------------------------------------------------------
# Machinery the tools share
#
# Reaching the server, and the per-turn context a tool reads its budget and its formatters from.
# ------------------------------------------------------------------------------------------------


def _client_api():
    """Return `raven.client.api`, imported on first use and initialized.

    `initialize_api` is idempotent — first settings win, later calls are logged and ignored — so an app that
    wants its own executor (as `librarian.app` does) simply initializes at startup, before any tool can run.
    """
    from ..client import api  # noqa: PLC0415 -- deferred on purpose; see the note above
    from ..client import config as client_config  # noqa: PLC0415 -- same
    api.initialize(raven_server_url=client_config.raven_server_url,
                   raven_api_key_file=client_config.raven_api_key_file)  # let it create a default executor
    return api

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
# What the context carries, and why each field is there, is documented where it is built:
# `scaffold.make_tool_context`. Listing the fields here as well only produced a list that fell behind it.
# The one worth repeating is the security-relevant one:
#   webfetch_allowed_hosts : frozenset[str]  — hosts auto-allowed for this turn (URLs the user typed,
#                                              plus, if `webfetch_trust_search_results`, this turn's
#                                              websearch-result hosts). Read by `webfetch`.
#                                              Absent -> treated as empty (fail closed: no auto-allow).
#
# The process-wide default (an empty env) means a thread that never entered a `dyn.let` — e.g. a
# direct unit-test call of an entrypoint — still reads a valid, empty context instead of erroring.
make_dynvar(tool_context=env())

def _formatters() -> env:
    """The turn's model-facing formatters, from `dyn.tool_context`, falling back to the defaults.

    Entrypoints reach `settings` only through the tool context, and that context legitimately carries no
    settings at all: `scaffold.make_tool_context(llm_settings=None, ...)` is the documented shape for a
    caller that is not going to run a tool needing them. Two entrypoints here format their result without
    otherwise wanting settings, and they worked in that shape before formatters became overridable; falling
    back keeps them working rather than making them the reason a probe needs a full settings object.
    """
    settings = getattr(dyn.tool_context, "llm_settings", None)
    if settings is None or "formatters" not in settings:
        return chatutil.default_formatters()
    return settings.formatters


# ------------------------------------------------------------------------------------------------
# The tools that reach the network
#
# The two that leave this machine, and the allowlist that decides whether the second one may.
# ------------------------------------------------------------------------------------------------


def websearch(query: str,
              engine: str | None = None) -> list[dict[str, str]]:
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
    api = _client_api()
    websearch_results = api.websearch_search(query,
                                             engine,
                                             librarian_config.web_num_results)  # -> {"results": preformatted_text, "data": structured_results}
    structured_results = websearch_results["data"]

    def format_result_part(result: dict[str, str]) -> dict[str, str]:
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

# Canonical user-facing string for an allowlist refusal — the client-side counterpart to the
# server-side SSRF / scheme / SPA strings in `raven.server.modules.webfetch`. Pre-templated so the
# model copies it verbatim instead of improvising an explanation.
CANONICAL_NOT_ON_ALLOWLIST = ("The host {host} is not on the configured allowlist. The user can add it to the "
                              "webfetch_allowlist setting if you should be able to access this site.")

# Hosts the user has explicitly approved during this session (in-memory; NOT persisted). Populated by
# the GUI "allow this fetch" override when the user approves a host that `webfetch` denied. Consulted
# by `webfetch`'s gate alongside the configured allowlist and the per-turn auto-allow set.
# Session-scoped by design: persisting approvals is deferred to a future JSON-config migration — we do
# NOT programmatically rewrite the `.py` config files (that reads as dangerous and is fragile).
_session_approved_hosts: set[str] = set()

def approve_host_for_session(host: str) -> None:
    """Approve `host` for `webfetch` for the rest of this session (in-memory, not persisted).

    Used by the GUI override when the user allows a host the allowlist denied. Afterward,
    `webfetch` fetches from `host` even if it is not on `librarian_config.webfetch_allowlist`.
    """
    _session_approved_hosts.add(host.lower())

# !!! DO NOT memoize `webfetch` (or anything that wraps it). !!!
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
def webfetch(url: str) -> tuple[str, dict]:
    """Fetch a web page's main content, gated by the client-side domain allowlist.

    Tool entrypoint for the LLM's `webfetch` tool. Enforces the allowlist policy (which constrains
    the AI's *initiative*), then delegates the actual fetch to Raven-server, which enforces the
    network-level safety (SSRF / scheme blocking) and does the two-tier extraction.

    Returns `(what the model reads, metadata for the frontends)` on *every* path, refusals included —
    `perform_tool_calls` unpacks the pair and hands the second half on as `tool_metadata`. One of two
    keys is present: `fetched_document` when something was fetched, naming the URL the server actually
    landed on and the page title, which is what lets `scaffold` store a long page as an attachment
    rather than dumping it into the chat log; or `webfetch_denied_host` when the allowlist refused,
    which is what the GUI's "approve this host and retry" override reads off the tool node.

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
            logger.info(f"webfetch: refusing '{url}': host '{host}' not on allowlist, not user-allowed this turn, not session-approved.")
            # Structured return: the canonical refusal for the model, plus metadata the GUI override reads
            # (on the resulting tool node) to offer "approve this host" and re-run with the fetch allowed.
            return (CANONICAL_NOT_ON_ALLOWLIST.format(host=(host or "(none)")),
                    {"webfetch_denied_host": host})

    api = _client_api()
    result = api.webfetch_fetch(url)  # server enforces SSRF/scheme, fetches, returns {"content", "url", "spaSuspected", "title"}
    if result.get("spaSuspected"):
        logger.info(f"webfetch: '{result.get('url', url)}' flagged spaSuspected (neither fetch tier extracted usable content).")
    # Declare the result a fetched document, so `scaffold` can store a long one as an attachment sidecar
    # instead of dumping it into the chat log. Declared rather than inferred from the tool's name: the URL
    # the server actually ended up at (after rewriting and redirects) and the page's title are known only
    # here, and they are exactly what the sidecar's provenance and chip want.
    #
    # Declared on the refusal paths too — the network refusal, the HTTP error, the SPA notice. Those are
    # canonical one-sentence strings and so never reach the size threshold that decides whether to store
    # anything, which makes a special case for them machinery with no effect to have.
    effective_url = result.get("url") or url
    return (result["content"],
            {"fetched_document": {"url": effective_url,
                                  "name": result.get("title") or effective_url}})


# ------------------------------------------------------------------------------------------------
# The tools that read the document database
#
# Kept together so the canned replies they share sit above all of them, and the ones belonging to a
# single tool sit directly above it.
# ------------------------------------------------------------------------------------------------


# Canonical user-facing strings for the two ways a document search can come back empty-handed. Pre-templated
# so the model reports the situation in consistent words instead of improvising an explanation of Raven's
# internals - and deliberately phrased as *statements of fact*, never as prohibitions. A tool result that
# tells the model what it may not do is the shape that measured 29000 characters of deliberation without
# producing a reply; see `investigations/context-injects/context-inject-shape-measurements.md`.
CANONICAL_NO_DOCUMENT_DATABASE = ("The document database is not available in this conversation, so it cannot "
                                  "be searched.")

CANONICAL_NO_DOCUMENT_MATCHES = ("The document database contains no matches for that query. A differently "
                                 "worded query may match, since this search is over the documents' own wording.")

def search_documents(query: str) -> tuple[str, dict]:
    """Search the local document database (RAG); return the matches formatted for the LLM.

    Tool entrypoint for the LLM's `search_documents` tool - the model-driven counterpart to the automatic
    search `raven.librarian.scaffold` runs on the user's behalf before each turn. Both exist on purpose:
    the automatic one buys a zero-latency first pass from a cheap heuristic query, this one buys a query
    written by something that has *read* that first pass and can aim better.

    The retriever arrives through `dyn.tool_context` (harness-supplied, never model-supplied), because
    `llmclient.setup` runs before `hybridir.setup` in both clients and so no entrypoint can close over it
    at registration time. An absent retriever means the document database is not in play this turn - either
    the app has none, or the user has switched it off - and *that is the whole gate*: this function is
    reachable only if the harness handed it a retriever, so a model that calls the tool anyway (it is not
    advertised when unavailable) gets a plain refusal rather than access.

    Results are formatted by `chatutil.format_docs_matches`, the same formatter the automatic search uses,
    so a match reads identically whoever asked for it.

    Returns `(output, metadata)`. The output is text for the model to read: either the matches as
    `chatutil.format_docs_matches` renders them, or one of the two canonical sentences saying why there are
    none. The metadata declares whether the result is grounding material, which
    `scaffold._record_grounding` folds into the turn's state. Declaring it matters here because "no
    matches" is a perfectly non-empty string that grounds nothing at all. It also names which documents
    were reached and by what query, which is what lets a later turn list them once their text has scrolled
    out of the window (`scaffold._collect_consulted_documents`).
    """
    retriever = getattr(dyn.tool_context, "retriever", None)
    if retriever is None:
        logger.info("search_documents: no retriever in the tool context; document database not in play this turn.")
        return (CANONICAL_NO_DOCUMENT_DATABASE, {"grounding": False})

    matches = retriever.query(query,
                              k=librarian_config.docs_num_results,
                              max_span_length=librarian_config.docs_max_result_length,
                              return_extra_info=False)
    plural_s = "es" if len(matches) != 1 else ""
    logger.info(f"search_documents: {len(matches)} match{plural_s} for '{query}'.")
    if not matches:
        return (CANONICAL_NO_DOCUMENT_MATCHES, {"grounding": False, "docs_query": query})
    return (_formatters().docs_matches(matches),
            {"grounding": True,
             "docs_query": query,
             "document_ids": list(uniqify(match["document_id"] for match in matches))})

CANONICAL_NOTHING_CONSULTED = ("This conversation has not looked at any documents from the knowledge base yet. "
                               "Search for some with `search_documents`.")

def list_consulted_documents() -> tuple[str, dict]:
    """List the knowledge-base documents this conversation has already looked at, by ID.

    Tool entrypoint for the LLM's `list_consulted_documents` tool. It answers a question the transcript
    cannot: the automatic search injects its matches for one turn and then drops them, so a follow-up
    question arrives with the model's own earlier reply visible and the material it was based on gone. The
    IDs survive, and `fetch_document` turns one back into text. (A document the model *fetched* is a stored
    node and is still written out where the window reaches; the list covers both, because from here they
    are indistinguishable, and a pointer to something already in view costs one line.)

    Pointers, not text, and that is the point - re-injecting the material would grow without bound.
    Consequently this grounds *nothing*: a list of what one has read is not a thing one has read. The model
    that wants the material has to go and fetch it, which is exactly the step the badge is measuring.

    The list is assembled per turn by `scaffold` (which is where the branch is) and arrives through
    `dyn.tool_context`, in the same manner as the retriever.
    """
    entries = getattr(dyn.tool_context, "consulted_documents", None)
    if not entries:
        logger.info("list_consulted_documents: nothing consulted on this branch yet.")
        return (CANONICAL_NOTHING_CONSULTED, {"grounding": False})
    logger.info(f"list_consulted_documents: {len(entries)} document(s).")
    return (_formatters().consulted_documents(entries), {"grounding": False})

CANONICAL_NO_SUCH_DOCUMENT = ("There is no document with that ID in the database. Document IDs come from "
                              "search results; search first, then fetch by the ID a result reports.")

# Canonical refusal for a fetch that cannot fit, in the manner of `CANONICAL_NOT_ON_ALLOWLIST`. Phrased as a
# statement of the situation with the remedy attached, never as a prohibition: a tool result that tells the
# model what it may not do is the shape that measured 29000 characters of deliberation without producing a
# reply. See `investigations/context-injects/context-inject-shape-measurements.md`.
CANONICAL_NO_ROOM_TO_FETCH = ("There is not enough room left in this conversation to read that document. "
                              "Suggest starting a new chat if the user wants to work through it.")

def fetch_document(document_id: str,
                   offset: int | None = None,
                   length: int | None = None) -> tuple[str, dict]:
    """Read a document from the local database by ID; return its text, cut to what the context can hold.

    Tool entrypoint for the LLM's `fetch_document` tool — the internal-engine counterpart of `webfetch`,
    and the follow-up to `search_documents`. A search match is a window onto a larger document and reports
    where that window sits, so the model can come back for the surrounding text, or for the whole thing.

    `document_id` must come from a search result: the model has no other way to learn one, which makes this
    search-then-fetch by construction, exactly as websearch precedes webfetch.

    `offset`, `length`: character span to read, both optional. Omit both for the document from the start.
    Out-of-range values are clamped rather than refused — an off-by-a-bit span is an ordinary mistake to
    make about a document you have only seen one window of, and clamping answers the question that was
    meant instead of starting a correction round-trip.

    The text is fitted to what the conversation can still afford (`budget_for_fetched_text`), and truncated
    in the middle if it does not fit, so a long paper keeps its abstract and its conclusions. A fetch that
    cannot fit at all is refused with a canonical string rather than served as a sliver.

    Returns `(output, metadata)`, declaring grounding as `search_documents` does — including the refusals,
    which are non-empty strings that ground nothing.
    """
    retriever = getattr(dyn.tool_context, "retriever", None)
    if retriever is None:
        logger.info("fetch_document: no retriever in the tool context; document database not in play this turn.")
        return (CANONICAL_NO_DOCUMENT_DATABASE, {"grounding": False})

    text = document_text(retriever, document_id)
    if text is None:
        logger.info(f"fetch_document: no document with ID '{document_id}'.")
        return (CANONICAL_NO_SUCH_DOCUMENT, {"grounding": False})

    start = min(max(offset or 0, 0), len(text))
    end = min(start + length, len(text)) if length is not None and length > 0 else len(text)
    span = text[start:end]

    settings = dyn.tool_context.llm_settings
    # Deferred, and the only reach back into `llmclient`: it imports this module, so a
    # module-level import here would close the loop. How much room is left in the context is
    # its business — this tool just happens to be the one that has to fit inside the answer.
    from . import llmclient  # noqa: PLC0415 -- see above
    budget_tokens = llmclient.budget_for_fetched_text(settings, used_tokens=dyn.tool_context.used_tokens)
    fitted = llmclient.fit_text_to_token_budget(settings, span, budget_tokens)
    if not fitted:
        logger.info(f"fetch_document: no room to fetch '{document_id}' (budget {budget_tokens} tokens).")
        return (CANONICAL_NO_ROOM_TO_FETCH, {"grounding": False})

    logger.info(f"fetch_document: '{document_id}' [{start}:{end}] -> {len(fitted)} of {len(span)} characters.")
    header = (f"[System information: Document '{document_id}', characters {start} to {end} "
              f"of {len(text)} total.]")
    return (f"{header}\n\n{fitted}", {"grounding": True, "document_ids": [document_id]})

def document_text(retriever: "hybridir.HybridIR | None",
                  document_id: str) -> str | None:
    """Read one document's full text out of the retriever, or `None` if it has no such document.

    The lock is not optional: `documents` is rewritten wholesale when the retriever commits a batch of
    filesystem changes, so a read that races one sees a half-swapped mapping.
    """
    if retriever is None:
        return None
    with retriever.datastore_lock:
        document = retriever.documents.get(document_id)
        return document["text"] if document is not None else None

def document_path(retriever: "hybridir.HybridIR | None",
                  document_id: str) -> str | None:
    """Read one document's original file path out of the retriever, or `None` if it has no such document.

    The sibling of `document_text`, with the same locking discipline and for the same reason: `documents` is
    rewritten wholesale when the retriever commits a batch of filesystem changes, so a read that races one
    sees a half-swapped mapping.

    The path is what makes a docs-DB document *openable* — unlike a chat attachment, whose bytes Raven stores
    itself, an indexed document is a file the user already has, and the index only points at it. It may of
    course have moved or been deleted since indexing; the caller finds that out when it tries to open it.
    """
    if retriever is None:
        return None
    with retriever.datastore_lock:
        document = retriever.documents.get(document_id)
        return document["path"] if document is not None else None

def label_documents(retriever: "hybridir.HybridIR | None",
                    entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill in each entry's `label` from the document it names (`chatutil.document_label`), and whether it is `present`.

    `entries`: one dict per knowledge-base document the conversation has consulted, as
               `scaffold._collect_consulted_documents` assembles them by walking the branch. Each carries
               `document_id` (required - the handle `fetch_document` takes) and, where the document came from
               an automatic search, `query` (what surfaced it). `chatutil.format_consulted_documents` is what
               renders the result for the model.

    Returns new dicts, each the input entry plus `label` and `present`; the input is not modified. An entry
    naming a document that is no longer in the database keeps an empty label rather than being dropped - the
    conversation did consult it, and saying so with only an ID is more honest than pretending it was never
    there.

    `present` says which of those two an entry is, because the label cannot: an empty label means *either* a
    deleted document *or* one whose text yielded nothing to label it with, and those want opposite responses
    from the model. Without the flag the list advertises a deleted document as readable, the model spends a
    round on `fetch_document`, and gets a refusal - which a persistent model may answer by trying variations
    of the same ID (see `investigations/tool_refusal/`).
    """
    labelled = []
    for entry in entries:
        text = document_text(retriever, entry["document_id"])
        labelled.append({**entry, "label": chatutil.document_label(text) if text else "", "present": text is not None})
    return labelled


# ------------------------------------------------------------------------------------------------
# The tool that reads the clock
#
# Answers to neither switch: the current date is injected into every turn regardless, so withholding
# the tool would leave the model reading a call it could not resolve.
# ------------------------------------------------------------------------------------------------


def get_current_time() -> str:
    """Return the current local time.

    Tool entrypoint for the LLM's `get_current_time` tool. Returns exactly what the per-turn clock inject
    delivers, from the same formatter, so the two cannot drift into telling the model the time in two
    different shapes — which matters more here than it would for another tool, because the inject presents
    itself *as a call to this one*.

    Why the tool exists at all, given every turn already carries the time: without it the inject names a
    function the model was never offered, and a model reading its own transcript takes that call for its
    own. See the note beside the spec in `setup`.
    """
    return _formatters().time_now()


CANONICAL_BAD_EXPRESSION = ("That is not an expression this calculator can evaluate: {reason}. It takes "
                            "arithmetic in Python syntax - numbers, the usual operators, and functions from "
                            "the math module - and nothing else. Statements, assignments, variables and "
                            "attribute access are not available. Rewrite it as a single expression, or work "
                            "it out yourself and say so.")

# What the calculator may call. `simpleeval`'s own defaults are dropped rather than extended: they include
# `rand` and `randint`, and a tool named `calculate` that can silently return a different answer to the same
# question is a trap - the model has no way to tell that it happened, and neither has the user reading the
# reply.
#
# Everything here is a pure function of its arguments. `math` supplies the rest, by name, below.
_CALCULATOR_FUNCTIONS = {"abs": abs,
                         "min": min, "max": max,
                         "round": round,
                         "int": int, "float": float,
                         **{name: getattr(math, name) for name in
                            ("acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh",
                             "cbrt", "ceil", "comb", "copysign", "cos", "cosh", "degrees",
                             "dist", "erf", "erfc", "exp", "expm1", "fabs", "factorial",
                             "floor", "fmod", "gamma", "gcd", "hypot", "isqrt", "lcm",
                             "lgamma", "log", "log10", "log1p", "log2", "perm", "pow",
                             "radians", "remainder", "sin", "sinh", "sqrt", "tan", "tanh",
                             "trunc")}}

_CALCULATOR_NAMES = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf, "nan": math.nan}


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression, and return the result as text.

    Tool entrypoint for the LLM's `calculate` tool.

    `expression`: arithmetic in Python syntax. Expressions only - no statements, no assignments.

    Never raises: a malformed or rejected expression comes back as `CANONICAL_BAD_EXPRESSION`, which tells
    the model what this accepts and what to do instead. A tool that raises here would abort the turn over a
    typo the model could have corrected.
    """
    # `simpleeval` walks the AST and evaluates only the node types it knows, so what is *not* listed above
    # is not reachable rather than merely discouraged - measured 2026-08-25 on 1.0.7: attribute access
    # (`(1).__class__`, the usual way out of a sandbox) and comprehensions raise `FeatureNotAvailable`, an
    # undefined name raises `NameNotDefined`, and the resource bombs `9**9**9` and `'a'*10**10` raise
    # `NumberTooHigh` and `IterableTooLong` from its own limits. So "sandboxing" here is the choice of
    # allowed names, which is what the two tables above are.
    evaluator = simpleeval.SimpleEval(functions=_CALCULATOR_FUNCTIONS, names=_CALCULATOR_NAMES)
    try:
        # Parsed here rather than left to `simpleeval`, for two reasons. `eval` mode is what rejects
        # statements - `simpleeval` does not: measured 2026-08-25 on 1.0.7, `x = 1` raises nothing, *warns*
        # that the assignment was ignored, and returns a value, so the model would read back `x = 1 = 1` and
        # take it for a result. And handing the tree over as `previously_parsed` saves parsing it twice.
        #
        # It is also where an expander would go, should the expression language ever want to be richer than
        # Python's: `mcpyrate` can expand macros in a tree before `simpleeval` walks it. Not done, and the
        # reason is neither taste nor feasibility - models are not trained to write macro syntax, so the
        # feature would go unused by the only caller there is.
        result = evaluator.eval(expression, previously_parsed=ast.parse(expression, mode="eval").body)
    except Exception as exc:
        # Broad on purpose: `simpleeval` raises a family of its own exceptions plus whatever the arithmetic
        # itself raises (`ZeroDivisionError`, `ValueError` from `sqrt(-1)`, `OverflowError`), and every one
        # of them is the same thing as far as the model is concerned - this expression did not work, here is
        # why, try another.
        logger.info(f"calculate: rejected '{expression}': {type(exc)}: {exc}")
        return CANONICAL_BAD_EXPRESSION.format(reason=f"{type(exc).__name__}: {exc}")
    return f"{expression} = {result}"


# ------------------------------------------------------------------------------------------------
# The registry, completed
#
# Now that the functions exist, the third table can name them.
# ------------------------------------------------------------------------------------------------


TOOL_ENTRYPOINTS = {"websearch": websearch,
                    "webfetch": webfetch,
                    "get_current_time": get_current_time,
                    "calculate": calculate,
                    "search_documents": search_documents,
                    "fetch_document": fetch_document,
                    "list_consulted_documents": list_consulted_documents}


# ------------------------------------------------------------------------------------------------
# Offering them, and running what was asked for
#
# Which tools this turn gets, and the dispatch that turns a model's request into `role="tool"` messages.
# ------------------------------------------------------------------------------------------------


def maybe_tool_names_for_turn(settings: env,
                              documents_available: bool,
                              internet_available: bool) -> tuple[str, ...] | None:
    """Which tools to offer on one AI turn, as a `tool_names` value for `invoke` (or `prefill`).

    Returns `None` when every registered tool is on offer. Note that `None` is the *permissive* value here,
    not the restrictive one — hence the `maybe_` on the name, at every call site.

    Sorted, and a tuple rather than the set it is computed from, so that the same turn always produces the
    same sequence: set iteration order for strings varies with the interpreter's hash seed, i.e. between
    runs of the same code. That would make logs of two identical turns differ for no reason.

    It does not reach the wire — `invoke` filters the tool *spec list*, so the advertised order comes from
    the hand-written list in `setup` and is already stable — but that is a property of code elsewhere, and
    the next reader should not have to go and verify it before trusting a log line.

    Each argument owns one gated group outright, and a tool in neither group is always offered:

    `documents_available`: whether the document database is in play this turn, i.e. the user has it switched
                           on *and* this app has one. Gates `settings.document_tool_names`.

    `internet_available`: whether the network-reaching tools are in play this turn. Gates
                          `settings.network_tool_names`.

    With both `False` the ungated tools are still on offer — `get_current_time` today — so the result is
    that group rather than the empty tuple. (An empty tuple would be handled: `invoke` drops an emptied
    `tools` field rather than sending one, since some backends reject an empty list.)

    Shared so that the two callers cannot disagree. `scaffold.ai_turn` uses it to build the real request,
    and the GUI's context prefill uses it to warm the KV cache — and those must produce the same list, since
    tool definitions are expanded into the system block at the very front of the prompt. A prefill that
    warms a different tool list warms a prefix the real turn never sends, so the full prompt is reprocessed
    anyway, and the warm-up has cost time to achieve nothing.
    """
    if documents_available and internet_available:
        return None
    withheld = set()
    if not documents_available:
        withheld |= set(settings.document_tool_names)
    if not internet_available:
        withheld |= set(settings.network_tool_names)
    return tuple(sorted(set(settings.tool_entrypoints) - withheld))

def perform_tool_calls(settings: env,
                       message: dict,
                       on_call_start: Callable | None,
                       on_call_done: Callable | None,
                       maybe_refusal_text: str | None = None) -> list[env]:
    """Perform tool calls as requested in `message["tool_calls"]`.

    Returns a list of chat payloads (where each message's `role="tool"`) containing the tool outputs,
    one for each tool call.

    If the "tool_calls" field of `message` is missing or if it is empty, return the empty list.

    `maybe_refusal_text`: If given, no tool is called at all. Every requested call is answered with this
                          text as an `status="error"` result, and `on_call_start` never fires (nothing
                          started). This is how the caller declines a whole round of calls while leaving
                          the tools themselves on offer - see `raven.librarian.scaffold.ai_turn`, which
                          uses it when the turn's tool-call budget is spent.

    `on_call_start`: 3-argument callable: `(tool_call_id: str, function_name: str, arguments: dict[str, Any])`.

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
    def add_tool_response_record(output: str | list[dict], *,
                                 status: str,
                                 tool_call_id: str | None,
                                 function_name: str | None = None,  # unknown when the request was too malformed to name a tool
                                 dt: float | None = None,  # absent when nothing was called, so nothing was timed
                                 tool_metadata: dict | None = None) -> None:
        """Add a tool response record to `tool_response_records`.

        `output` is the tool result: either a plain string (wrapped as a single text content-part) or an
        already-built content-parts list — e.g. `websearch`'s one-text-part-per-result output.
        Error reports are passed as plain strings.

        The record is an `unpythonic.env.env` with the following attributes:

            `data`: dict: chat message object, with `role="tool"` and `content` the content-parts list.

            `status`: str: Values "success" or "error" are recommended.

            `tool_call_id`: str | None: ID of this tool call (can be matched against the `id` in the
                           `tool_calls` list of the AI chat message that spawned this call).

                           The ID should be included whenever it was present in the tool call request record.

            `function_name`: str | None: Which tool was called (or at least attempted),
                             if the call got that far. If it didn't, this is `None`.

            `dt`: float | None: Duration of this tool call, in seconds. Recommended to be included whenever
                                   the request was valid enough to actually proceed to call the function
                                   (so that the call timing can be measured).

            `tool_metadata`: dict | None: Structured metadata the entrypoint attached to this result
                             (by returning `(output, metadata)` instead of a bare `output`). The caller
                             (`scaffold`) merges it into the tool node's `generation_metadata`. Used e.g.
                             by `webfetch` to record `webfetch_denied_host` for the GUI override.
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

    # Declining the whole round. Deliberately ahead of the per-request validation below: a malformed request
    # is not worth reporting when nothing was going to run anyway, and the model's next move is to answer,
    # not to fix its JSON.
    if maybe_refusal_text is not None:
        logger.info(f"perform_tool_calls: refusing {len(tool_calls)} tool call{plural_s} without calling anything: {maybe_refusal_text}")
        for request_record in tool_calls:
            add_tool_response_record(maybe_refusal_text,
                                     status="error",
                                     tool_call_id=request_record.get("id", None),
                                     function_name=request_record.get("function", {}).get("name", None))
        return tool_response_records

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
