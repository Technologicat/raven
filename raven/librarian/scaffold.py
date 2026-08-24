"""Scaffolding for a multi-turn conversation with automatic RAG search and tool-calling."""

__all__ = ["user_turn",
           "ai_turn", "retry_tool_calls", "action_ack", "action_stop",

           # For scripting: the prompt a turn would send, without sending it
           "build_turn_prompt", "make_tool_context",
           "build_system_injects"]  # also what the chat view shows, so the log matches what is sent

import logging
logger = logging.getLogger(__name__)

import json

from typing import Any, Callable, Collection, Dict, List, Optional, Tuple, TYPE_CHECKING

from unpythonic import dyn
from unpythonic.env import env

from ..common import netutil

from . import chattree
from . import chatutil
from . import config as librarian_config
from . import llmclient

# `hybridir` is only referenced by scaffold for type annotations (retriever
# parameters); all runtime access to the retriever goes through duck-typed
# `.query(...)` calls. Importing it at runtime would drag in the full
# `chromadb`/`bm25s`/`watchdog` stack, which isn't needed by scaffold itself
# or by scaffold's test suite — so defer the import to type-checking only.
if TYPE_CHECKING:
    from . import hybridir

action_ack = llmclient.action_ack
action_stop = llmclient.action_stop

# --------------------------------------------------------------------------------
# User's turn

def user_turn(llm_settings: env,
              datastore: chattree.Forest,
              head_node_id: str,
              user_message_text: str,
              staged_images: Optional[List[env]] = None,
              staged_files: Optional[List[env]] = None) -> str:
    """Add the user's message with content `user_message_text` to `datastore`.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.

    `datastore`: The chat datastore.

    `head_node_id`: Current HEAD node of the chat. Used as the parent for the no-match message, if needed.

    `user_message_text`: The message text to add.

    `staged_images`: Images the user attached to this message, or `None` for a text-only message. Each entry is
                     an `env` with `raw` (the image bytes), `provenance_url` (recorded provenance — where the
                     image came from), and `provenance_source` (the categorical pathway, e.g.
                     `"user_attachment"`). Each is stored as a datastore sidecar via
                     `imagestore.store_image_as_sidecar`; the resulting `image_url` parts are appended to the
                     message content (after the text part), and the per-image provenance metadata is recorded
                     under the node's `general_metadata["sidecars"]`. The heavy work (decode + downsample) runs
                     here, so call this off the GUI thread.

    `staged_files`: Documents (plain text / PDF) the user attached, or `None`. Each entry is an `env` with `raw`
                    (the file bytes), `name` (the original filename — for display and to derive the sidecar
                    extension), `provenance_url`, and `provenance_source` (as for `staged_images`). Each is stored
                    verbatim as a datastore sidecar via `textfilestore.store_file_as_sidecar`; the resulting
                    `text_file` parts are appended to the message content and their provenance is recorded
                    alongside the images' under `general_metadata["sidecars"]`. Unlike an image, a document works
                    with any model (its text is folded into the prompt at wire-build), so this is not gated on
                    vision capability.

    Returns the new HEAD node ID (i.e. the chat node that was just added).
    """
    message = chatutil.create_chat_message(llm_settings=llm_settings,
                                           role="user",
                                           text=user_message_text)
    sidecar_metadata_by_filename = {}
    if staged_images:
        from . import imagestore  # deferred: only pulls Pillow/torch when an image is actually attached
        for staged in staged_images:
            result = imagestore.store_image_as_sidecar(datastore=datastore,
                                                       image_source=staged.raw,
                                                       provenance_url=staged.provenance_url,
                                                       provenance_source=staged.provenance_source)
            message["content"].append(result.part)  # a fresh list from create_chat_message; safe to extend
            sidecar_metadata_by_filename[result.filename] = result.sidecar_metadata
    if staged_files:
        from . import textfilestore  # deferred: only pulls docextract/pypdf when a document is actually attached
        for staged in staged_files:
            result = textfilestore.store_file_as_sidecar(datastore=datastore,
                                                         file_source=staged.raw,
                                                         name=staged.name,
                                                         provenance_url=staged.provenance_url,
                                                         provenance_source=staged.provenance_source)
            message["content"].append(result.part)
            sidecar_metadata_by_filename[result.filename] = result.sidecar_metadata

    payload = chatutil.create_payload(llm_settings=llm_settings,
                                      message=message)
    if sidecar_metadata_by_filename:
        payload["general_metadata"]["sidecars"] = sidecar_metadata_by_filename

    user_message_node_id = datastore.create_node(payload=payload,
                                                 parent_id=head_node_id)
    return user_message_node_id


# --------------------------------------------------------------------------------
# AI's turn

def _create_synthetic_assistant_node(llm_settings: env,
                                     datastore: "chattree.Forest",
                                     parent_node_id: str,
                                     text: str,
                                     add_persona: bool = True) -> str:
    """Create and persist a *synthetic* assistant message node, returning its id.

    "Synthetic" = the text is authored by Raven, not generated by the LLM. One case uses this: the
    backend-error notice, when the LLM call raised.

    Like a real assistant reply it is a `role="assistant"` node, so the usual message affordances apply —
    notably reroll, which re-runs the AI turn from the parent and thus doubles as "retry". It carries no
    `generation_metadata` (there was no generation), so the renderer shows no token-stats line, which is how
    these non-generated messages have always appeared.

    Callers add their own payload fields afterward if needed (e.g. the no-match node's `retrieval`) and fire
    whichever GUI done-callback fits their path.

    `add_persona`: Whether to prefix the character's name, as a real reply carries it. Pass `False` on a
                   turn taken without a character - Raven authored the text either way, but on such a turn
                   there is no character it could be speaking as.
    """
    message = chatutil.create_chat_message(llm_settings=llm_settings,
                                           role="assistant",
                                           text=text,
                                           add_persona=add_persona)
    return datastore.create_node(payload=chatutil.create_payload(llm_settings=llm_settings,
                                                                 message=message),
                                 parent_id=parent_node_id)


def _search_docs(retriever: "hybridir.HybridIR",
                 query: str,
                 k: Optional[int] = None) -> List[Dict]:
    """Helper for `ai_turn`. Search the document database (`retriever`) for `query`, returning `k` best matches.

    `retriever`: A `raven.librarian.hybridir.HybridIR` retriever connected to the document database.

    `query`: The query string to search with in the document database. (Note "with", not "for"; the query may
             undergo processing. As of v0.2.3, it is directly tokenized for keyword search, but the semantic
             search uses the "qa" role, which maps questions and possible corresponding answers near each other.)

    `k`: Return up to this many best matches. Note that there is an internal threshold, which automatically drops
         any very low-quality semantic matches.

         The default `None` means `k=10`.

    An empty result is an ordinary outcome, reported as an empty list. It used to end the turn before the LLM
    ran, on the reasoning that a model with no documents would confabulate; that guard is now a *badge* on
    the reply instead. Two things made the trade worth it. The guard could not tell a question about the
    documents from a general-knowledge aside, so in the default configuration it answered "what is 2+2?" with
    "No matches in document database" - a false-positive rate near 100% on such questions. And once the model
    can search for itself, an empty first pass is the *strongest* signal that the heuristic query was bad,
    which made this the one case where the model was forbidden from improving it.
    """
    if k is None:
        k = 10
    return retriever.query(query,
                           k=k,
                           max_span_length=librarian_config.docs_max_result_length,
                           return_extra_info=False)

def _grounding_was_declared(content: List[Dict],
                            maybe_metadata: Optional[Dict]) -> bool:
    """Whether one tool result counts as material to answer from, per what produced it.

    Declared results say so themselves, by returning `(output, {"grounding": ...})` from the entrypoint.
    `webfetch` is the case that must: its allowlist refusal is a perfectly non-empty string that grounds
    nothing at all.

    Undeclared results fall back to "it returned something", which amounts to trusting the tool. For Raven's
    own tools that is a default to be overridden where it is wrong; for a third-party tool reached over MCP
    it is the policy, since its author has no reason to annotate anything for Raven's benefit. Trusting them
    is defensible - producing material is what a tool is *for* - and if some tool ever turns out to return
    confident noise, the answer is a Raven-side setting, not an expectation that upstream will mark it.

    `maybe_metadata`: the entrypoint's structured metadata, or `None` when it attached none. Its two homes
                      are the live `tool_metadata` on a response record and, once stored, the tool node's
                      `generation_metadata` - which is what lets an earlier turn's declaration still be read
                      back off the branch.
    """
    declared = (maybe_metadata or {}).get("grounding")
    if declared is None:
        declared = bool(chatutil.content_to_text(content).strip())
    return bool(declared)

def _branch_grounding_is_present(datastore: chattree.Forest,
                                 head_node_id: str) -> bool:
    """Whether any tool result already on this branch provided material to answer from.

    Walks the stored nodes rather than the linearized history, because the declaration lives in each tool
    node's `generation_metadata` and `chatutil.linearize_chat` hands out bare message dicts.

    Scoped to the whole branch, on the same rule as an attachment: the question is *is this material still
    in the context*, which is answerable, rather than *has it gone stale*, which is not. A tool result is a
    persisted node, so it remains in the model's context verbatim for as long as the window holds it. (The
    automatic pre-turn search is the one thing that does not qualify - it is never persisted, so next turn
    it is genuinely gone, and that is a fact about the data rather than a policy about lifetimes.)
    """
    node_id = head_node_id
    while node_id is not None:
        payload = datastore.get_payload(node_id)
        generation_metadata = payload.get("generation_metadata") or {}
        if payload["message"]["role"] == "tool" and generation_metadata.get("status", "success") == "success":
            if _grounding_was_declared(payload["message"]["content"], generation_metadata):
                return True
        node_id = datastore.get_parent(node_id)
    return False

def _documents_named_by(payload: Dict) -> List[Tuple[str, Optional[str]]]:
    """The knowledge-base documents one stored node reached, as `(document_id, query)` pairs.

    Two producers write into a node, and both are read here, because "consulted" is deliberately silent
    about who did the consulting:

      - An **assistant** node carries the automatic pre-turn search in its `retrieval` payload - the query
        Raven guessed from the user's message, and the matches it found.
      - A **tool** node carries what the model asked for itself, in the metadata its entrypoint declared
        (`llmclient.search_documents_wrapper`, `llmclient.fetch_document_wrapper`). A fetch has no query,
        which is why the query half of the pair is optional rather than a placeholder string.
    """
    named = []
    retrieval = payload.get("retrieval") or {}
    query = retrieval.get("query")
    for result in retrieval.get("results") or []:
        if result.get("document_id"):
            named.append((result["document_id"], query))
    generation_metadata = payload.get("generation_metadata") or {}
    for document_id in generation_metadata.get("document_ids") or []:
        named.append((document_id, generation_metadata.get("docs_query")))
    return named

def _collect_consulted_documents(datastore: chattree.Forest,
                                 head_node_id: str,
                                 exclude_document_ids: Collection[str]) -> List[Dict[str, Any]]:
    """The knowledge-base documents this branch has already looked at, newest first, deduplicated.

    Closes a hole the automatic search leaves open. Its matches are injected for one turn and then dropped,
    never persisted - so a follow-up question arrives with the model's own earlier reply in view and the
    material behind it gone, with nothing to signal that. The IDs survive in the stored `retrieval` payload;
    handing those back turns "I no longer know where that came from" into "read it again with
    `fetch_document`".

    Pointers rather than text, on purpose. Re-injecting the material grows without bound as a conversation
    goes on; a list of IDs grows slowly, and is capped anyway
    (`config.max_consulted_documents_listed`).

    `exclude_document_ids`: documents whose full text is already in this turn's context - the current
                            auto-search matches. Listing those would be redundancy in the one place that
                            has to stay compact, since the material is sitting right beside the list.
    """
    seen = set(exclude_document_ids)
    entries = []
    node_id = head_node_id
    while node_id is not None:
        for document_id, query in _documents_named_by(datastore.get_payload(node_id)):
            if document_id in seen:
                continue
            seen.add(document_id)
            entries.append({"document_id": document_id, "query": query})
        node_id = datastore.get_parent(node_id)

    cap = librarian_config.max_consulted_documents_listed
    if len(entries) > cap:
        logger.info(f"_collect_consulted_documents: {len(entries)} documents consulted on this branch; "
                    f"listing the {cap} most recent (config.max_consulted_documents_listed).")
        entries = entries[:cap]
    return entries

def _attachment_is_present(history: List[Dict]) -> bool:
    """Return whether the user has attached an image or a document anywhere on this branch.

    The other half of the same rule `_branch_grounding_is_present` applies to tool results: an attachment is
    material the user placed in the context directly, and stays usable for as long as it is in the window.
    It differs only in having no producer to ask, so it is found by looking rather than by declaration.

    Not counted here, and not anywhere: the conversation itself, nor the AI's own earlier replies. A model
    summarizing what it previously said is exactly the ungrounded answer this is used to guard against.
    """
    for message in history:
        for part in message.get("content", []):
            if isinstance(part, dict) and part.get("type") in ("image_url", "text_file"):
                return True
    return False

def _add_to_system_message(llm_settings: env,
                           history: List[Dict],  # mutated!
                           texts: List[str]) -> None:
    """Append `texts` to the text content of the leading system message of `history`.

    Injects that are *instruction-like* go here rather than into a message of their own. Measured across
    the supported model families, the leading system block was the cheapest placement in deliberation
    tokens and the only one that never provoked the model into remarking on the inject as if the user had
    typed it. That is affordable only because these injects are constant within a session (or within a
    day, for the date), so hoisting them costs the backend's prompt-prefix cache nothing.

    Material that is *data* - retrieval results, the clock - does not belong here; see
    `_synthetic_tool_exchange` for that, and note that the strictest chat templates permit exactly one
    system message anyway, so a second one is not available to us even if we wanted it.

    NOTE: `chatutil.linearize_chat` hands out the datastore's own message dicts, not copies. Hence the
    system message is *replaced* with a modified copy rather than edited in place - editing in place
    would write the injects into the stored system prompt, permanently, once per turn.
    """
    if not texts:
        return
    has_system_message = bool(history) and history[0]["role"] == "system"
    old_texts = [chatutil.content_to_text(history[0]["content"])] if has_system_message else []
    system_message = chatutil.create_chat_message(llm_settings=llm_settings,
                                                  role="system",
                                                  add_persona=False,
                                                  text="\n\n".join([*old_texts, *texts]))
    if has_system_message:
        history[0] = system_message
    else:  # no system prompt in this chat (unusual, but a client may build a history without one)
        history.insert(0, system_message)

def _synthetic_tool_exchange(llm_settings: env,
                             call_id: str,
                             function_name: str,
                             arguments: Dict,
                             result_text: str) -> List[Dict]:
    """Return `[assistant message requesting a tool call, tool message answering it]`, for injected data.

    "Synthetic" = neither message is real. The AI never asked for this call; Raven made the call on its
    own initiative and is presenting the answer in the shape the model expects for one.

    The assistant message is load-bearing, not decoration: given a bare `tool` message with no call to
    answer, Gemma 4 ignores the material entirely and confabulates a confident wrong answer in its place -
    reproducibly, across packagings and backend versions. Handing it the call it never made is what makes
    the result legible as a result.

    One `tool` message per call, as the OpenAI schema describes - a `tool` message answers exactly one
    `tool_call_id`. So a set of retrieval matches rides as a single message, not one message per match;
    sharing one call id across many messages is the shape that Gemma4-E4B reads as nothing at all.
    """
    call_message = chatutil.create_chat_message(llm_settings=llm_settings,
                                                role="assistant",
                                                add_persona=False,
                                                text="",
                                                tool_calls=[{"id": call_id,
                                                             "type": "function",
                                                             "function": {"name": function_name,
                                                                          "arguments": json.dumps(arguments)}}])
    result_message = chatutil.create_chat_message(llm_settings=llm_settings,
                                                  role="tool",
                                                  add_persona=False,
                                                  text=result_text)
    result_message["tool_call_id"] = call_id  # OAI spec: the linkage lives on the tool-response message
    return [call_message, result_message]

def build_system_injects(llm_settings: env,
                         grounding_material_exists: bool,
                         tools_are_spent: bool = False) -> List[str]:
    """The instruction-like texts this turn appends to the leading system message.

    Split out of `build_turn_prompt` so that the chat view can show them. The log's promise is that it shows
    what was said, and these are said every turn while appearing nowhere in it - so the view needs the same
    list the prompt is built from, and getting it by re-deriving the wording in the GUI would be two sources
    of truth for text the model actually reads.

    The two flags are what makes an inject conditional, and neither is a property of the conversation
    alone - which is why they are arguments rather than something this function could work out:

    `grounding_material_exists`: whether anything gave the model material to answer *from* this turn:
                                 retrieval results, a tool result that declared grounding, or an attachment
                                 on the branch. Adds the reminder to base claims about the provided
                                 documents on them. Sent with nothing to ground in, that reminder is a
                                 self-contradiction the model tries to resolve, at up to 37x the
                                 deliberation and sometimes without terminating - so the condition is
                                 load-bearing rather than an optimization.

    `tools_are_spent`: whether the turn has used its tool-call budget, so the model should answer from what
                       it already has rather than reaching for another call it will not get.
    """
    formatters = llm_settings.formatters
    injects = [formatters.date_now(),
               formatters.loaded_model(llm_settings.model, llm_settings.context_length),
               formatters.reminder_to_write_conversationally()]
    if grounding_material_exists:
        injects.append(formatters.reminder_to_use_information_from_context_only())
    if tools_are_spent:
        injects.append(formatters.notice_that_tools_are_spent())
    return injects

def build_turn_prompt(llm_settings: env,
                      history: List[Dict],
                      docs_query: Optional[str],
                      docs_matches: List[Dict],
                      tool_context: env,
                      tools_are_spent: bool = False,
                      tools_enabled: bool = True,
                      use_character_card: bool = True) -> List[Dict]:
    """Return the message history for the AI's turn: `history` with the temporary injects added.

    These are not meant to be persistent, so we don't even add them to the datastore,
    but only insert them into the temporary linearized history that is fed to the LLM.

    **`history` is not modified.** A new list is returned; the caller rebinds. The message dicts themselves
    are shared with the input, so treat the result as read-only — `chatutil.linearize_chat` hands out the
    datastore's own dicts, and the one message this function does need to change (the leading system
    message, which the instruction-like injects join) is *replaced* by a modified copy rather than edited.

    Two kinds of material go in, and they are shaped differently because they are asking for different
    things. Instructions - the reminders, and the date - want to be obeyed, so they join the leading
    system message. Data - the clock time, the document-database matches - wants to be read but not
    obeyed, so it arrives as the answer to a tool call Raven makes on the model's behalf.

    All of it lands *before* the user's latest message. That position is what keeps the model answering
    the user instead of continuing the agent loop: with a tool result as the very last message, Qwen 3.6
    will sometimes reply by emitting another `search_documents` call rather than an answer. It also keeps
    the whole conversation ahead of the injects byte-identical from turn to turn, so the backend reprocesses
    only the tail. (The front of the history, where the retrieval results used to go, is the worst of both:
    a fresh prompt prefix every turn.) The rationale, and the measurements behind every choice here, are in
    `investigations/context-injects/context-inject-shape-measurements.md`.

    The position is also why this needs no `continue_` flag: when continuing the AI's interrupted message,
    the history must look as it did at the moment of interruption, and everything we add sits ahead of the
    user's message, which is ahead of the message being continued either way.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.
                    Contains (among other things) a mapping of roles to persona names.

    `history`: Linearized message history in the OpenAI format sent to the LLM. Not modified.

    `docs_query`: The query string the document database was searched with, or `None` if it wasn't searched.
                  Reported to the model as the arguments of the synthetic search call, so that the matches
                  arrive as the answer to a legible question rather than as free-floating material.

    `docs_matches`: Docs search matches returned by `HybridIR` (see `_search_docs`).

    `tool_context`: The turn's request context (`make_tool_context`). Read for `grounded`, which is where
                    retrieval results and tool results report whether they actually provided material.

    `tools_are_spent`: Whether this turn has used its tool-call budget; adds the notice saying so.

    `tools_enabled`: Whether the turn offers the model any tools. `False` also withholds the clock, which is
                     delivered as a synthetic call to `get_current_time` — see the comment at that inject
                     for why the two travel together in both directions.
    """
    # Work on our own list from here on. Everything below inserts into it, and `_add_to_system_message`
    # replaces its leading element; doing that to the caller's list is what this function used to do, and
    # what made it impossible to ask "what would Raven send?" without handing over a list to be altered.
    history = list(history)

    # Two sources, because they are scoped differently (see `_attachment_is_present`). `grounded` is
    # *declared* by whatever produced the material; an attachment is not produced by anything, so it is
    # still found by walking the branch.
    grounding_material_exists = tool_context.grounded or _attachment_is_present(history)

    # The instruction injects belong to the character, and go when she does. Not merely a preference: with
    # no persona there may be no system message at all (Raven's shipped `system_prompt` is empty, all of the
    # content being the character card), and `_add_to_system_message` *inserts* one when the history has
    # none — so leaving these on would hand a deliberately bare model a system message containing nothing
    # but today's date.
    if use_character_card:
        _add_to_system_message(llm_settings=llm_settings,
                               history=history,
                               texts=build_system_injects(llm_settings=llm_settings,
                                                          grounding_material_exists=grounding_material_exists,
                                                          tools_are_spent=tools_are_spent))

    # The data-like injects below go into synthetic tool exchanges of their own, not into the system
    # message; see `_add_to_system_message` for why the split is not a stylistic one.
    formatters = llm_settings.formatters

    # Data-like injects -> synthetic tool calls, placed just before the user's latest message.
    #
    # The time is *injected* rather than left for the model to fetch, because a conversation needs the date
    # without ever asking for it — "is this paper recent", "what shall we do this week" — and a tool call
    # happens only if the model decides to make one. Handing it over unasked costs one synthetic exchange
    # and removes the decision.
    #
    # What that is not evidence for: the tool path being unreliable. Asked the time with a clock tool
    # actually on offer, qwen3.5-9b called it 24 times out of 24 — with and without being told to, so it
    # reaches for it unprompted (`investigations/absent-tool-behaviour/`, the 2x2). Every measured failure
    # there — refusal, invention, a reasoning loop returning nothing — comes from the cells where the tool
    # is *absent*, which is not this situation.
    #
    # What stays untested is the case this inject is actually for. Those samples all ask *about the time*,
    # so reaching for a clock is the obvious move. The turn that needs today's date is usually the one where
    # nobody mentions it — "is this paper recent" — and whether a model calls the clock when the date is
    # merely relevant is a different question, unasked.
    #
    # The clock goes in only when tools are on offer, and the two are tied both ways. Normally
    # `get_current_time` is offered whatever the group switches say, *because* the time is injected as a
    # call to it and a history calling an undeclared tool is a shape models handle badly. Withdraw every
    # tool and the same reasoning runs backwards: staging a call to a tool that now does not exist is the
    # confusing shape rather than the safe one. A one-shot scripted job also rarely wants the time at all.
    data_injects = []
    if tools_enabled and use_character_card:
        data_injects.extend(_synthetic_tool_exchange(llm_settings=llm_settings,
                                                     call_id="raven_clock",
                                                     function_name="get_current_time",
                                                     arguments={},
                                                     result_text=formatters.time_now()))
    # Order is load-bearing: the earlier conversation's documents, then this turn's search results. The two
    # lists look alike, and whichever sits closest to the user's message reads as the answer to it — so with
    # the consulted list last, a model could take a document it read three turns ago for something the
    # current search just returned. Chronological order says what the wording says, and the two agree.
    consulted_documents = getattr(tool_context, "consulted_documents", None)
    if consulted_documents:
        # Pushed, not merely offered as a tool, because the model cannot detect the gap it fills. At a
        # follow-up question its own transcript shows it answering from documents, so nothing signals that
        # the automatic search's matches were dropped after that turn. A tool covers the case where it knows
        # to pull; this covers the case where it does not.
        data_injects.extend(_synthetic_tool_exchange(llm_settings=llm_settings,
                                                     call_id="raven_consulted",
                                                     function_name="list_consulted_documents",
                                                     arguments={},
                                                     result_text=formatters.consulted_documents(consulted_documents)))
    if docs_matches:
        # The synthetic call names the real tool, which is no longer a fiction: asked about something these
        # matches do not cover, the model reaches for a second, better-aimed search, and now there is one to
        # reach for. Before that tool existed it wrote the call out as literal text and the user got that
        # instead of an answer, roughly one turn in three on Qwen3.6-27B.
        data_injects.extend(_synthetic_tool_exchange(llm_settings=llm_settings,
                                                     call_id="raven_docs",
                                                     function_name="search_documents",
                                                     arguments={"query": docs_query if docs_query is not None else ""},
                                                     result_text=formatters.docs_matches(docs_matches)))

    for position in range(len(history) - 1, -1, -1):
        if history[position]["role"] == "user":
            break
    else:  # No user message to place the injects ahead of. Nothing to reply to either, so this shouldn't happen.
        logger.warning("build_turn_prompt: no user message in history; appending injects at the end.")
        position = len(history)
    history[position:position] = data_injects

    return history


def make_tool_context(llm_settings: Optional[env],
                      retriever: "Optional[hybridir.HybridIR]") -> env:
    """Create the per-turn tool-call request context (the `dyn.tool_context` payload).

    `build_turn_prompt` requires one of these, so a script asking "what would Raven send?" needs this too;
    passing `None` for both arguments gives the plain shape that question usually wants - no retriever, no
    tools that size themselves against the context window.

    One env per AI turn, not per tool round, because two kinds of field live here and only one of them is
    per-round:

      - *Accumulating* fields carry information forward across the rounds of a single turn. `grounded` is
        the case that forces the per-turn lifetime: whether anything this turn gave the model material to
        answer from is a question about the turn, and a tool call in round 1 must still count in round 3.
      - *Volatile* fields are recomputed by `_perform_and_store_tool_calls` before each round, because
        their correct value depends on what the earlier rounds did. `webfetch_allowed_hosts` is one: a
        websearch in round 1 can auto-allow the hosts a webfetch reaches for in round 2. `used_tokens` is
        another: each round adds to the branch, so how much room is left changes as the turn proceeds.

    Everything here is harness-supplied, never model-supplied - that separation is the point, and it is why
    the retriever is handed over this way rather than being closed over at tool-registration time (it could
    not be: `llmclient.setup` runs before `hybridir.setup` in both clients).

    `llm_settings`: Needed by tools that have to reason about the context window - `fetch_document` sizes
                    what it returns against what is left of it. Carried here rather than closed over,
                    because an entrypoint is called with the model's arguments and nothing else. `None` is
                    allowed, and means no tool that needs it may run; nothing else reads it.

    `retriever`: The document-database retriever the document tools search, or `None` if this app has no
                 document database. The tools are duck-typed against `.query(...)`; see the module header
                 for why `hybridir` is not imported at runtime.
    """
    return env(llm_settings=llm_settings,
               retriever=retriever,
               webfetch_allowed_hosts=frozenset(),  # volatile: recomputed per round
               used_tokens=0,  # volatile: recomputed per round
               grounded=False,  # accumulating: did anything this turn provide grounding material?
               consulted_documents=[])  # fixed for the turn: what the branch had already read when it began

def _record_grounding(tool_context: env,
                      tool_response_record: env) -> None:
    """Fold one tool result into `tool_context.grounded`. Monotonic: once grounded, stays grounded.

    Grounding is *declared at the source* rather than inferred from message shape; see
    `_grounding_was_declared` for what a declaration is and what an undeclared result falls back to.

    Why this matters enough to have its own mechanism: the reminder to base claims on the provided context
    is only sound when there *is* context. Sent with nothing to ground in, it is a self-contradiction that
    measured 5-37x the deliberation of sending nothing, with one model never terminating at all.
    """
    if tool_context.grounded:  # already grounded; nothing can un-ground it
        return
    if tool_response_record.status != "success":
        return
    maybe_metadata = tool_response_record.tool_metadata if "tool_metadata" in tool_response_record else None
    tool_context.grounded = _grounding_was_declared(tool_response_record.data["content"], maybe_metadata)

# Characters a filesystem (or a human reading a folder listing) would rather not meet in a filename. The
# sidecar itself is content-addressed and so never carries this name, but `cleanup.rescue_to_staging` writes
# the rescued copy out under it, and a page title is arbitrary text from the open web.
_UNSAFE_IN_FILENAME = str.maketrans({c: "-" for c in '/\\:*?"<>|\n\r\t'})

def _document_display_name(document: Dict[str, str]) -> str:
    """A human-readable, filesystem-safe name for a fetched document: `"<host> - <title>"`.

    Leads with the host because that is the discriminator that survives leaving the conversation — in a
    folder of rescued attachments, "Attention Is All You Need" alone says nothing about where it came from,
    and two sites can publish that title. Falls back to the host alone for a titleless page, and to the whole
    URL when there is no host to extract (which should not happen for an `http(s)` fetch, but the name must
    exist either way).

    Deliberately *not* made unique. Uniqueness on disk is the content hash's job and it already has it;
    what a name has to do is let a reader tell two things apart, and the remaining collision — same host,
    same title, genuinely different bytes — is handled where it actually bites, by the ` (2)` suffix
    `cleanup.rescue_to_staging` already applies.
    """
    url = document.get("url") or ""
    title = (document.get("name") or "").strip()
    host = netutil.url_host(url) or ""
    if title and title != url:  # `webfetch_wrapper` falls back to the URL when a page has no title
        name = f"{host} - {title}" if host else title
    else:
        name = host or url or "fetched document"
    return name.translate(_UNSAFE_IN_FILENAME).strip()

def _attachmentify_tool_result(datastore: chattree.Forest,
                               tool_response_record: env) -> Dict[str, Dict]:
    """Store an over-long fetched document as a sidecar, rewriting the tool result to an excerpt plus a chip.

    Modifies `tool_response_record.data["content"]` in place, and removes the `fetched_document` key from
    `tool_response_record.tool_metadata` (it is consumed here rather than recorded as generation metadata).
    Returns `{sidecar filename: provenance}` for the caller to put under `general_metadata["sidecars"]`, or
    `{}` when nothing was stored — no document was declared, the result is short enough to read inline, or
    the store failed.

    Eligibility is *declared by the tool*, via a `fetched_document` entry in its returned metadata naming
    the document's URL and title (`llmclient.webfetch_wrapper` is the one that does). Declaring rather than
    matching on the tool's name is what keeps `websearch` inline at any length: its result is a list of
    links, and the links are the thing the user wants to click, so hiding them behind a chip would be a
    regression rather than a tidying. It also means a tool that returns a document opts in by saying so.

    The model reads the same bytes either way. A `text_file` part is folded back into the message text at
    wire-build (`llmclient.serialize_history_for_wire`), so what changes is the chat log and the datastore
    JSON, not the conversation — which is the property that makes this safe to do behind the user's back.
    Two things do change for the better: the fetched text is now content-addressed on disk, so it survives
    the page going away, and it is sized against the context window along with every other attachment
    instead of being sent whole.

    A failure to store is not allowed to lose the result: the log keeps the full text inline, which is what
    it did before this existed.
    """
    if "tool_metadata" not in tool_response_record:
        return {}
    document = tool_response_record.tool_metadata.pop("fetched_document", None)
    if document is None:
        return {}

    text = chatutil.content_to_text(tool_response_record.data["content"])
    if len(text) <= librarian_config.tool_result_attachment_threshold:
        return {}

    from . import textfilestore  # deferred: only pulls docextract/pypdf when a document is actually stored
    # The name is for humans, and it has one job the sidecar's own filename cannot do: tell two documents
    # apart. On disk they are content-addressed, so nothing collides and nothing is ever overwritten — two
    # fetches of one URL are one file when the bytes match and two when the page changed, which is what a
    # message wanting the version it actually saw needs. But a *name* is what the chip shows and what
    # `cleanup.rescue_to_staging` writes out, and two pages can share a title. Leading with the host is what
    # keeps the name meaningful outside the conversation, months later, in a folder of rescued files.
    name = _document_display_name(document)
    try:
        result = textfilestore.store_file_as_sidecar(datastore=datastore,
                                                     file_source=text.encode("utf-8"),
                                                     # The extension decides how the text is extracted back out
                                                     # later; the server hands us markdown, which `docextract`
                                                     # reads verbatim. The name carries no extension of its own
                                                     # (a page title may contain anything at all).
                                                     name=f"{name}.md",
                                                     provenance_url=document["url"],
                                                     provenance_source="tool_result",
                                                     content_type="text/markdown")
    except Exception as exc:  # noqa: BLE001 -- a failed store must not lose the tool result
        logger.warning(f"_attachmentify_tool_result: could not store '{name}' as a sidecar, leaving it inline: {type(exc)}: {exc}")
        return {}

    # The chip carries the title, so the excerpt is free to be only the opening of the document. Both are
    # kept: the chip alone would make the result invisible, the excerpt alone would lose the handle on it.
    excerpt = chatutil.excerpt(text, librarian_config.tool_result_preview_characters)
    tool_response_record.data["content"] = [chatutil.text_content_part(excerpt), result.part]
    logger.info(f"_attachmentify_tool_result: stored '{name}' ({len(text)} characters) as sidecar '{result.filename}'.")
    return {result.filename: result.sidecar_metadata}

def _perform_and_store_tool_calls(llm_settings: env,
                                  datastore: chattree.Forest,
                                  assistant_message: Dict,
                                  parent_node_id: str,
                                  tool_context: env,
                                  maybe_refusal_text: Optional[str] = None,
                                  on_tools_start: Optional[Callable] = None,
                                  on_call_lowlevel_start: Optional[Callable] = None,
                                  on_call_lowlevel_done: Optional[Callable] = None,
                                  on_tool_done: Optional[Callable] = None,
                                  on_tools_done: Optional[Callable] = None) -> str:
    """Execute the tool calls in `assistant_message`, storing each result as a `role="tool"` chat node.

    The result nodes are chained under `parent_node_id` (normally the assistant message that requested
    the calls), one node per call, in call order. Returns the node ID of the last one created — the new HEAD.

    Shared by `ai_turn`'s agent loop and by `retry_tool_calls` (the GUI "approve denied host" override),
    so the per-turn request-context binding (`dyn.tool_context`), the `perform_tool_calls` dispatch, and
    the result→`generation_metadata` mapping all live in exactly one place.

    `tool_context`: The turn's request context, from `make_tool_context` (which see for what belongs in
                    it and why it outlives a single round). Bound to `dyn.tool_context` for the dynamic
                    extent of the dispatch — the request-context pattern (cf. Racket's `parameterize`,
                    Flask's `g`). Entrypoints that need it read `dyn.tool_context`; see the field registry
                    at `llmclient.make_dynvar(tool_context=...)`.

                    Its volatile fields are refreshed here, per round. `webfetch_allowed_hosts` is
                    recomputed from `parent_node_id`, so the walk sees this turn's user message and any
                    prior tool results on the branch — that is what lets a websearch in an earlier round
                    auto-allow the hosts a webfetch reaches for in this one.

    `maybe_refusal_text`: If given, nothing is called: every requested call is answered with this text as
                         an error result. The results are stored as ordinary `role="tool"` nodes, because
                         from the model's side that is exactly what they are — see `ai_turn` for when the
                         turn declines a round.
    """
    head_node_id = parent_node_id
    if on_tools_start is not None:
        on_tools_start(assistant_message["tool_calls"])

    if maybe_refusal_text is None:
        tool_context.webfetch_allowed_hosts = chatutil.compute_auto_allowed_hosts(
            datastore, head_node_id,
            trust_search_results=librarian_config.webfetch_trust_search_results)
        # How full the context already is, for the tools that have to fit something into what is left. Volatile
        # for a reason worth stating: by round three the model's own reasoning and the earlier rounds' results
        # are in the branch, so a figure taken at turn start would claim room that has since been spent.
        # Skipped when refusing, because it walks the whole branch to serve entrypoints that will not run.
        tool_context.used_tokens = llmclient.count_branch_tokens(llm_settings, datastore, head_node_id)[0]

    # Each tool call produces exactly one response. No-ops if the message contains no tool calls.
    with dyn.let(tool_context=tool_context):
        tool_response_records = llmclient.perform_tool_calls(llm_settings,
                                                             message=assistant_message,
                                                             on_call_start=on_call_lowlevel_start,
                                                             on_call_done=on_call_lowlevel_done,
                                                             maybe_refusal_text=maybe_refusal_text)

    for tool_response_record in tool_response_records:
        _record_grounding(tool_context, tool_response_record)

        def create_tool_payload() -> Dict:
            # OAI spec puts the tool-call linkage on the tool-response *message* as `tool_call_id` (matching the
            # `id` of the assistant's `tool_calls[i]` entry). The tool *execution* metadata (status, function
            # name, timing) stays in `generation_metadata`.
            if "tool_call_id" in tool_response_record:
                tool_response_record.data["tool_call_id"] = tool_response_record.tool_call_id

            # A long fetched document goes to a sidecar, leaving an excerpt and a chip in the log. This runs
            # before the payload is built, because it rewrites the message content and produces the sidecar
            # provenance that goes beside it.
            sidecar_metadata_by_filename = _attachmentify_tool_result(datastore, tool_response_record)

            payload = chatutil.create_payload(llm_settings=llm_settings,
                                              message=tool_response_record.data)
            if sidecar_metadata_by_filename:
                payload["general_metadata"]["sidecars"] = sidecar_metadata_by_filename

            generation_metadata = {"status": tool_response_record.status}  # status is "success" or "error"
            if "function_name" in tool_response_record:
                generation_metadata["function_name"] = tool_response_record.function_name
            if "dt" in tool_response_record:
                generation_metadata["dt"] = tool_response_record.dt  # elapsed wall time, seconds
            if "tool_metadata" in tool_response_record:  # structured metadata the entrypoint attached (e.g. webfetch_denied_host)
                generation_metadata.update(tool_response_record.tool_metadata)

            payload["generation_metadata"] = generation_metadata
            return payload

        tool_response_message_node_id = datastore.create_node(payload=create_tool_payload(),
                                                              parent_id=head_node_id)
        head_node_id = tool_response_message_node_id

        if on_tool_done is not None:
            on_tool_done(head_node_id)

    if on_tools_done is not None:
        on_tools_done()
    return head_node_id


def ai_turn(llm_settings: env,
            datastore: chattree.Forest,
            retriever: "Optional[hybridir.HybridIR]",
            head_node_id: str,
            internet_enabled: bool,
            continue_: bool,
            docs_enabled: bool,
            docs_query: Optional[str],
            docs_num_results: Optional[int],
            markup: Optional[str],
            on_docs_start: Optional[Callable],
            on_docs_done: Optional[Callable],
            on_prompt_ready: Optional[Callable],
            on_llm_start: Optional[Callable],
            on_llm_progress: Optional[Callable],
            on_llm_done: Optional[Callable],
            on_tools_start: Optional[Callable],
            on_call_lowlevel_start: Optional[Callable],
            on_call_lowlevel_done: Optional[Callable],
            on_tool_done: Optional[Callable],
            on_tools_done: Optional[Callable],
            tool_context: Optional[env] = None,
            tools_enabled: bool = True,
            use_character_card: bool = True) -> str:
    """AI's turn: LLM generation interleaved with tool responses, until there are no tool calls in the LLM's latest reply.

    This continues the current branch with as many chat nodes as needed: one for each LLM response, and one for each tool call.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.

    `datastore`: The chat datastore.

    `retriever`: A `raven.librarian.hybridir.HybridIR` retriever connected to the document database,
                 if there is a document database.

    `head_node_id`: Current HEAD node of the chat. Used as the parent for the no-match message, if needed.

    `internet_enabled`: Whether the LLM may reach the network — the user-facing "Internet" switch. Gates
                        `llm_settings.network_tool_names` (`websearch`, `webfetch`) and nothing else.

                        It and `docs_enabled` each own one group of tools outright, so the four combinations
                        all mean something. A tool answering to neither switch — `get_current_time` — is
                        always offered.

    `continue_`: If `False` (default), generate a new AI message. Most of the time, this is what you want.
                 A new chat node is created.

                 If `True`, continue an incomplete AI message, which must be the message at `head_node_id`.
                 The chat node will be updated with the continued message, creating a new revision.
                 The new revision is set as active. The old revision is not removed.

    `docs_enabled`: Whether the document database is in play at all this turn — the user-facing "Documents"
                    switch. When `False`, no automatic search runs (whatever `docs_query` says) and the
                    document tools are not offered to the LLM.

                    Distinct from `docs_query` because the two answer different questions: this one is
                    *may the documents be used*, `docs_query` is *search for this before the model runs*.
                    They collapsed into one switch while the automatic search was the only way to reach the
                    documents; with the tools, a turn that performs no automatic search may still legitimately
                    let the model search for itself — which is exactly what a continuation turn does.

    `docs_query`: Optional query string to search with in the document database.

                  If this, `retriever` and `docs_enabled` are all supplied, `retriever` is queried, and the
                  search results are injected into the context before sending the context to the LLM.

                  If `None`, no automatic search is performed. Note this does not withdraw the document
                  tools; use `docs_enabled=False` for that.

                  NOTE: The official way to NOT search for anything, when you have a document database,
                  is to set `docs_query=None`. If you instead disconnect by setting `retriever=None`,
                  a warning will be logged every time `docs_query` is supplied (because a query requires
                  a retriever).

    `docs_num_results`: How many `docs_query` results to return, at most. Used only if `docs_query` is supplied.

                        If not supplied, use the default of `_search_docs`, which see.

    `markup`: Markup type to use for marking thought blocks, or `None` for no markup. One of:
        "ansi": ANSI terminal color codes.
        "markdown": Markdown markup, with HTML tags for colors.
        `None` (the special value): no markup, keep thought blocks as-is.

    We provide the following optional callbacks/events, which are useful for live UI updates.

    `on_docs_start`: 0-argument callable.
                     The return value is ignored.

                     Called just before searching the document database. Meant as an optional UI hook
                     to show that the document database search (RAG) is starting.

                     Only called if `docs_query is not None`.

    `on_docs_done`: 1-argument callable, with argument `matches: List[Dict]`. For the exact format,
                    see `raven.librarian.hybridir.HybridIR.query`; this is the return value from that.
                    Note that `matches` may be empty.

                    The return value of the event is ignored.

                    Called just after searching the document database. Meant as an optional UI hook
                    to show that the document database search (RAG) is completed.

                    Only called if `docs_query is not None`.

    `on_llm_start`: 0-argument callable. Called just before we call `llmclient.invoke` and the LLM starts
                    parsing the prompt, and eventually streaming a response.
                    The return value is ignored.

                    The LLM will start once at the beginning of the AI's turn, and then once after each set
                    of tool calls.

    `on_prompt_ready`: 1-argument callable, with argument `history: List[Dict]`. Debug/info hook.
                       The return value is ignored.

                       Called after the LLM context has been completely prepared, before sending it to the LLM.

                       This is the modified history, after including document search results and temporary injects,
                       and after scrubbing thought blocks.

                       Each element of the list is a chat message in the format accepted by the LLM backend,
                       with "role" and "content" fields.

    `on_llm_progress`: 1-argument callable taking a typed stream event `event: Dict`; forwarded verbatim to
                       `llmclient.invoke`'s `on_progress` (which see for the event shapes — `content`,
                       `reasoning`, `tool_call`). Called while streaming the response, typically once per
                       generated token. `invoke` is the single parser; this callback just dispatches on
                       `event["type"]`.

           Return value: `action_ack` to let the LLM keep generating, `action_stop` to interrupt and finish forcibly.

           If you interrupt the LLM by returning `action_stop`, normal finalization still takes place, and you'll get
           a chat message populated with the content received so far. It is up to the caller what to do with that data.

    `on_llm_done`: 1-argument callable, with argument `node_id: str`.
                   The return value is ignored.

                   Called after the LLM is done writing and the new chat node has been added to the chat datastore.
                   If there are tool calls in the LLM response, this is called before the tool calls are processed
                   (and before `on_tools_start`).

                   The argument is the node ID of this new chat node.

    `on_tools_start`: 1-argument callable, with argument `tool_calls: List[Dict]`, containing the raw tool call requests
                      in the OpenAI format.

                      Called just before processing the tool calls.

                      The return value is ignored.

                      This is called ONLY IF there is at least one tool call in the LLM's response.

                      This is meant as an optional UI hook to show that tool calls will be processed next.

                      Each completed tool call (regardless of whether success or failure) then triggers
                      one `on_tool_done` event, in a postprocessing loop that creates the chat nodes.

                      After *all* tool calls have completed, the `on_tools_done` (note plural) event triggers.

    `on_call_lowlevel_start`: Called when a tool call has been successfully parsed and the
                              tool is about to be invoked.

                              Main use case is to turn on tool-specific GUI indicators.

                              See `llmclient.perform_tool_calls` for arguments.

    `on_call_lowlevel_done`: Called when a tool call is completed, or when it has failed.

                             Called also for broken tool call requests, without a corresponding
                             `on_call_start`, in order to report the error.

                             Main use case is to turn off tool-specific GUI indicators.

                             See `llmclient.perform_tool_calls` for arguments.

    `on_tool_done`: 1-argument callable, with argument `node_id: str`.
                    The return value is ignored.

                    Called *after* `on_llm_done`, once per tool call result, if there were tool calls,
                    after the tool's response chat node has been added to the chat datastore.

                    The argument is the node ID of this new chat node.

                    Note that all tools have already run when the first `on_tool_done` is called,
                    because the chat nodes are created in a postprocessing loop.

                    If you need an event that triggers when a tool is about to start or has just finished,
                    use `on_call_lowlevel_start` and `on_call_lowlevel_done` instead.

    `on_tools_done`: 0-argument callable.
                     The return value is ignored.

                     Called just after the last tool call has completed.

                     This is called ONLY IF there is at least one tool call in the LLM's response.

                     This is meant as an optional UI hook to show that tool calls have finished processing.

    `tool_context`: Not for application code, which should leave this at `None` so that the turn gets a
                    fresh context (see `make_tool_context`). It exists for callers that are *continuing*
                    a turn already in progress — `retry_tool_calls` runs a tool call of its own before
                    handing control back here, and its result must keep counting toward this turn's
                    accumulated state rather than being forgotten at the handover.

    `tools_enabled`: Whether to offer the LLM any tools at all. `True` (default) is the normal agent loop,
                     with `internet_enabled` and `docs_enabled` deciding which groups are on offer.

                     `False` withdraws every tool, which those two switches cannot express between them:
                     `get_current_time` answers to neither, so it is offered even with both off. A turn with
                     no tools cannot ask for one, so the agent loop runs exactly once — which is what a
                     one-shot scripted completion wants.

                     A blanket switch like this was removed from here once, when the user-facing "Tools"
                     toggle was replaced by the two group switches — on the grounds that a GUI user is never
                     served by one, since it overruled the switch named after the thing it overruled. That
                     reasoning is about a *toggle a person operates*, and it still holds: nothing here is
                     reachable from the GUI. What this parameter says is what kind of call the caller is
                     making, which is a thing only a caller can know.

                     Deliberately a boolean rather than the tool-name list the old marker wished for. A list
                     would decide the same question as the group switches by a second route, and two
                     mechanisms disagreeing about which tools are on offer is the incoherence those switches
                     were introduced to end. Spelled as `llmclient.invoke`'s parameter of the same name and
                     meaning, which it feeds.

    Returns the new HEAD node ID (i.e. the last chat node that was just added).
    """
    # Sanity check
    if continue_:
        head_node_payload = datastore.get_payload(head_node_id)
        if head_node_payload["message"]["role"] != "assistant":
            error_message = f"node '{head_node_id}' is not an AI message (role is '{head_node_payload['message']['role']}'), cannot continue it."
            logger.error(f"ai_turn: {error_message}")
            raise ValueError(error_message)

    documents_available = docs_enabled and retriever is not None

    # Search document database if requested
    if documents_available and docs_query is not None:
        if on_docs_start is not None:
            on_docs_start()
        docs_matches = []  # bound before the `try` so the `finally` can report it even if the search raises
        try:
            docs_matches = _search_docs(retriever=retriever,
                                        query=docs_query,
                                        k=docs_num_results)
        finally:
            # Ensure `on_docs_done` always fires - including when the search raises mid-flight - so GUI
            # state (e.g. `_docs_reading`) recovers cleanly.
            if on_docs_done is not None:
                on_docs_done(docs_matches)
    else:
        if retriever is None and docs_query is not None:
            logger.warning("ai_turn: A `docs_query` was supplied without a `retriever` to search with. Ignoring the query.")
        docs_matches = []

    if tool_context is None:  # normal case; `retry_tool_calls` passes the context it already started
        # The retriever goes in only when the documents are actually in play, so that its presence is the
        # single gate the document tools read. Fails closed: a model that calls a tool we did not advertise
        # finds no retriever there and gets a refusal, rather than reaching around the user's switch.
        tool_context = make_tool_context(llm_settings=llm_settings,
                                         retriever=(retriever if documents_available else None))
        # Material an earlier turn's tools brought in is still sitting in the context, so it still grounds.
        tool_context.grounded = _branch_grounding_is_present(datastore, head_node_id)
    if docs_matches:  # the auto-search grounds this turn as much as a tool call would
        tool_context.grounded = True

    # What this branch has already read, for the model to ask about and for the inject to push. Computed
    # once per turn, from the branch as it stood when the turn began: a document a tool reaches for later in
    # this same turn is still written out in full further down the history, so listing it as well would say
    # the same thing twice in the one place that has to stay compact.
    if documents_available:
        tool_context.consulted_documents = llmclient.label_documents(
            retriever,
            _collect_consulted_documents(datastore=datastore,
                                         head_node_id=head_node_id,
                                         exclude_document_ids=[match["document_id"] for match in docs_matches]))

    # Which tools to offer this turn (`None` = all of them; see the helper for why that reading is the
    # permissive one). Shared with the GUI's context prefill, which must warm the same list.
    maybe_tool_names = llmclient.maybe_tool_names_for_turn(llm_settings,
                                                           documents_available=documents_available,
                                                           internet_available=internet_enabled)
    # `None` means every tool, so an empty tuple is the only "nothing on offer" case. There is none today —
    # `get_current_time` answers to neither switch — but the budget machinery below asks "are there tools at
    # all", and asking the list is the answer that stays true if the ungated group ever empties.
    #
    # `tools_enabled` sits above both group switches rather than beside them, which is why it can say what
    # they cannot: with documents and internet both off, `get_current_time` is still offered, so there is no
    # combination of the two that means "no tools at all".
    any_tools_available = tools_enabled and ((maybe_tool_names is None) or bool(maybe_tool_names))

    continue_this_message = continue_  # we need to continue at most the first message in the agent loop
    completed_tool_rounds = 0  # rounds in which tools actually ran
    refused_tool_rounds = 0  # rounds declined because the budget was already spent
    while True:  # LLM agent loop - interleave LLM responses, tool calls and tool call results, until the LLM is done (no more tool calls).
        # Backstop against a model that keeps rephrasing a search that keeps finding nothing. Past the cap
        # the tools stay in the schema and any call is *refused* instead: changing the loadout mid-turn
        # invalidates the backend's KV cache from that point on, and a history calling a tool the current
        # request no longer declares is a shape models see little of in training, whereas a tool answering
        # "not now" is one they see plenty of.
        #
        # Withdrawing them is the terminator of last resort, and it has to exist, because a refusal cannot
        # by itself guarantee the loop ends. It is a `tools_enabled=False` invocation rather than a `break`:
        # breaking would leave the turn's last message a tool result, which reads as a paused agent loop and
        # is answered with yet another tool call, whereas offering no tools leaves the model no move except
        # to reply.
        budget_spent = completed_tool_rounds >= librarian_config.max_tool_call_rounds
        tools_offered = any_tools_available and (not budget_spent or
                                                 refused_tool_rounds < librarian_config.max_tool_call_refusal_rounds)
        if any_tools_available and not tools_offered:
            logger.info(f"ai_turn: tool-call round cap ({librarian_config.max_tool_call_rounds}) reached and "
                        f"{refused_tool_rounds} refusal round(s) did not end the turn; "
                        "requesting the final reply with no tools offered.")
        message_history = chatutil.linearize_chat(datastore=datastore,
                                                  node_id=head_node_id)

        # Prepare the final LLM prompt, by including the temporary injects (the document search results, too).
        message_history = build_turn_prompt(llm_settings=llm_settings,
                                            history=message_history,
                                            docs_query=docs_query,
                                            docs_matches=docs_matches,
                                            tool_context=tool_context,
                                            # Told the moment the budget runs out, not the moment the tools go
                                            # away — the point of the notice is to make the doomed call
                                            # unnecessary, which is too late once the model has already tried it.
                                            tools_are_spent=(any_tools_available and budget_spent),
                                            tools_enabled=tools_enabled,
                                            use_character_card=use_character_card)

        if on_llm_start is not None:
            on_llm_start()
        try:
            out = llmclient.invoke(settings=llm_settings,
                                   history=message_history,
                                   on_prompt_ready=on_prompt_ready,
                                   on_progress=on_llm_progress,  # this handles `action_stop` from `on_llm_progress`
                                   tools_enabled=tools_offered,
                                   tool_names=maybe_tool_names,
                                   continue_=continue_this_message,
                                   datastore=datastore)  # resolve any sidecar: image refs to data: URLs on the wire
        except Exception as exc:  # noqa: BLE001 -- any backend failure becomes a visible, rerollable message rather than a silent crash
            # Materialize the failure as a synthetic assistant message (rerollable — reroll re-runs the turn).
            # We fire `on_llm_done` because `on_llm_start` already created a streaming message in the GUI,
            # and that is the event which demolishes it.
            logger.error(f"ai_turn: LLM backend invocation failed: {type(exc)}: {exc}")
            # No emoji/symbol glyphs and no fenced code block here: the chat renders via `dpg_markdown`, whose
            # fonts lack emoji (a bare box would show) and which doesn't render ``` code fences (they show up
            # literally). Bold carries the emphasis; the backend's reason string goes in as plain text.
            error_text = ("**The language model backend returned an error, so no reply could be generated.**\n\n"
                          f"Reason: {exc}\n\n"
                          "You can reroll this message to try again.")
            head_node_id = _create_synthetic_assistant_node(llm_settings=llm_settings,
                                                            datastore=datastore,
                                                            parent_node_id=head_node_id,
                                                            text=error_text,
                                                            add_persona=use_character_card)
            if on_llm_done is not None:
                on_llm_done(head_node_id)
            return head_node_id
        # `out.data` is now the complete message object (in the format returned by `create_chat_message`)

        # Clean up the LLM's reply (heuristically). This version goes into the chat history.
        # Content-parts: the reply carries a single text part; scrub its text, re-wrap as a text part.
        #
        # The persona is prefixed only when there is a character to name. Without one the model was never
        # told it was anybody, so a "<char>: " in front of its reply is a claim about the run that is not
        # true - and it reaches whatever reads the stored text, which for a scripted turn is a parser.
        scrubbed_text = chatutil.scrub(persona=llm_settings.personas.get("assistant", None),
                                       text=chatutil.content_to_text(out.data["content"]),
                                       thoughts_mode="keep",
                                       markup=markup,
                                       add_persona=use_character_card)
        out.data["content"] = [chatutil.text_content_part(scrubbed_text)]

        # Add the LLM's message to the chat.
        #
        # Note the token count of the message actually saved into the chat log may be different from `out.n_tokens`, e.g. if the AI is interrupted.
        # However, to correctly compute the generation speed (which is done by the GUI, based on the data we store here), we need to use the original count
        # before any editing, since `out.dt` was measured for that.
        def create_ai_payload() -> Dict:
            payload = chatutil.create_payload(llm_settings=llm_settings,
                                              message=out.data)
            payload["generation_metadata"] = {"model": out.model,
                                              "n_tokens": out.n_tokens,
                                              "dt": out.dt}
            # Record whether this reply had anything to stand on besides the model's own knowledge, so the
            # GUI can say so.
            #
            # Only recorded when the documents are in play, and that condition is the whole content of the
            # marker's honesty: with documents switched off, "no sources retrieved" would announce what the
            # user just chose, and the state that is actually worth reporting - documents on, nothing came
            # back - would be indistinguishable from it. An attachment still grounds a reply either way, so
            # it is read regardless. Absent means "nothing to say", which beats a third state.
            #
            # Not a guard: it does not withhold the reply. The guard it replaces could not tell a question
            # about the documents from a general-knowledge aside, and in the default configuration answered
            # "what is 2+2?" with "No matches in document database."
            attachment_grounds = _attachment_is_present(message_history)
            if documents_available or attachment_grounds:
                payload["generation_metadata"]["grounded"] = bool(tool_context.grounded or attachment_grounds)
            if docs_query is not None:
                payload["retrieval"] = {"query": docs_query,
                                        "results": docs_matches}  # store RAG results in the chat node that was generated based on them, for later use (upcoming citation mechanism)
            return payload
        if not continue_this_message:  # new message (usual case)
            ai_message_node_id = datastore.create_node(payload=create_ai_payload(),
                                                       parent_id=head_node_id)
        else:  # continue existing message
            ai_message_node_id = head_node_id
            datastore.add_revision(node_id=ai_message_node_id,
                                   payload=create_ai_payload())
            continue_this_message = False  # any further messages during this AI turn should be created normally
        head_node_id = ai_message_node_id
        if on_llm_done is not None:
            on_llm_done(head_node_id)

        # Handle tool calls, if any.
        #
        # Call the tool(s) specified by the LLM, with arguments specified by the LLM, and add the result to the chat.
        #
        # Each response goes into its own message, with `role="tool"`.
        #
        have_tool_calls = (out.data["tool_calls"] is not None and len(out.data["tool_calls"]))
        if have_tool_calls:
            head_node_id = _perform_and_store_tool_calls(llm_settings=llm_settings,
                                                         datastore=datastore,
                                                         assistant_message=out.data,
                                                         parent_node_id=head_node_id,
                                                         tool_context=tool_context,
                                                         maybe_refusal_text=(llm_settings.formatters.error_that_tools_are_spent() if budget_spent else None),
                                                         on_tools_start=on_tools_start,
                                                         on_call_lowlevel_start=on_call_lowlevel_start,
                                                         on_call_lowlevel_done=on_call_lowlevel_done,
                                                         on_tool_done=on_tool_done,
                                                         on_tools_done=on_tools_done)
            if budget_spent:
                refused_tool_rounds += 1
            else:
                completed_tool_rounds += 1
        else:
            # When there are no more tool calls, the LLM is done replying.
            break

    return head_node_id


def _next_tool_node_on_branch(datastore: chattree.Forest, node_id: str) -> Optional[str]:
    """Return the (single) `role="tool"` child of `node_id`, or `None`.

    A tool-result node created by the agent loop has at most one tool-role child (the next tool result
    of the same assistant turn); the assistant's reply that follows the tool round is `role="assistant"`,
    which stops the walk. Used to collect the suffix of a tool-call chain in `retry_tool_calls`.
    """
    for child_id in datastore.get_children(node_id):
        if datastore.get_payload(child_id)["message"]["role"] == "tool":
            return child_id
    return None


def retry_tool_calls(llm_settings: env,
                     datastore: chattree.Forest,
                     retriever: "Optional[hybridir.HybridIR]",
                     tool_node_id: str,
                     internet_enabled: bool,
                     docs_enabled: bool,
                     markup: Optional[str],
                     docs_num_results: Optional[int],
                     on_docs_start: Optional[Callable] = None,
                     on_docs_done: Optional[Callable] = None,
                     on_prompt_ready: Optional[Callable] = None,
                     on_llm_start: Optional[Callable] = None,
                     on_llm_progress: Optional[Callable] = None,
                     on_llm_done: Optional[Callable] = None,
                     on_tools_start: Optional[Callable] = None,
                     on_call_lowlevel_start: Optional[Callable] = None,
                     on_call_lowlevel_done: Optional[Callable] = None,
                     on_tool_done: Optional[Callable] = None,
                     on_tools_done: Optional[Callable] = None) -> str:
    """Re-run a single previously-denied tool call on a NEW branch, then continue the AI's turn.

    This is the backend of the GUI "approve this denied host & retry" override. The user has just approved
    a host (via `llmclient.approve_host_for_session`) that `webfetch` refused; this re-runs *only* that one
    call so the now-allowed fetch can succeed, WITHOUT re-invoking the LLM — the AI's decision to call those
    tools is preserved.

    `tool_node_id` is the denied `role="tool"` node (the one carrying `webfetch_denied_host` in its
    `generation_metadata`). Mechanism:

      1. Walk up past the contiguous tool-result chain to the assistant that requested the calls, and read
         that one call (matched by `tool_call_id`) from its `tool_calls`.
      2. Re-run ONLY that call, as a new sibling of the old denied node (branching at its parent). Every
         other tool result of the same turn is preserved verbatim, NOT re-run: the nodes *before* the denied
         one are shared ancestors of the new branch, and any *after* it are copied across (step 3). This is
         deliberate — re-running a websearch would re-query the engine (the server-side `@memoize` is in-RAM
         and empty after a restart / on a chat reloaded from disk), yielding a SERP the model never reasoned
         about. "Approve this fetch" must change only this fetch.
      3. Copy the suffix tool results (those after the denied one in the same turn — rare; present only if
         the model ordered another call after `webfetch` in one message) onto the new branch. The turn's
         calls are issued together, so a suffix result cannot depend on the re-run call's new output.
      4. Continue from the rebuilt tool head via `ai_turn(continue_=False)` — the LLM responds to the now-
         complete results and the agent loop proceeds. No new *automatic* RAG search (`docs_query=None`,
         matching loop continuation), but the document tools stay available if `docs_enabled` — a
         continuation is mid-turn, and a tool that vanishes between rounds of one agent loop is a shape
         models read as noise.

    Returns the new HEAD node ID.
    """
    denied_payload = datastore.get_payload(tool_node_id)
    if denied_payload["message"]["role"] != "tool":
        raise ValueError(f"retry_tool_calls: node '{tool_node_id}' is not a tool-result node (role is '{denied_payload['message']['role']}').")
    denied_tool_call_id = denied_payload["message"].get("tool_call_id")  # OAI: linkage lives on the tool message

    # 1. Walk up the tool-result chain to the assistant that requested the calls.
    parent_node_id = datastore.get_parent(tool_node_id)
    assistant_node_id = parent_node_id
    while assistant_node_id is not None and datastore.get_payload(assistant_node_id)["message"]["role"] == "tool":
        assistant_node_id = datastore.get_parent(assistant_node_id)
    if assistant_node_id is None:
        raise ValueError(f"retry_tool_calls: could not find the tool-calling assistant above tool node '{tool_node_id}'.")
    assistant_message = datastore.get_payload(assistant_node_id)["message"]
    all_tool_calls = assistant_message.get("tool_calls") or []

    # Resolve the single call to re-run. Match by stored tool_call id; fall back to the lone call only if
    # the assistant issued exactly one (older nodes may predate the stored id).
    if denied_tool_call_id is not None:
        calls_to_rerun = [tc for tc in all_tool_calls if tc.get("id") == denied_tool_call_id]
    else:
        calls_to_rerun = list(all_tool_calls) if len(all_tool_calls) == 1 else []
    if not calls_to_rerun:
        raise ValueError(f"retry_tool_calls: could not match denied tool node '{tool_node_id}' to a call on assistant '{assistant_node_id}'.")

    # 3 (collect, before mutating). The suffix tool nodes after the denied one on this branch.
    suffix_node_ids: List[str] = []
    node_id = _next_tool_node_on_branch(datastore, tool_node_id)
    while node_id is not None:
        suffix_node_ids.append(node_id)
        node_id = _next_tool_node_on_branch(datastore, node_id)

    # 2. Re-run only the denied call, as a new sibling branch under the assistant's tool chain. Fire
    #    `on_tools_start` (GUI: tools starting), but defer `on_tools_done` until the suffix is also in place.
    synthetic_message = {**assistant_message, "tool_calls": calls_to_rerun}
    # Handed to `ai_turn` below, so the re-run call's grounding carries across the handover instead of being
    # forgotten. Same gate as `ai_turn` applies: no retriever in the context unless the documents are in play.
    tool_context = make_tool_context(llm_settings=llm_settings,
                                     retriever=(retriever if docs_enabled else None))
    if docs_enabled:  # the re-run call may be, or may lead to, a `list_consulted_documents`
        tool_context.consulted_documents = llmclient.label_documents(
            retriever,
            _collect_consulted_documents(datastore=datastore,
                                         head_node_id=parent_node_id,
                                         exclude_document_ids=[]))
    head_node_id = _perform_and_store_tool_calls(llm_settings=llm_settings,
                                                 datastore=datastore,
                                                 assistant_message=synthetic_message,
                                                 parent_node_id=parent_node_id,
                                                 tool_context=tool_context,
                                                 on_tools_start=on_tools_start,
                                                 on_call_lowlevel_start=on_call_lowlevel_start,
                                                 on_call_lowlevel_done=on_call_lowlevel_done,
                                                 on_tool_done=on_tool_done,
                                                 on_tools_done=None)

    # 3. Copy the suffix tool results verbatim onto the new branch (reboot-safe: not re-fetched).
    #    `copy_node` duplicates the whole node (full revision history + names + timestamp), not just the
    #    active payload — the Forest-aware way to clone a node into a new place.
    for old_node_id in suffix_node_ids:
        head_node_id = datastore.copy_node(old_node_id, new_parent_id=head_node_id)
        if on_tool_done is not None:
            on_tool_done(head_node_id)
    if on_tools_done is not None:
        on_tools_done()

    # 4. Continue the AI turn from the rebuilt tool head.
    return ai_turn(llm_settings=llm_settings,
                   datastore=datastore,
                   retriever=retriever,
                   head_node_id=head_node_id,
                   internet_enabled=internet_enabled,
                   continue_=False,
                   docs_enabled=docs_enabled,
                   docs_query=None,
                   docs_num_results=docs_num_results,
                   markup=markup,
                   on_docs_start=on_docs_start,
                   on_docs_done=on_docs_done,
                   on_prompt_ready=on_prompt_ready,
                   on_llm_start=on_llm_start,
                   on_llm_progress=on_llm_progress,
                   on_llm_done=on_llm_done,
                   on_tools_start=on_tools_start,
                   on_call_lowlevel_start=on_call_lowlevel_start,
                   on_call_lowlevel_done=on_call_lowlevel_done,
                   on_tool_done=on_tool_done,
                   on_tools_done=on_tools_done,
                   tool_context=tool_context)
