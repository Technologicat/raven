"""Scaffolding for a multi-turn conversation with automatic RAG search and tool-calling."""

__all__ = ["user_turn",
           "ai_turn", "retry_tool_calls", "action_ack", "action_stop"]

import logging
logger = logging.getLogger(__name__)

import json

from typing import Callable, Dict, List, Optional, TYPE_CHECKING

from unpythonic import dyn, sym, Values
from unpythonic.env import env

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

action_continue = sym("continue")  # continue this turn (e.g. when docs were searched and at least one match was found)
action_done = sym("done")  # this turn (user/AI) is complete

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
                                     text: str) -> str:
    """Create and persist a *synthetic* assistant message node, returning its id.

    "Synthetic" = the text is authored by Raven, not generated by the LLM. Two cases use this: the no-match
    notice (docs on, no results, speculation off — the anti-hallucination bypass) and the backend-error notice
    (the LLM call raised). Both want the same shape, so it lives here so the two can't drift.

    Like a real assistant reply it is a `role="assistant"` node, so the usual message affordances apply —
    notably reroll, which re-runs the AI turn from the parent and thus doubles as "retry". It carries no
    `generation_metadata` (there was no generation), so the renderer shows no token-stats line, which is how
    these non-generated messages have always appeared.

    Callers add their own payload fields afterward if needed (e.g. the no-match node's `retrieval`) and fire
    whichever GUI done-callback fits their path.
    """
    message = chatutil.create_chat_message(llm_settings=llm_settings,
                                           role="assistant",
                                           text=text)
    return datastore.create_node(payload=chatutil.create_payload(llm_settings=llm_settings,
                                                                 message=message),
                                 parent_id=parent_node_id)


def _search_docs_with_bypass(llm_settings: env,
                             datastore: chattree.Forest,
                             retriever: "hybridir.HybridIR",
                             head_node_id: str,
                             speculate: bool,
                             query: str,
                             k: Optional[int] = None) -> Values:
    """Helper for `ai_turn`. Search the document database (`retriever`) for `query`, returning `k` best matches.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.
                    Contains (among other things) a mapping of roles to persona names.

    `datastore`: The chat datastore. Used for adding a no-match chat message when the search comes up empty
                 and `speculate` is `False`. In such situations, the no-match message is used the response,
                 instead of invoking the LLM.

    `retriever`: A `raven.librarian.hybridir.HybridIR` retriever connected to the document database.

    `head_node_id`: Current HEAD node of the chat. Used as the parent for the no-match message, if needed.

    `speculate`: If `False`, and the search returns no matches, bypass the LLM, and creating a no-match chat node.
                 If `True`, always just return the search results.

    `query`: The query string to search with in the document database. (Note "with", not "for"; the query may
             undergo processing. As of v0.2.3, it is directly tokenized for keyword search, but the semantic
             search uses the "qa" role, which maps questions and possible corresponding answers near each other.)

    `k`: Return up to this many best matches. Note that there is an internal threshold, which automatically drops
         any very low-quality semantic matches.

         The default `None` means `k=10`.

    If there are no matches, add a no-match message to the chat log (to be shown instead of the AI's reply).
    """
    if k is None:
        k = 10
    docs_results = retriever.query(query,
                                   k=k,
                                   return_extra_info=False)

    # First line of defense (against hallucinations): docs on, no matches for given query, speculate off -> bypass LLM
    if not docs_results and not speculate:
        nomatch_text = "No matches in document database. Please try another query."
        nomatch_message_node_id = _create_synthetic_assistant_node(llm_settings=llm_settings,
                                                                   datastore=datastore,
                                                                   parent_node_id=head_node_id,
                                                                   text=nomatch_text)
        nomatch_message_node_payload = datastore.get_payload(nomatch_message_node_id)  # get current revision (which is the only revision since we just created the node)
        nomatch_message_node_payload["retrieval"] = {"query": query,
                                                     "results": []}  # store RAG results in the chat node that was generated based on them, for later use (upcoming citation mechanism)
        return Values(action=action_done, new_head_node_id=nomatch_message_node_id)

    # Whether we got any results or not, return them to the caller and let the caller proceed.
    return Values(action=action_continue, matches=docs_results)


def _context_is_present(history: List[Dict],
                        docs_matches: List[Dict]) -> bool:
    """Return whether the AI has any material to ground an answer in, beyond its own static knowledge.

    Call this *before* adding any injects to `history`; the injects we add are themselves context-shaped,
    so afterwards the answer is always `True` and the question no longer means anything.

    Counted as context: this turn's document-database matches, any attachment the user has made anywhere
    on this branch (an image or a document), and tool results *from this turn* (a web search, a fetched
    page). The two are scoped differently on purpose, because they age differently. An attachment is
    material sitting in the context, and stays usable for as long as it is in the window. A tool result
    is the answer to one specific earlier question, and goes stale with it: last week's weather lookup is
    no grounding at all for today's question about instrument calibration, so treating it as context would
    leave the reminder stuck on for the rest of the conversation. Scoping tool results to the turn that
    asked for them follows the same one-hop rule as `chatutil.compute_auto_allowed_hosts`.

    Not counted: the conversation itself, nor the AI's own earlier replies - a model summarizing what it
    previously said is exactly the ungrounded answer this is used to guard against.
    """
    if docs_matches:
        return True
    latest_user_position = max((position for position, message in enumerate(history) if message["role"] == "user"),
                               default=-1)
    for position, message in enumerate(history):
        if message["role"] == "tool" and position > latest_user_position:  # this turn's tool results only
            return True
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

def _perform_injects(llm_settings: env,
                     history: List[Dict],  # mutated!
                     speculate: bool,
                     docs_query: Optional[str],
                     docs_matches: List[Dict]) -> None:
    """Perform the temporary injects to prepare for the AI's turn.

    These are not meant to be persistent, so we don't even add them to the datastore,
    but only insert them into the temporary linearized history that is fed to the LLM.

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
    `briefs/context-inject-shape-measurements.md`.

    The position is also why this needs no `continue_` flag: when continuing the AI's interrupted message,
    the history must look as it did at the moment of interruption, and everything we add sits ahead of the
    user's message, which is ahead of the message being continued either way.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.
                    Contains (among other things) a mapping of roles to persona names.

    `history`: Linearized message history in the OpenAI format sent to the LLM.

    `speculate`: If `False`, and there is context to ground an answer in, remind the LLM to base its claims
                 about that context on the context. With nothing to ground in, the reminder is skipped -
                 asking a model to stick to documents that were never provided is a contradiction it will
                 dutifully try to resolve, at a cost of up to 37x the deliberation, sometimes never
                 terminating at all.

    `docs_query`: The query string the document database was searched with, or `None` if it wasn't searched.
                  Reported to the model as the arguments of the synthetic search call, so that the matches
                  arrive as the answer to a legible question rather than as free-floating material.

    `docs_matches`: Docs search matches returned by `HybridIR` (see `_search_docs_with_bypass`).
    """
    grounding_material_exists = _context_is_present(history, docs_matches)  # ask before we add any ourselves

    # Instruction-like injects -> leading system message.
    system_injects = [chatutil.format_date_now(),
                      chatutil.format_reminder_to_write_conversationally()]
    if not speculate and grounding_material_exists:
        system_injects.append(chatutil.format_reminder_to_use_information_from_context_only())
    _add_to_system_message(llm_settings=llm_settings,
                           history=history,
                           texts=system_injects)

    # Data-like injects -> synthetic tool calls, placed just before the user's latest message.
    data_injects = _synthetic_tool_exchange(llm_settings=llm_settings,
                                            call_id="raven_clock",
                                            function_name="get_current_time",
                                            arguments={},
                                            result_text=chatutil.format_time_now())
    if docs_matches:
        # Asked about something the matches do not cover, the model sometimes reaches for another search -
        # and since this search is Raven's own, run before the turn began, there is no such tool for it to
        # reach for. It then writes the call out as literal text, and the user gets that instead of an
        # answer (~1 turn in 3 on Qwen3.6-27B, asking for a figure the documents do not contain).
        #
        # Deliberately unmitigated. Telling the model it may not search again reads as a prohibition, and a
        # prohibition is the thing this whole inject rework exists to stop handing it: that wording ran
        # 29000 characters of deliberation without producing a reply, where saying nothing answered cleanly
        # in 3000. The model's instinct is right - it *should* want a second, better-targeted search - so
        # the fix is to let it have one.
        # TODO (brief 10): Expose the retriever as a real `search_documents` tool the model may call, which
        # TODO (brief 10): turns this failure into the feature it is reaching for. The synthetic call below
        # TODO (brief 10): already uses that name, so it becomes honest rather than a fiction once it lands.
        data_injects.extend(_synthetic_tool_exchange(llm_settings=llm_settings,
                                                     call_id="raven_docs",
                                                     function_name="search_documents",
                                                     arguments={"query": docs_query if docs_query is not None else ""},
                                                     result_text=chatutil.format_docs_matches(docs_matches)))

    for position in range(len(history) - 1, -1, -1):
        if history[position]["role"] == "user":
            break
    else:  # No user message to place the injects ahead of. Nothing to reply to either, so this shouldn't happen.
        logger.warning("_perform_injects: no user message in history; appending injects at the end.")
        position = len(history)
    history[position:position] = data_injects


def _make_tool_context(retriever: "Optional[hybridir.HybridIR]") -> env:
    """Create the per-turn tool-call request context (the `dyn.tool_context` payload).

    One env per AI turn, not per tool round, because two kinds of field live here and only one of them is
    per-round:

      - *Accumulating* fields carry information forward across the rounds of a single turn. `grounded` is
        the case that forces the per-turn lifetime: whether anything this turn gave the model material to
        answer from is a question about the turn, and a tool call in round 1 must still count in round 3.
      - *Volatile* fields are recomputed by `_perform_and_store_tool_calls` before each round, because
        their correct value depends on what the earlier rounds did. `webfetch_allowed_hosts` is one: a
        websearch in round 1 can auto-allow the hosts a webfetch reaches for in round 2.

    Everything here is harness-supplied, never model-supplied - that separation is the point, and it is why
    the retriever is handed over this way rather than being closed over at tool-registration time (it could
    not be: `llmclient.setup` runs before `hybridir.setup` in both clients).

    `retriever`: The document-database retriever the document tools search, or `None` if this app has no
                 document database. The tools are duck-typed against `.query(...)`; see the module header
                 for why `hybridir` is not imported at runtime.
    """
    return env(retriever=retriever,
               webfetch_allowed_hosts=frozenset(),  # volatile: recomputed per round
               grounded=False)  # accumulating: did anything this turn provide grounding material?

def _record_grounding(tool_context: env,
                      tool_response_record: env) -> None:
    """Fold one tool result into `tool_context.grounded`. Monotonic: once grounded, stays grounded.

    Grounding is *declared at the source* rather than inferred from message shape. An entrypoint that
    knows whether its output is material to answer from says so, by returning `(output, {"grounding": ...})`
    instead of a bare output; `webfetch` is the case that has to, because its allowlist refusal is a
    perfectly non-empty string that grounds nothing.

    Undeclared results fall back to "a successful call that returned something is material". That is a
    good default for an unknown tool (notably a future MCP one), and it is wrong only in the direction
    that a tool author can correct by declaring.

    Why this matters enough to have its own mechanism: the reminder to base claims on the provided context
    is only sound when there *is* context. Sent with nothing to ground in, it is a self-contradiction that
    measured 5-37x the deliberation of sending nothing, with one model never terminating at all.
    """
    if tool_context.grounded:  # already grounded; nothing can un-ground it
        return
    if tool_response_record.status != "success":
        return
    declared = (tool_response_record.tool_metadata or {}).get("grounding") if "tool_metadata" in tool_response_record else None
    if declared is None:
        declared = bool(chatutil.content_to_text(tool_response_record.data["content"]).strip())
    tool_context.grounded = bool(declared)

def _perform_and_store_tool_calls(llm_settings: env,
                                  datastore: chattree.Forest,
                                  assistant_message: Dict,
                                  parent_node_id: str,
                                  tool_context: env,
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

    `tool_context`: The turn's request context, from `_make_tool_context` (which see for what belongs in
                    it and why it outlives a single round). Bound to `dyn.tool_context` for the dynamic
                    extent of the dispatch — the request-context pattern (cf. Racket's `parameterize`,
                    Flask's `g`). Entrypoints that need it read `dyn.tool_context`; see the field registry
                    at `llmclient.make_dynvar(tool_context=...)`.

                    Its volatile fields are refreshed here, per round. `webfetch_allowed_hosts` is
                    recomputed from `parent_node_id`, so the walk sees this turn's user message and any
                    prior tool results on the branch — that is what lets a websearch in an earlier round
                    auto-allow the hosts a webfetch reaches for in this one.
    """
    head_node_id = parent_node_id
    if on_tools_start is not None:
        on_tools_start(assistant_message["tool_calls"])

    tool_context.webfetch_allowed_hosts = chatutil.compute_auto_allowed_hosts(
        datastore, head_node_id,
        trust_search_results=librarian_config.webfetch_trust_search_results)

    # Each tool call produces exactly one response. No-ops if the message contains no tool calls.
    with dyn.let(tool_context=tool_context):
        tool_response_records = llmclient.perform_tool_calls(llm_settings,
                                                             message=assistant_message,
                                                             on_call_start=on_call_lowlevel_start,
                                                             on_call_done=on_call_lowlevel_done)

    for tool_response_record in tool_response_records:
        _record_grounding(tool_context, tool_response_record)

        def create_tool_payload() -> Dict:
            # OAI spec puts the tool-call linkage on the tool-response *message* as `tool_call_id` (matching the
            # `id` of the assistant's `tool_calls[i]` entry). The tool *execution* metadata (status, function
            # name, timing) stays in `generation_metadata`.
            if "tool_call_id" in tool_response_record:
                tool_response_record.data["tool_call_id"] = tool_response_record.tool_call_id

            payload = chatutil.create_payload(llm_settings=llm_settings,
                                              message=tool_response_record.data)

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


# TODO: `tools_enabled` is a blunt hammer; maybe have also an optional tool name list for fine-grained control?
def ai_turn(llm_settings: env,
            datastore: chattree.Forest,
            retriever: "Optional[hybridir.HybridIR]",
            head_node_id: str,
            tools_enabled: bool,
            continue_: bool,
            docs_enabled: bool,
            docs_query: Optional[str],
            docs_num_results: Optional[int],
            speculate: bool,
            markup: Optional[str],
            on_docs_start: Optional[Callable],
            on_docs_done: Optional[Callable],
            on_prompt_ready: Optional[Callable],
            on_llm_start: Optional[Callable],
            on_llm_progress: Optional[Callable],
            on_llm_done: Optional[Callable],
            on_nomatch_done: Optional[Callable],
            on_tools_start: Optional[Callable],
            on_call_lowlevel_start: Optional[Callable],
            on_call_lowlevel_done: Optional[Callable],
            on_tool_done: Optional[Callable],
            on_tools_done: Optional[Callable],
            tool_context: Optional[env] = None) -> str:
    """AI's turn: LLM generation interleaved with tool responses, until there are no tool calls in the LLM's latest reply.

    This continues the current branch with as many chat nodes as needed: one for each LLM response, and one for each tool call.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.

    `datastore`: The chat datastore.

    `retriever`: A `raven.librarian.hybridir.HybridIR` retriever connected to the document database,
                 if there is a document database.

    `head_node_id`: Current HEAD node of the chat. Used as the parent for the no-match message, if needed.

    `tools_enabled`: Whether the LLM is allowed to use the tools available in `llmclient.setup`.
                     This can be disabled e.g. to temporarily turn off websearch.

    `continue_`: If `False` (default), generate a new AI message. Most of the time, this is what you want.
                 A new chat node is created.

                 If `True`, continue an incomplete AI message, which must be the message at `head_node_id`.
                 The chat node will be updated with the continued message, creating a new revision.
                 The new revision is set as active. The old revision is not removed.

    `docs_enabled`: Whether the document database is in play at all this turn — the user-facing "docs on/off"
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

                        If not supplied, use the default of `_search_docs_with_bypass`, which see.

    `speculate`: Used only if `docs_query` is supplied.

                 If `False`:

                     If the search returns no matches, bypass the LLM, creating a no-match chat node.

                     If the search returns at least one match, then remind the LLM to base its reply on the
                     information provided in the context only. How well this works depends on the LLM used;
                     Qwen3 2507 30B-A3B mostly seems to do fine.

                 If `True`, allow the LLM to respond regardless.

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

                     NOTE: The role of `on_docs_done` differs from that of `on_nomatch_done`:
                       - `on_docs_done` signals that the documents database search has completed.
                       - `on_nomatch_done` signals that the whole AI turn has completed due to the no-match LLM bypass.

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

    `on_nomatch_done`: 1-argument callable, with argument `node_id: str`.
                       The return value is ignored.

                       Called instead of `on_llm_start`/`on_llm_progress`/`on_llm_done` if the LLM was bypassed,
                       after the new chat node has been added to the chat datastore.

                       The argument is the node ID of this new chat node.

                       NOTE: The role of `on_nomatch_done` differs from that of `on_docs_done`:
                         - `on_docs_done` signals that the documents database search has completed.
                         - `on_nomatch_done` signals that the whole AI turn has completed due to the no-match LLM bypass.

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
                    fresh context (see `_make_tool_context`). It exists for callers that are *continuing*
                    a turn already in progress — `retry_tool_calls` runs a tool call of its own before
                    handing control back here, and its result must keep counting toward this turn's
                    accumulated state rather than being forgotten at the handover.

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
        docs_matches_to_report: List[Dict] = []  # captured for the `finally` so `on_docs_done` always fires
        try:
            docs_result = _search_docs_with_bypass(llm_settings=llm_settings,
                                                   datastore=datastore,
                                                   retriever=retriever,
                                                   head_node_id=head_node_id,
                                                   speculate=speculate,
                                                   query=docs_query,
                                                   k=docs_num_results)
            if docs_result["action"] is action_done:  # no-match bypass triggered, we have a response chat node already
                head_node_id = docs_result["new_head_node_id"]
                if on_nomatch_done is not None:
                    on_nomatch_done(head_node_id)
                return head_node_id
            else:
                docs_matches = docs_result["matches"]
                docs_matches_to_report = docs_matches
        finally:
            # Ensure `on_docs_done` always fires — including when the search raises mid-flight or when
            # the no-match-bypass `return` exits early — so GUI state (e.g. `_docs_reading`) recovers
            # cleanly. With this finally in place, leaving `on_docs_done` out of the bypass branch
            # above is intentional: the finally calls it on the way out.
            if on_docs_done is not None:
                on_docs_done(docs_matches_to_report)
    else:
        if retriever is None and docs_query is not None:
            logger.warning("ai_turn: A `docs_query` was supplied without a `retriever` to search with. Ignoring the query.")
        docs_matches = []

    if tool_context is None:  # normal case; `retry_tool_calls` passes the context it already started
        # The retriever goes in only when the documents are actually in play, so that its presence is the
        # single gate the document tools read. Fails closed: a model that calls a tool we did not advertise
        # finds no retriever there and gets a refusal, rather than reaching around the user's switch.
        tool_context = _make_tool_context(retriever=(retriever if documents_available else None))
    if docs_matches:  # the auto-search grounds this turn as much as a tool call would
        tool_context.grounded = True

    # Which tools to offer this turn (`None` = all of them; see the helper for why that reading is the
    # permissive one). Shared with the GUI's context prefill, which must warm the same list.
    maybe_tool_names = llmclient.maybe_tool_names_for_turn(llm_settings, documents_available=documents_available)

    continue_this_message = continue_  # we need to continue at most the first message in the agent loop
    completed_tool_rounds = 0
    while True:  # LLM agent loop - interleave LLM responses, tool calls and tool call results, until the LLM is done (no more tool calls).
        # Backstop against a model that keeps rephrasing a search that keeps finding nothing. Offering no
        # tools is what ends the loop, rather than breaking out of it: a `break` here would leave the turn's
        # last message a tool result, which reads as a paused agent loop and is answered with yet another
        # tool call. Withdrawing the tools instead leaves the model no move except to reply.
        tools_offered = tools_enabled and completed_tool_rounds < librarian_config.max_tool_call_rounds
        if tools_enabled and not tools_offered:
            logger.info(f"ai_turn: tool-call round cap ({librarian_config.max_tool_call_rounds}) reached; "
                        "requesting the final reply with no tools offered.")
        message_history = chatutil.linearize_chat(datastore=datastore,
                                                  node_id=head_node_id)

        # Prepare the final LLM prompt, by including the temporary injects (the document search results, too).
        _perform_injects(llm_settings=llm_settings,
                         history=message_history,
                         speculate=speculate,
                         docs_query=docs_query,
                         docs_matches=docs_matches)

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
            # Materialize the failure as a synthetic assistant message (rerollable — reroll re-runs the turn),
            # via the same helper as the no-match bypass. Unlike no-match we fire `on_llm_done`, not
            # `on_nomatch_done`, because `on_llm_start` already created a streaming message that `on_llm_done`
            # demolishes.
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
                                                            text=error_text)
            if on_llm_done is not None:
                on_llm_done(head_node_id)
            return head_node_id
        # `out.data` is now the complete message object (in the format returned by `create_chat_message`)

        # Clean up the LLM's reply (heuristically). This version goes into the chat history.
        # Content-parts: the reply carries a single text part; scrub its text, re-wrap as a text part.
        scrubbed_text = chatutil.scrub(persona=llm_settings.personas.get("assistant", None),
                                       text=chatutil.content_to_text(out.data["content"]),
                                       thoughts_mode="keep",
                                       markup=markup,
                                       add_persona=True)
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
                                                         on_tools_start=on_tools_start,
                                                         on_call_lowlevel_start=on_call_lowlevel_start,
                                                         on_call_lowlevel_done=on_call_lowlevel_done,
                                                         on_tool_done=on_tool_done,
                                                         on_tools_done=on_tools_done)
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
                     tools_enabled: bool,
                     docs_enabled: bool,
                     speculate: bool,
                     markup: Optional[str],
                     docs_num_results: Optional[int],
                     on_docs_start: Optional[Callable] = None,
                     on_docs_done: Optional[Callable] = None,
                     on_prompt_ready: Optional[Callable] = None,
                     on_llm_start: Optional[Callable] = None,
                     on_llm_progress: Optional[Callable] = None,
                     on_llm_done: Optional[Callable] = None,
                     on_nomatch_done: Optional[Callable] = None,
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
    tool_context = _make_tool_context(retriever=(retriever if docs_enabled else None))
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
                   tools_enabled=tools_enabled,
                   continue_=False,
                   docs_enabled=docs_enabled,
                   docs_query=None,
                   docs_num_results=docs_num_results,
                   speculate=speculate,
                   markup=markup,
                   on_docs_start=on_docs_start,
                   on_docs_done=on_docs_done,
                   on_prompt_ready=on_prompt_ready,
                   on_llm_start=on_llm_start,
                   on_llm_progress=on_llm_progress,
                   on_llm_done=on_llm_done,
                   on_nomatch_done=on_nomatch_done,
                   on_tools_start=on_tools_start,
                   on_call_lowlevel_start=on_call_lowlevel_start,
                   on_call_lowlevel_done=on_call_lowlevel_done,
                   on_tool_done=on_tool_done,
                   on_tools_done=on_tools_done,
                   tool_context=tool_context)
