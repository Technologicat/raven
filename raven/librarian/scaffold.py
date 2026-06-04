"""Scaffolding for a multi-turn conversation with automatic RAG search and tool-calling."""

__all__ = ["user_turn",
           "ai_turn", "retry_tool_calls", "action_ack", "action_stop"]

import logging
logger = logging.getLogger(__name__)

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
              user_message_text: str) -> str:
    """Add the user's message with content `user_message_text` to `datastore`.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.

    `datastore`: The chat datastore.

    `head_node_id`: Current HEAD node of the chat. Used as the parent for the no-match message, if needed.

    `user_message_text`: The message text to add.

    Returns the new HEAD node ID (i.e. the chat node that was just added).
    """
    # Add the user's message to the chat.
    user_message_node_id = datastore.create_node(payload=chatutil.create_payload(llm_settings=llm_settings,
                                                                                 message=chatutil.create_chat_message(llm_settings=llm_settings,
                                                                                                                      role="user",
                                                                                                                      text=user_message_text)),
                                                 parent_id=head_node_id)
    return user_message_node_id


# --------------------------------------------------------------------------------
# AI's turn

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
        nomatch_message_node_id = datastore.create_node(payload=chatutil.create_payload(llm_settings=llm_settings,
                                                                                        message=chatutil.create_chat_message(llm_settings=llm_settings,
                                                                                                                             role="assistant",
                                                                                                                             text=nomatch_text)),
                                                        parent_id=head_node_id)
        nomatch_message_node_payload = datastore.get_payload(nomatch_message_node_id)  # get current revision (which is the only revision since we just created the node)
        nomatch_message_node_payload["retrieval"] = {"query": query,
                                                     "results": []}  # store RAG results in the chat node that was generated based on them, for later use (upcoming citation mechanism)
        return Values(action=action_done, new_head_node_id=nomatch_message_node_id)

    # Whether we got any results or not, return them to the caller and let the caller proceed.
    return Values(action=action_continue, matches=docs_results)


injectors = [chatutil.format_chat_datetime_now,  # let the LLM know the current local time and date
             chatutil.format_reminder_to_focus_on_latest_input]  # remind the LLM to focus on user's last message (some models such as the distills of DeepSeek-R1 need this to support multi-turn conversation)
def _perform_injects(llm_settings: env,
                     history: List[Dict],  # mutated!
                     continue_: bool,
                     speculate: bool,
                     docs_matches: List[Dict]) -> None:
    """Perform the temporary injects to prepare for the AI's turn.

    These are not meant to be persistent, so we don't even add them to the datastore,
    but only insert them into the temporary linearized history that is fed to the LLM.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.
                    Contains (among other things) a mapping of roles to persona names.

    `history`: Linearized message history in the OpenAI format sent to the LLM.

    `continue_`: Whether to continue AI's last message. Affects the inject position:

                 If `False`, always-on injects are appended to the end. Usually you want this.

                 If `True`, always-on injects are added just before the last message,
                            which is the message being continued.

    `speculate`: If `False`, remind the LLM to respond using in-context information only.

    `docs_matches`: Docs search matches returned by `HybridIR` (see `_search_docs_with_bypass`).
    """
    # # This causes Qwen3 to miss the user's last message. Maybe better to put the RAG results at another position.
    # #
    # # Format RAG results like a tool-call reply to the user's message.
    # # First, find the user's latest message in the linearized history.
    # for depth, message in enumerate(reversed(history)):
    #     if message["role"] == "user":
    #         break
    # else:  # no user message found (should not happen)
    #     depth = None
    #     message = None
    #
    # if message is not None:
    #     position = len(history) - depth
    #     for docs_result in reversed(docs_matches):  # reverse to keep original order, because we insert each item at the same position.
    #         # TODO: Should the RAG match notification show the query string, too?
    #         search_result_text = f"Knowledge-base match from '{docs_result['document_id']}':\n\n{docs_result['text'].strip()}\n-----"
    #         message_to_inject = chatutil.create_chat_message(llm_settings=settings,
    #                                                           role="tool",
    #                                                           text=search_result_text)
    #         history.insert(position, message_to_inject)

    # Insert RAG results at the start of the history, as system messages.
    # TODO: This causes a full KV cache rebuild. Could we place them later in the chat history?
    for docs_result in reversed(docs_matches):  # reverse to keep original order, because we insert each item at the same position.
        # TODO: Should the RAG match notification show the query string, too?
        search_result_text = f"[System information: Knowledge-base match from '{docs_result['document_id']}'.]\n\n{docs_result['text'].strip()}\n-----"
        message_to_inject = chatutil.create_chat_message(llm_settings=llm_settings,
                                                         role="system",
                                                         text=search_result_text)
        history.insert(1, message_to_inject)  # after system prompt / character card combo
    if docs_matches:
        n_matches_text = f"[System information: Knowledge base matched {len(docs_matches)} items.]"
        message_to_inject = chatutil.create_chat_message(llm_settings=llm_settings,
                                                         role="system",
                                                         text=n_matches_text)
        history.insert(1, message_to_inject)

    # When we're continuing AI's latest message, the history should appear as it was at the point where generation was interrupted.
    # Hence the always-on injects at the end of the history should be placed just before the AI's incomplete message, which will be continued.
    #
    # Otherwise, those injects should be placed at the end, since the AI will add a new message to the end.
    def inject(message):
        if continue_:
            history.insert(-1, message_to_inject)
        else:
            history.append(message_to_inject)

    # Always-on injects, e.g. current local datetime
    for thunk in injectors:
        message_to_inject = chatutil.create_chat_message(llm_settings=llm_settings,
                                                          role="system",
                                                          text=thunk())
        inject(message_to_inject)

    # If speculation is off, remind the LLM to use information from the context only.
    if not speculate:
        message_to_inject = chatutil.create_chat_message(llm_settings=llm_settings,
                                                          role="system",
                                                          text=chatutil.format_reminder_to_use_information_from_context_only())
        inject(message_to_inject)


def _perform_and_store_tool_calls(llm_settings: env,
                                  datastore: chattree.Forest,
                                  assistant_message: Dict,
                                  parent_node_id: str,
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

    The tool-call request context (harness-supplied, NOT model-supplied) is assembled and bound here for
    the dynamic extent of the dispatch — the request-context pattern (cf. Racket's `parameterize`, Flask's
    `g`). Entrypoints that need it read `dyn.tool_context`; see the field registry at
    `llmclient.make_dynvar(tool_context=...)`. It is computed from `parent_node_id`, so the walk sees this
    turn's user message and any prior tool results on the branch (e.g. a websearch whose result hosts may
    inform a later webfetch auto-allow).
    """
    head_node_id = parent_node_id
    if on_tools_start is not None:
        on_tools_start(assistant_message["tool_calls"])

    tool_context = env(webfetch_allowed_hosts=chatutil.compute_auto_allowed_hosts(
        datastore, head_node_id,
        trust_search_results=librarian_config.webfetch_trust_search_results))

    # Each tool call produces exactly one response. No-ops if the message contains no tool calls.
    with dyn.let(tool_context=tool_context):
        tool_response_records = llmclient.perform_tool_calls(llm_settings,
                                                             message=assistant_message,
                                                             on_call_start=on_call_lowlevel_start,
                                                             on_call_done=on_call_lowlevel_done)

    for tool_response_record in tool_response_records:
        def create_tool_payload() -> Dict:
            payload = chatutil.create_payload(llm_settings=llm_settings,
                                              message=tool_response_record.data)

            generation_metadata = {"status": tool_response_record.status}  # status is "success" or "error"
            if "tool_call_id" in tool_response_record:
                generation_metadata["toolcall_id"] = tool_response_record.tool_call_id  # storage key relocated/renamed to message.tool_call_id in §11 migration
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
            on_tools_done: Optional[Callable]) -> str:
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

    `docs_query`: Optional query string to search with in the document database.

                  If both this and `retriever` are supplied, `retriever` is queried, and the search results
                  are injected into the context before sending the context to the LLM.

                  If `None`, no search is performed.

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

    Returns the new HEAD node ID (i.e. the last chat node that was just added).
    """
    # Sanity check
    if continue_:
        head_node_payload = datastore.get_payload(head_node_id)
        if head_node_payload["message"]["role"] != "assistant":
            error_message = f"node '{head_node_id}' is not an AI message (role is '{head_node_payload['message']['role']}'), cannot continue it."
            logger.error(f"ai_turn: {error_message}")
            raise ValueError(error_message)

    # Search document database if requested
    if retriever is not None and docs_query is not None:
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

    continue_this_message = continue_  # we need to continue at most the first message in the agent loop
    while True:  # LLM agent loop - interleave LLM responses, tool calls and tool call results, until the LLM is done (no more tool calls).
        message_history = chatutil.linearize_chat(datastore=datastore,
                                                  node_id=head_node_id)

        # Prepare the final LLM prompt, by including the temporary injects (the document search results, too).
        _perform_injects(llm_settings=llm_settings,
                         history=message_history,
                         continue_=continue_this_message,
                         speculate=speculate,
                         docs_matches=docs_matches)

        if on_llm_start is not None:
            on_llm_start()
        out = llmclient.invoke(settings=llm_settings,
                               history=message_history,
                               on_prompt_ready=on_prompt_ready,
                               on_progress=on_llm_progress,  # this handles `action_stop` from `on_llm_progress`
                               tools_enabled=tools_enabled,
                               continue_=continue_this_message)
        # `out.data` is now the complete message object (in the format returned by `create_chat_message`)

        # Clean up the LLM's reply (heuristically). This version goes into the chat history.
        out.data["content"] = chatutil.scrub(persona=llm_settings.personas.get("assistant", None),
                                             text=out.data["content"],
                                             thoughts_mode="keep",
                                             markup=markup,
                                             add_persona=True)

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
                                                         on_tools_start=on_tools_start,
                                                         on_call_lowlevel_start=on_call_lowlevel_start,
                                                         on_call_lowlevel_done=on_call_lowlevel_done,
                                                         on_tool_done=on_tool_done,
                                                         on_tools_done=on_tools_done)
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
         complete results and the agent loop proceeds. No new RAG search (matches loop continuation).

    Returns the new HEAD node ID.
    """
    denied_payload = datastore.get_payload(tool_node_id)
    if denied_payload["message"]["role"] != "tool":
        raise ValueError(f"retry_tool_calls: node '{tool_node_id}' is not a tool-result node (role is '{denied_payload['message']['role']}').")
    denied_tool_call_id = denied_payload.get("generation_metadata", {}).get("toolcall_id")  # storage key relocated/renamed to message.tool_call_id in §11 migration

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
    head_node_id = _perform_and_store_tool_calls(llm_settings=llm_settings,
                                                 datastore=datastore,
                                                 assistant_message=synthetic_message,
                                                 parent_node_id=parent_node_id,
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
                   on_tools_done=on_tools_done)
