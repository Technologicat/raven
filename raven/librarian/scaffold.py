"""Scaffolding for a multi-turn conversation with automatic RAG search and tool-calling."""

__all__ = ["user_turn",
           "ai_turn", "action_ack", "action_stop"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Callable, Dict, List, Optional

from unpythonic import sym, Values
from unpythonic.env import env

from . import chattree
from . import chatutil
from . import hybridir
from . import llmclient

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
    timestamp, unused_weekday, isodate, isotime = chatutil.make_timestamp()
    user_message_node_id = datastore.create_node(payload={"message": chatutil.create_chat_message(llm_settings=llm_settings,
                                                                                                   role="user",
                                                                                                   text=user_message_text),
                                                          "general_metadata": {"timestamp": timestamp,
                                                                               "datetime": f"{isodate} {isotime}",
                                                                               "persona": llm_settings.personas.get("user", None)}},
                                                 parent_id=head_node_id)
    return user_message_node_id


# --------------------------------------------------------------------------------
# AI's turn

def _search_docs_with_bypass(llm_settings: env,
                             datastore: chattree.Forest,
                             retriever: hybridir.HybridIR,
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
        timestamp, unused_weekday, isodate, isotime = chatutil.make_timestamp()
        nomatch_message_node_id = datastore.create_node(payload={"message": chatutil.create_chat_message(llm_settings=llm_settings,
                                                                                                         role="assistant",
                                                                                                         text=nomatch_text),
                                                                 "general_metadata": {"timestamp": timestamp,
                                                                                      "datetime": f"{isodate} {isotime}",
                                                                                      "persona": llm_settings.personas.get("assistant", None)}},
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
                     speculate: bool,
                     docs_matches: List[Dict]) -> None:
    """Perform the temporary injects to prepare for the AI's turn.

    These are not meant to be persistent, so we don't even add them to the datastore,
    but only insert them into the temporary linearized history that is fed to the LLM.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.
                    Contains (among other things) a mapping of roles to persona names.

    `history`: Linearized message history in the OpenAI format sent to the LLM.

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

    # Always-on injects, e.g. current local datetime
    for thunk in injectors:
        message_to_inject = chatutil.create_chat_message(llm_settings=llm_settings,
                                                          role="system",
                                                          text=thunk())
        history.append(message_to_inject)

    # If speculation is off, remind the LLM to use information from the context only.
    if not speculate:
        message_to_inject = chatutil.create_chat_message(llm_settings=llm_settings,
                                                          role="system",
                                                          text=chatutil.format_reminder_to_use_information_from_context_only())
        history.append(message_to_inject)


# TODO: `raven.librarian.scaffold.ai_turn`: implement continue mode, to continue an interrupted generation (or one that ran out of max output length)
# TODO: `tools_enabled` is a blunt hammer; maybe have also an optional tool name list for fine-grained control?
def ai_turn(llm_settings: env,
            datastore: chattree.Forest,
            retriever: Optional[hybridir.HybridIR],
            head_node_id: str,
            tools_enabled: bool,
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

    `on_llm_progress`: 2-argument callable, with arguments `(n_chunks: int, chunk_text: str)`.
                       Called while streaming the response from the LLM, typically once per generated token.

           `n_chunks: int`: How many chunks have been generated so far, for this invocation.
                            Useful for live UI updates.

           `chunk_text: str`: The text of the current chunk (typically a token).

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
    # Search document database if requested
    if retriever is not None and docs_query is not None:
        if on_docs_start is not None:
            on_docs_start()
        docs_result = _search_docs_with_bypass(llm_settings=llm_settings,
                                               datastore=datastore,
                                               retriever=retriever,
                                               head_node_id=head_node_id,
                                               speculate=speculate,
                                               query=docs_query,
                                               k=docs_num_results)
        if docs_result["action"] is action_done:  # no-match bypass triggered, we have a response chat node already
            head_node_id = docs_result["new_head_node_id"]
            if on_docs_done is not None:
                on_docs_done([])  # no matches
            if on_nomatch_done is not None:
                on_nomatch_done(head_node_id)
            return head_node_id
        else:
            docs_matches = docs_result["matches"]
            if on_docs_done is not None:
                on_docs_done(docs_matches)
    else:
        if retriever is None and docs_query is not None:
            logger.warning("ai_turn: A `docs_query` was supplied without a `retriever` to search with. Ignoring the query.")
        docs_matches = []

    while True:  # LLM agent loop - interleave LLM responses, tool calls and tool call results, until the LLM is done (no more tool calls).
        message_history = chatutil.linearize_chat(datastore=datastore,
                                                  node_id=head_node_id)

        # Prepare the final LLM prompt, by including the temporary injects (the document search results, too).
        _perform_injects(llm_settings=llm_settings,
                         history=message_history,
                         speculate=speculate,
                         docs_matches=docs_matches)

        if on_llm_start is not None:
            on_llm_start()
        out = llmclient.invoke(settings=llm_settings,
                               history=message_history,
                               on_prompt_ready=on_prompt_ready,
                               on_progress=on_llm_progress,  # this handles `action_stop` from `on_llm_progress`
                               tools_enabled=tools_enabled)
        # `out.data` is now the complete message object (in the format returned by `create_chat_message`)

        # Clean up the LLM's reply (heuristically). This version goes into the chat history.
        out.data["content"] = chatutil.scrub(persona=llm_settings.personas.get("assistant", None),
                                             text=out.data["content"],
                                             thoughts_mode="keep",
                                             markup=markup,
                                             add_persona=True)

        # Add the LLM's message to the chat.
        #
        # Note the token count of the message actually saved into the chat log may be different from `out.n_tokens`, e.g. if the AI is interrupted or when thoughts blocks are discarded.
        # However, to correctly compute the generation speed, we need to use the original count before any editing, since `out.dt` was measured for that.
        timestamp, unused_weekday, isodate, isotime = chatutil.make_timestamp()
        ai_message_node_id = datastore.create_node(payload={"message": out.data,
                                                            "generation_metadata": {"model": out.model,
                                                                                    "n_tokens": out.n_tokens,  # could count final tokens with `llmclient.token_count(settings, out.data["content"])`
                                                                                    "dt": out.dt},
                                                            "general_metadata": {"timestamp": timestamp,
                                                                                 "datetime": f"{isodate} {isotime}",
                                                                                 "persona": llm_settings.personas.get("assistant", None)}},
                                                   parent_id=head_node_id)
        ai_message_node_payload = datastore.get_payload(ai_message_node_id)
        if docs_query is not None:
            ai_message_node_payload["retrieval"] = {"query": docs_query,
                                                    "results": docs_matches}  # store RAG results in the chat node that was generated based on them, for later use (upcoming citation mechanism)
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
            if on_tools_start is not None:
                on_tools_start(out.data["tool_calls"])

            # Each tool call produces exactly one response.
            # This will no-op if the message contains no tool calls.
            tool_response_records = llmclient.perform_tool_calls(llm_settings,
                                                                 message=out.data,
                                                                 on_call_start=on_call_lowlevel_start,
                                                                 on_call_done=on_call_lowlevel_done)

            # Add the tool response messages to the chat.
            for tool_response_record in tool_response_records:
                generation_metadata = {"status": tool_response_record.status}  # status is "success" or "error"
                if "toolcall_id" in tool_response_record:
                    generation_metadata["toolcall_id"] = tool_response_record.toolcall_id
                if "function_name" in tool_response_record:
                    generation_metadata["function_name"] = tool_response_record.function_name
                if "dt" in tool_response_record:
                    generation_metadata["dt"] = tool_response_record.dt  # elapsed wall time, seconds

                timestamp, unused_weekday, isodate, isotime = chatutil.make_timestamp()
                payload = {"message": tool_response_record.data,
                           "generation_metadata": generation_metadata,
                           "general_metadata": {"timestamp": timestamp,
                                                "datetime": f"{isodate} {isotime}",
                                                "persona": llm_settings.personas.get("tool", None)}}
                tool_response_message_node_id = datastore.create_node(payload=payload,
                                                                      parent_id=head_node_id)
                head_node_id = tool_response_message_node_id

                if on_tool_done is not None:
                    on_tool_done(head_node_id)

            if have_tool_calls and on_tools_done is not None:
                on_tools_done()
        else:
            # When there are no more tool calls, the LLM is done replying.
            break

    return head_node_id
