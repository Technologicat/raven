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
    user_message_node_id = datastore.create_node(payload={"message": chatutil.create_chat_message(llm_settings=llm_settings,
                                                                                                   role="user",
                                                                                                   text=user_message_text)},
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
                             k: int = 10) -> Values:
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

    If there are no matches, add a no-match message to the chat log (to be shown instead of the AI's reply).
    """
    docs_results = retriever.query(query,
                                   k=k,
                                   return_extra_info=False)

    # First line of defense (against hallucinations): docs on, no matches for given query, speculate off -> bypass LLM
    if not docs_results and not speculate:
        nomatch_text = "No matches in knowledge base. Please try another query."
        nomatch_message_node_id = datastore.create_node(payload={"message": chatutil.create_chat_message(llm_settings=llm_settings,
                                                                                                          role="assistant",
                                                                                                          text=nomatch_text)},
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
def ai_turn(llm_settings: env,
            datastore: chattree.Forest,
            retriever: hybridir.HybridIR,
            head_node_id: str,
            docs_query: Optional[str],  # if supplied, search the document database with this query and inject the results
            speculate: bool,  # if `False`, remind the LLM to respond using in-context information only
            markup: Optional[str],
            on_prompt_ready: Optional[Callable],
            on_llm_start: Optional[Callable],
            on_llm_progress: Optional[Callable],
            on_llm_done: Optional[Callable],
            on_docs_nomatch_done: Optional[Callable],
            on_tool_done: Optional[Callable]) -> str:
    """AI's turn: LLM generation interleaved with tool responses, until there are no tool calls in the LLM's latest reply.

    This continues the current branch with as many chat nodes as needed: one for each LLM response, and one for each tool call.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.

    `datastore`: The chat datastore.

    `retriever`: A `raven.librarian.hybridir.HybridIR` retriever connected to the document database.

    `head_node_id`: Current HEAD node of the chat. Used as the parent for the no-match message, if needed.

    `query`: Optional query string to search with in the document database.

             If supplied, `retriever` is queried, and the search results are injected into the context
             before sending the context to the LLM.

             If `None`, no search is performed.

    `speculate`: Used only if `query` is supplied.

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

    `on_prompt_ready`: 1-argument callable, with argument `history: List[Dict]`. Debug/info hook.
                       The return value is ignored.

                       Called after the LLM context has been completely prepared, before sending it to the LLM.

                       This is the modified history, after including document search results and temporary injects.
                       Each element of the list is a chat message in the format accepted by the LLM backend,
                       with "role" and "content" fields.

    `on_llm_start`: 0-argument callable. Called just before we call `llmclient.invoke` and the LLM starts
                    streaming a response.
                    The return value is ignored.

                    The LLM will start once at the beginning of the AI's turn, and then once after each set
                    of tool calls.

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

                   The argument is the node ID of this new chat node.

    `on_docs_nomatch_done`: 1-argument callable, with argument `node_id: str`.
                            The return value is ignored.

                            Called instead of `on_llm_start`/`on_llm_progress`/`on_llm_done` if the LLM was bypassed,
                            after the new chat node has been added to the chat datastore.

                            The argument is the node ID of this new chat node.

    `on_tool_done`: 1-argument callable, with argument `node_id: str`.
                    The return value is ignored.

                    Called *after* `on_llm_done`, once per tool call result, if there were tool calls, after the
                    tool's response chat node has been added to the chat datastore.

                    The argument is the node ID of this new chat node.

    Returns the new HEAD node ID (i.e. the last chat node that was just added).
    """
    # Search document database if requested
    if docs_query is not None:
        docs_result = _search_docs_with_bypass(llm_settings=llm_settings,
                                               datastore=datastore,
                                               retriever=retriever,
                                               head_node_id=head_node_id,
                                               speculate=speculate,
                                               query=docs_query)
        if docs_result["action"] is action_done:  # bypass triggered, we have a response chat node already
            head_node_id = docs_result["new_head_node_id"]
            if on_docs_nomatch_done is not None:
                on_docs_nomatch_done(head_node_id)
            return head_node_id
        docs_matches = docs_result["matches"]
    else:
        docs_matches = []

    while True:
        message_history = chatutil.linearize_chat(datastore=datastore,
                                                  node_id=head_node_id)

        # Prepare the final LLM prompt, by including the temporary injects (the document search results, too).
        _perform_injects(llm_settings=llm_settings,
                         history=message_history,
                         speculate=speculate,
                         docs_matches=docs_matches)
        if on_prompt_ready is not None:
            on_prompt_ready(message_history)

        if on_llm_start is not None:
            on_llm_start()
        out = llmclient.invoke(llm_settings, message_history, on_llm_progress)  # `out.data` is now the complete message object (in the format returned by `create_chat_message`)

        # Clean up the LLM's reply (heuristically). This version goes into the chat history.
        # TODO: Keep the thought blocks; strip them only when sending the history to the LLM.
        out.data["content"] = chatutil.scrub(llm_settings,
                                             out.data["content"],
                                             thoughts_mode="discard",
                                             markup=markup,
                                             add_ai_role_name=True)

        # Add the LLM's message to the chat.
        #
        # Note the token count of the message actually saved into the chat log may be different from `out.n_tokens`, e.g. if the AI is interrupted or when thoughts blocks are discarded.
        # However, to correctly compute the generation speed, we need to use the original count before any editing, since `out.dt` was measured for that.
        ai_message_node_id = datastore.create_node(payload={"message": out.data,
                                                            "generation_metadata": {"model": out.model,
                                                                                    "n_tokens": out.n_tokens,  # could count final tokens with `llmclient.token_count(settings, out.data["content"])`
                                                                                    "dt": out.dt}},
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
        tool_response_records = llmclient.perform_tool_calls(llm_settings, message=out.data)

        # When there are no more tool calls, the LLM is done replying.
        # Each tool call produces exactly one response, so we may as well check this from the number of responses.
        if not tool_response_records:
            break

        # Add the tool response messages (if any) to the chat.
        for tool_response_record in tool_response_records:
            payload = {"message": tool_response_record.data,
                       "generation_metadata": {"status": tool_response_record.status}}  # status is "success" or "error"
            if "toolcall_id" in tool_response_record:
                payload["generation_metadata"]["toolcall_id"] = tool_response_record.toolcall_id
            if "dt" in tool_response_record:
                payload["generation_metadata"]["dt"] = tool_response_record.dt
            tool_response_message_node_id = datastore.create_node(payload=payload,
                                                                  parent_id=head_node_id)
            head_node_id = tool_response_message_node_id

            if on_tool_done is not None:
                on_tool_done(head_node_id)

    return head_node_id
