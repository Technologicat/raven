"""Utilities for formatting LLM chat messages."""

__all__ = ["format_message_number",
           "format_persona",
           "format_message_heading",
           "format_chat_datetime_now", "format_chatlog_datetime_now",
           "format_reminder_to_focus_on_latest_input",
           "format_reminder_to_use_information_from_context_only",
           "create_chat_message",
           "create_initial_system_message",
           "linearize_chat",
           "upgrade_datastore",
           "remove_role_name_from_start_of_line",
           "scrub"]

import copy
import datetime
import re
from typing import Dict, List, Optional

from mcpyrate import colorizer

from unpythonic.env import env

from . import chattree

# --------------------------------------------------------------------------------
# Display formatting utilities (markdown, ansi)

def _yell_if_unsupported_markup(markup):
    if markup not in ("ansi", "markdown", None):
        raise ValueError(f"unknown markup kind '{markup}'; valid values: 'ansi' (*nix terminal), 'markdown', and the special value `None`.")

def format_message_number(message_number: Optional[int],
                          markup: Optional[str]) -> str:
    """Format the number of a chat message, e.g. '[#42]'.

    `message_number`: The number to format. If `None`, this returns the empty string, for convenience.
    `markup`: Which markup kind to use, or `None` for no markup. One of:
        "ansi": ANSI terminal color codes
        "markdown": Markdown markup
        `None` (the special value): no markup.

    Returns the formatted number.
    """
    _yell_if_unsupported_markup(markup)
    if message_number is not None:
        out = f"[#{message_number}]"
        if markup == "ansi":
            out = colorizer.colorize(out, colorizer.Style.DIM)
        elif markup == "markdown":
            out = f"*{out}*"
        return out
    return ""

def format_persona(llm_settings: env,
                   role: str,
                   markup: Optional[str]) -> str:
    """Format the persona name for `role`.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.
                    Contains (among other things) a mapping of roles to persona names.

    `role`: One of the roles supported by `raven.librarian.llmclient`.
            Typically, one of "assistant", "system", "tool", or "user".

    `markup`: Which markup kind to use, or `None` for no markup. One of:
        "ansi": ANSI terminal color codes
        "markdown": Markdown markup
        `None` (the special value): no markup.

    Returns the formatted persona name.
    """
    _yell_if_unsupported_markup(markup)
    persona = llm_settings.role_names.get(role, None)
    if persona is None:
        out = f"<<{role}>>"  # currently, this include "<<system>>" and "<<tool>>"
        if markup == "ansi":
            out = colorizer.colorize(out, colorizer.Style.DIM)
        elif markup == "markdown":
            out = f"`{out}`"  # use verbatim mode; otherwise looks like an HTML tag
        return out
    else:
        out = persona
        if markup == "ansi":
            out = colorizer.colorize(out, colorizer.Style.BRIGHT)
        elif markup == "markdown":
            out = f"**{out}**"
        return out

def format_message_heading(llm_settings: env,
                           message_number: Optional[int],
                           role: str,
                           markup: Optional[str]) -> str:
    """Format a chat message heading.

    Calls `format_message_number` and `format_persona`, which see.

    Returns the formatted message heading.

    For example, in:

        [#1] Aria: How can I help you today?

    the heading is the "[#1] Aria: " part, including the final space.
    """
    _yell_if_unsupported_markup(markup)
    markedup_number = format_message_number(message_number, markup)
    markedup_persona = format_persona(llm_settings, role, markup)
    if message_number is not None:
        return f"{markedup_number} {markedup_persona}: "
    else:
        return f"{markedup_persona}: "


# --------------------------------------------------------------------------------
# Stock message formatting utilities

_weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
def format_chat_datetime_now() -> str:
    """Return the text content of a dynamic system message containing the current date, weekday, and local time."""
    now = datetime.datetime.now()
    weekday = _weekdays[now.weekday()]
    date = now.date().isoformat()
    isotime = now.time().replace(microsecond=0).isoformat()
    return f"[System information: Today is {weekday}, {date} (in ISO format). The local time now is {isotime}.]"

def format_chatlog_datetime_now() -> str:
    """Return the current date, weekday, and local time in a human-readable format."""
    now = datetime.datetime.now()
    weekday = _weekdays[now.weekday()]
    date = now.date().isoformat()
    isotime = now.time().replace(microsecond=0).isoformat()
    return f"{weekday} {date} {isotime}"

def format_reminder_to_focus_on_latest_input() -> str:
    """Return the text content of a system message that reminds the LLM to focus on the user's latest input.

    Some models such as the distills of DeepSeek-R1 need this to enable multi-turn conversation to work correctly.
    """
    return "[System information: IMPORTANT: Reply to the user's most recent message. In a discussion, prefer writing your raw thoughts rather than a structured report.]"

def format_reminder_to_use_information_from_context_only() -> str:
    """Return the text content of a system message that reminds the LLM to use the information from the context only (not its internal static knowledge).

    As with all things LLM, this isn't completely reliable, but tends to increase the chances of the model NOT responding based on its static knowledge.
    This is useful when summarizing or extracting information from RAG search results.

    (The first line of defense is not giving control to the LLM when the search comes up empty. This reminder helps when the search returns results,
     but their content is irrelevant to the query.)
    """
    return "[System information: NOTE: Please answer based on the information provided in the context only.]"


# --------------------------------------------------------------------------------
# Chat message creation utilities

def create_chat_message(llm_settings: env,
                        role: str,
                        text: str,
                        add_role_name: bool = True,
                        tool_calls: List[str] = None) -> Dict:
    """Create a new chat message, compatible with the chat history format sent to the LLM.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.

    `role`: One of "user", "assistant", "system", "tool".

            Typically, "system" is used for the initial system prompt / character card combo,
            and "tool" is used for tool responses from tool-calls made by the LLM.

    `text`: The text content of the message.

    `add_role_name`: If `True`, we prepend the name of `role` (e.g. "AI: ..." when
                     `role='assistant'`) to the text content, if `settings.role_names`
                     has a name defined for that role.

                     Usually this is the right thing to do, but there are some occasions
                     (e.g. internally in `invoke`) where we need to skip this.

    `tool_calls`: Tool call requests; a list of JSON strings generated by the LLM.
                  These are pre-parsed from the raw text output by the LLM backend.

                  Mostly for use by `invoke`.

                  If `None`, an empty list is created. This is usually the right thing to do.

    Returns the new message: `{"role": ..., "content": ...}`.

    """
    if role not in ("user", "assistant", "system", "tool"):
        raise ValueError(f"Unknown role '{role}'; valid: one of 'user', 'assistant', 'system', 'tool'.")

    if add_role_name and llm_settings.role_names[role] is not None:
        content = f"{llm_settings.role_names[role]}: {text}"  # e.g. "User: ..."
    else:  # System and tool messages typically do not use a speaker tag in the text content.
        content = text

    data = {"role": role,
            "content": content,
            "tool_calls": tool_calls if tool_calls is not None else []}
    return data

def create_initial_system_message(llm_settings: env) -> Dict:
    """Create a chat message containing the system prompt and the AI's character card as specified in `llm_settings`.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.
    """
    if llm_settings.system_prompt and llm_settings.character_card:
        # The system prompt is stripped, so we need two linefeeds to have one blank line in between.
        text = f"{llm_settings.system_prompt}\n\n{llm_settings.character_card}\n\n-----"
    elif llm_settings.system_prompt:
        text = f"{llm_settings.system_prompt}\n\n-----"
    elif llm_settings.character_card:
        text = f"{llm_settings.character_card}\n\n-----"
    else:
        raise ValueError("create_initial_system_message: Need at least a system prompt or a character card.")
    return create_chat_message(llm_settings,
                               role="system",
                               text=text)


# --------------------------------------------------------------------------------
# Chat datastore utilities

def linearize_chat(datastore: chattree.Forest, node_id: str) -> List[Dict]:
    """In the chat `datastore`, walking up from `node_id` up to and including a root node, return a linearized representation of that branch.

    This collects the active revision of the data from each node, ignores everything except the chat message data
    (i.e. ignores any metadata added by the chat client, such as RAG retrieval attributions, AI token counts, etc.)
    and puts the messages into a list, in depth order (root node first).

    Note `node_id` doesn't need to be a leaf node; but it will be the last node of the linearized representation;
    children are not scanned.

    NOTE: The difference between this function and `chattree.Forest.linearize_up` is that this will
    automatically extract the "message" field (OpenAI-compatible chat message record) from each node,
    using the active revision of the payload, whereas that other function returns the node IDs.

    Hence, this is a convenience function for populating a linear chat history for chat clients that use
    the OpenAI format to communicate with the LLM server.
    """
    node_id_history = datastore.linearize_up(node_id)
    payload_history = [datastore.get_payload(node_id=node_id) for node_id in node_id_history]  # this auto-selects the active revision of the payload of each node
    message_history = [payload["message"] for payload in payload_history]
    return message_history

# v0.2.3+: data format change
def upgrade_datastore(datastore: chattree.Forest, system_prompt_node_id: str) -> None:
    """Upgrade a chat datastore's payloads to the latest format, modifying the datastore in-place.

    If the chat datastore's payloads are already in the latest format, no changes are made.

    `system_prompt_node_id`: The ID of the initial system prompt node (root node)
                             that starts a chat.

                             The reason we need this is that even in the old format (up to v0.2.2),
                             the system prompt node has no extra fluff saved on it, so we can use it
                             to get a list of system-level keys a chat node *should* have.

                             On other nodes, any keys that do NOT match those system-level keys
                             are assumed to be metadata added by the chat client. They are copied
                             to each existing data revision on the node (independent deepcopy for
                             each revision), and deleted from the top level of the node, so that
                             the top level contains only the system keys.

    NOTE: There are two upgrade functions for the chat datastore.

    The forest datastore itself also changed in v0.2.3 to allow for data revisioning.
    That part is automatically handled when an old datastore is loaded.
    See `chattree.PersistentForest._upgrade`.

    This function is meant to be explicitly called by a chat client. This upgrades
    the chat payload format.

    Up to v0.2.2, the chat message was stored in `node["data"]` directly, so that
    a node's "data" field content was an OpenAI-compatible chat message record::

        {"role": ..., "content": ..., "tool_calls": ...}

    In v0.2.3+, the `node["data"]` field is revisioned:

        {revision_id: payload,
         ...}

    Additionally, in the payload, the OpenAI-compatible chat message record
    now lives under the "message" key inside the `payload` part:

        {revision_id: {"message": {"role": ..., "content": ..., "tool_calls": ...},
                       "retrieval": {"query": ..., "results": ...},
                       ...},
         ...}

    thus allowing the chat client to add arbitrary other keys to the payload.
    These can be used to store metadata (for the chat client and/or for the user).

    For example, the "retrieval" key stores the RAG query and its retrieval results,
    which is useful for collecting attributions in the chat client (as well as for debugging).
    """
    # Get the names of system-level keys a chat node should have. Even in the old format (up to v0.2.2),
    # no extra keys are ever created on the system prompt node, so we can use this node to get an
    # up-to-date list (since `PersistentForest` auto-upgrades upon loading if the data format has changed).
    system_keys = set(datastore.nodes[system_prompt_node_id].keys())

    for node in datastore.nodes.values():
        payload_revisions = node["data"]  # {revision_id: payload, ...}

        # v0.2.3: Upgrade payload format
        for payload in payload_revisions.values():
            if "message" not in payload:  # old format?
                message = copy.copy(payload)
                payload.clear()
                payload["message"] = message

        # v0.2.3: Move any non-system keys on the node to under the revisioned data (one copy per revision; will become copies upon JSON saving anyway)
        existing_keys = list(node.keys())
        for key in existing_keys:
            if key not in system_keys:
                value = node.pop(key)
                for payload in payload_revisions.values():
                    payload[key] = copy.deepcopy(value)

def factory_reset_datastore(datastore: chattree.Forest, llm_settings: env) -> str:
    """Reset `datastore` to its "factory-default" state.

    **IMPORTANT**: This deletes all existing chat nodes in the datastore, and CANNOT BE UNDONE.

    The primary purpose of this function is to initialize the chat datastore when it hasn't been created yet.

    This creates a root node containing the system prompt (including the character card), and a node for the AI's initial greeting.

    Returns the unique ID of the initial greeting node, so you can start building chats on top of that.

    You can obtain the `settings` object by first calling `setup`.
    """
    datastore.purge()
    root_node_id = datastore.create_node(payload={"message": create_initial_system_message(llm_settings)},
                                         parent_id=None)
    new_chat_node_id = datastore.create_node(payload={"message": create_chat_message(llm_settings,
                                                                                     role="assistant",
                                                                                     text=llm_settings.greeting)},
                                             parent_id=root_node_id)
    return new_chat_node_id


# --------------------------------------------------------------------------------
# Chat message cleanup utilities

_complete_thought_block = re.compile(r"([<\[])(think(ing)?[>\]])(.*?)\1/\2\s*", flags=re.IGNORECASE | re.DOTALL)  # opened and closed correctly; thought contents -> group 4
_incomplete_thought_block = re.compile(r"([<\[])(think|thinking)([>\]])(?!.*?\1/\2\3)(.*)", flags=re.IGNORECASE | re.DOTALL)  # opened but not closed; thought contents -> group 4
_doubled_think_tag = re.compile(r"([<\[])(think|thinking)([>\]])\n([<\[])(think|thinking)([>\]])", flags=re.IGNORECASE | re.DOTALL)
_nan_thought_block = re.compile(r"([<\[])(think|thinking)([>\]])\nNaN\n([<\[])/(think|thinking)([>\]])\n", flags=re.IGNORECASE | re.DOTALL)
_thought_begin_tag = re.compile(r"([<\[])(think|thinking)([>\]])", flags=re.IGNORECASE | re.DOTALL)
_thought_end_tag = re.compile(r"([<\[])/(think|thinking)([>\]])", flags=re.IGNORECASE | re.DOTALL)

def remove_role_name_from_start_of_line(llm_settings: env,
                                        role: str,
                                        text: str) -> str:
    """Transform e.g. "User: blah blah" -> "blah blah", for every line in `text`.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.

    `role`: One of the roles supported by `raven.librarian.llmclient`.
            Typically, one of "assistant", "system", "tool", or "user".

    `text`: The text to process.

    Returns the processed text.
    """
    persona = llm_settings.role_names.get(role, None)
    if persona is None:
        return text
    _role_name_at_start_of_line = re.compile(f"^{persona}:\\s+", re.MULTILINE)
    text = re.sub(_role_name_at_start_of_line, r"", text)
    return text

def scrub(llm_settings: env,
          text: str,
          thoughts_mode: str,
          markup: Optional[str],
          add_ai_role_name: bool) -> str:
    """Heuristically clean up the text content of an LLM-generated message.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.

    `text`: The text content of the message to scrub.

    `thoughts_mode`: one of "discard", "markup". or "keep". What to do with thought blocks,
                     for thinking models.

    `markup`: used when `thoughts_mode='markup'`. Which markup kind to use, or `None` for no markup. One of:
        "ansi": ANSI terminal color codes.
        "markdown": Markdown markup, with HTML tags for colors.
        `None` (the special value): no markup. (Same effect as setting `thoughts_mode='keep'`.)

    `add_ai_role_name`: Whether to format the final text as "AI: blah blah" or just "blah blah".

    Returns the scrubbed text content.
    """
    _yell_if_unsupported_markup(markup)

    if thoughts_mode not in ("discard", "markup", "keep"):
        raise ValueError("scrub: Unknown thoughts_mode '{thoughts_mode}'; valid values: 'discard', 'markup', 'keep'.")

    # First remove any mentions of the AI persona's name at the start of any line in the text.
    # The model might generate this anywhere - before the thought block, or after the thought block.
    #
    # E.g. "AI: blah" -> "blah".
    #
    # This is important for consistency, since many models randomly sometimes add the persona name, and sometimes don't.
    #
    text = remove_role_name_from_start_of_line(llm_settings=llm_settings,
                                               role="assistant",
                                               text=text)

    # Fix the most common kinds of broken thought blocks (for thinking models)
    text = re.sub(_doubled_think_tag, r"\1\2\3", text)  # <think><think>...
    text = re.sub(_nan_thought_block, r"", text)  # <think>NaN</think>

    # September 2025 update: This seems to work with Qwen 3 2507, too.
    #
    # QwQ-32B: the model was trained not to emit the opening <think> tag, but to begin thinking right away. Still, it sometimes inserts that tag, but not always.
    #
    # Also sometimes, the model skips thinking and starts writing the final answer immediately (although it shouldn't do that). There's no way to detect this case
    # on the fly, because the opening <think> tag is *supposed to* be missing from the output when the model works correctly. The only way we can detect this is
    # when the output is complete; there won't be a closing </think> tag in it.
    #
    # At least in my tests, QwQ-32B always closes its thought blocks correctly, so if </think> is missing, it means that the model didn't generate a thought block.
    # If </think> is there, then it did.
    #
    # So we search for a closing </think>, and if that's there, but there is no opening <think>, we add the opening tag.
    #
    # What we have here works when there is at most one think block in the message - should be sufficient in practice.
    # TODO: Should we add the opening <think> already when streaming, or even add it to the prompt? How can we add a partial message with the API? Drawback: prevents the model from replying without thinking even in simple cases.
    #
    g = re.search(_thought_end_tag, text)
    if g is not None and re.search(_thought_begin_tag, text) is None:
        text = f"{g.group(1)}{g.group(2)}{g.group(3)}\n{text}"  # Prepend the message with a matching beginning think tag (for QwQ-32B, it's "<think>", but let's be general)

    # Now we should have clean thought blocks.
    # Treat them next.
    if thoughts_mode == "discard":  # for cases where we're not going to read them anyway (e.g. when we pipe the output to a script that only needs the final answer)
        text = re.sub(_complete_thought_block, r"", text)
        text = re.sub(_incomplete_thought_block, r"", text)
    elif thoughts_mode == "markup":  # For cases where we want to see the thought blocks. Colorize them. (TODO: Maybe make some kind of data structure instead.)
        # Colorize thought blocks (thinking models)
        #
        # TODO: This colorizes for text terminals for now; support also HTML colorization. Something like:
        # r"<hr><font color="#a0a0a0">\4</font><hr>"  -- simple variant
        # r"<hr><font color="#8080ff"><details name="thought"><summary><i>Thought</i></summary><font color="#a0a0a0">$4</font></details></font><hr>"  -- complete thought
        # r"<hr><font color="#8080ff"><i>Thinking...</i><br><font color="#a0a0a0">$4<br></font><i>Thinking...</i></font><hr>"  -- incomplete thought
        #
        if markup == "ansi":
            blue_thought = colorizer.colorize("Thought", colorizer.Fore.BLUE)
            def _colorize(match_obj):
                s = match_obj.group(4)
                s = colorizer.colorize(s, colorizer.Style.DIM)
                return f"⊳⊳⊳{blue_thought}⊳⊳⊳\n{s}⊲⊲⊲{blue_thought}⊲⊲⊲\n"
        elif markup == "markdown":
            blue_thought = '<font color="#808080ff">Thought</font>'
            def _colorize(match_obj):
                s = match_obj.group(4)
                s = f'<font color="#a0a0a0">{s}</font>'
                return f"⊳⊳⊳{blue_thought}⊳⊳⊳\n-----\n{s}\n-----\n⊲⊲⊲{blue_thought}⊲⊲⊲\n"

        if markup is not None:  # one of the supported markup types was picked?
            text = re.sub(_complete_thought_block, _colorize, text)
            text = re.sub(_incomplete_thought_block, _colorize, text)
    # else do nothing, i.e. keep thought blocks as-is.

    # Remove whitespace surrounding the whole text content. (Do this last.)
    text = text.strip()

    # Postprocess:
    #
    # If we should add the AI persona's name, now do so at the beginning of the text content, for consistency.
    # It will appear before the thought block, if any, because this is the easiest to do. :)
    #
    # This is also good for detecting the persona name later. The OpenAI-compatible chat log format expects the persona name
    # at the start of the first line of each chat message ("User: Blah..." or "AI: Blah..."). Hence we should keep it
    # *only* there, to avoid duplicating information in the chat datastore. (This works as long as characters have unique names.)
    #
    # The main case where we DON'T need to do this is when piping the output to a script, in which case the chat framework
    # is superfluous. In that use case, we really use the LLM as an instruct-tuned model, i.e. a natural language processor
    # that is programmed via free-form instructions in English. Raven's PDF importer does this a lot.
    if add_ai_role_name:
        text = f"{llm_settings.char}: {text}"

    return text
