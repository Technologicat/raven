"""Utilities for formatting LLM chat messages."""

__all__ = ["format_message_number",
           "format_persona",
           "format_message_heading",
           "format_chat_datetime_now",
           "format_reminder_to_focus_on_latest_input",
           "format_reminder_to_use_information_from_context_only",
           "scrub"]

import datetime
import re
from typing import Optional

from mcpyrate import colorizer

from unpythonic.env import env

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
# stock message formatting utilities

_weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
def format_chat_datetime_now() -> str:
    """Return the text content of a dynamic system message containing the current date, weekday, and local time."""
    now = datetime.datetime.now()
    weekday = _weekdays[now.weekday()]
    date = now.date().isoformat()
    isotime = now.time().replace(microsecond=0).isoformat()
    return f"[System information: Today is {weekday}, {date} (in ISO format). The local time now is {isotime}.]"

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
# cleanup utilities

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
          add_ai_role_name: bool) -> str:
    """Heuristically clean up the text content of an LLM-generated message.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.

    `text`: The text content of the message to scrub.

    `thoughts_mode`: one of "discard", "colorize". or "keep". What to do with thought blocks,
                     for thinking models.

    `add_ai_role_name`: Whether to format the final text as "AI: blah blah" or just "blah blah".

    Returns the scrubbed text content.
    """

    # First remove any mentions of the AI persona's name at the start of any line in the text.
    # The model might generate this anywhere - before the thought block, or after the thought block.
    #
    # E.g. "AI: blah" -> "blah".
    #
    # This is important for consistency, since many models randomly sometimes add the persona name, and sometimes don't.
    #
    text = remove_role_name_from_start_of_line(llm_settings=llm_settings, role="assistant", text=text)

    # Fix the most common kinds of broken thought blocks (for thinking models)
    text = re.sub(_doubled_think_tag, r"\1\2\3", text)  # <think><think>...
    text = re.sub(_nan_thought_block, r"", text)  # <think>NaN</think>

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
    elif thoughts_mode == "colorize":  # For cases where we want to see the thought blocks. Colorize them. (TODO: Maybe make some kind of data structure instead.)
        # Colorize thought blocks (thinking models)
        #
        # TODO: This colorizes for text terminals for now; support also HTML colorization. Something like:
        # r"<hr><font color="#a0a0a0">\4</font><hr>"  -- simple variant
        # r"<hr><font color="#8080ff"><details name="thought"><summary><i>Thought</i></summary><font color="#a0a0a0">$4</font></details></font><hr>"  -- complete thought
        # r"<hr><font color="#8080ff"><i>Thinking...</i><br><font color="#a0a0a0">$4<br></font><i>Thinking...</i></font><hr>"  -- incomplete thought
        #
        blue_thought = colorizer.colorize("Thought", colorizer.Fore.BLUE)
        def _colorize(match_obj):
            s = match_obj.group(4)
            s = colorizer.colorize(s, colorizer.Style.DIM)
            return f"⊳⊳⊳{blue_thought}⊳⊳⊳\n{s}⊲⊲⊲{blue_thought}⊲⊲⊲\n"
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
