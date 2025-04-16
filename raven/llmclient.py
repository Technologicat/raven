__all__ = ["list_models", "setup",
           "new_chat", "add_chat_message", "inject_chat_message",
           "format_chat_datetime_now", "format_reminder_to_focus_on_latest_input",
           "invoke"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse
import atexit
from copy import copy, deepcopy
import datetime
import io
import json
import os
import pathlib
import re
import requests
import sys
from textwrap import dedent
from typing import Dict, List, Tuple, Optional

import sseclient  # pip install sseclient-py

from mcpyrate import colorizer

from unpythonic import timer
from unpythonic.env import env

from . import config

# --------------------------------------------------------------------------------
# Setup for minimal chat client (for testing/debugging)

_config_dir = "~/.config/raven"  # for chat history

output_line_width = 160  # for text wrapping in live update

# --------------------------------------------------------------------------------
# LLM API setup

# !!! oobabooga/text-generation-webui users: If you need to see the final prompt in instruct or chat mode, start your server in `--verbose` mode.

headers = {
    "Content-Type": "application/json"
}

def list_models(backend_url):
    """List all models available at `backend_url`."""
    response = requests.get(f"{config.llm_backend_url}/v1/internal/model/list",
                            headers=headers,
                            verify=False)
    payload = response.json()
    return [model_name for model_name in sorted(payload["model_names"], key=lambda s: s.lower())]

def setup(backend_url: str) -> env:
    """Connect to LLM at `backend_url`, and return an `env` object (a fancy namespace) populated with the following fields:

        `user`: Name of user's character.
        `char`: Name of AI assistant.
        `model`: Name of model running at `backend_url`, queried automatically from the backend.
        `system_prompt`: Generic system prompt for the LLM (this is the LLaMA 3 preset from SillyTavern), to make it follow the character card.
        `character_card`: Character card that configures the AI assistant to improve the model's performance.
        `greeting`: The AI assistant's first message, used later for initializing the chat history.
        `prompts`: Prompts for the LLM, BibTeX field processing functions, and any literal info to fill in the output BibTeX. See the main program for details.
        `request_data`: Generation settings for the LLM backend.
        `role_names`: A `dict` with keys "user", "assistant", "system", used for constructing chat messages (see `add_chat_message`).
    """
    user = "User"
    char = "Aria"

    # Fill the model name from the backend, for the character card.
    #
    # https://github.com/oobabooga/text-generation-webui/discussions/1713
    # https://stackoverflow.com/questions/78690284/oobabooga-textgen-web-ui-how-to-get-authorization-to-view-model-list-from-port-5
    # https://github.com/oobabooga/text-generation-webui/blob/main/extensions/openai/script.py
    response = requests.get(f"{backend_url}/v1/internal/model/info",
                            headers=headers,
                            verify=False)
    payload = response.json()
    model = payload["model_name"]

    # ----------------------------------------
    # System prompt and character card

    # For recent models as of April 2025, e.g. QwQ-32B, the system prompt itself can be blank. The character card is enough.
    # Older models may need a general briefing first.
    #
    # system_prompt = dedent(f"""You are an expert actor that can fully immerse yourself into any role given. You do not break character for any reason, even if someone tries addressing you as an AI or language model. Currently your role is {char}, which is described in detail below. As {char}, continue the exchange with {user}.""")  # "Actor" preset from SillyTavern.
    system_prompt = ""

    # This is a minimal setup, partially copied from my personal AI assistant, meant to be run against locally hosted models.
    # This gives better performance (accuracy, instruction following) vs. querying the LLM directly without any system prompt.
    #
    # TODO: "If unsure" and similar tricks tend to not work for 8B models. At LLaMA 3.1 70B and better, it should work, but running that requires at least 2x24GB VRAM.
    # TODO: Query the context size from the backend if possible. No, doesn't seem to be possible. https://github.com/oobabooga/text-generation-webui/discussions/5317
    #
    character_card = dedent(f"""Note that {user} cannot see this introductory text; it is only used internally, to initialize the LLM (large language model).

    **About {char}**

    You are {char} (she/her), an AI assistant. You are highly intelligent. You have been trained to answer questions, provide recommendations, and help with decision making.

    **About the system**

    The LLM version is "{model}".

    The knowledge cutoff date of the model is not specified, but is most likely within the year 2024. The knowledge cutoff date applies only to your internal knowledge. Any information provided in the context may be newer.

    You are running on a private, local system.

    **Interaction tips**

    - Be polite, but go straight to the point.
    - Provide honest answers.
    - If you are unsure or cannot verify a fact, admit it.
    - If you think what the user says is incorrect, say so, and provide justification.
    - Cite sources when possible. IMPORTANT: Cite only sources listed in the context.
    - When given a complex problem, take a deep breath, and think step by step. Report your train of thought.
    - When given web search results, and those results are relevant to the query, use the provided results, and report only the facts as according to the provided results. Ignore any search results that do not make sense. The user cannot directly see your search results.
    - Be accurate, but diverse. Avoid repetition.
    - Use the metric unit system, with meters, kilograms, and celsius.
    - Use Markdown for formatting when helpful.
    - Believe in your abilities and strive for excellence. Take pride in your work and give it your best. Your hard work will yield remarkable results.

    **Known limitations**

    - You are NOT automatically updated with new data.
    - You have limited long-term memory within each chat session.
    - The length of your context window is 65536 tokens.

    **Data sources**

    - The system accesses external data beyond its built-in knowledge through:
      - Additional context that is provided by the software this LLM is running in.
    """)

    greeting = "How can I help you today?"

    # Generation settings for the LLM backend.
    request_data = {
        "mode": "instruct",  # instruct mode: when invoking the LLM, send it instructions (system prompt and character card), followed by a chat transcript to continue.
        "max_tokens": 3200,  # 800 is usually good, but thinking models may need (much) more. For them, 1600 or 3200 are good.
        # Correct sampler order is min_p first, then temperature (and this is also the default).
        #
        # T = 1: Use the predicted logits as-is (as of early 2025, a good default; older models may need T = 0.7).
        # T = 0: Greedy decoding, i.e. always pick the most likely token. Prone to getting stuck in a loop. For fact extraction (for some models).
        # T > 1: Skew logits to emphasize rare continuations ("creative mode").
        # 0 < T < 1: Skew logits to emphasize common continuations.
        "temperature": 1,
        # min_p a.k.a. "you must be this tall". Good default sampler, with 0.02 a good value for many models.
        # This is a tail-cutter. The value is the minimum probability a token must have to admit sampling that token,
        # as a fraction of the probability of the most likely option (locally, at each position).
        #
        # Once min_p cuts the tail, then the remaining distribution is given to the temperature mechanism for skewing.
        # Then a token is sampled, weighted by the probabilities represented by the logits (after skewing).
        "min_p": 0.02,
        "seed": -1,  # 558614238,  # -1 = random; unused if T = 0
        "stream": True,  # When the LLM is generating text, send each token to the client as soon as it is available. For live-updating the UI.
        "messages": [],  # Chat transcript, including system messages. Populated later by `invoke`.
        "name1": user,  # Name of user's persona in the chat.
        "name2": char,  # Name of AI's persona in the chat.
    }

    # For easily populating chat messages.
    role_names = {"user": user,
                  "assistant": char,
                  "system": None}

    settings = env(user=user, char=char, model=model,
                   system_prompt=system_prompt,
                   character_card=character_card,
                   greeting=greeting,
                   backend_url=backend_url,
                   request_data=request_data,
                   role_names=role_names)
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

# --------------------------------------------------------------------------------
# Utilities

def new_chat(settings: env) -> List[Dict[str, str]]:
    """Initialize a new chat.

    Returns the chat history object for the new chat.

    The new history begins with the system prompt, followed by the character card, and then the AI assistant's greeting message.

    You can add more messages to the chat by calling `add_chat_message`.

    You can obtain the `settings` object by first calling `setup`.
    """
    history = []
    history = add_chat_message(settings, history, role="system", message=f"{settings.system_prompt}\n\n{settings.character_card}")
    history = add_chat_message(settings, history, role="assistant", message=settings.greeting)
    return history

def add_chat_message(settings: env, history: List[Dict[str, str]], role: str, message: str) -> List[Dict[str, str]]:
    """Append a new message to a chat history, functionally (without modifying the original history instance).

    Returns the updated chat history object.

    `role`: one of "user", "assistant", "system"
    """
    return inject_chat_message(0, settings, history, role, message)

def inject_chat_message(depth: int, settings: env, history: List[Dict[str, str]], role: str, message: str) -> List[Dict[str, str]]:
    """Add a new message to a chat history at arbitrary position, functionally (without modifying the original history instance).

    Returns the updated chat history object.

    `depth`: int, nonnegative. Inject position, counting from current end of `history` (0 = append at end).
             If `depth = -1` or `depth > len(history)`, insert at the beginning.
    `role`: one of "user", "assistant", "system"
    """
    if role not in ("user", "assistant", "system"):
        raise ValueError(f"Unknown role '{role}'; valid: one of 'user', 'assistant', 'system'.")
    if depth == -1:  # insert at beginning?
        idx = 0
    else:
        idx = max(0, len(history) - depth)
    if settings.role_names[role] is not None:
        content = f"{settings.role_names[role]}: {message}"  # e.g. "User: ..."
    else:  # System messages typically do not have a speaker tag for the line.
        content = message
    new_history = copy(history)
    new_history.insert(idx, {"role": role, "content": content})
    return new_history

_weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
def format_chat_datetime_now() -> str:
    """Return a dynamic system message containing the current date, weekday, and local time.

    This message can be injected to the chat history to show it to the LLM.
    """
    now = datetime.datetime.now()
    weekday = _weekdays[now.weekday()]
    date = now.date().isoformat()
    isotime = now.time().replace(microsecond=0).isoformat()
    return f"[System information: Today is {weekday}, {date} (in ISO format). The local time now is {isotime}.]"

def format_reminder_to_focus_on_latest_input() -> str:
    """Return a system message that reminds the LLM to focus on the user's latest input.

    Some models such as the distills of DeepSeek-R1 need this to enable multi-turn conversation to work correctly.

    This message can be injected to the chat history to show it to the LLM.
    """
    return "[System information: IMPORTANT: Reply to the user's most recent message. In a discussion, prefer writing your raw thoughts rather than a structured report.]"


_complete_thought_block = re.compile(r"([<\[])(think(ing)?[>\]])(.*?)\1/\2\s*", flags=re.IGNORECASE | re.DOTALL)  # opened and closed correctly; thought contents -> group 4
_incomplete_thought_block = re.compile(r"([<\[])(think|thinking)([>\]])(?!.*?\1/\2\3)(.*)", flags=re.IGNORECASE | re.DOTALL)  # opened but not closed; thought contents -> group 4
_doubled_think_tag = re.compile(r"([<\[])(think|thinking)([>\]])\n([<\[])(think|thinking)([>\]])", flags=re.IGNORECASE | re.DOTALL)
_nan_thought_block = re.compile(r"([<\[])(think|thinking)([>\]])\nNaN\n([<\[])/(think|thinking)([>\]])\n", flags=re.IGNORECASE | re.DOTALL)
_thought_begin_tag = re.compile(r"([<\[])(think|thinking)([>\]])", flags=re.IGNORECASE | re.DOTALL)
_thought_end_tag = re.compile(r"([<\[])/(think|thinking)([>\]])", flags=re.IGNORECASE | re.DOTALL)

def remove_persona_from_start_of_line(settings: env, role: str, content: str) -> str:
    persona = settings.role_names.get(role, None)
    if persona is None:
        return content
    _persona_name_at_start_of_line = re.compile(f"^{persona}:\\s+", re.MULTILINE)
    content = re.sub(_persona_name_at_start_of_line, r"", content)
    return content

def clean(settings: env, message: str, thoughts_mode: str, add_ai_persona_name: bool) -> str:
    """Heuristically clean up an LLM-generated message."""

    # First remove any mentions of the AI persona's name at the start of any line in the message.
    # The model might generate this anywhere - before the thought block, or after the thought block.
    #
    # E.g. "AI: blah" -> "blah".
    #
    # This is important for consistency, since many models randomly sometimes add the persona name, and sometimes don't.
    #
    message = remove_persona_from_start_of_line(settings=settings, role="assistant", content=message)

    # Fix the most common kinds of broken thought blocks (for thinking models)
    message = re.sub(_doubled_think_tag, r"\1\2\3", message)  # <think><think>...
    message = re.sub(_nan_thought_block, r"", message)  # <think>NaN</think>

    # QwQ-32B: the model was trained not to emit the opening <think> tag, but to begin thinking right away. Still, it sometimes inserts that tag, but not always.
    #
    # Also sometimes, the model skips thinking and starts writing the final answer immediately (although it shouldn't do that). There's no way to detect this case
    # on the fly, because the opening <think> tag is *supposed to* be missing from the output when the model works correctly. The only way we can detect this is
    # when #he output is complete; there won't be a closing </think> tag in it.
    #
    # At least in my tests, QwQ-32B always closes its thought blocks correctly, so if </think> is missing, it means that the model didn't generate a thought block.
    # If </think> is there, then it did.
    #
    # So we search for a closing </think>, and if that's there, but there is no opening <think>, we add the opening tag.
    #
    # What we have here works when there is at most one think block in the message - should be sufficient in practice.
    # TODO: Should we add the opening <think> already when streaming, or even add it to the prompt? How can we add a partial message with the API? Drawback: prevents the model from replying without thinking even in simple cases.
    #
    g = re.search(_thought_end_tag, message)
    if g is not None and re.search(_thought_begin_tag, message) is None:
        message = f"{g.group(1)}{g.group(2)}{g.group(3)}\n{message}"  # Prepend the message with a matching beginning think tag (for QwQ-32B, it's "<think>", but let's be general)

    # Now we should have clean thought blocks.
    # Treat them next.
    if thoughts_mode == "discard":  # for cases where we're not going to read them anyway (e.g. when we pipe the output to a script that only needs the final answer)
        message = re.sub(_complete_thought_block, r"", message)
        message = re.sub(_incomplete_thought_block, r"", message)
    elif thoughts_mode == "colorize":  # For cases where we want to see the thought blocks. Colorize them. (TODO: Maybe make some kind of data structure instead.)
        # Colorize thought blocks (thinking models)
        #
        # TODO: this is for text terminals now; support also HTML colorization. Something like:
        # r"<hr><font color="#a0a0a0">\4</font><hr>"  -- simple variant
        # r"<hr><font color="#8080ff"><details name="thought"><summary><i>Thought</i></summary><font color="#a0a0a0">$4</font></details></font><hr>"  -- complete thought
        # r"<hr><font color="#8080ff"><i>Thinking...</i><br><font color="#a0a0a0">$4<br></font><i>Thinking...</i></font><hr>"  -- incomplete thought
        #
        blue_thought = colorize("Thought", colorizer.Fore.BLUE)
        def _colorize(match_obj):
            s = match_obj.group(4)
            s = colorize(s, colorizer.Style.DIM)
            return f"⊳⊳⊳{blue_thought}⊳⊳⊳\n{s}⊲⊲⊲{blue_thought}⊲⊲⊲\n"
        message = re.sub(_complete_thought_block, _colorize, message)
        message = re.sub(_incomplete_thought_block, _colorize, message)
    # else do nothing, i.e. keep thought blocks as-is.

    # Remove whitespace surrounding the whole message. (Do this last.)
    message = message.strip()

    # Postprocess:
    #
    # If we should add the AI persona's name, now do so at the beginning of the message, for consistency.
    # It will appear before the thought block, if any, because this is the easiest to do. :)
    #
    # Cases where we DON'T need to do this:
    #   - Chat app, which usually has a separate UI element for the persona name, aside from the actual chat text content UI element.
    #   - Piping output to a script, in which case the chat framework is superfluous. In that use case, we really use the LLM
    #     as an instruct-tuned model, i.e. a natural language processor that is programmed via free-form instructions in English.
    #     Raven's PDF importer does this a lot.
    if add_ai_persona_name:
        message = f"{settings.char}: {message}"

    return message


# --------------------------------------------------------------------------------
# The most important function - call LLM, parse result

def invoke(settings: env, history: List[Dict[str, str]], progress_callback=None) -> Tuple[str, int]:
    """Invoke the LLM with the given chat history.

    This is typically done after adding the user's message to the chat history, to ask the LLM to generate a reply.

    `progress_callback`: callable, optional.
        If provided, this is called for each chunk. It is expected to take two arguments; the signature is
        `progress_callback(n_chunks, chunk_text)`. Here:
            `n_chunks`: int, how many chunks have been generated so far (for this message).
                        This is useful if you want to e.g. print a progress symbol to the terminal every ten chunks.
            `chunk_text` str, the actual text of the current chunk.
        Typically, at least with `oobabooga/text-generation-webui`, one chunk = one token.

    Returns an `unpythonic.env.env` WITHOUT adding the LLM's reply to `history`.

    The returned `env` has the following attributes:

        `data`: str, The new message generated by the LLM.
                     If it begins with the assistant character's name (e.g. "AI: ..."), this is automatically stripped.
        `n_tokens`: int, Number of tokens emitted by the LLM.
        `dt`: float, Wall time elapsed for generating this message, in seconds.

    If you want to add `data` to `history`, use `history = add_chat_message(settings, history, role='assistant', message=data)`.
    """
    data = deepcopy(settings.request_data)
    data["messages"] = history
    stream_response = requests.post(f"{settings.backend_url}/v1/chat/completions", headers=headers, json=data, verify=False, stream=True)
    client = sseclient.SSEClient(stream_response)

    llm_output = io.StringIO()
    n_chunks = 0
    try:
        with timer() as tim:
            for event in client.events():
                payload = json.loads(event.data)
                chunk = payload['choices'][0]['delta']['content']
                n_chunks += 1
                # TODO: we should implement some stopping strings, just to be sure.
                llm_output.write(chunk)
                if progress_callback is not None:
                    progress_callback(n_chunks, chunk)
    except requests.exceptions.ChunkedEncodingError as exc:
        logger.error(f"Connection lost. Please check if your LLM backend is still alive (was at {settings.backend_url}). Original error message follows.")
        logger.error(f"{type(exc)}: {exc}")
        raise

    llm_output = llm_output.getvalue()
    n_tokens = n_chunks - 2  # No idea why, but that's how it empirically is (see ooba server terminal output).  # TODO: Investigate later.

    return env(data=llm_output,
               n_tokens=n_tokens,
               dt=tim.dt)


# --------------------------------------------------------------------------------
# Minimal chat client.
#
# Also a usage example for the API of this module.

# TODO: fix `mcpyrate.colorizer` and release new version, this belongs there. This version works also in `input` when `readline` is enabled.
def colorize(text, *colors):
    """Colorize string `text` for terminal display.

    Always reset style and color at the start of `text`, as well as after it.

    Returns `text`, augmented with color and style commands for terminals.

    For available `colors`, see `Fore`, `Back` and `Style`.

    Usage::

        print(colorize("I'm new here", Fore.GREEN))
        print(colorize("I'm bold and bluetiful", Style.BRIGHT, Fore.BLUE))

    Each entry can also be a tuple (arbitrarily nested), which is useful
    for defining compound styles::

        BRIGHT_BLUE = (Style.BRIGHT, Fore.BLUE)
        ...
        print(colorize("I'm bold and bluetiful, too", BRIGHT_BLUE))

    **CAUTION**: Does not nest. If you want to set a color and style
    until further notice, use `setcolor` instead.
    """
    # To allow the `readline` module to calculate the visual length of colored text correctly, we can wrap the ANSI escape sequences
    # in ASCII escape sequences that temporarily disable `readline`'s length counting:
    #     \x01 is ASCII "Start of Heading" (SOH) character.
    #     \x02 is ASCII "Start of Text" (STX) character.
    #
    #     https://www.reddit.com/r/commandline/comments/1d4t3xz/gnu_readline_issues_setting_up_a_python_prompt/
    #
    # Also, to use `readline` correctly, the prompt must be supplied to your call to `input` so that `readline` can perform
    # the prompt printing, because it's not possible to get it back from the terminal (were we to print it from our own code).
    # When you do that, browsing history entries no longer clears the prompt, and the prompt is protected from backspacing.
    #     https://stackoverflow.com/questions/75987688/how-can-readline-be-told-not-to-erase-externally-supplied-prompt
    return "\x01{}\x02{}\x01{}\x02".format(colorizer.setcolor(colors),
                                           text,
                                           colorizer.setcolor())

def chat(backend_url):
    """Minimal LLM chat client, for testing/debugging."""
    import readline  # noqa: F401, side effect: enable GNU readline in input()
    # import rlcompleter  # noqa: F401, side effects: readline tab completion

    config_dir = pathlib.Path(_config_dir).expanduser().resolve()
    config_file_location = config_dir / "llmclient_history"
    print(colorize(f"GNU readline available. Saving user inputs to {str(config_file_location)}.", colorizer.Style.BRIGHT))
    print(colorize("Use up/down arrows to browse previous inputs. Enter to send. ", colorizer.Style.BRIGHT))
    print()
    try:
        readline.read_history_file(config_file_location)
    except FileNotFoundError:
        pass

    def save_history():
        config_dir.mkdir(parents=True, exist_ok=True)
        readline.set_history_length(1000)
        readline.write_history_file(config_file_location)
    atexit.register(save_history)

    try:
        try:
            print(colorize(f"Connecting to LLM backend at {backend_url}", colorizer.Style.BRIGHT))
            list_models(backend_url)  # just do something, to try to connect
            print(colorize("    Connected!", colorizer.Style.BRIGHT, colorizer.Fore.GREEN))
            print()
            settings = setup(backend_url=backend_url)  # If this succeeds, then we know the backend is alive.
        except Exception as exc:
            print(colorize("    Failed!", colorizer.Style.BRIGHT, colorizer.Fore.YELLOW))
            msg = f"Failed to connect to LLM backend at {backend_url}, reason {type(exc)}: {exc}"
            logger.error(msg)
            raise RuntimeError(msg)

        def chat_show_model_info():
            print(f"    {colorize('Current model', colorizer.Style.BRIGHT)}: {settings.model}")
            print(f"    {colorize('Character', colorizer.Style.BRIGHT)}: {settings.char} [defined in this client]")
            print()
        chat_show_model_info()

        print(colorize("Starting chat.", colorizer.Style.BRIGHT))
        print()
        def chat_show_help():
            print(colorize("=" * 80, colorizer.Style.BRIGHT))
            print("    llmclient.py - Minimal LLM client for testing/debugging.")
            print()
            print("    Special commands (tab-completion available):")
            print("        !clear   - Start new chat (clear history)")
            print("        !history - Print a cleaned-up transcript of the chat history")
            print("        !model   - Show which model is in use")
            print("        !models  - List all models available at connected backend")
            print("        !help    - Show this message again")
            print()
            print("    Press Ctrl+D to exit chat.")
            print(colorize("=" * 80, colorizer.Style.BRIGHT))
            print()
        chat_show_help()

        # https://docs.python.org/3/library/readline.html#readline-completion
        candidates = ["clear", "history", "model", "models", "help"]
        def completer(text, state):  # completer for special commands
            if not text.startswith("!"):  # Not a command -> no completions.
                return None
            text = text[1:]

            # Just a "!"? -> List all commands.
            if not text:
                if state < len(candidates):
                    return f"!{candidates[state]}"
                return None

            # Score the possible completions for the given prefix `text`.
            # https://stackoverflow.com/questions/6718196/determine-the-common-prefix-of-multiple-strings
            scores = [len(os.path.commonprefix([text, candidate])) for candidate in candidates]
            max_score = max(scores)
            if max_score == 0:  # no match
                return None

            # Accept only completions that scored best (i.e. those that match the longest matched prefix).
            # completions = [(c, s) for c, s in zip(candidates, scores) if s == max_score]
            completions = [c for c, s in zip(candidates, scores) if s == max_score]
            if state >= len(completions):  # No more completions for given prefix `text`.
                return None
            # completions = list(sorted(completions, key=lambda item: -item[1]))  # Sort by score, descending.
            # completion, score = completions[state]
            completion = completions[state]
            return f"!{completion}"
        readline.set_completer(completer)
        readline.set_completer_delims(" ")
        readline.parse_and_bind("tab: complete")

        def chat_show_list_of_models():
            available_models = list_models(backend_url)
            print(colorize("    Available models:", colorizer.Style.BRIGHT))
            for model_name in available_models:
                print(f"        {model_name}")
            print()

        # TODO: we don't need the `color=False` option now that we fixed `colorize` to work with `readline`/`input`.
        def format_message_number(message_number: Optional[int], color: bool) -> None:
            if message_number is not None:
                out = f"[#{message_number}]"
                if color:
                    out = colorize(out, colorizer.Style.DIM)
                return out
            return ""

        def format_persona(role: str, color: bool) -> None:
            persona = settings.role_names.get(role, None)
            if role == "system" and persona is None:
                out = "<<system>>"
                if color:
                    out = colorize(out, colorizer.Style.DIM)
                return out
            else:
                out = persona
                if color:
                    out = colorize(out, colorizer.Style.BRIGHT)
                return f"{out}:"

        def format_message_heading(message_number: Optional[int], role: str, color: bool):
            colorful_number = format_message_number(message_number, color)
            colorful_persona = format_persona(role, color)
            if message_number is not None:
                return f"{colorful_number} {colorful_persona} "
            else:
                return f"{colorful_persona} "

        def chat_print_message(message_number: Optional[int], role: str, content: str) -> None:
            print(format_message_heading(message_number, role, color=True), end="")
            print(remove_persona_from_start_of_line(settings=settings, role=role, content=content))

        def chat_print_history(history: List[Dict[str, str]]) -> None:
            for k, item in enumerate(history):
                chat_print_message(k, item["role"], item["content"])
                print()

        history = new_chat(settings)
        new_chat_history = history
        chat_print_history(history)  # show initial blank history
        injectors = [format_chat_datetime_now,  # let the LLM know the current local time and date
                     format_reminder_to_focus_on_latest_input]  # remind the LLM to focus on user's last message (some models such as the distills of DeepSeek-R1 need this to support multi-turn conversation)
        while True:
            original_history = history
            user_message_number = len(history)
            ai_message_number = user_message_number + 1

            # User's turn - print a user input prompt and get the user's input.
            #
            # The `readline` module takes its user input prompt from what we supply to `input`, so we must print the prompt via `input`, colors and all.
            # The colorizer automatically wraps the ANSI color escape codes (for the terminal app) in ASCII escape codes (for `readline` itself)
            # that tell `readline` not to include them in its visual length calculation.
            #
            # This avoids the input prompt getting overwritten when browsing history entries, and prevents backspacing over the input prompt.
            # https://stackoverflow.com/questions/75987688/how-can-readline-be-told-not-to-erase-externally-supplied-prompt
            input_prompt = format_message_heading(user_message_number, role="user", color=True)
            user_message = input(input_prompt)
            print()

            # Interpret special commands for this LLM client
            if user_message == "!clear":
                print(colorize("Starting new chat session (resetting history)", colorizer.Style.BRIGHT))
                history = new_chat_history
                chat_print_history(history)
                continue
            elif user_message == "!history":
                print(colorize("Chat history (cleaned up):", colorizer.Style.BRIGHT))
                print(colorize("=" * 80, colorizer.Style.BRIGHT))
                chat_print_history(history)
                print(colorize("=" * 80, colorizer.Style.BRIGHT))
                print()
                continue
            elif user_message == "!model":
                chat_show_model_info()
                continue
            elif user_message == "!models":
                chat_show_list_of_models()
                continue
            elif user_message == "!help":
                chat_show_help()
                continue

            # Prepare to prompt the LLM.

            # Perform the temporary injects.
            for thunk in injectors:
                history = inject_chat_message(depth=0, settings=settings, history=history, role="system", message=thunk())

            # Add the user's message to the context.
            history = add_chat_message(settings, history, role="user", message=user_message)

            # Now:
            #   -1 = user's last message
            #   -2 = last inject

            # AI's turn - prompt the LLM.
            print(format_message_number(ai_message_number, color=True))
            chars = 0
            def progress_callback(n_chunks, chunk_text):  # any UI live-update code goes here, in the callback
                # TODO: think of a better way to split to lines
                nonlocal chars
                chars += len(chunk_text)
                if "\n" in chunk_text:  # one token at a time; should have either one linefeed or no linefeed
                    chars = 0  # good enough?
                elif chars >= output_line_width:
                    print()
                    chars = 0
                print(chunk_text, end="")
                sys.stdout.flush()
            out = invoke(settings, history, progress_callback)  # `out.data` is now the complete response
            print()  # add the final newline

            # Clean up the AI's reply (heuristically).
            out.data = clean(settings, out.data, thoughts_mode="discard", add_ai_persona_name=False)  # persona name is already added by `add_chat_message`

            # Show performance statistics
            print(colorize(f"[{out.n_tokens}t, {out.dt:0.2f}s, {out.n_tokens/out.dt:0.2f}t/s]", colorizer.Style.DIM))
            print()

            # Discard the temporary injects (resetting history to the start of this round, before user's message).
            history = original_history

            # Add the user's and the AI's messages to the chat history.
            history = add_chat_message(settings, history, role="user", message=user_message)
            history = add_chat_message(settings, history, role="assistant", message=out.data)
    except (EOFError, KeyboardInterrupt):
        print()
        print(colorize("Exiting chat.", colorizer.Style.BRIGHT))
        print()

def main():
    parser = argparse.ArgumentParser(description="""Minimal LLM chat client, for testing/debugging. You can use this for testing that Raven can connect to your LLM.""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(dest="backend_url", nargs="?", default=config.llm_backend_url, type=str, metavar="url", help="where to access the LLM API")
    opts = parser.parse_args()

    chat(opts.backend_url)

if __name__ == "__main__":
    main()
