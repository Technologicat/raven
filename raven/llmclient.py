"""LLM client for Raven.

NOTE for oobabooga/text-generation-webui users:

If you want to see the final prompt in instruct or chat mode, start your server in `--verbose` mode.
"""

__all__ = ["list_models", "setup",
           "create_chat_message", "create_initial_system_message",
           "factory_reset_chat_datastore",
           "format_chat_datetime_now", "format_reminder_to_focus_on_latest_input", "format_reminder_to_use_information_from_context_only",
           "scrub",
           "invoke", "perform_tool_calls"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse
import atexit
import collections
import copy
import datetime
import io
import json
import os
import pathlib
import platform
import re
import requests
import sys
from textwrap import dedent
from typing import Dict, List, Optional

import sseclient  # pip install sseclient-py

from mcpyrate import colorizer

from unpythonic import sym, timer, Values
from unpythonic.env import env

from . import chattree
from . import config
from . import hybridir

from .avatar.client import api as avatar_api  # raven-avatar client communication setup is in `raven.avatar.client.config`, used automatically.
from .avatar.client import config as client_config

# --------------------------------------------------------------------------------
# Module bootup

config_dir = pathlib.Path(config.llm_save_dir).expanduser().resolve()

avatar_api.init_module(avatar_url=client_config.avatar_url,
                       avatar_api_key_file=client_config.avatar_api_key_file,
                       tts_url=client_config.tts_url,
                       tts_api_key_file=client_config.tts_api_key_file,
                       tts_server_type=client_config.tts_server_type)  # let it create a default executor

# ----------------------------------------
# LLM communication setup

api_key_file = config_dir / "api_key.txt"

# HTTP headers for LLM requests
headers = {
    "Content-Type": "application/json"
}

# Read API key for cloud LLM support
if os.path.exists(api_key_file):  # TODO: test this (implemented according to spec)
    with open(api_key_file, "r", encoding="utf-8") as f:
        api_key = f.read()
    # "Authorization": "Bearer yourPassword123"
    # https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API
    headers["Authorization"] = api_key.strip()

# --------------------------------------------------------------------------------
# Websearch integration (requires `raven.avatar.server`)

def websearch_wrapper(query: str, engine: str = "duckduckgo", max_links: int = 10) -> str:
    """Perform a websearch, using the Avatar server to handle the interaction with the search engine and the parsing of the results page."""
    data = avatar_api.websearch_search(query, engine, max_links)
    return data["results"]  # TODO: our LLM scaffolding doesn't currently accept anything else but preformatted text

# --------------------------------------------------------------------------------
# Utilities

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
        `role_names`: A `dict` with keys "user", "assistant", "system", used for constructing chat messages (see `create_chat_message`).
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

    # For recent models as of April 2025, e.g. QwQ-32B, the system prompt itself can be blank.
    # The character card is enough.
    #
    # Older models may need a general briefing first.
    #
    # If you provide a system prompt, be sure to dedent and strip it
    # so the client can process whitespace in a unified way. See example.
    #
    # system_prompt = dedent(f"""You are an expert actor that can fully immerse yourself into any role given. You do not break character for any reason, even if someone tries addressing you as an AI or language model. Currently your role is {char}, which is described in detail below. As {char}, continue the exchange with {user}.""").strip()  # "Actor" preset from SillyTavern.
    system_prompt = ""

    # This is a minimal setup, partially copied from my personal AI assistant, meant to be run
    # against locally hosted models. This gives better performance (accuracy, instruction following)
    # vs. querying the LLM directly without any system prompt.
    #
    # TODO: "If unsure" and similar tricks tend to not work for 8B models. At LLaMA 3.1 70B and better, it should work, but running that requires at least 2x24GB VRAM.
    # TODO: Query the context size from the backend if possible. No, doesn't seem to be possible. https://github.com/oobabooga/text-generation-webui/discussions/5317
    #
    character_card = dedent(f"""
    Note that {user} cannot see this introductory text; it is only used internally, to initialize the LLM (large language model).

    **About {char}**

    You are {char} (she/her), an AI assistant. You are highly intelligent. You have been trained to answer questions, provide recommendations, and help with decision making.

    **About the system**

    The LLM version is "{model}".

    The knowledge cutoff date of the model is not specified, but is most likely within the year 2024. The knowledge cutoff date applies only to your internal knowledge. Any information provided in the context as well as web search results may be newer.

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
      - Tool calls.
      - Additional context that is provided by the software this LLM is running in.
    """).strip()

    # The AI's initial greeting. Used when a new chat is started.
    greeting = "How can I help you today?"

    # Tools (functions) to make available to the AI for tool-calling (for models that support that - as of May 2025, at least Qwen 2 or later do).
    tools = [
        {"type": "function",
         "function": {"name": "websearch",
                      "description": "Perform a web search.",
                      "parameters": {"type": "object",
                                     "required": ["query"],
                                     "properties": {"query": {"type": "string",
                                                              "description": "The search query."}}}}}
    ]
    tool_entrypoints = {"websearch": websearch_wrapper}

    if config.llm_send_toolcall_instructions:
        tools_json = "\n".join(json.dumps(tool) for tool in tools)

        # This comes from the template built into QwQ-32B.
        tools_info = dedent(f"""
        # Tools

        You may call one or more functions to assist with the user query.

        You are provided with function signatures within <tools></tools> XML tags:
        <tools>
        {tools_json}
        </tools>

        For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
        <tool_call>
        {{"name": <function-name>, "arguments": <args-json-object>}}
        </tool_call>
        """).strip()
        character_card = f"{character_card}\n\n{tools_info}"

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
        "tools": tools,  # Tools available for tool-calling, for models that support that (as of 16 May 2025, need dev branch of ooba).
        "name1": user,  # Name of user's persona in the chat.
        "name2": char,  # Name of AI's persona in the chat.
    }

    # See `create_chat_message`.
    role_names = {"user": user,
                  "assistant": char,
                  "system": None,
                  "tool": None}

    # List of strings after which to interrupt the LLM.
    # Useful mainly with older models that tend to speak on behalf of the user.
    stopping_strings = [f"\n{user}:"]

    settings = env(user=user, char=char, model=model,
                   system_prompt=system_prompt,
                   character_card=character_card,
                   stopping_strings=stopping_strings,
                   greeting=greeting,
                   tools=tools,  # for inspection
                   tool_entrypoints=tool_entrypoints,  # for our implementation to be able to call them
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

def create_chat_message(settings: env,
                        role: str,
                        text: str,
                        add_role_name: bool = True,
                        tool_calls: List[str] = None) -> Dict:
    """Create a new chat message, compatible with the chat history format sent to the LLM.

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

    if add_role_name and settings.role_names[role] is not None:
        content = f"{settings.role_names[role]}: {text}"  # e.g. "User: ..."
    else:  # System and tool messages typically do not use a speaker tag in the text content.
        content = text

    data = {"role": role,
            "content": content,
            "tool_calls": tool_calls if tool_calls is not None else []}
    return data

def create_initial_system_message(settings: env) -> Dict:
    """Create a chat message containing the system prompt and the AI's character card as specified in `settings`."""
    if settings.system_prompt and settings.character_card:
        # The system prompt is stripped, so we need two linefeeds to have one blank line in between.
        text = f"{settings.system_prompt}\n\n{settings.character_card}\n\n-----"
    elif settings.system_prompt:
        text = f"{settings.system_prompt}\n\n-----"
    elif settings.character_card:
        text = f"{settings.character_card}\n\n-----"
    else:
        raise ValueError("create_initial_system_message: Need at least a system prompt or a character card.")
    return create_chat_message(settings,
                               role="system",
                               text=text)

def factory_reset_chat_datastore(datastore: chattree.PersistentForest, settings: env) -> str:
    """Reset `datastore` to its "factory-default" state.

    **IMPORTANT**: This deletes all existing chat nodes in the datastore, and CANNOT BE UNDONE.

    The primary purpose of this function is to initialize the chat datastore when it hasn't been created yet.

    This creates a root node containing the system prompt (including the character card), and a node for the AI's initial greeting.

    Returns the unique ID of the initial greeting node, so you can start building chats on top of that.

    You can obtain the `settings` object by first calling `setup`.
    """
    datastore.purge()
    root_node_id = datastore.create_node(data=create_initial_system_message(settings),
                                         parent_id=None)
    new_chat_node_id = datastore.create_node(data=create_chat_message(settings,
                                                                      role="assistant",
                                                                      text=settings.greeting),
                                                parent_id=root_node_id)
    return new_chat_node_id

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


_complete_thought_block = re.compile(r"([<\[])(think(ing)?[>\]])(.*?)\1/\2\s*", flags=re.IGNORECASE | re.DOTALL)  # opened and closed correctly; thought contents -> group 4
_incomplete_thought_block = re.compile(r"([<\[])(think|thinking)([>\]])(?!.*?\1/\2\3)(.*)", flags=re.IGNORECASE | re.DOTALL)  # opened but not closed; thought contents -> group 4
_doubled_think_tag = re.compile(r"([<\[])(think|thinking)([>\]])\n([<\[])(think|thinking)([>\]])", flags=re.IGNORECASE | re.DOTALL)
_nan_thought_block = re.compile(r"([<\[])(think|thinking)([>\]])\nNaN\n([<\[])/(think|thinking)([>\]])\n", flags=re.IGNORECASE | re.DOTALL)
_thought_begin_tag = re.compile(r"([<\[])(think|thinking)([>\]])", flags=re.IGNORECASE | re.DOTALL)
_thought_end_tag = re.compile(r"([<\[])/(think|thinking)([>\]])", flags=re.IGNORECASE | re.DOTALL)

def remove_role_name_from_start_of_line(settings: env, role: str, text: str) -> str:
    """Transform e.g. "User: blah blah" -> "blah blah", for every line in `text`."""
    persona = settings.role_names.get(role, None)
    if persona is None:
        return text
    _role_name_at_start_of_line = re.compile(f"^{persona}:\\s+", re.MULTILINE)
    text = re.sub(_role_name_at_start_of_line, r"", text)
    return text

def scrub(settings: env, text: str, thoughts_mode: str, add_ai_role_name: bool) -> str:
    """Heuristically clean up the text content of an LLM-generated message.

    `settings`: Obtain this by calling `setup()` at app start time.
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
    text = remove_role_name_from_start_of_line(settings=settings, role="assistant", text=text)

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
    # Cases where we DON'T need to do this:
    #   - Chat app, which usually has a separate UI element for the persona name, aside from the actual chat text content UI element.
    #   - Piping output to a script, in which case the chat framework is superfluous. In that use case, we really use the LLM
    #     as an instruct-tuned model, i.e. a natural language processor that is programmed via free-form instructions in English.
    #     Raven's PDF importer does this a lot.
    if add_ai_role_name:
        text = f"{settings.char}: {text}"

    return text


# --------------------------------------------------------------------------------
# The most important function - call LLM, parse result

def invoke(settings: env, history: List[Dict], progress_callback=None) -> env:
    """Invoke the LLM with the given chat history.

    This is typically done after adding the user's message to the chat history, to ask the LLM to generate a reply.

    `settings`: Obtain this by calling `setup()` at app start time.

    `history`: List of chat messages, see `create_chat_message`.

    `progress_callback`: callable, optional.
        If provided, this is called for each chunk. It is expected to take two arguments; the signature is
        `progress_callback(n_chunks, chunk_text)`. Here:
            `n_chunks`: int, how many chunks have been generated so far (for this invocation).
                        This is useful if you want to e.g. print a progress symbol to the terminal every ten chunks.
            `chunk_text` str, the actual text of the current chunk.
        Typically, at least with `oobabooga/text-generation-webui`, one chunk = one token.

    Returns an `unpythonic.env.env` WITHOUT adding the LLM's reply to `history`.

    The returned `env` has the following attributes:

        `data`: dict, The new message generated by the LLM (for the format, see `create_chat_message`).
                      If the text content begins with the assistant character's name (e.g. "AI: ..."), this is automatically stripped.
        `n_tokens`: int, Number of tokens emitted by the LLM.
        `dt`: float, Wall time elapsed for this invocation, in seconds.
    """
    data = copy.deepcopy(settings.request_data)
    data["messages"] = history
    stream_response = requests.post(f"{settings.backend_url}/v1/chat/completions", headers=headers, json=data, verify=False, stream=True)

    if stream_response.status_code != 200:  # not "200 OK"?
        logger.error(f"LLM server returned error: {stream_response.status_code} {stream_response.reason}. Content of error response follows.")
        logger.error(stream_response.text)
        raise RuntimeError(f"While calling LLM: HTTP {stream_response.status_code} {stream_response.reason}")

    client = sseclient.SSEClient(stream_response)

    llm_output_text = io.StringIO()
    last_few_chunks = collections.deque([""] * 10)  # ring buffer for quickly checking a short amount of text at the current end; prepopulate with empty strings since `popleft` requires at least one element to be present
    n_chunks = 0
    stopped = False  # whether one of the stop strings triggered
    try:
        with timer() as tim:
            for event in client.events():
                payload = json.loads(event.data)
                # print(payload, file=sys.stderr)  # DEBUG / basic science against API
                delta = payload["choices"][0]["delta"]
                chunk = delta["content"]
                n_chunks += 1
                llm_output_text.write(chunk)

                # Check for stopping strings (helps with some models that start talking on behalf of the user)
                last_few_chunks.append(chunk)
                last_few_chunks.popleft()
                recent_text = "".join(last_few_chunks)  # Note start-of-word LLM tokens begin with a space.
                stop = [stopping_string in recent_text for stopping_string in settings.stopping_strings]  # check which stopping strings match (if any)
                if any(stop):
                    # The local LLM is OpenAI-compatible, so the same trick works - to tell the server to stop, just close the stream.
                    # https://community.openai.com/t/interrupting-completion-stream-in-python/30628/7
                    client.close()
                    stopped = True
                    break

                if progress_callback is not None:
                    progress_callback(n_chunks, chunk)
    except KeyboardInterrupt:  # stop generating on Ctrl+C
        client.close()
        raise
    except requests.exceptions.ChunkedEncodingError as exc:
        logger.error(f"Connection lost. Please check if your LLM backend is still alive (was at {settings.backend_url}). Original error message follows.")
        logger.error(f"{type(exc)}: {exc}")
        raise
    llm_output_text = llm_output_text.getvalue()

    # Tool calls come in the last chunk, with empty text content, when "finish_reason = tool_calls", e.g.:
    #
    # {'id': 'chatcmpl-1747386255850717184', 'object': 'chat.completion.chunk', 'created': 1747386255, 'model': 'Qwen_QwQ-32B-Q4_K_M.gguf',
    #  'choices': [{'index': 0,
    #               'finish_reason': 'tool_calls',
    #               'delta': {'role': 'assistant',
    #                         'content': '',
    #                         'tool_calls': [{'type': 'function',
    #                                         'function': {'name': 'websearch',
    #                                                      'arguments': '{"query": "Sharon Apple"}'},
    #                                         'id': 'call_m357947b',
    #                                         'index': '0'}]
    #                        }}],
    #  'usage': {'prompt_tokens': 674, 'completion_tokens': 264, 'total_tokens': 938}}
    #
    # See also the backend source code, particularly:
    #   https://github.com/oobabooga/text-generation-webui/blob/dev/extensions/openai/typing.py
    #   https://github.com/oobabooga/text-generation-webui/blob/dev/extensions/openai/completions.py

    tool_calls = None
    if stopped:  # due to a stopping string
        # From the final LLM output, remove the longest suffix that is in the stopping strings
        matched_stopping_strings = [stopping_string for is_match, stopping_string in zip(stop, settings.stopping_strings) if is_match]
        assert matched_stopping_strings  # we only get here if at least one stopping string matches
        stopping_string_start_positions = [llm_output_text.rfind(match) for match in matched_stopping_strings]
        assert not any(start_position == -1 for start_position in stopping_string_start_positions)  # we only checked matching strings
        chop_position = min(stopping_string_start_positions)
        llm_output_text = llm_output_text[:chop_position]
    else:  # normal finish, by LLM server
        if "tool_calls" in delta:
            tool_calls = delta["tool_calls"]

    # In streaming mode, the oobabooga backend always yields an initial chunk with empty content (perhaps to indicate that the connection was successful?)
    # and at the end, a final chunk with empty content, containing usage stats and possible tool calls.
    #
    # The correct way to get the number of tokens is to read the "usage" field of the final chunk.
    # Of course, if we have to close the connection on our end due to a stopping string, we won't get that chunk.
    # But this hack always works.
    n_tokens = n_chunks - 2

    message = create_chat_message(settings=settings,
                                  role="assistant",
                                  text=llm_output_text,
                                  add_role_name=False,
                                  tool_calls=tool_calls)
    return env(data=message,
               n_tokens=n_tokens,
               dt=tim.dt)

def perform_tool_calls(settings: env, message: Dict) -> List[Dict]:
    """Perform tool calls as requested in `message["tool_calls"]`.

    Return a list of messages (each with `role="tool"`) containing the tool outputs.

    If the `tool_calls` field of the message is missing or if it is empty, return the empty list.
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

    tool_response_messages = []
    def add_tool_response_message(text):
        tool_response_message = create_chat_message(settings=settings,
                                                    role="tool",
                                                    text=text,
                                                    add_role_name=False)
        tool_response_messages.append(tool_response_message)

    for request_record in tool_calls:
        if "type" not in request_record:
            # The response message is intended for the LLM, whereas the log message (with all technical details) goes into the log.
            add_tool_response_message("Tool call failed. The request is missing the 'type' field.")
            logger.warning(f"perform_tool_calls: missing 'type' field in request. Data: {request_record}")
            continue
        if request_record["type"] != "function":
            add_tool_response_message(f"Tool call failed. Unknown request type '{request_record['type']}'; expected 'function'.")
            logger.warning(f"perform_tool_calls: unknown type '{request_record['type']}' in request, expected 'function'. Data: {request_record}")
            continue
        if "function" not in request_record:
            add_tool_response_message("Tool call failed. The request is missing the 'function' field.")
            logger.warning(f"perform_tool_calls: missing 'function' field. Data: {request_record}")
            continue

        function_record = request_record["function"]
        if "name" not in function_record:
            add_tool_response_message("Tool call failed. The request's function record is missing the 'name' field.")
            logger.warning(f"perform_tool_calls: missing 'function.name' field in request. Data: {request_record}")
            continue

        function_name = function_record["name"]
        try:
            function = settings.tool_entrypoints[function_name]
        except KeyError:
            add_tool_response_message(f"Tool call failed. Function not found: '{function_name}'.")
            logger.warning(f"perform_tool_calls: unknown function '{function_name}'.")
            continue

        if "arguments" in function_record:
            try:
                kwargs = json.loads(function_record["arguments"])
            except Exception as exc:
                add_tool_response_message(f"Tool call failed. When calling '{function_name}', failed to parse the request's JSON for the function arguments.")
                logger.warning(f"perform_tool_calls: function '{function_name}': failed to parse JSON for arguments: {type(exc)}: {exc}")
                continue
            else:
                logger.debug(f"perform_tool_calls: calling '{function_name}' with arguments {kwargs}.")
        else:
            logger.debug(f"perform_tool_calls: for function '{function_name}: The request's function record is missing the 'arguments' field. Calling without arguments.")
            kwargs = {}

        # TODO: get the tool call ID (OpenAI compatible API) and add it to the message
        # TODO: websearch return format: for the chat history, need only the preformatted text, but for the eventual GUI, would be nice to have the links separately.
        try:
            tool_output_text = function(**kwargs)
        except Exception as exc:
            add_tool_response_message(f"Tool call failed. Function '{function_name}' exited with exception {type(exc)}: {exc}")
            logger.warning(f"perform_tool_calls: function '{function_name}': exited with exception {type(exc)}: {exc}")
        else:  # success!
            logger.debug(f"perform_tool_calls: Function '{function_name}' returned successfully.")
            add_tool_response_message(tool_output_text)

    return tool_response_messages

# --------------------------------------------------------------------------------
# Minimal chat client for testing/debugging that Raven can connect to your LLM.
#
# Although for a "minimal" client, this does have some fancy features, such as branching history and RAG.
#
# Also a usage example for the API of this module.

def minimal_chat_client(backend_url):
    """Minimal LLM chat client, for testing/debugging."""

    history_file = config_dir / "history"      # user input history (readline)
    datastore_file = config_dir / "data.json"  # chat node datastore
    state_file = config_dir / "state.json"     # important node IDs for the chat client state

    docs_dir = pathlib.Path(config.llm_docs_dir).expanduser().resolve()  # RAG documents (put your documents in this directory)
    db_dir = pathlib.Path(config.llm_database_dir).expanduser().resolve()  # RAG search indices datastore

    datastore = None  # initialized later, during app startup
    def load_app_state(settings: env) -> Dict:
        if datastore is None:
            assert False  # `datastore` must be initialized before this internal function is called

        def new_datastore():
            state["new_chat_HEAD"] = factory_reset_chat_datastore(datastore, settings)  # do this first - this creates the first two nodes (system prompt with character card, and the AI's initial greeting)
            state["HEAD"] = state["new_chat_HEAD"]  # current last node in chat; like HEAD pointer in git

        try:
            with open(state_file, "r", encoding="utf-8") as json_file:
                state = json.load(json_file)
        except FileNotFoundError:
            new_datastore()
            state["docs_enabled"] = True

        if not datastore.nodes:  # No stored chat history -> reset datastore
            logger.warning("load_app_state: no chat nodes in datastore, creating new datastore")
            new_datastore()

        if "new_chat_HEAD" not in state:  # New-chat start node ID missing -> reset datastore
            logger.warning(f"load_app_state: missing key 'new_chat_HEAD' in '{state_file}', creating new datastore")
            new_datastore()

        if "HEAD" not in state:  # Current chat node ID missing -> start at new chat
            logger.warning(f"load_app_state: missing key 'HEAD' in '{state_file}', resetting it to 'new_chat_HEAD'")
            state["HEAD"] = state["new_chat_HEAD"]

        if "docs_enabled" not in state:
            logger.warning(f"load_app_state: missing key 'docs_enabled' in '{state_file}', using default")
            state["docs_enabled"] = True

        if "speculate_enabled" not in state:
            logger.warning(f"load_app_state: missing key 'speculate_enabled' in '{state_file}', using default")
            state["speculate_enabled"] = False

        # Refresh the system prompt in the datastore (to the one in this client's source code)
        new_chat_node_id = state["new_chat_HEAD"]
        system_prompt_node_id = datastore.nodes[new_chat_node_id]["parent"]
        datastore.nodes[system_prompt_node_id]["data"] = create_initial_system_message(settings)

        print(colorizer.colorize(f"Loaded app state from '{state_file}'.", colorizer.Style.BRIGHT))
        return state

    def save_app_state(state: Dict) -> None:
        # validate
        required_keys = ("new_chat_HEAD",
                         "HEAD",
                         "docs_enabled",
                         "speculate_enabled")
        if any(key not in state for key in required_keys):
            raise KeyError  # at least one required setting missing from `state`

        with open(state_file, "w", encoding="utf-8") as json_file:
            json.dump(state, json_file, indent=2)

    # Ugh for the presentation order, but this is needed in two places, starting immediately below.
    def chat_show_model_info():
        print(f"    {colorizer.colorize('Model', colorizer.Style.BRIGHT)}: {settings.model}")
        print(f"    {colorizer.colorize('Character', colorizer.Style.BRIGHT)}: {settings.char} [defined in this client]")
        print()

    # Main program
    try:
        # API key already loaded during module bootup; here, we just inform the user.
        if "Authorization" in headers:
            print()
            print(f"Loaded LLM API key from '{str(api_key_file)}'.")
            print()
        else:
            print()
            print(f"No LLM API key configured. If your LLM needs an API key to connect, put it into '{str(api_key_file)}'.")
            print("This can be any plain-text data your LLM's API accepts in the 'Authorization' field of the HTTP headers.")
            print("For username/password, the format is 'user pass'. Do NOT use a plaintext password over an unencrypted http:// connection!")
            print()

        try:
            list_models(backend_url)  # just do something, to try to connect
        except requests.exceptions.ConnectionError as exc:
            print(colorizer.colorize(f"Cannot connect to LLM backend at {backend_url}.", colorizer.Style.BRIGHT, colorizer.Fore.RED) + " Is the LLM server running?")
            msg = f"Failed to connect to LLM backend at {backend_url}, reason {type(exc)}: {exc}"
            logger.error(msg)
            sys.exit(255)
        else:
            print(colorizer.colorize(f"Connected to LLM backend at {backend_url}", colorizer.Style.BRIGHT, colorizer.Fore.GREEN))
            settings = setup(backend_url=backend_url)
            chat_show_model_info()

        # Persistent, branching chat history.
        datastore = chattree.PersistentForest(datastore_file)  # This autoloads and auto-persists.
        if datastore.nodes:
            print(colorizer.colorize(f"Loaded chat datastore from '{datastore_file}'.", colorizer.Style.BRIGHT))
        state = load_app_state(settings)
        print()

        # Load RAG database (it will auto-persist at app exit).
        retriever, _unused_scanner = hybridir.setup(docs_dir=docs_dir,
                                                    recursive=config.llm_docs_dir_recursive,
                                                    db_dir=db_dir,
                                                    embedding_model_name=config.qa_embedding_model)
        docs_enabled_str = "ON" if state["docs_enabled"] else "OFF"
        colorful_rag_status = colorizer.colorize(f"RAG (retrieval-augmented generation) autosearch is currently {docs_enabled_str}.",
                                                 colorizer.Style.BRIGHT)
        print(f"{colorful_rag_status} Toggle with the `!docs` command.")
        print(f"    Document store is at '{config.llm_docs_dir}'.")
        # The retriever's `documents` attribute must be locked before accessing.
        with retriever.datastore_lock:
            plural_s = "s" if len(retriever.documents) != 1 else ""
            print(f"        {len(retriever.documents)} document{plural_s} loaded.")
        print(f"    Search indices are saved in '{config.llm_database_dir}'.")
        print()

        import readline  # noqa: F401, side effect: enable GNU readline in builtin input()
        # import rlcompleter  # noqa: F401, side effects: readline tab completion for Python code
        print(colorizer.colorize(f"GNU readline available. Saving user inputs to '{str(history_file)}'.", colorizer.Style.BRIGHT))
        print(colorizer.colorize("Use up/down arrows to browse previous inputs. Enter to send. ", colorizer.Style.BRIGHT))
        print()
        try:
            readline.read_history_file(history_file)
        except FileNotFoundError:
            pass

        # Set up autosave at exit.
        def persist():
            config_dir.mkdir(parents=True, exist_ok=True)

            # Save readline history
            readline.set_history_length(1000)
            readline.write_history_file(history_file)

            try:
                save_app_state(state)
            except KeyError:
                logger.warning(f"During app shutdown: app `state` missing at least one required key, cannot persist it. Existing keys: {list(state.keys())}")

            # Before saving (which happens automatically at exit),
            # remove any nodes not reachable from the initial message, and also remove dead links.
            # There shouldn't be any, but this way we exercise these features, too.
            try:
                new_chat_node_id = state["new_chat_HEAD"]
                system_prompt_node_id = datastore.nodes[new_chat_node_id]["parent"]
            except KeyError as exc:
                logger.warning(f"During app shutdown: while pruning chat forest: {type(exc)}: {exc}")
            else:
                datastore.prune_unreachable_nodes(system_prompt_node_id)
                datastore.prune_dead_links(system_prompt_node_id)
        # We register later than `chattree.PersistentForest` (which we already instantiated above), so ours runs first.
        # Hence we'll have the chance to prune before the forest is persisted to disk.
        #     https://docs.python.org/3/library/atexit.html
        atexit.register(persist)

        print(colorizer.colorize("Starting chat.", colorizer.Style.BRIGHT))
        print()
        def chat_show_help():
            print(colorizer.colorize("=" * 80, colorizer.Style.BRIGHT))
            print("    llmclient.py - Minimal LLM client for testing/debugging.")
            print()
            print("    Special commands (tab-completion available):")
            print("        !clear                  - Start new chat")
            print(f"        !docs [True|False]      - RAG autosearch on/off/toggle (currently {state['docs_enabled']}; document store at '{config.llm_docs_dir}')")
            print(f"        !speculate [True|False] - LLM speculate on/off/toggle (currently {state['speculate_enabled']}); used only if docs is True.")
            print("                                  If speculate is False, try to use only RAG results to answer.")
            print("                                  If speculate is True, let the LLM respond however it wants.")
            print("        !dump                   - See raw contents of chat node datastore")
            print("        !head some-node-id      - Switch to another chat branch (get the node ID from `!dump`)")
            print("        !history                - Print a cleaned-up transcript of the current chat branch")
            print("        !model                  - Show which model is in use")
            print("        !models                 - List all models available at connected backend")
            print("        !help                   - Show this message again")
            print()
            print("    Press Ctrl+D to exit chat.")
            print(colorizer.colorize("=" * 80, colorizer.Style.BRIGHT))
            print()
        chat_show_help()

        # We prefill the space for commands that take an argument.
        commands = ["!clear",
                    "!docs ",
                    "!dump",
                    "!head ",
                    "!help",
                    "!history",
                    "!model",
                    "!models",
                    "!speculate"]
        def get_completions(candidates, text):
            """Return matching completions for `text`.

            `candidates`: Every possible completion the system knows of, in the current context.
            `text`: Prefix text to complete. Can be the empty string, which matches all candidates.
            """
            if not text:  # If no text, all candidates match.
                return candidates
            assert text  # we have text

            # Score the completions for the given prefix `text`.
            # https://stackoverflow.com/questions/6718196/determine-the-common-prefix-of-multiple-strings
            scores = [len(os.path.commonprefix([text, candidate])) for candidate in candidates]
            max_score = max(scores)
            if max_score == 0:  # no match
                return None
            assert max_score > 0  # we have at least one match, of at least one character

            # Possible completions are those that scored best (i.e. match the longest matched prefix).
            completions = [candidate for candidate, score in zip(candidates, scores) if score == max_score]
            return completions
        # https://docs.python.org/3/library/readline.html#readline-completion
        def completer(text, state):  # completer for special commands
            buffer_content = readline.get_line_buffer()  # context: text before the part being completed (up to last delim)

            # TODO: fix one more failure mode, e.g. "!help !<tab>"
            if buffer_content.startswith("!") and text.startswith("!"):  # completing a command?
                candidates = commands
            elif buffer_content.startswith("!docs"):  # in `!docs` command, expecting an argument?
                candidates = ["True", "False"]
            elif buffer_content.startswith("!head"):  # in `!head` command, expecting an argument?
                candidates = list(sorted(datastore.nodes.keys()))
            elif buffer_content.startswith("!speculate"):  # in `!speculate` command, expecting an argument?
                candidates = ["True", "False"]
            else:  # anything else -> no completions
                return None

            completions = get_completions(candidates, text)
            if completions is None:  # no match
                return None
            if state >= len(completions):  # no more completions
                return None
            return completions[state]
        readline.set_completer(completer)
        readline.set_completer_delims(" ")

        # Support tab completion also on MacOSX. Not sure which way is better here.
        # Neither seems The Right Thing:
        #   - Detecting the platform (as we do now) assumes that MacOSX will always use `libedit`
        #     to provide `readline`.
        #   - Detecting the `readline` module's `__doc__` assumes that any future versions of `libedit`
        #     keep the mention of `libedit` in the docstring.
        # https://stackoverflow.com/questions/7116038/python-repl-tab-completion-on-macos
        # https://stackoverflow.com/questions/1854/how-to-identify-which-os-python-is-running-on
        #
        # if 'libedit' in readline.__doc__:  # MacOSX uses libedit, not GNU readline
        #     readline.parse_and_bind("bind ^I rl_complete")
        # else:  # Linux, Windows
        #     readline.parse_and_bind("tab: complete")
        if platform.system() == "Darwin":  # MacOSX
            readline.parse_and_bind("bind ^I rl_complete")
        else:  # "Linux", "Windows"
            readline.parse_and_bind("tab: complete")

        def chat_show_list_of_models():
            available_models = list_models(backend_url)
            print(colorizer.colorize("    Available models:", colorizer.Style.BRIGHT))
            for model_name in available_models:
                print(f"        {model_name}")
            print()

        # TODO: we don't need the `color=False` option now that we fixed `colorize` to work with `readline`/`input`.
        def format_message_number(message_number: Optional[int], color: bool) -> None:
            if message_number is not None:
                out = f"[#{message_number}]"
                if color:
                    out = colorizer.colorize(out, colorizer.Style.DIM)
                return out
            return ""

        def format_persona(role: str, color: bool) -> None:
            persona = settings.role_names.get(role, None)
            if persona is None:
                out = f"<<{role}>>"  # currently, this include "<<system>>" and "<<tool>>"
                if color:
                    out = colorizer.colorize(out, colorizer.Style.DIM)
                return out
            else:
                out = persona
                if color:
                    out = colorizer.colorize(out, colorizer.Style.BRIGHT)
                return f"{out}:"

        def format_message_heading(message_number: Optional[int], role: str, color: bool):
            colorful_number = format_message_number(message_number, color)
            colorful_persona = format_persona(role, color)
            if message_number is not None:
                return f"{colorful_number} {colorful_persona} "
            else:
                return f"{colorful_persona} "

        def chat_print_message(message_number: Optional[int], role: str, text: str) -> None:
            print(format_message_heading(message_number, role, color=True), end="")
            print(remove_role_name_from_start_of_line(settings=settings, role=role, text=text))

        def chat_print_history(history: List[Dict], show_numbers: bool = True) -> None:
            if show_numbers:
                for k, message in enumerate(history):
                    chat_print_message(message_number=k, role=message["role"], text=message["content"])
                    print()
            else:
                for message in history:
                    chat_print_message(message_number=None, role=message["role"], text=message["content"])
                    print()

        action_proceed = sym("proceed")  # proceed current round as normal
        action_next_round = sym("next_round")  # skip to start of next round, e.g. after a special command

        def user_turn() -> Values:
            history = datastore.linearize_up(state["HEAD"])
            user_message_number = len(history)

            # Print a user input prompt and get the user's input.
            #
            # The `readline` module takes its user input prompt from what we supply to `input`, so we must print the prompt via `input`, colors and all.
            # The colorizer automatically wraps the ANSI color escape codes (for the terminal app) in ASCII escape codes (for `readline` itself)
            # that tell `readline` not to include them in its visual length calculation.
            #
            # This avoids the input prompt getting overwritten when browsing history entries, and prevents backspacing over the input prompt.
            # https://stackoverflow.com/questions/75987688/how-can-readline-be-told-not-to-erase-externally-supplied-prompt
            input_prompt = format_message_heading(user_message_number, role="user", color=True)
            user_message_text = input(input_prompt)
            print()

            # Interpret special commands for this LLM client
            if user_message_text == "!clear":
                print(colorizer.colorize("Starting new chat session.", colorizer.Style.BRIGHT))
                state["HEAD"] = state["new_chat_HEAD"]
                print(f"HEAD is now at '{state['HEAD']}'.")
                print()
                chat_print_history(datastore.linearize_up(state["HEAD"]))
                return Values(action=action_next_round)
            elif user_message_text.startswith("!docs"):  # TODO: refactor
                split_command_text = user_message_text.split()
                nargs = len(split_command_text) - 1
                if nargs == 0:
                    state["docs_enabled"] = not state["docs_enabled"]
                elif nargs == 1:
                    arg = split_command_text[-1]
                    if arg == "True":
                        state["docs_enabled"] = True
                    elif arg == "False":
                        state["docs_enabled"] = False
                    else:
                        print(f"!docs: unrecognized argument '{arg}'; expected 'True' or 'False'.")
                        print()
                        return Values(action_next_round)
                else:
                    print("!docs: wrong number of arguments; expected at most one, 'True' or 'False'.")
                    print()
                    return Values(action=action_next_round)
                docs_enabled_str = "ON" if state["docs_enabled"] else "OFF"
                print(f"RAG autosearch is now {docs_enabled_str}.")
                print()
                return Values(action=action_next_round)
            elif user_message_text == "!dump":
                print(colorizer.colorize("Raw datastore content:", colorizer.Style.BRIGHT) + f" (current HEAD is at {state['HEAD']})")
                print(colorizer.colorize("=" * 80, colorizer.Style.BRIGHT))
                print(f"{datastore}", end="")  # -> str; also, already has the final blank line
                print(colorizer.colorize("=" * 80, colorizer.Style.BRIGHT))
                print()
                return Values(action=action_next_round)
            elif user_message_text.startswith("!head"):  # switch to another chat branch
                try:
                    _, new_head_id = user_message_text.split()
                except ValueError:
                    print("!head: wrong number of arguments; expected exactly one, the node ID to switch to; see `!dump` for available chat nodes.")
                    print()
                    return Values(action=action_next_round)
                if new_head_id not in datastore.nodes:
                    print(f"!head: no such chat node '{new_head_id}'; see `!dump` for available chat nodes.")
                    print()
                    return Values(action=action_next_round)
                state["HEAD"] = new_head_id
                print(f"HEAD is now at '{state['HEAD']}'.")
                print()
                chat_print_history(datastore.linearize_up(state["HEAD"]))
                return Values(action=action_next_round)
            elif user_message_text == "!help":
                chat_show_help()
                return Values(action=action_next_round)
            elif user_message_text == "!history":
                print(colorizer.colorize("Chat history (cleaned up):", colorizer.Style.BRIGHT))
                print(colorizer.colorize("=" * 80, colorizer.Style.BRIGHT))
                chat_print_history(history)
                print(colorizer.colorize("=" * 80, colorizer.Style.BRIGHT))
                print()
                return Values(action=action_next_round)
            elif user_message_text == "!model":
                chat_show_model_info()
                return Values(action=action_next_round)
            elif user_message_text == "!models":
                chat_show_list_of_models()
                return Values(action=action_next_round)
            elif user_message_text.startswith("!speculate"):  # TODO: refactor
                split_command_text = user_message_text.split()
                nargs = len(split_command_text) - 1
                if nargs == 0:
                    state["speculate_enabled"] = not state["speculate_enabled"]
                elif nargs == 1:
                    arg = split_command_text[-1]
                    if arg == "True":
                        state["speculate_enabled"] = True
                    elif arg == "False":
                        state["speculate_enabled"] = False
                    else:
                        print(f"!speculate: unrecognized argument '{arg}'; expected 'True' or 'False'.")
                        print()
                        return Values(action=action_next_round)
                else:
                    print("!speculate: wrong number of arguments; expected at most one, 'True' or 'False'.")
                    print()
                    return Values(action=action_next_round)
                speculate_enabled_str = "ON" if state["speculate_enabled"] else "OFF"
                print(f"LLM speculation is now {speculate_enabled_str}.")
                print()
                return Values(action=action_next_round)
            elif user_message_text.startswith("!") and len(user_message_text.split("\n")) == 1:
                print(f"Unrecognized command '{user_message_text}'; use `!help` for available commands.")
                return Values(action=action_next_round)
            # Not a special command.

            # Add the user's message to the chat.
            user_message_node_id = datastore.create_node(data=create_chat_message(settings=settings,
                                                                                  role="user",
                                                                                  text=user_message_text),
                                                         parent_id=state["HEAD"])
            state["HEAD"] = user_message_node_id
            return Values(action=action_proceed, text=user_message_text)

        def rag_search_with_bypass(query: str) -> Values:
            if not state["docs_enabled"]:
                return Values(action=action_proceed, matches=[])

            docs_results = retriever.query(query,
                                           k=10,
                                           return_extra_info=False)

            # First line of defense: docs on, no matches for given query, speculate off -> bypass LLM
            if not docs_results and not state["speculate_enabled"]:
                nomatch_text = "No matches in knowledge base. Please try another query."
                nomatch_message_node_id = datastore.create_node(data=create_chat_message(settings=settings,
                                                                                         role="assistant",
                                                                                         text=nomatch_text),
                                                                parent_id=state["HEAD"])
                nomatch_message_node = datastore.nodes[nomatch_message_node_id]
                nomatch_message_node["retrieval"] = {"query": query,
                                                     "results": []}  # store RAG results in the chat node that was generated based on them, for later use (upcoming citation mechanism)
                state["HEAD"] = nomatch_message_node_id

                history = datastore.linearize_up(state["HEAD"])
                nomatch_message_number = len(history)
                chat_print_message(message_number=nomatch_message_number,
                                   role="assistant",
                                   text=nomatch_text)
                print()

                return Values(action=action_next_round)

            return Values(action=action_proceed, matches=docs_results)

        # Perform the temporary injects. These are not meant to be persistent, so we don't even add them
        # as nodes to the chat tree, but only into the temporary linearized history.
        injectors = [format_chat_datetime_now,  # let the LLM know the current local time and date
                     format_reminder_to_focus_on_latest_input]  # remind the LLM to focus on user's last message (some models such as the distills of DeepSeek-R1 need this to support multi-turn conversation)
        def perform_injects(history: List[Dict], docs_matches: List[Dict]) -> None:
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
            #         message_to_inject = create_chat_message(settings=settings,
            #                                                 role="tool",
            #                                                 text=search_result_text)
            #         history.insert(position, message_to_inject)

            # Insert RAG results at the start of the history, as system messages.
            for docs_result in reversed(docs_matches):  # reverse to keep original order, because we insert each item at the same position.
                # TODO: Should the RAG match notification show the query string, too?
                search_result_text = f"[System information: Knowledge-base match from '{docs_result['document_id']}'.]\n\n{docs_result['text'].strip()}\n-----"
                message_to_inject = create_chat_message(settings=settings,
                                                        role="system",
                                                        text=search_result_text)
                history.insert(1, message_to_inject)  # after system prompt / character card combo

            # Always-on injects, e.g. current local datetime
            for thunk in injectors:
                message_to_inject = create_chat_message(settings=settings,
                                                        role="system",
                                                        text=thunk())
                history.append(message_to_inject)

            # If docs on, speculate off (-> `perform_injects` gets called if there is at least one RAG match), remind the LLM to use information from context only.
            #                           This increases the changes of the user's query working correctly when the search returns irrelevant results.
            # If docs off, the whole point is to use the LLM's static knowledge, so in that case don't bother.
            if state["docs_enabled"] and not state["speculate_enabled"]:
                message_to_inject = create_chat_message(settings=settings,
                                                        role="system",
                                                        text=format_reminder_to_use_information_from_context_only())
                history.append(message_to_inject)

            # # DEBUG - show history with injects.
            # # Message numbers counted from the modified history (with injects) would be wrong, so don't show them.
            # chat_print_history(history, show_numbers=False)

        def ai_turn(user_message_text: str) -> Values:
            # Perform the RAG autosearch (if enabled; will check automatically).
            # If docs is on, no match, and speculate is off -> bypass the LLM.
            #
            # NOTE: This is very rudimentary.
            #   - We simply use the user's new message as-is as the query.
            #   - Hence, this does NOT match on any earlier message, and may result in spurious matches.
            #     E.g. "Can cats jump?" and "Does your knowledge base say if cats can jump?" return
            #     different results, because the term "knowledge base" in the latter may match e.g.
            #     AI/CS articles that the user happens to have included in the KB.
            #     - In this example, with the example data, the shorter query correctly returns no matches.
            #     - The longer query returns two AI agent abstracts, leaving it to the LLM to put the
            #       pieces together and notice that the user's query and provided KB context don't actually match.
            #   - This could be improved by querying the LLM itself - "given the chat history so far and
            #     the user's most recent message, please formulate query terms for a knowledge base search."
            #     and then run the search with the final output of that.
            #   - We could also build a slightly more complex scaffold to support tool-calling,
            #     and instruct the LLM to send a query when it itself thinks it needs to.
            rag_query = user_message_text
            rag_result = rag_search_with_bypass(query=rag_query)
            if rag_result["action"] is action_next_round:  # bypass triggered
                return Values(action=action_next_round)

            # AI's turn: LLM generation interleaved with tool responses, until there are no tool calls in the LLM's latest reply.
            while True:
                history = datastore.linearize_up(state["HEAD"])  # latest history
                ai_message_number = len(history)

                # Prepare the final LLM prompt, by including the temporary injects.
                perform_injects(history, docs_matches=rag_result["matches"])

                # Invoke the LLM.
                print(format_message_number(ai_message_number, color=True))
                chars = 0
                def progress_callback(n_chunks, chunk_text):  # any UI live-update code goes here, in the callback
                    # TODO: think of a better way to split to lines
                    nonlocal chars
                    chars += len(chunk_text)
                    if "\n" in chunk_text:  # one token at a time; should have either one linefeed or no linefeed
                        chars = 0  # good enough?
                    elif chars >= config.llm_line_wrap_width:
                        print()
                        chars = 0
                    print(chunk_text, end="")
                    sys.stdout.flush()
                # `invoke` uses a linearized history, as expected by the LLM API.
                out = invoke(settings, history, progress_callback)  # `out.data` is now the complete message object (in the format returned by `create_chat_message`)
                print()  # print the final newline

                # Clean up the LLM's reply (heuristically). This version goes into the chat history.
                out.data["content"] = scrub(settings, out.data["content"], thoughts_mode="discard", add_ai_role_name=True)

                # Show LLM performance statistics
                print(colorizer.colorize(f"[{out.n_tokens}t, {out.dt:0.2f}s, {out.n_tokens/out.dt:0.2f}t/s]", colorizer.Style.DIM))
                print()

                # Add the LLM's message to the chat.
                ai_message_node_id = datastore.create_node(data=out.data,
                                                           parent_id=state["HEAD"])
                ai_message_node = datastore.nodes[ai_message_node_id]
                if state["docs_enabled"]:
                    ai_message_node["retrieval"] = {"query": rag_query,
                                                    "results": rag_result["matches"]}  # store RAG results in the chat node that was generated based on them, for later use (upcoming citation mechanism)
                state["HEAD"] = ai_message_node_id

                # Handle tool calls requested by the LLM, if any.
                #
                # Call the tool(s) specified by the LLM, with arguments specified by the LLM, and add the result to the chat.
                #
                # Each response goes into its own message, with `role="tool"`.
                #
                tool_message_number = ai_message_number + 1
                tool_response_messages = perform_tool_calls(settings, message=out.data)

                # When there are no more tool calls, the LLM is done replying.
                # Each tool call produces exactly one response, so we may as well check this from the number of responses.
                if not tool_response_messages:
                    break

                for tool_response_message in tool_response_messages:
                    tool_response_message_node_id = datastore.create_node(data=tool_response_message,
                                                                          parent_id=state["HEAD"])
                    state["HEAD"] = tool_response_message_node_id

                    chat_print_message(message_number=tool_message_number,
                                       role="tool",
                                       text=tool_response_message["content"])
                    print()

                    tool_message_number += 1

                # # DEBUG - show history after the tool calls, before the LLM starts writing again.
                # history = datastore.linearize_up(state["HEAD"])
                # chat_print_history(history, show_numbers=False)

            return Values(action=action_proceed)

        # Show initial history (loaded from datastore, or blank upon first start)
        chat_print_history(datastore.linearize_up(state["HEAD"]))

        # Main loop
        while True:
            user_result = user_turn()
            if user_result["action"] is action_next_round:
                continue

            # The AI needs the text of the user's latest message for the RAG autosearch query.
            ai_result = ai_turn(user_message_text=user_result["text"])
            if ai_result["action"] is action_next_round:
                continue  # Silly, since this is the last thing in the loop, but for symmetry.

    except (EOFError, KeyboardInterrupt):
        print()
        print(colorizer.colorize("Exiting chat.", colorizer.Style.BRIGHT))
        print()

def main():
    parser = argparse.ArgumentParser(description="""Minimal LLM chat client, for testing/debugging. You can use this for testing that Raven can connect to your LLM.""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(dest="backend_url", nargs="?", default=config.llm_backend_url, type=str, metavar="url", help=f"where to access the LLM API (default, currently '{config.llm_backend_url}', is set in `raven/config.py`)")
    opts = parser.parse_args()

    print()
    if avatar_api.avatar_available():
        print(colorizer.colorize(f"Connected to avatar server at {client_config.avatar_url}", colorizer.Style.BRIGHT, colorizer.Fore.GREEN))
        print(colorizer.colorize("The LLM will have access to websearch.", colorizer.Style.BRIGHT, colorizer.Fore.GREEN))
    else:
        print(colorizer.colorize(f"WARNING: Cannot connect to avatar server at {client_config.avatar_url}", colorizer.Style.BRIGHT, colorizer.Fore.YELLOW))
        print(colorizer.colorize("The LLM will NOT have access to websearch.", colorizer.Style.BRIGHT, colorizer.Fore.YELLOW))

    # print(websearch_wrapper("what is the airspeed velocity of an unladen swallow"))
    minimal_chat_client(opts.backend_url)

if __name__ == "__main__":
    main()
