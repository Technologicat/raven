"""LLM client library functions for Raven.

For an example chat client built using these, see `raven.librarian.minichat`.

NOTE for oobabooga/text-generation-webui users:

If you want to see the final prompt in instruct or chat mode, start your server in `--verbose` mode.
"""

__all__ = ["list_models", "setup",
           "token_count",
           "create_chat_message",
           "linearize_chat",
           "upgrade",
           "create_initial_system_message",
           "factory_reset_chat_datastore",
           "format_chat_datetime_now", "format_reminder_to_focus_on_latest_input", "format_reminder_to_use_information_from_context_only",
           "scrub",
           "invoke", "perform_tool_calls"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import collections
import copy
import datetime
import io
import json
import os
import re
import requests
from textwrap import dedent
from typing import Dict, List

import sseclient  # pip install sseclient-py

from mcpyrate import colorizer

from unpythonic import timer
from unpythonic.env import env

from ..client import api
from ..client import config as client_config

from . import chattree
from . import config as librarian_config

# --------------------------------------------------------------------------------
# Module bootup

api.initialize(raven_server_url=client_config.raven_server_url,
               raven_api_key_file=client_config.raven_api_key_file,
               tts_server_type=client_config.tts_server_type,
               tts_url=client_config.tts_url,
               tts_api_key_file=client_config.tts_api_key_file)  # let it create a default executor

# ----------------------------------------
# LLM communication setup

# HTTP headers for LLM requests
headers = {
    "Content-Type": "application/json"
}

# Read API key for cloud LLM support
if os.path.exists(librarian_config.llm_api_key_file):  # TODO: test this (implemented according to spec)
    with open(librarian_config.llm_api_key_file, "r", encoding="utf-8") as f:
        api_key = f.read().replace('\n', '')
    # "Authorization": "Bearer yourPassword123"
    # https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API
    headers["Authorization"] = api_key.strip()

# --------------------------------------------------------------------------------
# Websearch integration (requires `raven.server` to be running)

def websearch_wrapper(query: str, engine: str = "duckduckgo", max_links: int = 10) -> str:
    """Perform a websearch, using Raven-server to handle the interaction with the search engine and the parsing of the results page."""
    data = api.websearch_search(query, engine, max_links)
    return data["results"]  # TODO: our LLM scaffolding doesn't currently accept anything else but preformatted text

# --------------------------------------------------------------------------------
# Utilities

def list_models(backend_url: str) -> List[str]:
    """List all models available at `backend_url`."""
    response = requests.get(f"{backend_url}/v1/internal/model/list",
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

    if librarian_config.llm_send_toolcall_instructions:
        tools_json = "\n".join(json.dumps(tool) for tool in tools)

        # This comes from the template built into QwQ-32B.
        #
        # Note QwQ-32B and Qwen3 don't actually need this, because they have a template built in;
        # this is for slightly older models that support tool-calling but lack the built-in template,
        # such as DeepSeek-R1-Distill-Qwen-7B.
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

def token_count(settings: env, text: str) -> int:
    """Get number of tokens in `text`, according to the model currently loaded at the LLM backend.

    This is useful for checking how long the prompt is (after you have injected all RAG context etc.).
    """
    # In oobabooga, undocumented web API endpoints can be found at `text-generation-webui/extensions/openai/script.py`
    data = {"text": text}
    response = requests.post(f"{settings.backend_url}/v1/internal/token-count",
                             headers=headers,
                             json=data)
    output_data = response.json()
    return output_data["length"]

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

def linearize_chat(datastore: chattree.Forest, node_id: str) -> List[Dict]:
    """In the chat `datastore`, walking up from `node_id` up to and including a root node, return a linearized representation of that branch.

    This collects the active revision of the data from each node, ignores everything except the chat message data
    (i.e. ignores any metadata added by the chat client, such as RAG retrieval attributions, AI token counts, etc.)
    and puts the messages into a list, in depth order (root node first).

    Note `node_id` doesn't need to be a leaf node; but it will be the last node of the linearized representation;
    children are not scanned.

    NOTE: The difference between this function and `chattree.Forest.linearize_up` is that this will
    extract only the "message" field (OpenAI-compatible chat message record) from each payload, whereas
    that other function returns the full payloads.

    Hence, this is a convenience function for populating a linear chat history for chat clients that use
    the OpenAI format to communicate with the LLM server.
    """
    payload_history = datastore.linearize_up(node_id)  # this auto-selects the active revision of the payload of each node
    message_history = [payload["message"] for payload in payload_history]
    return message_history

# v0.2.3+: data format change
def upgrade(datastore: chattree.Forest, system_prompt_node_id: str) -> None:
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

def factory_reset_chat_datastore(datastore: chattree.Forest, settings: env) -> str:
    """Reset `datastore` to its "factory-default" state.

    **IMPORTANT**: This deletes all existing chat nodes in the datastore, and CANNOT BE UNDONE.

    The primary purpose of this function is to initialize the chat datastore when it hasn't been created yet.

    This creates a root node containing the system prompt (including the character card), and a node for the AI's initial greeting.

    Returns the unique ID of the initial greeting node, so you can start building chats on top of that.

    You can obtain the `settings` object by first calling `setup`.
    """
    datastore.purge()
    root_node_id = datastore.create_node(payload={"message": create_initial_system_message(settings)},
                                         parent_id=None)
    new_chat_node_id = datastore.create_node(payload={"message": create_chat_message(settings,
                                                                                     role="assistant",
                                                                                     text=settings.greeting)},
                                             parent_id=root_node_id)
    return new_chat_node_id

# --------------------------------------------------------------------------------
# stock message formatting itulities

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
                    # Alternatively, in oobabooga, we could call the undocumented "/v1/internal/stop-generation" endpoint.
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
               model=settings.model,
               n_tokens=n_tokens,
               dt=tim.dt)

# --------------------------------------------------------------------------------
# For tool-using LLMs: tool-calling

def perform_tool_calls(settings: env, message: Dict) -> List[Dict]:
    """Perform tool calls as requested in `message["tool_calls"]`.

    Returns a list of messages (each with `role="tool"`) containing the tool outputs,
    one for each tool call.

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
        # TODO: websearch return format: for the chat history, need only the preformatted text, but for the eventual GUI, would be nice to have the links separately. Could use a new metadata field in the chat datastore for this.
        try:
            tool_output_text = function(**kwargs)
        except Exception as exc:
            add_tool_response_message(f"Tool call failed. Function '{function_name}' exited with exception {type(exc)}: {exc}")
            logger.warning(f"perform_tool_calls: function '{function_name}': exited with exception {type(exc)}: {exc}")
        else:  # success!
            logger.debug(f"perform_tool_calls: Function '{function_name}' returned successfully.")
            add_tool_response_message(tool_output_text)

    return tool_response_messages
