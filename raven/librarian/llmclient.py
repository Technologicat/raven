"""LLM client low-level library functions for Raven.

See `raven.librarian.scaffold` for the higher-level scaffolding that goes on top of this,
e.g. automatically applying tool-calls.

For an example chat client built using these, see `raven.librarian.minichat`.

NOTE for oobabooga/text-generation-webui users:

If you want to see the final prompt in instruct or chat mode, start your server in `--verbose` mode.
"""

__all__ = ["list_models",
           "setup",
           "token_count",
           "invoke", "action_ack", "action_stop",
           "perform_tool_calls"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import collections
import copy
import io
import json
import os
import requests
from textwrap import dedent
from typing import Dict, List, Optional

import sseclient  # pip install sseclient-py

from mcpyrate import colorizer

from unpythonic import sym, timer
from unpythonic.env import env

from ..client import api
from ..client import config as client_config

from . import chatutil
from . import config as librarian_config

action_ack = sym("ack")  # acknowledge LLM progress, keep generating
action_stop = sym("stop")  # interrupt the LLM, stop generating now

# --------------------------------------------------------------------------------
# Module bootup

api.initialize(raven_server_url=client_config.raven_server_url,
               raven_api_key_file=client_config.raven_api_key_file,
               tts_server_type=client_config.tts_server_type,
               tts_url=client_config.tts_url,
               tts_api_key_file=client_config.tts_api_key_file,
               tts_playback_audio_device=client_config.tts_playback_audio_device)  # let it create a default executor

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

def websearch_wrapper(query: str,
                      engine: str = "duckduckgo",
                      max_links: int = 10) -> str:
    """Perform a websearch, using Raven-server to handle the interaction with the search engine and the parsing of the SERP (search engine results page)."""
    # TODO: The ANSI coloring isn't useful in `websearch_wrapper`, because its output goes to the LLM. We should separate data/presentation if we want to do this. Currently, should always use markdown, which the LLM also understands.
    markup = "markdown"
    chatutil._yell_if_unsupported_markup(markup)

    websearch_results = api.websearch_search(query, engine, max_links)  # -> {"results": preformatted_text, "data": structured_results}
    structured_results = websearch_results["data"]

    def highlight(text: str) -> str:
        if markup == "ansi":
            text = colorizer.colorize(text, colorizer.Style.BRIGHT)
        elif markup == "markdown":
            text = f"**{text}**"
        return text

    def format_link(url: str) -> str:
        if markup == "ansi":
            url = colorizer.colorize(url, colorizer.Style.BRIGHT, colorizer.Fore.BLUE)
        elif markup == "markdown":
            url = f"[{url}]({url})"
        return url

    # See also `raven.server.modules.websearch`, which has a version of this without markup.
    def format_result(result: dict) -> str:
        if "title" in result and "link" in result:
            heading = f"{highlight('Web result from')}: {format_link(result['link'])}\n{highlight(result['title'])}"
        elif "title" in result:
            heading = highlight(result["title"])
        elif "link" in result:
            heading = f"{highlight('Web result from')}: {format_link(result['link'])}"
        else:
            return f"{result['text']}\n"
        return f"{heading}\n\n{result['text']}\n"

    formatted_text = "\n\n".join(format_result(result) for result in structured_results)
    return formatted_text  # TODO: our LLM scaffolding doesn't currently accept anything else but preformatted text

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
    """Connect to LLM at `backend_url`.

    Return an `env` object (a fancy namespace) populated with the following fields:

        `user`: Name of user's character.
        `char`: Name of AI assistant.
        `model`: Name of model running at `backend_url`, queried automatically from the backend.
        `system_prompt`: Generic system prompt for the LLM (this is the LLaMA 3 preset from SillyTavern), to make it follow the character card.
        `character_card`: Character card that configures the AI assistant to improve the model's performance.
        `greeting`: The AI assistant's first message, used later for initializing the chat history.
        `prompts`: Prompts for the LLM, BibTeX field processing functions, and any literal info to fill in the output BibTeX. See the main program for details.
        `request_data`: Generation settings for the LLM backend.
        `role_names`: A `dict` with keys "user", "assistant", "system", used for constructing chat messages (see `raven.librarian.chatutil.create_chat_message`).
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
    #
    # For the sampler settings, below are some sensible defaults.
    # But for best results, prefer using the values recommended in your LLM's model card, if known.
    # E.g. Qwen3-30B-A3B-Thinking-2507 was tuned for T = 0.6, top_k = 20, top_p = 0.95, min_p = 0.
    request_data = {
        "mode": "instruct",  # instruct mode: when invoking the LLM, send it instructions (system prompt and character card), followed by a chat transcript to continue.
        "max_tokens": 6400,  # 800 is usually good, but thinking models may need (much) more. For them, 1600 or 3200 are good. 6400 if you want to be sure.
        # Correct sampler order is tail-cutters (such as top_k, top_p, min_p) first, then temperature. In oobabooga, this is also the default.
        #
        # T = 1: Use the predicted logits as-is.
        # T = 0: Greedy decoding, i.e. always pick the most likely token. Prone to getting stuck in a loop. For fact extraction (for some models).
        # T > 1: Skew logits to emphasize rare continuations ("creative mode").
        # 0 < T < 1: Skew logits to emphasize common continuations.
        #
        # Usually T = 1 is a good default; but a particular LLM may have been tuned to use some other value, e.g. 0.7 or 0.6. See the model card of your LLM.
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

    # See `raven.librarian.chatutil.create_chat_message`.
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

# --------------------------------------------------------------------------------
# The most important function - call LLM, parse result

def invoke(settings: env, history: List[Dict], on_progress=None) -> env:
    """Invoke the LLM with the given chat history.

    This is typically done after adding the user's message to the chat history, to ask the LLM to generate a reply.

    `settings`: Obtain this by calling `setup()` at app start time.

    `history`: List of chat messages, see `raven.librarian.chatutil.create_chat_message`.

    `on_progress`: 2-argument callable, optional, with arguments `(n_chunks: int, chunk_text: str)`.
                   Called while streaming the response from the LLM, typically once per generated token.

           `n_chunks: int`: How many chunks have been generated so far, for this invocation.
                            Useful for live UI updates.

           `chunk_text: str`: The text of the current chunk (typically a token).

           Return value: `action_ack` to let the LLM keep generating, `action_stop` to interrupt and finish forcibly.

           If you interrupt the LLM by returning `action_stop`, normal finalization still takes place, and you'll get
           a chat message populated with the content received so far. It is up to the caller what to do with that data.

    Returns an `unpythonic.env.env` WITHOUT adding the LLM's reply to `history`.

    The returned `env` has the following attributes:

        `data: dict`: The new message generated by the LLM (for the format, see `raven.librarian.chatutil.create_chat_message`).
                      If the text content begins with the assistant character's name (e.g. "AI: ..."), this is automatically stripped.
        `n_tokens: int`: Number of tokens emitted by the LLM.
        `dt: float`: Wall time elapsed for this invocation, in seconds.
        `interrupted: bool`: Whether the invocation was interrupted by the `on_progress` callback.
                             This is provided for convenience.
    """
    data = copy.deepcopy(settings.request_data)
    data["messages"] = history
    stream_response = requests.post(f"{settings.backend_url}/v1/chat/completions", headers=headers, json=data, verify=False, stream=True)

    if stream_response.status_code != 200:  # not "200 OK"?
        logger.error(f"LLM server returned error: {stream_response.status_code} {stream_response.reason}. Content of error response follows.")
        logger.error(stream_response.text)
        raise RuntimeError(f"While calling LLM: HTTP {stream_response.status_code} {stream_response.reason}")

    client = sseclient.SSEClient(stream_response)
    def stop_generating():
        # The local LLM is OpenAI-compatible, so the same trick works - to tell the server to stop, just close the stream.
        # https://community.openai.com/t/interrupting-completion-stream-in-python/30628/7
        # Alternatively, in oobabooga, we could call the undocumented "/v1/internal/stop-generation" endpoint.
        client.close()

    llm_output_text = io.StringIO()
    last_few_chunks = collections.deque([""] * 10)  # ring buffer for quickly checking a short amount of text at the current end; prepopulate with empty strings since `popleft` requires at least one element to be present
    n_chunks = 0
    stopped = False  # whether one of the stop strings triggered
    interrupted = False  # whether the progress callback interrupted generation
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

                action = action_ack
                if on_progress is not None:
                    action = on_progress(n_chunks, chunk)

                if any(stop):  # should stop due to a stopping string?
                    stop_generating()
                    stopped = True
                    break
                if action is action_stop:  # did the callback tell us to interrupt the LLM generation?
                    stop_generating()
                    interrupted = True
                    break
    except KeyboardInterrupt:  # on Ctrl+C, stop generating, and let the exception propagate
        stop_generating()
        raise
    except requests.exceptions.ChunkedEncodingError as exc:
        logger.error(f"invoke: Connection lost. Please check if your LLM backend is still alive (was at {settings.backend_url}). Original error message follows.")
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
    else:  # normal finish by LLM server, or callback interrupt
        if "tool_calls" in delta:
            tool_calls = delta["tool_calls"]

    # In streaming mode, the oobabooga backend always yields an initial chunk with empty content (perhaps to indicate that the connection was successful?)
    # and at the end, a final chunk with empty content, containing usage stats and possible tool calls.
    #
    # The correct way to get the number of tokens is to read the "usage" field of the final chunk.
    # Of course, if we have to close the connection on our end due to a stopping string, or if the callback tells us to stop, we won't get that chunk.
    # But this hack always works.
    n_tokens = n_chunks - 2

    message = chatutil.create_chat_message(llm_settings=settings,
                                           role="assistant",
                                           text=llm_output_text,
                                           add_role_name=False,
                                           tool_calls=tool_calls)
    return env(data=message,
               model=settings.model,
               n_tokens=n_tokens,
               dt=tim.dt,
               interrupted=interrupted)

# --------------------------------------------------------------------------------
# For tool-using LLMs: tool-calling

def perform_tool_calls(settings: env, message: Dict) -> List[env]:
    """Perform tool calls as requested in `message["tool_calls"]`.

    Returns a list of chat payloads (where each message's `role="tool"`) containing the tool outputs,
    one for each tool call.

    If the `tool_calls` field of the message is missing or if it is empty, return the empty list.

    Each returned `env` has the following attributes:

        `data`: dict, The new message containing the tool response (for the format, see `raven.librarian.chatutil.create_chat_message`).

        `status`: str, one of "success" or "error".

            When an error occurs, the text of the output message will describe the error instead,
            and the full error message is posted to the server's log at warning level.

            Even if a tool call errors out, processing continues with the remaining tool calls, if any.

        `toolcall_id`: str. The ID of the tool call, copied from the input `message`.
                       Missing if no ID was provided.

        `dt`: float, Wall time elapsed for the call, in seconds.
              Missing if something went wrong before the tool was called (usually, bad input).

    Usually the input `message` looks something like this::

        message = {'role': 'assistant',
                   'content': '',
                   'tool_calls': [{'type': 'function',
                                   'function': {'name': 'websearch',
                                                'arguments': '{"query": "Sharon Apple"}'},
                                   'id': 'call_m357947b',
                                   'index': '0'}],
                  }
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

    tool_response_records = []
    def add_tool_response_record(text: str, *,
                                 status: str,
                                 toolcall_id: Optional[str],
                                 dt: Optional[float]) -> None:
        """Add a tool response record to `tool_response_records`.

        The record is an `unpythonic.env.env` with the following attributes:

            `data`: dict: chat message object, with `role="tool"`, and `content=text`.

            `status`: str: Values "success" or "error" are recommended.

            `toolcall_id`: Optional[str]: ID of this tool call (can be matched against the `id` in the `tool_calls` list
                           of the AI chat message that spawned this call).

                           The ID should be included whenever it was present in the tool call request record.

            `dt`: Optional[float]: Duration of this tool call, in seconds. Recommended to be included whenever
                                   the request was valid enough to actually proceed to call the function
                                   (so that the call timing can be measured).
        """
        tool_response_message = chatutil.create_chat_message(llm_settings=settings,
                                                             role="tool",
                                                             text=text,
                                                             add_role_name=False)
        record = env(data=tool_response_message,
                     status=status)
        if toolcall_id is not None:
            record.toolcall_id = toolcall_id
        if dt is not None:
            record.dt = dt
        tool_response_records.append(record)

    for request_record in tool_calls:
        toolcall_id = request_record["id"] if "id" in request_record else None

        if "type" not in request_record:
            # The response message is intended for the LLM, whereas the log message (with all technical details) goes into the log.
            logger.warning(f"perform_tool_calls: {toolcall_id}: missing 'type' field in request. Data: {request_record}")
            add_tool_response_record("Tool call failed. The request is missing the 'type' field.", status="error", toolcall_id=toolcall_id)
            continue
        if request_record["type"] != "function":
            logger.warning(f"perform_tool_calls: {toolcall_id}: unknown type '{request_record['type']}' in request, expected 'function'. Data: {request_record}")
            add_tool_response_record(f"Tool call failed. Unknown request type '{request_record['type']}'; expected 'function'.", status="error", toolcall_id=toolcall_id)
            continue
        if "function" not in request_record:
            logger.warning(f"perform_tool_calls: {toolcall_id}: missing 'function' field. Data: {request_record}")
            add_tool_response_record("Tool call failed. The request is missing the 'function' field.", status="error", toolcall_id=toolcall_id)
            continue

        function_record = request_record["function"]
        if "name" not in function_record:
            logger.warning(f"perform_tool_calls: {toolcall_id}: missing 'function.name' field in request. Data: {request_record}")
            add_tool_response_record("Tool call failed. The request's function record is missing the 'name' field.", status="error", toolcall_id=toolcall_id)
            continue

        function_name = function_record["name"]
        try:
            function = settings.tool_entrypoints[function_name]
        except KeyError:
            logger.warning(f"perform_tool_calls: {toolcall_id}: unknown function '{function_name}'.")
            add_tool_response_record(f"Tool call failed. Function not found: '{function_name}'.", status="error", toolcall_id=toolcall_id)
            continue

        if "arguments" in function_record:
            try:
                kwargs = json.loads(function_record["arguments"])
            except Exception as exc:
                logger.warning(f"perform_tool_calls: {toolcall_id}: function '{function_name}': failed to parse JSON for arguments: {type(exc)}: {exc}")
                add_tool_response_record(f"Tool call failed. When calling '{function_name}', failed to parse the request's JSON for the function arguments.", status="error", toolcall_id=toolcall_id)
                continue
            else:
                logger.debug(f"perform_tool_calls: {toolcall_id}: calling '{function_name}' with arguments {kwargs}.")
        else:
            logger.debug(f"perform_tool_calls: {toolcall_id}: for function '{function_name}: The request's function record is missing the 'arguments' field. Calling without arguments.")
            kwargs = {}

        # TODO: websearch return format: for the chat history, need only the preformatted text, but for the eventual GUI, would be nice to have the links separately. Could use a new metadata field in the chat datastore for this.
        try:
            with timer() as tim:
                tool_output_text = function(**kwargs)
        except Exception as exc:
            logger.warning(f"perform_tool_calls: {toolcall_id}: function '{function_name}': exited with exception {type(exc)}: {exc}")
            add_tool_response_record(f"Tool call failed. Function '{function_name}' exited with exception {type(exc)}: {exc}", status="error", toolcall_id=toolcall_id, dt=tim.dt)
        else:  # success!
            logger.debug(f"perform_tool_calls: {toolcall_id}: Function '{function_name}' returned successfully.")
            add_tool_response_record(tool_output_text, status="success", toolcall_id=toolcall_id, dt=tim.dt)

    return tool_response_records
