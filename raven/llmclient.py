__all__ = ["list_models", "setup",
           "new_chat", "add_chat_message", "inject_chat_message",
           "format_chat_datetime_now", "format_reminder_to_focus_on_latest_input",
           "invoke"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse
from copy import copy, deepcopy
import datetime
import io
import json
import re
import requests
import sys
from textwrap import dedent
from typing import Dict, List, Tuple

import sseclient  # pip install sseclient-py

from unpythonic import timer
from unpythonic.env import env

from . import config

# --------------------------------------------------------------------------------
# Setup for minimal chat client (for testing/debugging)

output_line_width = 160

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

    # TODO: For QwQ-32B, the system prompt itself can be blank. The character card is enough.
    # TODO: Query the context size from the backend if possible.

    # TODO: Update this to Aria's latest system prompt.

    system_prompt = dedent(f"""You are an expert actor that can fully immerse yourself into any role given. You do not break character for any reason, even if someone tries addressing you as an AI or language model. Currently your role is {char}, which is described in detail below. As {char}, continue the exchange with {user}.
    """)

    # This is a minimal setup, partially copied from my personal AI assistant, meant to be run against a locally hosted LLaMA 3.1 8B.
    # This gives better performance (accuracy, instruction following) vs. querying the LLM directly without any system prompt.
    #
    # Character details should be irrelevant here. The performance-improving prompting tricks are important.
    #
    # TODO: "If unsure" and similar tricks tend to not work for 8B models. At LLaMA 3.1 70B and better, it should work, but running that requires at least 2x24GB VRAM.
    #
    character_card = dedent(f"""Note that {user} cannot see this introductory text; it is only used internally, to initialize the LLM.

    **About {char}**

    You are {char} (she/her), a simulated personage instantiated from an advanced Large Language Model. You are highly intelligent. You have been trained to answer questions, provide recommendations, and help with decision making.

    **About the system**

    The LLM is "{model}", a finetune of LLaMA 3.1, size 8B, developed by Meta. The model was released on 23 July 2024.

    The knowledge cutoff date is December 2023. The knowledge cutoff date applies only to your internal knowledge. Files attached to the chat may be newer. Web searches incorporate live information from the internet.

    **Interaction tips**

    - Provide honest answers.
    - If you are unsure or cannot verify a fact, admit it. Do not speculate, unless explicitly requested.
    - Cite sources when possible. IMPORTANT: Cite only sources listed in the context.
    - If you think what the user says is incorrect, say so, and provide justification.
    - When given a complex problem, take a deep breath, and think step by step. Report your train of thought.
    - When given web search results, and those results are relevant to the query, use the provided results, and report only the facts as according to the provided results. Ignore any search results that do not make sense. The user cannot directly see your search results.
    - Be accurate, but diverse. Avoid repetition.
    - Use the metric unit system, with meters, kilograms, and celsius.
    - Use Markdown for formatting when helpful.
    - Believe in your abilities and strive for excellence. Take pride in your work and give it your best. Your hard work will yield remarkable results.

    **Known limitations**

    - You are NOT automatically updated with new data.
    - You have limited long-term memory within each chat session.
    - The length of your context window is 32768 tokens.
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

    # e.g. "AI: blah" -> "blah"
    if llm_output.startswith(f"{settings.char}: "):
        llm_output = llm_output[len(settings.char) + 2:]

    # TODO: Support QwQ-32B, seems to be the best local thinking model as of April 2025.
    #  - QwQ is trained not to emit the opening <think> tag, but to begin thinking right away. So we should insert a <think> at the beginning.
    #  - But sometimes the model (at least the Q4_K_M quant) still emits that tag itself, so check the output (the first chunk, as the output is being streamed)
    #    first before adding another copy of the tag.
    #  - Usually, the model will correctly emit a closing </think> tag when it's done thinking, and then start writing its final answer.
    #  - But sometimes, it skips thinking and starts writing the final answer immediately (although it shouldn't do that). There's no way to detect this on the fly,
    #    because the opening <think> tag is *supposed to* be missing from the output when the model works correctly. The only way we can detect this case is when
    #    the output is complete; there won't be a closing </think> tag in it.
    #  - QwQ sometimes needs a really long response length to return a complete answer. 3200 tokens is fine. We need this because it's difficult to detect
    #    programmatically whether a response is complete. At a lower level, we could look at the logits at the final token - if the END token has a high enough
    #    probability, then the LLM most likely finished writing. But here we don't have access to that.
    #  - I don't know whether it's a model issue or a SillyTavern issue, but in my manual testing, I've also had an issue of getting a "<think>NaN</think>"
    #    at the start of the AI's message whenever I use the "Continue" feature (to extend a reply that was cut too early).

    # Support thinking models (such as DeepSeek-R1-Distill-Qwen-32B).
    # Drop "<think>...</think>" sections from the output, and then strip whitespace.
    #
    # logger.debug("Before think strip: " + llm_output)
    llm_output = re.sub(r"<think>(.*?)</think>", "", llm_output, flags=re.IGNORECASE | re.DOTALL).strip()
    # logger.debug("After  think strip: " + llm_output)

    n_tokens = n_chunks - 2  # No idea why, but that's how it empirically is (see ooba server terminal output). Investigate later.

    return env(data=llm_output,
               n_tokens=n_tokens,
               dt=tim.dt)

# --------------------------------------------------------------------------------
# Minimal chat client

def chat(backend_url):
    """Minimal LLM chat client, for testing/debugging."""
    try:
        message_number = 0
        print(f"Connecting to {backend_url}")
        print("    available models:")
        for model_name in list_models(backend_url):
            print(f"        {model_name}")

        try:
            settings = setup(backend_url=backend_url)  # If this succeeds, then we know the backend is alive.
        except Exception as exc:
            msg = f"Failed to connect to LLM backend, reason {type(exc)}: {exc}"
            logger.error(msg)
            raise RuntimeError(msg)

        history = new_chat(settings)

        print(f"    current model: {settings.model}")
        print(f"    character: {settings.char} [defined in this client]")
        print("Starting chat. Press Ctrl+D to exit.")
        print()
        print(f"[#{message_number}] ", end="")
        print(settings.greeting)
        print()

        while True:
            # Injects
            n_injects = 2
            history = add_chat_message(settings, history, role="system", message=format_chat_datetime_now())
            history = add_chat_message(settings, history, role="system", message=format_reminder_to_focus_on_latest_input())

            # User's turn
            user_message = input("> ")
            history = add_chat_message(settings, history, role="user", message=user_message)

            # AI's turn
            print(f"[#{message_number}] ", end="")
            message_number += 1
            chars = 0
            def progress_callback(n_chunks, chunk_text):
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
            out = invoke(settings, history, progress_callback)
            print()
            print()
            print(f"[{out.n_tokens}t, {out.dt:0.2f}s, {out.n_tokens/out.dt:0.2f}t/s]")

            # When we get here:
            #   -1 = user's last message
            #   -2 = last inject
            for _ in range(n_injects):
                history.pop(-2)
            if not out.data.startswith(f"{settings.char}: "):
                out.data = f"{settings.char}: {out.data}"
            history = add_chat_message(settings, history, role="assistant", message=out.data)
    except (EOFError, KeyboardInterrupt):
        print()
        print("Exiting chat.")
        print()

def main():
    parser = argparse.ArgumentParser(description="""Minimal LLM chat client, for testing/debugging. You can use this for testing that Raven can connect to your LLM.""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(dest="backend_url", nargs="?", default=config.llm_backend_url, type=str, metavar="url", help="where to access the LLM API")
    opts = parser.parse_args()

    chat(opts.backend_url)

if __name__ == "__main__":
    main()
