"""Does flipping the thinking toggle throw away the KV cache, and how much of it?

Where each family's thinking marker sits in the rendered prompt predicts the answer:

  - Qwen puts it at the *generation prompt*, i.e. the tail. Toggling should invalidate only the last few
    tokens, so a toggled request still lands on a warm cache.
  - Gemma puts `<|think|>` at the top of the *first system turn*. Toggling should invalidate from the very
    first token, so a toggled request re-processes the whole conversation.

**Time, not `prompt_tokens`.** The cache-relative reporting that would have answered this directly does not
engage at this scale — measured 2026-08-26, a 682-token prompt reports 682 warm or cold — so the readout is
how long the backend takes before it can answer. A warm cache is quick; a discarded one is a full
re-prefill and shows as seconds.

Sequence: A A B B A, where A is thinking-on and B is `reasoning_effort: "none"`. The repeats establish what
a warm hit costs for each, so the *switch* timings are read against a measured baseline rather than a guess.

`max_tokens=1`, so this times prompt handling and not generation.
"""

import sys
import time

import requests

URL = sys.argv[1] if len(sys.argv) > 1 else "http://maia.local:1234"

# Big enough that a full re-prefill is unmistakable against a cache hit. The content is filler on purpose:
# what is measured is the prompt's length and its prefix, not what the model makes of it.
_FILLER = ("Consider a long-running experiment in which a sequence of measurements is recorded, each with "
           "its own uncertainty, and the results are later aggregated into a single estimate. ")
HISTORY = [{"role": "user", "content": "Here are some notes. " + _FILLER * 190},
           {"role": "assistant", "content": "Noted. " + _FILLER * 130},
           {"role": "user", "content": "Summarize the notes in one sentence."}]


def ask(label: str, extra: dict) -> None:
    body = {"messages": HISTORY, "stream": False, "max_tokens": 1, **extra}
    t0 = time.perf_counter()
    r = requests.post(f"{URL}/v1/chat/completions", json=body, timeout=900)
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        print(f"{label}: HTTP {r.status_code}: {r.text[:200]}")
        return
    usage = r.json().get("usage") or {}
    print(f"{label}: {dt:6.2f} s   prompt_tokens={usage.get('prompt_tokens')}")


ON: dict = {}
OFF = {"reasoning_effort": "none"}
for label, extra in (("A1  thinking on           (cold)          ", ON),
                     ("A2  thinking on           (warm baseline) ", ON),
                     ("B1  reasoning_effort=none (THE SWITCH)    ", OFF),
                     ("B2  reasoning_effort=none (warm baseline) ", OFF),
                     ("A3  thinking on           (switch back)   ", ON)):
    ask(label, extra)
