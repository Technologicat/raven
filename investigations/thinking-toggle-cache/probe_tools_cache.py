"""Does flipping a tool toggle throw away the KV cache, and how much of it?

The sibling question to `probe_toggle_cache.py`, and the same apparatus. Librarian's **Internet** and
**Documents** checkboxes change which tools are declared on the next turn, and a chat template renders the
declarations somewhere in the prompt. Where it puts them decides the cost, exactly as the thinking marker's
position did:

  - Declared in the **first system turn** — the usual place — a changed tool set invalidates the prompt from
    token zero, so the switch should cost a full re-prefill of the whole conversation.
  - Declared anywhere later, or not rendered into the prompt at all, and prefix matching survives.

**Time, not `prompt_tokens`**, for the reason the sibling probe records: cache-relative reporting does not
engage at this scale, so the readout is how long the backend takes before it can answer.

Sequence: A A B B A, where A declares every tool and B declares only the ones that reach nothing outside the
process — which is exactly what switching **Internet** and **Documents** both off does. The repeats
establish what a warm hit costs for each, so the *switch* timings are read against a measured baseline.

`max_tokens=1`, so this times prompt handling and not generation.

Run it against the backend Librarian is configured for:

    python probe_tools_cache.py http://localhost:1234
"""

import sys
import time

import requests

from raven.librarian import llmtools

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:1234"

# Big enough that a full re-prefill is unmistakable against a cache hit. The content is filler on purpose:
# what is measured is the prompt's length and its prefix, not what the model makes of it.
_FILLER = ("Consider a long-running experiment in which a sequence of measurements is recorded, each with "
           "its own uncertainty, and the results are later aggregated into a single estimate. ")
HISTORY = [{"role": "user", "content": "Here are some notes. " + _FILLER * 190},
           {"role": "assistant", "content": "Noted. " + _FILLER * 130},
           {"role": "user", "content": "Summarize the notes in one sentence."}]

_GATED = llmtools.NETWORK_TOOL_NAMES | llmtools.DOCUMENT_TOOL_NAMES
ALL_TOOLS = llmtools.TOOLS
LOCAL_TOOLS = [tool for tool in llmtools.TOOLS if tool["function"]["name"] not in _GATED]
assert LOCAL_TOOLS, "no ungated tools left; this probe cannot tell the two tool sets apart"
assert len(LOCAL_TOOLS) < len(ALL_TOOLS), "the two tool sets are identical; nothing would be switched"


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


print(f"{len(ALL_TOOLS)} tools declared in A, {len(LOCAL_TOOLS)} in B\n")
A = {"tools": ALL_TOOLS}
B = {"tools": LOCAL_TOOLS}
for label, extra in (("A1  all tools     (cold)          ", A),
                     ("A2  all tools     (warm baseline) ", A),
                     ("B1  local only    (THE SWITCH)    ", B),
                     ("B2  local only    (warm baseline) ", B),
                     ("A3  all tools     (switch back)   ", A)):
    ask(label, extra)
