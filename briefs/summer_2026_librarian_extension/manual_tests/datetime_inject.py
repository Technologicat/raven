#!/usr/bin/env python3
"""Manual live probe: can we tell the model what day it is, and will it believe us?

NOT a pytest test — it needs a running backend with a model loaded, so it lives here
under `briefs/` rather than in the suite. Stdlib only, so it can be piped over ssh to
whichever machine has the models:

    ssh host 'python3 - http://localhost:1234 <model>' < datetime_inject.py

Raven injects the current date and time every AI turn. Unlike the two behavioural
reminders, this one is **data, not instruction** — the model only has to read it — and
unlike the RAG matches it **changes every single turn**, so it can never be part of a
stable cached prefix. That combination is what makes its shape a separate question.

There is also a known behavioural problem that the shape may or may not fix. Qwen models
carry a strong prior that it is still spring 2024 and will argue with a date that
contradicts it — observed 2026-07-19 rationalizing the conflict by concluding it must be
running in a test environment, "since 2026 hasn't happened yet". Whether delivering the
date as tool output (which is what it honestly is: the answer a clock would give) changes
that, rather than merely changing where it sits, is the thing worth measuring.

Two questions per shape, because parroting a date back is easier than using it:

  - "What is today's date?" — is the injected date repeated, or disputed?
  - "How many days until <a date ~2 months out>?" — the model has to *act* on it. A model
    that recites the date and then counts from its training-data prior has not accepted it.

Usage:
    python datetime_inject.py                       # localhost:1234, first model
    python datetime_inject.py <base_url> [model]
"""

import datetime
import json
import re
import sys
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE = "http://localhost:1234"
TIMEOUT = 600

SYSTEM_PROMPT = "You are Aria, a helpful research assistant. Answer the user's questions accurately and concisely."

# Mirrors `chatutil.format_chat_datetime_now`. Duplicated rather than imported so this stays
# stdlib-only and can be piped to a machine with no venv; keep the wording in step with it.
NOW = datetime.datetime.now()
WEEKDAY = NOW.strftime("%A")
ISODATE = NOW.strftime("%Y-%m-%d")
ISOTIME = NOW.strftime("%H:%M:%S")
DATETIME_INJECT = (f"[System information: Today is {WEEKDAY}, {ISODATE} (in ISO format). "
                   f"The local time now is {ISOTIME}.]")

# For the `split` shape. The date is constant for a whole day, so it can live in the leading system
# block without costing the KV-cache prefix anything — the prefix only changes at midnight, which the
# app can watch for and patch. Only the clock time has to arrive per-turn.
DATE_INJECT = f"[System information: Today is {WEEKDAY}, {ISODATE} (in ISO format).]"
TIME_INJECT = f"[System information: The local time now is {ISOTIME}.]"

# The target is far enough out that the answer is unmistakably a computation, and close enough
# that a model reasoning from a 2024 prior gets a wildly different number rather than a near miss.
TARGET = NOW.date() + datetime.timedelta(days=60)
DAYS_AWAY = (TARGET - NOW.date()).days

QUESTIONS = {"recite": "What is today's date?",
             "compute": f"How many days are there from today until {TARGET.isoformat()}? "
                        f"Answer with the number of days."}

SHAPES = ("none", "user", "tool", "tool+call", "system_front", "split")

# Markers of the model rejecting the supplied date rather than using it. "2024" and "2025" catch
# the training-prior answer directly; the rest catch the hedge that precedes it.
DISPUTE = ("2024", "2025", "training data", "last update", "knowledge cutoff", "cutoff",
           "cannot know", "can't know", "don't have access to real-time", "do not have access",
           "simulation", "hypothetical", "fictional", "future date", "as an ai")


def post(base: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(f"{base}/v1/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"_error": f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:160]}"}
    except Exception as e:  # noqa: BLE001 -- a probe reports failures rather than raising
        return {"_error": f"{type(e).__name__}: {e}"}


def build(shape: str, question: str) -> list[dict]:
    """Place the datetime inject around `question` according to `shape`.

    The inject text is identical in every shape, so the only variable is role and position.
    (As tool output the "[System information: ...]" wrapper is arguably redundant — but
    changing two things at once is how the previous attempt at this failed to teach anybody
    anything, so it stays.)
    """
    system_text, extra = SYSTEM_PROMPT, []
    if shape == "none":
        pass
    elif shape == "system_front":
        system_text = SYSTEM_PROMPT + "\n\n" + DATETIME_INJECT
    elif shape == "user":
        extra = [{"role": "user", "content": DATETIME_INJECT}]
    elif shape == "tool":
        extra = [{"role": "tool", "content": DATETIME_INJECT}]
    elif shape == "tool+call":
        extra = [{"role": "assistant", "content": "",
                  "tool_calls": [{"id": "call_clock", "type": "function",
                                  "function": {"name": "get_current_datetime", "arguments": "{}"}}]},
                 {"role": "tool", "tool_call_id": "call_clock", "content": DATETIME_INJECT}]
    elif shape == "split":
        # Date in the stable system block, clock time as tool output. Buys system-level placement for
        # the part that provokes the least argument, while the per-turn part stays small.
        system_text = SYSTEM_PROMPT + "\n\n" + DATE_INJECT
        extra = [{"role": "assistant", "content": "",
                  "tool_calls": [{"id": "call_clock", "type": "function",
                                  "function": {"name": "get_current_time", "arguments": "{}"}}]},
                 {"role": "tool", "tool_call_id": "call_clock", "content": TIME_INJECT}]
    else:
        raise ValueError(f"unknown shape {shape!r}; expected one of {SHAPES}")

    # The inject goes *before* the user's question: measured as the safe placement for tool-role
    # material, since a trailing tool result invites the model to emit another tool call instead
    # of an answer. See briefs/context-inject-shape-measurements.md.
    return [{"role": "system", "content": system_text}, *extra, {"role": "user", "content": question}]


def ask(base: str, model: str, messages: list[dict]) -> dict[str, Any]:
    body = post(base, {"model": model, "messages": messages, "max_tokens": 1500, "temperature": 0.0})
    if "_error" in body:
        return {"error": body["_error"], "content": "", "reasoning": "", "finish": "error"}
    choice = body.get("choices", [{}])[0]
    msg = choice.get("message", {})
    return {"error": None,
            "content": (msg.get("content") or "").strip(),
            "reasoning": (msg.get("reasoning_content") or msg.get("reasoning") or "").strip(),
            "finish": choice.get("finish_reason") or "?"}


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE
    model = sys.argv[2] if len(sys.argv) > 2 else None
    if model is None:
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=TIMEOUT) as r:
                ids = [m["id"] for m in json.loads(r.read().decode("utf-8")).get("data", [])]
        except Exception as e:  # noqa: BLE001
            print(f"cannot reach {base}: {type(e).__name__}: {e}")
            return
        if not ids:
            print(f"{base} reports no models loaded")
            return
        model = ids[0]

    print(f"probing {base} with model {model!r}")
    print(f"injecting: {DATETIME_INJECT}")
    print(f"expecting: date {ISODATE}, and {DAYS_AWAY} days until {TARGET.isoformat()}\n")

    for shape in SHAPES:
        print(f"  --- datetime as {shape} ---")
        for kind, question in QUESTIONS.items():
            got = ask(base, model, build(shape, question))
            if got["error"]:
                print(f"    {kind:<8} REJECTED -- {got['error'][:90]}")
                continue
            blob = f"{got['content']}\n{got['reasoning']}".lower()
            if kind == "recite":
                ok = ISODATE in got["content"] or NOW.strftime("%B %-d, %Y").lower() in blob
            else:
                # Accept the number alone or with the word "days"; a model reasoning from its
                # training prior lands tens or hundreds of days away, not one or two.
                ok = bool(re.search(rf"\b{DAYS_AWAY}\b", got["content"]))
            disputes = sorted({d for d in DISPUTE if d in blob})
            verdict = "ok" if ok else "WRONG"
            if disputes:
                verdict += f"  (disputes: {', '.join(disputes[:3])})"
            print(f"    {kind:<8} {verdict:<44} finish={got['finish']} "
                  f"reasoning={len(got['reasoning'])}ch")
            print(f"             reply: {' '.join(got['content'].split())[:110]!r}")
            if not ok and got["reasoning"]:
                # The failing case is exactly the one whose reasoning is worth reading — how the
                # model talked itself out of the answer. Show the tail, where it lands on a
                # conclusion, rather than the opening restatement of the question.
                print(f"             reasoning tail: {' '.join(got['reasoning'].split())[-400:]!r}")
    print("\ndone")


if __name__ == "__main__":
    main()
