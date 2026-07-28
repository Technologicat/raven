#!/usr/bin/env python3
"""Manual live probe: what shape should Librarian's temporary context injects take?

NOT a pytest test — it needs a running backend with a model loaded, so it lives here
under `briefs/` rather than in the suite.

Each AI turn, `scaffold._perform_injects` adds material the user never typed: the current
date and time, two behavioural reminders, and one message per RAG match. The open question
is not *whether* to send them but in which **role** and at which **position**, and the
candidate answers trade against each other in ways that argue equally well on paper:

  user-role at the end   what we ship today. Every template accepts it, but the model sees
                         machine-looking text as the user's words — and "reply to the user's
                         most recent message" then refers to itself.
  tool-role at the end    reads as tool output, which is what a RAG match is. On Qwen the
                         template also excludes tool messages from its last-user-query scan,
                         so the real question keeps its place.
  folded into the user's message   no extra turns at all, so nothing for a template guard to
                         object to — but the text is still the user's, so narration may stay.
  merged into the leading system message   restores system-level authority, at the cost of
                         recency and of the KV cache: the leading block is the prompt's prefix,
                         so rewriting it per turn invalidates everything after it.
  system-role at the end  the original design. Included as a control: strict templates
                         (Qwen3.5) permit exactly one system message and only as the first.

The probes below measure what those paragraphs assert. Each reports a mechanical read
(did the answer contain the needle, did the request survive) *and* prints the model's own
words, because the failures being hunted here — the model narrating the injects, or spending
its reasoning budget negotiating a reminder — are visible in the text and not in a token count.

Usage:
    python inject_shapes.py                       # localhost:1234, first model
    python inject_shapes.py <base_url> [model]
"""

import json
import sys
import urllib.error
import urllib.request
from typing import Any

from raven.librarian import chatutil

DEFAULT_BASE = "http://localhost:1234"
TIMEOUT = 300

# Prefilling this reproduces Qwen's own non-thinking mode; `chat_template_kwargs` is ignored
# by LM Studio, so prefill is the only thinking toggle available on the OpenAI-compatible
# endpoint. Used where a probe wants the answer rather than the deliberation.
CLOSED_THINK = "<think>\n\n</think>\n\n"

SYSTEM_PROMPT = ("You are Aria, a helpful research assistant. Answer the user's questions "
                 "accurately and concisely.")

# The real thing, not a paraphrase — these are what Raven puts on the wire every turn.
DATETIME_INJECT = chatutil.format_chat_datetime_now()
FOCUS_INJECT = chatutil.format_reminder_to_focus_on_latest_input()
CONTEXT_ONLY_INJECT = chatutil.format_reminder_to_use_information_from_context_only()
ALWAYS_ON_INJECTS = [DATETIME_INJECT, FOCUS_INJECT, CONTEXT_ONLY_INJECT]

# A fact no model can hold, so an answer containing it proves the material was read rather
# than recalled. Deliberately dull — nothing here invites the model to reason around it.
NEEDLE_FACT = ("The Kuiper-7 sensor array reports a baseline drift of 4.2 millikelvin per hour "
               "under nominal load.")
NEEDLE_QUESTION = "What baseline drift does the Kuiper-7 sensor array report?"
NEEDLE = "4.2"

SHAPES = ("user", "tool", "folded", "system_front", "system_end")

# Thinking probes need headroom, or the reasoning eats the whole budget and the empty answer
# reads as a refusal when it was a truncation. The current context-only reminder alone spends
# ~5000 characters deliberating, so leave several times that.
THINK_BUDGET = 3000


def post(base: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST JSON, returning the parsed body or a dict with an `_error` key."""
    req = urllib.request.Request(f"{base}{path}",
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


def ask(base: str, model: str, messages: list[dict], think: bool = False,
        max_tokens: int = 500) -> dict[str, str]:
    """Send `messages`, returning `content` and `reasoning` as plain strings.

    With `think=False`, a closed-thought prefill suppresses the reasoning phase, so the whole
    token budget goes to the answer. An `_error` key survives into `content`.
    """
    wire = list(messages)
    if not think:
        wire = wire + [{"role": "assistant", "content": CLOSED_THINK}]
    body = post(base, "/v1/chat/completions",
                {"model": model, "messages": wire, "max_tokens": max_tokens, "temperature": 0.0})
    if "_error" in body:
        return {"content": body["_error"], "reasoning": ""}
    msg = body.get("choices", [{}])[0].get("message", {})
    return {"content": (msg.get("content") or "").strip(),
            "reasoning": (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()}


def build(shape: str, user_text: str, injects: list[str]) -> list[dict]:
    """Build a wire history placing `injects` around `user_text` according to `shape`.

    This mirrors what `scaffold._perform_injects` does, with the role and position as the
    free parameters. See the module docstring for what each shape is arguing for.
    """
    system_text = SYSTEM_PROMPT
    trailing: list[dict] = []
    user_content = user_text

    if shape == "system_front":
        system_text = SYSTEM_PROMPT + "\n\n" + "\n".join(injects)
    elif shape == "folded":
        user_content = user_text + "\n\n" + "\n".join(injects)
    elif shape in ("user", "tool", "system_end"):
        role = "system" if shape == "system_end" else shape
        trailing = [{"role": role, "content": text} for text in injects]
    else:
        raise ValueError(f"unknown shape {shape!r}; expected one of {SHAPES}")

    return [{"role": "system", "content": system_text},
            {"role": "user", "content": user_content},
            *trailing]


def narrates(text: str) -> bool:
    """Heuristic: does the reply talk *about* the injected notes rather than using them?

    Catches the observed failure ("Got it, this seems to be a system test. I've received your
    messages and the system information provided"). Only a hint — the printed text decides.
    """
    lowered = text.lower()
    return any(phrase in lowered for phrase in ("system information", "system test",
                                                "your instructions", "the instructions",
                                                "these notes", "system note"))


def report(label: str, verdict: str, detail: str = "") -> None:
    print(f"  {label:<22} {verdict}")
    if detail:
        print(f"      {detail}")


def show(field: str, text: str, limit: int = 220) -> str:
    flat = " ".join(text.split())
    return f"{field}: {flat[:limit]!r}" if flat else f"{field}: (empty)"


# --------------------------------------------------------------------------------

def probe_tool_role_accepted(base: str, model: str) -> None:
    print("\n[1] Is a standalone tool message accepted, with no assistant tool_call before it?")
    print("    Raven's RAG matches are not answers to a call the model made, so if the backend or")
    print("    template insists on the pairing, the tool role is off the table before anything else.")
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": NEEDLE_QUESTION},
                {"role": "tool", "content": NEEDLE_FACT}]
    got = ask(base, model, messages)
    if got["content"].startswith("HTTP "):
        report("standalone tool", f"REJECTED -- {got['content']}")
        return
    used = NEEDLE in got["content"]
    report("standalone tool", "accepted, material used" if used else "accepted, material NOT used",
           show("reply", got["content"]))

    print("    And with a tool_call_id, which the OpenAI schema normally requires:")
    paired = [{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": NEEDLE_QUESTION},
              {"role": "assistant", "content": "",
               "tool_calls": [{"id": "call_probe", "type": "function",
                               "function": {"name": "search_documents",
                                            "arguments": json.dumps({"query": "Kuiper-7 drift"})}}]},
              {"role": "tool", "tool_call_id": "call_probe", "content": NEEDLE_FACT}]
    got = ask(base, model, paired)
    used = NEEDLE in got["content"]
    report("with synthetic call", "accepted, material used" if used else got["content"][:70],
           show("reply", got["content"]))


def probe_last_question_survives(base: str, model: str) -> None:
    print("\n[2] With injects sitting after the user's question, is the question still answered?")
    print("    The self-reference wart: user-role injects become the most recent user message, so")
    print("    'reply to the user's most recent message' points at itself.")
    for shape in SHAPES:
        got = ask(base, model, build(shape, "What is the capital of France?", ALWAYS_ON_INJECTS))
        if got["content"].startswith("HTTP "):
            report(shape, f"REJECTED -- {got['content'][:80]}")
            continue
        answered = "paris" in got["content"].lower()
        verdict = "answered" if answered else "DID NOT ANSWER"
        if narrates(got["content"]):
            verdict += " + narrates the injects"
        report(shape, verdict, show("reply", got["content"]))


def probe_narration(base: str, model: str) -> None:
    print("\n[3] Does the model remark on the injects instead of just answering?")
    print("    Reproduces the 2026-07-19 observation, where the reasoning enumerated the injects as")
    print("    things the user had sent and the reply called the exchange a system test.")
    for shape in SHAPES:
        got = ask(base, model, build(shape, "Testing 1 2 3", ALWAYS_ON_INJECTS),
                  think=True, max_tokens=THINK_BUDGET)
        if got["content"].startswith("HTTP "):
            report(shape, f"REJECTED -- {got['content'][:80]}")
            continue
        # Only the reply decides. Reasoning that enumerates the constraints is ordinary model
        # behaviour; the failure is the model addressing the injects where the user can see it.
        verdict = "REPLY NARRATES" if narrates(got["content"]) else "reply is clean"
        if not got["content"]:
            verdict = "no reply (budget exhausted?)"
        report(shape, f"{verdict}, reasoning {len(got['reasoning'])} chars",
               show("reply", got["content"]))
        if got["reasoning"]:
            print(f"      {show('reasoning', got['reasoning'])}")


def probe_material_placement(base: str, model: str) -> None:
    print("\n[4] Retrieved material at the FRONT (what we ship) vs at the END (what the KV cache wants).")
    print("    Front placement dates from Qwen 3.0, which ignored material injected late. If that no")
    print("    longer holds, the per-turn full-prefix rebuild buys nothing.")
    filler = [{"role": "user", "content": "Remind me to check the calibration log later."},
              {"role": "assistant", "content": "Noted — I'll remind you about the calibration log."}]
    material = f"[System information: Knowledge-base match from 'kuiper7.txt'.]\n\n{NEEDLE_FACT}\n-----"

    for role in ("user", "tool"):
        front = [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": role, "content": material},
                 *filler,
                 {"role": "user", "content": NEEDLE_QUESTION}]
        end = [{"role": "system", "content": SYSTEM_PROMPT},
               *filler,
               {"role": "user", "content": NEEDLE_QUESTION},
               {"role": role, "content": material}]
        for where, messages in (("front", front), ("end", end)):
            got = ask(base, model, messages)
            if got["content"].startswith("HTTP "):
                report(f"{role} @ {where}", f"REJECTED -- {got['content'][:80]}")
                continue
            report(f"{role} @ {where}", "material used" if NEEDLE in got["content"] else "material IGNORED",
                   show("reply", got["content"]))


def probe_context_only_wording(base: str, model: str) -> None:
    print("\n[5] The 'answer from context only' reminder, with no context supplied.")
    print("    Observed: the model spends its reasoning deciding whether general knowledge is allowed.")
    print("    A general-knowledge question with an empty document database must still work — that is")
    print("    exactly what a live audience will ask.")
    variants = {"none (control)": None,
                "current": CONTEXT_ONLY_INJECT,
                "prefer-context": ("[System information: Prefer information from the context when it is "
                                   "relevant. Say so if you are drawing on general knowledge instead.]"),
                "cite-or-say-so": ("[System information: Base claims about the provided documents on those "
                                   "documents. Answer general questions normally.]")}
    for label, inject in variants.items():
        injects = [DATETIME_INJECT] + ([inject] if inject else [])
        got = ask(base, model, build("user", "What is 2+2?", injects), think=True, max_tokens=THINK_BUDGET)
        if got["content"].startswith("HTTP "):
            report(label, f"REJECTED -- {got['content'][:80]}")
            continue
        answered = "4" in got["content"]
        report(label, f"{'answered' if answered else 'DID NOT ANSWER'}, "
                      f"reasoning {len(got['reasoning'])} chars",
               show("reply", got["content"], limit=120))
        if got["reasoning"]:
            print(f"      {show('reasoning', got['reasoning'], limit=300)}")


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
        print(f"models available: {ids}")

    probes = (probe_tool_role_accepted, probe_last_question_survives, probe_narration,
              probe_material_placement, probe_context_only_wording)
    if len(sys.argv) > 3:  # e.g. "3,5" -- a full pass is minutes of generation, so allow a subset
        wanted = {int(n) for n in sys.argv[3].split(",")}
        probes = tuple(probe for n, probe in enumerate(probes, start=1) if n in wanted)

    print(f"probing {base} with model {model!r}")
    for probe in probes:
        probe(base, model)
    print("\ndone")


if __name__ == "__main__":
    main()
