#!/usr/bin/env python3
"""Manual live probe: asked something the retrieved documents do not answer, what does the model do?

NOT a pytest test — it needs a running backend with a model loaded, so it lives here under `briefs/`
rather than in the suite.

Unlike the other probes here, this one builds its wire history through Raven's own code
(`scaffold._perform_injects` + `llmclient._serialize_history_for_wire`) rather than reimplementing the
shape, so it measures what Raven actually sends. That costs it the stdlib-only property the others have:
it needs the venv, and cannot be piped to a machine that lacks one.

The case is the one where retrieval is least comfortable: the documents are relevant to the *topic* but
silent on the *question*. A model that wants more material will try to get more material — and Raven's
document search is not a tool the model may call, it is a search Raven ran on the model's behalf before
the turn began. So the request has nowhere to go: with no tools declared, the model writes the call out
as literal `<tool_call>` text, and the user gets that instead of an answer.

Measured on Qwen3.6-27B, nine samples per variant at temperature 1.0:

    as-shipped         3/9 asked for another search
    closing-note       0/9   <- the tool result ends by saying the search is already done
    no-synthetic-call  1/3 (and one reply lost track of who had said what)

The third variant is the control that rules out the obvious suspect: removing the synthetic call does not
fix this, so the reaching-for-a-tool behaviour comes from the empty-handed question, not from having been
shown a call.

The second variant was **rejected despite the 0/9**, which is why this probe is worth keeping around. Run
it at `temperature=0` with a large budget and that wording spends 29000 characters of reasoning without
producing a reply, where sending nothing extra answers cleanly in 3000: it reads as a prohibition, and
prohibitions are what the whole inject rework exists to stop handing the model. Sampling at one
temperature was enough to make a bad shape look like a fix. So: run both temperatures, and read the
reasoning length, not only the verdict. See `investigations/context-injects/context-inject-shape-measurements.md`.

Usage:
    python absent_fact.py [base_url] [model] [samples_per_variant]
"""

import json
import re
import sys
import urllib.request

from unpythonic.env import env

from raven.librarian import chatutil, llmclient, scaffold

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:1234"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.6-27b"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 3

settings = env(user="User", char="Aria", model="test-model",
               system_prompt="You are a helpful assistant.", character_card="Name: Aria",
               greeting="How can I help you today?",
               personas={"user": "User", "assistant": "Aria", "system": None, "tool": None})

MATCHES = [{"document_id": "abstract_001.txt", "text": "Alkaline electrolysis remains the workhorse of industrial hydrogen production.", "score": 0.6, "offset": 0},
           {"document_id": "vantaa3_stack_report.txt", "text": "The Vantaa-3 pressurized alkaline stack draws 41.7 kWh/kg under nominal load.", "score": 0.9, "offset": 0},
           {"document_id": "abstract_002.txt", "text": "PEM electrolyzers offer faster load following at higher capital cost.", "score": 0.5, "offset": 0}]

QUESTION = "What is the specific energy consumption of the Kelvin-7 stack?"

CLOSING_NOTE = ("[System information: These are all the matches for this turn. The document database has "
                "already been searched; you cannot search it again yourself.]")


def build(variant):
    history = [chatutil.create_chat_message(llm_settings=settings, role="system",
                                            text="You are Aria, a helpful research assistant working at a university."),
               chatutil.create_chat_message(llm_settings=settings, role="assistant", text="How can I help you today?"),
               chatutil.create_chat_message(llm_settings=settings, role="user", text="I'm reviewing the hydrogen production literature this week."),
               chatutil.create_chat_message(llm_settings=settings, role="assistant", text="Happy to help - tell me what you need from it."),
               chatutil.create_chat_message(llm_settings=settings, role="user", text=QUESTION)]
    # `grounded=True` is what the retrieval would have declared in a real turn: the matches are on topic,
    # they simply do not answer the question. It keeps the context-only reminder in the prompt, which is the
    # instruction this probe is measuring the model against.
    tool_context = scaffold._make_tool_context(llm_settings=settings, retriever=None)
    tool_context.grounded = True
    scaffold._perform_injects(llm_settings=settings, history=history,
                              docs_query="Kelvin-7 specific energy consumption", docs_matches=MATCHES,
                              tool_context=tool_context)
    wire = llmclient._serialize_history_for_wire(settings, history, continue_=False)

    if variant == "as-shipped":
        return wire
    if variant == "closing-note":
        for message in wire:
            if message["role"] == "tool" and "Knowledge-base match" in message["content"][0]["text"]:
                message["content"][0]["text"] += "\n\n" + CLOSING_NOTE
        return wire
    if variant == "no-synthetic-call":
        # Strip the docs exchange entirely and hand the same material as a plain user-role note, to see
        # whether the invitation to search again comes from the call or from the empty-handed question.
        blocks = None
        stripped = []
        for message in wire:
            text = message["content"][0]["text"] if message["content"] else ""
            if message["role"] == "tool" and "Knowledge-base match" in text:
                blocks = text
                continue
            if message.get("tool_calls") and message["tool_calls"][0]["function"]["name"] == "search_documents":
                continue
            stripped.append(message)
        stripped.insert(-1, {"role": "user", "content": [{"type": "text", "text": blocks}], "tool_calls": []})
        return stripped
    raise ValueError(variant)


def ask(messages, temperature):
    # A budget the model cannot exhaust. At 2000 tokens a runaway deliberation truncates and reads as a
    # refusal, which is the mistake that made the rejected variant look like a fix in the first place.
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps({"model": MODEL, "messages": messages, "max_tokens": 8000,
                                                  "temperature": temperature}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=1200) as r:
        body = json.loads(r.read().decode("utf-8"))
    choice = body.get("choices", [{}])[0]
    msg = choice.get("message", {})
    return {"content": (msg.get("content") or "").strip(),
            "reasoning": (msg.get("reasoning_content") or msg.get("reasoning") or "").strip(),
            "tool_calls": msg.get("tool_calls") or [],
            "finish": choice.get("finish_reason") or "?"}


def requested_a_search(got):
    """Native tool call, or the inline `<tool_call>` text a model emits when no tools were declared."""
    return bool(got["tool_calls"]) or bool(re.search(r"<tool_call>|<function=", got["content"]))


def main():
    # Both temperatures, because they disagree. T=1 samples the behaviour a user actually meets; T=0 is
    # where a wording that quietly costs the model its whole reasoning budget shows up.
    for temperature, samples in ((0.0, 1), (1.0, N)):
        for variant in ("as-shipped", "closing-note", "no-synthetic-call"):
            print(f"\n--- {variant} @ T={temperature} ---")
            asked_again = 0
            for i in range(samples):
                got = ask(build(variant), temperature=temperature)
                again = requested_a_search(got)
                asked_again += again
                print(f"  [{i + 1}] {'ASKED FOR ANOTHER SEARCH' if again else 'answered':<24} "
                      f"finish={got['finish']}, reasoning={len(got['reasoning'])} chars")
                print(f"      reply: {' '.join(got['content'].split())[:180]!r}")
            print(f"  => {asked_again}/{samples} tried to search again")


if __name__ == "__main__":
    main()
