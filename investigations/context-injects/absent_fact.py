#!/usr/bin/env python3
"""Manual live probe: asked something the retrieved documents do not answer, what does the model do?

NOT a pytest test — it needs a running backend with a model loaded, so it lives here under `briefs/`
rather than in the suite.

Unlike the other probes here, this one builds its wire history through Raven's own code
(`llmclient.configure` + `scaffold.build_turn_prompt` + `llmclient.serialize_history_for_wire`) rather than
reimplementing the shape, so it measures what Raven actually sends — settings, system prompt and character
card included, not just the injects. That costs it the stdlib-only property the others have: it needs the
venv, and cannot be piped to a machine that lacks one.

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

**That rejection is model-specific, and the model has moved.** On qwen3.6-35b-a3b the runaway belongs to
the *as-shipped* wording instead — three of four samples at T=0 — while `closing-note` answers cleanly in
0 of 3. Treat the paragraph above as the 27B result it is, and re-run the sweep before defending either
wording on it.

**Three samples per arm at T=0, not one.** Identical requests produced 2484, 30757 and 29684 characters of
reasoning in the same run: greedy decoding is deterministic given identical numerics, and a GPU does not
provide those. A single T=0 observation here carries no information.

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

# Raven's own settings, built without contacting a backend — the same object `llmclient.setup` returns, and
# therefore the same system prompt and character card a real turn is given. This probe used to hand-assemble
# an `env` with a handful of plausible fields and a stand-in system prompt, which quietly undercut its own
# claim to measure what Raven sends: the injects were real and the prompt they landed in was not.
#
# `configure` rather than `setup` because the model under test is a command-line argument, so the card should
# name *that* model rather than whatever the backend happens to have loaded. The context length is stated
# rather than discovered for the same reason; nothing here measures context-length behaviour.
settings = llmclient.configure(model_info=env(label=MODEL, model_id=MODEL,
                                              context_length=131072, is_vlm=None),
                               backend_flavor="generic",
                               backend_url=BASE,
                               quiet=True)

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
    tool_context = scaffold.make_tool_context(llm_settings=settings, retriever=None)
    tool_context.grounded = True
    history = scaffold.build_turn_prompt(llm_settings=settings, history=history,
                                         docs_query="Kelvin-7 specific energy consumption", docs_matches=MATCHES,
                                         tool_context=tool_context)
    wire = llmclient.serialize_history_for_wire(settings, history, continue_=False)

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
    # Raven's own request template, tools and samplers included, so that what varies between the arms is the
    # thing under test. Sampling is part of what Raven sends: the shipped configuration is `min_p=0.02` ahead
    # of the temperature, and a probe that sends a bare temperature is measuring a distribution nobody runs.
    payload = dict(settings.request_data)
    payload["messages"] = messages
    payload["temperature"] = temperature  # the variable this probe sweeps
    # A budget the model cannot exhaust, but small enough to keep a runaway tractable — Raven itself ships no
    # cap, so an arm that ends `finish=length` here is one Raven would have let run to the context window.
    payload["max_tokens"] = 8000
    payload.pop("stream", None)  # this probe reads one whole response; `invoke` is the streaming caller
    # The controlled condition, and the reason the numbers here compare with the earlier nine-sample runs:
    # no tools are declared, so a model that wants to search has nowhere to put the request and writes it
    # out as text. Raven's real turns *do* declare `search_documents`, so this arm is deliberately narrower
    # than a live turn rather than a replica of one.
    payload.pop("tools", None)
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"),
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
    #
    # T=0 is sampled repeatedly too. Greedy decoding is deterministic on paper, and on a GPU only nearly so:
    # kernel choice and float non-associativity can flip a near-tie, after which the trajectory diverges. The
    # runaway arm is where that bites hardest, being some 8000 sampling decisions long rather than a few
    # hundred. Whether the runaway repeats is the open question, so it is worth the samples until answered -
    # if it repeats, one sample is enough here afterwards.
    for temperature, samples in ((0.0, N), (1.0, N)):
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
