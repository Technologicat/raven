#!/usr/bin/env python3
"""Manual live check: does Raven's *assembled* context inject still behave as the sweep predicted?

NOT a pytest test — it needs a running backend with a model loaded, so it lives here under `briefs/`
rather than in the suite.

The other probes here hand-build a wire history to compare candidate shapes. This one does the opposite:
it builds the history through Raven's own code (`llmclient.configure` + `scaffold.build_turn_prompt` +
`llmclient._serialize_history_for_wire`) and asks whether the shapes that *won* the sweep still deliver
once assembled together. A shape can measure well in isolation and interact badly in company — the date
inject, the clock tool call and the retrieval tool call all land in the same turn.

Settings included: the system prompt and character card are Raven's real ones, not stand-ins, so what is
measured is the whole assembled prompt rather than the injects sitting in a placeholder.

The four checks are the four things the shapes were chosen to buy:

  1. retrieval lands       the planted figure comes back, and the reply is an answer rather than another
                           tool call (the failure the `before` placement exists to prevent)
  2. grounding holds       asked for a figure the documents do not contain, the model declines instead of
                           inventing one
  3. general questions     "what is 2+2?" is answered, not deliberated (the failure Q4 is about)
  4. the clock is believed the injected date and time come back, rather than the model's training prior

Read `finish` and the reasoning length, not just the verdict: an empty reply at `finish=length` is a
truncated deliberation, not a refusal, and the two want opposite responses.

Requires the venv (it imports Raven); the backend may be remote.

Usage:
    python assembled_shape.py [base_url] [model[,model,...]]
"""

import datetime
import json
import sys
import urllib.request

from unpythonic.env import env

from raven.librarian import chatutil, llmclient, scaffold

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:1234"
MODELS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["qwen3.6-27b"]

NEEDLE_FACT = "The Vantaa-3 pressurized alkaline stack draws 41.7 kWh/kg under nominal load."


def settings_for(model):
    """Raven's own settings for `model`, built without contacting a backend.

    The same object `llmclient.setup` returns — same system prompt, same character card — so the assembled
    prompt this probe measures is the one a real turn is given. It used to hand-assemble an `env` with a
    handful of plausible fields and a stand-in system prompt, which undercut the claim in the module
    docstring: the injects were real, the prompt they landed in was not.

    Per model, because the card names the model, and this probe sweeps several. The context length is
    stated rather than discovered; nothing here measures context-length behaviour.
    """
    return llmclient.configure(model_info=env(label=model, model_id=model,
                                              context_length=131072, is_vlm=None),
                               backend_flavor="generic",
                               backend_url=BASE,
                               quiet=True)


def build(settings, question, docs_query, matches):
    history = [chatutil.create_chat_message(llm_settings=settings, role="system",
                                            text="You are Aria, a helpful research assistant working at a university."),
               chatutil.create_chat_message(llm_settings=settings, role="assistant", text="How can I help you today?"),
               chatutil.create_chat_message(llm_settings=settings, role="user", text="I'm reviewing the hydrogen production literature this week."),
               chatutil.create_chat_message(llm_settings=settings, role="assistant", text="Happy to help - tell me what you need from it."),
               chatutil.create_chat_message(llm_settings=settings, role="user", text=question)]
    # `grounded=True` is what the retrieval would have declared in a real turn, and it is what keeps the
    # context-only reminder in the assembled prompt — which is one of the four shapes this probe measures.
    tool_context = scaffold._make_tool_context(llm_settings=settings, retriever=None)
    tool_context.grounded = True
    history = scaffold.build_turn_prompt(llm_settings=settings, history=history,
                                         docs_query=docs_query, docs_matches=matches,
                                         tool_context=tool_context)
    return llmclient._serialize_history_for_wire(settings, history, continue_=False)


def ask(model, messages, max_tokens=8000):
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps({"model": model, "messages": messages,
                                                  "max_tokens": max_tokens, "temperature": 0.0}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=900) as r:
        body = json.loads(r.read().decode("utf-8"))
    choice = body.get("choices", [{}])[0]
    msg = choice.get("message", {})
    return {"content": (msg.get("content") or "").strip(),
            "reasoning": (msg.get("reasoning_content") or msg.get("reasoning") or "").strip(),
            "tool_calls": msg.get("tool_calls") or [],
            "finish": choice.get("finish_reason") or "?"}


def show(label, got, extra=""):
    one_line = " ".join(got["content"].split())
    print(f"    {label:<22} {extra}  [finish={got['finish']}, reasoning={len(got['reasoning'])} chars, "
          f"tool_calls={len(got['tool_calls'])}]")
    print(f"      reply: {one_line[:260]!r}")


def main():
    matches = [{"document_id": "abstract_001.txt", "text": "Alkaline electrolysis remains the workhorse of industrial hydrogen production.", "score": 0.6, "offset": 0},
               {"document_id": "vantaa3_stack_report.txt", "text": NEEDLE_FACT, "score": 0.9, "offset": 0},
               {"document_id": "abstract_002.txt", "text": "PEM electrolyzers offer faster load following at higher capital cost.", "score": 0.5, "offset": 0}]
    today = datetime.date.today()

    for model in MODELS:
        print(f"\n=== {model} ===")
        settings = settings_for(model)

        print("  [1] Retrieval: does the answer use the planted figure, and stay an answer (no further tool call)?")
        got = ask(model, build(settings, "What is the specific energy consumption of the Vantaa-3 stack?",
                               "Vantaa-3 specific energy consumption", matches))
        show("needle", got, "HIT " if "41.7" in got["content"] else "MISS")

        print("  [2] Absent fact: does the reminder still constrain, with documents present?")
        got = ask(model, build(settings, "What is the specific energy consumption of the Kelvin-7 stack?",
                               "Kelvin-7 specific energy consumption", matches))
        declines = any(w in got["content"].lower() for w in ("no ", "not ", "doesn't", "does not", "don't", "cannot", "can't"))
        show("absent fact", got, "DECLINED" if declines else "CHECK BY EYE")

        print("  [3] General question, no context: does it just answer?")
        got = ask(model, build(settings, "What is 2+2?", None, []))
        show("2+2", got, "OK  " if "4" in got["content"] else "CHECK")

        print("  [4] Date and time: is the injected clock believed?")
        got = ask(model, build(settings, "What is today's date, and what time is it now?", None, []))
        show("datetime", got, "OK  " if today.isoformat() in got["content"] else "CHECK")


if __name__ == "__main__":
    main()
