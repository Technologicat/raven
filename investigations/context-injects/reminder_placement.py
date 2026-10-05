#!/usr/bin/env python3
"""Manual live probe: the grounding reminder sent only when there is material (A), or always (C′)?

NOT a pytest test — it needs an LM Studio backend, and it loads and unloads the models it measures.

Today the reminder ("Base claims about the provided documents on those documents…") is appended to the
system message only on a turn that has material to ground in. The system message is the start of the
prompt, so on a branch with nothing grounded before, the first grounded turn changes it and the backend
reprocesses the whole conversation. C′ sends a conditionally worded reminder always, so the system message
never changes:

    A    as shipped: the reminder only when there is material.
    C′   "When documents, attachments or tool results are in the conversation, base claims about them on
         them. Answer general questions normally." — always, as the system prompt.

Built through Raven's own code (`llmclient.setup`, `agent.turn`, the real character card), with the arm set
by per-run overrides on the settings object: C′ sets `system_prompt` (which ships empty and leads the system
message when set) and silences the conditional reminder's formatter. Nothing shipped changes.

Scenarios, the first three the reminder's job and the fourth its cost:

    general   "Who wrote Hamlet?", nothing retrieved. C′'s risk: a reminder about material with none present
              (context-injects Q4 measured the old wording at 5–37× the deliberation here). Not Q4's
              "What is 2+2?": with tools on offer the model calls `calculate`, whose result counts as material,
              so the turn ends grounded in both arms and stops being the no-material case.
    absent    the Kuiper-7 document retrieved, asked about Kuiper-9: say so, or invent?
    present   the same document, asked about Kuiper-7: answer from it?
    cache     a long ungrounded turn, then a grounded one; the second turn's prefill time.

Every turn is recorded whole (`TurnRecord.to_dict`, phases included) as it finishes, and streamed to a log
as it runs. The run resumes: a sample already in the results file is skipped.

    python reminder_placement.py http://host:1234 MODEL_KEY [MODEL_KEY ...]
"""

import json
import pathlib
import re
import sys
import threading
import time

import requests

from raven.client import api as client_api, config as client_config
from raven.librarian import agent, config as librarian_config, llmclient

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "reminder_placement.jsonl"
STREAM_LOG = HERE / "reminder_placement-stream.log"
CONTEXT_LENGTH = 32768
MAX_REPLY_TOKENS = 16384  # a runaway shows as `finish=length` rather than running on

C_PRIME = ("[System information: When documents, attachments or tool results are in the conversation, base "
           "claims about them on them. Answer general questions normally.]")

# From `inject_shapes.py`: a fact no model can hold, so an answer containing it was read, not recalled.
NEEDLE_FACT = ("The Kuiper-7 sensor array reports a baseline drift of 4.2 millikelvin per hour "
               "under nominal load.")
NEEDLE = "4.2"
QUESTIONS = {"general": "Who wrote Hamlet?",
             "absent": "What baseline drift does the Kuiper-9 sensor array report?",
             "present": "What baseline drift does the Kuiper-7 sensor array report?"}
SAMPLING = [(0.0, 3), (1.0, 9)]  # (temperature, samples): a single T=0 sample carries no information here
CACHE_SAMPLES = 3
# The cache case's first turn: long, and answerable without the documents.
LONG_TEXT = (HERE.parent.parent / "raven" / "librarian" / "README.md").read_text(encoding="utf-8")[:30000]


class KuiperRetriever:
    """Stands in for `HybridIR`: one document, returned for any query."""

    def __init__(self) -> None:
        self.documents = {"kuiper7.txt": {"text": NEEDLE_FACT, "path": "/kuiper7.txt"}}
        self.datastore_lock = threading.RLock()

    def query(self, q, k=10, return_extra_info=False, **kwargs):
        return [{"document_id": "kuiper7.txt", "text": NEEDLE_FACT, "score": 0.9, "offset": 0}]


def lmstudio(base: str, path: str, payload: dict) -> dict:
    response = requests.post(f"{base}{path}", json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def settings_for(base: str, arm: str, temperature: float):
    settings = llmclient.setup(backend_url=base, quiet=True)
    settings.request_data["temperature"] = temperature
    settings.request_data["max_tokens"] = MAX_REPLY_TOKENS
    if arm == "C'":
        settings.system_prompt = C_PRIME
        settings.formatters.reminder_to_use_information_from_context_only = lambda: ""
    return settings


def verdict(scenario: str, reply: str) -> str:
    """A hint only; the stored reply decides. Regexes from `inject_shapes.py`, which learned them the hard way."""
    lowered = reply.lower()
    if not reply.strip():
        return "no reply"
    if scenario == "absent":
        declined = bool(re.search(r"(does not|do not|doesn't|don't|cannot|can't|unable to)\s+"
                                  r"(contain|include|have|mention|provide|find|specify)", lowered)
                        or re.search(r"no (information|data|mention|record|reference|details|specific)", lowered)
                        or re.search(r"only (contains?|reports?|provides?|includes?|has|covers)", lowered)
                        or any(phrase in lowered for phrase in ("not mentioned", "not provided", "not available", "not specified")))
        invented = bool({m for m in re.findall(r"(\d+(?:\.\d+)?)\s*(?:mk\b|millikelvin)", lowered)} - {NEEDLE})
        return "confabulated" if invented else ("declined" if declined else "unclear")
    if scenario == "present":
        return "answered from the document" if NEEDLE in reply else "unclear"
    return "answered" if "shakespeare" in lowered else "unclear"


def main() -> None:
    base, models = sys.argv[1].rstrip("/"), sys.argv[2:]
    librarian_config.llm_backend_url = base
    client_api.initialize(raven_server_url=client_config.raven_server_url,
                          raven_api_key_file=client_config.raven_api_key_file)
    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["model"], r["arm"], r["scenario"], r["temperature"], r["sample"]))

    with RESULTS.open("a", encoding="utf-8") as results, STREAM_LOG.open("a", encoding="utf-8") as stream:
        def record(**fields) -> None:
            results.write(json.dumps(fields, ensure_ascii=False, default=str) + "\n")
            results.flush()

        for model in models:
            pending = [(scenario, temperature, k, arm)
                       for scenario in ("general", "absent", "present") for temperature, n in SAMPLING
                       for k in range(n) for arm in ("A", "C'")]
            pending += [("cache", 0.0, k, arm) for k in range(CACHE_SAMPLES) for arm in ("A", "C'")]
            pending = [item for item in pending if (model, item[3], item[0], item[1], item[2]) not in done]
            if not pending:
                continue
            loaded = lmstudio(base, "/api/v1/models/load", {"model": model, "context_length": CONTEXT_LENGTH, "parallel": 1})
            print(f"{model}: loaded in {loaded.get('load_time_seconds')} s; {len(pending)} turns to go", flush=True)
            try:
                for scenario, temperature, k, arm in pending:
                    settings = settings_for(base, arm, temperature)
                    label = f"{model} {arm} {scenario} T={temperature} #{k}"
                    t0 = time.monotonic()
                    if scenario == "cache":
                        retriever = KuiperRetriever()
                        first = agent.turn(settings, f"In one sentence, what is the following text about?\n\n{LONG_TEXT}",
                                           retriever=retriever, docs_query=None,
                                           on_progress=agent.stream_log(stream, label=f"{label} first"))
                        out = agent.turn(settings, QUESTIONS["present"], datastore=first.datastore,
                                         head_node_id=first.head_node_id, retriever=retriever,
                                         on_progress=agent.stream_log(stream, label=label))
                        extra = {"first_turn": first.to_dict()}
                    else:
                        retriever = KuiperRetriever() if scenario != "general" else None
                        out = agent.turn(settings, QUESTIONS[scenario], retriever=retriever,
                                         on_progress=agent.stream_log(stream, label=label))
                        extra = {}
                    wall = time.monotonic() - t0
                    hint = verdict("present" if scenario == "cache" else scenario, out.reply)
                    record(model=model, arm=arm, scenario=scenario, temperature=temperature, sample=k,
                           wall_s=wall, verdict=hint, record=out.to_dict(), **extra)
                    reasoning = sum(len(r) for r in out.reasoning)
                    print(f"  {label}: {hint}, reasoning {reasoning} chars, wall {wall:.1f} s", flush=True)
            finally:
                lmstudio(base, "/api/v1/models/unload", {"instance_id": loaded["instance_id"]})


if __name__ == "__main__":
    main()
