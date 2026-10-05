"""How fast each model processes a prompt, cold and with the prefix cached, on an LM Studio backend.

For each model: load it, run three cold requests (a fresh nonce opens the system text, so nothing is
cached), then two tail requests (the last nonce again, with a different question, so only the tail is new -
the shape of Librarian's query request), then unload it. Thinking is off and the reply capped, so the
measurement is prompt processing first and generation second.

Timings are LM Studio's own: `/api/v0/chat/completions` returns `stats` with `time_to_first_token` and
`tokens_per_second` beside `usage`. Note `usage.prompt_tokens` counts only what the cache did not already
hold (`investigations/prompt-size-cache-relative/`), which is what makes the tail runs' figure the tail.

Every response is written to `results.jsonl` as it arrives, whole.

    python investigations/prefill-by-model/probe_prefill.py http://host:1234 MODEL_KEY [MODEL_KEY ...]
"""

import json
import pathlib
import sys
import time
import uuid

import requests

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results.jsonl"
CONTEXT_LENGTH = 32768
MAX_TOKENS = 300
# A fixed, long, ordinary document: Librarian's own user manual. About 10k tokens.
DOCUMENT = (HERE.parent.parent / "raven" / "librarian" / "README.md").read_text(encoding="utf-8")[:40000]
QUESTIONS = ["Summarize the text above in about 150 words.",
             "List five features the text above describes, one line each.",
             "What does the text above say about the chat graph? Answer in about 100 words."]


def post(url: str, payload: dict, timeout: float = 600) -> dict:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def ask(base: str, model: str, nonce: str, question: str) -> tuple[dict, float]:
    payload = {"model": model,
               "messages": [{"role": "system", "content": f"Session {nonce}.\n\n{DOCUMENT}"},
                            {"role": "user", "content": question}],
               "max_tokens": MAX_TOKENS,
               "temperature": 0.7,
               "reasoning_effort": "none",  # what Raven sends for thinking off; see `llmclient.thinking_request_fields`
               "stream": False}
    t0 = time.monotonic()
    out = post(f"{base}/api/v0/chat/completions", payload)
    return out, time.monotonic() - t0


def main() -> None:
    base, models = sys.argv[1].rstrip("/"), sys.argv[2:]
    with RESULTS.open("a", encoding="utf-8") as results:
        def record(**fields) -> None:
            results.write(json.dumps({"t": time.time(), **fields}) + "\n")
            results.flush()

        for model in models:
            # `parallel` pinned so that every model is measured under the same load settings: the stored
            # defaults are per model, and some said 4.
            loaded = post(f"{base}/api/v1/models/load", {"model": model, "context_length": CONTEXT_LENGTH, "parallel": 1})
            info = [m for m in requests.get(f"{base}/api/v1/models", timeout=30).json()["models"] if m["key"] == model]
            record(event="load", model=model, response=loaded, instances=info[0]["loaded_instances"] if info else None)
            print(f"{model}: loaded in {loaded.get('load_time_seconds')} s", flush=True)
            try:
                nonce = None
                for kind, n in (("cold", 3), ("tail", 2)):
                    for k in range(n):
                        if kind == "cold":
                            nonce = uuid.uuid4().hex
                        question = QUESTIONS[k % len(QUESTIONS)] if kind == "cold" else QUESTIONS[(k + 1) % len(QUESTIONS)]
                        out, wall = ask(base, model, nonce, question)
                        record(event="request", model=model, kind=kind, nonce=nonce, question=question,
                               wall_s=wall, response=out)
                        stats, usage = out.get("stats", {}), out.get("usage", {})
                        ttft = stats.get("time_to_first_token")
                        prompt_tokens = usage.get("prompt_tokens")
                        rate = prompt_tokens / ttft if ttft and prompt_tokens else None
                        print(f"  {kind}: prompt_tokens {prompt_tokens}, ttft {ttft} s"
                              + (f" ({rate:.0f} t/s)" if rate else "")
                              + f", gen {stats.get('tokens_per_second')} t/s, wall {wall:.1f} s", flush=True)
            finally:
                post(f"{base}/api/v1/models/unload", {"instance_id": loaded["instance_id"]})
                record(event="unload", model=model)


if __name__ == "__main__":
    main()
