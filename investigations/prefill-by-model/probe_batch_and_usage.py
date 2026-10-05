"""Two follow-ups to `probe_prefill.py`, on one model, and the circle prompt while it is loaded.

1. Does a larger physical batch speed up prompt processing? Cold prefill at LM Studio's default
   (`physical_batch_size` 512) against larger ones, the same prompt shape as `probe_prefill.py`.
2. Does the OpenAI-compatible endpoint, the one Raven uses, report `usage.prompt_tokens` relative to the
   cache? The same request twice to `/v1/chat/completions`: the second hits the cache, so a cache-relative
   count drops and a full one does not. `/api/v0` reported the full count in `probe_prefill.py`.
3. The circle prompt ("draw an svg of a circle", from Simon Willison's post on Qwen 3.8 27B) with thinking
   on, under whatever reasoning effort the server's chat template sets: reasoning length and time.

Every response is written to `results.jsonl`, whole.

    python investigations/prefill-by-model/probe_batch_and_usage.py http://host:1234 MODEL_KEY
"""

import json
import sys
import time
import uuid

import requests

import probe_prefill as base_probe

PHYSICAL_BATCH_SIZES = [512, 1024, 2048]


def main() -> None:
    base, model = sys.argv[1].rstrip("/"), sys.argv[2]
    with base_probe.RESULTS.open("a", encoding="utf-8") as results:
        def record(**fields) -> None:
            results.write(json.dumps({"t": time.time(), **fields}) + "\n")
            results.flush()

        def load(**config) -> str:
            loaded = base_probe.post(f"{base}/api/v1/models/load",
                                     {"model": model, "context_length": base_probe.CONTEXT_LENGTH, "parallel": 1, **config})
            info = [m for m in requests.get(f"{base}/api/v1/models", timeout=30).json()["models"] if m["key"] == model]
            instances = info[0]["loaded_instances"] if info else None
            record(event="load", model=model, requested=config, response=loaded, instances=instances)
            actual = instances[0]["config"] if instances else {}
            print(f"{model}: loaded with physical_batch_size {actual.get('physical_batch_size')}, "
                  f"eval_batch_size {actual.get('eval_batch_size')}", flush=True)
            return loaded["instance_id"]

        def unload(instance_id: str) -> None:
            base_probe.post(f"{base}/api/v1/models/unload", {"instance_id": instance_id})
            record(event="unload", model=model)

        # 1. Physical batch size.
        for physical in PHYSICAL_BATCH_SIZES:
            instance_id = load(physical_batch_size=physical, eval_batch_size=max(2048, physical))
            try:
                for k in range(2):
                    out, wall = base_probe.ask(base, model, uuid.uuid4().hex, base_probe.QUESTIONS[k])
                    record(event="request", model=model, kind="cold", physical_batch_size=physical, wall_s=wall, response=out)
                    stats, usage = out.get("stats", {}), out.get("usage", {})
                    ttft = stats.get("time_to_first_token")
                    print(f"  cold: prompt_tokens {usage.get('prompt_tokens')}, ttft {ttft:.2f} s "
                          f"({usage.get('prompt_tokens') / ttft:.0f} t/s)", flush=True)
            finally:
                if physical != PHYSICAL_BATCH_SIZES[-1]:
                    unload(instance_id)

        # 2. `usage` on the OpenAI-compatible endpoint, with the last load still in place.
        try:
            payload = {"model": model,
                       "messages": [{"role": "system", "content": f"Session {uuid.uuid4().hex}.\n\n{base_probe.DOCUMENT}"},
                                    {"role": "user", "content": base_probe.QUESTIONS[0]}],
                       "max_tokens": 50, "reasoning_effort": "none", "stream": False}
            for attempt in ("first", "repeat"):
                out = base_probe.post(f"{base}/v1/chat/completions", payload)
                record(event="usage_check", model=model, attempt=attempt, response=out)
                print(f"  /v1 {attempt}: usage {out.get('usage')}", flush=True)

            # 3. The circle prompt, thinking on: no `reasoning_effort` sent, so the template's setting governs.
            t0 = time.monotonic()
            out = base_probe.post(f"{base}/api/v0/chat/completions",
                                  {"model": model, "messages": [{"role": "user", "content": "draw an svg of a circle"}],
                                   "max_tokens": 16384, "stream": False}, timeout=1800)
            wall = time.monotonic() - t0
            record(event="circle", model=model, wall_s=wall, response=out)
            message = out["choices"][0]["message"]
            print(f"  circle: wall {wall:.1f} s, usage {out.get('usage')}, stats {out.get('stats')}, "
                  f"reasoning {len(message.get('reasoning_content') or '')} chars, "
                  f"answer {len(message.get('content') or '')} chars", flush=True)
        finally:
            unload(instance_id)


if __name__ == "__main__":
    main()
