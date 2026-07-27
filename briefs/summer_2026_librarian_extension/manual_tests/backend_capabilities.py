#!/usr/bin/env python3
"""Manual live probe: what does an LLM backend's HTTP API *actually* support?

NOT a pytest test — it needs a running backend with a model loaded, so it lives here
under `briefs/` rather than in the suite.

Exists because backend documentation is unreliable in both directions. LM Studio's
documented parameter list omits `min_p`, which it honours; and it documents nothing
that would suggest `chat_template_kwargs` is dropped, which it is. Worse, it answers
HTTP 200 to a parameter name invented on the spot — so **a request being accepted
proves nothing at all**. Every check below therefore observes behaviour: token counts,
reasoning lengths, or whether output varies with the seed.

Run it after upgrading a backend, or when pointing Raven at a new one, to see which
mechanisms are available before writing code against them.

Findings on LM Studio 0.4.19 (Build 2), 2026-07-27, for comparison against whatever
you get. Models: unsloth Qwen3.5-9B / Qwen3.6-35B-A3B, lmstudio-community Gemma4-26B-A4B.

  - unknown parameters              : silently ignored (so acceptance means nothing)
  - min_p                           : honoured, despite being undocumented
  - assistant prefill               : works on both endpoints
  - chat_template_kwargs            : ignored on both endpoints
  - thinking toggle, OpenAI-compat  : unavailable (prefill is the workaround)
  - thinking toggle, Anthropic-compat: works, via Anthropic's native `thinking` field
  - thinking history fed back       : Qwen drops it; Gemma 4 keeps it

Prefill suppresses thinking because a trailing assistant message leaves no generation
prompt, and the generation prompt is where the template emits its thinking prefix --
the prefill's *content* is incidental, as a bare "The" suppresses it just as well.
`CLOSED_THINK` below is nonetheless the Qwen-correct string, so the model sees
well-formed markup rather than a foreign tag. Gemma's equivalent is
`<|channel>thought\n<channel|>`.

Note the Gemma quant is not interchangeable: the unsloth build fails to load, because
LM Studio's workaround for Gemma's template fires only for the lmstudio-community
build with its bundled template unoverridden.

Usage:
    python backend_capabilities.py                       # localhost:1234, first model
    python backend_capabilities.py <base_url> [model]
"""

import json
import sys
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE = "http://localhost:1234"
TIMEOUT = 300

# Long enough that retaining it would move a token count unmistakably.
LONG_REASONING = ("The user is asking about the capital of France. " * 60).strip()

# What the Qwen chat template emits itself when thinking is off, so prefilling it
# reproduces non-thinking mode exactly rather than approximating it. Template-shaped:
# a model whose reasoning markup differs (Gemma) needs a different string.
CLOSED_THINK = "<think>\n\n</think>\n\n"


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
        return {"_error": f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:200]}"}
    except Exception as e:  # noqa: BLE001 -- a probe reports failures rather than raising
        return {"_error": f"{type(e).__name__}: {e}"}


def reasoning_len(body: dict[str, Any]) -> int | str:
    """Length of an OpenAI-compat reply's reasoning trace, or an error string."""
    if "_error" in body:
        return body["_error"]
    msg = body.get("choices", [{}])[0].get("message", {})
    return len(msg.get("reasoning_content") or msg.get("reasoning") or "")


def content_of(body: dict[str, Any]) -> str:
    if "_error" in body:
        return body["_error"]
    return (body.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()


def chat(base: str, model: str, **extra: Any) -> dict[str, Any]:
    ask = [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}]
    return post(base, "/v1/chat/completions", {"model": model, "messages": ask, "max_tokens": 400, **extra})


def report(label: str, result: Any) -> None:
    print(f"  {label:<38} {result}")


def probe_meta(base: str, model: str) -> None:
    print("\n[meta] Does an unknown parameter error, or get ignored?")
    print("       If ignored, no check below can be read from its status code.")
    body = chat(base, model, max_tokens=16, this_parameter_does_not_exist_xyzzy={"nonsense": True})
    report("bogus parameter", "IGNORED (200)" if "_error" not in body else f"rejected: {body['_error']}")


def probe_sampler(base: str, model: str) -> None:
    print("\n[sampler] Is min_p honoured? (absent from LM Studio's documented list)")
    print("          Read: at high temperature, an honoured clamp makes output seed-invariant.")
    prompt = [{"role": "user", "content": "Write two sentences about the sea."},
              {"role": "assistant", "content": CLOSED_THINK}]

    def sample(seed: int, **extra: Any) -> str:
        body = post(base, "/v1/chat/completions",
                    {"model": model, "messages": prompt, "max_tokens": 80,
                     "temperature": 2.0, "seed": seed, **extra})
        return " ".join(content_of(body).split())

    for label, extra in (("no clamp", {}), ("top_k=1 (documented)", {"top_k": 1}),
                         ("min_p=0.9 (undocumented)", {"min_p": 0.9})):
        a, b = sample(1000, **extra), sample(1001, **extra)
        report(label, "seed-invariant -> honoured" if a == b else "varies with seed -> no effect")


def probe_thinking_openai(base: str, model: str) -> None:
    print("\n[thinking / OpenAI-compat]")
    report("default", f"{reasoning_len(chat(base, model))} chars of reasoning")
    for flag in (False, True):
        n = reasoning_len(chat(base, model, chat_template_kwargs={"enable_thinking": flag}))
        report(f"chat_template_kwargs={flag}", f"{n} chars")
    print("       Read: all three alike means chat_template_kwargs is dropped.")

    ask = [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."},
           {"role": "assistant", "content": CLOSED_THINK}]
    body = post(base, "/v1/chat/completions", {"model": model, "messages": ask, "max_tokens": 400})
    report("closed-<think> prefill", f"{reasoning_len(body)} chars -> {content_of(body)[:48]!r}")


def probe_prefill(base: str, model: str) -> None:
    print("\n[prefill] Trailing assistant message -- continued, or answered afresh?")
    ask = [{"role": "user", "content": "Name the four seasons, in order."},
           {"role": "assistant", "content": "The four seasons are spring, summer,"}]
    body = post(base, "/v1/chat/completions", {"model": model, "messages": ask, "max_tokens": 64})
    report("OpenAI-compat", repr(content_of(body)[:70]))

    body = post(base, "/v1/messages", {"model": model, "messages": ask, "max_tokens": 64})
    if "_error" in body:
        report("Anthropic-compat", body["_error"])
    else:
        blocks = body.get("content", [])
        report("Anthropic-compat", repr((blocks[0].get("text", "") if blocks else "")[:70]))


def probe_anthropic(base: str, model: str) -> None:
    print("\n[Anthropic-compat] endpoint, native thinking control, streaming")
    body = post(base, "/v1/messages",
                {"model": model, "max_tokens": 64,
                 "messages": [{"role": "user", "content": "What is 2+2?"}]})
    if "_error" in body:
        report("/v1/messages", body["_error"])
        return
    report("/v1/messages", f"present; default block types={[c.get('type') for c in body.get('content', [])]}")

    for cfg in ({"type": "disabled"}, {"type": "enabled", "budget_tokens": 1024}):
        b = post(base, "/v1/messages",
                 {"model": model, "max_tokens": 256, "thinking": cfg,
                  "messages": [{"role": "user", "content": "What is 2+2?"}]})
        kinds = b.get("_error") or [c.get("type") for c in b.get("content", [])]
        report(f"thinking={cfg['type']}", kinds)
    print("       Read: a 'thinking' block appearing only when enabled means the toggle works.")

    req = urllib.request.Request(f"{base}/v1/messages",
                                 data=json.dumps({"model": model, "max_tokens": 32, "stream": True,
                                                  "messages": [{"role": "user", "content": "Say hello."}]}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            first = r.read(60).decode("utf-8", errors="replace").split("\n")[0]
        report("streaming", f"yes ({first!r})")
    except Exception as e:  # noqa: BLE001
        report("streaming", f"{type(e).__name__}: {e}")


def probe_thinking_history(base: str, model: str) -> None:
    print("\n[thinking history] Can prior reasoning be fed back into the prompt?")
    print("                   Measured via prompt size; a ~700-token jump means it landed.")
    head = [{"role": "user", "content": "What is the capital of France?"}]
    tail = [{"role": "user", "content": "And of Italy?"}]
    plain = {"role": "assistant", "content": "The capital of France is Paris."}
    sibling = {**plain, "reasoning_content": LONG_REASONING}
    native = {"role": "assistant",
              "content": [{"type": "thinking", "thinking": LONG_REASONING, "signature": ""},
                          {"type": "text", "text": "The capital of France is Paris."}]}

    def oai(msgs: list[dict], **extra: Any) -> Any:
        b = post(base, "/v1/chat/completions", {"model": model, "messages": msgs, "max_tokens": 8, **extra})
        return b.get("_error") or b.get("usage", {}).get("prompt_tokens")

    def ant(msgs: list[dict], **extra: Any) -> Any:
        b = post(base, "/v1/messages", {"model": model, "messages": msgs, "max_tokens": 8, **extra})
        return b.get("_error") or b.get("usage", {}).get("input_tokens")

    report("baseline, no reasoning", oai(head + [plain] + tail))
    report("reasoning_content sibling", oai(head + [sibling] + tail))
    report("+ preserve_thinking=True", oai(head + [sibling] + tail,
                                           chat_template_kwargs={"preserve_thinking": True}))
    report("Anthropic thinking block", ant(head + [native] + tail))


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

    print(f"probing {base} with model {model!r}")
    for probe in (probe_meta, probe_sampler, probe_thinking_openai,
                  probe_prefill, probe_anthropic, probe_thinking_history):
        probe(base, model)
    print("\ndone")


if __name__ == "__main__":
    main()
