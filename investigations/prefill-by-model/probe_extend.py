"""When does LM Studio reuse a cached prompt that the next request *extends*, rather than repeats?

Found from inside Librarian on 2026-10-05: with Qwen 3.8 27B, a prompt of 2k-6k tokens sent twice is reused,
but the same prompt with a reply and a new message after it is processed from the start, as is one whose
last message differs. This morning's `probe_prefill.py` had measured a changed tail on a ~10k-token prompt
as cheap, though. One hypothesis fits both and is untested: a model whose layers are not all full attention
(Qwen 3.5 onward is hybrid, with Gated DeltaNet layers; Gemma 4 uses sliding-window attention) can resume
only from a saved checkpoint, and a short prompt has none to resume from.

So, per model and per prefix length, four requests:

    cold      a fresh nonce opens the system text, so nothing is cached
    repeat    the same request again                                  (the control: an identical prompt)
    extend    cold's messages, then a short reply and a new question  (Librarian's turn after a prefill)
    tail      cold's system text with a different question            (`probe_prefill.py`'s shape)

Timings are LM Studio's own, `stats.time_to_first_token` from `/api/v0/chat/completions`. A model already
loaded is used as it is and left loaded; any other is loaded at 32k context with `parallel` 1 and unloaded
afterwards. Thinking off, replies capped at a few tokens: prompt processing is the whole measurement.

    python investigations/prefill-by-model/probe_extend.py [--parts] [--tools] [--lengths N,N,...] http://host:1234 MODEL_KEY [...]

`--parts` sends each message's content as a list of parts, `[{"type": "text", "text": ...}]`, which is the
form Raven sends; without it, plain strings. `--tools` adds one tool definition to every request, as Raven
offers tools on every turn. `--lengths` overrides the prefix lengths, in characters.
"""

import json
import pathlib
import sys
import time
import uuid

import requests

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "extend_results.jsonl"
CONTEXT_LENGTH = 32768
MAX_TOKENS = 4
# Long ordinary text from this repository, cut to each length. About four characters to a token.
ROOT = HERE.parent.parent
TEXT = "\n\n".join((ROOT / path).read_text(encoding="utf-8")
                   for path in ("raven/librarian/README.md", "README.md", "raven/visualizer/README.md"))
LENGTHS_CHARS = [8000, 16000, 32000, 48000, 64000, 96000]


def post(url: str, payload: dict, timeout: float = 900) -> dict:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


TOOL = {"type": "function",
        "function": {"name": "search_documents", "description": "Search the user's documents.",
                     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}


def ask(base: str, model: str, messages: list[dict], *, parts: bool, tools: bool) -> dict:
    if parts:
        messages = [{**m, "content": [{"type": "text", "text": m["content"]}]} for m in messages]
    payload = {"model": model, "messages": messages, "max_tokens": MAX_TOKENS, "temperature": 0.7,
               "reasoning_effort": "none", "stream": False}
    if tools:
        payload["tools"] = [TOOL]
    return post(f"{base}/api/v0/chat/completions", payload)


def main() -> None:
    args = sys.argv[1:]
    parts, tools = "--parts" in args, "--tools" in args
    lengths = LENGTHS_CHARS
    if "--lengths" in args:
        lengths = [int(n) for n in args[args.index("--lengths") + 1].split(",")]
        del args[args.index("--lengths"):args.index("--lengths") + 2]
    args = [a for a in args if a not in ("--parts", "--tools")]
    base, models = args[0].rstrip("/"), args[1:]
    assert len(TEXT) >= max(LENGTHS_CHARS), f"only {len(TEXT)} characters of text for {max(LENGTHS_CHARS)}"
    with RESULTS.open("a", encoding="utf-8") as results:
        def record(**fields) -> None:
            results.write(json.dumps({"t": time.time(), **fields}) + "\n")
            results.flush()

        for model in models:
            listed = [m for m in requests.get(f"{base}/api/v1/models", timeout=30).json()["models"] if m["key"] == model]
            was_loaded = bool(listed and listed[0].get("loaded_instances"))
            loaded = None
            if not was_loaded:
                loaded = post(f"{base}/api/v1/models/load", {"model": model, "context_length": CONTEXT_LENGTH, "parallel": 1})
                listed = [m for m in requests.get(f"{base}/api/v1/models", timeout=30).json()["models"] if m["key"] == model]
            record(event="model", model=model, was_loaded=was_loaded, parts=parts, tools=tools, instances=listed[0]["loaded_instances"] if listed else None)
            print(f"{model}: {'already loaded' if was_loaded else 'loaded'}, parts={parts}, tools={tools}", flush=True)
            try:
                for n_chars in lengths:
                    system = {"role": "system", "content": f"Session {uuid.uuid4().hex}.\n\n{TEXT[:n_chars]}"}
                    first = [system, {"role": "user", "content": "In one word, what is the text above about?"}]
                    requests_by_kind = {
                        "cold": first,
                        "repeat": first,
                        "extend": first + [{"role": "assistant", "content": "Software."},
                                           {"role": "user", "content": "And in two words?"}],
                        "tail": [system, {"role": "user", "content": "In one word, what is the text above for?"}],
                    }
                    line = []
                    for kind, messages in requests_by_kind.items():
                        if kind == "tail":  # back to the cold prompt, so the tail is measured from it
                            ask(base, model, first, parts=parts, tools=tools)
                        out = ask(base, model, messages, parts=parts, tools=tools)
                        ttft = (out.get("stats") or {}).get("time_to_first_token")
                        record(event="request", model=model, n_chars=n_chars, kind=kind, parts=parts, tools=tools,
                               prompt_tokens=(out.get("usage") or {}).get("prompt_tokens"), ttft=ttft, response=out)
                        line.append(f"{kind} {ttft:.2f}s" if ttft is not None else f"{kind} ?")
                    print(f"  {n_chars:6d} chars: " + ", ".join(line), flush=True)
            finally:
                if loaded is not None:
                    post(f"{base}/api/v1/models/unload", {"instance_id": loaded["instance_id"]})


if __name__ == "__main__":
    main()
