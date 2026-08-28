"""What does a larger retrieval `k` cost per chat turn?

The recall curve says the gold document is often present but deep — 74.7% within k=20 against 89.9%
within k=100 on hydrogen — so raising `k` is the one lever with a large measured effect and no new
model. What stops it is not the context window (128k is ample) but **prefill time**: the retrieved block
changes every turn, so it can never be cached, and the model re-reads all of it before answering.

This measures that, per `k`, as **time to first token** on a streaming request. TTFT is dominated by
prefill at these prompt sizes, it is what the user actually waits through, and it needs no
backend-specific timing fields — so the number survives a change of backend.

**Run it when nothing else is using the GPU.** An indexing run, an avatar, or a second model loaded will
inflate every figure and the comparison between `k` values is what matters here. (Measured against a
live index build once by accident; the numbers were useless.)

Raven already injects retrieved context in the cheapest available position — immediately before the
user's latest message, as a synthetic tool call — so the chat history stays cacheable and only the
retrieved block plus the new message are re-read. That is what this measures: the irreducible part.

    python prefill_cost.py [--db-dir DIR] [--corpus hydrogen] [--k 20,50,100,200] [--repeats 3]
"""

from __future__ import annotations

__all__ = ["K_VALUES", "measure_ttft"]

import concurrent.futures
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sharpness  # noqa: E402

from raven.client import api as client_api  # noqa: E402
from raven.client import config as client_config  # noqa: E402
from raven.librarian import config as librarian_config  # noqa: E402
from raven.librarian import hybridir  # noqa: E402
from raven.librarian import llmclient  # noqa: E402

K_VALUES = (20, 50, 100, 200)


def measure_ttft(llm_settings, messages: list[dict]) -> float:
    """Seconds for a request capped at one generated token — i.e. prefill plus a single decode step.

    Capping generation is what makes this a *prefill* measurement rather than a whole-reply one:
    decoding speed does not vary with `k`, so letting the model finish would add minutes of noise per
    sample and bury the effect being measured. One token is not free, but it is the same constant in
    every row, so the comparison across `k` is unaffected.

    Tools are disabled: a tool definition block is a fixed prompt cost that would sit in every
    measurement without depending on `k`.
    """
    t0 = time.perf_counter()
    llmclient.invoke(llm_settings, messages, tools_enabled=False, max_tokens=1)
    return time.perf_counter() - t0


def main() -> None:  # pragma: no cover
    argv = sys.argv[1:]

    def opt(name, default):
        if name in argv:
            at = argv.index(name)
            value = argv[at + 1]
            del argv[at:at + 2]
            return value
        return default

    corpus = opt("--corpus", "hydrogen")
    db_dir = pathlib.Path(opt("--db-dir", str(pathlib.Path.home() / f".config/raven/librarian/rag_index_{corpus}"))).expanduser()
    k_values = tuple(int(x) for x in opt("--k", ",".join(str(k) for k in K_VALUES)).split(","))
    repeats = int(opt("--repeats", "3"))

    items = [i for i in sharpness.build_workload(corpus)[0] if i["on_corpus"]][:repeats]
    if not items:
        print("no questions available for this corpus")
        return

    ex = concurrent.futures.ThreadPoolExecutor()
    client_api.initialize(raven_server_url=client_config.raven_server_url,
                          raven_api_key_file=client_config.raven_api_key_file, executor=ex)
    hybridir.init(executor=ex)
    retriever = hybridir.HybridIR(datastore_base_dir=db_dir,
                                  embedding_model_name=librarian_config.qa_embedding_model)
    llm_settings = llmclient.setup(backend_url=librarian_config.llm_backend_url)
    print(f"  backend: {librarian_config.llm_backend_url}")
    print(f"  model:   {getattr(llm_settings, 'model', '?')}")
    print(f"  corpus:  {corpus}   index: {db_dir}")
    print()

    results = {}
    for k in k_values:
        samples, token_counts = [], []
        for item in items:
            docs, _rep = retriever.query(item["query"], k=k, multi_query=False, return_extra_info=True)
            context = "\n\n".join(d.get("text", "") for d in docs)
            messages = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful research assistant."}]},
                        {"role": "user", "content": [{"type": "text",
                                                      "text": f"Context:\n{context}\n\nQuestion: {item['query']}"}]}]
            n_tokens, _ok = llmclient.count_tokens(llm_settings, context)
            token_counts.append(n_tokens)
            dt = measure_ttft(llm_settings, messages)
            samples.append(dt)
            print(f"    k={k:<4} {dt:6.2f} s   {token_counts[-1]:>7} tokens", flush=True)
        results[k] = {"ttft_s": samples, "tokens": token_counts}

    print()
    print(f"  {'k':>5} {'median TTFT':>12} {'median tokens':>14} {'ms/token':>10}")
    for k in k_values:
        r = results[k]
        t = statistics.median(r["ttft_s"])
        n = statistics.median(r["tokens"])
        print(f"  {k:>5} {t:>11.2f}s {n:>14.0f} {1000 * t / max(n, 1):>10.2f}")

    out = pathlib.Path(__file__).parent / f"prefill_cost_{corpus}.json"
    out.write_text(json.dumps({"corpus": corpus, "results": {str(k): v for k, v in results.items()}},
                              indent=1), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
