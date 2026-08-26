"""On a real backend, where does the long prompt-processing wait sit — and can it be aborted?

If the wait falls *after* the response headers, the client holds a socket throughout the expensive part
and `socket.shutdown` can reach it. If it falls before them, there is nothing to hold and an abort has
nowhere to land. `probe_closers.py` establishes which closers work at all; this one establishes whether
the production case is the one they work on.

Two runs, with different filler so the first cannot warm the KV cache for the second:

  A: baseline. How long does a cold ~90k-token prefill take, and where is the gap?
  B: the same request, aborted partway through.

Sends real requests and occupies the backend for tens of seconds.

Usage::

    python probe_phases.py [--url URL] [--model MODEL]
"""

import argparse
import socket
import threading
import time

import requests

DEFAULT_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_MODEL = "qwen3.6-35b-a3b"
ABORT_AFTER = 5.0  # seconds into run B

def make_filler(salt: str, n_words: int = 15000) -> str:
    """Build a prompt of unique words, so the backend cannot answer from a warm KV cache.

    ~90k tokens at the default size, which fits a 131072-token context with room to spare. Each word
    tokenizes to several tokens, so the word count is well below the token count.
    """
    return " ".join(f"{salt}{i:05d}" for i in range(n_words))

def run(url: str, model: str, label: str, salt: str, abort_after: float | None = None) -> None:
    """Send one cold prefill, printing a timestamp per SSE line; optionally abort it partway."""
    data = {"model": model,
            "messages": [{"role": "user", "content": make_filler(salt)}],
            "max_tokens": 1,
            "stream": True}

    print(f"--- {label}")
    t0 = time.monotonic()
    response = requests.post(url, json=data, stream=True, timeout=300)
    t_headers = time.monotonic() - t0
    print(f"{t_headers:7.3f}s  headers (status {response.status_code})")

    if abort_after is not None:
        def aborter() -> None:
            time.sleep(abort_after)
            try:
                response.raw._fp.fp.raw._sock.shutdown(socket.SHUT_RDWR)
                print(f"{time.monotonic() - t0:7.3f}s  [aborter] shutdown() returned")
            except BaseException as exc:  # noqa: BLE001 -- reporting what happened *is* the measurement
                print(f"{time.monotonic() - t0:7.3f}s  [aborter] raised {type(exc)}: {exc}")
        threading.Thread(target=aborter, daemon=True).start()

    # Count *lines*, not bytes. An OpenAI-style stream emits a role delta immediately, so a probe that
    # stops at the first body byte measures the framing rather than the prompt-processing pass — which is
    # how an earlier version of this put the whole wait at 0.586 s and reported no gap at all.
    n_lines = 0
    outcome = "stream ended"
    try:
        for line in response.iter_lines():
            if not line:
                continue
            n_lines += 1
            t = time.monotonic() - t0
            if n_lines <= 2 or t - t_headers > 1.0:
                print(f"{t:7.3f}s  line {n_lines}: {line.decode('utf-8', 'replace')[:70]}")
            if n_lines > 6:
                break
    except BaseException as exc:  # noqa: BLE001 -- likewise
        outcome = f"{type(exc)}: {str(exc)[:60]}"
    print(f"{time.monotonic() - t0:7.3f}s  -> {outcome}\n")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_URL, help="OpenAI-compatible chat-completions endpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model to load the prompt into")
    args = parser.parse_args()

    run(args.url, args.model, "A baseline (no abort)", "alpha")
    run(args.url, args.model, f"B abort {ABORT_AFTER:g} s in", "bravo", abort_after=ABORT_AFTER)

if __name__ == "__main__":
    main()
