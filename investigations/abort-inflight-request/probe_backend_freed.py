"""After aborting a prefill, is the backend actually free — or still chewing the abandoned prompt?

This is the point of the exercise. Waking the client thread is worth little on its own: what the user
waits for is the backend, so a real turn must not queue behind speculative work that has been abandoned.

Starts a cold ~90k-token prefill, aborts it partway, then immediately times a small request. If the
backend dropped the abandoned work, that small request answers in well under a second; if it did not, it
waits out the remaining prompt processing.

Sends real requests and occupies the backend for several seconds.

Usage::

    python probe_backend_freed.py [--url URL] [--model MODEL]
"""

import argparse
import socket
import threading
import time

import requests

DEFAULT_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_MODEL = "qwen3.6-35b-a3b"
ABORT_AFTER = 5.0
FREED_THRESHOLD = 3.0  # a follow-up faster than this cannot have queued behind the abandoned prefill

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_URL, help="OpenAI-compatible chat-completions endpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model to load the prompt into")
    args = parser.parse_args()

    # Unique filler, so the backend cannot answer from a warm KV cache and really pays for the prompt.
    big_prompt = " ".join(f"charlie{i:05d}" for i in range(15000))
    data = {"model": args.model,
            "messages": [{"role": "user", "content": big_prompt}],
            "max_tokens": 1,
            "stream": True}

    t0 = time.monotonic()
    response = requests.post(args.url, json=data, stream=True, timeout=300)

    def aborter() -> None:
        time.sleep(ABORT_AFTER)
        response.raw._fp.fp.raw._sock.shutdown(socket.SHUT_RDWR)
    threading.Thread(target=aborter, daemon=True).start()

    try:
        for _ in response.iter_lines():
            pass
    except Exception as exc:  # noqa: BLE001 -- reporting what happened *is* the measurement
        print(f"{time.monotonic() - t0:7.3f}s  prefill aborted: {type(exc)}")

    t1 = time.monotonic()
    follow_up = requests.post(args.url, timeout=120,
                              json={"model": args.model,
                                    "messages": [{"role": "user", "content": "Say OK."}],
                                    "max_tokens": 3,
                                    "stream": False})
    dt = time.monotonic() - t1
    print(f"{dt:7.3f}s  small follow-up request answered (status {follow_up.status_code})")
    print()
    print("-> backend was", "FREED" if dt < FREED_THRESHOLD else "STILL BUSY with the abandoned prefill")

if __name__ == "__main__":
    main()
