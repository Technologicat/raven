"""Does a streamed Raven-server response reach the client at once, and can the server tell the client has gone?

Two questions behind making a slow server-side job (a web search, a page fetch) abandonable from both ends:

  1. Under waitress, does yielding a first byte from a Flask generator send the response headers
     immediately, so that the client holds a `Response` - and so a socket `netutil.Abort` can shut down -
     within moments rather than when the job finishes?
  2. After the client abandons the request, what tells the server? Waitress's `waitress.client_disconnected`
     callable, with and without `channel_request_lookahead`, and with and without the server writing
     keepalive bytes while it works.

Each variant runs its own waitress server on a free port, in a thread. The endpoint yields one byte, then
"works" for up to 10 s in 0.1 s steps, writing a keepalive byte every second if asked to, and records the
moment it first sees the client gone. The client aborts 1.5 s in. Everything is local; nothing is contacted.

Run: python probe_server_side.py
"""

import socket
import threading
import time

import requests
from flask import Flask, Response
from waitress.server import create_server

from raven.common import netutil

WORK_S = 10.0
ABORT_AT_S = 1.5


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_variant(lookahead: int, keepalive: bool) -> dict:
    app = Flask(__name__)
    seen = {}

    @app.route("/work", methods=["POST"])
    def work():
        from flask import request
        client_disconnected = request.environ.get("waitress.client_disconnected")
        seen["has_hook"] = client_disconnected is not None

        def generate():
            t0 = time.monotonic()
            seen["t0"] = t0
            yield b" "  # commits the headers, if waitress sends them on the first chunk
            last_keepalive = t0
            try:
                while time.monotonic() - t0 < WORK_S:
                    time.sleep(0.1)
                    if client_disconnected is not None and client_disconnected():
                        seen.setdefault("hook_says_gone_at", time.monotonic() - t0)
                        return
                    if keepalive and time.monotonic() - last_keepalive >= 1.0:
                        last_keepalive = time.monotonic()
                        yield b" "
                seen["finished_work"] = True
                yield b'{"done": true}'
            except GeneratorExit:  # waitress closes the iterator when a write fails
                seen.setdefault("generator_closed_at", time.monotonic() - t0)
                raise
        return Response(generate(), mimetype="application/json")

    port = free_port()
    server = create_server(app, host="127.0.0.1", port=port, channel_request_lookahead=lookahead)
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(0.3)

    abort = netutil.Abort()
    result = {"lookahead": lookahead, "keepalive": keepalive}
    t0 = time.monotonic()
    threading.Timer(ABORT_AT_S, abort.abort).start()
    try:
        response = requests.post(f"http://127.0.0.1:{port}/work", json={}, stream=True, timeout=(5, 30))
        result["headers_at"] = round(time.monotonic() - t0, 3)
        abort.arm(response)
        try:
            body = response.content
            result["client"] = f"finished, {len(body)} bytes"
        except Exception as exc:  # noqa: BLE001 -- probe: whatever the read raises is the finding
            result["client"] = f"read ended at {time.monotonic() - t0:.2f} s: {type(exc).__name__}"
        finally:
            abort.disarm()
    except Exception as exc:  # noqa: BLE001
        result["client"] = f"request failed: {type(exc).__name__}: {exc}"

    time.sleep(WORK_S + 1.0)  # let the server side run its course
    for key in ("has_hook", "hook_says_gone_at", "generator_closed_at", "finished_work"):
        if key in seen:
            value = seen[key]
            result[key] = round(value, 2) if isinstance(value, float) else value
    server.close()
    return result


if __name__ == "__main__":
    for lookahead in (0, 1):
        for keepalive in (False, True):
            print(run_variant(lookahead, keepalive), flush=True)
