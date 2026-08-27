"""A backend proxy that can fail a turn on demand, so "what happens when the backend errors" is testable.

Sits between Librarian and the real OpenAI-compatible backend and forwards everything verbatim, except
that `/v1/chat/completions` consults a control file first. That makes the *moment* of the failure something
the test chooses, which is the whole point: the interesting cases are the ones where the failure lands while
the user is somewhere else.

Control file (default `/tmp/faultproxy.mode`), read per request:

    pass            forward to the real backend (default when the file is absent)
    error           answer 200 and then an SSE `event: error`, the way LM Studio reports a backend fault
    error:<secs>    the same, after stalling that many seconds -- long enough to navigate away first
    hang:<secs>     headers, then silence, then close: a backend that stops talking mid-turn

Usage::

    python faultproxy.py --port 8998 --upstream http://maia.local:1234
    raven-librarian --backend-url http://127.0.0.1:8998

Then, from anywhere::

    echo 'error:8' > /tmp/faultproxy.mode
"""

import argparse
import http.server
import json
import pathlib
import time
import urllib.request

CONTROL_FILE = pathlib.Path("/tmp/faultproxy.mode")

def read_mode() -> tuple[str, float]:
    """Return `(mode, delay_seconds)` from the control file. Absent or unreadable means pass-through."""
    try:
        raw = CONTROL_FILE.read_text().strip()
    except OSError:
        return ("pass", 0.0)
    if not raw:
        return ("pass", 0.0)
    mode, _, delay = raw.partition(":")
    try:
        return (mode, float(delay) if delay else 0.0)
    except ValueError:
        return (mode, 0.0)

class Proxy(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream = "http://localhost:1234"

    def log_message(self, fmt, *args):
        print(f"[faultproxy] {fmt % args}", flush=True)

    def _send_sse_error(self, delay: float) -> None:
        """Answer the way a backend reporting a fault mid-stream does: 200, then an SSE error event."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.wfile.flush()
        if delay:
            self.log_message("stalling %.1fs before the error", delay)
            time.sleep(delay)
        body = json.dumps({"error": {"message": "faultproxy: synthetic backend failure"}})
        for line in ("event: error\r\n", f"data: {body}\r\n", "\r\n"):
            chunk = line.encode()
            self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _hang(self, delay: float) -> None:
        """Headers, then silence, then close — a backend that stops talking without saying so."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.wfile.flush()
        time.sleep(delay or 30.0)
        self.close_connection = True

    def _forward(self, body: bytes) -> None:
        request = urllib.request.Request(f"{self.upstream}{self.path}", data=body,
                                         method=self.command,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:  # noqa: S310 -- the upstream is ours, from the command line
            self.send_response(response.status)
            self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = response.read(1024)
                if not chunk:
                    break
                self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    def _handle(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        mode, delay = read_mode()
        if self.path.endswith("/chat/completions") and mode != "pass":
            self.log_message("intercepting %s as mode=%s delay=%.1f", self.path, mode, delay)
            if mode == "error":
                self._send_sse_error(delay)
            elif mode == "hang":
                self._hang(delay)
            else:
                self.log_message("unknown mode %r; forwarding instead", mode)
                self._forward(body)
            return
        self._forward(body)

    do_POST = _handle
    do_GET = _handle

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--upstream", default="http://localhost:1234")
    args = parser.parse_args()

    Proxy.upstream = args.upstream.rstrip("/")
    print(f"[faultproxy] {args.upstream} -> http://127.0.0.1:{args.port}  (control: {CONTROL_FILE})", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Proxy).serve_forever()

if __name__ == "__main__":
    main()
