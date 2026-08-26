"""Which way of closing a streaming `requests` response actually aborts a blocked read?

Needs no backend: a local HTTP server supplies the stall.

A state-machine test rather than a timing one. The reader signals the moment it is genuinely blocked;
every ceiling is 15 s, so a failed approach costs seconds; the observation window is 1.5 s, because an
abort that works lands in milliseconds.

Measures two things per closer, not one:

  - does the READER wake?
  - does the CLOSER itself return promptly? A closer that blocks cannot be called from a GUI callback,
    whatever else it does.

Two cases, because they would need different fixes:

  A: blocked in the body read (headers arrived, no data yet) -> we hold the Response
  B: blocked in `requests.post` itself (no headers yet)      -> we hold nothing

`probe_phases.py` establishes which of the two a real backend puts us in.

Usage::

    python probe_closers.py
"""

import http.server
import socket
import threading
import time

import requests

CEILING = 15.0  # server stall and client read timeout alike; never reached, so a wait that ends is unambiguous
WATCH = 1.5  # how long we are willing to wait before calling it "did not wake"

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path.endswith("/nohdr"):  # case B: stall before responding at all
            time.sleep(CEILING)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.wfile.flush()
        time.sleep(CEILING)  # case A: headers out, body stalls

    def log_message(self, *args) -> None:  # keep the probe's own output readable
        pass

def socket_of(response: requests.Response) -> socket.socket:
    """Dig the raw socket out of a `requests` response.

    Four layers of private attribute across `requests`, `urllib3` and `http.client`. That is the point of
    the probe: the public `close()` does not do the job, so this is what the real implementation has to
    reach for (guarded).
    """
    return response.raw._fp.fp.raw._sock

CLOSERS = {"response.close()": lambda r: r.close(),
           "raw.close()": lambda r: r.raw.close(),
           "raw._fp.fp.close()": lambda r: r.raw._fp.fp.close(),
           "socket.close()": lambda r: socket_of(r).close(),
           "socket.shutdown()": lambda r: socket_of(r).shutdown(socket.SHUT_RDWR)}

def run_case(port: int, label: str, path: str, closer_name: str) -> None:
    """Block a reader on `path`, apply `closer_name` from another thread, and report what each thread did."""
    blocked = threading.Event()
    closed = threading.Event()
    holder = {}
    result = {}

    def reader() -> None:
        try:
            response = requests.post(f"http://127.0.0.1:{port}{path}", json={"x": 1},
                                     stream=True, timeout=CEILING)
            holder["response"] = response
            blocked.set()  # headers are in, so the next read is the blocking one
            for _ in response.iter_lines():
                pass
            result["outcome"] = "stream ended"
        except BaseException as exc:  # noqa: BLE001 -- reporting what happened *is* the measurement
            result["outcome"] = f"{type(exc)}: {str(exc)[:44]}"

    def closer() -> None:
        try:
            CLOSERS[closer_name](holder["response"])
        except BaseException as exc:  # noqa: BLE001 -- likewise
            result["closer"] = f"raised {type(exc)}"
        closed.set()

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()
    armed = blocked.wait(timeout=5.0)
    time.sleep(0.2)  # let it actually enter recv

    t0 = time.monotonic()
    if armed:
        threading.Thread(target=closer, daemon=True).start()
    reader_thread.join(timeout=WATCH)
    reader_woke = not reader_thread.is_alive()
    closer_returned = closed.wait(timeout=max(0.0, WATCH - (time.monotonic() - t0)))

    print(f"{label:14s} {closer_name:20s} armed={str(armed):5s} "
          f"reader_woke={str(reader_woke):5s} closer_returned={str(closer_returned):5s} "
          f"in {time.monotonic() - t0:4.2f}s  "
          f"{result.get('outcome', '(still blocked)')} {result.get('closer', '')}")

def main() -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    for closer_name in CLOSERS:
        run_case(port, "A body-stall", "/hdr", closer_name)
    run_case(port, "B header-stall", "/nohdr", "response.close()")

if __name__ == "__main__":
    main()
