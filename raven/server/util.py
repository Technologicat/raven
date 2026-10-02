"""Helpers shared by Raven-server's endpoints, the server-side counterpart of `raven.client.util`."""

__all__ = ["stream_job"]

import logging
logger = logging.getLogger(__name__)

import json
import queue
import threading
from collections.abc import Callable

from flask import Response, request


def stream_job(job: Callable[[Callable[[], bool], Callable[[str], None]], dict]) -> Response:
    """Answer with `job`'s progress and then its result, streamed, so that the client holds the response from the start.

    For an endpoint whose answer is a JSON result that takes a while to compute — a web search, a page
    fetch. Not for one answering with an image or audio, or with a stream of its own (the avatar's frames,
    TTS), which have their own shapes. The client side is `raven.client.util.post_streamed_job`.

    `job`: called as `job(is_cancelled, report)`, returning the result. `is_cancelled()` answers whether the
           client has gone; a job worth abandoning checks it between its steps and stops early, and what it
           returns then is never sent. `report(text)` tells the client which step the job is on, as a short
           human-readable line such as `"Loading the page…"`; the client may show it.

    The response opens with a space, which a JSON parser skips. Then come newline-terminated JSON records:
    any number of `{"progress": "<text>"}`, and last either `{"result": <the result>}` or, when `job` raised,
    `{"error": "<type>: <message>"}` — a failure cannot become a status once the space has gone out.
    Validate the request before calling this, while a status can still say what was wrong with it.

    Call it from a request handler. `is_cancelled` answers `False` throughout when the server is not
    waitress, which is the only server that says.
    """
    # Why stream at all: answered with one body at the end, the client holds no response while the job
    # runs, so it has nothing to close when it stops waiting, and the job has nobody to ask. Streamed, the
    # headers go out at once. It takes two settings in `raven.server.app` to work: Flask-Compress must not
    # compress streams (the compressor holds back the first bytes, and the headers with them), and waitress
    # needs a request lookahead (without one, `waitress.client_disconnected` never sees a client go). Both
    # measured in `investigations/abort-inflight-request/`.
    #
    # Read in the request context, which the generator below runs outside of.
    maybe_client_disconnected = request.environ.get("waitress.client_disconnected")

    def is_cancelled() -> bool:
        return maybe_client_disconnected is not None and maybe_client_disconnected()

    def record(**fields) -> bytes:
        return json.dumps(fields).encode("utf-8") + b"\n"

    def generate():
        yield b" "  # sends the headers, so the client has something to abandon

        # The job runs on a thread of its own so that this generator is free to send its progress as it comes:
        # a `report` called from inside the job cannot yield from here.
        events: queue.Queue = queue.Queue()
        def run() -> None:
            try:
                events.put(("result", job(is_cancelled, lambda text: events.put(("progress", text)))))
            except Exception as exc:  # noqa: BLE001 -- reported to the client in the body, the only channel left
                events.put(("exception", exc))
        threading.Thread(target=run, name="stream_job", daemon=True).start()

        while True:
            try:
                kind, value = events.get(timeout=0.1)
            except queue.Empty:
                if is_cancelled():  # nobody to send to; the job sees the same and stops at its next step
                    logger.info("stream_job: the client has gone while the job ran.")
                    return
                continue
            if kind == "progress":
                yield record(progress=value)
                continue
            if is_cancelled():
                if kind == "exception":
                    logger.info(f"stream_job: the client has gone; the job stopped with {type(value)}: {value}")
                else:
                    logger.info("stream_job: the client has gone; not sending the result.")
                return
            if kind == "exception":
                logger.error(f"stream_job: the job failed: {type(value)}: {value}", exc_info=value)
                yield record(error=f"{type(value).__name__}: {value}")
            else:
                yield record(result=value)
            return

    return Response(generate(), mimetype="application/x-ndjson")
