"""Helpers shared by Raven-server's endpoints, the server-side counterpart of `raven.client.util`."""

__all__ = ["stream_job"]

import logging
logger = logging.getLogger(__name__)

import json
from collections.abc import Callable

from flask import Response, request


def stream_job(job: Callable[[Callable[[], bool]], dict]) -> Response:
    """Answer with `job`'s result as JSON, streamed, so that the client holds the response from the start.

    For an endpoint whose answer is a JSON result that takes a while to compute — a web search, a page
    fetch. Not for one answering with an image or audio, or with a stream of its own (the avatar's frames,
    TTS), which have their own shapes. The client side is `raven.client.util.post_streamed_job`.

    `job`: called as `job(is_cancelled)`, returning the result. `is_cancelled()` answers whether the client
           has gone; a job worth abandoning checks it between its steps and stops early, and what it
           returns then is never sent.

    The response opens with a space, which a JSON parser skips. A failure in `job` cannot become a status
    once that has gone out, so it arrives in the body instead, as `{"error": "<type>: <message>"}`.
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

    def generate():
        yield b" "  # sends the headers, so the client has something to abandon
        try:
            result = job(is_cancelled)
        except Exception as exc:  # noqa: BLE001 -- reported to the client in the body, the only channel left
            if is_cancelled():
                logger.info(f"stream_job: the client has gone; the job stopped with {type(exc)}: {exc}")
                return
            logger.error(f"stream_job: the job failed: {type(exc)}: {exc}", exc_info=True)
            result = {"error": f"{type(exc).__name__}: {exc}"}
        if is_cancelled():
            logger.info("stream_job: the client has gone; not sending the result.")
            return
        yield json.dumps(result).encode("utf-8")

    return Response(generate(), mimetype="application/json")
