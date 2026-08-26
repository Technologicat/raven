"""Networking-related utilities.

This module is licensed under the 2-clause BSD license.
"""

__all__ = ["multipart_x_mixed_replace_payload_extractor",
           "pack_parameters_into_json_file_attachment", "unpack_parameters_from_json_file_attachment",
           "extract_urls", "url_host", "host_matches_allowlist",

           "Abort", "Aborted"]

import _socket  # the C-level socket type, for the one close Python's wrapper will not perform; see `Abort.abort`
import io
import json
import logging
import re
import socket
import sys
import threading
from collections.abc import Generator, Iterator
from typing import Any
import urllib.parse

import requests

from unpythonic.net.util import ReceiveBuffer

logger = logging.getLogger(__name__)

class Aborted(Exception):
    """Raised in the thread reading a stream that an `Abort` handle abandoned.

    Distinct from a connection failure, which the same underlying error would otherwise look like: this one
    means we asked, so it is not a fault and callers should not report it as one.
    """

def _maybe_socket_of(response: requests.Response) -> socket.socket | None:
    """Return the socket underneath a streaming `requests` response, or `None` if it can't be reached.

    `None` means only that this response's internals are not the shape we know; it is never an error.
    """
    # Four layers of private attribute, across `requests`, `urllib3` and `http.client`. There is no public
    # spelling: `Response.close()` is the API for this and it does not do the job (see `Abort.abort`).
    # Guarded rather than trusted, because a version bump on any of the three can change the shape, and an
    # abandoned request that cannot be abandoned is a missed optimization where an AttributeError raised
    # mid-stream would be a crash.
    try:
        return response.raw._fp.fp.raw._sock
    except AttributeError:
        logger.warning("_maybe_socket_of: cannot reach the socket under this response; abort will be a no-op")
        return None

class Abort:
    """A handle for abandoning an in-flight streaming `requests` call from another thread.

    A thread blocked reading a response cannot see a cancellation flag: it is inside a socket read, and
    will not look at anything until that read returns — which, for a backend that has gone quiet, means the
    read timeout. This handle is what ends the wait from outside.

    Hand one to whatever issues the request, have it call `arm` once the response exists and `disarm` when
    the stream is done, and call `abort` from anywhere to give up. The blocked reader then raises some
    connection error, which the issuing code turns into `Aborted` by asking `aborted` whose doing it was.

    `abort` is idempotent, safe from any thread, never raises, and returns immediately — so it is safe to
    call from a GUI callback. Aborting before the request is even sent is fine: the abort is remembered and
    applied as soon as there is something to apply it to.

    One handle serves one request. Aborting is permanent, so a handle that has been used is spent.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._aborted = False
        self._maybe_response = None

    def _get_aborted(self) -> bool:
        """Whether `abort` has been called on this handle."""
        with self._lock:
            return self._aborted

    aborted = property(fget=_get_aborted,
                       doc="Whether this handle has been aborted. Once `True`, it stays `True`.")

    def arm(self, response: requests.Response) -> None:
        """Register the response to abandon. Call this as soon as the request has one."""
        with self._lock:
            self._maybe_response = response
            already_aborted = self._aborted
        if already_aborted:  # the abort arrived while we were still waiting for the response headers
            self.abort()

    def disarm(self) -> None:
        """Forget the response. Call this when the stream is finished, however it ended."""
        with self._lock:
            self._maybe_response = None

    def abort(self) -> None:
        """Abandon the request this handle was armed with. Safe from any thread; never raises."""
        with self._lock:
            self._aborted = True
            maybe_response = self._maybe_response
        if maybe_response is None:  # nothing in flight yet, or it already finished
            return
        maybe_sock = _maybe_socket_of(maybe_response)
        if maybe_sock is None:
            return
        # Going at the socket rather than at the response, and the difference is not cosmetic. Measured
        # 2026-08-26 (`investigations/abort-inflight-request/`): `Response.close()`, `raw.close()` and
        # `raw._fp.fp.close()` all leave the reader blocked in `recv` *and* block the thread that called them
        # until the read timeout expires — 59.86 s against a 60 s timeout, which from a GUI callback is a
        # minute of frozen app.
        #
        # Both calls, because the platforms disagree about which one ends a blocked read, and each is
        # harmless where the other is the effective one:
        #
        #   - `shutdown` wakes the reader at once on Linux and macOS. On Windows it does not (Winsock
        #     cancels pending operations on `closesocket`, not on `shutdown`) — which is a CI failure on
        #     windows-latest, not a reading of the documentation.
        #   - `close` is what Winsock acts on. On Linux it returns promptly but does *not* wake the reader,
        #     so it cannot stand alone either.
        #
        try:
            maybe_sock.shutdown(socket.SHUT_RDWR)
        except OSError:  # already closed, or the peer got there first: either way there is nothing to abandon
            pass
        if sys.platform == "win32":
            # `socket.close()` is not the close Winsock needs to see. urllib3 reads the response through a
            # `makefile()` wrapper, so the socket carries an outstanding io-ref, and Python's `close` then
            # only marks the object closed and defers the real `closesocket` until that wrapper goes --
            # which is the reader we are trying to interrupt. Measured 2026-08-26: `_io_refs = 1` at the
            # moment of the abort. So the wrapper's `close` is a no-op here, and the C-level one is the
            # call that actually reaches the handle.
            #
            # POSIX is deliberately excluded rather than merely not needing it: `shutdown` has already woken
            # the reader there, and closing the descriptor under a thread that is still unwinding is how a
            # later reader ends up reading a descriptor that has been reused. On Windows that trade is
            # forced, because nothing else ends the wait.
            try:
                _socket.socket.close(maybe_sock)
            except OSError:
                pass

def multipart_x_mixed_replace_payload_extractor(source: Iterator[bytes],
                                                boundary_prefix: str,
                                                expected_mimetype: str | None) -> Generator[tuple[str | None, dict[str, str], bytes], None, None]:
    """Instantiate a generator that yields payloads from `source`, which is reading from a "multipart/x-mixed-replace" stream.

    The yielded value is the tuple `(received_mimetype, extra_headers, payload)`:

    - `received_mimetype` is whatever the server sent in the Content-Type header, or `None` if not sent.
    - `extra_headers` is a lowercase-keyed dict of every non-`Content-*` header from the part (e.g. custom
      `X-*` headers). Values are the raw strings as sent. Empty dict if the part had no extras.
    - `payload` is the body bytes.

    The server MUST send the Content-Length header for this reader to work. If it is missing, `ValueError` is raised.

    If `expected_mimetype` is provided, the server MUST send the Content-Type header, and it must match `expected_mimetype`,
    e.g. "image/png". If it is missing or does not match, `ValueError` is raised.

    If `expected_mimetype` is not provided, this reader does not care about the Content-Type header.

    Loosely based on `unpythonic.net.msg.decodemsg`.
    """
    stream_iterator = iter(source)
    boundary_prefix = boundary_prefix.encode()  # str -> bytes
    payload_buffer = ReceiveBuffer()

    def read_more_input() -> None:
        try:
            data = next(stream_iterator)
        except StopIteration:
            raise EOFError
        payload_buffer.append(data)

    def synchronize() -> None:
        """Synchronize `payload_buffer` to the start of the next payload boundary marker (e.g. "--frame")."""
        while True:
            val = payload_buffer.getvalue()
            idx = val.rfind(boundary_prefix)
            if idx != -1:
                junk, start_of_payload = val[:idx], val[idx:]  # noqa: F841
                payload_buffer.set(start_of_payload)
                return
            # Clear the receive buffer after each chunk that didn't have a sync
            # marker in it. This prevents a malicious sender from crashing the
            # receiver by flooding it with nothing but junk.
            payload_buffer.set(b"")
            read_more_input()

    def read_headers() -> tuple[str | None, dict[str, str], int]:
        """Read and validate headers for one payload.

        Return `(received_mimetype, extra_headers, body_length_bytes)`.
        """
        while True:
            val = payload_buffer.getvalue()
            end_of_headers_idx = val.find(b"\r\n\r\n")
            if end_of_headers_idx != -1:  # headers completely streamed? (have a blank line at the end)
                break
        headers, start_of_body = val[:end_of_headers_idx], val[end_of_headers_idx + 4:]
        headers = headers.split(b"\r\n")
        if headers[0] != boundary_prefix:  # after sync, we should always have the payload boundary marker at the start of the buffer
            assert False
        received_mimetype = None
        body_length_bytes = None
        extra_headers: dict[str, str] = {}
        for field in headers[1:]:
            field = field.decode("utf-8")
            # Split on the first colon only: header values (e.g. a JSON-encoded dict) may contain colons.
            field_name, _, field_value = field.partition(":")
            field_name = field_name.strip().lower()
            field_value = field_value.strip()
            if field_name == "content-type":
                lower_value = field_value.lower()
                if expected_mimetype is not None and lower_value != expected_mimetype:  # wrong type of data?
                    raise ValueError(f"multipart_x_mixed_replace_payload_extractor.read_headers: expected mimetype '{expected_mimetype}', got '{lower_value}'")
                received_mimetype = lower_value
            elif field_name == "content-length":
                body_length_bytes = int(field_value)  # and let it raise if the value is invalid
            elif field_name.startswith("content-"):
                pass  # any other Content-* headers are transport details; drop them
            else:
                extra_headers[field_name] = field_value
        if expected_mimetype is not None and received_mimetype is None:
            raise ValueError(f"read_headers: payload is missing the 'Content-Type' header (mandatory when `expected_mimetype` is specified; it is '{expected_mimetype}')")
        if body_length_bytes is None:
            raise ValueError("read_headers: payload is missing the 'Content-Length' header (mandatory for this client)")
        payload_buffer.set(start_of_body)
        return received_mimetype, extra_headers, body_length_bytes

    def read_body(body_length_bytes: int) -> bytes:
        """Read the payload body and return it as a `bytes` object."""
        while True:
            val = payload_buffer.getvalue()
            if len(val) >= body_length_bytes:
                break
            read_more_input()
        body, leftovers = val[:body_length_bytes], val[body_length_bytes:]
        payload_buffer.set(leftovers)
        return body

    while True:
        synchronize()
        received_mimetype, extra_headers, body_length_bytes = read_headers()
        payload = read_body(body_length_bytes)
        yield received_mimetype, extra_headers, payload

def pack_parameters_into_json_file_attachment(parameters: dict[str, Any]) -> str:
    """Pack API call parameters from a `dict`, for sending in the request as a JSON file attachment.

    The return value can be used as a value in the `files` argument of a `requests.post` call::

        files={"my_param_file": pack_parameters_into_json_file_attachment({param_name0: value0, ...}),
               "my_data_file": ...}

    This is meant for endpoints that on the server side receive "multipart/form-data" because
    they need a file input, but also simultenously need a JSON input to pass some API call parameters.

    The counterpart is `unpack_parameters_from_json_file_attachment`.
    """
    return ("parameters.json", json.dumps(parameters, indent=2), "application/json")

def unpack_parameters_from_json_file_attachment(stream) -> dict[str, Any]:
    """Return API call parameters as `dict`, that came in the request as a JSON file.

    `stream`: the `request.files["my_param_file"].stream`.

    Returns a dictionary `{param_name0: value0, ...}`.

    This is meant for endpoints that receive "multipart/form-data" because they need a file input,
    but also simultenously need a JSON input to pass some API call parameters.

    The counterpart is `pack_parameters_into_json_file_attachment`.
    """
    # TODO: Do we need to run this through a `BytesIO` to copy the data? Probably not?
    # The internet says that in some versions of Flask, touching most of the attributes
    # of a `FileStorage` causes a disk write to a temporary file, but `.stream` can be
    # safely accessed in-memory.
    buffer = io.BytesIO()
    buffer.write(stream.read())
    parameters_bytes = buffer.getvalue()
    parameters_python = json.loads(parameters_bytes)

    # # Simpler way without `BytesIO`:
    # parameters_filestorage = request.files["json"]
    # parameters_bytes = parameters_filestorage.read()
    # parameters_python = json.loads(parameters_bytes)

    return parameters_python

# --------------------------------------------------------------------------------
# URL / host utilities

# Match http(s) URLs in free text. We stop at whitespace and at characters that commonly
# *delimit* a URL rather than belong to it — quotes, angle brackets, and the closing
# delimiters of Markdown/parenthetical contexts (`)`, `]`) — so a URL inside `[text](url)`
# or "(see https://example.com)" extracts cleanly. Trailing sentence punctuation is trimmed
# separately in `extract_urls`.
_URL_RE = re.compile(r"""https?://[^\s<>"'`)\]}]+""", re.IGNORECASE)

# Trailing characters trimmed from a matched URL: sentence punctuation that almost never
# ends a real URL but routinely follows one in prose.
_URL_TRAILING_TRIM = ".,;:!?"

def extract_urls(text: str) -> list[str]:
    """Return the http(s) URLs found in `text`, in order of appearance (duplicates kept).

    Trailing sentence punctuation (`.`, `,`, `;`, `:`, `!`, `?`) is trimmed from each match.
    Intended for pulling URLs out of chat prose (e.g. a user-typed message), not for
    validating or normalizing them.
    """
    return [m.group(0).rstrip(_URL_TRAILING_TRIM) for m in _URL_RE.finditer(text)]

def url_host(url: str) -> str:
    """Return the lowercased host of `url`, or "" if it has none (e.g. a relative or malformed URL)."""
    return (urllib.parse.urlsplit(url).hostname or "").lower()

def host_matches_allowlist(host: str, allowlist: list[str]) -> bool:
    """Return whether `host` is permitted by `allowlist`.

    An allowlist entry matches case-insensitively, in one of two ways:

    - Exact: `"example.com"` matches only `example.com`.
    - Wildcard: `"*.example.com"` matches the apex `example.com` *and* any subdomain
      (`sub.example.com`). The apex match matters in practice — e.g. `*.arxiv.org` must
      admit a bare `arxiv.org` host (the arXiv HTML rewrite targets `arxiv.org/html/...`).

    An empty `host` never matches.
    """
    if not host:
        return False
    host = host.lower()
    for entry in allowlist:
        entry = entry.lower()
        if entry.startswith("*."):
            apex = entry[2:]
            if host == apex or host.endswith(f".{apex}"):
                return True
        elif host == entry:
            return True
    return False
