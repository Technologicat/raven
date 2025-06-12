"""Networking-related utilities.

This module is licensed under the 2-clause BSD license.
"""

__all__ = ["multipart_x_mixed_replace_payload_extractor"]

from typing import Generator, Iterator, Optional, Tuple

from unpythonic.net.util import ReceiveBuffer

def multipart_x_mixed_replace_payload_extractor(source: Iterator[bytes],
                                                boundary_prefix: str,
                                                expected_mimetype: Optional[str]) -> Generator[Tuple[Optional[str], bytes], None, None]:
    """Instantiate a generator that yield payloads from `source`, which is reading from a "multipart/x-mixed-replace" stream.

    The yielded value is the tuple `(received_mimetype, payload)`, where `received_mimetype` is set to whatever the server
    sent in the Content-Type header. If Content-Type was not set, then `received_mimetype is None`.

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

    def read_headers() -> int:
        """Read and validate headers for one payload. Return the length of the payload body, in bytes."""
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
        for field in headers[1:]:
            field = field.decode("utf-8")
            field_name, field_value = [text.strip().lower() for text in field.split(":")]
            if field_name == "content-type":
                if expected_mimetype is not None and field_value != expected_mimetype:  # wrong type of data?
                    raise ValueError(f"multipart_x_mixed_replace_payload_extractor.read_headers: expected mimetype '{expected_mimetype}', got '{field_value}'")
                received_mimetype = field_value
            if field_name == "content-length":
                body_length_bytes = int(field_value)  # and let it raise if the value is invalid
        if expected_mimetype is not None and received_mimetype is None:
            raise ValueError(f"read_headers: payload is missing the 'Content-Type' header (mandatory when `expected_mimetype` is specified; it is '{expected_mimetype}')")
        if body_length_bytes is None:
            raise ValueError("read_headers: payload is missing the 'Content-Length' header (mandatory for this client)")
        payload_buffer.set(start_of_body)
        return received_mimetype, body_length_bytes

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
        received_mimetype, body_length_bytes = read_headers()
        payload = read_body(body_length_bytes)
        yield received_mimetype, payload
