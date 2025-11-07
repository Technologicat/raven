"""Networking-related utilities.

This module is licensed under the 2-clause BSD license.
"""

__all__ = ["multipart_x_mixed_replace_payload_extractor",
           "pack_parameters_into_json_file_attachment", "unpack_parameters_from_json_file_attachment"]

import io
import json
from typing import Any, Dict, Generator, Iterator, Optional, Tuple

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

def pack_parameters_into_json_file_attachment(parameters: Dict[str, Any]) -> str:
    """Pack API call parameters from a `dict`, for sending in the request as a JSON file attachment.

    The return value can be used as a value in the `files` argument of a `requests.post` call::

        files={"my_param_file": pack_parameters_into_json_file_attachment({param_name0: value0, ...}),
               "my_data_file": ...}

    This is meant for endpoints that on the server side receive "multipart/form-data" because
    they need a file input, but also simultenously need a JSON input to pass some API call parameters.

    The counterpart is `unpack_parameters_from_json_file_attachment`.
    """
    return ("parameters.json", json.dumps(parameters, indent=2), "application/json")

def unpack_parameters_from_json_file_attachment(stream) -> Dict[str, Any]:
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
