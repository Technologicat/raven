"""Unit tests for raven.common.netutil."""

import io
import json

import pytest

from raven.common import netutil


# --------------------------------------------------------------------------------
# JSON file-attachment pack/unpack

class TestPackJsonAttachment:
    def test_returns_three_tuple(self):
        packed = netutil.pack_parameters_into_json_file_attachment({"x": 1})
        assert isinstance(packed, tuple)
        assert len(packed) == 3

    def test_filename_and_mimetype(self):
        filename, _body, mimetype = netutil.pack_parameters_into_json_file_attachment({})
        assert filename == "parameters.json"
        assert mimetype == "application/json"

    def test_body_is_valid_json(self):
        params = {"voice": "af_heart", "speed": 1.0, "flags": [1, 2, 3]}
        _fn, body, _mt = netutil.pack_parameters_into_json_file_attachment(params)
        assert json.loads(body) == params


class TestUnpackJsonAttachment:
    def test_parses_stream(self):
        params = {"foo": "bar", "n": 42}
        stream = io.BytesIO(json.dumps(params).encode("utf-8"))
        assert netutil.unpack_parameters_from_json_file_attachment(stream) == params

    def test_empty_dict(self):
        stream = io.BytesIO(b"{}")
        assert netutil.unpack_parameters_from_json_file_attachment(stream) == {}

    def test_invalid_json_raises(self):
        stream = io.BytesIO(b"{not valid json")
        with pytest.raises(json.JSONDecodeError):
            netutil.unpack_parameters_from_json_file_attachment(stream)


class TestJsonAttachmentRoundtrip:
    def test_roundtrip_preserves_data(self):
        params = {"model": "whisper-base", "temperature": 0.0, "languages": ["en", "fi"]}
        _fn, body, _mt = netutil.pack_parameters_into_json_file_attachment(params)
        stream = io.BytesIO(body.encode("utf-8"))
        assert netutil.unpack_parameters_from_json_file_attachment(stream) == params

    def test_roundtrip_unicode(self):
        params = {"greeting": "Hyvää päivää", "symbol": "ϕ"}
        _fn, body, _mt = netutil.pack_parameters_into_json_file_attachment(params)
        stream = io.BytesIO(body.encode("utf-8"))
        assert netutil.unpack_parameters_from_json_file_attachment(stream) == params


# --------------------------------------------------------------------------------
# Multipart x-mixed-replace extractor
#
# The extractor reads bytes iteratively from `source`.  Each part has the shape:
#
#     --frame\r\n
#     Content-Type: <mimetype>\r\n
#     Content-Length: <n>\r\n
#     \r\n
#     <n bytes of body>
#
# followed by the next `--frame...` part.

BOUNDARY = "--frame"


def _build_part(body: bytes, mimetype: str = "image/jpeg", extra_headers: dict = None) -> bytes:
    extras = ""
    if extra_headers:
        extras = "".join(f"{k}: {v}\r\n" for k, v in extra_headers.items())
    return (
        f"{BOUNDARY}\r\n"
        f"Content-Type: {mimetype}\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"{extras}"
        f"\r\n"
    ).encode("utf-8") + body


class TestMultipartExtractor:
    def test_single_part(self):
        body = b"\xff\xd8hello-payload"
        stream = iter([_build_part(body)])
        gen = netutil.multipart_x_mixed_replace_payload_extractor(stream, BOUNDARY, "image/jpeg")
        mime, _headers, payload = next(gen)
        assert mime == "image/jpeg"
        assert payload == body

    def test_multiple_parts(self):
        payloads = [b"first", b"SECOND", b"\x00\x01\x02third"]
        chunks = [_build_part(p) for p in payloads]
        gen = netutil.multipart_x_mixed_replace_payload_extractor(iter(chunks), BOUNDARY, "image/jpeg")
        for expected in payloads:
            _mime, _headers, got = next(gen)
            assert got == expected

    def test_mimetype_passthrough_when_no_expected(self):
        # `expected_mimetype=None` means any mimetype is accepted — report what arrived.
        stream = iter([_build_part(b"x", mimetype="image/png")])
        gen = netutil.multipart_x_mixed_replace_payload_extractor(stream, BOUNDARY, expected_mimetype=None)
        mime, _headers, _payload = next(gen)
        assert mime == "image/png"

    def test_mimetype_mismatch_raises(self):
        stream = iter([_build_part(b"x", mimetype="image/png")])
        gen = netutil.multipart_x_mixed_replace_payload_extractor(stream, BOUNDARY, expected_mimetype="image/jpeg")
        with pytest.raises(ValueError, match="expected mimetype"):
            next(gen)

    def test_missing_content_length_raises(self):
        # Build a part by hand without Content-Length.
        part = (
            f"{BOUNDARY}\r\n"
            f"Content-Type: image/jpeg\r\n"
            f"\r\n"
        ).encode("utf-8") + b"body"
        gen = netutil.multipart_x_mixed_replace_payload_extractor(iter([part]), BOUNDARY, "image/jpeg")
        with pytest.raises(ValueError, match="Content-Length"):
            next(gen)

    def test_missing_content_type_when_expected_raises(self):
        part = (
            f"{BOUNDARY}\r\n"
            f"Content-Length: 4\r\n"
            f"\r\n"
        ).encode("utf-8") + b"body"
        gen = netutil.multipart_x_mixed_replace_payload_extractor(iter([part]), BOUNDARY, "image/jpeg")
        with pytest.raises(ValueError, match="Content-Type"):
            next(gen)

    def test_body_split_across_chunks(self):
        body = b"abcdefghij"  # 10 bytes
        part = _build_part(body)
        split = len(part) - 5  # split the last 5 body bytes off into a second chunk
        stream = iter([part[:split], part[split:]])
        gen = netutil.multipart_x_mixed_replace_payload_extractor(stream, BOUNDARY, "image/jpeg")
        _mime, _headers, payload = next(gen)
        assert payload == body

    def test_no_extra_headers_yields_empty_dict(self):
        stream = iter([_build_part(b"x")])
        gen = netutil.multipart_x_mixed_replace_payload_extractor(stream, BOUNDARY, "image/jpeg")
        _mime, headers, _payload = next(gen)
        assert headers == {}

    def test_extra_headers_captured(self):
        stream = iter([_build_part(b"x", extra_headers={"X-Foo": "bar", "X-Baz": "qux"})])
        gen = netutil.multipart_x_mixed_replace_payload_extractor(stream, BOUNDARY, "image/jpeg")
        _mime, headers, _payload = next(gen)
        assert headers == {"x-foo": "bar", "x-baz": "qux"}

    def test_extra_header_keys_are_lowercased(self):
        stream = iter([_build_part(b"x", extra_headers={"X-MiXeD-CaSe": "preserved"})])
        gen = netutil.multipart_x_mixed_replace_payload_extractor(stream, BOUNDARY, "image/jpeg")
        _mime, headers, _payload = next(gen)
        assert "x-mixed-case" in headers
        assert headers["x-mixed-case"] == "preserved"

    def test_extra_header_value_with_colons(self):
        # Real-world case: avatar stream sends JSON-encoded dicts in X-* headers
        # (e.g. x-crop bbox, x-server-stats). The JSON contains colons — the parser
        # must split on the FIRST colon only to avoid corrupting the value.
        json_value = json.dumps({"x": 10, "y": 20, "w": 100, "h": 200})
        stream = iter([_build_part(b"x", extra_headers={"X-Crop": json_value})])
        gen = netutil.multipart_x_mixed_replace_payload_extractor(stream, BOUNDARY, "image/jpeg")
        _mime, headers, _payload = next(gen)
        assert json.loads(headers["x-crop"]) == {"x": 10, "y": 20, "w": 100, "h": 200}

    def test_extra_content_star_headers_are_dropped(self):
        # Content-Encoding / Content-Language and other Content-* headers are
        # transport details — explicitly not surfaced via extra_headers.
        stream = iter([_build_part(b"x", extra_headers={"Content-Encoding": "identity",
                                                         "X-Keep": "me"})])
        gen = netutil.multipart_x_mixed_replace_payload_extractor(stream, BOUNDARY, "image/jpeg")
        _mime, headers, _payload = next(gen)
        assert "content-encoding" not in headers
        assert headers == {"x-keep": "me"}

    def test_extra_headers_persist_across_parts(self):
        # Each part's extras are independent; no leakage from one part to the next.
        parts = [
            _build_part(b"first", extra_headers={"X-Index": "0"}),
            _build_part(b"second", extra_headers={"X-Index": "1", "X-Extra": "only-here"}),
            _build_part(b"third"),
        ]
        gen = netutil.multipart_x_mixed_replace_payload_extractor(iter(parts), BOUNDARY, "image/jpeg")
        _m, h0, _p = next(gen)
        _m, h1, _p = next(gen)
        _m, h2, _p = next(gen)
        assert h0 == {"x-index": "0"}
        assert h1 == {"x-index": "1", "x-extra": "only-here"}
        assert h2 == {}

    def test_exhausted_source_raises_eof_on_next(self):
        # After yielding the one part, the generator tries to resync for the next;
        # with no more data, `read_more_input` raises EOFError.
        stream = iter([_build_part(b"payload")])
        gen = netutil.multipart_x_mixed_replace_payload_extractor(stream, BOUNDARY, "image/jpeg")
        next(gen)  # first part OK
        with pytest.raises(EOFError):
            next(gen)


# --------------------------------------------------------------------------------
# URL / host utilities

class TestExtractUrls:
    def test_plain_url(self):
        assert netutil.extract_urls("see https://example.com/x here") == ["https://example.com/x"]

    def test_trailing_punctuation_trimmed(self):
        assert netutil.extract_urls("go to https://example.com/x.") == ["https://example.com/x"]
        assert netutil.extract_urls("https://example.com/x, and more") == ["https://example.com/x"]

    def test_markdown_link(self):
        assert netutil.extract_urls("[paper](https://arxiv.org/abs/2301.1)") == ["https://arxiv.org/abs/2301.1"]

    def test_parenthetical(self):
        assert netutil.extract_urls("(see https://foo.org/y)") == ["https://foo.org/y"]

    def test_multiple_in_order_with_duplicates(self):
        text = "https://a.com https://b.com https://a.com"
        assert netutil.extract_urls(text) == ["https://a.com", "https://b.com", "https://a.com"]

    def test_non_http_ignored(self):
        assert netutil.extract_urls("ftp://host/x and file:///etc/passwd") == []

    def test_no_urls(self):
        assert netutil.extract_urls("just some prose, no links") == []


class TestUrlHost:
    def test_lowercased(self):
        assert netutil.url_host("https://Old.Reddit.COM/r/x/") == "old.reddit.com"

    def test_strips_port_and_path(self):
        assert netutil.url_host("http://example.com:8080/a/b?c=d") == "example.com"

    def test_no_host(self):
        assert netutil.url_host("/relative/path") == ""


class TestHostMatchesAllowlist:
    def test_exact(self):
        assert netutil.host_matches_allowlist("doi.org", ["doi.org"]) is True
        assert netutil.host_matches_allowlist("evil.com", ["doi.org"]) is False

    def test_wildcard_matches_apex(self):
        # The apex match matters: the arXiv HTML rewrite targets a bare arxiv.org host.
        assert netutil.host_matches_allowlist("arxiv.org", ["*.arxiv.org"]) is True

    def test_wildcard_matches_subdomain(self):
        assert netutil.host_matches_allowlist("export.arxiv.org", ["*.arxiv.org"]) is True

    def test_wildcard_does_not_match_suffix_lookalike(self):
        assert netutil.host_matches_allowlist("notarxiv.org", ["*.arxiv.org"]) is False

    def test_case_insensitive(self):
        assert netutil.host_matches_allowlist("ARXIV.ORG", ["*.Arxiv.Org"]) is True

    def test_empty_host_never_matches(self):
        assert netutil.host_matches_allowlist("", ["*.arxiv.org", "doi.org"]) is False

    def test_empty_allowlist(self):
        assert netutil.host_matches_allowlist("arxiv.org", []) is False
