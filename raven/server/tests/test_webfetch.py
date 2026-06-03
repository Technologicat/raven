"""Unit tests for the pure helpers of raven.server.modules.webfetch.

Covers URL rewriting, scheme checking, SSRF IP classification, and the network-safety
gate (with DNS resolution monkeypatched). The actual fetch/extraction (trafilatura,
Selenium, youtube-transcript-api) is network-bound and not exercised here.
"""

import socket

import pytest

from raven.server.modules import webfetch


class TestSchemeCheck:
    def test_http_allowed(self):
        assert webfetch._check_scheme("http://example.com/x") is None

    def test_https_allowed(self):
        assert webfetch._check_scheme("https://example.com/x") is None

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host/x", "gopher://host/", "data:text/plain,hi"])
    def test_non_http_refused(self, url):
        refusal = webfetch._check_scheme(url)
        assert refusal is not None
        assert "Only HTTP and HTTPS" in refusal


class TestBlockedIP:
    @pytest.mark.parametrize("ip", [
        "127.0.0.1",          # loopback
        "10.0.0.5",           # private
        "172.16.3.4",         # private
        "192.168.1.1",        # private
        "169.254.169.254",    # link-local / cloud metadata
        "0.0.0.0",            # unspecified
        "224.0.0.1",          # multicast
        "::1",                # IPv6 loopback
        "fc00::1",            # IPv6 ULA
        "fe80::1",            # IPv6 link-local
        "not-an-ip",          # unparseable -> fail closed
    ])
    def test_blocked(self, ip):
        assert webfetch._is_blocked_ip(ip) is True

    @pytest.mark.parametrize("ip", ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"])
    def test_public_allowed(self, ip):
        assert webfetch._is_blocked_ip(ip) is False


class TestRewriteUrl:
    def test_arxiv_abs_to_html(self):
        url, extractor = webfetch._rewrite_url("https://arxiv.org/abs/2301.00001")
        assert url == "https://arxiv.org/html/2301.00001"
        assert extractor is webfetch._extract_arxiv

    def test_arxiv_pdf_with_version_to_html(self):
        url, extractor = webfetch._rewrite_url("https://arxiv.org/pdf/2301.00001v2")
        assert url == "https://arxiv.org/html/2301.00001v2"
        assert extractor is webfetch._extract_arxiv

    def test_arxiv_pdf_extension_stripped(self):
        url, _ = webfetch._rewrite_url("https://arxiv.org/pdf/2301.00001.pdf")
        assert url == "https://arxiv.org/html/2301.00001"

    def test_reddit_to_old_reddit(self):
        url, extractor = webfetch._rewrite_url("https://www.reddit.com/r/x/comments/abc/title/")
        assert url == "https://old.reddit.com/r/x/comments/abc/title/"
        assert extractor is None

    def test_reddit_bare_host(self):
        url, _ = webfetch._rewrite_url("https://reddit.com/r/x/")
        assert url == "https://old.reddit.com/r/x/"

    def test_old_reddit_not_rewritten(self):
        url, _ = webfetch._rewrite_url("https://old.reddit.com/r/x/")
        assert url == "https://old.reddit.com/r/x/"

    def test_youtube_watch_gets_transcript_extractor(self):
        url, extractor = webfetch._rewrite_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extractor is webfetch._extract_youtube_transcript

    def test_youtube_short_url_gets_transcript_extractor(self):
        _, extractor = webfetch._rewrite_url("https://youtu.be/dQw4w9WgXcQ")
        assert extractor is webfetch._extract_youtube_transcript

    def test_plain_url_unchanged(self):
        url, extractor = webfetch._rewrite_url("https://example.com/article?id=5")
        assert url == "https://example.com/article?id=5"
        assert extractor is None


class TestNetworkSafetyGate:
    """`_classify_url_network_safety` with DNS resolution stubbed."""

    @staticmethod
    def _fake_getaddrinfo(ip):
        def _impl(host, *args, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]
        return _impl

    def test_public_host_allowed(self, monkeypatch):
        monkeypatch.setattr(socket, "getaddrinfo", self._fake_getaddrinfo("93.184.216.34"))
        assert webfetch._classify_url_network_safety("https://example.com/x", allow_private=False) is None

    def test_private_resolution_blocked(self, monkeypatch):
        # A public-looking host that resolves to a private address (the SSRF / DNS-rebind shape).
        monkeypatch.setattr(socket, "getaddrinfo", self._fake_getaddrinfo("192.168.1.1"))
        refusal = webfetch._classify_url_network_safety("https://sneaky.example/x", allow_private=False)
        assert refusal is not None
        assert "private-network address" in refusal

    def test_private_allowed_with_optout(self, monkeypatch):
        monkeypatch.setattr(socket, "getaddrinfo", self._fake_getaddrinfo("127.0.0.1"))
        assert webfetch._classify_url_network_safety("http://localhost/x", allow_private=True) is None

    def test_bad_scheme_refused_before_dns(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("DNS should not be consulted for a bad scheme")
        monkeypatch.setattr(socket, "getaddrinfo", _boom)
        refusal = webfetch._classify_url_network_safety("file:///etc/passwd", allow_private=False)
        assert refusal is not None
        assert "Only HTTP and HTTPS" in refusal

    def test_dns_failure_message(self, monkeypatch):
        def _fail(*args, **kwargs):
            raise socket.gaierror("name resolution failed")
        monkeypatch.setattr(socket, "getaddrinfo", _fail)
        refusal = webfetch._classify_url_network_safety("https://no-such-host.invalid/x", allow_private=False)
        assert refusal is not None
        assert "Could not resolve the host name" in refusal
