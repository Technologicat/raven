"""Unit tests for the pure helpers of raven.server.modules.webfetch.

Covers URL rewriting, scheme checking, SSRF IP classification, and the network-safety
gate (with DNS resolution monkeypatched). The actual fetch/extraction (trafilatura,
Selenium, youtube-transcript-api) is network-bound and not exercised here.
"""

import socket

import pytest

# webfetch imports `colorama` at module top (and, lazily, trafilatura/selenium/youtube-transcript-api).
# The CI test job installs a hand-picked minimal dep subset that omits colorama; skip this whole
# module when the import can't be satisfied. Mirrors test_scaffold.py / test_hybridir.py.
webfetch = pytest.importorskip("raven.server.modules.webfetch",
                               reason="webfetch needs colorama (not in the CI minimal dep subset)")
from raven.server.modules import webcommon  # noqa: E402 -- after the guard, which covers it too


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


class TestExtractTitle:
    def test_extracts_and_normalizes_title(self):
        # `_extract_title` lazily imports trafilatura (a heavy dep present in the full install but not in
        # the CI coverage job's dep subset). The colorama importorskip at module top doesn't cover it
        # (trafilatura is imported lazily inside the function, not at module load), so guard here.
        pytest.importorskip("trafilatura", reason="webfetch._extract_title lazily imports trafilatura")
        html = "<html><head><title>My Title | Some Site</title></head><body><h1>My Title</h1><p>" + ("word " * 60) + "</p></body></html>"
        assert webfetch._extract_title(html) == "My Title"  # trafilatura strips the site suffix

    def test_none_without_html(self):
        assert webfetch._extract_title(None) is None
        assert webfetch._extract_title("") is None


class TestFetchOutcome:
    """Which canonical answer `fetch` gives when both tiers come up short, with the tiers themselves faked."""

    @pytest.fixture
    def short_tier1(self, monkeypatch):
        """Tier 1 got nothing and no status, as a failed GET leaves it; the gate lets every URL through."""
        monkeypatch.setattr(webfetch, "_classify_url_network_safety", lambda url, allow_private: None)
        monkeypatch.setattr(webfetch, "_http_get", lambda url: (None, None))
        monkeypatch.setattr(webfetch, "_extract_clean_text", lambda html, url, output_format="markdown": "")
        monkeypatch.setattr(webfetch, "_extract_title", lambda html: None)

    def test_a_page_that_did_not_finish_loading_says_so(self, monkeypatch, short_tier1):
        monkeypatch.setattr(webfetch, "_fetch_tier2", lambda url, output_format, is_cancelled, on_progress: None)
        result = webfetch.fetch("https://slow.example/p")
        assert result["content"] == webfetch.CANONICAL_PAGE_TIMEOUT.format(url="https://slow.example/p")
        assert not result["spaSuspected"]
        assert result["fetched"] is False

    def test_a_page_that_loaded_empty_is_still_a_js_only_page(self, monkeypatch, short_tier1):
        # The control for the one above: the same fixture, and only the Tier 2 outcome differs.
        monkeypatch.setattr(webfetch, "_fetch_tier2", lambda url, output_format, is_cancelled, on_progress: "")
        result = webfetch.fetch("https://spa.example/p")
        assert result["content"] == webfetch.CANONICAL_SPA_SUSPECTED
        assert result["spaSuspected"]
        assert result["fetched"] is False

    def test_a_server_with_no_browser_says_so(self, monkeypatch, short_tier1):
        # Rather than the JS-only notice, which would blame the site for what the server lacks.
        def no_browser(url, output_format, is_cancelled, on_progress):
            raise webfetch._NoBrowser
        monkeypatch.setattr(webfetch, "_fetch_tier2", no_browser)
        result = webfetch.fetch("https://spa.example/p")
        assert result["content"] == webfetch.CANONICAL_NO_BROWSER.format(url="https://spa.example/p")
        assert not result["spaSuspected"]
        assert result["fetched"] is False

    def test_only_the_page_itself_is_fetched(self, monkeypatch, short_tier1):
        # The control for the notices above: `fetched` is how the client tells the page from a notice about it.
        page = "The page's own words. " * 100
        monkeypatch.setattr(webfetch, "_fetch_tier2", lambda url, output_format, is_cancelled, on_progress: page)
        result = webfetch.fetch("https://spa.example/p")
        assert result["content"] == page
        assert result["fetched"] is True

    def test_a_cancelled_fetch_never_starts_the_browser(self, monkeypatch, short_tier1):
        started = []
        monkeypatch.setattr(webfetch, "_fetch_tier2",
                            lambda url, output_format, is_cancelled, on_progress: started.append(url) or "")
        with pytest.raises(webcommon.Cancelled):
            webfetch.fetch("https://spa.example/p", is_cancelled=lambda: True)
        assert started == []
        webfetch.fetch("https://spa.example/p", is_cancelled=lambda: False)  # the control: not cancelled, it does
        assert started == ["https://spa.example/p"]


class TestTier2DeadBrowser:
    """Tier 2's browser can stop answering (a crashed tab); a fetch then starts a new one, once."""

    @pytest.fixture
    def extraction(self, monkeypatch):
        """Extraction that returns the HTML it was given, so a test can tell which browser rendered the page."""
        monkeypatch.setattr(webfetch, "_extract_clean_text", lambda html, url, output_format: html)

    def _fetch(self):
        return webfetch._fetch_tier2("https://example.com", output_format="markdown",
                                     is_cancelled=lambda: False, on_progress=lambda text: None)

    def test_a_dead_browser_is_replaced_and_the_page_renders(self, extraction, fake_browser, monkeypatch):
        dead, fresh = fake_browser(answers=False), fake_browser(page="<html>from the new browser</html>")
        monkeypatch.setattr(webfetch, "_driver", dead)
        monkeypatch.setattr(webcommon, "get_driver", lambda page_load_timeout=None: fresh)
        assert self._fetch() == "<html>from the new browser</html>"
        assert dead.quit_called
        assert webfetch._driver is fresh

    def test_an_ordinary_failure_does_not_restart_the_browser(self, extraction, fake_browser, monkeypatch):
        # The control for the one above: the navigation fails the same way, but the browser still answers.
        failing = fake_browser(navigation_fails=True)
        monkeypatch.setattr(webfetch, "_driver", failing)
        monkeypatch.setattr(webcommon, "get_driver", lambda page_load_timeout=None: pytest.fail("restarted a browser that answers"))
        assert self._fetch() == ""
        assert webfetch._driver is failing
        assert len(failing.visited) == 1

    def test_a_browser_that_stays_dead_is_not_restarted_again(self, extraction, fake_browser, monkeypatch):
        starts = []
        def get_dead_driver(page_load_timeout=None):
            starts.append(1)
            return fake_browser(answers=False)
        monkeypatch.setattr(webfetch, "_driver", fake_browser(answers=False))
        monkeypatch.setattr(webcommon, "get_driver", get_dead_driver)
        assert self._fetch() == ""
        assert len(starts) == 1, "one restart per fetch, not a loop"

    def test_a_browser_that_cannot_be_restarted_reports_no_browser(self, extraction, fake_browser, monkeypatch):
        monkeypatch.setattr(webfetch, "_driver", fake_browser(answers=False))
        monkeypatch.setattr(webcommon, "get_driver", lambda page_load_timeout=None: None)
        with pytest.raises(webfetch._NoBrowser):
            self._fetch()


class TestExtractionKeepsLinks:
    """Markdown extraction keeps each link's target, so a page whose payload is its links is still usable."""

    PARAGRAPH = ("<p>Paragraph {i} discusses the subject at some length, so that the extractor sees prose rather "
                 "than boilerplate; it mentions <a href=\"https://example.org/page-{i}\">the related page number "
                 "{i}</a> in the middle of a sentence, and then carries on for a while longer before it ends.</p>")

    @pytest.fixture
    def page(self):
        pytest.importorskip("trafilatura", reason="trafilatura is not installed")
        return ("<html><head><title>Fixture</title></head><body><article><h1>A heading</h1>"
                + "".join(self.PARAGRAPH.format(i=i) for i in range(4)) + "</article></body></html>")

    def test_markdown_keeps_the_link_targets(self, page):
        import trafilatura
        without_links = trafilatura.extract(page, output_format="markdown", include_tables=True, favor_recall=True)
        assert "](https://" not in without_links, "trafilatura kept links by itself, so this fixture tests nothing"
        extracted = webfetch._extract_clean_text(page, url="https://example.org/", output_format="markdown")
        for i in range(4):
            assert f"](https://example.org/page-{i})" in extracted

    def test_plain_text_has_no_link_markup(self, page):
        extracted = webfetch._extract_clean_text(page, url="https://example.org/", output_format="txt")
        assert "](https://" not in extracted
        assert "the related page number 0" in extracted


class TestIsTextType:
    @pytest.mark.parametrize("content_type", ["text/html; charset=utf-8", "text/x-wiki", "TEXT/PLAIN",
                                              "application/json", "application/ld+json", "application/xml",
                                              "application/atom+xml", "application/javascript", "text/csv"])
    def test_text_and_structured_data_are_text(self, content_type):
        assert webfetch._is_text_type(content_type)

    @pytest.mark.parametrize("content_type", ["image/png", "application/pdf", "application/octet-stream",
                                              "video/mp4", "", "nonsense"])
    def test_anything_else_is_not(self, content_type):
        assert not webfetch._is_text_type(content_type)


class TestRawFetch:
    """`fetch(..., output_format="raw")`: the body as the site sent it, with the network layer scripted."""

    URL = "https://example.org/w/index.php?title=Foo&action=raw"

    @pytest.fixture
    def network(self, monkeypatch):
        """Pass the safety gate, forbid URL rewriting, and script the plain GET and the browser; returns the script."""
        script = {"get": (None, None, "", self.URL), "browser": ""}
        monkeypatch.setattr(webfetch, "_classify_url_network_safety", lambda url, allow_private: None)
        monkeypatch.setattr(webfetch, "_rewrite_url", lambda url: pytest.fail("a raw fetch rewrote the URL"))
        monkeypatch.setattr(webfetch, "_http_get_raw", lambda url: script["get"])

        def browser(url, *, output_format, is_cancelled, on_progress):
            assert output_format == "raw"
            if script["browser"] == "no browser":
                raise webfetch._NoBrowser
            return script["browser"]
        monkeypatch.setattr(webfetch, "_fetch_tier2", browser)
        return script

    def test_a_text_body_comes_back_as_sent_less_invisible_characters(self, network):
        body = "== Heading ==\n'''Foo''' is a​ thing.\n\t{{Infobox}}"  # ​: zero-width space
        network["get"] = (body, 200, "text/x-wiki; charset=UTF-8", self.URL)
        result = webfetch.fetch(self.URL, output_format="raw")
        assert result["fetched"]
        assert result["content"] == body.replace("​", "")

    def test_the_url_reported_is_where_redirects_ended(self, network):
        network["get"] = ("{}", 200, "application/json", "https://example.org/final")
        assert webfetch.fetch(self.URL, output_format="raw")["url"] == "https://example.org/final"

    def test_a_type_that_is_not_text_is_refused(self, network):
        network["get"] = (None, 200, "image/png", self.URL)
        result = webfetch.fetch(self.URL, output_format="raw")
        assert not result["fetched"]
        assert "image/png" in result["content"] and "not text" in result["content"]

    def test_a_refused_request_is_tried_in_the_browser(self, network):
        network["get"] = ("Forbidden", 403, "text/html", self.URL)
        network["browser"] = "<html>rendered</html>"
        result = webfetch.fetch(self.URL, output_format="raw")
        assert result["fetched"] and result["content"] == "<html>rendered</html>"

    def test_a_refusal_the_browser_cannot_get_past_reports_the_status(self, network):
        network["get"] = ("Forbidden", 403, "text/html", self.URL)
        result = webfetch.fetch(self.URL, output_format="raw")
        assert not result["fetched"] and "HTTP 403" in result["content"]

    def test_an_unreachable_site_whose_page_never_loads_times_out(self, network):
        network["browser"] = None
        assert "did not finish loading" in webfetch.fetch(self.URL, output_format="raw")["content"]

    def test_an_unreachable_site_with_no_browser_says_so(self, network):
        network["browser"] = "no browser"
        assert "none installed" in webfetch.fetch(self.URL, output_format="raw")["content"]

    def test_an_unreachable_site_the_browser_cannot_load_is_unreachable(self, network):
        assert webfetch.fetch(self.URL, output_format="raw")["content"].startswith("Could not reach")


class TestRawBodyLimit:
    """`_http_get_raw` reads at most `webfetch_raw_max_bytes`, and says so when it cut a body off."""

    class FakeResponse:
        def __init__(self, body: bytes, content_type: str):
            self.body, self.headers = body, {"Content-Type": content_type}
            self.status_code, self.url, self.encoding, self.apparent_encoding = 200, "https://example.org/", "utf-8", "utf-8"

        def iter_content(self, chunk_size):
            for start in range(0, len(self.body), chunk_size):
                yield self.body[start:start + chunk_size]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    @pytest.fixture
    def serve(self, monkeypatch):
        from raven.server import config as server_config
        monkeypatch.setattr(server_config, "webfetch_raw_max_bytes", 100)
        def serve(body, content_type="text/plain"):
            monkeypatch.setattr(webfetch.requests, "get", lambda *args, **kwargs: self.FakeResponse(body, content_type))
        return serve

    def test_a_body_over_the_limit_is_cut_off_and_says_so(self, serve):
        serve(b"x" * 250)
        body, status, _, _ = webfetch._http_get_raw("https://example.org/")
        assert status == 200
        assert body.startswith("x" * 100) and "x" * 101 not in body
        assert "longer than the 100 bytes" in body

    def test_a_body_within_the_limit_is_whole(self, serve):
        serve(b"x" * 100)
        assert webfetch._http_get_raw("https://example.org/")[0] == "x" * 100

    def test_a_type_that_is_not_text_is_not_read(self, serve):
        serve(b"\x89PNG", content_type="image/png")
        body, status, content_type, _ = webfetch._http_get_raw("https://example.org/")
        assert body is None and status == 200 and content_type == "image/png"
