"""Unit tests for raven.common.docextract.

Every binary format is generated on the fly from the shared test utilities rather than committed as a fixture
file. For PDF that means assembling the bytes by hand (`make_minimal_pdf` writes a correct xref table, so pypdf
reads it as a real born-digital PDF); for the office formats it means writing the file with the same library
family that reads it back. The office round-trips are therefore weaker evidence than the PDF ones — a shared
misunderstanding of the format would cancel out — so they are aimed at what Raven decides rather than at what
the library does: which blocks get collected, and in what order.
"""

import pytest

from raven.common import docextract
from raven.common.tests import make_docx, make_minimal_pdf, make_odp, make_odt, make_pptx, make_textless_pdf


# ---------------------------------------------------------------------------
# Capability queries
# ---------------------------------------------------------------------------

def test_supported_extensions_is_the_full_advertised_set():
    # Pinned as an exact set rather than spot-checked, because this list is the contract that several other
    # places are kept in sync with by hand: `raven.librarian.config.llm_docs_exts`, the attachment dialog's
    # filter, and the format list in the Librarian README. Spot-checking a few members would let any of the
    # others fall out unnoticed; pinning the set makes both directions a deliberate edit.
    #
    # `.bib` in particular has to stay: `raven-burstbib` exists to split one BibTeX file into per-entry `.bib`
    # files precisely so they can be dropped into the document database as individual documents. Dropping the
    # extension here would break that workflow at the far end of the constellation, where nothing points back.
    assert set(docextract.supported_extensions()) == {".txt", ".md", ".rst", ".org", ".bib", ".tex",
                                                      ".pdf",
                                                      ".docx", ".pptx", ".odt", ".odp",
                                                      ".html", ".htm"}


def test_supported_extensions_has_no_duplicates():
    # `supported_extensions` splices two independently maintained lists (the plaintext tuple and the parser
    # dispatch table), and the set comparison above would hide an extension that appears in both. It matters
    # because the dispatch table wins: a format listed in both would quietly stop being read as plain text.
    exts = docextract.supported_extensions()
    assert len(exts) == len(set(exts))


@pytest.mark.parametrize("name, expected", [
    ("notes.txt", True),
    ("README.MD", True),   # case-insensitive
    ("paper.pdf", True),
    ("report.docx", True),
    ("REPORT.DOCX", True),
    ("deck.pptx", True),
    ("notes.odt", True),
    ("deck.odp", True),
    ("saved-page.html", True),
    ("saved-page.htm", True),
    ("sheet.xlsx", False),  # spreadsheets are a separate problem class; see TODO_DEFERRED.md
    ("legacy.doc", False),
    ("photo.png", False),
    ("archive.tar.gz", False),
    ("noext", False),
])
def test_is_supported(name, expected):
    assert docextract.is_supported(name) is expected


# ---------------------------------------------------------------------------
# Plain-text extraction
# ---------------------------------------------------------------------------

def test_plaintext_roundtrip(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("Hello Raven", encoding="utf-8")
    assert docextract.extract_text(p) == "Hello Raven"


def test_plaintext_is_stripped(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("\n\n  # Title  \n\n", encoding="utf-8")
    assert docextract.extract_text(p) == "# Title"


def test_whitespace_only_returns_none(tmp_path):
    p = tmp_path / "blank.txt"
    p.write_text("   \n\t \n", encoding="utf-8")
    assert docextract.extract_text(p) is None


def test_accepts_str_path(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("string path ok", encoding="utf-8")
    assert docextract.extract_text(str(p)) == "string path ok"


# ---------------------------------------------------------------------------
# Error situations raise (never silently return None)
# ---------------------------------------------------------------------------

def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        docextract.extract_text(tmp_path / "nope.txt")


def test_non_utf8_text_raises(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_bytes(b"\xff\xfe\x00garbage\x80\x81")
    with pytest.raises(docextract.DocumentExtractionError):
        docextract.extract_text(p)


def test_non_utf8_error_chains_cause(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_bytes(b"\x80\x81\x82")
    with pytest.raises(docextract.DocumentExtractionError) as excinfo:
        docextract.extract_text(p)
    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)


def test_not_a_pdf_raises(tmp_path):
    p = tmp_path / "fake.pdf"
    p.write_bytes(b"this is plainly not a PDF")
    with pytest.raises(docextract.DocumentExtractionError):
        docextract.extract_text(p)


def test_empty_file_with_pdf_extension_raises(tmp_path):
    p = tmp_path / "empty.pdf"
    p.write_bytes(b"")
    with pytest.raises(docextract.DocumentExtractionError):
        docextract.extract_text(p)


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def test_pdf_roundtrip(tmp_path):
    p = tmp_path / "sample.pdf"
    p.write_bytes(make_minimal_pdf("Hello Raven PDF extraction"))
    assert docextract.extract_text(p) == "Hello Raven PDF extraction"


def test_pdf_uppercase_extension(tmp_path):
    p = tmp_path / "SAMPLE.PDF"
    p.write_bytes(make_minimal_pdf("Case insensitive"))
    assert docextract.extract_text(p) == "Case insensitive"


def test_pdf_without_text_layer_returns_none(tmp_path):
    # A parseable PDF that has no text to extract (stand-in for a scanned/image-only page) is "empty", not an
    # error -> None, so the caller skips it rather than treating it as a failure.
    p = tmp_path / "scanned.pdf"
    p.write_bytes(make_textless_pdf())
    assert docextract.extract_text(p) is None


# ---------------------------------------------------------------------------
# Word documents (.docx)
# ---------------------------------------------------------------------------

def test_docx_paragraphs(tmp_path):
    p = tmp_path / "report.docx"
    p.write_bytes(make_docx(["First para", "Second para"]))
    assert docextract.extract_text(p) == "First para\nSecond para"


def test_docx_table_is_read_in_place(tmp_path):
    # The interesting property is placement, not presence: a table has to come back between the paragraphs that
    # surround it, because retrieval chunks on a sliding window and a chunk spanning the boundary is only
    # meaningful if the reading order survived.
    p = tmp_path / "report.docx"
    p.write_bytes(make_docx(["Before", [["Name", "Value"], ["alpha", "42"]], "After"]))
    assert docextract.extract_text(p) == "Before\nName\tValue\nalpha\t42\nAfter"


def test_docx_empty_returns_none(tmp_path):
    p = tmp_path / "blank.docx"
    p.write_bytes(make_docx([]))
    assert docextract.extract_text(p) is None


# ---------------------------------------------------------------------------
# PowerPoint presentations (.pptx)
# ---------------------------------------------------------------------------

def test_pptx_slides_in_order(tmp_path):
    p = tmp_path / "deck.pptx"
    p.write_bytes(make_pptx([{"text": ["Slide one"]}, {"text": ["Slide two"]}]))
    assert docextract.extract_text(p) == "Slide one\nSlide two"


def test_pptx_includes_presenter_notes(tmp_path):
    # Notes are part of what the deck says even though they are never projected — on a lecture deck they often
    # carry the argument the slide only asserts.
    p = tmp_path / "deck.pptx"
    p.write_bytes(make_pptx([{"text": ["Bullet"], "notes": "The reasoning behind the bullet"}]))
    assert docextract.extract_text(p) == "Bullet\nThe reasoning behind the bullet"


def test_pptx_reads_grouped_shapes(tmp_path):
    # Text inside a group is invisible to a flat pass over `slide.shapes`, and grouping is something a person
    # does for layout reasons without any expectation that it hides the words.
    p = tmp_path / "deck.pptx"
    p.write_bytes(make_pptx([{"text": ["Loose"], "group": ["Grouped"]}]))
    assert docextract.extract_text(p) == "Loose\nGrouped"


def test_pptx_table(tmp_path):
    p = tmp_path / "deck.pptx"
    p.write_bytes(make_pptx([{"table": [["k", "v"], ["x", "9"]]}]))
    assert docextract.extract_text(p) == "k\tv\nx\t9"


def test_pptx_empty_returns_none(tmp_path):
    p = tmp_path / "blank.pptx"
    p.write_bytes(make_pptx([]))
    assert docextract.extract_text(p) is None


# ---------------------------------------------------------------------------
# OpenDocument (.odt, .odp)
# ---------------------------------------------------------------------------

def test_odt_headings_and_paragraphs_in_order(tmp_path):
    # ODF marks a heading and a paragraph with different element names, so an extractor that collects them by
    # type rather than by position reads back every heading first and every paragraph after.
    p = tmp_path / "notes.odt"
    p.write_bytes(make_odt([("h", "Chapter one"), ("p", "Body of one"),
                            ("h", "Chapter two"), ("p", "Body of two")]))
    assert docextract.extract_text(p) == "Chapter one\nBody of one\nChapter two\nBody of two"


def test_odt_empty_returns_none(tmp_path):
    p = tmp_path / "blank.odt"
    p.write_bytes(make_odt([]))
    assert docextract.extract_text(p) is None


def test_odp_slides_in_order(tmp_path):
    p = tmp_path / "deck.odp"
    p.write_bytes(make_odp([["Slide one title", "point a"], ["Slide two title"]]))
    assert docextract.extract_text(p) == "Slide one title\npoint a\nSlide two title"


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

# Long enough that a readability extractor scores the paragraphs as body content rather than as page chrome.
_ARTICLE_HTML = b"""<!DOCTYPE html><html><head><title>On Alignment</title></head><body>
<nav>Home | Archive | About</nav>
<article>
<p>The first paragraph makes a claim about optimization pressure and how it interacts with proxy
measures over long horizons.</p>
<p>The second paragraph elaborates, at sufficient length that a readability extractor treats it as
real body content rather than as chrome.</p>
</article>
<footer>Copyright 2026</footer></body></html>"""


def test_html_extracts_body_and_drops_chrome(tmp_path):
    p = tmp_path / "post.html"
    p.write_bytes(_ARTICLE_HTML)
    text = docextract.extract_text(p)
    assert "optimization pressure" in text
    assert "Home | Archive | About" not in text  # nav
    assert "Copyright 2026" not in text          # footer


# Each article has to be *big* for this to test anything. Readability extraction keeps every article on a
# page of short ones and switches to picking a single block once they pass roughly 5000 characters — which is
# the size a real chapter is, and why the defect shows up on saved fiction and not on a hand-written fixture.
# Measured against trafilatura 2.1.0: at 2000 characters per article all 13 survive, at 5000 exactly one does.
_ARTICLE_CHARS = 6000
_FILLER = ("The narrator considered the situation at some length, and then considered it again from a "
           "second angle entirely, which did not help as much as had been hoped. ")


def _multi_article_html(n_articles: int) -> bytes:
    """A saved page holding several full-length articles — the shape a multi-chapter story is archived in."""
    filler = (_FILLER * (_ARTICLE_CHARS // len(_FILLER) + 1))[:_ARTICLE_CHARS]
    articles = "".join(
        f'<article class="chapter"><header><h1>{i}. Chapter Title {i}</h1></header>'
        f"<p>Chapter {i} opens on the marker phrase chaptermarker{i}. {filler}</p>"
        f'<footer><p><a href="#top">Jump to top</a></p></footer></article>'
        for i in range(1, n_articles + 1))
    return (f"<!DOCTYPE html><html><head><title>A Serial</title></head><body>"
            f"<nav>Home | Archive</nav>{articles}<footer>Copyright 2026</footer>"
            f"</body></html>").encode("utf-8")


def test_html_with_many_articles_keeps_all_of_them(tmp_path):
    # Readability extraction selects one main content block. On a page that holds several — a story archived
    # as one article per chapter — that silently discards the rest, and the document lands in the index
    # present, findable and a fraction complete. Measured at 6% of the text on a real saved page.
    p = tmp_path / "serial.html"
    p.write_bytes(_multi_article_html(13))
    text = docextract.extract_text(p)
    missing = [i for i in range(1, 14) if f"chaptermarker{i}" not in text]
    assert not missing, f"chapters dropped from the extraction: {missing}"


def test_html_article_headings_are_recovered(tmp_path):
    # Each article's own heading is dropped by the readability pass the same way the page `<title>` is, and
    # on a serial those are the chapter titles — worth having in a retrieval index.
    #
    # Only the per-article headings are asserted, not the page heading above them: which string trafilatura's
    # metadata pass calls the title is its own business, and on a synthetic fixture it takes the first `<h1>`
    # rather than `<title>`. That pairing is covered by `test_html_title_is_recovered`, against a page shaped
    # the way the extractor expects.
    p = tmp_path / "serial.html"
    p.write_bytes(_multi_article_html(3))
    text = docextract.extract_text(p)
    assert ["## 1. Chapter Title 1", "## 2. Chapter Title 2", "## 3. Chapter Title 3"] == \
        [line for line in text.splitlines() if line.startswith("## ")]


def test_html_single_article_still_drops_page_chrome(tmp_path):
    # The guard against the fix above defeating the feature it guards: a page with one content block must
    # still get boilerplate removal rather than being handed back whole.
    p = tmp_path / "post.html"
    p.write_bytes(_ARTICLE_HTML)
    text = docextract.extract_text(p)
    assert "optimization pressure" in text
    assert "Home | Archive | About" not in text
    assert "Copyright 2026" not in text


def test_html_title_is_recovered(tmp_path):
    # Readability extraction drops `<title>`, and on a saved page it is often the only thing naming the
    # document — filenames off the web are routinely useless.
    p = tmp_path / "post.html"
    p.write_bytes(_ARTICLE_HTML)
    assert docextract.extract_text(p).startswith("# On Alignment")


def test_html_title_not_duplicated_when_body_already_opens_with_it(tmp_path):
    # A page whose `<h1>` restates its `<title>` is the common case, and the heading survives extraction.
    p = tmp_path / "post.html"
    p.write_bytes(b"""<!DOCTYPE html><html><head><title>On Alignment</title></head><body><article>
<h1>On Alignment</h1>
<p>A paragraph long enough to be scored as real body content by a readability extractor rather than
being discarded as page chrome or navigation.</p></article></body></html>""")
    assert docextract.extract_text(p).count("On Alignment") == 1


# The dedup decision is tested directly as well as through `extract_text`, because which of the two body
# shapes below the extractor returns for a given page depends on the installed trafilatura version. Going
# through the library alone would leave whichever shape it does not currently produce untested — which is how
# the equality-based version of this check passed locally while failing CI.
@pytest.mark.parametrize("body, expected", [
    ("# On Alignment\n\nA paragraph.", True),                 # Markdown heading on its own line
    ("On Alignment A paragraph continues here.", True),       # flat text, title runs into the first paragraph
    ("  \n\n# On Alignment\n\nA paragraph.", True),           # leading blank lines
    ("## On Alignment\n\nA paragraph.", True),                # deeper heading level
    ("A paragraph that does not name the page.", False),      # nothing to dedup: title must be prepended
    ("", False),
])
def test_html_body_opens_with_title(body, expected):
    assert docextract._body_opens_with(body, "On Alignment") is expected


def test_html_declared_encoding_is_honored(tmp_path):
    # The file is handed to the extractor as bytes precisely so its own declaration decides the encoding. Were
    # it decoded as UTF-8 here, this well-formed page would raise instead of reading.
    p = tmp_path / "page.html"
    p.write_bytes('<html><head><meta charset="iso-8859-1"><title>P\xe4iv\xe4</title></head><body>'
                  '<p>Sein\xe4joen kaupunki on suomalainen kaupunki jossa on paljon asukkaita ja se '
                  'sijaitsee Etel\xe4-Pohjanmaalla.</p></body></html>'.encode("iso-8859-1"))
    text = docextract.extract_text(p)
    assert "Seinäjoen" in text
    assert "Päivä" in text


def test_html_htm_extension(tmp_path):
    p = tmp_path / "post.htm"
    p.write_bytes(_ARTICLE_HTML)
    assert "optimization pressure" in docextract.extract_text(p)


def test_html_script_built_page_still_indexes_under_its_title(tmp_path):
    # A self-contained app that carries its data as a script literal and builds the DOM at load — the shape a
    # chat assistant emits as an artifact. Its body is unreadable to us, but the title is recovered separately,
    # so the document is at least findable by name rather than vanishing from the database entirely. Reading
    # the data itself is a separate, larger question; see TODO_DEFERRED.md.
    p = tmp_path / "diva_modules.html"
    p.write_bytes(b"""<!DOCTYPE html><html><head><title>Module List</title></head><body>
<div id="app"></div>
<script>const MODULES = [{name: "Miku", cost: 3}]; render(MODULES);</script>
</body></html>""")
    assert docextract.extract_text(p) == "# Module List"


def test_html_single_page_app_shell_returns_none(tmp_path):
    # The bare shell of a JS-rendered page holds no content: it was never in the file, and nothing here runs the
    # scripts that would fetch it. That is "empty", not an error — the same answer a scanned PDF gets.
    p = tmp_path / "app.html"
    p.write_bytes(b'<!DOCTYPE html><html><head></head><body><div id="root"></div>'
                  b'<script src="bundle.js"></script></body></html>')
    assert docextract.extract_text(p) is None


# ---------------------------------------------------------------------------
# Office formats: malformed input raises rather than returning empty
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["fake.docx", "fake.pptx", "fake.odt", "fake.odp"])
def test_office_file_that_is_not_a_zip_raises(tmp_path, name):
    # Every office format here is a zip container, so the cheapest wrong file is one that is not a zip at all.
    # It must raise: a file the user deliberately handed us and that we cannot read is an error situation, not
    # an empty document, and the two are what the raise-vs-None split exists to keep apart.
    p = tmp_path / name
    p.write_bytes(b"this is plainly not an office document")
    with pytest.raises(docextract.DocumentExtractionError):
        docextract.extract_text(p)


@pytest.mark.parametrize("name", ["empty.docx", "empty.pptx", "empty.odt", "empty.odp"])
def test_zero_byte_office_file_raises(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"")
    with pytest.raises(docextract.DocumentExtractionError):
        docextract.extract_text(p)


def test_office_extraction_error_chains_cause(tmp_path):
    # The underlying exception is what says *why* the file could not be read, and an interactive attach site
    # shows that to the user; dropping the chain would leave only "could not be read".
    p = tmp_path / "fake.docx"
    p.write_bytes(b"not a zip")
    with pytest.raises(docextract.DocumentExtractionError) as excinfo:
        docextract.extract_text(p)
    assert excinfo.value.__cause__ is not None
