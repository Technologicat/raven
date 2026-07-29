"""Extract indexable plaintext from document files (plain text, PDF, office documents, and HTML).

This is the single source of truth for "given a document file, give me its text" across Raven. The RAG
document-database ingester (`raven.librarian.hybridir`), the chat document attachment feature, and the
`raven-pdf2bib` tool all route through here, so there is exactly one backend per format and one
supported-format list. Adding a format here adds it to every one of those surfaces at once.

Contract of `extract_text`:

  - raises `FileNotFoundError` if the path does not exist;
  - raises `DocumentExtractionError` if the file exists but cannot be parsed as its format (a corrupt or
    encrypted PDF, a text file that is not valid UTF-8) — the underlying cause is chained;
  - returns `None` if the file parses cleanly but yields no text (a scanned PDF with no text layer, a
    whitespace-only text file);
  - otherwise returns the extracted text.

The raise-vs-`None` split is deliberate: an *error situation* (missing or unreadable) is a different thing from
an *empty but valid* document. The rule is *parse failure → raise; parsed-but-empty → `None`*. Callers apply
their own policy on top of it — a background batch ingester catches the exceptions and skips the offending file
so one bad document does not abort the batch, whereas an interactive attach site lets the exception surface so
it can tell the user *why* their file could not be read.

PDF text extraction handles born-digital PDFs (a real text layer). A scanned/image-only PDF has no text to
extract and comes back as `None`; OCR for those is a separate, later concern.

HTML goes through the same readability extraction the web-fetch tool uses, so a page saved to disk reads the way
the same page fetched live would. What that misses is a page whose text is produced by *running* it: the bare
shell of a JS-rendered site, and equally a self-contained single-file app that carries its data inline as a
script literal and builds the DOM at load. In the second case the content genuinely is in the file — it is just
in a `<script>` element, which readability extraction ignores, and reading it would mean deciding where inline
data ends and a minified bundle begins. Neither is attempted here; both extract as empty, the way a scanned PDF
does.

Every format here is read for its *text layer* only. Whatever a document says through pictures — a figure, a
photograph, a typeset equation rendered as an image — is not recovered, and a file whose content is entirely
such material extracts as empty. Legacy binary office formats (`.doc`, `.ppt`) are not supported: reading them
means an external converter process, and PDF handling was deliberately moved off one of those.
"""

__all__ = ["DocumentExtractionError",
           "supported_extensions",
           "is_supported",
           "extract_text"]

import logging
import pathlib
from typing import Any, Iterable, Iterator

import docx
import docx.table
import odf.namespaces
import odf.opendocument
import odf.teletype
import pptx
import pptx.enum.shapes
import pypdf

logger = logging.getLogger(__name__)


class DocumentExtractionError(Exception):
    """A document file exists but its text could not be extracted (corrupt/encrypted PDF, wrong text encoding)."""


# Extensions read verbatim as UTF-8 plain text. Beyond `.txt` these are all markup a language model reads
# perfectly well as-is, so the list is really a list of who turns up carrying which format:
#
#   - `.md`, and `.rst` as its cousin — the latter is the scientific Python stack's house markup, so it arrives
#     with anyone whose reading is open-source documentation;
#   - `.bib`, and `.tex` alongside it, because someone who has the bibliography almost always has the papers
#     that cite it as well;
#   - `.org`, for those who count the Church of Emacs among their affiliations.
#
# Keep in sync with the formats the librarian offers for ingestion (`raven.librarian.config.llm_docs_exts`) —
# this tuple is the extractor's own capability, that config is the user-facing enable list.
_PLAINTEXT_EXTS = (".txt", ".md", ".rst", ".org", ".bib", ".tex")


def _extract_plaintext(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentExtractionError(f"extract_text: '{path}' is not valid UTF-8 text.") from exc


def _extract_pdf(path: pathlib.Path) -> str:
    # Force the whole page list up front so encryption/corruption errors (which pypdf raises lazily on page
    # access) surface here, as one clean `DocumentExtractionError`, rather than mid-iteration below.
    try:
        reader = pypdf.PdfReader(path)
        pages = list(reader.pages)
    except Exception as exc:  # noqa: BLE001 -- pypdf raises many types on malformed/encrypted input; normalize them
        raise DocumentExtractionError(f"extract_text: '{path}' could not be read as a PDF: "
                                      f"{type(exc).__name__}: {exc}") from exc
    chunks = []
    for page_number, page in enumerate(pages):
        try:
            chunks.append(page.extract_text())
        except Exception as exc:  # noqa: BLE001 -- one unreadable page must not lose the rest of the document
            logger.warning(f"extract_text: '{path}': skipping unreadable page {page_number}: "
                           f"{type(exc).__name__}: {exc}")
    return "\n".join(chunks)


# Cells of one table row are joined by a tab rather than a newline, so that a row survives as a row. Retrieval
# chunks on a sliding window of text, so a table flattened one-cell-per-line loses the association between a
# label and its value — which for a table is most of what it says.
_TABLE_CELL_SEPARATOR = "\t"


def _docx_content_text(container: Any) -> str:
    """Text of a `python-docx` block container — a whole document, or one table cell — in document order.

    Recursive, because a table cell is itself a block container: a table nested in a cell comes out in place.
    """
    chunks = []
    for block in container.iter_inner_content():
        if isinstance(block, docx.table.Table):
            for row in block.rows:
                chunks.append(_TABLE_CELL_SEPARATOR.join(_docx_content_text(cell) for cell in row.cells))
        else:  # a paragraph
            chunks.append(block.text)
    return "\n".join(chunks)


def _extract_docx(path: pathlib.Path) -> str:
    # Headers, footers and comments are deliberately not collected: they repeat on every page or annotate rather
    # than state, so folding them into the body text mostly adds boilerplate for a retrieval index to match on.
    try:
        return _docx_content_text(docx.Document(str(path)))
    except Exception as exc:  # noqa: BLE001 -- python-docx surfaces malformed input as several unrelated types
        raise DocumentExtractionError(f"extract_text: '{path}' could not be read as a Word document: "
                                      f"{type(exc).__name__}: {exc}") from exc


def _pptx_shapes_text(shapes: Iterable[Any]) -> str:
    """Text of a `python-pptx` shape collection — a slide's shapes, or a group's members — in document order.

    Recursive, so text inside a grouped shape is not lost.
    """
    chunks = []
    for shape in shapes:
        if shape.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.GROUP:
            chunks.append(_pptx_shapes_text(shape.shapes))
        elif shape.has_table:
            for row in shape.table.rows:
                chunks.append(_TABLE_CELL_SEPARATOR.join(cell.text for cell in row.cells))
        elif shape.has_text_frame:
            chunks.append(shape.text_frame.text)
    return "\n".join(chunk for chunk in chunks if chunk)


def _extract_pptx(path: pathlib.Path) -> str:
    try:
        presentation = pptx.Presentation(str(path))
        chunks = []
        for slide in presentation.slides:
            chunks.append(_pptx_shapes_text(slide.shapes))
            # Presenter notes are included. Slides tend to carry the claim and the notes the argument for it, so
            # on lecture or conference decks the notes are often where the substance actually is — and a reader
            # asking the corpus a question wants that, projected or not.
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
                chunks.append(slide.notes_slide.notes_text_frame.text)
        return "\n".join(chunk for chunk in chunks if chunk)
    except Exception as exc:  # noqa: BLE001 -- python-pptx surfaces malformed input as several unrelated types
        raise DocumentExtractionError(f"extract_text: '{path}' could not be read as a PowerPoint presentation: "
                                      f"{type(exc).__name__}: {exc}") from exc


# ODF marks a paragraph as `text:p` and a heading as `text:h`; both are leaves as far as we are concerned.
_ODF_PARAGRAPH_QNAMES = frozenset([(odf.namespaces.TEXTNS, "p"),
                                   (odf.namespaces.TEXTNS, "h")])


def _odf_paragraphs(node: Any) -> Iterator[Any]:
    """Yield the paragraph and heading elements under an ODF node, in document order.

    Stops descending at each one it yields. `odf.teletype.extractText` already gathers the text of everything
    nested *inside* a paragraph — a footnote body, a text box anchored as a character — so going deeper would
    emit that text a second time.
    """
    for child in getattr(node, "childNodes", ()):
        if getattr(child, "qname", None) in _ODF_PARAGRAPH_QNAMES:
            yield child
        else:
            yield from _odf_paragraphs(child)


def _extract_odf(path: pathlib.Path) -> str:
    # One reader for both OpenDocument formats we accept: a text document and a presentation differ in how the
    # body is structured, but a walk that only looks for paragraphs does not have to care which it is holding.
    try:
        document = odf.opendocument.load(str(path))
        return "\n".join(odf.teletype.extractText(paragraph) for paragraph in _odf_paragraphs(document.body))
    except Exception as exc:  # noqa: BLE001 -- odfpy surfaces malformed input as several unrelated types
        raise DocumentExtractionError(f"extract_text: '{path}' could not be read as an OpenDocument file: "
                                      f"{type(exc).__name__}: {exc}") from exc


def _extract_html(path: pathlib.Path) -> str:
    # `trafilatura` is imported here rather than at module level because it costs about 0.3 s — three times the
    # whole office stack put together — and most sessions never open an HTML document. `raven.server.modules
    # .webfetch` defers it at its use site for the same reason, and this is the same extractor doing the same
    # job on bytes that arrived from disk instead of from the network.
    import trafilatura  # noqa: PLC0415 -- deferred: keep the readability stack off the common import path

    # Handed over as bytes, not text: an HTML file declares its own encoding in a meta tag or an XML
    # declaration, and a page saved off the web is about as likely to be Latin-1 as UTF-8. Decoding it ourselves
    # would mean either guessing or raising on a file that is perfectly well-formed and says so in its header.
    raw = path.read_bytes()
    try:
        # Markdown rather than bare text, so headings, lists and tables survive as structure the chunker can see
        # — the document database already accepts Markdown as an input format, so nothing downstream is
        # surprised by it. `favor_recall` errs toward keeping borderline content: for a retrieval index, a
        # paragraph wrongly kept costs far less than one wrongly discarded.
        body = trafilatura.extract(raw, output_format="markdown", include_tables=True, favor_recall=True) or ""
        title = _html_title(trafilatura, raw)
    except Exception as exc:  # noqa: BLE001 -- trafilatura surfaces malformed input as several unrelated types
        raise DocumentExtractionError(f"extract_text: '{path}' could not be read as an HTML document: "
                                      f"{type(exc).__name__}: {exc}") from exc

    # Readability extraction keeps the article's own headings but drops `<title>`, which on a saved page is
    # often the only thing that names it — the filename frequently does not. Skipped when the body already
    # opens with that same heading, which is the common case for a page whose `<h1>` restates its title.
    if title and body.lstrip().split("\n", 1)[0].lstrip("#").strip() != title:
        return f"# {title}\n\n{body}" if body else f"# {title}"
    return body


def _html_title(trafilatura: Any, raw: bytes) -> str | None:
    """Best-effort page title from HTML bytes. `None` when there isn't one, or the metadata pass fails."""
    try:
        metadata = trafilatura.extract_metadata(raw)
    except Exception:  # noqa: BLE001 -- a title is a nicety; never let its absence cost us the body text
        return None
    title = getattr(metadata, "title", None) if metadata is not None else None
    return title.strip() if title else None


# The formats that need a real parser, and who parses them. Anything not listed here is read as plain text, so
# this table plus `_PLAINTEXT_EXTS` is the whole capability list — `supported_extensions` derives from the two
# rather than repeating them, which is what keeps the dispatch and the advertised formats from drifting apart.
_EXTRACTORS = {".pdf": _extract_pdf,
               ".docx": _extract_docx,
               ".pptx": _extract_pptx,
               ".odt": _extract_odf,
               ".odp": _extract_odf,
               ".html": _extract_html,
               ".htm": _extract_html}


def supported_extensions() -> tuple[str, ...]:
    """Return all file extensions `extract_text` can handle (lowercase, with the leading dot)."""
    return _PLAINTEXT_EXTS + tuple(_EXTRACTORS)


def is_supported(path: str | pathlib.Path) -> bool:
    """Return whether `extract_text` recognizes `path`'s file extension."""
    return pathlib.Path(path).suffix.lower() in supported_extensions()


def extract_text(path: str | pathlib.Path) -> str | None:
    """Extract indexable plaintext from a document file. See the module docstring for the full contract."""
    p = pathlib.Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"extract_text: no such file: '{p}'")
    # Plain text for the known text extensions, and as a defensive fallback for anything else (the ingester
    # filters by extension upstream, so an unknown suffix reaching here is already an unusual case).
    extract = _EXTRACTORS.get(p.suffix.lower(), _extract_plaintext)
    text = extract(p).strip()
    return text or None
