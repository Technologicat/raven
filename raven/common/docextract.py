"""Extract indexable plaintext from document files (plain text and PDF).

This is the single source of truth for "given a document file, give me its text" across Raven. The RAG
document-database ingester (`raven.librarian.hybridir`), the chat text/PDF attachment feature, and the
`raven-pdf2bib` tool all route through here, so there is exactly one PDF backend and one supported-format list.

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
"""

__all__ = ["DocumentExtractionError",
           "supported_extensions",
           "is_supported",
           "extract_text"]

import logging
import pathlib

import pypdf

logger = logging.getLogger(__name__)


class DocumentExtractionError(Exception):
    """A document file exists but its text could not be extracted (corrupt/encrypted PDF, wrong text encoding)."""


# Extensions read verbatim as UTF-8 plain text. Keep in sync with the formats the librarian offers for ingestion
# (`raven.librarian.config.llm_docs_exts`) — this tuple is the extractor's own capability, that config is the
# user-facing enable list.
_PLAINTEXT_EXTS = (".txt", ".md", ".rst", ".org", ".bib", ".tex")
_PDF_EXTS = (".pdf",)


def supported_extensions() -> tuple[str, ...]:
    """Return all file extensions `extract_text` can handle (lowercase, with the leading dot)."""
    return _PLAINTEXT_EXTS + _PDF_EXTS


def is_supported(path: str | pathlib.Path) -> bool:
    """Return whether `extract_text` recognizes `path`'s file extension."""
    return pathlib.Path(path).suffix.lower() in supported_extensions()


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


def extract_text(path: str | pathlib.Path) -> str | None:
    """Extract indexable plaintext from a document file. See the module docstring for the full contract."""
    p = pathlib.Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"extract_text: no such file: '{p}'")
    suffix = p.suffix.lower()
    if suffix in _PDF_EXTS:
        text = _extract_pdf(p)
    else:
        # Plain text for the known text extensions, and as a defensive fallback for anything else (the ingester
        # filters by extension upstream, so an unknown suffix reaching here is already an unusual case).
        text = _extract_plaintext(p)
    text = text.strip()
    return text or None
