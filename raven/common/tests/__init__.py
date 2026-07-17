"""Shared test utilities for Raven."""

__all__ = ["approx",
           "make_minimal_pdf",
           "make_textless_pdf"]


def approx(a, b, tol=0.01):
    """Check approximate float equality."""
    return abs(a - b) < tol


def make_minimal_pdf(text: str) -> bytes:
    """Build a minimal but valid single-page PDF whose text layer is `text`, with a correct xref table.

    `text` is embedded in a `Tj` text-showing operator, so a born-digital PDF text extractor (pypdf) reads it
    back verbatim. Restricted to Latin-1 for the content stream. Returns the PDF file bytes, ready to write to
    disk — used as an on-the-fly fixture so tests need no committed binary.
    """
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        None,  # contents stream, filled in below (its length depends on `text`)
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = b"BT /F1 24 Tf 72 700 Td (" + text.encode("latin-1") + b") Tj ET"
    objs[3] = b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)
    return _assemble_pdf(objs)


def make_textless_pdf() -> bytes:
    """Build a valid single-page PDF with no text-showing operators — a stand-in for a scanned/image-only page.

    A born-digital text extractor finds no text in it, so it exercises the "parses cleanly but yields no text"
    path (which should come back as empty rather than as an error).
    """
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
        b"<< /Length 0 >>\nstream\n\nendstream",  # empty content stream: nothing to extract
    ]
    return _assemble_pdf(objs)


def _assemble_pdf(objs: list) -> bytes:
    """Assemble a list of PDF object bodies (1-indexed) into a complete PDF with a correct xref table."""
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, body)
    xref_pos = len(out)
    out += b"xref\n0 %d\n" % (len(objs) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref_pos)
    return bytes(out)
