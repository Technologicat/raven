"""Fill a bibliography's missing and truncated abstracts from Crossref.

A literature search does not always deliver the abstract a publisher wrote. Some records arrive with none,
and some with a teaser cut at a fixed length by the database or its export — a couple of hundred
characters ending in `...`. Both can often be put right from Crossref, which serves the abstract as the
publisher deposited it, looked up by DOI.

    raven-fillbib corpus.bib

**A record's abstract is replaced only when ours is a prefix of Crossref's**, compared on letters and
digits alone, so that markup, punctuation and spacing cannot make two copies of one text look different.
That one rule both finds the truncated abstracts — no heuristic for "looks cut off" is needed — and
checks that what replaces them is the same text continued, rather than another version of it. Where the
two differ, the record keeps its own and the audit lists it for a person to look at. A record with no
abstract at all takes Crossref's.

**What it cannot do.** Only records carrying a DOI can be looked up: finding the DOI of a record that has
none would mean matching it by title and authors, which can pick the wrong paper, and a wrong abstract
filled in silently is worse than a missing one. And Crossref has only what publishers deposit, which
varies by publisher from nearly everything to nothing at all. Every record this run could not help is
in the audit, with the reason.

**The abstracts Crossref serves remain their publisher's or authors' copyright.** Crossref's own metadata
is free to reuse; abstracts are the exception, distributed under its agreement with its members. Use the
filled bibliography to screen and study the literature, and do not publish it.

The filled abstract is written as LaTeX, BibTeX's own markup: italics, bold, super- and subscripts, and a
structured abstract's section headings in bold, each section its own paragraph. A plain language
summary, which some publishers deposit as a section of the abstract, is left out.

Some database exports append a second text to the abstract — a plain language summary, a graphical
abstract, an alternate abstract, each under its heading. Where a record keeps its own abstract, that is
removed too, and the audit lists the record; `--keep-appended` leaves it in.
"""

from __future__ import annotations

__all__ = ["render_jats", "comparable",

           "FILLED", "REPAIRED", "DIFFERS", "SAME", "NO_DOI", "NOT_IN_CROSSREF", "NO_ABSTRACT_IN_CROSSREF",
           "LOOKUP_FAILED", "LOOKS_TRUNCATED", "APPENDED_REMOVED",
           "AuditRow", "AUDIT_COLUMNS", "words_shared", "fill", "audit_rows", "write_audit",

           "fetch_crossref",

           "main"]

import argparse
import copy
import dataclasses
import difflib
import html
import json
import logging
import pathlib
import re
import sys
import time
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Iterable, Mapping

import bibtexparser
from bibtexparser.model import Field

from ..common import tabular
from ..common import text as textutil
from ..common import utils as common_utils

from .. import __version__

from . import bibtex
from . import config as papers_config
from . import utils as papers_utils

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------
# JATS → LaTeX

# Crossref's abstracts are JATS XML fragments using these prefixes without declaring them, which a parser
# refuses; wrapping the fragment in an element that declares them is what makes it parseable.
_NAMESPACES = {"jats": "http://www.ncbi.nlm.nih.gov/JATS1",
               "mml": "http://www.w3.org/1998/Math/MathML",
               "xlink": "http://www.w3.org/1999/xlink",
               "ali": "http://www.niso.org/schemas/ali/1.0/"}
_WRAPPER_OPEN = "<fillbib-root " + " ".join(f'xmlns:{prefix}="{uri}"' for prefix, uri in _NAMESPACES.items()) + ">"
_WRAPPER_CLOSE = "</fillbib-root>"

_INLINE_STYLES = {"italic": r"\textit", "bold": r"\textbf", "underline": r"\underline",
                  "sup": r"\textsuperscript", "sub": r"\textsubscript"}

_HEADING_ONLY = re.compile(r"^\s*(abstract|summary)\s*[:.]?\s*$", re.IGNORECASE)
_PLAIN_LANGUAGE_SUMMARY = re.compile(r"\bplain[\s-]+language\s+summary\b", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


def _local(tag: str) -> str:
    """An element's tag without its namespace: `{uri}italic` → `italic`."""
    return tag.rpartition("}")[2]


def _escaped(text: str | None) -> str:
    return papers_utils.bibtex_escape(text) if text else ""


def _inline(element: ElementTree.Element) -> str:
    """The content of `element` as LaTeX, its markup reduced to what BibTeX can carry."""
    parts = [_escaped(element.text)]
    for child in element:
        parts.append(_inline_element(child))
        parts.append(_escaped(child.tail))
    return "".join(parts)


def _inline_element(element: ElementTree.Element) -> str:
    name = _local(element.tag)
    if name in _INLINE_STYLES:
        content = _inline(element)
        return f"{_INLINE_STYLES[name]}{{{content}}}" if content.strip() else content
    if name == "math":
        return _math(element)
    if name == "alternatives":
        # The same formula more than once — MathML, TeX, an image. MathML is the one we can render as text.
        for child in element:
            if _local(child.tag) == "math":
                return _inline_element(child)
        return ""
    if name == "tex-math":
        return ""  # raw TeX, often a whole document preamble; not text
    # Everything else — small caps, links, cross-references, the MathML token elements — keeps its text.
    return _inline(element)


_MATH_TOKENS = ("mi", "mn", "mo", "mtext", "ms")
_SPACED_OPERATORS = frozenset("=<>≤≥≈≠±×→")


def _math(element: ElementTree.Element) -> str:
    """A MathML formula as text, its scripts as LaTeX text-mode ones.

    Text-mode scripts are what the import pipeline turns into Unicode characters where they exist (H₂O,
    x², ¹³C). Whitespace between MathML elements means nothing, and Crossref's is indentation, so only the
    tokens' own text is kept.
    """
    name = _local(element.tag)
    children = list(element)
    if name in _MATH_TOKENS:
        text = _escaped("".join(element.itertext()).strip())
        if name == "mo" and text in _SPACED_OPERATORS:
            return f" {text} "
        if name == "mo" and text in (",", ";"):
            return f"{text} "
        return text
    if name == "mspace":
        return " "
    if name in ("msub", "msup") and len(children) == 2:
        command = r"\textsubscript" if name == "msub" else r"\textsuperscript"
        return f"{_math(children[0])}{command}{{{_math(children[1]).strip()}}}"
    if name == "msubsup" and len(children) == 3:
        return (rf"{_math(children[0])}\textsubscript{{{_math(children[1]).strip()}}}"
                rf"\textsuperscript{{{_math(children[2]).strip()}}}")
    if name == "mfrac" and len(children) == 2:
        return f"{_math(children[0]).strip()}/{_math(children[1]).strip()}"
    if name == "annotation":
        return ""  # another rendering of the same formula
    return "".join(_math(child) for child in children)


def _paragraph(text: str) -> str:
    return " ".join(text.split())


def _blocks(element: ElementTree.Element, *, top: bool) -> list[str]:
    """The paragraphs of `element`, in order. A structured abstract's section heading is one of them."""
    blocks = []
    if element.text and element.text.strip():
        blocks.append(_paragraph(_escaped(element.text)))
    skipping = False  # after a top-level "Plain language summary" heading, until the next heading
    for child in element:
        name = _local(child.tag)
        if name == "sec":
            skipping = False  # a section ends whatever a loose heading started
            titles = [grandchild for grandchild in child if _local(grandchild.tag) == "title"]
            if not (titles and _PLAIN_LANGUAGE_SUMMARY.search("".join(titles[0].itertext()))):
                blocks.extend(_blocks(child, top=False))
        elif name == "title":
            title_text = "".join(child.itertext())
            if _PLAIN_LANGUAGE_SUMMARY.search(title_text):
                skipping = True
            elif top and _HEADING_ONLY.match(title_text):
                skipping = False  # the word "Abstract" heading the abstract says nothing
            else:
                skipping = False
                heading = _paragraph(_inline(child))
                if heading:
                    blocks.append(rf"\textbf{{{heading}}}")
        elif not skipping:
            if name in ("p", "list-item", "disp-quote", "statement"):
                blocks.append(_paragraph(_inline(child)))
            elif name in ("list", "boxed-text"):
                blocks.extend(_blocks(child, top=False))
            else:
                blocks.append(_paragraph(_inline_element(child)))
        if child.tail and child.tail.strip() and not skipping:
            blocks.append(_paragraph(_escaped(child.tail)))
    return [block for block in blocks if block]


def render_jats(jats: str) -> str:
    """Render a JATS abstract, as Crossref serves it, as a BibTeX field value in LaTeX.

    Paragraphs are separated by a blank line. A structured abstract's section headings are bold paragraphs
    of their own; a leading heading that only says "Abstract" is dropped, and so is a plain language
    summary. Italic, bold, underline and super- and subscripts carry over as LaTeX commands, MathML's
    scripts included, and the text is escaped for BibTeX.

    Input that is not well-formed XML falls back to its text with the tags removed.
    """
    # Refused rather than parsed: a document type declaration is how an XML entity attack arrives, and no
    # abstract needs one.
    if "<!DOCTYPE" in jats or "<!ENTITY" in jats:
        return _paragraph(_escaped(html.unescape(_TAG.sub(" ", jats))))
    try:
        root = ElementTree.fromstring(_WRAPPER_OPEN + jats + _WRAPPER_CLOSE)
    except ElementTree.ParseError:
        logger.debug("render_jats: not well-formed XML; using its text with the tags removed")
        return _paragraph(_escaped(html.unescape(_TAG.sub(" ", jats))))
    return "\n\n".join(_blocks(root, top=True))


# --------------------------------------------------------------------------------
# Comparing two copies of an abstract


# Below this many letters and digits, a matching prefix says too little to identify the text it begins:
# a stray "[No abstract available]" or a lone heading would match any abstract opening with those words.
_MIN_PREFIX_CHARS = 40

# The headings under which a database export appends a second text to the abstract. ProQuest's field can
# hold the abstract, then one of these, then the start of the second text, each part cut short:
# `…a promising approach ...Plain language summaryThis study explores…`. The same words also occur in
# prose about plain language summaries, so a heading must stand where a heading can: at the start of the
# text or of a line, or after a sentence end — the teaser's `...` included, and any closing brackets and
# quotes after it, in every spelling an export uses, since nothing has normalized them at this point.
# What must follow it is checked by `_starts_appended_text`.
_APPENDED_TEXT = re.compile(r"""(?imx)
    (?: ^ | [.!?…] [)\]"'’”»]* )  # `…` …, `’` ’, `”` ”, `»` »
    \s*
    (?P<heading> plain[\s-]+language\s+summary | graphical\s+abstract | alternate\s+abstract )
""")


def _starts_appended_text(text: str, end: int) -> bool:
    """Whether a heading ending at `end` in `text` is followed as a heading is: by a colon, a line break,
    the end of the text, or a capital letter — glued on, as the exports have it, or after spaces."""
    rest = text[end:].lstrip(" \t")
    return not rest or rest[0] in ":\n" or rest[0].isupper()


def _without_appended_text(text: str) -> str:
    """`text` up to the heading of a second text appended to it, or all of `text` where there is none."""
    for match in _APPENDED_TEXT.finditer(text):
        if _starts_appended_text(text, match.end("heading")):
            return text[:match.start("heading")]
    return text


def comparable(text: str) -> str:
    """The form in which two copies of an abstract are compared: letters and digits only, casefolded.

    Before that, everything from a second text appended to the abstract onward is removed — a plain
    language summary, a graphical abstract, an "alternate abstract" in another language — and so are a
    trailing ellipsis, a leading "Abstract:" label and a trailing rights notice, and LaTeX and HTML markup
    is reduced to text. What is left is the abstract's wording and nothing else, so that two exports of one
    text compare equal however each one punctuated, spaced, marked up, labelled or extended it.
    """
    text = _without_appended_text(text)
    text = papers_utils.without_trailing_ellipsis(text)
    text = common_utils.unicodize_basic_markup(text)
    text = _TAG.sub(" ", text)
    text = textutil.strip_boilerplate(text)
    return "".join(character for character in text.casefold() if character.isalnum())


# --------------------------------------------------------------------------------
# Filling

# The outcomes a record can have, as the audit names them. Written for a reader who sees only the audit:
# each says what happened, and where nothing did, why.
FILLED = "filled: had no abstract"
REPAIRED = "repaired: ours was the start of Crossref's"
DIFFERS = "kept ours: Crossref's is a different text"
SAME = "kept ours: the same text as Crossref's"
NO_DOI = "no DOI, cannot retrieve"
NOT_IN_CROSSREF = "DOI not found in Crossref"
NO_ABSTRACT_IN_CROSSREF = "Crossref has no abstract"
LOOKUP_FAILED = "lookup failed"

# Appended to the outcome of a record that kept an abstract ending in an ellipsis. The ellipsis itself is
# at the far end of a long cell, where a spreadsheet puts it a screen or more out of view.
LOOKS_TRUNCATED = "ours looks truncated"

# Appended to the outcome of a record that kept its own abstract and had a second text removed from it.
APPENDED_REMOVED = "second text removed"

_CHANGED = (FILLED, REPAIRED)


@dataclasses.dataclass(frozen=True)
class AuditRow:
    """One record the run changed, or could not help.

    `before` is the record's abstract as it was and `crossref` the one Crossref has, both in readable form;
    the record now carries `crossref` where the outcome is a fill or a repair, and `before` otherwise.
    `maybe_shared` is `words_shared` of the two, or `None` where there are not two to compare. `link` is where
    to read the paper (see `raven.papers.utils.paper_url`).
    """
    key: str
    outcome: str
    title: str
    venue: str
    link: str = ""
    maybe_shared: float | None = None
    before: str = ""
    crossref: str = ""


AUDIT_COLUMNS = ("key", "outcome", "% of ours in Crossref's", "title", "venue", "link",
                 "abstract before", "Crossref's abstract")


def _words(text: str) -> list[str]:
    """`text` as the words `comparable` would keep, in order."""
    text = _without_appended_text(text)
    text = textutil.strip_boilerplate(_TAG.sub(" ", common_utils.unicodize_basic_markup(text)))
    return re.findall(r"\w+", text.casefold())


def words_shared(ours: str, theirs: str) -> float:
    """The share of the words of `ours` that appear in `theirs`, in the same order: 1.0 when `ours` is all in
    there, near 0.0 when the two are different texts.

    Measured against `ours` rather than against both, so that a short teaser that is the start of a long
    abstract scores high: the question is whether our text is in Crossref's, not whether the two are equally
    long. Compared as `comparable` compares, on words rather than letters.
    """
    ours_words, theirs_words = _words(ours), _words(theirs)
    if not ours_words:
        return 0.0
    matcher = difflib.SequenceMatcher(None, ours_words, theirs_words, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / len(ours_words)


def _needs_one(abstract: str) -> bool:
    """Whether a record visibly lacks a whole abstract: it has none, or one ending in an ellipsis.

    Used only to decide which records *without a lookup* are worth listing in the audit. A record that was
    looked up is judged by the prefix rule instead, which needs no guess about what a cut-off looks like.
    """
    return not abstract.strip() or papers_utils.looks_truncated(abstract)


def _without_appended_abstract(abstract: str) -> str:
    """`abstract` without a second text appended to it, or unchanged where removing one would leave no text."""
    cut = _without_appended_text(abstract)
    return cut.rstrip() if cut != abstract and comparable(cut) else abstract


def fill(library: bibtexparser.Library,
         abstracts: Mapping[str, str | None],
         *,
         failed: Mapping[str, str] | None = None,
         strip_appended: bool = True) -> tuple[bibtexparser.Library, list[AuditRow]]:
    """Fill and repair `library`'s abstracts from `abstracts`, and account for what happened.

    `abstracts` maps a normalized DOI (see `raven.papers.utils.normalize_doi`) to the abstract found for it,
    as a BibTeX field value, or to `None` where the record was found and has no abstract. A DOI absent from
    it was not found. `failed` maps a DOI to why its lookup failed.

    `strip_appended`: where a record keeps its own abstract, remove a second text appended to it — a plain
    language summary, a graphical abstract, an alternate abstract — from its heading on. Not done where
    the second text is all the abstract there is.

    Returns `(filled, audit)`. `filled` is a copy of `library`; the input is not modified. The audit lists
    every record that was changed, every record whose abstract differs from the one found, and every record
    that has no abstract or one ending in an ellipsis and could not be helped — with no DOI, not found,
    nothing found, or a failed lookup.
    """
    failed = failed or {}
    filled = copy.deepcopy(library)
    audit = []
    for entry in filled.entries:
        ours = bibtex.entry_field(entry, "abstract")
        doi = papers_utils.normalize_doi(bibtex.entry_field(entry, "doi"))
        theirs = None
        if doi is None:
            outcome = NO_DOI
        elif doi in failed:
            outcome = f"{LOOKUP_FAILED}: {failed[doi]}"
        elif doi not in abstracts:
            outcome = NOT_IN_CROSSREF
        elif not abstracts[doi]:
            outcome = NO_ABSTRACT_IN_CROSSREF
        else:
            theirs = abstracts[doi]
            ours_key, theirs_key = comparable(ours), comparable(theirs)
            if not ours_key:
                outcome = FILLED
            elif ours_key == theirs_key:
                outcome = SAME
            elif len(ours_key) >= _MIN_PREFIX_CHARS and theirs_key.startswith(ours_key):
                outcome = REPAIRED
            else:
                outcome = DIFFERS

        if outcome in _CHANGED:
            entry.set_field(Field("abstract", theirs))
        else:
            kept = _without_appended_abstract(ours) if strip_appended else ours
            if kept != ours:
                entry.set_field(Field("abstract", kept))
                outcome = f"{outcome}; {APPENDED_REMOVED}"
            elif outcome == SAME or (outcome != DIFFERS and not _needs_one(ours)):
                continue  # nothing changed, and nothing visibly missing: not worth a reader's time
            if papers_utils.looks_truncated(kept):
                outcome = f"{outcome}; {LOOKS_TRUNCATED}"
        audit.append(AuditRow(key=entry.key,
                              outcome=outcome,
                              title=bibtex.readable_text(bibtex.entry_field(entry, "title")),
                              venue=bibtex.entry_venue(entry),
                              link=papers_utils.paper_url(bibtex.entry_field(entry, "doi"),
                                                          bibtex.entry_field(entry, "url")),
                              maybe_shared=words_shared(ours, theirs) if (ours.strip() and theirs) else None,
                              before=bibtex.readable_text(ours),
                              crossref=bibtex.readable_text(theirs or "")))
    return filled, audit


def audit_rows(audit: list[AuditRow]) -> list[tuple]:
    """`audit` as table rows, in the order of `AUDIT_COLUMNS`."""
    return [(row.key, row.outcome,
             "" if row.maybe_shared is None else round(100 * row.maybe_shared),
             row.title, row.venue, row.link, row.before, row.crossref)
            for row in audit]


def write_audit(path: pathlib.Path, audit: list[AuditRow], sources: list[str]) -> None:
    """Write the audit to `path`, as TSV, `.xlsx` or `.ods` by its extension, with lines naming the tool
    version and inputs: comment lines above the header in a TSV, a *Notes* sheet in a spreadsheet."""
    tabular.write_table(path,
                        [f"raven-fillbib {__version__}",
                         f"input: {'; '.join(sources)}",
                         "abstracts from: Crossref (api.crossref.org); copyright of each abstract rests with "
                         "its publisher or authors",
                         f"records listed: {len(audit)}"],
                        AUDIT_COLUMNS,
                        audit_rows(audit))


# --------------------------------------------------------------------------------
# Crossref

def _load_cache(path: pathlib.Path) -> dict[str, dict]:
    cache = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                cache[record["doi"]] = record
    return cache


def fetch_crossref(dois: Iterable[str],
                   *,
                   maybe_cache_path: pathlib.Path | None = None,
                   batch_size: int | None = None,
                   delay: float | None = None,
                   get: Callable | None = None) -> tuple[dict[str, str | None], dict[str, str]]:
    """Look up `dois` in Crossref, and return their abstracts rendered as BibTeX field values.

    `dois` are normalized DOIs (see `raven.papers.utils.normalize_doi`). Returns `(abstracts, failed)` in the
    shape `fill` takes: `abstracts` maps each DOI Crossref knows to its rendered abstract, or to `None`
    where it has none; `failed` maps a DOI whose lookup failed to why.

    `maybe_cache_path`: a JSONL file of Crossref's answers. DOIs answered there are not asked again, and
    new answers are appended — so a rerun asks Crossref nothing it has answered, which Crossref asks of
    its users. The raw answer is what is stored, so a change in rendering does not need a new lookup.

    `batch_size` and `delay` default to `raven.papers.config`'s `crossref_doi_batch_size` and
    `crossref_request_delay`. `get` defaults to `raven.papers.httpfetch.polite_get`, and is there for tests.
    """
    if get is None:
        from . import httpfetch
        get = httpfetch.polite_get
    batch_size = batch_size or papers_config.crossref_doi_batch_size
    delay = papers_config.crossref_request_delay if delay is None else delay

    cache = _load_cache(maybe_cache_path) if maybe_cache_path else {}
    wanted = list(dict.fromkeys(dois))
    todo = [doi for doi in wanted if doi not in cache]
    # A comma separates the values of Crossref's filter, so a DOI containing one is asked for on its own.
    batches = [[doi] for doi in todo if "," in doi]
    plain = [doi for doi in todo if "," not in doi]
    batches += [plain[i:i + batch_size] for i in range(0, len(plain), batch_size)]

    failed = {}
    cache_file = maybe_cache_path.open("a", encoding="utf-8") if maybe_cache_path else None
    try:
        for index, batch in enumerate(batches):
            if index:
                time.sleep(delay)
            logger.info(f"fetch_crossref: batch {index + 1}/{len(batches)}, {len(batch)} DOIs")
            try:
                response = get(papers_config.crossref_api_url,
                               params={"filter": ",".join(f"doi:{doi}" for doi in batch),
                                       "select": "DOI,abstract",
                                       "rows": len(batch)},
                               timeout=60)
                response.raise_for_status()
                items = response.json()["message"]["items"]
            except Exception as exc:  # one failed batch costs its own DOIs, not the run
                logger.warning(f"fetch_crossref: batch {index + 1} failed: {type(exc).__name__}: {exc}")
                failed.update((doi, f"{type(exc).__name__}: {exc}") for doi in batch)
                continue
            answered = {papers_utils.normalize_doi(item.get("DOI")): item.get("abstract") for item in items}
            for doi in batch:
                record = {"doi": doi, "found": doi in answered, "abstract": answered.get(doi)}
                cache[doi] = record
                if cache_file:
                    cache_file.write(json.dumps(record) + "\n")
                    cache_file.flush()
    finally:
        if cache_file:
            cache_file.close()

    abstracts = {}
    for doi in wanted:
        record = cache.get(doi)
        if record is None or not record["found"]:
            continue
        abstracts[doi] = render_jats(record["abstract"]) if record["abstract"] else None
    return abstracts, failed


# --------------------------------------------------------------------------------
# Command line

_OUTCOME_ORDER = (FILLED, REPAIRED, DIFFERS, SAME, NO_DOI, NOT_IN_CROSSREF, NO_ABSTRACT_IN_CROSSREF)


def _report(total: int, audit: list[AuditRow]) -> None:
    """Print what the run did, one line per outcome."""
    print(f"{total} records")
    counts: dict[str, int] = {}
    truncated: dict[str, int] = {}
    removed = 0
    for row in audit:
        outcome = LOOKUP_FAILED if row.outcome.startswith(LOOKUP_FAILED) else row.outcome.split("; ")[0]
        counts[outcome] = counts.get(outcome, 0) + 1
        if row.outcome.endswith(LOOKS_TRUNCATED):
            truncated[outcome] = truncated.get(outcome, 0) + 1
        if f"; {APPENDED_REMOVED}" in row.outcome:
            removed += 1
    for outcome in (*_OUTCOME_ORDER, LOOKUP_FAILED):
        if counts.get(outcome):
            maybe_truncated = f" ({truncated[outcome]} of them {LOOKS_TRUNCATED})" if truncated.get(outcome) else ""
            print(f"  {counts[outcome]:>6}  {outcome}{maybe_truncated}")
    if removed:
        print(f"  {removed:>6}  of the abstracts kept: {APPENDED_REMOVED}")


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Fill missing and truncated abstracts in a BibTeX file from "
                                                 "Crossref, by DOI, and write an audit of what changed and "
                                                 "what could not be helped. An abstract is replaced only "
                                                 "where ours is the start of Crossref's. Abstracts remain "
                                                 "their publisher's or authors' copyright: use the result "
                                                 "for your own screening, and do not publish it.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-o", "--output-suffix", dest="output_suffix", default="_filled", type=str,
                        metavar="suf", help="Suffix for naming output files (file.bib -> file_filled.bib).")
    parser.add_argument("--audit-suffix", dest="audit_suffix", default="_fill_audit", type=str,
                        metavar="suf", help="Suffix for naming audit files (file.bib -> file_fill_audit.xlsx).")
    parser.add_argument("--audit-format", dest="audit_format", default="xlsx", choices=tabular.FORMATS,
                        help="Format of the audit files: Excel, OpenDocument or TSV.")
    parser.add_argument("--out-dir", dest="out_dir", default=None, type=str, metavar="DIR",
                        help="Write the filled bibliographies and their audits here, instead of beside each input file. Created if it does not exist.")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--cache", dest="cache", default=None, type=str, metavar="FILE",
                       help="Keep Crossref's answers in this JSONL file, and ask only for DOIs it does not hold. Crossref asks its users to cache. Default: <bib stem>.crossref.jsonl beside the output.")
    cache.add_argument("--no-cache", dest="no_cache", action="store_true", default=False,
                       help="Keep no cache: ask Crossref about every DOI, every run.")
    parser.add_argument("--keep-appended", dest="keep_appended", action="store_true", default=False,
                        help="Where a record keeps its own abstract, keep a second text appended to it — a plain language summary, a graphical abstract, an alternate abstract. By default it is removed, and the audit lists the record.")
    parser.add_argument("-n", "--dry-run", dest="dry_run", action="store_true", default=False,
                        help="Look up and report what would change, and write nothing but the cache.")
    parser.add_argument("--log", metavar="PATH", default=None,
                        help="Write the log here instead of to stderr.")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
                        help="How much to log.")
    parser.add_argument(dest="filenames", nargs="+", default=None, type=str, metavar="myreferences.bib",
                        help="BibTeX file(s) to fill.")
    opts = parser.parse_args()

    from ..common import logsetup
    logsetup.configure(level=getattr(logging, opts.log_level), logfile=opts.log)

    # `bibtexparser` logs a warning per record it cannot read. Those are `raven-fixbib`'s business.
    logging.getLogger("bibtexparser").setLevel(logging.ERROR)

    maybe_given_cache = pathlib.Path(opts.cache).expanduser().resolve() if opts.cache else None
    for filename in opts.filenames:
        path = pathlib.Path(filename).expanduser().resolve()
        try:
            library = bibtex.parse_file(str(path), split_names=False)
        except OSError as exc:
            print(f"{path}: cannot read ({type(exc).__name__}: {exc})", file=sys.stderr)
            continue
        directory = pathlib.Path(opts.out_dir).expanduser().resolve() if opts.out_dir else path.parent
        if opts.no_cache:
            maybe_cache_path = None
        else:
            maybe_cache_path = maybe_given_cache or directory / f"{path.stem}.crossref.jsonl"
            maybe_cache_path.parent.mkdir(parents=True, exist_ok=True)

        dois = [doi for doi in (papers_utils.normalize_doi(bibtex.entry_field(entry, "doi"))
                                for entry in library.entries) if doi]
        abstracts, failed = fetch_crossref(dois, maybe_cache_path=maybe_cache_path)
        filled, audit = fill(library, abstracts, failed=failed, strip_appended=not opts.keep_appended)
        print(f"\n{path}")
        _report(len(library.entries), audit)

        if opts.dry_run:
            continue
        directory.mkdir(parents=True, exist_ok=True)
        out_path = directory / f"{path.stem}{opts.output_suffix}{path.suffix}"
        out_path.write_text(bibtex.write_string(filled), encoding="utf-8")
        print(f"  wrote {out_path}")
        audit_path = directory / f"{path.stem}{opts.audit_suffix}.{opts.audit_format}"
        write_audit(audit_path, audit, [str(path)])
        print(f"  wrote {audit_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
