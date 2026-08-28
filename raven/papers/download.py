"""Download papers from arXiv by their IDs.

Fetches metadata from the arXiv API and downloads PDFs, naming files
automatically from the paper metadata.

Thanks to Qwen3-30B-A3B-Thinking-2507 and the documentation:
   https://info.arxiv.org/help/api/user-manual.html
   https://info.arxiv.org/help/arxiv_identifier.html
"""

from __future__ import annotations

__all__ = [
    "ArxivMetadataError",
    "format_years",
    "format_filename",

    "parse_metadata_response",
    "metadata_to_feed_entry",
    "get_paper_metadata",
    "parse_metadata_responses",
    "get_papers_metadata",

    "download_papers",
    "extract_ids_from_bib",
    "main",
]

import argparse
import collections
import os
import pathlib
import re
import sys

import traceback
from typing import Dict, List, Union
import xml.etree.ElementTree as ET

import bibtexparser

from mcpyrate import colorizer

from .. import __version__
from ..common import stringmaps
from . import bibtex
from . import httpfetch
from . import identifiers
from .ratelimit import RateLimiter
from .utils import deduplicate_arxiv_ids

GLOBE = "\U0001f310"  # 🌐 — for progress messages indicating internet access
CHECKMARK = "\u2713"  # ✓
CROSS = "\u2717"      # ✗


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_NS = {"arxiv": "http://arxiv.org/schemas/atom"}

# Identifiers per metadata request. arXiv answers an `id_list` naming many papers in one response, and
# the rate limit is per *request*, so the metadata for a whole run costs ceil(N / 100) waits instead of
# N. The PDFs still cost one wait each, so this halves a run's wall time rather than eliminating it.
METADATA_BATCH_SIZE = 100


class ArxivMetadataError(ValueError):
    """Raised when an arXiv API response carries no usable paper metadata.

    Typically a nonexistent or malformed arXiv ID (e.g. a typoed month):
    arXiv answers with a well-formed but entry-less Atom feed, so there is
    no paper to parse. Inherits `ValueError` so existing broad handlers
    still catch it; `download_papers` catches it specifically to report the
    offending ID without a traceback — an expected user error, not a bug.
    """


def format_years(original_year: str,
                 version_year: str | None) -> str:
    """Render the publication-year parenthetical for a reference.

    ``"(2023)"`` normally, or ``"(2023, revised 2024)"`` when *version_year*
    differs from *original_year* (a later revision of the paper).
    """
    if version_year is not None and version_year != original_year:
        return f"({original_year}, revised {version_year})"
    return f"({original_year})"


def format_filename(arxiv_id: str,
                    authors: list[str],
                    original_year: str,
                    version_year: str | None,
                    title: str,
                    version: str,
                    title_length_limit: int = 128) -> tuple[str, str, str]:
    """Build the canonical output filename for an arXiv paper.

    Returns ``(author_str, resolved_id, filename)`` where *filename* has the
    shape ``"Authors (Year[, revised Year2]) - Title - arxivid.pdf"`` and
    *resolved_id* is the input *arxiv_id* with its version suffix replaced
    by the supplied *version* (so bare IDs get canonicalized to include
    their version, and a mismatched version is overwritten).
    """
    author_str = " and ".join(authors[:2])
    if len(authors) > 2:
        author_str += " et al."
    elif not authors:
        author_str = "Unknown"

    # Normalize separators that the safe-char filter (below) would otherwise
    # drop, leaving the title mashed together. A ":" / "?" / "!" / ";" used as
    # a clause boundary (punctuation + space) becomes " - "
    # (e.g. "…Own Exploration? Gradient-Guided…" → "…Own Exploration - Gradient-Guided…").
    # Em/en dashes (dropped, leaving a double space) and a compound-joining "/"
    # (dropped, mashing the two sides — "Twitter/X" → "TwitterX") become a plain
    # "-", which is in the safe set. "/" has too many senses (or / and / per /
    # ratio) for any word to fit, so "-" is a neutral stand-in that at least
    # keeps the sides distinct.
    for separator in (": ", "? ", "! ", "; "):
        title = title.replace(separator, " - ")
    title = title.replace("—", "-").replace("–", "-").replace("/", "-")
    safe_title = "".join(c for c in title if c.isalnum() or c in stringmaps.filename_safe_nonalphanum)
    safe_title = safe_title[:title_length_limit] + ("..." if len(title) > title_length_limit else "")

    # Canonize ID to always include the version
    clean_id = identifiers.strip_version(arxiv_id)
    resolved_id = f"{clean_id}{version}"

    safe_resolved_id = resolved_id.replace("/", "_")
    safe_resolved_id = "".join(c for c in safe_resolved_id if c.isalnum() or c in stringmaps.filename_safe_nonalphanum)

    filename = f"{author_str} {format_years(original_year, version_year)} - {safe_title} - {safe_resolved_id}.pdf"
    return author_str, resolved_id, filename


def parse_metadata_response(xml_content: bytes,
                            arxiv_id: str,
                            title_length_limit: int = 128) -> Dict[str, str]:
    """Parse an arXiv API Atom response into our internal metadata dict.

    Pure function — no network access.  *xml_content* is the raw body from
    a call to ``http://export.arxiv.org/api/query?id_list=<arxiv_id>``;
    *arxiv_id* is the original query ID, used to derive ``resolved_id``.
    """
    root = ET.fromstring(xml_content)
    entry = root.find(".//atom:entry", ATOM_NS)
    if entry is None:
        # arXiv returns an entry-less feed for a nonexistent or malformed ID
        # (e.g. a typoed month, as in "2614.19062"). Fail with something
        # readable instead of an AttributeError from the next .find().
        raise ArxivMetadataError(f"no arXiv entry for ID '{arxiv_id}' (nonexistent or malformed ID?)")
    return _metadata_from_entry(entry, arxiv_id, title_length_limit)


def _metadata_from_entry(entry: ET.Element,
                         arxiv_id: str,
                         title_length_limit: int = 128) -> Dict[str, str]:
    """Build the metadata dict from one already-located Atom ``<entry>`` element.

    Split out from `parse_metadata_response` so that the single-ID and batched paths parse identically;
    the two differ only in how they find the entry.
    """
    ns = ATOM_NS
    title_elem = entry.find(".//atom:title", ns)
    title = title_elem.text.strip() if title_elem is not None else "untitled"

    authors = []
    for author_elem in entry.findall(".//atom:author", ns):
        name_elem = author_elem.find("atom:name", ns)
        if name_elem is not None:
            authors.append(name_elem.text.strip())

    published_elem = entry.find(".//atom:published", ns)
    updated_elem = entry.find(".//atom:updated", ns)

    original_year = "unknown"
    version_year = None
    if published_elem is not None and published_elem.text:
        original_year = published_elem.text[:4]
    if updated_elem is not None and updated_elem.text:
        version_year = updated_elem.text[:4]

    summary_elem = entry.find(".//atom:summary", ns)
    abstract = summary_elem.text.strip() if summary_elem is not None else "No abstract available"

    pdf_url = None
    for link_elem in entry.findall(".//atom:link", ns):
        if link_elem.get("title") == "pdf" and link_elem.get("rel") == "related":
            pdf_url = link_elem.get("href")
            break

    # Extract version from entry ID URL, e.g. http://arxiv.org/abs/hep-ex/0307015v1
    id_elem = entry.find(".//atom:id", ns)
    version = "v1"
    if id_elem is not None and "http://arxiv.org/abs/" in id_elem.text:
        abs_url = id_elem.text
        if "v" in abs_url:
            version = f"v{abs_url.split('v')[-1].split('/')[0]}"

    author_str, resolved_id, filename = format_filename(
        arxiv_id, authors, original_year, version_year, title, version, title_length_limit
    )

    # Human-readable one-line reference, e.g.
    # "Zhang and Hu et al. (2026) - Is One Layer Enough? ...". Uses the real
    # title (not the filename-safe one), so punctuation stays intact.
    citation = f"{author_str} {format_years(original_year, version_year)} - {title}"

    # arXiv's own extensions, in a second namespace. Only needed for BibTeX output, and all optional —
    # a preprint that was never published has no DOI and no journal reference.
    primary_category_elem = entry.find(".//arxiv:primary_category", ARXIV_NS)
    doi_elem = entry.find(".//arxiv:doi", ARXIV_NS)
    journal_ref_elem = entry.find(".//arxiv:journal_ref", ARXIV_NS)

    return {
        "original_id": arxiv_id,
        "resolved_id": resolved_id,
        "version": version,
        "authors": author_str,
        "author_names": authors,
        "original_year": original_year,
        "version_year": version_year,
        "title": title,
        "citation": citation,
        "abstract": abstract,
        "pdf_url": pdf_url,
        "filename": filename,
        "primary_category": (primary_category_elem.get("term")
                             if primary_category_elem is not None else None),
        "doi": doi_elem.text.strip() if doi_elem is not None and doi_elem.text else None,
        "journal_ref": (journal_ref_elem.text.strip()
                        if journal_ref_elem is not None and journal_ref_elem.text else None),
    }


class _FeedLikeEntry(dict):
    """A `dict` that also answers attribute access, matching what `bibtex.entries_to_bibtex` consumes.

    That function is written against `feedparser` entries, which support both `entry.published` and
    `entry.get("authors")`. This module parses the same Atom with `ElementTree` instead, so rather than
    grow a second BibTeX writer it hands `bibtex` something shaped the way it already expects.
    """
    __getattr__ = dict.__getitem__


def metadata_to_feed_entry(metadata: Dict[str, str]) -> _FeedLikeEntry:
    """Reshape one metadata dict into the entry shape `bibtex.entries_to_bibtex` reads.

    Keeps the version in the `id`, since a caller that downloaded a specific version wants a
    bibliography naming it — `entries_to_bibtex(..., keep_versions=True)` is the matching argument.
    """
    return _FeedLikeEntry(
        id=f"http://arxiv.org/abs/{metadata['resolved_id']}",
        published=metadata["original_year"],  # only [:4] is ever read, so the bare year suffices
        authors=[{"name": name} for name in metadata["author_names"]],
        title=metadata["title"],
        summary=metadata["abstract"],
        arxiv_primary_category=({"term": metadata["primary_category"]}
                                if metadata.get("primary_category") else {}),
        arxiv_doi=metadata.get("doi"),
        arxiv_journal_ref=metadata.get("journal_ref"),
    )


def get_paper_metadata(arxiv_id: str,
                       title_length_limit: int = 128) -> Dict[str, str]:
    """Fetch and parse metadata from arXiv API, including PDF link."""
    api_url = f"{ARXIV_API_URL}?id_list={arxiv_id}"
    response = httpfetch.arxiv_get(api_url)
    response.raise_for_status()
    return parse_metadata_response(response.content, arxiv_id, title_length_limit)


def parse_metadata_responses(xml_content: bytes,
                             arxiv_ids: List[str],
                             title_length_limit: int = 128) -> Dict[str, Dict[str, str]]:
    """Parse a multi-entry arXiv Atom response into ``{requested id: metadata}``.

    Pure function — no network access. `arxiv_ids` are the identifiers that were asked for, and the
    result is keyed by those strings rather than by what arXiv answered with, so the caller can look up
    what it requested without re-deriving anything.

    Entries are matched to requests by identifier, never by position. arXiv returns entries in `id_list`
    order in practice, but relying on that would fail silently and paper-by-paper the first time it did
    not — every filename after a misalignment would be built from the wrong paper's metadata, which is a
    far worse outcome than a missing entry.

    **A request that names a version is matched on that exact version**, and only a request without one
    falls back to the base identifier, taking the highest version returned. This mirrors what the two
    forms mean to arXiv — `2301.12345v2` is that revision, `2301.12345` is whatever is current — and it
    is what makes a batch holding two versions of one paper safe. Matching on the base alone would map
    both requests onto whichever entry came back first, so one of them would quietly receive the other's
    `pdf_url` and filename: the wrong PDF, saved under a name asserting it is the right one.

    Requested identifiers with no matching entry are simply absent from the result. That is the
    batching analogue of `ArxivMetadataError` for a single ID, and it is left to the caller because one
    unusable identifier must not cost the other 99 in its batch.
    """
    root = ET.fromstring(xml_content)
    by_exact: Dict[str, ET.Element] = {}
    by_base_latest: Dict[str, tuple[int, ET.Element]] = {}
    for entry in root.findall(".//atom:entry", ATOM_NS):
        id_elem = entry.find(".//atom:id", ATOM_NS)
        if id_elem is None or not id_elem.text:
            continue
        returned_id = id_elem.text.rsplit("/abs/", 1)[-1]
        base, version = identifiers.split_version(returned_id)
        by_exact.setdefault(returned_id, entry)
        if version >= by_base_latest.get(base, (0, None))[0]:
            by_base_latest[base] = (version, entry)

    found: Dict[str, Dict[str, str]] = {}
    for arxiv_id in arxiv_ids:
        base, _version = identifiers.split_version(arxiv_id)
        if arxiv_id != base:  # the request named a version; nothing else will do
            entry = by_exact.get(arxiv_id)
        else:
            entry = by_base_latest.get(base, (0, None))[1]
        if entry is not None:
            found[arxiv_id] = _metadata_from_entry(entry, arxiv_id, title_length_limit)
    return found


def get_papers_metadata(arxiv_ids: List[str],
                        batch_size: int = METADATA_BATCH_SIZE,
                        title_length_limit: int = 128,
                        rate_limiter: RateLimiter | None = None) -> Dict[str, Dict[str, str]]:
    """Fetch metadata for many papers at once, keyed by requested identifier.

    One request per `batch_size` identifiers rather than one per paper, which is where the wall-clock
    saving lives: the politeness delay is charged per request, and a personal collection runs to
    hundreds of papers.

    `batch_size`: identifiers per request.
    `rate_limiter`: share the caller's limiter so the metadata and PDF requests are paced against one
                    budget. A fresh one is used if not given.

    Identifiers arXiv returns nothing for are absent from the result rather than raising, so the caller
    reports the gap and proceeds with what came back.
    """
    if rate_limiter is None:
        rate_limiter = RateLimiter()

    found: Dict[str, Dict[str, str]] = {}
    for start in range(0, len(arxiv_ids), batch_size):
        batch = arxiv_ids[start:start + batch_size]
        print(f"{colorizer.colorize(GLOBE, colorizer.Style.BRIGHT, colorizer.Fore.BLUE)} "
              f"fetching metadata for {len(batch)} papers "
              f"({start + 1}-{start + len(batch)} of {len(arxiv_ids)})")
        rate_limiter.wait()
        # A failed request costs its batch, not the run. Batching trades granularity for wall time, and
        # without this that trade would extend to failures too: one blip mid-run would abort a job that
        # had already downloaded hundreds of papers. The batch's identifiers are simply absent from the
        # result, which the caller already reports paper by paper.
        try:
            response = httpfetch.arxiv_get(ARXIV_API_URL,
                                           params={"id_list": ",".join(batch), "max_results": len(batch)})
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 -- one bad batch must not abort the run
            print(f"{colorizer.colorize(CROSS, colorizer.Style.BRIGHT, colorizer.Fore.RED)} "
                  f"metadata request for {len(batch)} papers failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue
        found.update(parse_metadata_responses(response.content, batch, title_length_limit))
    return found


def _write_bibtex(metadata_by_id: Dict[str, Dict[str, str]],
                  arxiv_ids: List[str],
                  path: Union[str, pathlib.Path]) -> None:
    """Write the fetched metadata as BibTeX, in the order the identifiers were requested."""
    entries = [metadata_to_feed_entry(metadata_by_id[i]) for i in arxiv_ids if i in metadata_by_id]
    if not entries:
        print(f"{colorizer.colorize(CROSS, colorizer.Style.BRIGHT, colorizer.Fore.RED)} "
              f"no metadata to write to '{path}'")
        return
    path = pathlib.Path(path).expanduser()
    text = bibtex.entries_to_bibtex(entries, keep_versions=True)
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:  # an unwritable path must not cost the downloads that follow
        print(f"{colorizer.colorize(CROSS, colorizer.Style.BRIGHT, colorizer.Fore.RED)} "
              f"could not write '{path}': {exc}")
        return
    # Counted from the text rather than from `entries`, because the writer folds identifiers naming the
    # same record into one - the same identifier typed twice, most simply. Two *versions* of a paper are
    # not that case and stay separate, the version being part of the key here: asking for v3 and v5 is
    # asking for both. Reporting the requested count would overstate what is in the file, which is the
    # wrong direction for a message whose whole job is to say what was written.
    written = len(re.findall(r"^@", text, re.MULTILINE))
    print(f"{colorizer.colorize(CHECKMARK, colorizer.Style.BRIGHT, colorizer.Fore.GREEN)} "
          f"wrote {written} BibTeX entries to '{path}'")


def download_papers(arxiv_ids: List[str],
                    output_dir: str = "papers",
                    batch_size: int = METADATA_BATCH_SIZE,
                    save_bib: Union[str, pathlib.Path, None] = None) -> None:
    """Download papers from arXiv, naming files from their metadata.

    Skips papers already present in *output_dir* (matched by arXiv ID in filename).

    `batch_size`: identifiers per metadata request, passed to `get_papers_metadata`. Lower it to make
                  a failure lose fewer papers; the cost is one rate-limit wait per extra request.

    `save_bib`: if given, also write the fetched metadata to this path as BibTeX.

                This costs no extra requests. Downloading a paper already requires its metadata, to
                build the filename, so the bibliography is made from what is in hand — which is the
                whole reason to do it here rather than by running `raven-arxiv2bib` over the same
                identifiers afterwards.

                Written even for papers that were skipped as already present, since the bibliography
                describes the *set that was asked for*, not the subset that happened to be missing.
    """
    output_dir = str(pathlib.Path(output_dir).expanduser().resolve())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Scan for existing arXiv PDFs to omit them from processing.
    #
    # This is used to skip downloading duplicates when there are manually named
    # files (not downloaded by this tool) that contain a matching arXiv ID in
    # the filename.
    #
    # Filenames where the ID has no paper version part (e.g. "v2") are assumed
    # to refer to "v1". In 2025, arXiv started always adding the version to the
    # suggested filename when manually downloading a paper, even when "v1" is
    # the only existing version. However, old downloads might not have a version
    # in the filename. To be safe, we assume that any such old downloads refer
    # to the first version of the paper.
    arxiv_pdf_files_in_output_dir = identifiers.extract_ids_from_filenames(
        identifiers.list_pdf_files(output_dir), canonize=True,
    )
    output_dir_existing_arxiv_ids = [aid for aid, unused_filename in arxiv_pdf_files_in_output_dir]

    rate_limiter = RateLimiter()

    # Drop exact repeats before fetching anything, which is what `raven-arxiv2bib` does with its input and
    # what this did not: a repeated identifier was carried all the way to the download step before being
    # recognized, costing a slot in the metadata batch on the way. Two *spellings* of one paper are a
    # different matter — `2301.12345` and `2301.12345v1` are the same PDF and cannot be known to be until
    # the metadata says so — so the loop below still deduplicates on the resolved id.
    unique_ids = list(dict.fromkeys(arxiv_ids))

    # Metadata for the whole run first, batched — see `get_papers_metadata`. The PDFs below still cost
    # one rate-limited request each, so this roughly halves the run rather than making it instant.
    metadata_by_id = get_papers_metadata(unique_ids, batch_size=batch_size, rate_limiter=rate_limiter)

    if save_bib is not None:
        _write_bibtex(metadata_by_id, unique_ids, save_bib)

    # Counted by outcome rather than by a single total, because "processed 170" answers nothing on a rerun:
    # the whole point of the skip-existing behaviour is that most of a repeat run does nothing, and the
    # number worth seeing is how much of it was actually fetched.
    tally: dict[str, int] = collections.Counter()
    tally["duplicate identifier"] += len(arxiv_ids) - len(unique_ids)

    seen: set[str] = set()
    for arxiv_id in unique_ids:
        try:
            metadata = metadata_by_id.get(arxiv_id)
            if metadata is None:
                raise ArxivMetadataError(f"no arXiv entry for ID '{arxiv_id}' (nonexistent or malformed ID?)")
            resolved_id = metadata["resolved_id"]
            resolved_id_str = f" (\u2192 {resolved_id})" if resolved_id != arxiv_id else ""
            if resolved_id not in seen:
                seen.add(resolved_id)
                if resolved_id not in output_dir_existing_arxiv_ids:
                    save_path = os.path.join(output_dir, metadata["filename"])
                    if not os.path.exists(save_path):
                        pdf_url = metadata["pdf_url"]
                        if pdf_url is not None:
                            # Show which paper this resolved to before the
                            # rate-limit wait — the one branch that actually
                            # waits, and the one where a wrong-ID typo would
                            # otherwise cost a full download before you notice.
                            print(f"  {metadata['citation']}")
                            print(f"{colorizer.colorize(GLOBE, colorizer.Style.BRIGHT, colorizer.Fore.BLUE)} {arxiv_id}{resolved_id_str}: downloading PDF")
                            rate_limiter.wait()
                            pdf_response = httpfetch.arxiv_get(pdf_url)
                            pdf_response.raise_for_status()
                            with open(save_path, "wb") as f:
                                f.write(pdf_response.content)
                            print(f"{colorizer.colorize(CHECKMARK, colorizer.Style.BRIGHT, colorizer.Fore.GREEN)} {arxiv_id}{resolved_id_str} PDF saved as '{save_path}'")
                            tally["downloaded"] += 1
                        else:
                            print(f"{colorizer.colorize(CROSS, colorizer.Style.BRIGHT, colorizer.Fore.RED)} {arxiv_id}{resolved_id_str} no PDF found")
                            tally["no PDF available"] += 1
                    else:
                        print(f"{colorizer.colorize('-', colorizer.Style.BRIGHT, colorizer.Fore.YELLOW)} {arxiv_id}{resolved_id_str} already downloaded (by this tool) as '{save_path}'")
                        tally["already present"] += 1
                else:
                    idx = output_dir_existing_arxiv_ids.index(resolved_id)
                    save_path = arxiv_pdf_files_in_output_dir[idx][1]
                    print(f"{colorizer.colorize('-', colorizer.Style.BRIGHT, colorizer.Fore.YELLOW)} {arxiv_id}{resolved_id_str} already exists as '{save_path}'")
                    tally["already present"] += 1
            else:
                print(f"{colorizer.colorize('-', colorizer.Style.BRIGHT, colorizer.Fore.YELLOW)} {arxiv_id}{resolved_id_str} already processed (during this session), skipping")
                tally["duplicate identifier"] += 1
        except ArxivMetadataError as e:
            # Expected user error (bad ID) — a one-line message is enough,
            # no traceback.
            print(f"{colorizer.colorize(CROSS, colorizer.Style.BRIGHT, colorizer.Fore.RED)} {arxiv_id} failed: {e}")
            tally["failed"] += 1
        except Exception as e:
            # Unexpected (network blip, parse bug, …) — keep the traceback
            # for debugging.
            print(f"{colorizer.colorize(CROSS, colorizer.Style.BRIGHT, colorizer.Fore.RED)} {arxiv_id} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            tally["failed"] += 1

    # Ordered so the line reads as an account of the run rather than as a dictionary dump, and listing only
    # what happened: a clean run should not have to say "0 failed" for the reader to notice that it did not.
    summary = ", ".join(f"{tally[outcome]} {outcome}"
                        for outcome in ("downloaded", "already present", "duplicate identifier",
                                        "no PDF available", "failed")
                        if tally[outcome])
    print(f"\n{colorizer.colorize(CHECKMARK, colorizer.Style.BRIGHT, colorizer.Fore.GREEN)} "
          f"{len(arxiv_ids)} identifier{'s' if len(arxiv_ids) != 1 else ''} processed"
          f"{': ' + summary if summary else ''}.")


def extract_ids_from_bib(bib_path: str) -> list[str]:
    """Extract arXiv IDs from the ``eprint`` fields of a BibTeX file.

    Returns a list of arXiv ID strings. Entries without an ``eprint``
    field (or with ``archiveprefix`` other than ``arXiv``) are skipped.
    """
    library = bibtexparser.parse_file(bib_path)
    if library.failed_blocks:
        print(f"Warning: {len(library.failed_blocks)} entries failed to parse in {bib_path}",
              file=sys.stderr)

    raw_ids: list[str] = []
    for entry in library.entries:
        fields = entry.fields_dict
        eprint = fields.get("eprint")
        if eprint is None:
            continue
        # Only accept arXiv eprints (skip e.g. SSRN or other archives)
        prefix = fields.get("archiveprefix")
        if prefix is not None and prefix.value.lower() != "arxiv":
            continue
        raw_ids.append(eprint.value)
    return deduplicate_arxiv_ids(raw_ids)


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Download arXiv papers by their IDs, and name the files "
        "automatically using the metadata. If an ID specifies a version, "
        "that version of the paper is downloaded; otherwise the latest "
        "version is downloaded.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(dest="arxiv_ids", nargs="*", default=None,
                        type=str, metavar="id",
                        help="arXiv IDs of papers to download (ID format e.g. "
                        "2511.22570, 2411.17075v5, cond-mat/0207270, "
                        "math/0501001v2)")
    parser.add_argument("-b", "--from-bib", dest="bib_file",
                        type=str, metavar="file.bib", default=None,
                        help="Read arXiv IDs from the eprint fields of a "
                        "BibTeX file (e.g. output of raven-arxiv-search). "
                        "Can be combined with positional IDs.")
    parser.add_argument("-o", "--output-dir", dest="output_dir", default=".",
                        type=str, metavar="output_dir",
                        help="Output directory where to write the PDF file(s). "
                        "Can be a relative or absolute path. Default: current "
                        "working directory.")
    parser.add_argument("-s", "--save-bib", dest="save_bib", default=None,
                        type=str, metavar="file.bib",
                        help="Also write the papers' metadata to this file as BibTeX. Free: the "
                        "metadata is already fetched in order to name the PDFs, so this costs no "
                        "extra requests and no extra waiting, unlike running raven-arxiv2bib over "
                        "the same identifiers afterwards.")
    parser.add_argument('-v', '--version', action='version',
                        version=('%(prog)s ' + __version__))
    opts = parser.parse_args()

    arxiv_ids = list(opts.arxiv_ids or [])
    if opts.bib_file is not None:
        bib_ids = extract_ids_from_bib(opts.bib_file)
        print(f"Read {len(bib_ids)} arXiv IDs from {opts.bib_file}", file=sys.stderr)
        arxiv_ids.extend(bib_ids)
    if not arxiv_ids:
        parser.error("no arXiv IDs specified (provide IDs on the command line and/or via --from-bib)")

    download_papers(arxiv_ids=arxiv_ids,
                    output_dir=opts.output_dir,
                    save_bib=opts.save_bib)


if __name__ == "__main__":
    main()
