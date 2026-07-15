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
    "get_paper_metadata",
    "download_papers",
    "extract_ids_from_bib",
    "main",
]

import argparse
import os
import pathlib
import sys

import traceback
from typing import Dict, List
import xml.etree.ElementTree as ET

import bibtexparser

from mcpyrate import colorizer

from .. import __version__
from ..common import stringmaps
from . import httpfetch
from . import identifiers
from .ratelimit import RateLimiter
from .utils import deduplicate_arxiv_ids

GLOBE = "\U0001f310"  # 🌐 — for progress messages indicating internet access
CHECKMARK = "\u2713"  # ✓
CROSS = "\u2717"      # ✗


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
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_content)
    entry = root.find(".//atom:entry", ns)
    if entry is None:
        # arXiv returns an entry-less feed for a nonexistent or malformed ID
        # (e.g. a typoed month, as in "2614.19062"). Fail with something
        # readable instead of an AttributeError from the next .find().
        raise ArxivMetadataError(f"no arXiv entry for ID '{arxiv_id}' (nonexistent or malformed ID?)")

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

    return {
        "original_id": arxiv_id,
        "resolved_id": resolved_id,
        "version": version,
        "authors": author_str,
        "original_year": original_year,
        "version_year": version_year,
        "title": title,
        "citation": citation,
        "abstract": abstract,
        "pdf_url": pdf_url,
        "filename": filename,
    }


def get_paper_metadata(arxiv_id: str,
                       title_length_limit: int = 128) -> Dict[str, str]:
    """Fetch and parse metadata from arXiv API, including PDF link."""
    api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = httpfetch.arxiv_get(api_url)
    response.raise_for_status()
    return parse_metadata_response(response.content, arxiv_id, title_length_limit)


def download_papers(arxiv_ids: List[str],
                    output_dir: str = "papers") -> None:
    """Download papers from arXiv, naming files from their metadata.

    Skips papers already present in *output_dir* (matched by arXiv ID in filename).
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
    seen: set[str] = set()
    for arxiv_id in arxiv_ids:
        try:
            # TODO: We could reduce total wait time by batching the metadata
            # fetch into sets of up to 100 IDs each, needing just one metadata
            # request per set (instead of per paper as now). For how, see the
            # external `arxiv2bib` tool.
            print(f"{colorizer.colorize(GLOBE, colorizer.Style.BRIGHT, colorizer.Fore.BLUE)} {arxiv_id}: fetching metadata")
            rate_limiter.wait()
            metadata = get_paper_metadata(arxiv_id)
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
                        else:
                            print(f"{colorizer.colorize(CROSS, colorizer.Style.BRIGHT, colorizer.Fore.RED)} {arxiv_id}{resolved_id_str} no PDF found")
                    else:
                        print(f"{colorizer.colorize('-', colorizer.Style.BRIGHT, colorizer.Fore.YELLOW)} {arxiv_id}{resolved_id_str} already downloaded (by this tool) as '{save_path}'")
                else:
                    idx = output_dir_existing_arxiv_ids.index(resolved_id)
                    save_path = arxiv_pdf_files_in_output_dir[idx][1]
                    print(f"{colorizer.colorize('-', colorizer.Style.BRIGHT, colorizer.Fore.YELLOW)} {arxiv_id}{resolved_id_str} already exists as '{save_path}'")
            else:
                print(f"{colorizer.colorize('-', colorizer.Style.BRIGHT, colorizer.Fore.YELLOW)} {arxiv_id}{resolved_id_str} already processed (during this session), skipping")
        except ArxivMetadataError as e:
            # Expected user error (bad ID) — a one-line message is enough,
            # no traceback.
            print(f"{colorizer.colorize(CROSS, colorizer.Style.BRIGHT, colorizer.Fore.RED)} {arxiv_id} failed: {e}")
        except Exception as e:
            # Unexpected (network blip, parse bug, …) — keep the traceback
            # for debugging.
            print(f"{colorizer.colorize(CROSS, colorizer.Style.BRIGHT, colorizer.Fore.RED)} {arxiv_id} failed: {type(e).__name__}: {e}")
            traceback.print_exc()


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
                    output_dir=opts.output_dir)


if __name__ == "__main__":
    main()
