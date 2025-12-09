"""Download papers from arXiv by their IDs.

Thanks to Qwen3-30B-A3B-Thinking-2507 and the documentation:
   https://info.arxiv.org/help/api/user-manual.html
   https://info.arxiv.org/help/arxiv_identifier.html
"""

import argparse
import math
import os
import pathlib
import requests
import time
from tqdm import tqdm
import traceback
from typing import Dict, List
import xml.etree.ElementTree as ET

from mcpyrate import colorizer

from .. import __version__

from . import arxiv2id

# Here "." is important for `raven-arxiv2id` to handle the downloaded files correctly.
filename_safe_nonalphanum = " -_',."
GLOBE = "🌐"  # for progress messages indicating internet access

# arXiv API TOS requires waiting a minimum of 3 seconds between requests
#     https://info.arxiv.org/help/api/tou.html
class RateLimiter:
    def __init__(self, delay: float = 3.0):
        """Rate-limit an action.

        For single-threaded use only.

        `delay`: minimum required delay between actions, seconds
        """
        self.delay = delay
        self.timestamp = 0  # can use any value that causes the first `wait()` to return immediately

    def wait(self, show_progress: bool = True) -> None:
        """Wait until `self.delay` seconds of wall time has elapsed since the last action.

        `show_progress`: If `True`, show a `tqdm` progress bar while waiting.
                         If `False`, wait silently.

        Call this just before performing the actual action to honor the rate limit for that action.

        When the `RateLimiter` starts up, the first `wait` returns immediately,
        so that no special handling is needed for the first action at the calling end.

        Further calls to `wait` the time from when the previous call to `wait` finished.
        """
        t = time.time_ns()
        delay_ns = self.delay * 10**9
        wait_duration_ns = delay_ns - (t - self.timestamp)
        if wait_duration_ns > 0:
            if show_progress:
                with tqdm(desc="Waiting for API rate limit", leave=False) as pbar:
                    pbar.total = math.ceil((wait_duration_ns / 10**9) * 10)  # segments of 0.1 seconds (last one may be shorter)
                    pbar.n = 0
                    while (time.time_ns() - t) < wait_duration_ns:
                        time.sleep(min(0.1, wait_duration_ns / 10**9))
                        pbar.update()
            else:
                time.sleep(wait_duration_ns / 10**9)
        self.timestamp = time.time_ns()

def clean_arxiv_id(arxiv_id: str) -> str:
    """Remove version suffix (e.g., 'v1') from arXiv ID."""
    if 'v' in arxiv_id:
        return arxiv_id.split('v')[0]
    return arxiv_id

def get_paper_metadata(arxiv_id: str,
                       title_length_limit: int = 128) -> Dict[str, str]:
    """Fetch and parse metadata from arXiv API, including PDF link."""
    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

    response = requests.get(api_url)
    response.raise_for_status()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(response.content)

    # Descend into entry element
    entry = root.find(".//atom:entry", ns)

    # Extract title
    title_elem = entry.find(".//atom:title", ns)
    title = title_elem.text.strip() if title_elem is not None else "untitled"

    # Extract authors
    authors = []
    for author_elem in entry.findall(".//atom:author", ns):
        name_elem = author_elem.find("atom:name", ns)
        if name_elem is not None:
            authors.append(name_elem.text.strip())

    # Extract both years (original posting, current revision)
    published_elem = entry.find(".//atom:published", ns)
    updated_elem = entry.find(".//atom:updated", ns)

    original_year = "unknown"
    version_year = None

    if published_elem is not None and published_elem.text:
        original_year = published_elem.text[:4]

    if updated_elem is not None and updated_elem.text:
        version_year = updated_elem.text[:4]

    # Extract abstract
    summary_elem = entry.find(".//atom:summary", ns)
    abstract = summary_elem.text.strip() if summary_elem is not None else "No abstract available"

    # Extract PDF link
    pdf_url = None
    for link_elem in entry.findall(".//atom:link", ns):
        if link_elem.get("title") == "pdf" and link_elem.get("rel") == "related":
            pdf_url = link_elem.get("href")
            break

    # Extract version from ID URL
    # Example URL: http://arxiv.org/abs/hep-ex/0307015v1
    id_elem = entry.find(".//atom:id", ns)
    version = "v1"  # default version
    if id_elem is not None and "http://arxiv.org/abs/" in id_elem.text:
        abs_url = id_elem.text
        if "v" in abs_url:
            version = f"v{abs_url.split('v')[-1].split('/')[0]}"

    # Format filename
    author_str = " and ".join(authors[:2])
    if len(authors) > 2:
        author_str += " et al."
    elif not authors:
        author_str = "Unknown"

    version_year_str = ""
    if version_year is not None and version_year != original_year:
        version_year_str = f", revised {version_year}"

    title = title.replace(": ", " - ")
    safe_title = "".join(c for c in title if c.isalnum() or c in filename_safe_nonalphanum)
    safe_title = safe_title[:title_length_limit] + ("..." if len(title) > title_length_limit else "")

    # Canonize ID to always include the version
    clean_id = clean_arxiv_id(arxiv_id)
    resolved_id = f"{clean_id}{version}"

    safe_resolved_id = resolved_id.replace("/", "_")
    safe_resolved_id = "".join(c for c in safe_resolved_id if c.isalnum() or c in filename_safe_nonalphanum)

    filename = f"{author_str} ({original_year}{version_year_str}) - {safe_title} - {safe_resolved_id}.pdf"

    return {
        "original_id": arxiv_id,
        "resolved_id": resolved_id,
        "version": version,
        "authors": author_str,
        "original_year": original_year,
        "version_year": version_year,
        "title": title,
        "abstract": abstract,
        "pdf_url": pdf_url,
        "filename": filename,
    }

def download_papers(arxiv_ids: List[str],
                    output_dir: str = "papers") -> None:
    """Given a list of arXiv IDs, download papers from arXiv and name the files automatically according to the paper metadata."""
    output_dir = str(pathlib.Path(output_dir).expanduser().resolve())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Scan for existing arXiv PDFs to omit them from processing.
    #
    # This is used to skip downloading duplicates when there are manually named files (not downloaded by this tool) that contain a matching arXiv ID in the filename.
    #
    # Filenames where the ID has no paper version part (e.g. "v2") are assumed to refer to "v1". In 2025, arXiv started always adding the version to the suggested filename
    # when manually downloading a paper, even when "v1" is the only existing version. However, old downloads might not have a version in the filename. To be safe, we assume
    # that any such old downloads refer to the first version of the paper.
    #
    arxiv_pdf_files_in_output_dir = arxiv2id.get_arxiv_identifiers_from_filenames(arxiv2id.list_pdf_files(output_dir),
                                                                                  canonize=True)
    output_dir_existing_arxiv_ids = [arxiv_id for arxiv_id, unused_filename in arxiv_pdf_files_in_output_dir]

    rate_limiter = RateLimiter()
    seen = set()
    for arxiv_id in arxiv_ids:
        try:
            # TODO: We could reduce total wait time by batching the metadata fetch into sets of up to 100 IDs each, needing just one metadata request per set (instead of per paper as now). For how, see the external `arxiv2bib` tool.
            print(f"{colorizer.colorize(GLOBE, colorizer.Style.BRIGHT, colorizer.Fore.BLUE)} {arxiv_id}: fetching metadata")
            rate_limiter.wait()
            metadata = get_paper_metadata(arxiv_id)
            resolved_id = metadata["resolved_id"]
            resolved_id_str = f" (→ {resolved_id})" if resolved_id != arxiv_id else ""
            if resolved_id not in seen:
                seen.add(resolved_id)
                if resolved_id not in output_dir_existing_arxiv_ids:
                    save_path = os.path.join(output_dir, metadata["filename"])
                    if not os.path.exists(save_path):
                        pdf_url = metadata["pdf_url"]
                        if pdf_url is not None:
                            print(f"{colorizer.colorize(GLOBE, colorizer.Style.BRIGHT, colorizer.Fore.BLUE)} {arxiv_id}{resolved_id_str}: downloading PDF")
                            rate_limiter.wait()
                            pdf_response = requests.get(pdf_url)
                            pdf_response.raise_for_status()
                            with open(save_path, "wb") as f:
                                f.write(pdf_response.content)
                            print(f"{colorizer.colorize('✓', colorizer.Style.BRIGHT, colorizer.Fore.GREEN)} {arxiv_id}{resolved_id_str} PDF saved as '{save_path}'")
                        else:  # could not get URL
                            print(f"{colorizer.colorize('✗', colorizer.Style.BRIGHT, colorizer.Fore.RED)} {arxiv_id}{resolved_id_str} no PDF found")
                    else:  # existing file downloaded by this tool
                        print(f"{colorizer.colorize('-', colorizer.Style.BRIGHT, colorizer.Fore.YELLOW)} {arxiv_id}{resolved_id_str} already downloaded (by this tool) as '{save_path}'")
                else:  # existing file from some other source
                    idx = output_dir_existing_arxiv_ids.index(resolved_id)
                    save_path = arxiv_pdf_files_in_output_dir[idx][1]
                    print(f"{colorizer.colorize('-', colorizer.Style.BRIGHT, colorizer.Fore.YELLOW)} {arxiv_id}{resolved_id_str} already exists as '{save_path}'")
            else:  # duplicate IDs from the command line that resolve to the same version of the same paper
                print(f"{colorizer.colorize('-', colorizer.Style.BRIGHT, colorizer.Fore.YELLOW)} {arxiv_id}{resolved_id_str} already processed (during this session), skipping")
        except Exception as e:  # general failure, e.g. error during HTTP request
            print(f"{colorizer.colorize('✗', colorizer.Style.BRIGHT, colorizer.Fore.RED)} {arxiv_id} failed: {str(e)}")
            traceback.print_exc()  # useful if the error was actually triggered by a bug in this program

def main() -> None:
    parser = argparse.ArgumentParser(description="""Download arXiv papers by their IDs, and name the files automatically using the metadata. If an ID specifies a version, that version of the paper is downloaded; otherwise the latest version is downloaded.""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest="arxiv_ids", nargs="+", default=None, type=str, metavar="id", help="arXiv IDs of papers to download (ID format e.g. 2511.22570, 2411.17075v5, cond-mat/0207270, math/0501001v2)")
    parser.add_argument("-o", "--output-dir", dest="output_dir", default=".", type=str, metavar="output_dir", help="Output directory where to write the PDF file(s). Can be a relative or absolute path. Default: current working directory.")
    parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
    opts = parser.parse_args()

    download_papers(arxiv_ids=opts.arxiv_ids,
                    output_dir=opts.output_dir)

if __name__ == "__main__":
    main()
