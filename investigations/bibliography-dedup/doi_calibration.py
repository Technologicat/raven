"""Where to put the threshold that decides a shared DOI is wrong, and why it needs two conditions.

    python doi_calibration.py path/to/search.bib

`raven.papers.deduplicate._doi_edge_holds` refuses to merge two records that share a DOI when their titles
are unalike **and** they name different first authors. Both, and this is the measurement that says so: the
title condition alone cannot be calibrated, and the corpus is what shows it.

Run this against a corpus before moving `config.doi_title_floor`, or when a new corpus makes the rule
refuse something it should not have. Writes nothing.

## The three sections

  1. **How alike DOI-joined titles are.** The distribution, and the least alike pairs named. Every one of
     these is a real merge somebody wants: two databases spelling one title differently, a subtitle kept
     by one and dropped by the other, a paper retitled between its preprint and its publication.
  2. **Whether any of them also disagree about the first author.** The question section 1 forces, once the
     distribution turns out to overlap the counter-example.
  3. **The negative control**, which no corpus of correct records can supply, so it is built here.

## What it found on 2026-08-31

The corpus was 6934 records of a multi-database search on AI in education, and it is not committed — a
search export, in a public repository. The numbers stand as the reason the rule has the shape it has:

  - 27 DOI-joined pairs whose titles were not character-identical. The least alike scored **0.327**, a
    paper retitled between its preprint and its publication, and it is one work.
  - The two genuinely unrelated papers claiming one DOI scored **0.260**.
  - Seven hundredths between them, from one corpus and one counter-example. A title threshold alone has
    to be threaded through that gap, and nobody can honestly place it.
  - **All 27 agreed about the first author.** So requiring an author disagreement as well costs nothing
    that corpus contained, and moves the burden off a number that cannot bear it.

The asymmetry that makes a quiet rule the right one: a missed wrong DOI leaves a bibliography as it already
was, and a wrongly refused merge splits a real paper in two.
"""

import argparse
import collections
import logging
import pathlib
import sys

from raven.papers import config as papers_config
from raven.papers import deduplicate as dd


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def ratio(a: dd.Record, b: dd.Record) -> float:
    """The true similarity of two records' normalized titles.

    Through `deduplicate`'s own helper, with the threshold dropped to zero so that it reports rather than
    rejecting — the default floors everything under `config.title_similarity` to 0.0, which is the whole
    range this script exists to look at.
    """
    return dd._title_similarity(a.title, b.title, 0.0)


def doi_joined_pairs(records: list[dd.Record]) -> list[tuple[float, str, dd.Record, dd.Record]]:
    """Every pair of records sharing a DOI and differing in title, least alike first."""
    groups = collections.defaultdict(list)
    for record in records:
        if record.doi is not None:
            groups[record.doi].append(record)

    seen, pairs = set(), []
    for doi, group in groups.items():
        for position, a in enumerate(group):
            for b in group[position + 1:]:
                if a.title is None or b.title is None or a.title == b.title:
                    continue
                # Two databases' copies of one record are themselves duplicated across the corpus, so the
                # same pair of titles arrives several times under one DOI. Counting it once keeps the
                # distribution a distribution rather than a census of export volume.
                key = (doi, min(a.title, b.title), max(a.title, b.title))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((ratio(a, b), doi, a, b))
    return sorted(pairs, key=lambda pair: pair[0])


# The control, which a corpus of correct records cannot provide: two papers that really do claim one DOI.
# arXiv's metadata for eprint 2405.00291 carries the journal reference and DOI of an unrelated astronomy
# paper, so the education record below ships with them. The astronomy title and first author are what
# `https://doi.org/10.1051/0004-6361/202349120` answers with, read as CSL JSON on 2026-08-31.
_CONTROL = """\
@article{education,
  title = {How Can I Improve? Using GPT to Highlight the Desired and Undesired Parts of Open-ended Responses},
  author = {Lin, Jionghao and Chen, Eason},
  doi = {10.1051/0004-6361/202349120},
}
@article{astronomy,
  title = {Modifications of astrophysical ices induced by cosmic rays},
  author = {Mej\\'ia, C. and de Barros, A. L. F.},
  doi = {10.1051/0004-6361/202349120},
}
"""


def report_control() -> float:
    """Print what the guard does with a pair that genuinely should not merge. Returns their similarity."""
    education, astronomy = dd.read_records(_CONTROL)[0]
    score = ratio(education, astronomy)
    print(f"  two papers claiming 10.1051/0004-6361/202349120, title similarity {score:.3f}")
    print(f"    first authors differ:  {dd._disagree_on_author(education, astronomy)}")
    print(f"    below the floor:       {score < papers_config.doi_title_floor} "
          f"(config.doi_title_floor = {papers_config.doi_title_floor})")
    holds = dd._doi_edge_holds(education, astronomy)
    print(f"    _doi_edge_holds:       {holds}  <- must be False, or the guard does not fire on the one "
          f"case it was written for")
    return score


def main(path: str) -> int:
    logging.getLogger("bibtexparser").setLevel(logging.ERROR)
    records, unreadable = dd.read_records(pathlib.Path(path).read_text(encoding="utf-8"))
    pairs = doi_joined_pairs(records)
    print(f"{pathlib.Path(path).name}: {len(records)} records read, {len(unreadable)} unreadable")

    section("How alike are two titles that a shared DOI joined?")
    print(f"  {len(pairs)} pair(s) share a DOI and spell their title differently\n")
    buckets = collections.Counter(round(score, 1) for score, *_ in pairs)
    for edge in sorted(buckets):
        print(f"    ~{edge:.1f}   {buckets[edge]:5}  {'#' * min(60, buckets[edge])}")
    print("\n  the least alike, which are the ones a threshold would refuse first:")
    for score, doi, a, b in pairs[:8]:
        print(f"    {score:.3f}  {doi}")
        print(f"           {(a.field('title') or '')[:96]}")
        print(f"           {(b.field('title') or '')[:96]}")

    section("Do any of them disagree about the first author?")
    disagreeing = [(score, doi, a, b) for score, doi, a, b in pairs if dd._disagree_on_author(a, b)]
    print(f"  {len(disagreeing)} of {len(pairs)} pair(s) name different first authors")
    for score, doi, a, b in disagreeing:
        print(f"    {score:.3f}  {doi}  {a.surname} vs {b.surname}")
    if not disagreeing:
        print("    -> the author clause refuses nothing this corpus contains, which is what lets it be")
        print("       required alongside the title condition rather than replacing it")

    section("The negative control, and what the guard makes of it")
    control = report_control()
    if pairs:
        print(f"\n  lowest real pair in this corpus: {pairs[0][0]:.3f}")
        print(f"  the control:                     {control:.3f}")
        gap = pairs[0][0] - control
        print(f"  daylight between them:           {gap:.3f}")
        if gap < 0.15:
            print("    -> too narrow to place a title threshold in. This is the finding that made the")
            print("       author disagreement a requirement rather than a tiebreak.")

    section("What the guard actually refuses here")
    refused = dd.refused_doi_edges(records)
    print(f"  {len(refused)} pair(s) refused in this corpus")
    for a, b in refused:
        print(f"    {a.doi}: `{a.key}` and `{b.key}`")
    if not refused:
        print("    -> nothing, which is the false-positive rate this rule needed to have")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest="filename", type=str, metavar="search.bib",
                        help="BibTeX file to calibrate against. Read, never written.")
    sys.exit(main(parser.parse_args().filename))
