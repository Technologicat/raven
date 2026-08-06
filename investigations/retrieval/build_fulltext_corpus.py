"""Assemble the arXiv AI fulltext corpus, matching the abstract corpus document-for-document.

The abstract corpus is `00_stuff/datasets/ai_papers/burst`, one `.bib` per paper, named for the
*pinned* arXiv identifier — `2506.19823v2.bib`. Every gold label in `arxiv_ai_questions.json` is one
of those filenames, so the fulltext corpus is only a controlled comparison if it holds the same 1268
documents under the same identifiers, differing in nothing but what is indexed. That constraint drives
every decision here.

Two subcommands, run in order:

    python build_fulltext_corpus.py plan       # what is on disk, what has to be fetched
    # ...then raven-arxiv-download the missing ones (plan prints the command)...
    python build_fulltext_corpus.py assemble   # canonical-named corpus, ready for raven-indexer

`plan` matches the pinned set against a directory of already-downloaded PDFs, **on the exact
versioned identifier**. A newer version of the same paper does not count as a match: the identifier is
what the gold labels key on, so substituting v3 for the pinned v2 would silently change the document
ID and turn a correct retrieval into a scored miss. 243 of the 1268 had drifted that way when this was
first measured (2026-08-06) — 19%, far too many to absorb.

`assemble` symlinks rather than copies. The corpus is 6 GB of PDFs that already exist on this disk,
and the extraction step only reads them.

**Filenames are canonicalized to `<pinned-id>.pdf`.** The sources disagree about naming — the existing
stash mostly uses `Authors (Year) - Title - id.pdf`, `raven-arxiv-download` writes that same shape,
and a few files are bare identifiers — while `document_id` in the index is the filename. Rewriting all
of them to the identifier is what makes the gold labels transfer: a label reads `2506.19823v2.bib` and
the retrieved document reads `2506.19823v2.pdf`, so scoring differs from the abstract side by a suffix
substitution and nothing else.
"""

from __future__ import annotations

__all__ = [
    "pinned_identifiers",
    "identifiers_on_disk",
    "match",
    "assemble",
]

import argparse
import collections
import sys
from pathlib import Path

from raven.papers import identifiers as arxiv_identifiers

BURST_DIR = Path("00_stuff/datasets/ai_papers/burst")
DOWNLOADS_DIR = Path.home() / "Downloads" / "papers"
FETCHED_DIR = Path("00_stuff/datasets/ai_papers/fulltext_fetched")
CORPUS_DIR = Path("00_stuff/datasets/ai_papers/fulltext")

MISSING_IDS_PATH = Path("investigations/retrieval/fulltext_missing_ids.txt")


def pinned_identifiers(burst_dir: Path = BURST_DIR) -> list[str]:
    """The versioned identifiers the abstract corpus (and every gold label) is keyed on.

    Read from the `.bib` filenames rather than the file contents: the filename *is* the document ID
    on the abstract side, so anything else would be answering a slightly different question.
    """
    return sorted(path.stem for path in burst_dir.glob("*.bib"))


def identifiers_on_disk(pdf_dir: Path) -> dict[str, Path]:
    """Map versioned arXiv identifier -> PDF path, for every PDF under `pdf_dir` that names one.

    **Searched recursively**, which is load-bearing rather than a convenience: the stash sorts some
    of its papers into topic subdirectories (`materiaali2`, `temp`, `qualitative_analysis`), and a
    top-level-only scan reported five pinned papers as absent from disk entirely when all five were
    simply one level down. That failure is quiet in the worst way — it reads as "arXiv no longer has
    these" rather than as a bug in the search.

    Files whose names contain no identifier are skipped silently; the stash is a general papers
    directory holding course material and non-arXiv PDFs alongside the collection. Files naming
    *two* identifiers are skipped with a warning — `extract_id` asserts on that case, and a title
    that happens to quote an identifier is not worth crashing the build over.

    Identifiers with no version suffix are recorded as v1, per `split_version`, and for this stash
    that is a fact about how the files got here rather than a default: the unversioned names predate
    `raven-arxiv-download` and carry whatever arXiv suggested at save time, where an absent suffix
    means v1. 73 of the 1268 pinned papers match only through this rule, so reading an unversioned
    name as "version unknown" instead would send 6% of the corpus back to arXiv for files already on
    disk.
    """
    found: dict[str, Path] = {}
    for path in sorted(pdf_dir.rglob("*.pdf")):
        try:
            raw = arxiv_identifiers.extract_id(path.name)
        except AssertionError:
            print(f"  skipping (multiple identifiers in name): {path.name}", file=sys.stderr)
            continue
        if raw is None:
            continue
        base, version = arxiv_identifiers.split_version(raw)
        found.setdefault(f"{base}v{version}", path)
    return found


def match(pinned: list[str], on_disk: dict[str, Path]) -> tuple[dict[str, Path], list[str]]:
    """Split `pinned` into (exact matches -> path, missing identifiers)."""
    matched = {identifier: on_disk[identifier] for identifier in pinned if identifier in on_disk}
    missing = [identifier for identifier in pinned if identifier not in on_disk]
    return matched, missing


def _report(pinned: list[str], on_disk: dict[str, Path], matched: dict[str, Path], missing: list[str]) -> None:
    superseded = collections.Counter(arxiv_identifiers.split_version(i)[0] for i in on_disk)
    pinned_bases = {arxiv_identifiers.split_version(i)[0] for i in pinned}
    present_either_version = len(pinned_bases & set(superseded))

    print()
    print(f"  identifiers found on disk          {len(on_disk)}")
    print(f"  identifiers in the pinned set      {len(pinned)}")
    print(f"  papers present, either version     {present_either_version}")
    print(f"  exact matches including version    {len(matched)}")
    print(f"  pinned version not on disk         {len(missing)}")
    print()


def do_plan(args: argparse.Namespace) -> None:
    pinned = pinned_identifiers(args.burst_dir)
    on_disk = identifiers_on_disk(args.pdf_dir)
    matched, missing = match(pinned, on_disk)
    _report(pinned, on_disk, matched, missing)

    if not missing:
        print("Nothing to fetch; run `assemble`.")
        return

    args.missing_out.write_text("\n".join(missing) + "\n", encoding="utf-8")
    print(f"Wrote {len(missing)} identifiers to {args.missing_out}")
    print()
    print("Fetch them with:")
    print(f"  raven-arxiv-download -o {args.fetched_dir} $(cat {args.missing_out})")


def assemble(pinned: list[str], sources: list[Path], corpus_dir: Path) -> tuple[int, list[str]]:
    """Symlink one PDF per pinned identifier into `corpus_dir`, named `<identifier>.pdf`.

    `sources` are searched in order, so a later directory cannot shadow an earlier one — put the
    authoritative stash first. Returns (linked count, still-missing identifiers).

    Symlinks, because the corpus is a *view* over papers that live elsewhere: nothing is copied, and
    the originals stay the only real files. Hard links would also avoid the copy, but a hard link is
    indistinguishable from the original, so anything that later rewrites a file here would silently
    rewrite the collection too. (`hybridir` grew symlink support for this; before that it resolved
    each document path and rejected anything landing outside the indexed directory.)
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    available: dict[str, Path] = {}
    for source in sources:
        if not source.is_dir():
            continue
        for identifier, path in identifiers_on_disk(source).items():
            available.setdefault(identifier, path)

    linked = 0
    missing: list[str] = []
    for identifier in pinned:
        path = available.get(identifier)
        if path is None:
            missing.append(identifier)
            continue
        link = corpus_dir / f"{identifier}.pdf"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(path.resolve())
        linked += 1
    return linked, missing


def do_assemble(args: argparse.Namespace) -> None:
    pinned = pinned_identifiers(args.burst_dir)
    linked, missing = assemble(pinned, [args.pdf_dir, args.fetched_dir], args.corpus_dir)

    print()
    print(f"  linked into {args.corpus_dir}: {linked} / {len(pinned)}")
    if missing:
        print(f"  still missing: {len(missing)} -> {', '.join(missing[:10])}"
              + (" ..." if len(missing) > 10 else ""))
        print()
        print("The corpus is NOT document-for-document comparable with the abstract side.")
        sys.exit(1)
    print()
    print("Complete. Index it with:")
    print(f"  raven-indexer {args.corpus_dir} -d <db-dir>")


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--burst-dir", type=Path, default=BURST_DIR)
    ap.add_argument("--pdf-dir", type=Path, default=DOWNLOADS_DIR)
    ap.add_argument("--fetched-dir", type=Path, default=FETCHED_DIR)
    ap.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    ap.add_argument("--missing-out", type=Path, default=MISSING_IDS_PATH)

    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="match the stash against the pinned set; list what must be fetched")
    sub.add_parser("assemble", help="symlink a canonically named corpus from stash + fetched")

    args = ap.parse_args(argv)
    {"plan": do_plan, "assemble": do_assemble}[args.command](args)


if __name__ == "__main__":
    main()
