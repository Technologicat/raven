"""Remove kept records that the extracted fields positively rule out, and list the ones they only suspect.

The judge keeps a record whenever no test can be positively established, so its keeps carry no evidence
and cannot be audited. `extract_fields.py` records what each one *says*; this applies a rule to those
fields, and the rule is the cheap half — the fields are already stored, so a cutoff can be changed and
argued about without another model call.

Four tiers, separated because they are not equally reliable and a single removal count hides that — and
because two of them are not about reliability at all:

  A   `level` is `school` or `vocational` — a level positively named, and quoted. Every such call in the
      corpus carried a quoted phrase from the record's own text, which is the check on it.
  B1  `level` is `not_applicable` and `human_learning` is false — not set in education, corroborated by
      a second field. Machine-learning methods papers, mostly.
  B2  `level` is `not_applicable` with no such corroboration. **Not removed.** These are where the
      extraction's errors concentrate: a study whose level is merely unstated gets called
      `not_applicable`, and the removal would be wrong rather than merely unlucky.
  C   `level` is `professional_training` or `informal` — learning that is really happening, somewhere
      this review may or may not ask about. **Not removed**, and for a different reason than B2: these
      are correctly labelled, and the open question is the review's scope rather than the extraction's
      accuracy. `--remove-outside-institutions` is where that decision goes when it is made.

So A and B1 are removals; B2 and C are lists for a person, and they ask different questions of that
person — *is this label right?* and *is this in scope?* A record this cannot rule out stays, on the same
asymmetry the judge uses: a false keep costs a reader one line, a false drop loses a study with nothing
left behind to notice it by.

Writes an audit of everything it removed, with the fields and the quoted evidence that removed it —
following the convention `raven.papers` states: a tool that removes records owes an audit.
"""

import argparse
import collections
import json
import logging
import pathlib
import sys

import bibtexparser
from bibtexparser.model import Entry

from raven.common import utils as common_utils
from raven.papers import bibtex

TIER_A = "wrong level, quoted"
TIER_B1 = "not education, corroborated"
TIER_B2 = "not education, uncorroborated"
TIER_C = "learning, but not in an institution"


def tier_of(fields: dict) -> str | None:
    """Which tier a record's extracted fields put it in, or `None` for none of them.

    `professional_training` and `informal` exist because one value was carrying two meanings — *not about
    education* and *about education, somewhere this review does not ask about* — and separating them was
    the point of the last re-extraction. So they get a tier rather than falling through: a record that
    lands in one is a real finding, not an absence of one.

    That tier is held rather than removed, because whether a psychotherapy training study or a lifelong
    learning study belongs in a review of higher education is a question about the review's scope. The
    extractor can say where a study is set; it cannot say what the review wants.
    """
    if fields["level"] in ("school", "vocational"):
        return TIER_A
    if fields["level"] in ("professional_training", "informal"):
        return TIER_C
    if fields["level"] == "not_applicable":
        return TIER_B1 if fields["human_learning"] is False else TIER_B2
    return None


def newest_extraction(directory: pathlib.Path) -> pathlib.Path:
    """The most recent `extracted-<instrument>.jsonl`, since the filename now names the instrument.

    Raises `FileNotFoundError` if there is none, rather than falling back to an older naming scheme: a
    filter that silently reads a file from a superseded extractor produces a plausible corpus from the
    wrong measurements, which is worse than not running.
    """
    candidates = [p for p in directory.glob("extracted-*.jsonl") if not p.name.endswith("-traces.jsonl")]
    if not candidates:
        raise FileNotFoundError(f"no extracted-*.jsonl in {directory}; run extract_fields.py first")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_extracted(path: pathlib.Path) -> tuple[dict[str, dict], set[str]]:
    """The extracted fields per citekey, and every instrument fingerprint the file turned out to hold.

    A later line supersedes an earlier one. More than one fingerprint means the file is not what its name
    says — two runs concatenated, or a rename — and the caller has to decide, because the mixture cannot
    be seen in the values: a level name that exists in two vocabularies means different things on either
    side of the change, and nothing errors.
    """
    out, seen = {}, set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            fields = json.loads(line)
            out[fields["key"]] = fields
            seen.add(fields.get("v"))
    return out, seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bib", required=True, help="the .bib to filter (the judge's in-scope output)")
    parser.add_argument("--extracted", default=None,
                        help="the extraction JSONL (default: the newest extracted-*.jsonl beside this "
                             "script, since the filename names the instrument that wrote it)")
    parser.add_argument("--out-dir", default=None, help="where the outputs go (default: beside this)")
    parser.add_argument("--keep-uncorroborated", action="store_true",
                        help="also remove tier B2, which this refuses to do by default. Read "
                             "`held-for-review.tsv` before reaching for it: that tier is where the "
                             "extraction's errors are, and its mistakes are false drops")
    parser.add_argument("--remove-outside-institutions", action="store_true",
                        help="also remove workplace training and informal learning. A scope decision "
                             "rather than a correctness one: those records are correctly labelled, and "
                             "the question is whether a review of higher education wants them")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="report what would go, write nothing")
    opts = parser.parse_args()

    logging.getLogger("bibtexparser").setLevel(logging.ERROR)
    here = pathlib.Path(__file__).resolve().parent
    out_dir = pathlib.Path(opts.out_dir) if opts.out_dir else here
    bib_path = pathlib.Path(opts.bib).expanduser().resolve()

    extracted_path = (pathlib.Path(opts.extracted).expanduser().resolve() if opts.extracted
                      else newest_extraction(here))
    extracted, fingerprints = load_extracted(extracted_path)
    print(f"reading {extracted_path.name}: {len(extracted)} records")
    if len(fingerprints) > 1:
        print(f"  WARNING: {len(fingerprints)} instruments in one file ({', '.join(sorted(map(str, fingerprints)))}).\n"
              f"  Its name says one. The values cannot be told apart by reading them, so this filter is\n"
              f"  mixing measurements — fix the file before believing the result.")
    library = bibtex.parse_file(str(bib_path), split_names=False)

    tiers = {}
    for key, fields in extracted.items():
        tier = tier_of(fields)
        if tier is not None:
            tiers[key] = tier

    removing = {TIER_A, TIER_B1} | ({TIER_B2} if opts.keep_uncorroborated else set())
    removing |= {TIER_C} if opts.remove_outside_institutions else set()
    doomed = {key for key, tier in tiers.items() if tier in removing}
    held = {key for key, tier in tiers.items() if tier not in removing}

    counts = collections.Counter(tiers.values())
    print(f"{bib_path.name}: {len(library.entries)} records, {len(extracted)} with extracted fields")
    for tier in (TIER_A, TIER_B1, TIER_B2, TIER_C):
        verdict = "removing" if tier in removing else "HELD for hand-check"
        print(f"  {tier:<32}{counts[tier]:>5}   {verdict}")
    print(f"\n{len(doomed)} to remove, {len(library.entries) - len(doomed)} to keep")
    if opts.dry_run:
        print("\ndry run: nothing written")
        return 0

    # Filtered at the block level so a preamble, `@string` definitions and comments between records
    # survive into the filtered copy rather than being dropped along with the records.
    kept = bibtexparser.Library()
    for block in library.blocks:
        if isinstance(block, Entry) and block.key in doomed:
            continue
        kept.add(block)
    out_path = out_dir / f"{bib_path.stem}_filtered.bib"
    out_path.write_text(bibtex.write_string(kept), encoding="utf-8")

    # Resolved rather than brace-stripped: these columns exist to be read, and `{\o}nly` collapses to
    # `\only` if the braces go first, the braces being what terminates the command.
    titles = {entry.key: common_utils.normalize_whitespace(
                  common_utils.unicodize_basic_markup(entry.fields_dict["title"].value))
              for entry in library.entries if "title" in entry.fields_dict}

    def write_table(path: pathlib.Path, keys, header: str) -> None:
        with path.open("w", encoding="utf-8") as f:
            f.write(header)
            for key in sorted(keys, key=lambda k: (tiers[k], k)):
                fields = extracted[key]
                title = titles.get(key, "")
                f.write(f"\t{tiers[key]}\t{key}\t{fields['level']}\t{fields['population']}\t"
                        f"{fields['human_learning']}\t{fields['evidence']}\t{fields['ai_role']}\t"
                        f"{title}\n")

    columns = "mark\ttier\tkey\tlevel\tpopulation\thuman_learning\tevidence\tai_role\ttitle\n"
    audit_path = out_dir / "filtered-out.tsv"
    write_table(audit_path, doomed, columns)
    held_path = out_dir / "held-for-review.tsv"
    write_table(held_path, held, columns)

    print(f"\nwrote {out_path}  ({len(kept.entries)} of {len(library.entries)} records kept)")
    print(f"wrote {audit_path}  ({len(doomed)} removed, with the fields that removed them)")
    print(f"wrote {held_path}  ({len(held)} held back, sorted by tier — the ones needing a person)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
