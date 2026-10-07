"""Rebuild the whole pipeline's outputs into one directory, numbered by stage, without asking a model anything.

    python investigations/aokk-corpus-scope/regenerate_reports.py --out-dir OUT --format xlsx

The stages and what each leaves, named so that listing the directory shows the pipeline in order:

    01-deduped.bib        01-dedup-merged.<fmt>                  raven-deduplicate, the judge's answers replayed
    02-sifted.bib         02-sift-removed.<fmt>                  raven-siftbib --require abstract
    03-in-scope.bib       03-judge-dropped.<fmt>                 judge_scope's verdicts, replayed
    04-filtered.bib       04-filter-removed.<fmt>,
                          04-filter-held-for-review.<fmt>        filter_keeps over the saved extraction

Every model answer comes from a state file a previous run left: `dedup_judge.jsonl` beside the raw export,
`judged.jsonl` and the newest `extracted-*.jsonl` beside this script. Those are keyed by citekey, so a
record the earlier runs never saw has no answer here; stage 3 keeps such a record, and this script says
how many there were rather than leaving the reader to wonder. The dedup judge's state is copied into the
output directory before use, so that a re-run which did have to ask something writes there rather than
into the research data.

`raven-deduplicate --judge` still checks that a backend answers before it reads its cached answers, so one
must be reachable even though nothing is asked of it.
"""

import argparse
import pathlib
import shutil
import subprocess
import sys

from raven.common import tabular

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
RAW_DIR = REPO / "00_stuff" / "rawdata" / "AOKK" / "multisource"


def run(*command: str) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True, help="where the numbered outputs go; created if missing")
    parser.add_argument("--format", default="xlsx", choices=tabular.FORMATS,
                        help="format of every table written")
    parser.add_argument("--raw", default=str(RAW_DIR / "tekoalyagentti_tutkimus.bib"),
                        help="the raw concatenated search export")
    parser.add_argument("--dedup-judge-state", default=str(RAW_DIR / "dedup_judge.jsonl"),
                        help="the dedup judge's saved answers")
    parser.add_argument("--judged", default=str(HERE / "judged.jsonl"), help="the scope judge's saved answers")
    opts = parser.parse_args()

    out = pathlib.Path(opts.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    fmt = opts.format

    # 1. Deduplicate, replaying the judge.
    dedup_state = out / "01-dedup-judge.jsonl"
    shutil.copyfile(opts.dedup_judge_state, dedup_state)
    run("raven-deduplicate", opts.raw, "-o", str(out / "01-deduped.bib"), "-a", str(out / f"01-dedup-merged.{fmt}"),
        "--judge", "--judge-state", str(dedup_state))

    # 2. Sift out the records with no abstract. raven-siftbib names its outputs after its input, so they are
    #    renamed into the sequence afterwards.
    run("raven-siftbib", str(out / "01-deduped.bib"), "--require", "abstract", "--out-dir", str(out),
        "--audit-format", fmt)
    (out / "01-deduped_sifted.bib").rename(out / "02-sifted.bib")
    (out / f"01-deduped_removed.{fmt}").rename(out / f"02-sift-removed.{fmt}")

    # 3. The scope judge's verdicts, replayed: its outputs, without its model passes.
    sys.path.insert(0, str(HERE))
    import judge_scope
    records = judge_scope.load_records(out / "02-sifted.bib")
    done = judge_scope.load_state(pathlib.Path(opts.judged))
    unanswered = [record.key for record in records if record.key not in done]
    print(f"\n$ judge_scope (replayed from {opts.judged})", flush=True)
    judge_scope.write_outputs(records, done, out / "02-sifted.bib",
                              out / "03-in-scope.bib", out / f"03-judge-dropped.{fmt}")
    if unanswered:
        print(f"  NOTE: {len(unanswered)} records have no saved verdict, so were kept unjudged; "
              f"e.g. {', '.join(unanswered[:5])}")

    # 4. The field filter over the saved extraction. It names its outputs itself; renamed likewise.
    run(sys.executable, str(HERE / "filter_keeps.py"), "--bib", str(out / "03-in-scope.bib"),
        "--out-dir", str(out), "--format", fmt)
    (out / "03-in-scope_filtered.bib").rename(out / "04-filtered.bib")
    (out / f"filtered-out.{fmt}").rename(out / f"04-filter-removed.{fmt}")
    (out / f"held-for-review.{fmt}").rename(out / f"04-filter-held-for-review.{fmt}")

    print(f"\nwrote, in {out}:")
    for path in sorted(out.iterdir()):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
