"""Ask a model the extractor's questions over the records a decisions table names, and set the answers beside it.

For whether a newer model reads a record's level and population better, on records people have decided, without a
whole run. Goes through the tool's own extract stage and ledger, so the answers count toward a run with this definition
and model.

    python investigations/aokk-corpus-scope/rerun_extract_on.py STUDIES.bib --decisions TABLE --definition DEF.toml \\
        --ledgers DIR --require abstract --fill --out OUT.tsv

Prints, for each decision, how the answers fall.
"""

import argparse
import collections
import functools
import pathlib
import sys

from raven.common import tabular
from raven.librarian import sweep
from raven.papers.scopebib import cli as scope_cli
from raven.papers.scopebib import definition as scope_definition
from raven.papers.scopebib import extract as scope_extract
from raven.papers.scopebib import pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bib")
    parser.add_argument("--decisions", required=True, help="a table with 'key' and 'decision' columns")
    parser.add_argument("--definition", required=True)
    parser.add_argument("--ledgers", required=True)
    parser.add_argument("--require", action="append", default=[])
    parser.add_argument("--fill", action="store_true")
    parser.add_argument("--backend-url", default=None)
    parser.add_argument("--out", required=True)
    opts = parser.parse_args()

    decided = {row["key"]: row for row in tabular.read_table(opts.decisions)}
    ledger_dir = pathlib.Path(opts.ledgers)
    definition = scope_definition.load_definition(opts.definition)
    nlp, _ = scope_cli._nlp(on_server=False, server_url=None)
    corpus = pipeline.load_corpus(opts.bib, nlp=nlp, require=opts.require, fill=opts.fill,
                                  maybe_crossref_cache=ledger_dir / scope_cli.CROSSREF_CACHE)
    records = [record for record in corpus.records if record.key in decided]
    missing = sorted(set(decided) - {record.key for record in records})
    print(f"{len(records)} of {len(decided)} records in the corpus" + (f"; absent: {missing}" if missing else ""))

    settings = scope_cli._connector(opts.backend_url)()
    fingerprint = sweep.fingerprint(settings, pipeline.materials(definition)["extract"])
    print(f"model {settings.model_record.model_id}, extract instrument {fingerprint}")
    ledger = scope_extract.ledger(ledger_dir, fingerprint)
    digests = {record.key: scope_extract.digest(record) for record in records}
    done = ledger.answers(digests)
    result = sweep.Sweep(settings, functools.partial(scope_extract.ask_fields, definition=definition),
                         [(record.key, record) for record in records if record.key not in done], ledger,
                         batch_size=pipeline.EXTRACT_BATCH, header=sweep.instrument_header(settings),
                         digest=scope_extract.digest).run()
    if result.outcome is not sweep.completed:
        print(f"stopped: {result.message or result.outcome}")
        return 1

    answers = ledger.answers(digests)
    rows, counts = [], collections.Counter()
    for key, row in decided.items():
        if key not in answers:
            continue
        fields = answers[key]["fields"]
        rows.append([key, row["decision"], row.get("level (prototype)", ""), fields.get("level"), row.get("population (prototype)", ""),
                     fields.get("population"), fields.get("human_learning"), fields.get("ai_role"), row.get("title", "")])
        counts[row["decision"] or "unsettled", fields.get("level")] += 1
    tabular.write_table(opts.out, [f"Extraction by {settings.model_record.model_id}, instrument {fingerprint}"],
                        ["key", "decision", "level before", "level", "population before", "population", "human_learning", "ai_role",
                         "title"], rows)
    for (decision, level), n in sorted(counts.items()):
        print(f"{decision:>10}  {level:<24}{n}")
    print("wrote", opts.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
