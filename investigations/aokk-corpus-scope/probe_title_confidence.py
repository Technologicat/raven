"""Does a model's title-pass confidence say when its title-only verdict will be wrong?

The judge escalates a record to the abstract pass on *low* confidence, a title too thin to judge from, or any
test answered true. Everything else — medium and high confidence keeps — is settled on the title alone. So
whether a model says *low* when it should is what decides how many wrong keeps pass on a title. This probe
measures it, one model per run:

1. Sample records (seeded, after the sift and the fill, as `raven-scopebib` reads them).
2. Ask the title pass about each, as the tool does.
3. Ask the abstract pass about **every** sampled record, the ones the tool would not have escalated included.
4. For each title confidence, and whether the tool would have escalated, count how often the abstract pass
   says something different about scope than the title pass did: in, out, or unknown.

A calibrated model is overturned less at high than at medium, and the records it would have escalated hold
most of the overturns. The reference is the same model with more text rather than ground truth; a blind
human rating of a stratified subset checks it (`README.md` → *The run queued next*).

Answers go to the tool's own ledgers, under the instrument the definition and the model make, so a title answer
is one the full run would have asked anyway, and counts toward it. A forced abstract answer is the abstract
pass's own answer to that record and text, read by a run only if the record is ever escalated.

Usage, once per model:

    python investigations/aokk-corpus-scope/probe_title_confidence.py \\
        00_stuff/rawdata/AOKK/multisource/tekoalyagentti_tutkimus_deduped.bib \\
        --definition investigations/aokk-corpus-scope/aokk.scope.toml \\
        --ledgers 00_stuff/rawdata/AOKK/scopebib-2026-10-26/ledgers \\
        --require abstract --fill --sample 300 --model qwen3.6-35b-a3b \\
        --out 00_stuff/rawdata/AOKK/confidence-probe/qwen3.6.tsv

`--model` loads that model through LM Studio for the run and puts the one loaded before back afterwards.
Without it, whatever is loaded is asked.
"""

import argparse
import collections
import functools
import logging
import pathlib
import sys

from raven.common import tabular
from raven.librarian import sweep
from raven.papers.scopebib import cli as scope_cli
from raven.papers.scopebib import definition as scope_definition
from raven.papers.scopebib import judge, pipeline
from raven.papers.scopebib import models as scope_models

COLUMNS = ("key", "title confidence", "title says", "escalated", "why escalated",
           "abstract confidence", "abstract says", "overturned", "title")


def _stance(answer: dict) -> str:
    """What an answer says about scope: `"out"` if a test came back true, `"in"` if every one came back false,
    `"unknown"` if one was left unanswered.

    Compared rather than `judge.verdict_of`, whose `"review"` for a title-pass true at medium confidence is
    the tool's routing — ask again with the abstract — and so never equals what the abstract pass says.
    """
    tests = answer["tests"].values()
    if True in tests:
        return "out"
    return "unknown" if None in tests else "in"


def _sweep(settings, ask, records, ledger, *, definition, batch_size, digest) -> None:
    result = sweep.Sweep(settings, functools.partial(ask, definition=definition),
                         [(record.key, record) for record in records], ledger, batch_size=batch_size,
                         header=sweep.instrument_header(settings), digest=digest).run()
    if result.outcome is not sweep.completed:
        raise SystemExit(f"probe_title_confidence: stopped: {result.message or result.outcome}")
    if result.failed:
        print(f"note: {result.failed} records failed and are asked again on the next run of this probe")


def _why_escalated(answer: dict, measure) -> str:
    reasons = [name for name, fired in (("low confidence", answer["confidence"] == "low"),
                                        ("thin title", measure.uninformative),
                                        ("a test true", True in answer["tests"].values())) if fired]
    return ", ".join(reasons)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bib")
    parser.add_argument("--definition", required=True)
    parser.add_argument("--ledgers", required=True)
    parser.add_argument("--require", action="append", default=[])
    parser.add_argument("--fill", action="store_true")
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None, help="LM Studio's key for the model to ask")
    parser.add_argument("--backend-url", default=None)
    parser.add_argument("--out", required=True, help="per-record table, .tsv/.xlsx/.ods")
    opts = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("bibtexparser").setLevel(logging.ERROR)

    ledger_dir = pathlib.Path(opts.ledgers)
    definition = scope_definition.load_definition(opts.definition)
    nlp, _ = scope_cli._nlp(on_server=False, server_url=None)
    corpus = pipeline.load_corpus(opts.bib, nlp=nlp, require=opts.require, fill=opts.fill,
                                  maybe_crossref_cache=ledger_dir / scope_cli.CROSSREF_CACHE,
                                  maybe_sample=opts.sample, seed=opts.seed)
    records = corpus.records
    print(f"{len(records)} records sampled")

    connect = scope_cli._connector(opts.backend_url)
    settings = connect()
    initial = settings.model_record.model_id
    maybe_switcher = None
    if opts.model and opts.model != initial:
        maybe_switcher = scope_models.switcher_for(settings, reconnect=connect)
        if maybe_switcher is None:
            raise SystemExit("probe_title_confidence: --model needs an LM Studio backend")
        settings = maybe_switcher.load(opts.model)
    try:
        fingerprint = sweep.fingerprint(settings, pipeline.materials(definition)["judge"])
        print(f"model {settings.model_record.model_id}, judge instrument {fingerprint}")
        titles, abstracts = judge.ledgers(ledger_dir, fingerprint)
        _sweep(settings, judge.ask_titles, judge.todo_titles(records, titles=titles), titles,
               definition=definition, batch_size=pipeline.TITLE_BATCH, digest=judge.title_digest)
        by_abstract = abstracts.answers({record.key: judge.abstract_digest(record) for record in records})
        _sweep(settings, judge.ask_abstracts, [record for record in records if record.key not in by_abstract],
               abstracts, definition=definition, batch_size=pipeline.ABSTRACT_BATCH, digest=judge.abstract_digest)
    finally:
        if maybe_switcher is not None:
            maybe_switcher.load(initial)

    by_title = titles.answers({record.key: judge.title_digest(record) for record in records})
    by_abstract = abstracts.answers({record.key: judge.abstract_digest(record) for record in records})
    rows, cells = [], collections.defaultdict(collections.Counter)
    for record in records:
        if record.key not in by_title or record.key not in by_abstract:
            continue  # failed this run; asked again on the next
        title_answer, abstract_answer = by_title[record.key], by_abstract[record.key]
        title_verdict, abstract_verdict = _stance(title_answer), _stance(abstract_answer)
        escalated = judge.needs_escalation(title_answer, corpus.measures[record.key])
        overturned = title_verdict != abstract_verdict
        rows.append((record.key, title_answer["confidence"], title_verdict, "yes" if escalated else "no",
                     _why_escalated(title_answer, corpus.measures[record.key]), abstract_answer["confidence"],
                     abstract_verdict, "yes" if overturned else "no", record.title))
        cell = cells[(title_answer["confidence"], escalated)]
        cell["records"] += 1
        cell["overturned"] += overturned

    out = pathlib.Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tabular.write_table(out, [f"model {settings.model_record.model_id}, judge instrument {fingerprint}",
                              f"sample {opts.sample}, seed {opts.seed}, from {opts.bib}"], COLUMNS, rows)

    print(f"\n{'title confidence':<18}{'escalated':<11}{'records':>8}{'overturned':>12}{'rate':>8}")
    for confidence in judge.CONFIDENCES:
        for escalated in (False, True):
            cell = cells.get((confidence, escalated))
            if cell:
                print(f"{confidence:<18}{'yes' if escalated else 'no':<11}{cell['records']:>8}"
                      f"{cell['overturned']:>12}{cell['overturned'] / cell['records']:>8.1%}")
    total = sum(cell["overturned"] for cell in cells.values())
    caught = sum(cell["overturned"] for (_, escalated), cell in cells.items() if escalated)
    if total:
        print(f"\noverturns the tool would have caught by escalating: {caught} of {total} ({caught / total:.0%})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
