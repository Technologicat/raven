"""Ask what a record *says*, not whether it belongs, and filter on the answers afterwards.

The scope judge decides. That is the right shape for finding records to throw away — a drop needs a
reason — and the wrong shape for the records it *keeps*, because a keep needs no reason at all. A record
survives the rubric whenever no test can be positively established, so a study whose level is simply
never stated is kept by the same rule that keeps a study plainly set in a university, and afterwards the
two are indistinguishable. Reviewing every dropped record found 3.9% of the confident title-only drops
contested; the corresponding question about the keeps has no answer, because nothing was recorded.

So this asks for **fields rather than a verdict**:

  - `population`      who the work actually studies
  - `level`           the educational level, with "not stated" a first-class answer
  - `human_learning`  whether a PERSON is being taught or learning, as against a model being trained
  - `ai_role`         what the AI does in the work, in a few words, or "none"
  - `evidence`        the phrase in the text that settles `level`, quoted

Three things follow, and the third is why this is worth a run of its own.

**Extraction is an easier task than judgement.** Reporting what a text says does not require weighing it
against a rubric, and does not need the model to be confident enough to assert a negative.

**A removal carries its reason.** `level: school, evidence: "sixth-grade pupils"` is checkable at a
glance; `wrong_level: true` is not.

**Re-filtering is free.** The fields are stored, so a cutoff can be changed, argued about and changed
again without another model call. That is the opposite of the judge, where every adjustment cost a run —
and it is what lets the filter be tuned against the corpus rather than against the examples that
motivated it.

The run is a `raven.librarian.sweep`: one ledger line per record as its answer arrives, so an
interrupted run resumes by running it again, and records that failed are asked again. The ledger is named
for the instrument — the vocabularies, the prompt, the model and its quantization, and what `--declare`
states — so a run under a changed instrument writes its own file. Reasoning traces go to a file beside
it, one entry per model call, naming the records that shared the call.
"""

import argparse
import collections
import json
import logging
import pathlib
import random
import re
import sys

from unpythonic.env import env

from raven.common import netutil
from raven.librarian import agent, config as librarian_config, llmclient, sweep

logger = logging.getLogger(__name__)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import judge_scope  # noqa: E402 -- the path has to be set up first; this is the sibling apparatus

SCOPE_QUESTION = judge_scope.SCOPE_QUESTION

ABSTRACT_CHARS = 4000

# The vocabularies. Closed sets, because an open-ended answer cannot be filtered on without a second pass
# to normalize it, and normalizing free text is how a field stops meaning one thing.
POPULATIONS = ("university_students", "school_pupils", "preservice_teachers", "inservice_teachers",
               "researchers", "professionals", "general_public", "none", "unclear")
LEVELS = ("higher_education", "school", "vocational", "professional_training", "informal",
          "mixed", "not_applicable", "not_stated")


INSTRUCTIONS = """\
You are reading records from a literature search, to record what each one says. You are NOT deciding \
whether any of them belongs anywhere - do not screen, include or exclude. Report what is there.

For each numbered record, answer:
  "i"               the record's number, copied exactly
  "population"      who the work studies, one of: {populations}
  "level"           the educational level the work is set at, one of: {levels}
  "human_learning"  true if a PERSON is taught or is learning in this work; false if the only thing being \
taught or trained is software (a model, an agent, a policy); "unclear" if you cannot tell
  "ai_role"         what the AI does in this work, at most eight words, or "none" if there is no AI in it
  "evidence"        the words in the record that settle "level", quoted exactly, or "" if nothing does

Rules that decide most cases:

"not_stated" is a real answer and often the correct one. A great many studies never say what level their \
participants are at. Answer "not_stated" rather than inferring from the topic, the venue or what would be \
usual - if the text does not say, you do not know. Guessing here destroys the only thing this task is for.

"not_applicable" is different from "not_stated": it means the work is not set in education at all, so the \
question does not arise. A study of clinical decision-making has level "not_applicable"; a study of \
students whose year is never given has level "not_stated". Reserve "not_applicable" for work with no \
teaching or learning in it whatsoever - a methods paper, a recommender, a clinical tool. If somebody is \
learning something, the work IS set in education and one of the other values applies.

Four values name the places learning happens, and the distinction between them is the SETTING, not the \
subject matter or the age of the learner:
  "higher_education"      a university, college, polytechnic or their students and faculty - degree study
  "school"                primary or secondary schooling, K-12, pupils
  "vocational"            formal vocational or trade education leading to a qualification
  "professional_training" workplace and professional development: practitioners already qualified, being \
coached, upskilled or trained on the job. Psychotherapist training, teacher CPD, staff onboarding
  "informal"              learning outside any institution - self-directed learners, the general public, \
lifelong learning, patient or rehabilitation training, museum and hobby settings

Two of those are new and easy to over-apply, so note what they are NOT. A course taught to university \
students is "higher_education" however professional its subject; medicine, law and teacher preparation \
are degree study. And "informal" is about the absence of an institution, not the absence of a stated \
year - a study of undergraduates whose year is never given is "not_stated", not "informal".

A review, survey or meta-analysis has no setting of its own: it reports on other people's studies, so \
nobody is being taught inside it. That does NOT make it "not_applicable". A review of AI in education is \
about education. Give it the level its own scope names - "school" for a review of classroom chatbots - \
and "not_stated" when it names none, which is the common case.

"human_learning" is the distinction between a person being educated and a model being trained. \
"Teaching", "training" and "learning" are the machine-learning field's own words for what is done to \
software. A "teachable agent" corrected by crowdworkers, a system "trained" on a corpus, a paper about \
"learning" a control policy - in all of those the thing being taught is software, and "human_learning" is \
false.

"evidence" must be words that actually appear in the record. If you cannot find any, answer "" - do not \
paraphrase and do not compose a sentence. An empty "evidence" beside a confident "level" is a \
contradiction, so let the evidence decide.

Answer with a JSON array of objects and nothing else. One object per record, in order, no commentary, no \
markdown fences.

{items}
"""


def instrument_material() -> str:
    """The extractor's half of the instrument, for `sweep.fingerprint`: the vocabularies and the prompt.

    The prompt is in here because the vocabularies alone are not the instrument. A value set can stay
    fixed while the paragraph explaining it changes what the model does with it — telling it what to do
    with a systematic review moved records out of one value and into another without touching either
    list — so answers from before and after that edit are not the same measurement and must not be
    pooled. Hashing both means any change to either starts a fresh run, and the safe direction is the
    cheap one: at worst a typo fix costs a partial re-run.
    """
    return "|".join(POPULATIONS) + "//" + "|".join(LEVELS) + "//" + INSTRUCTIONS


def quote_bare_vocabulary(reply: str) -> str:
    """Put quotes back on a vocabulary word the model emitted as a bareword.

    `"human_learning": unclear` is not JSON, and the whole batch fails to parse over it — ten records
    lost because one field in one of them came back unquoted. The schema invites it: that field's three
    values are `true`, `false` and `"unclear"`, so two of them are JSON literals and the third is a
    string, and generalising from the two to the third is the obvious thing to do.

    Repairing it here is safe in a way a general bareword repair would not be, because every value this
    touches comes from a closed set: a bare `unclear` or `school` in this reply can only be the token,
    since nothing else in the schema is unquoted except `true`, `false` and the numbers.

    The better fix is to stop mixing the two kinds — three quoted strings, or a nullable boolean — but
    that is a change to the prompt, and so to the instrument, which would discard a run in progress to
    buy nothing this does not already buy.
    """
    tokens = sorted(set(POPULATIONS) | set(LEVELS) | {"unclear"}, key=len, reverse=True)
    pattern = r':\s*(' + "|".join(re.escape(t) for t in tokens) + r')\s*(?=[,}])'
    return re.sub(pattern, lambda m: f': "{m.group(1)}"', reply)


def format_record(index: int, record: env) -> str:
    """One record as the extractor sees it. Same shape pass 2 of the judge uses, minus the verdict."""
    venue = f"Published in: {record.venue}\n" if record.venue else ""
    abstract = record.abstract[:ABSTRACT_CHARS] if record.abstract else "(none)"
    return (f"--- record {index} ---\n"
            f"Title: {record.title}\n"
            f"{venue}"
            f"Abstract: {abstract}\n")


def extract_batch(llm_settings: env, batch: list[tuple[str, env]],
                  maybe_abort: netutil.Abort | None = None) -> tuple[dict[str, dict], list[agent.TurnRecord]]:
    """Fields for one batch of `(citekey, record)` pairs, plus the call that answered them.

    The `ask_batch` of a `sweep.Sweep`. Returns `({citekey: fields}, [turn])`. An answer whose number does
    not resolve is dropped rather than guessed at, exactly as the judge's batched calls do, and the sweep
    records the record it belonged to as failed.
    """
    items = "\n".join(format_record(i, record) for i, (_, record) in enumerate(batch))
    record = agent.ask_record(llm_settings,
                              INSTRUCTIONS.format(populations=" / ".join(POPULATIONS),
                                                  levels=" / ".join(LEVELS),
                                                  items=items),
                              maybe_abort=maybe_abort)
    # A reply that does not parse answers nothing rather than raising, so that the sweep keeps the turn and
    # the reply can be read afterwards; every record in the batch is then recorded as failed.
    try:
        answers = agent.parse_json_reply(quote_bare_vocabulary(record.reply or ""))
    except ValueError:
        answers = None
    if not isinstance(answers, list):
        logger.warning(f"extract_batch: expected a JSON array, got {type(answers).__name__}; the reply is in the traces")
        return {}, [record]
    out = {}
    for answer in answers:
        if not isinstance(answer, dict) or "i" not in answer:
            continue
        try:
            i = int(answer["i"])
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(batch):
            out[batch[i][0]] = normalize(answer)
    return out, [record]


def normalize(answer: dict) -> dict:
    """One extraction, with its vocabularies checked.

    An out-of-vocabulary value becomes "unclear"/"not_stated" rather than being stored as given: a field
    whose set of values is open cannot be filtered on, and the filter is the entire point. The original
    is kept in `raw` so a systematic mismatch is visible rather than silently flattened.
    """
    population = str(answer.get("population", "")).strip().lower()
    level = str(answer.get("level", "")).strip().lower()
    learning = answer.get("human_learning")
    if learning not in (True, False):
        learning = "unclear"
    out = {"population": population if population in POPULATIONS else "unclear",
           "level": level if level in LEVELS else "not_stated",
           "human_learning": learning,
           "ai_role": " ".join(str(answer.get("ai_role") or "").split())[:80],
           "evidence": " ".join(str(answer.get("evidence") or "").split())[:200]}
    if out["population"] != population or out["level"] != level:
        out["raw"] = {"population": population, "level": level}
    return out


def seed_from(path: pathlib.Path, out_dir: pathlib.Path, run_name: str) -> sweep.Ledger:
    """A ledger holding the answers of an extraction written before the sweep, under its own fingerprint.

    Those files have one flat line per record, `{"key", "v", <fields>}`. The fingerprint is the one the
    file records, not this script's, since the answers were made by that instrument: a run under the
    current one never reads them. Refuses a file holding more than one, which is the mixture the
    fingerprint exists to prevent.
    """
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fingerprints = {line.get("v") for line in lines}
    if len(fingerprints) != 1 or None in fingerprints:
        raise ValueError(f"{path.name} holds answers from {len(fingerprints)} instruments ({sorted(map(str, fingerprints))}); "
                         f"seed from a file holding one")
    ledger = sweep.Ledger(out_dir, run_name, fingerprints.pop())
    if ledger.path.resolve() == path.resolve():
        raise ValueError(f"seeding {path.name} into itself; pass --out-dir to put the ledger elsewhere")
    for line in lines:
        ledger.write_answer(line["key"], {name: value for name, value in line.items() if name not in ("key", "v")})
    return ledger


def unsure_keeps(judged: dict[str, dict]) -> set[str]:
    """The kept records whose keep rests on nothing: hedged, unsure, or a withheld verdict.

    A confident keep was reached by the model positively failing to find any evidence of being off topic,
    which is the rubric working. These are the ones where it did not look hard enough to say.
    """
    out = set()
    for key, answer in judged.items():
        verdict = judge_scope.verdict_of(answer)
        if verdict == "unknown" or (verdict == "keep" and answer["confidence"] != "high"):
            out.add(key)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bib", required=True, help="the .bib the records come from")
    parser.add_argument("--judged", default=None, help="the judge's state JSONL (default: beside this)")
    parser.add_argument("--all-keeps", action="store_true",
                        help="extract from every kept record, not only the ones kept on a hedge")
    parser.add_argument("--pilot", type=int, default=None, metavar="N",
                        help="extract a random N of the selection and stop, to read before committing "
                             "to the whole run")
    parser.add_argument("--seed", type=int, default=23, help="which random sample --pilot takes")
    parser.add_argument("--batch", type=int, default=10, help="records per model call")
    parser.add_argument("--backend-url", default=None,
                        help=f"the LLM backend (default: {librarian_config.llm_backend_url})")
    parser.add_argument("--declare", default=None, metavar="TEXT",
                        help="backend settings the backend does not report, e.g. 'KV cache q4_0'. Part of "
                             "the instrument, so it is in the fingerprint and in the ledger's header")
    parser.add_argument("--seed-from", default=None, metavar="FILE",
                        help="instead of asking anything, turn an extraction written before the sweep into "
                             "a ledger, under the fingerprint that file records")
    parser.add_argument("--out-dir", default=None, help="where the outputs go (default: beside this)")
    opts = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("bibtexparser").setLevel(logging.ERROR)
    here = pathlib.Path(__file__).resolve().parent
    out_dir = pathlib.Path(opts.out_dir) if opts.out_dir else here

    records = judge_scope.load_records(pathlib.Path(opts.bib).expanduser().resolve())
    judged = judge_scope.load_state(pathlib.Path(opts.judged) if opts.judged else here / "judged.jsonl")

    if opts.all_keeps:
        wanted = {key for key, answer in judged.items() if judge_scope.verdict_of(answer) != "drop"}
        run_name = "extracted-all-keeps"
    else:
        wanted = unsure_keeps(judged)
        run_name = "extracted"
    selection = [record for record in records if record.key in wanted]
    if opts.pilot is not None:
        selection = random.Random(opts.seed).sample(selection, min(opts.pilot, len(selection)))
        run_name = f"{run_name}-pilot-{opts.seed}-{len(selection)}"
    print(f"{len(wanted)} records selected; {len(selection)} in this run")

    if opts.seed_from:
        ledger = seed_from(pathlib.Path(opts.seed_from), out_dir, run_name)
        print(f"seeded {ledger.path} from {opts.seed_from}")
    else:
        # The instrument is in the filename, so a changed vocabulary, prompt or model writes a new file
        # rather than appending to one whose earlier answers mean something subtly different.
        llm_settings = llmclient.setup(backend_url=opts.backend_url or librarian_config.llm_backend_url,
                                       quiet=True)
        ledger = sweep.Ledger(out_dir, run_name,
                              sweep.fingerprint(llm_settings, instrument_material(), declared=opts.declare))
        print(f"instrument {ledger.fingerprint} -> {ledger.path.name}")
        result = sweep.Sweep(llm_settings, extract_batch,
                             [(record.key, record) for record in selection],
                             ledger,
                             batch_size=opts.batch,
                             header=sweep.instrument_header(llm_settings, declared=opts.declare)).run()
        if result.outcome is not sweep.completed:
            print(f"\nstopped ({result.outcome}): {result.message or 'cancelled'}\nrun again to resume")
            return 1

    answers = ledger.answers()
    mine = {key: answers[key] for key in (record.key for record in selection) if key in answers}
    print(f"\n{len(mine)} extracted")
    for field in ("level", "population", "human_learning"):
        counts = collections.Counter(value[field] for value in mine.values())
        print(f"\n  {field}:")
        for name, count in counts.most_common():
            print(f"    {str(name):<22}{count:>5}")
    print(f"\nwrote {ledger.path}\nwrote {ledger.traces_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
