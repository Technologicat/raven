"""Sort the AOKK corpus into in-scope and out-of-scope, so the false positives can be reviewed.

The corpus was assembled by a boolean query — an AI-agent term AND a collaborative-learning term AND
`student*` AND a higher-education term — and the other three blocks are broad enough that a paper about
something else entirely can clear all four. Two shapes of false positive are confirmed in it: a term
matching outside its intended sense (`"conversational agent"` catching a child-helpline paper), and a term
of art the query did not mean (`"learning assistant"` also being the established name for a *human*
undergraduate teaching helper, which has no AI in it at all).

The judgement is against the broad question the search actually asked:

    Studies on different aspects of the use of AI agents in higher education.

It is asked as three booleans rather than one, because a drop is only reviewable if it says which test
failed: no AI in it, no education in it, or the wrong level of education. Each is asked as evidence of
being *off* topic rather than as a test of being on it — every record here already matched the search, so
a title that does not restate the setting says nothing against it. Asked the other way round, the model
answers "not higher education" for any title that merely fails to say, and drops real studies with
nothing in the output looking wrong.

Which is why "no education at all" is a test of its own rather than part of the level question, and it is
the one place where absence *is* the evidence. A machine-learning methods paper is not set at the wrong
level; it is not set anywhere. Folded into the level test it slips through — the model reads "positively
set somewhere other than higher education", finds no setting at all, and keeps a link-prediction survey.

Two passes. Pass 1 asks about titles in batches, which is enough for the clear cases and is what makes
5167 records affordable. Pass 2 re-asks about title *and* abstract, one record at a time, for everything
pass 1 was unsure about. 83% of this corpus has an abstract — though a tenth of those are publisher
teasers that break off mid-sentence, which pass 2 is told about, since a blurb read as a whole abstract
invites exactly the concluding-from-absence the rubric otherwise forbids.

Escalation fires on either a low-confidence answer or a title with too few words in it. The second
condition is computed here rather than asked of the model, and that is the point: a sibling run over a
paper pile found the model at its most confident exactly where the input carried least — confidently
naming the subject of a file called `2006.05563.pdf` — so a rule driven by the model's own confidence is
blind in precisely the place it most needs to look.

Resumable. Every answer is appended to a JSONL as it arrives and a re-run skips what is already there, so
a backend hiccup two-thirds of the way through costs the current batch rather than the run.

Reads a `.bib` rather than the Visualizer dataset built from it: the citekey is a key the outputs can be
written against, and the filtered `.bib` is one of the two things this produces.

    python judge_scope.py --bib .../tekoalyagentti_tutkimus_deduped.bib --pilot 200
    python judge_scope.py --bib .../tekoalyagentti_tutkimus_deduped.bib
"""

import argparse
import json
import logging
import pathlib
import random
import re
import sys

import bibtexparser
from bibtexparser.model import Entry

from unpythonic import timer
from unpythonic.env import env

from raven.librarian import agent, config as librarian_config, llmclient
from raven.papers import bibtex

# The question the corpus is supposed to answer, quoted into both prompts so the two passes judge the
# same thing. `00_stuff/rawdata/AOKK/search-phrase.txt` is the boolean query this paraphrases; the
# research questions narrow it further and are deliberately *not* asked here, being a separate pass over
# a corpus that has already had its obvious strays taken out.
SCOPE_QUESTION = "studies on different aspects of the use of AI agents in higher education"

# What counts as too little to judge from. Set from the corpus rather than by taste: titles here run to a
# median of 13 words with a 5th percentile of 7, and the 41 records (0.8%) below this bound are genuinely
# the uninformative ones — "Book Review", "Machine culture", "Generative AI".
MIN_INFORMATIVE_WORDS = 5

# How much of an abstract pass 2 sends. Comfortably more than an abstract normally runs to; the cap is
# against the occasional record whose "abstract" field holds a whole introduction.
ABSTRACT_CHARS = 4000

# How much of an abstract the pilot TSV shows, so that a title too thin to judge from can still be
# adjudicated by hand without opening the .bib.
ABSTRACT_PREVIEW_CHARS = 300

# Below this, an abstract carrying an ellipsis is taken to be a truncated teaser rather than a short
# abstract. See `looks_truncated`; the corpus's median abstract is 1334 characters, so this is well clear
# of anything complete.
TEASER_CHARS = 600

_SCOPE_RUBRIC = """\
These records came out of a literature search for: {question}. Every one of them already matched that \
search. Your job is to find the ones that matched it by accident - the ones that are demonstrably about \
something else.

So do NOT ask whether each record is on topic. Ask whether there is positive evidence that it is off \
topic, and answer three things separately. Answer each one on its own; a record can fail any one of them \
and pass the other two.

  "no_ai"          true ONLY if the work is positively about something with no artificial intelligence in \
it - human teaching staff, human tutors, human undergraduate learning assistants, or a non-AI technology. \
False when AI is present in any way (an AI agent, assistant, chatbot, tutor or LLM-based tool, or AI in \
general - including its use, its effects, attitudes towards it, policy and ethics about it, and AI \
literacy), AND false when the work does not say what it used.
  "not_education"  answered in two steps, in this order. FIRST: can you name what this work is about, \
from what you were given? If you cannot - the title is generic, an editorial or a section heading, or a \
proper noun or acronym you do not recognize - then answer FALSE and "low", and stop. SECOND, only if you \
can name the subject: is that subject entirely outside education? Answer true only for a subject you can \
name that plainly involves no teaching, learning, students or courses - a machine-learning methods paper, \
a computer-vision or natural-language-processing study, a finance, engineering, energy or medical \
application. Teaching, learning, students, courses, curricula, training and education policy all count as \
an educational dimension, at ANY level.
  "wrong_level"    true ONLY if the work is positively set at a level other than higher education - \
preschool, primary or secondary school, K-12, workplace or professional training outside a university, or \
the general public. False when the setting is a university, college or polytechnic, or its students, \
teachers or courses, AND false when the work does not say what level it is set at.

Two of these are about silence and one is not, so read this carefully:

  - If a record does not say WHAT LEVEL it is set at, that is NOT "wrong_level". These records all matched \
a higher-education search term already. Answer false and lower your confidence.
  - If a record does not say WHAT METHOD it used, that is NOT "no_ai". Answer false and lower your \
confidence.
  - But a work whose subject you CAN name, where that subject has no teaching, no learning, no students \
and no courses in it, IS "not_education". There the absence is evidence, because a study about education \
says so. A survey of machine-learning techniques that never mentions a learner is "not_education" true, \
however much AI is in it.

That last one is the dangerous test, so guard it: a short title is mostly absence no matter what the \
paper is, and absence you cannot interpret is not evidence. An identifier is not a description. If you \
are given a bare proper noun, an acronym, a project name, a journal section or a heading, you do NOT know \
what the work is about - answer false to all three and say "low". Do NOT claim to recognize what an \
unfamiliar name refers to; you cannot, and a guess dressed as recognition is worse than admitting the \
title says nothing.

Some records also carry the journal, book or conference they appeared in, shown as "[published in: ...]". Use it - it is often the strongest evidence there is, especially when the title is short. A venue named for a field tells you the field: "Journal of Chemical Education" or "Proceedings of the Learning Analytics and Knowledge Conference" is educational research, and "IEEE Transactions on Smart Grid" is not. Two cautions. A venue rarely says the LEVEL - an education journal covers schools and universities alike - so it is weak evidence for "wrong_level". And it describes where the work was published, not what the work is: a book review in an education journal is still a book review.

A phrase can match in a sense the search did not mean, and this is the main thing to look for. "Learning \
assistant" names an AI tool, and it is ALSO the established term in STEM education research for a HUMAN \
undergraduate who helps teach a course - that second sense has no AI in it and is "no_ai" true. \
"Conversational agent" and "intelligent agent" appear in fields with nothing to do with education. Judge \
the work, not the phrase that matched.

Report your confidence honestly. A generic title - "Book Review", "Generative AI", "Machine culture" - \
gives you no evidence either way: answer false to all three and say "low". Do not guess confidently."""

PASS1_INSTRUCTIONS = """\
You are screening a literature-search result for records that are not about the topic it was searching for.

{rubric}

For each numbered item below you are given a paper's title, and nothing else. For each item, answer:
  "i"              the item's number, copied exactly
  "no_ai"          true or false, as above
  "not_education"  true or false, as above
  "wrong_level"    true or false, as above
  "confidence"     "high", "medium" or "low" - how sure you are, given only the title
  "why"            at most twelve words, why you answered that way

Answer with a JSON array of objects and nothing else. One object per item, in order, no commentary, no \
markdown fences.

Items:
{items}
"""

TRUNCATION_CAVEAT = """
NOTE: this abstract is a truncated preview - the publisher cut it off, and it ends mid-sentence. Judge \
ONLY from what is actually present. Whatever the abstract has not reached yet is not evidence of \
anything, so lower your confidence rather than concluding from what is missing.
"""

PASS2_INSTRUCTIONS = """\
You are screening a literature-search result for records that are not about the topic it was searching for.

{rubric}

The title of this record said too little to judge it from, so here is its abstract as well.
{caveat}
Title: {title}
{venue}
--- abstract ---
{abstract}
--- end ---

Answer with a single JSON object and nothing else:
  "no_ai"          true or false, as above
  "not_education"  true or false, as above
  "wrong_level"    true or false, as above
  "confidence"     "high", "medium" or "low"
  "why"            at most twelve words
"""


def load_records(bib_path: pathlib.Path) -> list[env]:
    """Every record of a `.bib`, as `env(key, title, abstract)`, in file order.

    `abstract` is the empty string where the record has none. Braces are stripped from the title — BibTeX
    uses them to protect capitalization, and they are noise in a prompt and in a TSV alike.
    """
    library = bibtex.parse_file(str(bib_path), split_names=False)
    records = []
    for entry in library.entries:
        fields = entry.fields_dict
        title = fields["title"].value if "title" in fields else ""
        abstract = fields["abstract"].value if "abstract" in fields else ""
        records.append(env(key=entry.key,
                           title=_clean(title.replace("{", "").replace("}", "")),
                           venue=_venue(fields),
                           abstract=_clean(abstract)))
    return records


def _venue(fields) -> str:
    """Where the record was published, from the first field that names it.

    Worth sending because it is evidence the title often does not carry, and it is *there*: 85.3% of this
    corpus has one, and — the case that matters — so do 852 of the 853 records with no abstract at all.
    A venue like "CEPS Journal: Center for Educational Policy Studies" or "Proceedings of the Learning
    Analytics and Knowledge Conference" settles the domain question outright.
    """
    for key in ("journal", "booktitle", "series", "publisher"):
        if key in fields and str(fields[key].value).strip():
            return _clean(str(fields[key].value).replace("{", "").replace("}", ""))
    return ""


def _clean(text: str) -> str:
    """One line of whitespace-normalized text, which is what both a prompt item and a TSV cell want."""
    return " ".join((text or "").split())


def informative_words(title: str) -> int:
    """How many words of the title carry letters — digits and punctuation say nothing about a subject."""
    return len([word for word in re.split(r"[\s\-_/]+", title) if any(c.isalpha() for c in word)])


def looks_uninformative(title: str) -> bool:
    """Whether a title is too thin to judge from — the half of the escalation rule the model cannot veto."""
    return informative_words(title) < MIN_INFORMATIVE_WORDS


def looks_truncated(abstract: str) -> bool:
    """Whether an "abstract" is really a publisher's teaser, cut off before it says anything.

    Measured on this corpus: 435 records (10.1% of those that have an abstract) both carry an ellipsis
    and run under `TEASER_CHARS`, against a median abstract of 1334 characters. They break off
    mid-sentence — "Given this ...", "with an increased ..." — so they stop well before any statement of
    method or setting.

    Both conditions together, because either alone is wrong: a full abstract may quote an ellipsis from a
    title, and a genuinely terse abstract is short without being cut off.
    """
    return bool(abstract) and len(abstract) < TEASER_CHARS and ("..." in abstract or "…" in abstract)


def describe_abstract(record: env) -> str:
    """What kind of evidence this record's abstract is, for the reviewer's column: none, teaser, or full."""
    if not record.abstract:
        return "none"
    return "teaser" if looks_truncated(record.abstract) else "full"


def judge_titles(llm_settings: env, batch: list[env]) -> dict[int, dict]:
    """Pass 1 over one batch. Returns `{position within batch: answer}`.

    Each item is numbered and the model returns those numbers back, so an answer whose index does not
    resolve is dropped rather than guessed at. A batch that comes back short simply leaves those records
    unanswered, which puts them in the next run's to-do list — and is what makes a re-run the recovery
    path for a failed batch.
    """
    items = "\n".join(f"{i}. {record.title}" + (f"\n   [published in: {record.venue}]" if record.venue else "")
                      for i, record in enumerate(batch))
    reply = agent.ask(llm_settings, PASS1_INSTRUCTIONS.format(rubric=_SCOPE_RUBRIC.format(question=SCOPE_QUESTION),
                                                        items=items))
    answers = agent.parse_json_reply(reply)
    if not isinstance(answers, list):
        raise ValueError(f"expected a JSON array, got {type(answers).__name__}")
    out = {}
    for answer in answers:
        if not isinstance(answer, dict) or "i" not in answer:
            continue
        try:
            i = int(answer["i"])
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(batch):
            out[i] = answer
    return out


def judge_abstract(llm_settings: env, record: env) -> dict:
    """Pass 2 over one record, reading its abstract as well as its title."""
    if not record.abstract:
        return {"no_ai": None, "not_education": None, "wrong_level": None, "confidence": "low",
                "why": "no abstract to read"}
    # A tenth of this corpus's abstracts are publisher teasers that break off mid-sentence, so the model
    # is told when it is looking at one. Without that it reads a truncated blurb as a whole abstract and
    # concludes from what is missing — which is the same silence-is-not-evidence mistake the rubric spends
    # three paragraphs on, arriving by a different door.
    caveat = (TRUNCATION_CAVEAT if looks_truncated(record.abstract) else "")
    reply = agent.ask(llm_settings, PASS2_INSTRUCTIONS.format(rubric=_SCOPE_RUBRIC.format(question=SCOPE_QUESTION),
                                                              caveat=caveat,
                                                              title=record.title,
                                                              venue=(f"Published in: {record.venue}\n"
                                                                     if record.venue else ""),
                                                              abstract=record.abstract[:ABSTRACT_CHARS]))
    answer = agent.parse_json_reply(reply)
    if isinstance(answer, list) and answer:
        answer = answer[0]
    return answer


def normalize(answer: dict, key: str, source: str) -> dict:
    """One recorded answer, with the model's fields coerced into shapes the outputs can rely on."""
    def boolean(name):
        value = answer.get(name)
        return value if value in (True, False) else None

    confidence = str(answer.get("confidence") or "").strip().lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
    return {"key": key,
            "no_ai": boolean("no_ai"),
            "not_education": boolean("not_education"),
            "wrong_level": boolean("wrong_level"),
            "confidence": confidence,
            "why": _clean(str(answer.get("why") or "")),
            "source": source}


def verdict_of(answer: dict) -> str:
    """"keep", "drop" or "unknown" for one answer.

    Both halves report *evidence of being off topic*, so either one is enough to drop, and a record with
    neither stays. An unanswered half withholds the verdict rather than deciding it — an unknown is a
    record for a reader to look at, not a record to throw away.
    """
    halves = (answer["no_ai"], answer["not_education"], answer["wrong_level"])
    if True in halves:
        return "drop"
    if None in halves:
        return "unknown"
    return "keep"


def load_state(state_path: pathlib.Path) -> dict[str, dict]:
    """Answers already recorded, keyed by citekey. A later line for a key supersedes an earlier one."""
    done = {}
    if state_path.exists():
        for line in state_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[record["key"]] = record
    return done


def append_state(state_path: pathlib.Path, record: dict) -> None:
    with state_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_pass1(llm_settings: env, todo: list[env], state_path: pathlib.Path, batch_size: int) -> None:
    """Judge every record of `todo` from its title, in batches, appending answers as they arrive."""
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    for n, batch in enumerate(batches, start=1):
        with timer() as tim:
            try:
                answers = judge_titles(llm_settings, batch)
            except Exception as exc:  # noqa: BLE001 -- one bad batch must not end the run
                print(f"  batch {n}/{len(batches)}: FAILED ({type(exc)}: {exc}); will retry on a later run",
                      flush=True)
                continue
        for i, record in enumerate(batch):
            if i in answers:
                append_state(state_path, normalize(answers[i], record.key, source="title"))
        print(f"  batch {n}/{len(batches)}: {len(answers)}/{len(batch)} answered in {tim.dt:.1f}s", flush=True)


def needs_escalation(answer: dict, record: env) -> bool:
    """Whether pass 1's answer for this record should be re-asked with the abstract in hand.

    Three triggers, and only the first is the model's own opinion of itself:

      - it said it was unsure;
      - the title has too little in it for the question to have been answerable, measured from the input,
        so a model confidently wrong about a thin title cannot talk its way out of a second look;
      - it wants to *drop* the record and was less than certain.

    The third is the asymmetry rather than a second guess at the confidence. A false keep costs a reader
    one line; a false drop removes a study from the review and leaves nothing behind to notice it by. So
    the two verdicts do not deserve the same standard of proof, and a drop has to clear a higher one.

    Both hand-checked false drops found during calibration were of exactly this shape: a medium-confidence
    drop of a record whose abstract, once read, put it plainly in scope. Both happened to escalate anyway
    on the thin-title rule, which is luck — the same answers on a normal-length title would have stood.
    """
    return (answer["confidence"] == "low"
            or looks_uninformative(record.title)
            or (verdict_of(answer) == "drop" and answer["confidence"] != "high"))


CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}


def write_review_tsv(rows: list[tuple[env, dict]], path: pathlib.Path) -> None:
    """A hand-checkable table: one row per record, an empty first column to mark disagreements in.

    Sorted so that the four cells of verdict × confidence are contiguous, because the two kinds of error
    are not equally bad — a false drop loses a study silently, a false keep costs a reader one line — and
    they are only countable separately if they can be read separately.
    """
    with path.open("w", encoding="utf-8") as f:
        f.write("mark\tn\tkey\tverdict\tconfidence\tno_ai\tnot_edu\twrong_level\tabstract\tescalates"
                "\twhy\ttitle\tvenue\tabstract_head\n")
        ordered = sorted(rows, key=lambda row: (verdict_of(row[1]),
                                                CONFIDENCE_ORDER[row[1]["confidence"]],
                                                row[0].key))
        for n, (record, answer) in enumerate(ordered, start=1):
            def shown(value):
                return {True: "yes", False: "no", None: "?"}[value]
            preview = record.abstract[:ABSTRACT_PREVIEW_CHARS]
            f.write(f"\t{n}\t{record.key}\t{verdict_of(answer)}\t{answer['confidence']}\t"
                    f"{shown(answer['no_ai'])}\t{shown(answer['not_education'])}\t"
                    f"{shown(answer['wrong_level'])}\t{describe_abstract(record)}\t"
                    f"{'yes' if needs_escalation(answer, record) else 'no'}\t"
                    f"{answer['why']}\t{record.title}\t{record.venue}\t{preview}\n")


def summarize(rows: list[tuple[env, dict]]) -> None:
    """Print the verdict × confidence table, which is what says where the hand-check should look hardest."""
    counts = {}
    for _record, answer in rows:
        counts[(verdict_of(answer), answer["confidence"])] = counts.get((verdict_of(answer), answer["confidence"]), 0) + 1
    print(f"\n{len(rows)} judged")
    for verdict in ("keep", "drop", "unknown"):
        total = sum(n for (v, _c), n in counts.items() if v == verdict)
        if not total:
            continue
        parts = ", ".join(f"{confidence} {counts[(verdict, confidence)]}"
                          for confidence in ("high", "medium", "low")
                          if (verdict, confidence) in counts)
        print(f"  {verdict:<8} {total:>5}   ({parts})")


def write_outputs(records: list[env], done: dict[str, dict],
                  bib_path: pathlib.Path, kept_path: pathlib.Path, dropped_path: pathlib.Path) -> None:
    """The two things this produces: the corpus with the strays taken out, and a list of what went.

    The source `.bib` is not modified. A record with no answer is kept, so that an incomplete run
    under-filters rather than silently losing studies.
    """
    library = bibtex.parse_file(str(bib_path), split_names=False)
    # A record leaves the bibliography for one reason here — the model judged it off topic — and the
    # reason is in `dropped.tsv` beside it. Records that cannot be screened at all, having no abstract,
    # are `raven-siftbib`'s business and are expected to be gone before this runs.
    judged_off_topic = {key for key, answer in done.items() if verdict_of(answer) == "drop"}
    dropped_keys = judged_off_topic

    # Filtered at the *block* level rather than by rebuilding from `library.entries`, so that whatever
    # else the file carries — a preamble, `@string` definitions, comments between records — survives into
    # the filtered copy instead of being quietly dropped along with the strays.
    kept = bibtexparser.Library()
    for block in library.blocks:
        if isinstance(block, Entry) and block.key in dropped_keys:
            continue
        kept.add(block)
    kept_path.write_text(bibtex.write_string(kept), encoding="utf-8")

    by_key = {record.key: record for record in records}
    with dropped_path.open("w", encoding="utf-8") as f:
        f.write("key\tno_ai\tnot_edu\twrong_level\tconfidence\tsource\tabstract\twhy\ttitle\n")
        for key in sorted(judged_off_topic):
            answer = done[key]
            record = by_key.get(key)
            def shown(value):
                return {True: "yes", False: "no", None: "?"}[value]
            f.write(f"{key}\t{shown(answer['no_ai'])}\t{shown(answer['not_education'])}\t"
                    f"{shown(answer['wrong_level'])}\t{answer['confidence']}\t{answer['source']}\t"
                    f"{describe_abstract(record) if record else '?'}\t{answer['why']}\t"
                    f"{record.title if record else ''}\n")

    unknown = sum(1 for answer in done.values() if verdict_of(answer) == "unknown")
    unanswered = len(records) - len(done)
    print(f"\nwrote {kept_path}  ({len(kept.entries)} of {len(library.entries)} records kept)")
    print(f"wrote {dropped_path}  ({len(judged_off_topic)} judged off topic)")
    if unknown or unanswered:
        print(f"  kept anyway: {unknown} judged unknown, {unanswered} never answered")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bib", required=True, help="the .bib to judge")
    parser.add_argument("--pilot", type=int, default=None, metavar="N",
                        help="calibration run: judge a random sample of N records from their titles only, "
                             "and write a TSV to hand-check. Does not write the filtered .bib")
    parser.add_argument("--seed", type=int, default=42, help="which random sample --pilot takes")
    parser.add_argument("--thin", action="store_true",
                        help="calibration run over every record whose title is under the informative bound, "
                             "rather than a random sample. There are few enough of them to read in full, and "
                             "a random sample of any affordable size contains almost none — so the criterion "
                             "they exist to test cannot be tested by drawing one")
    parser.add_argument("--batch", type=int, default=40, help="titles per model call in pass 1")
    parser.add_argument("--no-escalate", action="store_true",
                        help="skip pass 2, leaving pass 1's answers as the verdict")
    parser.add_argument("--backend-url", default=None,
                        help=f"the LLM backend (default: {librarian_config.llm_backend_url})")
    parser.add_argument("--model", default=None, help="model id to judge with (default: whatever the backend offers)")
    parser.add_argument("--out-dir", default=None, help="where the outputs go (default: beside this script)")
    opts = parser.parse_args()

    # `bibtexparser` logs a warning per record it cannot read, which would bury the progress report.
    logging.getLogger("bibtexparser").setLevel(logging.ERROR)

    bib_path = pathlib.Path(opts.bib).expanduser().resolve()
    out_dir = pathlib.Path(opts.out_dir) if opts.out_dir else pathlib.Path(__file__).resolve().parent

    records = load_records(bib_path)
    print(f"{bib_path.name}: {len(records)} records, "
          f"{sum(1 for r in records if r.abstract)} with an abstract, "
          f"{sum(1 for r in records if looks_uninformative(r.title))} with a title under "
          f"{MIN_INFORMATIVE_WORDS} words")

    piloting = opts.thin or opts.pilot is not None
    if opts.thin:
        sample = [record for record in records if looks_uninformative(record.title)]
        run_name = f"pilot-thin-{MIN_INFORMATIVE_WORDS}"
        print(f"calibration pilot: all {len(sample)} records with a title under "
              f"{MIN_INFORMATIVE_WORDS} words, titles only")
    elif opts.pilot is not None:
        sample = random.Random(opts.seed).sample(records, min(opts.pilot, len(records)))
        run_name = f"pilot-{opts.seed}-{len(sample)}"
        print(f"calibration pilot: {len(sample)} records, seed {opts.seed}, titles only")
    else:
        sample = records
        run_name = "judged"
    state_path = out_dir / f"{run_name}.jsonl"

    backend_url = opts.backend_url or librarian_config.llm_backend_url
    llm_settings = llmclient.setup(backend_url=backend_url, quiet=True)
    if opts.model:
        # `request_data["model"]` is the field actually sent; `model_id` is what gets reported.
        llm_settings.request_data["model"] = opts.model
        llm_settings.model_id = opts.model
    print(f"backend: {backend_url}, model: {llm_settings.model_id}")

    done = load_state(state_path)
    todo = [record for record in sample if record.key not in done]
    print(f"already judged: {len(done)};  to do now: {len(todo)}")

    print("pass 1 — titles:")
    run_pass1(llm_settings, todo, state_path, opts.batch)

    done = load_state(state_path)
    by_key = {record.key: record for record in sample}

    if piloting:
        rows = [(by_key[key], answer) for key, answer in done.items() if key in by_key]
        summarize(rows)
        would_escalate = sum(1 for record, answer in rows if needs_escalation(answer, record))
        print(f"  pass 2 would take {would_escalate} of these "
              f"({100 * would_escalate / max(len(rows), 1):.0f}%)")
        tsv_path = out_dir / f"{run_name}.tsv"
        write_review_tsv(rows, tsv_path)
        print(f"\nwrote {tsv_path} — put an x in the first column of every row you disagree with")
        return 0

    if not opts.no_escalate:
        escalate = [record for record in sample
                    if record.key in done
                    and done[record.key]["source"] == "title"
                    and needs_escalation(done[record.key], record)]
        print(f"pass 2 — abstracts, {len(escalate)} records:")
        for n, record in enumerate(escalate, start=1):
            try:
                answer = judge_abstract(llm_settings, record)
            except Exception as exc:  # noqa: BLE001 -- one bad record must not end the run
                print(f"  {n}/{len(escalate)}: {record.key}: FAILED ({type(exc)}: {exc})", flush=True)
                continue
            normalized = normalize(answer, record.key, source="abstract")
            append_state(state_path, normalized)
            print(f"  {n}/{len(escalate)}: {record.title[:60]:<60} -> {verdict_of(normalized)}", flush=True)
        done = load_state(state_path)

    summarize([(by_key[key], answer) for key, answer in done.items() if key in by_key])
    write_outputs(records, done, bib_path,
                  out_dir / f"{bib_path.stem}_in_scope.bib",
                  out_dir / "dropped.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
