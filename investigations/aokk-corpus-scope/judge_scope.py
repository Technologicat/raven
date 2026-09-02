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

Escalation fires on a low-confidence answer, on a title too thin to have been answerable, or on any drop
the model was not certain of. The last two are decided here rather than asked of the model, and that is
the point: a sibling run over a paper pile found the model at its most confident exactly where the input
carried least — confidently naming the subject of a file called `2006.05563.pdf` — so a rule driven by
the model's own confidence is blind in precisely the place it most needs to look. And the two verdicts do
not deserve the same standard of proof, a false keep costing a reader one line where a false drop removes
a study and leaves nothing behind to notice it by.

Resumable. Every answer is appended to a JSONL as it arrives and a re-run skips what is already there, so
a backend hiccup two-thirds of the way through costs the current batch rather than the run.

Reads a `.bib` rather than the Visualizer dataset built from it: the citekey is a key the outputs can be
written against, and the filtered `.bib` is one of the two things this produces.

    python judge_scope.py --bib .../tekoalyagentti_tutkimus_deduped.bib --pilot 200
    python judge_scope.py --bib .../tekoalyagentti_tutkimus_deduped.bib
"""

import argparse
import functools
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

from raven.common import nlptools
from raven.librarian import agent, config as librarian_config, llmclient
from raven.papers import bibtex
from raven.visualizer import config as visualizer_config

# The question the corpus is supposed to answer, quoted into both prompts so the two passes judge the
# same thing. `00_stuff/rawdata/AOKK/search-phrase.txt` is the boolean query this paraphrases; the
# research questions narrow it further and are deliberately *not* asked here, being a separate pass over
# a corpus that has already had its obvious strays taken out.
SCOPE_QUESTION = "studies on different aspects of the use of AI agents in higher education"

# When a title is too thin to judge from. Two tests, because a title can fail in two unrelated ways and
# no single number catches both.
#
# **A count catches the short and empty.** "Rethinking the paradigm" names nothing, and neither does a
# heading or a bare proper noun.
#
# **A density catches the long and padded**, which a count cannot: academic prose pads indefinitely, so
# "Tackling the possible advantages and potential challenges of a technical understanding in the present
# context" runs to fifteen words and three content words. Raising the count bound until it caught that
# was the wrong repair — measured across five corpora it flagged a quarter of one of them, because a
# *short and precise* title scores the same three or four. "Distilling the Knowledge in a Neural
# Network", "Unfitted finite element methods" and "On the instability of an axially moving elastic
# plate" all score 4, and all say exactly what they are about. Count does not separate the populations;
# the share of the title that is load-bearing does.
#
# Calibrated on five corpora in unrelated domains — education and AI, computational mechanics, applied
# mechanics, arXiv AI, hydrogen production — where these bounds flag 0.7% to 4.9%, and on a set of
# hand-written filler and genuine titles that it separates completely. Note this is *not* a classifier
# for vagueness and cannot be one: it is the backstop for the single case the model's own confidence
# cannot cover, which is being sure about an input that says nothing.
MIN_CONTENT_WORDS = 3
MIN_CONTENT_DENSITY = 0.40

# Below this many words, density says nothing — one function word in a four-word title swings it by a
# quarter — so a short title is judged on the count alone.
MIN_WORDS_FOR_DENSITY = 8

# Function words plus the project's own hand-tuned academic-prose list, which is the half that matters
# here: a title reading "Exploring the Potential of..." is padding a reader recognizes instantly and a
# word count does not. Reusing the Visualizer's list rather than writing a second one keeps "content
# word" meaning the same thing in the screen as it does in the keyword extraction downstream.
STOPWORDS = frozenset(nlptools.default_stopwords) | frozenset(visualizer_config.custom_stopwords)

# Small and CPU-friendly, which is what this needs: a few thousand short titles, lemmas only, no GPU.
SPACY_MODEL = "en_core_web_sm"

# How much of an abstract pass 2 sends. Comfortably more than an abstract normally runs to; the cap is
# against the occasional record whose "abstract" field holds a whole introduction.
ABSTRACT_CHARS = 4000

# How much of an abstract the pilot TSV shows, so that a title too thin to judge from can still be
# adjudicated by hand without opening the .bib.
ABSTRACT_PREVIEW_CHARS = 300

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

PASS2_INSTRUCTIONS = """\
You are screening a literature-search result for records that are not about the topic it was searching for.

{rubric}

The titles of the records below said too little to judge them from, so each is given with its abstract.
Judge each record only on its own text; they are unrelated to one another and their order means nothing.

An abstract marked TRUNCATED is a publisher's preview that breaks off mid-sentence. Judge it only on what \
is actually there - whatever it has not reached is not evidence of anything, so lower your confidence \
rather than concluding from what is missing.

For each item, answer:
  "i"              the item's number, copied exactly
  "no_ai"          true or false, as above
  "not_education"  true or false, as above
  "wrong_level"    true or false, as above
  "confidence"     "high", "medium" or "low"
  "why"            at most twelve words
  "truncated"      true if the abstract stops in the middle - it ends mid-sentence, or breaks off before \
the work has said what it did. Judge the text, not its length: a short abstract that finishes its point is \
NOT truncated. Answering this honestly costs nothing, a truncated abstract being a record that tells you \
less rather than a worse one.

Answer with a JSON array of objects and nothing else. One object per item, in order, no commentary, no \
markdown fences.

{items}
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


@functools.lru_cache(maxsize=None)
def _nlp():
    """The spaCy pipeline, loaded once on first use.

    On the CPU deliberately. It is a small model doing a few thousand short titles, and the GPU may be
    busy serving the very backend this script is talking to — a single card for everything is an ordinary
    way to run Raven rather than a degraded one.
    """
    return nlptools.load_spacy_pipeline(SPACY_MODEL, "cpu")


@functools.lru_cache(maxsize=None)
def content_words(title: str) -> int:
    """How many words of the title are neither function words nor academic filler.

    Counted on *lemmas*, which is the part that cannot be skipped: the stopword lists enumerate base
    forms, so a surface-form match lets any inflection through. "Tackling possible advantage of technical
    understanding" scores zero and the same sentence pluralized scores three, which is a gap ordinary
    English steps into constantly rather than one an adversary has to look for.

    Lemmatizing also makes this agree with the Visualizer's keyword extraction, which lemmatizes before
    applying the same lists — the reason for borrowing them rather than writing a second set was that
    "content word" should mean one thing across the pipeline, and a surface-form count quietly broke that.

    Both the lemma and the surface form are checked, since a list holding an irregular form directly is
    still a hit.
    """
    return len([token for token in _nlp()(title)
                if token.is_alpha
                and token.lemma_.lower() not in STOPWORDS
                and token.text.lower() not in STOPWORDS])


def informative_words(title: str) -> int:
    """How many words of the title carry letters — digits and punctuation say nothing about a subject."""
    return len([word for word in re.split(r"[\s\-_/]+", title) if any(c.isalpha() for c in word)])


def looks_uninformative(title: str) -> bool:
    """Whether a title is too thin to judge from — the trigger of the escalation rule the model cannot veto."""
    content = content_words(title)
    if content < MIN_CONTENT_WORDS:
        return True
    total = informative_words(title)
    return total >= MIN_WORDS_FOR_DENSITY and content / total < MIN_CONTENT_DENSITY


def looks_truncated(abstract: str) -> bool:
    """Whether an abstract visibly breaks off — a publisher's teaser rather than the whole thing.

    Keyed on an ellipsis **at the end**, which is the only place one settles anything. Mid-text an
    ellipsis is ordinary rhetoric — an elided quotation, a trailing "and so on" — so its presence there
    says nothing about whether the text is complete. About a tenth of the abstracts here end in one,
    breaking off as "Given this ..." or "with an increased ...".

    Length is deliberately not part of it. Some abstracts are simply short, and measured on this corpus
    the bound never fired on its own anyway: no ellipsis-ended abstract ran past it, so it could only ever
    have excluded a genuine teaser for being long.

    **It cannot see a publisher who truncates silently**, and no text-level rule can. "Ends without
    terminal punctuation" was tried as a second signal and rejected on the data — it selects records that
    are complete, ending in a URL, a DOI or a keyword list, with a median length *longer* than the
    corpus's. That gap is what the judge's own `truncated` answer is for; this half is the one
    that needs no model and cannot be argued with.
    """
    return bool(abstract) and bool(re.search(r"(\.\.\.|\u2026)\s*$", abstract.strip()))


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
    reply = agent.ask(llm_settings,
                      PASS1_INSTRUCTIONS.format(rubric=_SCOPE_RUBRIC.format(question=SCOPE_QUESTION),
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


def _format_for_pass2(index: int, record: env) -> str:
    """One record as pass 2 sees it: its number, title, venue, and abstract, with a truncation marker."""
    marked = " (TRUNCATED)" if looks_truncated(record.abstract) else ""
    venue = f"Published in: {record.venue}\n" if record.venue else ""
    return (f"--- item {index} ---\n"
            f"Title: {record.title}\n"
            f"{venue}"
            f"Abstract{marked}: {record.abstract[:ABSTRACT_CHARS]}\n")


def judge_abstracts(llm_settings: env, batch: list[env]) -> dict[int, dict]:
    """Pass 2 over a batch of records, reading each one's abstract as well as its title.

    Returns `{position within batch: answer}`, index-keyed and short-tolerant exactly as `judge_titles`
    is — an answer whose number does not resolve is dropped rather than guessed at, and a record left
    unanswered stays in the next run's to-do list.

    Batched because pass 2 is the expensive half: one call per record put it at several hours against
    pass 1's one, for a fifth of the corpus. Abstracts here average well under two thousand characters,
    so a batch of ten is a few thousand tokens of prompt — the same shape pass 1 already runs in.
    """
    items = "\n".join(_format_for_pass2(i, record) for i, record in enumerate(batch))
    reply = agent.ask(llm_settings,
                      PASS2_INSTRUCTIONS.format(rubric=_SCOPE_RUBRIC.format(question=SCOPE_QUESTION),
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


def temper_for_truncation(answer: dict, record: env) -> dict:
    """Withhold a *drop* that rests on a publisher's teaser, and never let one read as certain.

    A truncated abstract breaks off before it reaches the methods, so an absence in it is not evidence
    of anything — which is what the rubric says at length and what the prompt marks each such item with.
    Measured: the model reads the marker and concludes from the absence anyway, at *high* confidence, on
    a 511-character preview that never reaches the method it is asserting about. So this is decided here
    instead, where the model cannot argue with it — the same reason the thin-title trigger is computed
    rather than asked.

    Deterministic *given* the detection, which is itself a heuristic — see `looks_truncated`, which can
    only see a publisher who marks the cut. This tempers what it catches and cannot temper what it
    misses.

    A drop becomes an unknown, which is kept and flagged for a reader rather than removed, and the
    confidence drops to "low" whatever the model claimed. The `why` keeps what the model said, prefixed
    so the record explains itself. Positive evidence is not rescued from this: a teaser naming a medical
    congress is still naming one, and a reader looking at the reason can see which kind it is.

    Rarely reached in the real pipeline — `raven-siftbib --require abstract --min-chars abstract=600`
    removes teasers before the judge sees them — which is exactly why it belongs here as well. A guard
    that only holds when an upstream flag was remembered is not a guard.
    """
    # Either signal is enough. The visible one cannot see a silent cut and the model's cannot be trusted
    # to *act*, but it can be trusted to *look* — which is the division this rests on.
    if not (looks_truncated(record.abstract) or answer.get("truncated") is True):
        return answer
    tempered = dict(answer)
    tempered["confidence"] = "low"
    if verdict_of(normalize(answer, record.key, "abstract")) == "drop":
        for name in ("no_ai", "not_education", "wrong_level"):
            if tempered.get(name) is True:
                tempered[name] = None
        tempered["why"] = f"withheld, abstract is a truncated preview: {answer.get('why', '')}".strip()
    return tempered


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
        # Least-defended first, because this list is read top-down and never to the end. A run over a
        # corpus this size drops hundreds of records, so the order decides which of them a reader
        # actually sees: the shakiest verdicts, not the alphabetically luckiest.
        #
        # Two keys, in this order. Confidence, obviously. Then whether the record was judged from its
        # abstract or only from its title — a drop that pass 2 never re-examined rests on less evidence
        # than one that did, whatever the model said about its own certainty.
        def least_defended(key):
            answer = done[key]
            return (CONFIDENCE_ORDER[answer["confidence"]],
                    0 if answer["source"] == "title" else 1,
                    key)

        for key in sorted(judged_off_topic, key=least_defended):
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
    parser.add_argument("--batch2", type=int, default=10, metavar="N",
                        help="records per model call in pass 2. Smaller than pass 1's, each item carrying "
                             "a whole abstract rather than a title")
    parser.add_argument("--no-escalate", action="store_true",
                        help="skip pass 2, leaving pass 1's answers as the verdict")
    parser.add_argument("--backend-url", default=None,
                        help=f"the LLM backend (default: {librarian_config.llm_backend_url})")
    parser.add_argument("--model", default=None, help="model id to judge with (default: whatever the backend offers)")
    parser.add_argument("--max-reply-tokens", type=int, default=agent.DEFAULT_MAX_REPLY_TOKENS, metavar="N",
                        help="cap on one reply. Raven's default is the whole context window, which for an "
                             "unattended batch is no cap at all: a model that falls into a repetition "
                             "loop generates until it fills 128k, taking half an hour to produce a reply "
                             "that cannot parse. Measured: a batch of forty needs about 10.5k output "
                             "tokens, most of it thinking, so the default has room and still fails a "
                             "runaway several times sooner")
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
          f"{MIN_CONTENT_WORDS} content words or too padded to judge")

    piloting = opts.thin or opts.pilot is not None
    if opts.thin:
        sample = [record for record in records if looks_uninformative(record.title)]
        run_name = "pilot-thin"
        print(f"calibration pilot: all {len(sample)} records whose title is too thin "
              f"to judge from, titles only")
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
    llm_settings.request_data["max_tokens"] = opts.max_reply_tokens
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
        # A record with nothing to read is answered here rather than sent: an empty abstract asks the
        # model a question it has no material for, and the verdict is already known — unknown, which is
        # kept. Sending them would also dilute a batch with items carrying no text.
        unreadable = [record for record in escalate if not record.abstract]
        readable = [record for record in escalate if record.abstract]
        for record in unreadable:
            append_state(state_path, normalize({"no_ai": None, "not_education": None, "wrong_level": None,
                                                "confidence": "low", "why": "no abstract to read"},
                                               record.key, source="abstract"))

        batches = [readable[i:i + opts.batch2] for i in range(0, len(readable), opts.batch2)]
        print(f"pass 2 — abstracts, {len(readable)} records in {len(batches)} batches "
              f"({len(unreadable)} had no abstract and were answered without asking):")
        for n, batch in enumerate(batches, start=1):
            with timer() as tim:
                try:
                    answers = judge_abstracts(llm_settings, batch)
                except Exception as exc:  # noqa: BLE001 -- one bad batch must not end the run
                    print(f"  batch {n}/{len(batches)}: FAILED ({type(exc)}: {exc}); "
                          f"will retry on a later run", flush=True)
                    continue
            for i, record in enumerate(batch):
                if i in answers:
                    answer = temper_for_truncation(answers[i], record)
                    append_state(state_path, normalize(answer, record.key, source="abstract"))
            print(f"  batch {n}/{len(batches)}: {len(answers)}/{len(batch)} answered in {tim.dt:.1f}s",
                  flush=True)
        done = load_state(state_path)

    summarize([(by_key[key], answer) for key, answer in done.items() if key in by_key])
    write_outputs(records, done, bib_path,
                  out_dir / f"{bib_path.stem}_in_scope.bib",
                  out_dir / "dropped.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
