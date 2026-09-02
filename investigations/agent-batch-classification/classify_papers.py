"""Classify a pile of papers by field of science, to find the ones that do not belong in an AI pile.

Titles carry almost all of the signal — 98% of the filenames in the target pile are
"Authors YEAR - Title[ - id][ - note].ext" — so the model is asked about titles in batches, and only the
entries whose filename says too little are escalated to reading the document's first page. Nothing here
assumes arXiv: the id, where there is one, is just more filename.

Resumable. Every answer is appended to a JSONL as it arrives, and a re-run skips what is already there, so a
backend hiccup two-thirds of the way through costs the current batch rather than the run.

Writes a TSV of the classification, and a mover script that is a dry run until given `--commit`.

    python classify_papers.py --dir ~/papers
"""

import argparse
import json
import pathlib
import re
import shlex
import sys

from unpythonic import timer
from unpythonic.env import env

from raven.common import docextract
from raven.librarian import agent, config as librarian_config, llmclient

# Entries worth asking about. `.bib` is a bibliography rather than a paper, and a saved page's `_files`
# directory is assets; both are reported and skipped rather than silently ignored.
CLASSIFIABLE_SUFFIXES = {".pdf", ".txt", ".html", ".htm", ".md", ".djvu", ".ps", ".epub"}

# Offered to the model so that field labels can become directory names without a second consolidation pass.
# An unlisted field is allowed — the instruction says to coin a slug in the same style — because a pile
# assembled over years will contain something this list does not anticipate.
FIELDS = [
    "artificial-intelligence", "computer-science", "mathematics", "numerical-analysis", "statistics",
    "physics", "fluid-dynamics", "solid-mechanics", "materials-science", "chemistry", "biology",
    "neuroscience", "medicine", "psychology", "cognitive-science", "economics", "policy-and-governance",
    "philosophy", "history", "linguistics", "education", "engineering", "other",
]

INSTRUCTIONS = """\
You are sorting a researcher's paper collection. The collection is supposed to be about artificial \
intelligence; the task is to find the papers that are about something else.

For each numbered item below you are given a filename, which normally contains the authors, the year and \
the paper's title. Judge from the title.

For each item, answer:
  "i"          the item's number, copied exactly
  "about_ai"   true if the work is about artificial intelligence or machine learning IN ANY WAY - including \
the economics of AI, AI policy and governance, AI safety and alignment, philosophy of AI, and applications \
of machine learning to another field. False only if AI plays no part in it.
  "field"      the field of science the work belongs to, as a lowercase-hyphenated slug, preferably one of: \
{fields}. If none fits, coin one in the same style.
  "confidence" "high", "medium" or "low" - how sure you are, given only the filename
  "why"        at most twelve words, why you chose that field

Use "low" confidence when the filename does not say enough to judge. Do not guess confidently.

IMPORTANT: judge ONLY from words that describe the work. An identifier is not a description. If a filename \
is just a number, a code, an arXiv id, a date or a page number, you do not know what the paper is about - \
answer "low" confidence and say the filename carries no topic. Do NOT claim to recognize which paper an \
arXiv id refers to; you cannot, and a guess dressed as recognition is worse than admitting the filename is \
uninformative.

Answer with a JSON array of objects and nothing else. One object per item, in order, no commentary, no \
markdown fences.

Items:
{items}
"""

ESCALATION_INSTRUCTIONS = """\
You are sorting a researcher's paper collection, which is supposed to be about artificial intelligence.

The filename of this document says too little to classify it, so here is its beginning instead.

Filename: {name}

--- first page ---
{excerpt}
--- end ---

Answer with a single JSON object and nothing else:
  "about_ai"   true if the work is about artificial intelligence or machine learning in any way - including \
the economics of AI, AI policy, AI safety, philosophy of AI, and applications of machine learning to another \
field. False only if AI plays no part in it.
  "field"      the field of science, as a lowercase-hyphenated slug, preferably one of: {fields}
  "confidence" "high", "medium" or "low"
  "why"        at most twelve words
"""

EXCERPT_CHARS = 3000


def looks_uninformative(stem: str) -> bool:
    """Whether a filename stem is too thin to classify from — the escalation trigger.

    A name earns its keep by containing prose. The pile's convention is "Authors YEAR - Title", so a stem
    with a separator and several words is informative; a bare identifier, a page number, or a couple of
    tokens is not. Deliberately generous: escalation costs one document read, and a wrong "informative"
    verdict costs a wrong answer.
    """
    words = [w for w in re.split(r"[\s_\-]+", stem) if any(c.isalpha() for c in w)]
    if len(words) >= 5:
        return False
    return " - " not in stem


def collect_entries(directory: pathlib.Path) -> tuple[list[pathlib.Path], dict[str, int]]:
    """The files to classify, and a census of what was skipped and why."""
    entries = []
    skipped = {}
    for path in sorted(directory.iterdir()):
        if path.is_dir():
            skipped["directories"] = skipped.get("directories", 0) + 1
        elif path.suffix.lower() in CLASSIFIABLE_SUFFIXES:
            entries.append(path)
        else:
            key = f"other ({path.suffix.lower() or 'no suffix'})"
            skipped[key] = skipped.get(key, 0) + 1
    return entries, skipped


def classify_batch(llm_settings: env, batch: list[pathlib.Path]) -> dict[int, dict]:
    """Classify a batch of filenames. Returns {index within batch: answer}; missing entries are the caller's problem."""
    items = "\n".join(f"{i}. {path.name}" for i, path in enumerate(batch))
    reply = agent.ask(llm_settings, INSTRUCTIONS.format(fields=", ".join(FIELDS), items=items))
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


def escalate(llm_settings: env, path: pathlib.Path) -> dict:
    """Classify one entry by reading the start of the document itself."""
    try:
        text = docextract.extract_text(str(path)) or ""
    except Exception as exc:  # noqa: BLE001 -- an unreadable file is a result, not a crash
        return {"about_ai": None, "field": "unknown", "confidence": "low",
                "why": f"could not read: {type(exc).__name__}"}
    if not text.strip():
        return {"about_ai": None, "field": "unknown", "confidence": "low",
                "why": "no extractable text"}
    reply = agent.ask(llm_settings, ESCALATION_INSTRUCTIONS.format(name=path.name,
                                                             excerpt=text[:EXCERPT_CHARS],
                                                             fields=", ".join(FIELDS)))
    answer = agent.parse_json_reply(reply)
    if isinstance(answer, list) and answer:
        answer = answer[0]
    return answer


def load_state(state_path: pathlib.Path) -> dict[str, dict]:
    """Answers already recorded, keyed by filename."""
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
            done[record["name"]] = record
    return done


def append_state(state_path: pathlib.Path, record: dict) -> None:
    with state_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize(answer: dict, name: str, source: str) -> dict:
    """One recorded answer, with the model's fields coerced into shapes the outputs can rely on."""
    field = str(answer.get("field") or "unknown").strip().lower()
    field = re.sub(r"[^a-z0-9]+", "-", field).strip("-") or "unknown"
    confidence = str(answer.get("confidence") or "").strip().lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
    about_ai = answer.get("about_ai")
    if about_ai not in (True, False):
        about_ai = None
    return {"name": name,
            "about_ai": about_ai,
            "field": field,
            "confidence": confidence,
            "why": str(answer.get("why") or "").strip(),
            "source": source}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="the directory of papers to classify")
    parser.add_argument("--out", default=None, help="TSV output path (default: <dir>/../papers-classified.tsv)")
    parser.add_argument("--state", default=None, help="resumable JSONL of answers")
    parser.add_argument("--script", default=None, help="path for the generated mover script")
    parser.add_argument("--batch", type=int, default=40, help="filenames per model call")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many new classifications (for a trial run)")
    parser.add_argument("--no-escalate", action="store_true", help="skip reading documents whose filename says too little")
    opts = parser.parse_args()

    directory = pathlib.Path(opts.dir).expanduser().resolve()
    here = pathlib.Path(__file__).resolve().parent
    state_path = pathlib.Path(opts.state) if opts.state else here / "papers-classified.jsonl"
    tsv_path = pathlib.Path(opts.out) if opts.out else here / "papers-classified.tsv"
    script_path = pathlib.Path(opts.script) if opts.script else here / "move-stragglers.sh"

    entries, skipped = collect_entries(directory)
    print(f"{directory}: {len(entries)} classifiable entries")
    for what, count in sorted(skipped.items()):
        print(f"  skipped {count} {what}")

    done = load_state(state_path)
    print(f"already classified: {len(done)}")
    todo = [path for path in entries if path.name not in done]
    if opts.limit is not None:
        todo = todo[:opts.limit]
    print(f"to do now: {len(todo)}")

    llm_settings = llmclient.setup(backend_url=librarian_config.llm_backend_url, quiet=True)
    print(f"backend: {librarian_config.llm_backend_url}, model: {llm_settings.model_id}")

    # Pass 1 - titles, in batches.
    batches = [todo[i:i + opts.batch] for i in range(0, len(todo), opts.batch)]
    for n, batch in enumerate(batches, start=1):
        with timer() as tim:
            try:
                answers = classify_batch(llm_settings, batch)
            except Exception as exc:  # noqa: BLE001 -- one bad batch must not end the run
                print(f"  batch {n}/{len(batches)}: FAILED ({type(exc).__name__}: {exc}); will retry on a later run")
                continue
        for i, path in enumerate(batch):
            if i in answers:
                append_state(state_path, normalize(answers[i], path.name, source="title"))
        print(f"  batch {n}/{len(batches)}: {len(answers)}/{len(batch)} answered in {tim.dt:.1f}s", flush=True)

    # Pass 2 - the ones the title could not carry.
    if not opts.no_escalate:
        done = load_state(state_path)
        needs_reading = [path for path in entries
                         if path.name in done
                         and done[path.name]["source"] == "title"
                         and (done[path.name]["confidence"] == "low" or looks_uninformative(path.stem))]
        print(f"escalating {len(needs_reading)} entries to reading the document")
        for n, path in enumerate(needs_reading, start=1):
            try:
                answer = escalate(llm_settings, path)
            except Exception as exc:  # noqa: BLE001
                print(f"  {n}/{len(needs_reading)}: {path.name}: FAILED ({type(exc).__name__}: {exc})")
                continue
            append_state(state_path, normalize(answer, path.name, source="fulltext"))
            print(f"  {n}/{len(needs_reading)}: {path.name[:70]} -> {answer.get('field')}", flush=True)

    write_outputs(load_state(state_path), directory, tsv_path, script_path)
    return 0


def write_outputs(done: dict[str, dict], directory: pathlib.Path,
                  tsv_path: pathlib.Path, script_path: pathlib.Path) -> None:
    """Write the classification table, and the mover script derived from it."""
    rows = sorted(done.values(), key=lambda r: (r["about_ai"] is not False, r["field"], r["name"]))
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("name\tabout_ai\tfield\tconfidence\tsource\twhy\n")
        for r in rows:
            about = {True: "yes", False: "no", None: "unknown"}[r["about_ai"]]
            f.write(f"{r['name']}\t{about}\t{r['field']}\t{r['confidence']}\t{r['source']}\t{r['why']}\n")

    stragglers = [r for r in rows if r["about_ai"] is False]
    unknown = [r for r in rows if r["about_ai"] is None]
    lines = ["#!/bin/bash",
             "# Move the papers that are not about AI out of the main pile, filing them by field.",
             "#",
             "# Generated by classify_papers.py. A DRY RUN unless given --commit:",
             "#     ./move-stragglers.sh            # print what would happen",
             "#     ./move-stragglers.sh --commit   # do it",
             "#",
             "# Entries classified 'unknown' are NOT moved; they are listed at the end for you to look at.",
             "set -euo pipefail",
             "",
             'COMMIT=0; [ "${1:-}" = "--commit" ] && COMMIT=1',
             f"SRC={shlex.quote(str(directory))}",
             f"DEST={shlex.quote(str(directory / 'by-field'))}",
             "",
             "move() {  # $1 = filename, $2 = field",
             '  local from="$SRC/$1" to="$DEST/$2/$1"',
             '  if [ ! -e "$from" ]; then echo "MISSING  $1"; return; fi',
             '  if [ "$COMMIT" = 1 ]; then mkdir -p "$DEST/$2"; mv -n -- "$from" "$to"; echo "moved    $2/$1";',
             '  else echo "would move  $2/$1"; fi',
             "}",
             ""]
    for r in stragglers:
        lines.append(f"move {shlex.quote(r['name'])} {shlex.quote(r['field'])}")
    lines += ["",
              f'echo "{len(stragglers)} straggler(s); $( [ "$COMMIT" = 1 ] && echo moved || echo "dry run, nothing moved" )"',
              ""]
    if unknown:
        lines.append('echo; echo "Not classified - look at these yourself:"')
        for r in unknown:
            lines.append(f"echo '  {r['name'].replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'")
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)

    by_field = {}
    for r in stragglers:
        by_field[r["field"]] = by_field.get(r["field"], 0) + 1
    print(f"\nwrote {tsv_path}  ({len(rows)} rows)")
    print(f"wrote {script_path}  ({len(stragglers)} stragglers, {len(unknown)} unknown)")
    for field, count in sorted(by_field.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>4}  {field}")


if __name__ == "__main__":
    sys.exit(main())
