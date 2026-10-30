"""Group a review's accepted studies into themes with an LLM, several times over.

The question: can a model, given every accepted study's title and abstract in one prompt, produce the
top level of a grouping like the hand-made cluster tables of the earlier studies in the series — themes, a
reason per study, a study allowed under two themes — how stable is it from run to run, and how does what it
is told about the review (an aim, an extra instruction) change the grouping?

    python investigations/llm-thematic-grouping/group_studies.py STUDIES.bib --aim AIM.txt \\
        --guidance GUIDANCE.txt --runs 10 --declare "reasoning effort xhigh, thinking budget 12k"

Themes only: sub-themes proved unstable from run to run (the bundle's README), and are for people to read
and name. A reply with slips — a study placed nowhere, an item number naming no study, a study with no
primary placement or several, a study placed twice in one theme, or no JSON at all — is answered with a
repair turn naming each slip, up to `MAX_REPAIRS` times; whatever remains after that is fixed up as
`read_grouping` says and recorded with the run.

Each run is one item of a resumable `raven.librarian.sweep`, so a run stopped partway is finished by running
the same command again, and every model call is kept whole in the ledger's traces. The records are shuffled
per run, with the run's number as the seed, so that the order they are presented in cannot pass for a
stable grouping. Each turn streams to `stream.log` as it runs.

What a run writes, into `--out-dir` (default: `llm-grouping/<condition>/` beside the `.bib`, with the private
data, where the condition is the aim file's stem or `no-aim`, followed by `+` and the guidance file's stem
when there is one):
    `grouping-<fingerprint>.jsonl`, and `-traces.jsonl` beside it: the ledger, one answer per run.
    And in `<fingerprint>/`, so that a changed prompt cannot overwrite an earlier one's, per run:
    `run-NN.ods`: the grouping as a table shaped like the published ones, a row per theme.
    `run-NN-placements.ods`: a row per placement, with the model's reason.
    `run-NN.pickle`: a Raven-visualizer dataset, laid out by hand (see `layout`), one point per study.

The aim is given as context and not as the themes: the published tables group studies by what they are
about, and one theme per research question would be a coding of the studies instead. Without `--aim`, the
model is told nothing about the review but that it is one. `--guidance` adds an instruction of the user's
own. The same seeds are used whatever the condition, so a run of one condition shows the records in the same
order as the same-numbered run of another.
"""

import argparse
import hashlib
import json
import logging
import math
import pathlib
import pickle
import random
import sys
from typing import TextIO

from unpythonic.env import env

from raven.common import netutil, tabular
from raven.common import utils as common_utils
from raven.librarian import agent, config as librarian_config, llmclient, sweep
from raven.papers import bibtex

logger = logging.getLogger(__name__)

PROMPT = """You are helping to write a systematic literature review.{aim}

Below are all {n} studies the review accepted, numbered, each with its citation, title, keywords where the record has them, and abstract.

Group the studies into themes by what they are about, the way a review presents its accepted studies as a table of thematic clusters.{aim_note} Choose the themes the material itself suggests; how many is up to you.{guidance}

Place every study at least once. A study that genuinely belongs under two themes may be placed under both, but treat that as the exception. For each placement, give a one-sentence reason why the study belongs there. Mark exactly one placement of each study as its primary one: where it belongs most.

Answer with JSON only, in this shape:

{{"themes": [{{"name": "...", "description": "one sentence", "studies": [{{"item": 1, "primary": true, "reason": "..."}}]}}]}}

The studies:

{studies}
"""

#: Filled into the prompt when an aim is given.
AIM = " The review's aim:\n\n{aim}"
AIM_NOTE = (" The aim is context: it says what the review is looking for, but do not make one theme per part of"
            " the aim.")

#: Filled into the prompt when guidance is given.
GUIDANCE = "\n\n{guidance}"

#: Asked after a reply with slips: the original prompt, the reply, and what was wrong with it. Built on the
#: original prompt so that the backend's prefix cache covers it.
REPAIR = """{prompt}
Your answer was:

{reply}

It has these problems:

{problems}

Answer again with the whole corrected JSON, in the same shape, and nothing else.
"""

#: How many repair turns a run may take before what remains is fixed up and recorded.
MAX_REPAIRS = 2

MAX_TOKENS = 16384


# --------------------------------------------------------------------------------
# The studies

def citation(fields: dict) -> str:
    """'Author (2024)', 'Author & Another (2024)', 'Author et al. (2024)'."""
    authors = common_utils.format_bibtex_authors(fields["author"].value, conjunction="&") if "author" in fields else ""
    year = fields["year"].value if "year" in fields else "n.d."
    return f"{authors or 'Anonymous'} ({year})"


def _text(fields: dict, name: str) -> str:
    if name not in fields or not fields[name].value:
        return ""
    return common_utils.normalize_whitespace(common_utils.unicodize_basic_markup(fields[name].value))


def load_studies(bib_path: pathlib.Path) -> list[env]:
    """Every record of `bib_path`, in file order, as an `env` of `key`, `citation`, `title`, `keywords`, `abstract`."""
    library = bibtex.parse_file(bib_path)
    return [env(key=entry.key, citation=citation(entry.fields_dict),
                title=_text(entry.fields_dict, "title"), keywords=_text(entry.fields_dict, "keywords"),
                abstract=_text(entry.fields_dict, "abstract"))
            for entry in library.entries]


def study_item(i: int, study: env) -> str:
    """Study `study` as the prompt shows it, numbered `i` from 1."""
    lines = [f"[{i}] {study.citation}. {study.title}"]
    if study.keywords:
        lines.append(f"Keywords: {study.keywords}")
    lines.append(f"Abstract: {study.abstract or '(none)'}")
    return "\n".join(lines)


def run_order(n: int, seed: int) -> list[int]:
    """The order of `n` studies in the run with `seed`: a shuffle of `range(n)`."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


def prompt_for(maybe_aim: str | None, maybe_guidance: str | None, studies: list[env], order: list[int]) -> str:
    return PROMPT.format(aim=AIM.format(aim=maybe_aim.strip()) if maybe_aim else "",
                         aim_note=AIM_NOTE if maybe_aim else "",
                         guidance=GUIDANCE.format(guidance=maybe_guidance.strip()) if maybe_guidance else "",
                         n=len(studies),
                         studies="\n\n".join(study_item(i, studies[j]) for i, j in enumerate(order, start=1)))


# --------------------------------------------------------------------------------
# Reading the answer back

def read_grouping(reply: str, studies: list[env], order: list[int]) -> tuple[list[dict], list[str]]:
    """Parse `reply` into themes whose placements name studies by key; and the problems found.

    Returns `(themes, problems)`: `themes` a list of `{"name", "description", "subthemes": [{"name",
    "studies": [{"key", "primary", "reason"}]}]}`. A theme answered without sub-themes, as the prompt now
    asks, gets a single one with no name, so that a grouping has one shape whichever prompt made it. An item
    number that names no study is dropped; a study placed nowhere, or with no primary placement or several,
    is reported. Several primaries keep the first; none makes the first placement primary. Each problem names
    the study by its item number in this run and its citation, being worded for the model as much as for a
    person.

    Raises `ValueError` when the reply is not JSON of the asked shape.
    """
    parsed = agent.parse_json_reply(reply)
    if not (isinstance(parsed, dict) and isinstance(parsed.get("themes"), list)):
        raise ValueError(f"expected an object with a 'themes' list, got {type(parsed).__name__}")
    item_of = {studies[j].key: i for i, j in enumerate(order, start=1)}

    def named(key: str) -> str:
        return f"study [{item_of[key]}] ({next(s.citation for s in studies if s.key == key)})"
    problems = []
    themes = []
    placements_of = {}  # key -> list of placement dicts, in reply order
    for theme in parsed["themes"]:
        if not isinstance(theme, dict):
            raise ValueError(f"expected each theme to be an object, got {type(theme).__name__}")
        subthemes = []
        given = theme["subthemes"] if "subthemes" in theme else [{"name": "", "studies": theme.get("studies", [])}]
        for subtheme in given:
            placed = []
            for study in subtheme.get("studies", []):
                item = study.get("item") if isinstance(study, dict) else None
                if not (isinstance(item, int) and 1 <= item <= len(order)):
                    problems.append(f"theme '{theme.get('name')}': item {item!r} names no study")
                    continue
                key = studies[order[item - 1]].key
                if any(p["key"] == key for p in placed):
                    problems.append(f"theme '{theme.get('name')}': {named(key)} is placed in it twice")
                    continue
                placement = {"key": key, "primary": study.get("primary") is True, "reason": str(study.get("reason", ""))}
                placed.append(placement)
                placements_of.setdefault(key, []).append(placement)
            subthemes.append({"name": str(subtheme.get("name", "")), "studies": placed})
        themes.append({"name": str(theme.get("name", "")), "description": str(theme.get("description", "")),
                       "subthemes": subthemes})
    for study in studies:
        placements = placements_of.get(study.key, [])
        primaries = [p for p in placements if p["primary"]]
        if not placements:
            problems.append(f"{named(study.key)} is placed nowhere")
        elif not primaries:
            problems.append(f"{named(study.key)} has no placement marked primary")
            placements[0]["primary"] = True
        elif len(primaries) > 1:
            problems.append(f"{named(study.key)} has {len(primaries)} placements marked primary; it needs exactly one")
            for p in primaries[1:]:
                p["primary"] = False
    return themes, problems


def ask_with_repairs(settings: env, prompt: str, studies: list[env], order: list[int], *,
                     maybe_abort: netutil.Abort | None, stream: TextIO,
                     label: str) -> tuple[dict | str, list[agent.TurnRecord]]:
    """Ask `prompt`, and while the reply has slips, ask again naming them, up to `MAX_REPAIRS` times.

    `stream`: where each turn streams as it runs, under `label` (and `label, repair N` for the repairs).

    Returns `(outcome, records)`: `records` every turn taken, and `outcome` either a dict of `themes`,
    `problems` (what was left after the last turn, fixed up as `read_grouping` says), `repaired` (the problems
    each repair turn was asked to fix) and `reply` (the last reply) — or, when no usable grouping came back,
    a string saying why.
    """
    records, repaired = [], []
    question = prompt
    while True:
        tag = label if not repaired else f"{label}, repair {len(repaired)}"
        record = agent.ask_record(settings, question, max_tokens=None, maybe_abort=maybe_abort,
                                  on_progress=agent.stream_log(stream, label=tag))
        records.append(record)
        if record.truncated:
            return "cut off at the token cap; the trace holds what there was", records
        try:
            themes, problems = read_grouping(record.reply or "", studies, order)
        except ValueError as exc:
            themes, problems = None, [f"the answer is not JSON of the asked shape: {exc}"]
        if not problems or len(repaired) == MAX_REPAIRS:
            break
        repaired.append(problems)
        question = REPAIR.format(prompt=prompt, reply=(record.reply or "").strip(),
                                 problems="\n".join(f"- {p}" for p in problems))
    if themes is None:
        return f"unreadable after {len(repaired)} repairs: {problems[0]}", records
    return {"themes": themes, "problems": problems, "repaired": repaired, "reply": record.reply}, records


# --------------------------------------------------------------------------------
# Laying it out for the Visualizer

def layout(themes: list[dict], keys: list[str], *, spacing: float = 1.0) -> tuple[list[tuple[float, float]], list[int]]:
    """Made-up 2D coordinates for `keys`, one point per study at its primary placement.

    One circle per theme, its studies around the perimeter in sub-theme order, so a sub-theme is an arc; a
    circle's radius grows with its count, keeping neighbours `spacing` apart. The circles' centres sit on a
    maximin Latin square, as Librarian's attachment piles do, so they read as scattered rather than gridded.

    Returns `(points, labels)`, each in the order of `keys`; a study with no placement gets label -1 at the
    origin.
    """
    from raven.librarian import chatgraph  # the pile layout; private there, borrowed for the prototype

    members = []  # per theme: keys in perimeter order
    for theme in themes:
        members.append([p["key"] for sub in theme["subthemes"] for p in sub["studies"] if p["primary"]])
    radii = [max(spacing, len(m) * spacing / (2 * math.pi)) if len(m) > 1 else 0.0 for m in members]
    step = 2.5 * max(radii + [spacing])
    columns = chatgraph._pile_columns(len(themes), step, step) if themes else ()
    where = {}
    for t, (m, r) in enumerate(zip(members, radii)):
        cx, cy = columns[t] * step, -t * step
        for j, key in enumerate(m):
            angle = math.pi / 2 - 2 * math.pi * j / len(m)
            where[key] = (cx + r * math.cos(angle), cy + r * math.sin(angle), t)
    points = [where[k][:2] if k in where else (0.0, 0.0) for k in keys]
    labels = [where[k][2] if k in where else -1 for k in keys]
    return points, labels


def write_dataset(path: pathlib.Path, bib_path: pathlib.Path, themes: list[dict], keys: list[str]) -> None:
    """A Raven-visualizer dataset of `bib_path`, laid out by `layout`, the theme name and any sub-theme names
    as each cluster's keywords."""
    import numpy as np
    from raven.client import api, config as client_config
    from raven.visualizer import importer

    # The importer's dehyphenator runs on Raven-server when one answers, and locally otherwise; either way
    # the client must be set up first, as the importer's own CLI does. A second call is a no-op.
    api.initialize(raven_server_url=client_config.raven_server_url,
                   maybe_network_retry=client_config.network_retry_batch,
                   raven_api_key_file=client_config.raven_api_key_file)
    input_data = importer._parse_input_files(bib_path)
    vis_data = [entry for entries in input_data.parsed_data_by_filename.values() for entry in entries]
    if len(vis_data) != len(keys):
        raise RuntimeError(f"the importer read {len(vis_data)} records and the probe {len(keys)}; cannot align them")
    points, labels = layout(themes, keys)
    for entry, label in zip(vis_data, labels):
        entry.cluster_id = label
        entry.cluster_probability = 1.0
    name = common_utils.strip_ext(bib_path.name)
    output = {"version": 1,
              "all_input_filenames_raw": input_data.resolved_filenames,
              "all_input_filenames_list": [name],
              "all_input_filenames_str": name,
              "embedding_model": "none (grouped by an LLM)",
              "vis_method": "hand layout of an LLM grouping",
              "n_vis_clusters": len(themes),
              "n_vis_outliers": sum(1 for label in labels if label == -1),
              "labels": np.array(labels),
              "vis_data": vis_data,
              "lowdim_data": np.array(points, dtype=np.float64),
              "keywords_available": True,
              "clusters_keyword_method": "llm grouping",
              "all_keywords": {},
              "vis_keywords_by_cluster": [[theme["name"]] + [sub["name"] for sub in theme["subthemes"] if sub["name"]]
                                          for theme in themes]}
    with path.open("wb") as f:
        pickle.dump(output, f)


# --------------------------------------------------------------------------------
# Tables

def write_tables(out_dir: pathlib.Path, run_id: str, themes: list[dict], studies: list[env], problems: list[str],
                 comments: list[str]) -> None:
    by_key = {s.key: s for s in studies}
    rows = []
    for t, theme in enumerate(themes, start=1):
        keys = {p["key"] for sub in theme["subthemes"] for p in sub["studies"]}
        for sub in theme["subthemes"]:
            rows.append([t, theme["name"], len(keys), sub["name"], len(sub["studies"]),
                         "; ".join(by_key[p["key"]].citation for p in sub["studies"])])
    notes = comments + [f"problem: {p}" for p in problems]
    tabular.write_table(out_dir / f"{run_id}.ods", notes,
                        ["#", "theme", "studies in theme", "sub-theme", "count", "studies"], rows)
    placements = [[t, theme["name"], sub["name"], p["key"], by_key[p["key"]].citation,
                   "yes" if p["primary"] else "", p["reason"]]
                  for t, theme in enumerate(themes, start=1) for sub in theme["subthemes"] for p in sub["studies"]]
    tabular.write_table(out_dir / f"{run_id}-placements.ods", notes,
                        ["#", "theme", "sub-theme", "key", "citation", "primary", "reason"], placements)


def write_outputs(ledger: sweep.Ledger, bib_path: pathlib.Path, studies: list[env], condition: str, *,
                  maybe_run_ids: list[str] | None = None) -> None:
    """Write the tables and the dataset of each answered run in `ledger`, into `<ledger's folder>/<fingerprint>/`.

    `maybe_run_ids`: the runs to write; `None` writes every run the ledger answers.

    Everything comes from the ledger: the groupings from its answers, the model and the declarations from its
    latest header. `studies` (from `bib_path`) only names the studies, so it must hold every key the ledger
    places; `ValueError` otherwise.
    """
    answers = ledger.answers()
    run_ids = sorted(answers) if maybe_run_ids is None else [r for r in maybe_run_ids if r in answers]
    keys = [s.key for s in studies]
    missing = sorted({p["key"] for r in run_ids for t in answers[r]["themes"] for sub in t["subthemes"]
                      for p in sub["studies"]} - set(keys))
    if missing:
        raise ValueError(f"{bib_path.name} lacks {len(missing)} of the keys the ledger places, e.g. {missing[0]}")
    header = ledger.headers()[-1] if ledger.headers() else {}
    model_line = (f"model: {header.get('label', 'not recorded')}; "
                  f"declared: {header.get('declared', 'not recorded')}")
    run_dir = ledger.directory / ledger.fingerprint
    run_dir.mkdir(parents=True, exist_ok=True)
    for run_id in run_ids:
        answer = answers[run_id]
        comments = [f"{run_id} of an LLM grouping of {bib_path.name}, condition: {condition}, "
                    f"instrument {ledger.fingerprint}, seed {answer['seed']}", model_line]
        write_tables(run_dir, run_id, answer["themes"], studies, answer["problems"], comments)
        write_dataset(run_dir / f"{run_id}.pickle", bib_path, answer["themes"], keys)
        n_double = sum(1 for k in keys
                       if sum(1 for t in answer["themes"] for s in t["subthemes"] for p in s["studies"] if p["key"] == k) > 1)
        print(f"{run_id}: {len(answer['themes'])} themes, "
              f"{n_double} placed twice or more, {len(answer.get('repaired', []))} repair turns, "
              f"{len(answer['problems'])} problems left")


# --------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bib", help="the accepted studies")
    parser.add_argument("--aim", default=None, metavar="FILE",
                        help="a text file saying what the review is for, e.g. its research questions; without it, "
                             "the model is told nothing about the review")
    parser.add_argument("--guidance", default=None, metavar="FILE",
                        help="a text file holding an instruction of your own, added after the one to group by theme")
    parser.add_argument("--runs", type=int, default=5, help="how many runs (default: 5)")
    parser.add_argument("--out-dir", default=None,
                        help="where the outputs go (default: llm-grouping/<condition>/ beside the .bib)")
    parser.add_argument("--backend-url", default=None,
                        help=f"the LLM backend (default: {librarian_config.llm_backend_url})")
    parser.add_argument("--declare", default=None, metavar="TEXT",
                        help="backend settings the backend does not report, e.g. 'reasoning effort xhigh'")
    parser.add_argument("--rebuild", default=None, metavar="FINGERPRINT",
                        help="write the tables and datasets of the ledger with this fingerprint from its stored answers, "
                             "asking no model: after the output format or the .bib changed, or with no backend at hand. "
                             "The .bib only names the studies, so give the one the runs saw if their citations matter")
    opts = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bib_path = pathlib.Path(opts.bib).resolve()
    maybe_aim = pathlib.Path(opts.aim).read_text(encoding="utf-8") if opts.aim else None
    maybe_guidance = pathlib.Path(opts.guidance).read_text(encoding="utf-8") if opts.guidance else None
    condition = ((pathlib.Path(opts.aim).stem if opts.aim else "no-aim") +
                 (f"+{pathlib.Path(opts.guidance).stem}" if opts.guidance else ""))
    out_dir = pathlib.Path(opts.out_dir) if opts.out_dir else bib_path.parent / "llm-grouping" / condition
    studies = load_studies(bib_path)
    keys = [s.key for s in studies]
    print(f"{len(studies)} studies from {bib_path.name}")

    if opts.rebuild:
        ledger = sweep.Ledger(out_dir, "grouping", opts.rebuild)
        if not ledger.path.exists():
            print(f"no ledger at {ledger.path}")
            return 1
        write_outputs(ledger, bib_path, studies, condition)
        return 0

    llm_settings = llmclient.setup(backend_url=opts.backend_url or librarian_config.llm_backend_url, quiet=True)
    llm_settings.request_data["max_tokens"] = MAX_TOKENS
    material = json.dumps({"prompt": PROMPT, "aim": maybe_aim, "guidance": maybe_guidance,
                           "repair": REPAIR, "max_repairs": MAX_REPAIRS, "max_tokens": MAX_TOKENS,
                           "studies": hashlib.sha256(bib_path.read_bytes()).hexdigest(),
                           "sampler": {k: llm_settings.request_data.get(k) for k in ("temperature", "min_p", "top_p", "top_k")}},
                          sort_keys=True)
    ledger = sweep.Ledger(out_dir, "grouping", sweep.fingerprint(llm_settings, material, declared=opts.declare))
    print(f"instrument {ledger.fingerprint} -> {ledger.path}")
    stream_path = out_dir / "stream.log"
    out_dir.mkdir(parents=True, exist_ok=True)

    def ask_batch(settings: env, batch: list[tuple[str, int]],
                  maybe_abort: netutil.Abort | None) -> tuple[dict, list[agent.TurnRecord]]:
        (run_id, seed), = batch
        order = run_order(len(studies), seed)
        prompt = prompt_for(maybe_aim, maybe_guidance, studies, order)
        with stream_path.open("a", encoding="utf-8") as stream:
            outcome, records = ask_with_repairs(settings, prompt, studies, order, maybe_abort=maybe_abort,
                                                stream=stream, label=run_id)
        if isinstance(outcome, str):
            return {run_id: sweep.Failure(outcome)}, records
        return {run_id: {"seed": seed, "order": [keys[j] for j in order], **outcome}}, records

    items = [(f"run-{i:02d}", i) for i in range(1, opts.runs + 1)]
    result = sweep.Sweep(llm_settings, ask_batch, items, ledger,
                         header=sweep.instrument_header(llm_settings, declared=opts.declare)).run()

    write_outputs(ledger, bib_path, studies, condition, maybe_run_ids=[run_id for run_id, _ in items])
    if result.outcome is not sweep.completed:
        print(f"\nstopped ({result.outcome}): {result.message or 'cancelled'}\nrun again to resume")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
