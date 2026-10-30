"""Group a review's accepted studies into themes and sub-themes with an LLM, several times over.

The question: can a model, given a review's research questions and every accepted study's title and
abstract in one prompt, produce a grouping like the hand-made cluster tables of the earlier studies in the
series — themes, sub-themes within them, a reason per study, a study allowed under two themes — and how
stable is it from run to run?

    python investigations/llm-thematic-grouping/group_studies.py STUDIES.bib --questions RQ.txt \\
        --runs 5 --declare "reasoning effort xhigh, thinking budget 12k"

Each run is one item of a resumable `raven.librarian.sweep`, so a run stopped partway is finished by running
the same command again, and every model call is kept whole in the ledger's traces. The records are shuffled
per run, with the run's number as the seed, so that the order they are presented in cannot pass for a
stable grouping. Each turn streams to `stream.log` as it runs.

What a run writes, into `--out-dir` (default: `llm-grouping/` beside the `.bib`, with the private data):
    `grouping-<fingerprint>.jsonl`, and `-traces.jsonl` beside it: the ledger, one answer per run.
    `run-NN.ods`: the grouping as a table shaped like the published ones, a row per sub-theme.
    `run-NN-placements.ods`: a row per placement, with the model's reason.
    `run-NN.pickle`: a Raven-visualizer dataset, laid out by hand (see `layout`), one point per study.

The research questions are given as context and not as the themes: the published tables group studies by
what they are about, and one theme per question would be a coding of the studies instead.
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

from unpythonic.env import env

from raven.common import netutil, tabular
from raven.common import utils as common_utils
from raven.librarian import agent, config as librarian_config, llmclient, sweep
from raven.papers import bibtex

logger = logging.getLogger(__name__)

PROMPT = """You are helping to write a systematic literature review. The review asks these research questions:

{questions}

Below are all {n} studies the review accepted, numbered, each with its citation, title, keywords where the record has them, and abstract.

Group the studies into themes by what they are about, the way a review presents its accepted studies as a table of thematic clusters. The research questions are context: they say what the review is looking for, but do not make one theme per research question. Choose the themes the material itself suggests; how many is up to you.

Within each theme, divide the studies into sub-themes. A theme that has no meaningful division has a single sub-theme.

Place every study at least once. A study that genuinely belongs under two themes may be placed under both, but treat that as the exception. For each placement, give a one-sentence reason why the study belongs there. Mark exactly one placement of each study as its primary one: where it belongs most.

Answer with JSON only, in this shape:

{{"themes": [{{"name": "...", "description": "one sentence", "subthemes": [{{"name": "...", "studies": [{{"item": 1, "primary": true, "reason": "..."}}]}}]}}]}}

The studies:

{studies}
"""

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


def prompt_for(questions: str, studies: list[env], order: list[int]) -> str:
    return PROMPT.format(questions=questions.strip(), n=len(studies),
                         studies="\n\n".join(study_item(i, studies[j]) for i, j in enumerate(order, start=1)))


# --------------------------------------------------------------------------------
# Reading the answer back

def read_grouping(reply: str, studies: list[env], order: list[int]) -> tuple[list[dict], list[str]]:
    """Parse `reply` into themes whose placements name studies by key; and the problems found.

    Returns `(themes, problems)`: `themes` a list of `{"name", "description", "subthemes": [{"name",
    "studies": [{"key", "primary", "reason"}]}]}`. An item number that names no study is dropped; a study
    placed nowhere, or with no primary placement or several, is reported. Several primaries keep the first;
    none makes the first placement primary.

    Raises `ValueError` when the reply is not JSON of the asked shape.
    """
    parsed = agent.parse_json_reply(reply)
    if not (isinstance(parsed, dict) and isinstance(parsed.get("themes"), list)):
        raise ValueError(f"expected an object with a 'themes' list, got {type(parsed).__name__}")
    problems = []
    themes = []
    placements_of = {}  # key -> list of placement dicts, in reply order
    for theme in parsed["themes"]:
        subthemes = []
        for subtheme in theme.get("subthemes", []):
            placed = []
            for study in subtheme.get("studies", []):
                item = study.get("item")
                if not (isinstance(item, int) and 1 <= item <= len(order)):
                    problems.append(f"'{theme.get('name')}' / '{subtheme.get('name')}': item {item!r} names no study")
                    continue
                key = studies[order[item - 1]].key
                if any(p["key"] == key for p in placed):
                    problems.append(f"'{subtheme.get('name')}': {key} placed twice in the same sub-theme")
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
            problems.append(f"{study.key} ({study.citation}) is placed nowhere")
        elif not primaries:
            problems.append(f"{study.key}: no primary placement; the first is taken")
            placements[0]["primary"] = True
        elif len(primaries) > 1:
            problems.append(f"{study.key}: {len(primaries)} primary placements; the first is kept")
            for p in primaries[1:]:
                p["primary"] = False
    return themes, problems


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
    """A Raven-visualizer dataset of `bib_path`, laid out by `layout`, the theme and sub-theme names as each
    cluster's keywords."""
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
              "vis_keywords_by_cluster": [[theme["name"]] + [sub["name"] for sub in theme["subthemes"]]
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


# --------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bib", help="the accepted studies")
    parser.add_argument("--questions", required=True, help="a text file holding the research questions")
    parser.add_argument("--runs", type=int, default=5, help="how many runs (default: 5)")
    parser.add_argument("--out-dir", default=None, help="where the outputs go (default: llm-grouping/ beside the .bib)")
    parser.add_argument("--backend-url", default=None,
                        help=f"the LLM backend (default: {librarian_config.llm_backend_url})")
    parser.add_argument("--declare", default=None, metavar="TEXT",
                        help="backend settings the backend does not report, e.g. 'reasoning effort xhigh'")
    opts = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bib_path = pathlib.Path(opts.bib).resolve()
    out_dir = pathlib.Path(opts.out_dir) if opts.out_dir else bib_path.parent / "llm-grouping"
    questions = pathlib.Path(opts.questions).read_text(encoding="utf-8")
    studies = load_studies(bib_path)
    keys = [s.key for s in studies]
    print(f"{len(studies)} studies from {bib_path.name}")

    llm_settings = llmclient.setup(backend_url=opts.backend_url or librarian_config.llm_backend_url, quiet=True)
    llm_settings.request_data["max_tokens"] = MAX_TOKENS
    material = json.dumps({"prompt": PROMPT, "questions": questions, "max_tokens": MAX_TOKENS,
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
        with stream_path.open("a", encoding="utf-8") as stream:
            record = agent.ask_record(settings, prompt_for(questions, studies, order), max_tokens=None,
                                      maybe_abort=maybe_abort, on_progress=agent.stream_log(stream, label=run_id))
        if record.truncated:
            return {run_id: sweep.Failure("cut off at the token cap; the trace holds what there was")}, [record]
        try:
            themes, problems = read_grouping(record.reply or "", studies, order)
        except ValueError as exc:
            return {run_id: sweep.Failure(f"unreadable reply: {exc}")}, [record]
        return {run_id: {"seed": seed, "order": [keys[j] for j in order], "themes": themes, "problems": problems,
                         "reply": record.reply}}, [record]

    items = [(f"run-{i:02d}", i) for i in range(1, opts.runs + 1)]
    result = sweep.Sweep(llm_settings, ask_batch, items, ledger,
                         header=sweep.instrument_header(llm_settings, declared=opts.declare)).run()

    answers = ledger.answers()
    for run_id, _ in items:
        if run_id not in answers:
            continue
        answer = answers[run_id]
        comments = [f"{run_id} of an LLM grouping of {bib_path.name}, instrument {ledger.fingerprint}, seed {answer['seed']}",
                    f"model: {llm_settings.model}" + (f"; {opts.declare}" if opts.declare else "")]
        write_tables(out_dir, run_id, answer["themes"], studies, answer["problems"], comments)
        write_dataset(out_dir / f"{run_id}.pickle", bib_path, answer["themes"], keys)
        n_double = sum(1 for k in keys
                       if sum(1 for t in answer["themes"] for s in t["subthemes"] for p in s["studies"] if p["key"] == k) > 1)
        print(f"{run_id}: {len(answer['themes'])} themes, "
              f"{sum(len(t['subthemes']) for t in answer['themes'])} sub-themes, "
              f"{n_double} placed twice or more, {len(answer['problems'])} problems")
    if result.outcome is not sweep.completed:
        print(f"\nstopped ({result.outcome}): {result.message or 'cancelled'}\nrun again to resume")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
