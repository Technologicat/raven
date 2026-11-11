"""Cluster a small dataset by asking an LLM to group it into themes, and lay the themes out as circles.

For a corpus small enough that every title and abstract fits in one prompt, a few hundred records at most.
The model sees them all at once and names the themes itself, so the clusters come with names. The map is
made up rather than computed: one circle per theme, its records around the perimeter, the circles' centres
on a maximin Latin square so that they read as scattered rather than gridded.

A record may be placed under two themes. Each has one primary placement, which is where its point is drawn;
the others are kept in the themes this returns.

Developed in `investigations/llm-thematic-grouping/`, which also measured how much the grouping varies from
run to run; this is one run of it.
"""

__all__ = ["PROMPT", "AIM", "REPAIR", "MAX_REPAIRS", "MAX_REPLY_TOKENS",
           "PromptTooLarge",
           "build_prompt", "read_grouping", "ask_with_repairs", "layout", "cluster"]

import logging
import math

import numpy as np

from ..common import numutils

logger = logging.getLogger(__name__)

PROMPT = """You are helping to make sense of a collection of research papers.{aim}

Below are all {n} papers in the collection, numbered, each with its authors, year, title and abstract.

Group the papers into themes by what they are about.{aim_note} Choose the themes the material itself suggests; how many is up to you.

Place every paper at least once. A paper that genuinely belongs under two themes may be placed under both, but treat that as the exception. For each placement, give a one-sentence reason why the paper belongs there. Mark exactly one placement of each paper as its primary one: where it belongs most.

Answer with JSON only, in this shape:

{{"themes": [{{"name": "...", "description": "one sentence", "papers": [{{"item": 1, "primary": true, "reason": "..."}}]}}]}}

The papers:

{papers}
"""

#: Filled into the prompt when an aim is given.
AIM = " The collection's aim:\n\n{aim}"
AIM_NOTE = (" The aim is context: it says what the collection is about, but do not make one theme per part of"
            " the aim.")

#: Asked after a reply with slips: the original prompt, the reply, and what was wrong with it. Built on the
#: original prompt so that the backend's prefix cache covers it.
REPAIR = """{prompt}
Your answer was:

{reply}

It has these problems:

{problems}

Answer again with the whole corrected JSON, in the same shape, and nothing else.
"""

#: How many repair turns a grouping may take before what remains is fixed up.
MAX_REPAIRS = 2

#: The reply's token cap, kept free of the context window when checking that the prompt fits.
MAX_REPLY_TOKENS = 16384


class PromptTooLarge(ValueError):
    """The dataset does not fit in one prompt, so it cannot be grouped this way."""


def _paper_item(i: int, entry) -> str:
    """`entry` as the prompt shows it, numbered `i` from 1. `entry` has `author`, `year`, `title`, `abstract`."""
    return (f"[{i}] {entry.author or 'Anonymous'} ({entry.year or 'n.d.'}). {entry.title}\n"
            f"Abstract: {entry.abstract or '(none)'}")


def build_prompt(entries: list, *, maybe_aim: str | None = None) -> str:
    """The grouping prompt for `entries`, numbered from 1 in their given order.

    `maybe_aim`: what the collection is for, given as context; empty or `None` to say nothing about it.
    """
    aim = (maybe_aim or "").strip()
    return PROMPT.format(aim=AIM.format(aim=aim) if aim else "",
                         aim_note=AIM_NOTE if aim else "",
                         n=len(entries),
                         papers="\n\n".join(_paper_item(i, entry) for i, entry in enumerate(entries, start=1)))


def read_grouping(reply: str, n: int) -> tuple[list[dict], list[str]]:
    """Parse `reply` into themes, and the problems found in it.

    `n`: how many papers the prompt numbered.

    Returns `(themes, problems)`: `themes` a list of `{"name", "description", "placements": [{"index",
    "primary", "reason"}]}`, `index` counted from 0. An item number that names no paper is dropped, and so
    is a paper placed twice in one theme; a paper placed nowhere is reported. A paper with several primary
    placements keeps the first, and one with none has its first placement made primary. Each problem is
    worded for the model as much as for a person, since a repair turn quotes it.

    Raises `ValueError` when the reply is not JSON of the asked shape.
    """
    from ..librarian import agent  # noqa: PLC0415 -- the LLM side is loaded only when a run needs it

    parsed = agent.parse_json_reply(reply)
    if not (isinstance(parsed, dict) and isinstance(parsed.get("themes"), list)):
        raise ValueError(f"expected an object with a 'themes' list, got {type(parsed).__name__}")
    problems = []
    themes = []
    placements_of: dict[int, list[dict]] = {}
    for theme in parsed["themes"]:
        if not isinstance(theme, dict):
            raise ValueError(f"expected each theme to be an object, got {type(theme).__name__}")
        name = str(theme.get("name", ""))
        placements = []
        for paper in theme.get("papers", []):
            item = paper.get("item") if isinstance(paper, dict) else None
            if not (isinstance(item, int) and 1 <= item <= n):
                problems.append(f"theme '{name}': item {item!r} names no paper")
                continue
            index = item - 1
            if any(p["index"] == index for p in placements):
                problems.append(f"theme '{name}': paper [{item}] is placed in it twice")
                continue
            placement = {"index": index, "primary": paper.get("primary") is True, "reason": str(paper.get("reason", ""))}
            placements.append(placement)
            placements_of.setdefault(index, []).append(placement)
        themes.append({"name": name, "description": str(theme.get("description", "")), "placements": placements})
    for index in range(n):
        placements = placements_of.get(index, [])
        primaries = [p for p in placements if p["primary"]]
        if not placements:
            problems.append(f"paper [{index + 1}] is placed nowhere")
        elif not primaries:
            problems.append(f"paper [{index + 1}] has no placement marked primary")
            placements[0]["primary"] = True
        elif len(primaries) > 1:
            problems.append(f"paper [{index + 1}] has {len(primaries)} placements marked primary; it needs exactly one")
            for p in primaries[1:]:
                p["primary"] = False
    return themes, problems


def ask_with_repairs(llm_settings, prompt: str, n: int) -> tuple[list[dict], list[str]]:
    """Ask `prompt`, and while the reply has slips, ask again naming them, up to `MAX_REPAIRS` times.

    Returns `(themes, problems)` as `read_grouping` gives them, the problems being whatever was left after the
    last turn, fixed up as it says.

    Raises `RuntimeError` when no usable grouping came back: a reply cut off at the token cap, or one still not
    JSON of the asked shape after the repairs.
    """
    from ..librarian import agent  # noqa: PLC0415 -- the LLM side is loaded only when a run needs it

    question = prompt
    for attempt in range(MAX_REPAIRS + 1):
        record = agent.ask_record(llm_settings, question, max_tokens=MAX_REPLY_TOKENS)
        for reasoning in record.reasoning:
            logger.info(f"ask_with_repairs: LLM thoughts:\n{reasoning}")
        if record.truncated:
            raise RuntimeError("ask_with_repairs: the reply was cut off at the token cap")
        try:
            themes, problems = read_grouping(record.reply or "", n)
        except ValueError as exc:
            themes, problems = None, [f"the answer is not JSON of the asked shape: {exc}"]
        if not problems:
            return themes, problems
        logger.info(f"ask_with_repairs: {len(problems)} problem(s) in the reply: {problems}")
        if attempt < MAX_REPAIRS:
            question = REPAIR.format(prompt=prompt, reply=(record.reply or "").strip(),
                                     problems="\n".join(f"- {p}" for p in problems))
    if themes is None:
        raise RuntimeError(f"ask_with_repairs: unreadable after {MAX_REPAIRS} repairs: {problems[0]}")
    return themes, problems


def layout(themes: list[dict], n: int, *, spacing: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Made-up 2D coordinates for `n` papers, one point per paper at its primary placement.

    One circle per theme, its papers around the perimeter; a circle's radius grows with its count, keeping
    neighbours `spacing` apart. The circles' centres sit on a maximin Latin square
    (`raven.common.numutils.maximin_latin_square`). Papers placed nowhere get a circle of their own after the
    themes, labelled as outliers.

    Returns `(points, labels)`: a `[n, 2]` array, and each paper's theme index, -1 for an outlier.
    """
    members = [[p["index"] for p in theme["placements"] if p["primary"]] for theme in themes]
    placed = {index for group in members for index in group}
    outliers = [index for index in range(n) if index not in placed]
    groups = members + ([outliers] if outliers else [])
    group_labels = list(range(len(members))) + ([-1] if outliers else [])
    radii = [max(spacing, len(group) * spacing / (2 * math.pi)) if len(group) > 1 else 0.0 for group in groups]
    step = 2.5 * max(radii + [spacing])
    columns = numutils.maximin_latin_square(len(groups), step, step)

    points = np.zeros((n, 2), dtype=np.float64)
    labels = np.full(n, -1, dtype=np.int64)
    for row, (group, radius, label) in enumerate(zip(groups, radii, group_labels)):
        cx, cy = columns[row] * step, -row * step
        for j, index in enumerate(group):
            angle = math.pi / 2 - 2 * math.pi * j / len(group)
            points[index] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
            labels[index] = label
    return points, labels


def cluster(llm_settings, entries: list, *, maybe_aim: str | None = None) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Group `entries` into themes with one LLM call, and lay them out.

    `entries`: each with `author`, `year`, `title` and `abstract`, as the importer parses them.
    `maybe_aim`: what the collection is for, given to the model as context.

    Returns `(labels, points, themes)`: each entry's theme index (-1 for one placed nowhere), the made-up 2D
    points, and the themes as `read_grouping` gives them.

    Raises `PromptTooLarge` before asking when the prompt and the reply's cap do not fit in the loaded context
    window, and `RuntimeError` when no usable grouping came back.
    """
    from ..librarian import llmclient  # noqa: PLC0415 -- the LLM side is loaded only when a run needs it

    prompt = build_prompt(entries, maybe_aim=maybe_aim)
    llmclient.wait_for_tokenizer(llm_settings)  # an exact count where one is available; it loads in the background
    n_tokens, exact = llmclient.count_tokens(llm_settings, prompt)
    budget = llm_settings.context_length - MAX_REPLY_TOKENS
    if n_tokens > budget:
        raise PromptTooLarge(f"{len(entries)} records make a prompt of {'' if exact else 'about '}{n_tokens} tokens, "
                             f"and {budget} fit beside the reply in the loaded model's context window. "
                             'Group a dataset this size with `clustering_method = "agglomerative"` instead.')
    logger.info(f"cluster: asking for themes over {len(entries)} records, a prompt of {'' if exact else 'about '}{n_tokens} tokens.")
    themes, problems = ask_with_repairs(llm_settings, prompt, len(entries))
    if problems:
        logger.warning(f"cluster: the grouping kept {len(problems)} problem(s) after the repairs, fixed up as far as possible: {problems}")
    points, labels = layout(themes, len(entries))
    return labels, points, themes
