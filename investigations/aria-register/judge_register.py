"""Score the replies `probe_register.py` generated: does each one carry the voice?

The voice is easier to recognize than to name, so the rubric describes it by its features and by examples
on either side of the line, and rates the reply as a whole:

  - `grade`             `clear`, `borderline` (elements of the voice, too few or too diluted to carry the
                        reply) or `none`
  - `unsure`            whether the reply is genuinely hard to call — orthogonal to the grade, since a reply
                        can be plainly borderline
  - `directness`        the reply sets aside a convention an assistant would observe, by understatement:
                        the edge is implied and left for the reader to notice
  - `deadpan_absurd`    the reply commits to an absurd premise, answer or framing and plays it entirely
                        straight, without signalling the joke
  - `technical_wit`     the reply brings a technical tool to bear on an everyday problem it genuinely
                        solves; jargon pasted onto a situation for its own sake does not count
  - `quotes`            for each form present, where it shows most clearly, or empty for the whole reply
  - `funny`             how well the humour lands, recorded for information and never scored
  - `self_description`  does the reply talk about its own personality, manner or style; makes it `none`
  - `confrontational`   does the reply insult the user or pick a fight; makes it `none`

The forms are reported at any grade, so that a later step can see which part of the configuration drives
which. `voice` is `grade == "clear"`. The judge sees the user's last message and the reply, and nothing
about which condition produced it.

The judge is calibrated against the maintainer's ratings of a blind sample, since a rubric otherwise ends
up fitted to the examples that motivated it. `--blind N` writes that sample instead of judging: N replies
drawn at random across conditions, under opaque labels, with the key in a separate file, and none of them
a reply an earlier blind sample in the same directory already holds; `--aim` points it at what a judge
marked. `--blind-check SAMPLE` then judges only that sample's replies and prints the judge against the
ratings, which is the cheap test of a rubric before it is run over everything. The rubric's examples are
therefore all from outside the sample it is checked on.

The run is a `raven.librarian.sweep` over the probe's ledger, so it resumes; the fingerprint covers the
rubric, the sampler and the probe ledger it reads. A blind check and a full run under one rubric share a
ledger, so the full run asks only what the check did not.
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

JUDGE_TEMPERATURE = 0.3

RUBRIC = """\
You are rating one reply from an AI assistant in a chat, for a study of conversational register. Do not \
judge whether the reply is correct or helpful. Read the reply as a whole and report what it does.

We are looking for a particular kind of wit. It has three forms, and a reply may show any of them, \
anywhere in the reply - in one sentence or across the whole of it:

"directness": the reply sets aside a social convention a polite assistant would observe, and does it by \
understatement. The edge is implied rather than said - carried by a single word or a turn of phrase, and \
left for the reader to notice. It may question the premise of the exchange itself, tease across the \
fourth wall, or let it show between the lines that the user is being unreasonable or silly. Example: \
after a run of test messages, "Did you have any real work for me, or shall we keep testing?" Saying the \
edge outright, bluntly, overshoots it. So does commenting on the user's motives or calling the exchange \
a test of the assistant: that is commentary, not wit. A correction the user's message invited is NOT \
directness - if the user states something false and the reply says so, that is just the answer. Being \
concise or skipping pleasantries is not enough either.

"deadpan_absurd": the reply commits to an absurd premise, answer or framing and plays it entirely straight, \
as though it were perfectly normal, without winking. Treating a made-up or silly thing with full \
seriousness counts, and a whole reply played that way is the strongest form of it; announcing that \
something is funny, or adding an emoji or "just kidding", does not.

"technical_wit": the reply brings a technical tool to bear on an everyday problem that the tool genuinely \
solves, so that the joke is that it really is the right tool. Example: to always grab the juice carton \
with the earliest best-before date, pay O(n log n) once to keep the fridge sorted, and the pick becomes \
O(1) - which is exactly the problem a priority queue exists for. Programming jargon pasted onto a \
situation for its own sake does NOT count; that reads as inept IT humour, and it dilutes any real wit \
next to it.

None of these is: extra information, tips, caveats, rules of thumb; ordinary politeness; an offer of \
further help. Stock phrasing counts for nothing: openers such as "Ah, a classic!" or "Great question!", \
and ready-made aphorisms any assistant would produce.

Weigh the wit against the rest of the reply. Reassurance, suggestions and offers of help that follow it \
dilute it, in proportion to how much of the reply they take: a brief question afterwards does not cancel \
it, a list of things the user might do instead does. Where it sits matters too: an edge the reply ends on \
lands harder than one with more after it. This is NOT about length: a long reply that explains clearly can \
carry the voice, and clarity is never a fault. Example: asked to make code faster with no code attached, \
"No code yet, so no." standing as the answer carries it; the same point followed by a list of what the \
user might send and how the reply could then help does not.

Then grade the reply as a whole:
  "voice": "clear" if the wit is plainly there and holds up against the rest of the reply; "borderline" \
if elements of it are there but too few, or too diluted, to carry the reply; "none" otherwise.
  "unsure": "yes" if this reply is genuinely hard to call, "no" if it is not. This is separate from the \
grade: a reply can be plainly borderline, and a hard call can come out at any grade.

Also report:
  "self_description": true if the reply talks about its own personality, manner, tone or style - \
announces that it is being blunt, candid, sassy, playful and so on, or describes what kind of assistant \
it is. Describing what it can and cannot technically do does not count. If true, the grade is "none".
  "confrontational": true if the reply insults the user, calls them stupid, lectures them, or is \
combative for its own sake - an edge that would actually offend, as against one a reader has to notice. \
If true, the grade is "none".
  "funny": how well the reply's humour lands, as you read it: "very", "somewhat", "not", or "no_attempt" \
if the reply makes no attempt at humour. This does not affect the grade.

Answer with one JSON object and nothing else, no markdown fences. For each of the three forms, true if \
it is present at all, at any grade, and false otherwise; in "quotes" the passage of the reply where it \
shows most clearly, copied exactly, at most 200 characters. Leave the quote "" when the form is false, \
and also when the reply as a whole carries it, so that no single passage stands for it:
{{"directness": ..., "deadpan_absurd": ..., "technical_wit": ..., \
"quotes": {{"directness": "...", "deadpan_absurd": "...", "technical_wit": "..."}}, \
"self_description": ..., "confrontational": ..., "voice": "...", "unsure": "...", "funny": "..."}}

The user's last message:
<<<
{user}
>>>

The assistant's reply:
<<<
{reply}
>>>
"""

FEATURES = ("directness", "deadpan_absurd", "technical_wit")
GRADES = ("clear", "borderline", "none")
FUNNY = ("very", "somewhat", "not", "no_attempt")


def last_user_message(trace: dict) -> str:
    """The text of the user message the reply answered, from the probe's saved turn."""
    for prompt in reversed(trace["prompts"]):
        for message in reversed(prompt):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content
                return "".join(part.get("text", "") for part in content if part.get("type") == "text")
    raise ValueError("no user message in the trace")


def load_probe(path: pathlib.Path) -> tuple[dict[str, dict], dict[str, str]]:
    """The probe's answers, and the user message each one answered, keyed by item id."""
    fingerprint = path.stem.rsplit("-", 1)[1]
    probe = sweep.Ledger(path.parent, path.stem.rsplit("-", 1)[0], fingerprint)
    answers = probe.answers()
    users = {}
    for line in probe.traces_path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("v") == fingerprint:
            users[entry["ids"][0]] = last_user_message(entry["turns"][-1])
    return answers, users


def normalize(answer: dict, reply: str) -> dict:
    """One rating, with its vocabularies checked and each quote checked against the reply. A quote that is
    not in the reply is kept, and flagged, since a judge paraphrasing is worth knowing about rather than
    worth hiding."""
    out = {name: answer.get(name) is True for name in (*FEATURES, "self_description", "confrontational")}
    quotes = answer.get("quotes") if isinstance(answer.get("quotes"), dict) else {}
    flat_reply = " ".join(reply.split())
    out["quotes"] = {}
    out["quotes_missing"] = []
    for name in FEATURES:
        quote = str(quotes.get(name) or "").strip()[:200]
        if out[name]:
            out["quotes"][name] = quote
            # Empty is an answer: the whole reply carries it. Only a quote given and not found is missing.
            if quote and " ".join(quote.split()) not in flat_reply:
                out["quotes_missing"].append(name)
    grade = str(answer.get("voice") or "").strip().lower()
    out["grade"] = grade if grade in GRADES else "none"
    if out["self_description"] or out["confrontational"]:
        out["grade"] = "none"  # the rubric says so, and a judge that forgot is overruled rather than trusted
    out["unsure"] = str(answer.get("unsure") or "").strip().lower() == "yes"
    funny = str(answer.get("funny") or "").strip().lower()
    out["funny"] = funny if funny in FUNNY else None
    out["voice"] = out["grade"] == "clear"
    return out


CONDITIONS = ("card_on", "card_off", "plain")
AIMED_SHARE = 0.7  # of an aimed sample, the share drawn from what the judge marked
PER_PROMPT = 2     # at most this many replies per prompt, per condition and per stratum


def _draw(rng: random.Random, pool: list[str], n: int) -> list[str]:
    """`n` ids from `pool`, at most `PER_PROMPT` per prompt, prompts taken in turn so none dominates."""
    by_prompt: dict[str, list[str]] = collections.defaultdict(list)
    for item_id in rng.sample(pool, len(pool)):
        by_prompt[item_id.split("/")[1]].append(item_id)
    prompts = sorted(by_prompt)
    rng.shuffle(prompts)
    out = []
    for round_ in range(PER_PROMPT):
        for prompt in prompts:
            if len(out) < n and round_ < len(by_prompt[prompt]):
                out.append(by_prompt[prompt][round_])
    return out


def write_blind_sample(answers: dict[str, dict], users: dict[str, str], n: int, seed: int,
                       out_dir: pathlib.Path, maybe_marks: dict[str, bool] | None = None) -> None:
    """Write `n` replies for rating by hand, with their key in a separate file.

    `maybe_marks`: a judge's verdict per reply, to aim the sample with: `AIMED_SHARE` of it from what the
                   judge marked, the rest from elicitor replies it did not, both split evenly across the
                   conditions. `None` draws uniformly.
    """
    # Never a reply already rated: the rubric takes its examples from rated replies, so a second sample has
    # to be fresh for it to test the rubric rather than recite it.
    rated = set()
    for key_path in out_dir.glob("blind-key-*.json"):
        rated |= set(json.loads(key_path.read_text(encoding="utf-8")).values())
    candidates = sorted(set(answers) - rated)
    rng = random.Random(seed)
    if maybe_marks is None:
        chosen = rng.sample(candidates, min(n, len(candidates)))
    else:
        # Aimed where the judge is unsure of itself, which is in what it marked: precision is the number in
        # doubt. The unmarked share keeps recall in view, from the elicitors only, since an unmarked reply
        # to a neutral prompt is a near-certain no and tells nothing.
        n_marked = round(n * AIMED_SHARE)
        chosen = []
        for stratum_n, wanted in ((n_marked, True), (n - n_marked, False)):
            for i, condition in enumerate(CONDITIONS):
                per_condition = stratum_n // len(CONDITIONS) + (i < stratum_n % len(CONDITIONS))
                pool = [item_id for item_id in candidates
                        if item_id.startswith(condition + "/") and maybe_marks.get(item_id) is wanted
                        and (wanted or answers[item_id]["kind"] == "elicitor")]
                chosen += _draw(rng, pool, per_condition)
        rng.shuffle(chosen)
    key = {}
    lines = ["# Blind sample\n",
             "For each reply: does it carry the voice? Read the whole reply. The voice is wit of one of "
             "three kinds: directness by understatement, absurdity played entirely straight, or a technical "
             "tool brought to an everyday problem it really solves. It does not count when the reply "
             "announces a personality, and it is diluted when the reply cushions the edge. Mark **yes**, "
             "**borderline** (elements of it, too few or too diluted) or **no** after *Rating:*, adding "
             "**unsure** where the call is genuinely hard, and quote what carries it, or say that it is the "
             "whole reply.\n"]
    for i, item_id in enumerate(chosen, start=1):
        label = f"R{i:02d}"
        key[label] = item_id
        lines += [f"\n---\n\n## {label}\n", "**User:**\n", f"> {users[item_id].strip()}".replace("\n", "\n> ") + "\n",
                  "**Reply:**\n", answers[item_id]["reply"].strip() + "\n", "*Rating:* \n"]
    (out_dir / f"blind-sample-{seed}.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / f"blind-key-{seed}.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / f'blind-sample-{seed}.md'} and its key")


HAND_VERDICTS = ("yes", "borderline", "unsure", "no")


def read_blind_ratings(sample_path: pathlib.Path) -> dict[str, tuple[str, str]]:
    """The hand ratings in a filled-in blind sample, by item id: `(verdict, note)`, the verdict being the
    note's first word if it is one of `HAND_VERDICTS`, and "unsure" otherwise."""
    key = json.loads(sample_path.with_name(sample_path.name.replace("blind-sample-", "blind-key-")
                                           .replace(".md", ".json")).read_text(encoding="utf-8"))
    text = sample_path.read_text(encoding="utf-8")
    out = {}
    for match in re.finditer(r"^## (R\d+)\n(.*?)^\*Rating:\*[ \t]*(.*?)$", text, flags=re.S | re.M):
        note = match.group(3).strip()
        first = note.lower().split(maxsplit=1)[0].strip("!.,;:") if note else ""
        out[key[match.group(1)]] = (first if first in HAND_VERDICTS else "unsure", note)
    return out


def compare_with_blind(ratings: dict[str, dict],
                       hand: dict[str, tuple[str, str]]) -> None:
    """The judge against the hand ratings, one line per reply, and the counts by hand verdict and grade.

    A rating from a judge without grades (an older rubric) counts as `clear` where it marked the voice and
    `none` where it did not.
    """
    counts = collections.Counter()
    unsure = collections.Counter()
    for item_id, (verdict, _note) in sorted(hand.items(), key=lambda kv: (HAND_VERDICTS.index(kv[1][0]), kv[0])):
        rating = ratings.get(item_id)
        if rating is None:
            continue
        grade = rating.get("grade", "clear" if rating["voice"] else "none")
        counts[(verdict, grade)] += 1
        unsure[verdict] += bool(rating.get("unsure"))
        found = ", ".join(f"{name}: {rating['quotes'][name][:50]!r}" for name in rating["quotes"])
        flags = ("?" if rating.get("unsure") else " ") + (f" funny:{rating['funny']}" if rating.get("funny") else "")
        print(f"  {verdict:<11}{grade:<11}{flags:<20}{item_id:<38}{found}")
    print(f"\n  hand \\ judge{'':<4}" + "".join(f"{g:>12}" for g in GRADES) + f"{'judge unsure':>15}")
    for verdict in HAND_VERDICTS:
        print(f"  {verdict:<17}" + "".join(f"{counts[(verdict, g)]:>12}" for g in GRADES) + f"{unsure[verdict]:>15}")


def summarize(answers: dict[str, dict], ratings: dict[str, dict]) -> None:
    cells = collections.defaultdict(lambda: collections.Counter())
    for item_id, rating in ratings.items():
        answer = answers[item_id]
        for key in ((answer["condition"], answer["kind"]), (answer["condition"], answer["prompt"])):
            cells[key]["n"] += 1
            cells[key][rating["grade"]] += 1
            for name in FEATURES:
                cells[key][name] += rating[name]
    print(f"\n{'':<30}{'clear':>12}{'borderline':>12}  " + "".join(f"{name:>16}" for name in FEATURES))
    for condition in CONDITIONS:
        print(f"\n{condition}")
        keys = sorted(what for c, what in cells if c == condition)
        for what in [k for k in keys if k in ("elicitor", "neutral")] + [k for k in keys if k not in ("elicitor", "neutral")]:
            cell = cells[(condition, what)]
            indent = "  " if what in ("elicitor", "neutral") else "    "
            print(f"{indent}{what:<{28 - len(indent) + 2}}{cell['clear']:>5} / {cell['n']:<4}{cell['borderline']:>10}  " +
                  "".join(f"{cell[name]:>16}" for name in FEATURES))
    told = sum(r["self_description"] for r in ratings.values())
    missing = sum(bool(r["quotes_missing"]) for r in ratings.values())
    fights = sum(r["confrontational"] for r in ratings.values())
    hard = sum(r["unsure"] for r in ratings.values())
    print(f"\nself-description in {told} of {len(ratings)}, confrontational in {fights}, hard to call in {hard}; "
          f"a quote not found in its reply in {missing}")
    funny = collections.Counter((answers[i]["condition"], r["funny"]) for i, r in ratings.items())
    print("\nfunny, for information:")
    for condition in CONDITIONS:
        print(f"  {condition:<10}" + "  ".join(f"{f}: {funny[(condition, f)]}" for f in FUNNY))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("probe", help="the probe's ledger, register-step1-<fingerprint>.jsonl")
    parser.add_argument("--blind", type=int, default=None, metavar="N",
                        help="write a blind sample of N replies for rating by hand, instead of judging")
    parser.add_argument("--seed", type=int, default=7, help="which random sample --blind takes")
    parser.add_argument("--aim", default=None, metavar="JUDGED",
                        help="with --blind: a judge's ledger, register-judged-<fingerprint>.jsonl, to aim the "
                             "sample at what that judge marked")
    parser.add_argument("--blind-check", default=None, metavar="SAMPLE",
                        help="judge only the replies in this filled-in blind sample, and compare the judge "
                             "with the ratings in it; its key is read from beside it")
    parser.add_argument("--backend-url", default=None,
                        help=f"the LLM backend (default: {librarian_config.llm_backend_url})")
    parser.add_argument("--declare", default=None, metavar="TEXT",
                        help="backend settings the backend does not report, e.g. 'KV cache q8_0'")
    opts = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    probe_path = pathlib.Path(opts.probe).resolve()
    out_dir = probe_path.parent
    answers, users = load_probe(probe_path)
    print(f"{len(answers)} replies in {probe_path.name}")

    if opts.blind is not None:
        maybe_marks = None
        if opts.aim:
            judged = sweep.Ledger(pathlib.Path(opts.aim).parent, pathlib.Path(opts.aim).stem.rsplit("-", 1)[0],
                                  pathlib.Path(opts.aim).stem.rsplit("-", 1)[1])
            maybe_marks = {item_id: rating["voice"] for item_id, rating in judged.answers().items()}
        write_blind_sample(answers, users, opts.blind, opts.seed, out_dir, maybe_marks)
        return 0

    llm_settings = llmclient.setup(backend_url=opts.backend_url or librarian_config.llm_backend_url, quiet=True)
    llm_settings.request_data["temperature"] = JUDGE_TEMPERATURE
    material = json.dumps({"rubric": RUBRIC, "temperature": JUDGE_TEMPERATURE, "probe": probe_path.name})
    ledger = sweep.Ledger(out_dir, "register-judged", sweep.fingerprint(llm_settings, material, declared=opts.declare))
    print(f"instrument {ledger.fingerprint} -> {ledger.path.name}")

    def ask_batch(settings: env, batch: list[tuple[str, dict]],
                  maybe_abort: netutil.Abort | None) -> tuple[dict, list[agent.TurnRecord]]:
        (item_id, answer), = batch
        record = agent.ask_record(settings, RUBRIC.format(user=users[item_id].strip(), reply=answer["reply"].strip()),
                                  maybe_abort=maybe_abort)
        try:
            rating = agent.parse_json_reply(record.reply or "")
        except ValueError:
            rating = None
        if not isinstance(rating, dict):
            return {item_id: sweep.Failure("the judge's reply did not parse; it is in the traces")}, [record]
        return {item_id: normalize(rating, answer["reply"])}, [record]

    hand = read_blind_ratings(pathlib.Path(opts.blind_check)) if opts.blind_check else None
    items = sorted((item_id, answer) for item_id, answer in answers.items() if hand is None or item_id in hand)
    result = sweep.Sweep(llm_settings, ask_batch, items, ledger,
                         header=sweep.instrument_header(llm_settings, declared=opts.declare)).run()
    if result.outcome is not sweep.completed:
        print(f"\nstopped ({result.outcome}): {result.message or 'cancelled'}\nrun again to resume")
        return 1
    ratings = {item_id: rating for item_id, rating in ledger.answers().items() if item_id in answers}
    if hand is not None:
        compare_with_blind(ratings, hand)
    else:
        summarize(answers, ratings)
    print(f"\nwrote {ledger.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
