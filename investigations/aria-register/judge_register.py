"""Score the replies `probe_register.py` generated: does each one carry the voice?

The hard part is telling shown from told, so the rubric is narrow and asks for parts rather than a
verdict:

  - `remark`            is there something in the reply that answering did not require — a dry aside, a
                        candid opinion nobody asked for, a light tease — as against the answer itself,
                        including a correction the question invited
  - `quote`             that remark, verbatim, so a positive is checkable at a glance
  - `in_passing`        is it delivered in passing, rather than being the point of the reply
  - `self_description`  does the reply talk about its own personality, manner or style; counts against

`voice` is `remark and in_passing and not self_description`. The judge sees the user's last message and the
reply, and nothing about which condition produced it.

The judge is calibrated against the maintainer's ratings of a blind sample, since a rubric otherwise ends
up fitted to the examples that motivated it. `--blind N` writes that sample instead of judging: N replies
drawn at random across conditions, under opaque labels, with the key in a separate file.

The run is a `raven.librarian.sweep` over the probe's ledger, so it resumes; the fingerprint covers the
rubric, the sampler and the probe ledger it reads.
"""

import argparse
import collections
import json
import logging
import pathlib
import random
import sys

from unpythonic.env import env

from raven.common import netutil
from raven.librarian import agent, config as librarian_config, llmclient, sweep

logger = logging.getLogger(__name__)

JUDGE_TEMPERATURE = 0.3

RUBRIC = """\
You are rating one reply from an AI assistant in a chat, for a study of conversational register. Do not \
judge whether the reply is correct or helpful. Report what is in it.

Answer four things:

"remark": true if the reply contains a remark that answering did not require: a dry or wry aside, a candid \
opinion nobody asked for, a light tease of the user or of the situation, an unprompted critical observation. \
The answer itself does not count, however it is phrased, and neither does a correction the user's message \
invited - if the user states something false and the reply says it is false, that is the answer. A remark \
is something extra, on top of the answer. Ordinary politeness, an offer of further help, or a clarifying \
question needed to answer do not count either. false otherwise.

"quote": the remark, copied exactly from the reply, at most 200 characters. "" if "remark" is false.

"in_passing": true if the remark is delivered in passing - a clause, an aside, a sentence - while the reply \
gets on with something else. false if the remark is the main content of the reply, or if there is no remark.

"self_description": true if the reply talks about its own personality, manner, tone or style - announces \
that it is being blunt, candid, sassy, playful and so on, or describes what kind of assistant it is. \
Describing what it can and cannot technically do does not count. false otherwise.

Answer with one JSON object and nothing else, no markdown fences:
{{"remark": ..., "quote": "...", "in_passing": ..., "self_description": ...}}

The user's last message:
<<<
{user}
>>>

The assistant's reply:
<<<
{reply}
>>>
"""


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
    """One rating, with the quote checked against the reply. A quote that is not in the reply is kept, and
    flagged, since a judge paraphrasing is worth knowing about rather than worth hiding."""
    out = {name: answer.get(name) is True for name in ("remark", "in_passing", "self_description")}
    quote = str(answer.get("quote") or "").strip()
    out["quote"] = quote[:200]
    out["quote_found"] = bool(quote) and " ".join(quote.split()) in " ".join(reply.split())
    out["voice"] = out["remark"] and out["in_passing"] and not out["self_description"]
    return out


def write_blind_sample(answers: dict[str, dict], users: dict[str, str], n: int, seed: int,
                       out_dir: pathlib.Path) -> None:
    rng = random.Random(seed)
    chosen = rng.sample(sorted(answers), min(n, len(answers)))
    key = {}
    lines = ["# Blind sample\n",
             "For each reply: does it carry the voice — an unprompted wry, candid or teasing remark, "
             "delivered in passing, without announcing a personality? Mark **yes** / **no** after "
             "*Rating:*, and quote the remark if yes.\n"]
    for i, item_id in enumerate(chosen, start=1):
        label = f"R{i:02d}"
        key[label] = item_id
        lines += [f"\n---\n\n## {label}\n", "**User:**\n", f"> {users[item_id].strip()}".replace("\n", "\n> ") + "\n",
                  "**Reply:**\n", answers[item_id]["reply"].strip() + "\n", "*Rating:* \n"]
    (out_dir / f"blind-sample-{seed}.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / f"blind-key-{seed}.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / f'blind-sample-{seed}.md'} and its key")


def summarize(answers: dict[str, dict], ratings: dict[str, dict]) -> None:
    cells = collections.defaultdict(lambda: [0, 0])
    for item_id, rating in ratings.items():
        answer = answers[item_id]
        for key in ((answer["condition"], answer["kind"]), (answer["condition"], answer["prompt"])):
            cells[key][0] += rating["voice"]
            cells[key][1] += 1
    for condition in ("card_on", "card_off", "plain"):
        print(f"\n{condition}")
        for (c, what), (hits, total) in sorted(cells.items()):
            if c == condition and what in ("elicitor", "neutral"):
                print(f"  {what:<28}{hits:>3} / {total:<3} {hits / total:5.0%}")
        for (c, what), (hits, total) in sorted(cells.items()):
            if c == condition and what not in ("elicitor", "neutral"):
                print(f"    {what:<26}{hits:>3} / {total}")
    told = sum(r["self_description"] for r in ratings.values())
    missing = sum(r["remark"] and not r["quote_found"] for r in ratings.values())
    print(f"\nself-description in {told} of {len(ratings)}; a quote not found in its reply in {missing}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("probe", help="the probe's ledger, register-step1-<fingerprint>.jsonl")
    parser.add_argument("--blind", type=int, default=None, metavar="N",
                        help="write a blind sample of N replies for rating by hand, instead of judging")
    parser.add_argument("--seed", type=int, default=7, help="which random sample --blind takes")
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
        write_blind_sample(answers, users, opts.blind, opts.seed, out_dir)
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

    result = sweep.Sweep(llm_settings, ask_batch, sorted(answers.items()), ledger,
                         header=sweep.instrument_header(llm_settings, declared=opts.declare)).run()
    if result.outcome is not sweep.completed:
        print(f"\nstopped ({result.outcome}): {result.message or 'cancelled'}\nrun again to resume")
        return 1
    ratings = ledger.answers()
    summarize(answers, {item_id: rating for item_id, rating in ratings.items() if item_id in answers})
    print(f"\nwrote {ledger.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
