"""Step 1 of the register investigation: does the voice go with the user card?

Three conditions, which differ only in the system text the chat is rooted at:

  - `card_on`   Aria's card and the interaction block as shipped, plus the maintainer's user card. What
                produced the October examples.
  - `card_off`  the same, with the user card emptied.
  - `plain`     a one-line character card, no user card. The base rate: a voice that appears here too
                belongs to the model rather than to anything we wrote.

Everything else is held fixed across the three: the greeting, the persona prefixes, the per-turn system
postamble (date, loaded model), the tools offered, the sampler. So whatever differs between the rates is
the system text.

The prompts are of two kinds. *Elicitors* invite the voice without asking for it — a vague request, a wrong
premise, a system handling something clumsily, and the October exchanges replayed from just before the
remark. *Neutral* prompts should not draw it, and are the false-positive check: a condition that puts the
voice everywhere fails as surely as one that removes it.

A replay copies the stored branch — every message after the greeting, attachments included — into a fresh
in-memory chat rooted at the condition's system text, and runs the assistant's turn from the last user
message. The two October examples that reacted to the automatic document search are not replayed: that
search keyed on the whole message and on a garbled query, both since fixed, so the stimulus no longer
exists.

The prompts come in sets, by the date each was written (`--prompts`, the newest by default), each writing
its own ledger: `register-step1-*` for the first, `register-prompts2-*` for the second. A second set was
needed because the first set's replies were used up for blind rating. The second has no replays, and adds
an everyday problem a technical tool genuinely solves, which invites `technical_wit`.

The run is a `raven.librarian.sweep`. An item is `condition/prompt/sample`, ordered sample-major so that a
run stopped early is balanced across conditions and prompts. The prompt's name carries a hash of its text,
so editing one prompt asks only that prompt again; the fingerprint covers the conditions' system texts and
the sampler, so changing those starts a new ledger. Each turn is saved whole in the traces file, and streams
to `stream.log` while it runs.

Scoring the replies is a separate pass; this only generates them.
"""

import argparse
import hashlib
import json
import logging
import pathlib
import sys

from unpythonic.env import env

from raven.common import netutil
from raven.librarian import agent, chattree, chatutil, config as librarian_config, llmclient, sweep

logger = logging.getLogger(__name__)

MAX_TOKENS = 8192  # a guard against a runaway turn; the October replies were a few hundred tokens

PLAIN_CARD = "You are {char}, an AI assistant."

# Where the October chats are. The chat these two replays come from was moved aside on 2026-10-08, so it is
# not in the live datastore; its attachments still are, the sidecar store being content-addressed.
SOURCE_DATASTORE = pathlib.Path.home() / ".config" / "raven" / "librarian" / "chat_orig.json"
SOURCE_SIDECARS = pathlib.Path.home() / ".config" / "raven" / "librarian" / "chat.sidecars"

# The prompt sets, by the date each was written. Each writes its own ledger, so a set's replies are never
# judged together with another's, and the first set's fingerprint is what it was when it ran.
PROMPT_SETS = {
    "2026-10-21": env(
        ledger="register-step1",
        elicitors={
            # Replayed in content from 2026-09-25, where it drew "are you testing my creativity?" under another
            # character. A fresh chat, so it needs nothing stored.
            "shoelaces": "How to tie your shoelaces without touching the shoelaces?",
            "vague": "Can you make my code faster?",
            "wrong_premise": "Since the GIL was removed in Python 3.12, threads are now the obvious choice for "
                             "CPU-bound work, right?",
            "creative": "Propose a name for a new physical unit: one that measures how far a meeting runs over "
                        "its schedule.",
            "clumsy_system": "Did the attachment I just sent come through?",  # nothing is attached
        },
        # Replays: the node holding the remark. The turn is re-run from its parent.
        replays={
            "replay_pipe": "gensym#forest-node:6276d1e3-b961-4596-add0-4b9672cf96f7",  # "...your side of the pipe"
            "replay_thanks": "gensym#forest-node:73906825-bcb0-428e-966b-4ba736f6bab4",  # "...or for the actual work"
        },
        neutral={
            "boiling": "What is the boiling point of water at 2000 m altitude?",
            "fourier": "Briefly explain what a Fourier transform does.",
            "sort": "How do I sort a list of dicts by one key in Python?",
        }),
    # New prompts of the same kinds, because the first set's replies are used up for blind rating: the same
    # prompt recurring in every sample makes its replies read as less witty to the rater. No replays: no
    # other stored remark with the same vibe was at hand. One new kind, for `technical_wit`, which no prompt
    # of the first set invited.
    "2026-10-28": env(
        ledger="register-prompts2",
        elicitors={
            "absurd": "How can I be on time for a meeting that started yesterday?",
            "vague": "Can you fix the bug?",
            "wrong_premise": "Since Python 3.13 made `list.sort()` return the sorted list, I can finally chain "
                             "it, right?",
            "creative": "Propose a name for a unit that measures how long a 'quick question' actually takes.",
            "clumsy_system": "Which of the two pictures I attached do you like better?",  # nothing is attached
            "everyday_technical": "My nightstand has a pile of unread books and I keep buying more. How do I "
                                  "decide what to read next?",
        },
        replays={},
        neutral={
            "capital": "What is the capital of Australia?",
            "hash_table": "Briefly explain what a hash table is.",
            "reverse": "How do I reverse a string in Python?",
        }),
}
LATEST = "2026-10-28"


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:6]


def condition_texts(llm_settings: env) -> dict[str, dict[str, str]]:
    """The system text of each condition, as overrides on `llm_settings`. Read from the settings as built,
    so `card_on` is exactly what Librarian sends."""
    shipped = {"system_prompt": llm_settings.system_prompt,
               "character_card": llm_settings.character_card,
               "user_card": llm_settings.user_card}
    if not shipped["user_card"]:
        raise RuntimeError("no user card in the settings, so `card_on` and `card_off` would be the same "
                           "condition; check `llm_user_name` and the profile under "
                           "~/.config/raven/librarian/users/")
    char = llm_settings.personas["assistant"]
    return {"card_on": shipped,
            "card_off": {**shipped, "user_card": ""},
            "plain": {"system_prompt": "", "character_card": PLAIN_CARD.format(char=char), "user_card": ""}}


def load_replay(node_id: str) -> list[dict]:
    """The stored payloads of the branch ending at `node_id`'s parent, after the system message and the
    greeting — i.e. the conversation as the user had it just before the remark."""
    nodes = json.loads(SOURCE_DATASTORE.read_text(encoding="utf-8"))
    chain = []
    current = nodes[node_id]["parent"]
    while current is not None:
        node = nodes[current]
        chain.append(node["data"][str(node["active_revision"])])
        current = node.get("parent")
    chain.reverse()
    if chain[0]["message"]["role"] != "system" or chain[1]["message"]["role"] != "assistant":
        raise ValueError(f"load_replay: {node_id} does not hang from a system message and a greeting")
    if chain[-1]["message"]["role"] != "user":
        raise ValueError(f"load_replay: {node_id}'s parent is not a user message")
    return chain[2:]


def build_replay(llm_settings: env, payloads: list[dict]) -> tuple[chattree.Forest, str]:
    """A fresh chat rooted at the current settings' system text, carrying `payloads` after the greeting.
    Returns the datastore and the id of the last node."""
    datastore = chattree.Forest()
    head = chatutil.factory_reset_datastore(datastore, llm_settings)
    for payload in payloads:
        for filename, metadata in payload.get("general_metadata", {}).get("sidecars", {}).items():
            data = (SOURCE_SIDECARS / filename).read_bytes()
            stored = datastore.store_sidecar(data, filename.rsplit(".", 1)[1], metadata)
            assert stored == filename  # content-addressed, so the stored URLs stay valid
        head = datastore.create_node(payload=json.loads(json.dumps(payload)), parent_id=head)
    return datastore, head


def make_items(prompt_set: env, n_samples: int, n_neutral_samples: int | None = None) -> list[tuple[str, dict]]:
    """`condition/prompt/sample`, sample-major, so an early stop leaves every cell with the same count
    give or take one. The neutral prompts stop at `n_neutral_samples`, when given."""
    prompts = ([(f"{name}-{short_hash(text)}", "elicitor", {"text": text})
                for name, text in prompt_set.elicitors.items()] +
               [(f"{name}-{short_hash(node_id)}", "elicitor", {"replay": node_id})
                for name, node_id in prompt_set.replays.items()] +
               [(f"{name}-{short_hash(text)}", "neutral", {"text": text}) for name, text in prompt_set.neutral.items()])
    items = []
    for sample in range(n_samples):
        for condition in ("card_on", "card_off", "plain"):
            for prompt_id, kind, stimulus in prompts:
                if kind == "neutral" and n_neutral_samples is not None and sample >= n_neutral_samples:
                    continue
                items.append((f"{condition}/{prompt_id}/{sample}",
                              {"condition": condition, "prompt": prompt_id, "kind": kind, **stimulus}))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", default=LATEST, choices=sorted(PROMPT_SETS),
                        help="which prompt set, by the date it was written")
    parser.add_argument("--samples", type=int, default=5, help="replies per condition and prompt")
    parser.add_argument("--neutral-samples", type=int, default=None, metavar="N",
                        help="replies per condition for the neutral prompts, when fewer than --samples: the "
                             "elicitors are where the rates are low enough to want more")
    parser.add_argument("--backend-url", default=None,
                        help=f"the LLM backend (default: {librarian_config.llm_backend_url})")
    parser.add_argument("--declare", default=None, metavar="TEXT",
                        help="backend settings the backend does not report, e.g. 'KV cache q8_0'")
    parser.add_argument("--out-dir", default=None, help="where the outputs go (default: beside this)")
    opts = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = pathlib.Path(opts.out_dir) if opts.out_dir else pathlib.Path(__file__).resolve().parent

    llm_settings = llmclient.setup(backend_url=opts.backend_url or librarian_config.llm_backend_url, quiet=True)
    llm_settings.request_data["max_tokens"] = MAX_TOKENS
    conditions = condition_texts(llm_settings)
    prompt_set = PROMPT_SETS[opts.prompts]
    replays = {node_id: load_replay(node_id) for node_id in prompt_set.replays.values()}

    material = json.dumps({"conditions": conditions, "max_tokens": MAX_TOKENS,
                           "sampler": {k: llm_settings.request_data.get(k) for k in ("temperature", "min_p", "top_p", "top_k")},
                           "retain_thinking": llm_settings.retain_thinking},
                          sort_keys=True)
    ledger = sweep.Ledger(out_dir, prompt_set.ledger, sweep.fingerprint(llm_settings, material, declared=opts.declare))
    print(f"instrument {ledger.fingerprint} -> {ledger.path.name}")

    stream_path = out_dir / "stream.log"

    def ask_batch(settings: env, batch: list[tuple[str, dict]],
                  maybe_abort: netutil.Abort | None) -> tuple[dict, list[agent.TurnRecord]]:
        (item_id, item), = batch
        for name, value in conditions[item["condition"]].items():
            setattr(settings, name, value)
        with stream_path.open("a", encoding="utf-8") as stream:
            on_progress = agent.stream_log(stream, label=item_id)
            if "replay" in item:
                datastore, head = build_replay(settings, replays[item["replay"]])
                record = agent.turn(settings, None, datastore=datastore, head_node_id=head,
                                    on_progress=on_progress, maybe_abort=maybe_abort)
            else:
                record = agent.turn(settings, item["text"], on_progress=on_progress, maybe_abort=maybe_abort)
        if record.generation is None:
            return {item_id: sweep.Failure(f"no generation: {record.reply[:200]}")}, [record]
        return {item_id: {**{k: v for k, v in item.items() if k != "replay"},
                          "reply": record.reply,
                          "reasoning": list(record.reasoning),
                          "tool_calls": dict(record.tool_calls)}}, [record]

    result = sweep.Sweep(llm_settings, ask_batch, make_items(prompt_set, opts.samples, opts.neutral_samples), ledger,
                         header=sweep.instrument_header(llm_settings, declared=opts.declare)).run()
    if result.outcome is not sweep.completed:
        print(f"\nstopped ({result.outcome}): {result.message or 'cancelled'}\nrun again to resume")
        return 1
    print(f"\n{len(ledger.answers())} replies\nwrote {ledger.path}\nwrote {ledger.traces_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
