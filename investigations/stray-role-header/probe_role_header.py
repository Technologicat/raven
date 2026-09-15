"""Does a fresh chat's reply sometimes open with a bare `assistant`? Run turns and keep everything.

Runs N turns of "Hi!" against the configured LLM backend, each in a fresh in-memory chat built from the
character card, with the settings Raven-librarian had when the leaks were seen: thinking on, internet tools
on, documents off. Optionally prefills the greeting branch first, as Librarian's idle prefill does.

Writes, per run:
  - `<out>.jsonl`: one line per turn — whether it leaked, every request body sent (the prefill included),
    and the whole `TurnRecord`.
  - `<out>.live.txt`: the streamed reasoning, content and tool calls, as they arrive, so a turn that runs
    away can be read before it returns.

Usage:
    python investigations/stray-role-header/probe_role_header.py -n 30 --out data/baseline
    python investigations/stray-role-header/probe_role_header.py -n 30 --prefill --out data/prefill
"""

import argparse
import json
import pathlib
import time

from raven.librarian import agent, chattree, chatutil, llmclient
from raven.librarian import config as librarian_config

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("-n", type=int, default=30, help="how many turns (default 30)")
parser.add_argument("--out", type=pathlib.Path, required=True, help="output path prefix; `.jsonl` and `.live.txt` are added")
parser.add_argument("--prefill", action="store_true", help="prefill the greeting branch before each turn, as Librarian's idle prefill does")
parser.add_argument("--no-thinking", action="store_true", help="turn thinking off")
args = parser.parse_args()
args.out.parent.mkdir(parents=True, exist_ok=True)

# The request bodies as posted. `TurnRecord.prompts` has the turn's histories, but not the prefill's, nor
# the other request fields.
captured = []
real_post = llmclient.requests.post
def capturing_post(url, *a, **kw):
    if url.endswith("/v1/chat/completions"):
        captured.append(json.loads(json.dumps(kw.get("json"), default=str)))
    return real_post(url, *a, **kw)
llmclient.requests.post = capturing_post

llm_settings = llmclient.setup(backend_url=librarian_config.llm_backend_url, quiet=True)
print(f"backend {llm_settings.backend_url}, model {getattr(llm_settings, 'model', '?')}, prefill {args.prefill}")

leaks = 0
with open(args.out.with_suffix(".jsonl"), "w", encoding="utf-8") as records, \
     open(args.out.with_suffix(".live.txt"), "w", encoding="utf-8") as live:
    for k in range(args.n):
        captured.clear()
        t0 = time.monotonic()
        datastore = chattree.Forest()
        head = chatutil.factory_reset_datastore(datastore, llm_settings)
        if args.prefill:
            history = chatutil.linearize_chat(datastore=datastore, node_id=head)
            tool_names = llmclient.maybe_tool_names_for_turn(llm_settings, documents_available=False, internet_available=True)
            llmclient.prefill(llm_settings, history, tools_enabled=True, tool_names=tool_names, datastore=datastore)
        record = agent.turn(llm_settings, "Hi!",
                            datastore=datastore, head_node_id=head,
                            thinking_enabled=not args.no_thinking,
                            internet_enabled=True,
                            docs_enabled=False,
                            on_progress=agent.stream_log(live, label=f"turn {k + 1}"))
        leaked = record.reply.lstrip().lower().startswith("assistant")
        leaks += leaked
        print(f"{k + 1}/{args.n} leaked={leaked} dt={time.monotonic() - t0:.1f}s reply={record.reply[:60]!r}")
        records.write(json.dumps({"k": k, "leaked": leaked, "requests": list(captured),
                                  "record": record.to_dict()}, default=str) + "\n")
        records.flush()
print(f"total: {leaks} of {args.n} replies opened with 'assistant'")
