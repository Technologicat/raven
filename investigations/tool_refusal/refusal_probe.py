#!/usr/bin/env python3
"""Live probe: past the tool-call cap, does telling the model so make it stop asking?

Raven's agent loop gives a turn `max_tool_call_rounds` of tool calls. Past that it injects a system notice
saying the budget is spent, keeps the tools in the schema, and answers any further call with an error
result (`chatutil.format_error_that_tools_are_spent`). Only if the model calls again anyway are the tools
withdrawn outright, which leaves it no move but to reply.

The question here is which of those two ends the turn. The refusal path is the cheap one - it leaves the
backend's cached prompt prefix intact - so it matters how often the model actually takes it.

NOT a pytest test: it needs a running backend with a model loaded. The mechanism itself is pinned by
`TestToolCallRoundCap` in `raven/librarian/tests/test_scaffold.py`; what cannot be unit-tested is whether a
real model heeds the notice, which is the whole question.

Usage:

    python refusal_probe.py <backend-url> [samples] [cap]

The retriever finds nothing, ever, so the model is pushed into the escalation on purpose by rephrasing a
search that cannot succeed. That is the failure the round cap was originally added for
(`briefs/librarian-extension/manual_tests/rag_tool_rescue.py`), and it is deliberately the
*hardest* case for the notice to land: the model is empty-handed, so it has every reason to keep trying.
The scenario that motivated the budget work - working through a list of documents one fetch at a time,
where the model has material in hand by the time the budget runs out - is a different shape and is not
measured here.
"""

import json
import pathlib
import sys
import threading

from raven.librarian import chattree, chatutil, config as librarian_config, llmclient, scaffold

BASE = sys.argv[1] if len(sys.argv) > 1 else librarian_config.llm_backend_url

QUESTION = ("What does the local document database say about the Kelvin-3 electrolysis stack's "
            "operating pressure? Check thoroughly, searching more than once if needed.")


class EmptyRetriever:
    """Finds nothing, ever - so the model keeps rephrasing and runs into the cap.

    Implements the retriever surface the librarian touches: `.query(...)`, plus the `documents` mapping and
    the `datastore_lock` guarding it.
    """

    def __init__(self):
        self.queries = []
        self.documents = {}
        self.datastore_lock = threading.Lock()

    def query(self, q, k=10, return_extra_info=False):
        self.queries.append(q)
        return []


def run_once(llm_settings):
    """One full AI turn against the live backend. Returns how the turn ended, not what it said."""
    datastore = chattree.Forest()
    head = chatutil.factory_reset_datastore(datastore, llm_settings)
    head = scaffold.user_turn(llm_settings=llm_settings, datastore=datastore,
                              head_node_id=head, user_message_text=QUESTION)

    retriever = EmptyRetriever()
    final = scaffold.ai_turn(llm_settings=llm_settings, datastore=datastore, retriever=retriever,
                             head_node_id=head, tools_enabled=True, continue_=False,
                             docs_enabled=True, docs_query=QUESTION, docs_num_results=None,
                             speculate=False, markup=None,
                             on_docs_start=None, on_docs_done=None, on_prompt_ready=None,
                             on_llm_start=None, on_llm_progress=None, on_llm_done=None,
                             on_tools_start=None,
                             on_call_lowlevel_start=None, on_call_lowlevel_done=None,
                             on_tool_done=None, on_tools_done=None)

    # Walk the branch the turn built.
    #
    # `rounds` has to be counted separately from `refused`, and conflating them is the trap: a turn with no
    # refusals may mean the notice landed, or may mean the model stopped searching of its own accord and
    # never reached the cap at all. Only a turn that *reached* the cap says anything about the notice.
    refused = 0
    rounds = 0
    reasoning = []
    node_id = final
    while node_id is not None:
        payload = datastore.get_payload(node_id)
        message = payload["message"]
        text = chatutil.content_to_text(message["content"])
        if message["role"] == "tool" and "budget for this reply is spent" in text:
            refused += 1
        if message["role"] == "assistant" and message.get("tool_calls"):
            rounds += 1
        if message["role"] == "assistant" and message.get("reasoning_content"):
            reasoning.append(message["reasoning_content"])
        node_id = datastore.get_parent(node_id)
    reasoning.reverse()  # the walk goes leaf-to-root

    reply = chatutil.content_to_text(datastore.get_payload(final)["message"]["content"])
    reached_the_cap = rounds >= librarian_config.max_tool_call_rounds
    return {"rounds": rounds,
            "reached_the_cap": reached_the_cap,
            "refused_calls": refused,
            # Only meaningful when the cap was reached; `None` means the question did not arise.
            "stopped_when_told": (refused == 0) if reached_the_cap else None,
            "reply_chars": len(reply.strip()),
            "reasoning": reasoning,
            "reply": reply}


def main():
    librarian_config.max_tool_call_rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    librarian_config.max_tool_call_refusal_rounds = 1

    try:
        llm_settings = llmclient.setup(backend_url=BASE, quiet=True)
    except Exception as exc:  # noqa: BLE001 -- unreachable backend means "skip", not "crash"
        print(f"No LLM backend at {BASE} ({type(exc).__name__}); skipping.")
        return

    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    cap = librarian_config.max_tool_call_rounds
    print(f"cap = {cap}, samples = {n}")
    results = []
    for i in range(n):
        result = run_once(llm_settings)
        results.append(result)
        print(f"sample {i}: " + json.dumps({k: v for k, v in result.items()
                                            if k not in ("reasoning", "reply")}), flush=True)

    # The reasoning traces are where the diagnosis lives, so they are written out rather than summarized.
    out_path = pathlib.Path(__file__).parent / f"traces-cap{cap}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ntraces -> {out_path}")

    at_cap = [r for r in results if r["reached_the_cap"]]
    heeded = sum(r["stopped_when_told"] for r in at_cap)
    empty = sum(r["reply_chars"] == 0 for r in results)
    print(f"reached the cap:                     {len(at_cap)}/{n}")
    if at_cap:
        print(f"  ...of those, stopped when told:    {heeded}/{len(at_cap)}")
        print(f"  ...of those, needed the withdrawal: {len(at_cap) - heeded}/{len(at_cap)}")
    print(f"empty replies:                       {empty}/{n}")


if __name__ == "__main__":
    main()
