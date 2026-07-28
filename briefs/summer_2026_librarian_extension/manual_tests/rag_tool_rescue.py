#!/usr/bin/env python3
"""Manual live probe: given a document search it may call, does the model use it, and does that help?

NOT a pytest test — it needs a running backend with a model loaded, so it lives here rather than in the
suite. It skips cleanly (exit 0, one line) when no backend answers, so it is safe to run blind.

This is the successor to `absent_fact.py`, which measured the failure this exists to fix: asked something
the retrieved documents were silent on, Qwen3.6-27B wrote a `<tool_call>` block as literal text roughly one
turn in three, because Raven's document search was something Raven had already run, not something the model
could call. The corpus and the question here are deliberately *identical* to that probe's, so the numbers
are comparable: same three hydrogen-electrolysis matches, same question about a stack they never mention.

What differs is everything below the question. `absent_fact.py` hand-builds one wire history and posts it
once; this drives the real `scaffold.ai_turn`, so the whole path is under test — the tool is advertised by
`llmclient.setup`, dispatched through `perform_tool_calls`, answered by `search_documents_wrapper` reading
the retriever out of `dyn.tool_context`, and fed back into the loop as a real `role="tool"` node. Nothing
here is faked except the retriever, because retrieval *quality* is not what is being measured and a stub
keeps this probe independent of raven-server.

Two scenarios, because the tool can help in two different ways and only one of them is a rescue:

    absent     the second search also finds nothing. The model should say so and stop. This is the path
               that could loop forever, so it is also where the round cap earns its keep.
    findable   the fact is in the corpus under wording the first query missed, and the model's own query
               reaches it. This is the rescue the tool was built for: an answer that was unreachable before.

Read both numbers per sample, not just the verdict:

    called       how many real `search_documents` nodes appeared — the behaviour `absent_fact.py` could
                 only watch leak out as literal text. Doubles as the round count for this tool, so a run
                 that sits at the cap means the model is rephrasing rather than concluding, which is a
                 wording problem and not a loop problem.
    leaked       did `<tool_call>` text survive into the reply anyway? This should be zero; anything else
                 means the tool is present but the model is not finding it.

First run, Qwen3.6-35B-A3B, three samples per scenario, backend defaults:

    absent      2/3 searched again; 3/3 answered correctly (said the stack is absent, offered the Vantaa-3
                figure they did have); peak 2 rounds, well under the cap
    findable    3/3 searched again; 3/3 recovered the fact that was unreachable before the tool existed
    both        0/6 leaked literal tool-call text

Against `absent_fact.py`'s recorded 3/9, that reads like the fix landing — but **the comparison is not yet
apples-to-apples**, and three things have to be said about it rather than implied:

  - *Different model.* That 3/9 was Qwen3.6-27B; this run is 35B-A3B. Re-run against the 27B before
    treating 0/6 as the same measurement.
  - *One temperature, six samples.* `absent_fact.py` exists partly because sampling at a single temperature
    once made a bad wording look like a fix. Its protocol — both temperatures, and read the reasoning
    length, not only the verdict — has not been applied here.
  - *The stub is lenient.* It matches any query containing "kelvin", so the `findable` sample that queried
    "Kelvin-3" succeeded on the stub's generosity rather than on a good query. Tightening it would measure
    query quality; leaving it loose measures tool *use*, which is what this probe is for.

**Unprompted finding: the models narrate tool calls they did not make.** Replies in the `absent` scenario
said "I checked the local document database and performed a web search" — including a sample that called no
tool at all, and one that ran two document searches and no web search. The answers were correct; the
account of how they were reached was not. Nothing here depends on that account, but the citation and
provenance work does, so it should not be taken on trust when it lands.

Usage:
    python rag_tool_rescue.py [base_url] [model] [samples_per_scenario]

Naming a model matters on a backend that loads them on demand: an idle LM Studio answers `/v1/models`
happily and then refuses to generate, so "the backend is up" is not the same question as "the backend can
answer". The gate below asks the second one.
"""

import re
import sys

from raven.librarian import chattree, chatutil, config as librarian_config, llmclient, scaffold

BASE = sys.argv[1] if len(sys.argv) > 1 else librarian_config.llm_backend_url
MODEL = sys.argv[2] if len(sys.argv) > 2 else None
N = int(sys.argv[3]) if len(sys.argv) > 3 else 3

if MODEL is not None:
    librarian_config.llm_model = MODEL  # `setup` reads it from here; naming it lets the backend load on demand

# Same three matches and the same question as `absent_fact.py`: relevant to the topic, silent on the
# question. Changing either would cost the comparison against that probe's recorded 3/9.
TOPIC_MATCHES = [{"document_id": "abstract_001.txt", "text": "Alkaline electrolysis remains the workhorse of industrial hydrogen production.", "score": 0.6, "offset": 0},
                 {"document_id": "vantaa3_stack_report.txt", "text": "The Vantaa-3 pressurized alkaline stack draws 41.7 kWh/kg under nominal load.", "score": 0.9, "offset": 0},
                 {"document_id": "abstract_002.txt", "text": "PEM electrolyzers offer faster load following at higher capital cost.", "score": 0.5, "offset": 0}]

# The document the first query misses and a targeted one finds. Its offset is nonzero on purpose: the
# formatter reports the span, and a plausible span is part of what `fetch_document` will later rely on.
KELVIN_MATCH = {"document_id": "kelvin7_commissioning.txt",
                "text": "Commissioning trials of the Kelvin-7 stack recorded a specific energy consumption of 47.2 kWh/kg at nominal load, falling to 45.8 kWh/kg after the electrode conditioning cycle.",
                "score": 0.92,
                "offset": 8431}

QUESTION = "What is the specific energy consumption of the Kelvin-7 stack?"


class StubRetriever:
    """Answers the automatic search with topic matches, and a model-authored query on its own terms.

    Duck-types `hybridir.HybridIR.query`, which is all scaffold and the tool entrypoint ever call. Standing
    in for the real retriever keeps this probe independent of raven-server (no embeddings, no spaCy) and
    keeps the measurement pointed at tool *use* rather than retrieval quality.
    """

    def __init__(self, fact_is_findable):
        self.fact_is_findable = fact_is_findable
        self.queries = []

    def query(self, q, k=10, return_extra_info=False):
        self.queries.append(q)
        if len(self.queries) == 1:  # the automatic pre-turn search, run with the user's own words
            return list(TOPIC_MATCHES)
        if self.fact_is_findable and "kelvin" in q.lower():
            return [KELVIN_MATCH]
        return []


def connect():
    """Return live `llm_settings`, or `None` if the backend cannot actually answer.

    Deliberately stronger than a liveness check. A backend that loads models on demand serves `/v1/models`
    and reports a flavour while holding no model at all, and then fails every generation with HTTP 400 — so
    a probe gated on reachability runs its whole sample budget producing identical errors. Sending one real
    (if tiny) request is the only question worth asking, and `prefill` is Raven's own way of asking it.
    """
    try:
        llm_settings = llmclient.setup(backend_url=BASE, quiet=True)
    except Exception as exc:  # noqa: BLE001 -- unreachable backend means "skip", not "crash"
        print(f"No LLM backend at {BASE} ({type(exc).__name__}); skipping.")
        return None
    probe = [chatutil.create_chat_message(llm_settings=llm_settings, role="user", text="ping")]
    if llmclient.prefill(llm_settings, probe, tools_enabled=False) is None:
        print(f"Backend at {BASE} is reachable but cannot generate (no model loaded?); skipping. "
              f"Name a model as the second argument to have it loaded on demand.")
        return None
    return llm_settings


def run_once(llm_settings, fact_is_findable):
    """One full AI turn against the live backend. Returns what the turn did, not what it said."""
    datastore = chattree.Forest()
    head = chatutil.factory_reset_datastore(datastore, llm_settings)
    head = scaffold.user_turn(llm_settings=llm_settings, datastore=datastore,
                              head_node_id=head, user_message_text=QUESTION)

    retriever = StubRetriever(fact_is_findable=fact_is_findable)
    final = scaffold.ai_turn(llm_settings=llm_settings, datastore=datastore, retriever=retriever,
                             head_node_id=head, tools_enabled=True, continue_=False,
                             docs_enabled=True, docs_query=QUESTION, docs_num_results=None,
                             speculate=False, markup=None,
                             on_docs_start=None, on_docs_done=None, on_prompt_ready=None,
                             on_llm_start=None, on_llm_progress=None, on_llm_done=None,
                             on_tools_start=None,
                             on_call_lowlevel_start=None, on_call_lowlevel_done=None,
                             on_tool_done=None, on_tools_done=None)

    # Walk the branch the turn built. Tool nodes are the evidence the model actually used the tool; the
    # literal-text check is the failure mode this whole exercise exists to retire.
    searches = 0
    node_id = final
    while node_id is not None:
        payload = datastore.get_payload(node_id)
        message = payload["message"]
        if message["role"] == "tool" and payload.get("generation_metadata", {}).get("function_name") == "search_documents":
            searches += 1
        node_id = datastore.get_parent(node_id)

    reply = chatutil.content_to_text(datastore.get_payload(final)["message"]["content"])
    return {"searches": searches,
            "leaked": bool(re.search(r"<tool_call>|<function=", reply)),
            "queries": retriever.queries[1:],  # the model's own, excluding the automatic first pass
            "reply": " ".join(reply.split())}


def main():
    llm_settings = connect()
    if llm_settings is None:
        return
    print(f"backend {BASE}, model {llm_settings.model}")

    for scenario in ("absent", "findable"):
        print(f"\n--- {scenario} @ {N} samples ---")
        used_the_tool = 0
        leaked = 0
        for i in range(N):
            got = run_once(llm_settings, fact_is_findable=(scenario == "findable"))
            used_the_tool += bool(got["searches"])
            leaked += got["leaked"]
            print(f"  [{i + 1}] called={got['searches']}  leaked={'YES' if got['leaked'] else 'no'}")
            if got["queries"]:
                print(f"      its own queries: {got['queries']}")
            print(f"      reply: {got['reply'][:200]!r}")
        print(f"  => used the tool {used_the_tool}/{N}; leaked literal tool-call text {leaked}/{N}")
        if scenario == "findable" and used_the_tool < N:
            print("     (a sample that never searched cannot have found the fact - check its reply)")


if __name__ == "__main__":
    main()
