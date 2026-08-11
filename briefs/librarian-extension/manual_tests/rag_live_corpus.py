#!/usr/bin/env python3
"""Manual live probe: do the document tools work against a *real* corpus and a real `HybridIR`?

NOT a pytest test — it needs a built RAG index (and, for the last phase, a backend with a model loaded), so
it lives here rather than in the suite. Each phase skips cleanly and says why, so it is safe to run blind.

The companion probe, `rag_tool_rescue.py`, measures whether the model *reaches for* the document tools. It
fakes the retriever on purpose, so that it measures tool use rather than retrieval quality. That leaves a
gap this probe exists to close: everything below the stub. A stub invents the `document_id`s it then
resolves, so it cannot possibly catch the two of them disagreeing — and if they disagree, `fetch_document`
answers "there is no document with that ID" for *every* call, forever, with no error anywhere. The tool
would look present and be dead.

Phases, cheapest first, each printing PASS / FAIL / SKIP:

    A  load        the real `HybridIR` over the configured docs dir and index
    B  round-trip  ids from `query` resolve in `documents` — the failure described above
    C  tools       `search_documents` and `fetch_document` through their real entrypoints and tool context
    D  labels      `label_documents` over real search hits: how many get a title rather than an ID alone
    E  budget      where `fetch_document` starts refusing as the conversation fills the window
    F  live turn   a real `ai_turn` against the real retriever, and whether the provenance list comes back
                   (repeatable: the failure it hunts turns up in some runs and not others)

Phases A–E need no LLM backend at all. Only F does.

**Scope note on E.** The hydrogen corpus is `raven-burstbib` output: one BibTeX record per file, a few
kilobytes each. Nothing in it is long enough to be truncated, so E can show that the budget does not
*refuse* a fetch it should serve, and cannot exercise middle-truncation on real data. That wants a corpus
of full papers (an arXiv download stash) to be indexed first.

Usage:
    python rag_live_corpus.py [base_url] [model] [phase F samples] [results file] [on|off|ab]

A *results file* is appended to, one JSON object per phase F sample, as each sample finishes. Give one for
any run worth keeping: samples pool across sessions and machines, and this is the only thing that survives
a reboot (`/tmp` is a ramdisk here). The last argument switches the spent-tools notice on or off, or `ab`
to alternate it between samples — which is the honest way to compare, since it pairs the two arms inside
one run rather than across runs that may differ in ways nobody wrote down.

Naming a model matters on a backend that loads them on demand: an idle LM Studio answers `/v1/models` and
then refuses to generate, so "the backend is up" is not the same question as "the backend can answer".
"""

import json
import pathlib
import re
import sys

from unpythonic import dyn, timer
from unpythonic.env import env

from raven.common import docextract

from raven.librarian import chattree, chatutil, config as librarian_config, hybridir, llmclient, scaffold

BASE = sys.argv[1] if len(sys.argv) > 1 else librarian_config.llm_backend_url
MODEL = sys.argv[2] if len(sys.argv) > 2 else None
N_LIVE = int(sys.argv[3]) if len(sys.argv) > 3 else 1  # phase F samples; the failure it hunts is intermittent
RESULTS_PATH = sys.argv[4] if len(sys.argv) > 4 else None  # append one JSON object per sample, for pooling later
NOTICE_MODE = sys.argv[5] if len(sys.argv) > 5 else "on"  # "on", "off", or "ab" to alternate between samples

if MODEL is not None:
    librarian_config.llm_model = MODEL  # `setup` reads it from here; naming it lets the backend load on demand

# Queries aimed at the hydrogen corpus this was written against. Several, because a single query that
# happens to return nothing would make phase B vacuously pass.
QUERIES = ["hydrogen production by anaerobic bacteria",
           "alkaline water electrolysis efficiency",
           "photocatalytic hydrogen evolution"]

results = []


def report(phase, ok, message):
    """Record and print one phase's verdict. `ok` may be `None` for a skip.

    Also kept for the recap at the end, because these lines do not survive on their own: `bm25s` draws a
    progress bar with carriage returns, which overwrites whatever shares its physical line. A verdict that
    scrolled past invisibly is worse than no verdict, since the run still reports itself as passing.
    """
    verdict = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    results.append((phase, ok, message))
    print(f"[{verdict}] {phase}: {message}")


# --------------------------------------------------------------------------------
# A. Load the real retriever

def load_retriever():
    """Open the configured document store. Returns the retriever, or `None` if there is nothing to open.

    `hybridir.setup` rescans the documents directory, which is how it notices edits made while the app was
    closed. That is an mtime pass, not a reindex, so an unchanged corpus costs seconds rather than the hour
    a rebuild would — but it is worth knowing that this probe is not read-only on the index.
    """
    docs_dir = pathlib.Path(librarian_config.llm_docs_dir).expanduser().resolve()
    db_dir = pathlib.Path(librarian_config.llm_database_dir).expanduser().resolve()
    if not docs_dir.is_dir():
        report("A load", None, f"no documents directory at {docs_dir}")
        return None
    try:
        with timer() as tim:
            retriever, _scanner = hybridir.setup(docs_dir=docs_dir,
                                                 recursive=librarian_config.llm_docs_dir_recursive,
                                                 db_dir=db_dir,
                                                 exts=librarian_config.llm_docs_exts,
                                                 callback=docextract.extract_text,
                                                 embedding_model_name=librarian_config.qa_embedding_model,
                                                 local_model_loader_fallback=True)
    except Exception as exc:  # noqa: BLE001 -- a missing index or an absent server means "skip", not "crash"
        report("A load", None, f"could not open the document store ({type(exc).__name__}: {exc})")
        return None
    with retriever.datastore_lock:
        n_documents = len(retriever.documents)
    if not n_documents:
        report("A load", None, "the document store is empty")
        return None
    report("A load", True, f"{n_documents} documents, opened in {tim.dt:0.1f} s")
    return retriever


# --------------------------------------------------------------------------------
# B. The round-trip that a stub retriever cannot test

def check_round_trip(retriever):
    """Every `document_id` a search reports must resolve back to a document. Returns the search hits.

    This is the whole reason the probe exists. `HybridIRFileSystemEventHandler` derives an id from a path
    and `HybridIR.query` reports it; `fetch_document` looks that id up in `retriever.documents`. Nothing
    forces those two to agree, and if they ever stop agreeing the symptom is not an exception — it is
    `fetch_document` politely denying that any document exists, on every call, with the model left to
    conclude the database is empty.
    """
    hits = []
    for query in QUERIES:
        hits.extend(retriever.query(query, k=5, return_extra_info=False))
    if not hits:
        report("B round-trip", None, f"no matches for any of {len(QUERIES)} queries; corpus is off-topic for this probe")
        return []
    document_ids = list(dict.fromkeys(hit["document_id"] for hit in hits))
    unresolved = [document_id for document_id in document_ids
                  if llmclient.document_text(retriever, document_id) is None]
    if unresolved:
        report("B round-trip", False,
               f"{len(unresolved)} of {len(document_ids)} ids do not resolve, e.g. {unresolved[:3]} "
               f"-- fetch_document would refuse every one of these")
        return hits
    report("B round-trip", True, f"all {len(document_ids)} ids from {len(hits)} matches resolve")
    return hits


# --------------------------------------------------------------------------------
# C. The tool entrypoints, through their real dispatch path

def check_tools(retriever, hits):
    """Call the document tools as the model would, with a real tool context around them."""
    if not hits:
        report("C tools", None, "no matches to work from")
        return
    tool_context = scaffold.make_tool_context(llm_settings=None, retriever=retriever)
    document_id = hits[0]["document_id"]
    with dyn.let(tool_context=tool_context):
        search_output, search_metadata = llmclient.search_documents_wrapper(QUERIES[0])
        if not search_metadata.get("grounding"):
            report("C tools", False, f"search_documents declared no grounding: {search_output[:120]!r}")
            return
        if document_id not in search_output:
            report("C tools", False, "search_documents output does not name the document it matched")
            return

        tool_context.llm_settings = _settings_for_budget()
        fetch_output, fetch_metadata = llmclient.fetch_document_wrapper(document_id)
        if not fetch_metadata.get("grounding"):
            report("C tools", False, f"fetch_document declared no grounding: {fetch_output[:160]!r}")
            return
        span_output, _ = llmclient.fetch_document_wrapper(document_id, offset=10, length=50)
    if "characters 10 to 60" not in span_output:
        report("C tools", False, f"a requested span was not honoured: {span_output[:160]!r}")
        return
    report("C tools", True,
           f"search named {len(search_metadata.get('document_ids', []))} documents; "
           f"fetch returned {len(fetch_output)} characters, and a 50-character span was honoured")


def _settings_for_budget(context_length=32768):
    """The subset of `llm_settings` the budget helpers read, without needing a backend to build it."""
    return env(context_length=context_length,
               tokens_per_character=0.27,
               tokenizer=None,
               backend_flavor="lmstudio")


# --------------------------------------------------------------------------------
# D. Labels, over documents that actually exist

def check_labels(retriever, hits):
    """How many real search hits get a label a model could triage on, rather than an ID alone."""
    if not hits:
        report("D labels", None, "no matches to label")
        return
    entries = [{"document_id": document_id}
               for document_id in dict.fromkeys(hit["document_id"] for hit in hits)]
    labelled = llmclient.label_documents(retriever, entries)
    with_labels = [entry for entry in labelled if entry["label"]]
    report("D labels", bool(with_labels),
           f"{len(with_labels)} of {len(labelled)} labelled; e.g. "
           f"{[entry['label'][:70] for entry in with_labels[:2]]}")


# --------------------------------------------------------------------------------
# E. Where the budget starts refusing

def check_budget(retriever, hits):
    """Report the fill level at which `fetch_document` stops serving. Short items cannot exercise truncation."""
    if not hits:
        report("E budget", None, "no matches to fetch")
        return
    document_id = hits[0]["document_id"]
    length = len(llmclient.document_text(retriever, document_id) or "")
    settings = _settings_for_budget()
    served, refused = [], []
    for fill in (0.0, 0.25, 0.5, 0.7, 0.9):
        used_tokens = int(fill * settings.context_length)
        tool_context = scaffold.make_tool_context(llm_settings=settings, retriever=retriever)
        tool_context.used_tokens = used_tokens
        with dyn.let(tool_context=tool_context):
            output, metadata = llmclient.fetch_document_wrapper(document_id)
        (served if metadata.get("grounding") else refused).append(f"{fill:.0%}")
    report("E budget", bool(served),
           f"document is {length} characters; served at {served or 'no fill level'}, "
           f"refused at {refused or 'none'} "
           f"(items this short cannot exercise middle-truncation -- that wants a corpus of full papers)")


# --------------------------------------------------------------------------------
# F. A real turn, with a real retriever behind the tools

def connect():
    """Return live `llm_settings`, or `None` if the backend cannot actually answer.

    Deliberately stronger than a liveness check: a backend that loads models on demand serves `/v1/models`
    while holding no model, then fails every generation. `prefill` is Raven's own way of asking the second
    question, so one tiny real request settles it.
    """
    try:
        llm_settings = llmclient.setup(backend_url=BASE, quiet=True)
    except Exception as exc:  # noqa: BLE001 -- unreachable backend means "skip", not "crash"
        print(f"    (no LLM backend at {BASE}: {type(exc).__name__})")
        return None
    probe = [chatutil.create_chat_message(llm_settings=llm_settings, role="user", text="ping")]
    if llmclient.prefill(llm_settings, probe, tools_enabled=False) is None:
        print(f"    (backend at {BASE} cannot generate; name a model as the second argument)")
        return None
    return llm_settings


def _ai_turn(llm_settings, datastore, retriever, head, question):
    """One AI turn with every callback off, capturing the wire history that was actually sent."""
    seen = {}

    def on_prompt_ready(history):
        seen["history"] = history

    final = scaffold.ai_turn(llm_settings=llm_settings, datastore=datastore, retriever=retriever,
                             head_node_id=head, tools_enabled=True, continue_=False,
                             docs_enabled=True, docs_query=question, docs_num_results=None,
                             speculate=False, markup=None,
                             on_docs_start=None, on_docs_done=None, on_prompt_ready=on_prompt_ready,
                             on_llm_start=None, on_llm_progress=None, on_llm_done=None,
                             on_tools_start=None, on_call_lowlevel_start=None,
                             on_call_lowlevel_done=None, on_tool_done=None, on_tools_done=None)
    return final, seen.get("history", [])


def _append_result(record):
    """Append one sample to the results file, flushed immediately.

    Written per sample rather than per run because these runs are long, get interrupted, and are worth
    pooling afterwards. A batch that dies at sample 9 of 12 should still contribute its first 8.
    """
    if RESULTS_PATH is None:
        return
    with open(RESULTS_PATH, "a", encoding="utf-8") as results_file:
        results_file.write(json.dumps(record, sort_keys=True) + "\n")
        results_file.flush()


def _make_datastore(sample_index, notice_enabled):
    """A datastore for one sample: persistent beside the results file when there is one, else in memory."""
    if RESULTS_PATH is None:
        return chattree.Forest()
    results_path = pathlib.Path(RESULTS_PATH)
    arm = "notice" if notice_enabled else "control"
    chat_path = results_path.with_name(f"{results_path.stem}-{sample_index:03d}-{arm}.json")
    return chattree.PersistentForest(chat_path, autosave=False)


def check_live_turn(retriever, hits, notice_enabled=True, sample_index=0):
    """Two turns against the live model: does it use the tools, and does the provenance list come back?

    Two, because the second is where the interesting thing happens. The automatic search's matches are
    injected for one turn and dropped, so at turn two the material is gone and only the IDs survive — which
    is exactly what `list_consulted_documents` is for. A single-turn probe cannot see it at all.
    """
    if not hits:
        report("F live turn", None, "no matches to ask about")
        return
    llm_settings = connect()
    if llm_settings is None:
        report("F live turn", None, "no usable LLM backend")
        return

    # The A/B switch. Silencing the formatter is enough: `build_turn_prompt` appends whatever it returns,
    # and an empty string adds no system line at all. The override is a field on this run's own settings,
    # so nothing process-wide changes and there is no original to put back.
    if not notice_enabled:
        llm_settings.formatters.notice_that_tools_are_spent = lambda: ""
    try:
        # A `PersistentForest` rather than an in-memory `Forest`, when there is somewhere to put it. The
        # whole conversation is then on disk in Raven's own format - every message, the reasoning that
        # never reaches `content`, the tool nodes and their metadata - so a later analysis is not limited
        # to whatever this probe thought to summarize tonight. These runs are slow enough that throwing the
        # evidence away and re-running would be the expensive mistake.
        datastore = _make_datastore(sample_index, notice_enabled)
        head = chatutil.factory_reset_datastore(datastore, llm_settings)
        head = scaffold.user_turn(llm_settings=llm_settings, datastore=datastore,
                                  head_node_id=head, user_message_text=QUERIES[0])
        head, _ = _ai_turn(llm_settings, datastore, retriever, head, QUERIES[0])

        follow_up = "Which of those documents said that, and what else does it say?"
        head = scaffold.user_turn(llm_settings=llm_settings, datastore=datastore,
                                  head_node_id=head, user_message_text=follow_up)
        final, history = _ai_turn(llm_settings, datastore, retriever, head, follow_up)
    finally:
        if isinstance(datastore, chattree.PersistentForest):
            datastore.save()

    wire = "\n".join(chatutil.content_to_text(message.get("content")) for message in history)
    listed = "consulted" in wire
    reply = chatutil.content_to_text(datastore.get_payload(final)["message"]["content"])
    leaked = bool(re.search(r"<tool_call>|<function=", reply))

    # Walk the branch the second turn built. The shape matters as much as the counts: an empty reply is
    # ambiguous on its own (a model that said nothing, a reply that went out as reasoning, or a turn that
    # ended on a tool node), and those want different fixes.
    tool_calls = {}
    rounds = {"n": 0}
    shape = []
    node_id = final
    while node_id is not None:
        payload = datastore.get_payload(node_id)
        message = payload["message"]
        role = message["role"]
        if role == "tool":
            name = payload.get("generation_metadata", {}).get("function_name", "?")
            tool_calls[name] = tool_calls.get(name, 0) + 1
        n_requested = len(message.get("tool_calls") or [])
        if n_requested:
            rounds["n"] += 1  # a *round* is one assistant message asking for tools, however many it asks for
        shape.append(f"{role}"
                     f"(text {len(chatutil.content_to_text(message.get('content')))}"
                     f", reasoning {len(message.get('reasoning_content') or '')}"
                     + (f", requested {n_requested}" if n_requested else "") + ")")
        node_id = datastore.get_parent(node_id)
    shape.reverse()

    final_payload = datastore.get_payload(final)
    generation_metadata = final_payload.get("generation_metadata") or {}
    report("F live turn", not leaked and bool(reply.strip()),
           f"{rounds['n']} tool rounds (cap {librarian_config.max_tool_call_rounds}), "
           f"calls {tool_calls or 'none'}; provenance list injected: {'yes' if listed else 'NO'}; "
           f"literal tool-call text leaked: {'YES' if leaked else 'no'}; "
           f"final node status {generation_metadata.get('status', '?')}, "
           f"grounded {generation_metadata.get('grounded', '(not recorded)')}; "
           f"reply: {' '.join(reply.split())[:160]!r}")
    print("    branch: " + " -> ".join(shape[-8:]))

    # When the reply is empty, what the model *did* emit is the whole story. Identical reasoning across
    # rounds means a loop (the cap doing its job); a one-off means something else stopped the generation.
    reasonings = []
    node_id = final
    while node_id is not None:
        payload = datastore.get_payload(node_id)
        if payload["message"]["role"] == "assistant":
            reasonings.append(payload["message"].get("reasoning_content") or "")
        node_id = datastore.get_parent(node_id)
    reasonings = [reasoning for reasoning in reversed(reasonings) if reasoning]
    if reasonings:
        distinct = len(set(reasonings))
        print(f"    {len(reasonings)} assistant turns emitted reasoning, {distinct} distinct")
        print(f"    last reasoning: {' '.join(reasonings[-1].split())[:400]!r}")

    _append_result({"sample": sample_index,
                    "notice": notice_enabled,
                    "model": llm_settings.model,
                    "answered": bool(reply.strip()),
                    "leaked": leaked,
                    "rounds": rounds["n"],
                    "cap": librarian_config.max_tool_call_rounds,
                    "calls": tool_calls,
                    "provenance_list_injected": listed,
                    "reasoning_turns": len(reasonings),
                    "reasoning_distinct": len(set(reasonings)),
                    "reply_characters": len(reply)})


# --------------------------------------------------------------------------------

def main():
    print(f"docs dir {librarian_config.llm_docs_dir}\nindex    {librarian_config.llm_database_dir}\n")
    retriever = load_retriever()
    if retriever is None:
        return
    hits = check_round_trip(retriever)
    check_tools(retriever, hits)
    check_labels(retriever, hits)
    check_budget(retriever, hits)
    # Resume where a previous run stopped. The results file is the ledger: one line per finished sample,
    # so its length *is* the count of work already done, and re-running the same command continues instead
    # of starting over. These runs take an hour and the machines reboot; a batch that has to restart from
    # zero is a batch that never finishes.
    done = 0
    if RESULTS_PATH is not None and pathlib.Path(RESULTS_PATH).exists():
        done = sum(1 for line in pathlib.Path(RESULTS_PATH).read_text(encoding="utf-8").splitlines() if line.strip())
        if done:
            print(f"\nresuming: {done} sample(s) already in {RESULTS_PATH}")
    for sample in range(done, N_LIVE):
        # The arm follows from the sample index, so resuming keeps the two arms balanced rather than
        # restarting the alternation and over-sampling whichever one comes first.
        notice_enabled = {"on": True, "off": False}.get(NOTICE_MODE, sample % 2 == 0)
        if N_LIVE > 1:
            print(f"\n--- live turn, sample {sample + 1} of {N_LIVE}"
                  f" ({'notice' if notice_enabled else 'control'}) ---")
        check_live_turn(retriever, hits, notice_enabled=notice_enabled, sample_index=sample)

    print("\n" + "=" * 78 + "\nrecap\n" + "=" * 78)
    for phase, ok, message in results:
        verdict = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        print(f"[{verdict}] {phase}: {message}")
    failed = [phase for phase, ok, _ in results if ok is False]
    print(f"\n{len(failed)} failed, "
          f"{len([1 for _, ok, _ in results if ok is True])} passed, "
          f"{len([1 for _, ok, _ in results if ok is None])} skipped")
    if failed:
        print("  failed:", ", ".join(failed))


if __name__ == "__main__":
    main()
