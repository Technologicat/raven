"""Scripting surface over the agent loop: run a turn, and get back what happened.

`scaffold.ai_turn` is built for a live frontend — fifteen mandatory callbacks, and a node id as the return
value. A script wants neither. It will never draw a progress bar, and the node id is the start of a branch
walk it then has to write itself: count the tool nodes by name, count the rounds, collect the reasoning that
never reached `content`, find the reply. That walk is the actual result of a turn, and every probe that has
needed it has written it out longhand, differently.

So this module is the same loop with the events turned inside out: no callbacks at all, and a `TurnRecord`
describing what the turn did. The loop itself is not reimplemented — `turn` calls `ai_turn` and reads the
branch afterwards, which is what makes it the same behaviour rather than a replica of it.

Vocabulary, because hand-rolled walks get this wrong: a **round** is one assistant message asking for tools,
however many calls it asks for; a **call** is one tool invocation. A model asking for three searches in one
message has taken one round and made three calls.

This is a programming library and not a product: what it offers is programmatic access to Raven's own
corpus, chattree and provenance machinery. It is deliberately not a generic agent harness — no plugin
system, no workflow DSL, no orchestration layer.

**Nothing here overwrites anything, and that is worth knowing up front**, because a scripting surface over
an LLM is usually built on a transcript, where a retry has nowhere to put the attempt it replaces. The
chat is a tree. A retry, a reroll, or a second phrasing of the same question is a new branch off the same
parent, so the failed attempt and the one that worked are both in the chat afterwards — as are all four
samples of a turn that was sampled four times, side by side under the message that prompted them. Nothing
has to be switched on for that, and there is no separate place for the run's history to live: a batch's
whole record of what it did is a chat, openable in Librarian and readable by a person.

`turn`'s docstring carries worked examples. The *executable* ones are in
`raven/librarian/tests/test_agent.py`, which is the better place to look for a pattern this docstring does
not cover: it drives the real loop against a faked backend, so every example in it is one CI runs.

Two properties worth knowing before scripting against it:

  - **The network is off by default.** `internet_enabled=False`, so `websearch` and `webfetch` are not
    offered. A probe that silently reaches the internet is the more expensive mistake, and turning it on is
    one keyword.
  - **Per-run configuration lives on `llm_settings`.** To A/B a wording, assign to `settings.formatters`
    (see `chatutil.default_formatters`) on the settings object this run uses. Nothing here is process-wide,
    so two runs in one process cannot contaminate each other.
"""

__all__ = ["TurnRecord", "describe_turn", "turn"]

import logging
logger = logging.getLogger(__name__)

import dataclasses

from collections.abc import Mapping
from typing import TYPE_CHECKING

from unpythonic import sym
from unpythonic.env import env

from . import chattree
from . import chatutil
from . import scaffold

if TYPE_CHECKING:
    from . import hybridir

# `docs_query`'s default: search with the user's own message, which is what the apps do. Distinct from
# `None`, which means "run no automatic search" — a turn that legitimately offers the document tools without
# having searched first.
from_user_message = sym("from_user_message")


@dataclasses.dataclass(frozen=True)
class TurnRecord:
    """What one assistant turn did.

    Frozen, and the odd one out against this codebase's `env` habit: a probe asserting on `rounds` wants a
    mistyped field to fail loudly rather than to read as `None`.

    Immutability stops at the fields. `messages` and `prompts` hold the datastore's own dicts rather than
    copies — a turn's tool results can be document-sized, and copying them to enforce a promise Python
    cannot enforce deeply anyway would be a real cost for a nominal gain. Read them; do not edit them.

    Fields:

    `datastore`: The chat this turn happened in. Carried so the record is a complete handle on where the
                 conversation now stands: `turn(..., datastore=record.datastore,
                 head_node_id=record.head_node_id)` continues it, whether or not the caller was the one who
                 built the datastore — the one-liner form builds its own, and without this there would be
                 no way back to it.

    `head_node_id`: The chat node the turn ended on — pass it as the next turn's `head_node_id`.

                    Any *other* node of the same datastore works there too, which is what makes this a
                    branching chat rather than a transcript: continuing from a node this turn passed
                    through leaves the branch just taken intact beside the new one, so a script can try
                    several continuations of the same prefix and keep them all.

    `node_ids`: The nodes this turn added, oldest first, ending with `head_node_id`.

    `messages`: Their messages, in the same order, in Raven's internal format (`role`, `content` as a list
                of typed parts, optionally `reasoning_content` and `tool_calls`). Anything this record does
                not summarize is in here, so a caller with an unanticipated question walks a list rather
                than the tree.

    `reply`: The text a user would read: the final message's text with the persona prefix removed, as both
             frontends render it, if the turn ended on an assistant message — and `""` otherwise. Read
             `messages[-1]` for the stored form.

             Empty is a real answer and a diagnostic one — a model that said nothing, a reply that went out
             as reasoning, or a turn that ended on a tool node. `messages` distinguishes them.

    `reasoning`: The reasoning traces the assistant emitted, in order, skipping messages that emitted none.
                 Identical traces across rounds are the signature of a loop; a single long one is not.

    `rounds`: How many rounds this turn took (see the vocabulary note above). Compare against
              `librarian_config.max_tool_call_rounds` to tell "the model was satisfied" from "the cap
              stopped it".

    `tool_calls`: How many calls each tool received, by name. Empty when the model called nothing.

    `grounded`: What the final reply recorded about having material to stand on: `True`/`False`, or `None`
                for "not recorded", which is what a turn with the documents switched off and no attachment
                stores. Not a verdict on the reply's accuracy — it says whether anything was retrieved.

    `generation`: The final reply's generation metadata — `model`, `n_tokens`, `dt` — or **`None` when
                  Raven wrote that message rather than the model.** A turn never raises on a backend
                  failure: it materializes the failure as an assistant message, which is right for a person
                  (it is visible and rerollable) and a trap for an unattended batch, where a dead backend
                  otherwise yields a run of plausible-looking replies that all say the same thing. This is
                  the field that distinguishes them, and it is the one to check before trusting a batch.

    `prompts`: The wire histories actually sent, one per model call, so `prompts[-1]` is the one that
               produced `reply`. This is the assembled prompt including the per-turn injects, after
               attachments are resolved — what a script asserting on "what was actually sent" needs.
               Empty when the record was reconstructed from a stored branch, which cannot know it.
    """
    datastore: chattree.Forest
    head_node_id: str
    node_ids: tuple[str, ...]
    messages: tuple[dict, ...]
    reply: str
    reasoning: tuple[str, ...]
    rounds: int
    tool_calls: Mapping[str, int]
    grounded: bool | None
    generation: dict | None
    prompts: tuple[list[dict], ...]


def describe_turn(datastore: chattree.Forest,
                  head_node_id: str,
                  since_node_id: str | None = None,
                  prompts: tuple[list[dict], ...] = ()) -> TurnRecord:
    """Read a branch and summarize it as a `TurnRecord`. The walk `turn` performs, exposed on its own.

    `head_node_id`: The last node of the span to read.

    `since_node_id`: The node the span starts *after* — the HEAD the turn began from. `None` reads all the
                     way to the root, which for a chat containing several turns totals them all: the
                     round and call counts then answer "on this branch" rather than "in this turn". That
                     is a different question, and rarely the one being asked.

    `prompts`: What was sent, if known. There is no way to recover it from stored nodes, so a record built
               from a datastore alone leaves it empty.

    Useful on its own for reading a `chattree.PersistentForest` written by an earlier run: a batch that
    saved its conversations can be analyzed afterwards with the same counting the live path uses, rather
    than with a second walk that has to agree with it.
    """
    node_ids = []
    node_id = head_node_id
    while node_id is not None and node_id != since_node_id:
        node_ids.append(node_id)
        node_id = datastore.get_parent(node_id)
    node_ids.reverse()

    payloads = [datastore.get_payload(node_id) for node_id in node_ids]
    messages = [payload["message"] for payload in payloads]

    rounds = 0
    tool_calls: dict[str, int] = {}
    reasoning = []
    for payload in payloads:
        message = payload["message"]
        if message.get("tool_calls"):
            rounds += 1  # one assistant message asking for tools is one round, however many it asks for
        if message["role"] == "tool":
            name = (payload.get("generation_metadata") or {}).get("function_name", "?")
            tool_calls[name] = tool_calls.get(name, 0) + 1
        if message["role"] == "assistant" and message.get("reasoning_content"):
            reasoning.append(message["reasoning_content"])

    ended_on_a_reply = bool(messages) and messages[-1]["role"] == "assistant"
    if ended_on_a_reply:
        # The persona prefix comes off, as it does in both frontends: it is part of how the message is
        # stored, not part of what was said. The *stored* persona rather than the session's, since a chat
        # can hold nodes generated under a different character.
        reply = chatutil.remove_persona_from_start_of_line(
            persona=(payloads[-1].get("general_metadata") or {}).get("persona"),
            text=chatutil.content_to_text(messages[-1]["content"]))
    else:
        reply = ""
    # `None` rather than `{}` when there is none: a message Raven authored has no generation to describe,
    # and that absence is what a batch reads to tell it from a real reply.
    generation = payloads[-1].get("generation_metadata") if ended_on_a_reply else None
    grounded = (generation or {}).get("grounded")

    return TurnRecord(datastore=datastore,
                      head_node_id=head_node_id,
                      node_ids=tuple(node_ids),
                      messages=tuple(messages),
                      reply=reply,
                      reasoning=tuple(reasoning),
                      rounds=rounds,
                      tool_calls=tool_calls,
                      grounded=grounded,
                      generation=generation,
                      prompts=tuple(prompts))


def turn(llm_settings: env,
         user_message_text: str | None = None,
         staged_images: list[env] | None = None,
         staged_files: list[env] | None = None,
         datastore: chattree.Forest | None = None,
         head_node_id: str | None = None,
         retriever: "hybridir.HybridIR | None" = None,
         internet_enabled: bool = False,
         docs_enabled: bool = True,
         docs_query: str | None | sym = from_user_message,
         docs_num_results: int | None = None,
         continue_: bool = False,
         markup: str | None = None) -> TurnRecord:
    """Run one assistant turn — the full agent loop — and return what it did.

    The one-liner form starts a fresh conversation in a throwaway in-memory datastore:

        record = agent.turn(llm_settings, "What is the Kelvin-7 stack's energy consumption?")

    A script that wants to look at the answer before deciding what to ask next continues from the record,
    which carries both halves of where the conversation stands:

        record = agent.turn(llm_settings, "What is the Kelvin-7 stack's energy consumption?",
                            retriever=retriever)
        if not record.tool_calls:
            record = agent.turn(llm_settings, "Please search the documents before answering.",
                                datastore=record.datastore, head_node_id=record.head_node_id,
                                retriever=retriever)

    Branching works the same way, since any node of that datastore is a valid `head_node_id`: asking a
    second question from `record.node_ids[0]` rather than from `record.head_node_id` grows a sibling branch
    and leaves the first one intact, so several continuations of one prefix can be compared afterwards.

    To interrogate the corpus you actually indexed, from the chat you actually use. One question is
    something to type into Librarian; two hundred of them out of a file is what this is for:

        llm_settings = llmclient.setup()
        datastore, state = appstate.load(llm_settings,
                                         librarian_config.llm_datastore_file,
                                         librarian_config.llm_state_file,
                                         autosave=False)
        retriever, _scanner = hybridir.setup(...)  # over the configured docs dir and index; see its docstring
        for question in questions:
            record = agent.turn(llm_settings, question, datastore=datastore,
                                head_node_id=state["new_chat_HEAD"], retriever=retriever)
            write_result(question, record.reply, record.grounded, dict(record.tool_calls))

    Four things in that loop are the point of writing it this way:

      - **`appstate.load`, rather than a `chattree.PersistentForest` over the configured path.** It
        validates the stored HEAD, refreshes the system prompt and greeting, migrates older formats, and
        wires the sidecar reader the attachment GC needs — none of which a bare constructor does.
      - **`autosave=False`**, so an interrogation run writes nothing. `datastore.save()` keeps it if you
        want it kept, but only when Librarian is not open on the same file: of two writers, the one who
        saved first loses.
      - **`state["new_chat_HEAD"]` — the greeting — and not `state["HEAD"]`.** Every question then branches
        off the same point, so question 2 is not answered in the context of question 1's reply, and none of
        them lands on top of the conversation the user was in the middle of. All the threads are in the one
        chat, which is what the tree is for: open Librarian afterwards and they are there to read.
      - **The retriever is opened once**, outside the loop. Opening it loads the index and the embedding
        model, which is the expensive part and is not per-question.

    Such a run is an overnight job on a local model, which is unattended, and two things follow that are
    easy to leave until the morning after:

      - **Write each result as it finishes, and make the results file the ledger.** Its length is then the
        count already done, so re-running the same command continues instead of starting over. A batch that
        dies at question 140 should contribute its first 139 rather than nothing.
      - **Record `record.generation is None` per question, and re-run those.** A turn does not raise when
        the backend fails — the failure arrives as an assistant message. Overnight, a backend that stopped
        answering at question 12 yields 188 replies that read like replies, and a run that looks complete.
        That predicate is what says the model did not write one, so the morning's second pass can be over
        the failures alone rather than over everything. Re-running a question grows a new branch beside the
        failed attempt rather than replacing it, so what went wrong stays readable in the chat.

    To keep a run without touching the real chat, build the datastore yourself and make it file-backed —
    that is also what attaching anything requires:

        datastore = chattree.PersistentForest(path)
        record = agent.turn(llm_settings, "First question", datastore=datastore, retriever=retriever)
        record = agent.turn(llm_settings, "Follow-up", datastore=datastore,
                            head_node_id=record.head_node_id, retriever=retriever)
        datastore.save()

    `llm_settings`: From `llmclient.setup` (which asks the backend what it has) or `llmclient.configure`
                    (which is told). This is also where per-run overrides live — see `settings.formatters`.

    `user_message_text`: Posted as the user's message before the assistant's turn, via `scaffold.user_turn`.
                         `None` runs the assistant's turn against the branch as it stands, which is what a
                         reroll or a continuation does.

    `staged_images`, `staged_files`: Images and documents attached to that message; see `scaffold.user_turn`
                                     for the shape of an entry. **Either one requires a file-backed
                                     `datastore`**: an attachment is stored as a content-addressed sidecar
                                     file beside the datastore, and the in-memory default has nowhere to
                                     put it. Refused up front rather than partway through storing.

                                     `staged_images` additionally needs a model that accepts image input,
                                     and is refused when `llm_settings.model_is_vlm` says otherwise —
                                     `False` only, since `None` means the backend did not say. A document
                                     needs no such capability: its text is folded into the prompt.

    `datastore`: Where the conversation lives. `None` (default) builds a throwaway `chattree.Forest`, which
                 is in-memory and costs nothing. Pass a `chattree.PersistentForest` to keep the run: the
                 whole conversation then lands on disk in Raven's own format — every message, the reasoning
                 that never reached `content`, the tool nodes and their metadata — so a later analysis is
                 not limited to what the script thought to summarize at the time. It is also what attaching
                 anything requires, per above.

    `head_node_id`: Where in the conversation to continue from. `None` starts a fresh conversation, which
                    means a factory reset of `datastore` — system prompt, character card and greeting. That
                    deletes whatever was there, so it is refused on a non-empty datastore: pass a head, or
                    call `chatutil.factory_reset_datastore` yourself if wiping it is what you meant.

    `retriever`: A `hybridir.HybridIR` over the document corpus, or anything with its `query` method (a
                 stub keeps a probe independent of raven-server when retrieval quality is not what is being
                 measured). `None` means no documents: no automatic search, and the document tools are not
                 offered.

    `internet_enabled`: Whether `websearch` and `webfetch` are offered. **`False` by default**, unlike the
                        apps — a run with tools enabled performs *real* calls, and a probe that reaches the
                        network without having asked to is the more expensive mistake.

    `docs_enabled`: Whether the documents are in play at all. `False` withdraws the document tools and
                    suppresses the automatic search whatever `docs_query` says.

    `docs_query`: What to search the corpus for before the model runs. Defaults to `user_message_text` when
                  there is a `retriever`, which is what the apps do. `None` runs no automatic search while
                  still offering the document tools — the shape a continuation turn has, and the control
                  arm for measuring what the automatic search is worth.

    `docs_num_results`: How many matches the automatic search returns, at most. `None` takes the default.

    `continue_`: Continue the incomplete assistant message at `head_node_id` instead of writing a new one.

    `markup`: How to mark thought blocks in the stored reply text: `"ansi"`, `"markdown"`, or `None` to
              keep them as they arrived. `None` is what a script wants.

    Returns a `TurnRecord`, which see. A backend failure is not raised: `ai_turn` materializes it as an
    assistant message, so the record comes back with that text as its `reply` — the same thing a user
    would see, and rerollable in the same way.
    """
    if datastore is None:
        datastore = chattree.Forest()
    if (staged_images or staged_files) and not hasattr(datastore, "store_sidecar"):
        raise ValueError("agent.turn: attachments need a datastore with a sidecar store — a "
                         "`chattree.PersistentForest`, not the in-memory `Forest` this defaults to.")
    if staged_images and llm_settings.model_is_vlm is False:
        # Only on a confirmed `False`: the tri-state's `None` is "the backend does not say", and refusing
        # on that would block every backend that reports nothing. Librarian's attach button reads it the
        # same way. A batch feeding page images to a text-only model would otherwise pay for every call
        # and get an answer about nothing.
        raise ValueError("agent.turn: this model does not accept image input, so `staged_images` cannot "
                         "be used with it. Documents (`staged_files`) work with any model.")
    if head_node_id is None:
        if datastore.nodes:
            raise ValueError("agent.turn: `head_node_id` is required for a datastore that already has "
                             "nodes; starting a new conversation would delete them.")
        head_node_id = chatutil.factory_reset_datastore(datastore, llm_settings)

    if docs_query is from_user_message:
        # With no corpus there is nothing to search, and passing a query anyway makes `ai_turn` warn about a
        # query it was never going to run — on every call of the one-liner form, which has no retriever.
        docs_query = user_message_text if retriever is not None else None

    if user_message_text is not None:
        head_node_id = scaffold.user_turn(llm_settings=llm_settings,
                                          datastore=datastore,
                                          head_node_id=head_node_id,
                                          user_message_text=user_message_text,
                                          staged_images=staged_images,
                                          staged_files=staged_files)
    started_from_node_id = head_node_id

    # The surface takes no callbacks; this one is internal, and it is how the record gets the prompts. The
    # agent loop calls it once per model call, with the history as it goes on the wire.
    prompts: list[list[dict]] = []

    final_node_id = scaffold.ai_turn(llm_settings=llm_settings,
                                     datastore=datastore,
                                     retriever=retriever,
                                     head_node_id=head_node_id,
                                     internet_enabled=internet_enabled,
                                     continue_=continue_,
                                     docs_enabled=docs_enabled,
                                     docs_query=docs_query,
                                     docs_num_results=docs_num_results,
                                     markup=markup,
                                     on_docs_start=None, on_docs_done=None,
                                     on_prompt_ready=prompts.append,
                                     on_llm_start=None, on_llm_progress=None, on_llm_done=None,
                                     on_tools_start=None,
                                     on_call_lowlevel_start=None, on_call_lowlevel_done=None,
                                     on_tool_done=None, on_tools_done=None)

    # A continuation revises the node it was given rather than adding one, so the turn's span is that node
    # onward, not everything after it.
    since_node_id = datastore.get_parent(started_from_node_id) if continue_ else started_from_node_id
    return describe_turn(datastore=datastore,
                         head_node_id=final_node_id,
                         since_node_id=since_node_id,
                         prompts=tuple(prompts))
