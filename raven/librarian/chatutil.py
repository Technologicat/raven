"""Utilities for formatting LLM chat messages."""

__all__ = ["format_message_number",
           "format_persona",
           "format_message_heading",
           "format_date_now", "format_time_now",
           "format_chatlog_datetime_now",
           "format_message_text_for_export",
           "format_disclosure_manifest",
           "format_reminder_to_write_conversationally",
           "format_reminder_to_use_information_from_context_only",
           "format_notice_that_tools_are_spent",
           "format_error_that_tools_are_spent",
           "format_docs_match", "format_docs_matches",
           "document_label", "excerpt", "format_consulted_documents",

           "default_formatters",  # the eight above that the model reads, as a namespace for `settings`

           "make_timestamp",
           "text_content_part",
           "image_content_part",
           "text_file_content_part",
           "normalize_content",
           "content_to_text",
           "create_message_from_parts",
           "create_chat_message",
           "create_initial_system_message",
           "create_payload",
           "linearize_chat",
           "compute_auto_allowed_hosts",
           "upgrade_datastore",
           "remove_persona_from_start_of_line",
           "get_node_message_text_without_persona",
           "scrub"]

import logging
logger = logging.getLogger(__name__)

import copy
import datetime
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml

from mcpyrate import colorizer

from unpythonic import si_prefix
from unpythonic.env import env

from .. import __version__

from ..common import netutil
from ..common import utils as common_utils

from ..papers import bibtex

from . import chattree
from . import sidecarstore

# --------------------------------------------------------------------------------
# Content parts (OpenAI multimodal content schema)
#
# A chat message's `content` is a list of typed parts. Even a text-only message is
# `[{"type": "text", "text": "..."}]`. This is OpenAI's multimodal `content` shape used directly as Raven's
# internal representation, so the wire format needs no translation. v0 part types: "text" and "image_url".
# A later addition, "text_file", references an attached document (plain text or PDF) by its sidecar; it has no
# native wire form and is expanded into the message's text part at wire-build (`llmclient.invoke`), so it never
# leaves as-is. Reading code must never treat `content` as a string — funnel through `content_to_text` for the
# text, or dispatch on each part's "type".

def text_content_part(text: str) -> Dict[str, str]:
    """Wrap a plain string as a single text content-part: `{"type": "text", "text": text}`."""
    return {"type": "text", "text": text}

def image_content_part(url: str) -> Dict[str, Any]:
    """Wrap an image URL as an image content-part: `{"type": "image_url", "image_url": {"url": url}}`.

    The multimodal sibling of `text_content_part`. In a *stored* message, `url` is always a Raven-internal
    `sidecar:<filename>` reference (never an `https://` URL, so stored chats stay offline-reloadable and don't
    phone home); `llmclient.invoke` substitutes a real `data:` URL from the sidecar bytes just before sending
    on the wire.
    """
    return {"type": "image_url", "image_url": {"url": url}}

def text_file_content_part(url: str, name: str, source: str) -> Dict[str, Any]:
    """Wrap an attached document as a content-part: `{"type": "text_file", "text_file": {"url", "name", "source"}}`.

    A Raven-internal part type for a document (plain text or PDF) attached to a message. As with
    `image_content_part`, in a *stored* message `url` is a `sidecar:<filename>` reference — the document bytes
    live in the datastore's sidecar directory, never inline in the chat JSON — and `name` is the original
    filename, kept for display and for the wire header. Unlike an image, this part has no native wire form:
    `llmclient.invoke` reads the sidecar, extracts its plaintext (`raven.common.docextract`), and folds it into
    the message's text part just before sending, so any model can use it (no vision capability required).
    `content_to_text` skips this part, so an attached document never leaks into the message's own displayed text.

    `source` names how the document got here, in the vocabulary `sidecarstore.base_provenance` documents
    (`"user_attachment"`, `"tool_result"`, ...). The same value is recorded in the sidecar provenance, and this
    is a deliberate second copy rather than a redundancy: how much of the window a document may occupy depends
    on who asked for it — a paper the user attached says *read this*, a page the model fetched on a hunch does
    not — and `llmclient.serialize_history_for_wire` decides that from bare messages, which carry no
    `general_metadata` and so cannot reach the provenance. Keeping it on the part is also the right shape on its
    own: what a message means on the wire should not depend on metadata travelling separately.
    """
    return {"type": "text_file", "text_file": {"url": url, "name": name, "source": source}}

def normalize_content(content: Any) -> List[Dict[str, Any]]:
    """Return `content` as a content-parts list: a bare string becomes a single text part; a list passes through.

    This is the *one* place that converts a legacy bare string into parts. It exists for the load-time
    migration (`upgrade_datastore`) — old datastores stored `content` as a string; the migration wraps each
    into `[{"type": "text", "text": <string>}]` so that everything in memory, and everything written back, is
    parts. After migration, readers never see a string; they assume the parts list (see `content_to_text`).
    Idempotent on already-parts content. Raises on anything that is neither str nor list (datastore
    corruption), rather than silently coercing — the migration relies on this to surface bad data.
    """
    if isinstance(content, str):
        return [text_content_part(content)]
    if isinstance(content, list):
        return content
    raise TypeError(f"normalize_content: expected message content to be str or list, got {type(content)}")

def content_to_text(content: Optional[List[Dict[str, Any]]]) -> str:
    """Concatenate the text of a message's content-parts `content` — the universal "give me the text" accessor.

    Text parts are concatenated in order; non-text parts (e.g. images) are skipped. `None` (absent content)
    yields `""`. Use this anywhere that needs the message's text for counting, matching, scrubbing, or
    single-string rendering. `content` is a parts list, never a string: legacy bare strings are migrated to a
    single text part at load time (`upgrade_datastore` via `normalize_content`), so a string reaching here is a
    bug — it raises loudly rather than being silently tolerated.
    """
    if content is None:
        return ""
    return "".join(part["text"] for part in content if part.get("type") == "text")

# --------------------------------------------------------------------------------
# Display formatting utilities (markdown, ansi)

def _yell_if_unsupported_markup(markup):
    if markup not in ("ansi", "markdown", None):
        raise ValueError(f"unknown markup kind '{markup}'; valid values: 'ansi' (*nix terminal), 'markdown', and the special value `None`.")

def format_message_number(message_number: Optional[int],
                          markup: Optional[str]) -> str:
    """Format the number of a chat message, e.g. '[#42]'.

    `message_number`: The number to format. If `None`, this returns the empty string, for convenience.
    `markup`: Which markup kind to use, or `None` for no markup. One of:
        "ansi": ANSI terminal color codes
        "markdown": Markdown markup
        `None` (the special value): no markup.

    Returns the formatted number.
    """
    _yell_if_unsupported_markup(markup)
    if message_number is not None:
        out = f"[#{message_number}]"
        if markup == "ansi":
            out = colorizer.colorize(out, colorizer.Style.DIM)
        elif markup == "markdown":
            out = f"*{out}*"
        return out
    return ""

def format_persona(role: str,
                   persona: Optional[str],
                   markup: Optional[str]) -> str:
    """Format the persona name for `role`.

    `role`: One of the roles supported by `raven.librarian.llmclient`.
            Typically, one of "assistant", "system", "tool", or "user".

    `persona`: The persona name speaking, or `None` if the role has no persona name ("system" and "tool" are like this).

               To get the **current session's** persona, use::

                   persona=llm_settings.personas.get(role, None)

               where `role` is one of "assistant", "system", "tool", "user".

               To get the **stored** persona from a chat node::

                   persona=node_payload["general_metadata"]["persona"]

               This may differ from the current session's persona, e.g. if the chat node was generated with a different AI character.

    `markup`: Which markup kind to use, or `None` for no markup. One of:
        "ansi": ANSI terminal color codes
        "markdown": Markdown markup
        `None` (the special value): no markup.

    Returns the formatted persona name.
    """
    _yell_if_unsupported_markup(markup)
    if persona is None:
        out = f"<<{role}>>"  # currently, this include "<<system>>" and "<<tool>>"
        if markup == "ansi":
            out = colorizer.colorize(out, colorizer.Style.DIM)
        elif markup == "markdown":
            out = f"`{out}`"  # use verbatim mode; otherwise looks like an HTML tag
        return out
    else:
        out = persona
        if markup == "ansi":
            out = colorizer.colorize(out, colorizer.Style.BRIGHT)
        elif markup == "markdown":
            out = f"**{out}**"
        return out

def format_message_heading(message_number: Optional[int],
                           role: str,
                           persona: Optional[str],
                           markup: Optional[str]) -> str:
    """Format a chat message heading.

    Calls `format_message_number` and `format_persona`, which see.

    Returns the formatted message heading.

    For example, in:

        [#1] Aria: How can I help you today?

    the heading is the "[#1] Aria: " part, including the final space.
    """
    _yell_if_unsupported_markup(markup)
    markedup_number = format_message_number(message_number, markup)
    markedup_persona = format_persona(role, persona, markup)
    if message_number is not None:
        return f"{markedup_number} {markedup_persona}: "
    else:
        return f"{markedup_persona}: "


# --------------------------------------------------------------------------------
# Stock message formatting utilities

_weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
def _format_isodatetime(d: datetime.datetime) -> Tuple[str, str, str]:
    """datetime.datetime -> ['Wednesday', '2025-09-24', '08:57:00']"""
    weekday = _weekdays[d.weekday()]
    isodate = d.date().isoformat()
    isotime = d.time().replace(microsecond=0).isoformat()
    return weekday, isodate, isotime

def format_date_now() -> str:
    """Return the text content of a dynamic system message containing the current date and weekday.

    This is for a dynamic injection.

    Split from `format_time_now` because the two differ in both shelf life and credibility:

      - The date is good for a whole day, so it can ride in the leading system block, where instructions
        are followed most reliably - and it needs that, because a date years past the model's training
        cutoff contradicts its prior, and it will argue with anything it trusts less.
      - The clock time changes every turn, so it stays out of the prefix that the backend would otherwise
        have to reprocess on each turn. It can afford the weaker placement: every clock time already
        existed when the model was trained, so there is nothing there to dispute.
    """
    weekday, isodate, isotime = _format_isodatetime(datetime.datetime.now())
    return f"[System information: Today is {weekday}, {isodate} (in ISO format).]"

def format_time_now() -> str:
    """Return the text content of a dynamic system message containing the current local time.

    This is for a dynamic injection. See `format_date_now` for why the two are separate.
    """
    weekday, isodate, isotime = _format_isodatetime(datetime.datetime.now())
    return f"[System information: The local time now is {isotime}.]"

def format_chatlog_datetime_now() -> str:
    """Return the current date, weekday, and local time in a human-readable format.

    As of v0.2.3, only used by Raven-librarian for logging the export date and time of an exported chatlog.
    """
    weekday, isodate, isotime = _format_isodatetime(datetime.datetime.now())
    return f"{weekday} {isodate} {isotime}"

def format_message_text_for_export(message: Dict[str, Any]) -> str:
    """Return a message's full text for export: its thinking trace, marked, followed by its visible reply.

    A thinking model produces two things, and an export that runs them together is not a record of what it
    said — a reader cannot tell which sentences were the answer and which were the model talking itself
    towards one, and neither can a parser. The two live in separate fields (`reasoning_content` beside
    `content`), so the distinction is not being *recovered* here; it is being carried across instead of
    dropped on the way out.

    The trace is wrapped in `<think>`/`</think>`. Synthetic in the sense that no backend sends those tags
    any more — `reasoning_content` arrives as its own field — but *restoring* rather than inventing. Up to
    v0.2.7 Raven spoke to one backend, which delivered the whole reply in one channel with the tags in it,
    and both exports carried them through without having to try. The June 2026 migration normalized every
    backend onto `reasoning_content`, which is the right shape for everything except these two call sites,
    where it silently removed a boundary that had always been in the text. So this emits what a reader of
    an older log already expects, and `_migration_think_block` still recognizes it.

    Messages with no trace (every user message, and any reply from a non-thinking model) come back as just
    their text, with no empty block to step over.
    """
    reasoning = (message.get("reasoning_content") or "").strip()
    text = content_to_text(message.get("content"))
    if not reasoning:
        return text
    return f"<think>\n{reasoning}\n</think>\n\n{text}"

def format_disclosure_manifest(payloads: List[Dict[str, Any]],
                               exported_at: Optional[str] = None) -> str:
    """Return a YAML front-matter block disclosing which messages in `payloads` were AI-generated.

    This is the origin disclosure that travels with exported text: which messages were written
    by a human, which were generated by an AI (and by which model, and when), and which are
    tool output. The EU AI Act's Article 50(2) asks providers of generative systems to mark
    AI-generated output so that a downstream reader - human or parser - can detect it as such.

    *Disclosure*, not *provenance*, and the distinction is worth holding: this answers "was a
    machine involved in writing this", which is a compliance question about the artifact.
    Provenance answers "which source material did this claim come from", which is the research
    question, and it is what the sidecar stores mean by the word (`sidecarstore.base_provenance`
    and the `provenance_url` / `provenance_source` attachment fields). Collapsing the two loses
    the one researchers actually care about.

    A robust mark for text is a generation-time watermark, applied inside the sampling loop.
    Librarian samples third-party weights through an OpenAI-compatible backend and so has no
    access to the logits; there is no watermarker anywhere in the path. What a system on this
    side of the boundary can do is attach the origin metadata it does know, which is what this is.

    YAML front matter, rather than a sentence in the prose, because a mark that only a human can
    read is only half a mark: the block is delimited, keyed, and conventional enough that
    downstream Markdown tooling already looks for it. It is emitted first in the document, since
    a front-matter parser only recognizes it in that position.

    `payloads`: Chat node payloads, in the order they appear in the exported text. One manifest
                entry is emitted per payload, numbered by position. Each is read for
                `message["role"]`, `general_metadata["datetime"]`, and
                `generation_metadata["model"]`; all but the role are optional, so a payload
                whose generation was interrupted (no `generation_metadata`) is fine.

    `exported_at`: Timestamp for the export itself, as an ISO 8601 string. If `None`, stamp now,
                   with the local UTC offset. (The per-message `generated_at` values carry no
                   offset - they are reproduced as stored, and the stored format has none.)

    Example, for a question and its answer::

        ---
        generator: raven-librarian
        generator_version: 0.2.8-dev
        exported_at: '2026-07-29T14:23:11+03:00'
        ai_generated: true
        messages:
        - n: 0
          origin: user
        - n: 1
          origin: assistant
          model: Qwen3-VL-30B-A3B
          generated_at: '2026-07-29 14:22:58'
        ---
    """
    if exported_at is None:
        exported_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")

    messages = []
    for message_number, payload in enumerate(payloads):
        role = payload["message"]["role"]
        entry = {"n": message_number,
                 "origin": role}
        # Optional keys are omitted rather than emitted as null: a reader should be able to take the
        # presence of `model` as a claim, and its absence as "not recorded", without a third state.
        if (model := payload.get("generation_metadata", {}).get("model", None)) is not None:
            entry["model"] = model
        if (generated_at := payload.get("general_metadata", {}).get("datetime", None)) is not None:
            entry["generated_at"] = generated_at
        messages.append(entry)

    # Only the assistant role is model-generated prose. Tool messages are retrieved external content
    # that the model requested but did not write, so they do not by themselves make an export
    # AI-generated - labeling them `origin: tool` states what actually happened.
    manifest = {"generator": "raven-librarian",
                "generator_version": __version__,
                "exported_at": exported_at,
                "ai_generated": any(entry["origin"] == "assistant" for entry in messages),
                "messages": messages}

    # `safe_dump` rather than a hand-built string: a model name or a persona is arbitrary text, and
    # a stray colon in one would otherwise silently produce a manifest that parses as something else.
    body = yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return f"---\n{body}---\n"

def format_reminder_to_write_conversationally() -> str:
    """Return the text content of a system message that asks the LLM to answer in prose rather than in report form.

    Aimed at the reflex, strong in the Qwen 3 family, to answer a conversational question with headings and
    bulleted sections. A chat is a discussion, and a discussion in outline form reads as a briefing document.

    This used to also carry "Reply to the user's most recent message", for models - the distills of
    DeepSeek-R1, early 2025 - that would otherwise answer some earlier turn instead. No model in the current
    lineup needs telling: the temporary injects now sit ahead of the user's latest message rather than after
    it, which leaves the question last, where a chat template already points the model at it.

    This is for a dynamic injection.
    """
    return "[System information: In a discussion, prefer writing your raw thoughts rather than a structured report.]"

def format_reminder_to_use_information_from_context_only() -> str:
    """Return the text content of a system message that reminds the LLM to ground its claims in the provided context (not its internal static knowledge).

    As with all things LLM, this isn't completely reliable, but tends to increase the chances of the model NOT responding based on its static knowledge.
    This is useful when summarizing or extracting information from RAG search results.

    The first line of defense is not giving control to the LLM when the search comes up empty. This reminder helps when the search returns results,
    but their content is irrelevant to the query - or when docs are not enabled, but there is some other data in the context, and the answer should be
    based on that.

    The wording is deliberately about *grounding* rather than about a prohibition. Asking for context-only answers
    reads, to a model that takes instructions literally, as a ban on general knowledge - and then a question like
    "what is 2+2?" becomes a dilemma to be reasoned through instead of answered. Measured across the supported
    model families, that phrasing cost 5-37x the deliberation of sending no reminder at all; one model never
    terminated, and another refused outright. This phrasing measured within noise of sending nothing, while still
    declining correctly when asked about something the documents do not contain. Caller-side, the reminder is sent
    only when there actually is context to ground in - a reminder about "the provided documents" with no documents
    provided is the self-contradiction that started the problem.

    This is for a dynamic injection.
    """
    return "[System information: Base claims about the provided documents on those documents. Answer general questions normally.]"

def format_notice_that_tools_are_spent() -> str:
    """Return the text of a system message telling the LLM that this reply gets no more tool calls.

    Sent on every invocation of an AI turn after the tool-call round cap is reached, and only then. In the
    common case that is exactly one invocation, since the model answers when told to.

    It exists because the model is otherwise never *told* that the gathering is over - it finds out by
    reaching for a tool and being refused. Given a list of documents to work through, the model spends its
    rounds fetching them one at a time, and on the invocation after the last one it announces the *next*
    fetch ("Now let's get the ABR reactor document") and then stops, having written no reply at all.

    Paired with `format_error_that_tools_are_spent`, which says the same thing in the tool-result channel
    for a model that reaches for a tool anyway. This one aims to make that call unnecessary; that one
    handles it when it happens.

    **Reaching the cap is what produces the empty reply, and this notice does not measurably prevent it.**
    Measured over 24 paired samples (`investigations/tool_budget/`, qwen3.6-35b-a3b): turns that reached the cap
    answered 5 of 14, turns that did not answered 9 of 10 (Fisher exact p = 0.013). The notice itself moved
    nothing - 8 of 12 answered with it against 6 of 12 without, p = 0.68, and the two arms disagree about
    the sign once restricted to cap-reaching turns. So this is kept on the strength of the mechanism it
    addresses, which was directly observed, and not on evidence that it works. The fix with the evidence
    behind it is a larger budget for fetch-shaped calls; see the tool-budget item in `TODO.md`.

    Worded as a statement of the situation with the required action attached, never as a prohibition. "You
    may not call any more tools" is the shape that measured 5-37x the deliberation elsewhere in this file,
    and it would land here on a model that is *already* mid-task and looking for a way to continue.
    Permission to say the answer is incomplete is part of that: without it, a model whose gathering was cut
    short has a reason to keep trying rather than to report what it has.

    This is for a dynamic injection.
    """
    return ("[System information: No further tool calls are available for this reply. Write the answer now, "
            "from the information gathered above. If something you wanted is missing, say so in the answer.]")

def format_error_that_tools_are_spent() -> str:
    """Return the text of the tool result given when a call is refused because the turn's budget is spent.

    Sent in place of the tool's output, as an `status="error"` result, once the AI turn has used up
    `max_tool_call_rounds`. The tool stays in the schema and the call is answered rather than made.

    Why an error result and not an absent tool: a mid-turn change to the tool loadout invalidates the
    backend's KV cache for everything after it, and a history whose earlier messages call a tool that the
    current request does not declare is a shape the model saw little of in training. A tool that answers
    "not now" is a shape it saw plenty of. Withdrawing the tools remains the terminator of last resort
    (`max_tool_call_refusal_rounds`), because a refusal, however well formed, cannot by itself guarantee
    that the loop ends.

    Worded the same way as `format_notice_that_tools_are_spent`: the situation, plus the action that
    follows from it, and permission to answer incompletely. Not a prohibition - "you may not call any more
    tools" is the shape that measured 5-37x the deliberation elsewhere in this file, and it would land here
    on a model that is already mid-task and looking for a way to continue.
    """
    return ("The tool-call budget for this reply is spent, so this call was not made. Write the answer now, "
            "from the information gathered above. If something you wanted is missing, say so in the answer.")


# --------------------------------------------------------------------------------
# Document database match formatting

def format_docs_match(match: Dict[str, Any]) -> str:
    """Format one document-database match (a result dict from `hybridir.HybridIR.query`) for the LLM.

    The single formatter for a match, shared by the two paths that produce one: the auto-search results
    that `scaffold` injects on the model's behalf, and the results of a `search_documents` tool call the
    model made itself. If these two ever drift apart, the model is being told two different things about
    what a match *is* - so they are one function, not two similar ones.

    The span (`offset`, `length`) is reported so that a match can be followed up with `fetch_document`:
    a match is a window onto a larger document, and without its coordinates the model can ask only for
    the whole thing. Both are in characters, matching the retriever's own units and the tool's parameters.

    NOTE: `length` is that of the *unstripped* match text, and `offset` is its true offset in the source
    document, so `document[offset:offset + length]` is exactly the matched span. The text is shown
    stripped, purely because chunk boundaries tend to land in the middle of whitespace - so the displayed
    block can start a few characters after `offset`. That mismatch is intentional; do not "fix" it by
    reporting the offset of the stripped text, which would make the reported span no longer the span the
    retriever found.
    """
    return (f"[System information: Knowledge-base match from '{match['document_id']}', "
            f"at offset {match['offset']}, length {len(match['text'])} characters.]\n\n"
            f"{match['text'].strip()}\n-----")

def format_docs_matches(matches: List[Dict[str, Any]]) -> str:
    """Format a list of document-database matches for the LLM, as one text blob. See `format_docs_match`."""
    return "\n\n".join(format_docs_match(match) for match in matches)

# How much of a document's beginning `document_label` may look at. A title lives near the front, and
# scanning a multi-megabyte reference database end to end - once per listed document, once per turn - would
# be a lot of work for one line of display text.
_DOCUMENT_LABEL_SCAN_LIMIT = 8192

# Above this, a `.bib` is described as a database by size rather than parsed. A single BibTeX record with a
# full abstract runs a few kilobytes; anything much larger holds many records, and reading a multi-megabyte
# reference database end to end - once per listed document, once per turn - is a lot of work for one line
# of display text.
_BIBTEX_SINGLE_RECORD_LIMIT = 65536

# Shorter than this, a line is punctuation or a page number rather than a title.
_MINIMUM_PSEUDO_TITLE_LENGTH = 8

# Long enough to identify a title, short enough that a list of twenty stays readable.
_MAXIMUM_LABEL_LENGTH = 200

def _shorten(text: str,
             max_length: int) -> str:
    """Cut `text` to `max_length`, marking that it was cut. For display labels, not for content."""
    text = " ".join(text.split())  # a title wrapped across source lines reads as one line here
    if len(text) <= max_length:
        return text
    return text[:max_length - 1].rstrip() + "…"

def _bibtex_library(text: str) -> Optional[Any]:
    """Parse `text` as BibTeX, returning the `bibtexparser` library, or `None` if it will not parse at all.

    Returning `None` rather than raising is the point of this wrapper: the caller is sniffing arbitrary
    pasted text to see whether it happens to be BibTeX, so failure to parse is an expected answer and not
    an error. `raven.papers.bibtex.parse_string` supplies the middleware chain and the rationale for it.
    """
    try:
        library = bibtex.parse_string(text)
    except Exception as exc:
        logger.debug(f"_bibtex_library: not readable as BibTeX ({type(exc)}: {exc}).")
        return None
    return library

def _bibtex_entry_label(entry: Any) -> str:
    """Label one parsed BibTeX entry from its own fields, or `""` if it has no title to show."""
    fields = {field.key: field.value for field in entry.fields}
    title = str(fields.get("title", "")).strip("{} ")
    if not title:
        return ""
    authors = fields.get("author") or []  # guarded: an authorless record is ordinary here, not worth a warning
    year = str(fields.get("year", "")).strip("{} ")
    attribution = " ".join(part for part in (common_utils.format_bibtex_authors(authors) if authors else "", year) if part)
    return f'"{title}"' + (f" ({attribution})" if attribution else "")

def document_label(text: str) -> str:
    """Describe a document in one line, from the document itself. `""` when nothing in it describes it.

    A search can return twenty documents, and fetching each one to find out what it is would be exactly the
    waste that listing them is meant to avoid - so a list of document IDs needs something beside each ID
    that is *decision-grade*: enough to tell whether this is the one worth reading in full.

    A fallback chain over the best structured signal the document carries, rather than a heuristic:

      1. **A BibTeX record** - its own `title` / `author` / `year`. Exact, no guessing. A `.bib` holding
         *many* records is described as the database it is, by record count, because the useful decision
         about one of those is not to fetch it: it is one document as far as the retriever is concerned,
         however many works it lists. (Nothing stops a user dropping a whole reference database into the
         document directory. `raven-burstbib` bursting it into one file per record is the better shape, and
         is what `raven.papers` produces, but it is a recommendation rather than a requirement.)
      2. **A BibTeX record that will not parse** - its `title` field, dug out by pattern. Real corpora
         contain records that are not quite valid BibTeX: one Web of Science export in ~12000 carries an
         abstract with unbalanced braces, which aborts the parse of the whole record. The title is
         nonetheless sitting right there on its own line, and a label is a place where a good guess beats
         nothing.
      3. **Anything else** - the first substantial line, as a pseudo-title. Weak, but it costs nothing and
         is right surprisingly often (papers, notes and reports tend to open with their own titles).

    There is deliberately no filename case in the chain, though a hand-curated stash often has descriptive
    filenames carrying author, year and title. The caller shows the document ID anyway - it is the key to
    fetch by - so a label that repeated it would say nothing twice, and `""` reads correctly as "the ID is
    all there is".
    """
    lines = text[:_DOCUMENT_LABEL_SCAN_LIMIT].splitlines()
    first_nonblank = next((line.lstrip() for line in lines if line.strip()), "")
    looks_like_bibtex = first_nonblank.startswith("@")

    if looks_like_bibtex:  # cheap gate, so a plain document never pays for a BibTeX parse
        if len(text) > _BIBTEX_SINGLE_RECORD_LIMIT:  # too big to be one record; no need to read it to know
            return f"BibTeX database, {si_prefix(len(text), precision=0)} characters"
        library = _bibtex_library(text)
        if library is not None:
            # Failed blocks count as records: a duplicate BibTeX key makes `bibtexparser` refuse the second
            # entry, and a file with a repeated key is still plainly a file of many records.
            n_records = len(library.entries) + len(library.failed_blocks)
            if n_records > 1:
                return f"BibTeX database of {n_records} records"
            if library.entries:
                label = _bibtex_entry_label(library.entries[0])
                if label:
                    return _shorten(label, _MAXIMUM_LABEL_LENGTH)
        # Records that are not quite valid BibTeX do occur: an abstract with unbalanced braces aborts the
        # parse of the whole record, title and all. Since the alternative is no label, read the one field
        # we need by pattern - after the real parser has had its say, never instead of it.
        salvaged = common_utils.bibtex_field_value(text, "title")
        if salvaged:
            return _shorten(f'"{salvaged}"', _MAXIMUM_LABEL_LENGTH)

    for line in lines:
        stripped = line.strip()
        if len(stripped) < _MINIMUM_PSEUDO_TITLE_LENGTH:
            continue
        # In a record that failed to parse, every line is a `Key = {value}` pair, and the first one that is
        # long enough is whichever field happens to come first - `Author = {...}` in Web of Science output.
        # A label reading "Author = {Afgan, Nain H. and ...}" looks like a bug even though nothing went
        # wrong that was ours, so a BibTeX file that gets this far gets no label rather than a silly one.
        if looks_like_bibtex:
            continue
        return _shorten(stripped, _MAXIMUM_LABEL_LENGTH)
    return ""

# What marks an excerpt as having more behind it. On its own line, so it reads as an omission rather than
# as the writer trailing off mid-thought - a stored document's opening often ends on a colon.
_EXCERPT_CONTINUES_MARKER = "…"

# How far into the budget a paragraph break must sit before `excerpt` will end on it instead of filling the
# budget and cutting mid-paragraph. Ending on a paragraph reads better, but only when it costs little: a
# fetched page opens with a source header, a title and often a licence notice, so the last break before a
# few hundred characters tends to be the one right *before* the first real prose. Snapping to it spends the
# whole budget on boilerplate and stops exactly where the document starts saying something.
_EXCERPT_PARAGRAPH_SNAP_FRACTION = 0.5

def excerpt(text: str, max_characters: int) -> str:
    """Return the opening of `text`, at most about `max_characters` long, cut on a sensible boundary.

    Unlike `document_label`, which distils a document down to one line to choose *between* documents, this
    shows the reader the beginning of a document they already have: the head of a long tool result, with the
    rest of it available as an attachment beside it.

    The budget is *filled*, then the cut is backed off to a boundary — a paragraph break if one sits in the
    later part of the budget, otherwise a word boundary, otherwise (a single unbroken token) the budget
    itself. Filling first is what makes the result informative on a real document: pages open with headers,
    titles and notices, so a rule that always ended on the last complete paragraph would reliably stop just
    before the first sentence worth reading.

    A cut is marked, so the reader can tell "this is the whole result" from "this is where it was cut". Text
    that fits entirely is returned unchanged, with no marker.

    `max_characters` is a budget rather than a hard limit: the marker is added on top of it. Callers wanting
    an exact bound should cut the result themselves.
    """
    text = text.strip()
    if len(text) <= max_characters:
        return text

    head = text[:max_characters]
    paragraph_breaks = list(re.finditer(r"\n\s*\n", head))
    if paragraph_breaks and paragraph_breaks[-1].start() >= max_characters * _EXCERPT_PARAGRAPH_SNAP_FRACTION:
        cut_at = paragraph_breaks[-1].start()
    else:
        cut_at = head.rfind(" ")
        if cut_at <= 0:  # a single unbroken token; a hard cut beats showing nothing
            cut_at = max_characters
    return f"{head[:cut_at].rstrip()}\n\n{_EXCERPT_CONTINUES_MARKER}"

# The user's whole message is the auto-search query, so this can be an essay. It is shown to say *why* a
# document is on the list, which the first line of it does.
_MAXIMUM_SHOWN_QUERY_LENGTH = 120

def format_consulted_documents(entries: List[Dict[str, Any]]) -> str:
    """Format the documents this conversation has already looked at, for the LLM.

    Each entry is a dict with `document_id`, and optionally `label` (see `document_label`) and `query` (what
    surfaced it). Pointers, not text: re-injecting the material would grow without bound, while a list of
    IDs does not - and `fetch_document` is what makes a pointer worth having.

    "Consulted" is deliberately silent about *who* consulted: the list merges what Raven searched on the
    user's behalf with what the model went and fetched itself, and naming an actor in either direction
    would make half the entries read wrong.

    Which is also why the header does not claim the text is gone. That is true only of the automatic
    search, whose matches are injected for one turn and never persisted; a document the model fetched is a
    stored tool node, still written out verbatim wherever the window still reaches. The list cannot tell
    those apart from here, so it says what is true of both - these were consulted, and any that are no
    longer written out can be read again.
    """
    lines = []
    for entry in entries:
        line = f"- {entry['document_id']}"
        label = entry.get("label")
        if label:
            line += f" - {label}"
        query = entry.get("query")
        if query:
            line += f" [surfaced by: {_shorten(query, _MAXIMUM_SHOWN_QUERY_LENGTH)}]"
        # A document the user has since deleted stays listed - the conversation did read it - but says so,
        # or the model spends a `fetch_document` round discovering it. A statement of the situation rather
        # than a prohibition, in the manner of `CANONICAL_NO_ROOM_TO_FETCH`: a "you may not" in an inject
        # measured badly (`investigations/context-injects/`). Absent `present` means present, so entries
        # assembled without `llmclient.label_documents` read unchanged.
        if entry.get("present") is False:
            line += " [no longer in the database]"
        lines.append(line)
    # "Previously" rather than "already", because this list and the current turn's search results look
    # alike, and without it the model can read a document from three turns ago as something the search just
    # returned. The inject ordering in `scaffold.build_turn_prompt` says the same thing by position - this
    # list first, then the current turn's matches - and the two are meant to agree.
    header = ("[System information: Documents from the knowledge base that this conversation consulted "
              "previously, on earlier turns. Any whose text is no longer written out above can be read "
              "again with `fetch_document`, by the ID shown.]")
    return f"{header}\n\n" + "\n".join(lines)


def default_formatters() -> env:
    """The model-facing formatters, as a namespace, for `settings.formatters`.

    These eight are the ones whose output reaches the LLM: the per-turn injects, the two tool notices, and
    the two tool results that are text rather than data. Everything else named `format_*` here writes for
    the chat log or an export, where the reader is a person and a run has no reason to vary it.

    They live on `settings` for the sake of experiments that A/B a wording, which is the only thing that
    wants them to vary - ordinary use never touches them, and gets exactly the functions below. The
    alternative an experiment had before was assigning to a global in this module, which is process-wide
    and therefore shared with any concurrent turn; a settings field belongs to the one run.

    The names drop the `format_` prefix, the namespace having said it already.
    """
    return env(date_now=format_date_now,
               time_now=format_time_now,
               reminder_to_write_conversationally=format_reminder_to_write_conversationally,
               reminder_to_use_information_from_context_only=format_reminder_to_use_information_from_context_only,
               notice_that_tools_are_spent=format_notice_that_tools_are_spent,
               error_that_tools_are_spent=format_error_that_tools_are_spent,
               docs_matches=format_docs_matches,
               consulted_documents=format_consulted_documents)


# --------------------------------------------------------------------------------
# Chat message creation utilities

def make_timestamp(timestamp: Optional[int] = None) -> Tuple[int, str, str, str]:
    """Stamp the date and time.

    `timestamp`: Nanoseconds since epoch, as returned by `time.time_ns()`.
                 If `None`, stamp the current date and time.

    Returns the tuple `(nanoseconds_since_epoch, weekday, isodate, isotime)`.

    Useful e.g. for timestamping new chat messages.
    """
    if timestamp is None:
        timestamp = time.time_ns()  # authoritative timestamp: nanoseconds since epoch
    now = datetime.datetime.fromtimestamp(timestamp / 1e9)  # UGH, `datetime.datetime.now()` would be nicer, but then we would risk losing resolution in the authoritative timestamp.
    weekday, isodate, isotime = _format_isodatetime(now)
    return timestamp, weekday, isodate, isotime

def create_message_from_parts(role: str,
                              content: List[Dict[str, Any]],
                              *,
                              tool_calls: Optional[List[Dict]] = None,
                              reasoning_content: Optional[str] = None) -> Dict:
    """Build a chat message whose `content` is an already-structured content-parts list.

    The counterpart to `create_chat_message` for callers that have parts in hand rather than a plain string —
    tool results split into one part per item, multimodal input. No persona prefix, no string wrapping, no
    `llm_settings` needed. `create_chat_message` builds its single persona-prefixed text part and then delegates
    here, so the message-dict shape and role validation live in one place.

    `role`: one of "user", "assistant", "system", "tool".
    `content`: the content-parts list, used as the message content verbatim.
    `tool_calls`: structured OAI tool-call dicts; `None` -> empty list.
    `reasoning_content`: thinking-trace string stored as a sibling field; when `None`, no key is added.
    """
    if role not in ("user", "assistant", "system", "tool"):
        raise ValueError(f"Unknown role '{role}'; valid: one of 'user', 'assistant', 'system', 'tool'.")
    data = {"role": role,
            "content": content,
            "tool_calls": tool_calls if tool_calls is not None else []}
    if reasoning_content is not None:
        data["reasoning_content"] = reasoning_content
    return data

def create_chat_message(llm_settings: env,
                        role: str,
                        text: str,
                        add_persona: bool = True,
                        persona: Optional[str] = None,
                        tool_calls: Optional[List[Dict]] = None,
                        reasoning_content: Optional[str] = None) -> Dict:
    """Create a new chat message, compatible with the chat history format sent to the LLM.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.

    `role`: One of "user", "assistant", "system", "tool".

            Typically, "system" is used for the initial system prompt / character card combo,
            and "tool" is used for tool responses from tool-calls made by the LLM.

            Because this function creates a new chat message, the persona is always the current
            session's persona for `role`, automatically read from `llm_settings`.

    `text`: The text content of the message, as a plain string. Wrapped as a single text content-part
            (`[{"type": "text", "text": ...}]`); the persona prefix is prepended first if `add_persona` applies.

            Multi-part content (a tool result split into several parts, an image attachment) is not built
            here — that path is introduced where a real caller needs it. This function is the everyday
            text-message constructor (user / assistant / system / single-text tool messages).

    `add_persona`: If `True`, we prepend the persona of `role` to the text content,
                   if `llm_settings.personas` has a name defined for that role.

                   E.g., if `role='assistant'`, format output as "AI: ...",
                   where "AI" is the persona.

                   Usually this is the right thing to do, but there are some occasions
                   (e.g. internally in `invoke`) where we need to skip this.

    `persona`: Persona name override, used when `add_persona=True`.

               When `None`, uses the current persona name for `message["role"]`, from `llm_settings`,
               as explained above. This is the Right Thing when creating a new chat message.

               If creating a new revision of an existing message (editing an existing chat message
               from the datastore), you should pass the old persona, which is available in the
               return value of `get_node_message_text_without_persona`, or equivalently,
               in `old_payload["general_metadata"]["persona"]`.

    `tool_calls`: Tool call requests; a list of structured OAI tool-call dicts
                  (`{"type": "function", "function": {"name", "arguments"}, "id", ...}`),
                  where `arguments` is itself a JSON string per the OAI convention.
                  These are parsed from the LLM's response by `invoke`.

                  Mostly for use by `invoke`.

                  If `None`, an empty list is created. This is usually the right thing to do.

    `reasoning_content`: The message's reasoning (thinking) trace, as a plain string, stored as a
                         sibling field of `content` (matching the llama.cpp / LM Studio convention).
                         `invoke` populates this from the model's reasoning channel — either the native
                         `reasoning_content` stream deltas, or inline `<think>...</think>` blocks it
                         parses out of the content stream. Never embedded in `content`.

                         If `None` (the default), no `reasoning_content` key is added — appropriate for
                         user/system/tool messages and assistant messages with no thinking trace.

    Returns the new message: `{"role": ..., "content": [<part>, ...], "tool_calls": ...}`, where `content`
    is always a content-parts list; plus a `"reasoning_content"` key when `reasoning_content` is given.
    """
    persona_name = llm_settings.personas.get(role, None) if (persona is None) else persona
    if add_persona and persona_name is not None:
        text = f"{persona_name}: {text}"  # e.g. "User: ..."
    # else: system and tool messages typically do not include a persona name in the text content.

    return create_message_from_parts(role,
                                     [text_content_part(text)],
                                     tool_calls=tool_calls,
                                     reasoning_content=reasoning_content)

def create_initial_system_message(llm_settings: env, use_character_card: bool = True) -> Optional[Dict]:
    """Create a chat message containing the system prompt and the AI's character card as specified in `llm_settings`.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.

    `use_character_card`: Whether the AI character is present. `False` builds the message from `system_prompt`
                   alone — the half of the configuration that holds whatever character it is wearing — and
                   returns `None` when that is empty, which is how Raven ships it. The caller then creates
                   no system node at all, which is the correct shape for a bare-model call: nothing is lost,
                   because there was nothing character-independent to say.

                   This is the one place that knows how a system message is assembled, so that a deployment
                   which does fill `system_prompt` keeps it in both settings without every caller having to
                   remember that it might be there.
    """
    if not use_character_card:
        return (create_chat_message(llm_settings, role="system", add_persona=False,
                                    text=f"{llm_settings.system_prompt}\n\n-----")
                if llm_settings.system_prompt else None)
    if llm_settings.system_prompt and llm_settings.character_card:
        # The system prompt is stripped, so we need two linefeeds to have one blank line in between.
        text = f"{llm_settings.system_prompt}\n\n{llm_settings.character_card}\n\n-----"
    elif llm_settings.system_prompt:
        text = f"{llm_settings.system_prompt}\n\n-----"
    elif llm_settings.character_card:
        text = f"{llm_settings.character_card}\n\n-----"
    else:
        raise ValueError("create_initial_system_message: Need at least a system prompt or a character card.")
    return create_chat_message(llm_settings,
                               role="system",
                               text=text)

def create_payload(llm_settings: env,
                   message: Dict[str, Any],
                   persona: Optional[str] = None) -> Dict[str, Any]:
    """Create a payload for a chat node.

    This sets the "message" field of the payload to `message` (see `create_chat_message`),
    and automatically populates the "general_metadata" field. The general metadata contains
    the payload creation timestamp, and persona name. Roles that do not have a persona name
    (namely, "tool" and "system") have `None` as the persona name in the metadata.

    The "generation_metadata" field (note "generation", not "general"!), on the other hand,
    depends on the message role. It is populated elsewhere, at the call sites. Only messages
    with role "assistant" or "tool" have that field.

    RAG retrieval (document database autosearch) may also add a "retrieval" field,
    e.g. to the payload of an AI message.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.

    `message`: Return value of `create_chat_message`, which see.

    `persona`: Persona name, to be stored in "general_metadata".

               If `None`, uses the current persona name for `message["role"]`, from `llm_settings`.
               This is the Right Thing when creating a new chat message.

               If creating a new revision of an existing payload (editing an existing chat message
               from the datastore), you should pass the old persona, which is available in the
               return value of `get_node_message_text_without_persona`, or equivalently,
               in `old_payload["general_metadata"]["persona"]`.

    The return value is the payload.

    **WHAT, WHY, HOW**:

    The chat nodes (in `chattree.Forest`) store and revision payloads.
    This function creates one such payload.

    - When creating a new node, the payload can be passed to `chattree.Forest.create_node`.
      It is then added as the first revision of the payload to that node.

    - When adding a new revision to an existing node (e.g. when the user fixes a typo),
      the payload can be passed to `chattree.Forest.add_revision`, which adds it to the
      desired node, and sets the new revision as active.

      The old revision is NOT deleted unless explicitly requested by calling
      `chattree.Forest.delete_revision` separately.

      By design, it is not possible to overwrite a revision; revisions are immutable.
      If you want to emulate an overwrite (doing so increases the revision number),
      you can first `chattree.Forest.get_revision` to get the old active revision
      number, then add your new revision, and finally delete the old one. Note the
      ordering; a chat node must contain at least one payload revision at all times.

    So, roughly, to create a new node in the datastore::

        # Raw OAI-compatible message record
        message = chatutil.create_chat_message(llm_settings, role=..., text=...)

        # Payload; message record and Raven-librarian metadata
        payload = chatutil.create_payload(llm_settings, message)

        # Chat node, which can contain one or more payload revisions
        node = datastore.create_node(payload, parent_id=...)
    """
    # NOTE: If you add or change fields here, update also `upgrade_datastore`.
    timestamp, unused_weekday, isodate, isotime = make_timestamp()
    message_role = message["role"]
    persona_name = llm_settings.personas.get(message_role, None) if (persona is None) else persona
    payload = {"message": message,
               "general_metadata": {"timestamp": timestamp,
                                    "datetime": f"{isodate} {isotime}",
                                    "persona": persona_name}}
    return payload

# --------------------------------------------------------------------------------
# Chat datastore utilities

def linearize_chat(datastore: chattree.Forest, node_id: str) -> List[Dict]:
    """In the chat `datastore`, walking up from `node_id` up to and including a root node, return a linearized representation of that branch.

    This collects the active revision of the data from each node, ignores everything except the chat message data
    (i.e. ignores any metadata added by the chat client, such as RAG retrieval attributions, AI token counts, etc.)
    and puts the messages into a list, in depth order (root node first).

    Note `node_id` doesn't need to be a leaf node; but it will be the last node of the linearized representation;
    children are not scanned.

    NOTE: The difference between this function and `chattree.Forest.linearize_up` is that this will
    automatically extract the "message" field (OpenAI-compatible chat message record) from each node,
    using the active revision of the payload, whereas that other function returns the node IDs.

    Hence, this is a convenience function for populating a linear chat history for chat clients that use
    the OpenAI format to communicate with the LLM server.
    """
    node_id_history = datastore.linearize_up(node_id)
    payload_history = [datastore.get_payload(node_id=node_id) for node_id in node_id_history]  # this auto-selects the active revision of the payload of each node
    message_history = [payload["message"] for payload in payload_history]
    return message_history

def compute_auto_allowed_hosts(datastore: chattree.Forest,
                               node_id: str,
                               *,
                               trust_search_results: bool = False) -> frozenset:
    """Return the set of hosts to *auto-allow* for `webfetch` during the current turn.

    The `webfetch` domain allowlist constrains the AI's *initiative* — the sites the model
    may decide to visit on its own. A URL the user explicitly typed is the user's intent, not
    the model's, so its host is allowed for that turn without the user having to edit config.
    This computes those temporarily-allowed hosts by walking the branch ending at `node_id`
    (same convention as `linearize_chat`).

    Scope is the **current turn only** — from the most recent user-role message onward. A URL
    the user pasted in an *earlier* turn does not keep widening the trust surface (one-hop
    trust: trust user input one step, never transitively).

    - Always: hosts of URLs found in the latest user-role message.
    - If `trust_search_results` (the DANGEROUS, default-off power-user opt-in): also hosts of
      URLs found in this turn's `websearch` tool-results. Tool-result provenance is read from
      `generation_metadata.function_name`; only `websearch` results count. Crucially, `webfetch`'s
      *own* output is excluded, so a URL the model discovered inside a fetched page never gains
      auto-allow — content discovered by tools is not user intent.

    Returns a `frozenset` of lowercased hosts (possibly empty).
    """
    node_id_history = datastore.linearize_up(node_id)
    payload_history = [datastore.get_payload(node_id=nid) for nid in node_id_history]

    # Find the turn boundary: the most recent user-role message.
    last_user_index = None
    for index in range(len(payload_history) - 1, -1, -1):
        if payload_history[index]["message"].get("role") == "user":
            last_user_index = index
            break
    if last_user_index is None:
        return frozenset()

    def hosts_in(text: str):
        return (host for url in netutil.extract_urls(text) if (host := netutil.url_host(url)))

    hosts = set(hosts_in(content_to_text(payload_history[last_user_index]["message"].get("content"))))

    if trust_search_results:
        for payload in payload_history[last_user_index + 1:]:  # this-turn tool results only
            message = payload["message"]
            if message.get("role") != "tool":
                continue
            if payload.get("generation_metadata", {}).get("function_name") != "websearch":
                continue  # one-hop trust: search results yes, webfetch's own output (or anything else) no
            hosts.update(hosts_in(content_to_text(message.get("content"))))

    return frozenset(hosts)

# --------------------------------------------------------------------------------
# Datastore migration helpers: inline reasoning / tool-call normalization.
#
# Old chats stored reasoning inline as `<think>...</think>` in `message["content"]`, and (for tool-response
# messages) the tool-call linkage under `generation_metadata["toolcall_id"]`. The current format keeps reasoning
# in a `reasoning_content` sibling field, tool calls in structured `message["tool_calls"]`, and the linkage on
# `message["tool_call_id"]` (OAI spelling). These helpers normalize old payloads to the current format, each
# guarded for idempotency (a second pass over already-migrated data produces no changes).

# `\s*<think>(.*?)</think>\s*` (DOTALL): captures the inner reasoning and consumes the surrounding whitespace
# (incl. newlines) on both sides, so replacing with a single space preserves a persona prefix cleanly
# (`"Aria: <think>...</think>\nBla"` -> `"Aria: Bla"`).
_migration_think_block = re.compile(r"\s*<think>(.*?)</think>\s*", flags=re.DOTALL | re.IGNORECASE)
_migration_tool_call_block = re.compile(r"\s*<tool_call>(.*?)</tool_call>\s*", flags=re.DOTALL | re.IGNORECASE)

def _normalize_tool_call_arguments(arguments: str) -> str:
    """Normalize a tool call's JSON-string `arguments` for dedup comparison (sorted keys); fall back to stripped raw."""
    try:
        return json.dumps(json.loads(arguments), sort_keys=True)
    except (json.JSONDecodeError, ValueError, TypeError):
        return (arguments or "").strip()

def _migrate_inline_reasoning(message: Dict[str, Any]) -> None:
    """Extract inline `<think>...</think>` blocks from `message["content"]` into `message["reasoning_content"]`.

    Idempotent: once the tags are stripped from content, a later pass finds nothing to do. Mutates `message`.
    """
    content = message.get("content")
    if not isinstance(content, str) or "<think>" not in content.lower():
        return
    blocks = [block.strip() for block in _migration_think_block.findall(content)]
    blocks = [block for block in blocks if block]  # drop empty (e.g. Gemma's ghost `<think></think>`)
    message["content"] = _migration_think_block.sub(" ", content).strip()
    if blocks:
        extracted = "\n\n".join(blocks)
        existing = message.get("reasoning_content") or ""
        message["reasoning_content"] = f"{existing}\n\n{extracted}" if existing else extracted

def _migrate_inline_tool_calls(message: Dict[str, Any]) -> None:
    """Dedup inline `<tool_call>...</tool_call>` blocks in content against `message["tool_calls"]`, then strip them.

    If a block isn't already represented structurally, parse and add it. On malformed JSON, leave the message's
    content unchanged (lossless fallback — better to show the literal tags than to crash on load). Idempotent.
    Mutates `message`.
    """
    content = message.get("content")
    if not isinstance(content, str) or "<tool_call>" not in content.lower():
        return
    existing = message.get("tool_calls") or []
    seen_keys = {(((tc.get("function") or {}).get("name", "")),
                  _normalize_tool_call_arguments((tc.get("function") or {}).get("arguments", "")))
                 for tc in existing}
    new_calls: List[Dict] = []
    for raw in _migration_tool_call_block.findall(content):
        try:
            parsed = json.loads(raw.strip())
        except (json.JSONDecodeError, ValueError):
            logger.warning("upgrade_datastore: malformed inline <tool_call> JSON; leaving this message's content unchanged.")
            return  # degraded but lossless — don't strip, don't extract
        name = parsed.get("name", "")
        arguments = parsed.get("arguments", {})
        if not isinstance(arguments, str):  # OAI convention stores arguments as a JSON *string*
            arguments = json.dumps(arguments)
        key = (name, _normalize_tool_call_arguments(arguments))
        if key not in seen_keys:  # not already in the structured tool_calls -> add it
            seen_keys.add(key)
            position = len(existing) + len(new_calls)
            new_calls.append({"type": "function",
                              "function": {"name": name, "arguments": arguments},
                              "id": f"migrated_{position}",
                              "index": str(position)})
    message["content"] = _migration_tool_call_block.sub(" ", content).strip()
    if new_calls:
        message["tool_calls"] = existing + new_calls

def _migrate_tool_call_id(payload: Dict[str, Any]) -> None:
    """Move `generation_metadata["toolcall_id"]` -> `message["tool_call_id"]` (OAI spelling). Idempotent. Mutates `payload`."""
    message = payload.get("message", {})
    if message.get("role") != "tool":
        return
    generation_metadata = payload.get("generation_metadata")
    if not generation_metadata or "toolcall_id" not in generation_metadata:
        return
    message["tool_call_id"] = generation_metadata.pop("toolcall_id")  # full move: single source of truth on the message

def _migrate_content_to_parts(message: Dict[str, Any]) -> None:
    """Wrap a legacy bare-string `message["content"]` into a single text content-part. Mutates `message`.

    Runs *after* the reasoning / tool-call normalization stanzas above, which operate on the string form of
    `content`; this one changes the shape. The persona prefix in old assistant content (`"Aria: ..."`) rides
    along inside the wrapped text
    verbatim — no special handling. Idempotent: already-parts content passes through unchanged. Raises (via
    `normalize_content`) on content that is neither str nor list, surfacing datastore corruption.
    """
    message["content"] = normalize_content(message["content"])

def _migrate_text_file_source(payload: Dict[str, Any]) -> None:
    """Backfill the `source` field on `text_file` content-parts from the sidecar provenance. Mutates `payload`.

    Attached documents predate the `source` field on the part (added 0.2.8 so that the wire builder can tell a
    document the user handed over from a page the model fetched — see `text_file_content_part`). The value was
    always recorded, just elsewhere: `general_metadata["sidecars"][<filename>]["source"]`, written when the
    sidecar was stored.

    So this recovers the *true* value rather than assuming one, which matters because both kinds already exist
    in stored chats — any datastore that saw a long `webfetch` before this migration was written has
    `"tool_result"` documents in it, and defaulting them to a user attachment would hand the model an
    unceilinged page forever after. Takes the whole `payload` rather than the message, because provenance and
    parts live on opposite sides of that boundary; that split is the reason the field has to be copied at all.

    A part whose sidecar has no recorded provenance falls back to `"user_attachment"`, matching
    `attachment_budget_kind`'s treatment of an unknown source: send it whole rather than silently truncate
    something that may have been asked for. Idempotent - a part that already has a `source` is left alone.
    """
    sidecars = (payload.get("general_metadata") or {}).get("sidecars") or {}
    for part in payload["message"].get("content") or []:
        if not isinstance(part, dict) or part.get("type") != "text_file":
            continue
        text_file = part.setdefault("text_file", {})
        if text_file.get("source"):
            continue
        url = text_file.get("url", "")
        filename = url[len(sidecarstore.SIDECAR_SCHEME):] if url.startswith(sidecarstore.SIDECAR_SCHEME) else ""
        text_file["source"] = (sidecars.get(filename) or {}).get("source") or "user_attachment"

# v0.2.3+: data format change
def upgrade_datastore(llm_settings: env,
                      datastore: chattree.Forest,
                      system_prompt_node_id: str) -> None:
    """Upgrade the chat `datastore` payloads to the latest format, modifying `datastore` in-place.

    If the chat datastore's payloads are already in the latest format, no changes are made.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.

    `system_prompt_node_id`: The ID of the initial system prompt node (root node)
                             that starts a chat.

                             The reason we need this is that even in the old format (up to v0.2.2),
                             the system prompt node has no extra fluff saved on it, so we can use it
                             to get a list of system-level keys a chat node *should* have.

                             On other nodes, any keys that do NOT match those system-level keys
                             are assumed to be metadata added by the chat client. They are copied
                             to each existing data revision on the node (independent deepcopy for
                             each revision), and deleted from the top level of the node, so that
                             the top level contains only the system keys.

    This also normalizes reasoning and tool-call storage to the current format, independently
    of the v0.2.3 payload-format change above:

      - inline `<think>...</think>` reasoning in `message["content"]` is moved to the `reasoning_content`
        sibling field;
      - inline `<tool_call>...</tool_call>` blocks in content are deduped against / merged into the structured
        `message["tool_calls"]`, then stripped from content;
      - the tool-response linkage moves from `generation_metadata["toolcall_id"]` to `message["tool_call_id"]`
        (OAI spelling).

    Every step is idempotent, so a second load over already-migrated data produces no changes.

    NOTE: There are two upgrade functions for the chat datastore.

    The forest datastore itself also changed in v0.2.3 to allow for data revisioning.
    That part is automatically handled when an old datastore is loaded.
    See `chattree.PersistentForest._upgrade`.

    This function is meant to be explicitly called by a chat client. This upgrades
    the chat payload format.

    Up to v0.2.2, the chat message was stored in `node["data"]` directly, so that
    a node's "data" field content was an OpenAI-compatible chat message record::

        {"role": ..., "content": ..., "tool_calls": ...}

    In v0.2.3+, the `node["data"]` field is revisioned:

        {revision_id: payload,
         ...}

    Additionally, in the payload, the OpenAI-compatible chat message record
    now lives under the "message" key inside the `payload` part:

        {revision_id: {"message": {"role": ..., "content": ..., "tool_calls": ...},
                       "retrieval": {"query": ..., "results": ...},
                       ...},
         ...}

    thus allowing the chat client to add arbitrary other keys to the payload.
    These can be used to store metadata (for the chat client and/or for the user).

    For example, the "retrieval" key stores the RAG query and its retrieval results,
    which is useful for collecting attributions in the chat client (as well as for debugging).
    """
    with datastore.lock:
        # Get the names of system-level keys a chat node should have. Even in the old format (up to v0.2.2),
        # no extra keys are ever created on the system prompt node, so we can use this node to get an
        # up-to-date list (since `PersistentForest` auto-upgrades upon loading if the data format has changed).
        system_keys = set(datastore.nodes[system_prompt_node_id].keys())

        for node in datastore.nodes.values():
            payload_revisions = node["data"]  # {revision_id: payload, ...}

            # v0.2.3: Upgrade payload format
            for payload in payload_revisions.values():
                if "message" not in payload:  # old format?
                    message = copy.copy(payload)
                    payload.clear()
                    payload["message"] = message

            # v0.2.3: Move any non-system keys on the node to under the revisioned data (one copy per revision; will become copies upon JSON saving anyway)
            existing_keys = list(node.keys())
            for key in existing_keys:
                if key not in system_keys:
                    value = node.pop(key)
                    for payload in payload_revisions.values():
                        payload[key] = copy.deepcopy(value)

            # v0.2.3: Add general metadata (message timestamp and persona name)
            #
            # For nodes with missing general metadata, we copy the node's top-level timestamp to all revisions.
            #
            # Also, we populate the "persona" field from the current `llm_settings`, as this is more predictable than parsing it from the text content.
            # Versions prior to 0.2.3 have no support for changing the personas, anyway, so this should also be correct (in most cases).
            timestamp, unused_weekday, isodate, isotime = make_timestamp(node["timestamp"])
            for payload in payload_revisions.values():
                if "general_metadata" not in payload:
                    role = payload["message"]["role"]
                    payload["general_metadata"] = {"timestamp": timestamp,
                                                   "datetime": f"{isodate} {isotime}",
                                                   "persona": llm_settings.personas.get(role, None)}

            # Normalize reasoning + tool-call storage to the current format. Move inline `<think>` reasoning
            # into the `reasoning_content` sibling field; dedup/extract inline `<tool_call>` into structured
            # `tool_calls`; move the tool-response linkage onto `message["tool_call_id"]`. Each step is
            # idempotent, so this is safe to run on every load (old data and already-migrated alike).
            for payload in payload_revisions.values():
                message = payload["message"]
                _migrate_inline_reasoning(message)
                _migrate_inline_tool_calls(message)
                _migrate_tool_call_id(payload)
                _migrate_content_to_parts(message)  # wrap legacy string content as parts (runs last, since it changes the shape)
                _migrate_text_file_source(payload)  # needs the parts list, so it runs after the shape change

def factory_reset_datastore(datastore: chattree.Forest, llm_settings: env) -> str:
    """Reset `datastore` to its "factory-default" state.

    **IMPORTANT**: This deletes all existing chat nodes in the datastore, and CANNOT BE UNDONE.

    The primary purpose of this function is to initialize the chat datastore when it hasn't been created yet.

    This creates a root node containing the system prompt (including the character card), and a node for the AI's initial greeting.

    Returns the unique ID of the initial greeting node, so you can start building chats on top of that.

    You can obtain the `settings` object by first calling `setup`.
    """
    with datastore.lock:
        datastore.purge()
        root_node_id = datastore.create_node(payload=create_payload(llm_settings=llm_settings,
                                                                    message=create_initial_system_message(llm_settings)),
                                             parent_id=None)
        new_chat_node_id = datastore.create_node(payload=create_payload(llm_settings=llm_settings,
                                                                        message=create_chat_message(llm_settings,
                                                                                                    role="assistant",
                                                                                                    text=llm_settings.greeting)),
                                                 parent_id=root_node_id)
        return new_chat_node_id


# --------------------------------------------------------------------------------
# Chat message cleanup utilities

_complete_thought_block = re.compile(r"([<\[])(think(ing)?[>\]])(.*?)\1/\2\s*", flags=re.IGNORECASE | re.DOTALL)  # opened and closed correctly; thought contents -> group 4
_incomplete_thought_block = re.compile(r"([<\[])(think|thinking)([>\]])(?!.*?\1/\2\3)(.*)", flags=re.IGNORECASE | re.DOTALL)  # opened but not closed; thought contents -> group 4
_doubled_think_tag = re.compile(r"([<\[])(think|thinking)([>\]])\n([<\[])(think|thinking)([>\]])", flags=re.IGNORECASE | re.DOTALL)
_nan_thought_block = re.compile(r"([<\[])(think|thinking)([>\]])\nNaN\n([<\[])/(think|thinking)([>\]])\n", flags=re.IGNORECASE | re.DOTALL)
_thought_begin_tag = re.compile(r"([<\[])(think|thinking)([>\]])", flags=re.IGNORECASE | re.DOTALL)
_thought_end_tag = re.compile(r"([<\[])/(think|thinking)([>\]])", flags=re.IGNORECASE | re.DOTALL)

def remove_persona_from_start_of_line(persona: Optional[str],
                                      text: str) -> str:
    """Transform e.g. "User: blah blah" -> "blah blah", for every line in `text`.

    `persona`: The persona name speaking `text`, or `None` if the role has no persona name ("system" and "tool" are like this).

               To get the **current session's** persona, use::

                   persona=llm_settings.personas.get(role, None)

               where `role` is one of "assistant", "system", "tool", "user".

               To get the **stored** persona from a chat node::

                   persona=node_payload["general_metadata"]["persona"]

               This may differ from the current session's persona, e.g. if the chat node was generated with a different AI character.

    `text`: The text to process.

    Returns the processed text.
    """
    if persona is None:
        return text
    # `[ \t]*` rather than `\s+`, for two reasons. It must match with *nothing* after the colon: an assistant
    # turn that only makes tool calls has "Aria:" as its entire text, and requiring trailing whitespace left
    # that marker on screen with no message under it. And it must not cross a line break, or "Aria:\n\ntext"
    # would lose the blank line that separates its paragraphs. The optional `\n` then removes the line the
    # marker was alone on, instead of leaving it empty.
    _persona_at_start_of_line = re.compile(f"^{re.escape(persona)}:[ \t]*\n?", re.MULTILINE)
    text = re.sub(_persona_at_start_of_line, r"", text)
    return text

def get_node_message_text_without_persona(datastore: chattree.Forest,
                                          node_id: str) -> str:
    """Format a chat message from `node_id` in the datastore, by stripping the persona name from the front.

    This is useful e.g. for displaying the message text in the linearized chat view,
    or for sending the message into TTS preprocessing (`avatar_controller.send_text_to_tts`).

    Returns the tuple `(role, persona, text)`, where:

        `role`: One of the roles supported by `raven.librarian.llmclient`.
                Typically, one of "assistant", "system", "tool", or "user".

        `persona`: The persona name of `role`, as it was stored in the chat node.
                   If the role has no persona name, then this is `None`.

        `text`: The text content of the chat message with the persona name stripped,
                at the node's current payload revision.
    """
    node_payload = datastore.get_payload(node_id)  # auto-selects active revision  TODO: later (chat editing), we need to set the revision to load
    message = node_payload["message"]
    role = message["role"]
    persona = node_payload["general_metadata"]["persona"]  # stored persona for this chat message
    text = content_to_text(message["content"])
    text = remove_persona_from_start_of_line(persona=persona,
                                             text=text)
    return role, persona, text

def scrub(persona: Optional[str],
          text: str,
          thoughts_mode: str,
          markup: Optional[str],
          add_persona: bool) -> str:
    """Heuristically clean up the text content of an LLM-generated message.

    `persona`: The persona name speaking `text`, or `None` if the role has no persona name ("system" and "tool" are like this).

               To get the **current session's** persona, use::

                   persona=llm_settings.personas.get(role, None)

               where `role` is one of "assistant", "system", "tool", "user".

               To get the **stored** persona from a chat node::

                   persona=node_payload["general_metadata"]["persona"]

               This may differ from the current session's persona, e.g. if the chat node was generated with a different AI character.

    `text`: The text content of the message to scrub.

    `thoughts_mode`: one of "discard", "markup". or "keep". What to do with thought blocks,
                     for thinking models.

    `markup`: used when `thoughts_mode='markup'`. Which markup kind to use, or `None` for no markup. One of:
        "ansi": ANSI terminal color codes.
        "markdown": Markdown markup, with HTML tags for colors.
        `None` (the special value): no markup. (Same effect as setting `thoughts_mode='keep'`.)

    `add_persona`: Whether to format the scrubbed text as e.g. "AI: blah blah" (standard chat storage convention),
                   or just "blah blah" (e.g. for feeding into scripts).

                   This formatting is only added if `persona is not None`.

    Returns the scrubbed text content.
    """
    _yell_if_unsupported_markup(markup)

    if thoughts_mode not in ("discard", "markup", "keep"):
        raise ValueError("scrub: Unknown thoughts_mode '{thoughts_mode}'; valid values: 'discard', 'markup', 'keep'.")

    # First remove any mentions of the AI persona's name at the start of any line in the text.
    # The model might generate this anywhere - before the thought block, or after the thought block.
    #
    # E.g. "AI: blah" -> "blah".
    #
    # This is important for consistency, since many models randomly sometimes add the persona name, and sometimes don't.
    #
    text = remove_persona_from_start_of_line(persona=persona,
                                             text=text)

    # Fix the most common kinds of broken thought blocks (for thinking models)
    text = re.sub(_doubled_think_tag, r"\1\2\3", text)  # <think><think>...
    text = re.sub(_nan_thought_block, r"", text)  # <think>NaN</think>

    # September 2025 update: This seems to work with Qwen 3 2507, too.
    #
    # QwQ-32B: the model was trained not to emit the opening <think> tag, but to begin thinking right away. Still, it sometimes inserts that tag, but not always.
    #
    # Also sometimes, the model skips thinking and starts writing the final answer immediately (although it shouldn't do that). There's no way to detect this case
    # on the fly, because the opening <think> tag is *supposed to* be missing from the output when the model works correctly. The only way we can detect this is
    # when the output is complete; there won't be a closing </think> tag in it.
    #
    # At least in my tests, QwQ-32B always closes its thought blocks correctly, so if </think> is missing, it means that the model didn't generate a thought block.
    # If </think> is there, then it did.
    #
    # So we search for a closing </think>, and if that's there, but there is no opening <think>, we add the opening tag.
    #
    # What we have here works when there is at most one think block in the message - should be sufficient in practice.
    # TODO: Should we add the opening <think> already when streaming, or even add it to the prompt? How can we add a partial message with the API? Drawback: prevents the model from replying without thinking even in simple cases.
    #
    g = re.search(_thought_end_tag, text)
    if g is not None and re.search(_thought_begin_tag, text) is None:
        text = f"{g.group(1)}{g.group(2)}{g.group(3)}\n{text}"  # Prepend the message with a matching beginning think tag (for QwQ-32B, it's "<think>", but let's be general)

    # Now we should have clean thought blocks.
    # Treat them next.
    if thoughts_mode == "discard":  # for cases where we're not going to read them anyway (e.g. when we pipe the output to a script that only needs the final answer)
        text = re.sub(_complete_thought_block, r"", text)
        text = re.sub(_incomplete_thought_block, r"", text)
    elif thoughts_mode == "markup":  # For cases where we want to see the thought blocks. Colorize them. (TODO: Maybe make some kind of data structure instead.)
        # Colorize thought blocks (thinking models)
        #
        # TODO: This colorizes for text terminals for now; support also HTML colorization. Something like:
        # r"<hr><font color="#a0a0a0">\4</font><hr>"  -- simple variant
        # r"<hr><font color="#8080ff"><details name="thought"><summary><i>Thought</i></summary><font color="#a0a0a0">$4</font></details></font><hr>"  -- complete thought
        # r"<hr><font color="#8080ff"><i>Thinking...</i><br><font color="#a0a0a0">$4<br></font><i>Thinking...</i></font><hr>"  -- incomplete thought
        #
        if markup == "ansi":
            blue_thought_start = colorizer.colorize("⊳⊳⊳Thought⊳⊳⊳", colorizer.Fore.BLUE)
            blue_thought_end = colorizer.colorize("⊲⊲⊲Thought⊲⊲⊲", colorizer.Fore.BLUE)
            def _colorize(match_obj):
                s = match_obj.group(4)
                s = colorizer.colorize(s, colorizer.Style.DIM)
                return f"{blue_thought_start}\n{s}{blue_thought_end}\n"
        elif markup == "markdown":
            blue_thought_start = '<font color="#808080ff">⊳⊳⊳Thought⊳⊳⊳</font>'
            blue_thought_end = '<font color="#808080ff">⊲⊲⊲Thought⊲⊲⊲</font>'
            def _colorize(match_obj):
                s = match_obj.group(4)
                s = f'<font color="#a0a0a0">{s}</font>'
                return f"{blue_thought_start}\n-----\n{s}\n-----\n{blue_thought_end}\n"

        if markup is not None:  # one of the supported markup types was picked?
            text = re.sub(_complete_thought_block, _colorize, text)
            text = re.sub(_incomplete_thought_block, _colorize, text)
    # else do nothing, i.e. keep thought blocks as-is.

    # Remove whitespace surrounding the whole text content. (Do this last.)
    text = text.strip()

    # Postprocess:
    #
    # If we should add the persona name, now do so at the beginning of the text content, for consistency.
    # It will appear before the thought block, if any, because this is the easiest to do. :)
    #
    # The main case where we DON'T need to do this is when piping the output to a script, in which case the chat framework
    # is superfluous. In that use case, we really use the LLM as an instruct-tuned model, i.e. a natural language processor
    # that is programmed via free-form instructions in English. Raven's PDF importer does this a lot.
    # Empty content is left empty: a message whose whole content would be "Aria: " is a speaker label with
    # nothing spoken. That shape is not hypothetical - an assistant message that only requests a tool call
    # carries no text, and Raven's own injected tool calls are of exactly that kind.
    if add_persona and persona is not None and text:
        text = f"{persona}: {text}"

    return text
