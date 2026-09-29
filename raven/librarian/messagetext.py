"""How a stored chat message reads as text.

The small grey line above a message, the generation statistics under it and their per-phase breakdown, the
note under a reply that stopped early, and the forms a message takes when it leaves the chat log — copied on
its own, or as part of an exported log.

No DearPyGui anywhere in this module: everything here reads a datastore and a payload, and returns a string,
which is what lets it be tested without a GUI. `chat_controller` draws what this formats.
"""

__all__ = ["format_chat_message_for_clipboard",
           "format_excerpt_notice",
           "format_message_metadata_line", "format_message_metadata_parts",
           "format_generation_stats",
           "phase_breakdown_rows",
           "node_is_unfinished",
           "incompleteness_note",
           "docs_match_snippet", "collapse_docs_match",
           "document_body",
           "clipboard_text",
           "export_text",
           "format_message_for_clipboard"]

import logging
logger = logging.getLogger(__name__)

from typing import Any

from . import chattree
from . import chatutil
from . import sidecarstore
from . import textfilestore

# --------------------------------------------------------------------------------
# A message in the chat log: its grey line, its statistics, and why it stopped

def format_chat_message_for_clipboard(message_number: int | None,
                                      role: str,
                                      persona: str | None,
                                      text: str,
                                      add_heading: bool,
                                      tool_name: str | None = None) -> str:
    """Format a chat message for copying to clipboard, by adding a metadata header as Markdown.

    As a preprocessing step, `persona` is stripped from the beginning of each line in `message_text`.
    It is then re-added in a unified form.

    `message_number`: The sequential number of the message in the current linearized view.
                      If `None`, the number part in the formatted output is omitted.

    `role`: One of the roles supported by `raven.librarian.llmclient`.
            Typically, one of "assistant", "system", "tool", or "user".

    `persona`: The persona name speaking `text`, or `None` if the role has no persona name ("system" and "tool" are like this).

               To get the **current session's** persona, use::

                   persona=llm_settings.personas.get(role, None)

               where `role` is one of "assistant", "system", "tool", "user".

               To get the **stored** persona from a chat node::

                   persona=node_payload["general_metadata"]["persona"]

               This may differ from the current session's persona, e.g. if the chat node was generated with a different AI character.

    `text`: The text content of the chat message to format.
            The content is pasted into the output as-is.

    `add_heading`: Whether to include the message number and role's character name
                   in the final output.

                   Example. If `add_heading` is `True`, then both::

                       Lorem ipsum.

                   and::

                       Aria: Lorem ipsum.

                   become::

                       *[#42]* **Aria**: Lorem ipsum.

                   If `add_heading` is `False`, then both become just::

                       Lorem ipsum.

    `tool_name`: Which tool produced a `role="tool"` message, from `chatutil.tool_name_of`, or `None`
                 where the node does not say or the role is not "tool". Appears in the heading, so a
                 pasted tool result names the tool that wrote it: `` `<<tool [websearch]>>` ``.

    Returns the formatted message.
    """
    if add_heading:
        message_heading = chatutil.format_message_heading(message_number=message_number,
                                                          role=role,
                                                          persona=persona,
                                                          markup="markdown",
                                                          tool_name=tool_name)
    else:
        message_heading = ""
    message_text = chatutil.remove_persona_from_start_of_line(persona=persona,
                                                              text=text)
    return f"{message_heading}{message_text}"

def format_excerpt_notice(node_payload: dict, full_length: int) -> str:
    """Format the line marking a tool result whose text in the log is only an excerpt of its document.

    `node_payload`: The chat node payload, at the revision being shown.
    `full_length`: How many characters the whole document runs to.

    Names the document where the message says what it is called, so a reader can match the note against
    the attachment rather than merely learning that something is missing. Falls back to the indefinite
    article for a message that names nothing, which is honest and still says the useful half.

    Italic, and bracketed as the constellation's own voice: this is the log remarking on the message, not
    the message speaking.

    Returns the formatted line.
    """
    name = None
    for part in node_payload.get("message", {}).get("content") or []:
        if part.get("type") == "text_file":
            name = (part.get("text_file") or {}).get("name")
            break
    what = f'"{name}"' if name else "the attached document"
    return f"*[Excerpt. The full text of {what} — {full_length:,} characters — is attached to this message.]*"


def format_message_metadata_line(node_payload: dict, role: str, revision: int) -> str:
    """Format the small grey line above a message: when it was written, which revision, and what produced
    it — the tool, for a result; the model, for a reply.

    `node_payload`: The chat node payload, at the revision on screen.
    `role`: The message's role. `"tool"` contributes the tool that answered and `"assistant"` the model
            that wrote the reply; every other role has nothing to add.
    `revision`: Which stored revision of the payload is being shown.

    The bracketed part is spelled as the chat graph spells it in a box's speaker line, so the two views
    name one thing one way. It is absent wherever the node does not record it — a call that failed before
    it had a function to name, a reply interrupted before its model was written down, anything stored
    before either field existed — and the line then carries what it always did.

    Returns the formatted line.
    """
    return " ".join(part for part in format_message_metadata_parts(node_payload, role, revision) if part)

def format_message_metadata_parts(node_payload: dict, role: str, revision: int) -> tuple[str, str, str]:
    """The three parts of `format_message_metadata_line`, for a view that draws them separately.

    Returns `(when, revision_label, producer_label)`, where `producer_label` is `""` when there is nothing
    to say. The chat log draws the revision label as a link, which is why it is a part of its own.
    """
    # What produced this message, for the two roles that have an answer: the tool for a result, the model
    # for a reply. One bracket serves both, because a reader asking where a message came from is asking
    # one question and does not care which kind of answer comes back.
    if role == "tool":
        maybe_producer = chatutil.tool_name_of(node_payload)
    elif role == "assistant":
        maybe_producer = chatutil.model_of(node_payload)
    else:
        maybe_producer = None
    return (node_payload['general_metadata']['datetime'],
            f"R{revision}",
            f"[{maybe_producer}]" if maybe_producer else "")

def format_generation_stats(*, n_tokens: int, dt: float, exact: bool = True, label: str | None = None) -> str:
    """Format a token count, a wall time and the speed between them, as the chat log shows them.

    `exact`: whether `n_tokens` is a count rather than an estimate. An estimate is marked with a `~`, the
             same way the context-fill readout marks one — a number that only claims to be about right
             should say so, or it will be quoted back as though it were measured.
    `label`: an optional lead-in *inside* the brackets, e.g. `"Thought for"`. The message's own figures need
             none — they sit under the message and are obviously about it — but a second set of figures
             elsewhere on screen has to say what it counts, or the two look alike and mean different things.

    One function because the message's own line and its thinking trace's line have to look alike: they are
    the same three quantities over different spans, and a reader compares them at a glance.
    """
    tilde = "" if exact else "~"
    # No speed where there is no time to divide by. A phase can genuinely have none — a turn that thought
    # and then asked for a tool has no answer phase, and its leftover tokens are the tool call, generated
    # inside a span this split cannot see into. Printing `0.00t/s` for them states a measurement that was
    # never made; showing two figures instead says exactly what is known.
    lead = "" if label is None else f"{label} "
    if dt < 0.005:
        return f"[{lead}{tilde}{n_tokens}t, {dt:0.2f}s]"
    return f"[{lead}{tilde}{n_tokens}t, {dt:0.2f}s, {tilde}{n_tokens / dt:0.2f}t/s]"

def phase_breakdown_rows(generation_metadata: dict, *,
                         ended_in_tool_call: bool = False) -> list[tuple[str, str, str, str]] | None:
    """Where a reply's wall time went, as `(label, time, tokens, speed)` rows. `None` when none was recorded.

    The cells carry bare numbers; their units belong in the table's header, where they are stated once.

    Up to four rows: prompt processing, thinking, answer, total. Only the first two phases are *stored* —
    the answer is whatever is left of the total once they are taken off, which is why nothing here can
    disagree with the line it explains.

    **A cell is `""` where the quantity does not apply**, and that is the point of the shape. Prompt
    processing has no token count of its own worth showing and no honest speed (a warm KV cache still
    reports the whole prompt as its size). A turn that thought and then asked for a tool has no answer
    *duration*, so no speed either, though its leftover tokens — the tool call — are real and shown.

    One cell per quantity, rather than a bracketed triple per row, because the columns are what a reader
    compares: which phase took the time, and which produced the tokens.

    `ended_in_tool_call`: whether this message asked for a tool instead of replying. Then there is no answer
                          at all — its tokens are the call itself, and they were generated inside a span
                          this split cannot see into, since tool-call deltas arrive through the structured
                          accumulator and never raise a content event. The row says so by name, and leaves
                          the time blank rather than claiming the zero that subtraction produces.
    """
    phases = generation_metadata.get("phases")
    if not phases:  # absent on a node stored before this was recorded, and on a reply that generated no text
        return None
    total_tokens = generation_metadata["n_tokens"]
    total_dt = generation_metadata["dt"]
    prefill_dt = (phases.get("prefill") or {}).get("dt", 0.0)
    thinking = phases.get("thinking")

    def row(label: str, dt: float | None, n_tokens: int | None = None, exact: bool = True):
        tilde = "" if exact else "~"
        time_cell = "" if dt is None else f"{dt:0.2f}"
        tokens_cell = "" if n_tokens is None else f"{tilde}{n_tokens}"
        # No speed without a time to divide by, and none for a phase whose tokens we do not count.
        speed_cell = "" if (n_tokens is None or dt is None or dt < 0.005) else f"{tilde}{n_tokens / dt:0.2f}"
        return (label, time_cell, tokens_cell, speed_cell)

    rows = [row("Prompt processing", prefill_dt)]
    if thinking is not None:
        exact = thinking.get("tokens_exact", False)
        rows.append(row("Thinking", thinking["dt"], thinking["n_tokens"], exact))
        answer_dt = total_dt - prefill_dt - thinking["dt"]
        if ended_in_tool_call:
            rows.append(row("Tool call", None, total_tokens - thinking["n_tokens"], exact))
        else:
            rows.append(row("Answer", answer_dt, total_tokens - thinking["n_tokens"], exact))
    elif ended_in_tool_call:
        rows.append(row("Tool call", total_dt - prefill_dt, total_tokens))
    else:
        rows.append(row("Answer", total_dt - prefill_dt, total_tokens))
    rows.append(row("Total", total_dt, total_tokens))
    return rows

def node_is_unfinished(datastore: chattree.Forest, node_id: str) -> bool:
    """Whether `node_id` holds a reply that never finished arriving.

    True while a turn is streaming into it, and true afterwards for a reply the app was interrupted in the
    middle of — a turn cannot outlive the process, so a node still marked this way when a datastore is read
    back was cut short. The two are told apart by whether a turn is running, not by the node.

    Written by `scaffold.ai_turn`, which creates the node before the reply and clears the marker when the
    reply is complete.
    """
    generation_metadata = datastore.get_payload(node_id).get("generation_metadata") or {}
    return generation_metadata.get("status") == "incomplete"

def incompleteness_note(generation_metadata: dict) -> str | None:
    """What to say under a reply that stopped early, or `None` for one that ended on its own.

    `generation_metadata`: the node's, or `{}` for a node that has none.

    Two ways a reply ends before the model was finished, and they are different events, so they get
    different words. The reader needs no more than that; the *reason* a backend failed is already in the
    message text, which is why an error is not one of the cases here.
    """
    # Rendered rather than stored, and that is the whole design. Storing the line would put words into the
    # assistant's mouth: the continue machinery would then have to strip them again before sending the
    # message back to the model, and anything else reading the stored text — a script, an export — would
    # have to know to ignore them. A line that never enters the text has nothing to strip.
    #
    # Worth having at all because a stopped reply looks exactly like a finished one. Both end mid-thought
    # often enough, and by the next session nobody remembers pressing the button.
    if generation_metadata.get("interrupted"):
        return "[Interrupted — the reply was stopped here]"
    if generation_metadata.get("status") == "incomplete":
        # Only reachable from disk: a live turn cannot outlive the process, so a node still marked this way
        # when the datastore is read back was cut off by the app going away rather than by the user.
        return "[Incomplete — Raven exited while this reply was being written]"
    return None

def docs_match_snippet(text: str, max_characters: int = 200) -> str:
    """A few lines from one match of a document search, for the collapsed chat log. `""` for a part with no text.

    `text`: A match as `chatutil.format_docs_match` writes it — a line naming the document and the offset,
            then the matched text, then a rule. The first line is not part of the snippet.
    `max_characters`: About how long the snippet may be; see `chatutil.excerpt`, which cuts it.

    The whitespace is folded to single spaces, and a cut is marked at the end of the last line rather than on
    a line of its own, so the snippet runs as prose however the document broke its lines — as a websearch
    result's snippet does.
    """
    _first_line, _, rest = text.strip().partition("\n")
    rest = rest.strip().removesuffix("-----")
    snippet = " ".join(rest.split())
    return chatutil.excerpt(snippet, max_characters, inline_marker=True) if snippet else ""

def collapse_docs_match(text: str, max_characters: int = 200) -> str:
    """Shorten one part of a document search's result to its first line and a snippet of the rest.

    `text`: A match as `chatutil.format_docs_match` writes it, or the result's one-line heading, which is
            returned as is. See `docs_match_snippet`.

    For a result whose parts cannot be matched to their documents, which is one stored before the tool
    recorded that; the first line is then the only place the document is named.
    """
    first_line = text.strip().partition("\n")[0]
    snippet = docs_match_snippet(text, max_characters)
    return f"{first_line}\n\n{snippet}" if snippet else first_line

# --------------------------------------------------------------------------------
# A message leaving the chat log: copied on its own, or as part of an exported log

def document_body(datastore: chattree.Forest, node_payload: dict[str, Any]) -> str | None:
    """The full text of the document this message reports, or `None` if it does not report one.

    `datastore`: The datastore holding the message, whose sidecars hold any attached document.
    `node_payload`: The chat node payload, at the revision on screen.

    "Document" is the category the chat log gives a handle to, and membership is *declared*, never guessed
    from length. Two ways in, matching the two ways a document reaches a message:

      - a `text_file` part, whose sidecar holds the text (a page `webfetch` stored, or a file the user
        attached) — the stored text part is only an excerpt, so the body comes from the sidecar; and
      - a `fetch_document` result, whose text *is* the body, sitting inline because a knowledge-base
        document has no sidecar and should not get one (the file is already the user's).

    Everything else answers `None` and renders unchanged — notably `websearch`, whose result can be long
    but is a list of links the user wants to see and click, not a document to put behind a toggle.

    **Tool messages only**, which is load-bearing rather than a narrowing for tidiness. A user message
    carrying an attached document has a `text_file` part too, and its text part is the user's own words;
    treating that as a document result would replace what they wrote with an excerpt of what they
    attached. An attached document is not inlined into the chat log at all, by design — the chip is its
    handle — and that stays true.

    An unreadable sidecar also answers `None`, which degrades to rendering the stored excerpt as ordinary
    text: less than we wanted, but never a message that shows nothing. That arrives as a *value* rather
    than an exception — `textfilestore.sidecar_to_text` answers `NO_EXTRACTABLE_TEXT` instead of raising,
    so that one bad attachment can never break a wire-build — which is why it is compared for.
    """
    message = node_payload["message"]
    if message.get("role") != "tool":
        return None
    for part in message.get("content") or []:
        if part.get("type") == "text_file":
            url = (part.get("text_file") or {}).get("url", "")
            if url.startswith(sidecarstore.SIDECAR_SCHEME):
                try:
                    body = textfilestore.sidecar_to_text(datastore, url)
                except Exception as exc:  # noqa: BLE001 -- rendering must not fail on one unreadable sidecar
                    logger.warning(f"document_body: could not read '{url}': {type(exc)}: {exc}")
                    return None
                # `sidecar_to_text` answers a placeholder rather than raising, so the `except` above
                # catches nothing this case can throw and the failure arrives as a *value*. Treated as
                # no body at all, which falls back to the stored excerpt: 800 real characters of the
                # page beat a notice saying we could not read it, and the excerpt is right there in
                # the message. The notice would otherwise reach the clipboard as the message's text.
                return None if body == textfilestore.NO_EXTRACTABLE_TEXT else body
    if (node_payload.get("generation_metadata") or {}).get("function_name") == "fetch_document":
        return chatutil.content_to_text(message.get("content"))
    return None

def clipboard_text(datastore: chattree.Forest, node_payload: dict[str, Any]) -> str:
    """The text a copy of *one message* should carry: what it holds, in full.

    `datastore`: The datastore holding the message.
    `node_payload`: The chat node payload, at the revision on screen.

    The stored text, except for a tool result whose document was moved to a sidecar. There the stored
    text is an 800-character excerpt, so returning it hands back a truncation with nothing saying it
    was one — lossy in a way an ordinary attachment is not, since a user message keeps its own words
    and *gains* a chip, where this message's content was replaced.

    The sibling of `export_text`, which answers the same question for a whole log and answers it
    differently. The gestures differ, not the mechanism: asking for one message is asking for its data as
    it stands, where a log is a document to be read and passed on.

    Returns the text.
    """
    # The defect this fixes, seen in the live app: a webfetch result too long for the chat log copied
    # as far as the log had shown it, while the attached file was intact.
    maybe_body = document_body(datastore, node_payload)
    if maybe_body is not None:
        return maybe_body
    return chatutil.format_message_text_for_export(node_payload["message"])

def export_text(datastore: chattree.Forest, node_payload: dict[str, Any]) -> str:
    """The text of one message as a whole-log export should carry it.

    `datastore`: The datastore holding the message.
    `node_payload`: The chat node payload, at the revision on screen.

    The stored text, with a line naming the attachment wherever that text is only an excerpt of one —
    so a reader of the log can tell that something was elided and what it was. The sibling of
    `clipboard_text`, which answers the same question for a single message and answers it with the
    whole document, that gesture being a request for one message's data as it stands.

    Returns the text.
    """
    # Two reasons the log keeps the excerpt where a single copied message takes the document. The
    # second is not about the code at all, which is why it is written down:
    #
    #   - Several fetched pages inlined at full length make the log unreadable, which is most of what
    #     a log is copied *for*. One message is a lift of that message; a log is a document.
    #   - A log is the artifact that gets *shared*, where a single copied message is typically pasted
    #     somewhere the person is working. So a log carrying complete copies of pages that were only
    #     ever quoted invites a complaint that an excerpt would not — the same content is unremarkable
    #     as a quotation and awkward as a redistribution, and only the log is likely to travel.
    #
    # So the elision stays and stops being silent, which was the whole of the original defect.
    maybe_body = document_body(datastore, node_payload)
    stored_text = chatutil.format_message_text_for_export(node_payload["message"])
    if maybe_body is None:
        return stored_text
    return f"{stored_text}\n\n{format_excerpt_notice(node_payload, len(maybe_body))}"

def format_message_for_clipboard(datastore: chattree.Forest,
                                 node_id: str,
                                 *,
                                 role: str,
                                 persona: str | None,
                                 include_node_id: bool) -> str:
    """Return exactly what copying one message puts on the clipboard.

    `datastore`: The datastore holding the message.
    `node_id`: The message's chat node. Its active revision is the one copied.
    `role`, `persona`: As shown on screen for the message. See `format_chat_message_for_clipboard`.
    `include_node_id`: also name the speaker, and prefix the message with its node ID, the timestamp
                       of the active payload revision, and that revision's number. This is what
                       holding Shift over the copy button asks for.
    """
    # The keyboard is read by the caller and arrives here as an argument, which is what lets the
    # result be checked without one. Same split as the Visualizer's report copy, and for the same
    # reason: a function that reads the modifier itself can only be exercised by faking `is_key_down`.
    node_payload = datastore.get_payload(node_id)  # auto-selects active revision

    # The speaker's name rides along with the node ID and not otherwise: omitting it makes a copied
    # question convenient to paste back into the chat field and edit before re-submitting.
    #
    # Text from the stored payload rather than from the widget's rendered text, so that this and the
    # full-log export say the same thing about the same message. The rendered form joins the widget's
    # paragraphs and drops their `is_thought` flag, so a thinking model's trace came out welded to its
    # answer with nothing between them, and a reader could not tell where one ended.
    formatted_message = format_chat_message_for_clipboard(message_number=None,  # a single message copied to clipboard does not need a sequential number
                                                          role=role,
                                                          persona=persona,
                                                          text=clipboard_text(datastore, node_payload),
                                                          add_heading=include_node_id,
                                                          tool_name=chatutil.tool_name_of(node_payload))

    # A lifted fragment travels without the document manifest the full-log export carries, so it needs
    # its own - same format, because a one-message manifest and a fifty-message one should not need two
    # parsers. Human turns get none: there is no AI generation to disclose, and a YAML block on a copied
    # question would just be something to delete before pasting it back into the chat field.
    if role != "user":
        manifest = f"{chatutil.format_disclosure_manifest([node_payload])}\n"
    else:
        manifest = ""

    if include_node_id:
        payload_datetime = node_payload["general_metadata"]["datetime"]  # of the active payload revision!
        node_active_revision = datastore.get_revision(node_id)
        header = f"*Node ID*: `{node_id}` {payload_datetime} R{node_active_revision}\n\n"
    else:
        header = ""
    return f"{manifest}{header}{formatted_message}\n"
