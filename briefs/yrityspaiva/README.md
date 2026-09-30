# Yrityspäivä sprint

*A sprint README is the sprint's decision log, not an introduction to it: what was scheduled and in which
order, what was cut and on what argument, what was learned while building each item. Most of it is here
because it produced no diff and would otherwise be re-decided from scratch.*

**Deadline: 8 October 2026** — Yrityspäivä. The same system as Researchers' Night, demoed to an SME
audience. Opened 2026-09-28, from the maintainer's post-event list and the issues flagged
live on the Night.

## Done

- **Librarian's attach dialog closed itself on the first Ctrl+Shift+O after launch** (flagged live on the
  Night). Traced to a hidden `Tooltip` window being shown in the frame the modal opened; fixed in
  `raven.common.gui.tooltip`, which now shows no window for a text change nobody is looking at. The same fix
  removed the blue-border flicker the `give_caret` retry had made visible on Tab. Write-up in `dpg-notes.md`.
- **Tab flashed a caret in Librarian's search field for a frame.** ImGui's own Tab step, confined with
  `flattened_navigation=False` on the composer's child window. The Visualizer was checked live and needs
  nothing; Raven-cherrypick gives nothing focus programmatically and so sees no step.
  - **That fix's sequel, 2026-09-29:** Tab or Shift+Tab from the log or the graph to the search field then
    flashed the *composer*. The step starts from the send button, where those two panes park focus, and
    the button now shared a scope with the composer alone. The toolbar row got a scope of its own.
    Confirmed live in both directions.
- **`FileDialog` and the thumbnail grid, found by following the Tab thread** (all 2026-09-28):
  - The path field went active for a frame on every Tab: ImGui's own Tab step lands on another text field
    in the same navigation scope. Each field now has a scope of its own. This closed the open question in
    `investigations/dpg-focus/`.
  - Clicking the find field, a row or a tile now moves the keyboard mark there, and a row click moves the
    table's cursor. A click on the blank space below a short listing does not claim the keys:
    `is_item_hovered` on a child window answers only for that exact window, so there is nothing to ask.
  - A tile click was sometimes dropped mid-tile: a press makes ImGui report the panel unhovered from the
    next frame, and the click handler read the hover live. It now reads a sample taken while the button
    was up.
  - A tile's tooltip wore the listing's blue border when the listing had the keys. Fixed with
    `keyboardmark.shield_tooltip`.
- **The XDot viewer's search keys follow the Visualizer's** — Tab between the field and the graph,
  Ctrl+Shift+F to clear, and every hand-back of the keys parks focus on a button rather than on the graph
  widget's group. Its open dialog also gained a *Graph files* filter, default, covering every supported type.
- **Librarian's indicators say DOCUMENTS and INTERNET**, as the toggles do, and only their icons pulsate;
  the names hold still. The phase reset per indicator stays, out of step with INDEXING's red: visibility of a
  briefly shown indicator was preferred over breathing in step, which the code now says.
- **Every app's window title is `Raven-<app>`**, as its command is.
- **A Librarian quickstart**, at the top of its README: LM Studio set up, Raven-server started, the model
  loaded, then Librarian — server before model, as the troubleshooting section explains for a shared GPU.
  The appendix now names LM Studio first, and says an 8 GB card runs Librarian with a 4B-class model and the
  server in low-VRAM mode. The main README says which GPU backends are untested (ROCm, XPU) and asks for
  reports.
- **The avatar's render time broken down by phase** on the debug overlay, timed with `torch.Event` so the
  numbers are true without slowing the renderer. That retired `metrics_enabled`, whose device syncs existed
  only to make the old per-phase timers true, and the server's DEBUG log lost its per-frame lines.
- **Delete a subtree from the chat graph** (2026-09-28). A trash can on the toolbar after the branch button,
  and **Shift+Delete** — chosen as the file-manager "delete, no wastebasket" key, and different from the chat
  log's Ctrl+Shift+Delete so a habit cannot carry across. Two presses, and the second confirms only the
  message the first one armed, since this button follows the cursor.
  - Both views go through `DPGChatController.delete_subtree`, over `chatutil.delete_subtree` and
    `chatutil.is_deletable`, which moved down so the graph could reach them. HEAD moves only when the subtree
    held it, and the chat log rebuilds only when the deleted node's parent is on its branch.
  - Found on the way: the chat log's delete ran during a reply, and could delete the node being written
    into. Now refused while a turn is in flight, on the first press. A per-delete check was priced and
    rejected — a turn's write point does not exist yet while it is queued, which is where it would go wrong.
  - Found in live testing: the click that gave the graph the keyboard committed, because taking the keyboard
    puts the cursor on HEAD and the click then read as a second click. The cursor now records whether the
    panel placed it, and coming back to the graph never counts as a second click on the ringed box
    (maintainer's call).
- **Server autostart** and **the DB behind the server** are written up for later —
  `briefs/server-autostart-brief.md`, and brief 13's *Where the database lives*. Neither is for this demo.

- **Message editing v1** (2026-09-28). Ctrl+E or the pencil edits a user message or an AI reply in place, as
  a new revision; a message with more than one revision draws its `R` number as a link and says how many it
  has, and the link or Ctrl+Shift+E opens `DPGRevisionPanel` to show or delete each. The background was in
  `briefs/researchers-night/README.md`, "Message editing joins the slack".
  - Found on the way, and fixed: a keyboard mark moved onto a widget another thread had just deleted took
    the render loop down (`keyboardmark.Mark._set_target`), which is how Librarian died once at startup.
  - New in the shared layer: `gui_animation.give_focus`, `give_caret`'s sibling for widgets that hold focus
    rather than a caret — a single `focus_item` from a mouse click did not land.
  - The eight `TODO: later (chat editing)` markers are retired: every reader wants the active revision.
   - **Settled 2026-09-28** (design session with the maintainer):
     - **Inline editing, GitHub-style**: the message's Markdown is replaced in place by a multiline field
       with Save/Cancel, via `rebuild_in_place`. Ctrl+Enter saves (gated on `is_item_focused` — it is the
       committing chord), Esc cancels, global hotkeys yield while the field has the caret. Rejected: a
       modal (cheapest, but edits away from what is being read), and loading into the composer (a mode in
       the send path, the attachment pile and the user's draft — a leak waiting to happen). The modal is
       the fallback if sizing the field proves fiddly. **The field does not word-wrap** — DPG 2.3.1's
       `InputText` has no such option, and the composer lives with the same limit. Accepted for v1; a
       custom multiline text box is the eventual fix for both (maintainer, 2026-09-28).
     - **User and AI message text only.** System prompt, greeting and tool messages are not editable.
       Thinking traces and attachments carry over unchanged: the new revision is the old payload with its
       text parts replaced and a fresh timestamp. **An edited AI message keeps its
       `generation_metadata`** — the edits are typo fixes on user messages and, on AI messages, trimming
       excess length before sharing a chat log; neither changes which model wrote it.
       **Editing the thinking trace is a later version's**, and it has a use: a model stuck in a loop can
       sometimes be unwedged by cutting the loop out of its trace and continuing (maintainer, 2026-09-28).
       That depends on Continue resuming an *incomplete thinking trace*, which is probably not supported
       yet (maintainer's recollection). Checked only this far: on a prefill backend `llmclient.invoke`
       seeds the old `reasoning_content` into the stored result, but whether the trace reaches the model
       on the wire is unchecked.
     - **The replies below an edit get nothing** — no stale mark, no reroll prompt. The manual's own
       promise (small edits that don't change the flow) is the scope, and the `R` number is the "edited"
       marker.
     - **The revision history is in v1**: clicking the `R` number opens a list of revisions (number,
       date, first line) with *show* (make active) and *delete*. The `R` must look clickable. It is also
       the first way to reach the revisions Continue has always been making. First to cut if time runs out.
     - **Refused while a reply is being written**, as delete is. **Ctrl+E**, into the help card and README.
       Beside the data-integrity reason in the code, there is a UX one: editing while the AI is still
       writing has no known use case, so not supporting it costs nothing until one turns up (maintainer).
     - **Settled in live testing**: the field's size is fine for v1; one message open at a time; the save
       chord is the composer's send key; an empty message with nothing else in it is refused; unchanged
       text saves nothing.
     - **The history view is non-modal** (maintainer, 2026-09-28): a DPG modal dims the whole window,
       which would look bad for this. So it takes the non-modal pane pattern — `has_keyboard()` before
       `handle_key()` in the app's key chain, like the audio input panel. The `R` number is drawn in a
       link colour with a tooltip. **Ctrl+Shift+E** opens it for the message with the keyboard mark.
     - Order: editing first, then the history view.
     - Already in hand: `rebuild_in_place` repaints one message, which settles the "switchable without
       regenerating the whole view" marker.
     - **Settled in live testing of the history view**: Enter or a click on the revision already shown
       closes the panel, so a double-click shows and closes (maintainer's idea); Ctrl+Shift+E does nothing
       on a message with one revision, as its `R` is then no link.
     - **The count reads "(3 revisions)", right after the `R` link** (2026-09-29), where it had been
       "(3 revisions available)" after the model name. Rejected: `R2/3`, since revision numbers stay unique
       after a deletion — a message holding R1 and R3 would read `R3/2` (maintainer).
- **Rerolling a tool-calling reply no longer leaves a gap** (2026-09-29). `DPGChatMessage.demolish` emptied a
  message's container and left it standing, and an empty DPG group still takes 4 px of item spacing
  (measured), so each message a reroll rewound left 4 px behind, as did every finished streaming message.
  `demolish` is now a teardown, container included, and `build` refuses a demolished instance, pointing at
  `rebuild_in_place` — the flicker-free redraw, which made a demolish-then-build cycle useless anyway
  (maintainer's call, over keeping the container for a rebuild). Confirmed live.
- **`chat_controller.py` extractions** (2026-09-29): 6238 → 5237 lines. An analysis found about half of it in
  blocks with narrow interfaces, and the rest Kolmogorov-hard (`ai_turn`'s closures, the message rendering
  core, the view's scrolling) — worth doing even so, since the prose around parenthetical material is its own
  cognitive load (maintainer). Rejected as low value for the untangling: context fill, the per-message button row.
  - `messagetext` (formatters, copy/export text; DPG-free, tests now in CI); `layout_math.decide_tail_follow`
    (fuzz-checked equivalent); `chattextures` (`SpeakerGlyphs`, `AttachmentTextures`; tags now unique per
    instance; glyphs looked up by name for the AI as for the user — maintainer's review); `chatlog_search`
    (`DPGChatLogSearch`, given the chat graph panel's search interface, so the app builds one query for both).
  - Also: a view rebuild now demolishes the messages it drops, so `get_current_message`'s promise holds.
- **Found in live testing the same day, and fixed**: a failed webfetch named no URL (every webfetch result now
  opens "Webfetch result for" + URL — "for", since the result may be an error; maintainer); a chat graph
  search step sometimes glided past its match (a rebuild mid-morph shifted the pan's destination by the morph
  still to run — `XDotWidget.set_graph`); committing a graph box with a thinking-trace hit did not open the
  trace (a stored message's paragraphs spent the request while it was still being built).
  - And a graph search step rebuilt three times — cursor, then the graph's keyboard flag switched off and on by
    release-then-claim — each mid-morph rebuild restarting the morph. The claimant now names the pane it is
    claiming, which the release leaves alone: one rebuild per step.
- **Document search results no longer flood the chat log** (2026-09-29). `search_documents` returned up to 50
  matches of up to 2000 characters as one string, rendered in full — `k` = 50 is deliberate, being the one
  retrieval knob that measurably mattered. Now a heading with the count and query, then one text part per
  match; a long result opens collapsed to a handle on each match's document (title, open, folder — as
  `fetch_document`'s) and a two-line snippet, cut inline as a websearch snippet is. Both search paths give
  the model the same text (`chatutil.format_docs_search_result`).
  - **Decided along the way**: B over A (an excerpt of the whole, which showed one match and a half) and over
    C for now; the count in the heading rather than the tooltip, for model and reader alike; the snippet cut
    by **character budget, not by sentence** — matches are sliding-window chunks that start mid-sentence
    anyway, much of a corpus is not prose (BibTeX), and spaCy per match per render is a model call ×50
    (maintainer agreed). If a snippet reads badly, snapping to a sentence end late in the budget is a regex,
    as `excerpt` already snaps to a paragraph break.
  - **C followed the same afternoon**: each match has its own chevron, on its handle row. The result's own
    chevron commands them, as expand-all / collapse-all toggles usually do: it opens all while any is
    closed, and closes all once every one is open (maintainer's refinement). So the per-match set
    (`expanded_parts`) is the only state a match has; `show_full_text` serves the results with no parts.
  - The per-match handle is also a start on the deferred item on RAG citations' source files; a reply's own
    citations are still open there.
- **Also from the same live test, done**: empty send always answers the user's own last message (so a
  question can be asked again after deleting its replies), whatever `llm_allow_empty_send` says; Ctrl+T now
  closes a trace scrolled out of view (`is_item_shown`, not `is_item_visible` — swept for the pattern, which
  found one more in the pose editor); `fetch_document`'s refusal names the unknown ID; a rescan repoints
  documents whose recorded path is gone, which repaired the indexes v0.2.9's `llmclient/` → `librarian/`
  move had stranded (2520 documents in one, no reindex).
  - **Rejected**: a Raven-side per-extension override for the document open button, when `.bib` opened in
    Zotero. Desktop file associations are the uniform way users know (maintainer).
- **Brief 13 gained §4a** (2026-09-29): with a scope TOC published to the model, the automatic search goes
  away. Not for this sprint.
- **Librarian at 4K, and the avatar editors' keyboard** (2026-09-30, prompted by Qwen 3.8's volume of tool
  calls and a 4K display):
  - A window wider than its default gives the extra width to the chat graph; the chat log stays at its
    default width.
  - The avatar fills 98% of its panel's height at any size. Past the configured `upscale` ceiling the
    client enlarges the frames itself, bilinearly (`investigations/dpg-texture-filtering/`), with
    `avatar_config.display_scaling` "fit" (default, chosen after trying both) / "integer" / "off". The
    settings editor previews it with an "x drawn" slider, preview only. Resizing no longer triple-swaps the
    avatar's texture.
  - Ctrl+Shift+click on a message's copy button copies its node ID alone, for reporting a message.
  - Avatar editors: keys for every chooser, Esc out of one, the record button on Ctrl+Shift+Enter, the
    settings editor's voice key off Ctrl+V, its stats overlay behind a checkbox and Ctrl+M.
  - `keyboardmark.focus` keeps a focus move's transit from lighting a mark on the way.
  - **Rejected:** a settle time before a mark lights, which would have delayed the feedback a fast typist
    steers by (maintainer); Ctrl+Shift+Space for record, which IBus claims.
  - **Open, not filed:** the parks in Librarian, the Visualizer and the XDot viewer still use
    `dpg.focus_item`. Moving them to `keyboardmark.focus` needs `gui_animation.give_caret` to set the
    expectation too, or a park would hold back the next move's mark. No flash has been seen there.

## Queue, in order

1. **Make the tools cancellable, and the web tools fail in prose** — filed on the Night, 2026-09-25, in
   `TODO_DEFERRED.md` (`investigations/abort-inflight-request/`). Designed 2026-09-29 with the maintainer.
   - **Survey.** Only `websearch` and `webfetch` can wait long (Raven-server; `webfetch`'s headless tier the
     slowest). `search_documents` may make server round-trips through its `MaybeRemote` embedder and
     tokenizer: seconds. The rest are local and instant.
   - **Mechanism: (b) now, then (c).**
     - (b) Each tool call runs on a worker thread; the turn waits on the call *or* the abort, and on abort
       stops waiting. One place, the `perform_tool_calls` dispatch, covering every tool. The orphaned thread
       runs until the server answers or the timeout expires, and its result is discarded.
     - (c) The web endpoints send headers at once and the result as a streamed body, so `Abort.arm(response)`
       works unchanged, and the server can notice the client has gone and stop scraping. Changing the API
       is fine: Raven-server always ships version-matched with the client, so both ends change together.
     - Rejected: (a), reaching the socket at connect time through a custom transport adapter. More private
       urllib3 internals and a Windows variant, for less coverage than (b), since `MaybeRemote` calls
       bypass it.
   - **What a Stop mid-round leaves.** Finished calls keep their results; each unfinished call gets a tool
     result saying the user cancelled it; the turn ends, with no further LLM round. A `tool_calls` message
     without a result per call would make the next request malformed under the OpenAI schema.
   - **Stop ends the turn.** The other verb — go on without that tool — is the empty send, which becomes
     always allowed on a tool node, as it is on a user message.
   - **The GUI's cancel hook** aborts during a tool round as well as before anything has streamed: one more
     flag on `task_env`. `retry_tool_calls` goes through the same dispatch and gets all of this for free.
   - **The web tools' errors, from the Night.** DuckDuckGo flaked, and each search waited about 30 s and
     returned an empty result: the server's wait for the results element gives up after 5 s and carries on,
     the scroll loop then waits up to 25 s more, and zero links go back as a success. So:
     - the server tells "no results" from "the results never appeared", and sets a page-load timeout on the
       driver;
     - `llmtools` answers in canonical prose, as the document tools do — no results; the engine did not
       respond; web search unavailable — in place of both the empty list and the exception text;
     - a web-tool timeout of its own in Librarian's config, passed through as an optional `timeout=` on the
       two api calls, since `network_timeout` covers every server call. 30–45 s, to leave room for
       `webfetch`'s headless tier.
   - Order: the empty send on tool nodes, the web tools' errors and timeout, (b), then (c). **All but (c)
     done 2026-09-29**, (b) confirmed live by the maintainer — Stop during a slowed websearch, then an empty
     send to go on. **(c) done 2026-09-30**, confirmed live — Stop during a slow fetch, and the server logged
     the client gone:
     - `raven.server.util.stream_job` streams a slow job's JSON; `raven.client.util.post_streamed_job` is its
       client side. It took two server settings, both measured in `investigations/abort-inflight-request/`:
       `COMPRESS_STREAMS = False` (Flask-Compress held the headers back) and waitress's
       `channel_request_lookahead=1` (without it the server never sees a client go).
     - The server stops between steps only. A page load already in progress runs to its end, which in the
       live test was six seconds after the Stop.
     - From the maintainer's review: `raven.server.util` is the shared server-side helper module, `stream_job`
       its first occupant; `raven/server/modules/webcommon.py` holds what the two web modules share (driver
       factory, user agent, the `WebToolException` family, `lock_unless_cancelled`). Nothing else in `app.py`
       or the modules was found to belong in either.
2. **Sprint cleanup**: `researchers-night/` still holds five open briefs, none of which shipped for the
   Night. Rehome them — here if anything is for the 8th, otherwise to `design/` or the top level — and close
   that folder into `done/`.
