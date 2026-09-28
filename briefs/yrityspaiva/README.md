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

## Queue, in order

1. **Message editing v1** (added 2026-09-28). Floated for the Night and set aside for the docs pass, which
   was the right call — people scanned the QR code there, and an up-to-date manual is what they meet. The
   material is in `briefs/researchers-night/README.md`, "Message editing joins the slack" and "Message editing
   v1 moves behind all of it": the backend is complete and tested, every message row already carries a
   disabled pencil button with `Ctrl+E` reserved, and the open question is what the UI does about replies
   below an edited message. The manual's own promise bounds it — small edits that do not change the flow of
   the chat — which may settle most of that question; the building session decides.
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
     - Order: editing first, then the history view.
     - Already in hand: `rebuild_in_place` repaints one message, which settles the "switchable without
       regenerating the whole view" marker.
2. **The two items filed on the Night, 2026-09-25** (added 2026-09-28, after message editing): in
   `TODO_DEFERRED.md`, rerolling a tool-calling reply can leave the vertical spacing wrong, and making
   `websearch` cancellable (`investigations/abort-inflight-request/`). **The second is wider than its
   title**: `webfetch` at least, and possibly other tools — survey them all when it is picked up
   (maintainer, 2026-09-28).
3. **Sprint cleanup**: `researchers-night/` still holds five open briefs, none of which shipped for the
   Night. Rehome them — here if anything is for the 8th, otherwise to `design/` or the top level — and close
   that folder into `done/`.
