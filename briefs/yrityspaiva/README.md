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
- **Server autostart** and **the DB behind the server** are written up for later —
  `briefs/server-autostart-brief.md`, and brief 13's *Where the database lives*. Neither is for this demo.

## Queue, in order

1. **The small items**, in this order:
   1. **A Librarian quickstart**: install an LLM backend such as LM Studio; in LM Studio, enable the
      Developer tab in its settings, switch the API on in that tab, and load the model *on the Developer
      tab* so it is served through the API; then start Raven-server; then start Librarian. Also fix the
      README's backend section, which says LM Studio "needs no setup beyond starting its local server".
   2. **Avatar timing breakdown in the debug overlay**: pose, upscale and postprocessor times, beside the
      render/encode/output ones already there. The server's `render` figure covers all three today. The
      animator very likely has the timings already (the upscaler's is `tim_upscale`); this is sending them
      in `X-Server-Stats` and showing them.
2. **Delete a subtree from the chat graph** — medium. A toolbar button with the usual double-press and red
   warning flash, deleting the subtree at the cursor. Disabled on the active system prompt node and the
   active AI greeting, which are the only nodes whose deletion would break the running instance. A hotkey
   *different* from the chat log's Ctrl+Shift+Delete, so a reader cannot delete the wrong data by habit,
   and as hard to hit by accident — to be proposed.
3. **Sprint cleanup**: `researchers-night/` still holds five open briefs, none of which shipped for the
   Night. Rehome them — here if anything is for the 8th, otherwise to `design/` or the top level — and close
   that folder into `done/`.
