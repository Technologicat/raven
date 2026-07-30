# Deferred TODOs

New items go at the **top**. (Both ends were in use up to 2026-07-27, which is how the two halves of the same
Librarian session ended up ~1000 lines apart.)

## The avatar upscaler offers bilinear and bicubic, but not Lanczos

`raven.common.video.upscaler.Upscaler`'s `quality` parameter takes `"low"` / `"high"` (Anime4K model sizes)
or `"bilinear"` / `"bicubic"` (bypass Anime4K entirely, straight to `torch.nn.functional.interpolate`).
`raven.common.image.lanczos` belongs in that second group and is missing from it — it is GPU-enabled, already
a Raven dependency, and takes `(B, C, H, W)` in and out, which is exactly the shape the bypass branch already
juggles with its `.unsqueeze(0)` / `[0]`. Its docstring states it works for both directions, so nothing about
the upscaling use is out of scope for it.

Two things to get right in the bypass branch (`Upscaler.upscale`):

- **Alpha stays bilinear.** The branch already splits RGB from alpha for bicubic, because bicubic's negative
  lobes ring at silhouette edges. Lanczos is a windowed sinc with *several* lobes, so it has more negative
  lobe than bicubic, not less — the same reasoning applies with more force. What changes is that the split's
  hardcoded `mode="bicubic"` has to become the selected filter, and the `c == 3 or quality == "bilinear"`
  fast path stays as it is.
- **`order` stays at `DEFAULT_ORDER`** (decided by Juha, 2026-07-29). `lanczos.resize` takes an `order`
  (kernel size / ringing trade-off) that `F.interpolate` has no equivalent for, so the question came up of
  whether to surface it. It should not be: `quality` is a short list of named presets rather than a knob
  panel, and the bypass filters it would sit beside take no parameters either. Pass the default and leave
  the parameter alone.

The real cost is not the code but the **duplicated option list**: the valid values appear in `Upscaler`'s
validation and docstring, `raven/avatar/settings_editor/app.py:625` (hardcoded list) and `:477`,
`raven/server/config.py:278`, `raven/librarian/config.py:470`, and `raven.client.mayberemote`'s docstring —
several of them carrying the same hand-maintained "what each value means" comment. Adding one entry means
editing all of them, and the next addition will too. Worth considering whether the list should have a single
source of truth while touching them anyway.

Note also that `quality` already mixes two different axes — model size for Anime4K, filter choice for the
bypass. Lanczos joins the second axis, so it fits the existing shape; it does make the conflation more
visible, but untangling it is a separate (and API-breaking) question.

Raised by Juha (2026-07-29).

## `app.py` and `minichat.py` each name the datastore paths separately

Both frontends independently build the same two paths:

```python
datastore_file = librarian_config.llmclient_userdata_dir / "data.json"  # chat node datastore
state_file = librarian_config.llmclient_userdata_dir / "state.json"     # important node IDs for the chat client state
```

(`app.py` ≈ line 246, `minichat.py` ≈ line 70.) They agree today, which is load-bearing — the GUI and the CLI
are supposed to share one datastore, and that only holds because two separate literals happen to match.
Nothing enforces it, and nothing would notice if one drifted; the symptom would be a CLI that quietly cannot
see the chats the GUI made.

The directory is already config (`llmclient_userdata_dir`), so the fix is to move the two filenames there
too and have both frontends read them. Small, but it converts an invariant maintained by coincidence into one
maintained by construction.

Noticed while documenting the on-disk layout (2026-07-30, Juha — "which is silly").

## Updating the vendored FontAwesome means both files, not just the header

`CLAUDE.md` notes the vendored `IconsFontAwesome6.py` is an outdated version. Measured 2026-07-30, the
situation is more specific than that: the header and the shipped fonts are **exactly in sync** —
`fa-solid-900.ttf` carries 1969 codepoints over 1395 distinct glyph names, the header names all 1395, and no
glyph in the font lacks a constant. (The 574-codepoint surplus is pure aliasing: 461 names have two
codepoints, e.g. `angle-down` at both U+2304 and U+F107. It is *not* a set of icons waiting to be exposed.)

So regenerating the header alone would gain nothing. An update means fetching newer `fa-solid-900.ttf` /
`fa-regular-400.ttf` webfonts **and** regenerating the header from the matching `icons.yml`, as one change —
and then checking the font atlas still fits (see `dpg-notes.md`, "Font atlas limits"; more glyphs is exactly
the direction that breaks it).

Concrete motivation, for whenever this is picked up: `arrow-down-to-bracket` does not exist in this version,
which is why the tool-call navigation links use the symmetric `ARROW_UP_LONG` / `ARROW_DOWN_LONG` pair
instead of the `arrow-up-from-bracket` / `arrow-down-to-bracket` pair brief 03 suggested. If the update
happens, that icon choice is worth revisiting — it is a two-line change in `add_tool_call_invocation` and
`build_buttons`.

Discovered while picking icons for the tool-call navigation links (2026-07-30).

## Chat view scrolling: keys, smoothness, and end-of-scroll feedback

Verified 2026-07-30: Librarian's key handler (`app._on_key`) covers F1, F8, F11, Return, the arrows (sibling
navigation), a handful of letters and Escape — and nothing for scrolling the chat log. The only way through a
long chat is the mouse wheel, which is painful when a single tool result runs to dozens of screens (see the
sibling item on webfetch results).

**Visualizer's info panel is the reference implementation**, not a thing to invent: `scroll_to_position`,
`go_to_top`, `go_to_bottom`, the page-up/page-down handlers and `update_navigation_controls` (which
enables/disables the buttons by scroll position) already exist there with the semantics wanted here. Port the
shape rather than a new one, and Librarian gets the on-screen buttons for free as well.

Two traps specific to this:

- **Page Up / Page Down do not match their constants.** They arrive as **517 / 518**, while `dpg.mvKey_Prior`
  / `dpg.mvKey_Next` are stale DPG-1.x values (266 / 267). Comparing against the constant silently never
  fires — no error, just a dead key. Documented in `dpg-notes.md` ("Keyboard input") and CLAUDE.md pitfall 7;
  this is precisely the feature that would hit it.
- **Home / End are ambiguous while the composer has focus.** In a text field they mean start/end of line, and
  the user will expect that. The handler needs to route by focus rather than claim the keys globally.

**Two more gaps in the same area, noticed alongside** (Juha, 2026-07-30) — and all three share the same
reference implementation, so they are one job rather than three:

- **No smooth scrolling.** `scroll_view` sets `dpg.set_y_scroll` directly, so every jump is instantaneous and
  the reader loses their place — including the new tool-call navigation links, where "where did it take me?"
  is the whole question. `raven.common.gui.animation.SmoothScrolling` already exists and is what Visualizer's
  `scroll_to_position` uses.
- **No scroll-past-end feedback.** Visualizer has `ScrollEndFlasher` (an animated overlay, arrows at top and
  bottom) so that hitting the end of the content is visibly the end rather than an unresponsive view.
  Librarian has nothing.

Doing the keys without the other two would be the wrong order: page-down onto an instant jump with no
end-of-content signal is worse than the mouse wheel it replaces.

Raised by Juha (2026-07-30), while scrolling through a chat with several full-page webfetch results.

## Store large tool results as attachments instead of dumping them into the chat log

A `webfetch` result is currently rendered inline as the tool message's text, so fetching a paper drops its
entire body into the chat log — dozens of screens to scroll past, and the same bytes into the datastore JSON
(the 1.1 MB test datastore is mostly fetched article text). It is unreadable as a log and it is the reason
the missing PageUp/PageDown above hurts.

**The attachment machinery already does exactly what is wanted here.** A `text_file` content part stores the
bytes as a content-addressed sidecar, the chat log shows a compact chip instead of the content, and
`llmclient._serialize_history_for_wire` folds the extracted text back into the message at wire-build time —
so *the model sees no difference*, which is the property that makes this safe. Store the fetched document as
a sidecar and the tool message becomes a chip plus, say, the first paragraph.

What falls out for free, beyond the readability:

- **Content addressing.** Fetching the same URL in two branches costs one file instead of two copies.
- **Provenance.** The fetch URL is exactly what `sidecarstore.base_provenance` records, and the existing
  "Open source" / "Show original" buttons then work on tool results too.
- **The chip's name shortening**, and the rest of the attachment display logic.
- **A smaller datastore**, since the JSON keeps a `sidecar:` reference rather than the text.

Design questions to settle first:

- **A size threshold.** A short websearch summary is *better* inline; only large results should become
  attachments. Where the line sits is a judgment call, and the behaviour should not flip confusingly around it.
- **What the message shows instead.** Nothing, a title, or the first paragraph — the last is probably right,
  since a tool result the user cannot see at all is a step backwards from the current what-you-see-is-what-you-get
  design.
- **Whether it applies to `websearch` too**, whose results are a list of snippets rather than one document.
- **Interaction with the context-fill indicator**, which counts document tokens by appending `sidecar_to_text`
  output — this path already exists, so it should just work, but confirm rather than assume.

Raised by Juha (2026-07-30).

## Make the DPG reference a skill, so it loads when it is needed

`CLAUDE.md` says "**Before editing any DPG code, read `dpg-notes.md` first**" and defines what counts as DPG
code. That is about as strong as prose gets, and it still depends on the agent noticing and obeying a line —
which is a weak trigger for a 519-line reference that matters on exactly the tasks where getting it wrong is
expensive.

A **project-scoped skill** in `raven/.claude/skills/` fixes the trigger mechanically: skills are surfaced by
description match, so "editing DPG code" pulls it in without anyone remembering to. Project-scoped rather
than fleet-wide (`~/.claude/skills/`) because the notes cite Raven modules throughout, and a repo-local skill
is version-controlled with the code it documents — so it travels between machines and cannot drift from the
tree it describes. If `raven.common` is ever extracted as `corvid`, promoting it is a move, not a rewrite.

**The skill body must be a router, not a copy.** A short index saying which section of `dpg-notes.md` answers
which question, and nothing else. Duplicating the content is how one of the two copies goes stale, and the
human-facing file has to stay authoritative — that is what a person reads in an IDE, where no skill exists.

Explicitly *not* an `@include` of `dpg-notes.md` in `CLAUDE.md`: that loads all 519 lines into every
conversation, including the ones about BibTeX parsing. See the sibling item "Audit and slim down project
CLAUDE.md" — Raven's has not been through an optimization pass yet (the global one has, and has nothing left
to trim), so adding to it is the wrong direction.

Raised by Juha (2026-07-30), after noticing `dpg-notes.md` is not auto-loaded and so is unlikely to be seen
at the moment it is needed.

## GUI: hardcoded stand-ins for values DPG has no getter for

DPG exposes very few getters for theme state — there is no way to ask a theme for its colors or spacings —
so code that wants to *restore* a value ends up guessing a literal instead. Found while widening `ButtonFlash`
into `WidgetFlash`, where the theme-restore case turned out to be exactly this bug: `finish` rebound a fixed
theme rather than the one the widget actually had, so flashing a widget that had no theme silently left one
on it.

**The tell is intent, not syntax.** A literal is an instance of this whenever the code's *intent* is "be
whatever the theme is" and the literal is standing in for a question it cannot ask. That covers cases which
look at first like ordinary configuration — `plotter_background_color=(37, 37, 38)  # measured from DPG
default theme using GIMP` is not a color someone chose, it is a getter call performed with GIMP. Sorting by
"is it in config.py" gets this wrong; sorting by "would this value have to change if the theme changed" gets
it right.

By that test the scan finds, in Raven's own code:

- `raven/common/gui/animation.py` — `WidgetFlash`'s button path fades to `45, 45, 48` because it cannot ask
  what the background was. Its text path escapes this: `get_item_configuration` *does* report a widget's own
  color, so it fades back to the real value. Where a per-widget getter exists, use it — the gap is theme
  state specifically, not everything.
- `raven/common/gui/utils.py` (≈ line 293) — `disablable_widget_theme`'s disabled colors, with the TODO right
  above them.
- `raven/visualizer/info_panel.py` (≈ line 849) — "the info panel content background color in the default
  theme".
- `raven/visualizer/config.py` — `plotter_background_color` and `plotter_grid_color`, both annotated
  "measured from DPG default theme using GIMP".
- `raven/librarian/config.py` — the four `chat_color_*_back`, all `(45, 45, 48)`: the chat backgrounds want to
  be the app background.
- `raven/librarian/config.py` — `margin=8  # the DPG default theme uses 8 elsewhere`. Worth noting because it
  is a **spacing**, not a color: the missing getters are not only about color, so an audit that greps for RGB
  triples will miss half of it.

Genuinely *not* an instance, for contrast: `vumeter.py`'s `bgcolor = (64, 64, 64)  # cf. DPG default gray:
(45, 45, 48)`. That one deliberately differs from the default and cites it only to say so — its value would
not change if the theme did.

**A naming convention already exists, in two of the smaller apps** (found 2026-07-30 while scanning for the
spacing literals). `raven/xdot_viewer/config.py` and `raven/conference_timer/config.py` name each constant
after the DPG style variable it mirrors:

```python
DPG_WINDOW_PADDING_Y = 8    # mvStyleVar_WindowPadding[1]
DPG_FRAME_PADDING_Y  = 3    # mvStyleVar_FramePadding[1]
DPG_ITEM_SPACING_Y   = 4    # mvStyleVar_ItemSpacing[1]
```

**Adopt that fleet-wide, in preference to naming the constant after its use.** Librarian and Visualizer now
have `margin` / `panel_inner_padding`, which say what the value is *for* rather than what it *is* — a weaker
choice, and demonstrably so: `raven/avatar/settings_editor/app.py:490` computes `2 * 8 + 2 * 8` because
`WindowPadding.x` and `ItemSpacing.x` are **two distinct quantities that both happen to be 8 today**. One
`margin` constant would assert they are the same thing, and the day DPG changes one of them, every use site
has to be re-derived from scratch. Note too that xdot_viewer calls the 3 `FRAME_PADDING`, which suggests
`panel_inner_padding` is misnaming something DPG already has a name for — worth confirming by measurement
(compare `get_item_rect_min` of a child window against that of its first child) rather than by assumption.

Sites found by the scan:

- `raven/visualizer/info_panel.py:248` — `_get_content_area_start_pos`, now `margin + panel_inner_padding`.
- `raven/visualizer/info_panel.py:639` — the same `8 + 3` pair again, computing the content area's *bottom*
  edge (`y0_content + h_content - 8 - 3`). Still literal.
- `raven/librarian/chat_controller.py` — `scroll_view`, now `margin + panel_inner_padding`.
- `raven/avatar/settings_editor/app.py:490, 516, 530` — `2 * 8 + 2 * 8`, a `-16`, and a `pos=(8, 32)`.
- `raven/avatar/pose_editor/app.py` — several `image_size - 16` (i.e. 2×8) plus an `add_spacer(height=8)`.

**Both avatar editors have no `gui_config` at all**, so adopting the convention there means introducing one
(or importing the shared constants from wherever they end up living — arguably `raven.common.gui`, since
these are DPG facts rather than per-app choices).

Two separable pieces of work, and the cheap one is worth doing even if the other never is:

- **Collapse the duplication**, on the convention above. Costs nothing, and makes the blast radius visible —
  right now, "what breaks if the DPG default theme changes?" can only be answered by grepping for magic
  numbers across six modules in five apps.
- **Restore rather than guess, wherever DPG allows it.** Per-widget getters exist for some properties and not
  others; the audit is to find which sites can be converted (as the `WidgetFlash` text path was) and which
  are genuinely stuck until upstream grows a getter. Those that are stuck should at least say so, so the next
  reader knows it is a workaround rather than a choice.

Raised by Juha (2026-07-30), while reviewing the `WidgetFlash` theme-restore fix.

## Web status panel: check on a long job without being at the machine

The motivating case is concrete: a ~12k-abstract hydrogen indexing run, and no way to see how it is doing
except the Librarian window and the terminal that launched it. From a phone, from another room, there is
nothing. "Is it still going, and should I wait?" is the question, and it currently has no remote answer.

**The server half is nearly built.** `raven.server.app` already serves `/` (rendered HTML, from Markdown),
`/health`, and `/api/modules` (which modules are enabled). So "which modules are up" is already there, and
mostly wants a nicer presentation.

**VRAM is the part that is not already solved.** `raven.common.deviceinfo` records what each *load step*
consumes, which is load-time accounting, not live usage — and the two diverge badly for exactly the module
one would most want to watch: the avatar allocates most of its memory when it *starts*, not when it loads.
So a panel built on the load figures would confidently under-report. Live usage wants
`torch.cuda.memory_allocated` (and `mem_get_info` for the whole-device picture) sampled at request time,
plus a decision about attribution — torch reports per *device*, not per module, so per-module numbers are
inference from load-order bookkeeping rather than measurement. Worth deciding whether the panel promises
per-module attribution at all, or just shows the device total plus which modules are resident.

**The Librarian half is the real problem, and it is a process problem, not a data problem.**
`hybridir.get_indexing_progress_text()` already produces the string the GUI mirrors. It just lives in a
desktop GUI process with no listening socket, while the panel needs to be somewhere a phone can reach.

**Note this is the same missing piece that Librarian↔Visualizer talk needs**, which raises the value a lot:
the panel is one customer of an inter-app channel, not a one-off. Solve it once and "show me on the map what
you found" / "discuss *these* items" become wiring rather than architecture. Three shapes, and the middle one
looks best:

- *Librarian opens its own HTTP port.* A listening socket and a firewall hole per desktop app, plus a second
  web stack to maintain. Scales badly the moment Visualizer wants in.
- *Librarian pushes status up to raven-server, which serves the panel.* It is **already** a client of that
  server, so this rides a connection that exists, needs no new socket on the desktop side, and gives one
  address to point a phone at and one auth story to get right. Generalizes to any long-running job in any
  Raven app.
- *The panel queries both.* Requires the desktop app to be reachable, i.e. the first option wearing a hat.

**Push downward is possible, and the machinery is already in the building.** The worry with "the desktop app
has no listening socket" is that it can then only poll. It can do better: the app opens a long-lived GET to
the server and holds it, and the server writes events down that pipe as they happen. The connection is
client-initiated (so no inbound port, no firewall hole, NAT-friendly) while the *data* flows server→client,
which is exactly the shape selection-syncing wants. Both halves are proven here already — `raven.server.app`
serves a long-lived streaming response today (`/api/avatar/result_feed`, `multipart/x-mixed-replace`), and
`llmclient` already consumes SSE from the LLM backend, so neither end needs a new skill. Server-Sent Events
is the natural fit for the panel and for app-to-app events alike: text, one direction, auto-reconnecting.

Two constraints that will decide the design, though:

- **Each held connection costs a waitress thread.** The server runs `serve(app, ...)` at its default thread
  count, and the avatar feed already holds one of them. Three desktop apps plus a phone is fine; this is not
  a design that scales past a handful of subscribers, so bound it deliberately rather than discovering the
  ceiling.
- **It makes raven-server the broker for app-to-app talk**, which cuts against the documented server-optional
  story (Visualizer is deployable standalone by design, via `MaybeRemote`). Acceptable if inter-app sync is
  explicitly a server-present feature that degrades to "apps don't see each other" — but that is a decision to
  make on purpose, not to back into.

Polling is still the right first step for the *status panel alone*: a phone refreshing a page every few
seconds needs nothing built. The push channel earns its keep when selection-syncing arrives, where a
round-trip delay is felt directly.

**This does not have to wait for database unification.** Unification decides whether there is one importer or
several; a generic "job reports its progress" channel does not care either way, and Visualizer's importer can
push through the same channel whenever it is convenient. Worth decoupling so neither blocks the other.

**Status and control are not the same feature, and the difference is the threat model.** Raven-server is
documented as trusted-network-only, unencrypted and unauthenticated. A *read-only* status page on that
footing exposes nothing new. A *control* panel that loads and unloads modules is an unauthenticated "make
this machine allocate 20 GB of VRAM" button — and the phone use case means it is by definition no longer
LAN-only. So:

- **Decided (Juha, 2026-07-30): read-only for now.** Ship status; leave control out entirely.
- Control needs the auth and transport story decided *before* it exists, not patched on after. That is the
  honest answer to whether dynamic load/unload is YAGNI: not that it is too niche to want, but that it is the
  half that carries the whole cost.
- Note the overlap with the existing "Uniform load-on-demand for Raven-server modules" item. If that lands,
  loading and unloading become automatic and a manual control surface is mostly redundant — which is a good
  reason not to build the manual one first.

**Worth showing, beyond "is it indexing":**

- **An ETA and a progress bar, not just a progress string.** `get_indexing_progress_text` is prose for a GUI
  label. On a 12k-item job the question is not "is it running" but "should I wait", and that needs counts —
  which the string is *built from*, so they exist and are merely being formatted away before anyone outside
  can use them. Report the numbers and let the panel render the bar.
  `unpythonic.timeutil.ETAEstimator` is already a dependency.
- **Whether it died.** A page that says "still indexing" is useful; one that can say "stopped 40 minutes ago,
  here is the error" is what actually saves the trip to the machine.
- **The docs-DB pending-edit queue depth** (`hybridir._pending_edits`) — how much is still waiting, as
  distinct from what is being chewed now.
- **Datastore and sidecar directory sizes.** The attachment-browser item below already computes exactly this.

**Design constraints that fall out of "on a phone":** plain server-rendered HTML that reflows, no DPG, and no
dependence on a live socket to show anything — a meta-refresh or a small poll beats a JS app that renders
blank until a websocket connects, over a link that may be poor. And no CDN assets: Raven is local-first and
must work with no internet at all, so whatever CSS/JS there is ships in the repo.

Raised by Juha (2026-07-30), from wanting to check the hydrogen indexing run from a phone.

## Browse *all* attachments in the datastore, not just the orphaned ones

The cleanup dialog (`raven/librarian/cleanup_dialog.py`) turned out to be a decent attachment browser that
happens to be filtered to orphans. Point the same machinery at `list_sidecar_files()` instead of
`list_unreferenced_sidecars()` and it browses the whole sidecar directory. Two use cases, and the second is
the one that recurs:

- A human-readable view of `<datastore>.images/`, which a file manager cannot give: the grid shows titles
  rather than SHA-256 filenames, and folds a downscaled image and its preserved original into one item.
- *"I'm sure I attached paper X at some point, but which chat was it?"* — which the file manager cannot
  answer at all.

**Most of it is already built.** `cleanup.describe_sidecar`, the companion fold that merges original with
downscaled, `SidecarEntry.archival_filename`, the thumbnail grid, and click-to-open-the-original all work off
a list of filenames and do not care where that list came from. What is genuinely new is one datastore method
and one labelling decision.

**The datastore side is one method.** `PersistentForest._referenced_sidecars` already walks every revision of
every node, calls the `sidecar_extractor`, and discards *which* node each hit came from. Collect it instead
and you have `sidecar_reference_map() -> dict[str, set[str]]`; the existing mark phase becomes a projection
of that map, and `excluding_nodes=` a filter on it. Orphans then fall out as the empty-set entries, so the
browser can show them inline, marked, with the same rescue/delete affordances the cleanup dialog gives them
— one code path, not a parallel one.

**Navigating to the message is likewise assembled from parts that exist.** A referencing node names its
place by walking up; `chat_controller.py:924`'s `descend` already picks a leaf below a given node
(it is what the branch switcher uses); `DPGLinearizedChatView.scroll_view(scroll_target_node_id=...)` already
scrolls to a message. Three wrinkles decide the UX, and only the first is obvious:

- **"Which chat" is not a well-formed question** — see the labelling paragraph below. What the picker is
  really choosing between is *labelled subtrees*, and two branches of one root can be two of them. Once the
  labels are right this collapses to one level of ambiguity rather than two; auto-descend then resolves only
  what is left *below* the chosen label.
- **A reference can live in an old revision.** Sidecar references are collected per revision, so an
  attachment may be referenced by a superseded revision of a node whose current revision no longer shows it.
  Scrolling there lands the user on a message that visibly does not contain the thing they searched for.
  Either scope the map to current revisions, or say so in the UI — but decide deliberately rather than
  discovering it in testing.
- **Orphans have no chat to open**, which is exactly why they get the delete button instead. The affordance
  set is per-item, not per-dialog.

**The labelling is the real work, and it is wanted elsewhere anyway.** Nothing in `chattree` or `chatutil`
labels a chat today — a picker listing three of them by node UUID is useless, so this item is gated on it.
But "give each chat a title" is the wrong shape, and the branching history is why: **a label on the root
labels the whole multiverse**, so every branch under it inherits one name — which is precisely the case the
picker exists to tell apart.

Git has the answer already, and it is worth taking wholesale: labels attach to **nodes**, sparsely, and
"where am I" is resolved by *reachability* rather than by a field on the thing you are in. Tags name commits;
branch names name tips; a commit with no name of its own is described by what it is reachable from. The chat
analogue: a label on any node applies to its subtree until a deeper label overrides it, so a node's
displayed name is its **nearest labelled ancestor** (plus the message itself, when that is the finer
distinction the user needs). The root carries an automatic label so there is always a fallback, and a branch
point is exactly where a user would want to add one by hand.

The tiering then has to change with it. `chatutil.document_label`'s *shape* still applies — use the good
source when there is one, fall back to something dug out of the content — but the tiers must be computed
**at the labelled node**, not at the root:

- Stored label, if a human set one. This is the one that matters; the rest exist so the list is usable before
  anyone has.
- Auto-generated from **what diverged at that node**, not from the opening turns. Two branches that split at
  message 20 share every one of those turns, so an opening-turns summary hands them identical names — worst
  exactly where the picker is needed. The divergent subtree is what distinguishes them, so that is what the
  generator has to read.
- The node's own message, truncated.

Store the label on the node, make it editable, and the chat tree browser gets the same field for free.

**One dialog or two** is open. The cleanup dialog's safety framing (dry run, then an explicit destructive
commit) does not belong on a browse-everything window, so the likely answer is two thin dialogs over one
shared grid widget rather than one dialog with a filter dropdown.

Raised by Juha (2026-07-29), right after the cleanup dialog landed.

## The attachment sidecar directory is called `<datastore>.images/`

The name dates from when images were the only kind of attachment; it now also holds PDFs and text documents,
and will hold office formats. It reads as "this was not thought out in advance", and it costs discoverability
for anyone poking around the datastore looking for *attachments*.

`chattree.py`'s comment above `_get_sidecar_dir` argues for keeping it: renaming would strand the sidecars of
every existing datastore, and it is a directory name rather than a description. The first half is the real
objection, and it is answerable — a rename-if-present at load time is a small, one-shot migration. Note it
does not fit in `_upgrade`, which migrates the loaded *nodes* dict and knows nothing about the filesystem, so
it needs its own step in the load path (and the payloads need no change at all: `sidecar:<filename>` URLs
name the file, not the directory).

Worth doing together with the browser item above, since that one puts the directory in front of users for the
first time.

Raised by Juha (2026-07-29), while finishing brief 03.

## Move the avatar backdrop onto `image.utils.fit_cover`

`DPGAvatarRenderer.configure_backdrop` (`raven/client/avatar_renderer.py`) scales its backdrop with PIL —
`scale = max(...)`, resize, crop — which is exactly what `raven.common.image.utils.fit_cover` now does. Porting
it would leave one resampler in the constellation instead of two, and fold in the sibling `# TODO` already
sitting there ("if the backdrop image is small and/or has a wild aspect ratio, would be more efficient to cut
first, then scale").

**Straight port ≠ speedup, so decide which of the two jobs this is.** A local `fit_cover` would run CPU torch
against PIL's C implementation — a wash at best, because `configure_backdrop` is client-side and Librarian's
client does no local GPU work by design. Real acceleration means doing the resize *server-side*, in `imagefx`.

**That is more expensive than it first looks, because the blur round trip cannot carry the resize.** The two
jobs are separate endpoints of the `imagefx` module, not two filters in one chain:

- `/api/imagefx/process` runs the postprocessor chain, and is **resolution-preserving by construction** —
  `postprocessor.render_into(image_rgba)` writes into the tensor it was given. There is no chain entry that
  could change the output size, so a "resize filter" is not a thing that can be added to the blur call.
- `/api/imagefx/upscale` is where resolution changes live, and it is backed by Anime4K — an *upscaler*. A
  backdrop cover-fit is usually a **downscale** (a large wallpaper into a window), so this is likely the
  wrong tool for the job rather than merely a second call.

So server-side resize means either two round trips (send, resize, receive, send again, blur, receive) or a
new Lanczos resize endpoint alongside the two — and the blur is **conditional** (`if new_blur_state:`), so in
the unblurred case there is no server call today to piggyback on at all. Any of those is a real piece of
work, not a redirect of an existing one.

Then there is the byte cost, which points the same way. The resize currently runs **before** the blur, so it
is the already-downscaled image that goes over the wire; delegating inverts that — full-size pixels out,
small ones back.

None of this is urgent: the resize fires on a window resize, never in a hot loop.

Noticed while extracting `fit_contain` / `fit_cover` for the cleanup preview's thumbnail grid (2026-07-29).

## TODO.md goes stale because nothing in the workflow makes anyone visit it

The two lists have different failure rates and the reason is mechanical rather than a matter of discipline.
`TODO_DEFERRED.md` stays fresh because the workflow *writes to it as a side effect of doing other work*:
there is a trigger (notice something unrelated mid-task, append it), a closing ritual (resolved items are
deleted), and a reminder (mention new entries after committing). It is maintained by accident.

`TODO.md` has none of those. It is a plan rather than a byproduct, so nothing forces a visit, and items rot
quietly. Verified tonight: the BibTeX umlaut item had been fully implemented for some unknown length of
time and still sat there marked `[Verify]`, while the RAG-by-tool-call item on the same page was accurate —
because that one pointed at a brief, and closing the brief was a ritual that made someone read it.

That contrast suggests the fix rather than more diligence. **Briefs do not go stale, because closing one is
an event.** So candidate directions, in rough order of how much they change:

- Make TODO.md items *point at* briefs wherever a brief exists, so the brief's status line is the truth and
  TODO.md degrades to an index. Cheap, and already how the accurate items behave.
- Give the `[Verify]` marker a ritual — a periodic sweep that actually runs the checks. Tonight showed
  these markers work exactly when someone runs them and never otherwise.
- Reconcile TODO.md as part of closing a brief, explicitly, the way `CHANGELOG.md` is written alongside a
  fix rather than reconstructed at release time.

Worth thinking about before the list grows further; not worth a big refactor of the file itself.

Discovered while closing brief 10 and finding a stale item next to an accurate one (2026-07-29).

## EU AI Act Article 50 (transparency) compliance

`briefs/ai-act-article-50-summary.md` has the analysis; Commission guidelines were adopted 20 July 2026 and the
Article applies from **2 August 2026**. Raven has been available since 2024, so it is a system already on the
market before that date, which means the **2 December 2026** grace period applies — but only to the 50(2)
machine-readable marking of generated content. The rest applies from August with no grace period.

The implementation is briefed: `briefs/summer_2026_librarian_extension/07_export-provenance-brief.md`, which
scopes it to attaching system-level provenance to exported chatlogs and messages, and explicitly rules out
building text watermarking — the robust 50(2) mark acts on the logits during sampling, and Librarian samples
un-watermarked third-party weights through an OpenAI-compatible backend, so there is nothing post-hoc to add.
Depends on content-parts (brief 03).

Work backwards from December, and note it lands right after Researchers' Night, so the demo build and the
compliance build are the same autumn's work.

Raised by Juha (2026-07-28).

## Make the canned AI greeting optional

A new chat opens with a canned greeting from the AI (`raven.librarian.config`, "Names, AI's greeting"). That is a
2024-ism: as of mid-2026 the first message after the system prompt can just as well be the user's, and an opening
line the AI did not choose is one more thing asserting a personality at a model that does not need it.

Make it optional: blank in config means no greeting node at all, and starting a new chat then points HEAD at the
system prompt instead. The assumption is baked into several places, and one of them fails silently rather than
loudly:

- `chat_controller._get_all_greeting_node_ids` identifies a greeting *structurally*, as any direct child of a
  root (system prompt) node — deliberately, so that a chat stored under an older config's greeting is still
  recognized. With no greeting node, the direct children of the system prompt are **the user's first messages**,
  so every first message in every chat would be classified as a greeting. Reroll, continue, branch-from-here and
  delete are all gated on `node_id not in greeting_node_ids`, so they would quietly go dead on exactly those
  messages. This one needs a real fix, not a length tweak.
- `chat_controller.py:653`, an `assert k < len(...) - 3` whose comment enumerates system prompt + greeting +
  first user message.
- `minichat.py:453`, `len(node_id_history) < 4`, counting the greeting as one of four expected nodes.
- `chatutil.factory_reset_datastore` (creates the node) and `appstate._refresh_greeting` (rewrites it on load),
  which is also where "blank means omit" has to be honoured.

Related: [Modernize the Librarian system prompt / character card] — same question of how much identity the
frontend should assert at a modern model.

Raised by Juha (2026-07-28).

## TTS reads arXiv IDs digit by digit

Qwen likes to cite arXiv papers by their full identifier, and the TTS then says
"twenty twenty six dot zero five ... v three" — long, and the least informative part of the sentence gets the most
speaking time. Detect arXiv IDs in the TTS input (`avatar_controller.preprocess_task`, alongside the existing
Markdown/emoji stripping) and either shorten them to something sayable or drop them from the spoken text while
keeping them in the subtitle, which is where a reader can actually use them.

The spoken and written forms can differ cheaply: `preprocess_task` already holds the sentence in one local and
feeds it to `self.tts.synthesize` and to the subtitler separately, so a second, spoken-form local splits them
without touching the surrounding structure. Prefer that over shortening in place — the ID is exactly the part a
reader wants in full and a listener does not want at all.

Discovered while fixing the zero-segment TTS crash (2026-07-28, reported by Juha).

## DPG now sets up font ranges itself; `setup_font_ranges` is a no-op that logs loudly

Recent DearPyGui configures font ranges automatically, which makes `dpg.add_font_range` redundant — so
`raven.common.gui.fontsetup.setup_font_ranges` (which requests `0x100`–`0x2fff`) no longer does anything, and DPG
logs about it on every Raven GUI app start. Cosmetically noisy now; misleading later, since the function reads as
load-bearing and isn't.

Called from four places: `raven/avatar/pose_editor/app.py`, `raven/common/gui/utils.py` (twice), and
`fontsetup.markdown_add_font_callback`. Decide between version-gating the call (if we still support DPG versions
that need it) and removing it outright, then update `dpg-notes.md`, whose "`setup_font_ranges` and extended
Unicode" section documents the old behaviour as current.

Note the two dev machines are on different DPG versions as of 2026-07-27, so this reproduces on one and not the
other — check the installed version before concluding anything about it.

Discovered while reconciling the TODO lists (2026-07-27, reported by Juha).

Several items are siblings under one root cause and are cheaper to fix as a package than one at a time. The
clusters, as of 2026-07-27:

- **Temporary context injects** — how much goes on the wire each turn, in which role, at which position.
  **Built 2026-07-28**, closing "RAG injects: sent in the user role as a workaround", "Fold the temporary
  context injects…" (measured, and rejected in favour of the system block plus a tool result) and "Revisit
  the 'answer from context only' reminder". Measurements in `briefs/context-inject-shape-measurements.md`,
  the plan they argued for in `briefs/summer_2026_librarian_extension/done/08_context-injects-brief.md`. Still
  open in this cluster: "RAG: rerank retrieved chunks…" and "Modernize the Librarian system prompt /
  character card", plus the new "RAG access via tool-call" motivation recorded under Q11 of the
  measurements — the model asks for a second, better-aimed search and currently has no way to get one.
- **FileDialog** — "slow open and a teardown input-dead-window", "smart-case the Find field", "image thumbnail
  previews", "multi-extension filter as one labelled item", "reduce per-use-site boilerplate", plus "OS
  drag-and-drop of files into DPG apps" (which is why the picker has to be good — it's the only entry path).
- **Markdown renderer** (the vendored `DearPyGui_Markdown`) — "Markdown ATX headings don't render", "Fenced
  code block support", "Reasoning traces with indented bullets mis-render", "inline-code background boxes are
  stranded on dynamic reflow", "Emoji support in the Markdown renderer". Adjacent: "Super/subscript font
  coverage in the GUI" is an *atlas* problem rather than a renderer one (`fontsetup` serves both plain DPG
  text and `dpg_markdown`), but it shares the font-survey work with the emoji item's monochrome-font route.

## Attachment + docs-DB: support office document formats (MS Office / LibreOffice)

**Mostly landed 2026-07-29** (`093c400`): `docextract` now reads word-processor documents (`.docx`, `.odt`),
presentations (`.pptx`, `.odp`) and saved web pages (`.html`, `.htm`) alongside plain text and PDF, for both
the attach path and the docs-DB ingester.

**Spreadsheets (`.xlsx`, `.ods`) remain**, and they are the awkward ones — which is why they were left. A
sheet is not a linear document, so "the text of a spreadsheet" is a design decision before it is an
`openpyxl` call.

The agreed first approximation (Juha, 2026-07-30): **emit Markdown tables.** One table per detected table
region, regions delimited by at least one fully blank row or column, taken in Western reading order (left to
right, then top to bottom). Markdown is the right target — the models are steeped in it, and `docextract`'s
other formats already produce prose that the chat view renders as Markdown, so it needs no new convention.

One substitution worth making on the sketch: **separate sheets with a heading carrying the sheet name**
(`## Sheet: Q3 Budget`) rather than a bare `-----`. A horizontal rule says "something else starts here" and
throws away the name, which is often the single most informative string in the file — "Assumptions" versus
"Raw data" tells a reader, and a model, what it is looking at. Same cost, strictly more information.

Then the details that decide whether the output is useful or merely plausible:

- **Values, not formulas.** `openpyxl`'s `data_only=True` yields the *cached* result, which is present only if
  a real spreadsheet application last saved the file.

  The empty-cell case needs **two** things true together: the file contains formulas *and* was never saved by
  an application that computes them. Most inputs fail one of those, which is why the expected sources look
  safe — a report downloaded from a web dashboard is usually pure values with no formulas at all (so
  `data_only` is moot), and a human-authored workbook has been through Excel or LibreOffice (so the cache is
  populated). The gap is narrow: a formula-bearing file written by a library (`openpyxl`, `xlsxwriter`,
  `pandas`) and never opened.

  What makes it worth handling anyway is that the failure is **silent** — blank cells, not an exception — so
  it surfaces as a confidently empty table rather than an error. Cheap insurance for a narrow case, not a
  workaround for a common one: fall back to the formula text, and never emit a table that is entirely blank
  without saying why. Confirm the behaviour against a file written by `openpyxl` itself before relying on any
  of this.
- **Merged cells.** Markdown cannot express a merge. `openpyxl` reports the value in the top-left cell and
  `None` for the rest of the range; repeating the value across the merged span usually retrieves better than
  leaving blanks, since a row then still reads as a complete record.
- **The used range lies.** One stray cell far out to the right makes a sheet nominally enormous. Bound the
  emitted region by actual content, and cap total output — a 50k-row sheet rendered in full is a wall of text
  that crowds out the question being asked about it.
- **Charts, images and pivot caches: skip.** No text to extract, and a placeholder line invites the model to
  comment on something it cannot see.
- **`.ods` may be nearly free.** `odfpy` is already a dependency (it backs `_extract_odf` for `.odt`/`.odp`)
  and handles spreadsheets too, so the second format is likely a different reader over the same
  region-detection and Markdown-emission logic. Worth structuring the code that way from the start.

The legacy binary formats (`.doc`, `.ppt`) are deliberately out of scope: reading them means shelling out to
a separate converter.

Discovered during the document-attach test-drive (2026-07-18, Juha — "the software category that spends its time
disproving the claim on the tin").

## FileDialog: slow open and a teardown input-dead-window on huge directories

`FileDialog.show_file_dialog` → `chdir` → `reset_dir` rebuilds the *entire* file listing — one widget row per
entry — every time the dialog opens. On a directory with thousands of files (Juha's papers dir) this takes a
couple of seconds to show, and there is a second symptom: right after *closing* the dialog, clicking the opener
again does nothing (not even the opener button's own flash fires) for a similar couple of seconds, then works.
The modal window is still tearing down its thousands of child widgets, and while it does, input to the button
behind it is swallowed — the click never reaches the callback. Both symptoms share one root cause: the listing is
fully materialized as DPG widgets. Fixes to weigh: virtualize the listing (render only the visible rows), or
cache/reuse the built listing when the directory is unchanged across opens (the common reopen-same-dir case), so
a reopen is instant and there is no thousands-of-widgets teardown to block on.

Discovered during the document-attach test-drive (2026-07-18, Juha).

## FileDialog: smart-case the Find (search) field

The `FileDialog` search/filter field matches case-sensitively, which is the wrong default for a file finder. Make
it smart-case: case-insensitive when the query is all-lowercase, case-sensitive when it contains an uppercase
letter (the Emacs / ripgrep convention).

Discovered during the document-attach test-drive (2026-07-18, Juha).

## FileDialog: image thumbnail previews (Lanczos'd)

The vendored `FileDialog` (`raven/vendor/file_dialog/`) lists files by name only — no image previews. For picking
*image* files (the multimodal image-attach feature, brief 03 Half 2), a thumbnail per image would make selection
usable — you pick by looking, not by guessing from the filename. This matters more than for most file types:
photos and AI-generated images usually have non-descriptive filenames (hashes, timestamps, auto-names), so the
image *data* is the only reliable way to identify the right file — and doubly so because DPG apps have no OS
drag-and-drop (see the drag-and-drop item below), making the in-app picker the *only* way to bring an image in.
Add Lanczos-downsampled thumbnail previews to the file listing when the filter is image-typed. Reuse
`raven.common.image.lanczos` + the `add_dynamic_texture` path; the Nvidia/Linux texture-deletion-segfault
workaround (`__GLVND_DISALLOW_PATCHING=1`) is already set in the apps.

This is a UX enhancement for the image-attach picker, not a blocker: the attach feature ships first with the
basic (filename) FileDialog listing; this improves it (a filename-only picker is a poor fit for choosing images).
When built, mind DPG texture lifecycle for the many small preview textures (create/destroy as the user navigates
directories).

Discovered during brief-03 Half-2 multimodal work (2026-07-17, flagged by Juha).

## FileDialog: multi-extension filter as one labelled item

The vendored `FileDialog`'s type filter is single-extension: each `filter_list` entry matches exactly one
extension (`.png`), and the "show everything" option is the bare `.*`. There is no way to offer a single filter
item that matches a *set* of extensions under a descriptive label — e.g. "All images (`.png .jpg .jpeg .webp
.bmp .gif .tiff`)". Librarian's image-attach dialog works around this by defaulting to `.*` (so images of every
type show at once, at the cost of also listing non-images). Add multi-extension filter items with custom labels:
a `filter_list` entry should be able to carry a label plus a set of extensions, and the listing filter should
match any extension in the set. Then image pickers can offer one "All images" item instead of `.*`.

Once this exists, it also lets the Librarian **attach** dialog gate the *offered* types by model capability
(Juha, 2026-07-18): show "All files (images + documents)" with a vision model, "Documents only" with a text-only
model — so wrong types can't be picked at all. Today the attach dialog offers everything and does the image
gating at *routing* time (`app._attach_callback` rejects an image on a confirmed text-only model with a dialog);
picker-level filtering would replace that after-the-fact rejection with up-front unavailability.

Discovered during brief-03 Half-2 multimodal work (2026-07-17, flagged by Juha).

## FileDialog: reduce per-use-site boilerplate

Every `FileDialog` use site repeats a verbose constructor (title, tag, callback, modal, `filter_list`,
`file_filter`, `multi_selection`/`save_mode`/`dirs_only`, `allow_drag`, `default_path`, …) plus a `.show_file_dialog()`
call and a `selected_files` callback. Recurring across `raven-visualizer`, `raven-cherrypick`, and the avatar
pose/settings editors — and it was about to be repeated in `raven-librarian`. Wrap the common shapes (open-file,
open-files, save-file, pick-dir) into thin helpers so a use site is roughly one call. Long-standing "meaning to
fix this" item (Juha).

Discovered during brief-03 Half-2 multimodal work (2026-07-17, flagged by Juha).

## OS drag-and-drop of files into DPG apps (cross-platform)

DPG apps can't receive files dragged in from the OS file manager — you must go through the in-app `FileDialog`
every time. A recurring pain point across the fleet (Juha), and it compounds the image-picker problem above:
with no drag-and-drop, the picker is the sole entry path, so the picker has to be good. There's a Windows-only
extension for this, but nothing for Linux/macOS. Investigate whether cross-platform OS→app file drop is feasible
(SDL/GLFW-level drop events, a platform-specific shim per OS, or an out-of-process helper) and, if so, wire it as
a general capability the apps can opt into — image attach and `FileDialog` both benefit.

Discovered during brief-03 Half-2 multimodal work (2026-07-17, flagged by Juha as a constant pain point).

## Modernize the Librarian system prompt / character card

The default system prompt (`raven.librarian.config`) reads as dated for current instruction-tuned models —
"take a deep breath and think step by step", "believe in your abilities and strive for excellence", "you are
NOT automatically updated with new data", an explicit context-window-size line, etc. Much of this is Bronze-Age
prompt-craft: modern models don't need the pep-talk hand-holding, and some of it is mildly counterproductive
(asserting the model's identity/limitations *to* the model). Revisit the whole prompt + character-card default:
keep the genuinely load-bearing behavioral constraints (cite only provided sources, metric units, admit
uncertainty), drop the motivational filler, and reconsider how much identity the frontend should assert at a
modern model at all. Noticed during brief-03 Half-2 image-attach testing (2026-07-17, Juha).

**While rewriting it, add the supported attachment formats.** A user with a file in hand and a doubt about it
should be able to just ask, and get an answer instead of a guess — which matters for the digital-colleague
track, where asking the colleague is the natural move and hunting for a file-type list in the docs is not.
Four things make this less obvious than it looks:

- **Generate the list; never write it down.** `docextract.supported_extensions()` derives from
  `_PLAINTEXT_EXTS + _EXTRACTORS` specifically so the advertised formats cannot drift from the dispatch
  table. A hand-typed list in the prompt reintroduces exactly that drift, and in its worst form: a stale
  prompt does not fail, it makes the model *confidently* tell the user the wrong thing. This is not a
  hypothetical: the attach tooltip drifted this way within days of the office formats landing, by spelling
  the list out instead of asking for it (fixed 2026-07-30 — `_ATTACH_DOC_EXTS_TEXT`, which is the shape the
  prompt should copy).
- **The image list has no `docextract` equivalent, and lives in the wrong place.** `_ATTACH_IMAGE_EXTS` is a
  hardcoded private tuple in `raven/librarian/app.py` — no derived source, and unreachable from anywhere
  `minichat` could share. Assembling the prompt from a single source therefore needs it moved somewhere
  common first (`imagestore` is the natural home, beside the rest of the image-attach knowledge). Its two
  current read sites are both inside `app.py`, which argues for nothing; what forces the move is that the
  next reader is *outside* it.
- **It is model-dependent, so it cannot be one static string.** Documents work on any model; images need a
  VLM. The prompt has to be assembled against the same capability check `app._attach_callback` already
  performs, or a text-only backend will cheerfully promise image support.
- **The system block, not a per-turn inject.** Brief 08 §4 settled the general rule by measurement — what is
  stable for the session lives in the leading system message and stays in the cached prefix; only what
  actually changes per turn is injected. A format list changes only when the model does. (This does not
  contradict that brief's §5 "no capability check in `_perform_injects`": the check above happens once, at
  system-block build time, not per turn.)
- **Say how to attach, not just what.** "Yes, `.docx` works" is a dead end if the model cannot then point at
  the paperclip. The prompt should name the affordance alongside the formats.

Extension raised by Juha (2026-07-30).

## RAG: rerank retrieved chunks and inject only the best few

**Do `briefs/summer_2026_librarian_extension/09_retrieval-query-side-brief.md` first.** It documents a
verified finding that changes the diagnosis below: `_query_body` applies each engine's quality signal as
an absolute cutoff *before* fusion, and `reciprocal_rank_fusion` then sums `1 / (rank + K)` over
positions only. So the score-to-quality mapping is discarded one line before the rank that is supposed to
carry it, and a top-of-a-bad-batch result is indistinguishable from a top-of-a-good-batch one. That is
the reported symptom (less topical matches outscoring the ones that answer the question), and it gets
*worse* as retrieval widens — which is the direction this item wants to push. The brief also carries the
labelled-set setup that makes any of this measurable.

`docs_num_results = 20` (`raven.librarian.config`), and `scaffold._perform_injects` injects *all* of
them into the prompt, as one merged tool message placed before the user's latest message. That is a lot
of material to hand a model for one question, and it costs three ways at once:

- **Context.** Twenty chunks of scientific fulltext is a large fraction of the window before the
  conversation has even started, and the "Context-window budgeting and conversation compaction" item
  below has no enforcement yet.
- **KV cache.** They go in at the front, so every one of them is part of the prefix that gets rebuilt
  each turn (see the fold item's discussion of insert position).
- **Attention.** A model given twenty candidate passages, most of them irrelevant, has to do the
  relevance filtering itself — and long-context attentiveness is exactly what degrades as the prompt
  fills. Handing it three good passages is a different task from handing it twenty mediocre ones.

The standard shape for this is a **reranking stage**: retrieve broadly (the current hybrid BM25 +
vector + reciprocal-rank-fusion pass is a good recall stage), then score each candidate against the
query with a cross-encoder — which reads query and passage *together*, rather than comparing
independently-computed embeddings — and keep only the top few. Retrieval stays wide; what reaches the
prompt is narrow. See sentence-transformers' cross-encoder documentation
(https://www.sbert.net/examples/applications/cross-encoder/README.html) for the usual implementation,
and note the recall/precision division of labour is the whole point: the fusion pass is cheap and
approximate, the reranker is expensive per candidate but only runs on a shortlist.

Fits the three-layer pattern the other ML subsystems use: a `raven.common.rerank` implementation, a
`raven.server.modules.rerank` shim with its route, and a `raven.client.mayberemote.Reranker`. It is a
separate model from the embedder (cross-encoder, not bi-encoder), so it is a new load on whichever
device serves it — worth weighing against the VRAM budget on single-GPU setups, and a good candidate
for the CPU in `config_lowvram` since it runs on a shortlist rather than the whole corpus.

Open question worth settling with the same experiment: how many chunks should actually reach the
prompt. The answer interacts with the "answer from context only" reminder below — fewer, better
passages may make that reminder unnecessary rather than merely better-worded.

Discovered during Librarian↔LM Studio connectivity work (2026-07-19, Juha).

## Decide the public name: "Raven" is taken, and the project has outgrown "raven-visualizer"

Raven has no PyPI package, and can't easily get one under either candidate name.
`raven` on PyPI is Sentry's old client, and the name is common enough to be crowded
generally — there is now an AI product using it too. Meanwhile `raven-visualizer`
(the name in `pyproject.toml`) no longer describes the thing: Raven is a constellation
of apps now, and the visualizer is one of them.

So this is a naming *and* branding decision, not a packaging chore, and it needs an
in-house discussion before anything is chosen.

What any replacement has to preserve, since the current name is doing several jobs at once:

- **The local in-joke.** Jyväskylä once ran Korppi, a course-management system built
  in-house at the university before a commercial product replaced it. "Jyväskylä develops
  ravens" is the tradition — cheekily generalized from a single data point.
- **The literal aptness.** Ravens collect shiny things, which is precisely what the
  visualizer does. This is where the name came from, back in the visualizer-only days
  (~2024).
- **The constellation pun.** Corvus *is* an actual constellation, which landed retroactively
  once Raven became a constellation of apps rather than one tool.

Decision inputs: discoverability (a crowded name costs visibility), namespace availability on
PyPI, and whatever branding constraints the in-house discussion surfaces. Note that keeping
"Raven" as the *project* name while publishing under a distinct PyPI name is also on the
table — the two don't have to match.

**Before closing this item, move the three-part etymology above somewhere permanent.** It is
filed here because it constrains the naming decision, but it long outlives it — and this file's
convention is that resolved items are *deleted*, git being the history. Whatever is decided, the
reason the thing was ever called Raven should not be deleted along with the question of what to
call it next.

## Audit fleet for dict constants that should be `frozendict`

Several modules across Raven hold module-level dict constants that are used as immutable defaults or lookup tables, relying on "don't mutate this" by convention. `unpythonic.frozendict` (already a Raven dep) enforces it with teeth and costs nothing extra; Python 3.15 will also ship a stdlib `frozendict`.

Worth a pass across the fleet to find these and convert them. Low-risk (any call site that was mutating a shared default was a bug anyway), non-urgent.

Discovered during avatar-client-crop brief review (2026-04-20).

## Split `raven.common.nlptools` per backend (reduce import weight)

`raven.common.nlptools` is a hub module: it imports `torch`, `transformers`, `sentence_transformers`, `flair`, `dehyphen`, and `spacy`. All five ML-engine loaders (spaCy, classifier, dehyphenator, embedder, translator) live in it, so importing the module drags the entire ML stack into any process that touches it.

`raven.client.api` currently imports `nlptools` purely to reach `deserialize_spacy_docs` for the natlang response reconstruction. If a *lighter* client module ever wants to reconstruct spaCy Docs from wire data without pulling transformers/flair/dehyphen, a clean way to do it would be: extract each backend into its own module (e.g. `raven.common.spacy_wire` for the spaCy serialize/deserialize pair, parallel modules for each of classifier, dehyphenator, embedder, translator), and leave `nlptools` as a thin aggregator that re-exports them.

Why this is deferred: `api.py` already imports torch, qoi, spaCy, etc. for its other endpoints — so `nlptools` riding along costs `api.py` nothing extra *today*. The split pays off only when some caller wants a minimal "just reconstruct a Doc from JSON" importable, which no one currently needs. Also ties into the companion item "Lazy `api.initialize` in `llmclient`" — that one is about letting `llmclient` be imported in minimal-deps CI without triggering the full chain; slimming `nlptools` becomes relevant only once that parent effort is on the table.

For symmetry, if we ever start splitting, we should do all five backends, not just spaCy.

Discovered during natlang wire-format migration (2026-04-21). Original framing lived in the now-resolved "Language-neutral wire format for the natlang (spaCy) endpoint" item, superseded by this follow-up.

## Enable HTTP response compression on raven-server

The natlang wire-format migration (JSON via `Doc.to_json()` instead of DocBin) lost DocBin's vocab-sharing optimization — categorical strings (POS tags, dep labels, lemmas) now appear once per token rather than once per batch. gzip/deflate recovers most of the loss because those are exactly the patterns dictionary-based compression eats for breakfast; natively we're probably 1.5×–2.5× bigger uncompressed, within 10–20% after gzip.

Raven-server uses Flask/waitress without response compression currently. Adding `Flask-Compress` or a waitress pre-filter is ~one line. Other endpoints would benefit too (imagefx JSON metadata, the server's HTML index page).

Not urgent — Raven's trusted-LAN-or-localhost deployment means bandwidth isn't the bottleneck for typical payloads (KB-range, not MB-range). Revisit if profiling shows wire time becoming a meaningful fraction of end-to-end latency, or when a JS client on a WAN-ish link enters the picture.

Discovered during natlang wire-format migration (2026-04-21).

## Uniform load-on-demand for Raven-server modules

Raven-server keeps every model resident. That is the right default when the server owns the GPU, and the
wrong one on a laptop dGPU where the VRAM budget is already spent on the LLM and the avatar — any new
model (a reranker is the live example) has to displace something or not run at all.

What this wants is not a one-off for whichever module needs it first, but *uniform* load-on-demand across
the server modules: a module declares itself loadable-on-demand, and the server loads on first request
and evicts under pressure. Bolting it onto one module invites each subsequent one to invent its own
policy, which is how the eviction bugs get interesting.

Unverified prior worth measuring before designing around it: on a laptop the PCIe link is narrow (x8 on
a dGPU, x4 over a Thunderbolt eGPU), so load time may make on-demand a non-starter for anything but the
smallest models. Nobody has timed it — and if it *is* slow, running the small model on CPU may beat
loading it to GPU per turn.

Raised while scoping RAG reranking (2026-07-28, Juha).

## Remaining server modules without a MaybeRemote

With `Classifier`, `Translator`, `Postprocessor`, `Upscaler` landed (2026-04-22), the following server modules still don't participate in the MaybeRemote pattern:

- `avatar`, `avatarutil` — licensing-constrained (see "Client-local avatar animator" below). Also the rendering pipeline is tied to real-time animation driver state that's server-local; a client-local path would be effectively a parallel rewrite, not a wrapper.
- `websearch` — AGPL-constrained (~90 % from SillyTavern-extras, rest ported from SillyTavern-selenium's JS version — see the licensing item below). Also heavy to run locally: Selenium + headless browser.

These are both intentional omissions, not TODO gaps. Kept as a navigational note so future readers can see the coverage at a glance.

## Client-local avatar animator (licensing-bounded)

The avatar animator currently lives only in `raven.server.modules.avatar` under AGPL. THA3 upstream (the underlying ML model, vendored in `raven/vendor/tha3/`) is actually MIT — so the AGPL tax comes from Raven-side extensions, not the model itself.

A client-local animator would be valuable even though the server one stays:

- It extends the "server-optional" story (the goal behind the existing MaybeRemote pattern) to the avatar: a Raven app running standalone could still show the avatar, without requiring the server to be running.
- It enables a **fully-BSD Raven distribution** — simpler to configure for single-app users, and avoids the "license: it's complicated" friction that tends to drive people away from otherwise-perfectly-serviceable software.
- It skips the QOI encode/decode + loopback-socket round-trip. This *may* be a meaningful latency contributor even on localhost setups — needs measuring before being used as justification. On a non-localhost server setup, the user has put the server elsewhere for a reason (shared GPU across machines, a specific box with the VRAM, etc.), so "skip the network" isn't really the escape for those cases — a client-local animator helps only standalone / localhost use.

**Per-module authorship provenance on the server side** (for scoping what can / can't be unilaterally relicensed):

- `raven.server.app` — the Flask application proper. Has external contributors from the SillyTavern-extras era (and possibly earlier). Shared authorship.
- `raven.server.modules.websearch` — ~90 % from SillyTavern-extras, then patched by porting later modifications from SillyTavern-selenium (JS) into Python. AGPL lineage; not the user's code to relicense.
- `raven.server.modules.avatar` — mostly user-authored *now*, but with a tangled lineage:
  - The very original avatar module (in SillyTavern-extras, pre-user) was based on example code from THA3 (MIT).
  - Just before the user's first commit on it, it had a `result_feed`, a rudimentary `Animator` class (including its Discordian class docstring, which was a keeper), and little else.
  - Everything that makes the current avatar actually work at 10+ FPS — the optimised animator, sway animation, breathing, blinking, the postprocessor, the addon cel machinery, animefx, morph overriding — is user-authored. Some of this landed already during the SillyTavern-extras era (inside the AGPL project, but authored by the user and therefore his to relicense), the rest was added during Raven's development.
  - Clean-room scoping therefore isn't just "Raven-era = clean": the user's authored contributions pre-date the ST-extras → Raven conversion. The actual scoping would be per-line `git blame` on the final post-deprecation ST-extras snapshot to identify user-authored lines there, combined with all of Raven-era avatar work. Non-user-authored lines from the ST-extras era are the only ones that must not be reused.
- `raven.server.modules.classify` — thin shim over `raven.common.nlptools`. User-authored.
- `raven.server.modules.embeddings` — thin shim over `raven.common.nlptools`. User-authored.
- `raven.server.modules.imagefx` — new in Raven, 100 % user-authored.
- `raven.server.modules.natlang` — new in Raven, 100 % user-authored.
- `raven.server.modules.sanitize` — new in Raven, 100 % user-authored.
- `raven.server.modules.translate` — new in Raven, 100 % user-authored.
- `raven.server.modules.tts`, `raven.server.modules.stt` — new in Raven; the ST-extras versions were discarded when the final ST-extras was converted into the first Raven-Server. 100 % user-authored.

So the AGPL tax on the server side is concentrated in three places: the Flask app scaffolding (`server.app`), `websearch`, and the pre-rewrite foundation of `avatar`. Everything else is user-authored and could in principle be relicensed BSD — but the server as a whole still ships AGPL because of those three.

**Licensing distinction the open question really hinges on:**

- **Web-API RPC between processes** (the current MaybeRemote pattern): two separate products, networked, already clearly fine. No combined-work question.
- **In-process linking** (importing the AGPL module from a BSD caller): analogous to linking object files. *This* is the real AGPL question — does the combined work inherit AGPL, and can AGPL make the BSD caller's effective requirements *stricter* than BSD's?
  - GPL-family consensus: yes, a combined work inherits the strictest applicable license, which is AGPL's intent (prevent embrace-and-extinguish).
  - BSD's permissive posture doesn't change that — BSD code can be used *in* a stricter work, it just can't be *stripped of* attribution.
  - Practical consequence: if a BSD client module imports AGPL server code into the same process, the resulting program is effectively AGPL for distribution purposes.
  - Takeaway: the server stays server-only (RPC boundary, clean); an in-process client-local animator has to be a separate implementation that doesn't import the AGPL module.

**What a client-local animator would require:**

- A clean-room implementation built on MIT-licensed THA3 plus the user's own BSD-licensable contributions (the bulk of the current animator — see authorship breakdown above).
- Per-line `git blame` on the final post-deprecation SillyTavern-extras avatar module to classify each line by author. User-authored lines (from either the ST-extras era or Raven) are reusable; non-user lines from the ST-extras era are the only ones that must not be copied. The Raven-era delta is entirely user-authored and reusable wholesale.
- No code copy-paste from the non-user-authored ST-era portions — even small fragments with shared authorship would re-infect.
- The server-side animator keeps its tangled lineage and stays AGPL; the BSD client-local animator lives in `raven.common.avatar` (new) or similar, with a cleaner, from-scratch scaffolding around THA3.

The server animator is not going away — it remains indefinitely useful, especially once a JavaScript avatar client exists. A BSD client-local animator is purely additive.

`raven-avatar-pose-editor` should gain a mayberemote mode as well: it currently loads THA3 in-process, which collides with other local GPU consumers (observed 2026-04-24 — CUDA OOM on a 3070 Ti with Qwen + one THA3 instance already resident). Remote mode would let the pose editor run against a separate server process, or share a single THA3 instance with the live animator on the same box.

**This does *not* depend on the client-local animator, and is not licensing-gated** (corrected 2026-07-27 — the earlier wording sequenced it after the clean-room work, which was wrong). Both the server and the pose editor are already AGPL, so a remote mode is just an AGPL client calling an AGPL server over HTTP; no clean-room question arises. The actual blocker is technical: the avatar web API was designed for live animation and does not expose the primitives the pose editor needs. Scoping that gap is the first step.

No action until the user decides whether to pursue the clean-room path. Discovered during speech-extract-to-common discussion (2026-04-17).

## Untested but test-worthy modules in `raven.common`

Cross-referencing `raven/common/**/*.py` against existing `tests/` dirs, the following have non-trivial algorithmic content but no tests:

- `raven/common/bgtask.py` — background task queue / lifecycle primitives. Pure orchestration; testable with fake tasks.
- `raven/common/gui/layout_math.py` — coordinate / packing math for DPG layouts. Pure functions despite living under `gui/`; testable the same way `viewport_math` and the xdotwidget math modules are.
- `raven/common/hfutil.py` — HuggingFace model installer. Side-effectful but the path-computation / repo-name-parsing parts are testable with tmpdir + monkeypatched `snapshot_download`.
- `raven/common/deviceinfo.py` — GPU detection, dual-GPU ordering. Small surface area; the logic that matters (device counting, visibility filtering, user-facing string formatting) can be tested with a monkeypatched `torch.cuda`.

The following are untested, and were long assumed untestable (the "test the algorithm layer, not GUI code"
principle):

- `raven/common/audio/{player,recorder}.py` — audio hardware I/O. Genuinely hardware-bound; leave.
- `raven/common/gui/{fontsetup,helpcard,messagebox,utils,vumeter,widgetfinder}.py` — DPG glue.
- `raven/common/gui/xdotwidget/{widget,renderer,highlight,constants}.py` — rendering / DPG-bound.

**That blanket exclusion is now known to be too broad** (2026-07-30). DPG runs headlessly enough to test:
`create_context()` + `create_viewport()` + `setup_dearpygui()`, *without* `show_viewport()`, gives real
widgets, real themes and a steppable animator with no window mapped and no focus taken — so a test can drive
actual DPG state rather than mocks. `raven/common/gui/tests/test_animation.py` is the first case
(`WidgetFlash`'s restore contract and its ghost/reified de-duplication), guarded by
`importorskip("dearpygui")` so it runs locally and skips in CI, which installs no GUI toolkit. Technique
recorded in `dpg-notes.md`.

So the list above is a list of *candidates*, not exclusions. What to select for is behavior a human cannot
reliably eyeball: state machines, restore-what-you-borrowed contracts, teardown ordering, anything with a
lock. What still does not pay is appearance — whether a layout looks right is a screenshot's job, not an
assertion's.

Worth a pass in that spirit over `messagebox` (modal state and the `split_frame` dance), `widgetfinder`, and
`vumeter`'s level math.

Priority if picking one up: `bgtask` (most likely to harbour concurrency bugs; test-time cost is low), then `layout_math` (easy win), then `hfutil`/`deviceinfo` (requires monkeypatching but small).

Discovered during speech-extract-to-common discussion (2026-04-17).

## torch / torchaudio CUDA version alignment on fresh installs

`torchaudio>=2.4.0` was added as a direct dep alongside the existing `torch>=2.4.0`. Bare `pip install torchaudio` on a machine with `torch==2.10.0+cu128` fetched `torchaudio==2.11.0` from PyPI, which is built against CUDA 13 and fails to load (`libcudart.so.13: cannot open shared object file`). Workaround used on the dev box: `pip install "torchaudio==2.10.0" --index-url https://download.pytorch.org/whl/cu128`.

This is a broader torch-ecosystem packaging issue (torch/torchvision/torchaudio minor versions must match, and PyPI's default wheels track the latest CUDA while most installed torch is older). Not fixable from within raven's `pyproject.toml` without pinning a specific torch build — which would create its own problems across Linux/Mac/Windows and CPU-only/CUDA users.

Follow-up options to consider:

- Document the issue in `README.md` / install instructions: if `pip install` from PyPI pulls a torchaudio that fails to load, install it from `https://download.pytorch.org/whl/<your-cuda-or-cpu>` matching the installed torch minor.
- Check whether PDM respects PyTorch's index-url convention if we add it to `[[tool.pdm.source]]` — might auto-resolve correctly on fresh installs.
- Revisit once torchvision is pinned somewhere (same class of problem).

No code change; this is a documentation / install-experience issue. Discovered during speech-extract-to-common step 2 (2026-04-17).

## Lazy `api.initialize` in `llmclient` and `hybridir` (would unblock `test_scaffold` in minimal CI)

`raven/librarian/llmclient.py` calls `api.initialize(...)` at module top (lines 55–58). This means `from raven.librarian import llmclient` both (a) requires the full `raven.client.api` import chain to succeed (qoi, spaCy, Kokoro TTS, …), and (b) runs the initialization side effect. As a result, `scaffold` — which imports `llmclient` at module level — is not importable in environments without the full dep stack.

The same anti-pattern also lives in `raven/librarian/hybridir.py` (same line-range).

Concrete cost observed 2026-04-17: `test_scaffold.py` has to `pytest.importorskip("raven.librarian.scaffold")` at the top, so the scaffold tests skip entirely in the CI minimal-deps job (matching the existing pattern for `test_api.py` and `test_hybridir.py`). Scaffold coverage is visible only in dev environments — not a regression, just a cap on what CI can report.

Refactor sketch:

- Move `api.initialize(...)` out of the module body into a lazy setup function. The natural home in `llmclient` is probably `llmclient.setup`, which app startup already calls; `hybridir` has an analogous setup path.
- Audit `llmclient`'s / `hybridir`'s module-top imports for other side effects; move to lazy/TYPE_CHECKING where possible. `scaffold.py` now uses `TYPE_CHECKING` for its `hybridir` import, which is a good model.
- Verify no other module relies on `api.initialize` being called as a side effect of importing `llmclient` / `hybridir`.

Once done, remove the `pytest.importorskip` from `test_scaffold.py`; the scaffold tests then contribute to CI coverage too (~90% of scaffold.py's 119 statements).

Fleet status as of 2026-04-24: `raven/visualizer/importer.py` used to have the same pattern but was cleaned up — `api.initialize(...)` now lives in that module's `main()` (for the `raven-importer` CLI) and in `raven/visualizer/app.py` (for the GUI). Same shape as `librarian/app.py` already uses. Use as a reference when tackling `llmclient` / `hybridir`.

Discovered during scaffold/appstate test work (2026-04-17).


## torch.compile for the postprocessor

`torch.compile()` on THA3 was investigated (2026-04-09) and yields only ~6% speedup (20.3ms → 19.0ms on 3070 Ti) at the cost of 37s compilation startup. Not worth it for THA3 — the model is already lean with separable convolutions + FP16. Also hangs in the server (works in standalone; cause unresolved — possibly Triton subprocess interaction with waitress/threads).

The postprocessor (`raven.common.video.postprocessor`) might benefit more from compilation (20–60 kernel launches per frame, more fusible elementwise ops). Worth investigating separately. See `briefs/tha3-performance-audit.md`.

Discovered during THA3 performance optimization work (2026-04-09).

## MPS (Apple Silicon) device synchronization

`torch.cuda.synchronize()` calls throughout the codebase (preload cache, imageview mip loading) only handle CUDA/ROCm. Apple MPS (`torch.device("mps")`) needs `torch.mps.synchronize()` instead. Audit all `torch.cuda.synchronize` call sites and add MPS equivalents. Consider a `deviceinfo.synchronize(device)` helper.

Discovered during raven-cherrypick compare mode review (2026-03-30).

## Audit unnamed lambdas

Unnamed lambdas produce unhelpful `<lambda>` in stack traces. Audit all Raven apps for unnamed lambdas and name them using either `unpythonic.namelambda` or by hoisting to a `def`. Start with raven-cherrypick and raven-xdot-viewer.

Discovered during raven-cherrypick compare mode review (2026-03-30).

## AMD GPU (ROCm) support audit

ROCm presents as `"cuda"` in PyTorch, so our Lanczos kernel and `deviceinfo` validation should already work on AMD GPUs. However, the rest of the codebase needs auditing:

- All custom Torch code (postprocessor filters, Anime4K upscaler, avatar renderer) — likely fine, but verify.
- Third-party ML libraries: `transformers`, `sentence-transformers`, Flair, Whisper, Kokoro TTS — check ROCm compatibility status for each.
- THA3 (vendored) — uses standard `nn.Module`, probably fine.

Discovered while implementing `raven/common/lanczos.py`.

## pillow-simd for faster PIL image processing

`pillow-simd` is a drop-in Pillow replacement with SIMD-optimized processing (resize, convert, transpose, etc.). Doesn't accelerate format decoders (libjpeg, libpng), but Raven has real PIL `.resize()` calls that would benefit:

- `raven/client/avatar_renderer.py` — 4 resize calls (backdrop, frame resizing)
- `raven/vendor/tha3/util.py` — Lanczos resize for character images

Limitations: x86-only (no ARM/Mac M-series), may lag behind Pillow releases. Needs `pip uninstall pillow && pip install pillow-simd`.

Discovered during raven-cherrypick loader pipeline design.


## Consolidate remaining numpy/tensor/DPG image conversions

`raven/common/image/utils.py` provides canonical `np_to_tensor`, `tensor_to_np`, `tensor_to_dpg_flat`. The `imagefx.py` conversions have been migrated. Remaining sites have intentional differences that make direct replacement impractical:

- `raven/server/modules/avatarutil.py` — involves sRGB ↔ linear colorspace conversion (domain-specific preprocessing, not just axis reordering)
- `raven/avatar/pose_editor/app.py` — pure numpy `.ravel()` for DPG, too simple to benefit from abstraction
- `raven/client/avatar_renderer.py` — pure numpy `/ 255` + `.ravel()` for DPG (3 sites), same
- `raven/vendor/tha3/util.py` — vendored, with custom scale/offset normalization (THA3-specific)

The remaining gain would be single-source-of-truth, not code reduction. Revisit if the avatar pipeline is ever refactored.

Discovered during raven-cherrypick imageutil extraction.

## Adopt dotted import style in remaining modules

Raven style is `from ..common.gui import utils as guiutils` + `guiutils.func()`, not
`from ..common.gui.utils import func` + bare `func()`. The dotted style makes it clear
at the call site where a function comes from. Modules with ambiguous names get an alias
(e.g. `guiutils`, `server_config`, `client_config`).

Cherrypick and xdot_viewer migrated (session 6). The xdotwidget internal
sibling imports (Node, Edge, etc.) are fine as-is — tightly coupled types.

Remaining: audit other Raven apps (Librarian, Visualizer, Server) if desired.

Discovered during raven-cherrypick imageview review.

## Triage CLAUDE.md style conventions: global vs project-specific

Many code style conventions currently in Raven's `CLAUDE.md` apply to all of Juha's projects (import style, naming, docstrings, log format, license DRY, sentence spacing). These should be moved to `~/.claude/CLAUDE.md` so they're picked up everywhere. Review each entry and split accordingly.

Discovered during raven-cherrypick development.

## Robust public API auditing tool

A tool that checks all public symbols are listed in `__all__` (PEP 8 compliance). The simple AST approach works for straightforward modules but misses re-exports, macro-generated symbols, and barrel `__init__.py` patterns. See mcpyrate's troubleshooting docs for the full complexity: https://github.com/Technologicat/mcpyrate/blob/master/doc/troubleshooting.md#how-to-list-the-whole-public-api-and-only-the-public-api

Could be a useful addition to pyan3 (static call graph generator already understands Python module structure).

Discovered during raven-cherrypick development.

## Faster PNG decoder

PIL's PNG decode via libpng is slow (~59 ms for a 1 MP image). Unlike JPEG (where turbojpeg provides scaled decode), libpng has no equivalent fast path. Options to investigate:
- `cv2.imread` — uses libpng but OpenCV's memory handling may be faster
- `fpng` / `fpnge` — fast PNG codecs, but Python bindings may not exist
- `spng` — simpler PNG library, sometimes faster than libpng
- For thumbnails specifically, could decode at reduced bit depth or skip interlacing

Discovered during raven-cherrypick test drive.

## Preload cache: 16MP image optimization

With 16MP images (4624×3472), each cached mipchain is ~342MB as flat arrays. The current 1500MB budget fits only ~4 images, causing most preloads to be dropped after doing the full GPU work (wasted ~530ms each, with GPU contention degrading frame times to ~90ms).

Three improvements needed:
1. **Cap preload mip resolution** — skip the full-res mip in preload (only needed at 1:1 zoom, rare during triage). At 0.5× max, per-image cost drops from 342MB to ~85MB → 17 images fit.
2. **Check budget before submitting** — currently the budget check is in `_on_task_done` (after all the work). Reject early in `schedule()` to avoid wasted GPU work and contention.
3. **Move decode to background thread** — `decode_image` (50-110ms) still runs on the main thread for cache misses. Add `set_image_path`/`set_image_bytes` to ImageView (sans-IO style), folding decode into the bg mip task.
4. **Profile ~300ms mipgen anomaly on non-sequential navigation** — `lanczos.mipchain` takes ~300ms wall-clock on click-after-scroll, but 0ms on End key. Both are cache misses, same image size, same allocator state, no thumbnail or preload contention (confirmed by cancelling all background work). The function contains only async CUDA kernel launches (F.pad, F.conv2d) — nothing that should block. Needs py-spy (GIL analysis) or nsight (CUDA timeline) to identify what's actually blocking. Might be cuDNN autotuning, CUDA memory allocator fragmentation, or something else entirely.
5. **Wait for preload CUDA completion on cancel** — `cancel_pending()` is cooperative (sets flag, doesn't wait). Cancelled preload tasks may still be mid-CUDA-operation (Lanczos mipchain, tensor transfers). The bg_mip_task's `cuda.synchronize` then blocks on both its own work AND the lingering preload ops. Observed: `mipgen=508ms` for 1024×1024 (should be ~1ms) after a far jump. Consider `cuda.synchronize` before starting the bg_mip_task, or use CUDA streams to isolate preload vs display work.

Discovered during raven-cherrypick preload performance session.

## raven-cherrypick: export image sequence (QOI→PNG batch conversion)

raven-cherrypick is effectively an image viewer with QOI support, which is rare. This makes it ideal for previewing avatar recordings frame-by-frame. Integrate `raven-qoi2png` CLI functionality so that raven-cherrypick can export avatar recordings for external consumption (e.g. as a PNG image sequence for OpenShot or other video editors).

Discovered during raven-cherrypick preload performance session.



## pygame pkg_resources deprecation warning

pygame 2.6.1 emits a deprecation warning: `pkg_resources is deprecated as an API` (from `pygame/pkgdata.py`). Functional but noisy. Check if a newer pygame version fixes this, or if pygame has moved to `importlib.resources`.

Discovered during smoke-testing on new machine (2026-03-25). Re-checked 2026-05-06: still pygame 2.6.1 on PyPI, and `pkgdata.py` on pygame's `main` branch is unchanged.

The fix isn't missing — it's queued. Last commit on pygame's `main` was 2025-10-05 (the v2.6.1 merge); nothing in ~7 months, 754 open issues. Three open PRs already replace `pkg_resources` with `importlib.resources` — #4792 (2026-03-12), #4583 and #4511 (both 2025-09-23) — plus several user-side warning reports (#4557, #4769, …). Repo is not archived, just review/merge-throughput limited. Nothing for us to do but wait for a release that picks one of those PRs up.

## raven-cherrypick: further reduce idle CPU/GPU load

Idle throttle (2026-04-05) reduced CPU load from ~80% to ~20% of one core by sleeping ~80ms between frames when nothing needs updating. The remaining ~20% is the floor cost of `render_dearpygui_frame()` at ~12fps — ImGui resubmits the entire UI each call. Further reduction options: adaptive sleep ramp (80ms → 500ms over ~5s idle, snap back on input), or skipping `render_dearpygui_frame()` entirely (risky — event processing is tied to the render call).

Originally discovered during raven-cherrypick session 5 (2026-03-19).

## Idle throttle for Librarian

Librarian has an avatar idle auto-off, and a no-avatar mode is under consideration. When the avatar is off and no LLM generation is in progress, the GUI is mostly static — same pattern as cherrypick/xdot-viewer. Busy sources: avatar rendering, LLM streaming, RAG indexing, pulsating color animations (audit which are always-on vs. conditional), recent user input. The existing cherrypick/xdot-viewer pattern (`_is_busy()` + sleep) should port directly.

Discovered during idle throttle discussion (2026-04-05).

## raven-cherrypick: low FPS with large images

With large images (e.g. 4247×891, 5203×1313), steady-state FPS drops to 10–15 (66ms/frame) compared to ~30 FPS for 1MP images. DPG metrics show the bottleneck is in presentation/rendering, not input routing. Likely causes:

- Large `draw_image` textures are expensive to blit every frame.
- Texture pool growth — `_release_texture` accumulates pooled dynamic textures; DPG scans all registered textures O(n) per frame even when not drawn.
- The double `split_frame()` workaround (needed because DPG doesn't guarantee texture upload before rendering within a single frame) adds ~16ms latency per mip during loading, but shouldn't affect steady-state FPS.

Related: the existing "investigate GPU/CPU load at idle" item. Both may benefit from frame-skip when nothing changes, and/or pool trimming to reduce registered texture count.

Discovered during raven-cherrypick deadlock/flash fix session (2026-03-28).

## CLAUDE.md: rephrase DPG pitfall #5 to avoid Claude thinking loops

DPG pitfall #5 (callback thread deadlock pattern) was temporarily removed from CLAUDE.md because it causes Claude Opus and Sonnet to hang when analyzing cherrypick concurrency code. The model reads the complex three-way deadlock description, then enters an unproductive reasoning loop — consistently stalls at ~250–300 output tokens across multiple retries and effort levels.

The information is correct and important (confirmed by C++ source analysis — see `dpg-threading-notes.md`). Needs rephrasing in a way that conveys the same constraints without the chain-of-reasoning structure that triggers the loop. The original text is recoverable from git history.

Discovered during raven-cherrypick debugging session (2026-03-28).

## Audit and slim down project CLAUDE.md

Raven's CLAUDE.md is growing long, which increases token cost per conversation and may contribute to reasoning issues (see pitfall #5 incident above). Audit for:

- Material that could move to **project-specific skills** (e.g. "how to set up a new Raven DPG app" — the DPG app structure, startup sequence, and key patterns sections are reference material, not per-conversation instructions).
- Material already covered by **sub-project CLAUDE.md files** (Visualizer and Librarian have their own — check for redundancy).
- Material that belongs in the **global `~/.claude/CLAUDE.md`** (see existing deferred item "Triage CLAUDE.md style conventions").
- Sections that are **too detailed for instructions** and would be better as standalone reference docs (like `dpg-threading-notes.md`).

Goal: CLAUDE.md should be concise instructions and constraints, not an encyclopedia. Reference material goes in separate files that can be read on demand.

Discovered during raven-cherrypick debugging session (2026-03-28). Reinforced 2026-04-06: instruction volume caused Claude to lint a .md file despite existing memory saying not to.

## Audit typing: abstract parameter types, concrete return types

Raven convention: parameters should use abstract types from `collections.abc` (`Mapping`, `Sequence`, `Iterable`) for widest-possible-accepted semantics. Return types should use concrete lowercase builtins (`tuple[int, int]`, `list[int]`, `dict[str, int]`) — PEP 585, Python 3.9+. The capitalized `typing` forms (`Dict`, `List`, `Tuple`) are deprecated aliases for the builtins and offer no extra width — avoid them. Audit existing type hints across the codebase for consistency.

Discovered during raven-cherrypick compare mode planning (2026-03-30).

## Audit toolbar buttons for ButtonFlash acknowledgment

Check existing toolbar buttons in raven-cherrypick and raven-xdot-viewer for whether their actions should flash green on activation (Raven's convention for acknowledging a click or hotkey press). Other Raven apps (Librarian, Visualizer) already use `ButtonFlash` consistently — cherrypick and xdot-viewer may be missing it.

Discovered during raven-cherrypick compare mode planning (2026-03-30).

## Extract `raven.common` into an upstream library ("corvid")

Raven's `common/` package has grown into a general-purpose DPG toolkit: GUI widgets (file dialog, markdown, helpcard, xdot widget, animation framework, VU meter), video/audio processing, networking utils, bgtask infrastructure. This creates a gravitational well — new apps land in Raven because the batteries are there, even when they have nothing to do with NLP/ML.

Extracting `raven.common` (and the vendored DPG extensions) into a standalone library would:
- Let pyan-gui and other non-Raven DPG apps use the toolkit without vendoring
- Move the general DPG notes (`dpg-notes.md`) upstream with the code they document
- Reduce Raven to domain apps (Visualizer, Librarian, Server, Avatar) + ML-specific code
- Clarify the dependency direction: corvid → DPG, Raven → corvid + ML

Short-term: vendor the xdot widget into pyan for pyan-gui. Long-term: extract properly.

Discovered during tooltip feature session (2026-04-03).

## Avatar settings editor: custom postprocessor chain ordering

**This is a GUI limitation only** — `briefs/crt-display.md` §0 establishes that the backend has always
supported multiple instances at arbitrary positions: `render_into` applies the chain *positionally*, and
`_priority` is consumed only by `get_filters` to sort the settings-editor panels. The `name` parameter on every
caching filter exists precisely so multiple instances key their caches apart. So this item is "build the
add/remove/reorder GUI", not "make the engine support it".

**What the GUI unlocks, which is more than convenience.** The eventual intent is two raster filters coexisting —
the old, simple `scanlines` and the new `crt` — each usable at *either* diegetic layer: Scene band for a sci-fi
hologram projected into the character's world, Display band for the viewer's own CRT monitor. The filters stop
being "the hologram one" and "the monitor one" and become two looks placeable at either position, or both at
once. That is exactly the freedom the fixed-order, one-instance-per-filter GUI currently withholds, which makes
this item the prerequisite for the whole idea rather than a nicety alongside it.

Note it is **not** a Researchers' Night blocker: the demo needs `crt` working at its default Scene-band
position, which the existing autodiscovery gives it. The placement freedom is the follow-up.

The settings editor currently presents filters in a fixed priority order, with at most one copy of each filter. With the desaturate/monochrome_display and noise/analog_vhs_noise splits, the signal pipeline model is becoming richer — users may want to reorder filters or have multiple instances. The GUI needs drag-and-drop chain building: add/remove filters, reorder freely, support multiple instances of the same filter (with independent `name` keys). Currently, `strip_postprocessor_chain_for_gui` enforces fixed ordering and single instances.

Discovered during postprocessor chain ordering redesign (2026-04-09).

## raven.papers user manual

The `raven.papers` tool collection has grown to the point where it deserves its own user manual, like Visualizer, Librarian and Server already have.

There are existing usage instructions for `raven-arxiv-search` in the README of the separate `arxiv-api-search` project, which the tool was created from. These should be included in the manual.

For the others, some instructions are scattered in Raven's main `README.md`.

Some instructions don't yet exist, and need to be written.

## Hybridir: cover the edit-queueing layer with tests

`raven/librarian/tests/test_hybridir.py`'s original 18 tests all target the post-commit query side — corpus is added once, committed, queried. The edit-queueing layer (`_pend_edit` dedup, update/delete paths, the add-then-update-same-doc race) was untested. Two latent bugs survived this gap: a `_pend_edit` shape-mismatch (triggered by dropping ~200 .bib files into the docs dir at once), and a spurious delete queued for a brand-new file whose watchdog create+modify events both landed before the first commit (triggered by ingesting a PDF — the first large files the docs DB handled).

A first batch of `_pend_edit` collapse tests landed 2026-07-18 (covering: update of an existing document → delete+add; delete of an existing document; add-then-update dedup for a new file → single add; the observed create/modify event-flurry ordering → single add). Remaining coverage to add:

- More dedup shapes: queue delete then add same id; queue add for two docs and update one; etc.
- Idempotency of `commit()` on empty queue.
- `is_indexing()` reference-counting under threaded concurrent `commit()` calls — mock the slow inner work, have two threads enter, verify `is_indexing` stays True throughout and goes False only when both have exited (also covers same-thread re-entry under the existing `datastore_lock` RLock).
- BM25 + semantic search is becoming the de-facto standard hybrid retrieval shape, so the layer is worth investing in regardless.

The watchdog-driven flow (tmpdir + `Path.touch` / `unlink` to drive `HybridIRFileSystemEventHandler`) crosses into bgtask scheduling and is harder to make deterministic — separate, lower-priority follow-up.

Discovered during DOCS-indexing-indicator smoke test (2026-04-27).


## Easy install with a chosen CUDA version (and a sensible CPU default)

Raven's `[cuda]` extra currently pulls a torch / torchaudio / torchvision combo pinned to one CUDA toolchain (currently `+cu128`). The PyTorch project ships these via `--index-url https://download.pytorch.org/whl/cuXXX`, and the matching `nvidia-cuda-runtime-cuYY` runtime is also installable as a Python package — so a Raven install could in principle bundle a complete CUDA stack from PyPI without touching the host's toolchain.

Today, switching between machines on different CUDA versions (e.g. one at CUDA 12.8, another at CUDA 13) requires hand-editing `pyproject.toml` and re-running `pdm install`. Worse, a plain `pdm install` quietly upgraded `torchaudio` from `2.10.0+cu128` to `2.11.0` (the latter wants CUDA 13 and silently broke imports on the CUDA-12.8 machine). The fix: pin torch + torchaudio + torchvision together as a CUDA-version-matched group, expose `pdm install -G cuda12` / `-G cuda13` extras, and document the per-machine choice.

In particular, `torchaudio` should be part of the CUDA dep set, pinned to the matching CUDA version — not a free-floating dep that PDM resolves to whatever's latest.

CPU-only path: someone who just wants Raven-visualizer on a laptop without a GPU shouldn't have to learn about extra-dep groups. The default install (`pdm install` with no extras) should pull the CPU build of torch/torchaudio/torchvision; the `-G cudaXX` extras only add CUDA-build alternates. Today GPU support being opt-in is fine (it's the heavy/optional capability), but the CPU torch build still has to *appear*, otherwise `import torch` fails outright. Probably means listing the CPU-build versions in the base `[project] dependencies` (with PyTorch's CPU index URL) and having the `-G cudaXX` extras override the base pins via a higher-priority constraint.

Discovered during the logsetup smoke test (2026-04-29) when a routine `pdm install` (run to refresh a console-script entry point) bumped torchaudio and broke the visualizer's import path. Recurred 2026-06-03 on the CUDA 12.8 machine: adding `trafilatura` triggered a re-resolve that again bumped `torchaudio 2.10.0+cu128 → 2.11.0` (CUDA-13 build, `OSError: libcudart.so.13`); restored with `python -m pip install "torchaudio==2.10.0" --index-url https://download.pytorch.org/whl/cu128`. Second occurrence — this is a recurring tax on every dependency change, not a one-off.

## Convert startup `print()`s to `logger.info()` where appropriate

`raven/server/app.py:11` has had a standing `# TODO: convert prints to use logger where appropriate` for a while; the smoke tests for the new `--log` flag made it concrete. Server startup currently does much of its progress via `print()` ("Server config loaded from '…'", "No API key, accepting all requests", "Initializing avatar on device 'cuda:0' …", etc.) — all log-worthy, none captured by `--log` today. Same pattern in a few other apps where startup status got `print()` instead of `logger.info()` historically.

Pass: grep `print(` in each app's startup region, decide per-call whether it's user-facing tool output (keep as `print`, e.g. `raven-check-cuda`'s ✅/❌ markers) or app status (promote to `logger.info`). The pre-PR `--log` smoke-test diff (`stderr - logfile`) is the easiest way to find candidates.

Vendored prints (e.g. THA3's "Loading the eyebrow decomposer … DONE!!!" at `raven/vendor/tha3/poser/modes/load_poser.py`) are judgment calls — leave alone unless we're already touching the file.

Discovered during the logsetup smoke test (2026-04-30).

## Hybridir: BM25 backend migration for larger corpora

`bm25s` rebuilds the entire keyword index on every commit (full corpus → full reindex; IDF changes mean it can't be incremental in this design). Sub-second on ~1k small documents, so a non-issue today. Will start to pinch around the 10k–100k mark.

The standard fix is the **segmented index** model: each batch of writes lands in a small immutable segment with deletes-as-tombstones, IDF is computed across segments at query time (or partial-pre-aggregated), and a background merge thread occasionally consolidates segments to keep the count bounded. Writes become O(batch); reindex cost is amortized through merges. This is what Lucene / Elasticsearch / Solr / Tantivy all do.

For Raven, the natural migration target is **Tantivy** via the `tantivy-py` Python bindings — a Rust port of the Lucene model, MIT-licensed, no JVM, decent Python ergonomics. Would replace `bm25s` end-to-end. The semantic side (ChromaDB) and the hybrid-fusion logic above stay the same.

Plan when this becomes necessary:

- Audit `_rebuild_keyword_search_index` and `_keyword_retriever` callsites to extract the BM25-specific surface from `HybridIR`.
- Introduce a thin keyword-index abstraction (add / delete / search) so the backend swap touches one module.
- Migrate index storage on first run; existing `bm25s` indices on disk get rebuilt into Tantivy form.

Approximate alternatives if we want to stay on `bm25s`: rebuild on a schedule (every Nth commit, or every M seconds of accumulated edits) rather than on every commit — relevance drifts a tiny bit between rebuilds in exchange for cheaper writes. Cheaper than a full backend swap; doesn't help asymptotically.

Discovered during cancellable-commit work (2026-04-27).

## webfetch "approve denied host" button relocates in brief 03

The brief-01 override affordance (approve a denied host for the session, then re-run the fetch on
a new branch — `scaffold.retry_tool_calls`) is wired to a button in `chat_controller.build_buttons`,
attached to the denied `role="tool"` node's button row. That attachment point is **provisional**:
brief 03 (content-parts) moves tool-result rendering into the assistant message body (tool calls
become gear-icon sub-elements, results become content-parts). When that lands, relocate the approve
button to wherever the denied fetch's result then renders, and drop the special `role == "tool"`
button-row branch. The backend (`retry_tool_calls`, `approve_host_for_session`, the
`webfetch_denied_host` marker) is rendering-independent and stays as-is.

Discovered while implementing the brief-01 GUI override (2026-06-04).

## webfetch: batch-approve several denied hosts at once

The "approve denied host & retry" override creates a NEW branch per approval (correct, given the
chat store is a forest). But approving several denied fetches one at a time leaves all-but-the-last
branch as noise — each intermediate branch is a dead end the user doesn't want. For a v1 follow-up:
let the user approve a *set* of denied hosts in one action (e.g. multi-select the denied tool nodes,
or an "approve all denials in this turn" button), then re-run all of them on a single new branch.
`retry_tool_calls` currently re-runs exactly one call; the batch version would re-run the union of
approved calls and copy/share the rest — a natural generalization of the same branch-and-rebuild logic.

Discovered during the brief-01 GUI override session (2026-06-04).

## "Internet" toggle: scope `tools_enabled` to a clear security boundary in the GUI

`scaffold.ai_turn`'s `tools_enabled` is a blunt all-or-nothing hammer (standing TODO on the param).
The intent behind the GUI toggle is to make the **network/security boundary blindingly obvious** to
the user. Direction: rename/scope the toggle to "Internet" and have it gate the network-reaching tools
(`websearch`, `webfetch`) specifically, rather than all tools indiscriminately. A separate toggle for
MCP tools is likely wanted later (different trust surface). This needs `ai_turn` to accept a per-tool
enable set (or an allowed-tool-name list) instead of a single bool, and the controller/app to map each
GUI toggle onto the relevant tool names. Pairs with the brief's tool-description-generation work.

Discovered during the brief-01 GUI override session (2026-06-04).

## scaffold: collect `ai_turn`'s callbacks into a single bundle object

`ai_turn` (and now `retry_tool_calls`) take ~12 individual `on_*` callback parameters, threaded
verbatim through the controller, the test helpers (`run_ai_turn` / `run_retry` rebuild the same dict),
and `retry_tool_calls` → `ai_turn`. The brief flags this as a later cleanup: replace the loose
parameters with one callback-bundle object (an `unpythonic.env` namespace or a small frozen dataclass
of optional callables), constructed once and passed as a unit. Shrinks every signature and call site,
and makes "the AI-turn callback set" a named thing. Do it as a focused refactor so the diff is
mechanical (one bundle type, update each producer/consumer), not entangled with feature work.

Discovered during the brief-01 GUI override session (2026-06-04).

## Headless agent-harness mode for `ai_turn` (scriptable agent layer)

`llmclient` already acts as an LLM *scripting* layer — a non-interactive way to drive the model
for one-shot tasks (used by `raven-pdf2bib` and friends). What's missing is the equivalent one
level up: a way to drive the full **agent** loop (`scaffold.ai_turn`) — LLM plus tool-calling,
branching chat tree, RAG — programmatically, with **no UI of any kind**.

Note the distinction from the existing frontends: Librarian (`app.py`) is the GUI client and
`minichat` is a TUI client, but *both* are interactive UIs. `scaffold` is already
*frontend*-agnostic (its ~15 callbacks are the seam — `minichat` proves the same backend drives
a terminal as well as the GUI), so the building blocks exist. The harness is not a third
frontend; it's the *no-frontend*, non-interactive, programmatic caller. What's wanted is a small,
ergonomic layer: feed it a backend (real or scripted), a datastore, and an initial message; let
it run `user_turn` + `ai_turn` to completion with tools enabled; return the resulting nodes /
tool transcript. Think "`llmclient` for agents".

Value:
- **Testing**: exercise agentic flows end-to-end without a live, nondeterministic LLM — e.g. the
  websearch -> webfetch chain, canonical-phrase copying, multi-step tool loops. Today the
  structural tests mock `invoke` / `perform_tool_calls`; a harness with a *scripted* backend
  could drive the real `ai_turn` against canned model turns deterministically.
- **Automation**: headless agent runs for batch/offline tasks, cron-style jobs, evaluation
  harnesses — the same way `raven-pdf2bib` scripts the plain LLM today.

Natural to build somewhere in the summer 2026 librarian six-part sprint; it would make every
later brief's agent behavior far easier to verify. Likely lands near `scaffold` / `minichat`
(a programmatic sibling of the CLI client) or as a thin `raven.librarian.agentharness`.

Discovered during webfetch implementation (2026-06-03), when validating the agent loop required
a live Qwen backend that wasn't available.

## Markdown ATX headings (`### ...`) don't render in the chat view

LLM replies that use ATX headings (`# `, `## `, `### `) show the literal `#` markers in
Librarian's chat history instead of rendering as headings. Confirmed it is NOT a data/websearch
problem: the stored content is valid markdown, and `mistletoe.markdown()` produces a correct
`<h3>...</h3>` even with the `<font color='...'>` wrapper that `chat_controller._render_text`
(chat_controller.py ~455) adds. So the heading is lost downstream, in the adopted
`raven/vendor/DearPyGui_Markdown` renderer's HTML→widget stage (`_HTMLToParser.handle_starttag`
in `parser.py` plus the entity rendering) — it doesn't appear to map `<h1>`–`<h6>` to heading
entities. Raven's own help text sidesteps this by using `**bold**` as pseudo-headings instead of
`###`, consistent with headings never having rendered.

Fix when convenient: add `<h1>`–`<h6>` handling to the renderer's HTML parser, wiring them to the
existing `H1`–`H6` font attributes (already defined in `font_attributes.py`; the markdown path may
just not be connecting them). Separately, emoji in replies render as tofu/`?` because the body font
has no emoji glyphs — cosmetic, lower priority (would need an emoji fallback font).

Discovered while smoke-testing the webfetch send-to-AI affordance (2026-06-03).

## Fenced code block (```` ``` ````) support in the Markdown renderer

`dpg_markdown` (vendored) doesn't render triple-backtick fenced code blocks: the fences show up literally and
the content between them renders as ordinary prose (no monospace, no background box). Add fenced-code-block
parsing plus a styled render — monospace font, background fill, and horizontal scroll (or wrap) for long lines.
Would let LLM replies containing code display properly, and let Raven's own system/error messages show verbatim
technical text cleanly. (Librarian's backend-error message wanted a code box for the raw error string; it falls
back to plain text for now.)

Discovered during brief-03 Half-2 error-message work (2026-07-17, flagged by Juha).

## Reasoning traces with indented bullets mis-render (Markdown indented-code-block collision)

A model's reasoning trace that indents its bullets — Gemma 4 emits `    *   Role: ...` with **four leading
spaces** — collides with standard Markdown semantics: 4+ leading spaces is an **indented code block**, so the
whole bullet list renders as verbatim/`Pre`, drawing a grey background box around it. Confirmed it is the
indentation, not stray markup: the stored `reasoning_content` has **zero backticks and zero font tags**
(grepped `data.json`); the offending lines are literally `'\n    *   Role: Aria...'`.

Two visual manifestations of the same input:
- **On reload / completed message** (whole reasoning rendered as one Markdown block): the 4-space indent fires
  the code-block rule → a grey `Pre` box over the bullets, whose border doesn't match its fill — that mismatch
  is the *existing* stranded-`Pre`-box reflow bug (see the inline-code-box item below), here triggered by the
  indented-code block instead of an inline-code span.
- **While streaming** (reasoning built incrementally): the code-block rule doesn't fire consistently, so the
  `*` markers get mis-parsed as emphasis delimiters across lines — random words tinted pink/teal — and a raw
  `</font>` leaks at the end (the thought-blue color wrapping broken by the list parse).

Not a content-parts (brief 03) regression: the reasoning-bubble rendering is unchanged
(`add_paragraph(reasoning_content, is_thought=True)`); the model's indented output simply meets standard
Markdown. A reroll whose reasoning used `1.`/`2.` numbers at column 0 rendered cleanly. Likely fix: **dedent /
normalize the reasoning trace's leading indentation before Markdown rendering** (strip the common/per-line
leading whitespace so indented bullets become real bullets, not code), and/or fix the vendored renderer's
color-on-list and `Pre`-box-position handling. Separately note: paragraph font color is also not applied to
list-item *markers* (bullets/numbers keep the default color) even when the list renders correctly.

Discovered during brief 03 §4 live validation, reported by Juha (2026-06-05).

## DearPyGui_Markdown inline-code background boxes are stranded on dynamic reflow

Inline-code spans (`` `like this` ``) render a grey rounded background box behind the text. The box position is
correct on first render, but when the layout above the span *reflows* — e.g. expanding/collapsing a message's
thinking trace, which pushes the whole answer down — the text moves but the background box stays put, leaving
empty grey rectangles stranded mid-paragraph (observed live: three stranded boxes after expanding a thought
bubble whose answer contained `` `code` `` spans).

Root cause in the vendored renderer: `DearPyGui_Markdown/text_attributes.py` `Code.render` captures the text
group's **absolute** screen position once via `dpg.get_item_pos(dpg_text_group)` and creates the background as
an absolutely-positioned `dpg.add_group(pos=pos)` + drawlist quad. Absolute position doesn't track normal-flow
reflow, so any layout change above the span leaves the box behind. (The thought-bubble show/hide toggle is the
easy repro now that §9/§10 made thinking traces prominent, but window-resize reflow and message edits would
trigger it too.)

Same class as the list-item bullet-point position bug fixed earlier — a decoration positioned from a captured
absolute pos rather than following the live layout. Fix directions: draw the background relative to the text in
normal flow instead of an absolute-pos group, or recompute/redraw the Code (and Pre) backgrounds when the
containing layout changes (the renderer's `post_render` / `CallInNextFrame` machinery already exists for `Pre`).

Discovered during brief 02 §9/§10 live validation (2026-06-04).

## Emoji support in the Markdown renderer (color emoji as inline images)

`dpg_markdown` (vendored) can't show color emoji: DPG/ImGui rasterizes a font's glyphs into a single monochrome
atlas, so emoji code points render as blank boxes (and emoji in LLM replies come out as tofu/`?`, since the body
font has no emoji glyphs at all). This is why Librarian's system/error chat messages avoid symbols like `⚠`/`✓`
(see the error-message construction in `scaffold.py`'s `ai_turn`).

Precisely: full-color emoji would need Dear ImGui's FreeType backend with `LoadColor` (COLR/CPAL or CBDT/CBLC),
which DPG does not expose. That leaves two realistic routes, and they sit at different layers:

- **(a) A monochrome outline emoji font** with a permissive license (e.g. an OpenMoji-Black or Twemoji-mono
  build), added to the atlas. Cheap, but flat glyphs — decide whether that's acceptable before building it.
  This is an atlas-level fix, so it shares machinery with "Super/subscript font coverage in the GUI" above.
- **(b) Inline images**, i.e. the sketch below — richer, renderer-level, and the only route that gets actual
  color.

The renderer is the natural home for a fix: it already splits a text run into extents to apply styling, so it
could detect emoji code points and substitute an inline image per emoji instead of a text glyph, sized to the
surrounding text. Implementation sketch: bundle a permissively-licensed emoji set — candidates: Twemoji, Noto
Emoji, OpenMoji (confirm the exact license when picking) — and rasterize **lazily**: a cache (defaultdict-style)
keyed by emoji code point, populated on first encounter, so only emoji that actually appear get turned into
textures. This sidesteps rasterizing the whole set up front (cf. the font-atlas size limits in `dpg-notes.md`).
Emit the cached texture inline where the emoji appears. Enables emoji in chat messages, tooltips, and anywhere
`dpg_markdown` renders.

Discovered during brief-03 Half-2 error-message work (2026-07-17, flagged by Juha).

## Super/subscript font coverage in the GUI

Math superscripts and chemistry subscripts, for the letters and numbers Unicode provides
(U+2070–U+209F etc.), need a font that actually carries those glyphs. Raven currently has no single
font covering both well; the gap shows up first in Visualizer.

This is a **font-coverage** problem, not a renderer one: `raven.common.gui.fontsetup` serves both
plain DPG text and the vendored markdown renderer (`markdown_add_font_callback` supplies
`dpg_markdown`'s fonts), so the glyphs either exist in the atlas for everything or for nothing.
Visualizer wants them in labels and tooltips, which never go through `dpg_markdown`.

**The Unicode range is not the gap.** `setup_font_ranges` already requests `0x100`–`0x2fff`, which
covers the subscript/superscript blocks outright — and on recent DPG the ranges are set up
automatically, so that call is a no-op anyway (see the separate item on it). What's missing is a font
that actually *carries* the glyphs. So the work is a survey of permissively-licensed fonts for
coverage of U+2070–U+209F and friends, and picking one — not range configuration, and not the
renderer.

Raised during webfetch GUI smoke-testing (2026-06-03); flagged for a dedicated discussion. Split out
from a combined emoji + super/subscript item on 2026-07-27 — the emoji half is a separate problem with
its own fix, and lives in "Emoji support in the Markdown renderer" below.

## Chat view scroll position jumps back down while the model is writing

While a reply streams, the chat view's scroll position keeps being pulled back to the bottom, so
scrolling up to re-read an earlier message during generation does not stay put — the next streamed
chunk yanks the view down again. The user has to wait for the turn to finish before they can read
anything else, which on a thinking model is a long time.

Demo-facing (Researchers' Night, 2026-09-26), and arguably the most *felt* of the chat-view defects:
it fires on every single turn, unlike the Markdown cases which need particular content. On stage it
also removes the natural thing to do while the model thinks, which is to scroll back and talk about
what it said last.

Presumably the follow-the-tail autoscroll is unconditional, where it should engage only when the
view is already at (or near) the bottom — the standard "stick to bottom unless the user has scrolled
away" rule. Not investigated, so that is a hypothesis about the cause, not a diagnosis.

Discussed in an earlier session and believed to be recorded here; it was not. Written down
2026-07-28 after failing to find it (reported by Juha).

## Chat view drops a character mid-message ("What" renders as " hat")

Observed 2026-07-18 in Librarian's chat view: an assistant greeting displayed as
"Hello! I'm here and ready to help.    hat can I do for you?" — the `W` missing, with a visible
run of extra whitespace where it should be.

The data is fine and the markdown stage is fine; both were checked:

- Stored verbatim in the datastore: `"Aria: Hello! I'm here and ready to help. What can I do for you?"`.
- `mistletoe.markdown()` on the scrubbed text returns `<p>...What can I do for you?</p>` intact.
- `chatutil.scrub` only does thought-block surgery plus a final `strip()` — nothing that removes a
  mid-string character.

So the loss happens after markdown, in the rendering stage. The streaming path is ruled out: the
observation was of a *stored* message re-rendered on load, in a later session than the one that
generated it.

**Still unreproduced as of 2026-07-28** — one sighting, ten days, no recurrence. Treat it as an open
report rather than a known defect: it is not a Researchers' Night blocker on current evidence, and
it should not be *worked* until it reappears, because there is nothing to test a fix against. If it
does reappear, the thing to capture is the node ID and whether the window had just been resized.

**It is intermittent.** Restarting Librarian re-rendered the same stored node correctly, and it had
never been seen before that one occurrence. So this is not a deterministic function of the input —
feeding the stored string to the renderer in isolation may well render it correctly and prove
nothing. Budget for a race, not for a parsing bug: the vendored `DearPyGui_Markdown` renders from a
persistent worker thread (`CallInNextFrame._worker`), which is already known to be the shaky part of
that module (see the shutdown-segfault note in the vendored-dependencies section of `CLAUDE.md`, and
the untracked URL-highlight bug). A dropped character is consistent with a widget being built from
partially-updated state.

The shape of the loss is still worth recording, since it may fingerprint the race: a capital letter
starting a word after a sentence-ending period, replaced by whitespace rather than deleted outright.

Reproduction is the hard part and the first task — without it, any fix is unfalsifiable. Rendering
the same long-ish message repeatedly (reload the chat in a loop, or rebuild the view many times) and
diffing rendered text against stored text would be the way in.

One lead worth checking first, on the grounds that it is new in this same release cycle: the
Librarian **idle throttle** drops the render loop to ~12 fps when nothing needs updating, and the
vendored renderer hands work back to the main thread via `CallInNextFrame` — so frame cadence is part
of that handoff's timing, and throttling changes the width of any race window in it. A message
rendered on chat load, with the app otherwise idle, is precisely the case the throttle governs. This
is a hypothesis from timing coincidence, not from reading the interaction; if it doesn't pan out
quickly, drop it rather than building on it. (It is also why the observation is treated as a genuine
first occurrence rather than a long-standing bug finally noticed: a dropped letter is the kind of
thing this project's author does reliably catch.)

Discovered while committing the chat-template fix (2026-07-19).

## webfetch local (client-side) mode

`websearch` and `webfetch` currently live server-side only. `webfetch` is a candidate for running
client-side too (e.g. to fetch from the client's network vantage — VPN-internal sites the server
can't reach; the SSRF `allow_private_networks` opt-out would come into play there). Two constraints
shape how, and they say "defer", not "do now":

- **Licensing splits the two.** `websearch` is AGPL (a Python port of the SillyTavern-Selenium
  extension) and MUST stay server-side — pulling it into a client process would infect Raven's
  otherwise-BSD client. `webfetch` is Raven's own and could be BSD'd — EXCEPT its Tier-2 fallback
  currently borrows websearch's Selenium driver (`_fetch_tier2` → `websearch.get_driver()`), which is
  in the AGPL module. So a prerequisite is a **clean-room BSD Selenium driver factory** (e.g.
  `raven.common.webdriver`) that both webfetch and websearch use.

- **Not a `MaybeRemoteService` — the imagefx pattern.** webfetch is *stateless* (no cache, like
  nlp/stt/embeddings), so the 4-layer transparent-dispatch machinery (which earns its keep only when
  there's cache/shape state to hide, as in tts) is overkill. If local mode is built, do it
  imagefx-style: a clean `raven.common.webfetch` impl the client can call explicitly, with the server
  module delegating to it. No mayberemote class.

- **Low urgency.** Librarian already hard-depends on the server for websearch and RAG embeddings, so
  client-local webfetch does not unlock a standalone Librarian. Build only when a concrete
  client-vantage need appears.

Discovered while wrapping up brief 01 webfetch (2026-06-03).

## Context-window budgeting and conversation compaction (Librarian)

Librarian does not yet budget the prompt against the model's context window, nor compact long
conversations. After brief 02 (LM Studio compat), the loaded context-window figure captured per
backend in `llmclient.setup` (§5) feeds only two consumers: the identity/context line in the
character card (what Librarian *tells the model*) and the context-fill GUI indicator (§7). There
is no enforcement — nothing trims or summarizes history as the prompt approaches the window.

This will start to matter as the tool set grows and a direct file-upload affordance lands (a
scientific fulltext can push the user against the ceiling). When built: budget against the
*loaded* context length (never the model's theoretical max), prefer real `usage.prompt_tokens`
where the backend reports it, and add a compaction strategy (summarize-older-turns or
sliding-window) gated on the fill fraction. The §7 token-counting tiers (local tokenizer /
usage-calibration / idle-prefill) are the measurement substrate this would build on.

Discovered during brief 02 (LM Studio compat) kickoff (2026-06-04).

## Librarian chat input: make it multiline (Shift+Enter = newline)

The chat input (`app.py`, `dpg.add_input_text(tag="chat_field", ...)`) is single-line. This blocks
serious usage — pasting or composing a multi-paragraph prompt (a fulltext excerpt, a structured
question) is impractical. Make it a multiline input (`multiline=True`), with **Enter = send** and
**Shift+Enter = newline**. The send hotkey handler already intercepts Enter; it needs to check the
Shift modifier and insert a newline instead of sending when Shift is held (and a multiline box also
wants a sensible default height + growth behaviour). Pairs with the eventual file-upload affordance
(both feed larger prompts into the box).

Discovered during brief 02 GUI work (2026-06-04).

## Fleet-wide: shared two-phase DPG shutdown helper + audit

The DPG apps each hand-roll their render-loop teardown, and the pattern is fragile — `raven-librarian`
and `raven-avatar-settings-editor` both got it *wrong* independently, which is the signal it should be a
shared utility, not copy-pasted boilerplate. The correct shape (already in `raven.cherrypick.app`):

1. **Exit callback = cancel only, NO waiting.** DPG dispatches it from inside `render_dearpygui_frame`; a
   `wait=True` there deadlocks any task parked in `dpg.split_frame` (the frame can't complete while we wait,
   and `split_frame` needs the frame to complete). Signal cancellation only.
2. **Blocking drain + teardown in the render-loop `finally`,** on the main thread, BEFORE `destroy_context()`.
   And **drive both phases from the `finally` yourself** — do NOT rely on DPG having run the exit callback: on
   a fast/mid-boot close its callback-thread slot can be occupied, so it may never fire (this was the librarian
   hang). Call the cancel explicitly at the top of the `finally`, then the waiting drain, then `destroy_context`.

The librarian fix (commits TBD on `feature/librarian-lmstudio-compat`, 2026-06-04) also surfaced gotchas any
shared helper / per-app conversion must handle, beyond the two-phase skeleton:
- **`split_frame` after loop-exit hangs forever** (no frame will ever complete). Any background task that calls
  it (avatar renderer texture reconfigure; chat-view rebuild) must skip it once cancelled/shutting-down — see
  `DPGAvatarRenderer._split_frame_unless_stopping` and the `gui_updates_safe` guards in
  `DPGLinearizedChatView.build`.
- **Late submissions slip past the cancel.** A startup callback's tail can *submit* a debounced rebuild task
  AFTER teardown's cancel ran (librarian's `_resize_gui` → `_resize_gui_task` → `view.build`). Guard the
  GUI-mutating op itself (a top-level `gui_updates_safe`/`_shutting_down` bail), not just the task manager.
- **Startup frame callbacks race teardown.** `set_frame_callback`-deferred startup work runs on DPG's callback
  thread and can fire mid-teardown; guard each on a `_shutting_down` flag set at the very first action of shutdown.
- **In-flight GUI builds.** A top-of-function `gui_updates_safe` guard catches builds that *start* during
  shutdown, but not one already running when teardown begins; for full coverage the build loop must re-check
  per-iteration (currently `DPGLinearizedChatView.build` only guards entry + the scroll tail).
- **The `DearPyGui_Markdown` render worker is the worst offender — STILL OPEN.** `CallInNextFrame._worker`
  (`raven/vendor/DearPyGui_Markdown/__init__.py`) is a *persistent daemon* thread (not a managed task, no
  cancellation hook) that pulls a render queue and calls DPG — including `dpg.split_frame()` — on its own thread.
  Nothing stops it at shutdown, so on a mid-boot close with a URL-heavy message mid-render it keeps touching DPG
  across `destroy_context` → segfault (and its `split_frame` would park post-loop-exit). It needs a stop flag the
  worker checks (skip `split_frame` + stop processing when set), an app-side `markdown.shutdown()` called in the
  cancel phase, and ideally a drain (worker sets a "stopped" flag; teardown waits for it before `destroy_context`).
  This is the boundary layer the librarian whack-a-mole bottomed out on (2026-06-04) — the reason the shared
  helper must own a *global* "all DPG-touching threads stop before `destroy_context`" barrier, not just per-task
  drains. A `does_item_exist`/`nonexistent_ok` guard was added at the worker's `bind_item_handler_registry`
  (quieted the "Item not found" spam) but does NOT fix the segfault (other DPG calls + `split_frame` remain).

Per-app exposure (the bug needs a waiting drain in the exit callback AND a `split_frame`-using background task
that can be busy at close):

| App | Status |
|---|---|
| `cherrypick` | ✅ correct (the reference) |
| `librarian` | ✅ fixed 2026-06-04 (this saga) |
| `avatar-settings-editor` | 🔴 **identical bug** — only other `DPGAvatarRenderer` user, `stop(wait=True)` in exit callback, bare `finally` |
| `visualizer` | 🟠 partial — `finally` does `clear_background_tasks(wait=False)`, but no waiting drain there; `split_frame` in annotation/info_panel (interaction-triggered, narrower window) |
| `avatar-pose-editor` | 🟡 anti-pattern + 1 `split_frame`; exposure TBD |
| `xdot-viewer` | 🟢 anti-pattern structure, but no `split_frame` anywhere → this bug can't bite |
| `conference-timer` | 🟢 no exit callback, no heavy bg tasks |

Plan for a focused session: extract a shared `raven.common` helper that owns the cancel-in-exit-callback /
drain-in-finally / then-destroy_context sequence (and ideally the `split_frame`-skip-when-stopping idiom),
convert each exposed app to it, and test each with the finicky mid-boot-close repro. `avatar-settings-editor`
first (confirmed identical bug). Dovetails with the existing "extract `raven.common` into a toolkit" item.

Discovered during brief 02 §7 live testing (2026-06-04), when an accidental mid-boot Alt+F4 exposed the
librarian shutdown races.

## Librarian: in-flight AI turn bleeds into a new chat (turn-sequencing race)

Pressing "new chat" (or otherwise switching HEAD) while an `ai_turn` is still streaming doesn't cancel that
turn. `start_new_chat_callback` (`app.py`) just sets `app_state["HEAD"] = new_chat_HEAD` and rebuilds the
view; `ai_turn_task_manager` is `mode="concurrent"`, so the old turn keeps running and its `on_done`
unconditionally `add_complete_message(node_id)` + sets `app_state["HEAD"]` — so the previous chat's reply
appears in the fresh chat and clobbers the new HEAD. Observed live on a slow LM Studio generation: "new chat
→ AI starts writing immediately before I type → my message gets injected mid-stream." Same family as the
shutdown races (GUI state concurrency) — fold into that AoE session.

The fix is NOT just "cancel on new-chat": a cancelled turn still finalizes its partial message in `on_done`
(the cancellation path returns `action_stop`, then finalization runs), so it would still bleed. It needs (a)
HEAD-switching user actions (new-chat, branch nav, reroll) to cancel the in-flight turn, AND (b) the turn's
completion (`on_done`/`on_tool_done`/streaming render) to NOT touch the view or `app_state["HEAD"]` once the
user has navigated away — while still preserving the *stop-button* case (cancel but keep the partial reply in
the same chat). Cleanest approach: the turn captures its branch identity at start and its completion is a no-op
on view/HEAD if HEAD has since been switched to a different branch. Consider also disabling new-chat / nav
while a turn is in flight as a cheap interim guard.

Discovered during brief 02 LM Studio live validation (2026-06-04).

## Idle prefill fires even when the HEAD's token count is already exact (redundant, and slows the next turn)

`DPGChatController._context_prefill_entrypoint` schedules an idle prefill (5 s after a HEAD change) to get an
exact `usage.prompt_tokens` count and warm the backend KV cache. But it doesn't check whether the current
count is *already exact* — after a normal chat turn, the just-completed `invoke` already returned real `usage`
for this HEAD's prompt, so the indicator is already `X%` (exact) and the KV cache is already warm. The idle
prefill then fires anyway 5 s later, which is pure cost: a second full prompt-processing pass with no benefit.
Worse, if the user sends the next message while that redundant prefill is mid-flight, the two prefills (the
idle one and the real turn's) compete for backend compute and *slow down* the next turn. Observed live with
LM Studio: two expensive prefills running simultaneously.

Fix: the prefill entrypoint should cancel itself (no-op) when the current HEAD's count is already known-exact —
i.e. track whether the last count for this HEAD came from real `usage` (exact) vs. the calibrated estimate, and
only prefill when it's an estimate (or when the HEAD's prompt has changed since the last exact reading). The
exact/estimate bit already drives the `X%` vs `~X%` typography, so the signal is in hand; it just needs to gate
the prefill too.

Discovered during brief 02 PR-B work, reported by Juha from live LM Studio use (2026-06-04).

## Render the streaming thinking trace inside a bubble from the start, not just on completion

The thinking trace renders in two different shapes across a message's lifetime. *While streaming*, reasoning
paragraphs (`is_thought=True`) render as inline blue text — same flow as the visible answer, just tinted (blue
being, as we all know, the color of thoughts). *On completion*, `DPGCompleteChatMessage._render_text` turns the
same thought paragraph into a collapsible bubble with a cloud-icon toggle (Ctrl+T). So the user watches the
thinking stream inline, then it visually "snaps" into a bubble when the message finalizes.

It would be more coherent to render the streaming thinking inside a bubble *from the first reasoning token* —
the live thought bubble grows in place, then just gains its collapse toggle on completion, with no shape change.
`DPGStreamingChatMessage` would need its own thought-bubble container (analogous to the complete-message one)
that reasoning paragraphs render into, while content paragraphs render in the normal flow. Cosmetic/UX, not
correctness — the typed-event stream already cleanly separates the two channels (brief 02 §9), so the data is
in hand; this is purely a streaming-renderer presentation change.

Discovered during brief 02 §9 live validation, suggested by Juha (2026-06-04).

## Streaming thinking shows as gray (not blue) for models that pre-fill the opening `<think>` tag

QwQ-32B-style thinking models (and Qwen3-2507 as served by ooba) are trained to *begin thinking immediately*:
their chat template pre-fills `<think>\n` into the prompt tail, so the generated stream starts already inside
the think block and emits only the **closing** `</think>`. The `StreamParser` (`llmclient.py`) starts in
`_PS_TEXT` and only transitions to `_PS_THINK` on seeing an *opening* `<think>` — which never arrives — so the
entire thinking phase streams as `content` events and renders gray (visible answer), not blue (thought). The
*completed* message renders correctly: `chatutil.scrub` already recovers the orphan close (detects a `</think>`
with no matching open and prepends the opening tag — the long-standing QwQ-32B note at `chatutil.py:815-827`),
so the thinking consolidates into its bubble on finalize. Net effect: thinking is gray while streaming, then
"snaps" blue/into-a-bubble on completion. Stopping *mid-thinking* leaves it gray (the close never arrived).

**Not a brief-02 regression** — the pre-PR-B `on_llm_progress` had the identical limitation (its
`inside_think_block` flag also only flipped on a literal `<think>`, so a pre-filled open never registered live);
confirmed by diffing `09a88a7`. Models that emit the opening tag inline (e.g. ooba's Qwen3-VL-30B) are
unaffected — they stream blue correctly both before and after PR-B.

The hard part is that the opening tag's *absence is the correct, expected output* for these models, so there is
no in-stream signal until `</think>` lands — by which point the thinking has already streamed gray. Possible
fixes, none free: (a) start the parser in `_PS_THINK` when we know the model pre-fills the tag — needs a
per-model/backend signal we don't currently have at invoke time (the rendered prompt tail would show the open
`<think>`, but ooba renders server-side; an `/v1/internal/...` prompt-render probe could expose it); (b) on the
first orphan `</think>`, retroactively re-tint the already-committed streaming paragraphs as thought (recolor +
flip `is_thought`) — correct but invasive in the streaming renderer. Related to the "bubble from the first
token" item above, but distinct: there the thinking is *already* detected (blue, just inline); here it is *not
detected at all* until completion. This is the `chatutil.py:827` TODO ("add the opening `<think>` while
streaming, or to the prompt?") surfacing in the typed-event parser.

Discovered during the brief-02 ooba cross-backend regression test (2026-06-05).

## Parse Gemma's inline tool-call spelling if a raw-passthrough backend needs it

The `StreamParser` (`raven/librarian/llmclient.py`) parses the generic / Qwen inline tool-call form
(`<tool_call>{json}</tool_call>`) but not Gemma's: `<|tool_call>call:NAME{...}<tool_call|>` (inner pipes, a
`call:` prefix, and a non-JSON argument body using Gemma's `<|"|>`-quoted values). On LM Studio — the
live-verified Gemma 4 backend — tool calls arrive structured in the OpenAI `tool_calls` field, so there is
nothing to parse inline. The open question is whether a raw-passthrough backend (oobabooga, or a generic
OpenAI-compat server) serving Gemma emits the tool call inline in `content` instead — exactly the way it does
for the reasoning channel, which is why we added inline `<|channel>thought` parsing. If one does, Gemma
tool-calling on that backend would silently break: the call renders as text and never invokes the tool.

Deferred rather than built speculatively, because (a) the need is unverified — we don't yet know that any
backend we use passes Gemma tool calls inline-raw; and (b) the argument body is not JSON, so a correct parser
must be written against *real captured output*, not guessed. Resolve by: load Gemma 4 on ooba (currently on
Qwen3-2507; ooba is also overdue an update), enable tools, and capture the raw streamed `content` of a
tool-calling turn. If it carries `<|tool_call>call:...` inline, write a parser for that syntax (+ tests)
anchored on the capture. If ooba delivers the call structured instead, close this item — there is nothing to do.

Discovered during the brief-02 Gemma 4 reasoning-channel work (2026-06-05).

## Add built-in calculator and weather LLM tools (parked in brief 01 §6)

Two small built-in tools were scoped out of brief 01 (webfetch) v0 and parked for "after the retrieval
workstream wraps." Recording here so they survive brief archival:

- **Sandboxed expression calculator** via `simpleeval` — AST-walks the expression and restricts the allowed
  function set (math, abs, min/max, round, …) plus size limits; "sandboxing" reduces to picking the allowed
  set. Scope is *expressions*, not statements: `2+2`, `sqrt(...)`, arithmetic, comparisons — not "run a Python
  script with imports." ~a page of code, not its own brief.
- **Weather** via OpenMeteo — no API key, no cloud account; mirrors the `webfetch` tool shape. Small.

Both register as built-in tools alongside `websearch` / `webfetch` (tool registry in
`raven.librarian.llmclient`). Under content-parts (brief 03) their string output wraps as a single text part —
no special handling. Worked-out design and rationale: brief 01 §6 ("Out of scope for v0").

Flagged by Juha while wrapping brief 02 (2026-06-05).

## Reconsider the webfetch allowlist default: ship deny-by-default?

`librarian_config.webfetch_allowlist` defaults to `None`, which means **allow-all**: the allowlist gate in
`webfetch_wrapper` (`raven/librarian/llmclient.py`) is skipped entirely (`if allowlist is not None:`), so the
model may fetch any public URL — subject only to the server-side SSRF / private-network / scheme blocks. In
effect the webfetch "constrain the AI's initiative" power switch is **off by default**. This is arguably a
larger exposure than the opt-in `webfetch_trust_search_results` flag (which only does anything once an
allowlist is set): with allow-all, the model can already follow any link a poisoned search result or fetched
page feeds it — the prompt-injection→exfiltration vector the allowlist exists to bound — with no opt-in needed.

A curated, safe baseline already exists in the same config (`webfetch_default_allowlist`: DOI, arXiv, major
publishers, GitHub, Wikipedia, …); it's just not the default — the user opts in by assigning it. The question
is whether to flip the shipped default to deny-by-default, e.g. `webfetch_allowlist = webfetch_default_allowlist`,
so a fresh install is safe out of the box and the user *extends* rather than *enables* the list. Tradeoff:
convenience (allow-all "just works" for any site) vs. safety (the model can't reach an arbitrary host until the
user adds it — but user-typed URLs are already auto-allowed per turn, so the common "read this link I pasted"
flow is unaffected). Server-side network checks remain regardless; this is purely about bounding the model's
*initiative*.

Posture decision for Juha (security vs. convenience for the median scientific user). Noticed during brief-03
review (2026-06-05); pre-existing since the webfetch brief (brief 01) shipped, not introduced by the
content-parts refactor.

## Keyboard-layout-aware positional hotkeys across the fleet

Some Raven apps bind hotkeys *positionally* — keys chosen for their physical location rather than the letter they produce. The first case is `raven-cherrypick`'s WASD navigation (an alias for the arrow keys, plus `Q`/`E` for page up/down, for one-handed triage). On AZERTY (French) that cluster sits at ZQSD; on QWERTZ (German/Swiss) `Z` and `Y` are swapped — so positional bindings land under the wrong fingers for those users.

Offer a `config.py` option to remap the positional cluster per keyboard layout (at minimum WASD↔ZQSD), and ideally autodetect the active layout on the three OS families we support (Linux: `setxkbmap -query` / XKB, or the Wayland compositor; Windows: `GetKeyboardLayout`; macOS: `TISCopyCurrentKeyboardInputSource`). Same problem class as games that assume `[ ] \ '` exist as single keys and won't accept the AltGr'd equivalents on Nordic layouts — don't impose that UX pain on others.

When implementing, sweep the fleet for *every* positional binding, not just cherrypick's WASD. Until then, positional keys stay as aliases beside the layout-independent originals (see `raven-style-guide.md`, "Hotkey discoverability").

Detection mechanics for all three OS families are researched in `briefs/keyboard-layout-detection.md` — including the key finding that DPG reports *layout-translated* keys and hides scancodes (so physical binding isn't reachable, since Raven won't vendor/patch DPG), the two recommended strategies (config override, then OS position→char query for an auto-default), and the Wayland gap. Start there.

Discovered during cherrypick WASD navigation work (2026-06-07).

## Fleet audit: every hotkey discoverable in a tooltip + help card

Policy (now in `raven-style-guide.md`, "Hotkey discoverability"): every hotkey must be surfaced both in the `F1` help card *and* in the tooltip of the GUI control it triggers (bracketed, e.g. `"Open folder [Ctrl+O]"`). Most apps in the wild miss the tooltip half; Raven apps shouldn't.

Two prerequisites are already done as of the 2026-06-07 doc sweep, so this item is narrower than it looks:
- The standard "no shared keymap — keep surfaces in sync" warning comment is present at every hotkey handler.
- App-level help-card coverage is 100% — all seven DPG GUI apps (`visualizer`, `librarian`, `xdot_viewer`, `conference_timer`, `avatar/pose_editor`, `avatar/settings_editor`, `cherrypick`) construct a `HelpWindow`. No whole app is missing a card.

Remaining work is the *per-key* audit: for every bound key in each of the seven apps, confirm it is (a) listed in that app's help card and (b) named (bracketed) in the tooltip of the control it triggers, then fill the gaps. Note that filling a missing tooltip is a behavior change, not a doc edit — keep it as its own focused pass.

Discovered during cherrypick WASD navigation work (2026-06-07).


## Cherrypick: zoom-in doesn't upgrade already-cached preload neighbors

The preload cap is adaptive to the current zoom (`preload.mip_scale_for_zoom`): `schedule_neighbors` prefetches each neighbor at the smallest mip that displays crisply at the zoom in effect when it runs. But an already-cached neighbor is skipped (`if idx in self._cache: continue`), so if the user zooms *in* after a neighbor was cached at a smaller scale, that entry keeps its now-too-small mips. The first navigation to it then triggers the on-arrival augment (a one-time re-sharpen); take/donate cycles heal it thereafter.

Correctness is fine (augment fallback covers it); only the instant-crisp guarantee lapses for that one step, in the non-primary zoom-in-mid-browse workflow. If it ever feels worth closing: have `schedule_neighbors` re-issue a neighbor whose cached largest scale (`entry.mips[0][0]`) is below the current `mip_scale_for_zoom`, mirroring the capped-entry eviction `schedule_compare` already does for full-chain upgrades.

Discovered during cherrypick preload adaptive-cap work (2026-06-09).

## Remove the dead inline-`<think>` handling in the chat renderer

`DPGChatMessage._render_text_paragraphs` (`chat_controller.py`) still splits inline `<think>...</think>` out of a text content-part into a collapsible thought paragraph. That path is dead: since the June 2026 `reasoning_content` migration, thinking is separated before render — at load by `chatutil.upgrade_datastore`, live by the stream parser — so a text part never contains inline `<think>`. It's leftover from last autumn's demo code, not dismantled when the June 2026 thinking-block handling landed.

Remove the `inside_think_block` / `<think>`-detection machinery from `_render_text_paragraphs` (and check whether the `<think>` → "**>>>Thinking>>>**" cosmetic replacement in `_render_text` and the `is_thought` plumbing it feeds become simplifiable too). Verify unreachability first (grep for any message-construction path that could inject raw `<think>` into a content text part, bypassing both migration and the parser). Small, but it's a behavior-adjacent change, so give it its own commit + a quick live check.

Discovered during brief-03 Half-2 doc pass (2026-07-16); the renderer comments already flag the path as dead.

## Upgrade oobabooga and re-check Raven's ooba support

text-generation-webui (oobabooga) hasn't been pulled in a long time; its OpenAI-compat API may have drifted from what Raven's `llmclient` assumes. Upgrade the local ooba install, then re-validate the ooba code paths against the current version: backend-flavor detection (`detect_backend_flavor`), model-info resolution (`_resolve_model_info` — the `/v1/internal/model/info` shape, and whether ooba now exposes a VLM-capability field so `model_is_vlm` can be better than `None`), the `mode: "instruct"` request field, the explicit `continue_` flag, the reasoning/tool-call streaming shape, and the exact token-count endpoint. Live-test a real generation + a tool call + (if supported) an image attach through ooba.

Discovered 2026-07-16 (noted by Juha during brief-03 Half-2 pause).

## Make the Librarian chat composer text field resizable

The composer's multiline text field (`chat_field`, `app.py`) is a fixed height (`gui_config.chat_field_h`, ~5 rows). For essay-length prompts — common in scientific use — a fixed box is a toilet-paper-roll view of the input. Add a drag-to-resize affordance (or a fixed/expand toggle) so the user can grow the field when composing long messages. The composer's outer height is currently fixed on purpose (so the chat/avatar panels don't jump when the staged-image strip appears), so a resize handle would need to grow the whole composer and re-run the panel layout — reuse `_resize_panels`.

Discovered during brief-03 Half-2 composer rework (2026-07-17, flagged by Juha).

## Attach an image from a web URL (paste-URL path)

The image-attach GUI only supports attaching *local files* (composer paperclip → FileDialog → a `file://`
provenance). The storage and provenance layers already anticipate a web source: `imagestore.store_image_as_sidecar`
accepts `provenance_source="paste_url"` (and `"mcp:<server>"`), records an `https://` provenance `url`, and the
inline-image "Open source" action opens an `https://` URL in the browser — but nothing yet *produces* such an
image, so that branch is unreachable in normal use today. Add a user path to attach an image by URL (paste a URL
into the composer, or a dedicated "attach from URL" affordance): fetch the bytes, run them through the same
`store_image_as_sidecar` (which downsamples + keeps the original per the existing cap policy), and record the
web source as provenance. The image itself is stored as a sidecar exactly like a local attachment, so a saved
chat still reloads offline — only the provenance `url` points at the web.

Discovered during brief-03 Half-2 (2026-07-17, noted by Juha — the backend already accounts for it).

## Datastore scaling: a single `data.json` (+ flat sidecar dir) won't hold years of chats

Librarian stores *every* chat — all nodes, all payload revisions, across the whole forest — in one
`data.json` (`chattree.PersistentForest`), and every attached image as a file in one flat `<datastore>.images/`
sidecar directory. Both are fine now and for a long while, but neither scales to months/years of daily use:

- **The JSON**: `PersistentForest.save` serializes and rewrites the *entire* file on every autosave. As the
  forest grows to thousands of nodes with revision history, load-at-startup and each save get linearly slower,
  and a corrupted write risks the whole history at once.
- **The sidecar dir**: a single flat directory of content-addressed images degrades on some filesystems once it
  holds many thousands of entries (directory-scan and lookup costs); `list_sidecar_files` (used by GC) reads the
  whole directory each time.

Directions to weigh when it matters (don't pre-build): shard the sidecar dir by hash prefix (`ab/cd/<sha>.png`);
move the forest to an on-disk store with incremental writes (SQLite, or append-only revisions) instead of
whole-file JSON rewrites; and/or split/rollover the datastore (per-chat files, or archive old chats out of the
hot store). Any change needs a migration from the current single-file format (`chattree._upgrade` is the hook).

Discovered during brief-03 Half-2 checkpoint C (2026-07-17, flagged by Juha — surfaced by the global "Open chat
data folder" button making the single-store design visible).

## Colorblind-safe status signaling (ok/error flashes distinguished by color alone)

`animation.flash_button(ok=...)` (and, more broadly, Raven's flash/highlight vocabulary) conveys success vs.
failure by *color alone* — green for ok, red for error. That's invisible to the ~8% of men with red–green
colorblindness. Add a redundant, non-color channel so the distinction survives without color: an icon glyph
(e.g. check vs. cross) folded into the flash message text, a brief symbol overlay, or a shape/position cue. The
flash message string is already shown during the animation, so prefixing the ok/error message with a
distinguishing glyph is the cheapest first step. Audit the other color-only signals too (the "search green"
found-indicator, the VU meter's green/yellow/red — the latter has position redundancy already).

Discovered during brief-03 Half-2 (2026-07-17, flagged by Juha).

## Consolidate flash palette + internalize `ButtonFlash.original_theme`

Two related cleanups in `raven.common.gui.animation`, both fleet-wide (touching multiple apps), so deferred out
of the checkpoint that surfaced them:

- **Named palette constants.** The flash colors are magic tuples repeated across the codebase: the ok-green
  `(96, 128, 96)` / `(180, 255, 180)` is *literally* `ButtonFlash`'s own default, re-hardcoded in `flash_button`;
  the text-green `(180, 255, 180)` ("search green") recurs in ~7 places (`vumeter`, `visualizer/app`,
  `visualizer/annotation`, `xdot_viewer/app`, …); the error-red `(150, 96, 96)` / `(255, 180, 180)` is newer.
  Extract a small named palette (success/failure flash background + text) and have `ButtonFlash`'s defaults,
  `flash_button`, and the scattered literals reference it.
- **Internalize `original_theme`.** `ButtonFlash` takes an `original_theme` argument that every caller fills with
  `dpg.get_item_theme(<the tooltip>)` — apparently boilerplate the animation could read itself, at reification
  time in `start()` (before it binds the flash theme), where the true pre-flash theme is still bound. Ghost-mode
  instances (which don't reallocate) never capture, so the reified instance's snapshot is always the correct
  one. Removing the parameter would delete the boilerplate at all 19 call sites (across `librarian/app`,
  `librarian/chat_controller`, vendored `file_dialog/fdialog`, `visualizer/word_cloud`, `visualizer/info_panel`,
  `cherrypick/app`) and remove the `else 0` placeholder in `flash_button`.
  - **Verify first — the parameter may exist for a reason.** The plan assumes `dpg.get_item_theme` returns a
    usable value for every item type ButtonFlash is applied to. Callers already call it successfully on the
    *tooltip* items they pass, so internalizing the tooltip snapshot is behavior-preserving there — but the same
    is unproven for `target_text` items and any other type. It's plausible the explicit parameter exists
    precisely because `get_item_theme` misbehaves on some item (and the reason was never written down). Before
    deleting the parameter, confirm `get_item_theme` round-trips on every item type actually flashed; if it
    doesn't somewhere, keep the parameter (or a per-widget capture) and document *that* as the reason.

Discovered during brief-03 Half-2 checkpoint C (2026-07-17, flagged by Juha while reviewing `flash_button`).

## Expose the docs-DB source files behind a reply's RAG citations

When the AI composes a reply using the document database, it sees a set of retrieved snippets, and that
provenance is already tracked per turn (the payload's `retrieval` field records the query and the snippets the
AI saw). But none of it is surfaced in the GUI — the user can't see *which* documents fed a given answer, nor
open the originals. Expose it: for each stored reply that used RAG, offer a way to view the source snippets and
open the original files they came from (each indexed doc carries its `path`, so "open file" / "open folder"
reuses the `common_utils.open_file` / `open_in_file_manager` provenance-button machinery already built for image
attachments). Design questions when we get to it: where the affordance lives (per-message expander? a side
panel?), and how much of the snippet vs. the whole document to show.

Discovered during the plain-text/PDF interlude (2026-07-18, requested by Juha).

## VLM reranking of mixed-modality search results (post-Nomic)

Once the search feature encodes both text and images (the Nomic plan, autumn 2026 — this supersedes the
CLIP/`clip-ViT-L-14` note still sitting in `hybridir.py`'s image-support TODO comment), a single result set can
mix text snippets and images. Idea: feed the candidate result set to the VLM and ask it to re-prioritize —
neural reranking that can actually *look at* the image hits, not just their embeddings. Composes naturally with
the "hand PDF pages to the VLM as ~1MP images" v2 idea from the attachment work: once images are first-class
search hits, VLM reranking over them is a small step.

Orthogonal sub-question that also needs answering at that time: what should *keyword* (BM25) search do for
images, which have no text? One option: ask the VLM to describe each image and store the description as a
BM25-able text representation. This is the same "the ingestor needs an embeddable/searchable representation of
each file" generalization that the PDF-extraction callback foreshadows — for text it's the extracted plaintext,
for an image it's a caption plus the image embedding.

Discovered during the plain-text/PDF interlude (2026-07-18, raised by Juha).

## Drop the Intel Mac / macOS 10.x install workaround

The README has an "Install on an Intel Mac with MacOSX 10.x" section (torch 2.2.x pin + removing ChromaDB) for
an Intel x86_64 Mac too old for modern PyTorch. That platform is now effectively dead — new Macs are Apple
Silicon (M-series) on recent macOS. Remove the section (and stop maintaining the torch-version snippet inside it,
which has to be kept in sync with the pinned trio in `pyproject.toml`). The general macOS caveat — remove the
`pytorch-cu128` source so torch resolves from PyPI — stays, since that applies to *all* Macs.

Discovered during the plain-text/PDF interlude (2026-07-18, Juha: a coworker's new M-series Mac on current macOS).

## Same file formats in the docs DB and in chat attachments

The docs database and chat attachments should accept the *same* set of formats. A user who can attach a file
to a message reasonably expects to be able to drop it in the documents folder, and vice versa; a split between
the two is arbitrary from outside.

This is cheap to hold to, because `raven.common.docextract` is already the single chokepoint for both — the RAG
ingester and the attachment path both call `extract_text`. Adding a format there serves both surfaces at once,
so the symmetry costs nothing extra as long as new formats are added *there* rather than at one call site.

Two halves, gated differently:

- **Office documents** (`.docx`, `.odt`, `.pptx`, `.odp`): no blocker; done for 0.2.8.
- **Images**: the asymmetry runs the other way. Attachments have supported images since brief 03 — it is the
  docs DB that lacks them, so this is the database catching up to a capability that already exists, not a new
  one. Blocked on the Nomic embedder (text and vision in one aligned space): until then there is no way to
  retrieve an image by a text query except captioning at ingest time. Tracked as part of the multimodal search
  plan in `TODO.md` ("RAG PDF ingestion — polish", and the Nomic multimodal-search item).
  - **Visualizer needs images too, and that is the expensive half.** Once the docs DB accepts images, a
    constellation where the Librarian can hold an image but the Visualizer cannot show one is incomplete in the
    same arbitrary way this item is about. Placing images *on the map* is the easy part — a shared text/vision
    embedding space means an image gets a position exactly the way an abstract does, with no new machinery. The
    cost is on the presentation side: the tooltip and the info panel are built around a record that is title +
    abstract + keywords, and neither has any notion of showing a picture. Expect real UX design work there
    (what a hovered image looks like at tooltip size, how a mixed text/image cluster reads, what the info panel
    does with a record whose content *is* the image), not just a rendering branch.

Raised during the 0.2.8 release scoping (2026-07-29, Juha).

## Spreadsheets in the docs DB and attachments (`.xlsx`, `.ods`)

Left out of the office-formats work deliberately: a spreadsheet is a different problem class wearing the same
file picker. Its content is tabular, so "the text of this file" is not well defined — reading a sheet row-major
into a paragraph produces something that chunks badly for retrieval and reads poorly when folded into a chat
message. Getting it right means deciding how a table becomes prose (or whether it should become Markdown table
syntax instead, which the model can actually read), and how a multi-sheet workbook maps onto one document.

Worth doing eventually — research data does arrive as spreadsheets — but as its own design question, not as
three more lines in the extractor's dispatch.

Raised while scoping office-format support (2026-07-29, Juha).

## Read documents as page images, for figure- and math-heavy sources

Current extraction is **text-layer only**, for PDFs and (as of 0.2.8) office formats alike. That loses exactly
what matters in the sources this project exists to read: equations, plots, diagrams, tables-as-figures. A paper
whose argument lives in its figures extracts to prose that omits the argument.

The other route is to render each page to an image and hand it to a vision model — which Raven can already do
for an *attached* image, so the missing piece is the rendering and the decision of when to use it. Design
questions: when to choose images over text (always for the figure pages? a heuristic? user's choice per file?),
the token cost (a page image is expensive next to a page of text — interacts with the attachment budget in
`llmclient.fit_attachments_to_context`), what to store in the sidecar (the original file, the rendered pages,
or both), and how a page image participates in RAG retrieval at all (this is the same wall the images item
hits — it wants the Nomic aligned space, or captioning).

Distinct from OCR for scanned documents (`TODO.md`, "RAG PDF ingestion — polish"): OCR recovers text that was
always meant to be text. This is about content that is *not* text and never was.

Raised while scoping office-format support (2026-07-29, Juha).

## Librarian doesn't check that the LLM backend has a model loaded

Starting Librarian against a backend with no model loaded produces no warning; the first turn fails with a raw
`HTTP 400 Bad Request` and the backend's own error text ("No models loaded. Please load a model in the developer
page or use the `lms load` command."). This surfaces as an assistant error message — the failure is visible, but
only after the user has composed and sent something, and the wording is the backend's rather than a Raven-side
"no model is loaded; load one in your backend and reroll."

The connection-time query in `llmclient.setup` already talks to the backend, so it is the natural place to notice
the empty-model case and say so up front — either at startup, or by disabling send with an explanatory tooltip
the way the paperclip does on a text-only model.

Discovered during brief 07 (2026-07-29, raised by Juha).

**This is one instance of a general gap: Librarian's error handling is a standing TODO**, deferred in favour of
prototyping. Beyond the LLM-failure path — which does surface properly, as a spoken and rerollable message
since 2026-07-27 — most failures are not reported to the user at all: no model loaded, an unreadable document,
the server going away mid-session. It has been survivable because the person running Librarian has so far been
the person who can read its log.

Two things change that, and both are on the design track rather than hypothetical. An **avatar-first mode**
(`briefs/design/lab-assistant-hci-sketch.md`) has no console to be sitting at and no log to fall back on, so an
unreported failure is indistinguishable from an assistant ignoring you. And anything **served to a phone**
(`briefs/design/corpus-interrogation-sketch.md`) reports to a device with no access to the log at all.

So the sweep is worth doing as its own pass rather than one dialog at a time — and worth doing before either of
those tracks, which will otherwise each build half of it differently.

## No way for the user to attach a document from a URL

The attach button takes a local file. There is no affordance for "attach *this URL* as a document to my
message", even though the storage layer was designed expecting one: `sidecarstore.base_provenance` names
`"paste_url"` as a `provenance_source`, and nothing in the tree emits it.

What exists today is not a substitute. A user who pastes a URL into the chat is relying on the *model* to
decide to call `webfetch` — which needs tools enabled, needs the model to actually make that call, fetches
again on every reroll, stores nothing locally (so the chat does not reload offline the way an attachment
does), and contributes nothing to the context-fill estimate before it happens. An attachment is a different
thing: pinned content, materialized once, owned by the message.

The work splits in two, and the halves are not the same size:

- **An article page.** Nearly free — `client.api.webfetch_fetch(url)` already returns `{"content": markdown,
  "url", "title", "spaSuspected"}` with the server doing the two-tier fetch, SSRF and scheme checks, and URL
  rewriting. Store `content` as a `.md` sidecar, `provenance_url=url`, `provenance_source="paste_url"`, and it
  arrives as a document chip like any other. Most of the cost is GUI: where the URL is entered, and what the
  chip shows while the fetch is in flight.
- **A file behind a link** — an arXiv PDF, a `.docx` on a course page. This one has no plumbing at all.
  `webfetch` is built to return *extracted text*, so there is no path that hands back raw bytes, and pointing
  it at a PDF gets a readability pass over binary rather than a stored document. Doing it properly means a new
  server endpoint that returns bytes plus content type, under the same SSRF and scheme checks, with a size cap
  (a fetch the user initiated is still a fetch of something unseen) and dispatch on the returned content type
  rather than on the URL's apparent extension. Worth noting that for an academic user this is probably the
  more valuable half — a paper is usually a PDF behind a link, not an article page.

Doing only the first half is defensible and would close most of the everyday gap, but it should be a deliberate
choice rather than an accident of which one was easier — the button says "attach from URL" either way, and a
user whose arXiv link silently produces a readability pass over PDF bytes has been misled by their own tool.

**The button has to DWIM, and that is the hard part rather than the fetching.** One URL, and the user's intent
is not recoverable from it. The canonical case is an arXiv abstract page, which is simultaneously:

- an article page whose abstract is exactly what someone might want attached, cheaply, without pulling down a
  40-page PDF;
- a page carrying a link to the full text, where what the user wanted was the *paper*, and the abstract page is
  merely how they navigated to it;
- and a page whose full text may also exist as HTML (`arxiv.org/html/...`), which `webfetch` already rewrites
  toward — so even "fetch the full text" has two answers with different fidelity.

No amount of content sniffing settles that, because the ambiguity is about intent, not about bytes. So the
design question is what the UI does with it: fetch one and say which, offer the choice when a page advertises a
full-text link, or attach the page and let the model follow the link with `webfetch`. Whatever is chosen, the
user has to be able to see which reading they got — silently picking one and being right half the time is the
outcome to avoid.

**Targeted at 0.2.9**, deliberately out of 0.2.8's scope: the second half is not small, and the DWIM question
above wants deciding rather than guessing. Until then the workaround is the obvious one and worth stating in
the docs if users ask — download the document, then attach the file.

The backend was built anticipating this, during brief 03; what is missing is the fetch machinery and the whole
UX side.

Raised during the 0.2.8 format work (2026-07-29, Juha).

## Version the chat datastore file, so migrations can be skipped once applied

Raised 2026-07-29 (Juha), from noticing that `appstate.backfill_sidecar_metadata` walks every revision of every
node at every load. Nothing today tells a loaded datastore apart from one that has already been through each
migration, so all of them run unconditionally and their cost is paid on every startup forever.

**The version belongs to the file, not to a node.** The first instinct is a lone root node holding it, and that
runs straight into two problems: `prune_unreachable_nodes` would collect it, and `_get_system_prompt_node_id`
takes root nodes to be system prompts, so it would be misread as one. Both are symptoms of the same thing —
metadata about the file is not data in the forest, and putting it there means every consumer of the forest has
to learn to ignore it.

So: an **envelope**. `PersistentForest._save` currently writes the bare nodes dict
(`json.dump(self.nodes, ...)`), so the change is to write `{"format_version": N, "nodes": {...}}` and have
`_load` detect which shape it is holding. Detection is unambiguous rather than heuristic: node IDs are gensym
strings (`gensym#forest-node:...`), so a top-level key named `nodes` or `format_version` cannot be one.

Preferred here over a companion `<datastore>.meta.json`, which can be separated from the data it describes — by
a partial copy, a backup that catches one file, a manual move. The envelope travels with the thing it versions.

Note this is not an argument against companion metadata files in general, and specifically not against the
per-sidecar `<filename>.meta.json` descriptions: those *want* to be separate files, because their whole point
is to be readable without the datastore, and a sidecar that loses its description degrades to a hash rather
than becoming dangerous. A format version is the opposite case — useless on its own and actively harmful when
it disagrees with the file it claims to describe.

Three migrations would be gated by it, and the version has to cover all of them: `chattree._upgrade` (forest
structure), `chatutil.upgrade_datastore` (payload schema), and `appstate.backfill_sidecar_metadata`. One
integer, each landed migration a step from N to N+1.

**Handle the too-new case from the start**, because it is the one that corrupts rather than merely wastes time.
A datastore written by a newer Raven and opened by an older one is the dangerous direction: the old build has
no way to know it is looking at a format it does not understand, and will happily write back a version of the
data with the newer fields dropped. Refusing to open — or opening read-only with a clear message — is the whole
reason to have a version number rather than just a migration marker.

## A no-avatar mode, with the chat tree in the panel the avatar vacates

Raised 2026-07-29 (Juha), and the opposite end of the same axis as the avatar-first mode sketched in
`briefs/design/lab-assistant-hci-sketch.md`. On the road, on a laptop, with no power outlet in sight, the avatar is
the most expensive thing on screen — it holds VRAM that the LLM would rather have, and it renders continuously,
which is exactly what a battery objects to. So: a mode that does not load it at all.

**The natural occupant of the vacated panel is a visualization of the chat tree.** Librarian's history is a
branching forest and always has been, but the GUI only ever shows the linearized path from HEAD to root — so
the structure the datastore is built around is invisible in the app built on it. A laptop-sized window is also
exactly where knowing "where am I in this conversation" pays most.

Which suggests a framing worth adopting before building any of the three: **the right-hand panel is what the
mode actually varies.** Avatar-first fills the window with the character; the standard mode shows avatar plus
toggles; road mode shows the tree. That is one mechanism with a swappable panel, not three layouts — and
deciding it now is much cheaper than discovering it after the second one is hardcoded.

The classic mode makes that concrete rather than merely tidy: there the chat tree is **toggleable, overlaying
the avatar panel exactly** when open. So the same rect has alternative occupants at runtime, and the no-avatar
mode is the degenerate case where one of them is never constructed at all. Which also means the classic mode
gets a slice of this item's benefit for free, if the avatar is *paused* while covered rather than left
rendering frames nobody can see.

**The view itself is not tracked here** — it is `TODO.md`'s "Nonlinear chat view / chat graph editor", which
already carries the mechanism (generate the layout ourselves, no GraphViz) and the constraints (limit visible
depth; the full tree will not render at interactive FPS). This item is only about the *mode*, and about the
observation that the two want designing together: the view needs a panel, and the mode is what frees one.

What the mode adds on top of that view: not loading the avatar at all. That is where the VRAM and the battery
saving come from, and it is a startup-path decision rather than a hide/show toggle — worth being explicit
about, since a mode that merely hides the avatar panel saves nothing that matters here.

## Visualizer's importer should read the document database, not just `.bib` files

Visualizer ingests BibTeX databases. That is where it started — scientific abstracts, one entry per paper — and
it is now the wrong boundary. What the importer actually wants as its source is **the same document database
Librarian searches**, with scope support, so that the two apps are two views of one corpus rather than two
corpora that happen to live on the same machine.

The payoff is the plain-language version: someone drops a pile of Word documents into the documents folder and
gets a semantic map of them. No conversion step, no separate import, no BibTeX in sight. That is a different
product from "a tool for visualizing bibliographies", and a much easier one to explain.

Several things that already exist turn out to be pointing here:

- **The document database already accepts far more than abstracts** — plain text, PDF, office formats, saved
  web pages (2026-07-29). Every one of those is a document with text; nothing about mapping them semantically
  is specific to a BibTeX entry.
- **Scoping is already planned for the docs DB** (subdirectories as umbrella topics — see the Librarian
  README's note that all documents currently share one search namespace). The same scope concept is what a map
  needs: map this topic, not the entire library.
- **`raven-burstbib` exists to bridge the gap in the other direction**, splitting a `.bib` into per-entry files
  so Librarian can read them. If the importer read the docs DB directly, that tool becomes a convenience rather
  than a required step, and BibTeX becomes one input format among several rather than the privileged one.

Gated on the Nomic migration, which is where the shared embedding space comes from — today the two apps embed
separately and there is no single space to place both in. Also interacts with the image-support and
cleanup-view items above: once both apps read one corpus in one space, "show me the map", "search my documents"
and "show me what cleanup is about to delete" are the same machinery pointed at different subsets.

Substantial enough to deserve a brief rather than an incremental change — it revisits what the importer's input
*is*, and the importer is already flagged for a stage-separation refactor.

Raised during the 0.2.8 format work (2026-07-29, Juha).

## Let the AI drive the constellation's own views (tools, and then voice)

Falls out of the item above, and is easy to miss because it looks like prose: *"show me the map"*, *"search my
documents"*, *"show me what cleanup is about to delete"* are not descriptions of features, they are things a
person would **say**. Once the apps share one corpus in one embedding space, each is a plausible tool call.

What makes this cheap rather than speculative is that the hard parts already exist and are load-bearing
elsewhere. Librarian has tool-calling with validation and dispatch (`websearch`, `webfetch` are the working
examples). It has Whisper STT on the mic button, and an avatar the user is already addressing. What is missing
is not the interface — it is that there is currently nothing for such a tool to *point at*, because the map and
the chat do not share a corpus. That is exactly what the importer item above changes.

The shape, roughly: a small family of view-control tools the model can call — open a map of a named scope,
show the retrieval hits for a query as a map rather than a list, show the cleanup preview. Each is a thin call
onto machinery that would already exist; the tool layer is what turns it from a button someone has to find into
something they can ask for.

Worth noting *why* this is more than a gimmick, since "voice control" usually is one. The scope names are the
user's own folder names, the queries are their own words, and the objects are their own documents — so the
utterance is short, unambiguous, and about material the model already has in context from the conversation that
prompted it. That is the narrow case where speech beats a mouse, rather than the broad case where it loses.

Sequenced after the importer/corpus work, not before: without the shared corpus there is nothing to point at,
and a tool that opens an empty map is worse than no tool.

Raised during the 0.2.8 format work (2026-07-29, Juha's observation, from noticing that the phrases were
commands).

## Semantic grouping in the sidecar cleanup preview (once Nomic lands)

The "Clean up & save" preview lists the orphaned sidecars it is about to delete. There is no grouping, because
today there is nothing to group *by*: sidecars are content-addressed, so the set is globally unordered, the
filenames are hashes, and the only other handle is a per-file provenance URL that may be absent. The preview
therefore shows an arbitrary wall of tiles, and the recovery decision is per-file.

The Nomic embedder (text and vision in one aligned space — see the format-symmetry item) changes what is
possible here, because it gives *every* orphan a vector regardless of kind. An image and a PDF land in the same
space, so the orphan set becomes clusterable as one collection rather than two.

The payoff is not prettier tiles, it is a better decision. Laid out by cluster with a label per group, the view
says "these nine are the plot figures from the thesis discussion, these three are the conference slides"
instead of showing twelve unlabelled squares — and recovery moves to *per cluster*, which is the granularity a
person actually thinks at when deciding what to keep.

**But this belongs in Visualizer, not in a Librarian dialog** (decided 2026-07-29). Visualizer already *is* a
semantic map: clustering, cluster labelling by keyword extraction, selection, tooltips, an info panel. A
cluster view built into Librarian's cleanup dialog would be a second, worse copy of an app we already ship,
maintained separately and diverging. The right shape is for the orphan set to become something Visualizer can
open — the cleanup dialog stays a plain list for the everyday "delete 40 MB of nothing" case, and the "I need
to actually look at these" case hands off to the tool built for looking.

That makes this dependent on Librarian↔Visualizer integration as well as on Nomic, and worth *not* starting
until both are in place. Note the overlap with the Visualizer image-support item above: showing an orphaned
attachment on the map needs the same tooltip and info-panel work, so the two land together or not at all.

One consequence to plan for rather than discover: a GC view is not a *reading* view, it is a *choosing* view —
the user has to mark items for deletion or rescue. Visualizer's selection model was built for inspecting a map,
not for accumulating a working set across clusters and acting on it. Expect the selection UX to want a revisit
at that point, and treat that as part of the cost of this item rather than as a surprise inside it.

Raised while implementing brief 03 D (2026-07-29, Juha's idea, and Juha's placement call).

## HTML pages whose content is produced by running them

`raven.common.docextract` reads HTML through `trafilatura`'s readability extraction, which looks at markup. A
page that has no text in its markup — because a script writes it at load — therefore extracts as empty, and the
attachment path folds in `[no extractable text]` while the docs DB skips the file.

Two rather different situations share that symptom, and only the second is worth solving:

- **The bare shell of a JS-rendered site.** The content is genuinely absent from the file; it lives on a server
  the page would have fetched. Nothing local can recover it, and the honest recourse is to fetch the URL with
  the `webfetch` tool instead of attaching the saved file. Not a gap — a correct refusal.
- **A self-contained single-file app.** The data *is* in the file, as a literal inside a `<script>` element,
  and the DOM is built from it at load. Here the extractor is leaving real, present content on the floor. The
  motivating example is a mod-listing table built with claude.ai (`diva_modules.html`), which renders exactly
  the sort of table a reader would want indexed.

  **Expect this to stop being a corner case.** Chat assistants now emit self-contained HTML artifacts as a
  routine output format, so a growing share of the documents a user has worth keeping will arrive in precisely
  this shape — and unlike a scanned PDF, there is no lossy origin to blame, the content is right there. That
  makes this the highest-value of the reading gaps on this list, and a fair candidate for its own brief rather
  than an incremental fix.

  Partly mitigated today: the `<title>` is recovered separately from the body, so such a file indexes under its
  own name instead of vanishing. That is enough to find the document, and not nearly enough to search it.

The hard part of the second case is telling inline *data* from inline *code*. Dumping every `<script>` body into
the index would work beautifully for a hand-built page carrying a JSON array, and catastrophically for a page
carrying a minified React bundle — hundreds of kilobytes of unreadable tokens poisoning retrieval for everything
else in the database. Some possible shapes, none yet chosen:

- Read only *declared* data: `<script type="application/json">` and `application/ld+json`. Unambiguous and
  cheap, but does not catch the motivating case, which uses a plain `<script>` with a JS array.
- Fall back to script text only when readability found nothing *and* the inline script volume is under some
  budget. Heuristic, but degrades in the right direction: it fires exactly when there is nothing to lose.
- Render the page in a headless browser and read the resulting DOM. Solves it completely, and is the one option
  that also handles a shell page with a reachable server. **Ruled out for the automatic paths**, and this is a
  settled decision rather than an open trade-off: dropping a file into a watched folder must never be enough to
  start executing that file's scripts. Webfetch's Tier 2 is a different posture — a URL someone explicitly
  asked to fetch — and the SSRF and sandboxing care taken there is the measure of what automatic rendering
  would have to earn. If it is ever built, it belongs behind a per-file action the user takes deliberately,
  never behind the ingester or the attach dialog.

Raised while adding HTML support (2026-07-29, Juha's example).

## Rendering LaTeX equations in the chat log

Models emit LaTeX — `$...$`, `$$...$$`, `\begin{equation}` — whenever the subject is mathematical, and Librarian
currently shows it as source. For a research assistant aimed at scientific work this is the wrong way round: the
notation exists because it is easier to *read* than the prose it replaces, and we are showing the half that is
harder.

This is the *output* side, and distinct from the two document-reading items it sounds like:

- Different from "Read documents as page images": that is about getting math *in* from a source whose equations
  are pictures. This is about getting math *out* to the screen, from source we already have in the best possible
  form.
- Different from `.tex` ingestion, which already works — the file is read as text, and the model handles it fine.

The work is in the renderer. `DearPyGui_Markdown` (vendored, and already substantially ours) knows nothing about
math, so something has to turn a LaTeX fragment into pixels and splice it into the flow of a message: a
typesetting pass producing a texture per equation, at the right baseline and the right size for the surrounding
font. Streaming makes it sharper — an equation is only renderable once its closing delimiter has arrived, so the
renderer needs to hold an incomplete fragment as source and swap in the rendered form when it completes, without
reflowing everything above it.

Not tracked before now because it was too large to take on solo; that calculus has changed.

Raised during the 0.2.8 format work (2026-07-29, Juha).

## Librarian leaks its server-side avatar instance when it doesn't exit normally

Librarian releases its avatar instance in `app_shutdown` (`raven/librarian/app.py`), which is registered with
`atexit`. That covers the normal exit, but `atexit` handlers run only when the interpreter shuts down cleanly —
so every abnormal exit leaves an orphaned instance on the server, holding VRAM and a render slot until the
server process is restarted. Observed: seven stale instances accumulated on the server during one session of
GUI testing.

Two paths reach it, and the first is by far the common one:

- **A signal kills the process.** Librarian installs no `signal` handlers at all, so `SIGTERM` (plain `kill`,
  a session manager logging out, a supervisor stopping the app) terminates it at the C level with no Python
  cleanup — `app_shutdown` never runs. This is what produced the seven instances: the test-harness `kill`s
  during this session. A `signal.signal(SIGTERM, ...)` handler that calls `app_shutdown` and then re-raises
  fixes the whole class, and is the actual ask: *if the process is still alive enough to make an HTTP call,
  it should make this one.*
- **`sys.exit` from a non-main thread.** `_load_initial_animator_settings` calls `sys.exit(255)` on two error
  paths, and by its own comment it runs on DPG's callback thread. `sys.exit` outside the main thread raises
  `SystemExit` in *that* thread only, so it neither runs `atexit` nor actually exits the process. Both paths
  are reached after `avatar_instance_id` is assigned, so both leak. (That this leaves the process running
  rather than exiting is inferred from how `SystemExit` propagates out of a worker thread; worth confirming
  against DPG's callback dispatch before fixing, since the fix differs depending on the answer.)

Worth keeping the unload best-effort in either case — the server may legitimately be gone first, which
`app_shutdown` already handles by swallowing `ConnectionError`.

Discovered during brief 07 GUI testing (2026-07-29, raised by Juha).
