# Brief 16: chat graph view

> **Line numbers verified and corrected 2026-09-01.** Every pointer in the 2026-08-04 draft had drifted —
> they were read from a shallow clone already a couple of hours behind a moving tree — but each named a
> thing that exists. The numbers below are the current ones. Two pointers resolved to nothing and are
> annotated where they appear: `README.md:590`'s "port xdottir" sentence has been removed (the port
> landed), and `README.md:588`'s recent-chats wish is now `README.md:776–777`.

**Researchers' Night work (2026-09-26), and it takes precedence over `atmospheric-dust.md`** (decided
2026-08-05: the graph adds more value; dust lands only if time remains after this). Written 2026-08-04. The
feature has been recorded in four places since roughly a year ago; what changes now is that the expensive
part is built, and that the demo gives it a purpose beyond navigation.

The ordering is safe in the direction it needs to be: `atmospheric-dust.md:30–35` takes its priority-band
scheme *from* `crt-display.md` §0, so the dependency runs dust → crt. Nothing points back at dust, and
dropping it leaves both `crt` and this brief intact — which is what makes it a sound last item rather than an
overcommitted one. (Both have since landed and live in `done/`, so the ordering argument is history.)

**Closes on landing**: `TODO.md`, *"Nonlinear chat view / chat graph editor"* (`TODO.md:658`), the
placeholder button at `app.py:1700–1708` — *deleted*, the control being a mode-toggle checkbox instead
(see below) — the recent-chats wish at `raven/librarian/README.md:776–777`, the right-panel occupant note
at `README.md:415`, and the help card's "not built yet" sentence at `app.py:1906`.

(An earlier draft also listed one of the help card's two "This is a tech demo" disclaimers, on the grounds
that `TODO_DEFERRED.md:1183` called it still true in substance — old chats stored but unreachable. Both
disclaimers were removed 2026-08-04. The substance still holds until this view exists: the datastore keeps
every branch and the GUI can reach none of them. It just no longer has a sentence to correct.)

## Why it is cheaper than the record says

Two entries are stale in the same direction:

- **The `TODO.md` item says Librarian must generate `.xdot` code.** It does not. `XDotWidget.set_graph(graph:
  Graph)` takes a `Graph` directly, and `Node(x, y, w, h, ...)` / `Edge(src, dst, points, shapes)` take
  explicit geometry. The graph is built in memory. No text format, no re-parse, and no DOT escaping applied
  to chat content.
- **`raven/librarian/README.md` said a suitable DPG widget was missing and xdottir should be ported.**
  That port happened; `raven/common/gui/xdotwidget/` is the result, and `raven-xdot-viewer` is built on it.
  The sentence has since been removed from the README, which is why it has no line number here.

The widget already exposes every hook this needs: `on_click(node_id, button)`, `on_hover`, `pan_to_node`,
`set_highlighted_nodes`, `search`, and `text_compaction_callback` for the label-does-not-fit case.

**Framing worth keeping (Juha, 2026-08-04):** Librarian's chat view was always going to be the first real
production client of the widget. `raven-xdot-viewer` is the bonus, not the target.

## Step zero — `set_graph` has no callers and no tests

> **Done 2026-08-25, and the answer is yes: the door opens.** A chat-shaped `Graph` built in memory — a root,
> two children, two more under the second — went into `set_graph` and came out rendered, with no GraphViz and
> no xdot text anywhere. After `zoom_to_fit` the drawlist held exactly the expected fifteen items: five nodes
> of two shapes each, four edge lines, one background. **So the rest of this brief can be costed on the
> assumption that "the hard part is already done" is true of this door and not only of the renderer.**
>
> Three things came out of the half hour beyond the yes/no:
>
> - **Every hook the brief relies on works against a hand-built graph**, not just against a parsed one:
>   `pan_to_node`, `set_highlighted_nodes`, and `search` (the index is built in `Graph.__init__`, so it needs
>   no parser). Hit-testing too.
> - **One defect found and fixed.** `XDotWidget` could not be instantiated twice in one DPG context — its
>   tooltip window and group took fixed tags, so the second died on "Alias already exists". An app holding one
>   widget never meets this; the first test to build two met it immediately. Now per-instance, via `gui_uuid`
>   as Raven's own widgets do.
> - **The tests are written** — `raven/common/gui/xdotwidget/tests/test_widget.py`, seven of them — which is
>   the debt item 1 below says is owed regardless of this brief. They are the fixture a chat-shaped graph
>   needs, so the feature work starts from a graph that is known to render.
>
> One measurement worth carrying: **culling is by viewport**, so a freshly-set graph is only partly drawn
> until something establishes the view. `zoom_to_fit` is what the chat view will want on load.

Checked 2026-08-04. Every production path goes through `set_xdotcode`; `set_graph` has never run outside the
widget, and has no direct test. (The `set_graph` calls in `tests/test_search.py` are `SearchState`'s method
of the same name, not the widget's.)

Two consequences, and the first stands on its own:

1. **The missing tests are owed regardless of this brief.** A public method with no caller and no coverage is
   a defect in the widget, not a footnote to a feature that happens to want it.
2. **Before the rest of this brief is costed**, hand-build a small `Graph` — a root, two children, one of them
   with two siblings — feed it to `set_graph`, and see it render. Half an hour. It either confirms the
   estimate or shows that "the hard part is already done" is true of the renderer and not of this particular
   door.

## The decision: build `Graph` objects, do not emit xdot

Recorded because it will be re-litigated, and because the argument against is a good one.

**The case for xdot text** is putting more of the codebase on the daily hot path: the parser
(`DotScanner` / `DotLexer` / `DotParser` / `XDotParser`) currently runs only when someone opens the viewer
app, and cold code hides bugs.

**Why it does not win here.** The recorded decision is manual layout with no GraphViz, so emitting xdot means
emitting *positioned* xdot — `pos`, `bb`, `_draw_` draw-command strings. That is a serializer for the format
whose only consumer is our own parser: more code than constructing `Node` objects, a round-trip through a
format neither end needs, and serializer bugs that present as parser bugs — which degrades exactly the
diagnostic value the exposure was meant to buy. The version of the argument that fully works is
DOT-plus-real-GraphViz, since then the reference implementation produces the xdot; but that is a hard
dependency plus a subprocess per graph update, and it kills live update, which is the demo move.

**Note what is common to both routes**: `Viewport`, the renderer, highlight state, search, `pan_to_node` and
the coordinate transforms go hot either way. The differential is one entry layer.

**Pay the hot-path debt explicitly** rather than pretending the argument was wrong: add chat-shaped fixtures
to `tests/test_parser.py` — deep chains, wide sibling fans, long labels that force compaction. Cheaper than a
serializer and aimed at the same decay.

## The framing that shapes v1: explanatory before navigational

The `TODO.md` item frames this as navigation — "jump to chat node by ID". The demo wants explanation: making
"an LLM is a multiverse generator" visible to a general audience who have never seen one.

Both are wanted, and where they pull apart v1 follows the demo. Two interactions carry the explanatory
weight, and they are equal partners (Juha, 2026-08-05):

- **Generating a branch.** Reroll from the graph and watch the new sibling appear. The thesis in one gesture,
  and the reason live update is a requirement rather than a nicety.
- **Exploring the branches that exist.** Seeing the multiverse laid out, and moving through it, is what makes
  the point land for someone who has only ever seen a chatbot produce one answer.

**Fragment search is wanted, and is v2** (decided 2026-08-05, scoped with brief 14). It is the only way to
find anything in a multiverse short of aimless browsing, and `raven-xdot-viewer` already has the machinery —
`XDotWidget.search` / `highlight_search_results` / `next_match` / `prev_match` are on the widget.

**What pushes it out of v1** is the interaction with windowed siblings, below: a match outside the current
window is not merely off-screen but absent from the built `Graph`, so `next_match` would have to move the
window and rebuild rather than pan. That is a different operation from what the viewer app does, and it is
not a small addition to it.

**The demo does not need it.** Aimless browsing is what a visitor with five minutes will do anyway, and
arguably what the exhibit wants them to do. The loss is to later working use, not to Researchers' Night —
worth writing down so it is not relitigated in September.

**It is the companion to brief 14, not a duplicate.** Brief 14 searches *within one chat* — the linearized
branch, message as the match unit. This searches *across the tree*. Same corpus, different scope, and they
must not diverge on what a user would notice: case sensitivity, regex, whether tool results and thinking
traces are searchable. Brief 14 lands first and decides those; this inherits them and adds the
window-moving `next_match`.

## Tool nodes will swamp the view unless collapsed

The tree does not alternate user / AI. `ai_turn` creates a node per LLM response **and** a node per tool
call, so with documents and tools enabled one conversational turn is three to six nodes. A visitor looking
for "the things it could have said instead" would mostly see plumbing.

**Decided 2026-08-05: collapse a turn's tool nodes into their assistant node**, with a count badge, and
**expandable on click** for a visitor who wants the machinery. Try this shape first; other options only if it
fails in practice. This also puts the round-versus-call distinction on screen in the same vocabulary brief 15
is fixing it in — a round is what gets a node, the calls within it are the badge's count.

**Use `ICON_GEARS` for the badge.** The chat log already marks tool-role messages with it (the three-cogs
glyph), and `chat_controller.py:1101–1103` records the reservation deliberately: `ICON_GEAR`, the universal
settings glyph, is held for the future settings dialog and must not be used here. Reusing the same symbol
means a visitor who expands a badge sees the marker they were just shown.

Worth deciding early, because it changes what the layout is laying out.

## The shape of real trees, and what the layout must handle

Measured by inspection rather than assumed (Juha, 2026-08-05): after the AI's canned greeting there is a
**huge fan-out at the first user message**, which is effectively the entry point of a chat session. Below
that, branches are mostly narrow chains with occasional reroll siblings.

So the layout problem is not "a tree" in general. It is one very wide level plus narrow chains, and the
horizontal extent of that one level is the bottleneck. Reingold–Tilford assumes bounded fan-out and is the
wrong default here.

**Windowed siblings with anchors.** Show the next and previous few siblings, plus the first and last ones.
This is a focus-plus-context layout rather than a full one, and it is what keeps the wide level renderable.
Navigation between windows: clicking a sibling re-centres, with the rebuild happening after the smooth scroll
finishes. Whether custom controls are needed to jump by several siblings at once, or whether click-to-recentre
covers it, is open.

**That wide level is also the recent-chats list.** `raven/librarian/README.md:776–777` wants one, and notes that
"recent chat" is ill-defined in a nonlinear format, guessing that the user's first message is a good enough
splitting point. The fan-out shape says the guess is right, and this view already renders that level. So the
level deserves special treatment — recency ordering, possibly timestamps — and this brief closes that item
too.

**Depth limiting** is recorded as a hard constraint (the `TODO.md` item: the full tree will not render at
interactive FPS). For the demo case it is less pressing than that suggests, since walk-up conversations are
short. What is large is *depth over a long-lived chat plus the width of the session level*, and scoping to
the tree containing HEAD plus windowed siblings is what handles it.

> **Corrected 2026-08-12.** An earlier version of that sentence said "the accumulated forest is what is
> large", and an earlier passage assumed an open-house evening accumulates dozens of roots. **A root is a
> distinct system prompt text**, not a session — `appstate` keeps one root per variety of card, so a chat
> written under an older card is rooted at its own. An evening at one system prompt produces **one** root and
> dozens of branches beneath the greeting. The forest is never wide at the root level; the width is one level
> down, at the first user message, which this brief already identifies correctly. The scoping decision may
> still be right; the reason given for it was not.

### Roots became first-class on 2026-08-12, after this brief was written

Multi-root system-prompt storage landed, so showing the root level is now a natural thing to consider — and
the out-of-scope entry below ("a forest view across all roots") was decided against a world where roots were
effectively singular.

**What the root level is, and is not.** A root is a distinct *system prompt text*; the character's name,
avatar and voice live in `config.py`. So the root set is largely **version history of one character's card**,
not a character selector. Two consequences:

- **Clicking a root does not switch character.** The app would keep rendering the configured avatar and voice
  while the chat sits under a different system prompt. That mismatch needs a decision, and it is the main
  reason not to expose roots casually.
- **What it does give is reachability of chats written under older cards** — precisely the "access old chats"
  gap this brief says it closes. Those chats are in the datastore and unreachable from the GUI today.
  **Archaeology rather than switching**, and worth having on those terms.

So the graph has **two wide levels doing different jobs** — cards at the root, sessions one level down — and
this brief designed for only one. Whether the root level gets the same windowed treatment, a different
affordance, or stays out of scope is now an open question rather than a settled one; see settle-item 7.

**Consequent cost.** `chattree.get_siblings` calls `get_all_root_nodes`, which is an O(n) scan over every
node in the forest. If the graph view shows the root level, every root-sibling lookup pays that — and the
memoization that exists is at the `chat_controller` layer (`_scan_for_root_nodes` plus a live-node filter),
not in `chattree`, so a graph view calling `chattree` directly does not inherit it. **This decision
determines whether the deferred item on that scan needs acting on at all.**

## Truncation must be visible, and it is one mechanism serving two cases

**Rule (Juha, 2026-08-05): a node with no visible links means the graph genuinely ends there.** Truncation
must therefore always show itself.

Note that this makes the sibling window and the depth limit the *same* problem: a windowed sibling row that
simply stops looks identical to a branch that really has no more siblings. So build one gap primitive — a
clickable "…N more" element, itself a node-like thing in the layout — and use it for both the horizontal
case (siblings beyond the window) and the vertical one (depth beyond the limit).

## Revisions: parked, and correctly so

`add_revision` stores multiple payloads on one node — a second multiverse axis that a node-link diagram does
not show at all. An earlier draft of this brief suggested a badge.

**Withdrawn**: revisions are not accessible anywhere in the GUI today, since everything picks the latest
(Juha, 2026-08-05). Exposing them here would be a dead affordance, and a view that ignores them is
*consistent* with the rest of the app rather than lying about the datastore.

They become visible when message editing lands — fixing a typo, continuing a prematurely sent user message —
and the badge becomes right at that moment. Park it there rather than here.

## Look and colour

The reference is the diagram in `raven/librarian/README.md` that explains the tree structure: rounded
rectangles, truncated labels, and pointer pills (`SYS`, `NEW`, `HEAD`) drawn as a **separate visual class**
from nodes.

The pointer pills earn their keep in the demo beyond navigation: `NEW` shows that starting a new chat is
just moving a pointer, which is the multiverse thesis in miniature.

**Fill encodes branch membership** — on the current linearized path versus off it — rather than role. That is
the right call for this audience, since the story is the paths not taken and that must read from across a
room.

**Which leaves role without a channel.** The `U:` / `AI:` prefix carries it textually, and the text is the
first thing to go when zoomed out and compacted (`text_compaction_callback` exists for exactly that case).

**Decided 2026-08-05: role glyphs**, reusing the ones the chat log already carries rather than picking
afresh. The tool role's is `ICON_GEARS`, and `chat_controller.py:1101–1103` documents why it is not `ICON_GEAR`
— that one is reserved for the settings dialog. Whatever the other roles use in the chat log, carry the same
symbols across, so that the two views name the same things the same way.

**Labels** come from `chatutil.content_to_text` (brief 14 uses it for the same reason), compacted by the
widget.

### The role glyphs are the PNGs, and the widget cannot draw images yet

The marks the chat log uses for turns are `raven/icons/system.png`, `user.png`, `ai.png` and `tool.png` —
image assets, not font glyphs. (`ICON_GEARS` is a different thing: it marks a tool call *inside* a message,
and the collapsed tool-count badge should reuse it, per above.)

**`XDotWidget` cannot render images — but not for the reason recorded.** `parser.py:633–635` parses xdot
image shapes and skips them with a warning, on the stated grounds that DPG drawlists lack image support.
**That premise is disproven by code in this repo**: `chat_controller.py:615–616` calls `dpg.draw_image` into a
drawlist to paint these very icons. Correct the comment while implementing; it will otherwise keep costing
someone the same investigation.

So the open question is not availability but integration — an image shape has to cooperate with `Viewport`'s
coordinate transform and zoom the way the other shapes do. Smaller, and not a probe question.

**Reuse `chat_controller.gui_role_icons` rather than loading the PNGs again.** It is a dict keyed by role
(`assistant`, `system`, `tool`, `user`) holding already-registered texture tags, and it is where the
per-character override is already resolved: `_load_instance_textures` looks for `{stem}_icon{ext}` beside the
character image and shadows the generic AI texture when found — which is how Aria gets her own icon
(`raven/avatar/assets/characters/other/aria1_icon.png`). Pointing at the dict inherits the override for free;
pointing at `raven/icons/ai.png` would silently lose it, and the loss would show up only when a character is
loaded.

### Resampling: scale with the node, clamped at native size

DPG samples nearest-neighbour, which is why the tree carries a custom GPU Lanczos scaler. But the shipped
role icons are **64×64 already, high-quality downscales meant for 1:1 use** (`aria1_icon.png` is 64×66;
originals live in `raven/icons/00_workfiles/cropped/`, e.g. 719×722 for the generic AI icon).

**Decided 2026-09-01: the icon is a fixed fraction of node height, clamped to 64 px on screen.** So it
never upsamples past the asset, and it shrinks with everything else when zooming out.

An earlier draft of this section recommended a *constant* 64 px screen size, on the grounds that a
fixed-size icon stays readable when zoomed out to take in a wide sibling fan. That argument inverts at the
zoom it was written for: a wide fan is exactly the view where nodes are a few dozen pixels across, and a
64 px icon is then wider than the node carrying it, so the row becomes overlapping icons with no nodes
visible behind them. Constant screen size fails first in the case it was chosen to serve.

Downsampling a 64 px texture through DPG's nearest-neighbour path is what the chat log already does today
(64 px assets into a 32 px rect), so this needs no new machinery and introduces no inconsistency. The
alternatives both cost real work and neither is needed:

- **A different fixed size** wants a new *asset*, generated offline from the originals, rather than a runtime
  Lanczos pass. That is how the 64 px files were produced in the first place, and an asset is cheaper and
  more inspectable than a startup GPU step.
- **Lanczos at render time** needs the full apparatus: `mipchain`
  (`raven/common/image/lanczos.py`) plus the selection rule in `mip_scale_for_zoom`
  (`raven/cherrypick/preload.py`). Note the trap if quality turns out to want it — `mipchain`'s `min_size`
  defaults to 64, tuned for Cherrypick's photographs, so a 64 px icon produces a chain of length one and the
  aliasing survives the machinery.

Make the size rule a parameter on the shape rather than a constant in the layout, so the two can be
compared by looking at them.

*One asset note found while checking this, not belonging to this brief*: the chat log draws these 64 px icons
into a `gui_config.chat_icon_size = 32` rect, so it takes DPG's nearest-neighbour path today. The fix is a
32 px asset or a 64 px display size, rather than a runtime pass.

*Asset provenance, for whoever needs to regenerate one*: the shipped generic AI icon comes from
`00_workfiles/cropped/ai_generic_original.png`, and Aria's character override `aria1_icon.png` is
byte-identical to `new_ai_64x64.png`, hence sourced from `new_ai_original.png`. The two generations in the
workfiles are what make this confusing from filenames alone — the *new* AI icon became a character icon
rather than replacing the generic one.

Scope, and note it runs opposite to the hot-path argument: because v1 constructs `Graph` objects directly,
it needs only an `ImageShape`, renderer support for it, and those existing textures. Teaching
`XDotAttrParser` to stop skipping `I` operations is a separate follow-on — cheap once the shape exists,
closes the TODO, and makes real xdot files carrying images render in `raven-xdot-viewer`.

## Interaction: preview and commit are separate

Decided 2026-08-05:

- **Clicking a node on the current branch** scrolls the chat log to that message. Reuses `scroll_view` plus
  `highlight_widget`, the pair `DPGChatMessage._make_jump_to_tool_call` already uses.
- **Clicking a node on a different branch** scrolls the *graph* — re-layout, refresh the nearby siblings — 
  rather than switching to it.
- **Committing (moving HEAD) is a second, deliberate gesture.** Click again, a button, or something else;
  this wants a decision or a prototype.

The two-tier shape is worth stating as a principle rather than a mechanism, because it is also a demo-safety
property: **browsing the multiverse is non-destructive, and only a deliberate act changes state.** A visitor
can explore freely without leaving the session somewhere the next visitor inherits.

**Right-click is ruled out** (2026-08-05): `XDotWidget.__init__` already binds `on_open_url` to right-click
for nodes carrying a URL. Left-click is spent on preview, so the commit gesture is a second left-click on an
already-previewed node, a GUI button, or a modifier-click. Prototype before choosing.

## Where the panel goes

**Corrected 2026-08-05.** An earlier draft suggested a separate OS window or a second display. Neither is
available: DPG gives one real OS window, which becomes the viewport. That is why the original plan in
`TODO_DEFERRED.md:4397`ff assumed a panel.

What *is* available is DPG windows — Visualizer's word cloud feature is the precedent. Whether that suits a
graph view is open.

**Constraint either way: keep the layout flexible enough for the road mode to be added later.** The road-mode
item's framing is that the right-hand panel is what the mode varies — avatar-first fills it with the
character, standard shows avatar plus toggles, road mode shows the tree. That is one mechanism with a
swappable panel rather than three layouts, and it is much cheaper to leave room for now than to retrofit.

Note the tension the demo creates, unresolved: the road-mode design has the tree overlay the avatar panel
with the avatar paused while covered. On Researchers' Night the avatar is the draw, so pausing it to show the
graph trades the more compelling artifact for the more explanatory one at the moment both are wanted.

## Live update

**Rebuild the whole `Graph` on each change** (decided 2026-08-05). Simple, and almost certainly fine at demo
tree sizes with windowed siblings. Revisit only if measurement says otherwise.

## Out of scope for v1

- **Editing the tree from the graph.** `chattree` has `delete_subtree`, `reparent_subtree` and friends, so it
  is tempting. Destructive operations driven from a graph a stranger is clicking is a bad first version.
- **Revisions**, per above — they belong with message editing.
- **A forest view across all roots.** The windowed wide level covers what the demo needs. Reopened when
  roots became first-class on 2026-08-12, and closed again on 2026-09-01 — see decision 7 below.

## Settled 2026-09-01

The seven open items are decided, and four more surfaced while deciding them. What follows is the
implementation contract; where a decision reverses something above, the section above says so too.

1. **Panel placement: two child windows sharing the avatar panel's rect, shown and hidden.** Not a DPG
   window and not a true overlay with its own z-order. This was already decided in `TODO.md:661` and is
   recorded here only because this brief re-opened it. The view is built as a component class taking a
   `gui_parent`, so the rect is one line in `app.py` if this is ever revisited.
   - **The avatar pauses while covered**, per `TODO.md:662–663`, and the gate needs its own path rather
     than a term in the idle branch. Two reasons, both checked: the existing pause is guarded by
     `config.idle_timeout is not None` (`avatar_controller.py:358`), so a visibility term added inside it
     would be dead for anyone who turned idle-off off; and `ping` resumes unconditionally
     (`avatar_controller.py:388–390`), so without a visibility term there, any chat rebuild un-pauses a
     covered avatar — and `view.build()` calls `ping`.
2. **Commit gesture: a second left-click on the already-previewed node, plus a button in the graph's own
   toolbar** enabled while a node is previewed. Both call the same function; the click is the fluent
   gesture and the button is the discoverable one. **The button's tooltip names the click**, which is what
   makes the fluent gesture findable at all. Preview state must be visibly highlighted, so that what a
   second click will do is on screen before it happens.
   - Note what the two-tier shape is *for*. Moving HEAD is itself non-destructive here — nothing is lost
     and the graph can always get you back — so this is about legibility rather than safety, and should
     not be hardened as if it were about safety.
3. **Tool-node expansion is per-node**, keyed by a set of expanded assistant-node IDs. Same cost as a
   global flag, since the whole `Graph` is rebuilt either way, and "what is inside this one" is the
   question a visitor actually has.
4. **Sibling window: the focused sibling ±2, plus the first and last as anchors, with a "…N more" gap
   between.** Clicking a gap re-centres the window on the middle sibling of that gap — so the gap
   primitive *is* the jump-by-several control and no extra widgets are needed. Up to nine items at a
   level.
   - **No ±10 buttons here**, unlike the chat log's sibling row. A gap click bisects the remaining run,
     which beats a fixed stride on a fan of unknown width. Whether that in turn makes the chat log's ±10
     buttons redundant is a real question and an open one — it wants empirics from this view first, and
     nothing here should change them pre-emptively.
5. **One ordering rule everywhere: left to right is creation order.** "Recent" is expressed by where the
   *window* sits — the newest end, by default, at the session level — rather than by reversing that one
   level, which would otherwise read backwards from every other level in the same picture.
   - **Timestamps go in the node tooltip to start with.** Whether the session level also wants one drawn
     into the node is deliberately left to look at in the running GUI; build the label so a second line
     can be added cheaply.
6. **An icon is a fixed fraction of node height, clamped to 64 px on screen.** This reverses the earlier
   recommendation of constant screen size; the reasoning is in the resampling section above.
7. **v1 renders HEAD's root only**, with a visible but inert "…N more cards" marker above it so the
   truncation stays honest. Cross-root navigation stays out: clicking through to another card's chats
   would leave the configured avatar and voice rendering against a different system prompt, and that
   mismatch wants a decision of its own. Consequence: `get_all_root_nodes` needs no index for v1, so the
   deferred item on that O(n) scan does not need acting on yet.
   - The cheap follow-on, if the archaeology is wanted before the mismatch is solved: let the marker
     re-scope the graph to another root as *preview only*, with commit disabled there. No HEAD move, so no
     mismatch, and old-card chats become visible even though they are not yet reachable.

### The greeting is becoming optional, so the wide level is defined structurally

**The wide level is the children of `app_state["new_chat_HEAD"]`** — not "depth 2", which is how the
sections above reason about it. The greeting node is on its way to being optional (Juha, 2026-09-01), and
*optional* means per-chat rather than a global switch: one datastore will hold chats in both shapes. When
there is no greeting, `new_chat_HEAD` is the root itself and the wide level moves up one.

**Consequence: pointer pills are a list per node, not one pill.** With no greeting, SYS and NEW land on the
same node. Cheap if built that way from the start.

### Four more, settled at the same time

**A. Module split.** `chatgraph.py` — the pure builder: datastore plus view state in, a `Graph` out, no DPG
beyond the shape classes, which are plain data. `chatgraph_panel.py` — the DPG panel: the `XDotWidget`, the
toolbar, the click wiring, the avatar pause. The split is what makes the layout unit-testable, which is the
whole difficulty of this feature; nothing about the layout is checkable through a widget. Standard agile
caveat applies — if the split fights the code, stop and reconsider rather than forcing it.

**B. Layout: layered, with HEAD's lineage as a centred spine.** One row per depth; siblings windowed around
the spine node at each level. Reingold–Tilford is the textbook fit for a tree in general and is overkill
once windowing bounds the visible set — and a fixed spine gives the position stability of `TODO.md:664`
nearly for free. The vertical window mirrors the horizontal one: root, gap, the last ~12 ancestors, HEAD,
HEAD's children. An off-spine sibling that has children gets a downward gap marker, so its truncation shows
the same way every other truncation does.

**Position stability is a UX requirement, not a nicety** (Juha, 2026-09-01), and where the graph must move
to accommodate new content, it moves *smoothly*. The widget already does this: `set_graph` replaces the
graph without touching pan or zoom (it only updates the bounds), `pan_to_point(animate=True)` runs through
`SmoothValue`, and the widget tracks a focus node of its own (`_focus_node_name` / `get_focus_node`). So the
recipe is a deterministic layout followed by `pan_to_node(focus, animate=True)` after each rebuild: the
anchor node stays put and the view slides, rather than the content jumping under a fixed viewport.

**C. Live update is driven by a monotone `generation` counter on `chattree.Forest`**, bumped on every
mutation; the panel reads it on the animator tick and rebuilds when it moves. About five lines in the
datastore, against edits at the dozen-odd mutation sites in `chat_controller` — and it is the honest
version of the alternative, which was to poll `(HEAD, len(datastore.nodes))` and would miss a delete and a
create landing in the same frame. `chattree` is foundation code, so this was asked rather than assumed.

**D. Dark mode by default, with a toggle.** Author the graph colours light and let the renderer's
`dark_mode=True` lightness inversion do the work — the same path a parsed graph takes, so the two cannot
diverge, and no API change. Raven's interface is dark everywhere; `raven-xdot-viewer` is the exception and
exposes a toggle, so this view carries the same toggle. Note `set_dark_mode` is module-global state in the
renderer, which is harmless with one widget in the process and worth remembering if that stops being true.

### Left open on purpose, to be settled by looking

Three things the builder has taken a position on that only the running GUI can confirm. Written down so
they are re-examined rather than inherited.

- **A gap click bisects the run it hides**, which is not what a reader expects from a "…12 more" control —
  the expectation is a step, and this jumps to the middle. It is worth trying anyway because it reaches
  anywhere in a fan in O(log n) clicks where a step takes O(n) (Juha, 2026-09-01: "not what people expect,
  but might be what they actually *want*").
- **Whether the layout wants a real tidy-tree algorithm.** Neither of the two limits that were expected to
  decide this actually does, so it comes down to how the picture reads:
  - *Screen space is not the binding one.* **The picture may spill past the panel's edges** rather than
    being sized to fit inside it — the view pans, and a graph running off the edge reads more like the
    shape the tree actually has than a graph trimmed to a rectangle does (Juha, 2026-09-01).
  - *Neither is rebuild speed*, which was the other candidate and has now been measured —
    `investigations/chatgraph-rebuild-cost/`. Cost tracks the number of boxes drawn, not the size of the
    forest: about 1 ms at `siblings_each_side = 5`, and a fifth of a frame at 20, on a twenty-thousand-node
    forest. `siblings_each_side` was raised from 2 to 5 on the strength of that. The measurement also
    settles decision 7's loose end — `get_all_root_nodes` is 0.37 ms at that size, once per rebuild, so it
    needs no index for this view's sake.
  - So the ceiling is **legibility**, which only looking can set.
- **Pointer pills are outlined rather than filled**, reasoned from the renderer's dark-mode contrast rule
  picking a text colour from the element's fill. That reasoning is from reading the renderer, not from
  seeing it drawn.

### Found by measuring: the depth window can hide the level the sibling window is for

Turned up on 2026-09-01 while timing rebuilds, and not asked by anything above. **With HEAD deep in a long
chat, the depth window elides the session level entirely.** A chat twenty messages deep gives a spine of
twenty-three; keeping the root and the last eleven drops everything between, and the wide level — the
children of `new_chat_HEAD` — is in that gap.

So the level this brief says doubles as the recent-chats list disappears exactly when the conversation is
long enough for the user to want out of it. The root is pinned against the depth window already, for
reasons that apply here just as well: it is what says where you are rather than what was said.

**Decided 2026-09-01: pin it.** The depth window keeps a prefix at the top of the tree and a run at the
bottom, with one gap between them. The prefix runs down to and including the session node — the child of
`new_chat_HEAD` this branch began at — rather than being the root alone.

Kept *whole* rather than as its two ends: `new_chat_HEAD` normally sits directly under the root, so the
prefix is three nodes, and pinning only the ends would split the elision into two runs and spend a gap box
on hiding a single node. Falls back to the root alone when `new_chat_HEAD` is not on this branch (a chat
under an older card), or when the prefix would take more than half the budget — a prefix that crowds out
the nodes near HEAD has answered *where am I* at the cost of *what is happening*.

### Road mode makes the avatar optional, so the panel cannot assume one

Noted 2026-09-01 (Juha). The on-the-road mode is coming, and in it the avatar is **not constructed at all** —
that is where its VRAM and battery saving come from, and `TODO_DEFERRED.md`'s item is explicit that a mode
which merely hides the avatar panel saves nothing that matters.

**The shape that makes this cheap** (Juha, 2026-09-01): road mode still constructs the avatar's child
window, blank and never shown, and **leaves the graph open permanently**. So the two occupants exist in
both modes and the rect arithmetic does not change; what road mode varies is that one of them is empty and
the other never hides.

That follows from where the contention actually is: *the two compete for screen space only when both are
available.* With no avatar there is nothing to trade against, so there is nothing to toggle, and the
question of what the panel shows by default answers itself.

**So the panel has a preference and a current occupant, and they are not the same thing** (Juha,
2026-09-01). Three rules, in order:

- **The control is a checkbox in the mode-toggle row, not the toolbar button** (Juha, 2026-09-01). It sits
  beside *Speech* and *Subtitles* — the group that already governs the right-hand panel — and it is
  offered **only when an avatar exists**, since with nothing to trade screen space against there is
  nothing to toggle.
  - Two reasons it belongs there rather than in the bottom toolbar. It is *persistent state*, and that row
    is where Librarian keeps persistent state, the toolbar being for one-shot actions. And it stays put
    while the graph is up: the row sits below the shared rect, so the switch that put the graph there does
    not disappear under it.
  - **So `chat_open_graph_button` is deleted rather than wired.** It has been a disabled placeholder since
    it was added; the brief's "closes on landing" list means it stops existing, not that it starts
    working. Note the label it never got to wear would have been wrong anyway — "Open graph view", for a
    control that has to close it again.
  - This widens the fourth toggle group's meaning from *what the avatar does* to *what the right-hand
    panel shows*, which is the same reframing road mode makes and worth making in the README's wording
    when this lands.
- **When the avatar's video auto-offs, the graph takes the panel** rather than the panel showing
  *"[Video is off]"*. A dead placeholder is the one thing that rect is certainly not worth.
- **When the video comes back, so does the avatar — if that is what the user asked for.** Hence the
  preference: the automatic switch is a loan of the rect, not a change of mind, and the button is what
  states the mind.

**The loop to avoid, which this arrangement invites.** Showing the graph covers the avatar; covering the
avatar pauses it; a paused avatar is "video off"; and "video off" is the condition that shows the graph. Read
naively, the avatar never comes back.

The cut is that **the auto-switch keys on the idle detector, not on whether the video is running.** Going
idle shows the graph; `avatar_controller.ping` — activity — hands the rect back. The visibility-driven pause
is downstream of the switch and must never be an input to it. Two signals that look like one, and the whole
mechanism turns on keeping them apart.

**A third consequence, in the help card** (Juha, 2026-09-01). The card tells the reader to switch on
*Chat graph* to reach an old chat, and in road mode there is no switch — the graph is already up — nor an
avatar to describe its position relative to. So that sentence has to vary by mode. For now it carries no
locator at all, a label being findable on its own; saying where the toggle is would be right in one mode
and wrong in the other, which is worse than saying nothing.

Two consequences for the panel commit, both cheap now and awkward later:

- **The pause gate is conditional on there being something to pause.** `avatar_renderer` may be absent, so
  the visibility term is "pause it if it exists", not "pause it". In road mode nothing ever pauses,
  because nothing was ever started.
- **The rect is not the avatar's to own.** Placement is two child windows sharing one rect
  (`TODO.md:661`), and in road mode one of them is blank — so whatever computes that rect has to run
  without an avatar in it. Today `_get_avatar_panel_size` is what computes it, which is the name to watch:
  the geometry is the *right-hand panel's*, and the avatar is one thing that can fill it.

### First live look, 2026-09-01

The panel rendered on the first run: branch colouring, dashed gap boxes, arrowheads, the HEAD pill, label
truncation. What did not work, and what wants changing. **Juha's list is his; the two defects below it are
mine, found in the same run.**

**Raised by Juha, and not yet acted on:**

- **Font size.** He did not say which direction and I have not asked. My own reading of the screenshot is
  that the labels sit small inside their boxes, but that is a guess about his complaint, not his words.
- **Nodes need a role name.** The role is currently carried by nothing at all: the plan was role *glyphs*
  (the PNGs the chat log uses), and those are the last commit of the sequence because they need an
  `ImageShape` the widget does not have. A text role is available now and does not wait on that. Worth
  noting that the brief already predicted this hole — *"which leaves role without a channel"* — and
  answered it with glyphs alone.
- **Defaulting the view to HEAD leaves the bottom half of the panel blank.** HEAD is at the bottom of its
  own spine, so centring on it spends half the panel on the empty space below the tree.

**Raised as an idea, to be discussed** (Juha, 2026-09-01): **a message with attachments should show that
it has them**, with images drawn as thumbnails once `ImageShape` exists. The datastore already knows what a
message carries — `general_metadata["sidecars"]` on the payload.

Checked 2026-09-01, because the guess was that thumbnails already exist somewhere. They do, and the shape
of the answer is not quite the shape of the guess:

- **The thumbnails exist, as DPG textures already on the GPU.**
  `DPGChatController.get_inline_image_texture(filename)` reads the sidecar, `fit_contain`s it (which routes
  through the GPU Lanczos in `common.image.lanczos`), uploads a *static* texture, and caches it under the
  content-addressed filename — so an image referenced by several messages decodes once, and survives a view
  rebuild. That cache is keyed by exactly what the graph would look an attachment up by.
- **The size is wrong, and the cache cannot hold two.** The inline box is
  `chat_inline_image_h × chat_inline_image_w` = 220×480; a node is 180×52. So the graph wants its own,
  much smaller texture, and the cache key is the filename alone. Either it becomes `(filename, size)` or
  the graph keeps its own.
- **The upload cannot happen during a graph rebuild, and this is the constraint that shapes the design.**
  `get_inline_image_texture` calls `dpg.split_frame()` twice, deliberately — DPG defers the OpenGL upload
  and one wait is not enough. The panel rebuilds from its animator hook, and `animator.render_frame()` is
  called from the app's manual render loop, so a rebuild runs **on the render thread**, where `split_frame`
  deadlocks. Thumbnails therefore have to be prepared on a background task with the graph drawing a
  placeholder until they land — the pattern `cleanup_dialog` already uses for its own image grid.
- **Role glyphs are the cheap case by comparison**, and it is worth not conflating the two:
  `chat_controller.gui_role_icons` are registered once at class init, so drawing one needs no upload and no
  `split_frame`. Only per-attachment thumbnails need the background path.

**The shape, sketched 2026-09-01 and settled:** the thumbnail **straddles the node's right edge**, sized
to about the node's height, with part of it hanging outside the box. Several attachments **stack, each
overhanging further**, like a fanned deck.

That placement is what makes the third-indicator problem go away. It costs no label width, it needs none of
the box's interior, and it leaves the left edge to the role glyph — so the node is not asked to hold three
things in a space that fits one.

- **The image gets a border, in the graph's own line pen rather than a special grey.** Every other element
  here is outlined — nodes, gaps, pills, edges — so a picture without one is the single element that does
  not belong to the drawing, which is exactly why it reads as an unanchored texture rather than a tile.
  - **With stacking the border stops being decoration.** Two overlapping photographs of similar tone merge
    into one shape, and the count is the whole point of showing more than one. The line is what separates
    them.
- **Watch that the dark-mode remap does not touch the picture.** The renderer inverts lightness for pen
  colours, which is right for the border and would be wrong for the image; an `ImageShape` drawn through
  `dpg.draw_image` takes no pen colour, so it should be unaffected — but that is reasoning, not a
  measurement, and it is worth a look on the first render.
- **The stack is capped, because somebody will attach fifty files** (Juha, 2026-09-01). Show them all up
  to five or six; past that, the first two, an ellipsis, and the last two.
  - **With a wrapping algorithm's tolerance**: a count a little over the limit is shown whole rather than
    abbreviated, since replacing three thumbnails with two and an ellipsis saves nothing and costs the
    reader a count. Cut only when clearly over.
  - Worth noticing that this is the **sibling window again, one axis down** — some items, anchors at both
    ends, a gap standing for the run between. Whether the two share code or merely a shape is an
    implementation question; that they should not disagree about the *rule* is not.
- **The overhang still eats the gap between siblings**, which is `horizontal_spacing = 24` against a stack
  of up to six. The cap bounds it but does not size it: either the spacing grows when a row has
  attachments, or the per-thumbnail offset is small enough that six fit. Wants a row of them to look at.
- **No test would currently notice the resulting overlap.** `overlapping_pairs` compares *node* boxes, and
  a thumbnail lives outside its node's box exactly as a pill does. Whichever way the overhang is bounded,
  that test wants extending to shape extents, or the bound is unenforced. The overlap meant here is
  visual — two things drawn on top of each other — rather than anything the layout or the hit test would
  call an error.
- **The overhang is outside the node's hit box, and that decides one thing later.** `hit_test` asks
  `Node.is_inside`, which is the rectangle; the part of a thumbnail hanging past the right edge belongs to
  no node. Harmless while a thumbnail is a marker. The day clicking one should open the attachment, the
  clickable area and the drawn area are not the same shape, and something has to give.

**Found while driving it:**

- **`XDotWidget.on_click` does not pass what its docstring says it passes.** The docstring reads
  *"Receives (node_id, button)"* and this brief quoted it as the hook the feature needs. It actually passes
  `_describe_element(element)` — a human-readable string, from which no caller can recover which node was
  clicked. `raven-xdot-viewer` is the only other consumer and prints it to a status bar, so nothing had
  ever needed the identity and the wrong docstring cost nobody anything until now.
  - **Proposed fix, on the widget rather than the panel: pass the element.** A description is derivable
    from identity and not the reverse, and it also covers edges, which have no `internal_name` to pass
    instead. So `on_click(element, button)` / `on_hover(element_or_None)`, `describe_element` made public,
    and one line changed in the viewer.
  - This is the second load-bearing claim in the brief's "why it is cheaper than the record says" section
    to turn out false on contact. The other was the parser's image comment. Both were read rather than run.
- **The initial view is never fitted.** `refresh` pans to its anchor and never sets zoom, so the first
  build renders at zoom 1.0 with the root and the left of the fan off-screen. It wants `zoom_to_fit` on the
  first build and pan-only afterwards — the pan-only part being what keeps the picture still while a reply
  is arriving, so the two cannot simply be swapped.

### How much of a branch to show: four rules

Settled 2026-09-01, from a case where clicking back onto the current branch left the message below it
collapsed as a bare "…1 more" — one hidden box, announced by a box.

1. **A gap that hides fewer boxes than it costs should not be drawn.** A gap occupies a slot, so hiding
   one is a pure loss and hiding two saves nothing worth the content. Draw a gap from **three** hidden
   upward and inline anything below that. The same threshold for sibling gaps and depth gaps, since two
   rules here would disagree in front of the reader.
   - With the same tolerance the attachment stack gets: a count a little over the limit is shown whole
     rather than abbreviated. Cut when clearly over, not at the first opportunity.
2. **The drawn branch is a whole branch, not a stump.** This is the one that removes the reported symptom
   rather than papering over it. Focusing a node currently *truncates* the spine there, so whatever
   continues below is somebody's subtree gap by construction. The focus should select a branch and the
   view should draw it to its tip — `chatutil.descend_to_latest` already walks exactly that. Then "one
   message below the one you clicked" never arises, because the branch simply carries on.
3. **The depth window centres on the focus, not on the end of the branch.** Pinned prefix, gap, a window
   reaching both ways from the focus, gap, tip. With a floor under the downward budget, because one is
   plainly too few and the current arrangement can produce exactly one.
4. **When the focus is not HEAD, both want to be on screen.** That comparison — where I would be going
   against where I am — is the whole reason a preview exists. If the window cannot hold both, the focus
   wins, and the gap that swallowed HEAD should say that it did rather than letting HEAD vanish without
   comment.

**The current-branch-is-special question underneath rule 2**: at present chats are short enough to draw
whole, and rules 3 and 4 are what will matter when they are not. Neither should be written as though the
short case were the only one.

### The preview mark should be part of the picture, not the widget's highlight

Noticed 2026-09-01: clicking a node leaves it lit, and the light does not go away. It is the preview mark,
and it is deliberate — a second click commits, so what that click would act on has to be visible first.

**The defect is that it wears hover's clothes.** Both go through `HighlightState` and share one pair of
highlight colours, so a previewed node and a hovered node are indistinguishable — and a mark that looks
like hover *ought* to leave when the pointer does.

**So it moved into the built `Graph`** (done 2026-09-01): `ViewState.previewed_node_id`, and `_box_shapes`
draws it a ring of its own. Three things fell out of that, in ascending order of how annoying they were —
it cannot be mistaken for hover; it survives a rebuild without help, where `_apply_preview_highlight` had
to be re-applied after every one; and the widget's highlight state goes back to meaning hover and the
deliberate attention-grabbing flash, and nothing else.

**Three states, three marks, and none of them may be mistaken for another** — that is the rule the whole
thing turns on, because the failure was exactly a collision between two of them:

| state | mark | why that one |
|---|---|---|
| hovered | the widget's highlight, fading | belongs to the pointer, and goes when the pointer does |
| HEAD | a heavier box outline, plus the pill | where the reader *is*; the loudest thing in the picture |
| previewed | a **dotted ring outside** the box | where a second click *would* take them |

- **HEAD got heavier because of what the collision revealed.** A lit box reads as "this is the current
  one" — so while the preview was borrowing the highlight, the picture had two things claiming to be HEAD
  and the real one was saying so with a small pill in the margin. Emphasis on the box itself is what makes
  that unambiguous (Juha, 2026-09-01).
- **The ring is outside the box, not a change to it.** The box's own outline is already carrying
  information — solid or dashed, heavy for HEAD — and a selection has to be legible over every combination
  of those without overwriting any.
- **Dotted, because the selection is tentative** until the second click (Juha, 2026-09-01). A finer
  pattern than the gaps' dash, so the two broken lines do not read as one mark: they are saying related
  but different things, *this is not here* against *this is not settled*.

### Sequencing

Four reviewable commits: the builder and its tests; the panel and its placement, including the avatar pause
gate; the interactions; the tool badges and role icons.

### Getting back: the view has no history

Raised 2026-09-01, from a live run. Switch to *"How can I help you today?"* — two clicks — and the position
you were reading from is gone, with no convenient way back. Panning to it by hand across a wide level is
not a way back; it is a search.

Three things came out of it, and only the first is clearly a feature:

1. **A navigation history, back and forward.** `raven/visualizer/selection.py` is the precedent and the
   shape to copy: a module-local stack with a cursor, `commit_change_to_undo_history` / `undo` / `redo`.
   Here it would be panel-local and the unit on the stack is a `ViewState` — the focus, the sibling
   windows, the expanded turns.
   - **It is a history of *views*, not of HEAD**, and conflating the two is the obvious mistake waiting to
     be made. Going back should restore where you were *looking*; it should not un-commit a branch switch.
     Moving HEAD is already reversible by navigating, and an undo that silently moved it back would break
     the one promise this view makes — that only a deliberate act changes state.
   - Open: whether pan and zoom go on the stack too. The framing rules would otherwise re-derive a
     position, which is not the same as the one you left.
2. **A way back to the newest sibling without panning.** Possibly already there and worth *checking before
   building*: the sibling window keeps the first and last as anchors, so at the session level the last
   anchor is the newest chat and clicking it should do it. If that works, what is missing is not the
   operation but knowing it is there.
   - Note this is the operation the chat log's *switch to last sibling* button performs, so the two should
     agree. It is also distinct from the ±10 stepping that the gap-click bisection replaced: an anchor
     jump, not a step.
3. **Auxiliary buttons drawn near a node**, as the general answer to "this view needs verbs". Open, and
   the largest of the three — it wants deciding what the verbs are before deciding they are buttons.

### Panel sizing on a wide screen, and the constraint underneath it

Raised 2026-09-01 (Juha), for discussion. **Not demo-critical** — Researchers' Night runs at 1080p — but it
matters for daily use.

Widen the window towards 4K and the extra width goes largely to the chat log, where the graph would use it
better. The obstacle is that the graph shares its rect with the avatar, and **the avatar cannot simply grow:
its cost is O(pixels), so a bigger panel is a bigger per-frame bill for the same character.**

Three directions, none chosen:

- **Cap the chat log's width.** Worth considering on its own merits rather than as a way to feed the graph:
  a chat log 1800 px wide is *harder* to read than one at 900, the eye losing the start of the next line —
  which is why typography has a measure at all. Surplus width then goes right by default.
- **Let the two occupants of the rect be different sizes.** They are alternatives, never both on screen, so
  nothing forces the graph to inherit the avatar's dimensions. The split could move when the graph is
  shown.
- **Cap what the avatar *renders* and letterbox it — which is already how it works.** Checked 2026-09-01:
  `DPGAvatarRenderer` is handed a rect and positions the character bottom-centred in it, and the
  character's pixel size follows `avatar_config`'s `upscale`, not the rect. So panel size and avatar cost
  are already decoupled, and this direction costs nothing to adopt.
  - **The backdrop is handled too**, which was the part expected to need care.
    `configure_backdrop`'s contract: if the loaded image does not match the requested size, it is
    "rescaled with Lanczos on CPU, and then cropped to fit the aspect ratio" — so widening the panel
    re-crops the background rather than stretching it, which is what
    `raven-avatar-settings-editor` needs and gets from the same method.
  - Note its threading constraint if this is wired to a resize: `configure_backdrop` waits for a frame, so
    it cannot be called from the render thread. `app.py` already calls it from the debounced resize task.

The first two are compatible and could both apply. The third is the one that makes the *avatar* panel
growable, which the other two deliberately avoid needing.

## Where this stands, end of 2026-09-01

The view is built, wired into Librarian, and has been driven live several times. It is *usable* — the
remaining work is the two features below plus the polish the four rules describe.

**Built and landed:**

- `chatgraph.py` — the layout, pure and DPG-free, with its own test suite covering geometry and selection.
- `chatgraph_panel.py` — the widget, the toolbar, click routing, and rebuild-on-change by polling
  `chattree.Forest.generation` (added for this) plus HEAD.
- Wired into `app.py`: the **Chat graph** checkbox in the mode-toggle row, the panel sharing the avatar's
  rect, `chat_open_graph_button` deleted. `DPGLinearizedChatView.jump_to_node` is the preview's other half.
- Opening frames the branch; the crosshair returns to HEAD at 1:1 with a fading flash.
- Two-line labels at the interface's own font size, speaker names from the stored persona, pointer pills,
  the four gap kinds, HEAD emphasis and the preview ring.
- In the shared widget, so `raven-xdot-viewer` has them too: `on_click`/`on_hover` hand over the element
  rather than a caption, a drag no longer navigates, `flash_nodes`, a 1 s fade-out, a 1.25 wheel notch.
- Docs updated — a **Chat graph** section in `raven/librarian/README.md`, a line in the main README, and
  the roadmap item shrunk to what is genuinely left.

**Not built, in the order worth doing:**

1. **The four rules above** (whole branch rather than a stump; no gap below three hidden; the depth window
   centred on the focus; focus and HEAD both on screen). Pure `chatgraph` work, visible immediately.
2. **A navigation history**, per the section above — the view keeps none, so a branch switch strands the
   reader. Cheap, and the precedent is written.
3. **`ImageShape` in the widget.** Unblocks two things at once and they are not the same size: role glyphs
   need no texture upload (`chat_controller.gui_role_icons` are registered at class init), attachment
   thumbnails do — and that upload cannot happen during a rebuild, since a rebuild runs on the render
   thread where `split_frame` deadlocks.
4. **Attachment thumbnails**, to the design above: straddling the right edge, stacked and capped, bordered
   in the graph's line pen, prepared on a background task with a placeholder meanwhile.
5. **The avatar pause gate.** Never built. `TODO.md:662–663` and the road-mode notes above have the
   traps: the visibility term needs its own path rather than a term in the idle branch, and the switch
   keys on the idle detector rather than on whether the video happens to be running.
6. **Fragment search across the tree**, which was always v2 and is brief 14's companion.

**Open, needing a decision rather than work:** whether `raven-xdot-viewer` gets the actual-size button
too, for the consistency that now runs the other way.
