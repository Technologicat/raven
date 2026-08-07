# Brief 16: chat graph view

> **Line numbers are as of 2026-08-04 and want verifying.** They were read from a shallow clone that was
> already a couple of hours behind a moving tree, and the files most often cited here (`app.py`,
> `chat_controller.py`) were among those being committed to that day. Treat every `file.py:NNN` below as a
> pointer to a thing that exists, not as a coordinate.

**Researchers' Night work (2026-09-26), and it takes precedence over `atmospheric-dust.md`** (decided
2026-08-05: the graph adds more value; dust lands only if time remains after this). Written 2026-08-04. The
feature has been recorded in four places since roughly a year ago; what changes now is that the expensive
part is built, and that the demo gives it a purpose beyond navigation.

The ordering is safe in the direction it needs to be: `atmospheric-dust.md:30–35` takes its priority-band
scheme *from* `crt-display.md` §0, so the dependency runs dust → crt. Nothing points back at dust, and
dropping it leaves both `crt` and this brief intact — which is what makes it a sound last item rather than an
overcommitted one.

**Closes on landing**: `TODO.md:480` ("Nonlinear chat view / chat graph editor"), the placeholder button at
`app.py:1083`, and `raven/librarian/README.md:377` / `:588` / `:590`.

(An earlier draft also listed one of the help card's two "This is a tech demo" disclaimers, on the grounds
that `TODO_DEFERRED.md:1183` called it still true in substance — old chats stored but unreachable. Both
disclaimers were removed 2026-08-04. The substance still holds until this view exists: the datastore keeps
every branch and the GUI can reach none of them. It just no longer has a sentence to correct.)

## Why it is cheaper than the record says

Two entries are stale in the same direction:

- **`TODO.md:480` says Librarian must generate `.xdot` code.** It does not. `XDotWidget.set_graph(graph:
  Graph)` takes a `Graph` directly, and `Node(x, y, w, h, ...)` / `Edge(src, dst, points, shapes)` take
  explicit geometry. The graph is built in memory. No text format, no re-parse, and no DOT escaping applied
  to chat content.
- **`raven/librarian/README.md:590` says a suitable DPG widget is missing and xdottir should be ported.**
  That port happened; `raven/common/gui/xdotwidget/` is the result, and `raven-xdot-viewer` is built on it.

The widget already exposes every hook this needs: `on_click(node_id, button)`, `on_hover`, `pan_to_node`,
`set_highlighted_nodes`, `search`, and `text_compaction_callback` for the label-does-not-fit case.

**Framing worth keeping (Juha, 2026-08-04):** Librarian's chat view was always going to be the first real
production client of the widget. `raven-xdot-viewer` is the bonus, not the target.

## Step zero — `set_graph` has no callers and no tests

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

`TODO.md:480` frames this as navigation — "jump to chat node by ID". The demo wants explanation: making
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
glyph), and `chat_controller.py:599` records the reservation deliberately: `ICON_GEAR`, the universal
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

**That wide level is also the recent-chats list.** `raven/librarian/README.md:588` wants one, and notes that
"recent chat" is ill-defined in a nonlinear format, guessing that the user's first message is a good enough
splitting point. The fan-out shape says the guess is right, and this view already renders that level. So the
level deserves special treatment — recency ordering, possibly timestamps — and this brief closes that item
too.

**Depth limiting** is recorded as a hard constraint (`TODO.md:480`: the full tree will not render at
interactive FPS). For the demo case it is less pressing than that suggests, since walk-up conversations are
short. The accumulated forest is what is large, and scoping to the tree containing HEAD plus windowed
siblings is what handles it.

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
afresh. The tool role's is `ICON_GEARS`, and `chat_controller.py:599–601` documents why it is not `ICON_GEAR`
— that one is reserved for the settings dialog. Whatever the other roles use in the chat log, carry the same
symbols across, so that the two views name the same things the same way.

**Labels** come from `chatutil.content_to_text` (brief 14 uses it for the same reason), compacted by the
widget.

### The role glyphs are the PNGs, and the widget cannot draw images yet

The marks the chat log uses for turns are `raven/icons/system.png`, `user.png`, `ai.png` and `tool.png` —
image assets, not font glyphs. (`ICON_GEARS` is a different thing: it marks a tool call *inside* a message,
and the collapsed tool-count badge should reuse it, per above.)

**`XDotWidget` cannot render images — but not for the reason recorded.** `parser.py:623–625` parses xdot
image shapes and skips them with a warning, on the stated grounds that DPG drawlists lack image support.
**That premise is disproven by code in this repo**: `chat_controller.py:375` calls `dpg.draw_image` into a
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

### Resampling: draw at native size and there is nothing to do

DPG samples nearest-neighbour, which is why the tree carries a custom GPU Lanczos scaler. But the shipped
role icons are **64×64 already, high-quality downscales meant for 1:1 use** (`aria1_icon.png` is 64×66;
originals live in `raven/icons/00_workfiles/cropped/`, e.g. 719×722 for the generic AI icon).

**So constant screen size at 64 px is the zero-work path**: draw the existing textures 1:1 and there is no
resampling, no mip chain, and no coupling to `Viewport`'s zoom at all. Combined with the legibility argument
— a fixed-size icon stays readable when zoomed out to take in a wide sibling fan — this is the recommended
answer, and the remaining alternatives cost real work:

- **A different fixed size** wants a new *asset*, generated offline from the originals, rather than a runtime
  Lanczos pass. That is how the 64 px files were produced in the first place, and an asset is cheaper and
  more inspectable than a startup GPU step.
- **Scaling with the node** needs the full apparatus: `mipchain`
  (`raven/common/image/lanczos.py`) plus the selection rule in `mip_scale_for_zoom`
  (`raven/cherrypick/preload.py`). Note the trap if this is chosen — `mipchain`'s `min_size` defaults to 64,
  tuned for Cherrypick's photographs, so a 64 px icon produces a chain of length one and the aliasing
  survives the machinery.

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
`TODO_DEFERRED.md:3060`ff assumed a panel.

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
- **A forest view across all roots.** The windowed wide level covers what the demo needs.

## What this brief must settle before implementation

1. **Panel placement** — DPG window, or a panel in the main layout — against the road-mode item's
   swappable-panel framing.
2. **The commit gesture**, right-click being unavailable. Prototype rather than argue.
3. **Whether tool-node expansion is per-node or a global toggle.** The collapse itself is decided; only the
   control is open.
4. **Sibling-window controls**: whether click-to-recentre suffices or jump-by-several controls are needed,
   and how many siblings the window holds either side.
5. **Ordering at the wide level**, given that it doubles as the recent-chats list.
6. **Whether an icon scales with its node or holds a constant screen size.** See the resampling note above:
   the two answers differ by an order of magnitude in cost, and constant size is also likely what reads
   better when zoomed out to take in a wide sibling fan.
