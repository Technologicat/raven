# Brief 14: search within the chat log

**v0.2.9 work.** Not admitted to v0.2.8: it is a feature, the freeze is explicit, and unlike the webfetch
attachment work it is not half of anything the release already ships. Written 2026-08-03, while the scrolling
work had the relevant machinery in view.

## The problem

A chat has no way to find anything in it. The only navigation is the scroll position and, as of the scrolling
work, Page Up/Down and Home/End. Long chats are the normal case rather than the exceptional one — a single
`webfetch` result can run to dozens of screens — so "what did it say about X earlier" currently means paging
by eye.

## The decision that makes this small: **the match unit is the message**

Not the character range. This mirrors the Visualizer's info panel, where the unit is the record, and it is
what keeps v1 cheap — because it sidesteps the one genuinely hard part.

**Why in-text highlighting is the hard part, checked rather than assumed.** The Visualizer highlights matches
by rebuilding the info panel *in full*, with highlight regexes applied at render time
(`entry_renderer.compile_search_highlight_regexes`, consumed during `_update_info_panel`). Its own comment
calls this Ship-of-Theseus: "the info panel is completely repopulated every time". That approach does not
transfer. The chat log is built incrementally, each message is a vendored-Markdown render, and re-rendering
the whole log on every keystroke of a search box is not a trade anyone would take.

Choosing the message as the unit removes the need for it entirely in v1.

## v1 — navigate to matching messages

**Almost all of this exists.** The scroll-and-flash operation is already built and already used for the
tool-call navigation links, in `DPGChatMessage._make_jump_to_tool_call`:

```python
self.parent_view.scroll_view(scroll_target_node_id=origin.node_id, user_initiated=True)
gui_animation.highlight_widget(...)
```

Search navigation is that same operation with a different rule for choosing the target. So v1 is:

1. **A search field** in the chat panel, with previous/next controls and an `[x/y]` counter. The Visualizer's
   info panel controls are the model for layout and for the enable/disable-by-position behaviour
   (`update_navigation_controls`).
2. **Find matching messages.** Walk `chat_controller.current_chat_history`, read each message's text with
   `chatutil.content_to_text`, match. Cheap, and no rendering involved. Decide: case sensitivity (the
   Visualizer compiles both a case-sensitive and a case-insensitive regex — see
   `compile_search_highlight_regexes`), and whether tool results and thinking traces are searchable.
3. **Jump.** `scroll_view(scroll_target_node_id=..., user_initiated=True)` plus `highlight_widget` on the
   message container.
4. **Status.** The counter, and a clear "no matches" state.

**Interaction with tail-following, which falls out correctly.** Searching scrolls away from the end, so
`should_follow_tail` stops following — which is right, since a reader who jumped to an old message does not
want to be yanked back by an arriving chunk. Getting back is the jump-to-latest pill's job, already planned
in the scrolling item. Nothing extra to build.

**A known trap, inherited from the reference implementation.** `info_panel.scroll_to_next_search_match`
carries a `try: ... except RuntimeError: pass` around a documented race: hammering the next-match button can
start the next update before the previous one has finished rendering, and there is no way to know when DPG
has finished updating viewport-coordinate item positions — one `split_frame` does not always suffice. Do not
copy the silencing without understanding it. The chat view's targets are message containers whose positions
come from the same kind of query, so the same race is reachable here.

## v2 — in-text highlighting, in completed messages only

Wanted, and deferred rather than dropped (Juha, 2026-08-03).

- **Highlight inside completed messages; leave the streaming one alone.** The streaming message is being
  re-rendered as it grows, which is exactly where a per-keystroke highlight pass would be most expensive and
  least useful. And the use case points the same way: search is for finding things in text that has already
  been written.
- **The message remains the unit that next/previous jumps to**, even once matches are highlighted
  in-text. Highlighting is presentation; navigation granularity is a separate decision, and message-level
  jumping is what a reader of a chat wants — the alternative walks the reader through six matches inside one
  long tool result.
- The open question is the mechanism: whether a completed message can be re-rendered with highlights
  cheaply enough to do on demand, or whether the vendored Markdown renderer needs a way to apply a highlight
  over already-rendered text. That is a renderer question, and it is the reason this is v2.

## What this brief must settle before implementation

1. Where the search field lives, and how it is opened and dismissed (a hotkey — `Ctrl+F` is the obvious
   one — and whether it is a persistent strip or an overlay).
2. Which content is searchable: message text certainly; tool results, thinking traces and attachment names
   are each a judgement call.
3. Case sensitivity, and whether to offer regex.
4. Whether matches survive a branch switch or a rebuild, or are recomputed.
5. The hammering race above.

## Settled 2026-09-17

**In-text highlighting is in v1 after all**, red and bold like the Visualizer's, re-rendered paragraph by paragraph
without the view moving. The mechanism was measured before it was built: `investigations/chat-search-highlight/`.
The pieces are `dpg_markdown`'s `highlight=` (matches marked after parsing, never spliced into the source),
`MarkdownText.rows` for predicting a paragraph's height, `gui_animation.WidgetSwap` for the swap, and decorations
that wait while their text is hidden. The streaming message is included: it is highlighted as it renders.

**Where the field lives** (Juha):

- **One search field for the whole app, not one per view.** Two text fields would be confusing, particularly
  with the composer as a third.
- **So the row spans the whole window, at the top**, and both panels shrink by its height. Left to right: the
  field, clear, and the two checkboxes below; then, at the right end of the chat half, the chat's `[x/y]` counter
  and previous/next. `Ctrl+F`, `Ctrl+Shift+F`, `F3`, `Shift+F3`.
- **Each view's navigation sits in the part of the row above that view.** The graph half stays empty until the
  graph's tree-wide search lands, when its own `[x/y]` and previous/next go there. (Revised the same afternoon:
  a row above the chat column only, with the graph's buttons in its own toolbar, was chosen first, and dropped
  because a shared field above one view reads as that view's.)

**What counts as a match** (Juha):

- **Tool results are searched like any message**, by default.
- **Thinking traces count, and a jump to a match inside a collapsed one opens it.**
- **Both are checkboxes in the search row**: whether thinking traces are searched, and whether tool messages are.

**Every search length highlights**, `e` included, if it is fast enough: a screenful costs roughly 150–250 ms at
the one-letter rate, and whether that feels acceptable is to be judged live (Juha).
