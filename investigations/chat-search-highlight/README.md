# In-place search highlighting in the chat view

What it costs to re-render chat paragraphs with search matches highlighted, whether a highlight changes a
paragraph's size, and how a re-rendered paragraph can replace the old one without the view moving. The
groundwork for search v1 in Raven-librarian (`briefs/researchers-night/14_chat-search-brief.md`, brief 16
item 8). Measured 2026-09-17, DPG 2.3.1, font size 20, wrap width 850 px.

## The script

**`probe_highlight.py`** — one mapped window, three parts. Samples 200 paragraphs (one per non-blank line,
which is the unit `DPGChatMessage._render_text` builds one widget for) from the Librarian chat datastore, and
prints only numbers. The highlight is prototyped as the proposed renderer feature rather than as markup in the
source: `parser.parse` is wrapped so that red `MessageEntityFont` spans (plus `MessageEntityBold`, for the bold
variant) are appended over each match in the parsed, visible text.

It takes keyboard focus while it runs, and leaves with `os._exit`.

## Results

### The corpus

2268 paragraphs. `e` occurs in 86% of them, `the` in 47%, `search` in 8%.

### Cost per paragraph

| variant | parse, median | build, median | build, p90 | build, max |
|---|---|---|---|---|
| plain | 0.44 ms | 1.3 ms | 3.9 ms | 13 ms |
| colour, fragment `e` | 0.75 ms | 5.2 ms | 18 ms | 56 ms |
| bold, fragment `e` | 0.77 ms | 5.7 ms | 18 ms | 57 ms |
| colour, fragment `the` | 0.49 ms | 2.8 ms | 6.9 ms | 21 ms |
| bold, fragment `the` | 0.56 ms | 2.7 ms | 6.4 ms | 71 ms |

Parsing is cheap. Building (wrapping plus one DPG text item per run) is where the time goes, and a highlight
multiplies the number of runs. Where inside the build that time is spent was not profiled.

At the `e` rate, highlighting every matching paragraph of this datastore would take on the order of 15 s.
A screenful of 30 paragraphs is on the order of 150–250 ms.

### A highlight changes the layout, in colour as much as in bold

Against a negative control (the same paragraph built plain twice: 0 of 200 changed in either dimension):

| variant | paragraphs matched | height changed | width changed |
|---|---|---|---|
| colour `e` | 174 | 4 | 167 |
| bold `e` | 174 | 4 | 167 |
| colour `the` | 96 | 0 | 69 |
| bold `the` | 96 | 0 | 81 |

**Colour-only highlighting is not layout-neutral.** Measuring a string whole against measuring it in the
pieces a highlight splits it into gives 0.25–1.0 px extra per split (median 0.61). That is consistent with
each text item's width being rounded up separately. The mechanism inside ImGui was not checked. Those
fractions add up along a line, move line breaks (widths changed by up to ±80 px at an unchanged line count),
and occasionally add a line.

**So neither variant can skip scroll compensation**, and bold costs nothing extra on that axis. The bold
variant does bind a different font to the red runs (checked), so the similar numbers are not a
failure to apply it.

### Swapping a paragraph in a scrolled view

A per-frame recorder on the render thread watched a paragraph below the target, a paragraph above it, and
the scroll offset. Each paragraph in the view sits in a wrapper group, the same shape as the replacement, so
the swap does not change the item count.

| strategy | the view | decorations (code background, link underline) |
|---|---|---|
| build hidden in place, show + delete old at once, from a background thread (3 reps) | still | correct |
| the same, show + delete in one render-thread slot (3 reps) | still | correct |
| the same, with a 3-frame delay between build and swap | still | **zero-size and misplaced** |
| build visible in a 1 px clipped staging child window, `move_item` into place | still | correct size, **placed ~1100 px away** |

- **A hidden build gets working decorations only if it is shown before the renderer's worker runs.** The
  worker waits one frame, then measures. Show immediately and it measures a laid-out widget. Batch several
  builds before swapping and it measures a hidden one: the fault recorded in
  `investigations/dpg-markdown-decorations/`, reached by a different route.
- **A clipped but shown widget is laid out.** In the staging window the height and the decorations were both
  correct, so this is a way to learn a replacement's height before it is on screen.
- **`move_item` breaks decorations**, since they are placed with an absolute `pos` inside their window.

No straddled frame (both copies visible) was seen from the background thread, in three reps. That is weak
evidence, and the render-thread slot rules it out by construction.

### Keeping the view still when a paragraph above it changes height

A paragraph above the viewport growing from 26 to 52 px, with the watched on-screen paragraph at y = 252:

| compensation | watched paragraph's y, frame by frame |
|---|---|
| none | 252 → 278 (jumps, stays) |
| `set_y_scroll(+Δ)` in the same render-thread slot as the swap | 252 → 278 → 252 (one-frame jump) |
| `set_y_scroll(+Δ)` one frame *before* the swap | 252 throughout |

**A `set_y_scroll` takes effect one frame later than a tree change made at the same moment**, so the
correction has to go in a frame early to land together with the swap. That was measured in one
configuration, a single case, and the ordering it depends on is between two render-thread slots.

This needs Δ before the swap, which a hidden build cannot provide (a hidden widget measures `[0, 0]`) and a
staged build can, at the price of the decorations above.

### Incidental

The renderer's decoration sites (`text_attributes.py`) use `nonexistent_ok()` without `parent_gone_ok`. So
a paragraph deleted while its decorations are still queued prints a `[1011] Parent could not be deduced`
traceback from the worker. Two to five per run here, from the view being rebuilt between cases. Harmless, but
a search that swaps many paragraphs would produce them routinely.

## Open

- Whether the renderer's own wrap (`wrap_text_entity`, which fixes the line breaks before layout) predicts a
  paragraph's rendered height exactly. If it does, Δ is available for a hidden build and the staging window is
  unnecessary.
- The colour precedence when a match falls inside a span that already has a colour of its own, such as a URL.
- Offsets: the prototype uses Python string offsets, and the parser's disagree outside the BMP.
