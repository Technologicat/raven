# What one chat-graph rebuild costs, and what it scales with

Measured 2026-09-01 and re-taken 2026-09-07, on `raven.librarian.chatgraph.build`. **Read the
re-take first** — the original figures were ten times off by the time anyone looked again, and why
they were is more useful than what they said.

The chat graph view rebuilds its whole `Graph` on every change to the forest — decided in brief 16 on the
grounds that it is simple and almost certainly fine, with a note to revisit if measurement said otherwise.
This is that measurement, taken because the sibling window's width was about to be picked by guesswork:
`siblings_each_side` had been set to 2 against the width of the panel, and the panel turned out to be the
wrong bound (the view pans, so the picture may spill past its edges). What is left bounding it is the cost
of a rebuild.

## The question

How large can the visible set be before a rebuild stops fitting in a frame? One frame at 60 fps is 16.7 ms,
and a rebuild happens on a tree change rather than per frame, so that is a generous bar rather than a tight
one.

## The answer

**Cost tracks the number of boxes drawn, not the size of the forest.** A twenty-thousand-node forest costs
about what a five-hundred-node one does, and the sibling window is what moves the figure:

```
                forest | each_side |      ms | boxes
   50 chats, 552 nodes |         2 |    0.66 |    24
                       |         5 |    0.94 |    36
                       |        20 |    3.38 |    96
 200 chats, 4202 nodes |         2 |    0.75 |    24
                       |         5 |    1.00 |    36
                       |        20 |    2.60 |    96
1000 chats, 21002 nodes|         3 |    1.14 |    28
                       |         5 |    1.36 |    36
                       |        20 |    2.96 |    96
```

Which makes sense: the builder reads only the nodes it is going to draw, plus one lineage walk. Nothing
scans the forest except `get_all_root_nodes`, and that is **0.37 ms at 20 502 nodes** — once per rebuild,
and not the term that decides anything. The deferred item proposing an index for it does not need acting on
for this view's sake.

**So the window is not speed-bound in any range worth using.** `siblings_each_side = 5` costs about 1 ms;
even 20 is a fifth of a frame. The setting was raised from 2 to 5 on the strength of this, and the ceiling
is legibility rather than time.

## Re-taken 2026-09-07, and the figures above had gone stale by a factor of ten

Prompted by the note at the bottom of this file — role glyphs were about to add a shape per box — and the
re-take found something larger than the thing it was checking.

**The September figures no longer held. The same script, same forests, reported 8.2 ms where it had
reported 0.66.** Not the glyphs, which were not in yet: `strip_markdown`, added to the label pipeline
after the original measurement. It renders the text to HTML and parses it back with BeautifulSoup,
building a fresh `markdown.Markdown` parser per call, and a profile put it at **0.20 s of a 0.29 s run —
68% of everything a rebuild did**. None of that is ours to make cheaper, so it is memoized instead; it is
a pure function of the message text, and a rebuild draws mostly the boxes the last one drew.

| `each_side` | before the memoize | after |
|---|---|---|
| 2 (the default was 5 when this was written; see below) | 8.2 ms | 1.6 ms |
| 5 | 8.8 | 2.6 |
| 20 | 26.3 | 8.8 |

**So the conclusion survives, and it had stopped being true in the meantime.** At 20 the rebuild was over
a 16.7 ms frame and nobody knew.

**The lesson is the one the note at the bottom was already reaching for**: this number is not a property of
the view, it is a property of whatever the label pipeline currently does, and something is added to that
pipeline every few weeks. A figure in a README does not notice.

**The fixture was also wrong, in a way that had been there from the start.** `measure_rebuild.py`'s
payloads carried no timestamp, so `chatutil.descend_to_latest` could not order siblings, and the builder
logged a warning per box and drew the branch only as far as the focus instead of on to its tip. It was
timing a picture the app never renders. Fixed here; the box counts in the tables above are from the old
fixture and the ones below are from the corrected one.

## Measuring the text costs almost nothing, so the labels are wrapped by measurement

The question that started the re-take. `chatgraph` can wrap a label to a measured width (`dpg.get_text_size`,
one call per candidate line per word) or estimate from an average glyph advance. The estimate is visibly
wrong in the direction that matters — the figure that keeps capitals inside the box cuts ordinary lowercase
prose short of the edge, so boxes come out with unused room on the right and messages wrap that would have
fitted on one line.

```
                forest | each_side | estimated | measured | boxes |  calls
   50 chats, 552 nodes |         2 |     1.58 |    2.18 |    33 |    234
                       |         5 |     4.83 |    2.92 |    51 |    336
                       |        20 |     8.32 |    7.40 |   141 |    846
 200 chats, 4202 nodes |         2 |     1.83 |    2.23 |    35 |    246
                       |         5 |     2.63 |    3.22 |    53 |    348
                       |        20 |     9.02 |   10.88 |   143 |    858
1000 chats, 21002 nodes|         2 |     2.21 |    2.57 |    35 |    246
                       |         5 |     3.00 |    3.57 |    53 |    348
                       |        20 |     9.81 |   11.15 |   143 |    858
```

**Under a millisecond at any sensible setting, and the run-to-run spread is the same size as the
difference** — two of the nine rows come out faster measured than estimated, which is noise rather than a
finding. So the exact answer is affordable and the estimate is kept only as the fallback for a caller with
no DPG: a test, or a build before the font atlas exists.

The alternative that was considered and is not needed: deriving a better average advance offline from the
real chat corpus. That would calibrate a path nobody looks at.

**`measure_measured_wrap.py` maps a window, and that is not incidental.** `dpg.get_text_size` answers only
once a font atlas exists, and an atlas is built by rendering a frame. A headless run would measure the
fallback twice and report no difference at all — a negative result from an instrument that was never
pointed at anything.

## What the first run got wrong, and why it is worth recording

The first attempt placed HEAD at the *end* of each chat, and reported that `siblings_each_side` made no
difference at all above 50 chats. That was true and useless: with a chat 20 messages deep, the depth window
(`max_visible_depth = 12`) elides everything between the root and the last eleven messages — **including
the session level**, the wide one the sibling window exists for. The measurement was of a picture that had
no fan in it.

The script now puts HEAD four messages into its chat so the session level is on screen. The accident is
worth keeping because it found a design question the brief had not asked: the wide level doubles as the
recent-chats list, and the depth window hides it exactly when the conversation is long enough to need it.
Whether that level should be pinned the way the root is, is open.

## Scripts

| Script | What it answers |
|---|---|
| `measure_rebuild.py` | Rebuild time against forest size and `siblings_each_side`, with HEAD positioned so the session level is inside the depth window |
| `measure_measured_wrap.py` | What wrapping labels by measured width costs against estimating them, and how many `get_text_size` calls a rebuild makes |

Run both from the repository root. `measure_rebuild.py` needs nothing outside the package — no server, no
models, no GUI. `measure_measured_wrap.py` needs DPG and **maps a window** for a few seconds, for the
reason above.

**Re-take these when the label pipeline changes**, which is the trigger that actually fires — the ten-fold
drift above came from one function added to it, not from the layout. "When the builder draws more per box"
was the trigger written here in September, and it did not catch the thing that mattered: the glyphs it was
warning about cost nothing measurable, and the Markdown strip that landed quietly in between cost
everything.
