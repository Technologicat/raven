# What one chat-graph rebuild costs, and what it scales with

Measured 2026-09-01, on `raven.librarian.chatgraph.build`.

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

Run it from the repository root: `python investigations/chatgraph-rebuild-cost/measure_rebuild.py`. It
needs nothing outside the package — no server, no models, no GUI.

**Re-take it when the builder draws more per box.** Role icons are not in it yet, and they add a shape to
every node; the figures above are for outline, label and pills.
