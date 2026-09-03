# How many result nodes a tool round actually folds

Measured 2026-09-03, against the maintainer's live Librarian datastore, with **Qwen 3.6** as the model that
wrote it.

## The question

The chat graph folds a tool round into the message that asked for it. Brief 16 then designed a way to
unfold one — a gap box below the call, opened the way every other gap is opened. Before building it, the
question underneath: *is the folding buying anything at all?* A round that folds a single node cannot be
hidden behind a gap box without spending a box to save a box.

## The answer

**It buys about ten boxes, and only on one round in seven.**

```
54 tool rounds
    46 rounds fold 1 result node    85.2%
     6 rounds fold 2 result nodes   11.1%
     2 rounds fold 3 result nodes    3.7%

  boxes a round could hide, in total: 64
  if a gap is drawn only where it hides more than one box:
    gap boxes added:  8
    boxes hidden:     18
    net saving:       10 boxes (16% of what a round could hide)
```

Two readings, and the second is the one that decided it:

- **Unconditional folding is worse than no folding.** In 85% of rounds a gap box would hide exactly one
  box — a wash on screen, and worse than a wash for the reader, who now has to gesture to see what was
  hidden.
- **With a threshold — never gap a single node — the mechanism is invisible in those 85%**, so the cost is
  not per-round friction at all. What remains is *conceptual*: another gesture in the vocabulary, and a
  rule that has to be explained — *tool results are usually drawn, but folded at three or more, and then
  you open the gap and press Backspace to close it again*. For ten boxes. (Not an argument about help-card
  space, which is a transient state and due for rework; the learning cost is the durable half.)

**Multi-call is not hypothetical even now** — 8 rounds of 54 asked for two or three tools. It is simply not
yet frequent enough to pay for the machinery.

## Why it is a script

The distribution is a property of the *model's* habits, not of Raven, and the argument for building the
machinery is a prediction: models are being built for agentic work, so multi-call rounds should become
common. That is a reasonable expectation and not a measurement, and the gap between them is exactly what
this file closes — re-running it after a model change is a command rather than an afternoon.

**Re-measure when the model family changes**, and reconsider brief 16's design if the single-node case
stops dominating. It was measured on one model and one corpus; a model that emits genuinely parallel calls
would move the distribution, and none here has.

One thing that will *not* change under it: the tree shape. `scaffold._perform_and_store_tool_calls` chains
one `role="tool"` node per call itself, so a model asking for five tools at once produces five chained
nodes exactly as five sequential asks would. Only the frequency of long chains can move — which is why the
design in brief 16 stays correct for that future and only its cost/benefit changes.

## What it cost, once built (2026-09-03, same corpus)

The design was built the same day, and the threshold went in as `_MIN_HIDDEN_FOR_GAP` — the number the
sibling and depth gaps already use. So a round of one or two results is now **drawn** where previously
*every* round folded. That is a visible change to the common case, and `measure_round_cost.py` is what
priced it against real branches rather than against a synthetic one.

**A branch pays in one of two currencies, and never both**, which is the finding worth carrying:

- **A branch that fits the depth window pays in height.** Every round costs one row where it used to cost
  none — a drawn result is a row, and a folded round's band is also a row. The tallest gain on this corpus
  was **+49%**, on a branch of five single-result rounds.
- **A branch that overruns the window pays in conversation instead.** The height is capped whatever
  happens, so the drawn results spend the depth budget: `max_visible_depth` counts boxes down the branch,
  and a result is now one of them. Four branches here lost messages behind the depth gap, the worst going
  from 14 conversational messages on screen to 10.

**They land on different branches, and that is a trap for whoever reads this next.** A branch that grows
tallest is one with room to grow, so it is not the one being squeezed — sorting the table by height puts
the squeezed branches nowhere near the top, and reading only that table says the squeeze never happens.
It was read that way once, and reported that way, before the separate scan was added. The script now
prints both, and says so explicitly when nothing was squeezed.

Neither cost was judged unacceptable; the threshold is the dial if it ever is, and it is one constant.

## Scripts

| Script | What it answers |
|---|---|
| `measure_rounds.py` | The distribution above: how many `role="tool"` nodes chain under each assistant message that requested tools. Takes an optional path to a `chat.json`; defaults to the configured datastore. Reads the JSON directly, so it needs no Raven imports and can be pointed at a backup or an export. |
| `measure_round_cost.py` | What drawing a round's results costs a real branch — height, and conversational messages inside the depth window, each against the pre-2026-09-03 behaviour. Builds the picture with the real `chatgraph.build` rather than counting rows, layout being the kind of thing that is wrong when reasoned about; the old behaviour is simulated by substituting `_collapse_tool_rounds`. Same optional path argument. Needs Raven's imports, unlike its sibling. |

No data is committed: the datastore is a private chat history, and the script reads whichever one it is
pointed at.
