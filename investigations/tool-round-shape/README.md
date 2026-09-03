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

## Scripts

| Script | What it answers |
|---|---|
| `measure_rounds.py` | The distribution above: how many `role="tool"` nodes chain under each assistant message that requested tools. Takes an optional path to a `chat.json`; defaults to the configured datastore. Reads the JSON directly, so it needs no Raven imports and can be pointed at a backup or an export. |

No data is committed: the datastore is a private chat history, and the script reads whichever one it is
pointed at.
