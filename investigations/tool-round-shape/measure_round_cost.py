#!/usr/bin/env python
"""What does drawing a tool round's results cost the picture?

Until 2026-09-03 the chat graph folded every tool round into the message that asked for it, drawing
nothing between the call and its answer. Now a round is folded only from `_MIN_HIDDEN_FOR_GAP` results up,
and a smaller one has its results drawn as ordinary boxes — which is what makes them reachable, and what
this measures the price of.

Two prices, and which one a branch pays turns on whether it fits the depth window:

- **Height**, for a branch that fits. A drawn result is a row; a folded round's gap box is a band, which
  is also a row. So every round now costs one row where it used to cost none, and the picture grows.
- **Conversation on screen**, for a branch that does not. There the height is capped by the window
  whatever happens, and the drawn results spend its budget instead — `max_visible_depth` counts boxes
  down the branch and a result is now one of them, so the messages at the far end go behind the depth gap.

**The two land on different branches, which is what makes reading one column misleading.** A branch that
grows tallest is one with room to grow, so it is by definition not the one being squeezed — sort by height
and the squeezed branches are nowhere near the top. Both are therefore reported: the table by height, and
a separate scan for any branch that lost a message.

Nor does the squeeze follow from a branch merely being longer than the budget; that was checked, and on
the corpus this was written against most of the over-budget branches lost nothing. It takes enough rounds
on the one branch, which is why this counts instead of reasoning about it.

Run it against the live datastore to see what real branches pay:

    python investigations/tool-round-shape/measure_round_cost.py [path/to/chat.json]

Measured with the real builder rather than by arithmetic over row counts. Layout is exactly the kind of
thing that is wrong when reasoned about and right when run.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from raven.librarian import chatgraph  # noqa: E402 -- after the path insert, by design
from raven.librarian import chattree  # noqa: E402 -- after the path insert, by design


def collapse_everything(datastore, lineage, expanded):
    """`_collapse_tool_rounds` as it behaved before 2026-09-03: every round folded, and no box for it.

    Returns no rounds at all, which is what suppresses the gap boxes — `build` draws one per folded round,
    and the old code had none to draw. `expanded` is ignored because nothing ever wrote to it, so the
    shipped behaviour was to fold unconditionally.
    """
    kept = []
    owner = None
    with datastore.lock:
        for node_id in lineage:
            role = (datastore.get_payload(node_id).get("message") or {}).get("role")
            if role == "tool" and owner is not None:
                continue
            owner = node_id if role == "assistant" else None
            kept.append(node_id)
    return kept, []


def measure(datastore, head, config, old: bool):
    """Build the picture around `head` and return `(height, conversational messages on screen)`.

    `old`: measure the pre-2026-09-03 behaviour instead of what the code does now.

    The second figure deliberately excludes the tool results themselves. What the depth budget is *for* is
    showing the conversation, and a result now spends one of its boxes — so counting every drawn box would
    report the cost as a gain, the tool nodes making the total go up while the conversation shrinks.
    """
    real = chatgraph._collapse_tool_rounds
    if old:
        chatgraph._collapse_tool_rounds = collapse_everything
    try:
        built = chatgraph.build(datastore, chatgraph.ViewState(head_node_id=head), config)
    finally:
        chatgraph._collapse_tool_rounds = real
    with datastore.lock:
        conversation = [node_id for node_id in built.spine
                        if node_id in built.refs
                        and (datastore.get_payload(node_id).get("message") or {}).get("role") != "tool"]
    return built.graph.height, len(conversation)


def main() -> int:
    path = (pathlib.Path(sys.argv[1]).expanduser() if len(sys.argv) > 1
            else pathlib.Path.home() / ".config/raven/librarian/chat.json")
    if not path.exists():
        print(f"no datastore at {path}", file=sys.stderr)
        return 1

    datastore = chattree.PersistentForest(path)
    config = chatgraph.LayoutConfig()

    def role_of(node_id):
        return (datastore.get_payload(node_id).get("message") or {}).get("role")

    # Every branch tip, and how many tool rounds its own lineage carries. A branch with none pays nothing
    # and is not what this is about.
    rows = []
    with datastore.lock:
        leaves = [node_id for node_id, node in datastore.nodes.items() if not node.get("children")]
        for leaf in leaves:
            try:
                lineage = datastore.linearize_up(leaf)
            except KeyError:
                continue
            runs, in_run = [], 0
            for node_id in lineage:
                if role_of(node_id) == "tool":
                    in_run += 1
                else:
                    if in_run:
                        runs.append(in_run)
                    in_run = 0
            if in_run:
                runs.append(in_run)
            if runs:
                rows.append((leaf, runs))

    if not rows:
        print(f"{path}: no branch carries a tool round; nothing to measure")
        return 0

    print(f"{path}")
    print(f"{len(leaves)} branches, {len(rows)} of them carrying a tool round\n")

    measured = []
    for leaf, runs in rows:
        new_h, new_spine = measure(datastore, leaf, config, old=False)
        old_h, old_spine = measure(datastore, leaf, config, old=True)
        measured.append((new_h - old_h, old_h, new_h, old_spine, new_spine, runs))

    measured.sort(reverse=True)
    print(f"  {'height':>16}   {'conversation on screen':>24}")
    print(f"  {'was':>7} {'now':>8}   {'was':>11} {'now':>12}   rounds (results each)")
    for delta, old_h, new_h, old_spine, new_spine, runs in measured[:12]:
        shape = "+".join(str(n) for n in runs)
        print(f"  {old_h:7.0f} {new_h:8.0f}   {old_spine:11} {new_spine:12}   "
              f"{len(runs):>2} ({shape}){'   <- worst' if delta == measured[0][0] else ''}")

    worst = measured[0]
    print(f"\n  tallest gain: {worst[1]:.0f} -> {worst[2]:.0f} graph units "
          f"({100 * worst[0] / worst[1]:+.0f}%)")

    # The other price, reported separately because a branch pays one or the other. Silence here is a
    # finding rather than an absence: it says every branch fits the depth window, so nothing was pushed
    # out of it -- and the height column above is then the whole cost.
    squeezed = [row for row in measured if row[4] < row[3]]
    if squeezed:
        lost = sum(row[3] - row[4] for row in squeezed)
        print(f"  {len(squeezed)} branch(es) lost conversation to the depth window, {lost} message(s) in all")
        for delta, old_h, new_h, old_conv, new_conv, runs in squeezed[:5]:
            print(f"    {old_conv} -> {new_conv} messages, {len(runs)} rounds")
    else:
        print("  no branch lost a conversational message: every one fits the depth window, so the")
        print("  height above is the whole of what this costs")

    print(f"\n  a row step is {config.node_h + config.vertical_spacing:.0f} units, "
          f"and the depth budget is {config.max_visible_depth} boxes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
