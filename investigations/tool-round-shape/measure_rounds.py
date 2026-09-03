#!/usr/bin/env python
"""How many result nodes does a tool round actually fold?

The chat graph collapses a tool round into the message that asked for it. Whether that is worth any
machinery depends entirely on this distribution: a round folding one node cannot be hidden behind a gap
box without spending a box to save a box.

Run it again whenever the model changes. That is the whole point of it being a script — the answer is a
property of the model's habits, not of Raven, and "models are becoming more agentic" is a prediction where
this is a measurement.

    python investigations/tool-round-shape/measure_rounds.py [path/to/chat.json]

Reads the datastore directly rather than through `chattree`, so it needs no Raven imports and can be
pointed at a copy, a backup, or someone else's export.
"""

import collections
import json
import pathlib
import sys


def main() -> int:
    if len(sys.argv) > 1:
        path = pathlib.Path(sys.argv[1]).expanduser()
    else:
        path = pathlib.Path.home() / ".config/raven/librarian/chat.json"
    if not path.exists():
        print(f"no datastore at {path}", file=sys.stderr)
        return 1

    # The on-disk shape is a flat mapping of node id -> node, with no wrapper key, and each node's `data`
    # maps revision -> payload. Assuming a "nodes" key yields zero nodes rather than an error.
    nodes = json.loads(path.read_text(encoding="utf-8"))

    def payload(node: dict) -> dict:
        data = node.get("data") or {}
        return (data.get(str(node.get("revision"))) or next(iter(data.values()), {})) or {}

    def role_of(node_id: str):
        return ((payload(nodes[node_id]).get("message") or {}).get("role")) if node_id in nodes else None

    chains = collections.Counter()
    for node in nodes.values():
        message = payload(node).get("message") or {}
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        # The agent loop chains one `role="tool"` node per call under the message that asked, so the depth
        # of that chain is the number of results the round would fold. It stops at the first non-tool
        # child, which is the next assistant message and the end of the round.
        depth, current = 0, node
        while True:
            tool_children = [child for child in current.get("children", []) if role_of(child) == "tool"]
            if not tool_children:
                break
            depth += 1
            current = nodes[tool_children[0]]
        chains[depth] += 1

    total = sum(chains.values())
    if not total:
        print(f"{path}: no tool rounds found")
        return 0

    print(f"{path}\n{total} tool rounds\n")
    for folded in sorted(chains):
        print(f"  {chains[folded]:>4} rounds fold {folded} result node(s)   {100 * chains[folded] / total:5.1f}%")

    hidden = sum(folded * count for folded, count in chains.items())
    gapped = sum(count for folded, count in chains.items() if folded >= 2)
    hidden_by_gaps = sum(folded * count for folded, count in chains.items() if folded >= 2)
    print(f"\n  boxes a round could hide, in total: {hidden}")
    print("\n  if a gap is drawn only where it hides more than one box:")
    print(f"    gap boxes added:  {gapped}")
    print(f"    boxes hidden:     {hidden_by_gaps}")
    print(f"    net saving:       {hidden_by_gaps - gapped} boxes "
          f"({100 * (hidden_by_gaps - gapped) / hidden:.0f}% of what a round could hide)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
