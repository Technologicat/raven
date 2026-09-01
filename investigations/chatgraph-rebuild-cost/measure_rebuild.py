"""Same measurement, with HEAD placed where the wide level is actually on screen."""
import time
from raven.librarian import chatgraph
from raven.librarian.chattree import Forest


def payload(role, text):
    return {"message": {"role": role, "content": [{"type": "text", "text": text}]},
            "general_metadata": {"persona": None}}


def make_forest(n_chats, depth_per_chat, head_depth):
    forest = Forest()
    root = forest.create_node(payload("system", "the card"), parent_id=None)
    greeting = forest.create_node(payload("assistant", "hello!"), parent_id=root)
    head = None
    for c in range(n_chats):
        node = forest.create_node(payload("user", f"chat {c} opening message"), parent_id=greeting)
        for d in range(depth_per_chat):
            node = forest.create_node(payload("assistant" if d % 2 == 0 else "user",
                                              f"message {d} of a fairly ordinary length, as these go"),
                                      parent_id=node)
            if c == n_chats // 2 and d == head_depth:
                head = node
    return forest, head


def timeit(forest, head, each_side, repeats=20):
    config = chatgraph.LayoutConfig(siblings_each_side=each_side)
    state = chatgraph.ViewState(head_node_id=head)
    built = chatgraph.build(forest, state, config)
    t0 = time.perf_counter()
    for _ in range(repeats):
        chatgraph.build(forest, state, config)
    return (time.perf_counter() - t0) / repeats * 1e3, len(built.graph.nodes)


print("Budget: one frame at 60 fps is 16.7 ms. HEAD is placed 4 messages into its chat, so the")
print("session level is inside the depth window and the sibling count actually bites.\n")
print(f"{'forest':>22} | {'each_side':>9} | {'ms':>7} | {'boxes':>5}")
for n_chats, depth in [(50, 10), (200, 20), (1000, 20)]:
    forest, head = make_forest(n_chats, depth, head_depth=4)
    for each_side in (2, 3, 5, 10, 20):
        ms, boxes = timeit(forest, head, each_side)
        label = f"{n_chats} chats, {len(forest.nodes)} nodes" if each_side == 2 else ""
        print(f"{label:>22} | {each_side:>9} | {ms:>7.2f} | {boxes:>5}")
    print()
