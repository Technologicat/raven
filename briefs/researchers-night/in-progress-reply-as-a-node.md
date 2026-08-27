# The in-progress reply becomes a node

**Decided 2026-08-27, and expected to be implemented immediately** rather than queued — the last outstanding
piece of *the turn-sequencing race with the abortable prefill* (`README.md` in this folder, band 2, item 9),
and groundwork the graph view will stand on.

## The problem this closes

An in-progress reply is not a node, so **nothing that reasons over the tree can see it**. Every hole found
while testing that work is a place where the tree is the source of truth and the reply is not in it:

- `chatutil.descend_to_latest` follows the most recent *stored* child, so navigating back to a branch whose
  reply is still being written walks straight past the turn's insertion point and lands on the previous
  stored sibling.
- The sibling counter cannot count it, so the reply the user is watching is not among the alternatives.
- `on_done`'s ownership test compares HEAD against the turn's parent, and after navigating back HEAD is a
  deep leaf, so the finished reply never becomes the active sibling. It appears only after the user
  navigates once more, by which time it *is* a node and the descent can find it.

Patching each walker was the alternative (`descend_to_latest(stop_at=...)`, cost S). Rejected because it
teaches one walker while the next one added has to remember — and because the structural fix **deletes**
the publish/reattach machinery rather than adding to it, which is the sign it is the right shape.

## The pivot

Today the assistant node is created *after* `invoke` returns. Instead: **create it before, and fill it in as
the text arrives.** The streaming widget then stops being a floating thing the view has to be told about,
and becomes the rendering of a node like any other — recreated by any rebuild, with the right content,
because the content is in the tree.

Writing partial text costs nothing on disk: `PersistentForest` saves at `atexit` only, so this is an
in-memory update until the app exits.

## What changes

**`chattree`** — `Forest.overwrite_active_revision`, replacing the current revision's payload in place.
`add_revision` is wrong here: revisions are a user-facing edit history, and a reply that streamed would fill
it with hundreds of entries. Foundation code, so it wants the usual care: under the lock, revision identity
unchanged — and see *Settled* below for the warning it has to carry.

**`scaffold.ai_turn`** — create the assistant node before `invoke` with an empty message and a status saying
it is incomplete; update its payload as paragraphs arrive; write the final payload on completion. The
backend-error path then *updates the node it already made* instead of creating a synthetic one, which also
removes an asymmetry: today an error message is built by a different code path from a real reply.

The partial writes happen in the scaffold, by wrapping the caller's `on_llm_progress` — so the frontends
need no new hook for it.

**`on_llm_start` gains the node id.** It currently takes no arguments. Consumers: `chat_controller`,
`minichat`, and the tests that assert the callback bundle.

**`chat_controller`** — the streaming widget binds to a node id. `build()` renders the tail node as a
streaming message when it is incomplete and a turn is live; `on_llm_progress` *looks up* the widget by node
id rather than holding a reference, so a rebuild that replaced it is transparent.

Deleted: `streaming_message`, `streaming_message_head`, `publish_streaming_message`,
`DPGStreamingChatMessage.reattach`, and the `turn_owns_the_view` checks that exist only to protect the
floating widget.

**The navigation holes close without being addressed.** The node is the newest child, so `descend_to_latest`
lands on it; the sibling counter includes it; `on_done` becomes "re-render this node", with HEAD already
there.

## Settled

- **Partial payload written per paragraph**, not per chunk — the paragraph is the renderer's unit already.
- **A stub left by a clean exit mid-turn is acceptable** (Juha): no worse than the backend-error message,
  which is stored by design. A crash loses the session either way — Librarian has no periodic autosave, and
  that is its own filed item.
- **Sibling churn during a turn is acceptable** (Juha): the count going up when the reply starts is honest.

## Settled before building (Juha, 2026-08-27)

**Overwriting a revision is allowed here, and the docstring has to say why it is not allowed elsewhere.**
Revisions are immutable by design, and the reason is not "bytes never change" — it is that a thread does not
record which revision of each ancestor it was written against (and arguably should not, since a typo fix is
meant to reach every thread). So the real guarantee is **no semantic change that would render an existing
thread invalid**, and a reply becoming itself as it streams is not one: the node has no descendants, so
nothing has been written against its *content* — the parent has listed it as a child since it was created,
which the guarantee does not speak to — and the successive payloads are one message arriving rather than
competing versions of it. Recording each chunk as a revision would bury the edit history the user is meant
to read. `Forest.overwrite_active_revision`
carries this argument in full, because a caller who has not read it should not be using it.

**"Incomplete" is three states, not one**, and only one of them is a runtime concern:

| State | What it means | What to render |
|---|---|---|
| still streaming | a turn is live and writing this node | the streaming message |
| cancelled by the user | arguably *complete* — the partial reply was kept, which is Cancel's promise | whatever is there |
| the app died mid-turn | only reachable at load time | whatever is there |

The runtime and load-time questions never need the same answer, and neither needs stored state beyond the
marker, because **a live turn cannot span a restart**: any node still marked incomplete when the datastore
is read back was interrupted, by definition.

*Considered and not done*: a `[Cancelled by user]` footer paragraph on the second case. It would have to be
stripped by the continue machinery, and that complication buys little — the message reads the same to a
person either way.

**`continue_` stays on `add_revision`.** Which is what keeps `overwrite_active_revision` down to one caller,
so it can be scoped to the streaming case rather than designed as a general operation.

*Noted, not acted on*: the multiversal-shaped answer for continuing is arguably a **sibling** rather than a
revision — the original message is one thing that historically happened, and the continued variant is an
alternative timeline. That is a separate question from this work, and left open.

**The class hierarchy stays.** The fork between a message being streamed and a stored one is genuine — live
paragraph updates and a thinking cloud against a full button row and sibling navigation — and the base class
exists to hold what they share. Worth stating the converse, since it is what makes the decision non-obvious:
if the streaming class were folded away there would be no reason for a hierarchy at all, the base becoming
the only class. The fork remaining is what keeps it.

## How it gets verified

Suite, then live: reroll → navigate away → back (streaming resumes, with the text so far); completion while
away (appears as the active sibling on return); Cancel mid-stream (partial reply kept); backend error (the
node is updated, not a second node created) — that last one via
`investigations/backend-fault-injection/faultproxy.py`, which is what found the orphaned-widget bug.
