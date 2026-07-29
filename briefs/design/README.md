# Design sketches

Architecture and product discussion documents — **not implementation briefs**.

The distinction is the point of keeping them apart. A brief in `briefs/` describes work that has been decided
on and is being or has been done; it gets archived to `done/` when the work lands. A sketch here describes a
direction where the *workflow* is clear and the *mechanism* is not, and writing it as a brief would freeze
decisions nobody has made yet. A sketch graduates by producing a brief, not by becoming one.

Each carries a status line saying which parts are decided and which are open. Read that before treating
anything in one as settled.

## The sketches

- **[`corpus-interrogation-sketch.md`](corpus-interrogation-sketch.md)** — what the whole tool is *for*:
  screening tens of thousands of sources down to the ones worth reading, in a domain you are approaching
  fresh. Map, select, interrogate the selection, get a handle back. Also carries the constraints from the
  ECCOMAS 2026 talk, which are not negotiable by a later design decision, and the argument that the apps
  should be views of one corpus rather than two collections sharing a machine.

- **[`lab-assistant-hci-sketch.md`](lab-assistant-hci-sketch.md)** — the avatar as the interface. Hide the
  Librarian GUI except the character; anything reasonable by voice goes through it, and the full GUI stays on
  the console for what a screen and a pointer do better. Most of the machinery already exists, which is the
  surprising part.

- **[`constellation-architecture-sketch.md`](constellation-architecture-sketch.md)** — how Raven's parts talk
  to each other, and where the division-of-concerns lines fall. Opened because phone uploads and the
  Visualizer↔Librarian handoff turned out to need the same missing thing. The other two are blocked on it in
  places.

They are not independent: the first two converge (the interrogation flow *is* a view-control problem, and the
avatar's HCI is how you address a view), and both need the third before parts of them can be built.

## Related, elsewhere

Cross-cutting work items live in the usual places, not here — `TODO.md` for planned work, `TODO_DEFERRED.md`
for things noticed mid-task and set aside. Where a sketch depends on one, it names it.
