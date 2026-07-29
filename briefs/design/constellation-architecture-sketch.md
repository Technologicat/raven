# Sketch: how Raven's parts talk to each other

**Status: a discussion sketch, not an implementation brief.** Opened 2026-07-29 (Juha) when two unrelated
features turned out to need the same missing thing. Sibling to `corpus-interrogation-sketch.md` and
`lab-assistant-hci-sketch.md`, both of which are blocked on it in places.

The question: **what is the architecture by which Raven's parts communicate, and where do the
division-of-concerns lines fall?** It has never needed answering, because there has only ever been one shape.

## The one shape there has been

Client → server, for GPU-bound inference. `raven-server` holds the models; the client apps call it over HTTP
(`raven.client.api`), or load the models in-process when it is absent (the `MaybeRemote` pattern). Everything
else lives on the client machine: the chat datastore, the documents folder, the app state, the GUI.

That has been enough because Raven ran on one machine. It is being stressed right now for a real reason and not
a hypothetical: the LLM backend has been run on a separate machine over 2026-07-28/29, partly for hardware
reasons (the eGPU is on the desk machine) and partly as early testing for the lab setup, where the LLM will
likely be served by dedicated hardware. So multi-machine is arriving as a fact, not a design exercise.

## What forced the question

Two features, arriving independently, both needing a part of Raven to receive something it did not ask for:

- **File upload from a phone** (`corpus-interrogation-sketch.md`) — a page served on the local network, so a
  paper on a phone can be pushed into the documents folder or into the current conversation.
- **Visualizer ↔ Librarian communication** — already on the prerequisite list for the interrogation track:
  hand a selection from the map to the chat, and get results back.

The first was initially answered with "then Librarian gains a server role". **That is rejected**, and the
reasons generalize well past uploads:

- **It puts an HTTP listener inside a client app.** Every client would eventually want one, and each is then a
  network service with its own lifecycle, port, and attack surface.
- **Which client?** Librarian and Visualizer both have claims, and choosing one makes the other a second-class
  citizen of its own constellation.
- **The constellation grows.** More apps will appear. An architecture where receiving anything means becoming a
  server scales by multiplication.
- **Not all members are equal, and this is worth making explicit.** Librarian and Visualizer are
  gravitationally bound — they share a corpus and will share selections. Cherrypick and the XDot viewer merely
  live in the same constellation; they happen to be shipped together and have no reason to talk to anything.
  An architecture should let a loosely-bound app opt out entirely rather than inherit machinery it will never
  use.

## Rejected: Warpinator

Considered for the upload case because it already runs on Linux, Windows and Android. Wrong shape, on two
counts. It delivers into **one configured folder**, so every transfer has the same destination — but the whole
design question for uploads is *scoping*, which destination this file is for. And in practice its receiver
tends to end up unreachable after running for a while, which is disqualifying for something meant to be
sitting there when you happen to want it.

Worth recording rather than silently omitting: the next person to look at this will have the same idea.

## A proposal, to be argued with

**The server is the constellation's switchboard; clients stay clients.** Uploads and inter-app messages land in
a server-side mailbox, and clients *pull* from it. No client ever listens on a port.

Why this looks right:

- **It preserves the invariant that is doing the work.** "Clients do not accept connections" survives intact.
  The server is already the thing with a stable address that others connect to; giving it a second role does
  not change any app's shape.
- **A stable address is what a QR code wants anyway.** Pointing the phone at the server beats pointing it at
  whichever laptop happens to be running Librarian today.
- **The push machinery exists.** `llmclient` already consumes SSE; a client holding an SSE connection to a
  mailbox is the same pattern pointed at a different endpoint, so "pull" need not mean polling.
- **Loosely-bound apps ignore it for free.** Cherrypick never opens a mailbox, and nothing about it changes.
- **It scales by addition.** A new app that wants to participate connects; one that does not, does not.

What it costs, and these are the parts to argue about:

- **Inter-app features would require the server**, which is currently optional — Visualizer is deployable
  standalone via `MaybeRemote`. Probably acceptable, since the features in question are inherently
  multi-machine, but it does mean the server stops being purely an inference accelerator and becomes
  infrastructure.
- **User documents would transit and briefly rest on the server.** That is a new data-at-rest location in a
  privacy-first tool, on a machine that may not be the user's own. Needs an explicit lifetime and deletion
  policy, not an implicit one.
- **It is still the first write surface**, wherever it lives. Capability tokens, size caps, extension filters,
  off by default — see the upload section of the interrogation sketch.
- **Is a mailbox even the right primitive?** Alternatives worth weighing before committing: an RPC surface
  (apps call each other's operations through the server), a shared-state model (the server holds the current
  selection and interested apps observe it), or a plain shared filesystem convention for the cases where both
  apps are on one machine anyway. The mailbox is the obvious first answer, which is exactly why it deserves
  scepticism.

## Open questions

1. **What is the minimum that solves both driving cases?** Uploads and a selection handoff are not obviously
   the same primitive, and building the general thing before either is a way to get neither.
2. **Does an app address another app, or a topic?** "Send this to Visualizer" and "here is the current
   selection, whoever cares" scale differently and imply different failure modes when the recipient is absent.
3. **What happens when the server is not running?** Both apps still work today; that should probably remain
   true, with inter-app features degrading rather than the apps failing.
4. **Where does discovery live?** How does Librarian learn that a Visualizer exists, on which machine, and
   whether it is the one the user means.
5. **Does any of this want to be MCP?** The lab-assistant track already names MCP, and an MCP server is a
   standard shape for "a thing that exposes operations to a client". Whether the constellation's internal
   plumbing should reuse it or stay separate is a real fork.

## What this is not

Not a distributed system, and not an excuse to build one. Every driving case is one user, one trusted local
network, a handful of processes. The measure of a good answer here is how little machinery it adds — and
whether an app that does not care can carry on not caring.
