# Follow-up edits to the triage pass

Two small additions, arising after the decision document was handed over. Apply alongside it, or as a
follow-up commit if the pass has already moved past these items.

## 1. `Visualizer's importer should read the document database, not just .bib files` — add a use and a trigger

That item is gated `next`. It has a second motivation that is not recorded anywhere, and a trigger that
currently exists only in conversation — which is exactly the failure mode the *webfetch "approve denied
host" button relocates in brief 03* item demonstrates: a conditional deferral whose condition is real,
nothing watches for it, and the follow-up sits for months.

**Add to that item:**

> **Second use: mapping the backlog itself, for convergence detection.** `TODO_DEFERRED.md` items,
> `TODO.md` entries and the briefs together are ~160 documents of roughly abstract length — a corpus
> Visualizer is sized for, and one where the answers are already known well enough to check the map against.
> The purpose is not scheduling: a semantic map gives proximity, and the axis that decides scheduling is
> operational (which component, which session, what blocks what), which correlates with topic only loosely.
> The purpose is finding items that have converged without anyone noticing.
>
> That failure is well evidenced. The 2026-08-10 triage found the scriptable-scaffold design filed in two
> files under different names, with the poorer copy driving decisions; three Markdown items sharing one
> cause that none of them named; six independent pointers at an unwritten brief; and the ingest-crash and
> mid-run-recovery items arriving separately at the same missing primitive. Every one was found by a human
> noticing, several after two days of reading.
>
> **Two prerequisites**: brief 11's double-clustering bugfix, since the 2D labels are what colour the map;
> and this item, since the importer wants `.bib` and a fake bibliography is the alternative.
>
> **One technical caveat if it is built**: a 400-line brief and a 15-line item embed very differently — the
> brief's vector is a topic average that will not sit near any specific item, so cross-file convergence,
> the thing most worth finding, is what gets washed out. Embed briefs per section rather than whole. That
> also gives finer hits: "this item is about brief 16 §4" rather than "about brief 16".
>
> t-SNE (Visualizer's current projection) suits this better than UMAP would, since local neighbourhoods are
> the whole question and inter-cluster distance is not. Perplexity does real work at n≈160.
>
> **Trigger**: bump this item's priority when the backlog becomes unmanageable by reading. See the header
> note below, which is where that gets checked.

## 2. `TODO_DEFERRED.md` header — record the trigger where it will be read

Add to the file's header, after whatever describes the metadata line and the sections:

> **When this file stops being readable**, that is the trigger to bump *Visualizer's importer should read
> the document database* — the map of this backlog is a convergence-detection tool, and it needs the
> importer first. Recorded here rather than in the item because a trigger nobody meets is not a trigger;
> the recurring moment to ask is the triage step in the release procedure.

**Rationale for putting it in the header**, worth keeping in the commit message rather than the file: the
tool for finding things in the backlog cannot itself be gated on someone remembering to look for it in the
backlog.

## Note, not an edit: the operational map

Raised 2026-08-12 and worth recording somewhere, though not as a task yet. A semantic map does not capture
the operational axis — which component an item touches, which session it belongs to, what blocks what. With
the agent-scripting layer (`raven.librarian.agent`) that axis is now *derivable*: a per-item pass could
extract component, dependency and blocking relations from the item text, and the result is a graph rather
than a projection.

Two things that makes it, both of which the semantic map cannot be: a **dependency graph**, which is what
scheduling actually needs; and a use case for the per-document LLM pass on a corpus already in the
repository.

**And it needs no GUI work.** The pass emits DOT, GraphViz lays it out, `raven-xdot-viewer` opens the
result — a stopgap that already exists. So where the semantic map is gated on brief 11's clustering fix, the
importer item, and a fake bibliography, this route needs the pass and a prompt.

**DOT is also diffable**, which the projection is not: regenerate after a triage session and the diff shows
what changed in the dependency structure. That is precisely the drift the two-file duplication kept
producing, and a t-SNE layout cannot show it — the coordinates move for reasons unrelated to content.

**The prompt is the hard part, not the plumbing.** "What does this item depend on" is a judgement across 130
items where the answer is usually another item's *heading*, and the model has to resolve references like
"the sibling item above" or "brief 03" into actual node names. Which argues for doing it **after** the
triage pass lands: the metadata line then carries `Cluster:` and `See also:`, both of which are edges a
human has already drawn, giving the pass a scaffold rather than a blank page.

Not scheduling this. Recording it so the idea is not re-derived from scratch.
