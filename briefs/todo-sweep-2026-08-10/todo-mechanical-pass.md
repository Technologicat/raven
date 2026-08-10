# TODO hygiene: the mechanical pass

**Unnumbered, working-set** — it belongs in the sprint folder for the same reason the Monday checklist does,
and it deliberately does not take number 17, which brief 15 reserves for the per-document LLM pass.

**Timing: 0.2.8 shipped 2026-08-07; this runs now, and it runs *before* the human triage discussion**
(sequencing settled 2026-08-10). The two were originally imagined as concurrent, with `TODO.md` and
`TODO_DEFERRED.md` frozen so CC could not edit them from under the discussion. Running the sweep first
removes the conflict instead of working around it, and the discussion then starts from swept input rather
than from fiction.

The pass touches all 135 items in `TODO_DEFERRED.md` (3,928 lines as of 2026-08-10, up from 132 items and
3,487 lines on 08-04 — note that the body grew far faster than the count, so existing items deepened rather
than the pile merely lengthening). Anything filed while it runs conflicts with it. It must not become a
reason to *stop* filing — capture is the one part of this system that works, and the point of the whole
exercise is to keep it that way.

## What this is, and what it is not

This is the mechanical half of a two-part reorganization, agreed 2026-08-05. It **installs conventions** and
**populates what can be populated without judgment**. The judgment half — classifying the ~98 unassigned
items, and collapsing clusters into project heads — is a separate discussion pass, and is out of scope here.

**Three standing rules, in priority order:**

1. **Never delete an item that has not been explicitly ruled on.** Five have been (below). Everything else
   stays, however stale it looks. A verification sweep *reports*; it does not act.
2. **Never rewrite item prose.** Each item carries the reasoning that stops it needing re-derivation; that is
   the file's value and it is not yours to compress. Metadata goes *around* the prose.
3. **Flag, do not decide.** Anything ambiguous goes on a list for the discussion pass. A flagged item costs a
   line; a wrongly-resolved one costs the reasoning behind it.

**Two parts are now time-sensitive, for different reasons.**

- **Part C is the input to the triage discussion**, which follows immediately. Stale items that survive the
  sweep get ranked as though they were real, which is the specific waste this whole exercise exists to
  avoid.
- **Part A has the Researchers' Night deadline.** 2026-09-26 is itself a live-testing event and will produce
  a burst of new items. Those should arrive into a file that already carries the conventions, so they are
  filed correctly from the start rather than becoming another tranche of unclassified heads.

**Part B is the deferrable one.** If time runs short, ship C and A.

---

## Part A — install the conventions

### A1. The metadata line

One line, immediately under each `##` heading, then a blank line, then the existing prose untouched:

```markdown
## Markdown ATX headings (`### ...`) don't render in the chat view

*Cluster: markdown-renderer · Cost: ? · Gate: ? · Filed: 2026-07-13*

LLM replies that use ATX headings (`# `, `## `, `### `) show the literal `#` markers in
```

**Four required fields, always present**, using `?` when unknown — `?` is greppable and an omission is not,
and the discussion pass will work by grepping for what is still unfilled.

| Field | Values | Meaning |
|---|---|---|
| `Cluster` | a cluster name, or `—` | Which project owns it. `—` means *no cluster*, distinct from `?` meaning *unclassified*. |
| `Cost` | `S` / `M` / `L` / `?` | Deliberately coarse. Anything finer is fiction. |
| `Gate` | `RN2026`, `0.2.9`, `—`, `?` | What it blocks, if anything. |
| `Filed` | ISO date, or `?` | When it entered the file. |

**Two optional fields, present only when they have content**, appended in this order:

- `Bitten: N` — a tally, incremented in passing whenever the item costs something again. This is the
  self-promotion mechanism: an item whose cost *recurs* is categorically different from one that costs an
  afternoon once, and nothing in the file currently records the difference.
- `See also: <reference>` — cross-file references. This field earns its keep: on 2026-08-04 the same work was
  found filed in both `TODO.md` and `TODO_DEFERRED.md` under different names, with the poorer copy the one
  driving decisions.

### A2. Two new sections

Append at the end of the file, before any existing trailing material:

```markdown
## Declined

Items closed without doing. A reason is recorded so the decision stays made — an undocumented discard gets
re-added by the next person who has the same thought.

- **<heading text>** — <one-line reason>. (Declined <date>.)

## Waiting on upstream

Not tasks. There is no action available on our side; what is recorded is the trigger to look again.

- **<heading text>** — <what we are waiting for>; re-check <when or on what event>.
```

### A3. Stamp the pre-assigned clusters

`todo-cluster-map.json` assigns 34 of the 132 items to five clusters — `markdown-renderer` (10),
`document-ingestion` (9), `filedialog` (6), `hygiene-sweep` (6), `abnormal-exit` (3). The judgment was made
2026-08-05; stamping carries none.

Match on **exact heading text**, not line number. The map was generated against `dc1d9e5`; **re-verified
2026-08-10 against the current tree, 34/34 headings still match exactly**, so it stamps cleanly. If a heading
no longer matches, that is drift — **report it, do not guess**.

All other items get `Cluster: ?`. That includes the five filed since the map was built — the ingest-crash
item, silent indexing, ligature mojibake, the source-code tokenizer, and per-call tokenization overhead.
Two of them look like obvious cluster members from the heading alone; assigning on that basis is exactly the
judgment this pass does not make. Leave them `?` and let the discussion rule.

### A4. Apply the rulings of 2026-08-05

All six items named below were **verified present in the current file on 2026-08-10**, so none of these
rulings has been overtaken.

**Move to `## Declined`:**

- **Librarian chat input: make it multiline** — already done; `app.py` passes `multiline=True`.
- **`torch.compile` for the postprocessor** — measured, answer was no. This is a finding rather than a task,
  and it duplicates a completed investigation: `investigations/tha3-performance/` holds
  `debug_torch_compile.py` and `tha3-performance-audit.md`. Point the reason at the write-up.
- **Drop the Intel Mac / macOS 10.x install workaround** — support is being dropped; the one Mac user is on
  Apple Silicon.
- **Attachment + docs-DB: support office document formats** — the important formats now work. Spreadsheets
  are the remaining gap and already have their own item; point the reason at it.
- **webfetch "approve denied host" button relocation** — *conditional*. Brief 03 is closed and the relocation
  is believed done. **Verify against the code first**; decline only if confirmed, and report otherwise.

**Close against a brief** (verdict SUPERSEDED, not declined):

- **Ligature mojibake in PDF-extracted text** — `briefs/ligature-repair-brief.md` now owns it, including the
  argument for why `normalize` must not be wired into `docextract`. Point the deferred entry at the brief
  rather than deleting it, since the brief may be split (the `raven-fixbib` half is a Researchers' Night
  candidate; the indexer half is not).

**Move to `## Waiting on upstream`:**

- **pygame `pkg_resources` deprecation warning** — no action available on our side; re-check periodically
  until pygame resolves it.

**Keep as tasks, unchanged:**

- **CLAUDE.md: rephrase DPG pitfall #5 to avoid Claude thinking loops** — wants investigation time, not now.
- **Decide the public name** and **extract `raven.common` into "corvid"** — both retained. Each is a decision
  with no forcing function, which is why neither has moved; add a `Gate:` value naming the trigger (for the
  public name, before the first PyPI upload) rather than leaving them open-ended.

---

## Part B — populate mechanically

**`Filed:` dates.** Two sources, in order of reliability:

1. Many items already say so in prose — "Discovered during webfetch implementation (2026-06-03)",
   "Discovered during the brief-01 GUI override session (2026-06-04)". Lift the date.
2. Otherwise `git log -S '<distinctive heading fragment>' -- TODO_DEFERRED.md` recovers the commit that
   introduced the heading. Use the author date.

Where neither works, `?`. Do not estimate.

**`Cluster:`, `Cost:`, `Gate:`** — `?` for everything not covered by A3 or A4. These are the discussion
pass's input, and a wrong value is worse than an honest `?`.

**`Bitten:` and `See also:`** — omit unless the item's own prose already states one (some items reference a
sibling item or a brief explicitly; lift those into `See also`).

---

## Part C — the verification sweep

For each item, identify the concrete claim it rests on — a file, a symbol, an observed behaviour — and check
it against the current code. **Report only.** Write the results to a separate file; do not touch
`TODO_DEFERRED.md` on the strength of them.

Five verdicts:

- **CONFIRMED** — the claim still holds.
- **STALE** — the claim is false; the thing was fixed, landed, or removed. Cite the evidence (file, symbol,
  commit).
- **MOVED** — still true, but the code it names has relocated. Cite the new location.
- **SUPERSEDED** — still true, but a brief now owns it, so the deferred entry is a duplicate rather than a
  task. Cite the brief. Added 2026-08-10; the instance that prompted it is *Ligature mojibake in
  PDF-extracted text*, now owned by `briefs/ligature-repair-brief.md`. Expect more of these, since the brief
  set has been growing faster than the deferred file has been drained.
- **UNCHECKABLE** — the claim is about behaviour that needs a running app, a live backend, or a human eye.

**Write the results to `investigations/todo-sweep-2026-08-10.md`**, one row per item: heading, verdict,
evidence. The triage discussion consumes this file directly, so keep the heading text verbatim — it is the
join key against `TODO_DEFERRED.md` and against the cluster map.

Expect a meaningful STALE rate, and note *why* that is expected rather than embarrassing: several of these
items date from a period when the working instruction was never to get sidetracked under momentum, so things
were filed rather than fixed and some were fixed later without the item being closed. That instruction was
amended in 2026-08; the residue is what this sweep finds.

**This sweep will need running again** before the discussion pass, if that pass does not follow immediately.
That is expected and cheap — write it so it can be re-run rather than as a one-off.

---

## Out of scope

- **Collapsing clusters into project heads.** Merging several items into one is editorial writing, and the
  load-bearing sentence goes missing invisibly. By hand, with CC drafting rather than authoring.
- **Classifying the ~98 unassigned items.** Discussion pass, roughly five sessions of ~20 items.
- **Deciding anything on the discard list beyond the five ruled above.**
- **`TODO.md` itself.** It has the same problems and is a separate job; touching both at once makes one
  reviewable diff into two unreviewable ones.

## Adjacent, and worth deciding rather than assuming

Making triage a step in the release procedure is what gives it a forcing function — `TODO.md`'s own "goes
stale because nothing in the workflow makes anyone visit it" item is asking for exactly that.

**The home exists and is not in this repo**: `dotclaude/skills/release/SKILL.md`, fleet-wide. Its
`## Post-release` section is the natural slot, and already carries the matching rationale — the changelog
stub is opened immediately after tagging rather than at the start of the next release, so entries get written
while the context is fresh instead of reconstructed from `git log` months later. Triage takes the same
argument: nothing is in flight, and the release is the only recurring moment that reliably happens.

**Do not add it as part of this pass.** It is a different repo, and a mechanical checklist that accumulates
judgment steps gets run less — which would cost more than the step gains. Flag it for a separate decision,
including whether it belongs inline or in a separate skill that `release` points at.
