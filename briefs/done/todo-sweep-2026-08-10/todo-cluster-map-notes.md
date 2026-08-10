# Cluster pre-assignment for `TODO_DEFERRED.md`

`todo-cluster-map.json` assigns **34 of 132** items to five clusters, agreed 2026-08-05. Generated against
the live file at `dc1d9e5`, with every heading matched exactly once — the generator asserts on ambiguous or
missing matches, so the keys are known-good as of that commit. Line numbers are informational; match on the
heading text.

Stamping these carries no judgment: the judgment was made here. The remaining ~98 items still want the
per-item classification pass.

| Cluster | Items | What makes it one thing |
|---|---:|---|
| `markdown-renderer` | 10 | One vendored component, `DearPyGui_Markdown` |
| `document-ingestion` | 9 | The formats 2×2 — docs DB and attachments, read and attach |
| `filedialog` | 6 | One vendored component, one session |
| `hygiene-sweep` | 6 | Codebase-wide sweeps; lint rules where possible, one hygiene day for the rest |
| `abnormal-exit` | 3 | What is lost or leaked when the process does not exit cleanly |

## Judgment calls worth knowing about

**Two markdown items look like the same bug.** "`dpg_markdown` intermittently drops a single letter from
rendered text" and "Chat view drops a character mid-message (`What` renders as ` hat`)" describe a dropped
character in the same renderer from two vantage points. Both are tagged `markdown-renderer`; whether they
merge is a call for the collapse pass, not the stamping pass. Flagged rather than merged.

**`OS drag-and-drop of files into DPG apps` is tagged `filedialog`** though it is really a DPG-platform item.
The tag records *which brief it would land in* rather than which subsystem it belongs to — bundling it with
the dialog work is what makes both worth one session.

**Emoji and LaTeX are in `markdown-renderer` but are separately schedulable.** Cluster is about which project
owns an item, not when it ships. Brief 16's neighbours already treat both as deferrable within the renderer
project.

**Deliberately excluded from `hygiene-sweep`**, though they match on the word "audit":

- *AMD GPU (ROCm) support audit* — a platform-support investigation, not a codebase sweep.
- *Robust public API auditing tool* — builds a tool, so it is a feature; it also now overlaps brief 15's
  `__all__` and library-declaration work.
- *Audit and slim down project CLAUDE.md* — documentation.
- *Fleet audit: every hotkey discoverable in a tooltip + help card* — user-facing UX with real value, not
  hygiene.
- *Fleet-wide: shared two-phase DPG shutdown helper + audit* — assigned to `abnormal-exit` instead.

**Excluded from `document-ingestion`**, though they match on "attachment": browsing all attachments, the
`<datastore>.images/` directory name, the help card's coverage of attachments, exposing docs-DB source files
behind citations, and the fetched-page budgeting item. Those are attachment *management*, naming,
documentation and budgeting — different concerns from which formats can be read.

**`abnormal-exit` is three items, not four.** An earlier count included a `DearPyGui_Markdown` worker-thread
teardown item that does not exist as a heading in the current file.
