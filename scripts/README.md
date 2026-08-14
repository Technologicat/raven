# Repository maintenance scripts

Tooling that operates on *this repository* — its CI config, its docs, its own consistency. Run by a
maintainer, never by a user, and not part of the installed package.

**This is not `raven/tools/`.** That package holds shipped console scripts (`raven-check-cuda`,
`raven-qoi2png`, …): user-facing utilities that happen to be small. Everything here is the opposite — it has
no users, only maintainers, and it would be noise in a wheel.

It is also not `investigations/`, which keeps a measurement together with the apparatus that produced it, and
not `briefs/`, which is prose about work to be done. A checker that runs indefinitely, answering the same
question after every change, is neither: it measures nothing once and describes nothing. That is what this
directory is for.

| Script | What it answers |
|---|---|
| `check_ci_imports.py` | Which test modules would fail to *collect* in CI, whose dependency list is hand-picked and hand-maintained. Walks the unguarded tests and their first-party imports, and reports any module-level third-party import that `requirements-ci.txt` and the workflows' inline `pip install` lines do not provide. |
| `check_todo_structure.py` | Whether `TODO_DEFERRED.md` still parses as a list of items. Catches an item whose `##` heading an edit removed — its body then reads as part of the item above and it vanishes from every future scan of the headings, silently — plus duplicate titles (items are cited by title), missing metadata fields, and headings without a blank line before them. |

**One checker deliberately lives elsewhere**, and is indexed here so that this table stays the place to look:
`.claude/skills/dpg/check_router.py` verifies that the `dpg` skill's router still names sections that exist in
`dpg-notes.md`. It sits beside the skill rather than here because the three files — notes, router, checker —
are one unit, and the argument for co-locating the first two applies to the third: if they ever move to
another repository, they move together and keep working. A checker in `scripts/` would be left behind
pointing at nothing.

## Why `check_ci_imports.py` exists

CI does not run `pdm install`. It installs a hand-picked subset, because Raven's full tree is multi-gigabyte
and a matrix would install it once per entry on every push (the reasoning is in `requirements-ci.txt`'s own
header). The cost of that choice is a second list with nothing enforcing the overlap: **a test that imports
something CI lacks passes locally and fails only on push.**

That is not hypothetical. Dropping an `importorskip` from `test_llmclient.py` made it collect in CI for the
first time, and `llmclient` imports `sseclient` at module level — a real dependency, declared in
`pyproject.toml`, absent from the CI list because until then nothing in CI had needed it. Red main, on a
docs-only commit, from a change two commits back.

Run it before pushing anything that adds an import or removes an `importorskip`:

```bash
python scripts/check_ci_imports.py
```

**Two things it knows that a naive version does not**, both learned by being wrong first:

- **`importorskip` guards are honoured, including in `conftest.py`.** `raven/client/tests/` guards its whole
  directory from the conftest, deliberately, so no individual file needs to. A checker reading only test
  files calls that directory broken while CI is green.
- **Only module-level imports count.** A function-local import is the standard way to let a heavy or optional
  dependency degrade gracefully, and flagging those buries the real finding.

The inline-pinned torch trio is read out of the workflow files rather than copied here, because a second
hand-maintained list is precisely what this script exists to catch.
