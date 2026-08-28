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
| `check_dependency_versions.py` | Whether `pyproject.toml`, `pdm.lock` and `requirements-ci.txt` still agree about versions. Reports a locked or CI-pinned version the metadata forbids (exit 1), a package CI pins that `pyproject.toml` never declares, and floors older than what resolves — the last as `old -> new`, so raising them is a read-off. |
| `check_module_maps.py` | Whether the per-package `CLAUDE.md` module maps still describe their packages. Catches a size that drifted (they are written rounded to two significant figures, so 5% off means the code moved, not the rounding) and — the quieter one — a module absent from the map, where nothing looks wrong and a reader concludes it does not exist. Both had happened: the librarian map read 30–45% low for three weeks, and `indexer.py` was unlisted from the day it was added. |
| `check_exports.py` | Whether each module's `__all__` still lists what the module has, in the order the module has it. The convention is that it mirrors the file, so a reader can predict where a name sits; drift makes it predict nothing, and a name absent from it silently stops being re-exported. It reports and never rewrites — `__all__` carries comments whose prose is positional, so sorting it automatically would scramble the commentary while making the names right. A public definition *outside* `__all__` is a notice rather than a failure (a DPG callback bound by name is not API), and `--strict` promotes those. |
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

## Why `check_dependency_versions.py` exists

The same shape one level up. `check_ci_imports.py` asks whether CI can *import* what the tests need;
this asks whether the three lists agree on *which version* of it.

They come apart quietly, because each is edited for its own reasons and nothing reads two of them at
once. `pyproject.toml` states floors that get raised when a feature is needed. Dependabot moves the CI
pins on its own schedule. `pdm.lock` moves when someone re-locks, which on this project is rarely, since
it is gitignored and therefore absent from `git status`.

Run 2026-08-23, the day it was written, against the state of that morning:

- **NumPy**: the metadata said `<2.0` while CI installed 2.4.6 and 2.5.1 — and never saw the cap, because
  CI installs with `--no-deps`. Every test that had run for months ran against a major version the
  project declared it did not support.
- **`unpythonic`**: the lockfile held 2.2.0 against a floor of `>=2.3.0`, so a plain `pdm install` would
  have *downgraded* the package out from under code written for 2.3.
- **Pillow**: pinned in CI, imported at module level in four subsystems, declared nowhere — arriving
  transitively through torchvision and wordcloud, and fine right up until one of those dropped it.

None of the three was found by a failing test, and none would have been: each was invisible from inside
the list it was consistent with.

Floors are reported but never edited. They carry rationale — `dearpygui>=2.3` names the release that
made font atlas ranges automatic — and a rewrite that raises the number flattens the sentence explaining
it. The script's job is to say *which* floors lag; keeping the reason is a person's.

**A URL-pinned requirement gets its own section** because it is invisible to every other check: it names
one artifact, so resolution never touches it, no upgrade proposes anything for it, and nothing makes it
expire. Raven has one, `en_core_web_sm`, and it is the benign kind — spaCy publishes its models as
release assets rather than on PyPI, so a URL is the only way to get one. It still shows the hazard: the
pinned model declares `spacy>=3.8.0,<3.9.0`, which is satisfied today and will not be on the day spaCy
reaches 3.9, with nothing between here and there to mention it. The other kind — a fork or an unreleased
fix, pinned to work around upstream breakage — has the same property and much less reason to survive.

**The unbounded section is meant to stay non-empty**, which makes it the one worth explaining. A requirement
naming no version bets that every release ever made works. For the Emacs integration tooling — `flake8`,
`autopep8`, `importmagic`, `epc` — that bet is fine and permanent: none of them has any bearing on what
Raven does, and pinning them would be noise. So the section's steady state is those four, and the signal
is *a fifth row appearing*. Each row is tagged with the table it came from, because a `runtime` entry
there is a bet nobody placed deliberately.
