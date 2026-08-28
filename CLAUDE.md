# Raven - CLAUDE.md

## Project Overview
Local research assistant constellation. Privacy-first, 100% local.

**Components:**
- **Visualizer** (`raven/visualizer/`): BibTeX topic analysis, semantic clustering, keyword extraction. The original app. See `raven/visualizer/CLAUDE.md` for architecture.
- **Librarian** (`raven/librarian/`): LLM chat frontend with tree-structured branching history, hybrid RAG, tool-calling, message attachments (images on a VLM + text/PDF documents on any model, stored as content-addressed sidecars), avatar integration. See `raven/librarian/CLAUDE.md` for architecture.
- **Server** (`raven/server/`): Web API for GPU-bound ML models. Primary inference endpoint.
- **Client** (`raven/client/`): Python bindings for Server API.
- **Avatar** (`raven/avatar/`): AI-animated anime character (THA3 engine, lipsync, cel animations). Some avatar-related code (video postprocessor, colorspace) lives in Common for licensing reasons.
- **Common** (`raven/common/`): Shared utilities (video processing, audio, GUI widgets, networking, document text extraction — `docextract`: plain text + PDF via pypdf, the single extraction backend for both RAG ingestion and chat attachments). Mostly BSD, but **not uniformly** — `common/gui/xdotwidget` is LGPL-3.0-or-later (derived from xdottir, and through it from Jose Fonseca's `xdot.py`) and `common/video/upscaler` is MIT (matching Anime4K). Server and Avatar pose editor are AGPL-3.0. The full picture is in `TODO_DEFERRED.md`; `raven/vendor/README.md` covers the adopted tree.
- **Papers** (`raven/papers/`): Academic paper tools — arXiv search/download, bibliography converters (WoS, CSV, PDF, BibTeX burst).
- **Tools** (`raven/tools/`): Miscellaneous CLI utilities (CUDA check, audio device listing, image format conversion, dehyphenation).

## Who develops Raven

Raven is what agentic development lets one researcher build. The maintainer directs and reviews; Claude
writes a large share of the code, the tests and the notes. Several conventions in this file look arbitrary
until you know that, so they are worth stating together:

- **Review is the binding constraint, not implementation.** Code arrives faster than it can be read, so a
  change that never renders as a diff has skipped the only review there is. Hence commits kept small and
  separately reviewable, and edits made so that they show up as diffs rather than as in-place rewrites
  nobody sees.
- **Sessions are bounded; the repository is not.** A decision that exists only in conversation is lost at
  the next context boundary — hence `briefs/`, `investigations/`, and writing findings to a file *before*
  a session ends rather than after.
- **Every reader is an outside reader**, the next session included. Documentation that assumes shared
  context has none.

## Where the non-source material lives

Four trees, sorted by what a document *is* rather than what it is about. Each has its own README; this is the
index.

- **`briefs/`** — prose. `design/` for sketches (direction clear, mechanism not), **one folder per sprint**,
  named for its scope (currently `librarian-extension/` and `researchers-night/`), `done/` for closed briefs
  and closed sprints, `reference/` for **durable knowledge** — documents that describe how something *is*
  rather than what to build, and so are consulted rather than finished (the EU AI Act summary, the DPG keycode
  table, an archived style snapshot). Who wrote them is not the test: **every document in
  `reference/` is ours**, written or compiled here. Our *measurements* go to `investigations/`, with the
  apparatus that produced them. A sprint folder is a working set: it holds its own `README.md`
  and `done/`, and unnumbered briefs alongside any numbered ones. **Numbering is discontinued** — the 01–16
  run is historical, and new briefs are named for what they are. `briefs/README.md` is authoritative, and
  explains why a single `active/` folder was tried and rejected on 2026-08-07.
- **`investigations/`** — things we measured, profiled or reproduced. **One directory per investigation, holding
  its write-up, its scripts and its data together**, because a measurement whose apparatus lives in another tree
  is not reproducible in practice however carefully it was written. **`investigations/README.md` lists what is
  there, one line each saying which question it answers** — read that rather than an enumeration here, which
  went stale at ten of seventeen before anyone noticed.
  - **The trigger: you are about to write a script to find out how something behaves. Read
    `investigations/README.md` first, and then the bundle's own README if one looks close.** Same shape as
    checking for an existing helper before writing one, and it fails the same way — the cost is not the
    wasted hour but a *second* answer to a settled question, which the next reader has to reconcile with
    the first. The list is one line per bundle; reading all of it is cheaper than one probe run.
  - **A close bundle is usually the right home for the new probe, rather than a new directory.** These
    accumulate — `dpg-focus/` has gathered probes over months, because each new question about focus and
    keyboard dispatch belonged beside the table it was extending. A bundle per probe would scatter one
    subject across the index and lose exactly the adjacency that makes the earlier answers findable.
- **`TODO.md`** for planned work, **`TODO_DEFERRED.md`** for things noticed mid-task and set aside.
- **`scripts/`** — repository-maintenance tooling: scripts that check *this repo*, run by a maintainer and
  not shipped in the wheel. Distinct from `raven/tools/`, which holds user-facing console scripts. Each is
  indexed in `scripts/README.md` with the question it answers.
- **`dpg-notes.md`**, **`raven-style-guide.md`** — at the root because they are consulted constantly.

A fourth kind of non-source material sits *inside* the package tree rather than beside it, because it belongs
with what it produced:

- **`00_workfiles/` holds originals, never runtime assets.** Wherever shipped art lives, the sources it was
  exported *from* live in a `00_workfiles/` subdirectory next to it — GIMP `.xcf`, Inkscape `.svg`, camera
  originals, intermediate crops. Currently `raven/icons/00_workfiles/` and
  `raven/avatar/assets/characters/00_workfiles/`.
  - **Nothing at runtime may read from one.** The name is the contract: an editable original is not an asset,
    and code that loads one is a bug, not a shortcut. What ships is the export beside the directory.
  - **They are excluded from the wheel** (`[tool.pdm.build].excludes` in `pyproject.toml`). Measured
    2026-08-12: they were 92.8 MB of a 107 MB wheel, which is now 14.4 MB. The exclusion is safe *because*
    of the rule above, so a new `00_workfiles/` anywhere in the tree is covered automatically by the
    `**/00_workfiles` glob.

Two conventions worth knowing before adding to any of them:

- **Keep an artifact with what produced it.** This is why `investigations/` exists, and it applies wherever the
  artifact lives — a completed brief with apparatus becomes a directory too (`briefs/done/dpg-markdown-bullet/`
  is a write-up plus its reproduction script). A **shared** instrument is pointed at by path, not copied into
  every bundle that used it.
- **Record the link in the bundle's README**, naming each script and what it answers. The doc↔script connection
  was not recorded consistently in the past, and recovering it meant asking git what landed in the same commit
  as each script. Writing it down is what stops that recurring.

## Build and Development

**The supported platforms are Linux, macOS, and Windows.** Development happens on Linux, so the other two
are reached only through CI and through the people running them — which makes a platform-specific CI
failure a report from a real user's environment rather than a curiosity to be waived. The test matrix
covers all three for that reason. (A Windows-only crash on 2026-08-11 turned out to be a latent mistake
that Linux and macOS had simply tolerated, which is the usual shape of these.)

**A single GPU with modest VRAM is a supported configuration, not a degraded one.** The maintainer's
development machine has an external GPU attached at the desk, so the LLM gets a card to itself and
everything else — avatar, TTS, and the rest of the server's models — gets another. Away from the desk there
is one card for all of it, and that is an ordinary way to run Raven rather than an edge case.

Worth knowing because **the desk setup hides a whole class of problem**: anything where two subsystems
contend for the same GPU simply does not happen where the work is done. So a contention issue will not
surface on its own, will not reproduce when reported, and has to be sought deliberately by hiding the
external GPU from the process. Treat "it is fine here" as saying nothing about the one-card case.

Uses PDM with `pdm-backend`. **Python 3.11–3.12** (see `pyproject.toml`: `requires-python = "<3.13,>=3.11"`). Optional CUDA extras via `pdm install -G cuda`.

### Why the 3.12 upper cap

The cap comes from `kokoro` (Kokoro TTS) and its phonemizer `misaki`, which currently require `<3.13`. Raven's own code and every other dependency (`mcpyrate`, `unpythonic`, `torch`, `Pillow`, `numpy`, …) already support Python 3.13 and 3.14. The plan to lift the cap has two branches:

- **(a)** Kokoro/Misaki upstream expand their supported Python range — in which case we just bump `requires-python` and widen the CI matrix.
- **(b)** If those projects look dead after a reasonable wait, we vendor both. Kokoro is the TTS engine, Misaki is its English phonemizer; together they're self-contained enough to be absorbed into `raven/vendor/` alongside `tha3/`, `DearPyGui_Markdown/`, etc.

**(c)** was added 2026-08-10, and is the branch to reach for if the synthesizers move on: replace Kokoro. The
constraint that pins us to it is *lipsync* — the avatar needs timestamped words plus per-word phonemes, from
which timestamped phonemes interpolate linearly, and few engines expose that. There is an audio-analysis route
to the same data from plain synthesized speech, which would make the engine choice free. So: (a) upstream
widens, (b) we vendor, (c) we pivot the engine once something is materially better and the analysis route is
built.

**The deadline is October 2028, when Python 3.12 goes EOL** — and there are now *two* items on that clock, not
one. Kokoro caps the Python version outright. `torchaudio` caps the *torch* version: its latest release is
2.11.0 (2026-03-23), it skipped both torch 2.12.0 and 2.13.0 while torchvision shipped same-day with each, and
`raven.common.audio.resample` is its only user. Decision 2026-08-10: **keep both and re-check later.** Two
years is enough time to build an alternative from whatever parts exist then, and charting the risk in advance
is what keeps it a plan rather than an emergency. What would make it urgent is either project going visibly
dead, or a synthesizer worth switching to arriving first.

Until one of those branches lands, **don't add `3.13`/`3.14` to the CI matrix** — it would fail at dependency resolution time. The test CI currently works around this by using `pip install -e . --no-deps` and hand-picking a minimal dependency subset for the test suite, which avoids pulling in kokoro/misaki at all. That's how the test matrix can stay lightweight even though kokoro lives in the full `[project] dependencies`.

### `source env.sh` too, not just the venv

Raven's CUDA libraries are the pip-installed ones under `.venv/.../site-packages/nvidia/`, and nothing
puts those on the loader path by default. `env.sh` does it — it appends every nvidia `lib/` directory to
`LD_LIBRARY_PATH` and adds `ptxas` to `PATH` — and the `~/.bashrc` wrappers for `raven-server` and the
other entry points source it before running the real command. So the *apps* always have it.

**A shell that only activated the venv does not.** In that shell `import cupy` succeeds while
`import cupy.cublas` raises `ImportError: libcublas.so.12`, so `thinc.compat.has_cupy` is False and
`spacy.require_gpu()` fails. Everything looks like a broken GPU installation, and none of it is true of
the processes that matter.

The failure this prevents is not a crash but a *wrong conclusion*: an environment-dependent probe run in
the wrong environment reads exactly like a bug report, and gets written up as one. (Live case,
2026-08-06 — a deferred TODO claiming "spaCy silently runs on CPU" was raised, argued and committed
before checking `/proc/<server-pid>/environ`, which showed sixteen nvidia directories on the path and
cupy mapped into the process.)

So when setting up a session that will touch CUDA, GPU device selection, or anything asking *which
device is this running on*, source it alongside the venv activation:

```bash
source env.sh          # after $(pdm venv activate)
```

To check the answer for a *running* process rather than for your shell, read its environment directly —
`tr '\0' '\n' < /proc/<pid>/environ | grep LD_LIBRARY_PATH` — or look for the library in
`/proc/<pid>/maps`. That is authoritative where your own shell is merely suggestive.

### Working-tree state: `config.py` files are edited in place

Raven is configured via in-place edits to tracked `config.py` files — paths, model choices, hardware-specific tweaks. On any dev machine, expect some subset of the following to show up as `M` in `git status` as the **normal steady state**, not as a pending change that needs committing:

- `raven/client/config.py`
- `raven/librarian/config.py`
- `raven/visualizer/config.py`

The specific files and the specific contents differ between dev machines; the pattern is the same everywhere — at least some config.py somewhere carries local overrides.

**Implication for `git add`**: add specific files by name. **Never** `git add -A`, `git add --all`, `git add .`, `git add -u`, `git add --update`, or `git add raven/`. If a commit you're working on touches one of these files coincidentally (e.g. a refactor sweeps through them), check with me before staging — there may be an unrelated local override mixed in that shouldn't be part of the commit.

**`-u` belongs on that list even though it looks narrower than `-A`.** It stages only files git already tracks, which reads as the safe one — and every `config.py` here is tracked, so it sweeps up exactly the overrides this section exists to protect. It is tempting for the same reason each time: after a wide refactor it is the short way to say "the files I touched", which it is not; it means "every tracked file that differs", and the difference is invisible until it is committed. (Live case 2026-08-07: a docs restructure staged with `-u` put a personal-machine hostname into `llm_backend_url` on a public repo, and the working tree looked *clean* afterwards, which is what made it noticeable at all.) All these forms are denied in the agent's permission settings, so the failure should now be a refused command rather than a bad commit.

**That sentence was not true of `git add raven/` until 2026-08-10, and the gap cost exactly what it looks like it would.** The directory form was on the never-list above but absent from the deny list, which covered only the flags — so the documentation asserted a guard that did not exist, and the command went through unremarked after a multi-file change ("add the files I touched"). It staged all three `config.py` overrides; caught in `git status` before committing, but only because the habit of reading the staged list survived the missing guard. The rule is now `Bash(git add raven)` and `Bash(git add raven/)` as **exact** matches, deliberately not `git add raven:*` — a prefix rule would also refuse `git add raven/librarian/app.py`, which is the correct way to stage and must stay frictionless.

Version is defined in `raven/__init__.py` (`__version__`), read by PDM via `[tool.pdm.version]` in `pyproject.toml`. Tag format: `vX.Y.Z`.

```bash
pdm install              # creates .venv/ and installs deps
pdm use --venv in-project
```

Prefix commands with `pdm run` if the venv is not active.

### Entry points: reach for the CLI before writing a script

All defined in `pyproject.toml` under `[project.scripts]`, which is authoritative — this table is an index
so the tools are *findable*, and its failure mode is a one-off script doing badly what a shipped tool does
well. (Live case: an ad-hoc `HybridIR(...).add(...).commit()` script written to re-ingest three documents
that `raven-indexer` indexes as a matter of course.) Before scripting against `raven.*` internals for an
operational task, check here.

**GUI apps** — `raven-visualizer` (the main app), `raven-librarian`, `raven-avatar-pose-editor`,
`raven-avatar-settings-editor`, `raven-xdot-viewer`, `raven-cherrypick`, `raven-conference-timer`.

**Servers and terminal frontends** — `raven-server` (the ML inference API), `raven-minichat` (readline REPL
on the Librarian backend).

**Headless pipeline tools**, the ones worth knowing before writing anything:

| tool | what it does |
|---|---|
| `raven-indexer` | build or refresh Librarian's RAG index over a documents directory. Takes `-d/--db-dir`, so it indexes any corpus into any index without touching config |
| `raven-importer` | Visualizer's BibTeX → dataset import pipeline |
| `raven-arxiv-search` | arXiv boolean search → identifiers |
| `raven-arxiv2id` | parse arXiv identifiers out of filenames or BibTeX; `--strip-versions` discards the pinned version, which is how a collection gets refreshed to latest |
| `raven-arxiv2bib` | arXiv identifiers → BibTeX (records the version arXiv *returned*) |
| `raven-arxiv-download` | fetch fulltext PDFs for identifiers; `--save-bib` writes the BibTeX from metadata it already fetched, so the two runs cost one set of politeness delays |
| `raven-deduplicate` | merge the duplicate records a multi-database literature search leaves. Matches on DOI and on normalized title, writes an audit TSV of every merge, and reads through `fixbib`'s repair so the count is honest. `--judge` adds an opt-in LLM pass over the near-misses |
| `raven-burstbib` | split a multi-entry BibTeX file into one file per entry |
| `raven-wos2bib`, `raven-csv2bib`, `raven-pdf2bib` | bibliography converters from Web of Science exports, CSV, and PDF metadata |
| `raven-dehyphenate` | undo line-break hyphenation in extracted text |
| `raven-qoi2png` | image format conversion |
| `raven-check-cuda`, `raven-check-audio-devices` | environment diagnostics |

**To see what an app is doing, use its own flags: `--log-level DEBUG` and `--log PATH`.** Every app with a
frontend takes both — the GUI apps (pose editor, settings editor, cherrypick, conference timer, librarian,
visualizer, xdot viewer), `raven-minichat` (a terminal REPL, not a GUI), the visualizer's importer CLI, and
the server. `--log` writes the log where you ask instead of leaving you to redirect a stream you also want
to watch.

- **`--debug` is not a logging flag.** In `raven-cherrypick` it turns on debug *overlays* — pan/zoom
  coordinates, click positions — and does nothing to the log level. Reaching for it and finding an empty
  log is a two-minute detour that has been taken at least once.
- **Twelve of the twenty-five console scripts still lack them**, all of them the smaller CLI tools:
  `raven-indexer`, the arXiv four, `raven-burstbib`, `raven-fixbib`, `raven-dehyphenate`, `raven-qoi2png`,
  the two `raven-check-*`, and `raven-pdf2bib` (which has `--log-level` but not `--log`). Audited
  2026-08-17; `raven-deduplicate` arrived 2026-08-28 with both, so the gap is not growing. That is a gap
  to close rather than a convention with exceptions.

**To point an app at a different endpoint — or at nothing — use `--backend-url` and `--server-url`**
(2026-08-20). `raven-librarian` takes both (the LLM backend and the Raven server); `raven-visualizer` takes
`--server-url`, the only one it uses. Each logs the override against the configured value it replaced.

This is how a *degraded* state gets exercised on purpose, which is otherwise awkward: aim `--backend-url`
at a port nothing is listening on and Librarian's backend-status pill appears and stays, and aim
`--server-url` likewise and the Visualizer's importer falls back to loading models locally. The alternative
was editing a `config.py` that carries local overrides and restoring it exactly afterwards — which is the
one file class this repo is most careful about, so a flag is worth having for that reason alone.

**Every console script that talks to Raven-server takes `--server-url`, and every one that talks to an LLM
backend takes `--backend-url`** — same spelling everywhere, no exceptions to look up. `raven-minichat` and
`raven-pdf2bib` took the backend as a positional `url` until 2026-08-20; it is a flag on both now.

### Running Tests

```bash
pytest                   # everything except the GUI and live-backend groups
pytest --run-gui         # ...including the GUI group. Takes keyboard focus — warn the user first
pytest --run-llm         # ...including the live-backend group, against the configured backend
pytest --backend-url URL # ...the same, against a backend elsewhere (implies --run-llm)
pytest -m "not ml"       # what CI runs; the ML stack isn't installed there
```

Three markers divide the suite, and every default is chosen so that the command a person types by
reflex is the safe one:

- **`gui`** — maps a real window. Focus is a single-holder resource on a shared desktop, so these
  **steal the keyboard** from whatever the user is typing into (see "Live GUI testing on a shared
  desktop" below). Skipped unless `--run-gui` is passed, and **tell the user before passing it.**
  Note that most tests touching DPG are *not* in this group: a DPG context with an unmapped viewport
  takes no focus, so only tests that genuinely need rendered frames — focus semantics, layout
  geometry — need the marker.
- **`ml`** — needs the real ML stack (spaCy, Flair, torch model weights). Runs locally by default and
  is skipped in CI, which installs a hand-picked dependency subset rather than the multi-gigabyte
  tree (see the `ci-setup` skill).
- **`llm`** — talks to a live OpenAI-compatible LLM backend, and so **opens a connection to it**. Skipped
  unless `--run-llm` (the configured `llm_backend_url`) or `--backend-url URL` (anywhere else) is passed.
  Opt-in for the connection rather than the cost: on by default, a CI runner would open a socket to
  whatever the committed URL names, on a machine nobody here controls — which a test suite has no business
  doing unasked, whatever is or is not listening. Having opted in, they still skip when nothing answers,
  and **every skip names the URL it tried**, a silent skip being indistinguishable from a pass.
  - These are the assertions a mock cannot make. Everything else in the suite mocks the backend, so a
    tool-calling regression in the inference engine — a changed chat template, a reworked tool parser —
    breaks Librarian at runtime with nothing going red. `raven/librarian/tests/test_live_backend.py` is
    the group; `get_current_time` is the one tool needing nothing outside the process, which is what lets
    it exercise the whole tool loop hermetically.

A test that needs none of the three takes no marker, which is the overwhelmingly common case.

### A test that fakes a home directory must set `USERPROFILE` as well as `HOME`

`expanduser` reads `HOME` through `posixpath` and `USERPROFILE` through `ntpath`, and **never consults
`HOME` on Windows** (`ntpath.expanduser` checks `USERPROFILE`, then `HOMEDRIVE`/`HOMEPATH`). So a fixture
that points `HOME` at a `tmp_path` redirects the suite on Linux and macOS while leaving the Windows runner
resolving `~` to the real profile of whoever is running it.

The failure is partial, which is what makes it confusing: `expandvars` *is* plain environment substitution
on every platform, so anything going through it — expanding the `$HOME` inside an XDG value, say — works
from `HOME` alone. Only the assertions that expand `~` themselves fail, and only on Windows.

Set both. (Live case 2026-08-18: `TestUserDirectory` went green on ubuntu and macOS and took 7 failures on
windows-latest.)

### Don't sleep through a duration to sample partway into it — wind the clock instead

A test that starts a timed animation, sleeps most of its duration and then takes one sample is betting that
everything in between fits in the remainder. On a dev machine it always does. **The macOS runner is where
that bet is lost**, repeatedly and across unrelated tests (Juha's observation; the Linux and Windows runners
in the same matrix pass the same test).

The correct form is already in the suite: set the animation's `t0` back by the fraction you want to sample
at, then render one frame. Deterministic, needs no sleep, and reads more clearly about what is being
sampled. `test_animation.py`'s `sample_at` helper is the worked example.

**The reason to care is not the flake, it is what the flake says.** These failures name the wrong cause.
When `test_a_widget_with_no_colour_of_its_own_fades_back_to_the_default` lost this bet (2026-08-20), the
flash had already finished and handed the widget back with no colour declared — and DPG reports an
undeclared colour as a sentinel whose red channel is the very colour the fade started from, so the assertion
fired with "the fade is heading for the default text colour, not for black" about a fade that had ended
perfectly. Half an investigation goes into the mechanism the message points at before anyone checks the
clock.

So treat a macOS-only failure as a report about the *test*, not about macOS. It is the runner that keeps the
suite honest about its timing assumptions, which is worth having.

### Naming and placing a test module

**`test_X.py` tests the module `X.py`, and lives in the `tests/` directory of X's own package.** So
`raven/common/text/normalize.py` is tested by `raven/common/text/tests/test_normalize.py`. Every subpackage
carries its own `tests/` (`audio/`, `gui/`, `image/`, `text/`, `video/`, …); `raven/common/tests/` is for the
modules sitting directly under `raven/common/`.

Where one module's tests are split by aspect, **every part still names the module** — `test_download.py` and
`test_download_metadata.py`, not `test_metadata.py`.

Two kinds of test have no single module to name, and take a descriptive name instead. Both must say what they
cover in the docstring's first line, since the filename no longer does:

- **Spanning several modules** — `test_tts_stt_roundtrip.py` (TTS → resample → STT).
- **Pinning a third-party library's behaviour**, where there is no Raven module in the picture at all —
  `test_focus_semantics.py`, which characterizes DearPyGui's focus model.

There is deliberately **no `integration` keyword** in the naming scheme. Two files out of 74 are in these
categories, and they are not the same kind of thing — one crosses Raven modules, the other tests no Raven
code whatsoever — so a shared keyword would assert a similarity that is not there.

**The failure this prevents:** a test named after a module that no longer exists. `layout_math.py` was once
`viewport_math.py` and its test kept the old name; a coverage audit on 2026-08-10 read that as "layout_math
is untested", which was wrong, and would have been filed as a real gap had the file not been opened. A stale
test name costs a re-investigation every time somebody checks, and nothing ever fails to make it visible.

### Linting

```bash
ruff check <changed .py files>   # primary linter (config in pyproject.toml)
```

Legacy `flake8rc` also present (used by Emacs flycheck, not by CI or CC).

### Workflow Rules

1. **Lint after every code change**: `ruff check <changed .py files>`. Do this before review, testing, or committing. Catches unused imports and dead names early.
2. **Run `python scripts/check_ci_imports.py` before pushing anything that adds an import or removes an `importorskip`.** CI installs a hand-picked dependency subset rather than the full tree, so a module-level import of something outside that list passes locally and fails only on push — as `sseclient` did, turning main red on a docs-only commit from a change two commits back. The script answers, in a second, which unguarded test modules would fail to *collect* in CI. A green local `pytest` cannot tell you this.
3. **Run `python scripts/check_dependency_versions.py` after touching any dependency list.** Same shape as the above, one level up: it asks whether `pyproject.toml`, `pdm.lock` and `requirements-ci.txt` still agree about *versions*. See the next section for the loop it belongs to.
4. **Run `python scripts/check_exports.py <the files you touched>` before committing a new public symbol** — a function, class or constant added to an `__all__`. The trigger is *adding a name to `__all__`*, which is the moment the list can stop mirroring the file: a name appended where it was convenient rather than where its definition sits makes the list predict nothing, and it is invisible in the diff of the change that caused it. The other half it catches is a public symbol that never reached `__all__` at all, so `import *` quietly does not bring it — which is how `guiutils.DEFAULT_BUTTON_BG_COLOR` sat outside its list while `animation.py` imported it by name. **Raven follows the plain convention here — no leading underscore means public** — so that is a failure rather than a matter of taste. What it cannot decide is which names *should* be public: an unexported one is either a missing export or a name that wanted an underscore. **It reports and never rewrites**: those lists carry comments whose prose is positional, and a sort would scramble the commentary while making the names right.
5. **Run `python scripts/check_module_maps.py` after adding or removing a module** in a package that has a `CLAUDE.md` module map (`raven/librarian/`, `raven/visualizer/`). A new module is simply absent from the map, and absence is the one error nobody spots: the table looks complete either way, and the next session reads it as the list of what exists. `indexer.py` was unlisted from the day it landed. The same run reports sizes that have drifted more than the rounding allows.

### Dependency versions: CI moves first, the floors follow a green local run

Three lists say what Raven depends on, and each answers a different question:

- **`.github/workflows/requirements-ci.txt`** — exact pins, what CI actually tests.
- **`pyproject.toml`** — floors, what an installer is permitted to resolve.
- **`pdm.lock`** — one resolved version per package, what a developer gets. Gitignored here, so it is
  invisible in `git status` and drifts unnoticed.

**The loop between them, in order:**

1. **Dependabot bumps the CI pins.** That is what it is for, and the PR arriving red is the point: an
   upstream release that breaks us shows up as its own reviewable failure rather than as a mystery
   attached to somebody's unrelated commit.
2. **Merge the ones that pass.** Green on the full matrix is the evidence.
3. **Re-lock and run the suite locally** — `pdm lock && pdm install`, then `pytest`, with `raven-server`
   and an LLM backend up so the server-dependent tests are not skipped. **Then `pytest --run-gui`**, as
   its own invocation. Both groups exist because CI cannot run them: `ml` needs the multi-gigabyte model
   stack, `gui` needs a display and a window that takes focus. A local run that skips them covers no
   more than the matrix already did.
   - **`--run-gui` takes the keyboard and can crash** (a resource-allocation fault on the deferred list),
     so warn before starting it and run it *after* the main suite — separately, so a crash cannot cost
     the results of everything else.
   - **A `dearpygui` bump is not green without it.** The GUI group is where a toolkit change actually
     lands: focus semantics, layout geometry, the file dialog's keyboard. The rest of the suite drives
     DPG with an unmapped viewport and would not notice.
4. **Only then raise the floors in `pyproject.toml`**, to the versions that run just passed.

**Do not raise a floor to a version nothing has tested.** The floor is a claim about the oldest version
that works, and the only evidence for it is a run. This is also why step 4 is last rather than bundled
into step 2: Dependabot's PR proves the *new* version works, not that the *old* floor is wrong.

**A floor is raised to the oldest version we exercise, not the newest.** CI's pin and the lockfile
routinely differ by a patch — whichever was refreshed most recently — and the floor belongs at the lower
of the two. Going higher declares a minimum that the other list then falls below, which the checker
correctly reports as a violation.

**Keep the reason when the number moves.** Several floors record *why* they are where they are —
`dearpygui>=2.3` for automatic font atlas ranges, `python-docx>=1.1` for `iter_inner_content`. When the
floor is raised for the unrelated reason above, the fact survives in the comment: `python-docx>=1.2.0,
# … `iter_inner_content` … arrived in 1.1`. The declared floor answers "what do we test"; the comment
answers "what do we actually need", and those stop being the same number the first time this loop runs.

**Two dependencies sit outside the loop deliberately.** The torch trio is pinned exactly, from its own
index, and is bumped by hand as a set. `PyTurboJPEG` is capped `<2` because 2.x needs a *system*
libjpeg-turbo 3.0+ that current distributions do not ship — a constraint pip cannot see, so the version
bound is the only place it can live.

### CHANGELOG layout: group by component

Raven ships many separate user-facing apps, so within each of **Added** / **Changed** / **Fixed**, entries are grouped under an italic component header and the per-entry `*Raven-<app>*:` prefix is dropped — the header carries it. Entries then read as continuations of the header, so they start lowercase.

Component order is fixed, so a reader learns where to look: *Raven-librarian*, *Raven-visualizer*, *Raven-server*, *Raven-avatar*, *Raven-cherrypick*, *Raven-arxiv-download*, *Raven-pdf2bib*, then *Constellation-wide* for anything cross-cutting (install, device strings, CLI options shared by every app, client-side HTTP behavior). Omit a component that has no entries in that section. An entry spanning two tools goes under the primary one and names the other inline ("with `raven-wos2bib`: …").

**File a new entry into its group when you write it.** The failure this prevents: 0.2.8 accumulated 58 flat entries — 24 of them opening with `*Raven-librarian*:` — before anyone noticed the prefix was a heading doing prose duty, and regrouping after the fact is a large, error-prone reshuffle that has to be verified entry by entry.

**Measurements from a particular corpus stay out.** "1598 of 6934 records", "1650 occurrences of `©` down to 1" — a number like that is evidence *for* the change, and it is evidence drawn from one dataset that happened to be on a JAMK researcher's disk. A reader of the changelog has a different corpus and wants to know what the feature does for theirs, so the entry says the *kind* of effect and its shape ("this can account for a large share of a file", "publisher names largely gone from the word cloud"), and the figures go where they belong: the commit message, and the brief if there is one.

The pull toward including them is strong and worth naming, because it is not laziness — a measured number is the most honest thing in the room, and it feels like the strongest sentence available. It is, for the audience that shares the corpus. The changelog's audience does not. (Caught twice in one session, 2026-08-28, in entries for `raven-fixbib` and the abstract boilerplate stripper — both written immediately after the measuring, which is exactly when the numbers are most vivid and least transferable.)

This is Raven-local, not fleet-wide: elsewhere in the fleet a project *is* the component, so a header would be noise. Wording rules (density, nesting, users-not-commits, "was it broken in the last tagged release?") are fleet-wide and live in the `changelog` skill.

### Live GUI testing on a shared desktop

Raven's apps are DPG, so verifying GUI work means running them — and the agent and the human are on the *same X session*. Keyboard focus is therefore a shared, single-holder resource: a window that maps or gets activated takes focus away from wherever the human is typing, and their next keystrokes land in the app instead of their editor or terminal. (Observed the obvious way: a launched Librarian window swallowed a half-typed message and its Enter, which sent an empty chat turn.)

**The recipes are in the `live-gui-testing` skill** (fleet-wide): finding the window, aiming a click at a widget, sending synthetic keys that behave like real ones, confirming an action landed, closing the app again, driving one from inside its own process, and putting a TCP relay in front of a dependency so its appearance is an event you time. Load it when you are about to do any of that.

What stays here is the short list that has to fire **before** the decision to launch — which is earlier than a skill can load, because the decision is usually incidental ("let me just check this renders"):

- **Announce a launch, and any input injection, before it happens — in the message *and* as a desktop toast.** The mapping window steals focus by itself, so this is owed even for a look-and-see run. Send both, because they fail in opposite directions and neither covers the other's case:
  - **The message line lands where the eyes already are.** The composer sits directly below the newest output, so a short, loud line there is in view while the user is typing. But it only reaches them if *this* session's window is the one on screen.
    - **Claude Code strips ANSI escapes from assistant text** (tested 2026-08-21 — an SGR orange-bold sequence arrived as plain text). So the only emphasis available is Markdown's, which on this project's terminal is barely brighter than body text. That is what the toast is compensating for; don't spend another round rediscovering that colour is unavailable.
  - **The toast reaches a session the user cannot see** — `cc-toast "taking keyboard focus"`, which also makes a sound. That is the dangerous case: a window mapping out of a backgrounded session steals focus from whatever they are actually doing, and there is no output for them to be reading. Its cost is position — it appears in the far corner of the display, about as far from the composer as the screen allows.
    - **Use the script, not a bare `notify-send`.** The toast is deliberately critical-urgency, and a critical notification never expires — so a bare `notify-send` per launch leaves one on the screen per launch. A dozen were waiting after one lunch break. `cc-toast` replaces the previous toast instead of adding to it, so there is at most one; `cc-toast --clear` takes it away. It lives in `~/.claude/scripts/` and its header records what was measured about the notification server, including why dropping to normal urgency is not the fix.
  - **The announcement does not protect someone who is already typing**, which is the case it most needs to. They are watching their own words, not the scrollback, so the window maps under their hands and the rest of the sentence goes into the app. Whatever those letters mean there, happens. (Live case 2026-08-19: a launched `raven-cherrypick` took the keyboard mid-message and a `g` in the sentence reached it, which is *cycle filter*; the grid emptied and the status line read `0 / 79`. It was investigated as a rendering bug, then misattributed to the agent's own `xdotool` — which had in fact hung before injecting anything.) **So prefer a check that needs no focus at all**: a screenshot of an unfocused window is never intrusive, and `--run-gui` tests exist for the rest.
- **The whole drive sequence goes in ONE Bash call.** Activate → click → type → send → restore focus, and any waiting in between, must be a single command. Two reasons, of which only the first is permanent:
  - **Shell state does not survive between calls.** `WID`, `PREV` and the launched app's job are locals of the shell that made them, so a split sequence has no window to drive and nothing to hand focus back to.
  - **Where each call raises its own permission prompt, the prompt itself steals focus** back to the Claude Code window; acknowledging it re-focuses CC, and the `xdotool` in the next call then drives the terminal rather than the app — silently, since `xdotool` reports success either way. This was the original reason for the rule and it does not apply while prompts are switched off, so don't reason from it when explaining the rule; reason from the line above, which always holds.
- **Restore focus afterwards** — capture `PREV=$(xdotool getactivewindow)` *before* launching or activating, and `xdotool windowactivate --sync "$PREV"` when done.
  - **But restoring also *raises* that window, burying the app under test.** `xdotool windowactivate` focuses and raises together, so the tidy last step puts Claude Code on top of the app just launched — which reads as the app having closed. Cost a false crash investigation once (process alive, window present, simply behind). Either say where the window went, or skip the restore when the human is about to drive the app themselves, which is the common case after a restart.
- **Never `pkill -f raven-<app>`.** The pattern matches the agent's own shell command line, so it kills the invoking shell (exit 144) and usually leaves the app running. Select real PIDs instead — `pgrep -af raven-librarian | awk '$2 ~ /python/ {print $1}' | xargs -r kill` — or better, close it through the window manager, which the skill covers with the measurements behind it.

**Why these four stay and the recipes go.** The costs are asymmetric: forgetting a recipe wastes a run of your own, while forgetting to announce spends the human's keystrokes on whatever the app makes of them. A rule that protects *someone else* cannot be demand-loaded, because the moment it is needed is the moment before the task has been recognised as the kind that needs it.

**A per-message hotkey needs the blue dot on the right message first — press End.** In Librarian the
per-message keys (Ctrl+T, Ctrl+R, Ctrl+U, Ctrl+S) act on *the message the keyboard mark is on*, which is
the bottommost message whose whole button row is on screen. A driven test that scrolls to look at something
and then presses one of these aims it wherever it happens to have stopped, and `End` is the one-key way to
put the mark on the last message. `raven/librarian/README.md` → *Chat message actions* states the rule for
users; this is the operational half.

The failure is silent in a way worth naming, because it looks like a bug in the feature: the key is
delivered, the handler runs, it acts on a message that is off screen or is the *user's* rather than the
AI's, and the visible result is nothing at all. (Live case 2026-08-28: Ctrl+T twice reported as "the trace
will not expand", with the app behaving correctly both times — the mark was on the user message, because
the previous step had scrolled up to read.) So before concluding that a hotkey does not work, check where
the mark is; a screenshot shows it, and costs no focus.

### DPG Pitfalls

**Before editing any DPG code, invoke the `dpg` skill** (`.claude/skills/dpg/`), which indexes the reference by
the question you arrived with and carries the one standing instruction: *measure a DPG claim before writing it
down*. The reference itself is **`dpg-notes.md`** (project root) — read the section the skill routes you to. The full DPG reference: threading model, callback dispatch, `split_frame` mechanics, texture upload ordering, keyboard input / keycode traps, window sizing gotchas, diagnosing background-task races. "DPG code" = anything importing `dearpygui`, the render loop, key/mouse handlers, or texture / `split_frame` work. The pitfalls listed below are an index, not a substitute for the full notes. **When you discover a new DPG gotcha, record it in `dpg-notes.md`** (and add a one-line pointer below if it's pitfall-grade).

1. **DPG threading — push work to background threads aggressively.** Unlike most GUI toolkits, DPG allows all operations from background threads: creating/deleting items, setting values, creating OpenGL textures. Resist the "standard GUI toolkit" instinct to marshal everything to the main thread — doing work on background threads simplifies code and reduces GUI stutter, especially when the heavy lifting is non-Python (C/CUDA) and can release the GIL.
2. **`dpg.split_frame()` — not in the render loop thread.** `split_frame()` waits for the render loop to complete one frame. Safe to call from background threads, DPG event callbacks, and frame callbacks (DPG dispatches these on a separate thread). **Deadlocks** if called from code that runs synchronously in the render loop — i.e. anything in the `while dpg.is_dearpygui_running(): dpg.render_dearpygui_frame()` loop body (e.g. animation frame updaters), or before the render loop starts (startup code). Common use: call from a background thread after creating textures, to ensure DPG processes them before the next render.
   - **The restriction is enforced, not lifted.** Use `raven.common.gui.utils.split_frame(operation=..., required=...)` in preference to the bare `dpg` call wherever the calling thread isn't obvious from two lines of context. It cannot wait in the render loop either — nothing can — but it *detects* that it was called there and reports it instead of hanging: `RuntimeError` when waiting is load-bearing (`required=True`, the default), or a warning and a stale-geometry fallback when the wait only improves the result (`required=False`). The trade is a silent hang for a named failure, not a lifted constraint. One predicate (`guiutils.is_render_thread`) covers both this pitfall and pitfall 4, since startup runs on the main thread too. Rationale and per-site policy in `dpg-notes.md`.
3. **`dpg.set_frame_callback(N, cb)` — one callback per frame number.** Only one callback can be registered for any given frame N. A second `set_frame_callback(N, ...)` silently overwrites the first. If you need multiple actions at the same frame, combine them into a single callback, or use different frame numbers.
4. **Defer startup work that may show error dialogs to a frame callback.** The modal messagebox uses `split_frame`, which deadlocks before the render loop is running. If startup code (e.g. loading a file from a CLI argument) may need to show an error dialog, defer it to `dpg.set_frame_callback(N, ...)` so the render loop is active. This is a standard Raven pattern — see `raven.avatar.settings_editor.app` and `raven.xdot_viewer.app`.
5. **DPG widget IDs must be unique — violating this crashes the process, not raises an exception.** Combined with Python's lazy garbage collection, explicit `dpg.delete_item(...)` does not guarantee the ID is free for reuse: the old widget may still be in DPG's registry for some unbounded time after the delete call. Raven's defensive pattern for any widget that gets dynamically recreated (tooltip groups, info-panel content, per-entry groups, etc.) is **version-counted tags**: every rebuild increments a monotonic counter, and every tag created during that rebuild embeds the counter (e.g. `f"cluster_{cid}_item_{data_idx}_annotation_title_build{build_number}"`). Even if the old widgets aren't collected yet, the new tags won't collide. The counter increments on *every* build attempt, including cancelled ones, so a cancelled build's partial widgets can't collide with the next build either. For the top-level "current vs. previous" swap (where the slot itself has a stable identity), track the current widget *ID* in a module-level Python variable rather than relying on an alias rebind — `dpg.set_item_alias(new_item, existing_alias)` does not reliably rebind after the aliased item is deleted.
6. **When rebinding an alias across a swap, delete the old item by widget ID, not by alias string.** The working pattern is: hold the current widget ID in a Python variable, call `dpg.delete_item(old_id)`, then `dpg.set_item_alias(new_id, alias_str)`. Calling `dpg.delete_item(alias_str)` instead appears to leave the alias→id mapping partially dirty, so the subsequent `set_item_alias` lands in an inconsistent state and later lookups by that alias return `0` (→ `configure_item(0, ...)` raises `SystemError: Item not found: 0`). This is observable even on DPG versions that fixed the older manual-alias-cleanup bug (hoffstadt/DearPyGui#1350). See `raven.visualizer.info_panel`'s content swap (app.py `_update_info_panel`) and `raven.visualizer.annotation`'s `_current_group` handling for the working pattern.
7. **Focus is not the caret, and `focus_item` cannot focus a child window.** ImGui auto-focuses the first navigable item of a window, so a text field reports `is_item_focused` True within a few frames of startup with nobody having touched it — a global hotkey handler gated on that silently swallows every key it delegates to the field. Gate on `dpg.is_item_active` instead, which is True only while the field owns the caret.
   - **Except for the chord that *commits the edit*, which needs `is_item_focused`.** Committing deactivates the field, so a handler gated on `is_item_active` can never fire — by the time it runs the state it tests is already cleared, and the hotkey dies silently. Which chord commits varies: bare Enter on a *single-line* `InputText`, and **Ctrl+Enter on a multiline one** (bare Enter there inserts a newline and leaves the field active). So `raven-visualizer` (single-line search) and `raven-librarian` (multiline composer, Ctrl+Enter to send) legitimately gate their send handlers on `is_item_focused` while both gate their bare-key branches on `is_item_active`. Ask which kind a chord is rather than reasoning from the field's kind; both variants shipped broken and were caught only in live testing.
   - Separately, `dpg.focus_item` on a *child window* does not focus it: focus lands on the enclosing window's first navigable item and is **activated**, so "park focus on the scroll panel" hands the caret to the composer instead. Park on a real widget; a focused button is safe (DPG leaves ImGui's keyboard-nav activation off, so it ignores Space/Enter). A child window is not unfocusable in general — a *click* focuses one; it is `focus_item` that has no working spelling for it.

   See `dpg-notes.md` "Keyboard input", and `investigations/dpg-focus/` for the probes.
8. **Keyboard input has two non-obvious traps.** (a) *Stale key constants*: some `dpg.mvKey_*` values are pre-2.0 codes that no longer match what a handler receives in `app_data` — Page Up arrives as `517` not `mvKey_Prior` (266), Page Down `518` not `mvKey_Next` (267), plus LWin/RWin and Quote/Colon/Plus/Tilde. Comparing against the constant silently never matches; compare against the literal code. (b) *Same-frame dispatch is by keycode, not press order*: a keyless key-press handler fires once per key pressed that frame in ascending keycode order, so two near-simultaneous keys — where one handler mutates state another reads — interact as if the lower-keycode key came first (e.g. cherrypick's fast `C`+`Right` tagged the *next* image until navigation was deferred a frame). See `dpg-notes.md` "Keyboard input"; full table + reproduction in `briefs/reference/dpg-keycodes.md`.

## Architecture

### Server/Client Split
All ML inference in `raven/server/modules/` when Server is running:
- `tts.py` - Kokoro TTS with phoneme timestamps (needed for lipsync)
- `stt.py` - Whisper speech recognition
- `embeddings.py` - Sentence embeddings (currently snowflake-arctic; Nomic-embed-text v1.5 + vision v1.5 migration pending, bundled with Visualizer importer rework)
- `translate.py` - Neural machine translation
- `classify.py` - Sentiment/emotion classification, to control avatar's facial expression
- `sanitize.py` - Text cleanup (dehyphenation etc.)
- `natlang.py` - spaCy NLP analysis
- `websearch.py` - Web search tool for LLM
- `avatar.py`, `avatarutil.py`, `imagefx.py` - Avatar rendering pipeline

Client apps call Server via `raven/client/api.py`. Server can run on a different machine (trusted network only — no encryption). When Server isn't running, Visualizer's importer uses the `MaybeRemoteService` pattern to load models in-process, making the Visualizer deployable standalone.

### The Raven Way: three-layer module organization for ML-bearing subsystems

Each subsystem that has both a local (in-process) and remote (HTTP) mode follows the same three-layer pattern:

1. **`raven.common.<subsystem>`** — the actual implementation, pure library code, runs on whichever machine calls it. Framed as "explicit local mode", but the framing is incidental: this is where the work happens regardless of which process is doing it.
2. **`raven.server.modules.<subsystem>`** — the server-side subsystem module, delegating to `raven.common.<subsystem>`. Defines request handlers but not the routes themselves — routes and Flask plumbing live in `raven.server.app`, which wires each `modules.<subsystem>` handler onto its `/api/<subsystem>/...` URL. On the server, "local" means "server-side" — the server loads the same common-layer module the client would have loaded.
3. **`raven.client.api.<subsystem>`** — explicit remote mode. Client functions that make HTTP calls to the server. Mirrors the server's API surface one-for-one. In practice most subsystems are *inlined* directly into `raven.client.api` (they're small — a handful of request-sending functions). Only `tts` got large enough to warrant its own `raven.client.tts` module, re-exported through `raven.client.api`. Whether we should split the others out for symmetry with `raven.server.modules.*` is an open design question; inlined is the current reality.
4. **`raven.client.mayberemote.<Subsystem>`** — transparent remote/local mode. A class per subsystem; in remote mode it delegates to `raven.client.api.*`, in local mode it delegates to `raven.common.<subsystem>.*`. Callers don't need to know which mode is active.

Concrete example — `speech.tts`:

| Layer | Module | Role |
|---|---|---|
| Common (impl) | `raven.common.audio.speech.tts` | `prepare` / `prepare_cached` (TTSResult), `prepare_encoded_cached` (EncodedTTSResult); `encode`, `decode`, `synthesize`, `finalize_metadata` |
| Server module | `raven.server.modules.tts` | request handlers; uses common `synthesize_iter`, `audio_codec.encode` |
| Server app | `raven.server.app` | registers `/api/tts/...` routes onto the handlers |
| Client remote | `raven.client.tts`, re-exported via `raven.client.api` | `tts_prepare` / `tts_prepare_cached` (EncodedTTSResult), `tts_prepare_decoded_cached` (TTSResult), `tts_list_voices`, `tts_speak`, … → HTTP |
| Client mayberemote | `raven.client.mayberemote.TTS` | pure 2×2 dispatch, no cache state of its own; delegates to the cached bottom functions per (location, shape) |

**Caching strategy** (used if a subsystem needs it — currently only `tts`; other subsystems like `nlp`, `stt`, `embeddings` don't cache because their inputs are essentially never repeated in a session). When a subsystem has two natural output shapes (e.g. raw vs. encoded for TTS), caching lives in the bottom layers, not in mayberemote. Each of `common` and `client.remote` exposes:

- The "natural" cached shape for that side — `TTSResult` in common (local synthesizes float natively), `EncodedTTSResult` in client.remote (server returns encoded over the wire).
- The other shape, composed on top via `encode` / `decode`, also cached.

Mayberemote's `synthesize(format=...)` is then pure 2×2 dispatch — it picks one of the four cached bottom functions by `(location, shape)`. No cache state in the mayberemote class itself. This keeps the cache next to the engine (natural single-source-of-truth) while still giving the mayberemote caller the same "call it twice, second one is free" guarantee regardless of mode.

Same shape applies to `nlp` (`nlptools` ↔ `natlang`), `stt`, `embeddings`, `sanitize`, etc. — cross-check `raven.client.mayberemote` for the current set.

**Implications:**
- New ML work goes in `raven.common.<subsystem>` first. The server module and mayberemote wrapper come after and are thin shims.
- Playback / audio output stays in `raven.client.*` even when synthesis is local — the user is on the client machine, audio hardware is local by definition.
- `raven.client.tts.tts_prepare` and friends are **not** obsolete when `MaybeRemote.TTS` exists. They remain the explicit-remote path, used by `MaybeRemote` itself and by any app that wants to force remote mode.
- Data conversion at the boundary: in-process uses dataclasses (`TTSResult`, `WordTiming`), HTTP wire uses JSON-friendly dicts. Converter functions (`decode`/`encode`, `finalize_metadata`) live in the common layer — neither "local" nor "remote", they're shape conversions.
- Engine-agnostic data shapes live in their own module, separate from the engine wrapper. For TTS, `WordTiming`, `TTSSegment`, `TTSResult`, `EncodedTTSResult` are in `raven.common.audio.speech.datatypes`; only `TTSPipeline` (which holds a `kokoro.KPipeline`) stays in `raven.common.audio.speech.tts`. This lets consumers that only need the shapes (e.g. `lipsync`) import them without dragging in Kokoro/PyAV/huggingface_hub.

### Common Subsystems
- `raven/common/video/` - Postprocessor, upscaler (PyTorch Anime4K), colorspace conversions, cel compositor
- `raven/common/audio/` - Player, recorder, codec (PyAV streaming)
- `raven/common/gui/` - Custom DearPyGui widgets and the shared GUI vocabulary. Widgets: VU meter, messagebox, self-sizing tooltip (`tooltip.Tooltip` — for a caption whose text *changes* after it is built; one written once and never touched wants `dpg.add_tooltip`, and inside a modal that is the only option that works), thumbnail grid and table cursor (one keyboard cursor, two views), help card, xdot canvas. Frameworks and vocabulary: the GUI animation framework, `filedrop` (OS file drag-and-drop, installed with one call per app), `keyboardmark` (the colour and pulse that say where the keyboard is, so every widget drawing that mark agrees), `layout_math`, `fontsetup`. `api-inventory raven/common/gui/` is the current list; this one is a sample.

**"Every app does X" belongs in `raven/common/gui/`, as a component each app opts into with one call.** Not
a base class to inherit and not a copy per app: `filedrop.install(...)` in all six GUI apps and
`ThumbnailGrid` under Cherrypick's `TriageGrid` are the worked examples, and the planned `--qr` overlay
follows them. Extension is by subclass hooks or callbacks, and *policy stays with the app* — the grid takes
a list of visible indices and knows nothing about what admitted them, which is what lets both a triage tool
and a file dialog drive it.

Two things this buys, both observed rather than predicted: **one fix serves every consumer** (the grid's
windowing problem is now one bug in one place), and **the second consumer finds the API gaps the first one
hid** — wiring the grid into a second app surfaced three missing pieces in `raven.common.gui.animation`
within an hour, none of them grid-specific.

**The bar tracks how deep the code sits, not which package it is in.** `raven/common/` is the clearest
case, but the lower layers inside an app package are held to it too — Librarian's `chattree` and `hybridir`
are the worked examples. These are the parts worth reusing even if Raven itself turns out to have been the
wrong idea, so they are written to outlive it.

An awkward shape down there is not paid once. It is paid by every future caller, possibly in another
project and possibly years from now, and by the time anyone minds it is load-bearing and expensive to move.
An app can carry an oddity that its own code works around; a foundation cannot, because the workaround has
to be written again at each call site by someone who no longer remembers why.

So things that are ordinarily fine to defer are worth settling here while the code is in front of you: an
asymmetry between two paths through the same class, a parameter that means two things, a documented
contract one branch honours and another does not. The test is not whether it bites today — no live caller
may reach it — but whether a caller arriving later would have to learn the exception before they could use
the thing.

**App-level code is judged differently, and that is not a grudging allowance.** For the frontends, GUI code
especially, the criteria are that it works and that it stays reasonably maintainable. Holding a DPG
callback to the foundation's bar spends review on the wrong thing: it has one consumer, it is greppable,
and it can be rewritten the day the app changes shape.

### Vendored / adopted dependencies (`raven/vendor/`)

**`raven/vendor/` is *adopted* code — effectively ours to fix and extend, not pristine upstream snapshots.**
Each of these has already diverged from upstream with Raven-specific robustifications and features (see notes
below). So when you hit a bug *in* vendored code, fix it like any other Raven code (with the usual care for a
foreign-API layer — match the wrapped library's conventions). We may upstream a given change later, or not;
either way, treat the in-tree copy as the source of truth. Don't reach for "it's vendored, leave it alone."

- `tha3/` - Talking Head Anime 3 neural network (avatar animation). Switched `no_grad` → `inference_mode` in the
  hot paths for a few-percent speedup.
- `DearPyGui_Markdown/` - MD renderer, substantially robustified for Raven's background-threaded rendering
  (most call sites guarded with `guiutils.nonexistent_ok` / `does_item_exist` against DPG's lazy GC). Known
  remaining issue: the persistent render worker thread (`CallInNextFrame._worker`) doesn't participate in app
  shutdown — it keeps calling DPG (incl. `split_frame`) during teardown, which can segfault on a mid-boot close
  while a URL-heavy message is mid-render. Tracked in `TODO_DEFERRED.md` (fleet shutdown item).
- `file_dialog/` - File dialog, extended (sortable, animated OK button, click twice when overwriting).
- `anime4k/` - PyTorch port of Anime4K upscaler (extracts kernels from GLSL), slightly cleaned up.
- `kokoro_fastapi/` - Streaming audio writer for TTS over network.
- `IconsFontAwesome6.py` - Icon font codepoint table. Not the newest FontAwesome release, but **in sync with
  the fonts we ship**: measured 2026-07-30, every glyph in `fa-solid-900.ttf` has a constant and vice versa,
  so regenerating the header alone would gain nothing. Updating means new webfonts *and* a header
  regenerated from the matching `icons.yml`, as one change.

## Code Style
All new and modified code must follow `raven-style-guide.md` (in the project root). **Read the full guide before implementing a new app.** The summary below covers the most commonly needed conventions.

- Impure functional, Lispy (closures, `unpythonic` patterns)
- `unpythonic` pure-Python features are fair game. Currently used: `env` (namespace), `Timer` (benchmarking), `@call` (scoping), `box`/`unbox`, `sym`, `dyn`. Other features welcome where they improve clarity. **Do not** use the macro layer (`unpythonic.syntax`) or features that primarily serve as macro backends (e.g. `let` bindings — these exist mainly as a code-generation target for the macros. They *are* usable by hand, since the machinery runs on `env`, but clumsily: the body has to be a callable taking an `env` parameter, so the bindings read as `env.x` rather than as plain names).
- OOP where appropriate (GUI components, stateful objects)
- Config via Python modules (`config.py` files, not YAML/JSON)
- Type hints on all new and modified functions (public and internal). Existing untyped code can be left as-is unless you're already editing it. Use the modern spelling — `X | None` (not `Optional[X]`), `list[X]`/`dict[K, V]` (not `typing.List`/`Dict`); the codebase is mid-migration. Full guidance in `raven-style-guide.md` under "Type hints".
- `__all__`: all public symbols must be listed in `__all__` (PEP 8). Whether locally defined or re-exported, doesn't matter. This allows star-importing a module in a REPL to bring in its public API only.
- Imports: prefer `import module` + `module.func()` (dotted style) over `from module import func`. Makes it clear at the call site where a function comes from. For modules with ambiguous names, use an alias: `from ..common.gui import utils as guiutils`, `from ..server import config as server_config`.
- Naming: don't repeat the module name in function names. With dotted imports, `lanczos.resize()` reads better than `lanczos.lanczos_resize()`. The module provides the namespace.
- Docstrings: use raw backtick names (`` `func_name` ``), not RST cross-reference markup (`:meth:`, `:func:`). The codebase is read as raw code, not via Sphinx. Single space after sentence-ending period (European convention), not double.
- Log messages: prefix with the function name (or `ClassName.method_name` for methods), e.g. ``logger.warning("TriageManager.scan: ...")``. Python's logging already shows the module name, but not the function/method name.
  - Background tasks: include the instance name — ``logger.info(f"speak_task: instance {task_env.task_name}: message")``. This groups log output from the same task instance when multiple run concurrently.
  - Classes with multiple instances: include instance identification — a natural name attribute (e.g. ``instance '{self.base_dir.name}'``) or ``instance 0x{id(self):x}`` as fallback. Not needed for obvious singletons (e.g. GUI app classes).
  - Exceptions: use ``{type(exc)}: {exc}`` in log messages, not bare ``{exc}``. The type name is cheap insurance against uninformative `str()` output.
- Timers: use the right clock for the job. ``time.perf_counter()``/``perf_counter_ns()`` for benchmarks (highest resolution, monotonic). ``time.monotonic()``/``monotonic_ns()`` for elapsed time in app code (animation, polling, timeouts — immune to NTP adjustments). ``time.time()``/``time_ns()`` only for wall-clock timestamps that need epoch identity (chat message timestamps, persistent records).
- License DRY: the project-level `LICENSE.md` is the single source of truth (2-clause BSD). Don't repeat the license in individual module docstrings unless a module has a *different* license from the project default (e.g. AGPL for Server and Avatar pose editor).
  - **The second reason a module carries the line is an invitation, not a disambiguation.** "This module is licensed under the 2-clause BSD license, to facilitate integration anywhere" says *lift this if it is useful to you* — the same move that got the BSD network stack adopted everywhere. So it belongs on the pieces that are worth taking on their own (`common/gui/utils`, `common/gui/tooltip`), independently of whether anything about the license was unclear. The distinction the licenses draw is worth keeping in view when choosing: the GPL family keeps *code* free, BSD keeps *people* free to do what they like with it.
- Blank lines in code are paragraph breaks — insert when the topic changes, not mechanically (e.g. not "always before `return`").
- Properties: define as `def get_x(...) ... def set_x(...) ... x = property(fget=..., fset=..., doc=...)` instead of the `@property`/`@x.setter` decorator syntax.
- **Super init fires first.** A subclass `__init__` calls `super().__init__(...)` before doing anything of its own. The base is then free to declare every attribute it owns — including the ones a subclass will fill in — because nothing a subclass wrote can be sitting there yet.
  - **The failure this prevents is silent.** A subclass that assigns before the super call has that assignment overwritten by the base's own declaration, and nothing reports it: the attribute simply holds the base's default afterwards. What surfaces is whatever reads it next, somewhere else — a lookup for `None`, a render of nothing. (Live case 2026-08-27: `DPGStreamingChatMessage` set `node_id` before the super call and then asked the datastore for the payload of node `None`.)
- DPG string tags: any line that mentions a DPG string tag must carry a ``# tag`` comment (for greppability across the codebase). The only exception is a line that already passes ``tag=...`` as a keyword argument — the word "tag" is right there in the parameter name, so the comment would be redundant. This applies to any API that takes a DPG tag/alias: ``dpg.add_*``, ``dpg.hide_item``, ``dpg.show_item``, ``dpg.set_value``, ``dpg.set_item_pos``, ``dpg.get_item_rect_size``, ``dpg.does_item_exist``, ``guiutils.wait_for_resize``, etc. If the line already has a trailing comment, keep both: ``dpg.show_item("foo_window")  # tag  # existing note``.
- **Changing Raven's own library code is fine when it yields a better design.** `raven.common.*` and friends are first-party: if a caller needs a shape the library does not offer, prefer improving the library over working around it at the call site. Much of the caution that would apply in a corporate multi-team setting — freeze the interface, add an adapter, coordinate with owners — has no counterpart here; this is a solo-maintained project, and every consumer is in the same tree and greppable. The vendored-code note below says the same thing about `raven/vendor/`, and first-party code is the easier case, not the harder one. Not a licence to churn: the test is whether the design comes out cleaner, not whether the change is possible.
- Contract-style preconditions/postconditions would be useful, but mostly not implemented yet

## Key Patterns

### DearPyGui App Structure
See `dpg-notes.md` "Raven DPG app structure" section for layout patterns, startup sequence, background work, thread safety, DPG item management, and texture handling.

**An `app.py` is an OS entry point, not a library module.** Most are wiring skeletons: parse the command line, build the GUI, instantiate the objects, run the DPG render loop (the manual `while dpg.is_dearpygui_running()` form, so the animator can be ticked). Anything that would be worth calling from elsewhere — or worth testing — belongs in another module, beside the thing it operates on.

Two consequences worth stating, because both were learned the expensive way:

- **Behaviour that lands in `app.py` is untested behaviour.** An `app.py` runs `parser.parse_args()` at module scope, so importing it under pytest fails on pytest's own argv. That is a fine property for an entry point and a fatal one for anything else: a function put there cannot be exercised at all. `raven.cherrypick.preload.donate_outgoing_image` lives where it does for this reason.
- **App-level state that mirrors a component's state will drift.** A module-level "the current image is N" tracked beside the widget that holds it is correct only while the app is the sole writer, and it fails silently the moment anything else drives that widget. Ask the component instead, or give it the datum to carry (`ImageView.image_key` is the worked example). The corruption this produced outlived the session that caused it, and gave no sign of itself anywhere near where it happened.

### Avatar Lipsync
TTS (Kokoro) provides timestamped phonemes → mapped to mouth morphs → THA3 animator. Audio playback occurs on the client side.
This coupling limits TTS engine choices (most don't expose timestamped phoneme data).

## Current State

### Well-structured (target style)
- `raven/librarian/` - Clean module separation (~20k lines across 18 modules, measured 2026-08-24). Note it has outgrown the per-module guideline below in several places — `chat_controller.py` is ~4.0k lines and `llmclient.py` ~2.8k — without losing the layering, which is the property that made it the target style. Size is a smell here, not a verdict. See `raven/librarian/CLAUDE.md` for the layer map.

### Needs refactoring

Target ~700 lines per module as a guideline, not a hard limit — some modules can be longer when appropriate (e.g. lots of simple related code).

**The number came from the macro projects, and it does not transfer unchanged.** `mcpyrate` and
`unpythonic` are dense in a way Raven mostly is not: there, 700 lines is plenty because the lines are
Kolmogorov-hard, and a longer module really is a sign that something wants splitting. Raven's frontends
and tools are the opposite — long stretches of simple related code, and prose. So a module over the
guideline is a prompt to look, not a finding: ask whether the *layering* has gone, which is the property
that made `raven/librarian/` the target style, rather than whether the line count has. `chat_controller.py`
is ~5k lines and keeps its layers, so it is fine where it is; `raven/papers/deduplicate.py` is ~1.1k of
which a third is docstrings, and splitting it would move prose around rather than simplify anything.
(Juha, 2026-08-28, settling exactly this question about the deduplicator.)

- `raven/visualizer/app.py` - ~1.9k lines. The split into `info_panel`, `selection`, `plotter`, `annotation`, `word_cloud`, `entry_renderer` and `app_state` has landed; what remains is ordinary size rather than a god object. See `raven/visualizer/CLAUDE.md` for the module map.
- `raven/visualizer/info_panel.py` - ~1.5k lines, the largest of the extracted modules; a candidate for further splitting, but not urgent.
- `raven/visualizer/importer.py` - ~1.3k lines, pipeline architecture, lower priority but could benefit from stage separation

### Test coverage

88 test modules as of 2026-08-20, ~2750 tests (`pytest -m "not ml"`). Library and utility code is broadly
covered, and the GUI layer is no longer the hole it was; what remains untested is the Visualizer and the
large DPG frontends.

- **`common/`** — bgtask, datastorelock, deviceinfo, docextract, filelisting, logsetup, netutil, nlptools, numutils, readcsv, running_average, smoothvalue, stringmaps, utils; `text/` (normalize, speakable); `audio/` (codec, resample, utils) and `audio/speech/` (tts, stt, lipsync, and a TTS→STT round trip); `image/` (codec, lanczos, utils); `video/` (colorspace, compositor, postprocessor, upscaler); `gui/` (animation, filedrop, filegrid, fontsetup, gridnav, helpcard, layout_math, messagebox, tablecursor, thumbnailgrid, tileicons, tooltip, utils, a characterization of DPG's own focus semantics, and all of `xdotwidget/`).
- **`librarian/`** — agent, appstate, chat_controller, chattree, chatutil, cleanup, hybridir, imagestore, indexer, llmclient, scaffold, sidecarstore, textfilestore.
- **Elsewhere** — `vendor/file_dialog` (the largest single module's worth, at ~175 tests), `client/` (api, mayberemote), `papers/*`, `cherrypick/*`, `server/webfetch`, `xdot_viewer/dot_utils`.

What is **not** covered:

- **Visualizer has zero tests.** Still the biggest gap, and the refactor that motivated writing them
  landed without them — so what they would pin now is the new module boundaries rather than a rewrite
  in flight.
- **The DPG frontends**: librarian `app` and `cleanup_dialog`, and every Visualizer GUI module. **Not
  because DPG resists testing** — it runs without a mapped window, and `common/gui/tests/` already drives a
  real context with an unmapped viewport. See `dpg-notes.md`, "Testing DPG code". The barrier is that nobody
  has written them for the large frontend modules, which is a different and more tractable problem than
  "untestable".
  - **`vendor/file_dialog` is the proof, and it is a large frontend module by any measure**: 175 tests over
    a 2900-line DPG widget, covering its listing rules, its keyboard, its colours and its promised target.
    They are what made a month of keyboard work safe to do. Whatever is in the way for `app.py`, it is not
    the toolkit.
  - **And they run in CI, on all three platforms**, since 2026-08-12: `dearpygui` and `mistletoe` are in
    `requirements-ci.txt`, which brought 57 tests that had only ever run on a dev machine into every push
    (2090 → 2147 passing). The open question was whether GLFW could get a context on a runner with no
    display server; it can, on ubuntu, macOS and Windows alike. Tests that *map* a window are a separate
    case and still stay out — they carry the `gui` marker and need `--run-gui`.
  - **What is still dev-machine-only is `test_chat_controller.py`**, and not for a GUI reason: importing
    `chat_controller` reaches spaCy through the avatar client, so the module skips on itself. Same
    anti-pattern as the `llmclient` one that lazy `api.initialize` fixed, one layer up.
  - Splitting an operation from its dialog is what makes the operation testable at all; `cleanup.py` /
    `cleanup_dialog.py` is the worked example, and its module docstring explains why.
- **`librarian/minichat`** — the readline REPL, and the odd one out: no DPG anywhere in it, so none of the
  above applies. It is a terminal app with the same backend as the GUI, which makes it the *cheapest*
  frontend to test rather than the hardest. Untested because nobody has, not because anything is in the way.
- `config.py` modules, which are configuration-as-code and carry local overrides anyway.

## Upstream warning noise in `pytest raven/`

The pytest summary normally shows a handful of `DeprecationWarning`/`UserWarning` captures. They look alarming but are **all upstream** and not fixable from raven's side. Catalogued here so we don't re-investigate each time. (This subsection is temporary; eventually factor it out to a dedicated `.md`.)

- **`DeprecationWarning: builtin type SwigPyPacked/SwigPyObject has no __module__ attribute`** — from `sentencepiece`, whose Python wrapper is SWIG-generated. Verify with `find .venv -name "*.so" | xargs -I{} sh -c 'if strings "{}" 2>/dev/null | grep -q swigvarlink; then echo "{}"; fi'`. Python 3.12+ warns when built-in types don't set `__module__`; SWIG's generated helper types (`SwigPyPacked`, `SwigPyObject`, `swigvarlink`) pre-date that convention. Upstream fix has to happen in the SWIG project itself; every SWIG-wrapped library inherits the warning. `sentencepiece` is a transitive dep via NLP tokenizers (`transformers`, `kokoro`'s phonemizer chain).

- **`DeprecationWarning: torch.jit.script is deprecated`** — from `transformers` (HuggingFace). Many of its model files use `@torch.jit.script` as a decorator at module load: `deberta`, `deberta_v2`, `gpt_bigcode`, `zoedepth`, `sew_d`, `vits`, `sam3_video`, … When raven's tests import `sentence-transformers` (via `raven.librarian.hybridir` for embeddings), transformers eagerly loads these model modules and the decorators fire. Verify with `grep -rn "@torch.jit.script" .venv/lib/python3.12/site-packages/transformers/`. Upstream fix waits on HuggingFace migrating these decorators to `torch.compile`/`torch.export`. Raven's own code no longer calls `torch.jit.script`.

- **`UserWarning: pkg_resources is deprecated as an API`** — from `pygame` 2.6.1 (currently the latest on PyPI). Its `pkgdata.py` still imports from `pkg_resources`. Upstream fix waits on a pygame release that stops using it. Pinning `Setuptools<81` would silence it but isn't worth the collateral; just wait for the next pygame.

- **`DeprecationWarning: Deprecated in 0.9.0: WordPiece.__init__ will not create from files anymore, try WordPiece.from_file instead`** — from `tokenizers` (the Rust library), raised by `transformers`' own `models/bert/tokenization_bert.py`, which builds a `WordPiece` from an in-memory vocab dict. Seen via `test_load_classifier_caches`, which loads the sentiment classifier. No Raven code constructs a `WordPiece`, so there is nothing here to migrate; the fix is HuggingFace's.
  - **This is not a signal that BERT is going away**, which is the natural worry and the reason this entry is longer than the others. Checked 2026-08-25 against transformers 5.15.1: `transformers/models/deprecated/` holds nothing but its `__init__.py`, and `models/bert/` ships *both* `tokenization_bert.py` and `tokenization_bert_legacy.py` — the shape of a model being carried through an API transition, not retired.
  - The comparable breakage has already happened and is already absorbed: transformers 5.0 **removed** `TranslationPipeline`, and `nlptools._Translator` replaces it with `AutoModelForSeq2SeqLM` plus `generate`. Its docstring records that, and it is the pattern to reach for if another pipeline task goes the same way.

### Fixed locally (for reference)

- **`RuntimeWarning: divide by zero encountered in divide` — `raven/common/numutils.py:psi()`**: the mollifier helper computes `np.exp(-1.0 / x**m) * (x > 0.0)` and relies on the `(x > 0.0)` mask to zero the divide-by-zero. A previous attempt used `warnings.filterwarnings(..., module="__main__")` which silently failed (numpy emits the warning from its own internal module, not `__main__`). Correct fix: `with np.errstate(divide='ignore', invalid='ignore'):` — numpy's own mechanism for suppressing float-error warnings within a dynamic extent.

## LLM Backend
Any OpenAI-compatible API. **LM Studio is what the team uses**, so it is the one to assume when a question
turns on backend behavior. text-generation-webui (oobabooga) is also supported — `llmclient` detects the
flavor and adapts — but that path has not been re-validated against a recent ooba release, so treat its
quirk handling as untested rather than known-good.

Model choice tracks whatever Qwen currently ships: the **VL line was folded into the main line at Qwen3.5**,
so a current Qwen3.5/3.6 release covers both text and vision, and the separate `-VL` builds are historical.
Size follows VRAM in the usual way — a ~30B MoE wants 24 GB or more, a ~4B fits in 8 GB.

## Known Issues / TODOs
- Visualizer: the `app.py` split has landed (see `raven/visualizer/CLAUDE.md` for the module map). What remains is ordinary tidying — `info_panel.py` at ~1.5k lines is the next split candidate, and `importer.py` could use stage separation — not a god-object rescue
- Visualizer has zero tests (the librarian gaps this used to list — `scaffold`, `appstate`, `llmclient` — are all covered now)
- DearPyGui_Markdown decorations land in the wrong place — now tracked in `TODO_DEFERRED.md`, "Markdown
  decorations are placed by measuring the text". All five of them are drawlists positioned from a
  measurement that nothing waits for. The URL *colour* sitting one character off is filed there as probably
  a different fault (segmentation rather than placement), and possibly already fixed
- Hindsight integration pending (PDM dependency conflicts; likely separate container with optional backend, keeping BM25+vector backend as primary)
- TTS engine expansion limited by phoneme timestamp requirement
- Many `# TODO: DRY duplicate definitions for labels` scattered through Visualizer `app.py`
- Annotation tooltip help section rebuilt every time (could be static with show/hide)
- `_update_info_panel` race condition: current item highlight sometimes doesn't update immediately after selection change
- Search match scrolling race condition: hammering the button can error out (`info_panel.py:670`/`685` — the code moved there in the refactor; the old `app.py:2978` pointer was past EOF. Not re-verified since the move, so it may or may not survive)
- XDot viewer: GraphViz `--concentrate` produces near-miss edge endpoints (0.02–0.09 graph units off) at edge split/merge points, visible as small gaps at high zoom. This is a GraphViz precision issue in the xdot data, not a rendering bug.
