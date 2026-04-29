# Fleet-wide logging setup

*2026-04-29, Claude Opus 4.7 — design brief, pre-implementation*

## Why

Two TODO_DEFERRED items, both surfaced during INDEXING-indicator debugging on 2026-04-28:

1. **`--log <file>` option for Raven apps** — uniform flag mirroring stderr to a file, so users can capture session logs for bug reports without redirecting their terminal. Especially useful for GUI apps where the originating terminal is often a side window the user has already closed.
2. **Logging is misconfigured fleet-wide** — 65 modules across `raven/` call `logging.basicConfig(level=logging.INFO)` at import time. The first import wins (subsequent `basicConfig` calls are no-ops once root has handlers), making the configuration order-dependent and fragile. Bumping a logger to `DEBUG` from the entry point is silently undone if any later code touches root.

These are one task: `--log` only makes sense once root configuration has a single owner.

## What's actually broken

Every Raven module currently does at module top:

```python
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

`basicConfig` is documented as a no-op once root has any handler, so most of the 65 calls don't *do* anything. They make the configuration **first-import-wins**, depending on import order. Even when the current behavior happens to be fine, the shape is wrong.

Visualizer additionally installs an allowlist filter at `raven/visualizer/app.py:71-85`:

```python
for handler in logging.root.handlers:
    handler.addFilter(UnionFilter(logging.Filter(__name__),
                                  logging.Filter("raven.client.mayberemote"),
                                  logging.Filter("raven.client.api"),
                                  ...))
```

A curated list of ~13 modules whose log output is signal; everything else (including some Raven modules) gets suppressed. The pattern works but is duplicated nowhere — every other app gets all the noise.

## Design

### `raven/common/logsetup.py`

```python
DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

def configure(*,
              level: int = logging.INFO,
              logfile: Optional[str] = None,
              allow: Optional[Iterable[str]] = None,
              fmt: Optional[str] = None) -> None:
    """Configure the root logger for a Raven app entry point.

    Call exactly once per app session, before heavy imports.
    Idempotent — uses force=True so re-entry from tests is safe.

    level:   root logger level (e.g. logging.INFO, logging.DEBUG).
    logfile: if given, also write to this file (overwrite mode, UTF-8).
    allow:   if given, only emit records whose logger name starts with one
             of these prefixes. None means emit everything. Implemented as
             an `unpythonic.UnionFilter` of `logging.Filter` instances,
             attached to every root handler (see "Allowlist via per-handler
             UnionFilter" below for why per-handler).
    fmt:     logging format string. Defaults to a timestamped format.
    """
```

### Allowlist via per-handler `unpythonic.UnionFilter`

For Raven's research-app context (many noisy ML deps), an allowlist beats the conventional denylist (`logging.getLogger("noisy.lib").setLevel(WARNING)`):

- **Allowlist (chosen)**: deterministic against unpredictable transitive logging — a new noisy dep doesn't leak in. Cost: noisy records still get created and propagated; only output is suppressed (a handful of string-prefix checks per record).
- **Denylist**: records never created (cheaper), but a new noisy dep leaks in until you add it.

Filter must attach to handlers, not to the root logger: Python's propagation rules consult only ancestor *handlers'* filters during propagation, not ancestor *loggers'* filters. A filter on root would only catch records logged directly to root (rare).

One `unpythonic.UnionFilter` instance shared across both handlers (stderr + optional file) keeps the cost minimal.

### Entry-point shape

Each entry point's top-of-file becomes:

```python
# logging
import argparse
import logging

parser = argparse.ArgumentParser(...)
parser.add_argument("--log", metavar="PATH", help="Mirror stderr log to this file (overwritten each run).")
parser.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    help="Root logger level.")
args = parser.parse_args()

from raven.common import logsetup
logsetup.configure(
    level=getattr(logging, args.log_level),
    logfile=args.log,
    allow=[__name__, ...],  # per-app curated list, or None
)

logger = logging.getLogger(__name__)

# version message
# heavy imports (with "Loading libraries..." status)
# everything else
```

This matches Raven's existing top-of-file convention for apps:

> `<logging>, <version message>, <imports (possibly with a "Loading libraries..." message, to show that the app has not hung)>, <everything else>`

For library modules, the `<logging>` block collapses to just `logger = logging.getLogger(__name__)`.

### Per-app allowlists

- **Visualizer**: ports its existing curated list verbatim.
- **All others**: `allow=None` initially. Same effective behavior as today (root-level INFO emits everything from `raven.*` plus whatever third-party libs log at INFO+). Curate per-app later as noise becomes visible.

### Dual-use modules: CLI shell split

Most entry points are pure CLI/GUI apps — nothing imports them as a library. They get the simple top-of-file pattern shown above.

The exception is `raven.visualizer.importer`, which is both a library (imported by `raven.visualizer.app` for the GUI's *Import BibTeX* window) and a CLI tool (`raven-importer`). Putting `configure()` at the top of `importer.py` would clobber the GUI's already-configured logging when the GUI imports importer. Putting `configure()` only inside `main()` would mean heavy imports run before logging is configured, losing import-time INFO records and the "Loading libraries..." status banner.

Resolution: extract the CLI shell into a new `raven/visualizer/importer_cli.py`:

```python
# raven/visualizer/importer_cli.py — CLI entry point only

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--log", ...)
    parser.add_argument("--log-level", ...)
    # ... existing importer CLI flags
    args = parser.parse_args()

    import logging
    from raven.common import logsetup
    logsetup.configure(level=getattr(logging, args.log_level),
                       logfile=args.log,
                       allow=[...])

    logger = logging.getLogger(__name__)
    from .. import __version__
    logger.info(f"Raven-importer version {__version__} loading.")
    logger.info("Loading libraries...")

    from raven.visualizer import importer  # heavy imports happen here, AFTER configure()
    importer.run_cli(args)  # or whatever the dispatch shape is
```

`pyproject.toml`: change `raven-importer = "raven.visualizer.importer:main"` → `... importer_cli:main"`.

`raven/visualizer/importer.py` becomes pure library: heavy imports at module top, `logger = logging.getLogger(__name__)`, no `main()`, no argparse, no logsetup call. Both `importer_cli.py` and `raven.visualizer.app` call into the library API.

This sidesteps the import-ordering problem entirely: by the time `importer.py` is imported (in either CLI or GUI mode), `logsetup.configure(...)` has already run.

## Plan

1. Write `raven/common/logsetup.py`. Docstring covers: call-once convention, allowlist vs denylist rationale, why filter must be per-handler not per-logger.
2. Port `raven/visualizer/app.py` as the reference: argparse first (`--log`, `--log-level`), then `logsetup.configure(...)` with the existing allowlist verbatim, then heavy imports.
3. Extract `raven/visualizer/importer_cli.py` from `importer.py`'s `main()`; update `pyproject.toml` entry point. `importer.py` becomes pure library.
4. Sweep the 65 module-body `basicConfig` calls. Drop the line; keep `logger = logging.getLogger(__name__)`.
5. Add `--log` / `--log-level` to the remaining entry points (`raven-{librarian,server,minichat,xdot-viewer,cherrypick,conference-timer,avatar-pose-editor,avatar-settings-editor}`) with `allow=None`.
6. Smoke-test all ten apps: launch each briefly with and without `--log`, confirm stderr looks right and the file mirrors it. Catch any case where heavy imports run before `logsetup.configure` due to subtle import ordering.
7. CHANGELOG entries; remove the two corresponding sections from `TODO_DEFERRED.md`.
8. Lint, branch off `main`, commit, push, open PR for CI gating.

## Open considerations

- **Idempotency guard.** `force=True` makes `configure()` safe to call multiple times (second call wins, first call's handlers torn down cleanly). We could add a `_called_already` warning, but it would fight tests that legitimately reconfigure. Skipped — trust the convention.
- **Loggers with `propagate=False`.** Records bypass root entirely, so root-handler filters don't see them. None of Raven's deps do this today, and the existing Visualizer setup has the same limitation, so we're not regressing.

## Self-healing logfile (flair survival)

Discovered during the first smoke test: `flair` (transitively imported by Visualizer via `raven.common.nlptools`) calls `logging.config.dictConfig` on import. With `incremental=False` (the default), Python's dictConfig invokes `logging.shutdown()` on every handler in `logging._handlerList` — closing the streams. The handlers stay attached to `root.handlers` (dictConfig only clears the handler-name-keyed dict and the handler list, not individual loggers' references), so records keep being routed to them. Per-handler behavior after the close:

- `StreamHandler` keeps working: its `close()` doesn't actually close `sys.stderr` (it doesn't own the stream).
- `FileHandler` opened in `mode='w'` is dead: `emit()`'s reopen guard (`mode != 'w' or not _closed`) refuses to re-open a `'w'`-mode file after close.
- `FileHandler` opened in `mode='a'` self-heals: the same guard short-circuits to True, `_open()` reopens the file in append mode, and the next emit succeeds.

Resolution: `configure(logfile=...)` truncates the file once via a bare `open(path, 'w').close()`, then opens the FileHandler in `mode='a'`. Fresh-per-session semantics preserved; full session captured even after flair clobbers `_handlerList`. No buffering, no `reapply()` API needed. Verified against a real Visualizer launch — the entire session including pre- and post-flair records reaches the logfile.

This pattern works against `flair`-class clobbering (which closes handlers but doesn't remove them from individual loggers). A more aggressive third-party library that explicitly cleared `root.handlers` would still defeat us; none of Raven's current deps do that.
