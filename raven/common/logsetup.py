"""Single owner of root-logger configuration for Raven app entry points.

Each Raven CLI/GUI app calls `configure(...)` exactly once per session. Two
placement patterns are in use:

- **Before heavy imports** — used by the GUI/server family (`raven-visualizer`,
  `raven-librarian`, `raven-server`, `raven-importer`, `raven-minichat`,
  `raven-xdot-viewer`, `raven-cherrypick`, `raven-conference-timer`,
  `raven-avatar-pose-editor`, `raven-avatar-settings-editor`). Captures
  import-time records from heavy ML deps in `--log` output.
- **Inside `main()` after heavy imports** — used by the smaller bibliography
  tools (`raven-pdf2bib`, `raven-wos2bib`, `raven-csv2bib`). Import-time
  records from third parties go to Python's `lastResort` (WARNING+ only),
  which is fine for these tools because they curate via `allow=[__name__]`
  anyway and only care about their own application-level logging.

Either way, the 65 modules across `raven/` that historically called
`logging.basicConfig(level=logging.INFO)` at import time should do only
``logger = logging.getLogger(__name__)`` — they don't own root configuration.

See `briefs/logsetup-fleet-wide.md` for the design rationale (allowlist vs
denylist, why filtering is per-handler not per-logger, dual-use module split,
and the self-healing logfile trick that survives `flair`-class third-party
clobbering).
"""

__all__ = ["DEFAULT_FORMAT", "configure"]

import logging
import pathlib
import sys
from typing import Iterable, Optional

from unpythonic import UnionFilter

logger = logging.getLogger(__name__)

DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

def configure(*,
              level: int = logging.INFO,
              logfile: Optional[str] = None,
              allow: Optional[Iterable[str]] = None,
              fmt: Optional[str] = None) -> None:
    """Configure the root logger for a Raven app entry point.

    Call exactly once per app session. Place the call before heavy imports
    if you want their import-time records captured by `--log` and the
    configured handlers (the GUI/server family does this); place it inside
    `main()` after heavy imports if you only care about your own
    application-level logging (the bibliography tools do this — records
    that emit before `configure(...)` runs go through Python's `lastResort`
    handler, visible at WARNING+, dropped silently below). The module
    docstring lists which apps use which placement.

    Idempotent — uses ``force=True`` so re-entry from tests is safe (the
    previous call's handlers are removed and closed before new ones install).

    `level`: root logger level (e.g. `logging.INFO`, `logging.DEBUG`).
    `logfile`: if given, also write to this file. Truncated at configure
               time (one fresh log per session), then opened in append mode
               so the handler self-heals from `logging.shutdown()` invoked
               by hostile third-party imports — see "Self-healing logfile"
               below.
    `allow`: if given, only emit records whose logger name starts with one
             of these prefixes. `None` means emit everything. Implemented as
             one `unpythonic.UnionFilter` of `logging.Filter` instances,
             attached to every root handler.
    `fmt`: logging format string. Defaults to `DEFAULT_FORMAT`
           (timestamp + level + name + message).

    Raises `ValueError` if `allow` is given but empty (an empty allowlist
    would silently drop every record — almost certainly a mistake; use
    `allow=None` instead to emit everything).

    Why filtering is per-handler, not per-logger: Python's propagation rules
    consult only ancestor *handlers'* filters during propagation, not
    ancestor *loggers'* filters. A filter on root would only catch records
    logged directly to root (rare in practice). Handlers are the only level
    that catches propagated records, which is ~all Raven records.

    Why an allowlist, not a denylist: in an ML-heavy research app the set
    of noisy transitive dependencies is unpredictable. An allowlist is
    deterministic against new noisy deps; a denylist would let them leak in
    until the user notices and adds them. The cost of allowlist filtering
    is a handful of string-prefix checks per record, negligible compared to
    actually formatting and writing the record.

    Self-healing logfile: some third-party libraries (notably `flair`) call
    `logging.config.dictConfig` on import, which invokes
    `logging.shutdown()` on every existing handler. The closed handlers
    stay in `logging.root.handlers` (dictConfig doesn't remove them from
    individual loggers), so records keep reaching them — and
    `FileHandler.emit` reopens its stream on the next write *if*
    ``mode != 'w'`` (see CPython's `FileHandler.emit`). Opening the file
    in append mode after an explicit pre-truncate gives both fresh-per-
    session semantics and survival across hostile shutdowns.

    Convention: only entry-point modules call this. Library modules
    (including dual-use ones like `raven.visualizer.importer` when imported
    by the GUI app) must not — they would clobber the host's configuration.
    See `raven.visualizer.importer_cli` for the dual-use split pattern.
    """
    if allow is not None:
        # Materialize once: a generator would be consumed by the empty-check
        # below before the UnionFilter constructor could see it.
        allow = list(allow)
        if not allow:
            raise ValueError("logsetup.configure: `allow` must be either `None` "
                             "(emit everything) or a non-empty iterable of "
                             "logger-name prefixes; got an empty iterable, which "
                             "would silently drop every record.")

    handlers = [logging.StreamHandler(sys.stderr)]
    if logfile is not None:
        # Normalize: expand `~`, resolve symlinks and `..`. Matches the path-handling
        # convention used elsewhere in Raven (pathlib `expanduser().resolve()`).
        logfile = str(pathlib.Path(logfile).expanduser().resolve())
        with open(logfile, "w", encoding="utf-8"):
            pass  # truncate to start fresh; FileHandler then opens in append mode
        handlers.append(logging.FileHandler(logfile, mode="a", encoding="utf-8"))

    logging.basicConfig(level=level,
                        handlers=handlers,
                        format=fmt or DEFAULT_FORMAT,
                        force=True)

    if allow is not None:
        union = UnionFilter(*(logging.Filter(name) for name in allow))
        for handler in logging.root.handlers:
            handler.addFilter(union)
