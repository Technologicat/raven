"""Single owner of root-logger configuration for Raven app entry points.

Each Raven CLI/GUI app calls `configure(...)` exactly once per session, before
heavy imports. The 65 modules across `raven/` that historically called
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

    Call exactly once per app session, before heavy imports. Idempotent —
    uses ``force=True`` so re-entry from tests is safe (the previous call's
    handlers are removed and closed before new ones install).

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
