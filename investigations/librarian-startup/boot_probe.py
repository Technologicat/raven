"""Run Raven-librarian with timers around its two slow startup steps. Prints a PROBE line per step, to stderr.

Usage: python boot_probe.py [raven-librarian options], e.g. `--log-level INFO --log boot.log`.

Each line gives the step's wall time, how much of it was spent in the cycle collector (with the number of
collections per generation), and which threads were alive when it began. Importing `raven.librarian.app`
runs the app, so this maps a window and takes keyboard focus; close it normally to end the run. The patches
are applied before that import, and `hybridir` is therefore imported early: the app's own "Libraries loaded"
figure is not comparable with an unpatched run's.
"""

import gc
import sys
import threading
import time

from raven.librarian import hybridir

gc_seconds = [0.0]
gc_collections = [0, 0, 0]
_gc_started = [0.0]

def _on_gc(phase: str, info: dict) -> None:
    if phase == "start":
        _gc_started[0] = time.perf_counter()
    else:
        gc_seconds[0] += time.perf_counter() - _gc_started[0]
        gc_collections[info["generation"]] += 1

gc.callbacks.append(_on_gc)

def timed(name, fn):
    def wrapper(*args, **kwargs):
        threads = sorted(thread.name for thread in threading.enumerate())
        gc_before, collections_before, t0 = gc_seconds[0], list(gc_collections), time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            collections = [after - before for after, before in zip(gc_collections, collections_before)]
            print(f"PROBE {name}: {dt:.2f} s wall, {gc_seconds[0] - gc_before:.2f} s in gc "
                  f"(collections by generation {collections}); {len(threads)} threads: {threads}",
                  file=sys.stderr, flush=True)
    return wrapper

hybridir.HybridIR._load_datastore = timed("_load_datastore", hybridir.HybridIR._load_datastore)
hybridir.HybridIRFileSystemEventHandler.rescan = timed("rescan", hybridir.HybridIRFileSystemEventHandler.rescan)
print(f"PROBE gc thresholds {gc.get_threshold()}, gc enabled {gc.isenabled()}", file=sys.stderr, flush=True)

sys.argv = ["raven-librarian"] + sys.argv[1:]
import raven.librarian.app  # noqa: E402, F401 -- importing it runs the app
