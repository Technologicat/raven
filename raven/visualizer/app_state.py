"""Shared app-level state for the Visualizer.

During the ongoing refactor (splitting `app.py` into focused submodules), this
module holds what used to be module-level globals in `app.py` that need to be
read or written by multiple extracted submodules. A central namespace beats
`from .app import name` / circular-import gymnastics, and matches the Zen of
Python's *explicit is better than implicit*: every cross-module access is
`app_state.foo`, not a bare name whose origin is ambiguous.

Entries get added as each submodule extraction surfaces a new cross-module
dependency. Entries leave once a later refactor pass wraps related state into
a class whose instance can live here under a single name.

Module-local state (state that only one submodule needs to read or write)
stays in that submodule as module-level variables — not here.
"""

__all__ = ["app_state"]

from unpythonic.env import env

app_state = env()
