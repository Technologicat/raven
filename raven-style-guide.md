# The Raven coding style

Derived from the [unpythonic style guide](unpythonic-style-guide.md), adapted for the Raven codebase. Documents actual patterns observed in the code.

## Philosophy

Raven inherits unpythonic's governing principle — *"find pythonic ways to do unpythonic things"* — but sits at a different point on the spectrum. Where unpythonic is a language extension library with deep metaprogramming, Raven is an application project that *uses* unpythonic idioms pragmatically:

- **Be correct.** Handle edge cases. Report errors clearly.
- **Be concise but readable.** No code golf, but no unnecessary ceremony either.
- **Closures over classes** when the state is simple. Classes when the state or interface is complex.
- **Keep it working.** Raven is built quickly and pragmatically. Polish where it matters (architecture, user-facing behavior), tolerate roughness elsewhere.
- **No macros.** Raven uses `mcpyrate` only for its `colorizer` utility. All logic is pure Python.
- **No currying.** `unpythonic.curry` is not used. Standard parameter ordering applies.

## Module structure

Modules follow a consistent layout:

```python
"""Short module description.

Longer explanation where useful.
"""

__all__ = ["public_name1", "public_name2"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# stdlib imports
import collections
import threading
from typing import Callable, Dict, List, Optional

# third-party imports
import numpy as np

# unpythonic imports
from unpythonic import sym, box, unbox
from unpythonic.env import env

# internal imports (relative)
from ..common import bgtask
from ..common import utils as common_utils
from . import config as librarian_config
```

Key points:

- `__all__` is mandatory and placed immediately after the module docstring, before imports. Populated explicitly.
- Logging setup immediately after `__all__`. The three-line `logging.basicConfig` / `logger = ...` pattern is standard.
- Imports use `from ... import ...` style (not bare `import ...`), except for large namespaces like `numpy`, `torch`, `dearpygui`, and `json`.
- Internal imports use relative paths (`.module`, `..module`).
- No star imports.
- `as` renaming is used sparingly and consistently: `env as envcls` (when `env` is also a parameter name), `config as librarian_config` (disambiguation), `utils as common_utils` / `utils as guiutils` (disambiguation).

### Application entry modules

For top-level app modules (`visualizer/app.py`, `server/app.py`, `librarian/app.py`), a heavier startup pattern is used:

```python
logger.info(f"App-name version {__version__} starting.")

logger.info("Loading libraries...")
from unpythonic import timer
with timer() as tim:
    import argparse
    import threading
    # ... all remaining imports ...
logger.info(f"Libraries loaded in {tim.dt:0.6g}s.")
```

This wraps all imports in a `timer` block to measure and log startup time. Imports go *inside* the `with` block.

## Module size

- Target ~100–500 SLOC per module (docstrings, comments, and blanks don't count).
- Rough upper bound ~800 lines total for library modules.
- App modules (`app.py`) are currently larger — `visualizer/app.py` is 4400+ lines and is the primary refactoring target.
- The librarian component (~8000 lines across 10 modules) is the target architecture: clean layered design with each module at ~300–800 lines.

## Naming

- **Functions**: `lowercase_with_underscores`.
- **Classes**: `PascalCase`, including exception classes.
- **Module-internal symbols**: single underscore prefix (`_update_annotation`, `_macrosteps_count`).
- **"Constants"**: lowercase, following Lisp/unpythonic tradition. (Python's `SCREAMING_CASE` is not used.)
- **Sentinel values**: `sym("name")` for human-readable sentinels:
  ```python
  action_ack = sym("ack")
  action_stop = sym("stop")
  status_pending = sym("pending")
  ```
- **Nonce objects**: `gensym("label")` when you need unique identity with readability.
- **Config modules**: Module-level variables, lowercase, with detailed comments.
- **DPG widget tags**: String literals, `snake_case`, commented with `# tag` on the same line for searchability.

## Docstrings

reStructuredText format. Extensive for public API, pragmatic for internals:

```python
def submit(self, function: Callable, env: env) -> Symbol:
    """Submit a new task.

    `function`: callable, must take one positional argument.
    `env`: `unpythonic.env.env`, passed to `function` as the only argument.

           When `submit` returns, `env` will contain two new attributes:

               `task_name`: str, unique name of the task, for use in log messages.

               `cancelled`: bool. This flag signals task cancellation.
    """
```

Patterns:

- One-line summary, then blank line, then details.
- Parameters documented inline with backtick-quoted names and indented descriptions.
- **NOTE** / **CAUTION** markers for gotchas.
- Reference external resources (URLs, other modules) directly in docstrings.
- Module docstrings list what the module contains and where it sits in the architecture.
- Having no docstring is better than having a placeholder — make the absence explicit.

## Comments

Comments read like prose and explain *why*, not *what*. The style has personality:

```python
# We do this as early as possible, because before the startup is complete,
# trying to `dpg.add_xxx` or `with dpg.xxx:` anything will segfault the app.

# But display at least one entry from each cluster.
if max_n is not None:
    ...
```

```python
import io  # we occasionally need one of Jupiter's moons
```

Recognized comment markers:

- `# TODO:` for known improvements, often with explanation of tradeoffs.
- `# HACK:` for acknowledged workarounds, with context on why.
- `# tag` on lines containing DPG widget tag string literals.
- `# pragma: no cover` always accompanied by an explanation.

## Horizontal separators

Major sections within a module are separated by:

```python
# --------------------------------------------------------------------------------
# Section title
```

This is used consistently throughout the codebase to visually group related functionality. A shorter variant without a title:

```python
# ----------------------------------------
```

is sometimes used for minor sub-sections within a major section.

## Formatting

- **Line width**: ~110 characters. Can locally go a few characters over for a more pleasing layout.
- **No line breaks in URLs**, even if over 110 characters. URLs must be copy-pasteable.
- **Blank lines**: Play the role of paragraph breaks in prose. Insert when the topic changes.
- **One blank line** after most function and class definitions.
- **Two blank lines** when the topic changes across a major boundary (before a horizontal separator, between classes).
- **f-strings** for all string formatting (not `%` or `.format()`).
- **European punctuation**: One space between full stop and next sentence.
- Timing values formatted with g-format: `f"{tim.dt:0.6g}s"`.

## Function signatures

### Parameter documentation in docstrings

Parameters that need explanation are documented with backtick-quoted names:

```python
def ai_turn(llm_settings: env,
            datastore: chattree.Forest,
            retriever: hybridir.HybridIR,
            head_node_id: str,
            ...):
    """Run the AI's response turn.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.

    `datastore`: The chat datastore.

    `head_node_id`: Current HEAD node of the chat.
    """
```

### Type hints

Type hints from `typing` should be used wherever they aid readability, on both public and internal functions. Common patterns:

```python
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

def create_node(self,
                payload: Any = None,
                parent_id: Optional[str] = None,
                timestamp: Optional[int] = None) -> str:
```

### Keyword-only arguments

Arguments without a standard ordering, or flags, use keyword-only syntax:

```python
def get_entries_for_selection(data_idxs, *, sort_field="title", max_n=None):
```

### Internal parameters

Parameters prefixed with `_` indicate internal use and should not be passed by normal callers:

```python
def reset_undo_history(_update_gui=True):
```

## Error handling

- **Error messages** report what was expected and what was actually received:
  ```python
  raise ValueError(f"Unknown mode '{mode}'; valid values: 'concurrent', 'sequential'.")
  ```
- **EAFP** (try/except) for performance-critical paths and thread-safety. Normal logic uses `if/elif/else`.
- Custom exceptions inherit from the most appropriate base.
- Logging of unexpected situations via `logger.error()` / `logger.warning()` before raising.

## Functional style

### Closures as the primary stateful pattern

State is captured in closure variables, not on objects, when the interface is simple:

```python
def make_copy_entry_to_clipboard(item):
    """Closure factory: create a callback that copies `item` to clipboard."""
    def copy_entry_to_clipboard():
        ...  # uses `item` from enclosing scope
    return copy_entry_to_clipboard
```

This pattern is ubiquitous for DPG button callbacks and event handlers.

### `@call` for scoped temporaries

`unpythonic.call` is used to limit the scope of temporary variables in script-style modules:

```python
from unpythonic import call

@call
def _():
    """Set up some config that requires temporary computation."""
    temp_value = expensive_computation()
    global_state.setting = transform(temp_value)
    # `temp_value` does not leak into module scope
```

### `unpythonic.env` as ad-hoc namespace

`env` from unpythonic replaces ad-hoc dictionaries and simple data classes:

```python
from unpythonic.env import env

llm_settings = env(model="Qwen3-VL-30B-A3B",
                   backend_url="http://localhost:5000",
                   personas={"assistant": "Aria"})

# Access as attributes
print(llm_settings.model)
```

Used throughout for passing related settings as a bundle. Particularly heavy in `llmclient` and `scaffold`.

### `unpythonic.box` / `unbox` for mutable references

When you need to replace an immutable value (like a `numpy` array) from inside a closure or across module boundaries:

```python
from unpythonic import box, unbox

selection_data_idxs_box = box(make_blank_index_array())

# Read
current = unbox(selection_data_idxs_box)

# Write (replace contents)
selection_data_idxs_box << new_array
```

### `unpythonic.sym` for sentinel values

Human-readable sentinel values that are distinct from any data value:

```python
from unpythonic import sym

action_continue = sym("continue")
action_done = sym("done")
status_pending = sym("pending")
status_running = sym("running")
```

These compare by identity (`is`) and print readably.

### Other unpythonic utilities used

- `gensym("label")` — unique identifiers with readable names (e.g. for tree node IDs)
- `timer()` — benchmarking context manager (startup timing, pipeline stages)
- `partition(pred, iterable)` — split iterable by predicate
- `ETAEstimator` — progress tracking in long-running pipelines
- `flatten` — flatten nested iterables
- `memoize` — function result caching
- `dyn` (dynamic variables) — implicit parameter passing through call chains (used in `importer.py` for status callbacks)
- `Values` — multiple named return values
- `islice` — lazy slicing
- `window` — sliding window over iterables

## OOP style

OOP is used when the state or interface demands it:

- **Data structures**: `Forest`, `PersistentForest` (tree storage with persistence)
- **Infrastructure**: `TaskManager` (background task scheduling), `HybridIR` (search index)
- **GUI components**: `DPGChatController`, `DPGChatMessage`, `Animator`, `Animation`
- **Server-side AI modules**: Each module in `raven/server/modules/` follows a consistent pattern with `init_module()`, `is_available()`, and task-specific functions.

```python
class TaskManager:
    def __init__(self, name: str, mode: str, executor: concurrent.futures.Executor):
        """..."""
        self.name = name
        self.mode = mode
        self.executor = executor
        self.tasks = {}
        self.lock = threading.RLock()
```

- `__repr__` / `__str__` implemented for debugging where useful.
- ABCs and metaclasses used only when needed, with detailed comments explaining why.

## Configuration

Configuration uses Python modules (config-as-code), not YAML/JSON:

```python
# raven/visualizer/config.py

vis_method = "tsne"  # good quality, fast (recommended)

extract_keywords = True

clusters_keyword_method = "frequencies"
# clusters_keyword_method = "llm"
```

Patterns:

- Module-level variables with descriptive comments.
- Commented-out alternatives show available options.
- `devices` dicts map task names to hardware settings (device string, dtype).
- Config imports flow downward: `raven.config` (global) → component configs (`librarian.config`, `visualizer.config`) → modules.
- A shorthand alias is common: `gui_config = librarian_config.gui_config`.
- Prompt templates use `textwrap.dedent("""...""").strip()`.

## Thread safety

### RLock for shared state

All shared mutable state uses `threading.RLock()`:

```python
self.lock = threading.RLock()

def some_operation(self):
    with self.lock:
        ...
```

`RLock` (reentrant) is preferred over `Lock` to allow the same thread to enter nested critical sections.

### Double-check after lock acquisition

For caches and registries:

```python
try:
    return self._cache[key]
except KeyError:
    with self._lock:
        if key not in self._cache:
            self._cache[key] = compute(key)
    return self._cache[key]
```

### Double-buffered GUI updates

Both the tooltip and info panel build new content in a hidden DPG group, then swap atomically:

1. Create new content in a hidden group (background thread)
2. Acquire content lock
3. Hide old group, show new group
4. `dpg.split_frame()` (wait for DPG to render)
5. Delete old group
6. Release lock

Each build gets a unique build number (appended to DPG tags as `_buildN`) for uniqueness.

### Cancellation via flag

Background tasks monitor a `cancelled` flag set by the task manager:

```python
def my_background_work(task_env):
    for item in items:
        if task_env.cancelled:
            return
        process(item)
```

## Background tasks

### `bgtask.TaskManager`

The standard pattern for background work in GUI apps:

```python
from ..common import bgtask

executor = concurrent.futures.ThreadPoolExecutor()  # default: number of CPU cores

# "sequential" mode: new task cancels previous one (for GUI updates)
info_panel_task_manager = bgtask.TaskManager("info_panel", mode="sequential", executor=executor)

# "concurrent" mode: tasks run independently
indexing_task_manager = bgtask.TaskManager("indexing", mode="concurrent", executor=executor)
```

Tasks are submitted with an `env` that receives `task_name` and `cancelled` attributes:

```python
task_env = env(data=my_data, callback=my_callback)
info_panel_task_manager.submit(update_info_panel_worker, task_env)
```

### Event-driven orchestration

High-level operations take optional callbacks for progress reporting:

```python
def ai_turn(llm_settings, datastore, ...,
            on_docs_start=None, on_docs_done=None,
            on_llm_start=None, on_llm_progress=None, on_llm_done=None,
            on_tools_start=None, on_tools_done=None,
            on_nomatch_done=None,
            on_prompt_ready=None):
```

The controller passes closures that update GUI state. This keeps the orchestration layer GUI-agnostic.

## DearPyGui patterns

### Widget tags

All widget tags are string literals (not integer IDs), using `snake_case`:

```python
dpg.add_button(label="Undo", tag="selection_undo_button")  # tag
```

The `# tag` comment marks lines containing widget tag references for searchability.

### Explicit `parent=` from background threads

DPG's container stack is global and not thread-safe. Background threads must always use explicit `parent=`:

```python
# Good: explicit parent, safe from any thread
dpg.add_text("hello", parent=my_group)

# Bad: uses implicit container stack, not thread-safe
with dpg.group():
    dpg.add_text("hello")
```

The `with` block style is fine in the main thread during GUI setup.

### Closure factories for per-item callbacks

Since DPG button callbacks can't receive custom arguments, closure factories are used:

```python
def make_select_cluster(cluster_id):
    def select_cluster():
        update_selection(get_data_idxs_for_cluster(cluster_id), mode="replace")
    return select_cluster

# In GUI setup:
dpg.add_button(label=f"Select #{cid}", callback=make_select_cluster(cid))
```

### `user_data` for widget metadata

DPG widgets store metadata in their `user_data` field as `(kind, data)` tuples:

```python
dpg.add_group(user_data=("entry_title_container", data_idx), parent=...)
```

Predicate functions check the kind for O(log n) lookups:

```python
def is_entry_title_container_group(item):
    ud = dpg.get_item_user_data(item)
    return ud is not None and ud[0] == "entry_title_container"
```

## Layered architecture

### Dependency direction

Dependencies flow strictly downward through layers:

```
Layer 5 - Applications:     app.py
Layer 4 - Controller:       chat_controller.py
Layer 3 - Orchestration:    scaffold.py
Layer 2 - Backends:         llmclient.py, hybridir.py
Layer 1 - Utilities:        chatutil.py, appstate.py
Layer 0 - Foundation:       config.py, chattree.py
```

Each layer only imports from layers below it. No circular dependencies. This pattern (demonstrated in `raven/librarian/`) is the target architecture for all components.

### Server/client split

All ML inference runs in `raven/server/modules/`. Client apps call the server via `raven/client/api.py`. Local fallback is available via `raven/client/mayberemote.py` when the server is not running.

## Testing

Tests use pytest and live in `tests/` subdirectories within each component:

```python
# raven/librarian/tests/test_chattree.py

import pytest
from raven.librarian.chattree import Forest, PersistentForest

@pytest.fixture
def forest():
    return Forest()

@pytest.fixture
def chain(forest):
    """A -> B -> C linear chain."""
    a = forest.create_node(payload="A")
    b = forest.create_node(payload="B", parent_id=a)
    c = forest.create_node(payload="C", parent_id=b)
    return forest, a, b, c

class TestCreateNode:
    def test_create_root_node(self, forest):
        node_id = forest.create_node(payload="root")
        assert forest.nodes[node_id]["parent"] is None

    def test_create_child_node(self, forest):
        parent_id = forest.create_node(payload="parent")
        child_id = forest.create_node(payload="child", parent_id=parent_id)
        assert forest.nodes[child_id]["parent"] == parent_id
```

Patterns:

- Fixtures for common setups (bare forest, linear chain, branching tree).
- Test classes group related tests by feature area.
- Tests use the public API, not internal state (except for verification assertions).
- `pytest.raises` for expected exceptions; `pytest.mark.xfail(strict=True)` for known bugs.
- Test file naming: `test_<module_name>.py`.

## External dependencies

Raven has many dependencies (ML frameworks, GUI toolkit, web server, etc.) — it's an application, not a library. However:

- Don't add dependencies without a reason. Prefer stdlib when reasonable.
- `unpythonic` is a core dependency used throughout.
- `mcpyrate` is used only for its `colorizer` utility (terminal colors). No macros.
- Heavy ML dependencies (`torch`, `transformers`, `sentence-transformers`, `spacy`) are confined to specific modules.
- Vendored dependencies live in `raven/vendor/` with attribution and modification notes.
