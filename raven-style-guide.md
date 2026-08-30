# The Raven coding style

Documents actual patterns observed in the Raven codebase. Adapted from unpythonic's house style — via [an extract made in 2026-02](briefs/reference/unpythonic-style-extract-2026-02.md), which is archived for provenance; unpythonic's maintained style guide lives in [its `CONTRIBUTING.md`](https://github.com/Technologicat/unpythonic/blob/master/CONTRIBUTING.md#style-guide).

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
from typing import Any, Callable

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
  - **What gets imported is the *module*, not the names inside it** — `from ..gui import keyboardmark`, then `keyboardmark.COLOR` at the call site. That is what keeps a use site saying where a name came from.
  - **For a module whose public names are deliberately short, this stops being a preference.** `keyboardmark.COLOR` and `keyboardmark.PULSE_SECONDS` are bare because the module already supplies the noun; `from ..gui.keyboardmark import COLOR` leaves a bare `COLOR` that says neither which colour nor why this widget wears it, which is precisely the information the short name was leaning on the namespace to carry.
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

A guideline and not a limit, and the number is worth less than the reason behind it.

**First, watch the units, because we quote both and they differ by about half.** This codebase runs ~40%
docstrings and comments — deliberately, see *Comments* — so a module's total line count roughly doubles its
SLOC. Measured 2026-08-29: `chat_controller.py` is 5090 lines and **2046 SLOC**; `papers/deduplicate.py` is
1225 lines and 556 SLOC; `papers/bibtex.py` is 831 lines and 341 SLOC. A "5k module" is usually a 2k one
wearing its prose, and shrinking the prose to make a number look better would be the wrong repair.

- Target roughly 300–800 SLOC for a library module. Over that is a prompt to look, not a finding.
- **What you are looking for is whether the layering has gone**, not whether a counter has passed a
  threshold. A 900-SLOC module with no discernible internal structure is worse off than a large one whose
  layers hold, and no line count will tell you which of the two you are holding.
- **`chat_controller.py` is not an endorsement.** It is four times the target, its layers do hold, and it
  is *tolerated for now, until we find a better solution — if ever* (Juha, 2026-08-29). That is a standing
  acknowledgement that nobody has a decomposition worth the disruption, not a demonstration that 5k is
  acceptable when the layering is good. Take a better decomposition if one turns up; do not cite this
  module as licence for the next one.
- **The ~700 figure came from `mcpyrate` and `unpythonic`, and does not transfer unchanged.** There the
  lines are Kolmogorov-hard, and length really is the signal.
  - The tempting next step is to say Raven is the prose-heavy opposite, and that is wrong: **most of
    `chat_controller.py` looks Kolmogorov-hard too** (Juha, 2026-08-29), and it is our largest module.
    Raven has both kinds, so density does not predict the right size either — which is the actual reason
    the guideline cannot be a threshold. `papers/deduplicate.py` at 556 SLOC is genuinely prose-heavy and
    sits inside the target; `chat_controller.py` is dense and sits far outside it; both are where they
    should be.
- App modules run larger, and that is not automatically a debt. As of 2026-08-29 the biggest by total lines
  are `librarian/chat_controller.py` ~5.1k, `librarian/app.py` ~2.6k, `librarian/llmclient.py` ~2.4k, and a
  cluster of `app.py` files around 1.9k (cherrypick, visualizer, avatar settings editor).
- **An `app.py` is GUI layout instantiation and wiring, and that much really is irreducible.** Forty
  distinct widgets with distinct bindings need forty statements; no refactor compresses a description
  below what it describes, so for the wiring part the line count measures the *GUI*, not the code.
  - It compresses in exactly two places. Repetition **across** apps belongs in `raven/common/gui/` as a
    component each app opts into with one call — `filedrop.install(...)` is that chunk of wiring deleted
    six times over. Repetition **within** one app is a loop or a local helper.
  - **Check the premise before leaning on it.** The argument licenses a long `app.py` only to the extent
    the file *is* wiring, and ours may not be: measured 2026-08-29, `visualizer/app.py` has 390 of 1919
    lines mentioning `dpg.`, `cherrypick/app.py` 194 of 1922, `avatar/settings_editor/app.py` 350 of 1916
    — and they hold 58, 54 and 86 functions. That grep is a crude lower bound on wiring (a `with
    dpg.group():` body is wiring too), so it is a hint rather than a verdict. But a file of layout and
    wiring holding eighty functions is mostly holding callback *bodies*, and a callback body is logic.
  - Which is already ruled on elsewhere: anything worth calling from elsewhere, or worth testing, belongs
    in another module (`CLAUDE.md`, "An `app.py` is an OS entry point"). So the order to check an oversized
    `app.py` is logic that is not wiring, then cross-app repetition, then intra-app repetition — and only
    then conclude the GUI is simply big.
- **`visualizer/` is what a resolved case looks like**, and worth contrasting with the one above.
  `app.py` was 4.4k and genuinely a god object — not fine, and nobody pretended otherwise — until it was
  split into `info_panel`, `selection`, `plotter`, `annotation`, `word_cloud`, `entry_renderer` and
  `app_state`. The result is *mostly* fine, which is a real verdict rather than a hedge: `importer.py` at
  595 SLOC is inside the target, `info_panel.py` at 828 is marginal and already flagged as the next split
  candidate, and `app.py` at 1204 is still half again over. Improved and largely settled, with the
  remainder named.
  - **The two large modules got large for different reasons, and that is the useful distinction.**
    Visualizer was an organically grown experiment nobody had stopped to refactor (Juha, 2026-08-29) — the
    size was deferral rather than a decision, and refactoring simply collected the debt.
    `chat_controller.py` was designed with its layers and is large anyway. So one was answerable by
    looking, and the other has been looked at. Worth asking which kind you have before reaching for a
    split: the first kind pays back immediately, and the second may have no better arrangement to move to.
- `raven/librarian/` (~22.8k lines across 20 modules) remains the target architecture — for its layering,
  which is what made it the model, rather than for its per-module line counts.

## Where a utility lives, and when that answer expires

`raven.common.utils` is the grab bag: the home for things with no more proper home. Grab bags balloon into
chaos unless actively mitigated, so it is worth knowing what the mitigation is.

**Ask what the code is *about*, not who calls it.** An operation you cannot describe without naming its
subject belongs with that subject — the surface-syntax BibTeX readers (`header_key`,
`brace_repair_candidates` and friends) are BibTeX, whoever happens to call them. The exception is an
operation that is genuinely general and merely got written the first time somebody needed it here; the tell
is that you *can* say what it does without naming the domain. Number formatting written while doing BibTeX
work is still number formatting.

**But the right answer changes over time, and this is the part worth internalizing.** Those four readers
sat in `common.utils` with a comment explaining that two of their three callers were not paper tooling.
That was correct: there was no `papers.bibtex` module then, and creating one *solely* to host four
commonutil-looking functions is structure for its own sake — Bach. Once `papers.bibtex` existed for its own
reasons, the cost of the proper home dropped to zero and the judgement flipped. Nobody erred; the ground
moved.

**So the trigger is the arrival of a new module, not a periodic audit.** The day a module about X is
created is the day `utils` may already be holding something about X — and nothing prompts anyone to look,
because nothing breaks. That silence is the whole problem: accretion is the passive state, and a grab bag
gets tidier only when somebody goes and looks.

Note none of this is a rule you can apply mechanically, and it should not be turned into one. It is a way
of looking, and the useful residue is the moment to look rather than a procedure for looking.

## Naming

- **Functions**: `lowercase_with_underscores`.
- **Classes**: `PascalCase`, including exception classes.
- **Module-internal symbols**: single underscore prefix (`_update_annotation`, `_macrosteps_count`).
- **"Constants": the casing says who the constant is for**, which is the distinction to get right — it is
  not a matter of taste, and a name in the wrong case is a category error rather than an inconsistency.
  - **lowercase** — a **user-exposed config knob**, following Lisp/unpythonic tradition. These live in a
    `config.py`, and all four of Raven's are lowercase throughout.
  - **`SCREAMING_CASE`** — a **file-local implementation detail**. Padding, a regex, a table the module
    reads and nobody outside it should. The package holds ~290 of these, correctly.

  So a lowercase module-level constant that no user is meant to touch is a category error. Deciding the
  case is deciding whether it is a knob, which is the useful question anyway.

  **The GUI apps' `config.py` files are `SCREAMING_CASE` throughout, and that is right rather than drift**
  — `cherrypick`, `conference_timer` and `xdot_viewer`, about a hundred names between them. What they hold
  is layout geometry and theme, which the app *computes with* (`TOOLBAR_H = FONT_SIZE + 2 *
  DPG_FRAME_PADDING_Y`) rather than reads as a setting. The lowercase config modules hold settings proper:
  model names, URLs, thresholds. Living in a file called `config.py` is not by itself what makes something
  a knob. (Juha, 2026-08-29, declining a rename sweep this entry as first written would have licensed.)

  (This entry previously read "lowercase … `SCREAMING_CASE` is not used", which the codebase contradicted
  about 290 times over. Corrected 2026-08-29.)
- **Sentinel values**: `sym("name")` for human-readable sentinels:
  ```python
  action_ack = sym("ack")
  action_stop = sym("stop")
  status_pending = sym("pending")
  ```
- **Nonce objects**: `gensym("label")` when you need unique identity with readability.
- **People and companies in test data and examples: invent them.** Names in a fixture, a docstring
  example or a comment are made up — `Bloggs, PhD, MSc, Joan`, `Holm Dahl, Aksel`, `Vantage Academic
  Press`. Real corpora produce the *formats*, so those stay real; the names do not have to. This applies
  whether or not the example is unflattering, though a real person illustrating malformed data is the
  worst case.
  - Two exceptions. A **stock example of the shape itself** — ask anyone for a name with a "van" in it
    and they will say `Ludwig van Beethoven` — where using it names the *shape* rather than a person, and
    an invented substitute would be less recognizable for no gain. (Two centuries dead helps; a living
    person is not a stock example of anything.) And a **name that is the fact**: "Elsevier" lemmatizes to
    "elsevi", the `_NN-V` DOI suffix is Springer's convention. Fictionalize those and the sentence stops
    being about anything.
- **Config modules**: Module-level variables, lowercase, with detailed comments.
- **DPG widget tags**: String literals, `snake_case`, commented with `# tag` on the same line for searchability.
- **String quotes**: double, `"like this"`. Single quotes are for the case that earns them — a value that
  itself contains a double quote, where `'he said "no"'` beats escaping. Not a rule anybody should have to
  think about; it exists so that greps for a literal find it. (The omission of this line let the tree drift
  to roughly a third single-quoted; a sweep is filed in `TODO_DEFERRED.md`.)

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

- One-line summary, then blank line, then details — where *details* means arguments, return value, and
  which cases are handled. Why the shape was chosen, what went wrong the other way, which failure it
  prevents: those go in a comment below the `def`, where the maintainer who needs them is reading. A second
  docstring paragraph that is none of the three is the tell.
- Parameters documented inline with backtick-quoted names and indented descriptions.
- **NOTE** / **CAUTION** markers for gotchas.
- Reference external resources (URLs, other modules) directly in docstrings.
- Module docstrings list what the module contains and where it sits in the architecture.
- Having no docstring is better than having a placeholder — make the absence explicit.

## Comments

**The standard is that prose must be as load-bearing for understanding as the code is for behaviour.**
Not "more comments is better": every line of prose earns its place the way a line of code does, and there
should be no *extra*. (Juha, 2026-08-29.)

Raven errs toward [literate programming](https://en.wikipedia.org/wiki/Literate_programming)
(Knuth, 1984), because dense code makes every reader re-derive the thinking that produced it — the same
objection as to theorem-proof-corollary mathematics monographs, which are excellent for *reference* and
poor for *learning*. Compressing a derivation moves the work from one writer to every reader, and here the
reader is usually the author six months on, or a session with no memory of the last one.

**The ~40% the codebase measures at is an observation, not a target.** It is what load-bearing prose
happens to cost here. Writing to hit a ratio makes the file worse, and two readings to refuse outright:

- *"Prose is cheap, so more of it is free."* It is cheaper per line than dense code, not free — someone
  reading the whole file reads all of it.
- *"Half is prose, so 5k total is really 2.5k."* Size is not bought back that way. Judge a module by its
  SLOC (see *Module size*), and note that a file needing that much explanation is saying something about
  its own complexity.

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
- **Blank lines**: Play the role of paragraph breaks in prose. Insert when the topic changes, not mechanically (e.g. not "always before `return`").
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

Type hints should be used wherever they aid readability, on both public and internal functions.

**Prefer the modern spelling** (PEP 604 unions, PEP 585 builtin generics) over the `typing` aliases:

- `X | None` instead of `Optional[X]`; `X | Y` instead of `Union[X, Y]`.
- `list[X]`, `dict[K, V]`, `tuple[X, ...]`, `set[X]` instead of `List`, `Dict`, `Tuple`, `Set` from `typing`.

Both are available on our baseline (Python 3.11+), read more cleanly, and need no import. `typing` is still the home for things without a builtin spelling — `Any`, `Callable`, `NamedTuple`, `Protocol`, `TypeVar`, etc. — so import those as needed.

```python
def create_node(self,
                payload: Any = None,
                parent_id: str | None = None,
                timestamp: int | None = None) -> str:
```

The codebase is mid-migration from the old `typing`-alias spelling. Write new and modified code in the modern spelling; existing old-style hints can be left as-is unless you're already editing that code (same rule as for adding hints to untyped code).

**Where a docstring names a type, use the same spelling** — `dict | None`, never `Optional[dict]`. One dialect, not two.

Which is a narrower rule than it looks, because a docstring should mostly *not* name a type. The split is by whether a signature already carries it:

- **Parameters: no type in the docstring.** It is two lines up in the signature, and a second copy is a second thing to keep current. Write `` `reasoning_content`: the accumulated thinking trace. `` and stop.
- **Things with no signature: the type belongs in the docstring**, because there is nowhere else it can go. The attributes of a returned `unpythonic.env.env`, the keys of a returned dict, the arguments a callback parameter will be called with — all of these are contracts a reader cannot get from the code above.

```python
    Returns an `env` with the following attributes:

        `n_tokens: int`: Number of tokens emitted by the LLM.
        `phases: dict | None`: Where the wall time went; `None` when the model generated no text.
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
- **Properties**: Use explicit `property(fget=..., fset=..., doc=...)` instead of the `@property`/`@x.setter` decorator syntax. Define `get_x` and `set_x` as regular methods first, then bind them:
  ```python
  def get_x(self) -> int:
      return self._x
  def set_x(self, value: int) -> None:
      self._x = value
  x = property(fget=get_x, fset=set_x, doc="The x coordinate.")
  ```

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

**A config value does what it says on the tin: the number written is the number used.** When a layout change
means a width has to shrink, change the default — never subtract at the use site, and never wrap the read in
a helper that adjusts it. Two reasons, and the second is why this is a rule rather than a preference:

- **A hidden adjustment breaks anyone who overrides the value.** They set the number the name promises and
  get something else, silently, and only for them — which is the hardest kind of report to act on.
- **The adjustment is invisible where the value is read.** A call site says `gui_config.title_wrap_w` and
  means it; a reader tracing a layout has no reason to suspect a subtraction sitting somewhere else.

The comment beside the value is where the arithmetic goes, naming what else occupies that space — which is
what lets the next person change it correctly. `title_wrap_w=486, # Note there will be the keyboard mark's
dot and two columns of buttons to the left of each item title.` (Juha's correction, 2026-08-28: the first
version of that change subtracted the dot's width in `info_panel`, behind a helper whose name promised the
configured width.)

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

### Hotkey discoverability

Every hotkey must be discoverable through the GUI, on two surfaces:

1. **The help card** (`F1`, via `helpcard.HelpWindow`) — the full keyboard reference.
2. **Tooltips** — every GUI control with a hotkey names it in its tooltip, in brackets:

```python
with dpg.tooltip(btn):
    dpg.add_text("Open folder [Ctrl+O]")
```

This is non-negotiable: surfacing the key right on the control it triggers is how users learn the shortcuts in flow. A feature reachable by keyboard but only listed in the help card is a UX miss — most software in the wild gets this wrong; Raven apps don't.

There is no shared keymap — bindings live in the hotkey handler, and these surfaces mirror them by hand (KISS; hotkeys change rarely). Keep a comment at the hotkey handler reminding the next dev to update both surfaces when adding, removing, or rebinding a key:

```python
# No shared keymap — bindings live here, and the surfaces that make them
# discoverable mirror them by hand (KISS; hotkeys change rarely). If you add,
# remove, or rebind a key, update those surfaces too:
#   - the help card (search "HelpWindow")
#   - any tooltip naming the key (search its bracketed hint, e.g. "[Ctrl+O]")
```

**Positional hotkeys** (keys chosen for physical location, e.g. WASD as an arrow-key alias for one-handed use) are keyboard-layout-dependent — WASD lands at ZQSD on AZERTY, and Z/Y swap on QWERTZ. Until the fleet grows layout-aware remapping, keep positional bindings as *aliases* alongside the layout-independent originals (the arrow keys), never as replacements.

### Never lose the reader's scroll position

**No action in the GUI may cost the user their place, so long as the corresponding new position is
computable.** Not "rarely", and not "except after a rebuild": if it can be worked out where the thing the
reader was looking at has gone, the view goes there. That covers every action, including the ones the app
takes on its own — a message finalizing, a background rebuild, a panel repopulating — and not only the ones
the user asked for.

It is a standard rather than a nicety because the alternative silently punishes reading. A scroll position
is the reader's own state, built by hand, and an app that discards it teaches them not to scroll away —
which in a chat log or a document panel is most of what the thing is for.

**Visualizer's info panel is the worked implementation, and it goes to considerable lengths to keep the
promise** — which is the measure of how seriously the standard is meant, rather than a sign that it was
overbuilt there. `_update_info_panel`'s `scroll_anchor_data`, tags with the build number stripped so an
anchor survives a rebuild, and `_find_next_or_prev_item` to choose a new anchor when the old one no longer
exists. Borrow that shape rather than inventing a second one — a chat message and an info-panel entry are
the same problem in different clothes.

**What it does not license is making a preference retroactive.** A switch that says how the *next* thing
should look has no business rewriting what is already on screen; there the right answer is to change
nothing, and no anchor is needed because nothing moved. Librarian's *Show thinking* is deliberately of that
kind. The standard governs layout changes that *do* happen.

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
