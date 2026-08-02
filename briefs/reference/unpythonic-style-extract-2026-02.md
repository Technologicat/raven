> **Archived snapshot, 2026-02-04. Not maintained; do not read as current documentation.**
>
> This is what one pass of LLM-assisted reading of the `unpythonic` codebase (plus its
> `CONTRIBUTING.md`) surfaced about its style, on the day Raven's own style guide was
> written. It was the input to [`raven-style-guide.md`](../raven-style-guide.md).
>
> It is kept because it is *evidence*, not documentation: an extraction like this is not a
> function of the codebase — re-running it would produce a different document, noticing
> different things. So it records a moment of attention, and that isn't recoverable by
> repeating the process.
>
> For unpythonic's actual, maintained style guide, see
> [its `CONTRIBUTING.md`](https://github.com/Technologicat/unpythonic/blob/master/CONTRIBUTING.md#style-guide).

# The unpythonic coding style

Extracted from the `unpythonic` codebase and its CONTRIBUTING.md. This documents actual patterns observed in the code, not just the aspirational guidelines.

## Philosophy

The governing principle is *"find pythonic ways to do unpythonic things"*. Fitting Python user expectations beats mathematical elegance. Beyond that:

- **Be obsessively correct.** Lack of robustness is a bug. Get terminology right. Handle edge cases.
- **Be obsessively complete** when the extra mile adds value. Features should compose and work together.
- **Be concise but readable**, like in mathematics. No code golf, but no unnecessary ceremony either.
- **Refactor aggressively.** Extract reusable utilities. Build a tower of abstractions. Internal dependencies are encouraged.
- **Macros are the nuclear option.** Only make a macro when a regular function can't do the job. Prefer a pure-Python core with a thin macro layer for UX.

## Module structure

Modules follow a consistent layout:

```python
# -*- coding: utf-8 -*-
"""Short module description.

Longer explanation, often with attribution and references to
Racket, Haskell, Common Lisp, SRFI, or academic papers.
"""

__all__ = ["public_name1", "public_name2"]

# stdlib imports
from functools import wraps
from threading import RLock

# internal imports (relative)
from .arity import _resolve_bindings
from .dynassign import dyn
```

Key points:
- `__all__` is mandatory and placed immediately after the module docstring, before imports. It is populated explicitly, manually—no tricks.
- Imports use `from ... import ...` style (not `import ...`). This is mandatory for macro imports and used consistently everywhere.
- Don't rename unpythonic features with `as`—macro code depends on original bare names at use sites.
- No star imports except in the top-level `__init__.py` for re-export.
- Internal imports use relative paths (`.module`, `..module`).

## Module size

- Target ~100–300 SLOC per module (docstrings, comments, and blanks don't count).
- Rough upper bound ~700 lines total. Python is expressive; that's a lot for a language extension library.
- Don't obsess over it if going over allows a better solution.

## Naming

- **One-word names preferred** for public API. If not possible, use `snake_case`. Coined-but-obvious words are fine (cf. SymPy's `lambdify`).
- **Module-internal symbols**: single underscore prefix (`_symbols`, `_jump`, `_global_dynvars`).
- **Sentinel values**: use `sym()` for human-readable sentinels: `_success = sym("_success")`.
- **Nonce objects**: `object()` when you only need identity, `gensym("label")` when you also need readability and pickle support.
- **Classes**: `PascalCase`, including exception classes.
- **Functions/methods**: `lowercase_with_underscores`.
- **"Constants"**: lowercase, following Lisp tradition. (Python's `SCREAMING_CASE` is not used.)
- When different subcultures have different names for the same idea, pick one but list alternatives for discoverability.

## Docstrings

reStructuredText format. Extensive, almost tutorial-like for complex features:

```python
def memoize(f):
    """Decorator: memoize the function f.

    All of the args and kwargs of ``f`` must be hashable.

    Any exceptions raised by ``f`` are also memoized. If the memoized function
    is invoked again with arguments with which ``f`` originally raised an
    exception, *the same exception instance* is raised again.

    **CAUTION**: ``f`` must be pure (no side effects, no internal state
    preserved between invocations) for this to make any sense.

    Beginning with v0.15.0, `memoize` is thread-safe even when the same memoized
    function instance is called concurrently from multiple threads. Exactly one
    thread will compute the result. If `f` is recursive, the thread that acquired
    the lock is the one that is allowed to recurse into the memoized `f`.
    """
```

Patterns:
- One-line summary, then blank line, then details.
- **CAUTION** / **NOTE** markers for gotchas.
- Use `::` and indented blocks for code examples. Examples often double as documentation of composability.
- Parameters documented either as a `Parameters:` block with indented descriptions, or inline in prose when simpler.
- Reference external resources (URLs, papers, other languages' docs) directly in docstrings.
- Having no docstring is better than having a placeholder docstring—make the absence explicit for static analyzers.

## Comments

Comments read like prose and explain *why*, not *what*. The style has personality:

```python
try:  # EAFP to eliminate TOCTTOU.
    kind, value = memo[k]
except KeyError:
    # But we still need to be careful to avoid race conditions.
    with lock:
```

Recognized comment markers:
- `# TODO:` for known improvements, often with explanation of tradeoffs.
- `# HACK:` for acknowledged workarounds, with context on why.
- `# pragma: no cover` always accompanied by an explanation.
- Commented-out "essential idea" versions of complex functions are intentionally retained for pedagogical clarity:

```python
#def memoize_simple(f):  # essential idea, without exception handling or thread-safety.
#    memo = {}
#    @wraps(f)
#    def memoized(*args, **kwargs):
#        ...
```

## Horizontal separators

Major sections within a module are separated by:

```python
# --------------------------------------------------------------------------------
```

This is used to visually group related functionality within a single file.

## Formatting

- **Line width**: ~110 characters. Can locally go a character or three over for a more pleasing layout (e.g. a long word at end of line, or a full stop that wouldn't fit).
- **No line breaks in URLs**, even if over 110 characters. URLs must be copy-pasteable.
- **Blank lines in code play the role of paragraph breaks in prose.** Insert when the topic changes or when doing so significantly increases clarity.
- **One blank line** after most function and class definitions.
- **Two blank lines** only when the situation already requires a blank line *and* the topic changes.
- A group of related very short methods/functions may omit blank lines between them, making the grouping explicit.
- **f-strings** for all string formatting (not `%` or `.format()`).
- European punctuation: one space between full stop and next sentence.

## Function signatures

### Curry-friendly parameter ordering

Parameters that change least often go on the left. For higher-order functions: user function on the left, data on the right.

```python
def scanl(proc, init, iterable0, *iterables, longest=False, fillvalue=None):
```

### Kleene-plus pattern

When at least one variadic argument is required, make it explicit so `curry` knows when to trigger:

```python
# Good: curry can see that at least one iterable is required
def foldl(proc, init, iterable0, *iterables):

# Bad: curry can't tell if zero iterables is a valid call
def foldl(proc, init, *iterables):
```

### Keyword-only arguments for disambiguation

Arguments without a standard ordering should be keyword-only to prevent accidental transposition:

```python
def scanl(proc, init, iterable0, *iterables, longest=False, fillvalue=None):
```

## Decorator patterns

Decorators follow a consistent structure:

```python
@register_decorator(priority=10)
def memoize(f):
    """..."""
    lock = RLock()
    memo = {}
    @wraps(f)
    def memoized(*args, **kwargs):
        ...
    return memoized
```

- Always `@wraps(f)` on wrapper functions.
- `register_decorator(priority=N)` for decorator ordering (used by macro layer).
- Decorators that are also usable as regular functions (like `curry`) handle both cases.
- State captured in closure variables (not on the wrapper function object).

## Thread-safety pattern

A recurring pattern for thread-safe singletons, caches, and registries:

```python
try:  # EAFP to eliminate TOCTTOU.
    return _registry[key]
except KeyError:
    # But we still need to be careful to avoid race conditions.
    with _lock:
        if key not in _registry:
            # We were the first thread to acquire the lock.
            _registry[key] = create_value()
        else:
            # Some other thread acquired the lock before us.
            pass
    return _registry[key]
```

This appears in `sym.__new__`, `gsym.__new__`, `Singleton.__new__`, and `memoize`. The structure is always: EAFP try → lock → double-check → act.

## Error handling

- **EAFP over LBYL** for performance-critical paths, but **not** as a blanket rule. Normal logic uses `if/elif/else`.
- Error messages report what was expected and what was actually received:

```python
raise TypeError(f"When partially applying {description}:\n"
                f"Parameter binding(s) do not match type specification: {mismatches_str}")
```

- `BaseException` caught only in specific cases (e.g. `memoize` caching exceptions).
- Custom exception classes inherit from the most appropriate base (`Exception`, `BaseException`).

## Functional style

- **Heavy use of closures** for state encapsulation. This is the primary stateful pattern, preferred over classes when the state is simple.
- **No mutation of input** (except in macros/AST transforms, where in-place editing is standard).
- **Return useful values** even from primarily side-effecting operations, to allow chaining.
- `functools.partial` used frequently for partial application in non-curried contexts.
- **Higher-order functions** are a fundamental building block. Custom HOFs are defined freely.
- `nonlocal` used without hesitation when needed, but sparingly.

## OOP style

OOP is used when it's the right tool:
- Custom container types (`box`, `frozendict`, `env`, `cons`).
- ABCs registered explicitly at module level: `Container.register(env)`.
- Metaclasses used when needed (e.g. `Singleton`), with detailed comments explaining why.
- `__new__` customized for singleton semantics and pickle support.
- `__repr__` and `__str__` always implemented for debugging.

## Dynamic variables (`dyn`)

When something needs to be implicitly passed through several layers of function calls, and a closure is not the right tool (because some functions in the chain are defined elsewhere), use `dyn`:

```python
from .dynassign import dyn, make_dynvar

make_dynvar(curry_context=[])

# At usage site:
with dyn.let(curry_context=["my context"]):
    ...
```

This pattern is especially used for passing `mcpyrate` `**kw` arguments through to syntax transformers without polluting intermediate function signatures.

## Testing

Custom framework based on conditions and restarts. Each test module exports a `runtests()` function:

```python
from ..syntax import macros, test, test_raises, fail, the  # noqa: F401
from ..test.fixtures import session, testset, returns_normally

def runtests():
    with testset("feature name"):
        test[2 + 2 == 4]
        test[some_func(42) == expected]

    with testset("error cases"):
        test_raises[TypeError, bad_call()]
```

- Tests grouped with `testset()` context managers.
- `test[]` macro for assertions (replaces `assert`).
- `test_raises[]` for expected exceptions.
- Tests double as usage examples; they often contain finer points that didn't make it to prose docs.

## Cross-cutting concerns

Some features have cross-cutting behavior that affects other modules:
- `curry` changes reduction semantics and has special handling scattered across the codebase.
- `@generic` (multiple dispatch) similarly touches several modules.
- `passthrough_lazy_args` and `maybe_force_args` appear in many modules to support the `lazify` macro.
- `register_decorator` is used throughout for macro-layer decorator ordering.

When modifying any of these, grep widely.

## External dependencies

Avoid them. `unpythonic` is meant as a standalone base to build on. `mcpyrate` is the only allowed external dep and must remain strictly optional for the pure-Python layer. The macro features (and `mcpyrate` dependency) are in `unpythonic.syntax`, strictly separated.
