"""Rewrite bare references to a set of module-level globals in `raven/visualizer/app.py`
into attribute access on `app_state` (the shared namespace from `raven.visualizer.app_state`),
using Python's tokenizer so that occurrences inside string literals, comments, and
f-string literal parts are left untouched.

Built 2026-04-17 for the Visualizer refactor's ``app_state`` migration. The Visualizer
currently has all shared-across-modules state as bare module-level globals in `app.py`;
as submodules get extracted (`word_cloud.py`, `selection.py`, `plotter.py`, …), those
globals need to be accessible from multiple files. Rather than sibling imports (which
create cycles with `app.py` at the top) we stash them as attributes on
`app_state.app_state`, an `unpythonic.env.env` namespace held in `raven/visualizer/app_state.py`.

## What it does

1. Rewrites every *code* occurrence of a NAME in `NAMES_TO_MIGRATE` into `app_state.NAME`.
   Uses the `tokenize` module so string literals and comments are correctly left alone —
   which the ~52 string/comment occurrences of `dataset` (as of the 2026-04-17 migration)
   made a non-negotiable requirement.

2. Skips NAME tokens that are the *name-being-defined* in a `def NAME(...)` or
   `class NAME(...)` statement — those binding sites must stay as top-level definitions;
   only call sites get rewritten. (You register the defined function on `app_state`
   manually in a later step, e.g. ``app_state.enter_modal_mode = enter_modal_mode``.)

3. Deletes any ``global NAME`` declaration where every name in the declaration is being
   migrated. Those become dead code once `NAME` is no longer a module-level binding.
   Declarations that mix migrated and non-migrated names are left alone (review manually).

## Usage

Pass the path to rewrite, followed by the bare identifiers to migrate::

    python briefs/tools_visualizer_rewrite_to_app_state.py raven/visualizer/app.py \
        dataset bg themes_and_fonts selection_data_idxs_box filedialog_save \
        enter_modal_mode exit_modal_mode

Follow up with ``python -c "import ast; ast.parse(open('raven/visualizer/app.py').read())"``
to catch any syntax errors before moving on, and scan `git diff` for any changes that
look wrong (e.g. an identifier that happens to live in a module other than `app.py`).

## Scope / limitations

- Only does *bare name → app_state.name*. Doesn't handle fully-qualified access or
  attribute chains. Fine for our case — everything we're migrating is a bare name at
  the `app.py` module level.
- Doesn't add ``app_state.NAME = value`` at the original definition site. Adding
  the initial binding is a one-line manual step per name.
- Doesn't add ``from .app_state import app_state`` — also a one-line manual step.
- Doesn't run `ast.parse` or `ruff check` — verify externally.

If the migration targets expand beyond `dataset`/`bg`/`themes_and_fonts`/etc., just
edit `NAMES_TO_MIGRATE` and re-run on a clean `app.py`.
"""

import io
import sys
import tokenize
from collections import defaultdict


def rewrite(source: str, names_to_migrate: set) -> str:
    NAMES_TO_MIGRATE = names_to_migrate  # keep the original variable name for readability below
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))

    # Pass 1: find `global NAME(, NAME)*` declarations whose names are all migrated;
    # those lines go away entirely (dead after the migration).
    global_lines_to_delete = set()  # 1-based line numbers

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == tokenize.NAME and tok.string == "global":
            j = i + 1
            global_names = []
            all_migrated = True
            while j < len(tokens) and tokens[j].type not in (tokenize.NEWLINE, tokenize.NL):
                t = tokens[j]
                if t.type == tokenize.NAME:
                    global_names.append(t.string)
                    if t.string not in NAMES_TO_MIGRATE:
                        all_migrated = False
                elif t.type == tokenize.OP and t.string == ",":
                    pass
                elif t.type == tokenize.COMMENT:
                    pass  # trailing comment is fine
                else:
                    all_migrated = False
                    break
                j += 1
            if global_names and all_migrated:
                global_lines_to_delete.add(tok.start[0])
                i = j + 1
                continue
        i += 1

    # Pass 2: collect (row, col_start, col_end, replacement) for each NAME to rewrite.
    # Skip NAMEs that are:
    # (a) on a line we're about to delete,
    # (b) the *name being defined* in a `def NAME(...)` or `class NAME(...)` statement —
    #     those binding sites stay; only call sites get rewritten,
    # (c) a kwarg name in a function call, i.e. a NAME inside parens that is immediately
    #     followed by `=` (e.g. `foo(themes_and_fonts=...)` — the LHS is a parameter
    #     name, which does *not* refer to the module-level binding).
    rewrites = []
    prev_name = None
    paren_depth = 0
    for i, tok in enumerate(tokens):
        is_target = (tok.type == tokenize.NAME and tok.string in NAMES_TO_MIGRATE)
        if is_target:
            is_def_or_class = (prev_name is not None and prev_name.string in ("def", "class"))
            is_kwarg_name = False
            if paren_depth > 0 and i + 1 < len(tokens):
                nxt = tokens[i + 1]
                if nxt.type == tokenize.OP and nxt.string == "=":
                    is_kwarg_name = True
            skip = (tok.start[0] in global_lines_to_delete
                    or is_def_or_class
                    or is_kwarg_name)
            if not skip:
                rewrites.append((tok.start[0], tok.start[1], tok.end[1], f"app_state.{tok.string}"))
        if tok.type == tokenize.NAME:
            prev_name = tok
        if tok.type == tokenize.OP:
            if tok.string in ("(", "[", "{"):
                paren_depth += 1
            elif tok.string in (")", "]", "}"):
                paren_depth -= 1

    # Apply rewrites line by line (right to left on each line so column positions stay valid).
    lines = source.splitlines(keepends=True)
    by_line = defaultdict(list)
    for row, col_start, col_end, repl in rewrites:
        by_line[row].append((col_start, col_end, repl))

    new_lines = []
    for lineno, line in enumerate(lines, start=1):
        if lineno in global_lines_to_delete:
            continue
        if lineno in by_line:
            edits = sorted(by_line[lineno], key=lambda e: e[0], reverse=True)
            buf = line
            for col_start, col_end, repl in edits:
                buf = buf[:col_start] + repl + buf[col_end:]
            new_lines.append(buf)
        else:
            new_lines.append(line)
    return "".join(new_lines)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <path> <name> [<name>...]", file=sys.stderr)
        sys.exit(2)
    path = sys.argv[1]
    names = set(sys.argv[2:])
    with open(path, encoding="utf-8") as f:
        src = f.read()
    new_src = rewrite(src, names)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_src)
    print(f"Rewrote {path}  (migrated: {', '.join(sorted(names))})")
