"""Part C, mechanical half (v2): which of each item's concrete references no longer exist anywhere.

v1 flagged every bare basename (`config.py`, `app.py`) as missing because it checked them against the repo
root. A reference is only evidence of staleness if *nothing* in the tree answers to it, so: resolve paths by
basename across all tracked files, and let a symbol be satisfied by a filename as well as by file content
(many references are to test modules, whose name is the thing being named).

Still evidence, not verdicts.
"""

import json
import pathlib
import re
import subprocess
from functools import lru_cache

TODO = pathlib.Path("TODO_DEFERRED.md")
DUMP = pathlib.Path("/tmp/todo_sweep_references.json")  # scratch; the README beside this script is the artifact

BACKTICKED = re.compile(r"`([^`]+)`")
PATHLIKE = re.compile(r"^[\w./-]+\.(py|md|json|toml|yml|yaml|txt|cfg)$")
SYMBOLISH = re.compile(r"^[a-zA-Z_][\w]*(\.[a-zA-Z_][\w]*)*(\(\))?$")

tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split()
tracked_set = set(tracked)
basenames = {}
for f in tracked:
    basenames.setdefault(pathlib.PurePath(f).name, []).append(f)
all_names_blob = "\n".join(tracked)


def path_exists(tok):
    if "/" in tok:
        return tok in tracked_set or pathlib.Path(tok).exists()
    return tok in basenames


@lru_cache(maxsize=None)
def symbol_exists(sym):
    leaf = sym.split(".")[-1]
    if leaf in basenames or f"{leaf}.py" in basenames:
        return True
    if leaf in all_names_blob:
        return True
    out = subprocess.run(["git", "grep", "-l", "-F", leaf], capture_output=True, text=True, check=False)
    return bool(out.stdout.strip())


def sections(lines):
    idxs = [i for i, ln in enumerate(lines) if ln.startswith("## ")]
    for n, i in enumerate(idxs):
        end = idxs[n + 1] if n + 1 < len(idxs) else len(lines)
        yield lines[i][3:].strip(), lines[i + 1:end]


def main():
    lines = TODO.read_text().splitlines()
    rows = []
    for heading, body in sections(lines):
        if heading in ("Declined", "Waiting on upstream"):
            continue
        tokens = [t.strip() for t in BACKTICKED.findall("\n".join(body))]
        paths = [t for t in tokens if PATHLIKE.match(t)]
        symbols = [t.rstrip("()") for t in tokens
                   if not PATHLIKE.match(t) and SYMBOLISH.match(t) and len(t.split(".")[-1]) > 3]
        paths = list(dict.fromkeys(paths))
        symbols = list(dict.fromkeys(symbols))[:8]

        gone_paths = [p for p in paths if not path_exists(p)]
        gone_syms = [s for s in symbols if not symbol_exists(s)]
        rows.append({"heading": heading, "gone_paths": gone_paths, "gone_symbols": gone_syms,
                     "n_refs": len(paths) + len(symbols)})

    with DUMP.open("w") as f:
        json.dump(rows, f, indent=1)
    flagged = [r for r in rows if r["gone_paths"] or r["gone_symbols"]]
    print(f"items scanned: {len(rows)}")
    print(f"items referencing something that no longer exists: {len(flagged)}\n")
    for r in flagged:
        print(f"- {r['heading'][:76]}")
        if r["gone_paths"]:
            print(f"    gone paths:   {r['gone_paths']}")
        if r["gone_symbols"]:
            print(f"    gone symbols: {r['gone_symbols']}")


main()
