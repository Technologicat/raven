#!/usr/bin/env python3
"""Which module-level imports would fail in CI, whose dependency list is hand-maintained.

CI installs a hand-picked subset (`.github/workflows/requirements-ci.txt` plus torch/torchvision pinned
inline in the workflows), so an import that is fine locally can be missing there — and the failure only
shows up on push. This walks every test module CI collects, follows the first-party imports, and reports
any *module-level* third-party import not satisfied by that list.

Function-local imports are excluded on purpose: they are the standard way to make a heavy or optional
dependency degrade gracefully, and flagging them would drown the real finding.
"""

import ast
import pathlib
import re
import subprocess
import sys

# Distribution name on PyPI -> the name you `import`. Only the ones that differ.
DIST_TO_IMPORT = {"pillow": "PIL", "pyyaml": "yaml", "python_docx": "docx", "python_pptx": "pptx",
                  "odfpy": "odf", "sseclient_py": "sseclient", "pytest_cov": "pytest_cov"}

# Installed transitively by something on the list, so present in CI without being named there.
TRANSITIVE = {"mcpyrate", "sympy"}  # both via unpythonic


def ci_import_names(root: pathlib.Path) -> set[str]:
    """Everything CI can import: the pinned requirements file, plus the workflow's inline `pip install`s."""
    names = set()

    def add(dist: str) -> None:
        dist = dist.strip().lower().replace("-", "_")
        if dist:
            names.add(DIST_TO_IMPORT.get(dist, dist))

    for line in (root / ".github/workflows/requirements-ci.txt").read_text().split("\n"):
        line = line.split("#")[0].strip()
        if line:
            add(re.split(r"[=<>;\[ ]", line)[0])

    # The torch trio is installed by a `pip install ... --index-url` line in the workflow rather than from
    # the requirements file (it needs PyTorch's own index). Read it from there rather than hardcoding a
    # copy here — a second hand-maintained list is the very thing this script exists to catch.
    for wf in ("ci.yml", "coverage.yml"):
        for line in (root / ".github/workflows" / wf).read_text().split("\n"):
            if "pip install" in line and "download.pytorch.org" in line:
                for token in line.split():
                    if "==" in token:
                        add(token.split("==")[0])

    names |= TRANSITIVE
    return {n.lower() for n in names}  # compared against a lowercased import name; `Pillow` -> `PIL` -> `pil`


def module_level_imports(path: pathlib.Path) -> set[str]:
    """Top-level-body imports only — not the ones inside functions."""
    out = set()
    for node in ast.parse(path.read_text(errors="replace")).body:
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0])
    return out


def first_party_deps(path: pathlib.Path, root: pathlib.Path) -> set[pathlib.Path]:
    """`raven.*` modules this file imports at module level, as paths."""
    deps, src = set(), path.read_text(errors="replace")
    for node in ast.parse(src).body:
        mods = []
        if isinstance(node, ast.ImportFrom):
            if node.level:  # relative
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
                if node.module:
                    base = base.joinpath(*node.module.split("."))
                mods += [base.joinpath(a.name) for a in node.names] + [base]
            elif node.module and node.module.startswith("raven"):
                base = root.joinpath(*node.module.split("."))
                # Both readings, because `from raven.librarian import llmclient` names a *module*, while
                # `from raven.librarian.chatutil import scrub` names a symbol inside one. Resolving only the
                # package silently skipped `llmclient.py` and every module imported that way — which is most
                # of them, and which made this script quietly unable to find anything.
                mods += [base.joinpath(a.name) for a in node.names] + [base]
        elif isinstance(node, ast.Import):
            mods += [root.joinpath(*a.name.split(".")) for a in node.names if a.name.startswith("raven")]
        for m in mods:
            for cand in (m.with_suffix(".py"), m / "__init__.py"):
                if cand.is_file():
                    deps.add(cand)
    return deps


def main() -> None:
    root = pathlib.Path(subprocess.run(["git", "rev-parse", "--show-toplevel"],
                                       capture_output=True, text=True).stdout.strip())
    allowed = ci_import_names(root)
    stdlib = set(sys.stdlib_module_names)

    findings: dict[str, list[str]] = {}
    n_modules = 0
    for test in sorted(root.glob("raven/**/tests/test_*.py")):
        # A module guarded by `importorskip` is allowed to need anything: CI skips it instead of erroring.
        # Without this the report is dominated by tests that are already correct — dearpygui, chromadb,
        # kokoro and the rest are all deliberately absent and deliberately guarded.
        #
        # The guard may live in a `conftest.py` rather than in the test file: `raven/client/tests/conftest.py`
        # guards the whole directory that way, on purpose, so each file does not need its own. Checking only
        # the test file reports those directories as broken while CI is green — which is how this script
        # learned to read conftests.
        sources = [test] + [p / "conftest.py" for p in test.parents
                            if (p / "conftest.py").is_file() and root in p.parents or p == root]
        if any(re.search(r"importorskip\(", s.read_text(errors="replace"))
               for s in sources if s.is_file()):
            continue

        seen, queue = set(), [test]
        while queue:
            f = queue.pop()
            if f in seen:
                continue
            seen.add(f)
            queue += [d for d in first_party_deps(f, root) if d not in seen]
            missing = sorted(m for m in module_level_imports(f)
                             if m not in stdlib and m != "raven" and m.lower() not in allowed)
            if missing:
                findings.setdefault(str(test.relative_to(root)), []).append(
                    f"{f.relative_to(root)} imports {missing}")
        n_modules += len(seen)

    if not findings:
        print(f"OK: {n_modules} module loads reachable from the unguarded tests; "
              "every module-level import is available in CI.")
        return
    print(f"{len(findings)} unguarded test module(s) would fail to collect in CI:\n")
    for test, reasons in sorted(findings.items()):
        print(f"  {test}")
        for r in reasons:
            print(f"      {r}")


if __name__ == "__main__":
    main()
