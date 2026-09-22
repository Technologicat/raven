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


# The repository's own import roots. `scripts` is one because `scripts/tests/` imports the checkers it
# tests, and a first-party name must not be reported as a package CI forgot to install.
FIRST_PARTY = ("raven", "scripts")


def first_party_deps(path: pathlib.Path, root: pathlib.Path) -> set[pathlib.Path]:
    """The repository's own modules that this file imports at module level, as paths."""
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
            elif node.module and node.module.split(".")[0] in FIRST_PARTY:
                base = root.joinpath(*node.module.split("."))
                # Both readings, because `from raven.librarian import llmclient` names a *module*, while
                # `from raven.librarian.chatutil import scrub` names a symbol inside one. Resolving only the
                # package silently skipped `llmclient.py` and every module imported that way — which is most
                # of them, and which made this script quietly unable to find anything.
                mods += [base.joinpath(a.name) for a in node.names] + [base]
        elif isinstance(node, ast.Import):
            mods += [root.joinpath(*a.name.split(".")) for a in node.names if a.name.split(".")[0] in FIRST_PARTY]
        for m in mods:
            for cand in (m.with_suffix(".py"), m / "__init__.py"):
                if cand.is_file():
                    deps.add(cand)
    return deps


_IMPORTORSKIP = re.compile(r"""importorskip\(\s*["']([^"']+)["']""")


def guard_targets(sources: list[pathlib.Path]) -> set[str]:
    """Every module named in an `importorskip(...)` across `sources`."""
    found = set()
    for source in sources:
        if source.is_file():
            found |= set(_IMPORTORSKIP.findall(source.read_text(errors="replace")))
    return found


def guard_fires_in_ci(target: str, root: pathlib.Path, allowed: set[str], stdlib: set[str]) -> bool:
    """Would `importorskip(target)` actually skip the module in CI?

    A guard naming something CI *has* protects nothing. This is the whole of what the check used to get
    wrong: a module carrying any `importorskip` at all was exempted, so one guarded on `dearpygui` — which
    CI installs — was free to go on and import the ML stack, and the report came back green while the push
    came back red.

    Third-party: it fires when the package is absent from CI's list. First-party: it fires when *that*
    module's own transitive imports are not satisfied there, which is how `importorskip` on a `raven`
    module works at all — the import runs and raises from somewhere further down.
    """
    top = target.split(".")[0]
    if top not in FIRST_PARTY:
        return top not in stdlib and top.lower() not in allowed
    for candidate in (root.joinpath(*target.split(".")).with_suffix(".py"),
                      root.joinpath(*target.split(".")) / "__init__.py"):
        if candidate.is_file():
            return bool(unsatisfied_imports(candidate, root, allowed, stdlib))
    return False  # names nothing that exists, so it cannot be relied on to skip


def unsatisfied_imports(start: pathlib.Path, root: pathlib.Path,
                        allowed: set[str], stdlib: set[str]) -> list[str]:
    """Walk first-party imports from `start`; report every module-level import CI could not satisfy."""
    findings, seen, queue = [], set(), [start]
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        queue += [d for d in first_party_deps(current, root) if d not in seen]
        missing = sorted(m for m in module_level_imports(current)
                         if m not in stdlib and m not in FIRST_PARTY and m.lower() not in allowed)
        if missing:
            findings.append(f"{current.relative_to(root)} imports {missing}")
    return findings


def test_modules(root: pathlib.Path) -> list[pathlib.Path]:
    """Every tracked test module, from wherever in the repository it lives.

    Asking git rather than globbing a named tree, because pytest has no `testpaths` setting: it collects
    from the whole repository, so a `scripts/tests/` module is collected in CI exactly like a package one.
    This globbed `raven/**/tests/test_*.py` until `scripts/tests/` appeared in September 2026 and landed
    outside it — a test there needing something CI lacks would have passed this check and failed on push,
    which is the one failure this script exists to prevent. Naming the trees that have tests today would
    reintroduce that the next time one appears; git knows them all.
    """
    tracked = subprocess.run(["git", "-C", str(root), "ls-files"],
                             capture_output=True, text=True).stdout.split()
    return sorted(root / p for p in tracked
                  if (q := pathlib.PurePosixPath(p)).parent.name == "tests"
                  and q.name.startswith("test_") and q.suffix == ".py")


def main() -> None:
    root = pathlib.Path(subprocess.run(["git", "rev-parse", "--show-toplevel"],
                                       capture_output=True, text=True).stdout.strip())
    allowed = ci_import_names(root)
    stdlib = set(sys.stdlib_module_names)

    findings: dict[str, list[str]] = {}
    n_modules = 0
    for test in test_modules(root):
        # A module whose `importorskip` actually fires in CI is allowed to need anything: CI skips it
        # instead of erroring. Without this the report is dominated by tests that are already correct —
        # dearpygui, chromadb, kokoro and the rest are all deliberately absent and deliberately guarded.
        #
        # **A guard naming something CI has protects nothing**, which is why this asks whether each one
        # fires rather than whether one exists. A module guarded on `dearpygui` — installed in CI — went
        # on to import the ML stack, and this script called it fine while the push came back red.
        #
        # The guard may live in a `conftest.py` rather than in the test file: `raven/client/tests/conftest.py`
        # guards the whole directory that way, on purpose, so each file does not need its own. Checking only
        # the test file reports those directories as broken while CI is green — which is how this script
        # learned to read conftests.
        sources = [test] + [p / "conftest.py" for p in test.parents
                            if (p / "conftest.py").is_file() and root in p.parents or p == root]
        if any(guard_fires_in_ci(target, root, allowed, stdlib)
               for target in guard_targets(sources)):
            continue

        for finding in unsatisfied_imports(test, root, allowed, stdlib):
            findings.setdefault(str(test.relative_to(root)), []).append(finding)
        n_modules += 1

    if not findings:
        print(f"OK: {n_modules} test module{'s' if n_modules != 1 else ''} whose guards would not fire "
              "in CI, and everything they reach; every module-level import is available there.")
        return
    print(f"{len(findings)} unguarded test module{'s' if len(findings) != 1 else ''} would fail to collect in CI:\n")
    for test, reasons in sorted(findings.items()):
        print(f"  {test}")
        for r in reasons:
            print(f"      {r}")
    sys.exit(1)  # as every other checker here does, so this one can gate a push too


if __name__ == "__main__":
    main()
