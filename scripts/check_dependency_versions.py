#!/usr/bin/env python3
"""Whether `pyproject.toml`, `pdm.lock` and the CI pin list still agree about versions.

Raven states its dependencies in three places, each answering a different question, and nothing keeps
them consistent:

  - `pyproject.toml` says what an *installer* may resolve — floors and the occasional exact pin.
  - `pdm.lock` says what a *developer* gets — one resolved version per package.
  - `.github/workflows/requirements-ci.txt` says what *CI* tests — pinned exactly, because an unpinned
    CI environment drifts without any commit saying so.

Three ways they come apart, one per section of the report:

  1. **Violations** — a locked or CI-pinned version that `pyproject.toml` does not permit. The serious
     one: it means the tests are not running against what the metadata promises to ship.
  2. **Undeclared** — a package CI pins that `pyproject.toml` never names. It reaches CI transitively,
     and keeps working until whichever package was dragging it in stops doing so.
  3. **Lagging floors** — a floor older than what actually resolves. Harmless in itself, and the reason
     this script prints `old -> new`: raising them is a read-off, not a research task.
  4. **Unbounded** — a requirement naming no version at all, which bets that every release ever made
     works. Each row says whether it is a runtime or a dev dependency, because that is what separates
     the ones worth acting on: Emacs integration tooling has no bearing on what Raven does and can sit
     here forever, while a *runtime* entry in this section is a bet nobody placed on purpose.
  5. **URL-pinned** — a requirement naming an artifact directly. Sometimes the only way to get a thing
     (spaCy publishes its models as release assets, not on PyPI), and sometimes a workaround for
     upstream breakage that was meant to be temporary. Either way resolution never touches it, so it is
     the one kind of dependency that cannot go stale loudly.

Nothing here is edited automatically. The floors carry rationale a rewrite would flatten, and section 1
usually wants a decision rather than a bump.
"""

import pathlib
import re
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

ROOT = pathlib.Path(__file__).resolve().parent.parent
CI_REQUIREMENTS = ROOT / ".github" / "workflows" / "requirements-ci.txt"


def canon(name: str) -> str:
    """PyPI's own name comparison: case-insensitive, with `-`, `_` and `.` all equivalent."""
    return re.sub(r"[-_.]+", "-", name).lower()


def declared() -> dict[str, tuple[Requirement, str]]:
    """What `pyproject.toml` permits, each paired with the table it came from: runtime or dev.

    The table is carried because it is what distinguishes an unbounded requirement worth looking at
    from one that is merely untidy — see the report's last section.

    Requirements pointing at a URL are excluded here and reported on their own by `url_requirements`:
    a direct reference names one artifact, so there is no range for the version checks to work with.
    """
    return {canon(req.name): (req, group)
            for group, req in _requirements() if req.url is None}


def url_requirements() -> list[tuple[str, str]]:
    """Requirements pinned to a URL rather than resolved from an index, as `(name, url)`.

    Worth a section of its own because the usual reason for one is a workaround — a fix that is merged
    upstream but unreleased, a fork carrying a patch — and a workaround pinned by URL has nothing to
    make it expire. It bypasses resolution entirely, so no upgrade ever touches it and no failing check
    ever mentions it.
    """
    return [(req.name, req.url) for _, req in _requirements() if req.url is not None]


def _requirements() -> list[tuple[str, Requirement]]:
    """Every declared requirement as `(group, requirement)`, runtime first, then the dev group."""
    pp = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return [(group, Requirement(spec))
            for group, specs in (("runtime", pp["project"]["dependencies"]),
                                 ("dev", pp.get("dependency-groups", {}).get("dev", [])))
            for spec in specs]


def locked() -> dict[str, Version]:
    """What `pdm.lock` resolves, or an empty mapping when the lockfile is absent.

    Raven gitignores its lockfile, so a fresh clone legitimately has none — the other two sections of
    the report still work without it.
    """
    path = ROOT / "pdm.lock"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    return {canon(m.group(1)): Version(m.group(2))
            for m in re.finditer(r'^name = "([^"]+)"\nversion = "([^"]+)"', text, re.M)}


def ci_pinned() -> dict[str, list[tuple[Version, str]]]:
    """What CI installs, per package, as `(version, environment marker)` pairs.

    A package may appear more than once under different markers — numpy does, because 2.5 raised its
    Python floor above the 3.11 matrix entry — so each pin is kept with the marker that selects it.
    """
    out: dict[str, list[tuple[Version, str]]] = {}
    for raw in CI_REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        req = Requirement(line)
        pins = [s.version for s in req.specifier if s.operator == "=="]
        if not pins:  # a floor or range in the CI list would not be a pin; nothing to compare
            continue
        try:
            version = Version(pins[0])
        except InvalidVersion:
            continue
        out.setdefault(canon(req.name), []).append((version, str(req.marker) if req.marker else ""))
    return out


def main() -> None:
    decls, locks, pins = declared(), locked(), ci_pinned()

    violations, undeclared, lagging, unbounded = [], [], [], []

    for name, versions in sorted(pins.items()):
        entry = decls.get(name)
        if entry is None:
            undeclared.append((name, ", ".join(str(v) for v, _ in versions)))
            continue
        req, _ = entry
        for version, marker in versions:
            if version not in req.specifier:
                where = f"CI [{marker}]" if marker else "CI"
                violations.append((name, f"{where} pins {version}, but {req} forbids it"))

    for name, version in sorted(locks.items()):
        entry = decls.get(name)
        if entry is None:  # transitive; pyproject has no opinion about it
            continue
        req, _ = entry
        if version not in req.specifier:
            violations.append((name, f"pdm.lock has {version}, but {req} forbids it"))

    # A floor lags when it sits below *every* version anything actually exercises — so the bar is the
    # oldest of them, not the newest. CI and the lockfile routinely differ by a patch release (whichever
    # was refreshed last), and a floor between the two is correct: raising it to the newer would declare
    # a minimum that the other one then falls below. Taking the maximum here would report those pairs
    # forever, and a section that is never empty is a section nobody reads.
    for name, (req, _) in sorted(decls.items()):
        if not req.specifier:  # nothing to be stale; the unbounded section covers these
            continue
        tested = [v for v, _ in pins.get(name, [])] + ([locks[name]] if name in locks else [])
        if tested and min(tested) > _floor(req.specifier):
            lagging.append((name, str(req), f">={min(tested)}"))

    for name, (req, group) in sorted(decls.items()):
        if not req.specifier:
            unbounded.append((name, group))

    _report("Constraint violations", violations,
            "these are what 'CI is not testing what we ship' looks like")
    _report("Pinned in CI, undeclared in pyproject.toml", undeclared,
            "present transitively, so it works until the package dragging it in stops")
    _report("Floors older than what resolves", lagging,
            "raise them when there is no reason to keep supporting the older release")
    _report("Declares no version at all", unbounded,
            "a 'runtime' here is the one to look at; dev-group editor tooling belongs in this list")
    _report("Pinned to a URL", url_requirements(),
            "resolution never touches these, so nothing makes a temporary workaround expire")

    if violations:
        sys.exit(1)


def _floor(specifier: SpecifierSet) -> Version:
    """The lowest version a specifier admits, or version 0 when it names no lower bound.

    Only `>=` and `==` establish one for our purposes; `>` and `~=` do not appear in Raven's metadata,
    and guessing at their boundaries would report bumps nobody asked for.
    """
    floors = [Version(s.version) for s in specifier if s.operator in (">=", "==")]
    return max(floors) if floors else Version("0")


def _report(title: str, rows: list[tuple], hint: str) -> None:
    if not rows:
        print(f"OK  {title}: none.")
        return
    print(f"\n{title} ({len(rows)}) — {hint}:\n")
    for row in rows:
        print(f"  {row[0]:28} {'  '.join(str(c) for c in row[1:])}")
    print()


if __name__ == "__main__":
    main()
