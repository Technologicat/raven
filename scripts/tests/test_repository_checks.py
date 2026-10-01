"""The repository checks CI runs in its lint job, run as part of a local `pytest`.

So that a push cannot fail on them for want of having run them: the test suite is what gets run before a
push, and the checks are a separate habit that keeps being skipped. CI runs `pytest raven/`, so this module
runs locally only, the lint job running the same checks there.

The list of checks is read out of the workflow's *Repository checks* step rather than written here, so a
checker added there is run here without anyone remembering to.
"""

import pathlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

# Needs the linters themselves, which a development install has and a minimal one may not.
_NEEDS = {"check_lint_canary.py": ("ruff", "pycodestyle")}


def _checks_in_the_workflow() -> list[str]:
    """Return the `scripts/check_*.py` that the workflow's *Repository checks* step runs, in its order."""
    text = WORKFLOW.read_text(encoding="utf-8")
    match = re.search(r"- name: Repository checks\n\s+run: \|\n((?:\s+python scripts/\S+\n)+)", text)
    assert match is not None, f"no 'Repository checks' step in {WORKFLOW}; update the pattern here if it was renamed"
    return re.findall(r"python scripts/(\S+\.py)", match.group(1))


_CHECKS = _checks_in_the_workflow()


def test_the_workflow_lists_some_checks():
    # Without this, a pattern that stopped matching would collect no checks below and pass vacuously.
    assert len(_CHECKS) >= 5, f"found only {_CHECKS} in the workflow's Repository checks step"


@pytest.mark.parametrize("script", _CHECKS)
def test_repository_check_passes(script):
    for tool in _NEEDS.get(script, ()):
        if shutil.which(tool) is None:
            pytest.skip(f"{script} needs `{tool}`, which is not installed here")
    result = subprocess.run([sys.executable, f"scripts/{script}"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, f"{script} failed:\n{result.stdout}{result.stderr}"
