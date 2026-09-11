"""Pins what `raven.visualizer.importer` is allowed to do when it is merely imported.

It is a library module, and says so: *"used both by the visualizer GUI … and by the `raven-importer` CLI
shell … this module stays free of argparse and logging configuration so the GUI can host it safely."* For a
while it was not. A module-level block connected to the LLM backend and called `sys.exit(255)` when none
answered, so with `clusters_keyword_method = "llm"` and the backend down:

  - `raven-visualizer` died at startup — no window, no message, exit 255 — because `app.py` imports the
    pipeline at module level. The app was killed by a feature that only matters during an import, which
    most sessions never do.
  - `pytest` could not even *collect* the suite: the interpreter exited mid-collection, an INTERNALERROR
    rather than a failure, taking every unrelated test with it.

**These read `importer.py` as source rather than importing it**, for two reasons. Importing is what the bug
did, so a test that imports either passes or takes the runner down with it — and reading the source needs
none of sklearn, torch or spaCy, so this runs in CI, where the module itself cannot.
"""

import ast
import pathlib

import pytest

_IMPORTER = pathlib.Path(__file__).resolve().parent.parent / "importer.py"

#: Calls that end the process. A library module may not make one at all: it is the caller's business
#: whether a failure is fatal, and both frontends have somewhere better to put the news.
_PROCESS_ENDING = frozenset({"sys.exit", "exit", "quit", "os._exit"})


@pytest.fixture(scope="module")
def module_body():
    """The statements of `importer.py` at module level — what runs on `import raven.visualizer.importer`."""
    return ast.parse(_IMPORTER.read_text(encoding="utf-8")).body


def _calls_in(nodes):
    """Yield every call node anywhere under `nodes`, as its unparsed function expression."""
    for node in nodes:
        for descendant in ast.walk(node):
            if isinstance(descendant, ast.Call):
                yield ast.unparse(descendant.func)


def _module_level_statements(body):
    """The statements that run on import: everything except the `def`s and `class`es, which only bind."""
    return [node for node in body
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]


def test_importing_the_pipeline_cannot_end_the_process(module_body):
    """The regression. `raven-visualizer` refused to start while this was false."""
    offenders = sorted(set(_calls_in(_module_level_statements(module_body))) & _PROCESS_ENDING)
    assert not offenders, f"importer.py ends the process at module level ({', '.join(offenders)}); importing a library module must not be able to do that"


def test_importing_the_pipeline_does_not_reach_the_network(module_body):
    """The other half: the exit was only fatal because a connection attempt preceded it.

    Even with the exit gone, connecting at import time would make `raven-visualizer` wait on a dead host
    before its first frame — the timeout rather than the crash, which is quieter and just as wrong.
    """
    reaching = sorted({call for call in _calls_in(_module_level_statements(module_body))
                       if "test_connection" in call or call.endswith("llmclient.setup") or call.endswith("backend_status")})
    assert not reaching, f"importer.py talks to the LLM backend at module level ({', '.join(reaching)}); that belongs in `_setup_llm_backend`, which a run calls"


def test_the_backend_check_still_happens_before_the_expensive_stages(module_body):
    """Moving the check must not have quietly removed it: an hour-long run has to fail in its first second.

    Asserts *position* as well as presence — ahead of the embeddings, which are the first costly stage and
    the thing a late diagnosis would waste.
    """
    import_bibtex = next(node for node in module_body
                         if isinstance(node, ast.FunctionDef) and node.name == "import_bibtex")
    calls = [ast.unparse(node.func) for node in ast.walk(import_bibtex) if isinstance(node, ast.Call)]
    assert "_setup_llm_backend" in calls, "nothing sets up the LLM backend, so a configured LLM stage would fail per cluster instead of once at the start"
    assert calls.index("_setup_llm_backend") < calls.index("_get_highdim_semantic_vectors"), \
        "the backend is checked after the embeddings are computed, so a dead backend now costs the user that time before saying so"


def test_the_failure_is_reported_as_an_exception_the_frontends_can_catch(module_body):
    """`LLMBackendUnavailable` is the contract that replaced the exit; both frontends are written to it."""
    names = [node.name for node in module_body if isinstance(node, ast.ClassDef)]
    assert "LLMBackendUnavailable" in names
