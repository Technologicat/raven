"""Tests for `raven.librarian.lorebookcore`: that it stands alone, as a port to another frontend takes it."""

import ast
import pathlib
import subprocess
import sys

from raven.librarian import lorebookcore

CORE = pathlib.Path(lorebookcore.__file__)


class TestStandalone:
    def test_it_imports_only_the_standard_library(self):
        tree = ast.parse(CORE.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, f"a relative import ties the core to its package: line {node.lineno}"
                imported.add(node.module.split(".")[0])
        assert "re" in imported, "no imports were found, so this test is not reading the module"
        assert imported <= sys.stdlib_module_names, f"not in the standard library: {imported - sys.stdlib_module_names}"

    def test_it_loads_by_path_in_an_interpreter_without_raven(self, tmp_path):
        script = ("import importlib.util, sys\n"
                  f"spec = importlib.util.spec_from_file_location('lorebookcore', {str(CORE)!r})\n"
                  "core = importlib.util.module_from_spec(spec)\n"
                  "spec.loader.exec_module(core)\n"
                  "entries = {e.id: e for e in core.parse('## Hydra tax\\nA.\\n', name='t')}\n"
                  "assert list(core.find_matches(entries, 'a hydra tax')) == ['hydra-tax']\n"
                  "assert not any(name.startswith('raven') for name in sys.modules), 'something of Raven was imported'\n"
                  "print('ok')\n")
        # `-I` keeps the current directory and the user's site-packages off the path, and the working directory
        # is not the repository, so nothing of Raven's can be found.
        result = subprocess.run([sys.executable, "-I", "-c", script], cwd=tmp_path, capture_output=True, text=True)
        assert result.stdout.strip() == "ok", result.stderr
