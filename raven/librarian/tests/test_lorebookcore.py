"""Tests for `raven.librarian.lorebookcore`: the conformance cases, and that it stands alone, as a port takes it.

The cases are `lorebook_conformance.json`, beside this file, which a port to another frontend runs too; this is
Raven's runner for them. Read it as the reference for what each section's fields mean.

A Python port copies this file unchanged, with the cases beside it, into a `tests/` directory beside its copy of
`lorebookcore.py`, and runs it under pytest's `--import-mode=importlib`, as Raven does.
"""

import ast
import json
import pathlib
import subprocess
import sys

import pytest

from .. import lorebookcore

CORE = pathlib.Path(lorebookcore.__file__)
CASES = json.loads((pathlib.Path(__file__).parent / "lorebook_conformance.json").read_text(encoding="utf-8"))


def _text(glossary: str | list[str]) -> str:
    """A case's glossary: the name of one in the file's `glossaries`, or its own lines."""
    lines = CASES["glossaries"][glossary] if isinstance(glossary, str) else glossary
    return "\n".join(lines) + "\n"

def _entries(glossary: str | list[str]) -> dict[str, lorebookcore.Entry]:
    return {entry.id: entry for entry in lorebookcore.parse(_text(glossary), name="glossary")}

def _expansions(raw: dict | None) -> dict[str, lorebookcore.Expansion]:
    """Expansions as a generator suggests them, cleaned as the core cleans them."""
    return {word: lorebookcore.clean_expansion(word, given["forms"], given["stems"]) for word, given in (raw or {}).items()}

def _cases(section: str) -> list:
    return [pytest.param(case, id=case["name"]) for case in CASES[section]]


class TestConformance:
    @pytest.mark.parametrize("case", _cases("parse"))
    def test_parse(self, case):
        entries = lorebookcore.parse(_text(case["glossary"]), name=case["file_name"], entry_level=case["entry_level"],
                                     first_order=case["first_order"])
        assert [entry.title for entry in entries] == case["titles"]
        by_id = {entry.id: entry for entry in entries}
        for entry_id, wanted in case.get("entries", {}).items():
            entry = by_id[entry_id]
            for key, value in wanted.items():
                if key == "content_includes":
                    assert all(part in entry.content for part in value), entry.content
                elif key == "content_ends_with":
                    assert entry.content.endswith(value), entry.content
                elif key == "content_not_ends_with":
                    assert not entry.content.endswith(value), entry.content
                elif key == "anchors_include":
                    assert set(value) <= set(entry.anchors), entry.anchors
                elif key in ("aliases", "crossrefs"):
                    assert list(getattr(entry, key)) == value
                else:
                    assert getattr(entry, key) == value, key

    @pytest.mark.parametrize("title, slug", CASES["slugify"])
    def test_slugify(self, title, slug):
        assert lorebookcore.slugify(title) == slug

    @pytest.mark.parametrize("case", _cases("load"))
    def test_load(self, case, tmp_path):
        for name, lines in case["files"].items():
            (tmp_path / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        entries, problems = lorebookcore.load(tmp_path, entry_levels=case.get("entry_levels"))
        assert list(entries) == case["ids"]
        assert sorted(problems) == case["problem_files"]
        if "orders" in case:
            assert [entry.order for entry in entries.values()] == case["orders"]
        for entry_id, content in case.get("contents", {}).items():
            assert entries[entry_id].content == content

    @pytest.mark.parametrize("case", _cases("find_matches"))
    def test_find_matches(self, case):
        found = lorebookcore.find_matches(_entries(case["glossary"]), case["text"],
                                          expansions=_expansions(case["expansions"]))
        assert sorted(found) == case["ids"]
        for entry_id, wording in case.get("wording", {}).items():
            assert found[entry_id].wording == wording

    @pytest.mark.parametrize("case", _cases("clean_expansion"))
    def test_clean_expansion(self, case):
        cleaned = lorebookcore.clean_expansion(case["word"], case["forms"], case["stems"])
        assert {"forms": list(cleaned.forms), "stems": list(cleaned.stems)} == case["expect"]

    @pytest.mark.parametrize("case", _cases("select"))
    def test_select(self, case):
        assert CASES["token_count"].startswith("characters"), "this runner counts tokens as characters"
        firings = lorebookcore.select(_entries(case["glossary"]), case["text"], count_tokens=len,
                                      budget=case["budget"], max_depth=case["max_depth"],
                                      already_fired=set(case["already_fired"]),
                                      expansions=_expansions(case["expansions"]))
        assert [{"id": firing.entry.id, "depth": firing.depth, "wording": firing.maybe_wording, "via": firing.maybe_via}
                for firing in firings] == case["firings"]

    @pytest.mark.parametrize("case", _cases("lookup"))
    def test_lookup(self, case):
        entries = _entries(case["glossary"])
        expansions = _expansions(case["expansions"])
        found = lorebookcore.lookup(entries, case["term"], expansions=expansions)
        assert (found.maybe_entry.id if found.maybe_entry else None) == case["entry"]
        assert found.exact == case["exact"]
        assert sorted(found.candidates) == case["candidates"]
        assert list(found.mentioned_in) == case["mentioned_in"]
        answer = lorebookcore.answer(entries, case["term"], expansions=expansions)
        if "answer" in case:
            assert answer == case["answer"]
        if "answer_starts_with" in case:
            assert answer.startswith(case["answer_starts_with"]), answer
        for part in case.get("answer_contains", ()):
            assert part in answer, answer


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
