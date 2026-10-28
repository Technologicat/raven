"""Tests for `raven.papers.scopebib.definition`: reading and checking a definition file."""

import pathlib
import textwrap

import pytest

from raven.papers.scopebib import definition as d

MINIMAL = """
scope = "studies of ravens"

[[test]]
name = "not_birds"
asks = "true if the work is not about birds"
"""

# Top-level keys before any table: after `[[test]]`, TOML would read `hints` as a key of that test.
FULL = 'hints = "Ravens and crows are easy to confuse."\n' + MINIMAL + """
[[field]]
name = "species"
asks = "which corvid the work studies"
quote = true
[field.values]
raven = "Corvus corax"
crow = "Corvus corone"

[[field]]
name = "method"
asks = "how the birds were studied"
max_words = 6

[[rule]]
label = "crows, stated in the text"
when = { species = ["crow"] }
action = "remove"
"""


def write(tmp_path, text, name="corpus.scope.toml"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


class TestLoading:
    def test_a_minimal_definition(self, tmp_path):
        definition = d.load_definition(write(tmp_path, MINIMAL))
        assert definition.scope == "studies of ravens"
        assert definition.tests == (d.Test("not_birds", "true if the work is not about birds"),)
        assert (definition.hints, definition.frame, definition.fields, definition.rules) == ("", None, (), ())

    def test_every_closed_vocabulary_ends_in_not_stated(self, tmp_path):
        species, method = d.load_definition(write(tmp_path, FULL)).fields
        assert list(species.values) == ["raven", "crow", d.NOT_STATED]
        assert species.quote == {"raven", "crow"}, "`quote = true` is every value but not_stated"
        assert species.max_words is None
        assert method.values is None and method.max_words == 6 and method.quote == frozenset()

    def test_quote_can_name_the_values_that_need_one(self, tmp_path):
        text = FULL.replace("quote = true", 'quote = ["crow"]')
        assert text != FULL, "this fixture changes nothing"
        species, _ = d.load_definition(write(tmp_path, text)).fields
        assert species.quote == {"crow"}

    def test_a_rule_reads_as_sets_of_values(self, tmp_path):
        (rule,) = d.load_definition(write(tmp_path, FULL)).rules
        assert rule == d.Rule("crows, stated in the text", {"species": frozenset({"crow"})}, "remove")

    def test_a_str_path_works(self, tmp_path):
        assert d.load_definition(str(write(tmp_path, MINIMAL))).scope == "studies of ravens"


class TestTheInstrument:
    """What changes the fingerprint and what does not."""

    def materials(self, tmp_path, text):
        definition = d.load_definition(write(tmp_path, text))
        return d.judge_material(definition), d.extract_material(definition)

    def test_rewrapping_prose_and_editing_comments_change_nothing(self, tmp_path):
        rewrapped = FULL.replace('asks = "true if the work is not about birds"',
                                 'asks = """\ntrue if the work\n   is not about birds\n"""  # reworded later')
        assert rewrapped != FULL, "this fixture rewraps nothing"
        assert self.materials(tmp_path, rewrapped) == self.materials(tmp_path, FULL)

    def test_a_paragraph_break_is_kept(self, tmp_path):
        two = FULL.replace('hints = "Ravens and crows are easy to confuse."',
                           'hints = """\nRavens and crows.\n\nAre easy to confuse.\n"""')
        definition = d.load_definition(write(tmp_path, two))
        assert definition.hints == "Ravens and crows.\n\nAre easy to confuse."

    def test_changing_what_is_asked_changes_its_half(self, tmp_path):
        judge, extract = self.materials(tmp_path, FULL)
        new_judge, new_extract = self.materials(tmp_path, FULL.replace("not about birds", "not about corvids"))
        assert new_judge != judge and new_extract == extract
        new_judge, new_extract = self.materials(tmp_path, FULL.replace("Corvus corone", "Corvus brachyrhynchos"))
        assert new_judge == judge and new_extract != extract

    def test_a_rule_is_in_neither(self, tmp_path):
        changed = FULL.replace('action = "remove"', 'action = "hold"')
        assert self.materials(tmp_path, changed) == self.materials(tmp_path, FULL)


class TestScopeNotes:
    NOTES = 'scope_notes = """\nCorvus corax only.\n\nCrows are out.\n"""\n'

    def test_they_are_read_and_absent_is_empty(self, tmp_path):
        assert d.load_definition(write(tmp_path, MINIMAL)).scope_notes == ""
        definition = d.load_definition(write(tmp_path, self.NOTES + MINIMAL))
        assert definition.scope_notes == "Corvus corax only.\n\nCrows are out."

    def test_they_go_after_the_first_paragraph(self, tmp_path):
        definition = d.load_definition(write(tmp_path, self.NOTES + MINIMAL))
        placed = d.with_scope_notes("Search for: ravens.\n\nAnswer each.\n\nItems.", definition)
        assert placed == ("Search for: ravens.\n\n" + d.SCOPE_NOTES_HEADING + "\nCorvus corax only.\n\nCrows are out."
                          "\n\nAnswer each.\n\nItems.")
        assert d.with_scope_notes("One paragraph.", definition).startswith("One paragraph.\n\n" + d.SCOPE_NOTES_HEADING)

    def test_without_them_a_prompt_is_untouched(self, tmp_path):
        definition = d.load_definition(write(tmp_path, MINIMAL))
        assert d.with_scope_notes("Search for: ravens.\n\nAnswer.", definition) == "Search for: ravens.\n\nAnswer."

    def test_without_them_the_material_does_not_mention_them(self, tmp_path):
        # So a definition written before the key existed keeps its fingerprints, and its saved answers.
        definition = d.load_definition(write(tmp_path, FULL))
        assert d.scope_notes_material(definition) == {}
        assert "scope_notes" not in d.judge_material(definition) and "scope_notes" not in d.extract_material(definition)

    def test_with_them_both_halves_change(self, tmp_path):
        plain = d.load_definition(write(tmp_path, FULL))
        noted = d.load_definition(write(tmp_path, self.NOTES + FULL, name="noted.scope.toml"))
        assert d.judge_material(noted) != d.judge_material(plain)
        assert d.extract_material(noted) != d.extract_material(plain)


class TestErrors:
    """Every error names the file and the entry, so it can be fixed without reading this code."""

    @pytest.mark.parametrize("text, expected", [
        (MINIMAL + 'scop = "typo"\n', "unknown key scop"),
        ('[[test]]\nname = "a"\nasks = "b"\n', "missing scope"),
        ('scope = "s"\ntest = []\n', "at least one [[test]]"),
        (MINIMAL + '[[test]]\nname = "not_birds"\nasks = "again"\n', "test 2 'not_birds': the name 'not_birds' is used twice"),
        ('scope = "s"\n[[test]]\nname = "two words"\nasks = "x"\n', "must be a word"),
        ('scope = "s"\n[[test]]\nname = "a"\nasks = "  "\n', "test 1 'a', asks: must be a non-empty string"),
        (MINIMAL + '[[field]]\nname = "f"\nasks = "x"\n[field.values]\nnot_stated = "x"\n', "part of every vocabulary"),
        (MINIMAL + '[[field]]\nname = "f"\nasks = "x"\nmax_words = 3\n[field.values]\na = "x"\n', "`max_words` is for free text"),
        (MINIMAL + '[[field]]\nname = "f"\nasks = "x"\nmax_words = 0\n', "positive whole number"),
        (MINIMAL + '[[field]]\nname = "f"\nasks = "x"\nquote = true\n', "`quote` needs `values`"),
        (FULL.replace("quote = true", 'quote = ["owl"]'), "`quote` names 'owl'"),
        (FULL.replace("quote = true", 'quote = ["not_stated"]'), "has nothing to quote"),
        (FULL.replace("quote = true", 'quote = "crow"'), "`quote` must be true, false, or a list"),
        (FULL + '[[rule]]\nlabel = "m"\nwhen = { method = ["x"] }\naction = "hold"\n', "not a field with `values`"),
        (FULL + '[[rule]]\nlabel = "m"\nwhen = { species = ["owl"] }\naction = "hold"\n', "'owl', which is not among its values"),
        (FULL + '[[rule]]\nlabel = "m"\nwhen = { species = ["crow"] }\naction = "drop"\n', "`action` must be one of remove, hold"),
        ('scope = "s"\n[[test\n', "not valid TOML"),
    ])
    def test_a_bad_definition_is_refused_with_a_reason(self, tmp_path, text, expected):
        path = write(tmp_path, text)
        with pytest.raises(d.DefinitionError) as info:
            d.load_definition(path)
        assert expected in str(info.value)
        assert str(path) in str(info.value)

    def test_a_top_level_key_below_the_first_entry_is_named_as_misplaced(self, tmp_path):
        with pytest.raises(d.DefinitionError, match="hints belongs at the top level: move it above"):
            d.load_definition(write(tmp_path, MINIMAL + 'hints = "written too late"\n'))

    def test_a_missing_file(self, tmp_path):
        with pytest.raises(d.DefinitionError, match="cannot be read"):
            d.load_definition(tmp_path / "absent.scope.toml")

    def test_not_stated_may_be_named_in_a_rule(self, tmp_path):
        text = FULL + '[[rule]]\nlabel = "unstated"\nwhen = { species = ["not_stated"] }\naction = "hold"\n'
        assert d.load_definition(write(tmp_path, text)).rules[-1].when == {"species": frozenset({"not_stated"})}


def test_the_aokk_definition_stays_valid():
    """The prototype's corpus, in the tool's format. Lives in the investigation bundle, so it is absent from
    an installed package, where this skips."""
    path = pathlib.Path(__file__).resolve().parents[4] / "investigations" / "aokk-corpus-scope" / "aokk.scope.toml"
    if not path.exists():
        pytest.skip(f"{path} is not here; it ships with the repository, not the package")
    definition = d.load_definition(path)
    assert [test.name for test in definition.tests] == ["no_ai", "not_education", "wrong_level"]
    assert [rule.action for rule in definition.rules] == ["remove", "remove", "hold", "hold"]
