"""Whether `scripts/check_todo_structure.py` would notice a `TODO_DEFERRED.md` whose introduction was broken into.

Covers the intro-end marker only, the newest of the checks. Each case feeds the checker a file broken on
purpose, against a baseline that must come out clean, or the broken cases prove nothing.
"""

from scripts.check_todo_structure import INTRO_END_MARKER, check

ITEM = """\
## An item

*Cluster: x · Cost: S · Gate: none · Filed: 2026-10-07*

Body.
"""

# Two paragraphs of introduction, then the marker, then an item: the shape the real file has.
GOOD = f"""\
# Deferred TODOs

New items go at the top.

A second paragraph of introduction.

{INTRO_END_MARKER}

{ITEM}"""


def _check(tmp_path, text: str) -> list[str]:
    path = tmp_path / "TODO_DEFERRED.md"
    path.write_text(text, encoding="utf-8")
    return check(path)


class TestTheIntroEndMarker:
    def test_the_baseline_is_clean(self, tmp_path):
        assert _check(tmp_path, GOOD) == []

    def test_an_item_inserted_after_the_first_intro_paragraph_is_reported(self, tmp_path):
        broken = GOOD.replace("A second paragraph", ITEM.replace("An item", "Misplaced") + "\nA second paragraph")
        assert broken != GOOD
        complaints = _check(tmp_path, broken)
        assert any("Misplaced" in complaint and "introduction" in complaint for complaint in complaints), complaints

    def test_a_missing_marker_is_reported(self, tmp_path):
        assert any("appears 0 times" in complaint for complaint in _check(tmp_path, GOOD.replace(INTRO_END_MARKER, "")))

    def test_a_second_marker_is_reported(self, tmp_path):
        doubled = GOOD + f"\n{INTRO_END_MARKER}\n"
        assert any("appears 2 times" in complaint for complaint in _check(tmp_path, doubled))
