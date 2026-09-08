"""Tests for `raven.avatar.characters` — finding a character by the name it goes by.

The whole point of the module is that a *name* is the only thing a chat message records, so everything
here is about what happens between a name and the files that depict it. Filesystem-only: no DPG, no server,
no chat.
"""

import json

import pytest

from raven.avatar import characters


def declare(directory, stem, metadata, *, image=True, icon=False, card=None):
    """Write one character declaration, plus whichever of its sidecars the test wants."""
    if metadata is not None:
        (directory / f"{stem}.json").write_text(json.dumps(metadata), encoding="utf-8")
    if image:
        (directory / f"{stem}.png").write_bytes(b"not really a png, and nothing here decodes one")
    if icon:
        (directory / f"{stem}_icon.png").write_bytes(b"nor this")
    if card is not None:
        (directory / f"{stem}_card.md").write_text(card, encoding="utf-8")


class TestFindingACharacter:
    def test_a_declaration_is_found_by_the_name_inside_it(self, tmp_path):
        # Not by its filename: `jj1.png` is the character *Juha*, which is the case the whole module
        # exists for and the one no filename rule recovers.
        declare(tmp_path, "jj1", {"name": "Juha"})
        found = characters.scan(tmp_path)
        assert list(found) == ["Juha"]
        assert found["Juha"].image_path.name == "jj1.png"

    def test_the_voice_comes_along(self, tmp_path):
        declare(tmp_path, "aria1", {"name": "Aria", "voice": "af_nova"})
        assert characters.scan(tmp_path)["Aria"].voice == "af_nova"

    def test_a_declaration_without_a_voice_says_so(self, tmp_path):
        """`None` rather than a guess: the caller's own default is the right answer, not ours."""
        declare(tmp_path, "aria1", {"name": "Aria"})
        assert characters.scan(tmp_path)["Aria"].voice is None

    def test_it_searches_recursively(self, tmp_path):
        # The shipped tree sorts characters into `other/`, `scientists/`, `tropes/` and so on.
        nested = tmp_path / "scientists"
        nested.mkdir()
        declare(nested, "jj1", {"name": "Juha"})
        assert list(characters.scan(tmp_path)) == ["Juha"]


class TestTheSidecarsAreFoundByConvention:
    """`aria1.png` -> `aria1_icon.png` and `aria1_card.md`, which is how the avatar loader finds its cels.

    Not named in the JSON, deliberately: one way to find a character's files rather than two.
    """

    def test_the_icon_is_found_when_it_is_there(self, tmp_path):
        declare(tmp_path, "aria1", {"name": "Aria"}, icon=True)
        assert characters.scan(tmp_path)["Aria"].icon_path.name == "aria1_icon.png"

    def test_a_character_without_an_icon_says_so(self, tmp_path):
        # An ordinary state, and the caller draws a generic glyph. `jj1` in the shipped tree is one.
        declare(tmp_path, "aria1", {"name": "Aria"}, icon=False)
        assert characters.scan(tmp_path)["Aria"].icon_path is None

    def test_the_card_is_read_on_demand(self, tmp_path):
        declare(tmp_path, "aria1", {"name": "Aria"}, card="You are {char}, and you talk to {user}.")
        character = characters.scan(tmp_path)["Aria"]
        assert character.card_path.name == "aria1_card.md"
        assert character.read_card() == "You are {char}, and you talk to {user}."

    def test_the_card_is_returned_as_a_template(self, tmp_path):
        """Unfilled. This module knows nothing about a chat, so who `{char}` is is not its question."""
        declare(tmp_path, "aria1", {"name": "Aria"}, card="I am {char}.")
        assert "{char}" in characters.scan(tmp_path)["Aria"].read_card()

    def test_a_character_without_a_card_says_so(self, tmp_path):
        declare(tmp_path, "aria1", {"name": "Aria"})
        character = characters.scan(tmp_path)["Aria"]
        assert character.card_path is None
        assert character.read_card() is None


class TestWhatIsNotACharacter:
    """The tree is shared with other files, and a bad one must cost its own character and no others."""

    def test_json_without_a_name_is_ignored(self, tmp_path):
        (tmp_path / "emotions.json").write_text('{"happy": {"eyebrow": 1.0}}', encoding="utf-8")
        declare(tmp_path, "aria1", {"name": "Aria"})
        assert list(characters.scan(tmp_path)) == ["Aria"], \
            "the stray file was taken for a character, or it took the real one down with it"

    def test_malformed_json_is_ignored(self, tmp_path):
        (tmp_path / "broken.json").write_text("{ this is not json", encoding="utf-8")
        declare(tmp_path, "aria1", {"name": "Aria"})
        assert list(characters.scan(tmp_path)) == ["Aria"]

    def test_a_character_with_no_face_is_still_a_character(self, tmp_path):
        """A face is one thing a character may have, not what makes it one.

        `raven-minichat` shows no avatar at all, and a Librarian may be run without one, so requiring an
        image would mean a faceless character could not be declared — and an undeclared character loses
        its card and its voice too, which is far more than the face.
        """
        declare(tmp_path, "sage", {"name": "Sage", "voice": "af_nova"}, image=False,
                card="You are {char}.")
        found = characters.scan(tmp_path)
        assert list(found) == ["Sage"]
        assert found["Sage"].image_path is None
        assert found["Sage"].voice == "af_nova", "the voice went with the missing face"
        assert found["Sage"].read_card() == "You are {char}.", "the card went with the missing face"

    def test_a_faceless_character_still_finds_its_sidecars(self, tmp_path):
        # They are keyed off the declaration's stem rather than the image's, which is what makes this work.
        declare(tmp_path, "sage", {"name": "Sage"}, image=False, icon=True, card="hello")
        sage = characters.scan(tmp_path)["Sage"]
        assert sage.icon_path.name == "sage_icon.png"
        assert sage.card_path.name == "sage_card.md"

    def test_one_name_declared_twice_keeps_the_first(self, tmp_path):
        declare(tmp_path, "aria1", {"name": "Aria"})
        declare(tmp_path, "aria2", {"name": "Aria"})
        found = characters.scan(tmp_path)
        assert list(found) == ["Aria"]
        assert found["Aria"].image_path.name == "aria1.png", "resolved in path order, so it is the first"


class TestTheShippedTree:
    """What Raven actually ships, which is the fixture no test can invent."""

    def test_the_two_used_characters_are_declared(self, tmp_path):
        found = characters.scan()
        assert "Aria" in found and "Juha" in found

    def test_aria_carries_a_voice_an_icon_and_a_card(self, tmp_path):
        aria = characters.scan()["Aria"]
        assert aria.image_path.name == "aria1.png"
        assert aria.voice is not None
        assert aria.icon_path is not None
        assert "{interaction_style}" in aria.read_card(), \
            "the card does not splice in the shared block, so every character would carry its own copy"

    def test_the_undeclared_characters_stay_undeclared(self):
        """Naming the rest of the cast needs an opinion nobody has spent yet; they fall back meanwhile."""
        assert characters.find("Nerd") is None


class TestFind:
    def test_a_name_nobody_claims_is_none(self):
        assert characters.find("Nobody At All") is None

    def test_none_in_none_out(self):
        # So a caller holding a message's recorded persona -- `None` for the roles with no character behind
        # them -- can ask without checking first.
        assert characters.find(None) is None


class TestTheScanIsCached:
    def test_rescan_picks_up_a_change(self, monkeypatch, tmp_path):
        """The cache is what makes a per-drawn-message lookup cheap; `rescan` is the way out of it."""
        declare(tmp_path, "aria1", {"name": "Aria"})
        monkeypatch.setattr(characters, "_cache", None)
        monkeypatch.setattr(characters, "assets_path", lambda *parts: tmp_path)
        assert list(characters.characters()) == ["Aria"]

        declare(tmp_path, "jj1", {"name": "Juha"})
        assert list(characters.characters()) == ["Aria"], "the scan was not cached at all"
        assert sorted(characters.rescan()) == ["Aria", "Juha"]
        monkeypatch.setattr(characters, "_cache", None)  # leave no test's tree in the process-wide cache


@pytest.fixture(autouse=True)
def _forget_the_cache():
    """The cache is module-global and outlives a test, so a tmp_path tree must not survive into the next."""
    characters._cache = None
    yield
    characters._cache = None
