"""Tests for `raven.librarian.userprofile` — finding the user by the name they go by.

The mirror of `raven/avatar/tests/test_characters.py`, and deliberately the same shape: a name is what a
caller holds, and everything here is about getting from it to the card. Filesystem-only.

What differs from the character side is which states are ordinary. Nothing ships a user profile, so *no
directory at all* is the out-of-the-box case rather than a fault, and has to read as an empty result.
"""

import json

import pytest

from raven.librarian import userprofile


def declare(directory, stem, metadata, *, card=None):
    """Write one profile declaration, and its card if the test wants one."""
    (directory / f"{stem}.json").write_text(json.dumps(metadata), encoding="utf-8")
    if card is not None:
        (directory / f"{stem}.md").write_text(card, encoding="utf-8")


def declared(**fields):
    """A profile's metadata: the version key, plus whatever the test is about."""
    return {userprofile.VERSION_KEY: userprofile.FORMAT_VERSION, **fields}


class TestFindingTheUser:
    def test_a_profile_is_found_by_the_name_inside_it(self, tmp_path):
        declare(tmp_path, "jj", declared(name="Juha"))
        found = userprofile.scan(tmp_path)
        assert list(found) == ["Juha"]

    def test_the_card_is_found_beside_it(self, tmp_path):
        declare(tmp_path, "jj", declared(name="Juha"), card="{user} prefers metric units.")
        profile = userprofile.scan(tmp_path)["Juha"]
        assert profile.card_path.name == "jj.md"
        assert profile.read_card() == "{user} prefers metric units."

    def test_the_card_is_returned_as_a_template(self, tmp_path):
        """Unfilled: who `{char}` is comes from somewhere this module knows nothing about."""
        declare(tmp_path, "jj", declared(name="Juha"), card="{user} is talking to {char}.")
        assert "{char}" in userprofile.scan(tmp_path)["Juha"].read_card()

    def test_the_icon_is_found_beside_it(self, tmp_path):
        """Same `_icon.png` convention a character's icon follows, so both sides look alike."""
        declare(tmp_path, "jj", declared(name="Juha"))
        (tmp_path / "jj_icon.png").write_bytes(b"not really a png")
        assert userprofile.scan(tmp_path)["Juha"].icon_path.name == "jj_icon.png"

    def test_a_profile_without_an_icon_says_so(self, tmp_path):
        # An ordinary state, and the caller draws the generic user glyph -- which is what every user got
        # before 0.2.9, there having been nowhere to declare another.
        declare(tmp_path, "jj", declared(name="Juha"))
        assert userprofile.scan(tmp_path)["Juha"].icon_path is None

    def test_a_profile_without_a_card_says_so(self, tmp_path):
        declare(tmp_path, "jj", declared(name="Juha"))
        profile = userprofile.scan(tmp_path)["Juha"]
        assert profile.card_path is None
        assert profile.read_card() is None

    def test_a_name_can_be_selected_from_several(self, tmp_path):
        """The reason it is a directory rather than one file: a work profile and a personal one."""
        declare(tmp_path, "work", declared(name="Juha at work"), card="Talk shop.")
        declare(tmp_path, "home", declared(name="Juha"), card="Talk anything.")
        found = userprofile.scan(tmp_path)
        assert sorted(found) == ["Juha", "Juha at work"]
        assert found["Juha at work"].read_card() == "Talk shop."


class TestTheOrdinaryStateIsHavingNone:
    """Nothing ships a profile, so an absence is the shipped configuration rather than a fault."""

    def test_a_missing_directory_is_empty_rather_than_an_error(self, tmp_path):
        assert userprofile.scan(tmp_path / "does_not_exist") == {}

    def test_an_empty_directory_is_empty(self, tmp_path):
        assert userprofile.scan(tmp_path) == {}

    def test_a_name_nobody_claims_is_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(userprofile, "_cache", {})
        assert userprofile.find("Nobody At All") is None

    def test_none_in_none_out(self, tmp_path, monkeypatch):
        monkeypatch.setattr(userprofile, "_cache", {})
        assert userprofile.find(None) is None


class TestWhatIsNotAProfile:
    def test_json_without_the_version_key_is_not_one(self, tmp_path):
        # Same guard as the character side, and for the same reason: anything might carry a `name`.
        (tmp_path / "settings.json").write_text('{"name": "not a person"}', encoding="utf-8")
        declare(tmp_path, "jj", declared(name="Juha"))
        assert list(userprofile.scan(tmp_path)) == ["Juha"]

    def test_a_profile_with_no_usable_name_is_ignored(self, tmp_path):
        declare(tmp_path, "nameless", {userprofile.VERSION_KEY: userprofile.FORMAT_VERSION})
        declare(tmp_path, "jj", declared(name="Juha"))
        assert list(userprofile.scan(tmp_path)) == ["Juha"]

    def test_a_profile_from_a_newer_raven_is_ignored(self, tmp_path):
        declare(tmp_path, "future", {userprofile.VERSION_KEY: userprofile.FORMAT_VERSION + 1,
                                     "name": "From The Future"})
        declare(tmp_path, "jj", declared(name="Juha"))
        assert list(userprofile.scan(tmp_path)) == ["Juha"]

    def test_malformed_json_is_ignored(self, tmp_path):
        (tmp_path / "broken.json").write_text("{ not json", encoding="utf-8")
        declare(tmp_path, "jj", declared(name="Juha"))
        assert list(userprofile.scan(tmp_path)) == ["Juha"]

    def test_one_name_declared_twice_keeps_the_first(self, tmp_path):
        declare(tmp_path, "a_first", declared(name="Juha"), card="first")
        declare(tmp_path, "b_second", declared(name="Juha"), card="second")
        assert userprofile.scan(tmp_path)["Juha"].read_card() == "first", \
            "resolved in path order, so it is the first"


@pytest.fixture(autouse=True)
def _forget_the_cache():
    """The cache is module-global and outlives a test, so a tmp_path tree must not survive into the next."""
    userprofile._cache = None
    yield
    userprofile._cache = None
