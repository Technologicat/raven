"""Unit tests for raven.librarian.chattextures."""

import pytest

pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from unpythonic.env import env  # noqa: E402 -- the class; `from unpythonic import env` gets the submodule

from raven.librarian import chattextures  # noqa: E402 -- after importorskip by design


class TestTheSpeakerGlyphFollowsTheStoredCharacter:
    """`icon_texture_for`, which both the chat log and the chat graph ask.

    A chat holds turns by whichever characters wrote them, and the character configured *now* is not who
    wrote the older ones. Keyed by role alone — which is how this worked until 0.2.9 — there is exactly
    one slot for the AI's face, so a stored "Juha" message was drawn wearing Aria's.

    Nothing here needs widgets: the textures are opaque handles and the method only chooses between them.
    """

    @staticmethod
    def _glyphs(monkeypatch, configured="Aria", character_icon=None,
                    configured_user="Juha", user_icon=None):
        """A `SpeakerGlyphs` with just the fields the resolver reads, and no textures loaded.

        `character_icon`, `user_icon`: what `__init__` would have set for a character or a user that ships
                                       an icon of its own, as an *instance* attribute shadowing the class's
                                       generic one. `None` leaves the generic showing, which is the case
                                       for somebody without one.
        """
        glyphs = chattextures.SpeakerGlyphs.__new__(chattextures.SpeakerGlyphs)
        # On the class, because that is where `_load_class_textures` puts them and where the fallback
        # reads them from. `monkeypatch` puts them back, so no other test inherits them.
        for attribute, value in (("icon_ai_texture", "tex_generic_ai"),
                                 ("icon_user_texture", "tex_generic_user")):
            monkeypatch.setattr(chattextures.SpeakerGlyphs, attribute, value, raising=False)
        # Only the two roles that have no speaker; the other two are resolved per persona.
        glyphs._role_icon_textures = {"system": "tex_system", "tool": "tex_tool"}
        glyphs.llm_settings = env(personas={"assistant": configured, "user": configured_user,
                                                "system": None, "tool": None})
        if character_icon is not None:
            glyphs.icon_ai_texture = character_icon
        if user_icon is not None:
            glyphs.icon_user_texture = user_icon
        return glyphs

    def test_the_configured_character_wears_its_own_face(self, monkeypatch):
        glyphs = self._glyphs(monkeypatch, configured="Aria", character_icon="tex_aria")
        assert glyphs.icon_texture_for("assistant", "Aria") == "tex_aria"

    def test_another_character_does_not_wear_it(self, monkeypatch):
        """The defect this exists to fix, with its own control beside it.

        The first assertion is the control: a resolver that answered the generic glyph for *everything*
        would satisfy the second one while fixing nothing, and would look exactly like a pass.
        """
        glyphs = self._glyphs(monkeypatch, configured="Aria", character_icon="tex_aria")
        assert glyphs.icon_texture_for("assistant", "Aria") == "tex_aria", \
            "the configured character has no icon of its own here, so borrowing it cannot be detected"
        assert glyphs.icon_texture_for("assistant", "Juha") == "tex_generic_ai"

    def test_an_assistant_message_with_no_recorded_character_gets_the_generic_glyph(self, monkeypatch):
        # The defensive branch rather than a case anyone meets: every payload gets a persona written with
        # it. Pinned because what it must not do is guess — drawing the configured character's face here
        # would assert something nothing recorded, which is the defect this whole method exists to remove.
        glyphs = self._glyphs(monkeypatch, configured="Aria", character_icon="tex_aria")
        assert glyphs.icon_texture_for("assistant", None) == "tex_generic_ai"

    def test_a_character_without_an_icon_of_its_own_gets_the_generic_one(self, monkeypatch):
        """Which is what happens today for such a character when it is the configured one."""
        glyphs = self._glyphs(monkeypatch, configured="Aria", character_icon=None)
        assert glyphs.icon_texture_for("assistant", "Aria") == "tex_generic_ai"

    def test_the_configured_user_wears_their_own_face(self, monkeypatch):
        """The user side, resolved exactly as the character side is — a conversation has two participants.

        Before 0.2.9 the user always got the generic glyph, there being nowhere to declare another; a
        profile can now carry a `_icon.png` the way a character does.
        """
        glyphs = self._glyphs(monkeypatch, configured_user="Juha", user_icon="tex_juha")
        assert glyphs.icon_texture_for("user", "Juha") == "tex_juha"

    def test_another_user_name_does_not_wear_it(self, monkeypatch):
        # Same reasoning as for a character, and the same control beside it: the configured user's turns
        # are theirs, and a turn stored under another name is somebody else's.
        glyphs = self._glyphs(monkeypatch, configured_user="Juha", user_icon="tex_juha")
        assert glyphs.icon_texture_for("user", "Juha") == "tex_juha", \
            "the configured user has no icon of their own here, so borrowing it cannot be detected"
        assert glyphs.icon_texture_for("user", "somebody else entirely") == "tex_generic_user"

    def test_a_user_without_a_profile_icon_gets_the_generic_one(self, monkeypatch):
        """Which is what every user got before 0.2.9, and what one without a profile still gets."""
        glyphs = self._glyphs(monkeypatch, configured_user="Juha", user_icon=None)
        assert glyphs.icon_texture_for("user", "Juha") == "tex_generic_user"

    def test_the_roles_with_no_speaker_answer_from_the_role_alone(self, monkeypatch):
        """A system prompt and a tool result are nobody's, so one glyph each is the whole answer.

        Two of the four roles have a speaker and two do not, and this is the half that does not: a persona
        must not change what is drawn for them.
        """
        glyphs = self._glyphs(monkeypatch, character_icon="tex_aria", user_icon="tex_juha")
        for role, expected in (("system", "tex_system"), ("tool", "tex_tool")):
            assert glyphs.icon_texture_for(role, None) == expected
            assert glyphs.icon_texture_for(role, "somebody else entirely") == expected, \
                f"a persona changed the glyph for role '{role}'"

    def test_an_unknown_role_draws_nothing(self, monkeypatch):
        glyphs = self._glyphs(monkeypatch)
        assert glyphs.icon_texture_for("narrator", None) is None
