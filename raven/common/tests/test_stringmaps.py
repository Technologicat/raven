"""Unit tests for raven.common.stringmaps."""

from raven.common import stringmaps


class TestSubscriptRoundtrip:
    def test_numbers(self):
        for ch, sub in stringmaps.regular_to_subscript_numbers.items():
            assert stringmaps.subscript_to_regular_numbers[sub] == ch

    def test_symbols(self):
        for ch, sub in stringmaps.regular_to_subscript_symbols.items():
            assert stringmaps.subscript_to_regular_symbols[sub] == ch

    def test_letters_roundtrip_except_phi_collision(self):
        # "ϕ" (symbol phi, U+03D5) and "φ" (letter phi, U+03C6) both map to "ᵩ"; the
        # inverse keeps whichever was inserted last — letter phi — so sym-phi is a
        # documented known loss in the inverse. All other letters round-trip cleanly.
        for ch, sub in stringmaps.regular_to_subscript_letters.items():
            if ch == "ϕ":
                assert stringmaps.subscript_to_regular_letters[sub] == "φ"
            else:
                assert stringmaps.subscript_to_regular_letters[sub] == ch

    def test_combined_map_is_union(self):
        assert set(stringmaps.regular_to_subscript.keys()) == (
            set(stringmaps.regular_to_subscript_numbers.keys())
            | set(stringmaps.regular_to_subscript_symbols.keys())
            | set(stringmaps.regular_to_subscript_letters.keys())
        )

    def test_numbers_match_unicode_codepoints(self):
        # SUBSCRIPT DIGIT ZERO is U+2080; "N" digits below start at that code point.
        for n in range(10):
            assert stringmaps.regular_to_subscript_numbers[str(n)] == chr(0x2080 + n)


class TestSuperscriptRoundtrip:
    def test_numbers(self):
        for ch, sup in stringmaps.regular_to_superscript_numbers.items():
            assert stringmaps.superscript_to_regular_numbers[sup] == ch

    def test_symbols(self):
        for ch, sup in stringmaps.regular_to_superscript_symbols.items():
            assert stringmaps.superscript_to_regular_symbols[sup] == ch

    def test_letters(self):
        for ch, sup in stringmaps.regular_to_superscript_letters.items():
            assert stringmaps.superscript_to_regular_letters[sup] == ch

    def test_combined_map_is_union(self):
        assert set(stringmaps.regular_to_superscript.keys()) == (
            set(stringmaps.regular_to_superscript_numbers.keys())
            | set(stringmaps.regular_to_superscript_symbols.keys())
            | set(stringmaps.regular_to_superscript_letters.keys())
        )

    def test_full_roundtrip(self):
        for ch, sup in stringmaps.regular_to_superscript.items():
            assert stringmaps.superscript_to_regular[sup] == ch

    def test_digit_two_and_three_are_latin1_codepoints(self):
        # Historical quirk: ², ³ live in Latin-1 (U+00B2, U+00B3), not in the main superscript block.
        assert stringmaps.regular_to_superscript_numbers["2"] == "\u00b2"
        assert stringmaps.regular_to_superscript_numbers["3"] == "\u00b3"


class TestTranslationTableUsage:
    # The dicts are designed to be fed to `str.translate(str.maketrans(...))`.

    def test_string_translate_to_subscript(self):
        table = str.maketrans(stringmaps.regular_to_subscript)
        assert "x2 + 1".translate(table) == "ₓ₂ ₊ ₁"

    def test_string_translate_to_superscript(self):
        table = str.maketrans(stringmaps.regular_to_superscript)
        assert "x2 + 1".translate(table) == "x² ⁺ ¹"  # lowercase x has no superscript form → passes through

    def test_inverse_translate(self):
        table = str.maketrans(stringmaps.subscript_to_regular)
        assert "ₓ₂ ₊ ₁".translate(table) == "x2 + 1"


class TestFilenameSafeNonAlphanum:
    def test_contains_dot(self):
        # Important for `raven.papers.identifiers` compatibility (see inline comment).
        assert "." in stringmaps.filename_safe_nonalphanum

    def test_no_path_separators(self):
        for sep in ("/", "\\", ":"):
            assert sep not in stringmaps.filename_safe_nonalphanum
