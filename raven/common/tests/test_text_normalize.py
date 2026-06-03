"""Unit tests for raven.common.text.normalize."""

from raven.common import text

ZWSP = chr(0x200B)             # zero-width space
ZWNJ = chr(0x200C)            # zero-width non-joiner
ZWJ = chr(0x200D)            # zero-width joiner
WJ = chr(0x2060)             # word joiner
MVS = chr(0x180E)            # Mongolian vowel separator
BOM = chr(0xFEFF)            # zero-width no-break space / BOM
COMBINING_ACUTE = chr(0x0301)


class TestTagCharacters:
    def test_ascii_smuggler_block_stripped(self):
        # A hidden instruction encoded in the U+E0000–U+E007F shadow-ASCII block.
        smuggled = "".join(chr(0xE0000 + ord(c)) for c in "ignore previous")
        assert text.normalize(f"Hello{smuggled} world") == "Hello world"

    def test_block_boundaries(self):
        assert text.normalize(f"a{chr(0xE0000)}{chr(0xE007F)}b") == "ab"
        # Just outside the block is left alone.
        assert text.normalize(f"a{chr(0xE0080)}b") == f"a{chr(0xE0080)}b"


class TestZeroWidth:
    def test_zero_width_chars_stripped(self):
        dirty = f"in{ZWSP}str{ZWNJ}uct{ZWJ}ions{WJ}{MVS} here"
        assert text.normalize(dirty) == "instructions here"

    def test_embedded_zwnbsp_stripped(self):
        assert text.normalize(f"a{BOM}b") == "ab"

    def test_single_leading_bom_preserved(self):
        assert text.normalize(f"{BOM}hello") == f"{BOM}hello"

    def test_second_leading_bom_treated_as_injection(self):
        # First BOM is the legitimate file-start marker; the second is an embedded ZWNBSP.
        assert text.normalize(f"{BOM}{BOM}hello") == f"{BOM}hello"


class TestControlCharacters:
    def test_tab_lf_cr_preserved(self):
        s = "line1\nline2\tcol\r\nline3"
        assert text.normalize(s) == s

    def test_c0_controls_stripped(self):
        assert text.normalize("a\x00\x01\x07\x1fb") == "ab"

    def test_del_and_c1_controls_stripped(self):
        assert text.normalize("a\x7f\x80\x9fb") == "ab"


class TestNFC:
    def test_combining_sequence_folded(self):
        # "e" + combining acute (U+0301) -> precomposed "é" (U+00E9).
        decomposed = f"e{COMBINING_ACUTE}"
        result = text.normalize(decomposed)
        assert result == "é"
        assert len(result) == 1

    def test_standalone_combining_mark_preserved(self):
        # No precomposed "d with acute" exists; NFC leaves the sequence, and we don't strip combining marks.
        s = f"d{COMBINING_ACUTE}"
        assert text.normalize(s) == s


class TestProseUntouched:
    def test_scientific_prose_unchanged(self):
        s = "The mollifier ψ(x) = exp(−1/xᵐ)·[x>0] is C∞ but non-analytic at 0.\n\nSee §2."
        assert text.normalize(s) == s

    def test_idempotent(self):
        dirty = f"He{ZWSP}llo{chr(0xE0041)}\x00 w{COMBINING_ACUTE}orld"
        once = text.normalize(dirty)
        assert text.normalize(once) == once
