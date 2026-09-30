"""`parse_color`: every spelling a `<font color=...>` attribute may carry, normalized to RGBA."""

import pytest

pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from raven.vendor.DearPyGui_Markdown.font_attributes import parse_color  # noqa: E402 -- after importorskip by design


class TestParseColor:
    def test_hex_is_padded_to_opaque_rgba(self):
        assert parse_color("#ff0000") == [255, 0, 0, 255]

    def test_hex_alpha_is_kept(self):
        assert parse_color("#ff000080") == [255, 0, 0, 128]

    def test_an_all_digit_hex_is_a_colour(self):
        """Evaluated as a literal first, as it once was, this became the integer 123456 and failed."""
        assert parse_color("123456") == [0x12, 0x34, 0x56, 255]

    def test_literal(self):
        assert parse_color("(10, 20, 30)") == [10, 20, 30, 255]

    def test_a_sequence_is_truncated_and_padded(self):
        assert parse_color((1, 2, 3, 4, 5)) == [1, 2, 3, 4]
        assert parse_color([7, 8]) == [7, 8, 255, 255]

    def test_a_string_that_is_not_a_colour_raises(self):
        with pytest.raises(ValueError):
            parse_color("banana")
