"""Unit tests for raven.common.text.shorten."""

import pytest

from raven.common import text

#: A monospace measurer: every character one unit wide. Lets a width budget be read as a character count.
def _monospace(s: str) -> float:
    return float(len(s))

#: A proportional measurer, "W" twice as wide as anything else. Used where a test has to be able to tell a
#: real measurement apart from a character count.
def _proportional(s: str) -> float:
    return float(sum(2 if c == "W" else 1 for c in s))


class TestEllipsize:
    def test_text_that_fits_is_returned_unchanged(self):
        assert text.ellipsize("short.pdf", 22) == "short.pdf"
        assert text.ellipsize("short.pdf", 22, middle=True) == "short.pdf"

    def test_exactly_the_budget_is_not_cut(self):
        assert text.ellipsize("abcde", 5) == "abcde"

    def test_the_end_goes_by_default(self):
        assert text.ellipsize("abcdefghij", 5) == "abcd…"

    def test_the_middle_goes_on_request_keeping_both_ends(self):
        """The topic is at the front of a filename and the file type at the back; both have to survive."""
        result = text.ellipsize("quarterly_report_2026_final.pdf", 22, middle=True)
        assert len(result) == 22
        assert result.startswith("quarterly_r")
        assert result.endswith(".pdf")

    @pytest.mark.parametrize("max_chars", range(1, 40))
    @pytest.mark.parametrize("middle", [False, True])
    def test_the_budget_is_never_exceeded(self, max_chars, middle):
        # A budget of one leaves room for the ellipsis and nothing else, which is the tightest case that
        # still says something; below that is covered separately.
        long_text = "the quick brown fox jumps over the lazy dog"
        assert len(text.ellipsize(long_text, max_chars, middle=middle)) <= max_chars

    @pytest.mark.parametrize("max_chars", [0, -1, -100])
    @pytest.mark.parametrize("middle", [False, True])
    def test_no_budget_yields_nothing(self, max_chars, middle):
        assert text.ellipsize("anything at all", max_chars, middle=middle) == ""

    def test_a_budget_of_one_is_the_ellipsis_alone(self):
        assert text.ellipsize("anything at all", 1) == "…"
        assert text.ellipsize("anything at all", 1, middle=True) == "…"

    def test_whitespace_is_taken_back_from_the_cut(self):
        # "the …" reads as a writer trailing off; "the…" reads as text that continues. The result is then
        # shorter than the budget, which is allowed — the budget is a ceiling.
        assert text.ellipsize("the quick brown fox", 5) == "the…"
        # Both edges of a middle cut, the tail's leading space going as well as the head's trailing one.
        assert text.ellipsize("aa bb cc dd", 7, middle=True) == "aa…dd"

    def test_the_empty_string_survives_any_budget(self):
        assert text.ellipsize("", 10) == ""
        assert text.ellipsize("", 0) == ""


class TestEllipsizeToWidth:
    def test_text_that_fits_is_returned_unchanged(self):
        assert text.ellipsize_to_width("abcde", 10.0, _monospace) == "abcde"

    def test_exactly_the_budget_is_not_cut(self):
        assert text.ellipsize_to_width("abcde", 5.0, _monospace) == "abcde"

    def test_text_that_does_not_fit_is_cut_and_marked(self):
        result = text.ellipsize_to_width("abcdefghij", 5.0, _monospace)
        assert result == "abcd…"
        assert _monospace(result) <= 5.0

    def test_the_budget_is_measured_rather_than_counted(self):
        # Six characters, ten units wide, against a budget of six. A character count would say it fits.
        wide = "WWWWWW"
        assert len(wide) == 6
        assert _proportional(wide) == 12.0, "this fixture cannot tell a measurement from a character count"
        assert text.ellipsize_to_width(wide, 6.0, _proportional) != wide
        assert _proportional(text.ellipsize_to_width(wide, 6.0, _proportional)) <= 6.0

    def test_a_budget_too_narrow_for_anything_still_says_something(self):
        # One character plus the ellipsis, overflowing, rather than an empty label: a box this narrow
        # cannot show text at all, and reporting that honestly beats showing nothing.
        assert text.ellipsize_to_width("abcdef", 0.5, _monospace) == "a…"


class TestLongestPrefixThatFits:
    def test_the_whole_text_when_it_fits(self):
        assert text.longest_prefix_that_fits("abcde", 10.0, _monospace) == "abcde"

    def test_the_longest_prefix_when_it_does_not(self):
        assert text.longest_prefix_that_fits("abcdefghij", 4.0, _monospace) == "abcd"

    def test_at_least_one_character_always(self):
        # A caller stepping through a word must be able to make progress; an empty answer would spin.
        assert text.longest_prefix_that_fits("abcdef", 0.0, _monospace) == "a"
        assert text.longest_prefix_that_fits("abcdef", -5.0, _monospace) == "a"

    @pytest.mark.parametrize("budget", [1.0, 2.5, 4.0, 7.0, 11.0, 13.5])
    def test_the_binary_search_agrees_with_a_walk(self, budget):
        # The reference is the obvious implementation the search replaced, so a disagreement is the
        # search's. Proportional widths, so the two cannot agree merely by both counting characters.
        subject = "aWbWcWdefWghi"
        walked = subject[:1]
        for length in range(1, len(subject) + 1):
            if _proportional(subject[:length]) <= budget:
                walked = subject[:length]
        assert text.longest_prefix_that_fits(subject, budget, _proportional) == walked
