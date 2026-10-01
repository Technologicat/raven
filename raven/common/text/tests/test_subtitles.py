"""Unit tests for raven.common.text.subtitles — cutting a sentence into subtitle cards, and timing them.

Widths are measured with `len`, a monospace font in effect, so that what fits is exact and readable off
the test.
"""

import pytest

from raven.common.text.subtitles import (Card, wrap_lines, split_into_cards,
                                         card_times_from_words, card_times_proportional)


class TestWrapLines:
    def test_short_text_is_one_line(self):
        assert wrap_lines("one two", 20, len) == ["one two"]

    def test_wraps_at_spaces(self):
        assert wrap_lines("aaa bbb ccc ddd", 7, len) == ["aaa bbb", "ccc ddd"]

    def test_a_word_wider_than_the_line_gets_a_line_of_its_own(self):
        assert wrap_lines("a enormousword b", 5, len) == ["a", "enormousword", "b"]

    def test_empty(self):
        assert wrap_lines("", 10, len) == []


class TestSplitIntoCards:
    def test_text_that_fits_is_one_card(self):
        text = "A short sentence."
        assert split_into_cards(text, 40, len) == [Card(0, text)]

    def test_offsets_point_into_the_text(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        cards = split_into_cards(text, 12, len, max_lines=1)
        assert len(cards) > 1
        for card in cards:
            assert text[card.offset:card.offset + len(card.text)] == card.text
        assert " ".join(card.text for card in cards) == text, "a word was lost or repeated at a cut"

    def test_every_card_fits(self):
        text = " ".join(f"word{i}" for i in range(40))
        for card in split_into_cards(text, 20, len, max_lines=2):
            assert len(wrap_lines(card.text, 20, len)) <= 2

    def test_cuts_at_a_clause_boundary_when_one_is_past_halfway(self):
        # Without the comma rule, the first card would take as much as fits: "one two three, four five".
        text = "one two three, four five six seven eight"
        cards = split_into_cards(text, 24, len, max_lines=1)
        assert len(wrap_lines("one two three, four five", 24, len)) == 1, \
            "the fixture no longer fits more than the clause on a line, so it cannot tell the two rules apart"
        assert cards[0].text == "one two three,"

    def test_ignores_a_clause_boundary_that_would_leave_the_card_less_than_half_full(self):
        text = "one, two three four five six seven eight nine"
        cards = split_into_cards(text, 24, len, max_lines=1)
        assert cards[0].text != "one,"
        assert len(cards[0].text) > 12

    def test_a_standalone_dash_opens_the_next_card(self):
        text = "one two three four — five six seven eight"
        cards = split_into_cards(text, 20, len, max_lines=1)
        assert cards[0].text == "one two three four"
        assert cards[1].text.startswith("—")

    def test_an_oversized_word_is_a_card_by_itself(self):
        text = "a b supercalifragilisticexpialidocious c d"
        cards = split_into_cards(text, 10, len, max_lines=1)
        assert Card(text.index("super"), "supercalifragilisticexpialidocious") in cards

    def test_empty(self):
        assert split_into_cards("   ", 10, len) == []


# A synthesizer's view of `_TEXT`: punctuation as tokens of its own, as Kokoro reports it, and a token with
# no timing, which `WordTiming` allows.
_TEXT = "Hello there, how are you doing today, my friend?"
_WORDS = [("Hello", 0.2, 0.5), ("there", 0.5, 0.8), (",", 0.8, 0.85), ("how", 0.85, 1.0),
          ("are", 1.0, 1.1), ("you", 1.1, 1.3), ("doing", 1.3, 1.6), ("today", 1.6, 2.0),
          (",", 2.0, 2.05), ("my", 2.05, 2.2), ("friend", 2.2, 2.6), ("?", None, None)]


class TestCardTimesFromWords:
    def test_each_card_goes_up_with_its_first_word(self):
        cards = [Card(0, "Hello there,"), Card(13, "how are you doing today,"), Card(38, "my friend?")]
        assert [_TEXT[card.offset:card.offset + len(card.text)] for card in cards] == [card.text for card in cards]
        assert card_times_from_words(_TEXT, cards, _WORDS) == [0.0, 0.85, 2.05]

    def test_the_first_card_goes_up_with_the_speech(self):
        cards = [Card(0, "Hello there,"), Card(13, "how are you doing today, my friend?")]
        assert card_times_from_words(_TEXT, cards, _WORDS)[0] == 0.0

    def test_a_card_with_no_word_found_falls_back_to_its_position(self):
        words = [(w if w != "my" else "mine", s, e) for w, s, e in _WORDS]  # spelled differently
        words = [(w if w != "friend" else "pal", s, e) for w, s, e in words]
        cards = [Card(0, "Hello there, how are you doing today,"), Card(38, "my friend?")]
        times = card_times_from_words(_TEXT, cards, words)
        assert times[1] == pytest.approx(card_times_proportional(len(_TEXT), cards, words)[1])

    def test_times_never_run_backwards(self):
        # A word found far ahead of where it belongs would otherwise drag a later card before an earlier one.
        words = [("friend", 0.3, 0.4)] + _WORDS
        cards = [Card(0, "Hello there,"), Card(13, "how are you doing today,"), Card(38, "my friend?")]
        times = card_times_from_words(_TEXT, cards, words)
        assert times == sorted(times)


class TestCardTimesProportional:
    def test_by_position_within_the_speech(self):
        words = [("a", 1.0, 2.0), ("b", 2.0, 5.0)]  # speech from 1.0 to 5.0
        cards = [Card(0, "x" * 50), Card(50, "y" * 50)]
        assert card_times_proportional(100, cards, words) == [0.0, 3.0]

    def test_no_timings_puts_every_card_at_zero(self):
        cards = [Card(0, "a"), Card(2, "b")]
        assert card_times_proportional(3, cards, [("a", None, None)]) == [0.0, 0.0]
