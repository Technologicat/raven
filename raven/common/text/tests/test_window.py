"""Unit tests for raven.common.text.window."""

from raven.common import text


def _feed(window, pieces):
    """Add `pieces` to `window` in order, returning what each `add` returned."""
    return [window.add(piece) for piece in pieces]


class TestEmotionWindow:
    def test_the_first_line_is_due_at_once(self):
        window = text.EmotionWindow(interval=5, size=20)
        assert window.add("a") == "a"

    def test_an_update_is_due_every_interval_lines(self):
        window = text.EmotionWindow(interval=3, size=6)
        results = _feed(window, [str(k) for k in range(10)])
        due = [k for k, result in enumerate(results) if result is not None]
        assert due == [0, 3, 6, 9]

    def test_an_interval_of_one_makes_every_line_due(self):
        window = text.EmotionWindow(interval=1, size=4, separator=" ")
        results = _feed(window, [str(k) for k in range(6)])
        assert results == ["0", "0 1", "0 1 2", "0 1 2 3", "1 2 3 4", "2 3 4 5"]

    def test_an_update_reads_only_the_last_size_lines(self):
        window = text.EmotionWindow(interval=4, size=3, separator=" ")
        results = _feed(window, [str(k) for k in range(9)])
        assert results[4] == "2 3 4"
        assert results[8] == "6 7 8"

    def test_lines_are_joined_with_the_separator(self):
        window = text.EmotionWindow(interval=2, size=3, separator=" | ")
        results = _feed(window, ["one", "two", "three"])
        assert results[2] == "one | two | three"

    def test_blank_lines_are_not_counted(self):
        # The same two paragraphs, streamed the two ways a model can space them: blank line as a piece of
        # its own, and blank line folded into the paragraph before it. Both must leave the same window.
        spaced = text.EmotionWindow(interval=2, size=4)
        folded = text.EmotionWindow(interval=2, size=4)
        spaced_results = _feed(spaced, ["first\n", "\n", "second\n", "\n", "third\n"])
        folded_results = _feed(folded, ["first\n\n", "second\n\n", "third\n"])
        assert [r for r in spaced_results if r is not None] == [r for r in folded_results if r is not None]
        assert spaced_results[-1] == "first\nsecond\nthird"

    def test_a_piece_holding_several_lines_counts_each(self):
        window = text.EmotionWindow(interval=2, size=10, separator=" ")
        assert window.add("a") == "a"  # line 0, due
        assert window.add("b\nc") == "a b c"  # lines 1 and 2; line 2 is due
        assert window.add("d") is None  # line 3

    def test_the_defaults_overlap_consecutive_updates_by_three_quarters(self):
        window = text.EmotionWindow(separator=" ")
        results = _feed(window, [f"p{k}" for k in range(60)])
        updates = [result.split() for result in results if result is not None]
        # Only once the window is full does the overlap mean anything; before that, an update simply reads
        # everything so far.
        full = [update for update in updates if len(update) == window.DEFAULT_SIZE]
        assert len(full) >= 2, "the window never filled, so this fixture cannot measure the overlap"
        for earlier, later in zip(full, full[1:]):
            shared = len(set(earlier) & set(later))
            assert shared == 3 * window.DEFAULT_SIZE // 4, (earlier, later)
