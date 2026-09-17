"""`predict_height_change`: a re-rendered paragraph's height, known before it is laid out.

Pure arithmetic over `MarkdownText.rows`; the measurement it rests on is `investigations/chat-search-highlight/`.
"""

import pytest

pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from raven.vendor.DearPyGui_Markdown import predict_height_change  # noqa: E402 -- after importorskip by design

LINE = 26  # a row of body text, as `rows` reports it
EXTRA = 6  # what layout adds to each row of text; measured, and deliberately not what the function assumes


def laid_out(rows):
    return sum(height + (0 if is_rule else EXTRA) for height, is_rule in rows)


class TestPredictHeightChange:
    def test_the_same_row_count_is_no_change(self):
        rows = [(LINE, False)] * 3
        assert predict_height_change(rows, laid_out(rows), [(LINE, False)] * 3) == 0

    def test_an_extra_row_adds_the_row_and_the_extra_read_off_the_old_render(self):
        old, new = [(LINE, False)] * 2, [(LINE, False)] * 3
        assert predict_height_change(old, laid_out(old), new) == laid_out(new) - laid_out(old) == LINE + EXTRA

    def test_the_extra_is_calibrated_rather_than_assumed(self):
        """With a different extra, the prediction follows it: nothing in the function knows 6 px."""
        old, new = [(LINE, False)] * 2, [(LINE, False)] * 3
        assert predict_height_change(old, 2 * (LINE + 10), new) == LINE + 10

    def test_a_rule_row_takes_no_extra(self):
        old = [(LINE, False), (4, True)]
        new = [(LINE, False), (LINE, False), (4, True)]
        assert predict_height_change(old, laid_out(old), new) == laid_out(new) - laid_out(old)

    @pytest.mark.parametrize("old_rows, old_height, new_rows", [
        (None, 100, [(LINE, False)]),        # the old render's rows are unknown
        ([(LINE, False)], 100, None),        # the new render's are
        ([(LINE, False)], 0, [(LINE, False)] * 2),  # the old render was never laid out
        ([(4, True)], 4, [(LINE, False)]),   # nothing in the old render to calibrate against
    ])
    def test_nothing_to_calibrate_from_is_no_change(self, old_rows, old_height, new_rows):
        assert predict_height_change(old_rows, old_height, new_rows) == 0
