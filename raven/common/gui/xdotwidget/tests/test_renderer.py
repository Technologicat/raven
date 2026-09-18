"""What colour a pen's text actually lands as — the one renderer decision that can silently discard a caller's intent.

Dark mode picks a contrast grey for text on a coloured fill, because the lightness inversion that serves
everything else can put near-white text on a mid-lightness fill. That is right for a graph whose colours
were chosen for paper, and wrong for text whose colour *is* the message: a search match drawn in a readable
grey is not a compromise, it is the wrong answer. `Pen.keep_color` is the distinction.

No DPG: `renderer.text_color` is pure, which is why it was split out of the draw call.
"""

import pytest

from ..graph import Pen
from .. import renderer


RED = (1.0, 0.0, 0.0, 1.0)
PALE_FILL = (0.93, 0.94, 0.94, 1.0)  # authored light, as a chat graph box's fill is


@pytest.fixture
def dark_mode():
    """Dark mode on for the test, and back to whatever it was after — the flag is module-global."""
    was = renderer.get_dark_mode()
    renderer.set_dark_mode(True)
    yield
    renderer.set_dark_mode(was)


def pen_for(color, keep_color=False):
    pen = Pen()
    pen.color = color
    pen.keep_color = keep_color
    return pen


class TestTextColor:
    def test_text_on_a_fill_is_greyed_for_contrast(self, dark_mode):
        drawn = renderer.text_color(pen_for(RED), PALE_FILL)
        assert drawn[0] == drawn[1] == drawn[2], f"expected a grey, got {drawn}"

    def test_keep_color_is_drawn_in_the_pen_s_own_colour(self, dark_mode):
        drawn = renderer.text_color(pen_for(RED, keep_color=True), PALE_FILL)
        assert drawn[:3] == renderer.color_to_dpg(RED)[:3]
        assert drawn[0] > drawn[1] and drawn[0] > drawn[2], f"a red that came out neutral: {drawn}"

    def test_the_two_differ_on_this_fill(self, dark_mode):
        """The control: a fill the contrast rule happened to grey *to* red would make the pair above vacuous."""
        assert (renderer.text_color(pen_for(RED), PALE_FILL)
                != renderer.text_color(pen_for(RED, keep_color=True), PALE_FILL))

    def test_without_a_fill_the_pen_decides_either_way(self, dark_mode):
        """An edge label, or a graph's background text: nothing to contrast against, so nothing to override."""
        assert (renderer.text_color(pen_for(RED), None)
                == renderer.text_color(pen_for(RED, keep_color=True), None))

    def test_opacity_reaches_the_alpha_on_both_paths(self, dark_mode):
        for keep in (False, True):
            drawn = renderer.text_color(pen_for(RED, keep_color=keep), PALE_FILL, opacity=0.5)
            assert drawn[3] == pytest.approx(127, abs=1), f"keep_color={keep} lost the opacity: {drawn}"
