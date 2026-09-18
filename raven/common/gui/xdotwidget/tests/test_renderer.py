"""What colour a pen's text actually lands as — the one renderer decision that can silently discard a caller's intent.

Dark mode picks a contrast grey for text on a coloured fill, because the lightness inversion that serves
everything else leaves hue and saturation alone: a saturated yellow stays bright however its lightness
moves, so light-grey-on-yellow measures about 1.3:1. That rule is right for a graph whose colours were
chosen for paper, and wrong for text whose colour *is* the message.

`Pen.keep_color` asks for the other trade: keep the hue, move only the lightness, and only as far as
legibility needs. Not an exemption — red on this widget's own orange node fill is 1.2:1, so a pen simply
let through would be as unreadable as the grey was uninformative.

No DPG: `renderer.text_color` is pure, which is why it was split out of the draw call.
"""

import colorsys

import pytest

from ..graph import Pen
from .. import renderer


RED = (1.0, 0.0, 0.0, 1.0)

# Authored for light, as this widget's callers write them — the renderer inverts on the way to the screen.
# The orange is the chat graph's own TOOL fill, which is the worst case in the constellation: bright enough
# after inversion that red has to go *darker* rather than lighter to be read on it.
FILLS = {"pale": (0.93, 0.94, 0.94, 1.0),
         "tool orange": (0.40, 0.27, 0.10, 1.0),
         "system green": (0.16, 0.40, 0.17, 1.0),
         "near white": (1.0, 1.0, 1.0, 1.0)}


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


def as_fraction(dpg_color):
    return tuple(c / 255 for c in dpg_color[:3])


class TestTextColor:
    def test_ordinary_text_on_a_fill_is_greyed_for_contrast(self, dark_mode):
        drawn = renderer.text_color(pen_for(RED), FILLS["pale"])
        assert drawn[0] == drawn[1] == drawn[2], f"expected a grey, got {drawn}"

    @pytest.mark.parametrize("fill_name", list(FILLS))
    def test_a_kept_colour_is_legible_on_every_fill(self, dark_mode, fill_name):
        fill = FILLS[fill_name]
        drawn = renderer.text_color(pen_for(RED, keep_color=True), fill)
        ratio = renderer._contrast_ratio(as_fraction(drawn), renderer._invert_lightness(fill))
        assert ratio >= renderer._MIN_TEXT_CONTRAST - 0.01, f"{fill_name}: {ratio:.2f}:1"

    @pytest.mark.parametrize("fill_name", list(FILLS))
    def test_a_kept_colour_is_still_the_colour_it_was(self, dark_mode, fill_name):
        """Lightness is what moves. Hue is what the caller was saying something with, so hue is what stays."""
        drawn = renderer.text_color(pen_for(RED, keep_color=True), FILLS[fill_name])
        red, green, blue = as_fraction(drawn)
        assert red > green and red > blue, f"{fill_name}: a red that came out neutral or otherwise: {drawn}"
        assert colorsys.rgb_to_hls(red, green, blue)[0] == pytest.approx(0.0, abs=0.02), "the hue moved"

    def test_the_two_paths_differ_on_the_fill_being_tested(self, dark_mode):
        """The control: a fill whose contrast grey happened to come out red would make the pair above vacuous."""
        assert (renderer.text_color(pen_for(RED), FILLS["pale"])
                != renderer.text_color(pen_for(RED, keep_color=True), FILLS["pale"]))

    def test_a_colour_that_already_reads_is_left_alone(self, dark_mode):
        """"As little as needed" is half the rule, and without it every kept colour would come out at the bar."""
        fill = FILLS["near white"]  # inverts to this mode's darkest, where plain red already has contrast
        shown_fill = renderer._invert_lightness(fill)
        untouched = renderer._invert_lightness(RED)
        assert renderer._contrast_ratio(untouched, shown_fill) >= renderer._MIN_TEXT_CONTRAST, \
            "this fixture needs a fill red already reads on, or it cannot tell 'left alone' from 'adjusted'"
        assert as_fraction(renderer.text_color(pen_for(RED, keep_color=True), fill)) == \
            pytest.approx(untouched[:3], abs=1 / 255)

    def test_without_a_fill_the_pen_decides_either_way(self, dark_mode):
        """An edge label, or a graph's background text: nothing to contrast against, so nothing to adjust."""
        assert (renderer.text_color(pen_for(RED), None)
                == renderer.text_color(pen_for(RED, keep_color=True), None))

    def test_opacity_reaches_the_alpha_on_every_path(self, dark_mode):
        for keep in (False, True):
            drawn = renderer.text_color(pen_for(RED, keep_color=keep), FILLS["pale"], opacity=0.5)
            assert drawn[3] == pytest.approx(127, abs=1), f"keep_color={keep} lost the opacity: {drawn}"


class TestLegibleAgainst:
    """The adjustment itself, where the awkward cases live."""

    def test_it_gives_up_gracefully_where_the_bar_cannot_be_reached(self):
        """A fill bright at every lightness the mode allows. A thin mark beats returning nothing."""
        on_yellow = renderer._legible_against((1.0, 1.0, 0.0, 1.0), (1.0, 1.0, 0.4, 1.0))
        assert on_yellow is not None and len(on_yellow) == 4

    @pytest.mark.parametrize("fill, expected", [((0.35, 0.35, 0.38, 1.0), "lighter"),
                                                ((0.85, 0.85, 0.50, 1.0), "darker")])
    def test_it_moves_toward_whichever_end_helps(self, fill, expected):
        """Not always lighter: a fill brighter than the text wants darker text, which one direction would miss."""
        def lightness(color):
            return colorsys.rgb_to_hls(*color[:3])[1]
        assert renderer._contrast_ratio(RED, fill) < renderer._MIN_TEXT_CONTRAST, \
            "red already reads on this fill, so nothing would move and the direction is untested"
        moved = lightness(renderer._legible_against(RED, fill))
        assert (moved > lightness(RED)) if expected == "lighter" else (moved < lightness(RED))
