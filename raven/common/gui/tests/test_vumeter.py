"""Tests for `raven.common.gui.vumeter`.

The meter is a drawlist, so what it shows is not a widget value anyone can read back — it is a
set of draw items. These tests read them: `mvDrawLine` items report their endpoints, which is
enough to say where the threshold and peak lines ended up, and whether they were drawn at all.

The positions asserted here are chosen so that the arithmetic is exact and independent of the
implementation's formula: over a -90..0 dBFS meter, -45 is the midpoint.

No layout is needed, so these run headless, in a DPG context whose viewport is never mapped.
"""

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed (GUI toolkit absent in CI)")

from raven.common.gui.vumeter import DPGVUMeter  # noqa: E402 -- after importorskip by design

HEIGHT = 100
BORDER = 1
# The drawable span is the height less the border at each end, so a value's distance from the
# bottom is that span times its position in the range, and `y` counts down from the top.
MIDPOINT_Y = HEIGHT - int((HEIGHT - 2 * BORDER) * 0.5)  # 51, for -45 dBFS on a -90..0 meter
TOP_Y = HEIGHT - (HEIGHT - 2 * BORDER)                  # 2, for 0 dBFS


@pytest.fixture(scope="module")
def dpg_context():
    """A DPG context with an unmapped viewport, torn down after the module."""
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: these tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


@pytest.fixture
def make_meter(dpg_context):
    """Build a meter of known geometry inside a fresh window."""
    def build(threshold_value=None):
        with dpg.window() as window:
            return DPGVUMeter(width=20,
                              height=HEIGHT,
                              border=BORDER,
                              min_value=-90.0,
                              max_value=0.0,
                              yellow_start=-24.0,
                              red_start=-6.0,
                              threshold_value=threshold_value,
                              parent=window)
    return build


def line_ys(meter):
    """The y coordinates of every horizontal line the meter has drawn."""
    return [dpg.get_item_configuration(item)["p1"][1]
            for item in dpg.get_item_children(meter.drawlist, slot=2)
            if dpg.get_item_type(item) == "mvAppItemType::mvDrawLine"]


class TestTheThresholdLine:
    def test_it_is_drawn_where_the_value_maps(self, make_meter):
        meter = make_meter(threshold_value=-45.0)
        assert MIDPOINT_Y in line_ys(meter)

    def test_no_threshold_draws_no_line_there(self, make_meter):
        # The control for the test above: without a threshold nothing is drawn at that height, so
        # the assertion there is about the threshold rather than about some other line the meter
        # happens to draw across the middle.
        meter = make_meter(threshold_value=None)
        assert MIDPOINT_Y not in line_ys(meter)

    def test_moving_it_moves_the_line(self, make_meter):
        meter = make_meter(threshold_value=-45.0)
        assert MIDPOINT_Y in line_ys(meter)
        meter.threshold = 0.0
        ys = line_ys(meter)
        assert TOP_Y in ys, f"the line did not move to the top of the meter: {ys}"
        assert MIDPOINT_Y not in ys, f"the line is still drawn where it used to be: {ys}"

    def test_it_can_be_taken_away_and_put_back(self, make_meter):
        meter = make_meter(threshold_value=-45.0)
        meter.threshold = None
        assert MIDPOINT_Y not in line_ys(meter)
        meter.threshold = -45.0
        assert MIDPOINT_Y in line_ys(meter)

    def test_the_property_reads_back_what_was_set(self, make_meter):
        meter = make_meter(threshold_value=-45.0)
        assert meter.threshold == -45.0
        meter.threshold = -30.0
        assert meter.threshold == -30.0

    @pytest.mark.parametrize("bad", [-95.0, 5.0])
    def test_a_value_outside_the_meter_is_refused(self, make_meter, bad):
        meter = make_meter(threshold_value=-45.0)
        with pytest.raises(ValueError):
            meter.threshold = bad
        assert meter.threshold == -45.0, "the refused value was applied anyway"


class TestThePeakLine:
    def test_it_is_drawn_where_the_value_maps(self, make_meter):
        meter = make_meter(threshold_value=None)
        meter.update(instant=-90.0, peak=-45.0)
        assert line_ys(meter) == [MIDPOINT_Y]

    def test_no_peak_draws_no_line(self, make_meter):
        meter = make_meter(threshold_value=None)
        meter.update(instant=-90.0, peak=None)
        assert line_ys(meter) == []
