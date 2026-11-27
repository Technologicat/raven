"""Unit tests for raven.visualizer.colorscale: where a cluster and a year take their colours."""

import numpy as np
import pytest

from raven.visualizer import colorscale


class TestTheClusterColourPosition:
    def test_outliers_sit_at_zero_and_the_clusters_spread_up_to_one(self):
        assert [colorscale.colormap_position(label, 3) for label in (-1, 0, 1, 2, 3)] == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_a_dataset_of_outliers_alone_puts_everything_at_zero(self):
        # The case that crashed the process: every label -1, so the position was 0 / 0, NaN in numpy integers.
        assert colorscale.colormap_position(np.int64(-1), np.int64(-1)) == 0.0

    def test_numpy_labels_give_a_finite_float(self):
        position = colorscale.colormap_position(np.int64(2), np.int64(4))
        assert isinstance(position, float) and position == 0.6


class TestTheYearColourPosition:
    def test_the_years_spread_above_misc_as_the_clusters_do(self):
        assert [colorscale.year_color_position(year, (2020, 2023)) for year in (2020, 2021, 2023)] == [0.25, 0.5, 1.0]


class TestFigureColours:
    @pytest.mark.parametrize("background, foreground", [("black", "white"), ("#202020", "white"),
                                                         ("white", "black"), ("#e8e0d0", "black")])
    def test_the_foreground_is_whichever_of_black_and_white_stands_out(self, background, foreground):
        assert colorscale.foreground_for(background) == foreground

    def test_colours_take_the_range_for_their_background(self, monkeypatch):
        monkeypatch.setattr(colorscale.gui_config, "colormap_ranges", {"dark": (0.2, 1.0), "light": (0.0, 0.85)})
        assert colorscale.colormap_range_for("black") == (0.2, 1.0)
        assert colorscale.colormap_range_for("white") == (0.0, 0.85)
        assert colorscale.colormap_range_for((37, 37, 38)) == (0.2, 1.0), "the GUI config writes 0-255 integers"
        assert colorscale.colormap_range_for((255, 255, 255)) == (0.0, 0.85)

    def test_the_plot_takes_the_range_for_its_own_background(self):
        # With the shipped config: the plot is dark, so its colours come from the dark range.
        assert colorscale.colormap_range_for(colorscale.gui_config.plotter_background_color) == \
            tuple(colorscale.gui_config.colormap_ranges["dark"])
