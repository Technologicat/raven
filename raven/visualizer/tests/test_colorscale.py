"""Unit tests for raven.visualizer.colorscale: where a cluster and a year take their colours."""

import numpy as np

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
