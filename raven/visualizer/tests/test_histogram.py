"""Unit tests for raven.visualizer.histogram: the figure, built without a GUI, and the cluster captions."""

import numpy as np
import pytest

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from unpythonic.env import env

from raven.visualizer import displaystate, histogram, savedstate


def labels_of(fig):
    """The axis labels written, as `{text: bold}`."""
    ax = fig.axes[0]
    return {t.get_text(): t.get_fontweight() == "bold" for t in ax.texts}


def figure(counts, **kwargs):
    kwargs.setdefault("grouping", displaystate.CLUSTER)
    return histogram.build_figure(counts, ["C0"] * len(counts), first=0, **kwargs)


# Wide-ranging counts, crowded at the low end, as a clustered corpus tends to give.
COUNTS = [1932, 1878, 300, 320, 140, 230, 66, 1050, 650, 20, 8, 400, 7, 90, 25, 14, 9, 30, 12, 5]


class TestTheFigure:
    @pytest.mark.parametrize("scale", histogram.SCALES)
    def test_every_scale_labels_zero_and_the_largest(self, scale):
        labels = labels_of(figure(COUNTS, scale=scale))
        assert "0" in labels and "1932" in labels
        assert labels["1932"] and not labels["0"], "a bar's count is bold, a round number is not"

    def test_the_tallest_bars_get_their_counts(self):
        labels = labels_of(figure(COUNTS, tallest=2))
        assert labels.get("1878") is True

    def test_without_tallest_the_second_tallest_is_unlabelled(self):
        """Control for the test above: 1878 is no round number, so only `tallest` puts it there."""
        assert "1878" not in labels_of(figure(COUNTS, tallest=0))

    def test_unreadable_counts_go_in_a_wide_gap(self):
        # On a linear axis from 0 to 1000 with ticks every 200, 500 and 700 sit midway between ticks.
        counts = [1000, 500, 700, 1]
        assert "500" not in labels_of(figure(counts, scale="linear", unreadable=False))
        assert labels_of(figure(counts, scale="linear", unreadable=True)).get("500") is True

    def test_captions_replace_the_numbers_along_the_bottom(self):
        fig = figure([3, 5], captions=["0 learning analytics", "1 chatbots"])
        assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["0 learning analytics", "1 chatbots"]

    def test_the_horizontal_axis_names_the_grouping(self):
        assert figure([3, 5], grouping=displaystate.YEAR).axes[0].get_xlabel() == "Year"
        assert figure([3, 5]).axes[0].get_xlabel() == "Cluster"

    def test_nothing_to_draw_or_an_unknown_scale_is_an_error(self):
        with pytest.raises(ValueError):
            figure([0, 0])
        with pytest.raises(ValueError):
            figure([1, 2], scale="cubic")

    def test_the_layout_has_settled(self):
        """The labels are placed for the height the layout chose, so laying out once more moves almost nothing."""
        fig = figure(COUNTS, tallest=2, unreadable=True, figsize=(8.0, 3.5), dpi=72)
        before = fig.axes[0].get_position().height * 3.5 * 72
        fig.tight_layout()
        after = fig.axes[0].get_position().height * 3.5 * 72
        assert abs(after - before) < 1.0, f"the axes moved {after - before:.2f} pt on another layout"

    def test_every_label_lands_inside_the_figure(self):
        captions = [f"{k} a long caption that says what cluster {k} is about" for k in range(len(COUNTS))]
        plain = Figure(figsize=(8.0, 3.5), dpi=72)
        FigureCanvasAgg(plain)
        plain_ax = plain.add_subplot(1, 1, 1)
        plain_ax.set_xticks(range(len(COUNTS)), captions, rotation=90)
        plain.canvas.draw()
        assert min(t.get_window_extent().y0 for t in plain_ax.get_xticklabels()) < 0, \
            "these captions fit a figure with no layout, so this fixture cannot tell a fitted layout from none"

        fig = figure(COUNTS, tallest=2, unreadable=True, captions=captions, figsize=(8.0, 3.5), dpi=72)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax = fig.axes[0]
        width, height = fig.bbox.width, fig.bbox.height
        for text in [*ax.texts, *ax.get_xticklabels(), ax.xaxis.label, ax.yaxis.label, ax.title]:
            box = text.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= width + 1 and box.y1 <= height + 1, \
                f"{text.get_text()!r} sticks out of the figure: {box}"

    @pytest.mark.parametrize("background, foreground", [("black", "white"), ("#1e1e1e", "white"),
                                                         ("white", "black"), ("#f0e8d8", "black")])
    def test_the_text_stands_out_against_the_background(self, background, foreground):
        import matplotlib.colors
        fig = figure(COUNTS, tallest=2, background=background)
        ax = fig.axes[0]
        assert matplotlib.colors.to_hex(fig.get_facecolor()) == matplotlib.colors.to_hex(background)
        expected = matplotlib.colors.to_rgba(foreground)
        for text in [*ax.texts, ax.xaxis.label, ax.yaxis.label]:
            assert matplotlib.colors.to_rgba(text.get_color()) == expected, f"{text.get_text()!r} on {background}"
        assert matplotlib.colors.to_rgba(ax.spines["left"].get_edgecolor()) == expected

    def test_it_renders_and_saves(self, tmp_path):
        fig = figure(COUNTS, tallest=2, unreadable=True, figsize=(4.0, 2.0), dpi=50)
        rgba = histogram._render_rgba(fig)
        assert rgba.shape == (100, 200, 4)
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(tmp_path / f"h.{suffix}")
            assert (tmp_path / f"h.{suffix}").stat().st_size > 0


class TestClusterCaptions:
    def test_the_leading_theme_or_else_the_leading_quote(self):
        file_content = env(cluster_llm_keywords=[["chatbots", "tutoring"], []],
                           cluster_keyword_lists=[[("chatbot", 1.0)], [("peer review", 0.8), ("feedback", 0.5)]],
                           keyphrase_display={}, vis_keywords_by_cluster=[["chatbot"], ["peer review"]])
        assert histogram._cluster_caption(file_content, 0) == "chatbots"
        assert histogram._cluster_caption(file_content, 1) == "peer review"

    def test_a_dataset_with_one_list_gives_its_first_keyword(self):
        file_content = env(vis_keywords_by_cluster=[["hydrogen", "electrolysis"]])
        assert histogram._cluster_caption(file_content, 0) == "hydrogen"


def test_every_scale_can_be_saved_between_sessions():
    # `savedstate` cannot import this module, so it lists the scales itself; a scale missing there is reset on load.
    assert savedstate.ALLOWED["histogram_scale"] == histogram.SCALES


def test_the_saved_defaults_are_the_module_defaults():
    assert savedstate.DEFAULTS["histogram_scale"] == histogram.get_scale()
    labels = histogram.get_labels()
    assert (savedstate.DEFAULTS["histogram_tallest"], savedstate.DEFAULTS["histogram_unreadable"],
            savedstate.DEFAULTS["histogram_themes"]) == (labels["tallest"], labels["unreadable"], labels["themes"])


def test_no_count_survives_as_a_float():
    labels = labels_of(figure([1.0, 2.5e3, 40.0]))
    assert all("." not in text for text in labels), labels
    assert np.isfinite([float(t) for t in labels]).all()
