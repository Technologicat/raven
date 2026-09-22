"""Whether `scripts/gen_postprocessor_manual.py` renders a filter the way the manual needs it.

**Every case here feeds the generator a filter shaped on purpose**, following the pattern set by
`test_check_doc_links.py`: a generator that has only ever seen the 21 real filters cannot show what it
would do with a malformed one, and the anomaly reports are the part that matters — they are what turns
a docstring somebody forgot into a line on the console rather than into a silently thinner manual.

The contents-listing case is a regression. The first draft built its anchors by replacing underscores
with hyphens, which is what GitHub does to *spaces*; every one of the 21 links landed nowhere, and
running the generator said nothing, because a generator cannot read its own output.

The generator imports `raven.common.video.postprocessor` inside `main` rather than at module scope, so
everything here runs without torch and therefore runs in CI.
"""

import pytest

from scripts.gen_postprocessor_manual import _one_line, _render_range, _split_tag, build


def make_filter(name="demo", docstring=None, defaults=None, ranges=None):
    """One entry shaped like `Postprocessor.get_filters()`'s, correct unless a case breaks it."""
    if docstring is None:
        docstring = ("[static] A demonstration filter.\n\n"
                     "`alpha`: How much of it to apply.\n")
    if defaults is None:
        defaults = {"alpha": 0.5}
    if ranges is None:
        ranges = {"alpha": [0.0, 1.0]}
    return (name, {"defaults": defaults, "ranges": ranges, "docstring": docstring})


class TestRanges:
    def test_a_numeric_range_is_rendered_as_bounds(self):
        assert _render_range([0.0, 1.0]) == "range `0.0` to `1.0`"

    def test_a_list_of_strings_is_rendered_as_a_choice(self):
        assert _render_range(["low", "high", "ultra"]) == "one of `low`, `high`, `ultra`"

    def test_a_known_gui_hint_is_rendered_in_words(self):
        assert _render_range(["!RGB"]) == "an RGB colour"

    def test_an_unknown_gui_hint_says_nothing_rather_than_guessing(self):
        # A hint added to the postprocessor later must not be printed raw into the manual as though it
        # were a range. Saying nothing is recoverable; `range !gamma` in shipped docs is not.
        assert _render_range(["!gamma"]) is None

    def test_no_range_says_nothing(self):
        assert _render_range([]) is None
        assert _render_range(None) is None


class TestSummaryTag:
    def test_the_marker_is_split_off_the_summary(self):
        kind, body = _split_tag("[dynamic] Tape noise.")
        assert (kind, body) == ("dynamic", "Tape noise.")

    def test_a_summary_without_a_marker_survives_whole(self):
        kind, body = _split_tag("Tape noise.")
        assert kind is None
        assert body == "Tape noise."

    def test_the_rest_of_a_multi_paragraph_summary_is_kept(self):
        kind, body = _split_tag("[static] Bloom.\n\nBleeds bright areas.")
        assert kind == "static"
        assert "Bleeds bright areas." in body


def test_a_docstrings_own_wrapping_is_collapsed():
    # The fleet's Markdown is unwrapped, and re-wrapping here would move the line breaks every time a
    # sentence changed length, making a regenerated file's diff unreadable.
    assert _one_line("one two\n   three\n\nfour") == "one two three four"


class TestGeneratedDocument:
    def test_every_contents_link_resolves_to_a_heading_it_generated(self):
        """The regression: anchors were built by a rule that does not match GitHub's."""
        from scripts.check_doc_links import dangling_links

        anomalies = []
        text = build([make_filter("analog_vhs_noise"), make_filter("crt")], anomalies)
        assert dangling_links(text.splitlines()) == []

    def test_the_contents_lists_every_filter(self):
        text = build([make_filter("crt"), make_filter("bloom")], anomalies=[])
        assert "- [`crt`](#crt)" in text
        assert "- [`bloom`](#bloom)" in text

    def test_a_parameter_is_rendered_with_its_default_its_range_and_its_help(self):
        text = build([make_filter()], anomalies=[])
        assert "- **`alpha`** — default `0.5`, range `0.0` to `1.0`" in text
        assert "How much of it to apply." in text

    def test_the_static_marker_reaches_the_page(self):
        text = build([make_filter()], anomalies=[])
        assert "*Static.*" in text

    def test_the_universal_name_parameter_is_left_to_the_preamble(self):
        """`name` is the same cache key in every filter that has one, so it is said once, up front."""
        f = make_filter(defaults={"alpha": 0.5, "name": "demo0"},
                        ranges={"alpha": [0.0, 1.0], "name": ["!ignore"]})
        text = build([f], anomalies=[])
        assert "- **`name`**" not in text
        assert "**Every filter takes a `name`**" in text


class TestAnomalies:
    def test_a_well_formed_filter_reports_nothing(self):
        # The negative control. Without it, every case below could be passing because the generator
        # complains about everything, and the tests would look exactly the same.
        anomalies = []
        build([make_filter()], anomalies)
        assert anomalies == []

    def test_a_parameter_with_no_help_is_reported(self):
        anomalies = []
        build([make_filter(docstring="[static] A filter.\n")], anomalies)
        assert any("alpha" in a and "no help text" in a for a in anomalies)

    def test_a_summary_with_no_marker_is_reported(self):
        anomalies = []
        build([make_filter(docstring="A filter.\n\n`alpha`: How much.\n")], anomalies)
        assert any("[static]/[dynamic]" in a for a in anomalies)

    def test_a_hidden_parameter_that_is_not_name_is_reported_and_still_documented(self):
        """`!ignore` means "hide from the GUI", which is not by itself a reason to hide it from a manual."""
        f = make_filter(defaults={"alpha": 0.5, "seed": 7},
                        ranges={"alpha": [0.0, 1.0], "seed": ["!ignore"]},
                        docstring="[static] A filter.\n\n`alpha`: How much.\n\n`seed`: The seed.\n")
        anomalies = []
        text = build([f], anomalies)
        assert any("seed" in a for a in anomalies)
        assert "- **`seed`**" in text

    def test_a_filter_with_no_summary_at_all_is_reported(self):
        anomalies = []
        build([make_filter(docstring="`alpha`: How much.\n")], anomalies)
        assert any("no summary" in a for a in anomalies)


if __name__ == "__main__":
    pytest.main([__file__])
