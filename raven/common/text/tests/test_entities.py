"""Unit tests for raven.common.text.entities."""

import html.entities

from raven.common.text import entities


class TestResolve:
    def test_a_named_entity_is_its_character(self):
        assert entities.resolve("eacute") == "é"

    def test_a_numeric_entity_is_its_character_in_either_base(self):
        assert entities.resolve("#8217") == "’"
        assert entities.resolve("#x2019") == "’"
        assert entities.resolve("#X2019") == "’"

    def test_a_name_that_means_nothing_resolves_to_none(self):
        assert entities.resolve("foo") is None

    def test_a_code_outside_unicode_resolves_to_none(self):
        assert entities.resolve("#0") is None
        assert entities.resolve("#x110000") is None

    def test_a_format_character_is_dropped(self):
        assert entities.resolve("zwj") == ""

    def test_a_control_or_line_separator_becomes_a_space(self):
        # A newline arriving mid-record would move every line after it.
        assert entities.resolve("#10") == " "
        assert entities.resolve("#x2028") == " "

    def test_a_no_break_space_is_kept_unless_folded(self):
        assert entities.resolve("nbsp") == " "  # no-break space
        assert entities.resolve("nbsp", fold_spaces=True) == " "

    def test_every_html5_name_resolves_without_raising(self):
        # Ninety-odd names stand for two code points, and `unicodedata.category` takes only one; this is
        # the test that would have caught the decoder raising `TypeError` on `&NotEqualTilde;`.
        names = [name[:-1] for name in html.entities.html5 if name.endswith(";")]
        assert any(len(html.entities.html5[name + ";"]) > 1 for name in names), \
            "no two-code-point names in the table, so this cannot exercise the case it is here for"
        for name in names:
            assert entities.resolve(name) is not None, name

    def test_a_two_code_point_entity_applies_the_rule_to_each(self):
        assert entities.resolve("NotEqualTilde") == html.entities.html5["NotEqualTilde;"]
        assert entities.resolve("ThickSpace", fold_spaces=True) == "  "


class TestDecode:
    def test_every_entity_in_the_text_is_decoded(self):
        assert entities.decode("caf&eacute; &amp; Smith&#8217;s") == "café & Smith’s"

    def test_one_pass_so_an_escaped_entity_stays_literal(self):
        assert entities.decode("&amp;lt;") == "&lt;"

    def test_folds_spaces_by_default(self):
        assert entities.decode("a&nbsp;b") == "a b"

    def test_a_spared_character_is_left_for_later(self):
        assert entities.decode("&lt;b&gt; &amp; &#38; &eacute;", spare="<>&") == "&lt;b&gt; &amp; &#38; é"

    def test_a_stray_or_unterminated_entity_is_left_alone(self):
        assert entities.decode("&foo; &copy x") == "&foo; &copy x"
