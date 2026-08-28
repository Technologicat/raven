"""Tests for `raven.common.text.boilerplate` — removing a publisher's rights notice from an abstract."""

from ..boilerplate import TAIL_BUDGET, find_rights_notice, strip_boilerplate

# A plausible abstract tail, long enough that a notice appended to it still leaves the abstract's own
# closing sentence outside a naive fixed-length window.
BODY = ("This study examines how generative AI reshapes assessment practice in higher education, and "
        "reports findings from a semester-long deployment across four undergraduate courses.")


class TestRealNotices:
    """The spellings that actually turn up in database exports."""

    def test_the_copyright_sign_opens_a_notice(self):
        text = f"{BODY} © The Author(s), under exclusive license to Springer Nature Singapore Pte Ltd. 2025."
        assert strip_boilerplate(text) == BODY

    def test_all_rights_reserved_opens_a_notice(self):
        assert strip_boilerplate(f"{BODY} All rights reserved.") == BODY

    def test_a_licence_grant_opens_a_notice(self):
        text = f"{BODY} This article is distributed under the terms of the Creative Commons Attribution 4.0 License."
        assert strip_boilerplate(text) == BODY

    def test_a_named_licensee_opens_a_notice(self):
        assert strip_boilerplate(f"{BODY} Licensee MDPI, Basel, Switzerland.") == BODY

    def test_copyright_qualified_by_a_year_opens_a_notice(self):
        assert strip_boilerplate(f"{BODY} Copyright 2024 by the authors.") == BODY

    def test_a_leading_label_is_removed(self):
        assert strip_boilerplate(f"Abstract: {BODY}") == BODY


class TestWhatMustSurvive:
    """The false positives. Both of these shipped in a first version and were caught by measurement."""

    def test_an_abstract_discussing_copyright_keeps_its_ending(self):
        # The case that motivated dropping a bare `copyright` from the pattern: an abstract about
        # AI-generated work whose closing sentence names copyright as a concern, not as a notice.
        text = (f"{BODY} Remaining challenges include copyright concerns, bias mitigation, computational "
                "demands, and the need for robust regulatory frameworks.")
        assert strip_boilerplate(text) == text

    def test_an_abstract_with_no_notice_keeps_its_full_stop(self):
        # A first version trimmed trailing punctuation unconditionally, silently shortening every
        # abstract in the corpus by one character. Trimming happens only after an actual cut.
        assert strip_boilerplate(BODY) == BODY
        assert strip_boilerplate(BODY).endswith(".")

    def test_a_notice_quoted_early_in_a_long_abstract_is_left_alone(self):
        # Position is half the test. Deep inside the text this is the abstract discussing its subject.
        text = "Our corpus study of the phrase 'All rights reserved' in scholarly abstracts. " + BODY * 4
        assert strip_boilerplate(text) == text.strip()

    def test_the_word_copyright_alone_is_not_a_notice(self):
        text = f"{BODY} We argue that copyright law lags behind generative models."
        assert strip_boilerplate(text) == text


class TestFindRightsNotice:
    def test_it_reports_where_the_notice_begins_so_a_caller_can_show_it(self):
        text = f"{BODY} © 2025 Elsevier Ltd."
        offset = find_rights_notice(text)
        assert text[offset:] == "© 2025 Elsevier Ltd."

    def test_it_reports_nothing_for_text_carrying_no_notice(self):
        assert find_rights_notice(BODY) is None

    def test_a_notice_beyond_the_tail_budget_is_not_one(self):
        # The negative control for the window: the same notice text, once inside the budget and once
        # pushed out of it by padding. Without this, a fixture that never exceeds TAIL_BUDGET would
        # pass whether the window were checked or ignored.
        notice = "© 2025 Elsevier Ltd."
        assert find_rights_notice(f"{BODY} {notice}") is not None
        pushed_out = f"{notice} " + "x" * (TAIL_BUDGET + 1)
        assert find_rights_notice(pushed_out) is None


class TestShapes:
    def test_empty_and_whitespace_survive_without_raising(self):
        assert strip_boilerplate("") == ""
        assert strip_boilerplate("   \n  ") == ""

    def test_an_abstract_that_is_only_a_notice_reduces_to_nothing(self):
        # Degenerate but real: a record whose "abstract" is the copyright line and nothing else. Better
        # an empty string, which a caller can test, than a stray fragment that reads like content.
        assert strip_boilerplate("© 2025 Elsevier Ltd. All rights reserved.") == ""

    def test_stripping_is_idempotent(self):
        text = f"Abstract: {BODY} © 2025 Elsevier Ltd."
        once = strip_boilerplate(text)
        assert strip_boilerplate(once) == once
