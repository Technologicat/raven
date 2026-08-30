"""Tests for `raven.common.text.boilerplate` — removing a publisher's rights notice from an abstract."""

from ..boilerplate import TAIL_BUDGET, find_rights_notice, split_rights_notice, strip_boilerplate

# A plausible abstract tail, long enough that a notice appended to it still leaves the abstract's own
# closing sentence outside a naive fixed-length window.
BODY = ("This study examines how generative AI reshapes assessment practice in higher education, and "
        "reports findings from a semester-long deployment across four undergraduate courses.")


class TestRealNotices:
    """The spellings that actually turn up in database exports."""

    def test_the_copyright_sign_opens_a_notice(self):
        text = f"{BODY} © The Author(s), under exclusive licence to Vantage Academic Press Ltd. 2025."
        assert strip_boilerplate(text) == BODY

    def test_all_rights_reserved_opens_a_notice(self):
        assert strip_boilerplate(f"{BODY} All rights reserved.") == BODY

    def test_a_licence_grant_opens_a_notice(self):
        text = f"{BODY} This article is distributed under the terms of the Creative Commons Attribution 4.0 License."
        assert strip_boilerplate(text) == BODY

    def test_a_named_licensee_opens_a_notice(self):
        assert strip_boilerplate(f"{BODY} Licensee Vantage Academic, Helsinki, Finland.") == BODY

    def test_copyright_qualified_by_a_year_opens_a_notice(self):
        assert strip_boilerplate(f"{BODY} Copyright 2024 by the authors.") == BODY

    def test_a_publisher_named_after_a_year_opens_a_notice(self):
        assert strip_boilerplate(f"{BODY} Copyright 2024, Society of Petroleum Engineers.") == BODY

    def test_an_open_access_grant_opens_a_notice(self):
        text = (f"{BODY} This is an open access article distributed under the terms of the Creative "
                "Commons Attribution 4.0 License.")
        assert strip_boilerplate(text) == BODY

    def test_a_licence_tag_wedged_before_the_notice_goes_with_it(self):
        # Publishers put a bracketed tag exactly where it breaks a naive sentence test, and cutting at
        # the marker alone leaves it dangling on the end of the abstract.
        text = (f"{BODY} (CC BY-NC 4.0) This article is licensed to you under a Creative Commons "
                "Attribution-NonCommercial 4.0 International License.")
        assert strip_boilerplate(text) == BODY
        assert "CC BY-NC" not in strip_boilerplate(text)

    def test_a_leading_label_is_removed(self):
        assert strip_boilerplate(f"Abstract: {BODY}") == BODY


class TestPapersAboutCopyright:
    """The hard case, and the reason for the two tiers.

    A corpus on AI in education contains papers about copyright and open licensing, so the phrases a
    notice is made of also occur as the *subject* of an abstract. Every ending here is prose and must
    survive whole. Each shipped as a false positive in some version of this module.
    """

    ENDINGS = [
        "Copyright remains a widely debated field of law, and further research into the topic is encouraged.",
        "Creative Commons was found to be a popular licensing model, and adoption is rising.",
        "Creative Commons Attribution licences are increasingly common in open education.",
        "The Copyright 1976 settlement still governs derivative works in this domain.",
        "We revisit the Copyright Act of 1976 in light of generative models.",
        "We show the phrase All rights reserved has no legal effect in these jurisdictions.",
        "We argue this work is licensed under terms too permissive for student data.",
        "Whether copyright held by an institution serves students remains unclear.",
        "The question of copyright by the author versus the publisher is unsettled.",
        "Remaining challenges include copyright concerns, bias mitigation and computational demands.",
    ]

    def test_prose_about_rights_is_not_a_rights_notice(self):
        for ending in self.ENDINGS:
            text = f"{BODY} {ending}"
            assert strip_boilerplate(text) == text, f"ate an abstract's own words: {ending!r}"


class TestWhatMustSurvive:
    def test_an_abstract_with_no_notice_keeps_its_full_stop(self):
        # A first version trimmed trailing punctuation unconditionally, silently shortening every
        # abstract in the corpus by one character. Trimming happens only after an actual cut.
        assert strip_boilerplate(BODY) == BODY
        assert strip_boilerplate(BODY).endswith(".")

    def test_a_notice_quoted_early_in_a_long_abstract_is_left_alone(self):
        # Position is the condition both tiers share. Deep inside the text this is an abstract
        # discussing its subject, whatever it says.
        text = "Our corpus study of the phrase 'All rights reserved' in scholarly abstracts. " + BODY * 6
        assert strip_boilerplate(text) == text.strip()


class TestFindRightsNotice:
    def test_it_reports_where_the_notice_begins_so_a_caller_can_show_it(self):
        text = f"{BODY} © 2025 Vantage Academic Press Ltd."
        offset = find_rights_notice(text)
        assert text[offset:] == "© 2025 Vantage Academic Press Ltd."

    def test_it_reports_nothing_for_text_carrying_no_notice(self):
        assert find_rights_notice(BODY) is None

    def test_a_notice_beyond_the_tail_budget_is_not_one(self):
        # The negative control for the window: the same notice text, once inside the budget and once
        # pushed out of it by padding. Without this, a fixture that never exceeds TAIL_BUDGET would
        # pass whether the window were checked or ignored.
        notice = "© 2025 Vantage Academic Press Ltd."
        assert find_rights_notice(f"{BODY} {notice}") is not None
        pushed_out = f"{notice} " + "x" * (TAIL_BUDGET + 1)
        assert find_rights_notice(pushed_out) is None


class TestSplitRightsNotice:
    """Both halves, for a caller writing a bibliography back out rather than analyzing text."""

    def test_it_hands_back_the_body_and_the_notice(self):
        body, notice = split_rights_notice(f"{BODY} © 2025 Vantage Academic Press Ltd.")
        assert body == BODY
        assert notice == "© 2025 Vantage Academic Press Ltd."

    def test_nothing_is_lost_between_the_two_halves(self):
        """The property that makes this safe to use for a move rather than a delete."""
        text = f"{BODY} © 2025 Vantage Academic Press Ltd."
        body, notice = split_rights_notice(text)
        assert set(text.split()) == set(body.split()) | set(notice.split())

    def test_text_with_no_notice_comes_back_whole_and_unaccompanied(self):
        assert split_rights_notice(BODY) == (BODY, None)

    def test_a_leading_label_is_dropped_rather_than_returned(self):
        # The one thing here that really is noise: no exporter means `Abstract:` as content.
        body, notice = split_rights_notice(f"Abstract: {BODY} © 2025 Vantage Academic Press Ltd.")
        assert body == BODY and notice is not None

    def test_strip_boilerplate_is_the_first_half(self):
        for text in (f"{BODY} © 2025 Vantage Academic Press Ltd.", BODY, f"Abstract: {BODY}", "", "   "):
            assert strip_boilerplate(text) == split_rights_notice(text)[0]


class TestShapes:
    def test_empty_and_whitespace_survive_without_raising(self):
        assert strip_boilerplate("") == ""
        assert strip_boilerplate("   \n  ") == ""
        assert split_rights_notice("") == ("", None)

    def test_an_abstract_that_is_only_a_notice_reduces_to_nothing(self):
        # Degenerate but real: a record whose "abstract" is the copyright line and nothing else. Better
        # an empty string, which a caller can test, than a stray fragment that reads like content.
        assert strip_boilerplate("© 2025 Vantage Academic Press Ltd. All rights reserved.") == ""

    def test_stripping_is_idempotent(self):
        text = f"Abstract: {BODY} © 2025 Vantage Academic Press Ltd."
        once = strip_boilerplate(text)
        assert strip_boilerplate(once) == once
