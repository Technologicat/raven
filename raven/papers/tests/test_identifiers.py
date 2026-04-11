"""Tests for arXiv identifier parsing and extraction."""

from raven.papers.identifiers import (
    ARXIV_NEW_ID_RE,
    ARXIV_OLD_ID_RE,
    extract_id,
    split_version,
    strip_version,
    extract_ids_from_filenames,
)


# ---- ARXIV_NEW_ID_RE -------------------------------------------------------

class TestArxivNewIdRegex:
    """Verify the new-style regex (YYMM.NNNN / YYMM.NNNNN)."""

    def test_5digit_bare(self):
        assert ARXIV_NEW_ID_RE.search("2301.12345")

    def test_5digit_with_version(self):
        m = ARXIV_NEW_ID_RE.search("2301.12345v3")
        assert m.group(1) == "2301.12345"
        assert m.group(2) == "v3"

    def test_4digit_bare(self):
        m = ARXIV_NEW_ID_RE.search("0704.0001")
        assert m.group(1) == "0704.0001"

    def test_4digit_with_version(self):
        m = ARXIV_NEW_ID_RE.search("0704.0001v2")
        assert m.group(1) == "0704.0001"
        assert m.group(2) == "v2"

    def test_no_match_3digit_suffix(self):
        assert ARXIV_NEW_ID_RE.search("2301.123") is None

    def test_no_match_6digit_suffix(self):
        """6 digits after the dot should not match at all."""
        m = ARXIV_NEW_ID_RE.search("2301.123456")
        assert m is None

    def test_no_match_random_text(self):
        assert ARXIV_NEW_ID_RE.search("hello world") is None

    # -- Boundary behavior (digit lookaround, not \b) --

    def test_underscore_adjacent(self):
        """Underscores are common in user-named paper files."""
        m = ARXIV_NEW_ID_RE.search("Smith_2301.12345v2_final.pdf")
        assert m.group(1) == "2301.12345"

    def test_letter_adjacent(self):
        m = ARXIV_NEW_ID_RE.search("x2301.12345y")
        assert m.group(1) == "2301.12345"

    def test_hyphen_adjacent(self):
        m = ARXIV_NEW_ID_RE.search("Smith-2301.12345v2-final.pdf")
        assert m.group(1) == "2301.12345"

    def test_digit_prefix_blocks(self):
        """A digit before the ID merges it into a longer number."""
        assert ARXIV_NEW_ID_RE.search("92301.12345") is None

    def test_digit_suffix_blocks(self):
        assert ARXIV_NEW_ID_RE.search("2301.123450") is None


# ---- ARXIV_OLD_ID_RE -------------------------------------------------------

class TestArxivOldIdRegex:
    """Verify the old-style regex (subject-class + separator + 7 digits)."""

    def test_slash_separator(self):
        m = ARXIV_OLD_ID_RE.search("hep-th/0601001")
        assert m.group(1) == "hep-th"
        assert m.group(2) == "0601001"

    def test_hyphen_separator(self):
        m = ARXIV_OLD_ID_RE.search("hep-th-0601001")
        assert m.group(1) == "hep-th"
        assert m.group(2) == "0601001"

    def test_underscore_separator(self):
        m = ARXIV_OLD_ID_RE.search("hep-th_0601001")
        assert m.group(1) == "hep-th"
        assert m.group(2) == "0601001"

    def test_dot_separator(self):
        m = ARXIV_OLD_ID_RE.search("hep-th.0601001")
        assert m.group(1) == "hep-th"
        assert m.group(2) == "0601001"

    def test_version_suffix(self):
        m = ARXIV_OLD_ID_RE.search("hep-ex/0307015v1")
        assert m.group(3) == "v1"

    def test_no_version(self):
        m = ARXIV_OLD_ID_RE.search("hep-ex/0307015")
        assert m.group(3) is None

    # -- Subject class variants --

    def test_single_word_class(self):
        m = ARXIV_OLD_ID_RE.search("math/0601001")
        assert m.group(1) == "math"

    def test_hyphenated_class(self):
        m = ARXIV_OLD_ID_RE.search("astro-ph/0601001")
        assert m.group(1) == "astro-ph"

    def test_short_hyphenated_class(self):
        m = ARXIV_OLD_ID_RE.search("gr-qc/0601001")
        assert m.group(1) == "gr-qc"

    def test_single_char_base(self):
        """q-bio, q-fin have a single-char base."""
        m = ARXIV_OLD_ID_RE.search("q-bio/0601001")
        assert m.group(1) == "q-bio"

    def test_dot_subcategory(self):
        m = ARXIV_OLD_ID_RE.search("math.GT/0309136")
        assert m.group(1) == "math.GT"

    def test_cs_dot_subcategory(self):
        m = ARXIV_OLD_ID_RE.search("cs.AI/0601001")
        assert m.group(1) == "cs.AI"

    # -- Boundary conditions --

    def test_preceded_by_letter_blocks(self):
        """Subject class must not be glued to a preceding word."""
        assert ARXIV_OLD_ID_RE.search("aboutmath/0601001") is None

    def test_preceded_by_digit_ok(self):
        m = ARXIV_OLD_ID_RE.search("2006hep-th_0601001.pdf")
        assert m.group(1) == "hep-th"

    def test_preceded_by_space_ok(self):
        m = ARXIV_OLD_ID_RE.search("Smith 2006 hep-th_0601001.pdf")
        assert m.group(1) == "hep-th"

    def test_trailing_digit_blocks(self):
        assert ARXIV_OLD_ID_RE.search("math/06010010") is None

    def test_no_match_too_few_digits(self):
        assert ARXIV_OLD_ID_RE.search("math/060100") is None

    def test_no_match_too_many_digits(self):
        assert ARXIV_OLD_ID_RE.search("math/06010012") is None

    def test_embedded_in_filename(self):
        m = ARXIV_OLD_ID_RE.search("Smith 2003 hep-ex_0307015v1.pdf")
        assert m.group(1) == "hep-ex"
        assert m.group(2) == "0307015"
        assert m.group(3) == "v1"


# ---- extract_id -------------------------------------------------------------

class TestExtractId:
    """Verify arXiv ID extraction from filenames."""

    # -- New-style --

    def test_5digit_in_filename(self):
        assert extract_id("2301.12345.pdf") == "2301.12345"

    def test_5digit_versioned(self):
        assert extract_id("2301.12345v2.pdf") == "2301.12345v2"

    def test_4digit_in_filename(self):
        assert extract_id("0704.0001.pdf") == "0704.0001"

    def test_4digit_versioned(self):
        assert extract_id("0704.0001v2.pdf") == "0704.0001v2"

    def test_underscore_surrounded(self):
        assert extract_id("Smith_2301.12345v2_preprint.pdf") == "2301.12345v2"

    def test_space_surrounded(self):
        assert extract_id("Smith 2023 2301.12345v1 draft.pdf") == "2301.12345v1"

    def test_double_digit_version(self):
        assert extract_id("2301.12345v12.pdf") == "2301.12345v12"

    def test_id_at_start(self):
        assert extract_id("2301.12345.tar.gz") == "2301.12345"

    def test_bare_id(self):
        assert extract_id("2301.12345") == "2301.12345"

    # -- Old-style --

    def test_old_style_slash(self):
        """Canonical form with /."""
        assert extract_id("hep-th/0601001.pdf") == "hep-th/0601001"

    def test_old_style_underscore(self):
        assert extract_id("hep-th_0601001v2.pdf") == "hep-th/0601001v2"

    def test_old_style_hyphen(self):
        assert extract_id("hep-ex-0307015v1.pdf") == "hep-ex/0307015v1"

    def test_old_style_dot_subcategory(self):
        assert extract_id("math.GT_0309136.pdf") == "math.GT/0309136"

    def test_old_style_in_user_named_file(self):
        assert extract_id("Smith 2003 hep-ex_0307015v1.pdf") == "hep-ex/0307015v1"

    # -- Non-matching --

    def test_no_id_returns_none(self):
        assert extract_id("my_paper_draft.pdf") is None

    def test_empty_string(self):
        assert extract_id("") is None

    def test_no_id_short_number(self):
        assert extract_id("12345.pdf") is None

    # -- Priority: new-style wins over old-style --

    def test_new_style_preferred(self):
        """If both patterns could match, new-style wins."""
        assert extract_id("2301.12345.pdf") == "2301.12345"


# ---- split_version ----------------------------------------------------------

class TestSplitVersion:
    """Verify version splitting from arXiv IDs."""

    def test_with_version(self):
        assert split_version("2103.12345v3") == ("2103.12345", 3)

    def test_without_version(self):
        assert split_version("2103.12345") == ("2103.12345", 1)

    def test_version_1_explicit(self):
        assert split_version("2103.12345v1") == ("2103.12345", 1)

    def test_high_version(self):
        assert split_version("2301.00001v15") == ("2301.00001", 15)

    def test_strips_whitespace(self):
        assert split_version("  2103.12345v2  ") == ("2103.12345", 2)

    def test_strips_whitespace_no_version(self):
        assert split_version("  2103.12345  ") == ("2103.12345", 1)

    def test_4digit_id(self):
        assert split_version("0704.0001v2") == ("0704.0001", 2)

    def test_old_style_id(self):
        assert split_version("hep-th/0601001v2") == ("hep-th/0601001", 2)

    def test_old_style_no_version(self):
        assert split_version("hep-th/0601001") == ("hep-th/0601001", 1)


# ---- strip_version ----------------------------------------------------------

class TestStripVersion:
    """Verify version suffix removal."""

    def test_removes_version(self):
        assert strip_version("2103.12345v2") == "2103.12345"

    def test_no_version_unchanged(self):
        assert strip_version("2103.12345") == "2103.12345"

    def test_old_style_id(self):
        assert strip_version("hep-ex/0307015v1") == "hep-ex/0307015"

    def test_high_version(self):
        assert strip_version("2301.00001v99") == "2301.00001"

    def test_4digit_id(self):
        assert strip_version("0704.0001v3") == "0704.0001"


# ---- extract_ids_from_filenames ---------------------------------------------

class TestExtractIdsFromFilenames:
    """Verify batch ID extraction from filename lists."""

    def test_basic_extraction(self):
        filenames = ["2301.12345.pdf", "2302.67890v2.pdf"]
        result = extract_ids_from_filenames(filenames)
        assert result == [("2301.12345", "2301.12345.pdf"),
                          ("2302.67890v2", "2302.67890v2.pdf")]

    def test_skips_non_matching(self):
        filenames = ["2301.12345.pdf", "notes.txt", "draft.pdf"]
        result = extract_ids_from_filenames(filenames)
        assert len(result) == 1
        assert result[0] == ("2301.12345", "2301.12345.pdf")

    def test_empty_list(self):
        assert extract_ids_from_filenames([]) == []

    def test_all_non_matching(self):
        filenames = ["notes.txt", "draft.pdf", "readme.md"]
        assert extract_ids_from_filenames(filenames) == []

    def test_canonize_adds_v1(self):
        filenames = ["2301.12345.pdf"]
        result = extract_ids_from_filenames(filenames, canonize=True)
        assert result == [("2301.12345v1", "2301.12345.pdf")]

    def test_canonize_preserves_existing_version(self):
        filenames = ["2301.12345v3.pdf"]
        result = extract_ids_from_filenames(filenames, canonize=True)
        assert result == [("2301.12345v3", "2301.12345v3.pdf")]

    def test_canonize_mixed(self):
        filenames = ["2301.12345.pdf", "2302.67890v2.pdf"]
        result = extract_ids_from_filenames(filenames, canonize=True)
        assert result == [("2301.12345v1", "2301.12345.pdf"),
                          ("2302.67890v2", "2302.67890v2.pdf")]

    def test_preserves_original_filename(self):
        filenames = ["Smith_2301.12345v2_preprint.pdf"]
        result = extract_ids_from_filenames(filenames)
        assert result[0][1] == "Smith_2301.12345v2_preprint.pdf"

    def test_4digit_ids(self):
        filenames = ["0704.0001.pdf", "0704.0001v2.pdf"]
        result = extract_ids_from_filenames(filenames)
        assert result[0][0] == "0704.0001"
        assert result[1][0] == "0704.0001v2"

    def test_old_style_ids(self):
        filenames = ["hep-th_0601001v2.pdf"]
        result = extract_ids_from_filenames(filenames)
        assert result == [("hep-th/0601001v2", "hep-th_0601001v2.pdf")]

    def test_old_style_canonize(self):
        filenames = ["hep-th_0601001.pdf"]
        result = extract_ids_from_filenames(filenames, canonize=True)
        assert result == [("hep-th/0601001v1", "hep-th_0601001.pdf")]
