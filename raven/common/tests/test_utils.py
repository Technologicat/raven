"""Unit tests for raven.common.utils."""

import os
import pathlib
import re
import subprocess
import sys
import time
import types

import numpy as np
import pytest

from raven.common import utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_name_parts(*, last, von=None, first=None, jr=None):
    """Create a mock bibtexparser NameParts-like object."""
    return types.SimpleNamespace(last=last,
                                 von=von or [],
                                 first=first or [],
                                 jr=jr or [])


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

class TestAbsolutizeFilename:
    def test_relative_becomes_absolute(self):
        result = utils.absolutize_filename("foo/bar.txt")
        assert os.path.isabs(result)

    def test_tilde_expanded(self):
        result = utils.absolutize_filename("~/foo.txt")
        assert "~" not in result

    def test_already_absolute(self):
        # Use this test file's own path as a known-absolute input: it exists,
        # it's absolute on every OS, and `absolutize_filename` (which calls
        # `Path.resolve()`) should return it as a fixed point.  A hardcoded
        # POSIX path like "/tmp/foo.txt" does NOT work cross-platform:
        #  - On macOS, `/tmp` is a symlink to `/private/tmp`, so `resolve()`
        #    returns `/private/tmp/foo.txt` — assertion fails.
        #  - On Windows, `/tmp/foo.txt` is not absolute at all; `resolve()`
        #    drive-prefixes it to something like `D:\tmp\foo.txt`.
        # Resolving the reference first makes the comparison symlink-proof.
        already_absolute = str(pathlib.Path(__file__).resolve())
        result = utils.absolutize_filename(already_absolute)
        assert result == already_absolute


class TestStripExt:
    def test_basic(self):
        assert utils.strip_ext("/foo/bar.bib") == "/foo/bar"

    def test_no_extension(self):
        assert utils.strip_ext("/foo/bar") == "/foo/bar"

    def test_multiple_dots(self):
        assert utils.strip_ext("/foo/bar.tar.gz") == "/foo/bar.tar"


class TestMakeCacheFilename:
    # `make_cache_filename` uses `os.path.join` internally, which returns
    # results with the native path separator — `/` on POSIX, `\` on
    # Windows.  Compare against `os.path.join` (same idiom) rather than
    # hardcoded forward-slash strings, so the expected value matches the
    # function's actual output on every OS.
    def test_basic(self):
        result = utils.make_cache_filename("data/papers.bib", "vectors", "npy")
        assert result == os.path.join("data", "papers_vectors.npy")

    def test_pathlib_input(self):
        result = utils.make_cache_filename(pathlib.Path("data/papers.bib"), "cache", "pkl")
        assert result == os.path.join("data", "papers_cache.pkl")

    def test_no_directory(self):
        result = utils.make_cache_filename("papers.bib", "vectors", "npy")
        assert result == "papers_vectors.npy"


class TestValidateCacheMtime:
    def test_cache_newer_than_original(self, tmp_path):
        orig = tmp_path / "orig.txt"
        orig.write_text("original")
        time.sleep(0.05)
        cache = tmp_path / "cache.txt"
        cache.write_text("cached")
        assert utils.validate_cache_mtime(str(cache), str(orig)) is True

    def test_cache_older_than_original(self, tmp_path):
        cache = tmp_path / "cache.txt"
        cache.write_text("cached")
        time.sleep(0.05)
        orig = tmp_path / "orig.txt"
        orig.write_text("original")
        assert utils.validate_cache_mtime(str(cache), str(orig)) is False


class TestOsOpen:
    """`open_file` / `open_in_file_manager` cross-platform dispatch and unified error contract.

    `subprocess.run` is always monkeypatched so no real file manager / viewer is ever launched by the suite.
    """
    def _capture_run(self, monkeypatch):
        """Patch `subprocess.run` to record the command instead of running it; return the record list."""
        calls = []
        def fake_run(cmd, *args, **kwargs):
            calls.append(cmd)
        monkeypatch.setattr(subprocess, "run", fake_run)
        return calls

    def test_missing_path_raises_filenotfounderror(self, tmp_path):
        # FileNotFoundError is an OSError subclass, so the unified `OSError` contract holds here too.
        with pytest.raises(FileNotFoundError):
            utils.open_file(tmp_path / "does_not_exist.png")

    def test_linux_open_file_dispatches_xdg_open(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        calls = self._capture_run(monkeypatch)
        target = tmp_path / "image.png"
        target.write_bytes(b"x")
        utils.open_file(target)
        assert calls == [["xdg-open", str(target.expanduser())]]

    def test_linux_open_in_file_manager_dispatches_xdg_open(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        calls = self._capture_run(monkeypatch)
        utils.open_in_file_manager(tmp_path)  # a directory
        assert calls == [["xdg-open", str(tmp_path.expanduser())]]

    def test_macos_dispatches_open(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        calls = self._capture_run(monkeypatch)
        target = tmp_path / "image.png"
        target.write_bytes(b"x")
        utils.open_file(target)
        assert calls == [["open", str(target.expanduser())]]

    def test_windows_dispatches_startfile(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        started = []
        # os.startfile exists only on Windows, so it isn't there to patch on Linux CI — inject it.
        monkeypatch.setattr(os, "startfile", (lambda p: started.append(p)), raising=False)
        target = tmp_path / "image.png"
        target.write_bytes(b"x")
        utils.open_file(target)
        assert started == [str(target.expanduser())]

    def test_called_process_error_is_reraised_as_oserror(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        def fake_run(cmd, *args, **kwargs):
            raise subprocess.CalledProcessError(returncode=4, cmd=cmd)
        monkeypatch.setattr(subprocess, "run", fake_run)
        target = tmp_path / "image.png"
        target.write_bytes(b"x")
        with pytest.raises(OSError) as excinfo:
            utils.open_file(target)
        assert not isinstance(excinfo.value, subprocess.CalledProcessError)  # unified to OSError, not the raw subprocess type
        assert isinstance(excinfo.value.__cause__, subprocess.CalledProcessError)  # but the cause is preserved


# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------

class TestMakeBlankIndexArray:
    def test_empty_int64(self):
        result = utils.make_blank_index_array()
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
        assert result.dtype == np.int64


# ---------------------------------------------------------------------------
# BibTeX author formatting
# ---------------------------------------------------------------------------

class TestFormatBibtexAuthor:
    def test_simple_last_name(self):
        author = _make_name_parts(last=["Knuth"])
        assert utils.format_bibtex_author(author) == "Knuth"

    def test_multi_word_last_name(self):
        author = _make_name_parts(last=["Brinch", "Hansen"])
        assert utils.format_bibtex_author(author) == "Brinch Hansen"

    def test_von_part(self):
        author = _make_name_parts(von=["van"], last=["Beethoven"])
        assert utils.format_bibtex_author(author) == "van Beethoven"

    def test_jr_part(self):
        author = _make_name_parts(last=["Beeblebrox"], jr=["IV"])
        assert utils.format_bibtex_author(author) == "Beeblebrox IV"

    def test_all_parts(self):
        author = _make_name_parts(von=["de", "la"], last=["Cruz"], jr=["III"], first=["Juan"])
        assert utils.format_bibtex_author(author) == "de la Cruz III"

    def test_empty_last_raises(self):
        author = _make_name_parts(last=[])
        with pytest.raises(ValueError, match="missing last name"):
            utils.format_bibtex_author(author)


class TestFormatBibtexAuthors:
    def test_single_author(self):
        authors = [_make_name_parts(last=["Knuth"])]
        assert utils.format_bibtex_authors(authors) == "Knuth"

    def test_two_authors(self):
        authors = [_make_name_parts(last=["Knuth"]),
                   _make_name_parts(last=["Lamport"])]
        assert utils.format_bibtex_authors(authors) == "Knuth and Lamport"

    def test_three_authors_et_al(self):
        authors = [_make_name_parts(last=["Knuth"]),
                   _make_name_parts(last=["Lamport"]),
                   _make_name_parts(last=["Dijkstra"])]
        assert utils.format_bibtex_authors(authors) == "Knuth et al."

    def test_empty_list(self):
        assert utils.format_bibtex_authors([]) == ""

    def test_invalid_author_returns_empty(self):
        # An author with no last name should cause a warning and return ""
        authors = [_make_name_parts(last=[])]
        assert utils.format_bibtex_authors(authors) == ""

    def test_latex_accents_are_decoded_like_titles_and_abstracts(self):
        """A citation string is for reading, so it gets the same treatment the title already got.

        `H{\\"a}m{\\"a}l{\\"a}inen` is a spelling of a name, not a name. Titles and abstracts have always
        run through `unicodize_basic_markup`; authors did not, which is how a citation could read
        `H{"a}m{"a}l{"a}inen` on screen while the title beside it rendered correctly.
        """
        authors = [_make_name_parts(last=[r'H{\"a}m{\"a}l{\"a}inen']),
                   _make_name_parts(last=[r'Erkkil{\"a}'])]
        assert utils.format_bibtex_authors(authors) == "Hämäläinen and Erkkilä"

    def test_the_raw_field_is_not_this_function_s_concern(self):
        # Lossless export is served by keeping the BibTeX field alongside (the Visualizer importer
        # stores `bibtex_author`), not by refusing to decode here.
        authors = [_make_name_parts(last=[r'Bj{\"o}rklund'])]
        assert utils.format_bibtex_authors(authors) == "Björklund"


# ---------------------------------------------------------------------------
# String normalization
# ---------------------------------------------------------------------------

class TestNormalizeWhitespace:
    def test_multiple_spaces(self):
        assert utils.normalize_whitespace("hello   world") == "hello world"

    def test_tabs_and_newlines(self):
        assert utils.normalize_whitespace("hello\t\nworld") == "hello world"

    def test_leading_trailing(self):
        assert utils.normalize_whitespace("  hello  ") == "hello"

    def test_empty_string(self):
        assert utils.normalize_whitespace("") == ""

    def test_already_normal(self):
        assert utils.normalize_whitespace("hello world") == "hello world"


class TestNormalizeUnicode:
    def test_nfkc_normalization(self):
        # Fullwidth "Ａ" (U+FF21) → regular "A"
        assert utils.normalize_unicode("\uff21") == "A"

    def test_regular_ascii_unchanged(self):
        assert utils.normalize_unicode("hello") == "hello"

    def test_compatibility_superscript(self):
        # Unicode superscript 2 (U+00B2) stays as ² in NFKC
        # But ﬁ (U+FB01, fi ligature) → "fi"
        assert utils.normalize_unicode("\ufb01") == "fi"


class TestUnicodizeBasicMarkup:
    def test_html_subscript(self):
        assert utils.unicodize_basic_markup("CO<sub>2</sub>") == "CO₂"

    def test_html_superscript(self):
        assert utils.unicodize_basic_markup("x<sup>2</sup>") == "x²"

    def test_latex_percent(self):
        assert utils.unicodize_basic_markup(r"100\%") == "100%"

    def test_latex_dollar(self):
        assert utils.unicodize_basic_markup(r"\$5") == "$5"

    def test_html_entity_le(self):
        assert utils.unicodize_basic_markup("x &le; y") == "x ≤ y"

    def test_html_entity_ge(self):
        assert utils.unicodize_basic_markup("x &ge; y") == "x ≥ y"

    def test_html_entity_auml(self):
        assert utils.unicodize_basic_markup("&auml;") == "ä"

    def test_html_entity_ouml(self):
        assert utils.unicodize_basic_markup("&Ouml;") == "Ö"

    def test_html_bold(self):
        assert utils.unicodize_basic_markup("<b>bold</b>") == "*bold*"

    def test_html_italic(self):
        assert utils.unicodize_basic_markup("<i>italic</i>") == "/italic/"

    def test_html_underline(self):
        assert utils.unicodize_basic_markup("<u>underline</u>") == "_underline_"

    def test_lt_gt_entities_last(self):
        # &lt; and &gt; should be replaced after HTML tags are processed
        assert utils.unicodize_basic_markup("a &lt; b &gt; c") == "a < b > c"

    def test_combined(self):
        result = utils.unicodize_basic_markup("H<sub>2</sub>O at 100&le;T")
        assert result == "H₂O at 100≤T"

    # BibTeX case-preservation grouping braces: `bibtexparser` hands us the raw
    # field value, and `{Word}` / `{ACRONYM}` grouping braces must be stripped
    # before the text is shown to the user.

    # Letter-named accent commands (`\c`, `\v`, `\u`, `\H`, `\k`, `\r`) accept their argument either
    # braced or space-separated: a LaTeX control word ends at the first non-letter, so the space *is*
    # the terminator. Real `.bib` files use the space form heavily, because the idiom wraps the whole
    # accent in a case-protecting group — `Tr{\c e}bicki` rather than `Tr\c{e}bicki`.

    def test_letter_accent_with_braced_argument(self):
        assert utils.unicodize_basic_markup(r"Tr\c{e}bicki") == "Trȩbicki"

    def test_letter_accent_with_space_separated_argument(self):
        assert utils.unicodize_basic_markup(r"Tr{\c e}bicki") == "Trȩbicki"

    def test_both_spellings_of_a_letter_accent_agree(self):
        for braced, spaced in ((r"\k{a}", r"{\k a}"),
                               (r"\v{s}", r"{\v s}"),
                               (r"\r{a}", r"{\r a}")):
            assert utils.unicodize_basic_markup(braced) == utils.unicodize_basic_markup(spaced)

    def test_punctuation_accents_do_not_take_a_space(self):
        # `\"` self-terminates, so a space after it is a literal space, not a separator. Decoding
        # `\" a` as "ä" would silently eat a word boundary.
        assert utils.unicodize_basic_markup(r'H\"am\"al\"ainen') == "Hämäläinen"
        assert "ä" not in utils.unicodize_basic_markup(r'x \" a y')

    def test_bibtex_case_group_single_word(self):
        assert utils.unicodize_basic_markup("{AutoPBL}") == "AutoPBL"

    def test_bibtex_case_groups_multiple(self):
        assert utils.unicodize_basic_markup("{An} {LLM}-powered") == "An LLM-powered"

    def test_bibtex_case_group_nested(self):
        # Doubly-wrapped forms like `{{AutoPBL}}` do occur (e.g. from Zotero export).
        assert utils.unicodize_basic_markup("{{AutoPBL}}") == "AutoPBL"

    def test_bibtex_case_group_realistic_title(self):
        src = "{AutoPBL}: {An} {LLM}-powered {Platform}"
        assert utils.unicodize_basic_markup(src) == "AutoPBL: An LLM-powered Platform"

    # `\{` / `\}` (literal escaped braces, produced by `raven.papers.utils.bibtex_escape`)
    # must survive the grouping-brace stripping pass. Regression guard for commit ea7a095.

    def test_latex_literal_braces_preserved(self):
        assert utils.unicodize_basic_markup(r"\{literal\}") == "{literal}"

    def test_latex_literal_braces_mixed_with_grouping(self):
        assert utils.unicodize_basic_markup(r"{Word} and \{literal\}") == "Word and {literal}"

    # LaTeX diacritics: both braced (`\X{c}`) and unbraced (`\Xc`) forms of accent
    # commands, plus single-token ligatures. Coverage spans the common European
    # cases that show up in author names and paper titles in bibliographies.

    def test_latex_diacritic_umlaut_unbraced(self):
        assert utils.unicodize_basic_markup(r'Jer\"onen') == "Jerönen"

    def test_latex_diacritic_acute_unbraced(self):
        assert utils.unicodize_basic_markup(r"caf\'e") == "café"

    def test_latex_diacritic_umlaut_braced(self):
        assert utils.unicodize_basic_markup(r'{\"u}ber') == "über"

    def test_latex_diacritic_umlaut_braced_capital(self):
        assert utils.unicodize_basic_markup(r'The \"{O}stberg effect') == "The Östberg effect"

    def test_latex_diacritic_cedilla_braced(self):
        assert utils.unicodize_basic_markup(r"\c{c}edilla") == "çedilla"

    def test_latex_diacritic_caron_braced(self):
        assert utils.unicodize_basic_markup(r"\v{s}afe") == "šafe"

    def test_latex_diacritic_dotless_i_under_umlaut(self):
        # `\"{\i}` uses dotless-i as a typesetting trick; intended letter is i.
        assert utils.unicodize_basic_markup(r'na\"{\i}ve') == "naïve"

    def test_latex_ligature_ae(self):
        assert utils.unicodize_basic_markup(r"\ae on") == "æ on"

    def test_latex_ligature_oslash_braced(self):
        # The idiomatic `{\o}` form — braces are the command terminator,
        # not case-preservation.
        assert utils.unicodize_basic_markup(r"{\o}nly") == "ønly"

    def test_latex_ligature_ss(self):
        assert utils.unicodize_basic_markup(r"\ss tr") == "ß tr"

    def test_latex_ligature_does_not_match_inside_identifier(self):
        # `\oethr` isn't a real LaTeX command; lookahead must prevent `\oe` from
        # being stripped out of longer backslash-sequences.
        assert utils.unicodize_basic_markup(r"\oethr") == r"\oethr"


# ---------------------------------------------------------------------------
# Search utilities
# ---------------------------------------------------------------------------

class TestNormalizeSearchString:
    def test_whitespace_and_unicode(self):
        assert utils.normalize_search_string("  hello   world  ") == "hello world"

    def test_subscript_to_regular(self):
        assert utils.normalize_search_string("O₂") == "O2"

    def test_superscript_to_regular(self):
        assert utils.normalize_search_string("x²") == "x2"

    def test_mixed(self):
        assert utils.normalize_search_string("CO₂  and  x²") == "CO2 and x2"


class TestSearchStringToFragments:
    def test_all_lowercase_is_case_insensitive(self):
        cs, ci = utils.search_string_to_fragments("cat photo", sort=False)
        assert cs == []
        assert ci == ["cat", "photo"]

    def test_uppercase_is_case_sensitive(self):
        cs, ci = utils.search_string_to_fragments("Cat photo", sort=False)
        assert cs == ["Cat"]
        assert ci == ["photo"]

    def test_sort_longest_first(self):
        cs, ci = utils.search_string_to_fragments("a bb ccc", sort=True)
        assert ci == ["ccc", "bb", "a"]

    def test_unsorted_preserves_order(self):
        cs, ci = utils.search_string_to_fragments("a bb ccc", sort=False)
        assert ci == ["a", "bb", "ccc"]

    def test_empty_string(self):
        cs, ci = utils.search_string_to_fragments("", sort=False)
        assert cs == []
        assert ci == []

    def test_single_fragment(self):
        cs, ci = utils.search_string_to_fragments("photocatalytic", sort=False)
        assert ci == ["photocatalytic"]

    def test_subscripts_normalized(self):
        cs, ci = utils.search_string_to_fragments("CO₂", sort=False)
        # "CO2" has uppercase letters, so it's case-sensitive
        assert cs == ["CO2"]
        assert ci == []


class TestMakeSearchMatcher:
    def test_lowercase_query_ignores_case(self):
        matches = utils.make_search_matcher("readme")
        assert matches("README.txt")
        assert matches("readme.txt")
        assert matches("ReadMe.TXT")

    def test_query_with_uppercase_is_case_sensitive(self):
        matches = utils.make_search_matcher("README")
        assert matches("README.txt")
        assert not matches("readme.txt")

    def test_fragments_match_anywhere_and_in_any_order(self):
        """The HELM/Firefox behavior: fragments are ANDed, position and order are free."""
        matches = utils.make_search_matcher("photo cat")
        assert matches("photocatalytic")
        assert matches("cat_photo.png")
        assert not matches("photosynthesis")

    def test_mixed_case_fragments_are_judged_separately(self):
        matches = utils.make_search_matcher("Cat photo")
        assert matches("Cat photo")
        assert matches("Cat PHOTO")  # "photo" is lowercase in the query, so its case is free
        assert not matches("cat photo")  # ...but "Cat" is not, so this one must match exactly

    def test_empty_query_accepts_everything(self):
        """So a call site with no query needs no special case, which is the reason it is spelled this way."""
        matches = utils.make_search_matcher("")
        assert matches("anything at all")
        assert matches("")

    def test_query_is_normalized(self):
        """`normalize_search_string` runs on the query, so a subscript typed by the user still finds "CO2"."""
        matches = utils.make_search_matcher("CO₂")
        assert matches("CO2 capture")

    def test_matcher_can_be_reused(self):
        """Compile once, test many — the whole point of returning a predicate."""
        matches = utils.make_search_matcher("report")
        assert [name for name in ["report.pdf", "notes.txt", "final_report.docx"] if matches(name)] == ["report.pdf", "final_report.docx"]


class TestSearchFragmentToHighlightRegex:
    def test_parens_escaped(self):
        result = utils.search_fragment_to_highlight_regex_fragment("f(x)")
        assert r"\(" in result
        assert r"\)" in result

    def test_dot_escaped(self):
        result = utils.search_fragment_to_highlight_regex_fragment("e.g.")
        assert r"\." in result

    def test_brackets_escaped(self):
        result = utils.search_fragment_to_highlight_regex_fragment("[a]")
        assert r"\[" in result
        assert r"\]" in result

    def test_digits_expanded(self):
        result = utils.search_fragment_to_highlight_regex_fragment("H2O")
        # "2" should be expanded to include subscript and superscript variants
        assert "₂" in result
        assert "²" in result

    def test_result_is_valid_regex(self):
        result = utils.search_fragment_to_highlight_regex_fragment("CO2")
        # Should compile without error
        re.compile(result)


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

class TestChunkifyText:
    def test_short_text_single_chunk(self):
        chunks = utils.chunkify_text("hello", chunk_size=100, overlap=0, extra=0.4)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "hello"
        assert chunks[0]["chunk_id"] == 0
        assert chunks[0]["offset"] == 0

    def test_exact_chunk_size_single_chunk(self):
        text = "a" * 100
        chunks = utils.chunkify_text(text, chunk_size=100, overlap=0, extra=0.4)
        assert len(chunks) == 1

    def test_multiple_chunks(self):
        text = "a" * 300
        chunks = utils.chunkify_text(text, chunk_size=100, overlap=0, extra=0.0)
        assert len(chunks) == 3
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_id"] == i

    def test_overlap_shared_content(self):
        text = "abcdefghij" * 10  # 100 chars
        chunks = utils.chunkify_text(text, chunk_size=40, overlap=10, extra=0.0)
        assert len(chunks) > 1
        # The end of chunk[0] should appear at the start of chunk[1]
        overlap_from_first = chunks[0]["text"][-10:]
        assert chunks[1]["text"].startswith(overlap_from_first)

    def test_zero_overlap(self):
        text = "a" * 200
        chunks = utils.chunkify_text(text, chunk_size=100, overlap=0, extra=0.0)
        total_len = sum(len(c["text"]) for c in chunks)
        assert total_len == 200

    def test_orphan_control_folds_remainder(self):
        # 130 chars with chunk_size=100 and extra=0.4:
        # remainder is 30, which is < 0.4*100=40, so it should fold into one chunk
        text = "a" * 130
        chunks = utils.chunkify_text(text, chunk_size=100, overlap=0, extra=0.4)
        assert len(chunks) == 1
        assert len(chunks[0]["text"]) == 130

    def test_orphan_control_separate_chunk(self):
        # 160 chars with chunk_size=100 and extra=0.1:
        # remainder is 60, which is > 0.1*100=10, so it becomes a separate chunk
        text = "a" * 160
        chunks = utils.chunkify_text(text, chunk_size=100, overlap=0, extra=0.1)
        assert len(chunks) == 2

    def test_offsets_correct(self):
        text = "a" * 300
        chunks = utils.chunkify_text(text, chunk_size=100, overlap=0, extra=0.0)
        for chunk in chunks:
            assert text[chunk["offset"]:chunk["offset"] + len(chunk["text"])] == chunk["text"]

    def test_chunk_ids_sequential(self):
        text = "a" * 500
        chunks = utils.chunkify_text(text, chunk_size=100, overlap=0, extra=0.0)
        ids = [c["chunk_id"] for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_with_trimmer(self):
        # Trimmer that strips 5 chars from the beginning (except first chunk) and 5 from the end (except last chunk)
        def trimmer(overlap, mode, text):
            if mode == "first":
                return text[:-5], 0
            elif mode == "last":
                return text[5:], 5
            else:  # middle
                return text[5:-5], 5
        text = "a" * 300
        chunks = utils.chunkify_text(text, chunk_size=100, overlap=10, extra=0.0, trimmer=trimmer)
        assert len(chunks) >= 2
        # First chunk should have trimmed end
        assert len(chunks[0]["text"]) == 95

    def test_empty_text(self):
        chunks = utils.chunkify_text("", chunk_size=100, overlap=0, extra=0.4)
        assert len(chunks) == 1
        assert chunks[0]["text"] == ""


# ---------------------------------------------------------------------------
# Reading BibTeX by its surface syntax, for when the parser has refused
# ---------------------------------------------------------------------------

class TestBibtexHeaderKey:
    def test_a_header_line_yields_its_key(self):
        assert utils.bibtex_header_key("@article{WOS:000258806000016,") == "WOS:000258806000016"

    def test_the_key_is_verbatim(self):
        # Unlike `papers.burstbib.get_slug`, which sanitizes for use as a filename.
        assert utils.bibtex_header_key("@misc{a/b:c,") == "a/b:c"

    def test_a_line_that_is_not_a_header_yields_nothing(self):
        assert utils.bibtex_header_key("Title = {Something}") == ""
        assert utils.bibtex_header_key("") == ""


class TestBibtexFieldValue:
    def test_a_field_is_found_by_name(self):
        assert utils.bibtex_field_value("@a{k,\n\tTitle = {Some Paper},\n}", "title") == "Some Paper"

    def test_the_key_case_does_not_matter(self):
        # A Web of Science export capitalizes its keys; the BibTeX literature does not.
        assert utils.bibtex_field_value("@a{k,\n\ttitle = {Lower},\n}", "Title") == "Lower"

    def test_it_reads_a_record_the_parser_would_refuse(self):
        # The whole point: an unbalanced brace elsewhere aborts a real parse, title and all.
        broken = "@a{k,\n\tAbstract = {System {[production]),\n\tTitle = {Still here},\n}"
        assert utils.bibtex_field_value(broken, "title") == "Still here"

    def test_an_absent_field_yields_nothing(self):
        assert utils.bibtex_field_value("@a{k,\n\tYear = {2020},\n}", "title") == ""


class TestBibtexUnbalancedFieldNames:
    def test_the_offending_field_is_named(self):
        broken = "@a{k,\n\tAbstract = {oops {,\n\tYear = {2020},\n}"
        assert utils.bibtex_unbalanced_field_names(broken) == ["Abstract"]

    def test_a_sound_record_names_nothing(self):
        assert utils.bibtex_unbalanced_field_names("@a{k,\n\tYear = {2020},\n}") == []

    def test_a_multiline_value_is_a_suspect_not_a_verdict(self):
        # An `Affiliation` listing one author per line is unbalanced line by line and perfectly valid.
        multiline = "@a{k,\n\tAffiliation = {First, Somewhere\nSecond, Elsewhere},\n}"
        assert utils.bibtex_unbalanced_field_names(multiline) == ["Affiliation"]


class TestBibtexBraceRepairCandidates:
    """The candidates are guesses; a parser decides. These pin what is proposed, and what never is."""

    def test_a_stray_opener_in_running_text_is_escaped(self):
        broken = "@a{k,\n\tAbstract = {set {0 <= rho for all rho},\n}"
        candidates = utils.bibtex_brace_repair_candidates(broken)
        assert candidates == ["@a{k,\n\tAbstract = {set \\{0 <= rho for all rho},\n}"]

    def test_a_stray_closer_in_running_text_is_escaped_first(self):
        # A stray literal falls between its value's opening brace and its terminator, so a surplus closer
        # is more likely the earlier candidate — the opposite bias from a surplus opener. Both are offered;
        # the order decides which one the oracle gets to accept.
        broken = "@a{k,\n\tAbstract = {closing } for nothing},\n}"
        assert utils.bibtex_brace_repair_candidates(broken)[0] == "@a{k,\n\tAbstract = {closing \\} for nothing},\n}"

    def test_a_repair_changes_nothing_but_the_escapes(self):
        broken = "@a{k,\n\tAbstract = {set {0 <= rho},\n}"
        candidate = utils.bibtex_brace_repair_candidates(broken)[0]
        assert candidate.replace("\\{", "{").replace("\\}", "}") == broken

    def test_a_balanced_record_is_left_alone(self):
        assert utils.bibtex_brace_repair_candidates("@a{k,\n\tYear = {2020},\n}") == []

    def test_the_structural_braces_are_never_proposed(self):
        # The header's `{`, each `Key = {` opener and the record's own closing `}` hold the record together.
        # A naive unmatched-bracket scan blames the header first, and escaping it destroys the record.
        broken = "@a{k,\n\tAbstract = {oops {,\n\tYear = {2020},\n}"
        for candidate in utils.bibtex_brace_repair_candidates(broken):
            assert candidate.startswith("@a{k,")
            assert "\\{2020" not in candidate and "Abstract = \\{" not in candidate
            assert candidate.endswith("\n}")

    def test_a_multiline_value_does_not_confuse_the_proposal(self):
        broken = ("@a{k,\n\tAbstract = {first line\nsecond {line\nthird line},\n\tYear = {2020},\n}")
        assert utils.bibtex_brace_repair_candidates(broken) == [
            "@a{k,\n\tAbstract = {first line\nsecond \\{line\nthird line},\n\tYear = {2020},\n}"]

    def test_a_record_too_tangled_to_guess_at_is_declined(self):
        # Two braces missing among many that legitimately pair up: the combinations explode, and proposing
        # hundreds of guesses is not repair, it is brute force with a parser attached.
        pairs = " ".join("{g%d}" % i for i in range(10))
        broken = "@a{k,\n\tAbstract = {%s {stray {stray},\n}" % pairs
        assert utils.bibtex_brace_repair_candidates(broken, max_candidates=10) == []

    def test_the_surplus_count_sets_how_many_braces_a_candidate_escapes(self):
        broken = "@a{k,\n\tAbstract = {a {b {c},\n}"  # two openers short of balance
        assert all(candidate.count("\\{") == 2 for candidate in utils.bibtex_brace_repair_candidates(broken))


class TestUserDirectory:
    """Finding the user's Pictures directory when it is not called "Pictures".

    On Linux the XDG user directories are renamed *on disk* — a Finnish desktop has `~/Kuvat`, not
    `~/Pictures` — so joining the home directory with an English name finds nothing. Windows and macOS
    localize what the file manager displays and leave the directory itself in English, so the join is the
    right answer there and asking XDG would be wrong.
    """

    @pytest.fixture
    def home(self, tmp_path, monkeypatch):
        """A home directory of our own, with no inherited XDG settings leaking in from the real one.

        Both variables are needed, and for different consumers. `expanduser` reads `HOME` through
        `posixpath` and `USERPROFILE` through `ntpath` — never `HOME` on Windows — so setting only the
        former leaves every fallback assertion pointing at the real profile of whoever runs the suite.
        `expandvars`, which expands the `$HOME` inside an XDG value, uses plain environment substitution
        on all platforms and so was satisfied by `HOME` alone; that is why the Windows runner failed on
        exactly the tests that do *not* read an XDG file.
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
        for key in utils._XDG_USER_DIRS.values():
            monkeypatch.delenv(key, raising=False)
        return tmp_path

    def write_user_dirs(self, home, text):
        config = home / ".config"
        config.mkdir(parents=True, exist_ok=True)
        (config / "user-dirs.dirs").write_text(text, encoding="utf-8")

    def test_home_is_the_home_directory(self, home, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        assert utils.user_directory("Home") == home

    def test_a_localized_directory_is_found_through_xdg(self, home, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        self.write_user_dirs(home, 'XDG_PICTURES_DIR="$HOME/Kuvat"\n')
        assert utils.user_directory("Pictures") == home / "Kuvat"

    def test_an_absolute_xdg_path_is_taken_as_given(self, home, monkeypatch):
        """The value need not live under the home directory — a photo library on another disk is normal."""
        monkeypatch.setattr(sys, "platform", "linux")
        self.write_user_dirs(home, 'XDG_PICTURES_DIR="/mnt/photos"\n')
        assert utils.user_directory("Pictures") == pathlib.Path("/mnt/photos")

    def test_comments_and_other_keys_are_skipped(self, home, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        self.write_user_dirs(home, '# generated by xdg-user-dirs-update\n'
                                   'XDG_MUSIC_DIR="$HOME/Musiikki"\n'
                                   'XDG_PICTURES_DIR="$HOME/Kuvat"\n')
        assert utils.user_directory("Pictures") == home / "Kuvat"
        assert utils.user_directory("Music") == home / "Musiikki"

    def test_an_exported_value_beats_the_file(self, home, monkeypatch):
        """The file is meant to be *sourced*, so an exported value is what sourcing it would have produced."""
        monkeypatch.setattr(sys, "platform", "linux")
        self.write_user_dirs(home, 'XDG_PICTURES_DIR="$HOME/Kuvat"\n')
        monkeypatch.setenv("XDG_PICTURES_DIR", str(home / "Bilder"))
        assert utils.user_directory("Pictures") == home / "Bilder"

    def test_a_missing_file_falls_back_to_the_english_name(self, home, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        assert utils.user_directory("Pictures") == home / "Pictures"

    def test_a_key_the_file_does_not_answer_falls_back(self, home, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        self.write_user_dirs(home, 'XDG_MUSIC_DIR="$HOME/Musiikki"\n')
        assert utils.user_directory("Pictures") == home / "Pictures"

    def test_an_empty_value_falls_back(self, home, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        self.write_user_dirs(home, 'XDG_PICTURES_DIR=""\n')
        assert utils.user_directory("Pictures") == home / "Pictures"

    @pytest.mark.parametrize("platform_name", ["win32", "darwin"])
    def test_windows_and_macos_ignore_xdg(self, home, monkeypatch, platform_name):
        """Those two keep the directory itself in English, so an XDG file — if any — is not theirs to read."""
        monkeypatch.setattr(sys, "platform", platform_name)
        self.write_user_dirs(home, 'XDG_PICTURES_DIR="$HOME/Kuvat"\n')
        assert utils.user_directory("Pictures") == home / "Pictures"

    def test_an_unknown_name_falls_back(self, home, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        assert utils.user_directory("Screenshots") == home / "Screenshots"

    def test_the_download_key_is_singular(self, home, monkeypatch):
        """`XDG_DOWNLOAD_DIR` where every sibling is plural — the spec's own inconsistency, easy to mistype."""
        monkeypatch.setattr(sys, "platform", "linux")
        self.write_user_dirs(home, 'XDG_DOWNLOAD_DIR="$HOME/Lataukset"\n')
        assert utils.user_directory("Downloads") == home / "Lataukset"
