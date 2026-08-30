"""Tests for the BibTeX reader and the arXiv feed → BibTeX converter."""

from unittest.mock import MagicMock

from raven.papers.bibtex import (_clean_whitespace, _field_spans, _make_key, _undelimit,
                                 brace_repair_candidates, decode_html_entities, entries_to_bibtex,
                                 field_value, header_key, parse_file, parse_string,
                                 relocate_rights_notices, repair_duplicate_field_keys,
                                 unbalanced_field_names, write_string)


def _fake_entry(
    arxiv_id="2103.12345v2",
    title="Some Title",
    authors=None,
    published="2021-03-23T00:00:00Z",
    summary="An abstract.",
    primary_category="quant-ph",
    doi=None,
    journal_ref=None,
):
    """Build a dict mimicking a feedparser entry for an arXiv result."""
    if authors is None:
        authors = [{"name": "Alice Smith"}, {"name": "Bob Jones"}]
    entry = MagicMock()
    entry.id = f"http://arxiv.org/abs/{arxiv_id}"
    entry.published = published
    entry.get = lambda key, default=None: {
        "title": title,
        "summary": summary,
        "authors": authors,
        "arxiv_primary_category": {"term": primary_category},
        "arxiv_doi": doi,
        "arxiv_journal_ref": journal_ref,
    }.get(key, default)
    return entry


class TestCleanWhitespace:
    def test_collapses_newlines(self):
        assert _clean_whitespace("a\n  b\n  c") == "a b c"

    def test_collapses_tabs(self):
        assert _clean_whitespace("a\t\tb") == "a b"

    def test_strips_leading_trailing(self):
        assert _clean_whitespace("  hello  ") == "hello"


class TestMakeKey:
    def test_basic(self):
        entry = _fake_entry()
        assert _make_key(entry) == "Smith_2021_2103.12345"

    def test_strips_version(self):
        entry = _fake_entry(arxiv_id="2103.12345v3")
        key = _make_key(entry)
        assert "v3" not in key
        assert "2103.12345" in key

    def test_old_style_id(self):
        entry = _fake_entry(arxiv_id="hep-ex/0307015v1")
        key = _make_key(entry)
        assert key == "Smith_2021_hep-ex_0307015"

    def test_no_authors(self):
        entry = _fake_entry(authors=[])
        key = _make_key(entry)
        assert key.startswith("Unknown_")


class TestEntriesToBibtex:
    def test_basic_output(self):
        entry = _fake_entry(
            title="Quantum Error Correction",
            doi="10.1234/test",
            journal_ref="Nature 605, 669 (2022)",
        )
        bib = entries_to_bibtex([entry])

        assert "@article{Smith_2021_2103.12345" in bib
        assert "Quantum Error Correction" in bib
        assert "Alice Smith and Bob Jones" in bib
        assert "2021" in bib
        assert "arXiv" in bib
        assert "10.1234/test" in bib
        assert "Nature 605, 669 (2022)" in bib

    def test_multiple_entries(self):
        e1 = _fake_entry(arxiv_id="2101.00001v1")
        e2 = _fake_entry(arxiv_id="2102.00002v1")
        bib = entries_to_bibtex([e1, e2])
        assert bib.count("@article{") == 2

    def test_no_doi_or_journal(self):
        entry = _fake_entry()
        bib = entries_to_bibtex([entry])
        assert "doi" not in bib.lower().split("archiveprefix")[0]  # no doi field
        assert "journal" not in bib.lower()


# ---------------------------------------------------------------------------
# The reader side: `parse_string` / `parse_file` and the middleware they carry
# ---------------------------------------------------------------------------

# These pin the middleware chain rather than `bibtexparser` itself. The chain is the reason the readers
# exist: two call sites (the Visualizer importer and Librarian's paste sniffer) had assembled it
# independently, and a silent divergence between them would mean the same `.bib` parsing differently
# depending on which door it came in by.

def _last_names(entry):
    """The `last` name parts of every author, as joined strings — what `SplitNameParts` should produce."""
    return [" ".join(name.last) for name in entry["author"]]


class TestParseString:
    def test_field_keys_are_normalized_to_lowercase(self):
        # A Web of Science export writes `Title = {...}`; the BibTeX literature writes `title = {...}`.
        library = parse_string("@article{k, Title={A Study}, YEAR={2024}}")
        assert sorted(f.key for f in library.entries[0].fields) == ["title", "year"]

    def test_coauthors_are_separated(self):
        library = parse_string("@article{k, author={Alice Smith and Bob Jones}, year={2024}}")
        assert _last_names(library.entries[0]) == ["Smith", "Jones"]

    def test_name_parts_survive_a_von_particle(self):
        library = parse_string("@article{k, author={Ludwig van Beethoven}, year={2024}}")
        name = library.entries[0]["author"][0]
        assert name.von == ["van"]
        assert name.last == ["Beethoven"]

    def test_name_parts_survive_a_compound_last_name(self):
        # "Brinch Hansen, Per" — the comma is what marks the whole of "Brinch Hansen" as the surname.
        library = parse_string("@article{k, author={Brinch Hansen, Per}, year={2024}}")
        name = library.entries[0]["author"][0]
        assert name.last == ["Brinch", "Hansen"]
        assert name.first == ["Per"]

    def test_name_parts_survive_a_suffix(self):
        library = parse_string("@article{k, author={Beeblebrox, IV, Zaphod}, year={2024}}")
        name = library.entries[0]["author"][0]
        assert name.last == ["Beeblebrox"]
        assert name.jr == ["IV"]
        assert name.first == ["Zaphod"]

    # The shapes below are the ones that actually occur in the axially-moving-materials bibliography
    # (538 author fields, 794 distinct names). Names are invented, formats are not — real corpora produce
    # these and invented test data tends not to.

    def test_a_particle_in_comma_form_is_still_a_particle(self):
        # "van Rijn, Charles F." reaches a different code path than "Ludwig van Beethoven": the comma
        # already marks the surname, so the particle has to be recognized inside it rather than before it.
        library = parse_string("@article{k, author={van Rijn, Rembrandt H.}, year={2024}}")
        name = library.entries[0]["author"][0]
        assert name.von == ["van"]
        assert name.last == ["Rijn"]
        assert name.first == ["Rembrandt", "H."]

    def test_a_brace_protected_suffix_is_read_as_a_suffix(self):
        # `{III}` — braces are BibTeX's "do not touch this" marker, and the suffix slot still has to see it.
        library = parse_string("@article{k, author={Aldrin, {III}, Edwin E.}, year={2024}}")
        name = library.entries[0]["author"][0]
        assert name.last == ["Aldrin"]
        assert name.jr == ["{III}"]

    def test_a_suffix_carrying_a_period_is_read_as_a_suffix(self):
        library = parse_string("@article{k, author={Fripp, Jr., R. A.}, year={2024}}")
        name = library.entries[0]["author"][0]
        assert name.last == ["Fripp"]
        assert name.jr == ["Jr."]
        assert name.first == ["R.", "A."]

    def test_hyphens_survive_in_both_initials_and_given_names(self):
        library = parse_string("@article{k, author={Zhou, X.-Y. and Liisa-Maria Koskinen}, year={2024}}")
        zhou, koskinen = library.entries[0]["author"]
        assert zhou.first == ["X.-Y."]      # a hyphenated initial pair is one token, not two
        assert koskinen.first == ["Liisa-Maria"]

    def test_tex_accent_escapes_are_preserved_verbatim_not_decoded(self):
        """The contract a display layer needs to know about: what comes out is still TeX, not Unicode.

        Both spellings of an umlaut survive as written, and neither becomes "ä". A consumer that renders
        author names is responsible for the conversion; nothing in this layer does it. Pinned because the
        alternative is discovering it in a UI, where it shows up as literal braces on screen.
        """
        library = parse_string(r'@article{k, author={H{\"a}kkinen, M. and H\"akkinen, T.}, year={2024}}')
        braced, bare = library.entries[0]["author"]
        assert braced.last == [r'H{\"a}kkinen']
        assert bare.last == [r'H\"akkinen']

    def test_an_unreadable_record_lands_in_failed_blocks_rather_than_raising(self):
        # The documented contract: a successful return is not a promise that every record was understood.
        # Librarian's paste sniffer relies on partial success, and the importer reports the failures.
        library = parse_string("@article{good, title={Fine}, year={2024}}\n@article{good, title={Dupe}}")
        assert len(library.entries) == 1
        assert len(library.failed_blocks) == 1


class TestParseFile:
    def test_reads_a_file_with_the_same_middleware_as_parse_string(self, tmp_path):
        path = tmp_path / "refs.bib"
        path.write_text("@article{k, Title={A Study}, author={Alice Smith and Bob Jones}, year={2024}}",
                        encoding="utf-8")

        from_file = parse_file(path)
        from_string = parse_string(path.read_text(encoding="utf-8"))

        assert [f.key for f in from_file.entries[0].fields] == [f.key for f in from_string.entries[0].fields]
        assert _last_names(from_file.entries[0]) == _last_names(from_string.entries[0]) == ["Smith", "Jones"]

    def test_accepts_a_path_object_and_a_string(self, tmp_path):
        # The importer passes a str, Librarian-side callers are likelier to hold a Path. Both must work.
        path = tmp_path / "refs.bib"
        path.write_text("@article{k, title={A Study}, year={2024}}", encoding="utf-8")

        assert len(parse_file(path).entries) == 1
        assert len(parse_file(str(path)).entries) == 1


class TestSplitNamesOff:
    def test_author_stays_the_string_the_file_had(self):
        library = parse_string("@article{k, author={Beeblebrox, IV, Zaphod and Alice Smith}, year={2024}}",
                               split_names=False)
        assert library.entries[0]["author"] == "Beeblebrox, IV, Zaphod and Alice Smith"

    def test_field_keys_are_still_normalized(self):
        # The one middleware that is not about names stays on: a rewriting caller wants it as much as a
        # reading one, since it is what makes `entry["title"]` work on a Web of Science export.
        library = parse_string("@article{k, Title={A Study}, YEAR={2024}}", split_names=False)
        assert sorted(f.key for f in library.entries[0].fields) == ["title", "year"]


class TestWriteString:
    # `author` here exercises all three shapes the name splitter handles, so a round trip that keeps it
    # intact has kept the hard cases; the other fields cover the delimiters and escapes.
    SOURCE = ("@article{k,\n"
              "  title = {A Study of {LaTeX} Braces and Ümlauts},\n"
              "  author = {Smith, Jane and van Beethoven, Ludwig and Beeblebrox, IV, Zaphod},\n"
              "  year = 2024,\n"
              "  note = \"a quoted value\",\n"
              "  abstract = {100\\% coverage},\n"
              "}\n")

    def test_upstream_writes_a_split_library_as_repr(self):
        """Why this module has a writer at all — pinned, because it is the failure it exists to prevent.

        `bibtexparser.write_string` renders a value it does not recognize with `repr()`, so a library read
        through the default chain comes back out with every author field replaced by the text of a Python
        object — a file that still parses as BibTeX, with the authors gone and nothing logged.
        """
        import bibtexparser
        mangled = bibtexparser.write_string(parse_string(self.SOURCE))
        assert "NameParts(" in mangled

    def test_a_split_library_round_trips(self):
        assert "NameParts(" not in write_string(parse_string(self.SOURCE))
        rewritten = parse_string(write_string(parse_string(self.SOURCE)))
        assert _last_names(rewritten.entries[0]) == ["Smith", "Beethoven", "Beeblebrox"]

    def test_an_unsplit_library_round_trips(self):
        rewritten = parse_string(write_string(parse_string(self.SOURCE, split_names=False)))
        assert _last_names(rewritten.entries[0]) == ["Smith", "Beethoven", "Beeblebrox"]

    def test_both_chains_write_the_same_text(self):
        """The property that lets `write_string` decide from the data instead of taking an argument.

        If the two disagreed, a caller would have to remember which reader produced the library, and the
        way to get that wrong silently is the one above.
        """
        assert write_string(parse_string(self.SOURCE)) == write_string(parse_string(self.SOURCE, split_names=False))

    def test_field_values_survive_byte_for_byte(self):
        written = write_string(parse_string(self.SOURCE, split_names=False))
        assert "{A Study of {LaTeX} Braces and Ümlauts}" in written  # inner groups and non-ASCII both
        assert "{100\\% coverage}" in written                        # and the escape stays an escape

    def test_writing_is_idempotent(self):
        # A bibliography that has been through the tool once must not keep changing on later passes;
        # otherwise every re-run of a rewriting tool shows a diff that means nothing.
        once = write_string(parse_string(self.SOURCE, split_names=False))
        assert write_string(parse_string(once, split_names=False)) == once

    def test_an_entry_with_no_names_at_all_is_written(self):
        # The detection asks whether *any* field holds a list. A library with no author anywhere must
        # take the no-merge path rather than fall off the end of `any()` into an error.
        written = write_string(parse_string("@book{b, title={No author at all}, year={2024}}"))
        assert "No author at all" in written


class TestDecodeHtmlEntities:
    def test_an_escaped_ampersand_entity_becomes_an_escaped_ampersand(self):
        """How this actually arrives: HTML exported into a `.bib`, with the `&` escaped on the way in.

        The escape in front belongs to the entity's own `&`, which is syntax, so it goes with it — and
        the decoded `&` needs an escape of its own, which is why this is not simply a text decode.
        """
        assert decode_html_entities(r"Q\&amp;A") == (r"Q\&A", 1)

    def test_an_unescaped_ampersand_entity_gains_the_escape_it_needs(self):
        assert decode_html_entities("Bell &amp; Howell") == (r"Bell \& Howell", 1)

    def test_an_even_backslash_run_is_left_alone(self):
        # `\\` is an escaped backslash — content, not an escape for what follows.
        assert decode_html_entities(r"a\\&amp;b") == (r"a\\\&b", 1)

    def test_a_named_entity_becomes_its_character(self):
        assert decode_html_entities("H&auml;kkinen") == ("Häkkinen", 1)

    def test_a_numeric_entity_becomes_its_character(self):
        assert decode_html_entities("Students&#8217; views") == ("Students’ views", 1)
        assert decode_html_entities("&#x2014;dash") == ("—dash", 1)

    def test_an_entity_naming_nothing_is_left_alone_and_not_counted(self):
        """The count is what a caller reports to a user, so it must be decodes and not matches.

        `re.subn` would have said 1 here, having substituted the text with itself.
        """
        assert decode_html_entities("&foo;") == ("&foo;", 0)

    def test_an_ampersand_that_is_not_an_entity_is_left_alone(self):
        """HTML5 also defines entities without the trailing semicolon, and honouring those here would
        rewrite `AT&T` at the first opportunity."""
        assert decode_html_entities("AT&T") == ("AT&T", 0)

    def test_an_escaped_entity_decodes_only_once(self):
        # `&amp;lt;` is how a source writes a literal `&lt;`; the result must still say that.
        assert decode_html_entities("&amp;lt;") == (r"\&lt;", 1)

    def test_a_format_character_is_dropped_rather_than_written(self):
        """A zero-width joiner or a directional mark says nothing a bibliography record needs, and
        leaves nothing on screen to explain the file it is in."""
        for source, expected in (("a&zwj;b", "ab"),          # zero-width joiner
                                 ("&NoBreak;x", "x"),        # word joiner
                                 ("a&lrm;b", "ab")):         # left-to-right mark
            assert decode_html_entities(source)[0] == expected

    def test_a_control_or_separator_becomes_a_space(self):
        """Not tidiness: `fixbib` decodes before repairing and reports faults by line number in the
        user's own file, so a newline arriving mid-record would move every line after it."""
        assert decode_html_entities("line&#10;break")[0] == "line break"

    def test_a_no_break_space_stays_a_no_break_space(self):
        """Not breaking the line there is the whole point of the character, and the source asked for it.

        The analysis path answers this differently and separately: `unicodize_basic_markup` folds it to
        a plain space on the way into the Visualizer, because a tokenizer *should* see a word boundary.
        """
        decoded, count = decode_html_entities(r"1)\&nbsp;more")
        assert count == 1
        assert decoded == "1)\u00a0more"  # no-break space; an escape, since a literal is unreadable here

    def test_the_other_typographic_spaces_survive_too(self):
        # A narrow no-break space is French typography, not an accident.
        assert decode_html_entities("narrow&#8239;space")[0] == "narrow\u202fspace"

    def test_the_line_count_cannot_change(self):
        """`raven-fixbib` reports faults by line number in the user's own file, and decodes before
        repairing — so an entity naming a newline must not move every line after it."""
        source = "@article{k,\n  title = {line&#10;break},\n  year = {2024},\n}\n"
        decoded, _count = decode_html_entities(source)
        assert decoded.count("\n") == source.count("\n")

    def test_text_with_no_entities_is_returned_unchanged(self):
        source = "@article{k,\n  title = {A perfectly ordinary title},\n}\n"
        assert decode_html_entities(source) == (source, 0)

    def test_the_result_still_parses_and_says_what_it_should(self):
        source = ("@article{k,\n  title = {Synchronous \\&amp; Remote},\n"
                  "  author = {H\\&auml;kkinen, P.},\n  year = {2024},\n}\n")
        decoded, count = decode_html_entities(source)
        assert count == 2
        fields = {f.key: f.value for f in parse_string(decoded, split_names=False).entries[0].fields}
        assert fields["title"] == r"Synchronous \& Remote"
        assert fields["author"] == "Häkkinen, P."


class TestRelocateRightsNotices:
    WITH_NOTICE = ("@article{a,\n"
                   "  title = {A Study},\n"
                   "  abstract = {We study things. © 2024 Elsevier Ltd. All rights reserved.},\n"
                   "  year = {2024},\n"
                   "}\n")

    def test_the_notice_moves_into_a_field_of_its_own(self):
        moved, count = relocate_rights_notices(self.WITH_NOTICE)
        assert count == 1
        fields = {f.key: f.value for f in parse_string(moved, split_names=False).entries[0].fields}
        assert fields["abstract"] == "We study things."
        assert fields["copyright"] == "© 2024 Elsevier Ltd. All rights reserved."

    def test_nothing_is_lost_in_the_move(self):
        """A move rather than a delete: the notice still says which export a record came from."""
        moved, _count = relocate_rights_notices(self.WITH_NOTICE)
        assert "Elsevier" in moved and "All rights reserved" in moved

    def test_the_new_field_matches_the_records_own_indentation(self):
        moved, _count = relocate_rights_notices(self.WITH_NOTICE)
        assert "\n  copyright = {" in moved

    def test_a_record_with_no_notice_is_untouched_byte_for_byte(self):
        source = "@article{b,\n  title = {B},\n  abstract = {No notice here at all.},\n}\n"
        assert relocate_rights_notices(source) == (source, 0)

    def test_a_record_with_no_abstract_is_untouched(self):
        source = "@article{b,\n  title = {B},\n  year = {2024},\n}\n"
        assert relocate_rights_notices(source) == (source, 0)

    def test_an_existing_field_is_appended_to_rather_than_clobbered_or_refused(self):
        """Both notices name a source the record came from, so they are joined.

        Same rule as `repair_duplicate_field_keys` merging repeated fields and `raven.papers.deduplicate`
        unioning this field across a merge — one way of combining, wherever the collision turns up.
        """
        source = ("@article{c,\n  title = {C},\n  abstract = {Body. © 2024 Someone.},\n"
                  "  copyright = {© 2019 Someone Else.},\n}\n")
        moved, count = relocate_rights_notices(source)
        assert count == 1
        fields = {f.key: f.value for f in parse_string(moved, split_names=False).entries[0].fields}
        assert fields["abstract"] == "Body."
        assert fields["copyright"] == "© 2019 Someone Else.\n© 2024 Someone."

    def test_a_notice_already_recorded_there_is_not_moved_twice(self):
        source = ("@article{c,\n  title = {C},\n  abstract = {Body. © 2024 Someone.},\n"
                  "  copyright = {© 2024 Someone.},\n}\n")
        assert relocate_rights_notices(source) == (source, 0)

    def test_the_records_around_it_are_untouched_byte_for_byte(self):
        untouched = "@article{b,\n  title = {Untouched},\n  abstract = {No notice.},\n}"
        source = f"{self.WITH_NOTICE}\n{untouched}\n"
        moved, count = relocate_rights_notices(source)
        assert count == 1
        assert untouched in moved

    def test_a_split_that_would_unbalance_the_braces_is_refused(self):
        """Cheap to check, and the alternative is writing a file that does not parse."""
        source = "@article{d,\n  abstract = {Body {group © 2024 Someone} more},\n}\n"
        moved, count = relocate_rights_notices(source)
        assert count == 0 and moved == source

    def test_the_field_name_is_the_callers_to_choose(self):
        moved, _count = relocate_rights_notices(self.WITH_NOTICE, field="rights")
        fields = {f.key: f.value for f in parse_string(moved, split_names=False).entries[0].fields}
        assert "rights" in fields and "copyright" not in fields

    def test_the_result_parses_and_the_run_is_idempotent(self):
        once, count = relocate_rights_notices(self.WITH_NOTICE)
        assert count == 1
        assert relocate_rights_notices(once) == (once, 0)


class TestFieldSpans:
    def test_the_three_value_shapes_are_all_located(self):
        raw = '@article{k,\n  title = {Braced},\n  journal = "Quoted",\n  year = 2024,\n}\n'
        assert [span[0] for span in _field_spans(raw)] == ["title", "journal", "year"]

    def test_a_value_may_contain_the_separators_that_end_a_bare_one(self):
        raw = "@article{k,\n  title = {Commas, braces {and} more},\n  year = {2024},\n}\n"
        spans = _field_spans(raw)
        assert [span[0] for span in spans] == ["title", "year"]
        assert raw[spans[0][3]:spans[0][4]] == "{Commas, braces {and} more}"

    def test_an_escaped_brace_does_not_close_a_value(self):
        # This is what `repair_record` writes, so the two repairs have to agree about what they read.
        raw = "@article{k,\n  abstract = {We require \\{0 <= rho <= 1 throughout},\n  year = {2024},\n}\n"
        assert [span[0] for span in _field_spans(raw)] == ["abstract", "year"]

    def test_a_value_that_never_ends_scans_to_nothing(self):
        assert _field_spans("@article{k,\n  abstract = {never closes\n") is None

    def test_field_names_are_lowercased_but_their_offsets_point_at_the_original(self):
        raw = "@article{k,\n  Title = {A Study},\n}\n"
        key, name_start, name_end, _value_start, _value_end = _field_spans(raw)[0]
        assert key == "title" and raw[name_start:name_end] == "Title"


class TestUndelimit:
    def test_strips_one_layer_of_either_delimiter(self):
        assert _undelimit("{A Study}") == "A Study"
        assert _undelimit('"A Study"') == "A Study"

    def test_leaves_a_bare_value_alone(self):
        assert _undelimit("2024") == "2024"

    def test_leaves_a_concatenation_alone_although_it_opens_and_closes_like_one_value(self):
        # `{a} # {b}` begins with `{` and ends with `}` and is nonetheless not one braced value; stripping
        # the pair would move the first value's end inwards and corrupt the text.
        assert _undelimit("{a} # {b}") == "{a} # {b}"

    def test_an_escaped_brace_does_not_end_the_value_early(self):
        assert _undelimit("{a \\} b}") == "a \\} b"


class TestRepairDuplicateFieldKeys:
    REPEATED = ("@article{k,\n"
                "  title = {A Study},\n"
                "  annote = {First note},\n"
                "  year = {2024},\n"
                "  annote = {Second note},\n"
                "}\n")

    def test_repeats_are_merged_into_the_first_occurrence(self):
        repaired = repair_duplicate_field_keys(self.REPEATED)
        fields = {f.key: f.value for f in parse_string(repaired).entries[0].fields}
        assert fields["annote"] == "First note\nSecond note"
        assert [f.key for f in parse_string(repaired).entries[0].fields] == ["title", "annote", "year"]

    def test_the_untouched_fields_keep_their_text_byte_for_byte(self):
        repaired = repair_duplicate_field_keys(self.REPEATED)
        assert "  title = {A Study},\n" in repaired
        assert "  year = {2024},\n" in repaired

    def test_removing_a_field_does_not_leave_the_blank_line_it_stood_on(self):
        repaired = repair_duplicate_field_keys(self.REPEATED)
        assert "\n\n" not in repaired

    def test_only_the_named_keys_are_merged_when_the_caller_names_any(self):
        raw = ("@article{k,\n  annote = {a},\n  keywords = {x},\n"
               "  annote = {b},\n  keywords = {y},\n}\n")
        repaired = repair_duplicate_field_keys(raw, {"annote"})
        # `keywords` still repeats, so the result is still unreadable and the repair reports failure
        # rather than handing back a record that only looks mended.
        assert repaired is None

    def test_a_record_naming_nothing_twice_is_not_a_candidate(self):
        assert repair_duplicate_field_keys("@article{k,\n  title = {A Study},\n}\n") is None

    def test_a_repair_that_does_not_read_back_as_an_entry_is_refused(self):
        # Nothing closes the value, so no merge can produce an entry and the oracle says so.
        raw = "@article{k,\n  annote = {a},\n  annote = {unterminated {value,\n"
        assert repair_duplicate_field_keys(raw) is None

    def test_an_unrelated_fault_does_not_veto_a_good_repair(self):
        """The oracle asks whether the *edit* worked, not whether the record is now perfect.

        This record carries two independent faults: repeated fields, and an author BibTeX cannot express
        — three commas where it allows two. Merging the fields is a complete repair of the first, and
        judging it against a parse that also splits names would throw it away for the second. Whether
        the result reads under Raven's full chain is the caller's question; `raven.papers.fixbib` asks
        it, and reports the author.
        """
        raw = self.REPEATED.replace("  year = {2024},\n", "  author = {Bloggs, PhD, MSc, Joan},\n")
        repaired = repair_duplicate_field_keys(raw)
        assert repaired is not None
        assert parse_string(repaired, split_names=False).entries[0]["annote"] == "First note\nSecond note"
        # ...and the second fault is untouched, so the record still does not read with names split.
        assert len(parse_string(repaired).failed_blocks) == 1


# ---------------------------------------------------------------------------
# Reading BibTeX by its surface syntax, for when the parser has refused
# ---------------------------------------------------------------------------

class TestHeaderKey:
    def test_a_header_line_yields_its_key(self):
        assert header_key("@article{WOS:000258806000016,") == "WOS:000258806000016"

    def test_the_key_is_verbatim(self):
        # Unlike `papers.burstbib.get_slug`, which sanitizes for use as a filename.
        assert header_key("@misc{a/b:c,") == "a/b:c"

    def test_a_line_that_is_not_a_header_yields_nothing(self):
        assert header_key("Title = {Something}") == ""
        assert header_key("") == ""


class TestFieldValue:
    def test_a_field_is_found_by_name(self):
        assert field_value("@a{k,\n\tTitle = {Some Paper},\n}", "title") == "Some Paper"

    def test_the_key_case_does_not_matter(self):
        # A Web of Science export capitalizes its keys; the BibTeX literature does not.
        assert field_value("@a{k,\n\ttitle = {Lower},\n}", "Title") == "Lower"

    def test_it_reads_a_record_the_parser_would_refuse(self):
        # The whole point: an unbalanced brace elsewhere aborts a real parse, title and all.
        broken = "@a{k,\n\tAbstract = {System {[production]),\n\tTitle = {Still here},\n}"
        assert field_value(broken, "title") == "Still here"

    def test_an_absent_field_yields_nothing(self):
        assert field_value("@a{k,\n\tYear = {2020},\n}", "title") == ""


class TestUnbalancedFieldNames:
    def test_the_offending_field_is_named(self):
        broken = "@a{k,\n\tAbstract = {oops {,\n\tYear = {2020},\n}"
        assert unbalanced_field_names(broken) == ["Abstract"]

    def test_a_sound_record_names_nothing(self):
        assert unbalanced_field_names("@a{k,\n\tYear = {2020},\n}") == []

    def test_a_multiline_value_is_a_suspect_not_a_verdict(self):
        # An `Affiliation` listing one author per line is unbalanced line by line and perfectly valid.
        multiline = "@a{k,\n\tAffiliation = {First, Somewhere\nSecond, Elsewhere},\n}"
        assert unbalanced_field_names(multiline) == ["Affiliation"]


class TestBraceRepairCandidates:
    """The candidates are guesses; a parser decides. These pin what is proposed, and what never is."""

    def test_a_stray_opener_in_running_text_is_escaped(self):
        broken = "@a{k,\n\tAbstract = {set {0 <= rho for all rho},\n}"
        candidates = brace_repair_candidates(broken)
        assert candidates == ["@a{k,\n\tAbstract = {set \\{0 <= rho for all rho},\n}"]

    def test_a_stray_closer_in_running_text_is_escaped_first(self):
        # A stray literal falls between its value's opening brace and its terminator, so a surplus closer
        # is more likely the earlier candidate — the opposite bias from a surplus opener. Both are offered;
        # the order decides which one the oracle gets to accept.
        broken = "@a{k,\n\tAbstract = {closing } for nothing},\n}"
        assert brace_repair_candidates(broken)[0] == "@a{k,\n\tAbstract = {closing \\} for nothing},\n}"

    def test_a_repair_changes_nothing_but_the_escapes(self):
        broken = "@a{k,\n\tAbstract = {set {0 <= rho},\n}"
        candidate = brace_repair_candidates(broken)[0]
        assert candidate.replace("\\{", "{").replace("\\}", "}") == broken

    def test_a_balanced_record_is_left_alone(self):
        assert brace_repair_candidates("@a{k,\n\tYear = {2020},\n}") == []

    def test_the_structural_braces_are_never_proposed(self):
        # The header's `{`, each `Key = {` opener and the record's own closing `}` hold the record together.
        # A naive unmatched-bracket scan blames the header first, and escaping it destroys the record.
        broken = "@a{k,\n\tAbstract = {oops {,\n\tYear = {2020},\n}"
        for candidate in brace_repair_candidates(broken):
            assert candidate.startswith("@a{k,")
            assert "\\{2020" not in candidate and "Abstract = \\{" not in candidate
            assert candidate.endswith("\n}")

    def test_a_multiline_value_does_not_confuse_the_proposal(self):
        broken = ("@a{k,\n\tAbstract = {first line\nsecond {line\nthird line},\n\tYear = {2020},\n}")
        assert brace_repair_candidates(broken) == [
            "@a{k,\n\tAbstract = {first line\nsecond \\{line\nthird line},\n\tYear = {2020},\n}"]

    def test_a_record_too_tangled_to_guess_at_is_declined(self):
        # Two braces missing among many that legitimately pair up: the combinations explode, and proposing
        # hundreds of guesses is not repair, it is brute force with a parser attached.
        pairs = " ".join("{g%d}" % i for i in range(10))
        broken = "@a{k,\n\tAbstract = {%s {stray {stray},\n}" % pairs
        assert brace_repair_candidates(broken, max_candidates=10) == []

    def test_the_surplus_count_sets_how_many_braces_a_candidate_escapes(self):
        broken = "@a{k,\n\tAbstract = {a {b {c},\n}"  # two openers short of balance
        assert all(candidate.count("\\{") == 2 for candidate in brace_repair_candidates(broken))
