"""Tests for the multi-database bibliography deduplicator.

The load-bearing half of this file is the set that tries to make the tool merge two *different* papers.
A cluster count says how much a rule caught and nothing about what else it caught, and both real defects
found while building this — a serial's recurring section heading merged across four issues, and two
unrelated editorials merged on the word `Editorial` — were invisible in statistics that looked correct.
"""

import pytest

from raven.papers import deduplicate as dd


def records(source: str):
    """The `Record`s of a BibTeX string, as the tool reads them."""
    parsed, _unreadable = dd.read_records(source)
    return parsed


def entry(key: str, **fields) -> str:
    """One BibTeX record, written the way a database export writes them."""
    body = "".join(f"  {name} = {{{value}}},\n" for name, value in fields.items())
    return f"@article{{{key},\n{body}}}\n"


def clusters_of(source: str, **kwargs):
    """Clusters of a BibTeX string, keyed by the set of BibTeX keys in each."""
    parsed = records(source)
    return [frozenset(record.key for record in cluster.records)
            for cluster in dd.cluster_records(parsed, **kwargs)]


class TestNormalizeDoi:
    def test_case_is_folded(self):
        assert dd.normalize_doi("10.1234/ABC-def") == "10.1234/abc-def"

    @pytest.mark.parametrize("prefix", ["https://doi.org/", "http://doi.org/", "https://dx.doi.org/",
                                        "http://dx.doi.org/", "doi:", "doi: ", "info:doi/"])
    def test_resolver_prefixes_are_stripped(self, prefix):
        assert dd.normalize_doi(f"{prefix}10.1234/abc") == "10.1234/abc"

    @pytest.mark.parametrize("dash", ["‐", "‑", "‒", "–", "—", "―", "−"])
    def test_every_unicode_dash_folds_to_ascii(self, dash):
        """Two databases exporting one DOI disagree about which dash it contains; the paper is one paper."""
        assert dd.normalize_doi(f"10.1234/abc{dash}def") == "10.1234/abc-def"

    def test_a_line_wrapped_doi_loses_its_whitespace(self):
        assert dd.normalize_doi("10.1234/abc\n  def") == "10.1234/abcdef"

    def test_trailing_sentence_punctuation_is_dropped(self):
        assert dd.normalize_doi("10.1234/abc.") == "10.1234/abc"

    def test_enclosing_braces_are_dropped(self):
        assert dd.normalize_doi("{10.1234/abc}") == "10.1234/abc"

    @pytest.mark.parametrize("value", ["", None, "   ", "n/a", "N/A", "not available",
                                       "https://example.com/article/123", "10.1234", "10.1234/",
                                       "10.12/x", "doi", "-"])
    def test_what_is_not_a_doi_is_refused(self, value):
        """A `doi` field regularly holds something that is not one, and those must not become a key.

        Two records both saying `n/a` are equal to each other, so admitting the value would merge papers
        with nothing whatsoever in common — the worst failure this tool has, since a merged record is
        gone from the review and nothing downstream can notice.
        """
        assert dd.normalize_doi(value) is None

    def test_a_suffix_full_of_punctuation_is_still_a_doi(self):
        # Real DOIs carry slashes, parentheses and dots in the suffix; the shape check must not be
        # so tight that it starts rejecting the thing it is checking for.
        assert dd.normalize_doi("10.1002/(SICI)1097-0258(19980815)17:15<1661::AID-SIM968>3.0.CO;2-2") \
            == "10.1002/(sici)1097-0258(19980815)17:15<1661::aid-sim968>3.0.co;2-2"


class TestNormalizeTitle:
    def test_case_spacing_and_punctuation_all_go(self):
        assert dd.normalize_title("Peer-Reviewed AI: A Study") == dd.normalize_title("peer reviewed ai - a study")

    def test_latex_grouping_braces_do_not_matter(self):
        assert dd.normalize_title("{AI} in {Education}") == dd.normalize_title("AI in Education")

    def test_a_unicode_accent_and_its_tex_spelling_agree(self):
        assert dd.normalize_title("Häkkinen") == dd.normalize_title(r"H{\"a}kkinen") == "hakkinen"

    def test_html_entities_resolve_before_reduction(self):
        # One exporter escapes the ampersand and another does not. Same paper.
        assert dd.normalize_title("Q&amp;A Generation") == dd.normalize_title("Q&A Generation")

    def test_a_numeric_entity_resolves_too(self):
        assert dd.normalize_title("Students&#8217; views") == dd.normalize_title("Students’ views")

    def test_markup_is_dropped_rather_than_reduced_to_letters(self):
        """`<i>` must not contribute an `i`, which is what makes an italicized word match a plain one."""
        assert dd.normalize_title("<i>Plasmodium</i> in schools") == dd.normalize_title("Plasmodium in schools")

    def test_ligatures_and_full_width_forms_are_flattened(self):
        assert dd.normalize_title("ﬁnding") == dd.normalize_title("finding")

    @pytest.mark.parametrize("value", ["", None, "   ", "---", "?!"])
    def test_a_title_with_no_letters_or_digits_is_no_title(self, value):
        assert dd.normalize_title(value) is None


class TestIsGenericTitle:
    def test_a_genre_label_is_generic(self):
        assert dd.is_generic_title(dd.normalize_title("Editorial"))
        assert dd.is_generic_title(dd.normalize_title("Book Review"))

    def test_a_real_title_is_not(self):
        assert not dd.is_generic_title(dd.normalize_title("AI in Education"))

    def test_a_short_but_distinctive_title_is_not(self):
        """The case a length threshold got wrong, which is why there is no length threshold.

        `Reportronic` is eleven characters and names exactly one thing; `Generative AI` is thirteen and
        could head any number of unrelated editorials. Length does not measure distinctiveness.
        """
        assert not dd.is_generic_title(dd.normalize_title("Reportronic"))

    def test_absence_is_not_generic(self):
        assert not dd.is_generic_title(None)


class TestTitleEdge:
    """Whether an equal title is allowed to merge two records — the guard against a false merge."""

    def _pair(self, a_fields, b_fields, title="A Study of Machine Learning in Higher Education"):
        source = entry("a", title=title, **a_fields) + entry("b", title=title, **b_fields)
        first, second = records(source)
        return first, second

    def test_the_same_paper_from_two_databases_is_joined(self):
        a, b = self._pair({"author": "Smith, Jane", "year": "2024"},
                          {"author": "Smith, Jane and Jones, Bob", "year": "2024"})
        assert dd._title_edge_holds(a, b)

    def test_a_different_author_alone_does_not_refuse_it(self):
        """Databases disagree about author order, so a surname mismatch alone is weak evidence."""
        a, b = self._pair({"author": "Smith, Jane and Jones, Bob", "year": "2024"},
                          {"author": "Jones, Bob and Smith, Jane", "year": "2024"})
        assert dd._disagree_on_author(a, b), "the fixture must actually disagree, or this proves nothing"
        assert dd._title_edge_holds(a, b)

    def test_a_year_apart_alone_does_not_refuse_it(self):
        """A preprint and its published version straddle a New Year all the time."""
        a, b = self._pair({"author": "Smith, Jane", "year": "2023"},
                          {"author": "Smith, Jane", "year": "2024"})
        assert dd._title_edge_holds(a, b)

    def test_a_different_author_and_a_year_gap_together_refuse_it(self):
        a, b = self._pair({"author": "Smith, Jane", "year": "2020"},
                          {"author": "Jones, Bob", "year": "2024"})
        assert not dd._title_edge_holds(a, b)

    def test_a_missing_author_is_not_a_disagreement(self):
        a, b = self._pair({"year": "2020"}, {"author": "Jones, Bob", "year": "2024"})
        assert dd._title_edge_holds(a, b)

    def test_a_generic_title_needs_positive_agreement(self):
        a, b = self._pair({"author": "Smith, Jane", "year": "2024"},
                          {"author": "Smith, Jane", "year": "2024"}, title="Editorial")
        assert dd._title_edge_holds(a, b)

    def test_a_generic_title_with_different_authors_is_refused(self):
        a, b = self._pair({"author": "Smith, Jane", "year": "2024"},
                          {"author": "Jones, Bob", "year": "2024"}, title="Editorial")
        assert not dd._title_edge_holds(a, b)

    def test_two_reviews_by_one_person_in_one_year_are_not_one_review(self):
        """Author and year agreement is not enough for a genre label — the same person writes several.

        Every one of them is titled `Book Review`, so what separates them is which item they are: a DOI,
        a page range, an issue. The corpus contains no such pair, which is exactly why a merge count
        could not have found this.
        """
        a, b = self._pair({"author": "Smith, Jane", "year": "2024", "pages": "101--103"},
                          {"author": "Smith, Jane", "year": "2024", "pages": "204--206"},
                          title="Book Review")
        assert not dd._title_edge_holds(a, b)

    def test_one_review_exported_twice_still_merges(self):
        """The negative control for the rule above: without it, refusing everything would also pass.

        Page ranges written with an en-dash and a double hyphen are one range, so the reduction has to see
        past that or every genuine pair is refused too.
        """
        a, b = self._pair({"author": "Smith, Jane", "year": "2024", "doi": "10.1234/x", "pages": "101--103"},
                          {"author": "Smith, Jane", "year": "2024", "doi": "10.1234/x", "pages": "101-103"},
                          title="Book Review")
        assert dd._title_edge_holds(a, b)

    def test_a_generic_title_with_no_author_is_refused(self):
        """Silence is not agreement: two authorless `Editorial`s are not one editorial."""
        a, b = self._pair({"year": "2024"}, {"year": "2024"}, title="Editorial")
        assert not dd._title_edge_holds(a, b)

    def test_two_authorless_records_disagreeing_about_the_doi_are_refused(self):
        """A serial's recurring section heading, which is an ordinary title by every other test."""
        a, b = self._pair({"year": "2024", "doi": "10.1177/00208345241262094"},
                          {"year": "2024", "doi": "10.1177/00208345241232770"},
                          title="II Political Science: Method and Theory")
        assert not dd._title_edge_holds(a, b)

    def test_two_authorless_records_with_nothing_to_contradict_them_are_joined(self):
        # A proceedings volume exported twice, no DOI either time. Nothing contradicts the title.
        a, b = self._pair({"year": "2024"}, {"year": "2024"},
                          title="25th International Conference on AI in Education, AIED 2024")
        assert dd._title_edge_holds(a, b)

    def test_two_volumes_of_one_proceedings_are_not_one_book(self):
        """Parts I and II share a title, carry no authors and often carry no DOIs — only `volume` differs.

        Merging them drops a whole volume out of the bibliography, and no DOI check can see it. Five such
        pairs were merging in the corpus this was built against.
        """
        a, b = self._pair({"year": "2022", "volume": "13355 LNCS"},
                          {"year": "2022", "volume": "13356 LNCS"},
                          title="23rd International Conference on AI in Education, AIED 2022")
        assert not dd._title_edge_holds(a, b)

    def test_an_authored_record_disagreeing_about_the_doi_is_still_joined(self):
        """The rule is about *authorless* records. A preprint and its published version differ by DOI."""
        a, b = self._pair({"author": "Smith, Jane", "year": "2024", "doi": "10.48550/arXiv.2401.00001"},
                          {"author": "Smith, Jane", "year": "2024", "doi": "10.1007/s10639-024-12764-2"})
        assert dd._title_edge_holds(a, b)


class TestClusterRecords:
    def test_an_equal_doi_merges(self):
        assert clusters_of(entry("a", title="One thing", doi="10.1234/x")
                           + entry("b", title="Another thing entirely", doi="10.1234/x")) == [{"a", "b"}]

    def test_an_equal_title_merges(self):
        assert clusters_of(entry("a", title="A Study of Learning Analytics", year="2024")
                           + entry("b", title="a study of learning analytics", year="2024")) == [{"a", "b"}]

    def test_matching_is_transitive_across_the_two_keys(self):
        """The property that makes DOI and title complementary rather than two separate passes."""
        source = (entry("a", title="A Study of Learning Analytics", doi="10.1234/x", year="2024")
                  + entry("b", title="Quite Different Wording Here", doi="10.1234/x", year="2024")
                  + entry("c", title="a study of learning analytics", year="2024"))
        assert clusters_of(source) == [{"a", "b", "c"}]

    def test_unrelated_records_stay_apart(self):
        source = (entry("a", title="A Study of Learning Analytics", doi="10.1234/x")
                  + entry("b", title="Something Else Entirely", doi="10.1/y"))
        assert clusters_of(source) == [{"a"}, {"b"}]

    def test_clusters_come_back_in_input_order(self):
        source = (entry("a", title="Alpha and its consequences")
                  + entry("b", title="Beta and its consequences")
                  + entry("c", title="alpha and its consequences"))
        assert clusters_of(source) == [{"a", "c"}, {"b"}]

    def test_which_keys_matched_is_recorded(self):
        parsed = records(entry("a", title="A Study of Learning Analytics", doi="10.1234/x", year="2024")
                         + entry("b", title="a study of learning analytics", doi="10.1234/x", year="2024"))
        assert dd.cluster_records(parsed)[0].rules == ("doi", "title")

    def test_two_editorials_by_different_people_do_not_merge(self):
        """The first false merge this tool produced: `Editorial` joined a 2022 paper to a 2024 one.

        The 2022 pair must still merge, which is what makes this a test of the guard rather than of
        refusing everything — and it is why the rule is applied per pair rather than per title.
        """
        source = (entry("bandyopadhyay_2022", title="Editorial", author="Bandyopadhyay, A.",
                        year="2022", doi="10.1177/26314541221107222")
                  + entry("bandyopadhyay_2022_dup", title="Editorial", author="Bandyopadhyay, A.",
                          year="2022", doi="10.1177/26314541221107222")
                  + entry("mcnally_2024", title="Editorial", author="McNally, S.",
                          year="2024", doi="10.1177/14697874241293205"))

        parsed = records(source)
        assert len({record.title for record in parsed}) == 1, \
            "all three must share a normalized title, or this fixture cannot exercise the guard at all"

        assert clusters_of(source) == [{"bandyopadhyay_2022", "bandyopadhyay_2022_dup"}, {"mcnally_2024"}]

    def test_a_serials_recurring_heading_does_not_merge_its_issues(self):
        """The second false merge: four issues of one journal, joined by the heading they all carry.

        Authorless, ordinary-looking title, same year, different DOIs — every signal but the DOI says
        these are one item. The pairs that share a DOI are the same item exported twice and must survive.
        """
        heading = "II Political Science: Method and Theory"
        source = (entry("ii_a", title=heading, year="2024", doi="10.1177/00208345241262094")
                  + entry("ii_b", title=heading, year="2024", doi="10.1177/00208345241232770")
                  + entry("ii_c", title=heading, year="2024", doi="10.1177/00208345241262094")
                  + entry("ii_d", title=heading, year="2024", doi="10.1177/00208345241232770"))

        parsed = records(source)
        assert len({record.title for record in parsed}) == 1, \
            "all four must share a normalized title, or the title rule is not what is being tested"
        assert not any(record.surname for record in parsed), \
            "the fixture must be authorless, since that is the branch under test"

        assert clusters_of(source) == [{"ii_a", "ii_c"}, {"ii_b", "ii_d"}]

    def test_a_judged_pair_is_merged(self):
        source = (entry("a", title="Not a Swiss Army Knife: Perceptions of Trade-Offs", year="2024")
                  + entry("b", title="Not a Swiss Army Knife: Perceptions of Trade Offs Around AI", year="2025"))
        assert clusters_of(source) == [{"a"}, {"b"}]
        assert clusters_of(source, maybe_judgements={(0, 1): True}) == [{"a", "b"}]

    def test_a_judged_refusal_withdraws_a_title_edge(self):
        source = (entry("a", title="A Study of Learning Analytics", year="2024")
                  + entry("b", title="A study of learning analytics", year="2024"))
        assert clusters_of(source) == [{"a", "b"}]
        assert clusters_of(source, maybe_judgements={(0, 1): False}) == [{"a"}, {"b"}]

    def test_a_judged_refusal_does_not_withdraw_a_doi_edge(self):
        """DOI equality is conclusive, and the judge is not asked to overrule it."""
        source = (entry("a", title="One phrasing of the title", doi="10.1234/x")
                  + entry("b", title="Another phrasing entirely", doi="10.1234/x"))
        assert clusters_of(source, maybe_judgements={(0, 1): False}) == [{"a", "b"}]


class TestMergeCluster:
    def test_a_singleton_is_returned_unchanged_and_unaudited(self):
        parsed = records(entry("a", title="A Study of Learning Analytics"))
        merged, row = dd.merge_cluster(dd.cluster_records(parsed)[0])
        assert merged is parsed[0].entry
        assert row is None

    def test_fields_are_filled_in_from_a_twin(self):
        source = (entry("rich", title="A Study of Learning Analytics", author="Smith, Jane",
                        year="2024", doi="10.1234/x", journal="Journal of Things")
                  + entry("poor", title="A Study of Learning Analytics", year="2024", doi="10.1234/x",
                          keywords="learning analytics"))
        merged, _row = dd.merge_cluster(dd.cluster_records(records(source))[0])
        fields = {field.key: field.value for field in merged.fields}
        assert fields["keywords"] == "learning analytics"       # filled in from the poorer record
        assert fields["journal"] == "Journal of Things"         # and the base kept its own

    def test_the_base_keeps_its_own_value_where_both_have_one(self):
        source = (entry("rich", title="A Study of Learning Analytics", author="Smith, Jane",
                        year="2024", doi="10.1234/x", journal="Journal of Things")
                  + entry("poor", title="A Study of Learning Analytics", year="2024", doi="10.1234/x",
                          journal="Some Other Journal"))
        merged, row = dd.merge_cluster(dd.cluster_records(records(source))[0])
        fields = {field.key: field.value for field in merged.fields}
        assert fields["journal"] == "Journal of Things"
        assert any("journal: kept" in difference for difference in row.differences)

    def test_the_richest_record_becomes_the_base(self):
        source = (entry("thin", title="A Study of Learning Analytics", doi="10.1234/x")
                  + entry("fat", title="A Study of Learning Analytics", doi="10.1234/x",
                          author="Smith, Jane", year="2024", journal="J", publisher="P"))
        merged, row = dd.merge_cluster(dd.cluster_records(records(source))[0])
        assert merged.key == "fat"
        assert row.kept == "fat"

    def test_the_published_version_beats_the_preprint_however_rich_the_preprint_is(self):
        """Publication status decides the base, not field count.

        Ranking by field count first usually picks the same record and sometimes picks the deposit,
        which leaves an arXiv DOI on a paper that has a journal one — wrong in a bibliography, and
        invisible once written.
        """
        source = (entry("published", title="A Study of Learning Analytics", author="Smith, Jane",
                        doi="10.1007/s10639-024-12764-2", year="2024")
                  + entry("preprint", title="A Study of Learning Analytics", author="Smith, Jane",
                          doi="10.48550/arXiv.2401.00001", year="2024", journal="J", publisher="P",
                          abstract="Longer.", keywords="k", note="n"))

        parsed = records(source)
        assert len(parsed[1].entry.fields) > len(parsed[0].entry.fields), \
            "the preprint must be the richer record, or this fixture cannot tell the two rules apart"

        merged, _row = dd.merge_cluster(dd.cluster_records(parsed)[0])
        assert merged.key == "published"
        assert dict((f.key, f.value) for f in merged.fields)["doi"] == "10.1007/s10639-024-12764-2"

    def test_the_highest_springer_chapter_version_becomes_the_base(self):
        source = (entry("v1", title="Use of AI in Singapore Education: Overview", author="Zhao, L.",
                        year="2026", doi="10.1007/978-981-96-3901-4_5-1")
                  + entry("v2", title="Use of AI in Singapore Education: Overview", author="Zhao, L.",
                          year="2026", doi="10.1007/978-981-96-3901-4_5-2"))
        merged, _row = dd.merge_cluster(dd.cluster_records(records(source))[0])
        assert merged.key == "v2"

    def test_the_merged_entry_shares_no_field_objects_with_its_inputs(self):
        # A caller writes the result out; that must not have disturbed the records it drew from.
        source = (entry("a", title="A Study of Learning Analytics", doi="10.1234/x", author="Smith, Jane")
                  + entry("b", title="A Study of Learning Analytics", doi="10.1234/x", keywords="k"))
        parsed = records(source)
        merged, _row = dd.merge_cluster(dd.cluster_records(parsed)[0])
        originals = {id(field) for record in parsed for field in record.entry.fields}
        assert not originals & {id(field) for field in merged.fields}

    def test_field_order_follows_the_base_with_the_filled_in_ones_after(self):
        source = (entry("a", title="A Study of Learning Analytics", author="Smith, Jane", doi="10.1234/x")
                  + entry("b", title="A Study of Learning Analytics", doi="10.1234/x", keywords="k"))
        merged, _row = dd.merge_cluster(dd.cluster_records(records(source))[0])
        assert [field.key for field in merged.fields] == ["title", "author", "doi", "keywords"]


class TestMergedAbstracts:
    NOTICE = " © 2024 Elsevier Ltd. All rights reserved."

    def test_the_publishers_notice_decides_nothing_and_survives_on_the_abstract_it_came_with(self):
        """Stripping chooses *which* abstract; it does not edit the one that wins.

        Writing the stripped text would leave the output carrying two kinds of abstract — trimmed where a
        record happened to have a twin, untouched where it did not — and would make the tool a content
        editor, which is what keeps the audit able to account for the whole input-to-output difference.
        """
        source = (entry("a", title="A Study of Learning Analytics", doi="10.1234/x",
                        abstract="We study things." + self.NOTICE)
                  + entry("b", title="A Study of Learning Analytics", doi="10.1234/x", keywords="k"))
        merged, _row = dd.merge_cluster(dd.cluster_records(records(source))[0])
        assert dict((f.key, f.value) for f in merged.fields)["abstract"] \
            == "We study things." + self.NOTICE

    def test_the_longest_abstract_wins_after_stripping_and_not_before(self):
        """Stripping first is what makes "longest" the right rule rather than a coin toss.

        On raw text the longest copy is usually just the one carrying the most boilerplate, so the rule
        would graft a publisher's copyright line onto the majority of merged records.
        """
        short_with_notice = "We study things." + self.NOTICE
        long_without = "We study things thoroughly and at some length."
        assert len(short_with_notice) > len(long_without), \
            "the boilerplate must make the shorter abstract the longer string, or nothing is being tested"

        source = (entry("a", title="A Study of Learning Analytics", doi="10.1234/x",
                        abstract=short_with_notice)
                  + entry("b", title="A Study of Learning Analytics", doi="10.1234/x",
                          abstract=long_without))
        merged, _row = dd.merge_cluster(dd.cluster_records(records(source))[0])
        assert dict((f.key, f.value) for f in merged.fields)["abstract"] == long_without

    def test_a_truncated_copy_loses_to_the_full_one_even_from_the_base(self):
        """The one field the base does not automatically keep: being the fullest record does not make
        its abstract the least truncated one."""
        source = (entry("base", title="A Study of Learning Analytics", doi="10.1234/x", author="Smith, Jane",
                        year="2024", journal="J", abstract="We study thi")
                  + entry("other", title="A Study of Learning Analytics", doi="10.1234/x",
                          abstract="We study things thoroughly and report what we found."))
        merged, _row = dd.merge_cluster(dd.cluster_records(records(source))[0])
        assert dict((f.key, f.value) for f in merged.fields)["abstract"].endswith("what we found.")

    def test_a_losing_abstract_that_said_something_different_is_audited(self):
        source = (entry("a", title="A Study of Learning Analytics", doi="10.1234/x",
                        abstract="We study things thoroughly and at length.")
                  + entry("b", title="A Study of Learning Analytics", doi="10.1234/x",
                          abstract="An entirely different summary."))
        _merged, row = dd.merge_cluster(dd.cluster_records(records(source))[0])
        assert any("abstract: kept" in difference for difference in row.differences)

    def test_two_copies_agreeing_after_stripping_leave_nothing_to_audit(self):
        source = (entry("a", title="A Study of Learning Analytics", doi="10.1234/x",
                        abstract="We study things." + self.NOTICE)
                  + entry("b", title="A Study of Learning Analytics", doi="10.1234/x",
                          abstract="We study things."))
        _merged, row = dd.merge_cluster(dd.cluster_records(records(source))[0])
        assert not any("abstract" in difference for difference in row.differences)


class TestMergedRightsNotices:
    """`copyright` is unioned across a cluster, not chosen — the plural case is the point.

    A merged record genuinely came from several exports, and each notice names one of them. Since nobody
    redistributes a bibliography pulled out of a paywalled aggregator, saying *which of your own exports*
    a record came from is most of what the notice is worth here, and picking one would delete it.
    """

    def _merged(self, source):
        merged, row = dd.merge_cluster(dd.cluster_records(records(source))[0])
        return {field.key: field.value for field in merged.fields}, row

    def test_two_exports_notices_both_survive_the_merge(self):
        source = (entry("scopus", title="A Study of Learning Analytics", doi="10.1234/x",
                        copyright="© 2024 Elsevier Ltd.")
                  + entry("springer", title="A study of learning analytics", doi="10.1234/x",
                          copyright="© The Author(s) 2024, Springer Nature."))
        fields, _row = self._merged(source)
        assert fields["copyright"] == "© 2024 Elsevier Ltd.\n© The Author(s) 2024, Springer Nature."

    def test_the_same_notice_twice_is_kept_once(self):
        source = (entry("a", title="A Study of Learning Analytics", doi="10.1234/x",
                        copyright="© 2024 Elsevier Ltd.")
                  + entry("b", title="A study of learning analytics", doi="10.1234/x",
                          copyright="© 2024 Elsevier Ltd."))
        fields, _row = self._merged(source)
        assert fields["copyright"] == "© 2024 Elsevier Ltd."

    def test_a_union_is_not_reported_as_a_dropped_value(self):
        """Nothing was dropped, so nothing belongs in the audit — the row would be noise."""
        source = (entry("a", title="A Study of Learning Analytics", doi="10.1234/x",
                        copyright="© 2024 Elsevier Ltd.")
                  + entry("b", title="A study of learning analytics", doi="10.1234/x",
                          copyright="© The Author(s) 2024."))
        _fields, row = self._merged(source)
        assert not any("copyright" in difference for difference in row.differences)

    def test_a_record_without_one_contributes_nothing(self):
        source = (entry("a", title="A Study of Learning Analytics", doi="10.1234/x",
                        copyright="© 2024 Elsevier Ltd.")
                  + entry("b", title="A study of learning analytics", doi="10.1234/x", keywords="k"))
        fields, _row = self._merged(source)
        assert fields["copyright"] == "© 2024 Elsevier Ltd."

    def test_a_cluster_with_no_notices_gains_no_field(self):
        source = (entry("a", title="A Study of Learning Analytics", doi="10.1234/x")
                  + entry("b", title="A study of learning analytics", doi="10.1234/x", keywords="k"))
        fields, _row = self._merged(source)
        assert "copyright" not in fields


class TestAudit:
    SOURCE = (entry("a", title="A Study of Learning Analytics", doi="10.1234/x", author="Smith, Jane",
                    year="2024")
              + entry("b", title="A Study of Learning Analytics", doi="10.48550/arXiv.2401.00001",
                      author="Smith, Jane", year="2024", keywords="analytics"))

    def _row(self):
        _merged, row = dd.merge_cluster(dd.cluster_records(records(self.SOURCE))[0])
        return row

    def test_the_row_names_what_was_kept_and_what_went(self):
        row = self._row()
        assert row.kept == "a"
        assert row.removed == ("b",)
        assert row.size == 2

    def test_every_distinct_doi_in_the_cluster_is_listed(self):
        assert self._row().dois == ("10.1234/x", "10.48550/arxiv.2401.00001")

    def test_cells_carry_no_tab_or_newline(self):
        """Either would end a cell or a row, and the differences column holds arbitrary field text."""
        source = self.SOURCE + entry("c", title="A Study of Learning Analytics", doi="10.1234/x",
                                     note="a note\twith a tab\nand a newline")
        _merged, row = dd.merge_cluster(dd.cluster_records(records(source))[0])
        assert all("\t" not in cell and "\n" not in cell and "\r" not in cell for cell in row.to_row())

    def test_a_row_has_one_cell_per_column(self):
        assert len(self._row().to_row()) == len(dd.AUDIT_COLUMNS)

    def test_a_long_value_is_clipped_rather_than_carried_whole(self):
        source = (entry("a", title="A Study of Learning Analytics", doi="10.1234/x", note="short")
                  + entry("b", title="A Study of Learning Analytics", doi="10.1234/x", note="x" * 5000))
        _merged, row = dd.merge_cluster(dd.cluster_records(records(source))[0])
        assert all(len(cell) < 2000 for cell in row.to_row())

    def test_the_file_is_stamped_with_the_tool_version(self, tmp_path):
        """What makes the audit citable: a method section names a versioned tool, not somebody's script."""
        from raven import __version__
        path = tmp_path / "audit.tsv"
        dd.write_audit(path, [self._row()], ["search.bib"])
        assert path.read_text(encoding="utf-8").splitlines()[0] == f"# raven-deduplicate {__version__}"

    def test_the_header_names_every_column(self, tmp_path):
        path = tmp_path / "audit.tsv"
        dd.write_audit(path, [self._row()], ["search.bib"])
        header = next(line for line in path.read_text(encoding="utf-8").splitlines()
                      if not line.startswith("#"))
        assert header.split("\t") == list(dd.AUDIT_COLUMNS)


class TestNothingDisappears:
    def test_every_value_is_either_kept_or_recorded(self):
        """The tool's actual promise, checked exhaustively on a cluster built to violate it.

        Every record carries a field the others do not, and two of them disagree about a field they
        share. Nothing may fall between the merged entry and the audit row.
        """
        source = (entry("a", title="A Study of Learning Analytics", doi="10.1234/x", author="Smith, Jane",
                        year="2024", journal="Journal A", abstract="A thorough summary of the work.")
                  + entry("b", title="A study of learning analytics", doi="10.1234/x", year="2024",
                          journal="Journal B", keywords="analytics", abstract="Short.")
                  + entry("c", title="A Study of Learning Analytics", doi="10.1234/x", year="2023",
                          publisher="Elsevier", note="a note"))

        cluster = dd.cluster_records(records(source))[0]
        assert len(cluster) == 3, "all three must cluster, or most of this fixture is not exercised"
        merged, row = dd.merge_cluster(cluster)

        kept = {field.key: field.value for field in merged.fields}
        recorded = " | ".join(row.differences)
        for record in cluster.records:
            for field in record.entry.fields:
                value = dd._field_value(record.entry, field.key)
                if value is None or kept.get(field.key) == value:
                    continue
                assert f"{field.key}: kept" in recorded, \
                    f"{record.key}'s {field.key} was neither kept nor recorded: {value!r}"

    def test_the_one_exemption_is_an_abstract_differing_only_by_its_rights_notice(self):
        """Stated as a test because it is the single hole in the promise above, and it is deliberate.

        A database's own notice is usually the only thing separating its copy of an abstract from
        another's, so reporting each one would bury the differences that are about the paper under
        hundreds that are about the exporter. Nothing else is compared this way.
        """
        source = (entry("a", title="A Study of Learning Analytics", doi="10.1234/x",
                        abstract="We study things. © 2024 Elsevier Ltd. All rights reserved.")
                  + entry("b", title="A Study of Learning Analytics", doi="10.1234/x",
                          abstract="We study things. © 2024 Springer Nature."))

        _merged, row = dd.merge_cluster(dd.cluster_records(records(source))[0])
        assert not any("abstract" in difference for difference in row.differences)

        # The negative control: change what the abstracts actually *say*, and it is reported again.
        louder = source.replace("We study things. © 2024 Springer Nature.",
                                "We study something else entirely. © 2024 Springer Nature.")
        _merged, row = dd.merge_cluster(dd.cluster_records(records(louder))[0])
        assert any("abstract: kept" in difference for difference in row.differences)


class TestReadRecords:
    def test_a_record_naming_a_field_twice_is_read(self):
        """Read through the repair: a database export's repeated `annote` would otherwise take the
        whole record with it, title, authors and all."""
        source = ("@article{k,\n  title = {A Study},\n  annote = {First note},\n"
                  "  annote = {Second note},\n  year = {2024},\n}\n")
        parsed, unreadable = dd.read_records(source)
        assert [record.key for record in parsed] == ["k"]
        assert unreadable == []
        assert parsed[0].field("annote") == "First note\nSecond note"

    def test_a_record_whose_author_bibtex_cannot_express_is_still_read(self):
        """Reading without name splitting is what buys this, and the author is only ever copied.

        `raven-fixbib` reports such a record, correctly — its job is to make a file readable by Raven's
        standard chain, and this one still is not. Deduplicating it needs no name parsing at all.
        """
        source = entry("k", title="A Study of Learning Analytics", author="Bloggs, PhD, MSc, Joan",
                       year="2024")
        parsed, unreadable = dd.read_records(source)
        assert [record.key for record in parsed] == ["k"]
        assert unreadable == []
        assert parsed[0].field("author") == "Bloggs, PhD, MSc, Joan"

    def test_a_genuinely_unreadable_record_is_reported_and_not_counted(self):
        source = entry("fine", title="A Study") + "@article{broken,\n  title = {Unterminated {value,\n"
        parsed, unreadable = dd.read_records(source)
        assert [record.key for record in parsed] == ["fine"]
        assert len(unreadable) == 1

    def test_an_arxiv_record_with_no_doi_gets_the_registered_one(self):
        """What lets a preprint match its published twin, whose `doi` field says the same thing."""
        source = entry("k", title="A Study", eprint="2401.00001", archiveprefix="arXiv")
        assert records(source)[0].doi == "10.48550/arxiv.2401.00001"

    def _surname(self, author):
        return records(entry("k", title="T", author=author))[0].surname

    def test_a_particle_blocks_the_same_in_both_bibtex_orders(self):
        """One person written two ways must land in one blocking key, or the fuzzy pass cannot see them.

        The particle belongs to the surname, and it is the whole reason this asks `bibtexparser` rather
        than splitting on the last space.
        """
        assert self._surname("van Beethoven, Ludwig") == self._surname("Ludwig van Beethoven") == "vanbeethoven"

    def test_a_comma_less_name_follows_bibtex_and_that_is_sometimes_a_miss(self):
        """`A B C` with no comma is ambiguous, and BibTeX resolves it by rule: the last token is it.

        Right for the common case — two given names and a surname — and wrong for a compound surname,
        which is what the comma form exists to say. Pinned because it is a *documented* blocking miss
        rather than a defect: guessing against the format would be worse, and the cost is one comparison
        the fuzzy pass does not get to make.
        """
        assert self._surname("Petra Johanna Lagerkvist") == self._surname("Lagerkvist, Petra Johanna")
        assert self._surname("Aksel Holm Dahl") == "dahl"        # BibTeX's reading, and it is the format's
        assert self._surname("Holm Dahl, Aksel") == "holmdahl"   # ...so the two spellings block apart

    def test_a_suffix_does_not_land_in_the_surname(self):
        assert self._surname("Fenwick, Jr., A. B.") == "fenwick"

    def test_an_author_bibtex_cannot_express_still_blocks(self):
        """Three commas where the format allows two. The record is still a paper, and dropping it from
        the fuzzy pass silently would be worse than reading up to the first comma."""
        assert self._surname("Bloggs, PhD, MSc, Joan") == "bloggs"

    @pytest.mark.parametrize("raw,expected", [("2024", 2024), ("2024-06", 2024), ("c2024", 2024),
                                              ("", None), ("in press", None)])
    def test_the_year_is_read_out_of_what_exporters_write(self, raw, expected):
        assert records(entry("k", title="T", year=raw))[0].year == expected


class TestFuzzyCandidates:
    def test_a_near_miss_the_exact_keys_missed_is_offered(self):
        source = (entry("a", title="A Large-Scale Real-World Evaluation of LLM-Based Virtual Teaching Assistants",
                        author="Kweon, S.", year="2025")
                  + entry("b", title="A Large-Scale Real-World Evaluation of an LLM-Based Virtual Teaching Assistant",
                          author="Kweon, S.", year="2025", doi="10.1234/x"))
        parsed = records(source)
        pairs = dd.fuzzy_candidates(parsed, dd.cluster_records(parsed))
        assert [(a.key, b.key) for a, b in pairs] == [("a", "b")]

    def test_records_already_in_one_cluster_are_not_offered(self):
        source = (entry("a", title="A Study of Learning Analytics", author="Smith, Jane", year="2024")
                  + entry("b", title="A study of learning analytics", author="Smith, Jane", year="2024"))
        parsed = records(source)
        assert dd.fuzzy_candidates(parsed, dd.cluster_records(parsed)) == []

    def test_two_different_papers_by_one_author_are_not_offered(self):
        source = (entry("a", title="Machine Learning for Chemistry", author="Smith, Jane", year="2024")
                  + entry("b", title="Deep Reinforcement Learning in Robotics", author="Smith, Jane",
                          year="2024"))
        parsed = records(source)
        assert dd.fuzzy_candidates(parsed, dd.cluster_records(parsed)) == []

    def test_blocking_needs_an_author_and_a_year(self):
        """What the blocking cannot see, stated as a test so it is a known cost rather than a surprise."""
        source = (entry("a", title="A Large-Scale Evaluation of Virtual Teaching Assistants", year="2025")
                  + entry("b", title="A Large-Scale Evaluation of a Virtual Teaching Assistant",
                          year="2025", doi="10.1234/x"))
        parsed = records(source)
        assert dd.fuzzy_candidates(parsed, dd.cluster_records(parsed)) == []

    def test_a_pair_reachable_from_two_year_blocks_is_offered_once(self):
        source = (entry("a", title="A Large-Scale Evaluation of Virtual Teaching Assistants",
                        author="Kweon, S.", year="2024")
                  + entry("b", title="A Large-Scale Evaluation of a Virtual Teaching Assistant",
                          author="Kweon, S.", year="2025", doi="10.1234/x"))
        parsed = records(source)
        assert len(dd.fuzzy_candidates(parsed, dd.cluster_records(parsed))) == 1


class TestConflictingClusters:
    def test_a_cluster_whose_records_disagree_about_the_doi_is_flagged(self):
        source = (entry("a", title="A Study of Learning Analytics", author="Smith, Jane", year="2024",
                        doi="10.1007/s10639-024-12764-2")
                  + entry("b", title="A study of learning analytics", author="Smith, Jane", year="2024",
                          doi="10.48550/arXiv.2401.00001"))
        parsed = records(source)
        assert len(dd.conflicting_clusters(dd.cluster_records(parsed))) == 1

    def test_a_springer_chapter_version_pair_is_settled_by_rule(self):
        """What the judge must not be asked about, because it gets it wrong.

        The version suffix is a documented Springer convention, and a model reading the DOI cold does
        not know it: shown four such pairs, Qwen3.6 refused all four as "separate chapters", which would
        have split four works the project decided are one.
        """
        source = (entry("v1", title="Use of AI in Singapore Education: Overview", author="Zhao, L.",
                        year="2026", doi="10.1007/978-981-96-3901-4_5-1")
                  + entry("v2", title="Use of AI in Singapore Education: Overview", author="Zhao, L.",
                          year="2026", doi="10.1007/978-981-96-3901-4_5-2"))
        parsed = records(source)
        assert dd.conflicting_clusters(dd.cluster_records(parsed)), \
            "the pair must reach the conflict list at all, or the exclusion is not what is being tested"
        assert dd.settled_by_rule(*parsed)

    def test_an_ordinary_doi_disagreement_is_not_settled_by_rule(self):
        source = (entry("a", title="A Study of Learning Analytics", author="Smith, Jane", year="2024",
                        doi="10.1145/3587103.3594165")
                  + entry("b", title="A study of learning analytics", author="Smith, Jane", year="2024",
                          doi="10.1016/j.softx.2023.101578"))
        assert not dd.settled_by_rule(*records(source))

    def test_a_cluster_that_agrees_is_not_flagged(self):
        source = (entry("a", title="A Study of Learning Analytics", doi="10.1234/x")
                  + entry("b", title="A study of learning analytics", doi="10.1234/x"))
        parsed = records(source)
        assert dd.conflicting_clusters(dd.cluster_records(parsed)) == []


class TestJudge:
    """The judge, with the backend replaced. Nothing here imports the LLM stack."""

    SOURCE = (entry("a", title="A Large-Scale Evaluation of Virtual Teaching Assistants",
                    author="Kweon, S.", year="2025")
              + entry("b", title="A Large-Scale Evaluation of a Virtual Teaching Assistant",
                      author="Kweon, S.", year="2025", doi="10.1234/x"))

    def _pairs(self):
        parsed = records(self.SOURCE)
        return parsed, [(a, b, "fuzzy") for a, b in
                        dd.fuzzy_candidates(parsed, dd.cluster_records(parsed))]

    def test_a_verdict_of_same_becomes_a_merge(self, monkeypatch):
        monkeypatch.setattr(dd, "_ask_judge", lambda _s, _p: '[{"i": 0, "same": true, "why": "one paper"}]')
        _parsed, pairs = self._pairs()
        assert dd.judge_pairs(None, pairs) == {(0, 1): True}

    def test_a_verdict_of_different_is_recorded_as_such(self, monkeypatch):
        monkeypatch.setattr(dd, "_ask_judge", lambda _s, _p: '[{"i": 0, "same": false, "why": "two papers"}]')
        _parsed, pairs = self._pairs()
        assert dd.judge_pairs(None, pairs) == {(0, 1): False}

    def test_an_answer_whose_index_does_not_resolve_is_dropped(self, monkeypatch):
        """Following the batch classifier: a batch that comes back short leaves those pairs unanswered,
        which is also what makes a re-run the recovery path."""
        monkeypatch.setattr(dd, "_ask_judge", lambda _s, _p: '[{"i": 7, "same": true}]')
        _parsed, pairs = self._pairs()
        assert dd.judge_pairs(None, pairs) == {}

    def test_a_reply_that_is_not_json_costs_the_batch_and_not_the_run(self, monkeypatch):
        monkeypatch.setattr(dd, "_ask_judge", lambda _s, _p: "I would rather not say.")
        _parsed, pairs = self._pairs()
        assert dd.judge_pairs(None, pairs) == {}

    def test_a_fenced_reply_is_read(self, monkeypatch):
        monkeypatch.setattr(dd, "_ask_judge",
                            lambda _s, _p: '```json\n[{"i": 0, "same": true}]\n```')
        _parsed, pairs = self._pairs()
        assert dd.judge_pairs(None, pairs) == {(0, 1): True}

    def test_a_hallucinated_merge_is_refused_by_python(self, monkeypatch):
        """The durable lesson of the batch-classification investigation, applied here.

        The model proposes and Python disposes: a "same work" verdict contradicted by the records
        themselves — a different first author *and* a year too far apart — is dropped whatever the model
        said about it. Without this the model's own judgement would be the only thing checking it.
        """
        source = (entry("a", title="Machine Learning for Chemistry", author="Smith, Jane", year="2015")
                  + entry("b", title="Machine Learning for Chemistry Today", author="Jones, Bob",
                          year="2025"))
        parsed = records(source)
        pairs = [(parsed[0], parsed[1], "fuzzy")]
        monkeypatch.setattr(dd, "_ask_judge", lambda _s, _p: '[{"i": 0, "same": true, "why": "sure"}]')

        assert dd._disagree_on_author(*parsed) and dd._disagree_on_year(*parsed), \
            "the records must actually contradict each other, or the guard has nothing to fire on"
        assert dd.judge_pairs(None, pairs) == {(0, 1): False}

    def test_answers_are_appended_to_the_state_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dd, "_ask_judge", lambda _s, _p: '[{"i": 0, "same": true, "why": "one paper"}]')
        _parsed, pairs = self._pairs()
        state = tmp_path / "judge.jsonl"
        dd.judge_pairs(None, pairs, state)
        assert '"same": true' in state.read_text(encoding="utf-8")

    def test_a_rerun_asks_nothing_it_has_already_asked(self, tmp_path, monkeypatch):
        state = tmp_path / "judge.jsonl"
        monkeypatch.setattr(dd, "_ask_judge", lambda _s, _p: '[{"i": 0, "same": true, "why": "one paper"}]')
        _parsed, pairs = self._pairs()
        dd.judge_pairs(None, pairs, state)

        def refuse(_settings, _prompt):
            raise AssertionError("a resumed run must not ask again")

        monkeypatch.setattr(dd, "_ask_judge", refuse)
        assert dd.judge_pairs(None, pairs, state) == {(0, 1): True}

    def test_a_state_file_with_a_half_written_last_line_is_still_usable(self, tmp_path, monkeypatch):
        """A run killed mid-write leaves one. The pair it names is simply asked again."""
        state = tmp_path / "judge.jsonl"
        state.write_text('{"pair": "x\\ty", "same": true}\n{"pair": "a\\tb", "sam', encoding="utf-8")
        monkeypatch.setattr(dd, "_ask_judge", lambda _s, _p: '[{"i": 0, "same": true, "why": "one paper"}]')
        _parsed, pairs = self._pairs()
        assert dd.judge_pairs(None, pairs, state) == {(0, 1): True}

    def test_the_pair_id_does_not_depend_on_input_order(self, tmp_path, monkeypatch):
        """Resumability keyed on BibTeX keys, not record indices, which shift when files are reordered.

        A resumable file that silently answers a different question after an argument is reordered
        would be worse than no resumability at all.
        """
        monkeypatch.setattr(dd, "_ask_judge", lambda _s, _p: '[{"i": 0, "same": true, "why": "one paper"}]')
        parsed, pairs = self._pairs()
        assert dd._pair_id(parsed[0], parsed[1]) == dd._pair_id(parsed[1], parsed[0])

    def test_the_prompt_carries_what_the_records_say_and_not_their_abstracts(self):
        """Two databases' copies of one abstract differ in ways that say nothing about whether the
        papers are the same, and it would be most of the prompt."""
        parsed = records(entry("a", title="A Study", author="Smith, Jane", year="2024",
                               journal="Journal of Things", doi="10.1234/x",
                               abstract="A long abstract that is not the judge's business."))
        described = dd._describe_for_judge(parsed[0])
        assert "Journal of Things" in described and "10.1234/x" in described
        assert "not the judge's business" not in described


class TestWholeRun:
    def test_a_multi_database_export_deduplicates_end_to_end(self):
        """The shape the tool is for: one paper from three databases, plus an unrelated record."""
        source = (entry("scopus", title="Peer-Reviewed AI: A Study", author="Smith, Jane and Jones, Bob",
                        year="2024", doi="10.1234/abc-def", journal="Journal of Things")
                  + entry("proquest", title="Peer reviewed AI - a study", year="2024",
                          doi="https://doi.org/10.1234/ABC–DEF",
                          abstract="We study things. © 2024 Elsevier Ltd. All rights reserved.",
                          annote="Copyright note")
                  + entry("arxiv", title="Peer-Reviewed AI: A Study", author="Smith, Jane", year="2023",
                          doi="10.48550/arXiv.2301.00001", abstract="We study things.")
                  + entry("other", title="Something Else Entirely", author="Doe, John", year="2024",
                          doi="10.9999/xyz"))

        parsed, unreadable = dd.read_records(source)
        assert unreadable == []
        clusters = dd.cluster_records(parsed)
        library, rows = dd.deduplicate(clusters)

        assert [entry_.key for entry_ in library.entries] == ["scopus", "other"]
        assert len(rows) == 1 and rows[0].removed == ("proquest", "arxiv")

        merged = {field.key: field.value for field in library.entries[0].fields}
        assert merged["doi"] == "10.1234/abc-def"      # the en-dash copy matched, and lost
        assert merged["annote"] == "Copyright note"    # filled in from the twin that had one
        # That twin's abstract wins on stripped length and arrives as its source wrote it, notice and
        # all — the comparison is what stripping is for.
        assert merged["abstract"].startswith("We study things.")
        assert merged["abstract"].endswith("All rights reserved.")

    def test_the_output_reads_back_as_the_records_that_were_written(self):
        from raven.papers import bibtex

        source = (entry("a", title="A Study of {LaTeX} Braces and Ümlauts", author="Smith, Jane",
                        year="2024", doi="10.1234/x")
                  + entry("b", title="A study of {LaTeX} braces and Ümlauts", year="2024", doi="10.1234/x",
                          keywords="k")
                  + entry("c", title="Something Else Entirely", author="Doe, John", doi="10.5678/y"))
        parsed, _unreadable = dd.read_records(source)
        library, _rows = dd.deduplicate(dd.cluster_records(parsed))
        written = bibtex.write_string(library)

        reread, unreadable = dd.read_records(written)
        assert unreadable == []
        assert [record.key for record in reread] == ["a", "c"]
        assert reread[0].field("title") == "A Study of {LaTeX} Braces and Ümlauts"
        assert reread[0].field("keywords") == "k"

    def test_output_keys_are_unique(self):
        """A repeated key does not merely duplicate: `bibtexparser` turns it into a failed block and
        then raises while writing, so the whole bibliography would be lost to it."""
        source = "".join(entry(f"k{i}", title=f"Title number {i}", doi=f"10.1234/{i}") for i in range(20))
        parsed, _unreadable = dd.read_records(source)
        library, _rows = dd.deduplicate(dd.cluster_records(parsed))
        keys = [entry_.key for entry_ in library.entries]
        assert len(keys) == len(set(keys))
