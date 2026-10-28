"""Unit tests for raven.papers.fillbib (filling missing and truncated abstracts from Crossref)."""

import json

import pytest

from raven.common import tabular
from raven.common import utils as common_utils
from raven.papers import bibtex, fillbib


def record(key: str, *, doi: str | None = None, abstract: str | None = None) -> str:
    fields = [f"title = {{The paper called {key}}}", "author = {Doe, Jane}", "year = {2024}"]
    if doi is not None:
        fields.append(f"doi = {{{doi}}}")
    if abstract is not None:
        fields.append(f"abstract = {{{abstract}}}")
    return f"@article{{{key},\n  " + ",\n  ".join(fields) + "\n}"


def library(*entries: str):
    return bibtex.parse_string("\n\n".join(entries), split_names=False)


def abstract_of(lib, key: str) -> str:
    return next(bibtex.entry_field(entry, "abstract") for entry in lib.entries if entry.key == key)


def _without_cut(text: str) -> str:
    """`comparable` as it would be if appended texts were not cut: the negative control's yardstick."""
    return "".join(character for character in common_utils.unicodize_basic_markup(text).casefold()
                   if character.isalnum())


WHOLE = ("We studied how students use language models when writing essays, across three courses and two "
         "terms, and found that the effect depends on the feedback they receive.")
TEASER = WHOLE[:90] + " ..."


class TestRenderJats:
    def test_paragraphs_are_separated_by_a_blank_line(self):
        assert fillbib.render_jats("<jats:p>One.</jats:p><jats:p>Two.</jats:p>") == "One.\n\nTwo."

    def test_a_heading_that_only_says_abstract_is_dropped(self):
        assert fillbib.render_jats("<jats:title>Abstract</jats:title><jats:p>Text.</jats:p>") == "Text."

    def test_a_structured_abstract_keeps_its_headings_in_bold(self):
        jats = ("<jats:sec><jats:title>Background</jats:title><jats:p>Why.</jats:p></jats:sec>"
                "<jats:sec><jats:title>Methods</jats:title><jats:p>How.</jats:p></jats:sec>")
        assert fillbib.render_jats(jats) == "\\textbf{Background}\n\nWhy.\n\n\\textbf{Methods}\n\nHow."

    def test_a_plain_language_summary_section_is_left_out(self):
        jats = ("<jats:sec><jats:title>Background</jats:title><jats:p>Why.</jats:p></jats:sec>"
                "<jats:sec><jats:title>Plain Language Summary</jats:title><jats:p>Simply.</jats:p></jats:sec>")
        assert fillbib.render_jats(jats) == "\\textbf{Background}\n\nWhy."

    def test_a_plain_language_summary_heading_takes_its_paragraphs_with_it(self):
        jats = ("<jats:p>The abstract.</jats:p>"
                "<jats:title>Plain language summary</jats:title><jats:p>Simply.</jats:p>")
        assert fillbib.render_jats(jats) == "The abstract."

    def test_inline_styles_become_latex(self):
        jats = ("<jats:p><jats:italic>In vitro</jats:italic> CO<jats:sub>2</jats:sub> at "
                "10<jats:sup>3</jats:sup> K, <jats:bold>twice</jats:bold>.</jats:p>")
        assert fillbib.render_jats(jats) == (r"\textit{In vitro} CO\textsubscript{2} at "
                                             r"10\textsuperscript{3} K, \textbf{twice}.")

    def test_mathml_indentation_is_not_text(self):
        jats = """<jats:p>Aged <jats:inline-formula><mml:math>
                      <mml:msub>
                        <mml:mi>M</mml:mi>
                        <mml:mi>age</mml:mi>
                      </mml:msub>
                      <mml:mo>=</mml:mo>
                      <mml:mn>12.9</mml:mn>
                    </mml:math></jats:inline-formula> years.</jats:p>"""
        assert fillbib.render_jats(jats) == r"Aged M\textsubscript{age} = 12.9 years."

    def test_a_tex_alternative_is_not_rendered_beside_the_mathml(self):
        jats = ("<jats:p><jats:inline-formula><jats:alternatives><jats:tex-math>$x^2$</jats:tex-math>"
                "<mml:math><mml:msup><mml:mi>x</mml:mi><mml:mn>2</mml:mn></mml:msup></mml:math>"
                "</jats:alternatives></jats:inline-formula></jats:p>")
        assert fillbib.render_jats(jats) == r"x\textsuperscript{2}"

    def test_text_is_escaped_for_bibtex(self):
        assert fillbib.render_jats("<jats:p>50% of R&amp;D</jats:p>") == r"50\% of R\&D"

    def test_malformed_xml_falls_back_to_its_text(self):
        assert fillbib.render_jats("<jats:p>Unclosed &nbsp; text") == "Unclosed text"

    def test_a_document_type_declaration_is_not_parsed(self):
        jats = '<!DOCTYPE x [<!ENTITY a "boom">]><jats:p>&a;</jats:p>'
        assert "boom" not in fillbib.render_jats(jats)

    def test_what_it_writes_imports_as_markdown(self):
        jats = ("<jats:sec><jats:title>Aim</jats:title><jats:p>An <jats:italic>in situ</jats:italic> "
                "study of H<jats:sub>2</jats:sub>.</jats:p></jats:sec>")
        assert common_utils.unicodize_basic_markup(fillbib.render_jats(jats)) == "**Aim**\n\nAn *in situ* study of H₂."


class TestComparable:
    def test_markup_spacing_and_case_do_not_count(self):
        assert fillbib.comparable(r"\textit{In vitro}  results, 50\%.") == fillbib.comparable("in vitro results 50%")

    def test_a_trailing_ellipsis_is_ignored(self):
        assert fillbib.comparable("Some text ...") == fillbib.comparable("Some text")
        assert fillbib.comparable("Some text…") == fillbib.comparable("Some text")

    def test_a_trailing_rights_notice_is_ignored(self):
        assert fillbib.comparable("Some text. © 2025 by the authors.") == fillbib.comparable("Some text.")

    @pytest.mark.parametrize("heading", ["Plain language summary", "Graphical abstract", "Alternate abstract:"])
    def test_everything_from_an_appended_texts_heading_on_is_ignored(self, heading):
        assert fillbib.comparable(f"Abstract part.{heading} Other part") == fillbib.comparable("Abstract part.")

    @pytest.mark.parametrize("joined", [
        "Abstract part ...Plain language summaryOther part",       # glued on, as the exports have it
        "Abstract part ...Plain Language Summary Other part",
        "Abstract part.Alternate abstract:Otra parte",
        "Abstract part.” Graphical abstract: Other part",      # `”` ”
        "Abstract part\nPlain language summary\nOther part",       # at the start of a line
    ])
    def test_a_heading_is_recognized_where_an_export_puts_one(self, joined):
        assert fillbib.comparable(joined) == fillbib.comparable("Abstract part")

    @pytest.mark.parametrize("prose", [
        "We asked whether a plain language summary helps readers. It did.",   # not after a sentence end
        "Readers liked it. Plain language summary writing is a skill.",       # lowercase after it
        "Readers liked it. Graphical abstracts are common in chemistry.",     # a longer word
        "Readers liked it. Plain language summaries vary.",
    ])
    def test_the_same_words_in_prose_are_not_cut(self, prose):
        assert fillbib.comparable(prose) == _without_cut(prose)

    def test_a_teaser_mentioning_a_heading_in_prose_is_repaired(self):
        # Cut at the mention, the teaser and the whole abstract compare equal, and the teaser is kept as whole.
        theirs = ("We asked whether a plain language summary helps readers of a review, and found that it "
                  "does in two of the three settings we tried.")
        ours = theirs[:70] + " ..."
        filled, audit = fillbib.fill(library(record("a", doi="10.1234/x", abstract=ours)), {"10.1234/x": theirs})
        assert [row.outcome for row in audit] == [fillbib.REPAIRED]
        assert abstract_of(filled, "a") == theirs

    def test_a_teaser_with_an_appended_text_is_repaired(self):
        # The shape a database export leaves: the abstract cut short, a heading glued on, the second text
        # cut short.
        ours = WHOLE[:100] + " ...Graphical abstractStudents at three universities were ..."
        assert not fillbib.comparable(WHOLE).startswith(_without_cut(ours)), \
            "without the cut ours is no prefix, or this fixture cannot show the cut matters"
        filled, audit = fillbib.fill(library(record("a", doi="10.1234/x", abstract=ours)), {"10.1234/x": WHOLE})
        assert audit[0].outcome == fillbib.REPAIRED

    def test_a_structured_abstracts_headings_compare_with_inline_ones(self):
        # A database export runs the headings into the text; the rendered copy has them as paragraphs.
        ours = "Background: Why it matters.Aim: This review"
        theirs = fillbib.render_jats("<jats:sec><jats:title>Background:</jats:title><jats:p>Why it matters."
                                     "</jats:p></jats:sec><jats:sec><jats:title>Aim:</jats:title><jats:p>This "
                                     "review covers it.</jats:p></jats:sec>")
        assert fillbib.comparable(theirs).startswith(fillbib.comparable(ours))


class TestWordsShared:
    def test_a_teaser_is_wholly_in_the_abstract_it_starts(self):
        assert fillbib.words_shared(TEASER, WHOLE) == 1.0

    def test_a_small_edit_costs_little(self):
        edited = WHOLE.replace("three courses", "3 courses")
        assert 0.95 < fillbib.words_shared(edited, WHOLE) < 1.0

    def test_a_different_text_shares_little(self):
        other = "Ce travail examine un autre sujet entièrement, avec une ouverture différente."
        assert fillbib.words_shared(other, WHOLE) < 0.1

    def test_order_counts(self):
        backwards = " ".join(reversed(WHOLE.split()))
        assert fillbib.words_shared(backwards, WHOLE) < 0.5


class TestFill:
    DOI = "10.1234/abc.5678"

    def test_a_missing_abstract_is_filled(self):
        filled, audit = fillbib.fill(library(record("a", doi=self.DOI)), {self.DOI: WHOLE})
        assert abstract_of(filled, "a") == WHOLE
        assert [(row.key, row.outcome) for row in audit] == [("a", fillbib.FILLED)]

    def test_a_truncated_abstract_is_repaired(self):
        filled, audit = fillbib.fill(library(record("a", doi=self.DOI, abstract=TEASER)), {self.DOI: WHOLE})
        assert abstract_of(filled, "a") == WHOLE
        assert audit[0].outcome == fillbib.REPAIRED
        assert audit[0].before.endswith("...") and audit[0].crossref == WHOLE
        assert audit[0].maybe_shared == 1.0

    def test_a_different_text_is_kept_and_listed(self):
        other = "This study investigates something else entirely, with a different opening sentence."
        filled, audit = fillbib.fill(library(record("a", doi=self.DOI, abstract=TEASER)), {self.DOI: other})
        assert abstract_of(filled, "a") == TEASER
        assert audit[0].outcome == f"{fillbib.DIFFERS}; {fillbib.LOOKS_TRUNCATED}"

    def test_a_kept_record_shows_crossrefs_text_and_how_much_they_share(self):
        edited = WHOLE.replace("three courses", "3 courses")
        _, audit = fillbib.fill(library(record("a", doi=self.DOI, abstract=edited)), {self.DOI: WHOLE})
        assert audit[0].outcome == fillbib.DIFFERS
        assert audit[0].crossref == WHOLE
        assert 0.95 < audit[0].maybe_shared < 1.0

    def test_a_different_whole_text_is_not_called_truncated(self):
        other = "This study investigates something else entirely, with a different opening sentence."
        _, audit = fillbib.fill(library(record("a", doi=self.DOI, abstract=WHOLE)), {self.DOI: other})
        assert audit[0].outcome == fillbib.DIFFERS

    def test_a_whole_abstract_is_left_alone_and_not_listed(self):
        filled, audit = fillbib.fill(library(record("a", doi=self.DOI, abstract=WHOLE)), {self.DOI: WHOLE})
        assert abstract_of(filled, "a") == WHOLE
        assert audit == []

    def test_a_short_fragment_is_not_evidence_of_the_same_text(self):
        fragment = WHOLE[:20]  # a prefix, but too short to say it begins this abstract rather than another
        assert fillbib.comparable(WHOLE).startswith(fillbib.comparable(fragment)), \
            "the fragment must be a prefix, or this fixture cannot tell the guard from the prefix test"
        _, audit = fillbib.fill(library(record("a", doi=self.DOI, abstract=fragment)), {self.DOI: WHOLE})
        assert audit[0].outcome == fillbib.DIFFERS

    def test_doi_spellings_match_after_normalization(self):
        bib = library(record("a", doi="https://doi.org/10.1234/ABC.5678"))
        filled, _ = fillbib.fill(bib, {self.DOI: WHOLE})
        assert abstract_of(filled, "a") == WHOLE

    def test_records_that_could_not_be_helped_are_listed_when_something_is_missing(self):
        bib = library(record("nodoi", abstract=TEASER),
                      record("absent", doi="10.1234/absent"),
                      record("empty", doi="10.1234/empty", abstract=TEASER),
                      record("broken", doi="10.1234/broken"))
        _, audit = fillbib.fill(bib, {"10.1234/empty": None}, failed={"10.1234/broken": "ConnectionError: x"})
        assert {row.key: row.outcome for row in audit} == {
            "nodoi": f"{fillbib.NO_DOI}; {fillbib.LOOKS_TRUNCATED}",
            "absent": fillbib.NOT_IN_CROSSREF,
            "empty": f"{fillbib.NO_ABSTRACT_IN_CROSSREF}; {fillbib.LOOKS_TRUNCATED}",
            "broken": f"{fillbib.LOOKUP_FAILED}: ConnectionError: x"}

    def test_a_whole_abstract_that_could_not_be_looked_up_is_not_listed(self):
        bib = library(record("nodoi", abstract=WHOLE), record("absent", doi="10.1234/absent", abstract=WHOLE))
        _, audit = fillbib.fill(bib, {})
        assert audit == []

    def test_a_second_text_is_removed_from_an_abstract_the_record_keeps(self):
        ours = "A whole abstract of our own. Plain language summaryWhat we did, simply."
        filled, audit = fillbib.fill(library(record("a", abstract=ours)), {})
        assert abstract_of(filled, "a") == "A whole abstract of our own."
        assert [(row.outcome, row.before) for row in audit] == [(f"{fillbib.NO_DOI}; {fillbib.APPENDED_REMOVED}", ours)]

    def test_a_second_text_is_removed_where_the_abstract_equals_crossrefs(self):
        ours = WHOLE + " Plain language summaryWhat we did, simply."
        filled, audit = fillbib.fill(library(record("a", doi=self.DOI, abstract=ours)), {self.DOI: WHOLE})
        assert abstract_of(filled, "a") == WHOLE
        assert audit[0].outcome == f"{fillbib.SAME}; {fillbib.APPENDED_REMOVED}"

    def test_a_teaser_keeps_its_ellipsis_when_its_second_text_goes(self):
        other = "This study investigates something else entirely, with a different opening sentence."
        ours = TEASER + "Plain language summaryWhat we did, simp ..."
        filled, audit = fillbib.fill(library(record("a", doi=self.DOI, abstract=ours)), {self.DOI: other})
        assert abstract_of(filled, "a") == TEASER
        assert audit[0].outcome == f"{fillbib.DIFFERS}; {fillbib.APPENDED_REMOVED}; {fillbib.LOOKS_TRUNCATED}"

    def test_a_second_text_that_is_all_there_is_stays(self):
        ours = "Plain language summary: What we did, simply, in a few words of our own."
        filled, audit = fillbib.fill(library(record("a", abstract=ours)), {})
        assert abstract_of(filled, "a") == ours
        assert audit == []

    def test_keeping_second_texts_leaves_the_abstract_as_it_was(self):
        ours = WHOLE + " Plain language summaryWhat we did, simply."
        bib = library(record("a", doi=self.DOI, abstract=ours))
        stripped, _ = fillbib.fill(bib, {self.DOI: WHOLE})
        assert abstract_of(stripped, "a") != ours, "nothing was removed by default, so this cannot test keeping"
        kept, audit = fillbib.fill(bib, {self.DOI: WHOLE}, strip_appended=False)
        assert abstract_of(kept, "a") == ours
        assert audit == []

    def test_the_input_library_is_not_modified(self):
        bib = library(record("a", doi=self.DOI))
        fillbib.fill(bib, {self.DOI: WHOLE})
        assert abstract_of(bib, "a") == ""

    def test_the_filled_library_round_trips_through_bibtex(self):
        theirs = "\\textbf{Background}\n\nFirst, 50\\% of it.\n\n\\textbf{Aim}\n\nSecond."
        filled, _ = fillbib.fill(library(record("a", doi=self.DOI)), {self.DOI: theirs})
        assert abstract_of(bibtex.parse_string(bibtex.write_string(filled)), "a") == theirs


class FakeResponse:
    def __init__(self, items):
        self._items = items

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"items": self._items}}


class FakeCrossref:
    """Answers `filter=doi:…` queries from a dict of DOI → JATS (or `None` for "found, no abstract")."""

    def __init__(self, known, *, fail_on=None):
        self.known = known
        self.fail_on = fail_on
        self.requests = []

    def __call__(self, url, params, timeout):
        dois = [value.removeprefix("doi:") for value in params["filter"].split(",doi:")]
        dois = [doi.removeprefix("doi:") for doi in dois]
        self.requests.append(dois)
        if self.fail_on and self.fail_on in dois:
            raise ConnectionError("no route")
        return FakeResponse([{"DOI": doi.upper(), **({"abstract": self.known[doi]} if self.known[doi] else {})}
                             for doi in dois if doi in self.known])


class TestFetchCrossref:
    def test_dois_are_asked_for_in_batches_and_rendered(self):
        crossref = FakeCrossref({"10.1000/a": "<jats:p>A.</jats:p>", "10.1000/b": None, "10.1000/c": "<jats:p>C.</jats:p>"})
        abstracts, failed = fillbib.fetch_crossref(["10.1000/a", "10.1000/b", "10.1000/c", "10.1000/d"],
                                                   batch_size=2, delay=0, get=crossref)
        assert crossref.requests == [["10.1000/a", "10.1000/b"], ["10.1000/c", "10.1000/d"]]
        assert abstracts == {"10.1000/a": "A.", "10.1000/b": None, "10.1000/c": "C."}  # 10.1/d: not found
        assert failed == {}

    def test_a_doi_with_a_comma_is_asked_for_alone(self):
        crossref = FakeCrossref({})
        fillbib.fetch_crossref(["10.1000/a", "10.1000/x,y", "10.1000/b"], batch_size=50, delay=0, get=crossref)
        assert sorted(crossref.requests) == [["10.1000/a", "10.1000/b"], ["10.1000/x,y"]]

    def test_a_failed_batch_costs_only_its_own_dois(self):
        crossref = FakeCrossref({"10.1000/a": "<jats:p>A.</jats:p>", "10.1000/c": "<jats:p>C.</jats:p>"}, fail_on="10.1000/a")
        abstracts, failed = fillbib.fetch_crossref(["10.1000/a", "10.1000/b", "10.1000/c"],
                                                   batch_size=2, delay=0, get=crossref)
        assert set(failed) == {"10.1000/a", "10.1000/b"}
        assert abstracts == {"10.1000/c": "C."}

    def test_the_cache_answers_a_rerun_without_asking(self, tmp_path):
        cache = tmp_path / "crossref.jsonl"
        crossref = FakeCrossref({"10.1000/a": "<jats:p>A.</jats:p>"})
        first = fillbib.fetch_crossref(["10.1000/a", "10.1000/b"], maybe_cache_path=cache, delay=0, get=crossref)
        assert crossref.requests, "the first run must ask, or this fixture cannot show the second one did not"
        crossref.requests.clear()
        second = fillbib.fetch_crossref(["10.1000/a", "10.1000/b"], maybe_cache_path=cache, delay=0, get=crossref)
        assert crossref.requests == []
        assert first == second == ({"10.1000/a": "A."}, {})

    def test_the_cache_keeps_the_raw_answer(self, tmp_path):
        cache = tmp_path / "crossref.jsonl"
        fillbib.fetch_crossref(["10.1000/a"], maybe_cache_path=cache, delay=0,
                               get=FakeCrossref({"10.1000/a": "<jats:p>A.</jats:p>"}))
        assert json.loads(cache.read_text())["abstract"] == "<jats:p>A.</jats:p>"

    def test_a_failed_lookup_is_not_cached(self, tmp_path):
        cache = tmp_path / "crossref.jsonl"
        fillbib.fetch_crossref(["10.1000/a"], maybe_cache_path=cache, delay=0,
                               get=FakeCrossref({}, fail_on="10.1000/a"))
        assert not cache.exists() or cache.read_text() == ""


def test_write_audit_names_the_tool_and_the_copyright(tmp_path):
    path = tmp_path / "audit.tsv"
    fillbib.write_audit(path, [fillbib.AuditRow(key="a", outcome=fillbib.NO_DOI, title="T", venue="V")],
                        ["in.bib"])
    text = path.read_text(encoding="utf-8")
    assert "raven-fillbib" in text and "copyright" in text
    assert "\t".join(fillbib.AUDIT_COLUMNS) in text


@pytest.mark.parametrize("fmt", tabular.FORMATS)
def test_write_audit_writes_every_format(tmp_path, fmt):
    fillbib.write_audit(tmp_path / f"audit.{fmt}", [], ["in.bib"])
    assert (tmp_path / f"audit.{fmt}").exists()
