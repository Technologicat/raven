"""Tests for `raven.papers.scopebib.records`: loading records, and the title and abstract checks."""

import types

import pytest

from raven.papers.scopebib import records


def _bib(tmp_path, text):
    path = tmp_path / "corpus.bib"
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadRecords:
    def test_a_record_carries_readable_text_and_where_to_read_it(self, tmp_path):
        path = _bib(tmp_path, "@article{smith_2024,\n"
                              "  title = {{\\o}nly a {Study}},\n"
                              "  abstract = {Some   text.},\n"
                              "  journal = {Journal of Things},\n"
                              "  doi = {10.1000/xyz},\n"
                              "}\n")
        (record,) = records.load_records(path)
        assert record == records.Record(key="smith_2024", title="ønly a Study", abstract="Some text.",
                                        venue="Journal of Things", link=record.link)
        assert "10.1000/xyz" in record.link

    def test_a_record_carries_its_authors_and_year_for_a_reader(self, tmp_path):
        (record,) = records.load_records(_bib(tmp_path, "@misc{k,\n  title = {T},\n"
                                                        "  author = {M{\\\"u}ller, Anna and Smith, Bo},\n"
                                                        "  year = { 2024 },\n}\n"))
        assert (record.authors, record.year) == ("Müller, Anna and Smith, Bo", "2024")

    def test_a_missing_field_is_empty_rather_than_absent(self, tmp_path):
        (record,) = records.load_records(_bib(tmp_path, "@misc{k,\n  title = {T},\n}\n"))
        assert (record.abstract, record.venue, record.link) == ("", "", "")

    def test_records_come_in_file_order_and_a_str_path_works(self, tmp_path):
        path = _bib(tmp_path, "@misc{b,\n  title = {B},\n}\n@misc{a,\n  title = {A},\n}\n")
        assert [record.key for record in records.load_records(str(path))] == ["b", "a"]


class _Token:
    def __init__(self, text, lemma=None):
        self.text = text
        self.lemma_ = lemma if lemma is not None else text.lower()
        self.is_alpha = text.isalpha()


class _FakeNLP:
    """Splits on spaces; a word may carry its lemma as `word/lemma`. Records each call."""

    def __init__(self):
        self.calls = []

    def analyze(self, texts):
        self.calls.append(list(texts))
        return [[_Token(*word.split("/")) for word in text.split()] for text in texts]


class TestContentWords:
    STOPWORDS = {"the", "of", "a", "potential", "advantage", "exploring"}

    def test_function_words_filler_and_non_words_do_not_count(self):
        counts = records.content_word_counts(["Exploring the potential of graph neural networks 2024"],
                                             _FakeNLP(), stopwords=self.STOPWORDS)
        assert counts == [3]  # graph, neural, networks; "2024" is not alphabetic

    def test_an_inflected_stopword_is_caught_by_its_lemma(self):
        # Without the lemma check, "advantages" would count: the list holds base forms only.
        title = "advantages/advantage of learning"
        assert records.content_word_counts([title], _FakeNLP(), stopwords=self.STOPWORDS) == [1]
        no_lemma = records.content_word_counts(["advantages of learning"], _FakeNLP(), stopwords=self.STOPWORDS)
        assert no_lemma == [2], "this fixture cannot show the lemma check: the surface form already misses"

    def test_all_titles_go_in_one_call(self):
        nlp = _FakeNLP()
        records.content_word_counts(["a b", "c d", "e f"], nlp, stopwords=self.STOPWORDS)
        assert len(nlp.calls) == 1

    def test_no_titles_ask_nothing(self):
        nlp = _FakeNLP()
        assert records.content_word_counts([], nlp, stopwords=self.STOPWORDS) == []
        assert nlp.calls == []


class TestIsUninformative:
    def test_too_few_content_words(self):
        assert records.is_uninformative(records.MIN_CONTENT_WORDS - 1, 3)
        assert not records.is_uninformative(records.MIN_CONTENT_WORDS, records.MIN_CONTENT_WORDS)

    def test_a_long_title_that_is_mostly_padding(self):
        words, content = 15, records.MIN_CONTENT_WORDS
        assert content / words < records.MIN_CONTENT_DENSITY, "this fixture is not padded enough to test density"
        assert records.is_uninformative(content, words)

    def test_informative_words_count_only_words_with_letters(self):
        # A hyphen splits too, so "Graph-based" is two words; "2024:" carries no letters.
        assert records.informative_words("Graph-based learning in 2024: a study") == 6


class TestTitleMeasures:
    def test_each_distinct_title_is_analysed_once(self):
        nlp = _FakeNLP()
        recs = [records.Record(key=k, title=t, abstract="", venue="", link="")
                for k, t in (("a", "graph neural networks"), ("b", "graph neural networks"), ("c", "the of"))]
        measures = records.title_measures(recs, nlp, stopwords=TestContentWords.STOPWORDS)
        assert nlp.calls == [["graph neural networks", "the of"]]
        assert measures["a"] == measures["b"] == records.TitleMeasure(3, 3, False)
        assert measures["c"] == records.TitleMeasure(0, 2, True)


class TestAbstracts:
    @pytest.mark.parametrize("abstract", ["Given this ...", "with an increased …", "It goes on... \n"])
    def test_an_ellipsis_at_the_end_is_a_teaser(self, abstract):
        assert records.looks_truncated(abstract)

    @pytest.mark.parametrize("abstract", ["", "As one author put it, ... and so we did.", "Complete."])
    def test_anything_else_is_not(self, abstract):
        assert not records.looks_truncated(abstract)

    def test_describe_abstract(self):
        def describe(abstract):
            return records.describe_abstract(records.Record("k", "t", abstract, "", ""))
        assert [describe(""), describe("Cut off ..."), describe("Whole.")] == ["none", "teaser", "full"]


@pytest.fixture(scope="module")
def nlp():
    """The real spaCy pipeline, behind the `analyze` method `content_word_counts` asks for."""
    pytest.importorskip("spacy")
    from raven.common import nlptools
    pipeline = nlptools.load_spacy_pipeline("en_core_web_sm", "cpu")
    return types.SimpleNamespace(analyze=lambda texts: [pipeline(text) for text in texts])


@pytest.mark.ml
class TestWithSpacy:
    """The real pipeline, which the counting was calibrated with: lemmas, and the stopwords as shipped."""

    def test_a_pluralized_filler_title_is_still_filler(self, nlp):
        singular, plural = records.content_word_counts(
            ["Tackling possible advantage of technical understanding",
             "Tackling possible advantages of technical understandings"], nlp)
        assert singular == plural, "an inflection let filler through: the lemma check is not working"
        assert singular < records.MIN_CONTENT_WORDS

    def test_a_short_precise_title_is_informative(self, nlp):
        title = "Unfitted finite element methods"
        (content,) = records.content_word_counts([title], nlp)
        assert not records.is_uninformative(content, records.informative_words(title))
