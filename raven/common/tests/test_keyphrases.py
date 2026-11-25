"""Tests for raven.common.keyphrases.

Everything but `extract` runs on hand-built inputs, so it needs no NLP stack and runs in CI. `extract` needs
spaCy and its small English pipeline, so its class carries the `ml` marker.
"""

import numpy as np
import pytest

from raven.common import keyphrases


def _extracted(phrases, definitions=(), nouns=(), casing=None):
    """One record as `extract` returns it."""
    return {"phrases": sorted(phrases), "definitions": list(definitions), "nouns": sorted(nouns),
            "casing": dict(casing or {})}


class TestSentenceCase:
    def test_title_case_is_lowered(self):
        assert keyphrases.sentence_case("Free Vibration Of Rectangular Plates") == "free vibration of rectangular plates"

    def test_all_caps_is_lowered(self):
        assert keyphrases.sentence_case("FREE VIBRATION OF A CLAMPED PLATE") == "free vibration of a clamped plate"

    def test_sentence_case_is_left_alone(self):
        title = "Stability of axially moving plates in Kelvin-Voigt materials"
        assert keyphrases.sentence_case(title) == title

    def test_a_short_title_is_left_alone(self):
        assert keyphrases.sentence_case("Mistral 7B") == "Mistral 7B"


class TestNfkc:
    def test_subscript_digits_become_digits(self):
        assert keyphrases.nfkc("TiO₂") == "TiO2"  # SUBSCRIPT TWO


class TestNormalize:
    def test_a_plural_folds_onto_a_singular_the_corpus_has(self):
        terms, _ = keyphrases.normalize([_extracted({"language models"}), _extracted({"language model"})])
        assert terms == [{"language model"}, {"language model"}]

    def test_a_word_merely_ending_in_s_is_kept(self):
        terms, _ = keyphrases.normalize([_extracted({"process"}), _extracted({"proces"})])
        assert terms[0] == {"process"}  # "ss" is never a plural ending

    def test_ics_folds_only_where_the_corpus_uses_the_ic_noun(self):
        records = [_extracted({"key characteristics", "physics"}, nouns={"characteristic", "physics"})]
        terms, _ = keyphrases.normalize(records)
        assert terms[0] == {"key characteristic", "physics"}

    def test_ics_is_kept_without_the_ic_noun(self):
        """Negative control for the fold above: the same term, with no "characteristic" in the corpus."""
        terms, _ = keyphrases.normalize([_extracted({"key characteristics"}, nouns={"characteristics"})])
        assert terms[0] == {"key characteristics"}

    def test_a_hyphen_and_a_space_are_one_term_spelled_as_most_records_spell_it(self):
        records = [_extracted({"retrieval-augmented generation"}), _extracted({"retrieval-augmented generation"}),
                   _extracted({"retrieval augmented generation"})]
        terms, display = keyphrases.normalize(records)
        assert all(t == {"retrieval-augmented generation"} for t in terms)
        assert display["retrieval-augmented generation"] == "retrieval-augmented generation"

    def test_the_majority_spelling_wins_either_way(self):
        """Control for the test above: with the majority reversed, the spaced spelling is the term."""
        records = [_extracted({"retrieval augmented generation"}), _extracted({"retrieval augmented generation"}),
                   _extracted({"retrieval-augmented generation"})]
        terms, _ = keyphrases.normalize(records)
        assert all(t == {"retrieval augmented generation"} for t in terms)

    def test_a_joined_spelling_is_one_term_with_the_hyphenated_one(self):
        records = [_extracted({"bitwidth"}), _extracted({"bitwidth"}), _extracted({"bit-width"})]
        terms, _ = keyphrases.normalize(records)
        assert all(t == {"bitwidth"} for t in terms)

    def test_different_letters_stay_apart(self):
        """Control for the folds above: only the separators may differ."""
        terms, _ = keyphrases.normalize([_extracted({"bit-width"}), _extracted({"bit-depth"})])
        assert terms == [{"bit-width"}, {"bit-depth"}]

    def test_the_display_form_takes_the_corpus_s_capitals(self):
        records = [_extracted({"solomonoff induction"}, casing={"solomonoff": "Solomonoff"}),
                   _extracted({"solomonoff induction"}, casing={"solomonoff": "Solomonoff"})]
        terms, display = keyphrases.normalize(records)
        assert terms[0] == {"solomonoff induction"}, "the term itself stays lowercase; only its display changes"
        assert display["solomonoff induction"] == "Solomonoff induction"

    def test_an_abbreviation_defined_by_two_records_applies_everywhere(self):
        definition = ("srl", "self-regulated learning")
        records = [_extracted({"srl"}, [definition]), _extracted({"srl"}, [definition]), _extracted({"srl"})]
        terms, _ = keyphrases.normalize(records)
        assert all(t == {"self-regulated learning"} for t in terms)

    def test_an_abbreviation_defined_by_one_record_stays_in_it(self):
        records = [_extracted({"care"}, [("care", "collaborative assistant for personalized exploration")]),
                   _extracted({"care"})]
        terms, _ = keyphrases.normalize(records)
        assert terms[0] == {"collaborative assistant for personalized exploration"}
        assert terms[1] == {"care"}

    def test_abbreviations_expand_inside_a_phrase(self):
        definitions = [("genai", "generative ai"), ("ai", "artificial intelligence")]
        records = [_extracted({"genai", "generative ai"}, definitions)] * 2
        terms, _ = keyphrases.normalize(records)
        assert terms[0] == {"generative artificial intelligence"}

    def test_a_known_abbreviation_displays_as_the_short_form(self):
        """Mostly used without a definition: the corpus's readers know it."""
        definition = ("chatgpt", "chat generative pre-trained transformer")
        records = [_extracted({"chatgpt"}, [definition])] * 2 + [_extracted({"chatgpt"})] * 3
        terms, display = keyphrases.normalize(records)
        assert display[next(iter(terms[0]))] == "chatgpt"

    def test_an_unfamiliar_abbreviation_displays_spelled_out(self):
        """Defined wherever it is used: the reader would not know it."""
        definition = ("fwp", "fast weight programmer")
        records = [_extracted({"fwp"}, [definition])] * 2 + [_extracted({"fwp"})]
        terms, display = keyphrases.normalize(records)
        assert display[next(iter(terms[0]))] == "fast weight programmer"

    def test_the_short_form_share_moves_the_line(self):
        definition = ("fwp", "fast weight programmer")
        records = [_extracted({"fwp"}, [definition])] * 2 + [_extracted({"fwp"})]  # one record in three undefined
        terms, display = keyphrases.normalize(records, short_form_share=0.3)
        assert display[next(iter(terms[0]))] == "fwp"


class TestCasing:
    def test_the_form_most_records_use_wins(self):
        records = [_extracted(set(), casing={"lora": "LoRA"}), _extracted(set(), casing={"lora": "LoRA"}),
                   _extracted(set(), casing={"lora": "lora"})]
        assert keyphrases.casing_table(records) == {"lora": "LoRA"}

    def test_a_tie_goes_to_lowercase(self):
        records = [_extracted(set(), casing={"model": "Model"}), _extracted(set(), casing={"model": "model"})]
        assert keyphrases.casing_table(records) == {}

    def test_a_known_word_takes_the_corpus_s_form(self):
        assert keyphrases.recase("solomonoff induction", {"solomonoff": "Solomonoff"}) == "Solomonoff induction"

    def test_random_title_case_is_undone(self):
        assert keyphrases.recase("Foundation Models", {}) == "foundation models"

    def test_title_case_does_not_undo_a_proper_name_the_corpus_knows(self):
        assert keyphrases.recase("Solomonoff Induction", {"solomonoff": "Solomonoff"}) == "Solomonoff induction"

    def test_a_partly_capitalized_term_keeps_its_unknown_capitals(self):
        """Control for the title-case rule: not every word capitalized, so the capitals are taken as meant."""
        assert keyphrases.recase("Bayes-optimal agents", {}) == "Bayes-optimal agents"

    def test_capitals_past_the_first_letter_are_kept(self):
        assert keyphrases.recase("Large LLMs", {}) == "large LLMs"

    def test_hyphenated_parts_are_recased_one_by_one(self):
        assert keyphrases.recase("qwen3-omni", {"qwen3": "Qwen3"}) == "Qwen3-omni"


class TestRankTerms:
    def test_the_weight_is_class_based_tf_idf_on_record_counts(self):
        terms = [{"a", "b"}, {"a"}, {"b"}, {"b"}]
        groups = [0, 0, 1, 1]
        ranked = keyphrases.rank_terms(terms, groups, min_support=1)
        average = (3 + 2) / 2  # term occurrences per group
        weight_a = 2 * np.log(1 + average / 2)
        assert ranked[0][0][0] == "a"
        assert ranked[0][0][1] == pytest.approx(weight_a)
        assert ranked[0][0][2] == 1.0  # both of group 0's records hold "a"

    def test_min_support_excludes_a_term_in_too_few_of_a_groups_records(self):
        terms = [{"a", "rare"}, {"a"}]
        ranked = keyphrases.rank_terms(terms, [0, 0], min_support=2)
        assert [t for t, _, _ in ranked[0]] == ["a"]

    def test_a_record_in_no_group_counts_toward_the_corpus_only(self):
        terms = [{"x"}, {"x"}, {"x"}]
        with_outlier = keyphrases.rank_terms(terms, [0, 0, None], min_support=1)
        without = keyphrases.rank_terms(terms[:2], [0, 0], min_support=1)
        assert set(with_outlier) == {0}
        assert with_outlier[0][0][1] < without[0][0][1]  # x is commoner in the corpus, so it weighs less

    def test_ties_break_by_term(self):
        ranked = keyphrases.rank_terms([{"b", "a"}, {"b", "a"}], [0, 0])
        assert [t for t, _, _ in ranked[0]] == ["a", "b"]


class TestUmbrellaTerms:
    def test_a_term_present_in_enough_groups_is_umbrella(self):
        # 10 groups: the threshold is max(min(5, 10), ceil(0.1 * 10)) = 5.
        terms, groups = [], []
        for g in range(10):
            terms += [{"everywhere", f"own{g}"}, {"everywhere", f"own{g}"}]
            groups += [g, g]
        assert keyphrases.umbrella_terms(terms, groups) == {"everywhere"}

    def test_a_single_group_has_no_umbrella_terms(self):
        """With one group, the threshold is one group, so every term would be "too common" without this."""
        assert keyphrases.umbrella_terms([{"a"}, {"a"}], [0, 0]) == set()

    def test_presence_needs_min_support(self):
        """Once in each of ten groups is not presence at `min_support=2`."""
        terms = [{"thin"} for _ in range(10)]
        assert keyphrases.umbrella_terms(terms, list(range(10))) == set()


class TestFuse:
    def test_a_term_fair_in_both_beats_one_high_in_only_one(self):
        first = [("interest",), ("x",), ("both",), ("y",)]
        second = [("z",), ("both",), ("w",), ("v",)]
        assert keyphrases.fuse([first, second])[0] == ("both",)

    def test_a_single_ranking_is_kept(self):
        """Negative control: with nothing to fuse, the order is the ranking's own."""
        ranking = [("c",), ("a",), ("b",)]
        assert keyphrases.fuse([ranking]) == ranking


def _toy_corpus():
    """Two groups of three records, each with its own topic, sharing one generic term."""
    terms = [{"deep learning", "learning", "neural network", "challenge"},
             {"deep learning", "learning", "neural network", "challenge"},
             {"deep learning", "learning", "sgd"},
             {"diffusion model", "model", "image generation", "challenge"},
             {"diffusion model", "model", "image generation", "challenge"},
             {"diffusion model", "model", "noise"}]
    groups = [0, 0, 0, 1, 1, 1]
    return terms, groups


def _embedder(dim=8, seed=0):
    """A deterministic stand-in embedder: each term gets a fixed random unit vector."""
    rng = np.random.default_rng(seed)
    table = {}

    def embed(terms):
        for t in terms:
            if t not in table:
                table[t] = rng.standard_normal(dim)
        return np.stack([table[t] for t in terms])
    return embed


class TestLabelGroups:
    def _labels(self, **kwargs):
        terms, groups = _toy_corpus()
        vectors = np.random.default_rng(1).standard_normal((len(terms), 8))
        return keyphrases.label_groups(terms, groups, display={}, record_vectors=vectors, embed_terms=_embedder(),
                                       **kwargs)

    def test_each_group_gets_its_own_terms(self):
        labels = self._labels()
        assert {t for t, _ in labels[0]} >= {"deep learning", "neural network"}
        assert {t for t, _ in labels[1]} >= {"diffusion model", "image generation"}

    def test_nested_terms_count_once(self):
        labels = self._labels()
        assert not ({"deep learning", "learning"} <= {t for t, _ in labels[0]})
        assert not ({"diffusion model", "model"} <= {t for t, _ in labels[1]})

    def test_an_excluded_term_is_never_given(self):
        labels = self._labels()
        assert all("challenge" not in [t for t, _ in label] for label in labels.values())  # a bare-keyword stopword

    def test_shares_are_of_the_group(self):
        labels = self._labels()
        assert dict(labels[0])["deep learning"] == 1.0

    def test_display_forms_are_used(self):
        terms, groups = _toy_corpus()
        vectors = np.ones((len(terms), 8))
        labels = keyphrases.label_groups(terms, groups, display={"deep learning": "Deep Learning"},
                                         record_vectors=vectors, embed_terms=_embedder())
        assert "Deep Learning" in [t for t, _ in labels[0]]

    def test_top_caps_the_list(self):
        labels = self._labels(top=2, shown=2)
        assert all(len(label) <= 2 for label in labels.values())

    def test_top_zero_gives_nothing(self):
        assert all(label == [] for label in self._labels(top=0).values())

    def test_the_list_begins_with_the_label_picked_alone(self):
        """A longer nested phrase further down must not rewrite the first `shown` terms."""
        # "alpha" is the head. Of the tail, "sampling" ranks first both ways (it points along the centroid),
        # and "latin hypercube sampling" covers enough of the group to replace it, were it allowed to.
        terms = [{"alpha", "sampling", "latin hypercube sampling"}, {"alpha", "sampling", "latin hypercube sampling"},
                 {"alpha", "sampling", "beta"}, {"alpha", "beta"},
                 {"other"}, {"other"}]
        groups = [0, 0, 0, 0, 1, 1]
        vectors = np.zeros((6, 4))
        vectors[:, 0] = 1.0

        def embed(ts):
            return np.array([[1.0, 0, 0, 0] if t == "sampling" else [0, 1.0, 0, 0] for t in ts])
        common = dict(display={}, record_vectors=vectors, embed_terms=embed, min_support=1)
        alone = keyphrases.label_groups(terms, groups, top=2, shown=2, **common)
        assert [t for t, _ in alone[0]] == ["alpha", "sampling"], "this fixture no longer puts the shorter phrase first"
        replaced = keyphrases.label_groups(terms, groups, top=30, shown=30, **common)
        assert "latin hypercube sampling" in [t for t, _ in replaced[0]][:2], (
            "the longer phrase does not replace the shorter even with everything picked at once, so this fixture "
            "cannot tell a rewritten label from a kept one")
        longer = keyphrases.label_groups(terms, groups, top=30, shown=2, **common)
        assert longer[0][:2] == alone[0]


class TestTopTerms:
    def test_terms_are_counted_by_record(self):
        assert keyphrases.top_terms([{"a", "b"}, {"a"}, {"a", "c"}], 10) == [("a", 3), ("b", 1), ("c", 1)]

    def test_a_dominant_longer_phrase_stands_in_for_the_shorter(self):
        records = [{"large language model", "language model"}] * 20 + [{"small language model", "language model"}] * 5
        terms = dict(keyphrases.top_terms(records, 10))
        assert "language model" not in terms
        assert terms == {"large language model": 20, "small language model": 5}

    def test_a_minor_longer_phrase_leaves_the_shorter(self):
        """Negative control for the test above: the longer form in under half the shorter's records."""
        records = [{"small language model", "language model"}] * 5 + [{"language model"}] * 20
        terms = dict(keyphrases.top_terms(records, 10))
        assert terms["language model"] == 25

    def test_excluded_terms_are_left_out(self):
        assert keyphrases.top_terms([{"challenge", "x"}] * 2, 10) == [("x", 2)]

    def test_top_caps_the_list(self):
        assert len(keyphrases.top_terms([{f"t{i}" for i in range(50)}], 7)) == 7


class TestWithoutUmbrellaTerms:
    def test_a_term_most_lists_share_goes_from_all_of_them(self):
        lists = [["Large Language Models", f"topic {i}"] for i in range(6)] + [["large language model", "other"]]
        out = keyphrases.without_umbrella_terms(lists)
        assert out == [[f"topic {i}"] for i in range(6)] + [["other"]]

    def test_a_term_few_lists_share_stays(self):
        """Control for the test above: the same term in two lists of seven is not shared widely enough."""
        lists = [["large language models", "a"], ["large language models", "b"]] + [[f"t{i}"] for i in range(5)]
        assert keyphrases.without_umbrella_terms(lists)[:2] == [["large language models", "a"],
                                                                ["large language models", "b"]]


class TestLeadWith:
    def test_an_abbreviation_shown_for_a_long_form_is_a_repeat(self):
        ranked = [("lora", 0.5), ("gpu", 0.4)]
        display = {"low-rank adaptation": "lora"}
        assert keyphrases.lead_with(["low-rank adaptation"], ranked, 3, display=display) == ["low-rank adaptation", "gpu"]
        assert keyphrases.lead_with(["LoRA"], [("low-rank adaptation", 0.5)], 3,
                                    display={"low-rank adaptation": "low-rank adaptation", "lora": "lora"}) == \
            ["LoRA", "low-rank adaptation"], "with no mapping between them, the two are different terms"

    def test_without_the_display_map_the_two_forms_are_not_matched(self):
        """Negative control for the test above."""
        assert keyphrases.lead_with(["low-rank adaptation"], [("lora", 0.5)], 3) == ["low-rank adaptation", "lora"]

    def test_the_leading_terms_come_first_and_the_ranking_fills_the_rest(self):
        assert keyphrases.lead_with(["AI safety"], [("prompt injection", 0.4), ("jailbreak", 0.3)], 3) == \
            ["AI safety", "prompt injection", "jailbreak"]

    def test_a_repeat_is_skipped_whatever_its_case_hyphens_or_plural(self):
        ranked = [("reinforcement learnings", 0.9), ("Reinforcement-Learning", 0.8), ("reward model", 0.5)]
        assert keyphrases.lead_with(["Reinforcement Learning"], ranked, 3) == ["Reinforcement Learning", "reward model"]

    def test_a_nested_term_is_skipped_either_way(self):
        ranked = [("deep reinforcement learning", 0.6), ("evaluation", 0.5), ("benchmark", 0.4)]
        assert keyphrases.lead_with(["reinforcement learning", "LLM evaluation"], ranked, 4) == \
            ["reinforcement learning", "LLM evaluation", "benchmark"]

    def test_top_caps_the_leading_terms_too(self):
        assert keyphrases.lead_with(["a", "b", "c"], [("d", 0.5)], 2) == ["a", "b"]

    def test_with_nothing_leading_it_is_the_ranking(self):
        """Negative control: the merge adds nothing of its own when there is nothing to lead."""
        assert keyphrases.lead_with([], [("x", 0.5), ("y", 0.4)], 6) == ["x", "y"]


@pytest.fixture(scope="module")
def nlp():
    spacy = pytest.importorskip("spacy")
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy's en_core_web_sm is not installed")


@pytest.fixture(scope="module")
def stopwords():
    from raven.common.text import stopwords as raven_stopwords
    nlptools = pytest.importorskip("raven.common.nlptools")
    return set(nlptools.default_stopwords) | set(raven_stopwords.academic_stopwords)


@pytest.mark.ml
class TestExtract:
    def test_a_noun_qualified_stopword_head_stays(self, nlp, stopwords):
        phrases = keyphrases.extract(nlp("We measure tool use in agents."), stopwords=stopwords)["phrases"]
        assert "tool use" in phrases

    def test_an_adjective_qualified_stopword_head_goes(self, nlp, stopwords):
        phrases = keyphrases.extract(nlp("Models show a remarkable ability."), stopwords=stopwords)["phrases"]
        assert not any(p.endswith("ability") for p in phrases)

    def test_the_possessive_is_dropped(self, nlp, stopwords):
        phrases = keyphrases.extract(nlp("We report GPT-4's performance."), stopwords=stopwords)["phrases"]
        assert any("performance" in p for p in phrases)
        assert not any("'s" in p for p in phrases)

    def test_a_subphrase_starting_with_a_number_is_skipped(self, nlp, stopwords):
        phrases = keyphrases.extract(nlp("It is released under the Apache 2.0 license."),
                                     stopwords=stopwords)["phrases"]
        assert "2.0 license" not in phrases

    def test_a_stopword_starting_a_hyphenated_compound_is_kept(self, nlp, stopwords):
        phrases = keyphrases.extract(nlp("We allocate test-time computation adaptively."), stopwords=stopwords)["phrases"]
        assert "test" in stopwords or "test" in {w.lower() for w in stopwords}, \
            "'test' is no longer a stopword, so this fixture tests nothing"
        assert "test-time computation" in phrases
        assert "time computation" not in phrases

    def test_physics_is_not_singularized(self, nlp, stopwords):
        phrases = keyphrases.extract(nlp("This paper concerns theoretical physics."), stopwords=stopwords)["phrases"]
        assert "physics" in phrases or "theoretical physics" in phrases
        assert not any(p.split()[-1] == "physic" for p in phrases)

    def test_casing_records_a_proper_name_and_skips_the_sentence_start(self, nlp, stopwords):
        doc = nlp("Induction is hard. We study Solomonoff induction with LoRA adapters.")
        casing = keyphrases.extract(doc, stopwords=stopwords)["casing"]
        assert casing.get("solomonoff") == "Solomonoff"
        assert casing.get("induction") == "induction", "the sentence-initial 'Induction' must not count"

    def test_an_abbreviation_definition_is_found(self, nlp, stopwords):
        doc = nlp("We use self-regulated learning (SRL) in class.")
        assert ("srl", "self-regulated learning") in keyphrases.extract(doc, stopwords=stopwords)["definitions"]

    def test_a_lowercase_word_in_parentheses_is_not_an_abbreviation(self, nlp, stopwords):
        doc = nlp("The gas evolved at the anode (oxygen) was measured.")
        assert keyphrases.extract(doc, stopwords=stopwords)["definitions"] == []
