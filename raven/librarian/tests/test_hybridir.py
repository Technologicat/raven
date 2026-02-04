"""Integration tests for raven.librarian.hybridir (hybrid semantic + keyword search).

Exercises indexing and querying of the HybridIR search engine with a small
corpus of AI paper abstracts. The embedding model is loaded locally (no
raven-server required), but the first run may be slow if the model is not
yet cached.
"""

import textwrap

import pytest

from raven.librarian import hybridir


# ---------------------------------------------------------------------------
# Test corpus — a few AI paper abstracts from arXiv
# ---------------------------------------------------------------------------

DOCS = {
    "arxiv_abstract_1": textwrap.dedent("""
        SCALING LAWS FOR A MULTI-AGENT REINFORCEMENT LEARNING MODEL

        Oren Neumann & Claudius Gros (2023)

        The recent observation of neural power-law scaling relations has made a signifi-
        cant impact in the field of deep learning. A substantial amount of attention has
        been dedicated as a consequence to the description of scaling laws, although
        mostly for supervised learning and only to a reduced extent for reinforcement
        learning frameworks. In this paper we present an extensive study of performance
        scaling for a cornerstone reinforcement learning algorithm, AlphaZero. On the ba-
        sis of a relationship between Elo rating, playing strength and power-law scaling,
        we train AlphaZero agents on the games Connect Four and Pentago and analyze
        their performance. We find that player strength scales as a power law in neural
        network parameter count when not bottlenecked by available compute, and as a
        power of compute when training optimally sized agents. We observe nearly iden-
        tical scaling exponents for both games. Combining the two observed scaling laws
        we obtain a power law relating optimal size to compute similar to the ones ob-
        served for language models. We find that the predicted scaling of optimal neural
        network size fits our data for both games. We also show that large AlphaZero
        models are more sample efficient, performing better than smaller models with the
        same amount of training data.""").strip(),

    "arxiv_abstract_2": textwrap.dedent("""
        A Generalist Agent

        Scott Reed et al. (2022)

        Inspired by progress in large-scale language modeling, we apply a similar approach towards
        building a single generalist agent beyond the realm of text outputs. The agent, which we
        refer to as Gato, works as a multi-modal, multi-task, multi-embodiment generalist policy.
        The same network with the same weights can play Atari, caption images, chat, stack blocks
        with a real robot arm and much more, deciding based on its context whether to output text,
        joint torques, button presses, or other tokens. In this report we describe the model and the
        data, and document the current capabilities of Gato.
        """).strip(),

    "arxiv_abstract_3": textwrap.dedent("""
        Unleashing the Emergent Cognitive Synergy in Large Language Models:
        A Task-Solving Agent through Multi-Persona Self-Collaboration

        Zhenhailong Wang et al. (2023)

        Human intelligence thrives on cognitive syn-
        ergy, where collaboration among different
        minds yield superior outcomes compared to iso-
        lated individuals. In this work, we propose Solo
        Performance Prompting (SPP), which trans-
        forms a single LLM into a cognitive synergist
        by engaging in multi-turn self-collaboration
        with multiple personas. A cognitive syner-
        gist is an intelligent agent that collaboratively
        combines multiple minds' strengths and knowl-
        edge to enhance problem-solving in complex
        tasks. By dynamically identifying and simu-
        lating different personas based on task inputs,
        SPP unleashes the potential of cognitive syn-
        ergy in LLMs. Our in-depth analysis shows
        that assigning multiple fine-grained personas
        in LLMs improves problem-solving abilities
        compared to using a single or fixed number
        of personas. We evaluate SPP on three chal-
        lenging tasks: Trivia Creative Writing, Code-
        names Collaborative, and Logic Grid Puzzle,
        encompassing both knowledge-intensive and
        reasoning-intensive types. Unlike previous
        works, such as Chain-of-Thought, that solely
        enhance the reasoning abilities in LLMs, ex-
        perimental results demonstrate that SPP effec-
        tively reduces factual hallucination, and main-
        tains strong reasoning capabilities. Addition-
        ally, comparative experiments show that cog-
        nitive synergy only emerges in GPT-4 and
        does not appear in less capable models, such
        as GPT-3.5-turbo and Llama2-13b-chat, which
        draws an interesting analogy to human devel-
        opment. Code, data, and prompts can be found
        at: https://github.com/MikeWangWZHL/
        Solo-Performance-Prompting.git
        """).strip(),

    "arxiv_abstract_4": textwrap.dedent("""
        AI Agents That Matter

        Sayash Kapoor et al. (2024)

        AI agents are an exciting new research direction, and agent development is driven
        by benchmarks. Our analysis of current agent benchmarks and evaluation practices
        reveals several shortcomings that hinder their usefulness in real-world applications.
        First, there is a narrow focus on accuracy without attention to other metrics. As
        a result, SOTA agents are needlessly complex and costly, and the community has
        reached mistaken conclusions about the sources of accuracy gains. Our focus on
        cost in addition to accuracy motivates the new goal of jointly optimizing the two
        metrics. We design and implement one such optimization, showing its potential
        to greatly reduce cost while maintaining accuracy. Second, the benchmarking
        needs of model and downstream developers have been conflated, making it hard
        to identify which agent would be best suited for a particular application. Third,
        many agent benchmarks have inadequate holdout sets, and sometimes none at all.
        This has led to agents that are fragile because they take shortcuts and overfit to the
        benchmark in various ways. We prescribe a principled framework for avoiding
        overfitting. Finally, there is a lack of standardization in evaluation practices, leading
        to a pervasive lack of reproducibility. We hope that the steps we introduce for
        addressing these shortcomings will spur the development of agents that are useful
        in the real world and not just accurate on benchmarks.
        """).strip(),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    """A committed HybridIR instance with the test corpus indexed.

    Module-scoped so the embedding model is loaded only once.
    """
    datastore_dir = tmp_path_factory.mktemp("hybridir_test")
    ret = hybridir.HybridIR(datastore_base_dir=datastore_dir,
                             embedding_model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1",
                             local_model_loader_fallback=True)
    for doc_id, doc_text in DOCS.items():
        ret.add(document_id=doc_id, path="<test>", text=doc_text)
    ret.commit()
    return ret


# ---------------------------------------------------------------------------
# Document storage
# ---------------------------------------------------------------------------

class TestDocumentStorage:
    def test_all_documents_stored(self, retriever):
        assert set(retriever.documents.keys()) == set(DOCS.keys())

    def test_document_count(self, retriever):
        assert len(retriever.documents) == len(DOCS)

    def test_stored_text_matches_input(self, retriever):
        for doc_id, doc_text in DOCS.items():
            assert retriever.documents[doc_id]["text"] == doc_text


# ---------------------------------------------------------------------------
# Query result structure
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_result_is_list_of_dicts(self, retriever):
        results = retriever.query("ai agents", k=5)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, dict)

    def test_result_fields(self, retriever):
        results = retriever.query("ai agents", k=5)
        assert len(results) > 0
        for r in results:
            assert "document_id" in r
            assert "text" in r
            assert "offset" in r
            assert "score" in r

    def test_extra_info_shape(self, retriever):
        results, (kw_results, kw_scores), (vec_results, vec_distances) = retriever.query(
            "ai agents", k=5, return_extra_info=True)
        assert isinstance(results, list)
        assert isinstance(kw_results, list)
        assert isinstance(kw_scores, list)
        assert len(kw_results) == len(kw_scores)
        assert isinstance(vec_results, list)
        assert isinstance(vec_distances, list)
        assert len(vec_results) == len(vec_distances)


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------

class TestKeywordSearch:
    def test_relevant_query_returns_results(self, retriever):
        _results, (kw_results, _kw_scores), _vec = retriever.query(
            "ai agents", k=5, return_extra_info=True)
        assert len(kw_results) > 0

    def test_unrelated_query_returns_few_or_no_results(self, retriever):
        _results, (kw_results, _kw_scores), _vec = retriever.query(
            "quantum physics", k=5,
            keyword_score_threshold=0.1,
            return_extra_info=True)
        # "quantum physics" doesn't appear in any document.
        assert len(kw_results) == 0

    def test_nonsense_returns_no_results(self, retriever):
        _results, (kw_results, _kw_scores), _vec = retriever.query(
            "blurba zaaaarrrgh blah qwertyuiop", k=5,
            keyword_score_threshold=0.1,
            return_extra_info=True)
        assert len(kw_results) == 0


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

class TestSemanticSearch:
    def test_relevant_query_returns_results(self, retriever):
        _results, _kw, (vec_results, _vec_distances) = retriever.query(
            "ai agents", k=5,
            semantic_distance_threshold=0.8,
            return_extra_info=True)
        assert len(vec_results) > 0

    def test_related_topic_returns_results(self, retriever):
        _results, _kw, (vec_results, _vec_distances) = retriever.query(
            "language models", k=5,
            semantic_distance_threshold=0.8,
            return_extra_info=True)
        assert len(vec_results) > 0

    def test_unrelated_topic_returns_few_or_no_results(self, retriever):
        _results, _kw, (vec_results, _vec_distances) = retriever.query(
            "quantum physics", k=5,
            semantic_distance_threshold=0.8,
            return_extra_info=True)
        # May return zero or very few; all should have high distance.
        if vec_results:
            for dist in _vec_distances:
                assert dist > 0.5  # anything returned should be weakly related at best


# ---------------------------------------------------------------------------
# Hybrid (combined) search
# ---------------------------------------------------------------------------

class TestHybridSearch:
    def test_relevant_query_ranks_related_doc_high(self, retriever):
        """Querying "ai agents" should return the "AI Agents That Matter" paper near the top."""
        results = retriever.query("ai agents", k=5)
        assert len(results) > 0
        top_doc_ids = [r["document_id"] for r in results[:2]]
        assert "arxiv_abstract_4" in top_doc_ids

    def test_llm_query_returns_results(self, retriever):
        """Querying "llms" or "language models" should return the LLM-related papers."""
        results = retriever.query("language models", k=5)
        assert len(results) > 0
        doc_ids = {r["document_id"] for r in results}
        # At least one of the LLM-related papers should appear.
        assert doc_ids & {"arxiv_abstract_3", "arxiv_abstract_2"}

    def test_completely_unrelated_returns_nothing(self, retriever):
        results = retriever.query("can cats jump", k=5,
                                  keyword_score_threshold=0.1,
                                  semantic_distance_threshold=0.8)
        # With strict thresholds, completely unrelated queries should return nothing.
        assert len(results) == 0

    def test_nonsense_returns_nothing(self, retriever):
        results = retriever.query("blurba zaaaarrrgh blah qwertyuiop", k=5,
                                  keyword_score_threshold=0.1,
                                  semantic_distance_threshold=0.8)
        assert len(results) == 0

    def test_result_text_comes_from_correct_document(self, retriever):
        """The result text should be a substring of the document it claims to come from."""
        results = retriever.query("ai agents", k=5)
        for r in results:
            doc = retriever.documents[r["document_id"]]
            assert r["text"] in doc["text"]

    def test_result_offset_is_consistent(self, retriever):
        """The offset should point to the correct position in the source document."""
        results = retriever.query("reinforcement learning", k=5)
        for r in results:
            doc = retriever.documents[r["document_id"]]
            offset = r["offset"]
            assert doc["text"][offset:offset + len(r["text"])] == r["text"]
