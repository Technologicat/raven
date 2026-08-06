"""Integration tests for raven.librarian.hybridir (hybrid semantic + keyword search).

Exercises indexing and querying of the HybridIR search engine with a small
corpus of AI paper abstracts. The embedding model is loaded locally (no
raven-server required), but the first run may be slow if the model is not
yet cached.
"""

import math
import pathlib
import textwrap
import threading
import types

import pytest

chromadb = pytest.importorskip("chromadb", reason="chromadb not installed (needs full dependency stack)")
bm25s = pytest.importorskip("bm25s", reason="bm25s not installed (needs full dependency stack)")
# The two above are named because the tests use them directly. This one guards the import below: `hybridir`
# pulls more than those two (spaCy among them), so guarding only on named packages leaves the skip depending
# on which dependency happens to be missing first, rather than on whether the module can be imported at all.
pytest.importorskip("raven.librarian.hybridir", reason="full dependency stack not installed")

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

    def test_a_rambling_message_retrieves_through_the_multi_query_path(self, retriever):
        # The short queries above never split, so they never exercise the per-query indexing that the
        # multi-query fusion introduced (`raw_keyword_results[i, j]`, and Chroma's per-query result lists).
        # This message does split, which is the point of it.
        rambling = ("I have been reading around this area for a while now and there is a lot of it. "
                    "My supervisor suggested I look into the practical side of things. "
                    "What do these papers say about agents built on language models?")
        assert hybridir.split_into_subqueries(rambling)  # guard: if this stops splitting, the test stops testing
        results = retriever.query(rambling, k=5, multi_query=True)  # explicit: the default is off
        assert len(results) > 0
        for r in results:
            assert "document_id" in r and "text" in r and "score" in r

    def test_the_multi_query_path_can_be_turned_off(self, retriever):
        # The evaluation harness needs both sides of the comparison from one code path.
        rambling = ("I have been reading around this area for a while now and there is a lot of it. "
                    "What do these papers say about agents built on language models?")
        assert len(retriever.query(rambling, k=5, multi_query=False)) > 0

    def test_extra_info_shape_survives_a_split_query(self, retriever):
        # The extra-info lists are the flat union across subqueries, so the per-entry pairing between a
        # result and its score has to survive the flattening.
        rambling = ("I have been reading around this area for a while now and there is a lot of it. "
                    "What do these papers say about agents built on language models?")
        results, report = retriever.query(
            rambling, k=5, multi_query=True, return_extra_info=True)  # explicit: the default is off
        assert len(report.keyword_results) == len(report.keyword_scores)
        assert len(report.vector_results) == len(report.vector_distances)
        assert len(results) > 0

    def test_an_empty_index_still_returns_the_pair_when_extra_info_was_asked_for(self, tmp_path):
        # An empty index is an ordinary state, not an error: `HybridIR` creates its datastore directory
        # rather than rejecting a path that does not exist yet, so anyone who mistypes a `--db-dir` gets
        # one of these. Returning a bare list here would raise `ValueError: not enough values to unpack`
        # at the call site — a report about the caller's tuple, naming nothing about the empty corpus.
        empty = hybridir.HybridIR(datastore_base_dir=tmp_path / "empty",
                                  embedding_model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1",
                                  local_model_loader_fallback=True)
        assert not empty.documents  # guard: if this ever indexes something, the test stops testing

        results, report = empty.query("ai agents", k=5, return_extra_info=True)
        assert results == []
        assert report.keyword_results == []
        assert report.vector_results == []
        assert len(report.per_query) == 1  # the whole query, `multi_query` being off
        assert report.per_query[0].candidate_keyword_scores == []

        assert empty.query("ai agents", k=5) == []  # and the plain shape is unchanged

    def test_extra_info_shape(self, retriever):
        results, report = retriever.query("ai agents", k=5, return_extra_info=True)
        assert isinstance(results, list)
        assert isinstance(report.keyword_results, list)
        assert isinstance(report.keyword_scores, list)
        assert len(report.keyword_results) == len(report.keyword_scores)
        assert isinstance(report.vector_results, list)
        assert isinstance(report.vector_distances, list)
        assert len(report.vector_results) == len(report.vector_distances)

    def test_the_report_breaks_the_candidate_scores_down_per_query(self, retriever):
        # The sharpness signal is a per-query reading, so the flat unions above cannot carry it: they merge
        # every subquery's candidates into one list, which is exactly the distinction being measured.
        rambling = ("I have been reading around this area for a while now and there is a lot of it. "
                    "What do these papers say about agents built on language models?")
        _results, report = retriever.query(rambling, k=5, multi_query=True, return_extra_info=True)
        assert len(report.per_query) == 1 + len(hybridir.split_into_subqueries(rambling))
        assert report.per_query[0].text == rambling  # the whole message is always queried, and always first
        for entry in report.per_query:
            assert entry.candidate_keyword_scores and entry.candidate_vector_distances

    def test_the_candidate_scores_are_not_thresholded(self, retriever):
        # `score_sharpness` needs the population the threshold cut from, not what survived it: a candidate the
        # threshold rejected is precisely a candidate the best result left behind. Ask with a threshold high
        # enough to admit nothing, and the candidates must still be there.
        _results, report = retriever.query("ai agents", k=5,
                                           keyword_score_threshold=1e9,
                                           semantic_distance_threshold=-1.0,
                                           return_extra_info=True)
        assert not report.keyword_results and not report.vector_results
        assert report.per_query[0].candidate_keyword_scores
        assert report.per_query[0].candidate_vector_distances


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------

class TestKeywordSearch:
    def test_relevant_query_returns_results(self, retriever):
        _results, report = retriever.query("ai agents", k=5, return_extra_info=True)
        assert len(report.keyword_results) > 0

    def test_unrelated_query_returns_few_or_no_results(self, retriever):
        _results, report = retriever.query(
            "quantum physics", k=5,
            keyword_score_threshold=0.1,
            return_extra_info=True)
        # "quantum physics" doesn't appear in any document.
        assert len(report.keyword_results) == 0

    def test_nonsense_returns_no_results(self, retriever):
        _results, report = retriever.query(
            "blurba zaaaarrrgh blah qwertyuiop", k=5,
            keyword_score_threshold=0.1,
            return_extra_info=True)
        assert len(report.keyword_results) == 0


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

class TestSemanticSearch:
    def test_relevant_query_returns_results(self, retriever):
        _results, report = retriever.query(
            "ai agents", k=5,
            semantic_distance_threshold=0.8,
            return_extra_info=True)
        assert len(report.vector_results) > 0

    def test_related_topic_returns_results(self, retriever):
        _results, report = retriever.query(
            "language models", k=5,
            semantic_distance_threshold=0.8,
            return_extra_info=True)
        assert len(report.vector_results) > 0

    def test_unrelated_topic_returns_few_or_no_results(self, retriever):
        _results, report = retriever.query(
            "quantum physics", k=5,
            semantic_distance_threshold=0.8,
            return_extra_info=True)
        # May return zero or very few; all should have high distance.
        if report.vector_results:
            for dist in report.vector_distances:
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


# ---------------------------------------------------------------------------
# Pending-edit collapse (`_pend_edit`) — unit-level, no real index needed
# ---------------------------------------------------------------------------

def _fake_ir_for_pend_edit(indexed_document_ids=()):
    """A minimal stand-in exposing just what `_pend_edit` touches, to unit-test its collapse logic.

    Avoids constructing a real `HybridIR` (which would load an embedding model and open the vector store).
    `indexed_document_ids` seeds the *committed* index membership that `_pend_edit` consults to decide whether
    an update needs a preceding delete.
    """
    fake = types.SimpleNamespace()
    fake._pending_edits = []
    fake._pending_edits_lock = threading.RLock()
    fake.documents = {doc_id: {"document_id": doc_id} for doc_id in indexed_document_ids}
    fake._stat = lambda path: {"size": 0, "mtime": 0.0}
    return fake


def _pending_kinds(fake):
    return [kind for (kind, _data) in fake._pending_edits]


# ---------------------------------------------------------------------------
# Indexing progress reporting — unit-level, no real index needed
# ---------------------------------------------------------------------------

def _fake_ir_for_prepare(chunk_size=1000, overlap=250):
    """A minimal stand-in exposing just what `_prepare_document_for_indexing` touches.

    Avoids constructing a real `HybridIR`, which would load an embedding model and reach the server for
    spaCy tokenization. Both slow steps are stubbed; what is under test is when progress gets reported.
    """
    fake = types.SimpleNamespace()
    fake.chunk_size = chunk_size
    fake.overlap = overlap
    fake._tokenize = lambda text: text.lower().split()
    fake._tokenize_many = lambda texts: [text.lower().split() for text in texts]
    fake.embedder = types.SimpleNamespace(encode=lambda texts: _FakeEmbeddings(len(texts)))
    return fake


class _FakeEmbeddings:
    def __init__(self, n):
        self.n = n

    def tolist(self):
        return [[0.0] for _ in range(self.n)]


def test_indexing_progress_keeps_moving_while_tokenizing():
    # A per-document update was fine when a document was a 1.3 kB abstract. On a 216 kB story it leaves the
    # indicator unchanged for tens of seconds while tokenizing, and a frozen indicator reads as a hung job
    # rather than a slow one — the user cannot tell the difference.
    #
    # What is pinned is that the report advances *during* tokenization and ends at the chunk count, not the
    # granularity: tokenization is sent in batches of `TOKENIZE_BATCH_SIZE`, so one report per chunk would
    # mean one round trip per chunk, which is the cost that batching exists to remove.
    fake = _fake_ir_for_prepare()
    text = "word " * 20000  # ~100k characters, so comfortably many chunks
    seen = []
    prepared = hybridir.HybridIR._prepare_document_for_indexing(
        fake, {"document_id": "big.txt", "text": text}, on_progress=seen.append)

    n_chunks = len(prepared["chunks"])
    assert n_chunks > hybridir.TOKENIZE_BATCH_SIZE  # guard: too few chunks and this stops testing anything
    tokenizing = [s for s in seen if s.startswith("tokenizing")]
    expected_reports = math.ceil(n_chunks / hybridir.TOKENIZE_BATCH_SIZE)
    assert len(tokenizing) == expected_reports
    assert expected_reports > 1  # it must advance, not report once at the end
    assert tokenizing[-1] == f"tokenizing {n_chunks} / {n_chunks}"
    assert seen[-1] == f"embedding {n_chunks} chunks"  # one report: it is 3% of the work


def test_preparing_a_document_without_a_progress_callback_still_works():
    # The callback is optional, and the full index rebuild path does not pass one.
    fake = _fake_ir_for_prepare()
    prepared = hybridir.HybridIR._prepare_document_for_indexing(
        fake, {"document_id": "small.txt", "text": "word " * 500})
    assert prepared["chunks"] and prepared["tokens"] and prepared["embeddings"]


def test_indexing_progress_line_omits_an_empty_detail():
    # A deletion has no inner steps to report, and an empty field would show as a stray separator.
    fake = types.SimpleNamespace(_indexing_progress_text="")
    eta = types.SimpleNamespace(formatted_eta="elapsed 6s, ETA 01:14, total 01:20")
    hybridir.HybridIR._set_indexing_progress(fake, 14, 186, "paper.bib", eta)
    assert fake._indexing_progress_text == "[14 / 186] | paper.bib | elapsed 6s, ETA 01:14, total 01:20"
    hybridir.HybridIR._set_indexing_progress(fake, 14, 186, "paper.bib", eta, "tokenizing 240 / 288")
    assert fake._indexing_progress_text == ("[14 / 186] | paper.bib | tokenizing 240 / 288 | "
                                            "elapsed 6s, ETA 01:14, total 01:20")


def test_pend_edit_new_file_add_then_modify_stays_single_add():
    # A brand-new file: watchdog fires create (-> add) then modify (-> update). The update must not queue a
    # delete for a document that was never indexed, or the commit no-ops it with a KeyError and the change count
    # reads 2 instead of 1.
    fake = _fake_ir_for_pend_edit(indexed_document_ids=())
    hybridir.HybridIR._pend_edit(fake, action="add", document_id="doc1", path="/docs/doc1.pdf", text="hello")
    hybridir.HybridIR._pend_edit(fake, action="update", document_id="doc1", path="/docs/doc1.pdf", text="hello")
    assert _pending_kinds(fake) == ["add"]


def test_pend_edit_new_file_event_flurry_collapses_to_single_add():
    # The exact ordering observed for a large new file (several modify events interleaved with the create):
    # update, update, add, update. It must still collapse to one clean add.
    fake = _fake_ir_for_pend_edit(indexed_document_ids=())
    for action in ("update", "update", "add", "update"):
        hybridir.HybridIR._pend_edit(fake, action=action, document_id="doc1", path="/docs/doc1.pdf", text="t")
    assert _pending_kinds(fake) == ["add"]


def test_pend_edit_update_of_indexed_document_is_delete_then_add():
    # A genuine update of an already-committed document still replaces it: delete, then add.
    fake = _fake_ir_for_pend_edit(indexed_document_ids=("doc1",))
    hybridir.HybridIR._pend_edit(fake, action="update", document_id="doc1", path="/docs/doc1.pdf", text="new")
    assert _pending_kinds(fake) == ["delete", "add"]


def test_pend_edit_delete_of_indexed_document():
    fake = _fake_ir_for_pend_edit(indexed_document_ids=("doc1",))
    hybridir.HybridIR._pend_edit(fake, action="delete", document_id="doc1")
    assert _pending_kinds(fake) == ["delete"]


# ---------------------------------------------------------------------------
# Weighted reciprocal rank fusion
# ---------------------------------------------------------------------------

class TestWeightedRRF:
    """The arms' relative vote is a knob, because the right value belongs to the collection.

    Measured across four evaluation corpora it runs from about 0.1 (a bibliography of bare titles, where
    BM25 has a dozen words to match on) to about 0.6 (scientific abstracts). Blending equally costs a
    corpus whose arms are that unequal.
    """

    def test_equal_weights_match_the_unweighted_default(self):
        a, b = ["x", "y", "z"], ["z", "y", "w"]
        assert (hybridir.reciprocal_rank_fusion(a, b, weights=[1.0, 1.0]) ==
                hybridir.reciprocal_rank_fusion(a, b))

    def test_only_the_ratio_matters(self):
        # RRF scores are compared against each other, so scaling every weight leaves the order alone.
        a, b = ["x", "y", "z"], ["z", "y", "w"]
        order = [item for item, _score in hybridir.reciprocal_rank_fusion(a, b, weights=[0.3, 0.7])]
        scaled = [item for item, _score in hybridir.reciprocal_rank_fusion(a, b, weights=[3.0, 7.0])]
        assert order == scaled

    def test_a_zero_weight_excludes_that_list_entirely(self):
        # This is how "search with one engine only" is spelled — the other arm's exclusives must not appear
        # at all, not merely rank low.
        keyword, vector = ["k1", "k2"], ["v1", "v2"]
        fused = [item for item, _score in
                 hybridir.reciprocal_rank_fusion(keyword, vector, weights=[1.0, 0.0])]
        assert fused == ["k1", "k2"]

    def test_weighting_reorders_the_documents_the_arms_disagree_about(self):
        # The point of the knob. Note what it does *not* touch: `shared` is in both lists, so it collects
        # from both arms and outranks either exclusive at any weight. That is RRF's agreement bonus
        # working as intended, and it is why this asserts on the disputed pair rather than on the winner.
        keyword, vector = ["doc_k", "shared"], ["doc_v", "shared"]

        def order(weights):
            ranked = [item for item, _score in
                      hybridir.reciprocal_rank_fusion(keyword, vector, weights=weights)]
            return ranked.index("doc_k") < ranked.index("doc_v")

        assert order([0.9, 0.1]) is True     # keyword-heavy: its exclusive wins the disputed slot
        assert order([0.1, 0.9]) is False    # semantic-heavy: the other one does

    def test_a_smaller_K_sharpens_the_advantage_of_rank_one(self):
        # What `rrf_k` controls, stated as the property rather than as a number: the gap between the top
        # two ranks of one list, relative to the scores themselves.
        def head_gap(K):
            scored = dict(hybridir.reciprocal_rank_fusion(["a", "b"], K=K))
            return (scored["a"] - scored["b"]) / scored["a"]
        assert head_gap(10) > head_gap(60) > head_gap(120)

    def test_mismatched_weight_count_is_an_error(self):
        # Silently padding or truncating would mis-weight an arm, which is invisible in the output.
        with pytest.raises(ValueError):
            hybridir.reciprocal_rank_fusion(["a"], ["b"], weights=[1.0])


# ---------------------------------------------------------------------------
# Stitching adjacent chunks back together
# ---------------------------------------------------------------------------

def _chunk(doc, offset, score, size=100, fill=None):
    """One retrieval hit. `fill` defaults to a per-offset letter, so a merged text says where it came from."""
    text = (fill or chr(ord("a") + (offset // size) % 26)) * size
    return {"document_id": doc, "offset": offset, "text": text, "score": score}


class TestMergeContiguousSpans:
    """Chunks that were adjacent in the document come back as one result, so the reader gets a passage."""

    def test_adjacent_chunks_become_one_span(self):
        merged = hybridir.merge_contiguous_spans([_chunk("d", 0, 0.9), _chunk("d", 100, 0.5)])
        assert len(merged) == 1
        assert merged[0]["offset"] == 0
        assert len(merged[0]["text"]) == 200

    def test_a_span_keeps_the_best_score_of_its_chunks(self):
        # It has to: the span occupies the rank its strongest evidence earned, not its weakest.
        merged = hybridir.merge_contiguous_spans([_chunk("d", 0, 0.2), _chunk("d", 100, 0.9)])
        assert merged[0]["score"] == 0.9

    def test_chunks_with_a_gap_between_them_stay_separate(self):
        merged = hybridir.merge_contiguous_spans([_chunk("d", 0, 0.9), _chunk("d", 5000, 0.5)])
        assert len(merged) == 2

    def test_chunks_of_different_documents_never_merge(self):
        # Offsets collide across documents, so grouping by offset alone would splice two documents together.
        merged = hybridir.merge_contiguous_spans([_chunk("a", 0, 0.9), _chunk("b", 100, 0.5)])
        assert len(merged) == 2
        assert {m["document_id"] for m in merged} == {"a", "b"}

    def test_results_come_back_ordered_by_score(self):
        merged = hybridir.merge_contiguous_spans([_chunk("a", 0, 0.1), _chunk("b", 0, 0.9), _chunk("c", 0, 0.5)])
        assert [m["score"] for m in merged] == [0.9, 0.5, 0.1]


class TestMergedSpanLengthCap:
    """A span is atomic to a caller spending a token budget, so its length has to be boundable.

    One longer than the budget contributes nothing at all — measured on a prose corpus at 7500 characters,
    the unbounded version returned zero results on two queries out of five. The cap splits such a run into
    several spans instead, which costs only the seam.
    """

    def _run(self, n, size=100):
        return [_chunk("d", i * size, 1.0 - i / 100, size=size) for i in range(n)]

    def test_a_long_run_is_split_rather_than_returned_whole(self):
        merged = hybridir.merge_contiguous_spans(self._run(4), max_span_length=200)
        assert len(merged) == 2
        assert all(len(m["text"]) <= 200 for m in merged)

    def test_splitting_loses_no_text(self):
        # The invariant that makes this safe: the same characters come back, in more pieces. A cap that
        # dropped the tail would look identical in a length assertion and be a data-loss bug.
        whole = hybridir.merge_contiguous_spans(self._run(6))
        assert len(whole) == 1
        pieces = hybridir.merge_contiguous_spans(self._run(6), max_span_length=200)
        assert len(pieces) > 1
        rejoined = "".join(p["text"] for p in sorted(pieces, key=lambda p: p["offset"]))
        assert rejoined == whole[0]["text"]

    def test_a_cap_wider_than_the_run_changes_nothing(self):
        assert (hybridir.merge_contiguous_spans(self._run(3), max_span_length=100_000) ==
                hybridir.merge_contiguous_spans(self._run(3)))

    def test_no_cap_means_unlimited(self):
        merged = hybridir.merge_contiguous_spans(self._run(20))
        assert len(merged) == 1 and len(merged[0]["text"]) == 2000

    def test_every_piece_still_carries_its_own_best_score(self):
        # Not the whole run's best: a later piece must not inherit the rank of an earlier piece's evidence,
        # or the split would silently promote text that nothing matched.
        pieces = hybridir.merge_contiguous_spans(self._run(4), max_span_length=200)
        by_offset = sorted(pieces, key=lambda p: p["offset"])
        assert by_offset[0]["score"] > by_offset[1]["score"]


# ---------------------------------------------------------------------------
# Rescan identifies documents by their id, not by where the collection sits
# ---------------------------------------------------------------------------

class TestRescanKeysOnDocumentId:
    """A document is identified by its path *relative* to the documents directory.

    So a collection reached through a renamed, moved or symlinked documents directory is still the same
    collection, and a rescan of it has nothing to do. Keying on the absolute path each record also stores
    made the same collection read as an entirely different one — every file new, every indexed document
    deleted — and then raised on the way there, because building the deletion task re-derived an id from a
    stored path that has no relative form under the new directory.
    """

    def _handler(self, docs_dir, indexed):
        """A handler wired to a stub retriever holding `indexed`, as `{document_id: absolute path}`."""
        retriever = types.SimpleNamespace(
            documents={document_id: {"path": path, "mtime": 0, "filesize": 1}
                       for document_id, path in indexed.items()},
            datastore_lock=threading.RLock(),
            _stat=lambda path: {"mtime": 0, "size": 1})
        handler = hybridir.HybridIRFileSystemEventHandler.__new__(hybridir.HybridIRFileSystemEventHandler)
        handler.docs_dir = pathlib.Path(docs_dir)
        handler.retriever = retriever
        handler.commit_task = lambda task_env: None  # bypassing __init__, so the real one is not bound
        return handler

    def test_the_same_collection_under_a_second_spelling_needs_no_work(self, tmp_path, monkeypatch):
        real = tmp_path / "documents_hydrogen"
        real.mkdir()
        (real / "paper.bib").write_text("@article{a, title={H2}}", encoding="utf-8")
        slot = tmp_path / "documents"
        slot.symlink_to(real)

        # Indexed by its real directory; rescanned through the symlink. Same document, both times.
        handler = self._handler(slot, {"paper.bib": str(real / "paper.bib")})
        monkeypatch.setattr(handler, "_sanity_check", lambda path: True)
        scheduled = []
        monkeypatch.setitem(hybridir.task_managers, "ingest",
                            types.SimpleNamespace(submit=lambda *a, **kw: scheduled.append(a)))

        handler.rescan(slot)
        assert scheduled == []   # not "every file new and every document deleted", and no ValueError

    def test_a_document_whose_file_is_gone_is_deleted_by_id(self, tmp_path, monkeypatch):
        # The deletion path specifically: the record's stored path lies outside the current documents
        # directory, so it has no relative form and must not be asked for one.
        slot = tmp_path / "documents"
        slot.mkdir()
        handler = self._handler(slot, {"vanished.bib": str(tmp_path / "somewhere_else" / "vanished.bib")})
        deleted = []
        monkeypatch.setattr(handler.retriever, "delete", deleted.append, raising=False)
        monkeypatch.setitem(hybridir.task_managers, "ingest",
                            types.SimpleNamespace(submit=lambda task, env: task(env)))
        monkeypatch.setitem(hybridir.task_managers, "commit",
                            types.SimpleNamespace(submit=lambda *a, **kw: None))

        handler.rescan(slot)
        assert deleted == ["vanished.bib"]


# ---------------------------------------------------------------------------
# Splitting a chat message into subqueries (lever 3 of brief 09)
# ---------------------------------------------------------------------------

class TestSplitIntoSubqueries:
    """A chat message is not a query, so it is also queried in pieces — beside the whole, never instead."""

    def test_a_single_sentence_adds_nothing(self):
        # The caller already has the whole text; returning it again would double its vote in the fusion.
        assert hybridir.split_into_subqueries("What is the specific energy consumption of alkaline electrolyzers?") == []

    def test_a_multi_sentence_message_splits(self):
        out = hybridir.split_into_subqueries("I am working on alkaline electrolyzers for a review. "
                                             "What is their specific energy consumption?")
        assert out == ["I am working on alkaline electrolyzers for a review.",
                       "What is their specific energy consumption?"]

    def test_the_context_carrying_sentence_survives_as_its_own_query(self):
        # The mirror-image failure the whole-text query exists to cover: split alone, "What is the specific
        # energy consumption?" is about nothing, because the topic lived in the previous sentence. Both
        # pieces are returned, so the fusion sees the topic sentence too.
        out = hybridir.split_into_subqueries("I'm working on alkaline electrolyzers. "
                                             "What is the specific energy consumption?")
        assert "I'm working on alkaline electrolyzers." in out

    def test_short_pleasantries_are_dropped(self):
        out = hybridir.split_into_subqueries("Good evening! Thanks. "
                                             "What is the specific energy consumption of alkaline electrolyzers?")
        assert all("evening" not in piece and "Thanks" not in piece for piece in out)

    def test_question_marks_and_exclamations_both_end_a_sentence(self):
        out = hybridir.split_into_subqueries("Is hydrogen storage solved yet? "
                                             "I keep reading that it is not solved at all!")
        assert len(out) == 2

    def test_a_closing_quote_after_the_stop_still_ends_the_sentence(self):
        out = hybridir.split_into_subqueries('The paper says "this is settled." '
                                             "I would like to know whether that is actually true.")
        assert len(out) == 2

    def test_the_number_of_pieces_is_capped(self):
        # Not for retrieval cost — the pieces batch into one call each — but for the fusion, where twenty
        # mediocre sentence-queries outvote the one good whole-message query.
        message = " ".join(f"This is sentence number {i} of a rambling message." for i in range(30))
        assert len(hybridir.split_into_subqueries(message)) <= 8

    def test_the_cap_keeps_the_end_of_the_message(self):
        # Recency is the usable prior: someone who has typed five paragraphs is asking about the end of them,
        # and the opening is scene-setting that the whole-text query already carries.
        message = " ".join(f"This is sentence number {i} of a rambling message." for i in range(30))
        assert "number 29" in hybridir.split_into_subqueries(message)[-1]

    def test_a_message_of_only_pleasantries_adds_nothing(self):
        assert hybridir.split_into_subqueries("Hi! Thanks. Bye.") == []

    def test_whitespace_only_input_is_safe(self):
        assert hybridir.split_into_subqueries("   \n  ") == []


# ---------------------------------------------------------------------------
# Reading the shape of a score distribution (lever 1 of brief 09)
# ---------------------------------------------------------------------------

class TestScoreSharpness:
    """Telling "this query found something" from "this query's best result is best by noise"."""

    def test_a_towering_top_hit_is_sharp(self):
        scores = [30.0] + [0.5] * 19
        assert hybridir.score_sharpness(scores, min_ratio=0.1) == pytest.approx(0.95)

    def test_a_flat_list_is_not_sharp(self):
        assert hybridir.score_sharpness([5.0] * 20, min_ratio=0.1) == 0.0

    def test_the_best_result_always_keeps_up_with_itself(self):
        # So the maximum is 1 - 1/n rather than 1, and a single candidate reports no sharpness at all —
        # there is no distribution to read off one number.
        assert hybridir.score_sharpness([42.0], min_ratio=0.1) == 0.0

    def test_nothing_found_is_not_confident(self):
        # A query matching nothing yields all-zero scores. That is the least confident case there is, so it
        # must not come out at the sharp end merely because the list is degenerate.
        assert hybridir.score_sharpness([0.0] * 20, min_ratio=0.1) == 0.0
        assert hybridir.score_sharpness([], min_ratio=0.1) == 0.0

    def test_the_reading_is_scale_free(self):
        # The property the whole design rests on: no constant fitted to a corpus, and immunity to an
        # embedder swap or a corpus changing character, because only a query's results are ever compared
        # to each other.
        scores = [30.0, 6.0, 3.0, 1.0]
        assert (hybridir.score_sharpness(scores, min_ratio=0.15) ==
                hybridir.score_sharpness([1000.0 * s for s in scores], min_ratio=0.15))

    def test_a_stricter_ratio_never_reads_sharper(self):
        # Raising the bar can only shrink the survivor set, so sharpness is monotone in `min_ratio`. This is
        # what makes sweeping the ratio against the evaluation set a well-behaved search.
        scores = [30.0, 12.0, 6.0, 3.0, 1.0, 0.2]
        readings = [hybridir.score_sharpness(scores, min_ratio=r) for r in (0.01, 0.05, 0.1, 0.3, 0.5, 0.9)]
        assert readings == sorted(readings)

    def test_order_does_not_matter(self):
        scores = [30.0, 6.0, 3.0, 1.0]
        assert (hybridir.score_sharpness(scores, min_ratio=0.15) ==
                hybridir.score_sharpness(list(reversed(scores)), min_ratio=0.15))
