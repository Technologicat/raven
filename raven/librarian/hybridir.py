"""A simple hybrid (keyword + semantic) information retrieval (IR) system.

While this is no Google, this can be useful for retrieval-augmented generation (RAG) for LLMs,
and for adding a semi-intelligent fulltext search to an app.

The search index is persisted automatically. Still, **everything** runs from RAM.

The implementation is rather memory-hungry, because we keep a second copy of chunks/tokens/embeddings
as well as the fulltext of each document. This keeps the code simple, and enables easy index rebuilds.
For example, if the fulltext of each document is 100 KB, and you have 1e4 such documents, you'll need
100 * 1e3 * 1e4 bytes = 1e9 bytes = 1 GB just to keep a copy of the fulltexts in memory; and likely a
couple more times this, to accommodate the two indexing mechanisms. But I'm thinking that nowadays
laptops have enough RAM for this not to be an issue with the dataset sizes needed in Raven.

QwQ-32B wrote a very first initial rough draft outline, from which this was then manually coded.
"""

__all__ = ["split_into_subqueries", "score_sharpness",
           "init", "shutdown", "HybridIR", "HybridIRFileSystemEventHandler", "setup"]

import logging
logger = logging.getLogger(__name__)

import atexit
from collections import defaultdict
import concurrent.futures
import copy
import functools
import json
import operator
import os
import pathlib
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import uuid

import watchdog.events
import watchdog.observers

import numpy as np

from unpythonic import allsame, box, ETAEstimator, uniqify
from unpythonic.env import env as envcls

# database
import bm25s  # keyword
import chromadb  # semantic (vector)

from ..client import api
from ..client import config as client_config
from ..client import mayberemote

from ..common import bgtask
from ..common import deviceinfo
from ..common import docextract
from ..common import nlptools
from ..common import utils as common_utils

from . import config as librarian_config

# --------------------------------------------------------------------------------
# Module bootup

deviceinfo.validate(librarian_config.devices)  # modifies in-place if CPU fallback needed

api.initialize(raven_server_url=client_config.raven_server_url,
               raven_api_key_file=client_config.raven_api_key_file)  # let it create a default executor

# --------------------------------------------------------------------------------

# Chunks per tokenization request. Amortizes the ~57 ms round trip over a batch (see
# `_prepare_document_for_indexing`) while keeping the per-document progress report updating a few times
# a second on a large document. At ~10 ms/chunk batched, 64 chunks is roughly 0.6 s per update.
TOKENIZE_BATCH_SIZE = 64


def format_chunk_full_id(document_id: str, chunk_id: str) -> str:
    """Generate an identifier for a chunk of a document, based on the given IDs."""
    return f"doc-{document_id}-chunk-{chunk_id}"

# A sentence end: `.`, `?` or `!`, optionally closed by a quote or bracket, followed by whitespace. Kept
# deliberately naive — it splits "et al. 2020" and "Fig. 3" in the middle, and that costs almost nothing
# here, because the pieces are *additional* queries beside the whole text rather than a partition of it. A
# fragment that means nothing retrieves nothing in particular and contributes a flat list to the fusion; a
# fragment that means something is exactly what we wanted. Buying spaCy's sentence segmentation (a server
# round trip, on the latency path before every reply) to avoid that is the wrong trade.
_SENTENCE_END = re.compile(r"(?<=[.?!][\"'’”)\]])\s+|(?<=[.?!])\s+")

# Below this many words a piece is not a query. "Thanks!", "Hmm.", "See below." retrieve noise, and every
# extra list handed to RRF dilutes the ones that mean something, since fusion weights by position only.
#
# It is a floor, not a solution, and the gap is worth naming: it catches "Good evening!" and lets "How are
# you doing today?" through at five words of pure pleasantry. A stoplist of greetings would be
# unmaintainable and language-specific, so the real answer is a per-subquery confidence test — a piece whose
# own score distribution comes back flat should not get a vote in the fusion, which is exactly what a
# pleasantry against a technical corpus produces. Until that exists, the damage is bounded rather than
# absent: the whole-text query is always in the fusion too, and RRF rewards agreement across lists, so an
# uncorroborated noise piece costs tail slots rather than the top of the ranking.
_MINIMUM_SUBQUERY_WORDS = 4

# At most this many pieces, beside the whole text. A cap is needed because the cost of a subquery is not
# the retrieval — those batch — but its vote in the fusion: twenty mediocre sentence-queries outvote the
# one good whole-message query. Eight is a guess, not a measurement; it wants sweeping against
# `investigations/retrieval/` alongside the rest of this lever.
_MAXIMUM_SUBQUERIES = 8

def split_into_subqueries(text: str) -> List[str]:
    """Split a chat message into the sentences worth querying separately. Returns them in document order.

    The user's message is not a query. With a multiline composer it can be several paragraphs of context
    ending in one specific question, and such a message embeds to a *centroid* — near nothing in particular,
    since a centroid's nearest neighbours are chosen by average topicality rather than by answering anything.
    Measured on `investigations/retrieval/`, that shape retrieves at 0.292 MRR against 0.562 for a focused
    question: the largest effect in that data by some way.

    **These are meant to be used *beside* the whole text, never instead of it**, which is what makes the
    naive splitting safe and is easy to get wrong. Splitting alone trades one failure for its mirror image:
    "I'm working on alkaline electrolyzers. What is the specific energy consumption?" yields a second
    sentence about nothing at all, because the topic lived in the first one. Querying with the whole message
    *and* each piece, then fusing all the result sets, needs no decision about which shape the message is —
    the whole carries the context, the pieces carry the specificity, and neither has to be right.

    Returns `[]` when there is nothing to add: a single-sentence message (the caller already has it), or one
    whose pieces are all too short to mean anything. A caller can therefore treat the empty list as "just
    query the text", with no special case.
    """
    pieces = [piece.strip() for piece in _SENTENCE_END.split(text.strip())]
    subqueries = [piece for piece in pieces if len(piece.split()) >= _MINIMUM_SUBQUERY_WORDS]
    if len(subqueries) < 2:  # nothing gained: one piece is the whole text again, none is nothing to add
        return []
    if len(subqueries) > _MAXIMUM_SUBQUERIES:
        # Keep the *last* ones. Recency is the usable prior in a chat message: a reader who has typed five
        # paragraphs is asking about the end of them, and the opening is scene-setting the whole-text query
        # already carries.
        subqueries = subqueries[-_MAXIMUM_SUBQUERIES:]
    return subqueries

def score_sharpness(scores: List[float], min_ratio: float) -> float:
    """How far the best of `scores` stands off from the rest. In `[0, 1)`; bigger is sharper.

    This answers a question nothing else in the pipeline answers: *did this query find anything, or is
    its best result merely the head of a flat list?* A query that found something produces a sharp head
    and a long tail; a query that found nothing produces a list of near-equals, whose "best" is best by
    noise. Both look identical once fusion has replaced every score with its rank.

    `scores`: One query's raw candidate scores, from one engine, in any order. **Bigger must mean
              better, and zero must mean no match** — which is true of BM25 as-is, and of a cosine
              *similarity*, but not of the cosine *distance* a vector store returns. Convert first
              (`similarity = 1 - distance`); passing distances measures the worst match instead.

              Use the full retrieved candidate list, not the survivors of a score threshold: the
              denominator has to be a fixed population for the fraction below to be comparable across
              queries, and a candidate the threshold rejected is precisely a candidate the best result
              left behind.

    `min_ratio`: A candidate counts as keeping up if it scores at least `min_ratio` times the best.

    Returns the fraction of candidates the best result left behind, i.e. `1 - survivors / len(scores)`.
    A query whose top hit towers over 19 of its 20 candidates scores 0.95; one where everything keeps up
    scores 0. The best result always keeps up with itself, so the maximum is `1 - 1/len(scores)` rather
    than 1. Degenerate cases — no candidates, or nothing scoring above zero — return 0.0, since a query
    that found nothing has no sharpness to report and should not read as a confident one.

    This is LLM sampling's `min_p` filter, transplanted, and it runs *backwards* here: on a flat
    distribution the bar is low and nearly everything survives, on a sharp one almost nothing does. So
    the survivor count is the signal rather than the selection, which is why this returns the complement
    and calls it sharpness — a consumer that has to remember an inversion will eventually forget it.

    **Borrow the idea, not the number.** `min_p` operates on a normalized probability distribution, and
    BM25 scores are neither normalized nor bounded. The ratio-to-best test survives that (it is in fact
    scale-free in a way `min_p` itself is not, which is what makes it immune to a corpus changing
    character or an embedder being swapped), but the values in circulation for LLM sampling were fitted
    against a distribution with different properties and carry no information about what to use here.
    The two engines will not want the same `min_ratio` either: BM25 scores range over orders of
    magnitude while cosine similarities sit in a narrow band, so the same ratio reads as a much weaker
    test on the vector arm.
    """
    if not scores:
        return 0.0
    best = max(scores)
    if best <= 0.0:  # nothing matched at all
        return 0.0
    survivors = sum(1 for score in scores if score >= min_ratio * best)
    return 1.0 - survivors / len(scores)

def reciprocal_rank_fusion(*item_lists: List[Any], K: int = 60) -> List[Tuple[Any, float]]:
    """Fuse rank from multiple IR systems using Reciprocal Rank Fusion (RRF).

    `item_lists`: Lists of search results, one list from each IR system.

                  Each list is assumed to be in the ranked order produced by that IR system
                  (descending, i.e. best matches first), but with no score information.

                  Each item must be hashable. It is recommended to use document IDs or similar.

    `K`: The constant used in the RRF formula. Default 60, a typical value from IR literature.

    Returns a list of tuples `(item, rrf_score)`, sorted by the RRF score, descending.

    Based on:
        https://gist.github.com/srcecde/eec6c5dda268f9a58473e1c14735c7bb
    """
    rrf_results = defaultdict(float)  # item -> score
    for items in item_lists:
        for rank, item in enumerate(items, start=1):
            rrf_results[item] += 1 / (rank + K)

    sorted_items = list(sorted(rrf_results.items(),
                               key=operator.itemgetter(1),
                               reverse=True))  # -> [(item0, score0), ...]
    return sorted_items

def merge_contiguous_spans(results: List[Dict]) -> List[Dict]:
    """Given a list of search results, merge overlapping/adjacent document chunks into contiguous spans.

    `results`: List of search hits with:
        - `document_id` (str)
        - `text` (str): The chunk text
        - `offset` (int): Start offset in original text
        - `score` (float): Search rank score

    Returns a list of the merged chunks, with each merged chunk (contiguous text from the same document)
    assigned the highest score of the original chunks that the merged chunk was built from.

    Each merged chunk has the same format as the input.

    The results are returned sorted by score, descending.
    """
    # Utility
    def merge_group(group: List[Dict]) -> List[Dict]:
        """Merge a group, i.e. a list of contiguous chunks the same document.

        `group` is assumed to be sorted by `offset` (so that we can perform the merge in one pass over the data).
        """
        if not group:
            return None
        if len(group) == 1:
            return group[0]

        # Gather all fields in one go
        document_ids = []
        offsets = []
        scores = []
        for hit in group:
            document_ids.append(hit["document_id"])
            offsets.append(hit["offset"])
            scores.append(hit["score"])
        assert allsame(document_ids), f"Expected all chunks to come from the same document; got multiple document IDs: {list(uniqify(document_ids))}"

        start_offset = min(offsets)
        local_offsets = [offset - start_offset for offset in offsets]  # local offsets, where the first chunk starts at zero

        out = {"document_id": document_ids[0],
               "offset": start_offset,
               "score": max(scores)}
        text = group[0]["text"]
        for hit_local_offset, hit in zip(local_offsets[1:], group[1:]):
            assert hit_local_offset <= len(text)
            text = text[:hit_local_offset] + hit["text"]  # TODO: Expensive for long texts with lots of contiguous chunks? Meybe RAG chunks are short enough for this to not matter.
        out["text"] = text

        return out

    # Group search results by document
    hits_by_document = defaultdict(list)
    for hit in results:
        doc_id = hit["document_id"]
        hits_by_document[doc_id].append(hit)

    # Find contiguous chunks in the search results (separately from each document)
    groups_by_document = defaultdict(list)
    for doc_id, hits in hits_by_document.items():
        # Sort chunks by their offset in the document
        sorted_hits = list(sorted(hits,
                                  key=operator.itemgetter("offset")))

        current_group = []
        current_end = 0
        for hit in sorted_hits:
            hit_start = hit["offset"]
            hit_end = hit_start + len(hit["text"])

            # Check if either this is the first group, or the current hit can join the existing group.
            if not current_group or (hit_start <= current_end):
                current_group.append(hit)
            else:
                # Commit current group if there was one, and start new group.
                if current_group:
                    groups_by_document[doc_id].append(current_group)
                current_group = [hit]
            current_end = hit_end
        # Commit the last group
        groups_by_document[doc_id].append(current_group)

    # Merge the contiguous chunks from each document.
    # After that, we don't need to group by document any more.
    merged_results = []
    for doc_id, groups in groups_by_document.items():
        for group in groups:
            merged_results.append(merge_group(group))

    # Sort the full set of merged results (across all documents) by descending score.
    return list(sorted(merged_results,
                       key=operator.itemgetter("score"),
                       reverse=True))

# --------------------------------------------------------------------------------

# TODO: `chunk_size` and `overlap_fraction` should probably also remain fixed after the datastore has been created.
class HybridIR:
    def __init__(self,
                 datastore_base_dir: Union[str, pathlib.Path],
                 embedding_model_name: str = "sentence-transformers/multi-qa-mpnet-base-cos-v1",
                 local_model_loader_fallback: bool = True,
                 chunk_size: int = 1000,
                 overlap_fraction: float = 0.25) -> None:
        """Hybrid information retrieval (IR) index, using both keyword and semantic search.

        `datastore_base_dir`: Where to store the data (for the specific collection you're creating/loading).
                              The data is persisted automatically.

        `embedding_model_name`: Semantic vector embedder, for semantic search.

                                Used only when the datastore has not been created yet.

                                After the datastore has been created, its embedding model cannot be changed,
                                and `HybridIR` will automatically use the model that was used to create
                                that datastore.

                                Basically you can put a HuggingFace model path here.
                                The default is a QA model that was trained specifically for semantic search.

                                For more details, see `sentence_transformers.SentenceTransformer`, and:
                                https://sbert.net/docs/sentence_transformer/pretrained_models.html

        `local_model_loader_fallback`: Whether to load models locally if Raven-server can't be reached.

                                       Apps that need the server also for other reasons may want to disable this.

                                       (Especially if the server is on another machine; then loading the models
                                        locally will download an extra copy of the models on the client machine.
                                        This could be undesirable if the app is not useful without the server.)

        `chunk_size`: Length of a search result chunk, in characters (Python native, so Unicode codepoints).

                      Smaller chunks gives more fine-grained search results inside each document, at the cost of
                      increasing the size of the index (because each chunk is a data record).

                      Note also it is possible that the neighbors of a matching chunk don't match the same search,
                      so if the chunk size is too small, you'll only get a very short snippet, with not much context.

                      But see the "offset" field of the chunk returned - you can then retrieve as much context as you want
                      from `your_hybrid_ir.documents[result["document_id"]]["text"]`, which contains the full text.

        `overlap_fraction`: For sliding window chunking, to avoid losing context at the seams of the chunks.
                            E.g. 0.25 means that the next chunk repeats the final 25% of the current chunk.

                            In search results, adjacent chunks are automatically seamlessly combined (removing overlaps),
                            so this only affects the performance (more overlap -> higher chance of having a better-matching
                            local excerpt as a chunk in the index) and the size of the index (more overlap -> more duplicated
                            text -> larger index -> slower search, uses more disk space for storage).
        """
        self.datastore_lock = threading.RLock()  # self.documents, and the keyword and vector search indices
        self._pending_edits_lock = threading.RLock()  # self._pending_edits

        self.datastore_base_dir = datastore_base_dir
        self.fulldocs_path = datastore_base_dir / "fulldocs"
        self.fulldocs_documents_file = self.fulldocs_path / "data.json"
        self.fulldocs_embeddings_file = self.fulldocs_path / "embeddings.npz"
        self.keyword_index_path = datastore_base_dir / "bm25s"
        self.semantic_index_path = datastore_base_dir / "chromadb"

        self.chunk_size = chunk_size
        self.overlap = int(overlap_fraction * chunk_size)

        # Load the main datastore. We use this to rebuild the BM25 index when documents are added/updated/deleted.
        # Note `self.documents` is technically part of the public API.
        stored_embedding_model_name, stored_documents = self._load_datastore()
        if stored_embedding_model_name is not None:  # load successful?
            if embedding_model_name != stored_embedding_model_name:
                logger.warning(f"HybridIR.__init__: Existing datastore at '{str(self.fulldocs_path)}' was created with embedding model '{stored_embedding_model_name}', which is different from the requested '{embedding_model_name}'. Using the datastore's model.")
            self.embedding_model_name = stored_embedding_model_name
            self.documents = stored_documents
        else:
            logger.info(f"HybridIR.__init__: Will create new datastore at '{str(self.fulldocs_path)}', with embedding model '{embedding_model_name}', at first commit.")
            self.embedding_model_name = embedding_model_name
            self.documents = {}

        # We compute vector embeddings manually (on Raven's side).
        self.embedder = mayberemote.Embedder(allow_local=local_model_loader_fallback,
                                             model_name=self.embedding_model_name,
                                             device_string=librarian_config.devices["embeddings"]["device_string"],
                                             dtype=librarian_config.devices["embeddings"]["dtype"])
        self.nlp = mayberemote.NLP(allow_local=local_model_loader_fallback,
                                   model_name=librarian_config.spacy_model,
                                   device_string=librarian_config.devices["nlp"]["device_string"])

        self._stopwords = nlptools.default_stopwords

        # Semantic search: ChromaDB vector storage
        # ChromaDB persists data automatically when we use the `PersistentClient`
        # https://docs.trychroma.com/docs/collections/create-get-delete
        self._vector_client = chromadb.PersistentClient(path=str(self.semantic_index_path),
                                                        settings=chromadb.Settings(anonymized_telemetry=False))
        try:
            self._vector_collection = self._vector_client.get_collection(name="embeddings")  # try loading existing vector index
        except Exception:  # vector index missing
            logger.warning(f"HybridIR.__init__: Caught exception while loading vector index from '{str(self.semantic_index_path)}'", exc_info=True)

            logger.info(f"HybridIR.__init__: Vector index not found. Creating new (blank) vector index at '{str(self.semantic_index_path)}'.")
            self._vector_collection = self._vector_client.create_collection(name="embeddings",
                                                                            metadata={"hnsw:space": "cosine"})  # we use normalized semantic vectors, so cosine distance is appropriate.

            if self.documents:  # suppress log message when no documents to process
                plural_s = "s" if len(self.documents) != 1 else ""
                logger.info(f"HybridIR.__init__: Rebuilding vector index for {len(self.documents)} document{plural_s} from main datastore. This may take a while.")
                # TODO: Full reindexing is slow. This should run as a bgtask. OTOH, the documents in the main datastore are pre-prepared (with embeddings), so here ChromaDB only needs to create the HNSW.
                # TODO: We could support changing the embedding model here, by preparing the documents again.
                for doc in self.documents.values():
                    self._add_document_to_vector_collection(doc)

        # Keyword search: BM25 index (tokenized documents as input)
        try:
            self._keyword_retriever = bm25s.BM25.load(str(self.keyword_index_path),
                                                      load_corpus=True)
            self._build_full_id_to_record_index()
        except Exception:  # keyword index missing
            logger.warning(f"HybridIR.__init__: Caught exception while loading keyword index from '{str(self.keyword_index_path)}'", exc_info=True)
            logger.info(f"HybridIR.__init__: Keyword index not found. Will create keyword index at '{str(self.keyword_index_path)}'.")

            if self.documents:  # suppress log message when no documents to process
                # TODO: full reindexing is slow. This should run as a bgtask.
                self._rebuild_keyword_search_index()
            else:  # No documents yet.
                self._keyword_retriever = None
                self.full_id_to_record_index = {}

        # Pending-edit mechanism so that we can add/update/delete a set of documents at once, and *then* rebuild the indices.
        self._pending_edits = []

        # Per-instance "currently indexing" reference counter, polled by GUI clients (e.g. Librarian's idle
        # throttle and DOCS indicator). A counter rather than a bool because `commit()` may run concurrently
        # from multiple threads — the outermost invocation's `finally` is the one that must zero the state.
        # Same-thread re-entry is also possible in principle, since `datastore_lock` is an RLock.
        self._indexing_lock = threading.Lock()
        self._indexing_count = 0

        # Human-readable progress messages for the GUI to mirror. Two independent channels — `commit()` and
        # `query()` can run concurrently (each acquires `datastore_lock` only briefly), so they need
        # separate strings to drive separate indicators in the GUI. String reads and writes are atomic
        # under the GIL; no lock needed. The GUI side polls each channel and re-sets the corresponding
        # DPG text widget on change.
        self._indexing_progress_text: str = ""  # set per-iteration in commit; "Saving…" during the tail
        self._query_progress_text: str = ""     # set per-phase in query (Tokenizing/Embedding/…)

        # Indexing lifecycle callbacks. Fired on the 0↔1 transitions of `_indexing_count`, so nested or
        # concurrent commits don't fire `start`/`done` multiple times — only the outermost invocation
        # triggers the visual events. See `set_indexing_callbacks`.
        self._on_indexing_start: Optional[Callable[[], None]] = None
        self._on_indexing_done: Optional[Callable[[], None]] = None

    def is_indexing(self) -> bool:
        """Return whether this instance is currently inside `commit()`.

        Per-instance: multiple `HybridIR`s report independently. Intended for GUI clients that want to surface
        index-rebuild activity (currently invisible heavy CPU/GPU work) and to gate idle-throttle predicates.
        """
        # int read is atomic under the GIL; no need to acquire the lock for a snapshot.
        return self._indexing_count > 0

    def get_indexing_progress_text(self) -> str:
        """Return the current human-readable indexing progress message, or `""` if not indexing.

        During commit: `"[14 / 186] | 2106.01345v2.bib | tokenizing 240 / 288 | elapsed 6s, ETA 01:14,
        total 01:20"` (the trailing part is `unpythonic.ETAEstimator.formatted_eta`). The middle field says
        where the *current* document is, and updates per chunk — without it a large document leaves this
        string unchanged for tens of seconds, which reads as a hung job rather than a slow one. It is absent
        for work that has no inner steps to report, such as a deletion. The "INDEXING" verb sits in the
        indicator's static label, so the progress text doesn't repeat it. During the rebuild + datastore
        save tail: `"Saving…"`. Outside of commit: `""`.

        Intended for GUI clients that poll once per frame and mirror the value into a DPG text widget.
        The underlying string is set from the worker thread that runs `commit()`; GIL-atomic, no lock.
        """
        return self._indexing_progress_text

    def _set_indexing_progress(self, edit_num: int, n_edits: int, document_id: str,
                               eta_estimator: ETAEstimator, detail: str = "") -> None:
        """Render one line of indexing progress into `get_indexing_progress_text`'s value.

        The commit loop binds everything but `detail` and hands the result to the per-document work, so that
        a long document reports where it is inside itself rather than going quiet until it finishes. The ETA
        is re-read on every call, which is what keeps its `elapsed` live between documents.
        """
        fields = [f"[{edit_num} / {n_edits}]", document_id]
        if detail:
            fields.append(detail)
        fields.append(eta_estimator.formatted_eta)
        self._indexing_progress_text = " | ".join(fields)

    def set_indexing_callbacks(self,
                               *,
                               on_start: Optional[Callable[[], None]] = None,
                               on_done: Optional[Callable[[], None]] = None) -> None:
        """Register callbacks fired on the indexing-busy state transitions.

        `on_start` fires (once) when an outermost `commit()` enters and `_indexing_count` goes 0→1.
        `on_done` fires (once) when the outermost `commit()` exits and `_indexing_count` goes 1→0. Nested or
        concurrent commits do not fire either callback — only the outer-edge transitions do.

        Both callbacks are invoked from the worker thread that runs `commit()`. Callees must be safe to
        call from a background thread. Exceptions raised by a callback are logged and swallowed; they
        never abort the indexing pipeline.

        Pass `on_start=None` / `on_done=None` (the defaults) to clear a previously registered callback.

        Why a setter rather than constructor arguments: `HybridIR` is constructed before its GUI client
        (`DPGChatController`) exists, so the natural Raven style of passing handlers at construction time
        isn't available here. The setter is the next-cleanest shape — handlers flow in via a single method
        call, no caller-side attribute mutation.
        """
        self._on_indexing_start = on_start
        self._on_indexing_done = on_done

    def get_query_progress_text(self) -> str:
        """Return the current human-readable query progress message, or `""` if no query is running.

        Per-phase during query: `"Tokenizing query…"`, `"Embedding query…"`, `"Keyword search…"`,
        `"Semantic search…"`, `"Merging results…"`. Outside of query: `""`.

        Intended for GUI clients that poll once per frame and mirror the value into a DPG text widget.
        The underlying string is set from the worker thread that runs `query()`; GIL-atomic, no lock.
        """
        return self._query_progress_text

    def _tokenize(self, text: str) -> List[str]:
        """Apply lowercasing, tokenization, lemmatization, stopword removal.

        Returns a list of tokens.

        We use a spaCy NLP pipeline to do the analysis.

        Lemmatization, not stemming - worth keeping straight, because the two fail differently. A stemmer
        chops suffixes by rule; a lemmatizer maps a word to its dictionary form, choosing by the word's
        part of speech. That is usually gentler, but it is not a guarantee, because the part of speech is
        itself a guess: spaCy's tagger is a neural model reading context, and a name that lands where an
        adjective would fit gets treated as one. "Elsevier" at the end of a copyright line is tagged ADJ
        and lemmatized to "elsevi", as though it were the comparative of "elsevi" (spaCy 3.8.14,
        en_core_web_sm 3.8.0; the lowercasing above makes no difference - it is the syntactic position that
        decides). So an unusual proper noun may or may not survive tokenization, and a keyword search for
        one is correspondingly a little lossy.
        """
        return self._tokenize_many([text])[0]

    def _tokenize_many(self, texts: List[str]) -> List[List[str]]:
        """`_tokenize` over several texts, in one call. Returns one token list per input, in order.

        Batched because the analysis is a round trip to raven-server, and at chunk sizes the round trip
        costs more than the analysis: measured 2026-08-06, ~90 ms end-to-end per chunk against ~25 ms of
        GPU compute, so roughly 57 ms of each call is HTTP and serialization. Handing spaCy a whole
        document's chunks at once pays that once instead of once per chunk.

        This mirrors what the embedding step in `_prepare_document_for_indexing` already does — one call
        per document — and the asymmetry was the reason tokenization dominated indexing while embedding
        was 3% of it.
        """
        if not texts:
            return []
        docs = self.nlp.analyze([text.lower() for text in texts])
        assert len(docs) == len(texts)
        return [[token.lemma_ for token in doc
                 if token.is_alpha and token.text not in self._stopwords]
                for doc in docs]

    def _stat(self, path: Union[pathlib.Path, str]) -> Dict:  # size, mtime
        p = pathlib.Path(path) if not isinstance(path, pathlib.Path) else path
        abspath = p.expanduser().resolve()
        if abspath.exists():
            stat = os.stat(abspath)
            return {"size": stat.st_size, "mtime": stat.st_mtime}
        return {"size": None, "mtime": None}  # could be an in-memory document

    def add(self, document_id: str, path: str, text: str) -> str:
        """Queue a document for adding into the index. To save changes, call `commit`.

        `document_id`: Must be unique, and **stable across runs** — `update` and `delete` match on it, so an
                       ID you cannot recompute later is an ID you cannot revise or remove. That rules out
                       anything freshly generated per call (a `gensym`, a UUID, a timestamp): re-indexing the
                       same file would mint a new ID and add a second copy rather than update the first.

                       Derive it from something the document has permanently. The built-in filesystem scanner
                       uses the path relative to `docs_dir` (see `HybridIRFileSystemEventHandler`), which is
                       unique, stable, and legible — worth keeping in mind, since a search result carries this
                       ID and it may end up in front of a human or a language model.

        `path`: Full path (or URL) of the original file.
                `HybridIR` uses this to check for changes to the file at datastore load time.
                You can use this to easily locate the original file a given search result refers to.

                If your document did not come from a file, use angle brackets
                to disable the file system event handler's rescan for that document,
                e.g. `path="<my in-memory document>"`.

        `text`: Plain-text content of the file, to be indexed for searching.

        Returns `document_id`, for convenience.
        """
        self._pend_edit(action="add", document_id=document_id, path=path, text=text)
        return document_id

    def update(self, document_id: str, path: str, text: str) -> str:
        """Queue a document for updating. To save changes, call `commit`.

        `document_id`: as in `add`.
        `path`: as in `add`. If you need the previous path, you can get it as `your_hybrid_ir.documents[document_id]["path"]`.
        `text`: as in `add`.

        Returns `document_id`, for convenience.
        """
        self._pend_edit(action="update", document_id=document_id, path=path, text=text)
        return document_id

    def delete(self, document_id: str) -> None:
        """Queue a document for deletion. To save changes, call `commit`.

        `document_id`: the ID you earlier gave to `add`.
        """
        self._pend_edit(action="delete", document_id=document_id)

    def _pend_edit(self,
                   action: str,
                   document_id: str,
                   path: Optional[str] = None,
                   text: Optional[str] = None):
        if action not in ("add", "update", "delete"):
            raise ValueError(f"Unknown action '{action}'; expected one of 'add', 'update', 'delete'.")
        logger.info(f"HybridIR._pend_edit: Queuing document '{document_id}' for {action}.")

        # Add or update -> prepare document record.
        if action != "delete":
            # Document-level data. This goes into the main datastore, which is also persisted so that we can rebuild indices when needed.
            #
            # Here we populate just those fields that can be filled quickly, so that the edit-queuing step can return instantly.
            # Fields that require expensive computation (chunks, tokens, embeddings) are added at commit time, by `_prepare_document_for_indexing`.
            #
            # Mind the insertion order of the fields - the resulting json should be easily human-readable, for debugging.
            # Metadata first, in a sensible order; fulltext last.
            stats = self._stat(path)
            document = {"document_id": document_id,
                        "path": path,  # path of original file (e.g. to be able to open it in a PDF reader)
                        "filesize": stats["size"],
                        "mtime": stats["mtime"],
                        "text": text,  # copy of original text as-is
                        }

        with self._pending_edits_lock:
            # Performance optimization: Drop any previous pending edits for the same document, since they'd be overwritten.
            new_edits = [(act, doc) for (act, doc) in self._pending_edits if doc["document_id"] != document_id]
            self._pending_edits.clear()
            self._pending_edits.extend(new_edits)

            # Pend the requested edit. Each queued entry is `(action, doc)` where `doc` is at minimum
            # a dict with a `document_id` key — the dedup pass above relies on that uniform shape.
            if action == "add":
                self._pending_edits.append((action, document))
            elif action == "delete":
                self._pending_edits.append((action, {"document_id": document_id}))
            else:  # action == "update":
                # Update is delete-then-add for an already-indexed document. But if the document isn't in the
                # committed index yet — a brand-new file whose watchdog `create` and `modify` events both landed
                # before the first commit (common for larger files, which emit several `modify` events while
                # being written) — there is nothing to delete. Queueing a delete then would only no-op with a
                # KeyError at commit time and inflate the change count, so emit it only when the document exists.
                # The membership read is a GIL-atomic dict lookup and takes no new lock; the two possible
                # stale-read races are already absorbed commit-side (the add path skips an already-present doc,
                # the delete path catches KeyError).
                if document_id in self.documents:
                    self._pending_edits.append(("delete", {"document_id": document_id}))
                self._pending_edits.append(("add", document))

    # TODO: Index rebuilding is slow. Maybe `commit` should run as a bgtask, like the BibTeX importer.
    #       Note `HybridIRFileSystemEventHandler` already does that.
    # TODO: `commit` is not as atomic as I'd like. If anything goes wrong, the vector index loses sync with the actual data, necessitating a full rebuild. Check if ChromaDB has transaction management.
    def commit(self, task_env: Optional[envcls] = None) -> None:
        """Commit pending changes (adds/deletes/updates), re-indexing the databases.

        An update is internally a delete, followed by an add for the updated version of the same document.

        `task_env`: Optional. A namespace carrying caller context — `unpythonic.env` is the typical
                    Raven-side carrier (it predates `types.SimpleNamespace` from the stdlib but plays the
                    same role). Any object with attribute access works. Currently inspected field is
                    `cancelled` (bool); when it flips `True`, the per-document loop exits cleanly after
                    the current document, partial state is persisted, and the unprocessed remainder is
                    requeued into `_pending_edits` so a later `commit` (in the same session) picks up
                    where this one left off. On app shutdown, that "later commit" never happens — the
                    leftovers sit in memory until the process exits, and `bootup.rescan` re-detects the
                    corresponding file changes via mtime on next startup. The `cancelled` attribute is
                    read defensively via `getattr`, so a namespace without it pre-set is accepted (and
                    treated as not-cancelled).
        """
        logger.info("HybridIR.commit: entered.")
        # Fire callbacks *outside* the lock — user code shouldn't run while we hold an internal lock,
        # in case it re-enters HybridIR. Capture the transition under the lock, fire after release.
        with self._indexing_lock:
            fire_start = (self._indexing_count == 0)
            self._indexing_count += 1
        # TEMP INSTRUMENTATION: INDEXING indicator debugging (2026-04-28)
        logger.info(f"HybridIR.commit: INSTR start: fire_start={fire_start}, _indexing_count(after_inc)={self._indexing_count}, on_indexing_start_wired={self._on_indexing_start is not None}")
        if fire_start and self._on_indexing_start is not None:
            try:
                self._on_indexing_start()
            except Exception:
                logger.exception("HybridIR.commit: on_indexing_start raised")
        try:
            self._commit_body(task_env=task_env)
        finally:
            self._indexing_progress_text = ""
            with self._indexing_lock:
                self._indexing_count -= 1
                fire_done = (self._indexing_count == 0)
            # TEMP INSTRUMENTATION: INDEXING indicator debugging (2026-04-28)
            logger.info(f"HybridIR.commit: INSTR done: fire_done={fire_done}, _indexing_count(after_dec)={self._indexing_count}, on_indexing_done_wired={self._on_indexing_done is not None}")
            if fire_done and self._on_indexing_done is not None:
                try:
                    self._on_indexing_done()
                except Exception:
                    logger.exception("HybridIR.commit: on_indexing_done raised")

    def _commit_body(self, task_env: Optional[envcls]) -> None:
        # Pop pending edits without `datastore_lock` — only `_pending_edits_lock` guards that list.
        with self._pending_edits_lock:
            if not self._pending_edits:
                logger.info("HybridIR.commit: No pending changes, exiting.")
                return
            pending_edits = copy.copy(self._pending_edits)
            self._pending_edits.clear()

        # Update `self.documents` and the semantic search index.
        # There is no "update" operation - to do that, first "delete", then "add".
        #
        # Lock granularity: we hold `datastore_lock` only briefly per iteration, around the actual
        # mutations. The slow `_prepare_document_for_indexing` (chunkify + tokenize + embed) is pure
        # — returns a new dict, doesn't touch self — and runs *outside* the lock. This lets a
        # concurrent `query()` interleave between iterations instead of blocking on the entire commit.
        logger.info("HybridIR.commit: Applying pending changes.")
        errors_occurred = 0
        cancelled_at = None  # set to the 1-based edit number of the iteration that observed cancellation
        eta_estimator = ETAEstimator(total=len(pending_edits), keep_last=50)
        for edit_num, (edit_kind, data) in enumerate(pending_edits, start=1):
            if task_env is not None and getattr(task_env, "cancelled", False):
                cancelled_at = edit_num
                remainder = pending_edits[edit_num - 1:]
                with self._pending_edits_lock:
                    self._pending_edits[0:0] = remainder  # prepend, preserving original order
                logger.info(f"HybridIR.commit: Cancelled before edit {edit_num} of {len(pending_edits)}; "
                            f"{len(remainder)} pending change(s) requeued for a later commit.")
                break
            # Both add and delete data shapes carry `document_id` (made uniform when the edit was queued).
            document_id = data["document_id"] if isinstance(data, dict) else "?"
            # Passed into the slow part below so it can report where it is *within* one document. A
            # per-document update was fine while a document was a 1.3 kB abstract and is not once it is a
            # 216 kB story: that one takes 23 s to prepare, of which 97% is tokenizing chunk by chunk, so
            # the indicator would sit unchanged for half a minute at a stretch and read as hung. Re-rendering
            # the line per chunk also makes `elapsed` in the ETA tick live, for free.
            report_progress = functools.partial(self._set_indexing_progress,
                                                edit_num, len(pending_edits), document_id, eta_estimator)
            report_progress()
            logger.info(f"HybridIR.commit: Applying change {edit_num} out of {len(pending_edits)}; {eta_estimator.formatted_eta}")
            try:
                if edit_kind == "add":
                    doc = data
                    document_id = doc["document_id"]
                    logger.info(f"HybridIR.commit: Adding document '{document_id}'.")

                    # The slow part runs *outside* `datastore_lock`. Pure: returns a new dict; no self-state mutation.
                    prepared = self._prepare_document_for_indexing(doc, on_progress=report_progress)

                    # Brief lock for the actual mutation. The dup check is here too so the check + insert is atomic.
                    with self.datastore_lock:
                        if document_id in self.documents:
                            logger.warning(f"HybridIR.commit: Document with ID '{document_id}' already exists in index; ignoring. If you meant to update, first delete, then add.")
                            continue
                        doc.update(prepared)
                        self.documents[document_id] = doc
                        self._add_document_to_vector_collection(doc)

                elif edit_kind == "delete":
                    document_id = data["document_id"]
                    logger.info(f"HybridIR.commit: Deleting document '{document_id}'.")
                    with self.datastore_lock:
                        try:
                            doc = self.documents[document_id]
                            old_chunk_ids = [format_chunk_full_id(document_id, chunk["chunk_id"]) for chunk in doc["chunks"]]
                            self.documents.pop(document_id)
                            self._vector_collection.delete(ids=old_chunk_ids)
                        except KeyError:
                            logger.warning(f"HybridIR.commit: Ignoring error: While deleting document with ID '{document_id}'", exc_info=True)

                else:  # should not happen, but let's log it
                    msg = f"HybridIR.commit: Unknown pending change type '{edit_kind}'. Ignoring."
                    logger.warning(msg)
            except Exception:
                errors_occurred += 1
                logger.exception("HybridIR.commit: Caught exception while applying changes. Attempting to continue with remaining edits, if any.")
            eta_estimator.tick()

        # Partial save: persist whatever was applied. The keyword index rebuilds from `self.documents`
        # (now reflecting the partial state); the vector index has been updated incrementally inside
        # the loop. Cheap for ~1k small docs; see TODO_DEFERRED for the segmented-backend story at scale.
        # Both methods take `datastore_lock` themselves, so the tail does serialize against `query()`.
        self._indexing_progress_text = "Saving…"
        self._rebuild_keyword_search_index()
        self._save_datastore()

        if cancelled_at is not None:
            logger.info(f"HybridIR.commit: Partial commit persisted ({cancelled_at - 1} of {len(pending_edits)} edit(s) applied before cancellation).")
        elif errors_occurred:
            plural_s = "s" if errors_occurred != 1 else ""
            logger.error(f"Error{plural_s} occurred while pending changes were being applied. This may cause the semantic search index to go out of sync with the actual data. Recommend deleting '{self.semantic_index_path}' and restarting the app to perform a full reindex.")
        else:
            logger.info("HybridIR.commit: All pending changes applied successfully.")

        logger.info("HybridIR.commit: Commit finished, exiting.")

    def _save_datastore(self) -> None:
        # We save embeddings separately, as compressed NumPy arrays, to save disk space.
        # Separate the embeddings from the rest of the data, being careful to not create extra in-memory copies (e.g. of the actual chunk texts or fulltexts).
        logger.info("HybridIR._save_datastore: entered. Preparing...")
        with self.datastore_lock:
            documents_without_embeddings = {}
            embeddings = []
            for document_id, doc in sorted(self.documents.items(),
                                           key=operator.itemgetter(0)):  # sort by document ID for debuggability
                tempdoc = copy.copy(doc)
                # `dict` preserves insertion order, so `embeddings` will be
                # in the same order as `self.documents.values()`
                embeddings.append(tempdoc.pop("embeddings"))
                documents_without_embeddings[document_id] = tempdoc
            data = {"embedding_model_name": self.embedding_model_name,
                    "documents": documents_without_embeddings}

            logger.info("HybridIR._save_datastore: Saving...")
            common_utils.create_directory(self.fulldocs_path)
            with open(self.fulldocs_documents_file, "w", encoding="utf-8") as json_file:
                # Keeping the amount of indentation small improves human-readability,
                # but also saves some disk space, as there are lots of indented lines in this file.
                json.dump(data, json_file, indent=2)

            # Note each document may have a different number of chunks, and each chunk
            # produces one embedding vector. This yields one 2D array per document (outer index = chunk).
            logger.info(f"HybridIR._save_datastore: Saving embeddings (model '{self.embedding_model_name}')...")
            np.savez_compressed(self.fulldocs_embeddings_file, *embeddings)

        logger.info("HybridIR._save_datastore: exiting, all done.")

    def _load_datastore(self) -> Tuple[Optional[str], Optional[str]]:
        logger.info("HybridIR._load_datastore: entered.")
        with self.datastore_lock:
            try:
                with open(self.fulldocs_documents_file, "r", encoding="utf-8") as json_file:
                    data = json.load(json_file)
                stored_embedding_model_name = data["embedding_model_name"]
                documents = data["documents"]

                # documents: {"document_id0": {...}, },  arrs: {"arr_0": np.array, ...};
                # arrs has one 2D array per document (outer index = chunk)
                arrs = np.load(self.fulldocs_embeddings_file)
                for doc, document_embeddings in zip(documents.values(), arrs.values()):
                    doc["embeddings"] = document_embeddings.tolist()  # same in-memory format as if freshly created

                plural_s = "s" if len(documents) != 1 else ""
                logger.info(f"HybridIR._load_datastore: Loaded datastore with embedding model '{stored_embedding_model_name}' from '{str(self.fulldocs_path)}' ({len(documents)} document{plural_s}).")
                return stored_embedding_model_name, documents
            except Exception:  # likely datastore not created yet
                logger.warning(f"HybridIR._load_datastore: While loading datastore from '{str(self.fulldocs_path)}'", exc_info=True)
                return None, None

    # TODO: support other media such as images (semantic embedding via `clip-ViT-L-14`, available in `sentence_transformers`; and keyword extraction by CLIP/Deepbooru)
    def _prepare_document_for_indexing(self, doc: Dict, on_progress: Optional[Callable[[str], None]] = None) -> Dict:
        """Chunk, tokenize and embed one document. Returns the new fields; does not mutate `self`.

        `on_progress`: Called with a short description of where this document is — `"tokenizing 240 / 288"`
                       — often enough to drive a live indicator. See `_set_indexing_progress`, which is what
                       the commit loop passes in.
        """
        document_id = doc["document_id"]
        text = doc["text"]

        # We split each document into chunks. The chunks themselves are useful
        # as the actual search results (the snippets that matched the search).
        logger.info(f"HybridIR._prepare_document_for_indexing: chunkifying document '{document_id}' ({len(text)} characters).")
        document_chunks = common_utils.chunkify_text(text, chunk_size=self.chunk_size, overlap=self.overlap, extra=0.4)  # -> [{"text": ..., "chunk_id": ..., "offset": ...}, ...]

        # Tokenizing each chunk enables keyword search. These are used by the keyword index (bm25s).
        #
        # Sent in batches, because the analysis is a round trip to raven-server and at chunk sizes the
        # round trip costs more than the analysis — ~57 ms of HTTP and serialization against ~25 ms of
        # GPU compute, measured 2026-08-06. Batching pays that once per batch instead of once per chunk:
        # on a 51-chunk document, 3.94 s -> 0.52 s, i.e. 77 ms -> 10 ms per chunk, a 7.5x speedup of what
        # was previously the dominant cost of indexing by a wide margin (22.5 s of 23.2 s on a 288-chunk
        # document).
        #
        # Batched rather than sent all at once so the progress report keeps moving: this is still the slow
        # step for a large document, and a single silent call would put a multi-second gap in the display
        # that reads as a stall. `TOKENIZE_BATCH_SIZE` is the trade — large enough to amortize the round
        # trip, small enough that the report updates a few times a second.
        logger.info(f"HybridIR._prepare_document_for_indexing: tokenizing document '{document_id}'.")
        tokenized_chunks = []
        for start in range(0, len(document_chunks), TOKENIZE_BATCH_SIZE):
            batch = document_chunks[start:start + TOKENIZE_BATCH_SIZE]
            if on_progress is not None:
                on_progress(f"tokenizing {start + len(batch)} / {len(document_chunks)}")
            tokenized_chunks.extend(self._tokenize_many([chunk["text"] for chunk in batch]))

        # Embedding each chunk enables semantic search. These are used by the vector index (chromadb).
        # One batched call for the whole document, and it does not need breaking up for progress: on the same
        # measurement it was 3% of the work (0.7 s for 288 chunks), because the embedder runs on the GPU.
        logger.info(f"HybridIR._prepare_document_for_indexing: computing semantic embeddings for document '{document_id}'.")
        if on_progress is not None:
            on_progress(f"embedding {len(document_chunks)} chunks")
        document_embeddings = self.embedder.encode([chunk["text"] for chunk in document_chunks])
        document_embeddings = document_embeddings.tolist()  # for JSON serialization

        prepdata = {"chunks": document_chunks,  # [{"text": ..., "chunk_id": ..., "offset": ...}, ...]
                    "tokens": tokenized_chunks,  # [[token0_of_chunk0, token1_of_chunk0, ...], [token0_of_chunk1, token1_of_chunk1, ...], ...]
                    "embeddings": document_embeddings,  # [vec_of_chunk0, vec_of_chunk1, ...]
                    }
        return prepdata

    # This is used both by the commit mechanism as well as the full index rebuild.
    def _add_document_to_vector_collection(self, doc: Dict) -> None:
        with self.datastore_lock:
            document_id = doc["document_id"]
            self._vector_collection.add(
                embeddings=doc["embeddings"],
                metadatas=[{"document_id": document_id,
                            "chunk_id": chunk["chunk_id"],
                            "full_id": format_chunk_full_id(document_id, chunk["chunk_id"]),
                            "offset": chunk["offset"],
                            "text": chunk["text"]} for chunk in doc["chunks"]],  # TODO: the vector storage technically doesn't need the "text" field, because we always read the full data records from the keyword index.
                ids=[format_chunk_full_id(document_id, chunk["chunk_id"]) for chunk in doc["chunks"]]
            )

    # TODO: We currently rebuild the whole BM25 index at every commit, which is slow.
    # The new document may have added new tokens so that the token vocabulary must be updated, and the `bm25s` library doesn't support adding documents to an existing index, anyway.
    def _rebuild_keyword_search_index(self) -> None:
        with self.datastore_lock:
            plural_s = "s" if len(self.documents) != 1 else ""
            logger.info(f"HybridIR._rebuild_keyword_search_index: Building keyword index for {len(self.documents)} document{plural_s} from main datastore. This may take a while.")
            corpus_records = []
            corpus_tokens = []
            for doc in self.documents.values():
                for chunk, tokens in zip(doc["chunks"], doc["tokens"]):
                    # All data here needs to be JSON serializable so that we can save these records to the BM25 corpus.
                    record = {"document_id": doc["document_id"],
                              "chunk_id": chunk["chunk_id"],
                              "full_id": format_chunk_full_id(doc["document_id"], chunk["chunk_id"]),
                              "offset": chunk["offset"],
                              "text": chunk["text"]}
                    corpus_records.append(record)
                    corpus_tokens.append(tokens)
            if self.documents:
                self._keyword_retriever = bm25s.BM25(corpus=corpus_records)
                self._keyword_retriever.index(corpus_tokens)

                # Save the updated index to disk.
                # NOTE: we don't save the vocab_dict, since we don't use the `Tokenizer` class from `bm25s`.
                logger.info("HybridIR._rebuild_keyword_search_index: Build complete. Saving keyword index.")
                self._keyword_retriever.save(str(self.keyword_index_path))
            else:  # No documents yet
                logger.info("HybridIR._rebuild_keyword_search_index: No documents. Doing nothing.")
                self._keyword_retriever = None

            self._build_full_id_to_record_index()

            logger.info("HybridIR._rebuild_keyword_search_index: done.")

    # We need to map from a chunk's "full_id" to the actual data record of that chunk when we fuse the search results.
    # Note the corresponding full document in the datastore is just `self.documents[record["document_id"]]`.
    #
    # This mapping is quick to build, so we don't bother persisting it to disk. (That does mean we have to regenerate just this part when loading the keyword index from disk.)
    def _build_full_id_to_record_index(self) -> None:
        with self.datastore_lock:
            if self._keyword_retriever is not None:
                self.full_id_to_record_index = {record["full_id"]: idx for idx, record in enumerate(self._keyword_retriever.corpus)}
            else:
                self.full_id_to_record_index = {}

    # TODO: add a variant of `query` with a fixed amount of context around each match (we can do this by looking up the fulltext of the matching chunk and taking the text from there)
    # TODO: do we need `exclude_documents`, for symmetry?
    def query(self,
              query: str,
              *,
              k: int = 10,
              alpha: float = 2.0,
              keyword_score_threshold: float = 0.1,
              semantic_distance_threshold: float = 0.8,
              include_documents: Optional[List[str]] = None,
              multi_query: bool = False,
              return_extra_info: bool = False) -> Union[List[Dict], Tuple[List[Dict], envcls]]:
        """Hybrid BM25 + Vector search with RRF fusion.

        `query`: Search query, of the kind you'd type into Google: space-separated keywords, or a natural-language question.
                 This is automatically tokenized for the keyword search, and semantically embedded for the semantic search.

        `k`: Return this many best results.

        `alpha`: Fudge factor. Retrieve `alpha * k` results, before cutting the final result at the best `k`.
                 If the initial results include adjacent chunks, those are auto-merged before the final list of results
                 is created. Hence it may be useful to first retrieve more than `k` best results, to increase the chances
                 of still having `k` results after any adjacent chunks have been combined.

        `keyword_score_threshold`: Ignore any keyword search results that have this score or less.
                                   The default `0.0` means to drop only results that did not match at all.

        `semantic_distance_threshold`: Ignore any semantic search results whose semantic distance to the query is this or more.
                                       Good values depend on the embedding you use, and possibly on the dataset.
                                       The default is for cosine distance using the default embedding model.

        `include_documents`: Optional list of document IDs. If provided, search only in the specified documents.

        `multi_query`: If `True`, also query with each sentence of `query` that is worth asking separately
                       (`split_into_subqueries`), and fuse all the result sets together.

                       **Defaults to `False`, because as it stands it does not work.** Measured against
                       `investigations/retrieval/` on 2026-08-05: no change on focused questions (0.535 MRR
                       either way, as expected — those do not split), and on the rambling ones it is
                       *worse* than not splitting at all (0.286 against 0.315 MRR, and R@20 0.64 → 0.50).

                       The mechanism is visible in the numbers and is worth knowing before trying again. A
                       rambling message yields five to seven subqueries, so the whole-message query holds
                       one vote in seven — and the context sentences it is outvoted by *agree with each
                       other* on generically topical documents, because that is what they are about. RRF
                       rewards agreement, so it promotes them. Which is the brief's own opening complaint,
                       reproduced by the fix for it: less topical matches outscoring the ones that answer
                       the question.

                       Kept rather than reverted because the machinery is right and reusable — the split,
                       the batched multi-query retrieval, the single flat fusion — and only the policy over
                       it is wrong. What it wants is fewer and better subqueries rather than every sentence
                       voting equally: see brief 09 lever 3 for the candidates.

                       Every backend here batches, so the extra queries cost extra rows rather than extra
                       round trips: one `bm25s.retrieve` over all token lists, one `encode` over all texts,
                       one Chroma query over all embeddings. Cost is not what is wrong with it.

        `return_extra_info`:
            If `True`: Return `final_results, report`, where `report` is an `env` with the fields

                           `keyword_results`, `keyword_scores`      what the BM25 arm found, and how well
                           `vector_results`, `vector_distances`     what the vector arm found, and how far

                       Each pair is index-aligned, and each is the union across all the queries actually
                       run (see `multi_query`) — a caller asking "what did the keyword arm find" wants
                       everything it found, not a per-query breakdown of it.

                       Plus `per_query`, which is the breakdown: one `env` per query actually run, with

                           `text`                         the query string
                           `candidate_keyword_scores`     BM25 scores of its retrieved candidates
                           `candidate_vector_distances`   cosine distances of its retrieved candidates

                       These are the raw scores as the engines returned them, in engine rank order and
                       *before* the two score thresholds — the score-to-quality mapping that fusion
                       discards. Read their shape with `score_sharpness` to tell a query that found
                       something from one whose best result is best by noise; note that the vector arm
                       reports distances, which have to be converted to similarities first.

                       This can be useful for debugging your knowledge base.
            If `False`: Return `final_results` only.

        In both return formats, the format of `final_results is`
            [{"document_id": the_id_string,
              "text": merged_contiguous_text,
              "offset": start_offset_in_document,
              "score": rrf_score},
             ...]
        """
        plural_s = "es" if k != 1 else ""
        logger.info(f"HybridIR.query: entered. Searching for {k} best match{plural_s} for '{query}'")
        try:
            return self._query_body(query=query,
                                    k=k,
                                    alpha=alpha,
                                    keyword_score_threshold=keyword_score_threshold,
                                    semantic_distance_threshold=semantic_distance_threshold,
                                    include_documents=include_documents,
                                    multi_query=multi_query,
                                    return_extra_info=return_extra_info)
        finally:
            self._query_progress_text = ""

    def _query_body(self,
                    query: str,
                    *,
                    k: int,
                    alpha: float,
                    keyword_score_threshold: float,
                    semantic_distance_threshold: float,
                    include_documents: Optional[List[str]],
                    multi_query: bool,
                    return_extra_info: bool):
        # The whole text is always queried; `split_into_subqueries` adds the sentences worth asking
        # separately, and returns nothing when there are none. Beside the whole rather than instead of it —
        # see that function for why splitting alone trades one failure for its mirror image.
        query_texts = [query]
        if multi_query:
            query_texts.extend(split_into_subqueries(query))
        if len(query_texts) > 1:
            logger.info(f"HybridIR.query: querying with the whole message and {len(query_texts) - 1} subqueries")

        # Prepare queries for keyword search (slow — runs the spaCy NLP pipeline; no datastore access).
        self._query_progress_text = "Tokenizing query…"
        query_tokens = [self._tokenize(text) for text in query_texts]

        # Prepare queries for vector search (slow — server roundtrip; no datastore access). One call for all
        # of them: the embedder batches, so the subqueries cost no extra round trip, only extra rows.
        self._query_progress_text = "Embedding query…"
        query_embeddings = self.embedder.encode(query_texts)

        # Pin the index references atomically and run both searches under the *same* `datastore_lock`
        # acquisition, so the keyword corpus and the chromadb state agree on the snapshot. A concurrent
        # `commit()` releases `datastore_lock` between iterations, so the wait here is bounded by one
        # iteration's mutation block (microseconds), not the whole commit.
        with self.datastore_lock:
            if not self.documents:
                logger.info("HybridIR.query: No documents in index, returning empty result.")
                # Shape-preserving: a caller that asked for extra info gets the pair it unpacks, with an
                # empty report, rather than a bare list that raises at the unpacking site. An empty index
                # is an ordinary state — a fresh datastore, or a `--db-dir` pointed at a directory that
                # does not exist yet, which `HybridIR` creates rather than rejects — so the caller most
                # likely to meet this is one that has not noticed anything is wrong yet.
                if return_extra_info:
                    return [], envcls(keyword_results=[],
                                      keyword_scores=[],
                                      vector_results=[],
                                      vector_distances=[],
                                      per_query=[envcls(text=text,
                                                        candidate_keyword_scores=[],
                                                        candidate_vector_distances=[])
                                                 for text in query_texts])
                return []
            if self._keyword_retriever is None:
                assert False  # we should have `self._keyword_retriever` as soon as we have at least one document
            keyword_retriever = self._keyword_retriever
            vector_collection = self._vector_collection
            full_id_to_record_index = self.full_id_to_record_index

            # `bm25s` requires `k ≤ corpus size`. Pinning above plus computing here keeps the size and the
            # subsequent `.retrieve` consistent.
            internal_k = min(int(alpha * k), len(keyword_retriever.corpus))
            if include_documents is None:
                keyword_k = internal_k
            else:  # return score for *every record in database*, for manual metadata-based filtering (document ID)
                keyword_k = len(keyword_retriever.corpus)

            # BM25 search
            self._query_progress_text = "Keyword search…"
            logger.info("HybridIR.query: keyword search")
            # Here we always search all documents; we filter afterward, if needed.
            raw_keyword_results, raw_keyword_scores = keyword_retriever.retrieve(query_tokens,  # list of list of tokens (outer list = one element per query; runs them all in one pass)
                                                                                  k=keyword_k)

            # Vector search
            self._query_progress_text = "Semantic search…"
            logger.info("HybridIR.query: semantic search")
            if include_documents is not None:  # search only documents with given IDs
                chroma_results = vector_collection.query(query_embeddings=list(query_embeddings),
                                                         n_results=internal_k,
                                                         include=["metadatas", "distances"],
                                                         where={"document_id": {"$in": include_documents}})
            else:  # search all documents
                chroma_results = vector_collection.query(query_embeddings=list(query_embeddings),
                                                         n_results=internal_k,
                                                         include=["metadatas", "distances"])
            # list of list of metadatas (outer list = one element per query)
            # https://github.com/chroma-core/chroma/blob/main/chromadb/api/types.py
            raw_vector_results = chroma_results["metadatas"]  # -> list (per query) of list of metadatas
            raw_vector_distances = chroma_results["distances"]  # -> list (per query) of list of float
        # Now we no longer need datastore access; the rest of the function uses the pinned references.

        # Filter each query's results by threshold (and by `include_documents`, if specified). One ranked
        # list per (query, engine) pair — the fusion below treats them all as peers, so a document has to
        # earn its place by being found repeatedly rather than by being found once, very well.
        keyword_hits = []  # per query: the corpus entries as-is
        vector_hits = []  # per query
        # Flat unions, for the log line and the report. A debugger asking "what did the keyword arm find"
        # wants everything it found; the report's `per_query` carries the breakdown separately.
        keyword_results, keyword_scores = [], []
        vector_results, vector_distances = [], []
        include_documents_set = set(include_documents) if include_documents is not None else set()  # for O(1) checking
        candidate_keyword_scores = []  # per query: raw scores, *before* the score threshold — see `score_sharpness`
        for i in range(len(query_texts)):
            per_query_keyword = []
            per_query_candidate_scores = []
            for j in range(raw_keyword_results.shape[1]):
                # https://github.com/xhluca/bm25s/blob/main/examples/save_and_reload_end_to_end.py
                keyword_result = raw_keyword_results[i, j]
                keyword_score = raw_keyword_scores[i, j]
                if include_documents is not None and keyword_result["document_id"] not in include_documents_set:
                    continue
                # With `include_documents` the BM25 arm scores the whole corpus and filters afterwards, so
                # take the best `internal_k` survivors of the filter — the same population it would have
                # retrieved had the filter been applied by the engine. Without it, `keyword_k` already is
                # `internal_k` and this truncation never fires.
                if len(per_query_candidate_scores) < internal_k:
                    per_query_candidate_scores.append(float(keyword_score))
                if keyword_score > keyword_score_threshold:
                    per_query_keyword.append(keyword_result)
                    keyword_results.append(keyword_result)
                    keyword_scores.append(keyword_score)
            keyword_hits.append(per_query_keyword)
            candidate_keyword_scores.append(per_query_candidate_scores)

            per_query_vector = []
            for vector_result, vector_distance in zip(raw_vector_results[i], raw_vector_distances[i]):
                if vector_distance < semantic_distance_threshold:
                    per_query_vector.append(vector_result)
                    vector_results.append(vector_result)
                    vector_distances.append(vector_distance)
            vector_hits.append(per_query_vector)

        logger.info("HybridIR.query: fusing results")

        # Fuse every list in one RRF call rather than fusing per query and then fusing the fusions. Nesting
        # would rank on `1/(rank + K)` values, which carry no information about how good a match was — the
        # same flattening RRF already performs once, applied twice.
        ranked_lists = []
        for i in range(len(query_texts)):
            # anything hashable that uniquely identifies each result -> use the full ID
            ranked_lists.append([record["full_id"] for record in keyword_hits[i]])
            ranked_lists.append([record["full_id"] for record in vector_hits[i]])
        rrf_results = reciprocal_rank_fusion(*ranked_lists)

        # Collect the actual data records for each full ID, and populate the fused scores.
        # Note we need the chunks, not the full documents. We can collect them from the
        # keyword-search corpus, regardless of which backend actually returned any specific result.
        fused_results = []
        for full_id, rrf_score in rrf_results:
            # Use the pinned references — `self._keyword_retriever` / `self.full_id_to_record_index` may
            # have been replaced by a concurrent `commit()` since we released the lock.
            #
            # Skip chunks unknown to the pinned BM25 corpus. ChromaDB is updated incrementally during
            # commit's per-doc loop, but the BM25 corpus is rebuilt only at commit's *end* — so during
            # a commit, chromadb may carry chunks for newly-added docs that aren't yet in BM25's index
            # (and so aren't in `full_id_to_record_index` either). Those chunks will become searchable
            # after the next commit's BM25 rebuild; skipping them here lets the search return promptly
            # with stable-corpus results instead of raising KeyError.
            if full_id not in full_id_to_record_index:
                continue
            record = copy.copy(keyword_retriever.corpus[full_id_to_record_index[full_id]])
            record["score"] = rrf_score
            fused_results.append(record)

        # Merge adjacent chunks, sorting the final results by the fused score.
        # Each merged chunk gets the score of the highest-scoring individual chunk that went into it.
        #
        # NOTE: Merged results don't have a "chunk_id" or "full_id" (design choice; multiple chunks may
        #       have been merged into each result, so chunk-specific fields wouldn't make sense), but only
        #       "document_id", "offset", "text", and "score" (RRF score).
        self._query_progress_text = "Merging results…"
        logger.info("HybridIR.query: merging contiguous spans in results")
        merged = merge_contiguous_spans(fused_results)

        kw_plural_s = "es" if len(keyword_results) != 1 else ""
        vec_plural_s = "es" if len(vector_results) != 1 else ""
        fused_plural_s = "es" if len(fused_results) != 1 else ""
        total_plural_s = "s" if len(merged) != 1 else ""
        logger.info(f"HybridIR.query: retrieved chunk statistics: {len(keyword_results)} keyword match{kw_plural_s}, {len(vector_results)} semantic match{vec_plural_s}; total {len(fused_results)} unique match{fused_plural_s}; {len(merged)} result{total_plural_s} after merging contiguous spans from each document.")

        # Drop extra results, if there are still too many at this point.
        plural_s = "s" if k != 1 else ""
        logger.info(f"HybridIR.query: Returning up to {k} best result{plural_s} (out of {len(merged)} retrieved), sorted by RRF score.")
        merged = merged[:k]

        logger.info("HybridIR.query: exiting. All done.")

        if return_extra_info:
            per_query = [envcls(text=query_texts[i],
                                candidate_keyword_scores=candidate_keyword_scores[i],
                                candidate_vector_distances=[float(d) for d in raw_vector_distances[i]])
                         for i in range(len(query_texts))]
            return merged, envcls(keyword_results=keyword_results,
                                  keyword_scores=keyword_scores,
                                  vector_results=vector_results,
                                  vector_distances=vector_distances,
                                  per_query=per_query)
        return merged

# --------------------------------------------------------------------------------

bg = None
task_managers = {}
def init(executor):
    """Initialize this module.

    If you use the all-in-one convenience function `setup`, you do not need `init`;
    `setup` calls `init` automatically.

    Otherwise, `init` must be called before `HybridIRFileSystemEventHandler`
    (including its `rescan` method) can be used.

    `executor`: A `ThreadPoolExecutor` or something duck-compatible with it.
                Used for running the background tasks for ingesting files
                and committing search index changes.
    """
    global bg
    if bg is not None:  # already initialized?
        return
    bg = executor
    try:
        # Ingestion for multiple files can proceed concurrently. The ingestion step might also be slow,
        # if the plaintext needs to be extracted from a binary file by a user callback.
        #
        # For search index commits, only one commit should be running at any given time.
        #
        # These share the same executor, so this takes no additional OS resources.
        task_managers["ingest"] = bgtask.TaskManager(name="hybridir_ingest",
                                                     mode="concurrent",
                                                     executor=bg)
        task_managers["commit"] = bgtask.TaskManager(name="hybridir_commit",
                                                     mode="sequential",  # for the auto-cancel mechanism
                                                     executor=bg)
        # Belt-and-suspenders: also signal cancellation via atexit, in case the app exits via a path that
        # doesn't go through `shutdown` (crashes, KeyboardInterrupt during startup, etc.). `wait=False` lets
        # this return immediately; the actual thread join happens later via concurrent.futures' `_python_exit`.
        atexit.register(lambda: shutdown(wait=False))
    except Exception:
        bg = None
        task_managers.clear()
        raise


def has_pending_work() -> bool:
    """Whether any document ingest or index commit is queued or running, module-wide.

    Complements `HybridIR.is_indexing`, which answers a narrower question — whether *that instance* is
    inside `commit()` — and is therefore False during the ingest phase, when documents are being read
    and their text extracted. A batch client that waits on `is_indexing` alone will conclude that a
    corpus of a thousand PDFs is finished a few seconds after starting, because the first commit has not
    been scheduled yet and reading them all takes minutes.

    Module-wide rather than per-instance because the task managers are: `init` builds one "ingest" and
    one "commit" manager for the module, shared by every `HybridIR`. A process driving two indexes at
    once cannot use this to tell them apart, which is fine for its intended caller (`raven-indexer`,
    one index per run) and is why the docstring says so rather than implying otherwise.
    """
    return any(manager.has_tasks() for manager in task_managers.values())


def shutdown(wait: bool = True) -> None:
    """Cancel any in-flight RAG indexing and ingestion tasks.

    On `wait=True` (default), block until the running `commit()` has observed the cancellation flag, exited
    its per-document loop, and finished its partial-save tail. Use this on app shutdown to ensure that
    whatever indexing work was applied before cancellation is persisted before the process exits.

    On `wait=False`, return immediately after signalling. The flag is set on each task's env, but the
    join happens later — at app exit, via `concurrent.futures._python_exit`. This is the right call from
    a crash-path atexit hook where blocking is undesirable.

    Drain order is **ingest before commit** (preserved by the dict's insertion-order semantics): a
    finishing ingest task submits a new commit on its way out, so commits must be drained *after* every
    ingest has exited or a fresh commit task could slip in behind us. Also worth noting: this only
    cancels module-level task work. Watchdog observers are owned by individual `HybridIRFileSystemEventHandler`
    instances; stop them via the instance's own `shutdown` method (also registered via `atexit`).
    """
    for task_manager in task_managers.values():
        task_manager.clear(wait=wait)

# --------------------------------------------------------------------------------

# See e.g. https://www.kdnuggets.com/monitor-your-file-system-with-pythons-watchdog
class HybridIRFileSystemEventHandler(watchdog.events.FileSystemEventHandler):
    def __init__(self,
                 docs_dir: Union[str, pathlib.Path],
                 recursive: bool,
                 retriever: HybridIR,
                 extractor: Optional[docextract.Extractor] = None) -> None:
        """Simple auto-updater that monitors a directory and auto-commits changes to a `HybridIR`.

        `docs_dir`: The path to monitor.

        `recursive`: Whether to monitor also subdirectories.

                     Cannot be changed while running.

                     If you need to re-instantiate, call the `shutdown` method of the old instance
                     before deleting it, to make its directory monitor exit.

                     If you never delete the instance, there is no need to bother - this constructor
                     sets up an exit trigger automatically, so that the directory monitor shuts down
                     cleanly when the app exits. If the instance has been deleted, the exit trigger no-ops.

        `retriever`: The `HybridIR` instance to send changes to, to automatically keep it up to date.

        `extractor`: A `raven.common.docextract.Extractor`: which file formats to monitor, and how to
                     read one. Files whose extension it does not claim are ignored; the text it returns
                     for the rest is what goes into `retriever`'s search index.

                     Defaults to `docextract.PLAINTEXT`, the formats readable without a parser. Pass
                     `docextract.ALL_FORMATS` for PDFs, the office formats and saved web pages —
                     which is what Librarian does, narrowed by `librarian_config.llm_docs_exts`.

                     The reader and the format list travel together on purpose: a reader handed a
                     format it cannot parse indexes line noise, and the document then sits in the
                     index findable and wrong, which is worse than never ingesting it.

        Uses the `watchdog` library.
        """
        # Canonical (not symlink-resolved) — `_make_document_id_from_path` takes each document's path
        # relative to this, so the two must be normalized the same way or every id lookup fails.
        self.docs_dir = common_utils.canonical_path(docs_dir)
        self.recursive = recursive
        self.retriever = retriever
        self.extractor = extractor if extractor is not None else docextract.PLAINTEXT

        self._docs_observer = None  # populated by `bootup`
        self._shutdown_lock = threading.RLock()

        # For delayed commit (commit when new/modified files stop appearing in quick succession)
        self._status_box = box()
        self._lock = threading.RLock()
        def commit(task_env: envcls) -> None:
            assert task_env is not None
            logger.debug(f"HybridIRFileSystemEventHandler.commit: {task_env.task_name}: Entered.")
            if task_env.cancelled:  # while waiting in queue
                logger.debug(f"HybridIRFileSystemEventHandler.commit: {task_env.task_name}: Cancelled.")
                return
            logger.debug(f"HybridIRFileSystemEventHandler.commit: {task_env.task_name}: Committing changes to HybridIR (may take a while; cancellable at per-document granularity).")
            self.retriever.commit(task_env=task_env)
            logger.debug(f"HybridIRFileSystemEventHandler.commit: {task_env.task_name}: Done.")
        self.uuid = str(uuid.uuid4())
        self.commit_task = bgtask.ManagedTask(category=f"raven_librarian_HybridIRFileSystemEventHandler_{self.uuid}_commit",
                                              entrypoint=commit,
                                              running_poll_interval=1.0,
                                              pending_wait_duration=1.0)
        self.bootup()

    def bootup(self):
        """Scan for offline changes, start the directory monitor, and set up the app-exit hook for monitor shutdown."""

        # Rescan docs directory for changes made while the app was not running.
        self.rescan(self.docs_dir,
                    recursive=self.recursive)

        # Register handler to auto-update search indices on live changes in docs directory.
        self._docs_observer = watchdog.observers.Observer()
        self._docs_observer.schedule(self,
                                     path=self.docs_dir,
                                     recursive=self.recursive)
        self._docs_observer.start()

        # And make sure it shuts down gracefully at app exit.
        atexit.register(self.shutdown)

    def shutdown(self):
        """Make the directory monitor exit gracefully.

        This is normally only used as the app-exit hook, but if you need to re-instantiate,
        then call the `shutdown` method of the old instance before creating the new one.
        """
        with self._shutdown_lock:
            try:  # EAFP
                self.docs_observer.stop()
                self.docs_observer.join()
            except AttributeError:  # `self.docs_observer is None` already
                pass
            self.docs_observer = None

    # `document_id` needs to be unique, but easily mappable from filename, persistently.
    def _make_document_id_from_path(self, path: Union[pathlib.Path, str]) -> str:
        p = pathlib.Path(path) if not isinstance(path, pathlib.Path) else path
        relp = p.relative_to(self.docs_dir)
        return str(relp)

    def _sanity_check(self, path: Union[pathlib.Path, str]) -> bool:
        if not task_managers:
            logger.warning("HybridIRFileSystemEventHandler._sanity_check: Module not initialized, cannot proceed.")
            return False
        abspath = str(common_utils.canonical_path(path))
        if not self.extractor.handles(abspath):
            logger.info(f"HybridIRFileSystemEventHandler._sanity_check: file '{abspath}': file extension not in monitored list {list(self.extractor.extensions)}, ignoring file.")
            return False
        return True

    def _read(self, path: Union[pathlib.Path, str]) -> Optional[str]:
        abspath = common_utils.canonical_path(path)
        # The extractor raises on an unreadable document — a corrupt/encrypted PDF, a non-UTF-8 text file, or a
        # file that vanished between the event and this read. In a background batch ingest the right policy is to
        # skip that one file and keep going, not to let the exception abort the ingest task, so we catch here and
        # treat it as "no content" downstream.
        try:
            content = self.extractor(abspath)
        except Exception as exc:  # noqa: BLE001 -- one unreadable document must not abort the whole batch ingest
            logger.warning(f"HybridIRFileSystemEventHandler._read: file '{abspath}': extraction failed, "
                           f"skipping: {type(exc)}: {exc}")
            return None
        if not content:
            return None
        if not isinstance(content, str):
            logger.error(f"HybridIRFileSystemEventHandler._read: file '{str(abspath)}': got non-string content. Ignoring file.")
            return None
        return content.strip()

    def _make_delete_task(self, document_id: str) -> Callable:
        """Deletion by id, for a document whose file is gone.

        Separate from `_make_task` because it is the one operation with no file to name: the caller has an
        index record and nothing on disk. Deriving the id from the record's stored path instead would fail
        exactly when the documents directory has been renamed, moved or symlinked since indexing — which is
        also when a rescan is most likely to be running.
        """
        def scheduled_delete(task_env: envcls) -> None:
            logger.debug(f"HybridIRFileSystemEventHandler.scheduled_delete: document '{document_id}': deleting from search indices.")
            self.retriever.delete(document_id)
            logger.debug(f"HybridIRFileSystemEventHandler.scheduled_delete: document '{document_id}': scheduling commit to save changes to HybridIR.")
            task_managers["commit"].submit(self.commit_task, envcls(wait=True))
        return scheduled_delete

    def _make_task(self, kind: str, path: Union[pathlib.Path, str]) -> Callable:
        abspath = common_utils.canonical_path(path)
        document_id = self._make_document_id_from_path(abspath)
        if kind == "add":
            def scheduled_add(task_env: envcls) -> None:
                logger.debug(f"HybridIRFileSystemEventHandler.scheduled_add: file '{path}': ingesting file content.")
                content = self._read(abspath)
                if content is None:
                    logger.debug(f"HybridIRFileSystemEventHandler.scheduled_add: file '{path}': got empty or non-string content, ignoring file.")
                    return
                self.retriever.add(document_id=document_id,
                                   path=str(abspath),
                                   text=content)
                logger.debug(f"HybridIRFileSystemEventHandler.scheduled_add: file '{path}': scheduling commit to save changes to HybridIR.")
                task_managers["commit"].submit(self.commit_task, envcls(wait=True))  # Schedule delayed commit after each add
            return scheduled_add

        elif kind == "update":
            def scheduled_update(task_env: envcls) -> None:
                logger.debug(f"HybridIRFileSystemEventHandler.scheduled_update: file '{path}': ingesting file content.")
                content = self._read(abspath)
                if content is None:
                    logger.warning(f"HybridIRFileSystemEventHandler.scheduled_update: file '{path}': got empty or non-string content from updated file; removing file from index.")
                    self.retriever.delete(document_id)
                else:
                    self.retriever.update(document_id=document_id,
                                          path=str(abspath),
                                          text=content)
                logger.debug(f"HybridIRFileSystemEventHandler.scheduled_update: file '{path}': scheduling commit to save changes to HybridIR.")
                task_managers["commit"].submit(self.commit_task, envcls(wait=True))  # Schedule delayed commit after each update
            return scheduled_update

        elif kind == "delete":
            def scheduled_delete(task_env: envcls) -> None:
                logger.debug(f"HybridIRFileSystemEventHandler.scheduled_delete: file '{path}': deleting from search indices.")
                self.retriever.delete(document_id)

                logger.debug(f"HybridIRFileSystemEventHandler.scheduled_delete: file '{path}': scheduling commit to save changes to HybridIR.")
                task_managers["commit"].submit(self.commit_task, envcls(wait=True))  # Schedule delayed commit after each delete
            return scheduled_delete

        else:
            raise ValueError(f"Unknown kind '{kind}'; expected one of 'add', 'update', 'delete'.")

    def rescan(self, path: Union[pathlib.Path, str], recursive: bool = False) -> None:
        """Rescan for documents at `path`.

        This adds/updates/deletes files from the retriever's index as necessary.

        Useful at app startup, since events only fire on live changes (while the app is running).
        """
        logger.info(f"HybridIRFileSystemEventHandler.rescan: Scanning '{path}' for offline changes (changes made while this app was not running).")
        abspath = common_utils.canonical_path(path)
        found_paths = []
        for root, dirs, files in os.walk(abspath):
            if not recursive:
                dirs.clear()
            for filename in files:
                filepath = os.path.join(root, filename)
                if self._sanity_check(filepath):
                    found_paths.append(str(common_utils.canonical_path(filepath)))
        plural_s = "s" if len(found_paths) != 1 else ""
        logger.info(f"HybridIRFileSystemEventHandler.rescan: Found {len(found_paths)} file{plural_s}.")

        # Compare by `document_id` -- the path *relative* to the documents directory -- and not by the
        # absolute path each record also stores. The relative path is what identifies a document, and it is
        # what survives the documents directory being renamed, moved, or reached through a symlink; the
        # absolute path is a fact about where the collection sits today. Keying on the latter makes a
        # collection reached by a second spelling of the same directory read as an entirely different one:
        # every file new, every indexed document deleted. It also *raised*, because building the deletion
        # task re-derived an id from the stored path, and a stored path outside the current documents
        # directory has no relative form -- the same "'<target>' is not in the subpath of '<documents dir>'"
        # that symlinked documents used to fail with, surviving on the deletion path.
        with self.retriever.datastore_lock:
            def came_from_file(doc: Dict) -> bool:  # convention: in-memory sources use paths of the form "<document_name_here>"
                return not (doc["path"].startswith("<") and doc["path"].endswith(">"))
            indexed_document_ids = {document_id for document_id, doc in self.retriever.documents.items()
                                    if came_from_file(doc)}
            # Safe to derive: every found path came from walking `self.docs_dir`, so it is under it by
            # construction. This is the direction that always works, which is why deletion is handled by
            # id below rather than by re-deriving one from a stored path.
            found_document_ids = {self._make_document_id_from_path(path): path for path in found_paths}

            def is_file_updated(document_id: str, path: str) -> bool:
                stats = self.retriever._stat(path)
                doc = self.retriever.documents[document_id]
                mtime_increased = (stats["mtime"] > doc["mtime"])
                filesize_changed = (stats["size"] != doc["filesize"])
                return mtime_increased or filesize_changed

            new_found_paths = [path for document_id, path in found_document_ids.items()
                               if document_id not in indexed_document_ids]
            updated_paths = [path for document_id, path in found_document_ids.items()
                             if document_id in indexed_document_ids and is_file_updated(document_id, path)]
            deleted_document_ids = [document_id for document_id in indexed_document_ids
                                    if document_id not in found_document_ids]

        new_plural_s = "s" if len(new_found_paths) != 1 else ""
        updated_plural_s = "s" if len(updated_paths) != 1 else ""
        deleted_plural_s = "s" if len(deleted_document_ids) != 1 else ""
        logger.info(f"HybridIRFileSystemEventHandler.rescan: Scan complete. Found {len(new_found_paths)} new file{new_plural_s}, {len(updated_paths)} updated file{updated_plural_s}, and {len(deleted_document_ids)} deleted file{deleted_plural_s}.")

        for path in new_found_paths:
            logger.info(f"HybridIRFileSystemEventHandler.rescan: File '{path}' is new: scheduling ingest.")
            task_managers["ingest"].submit(self._make_task(kind="add", path=path), envcls())
        for path in updated_paths:
            logger.info(f"HybridIRFileSystemEventHandler.rescan: File '{path}' was updated: scheduling ingest.")
            task_managers["ingest"].submit(self._make_task(kind="update", path=path), envcls())
        for document_id in deleted_document_ids:
            logger.info(f"HybridIRFileSystemEventHandler.rescan: Document '{document_id}' was deleted: scheduling deletion from index.")
            task_managers["ingest"].submit(self._make_delete_task(document_id), envcls())

    def on_created(self, event) -> None:
        path = event.src_path
        logger.info(f"HybridIRFileSystemEventHandler.on_created: File '{path}'.")
        if not self._sanity_check(path):
            return
        logger.info(f"HybridIRFileSystemEventHandler.on_created: File '{path}': scheduling ingest.")
        task_managers["ingest"].submit(self._make_task(kind="add", path=path), envcls())

    def on_modified(self, event) -> None:
        path = event.src_path
        logger.info(f"HybridIRFileSystemEventHandler.on_modified: File '{path}'.")
        if not self._sanity_check(path):
            return
        logger.info(f"HybridIRFileSystemEventHandler.on_created: File '{path}': scheduling ingest.")
        task_managers["ingest"].submit(self._make_task(kind="update", path=path), envcls())

    def on_deleted(self, event) -> None:
        path = event.src_path
        logger.info(f"HybridIRFileSystemEventHandler.on_deleted: File '{path}'.")
        if not self._sanity_check(path):
            return
        logger.info(f"HybridIRFileSystemEventHandler.on_deleted: File '{path}': scheduling deletion from index.")
        task_managers["ingest"].submit(self._make_task(kind="delete", path=path), envcls())

    # TODO: Do we need `on_moved`, too?

# --------------------------------------------------------------------------------

def setup(docs_dir: Union[pathlib.Path, str],
          recursive: bool,
          db_dir: Union[pathlib.Path, str],
          extractor: Optional[docextract.Extractor] = None,
          embedding_model_name: str = "sentence-transformers/multi-qa-mpnet-base-cos-v1",
          local_model_loader_fallback: bool = True,
          chunk_size: int = 1000,
          overlap_fraction: float = 0.25,
          executor: Optional[concurrent.futures.Executor] = None) -> Tuple[HybridIR, HybridIRFileSystemEventHandler]:
    """Set up hybrid keyword/semantic search for a directory containing document files.

    This is a convenience function that wires up both `HybridIR` and `HybridIRFileSystemEventHandler`
    in one go. When your app starts, point `setup` to the correct directories, and the search indices
    will work automagically. This includes an initial rescan when you call `setup`, to detect any changes
    to the documents folder that occurred while the app was not running.

    However, note that it is still the caller's responsibility to actually perform searches
    (see `HybridIR.query`) and to actually feed the search results to the LLM's context
    if you want to use `HybridIR` as a retrieval-augmented generation (RAG) backend.

    `docs_dir`: The directory the user puts documents in.
    `recursive`: Whether subdirectories of `docs_dir` are document directories, too.

    `db_dir`: The directory for storing search indices.

    `extractor`: Passed on to `HybridIRFileSystemEventHandler`, which see. Says both which formats to
                 ingest and how to read them; defaults to plain text only.

    `embedding_model_name`: passed on to `HybridIR`, which see.
    `local_model_loader_fallback`: passed on to `HybridIR`, which see.
    `chunk_size`: passed on to `HybridIR`, which see.
    `overlap_fraction`: passed on to `HybridIR`, which see.

    `executor`: A `ThreadPoolExecutor` or something duck-compatible with it.
                Passed on to `init`.

                If not provided, a new `ThreadPoolExecutor` is instantiated.

                Used for running the background tasks for ingesting files
                and committing search index changes.

    Returns the tuple `(retriever, scanner)`, where `retriever` is a `HybridIR` instance,
    and `scanner` is a `HybridIRFileSystemEventHandler` instance.
    """
    if executor is None:
        executor = concurrent.futures.ThreadPoolExecutor()
    init(executor=executor)

    # `HybridIR` also autoloads and auto-persists its search indices.
    retriever = HybridIR(datastore_base_dir=db_dir,
                         embedding_model_name=embedding_model_name,
                         local_model_loader_fallback=local_model_loader_fallback,
                         chunk_size=chunk_size,
                         overlap_fraction=overlap_fraction)

    # The watchdog observer can't watch a non-existent directory; without this, a fresh-install or
    # docs-dir-moved-aside startup raises FileNotFoundError from `inotify_add_watch` during `bootup()`.
    docs_dir_path = common_utils.canonical_path(docs_dir)
    if not docs_dir_path.is_dir():
        logger.info(f"setup: Documents directory '{str(docs_dir_path)}' does not exist; creating it.")
        common_utils.create_directory(docs_dir_path)

    scanner = HybridIRFileSystemEventHandler(docs_dir=docs_dir,
                                             recursive=recursive,
                                             retriever=retriever,
                                             extractor=extractor)

    return retriever, scanner
