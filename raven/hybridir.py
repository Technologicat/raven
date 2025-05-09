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

__all__ = ["HybridIR"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from collections import defaultdict
import copy
import json
import operator
import pathlib
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from unpythonic import allsame, uniqify

# NLP
from sentence_transformers import SentenceTransformer
import spacy

# database
import bm25s  # keyword
import chromadb  # semantic (vector)

from . import config
from . import utils

# --------------------------------------------------------------------------------
# Bootup

def load_stopwords():
    from spacy.lang.en import English
    nlp_en = English()
    stopwords = nlp_en.Defaults.stop_words
    return stopwords

def load_nlp_pipeline():
    spacy.require_gpu()
    nlp = spacy.load(config.spacy_model)

    try:
        spacy.require_gpu()
        logger.info("NLP model will run on GPU (if available).")
    except Exception as exc:
        logger.warning(f"exception while enabling GPU: {type(exc)}: {exc}")
        spacy.require_cpu()
        logger.info("NLP model will run on CPU.")

    try:
        nlp = spacy.load(config.spacy_model)
    except OSError:
        # https://stackoverflow.com/questions/62728854/how-to-place-spacy-en-core-web-md-model-in-python-package
        logger.info("Downloading language model for spaCy (don't worry, this will only happen once)...")
        from spacy.cli import download
        download(config.spacy_model)
        nlp = spacy.load(config.spacy_model)

    return nlp

stopwords = load_stopwords()
nlp = load_nlp_pipeline()

# --------------------------------------------------------------------------------

def format_chunk_full_id(document_id: str, chunk_id: str) -> str:
    """Generate an identifier for a chunk of a document, based on the given IDs."""
    return f"doc-{document_id}-chunk-{chunk_id}"

def chunkify(text: str, chunk_size: int, overlap: int, extra: float, trimmer: Optional[Callable] = None) -> List[Dict]:
    """Sliding window chunker with overlap, for chunking documents for fine-grained search.

    See also `merge_contiguous_spans`, which does unchunking for the search results.

    `text`: The text to be chunked.

    `chunk_size`: The length of one chunk, in characters (technically, Unicode codepoints, because Python's internal string format).

                  The final chunk may be up to 20% larger, to avoid leaving a very short chunk at the end (if the length of `text`
                  did not divide well with `chunk_size`).

    `extra`:   Orphan control parameter, as fraction of `chunk_size`, to avoid leaving a very small amount of text
               into a chunk of its own at the end of the document (in the common case where the length of the document
               does not divide evenly by `chunk_size`).

               E.g. `extra=0.4` allows placing an extra 40% of `chunk_size` of text into the last chunk of the document.
               Hence the remainder of text at the end of the document is split into a separate small chunk only if
               that extra 40% is not enough to accommodate it. If it fits into that, we instead make the previous chunk
               larger (by up to 40%), and place the remainder there.

    `overlap`: How much of the end of the previous chunk should be included in the next chunk,
               to avoid losing context at the seams.

               E.g. if `chunk_size` is 2000 characters and you want a 25% overlap, set `overlap=500`.

               For non-overlapping fixed-size chunking, set `overlap=0`.

    `trimmer`: Optional callback to clean up the end of a chunk, e.g. to a whole-sentence or whole-word boundary.
               Signature: str -> (str, int)

               The `trimmer` receives the text of the chunk as input. It must return a tuple `(trimmed_chunk, offset)`,
               where `offset` means how many characters were trimmed from the beginning. If you trim the end only,
               then `offset=0`.

               Note that when a trimmer is in use:
                   - The final size of any given chunk, after trimming, may be smaller than `chunk_size`.
                   - `overlap` is counted backward from the end of the *trimmed* chunk.

               An NLP pipeline can be useful as a component for building a high-quality trimmer.

    Returns a list of chunks of the form `{"text": actual_content, "chunk_id": running_number, "offset": start_offset_in_original_text}`.
    The `chunk_id` is provided primarily just for information and for debugging. The chunks are numbered 0, 1, ...
    The offsets are used by `merge_contiguous_spans` for unchunking search results.

    If `text` is at most `chunk_size` characters in length, returns a single chunk in the same format.
    """
    # TODO: better `extra` mechanism: adjust chunk size instead, to spread the extra content evenly?

    if len(text) <= (1 + extra) * chunk_size:
        return [{"text": text, "chunk_id": 0, "offset": 0}]

    chunks = []
    chunk_id = 0
    start = 0
    is_last = False
    while start < len(text):
        if len(text) - start <= (1 + extra) * chunk_size:
            chunk = text[start:]
            is_last = True
        else:
            chunk = text[start:start + chunk_size]

        if trimmer is not None:
            chunk, offset = trimmer(chunk)
            start = start + offset

        chunks.append({"text": chunk,
                       "chunk_id": chunk_id,
                       "offset": start})
        if is_last:
            break
        delta = len(chunk) - overlap
        if delta <= 0:
            assert False
        start += delta
        chunk_id += 1
    return chunks

def tokenize(text: str) -> List[str]:
    """Apply lowercasing, tokenization, stemming, stopword removal.

    Returns a list of tokens.

    We use a spaCy NLP pipeline to do the analysis.
    """
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stopwords]
    return tokens

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

class HybridIR:
    def __init__(self,
                 datastore_base_dir: Union[str, pathlib.Path],
                 embedding_model_name: str = "sentence-transformers/multi-qa-mpnet-base-cos-v1",
                 chunk_size: int = 1000,
                 overlap_fraction: float = 0.25):
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
        self._lock = threading.RLock()

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

        self._semantic_model = SentenceTransformer(self.embedding_model_name)  # we compute vector embeddings manually (on Raven's side)

        # Semantic search: ChromaDB vector storage
        # ChromaDB persists data automatically when we use the `PersistentClient`
        # https://docs.trychroma.com/docs/collections/create-get-delete
        self._vector_client = chromadb.PersistentClient(path=str(self.semantic_index_path),
                                                        settings=chromadb.Settings(anonymized_telemetry=False))
        try:
            self._vector_collection = self._vector_client.get_collection(name="embeddings")  # try loading existing vector index
        except Exception as exc:  # vector index missing
            logger.warning(f"HybridIR.__init__: While loading vector index from '{str(self.semantic_index_path)}': {type(exc)}: {exc}")

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
        except Exception as exc:  # keyword index missing
            logger.warning(f"HybridIR.__init__: While loading keyword index from '{str(self.keyword_index_path)}': {type(exc)}: {exc}")
            logger.info(f"HybridIR.__init__: Keyword index not found. Will create keyword index at '{str(self.keyword_index_path)}'.")

            if self.documents:  # suppress log message when no documents to process
                # TODO: full reindexing is slow. This should run as a bgtask.
                self._rebuild_keyword_search_index()
            else:  # No documents yet.
                self._keyword_retriever = None
                self.full_id_to_record_index = {}

        self._pending_edits = []

    def add(self, document_id: str, path: str, text: str) -> str:
        """Queue a document for adding into the index. To save changes, call `commit`.

        `document_id`: must be unique. Recommended to use `unpythonic.gensym(os.path.basename(path))` or something.
        `path`: Full path (or URL) of the original file. `HybridIR` itself doesn't use it; it's for your convenience
                so that you can easily locate the original file a given search result refers to.
        `text`: Plain-text content of the file, to be indexed for searching.

        Returns `document_id`, for convenience.
        """
        logger.info(f"HybridIR.add: Queuing document '{document_id}' for adding to index.")

        # Document-level data. This goes into the main datastore, which is also persisted so that we can rebuild indices when needed.
        #
        # Here we populate just those fields that can be filled quickly, so that `add` can return instantly.
        # Fields that require expensive computation (chunks, tokens, embeddings) are added at commit time, by `_prepare_document_for_indexing`.
        document = {"document_id": document_id,
                    "path": path,  # path of original file (e.g. to be able to open it in a PDF reader)
                    "text": text,  # copy of original text as-is
                    }

        # Pending document mechanism so that we can add a set of documents at once, and *then* rebuild the indices.
        with self._lock:  # may block until an ongoing `commit` (if any) finishes
            self._pending_edits.append(("add", document))

    def delete(self, document_id: str) -> None:
        """Queue a document for deletion. To save changes, call `commit`.

        `document_id`: the ID you earlier gave to `add`.
        """
        logger.info(f"HybridIR.add: Queuing document '{document_id}' for deletion from index.")
        with self._lock:
            self._pending_edits.append(("delete", document_id))

    def update(self, document_id: str, path: str, text: str) -> str:
        """Queue a document for updating. To save changes, call `commit`.

        `document_id`: as in `add`.
        `path`: as in `add`. If you need the previous path, you can get it as `your_hybrid_ir.documents[document_id]["path"]`.
        `text`: as in `add`.

        Returns `document_id`, for convenience.
        """
        logger.info(f"HybridIR.add: Queuing document '{document_id}' for update in index (will delete, then re-add).")
        with self._lock:
            self.delete(document_id)
            self.add(document_id, path, text)
            return document_id

    # TODO: Index rebuilding is slow. Maybe `commit` should run as a bgtask, like the BibTeX importer.
    # TODO: `commit` is not as atomic as I'd like. If anything goes wrong, the vector index loses sync with the actual data, necessitating a full rebuild. Check if ChromaDB has transaction management.
    def commit(self) -> None:
        """Commit pending changes (adds/deletes/updates), re-indexing the databases.

        An update is internally a delete, followed by an add for the updated version of the same document.
        """
        logger.info("HybridIR.commit: entered.")
        with self._lock:
            if not self._pending_edits:
                logger.info("HybridIR.commit: No pending changes, exiting.")
                return

            # Update `self.documents` and the semantic search index.
            # There is no "update" operation - to do that, first "delete", then "add".
            logger.info("HybridIR.commit: Applying pending changes.")
            try:
                for edit_kind, data in self._pending_edits:
                    if edit_kind == "add":
                        doc = data
                        document_id = doc["document_id"]
                        logger.info(f"HybridIR.commit: Adding document '{document_id}'.")

                        if document_id in self.documents:
                            logger.warning(f"HybridIR.commit: Document with ID '{document_id}' already exists in index; ignoring. If you meant to update, first delete, then add.")
                            continue

                        doc.update(self._prepare_document_for_indexing(doc))  # the slow part: chunkify, tokenize, embed

                        self.documents[document_id] = doc
                        self._add_document_to_vector_collection(doc)

                    elif edit_kind == "delete":
                        document_id = data

                        logger.info(f"HybridIR.commit: Deleting document '{document_id}'.")
                        try:
                            old_chunk_ids = [chunk["chunk_id"] for chunk in self.documents[document_id]["chunks"]]
                            self.documents.pop(document_id)
                            self._vector_collection.delete(ids=old_chunk_ids)
                        except KeyError as exc:
                            logger.warning(f"HybridIR.commit: Ignoring error: While deleting document with ID '{document_id}': {type(exc)}: {exc}")

                    else:  # should not happen, but let's log it
                        msg = f"HybridIR.commit: Unknown pending change type '{edit_kind}'. Ignoring."
                        logger.warning(msg)

            except Exception as exc:
                logger.error(f"While applying changes: {type(exc)}: {exc}")
                logger.error(f"The above error may have corrupted the semantic search index. Recommend deleting '{self.semantic_index_path}' and restarting the app to perform a full reindex.")
                raise

            self._rebuild_keyword_search_index()

            logger.info("HybridIR.commit: All changes applied, clearing pending changes list.")
            self._pending_edits = []

            self._save_datastore()

            logger.info("HybridIR.commit: Commit finished, exiting.")

    def _save_datastore(self):
        # We save embeddings separately, as compressed NumPy arrays, to save disk space.
        # Separate the embeddings from the rest of the data, being careful to not create extra in-memory copies (e.g. of the actual chunk texts or fulltexts).
        logger.info("HybridIR._save_datastore: entered. Preparing...")
        documents_without_embeddings = {}
        embeddings = []
        for document_id, doc in self.documents.items():
            tempdoc = copy.copy(doc)
            embeddings.append(tempdoc.pop("embeddings"))  # dict preserves insertion order, so this list is in the same order as `self.documents.values()`
            documents_without_embeddings[document_id] = tempdoc
        data = {"embedding_model_name": self.embedding_model_name,
                "documents": documents_without_embeddings}

        logger.info("HybridIR._save_datastore: Saving...")
        utils.create_directory(self.fulldocs_path)
        with open(self.fulldocs_documents_file, "w") as json_file:
            # Keeping the amount of indentation small improves human-readability, but also saves some disk space, as there are lots of indented lines in this file.
            json.dump(data, json_file, indent=2)

        # Note each document may have a different number of chunks, and each chunk produces one embedding vector. This yields one 2D array per document (outer index = chunk).
        logger.info(f"HybridIR._save_datastore: Saving embeddings (model '{self.embedding_model_name}')...")
        np.savez_compressed(self.fulldocs_embeddings_file, *embeddings)

        logger.info("HybridIR._save_datastore: exiting, all done.")

    def _load_datastore(self):
        try:
            with open(self.fulldocs_documents_file, "r") as json_file:
                data = json.load(json_file)
            stored_embedding_model_name = data["embedding_model_name"]
            documents = data["documents"]

            arrs = np.load(self.fulldocs_embeddings_file)
            for doc, document_embeddings in zip(documents.values(), arrs.values()):  # documents: {"document_id0": {...}, },  arrs: {"arr_0": np.array, ...}; arrs has one 2D array per document (outer index = chunk)
                doc["embeddings"] = document_embeddings.tolist()  # same in-memory format as if freshly created

            plural_s = "s" if len(documents) != 1 else ""
            logger.info(f"HybridIR._load_datastore: Loaded datastore with embedding model '{stored_embedding_model_name}' from '{str(self.fulldocs_path)}' ({len(documents)} document{plural_s}).")
            return stored_embedding_model_name, documents
        except Exception as exc:  # likely datastore not created yet
            logger.warning(f"HybridIR._load_datastore: While loading datastore from '{str(self.fulldocs_path)}': {type(exc)}: {exc}")
            return None, None

    # TODO: support other media such as images (semantic embedding via `clip-ViT-L-14`, available in `sentence_transformers`; and keyword extraction by CLIP/Deepbooru)
    def _prepare_document_for_indexing(self, doc):
        document_id = doc["document_id"]
        text = doc["text"]

        # We split each document into chunks. The chunks themselves are useful as the actual search results (the snippets that matched the search).
        logger.info(f"HybridIR._prepare_document_for_indexing: chunkifying document '{document_id}' ({len(text)} characters).")
        document_chunks = chunkify(text, chunk_size=self.chunk_size, overlap=self.overlap, extra=0.4)  # -> [{"text": ..., "chunk_id": ..., "offset": ...}, ...]

        # Tokenizing each chunk enables keyword search. These are used by the keyword index (bm25s).
        # NOTE: This can be slow, since we use spaCy's neural model for lemmatization.
        logger.info(f"HybridIR._prepare_document_for_indexing: tokenizing document '{document_id}'.")
        tokenized_chunks = [tokenize(chunk["text"]) for chunk in document_chunks]

        # Embedding each chunk enables semantic search. These are used by the vector index (chromadb).
        # NOTE: This can be slow, depending on the embedding model used, and whether GPU acceleration is available.
        logger.info(f"HybridIR._prepare_document_for_indexing: computing semantic embeddings for document '{document_id}'.")
        document_embeddings = self._semantic_model.encode([chunk["text"] for chunk in document_chunks],
                                                          show_progress_bar=True,
                                                          convert_to_numpy=True,
                                                          normalize_embeddings=True)  # SLOW; embeddings for each chunk
        document_embeddings = document_embeddings.tolist()  # for JSON serialization

        prepdata = {"chunks": document_chunks,  # [{"text": ..., "chunk_id": ..., "offset": ...}, ...]
                    "tokens": tokenized_chunks,  # [[token0_of_chunk0, token1_of_chunk0, ...], [token0_of_chunk1, token1_of_chunk1, ...], ...]
                    "embeddings": document_embeddings,  # [vec_of_chunk0, vec_of_chunk1, ...]
                    }
        return prepdata

    # This is used both by the commit mechanism as well as the full index rebuild.
    def _add_document_to_vector_collection(self, doc: Dict) -> None:
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
        if self._keyword_retriever is not None:
            self.full_id_to_record_index = {record["full_id"]: idx for idx, record in enumerate(self._keyword_retriever.corpus)}
        else:
            self.full_id_to_record_index = {}

    # TODO: add a variant of `query` that doesn't return debug information (but final fused search results only)
    # TODO: add a variant of `query` with a fixed amount of context around each match (we can do this by looking up the fulltext of the matching chunk and taking the text from there)
    def query(self,
              query: str,
              k: int = 10,
              alpha: float = 2.0,
              keyword_score_threshold: float = 0.1,
              semantic_distance_threshold: float = 0.8) -> List[Dict]:
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
        """
        logger.info(f"HybridIR.query: entered. Searching for {k} best matches for '{query}'")

        if not self.documents:
            logger.info("HybridIR.query: No documents in index, returning empty result.")
            return []
        if self._keyword_retriever is None:
            assert False  # we should have `self._keyword_retriever` as soon as we have at least one document

        internal_k = int(alpha * k)
        internal_k = min(internal_k, len(self._keyword_retriever.corpus))  # `bm25s` library requires `k ≤ corpus size`

        # BM25 search
        logger.info("HybridIR.query: keyword search")
        query_tokens = tokenize(query)
        raw_keyword_results, raw_keyword_scores = self._keyword_retriever.retrieve([query_tokens],  # list of list of tokens (outer list = one element per query; can run multiple queries at once)
                                                                                   k=internal_k)

        # Filter keyword results by threshold
        keyword_results = []
        keyword_scores = []
        for j in range(raw_keyword_results.shape[1]):
            # https://github.com/xhluca/bm25s/blob/main/examples/save_and_reload_end_to_end.py
            keyword_result = raw_keyword_results[0, j]
            keyword_score = raw_keyword_scores[0, j]
            if keyword_score > keyword_score_threshold:
                keyword_results.append(keyword_result)
                keyword_scores.append(keyword_score)
        # Now `keyword_results` contains the corpus entries as-is

        # Vector search
        logger.info("HybridIR.query: semantic search")
        query_embedding = self._semantic_model.encode([query],
                                                      show_progress_bar=True,
                                                      convert_to_numpy=True,
                                                      normalize_embeddings=True)[0].tolist()
        chroma_results = self._vector_collection.query(query_embeddings=[query_embedding],
                                                       n_results=internal_k,
                                                       include=["metadatas", "distances"])
        # list of list of metadatas (outer list = one element per query?)
        # https://github.com/chroma-core/chroma/blob/main/chromadb/api/types.py
        raw_vector_results = chroma_results["metadatas"][0]  # -> list of metadatas
        raw_vector_distances = chroma_results["distances"][0]  # -> list of float

        # Filter vector results by threshold
        vector_results = []
        vector_distances = []
        for vector_result, vector_distance in zip(raw_vector_results, raw_vector_distances):
            if vector_distance < semantic_distance_threshold:
                vector_results.append(vector_result)
                vector_distances.append(vector_distance)

        logger.info("HybridIR.query: fusing results")

        # Fuse results with RRF
        full_ids_bm25 = [record["full_id"] for record in keyword_results]  # anything hashable that uniquely identifies each result -> use the full ID
        full_ids_vector = [record["full_id"] for record in vector_results]
        rrf_results = reciprocal_rank_fusion(full_ids_bm25, full_ids_vector)

        # Collect the actual data records for each full ID, and populate the fused scores. Note we need the chunks, not the full documents.
        # We can collect them from the keyword-search corpus, regardless of which backend actually returned any specific result.
        fused_results = []
        for full_id, rrf_score in rrf_results:
            record = copy.copy(self._keyword_retriever.corpus[self.full_id_to_record_index[full_id]])
            record["score"] = rrf_score
            fused_results.append(record)

        # Merge adjacent chunks, sorting the final results by the fused score.
        # Each merged chunk gets the score of the highest-scoring individual chunk that went into it.
        #
        # NOTE: Merged results don't have a "chunk_id" or "full_id" (design choice; multiple chunks may have been merged into each result,
        #       so chunk-specific fields wouldn't make sense), but only "document_id", "offset", "text", and "score" (RRF score).
        logger.info("HybridIR.query: merging contiguous spans in results")
        merged = merge_contiguous_spans(fused_results)

        logger.info(f"HybridIR.query: retrieved chunk statistics: {len(keyword_results)} keyword matches, {len(vector_results)} semantic matches; total {len(fused_results)} unique matches; {len(merged)} results after merging contiguous spans from each document.")

        # Drop extra results, if there are still too many at this point.
        logger.info(f"HybridIR.query: Returning up to {k} best results (out of {len(merged)} retrieved), sorted by RRF score.")
        merged = merged[:k]

        logger.info("HybridIR.query: exiting. All done.")

        # Format of `merged`: [{"document_id": the_id_string, "text": merged_contiguous_text, "offset": start_offset_in_document, "score": rrf_score}, ...]
        return merged, (keyword_results, keyword_scores), (vector_results, vector_distances)

# --------------------------------------------------------------------------------

# Usage example / demo
if __name__ == "__main__":
    import textwrap
    from mcpyrate import colorizer

    # Create the retriever.
    config_dir = pathlib.Path(config.hybridir_demo_save_dir).expanduser().resolve()
    retriever = HybridIR(datastore_base_dir=config_dir,
                         embedding_model_name=config.qa_embedding_model)

    # Documents, plain text. Replace this with your data loading.
    #
    # This example is real-world data from a few AI papers from arXiv, copy'n'pasted from the PDFs.
    # We could get cleaner abstracts from the arXiv metadata, but fulltexts (after `pdftotext`) tend to look like this.
    docs_text = [textwrap.dedent("""
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

                 textwrap.dedent("""
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

                 textwrap.dedent("""
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
                 combines multiple minds’ strengths and knowl-
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

                 textwrap.dedent("""
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
                 """).strip()]

    # Add our documents to the index
    #
    # NOTE: The datastore is persistent, so you only need to do this when you add new documents.
    #
    # If you need to delete the index, open `config_dir` in a file manager, and delete the appropriate subdirectories:
    #   - "fulldocs" is the main datastore.
    #     - This is the master copy of the text data stored in the IR system, preprocessed into a format that can be indexed quickly
    #       (i.e. already chunkified, tokenized, and embedded).
    #     - This subdirectory alone is sufficient to rebuild the search indices, preserving all documents.
    # The other two subdirectories store the actual search indices:
    #   - "bm25s" is the keyword search index.
    #     - It is currently rebuilt at every `commit` due to technical limitations of the `bm25s` backend.
    #     - If you need to force a rebuild of the keyword index, shut down the app, delete this subdirectory, and then start the app again.
    #       `HybridIR` will then detect that the keyword index is missing, and rebuild it automatically (from the main datastore).
    #   - "chromadb" is the vector store for the semantic search.
    #     - It is currently never rebuilt automatically, but only updated at every `commit`.
    #     - If you need to force a rebuild of the semantic index, shut down the app, delete this subdirectory, and then start the app again.
    #       `HybridIR` will then detect that the semantic index is missing, and rebuild it automatically (from the main datastore).
    #
    # Queue each document for indexing.
    for m, doc_text in enumerate(docs_text, start=1):
        retriever.add(document_id=f"arxiv_abstract_{m}",
                      path="<locals>",  # in case of text coming from actual files, you can put the path here (to easily find the original file whose text data matched a search).
                      text=doc_text)
    # Write all pending changes, performing the actual indexing.
    retriever.commit()

    # Now we have a datastore. Run some searches.
    kw_threshold = 0.1
    vec_threshold = 0.8
    for query_string in ("ai agents",  # the actual test set topic
                         "llms",  # related topic
                         "language models",  # related topic, different wording
                         "quantum physics",  # completely unrelated technical topic
                         "can cats jump",  # completely unrelated non-technical topic
                         "blurba zaaaarrrgh blah qwertyuiop"):  # utter nonsense
        search_results, (keyword_results, keyword_scores), (vector_results, vector_distances) = retriever.query(query_string,
                                                                                                                k=5,
                                                                                                                keyword_score_threshold=kw_threshold,
                                                                                                                semantic_distance_threshold=vec_threshold)
        styled_query_string = colorizer.colorize(query_string, colorizer.Style.BRIGHT)  # for printing

        # DEBUG - you can obtain the raw results for keyword and semantic searches separately.
        # This data is useful e.g. for tuning the threshold hyperparameters.
        print()
        print(f"Keyword results for '{styled_query_string}' (BM25 score > {kw_threshold})")
        if keyword_results:
            for rank, (keyword_result, keyword_score) in enumerate(zip(keyword_results, keyword_scores), start=1):
                print(f"    {rank}. {keyword_result['full_id']} (score {keyword_score})")
        else:
            print(colorizer.colorize("    <no results>", colorizer.Style.DIM))
        print(f"Vector results for '{styled_query_string}' (semantic distance < {vec_threshold})")
        if vector_results:
            for rank, (vector_result, vector_distance) in enumerate(zip(vector_results, vector_distances), start=1):
                print(f"    {rank}. {vector_result['full_id']} (distance {vector_distance})")
        else:
            print(colorizer.colorize("    <no results>", colorizer.Style.DIM))

        print()
        print(f"Final search results for '{styled_query_string}':")
        print()

        # Show results to user. This is the final output that you'd normally show in the GUI (or paste into an LLM's context),
        # where contiguous spans from the same document have been merged.
        if search_results:
            for rank, result in enumerate(search_results, start=1):
                score = result["score"]  # final RRF score of search match
                document_id = result["document_id"]  # ID of document the search match came from
                result_text = result["text"]  # text of search match
                start_offset = result["offset"]  # start offset of `text` in document
                end_offset = start_offset + len(result_text)  # one past end
                document = retriever.documents[document_id]
                fulltext = document["text"]

                styled_rank = colorizer.colorize(f"{rank}.", colorizer.Style.BRIGHT)
                styled_docid = colorizer.colorize(document_id, colorizer.Style.BRIGHT)
                styled_extra_data = colorizer.colorize(f"(RRF score {score}, start offset in document {start_offset})", colorizer.Style.DIM)
                maybe_start_ellipsis = colorizer.colorize("...", colorizer.Style.DIM) if start_offset > 0 else ""
                maybe_end_ellipsis = colorizer.colorize("...", colorizer.Style.DIM) if end_offset < len(fulltext) else ""

                print(f"{styled_rank} {styled_docid} {styled_extra_data}\n\n{maybe_start_ellipsis}{result_text}{maybe_end_ellipsis}")
                print()
        else:
            print(colorizer.colorize("<no results>", colorizer.Style.DIM))
            print()
