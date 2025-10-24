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

__all__ = ["init", "HybridIR", "HybridIRFileSystemEventHandler", "setup"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import atexit
from collections import defaultdict
import concurrent.futures
import copy
import json
import operator
import os
import pathlib
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import uuid

import watchdog.events
import watchdog.observers

import numpy as np

from unpythonic import allsame, box, ETAEstimator, partition, uniqify
from unpythonic.env import env as envcls

# database
import bm25s  # keyword
import chromadb  # semantic (vector)

from ..client import api
from ..client import config as client_config
from ..client import mayberemote

from ..common import bgtask
from ..common import deviceinfo
from ..common import nlptools
from ..common import utils as common_utils

from . import config as librarian_config

# --------------------------------------------------------------------------------
# Module bootup

deviceinfo.validate(librarian_config.devices)  # modifies in-place if CPU fallback needed

api.initialize(raven_server_url=client_config.raven_server_url,
               raven_api_key_file=client_config.raven_api_key_file,
               tts_playback_audio_device=client_config.tts_playback_audio_device)  # let it create a default executor

# --------------------------------------------------------------------------------

def format_chunk_full_id(document_id: str, chunk_id: str) -> str:
    """Generate an identifier for a chunk of a document, based on the given IDs."""
    return f"doc-{document_id}-chunk-{chunk_id}"

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

        # Pending-edit mechanism so that we can add/update/delete a set of documents at once, and *then* rebuild the indices.
        self._pending_edits = []

    def _tokenize(self, text: str) -> List[str]:
        """Apply lowercasing, tokenization, stemming, stopword removal.

        Returns a list of tokens.

        We use a spaCy NLP pipeline to do the analysis.
        """
        docs = self.nlp.analyze(text.lower())
        assert len(docs) == 1
        doc = docs[0]
        tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in self._stopwords]
        return tokens

    def _stat(self, path: Union[pathlib.Path, str]) -> Dict:  # size, mtime
        p = pathlib.Path(path) if not isinstance(path, pathlib.Path) else path
        abspath = p.expanduser().resolve()
        if abspath.exists():
            stat = os.stat(abspath)
            return {"size": stat.st_size, "mtime": stat.st_mtime}
        return {"size": None, "mtime": None}  # could be an in-memory document

    def add(self, document_id: str, path: str, text: str) -> str:
        """Queue a document for adding into the index. To save changes, call `commit`.

        `document_id`: must be unique. Recommended to use `unpythonic.gensym(os.path.basename(path))` or something.
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

            # Pend the requested edit.
            if action == "add":
                self._pending_edits.append((action, document))
            elif action == "delete":
                self._pending_edits.append((action, document_id))
            else:  # action == "update":
                # Update = delete, then add.
                self._pending_edits.append(("delete", document_id))
                self._pending_edits.append(("add", document))

    # TODO: Index rebuilding is slow. Maybe `commit` should run as a bgtask, like the BibTeX importer.
    #       Note `HybridIRFileSystemEventHandler` already does that.
    # TODO: `commit` is not as atomic as I'd like. If anything goes wrong, the vector index loses sync with the actual data, necessitating a full rebuild. Check if ChromaDB has transaction management.
    def commit(self) -> None:
        """Commit pending changes (adds/deletes/updates), re-indexing the databases.

        An update is internally a delete, followed by an add for the updated version of the same document.
        """
        logger.info("HybridIR.commit: entered.")
        with self.datastore_lock:
            with self._pending_edits_lock:
                if not self._pending_edits:
                    logger.info("HybridIR.commit: No pending changes, exiting.")
                    return
                pending_edits = copy.copy(self._pending_edits)
                self._pending_edits.clear()
            # Now we can release the lock on the original pending edits list,
            # so that `add`/`update`/`delete` are available in case another thread
            # wants to queue new edits while we're committing the previous ones.

            # Update `self.documents` and the semantic search index.
            # There is no "update" operation - to do that, first "delete", then "add".
            logger.info("HybridIR.commit: Applying pending changes.")
            errors_occurred = 0
            eta_estimator = ETAEstimator(total=len(pending_edits), keep_last=50)
            for edit_num, (edit_kind, data) in enumerate(pending_edits, start=1):
                logger.info(f"HybridIR.commit: Applying change {edit_num} out of {len(pending_edits)}; {eta_estimator.formatted_eta}")
                try:
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
                            doc = self.documents[document_id]
                            old_chunk_ids = [format_chunk_full_id(document_id, chunk["chunk_id"]) for chunk in doc["chunks"]]
                            self.documents.pop(document_id)
                            self._vector_collection.delete(ids=old_chunk_ids)
                        except KeyError as exc:
                            logger.warning(f"HybridIR.commit: Ignoring error: While deleting document with ID '{document_id}': {type(exc)}: {exc}")

                    else:  # should not happen, but let's log it
                        msg = f"HybridIR.commit: Unknown pending change type '{edit_kind}'. Ignoring."
                        logger.warning(msg)
                except Exception as exc:
                    errors_occurred += 1
                    logger.error(f"While applying changes: {type(exc)}: {exc}")
                    logger.info("Attempting to continue with remaining edits, if any.")
                eta_estimator.tick()

            self._rebuild_keyword_search_index()
            self._save_datastore()

            if errors_occurred:
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
            except Exception as exc:  # likely datastore not created yet
                logger.warning(f"HybridIR._load_datastore: While loading datastore from '{str(self.fulldocs_path)}': {type(exc)}: {exc}")
                return None, None

    # TODO: support other media such as images (semantic embedding via `clip-ViT-L-14`, available in `sentence_transformers`; and keyword extraction by CLIP/Deepbooru)
    def _prepare_document_for_indexing(self, doc: Dict) -> Dict:
        document_id = doc["document_id"]
        text = doc["text"]

        # We split each document into chunks. The chunks themselves are useful
        # as the actual search results (the snippets that matched the search).
        logger.info(f"HybridIR._prepare_document_for_indexing: chunkifying document '{document_id}' ({len(text)} characters).")
        document_chunks = common_utils.chunkify_text(text, chunk_size=self.chunk_size, overlap=self.overlap, extra=0.4)  # -> [{"text": ..., "chunk_id": ..., "offset": ...}, ...]

        # Tokenizing each chunk enables keyword search. These are used by the keyword index (bm25s).
        # NOTE: This can be slow, since we use spaCy's neural model for lemmatization.
        logger.info(f"HybridIR._prepare_document_for_indexing: tokenizing document '{document_id}'.")
        tokenized_chunks = [self._tokenize(chunk["text"]) for chunk in document_chunks]

        # Embedding each chunk enables semantic search. These are used by the vector index (chromadb).
        # NOTE: This can be slow, depending on the embedding model, and whether GPU acceleration is available.
        logger.info(f"HybridIR._prepare_document_for_indexing: computing semantic embeddings for document '{document_id}'.")
        document_embeddings = self.embedder.encode([chunk["text"] for chunk in document_chunks])  # SLOW; embeddings for each chunk
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
              return_extra_info: bool = False) -> List[Dict]:
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

        `return_extra_info`:
            If `True`: Return
                       `final_results, (keyword_results, keyword_scores), (vector_results, vector_distances)`.
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

        # Prepare anything we can before locking the datastore
        internal_k = min(int(alpha * k),
                         len(self._keyword_retriever.corpus))  # `bm25s` library requires `k ≤ corpus size`
        if include_documents is None:
            keyword_k = internal_k
        else:  # return score for *every record in database*, for manual metadata-based filtering (document ID)
            keyword_k = len(self._keyword_retriever.corpus)

        # Prepare query for keyword search
        query_tokens = self._tokenize(query)

        # Prepare query for vector search
        query_embedding = self.embedder.encode([query])[0]

        with self.datastore_lock:
            if not self.documents:
                logger.info("HybridIR.query: No documents in index, returning empty result.")
                return []
            if self._keyword_retriever is None:
                assert False  # we should have `self._keyword_retriever` as soon as we have at least one document

            # BM25 search
            logger.info("HybridIR.query: keyword search")
            # Here we always search all documents; we filter afterward, if needed.
            raw_keyword_results, raw_keyword_scores = self._keyword_retriever.retrieve([query_tokens],  # list of list of tokens (outer list = one element per query; can run multiple queries at once)
                                                                                       k=keyword_k)

            # Vector search
            logger.info("HybridIR.query: semantic search")
            if include_documents is not None:  # search only documents with given IDs
                chroma_results = self._vector_collection.query(query_embeddings=[query_embedding],
                                                               n_results=internal_k,
                                                               include=["metadatas", "distances"],
                                                               where={"document_id": {"$in": include_documents}})
            else:  # search all documents
                chroma_results = self._vector_collection.query(query_embeddings=[query_embedding],
                                                               n_results=internal_k,
                                                               include=["metadatas", "distances"])
            # list of list of metadatas (outer list = one element per query?)
            # https://github.com/chroma-core/chroma/blob/main/chromadb/api/types.py
            raw_vector_results = chroma_results["metadatas"][0]  # -> list of metadatas
            raw_vector_distances = chroma_results["distances"][0]  # -> list of float
        # Now we no longer need datastore access to complete the search

        # Filter keyword results by threshold (and by `include_documents`, if specified)
        keyword_results = []
        keyword_scores = []
        include_documents_set = set(include_documents) if include_documents is not None else set()  # for O(1) checking
        for j in range(raw_keyword_results.shape[1]):
            # https://github.com/xhluca/bm25s/blob/main/examples/save_and_reload_end_to_end.py
            keyword_result = raw_keyword_results[0, j]
            keyword_score = raw_keyword_scores[0, j]
            if keyword_score > keyword_score_threshold:
                if include_documents is not None and keyword_result["document_id"] not in include_documents_set:
                    continue
                keyword_results.append(keyword_result)
                keyword_scores.append(keyword_score)
        # Now `keyword_results` contains the corpus entries as-is

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

        # Collect the actual data records for each full ID, and populate the fused scores.
        # Note we need the chunks, not the full documents. We can collect them from the
        # keyword-search corpus, regardless of which backend actually returned any specific result.
        fused_results = []
        for full_id, rrf_score in rrf_results:
            record = copy.copy(self._keyword_retriever.corpus[self.full_id_to_record_index[full_id]])
            record["score"] = rrf_score
            fused_results.append(record)

        # Merge adjacent chunks, sorting the final results by the fused score.
        # Each merged chunk gets the score of the highest-scoring individual chunk that went into it.
        #
        # NOTE: Merged results don't have a "chunk_id" or "full_id" (design choice; multiple chunks may
        #       have been merged into each result, so chunk-specific fields wouldn't make sense), but only
        #       "document_id", "offset", "text", and "score" (RRF score).
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
            return merged, (keyword_results, keyword_scores), (vector_results, vector_distances)
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
        def clear_background_tasks():
            for task_manager in task_managers.values():
                task_manager.clear(wait=False)  # signal background tasks to exit
        atexit.register(clear_background_tasks)
    except Exception:
        bg = None
        task_managers.clear()
        raise

# --------------------------------------------------------------------------------

# See e.g. https://www.kdnuggets.com/monitor-your-file-system-with-pythons-watchdog
class HybridIRFileSystemEventHandler(watchdog.events.FileSystemEventHandler):
    def __init__(self,
                 docs_dir: Union[str, pathlib.Path],
                 recursive: bool,
                 retriever: HybridIR,
                 exts: List[str] = [".txt", ".md", ".rst", ".org", ".bib"],
                 callback: Callable = None) -> None:
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

        `exts`: File extensions of files to monitor.

        `callback`: When new content arrives (a file is created or updated in the target directory),
                    this function is called.

                    Its only argument is the path to the file, and it must return the plaintext
                    content of the file. This plaintext content is then sent into `retriever`'s
                    search index.

                    If no callback is specified, the file is read as UTF-8 encoded text.
                    (This is enough for simple plaintext files.)

        Uses the `watchdog` library.
        """
        self.docs_dir = pathlib.Path(docs_dir) if not isinstance(docs_dir, pathlib.Path) else docs_dir
        self.recursive = recursive
        self.retriever = retriever
        self.exts = exts
        self.callback = callback

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
            logger.debug(f"HybridIRFileSystemEventHandler.commit: {task_env.task_name}: Committing changes to HybridIR (may take a while; this step cannot be cancelled).")
            self.retriever.commit()
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
        p = pathlib.Path(path) if not isinstance(path, pathlib.Path) else path
        abspath = p.expanduser().resolve()
        abspath = str(abspath)
        if not any(abspath.endswith(ext) for ext in self.exts):
            logger.info(f"HybridIRFileSystemEventHandler._sanity_check: file '{abspath}': file extension not in monitored list {self.exts}, ignoring file.")
            return False
        return True

    def _read(self, path: Union[pathlib.Path, str]) -> str:
        p = pathlib.Path(path) if not isinstance(path, pathlib.Path) else path
        abspath = p.expanduser().resolve()
        if self.callback:
            content = self.callback(abspath)
        else:
            with open(path, "r", encoding="utf-8") as document_file:
                content = document_file.read()
        if not content:
            return None
        if not isinstance(content, str):
            logger.error(f"HybridIRFileSystemEventHandler._read: file '{str(abspath)}': got non-string content. Ignoring file.")
            return None
        return content.strip()

    def _make_task(self, kind: str, path: Union[pathlib.Path, str]) -> Callable:
        p = pathlib.Path(path) if not isinstance(path, pathlib.Path) else path
        abspath = p.expanduser().resolve()
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
        p = pathlib.Path(path) if not isinstance(path, pathlib.Path) else path
        abspath = p.expanduser().resolve()
        found_paths = []
        for root, dirs, files in os.walk(abspath):
            if not recursive:
                dirs.clear()
            for filename in files:
                filepath = os.path.join(root, filename)
                if self._sanity_check(filepath):
                    found_paths.append(str(pathlib.Path(filepath).expanduser().resolve()))
        plural_s = "s" if len(found_paths) != 1 else ""
        logger.info(f"HybridIRFileSystemEventHandler.rescan: Found {len(found_paths)} file{plural_s}.")

        with self.retriever.datastore_lock:
            def came_from_file(doc: Dict) -> bool:  # convention: in-memory sources use paths of the form "<document_name_here>"
                return not (doc["path"].startswith("<") and doc["path"].endswith(">"))
            indexed_paths = [doc["path"] for doc in self.retriever.documents.values() if came_from_file(doc)]
            indexed_paths_set = set(indexed_paths)
            def is_in_index(path: str) -> bool:
                return path in indexed_paths_set
            def is_file_updated(path: str) -> bool:
                stats = self.retriever._stat(path)
                document_id = self._make_document_id_from_path(path)
                assert document_id in self.retriever.documents  # this is only ever called for already indexed documents
                doc = self.retriever.documents[document_id]
                mtime_increased = (stats["mtime"] > doc["mtime"])
                filesize_changed = (stats["size"] != doc["filesize"])
                return mtime_increased or filesize_changed

            new_found_paths, already_indexed_found_paths = partition(is_in_index, found_paths)
            new_found_paths = list(new_found_paths)
            already_indexed_found_paths = list(already_indexed_found_paths)
            updated_paths = [path for path in already_indexed_found_paths if is_file_updated(path)]
            found_paths_set = set(found_paths)
            deleted_paths = [path for path in indexed_paths if path not in found_paths_set]

        new_plural_s = "s" if len(new_found_paths) != 1 else ""
        updated_plural_s = "s" if len(updated_paths) != 1 else ""
        deleted_plural_s = "s" if len(deleted_paths) != 1 else ""
        logger.info(f"HybridIRFileSystemEventHandler.rescan: Scan complete. Found {len(new_found_paths)} new file{new_plural_s}, {len(updated_paths)} updated file{updated_plural_s}, and {len(deleted_paths)} deleted file{deleted_plural_s}.")

        for path in new_found_paths:
            logger.info(f"HybridIRFileSystemEventHandler.rescan: File '{path}' is new: scheduling ingest.")
            task_managers["ingest"].submit(self._make_task(kind="add", path=path), envcls())
        for path in updated_paths:
            logger.info(f"HybridIRFileSystemEventHandler.rescan: File '{path}' was updated: scheduling ingest.")
            task_managers["ingest"].submit(self._make_task(kind="update", path=path), envcls())
        for path in deleted_paths:
            logger.info(f"HybridIRFileSystemEventHandler.rescan: File '{path}' was deleted: scheduling deletion from index.")
            task_managers["ingest"].submit(self._make_task(kind="delete", path=path), envcls())

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
          exts=[".txt", ".md", ".rst", ".org"],
          callback: Optional[Callable] = None,
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

    `exts`: Passed on to `HybridIRFileSystemEventHandler`, which see.
    `callback`: Passed on to `HybridIRFileSystemEventHandler`, which see.

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

    scanner = HybridIRFileSystemEventHandler(docs_dir=docs_dir,
                                             recursive=recursive,
                                             retriever=retriever,
                                             exts=exts,
                                             callback=callback)

    return retriever, scanner
