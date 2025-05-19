"""Initialization for NLP (natural language processing) backends.

The main features are model caching (load only one copy of each) and device management (which device to load on).
"""

__all__ = {"load_nlp_pipeline", "load_stopwords",
           "load_embedding_model"}

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import List

# NLP
from sentence_transformers import SentenceTransformer
import spacy

from . import config

def load_stopwords() -> List[str]:
    """Load a default list of stopwords (English default list from spaCy)."""
    from spacy.lang.en import English
    nlp_en = English()
    stopwords = nlp_en.Defaults.stop_words
    return stopwords

_nlp_pipelines = {}
def load_nlp_pipeline(model_name: str):
    """Load and return the spaCy NLP pipeline.

    If the specified model is already loaded, return the already-loaded instance.
    """
    if model_name in _nlp_pipelines:
        logger.info(f"load_nlp_pipeline: '{model_name}' is already loaded, returning it.")
        return _nlp_pipelines[model_name]
    logger.info(f"load_nlp_pipeline: Loading '{model_name}'.")

    device_string = config.devices["nlp"]["device_string"]
    if device_string.startswith("cuda"):
        if ":" in device_string:
            _, gpu_id = device_string.split(":")
            gpu_id = int(gpu_id)
        else:
            gpu_id = 0

        try:
            spacy.require_gpu(gpu_id=gpu_id)
            logger.info("load_nlp_pipeline: spaCy will run on GPU (if available).")
        except Exception as exc:
            logger.warning(f"load_nlp_pipeline: exception while enabling GPU for spaCy: {type(exc)}: {exc}")
            spacy.require_cpu()
            logger.info("load_nlp_pipeline: spaCy will run on CPU.")
    else:
        spacy.require_cpu()
        logger.info("load_nlp_pipeline: spaCy will run on CPU.")

    try:
        nlp_pipeline = spacy.load(model_name)
    except OSError:
        # https://stackoverflow.com/questions/62728854/how-to-place-spacy-en-core-web-md-model-in-python-package
        logger.info(f"load_nlp_pipeline: Downloading spaCy model '{model_name}' (don't worry, this will only happen once)...")
        from spacy.cli import download
        download(model_name)
        nlp_pipeline = spacy.load(model_name)

    logger.info(f"load_nlp_pipeline: Loaded spaCy model '{model_name}'.")
    _nlp_pipelines[model_name] = nlp_pipeline
    return nlp_pipeline

# Cache the embedding models (to load only one copy of each model)
_embedding_models = {}
def load_embedding_model(model_name: str):
    """Load and return the embedding model (for vector storage).

    If the specified model is already loaded, return the already-loaded instance.
    """
    if model_name in _embedding_models:
        logger.info(f"load_embedding_model: '{model_name}' is already loaded, returning it.")
        return _embedding_models[model_name]
    logger.info(f"load_embedding_model: Loading '{model_name}'.")

    try:
        device_string = config.devices["embeddings"]["device_string"]
        embedding_model = SentenceTransformer(model_name, device=device_string)
    except RuntimeError as exc:
        logger.warning(f"load_embedding_model: exception while loading SentenceTransformer (will try again in CPU mode): {type(exc)}: {exc}")
        try:
            device_string = "cpu"
            embedding_model = SentenceTransformer(model_name, device=device_string)
        except RuntimeError as exc:
            logger.warning(f"load_embedding_model: failed to load SentenceTransformer: {type(exc)}: {exc}")
            raise
    logger.info(f"load_embedding_model: Loaded model '{model_name}' on device '{device_string}'.")
    _embedding_models[model_name] = embedding_model
    return embedding_model
