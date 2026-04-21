"""Server-side NLP analysis.

If you want these features locally, just directly use the spaCy-related functions in `raven.common.nlptools`;
they can run on a local GPU.

This module is intended for situations where it is preferred to use the server's GPU.
"""

__all__ = ["init_module", "is_available", "analyze"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import traceback
from typing import List, Optional, Union

from colorama import Fore, Style

from flask import jsonify, Response

from ...common import nlptools

nlp_pipeline = None

def init_module(model_name: str, device_string: str) -> None:
    global nlp_pipeline
    print(f"Initializing {Fore.GREEN}{Style.BRIGHT}natlang{Style.RESET_ALL} on device '{Fore.GREEN}{Style.BRIGHT}{device_string}{Style.RESET_ALL}' with model '{Fore.GREEN}{Style.BRIGHT}{model_name}{Style.RESET_ALL}'...")
    try:
        nlp_pipeline = nlptools.load_spacy_pipeline(model_name,
                                                    device_string)
    except Exception as exc:
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during init of module 'natlang'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"init_module: failed: {type(exc)}: {exc}")
        nlp_pipeline = None

def is_available() -> bool:
    """Return whether this module is up and running."""
    return (nlp_pipeline is not None)

def analyze(text: Union[str, List[str]], pipes: Optional[List[str]], with_vectors: bool = False) -> Response:
    """Perform NLP analysis on `text`.

    `pipes`: If provided, enable only the listed pipes. Which ones exist depend on the loaded spaCy model.
             If not provided, use the model's default pipes.

    `with_vectors`: If `True`, include `doc.tensor` per-doc in the output so `token.vector` is
                    available on the client-reconstructed docs. Default `False` to keep the wire small.

    Returns a Flask JSON response containing a list of per-doc items. Each item carries its own
    language code, making the format naturally multilingual-ready. See
    `raven.common.nlptools.serialize_spacy_docs` for the exact shape.
    """
    docs = nlptools.spacy_analyze(nlp_pipeline,
                                  text,
                                  pipes)
    return jsonify(nlptools.serialize_spacy_docs(docs, with_vectors=with_vectors))
