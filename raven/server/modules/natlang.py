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

from flask import Response

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

def analyze(text: Union[str, List[str]], pipes: Optional[List[str]]) -> Response:
    """Perform NLP analysis on `text`.

    `pipes`: If provided, enable only the listed pipes. Which ones exist depend on the loaded spaCy model.
             If not provided, use the model's default pipes.

    Returns an HTTP response that contains the analysis results as serialized binary data, with the
    Content-Type and Content-Length headers set. The type is "application/octet-stream" and the length
    is the data length in bytes.

    The model's language code (e.g. "en" for English) is sent in the custom header "x-langcode".
    The client needs the language code to be able to deserialize the data correctly.
    """
    if isinstance(text, str):  # always wrap in a list container
        text = [text]

    if pipes is not None:
        # TODO: This approach of selecting the NLP pipes is likely not thread-safe (think multiple concurrent requests to the server). Revisit this later.
        with nlp_pipeline.select_pipes(enable=pipes):  # only the requested pipes (e.g. `pipes=["tok2vec", "parser", "senter"]` to split text to sentences with "en_core_web_sm")
            docs = list(nlp_pipeline.pipe(text))
    else:  # default pipes
        docs = list(nlp_pipeline.pipe(text))

    docs_bytes = nlptools.serialize_spacy_docs(docs)

    output_headers = {"Content-Type": "application/octet-stream",
                      "Content-Length": len(docs_bytes),
                      "x-langcode": nlp_pipeline.lang}
    return Response(docs_bytes, headers=output_headers)
