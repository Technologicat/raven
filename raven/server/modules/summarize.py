"""Text summarization module for Raven-server."""

# TODO: LLM mode (call an LLM backend using `raven.librarian.llmclient`)

__all__ = ["init_module", "is_available", "summarize_text"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import traceback
from typing import Union

from colorama import Fore, Style

import torch

from ...common import nlptools

summarizer = None
nlp_pipe = None  # for breaking text into sentences (smart chunking)

def init_module(model_name: str,
                spacy_model_name: str,
                device_string: str,
                torch_dtype: Union[str, torch.dtype],
                summarization_prefix: str = "") -> None:
    global summarizer
    global nlp_pipe
    print(f"Initializing {Fore.GREEN}{Style.BRIGHT}summarize{Style.RESET_ALL} on device '{Fore.GREEN}{Style.BRIGHT}{device_string}{Style.RESET_ALL}' with summarization model '{Fore.GREEN}{Style.BRIGHT}{model_name}{Style.RESET_ALL}' and spaCy model '{Fore.GREEN}{Style.BRIGHT}{spacy_model_name}{Style.RESET_ALL}' on device '{Fore.GREEN}{Style.BRIGHT}cpu{Style.RESET_ALL}'...")
    try:
        summarizer = nlptools.load_summarizer(model_name,
                                              device_string,
                                              torch_dtype,
                                              summarization_prefix)
        nlp_pipe = nlptools.load_spacy_pipeline(spacy_model_name,
                                                "cpu")  # device_string    # seems faster to run sentence-splitting on the CPU, at least for short-ish (chat message) inputs.
    except Exception as exc:
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during init of module 'summarize'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"init_module: failed: {type(exc)}: {exc}")
        summarizer = None
        nlp_pipe = None

def is_available() -> bool:
    """Return whether this module is up and running."""
    return (summarizer is not None)

def summarize_text(text: str) -> str:
    """Return an abstractive summary of input text.

    This uses an AI summarization model (see `raven.server.config.summarization_model`),
    plus some heuristics to minimally clean up the result.
    """
    return nlptools.summarize(summarizer, nlp_pipe, text)
