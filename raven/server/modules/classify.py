"""Text sentiment classification for Raven-server."""

__all__ = ["init_module", "is_available", "get_info", "classify_text"]

import logging
logger = logging.getLogger(__name__)

from typing import Union

from colorama import Fore, Style

import torch

from ...common import nlptools

classifier = None
_model_name = None

def init_module(model_name: str, device_string: str, dtype: Union[str, torch.dtype]) -> None:
    global classifier
    global _model_name
    print(f"Initializing {Fore.GREEN}{Style.BRIGHT}classification{Style.RESET_ALL} on device '{Fore.GREEN}{Style.BRIGHT}{device_string}{Style.RESET_ALL}' with model '{Fore.GREEN}{Style.BRIGHT}{model_name}{Style.RESET_ALL}'...")
    try:
        classifier = nlptools.load_classifier(model_name,
                                              device_string,
                                              dtype)
        _model_name = model_name
    except Exception:
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during init of module 'classify'.{Style.RESET_ALL} See server log for details.")
        logger.exception("init_module: failed")
        classifier = None

def is_available() -> bool:
    """Return whether this module is up and running."""
    return (classifier is not None)

def get_info() -> dict:
    """Return engine metadata as a JSON-serializable dict.

    Current fields:
        `model`: HuggingFace repo identifier for the loaded classification model.
    """
    if classifier is None:
        raise RuntimeError("get_info: module not initialized (did `init_module` succeed?)")
    return {"model": _model_name}

def classify_text(text: str) -> list:
    return nlptools.classify(classifier, text)
