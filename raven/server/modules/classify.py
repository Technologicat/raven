"""Text sentiment classification for Raven-server."""

__all__ = ["init_module", "is_available", "classify_text"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import traceback
from typing import Union

from colorama import Fore, Style

import torch

from ...common import nlptools

classifier = None

def init_module(model_name: str, device_string: str, torch_dtype: Union[str, torch.dtype]) -> None:
    global classifier
    print(f"Initializing {Fore.GREEN}{Style.BRIGHT}classification{Style.RESET_ALL} on device '{Fore.GREEN}{Style.BRIGHT}{device_string}{Style.RESET_ALL}' with model '{Fore.GREEN}{Style.BRIGHT}{model_name}{Style.RESET_ALL}'...")
    try:
        classifier = nlptools.load_classifier(model_name,
                                              device_string,
                                              torch_dtype)
    except Exception as exc:
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during init of module 'classify'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"init_module: failed: {type(exc)}: {exc}")
        classifier = None

def is_available() -> bool:
    """Return whether this module is up and running."""
    return (classifier is not None)

def classify_text(text: str) -> list:
    return nlptools.classify(classifier, text)
