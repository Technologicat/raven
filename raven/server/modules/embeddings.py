"""Semantic embedding / sentence embedding / text vectorization for Raven-server."""

__all__ = ["init_module", "is_available", "embed_sentences"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import importlib
import traceback
from typing import List, Union

from colorama import Fore, Style

import torch

from ...common import hfutil
from ...common import nlptools

server_config = None
embedders = {}

def init_module(config_module_name: str, device_string: str, dtype: Union[str, torch.dtype]) -> None:
    global server_config
    print(f"Initializing {Fore.GREEN}{Style.BRIGHT}embeddings{Style.RESET_ALL} on device '{Fore.GREEN}{Style.BRIGHT}{device_string}{Style.RESET_ALL}'...")
    try:
        server_config = importlib.import_module(config_module_name)
        for role, model_name in server_config.embedding_models.items():
            logger.info(f"init_module: Ensuring model is installed for role '{Fore.GREEN}{Style.BRIGHT}{role}{Style.RESET_ALL}': model '{Fore.GREEN}{Style.BRIGHT}{model_name}{Style.RESET_ALL}'...")
            hfutil.maybe_install_models(model_name)
        for role, model_name in server_config.embedding_models.items():
            # `nlptools` already handles caching, loading only one copy of each unique model with the same device/dtype, so we don't have to.
            logger.info(f"init_module: Loading model for role '{Fore.GREEN}{Style.BRIGHT}{role}{Style.RESET_ALL}': model '{Fore.GREEN}{Style.BRIGHT}{model_name}{Style.RESET_ALL}'...")
            embedders[role] = nlptools.load_embedder(model_name,
                                                     device_string,
                                                     dtype)
            embedders[model_name] = embedders[role]  # also provide each model under its HuggingFace model name; this is convenient for HybridIR when it loads a database from disk.
    except Exception as exc:
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during init of module 'embeddings'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"init_module: failed: {type(exc)}: {exc}")
        server_config = None
        embedders.clear()

def is_available() -> bool:
    """Return whether this module is up and running."""
    return (server_config is not None)

def embed_sentences(text: Union[str, List[str]], model: str = "default") -> Union[List[float], List[List[float]]]:
    if model not in embedders:
        raise ValueError(f"embed_sentences: Unknown model '{model}'; valid: {list(embedders.keys())}. See server configuration file.")
    return nlptools.embed_sentences(embedders[model], text)
