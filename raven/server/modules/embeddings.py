__all__ = ["init_module", "is_available", "embed_sentences"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import traceback
from typing import List, Union

from colorama import Fore, Style

import torch

from ...common import nlptools

import numpy as np

sentence_embedder = None

def init_module(model_name: str, device_string: str, torch_dtype: Union[str, torch.dtype]) -> None:
    global sentence_embedder
    print(f"Initializing {Fore.GREEN}{Style.BRIGHT}embeddings{Style.RESET_ALL} on device '{Fore.GREEN}{Style.BRIGHT}{device_string}{Style.RESET_ALL}' with model '{Fore.GREEN}{Style.BRIGHT}{model_name}{Style.RESET_ALL}'...")
    try:
        sentence_embedder = nlptools.load_embedding_model(model_name,
                                                          device_string,
                                                          torch_dtype)
    except Exception as exc:
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during init of module 'embeddings'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"init_module: failed: {type(exc)}: {exc}")
        sentence_embedder = None

def is_available() -> bool:
    """Return whether this module is up and running."""
    return (sentence_embedder is not None)

def embed_sentences(text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    vectors: Union[np.array, List[np.array]] = sentence_embedder.encode(text,
                                                                        show_progress_bar=True,  # on console running this app
                                                                        convert_to_numpy=True,
                                                                        normalize_embeddings=True)
    # NumPy arrays are not JSON serializable, so convert to Python lists
    if isinstance(vectors, np.ndarray):
        vectors = vectors.tolist()
    else:  # isinstance(vectors, list) and all(isinstance(x, np.ndarray) for x in vectors)
        vectors = [x.tolist() for x in vectors]

    return vectors
