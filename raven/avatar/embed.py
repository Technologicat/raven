# The "embeddings" module is only provided for compatibility with the discontinued SillyTavern-extras,
# to provide a fast (GPU-accelerated, or at least CPU-native) embeddings API endpoint for SillyTavern.
#
# Raven loads its embedding module in the main app, not in the `avatar` subapp.

__all__ = ["init_module", "embed_sentences"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import List, Union

from colorama import Fore, Style

import numpy as np

from .. import nlptools

sentence_embedder = None

def init_module(model_name: str) -> None:
    global sentence_embedder
    print(f"Initializing {Fore.GREEN}{Style.BRIGHT}embeddings{Style.RESET_ALL} with model '{Fore.GREEN}{Style.BRIGHT}{model_name}{Style.RESET_ALL}'...")
    sentence_embedder = nlptools.load_embedding_model(model_name)

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
