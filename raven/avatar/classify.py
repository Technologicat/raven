"""
Classify module for SillyTavern Extras

Authors:
    - Tony Ribeiro (https://github.com/Tony-sama)
    - Cohee (https://github.com/Cohee1207)

Provides classification features for text

References:
    - https://huggingface.co/tasks/text-classification
"""

__all__ = ["init_module", "is_available", "classify_text"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import operator

from colorama import Fore, Style

import torch

from transformers import pipeline

text_emotion_pipe = None

def init_module(model_name: str, device_string: str, torch_dtype: str) -> None:
    global text_emotion_pipe

    print(f"Initializing {Fore.GREEN}{Style.BRIGHT}classification{Style.RESET_ALL} on device '{Fore.GREEN}{Style.BRIGHT}{device_string}{Style.RESET_ALL}' with model '{Fore.GREEN}{Style.BRIGHT}{model_name}{Style.RESET_ALL}'...")
    try:
        device = torch.device(device_string)
        text_emotion_pipe = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            device=device,
            torch_dtype=torch_dtype,
        )
    except Exception as exc:
        logger.warning(f"init_module: failed: {type(exc)}: {exc}")
        text_emotion_pipe = None

def is_available():
    """Return whether this module is up and running."""
    return (text_emotion_pipe is not None)

def classify_text(text: str) -> list:
    output = text_emotion_pipe(
        text,
        truncation=True,
        max_length=text_emotion_pipe.model.config.max_position_embeddings,
    )[0]
    return list(sorted(output,
                       key=operator.itemgetter("score"),
                       reverse=True))
