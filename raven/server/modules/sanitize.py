"""Text cleanup module for Raven-server."""

__all__ = ["init_module", "is_available", "dehyphenate"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import traceback
from typing import List, Union

from colorama import Fore, Style

from ...common import nlptools

dehyphenator = None

def init_module(model_name: str,
                device_string: str,
                summarization_prefix: str = "") -> None:
    global dehyphenator
    print(f"Initializing {Fore.GREEN}{Style.BRIGHT}sanitize{Style.RESET_ALL} on device '{Fore.GREEN}{Style.BRIGHT}{device_string}{Style.RESET_ALL}' with model '{Fore.GREEN}{Style.BRIGHT}{model_name}{Style.RESET_ALL}'...")
    try:
        dehyphenator = nlptools.load_dehyphenator(model_name, device_string)
    except Exception as exc:
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during init of module 'sanitize'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"init_module: failed: {type(exc)}: {exc}")
        dehyphenator = None

def is_available() -> bool:
    """Return whether this module is up and running."""
    return (dehyphenator is not None)

def dehyphenate(text: Union[str, List[str]]) -> Union[str, List[str]]:
    """Dehyphenate `text`.

    This can be used to clean up broken text e.g. as extracted
    from a PDF file:

        Text that was bro-
        ken by hyphenation.

    -->

        Text that was broken by hyphenation.

    If you intend to send the text to an LLM, having it broken
    by hyphenation doesn't matter much in practice, but this
    makes the text much nicer for humans to look at.

    Be aware that this often causes paragraphs to run together,
    because the likely-paragraph-split analyzer is not perfect.
    We could analyze one paragraph at a time, but we currently don't,
    because the broken input text could contain blank lines at
    arbitrary positions, so these are not a reliable indicator
    of actual paragraph breaks. If you have known paragraphs you
    want to preserve, you can send them as a list to process each
    separately.

    The primary use case for this in Raven is English text; but the
    backend (with the "multi" model) does autodetect 300+ languages,
    so give it a try.

    This is based on the `dehyphen` package. The analysis applies a small,
    specialized AI model (not an LLM) to evaluate the perplexity of the
    different possible hyphenation options (in the example, "bro ken",
    "bro-ken", "broken") that could have produced the hyphenated text.
    The engine automatically picks the choice with the minimal perplexity
    (i.e. the most likely according to the model).

    The AI is a character-level contextual embeddings model from the
    Flair-NLP project.

    We then apply some heuristics to clean up the output.
    """
    return nlptools.dehyphenate(dehyphenator, text)
