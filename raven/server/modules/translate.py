"""Machine translation (text, between natural languages) for Raven-server."""

__all__ = ["init_module", "is_available", "translate_text"]

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
translators = {}
nlp_pipe = None  # for breaking text into sentences (smart chunking)

def init_module(config_module_name: str,
                device_string: str,
                spacy_model_name: str,
                spacy_device_string: str,
                dtype: Union[str, torch.dtype]) -> None:
    global server_config
    global nlp_pipe
    print(f"Initializing {Fore.GREEN}{Style.BRIGHT}translate{Style.RESET_ALL} on device '{Fore.GREEN}{Style.BRIGHT}{device_string}{Style.RESET_ALL}'...")
    try:
        server_config = importlib.import_module(config_module_name)
        for target_lang, record in server_config.translation_models.items():
            for source_lang, model_name in record.items():
                logger.info(f"init_module: Ensuring model is installed for '{Fore.GREEN}{Style.BRIGHT}{source_lang}→{target_lang}{Style.RESET_ALL}': model '{Fore.GREEN}{Style.BRIGHT}{model_name}{Style.RESET_ALL}'...")
            hfutil.maybe_install_models(model_name)

        for target_lang, record in server_config.translation_models.items():
            for source_lang, model_name in record.items():
                # `nlptools` already handles caching, loading only one copy of each unique model with the same device/dtype, so we don't have to.
                logger.info(f"init_module: Loading model for '{Fore.GREEN}{Style.BRIGHT}{source_lang}→{target_lang}{Style.RESET_ALL}': model '{Fore.GREEN}{Style.BRIGHT}{model_name}{Style.RESET_ALL}'...")
                if target_lang not in translators:
                    translators[target_lang] = {}
                translators[target_lang][source_lang] = nlptools.load_translator(model_name,
                                                                                 device_string,
                                                                                 dtype)

        nlp_pipe = nlptools.load_spacy_pipeline(spacy_model_name,
                                                spacy_device_string)

    except Exception as exc:
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during init of module 'translate'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"init_module: failed: {type(exc)}: {exc}")
        server_config = None
        translators.clear()
        nlp_pipe = None

def is_available() -> bool:
    """Return whether this module is up and running."""
    return (server_config is not None)

def translate_text(text: Union[str, List[str]],
                   source_lang: str,
                   target_lang: str) -> Union[str, List[str]]:
    if target_lang not in translators:
        raise ValueError(f"translate_text: Unknown target language '{target_lang}'; valid: {list(translators.keys())}. See server configuration file.")
    translators_to_target_lang = translators[target_lang]

    if source_lang in translators_to_target_lang:  # specialized translator for this language pair
        translator = translators_to_target_lang[source_lang]
    elif None in translators_to_target_lang:  # fallback fallback translator to this target language (optional)
        translator = translators_to_target_lang[None]
    else:
        raise ValueError(f"translate_text: Don't know how to translate '{source_lang}' to '{target_lang}'; valid sources for '{target_lang}': {list(translators_to_target_lang.keys())}. See server configuration file.")

    return nlptools.translate(translator, nlp_pipe, text)
