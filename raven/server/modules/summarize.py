"""Summarization module for Raven-avatar.

Based on the `summarize` module of the old SillyTavern-extras server,
and extended from there.
"""

# TODO: LLM mode (call an LLM backend using `raven.librarian.llmclient`)

__all__ = ["init_module", "is_available", "summarize_text"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import re
import traceback
from typing import Union

from colorama import Fore, Style

import torch

from transformers import pipeline

from ...common import utils as common_utils
from ...common import nlptools
from ...common import numutils

text_summarization_pipe = None
text_summarization_prefix = None
nlp_pipe = None  # for breaking text into sentences (smart chunking)

def init_module(model_name: str,
                spacy_model_name: str,
                device_string: str,
                torch_dtype: Union[str, torch.dtype],
                summarization_prefix: str = "") -> None:
    global nlp_pipe
    global text_summarization_pipe
    global text_summarization_prefix
    print(f"Initializing {Fore.GREEN}{Style.BRIGHT}summarize{Style.RESET_ALL} on device '{Fore.GREEN}{Style.BRIGHT}{device_string}{Style.RESET_ALL}' with summarization model '{Fore.GREEN}{Style.BRIGHT}{model_name}{Style.RESET_ALL}' and spaCy model '{Fore.GREEN}{Style.BRIGHT}{spacy_model_name}{Style.RESET_ALL}' on device '{Fore.GREEN}{Style.BRIGHT}cpu{Style.RESET_ALL}'...")
    try:
        device = torch.device(device_string)
        text_summarization_pipe = pipeline(
            "summarization",
            model=model_name,
            device=device,
            torch_dtype=torch_dtype,
        )
        print(f"summarization model context window is {Fore.GREEN}{Style.BRIGHT}{text_summarization_pipe.tokenizer.model_max_length} tokens{Style.RESET_ALL}")
        # print(text_summarization_pipe.model.config)  # DEBUG
        # print(text_summarization_pipe.tokenizer)
        # print(dir(text_summarization_pipe.tokenizer))
        text_summarization_prefix = summarization_prefix
        nlp_pipe = nlptools.load_pipeline(spacy_model_name,
                                          "cpu")  # device_string    # seems faster to run sentence-splitting on the CPU, at least for short-ish (chat message) inputs.
    except Exception as exc:
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during init of module 'summarize'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"init_module: failed: {type(exc)}: {exc}")
        text_summarization_pipe = None
        text_summarization_prefix = None
        nlp_pipe = None

def is_available() -> bool:
    """Return whether this module is up and running."""
    return (text_summarization_pipe is not None)

def _summarize_one(text: str) -> str:
    """Summarize a piece of text that fits into the summarization model's context window.

    If the text does not fit, raises `IndexError`. See `_summarize_chunked` to handle arbitrary length texts.
    """
    tokens = text_summarization_pipe.tokenizer.tokenize(text)  # may be useful for debug...
    length_in_tokens = len(tokens)  # ...but this is what we actually need to set up the summarization lengths semsibly
    logger.info(f"Input text length is {len(text)} characters, {length_in_tokens} tokens.")

    if length_in_tokens > text_summarization_pipe.tokenizer.model_max_length:
        logger.info(f"summarize: text to be summarized does not fit into model's context window (text length {len(text)} characters, {length_in_tokens} tokens; model limit {text_summarization_pipe.tokenizer.model_max_length} tokens).")
        raise IndexError  # and let `_summarize_chunked` handle it
    if length_in_tokens <= 20:  # too short to summarize?
        return text

    # TODO: summary length: sensible limits that work for very short (one sentence) and very long (several pages) texts. This is optimized for a paragraph or a few at most.
    lower_limit = min(20, length_in_tokens)  # try to always use at least this many tokens in the summary
    upper_limit = min(120, length_in_tokens)  # and always try to stay under this limit
    max_length = numutils.clamp(length_in_tokens // 2, ell=lower_limit, u=upper_limit)
    min_length = numutils.clamp(length_in_tokens // 10, ell=lower_limit, u=upper_limit)
    logger.info(f"Setting summary min = {min_length} tokens, max = {max_length} tokens.")

    summary = text_summarization_pipe(
        text,
        truncation=False,
        min_length=min_length,
        max_length=max_length,
    )[0]["summary_text"]
    return summary

def _summarize_chunked(text: str) -> str:
    """Summarize a text that may require chunking before it fits into the summarization model's context window."""
    try:
        return _summarize_one(text)
    except IndexError:
        logger.info("summarize: input text (length {len(text)} characters) is long; cutting text in half at a sentence boundary and summarizing the halves separately.")

        with nlp_pipe.select_pipes(enable=['tok2vec', "parser", "senter"]):  # process faster by enabling only needed modules; https://stackoverflow.com/a/74907505
            doc = nlp_pipe(text)
        sents = list(doc.sents)
        mid = len(sents) // 2
        firsthalf = " ".join(str(sent).strip() for sent in sents[:mid])
        secondhalf = " ".join(str(sent).strip() for sent in sents[mid:])
        # print("=" * 80)
        # print("Splitting long text:")
        # print("-" * 80)
        # print(firsthalf)
        # print("-" * 80)
        # print(secondhalf)
        # print("-" * 80)
        return " ".join(
            [_summarize_chunked(firsthalf),
             _summarize_chunked(secondhalf)]
        )

        # # Sentence-splitting the output of the general sliding-window chunkifier doesn't seem to work that well here. It's easier to correctly split into sentences when we have all the text available at once.
        #
        # def full_sentence_trimmer(overlap, mode, text):
        #     @memoize  # from `unpythonic`
        #     def get_sentences(text):
        #         with nlp_pipe.select_pipes(enable=['tok2vec', "parser", "senter"]):  # process faster by enabling only needed modules; https://stackoverflow.com/a/74907505
        #             doc = nlp_pipe(text)
        #         return list(doc.sents)
        #
        #     offset = 0
        #     tmp = text.strip()  # ignore whitespace at start/end of chunk when detecting incomplete sentences
        #     if mode != "first":  # allowed to trim beginning?
        #         # Lowercase letter at the start of the chunk -> probably not the start of a sentence.
        #         if tmp[0].upper() != tmp[0]:
        #             sents = get_sentences(text)
        #             first_sentence_len = len(sents[0])
        #             offset = min(overlap, first_sentence_len)  # Prefer to keep incomplete sentence when there's not enough chunk overlap to trim it without losing text.
        #             text = text[offset:]
        #     if mode != "last":  # allowed to trim end?
        #         # No punctuation mark at the end of the chunk -> probably not a complete sentence.
        #         if tmp[-1] not in (".", "!", "?"):
        #             sents = get_sentences(text)
        #             last_sentence_len = len(sents[-1])
        #             text = text[:-last_sentence_len]
        #     return text, offset
        #
        # chunks = common_utils.chunkify_text(text,
        #                                     chunk_size=len(text) // 2,
        #                                     overlap=0,
        #                                     extra=0.2,
        #                                     trimmer=full_sentence_trimmer)
        # summary = " ".join(_summarize_chunked(chunk["text"] for chunk in chunks))
        # return summary

def summarize_text(text: str) -> str:
    """Return an abstractive summary of input text.

    This uses an AI summarization model (see `raven.server.config.summarization_model`),
    plus some heuristics to minimally clean up the result.
    """
    def normalize_sentence(sent: str) -> str:
        """Given a sentence, remove surrounding whitespace and capitalize the first word."""
        sent = str(sent).strip()  # `sent` might actually originally be a spaCy output
        sent = sent[0].upper() + sent[1:]
        return sent
    def sanitize(text: str) -> str:
        """Sanitize `text`.

        Specifically:
          - Normalize Unicode representation to NFKC
          - Normalize whitespace at sentence boundaries (as detected by the loaded spaCy NLP model)
          - Capitalize start of each sentence
          - Drop incomplete last sentence if any, but only if there's more than one sentence in total.
        """
        text = common_utils.normalize_unicode(text)
        text = text.strip()

        # Detect possible incomplete sentence at the end.
        #   - Summarizer AIs sometimes do that, especially if they run into the user-specified output token limit too soon.
        #   - The input text may have been cut off before it reaches us. When this happens, some summarizer AIs become confused.
        #     (E.g. Qiliang/bart-large-cnn-samsum-ChatGPT_v3, given the input:
        #          " The quick brown fox jumped over the lazy dog.  This is the second sentence! What?! This incomplete sentence"
        #      focuses only on the fact that the last sentence is incomplete, and reports that as the summary.)
        end = -1 if text[-1] not in (".", "!", "?") else None

        # Split into sentences via NLP. (This is the sane approach.)
        with nlp_pipe.select_pipes(enable=['tok2vec', "parser", "senter"]):  # Process faster by enabling only needed modules; https://stackoverflow.com/a/74907505
            doc = nlp_pipe(text)
        sents = list(doc.sents)
        if end is not None and len(sents) == 1:  # If only one sentence, keep it even if incomplete.
            end = None
        text = " ".join(normalize_sentence(sent) for sent in sents[:end])
        return text

    # Prompt the summarizer to write the raw summary (AI magic happens here)
    text = sanitize(text)
    text = f"{text_summarization_prefix}{text}"
    summary = _summarize_chunked(text)

    # Rudimentary check against AI hallucination: summarizing a very short text sometimes fails with the AI making up more text than there is in the original.
    if len(summary) > len(text):
        return text

    # Postprocess the summary
    summary = sanitize(summary)

    # At this point, depending on the AI model, we still sometimes have the spacing for the punctuation as "Blah blah blah . Bluh bluh..."

    # Normalize whitespace at full-stops (periods)
    parts = summary.split(".")
    has_period_at_end = summary.endswith(".")  # might have "!" or "?" instead
    parts = [x.strip() for x in parts]
    parts = [x for x in parts if len(x)]
    summary = ". ".join(parts) + ("." if has_period_at_end else "")
    summary = re.sub(r"(\d)\. (\d)", r"\1.\2", summary)  # Fix decimal numbers broken by the punctuation fix

    # Normalize whitespace at commas
    parts = summary.split(",")
    parts = [x.strip() for x in parts]
    parts = [x for x in parts if len(x)]
    summary = ", ".join(parts)
    summary = re.sub(r"(\d)\, (\d)", r"\1,\2", summary)  # Fix numbers with American thousands separators, broken by the punctuation fix

    # Convert some very basic markup (e.g. superscripts/subscripts) into their Unicode equivalents.
    summary = common_utils.unicodize_basic_markup(summary)

    return summary
