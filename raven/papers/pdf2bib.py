#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert PDF conference abstracts into a BibTeX database.

The heavy lifting is done by `raven.common.docextract` (PDF text extraction) and an OpenAI compatible LLM.

The command line is `raven-pdf2bib`, in `raven.papers.pdf2bib_cli`, which calls `run`.

A run is a `raven.librarian.sweep`: each PDF's fields go to a ledger as they arrive, so running the same
command again resumes, and the `.bib` outputs are rewritten from the ledger at the end of every run. With
`-o done`, each PDF is also moved into `done` once answered, which sorts what is processed from what is not.
"""

__all__ = ["full_output_trace", "extract", "setup_prompts",

           "listpdf", "extract_fields", "format_entry", "write_outputs",
           "collect_items", "sort_pdf", "process_abstracts",

           "run"]

import logging
logger = logging.getLogger(__name__)

import argparse
import collections
import io
import os
import pathlib
import re
import shutil
import sys
from textwrap import dedent
from collections.abc import Callable

from unpythonic import sym, timer, uniqify
from unpythonic.env import env

from ..client import api
from ..client import config as client_config
from ..client import mayberemote

from ..common import docextract
from ..common import netutil
from ..common import utils as common_utils

from ..librarian import agent
from ..librarian import chatutil
from ..librarian import llmclient
from ..librarian import sweep

from .utils import bibtex_escape

# --------------------------------------------------------------------------------
# Bootup

# How a document came out. Flagged means there is an answer and a person should look at it: the document
# seems to lack a field, or an answer looks wrong. A document with no usable answer at all is neither: it
# is a failed item of the sweep, and is asked again on the next run.
status_success = sym("success")
status_flagged = sym("flagged")

class _NoAnswer(Exception):
    """A step of one document gave no usable reply after its attempts. Not a property of the document, so
    the document is asked again on the next run rather than written out flagged."""

# What the prompts ask the LLM to reply when a field is absent from the abstract. Without one, the model
# describes the absence in prose, and the sentence lands in the BibTeX field as if it were the data.
_NO_AUTHORS = "NO AUTHORS LISTED"
_NO_TITLE = "NO TITLE"

# This tool's whole job is to drive an LLM over the text layer of free-form conference abstracts, so it
# always needs raven-server's client — unconditionally, unlike `visualizer.importer`, which needs it only
# when the cluster-keyword mode is `llm`. Initialize before the `MaybeRemoteService` below, whose
# constructor probes for the server and raises if the API is not up yet.
#
# It is declared here rather than inherited: this used to work only because importing `llmclient` happened
# to initialize the API as a module-import side effect, so the dependency was real but invisible, and
# nothing would have reported it if that import were ever reordered away.
#
# `main` calls this before anything else, and it must, because the `MaybeRemoteService` constructed below
# probes for the server. Doing it there rather than at import is what lets `--server-url` reach it at all:
# at module scope there is no parsed command line yet.
def _initialize_client_api(raven_server_url: str) -> None:
    """Bring up `raven.client.api` against `raven_server_url`. Call once, before any `mayberemote` use."""
    api.initialize(raven_server_url=raven_server_url,
                   maybe_network_retry=client_config.network_retry_batch,
                   raven_api_key_file=client_config.raven_api_key_file)  # let it create a default executor

_dehyphenator = None  # built on first use; see `_get_dehyphenator`

def _get_dehyphenator() -> "mayberemote.Dehyphenator":
    """The dehyphenator, built the first time something asks for it.

    Not built at import, because its constructor probes the Raven server: that needs the client API to be
    up, which needs the command line to have been parsed, since `--server-url` is what decides which server
    it probes.
    """
    global _dehyphenator
    if _dehyphenator is None:
        _dehyphenator = mayberemote.Dehyphenator(allow_local=True,
                                                 model_name=client_config.dehyphenation_model,
                                                 device_string=client_config.devices["sanitize"]["device_string"])
    return _dehyphenator

# --------------------------------------------------------------------------------
# Utilities

def _is_sentinel(reply: str, sentinel: str) -> bool:
    """Whether the LLM's `reply` is `sentinel`, ignoring case, surrounding whitespace, and the Markdown
    emphasis and trailing punctuation a model tends to add even when asked for plain text."""
    return reply.strip().strip("*_`\"'.! ").upper() == sentinel

def full_output_trace(record: agent.TurnRecord) -> str:
    """Format everything the LLM wrote in one extraction step, for the error report.

    When a step comes back empty, the reason is usually in the thinking trace - typically the model
    overthought until it ran out of token budget, leaving nothing to say. So the trace is what the report
    needs, and the final answer comes after it for context.

    The layout is the one Librarian's export buttons produce, so a trace in an error report and a trace in
    an exported chat read the same way.
    """
    return "\n\n".join(chatutil.format_message_text_for_export(message) for message in record.messages)

def extract(llm_settings: env,
            context: env,
            instruction: str,
            progress_symbol: str,
            label: str,
            attempts: int = 1,
            accept: Callable[[str], bool] = lambda reply: bool(reply.strip())) -> agent.TurnRecord:
    """Run one extraction step on the LLM, and return the resulting `agent.TurnRecord`.

    `context`: the document's: `unique_id`, `maybe_abort` (the sweep's handle for the call in flight), and
               `turns`, to which every attempt's record is appended for the sweep's traces.

    `instruction`: The complete prompt: what to extract, followed by the text to extract it from.

    `progress_symbol`: Printed to stderr as the LLM writes, so that a run of several thousand PDFs shows
                       which step is working. See `llmclient.make_console_progress_handler`.

    `label`: names the step in log messages and in the failure reason, e.g. "EXTRACT AUTHORS".

    `attempts`, `accept`: as `agent.ask_until`. The default accepts any reply that is not blank.

    Raises `_NoAnswer` when no attempt is accepted, and `RuntimeError` when the backend did not generate.

    The step runs without a character and without tools: every prompt here asks for a plain-text answer
    that this module then parses, whereas the character card asks for Markdown, for a reported train of
    thought, and for conversational prose. Each prompt states its own role and procedure, so what the card
    would contribute is exactly the part that would have to be undone afterwards.
    """
    # `ask_until` rather than `agent.turn`, which turns a backend failure into an assistant message: its text
    # is not blank, so it would pass `accept` and reach the BibTeX as a field.
    records, accepted = agent.ask_until(llm_settings,
                                        instruction,
                                        accept=accept,
                                        attempts=attempts,
                                        label=f"Input file '{context.unique_id}': {label}",
                                        maybe_abort=context.maybe_abort,
                                        on_progress=llmclient.make_console_progress_handler(progress_symbol))
    context.turns.extend(records)
    if not accepted:
        plural_s = "s" if attempts != 1 else ""
        raise _NoAnswer(f"{label}: no usable reply in {attempts} attempt{plural_s}")
    return records[-1]

def setup_prompts(llm_settings: env,
                  n_retries: int,
                  conference_year: str,
                  conference_booktitle: str | None = None,
                  conference_note: str | None = None,
                  conference_url: str | None = None) -> tuple[dict, str]:
    """Set up the LLM task handlers.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.

    `n_retries`: How many times to try (including the initial attempt) when author list extraction fails.
                 This is an LLM overthinking guard; the model may sometimes think so much that it fails to
                 produce a reply in the maximum token limit.

    Returns `(prompts, material)`. `material` is every prompt's text, for `sweep.fingerprint`. `prompts` is a
    dict, in the format `{bibtex_fieldname: (kind: str,
                                             thing: str | Callable,
                                             progress_symbol: str)}`,

    where the RHS are instructions for the main loop as to what to write into the field `bibtex_fieldname`.
    Although the content is hardcoded in the implementation of this function, we document the data structure here:

    `kind`: one of "literal", "prompt", "function":

       "literal": Inject the exact given text `thing: str` into the field.

       "prompt": Treat `thing: str` as a prompt for the LLM. Send in the prompt, a Markdown separator, and the complete text content of the PDF.
                 Use `progress_symbol` to indicate progress on the console. Inject the LLM's output (after heuristic cleaning) into the field.

       "function": Call `thing: Callable` with arguments `(context, unique_id, text)`, where `context` is the
                   document's (see `extract`) and `text` is the complete text content of the PDF.
                   Return value must be `(status, error_info, output)`.

                   `status` must be `status_success`, or `status_flagged` when a person should look at the
                   result. A step with no usable reply raises `_NoAnswer` instead, through `extract`.

                   Inject `output` into the field.
                   Except, if `output` is `None`, omit that field from the BibTeX.

    Obviously, the "function" kind is the most flexible. We use it to strip the reference list before handing over the text to the LLM.
    This tends to improve author list detection (by removing likely false positives before the LLM sees the text).
    """

    # Section headings that can be programmatically detected easily. These allow us to make some LLM processing simpler by removing confounders (e.g. reference lists have lots of author names and titles).
    kws_pattern = re.compile("^(Keywords|KEYWORDS|Key words|Key Words|KEY WORDS):", re.MULTILINE)
    ref_pattern = re.compile("^(References|REFERENCES)", re.MULTILINE)
    ack_pattern = re.compile("^(Acknowledgements|ACKNOWLEDGEMENTS|Acknowledgments|ACKNOWLEDGMENTS)", re.MULTILINE)

    # def _strip_pre(pattern: re.Pattern, text: str) -> str:  # TODO: not perfect, yaks destroy (doesn't work if multiple lines of keywords)
    #     match = pattern.search(text)
    #     if match is not None:
    #         header_end_pos = match.end()
    #     else:
    #         header_end_pos = None
    #     next_newline_pos = header_end_pos + text[header_end_pos:].find("\n")
    #     start_pos = next_newline_pos + 1
    #     return text[start_pos:].strip()

    def _strip_post(pattern: re.Pattern, text: str) -> str:
        """Strip the first match of `pattern` in `text`, and everything after it. Return the stripped text."""
        match = pattern.search(text)
        if match is not None:
            end_pos = match.start()
        else:
            end_pos = None
        return text[:end_pos].strip()
    def strip_postamble(text: str) -> str:
        """Strip reference list and acknowledgments from the conference abstract `text`."""
        text = _strip_post(ref_pattern, text)
        text = _strip_post(ack_pattern, text)
        return text

    # TODO: Sometimes this returns comma-separated authors. Can't easily detect programmatically, since "Ally Author" and "Author, Ally" are both valid BibTeX formats. Use LLM self-critique?
    prompt_get_authors = dedent(f"""Who are the authors of the abstract?

    This will be used for a bibliography. We need only the author names: NO institutions, NO street addresses, NO email addresses.

    Please reply only with the line(s) containing the author names, COPIED AS-IS from the original abstract.

    Just copy that line (or those lines) into the output.

    IMPORTANT: List the authors IN THE SAME ORDER as they appear in the original abstract.

    If the abstract lists no authors at all, reply only with: {_NO_AUTHORS}
    """)

    prompt_drop_author_affiliations = dedent("""Below is a list of author names for a bibliography:

    {author_names}

    **Notes on input format**

    - The list is automatically extracted raw data.
    - Author names may have affiliation markings such as '*', '1', '2', or similar. Sometimes, the same author may have several affiliations.
    - The list is comma-separated. It may use the word "and" to separate the last author.
    - Periods are used ONLY for abbreviation, NOT as a separator.
    - An author may have several first or middle names. They might or might not be abbreviated.
    - An author name may be followed by a parenthetical remark. Treat it as part of the author name.
    - Sometimes there is just one author.

    **Task**

    Please remove all affiliation markings, keeping everything else the same.

    IMPORTANT: Reply ONLY with the cleaned-up list.
    """)

    prompt_reformat_author_separators = dedent("""Below is a list of author names for a bibliography:

    {author_names}

    **Notes on input format**

    - In the original list, author names are separated by a comma, or by the word "and".
    - Periods are used ONLY for abbreviation, NOT as a separator.
    - An author may have several first or middle names. They might or might not be abbreviated.
    - An author name may be followed by a parenthetical remark. Treat it as part of the author name.
    - A last name may look like a first name. This is normal.
    - A name may look like an English word. This is normal.
    - Sometimes there is just one author.

    **Task**

    Please reformat the list, separating ALL authors using the word "and". Do NOT use a comma or a newline to separate authors.

    IMPORTANT: Reply ONLY with the reformatted list.
    """)

    def extract_authors(context: env, unique_id: str, text: str) -> tuple[sym, str, str | None]:
        status = status_success
        error_info = io.StringIO()
        text = strip_postamble(text)

        # TODO: Handle the case of a missing author list (there's at least one such abstract in the dataset). First check for author list presence using the LLM?
        # prompt_check_authorlist = dedent("""Consider the attached text snippet that begings after the separator mark.
        #
        # In that text, is there any text between the article title and the beginning of the main text?
        #
        # **Instructions**
        #
        # - When analyzing, IGNORE blank lines, if any.
        # - "Main text" is the actual main text body, EXCLUDING metadata such as authors, affiliations, and/or keywords.
        #
        #
        # **Example 1**
        #
        # Conference Name and Location
        #
        # Article Title
        #
        # Author A, Other O
        #
        # University of exampleness
        #
        # Keywords: blah, blah, ...
        #
        # In this paper we consider...
        #
        # --> YES, there is a list of authors and a list of keywords.
        #
        #
        # **Example 2**
        #
        # Conference Name and Location
        #
        # Article Title
        #
        # Author A, Other O
        #
        # University of exampleness
        #
        # In this paper we consider...
        #
        # --> YES, there is a list of authors.
        #
        #
        # **Example 3**
        #
        # Conference Name and Location
        #
        # Article Title
        #
        # Keywords: blah, blah, ...
        #
        # In this paper we consider...
        #
        # --> YES, there is a list of keywords.
        #
        #
        # **Example 4**
        #
        # Conference Name and Location
        #
        # Article Title
        #
        # In this paper we consider...
        #
        # --> NO, the main text begins immediately after the title.
        # """)
        #           In the ORIGINAL INPUT, is there any text that appears AFTER the title (which you extracted), BEFORE the start of the main text (whose first sentence you extracted)?
        #           The final "YES" or "NO" will be sent to a computer program, which cannot understand natural language, so it needs to be in a standard format. For that final word, use plain text, without formatting.
        #           IMPORTANT: Please print a final answer at the end of your reply; the analysis program needs it.
        prompt_check_authorlist = dedent(  # noqa: F841 -- WIP, used in commented-out call below
            """Attached is an ORIGINAL INPUT text.

        Please analyze it, following the detailed instructions below. As for how the ORIGINAL INPUT is formatted, consult the FORMAT EXAMPLE below.


        **FORMAT EXAMPLE**

        Conference Name and Location

        Example Article Title

        Author A*, Other O+ and Some S+

        * University of Exampleness
        + Technical University of Otherness

        Keywords: the, keywords, are, listed, here

        In this main text, we discuss new discoveries in science.
        This main text may go on for a few paragraphs.


        **Instructions**

        General:

        - When analyzing, IGNORE blank lines, if any.
        - MAIN TEXT is the actual main text body, excluding METADATA. The MAIN TEXT is written in free-form natural language.
        - METADATA consists of authors, affiliations, and/or keywords. METADATA is written in a LIST format.
        - The input text format is roughly as in FORMAT EXAMPLE. Note that the actual ORIGINAL INPUT is raw data, so one or more fields shown in the FORMAT EXAMPLE may be missing.
        - Note that if all METADATA is missing - this is a data error but it can happen - then the first sentence immediately after the title starts the MAIN TEXT.

        Step 1:

        - Start your reply with a section header for EXTRACTED PARTS.
        - Then find and report the article title, exactly as it appears in the ORIGINAL INPUT. Use the format "TITLE: ..."

          The article title usually fits onto one or two lines. The conference name is NOT the article title.

        - Then consider that title, and look at the lines in the ORIGINAL INPUT that follow the title. Where does the MAIN TEXT begin? Please report its first sentence. Use the format "FIRST SENTENCE: ..."
        - Move on to step 2. Do NOT extract any extra fields!

        Step 2:

        - Start a new section for ANALYSIS.
        - Please answer the following questions:

          In the ORIGINAL INPUT, is there any METADATA that appears AFTER the title (TITLE extracted in step 1), BEFORE the first sentence of the main text (FIRST SENTENCE extracted in step 1)?

          If so, what METADATA did you find? If you did not find a value for a METADATA item, just omit that item, do NOT report it.

          IMPORTANT: METADATA is NOT written in free-form natural language, but follows a LIST format.

          IMPORTANT: Analyze the ORIGINAL INPUT, not the FORMAT EXAMPLE.

          Hints:

          + Compare your extracted parts, from step 1, to the ORIGINAL INPUT.
          + Where do your extracted parts appear in the ORIGINAL INPUT?

        - Report the answer.
        - Move on to step 3.

        Step 3:

        - Start a new section for ANSWER.
        - Look at your analysis from step 2.
        - If you found metadata in step 2, reply "YES". Otherwise, if you did not find metadata in step 2, reply "NO".
        - This final answer will be sent to a computer program that cannot understand natural language, so please print it in plain text without formatting.

        -----

        **ORIGINAL INPUT, please analyze this**
        """)
        # # Sanity-check for presence of author list, for logging a warning if the LLM thinks the author list is missing.
        # # This doesn't work well with an 8B model, even with majority voting ( see Wang et al., 2023 https://arxiv.org/abs/2203.11171 ).
        # # So let's just skip this, and use a heuristic check on the final result (whether the extracted names are present in the original text).
        # T = llm_settings.request_data["temperature"]
        # llm_settings.request_data["temperature"] = 0.3
        # llm_outputs = []
        # answers = collections.Counter()
        # for _ in range(3):
        #     print(_ + 1, end="", file=sys.stderr)
        #     record = extract(llm_settings, f"{prompt_check_authorlist}\n\n{text}", "*")
        #     scrubbed_output_text = record.reply
        #     has_author_list = scrubbed_output_text[-20:].split()[-1].strip().upper()  # Last word of output, in uppercase.
        #     has_author_list = has_author_list.translate(str.maketrans('', '', string.punctuation))  # Strip punctuation, in case of spurious formatting.
        #     answers[has_author_list] += 1
        #     llm_outputs.append(scrubbed_output_text)
        # llm_settings.request_data["temperature"] = T
        # votes = answers.most_common()  # [(item0, count0), ...]
        #
        # logger.info("Detection reports follow.")
        # for llm_output in llm_outputs:
        #     logger.info(llm_output)
        #
        # if not any(count > 1 for answer, count in votes):  # no majority?
        #     logger.info(f"Input file '{unique_id}': LLM could not detect whether there is an author list. Votes: {votes}")
        # else:
        #     has_author_list = votes[0][0]
        #     if has_author_list in ("YES", "NO"):
        #         if has_author_list == "NO":
        #             logger.info(f"Input file '{unique_id}': LLM says the author list is missing; manual check recommended.")
        #     else:
        #         logger.info(f"Input file '{unique_id}': LLM returned unknown author list detection result '{has_author_list}', should be 'YES' or 'NO'; manual check recommended.")

        record_1 = extract(llm_settings, context, f"{prompt_get_authors}\n-----\n\n{text}", "A", "EXTRACT AUTHORS", n_retries)
        scrubbed_output_text_1 = record_1.reply
        logger.debug(f"\n        EXTRACT AUTHORS    : {scrubbed_output_text_1}")

        if _is_sentinel(scrubbed_output_text_1, _NO_AUTHORS):
            error_msg = f"Input file '{unique_id}': LLM found no author list; omitting the field, manual check recommended."
            logger.warning(error_msg)
            error_info.write(f"{error_msg}\n")
            error_info.write(f"Full LLM output trace for step EXTRACT AUTHORS:\n{'-' * 80}\n{full_output_trace(record_1)}\n{'-' * 80}\n")
            return status_flagged, error_info.getvalue(), None

        # Usually the output is correct, but sometimes:
        #   - The list may use commas instead of the word "and"
        #   - The list may have additional hallucinated authors
        #   - Some authors may be missing from the list
        # so we perform some post-processing and checking.

        # Here the LLM (Qwen3 2507) sometimes gets stuck overthinking.
        record_2 = extract(llm_settings, context, prompt_drop_author_affiliations.format(author_names=scrubbed_output_text_1), "a",
                           "DROP AFFILIATIONS", n_retries)
        scrubbed_output_text_2 = record_2.reply
        logger.debug(f"\n        DROP AFFILIATIONS  : {scrubbed_output_text_2}")

        record_3 = extract(llm_settings, context, prompt_reformat_author_separators.format(author_names=scrubbed_output_text_2), ".",
                           "REFORMAT SEPARATORS", n_retries)
        scrubbed_output_text_3 = record_3.reply
        logger.debug(f"\n        REFORMAT SEPARATORS: {scrubbed_output_text_3}")

        if scrubbed_output_text_3.endswith("and"):  # Remove spurious "and" with one author. Can happen especially if, in the original abstract, a comma follows the single author name.
            scrubbed_output_text_3 = scrubbed_output_text_3[:-3]

        authors = scrubbed_output_text_3.strip()  # Final result from LLM. Remove extra whitespace, just in case.

        # Sanity-check the LLM output.
        #
        # Initial list of authors. Using BibTeX's "and" convention is more reliable than a comma, because "Ally Author" and "Author, Ally" are both valid formats for a single name.
        authors_list = [author.strip() for author in authors.split(" and ")]

        # Each author name usually has at least two space-separated components (commonly, first and last name).
        # This doesn't always hold (e.g. "Google", "OpenAI"), so we just issue a warning.
        possibly_broken_names = []
        for author in authors_list:
            if len(author.split(" ")) < 2:
                possibly_broken_names.append(author)
        if len(possibly_broken_names):
            plural_s = "s" if len(possibly_broken_names) > 1 else ""
            error_msg = f"Input file '{unique_id}': Extractor returned one-component or blank author name{plural_s}; manual check recommended: {possibly_broken_names}"
            logger.warning(error_msg)
            error_info.write(f"{error_msg}\n")
            error_info.write(f"Final result for step EXTRACT AUTHORS: {scrubbed_output_text_1}\n")
            error_info.write(f"Final result for step DROP AFFILIATIONS: {scrubbed_output_text_2}\n")
            error_info.write(f"Final result for step REFORMAT SEPARATORS: {scrubbed_output_text_3}\n")
            error_info.write(f"Full LLM output trace for step EXTRACT AUTHORS:\n{'-' * 80}\n{full_output_trace(record_1)}\n{'-' * 80}\n")
            error_info.write(f"Full LLM output trace for step DROP AFFILIATIONS:\n{'-' * 80}\n{full_output_trace(record_2)}\n{'-' * 80}\n")
            error_info.write(f"Full LLM output trace for step REFORMAT SEPARATORS:\n{'-' * 80}\n{full_output_trace(record_3)}\n{'-' * 80}\n")
            status = status_flagged

        # No author should be listed more than once.
        authors_counter = collections.Counter(authors_list)
        duplicate_names = [author for author, count in authors_counter.items() if count > 1]
        if len(duplicate_names):
            plural_s = "s" if len(duplicate_names) > 1 else ""
            error_msg = f"Input file '{unique_id}': Extractor returned duplicate author name{plural_s}; de-duplicated, but manual check recommended: {duplicate_names}"
            logger.warning(error_msg)
            error_info.write(f"{error_msg}\n")
            error_info.write(f"Final result for step EXTRACT AUTHORS: {scrubbed_output_text_1}\n")
            error_info.write(f"Final result for step DROP AFFILIATIONS: {scrubbed_output_text_2}\n")
            error_info.write(f"Final result for step REFORMAT SEPARATORS: {scrubbed_output_text_3}\n")
            error_info.write(f"Full LLM output trace for step EXTRACT AUTHORS:\n{'-' * 80}\n{full_output_trace(record_1)}\n{'-' * 80}\n")
            error_info.write(f"Full LLM output trace for step DROP AFFILIATIONS:\n{'-' * 80}\n{full_output_trace(record_2)}\n{'-' * 80}\n")
            error_info.write(f"Full LLM output trace for step REFORMAT SEPARATORS:\n{'-' * 80}\n{full_output_trace(record_3)}\n{'-' * 80}\n")
            status = status_flagged

        # Add missing periods for abbreviated first and middle names. Sometimes these drop out during the LLM correction passes.
        def fix_abbrevs(author: str) -> str:
            """Ally A Author -> Ally A. Author"""
            parts = author.split()
            return " ".join(part if len(part) > 1 else f"{part}." for part in parts)
        authors_list = [fix_abbrevs(author) for author in authors_counter.keys()]  # de-duplicate, and fix abbreviated names

        authors = " and ".join(authors_list)

        logger.debug(f"\n        formatted : {authors}")

        # Authors should be separated by "and", not by commas.
        #
        # Heuristic check for final result:
        #   - Split at "and" (we already have this, as `authors_list`).
        #
        #   - Consider each item. Split it at comma.
        #
        #   - Consider each subitem. Check the number of components in a whitespace split.
        #
        #     For most names, each comma-separated part should have only one whitespace-separated component.
        #
        #     In rare cases, this is not true, e.g. "Edward von Example Jr." --BibTeX standard format--> "von Example, Jr., Edward".
        #     But the "von" part may be "van", "de", "de la", or whatever, so it can't be detected easily.
        #     Emitting a false positive warning in such rare cases should be fine in practice.
        #
        # Consider:
        #
        # OK, single author (plus von, Jr., and similar):
        #   - "Ally Author"
        #   - "Author, Ally"
        #   - "A. Author"
        #   - "Author, A."
        #
        # OK, two or more authors:
        #   - "Ally Author and Edward Example"
        #   - "Author, Ally and Example, Edward"
        #   - "A. Author and E. Example"
        #   - "Author, A. and Example, E."
        #
        # WRONG, comma-separated authors:
        #   - "Ally Author, Edward Example"
        #   - "A. Author, E. Example"
        #
        # WRONG, mix of comma and "and":
        #   - "Ally Author, Edward Example and Oscar Other"
        #
        # Inconsistent, but not strictly wrong:
        #   - "Ally Author and Example, Edward"
        #
        # The cases marked "WRONG" are caught by this check.
        format_warning_logged = False
        llm_warning_logged = False
        for author in authors_list:  # "and"-separated authors
            if "," in author:  # "Ally Author" -> OK; "Author, Ally" -> check it.
                parts = [part.strip() for part in author.split(",")]

                if not format_warning_logged:
                    for part in parts:  # comma-separated parts
                        components = part.split()
                        if len(components) > 1:
                            error_msg = f"Input file '{unique_id}': Possibly broken format in processed author list; manual check recommended: '{authors}'"
                            logger.warning(error_msg)
                            error_info.write(f"{error_msg}\n")
                            error_info.write(f"Final result for step EXTRACT AUTHORS: {scrubbed_output_text_1}\n")
                            error_info.write(f"Final result for step DROP AFFILIATIONS: {scrubbed_output_text_2}\n")
                            error_info.write(f"Final result for step REFORMAT SEPARATORS: {scrubbed_output_text_3}\n")
                            error_info.write(f"Full LLM output trace for step EXTRACT AUTHORS:\n{'-' * 80}\n{full_output_trace(record_1)}\n{'-' * 80}\n")
                            error_info.write(f"Full LLM output trace for step DROP AFFILIATIONS:\n{'-' * 80}\n{full_output_trace(record_2)}\n{'-' * 80}\n")
                            error_info.write(f"Full LLM output trace for step REFORMAT SEPARATORS:\n{'-' * 80}\n{full_output_trace(record_3)}\n{'-' * 80}\n")
                            format_warning_logged = True
                            status = status_flagged
                            break
            else:
                parts = [author.strip()]  # needed for the other check.

            # Re-scan once more.
            #
            # Each name should appear in the original `text`, because the name supposedly came from the author list of the original abstract.
            #
            # The check is very simplistic, and won't catch everything. We scan each part separately - so each first name and each last name,
            # ON ITS OWN (regardless of how they are paired in the original!), must appear in `text`, or we emit a warning.
            #
            # This fixes the "...and John F. Kennedy" issue, unless there is some other "John", some other "F.", AND some other "Kennedy"
            # mentioned in `text`, each separately.
            #
            # Note this emits a false positive for "Example, E.", if the original abstract spells the name as "Example E" (with no period,
            # so that the string "E." is not present in `text`).
            #
            # It is difficult to be more thorough or accurate (short of trusting an LLM to do it, which is the whole issue here) due to
            # the variety of name formats supported by BibTeX.
            if not llm_warning_logged:
                for part in parts:  # comma-separated parts
                    components = part.split()
                    if any(component not in text for component in components):
                        error_msg = f"Input file '{unique_id}': Possible LLM error in processed author list; manual check recommended: '{authors}'"
                        logger.warning(error_msg)
                        error_info.write(f"{error_msg}\n")
                        error_info.write(f"Final result for step EXTRACT AUTHORS: {scrubbed_output_text_1}\n")
                        error_info.write(f"Final result for step DROP AFFILIATIONS: {scrubbed_output_text_2}\n")
                        error_info.write(f"Final result for step REFORMAT SEPARATORS: {scrubbed_output_text_3}\n")
                        error_info.write(f"Full LLM output trace for step EXTRACT AUTHORS:\n{'-' * 80}\n{full_output_trace(record_1)}\n{'-' * 80}\n")
                        error_info.write(f"Full LLM output trace for step DROP AFFILIATIONS:\n{'-' * 80}\n{full_output_trace(record_2)}\n{'-' * 80}\n")
                        error_info.write(f"Full LLM output trace for step REFORMAT SEPARATORS:\n{'-' * 80}\n{full_output_trace(record_3)}\n{'-' * 80}\n")
                        llm_warning_logged = True
                        status = status_flagged
                        break

        return status, error_info.getvalue(), authors

    prompt_get_title = dedent(f"""What is the title of the abstract?

    Please reply only with the title.

    Use plain text.

    Do NOT place quotation marks around the title.

    A section heading such as "Abstract" is NOT the title.

    If the abstract has no title, reply only with: {_NO_TITLE}
    """)

    def extract_title(context: env, unique_id: str, text: str) -> tuple[sym, str, str | None]:
        """Extract the title from the fulltext of a conference abstract.

        `context`: the document's, as `extract` takes it.
        `unique_id`: input file identifier, for error messages
        `text`: the full text

        Returns `(status, notes, title)`, the title `None` when the document seems to have none.
        """
        status = status_success
        error_info = io.StringIO()
        text = strip_postamble(text)

        # Not blank once the quotation marks and periods it is stripped of below are gone.
        record = extract(llm_settings, context, f"{prompt_get_title}\n-----\n\n{text}", "T", "EXTRACT TITLE", n_retries,
                         accept=lambda reply: bool(reply.strip().strip("\"'.").strip()))
        scrubbed_output_text = record.reply
        logger.debug(f"\n        original : {scrubbed_output_text}")

        title = scrubbed_output_text.strip()

        # The prompt says a heading is not the title, and a model sometimes reports one anyway.
        if _is_sentinel(title, _NO_TITLE) or _is_sentinel(title, "ABSTRACT"):
            error_msg = f"Input file '{unique_id}': LLM found no title (reply: '{title}'); omitting the field, manual check recommended."
            logger.warning(error_msg)
            error_info.write(f"{error_msg}\n")
            error_info.write(f"Full LLM output trace for EXTRACT TITLE:\n{'-' * 80}\n{full_output_trace(record)}\n{'-' * 80}\n")
            return status_flagged, error_info.getvalue(), None

        # Strip spurious period
        while title.endswith("."):
            title = title[:-1]

        # Strip spurious quotation marks (they may still occasionally happen even though we instruct the LLM not to emit them)
        while (title.startswith('"') and title.endswith('"')) or (title.startswith("'") and title.endswith("'")):
            title = title[1:-1]

        # Strip spurious period, again (inside quotation marks)
        while title.endswith("."):
            title = title[:-1]

        # Sanity check: are all words of the LLM-extracted title present in the original text?
        components = title.split()
        if any(component not in text for component in components):
            error_msg = f"Input file '{unique_id}': Possible LLM error in processed title: one or more of the title words not found in original text; manual check recommended: '{title}'"
            logger.warning(error_msg)
            error_info.write(f"{error_msg}\n")
            error_info.write(f"Final result for EXTRACT TITLE: {scrubbed_output_text}\n")
            error_info.write(f"Full LLM output trace for EXTRACT TITLE:\n{'-' * 80}\n{full_output_trace(record)}\n{'-' * 80}\n")
            status = status_flagged

        logger.debug(f"\n        formatted: {title}")

        return status, error_info.getvalue(), title

    prompt_get_keywords = dedent("""What are the keywords, as given in the abstract?

    This will be used for a bibliography. We need only the keywords, COPIED AS-IS from the original abstract.

    You can find the keywords on a separate line that starts with "Keywords:" or "Key words:".

    Please reply only with a comma-separated list of keywords.

    Use plain text, no formatting.

    IMPORTANT: Only copy the existing list of keywords from the abstract; do NOT add your own keywords.

    IMPORTANT: The list of keywords in the original abstract may end abruptly. This is fine. Do NOT add additional keywords from the main text.
    """)

    def extract_keywords(context: env, unique_id: str, text: str) -> tuple[sym, str, str | None]:
        """Extract the keywords from the fulltext of a conference abstract.

        `context`: the document's, as `extract` takes it.
        `unique_id`: input file identifier, for error messages
        `text`: the full text

        Returns `(status, notes, keywords)`, the keywords comma-separated, or `None` when the document lists none.
        """
        status = status_success
        error_info = io.StringIO()
        text = strip_postamble(text)

        # Sanity check that the abstract has keywords before we run the LLM to extract them.
        match = kws_pattern.search(text)
        if match is None:
            logger.warning(f"Input file '{unique_id}': No keywords provided in original input, skipping keyword extraction.")
            keywords = None  # No keywords provided
        else:
            def strip_heading(reply: str) -> str:
                """Remove a spurious heading and surrounding whitespace."""
                for heading in ("Keywords:", "KEYWORDS:", "Key words:", "Key Words:", "KEY WORDS:"):
                    if reply.startswith(heading):
                        reply = reply[len(heading):]
                return reply.strip()
            record = extract(llm_settings, context, f"{prompt_get_keywords}\n-----\n\n{text}", "K", "EXTRACT KEYWORDS", n_retries,
                             accept=lambda reply: bool(strip_heading(reply)))
            scrubbed_output_text = record.reply
            logger.debug(f"\n        original : {scrubbed_output_text}")
            keywords = strip_heading(scrubbed_output_text)

            # Strip spurious period(s) at end
            while keywords[-1] == ".":
                keywords = keywords[:-1]

            keywords_list = [keyword.strip() for keyword in keywords.split(",")]

            # No keyword should be listed more than once.
            keywords_counter = collections.Counter(keywords_list)
            duplicate_keywords = [keyword for keyword, count in keywords_counter.items() if count > 1]
            if len(duplicate_keywords):
                plural_s = "s" if len(duplicate_keywords) > 1 else ""
                error_msg = f"Input file '{unique_id}': Extractor returned duplicate keyword{plural_s}; de-duplicated, but manual check recommended: {duplicate_keywords}"
                logger.warning(error_msg)
                error_info.write(f"{error_msg}\n")
                error_info.write(f"Final result for EXTRACT KEYWORDS:\n{'-' * 80}\n{scrubbed_output_text}\n")
                error_info.write(f"Full LLM output trace for EXTRACT KEYWORDS:\n{'-' * 80}\n{full_output_trace(record)}\n")
                status = status_flagged
            keywords = list(keywords_counter.keys())

            # Sanity check: are all LLM-extracted extracted keywords present in the original text?
            #
            # We check multiple-word keywords ("finite element method") one word at a time.
            #
            # This isn't 100% foolproof, but checking for the whole word runs into text layout issues very often,
            # producing lots of false positives. The keyword may be split across a linefeed in the original,
            # causing the exact complete keyword string not to appear in the input.
            for keyword in keywords:
                components = keyword.split()
                if any(component not in text for component in components):
                    error_msg = f"Input file '{unique_id}': Possible LLM error in processed keywords: one or more of the keywords not found in original text; manual check recommended"
                    logger.warning(error_msg)
                    error_info.write(f"{error_msg}\n")
                    error_info.write(f"Final result for EXTRACT KEYWORDS:\n{'-' * 80}\n{scrubbed_output_text}\n")
                    error_info.write(f"Full LLM output trace for EXTRACT KEYWORDS:\n{'-' * 80}\n{full_output_trace(record)}\n")
                    status = status_flagged
                    break

            keywords = ", ".join(keywords)  # de-duplicate

            # TODO: Other sanity checks.

            logger.debug(f"\n        formatted: {keywords}")

        return status, error_info.getvalue(), keywords

    # This is by far the most difficult part: grab text that has no obvious programmatically detectable starting delimiter.
    #
    # TODO: Using the LLM is the obvious general solution, but unreliable.
    #  - Sometimes extra metadata (e.g. title, institutions, keywords) is returned.
    #  - Sometimes parts of the main text are omitted.
    #  - Maybe this needs a larger LLM, or at least a higher quant of the 8B?
    # TODO: Self-critique the result. In the output, does the main text begin right away, or is there metadata at the beginning? Re-roll (at T=1) if unsuccessful.
    #
    prompt_get_abstract = dedent("""Please extract the COMPLETE main text of the abstract.

    This will be used as content for the Abstract field in a bibliography. We need only the main text: NO title, NO author names, NO institutions, NO street addresses, NO email addresses, NO keywords.

    Please reply only with the extracted MAIN TEXT.

    IMPORTANT: Do NOT summarize or reword the input in any way. Just copy the main text, and ONLY the main text, from the original abstract AS-IS.

    IMPORTANT: Please make sure to copy the COMPLETE main text, not just a part of it. The text may have several paragraphs.

    Note that the reference list has already been stripped. It is normal for the abstract to refer to citations not listed here.
    """)

    def extract_abstract(context: env, unique_id: str, text: str) -> tuple[sym, str, str | None]:
        """Extract the main text from the fulltext of a conference abstract.

        `context`: the document's, as `extract` takes it.
        `unique_id`: input file identifier, for error messages
        `text`: the full text

        Returns `(status, notes, abstract)`, the abstract being the main text of the conference abstract.
        """
        status = status_success
        error_info = io.StringIO()
        text = strip_postamble(text)

        record = extract(llm_settings, context, f"{prompt_get_abstract}\n-----\n\n{text}", ".", "EXTRACT ABSTRACT", n_retries)
        abstract = record.reply.strip()

        abstract = _get_dehyphenator().dehyphenate(abstract)

        return status, error_info.getvalue(), abstract

    # Format is {bibtex_fieldname: (kind, thing, progress_symbol)}.
    # Details in master copy, in the docstring of this function.
    prompts = {
        "author": ("function", extract_authors, None),
        "year": ("literal", conference_year, None),
        "title": ("function", extract_title, None),
    }
    if conference_booktitle is not None:
        prompts["booktitle"] = ("literal", conference_booktitle, None)
    if conference_note is not None:
        prompts["note"] = ("literal", conference_note, None)
    if conference_url is not None:
        prompts["url"] = ("literal", conference_url, None)
    prompts["keywords"] = ("function", extract_keywords, None)
    prompts["abstract"] = ("function", extract_abstract, None)
    material = "\x00".join((prompt_get_authors, prompt_drop_author_affiliations, prompt_reformat_author_separators,
                            prompt_get_title, prompt_get_keywords, prompt_get_abstract))
    return prompts, material


# --------------------------------------------------------------------------------
# Processing logic

def listpdf(path: str) -> list[str]:
    """Return a sorted list of the PDF filenames directly in `path` (this one directory, non-recursive).

    Recursion across a directory tree is the caller's job: `main` walks the input directory with `os.walk`
    (dropping the output / failed-output directories so processed files are not re-scanned) and hands
    `process_abstracts` the full list of directories, each of which is listed here.
    """
    return list(sorted(filename for filename in os.listdir(path) if filename.endswith(".pdf")))

def extract_fields(llm_settings: env,
                   prompts: dict,
                   unique_id: str,
                   text: str,
                   maybe_abort: netutil.Abort | None = None) -> tuple[dict | sweep.Failure, list[agent.TurnRecord]]:
    """Ask the LLM for one conference abstract's fields: the ledger's answer for the document.

    `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.
    `prompts`: Obtain by calling `setup_prompts` at app start time.
    `unique_id`: input file identifier, for error messages
    `text`: The text to process (extracted from a conference abstract PDF).
    `maybe_abort`: the sweep's handle for the call in flight, so that cancelling reaches it.

    Returns `(answer, turns)`. `answer` is `{"status", "fields", "notes"}`: `"success"` or `"flagged"`, the
    value per BibTeX field (`None` to omit the field), and the notes explaining a flag. Or a
    `sweep.Failure` when a step gave no usable reply, so that the document is asked again on the next run.
    `turns` are every model call made, for the sweep's traces. The conference literals are not in the
    answer; `format_entry` applies them, so a changed literal needs no re-asking.
    """
    # One field at a time: whatever can be filled programmatically is, and the LLM gets one small task at a
    # time, populating only the field's content.
    context = env(unique_id=unique_id, maybe_abort=maybe_abort, turns=[])
    status = status_success
    fields = {}
    notes = []
    maybe_failure = None
    with timer() as tim:
        print("    ", end="", file=sys.stderr)  # Indent the progress indicator
        sys.stderr.flush()
        try:
            for field_key, (data_kind, data, progress_symbol) in prompts.items():
                if data_kind == "literal":
                    continue
                elif data_kind == "prompt":
                    # A single-turn conversation per query, which typically costs a full prompt rescan each time.
                    fields[field_key] = extract(llm_settings, context, f"{data}\n-----\n\n{text}", "p", field_key).reply  # "p" for "prompt mode"
                elif data_kind == "function":
                    field_status, field_notes, fields[field_key] = data(context, unique_id, text)
                    if field_status is status_flagged:  # one flagged field flags the entry, which wants a manual check
                        status = status_flagged
                        notes.append(field_notes)
                else:
                    raise ValueError(f"Unknown data kind '{data_kind}'; please check your settings.")
        except _NoAnswer as exc:
            maybe_failure = sweep.Failure(str(exc))
    if maybe_failure is not None:
        print(f"no answer, in {tim.dt:0.2f}s", file=sys.stderr)
        return maybe_failure, context.turns
    print(f"done in {tim.dt:0.2f}s", file=sys.stderr)
    return {"status": "success" if status is status_success else "flagged",
            "fields": fields,
            "notes": "\n\n".join(notes)}, context.turns

def format_entry(prompts: dict, conference_slug: str, unique_id: str, answer: dict) -> str:
    """One BibTeX entry, from a ledger answer and the conference literals in `prompts`.

    Fields come in the order `prompts` lists them; one whose value is `None` is left out.
    """
    lines = [f"@incollection{{{conference_slug}-{unique_id},"]
    for field_key, (data_kind, data, _progress_symbol) in prompts.items():
        value = data if data_kind == "literal" else answer["fields"].get(field_key)
        if value is not None:
            lines.append(f"    {field_key} = {{{bibtex_escape(value)}}},")
    lines.append("}")
    return "\n".join(lines)

def write_outputs(ledger: sweep.Ledger, prompts: dict, opts: argparse.Namespace) -> collections.Counter:
    """Rewrite the BibTeX outputs from everything `ledger` holds, and return how many entries are which.

    From the whole ledger rather than this run's items: an earlier run's PDFs may have been moved out of the
    input directory, and their entries belong in the output all the same. Flagged entries go to the failed
    output when there is one, and to the success output otherwise; each also gets an `_errors.txt` with
    its notes and a `.bib` of its own, beside its PDF, when a directory for those was given.
    """
    answers = ledger.answers()
    counts = collections.Counter(answer["status"] for answer in answers.values())
    success, flagged = [], []
    for unique_id in sorted(answers):
        answer = answers[unique_id]
        entry = format_entry(prompts, opts.conference_slug, unique_id, answer)
        if answer["status"] == "success":
            success.append(entry)
            # Left behind by an earlier, flagged answer that asking again has replaced.
            for directory in (opts.failed_output_dir, opts.output_dir):
                for stale in (f"{unique_id}_errors.txt", f"{unique_id}.bib"):
                    if directory is not None and os.path.exists(path := os.path.join(directory, stale)):
                        os.remove(path)
            continue
        flagged.append(entry)
        maybe_flagged_dir = opts.failed_output_dir or opts.output_dir
        if maybe_flagged_dir is not None:
            with open(os.path.join(maybe_flagged_dir, f"{unique_id}_errors.txt"), "w", encoding="utf-8") as f:
                f.write(answer["notes"])
            with open(os.path.join(maybe_flagged_dir, f"{unique_id}.bib"), "w", encoding="utf-8") as f:
                f.write(f"{entry}\n\n")
    if opts.failed_filename is None:
        success, flagged = success + flagged, []
    for maybe_filename, entries in ((opts.success_filename, success), (opts.failed_filename, flagged)):
        text = "".join(f"{entry}\n\n" for entry in entries)  # one blank line after each entry
        if maybe_filename is None:
            sys.stdout.write(text)
        else:
            with open(maybe_filename, "w", encoding="utf-8") as f:
                f.write(text)
    return counts

def collect_items(paths: list[str], opts: argparse.Namespace) -> list[tuple[str, str]]:
    """The PDFs a run covers, as `(unique_id, fullpath)`: those in `paths`, and those already sorted into
    `-o` and `-of`, so that one asked again is found wherever it was moved. The sweep skips what is answered,
    so listing the sorted ones costs nothing.

    Raises `ValueError` when two PDFs have the same filename stem, since their BibTeX entries would get the
    same key.
    """
    directories = list(paths) + [directory for directory in (opts.output_dir, opts.failed_output_dir)
                                 if directory is not None and directory not in paths]
    items = []
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for filename in listpdf(directory):
            items.append((os.path.splitext(filename)[0], os.path.join(directory, filename)))  # "/foo/blah.pdf" -> "blah"
    stems = collections.Counter(unique_id for unique_id, _ in items)
    if duplicates := sorted(unique_id for unique_id, count in stems.items() if count > 1):
        raise ValueError(f"These filenames occur in more than one directory, and would give BibTeX entries the same key: "
                         f"{duplicates}. Rename them apart, or remove the copies.")
    return items

def sort_pdf(fullpath: str, status: str, opts: argparse.Namespace) -> str:
    """Move an answered PDF to where its outcome says, and return where it is now.

    Done goes to `-o`, flagged to `-of` (or `-o`, when there is no `-of`). Where the outcome names no
    directory, a PDF already sorted elsewhere goes back to the input directory, so that a flagged document
    cleared by asking again does not stay among the flagged ones.
    """
    maybe_target = opts.output_dir
    if status == "flagged" and opts.failed_output_dir is not None:
        maybe_target = opts.failed_output_dir
    current = os.path.dirname(fullpath)
    if maybe_target is None:
        if current not in (opts.output_dir, opts.failed_output_dir):
            return fullpath  # unsorted, and staying so
        maybe_target = opts.input_dir
    if os.path.abspath(current) == os.path.abspath(maybe_target):
        return fullpath
    destination = os.path.join(maybe_target, os.path.basename(fullpath))
    shutil.move(fullpath, destination)
    return destination

def process_abstracts(paths: list[str], opts: argparse.Namespace) -> None:
    """Process all PDFs under `paths`, recursively.

    Each PDF is assumed to contain a conference abstract (the final file sent to the conference organizers in PDF format), and nothing else.

    A run is a `raven.librarian.sweep` over the PDFs, keyed on the filename stem, which the BibTeX entry key
    is built from. Its ledger is what makes a run resumable: running the same command again asks only what
    is not yet answered, and asks again what had no usable reply. The `.bib` outputs are rewritten from the
    ledger at the end of every run, stopped ones included.
    """
    # A batch tool stops here rather than starting when there is no model: a precise diagnosis at document
    # 1 beats an extraction failure per document. A backend that goes away mid-run stops the run too; the
    # sweep sees to that.
    if not llmclient.test_connection(opts.backend_url):
        raise RuntimeError("Failed to connect to LLM backend, cannot proceed.")
    llm_settings = llmclient.setup(backend_url=opts.backend_url)
    if (status := llmclient.backend_status(llm_settings)) is llmclient.backend_has_no_model:
        headline, advice = llmclient.describe_backend_status(status, opts.backend_url)
        raise RuntimeError(f"{headline} {advice}")
    prompts, material = setup_prompts(llm_settings,
                                      n_retries=opts.retries,
                                      conference_year=opts.conference_year,
                                      conference_booktitle=opts.conference_booktitle,
                                      conference_note=opts.conference_note,
                                      conference_url=opts.conference_url)
    logger.info(f"Connected to LLM backend at {opts.backend_url}")
    plural_s = "s" if opts.retries != 1 else ""
    logger.info(f"    prompts set up to use up to {opts.retries} attempt{plural_s}")
    # No character logged: the extraction steps run with no character card, so `llm_settings.char` names a
    # persona that takes no part in this run.

    items = collect_items(paths, opts)
    ledger = sweep.Ledger(opts.ledger_dir, f"pdf2bib-{opts.conference_slug}",
                          sweep.fingerprint(llm_settings, material, declared=opts.declare))
    logger.info(f"Ledger: {ledger.path}")

    reask = set(opts.reask or ())
    if opts.reask_flagged:
        reask |= {unique_id for unique_id, answer in ledger.answers().items() if answer["status"] == "flagged"}
    if unknown := sorted(reask - {unique_id for unique_id, _ in items}):
        logger.warning(f"--reask: no PDF found for {unknown}, so these cannot be asked again.")
    if reask := sorted(reask - set(unknown)):
        logger.info(f"Asking again: {reask}")
        ledger.reask(reask)

    def ask_batch(llm_settings: env, batch: list, maybe_abort: netutil.Abort) -> tuple[dict, list]:
        ((unique_id, fullpath),) = batch  # one PDF per call; each PDF already takes several
        logger.info(f"{fullpath}")
        # The extraction need not be perfect, the LLM cleaning up what it does not get right (affiliation
        # symbols in author names, say). A PDF with no text layer (a scan) gives `None`, so the LLM gets
        # the empty string and says it found nothing.
        text_from_pdf = docextract.extract_text(fullpath) or ""
        answer, turns = extract_fields(llm_settings, prompts, unique_id, text_from_pdf, maybe_abort)
        return {unique_id: answer}, turns

    fullpaths = dict(items)

    def move_done(event: env) -> None:
        """Sort each PDF just answered, where directories for that were given. Optional: resuming does not
        depend on it, the ledger knowing what is done."""
        latest = ledger.latest()
        for unique_id in event.ids:
            entry = latest.get(unique_id)
            if entry is None or entry["status"] != "ok":  # failed: stays, and is asked again next run
                continue
            fullpaths[unique_id] = sort_pdf(fullpaths[unique_id], entry["answer"]["status"], opts)

    result = sweep.Sweep(llm_settings, ask_batch, items, ledger,
                         header=sweep.instrument_header(llm_settings, declared=opts.declare),
                         on_item=move_done).run()

    counts = write_outputs(ledger, prompts, opts)
    failed = sorted(unique_id for unique_id, entry in ledger.latest().items() if entry["status"] == "failed")
    # Counts moved behind labels, so that no verb has to agree with them.
    logger.info(f"Entries written: {counts['success']} done, {counts['flagged']} flagged for a manual check. "
                f"Documents without a usable answer, to be asked again on the next run: {len(failed)}.")
    for unique_id in failed:
        logger.info(f"    {unique_id}: {ledger.latest()[unique_id]['error']}")
    if result.outcome is not sweep.completed:
        raise RuntimeError(f"The run stopped ({result.outcome}): {result.message or 'cancelled'} Run again to resume.")

# --------------------------------------------------------------------------------
# Main program

def run(opts: argparse.Namespace) -> None:
    """Run the conversion, as `raven-pdf2bib` does. `opts` is its parsed command line; see `pdf2bib_cli`."""
    # The parser lives in `pdf2bib_cli`, so that `--help` does not wait for this module's imports. It also
    # configures logging before importing this module.
    raven_server_url = opts.server_url if opts.server_url is not None else client_config.raven_server_url
    if opts.server_url is not None:
        logger.info(f"Using Raven server '{raven_server_url}' from --server-url, overriding the configured '{client_config.raven_server_url}'.")
    _initialize_client_api(raven_server_url)

    if opts.retries < 1:
        logger.info(f"-r, --retries: At least one attempt is required, got {opts.retries}. Setting to 1.")
        opts.retries = 1

    if opts.success_filename is not None:
        success_filename = pathlib.Path(opts.success_filename).expanduser().resolve()
        success_bib_dir = success_filename.parent
        common_utils.create_directory(success_bib_dir)
        opts.success_filename = str(success_filename)

    if opts.failed_filename is not None:
        failed_filename = pathlib.Path(opts.failed_filename).expanduser().resolve()
        failed_bib_dir = failed_filename.parent
        common_utils.create_directory(failed_bib_dir)
        opts.failed_filename = str(failed_filename)

    if opts.input_dir is None:
        opts.input_dir = "."
    opts.input_dir = str(pathlib.Path(opts.input_dir).expanduser().resolve())
    logger.info(f"Processing PDF files from {opts.input_dir}.")

    if opts.ledger_dir is None:
        opts.ledger_dir = str(pathlib.Path(opts.success_filename).parent) if opts.success_filename is not None else "."
    opts.ledger_dir = str(pathlib.Path(opts.ledger_dir).expanduser().resolve())
    logger.info(f"Keeping the ledger in {opts.ledger_dir}.")

    if opts.output_dir is not None:
        opts.output_dir = str(pathlib.Path(opts.output_dir).expanduser().resolve())
        kind_str = "successful" if opts.failed_output_dir is not None else "all"
        logger.info(f"Moving {kind_str} done files to {opts.output_dir}.")
        common_utils.create_directory(opts.output_dir)

    if opts.failed_output_dir is not None:
        opts.failed_output_dir = str(pathlib.Path(opts.failed_output_dir).expanduser().resolve())
        logger.info(f"Moving failed done files to {opts.failed_output_dir}.")
        common_utils.create_directory(opts.failed_output_dir)

    blacklist = []
    paths = []
    for root, dirs, files in os.walk(opts.input_dir):
        paths.append(root)
        for x in blacklist:
            if x in dirs:
                dirs.remove(x)
    paths = list(uniqify(str(pathlib.Path(p).expanduser().resolve()) for p in paths))

    # Avoid recursing into output directories
    if opts.output_dir is not None:
        paths = [p for p in paths if not p.startswith(opts.output_dir)]
    if opts.failed_output_dir is not None:
        paths = [p for p in paths if not p.startswith(opts.failed_output_dir)]

    process_abstracts(sorted(paths), opts)
