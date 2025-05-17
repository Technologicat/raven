#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert PDF conference abstracts into a BibTeX database.

The heavy lifting is done by `pdftotext` (from poppler-utils) and an OpenAI compatible LLM.

USAGE:

    python pdf_abstracts_to_bibtex.py -o done 1>entries.bib

This will write `entries.bib`, and move the input PDFs into the `done` subdirectory as they are processed.
This allows easily continuing later, if there are lots of input files. A file is moved if and only if it was
successfully processed, AFTER printing its bibtex entry.
"""

# TODO: Apply BibTeX escapes (especially "%", "{", "}") - the LLM doesn't seem to do that even if we instruct it to do so. So we don't currently even try.

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output log messages only from this module. Handy for debugging.
# https://stackoverflow.com/questions/17275334/what-is-a-correct-way-to-filter-different-loggers-using-python-logging
for handler in logging.root.handlers:
    handler.addFilter(logging.Filter(__name__))

import argparse
import collections
from functools import partial
import io
import os
import pathlib
import re
import shutil
import subprocess
import sys
from textwrap import dedent
import traceback
from typing import List

from unpythonic import timer, ETAEstimator, uniqify
from unpythonic.env import env

from . import config
from . import chattree
from . import llmclient
from . import utils

# --------------------------------------------------------------------------------
# Bootup

datastore = chattree.Forest()

# --------------------------------------------------------------------------------
# Settings

# This conference information is automatically filled into all generated BibTeX entries.
# TODO: Add command-line options for conference info, to make this script into a properly reusable tool.
conference_slug = "ECCOMAS2024"  # BibTeX entry keys are generated as <slug>-<PDF_filename>
conference_year = "2024"
conference_booktitle = "The 9th European Congress on Computational Methods in Applied Sciences and Engineering (ECCOMAS Congress 2024)"
conference_note = "3--7 June 2024, Lisbon, Portugal"
conference_url = "https://eccomas2024.org/"

# --------------------------------------------------------------------------------
# Utilities

def print_progress(n_chunks, chunk, glyph="."):
    """Progress indicator while the LLM is processing. Callback for `llmclient.invoke`."""
    if (n_chunks == 1 or n_chunks % 10 == 0):  # in any message being written by the AI, print a progress symbol for the first chunk, and then again every 10 chunks.
        print(glyph, end="", file=sys.stderr)
        sys.stderr.flush()

def setup_prompts(settings: env) -> env:
    """Set up the task prompts for the LLM, writing them to `settings.prompts` (dict).

    Returns `settings` for convenience.
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
    prompt_get_authors = dedent("""Who are the authors of the abstract?

    This will be used for a bibliography. We need only the author names: NO institutions, NO street addresses, NO email addresses.

    Please reply only with the line(s) containing the author names, COPIED AS-IS from the original abstract.

    Just copy that line (or those lines) into the output.

    IMPORTANT: List the authors IN THE SAME ORDER as they appear in the original abstract.
    """)

    prompt_drop_author_affiliations = dedent("""Below is a list of author names for a bibliography:

    {author_names}

    **Notes**

    - The list is automatically extracted raw data.
    - Author names may have affiliation markings such as '*', '1', '2', or similar. Sometimes, the same author may have several affiliations.
    - The list is comma-separated. It may use the word "and" to separate the last author.
    - Periods are used ONLY for abbreviation, NOT as a separator.
    - An author may have several first or middle names. They might or might not be abbreviated.
    - An author name may be followed by a parenthetical remark. Treat it as part of the author name.
    - Sometimes there is just one author.

    **Task**

    Please remove all affiliation markings.

    Make NO other changes.

    Keep accented characters as-is.

    Keep names in the same order as in the original list.

    IMPORTANT: Reply ONLY with the cleaned-up list.
    """)

    prompt_reformat_author_separators = dedent("""Below is a list of author names for a bibliography:

    {author_names}

    **Notes**

    - In the original list, author names are separated by a comma, or by the word "and".
    - Periods are used ONLY for abbreviation, NOT as a separator.
    - An author may have several first or middle names. They might or might not be abbreviated.
    - An author name may be followed by a parenthetical remark. Treat it as part of the author name.
    - A last name may look like a first name. This is normal.
    - A name may look like an English word. This is normal.
    - Sometimes there is just one author.

    **Task**

    Please reformat the list, separating ALL authors using the word "and". Do NOT use a comma or a newline to separate authors.

    Make NO other changes.

    Keep accented characters as-is.

    IMPORTANT: Reply ONLY with the reformatted list.
    """)

    def extract_authors(uid: str, text: str) -> str:
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
        prompt_check_authorlist = dedent("""Attached is an ORIGINAL INPUT text.

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
        root_node_id = llmclient.factory_reset_chat_datastore(datastore, settings)

        # # Sanity-check for presence of author list, for logging a warning if the LLM thinks the author list is missing.
        # # This doesn't work well with an 8B model, even with majority voting ( see Wang et al., 2023 https://arxiv.org/abs/2203.11171 ).
        # # So let's just skip this, and use a heuristic check on the final result (whether the extracted names are present in the original text).
        # T = settings.request_data["temperature"]
        # settings.request_data["temperature"] = 0.3
        # llm_outputs = []
        # answers = collections.Counter()
        # for _ in range(3):
        #     print(_ + 1, end="", file=sys.stderr)
        #     request_node_id = datastore.create_node(llmclient.create_chat_message(settings,
        #                                                                           role="user",
        #                                                                           message=f"{prompt_check_authorlist}\n\n{text}"),
        #                                             parent_id=root_node_id)
        #     history = datastore.linearize_up(request_node_id)
        #     out = llmclient.invoke(settings, history, progress_callback=partial(print_progress, glyph="*"))
        #     out.data["content"] = llmclient.scrub(settings, out.data["content"], thoughts_mode="discard", add_ai_role_name=False)
        #     has_author_list = out.data["content"][-20:].split()[-1].strip().upper()  # Last word of output, in uppercase.
        #     has_author_list = has_author_list.translate(str.maketrans('', '', string.punctuation))  # Strip punctuation, in case of spurious formatting.
        #     answers[has_author_list] += 1
        #     llm_outputs.append(out.data["content"])
        # settings.request_data["temperature"] = T
        # votes = answers.most_common()  # [(item0, count0), ...]
        #
        # logger.info("Detection reports follow.")
        # for llm_output in llm_outputs:
        #     logger.info(llm_output)
        #
        # if not any(count > 1 for answer, count in votes):  # no majority?
        #     logger.info(f"Input file '{uid}': LLM could not detect whether there is an author list. Votes: {votes}")
        # else:
        #     has_author_list = votes[0][0]
        #     if has_author_list in ("YES", "NO"):
        #         if has_author_list == "NO":
        #             logger.info(f"Input file '{uid}': LLM says the author list is missing; manual check recommended.")
        #     else:
        #         logger.info(f"Input file '{uid}': LLM returned unknown author list detection result '{has_author_list}', should be 'YES' or 'NO'; manual check recommended.")

        request_node_id = datastore.create_node(llmclient.create_chat_message(settings,
                                                                              role="user",
                                                                              message=f"{prompt_get_authors}\n-----\n\n{text}"),
                                                parent_id=root_node_id)
        history = datastore.linearize_up(request_node_id)
        out = llmclient.invoke(settings, history, progress_callback=partial(print_progress, glyph="A"))
        out.data["content"] = llmclient.scrub(settings, out.data["content"], thoughts_mode="discard", add_ai_role_name=False)

        logger.debug(f"\n        extracted : {out.data}")

        # Usually the output is correct, but sometimes:
        #   - Some authors may be missing from the list
        #   - The list may have additional hallucinated authors
        #   - The list may use commas instead of the word "and"
        # so we perform some post-processing.

        request_node_id = datastore.create_node(llmclient.create_chat_message(settings,
                                                                              role="user",
                                                                              message=prompt_drop_author_affiliations.format(author_names=out.data["content"])),
                                                parent_id=root_node_id)
        history = datastore.linearize_up(request_node_id)
        out = llmclient.invoke(settings, history, progress_callback=partial(print_progress, glyph="a"))
        out.data["content"] = llmclient.scrub(settings, out.data["content"], thoughts_mode="discard", add_ai_role_name=False)

        logger.debug(f"\n        LLM pass 1: {out.data}")

        request_node_id = datastore.create_node(llmclient.create_chat_message(settings,
                                                                              role="user",
                                                                              message=prompt_reformat_author_separators.format(author_names=out.data["content"])),
                                                parent_id=root_node_id)
        history = datastore.linearize_up(request_node_id)
        out = llmclient.invoke(settings, history, progress_callback=partial(print_progress, glyph="."))
        out.data["content"] = llmclient.scrub(settings, out.data["content"], thoughts_mode="discard", add_ai_role_name=False)

        logger.debug(f"\n        LLM pass 2: {out.data}")

        if out.data["content"].endswith("and"):  # Remove spurious "and" with one author. Can happen especially if, in the original abstract, a comma follows the single author name.
            out.data["content"] = out.data["content"][:-3]

        authors = out.data["content"].strip()  # Final result from LLM. Remove extra whitespace, just in case.

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
            logger.warning(f"Input file '{uid}': Extractor returned one-component author name{plural_s}; manual check recommended: {possibly_broken_names}")

        # No author should be listed more than once.
        authors_counter = collections.Counter(authors_list)  # TODO: I hope `Counter` preserves insertion order?
        duplicate_names = [author for author, count in authors_counter.items() if count > 1]
        if len(duplicate_names):
            plural_s = "s" if len(duplicate_names) > 1 else ""
            logger.warning(f"Input file '{uid}': Extractor returned duplicate author name{plural_s}; de-duplicated, but manual check recommended: {duplicate_names}")

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
                            logger.warning(f"Input file '{uid}': Possibly broken format in processed author list; manual check recommended: '{authors}'")
                            format_warning_logged = True
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
                        logger.warning(f"Input file '{uid}': Possible LLM error with one or more spurious names in processed author list; manual check recommended: '{authors}'")
                        llm_warning_logged = True
                        break

        return authors

    prompt_get_title = dedent("""What is the title of the abstract?

    Please reply only with the title.

    Use plain text.

    Do NOT place quotation marks around the title.
    """)

    def extract_title(uid: str, text: str) -> str:
        """Extract the title from the fulltext of a conference abstract.

        `uid`: input file identifier, for error messages
        `text`: the full text

        Returns the title.
        """
        text = strip_postamble(text)

        root_node_id = llmclient.factory_reset_chat_datastore(datastore, settings)
        request_node_id = datastore.create_node(llmclient.create_chat_message(settings,
                                                                              role="user",
                                                                              message=f"{prompt_get_title}\n-----\n\n{text}"),
                                                parent_id=root_node_id)
        history = datastore.linearize_up(request_node_id)
        out = llmclient.invoke(settings, history, progress_callback=partial(print_progress, glyph="T"))
        out.data["content"] = llmclient.scrub(settings, out.data["content"], thoughts_mode="discard", add_ai_role_name=False)

        logger.debug(f"\n        original : {out.data}")

        title = out.data["content"].strip()

        # Strip spurious period
        while title[-1] == ".":
            title = title[:-1]

        # Strip spurious quotation marks (they may still occasionally happen even though we instruct the LLM not to emit them)
        while (title.startswith('"') and title.endswith('"')) or (title.startswith("'") and title.endswith("'")):
            title = title[1:-1]

        # Strip spurious period, again (inside quotation marks)
        while title[-1] == ".":
            title = title[:-1]

        logger.debug(f"\n        formatted: {title}")

        return title

    prompt_get_keywords = dedent("""What are the keywords, as given in the abstract?

    This will be used for a bibliography. We need only the keywords, COPIED AS-IS from the original abstract.

    You can find the keywords on a separate line that starts with "Keywords:" or "Key words:".

    Please reply only with a comma-separated list of keywords.

    Use plain text, no formatting.

    IMPORTANT: Only copy the existing list of keywords from the abstract; do NOT add your own keywords.

    IMPORTANT: The list of keywords in the original abstract may end abruptly. This is fine. Do NOT add additional keywords from the main text.
    """)

    def extract_keywords(uid: str, text: str) -> str:
        """Extract the keywords from the fulltext of a conference abstract.

        `uid`: input file identifier, for error messages
        `text`: the full text

        Returns the comma-separated keywords as a string.
        """
        text = strip_postamble(text)

        # Sanity check that the abstract has keywords before we run the LLM to extract them.
        match = kws_pattern.search(text)
        if match is None:
            logger.warning(f"Input file '{uid}': No keywords provided in original input, skipping keyword extraction.")
            return None  # No keywords provided

        root_node_id = llmclient.factory_reset_chat_datastore(datastore, settings)
        request_node_id = datastore.create_node(llmclient.create_chat_message(settings,
                                                                              role="user",
                                                                              message=f"{prompt_get_keywords}\n-----\n\n{text}"),
                                                parent_id=root_node_id)
        history = datastore.linearize_up(request_node_id)
        out = llmclient.invoke(settings, history, progress_callback=partial(print_progress, glyph="K"))
        out.data["content"] = llmclient.scrub(settings, out.data["content"], thoughts_mode="discard", add_ai_role_name=False)

        logger.debug(f"\n        original : {out.data}")

        # Remove spurious heading
        for heading in ("Keywords:", "KEYWORDS:", "Key words:", "Key Words:", "KEY WORDS:"):
            if out.data["content"].startswith(heading):
                out.data["content"] = out.data["content"][len(heading):]

        # Sanity-check the LLM output.
        #
        # Initial list of keywords.
        keywords = out.data["content"].strip()

        # Strip spurious period
        while keywords[-1] == ".":
            keywords = keywords[:-1]

        keywords_list = [keyword.strip() for keyword in keywords.split(",")]

        # No keyword should be listed more than once.
        keywords_counter = collections.Counter(keywords_list)  # TODO: I hope `Counter` preserves insertion order?
        duplicate_keywords = [author for author, count in keywords_counter.items() if count > 1]
        if len(duplicate_keywords):
            plural_s = "s" if len(duplicate_keywords) > 1 else ""
            logger.warning(f"Input file '{uid}': Extractor returned duplicate keyword{plural_s}; de-duplicated, but manual check recommended: {duplicate_keywords}")
        keywords = ", ".join(keywords_counter.keys())  # de-duplicate

        # TODO: Other sanity checks.

        logger.debug(f"\n        formatted: {keywords}")

        return keywords

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

    def extract_abstract(uid: str, text: str) -> str:
        """Extract the main text from the fulltext of a conference abstract.

        `uid`: input file identifier, for error messages
        `text`: the full text

        Returns the main text of the conference abstract.
        """
        text = strip_postamble(text)

        root_node_id = llmclient.factory_reset_chat_datastore(datastore, settings)
        request_node_id = datastore.create_node(llmclient.create_chat_message(settings,
                                                                              role="user",
                                                                              message=f"{prompt_get_abstract}\n-----\n\n{text}"),
                                                parent_id=root_node_id)
        history = datastore.linearize_up(request_node_id)
        out = llmclient.invoke(settings, history, progress_callback=partial(print_progress, glyph="."))
        out.data["content"] = llmclient.scrub(settings, out.data["content"], thoughts_mode="discard", add_ai_role_name=False)

        abstract = out.data["content"].strip()

        return abstract

    # {bibtex_fieldname: (kind, thing, progress_symbol_for_llm)}
    #
    # `kind`: one of "literal", "prompt", "function"
    #    "literal": Inject the exact given text `thing` (str) into the field.
    #    "prompt": Treat `thing` (str) as a prompt for the LLM. Send in the prompt and the complete text content of the PDF. Use `progress_symbol` to indicate progress. Inject the output into the field.
    #    "function": Call `thing` (callable) with arguments `uid, fulltext`. Inject the return value into the field. Except, if the return value is `None`, omit that field from the BibTeX.
    settings.prompts = {
                        "author": ("function", extract_authors, None),
                        "year": ("literal", conference_year, None),
                        "title": ("function", extract_title, None),
                        "booktitle": ("literal", conference_booktitle, None),
                        "note": ("literal", conference_note, None),
                        "url": ("literal", conference_url, None),
                        "keywords": ("function", extract_keywords, None),
                        "abstract": ("function", extract_abstract, None),
                       }
    return settings


# --------------------------------------------------------------------------------
# Processing logic

def listpdf(path: str) -> List[str]:
    """Return a list of all PDF files under `path`, recursively."""
    return list(sorted(filename for filename in os.listdir(path) if filename.endswith(".pdf")))

def process_abstracts(paths: List[str], opts: argparse.Namespace) -> None:
    """Process all PDFs under `paths`, recursively.

    Each PDF is assumed to contain a conference abstract (the final file sent to the conference organizers in PDF format), and nothing else.
    """
    # Connect to the LLM
    try:
        settings = llmclient.setup(backend_url=opts.backend_url)  # If this succeeds, then we know the backend is alive.
    except Exception as exc:
        msg = f"Failed to connect to LLM backend, reason {type(exc)}: {exc}"
        logger.error(msg)
        raise RuntimeError(msg)
    settings = setup_prompts(settings)
    logger.info(f"Connected to {opts.backend_url}")
    # logger.info("    available models:")
    # response = requests.get(f"{opts.backend_url}/v1/internal/model/list",
    #                         headers=headers,
    #                         verify=False)
    # payload = response.json()
    # for model_name in sorted(payload["model_names"], key=lambda s: s.lower()):
    #     logger.info(f"        {model_name}")
    logger.info(f"    model: {settings.model}")
    logger.info(f"    character: {settings.char} [defined in this client]")

    # Process the PDFs
    #
    # bibtex_entries = []
    uid = 0
    try:
        for path in paths:
            logger.info(f"Processing directory \"{path}\"...")
            # results = []
            filenames = listpdf(path)
            est = ETAEstimator(total=len(filenames), keep_last=10)
            for idx, filename in enumerate(filenames):
                fullpath = os.path.join(path, filename)
                uid = os.path.splitext(os.path.basename(fullpath))[0]  # "/foo/blah.pdf" -> "blah"
                logger.info(f"{fullpath} [{idx + 1} out of {len(filenames)}, {est.formatted_eta}]")

                # Extract the text content from the PDF.
                #
                # Since we'll be using an LLM for the processing step, it doesn't matter if the extraction isn't perfect
                # (e.g. the LLM will clean the author names if they have affiliation symbols or such).
                #
                cmd = ['pdftotext',
                       fullpath,
                       '-']  # output to stdout
                try:
                    completed = subprocess.run(cmd, check=True,
                                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except subprocess.CalledProcessError as err:
                    logger.error(f"When processing {fullpath}: subprocess returned non-zero exit status")
                    traceback.print_exc()
                    logger.error(err.stderr.decode("utf-8"))
                    raise

                text_from_pdf = completed.stdout.decode("utf-8")

                # Process one field at a time.
                #   - Fill whatever we can programmatically.
                #   - Use the LLM to populate the actual field content only, giving it one small task at a time.
                #
                entry_key = f"{conference_slug}-{uid}"
                bibtex_entry = io.StringIO()
                bibtex_entry.write(f"@incollection{{{entry_key},\n")
                with timer() as tim:
                    print("    ", end="", file=sys.stderr)  # Indent the progress indicator
                    sys.stderr.flush()
                    for field_key, (data_kind, data, progress_symbol) in settings.prompts.items():
                        if data_kind == "literal":
                            bibtex_entry.write(f"    {field_key} = {{{data}}},\n")
                        elif data_kind == "prompt":
                            # To keep things simple, we use a single-turn conversation for querying the LLM.
                            # Note this typically causes a full prompt rescan for every query.
                            root_node_id = llmclient.factory_reset_chat_datastore(datastore, settings)
                            request_node_id = datastore.create_node(llmclient.create_chat_message(settings,
                                                                                                  role="user",
                                                                                                  message=f"{data}\n-----\n\n{text_from_pdf}"),
                                                                    parent_id=root_node_id)
                            history = datastore.linearize_up(request_node_id)
                            out = llmclient.invoke(settings, history, progress_callback=partial(print_progress, glyph=progress_symbol))
                            out.data["content"] = llmclient.scrub(settings, out.data["content"], thoughts_mode="discard", add_ai_role_name=False)
                            bibtex_entry.write(f"    {field_key} = {{{out.data['content']}}},\n")
                        elif data_kind == "function":
                            function_output = data(uid, text_from_pdf)
                            if function_output is not None:  # A function can indicate "no data" by returning `None`. Inject the field only if data was returned.
                                bibtex_entry.write(f"    {field_key} = {{{function_output}}},\n")
                        else:
                            raise ValueError(f"Unknown data kind '{data_kind}'; please check your settings.")
                print(f"done in {tim.dt:0.2f}s", file=sys.stderr)
                bibtex_entry.write("}")
                bibtex_entry = bibtex_entry.getvalue()
                # bibtex_entries.append(bibtex_entry)

                print(bibtex_entry)
                print()  # one blank line after each entry
                sys.stdout.flush()

                # Move input file to done directory if specified (allows continuing later)
                if opts.output_dir is not None:
                    shutil.move(fullpath, os.path.join(opts.output_dir, os.path.basename(fullpath)))

                est.tick()
    finally:
        pass
        # bibtex_str = "\n\n".join(bibtex_entries)
        # print(bibtex_str)

# --------------------------------------------------------------------------------
# Main program

def main():
    parser = argparse.ArgumentParser(description="""Convert PDF conference abstracts into a BibTeX database. Works via pdftotext and an OpenAI compatible LLM.\nNo API key support as of yet, so please use a self-hosted LLM backend (e.g. AnythingLLM or oobabooga/text-generation-webui).""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(dest="backend_url", nargs="?", default=config.llm_backend_url, type=str, metavar="url", help="where to access the LLM API")
    parser.add_argument("-o", "--output-dir", dest="output_dir", default=None, type=str, metavar="output_dir", help="directory to move done files into (optional; allows easily continuing later)")
    parser.add_argument("-i", "--input-dir", dest="input_dir", default=None, type=str, metavar="input_dir", help="Input directory containing PDF file(s) to import (will be scanned recursively)")
    opts = parser.parse_args()

    if opts.input_dir is None:
        opts.input_dir = "."
    opts.input_dir = str(pathlib.Path(opts.input_dir).expanduser().resolve())
    logger.info(f"Processing PDF files from {opts.input_dir}.")

    if opts.output_dir is not None:
        opts.output_dir = str(pathlib.Path(opts.output_dir).expanduser().resolve())
        logger.info(f"Moving done files to {opts.output_dir}.")
        utils.create_directory(opts.output_dir)

    blacklist = []
    paths = []
    for root, dirs, files in os.walk(opts.input_dir):
        paths.append(root)
        for x in blacklist:
            if x in dirs:
                dirs.remove(x)
    paths = list(uniqify(str(pathlib.Path(p).expanduser().resolve()) for p in paths))

    # Avoid recursing into output directory
    if opts.output_dir is not None:
        paths = [p for p in paths if not p.startswith(opts.output_dir)]

    process_abstracts(sorted(paths), opts)

if __name__ == '__main__':
    main()
