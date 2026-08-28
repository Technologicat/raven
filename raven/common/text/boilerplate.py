"""Remove a publisher's rights notice from the end of an abstract.

Bibliographic databases append their own boilerplate to the abstracts they export, and it is not part of
what the paper says:

    ... contributing to improved educational outcomes and pedagogical strategies. © The Author(s), under
    exclusive license to Springer Nature Singapore Pte Ltd. 2025.

The text is legally required where it appears and useless everywhere it then travels. It pollutes anything
that reads an abstract as prose: it is why "Elsevier" turns up in a word cloud, and — per
`nlptools.default_stopwords` — why it lemmatizes to "elsevi", the copyright line putting it where an
adjective would go. Comparing two copies of one abstract from two databases, it is the *only* difference
in the overwhelming majority of cases.

`strip_boilerplate` also removes a leading `Abstract:` label, which some exporters prepend and which is
never content either.

## How a notice is recognized

Two conditions, and both must hold. Neither is sufficient alone, which is the whole design.

**It must look like the opening of a notice, not a mention of the subject.** The alternatives are the
copyright sign in its several spellings, `copyright` followed by something that makes it a notice (a year,
a sign, `held by`), `all rights reserved`, a licence-grant clause, `licensee <Name>`, or a Creative
Commons attribution. Deliberately absent: a bare `copyright`. An abstract may *discuss* copyright — one
about AI-generated work closes on "copyright concerns, bias mitigation, computational demands" — and a
bare-word match eats the end of it.

**It must sit within the last `TAIL_BUDGET` characters.** A rights notice goes at the end; a discussion of
rights may be anywhere. Position is what separates them when the wording alone cannot, and it is the
condition that makes the first one safe to relax slightly.

Trailing punctuation is trimmed only after an actual cut, so an abstract nothing matched keeps its final
full stop.

## What this deliberately does not do

It does not attempt to *parse* the notice, or to record what it said. A caller wanting the licence should
read the record's own fields, where an exporter that knows it puts it. And it does not touch anything
before the tail window, so an abstract that opens by quoting a licence keeps it.

Escaped and entity spellings (`\\copyright`, `&copy;`, `&#169;`) are matched because
`common.utils.unicodize_basic_markup` leaves them intact — verified, not assumed — so text that has been
through markup conversion can still carry them. They are here for that reason rather than because any
corpus to hand uses them.
"""

__all__ = ["TAIL_BUDGET", "RIGHTS_NOTICE", "LEADING_LABEL",
           "find_rights_notice", "strip_boilerplate"]

import re

# How far from the end a notice may begin. Generous enough for the longest real ones -- a Creative Commons
# grant with its URL runs to about 400 characters -- and short enough that an abstract's own closing
# sentences are out of reach. The longest genuine notice measured in a 6934-record multi-database export
# was 397 characters.
TAIL_BUDGET = 400

RIGHTS_NOTICE = re.compile(r"""(?ix)
      ©  |  \\copyright\b  |  &copy;  |  &\#169;      # the sign, and the spellings markup conversion leaves
    | \(\s*c\s*\)\s* \d{4}                            # (c) 2025
    | \bcopyright\s* (?: ©|\(c\)|\d{4}|held\s+by|by\s+the|[-\u2013\u2014]\s )
    | \ball\s+rights\s+reserved\b
    | \bthis\s+ (?:work|article|content|review|publication)\s+ is\s+
      (?:published|licensed|distributed|made\s+available)\s+ under
    | \blicensee\s+ [A-Z]                             # "Licensee MDPI", "licensee CEDTECH"
    | \bcreative\s+commons\s+ attribution\b
""")

LEADING_LABEL = re.compile(r"^\s*(?:abstract|summary)\s*[:.\u2013\u2014-]\s+", re.IGNORECASE)


def find_rights_notice(text: str) -> int | None:
    """Return the offset where a publisher's rights notice begins in `text`, or `None` if there is none.

    Exposed separately from `strip_boilerplate` so a caller can report or inspect a notice rather than
    discard it — an audit trail wanting to say what was removed needs the text, not just the absence.
    """
    window_start = max(0, len(text) - TAIL_BUDGET)
    for match in RIGHTS_NOTICE.finditer(text):
        if match.start() >= window_start:
            return match.start()
    return None


def strip_boilerplate(text: str) -> str:
    """Return `text` without a leading `Abstract:` label or a trailing publisher's rights notice.

    Returns the text unchanged when it carries neither, which is the common case for an abstract that
    reached the file from somewhere other than a database export.
    """
    text = LEADING_LABEL.sub("", text.strip())
    notice_start = find_rights_notice(text)
    if notice_start is None:
        return text.strip()
    # What is left dangling at the cut is a *separator* \u2014 the comma or dash that joined the notice on.
    # A full stop is not one: it ends the abstract's own last sentence and was there before any publisher
    # appended anything, so it stays. Getting this backwards silently shortens every abstract in a corpus
    # by a character, which no aggregate statistic will show.
    return re.sub(r"[\s,;:\u2013\u2014-]+$", "", text[:notice_start].rstrip())
