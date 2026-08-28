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

**Every candidate must sit within the last `TAIL_BUDGET` characters.** A rights notice goes at the end; a
discussion of rights may be anywhere. That is the one condition both tiers share.

Beyond it, the markers fall into two tiers, and the split is the whole design — because the hard case is
not an abstract that *has* a notice, it is an abstract *about copyright*, which a corpus on AI in
education has plenty of.

- **`UNMISTAKABLE` — not English.** The copyright sign and `(c) 2025`, in the spellings that survive
  markup conversion. Nobody writes these mid-sentence while arguing about licensing, so they are trusted
  anywhere in the window. This tier finds essentially every real notice.
- **`NOTICE_OPENER` — ordinary English, trusted only at a sentence boundary.** `All rights reserved` is a
  phrase a paper may quote and analyse; `copyright held by` is a clause a paper may argue about; a
  licence-grant clause is a sentence a paper may write about its own subject. Appended boilerplate opens
  its own sentence. A mention inside an argument does not, and that distinction is what tells
  `Copyright 2024, Society of Petroleum Engineers.` from `The Copyright 1976 settlement still governs
  derivative works.`

A bare `copyright` is in neither tier. An abstract may close on "copyright concerns, bias mitigation,
computational demands", and a bare-word match eats the end of it.

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

__all__ = ["TAIL_BUDGET", "UNMISTAKABLE", "NOTICE_OPENER", "LEADING_LABEL",
           "find_rights_notice", "split_rights_notice", "strip_boilerplate"]

import re

# How far from the end a notice may begin. A full Creative Commons grant -- the clause, the licence name,
# the URL, and the permissions sentence after it -- runs past 400 characters, and a budget cut to fit the
# *typical* notice silently misses the longest ones, which are the ones carrying the most junk. 600 still
# means "the end" for an abstract of any normal length, and the sentence-opening test below is what
# actually discriminates for the tier that needs discriminating.
TAIL_BUDGET = 600

# Markers that are not English. Nobody writes these in the middle of a sentence about copyright law, so
# they are trusted wherever they fall in the tail window. This tier does essentially all the work.
UNMISTAKABLE = re.compile(r"""(?ix)
      ©  |  \\copyright\b  |  &copy;  |  &\#169;      # the sign, and the spellings markup conversion leaves
    | \(\s*c\s*\)\s* \d{4}                            # (c) 2025
""")

# Markers that are ordinary English and therefore say nothing on their own. `All rights reserved` is a
# phrase a paper may quote and discuss; `copyright held by` is a clause a paper may argue about. These are
# trusted only when they *open a sentence*, which is what an appended notice does and what a mention
# inside an argument does not.
NOTICE_OPENER = re.compile(r"""(?ix)
      \bcopyright\s* (?: ©|\(c\)|\d{4}|held\s+by|by\s+the|[-\u2013\u2014]\s )
    | \ball\s+rights\s+reserved\b
    | \bthis\s+ (?:is\s+an?\s+)? (?:\w+\s+){0,3}? (?:work|article|content|review|publication|paper)\s+
      (?:is\s+)? (?:published|licensed|distributed|made\s+available)\s+ (?:to\s+you\s+)? under\b
    | \blicensee\s+ [A-Z]                             # "Licensee MDPI", "licensee CEDTECH"
""")

# Deliberately not a marker: `Creative Commons Attribution` on its own. It is a proper noun that starts
# sentences in prose -- "Creative Commons was found to be a popular licensing model" -- so even the
# sentence-opening test cannot save it, and an AI-in-education corpus contains papers about open
# licensing. Measured over a 6934-record export it was the sole evidence for 2 notices out of 1656, both
# of which the licence-grant clause above now reaches instead.

LEADING_LABEL = re.compile(r"^\s*(?:abstract|summary)\s*[:.\u2013\u2014-]\s+", re.IGNORECASE)

# What a sentence ends with: the stop, optionally a closing quote or bracket, and optionally a short
# bracketed aside before the next sentence starts. The aside is there because publishers put one exactly
# where it breaks a naive test -- `...within the domain. (CC BY-NC 4.0) This article is licensed to you
# under a ...` is one sentence ending and another beginning, with a licence tag wedged between them.
_SENTENCE_END = re.compile(r"""(?x)
    [.!?] ['"\u2019\u201d)\]]? \s*
    (?: (?P<aside> [(\[] [^()\[\]]{0,48} [)\]] ) \s* )?
    $""")


def _sentence_start_at(text: str, position: int) -> int | None:
    """Where the notice at `position` begins, or `None` if `position` sits mid-sentence.

    Usually `position` itself. It is earlier when a bracketed aside stands between the previous full stop
    and the notice, because the aside belongs to the boilerplate rather than to the prose — a dangling
    `(CC BY-NC 4.0)` left behind by cutting at the marker is exactly the text this reclaims.
    """
    before = text[:position]
    if not before.strip():
        return 0
    match = _SENTENCE_END.search(before)
    if match is None:
        return None
    return match.start("aside") if match.group("aside") else position


def find_rights_notice(text: str) -> int | None:
    """Return the offset where a publisher's rights notice begins in `text`, or `None` if there is none.

    Exposed separately from `strip_boilerplate` so a caller can report or inspect a notice rather than
    discard it — an audit trail wanting to say what was removed needs the text, not just the absence.
    """
    window_start = max(0, len(text) - TAIL_BUDGET)
    candidates = [match.start() for match in UNMISTAKABLE.finditer(text) if match.start() >= window_start]
    for match in NOTICE_OPENER.finditer(text):
        if match.start() < window_start:
            continue
        start = _sentence_start_at(text, match.start())
        if start is not None:
            candidates.append(start)
    return min(candidates) if candidates else None


def split_rights_notice(text: str) -> tuple[str, str | None]:
    """Split `text` into `(what it says, the publisher's rights notice)`, the second `None` if absent.

    Both halves come back, because the notice is **metadata in the wrong field rather than noise**: it
    says who holds the rights, which is worth keeping where a reader would look for it. A caller writing
    a bibliography back out should move it to a field of its own; one about to run text analysis wants
    only the first half, and `strip_boilerplate` is that call.

    A leading `Abstract:` label is removed from the first half. That one really is noise \u2014 no exporter
    means it as content, and nothing is lost by dropping it.
    """
    text = LEADING_LABEL.sub("", text.strip())
    notice_start = find_rights_notice(text)
    if notice_start is None:
        return text.strip(), None
    # What is left dangling at the cut is a *separator* \u2014 the comma or dash that joined the notice on.
    # A full stop is not one: it ends the abstract's own last sentence and was there before any publisher
    # appended anything, so it stays. Getting this backwards silently shortens every abstract in a corpus
    # by a character, which no aggregate statistic will show.
    body = re.sub(r"[\s,;:\u2013\u2014-]+$", "", text[:notice_start].rstrip())
    return body, text[notice_start:].strip() or None


def strip_boilerplate(text: str) -> str:
    """Return `text` without a leading `Abstract:` label or a trailing publisher's rights notice.

    Returns the text unchanged when it carries neither, which is the common case for an abstract that
    reached the file from somewhere other than a database export.

    `split_rights_notice` where the notice itself is wanted rather than discarded.
    """
    return split_rights_notice(text)[0]
