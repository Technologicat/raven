"""Shortening a string to fit, and marking that it was shortened.

Two budgets, because a caller has one or the other and never both. A **character count** is what text that
is not being laid out has — a log line, a label whose font nobody knows, a field with a documented maximum.
A **measured width** is what a GUI label in a proportional font has, where the number of characters says
very little about the room needed: "WWW" and "iii" are the same length and nowhere near the same width.

Both mark the cut with a single ellipsis character, and both return text that already fits exactly as it
is. What differs between them is only where the cut goes — the end by default, the middle on request.
"""

__all__ = ["ellipsize", "ellipsize_to_width",
           "longest_prefix_that_fits"]

from typing import Callable

#: What marks a string as having been cut. One character wide, so that even a one-character budget has room
#: to say that something was dropped.
_ELLIPSIS = "…"


def ellipsize(text: str, max_chars: int, *, middle: bool = False) -> str:
    """Return `text` at no more than `max_chars` characters, marked with an ellipsis if it had to be cut.

    `text`: What to shorten. Returned unchanged if it already fits.
    `max_chars`: The budget, the ellipsis counted against it. Zero or less gives the empty string, there
                 being no room even to report that something was dropped.
    `middle`: Take the cut out of the middle, keeping both ends: `"quarterly_re…port_2026.pdf"`.

              For a name whose two ends are the informative parts — the topic at the front, the file type
              at the back. Cutting the tail off a filename throws away the extension, which is exactly what
              tells a reader whether the thing they are about to delete is a paper or a slide deck.

    For a GUI label in a proportional font, `ellipsize_to_width` instead.
    """
    if len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""
    keep = max_chars - 1  # the ellipsis costs one character
    # Whitespace is taken back from each cut edge so that the result cannot read as "the word …", which
    # looks like a writer trailing off rather than like text that continues. It can only shorten the
    # result, so the budget still holds.
    if not middle:
        return f"{text[:keep].rstrip()}{_ELLIPSIS}"
    head = keep - keep // 2
    return f"{text[:head].rstrip()}{_ELLIPSIS}{text[len(text) - keep // 2:].lstrip()}"


def ellipsize_to_width(text: str,
                       max_width: float,
                       width_of: Callable[[str], float]) -> str:
    """Return `text` at no wider than `max_width`, marked with an ellipsis if it had to be cut.

    `ellipsize`, with the budget measured instead of counted — which is what a GUI label in a proportional
    font actually has to spend.

    `text`: What to shorten. Returned unchanged if it already fits.
    `max_width`: The budget, in whatever unit `width_of` reports.
    `width_of`: Measures a string in that same unit. A toolkit's text-measuring call, bound to the font
                the label will be drawn in.

    End-cut only; there is no `middle` here, since eliding a middle by measurement needs two searches
    rather than one and no caller has wanted it.

    A budget too narrow for even one character plus the ellipsis still yields one character plus the
    ellipsis, rather than nothing — `longest_prefix_that_fits` says why that is the honest answer.
    """
    if width_of(text) <= max_width:
        return text
    return f"{longest_prefix_that_fits(text, max_width - width_of(_ELLIPSIS), width_of)}{_ELLIPSIS}"


def longest_prefix_that_fits(text: str,
                             max_width: float,
                             width_of: Callable[[str], float]) -> str:
    """Return the longest prefix of `text` no wider than `max_width`, or its first character.

    `text`: What to take a prefix of.
    `max_width`: The budget, in whatever unit `width_of` reports.
    `width_of`: Measures a string in that same unit.

    At least one character, always, so that a caller stepping through a word cannot fail to make progress
    and spin. A single character too wide for the box is a box too narrow to draw text in at all, and the
    honest answer there is one character overflowing rather than an empty label.

    Binary search rather than a walk: width grows with length, and the alternative measures once per
    character dropped — which for a long identifier in a narrow box is dozens of measurements per box, on
    the render thread.
    """
    if width_of(text) <= max_width:
        return text
    low, high = 1, len(text)  # `low` always fits by fiat; `high` is known not to
    while low < high - 1:
        middle = (low + high) // 2
        if width_of(text[:middle]) <= max_width:
            low = middle
        else:
            high = middle
    return text[:low]
