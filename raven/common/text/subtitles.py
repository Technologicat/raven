"""Cutting a long sentence into subtitle cards, and timing each card against the speech.

A sentence shown as one subtitle can run to many lines and cover much of the picture. Subtitling splits it
across several cards instead, each up while its part is being spoken.

`split_into_cards` decides where the cuts go, by measured width, so that each card fits in a number of lines
at the width it will be drawn at. Two ways to time the cards, depending on whether the text on them is the
text being spoken:

  - `card_times_from_words`, for captions: the card text *is* the spoken text, so each card goes up when the
    speech synthesizer says its first word.
  - `card_times_proportional`, for a translation: its words have no timestamps, so a card goes up at the
    same fraction of the speech as it starts at in the text.

Nothing here knows about a toolkit or a speech engine. Widths come from a `width_of` function, as in
`raven.common.text.shorten`, and word timings arrive as `(word, start_time, end_time)` tuples.
"""

__all__ = ["Card",
           "wrap_lines",
           "split_into_cards",
           "card_times_from_words", "card_times_proportional"]

import re
from typing import Callable, NamedTuple, Sequence

# Punctuation after which a card may end, as a subtitler would cut: a clause boundary rather than mid-phrase.
# A sentence's own end is not in here, a card never containing more than one sentence.
_BREAK_AFTER = re.compile(r"[,;:]$|[—–]$")
_DASH = re.compile(r"^[—–]$")

_WORD = re.compile(r"\S+")


class Card(NamedTuple):
    """One subtitle card: its text, and where that text starts in the sentence it was cut from."""
    offset: int
    text: str


def wrap_lines(text: str, max_width: float, width_of: Callable[[str], float]) -> list[str]:
    """Return `text` word-wrapped into lines no wider than `max_width`.

    `text`: What to wrap. Runs of whitespace count as one space.
    `max_width`: The line width, in whatever unit `width_of` reports.
    `width_of`: Measures a string in that same unit. A toolkit's text-measuring call, bound to the font
                the text will be drawn in.

    Greedy, breaking only at spaces, which is how a GUI label wraps. A word wider than a whole line gets a
    line of its own and overflows it, there being nowhere to break it.
    """
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if current and width_of(candidate) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def split_into_cards(text: str,
                     max_width: float,
                     width_of: Callable[[str], float],
                     *,
                     max_lines: int = 2) -> list[Card]:
    """Cut `text` into cards that each wrap to at most `max_lines` lines at `max_width`.

    `text`: Usually one sentence.
    `max_width`: The width the cards will be drawn at, in whatever unit `width_of` reports.
    `width_of`: Measures a string in that same unit.
    `max_lines`: How many lines a card may wrap to.

    Text that already fits comes back as one card. Otherwise each card takes as many words as fit, and then
    gives back the words after the last clause boundary in it (a comma, a semicolon, a colon or a dash),
    provided that leaves the card at least half full. So a cut lands between clauses where one is near, and
    mid-clause only where none is. A card never ends on a single dash, which would read as a dangling
    break, and a word too wide for even one card goes alone on a card of its own.

    Each card's `offset` is where its text starts in `text`, which is what a caller times it by.
    """
    words = [(match.start(), match.end()) for match in _WORD.finditer(text)]
    if not words:
        return []

    def fits(first: int, last: int) -> bool:
        return len(wrap_lines(text[words[first][0]:words[last][1]], max_width, width_of)) <= max_lines

    cards: list[Card] = []
    first = 0
    while first < len(words):
        last = first  # a card holds at least one word, whatever its width
        while last + 1 < len(words) and fits(first, last + 1):
            last += 1
        if last + 1 < len(words):  # more to come, so the card may give some back
            half_full = (words[first][0] + words[last][1]) / 2
            for candidate in range(last, first, -1):
                word = text[words[candidate][0]:words[candidate][1]]
                if words[candidate][1] < half_full:
                    break
                if _BREAK_AFTER.search(word) and not _DASH.match(word):
                    last = candidate
                    break
                if _DASH.match(word) and candidate > first:  # a standalone dash opens the next card instead
                    last = candidate - 1
                    break
        start, end = words[first][0], words[last][1]
        cards.append(Card(offset=start, text=text[start:end]))
        first = last + 1
    return cards


def card_times_from_words(text: str,
                          cards: Sequence[Card],
                          words: Sequence[tuple[str, float | None, float | None]]) -> list[float]:
    """Return when each card goes up, in seconds from the start of the speech, for cards cut from `text`.

    `text`: What the cards were cut from, and what was spoken.
    `cards`: From `split_into_cards(text, ...)`.
    `words`: The speech synthesizer's word timings, `(word, start_time, end_time)`, in spoken order. A
             timing may be `None` where the synthesizer gives none.

    A card goes up when its first word is spoken. Each word is found in `text` by searching forward from the
    last one found, so a word the synthesizer spells differently from the text, or drops, is simply not
    found; a card with no word found in it falls back to `card_times_proportional`'s answer. The first card
    goes up at zero, with the speech, and the times never run backwards.
    """
    found: list[tuple[int, float]] = []  # (offset in text, start time)
    cursor = 0
    for word, start_time, _end_time in words:
        if start_time is None or not word:
            continue
        index = text.find(word, cursor)
        if index < 0:
            continue
        found.append((index, start_time))
        cursor = index + len(word)

    fallback = card_times_proportional(len(text), cards, words)
    times: list[float] = []
    for card_index, card in enumerate(cards):
        if card_index == 0:
            time = 0.0
        else:
            card_end = card.offset + len(card.text)
            in_card = [start_time for offset, start_time in found if card.offset <= offset < card_end]
            time = in_card[0] if in_card else fallback[card_index]
        times.append(max(time, times[-1]) if times else time)
    return times


def card_times_proportional(text_length: int,
                            cards: Sequence[Card],
                            words: Sequence[tuple[str, float | None, float | None]]) -> list[float]:
    """Return when each card goes up, in seconds from the start of the speech, by position alone.

    `text_length`: Length of the text the cards were cut from.
    `cards`: From `split_into_cards`.
    `words`: The speech synthesizer's word timings for what was *spoken*, `(word, start_time, end_time)`;
             only the first start and the last end are used.

    For cards whose text is not what was spoken, such as a translation: a card starting some fraction of the
    way into its text goes up the same fraction of the way through the speech. The first card goes up at
    zero, with the speech.
    """
    starts = [start for _word, start, _end in words if start is not None]
    ends = [end for _word, _start, end in words if end is not None]
    if not starts or not ends or text_length <= 0:
        return [0.0] * len(cards)
    speech_start, speech_end = min(starts), max(ends)
    return [0.0 if index == 0 else speech_start + (card.offset / text_length) * (speech_end - speech_start)
            for index, card in enumerate(cards)]
