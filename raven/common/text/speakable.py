"""Does a piece of text have anything for a speech synthesizer to say?

A TTS engine needs at least one *pronounceable* character. Markdown leftovers do not
qualify, and they reach the synthesizer more often than one would expect: an answer
ending `...to the naked eye:\n*` splits into a final "sentence" consisting of a lone
asterisk. That survives Markdown stripping — an unpaired `*` is not emphasis, so a
Markdown-to-plaintext pass correctly leaves it as literal text — and arrives at the
synthesizer as a sentence with no phonemes at all.

Kokoro answers such input with zero audio segments. Encoding zero segments is handled
gracefully now, but synthesizing-then-discarding still costs a network round trip and
a GPU pass per artifact, and a subtitle line reading `*` would be shown for a sentence
that is never spoken. The cheap check belongs at the front of the pipeline.

The test is Unicode-aware alphanumerics, which accepts every script the speech and
translation paths can encounter (Latin with diacritics, Cyrillic, CJK, digits) while
rejecting punctuation-and-symbol-only fragments such as `*`, `**`, `---`, `?!`, `-->`.
It is deliberately blunt: a stray `&` that a phonemizer might have voiced as "and" is
dropped too, which is the right trade when such a fragment is a formatting artifact
in essentially every case it occurs.
"""

__all__ = ["is_speakable"]


def is_speakable(text: str) -> bool:
    """Return whether `text` contains anything a speech synthesizer could pronounce.

    True when at least one character is alphanumeric in the Unicode sense. Blank text,
    whitespace, and punctuation-or-symbol-only fragments are not speakable.
    """
    return any(char.isalnum() for char in text)
