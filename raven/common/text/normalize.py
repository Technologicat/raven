"""Defensive normalization of untrusted retrieved text.

Untrusted text — web pages we fetch, search-result snippets, content pulled from
a RAG database — can carry characters that are invisible to a human reading the
rendered page but tokenize normally for an LLM. That gap is a prompt-injection
vector: hidden instructions smuggled past the human into the model's input.

`normalize` kills the high-priority cases with a cheap, deterministic pass:

- **Unicode tag characters (U+E0000–U+E007F).** The "ASCII smuggler": a full
  shadow ASCII alphabet in an invisible Unicode block, used to encode hidden
  instructions inside otherwise-innocent text. There are public PoCs (the
  embracethered.com ASCII-smuggler demo, the Google Jules invisible-injection
  writeup). Dropping the whole block is a one-line kill.
- **Other zero-width / invisible characters.** Zero-width space/joiner/non-joiner,
  word joiner, Mongolian vowel separator, ZWNBSP. All are injection-vector-shaped
  and almost never appear in legitimate scientific prose. A single *leading* BOM
  (U+FEFF) is preserved; embedded ZWNBSPs are stripped.
- **Control characters** other than TAB/LF/CR (C0, C1, and DEL). Same rationale.
- **Unicode NFC normalization.** Folds canonically-equivalent sequences into one
  form — standard hygiene that defangs lookalike-encoding tricks.

Deliberately *not* done (overactive — would break legitimate multilingual or code
content): full confusables/homoglyph detection, bidi-override stripping, and NFKC
compatibility folding. The simple zero-width + tag-character + NFC pass removes the
high-priority attack class with no false-positive risk to scientific prose.

This is a shared utility, not webfetch-private: any handler of untrusted retrieved
text (websearch results, MCP text content, future PDF ingestion) should run its
output through `normalize` before it reaches the model.
"""

__all__ = ["normalize"]

import unicodedata

# Codepoints deleted by `normalize`. See the module docstring for the threat model.
_TAG_CHARACTERS = range(0xE0000, 0xE0080)  # U+E0000–U+E007F: the ASCII-smuggler block
_ZERO_WIDTH = (0x200B,   # zero-width space
               0x200C,   # zero-width non-joiner
               0x200D,   # zero-width joiner
               0x2060,   # word joiner
               0x180E,   # Mongolian vowel separator
               0xFEFF)   # zero-width no-break space (ZWNBSP); a single leading BOM is exempted in `normalize`
_KEEP_CONTROLS = frozenset((0x09, 0x0A, 0x0D))  # TAB, LF, CR — the whitespace controls we keep

def _build_deletion_table() -> dict:
    """Build a `str.translate` table mapping every deleted codepoint to `None`."""
    table = {cp: None for cp in _TAG_CHARACTERS}
    for cp in _ZERO_WIDTH:
        table[cp] = None
    for cp in range(0x00, 0x20):  # C0 controls
        if cp not in _KEEP_CONTROLS:
            table[cp] = None
    table[0x7F] = None  # DEL
    for cp in range(0x80, 0xA0):  # C1 controls
        table[cp] = None
    return table

_DELETION_TABLE = _build_deletion_table()

def normalize(text: str) -> str:
    """Strip invisible-injection glyphs and control characters, then apply Unicode NFC.

    Idempotent: `normalize(normalize(text)) == normalize(text)`.

    A single leading BOM (U+FEFF) is preserved (it may legitimately mark the start
    of decoded file content); any other U+FEFF — including a second leading one — is
    treated as an embedded ZWNBSP and removed.
    """
    has_leading_bom = text.startswith("﻿")
    cleaned = text.translate(_DELETION_TABLE)
    if has_leading_bom:
        cleaned = f"﻿{cleaned}"
    return unicodedata.normalize("NFC", cleaned)
