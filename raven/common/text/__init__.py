"""Text utilities for Raven.

Currently:

  - `normalize`, defensive normalization of untrusted retrieved text (strips
    invisible-injection glyphs, applies Unicode NFC). Shared by webfetch,
    websearch-result handling, and future retrieved-text consumers.
  - `speakable`, whether a fragment has any content a TTS engine could pronounce.
    Used to drop Markdown artifacts before they reach the speech pipeline.
  - `boilerplate`, removing a publisher's rights notice from the end of an
    abstract. Used by the Visualizer's BibTeX importer, and by anything else
    reading a database-exported abstract as prose.
  - `plural`, agreeing a noun with a count the code already has, so that no
    message has to say `1 file(s)`.
  - `shorten`, cutting a string down to a budget of characters or of measured
    width, marking the cut with an ellipsis. Used wherever a name or a title has
    to fit a label.
  - `window`, the recent text to detect an emotion from while text arrives a
    piece at a time. Used for the avatar's expression, both while a reply
    streams in and while it is spoken.
  - `subtitles`, cutting a long sentence into subtitle cards that fit, and timing
    each card against the speech. Used by the avatar's subtitler.
  - `entities`, decoding HTML character entities. Used by the BibTeX tools and
    by `common.utils.unicodize_basic_markup`.

Submodules are independently importable; this package also re-exports the public
API, so callers can `from raven.common import text` and use `text.normalize(...)`.
`entities` is the exception: `decode` and `resolve` say what they do only beside
the module name, so it is imported as a module.
"""

from .normalize import normalize  # noqa: F401 -- re-export submodule public API
from .speakable import is_speakable  # noqa: F401 -- re-export submodule public API
from .boilerplate import find_rights_notice, split_rights_notice, strip_boilerplate  # noqa: F401 -- re-export submodule public API
from .plural import plural_s  # noqa: F401 -- re-export submodule public API
from .shorten import ellipsize, ellipsize_to_width, longest_prefix_that_fits, longest_suffix_that_fits  # noqa: F401 -- re-export submodule public API
from .window import EmotionWindow  # noqa: F401 -- re-export submodule public API
from .subtitles import Card, wrap_lines, split_into_cards, card_times_from_words, card_times_proportional  # noqa: F401 -- re-export submodule public API

__all__ = ["normalize",
           "is_speakable",

           "find_rights_notice", "split_rights_notice", "strip_boilerplate",

           "plural_s",

           "ellipsize", "ellipsize_to_width",
           "longest_prefix_that_fits", "longest_suffix_that_fits",

           "EmotionWindow",

           "Card", "wrap_lines", "split_into_cards",
           "card_times_from_words", "card_times_proportional"]
