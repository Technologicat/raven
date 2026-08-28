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

Submodules are independently importable; this package also re-exports the public
API, so callers can `from raven.common import text` and use `text.normalize(...)`.
"""

from .normalize import normalize  # noqa: F401 -- re-export submodule public API
from .speakable import is_speakable  # noqa: F401 -- re-export submodule public API
from .boilerplate import find_rights_notice, strip_boilerplate  # noqa: F401 -- re-export submodule public API

__all__ = ["normalize",
           "is_speakable",

           "find_rights_notice", "strip_boilerplate"]
