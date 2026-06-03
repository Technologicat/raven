"""Text utilities for Raven.

Currently: `normalize`, defensive normalization of untrusted retrieved text
(strips invisible-injection glyphs, applies Unicode NFC). Shared by webfetch,
websearch-result handling, and future retrieved-text consumers.

Submodules are independently importable; this package also re-exports the public
API, so callers can `from raven.common import text` and use `text.normalize(...)`.
"""

from .normalize import normalize  # noqa: F401 -- re-export submodule public API

__all__ = ["normalize"]
