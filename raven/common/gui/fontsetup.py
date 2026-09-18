"""Font loading related DPG GUI utilities.

DPG builds a font atlas covering whatever characters the app actually draws, so a font loaded here
carries every codepoint the TTF has glyphs for — Greek symbols and the math-related special
characters Raven's BibTeX importer introduces from its LaTeX and HTML conversions included. Where a
character comes out as a box, the font is missing the glyph; picking a different TTF is the fix.
"""

__all__ = ["FONT_VARIANTS_BY_FACE", "load_font_ladders",  # the faces a graph label can be drawn in

           "markdown_add_font_callback"]

import logging
logger = logging.getLogger(__name__)

from typing import Sequence, Union

import dearpygui.dearpygui as dpg

from . import utils as guiutils

# `(bold, italic)` -> the TTF variant that face is in, as `get_font_path` spells one.
FONT_VARIANTS_BY_FACE = {(False, False): "Regular",
                         (True, False): "Bold",
                         (False, True): "Italic",
                         (True, True): "BoldItalic"}


def load_font_ladders(themes_and_fonts,
                      sizes: Sequence[Union[int, float]],
                      font_basename: str = "OpenSans") -> dict:
    """Load `font_basename` in all four faces at each of `sizes`, and return them keyed by `(bold, italic)`.

    The shape `XDotWidget` takes as `graph_text_fonts`: a ladder of `(size, font id)` per face, from which
    its renderer picks the face a pen asks for at the size nearest the one it is drawing.

    `sizes` is a ladder rather than a range because a font atlas is built per size, and the renderer scales
    between rungs. Loading all four faces costs little: since DPG 2.3 a face is rasterized as its glyphs are
    drawn rather than over the whole range at load time, measured at 0.1–0.2 ms a face at any size up to
    1024 px (`investigations/graph-font-atlas/`).

    `themes_and_fonts`: from `raven.common.gui.utils.bootup`; the fonts are cached there.
    """
    return {face: [(size, guiutils.load_extra_font(themes_and_fonts, size, font_basename, variant)[1])
                   for size in sizes]
            for face, variant in FONT_VARIANTS_BY_FACE.items()}

def markdown_add_font_callback(file, size: int | float, parent=0, **kwargs) -> int:  # IMPORTANT: parameter names as in `dpg_markdown`, arguments are sent in by name.
    """Callback for `dpg_markdown` to load a font. Called whenever a new font size or family is needed."""
    if not isinstance(size, (int, float)):
        raise ValueError(f"markdown_add_font_callback: `size`: expected `int` or `float`, got `{type(size)}` with value `{size}`")
    return dpg.add_font(file, size, parent=parent, **kwargs)
