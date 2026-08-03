"""Font loading related DPG GUI utilities.

DPG builds a font atlas covering whatever characters the app actually draws, so a font loaded here
carries every codepoint the TTF has glyphs for — Greek symbols and the math-related special
characters Raven's BibTeX importer introduces from its LaTeX and HTML conversions included. Where a
character comes out as a box, the font is missing the glyph; picking a different TTF is the fix.
"""

__all__ = ["markdown_add_font_callback"]

import logging
logger = logging.getLogger(__name__)

import dearpygui.dearpygui as dpg

def markdown_add_font_callback(file, size: int | float, parent=0, **kwargs) -> int:  # IMPORTANT: parameter names as in `dpg_markdown`, arguments are sent in by name.
    """Callback for `dpg_markdown` to load a font. Called whenever a new font size or family is needed."""
    if not isinstance(size, (int, float)):
        raise ValueError(f"markdown_add_font_callback: `size`: expected `int` or `float`, got `{type(size)}` with value `{size}`")
    return dpg.add_font(file, size, parent=parent, **kwargs)
