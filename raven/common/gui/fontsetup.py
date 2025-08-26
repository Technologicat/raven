"""Font loading related DPG GUI utilities, especially for scientific use with Greek symbols and math-related special characters."""

__all__ = ["setup_font_ranges",
           "markdown_add_font_callback"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import dearpygui.dearpygui as dpg

def setup_font_ranges():
    """Set up special characters for a font.

    The price of GPU-accelerated rendering - font textures. In DPG, only Latin is enabled by default.
    We add anything that Raven's BibTeX importer may introduce from its LaTeX and HTML conversions.
    """
    # # Maybe just this?
    # dpg.add_font_range(0x300, 0x2fff)
    # return

    dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
    # Greek for math
    dpg.add_font_range(0x370, 0x3ff)
    # infinity symbol ∞
    dpg.add_font_chars([0x221e])
    # normal subgroup symbols ⊲, ⊳ (useful also as GUI arrows and similar)
    dpg.add_font_range(0x22b2, 0x22b3)
    # subscripts
    dpg.add_font_range(0x2080, 0x2089)  # zero through nine
    dpg.add_font_range(0x1d62, 0x1d65)  # i, r, u, v
    dpg.add_font_range(0x2090, 0x209c)  # a, e, o, x, schwa, h, k, l, m, n, p, s, t
    dpg.add_font_range(0x1d66, 0x1d6a)  # β, γ, ρ, φ, χ
    dpg.add_font_range(0x208a, 0x208e)  # +, -, =, (, )
    dpg.add_font_chars([0x2c7c])  # j
    # superscripts
    dpg.add_font_chars([0x2070, 0x00b9, 0x00b2, 0x00b3, 0x2074, 0x2075, 0x2076, 0x2077, 0x2078, 0x2079])  # zero through nine
    dpg.add_font_chars([0x2071, 0x207f])  # i, n
    dpg.add_font_range(0x207a, 0x207e)  # +, -, =, (, )
    # from biology dataset
    dpg.add_font_chars([0x0131])  # ı
    dpg.add_font_chars([0x2013])  # – (en dash)
    dpg.add_font_chars([0x2014])  # — (em dash)
    dpg.add_font_range(0x2018, 0x2019)  # ‘, ’
    dpg.add_font_range(0x201C, 0x201D)  # “, ”

def markdown_add_font_callback(file, size: int | float, parent=0, **kwargs) -> int:  # IMPORTANT: parameter names as in `dpg_markdown`, arguments are sent in by name.
    """Callback for `dpg_markdown` to load a font. Called whenever a new font size or family is needed.

    This calls our `setup_font_ranges` so that special characters work.
    """
    if not isinstance(size, (int, float)):
        raise ValueError(f"markdown_add_font_callback: `size`: expected `int` or `float`, got `{type(size)}` with value `{size}`")
    with dpg.font(file, size, parent=parent, **kwargs) as font:
        setup_font_ranges()
    return font
