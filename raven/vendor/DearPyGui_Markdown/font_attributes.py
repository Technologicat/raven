import ast
import os.path
import traceback

from ...common import color as common_color  # aliased: `color` is a parameter name here

from .attribute_types import Attribute, FontAttribute

__all__ = ["parse_color",

           "Font",
           "Default", "Bold", "Italic", "BoldItalic",
           "H1", "H2", "H3", "H4", "H5", "H6",

           "set_font"]


def parse_color(color: str | list | tuple) -> list[int, int, int, int]:
    '''Normalize a colour into RGBA, accepting every spelling a `<font color=...>` attribute may carry.

    A string in any spelling `raven.common.color.parse` accepts — `'#rrggbb'`, `'#rrggbbaa'`, the short
    `'#rgb'` forms, a `'(r, g, b)'` literal — or a list/tuple. Short forms are padded to opaque RGBA, and
    anything beyond four components of a list/tuple is dropped. Raises `ValueError` for a string that is
    not a colour.
    '''
    if isinstance(color, str):
        color = common_color.parse(color)
    color = list(color)[:4:]
    for i in range(4 - len(color)):
        color.append(255)
    return color


class Font(Attribute):
    color: list[int, int, int, int]
    size: int | None

    def __init__(self, color: str | list, size: str | float | int):
        if isinstance(size, str):
            try:
                size = ast.literal_eval(size)
            except Exception:
                traceback.print_exc()
                size = None
        self.size = size

        self.color = parse_color(color)


class Default(FontAttribute):
    ...


class Bold(FontAttribute):
    ...


class Italic(FontAttribute):
    ...


class BoldItalic(FontAttribute):
    ...


class H1(FontAttribute):
    font_multiply = 2


class H2(FontAttribute):
    font_multiply = 1.5


class H3(FontAttribute):
    font_multiply = 1.17


class H4(FontAttribute):
    font_multiply = 1


class H5(FontAttribute):
    font_multiply = 0.83


class H6(FontAttribute):
    font_multiply = 0.67


def set_font(font_size: int | float = 13, *,
             default: str | os.PathLike[str] = None,
             bold: str | os.PathLike[str] = None,
             italic: str | os.PathLike[str] = None,
             italic_bold: str | os.PathLike[str] = None) -> int:
    """
    :return: default font
    """
    fonts = {
        Default: default,
        Bold: bold,
        Italic: italic,
        BoldItalic: italic_bold,
    }
    for Font in fonts:
        font_path = fonts[Font]
        if font_path:
            Font.set_font(font_path, font_size)
        else:
            Font.set_font(Font.font_path, font_size)

    return Default.get_font()
