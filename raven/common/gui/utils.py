"""DPG GUI utilities.

This module is licensed under the 2-clause BSD license, to facilitate integration anywhere.
"""

__all__ = ["get_font_path", "bootup",
           "maybe_delete_item", "has_child_items",
           "get_widget_pos", "get_widget_size", "get_widget_relative_pos", "is_mouse_inside_widget",
           "recenter_window",
           "wait_for_resize",
           "compute_tooltip_position_scalar",
           "get_pixels_per_plotter_data_unit"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
import pathlib
from typing import Optional, Tuple, Union

from unpythonic.env import env

import dearpygui.dearpygui as dpg

from ...vendor import DearPyGui_Markdown as dpg_markdown  # https://github.com/IvanNazaruk/DearPyGui-Markdown
from ...vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa  # https://github.com/juliettef/IconFontCppHeaders

from .. import numutils

from . import fontsetup

def get_font_path(font_basename: str = "OpenSans",
                  variant: Optional[str] = "Regular") -> pathlib.Path:
    """Get the path of a TTF font installed with Raven.

    `font_basename`: The part of the TTF filename before the font variant and the file extension.
                     For example, the basename of "OpenSans-Regular.ttf" is "OpenSans".

    `variant`: Usually one of "Regular", "Bold", "Italic", or "BoldItalic".

               FontAwesome icon fonts are an exception; for them, put the full filename into
               `font_basename` (including the ".ttf" file extension) and use `variant=None`.
               Then, this will just prepend the Raven fonts path to the filename.

    Returns the path to the font file as a `pathlib.Path`.
    """
    if variant is None:
        variant_str = ""
        ext = ""  # icon font names include the file extension
    else:
        variant_str = f"-{variant}"  # e.g. "OpenSans-Regular"
        ext = ".ttf"
    filename = f"{font_basename}{variant_str}{ext}"
    logger.info(f"get_font_path: basename '{font_basename}', variant '{variant}' -> '{filename}'")
    return pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "..", "fonts", filename)).expanduser().resolve()

def bootup(font_size: int,
           font_basename: str = "OpenSans") -> env:
    """Perform GUI initialization common to the Raven constellation of apps.

    Must be called *after* `dpg.create_context()`, but *before* `dpg.create_viewport(...)`.

    `font_size`: in pixels, as in DPG functions that handle fonts.

                 This is used for initializing the default GUI font, the icon fonts (for toolbars etc.),
                 and the font for the Markdown renderer.

    `font_basename`: Fonts are looked up as:

            raven/fonts/<basename>-Regular.ttf
            raven/fonts/<basename>-Bold.ttf
            raven/fonts/<basename>-Italic.ttf
            raven/fonts/<basename>-BoldItalic.ttf

        Look in `raven/fonts/` for valid values. If you need to install more fonts, place them there.

        By default, Raven has the following fonts installed:

            `font_basename="OpenSans"`: https://fonts.google.com/specimen/Open+Sans
            `font_basename="InterTight"`: https://fonts.google.com/specimen/Inter+Tight

        For scientific text, OpenSans is otherwise better (e.g. has a subscript "x" glyph for chemical formulas),
        but it confuses subscripts and superscripts. InterTight is missing that subscript "x" glyph. No better options
        have been found so far.

    Returns an `unpythonic.env` with the following attributes:
        - `icon_font_regular (DPG font ID)
        - `icon_font_solid` (DPG font ID)
        - `global_theme` (DPG theme ID), in case you need to refer to the customized default theme explicitly.
        - `my_no_spacing_theme` (DPG theme ID), also registered under the DPG tag "my_no_spacing_theme".
        - `disablable_button_theme` (DPG theme ID), also registered under the DPG tag "disablable_button_theme".


    **Details**

    This function does the following font setup:

      - Sets up DPG with a default font that's easy to read and looks good (e.g. Open Sans).

      - Sets up DPG's font ranges so that e.g. Greek letters work (important for scientific text).

      - Loads the FontAwesome icon fonts (regular and solid) at the specified font size, for toolbar buttons and similar.

      - Hooks up the on-demand font loader for `DearPyGui_Markdown`.

        NOTE: It may still be a good idea to create a dummy GUI element that forces `DearPyGui_Markdown`
              to load its fonts at app startup time. Place it somewhere offscreen, and render it once.

              During app startup, do **NOT** add more than one Markdown GUI element (see below).
              Once startup is complete, then Markdown GUI elements can be added freely.

              See `raven.visualizer.app` for an example.

    and the following DPG theme setup:

      - Sets up DPG's global theme to use rounded widgets, making apps look more friendly.

      - Registers "my_no_spacing_theme", for tight text layout.

      - Registers "disablable_button_theme" (matching the colors of DPG's built-in default theme)
        so that a disabled button using this theme also looks disabled.


    **About the Markdown renderer**

    We use the `DearPyGui_Markdown` package:
        https://github.com/IvanNazaruk/DearPyGui-Markdown

    USAGE::
        dpg_markdown.add_text(some_markdown_string)

    For font color/size, use these HTML syntaxes::
        <font color="(255, 0, 0)">Test</font>
        <font color="#ff0000">Test</font>
        <font size="50">Test</font>
        <font size=50>Test</font>

    Color and size can be used in the same font tag.

    The first use (during an app session) of a particular font size/family loads the font into the renderer.

    During app startup (first frame?), don't call `dpg_markdown.add_text` more than once, or it'll crash the app
    (some kind of race condition in font loading?). After the app has started, it's fine to call it as often as needed.
    """
    # Initialize fonts. Must be done after `dpg.create_context`, or the app will just segfault at startup.
    # https://dearpygui.readthedocs.io/en/latest/documentation/fonts.html
    with dpg.font_registry() as the_font_registry:
        # Change the default font to something that looks clean and has good on-screen readability.
        # https://fonts.google.com/specimen/Open+Sans
        with dpg.font(get_font_path(font_basename, variant="Regular"),
                      font_size) as default_font:
            fontsetup.setup_font_ranges()
        dpg.bind_font(default_font)

        # FontAwesome 6 for symbols (toolbar button icons etc.).
        # We bind this font to individual GUI widgets as needed.
        with dpg.font(get_font_path(fa.FONT_ICON_FILE_NAME_FAR, variant=None),
                      font_size) as icon_font_regular:
            dpg.add_font_range(fa.ICON_MIN, fa.ICON_MAX_16)
        with dpg.font(get_font_path(fa.FONT_ICON_FILE_NAME_FAS, variant=None),
                      font_size) as icon_font_solid:
            dpg.add_font_range(fa.ICON_MIN, fa.ICON_MAX_16)

    # Configure fonts for the Markdown renderer.
    #     https://github.com/IvanNazaruk/DearPyGui-Markdown
    dpg_markdown.set_font_registry(the_font_registry)
    dpg_markdown.set_add_font_function(fontsetup.markdown_add_font_callback)
    dpg_markdown.set_font(font_size=font_size,
                          default=get_font_path(font_basename, variant="Regular"),
                          bold=get_font_path(font_basename, variant="Bold"),
                          italic=get_font_path(font_basename, variant="Italic"),
                          italic_bold=get_font_path(font_basename, variant="BoldItalic"))

    # Modify global theme
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            # dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (53, 168, 84))  # same color as Linux Mint default selection color in the green theme
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8, category=dpg.mvThemeCat_Core)
    dpg.bind_theme(global_theme)  # set this theme as the default

    # Add a theme for tight text layout
    with dpg.theme(tag="my_no_spacing_theme") as my_no_spacing_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 0, category=dpg.mvThemeCat_Core)

    # FIX disabled controls not showing as disabled.
    # DPG does not provide a default disabled-item theme, so we provide our own.
    # Everything else is automatically inherited from DPG's global theme.
    #     https://github.com/hoffstadt/DearPyGui/issues/2068
    # TODO: Figure out how to get colors from a theme. Might not always be `(45, 45, 48)`.
    #   - Maybe see how DPG's built-in theme editor does it - unless it's implemented at the C++ level.
    #   - See also the theme color editor in https://github.com/hoffstadt/DearPyGui/wiki/Tools-and-Widgets
    disabled_color = (0.50 * 255, 0.50 * 255, 0.50 * 255, 1.00 * 255)
    disabled_button_color = (45, 45, 48)
    disabled_button_hover_color = (45, 45, 48)
    disabled_button_active_color = (45, 45, 48)
    with dpg.theme(tag="disablable_button_theme") as disablable_button_theme:
        # We customize just this. Everything else is inherited from the global theme.
        with dpg.theme_component(dpg.mvButton, enabled_state=False):
            dpg.add_theme_color(dpg.mvThemeCol_Text, disabled_color, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Button, disabled_button_color, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, disabled_button_hover_color, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, disabled_button_active_color, category=dpg.mvThemeCat_Core)

    with dpg.theme(tag="disablable_red_button_theme") as disablable_red_button_theme:  # useful for dangerous delete buttons and such
        with dpg.theme_component(dpg.mvButton, enabled_state=False):
            dpg.add_theme_color(dpg.mvThemeCol_Text, disabled_color, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Button, disabled_button_color, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, disabled_button_hover_color, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, disabled_button_active_color, category=dpg.mvThemeCat_Core)
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 96, 96))

    out = env(font_size=font_size,  # for introspection
              font_basename=font_basename,  # for introspection
              font_registry=the_font_registry,  # for the app to be able to add more fonts while running
              icon_font_regular=icon_font_regular,
              icon_font_solid=icon_font_solid,
              global_theme=global_theme,
              my_no_spacing_theme=my_no_spacing_theme,
              disablable_button_theme=disablable_button_theme,
              disablable_red_button_theme=disablable_red_button_theme)
    return out

def load_extra_font(themes_and_fonts: env,
                    font_size: int,
                    font_basename: str,
                    variant: Optional[str]) -> Union[str, int]:
    """Load another (non-default) font.

    `themes_and_fonts`: obtain this from `bootup`; the font will be cached here.
    `font_size`: in pixels, as in DPG functions that handle fonts.
    `font_basename`: passed to `get_font_path`, which see.
    `variant`: passed to `get_font_path`, which see.

    Returns the tuple `(key, id)`, where:

      - `key` is the name of the font.
              Get the ID as `themes_and_fonts[key]`.

              Key depends on all of `font_basename`, `variant` (if applicable),
              and `font_size`.

      - `id` is DPG ID of the loaded font. For convenience,
             so you don't have to fish it out of `themes_and_fonts`.

    If the font is already loaded (same variant, at the same size),
    the cached font is returned.
    """
    if variant is None:
        variant_str = ""
    else:
        variant_str = f"_{variant}"  # e.g. "OpenSans-Regular"
    key = f"{font_basename}{variant_str}_{font_size}"

    if key not in themes_and_fonts:
        with dpg.font(get_font_path(font_basename, variant="Regular"),
                      font_size,
                      parent=themes_and_fonts.font_registry) as new_font:
            fontsetup.setup_font_ranges()
        themes_and_fonts[key] = new_font

    return key, themes_and_fonts[key]

def maybe_delete_item(item: Union[str, int]) -> None:
    """Delete `item` (DPG ID or tag), if it exists. If not, the error is ignored."""
    logger.info(f"maybe_delete_item: Deleting old GUI item '{item}', if it exists.")
    try:
        dpg.delete_item(item)
    except SystemError:  # does not exist
        pass

def has_child_items(widget: Union[str, int]) -> bool:
    """Return whether `widget` (DPG tag or ID) has child items in any of its slots."""
    for slot in range(4):
        if len(dpg.get_item_children(widget, slot=slot)):
            return True
    return False

def get_widget_pos(widget: Union[str, int]) -> Tuple[int, int]:
    """Return `widget`'s (DPG tag or ID) position `(x0, y0)`, in viewport coordinates.

    This papers over the fact that most items support `dpg.get_item_rect_min`,
    but e.g. with child windows, one needs to use `dpg.get_item_pos` instead.
    """
    try:
        x0, y0 = dpg.get_item_rect_min(widget)
    except KeyError:  # some items don't have `rect_min` (e.g. child windows)
        x0, y0 = dpg.get_item_pos(widget)
    return x0, y0

def get_widget_size(widget: Union[str, int]) -> Tuple[int, int]:
    """Return `widget`'s (DPG tag or ID) on-screen size `(width, height)`, in pixels.

    This papers over the fact that most items support `dpg.get_item_rect_size`,
    but e.g. child windows store their size in the item configuration instead.
    """
    try:
        w, h = dpg.get_item_rect_size(widget)
    except KeyError:  # e.g. child window
        config = dpg.get_item_configuration(widget)
        w = config["width"]
        h = config["height"]
    return w, h

def get_widget_relative_pos(widget: Union[str, int],
                            reference: Union[str, int]) -> Tuple[int, int]:
    """Return `widget`'s (DPG tag or ID) position, measured relative to the `reference` widget (DPG tag or ID).

    This is handy when you need child window coordinates (use the child window as `reference`).
    """
    x0, y0 = get_widget_pos(widget)  # in viewport coordinates  # tag
    x0_c, y0_c = get_widget_pos(reference)  # in viewport coordinates
    x0_local = x0 - x0_c
    y0_local = y0 - y0_c
    return x0_local, y0_local

def is_mouse_inside_widget(widget: Union[str, int]) -> bool:
    """Return whether the mouse cursor is inside `widget` (DPG ID or tag)."""
    x0, y0 = get_widget_pos(widget)
    w, h = get_widget_size(widget)
    m = dpg.get_mouse_pos(local=False)  # in viewport coordinates
    if m[0] < x0 or m[0] >= x0 + w or m[1] < y0 or m[1] >= y0 + h:
        return False
    return True

def wait_for_resize(widget: Union[str, int],
                    wait_frames_max: int = 10) -> bool:
    """Wait (calling `dpg.split_frame()`) until the on-screen size of `widget` (DPG tag or ID) changes.

    If `wait_frames_max` frames have elapsed without the size changing, return.

    Return `True` if the size changed, `False` otherwise.
    """
    waited = 0
    old_size = get_widget_size(widget)
    while waited < wait_frames_max:
        dpg.split_frame()  # let the autosize happen
        waited += 1

        new_size = get_widget_size(widget)
        if new_size != old_size:
            logger.debug(f"wait_for_resize: waited {waited} frame{'s' if waited != 1 else ''} for resize of DPG widget {widget}")
            return True
    else:
        logger.debug(f"wait_for_resize: timeout ({wait_frames_max} frames) when waiting for resize of DPG widget {widget}")
    return False

def recenter_window(thewindow: Union[str, int], *, reference_window: Union[str, int]) -> None:
    """Reposition `thewindow` (DPG ID or tag), if visible, so that it is centered on `reference_window`.

    To center on viewport, pass your maximized main window as `reference_window`.
    """
    if reference_window is None:
        return
    if thewindow is None:
        return
    # Sanity check. Just try to call *some* DPG function with `thewindow` to check that the handle is valid.
    try:
        dpg.get_item_alias(thewindow)
    except Exception:
        logger.debug(f"recenter_window: {thewindow} does not exist, skipping.")
        return

    reference_window_w, reference_window_h = get_widget_size(reference_window)
    logger.debug(f"recenter_window: Reference window (tag '{dpg.get_item_alias(reference_window)}', type {dpg.get_item_type(reference_window)}) size is {reference_window_w}x{reference_window_h}.")

    # Render offscreen so we get the final size. Only needed if the size can change.
    dpg.set_item_pos(thewindow,
                     (reference_window_w,
                      reference_window_h))
    dpg.show_item(thewindow)
    logger.debug(f"recenter_window: After show command: Window is visible? {dpg.is_item_visible(thewindow)}.")
    dpg.split_frame()  # wait for render
    logger.debug(f"recenter_window: After wait for render: Window is visible? {dpg.is_item_visible(thewindow)}.")

    w, h = get_widget_size(thewindow)
    logger.debug(f"recenter_window: Window {thewindow} (tag '{dpg.get_item_alias(thewindow)}', type {dpg.get_item_type(thewindow)}) size is {w}x{h}.")

    # Center the window in the viewport
    dpg.set_item_pos(thewindow,
                     (max(0, (reference_window_w - w) // 2),
                      max(0, (reference_window_h - h) // 2)))

def compute_tooltip_position_scalar(*,
                                    algorithm: str,
                                    cursor_pos: int,
                                    tooltip_size: int,
                                    viewport_size: int,
                                    offset: int = 20) -> int:
    """Compute x or y position for a tooltip. (Either one of them; hence "scalar".)

    This positions the tooltip elegantly, trying to keep it completely within the DPG viewport area.
    This is mostly useful for tooltips triggered by custom code, such as for a scatterplot dataset in a plotter.

    `algorithm`: one of "snap", "snap_old", "smooth".
                 "snap": Right/bottom side if the tooltip fits there, else left/top side.
                 "snap_old": Right/bottom side when the cursor is at the left/top side of viewport, else left/top side.
                 "smooth": Cursor at left edge -> right/bottom side; cursor at right edge -> left/top side; in between,
                           smoothly varying as a function of the cursor position. For the perfectionists.

                 If unsure, try "snap" for the x coordinate, and "smooth" for the y coordinate; usually looks good.

    `cursor_pos`: mouse cursor position (x or y) depending on which axis you are computing, in viewport coordinates.
    `tooltip_size`: width or height (depending on axis) of the tooltip window, in pixels.
    `viewport_size`: width or height (depending on axis), size of the DPG viewport (or equivalently, primary window), in pixels.
    `offset`: int. This allows positioning the tooltip a bit off from `cursor_pos`, so that the mouse cursor won't
              immediately hover over it when the tooltip is shown.

              This is important, because in DPG a tooltip is a separate window, so this would prevent further
              mouse hover events of the actual window under the tooltip from being triggered (until the mouse
              exits the tooltip area).

    Usage::

        mouse_pos = dpg.get_mouse_pos(local=False)  # in viewport coordinates
        tooltip_size = dpg.get_item_rect_size(my_tooltip_window)  # after `dpg.split_frame()` if needed
        w, h = dpg.get_item_rect_size(my_primary_window)
        xpos = compute_tooltip_position_scalar(algorithm="snap",
                                               cursor_pos=mouse_pos[0],
                                               tooltip_size=tooltip_size[0],
                                               viewport_size=w)
        ypos = compute_tooltip_position_scalar(algorithm="smooth",
                                               cursor_pos=mouse_pos[1],
                                               tooltip_size=tooltip_size[1],
                                               viewport_size=h)
        dpg.set_item_pos(my_tooltip_window, [xpos, ypos])
    """
    if algorithm not in ("snap", "snap_old", "smooth"):
        raise ValueError(f"Unknown `algorithm` '{algorithm}'; supported: 'snap', 'snap_old', 'smooth'.")

    if algorithm == "snap":  # Right/bottom side if the tooltip fits there, else left/top side.
        if cursor_pos + offset + tooltip_size < viewport_size:  # does it fit?
            return cursor_pos + offset
        elif cursor_pos - offset - tooltip_size >= 0:  # does it fit?
            return cursor_pos - offset - tooltip_size
        else:  # as far as it can go to the right/below while the right/bottom edge remains inside the viewport
            return viewport_size - tooltip_size

    elif algorithm == "snap_old":  # Right/bottom side when the cursor is at the left/top side of viewport, else left/top side.
        if cursor_pos < viewport_size / 2:
            return cursor_pos + offset
        else:
            return cursor_pos - offset - tooltip_size

    elif algorithm == "smooth":  # Cursor at left edge -> right/bottom side; cursor at right edge -> left/top side; in between, smoothly varying as a function of the cursor position.
        # Candidate position to the right/below (preferable in the left/top half of the viewport)
        if cursor_pos + offset + tooltip_size < viewport_size:  # does it fit?
            pos1 = cursor_pos + offset
        else:  # as far as it can go to the right/below while the right/bottom edge remains inside the viewport
            pos1 = viewport_size - tooltip_size

        # Candidate position to the left/above (preferable in the right/bottom half of the viewport)
        if cursor_pos - offset - tooltip_size >= 0:  # does it fit?
            pos2 = cursor_pos - offset - tooltip_size
        else:  # as far as it can go to the left/above while the left/top edge remains inside the viewport
            pos2 = 0

        # Weighted average of the two candidates, with a smooth transition.
        # This makes the tooltip x position vary smoothly as a function of the data point location in the plot window.
        # Due to symmetry, this places the tooltip exactly at the middle when the mouse is at the midpoint of the viewport (not necessarily at an axis line; that depends on axis limits).
        r = numutils.clamp(cursor_pos / viewport_size)  # relative coordinate, [0, 1]
        s = numutils.nonanalytic_smooth_transition(r, m=2.0)
        pos = (1.0 - s) * pos1 + s * pos2

        return pos

def get_pixels_per_plotter_data_unit(plot_widget: Union[str, int],
                                     xaxis: Union[str, int],
                                     yaxis: Union[str, int]) -> Tuple[int, int]:
    """Estimate pixels per DPG plotter data unit, for conversion between viewport space and data space.

    `plot_widget`: dpg tag or ID, the plotter widget (`dpg.plot`).
    `xaxis`: dpg tag or ID, the x axis widget of the plotter (`dpg.add_plot_axis(dpg.mvXAxis, ...)`).
    `yaxis`: dpg tag or ID, the y axis widget of the plotter (`dpg.add_plot_axis(dpg.mvYAxis, ...)`).

    This is subtly wrong, because the plot widget includes also the space for the axis labels and such.
    But there seems to be no way to get pixels per data unit from a plot in DPG (unless using a custom series,
    which we don't).

    Raven uses this for estimation of on-screen distances in data space. For that purpose this is good enough.

    Returns the tuple `(pixels_per_data_unit_x, pixels_per_data_unit_y)`.

    Note that if axes are not equal aspect, the x/y results may be different.
    """
    # x0, y0 = dpg.get_item_rect_min("plot")
    pixels_w, pixels_h = dpg.get_item_rect_size(plot_widget)
    xmin, xmax = dpg.get_axis_limits(xaxis)  # in data space
    ymin, ymax = dpg.get_axis_limits(yaxis)  # in data space
    data_w = xmax - xmin
    data_h = ymax - ymin
    if data_w == 0 or data_h == 0:  # no data in view (can happen e.g. if the plot hasn't been rendered yet)
        return [0.0, 0.0]
    pixels_per_data_unit_x = pixels_w / data_w
    pixels_per_data_unit_y = pixels_h / data_h
    return pixels_per_data_unit_x, pixels_per_data_unit_y
