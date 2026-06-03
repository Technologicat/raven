from typing import Callable

from . import get_text_size
from .attribute_types import Attribute, AttributeConnector, CallInNextFrame, HoverAttribute

import dearpygui.dearpygui as dpg

from ...common.gui import utils as guiutils


# --------------------------------------------------------------------------------
# Raven extension: optional per-URL "secondary action".
#
# When configured via `set_url_secondary_action`, each rendered URL gets a small clickable icon
# immediately to its left; clicking it invokes `callback(url)`. Raven-librarian uses this to offer
# a "send this link to the chat input" affordance, distinct from the URL's normal left/middle
# click (open in browser). Unset by default, so the upstream renderer behavior is unchanged.

_url_secondary_action: Callable[[str], None] | None = None
_url_secondary_glyph: str = ""
_url_secondary_font: int | str | None = None
_url_secondary_color: list[int, int, int, int] = (140, 160, 200, 255)
_url_secondary_tooltip: str = "{url}"

def set_url_secondary_action(callback: Callable[[str], None] | None,
                             *,
                             glyph: str = "",
                             font: int | str | None = None,
                             color: list[int, int, int, int] = (140, 160, 200, 255),
                             tooltip: str = "{url}") -> None:
    """Configure an optional secondary action, shown as a clickable icon to the left of each URL.

    `callback`: called with the URL string when the icon is clicked. `None` (default) disables the
                feature entirely — no icon, upstream behavior.
    `glyph`: the icon character to display (e.g. a FontAwesome glyph). Empty also disables.
    `font`: DPG font bound to the icon (e.g. an icon font); `None` uses the current font.
    `color`: icon color.
    `tooltip`: tooltip text; `{url}` is substituted with the URL.
    """
    global _url_secondary_action, _url_secondary_glyph, _url_secondary_font
    global _url_secondary_color, _url_secondary_tooltip
    _url_secondary_action = callback
    _url_secondary_glyph = glyph
    _url_secondary_font = font
    _url_secondary_color = color
    _url_secondary_tooltip = tooltip

def render_url_secondary_action_icon(url_attribute: "Url", parent: int | str, body_font: int | str | None = None) -> None:
    """Render the secondary-action icon to the left of a URL's first text run, if configured.

    No-op when no action is configured. Idempotent per URL: emits at most one icon even when the
    URL wraps across several runs, tracked via the shared `attribute_connector` (which survives the
    per-run deep-copy of the `Url` attribute). Called from `AttributeController.render`.

    `body_font`: the URL run's text font, used to render the non-breaking space that separates the
    icon from the link text — a real body font reliably has a space advance, whereas the icon font
    and the DPG default font may not.
    """
    if _url_secondary_action is None or not _url_secondary_glyph:
        return
    connector = url_attribute.attribute_connector
    if getattr(connector, "_raven_secondary_icon_done", False):
        return
    connector._raven_secondary_icon_done = True  # one icon per URL, not per wrapped run

    action = _url_secondary_action
    url = url_attribute.url
    with guiutils.nonexistent_ok():
        icon = dpg.add_text(_url_secondary_glyph, parent=parent, color=_url_secondary_color)
        if _url_secondary_font is not None:
            dpg.bind_item_font(icon, _url_secondary_font)
        icon_tooltip = dpg.add_tooltip(parent=icon)
        dpg.add_text(_url_secondary_tooltip.format(url=url), parent=icon_tooltip)
        handler = dpg.add_item_handler_registry()
        def on_secondary_icon_clicked(_sender, _app_data, _user_data) -> None:  # DPG click-callback signature; args unused (url/action captured from the closure)
            action(url)
        dpg.add_item_clicked_handler(parent=handler, callback=on_secondary_icon_clicked)
        dpg.bind_item_handler_registry(icon, handler)
        # One non-breaking space between the icon and the link text, in the body font (which has a
        # reliable space advance). The group's ItemSpacing is 0, so this is the only gap.
        gap = dpg.add_text(chr(0x00A0), parent=parent)
        if body_font is not None:
            dpg.bind_item_font(gap, body_font)


class Underline(Attribute):
    @staticmethod
    def render(dpg_text: int, dpg_text_group: int, font=None, parent=0, color=(255, 255, 255, 255)):
        '''
        :return: [drawlist, draw_line]
        '''
        with guiutils.nonexistent_ok() as nok:
            pos = dpg.get_item_pos(dpg_text_group)
            x, y = pos
            group_width, group_height = dpg.get_item_rect_size(dpg_text_group)
            text_width, text_height = get_text_size(dpg.get_value(dpg_text), font=font)
            y = y + (group_height - text_height) / 2
            drawlist_group = dpg.add_group(pos=[x, y], parent=parent)
            drawlist = dpg.add_drawlist(parent=drawlist_group, width=group_width, height=text_height)
            thickness = text_height / 15
            line_y = text_height - thickness + thickness / 5
            line = dpg.draw_line([0, line_y], [group_width, line_y], parent=drawlist, color=color, thickness=thickness)
            return drawlist, line
        if nok.errored:  # does not exist (most likely, container deleted in another thread while still rendering)
            return None, None


class Strike(Attribute):
    @staticmethod
    def render(dpg_text: int, dpg_text_group: int, font=None, parent=0, color=(255, 255, 255)):
        '''
        :return: [drawlist, draw_line]
        '''
        with guiutils.nonexistent_ok() as nok:
            pos = dpg.get_item_pos(dpg_text_group)
            x, y = pos
            group_width, group_height = dpg.get_item_rect_size(dpg_text_group)
            text_width, text_height = get_text_size(dpg.get_value(dpg_text), font=font)
            y = y + (group_height - text_height) / 2
            drawlist_group = dpg.add_group(pos=[x, y], parent=parent)
            drawlist = dpg.add_drawlist(parent=drawlist_group, width=group_width, height=text_height)
            thickness = text_height / 15
            line_y = text_height / 2 + thickness / 2 + text_height / 20
            line = dpg.draw_line([0, line_y], [group_width, line_y], parent=drawlist, color=color, thickness=thickness)
            return drawlist, line
        if nok.errored:  # does not exist (most likely, container deleted in another thread while still rendering)
            return None, None


class Code(Attribute):
    color = (55, 55, 65, 255)
    border_color = color

    @classmethod
    def render(cls, dpg_text_group: int):
        with guiutils.nonexistent_ok() as nok:
            width, height = dpg.get_item_rect_size(dpg_text_group)
            pos = dpg.get_item_pos(dpg_text_group)
            child = dpg.get_item_children(dpg_text_group, 1)[0]
            group = dpg.add_group(pos=pos, before=child)
            drawlist = dpg.add_drawlist(parent=group, width=width, height=height)
            dpg.draw_quad([0, 0], [width, 0],
                          [width, height], [0, height],
                          fill=cls.color,
                          color=cls.border_color,
                          parent=drawlist)
        if nok.errored:  # does not exist (most likely, container deleted in another thread while still rendering)
            return


class Pre(Attribute):
    color = (55, 55, 65, 255)
    border_color = (110, 110, 130, 200)

    def __init__(self, attribute_connector: AttributeConnector):
        self.attribute_connector = attribute_connector
        self.attribute_connector.max_width = 0
        self.attribute_connector.x0, self.attribute_connector.y0 = (None, None)
        self.attribute_connector.x1, self.attribute_connector.y1 = (None, None)
        self.attribute_connector.used_y = []

    def render(self, dpg_text_group: int):
        with guiutils.nonexistent_ok() as nok:
            self.width, self.height = dpg.get_item_rect_size(dpg_text_group)
            pos = dpg.get_item_pos(dpg_text_group)
            self.dpg_text_group = dpg_text_group

            pos_end = (self.width + pos[0], pos[1] + self.height)

            a_c = self.attribute_connector
            if a_c.x0 is None:
                a_c.x0, a_c.y0 = pos
                a_c.x1, a_c.y1 = pos_end

            if a_c.x0 > pos[0]:
                a_c.x0 = pos[0]
            if a_c.y0 > pos[1]:
                a_c.y0 = pos[1]
            if a_c.x1 < pos_end[0]:
                a_c.x1 = pos_end[0]
            if a_c.y1 < pos_end[1]:
                a_c.y1 = pos_end[1]
        if nok.errored:  # does not exist (most likely, container deleted in another thread while still rendering)
            return

    @CallInNextFrame
    def post_render(self, attributes_group=0):
        with guiutils.nonexistent_ok() as nok:
            width, height = dpg.get_item_rect_size(self.dpg_text_group)
            pos = dpg.get_item_pos(self.dpg_text_group)
            child = dpg.get_item_children(self.dpg_text_group, 1)[0]
            group = dpg.add_group(pos=pos, before=child)
            children = dpg.get_item_children(dpg.get_item_parent(self.dpg_text_group), 1)

            a_c = self.attribute_connector
            if children[-1] == self.dpg_text_group:
                width = a_c.x1 - pos[0]
                border_group = dpg.add_group(parent=attributes_group, pos=(a_c.x0, a_c.y0))
                border_width = a_c.x1 - a_c.x0
                border_height = a_c.y1 - a_c.y0
                border_drawlist = dpg.add_drawlist(parent=border_group, width=border_width, height=border_height)
                dpg.draw_quad([0, 0], [border_width, 0],
                              [border_width, border_height], [0, border_height],
                              color=self.border_color,
                              parent=border_drawlist)

            drawlist = dpg.add_drawlist(parent=group, width=width, height=height)
            dpg.draw_quad([0, 0], [width, 0],
                          [width, height], [0, height],
                          fill=self.color,
                          color=self.color,
                          parent=drawlist)
        if nok.errored:  # does not exist (most likely, container deleted in another thread while still rendering)
            return


class Url(HoverAttribute):
    color: list[int, int, int, int] = (85, 135, 205, 255)
    line_color: list[int, int, int, int] = (255, 255, 255, 0)
    hover_color: list[int, int, int, int] = (153, 187, 255, 255)

    url: str

    dpg_text_objects: list[int]
    underline_objects: list[int]

    def __init__(self, url: str, attribute_connector: AttributeConnector | None):
        super().__init__(attribute_connector)
        self.url = url
        self.dpg_text_objects = []
        self.underline_objects = []
        self.now_hover_item = None

    def render(self, dpg_text, font=None, parent=0):
        super().render()
        self.add_item_to_handler(dpg_text)
        self.dpg_text_objects.append(dpg_text)
        with guiutils.nonexistent_ok() as nok:
            dpg.configure_item(dpg_text, color=self.color)
            # Raven customization: show the linked URL as a tooltip
            url_tooltip = dpg.add_tooltip(parent=dpg_text)
            dpg.add_text(self.url, parent=url_tooltip)
        if nok.errored:  # does not exist (most likely, container deleted in another thread while still rendering)
            return

    def hover(self):
        with guiutils.nonexistent_ok() as nok:
            for item in self.dpg_text_objects:
                dpg.configure_item(item, color=self.hover_color)
            for item in self.underline_objects:
                dpg.configure_item(item, color=self.hover_color)
        if nok.errored:  # does not exist (most likely, container deleted in another thread while still rendering)
            return

    def unhover(self):
        with guiutils.nonexistent_ok() as nok:
            for item in self.dpg_text_objects:
                dpg.configure_item(item, color=self.color)
            for item in self.underline_objects:
                dpg.configure_item(item, color=self.line_color)
        if nok.errored:  # does not exist (most likely, container deleted in another thread while still rendering)
            return

    def click(self, mouse_button):
        if mouse_button in [2, 0]:
            import webbrowser
            webbrowser.open_new_tab(self.url)
