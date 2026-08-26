import dearpygui.dearpygui as dpg

from . import get_text_size
from .attribute_types import LineAttribute, AttributeConnector
from .font_attributes import Default

from ...common.gui import utils as guiutils

__all__ = ["Separator", "Blockquote", "List"]


class Separator(LineAttribute):
    @staticmethod
    def render(before: int, attributes_group=0):  # noqa
        # `before` is the row group this separator sits above; the separator
        # places itself as a sibling immediately preceding it, so the DPG
        # parent we add into is the row group's own parent.
        parent = dpg.get_item_parent(before)
        height = get_text_size('|', font=Default.get_font())[1]
        group = dpg.add_group(parent=parent, before=before)
        dpg.add_spacer(parent=group, height=int(height * 0.5))
        dpg.add_separator(parent=group)
        dpg.add_spacer(parent=group, height=int(height * 0.5))


class Blockquote(LineAttribute):
    width = 20
    line_width = 6

    depth: int
    color = [50, 55, 65, 255]

    def __init__(self, depth: int, attribute_connector: AttributeConnector):
        self.depth = depth
        self.attribute_connector = attribute_connector

    def __repr__(self):
        return f"<Blockquote.{self.depth}, id: {hex(id(self.attribute_connector))}>"

    def get_width(self) -> int | float:
        return self.width

    def render(self, text_height: int | float, parent=0, attributes_group=0):
        # The bar goes into the text flow, in the row of the line it marks, rather than being drawn at
        # coordinates read off that row once it had been laid out. Same reasoning as `List`: an absolute
        # position is correct exactly until something above the quote changes height, and then the bar is
        # left behind beside whatever text has moved into its place.
        #
        # It replaces the spacer that used to reserve the bar's width, so the row is laid out as before.
        with guiutils.nonexistent_ok():
            group = dpg.add_group(parent=parent)
            drawlist = dpg.add_drawlist(parent=group, width=self.get_width(), height=text_height)
            x_line = (self.get_width() / 2) - 1
            dpg.draw_line([x_line, 0],
                          [x_line, text_height],
                          parent=drawlist, color=self.color, thickness=self.line_width)


class List(LineAttribute):
    check_box_theme: int = None
    max_index_symbols_length = 4

    # A marker is structural, so its colour comes from the context the list sits in - never from anything
    # inside an item, which would let a coloured first word tint the bullet. `MarkdownText` resolves it and
    # passes it in; this is the fallback for a list built with no document behind it.
    color: list[int, int, int, int] = [255, 255, 255, 255]

    depth: int
    ordered: bool
    index: int
    task: bool
    task_done: bool

    spacer_group: int = None
    task_spacer: int = None

    def __new__(cls, *args, **kwargs):
        if cls.check_box_theme is None:
            with dpg.theme() as cls.check_box_theme:
                with dpg.theme_component(dpg.mvAll, enabled_state=False, parent=cls.check_box_theme) as theme_component:
                    dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (75, 255, 75, 255), category=dpg.mvThemeCat_Core, parent=theme_component)
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (0, 0, 0, 0), category=dpg.mvThemeCat_Core, parent=theme_component)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core, parent=theme_component)
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4, category=dpg.mvThemeCat_Core, parent=theme_component)
                    dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1, category=dpg.mvThemeCat_Core, parent=theme_component)
                    dpg.add_theme_color(dpg.mvThemeCol_Border, (255, 255, 255, 255), category=dpg.mvThemeCat_Core, parent=theme_component)
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (0, 0, 0, 0), category=dpg.mvThemeCat_Core, parent=theme_component)
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (0, 0, 0, 0), category=dpg.mvThemeCat_Core, parent=theme_component)

        return super().__new__(cls)

    def __init__(self, depth: int, attribute_connector: AttributeConnector,
                 ordered: bool = False,
                 index: int = 1,
                 task: bool = False,
                 task_done: bool = False,
                 color: list[int, int, int, int] | None = None):
        self.depth = depth
        self.attribute_connector = attribute_connector
        self.attribute_connector.first_line_objects = None
        self.ordered = ordered
        self.index = index
        self.task = task
        self.task_done = task_done
        if color is not None:
            self.color = color

    def __repr__(self):
        if self.ordered:
            return f"<List.{self.depth}, i: {self.index}, attr_id: {hex(id(self.attribute_connector))} id: {hex(id(self))}>"
        return f"<List.{self.depth}, attr_id: {hex(id(self.attribute_connector))} id: {hex(id(self))}>"

    def get_width(self) -> int | float:
        width = get_text_size(f"{'0' * self.max_index_symbols_length}.  ", font=Default.get_font())[0]
        width += self.get_task_width()
        return width

    def get_task_width(self) -> int | float:
        width = 0
        if not self.task:
            return width
        if self.attribute_connector.first_line_objects is not None:  # noqa
            if self in self.attribute_connector.first_line_objects:  # noqa
                width += Default.get_now_font_size() + get_text_size(" " * 2, font=Default.get_font())[0]
        else:
            width += Default.get_now_font_size() + get_text_size(" " * 2, font=Default.get_font())[0]
        return width

    def render(self, text_height: int | float, parent=0, attributes_group=0):
        spacer_group = dpg.add_group(parent=parent, horizontal=True)
        # The marker's slot in the text flow, a group rather than a bare spacer so that `post_render` can
        # fill it in place. That is what keeps the marker's position a matter of layout rather than of
        # coordinates: it sits in the same row as its text, and moves when the row does.
        #
        # It starts out holding a spacer the full width of the marker, which is what a *continuation* line
        # of a wrapped item keeps — only the first line of an item is given a marker.
        self.marker_slot = dpg.add_group(parent=spacer_group, horizontal=True)
        dpg.add_spacer(width=self.get_width() - self.get_task_width(), parent=self.marker_slot)
        if self.task:
            self.task_spacer = dpg.add_spacer(width=self.get_task_width(), parent=spacer_group)
        self.text_height = text_height
        self.spacer_group = spacer_group
        self.attribute_connector.append(self)

    def _fill_marker_slot(self, marker_width: int | float, top_padding: int | float):
        """Clear the slot and open a column in it for the marker, right-aligned and vertically centred.

        Returns the DPG ID of that column, for the caller to draw its own kind of marker into.

        `marker_width`: how wide the marker's own cell is. The slot's remaining width goes in front of it as
                        a spacer, which is what right-aligns the marker against the text that follows.
        `top_padding`: how far down the marker sits within the line. A line can be taller than the marker —
                       a larger font in the item, or a wrapped line — and the marker is centred against it.
        """
        dpg.delete_item(self.marker_slot, children_only=True)
        leading = (self.get_width() - self.get_task_width()) - marker_width
        if leading > 0:
            dpg.add_spacer(width=leading, parent=self.marker_slot)
        column = dpg.add_group(parent=self.marker_slot)
        if top_padding > 0:
            dpg.add_spacer(height=top_padding, parent=column)
        return column

    def post_render(self, attributes_group=0):
        # Not deferred to the next frame, unlike the base-class hook it overrides. That deferral paid for
        # reading a laid-out position, and nothing here reads one any more — the marker goes into a slot the
        # row already holds. `LineEntity.render` drains its queue after every line of the block is built, so
        # "which line is first" is answerable by then without waiting for a frame.
        if self != self.attribute_connector[0]:
            return
        self.attribute_connector.append(self)
        if self.ordered:
            self.ordered_render(attributes_group=attributes_group)
        else:
            self.unordered_render(attributes_group=attributes_group)

        if self.task:
            dpg.delete_item(self.task_spacer)
            checkbox = dpg.add_checkbox(enabled=False, default_value=self.task_done, parent=self.spacer_group)
            dpg.bind_item_theme(checkbox, self.check_box_theme)
            dpg.add_spacer(width=get_text_size(' ' * 2, font=Default.get_font())[0], parent=self.spacer_group)

    def ordered_render(self, attributes_group=0):
        with guiutils.nonexistent_ok():
            text = f'{str(self.index)[-4::]}.  '
            render_text_width, render_text_height = get_text_size(text, font=Default.get_font())
            column = self._fill_marker_slot(marker_width=render_text_width,
                                            top_padding=(self.text_height - render_text_height) / 2)
            dpg_text = dpg.add_text(text, parent=column, color=self.color)
            dpg.bind_item_font(dpg_text, font=Default.get_font())

    def unordered_render(self, attributes_group=0):
        with guiutils.nonexistent_ok():
            text = '0.  '
            render_text_width, render_text_height = get_text_size(text, font=Default.get_font())
            height = render_text_height / 2.5
            width = height
            thickness = height / 7
            # The bullet is smaller than a digit, so it sits lower in the line than a number would: centred
            # against the line, then dropped by most of the difference between a digit's height and its own.
            column = self._fill_marker_slot(marker_width=render_text_width,
                                            top_padding=((self.text_height - render_text_height) / 2 +
                                                         (render_text_height - height) * 0.77))
            # The drawlist is as wide as a number's cell even though the bullet drawn in it is a few pixels
            # across, so an unordered item's text starts where an ordered one's does. Sizing it to the glyph
            # instead shortens the whole row, and the two list kinds stop lining up.
            drawlist = dpg.add_drawlist(width=render_text_width, height=height, parent=column)

            depth = self.depth - self.depth // 4 * 4
            # The hollow variants take their colour from the outline rather than the fill, so `color`
            # has to be set on every case while `fill` alternates - a hollow marker left at DPG's
            # default would be the only white thing in a coloured list.
            match depth:
                case 1:
                    dpg.draw_circle([height / 2, width / 2],
                                    width / 2 - thickness / 2,
                                    parent=drawlist,
                                    thickness=thickness,
                                    color=self.color,
                                    fill=self.color)
                case 2:
                    dpg.draw_circle([height / 2, width / 2],
                                    width / 2 - thickness / 2,
                                    parent=drawlist,
                                    thickness=thickness,
                                    color=self.color,
                                    fill=(0, 0, 0, 0))
                case 3:
                    dpg.draw_quad([thickness, thickness], [width - thickness, thickness],
                                  [width - thickness, height - thickness], [thickness, height - thickness],
                                  parent=drawlist,
                                  thickness=thickness,
                                  color=self.color,
                                  fill=self.color)
                case _:
                    dpg.draw_quad([thickness, thickness], [width - thickness, thickness],
                                  [width - thickness, height - thickness], [thickness, height - thickness],
                                  parent=drawlist,
                                  thickness=thickness,
                                  color=self.color,
                                  fill=(0, 0, 0, 0))
