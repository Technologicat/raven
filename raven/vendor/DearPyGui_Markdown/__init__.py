import threading
import time
import traceback
from typing import List, Any, Callable, Union, Tuple

import dearpygui.dearpygui as dpg

# `raven.common.gui.utils` is imported inside the two functions that need it rather than here, because it
# imports *this* module at module level and a top-level import back would be a cycle. It does so to hand
# the renderer its fonts: `setup_markdown` calls `set_font_registry`, `set_add_font_function` and
# `set_font`, which is how a size Raven has not loaded yet gets loaded on demand.

# The setup functions are re-exported from the submodules further down, so that a caller configures the
# renderer through the package it imported rather than having to know which module owns which setter.
__all__ = ["get_text_size",
           "CallInNextFrame", "CallWhenDPGStarted",

           "set_font_registry", "set_add_font_function", "set_font", "set_url_secondary_action",

           "wrap_text_entity", "MarkdownText",

           "add_text",
           "add_text_italic", "add_text_bold", "add_text_bold_italic"]


def get_text_size(text: str, *, wrap_width: float = -1.0, font: int | str = 0, **kwargs) -> list[float | int] | tuple[float | int, ...]:
    strip_text = text.strip()
    while 1:
        size: list[float, float] = dpg.get_text_size(text, wrap_width=wrap_width, font=font, **kwargs)

        if size is None:
            continue
        if size[1] == 0:
            continue
        if size[0] == 0 and len(strip_text) != 0:
            continue
        break
    return size


class CallInNextFrame:
    __started = False
    now_frame_queue = []

    def __new__(cls, func):
        def decorator(*args, **kwargs):
            cls.append(func, *args, **kwargs)

        return decorator

    @classmethod
    def append(cls, func, *args, **kwargs):
        if cls.__started is False:
            cls.__started = True
            threading.Thread(target=cls._worker, daemon=True).start()
        cls.now_frame_queue.append(
            [func, args, kwargs]
        )

    @classmethod
    def _worker(cls):
        while True:
            if len(cls.now_frame_queue) == 0:
                time.sleep(0.015)
                continue
            next_frame_queue = cls.now_frame_queue.copy()
            cls.now_frame_queue.clear()
            # A bare `dpg.split_frame()` *raises* where no render loop is running — before one starts, and
            # after one exits — and on this thread the exception escapes and takes the worker with it.
            from ...common.gui import utils as guiutils  # local import: see the note beside the imports
            if not guiutils.split_frame(operation="DearPyGui_Markdown: deferring render work to the next frame",
                                        required=False):
                # No render loop, so no next frame will ever come and this queue cannot be served. Stand
                # down rather than poll: the condition arises at teardown, and every further call into DPG
                # from here lands in a context that has been destroyed, which *segfaults* rather than
                # raising. A worker that retries is a worker that eventually crashes the app on exit.
                #
                # This is what used to happen by accident — the bare `dpg.split_frame` raised, the
                # exception escaped, and the thread died with a traceback. Same outcome, on purpose and
                # quietly, with `split_frame` naming the reason in the log.
                return
            for func, args, kwargs in next_frame_queue:
                try:
                    func(*args, **kwargs)
                except Exception:
                    traceback.print_exc()


class CallWhenDPGStarted:
    __thread = None
    STARTUP_DONE = False
    functions_queue = []

    @classmethod
    def append(cls, func, *args, **kwargs):
        if cls.__thread is None:
            cls.__thread = True
            threading.Thread(target=cls._worker, daemon=True).start()
        if not cls.STARTUP_DONE:
            cls.functions_queue.append(
                [func, args, kwargs]
            )
            return
        try:
            func(*args, **kwargs)
        except Exception:
            traceback.print_exc()

    @classmethod
    def _worker(cls):
        while dpg.get_frame_count() <= 1:
            time.sleep(0.01)
        # Same reasoning as in `CallInNextFrame._worker`: a bare `dpg.split_frame()` raises where no render
        # loop is running, and the exception would kill this thread rather than delay it. Standing down
        # leaves `STARTUP_DONE` unset, which is the safe state — callers then queue their work instead of
        # rendering it, and nothing here touches DPG again.
        from ...common.gui import utils as guiutils  # local import: see the note beside the imports
        if not guiutils.split_frame(operation="DearPyGui_Markdown: waiting for the GUI to start",
                                    required=False):
            return
        cls.STARTUP_DONE = True
        for func, args, kwargs in cls.functions_queue:
            try:
                func(*args, **kwargs)
            except Exception:
                traceback.print_exc()
        del cls.functions_queue


from . import font_attributes
from . import line_attributes
from . import parser
from . import text_attributes
from . import text_entities
from .attribute_types import set_font_registry, set_add_font_function
from .font_attributes import set_font
from .text_attributes import set_url_secondary_action


def wrap_text_entity(text: text_entities.StrEntity | text_entities.TextEntity, width: int | float = -1) -> text_entities.LineEntity:
    def get_width(str_entity: text_entities.StrEntity | text_entities.TextEntity) -> int | float:
        return text_entities.LineEntity.get_width(str_entity)

    def to_words(str_entity: text_entities.StrEntity | text_entities.TextEntity) -> list[text_entities.StrEntity | text_entities.TextEntity]:
        words_list = []
        word = None
        chars = str_entity.chars()
        for char in chars:
            if str(char) == " ":
                if word is not None:
                    words_list.append(word)
                words_list.append(char)
                word = None
            else:
                if word is None:
                    word = char
                else:
                    word = word + char
        if word is not None:
            words_list.append(word)
        return words_list

    print_text = text_entities.LineEntity()
    paragraphs_list = text.split('\n')
    if width < 0:
        for paragraph in paragraphs_list:
            print_text.append(paragraph)
        return print_text

    for paragraph in paragraphs_list:
        sentence = text_entities.StrEntity('')
        if len(paragraph) == 0:
            print_text.append(paragraph)
            continue

        words_list = to_words(paragraph)
        for word in words_list:
            sentence_with_next_word = sentence + word
            if get_width(sentence_with_next_word) <= width:
                sentence = sentence_with_next_word
                continue
            if len(sentence) != 0:
                print_text.append(sentence)
                if get_width(word) <= width:
                    sentence = word
                    continue

            sentence = text_entities.StrEntity('')
            for i, char in enumerate(word.chars()):
                sentence_with_next_char = sentence + char
                if get_width(sentence_with_next_char) <= width:
                    sentence = sentence_with_next_char
                else:
                    if len(sentence) > 0:
                        print_text.append(sentence)
                    sentence = char
        print_text.append(sentence)

    return print_text


class _ConvertedMessageEntity:
    # Colour for a list marker, resolved once by `MarkdownText` and handed to every `List` this entity
    # mints. It has to live here rather than on the object, because `object` below is a property that
    # constructs a *new* attribute instance on every access - one per segment the list spans.
    marker_color: list[int, int, int, int] | None = None

    def __init__(self, entity: parser.MessageEntity):
        self.entity = entity
        self.offset = entity.offset
        match type(self.entity):
            case parser.MessageEntitySeparator:
                self.end = self.offset
            case _:
                self.end = self.offset + entity.length

    @property
    def object(self):
        match type(self.entity):
            case parser.MessageEntityBold:
                return font_attributes.Bold
            case parser.MessageEntityItalic:
                return font_attributes.Italic
            case parser.MessageEntityUnderline:
                return text_attributes.Underline
            case parser.MessageEntityStrike:
                return text_attributes.Strike
            case parser.MessageEntityCode:
                return text_attributes.Code
            case parser.MessageEntityPre:
                return text_attributes.Pre(attribute_connector=self.entity.attribute_connector)
            case parser.MessageEntityTextUrl:
                return text_attributes.Url(self.entity.url,
                                           attribute_connector=self.entity.attribute_connector)
            case parser.MessageEntityFont:
                return font_attributes.Font(self.entity.color,
                                            self.entity.size)
            case parser.MessageEntityBlockquote:
                return line_attributes.Blockquote(self.entity.depth,
                                                 attribute_connector=self.entity.attribute_connector)
            case parser.MessageEntityUnorderedList:
                return line_attributes.List(self.entity.depth,
                                           attribute_connector=self.entity.attribute_connector,
                                           task=self.entity.task,
                                           task_done=self.entity.task_done,
                                           color=self.marker_color)
            case parser.MessageEntityOrderedList:
                return line_attributes.List(self.entity.depth,
                                           attribute_connector=self.entity.attribute_connector,
                                           ordered=True,
                                           index=self.entity.index,
                                           task=self.entity.task,
                                           task_done=self.entity.task_done,
                                           color=self.marker_color)
            case parser.MessageEntitySeparator:
                return line_attributes.Separator
            case parser.MessageEntityH1:
                return font_attributes.H1
            case parser.MessageEntityH2:
                return font_attributes.H2
            case parser.MessageEntityH3:
                return font_attributes.H3
            case parser.MessageEntityH4:
                return font_attributes.H4
            case parser.MessageEntityH5:
                return font_attributes.H5
            case parser.MessageEntityH6:
                return font_attributes.H6
            case _:
                raise ValueError(f'Unidentified MessageEntity: {type(self.entity)}')

    def __repr__(self):
        return f'<{self.offset}, {self.end} {self.entity}>'

    def __eq__(self, index: int):
        if isinstance(self.entity, parser.MessageEntitySeparator):
            return self.offset == index
        return self.offset < index <= self.end


class MarkdownText:
    text_entity: text_entities.TextEntity | text_entities.StrEntity

    def __init__(self, markdown_text: str, color: str | list | tuple | None = None):
        '''
        :param markdown_text: The Markdown source to render.
        :param color: Colour for text the source does not colour itself, in any spelling a `<font
                      color=...>` attribute accepts. `None` keeps the renderer's white.
        '''
        clear_text, attributes = parser.parse(markdown_text)
        for i in range(len(attributes)):
            attributes[i] = _ConvertedMessageEntity(attributes[i])

        # Give each list marker the colour of the text it sits beside. Resolved here, on the raw entity
        # spans, rather than from the attributes that end up sharing a segment with the list: a segment
        # carries whatever markup happens to be inside the item, so taking the colour from there would let
        # an item beginning with a coloured span tint the bullet. Containment is the question a *marker*
        # asks - what encloses this list - and the innermost enclosing `<font>` is the answer.
        #
        # A `<font>` opening mid-list encloses nothing and so is correctly ignored, leaving that list on the
        # document colour.
        default_marker_color = font_attributes.parse_color(color) if color is not None else None
        for entity in attributes:
            if not isinstance(entity.entity, parser.MessageEntityList):
                continue
            enclosing = [other for other in attributes
                         if isinstance(other.entity, parser.MessageEntityFont)
                         and other.offset <= entity.offset and entity.end <= other.end]
            if enclosing:
                innermost = min(enclosing, key=lambda other: other.end - other.offset)
                entity.marker_color = font_attributes.parse_color(innermost.entity.color)
            else:
                entity.marker_color = default_marker_color

        attribute_points = []
        for entity in attributes:
            attribute_points.append(entity.offset)
            attribute_points.append(entity.end)

        attribute_points.append(len(clear_text))
        attribute_points = list(set(attribute_points))
        attribute_points.sort()

        if len(attribute_points) == 0:
            self.text_entity = text_entities.StrEntity(clear_text)
        else:
            self.text_entity = text_entities.TextEntity()

        for i, point in enumerate(attribute_points):
            str_attributes = []
            for entity in attributes:
                if point == entity:
                    str_attributes.append(entity.object)

            past_point = attribute_points[i - 1] if i != 0 else 0

            str_entity = text_entities.StrEntity(clear_text[past_point:point:])
            if line_attributes.Separator in str_attributes:
                del str_attributes[str_attributes.index(line_attributes.Separator)]
                if len(str(str_entity)) > 0:
                    str_entity = text_entities.StrEntity(str(str_entity).removesuffix('\n'))
                    str_entity.set_attributes(str_attributes)
                    self.text_entity.append(str_entity)

                if point != 0:
                    self.text_entity.append(
                        text_entities.StrEntity('\n')
                    )
                str_entity = text_entities.StrEntity(' ')
                str_entity.set_attributes([line_attributes.Separator])  # noqa
                self.text_entity.append(str_entity)
            else:
                str_entity.set_attributes(str_attributes)
                self.text_entity.append(str_entity)

        if color is not None:
            text_entities.set_default_text_color(self.text_entity,
                                                 font_attributes.parse_color(color))

    def add(self, wrap: int | float = -1, parent: int | str = 0, tag: int | str = 0):
        '''
        :param wrap: Number of pixels from the start of the item until wrapping starts.
        :param parent: Parent to add this item to. (runtime adding)
        :param tag: DPG tag/alias for the top-level group of the rendered Markdown.
        :return: group with rendered text
        '''
        print_text: text_entities.LineEntity = wrap_text_entity(self.text_entity, width=wrap)

        group = dpg.add_group(parent=parent, horizontal=True, tag=tag)
        text_group = dpg.add_group(parent=group)
        attributes_group = dpg.add_group(parent=group)

        if not CallWhenDPGStarted.STARTUP_DONE:
            CallWhenDPGStarted.append(dpg.bind_item_theme, group, text_entities.AttributeController.dpg_group_theme)
            CallWhenDPGStarted.append(print_text.render, parent=text_group, attributes_group=attributes_group)
        else:
            dpg.bind_item_theme(group, text_entities.AttributeController.dpg_group_theme)
            print_text.render(parent=text_group, attributes_group=attributes_group)
        return group


def add_text(markdown_text: str,
             wrap: float | int = -1,
             parent: int | str = 0,
             pos: list[int | float, int | float] | tuple[int | float, int | float] | None = None,
             tag: int | str = 0,
             color: str | list | tuple | None = None) -> int:
    ''' Adds Markdown text.
    :param wrap: Number of pixels from the start of the item until wrapping starts.
    :param parent: Parent to add this item to. (runtime adding)
    :pos: Places the item relative to window coordinates, [0,0] is top left.
    :param tag: DPG tag/alias for the top-level group of the rendered Markdown.
    :param color: Colour for text the Markdown does not colour itself, in any spelling a `<font
                  color=...>` attribute accepts. `None` keeps the renderer's white. Prefer this to
                  wrapping the source in a `<font>` tag: an open tag on the same line as the content
                  makes the whole thing one paragraph as far as CommonMark is concerned, and a heading
                  cannot occur inside a paragraph.
    :return: group with rendered Markdown text
    '''
    rendered_group = MarkdownText(markdown_text=markdown_text, color=color).add(wrap=wrap, parent=parent, tag=tag)
    if pos is not None:
        dpg.set_item_pos(rendered_group, pos)
    return rendered_group


def add_text_italic(default_value: str = '', *, label: str = None, user_data: Any = None, use_internal_label: bool = True, tag: Union[int, str] = 0, indent: int = -1, parent: Union[int, str] = 0, before: Union[int, str] = 0, source: Union[int, str] = 0, payload_type: str = '$$DPG_PAYLOAD', drag_callback: Callable = None, drop_callback: Callable = None, show: bool = True, pos: Union[List[int], Tuple[int, ...]] = [], filter_key: str = '', tracked: bool = False, track_offset: float = 0.5, wrap: int = -1, bullet: bool = False, color: Union[List[int], Tuple[int, ...]] = (-255, 0, 0, 255), show_label: bool = False, **kwargs) -> Union[int, str]:
    """	 Adds italic text. Text can have an optional label that will display to the right of the text.

        Args:
                default_value (str, optional):
                label (str, optional): Overrides 'name' as label.
                user_data (Any, optional): User data for callbacks
                use_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).
                tag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.
                indent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.
                parent (Union[int, str], optional): Parent to add this item to. (runtime adding)
                before (Union[int, str], optional): This item will be displayed before the specified item in the parent.
                source (Union[int, str], optional): Overrides 'id' as value storage key.
                payload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.
                drag_callback (Callable, optional): Registers a drag callback for drag and drop.
                drop_callback (Callable, optional): Registers a drop callback for drag and drop.
                show (bool, optional): Attempt to render widget.
                pos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.
                filter_key (str, optional): Used by filter widget.
                tracked (bool, optional): Scroll tracking
                track_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom
                wrap (int, optional): Number of pixels from the start of the item until wrapping starts.
                bullet (bool, optional): Places a bullet to the left of the text.
                color (Union[List[int], Tuple[int, ...]], optional): Color of the text (rgba).
                show_label (bool, optional): Displays the label to the right of the text.
                id (Union[int, str], optional): (deprecated)
        Returns:
                Union[int, str]
        """
    dpg_text = dpg.add_text(default_value, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, drag_callback=drag_callback, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, wrap=wrap, bullet=bullet, color=color, show_label=show_label, **kwargs)
    dpg.bind_item_font(dpg_text,
                       font=font_attributes.Italic.get_font())
    return dpg_text


def add_text_bold(default_value: str = '', *, label: str = None, user_data: Any = None, use_internal_label: bool = True, tag: Union[int, str] = 0, indent: int = -1, parent: Union[int, str] = 0, before: Union[int, str] = 0, source: Union[int, str] = 0, payload_type: str = '$$DPG_PAYLOAD', drag_callback: Callable = None, drop_callback: Callable = None, show: bool = True, pos: Union[List[int], Tuple[int, ...]] = [], filter_key: str = '', tracked: bool = False, track_offset: float = 0.5, wrap: int = -1, bullet: bool = False, color: Union[List[int], Tuple[int, ...]] = (-255, 0, 0, 255), show_label: bool = False, **kwargs) -> Union[int, str]:
    """	 Adds bold text. Text can have an optional label that will display to the right of the text.

        Args:
                default_value (str, optional):
                label (str, optional): Overrides 'name' as label.
                user_data (Any, optional): User data for callbacks
                use_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).
                tag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.
                indent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.
                parent (Union[int, str], optional): Parent to add this item to. (runtime adding)
                before (Union[int, str], optional): This item will be displayed before the specified item in the parent.
                source (Union[int, str], optional): Overrides 'id' as value storage key.
                payload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.
                drag_callback (Callable, optional): Registers a drag callback for drag and drop.
                drop_callback (Callable, optional): Registers a drop callback for drag and drop.
                show (bool, optional): Attempt to render widget.
                pos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.
                filter_key (str, optional): Used by filter widget.
                tracked (bool, optional): Scroll tracking
                track_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom
                wrap (int, optional): Number of pixels from the start of the item until wrapping starts.
                bullet (bool, optional): Places a bullet to the left of the text.
                color (Union[List[int], Tuple[int, ...]], optional): Color of the text (rgba).
                show_label (bool, optional): Displays the label to the right of the text.
                id (Union[int, str], optional): (deprecated)
        Returns:
                Union[int, str]
        """
    dpg_text = dpg.add_text(default_value, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, drag_callback=drag_callback, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, wrap=wrap, bullet=bullet, color=color, show_label=show_label, **kwargs)
    dpg.bind_item_font(dpg_text,
                       font=font_attributes.Bold.get_font())
    return dpg_text


def add_text_bold_italic(default_value: str = '', *, label: str = None, user_data: Any = None, use_internal_label: bool = True, tag: Union[int, str] = 0, indent: int = -1, parent: Union[int, str] = 0, before: Union[int, str] = 0, source: Union[int, str] = 0, payload_type: str = '$$DPG_PAYLOAD', drag_callback: Callable = None, drop_callback: Callable = None, show: bool = True, pos: Union[List[int], Tuple[int, ...]] = [], filter_key: str = '', tracked: bool = False, track_offset: float = 0.5, wrap: int = -1, bullet: bool = False, color: Union[List[int], Tuple[int, ...]] = (-255, 0, 0, 255), show_label: bool = False, **kwargs) -> Union[int, str]:
    """	 Adds bold italic text. Text can have an optional label that will display to the right of the text.

        Args:
                default_value (str, optional):
                label (str, optional): Overrides 'name' as label.
                user_data (Any, optional): User data for callbacks
                use_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).
                tag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.
                indent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.
                parent (Union[int, str], optional): Parent to add this item to. (runtime adding)
                before (Union[int, str], optional): This item will be displayed before the specified item in the parent.
                source (Union[int, str], optional): Overrides 'id' as value storage key.
                payload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.
                drag_callback (Callable, optional): Registers a drag callback for drag and drop.
                drop_callback (Callable, optional): Registers a drop callback for drag and drop.
                show (bool, optional): Attempt to render widget.
                pos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.
                filter_key (str, optional): Used by filter widget.
                tracked (bool, optional): Scroll tracking
                track_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom
                wrap (int, optional): Number of pixels from the start of the item until wrapping starts.
                bullet (bool, optional): Places a bullet to the left of the text.
                color (Union[List[int], Tuple[int, ...]], optional): Color of the text (rgba).
                show_label (bool, optional): Displays the label to the right of the text.
                id (Union[int, str], optional): (deprecated)
        Returns:
                Union[int, str]
        """
    dpg_text = dpg.add_text(default_value, label=label, user_data=user_data, use_internal_label=use_internal_label, tag=tag, indent=indent, parent=parent, before=before, source=source, payload_type=payload_type, drag_callback=drag_callback, drop_callback=drop_callback, show=show, pos=pos, filter_key=filter_key, tracked=tracked, track_offset=track_offset, wrap=wrap, bullet=bullet, color=color, show_label=show_label, **kwargs)
    dpg.bind_item_font(dpg_text,
                       font=font_attributes.BoldItalic.get_font())
    return dpg_text
