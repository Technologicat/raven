"""Chat controller.

This module renders a linearized chat view of the current branch, and contains the scaffold to GUI integration
that controls chatting with the AI.
"""

# TODO: check if we need to shuffle the abstraction levels around - e.g. if there are many references to `self.parent_view.chat_controller.something`, does `something` really belong to the controller level?

__all__ = ["DPGChatController"]

import logging
logger = logging.getLogger(__name__)

import collections
import concurrent.futures
import io
import json
import pathlib
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union
import urllib.parse
import uuid
import webbrowser

import numpy as np

import dearpygui.dearpygui as dpg

from unpythonic import flatten, memoize, sym
from unpythonic.env import env

from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa  # https://github.com/juliettef/IconFontCppHeaders
from ..vendor import DearPyGui_Markdown as dpg_markdown  # https://github.com/IvanNazaruk/DearPyGui-Markdown

from ..client import api  # Raven-server support
from ..client.avatar_controller import DPGAvatarController

from ..common import bgtask
from ..common import utils as common_utils

from ..common.gui import animation as gui_animation
from ..common.gui import utils as guiutils

from . import chattree
from . import chatutil
from . import config as librarian_config
from . import hybridir
from . import imagestore
from . import llmclient
from . import scaffold

gui_config = librarian_config.gui_config  # shorthand, this is used a lot

# --------------------------------------------------------------------------------

role_to_colors = {"assistant": {"front": gui_config.chat_color_ai_front, "back": gui_config.chat_color_ai_back},
                  "system": {"front": gui_config.chat_color_system_front, "back": gui_config.chat_color_system_back},
                  "tool": {"front": gui_config.chat_color_tool_front, "back": gui_config.chat_color_tool_back},
                  "user": {"front": gui_config.chat_color_user_front, "back": gui_config.chat_color_user_back},
                  }

# Built-in tools that reach out over the network -> light up the WEB (globe) indicator while they run.
web_access_tool_names = frozenset(("websearch", "webfetch"))


def _provenance_filename(maybe_url: Optional[str]) -> Optional[str]:
    """Best-effort original filename from an image's provenance URL: the basename of a `file://` or `https://` URL.

    Returns `None` for an inline `data:` URL (carries no filename), an empty URL, or a URL whose path has no
    basename (e.g. a bare host). Percent-escapes are decoded, so `.../my%20photo.png` -> `my photo.png`."""
    if not maybe_url or maybe_url.startswith("data:"):
        return None
    path = urllib.parse.urlparse(maybe_url).path
    name = pathlib.Path(urllib.parse.unquote(path)).name
    return name or None

def _open_source_url(url: str) -> None:
    """Open an image's recorded provenance source: a `file://` local original in its default application,
    anything else (an `https://` page) in the web browser. Raises like the underlying opener when a local
    original has moved or been deleted, so the caller can flash a non-intrusive failure acknowledgment."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        common_utils.open_file(urllib.parse.unquote(parsed.path))
    else:
        webbrowser.open(url)

# --------------------------------------------------------------------------------

def format_chat_message_for_clipboard(message_number: Optional[int],
                                      role: str,
                                      persona: Optional[str],
                                      text: str,
                                      add_heading: bool) -> str:
    """Format a chat message for copying to clipboard, by adding a metadata header as Markdown.

    As a preprocessing step, `persona` is stripped from the beginning of each line in `message_text`.
    It is then re-added in a unified form.

    `message_number`: The sequential number of the message in the current linearized view.
                      If `None`, the number part in the formatted output is omitted.

    `role`: One of the roles supported by `raven.librarian.llmclient`.
            Typically, one of "assistant", "system", "tool", or "user".

    `persona`: The persona name speaking `text`, or `None` if the role has no persona name ("system" and "tool" are like this).

               To get the **current session's** persona, use::

                   persona=llm_settings.personas.get(role, None)

               where `role` is one of "assistant", "system", "tool", "user".

               To get the **stored** persona from a chat node::

                   persona=node_payload["general_metadata"]["persona"]

               This may differ from the current session's persona, e.g. if the chat node was generated with a different AI character.

    `text`: The text content of the chat message to format.
            The content is pasted into the output as-is.

    `add_heading`: Whether to include the message number and role's character name
                   in the final output.

                   Example. If `add_heading` is `True`, then both::

                       Lorem ipsum.

                   and::

                       Aria: Lorem ipsum.

                   become::

                       *[#42]* **Aria**: Lorem ipsum.

                   If `add_heading` is `False`, then both become just::

                       Lorem ipsum.

    Returns the formatted message.
    """
    if add_heading:
        message_heading = chatutil.format_message_heading(message_number=message_number,
                                                          role=role,
                                                          persona=persona,
                                                          markup="markdown")
    else:
        message_heading = ""
    message_text = chatutil.remove_persona_from_start_of_line(persona=persona,
                                                              text=text)
    return f"{message_heading}{message_text}"

@memoize
def _get_all_system_prompt_node_ids(datastore: chattree.Forest) -> List[str]:
    """As it says on the tin.

    As of v0.2.4, there is just one system prompt node (which is dynamically updated at app startup),
    but this may change in the future, so we future-proof this by doing the semantically Right Thing:
    looking up all root nodes of the chat forest.

    Memoized; it's an O(n) search, which would need to run for each chat message widget created.
    The node ID of a system prompt node doesn't change once the node has been initially created.

    See also `_get_all_greeting_node_ids`.
    """
    system_prompt_node_ids = datastore.get_all_root_nodes()
    return system_prompt_node_ids

@memoize
def _get_all_greeting_node_ids(datastore: chattree.Forest) -> List[str]:
    """As it says on the tin.

    Since the AI's greeting can be changed in the config, the greeting used in any given stored chat
    is NOT necessarily the *current* greeting (`app_state["new_chat_HEAD"]`).

    Hence the reliable way to detect is to see whether the node a direct child node of a root node;
    any root node is a system prompt node.

    Same remarks as for `_get_all_system_prompt_node_ids`, which see.
    """
    system_prompt_node_ids = _get_all_system_prompt_node_ids(datastore=datastore)
    greeting_node_idss = [datastore.get_children(node_id) for node_id in system_prompt_node_ids]
    greeting_node_ids = flatten(greeting_node_idss)
    return greeting_node_ids

# --------------------------------------------------------------------------------

class DPGChatMessage:
    class_lock = threading.RLock()
    callbacks = {}

    @classmethod
    def run_callbacks(cls: type) -> None:
        with cls.class_lock:
            callbacks = list(cls.callbacks.items())
            for tag, function in callbacks:
                function()
                cls.callbacks.pop(tag)

    def __init__(self,
                 gui_parent: Union[str, int],
                 parent_view: "DPGLinearizedChatView"):
        """Base class for a chat message displayed in the linearized chat view.

        `gui_parent`: DPG tag or ID of the GUI widget (typically child window or group) to add the chat message to.
        `parent_view`: The linearized chat view widget this chat message is rendered in (and is owned by).
        """
        super().__init__()
        self.gui_parent = gui_parent  # GUI container to render in (DPG ID or tag)
        self.gui_uuid = str(uuid.uuid4())  # used in GUI widget tags
        self.gui_container_group = dpg.add_group(tag=f"chat_item_container_group_{self.gui_uuid}",
                                                 parent=self.gui_parent)
        self.parent_view = parent_view
        self.role = None  # populated by `build`
        self.persona = None  # populated by `build`
        self.paragraphs = []  # [{"text": ..., "rendered": True}, ...]
        self.paragraphs_lock = threading.RLock()
        self.node_id = None  # populated by `build`
        self.gui_text_group = None  # populated by `build`
        self.gui_button_callbacks = {}  # {name0: callable0, ...} - to trigger button features programmatically

        # for "delete subtree" confirmation (cannot be undone)
        self.last_delete_click_time = None
        self.confirm_duration = 2.0

    def _get_text(self) -> str:
        with self.paragraphs_lock:
            return "\n".join(paragraph["text"] for paragraph in self.paragraphs)
    text = property(fget=_get_text,
                    doc="Full text of this GUI chat message as `str`. Read-only.")

    def _get_next_or_prev_sibling_in_datastore(self,
                                               node_id: str,
                                               direction: str = "next",
                                               step: Optional[int] = 1) -> Optional[str]:
        """Get the next or previous sibling of `node_id` in the chat datastore.

        `direction`: One of "next", "prev".

        `step`: How many siblings to jump. Will jump up to as many as available in `direction`.
                Special value `None` means "jump to end" in the given `direction`.

        Returns the node ID of the sibling, or `None` if no such sibling.

        May return `node_id` itself.
        """
        siblings, this_node_index = self.parent_view.chat_controller.datastore.get_siblings(node_id)
        if siblings is None:  # can happen for root node
            return None
        if direction == "next":
            if step is None:  # jump to end
                return siblings[-1]
            elif this_node_index + step < len(siblings):
                return siblings[this_node_index + step]
            return siblings[-1]
        else:  # direction == "prev":
            if step is None:
                return siblings[0]
            elif this_node_index - step >= 0:
                return siblings[this_node_index - step]
            return siblings[0]

    def get_chat_text_width(self) -> int:
        """Get the current text wrap width of the chat."""
        w, h = guiutils.get_widget_size(self.parent_view.gui_parent)  # The view's GUI parent is the actual panel (DPG child window), whose width changes in a window resize.
        chat_text_w = w - gui_config.chat_text_right_margin_w
        return chat_text_w

    def build(self,
              role: str,
              persona: Optional[str],
              node_id: Optional[str]) -> None:
        """Build the GUI widgets for this chat message instance, thus rendering the chat message (and its buttons and such) in the GUI.

        `role`: One of the roles supported by `raven.librarian.llmclient`.
                Typically, one of "assistant", "system", "tool", or "user".

        `persona`: The persona name speaking `text`, or `None` if the role has no persona name ("system" and "tool" are like this).

                   To get the **current session's** persona, use::

                       persona=llm_settings.personas.get(role, None)

                   where `role` is one of "assistant", "system", "tool", "user".

                   To get the **stored** persona from a chat node::

                       persona=node_payload["general_metadata"]["persona"]

                   This may differ from the current session's persona, e.g. if the chat node was generated with a different AI character.

        `node_id`: The chat node ID of this message in the datastore, if applicable.

                   NOTE: Particularly, an incoming streaming message from the LLM does not have a node in the datastore.

        NOTE: You still need to `add_paragraph` the text you want to show in the chat message widget.

              We require explicit adding in order to be able to handle messages that *contain* thought blocks
              (i.e. any complete message from a thinking model), because the `is_thought` state (which is
              required when adding a paragraph) needs to be different for the think-block and final-message segments.

              The derived class `DPGCompleteChatMessage` automates this; it parses the content from a chat node,
              and adds the text to the widget.

              The derived class `DPGStreamingChatMessage`, on the other hand, requires full manual control, by design,
              so that the GUI driver handling the incoming message (`DPGChatController.ai_turn`) gets full control
              of what is displayed in the widget.
        """
        global role_to_colors  # intent only - we only read the color settings from this.

        self.role = role
        self.persona = persona
        self.node_id = node_id

        # clear old GUI content (needed if rebuilding)
        dpg.delete_item(self.gui_container_group, children_only=True)

        # --------------------------------------------------------------------------------
        # lay out the role icon and the text content areas horizontally

        icon_and_text_container_group = dpg.add_group(horizontal=True,
                                                      tag=f"chat_icon_and_text_container_group_{self.gui_uuid}",
                                                      parent=self.gui_container_group)

        # ----------------------------------------
        # role icon

        icon_drawlist = dpg.add_drawlist(width=(2 * gui_config.margin + gui_config.chat_icon_size),
                                         height=(2 * gui_config.margin + gui_config.chat_icon_size),
                                         tag=f"chat_icon_drawlist_{self.gui_uuid}",
                                         parent=icon_and_text_container_group)  # empty drawlist acts as placeholder if no icon
        if role in self.parent_view.chat_controller.gui_role_icons:
            dpg.draw_image(self.parent_view.chat_controller.gui_role_icons[role],
                           (gui_config.margin, gui_config.margin),
                           (gui_config.margin + gui_config.chat_icon_size, gui_config.margin + gui_config.chat_icon_size),
                           uv_min=(0, 0),
                           uv_max=(1, 1),
                           parent=icon_drawlist)

        # ----------------------------------------
        # text content

        # # colored border
        # dpg.add_drawlist(width=4,
        #                  height=4,  # to be updated after the text is rendered
        #                  tag=f"chat_colored_border_drawlist_{self.gui_uuid}",
        #                  parent=icon_and_text_container_group)

        # adjust text vertical positioning
        text_vertical_layout_group = dpg.add_group(tag=f"chat_message_vertical_layout_group_{self.gui_uuid}",
                                                   parent=icon_and_text_container_group)
        dpg.add_spacer(height=gui_config.margin,
                       parent=text_vertical_layout_group)

        # Render timestamp the revision number of the payload currently shown  TODO: later (chat editing): this needs to be switchable without regenerating the whole view
        if node_id is not None:
            node_payload = self.parent_view.chat_controller.datastore.get_payload(node_id)  # auto-selects active revision  TODO: later (chat editing), we need to set the revision to load
            payload_datetime = node_payload["general_metadata"]["datetime"]  # of the active payload revision!
            node_active_revision = self.parent_view.chat_controller.datastore.get_revision(node_id)
            dpg.add_text(f"{payload_datetime} R{node_active_revision}", color=(120, 120, 120), parent=text_vertical_layout_group)

        # render the actual text
        self.gui_text_group = dpg.add_group(tag=f"chat_message_text_container_group_{self.gui_uuid}",
                                            parent=text_vertical_layout_group)  # create another group to act as container so that we can update/replace just the text easily
        # NOTE: We now have an empty group, for `add_paragraph`/`replace_last_paragraph`.

        # Show LLM performance statistics for AI chat node, if linked to a chat node, and the chat node has them stored
        if role == "assistant" and node_id is not None:
            ai_message_node_payload = self.parent_view.chat_controller.datastore.get_payload(node_id)
            if (generation_metadata := ai_message_node_payload.get("generation_metadata", None)) is not None:
                n_tokens = generation_metadata["n_tokens"]
                dt = generation_metadata["dt"]
                speed = n_tokens / dt
                dpg.add_text(f"[{n_tokens}t, {dt:0.2f}s, {speed:0.2f}t/s]",
                             color=(120, 120, 120),
                             parent=text_vertical_layout_group)

        # If there is no linked chat node, this is a live streaming chat message, so the GUI widget should end here - it doesn't need the datastore control buttons or end spacers.
        # This makes the GUI look calmer while rendering a streaming message.
        if node_id is None:
            return

        # text area end spacer
        dpg.add_spacer(height=2,
                       parent=text_vertical_layout_group)

        # ----------------------------------------
        # buttons (below text)

        buttons_horizontal_layout_group = dpg.add_group(horizontal=True,
                                                        tag=f"chat_buttons_container_group_{self.gui_uuid}",
                                                        parent=text_vertical_layout_group)
        number_of_message_buttons = 14
        chat_text_w = self.get_chat_text_width()
        dpg.add_spacer(width=chat_text_w - number_of_message_buttons * (gui_config.toolbutton_w + 8) - 64,  # 8 = DPG outer margin; 64 = some space for sibling counter
                       parent=buttons_horizontal_layout_group)

        self.build_buttons(gui_parent=buttons_horizontal_layout_group)

        # ----------------------------------------
        # chat turn end spacers and line

        dpg.add_spacer(height=4,
                       tag=f"chat_turn_end_spacer1_{self.gui_uuid}",
                       parent=self.gui_container_group)

        if role in role_to_colors:
            dpg.add_drawlist(height=1,
                             width=(chat_text_w + 64),
                             tag=f"chat_turn_end_drawlist_{self.gui_uuid}",
                             parent=self.gui_container_group)
            dpg.draw_rectangle((64, 0), (chat_text_w + 64, 1),
                               color=(80, 80, 80),
                               fill=(80, 80, 80),
                               parent=f"chat_turn_end_drawlist_{self.gui_uuid}")  # tag

        dpg.add_spacer(height=4,
                       tag=f"chat_turn_end_spacer2_{self.gui_uuid}",
                       parent=self.gui_container_group)

    def add_paragraph(self, text: str, is_thought: bool) -> None:
        """Add a new paragraph of text to this widget.

        `is_thought`: Whether this paragraph is (part of) a `<think>...</think>` block.
                      The renderer selects the text color appropriately.
        """
        paragraph = {"text": text,
                     "is_thought": is_thought,
                     "rendered": False}
        with self.paragraphs_lock:
            self.paragraphs.append(paragraph)
            self._render_text()

    def replace_last_paragraph(self, text: str, is_thought: bool) -> None:  # TODO: Only last paragraph is replaceable for now, because it's easier for coding the GUI. :)
        """Replace the last paragraph of text in this widget. If there are no paragraphs yet, create one automatically.

       `is_thought`: Whether this paragraph is (part of) a `<think>...</think>` block.
                     The renderer selects the text color appropriately.

                     If needed, can be different from the old state of the same paragraph.
         """
        with self.paragraphs_lock:
            if not self.paragraphs:
                self.add_paragraph(text, is_thought)
                return
            paragraph = self.paragraphs[-1]

            # The mutex guarantees this section runs in the same frame.
            #     https://github.com/hoffstadt/DearPyGui/discussions/1002
            # TODO: Grabbing the mutex here causes the app to randomly hang during `on_llm_progress`. Debug why. Just disabling this for now.
            # with dpg.mutex():
            if "widget" in paragraph:
                dpg.delete_item(paragraph.pop("widget"))
            paragraph["text"] = text
            paragraph["is_thought"] = is_thought
            paragraph["rendered"] = False
            self._render_text()

        dpg.split_frame()  # ...and anything after this point runs in another frame.

    def _render_text(self) -> None:
        """Internal method. Render any pending new paragraphs. We assume new paragraphs are added only to the end."""
        with self.paragraphs_lock:
            if self.gui_text_group is None:
                assert False  # the chat message GUI widget did not fully initialize
            # dpg.delete_item(self.gui_text_group, children_only=True)  # how to clear all old text if we ever need to
            role = self.role
            role_color = role_to_colors[role]["front"] if role in role_to_colors else "#ffffff"
            think_color = librarian_config.gui_config.chat_color_think_front
            for idx, paragraph in enumerate(self.paragraphs):
                if paragraph["rendered"]:
                    continue
                assert "widget" not in paragraph  # a paragraph that hasn't been rendered has no GUI text widget associated with it
                text = paragraph["text"].strip()
                if text:  # don't bother if text is blank
                    # Replace known XML tokens with something that doesn't look like HTML to avoid confusing the Markdown renderer (which silently drops unknown tags).
                    text = text.replace("<tool_call>", "**>>>Tool call>>>**")
                    text = text.replace("</tool_call>", "**<<<Tool call<<<**")
                    text = text.replace("<think>", "**>>>Thinking>>>**")
                    text = text.replace("</think>", "**<<<Thinking<<<**")
                    color = think_color if paragraph["is_thought"] else role_color
                    colorized_text = f"<font color='{color}'>{text}</font>"

                    chat_text_w = self.get_chat_text_width()

                    if isinstance(self, DPGCompleteChatMessage) and paragraph["is_thought"]:  # make think blocks in complete messages collapsible (they are populated as a single paragraph)
                        widget = dpg.add_group(horizontal=True, parent=self.gui_text_group)
                        def toggle_message_think_callback():
                            with guiutils.nonexistent_ok() as nok:
                                v = dpg.is_item_visible(text_content)
                                if v:
                                    logger.info(f"DPGCompleteChatMessage._render_text.toggle_message_think_callback: hiding thinking trace for chat node '{self.node_id}'")
                                    dpg.hide_item(text_content)
                                else:
                                    logger.info(f"DPGCompleteChatMessage._render_text.toggle_message_think_callback: showing thinking trace for chat node '{self.node_id}'")
                                    dpg.show_item(text_content)
                            if nok.errored:
                                logger.info(f"DPGCompleteChatMessage._render_text.toggle_message_think_callback: GUI widget for chat node '{self.node_id}' does not exist, ignoring.")
                        self.gui_button_callbacks["toggle_thinking_trace"] = toggle_message_think_callback  # stash it so we can call it from the hotkey handler
                        dpg.add_button(label=fa.ICON_CLOUD,
                                       callback=toggle_message_think_callback,
                                       width=gui_config.toolbutton_w,
                                       tag=f"message_think_toggle_button_{self.gui_uuid}",
                                       parent=widget)
                        dpg.bind_item_font(f"message_think_toggle_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
                        # dpg.bind_item_theme(f"message_think_toggle_button_{self.gui_uuid}", "disablable_blue_widget_theme")  # tag
                        message_think_toggle_tooltip = dpg.add_tooltip(f"message_think_toggle_button_{self.gui_uuid}")  # tag
                        dpg.add_text("Show/hide thinking trace [Ctrl+T]", parent=message_think_toggle_tooltip)
                        text_content = dpg_markdown.add_text(colorized_text,
                                                             wrap=chat_text_w,
                                                             parent=widget)
                        dpg.hide_item(text_content)
                    else:
                        widget = dpg_markdown.add_text(colorized_text,
                                                       wrap=chat_text_w,
                                                       parent=self.gui_text_group)
                    paragraph["widget"] = widget
                    dpg.set_item_alias(widget, f"chat_message_text_{role}_paragraph_{idx}_{self.gui_uuid}")  # tag
                paragraph["rendered"] = True

    def add_tool_call_invocation(self, index: int, name: str, arguments: str) -> None:
        """Render one tool-call invocation as a visible sub-element: a meshing-cogs icon + the call signature.

        Raven's what-you-see-is-what-you-get design surfaces what the model did, so a tool-calling turn is not
        silently swallowed between an (often empty) assistant message and the subsequent tool result. The
        invocation may have arrived as a native `tool_calls` entry or as an inline `<tool_call>` tag — by the
        time it reaches here it's the same structured form (the `invoke` parser unified them).

        The icon is `ICON_GEARS` (meshing cogs), matching the tool-role result message's three-cogs badge
        (`icons/tool.png`) — invocation and result read as the same family. Deliberately *not* the single-gear
        `ICON_GEAR`, which is the universal "settings" glyph (reserved for the future settings dialog).

        `index`: position among this message's tool calls (for unique widget tags).
        `name`: the function name.
        `arguments`: the call arguments as a JSON string (OAI convention).
        """
        tool_color = role_to_colors["tool"]["front"]
        try:
            parsed_args = json.loads(arguments) if arguments else {}
        except (json.JSONDecodeError, ValueError):
            parsed_args = None
        if isinstance(parsed_args, dict):
            signature = ", ".join(f"{key}={value!r}" for key, value in parsed_args.items())
        else:  # non-dict / unparseable: show the raw arguments rather than nothing
            signature = (arguments or "").strip()

        with self.paragraphs_lock:
            row = dpg.add_group(horizontal=True, parent=self.gui_text_group)
            icon_tag = f"chat_message_toolcall_icon_{index}_{self.gui_uuid}"  # tag
            dpg.add_text(fa.ICON_GEARS, color=tool_color, tag=icon_tag, parent=row)  # tag
            dpg.bind_item_font(icon_tag, self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.add_text(f"{name}({signature})",
                         color=tool_color,
                         wrap=max(0, self.get_chat_text_width() - 40),  # leave room for the leading icon
                         parent=row)

    def demolish(self) -> None:
        """The opposite of `build`: delete all GUI widgets belonging to this instance.

        If you use `DPGLinearizedChatView.build`, it takes care of clearing all old chat message GUI widgets automatically,
        and you do not need to call this.

        If you are editing the GUI contents of the linearized chat view directly, this should be called before deleting
        the `DPGChatMessage` (or a derived class) instance.

        The main use case is switching a streaming message to a completed one when the streaming is done,
        without regenerating the whole linearized chat view (which may contain a lot of messages).
        """
        with self.paragraphs_lock:
            self.role = None
            self.persona = None
            self.paragraphs = []
            self.gui_text_group = None
            self.gui_button_callbacks = {}  # deleting all GUI widgets, so clear the stashed callbacks too.
            with guiutils.nonexistent_ok():
                dpg.delete_item(self.gui_container_group, children_only=True)  # clear old GUI content (needed if rebuilding)

    def build_buttons(self,
                      gui_parent: Union[str, int]) -> None:
        """Build the set of control buttons for a single chat message in the GUI.

        `gui_parent`: DPG tag or ID of the GUI widget (typically a group) to add the buttons to.

                      This is not simply `self.gui_parent` due to other layout performed by `build`;
                      the buttons go into a group.
        """
        # NOTE: If you add or remove buttons here, update also `number_of_message_buttons` (search for it in this module).

        role = self.role
        persona = self.persona
        node_id = self.node_id

        g = dpg.add_group(horizontal=True, tag=f"{role}_message_buttons_group_{self.gui_uuid}", parent=gui_parent)

        # dpg.add_spacer(tag=f"ai_message_buttons_spacer_{self.gui_uuid}",
        #                parent=g)

        def copy_message_to_clipboard_callback() -> None:
            shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
            # Note we only add the role name when we include also the node ID.
            # Omitting the speaker's name in regular mode improves convenience for copy-pasting an existing question into the chat field (to slightly modify it before re-submitting).
            formatted_message = format_chat_message_for_clipboard(message_number=None,  # a single message copied to clipboard does not need a sequential number
                                                                  role=role,
                                                                  persona=persona,
                                                                  text=self.text,
                                                                  add_heading=shift_pressed)

            if shift_pressed:
                node_payload = self.parent_view.chat_controller.datastore.get_payload(node_id)  # auto-selects active revision  TODO: later (chat editing), we need to set the revision to load
                payload_datetime = node_payload["general_metadata"]["datetime"]  # of the active payload revision!
                node_active_revision = self.parent_view.chat_controller.datastore.get_revision(node_id)
                header = f"*Node ID*: `{node_id}` {payload_datetime} R{node_active_revision}\n\n"  # yes, it'll say `None` when no node ID is available (incoming streaming message), which is exactly what we want.
            else:
                header = ""
            mode = "with node ID" if shift_pressed else "as-is"
            dpg.set_clipboard_text(f"{header}{formatted_message}\n")
            # Acknowledge the action in the GUI.
            gui_animation.animator.add(gui_animation.ButtonFlash(message=f"Copied to clipboard! ({mode})",
                                                                 target_button=copy_message_button,
                                                                 target_tooltip=copy_message_tooltip,
                                                                 target_text=copy_message_tooltip_text,
                                                                 original_theme=dpg.get_item_theme(copy_message_tooltip),
                                                                 duration=gui_config.acknowledgment_duration))
        self.gui_button_callbacks["copy"] = copy_message_to_clipboard_callback
        copy_message_button = dpg.add_button(label=fa.ICON_COPY,
                                             callback=copy_message_to_clipboard_callback,
                                             width=gui_config.toolbutton_w,
                                             tag=f"message_copy_to_clipboard_button_{self.gui_uuid}",
                                             parent=g)
        dpg.bind_item_font(copy_message_button, self.parent_view.themes_and_fonts.icon_font_solid)
        dpg.bind_item_theme(copy_message_button, "disablable_widget_theme")  # tag
        copy_message_tooltip = dpg.add_tooltip(copy_message_button)
        copy_message_tooltip_text = dpg.add_text("Copy message to clipboard\n    no modifier: as-is\n    with Shift: include message node ID", parent=copy_message_tooltip)

        # These are needed for enabling/disabling some buttons.
        system_prompt_node_ids = _get_all_system_prompt_node_ids(datastore=self.parent_view.chat_controller.datastore)
        greeting_node_ids = _get_all_greeting_node_ids(datastore=self.parent_view.chat_controller.datastore)

        # Rerolling for AI messages
        if role == "assistant":
            def reroll_message_callback():
                # Find this AI message in the chat history
                for k, dpg_chat_message in enumerate(reversed(self.parent_view.chat_controller.current_chat_history)):
                    if dpg_chat_message.node_id == node_id:
                        break
                else:  # not found
                    return
                # `k` is now how many messages must be popped from the end to reach this one
                assert k < len(self.parent_view.chat_controller.current_chat_history) - 3  # should have at least the system prompt, the AI's initial greeting, and the user's first message remaining
                # Rewind the linearized chat history in the GUI
                for _ in range(k):
                    old_dpg_chat_message = self.parent_view.chat_controller.current_chat_history.pop(-1)
                    old_dpg_chat_message.demolish()

                # Handle the RAG query: find the latest user message (above this AI message)
                user_message_text = None
                for dpg_chat_message in reversed(self.parent_view.chat_controller.current_chat_history):  # ...what's remaining of the history
                    if dpg_chat_message.role == "user":
                        user_message_text = dpg_chat_message.text
                        break

                # Remove the AI message from GUI
                self.parent_view.chat_controller.app_state["HEAD"] = self.parent_view.chat_controller.datastore.get_parent(node_id)
                old_dpg_chat_message = self.parent_view.chat_controller.current_chat_history.pop(-1)  # once more, with feeling!
                old_dpg_chat_message.demolish()

                # Generate new AI message
                self.parent_view.chat_controller.ai_turn(docs_query=user_message_text,
                                                         continue_=False)
            reroll_enabled = ((node_id is not None) and (node_id not in greeting_node_ids))  # The AI's initial greeting can't be rerolled
            if reroll_enabled:
                self.gui_button_callbacks["reroll"] = reroll_message_callback  # stash it so we can call it from the hotkey handler
            dpg.add_button(label=fa.ICON_DICE_D20,  # fa.ICON_RECYCLE,
                           callback=reroll_message_callback,
                           enabled=reroll_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_reroll_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_reroll_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_reroll_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
            reroll_tooltip = dpg.add_tooltip(f"message_reroll_button_{self.gui_uuid}")  # tag
            dpg.add_text("Reroll on a new branch [Ctrl+R]", parent=reroll_tooltip)
        else:
            dpg.add_spacer(width=gui_config.toolbutton_w, height=1, parent=g)

        if role == "assistant":
            def continue_message_callback():
                dpg_chat_message = self.parent_view.chat_controller.current_chat_history[-1]  # latest message
                if dpg_chat_message.node_id != node_id:  # latest message is not this message --> can't continue
                    return

                # Handle the RAG query: find the latest user message (above this AI message)
                user_message_text = None
                for dpg_chat_message in reversed(self.parent_view.chat_controller.current_chat_history):
                    if dpg_chat_message.role == "user":
                        user_message_text = dpg_chat_message.text
                        break

                # Continue the AI message
                self.parent_view.chat_controller.ai_turn(docs_query=user_message_text,
                                                         continue_=True)
                # No button flash, because the button will be deleted immediately, when the chat message widget is replaced.
            # We should enable continue only for the last message, but when we get here, this message isn't in the view yet.
            # We currently solve this by disabling continue buttons for old messages, from the outside, once we're done rendering the view.
            continue_enabled = ((node_id is not None) and (node_id not in greeting_node_ids))  # The AI's initial greeting can't be continued
            if continue_enabled:
                self.gui_button_callbacks["continue"] = continue_message_callback  # stash it so we can call it from the hotkey handler
            dpg.add_button(label=fa.ICON_PARAGRAPH,  # fa.ICON_RIGHT_LONG,  # fa.ICON_ARROW_RIGHT,
                           callback=continue_message_callback,
                           enabled=continue_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_continue_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_continue_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_continue_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
            continue_message_tooltip = dpg.add_tooltip(f"message_continue_button_{self.gui_uuid}")  # tag
            dpg.add_text("Ask the AI to continue this response (create new revision) [Ctrl+U]", parent=continue_message_tooltip)
        else:
            dpg.add_spacer(width=gui_config.toolbutton_w, height=1, parent=g)

        # TTS for AI messages
        if role == "assistant":
            def speak_message_callback():
                if self.parent_view.chat_controller.app_state["avatar_speech_enabled"]:
                    self.parent_view.chat_controller.avatar_controller.ping(config=self.parent_view.chat_controller.avatar_record)  # wake up the AI avatar before starting to speak
                    unused_message_role, message_persona, message_text = chatutil.get_node_message_text_without_persona(self.parent_view.chat_controller.datastore, node_id)
                    # Send only non-thought message content to TTS
                    message_text = chatutil.scrub(persona=message_persona,
                                                  text=message_text,
                                                  thoughts_mode="discard",
                                                  markup=None,
                                                  add_persona=False)
                    self.parent_view.chat_controller.avatar_controller.send_text_to_tts(config=self.parent_view.chat_controller.avatar_record,
                                                                                        text=message_text,
                                                                                        video_offset=librarian_config.avatar_config.video_offset)

                    # Acknowledge the action in the GUI.
                    gui_animation.animator.add(gui_animation.ButtonFlash(message="Sent to avatar!",
                                                                         target_button=speak_message_button,
                                                                         target_tooltip=speak_message_tooltip,
                                                                         target_text=speak_message_tooltip_text,
                                                                         original_theme=dpg.get_item_theme(speak_message_tooltip),
                                                                         duration=gui_config.acknowledgment_duration))
            speak_enabled = (role == "assistant")
            if speak_enabled:
                self.gui_button_callbacks["speak"] = speak_message_callback
            speak_message_button = dpg.add_button(label=fa.ICON_COMMENT,
                                                  callback=speak_message_callback,
                                                  enabled=speak_enabled,
                                                  width=gui_config.toolbutton_w,
                                                  tag=f"chat_speak_button_{self.gui_uuid}",
                                                  parent=g)
            dpg.bind_item_font(speak_message_button, self.parent_view.themes_and_fonts.icon_font_solid)
            dpg.bind_item_theme(speak_message_button, "disablable_widget_theme")  # tag
            speak_message_tooltip = dpg.add_tooltip(speak_message_button)
            speak_message_tooltip_text = dpg.add_text("Have the avatar speak this message [Ctrl+S]", parent=speak_message_tooltip)
        else:
            dpg.add_spacer(width=gui_config.toolbutton_w, height=1, parent=g)

        dpg.add_button(label=fa.ICON_PENCIL,
                       callback=lambda: None,  # TODO
                       enabled=False,
                       width=gui_config.toolbutton_w,
                       tag=f"chat_edit_button_{self.gui_uuid}",
                       parent=g)
        dpg.bind_item_font(f"chat_edit_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
        dpg.bind_item_theme(f"chat_edit_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
        edit_tooltip = dpg.add_tooltip(f"chat_edit_button_{self.gui_uuid}")  # tag
        dpg.add_text("Edit (revise)", parent=edit_tooltip)

        # Branch chat at this node
        # NOTE: We disallow branching from the system prompt, as well as from any message that is not linked to a chat node in the datastore.
        #       We also disallow using the "branch from here" feature on the AI's initial greeting, as that's an unnecessarily confusing way to say "start new chat".
        branch_enabled = ((node_id is not None) and
                          (node_id not in system_prompt_node_ids) and
                          (node_id not in greeting_node_ids))
        def branch_chat_callback():
            self.parent_view.chat_controller.app_state["HEAD"] = node_id
            self.parent_view.build()
        dpg.add_button(label=fa.ICON_CODE_BRANCH,
                       callback=branch_chat_callback,
                       enabled=branch_enabled,
                       width=gui_config.toolbutton_w,
                       tag=f"message_new_branch_button_{self.gui_uuid}",
                       parent=g)
        dpg.bind_item_font(f"message_new_branch_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
        dpg.bind_item_theme(f"message_new_branch_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
        new_branch_tooltip = dpg.add_tooltip(f"message_new_branch_button_{self.gui_uuid}")  # tag
        dpg.add_text("Branch from this node", parent=new_branch_tooltip)

        # Delete subtree starting from this node (requires a confirmation click)
        #
        # NOTE: We disallow deleting the system prompt and the AI's initial greeting, as well as any message that is not linked to a chat node in the datastore.
        delete_enabled = ((node_id is not None) and
                          (node_id not in system_prompt_node_ids) and
                          (node_id not in greeting_node_ids))
        def delete_subtree_callback():
            current_time = time.monotonic_ns()
            if self.last_delete_click_time is not None:
                double_okd = (current_time - self.last_delete_click_time < self.confirm_duration * 10**9)
            else:
                double_okd = False
            self.last_delete_click_time = current_time

            if double_okd:  # perform delete
                # Find which node to switch HEAD to after delete.
                #   - Switch to previous sibling, or if this was the first one, then the next one.
                #   - Switch to parent if no siblings remaining after delete.
                siblings, this_node_index = self.parent_view.chat_controller.datastore.get_siblings(node_id)
                assert len(siblings) >= 1  # should always have at least the node itself
                if len(siblings) == 1:  # no remaining siblings after delete --> set parent as HEAD
                    new_HEAD = self.parent_view.chat_controller.datastore.get_parent(node_id)
                # now `len(siblings) > 1`
                elif this_node_index == 0:
                    new_HEAD = siblings[1]
                # now `this_node_index > 0`
                else:
                    new_HEAD = siblings[this_node_index - 1]

                # Perform the delete
                self.parent_view.chat_controller.datastore.delete_subtree(node_id)

                # Refresh view
                self.parent_view.chat_controller.app_state["HEAD"] = new_HEAD
                self.parent_view.build()
            else:
                gui_animation.animator.add(gui_animation.ButtonFlash(message="Press again to confirm.\nDeletion CANNOT BE UNDONE.",
                                                                     target_button=delete_subtree_button,
                                                                     target_tooltip=delete_subtree_tooltip,
                                                                     target_text=delete_subtree_tooltip_text,
                                                                     original_theme=dpg.get_item_theme(delete_subtree_tooltip),
                                                                     flash_color=(255, 32, 32),  # orange for warning
                                                                     text_color=(255, 255, 255),
                                                                     duration=self.confirm_duration))
        delete_subtree_button = dpg.add_button(label=fa.ICON_TRASH_CAN,
                                               callback=delete_subtree_callback,
                                               enabled=delete_enabled,
                                               width=gui_config.toolbutton_w,
                                               tag=f"message_delete_branch_button_{self.gui_uuid}",
                                               parent=g)
        dpg.bind_item_font(f"message_delete_branch_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
        dpg.bind_item_theme(f"message_delete_branch_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
        delete_subtree_tooltip = dpg.add_tooltip(f"message_delete_branch_button_{self.gui_uuid}")  # tag
        delete_subtree_tooltip_text = dpg.add_text("Delete branch (subtree starting from this node, ALL descendants!)", parent=delete_subtree_tooltip)

        # "Approve denied host & retry" override. Appears ONLY on a webfetch tool result that the client-side
        # allowlist refused (such a node carries `webfetch_denied_host` in its generation_metadata, set by
        # `llmclient.webfetch_wrapper`). Clicking it approves the host for this session and re-runs that one
        # fetch on a new branch — see `scaffold.retry_tool_calls`.
        #
        # This is a conditional, rare button, so it is intentionally NOT counted in `number_of_message_buttons`
        # (bumping that would add left margin to EVERY message row for a button almost never shown). The cost:
        # the leading right-align spacer reserves space for the fixed button count, so the extra button pushes
        # the sibling counter ("1 / 2") further right and possibly off-view on a denied tool row. Acceptable
        # for this provisional affordance; revisit if/when it gets a permanent home.
        #
        # NOTE: provisional placement. Brief 03 (content-parts) moves tool-result rendering into the assistant
        # message body; when that lands, this affordance relocates there. See briefs/summer_2026_librarian_extension/.
        maybe_denied_host = None
        if role == "tool" and node_id is not None:
            denied_node_payload = self.parent_view.chat_controller.datastore.get_payload(node_id)
            maybe_denied_host = denied_node_payload.get("generation_metadata", {}).get("webfetch_denied_host")
        if maybe_denied_host is not None:
            def approve_and_retry_callback():
                chat_controller = self.parent_view.chat_controller
                llmclient.approve_host_for_session(maybe_denied_host)

                # Rewind the GUI to the branch point: pop every message after the denied tool result, then the
                # denied result itself. `retry_tool_calls` re-adds the new branch via the ai_turn callbacks.
                for k, dpg_chat_message in enumerate(reversed(chat_controller.current_chat_history)):
                    if dpg_chat_message.node_id == node_id:
                        break
                else:  # not found (shouldn't happen — the button lives on this message)
                    return
                for _ in range(k + 1):  # +1 to also pop the denied tool result itself
                    old_dpg_chat_message = chat_controller.current_chat_history.pop(-1)
                    old_dpg_chat_message.demolish()

                # Re-run the denied fetch on a new branch and continue. HEAD is updated by the callbacks.
                chat_controller.ai_turn(docs_query=None,
                                        continue_=False,
                                        _retry_tool_node_id=node_id)
            approve_retry_button = dpg.add_button(label=fa.ICON_UNLOCK,
                                                  callback=approve_and_retry_callback,
                                                  width=gui_config.toolbutton_w,
                                                  tag=f"message_approve_retry_button_{self.gui_uuid}",  # tag
                                                  parent=g)
            dpg.bind_item_font(approve_retry_button, self.parent_view.themes_and_fonts.icon_font_solid)
            dpg.bind_item_theme(approve_retry_button, "disablable_widget_theme")  # tag
            approve_retry_tooltip = dpg.add_tooltip(approve_retry_button)
            dpg.add_text(f"Approve host '{maybe_denied_host}' for this session, and retry the fetch (on a new branch)", parent=approve_retry_tooltip)

        # # TODO: Meh, `raven.common.gui.animation.ButtonFlash` doesn't play together with `dpg_markdown`.
        # c_red = '<font color="(255, 96, 96)">'
        # c_end = '</font>'
        # delete_subtree_tooltip_text = dpg_markdown.add_text(f"Delete branch (this node and {c_red}**all**{c_end} descendants!)", parent=delete_subtree_tooltip)

        datastore = self.parent_view.chat_controller.datastore
        def descend(start_node_id: str) -> str:
            node_ids = datastore.get_children(start_node_id)
            if not node_ids:
                return start_node_id
            payloads = [datastore.get_payload(node_id) for node_id in node_ids]
            timestamps = [payload["general_metadata"]["timestamp"] for payload in payloads]
            idx = np.argmax(timestamps)
            return descend(node_ids[idx])
        def make_navigate_to_sibling(message_node_id: str, direction: str, step: Optional[int]) -> Callable:
            # Pick the most recent subtree, greedily
            def navigate_to_sibling_callback():
                node_id = self._get_next_or_prev_sibling_in_datastore(message_node_id,
                                                                      direction=direction,
                                                                      step=step)
                if node_id is not None:
                    head_node_id = descend(node_id)
                    self.parent_view.chat_controller.app_state["HEAD"] = head_node_id
                    self.parent_view.build(scroll_target_node_id=node_id)
            return navigate_to_sibling_callback
        def make_show_chat_continuation(message_node_id: str) -> Callable:
            def show_chat_continuation_callback():
                head_node_id = descend(message_node_id)
                if head_node_id is not None:
                    self.parent_view.chat_controller.app_state["HEAD"] = head_node_id
                    self.parent_view.build()  # let it scroll to end
            return show_chat_continuation_callback

        # Only messages attached to a datastore chat node can have siblings or a chat continuation in the datastore
        if node_id is not None:
            siblings, this_node_index = self.parent_view.chat_controller.datastore.get_siblings(node_id)
            prev_enabled = (this_node_index is not None and this_node_index - 1 >= 0)
            next_enabled = (this_node_index is not None and this_node_index + 1 <= len(siblings) - 1)
            navigate_to_prev1_callback = make_navigate_to_sibling(node_id, direction="prev", step=1)
            navigate_to_next1_callback = make_navigate_to_sibling(node_id, direction="next", step=1)
            navigate_to_prev10_callback = make_navigate_to_sibling(node_id, direction="prev", step=10)
            navigate_to_next10_callback = make_navigate_to_sibling(node_id, direction="next", step=10)
            navigate_to_prevend_callback = make_navigate_to_sibling(node_id, direction="prev", step=None)
            navigate_to_nextend_callback = make_navigate_to_sibling(node_id, direction="next", step=None)
            if prev_enabled:
                self.gui_button_callbacks["prev1"] = navigate_to_prev1_callback
                self.gui_button_callbacks["prev10"] = navigate_to_prev10_callback
                self.gui_button_callbacks["prevend"] = navigate_to_prevend_callback
            if next_enabled:
                self.gui_button_callbacks["next1"] = navigate_to_next1_callback
                self.gui_button_callbacks["next10"] = navigate_to_next10_callback
                self.gui_button_callbacks["nextend"] = navigate_to_nextend_callback

            children = self.parent_view.chat_controller.datastore.get_children(node_id)
            show_chat_continuation_enabled = (len(children) > 0)
            show_chat_continuation_callback = make_show_chat_continuation(node_id)
            if show_chat_continuation_enabled:
                self.gui_button_callbacks["show_chat_continuation"] = show_chat_continuation_callback

            dpg.add_button(label=fa.ICON_BACKWARD_FAST,
                           callback=navigate_to_prevend_callback,
                           enabled=prev_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_prevend_branch_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_prevend_branch_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_prevend_branch_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
            prevend_branch_tooltip = dpg.add_tooltip(f"message_prevend_branch_button_{self.gui_uuid}")  # tag
            dpg.add_text("Switch to first sibling", parent=prevend_branch_tooltip)

            dpg.add_button(label=fa.ICON_BACKWARD,
                           callback=navigate_to_prev10_callback,
                           enabled=prev_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_prev10_branch_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_prev10_branch_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_prev10_branch_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
            prev10_branch_tooltip = dpg.add_tooltip(f"message_prev10_branch_button_{self.gui_uuid}")  # tag
            dpg.add_text("Switch 10 siblings left [Ctrl+Shift+Left]", parent=prev10_branch_tooltip)

            dpg.add_button(label=fa.ICON_CARET_LEFT,
                           callback=navigate_to_prev1_callback,
                           enabled=prev_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_prev1_branch_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_prev1_branch_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_prev1_branch_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
            prev1_branch_tooltip = dpg.add_tooltip(f"message_prev1_branch_button_{self.gui_uuid}")  # tag
            dpg.add_text("Switch to previous sibling [Ctrl+Left]", parent=prev1_branch_tooltip)

            dpg.add_button(label=fa.ICON_CARET_DOWN,
                           callback=show_chat_continuation_callback,
                           enabled=show_chat_continuation_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_show_chat_continuation_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_show_chat_continuation_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_show_chat_continuation_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
            show_chat_continuation_tooltip = dpg.add_tooltip(f"message_show_chat_continuation_button_{self.gui_uuid}")  # tag
            dpg.add_text("Show chat continuation (if any) [Ctrl+Down]", parent=show_chat_continuation_tooltip)

            dpg.add_button(label=fa.ICON_CARET_RIGHT,
                           callback=navigate_to_next1_callback,
                           enabled=next_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_next1_branch_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_next1_branch_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_next1_branch_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
            next1_branch_tooltip = dpg.add_tooltip(f"message_next1_branch_button_{self.gui_uuid}")  # tag
            dpg.add_text("Switch to next sibling [Ctrl+Right]", parent=next1_branch_tooltip)

            dpg.add_button(label=fa.ICON_FORWARD,
                           callback=navigate_to_next10_callback,
                           enabled=next_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_next10_branch_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_next10_branch_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_next10_branch_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
            next10_branch_tooltip = dpg.add_tooltip(f"message_next10_branch_button_{self.gui_uuid}")  # tag
            dpg.add_text("Switch 10 siblings right [Ctrl+Shift+Right]", parent=next10_branch_tooltip)

            dpg.add_button(label=fa.ICON_FORWARD_FAST,
                           callback=navigate_to_nextend_callback,
                           enabled=next_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_nextend_branch_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_nextend_branch_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_nextend_branch_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
            nextend_branch_tooltip = dpg.add_tooltip(f"message_nextend_branch_button_{self.gui_uuid}")  # tag
            dpg.add_text("Switch to last sibling", parent=nextend_branch_tooltip)

            if siblings is not None:
                dpg.add_text(f"{this_node_index + 1} / {len(siblings)}", parent=g)
        else:
            # Add the spacers separately so we get the same margins as with separate buttons
            for _ in range(6):
                dpg.add_spacer(width=gui_config.toolbutton_w, height=1, parent=g)


class DPGCompleteChatMessage(DPGChatMessage):
    def __init__(self,
                 node_id: str,
                 gui_parent: Union[str, int],
                 parent_view: "DPGLinearizedChatView"):
        """A complete chat message displayed in the linearized chat view, linked to a node ID in the datastore.

        `node_id`: The ID of the chat node, in the datastore, from which to extract the data to show.
        `gui_parent`: DPG tag or ID of the GUI widget (typically child window or group) to add the chat message to.
        `parent_view`: The linearized chat view widget this chat message is rendered in (and is owned by).
        """
        super().__init__(gui_parent=gui_parent,
                         parent_view=parent_view)
        self.node_id = node_id  # reference to the chat node (to ORIGINAL node data, not a copy)
        self.build()

    def build(self) -> None:
        """Build (or rebuild) the GUI widgets for this chat message.

        Automatically parse the content from the chat node, and add the text to the GUI.
        """
        node_payload = self.parent_view.chat_controller.datastore.get_payload(self.node_id)  # auto-selects active revision  TODO: later (chat editing), we need to set the revision to load
        message = node_payload["message"]
        role = message["role"]
        persona = node_payload["general_metadata"]["persona"]  # stored persona for this chat message
        sidecars_meta = node_payload["general_metadata"].get("sidecars", {})  # provenance per image sidecar (see imagestore)
        super().build(role=role,
                      persona=persona,
                      node_id=self.node_id)

        # Reasoning (thinking) trace lives in the message's `reasoning_content` sibling field, not in `content`.
        # Render it first, as a single collapsible thought paragraph. Migration (`upgrade_datastore`, at load)
        # and the live stream parser both move thinking into `reasoning_content` before it ever reaches here, so
        # `content` no longer carries inline `<think>`. The per-part splitter below still recognizes inline
        # `<think>`, but that path is dead — leftover from the pre-June-2026 inline handling, not yet removed.
        reasoning_content = message.get("reasoning_content") or ""
        if reasoning_content.strip():
            self.add_paragraph(reasoning_content, is_thought=True)

        # Render the content parts in order, stacked vertically. A text part renders as markdown
        # paragraphs; multiple text parts (e.g. one per websearch result) stack into the message's vertical
        # layout, giving per-result visual separation. The persona prefix on the first line of assistant content
        # ("Aria: ...") is stripped per part — a no-op for tool/system messages, which carry no persona.
        for part in message.get("content") or []:
            part_type = part.get("type")
            if part_type == "text":
                self._render_text_paragraphs(chatutil.remove_persona_from_start_of_line(persona=persona, text=part["text"]))
            elif part_type == "image_url":
                self._render_image_part(part, sidecars_meta)
            # else: unknown part type — skip (forward-compat)

        # Render any tool-call invocations this assistant message made, as visible sub-elements after the text.
        # Without this, a tool-calling turn — often with empty `content` — would show nothing
        # between the assistant message and the subsequent tool-result node.
        for index, tool_call in enumerate(message.get("tool_calls") or []):
            function = tool_call.get("function") or {}
            self.add_tool_call_invocation(index=index,
                                          name=function.get("name", "?"),
                                          arguments=function.get("arguments", ""))

    def _render_text_paragraphs(self, text: str) -> None:
        """Render one text content-part: split into paragraphs and add them.

        Also consolidates any inline `<think>...</think>` block into a single collapsible thought paragraph, but
        that handling is dead code: since the June 2026 `reasoning_content` migration, thinking is separated out
        before render (at load by `upgrade_datastore`, live by the stream parser), so `content` no longer
        carries inline `<think>`. Leftover from the pre-June-2026 inline handling; slated for removal."""
        paragraph_accumulator = io.StringIO()
        inside_think_block = False
        def commit_paragraph():
            nonlocal paragraph_accumulator
            text_to_commit = paragraph_accumulator.getvalue()
            if not text_to_commit:
                return
            self.add_paragraph(text_to_commit,
                               is_thought=inside_think_block)
            paragraph_accumulator = io.StringIO()

        paragraphs = text.split("\n")
        for idx, paragraph in enumerate(paragraphs):
            p = paragraph.strip()

            # Detect think block state (rudimentary; should detect from the token stream, not re-split a string).
            entering_think_block = (p == "<think>")
            exiting_think_block = (p == "</think>")

            if entering_think_block:
                commit_paragraph()  # commit previous text (if any) before start of think block
                inside_think_block = True

            paragraph_accumulator.write(f"{paragraph}\n")  # regardless of if it's just a newline

            # Consolidate "<think>...</think>" into one paragraph, so that we can hide/show it easily.
            # When at last paragraph, always commit (even if incomplete think block).
            if (inside_think_block and not exiting_think_block) and (idx < len(paragraphs) - 1):
                continue

            commit_paragraph()

            if exiting_think_block:
                inside_think_block = False

    def _render_image_part(self, part: Dict[str, Any], sidecars_meta: Dict[str, Any]) -> None:
        """Render one `image_url` content-part: an inline thumbnail plus a per-image provenance cluster.

        In a stored message the URL is always a Raven-internal `sidecar:<filename>` reference (see
        `chatutil.image_content_part`); the thumbnail texture is resolved and cached by the controller. A
        non-sidecar URL (shouldn't occur in stored data) is skipped for forward-compat; an unresolvable sidecar
        renders a small placeholder rather than nothing, so the message still reads as "an image was here".

        Provenance for this image lives in `sidecars_meta[filename]` (see `imagestore.store_image_as_sidecar`).
        The thumbnail carries the original filename as a tooltip, and a small action row below it offers, per
        image (a message may hold several): show the stored original at full size, open the recorded source (a
        `file://` original or an `https://` page — disabled when there is nothing openable), and reveal the
        chat's image-sidecar directory."""
        url = (part.get("image_url") or {}).get("url", "")
        if not url.startswith(imagestore.SIDECAR_SCHEME):
            return  # only local sidecar refs are resolvable here; skip anything else (forward-compat)
        filename = url[len(imagestore.SIDECAR_SCHEME):]
        meta = sidecars_meta.get(filename) or {}
        texture = self.parent_view.chat_controller.get_inline_image_texture(filename)
        datastore = self.parent_view.chat_controller.datastore
        with self.paragraphs_lock:
            if texture is None:
                dpg.add_text("[image unavailable]", color=(180, 120, 120), parent=self.gui_text_group)
                return

            cluster = dpg.add_group(parent=self.gui_text_group)  # thumbnail + its provenance action row, stacked
            image_id = dpg.add_image(texture.texture_tag,  # tag
                                     width=texture.w,
                                     height=texture.h,
                                     parent=cluster)
            with dpg.tooltip(image_id):  # original filename only; the action buttons carry their own tooltips
                dpg.add_text(_provenance_filename(meta.get("url")) or "attached image")

            # Per-image provenance actions. "Show original" resolves to the archival copy — the verbatim original
            # kept as a second sidecar (case 2 of the image store), or the primary itself when that is the
            # verbatim original (case 1); a downsample-only image (case 3) has no archival original, so the
            # primary is the best copy stored. "Open source" targets the recorded provenance URL, which is
            # fragile (the file may have moved, the page may 404) and absent for some images — disabled up front
            # when there is nothing openable. "Open folder" reveals the datastore's image-sidecar directory.
            archival_filename = meta.get("original_sidecar") or filename
            source_url = meta.get("url") or ""
            source_openable = bool(source_url) and not source_url.startswith("data:")
            actions = dpg.add_group(horizontal=True, parent=cluster)

            self._add_provenance_button(parent=actions,
                                        icon=fa.ICON_IMAGE,
                                        tooltip_text="Show full-size image\n(the saved copy, in the chat data folder)",
                                        ok_message="Opened image",
                                        action=lambda: common_utils.open_file(datastore.sidecar_path(archival_filename)))
            if source_openable:
                source_tooltip = f"Open original source\n{source_url}"
            elif source_url.startswith("data:"):
                source_tooltip = "Open original source — unavailable\n(the image was embedded inline; no external source)"
            else:
                source_tooltip = "Open original source — unavailable\n(no source location was recorded)"
            self._add_provenance_button(parent=actions,
                                        icon=fa.ICON_LINK,
                                        tooltip_text=source_tooltip,
                                        ok_message="Opened source",
                                        enabled=source_openable,
                                        action=lambda: _open_source_url(source_url))
            self._add_provenance_button(parent=actions,
                                        icon=fa.ICON_FOLDER_OPEN,
                                        tooltip_text="Open the image folder\n(where attached images are stored)",
                                        ok_message="Opened folder",
                                        action=lambda: common_utils.open_in_file_manager(datastore.sidecar_dir))

    def _add_provenance_button(self, *, parent: Union[str, int], icon: str, tooltip_text: str, ok_message: str,
                               action: Callable[[], None], enabled: bool = True) -> None:
        """Add one small provenance-action button (icon + tooltip) under an inline image, wired to run `action`.

        On click `action` runs; success flashes the button green with `ok_message`, any failure flashes it red
        (and logs) — a non-intrusive acknowledgment in place of a modal dialog, matching the global toolbar
        buttons. A disabled button (`enabled=False`) still shows its explanatory `tooltip_text` but does
        nothing, so a predictably-unavailable action (no recorded source, an inline `data:` image) is
        discoverable before the click rather than failing after it."""
        button_id = dpg.add_button(label=icon, width=gui_config.toolbutton_w, parent=parent, enabled=enabled)
        dpg.bind_item_font(button_id, self.parent_view.themes_and_fonts.icon_font_solid)
        dpg.bind_item_theme(button_id, "disablable_widget_theme")  # tag
        tooltip_id = dpg.add_tooltip(button_id)
        text_id = dpg.add_text(tooltip_text, parent=tooltip_id)
        if not enabled:
            return
        def callback() -> None:
            try:
                action()
                ok, message = True, ok_message
            except Exception as exc:  # noqa: BLE001 -- opening an external target must never crash the chat view
                logger.error(f"DPGCompleteChatMessage._add_provenance_button: action failed: {type(exc)}: {exc}")
                ok, message = False, "Couldn't open — it may have moved or been deleted"
            gui_animation.flash_button(button=button_id, tooltip=tooltip_id, text=text_id,
                                       ok=ok, message=message, duration=gui_config.acknowledgment_duration)
        dpg.set_item_callback(button_id, callback)


class DPGStreamingChatMessage(DPGChatMessage):
    def __init__(self,
                 gui_parent: Union[str, int],
                 parent_view: "DPGLinearizedChatView"):
        """A chat message being streamed live from the LLM, displayed in the linearized chat view.

        `gui_parent`: DPG tag or ID of the GUI widget (typically child window or group) to add the chat message to.
        `parent_view`: The linearized chat view widget this chat message is rendered in (and is owned by).

        Starts as blank. Use the `add_paragraph` and/or `replace_last_paragraph` methods to add text.

        To replace the streaming message with a completed message, call the streaming message's
        `demolish` method first. Doing so removes its widgets from the GUI.
        """
        super().__init__(gui_parent=gui_parent,
                         parent_view=parent_view)
        self.build()

    def build(self):
        super().build(role="assistant",  # TODO: parameterize this?
                      persona=self.parent_view.chat_controller.llm_settings.personas.get("assistant", None),
                      node_id=None)


class DPGLinearizedChatView:
    def __init__(self,
                 themes_and_fonts: env,
                 gui_parent: Union[str, int],
                 chat_controller: "DPGChatController"):
        """A view of the current chat branch, displayed as a linear chat.

        `themes_and_fonts`: Obtain by calling `raven.common.gui.utils.bootup` at app start time.

        `gui_parent`: DPG tag or ID of the panel (child window) you want the chat to be rendered in.

        `chat_controller`: The controller this view belongs to. Managed internally;
                           the `DPGLinearizedChatView` is instantiated and owned by the `DPGChatController`.
        """
        self.themes_and_fonts = themes_and_fonts
        self.gui_parent = gui_parent
        self.gui_uuid = str(uuid.uuid4())  # used in GUI widget tags
        self.chat_controller = chat_controller

        # TODO: We can later use the existence of this chat container group widget for double-buffering (can render a new group and then switch it in)
        self.chat_messages_container_group_widget = dpg.add_group(tag=f"chat_messages_container_group_{self.gui_uuid}",
                                                                  parent=gui_parent)

    def scroll_view(self,
                    max_wait_frames: int = 10,
                    scroll_target_node_id: Optional[str] = None) -> None:
        """Scroll this linearized chat view to the end.

        `max_wait_frames`: If `max_wait_frames > 0`, wait at most for that may frames
                           for the chat panel (`self.gui_parent`) to get a nonzero `max_y_scroll`.

                           Some waiting is usually needed at least at app startup
                           before the GUI settles.

        `target_y`: y coordinate to scroll to, in coordinate system of `self.gui_parent`.
                    If not provided (default), scroll to end.

        NOTE: When called from the main thread, `max_wait_frames` must be 0, as any
              attempt to wait would hang the main thread's explicit render loop.

              Setting `max_wait_frames=0` also has the effect of not logging the current
              frame number, because `dpg.get_frame_count()` would need the render thread mutex:
                  https://github.com/hoffstadt/DearPyGui/issues/2366

              When called from any other thread (also event handlers), waiting is fine.
        """
        max_y_scroll = dpg.get_y_scroll_max(self.gui_parent)
        for elapsed_frames in range(max_wait_frames):
            if max_y_scroll > 0:  # TODO: This approach fails when the content is less than one screenful in length. Think of a better way; currently we just use a small `max_wait_frames`.
                break
            dpg.split_frame()
            max_y_scroll = dpg.get_y_scroll_max(self.gui_parent)
        plural_s = "s" if elapsed_frames != 1 else ""
        waited_str = f" (after waiting for {elapsed_frames} frame{plural_s})" if elapsed_frames > 0 else " (no waiting was needed)"
        frames_str = f" frame {dpg.get_frame_count()}" if max_wait_frames > 0 else ""

        if scroll_target_node_id is not None:
            logger.info(f"DPGLinearizedChatView.scroll_view: Scroll target chat node is '{scroll_target_node_id}'")
            def get_target_widget() -> Optional[Union[str, int]]:
                for dpg_chat_message in self.chat_controller.current_chat_history:
                    if dpg_chat_message.node_id == scroll_target_node_id:  # found?
                        return dpg_chat_message.gui_container_group
                return None
            if (target_message_widget := get_target_widget()) is not None:
                x0, y0 = guiutils.get_widget_pos(target_message_widget)
                logger.info(f"DPGLinearizedChatView.scroll_view: Position of scroll target chat node is ({x0}, {y0}) (in GUI container coordinates).")
            else:
                y0 = max_y_scroll
                logger.warning(f"DPGLinearizedChatView.scroll_view: Scroll target chat node '{scroll_target_node_id}' not found in view, scrolling to end instead.")
            y_scroll = min(y0, max_y_scroll)
        else:
            logger.info("DPGLinearizedChatView.scroll_view: No scroll target chat node specified, scrolling to end.")
            y_scroll = max_y_scroll
        logger.info(f"DPGLinearizedChatView.scroll_view:{frames_str}{waited_str}: max_y_scroll = {max_y_scroll}, scrolling to y = {y_scroll}")
        dpg.set_y_scroll(self.gui_parent, y_scroll)

    def get_chatlog_as_markdown(self, include_metadata: bool) -> Optional[str]:
        """Format this linearized chat as Markdown, for e.g. copying to the clipboard or saving to a file.

        `include_metadata`: If `True`, the output will contain the node IDs, revision timestamps (ISO format), and revision numbers.

        Returns the chatlog as Markdown. If the view is empty, returns `None`.
        """
        with self.chat_controller.current_chat_history_lock:
            if not self.chat_controller.current_chat_history:
                return None

            output_text = io.StringIO()
            output_text.write(f"# Raven-librarian chatlog\n\n- *HEAD node ID*: `{self.chat_controller.current_chat_history[-1].node_id}`\n- *Log generated*: {chatutil.format_chatlog_datetime_now()}\n\n{'-' * 80}\n\n")
            for message_number, dpg_chat_message in enumerate(self.chat_controller.current_chat_history):
                node_payload = self.chat_controller.datastore.get_payload(dpg_chat_message.node_id)  # auto-selects active revision  TODO: later (chat editing), we need to set the revision to load
                message = node_payload["message"]
                role = message["role"]
                persona = node_payload["general_metadata"]["persona"]  # stored persona for this chat message
                text = chatutil.content_to_text(message["content"])
                formatted_message = format_chat_message_for_clipboard(message_number=message_number,
                                                                      role=role,
                                                                      persona=persona,
                                                                      text=text,
                                                                      add_heading=True)  # In the full chatlog, the message numbers and role names are important, so always include them.
                if include_metadata:
                    payload_datetime = node_payload["general_metadata"]["datetime"]  # of the active payload revision!
                    node_active_revision = self.chat_controller.datastore.get_revision(dpg_chat_message.node_id)
                    header = f"- *Node ID*: `{dpg_chat_message.node_id}`\n- *Revision date*: {payload_datetime}\n- *Revision number*: {node_active_revision}\n\n"  # yes, it'll say `None` when no node ID is available (incoming streaming message), which is exactly what we want.
                else:
                    header = ""
                output_text.write(f"{header}{formatted_message}\n\n{'-' * 80}\n\n")

            return output_text.getvalue()

    def add_complete_message(self,
                             node_id: str,
                             scroll_view: bool = True) -> DPGCompleteChatMessage:
        """Append the chat node with `node_id` to the end of the linearized chat view in the GUI.

        `scroll_view`: If `True`, then once the message has been added, wait for one frame
                       for the message to render, and scroll the chat view to the end.
        """
        with self.chat_controller.current_chat_history_lock:
            dpg_chat_message = DPGCompleteChatMessage(gui_parent=self.chat_messages_container_group_widget,
                                                      parent_view=self,
                                                      node_id=node_id)
            self.chat_controller.current_chat_history.append(dpg_chat_message)

            # Disable the "continue generation" and "show chat continuation" buttons on the old messages.
            # The latest message already has them *enabled* if it should.
            for dpg_old_message in self.chat_controller.current_chat_history[:-1]:
                if dpg_old_message.role == "assistant":  # only AI messages have a continue button
                    dpg.disable_item(f"message_continue_button_{dpg_old_message.gui_uuid}")
                dpg.disable_item(f"message_show_chat_continuation_button_{dpg_old_message.gui_uuid}")

        if scroll_view:
            dpg.split_frame()
            self.scroll_view()
        return dpg_chat_message

    # TODO: does this `build` really belong in `DPGLinearizedChatView` or in `DPGChatController`?
    def build(self,
              head_node_id: Optional[str] = None,
              scroll_target_node_id: Optional[str] = None) -> None:
        """Build the linearized chat view in the GUI, linearizing up from `head_node_id`.

        `scroll_target_node_id`: If provided, scroll to this node instead of to the end.
                                 Must be the chat node ID of a message shown in the view,
                                 i.e. either `head_node_id`, or one of its ancestors.

        As side effects:

          - Update the `current_chat_history` of the chat controller this view is bound to.
          - If `head_node_id` is an AI message, update the avatar's emotion from that
            (using the node's current payload revision).
        """
        # Shutdown guard (catch-all). `build` creates chat-message widgets, and several callers reach it on
        # background threads — the startup frame callback, but also the debounced resize-rebuild task, which
        # can be *submitted* after teardown has begun and so slip past the cancel. Creating widgets once the
        # app is tearing down races `destroy_context` → segfault. `gui_updates_safe` goes False as the very
        # first action of shutdown, so bailing on it here covers every path.
        if not self.chat_controller.gui_updates_safe:
            return
        if head_node_id is None:  # use current HEAD from app_state?
            head_node_id = self.chat_controller.app_state["HEAD"]
        node_id_history = self.chat_controller.datastore.linearize_up(head_node_id)
        with self.chat_controller.current_chat_history_lock:
            self.chat_controller.current_chat_history.clear()
            dpg.delete_item(self.chat_messages_container_group_widget,
                            children_only=True)  # clear old content from GUI
            for node_id in node_id_history:
                self.add_complete_message(node_id=node_id,
                                          scroll_view=False)  # we scroll just once, when done
        # Update avatar emotion from the final message text (use only non-thought message content)
        role, persona, text = chatutil.get_node_message_text_without_persona(self.chat_controller.datastore, head_node_id)
        if role == "assistant":
            logger.info("DPGLinearizedChatView.build: linearized chat view new HEAD node is an AI message; updating avatar emotion from (non-thought) message content")
            text = chatutil.scrub(persona=persona,
                                  text=text,
                                  thoughts_mode="discard",
                                  markup=None,
                                  add_persona=False)
            self.chat_controller.avatar_controller.update_emotion_from_text(config=self.chat_controller.avatar_record,
                                                                            text=text)
        self.chat_controller.avatar_controller.ping(config=self.chat_controller.avatar_record)  # wake up the AI avatar when the chat view is re-rendered
        self.chat_controller.update_context_fill_indicator()  # HEAD changed (rebuild / branch switch / initial load)
        # Skip the final settle-and-scroll during shutdown: once the render loop has stopped, `split_frame`
        # blocks forever (it waits for a frame that will never come). `gui_updates_safe` goes False as the very
        # first action of teardown, so a startup `build()` that races the close bails here instead of parking.
        if self.chat_controller.gui_updates_safe:
            dpg.split_frame()
            self.scroll_view(scroll_target_node_id=scroll_target_node_id)

# --------------------------------------------------------------------------------
# Scaffold to GUI integration

class DPGChatController:
    class_lock = threading.RLock()
    _class_initialized = False
    @classmethod
    def _load_class_textures(cls):
        """Load textures common to all instances of this class."""
        with cls.class_lock:
            if cls._class_initialized:
                return
            # Initialize textures.
            with dpg.texture_registry(tag="librarian_chat_controller_textures"):
                w, h, c, data = dpg.load_image(str(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "icons", "system.png")).expanduser().resolve()))
                cls.icon_system_texture = dpg.add_static_texture(w, h, data, tag="icon_system_texture")

                w, h, c, data = dpg.load_image(str(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "icons", "tool.png")).expanduser().resolve()))
                cls.icon_tool_texture = dpg.add_static_texture(w, h, data, tag="icon_tool_texture")

                w, h, c, data = dpg.load_image(str(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "icons", "user.png")).expanduser().resolve()))
                cls.icon_user_texture = dpg.add_static_texture(w, h, data, tag="icon_user_texture")

                w, h, c, data = dpg.load_image(str(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "icons", "ai.png")).expanduser().resolve()))   # generic AI icon
                cls.icon_ai_texture = dpg.add_static_texture(w, h, data, tag="icon_ai_texture_generic")
            cls._class_initialized = True

    def _load_instance_textures(self,
                                avatar_image_path: Optional[Union[str, pathlib.Path]]):
        """Load instance-specific textures.

        `avatar_image_path`: Path to the main character image of the AI's avatar.
                             Used for detecting the presence of a per-character icon.

                             If no per-character icon exists for this character,
                             a generic AI icon is used.
        """
        # Prefer per-character icon, if available. This intentionally shadows `type(self).icon_ai_texture`.
        character_dir = avatar_image_path.parent
        basename = os.path.basename(str(avatar_image_path))  # e.g. "/foo/bar/example.png" -> "example.png"
        stem, ext = os.path.splitext(basename)  # -> "example", ".png"
        character_icon_path = character_dir / f"{stem}_icon{ext}"
        if character_icon_path.exists():
            w, h, c, data = dpg.load_image(str(character_icon_path))
            self.icon_ai_texture = dpg.add_static_texture(w, h, data, tag=f"icon_ai_texture_0x{id(self):x}", parent="librarian_chat_controller_textures")  # tag

        self.gui_role_icons = {"assistant": self.icon_ai_texture,
                               "system": self.icon_system_texture,
                               "tool": self.icon_tool_texture,
                               "user": self.icon_user_texture,
                               }

    def __init__(self,
                 llm_settings: env,
                 datastore: chattree.Forest,
                 retriever: Optional[hybridir.HybridIR],
                 app_state: env,
                 avatar_controller: DPGAvatarController,
                 avatar_record: env,
                 avatar_image_path: Optional[Union[str, pathlib.Path]],
                 themes_and_fonts: env,
                 chat_panel_widget: Union[str, int],
                 chat_stop_generation_button_widget: Union[str, int],
                 indicator_glow_animation: Optional[gui_animation.PulsatingColor],
                 docs_indexing_glow_animation: Optional[gui_animation.PulsatingColor],
                 llm_indicator_widget: Union[str, int],
                 docs_indexing_indicator_widget: Union[str, int],
                 docs_indexing_progress_text_widget: Union[str, int],
                 docs_search_indicator_widget: Union[str, int],
                 docs_search_progress_text_widget: Union[str, int],
                 web_indicator_widget: Union[str, int],
                 executor: Optional[concurrent.futures.Executor] = None):
        """Controller for LLM scaffold to GUI integration.

        Owns a `DPGLinearizedChatView`, which displays the current branch of the chat.

        `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.

        `datastore`: The chat datastore.

        `retriever`: A `raven.librarian.hybridir.HybridIR` retriever connected to the document database.

        `app_state`: The chat's HEAD node ID, plus some persistent option flags.
                     See `raven.librarian.appstate`.

        `avatar_controller`: For TTS, and for controlling the "data eyes" effect of the avatar.

                             NOTE: In case of multiple avatars in the same app, there is still just one controller (to serialize TTS correctly).
                                   Each avatar instance has its own `avatar_record`.

        `avatar_record`: Control data for the avatar instance of the AI in this chat view.

                         See the `register_avatar_instance` method of `raven.client.avatar_controller.DPGAvatarController`.

        `avatar_image_path`: The file path to the main character image of the avatar of the AI speaking in this chat view.
                             This is used for detecting and loading the per-character icon. If the current character
                             has no per-character icon, a generic AI icon is used automatically.

        `themes_and_fonts`: Obtain by calling `raven.common.gui.utils.bootup` at app start time.

        `chat_panel_widget`: DPG tag or ID of the panel (child window) you want the chat to be rendered in.

        `chat_stop_generation_button_widget`: DPG tag or ID of the GUI button to interrupt the LLM (stop generating text).
                                              Will be auto-enabled only while the LLM is generating.

        `indicator_glow_animation`: When an indicator icon appears, the cycle of this animation will be reset,
                                    so that the glow always starts at the first animation frame.

                                    See `PulsatingColor` in `raven.common.gui.animation`.

        `docs_indexing_glow_animation`: Pulsator for the INDEXING indicator. Phase-reset on transition
                                        into the indexing state, so the glow always starts at the first
                                        animation frame when the indicator appears.

        `llm_indicator_widget`: DPG tag or ID of the widget to show while the prompt is being processed by
                                the LLM backend. Typically, a DPG group with items bound to the theme whose
                                color `indicator_glow_animation` pulsates.

        `docs_indexing_indicator_widget`: DPG tag or ID of the widget to show while the RAG database is
                                          being *indexed*. Independent from the search indicator —
                                          indexing and search can run concurrently, so they're separate
                                          stacked rows rather than two states of one widget.

        `docs_indexing_progress_text_widget`: DPG tag or ID of a text widget inside the indexing indicator;
                                              mirrors `retriever.get_indexing_progress_text()`.

        `docs_search_indicator_widget`: DPG tag or ID of the widget to show while the database is being
                                        *consulted* (search) by the LLM.

        `docs_search_progress_text_widget`: DPG tag or ID of a text widget inside the search indicator;
                                            mirrors `retriever.get_query_progress_text()`.

        `web_indicator_widget`: DPG tag or ID of the widget to show while a "websearch" tool call is in progress.

        `executor`: A `ThreadPoolExecutor` or something duck-compatible with it. Used for background tasks.
        """
        type(self)._load_class_textures()
        self._load_instance_textures(avatar_image_path)

        # Inline chat-image thumbnails get their own texture registry, separate from the role-icon textures
        # (`librarian_chat_controller_textures`). Cached by sidecar filename so an image referenced by several
        # messages — or re-encountered on a view rebuild — decodes and uploads once. The textures live for the
        # controller's lifetime and are never deleted (which also sidesteps the Nvidia/Linux texture-delete
        # segfault). The lock serializes get-or-create so two concurrent message builds can't both try to
        # create the same-tagged texture (a duplicate DPG tag crashes the process, not raises).
        self._inline_image_texture_registry = dpg.add_texture_registry(tag="librarian_chat_inline_image_textures")  # tag
        self._inline_image_textures = {}  # {sidecar_filename: env(texture_tag, w, h)}
        self._inline_image_lock = threading.RLock()

        self.llm_settings = llm_settings
        self.datastore = datastore
        self.retriever = retriever
        self.app_state = app_state
        self.avatar_controller = avatar_controller
        self.avatar_record = avatar_record
        self.chat_stop_generation_button_widget = chat_stop_generation_button_widget
        self.indicator_glow_animation = indicator_glow_animation
        self.docs_indexing_glow_animation = docs_indexing_glow_animation
        self.llm_indicator_widget = llm_indicator_widget
        self.docs_indexing_indicator_widget = docs_indexing_indicator_widget
        self.docs_indexing_progress_text_widget = docs_indexing_progress_text_widget
        self.docs_search_indicator_widget = docs_search_indicator_widget
        self.docs_search_progress_text_widget = docs_search_progress_text_widget
        self.web_indicator_widget = web_indicator_widget

        # Indicator wiring. Show/hide events are pushed via callbacks (symmetric across all four
        # indicators: on_docs_start/done from the chat scaffold drive DOCS / SYSTEM / WEB; the new
        # on_indexing_start/done on the retriever drive INDEXING). Progress text remains polled —
        # it's a continuously-updated state, not a discrete event, and polling models that shape
        # naturally with no per-update callback overhead.
        self._docs_indexing_progress_last = ""
        self._docs_search_progress_last = ""
        if self.retriever is not None:
            self.retriever.set_indexing_callbacks(on_start=self._on_indexing_start,
                                                  on_done=self._on_indexing_done)
        self.current_chat_history = []
        self.current_chat_history_lock = threading.RLock()

        self.gui_updates_safe = True  # At app shutdown, they aren't.

        # Sync the INDEXING indicator to any commit already in progress. The startup rescan
        # (`hybridir.setup`) can begin re-indexing before this controller exists to wire its callbacks, so
        # the 0→1 edge that fires `on_indexing_start` passes unheard — belongs with the indicator wiring
        # above, but must run after `gui_updates_safe`, which `_on_indexing_start` gates on.
        if self.retriever is not None and self.retriever.is_indexing():
            self._on_indexing_start()

        self.view = DPGLinearizedChatView(themes_and_fonts=themes_and_fonts,
                                          gui_parent=chat_panel_widget,
                                          chat_controller=self)

        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor()

        self.task_manager = bgtask.TaskManager(name="librarian_chat_controller",  # for most tasks
                                               mode="concurrent",
                                               executor=executor)
        self.ai_turn_task_manager = bgtask.TaskManager(name="librarian_chat_controller_ai_turn",  # for running the AI's turn, specifically (so that we can easily cancel just that one task when needed)
                                                       mode="concurrent",
                                                       executor=executor)  # same thread pool
        self.context_prefill_task_manager = bgtask.TaskManager(name="librarian_chat_controller_context_prefill",  # its own manager so a HEAD change cancels just the prefill
                                                               mode="sequential",  # only the latest HEAD's prefill matters; submitting a new one auto-cancels the previous
                                                               executor=executor)  # same thread pool
        # The debounced idle context-prefill. `ManagedTask` supplies the pending-wait debounce (cancellable in
        # `running_poll_interval` chunks) and the single-in-flight guarantee; we just submit one per HEAD change.
        # Created only when the feature is enabled (`config.context_prefill_idle_delay is not None`).
        self.context_prefill_task = None
        if librarian_config.context_prefill_idle_delay is not None:
            self.context_prefill_task = bgtask.ManagedTask(category="raven_librarian_chat_controller_context_prefill",
                                                           entrypoint=self._context_prefill_entrypoint,
                                                           running_poll_interval=0.25,
                                                           pending_wait_duration=librarian_config.context_prefill_idle_delay)

    def disable_gui_updates(self) -> None:
        """Stop the controller from firing GUI events.

        After this call:
          - `gui_updates_safe` is `False`, so any callback that gates on it (the on_docs_*,
            on_llm_*, on_tools_*, on_indexing_* handlers) becomes a no-op.
          - The retriever's indexing-lifecycle callbacks are cleared, so a cancelled `commit()`'s
            `finally` won't even reach the controller.

        Idempotent. Use as the first phase of app shutdown — run *before* `hybridir.shutdown()`
        and DPG teardown. The cancelled commit's `finally` block fires `on_indexing_done` from a
        worker thread, and any in-flight chat task can fire `on_docs_done` similarly; if those
        run while DPG widgets are already being torn down, `dpg.show/hide_item` raises against
        deleted widgets. Disabling the GUI-side hooks first sidesteps that race.

        The second phase is `shutdown()`, which drains background tasks. That has to run *after*
        `hybridir.shutdown()` because chat tasks blocked in `retriever.search` need
        `datastore_lock` to be released first.
        """
        self.gui_updates_safe = False
        if self.retriever is not None:
            self.retriever.set_indexing_callbacks(on_start=None, on_done=None)

    def cancel_tasks(self) -> None:
        """Signal all background tasks to stop, WITHOUT waiting. Idempotent.

        The non-blocking first phase of shutdown, meant to run from the app's DPG exit callback — i.e.
        from inside a render frame. A task parked in `dpg.split_frame` (e.g. the chat-streaming updater)
        can only be released by the render loop completing one more frame; waiting for it *here* would
        deadlock, because the render loop is currently sitting in the exit callback. So we only signal
        cancellation now (so the final frame releases the `split_frame` waiters, which then observe the
        flag and exit), and leave the blocking drain to `shutdown()`, called from the render loop's
        `finally` once the loop has exited.
        """
        self.disable_gui_updates()
        self.task_manager.clear(wait=False)
        self.ai_turn_task_manager.clear(wait=False)
        self.context_prefill_task_manager.clear(wait=False)

    def shutdown(self):
        """Prepare module for app shutdown.

        Second phase of shutdown: signal the background tasks to exit and wait for them.
        Calls `disable_gui_updates()` first (idempotent), so callers that haven't already
        invoked the first phase still get safe semantics.
        """
        self.disable_gui_updates()
        self.task_manager.clear(wait=True)
        self.ai_turn_task_manager.clear(wait=True)
        self.context_prefill_task_manager.clear(wait=True)

    def _on_indexing_start(self) -> None:
        """Show the INDEXING indicator. Called from `HybridIR.commit()`'s worker thread."""
        # TEMP INSTRUMENTATION: INDEXING indicator debugging (2026-04-28)
        logger.info(f"DPGChatController._on_indexing_start: INSTR entered: gui_updates_safe={self.gui_updates_safe}, widget={self.docs_indexing_indicator_widget!r}, exists={dpg.does_item_exist(self.docs_indexing_indicator_widget)}")
        if self.gui_updates_safe:
            if self.docs_indexing_glow_animation is not None:
                self.docs_indexing_glow_animation.reset()  # crisp phase on appear
            dpg.show_item(self.docs_indexing_indicator_widget)
            logger.info(f"DPGChatController._on_indexing_start: INSTR after show: visible={dpg.is_item_shown(self.docs_indexing_indicator_widget)}")

    def _on_indexing_done(self) -> None:
        """Hide the INDEXING indicator. Called from `HybridIR.commit()`'s worker thread."""
        # TEMP INSTRUMENTATION: INDEXING indicator debugging (2026-04-28)
        logger.info(f"DPGChatController._on_indexing_done: INSTR entered: gui_updates_safe={self.gui_updates_safe}, widget={self.docs_indexing_indicator_widget!r}, exists={dpg.does_item_exist(self.docs_indexing_indicator_widget)}")
        if self.gui_updates_safe:
            dpg.hide_item(self.docs_indexing_indicator_widget)

    def update_docs_indicator_progress_text(self) -> None:
        """Poll the retriever's two progress-text channels; mirror changes to the DPG widgets.

        Intended to be called once per frame from the app's `update_animations` tick. Cheap when nothing
        is changing (two string comparisons), only does GUI work on change.

        Indicator visibility is push-driven via callbacks — `on_docs_start`/`on_docs_done` from the chat
        scaffold for DOCS, `on_indexing_start`/`on_indexing_done` from the retriever for INDEXING. Only
        the progress texts (continuously-updated state, not discrete events) remain polled.
        """
        if self.retriever is None:
            return
        if not self.gui_updates_safe:
            return

        indexing_progress = self.retriever.get_indexing_progress_text()
        if indexing_progress != self._docs_indexing_progress_last:
            dpg.set_value(self.docs_indexing_progress_text_widget, indexing_progress)
            self._docs_indexing_progress_last = indexing_progress

        query_progress = self.retriever.get_query_progress_text()
        if query_progress != self._docs_search_progress_last:
            dpg.set_value(self.docs_search_progress_text_widget, query_progress)
            self._docs_search_progress_last = query_progress

    def is_generating(self) -> bool:
        """Return whether an AI turn is currently in flight (LLM streaming or tool calls).

        Intended for GUI clients that gate an idle-throttle predicate on "something is happening".
        """
        return self.ai_turn_task_manager.has_tasks()

    def get_last_message(self) -> Optional[DPGChatMessage]:
        """Return the `DPGChatMessage` for the last currently displayed message. Return `None` if the view is empty."""
        if not self.current_chat_history:
            return None
        dpg_chat_message = self.current_chat_history[-1]
        return dpg_chat_message

    def get_inline_image_texture(self, filename: str) -> Optional[env]:
        """Return a cached DPG texture for the chat sidecar `filename`, creating it on first use.

        Reads the sidecar bytes, downsamples to a thumbnail that fits the inline display box
        (`gui_config.chat_inline_image_h` × `chat_inline_image_w`, aspect preserved, never upscaled), uploads a
        static texture into the controller's inline-image registry, and caches it by filename — so the same
        image referenced by several messages, or re-encountered on a view rebuild, decodes once. Returns an
        `env(texture_tag, w, h)`, or `None` if the sidecar is missing or can't be decoded.

        Safe to call from a message-build background thread: texture creation is serialized (a duplicate DPG tag
        would crash the process), and two `split_frame`s after a fresh upload let DPG process the new texture
        before it is first drawn. (DPG defers the OpenGL upload to a render frame; a single wait empirically
        isn't enough — see dpg-notes.md "Texture upload ordering". A `static_texture` is correct here because
        these thumbnails are permanent — cached for the controller's lifetime, never deleted.)
        """
        with self._inline_image_lock:
            cached = self._inline_image_textures.get(filename)
            if cached is not None:
                return cached
            try:
                from ..common.image import codec, lanczos  # deferred: pulls torch / Pillow only when an image is shown
                from ..common.image import utils as image_utils
                raw = self.datastore.read_sidecar(filename)
                arr = image_utils.ensure_rgba(codec.decode(raw))  # (H, W, 4) uint8
                src_h, src_w = int(arr.shape[0]), int(arr.shape[1])
                scale = min(gui_config.chat_inline_image_h / src_h,
                            gui_config.chat_inline_image_w / src_w,
                            1.0)  # 1.0 cap: show a small image at native size, never upscale
                disp_h = max(1, round(src_h * scale))
                disp_w = max(1, round(src_w * scale))
                tensor = image_utils.np_to_tensor(arr, device="cpu")  # (1, 4, H, W) float32
                tensor = lanczos.resize(tensor, disp_h, disp_w)
                flat = image_utils.tensor_to_dpg_flat(tensor)  # flat float32 RGBA in [0, 1]
                texture_tag = f"chat_inline_image_{filename}"  # tag  # filename is a content-addressed sha256.ext, so unique
                dpg.add_static_texture(disp_w, disp_h, flat,
                                       tag=texture_tag,  # tag
                                       parent=self._inline_image_texture_registry)
                dpg.split_frame()  # trigger the deferred OpenGL upload...
                dpg.split_frame()  # ...and ensure it completed before the image widget draws it (single wait isn't enough; dpg-notes.md)
                result = env(texture_tag=texture_tag, w=disp_w, h=disp_h)
                self._inline_image_textures[filename] = result
                return result
            except Exception as exc:  # noqa: BLE001 -- a broken sidecar must not break rendering the rest of the chat
                logger.error(f"DPGChatController.get_inline_image_texture: failed to load sidecar '{filename}': {type(exc)}: {exc}")
                return None

    def _render_context_fill(self, count: int, is_exact: bool) -> None:
        """Set the bottom-toolbar context-fill readout text from a token `count`. Low-level; does no scheduling.

        `is_exact` drives the typography: `X%` when the count is exact (a local tokenizer, ooba's token-count
        endpoint, or a backend-reported `prompt_tokens` from `_context_prefill_task`), `~X%` when it is a
        calibrated estimate.
        """
        if not self.gui_updates_safe:
            return
        context_length = self.llm_settings.context_length
        percent = round(100 * count / context_length) if context_length else 0
        prefix = "" if is_exact else "~"
        with guiutils.nonexistent_ok():  # the readout widget may vanish under a shutdown race (background prefill caller)
            dpg.set_value("context_fill_text", f"{prefix}{percent}%  ({count} / {context_length})")  # tag

    def update_context_fill_indicator(self) -> None:
        """Refresh the bottom-toolbar context-fill readout: the current chat's token size vs the loaded window.

        Two-stage: this immediate pass approximates the prompt by the visible conversation content — the system
        prompt, RAG injects, and tool definitions add some tokens not counted here, so it slightly under-reports —
        and then schedules a debounced background prefill (`_schedule_context_prefill`) that, once the chat settles,
        replaces the estimate with the backend's exact full-prompt `prompt_tokens` (and warms the KV cache).

        Attached images each add a per-family estimate (`llmclient.image_token_cost`) — a VLM image consumes
        context the char->token ratio can't see. Any image present forces the `~X%` (estimate) typography until
        the background prefill lands the exact count.
        """
        if not self.gui_updates_safe:
            return
        try:
            node_ids = self.datastore.linearize_up(self.app_state["HEAD"])
            text_segments = []
            image_tokens = 0
            for node_id in node_ids:
                payload = self.datastore.get_payload(node_id)
                message = payload["message"]
                text_segments.append(chatutil.content_to_text(message.get("content")))
                sidecars_meta = payload.get("general_metadata", {}).get("sidecars", {})
                for part in message.get("content") or []:
                    if part.get("type") != "image_url":
                        continue
                    url = (part.get("image_url") or {}).get("url", "")
                    filename = url[len(imagestore.SIDECAR_SCHEME):] if url.startswith(imagestore.SIDECAR_SCHEME) else None
                    dims = (sidecars_meta.get(filename) or {}).get("stored_dimensions") if filename else None
                    image_h, image_w = dims if dims else (1024, 1024)  # fallback for pre-stored-dims data; only matters for resolution-scaling families
                    image_tokens += llmclient.image_token_cost(self.llm_settings, image_h, image_w)
            count, is_exact = llmclient.count_tokens(self.llm_settings, "".join(text_segments))
            if image_tokens:
                count += image_tokens
                is_exact = False  # per-image token cost is an estimate, so the whole readout is now approximate
            self._render_context_fill(count, is_exact)
        except Exception:  # noqa: BLE001 -- a status readout must never break the GUI or a chat turn
            logger.exception("DPGChatController.update_context_fill_indicator: failed to update the context-fill readout")
        self._schedule_context_prefill()

    def _schedule_context_prefill(self) -> None:
        """(Re)arm the debounced background context-prefill for the current HEAD.

        Submits a `ManagedTask`; the sequential `TaskManager` auto-cancels the previous pending/in-flight prefill
        (a HEAD change invalidates it), so this is safe to call from every HEAD-change site — it's driven from
        `update_context_fill_indicator`. The actual backend round-trip happens only after the `ManagedTask`'s
        pending wait (`config.context_prefill_idle_delay` seconds of quiet); see `_context_prefill_entrypoint`.
        No-op when the feature is disabled (the task wasn't created).
        """
        if not self.gui_updates_safe:
            return
        if self.context_prefill_task is None:  # feature disabled (config.context_prefill_idle_delay is None)
            return
        self.context_prefill_task_manager.submit(self.context_prefill_task,
                                                 env(wait=True, head_node_id=self.app_state["HEAD"]))

    def _context_prefill_entrypoint(self, task_env: env) -> None:
        """`ManagedTask` entrypoint: after the idle debounce, ask the backend for the exact prompt size of the captured branch.

        The pending-wait debounce and cancel-on-resubmit are handled by the `ManagedTask` / sequential-`TaskManager`
        machinery; we reach here only once the wait has elapsed without a newer HEAD superseding us. Sends the
        linearized branch to the backend via `llmclient.prefill` (generates ~nothing, but reports the exact templated
        `prompt_tokens` and warms the KV cache). On success, upgrades the indicator to `X%`.

        Bails (leaving the estimate in place) if cancelled, if the app is shutting down, if a real generation is in
        flight (that turn warms the cache and reports its own exact count), or if HEAD has moved off the branch this
        task captured — including a final re-check after the round-trip, so a late reply can't overwrite a newer
        branch's readout.
        """
        if task_env.cancelled or not self.gui_updates_safe or self.is_generating():
            return
        if self.app_state["HEAD"] != task_env.head_node_id:  # HEAD moved during the idle wait
            return

        history = chatutil.linearize_chat(datastore=self.datastore,
                                          node_id=task_env.head_node_id)
        out = llmclient.prefill(self.llm_settings,
                                history,
                                tools_enabled=self.app_state["tools_enabled"],  # match the next turn, so tool defs are counted/cached identically
                                datastore=self.datastore)  # resolve any sidecar: image refs so the exact prompt size counts image tokens

        if task_env.cancelled or not self.gui_updates_safe:
            return
        if out is None or out.usage is None or out.usage.get("prompt_tokens") is None:
            return  # backend didn't report usage; keep the estimate
        if self.app_state["HEAD"] != task_env.head_node_id:  # branch switched while we were waiting on the backend
            return
        logger.info(f"DPGChatController._context_prefill_entrypoint: exact prompt size for HEAD '{task_env.head_node_id}': {out.usage['prompt_tokens']} tokens")
        self._render_context_fill(out.usage["prompt_tokens"], is_exact=True)

    def chat_round(self, user_message_text: str, staged_images: Optional[List[env]] = None,
                   staged_files: Optional[List[env]] = None) -> None:
        """Run a chat round (user and AI).

        `user_message_text`: What the user wrote.

                             If `user_message_text` is the empty string *and* nothing is attached (no images and
                             no documents), the AI will generate another message without the user writing in
                             between.

        `staged_images`: Images the user attached to this message, or `None`. Each entry is an `env` with `raw`
                         (image bytes), `provenance_url`, and `provenance_source` (see `scaffold.user_turn`).
                         An attachment counts as user content: with images present, a round runs even when the
                         text is empty (rather than being treated as "let the AI take another turn").

        `staged_files`: Documents (plain text / PDF) the user attached, or `None` — the file counterpart of
                        `staged_images` (see `scaffold.user_turn`). Also counts as user content: a round runs with
                        attachments present even when the text is empty.

        The RAG query (for document database search) is taken from the latest available user message:

          - `user_message_text` if not the empty string.
          - Otherwise, automatically obtained by scanning the current chat for the user's latest message.

        This spawns a background task to avoid hanging GUI event handlers,
        since the typical use case is to call `chat_round` from a GUI event handler.
        """
        def chat_round_task(task_env: env) -> None:
            if task_env.cancelled:  # while the task was in the queue
                return

            # Add the user's message to the chat if the user entered any text or attached anything.
            if user_message_text or staged_images or staged_files:
                self.user_turn(text=user_message_text, staged_images=staged_images, staged_files=staged_files)
                # NOTE: Rudimentary approach to RAG search, using the user's message text as the query. (Good enough to demonstrate the functionality. Improve later.)
                docs_query = user_message_text or None  # image-only message: no text to search docs with
            else:
                # Handle the RAG query: find the latest existing user message
                docs_query = None  # if no user message, send `None` as query to AI -> no docs search
                for dpg_chat_message in reversed(self.current_chat_history):
                    if dpg_chat_message.role == "user":
                        docs_query = dpg_chat_message.text
                        break
            if task_env.cancelled:  # during user turn
                return
            self.ai_turn(docs_query=docs_query,
                         continue_=False)
        self.task_manager.submit(chat_round_task, env())

    def user_turn(self, text: str, staged_images: Optional[List[env]] = None,
                  staged_files: Optional[List[env]] = None) -> str:
        """Run the user's part of a chat round: create the user message node, update HEAD, append it to the view.

        Returns the new HEAD node id.

        Runs **synchronously on the caller's thread** — deliberately not as a task of its own, and deliberately
        asymmetric with `ai_turn`, which *is* task-based (see its docstring for why that one must be). The AI
        turn that follows in the same round must observe the completed user turn (its message node as the new
        HEAD, its sidecar images already written, and the message already in the view); if the two ran as
        separate concurrent tasks, that ordering would be a race — invisible while the AI turn takes seconds to
        reach its first output, but wrong the instant the backend errors immediately (the AI's error message
        would append before the user's message, and could even be parented to the pre-user HEAD). So
        `chat_round` calls this inline, then submits the AI turn.

        Call from a background thread (as `chat_round` does), never directly from a GUI event handler — it does
        datastore and (with attachments) image work. That constraint is exactly why this needs no task of its
        own: unlike `ai_turn`, it is never invoked straight from the GUI, so there is no GUI thread to keep free.

        `staged_images`: Images the user attached, or `None`. Passed through to `scaffold.user_turn`, which
                         stores each as a datastore sidecar (decode/downsample happens here, off the GUI thread).
        `staged_files`: Documents (plain text / PDF) the user attached, or `None`. Passed through to
                        `scaffold.user_turn`, which stores each verbatim as a datastore sidecar.
        """
        new_head_node_id = scaffold.user_turn(llm_settings=self.llm_settings,
                                              datastore=self.datastore,
                                              head_node_id=self.app_state["HEAD"],
                                              user_message_text=text,
                                              staged_images=staged_images,
                                              staged_files=staged_files)
        self.app_state["HEAD"] = new_head_node_id  # update HEAD before the AI turn reads it as the parent
        self.view.add_complete_message(new_head_node_id)
        self.update_context_fill_indicator()  # user message added -> context grew
        return new_head_node_id

    def ai_turn(self,
                docs_query: Optional[str],
                continue_: bool,
                _retry_tool_node_id: Optional[str] = None) -> None:
        """Run the AI's response part of a chat round.

        Spawns a background task (on its own `ai_turn_task_manager`) — deliberately, and deliberately asymmetric
        with `user_turn`, which runs synchronously. Three reasons this one must be tasked, none of which apply to
        `user_turn`:
          1. It is invoked *directly from GUI event handlers* — reroll, continue, and "approve denied host &
             retry" all call `ai_turn` from the DPG callback thread, which must return at once. (`user_turn` is
             only ever called from inside `chat_round`'s task, already off the GUI thread.)
          2. It needs *independent cancellation* — the Stop button clears just `ai_turn_task_manager`
             (`stop_ai_turn`), interrupting the AI turn without disturbing any other task.
          3. It is *long-running* — LLM streaming, tool calls, web fetches — the actual reason GUI responsiveness
             is at stake here.
        The underlying `scaffold.ai_turn` is itself synchronous; the tasking is the controller's concern (the CLI
        client `minichat` calls `scaffold.ai_turn` straight, and blocks, which is right for a REPL).

        `docs_query`: Query for RAG document database, or `None` for no search. Search results are auto-injected before the LLM replies.

        `continue_`: If `False`, create a new AI message. Most of the time, this is what you want.
                     If `True`, continue the AI's current message.

        `_retry_tool_node_id`: Internal. If set, this is the "approve denied host & retry" override: instead
                               of a normal AI turn, re-run the previously-denied tool call at this node on a
                               new branch (`scaffold.retry_tool_calls`) and continue from there. The same GUI
                               callback bundle is reused; `docs_query`/`continue_` are ignored in this mode.
        """
        docs_query = docs_query if self.app_state["docs_enabled"] else None

        def ai_turn_task(task_env: env) -> None:
            if task_env.cancelled:  # while the task was in the queue
                return

            # A live turn supersedes any pending idle-prefill: it warms the KV cache itself and reports its own
            # exact `prompt_tokens`, so a concurrent prefill round-trip would be wasted (and would contend with
            # the real request on a single-model backend).
            self.context_prefill_task_manager.clear()

            if self.gui_updates_safe:
                dpg.enable_item(self.chat_stop_generation_button_widget)

            speech_enabled = self.app_state["avatar_speech_enabled"]  # grab once, in case the user toggles it while this AI turn is being processed

            try:
                streaming_chat_message = None
                def delete_streaming_chat_message():  # for replacing with completed message
                    nonlocal streaming_chat_message
                    if streaming_chat_message is not None:
                        streaming_chat_message.demolish()
                        streaming_chat_message = None

                def on_docs_start() -> None:
                    if self.gui_updates_safe:
                        self.avatar_controller.start_data_eyes(config=self.avatar_record)
                        if self.indicator_glow_animation is not None:
                            self.indicator_glow_animation.reset()  # crisp phase on appear
                        dpg.show_item(self.docs_search_indicator_widget)

                def on_docs_done(matches: List[Dict]) -> None:
                    if self.gui_updates_safe:
                        dpg.hide_item(self.docs_search_indicator_widget)
                        self.avatar_controller.stop_data_eyes(config=self.avatar_record)

                def on_llm_start() -> None:
                    if self.gui_updates_safe:
                        nonlocal streaming_chat_message

                        # When continuing, delete the previous completed revision of the message from the GUI
                        if continue_:
                            old_dpg_chat_message = self.current_chat_history.pop(-1)
                            old_dpg_chat_message.demolish()

                        streaming_chat_message = DPGStreamingChatMessage(gui_parent=self.view.chat_messages_container_group_widget,
                                                                         parent_view=self.view)
                        dpg.split_frame()
                        self.view.scroll_view()

                        if self.indicator_glow_animation is not None:
                            self.indicator_glow_animation.reset()  # start new pulsation cycle
                        dpg.show_item(self.llm_indicator_widget)  # show prompt processing indicator

                task_env.text = io.StringIO()  # incoming, in-progress paragraph
                task_env.t0 = time.monotonic()  # timestamp of last GUI update
                task_env.n_chunks0 = 0  # chunks received since last GUI update

                task_env.current_is_thought = False  # which channel the in-progress paragraph belongs to (thought bubble vs visible answer)
                task_env.seen_content = False  # whether any visible-answer content has arrived yet (to fire the talking animation once)

                task_env.emotion_update_interval = 5  # how many lines of text to wait between emotion updates (NOTE: Qwen3 uses a double newline as its paragraph separator, so that eats an extra line)
                task_env.emotion_recent_paragraphs = collections.deque([""] * (4 * task_env.emotion_update_interval))  # buffer with 75% overlap between updates, to stabilize the detection
                task_env.emotion_update_calls = 0
                def _update_avatar_emotion_from_incoming_text(new_paragraph: str) -> None:
                    task_env.emotion_recent_paragraphs.append(new_paragraph)
                    task_env.emotion_recent_paragraphs.popleft()
                    if task_env.emotion_update_calls % task_env.emotion_update_interval == 0:
                        text = "".join(task_env.emotion_recent_paragraphs)
                        logger.info(f"ai_turn.ai_turn_task._update_avatar_emotion_from_incoming_text: updating emotion from {len(text)} characters of recent text")
                        self.avatar_controller.update_emotion_from_text(config=self.avatar_record,
                                                                        text=text)
                    task_env.emotion_update_calls += 1

                def on_llm_progress(event: Dict[str, Any]) -> Optional[sym]:
                    # `invoke` is the single parser; this handler is a pure renderer dispatching on the typed
                    # event. No regex-sniffing of the text stream; the event type *is* the state.

                    # If the task is cancelled (`stop_ai_turn` was called), interrupt the LLM, keeping the content received so far.
                    # The scaffold will automatically send the content to `on_llm_done`.
                    if task_env.cancelled or not self.gui_updates_safe:  # TODO: EAFP to avoid TOCTTOU
                        reason = "Cancelled" if task_env.cancelled else "App is shutting down"
                        logger.info(f"ai_turn.ai_turn_task.on_llm_progress: {reason}, stopping text generation.")
                        return llmclient.action_stop

                    event_type = event["type"]
                    if event_type == "tool_call":
                        # Structured tool-call invocations render when the completed message reloads. Nothing to stream live.
                        return llmclient.action_ack

                    chunk_text = event["text"]
                    n_chunks = event.get("n_chunks", 0)
                    is_thought = (event_type == "reasoning")  # reasoning -> thought bubble; content -> visible answer

                    if self.gui_updates_safe and chunk_text:  # avoid triggering on an empty event
                        dpg.hide_item(self.llm_indicator_widget)  # hide prompt processing indicator

                    # Fire the generic talking animation once, when the model transitions from thinking to the
                    # visible answer (replaces the old "</think> seen" trigger).
                    if not is_thought and not task_env.seen_content:
                        task_env.seen_content = True
                        logger.info("ai_turn.ai_turn_task.on_llm_progress: AI started writing the visible answer.")
                        if not speech_enabled:  # If TTS is NOT enabled, show the generic talking animation while the LLM is writing
                            api.avatar_start_talking(self.avatar_record.avatar_instance_id)

                    # If the channel changed mid-paragraph (thought <-> answer), commit the in-progress paragraph
                    # and start a fresh one in the new channel — the renderer colors per paragraph, so a thought
                    # and the answer must never share one.
                    if task_env.text.getvalue() and (is_thought != task_env.current_is_thought):
                        streaming_chat_message.replace_last_paragraph(task_env.text.getvalue(),
                                                                      is_thought=task_env.current_is_thought)
                        streaming_chat_message.add_paragraph("", is_thought=is_thought)
                        task_env.text = io.StringIO()
                        task_env.t0 = time.monotonic()
                        task_env.n_chunks0 = n_chunks
                        dpg.split_frame()
                        self.view.scroll_view()
                    task_env.current_is_thought = is_thought

                    # Accumulate the chunk, then render. Write *before* reading the paragraph so the chunk is
                    # never lost when it carries the paragraph-break newline (the trailing newline is stripped at render time).
                    task_env.text.write(chunk_text)
                    paragraph_text = task_env.text.getvalue()
                    time_now = time.monotonic()
                    dt = time_now - task_env.t0  # seconds since last GUI update
                    dchunks = n_chunks - task_env.n_chunks0  # chunks since last GUI update
                    if "\n" in chunk_text:  # start new paragraph?
                        task_env.t0 = time_now
                        task_env.n_chunks0 = n_chunks
                        # NOTE: The last paragraph of the AI's reply - for thinking models, commonly the final response - often never gets a "\n", and must be handled in `on_done`.
                        _update_avatar_emotion_from_incoming_text(paragraph_text)  # update emotion from recent received text (thoughts too)
                        streaming_chat_message.replace_last_paragraph(paragraph_text,
                                                                      is_thought=is_thought)
                        streaming_chat_message.add_paragraph("",
                                                             is_thought=is_thought)
                        task_env.text = io.StringIO()
                        dpg.split_frame()
                        self.view.scroll_view()
                    # - update at least every 0.5 sec, even if the LLM is slow
                    # - update after every 10 chunks, but with a rate limit
                    elif dt >= 0.5 or (dt >= 0.25 and dchunks >= 10):  # commit changes to in-progress last paragraph
                        task_env.t0 = time_now
                        task_env.n_chunks0 = n_chunks
                        streaming_chat_message.replace_last_paragraph(paragraph_text,
                                                                      is_thought=is_thought)  # at first paragraph, will auto-create the paragraph if not created yet
                        dpg.split_frame()
                        self.view.scroll_view()

                    # Let the LLM keep generating (if it wants to).
                    return llmclient.action_ack

                def on_done(node_id: str) -> None:   # For both `on_llm_done` and `on_nomatch_done`.
                    self.app_state["HEAD"] = node_id  # update just in case of Ctrl+C or crash during tool calls
                    task_env.text = io.StringIO()  # for next AI message (in case of tool calls)
                    if self.gui_updates_safe:
                        if not speech_enabled:  # If TTS is NOT enabled, stop the generic talking animation now that the LLM is done
                            api.avatar_stop_talking(self.avatar_record.avatar_instance_id)

                        unused_role, persona, text = chatutil.get_node_message_text_without_persona(self.datastore, node_id)

                        # Keep only non-thought content for TTS and final emotion update
                        text = chatutil.scrub(persona=persona,
                                              text=text,
                                              thoughts_mode="discard",
                                              markup=None,
                                              add_persona=False)

                        # Avatar speech and subtitling
                        if speech_enabled:  # If TTS enabled, send final message text to TTS preprocess queue (this always uses lipsync)
                            logger.info("ai_turn.ai_turn_task.on_done: sending final (non-thought) message content for translation, TTS, and subtitling")
                            self.avatar_controller.send_text_to_tts(config=self.avatar_record,
                                                                    text=text,
                                                                    video_offset=librarian_config.avatar_config.video_offset)

                        # Update avatar emotion one last time, from the final message text
                        logger.info("ai_turn.ai_turn_task.on_done: updating emotion from final (non-thought) message content")
                        self.avatar_controller.update_emotion_from_text(config=self.avatar_record,
                                                                        text=text)

                        # Update linearized chat view
                        logger.info("ai_turn.ai_turn_task.on_done: updating chat view with final message")
                        delete_streaming_chat_message()  # if we are called by docs nomatch, the in-progress message shouldn't exist in the GUI; then this no-ops.
                        self.view.add_complete_message(node_id)
                        self.update_context_fill_indicator()  # AI message completed -> context grew

                        logger.info("ai_turn.ai_turn_task.on_done: all done.")

                # def _parse_toolcall(request_record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
                #     """Given a tool call request record in OpenAI format, return tool call ID and function name."""
                #     tool_call_id = request_record["id"] if "id" in request_record else None
                #     function_name = None
                #     if "type" in request_record and request_record["type"] == "function":
                #         if "function" in request_record:
                #             function_record = request_record["function"]
                #             if "name" in function_record:
                #                 function_name = function_record["name"]
                #     return tool_call_id, function_name

                def on_tools_start(tool_calls: List[Dict]) -> None:
                    if self.gui_updates_safe:
                        self.avatar_controller.start_data_eyes(config=self.avatar_record)

                        # # HACK: If websearch is present *anywhere* among the tool calls in this message,
                        # #       light up the web access indicator for the whole tool call processing step.
                        # #       Often there is just one tool call, so it's fine.
                        # ids_and_names = [_parse_toolcall(request_record) for request_record in tool_calls]
                        # names = [name for _id, name in ids_and_names]
                        # if "websearch" in names:
                        #     if self.indicator_glow_animation is not None:
                        #         self.indicator_glow_animation.reset()  # start new pulsation cycle
                        #     dpg.show_item(self.web_indicator_widget)

                def on_call_lowlevel_start(tool_call_id: str, function_name: str, arguments: Dict[str, Any]) -> None:
                    if self.gui_updates_safe:
                        if function_name in web_access_tool_names:
                            if self.indicator_glow_animation is not None:
                                self.indicator_glow_animation.reset()  # start new pulsation cycle
                            dpg.show_item(self.web_indicator_widget)

                def on_call_lowlevel_done(tool_call_id: str, function_name: str, status: str, text: str) -> None:
                    if self.gui_updates_safe:
                        if function_name in web_access_tool_names:
                            dpg.hide_item(self.web_indicator_widget)

                def on_tool_done(node_id: str) -> None:
                    self.app_state["HEAD"] = node_id  # update just in case of Ctrl+C or crash during tool calls
                    task_env.text = io.StringIO()  # for next AI message (in case of tool calls)
                    if self.gui_updates_safe:
                        delete_streaming_chat_message()  # it shouldn't exist when this triggers, but robustness.
                        self.view.add_complete_message(node_id)
                        self.update_context_fill_indicator()  # tool result added -> context grew

                def on_tools_done() -> None:
                    if self.gui_updates_safe:
                        # dpg.hide_item(self.web_indicator_widget)
                        self.avatar_controller.stop_data_eyes(config=self.avatar_record)

                def on_prompt_ready(history) -> None:
                    # logger.info("DPGChatController.ai_turn.on_prompt_ready: full prompt (message history) that will be sent to the LLM:")
                    # logger.info("=" * 80)
                    # for item in history:
                    #     logger.info(item)
                    # logger.info("=" * 80)
                    pass

                # `scaffold.ai_turn` / `scaffold.retry_tool_calls` are synchronous calls, which lets us use
                # the context manager for the idle-off override. The same callback bundle serves both: the
                # override re-runs one denied tool call on a new branch, then continues via `ai_turn`.
                common_callbacks = dict(on_docs_start=on_docs_start,
                                        on_docs_done=on_docs_done,
                                        on_llm_start=on_llm_start,
                                        on_prompt_ready=on_prompt_ready,  # debug/info hook
                                        on_llm_progress=on_llm_progress,
                                        on_llm_done=on_done,
                                        on_nomatch_done=on_done,
                                        on_tools_start=on_tools_start,
                                        on_call_lowlevel_start=on_call_lowlevel_start,
                                        on_call_lowlevel_done=on_call_lowlevel_done,
                                        on_tool_done=on_tool_done,
                                        on_tools_done=on_tools_done)
                with self.avatar_controller.idle_override(config=self.avatar_record):
                    if _retry_tool_node_id is None:
                        new_head_node_id = scaffold.ai_turn(llm_settings=self.llm_settings,
                                                            datastore=self.datastore,
                                                            retriever=self.retriever,
                                                            head_node_id=self.app_state["HEAD"],
                                                            tools_enabled=self.app_state["tools_enabled"],
                                                            continue_=continue_,
                                                            docs_query=docs_query,
                                                            docs_num_results=librarian_config.docs_num_results,
                                                            speculate=self.app_state["speculate_enabled"],
                                                            markup="markdown",  # TODO: check if we actually use the `markup` argument for anything but thought blocks - those are in any case emitted as-is (and formatted at render time).
                                                            **common_callbacks)
                    else:
                        new_head_node_id = scaffold.retry_tool_calls(llm_settings=self.llm_settings,
                                                                     datastore=self.datastore,
                                                                     retriever=self.retriever,
                                                                     tool_node_id=_retry_tool_node_id,
                                                                     tools_enabled=self.app_state["tools_enabled"],
                                                                     speculate=self.app_state["speculate_enabled"],
                                                                     markup="markdown",
                                                                     docs_num_results=librarian_config.docs_num_results,
                                                                     **common_callbacks)
                self.app_state["HEAD"] = new_head_node_id
            finally:
                if self.gui_updates_safe:
                    dpg.disable_item(self.chat_stop_generation_button_widget)
                    self.avatar_controller.stop_data_eyes(config=self.avatar_record)  # make sure the data eyes effect ends (unless app shutting down, in which case we shouldn't start new GUI animations)
                    if not speech_enabled:  # make sure the generic talking animation ends (if we invoked it)
                        api.avatar_stop_talking(self.avatar_record.avatar_instance_id)
                    # Also make sure that the AI-turn-scoped processing indicators hide. The INDEXING
                    # indicator is intentionally *not* touched here — it has its own polling-driven
                    # lifecycle (background commits run independent of any AI turn).
                    dpg.hide_item(self.docs_search_indicator_widget)
                    dpg.hide_item(self.web_indicator_widget)
                    dpg.hide_item(self.llm_indicator_widget)
        self.ai_turn_task_manager.submit(ai_turn_task, env())

    def stop_ai_turn(self) -> None:
        """Interrupt the AI, i.e. stop ongoing text generation.

        Useful to have in case you (as the user) see the AI has misunderstood your question,
        so that there's no need to wait for a complete response.
        """
        if self.gui_updates_safe:
            dpg.disable_item(self.chat_stop_generation_button_widget)
        # Cancelling all background tasks from the AI turn specific task manager stops the task (co-operatively, so it shuts down gracefully).
        self.ai_turn_task_manager.clear()
