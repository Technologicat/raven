"""Chat controller.

This module renders a linearized chat view of the current branch, and contains the scaffold to GUI integration
that controls chatting with the AI.
"""

# TODO:
#   - `DPGChatController` orchestrates
#     - owns a `DPGLinearizedChatView`
#     - its `app_state` is really a controller-specific state (if several controllers in the same app)
#   - `DPGLinearizedChatView` displays
#     - owns chat message widgets
#     - `DPGChatMessage` and its descendants display individual messages
#     - the view's `gui_parent` should be a panel (DPG child window)
#
# TODO: check DPG tags - shouldn't directly use anything defined on the main app side (have a constructor parameter for each of these)
#
# TODO: Are these classes a bit too friendly with each other? A lot of state lives in the `DPGChatController`, and the lower levels reach into that.

__all__ = ["DPGChatController"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import collections
import concurrent.futures
import io
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union
import uuid

import dearpygui.dearpygui as dpg

from unpythonic.env import env

from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa  # https://github.com/juliettef/IconFontCppHeaders
from ..vendor import DearPyGui_Markdown as dpg_markdown  # https://github.com/IvanNazaruk/DearPyGui-Markdown

from ..client import api  # Raven-server support
from ..client.avatar_controller import DPGAvatarController

from ..common import bgtask

from ..common.gui import animation as gui_animation

from . import chattree
from . import chatutil
from . import config as librarian_config
from . import hybridir
from . import llmclient
from . import scaffold

gui_config = librarian_config.gui_config  # shorthand, this is used a lot

# --------------------------------------------------------------------------------

gui_role_icons = {"assistant": "icon_ai_texture",
                  "system": "icon_system_texture",
                  "tool": "icon_tool_texture",
                  "user": "icon_user_texture",
                  }
role_colors = {"assistant": {"front": gui_config.chat_color_ai_front, "back": gui_config.chat_color_ai_back},
               "system": {"front": gui_config.chat_color_system_front, "back": gui_config.chat_color_system_back},
               "tool": {"front": gui_config.chat_color_tool_front, "back": gui_config.chat_color_tool_back},
               "user": {"front": gui_config.chat_color_user_front, "back": gui_config.chat_color_user_back},
               }

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
                 gui_parent: Union[int, str],
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

    def _get_text(self):
        with self.paragraphs_lock:
            return "\n".join(paragraph["text"] for paragraph in self.paragraphs)
    text = property(fget=_get_text,
                    doc="Full text of this GUI chat message. Read-only.")

    def _get_next_or_prev_sibling_in_datastore(self,
                                               node_id: str,
                                               direction: str = "next") -> Optional[str]:
        siblings, node_index = self.parent_view.chat_controller.datastore.get_siblings(node_id)
        if siblings is None:
            return None
        if direction == "next":
            if node_index < len(siblings) - 1:
                return siblings[node_index + 1]
        else:  # direction == "prev":
            if node_index > 0:
                return siblings[node_index - 1]
        return None  # no sibling found

    def build(self,
              role: str,
              persona: Optional[str],
              node_id: Optional[str]) -> None:
        """Build the GUI widgets for this chat message instance, thus rendering the chat message (and its buttons and such) in the GUI.

        `role`: One of the roles supported by `raven.librarian.llmclient`.
                Typically, one of "assistant", "system", "tool", or "user".

        `persona`: The persona name speaking `text`, or `None` if the role has no persona name ("system" and "tool" are like this).

                   If you are creating a new chat message, use `persona=llm_settings.personas.get(role, None)`
                   (where `role` is one of "assistant", "system", "tool", "user") to get the current session's persona.

                   If you are editing a message from an existing chat node, use
                   `persona=node_payload["general_metadata"]["persona"]` to get the stored persona
                   (which may be different from the current session's, e.g. if the AI character has been changed).

        `node_id`: The chat node ID of this message in the datastore, if applicable.
                   (Streaming messages do not have a node yet.)

        NOTE: You still need to `add_paragraph` the text you want to show in the chat message widget.
              It's done this way to be able to handle messages that *contain* thought blocks
              (i.e. any complete message from a thinking model), because the `is_thought` state
              needs to be different for the think-block and final-message segments.

        NOTE: `DPGCompleteChatMessage` parses the content from the chat node add adds the text automatically.
        """
        global gui_role_icons  # intent only
        global role_colors  # intent only

        self.role = role
        self.persona = persona
        self.node_id = node_id

        # clear old GUI content (needed if rebuilding)
        dpg.delete_item(self.gui_container_group, children_only=True)

        # lay out the role icon and the text content horizontally
        icon_and_text_container_group = dpg.add_group(horizontal=True,
                                                      tag=f"chat_icon_and_text_container_group_{self.gui_uuid}",
                                                      parent=self.gui_container_group)

        # ----------------------------------------
        # role icon

        # TODO: add icons and colors for system and tool; improve user/AI icons
        icon_drawlist = dpg.add_drawlist(width=(2 * gui_config.margin + gui_config.chat_icon_size),
                                         height=(2 * gui_config.margin + gui_config.chat_icon_size),
                                         tag=f"chat_icon_drawlist_{self.gui_uuid}",
                                         parent=icon_and_text_container_group)  # empty drawlist acts as placeholder if no icon
        if role in gui_role_icons:
            dpg.draw_image(gui_role_icons[role],
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
            payload_datetime = node_payload["general_metadata"]["datetime"]  # of the active revision!
            node_active_revision = self.parent_view.chat_controller.datastore.get_revision(node_id)
            dpg.add_text(f"{payload_datetime} R{node_active_revision}", color=(120, 120, 120), parent=text_vertical_layout_group)

        # render the actual text
        self.gui_text_group = dpg.add_group(tag=f"chat_message_text_container_group_{self.gui_uuid}",
                                            parent=text_vertical_layout_group)  # create another group to act as container so that we can update/replace just the text easily
        # NOTE: We now have an empty group, for `add_paragraph`/`replace_last_paragraph`.

        # Show LLM performance statistics if linked to a chat node, and the chat node has them
        if role == "assistant" and node_id is not None:
            ai_message_node_payload = self.parent_view.chat_controller.datastore.get_payload(node_id)
            if (generation_metadata := ai_message_node_payload.get("generation_metadata", None)) is not None:
                n_tokens = generation_metadata["n_tokens"]
                dt = generation_metadata["dt"]
                speed = n_tokens / dt
                dpg.add_text(f"[{n_tokens}t, {dt:0.2f}s, {speed:0.2f}t/s]",
                             color=(120, 120, 120),
                             parent=text_vertical_layout_group)

        # If there is no datastore chat node attached to this message, it doesn't need the datastore control buttons.
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
        n_message_buttons = 8
        dpg.add_spacer(width=gui_config.chat_text_w - n_message_buttons * (gui_config.toolbutton_w + 8) - 64,  # 8 = DPG outer margin; 32 = some space for sibling counter
                       parent=buttons_horizontal_layout_group)

        self.build_buttons(gui_parent=buttons_horizontal_layout_group)

        # ----------------------------------------
        # chat turn end spacers and line

        dpg.add_spacer(height=4,
                       tag=f"chat_turn_end_spacer1_{self.gui_uuid}",
                       parent=self.gui_container_group)

        if role in role_colors:
            dpg.add_drawlist(height=1,
                             width=(gui_config.chat_text_w + 64),
                             tag=f"chat_turn_end_drawlist_{self.gui_uuid}",
                             parent=self.gui_container_group)
            dpg.draw_rectangle((64, 0), (gui_config.chat_text_w + 64, 1),
                               color=(80, 80, 80),
                               fill=(80, 80, 80),
                               parent=f"chat_turn_end_drawlist_{self.gui_uuid}")

        dpg.add_spacer(height=4,
                       tag=f"chat_turn_end_spacer2_{self.gui_uuid}",
                       parent=self.gui_container_group)

    def add_paragraph(self, text: str, is_thought: bool) -> None:
        """Add a new paragraph of text to this widget.

        `is_thought`: Whether this paragraph is (part of) a `<think>...</think>` block.
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
                     Can be different from the old state.
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
                assert False
            # dpg.delete_item(self.gui_text_group, children_only=True)  # how to clear all old text if we ever need to
            role = self.role
            role_color = role_colors[role]["front"] if role in role_colors else "#ffffff"
            think_color = librarian_config.gui_config.chat_color_think_front
            for idx, paragraph in enumerate(self.paragraphs):
                if paragraph["rendered"]:
                    continue
                assert "widget" not in paragraph  # a paragraph that hasn't been rendered has no GUI text widget associated with it
                text = paragraph["text"].strip()
                if text:  # don't bother if text is blank
                    # TODO: Add collapsible thought blocks to the GUI. For now, we just replace the tags with something that doesn't look like HTML to avoid confusing the Markdown renderer (which drops unknown tags).
                    text = text.replace("<tool_call>", "**>>>Tool call>>>**")
                    text = text.replace("</tool_call>", "**<<<Tool call<<<**")
                    text = text.replace("<think>", "**>>>Thinking>>>**")
                    text = text.replace("</think>", "**<<<Thinking<<<**")
                    color = think_color if paragraph["is_thought"] else role_color
                    colorized_text = f"<font color='{color}'>{text}</font>"
                    widget = dpg_markdown.add_text(colorized_text,
                                                   wrap=gui_config.chat_text_w,
                                                   parent=self.gui_text_group)
                    paragraph["widget"] = widget
                    dpg.set_item_alias(widget, f"chat_message_text_{role}_paragraph_{idx}_{self.gui_uuid}")
                paragraph["rendered"] = True

    def demolish(self) -> None:
        """The opposite of `build`: delete all GUI widgets belonging to this instance.

        If you use `DPGLinearizedChatView.build`, it takes care of clearing all chat message GUI widgets automatically,
        and you do not need to call this.

        If you are editing the linearized chat view directly, this should be called before deleting
        the `DPGChatMessage` instance.

        The main use case is switching a streaming message to a completed one when the streaming is done,
        without regenerating the whole linearized chat view.
        """
        with self.paragraphs_lock:
            self.role = None
            self.persona = None
            self.paragraphs = []
            self.gui_text_group = None
            self.gui_button_callbacks = {}  # deleting GUI items, so clear the stashed callbacks too.
            try:
                dpg.delete_item(self.gui_container_group, children_only=True)  # clear old GUI content (needed if rebuilding)
            except SystemError:  # the group went bye-bye (app shutdown)
                pass

    def build_buttons(self,
                      gui_parent: Union[int, str]) -> None:
        """Build the set of control buttons for a single chat message in the GUI.

        `gui_parent`: DPG tag or ID of the GUI widget (typically a group) to add the buttons to.

                      This is not simply `self.gui_parent` due to other layout performed by `build`.
        """
        role = self.role
        persona = self.persona
        node_id = self.node_id

        g = dpg.add_group(horizontal=True, tag=f"{role}_message_buttons_group_{self.gui_uuid}", parent=gui_parent)

        # dpg.add_text("[0 t, 0 s, ∞ t/s]", color=(180, 180, 180), tag=f"performance_stats_text_ai_{self.gui_uuid}", parent=g)  # TODO: add the performance stats

        # dpg.add_spacer(tag=f"ai_message_buttons_spacer_{self.gui_uuid}",
        #                parent=g)

        def copy_message_to_clipboard_callback() -> None:
            shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
            # Note we only add the role name when we include also the node ID.
            # Omitting the name in regular mode improves convenience for copy-pasting an existing question into the chat field.
            formatted_message = self.parent_view.format_chat_message_for_clipboard(message_number=None,  # a single message copied to clipboard does not need a sequential number
                                                                                   role=role,
                                                                                   persona=persona,
                                                                                   text=self.text,
                                                                                   add_heading=shift_pressed)

            if shift_pressed:
                node_payload = self.parent_view.chat_controller.datastore.get_payload(node_id)  # auto-selects active revision  TODO: later (chat editing), we need to set the revision to load
                payload_datetime = node_payload["general_metadata"]["datetime"]  # of the active revision!
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
                                                                 original_theme=self.parent_view.themes_and_fonts.global_theme,
                                                                 duration=gui_config.acknowledgment_duration))
        self.gui_button_callbacks["copy"] = copy_message_to_clipboard_callback
        copy_message_button = dpg.add_button(label=fa.ICON_COPY,
                                             callback=copy_message_to_clipboard_callback,
                                             width=gui_config.toolbutton_w,
                                             tag=f"message_copy_to_clipboard_button_{self.gui_uuid}",
                                             parent=g)
        dpg.bind_item_font(copy_message_button, self.parent_view.themes_and_fonts.icon_font_solid)
        dpg.bind_item_theme(copy_message_button, "disablable_button_theme")  # tag
        copy_message_tooltip = dpg.add_tooltip(copy_message_button)
        copy_message_tooltip_text = dpg.add_text("Copy message to clipboard\n    no modifier: as-is\n    with Shift: include message node ID", parent=copy_message_tooltip)

        # Only AI messages can be rerolled
        if role == "assistant":
            def reroll_message_callback():  # TODO: parameterize this - callback needs to come from main app (at least for the `ai_turn` part)
                # Find this AI message in the chat history
                for k, dpg_chat_message in enumerate(reversed(self.parent_view.chat_controller.current_chat_history)):
                    if dpg_chat_message.node_id == node_id:
                        break
                # `k` is now how many messages must be popped from the end to reach this one
                assert k < len(self.parent_view.chat_controller.current_chat_history) - 3  # should have at least the system prompt, the AI's initial greeting, and the user's first message remaining
                # Rewind the linearized chat history in the GUI
                for _ in range(k):
                    old_dpg_chat_message = self.parent_view.chat_controller.current_chat_history.pop(-1)
                    old_dpg_chat_message.demolish()

                # Handle the RAG query: find the latest user message (above this AI message)
                user_message_text = None
                for dpg_chat_message in reversed(self.parent_view.chat_controller.current_chat_history):  # ...what's remaining of the history, anyway
                    if dpg_chat_message.role == "user":
                        user_message_text = dpg_chat_message.text
                        break

                # Remove the AI message from GUI
                self.parent_view.chat_controller.app_state["HEAD"] = self.parent_view.chat_controller.datastore.get_parent(node_id)
                old_dpg_chat_message = self.parent_view.chat_controller.current_chat_history.pop(-1)  # once more, with feeling!
                old_dpg_chat_message.demolish()

                # Generate new AI message
                self.parent_view.chat_controller.ai_turn(docs_query=user_message_text)
            reroll_enabled = (node_id is not None and node_id != self.parent_view.chat_controller.app_state["new_chat_HEAD"])  # The AI's initial greeting can't be rerolled
            if reroll_enabled:
                self.gui_button_callbacks["reroll"] = reroll_message_callback  # stash it so we can call it from the hotkey handler
            dpg.add_button(label=fa.ICON_RECYCLE,
                           callback=reroll_message_callback,
                           enabled=reroll_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_reroll_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_reroll_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_reroll_button_{self.gui_uuid}", "disablable_button_theme")  # tag
            reroll_tooltip = dpg.add_tooltip(f"message_reroll_button_{self.gui_uuid}")  # tag
            dpg.add_text("Reroll AI response (create new sibling) [Ctrl+R]", parent=reroll_tooltip)
        else:
            dpg.add_spacer(width=gui_config.toolbutton_w, height=1, parent=g)

        if role == "assistant":
            def speak_message_callback():  # TODO: parameterize this - callback needs to come from main app
                if self.parent_view.chat_controller.app_state["avatar_speech_enabled"]:
                    unused_message_role, unused_message_persona, message_text = chatutil.get_node_message_text_without_persona(self.parent_view.chat_controller.datastore, node_id)
                    # Send only non-thought message content to TTS
                    message_text = chatutil.scrub(persona=self.parent_view.chat_controller.llm_settings.personas.get("assistant", None),
                                                  text=message_text,
                                                  thoughts_mode="discard",
                                                  markup=None,
                                                  add_persona=False)
                    self.parent_view.chat_controller.avatar_controller.send_text_to_tts(config=self.parent_view.chat_controller.avatar_record,
                                                                                        text=message_text,
                                                                                        voice=librarian_config.avatar_config.voice,
                                                                                        voice_speed=librarian_config.avatar_config.voice_speed,
                                                                                        video_offset=librarian_config.avatar_config.video_offset)

                    # Acknowledge the action in the GUI.
                    gui_animation.animator.add(gui_animation.ButtonFlash(message="Sent to avatar!",
                                                                         target_button=speak_message_button,
                                                                         target_tooltip=speak_message_tooltip,
                                                                         target_text=speak_message_tooltip_text,
                                                                         original_theme=self.parent_view.themes_and_fonts.global_theme,
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
            dpg.bind_item_theme(speak_message_button, "disablable_button_theme")  # tag
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
        dpg.bind_item_theme(f"chat_edit_button_{self.gui_uuid}", "disablable_button_theme")  # tag
        edit_tooltip = dpg.add_tooltip(f"chat_edit_button_{self.gui_uuid}")  # tag
        dpg.add_text("Edit (revise)", parent=edit_tooltip)

        dpg.add_button(label=fa.ICON_CODE_BRANCH,
                       callback=lambda: None,  # TODO
                       enabled=False,
                       width=gui_config.toolbutton_w,
                       tag=f"message_new_branch_button_{self.gui_uuid}",
                       parent=g)
        dpg.bind_item_font(f"message_new_branch_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
        dpg.bind_item_theme(f"message_new_branch_button_{self.gui_uuid}", "disablable_button_theme")  # tag
        new_branch_tooltip = dpg.add_tooltip(f"message_new_branch_button_{self.gui_uuid}")  # tag
        dpg.add_text("Branch from this node", parent=new_branch_tooltip)

        # NOTE: We disallow deleting the system prompt and the AI's initial greeting, as well as any message that is not linked to a chat node in the datastore.
        delete_enabled = (node_id is not None and
                          node_id not in (self.parent_view.chat_controller.app_state["system_prompt_node_id"],
                                          self.parent_view.chat_controller.app_state["new_chat_HEAD"]))
        dpg.add_button(label=fa.ICON_TRASH_CAN,
                       callback=lambda: None,  # TODO
                       enabled=False,  # TODO: use `delete_enabled` once delete is implemented
                       width=gui_config.toolbutton_w,
                       tag=f"message_delete_branch_button_{self.gui_uuid}",
                       parent=g)
        dpg.bind_item_font(f"message_delete_branch_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
        dpg.bind_item_theme(f"message_delete_branch_button_{self.gui_uuid}", "disablable_button_theme")  # tag
        delete_branch_tooltip = dpg.add_tooltip(f"message_delete_branch_button_{self.gui_uuid}")  # tag

        c_red = '<font color="(255, 96, 96)">'
        c_end = '</font>'
        dpg_markdown.add_text(f"Delete branch (this node and {c_red}**all**{c_end} descendants)", parent=delete_branch_tooltip)

        def make_navigate_to_prev_sibling(message_node_id: str) -> Callable:
            def navigate_to_prev_sibling_callback():
                node_id = self._get_next_or_prev_sibling_in_datastore(message_node_id, direction="prev")
                if node_id is not None:
                    self.parent_view.chat_controller.app_state["HEAD"] = node_id
                    self.parent_view.build()
            return navigate_to_prev_sibling_callback

        def make_navigate_to_next_sibling(message_node_id: str) -> Callable:
            def navigate_to_next_sibling_callback():
                node_id = self._get_next_or_prev_sibling_in_datastore(message_node_id, direction="next")
                if node_id is not None:
                    self.parent_view.chat_controller.app_state["HEAD"] = node_id
                    self.parent_view.build()
            return navigate_to_next_sibling_callback

        # Only messages attached to a datastore chat node can have siblings in the datastore
        if node_id is not None:
            siblings, node_index = self.parent_view.chat_controller.datastore.get_siblings(node_id)
            prev_enabled = (node_index is not None and node_index > 0)
            next_enabled = (node_index is not None and node_index < len(siblings) - 1)
            navigate_to_prev_sibling_callback = make_navigate_to_prev_sibling(node_id)
            navigate_to_next_sibling_callback = make_navigate_to_next_sibling(node_id)
            if prev_enabled:
                self.gui_button_callbacks["prev"] = navigate_to_prev_sibling_callback
            if next_enabled:
                self.gui_button_callbacks["next"] = navigate_to_next_sibling_callback

            dpg.add_button(label=fa.ICON_ANGLE_LEFT,
                           callback=navigate_to_prev_sibling_callback,
                           enabled=prev_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_prev_branch_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_prev_branch_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_prev_branch_button_{self.gui_uuid}", "disablable_button_theme")  # tag
            prev_branch_tooltip = dpg.add_tooltip(f"message_prev_branch_button_{self.gui_uuid}")  # tag
            dpg.add_text("Switch to previous sibling [Ctrl+Left]", parent=prev_branch_tooltip)

            dpg.add_button(label=fa.ICON_ANGLE_RIGHT,
                           callback=navigate_to_next_sibling_callback,
                           enabled=next_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_next_branch_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_next_branch_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_next_branch_button_{self.gui_uuid}", "disablable_button_theme")  # tag
            next_branch_tooltip = dpg.add_tooltip(f"message_next_branch_button_{self.gui_uuid}")  # tag
            dpg.add_text("Switch to next sibling [Ctrl+Right]", parent=next_branch_tooltip)

            if siblings is not None:
                dpg.add_text(f"{node_index + 1} / {len(siblings)}", parent=g)
        else:
            # Add the two spacers separately so we get the same margins as with two separate buttons
            dpg.add_spacer(width=gui_config.toolbutton_w, height=1, parent=g)
            dpg.add_spacer(width=gui_config.toolbutton_w, height=1, parent=g)


class DPGCompleteChatMessage(DPGChatMessage):
    def __init__(self,
                 node_id: str,
                 gui_parent: Union[int, str],
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
        role, persona, text = chatutil.get_node_message_text_without_persona(self.parent_view.chat_controller.datastore, self.node_id)  # TODO: later (chat editing), we need to set the revision to load
        super().build(role=role,
                      persona=persona,
                      node_id=self.node_id)

        paragraphs = text.split("\n")
        inside_think_block = False
        for paragraph in paragraphs:
            # Detect think block state (TODO: improve; very rudimentary and brittle for now)
            p = paragraph.strip()
            if p == "<think>":
                inside_think_block = True
            elif p == "</think>":
                inside_think_block = False

            self.add_paragraph(paragraph,
                               is_thought=(inside_think_block or (p == "</think>")))  # easiest to special-case the closing tag


class DPGStreamingChatMessage(DPGChatMessage):
    def __init__(self,
                 gui_parent: Union[int, str],
                 parent_view: "DPGLinearizedChatView"):
        """A chat message being streamed live from the LLM, displayed in the linearized chat view.

        `gui_parent`: DPG tag or ID of the GUI widget (typically child window or group) to add the chat message to.
        `parent_view`: The linearized chat view widget this chat message is rendered in (and is owned by).

        Starts as blank. Use the `add_paragraph` and/or `replace_last_paragraph` methods to add text.

        To replace the streaming message with a completed message, call the streaming message's
        `demolish` method first to remove it from the GUI.
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
                 gui_parent: Union[str, int],  # panel (DPG child window)
                 chat_controller: "DPGChatController"):
        self.themes_and_fonts = themes_and_fonts
        self.gui_parent = gui_parent
        self.gui_uuid = str(uuid.uuid4())  # used in GUI widget tags
        self.chat_controller = chat_controller

        # TODO: We can later use the existence of the chat container group widget for double-buffering (can render a new group and then switch it in)
        self.chat_messages_container_group_widget = dpg.add_group(tag=f"chat_messages_container_group_{self.gui_uuid}",
                                                                  parent=gui_parent)

    def scroll_to_end(self,
                      max_wait_frames: int = 50) -> None:
        """Scroll this linearized chat view to end.

        `max_wait_frames`: If `max_wait_frames > 0`, wait at most for that may frames
                           for the chat panel to get a nonzero `max_y_scroll`.

                           Some waiting is usually needed at least at app startup
                           before the GUI settles.

        NOTE: When called from the main thread, `max_wait_frames` must be 0, as any
              attempt to wait would hang the main thread's explicit render loop.

              Setting `max_wait_frames=0` also has the effect of not logging the current
              frame number, because `dpg.get_frame_count()` would need the render thread mutex:
                  https://github.com/hoffstadt/DearPyGui/issues/2366

              When called from any other thread (also event handlers), waiting is fine.
        """
        max_y_scroll = dpg.get_y_scroll_max(self.gui_parent)
        for elapsed_frames in range(max_wait_frames):
            if max_y_scroll > 0:
                break
            dpg.split_frame()
            max_y_scroll = dpg.get_y_scroll_max(self.gui_parent)
        plural_s = "s" if elapsed_frames != 1 else ""
        waited_str = f" (after waiting for {elapsed_frames} frame{plural_s})" if elapsed_frames > 0 else " (no waiting was needed)"
        frames_str = f" frame {dpg.get_frame_count()}" if max_wait_frames > 0 else ""
        logger.info(f"DPGLinearizedChatView.scroll_to_end:{frames_str}{waited_str}: max_y_scroll = {max_y_scroll}")
        dpg.set_y_scroll(self.gui_parent, max_y_scroll)

    def get_chatlog_as_markdown(self, include_metadata: bool) -> Optional[str]:
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
                text = message["content"]
                formatted_message = self.format_chat_message_for_clipboard(message_number=message_number,
                                                                           role=role,
                                                                           persona=persona,
                                                                           text=text,
                                                                           add_heading=True)  # In the full chatlog, the message numbers and role names are important, so always include them.
                if include_metadata:
                    payload_datetime = node_payload["general_metadata"]["datetime"]  # of the active revision!
                    node_active_revision = self.chat_controller.datastore.get_revision(dpg_chat_message.node_id)
                    header = f"- *Node ID*: `{dpg_chat_message.node_id}`\n- *Revision date*: {payload_datetime}\n- *Revision number*: {node_active_revision}\n\n"  # yes, it'll say `None` when no node ID is available (incoming streaming message), which is exactly what we want.
                else:
                    header = ""
                output_text.write(f"{header}{formatted_message}\n\n{'-' * 80}\n\n")

            return output_text.getvalue()

    # TODO: This wants to be a separate helper function (used by both `DPGChatMessage` and `DPGLinearizedChatView`)
    def format_chat_message_for_clipboard(self,
                                          message_number: Optional[int],
                                          role: str,
                                          persona: Optional[str],
                                          text: str,
                                          add_heading: bool) -> str:
        """Format a chat message for copying to clipboard, by adding a metadata header as Markdown.

        As a preprocessing step, the role name is stripped from the beginning of each line in `message_text`.
        It is then re-added in a unified form, using `message_role` as the role.

        `message_number`: The sequential number of the message in the current linearized view.
                          If `None`, the number part in the formatted output is omitted.

        `role`: One of the roles supported by `raven.librarian.llmclient`.
                Typically, one of "assistant", "system", "tool", or "user".

        `persona`: The persona name speaking `text`, or `None` if the role has no persona name ("system" and "tool" are like this).

                   If you are creating a new chat message, use `persona=llm_settings.personas.get(role, None)`
                   (where `role` is one of "assistant", "system", "tool", "user") to get the current session's persona.

                   If you are editing a message from an existing chat node, use
                   `persona=node_payload["general_metadata"]["persona"]` to get the stored persona
                   (which may be different from the current session's, e.g. if the AI character has been changed).

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

    def add_complete_message(self,
                             node_id: str,
                             scroll_to_end: bool = True) -> DPGCompleteChatMessage:
        """Append the given chat node to the end of the linearized chat view in the GUI."""
        with self.chat_controller.current_chat_history_lock:
            dpg_chat_message = DPGCompleteChatMessage(gui_parent=self.chat_messages_container_group_widget,
                                                      parent_view=self,
                                                      node_id=node_id)
            self.chat_controller.current_chat_history.append(dpg_chat_message)
        if scroll_to_end:
            dpg.split_frame()
            self.scroll_to_end()
        return dpg_chat_message

    # TODO: does this `build` really belong in `DPGLinearizedChatView` or in `DPGChatController`?
    def build(self,
              head_node_id: Optional[str] = None) -> None:
        """Build the linearized chat view in the GUI, linearizing up from `head_node_id`.

        As side effects:

          - Update the controller's `current_chat_history`.
          - If `head_node_id` is an AI message, update the avatar's emotion from that
            (using the node's current payload revision).
        """
        if head_node_id is None:  # use current HEAD from app_state?
            head_node_id = self.chat_controller.app_state["HEAD"]
        node_id_history = self.chat_controller.datastore.linearize_up(head_node_id)
        with self.chat_controller.current_chat_history_lock:
            self.chat_controller.current_chat_history.clear()
            dpg.delete_item(self.chat_messages_container_group_widget,
                            children_only=True)  # clear old content from GUI
            for node_id in node_id_history:
                self.add_complete_message(node_id=node_id,
                                          scroll_to_end=False)  # we scroll just once, when done
        # Update avatar emotion from the message text (use only non-thought message content)
        role, unused_persona, text = chatutil.get_node_message_text_without_persona(self.chat_controller.datastore, head_node_id)
        if role == "assistant":
            logger.info("DPGLinearizedChatView.build: linearized chat view new HEAD node is an AI message; updating avatar emotion from (non-thought) message content")
            text = chatutil.scrub(persona=self.chat_controller.llm_settings.personas.get("assistant", None),
                                  text=text,
                                  thoughts_mode="discard",
                                  markup=None,
                                  add_persona=False)
            self.chat_controller.avatar_controller.update_emotion_from_text(config=self.chat_controller.avatar_record,
                                                                            text=text)
        dpg.split_frame()
        self.scroll_to_end()

# --------------------------------------------------------------------------------
# Scaffold to GUI integration

class DPGChatController:
    def __init__(self,
                 llm_settings: env,
                 datastore: chattree.Forest,
                 retriever: Optional[hybridir.HybridIR],  # document database
                 app_state: env,  # mainly HEAD, but also some option flags
                 avatar_controller: DPGAvatarController,  # data eyes, TTS
                 avatar_record: env,  # the avatar instance of the AI in this chat view
                 themes_and_fonts: env,
                 chat_panel_widget: Union[str, int],  # panel / child window
                 indicator_glow_animation: Optional[gui_animation.PulsatingColor],  # the cycle of this animation will be reset when an indicator appears, to make glow work correctly
                 llm_indicator_widget: Union[str, int],  # DPG widget to show while the prompt is being processed by the LLM backend
                 docs_indicator_widget: Union[str, int],  # DPG widget to show while the docs database is being searched
                 web_indicator_widget: Union[str, int],  # DPG widget to show while the websearch tool is being called
                 executor: Optional = None):
        self.llm_settings = llm_settings
        self.datastore = datastore
        self.retriever = retriever
        self.app_state = app_state
        self.avatar_controller = avatar_controller
        self.avatar_record = avatar_record
        self.indicator_glow_animation = indicator_glow_animation
        self.llm_indicator_widget = llm_indicator_widget
        self.docs_indicator_widget = docs_indicator_widget
        self.web_indicator_widget = web_indicator_widget
        self.current_chat_history = []
        self.current_chat_history_lock = threading.RLock()

        self.gui_updates_safe = True  # At app shutdown, they aren't.

        self.view = DPGLinearizedChatView(themes_and_fonts=themes_and_fonts,
                                          gui_parent=chat_panel_widget,
                                          chat_controller=self)

        # TODO: task managers
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor()

        self.task_manager = bgtask.TaskManager(name="librarian_chat_controller",  # for most tasks
                                               mode="concurrent",
                                               executor=executor)
        self.ai_turn_task_manager = bgtask.TaskManager(name="librarian_chat_controller_ai_turn",  # for running the AI's turn, specifically (so that we can easily cancel just that one task when needed)
                                                       mode="concurrent",
                                                       executor=executor)  # same thread poool

    def shutdown(self):
        """Prepare module for app shutdown.

        This signals the background tasks to exit.
        """
        self.gui_updates_safe = False
        self.task_manager.clear(wait=True)
        self.ai_turn_task_manager.clear(wait=True)

    def get_last_message(self) -> Optional[DPGChatMessage]:
        """Return the `DPGChatMessage` for the last currently displayed message. Return `None` if the view is empty."""
        if not self.current_chat_history:
            return None
        dpg_chat_message = self.current_chat_history[-1]
        return dpg_chat_message

    def chat_round(self, user_message_text: str) -> None:  # message text comes from GUI
        """Run a chat round (user and AI).

        This spawns a background task to avoid hanging GUI event handlers,
        since the typical use case is to call `chat_round` from a GUI event handler.

        By sending empty `user_message_text`, it is possible to have the AI generate
        another message without the user writing in between.

        The RAG query is taken from the latest available user message.
        """
        def chat_round_task(task_env: env) -> None:
            if task_env.cancelled:  # while the task was in the queue
                return

            # Only add the user's message to the chat if the user entered any text.
            if user_message_text:
                self.user_turn(text=user_message_text)
                # NOTE: Rudimentary approach to RAG search, using the user's message text as the query. (Good enough to demonstrate the functionality. Improve later.)
                docs_query = user_message_text
            else:
                # Handle the RAG query: find the latest existing user message
                docs_query = None  # if no user message, send `None` as query to AI -> no docs search
                for dpg_chat_message in reversed(self.current_chat_history):
                    if dpg_chat_message.role == "user":
                        docs_query = dpg_chat_message.text
                        break
            if task_env.cancelled:  # during user turn
                return
            self.ai_turn(docs_query=docs_query)
        self.task_manager.submit(chat_round_task, env())

    def user_turn(self, text: str) -> None:
        """Add the user's message to the chat, and append it to the linearized chat view in the GUI."""
        def user_turn_task(task_env: env) -> None:
            if task_env.cancelled:  # while the task was in the queue
                return

            new_head_node_id = scaffold.user_turn(llm_settings=self.llm_settings,
                                                  datastore=self.datastore,
                                                  head_node_id=self.app_state["HEAD"],
                                                  user_message_text=text)
            self.app_state["HEAD"] = new_head_node_id  # as soon as possible, so that not affected by any errors during GUI building
            self.view.add_complete_message(new_head_node_id)
        self.task_manager.submit(user_turn_task, env())

    def ai_turn(self, docs_query: Optional[str]) -> None:  # TODO: implement continue mode
        """Run the AI's response part of a chat round.

        This spawns a background task to avoid hanging GUI event handlers,
        since the reroll GUI event handler calls `ai_turn` directly.
        """
        docs_query = docs_query if self.app_state["docs_enabled"] else None

        def ai_turn_task(task_env: env) -> None:
            if task_env.cancelled:  # while the task was in the queue
                return

            if self.gui_updates_safe:
                dpg.enable_item("chat_stop_generation_button")  # tag

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
                            self.indicator_glow_animation.reset()  # start new pulsation cycle
                        dpg.show_item(self.docs_indicator_widget)

                def on_docs_done(matches: List[Dict]) -> None:
                    if self.gui_updates_safe:
                        dpg.hide_item(self.docs_indicator_widget)
                        self.avatar_controller.stop_data_eyes(config=self.avatar_record)

                def on_llm_start() -> None:
                    if self.gui_updates_safe:
                        nonlocal streaming_chat_message
                        streaming_chat_message = DPGStreamingChatMessage(gui_parent=self.view.chat_messages_container_group_widget,
                                                                         parent_view=self.view)
                        dpg.split_frame()
                        self.view.scroll_to_end()

                        if self.indicator_glow_animation is not None:
                            self.indicator_glow_animation.reset()  # start new pulsation cycle
                        dpg.show_item(self.llm_indicator_widget)  # show prompt processing indicator

                task_env.text = io.StringIO()  # incoming, in-progress paragraph
                task_env.t0 = time.monotonic()  # timestamp of last GUI update
                task_env.n_chunks0 = 0  # chunks received since last GUI update

                task_env.inside_think_block = False

                task_env.emotion_update_interval = 5  # how many complete paragraphs (newline-separated text snippets) to wait between emotion updates (NOTE: Qwen3 likes using newlines liberally)
                task_env.emotion_recent_paragraphs = collections.deque([""] * (4 * task_env.emotion_update_interval))  # buffer with 75% overlap, to stabilize the detection
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

                def on_llm_progress(n_chunks: int, chunk_text: str) -> None:
                    if self.gui_updates_safe and chunk_text:  # avoid triggering on the initial empty chunk (ACK)
                        dpg.hide_item(self.llm_indicator_widget)  # hide prompt processing indicator

                    # If the task is cancelled (`stop_ai_turn` was called), interrupt the LLM, keeping the content received so far.
                    # The scaffold will automatically send the content to `on_llm_done`.
                    if task_env.cancelled or not self.gui_updates_safe:  # TODO: EAFP to avoid TOCTTOU
                        reason = "Cancelled" if task_env.cancelled else "App is shutting down"
                        logger.info(f"ai_turn.ai_turn_task.on_llm_progress: {reason}, stopping text generation.")
                        return llmclient.action_stop

                    # Detect think block state (TODO: improve; very rudimentary and brittle for now)
                    if "<think>" in chunk_text:
                        task_env.inside_think_block = True
                        logger.info("ai_turn.ai_turn_task.on_llm_progress: AI entered thinking state.")
                    elif "</think>" in chunk_text:
                        logger.info("ai_turn.ai_turn_task.on_llm_progress: AI exited thinking state.")
                        task_env.inside_think_block = False

                        if not speech_enabled:  # If TTS is NOT enabled, show the generic talking animation while the LLM is writing (after it is no longer thinking)
                            api.avatar_start_talking(self.avatar_record.avatar_instance_id)

                    task_env.text.write(chunk_text)
                    time_now = time.monotonic()
                    dt = time_now - task_env.t0  # seconds since last GUI update
                    dchunks = n_chunks - task_env.n_chunks0  # chunks since last GUI update
                    if "\n" in chunk_text:  # start new paragraph?
                        task_env.t0 = time_now
                        task_env.n_chunks0 = n_chunks
                        paragraph_text = task_env.text.getvalue()
                        # NOTE: The last paragraph of the AI's reply - for thinking models, commonly the final response - often never gets a "\n", and must be handled in `on_done`.
                        _update_avatar_emotion_from_incoming_text(paragraph_text)
                        # if speech_enabled:  # If TTS enabled, send complete paragraph to TTS preprocess queue
                        #     if not task_env.inside_think_block and "</think>" not in chunk_text:  # not enough, "</think>" can be in the previous chunk(s) in the same "paragraph".
                        #         avatar_controller.send_text_to_tts(config=avatar_record,
                        #                                            text=paragraph_text,
                        #                                            voice=librarian_config.avatar_config.voice,
                        #                                            voice_speed=librarian_config.avatar_config.voice_speed,
                        #                                            video_offset=librarian_config.avatar_config.video_offset)
                        streaming_chat_message.replace_last_paragraph(paragraph_text,
                                                                      is_thought=(task_env.inside_think_block or ("</think>" in chunk_text)))  # easiest to special-case the closing tag
                        streaming_chat_message.add_paragraph("",
                                                             is_thought=task_env.inside_think_block)
                        task_env.text = io.StringIO()
                        dpg.split_frame()
                        self.view.scroll_to_end()
                    # - update at least every 0.5 sec
                    # - update after every 10 chunks, but rate-limited (at least 0.1 sec must have passed since last update)
                    elif dt >= 0.5 or (dt >= 0.25 and dchunks >= 10):  # commit changes to in-progress last paragraph
                        task_env.t0 = time_now
                        task_env.n_chunks0 = n_chunks
                        streaming_chat_message.replace_last_paragraph(task_env.text.getvalue(),
                                                                      is_thought=task_env.inside_think_block)  # at first paragraph, will auto-create it if not created yet
                        dpg.split_frame()
                        self.view.scroll_to_end()

                    # Let the LLM keep generating (if it wants to).
                    return llmclient.action_ack

                def on_done(node_id: str) -> None:   # For both `on_llm_done` and `on_nomatch_done`.
                    self.app_state["HEAD"] = node_id  # update just in case of Ctrl+C or crash during tool calls
                    task_env.text = io.StringIO()  # for next AI message (in case of tool calls)
                    if self.gui_updates_safe:
                        if not speech_enabled:  # If TTS is NOT enabled, stop the generic talking animation now that the LLM is done
                            api.avatar_stop_talking(self.avatar_record.avatar_instance_id)

                        unused_role, unused_persona, text = chatutil.get_node_message_text_without_persona(self.datastore, node_id)

                        # Keep only non-thought content for TTS and emotion update
                        text = chatutil.scrub(persona=self.llm_settings.personas.get("assistant", None),
                                              text=text,
                                              thoughts_mode="discard",
                                              markup=None,
                                              add_persona=False)

                        # Avatar speech and subtitling
                        if speech_enabled:  # If TTS enabled, send final message text to TTS preprocess queue (this always uses lipsync)
                            logger.info("ai_turn.ai_turn_task.on_done: sending final (non-thought) message content for translation, TTS, and subtitling")
                            self.avatar_controller.send_text_to_tts(config=self.avatar_record,
                                                                    text=text,
                                                                    voice=librarian_config.avatar_config.voice,
                                                                    voice_speed=librarian_config.avatar_config.voice_speed,
                                                                    video_offset=librarian_config.avatar_config.video_offset)

                        # Update avatar emotion one last time, from the final message text
                        logger.info("ai_turn.ai_turn_task.on_done: updating emotion from final (non-thought) message content")
                        self.avatar_controller.update_emotion_from_text(config=self.avatar_record,
                                                                        text=text)

                        # Update linearized chat view
                        logger.info("ai_turn.ai_turn_task.on_done: updating chat view with final message")
                        delete_streaming_chat_message()  # if we are called by docs nomatch, the in-progress message shouldn't exist in the GUI; then this no-ops.
                        self.view.add_complete_message(node_id)

                        logger.info("ai_turn.ai_turn_task.on_done: all done.")

                # def _parse_toolcall(request_record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
                #     """Given a tool call request record in OpenAI format, return tool call ID and function name."""
                #     toolcall_id = request_record["id"] if "id" in request_record else None
                #     function_name = None
                #     if "type" in request_record and request_record["type"] == "function":
                #         if "function" in request_record:
                #             function_record = request_record["function"]
                #             if "name" in function_record:
                #                 function_name = function_record["name"]
                #     return toolcall_id, function_name

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

                def on_call_lowlevel_start(toolcall_id: str, function_name: str, arguments: Dict[str, Any]) -> None:
                    if self.gui_updates_safe:
                        if function_name == "websearch":
                            if self.indicator_glow_animation is not None:
                                self.indicator_glow_animation.reset()  # start new pulsation cycle
                            dpg.show_item(self.web_indicator_widget)

                def on_call_lowlevel_done(toolcall_id: str, function_name: str, status: str, text: str) -> None:
                    if self.gui_updates_safe:
                        if function_name == "websearch":
                            dpg.hide_item(self.web_indicator_widget)

                def on_tool_done(node_id: str) -> None:
                    self.app_state["HEAD"] = node_id  # update just in case of Ctrl+C or crash during tool calls
                    task_env.text = io.StringIO()  # for next AI message (in case of tool calls)
                    if self.gui_updates_safe:
                        delete_streaming_chat_message()  # it shouldn't exist when this triggers, but robustness.
                        self.view.add_complete_message(node_id)

                def on_tools_done() -> None:
                    if self.gui_updates_safe:
                        # dpg.hide_item(self.web_indicator_widget)
                        self.avatar_controller.stop_data_eyes(config=self.avatar_record)

                new_head_node_id = scaffold.ai_turn(llm_settings=self.llm_settings,
                                                    datastore=self.datastore,
                                                    retriever=self.retriever,
                                                    head_node_id=self.app_state["HEAD"],
                                                    tools_enabled=self.app_state["tools_enabled"],
                                                    docs_query=docs_query,
                                                    docs_num_results=librarian_config.docs_num_results,
                                                    speculate=self.app_state["speculate_enabled"],
                                                    markup="markdown",
                                                    on_docs_start=on_docs_start,
                                                    on_docs_done=on_docs_done,
                                                    on_llm_start=on_llm_start,
                                                    on_prompt_ready=None,  # debug/info hook
                                                    on_llm_progress=on_llm_progress,
                                                    on_llm_done=on_done,
                                                    on_nomatch_done=on_done,
                                                    on_tools_start=on_tools_start,
                                                    on_call_lowlevel_start=on_call_lowlevel_start,
                                                    on_call_lowlevel_done=on_call_lowlevel_done,
                                                    on_tool_done=on_tool_done,
                                                    on_tools_done=on_tools_done)
                self.app_state["HEAD"] = new_head_node_id
            finally:
                if self.gui_updates_safe:
                    dpg.disable_item("chat_stop_generation_button")  # tag
                    self.avatar_controller.stop_data_eyes(config=self.avatar_record)  # make sure the data eyes effect ends (unless app shutting down, in which case we shouldn't start new GUI animations)
                    if not speech_enabled:  # make sure the generic talking animation ends (if we invoked it)
                        api.avatar_stop_talking(self.avatar_record.avatar_instance_id)
                    # Also make sure that the processing indicators hide
                    dpg.hide_item(self.docs_indicator_widget)
                    dpg.hide_item(self.web_indicator_widget)
                    dpg.hide_item(self.llm_indicator_widget)
        self.ai_turn_task_manager.submit(ai_turn_task, env())

    def stop_ai_turn(self) -> None:
        """Interrupt the AI, i.e. stop ongoing text generation.

        Useful to have in case you see the AI has misunderstood your question,
        so that there's no need to wait for a complete response.
        """
        if self.gui_updates_safe:
            dpg.disable_item("chat_stop_generation_button")  # tag
        # Cancelling all background tasks from the AI turn specific task manager stops the task (co-operatively, so it shuts down gracefully).
        self.ai_turn_task_manager.clear()
