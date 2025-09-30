#!/usr/bin/env python
"""GUI LLM client with auto-persisted branching chat history and RAG (retrieval-augmented generation; query your plain-text documents)."""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .. import __version__

logger.info(f"Raven-librarian version {__version__} starting.")

logger.info("Loading libraries...")
from unpythonic import timer
with timer() as tim:
    import atexit
    import collections
    import concurrent.futures
    import io
    import json
    import os
    import pathlib
    import platform
    import requests
    import sys
    import threading
    import time
    import traceback
    from typing import Any, Callable, Dict, List, Optional, Union
    import uuid

    # WORKAROUND: Deleting a texture or image widget causes DPG to segfault on Nvidia/Linux.
    # https://github.com/hoffstadt/DearPyGui/issues/554
    if platform.system().upper() == "LINUX":
        os.environ["__GLVND_DISALLOW_PATCHING"] = "1"

    import dearpygui.dearpygui as dpg

    from mcpyrate import colorizer

    from unpythonic.env import env

    # Vendored libraries
    from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa  # https://github.com/juliettef/IconFontCppHeaders
    from ..vendor import DearPyGui_Markdown as dpg_markdown  # https://github.com/IvanNazaruk/DearPyGui-Markdown
    # from ..vendor.file_dialog.fdialog import FileDialog  # https://github.com/totallynotdrait/file_dialog, but with custom modifications

    from ..client import api  # Raven-server support
    from ..client import avatar_controller
    from ..client.avatar_renderer import DPGAvatarRenderer
    from ..client import config as client_config

    from ..common import bgtask

    from ..common.gui import animation as gui_animation
    from ..common.gui import utils as guiutils

    from . import appstate
    from . import chatutil
    from . import config as librarian_config
    # from . import chattree
    from . import hybridir
    from . import llmclient
    from . import scaffold

    gui_config = librarian_config.gui_config  # shorthand, this is used a lot
logger.info(f"Libraries loaded in {tim.dt:0.6g}s.")

# ----------------------------------------
# Module bootup

bg = concurrent.futures.ThreadPoolExecutor()
task_manager = bgtask.TaskManager(name="librarian",  # for most tasks
                                  mode="concurrent",
                                  executor=bg)
ai_turn_task_manager = bgtask.TaskManager(name="librarian_ai_turn",  # for running the AI's turn, specifically (so that we can easily cancel just that one task when needed)
                                          mode="concurrent",
                                          executor=bg)  # same thread poool
api.initialize(raven_server_url=client_config.raven_server_url,
               raven_api_key_file=client_config.raven_api_key_file,
               tts_server_type=client_config.tts_server_type,
               tts_url=client_config.tts_url,
               tts_api_key_file=client_config.tts_api_key_file,
               tts_playback_audio_device=client_config.tts_playback_audio_device,
               executor=bg)  # reuse our executor so the TTS audio player goes in the same thread pool  # TODO: there's currently a bug, because `llmclient` inits API first, with a default executor.

llm_backend_url = librarian_config.llm_backend_url

# These are initialized later, when the app starts
avatar_instance_id = None

current_chat_history = []
current_chat_history_lock = threading.RLock()

# --------------------------------------------------------------------------------
# Set up DPG - basic startup, load fonts, set up global theme

# We do this as early as possible, because before the startup is complete, trying to `dpg.add_xxx` or `with dpg.xxx:` anything will segfault the app.

logger.info("DPG bootup...")
with timer() as tim:
    dpg.create_context()

    themes_and_fonts = guiutils.bootup(font_size=gui_config.font_size)
    subtitle_font_key, subtitle_font = guiutils.load_extra_font(themes_and_fonts=themes_and_fonts,
                                                                font_size=gui_config.subtitle_font_size,
                                                                font_basename=gui_config.subtitle_font_basename,
                                                                variant=gui_config.subtitle_font_variant)

    # animation for document database and web access indicators (cyclic, runs in the background)
    with dpg.theme(tag="my_pulsating_gray_text_theme"):
        with dpg.theme_component(dpg.mvAll):
            pulsating_gray_color = dpg.add_theme_color(dpg.mvThemeCol_Text, (180, 180, 180))
    pulsating_gray_text_glow = gui_animation.PulsatingColor(cycle_duration=2.0,
                                                            theme_color_widget=pulsating_gray_color)
    gui_animation.animator.add(pulsating_gray_text_glow)

    # Initialize textures.
    with dpg.texture_registry(tag="librarian_app_textures"):
        # Prefer per-character icon; default to generic icon if not present.
        character_image_path = librarian_config.avatar_config.image_path
        character_dir = character_image_path.parent
        basename = os.path.basename(str(character_image_path))  # e.g. "/foo/bar/example.png" -> "example.png"
        stem, ext = os.path.splitext(basename)  # -> "example", ".png"
        character_icon_path = character_dir / f"{stem}_icon{ext}"
        if not character_icon_path.exists():
            character_icon_path = pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "icons", "ai.png")).expanduser().resolve()  # generic AI icon

        w, h, c, data = dpg.load_image(str(character_icon_path))
        icon_ai_texture = dpg.add_static_texture(w, h, data, tag="icon_ai_texture")

        w, h, c, data = dpg.load_image(str(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "icons", "system.png")).expanduser().resolve()))
        icon_system_texture = dpg.add_static_texture(w, h, data, tag="icon_system_texture")

        w, h, c, data = dpg.load_image(str(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "icons", "tool.png")).expanduser().resolve()))
        icon_tool_texture = dpg.add_static_texture(w, h, data, tag="icon_tool_texture")

        w, h, c, data = dpg.load_image(str(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "icons", "user.png")).expanduser().resolve()))
        icon_user_texture = dpg.add_static_texture(w, h, data, tag="icon_user_texture")

    if platform.system().upper() == "WINDOWS":
        icon_ext = "ico"
    else:
        icon_ext = "png"

    dpg.create_viewport(title=f"Raven-librarian {__version__}",
                        small_icon=str(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "icons", f"app_128_notext.{icon_ext}")).expanduser().resolve()),
                        large_icon=str(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "icons", f"app_256.{icon_ext}")).expanduser().resolve()),
                        width=gui_config.main_window_w,
                        height=gui_config.main_window_h)  # OS window (DPG "viewport")
    dpg.setup_dearpygui()
logger.info(f"    Done in {tim.dt:0.6g}s.")
print()

# --------------------------------------------------------------------------------
# Connect to servers, load datastores

if api.raven_server_available():
    print(f"{colorizer.Fore.GREEN}{colorizer.Style.BRIGHT}Connected to Raven-server at {client_config.raven_server_url}.{colorizer.Style.RESET_ALL}")
    print()
else:
    print(f"{colorizer.Fore.RED}{colorizer.Style.BRIGHT}ERROR: Cannot connect to Raven-server at {client_config.raven_server_url}.{colorizer.Style.RESET_ALL} Is Raven-server running?")
    logger.error(f"Failed to connect to Raven-server at '{client_config.raven_server_url}'.")
    sys.exit(255)

try:
    llmclient.list_models(llm_backend_url)  # just do something, to try to connect
except requests.exceptions.ConnectionError as exc:
    print(colorizer.colorize(f"Cannot connect to LLM backend at {llm_backend_url}.", colorizer.Style.BRIGHT, colorizer.Fore.RED) + " Is the LLM server running?")
    msg = f"Failed to connect to LLM backend at {llm_backend_url}, reason {type(exc)}: {exc}"
    logger.error(msg)
    sys.exit(255)
else:
    print(colorizer.colorize(f"Connected to LLM backend at {llm_backend_url}", colorizer.Style.BRIGHT, colorizer.Fore.GREEN))
    llm_settings = llmclient.setup(backend_url=llm_backend_url)
    print()

# API key already loaded during module bootup; here, we just inform the user.
if "Authorization" in llmclient.headers:
    print(f"{colorizer.Fore.GREEN}{colorizer.Style.BRIGHT}Loaded LLM API key from '{str(librarian_config.llm_api_key_file)}'.{colorizer.Style.RESET_ALL}")
    print()
else:
    print(f"{colorizer.Fore.YELLOW}{colorizer.Style.BRIGHT}No LLM API key configured.{colorizer.Style.RESET_ALL} If your LLM needs an API key to connect, put it into '{str(librarian_config.llm_api_key_file)}'.")
    print("This can be any plain-text data your LLM's API accepts in the 'Authorization' field of the HTTP headers.")
    print("For username/password, the format is 'user pass'. Do NOT use a plaintext password over an unencrypted http:// connection!")
    print()

logger.info("Loading chat datastore.")
with timer() as tim:
    datastore_file = librarian_config.llmclient_userdata_dir / "data.json"  # chat node datastore
    state_file = librarian_config.llmclient_userdata_dir / "state.json"     # important node IDs for the chat client state

    # Persistent, branching chat history, and app settings (these will auto-persist at app exit).
    datastore, app_state = appstate.load(llm_settings, datastore_file, state_file)
logger.info(f"Datastore loaded in {tim.dt:0.6g}s.")

logger.info("Loading RAG (retrieval-augmented generation) document store.")
with timer() as tim:
    docs_dir = pathlib.Path(librarian_config.llm_docs_dir).expanduser().resolve()  # RAG documents (put your documents in this directory)
    db_dir = pathlib.Path(librarian_config.llm_database_dir).expanduser().resolve()  # RAG search indices datastore

    # Load RAG database (it will auto-persist at app exit).
    retriever, _unused_scanner = hybridir.setup(docs_dir=docs_dir,
                                                recursive=librarian_config.llm_docs_dir_recursive,
                                                db_dir=db_dir,
                                                embedding_model_name=librarian_config.qa_embedding_model,
                                                local_model_loader_fallback=False)  # Librarian requires Raven-server for other reasons, too

    logger.info(f"RAG document store is at '{str(librarian_config.llm_docs_dir)}' (put your plain-text documents here).")
    # The retriever's `documents` attribute must be locked before accessing.
    with retriever.datastore_lock:
        plural_s = "s" if len(retriever.documents) != 1 else ""
        logger.info(f"RAG: {len(retriever.documents)} document{plural_s} loaded.")
    logger.info(f"RAG: Search indices are saved in '{str(librarian_config.llm_database_dir)}'.")
logger.info(f"RAG document store loaded in {tim.dt:0.6g}s.")


# --------------------------------------------------------------------------------
# Linear chat view (of current branch)

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

def _scroll_chat_view_to_end(max_wait_frames: int = 50) -> None:
    """Scroll chat view to end.

    `max_wait_frames`: If `max_wait_frames > 0`, wait at most for that may frames
                       for the chat panel to get a nonzero `max_y_scroll`.

                       Some waiting is usually needed at least at app startup
                       before the GUI settles.

    NOTE: When called from the main thread, `max_wait_frames` must be 0, as any
          attempt to wait would hang the main thread's explicit render loop.

          This also has the effect of not printing the current frame number,
          because `dpg.get_frame_count()` would need the render thread mutex:
              https://github.com/hoffstadt/DearPyGui/issues/2366

          When called from any other thread (also event handlers), waiting is fine.
    """
    max_y_scroll = dpg.get_y_scroll_max("chat_panel")
    for elapsed_frames in range(max_wait_frames):
        if max_y_scroll > 0:
            break
        dpg.split_frame()
        max_y_scroll = dpg.get_y_scroll_max("chat_panel")
    plural_s = "s" if elapsed_frames != 1 else ""
    waited_str = f" (after waiting for {elapsed_frames} frame{plural_s})" if elapsed_frames > 0 else " (no waiting was needed)"
    frames_str = f" frame {dpg.get_frame_count()}" if max_wait_frames > 0 else ""
    logger.info(f"_scroll_chat_view_to_end:{frames_str}{waited_str}: max_y_scroll = {max_y_scroll}")
    dpg.set_y_scroll("chat_panel", max_y_scroll)

def get_next_or_prev_sibling(node_id: str, direction: str = "next") -> Optional[str]:
    siblings, node_index = datastore.get_siblings(node_id)
    if siblings is None:
        return None
    if direction == "next":
        if node_index < len(siblings) - 1:
            return siblings[node_index + 1]
    else:  # direction == "prev":
        if node_index > 0:
            return siblings[node_index - 1]
    return None  # no sibling found

def format_chat_message_for_clipboard(message_number: Optional[int],
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

def get_node_message_text_without_persona(node_id: str) -> str:
    """Format a chat message from `node_id` in the datastore, by stripping the persona name from the front.

    This is useful e.g. for displaying the message text in the linearized chat view,
    or for sending the message into TTS preprocessing (`avatar_add_text_to_preprocess_queue`).

    Returns the tuple `(role, persona, text)`, where:

        `role`: One of the roles supported by `raven.librarian.llmclient`.
                Typically, one of "assistant", "system", "tool", or "user".

        `persona`: The persona name of `role`, as it was stored in the chat node.
                   If the role has no persona name, then this is `None`.

        `text`: The text content of the chat message with the persona name stripped,
                at the node's current payload revision.
    """
    node_payload = datastore.get_payload(node_id)  # auto-selects active revision  TODO: later (chat editing), we need to set the revision to load
    message = node_payload["message"]
    role = message["role"]
    persona = node_payload["general_metadata"]["persona"]  # stored persona for this chat message
    text = message["content"]
    text = chatutil.remove_persona_from_start_of_line(persona=persona,
                                                      text=text)
    return role, persona, text

class DisplayedChatMessage:
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
                 gui_parent: Union[int, str]):
        """Base class for a chat message displayed in the linearized chat view.

        `gui_parent`: DPG tag or ID of the GUI widget (typically child window or group) to add the chat message to.
        """
        super().__init__()
        self.gui_parent = gui_parent  # GUI container to render in (DPG ID or tag)
        self.gui_uuid = str(uuid.uuid4())  # used in GUI widget tags
        self.gui_container_group = dpg.add_group(tag=f"chat_item_container_group_{self.gui_uuid}",
                                                 parent=self.gui_parent)
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

    def build(self,
              role: str,
              persona: Optional[str],
              node_id: Optional[str]) -> None:
        """Build the GUI widgets for this instance, thus rendering the chat message (and buttons and such) in the GUI.

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

        NOTE: `DisplayedCompleteChatMessage` parses the content from the chat node add adds the text automatically.
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
            node_payload = datastore.get_payload(node_id)  # auto-selects active revision  TODO: later (chat editing), we need to set the revision to load
            payload_datetime = node_payload["general_metadata"]["datetime"]  # of the active revision!
            node_active_revision = datastore.get_revision(node_id)
            dpg.add_text(f"{payload_datetime} R{node_active_revision}", color=(120, 120, 120), parent=text_vertical_layout_group)

        # render the actual text
        self.gui_text_group = dpg.add_group(tag=f"chat_message_text_container_group_{self.gui_uuid}",
                                            parent=text_vertical_layout_group)  # create another group to act as container so that we can update/replace just the text easily
        # NOTE: We now have an empty group, for `add_paragraph`/`replace_last_paragraph`.

        # Show LLM performance statistics if linked to a chat node, and the chat node has them
        if role == "assistant" and node_id is not None:
            ai_message_node_payload = datastore.get_payload(node_id)
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

        # # Render background color after the content has been rendered once (so that we can get its size)
        # if message_role in role_colors:
        #     def add_colored_border():
        #         logger.info(f"{self.gui_uuid}: running callback ({message_role})")
        #
        #         # w, h = dpg.get_item_rect_size(self.gui_container_group)  # TODO: might not actually work
        #         w, h = dpg.get_item_rect_size(chat_message_widget)  # TODO: might not actually work
        #
        #         drawlist_tag = f"chat_colored_border_drawlist_{self.gui_uuid}"
        #         dpg.delete_item(drawlist_tag, children_only=True)
        #         dpg.configure_item(drawlist_tag, width=4, height=h)
        #         dpg.draw_rectangle((0, 0), (4, h),
        #                            color=role_colors[message_role]["front"],
        #                            fill=role_colors[message_role]["front"],
        #                            parent=drawlist_tag)
        #
        #         # xbase, ybase = guiutils.get_widget_pos("chat_panel")
        #         # logger.info(f"{self.gui_uuid}: xbase = {xbase}, ybase = {ybase}")
        #         # x0_local, y0_local = guiutils.get_widget_relative_pos(self.gui_container_group, reference="chat_panel")  # tag
        #         # w, h = dpg.get_item_rect_size(self.gui_container_group)  # TODO: might not actually work
        #         # logger.info(f"{self.gui_uuid}: x0 = {x0_local}, y0 = {y0_local}, w = {w}, h = {h}")
        #         # drawlist_bg_box = dpg.add_drawlist(width=w, height=h,
        #         #                                    pos=(x0_local, y0_local),
        #         #                                    tag=f"background_box_drawlist_{self.gui_uuid}",
        #         #                                    parent=self.gui_container_group,
        #         #                                    before=icon_and_text_container_group)
        #         # dpg.set_item_pos(self.gui_container_group, (x0_local, y0_local))
        #         # dpg.draw_rectangle((0, 0),
        #         #                    (w, h),
        #         #                    color=role_colors[message_role]["front"],
        #         #                    fill=role_colors[message_role]["back"],
        #         #                    rounding=8,
        #         #                    parent=drawlist_bg_box)
        #     # DPG can only assign one frame callback per frame, so we queue them and set a master callback to run through the queue.
        #     logger.info(f"{self.gui_uuid}: setting callback")
        #     with type(self).class_lock:
        #         type(self).callbacks[self.gui_uuid] = add_colored_border
        #     dpg.set_frame_callback(dpg.get_frame_count() + 10,
        #                            type(self).run_callbacks)

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

        If you use `build_linearized_chat_panel`, it takes care of clearing all chat message GUI widgets automatically,
        and you do not need to call this.

        If you are editing the linearized chat view directly, this should be called before deleting
        the `DisplayedChatMessage` instance.

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
            except SystemError:  # the group went bye-bye already (app shutdown)
                pass

    def build_buttons(self,
                      gui_parent: Union[int, str]) -> None:
        """Build the set of control buttons for a single chat message in the GUI.

        `gui_parent`: DPG tag or ID of the GUI widget (typically a group) to add the buttons to.

                      This is not simply `self.gui_parent` due to other layout performed by `build`.
        """
        role = self.role
        persona = self.persona
        text = self.text
        node_id = self.node_id

        g = dpg.add_group(horizontal=True, tag=f"{role}_message_buttons_group_{self.gui_uuid}", parent=gui_parent)

        # dpg.add_text("[0 t, 0 s, ∞ t/s]", color=(180, 180, 180), tag=f"performance_stats_text_ai_{self.gui_uuid}", parent=g)  # TODO: add the performance stats

        # dpg.add_spacer(tag=f"ai_message_buttons_spacer_{self.gui_uuid}",
        #                parent=g)

        def copy_message_to_clipboard_callback() -> None:
            shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
            # Note we only add the role name when we include also the node ID.
            # Omitting the name in regular mode improves convenience for copy-pasting an existing question into the chat field.
            formatted_message = format_chat_message_for_clipboard(message_number=None,  # a single message copied to clipboard does not need a sequential number
                                                                  role=role,
                                                                  persona=persona,
                                                                  text=text,
                                                                  add_heading=shift_pressed)

            if shift_pressed:
                node_payload = datastore.get_payload(node_id)  # auto-selects active revision  TODO: later (chat editing), we need to set the revision to load
                payload_datetime = node_payload["general_metadata"]["datetime"]  # of the active revision!
                node_active_revision = datastore.get_revision(node_id)
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
                                                                 original_theme=themes_and_fonts.global_theme,
                                                                 duration=gui_config.acknowledgment_duration))
        self.gui_button_callbacks["copy"] = copy_message_to_clipboard_callback
        copy_message_button = dpg.add_button(label=fa.ICON_COPY,
                                             callback=copy_message_to_clipboard_callback,
                                             width=gui_config.toolbutton_w,
                                             tag=f"message_copy_to_clipboard_button_{self.gui_uuid}",
                                             parent=g)
        dpg.bind_item_font(copy_message_button, themes_and_fonts.icon_font_solid)
        dpg.bind_item_theme(copy_message_button, "disablable_button_theme")  # tag
        copy_message_tooltip = dpg.add_tooltip(copy_message_button)
        copy_message_tooltip_text = dpg.add_text("Copy message to clipboard\n    no modifier: as-is\n    with Shift: include message node ID", parent=copy_message_tooltip)

        # Only AI messages can be rerolled
        if role == "assistant":
            def reroll_message_callback():
                global current_chat_history  # for documenting the intent only

                # Find this AI message in the chat history
                for k, displayed_message in enumerate(reversed(current_chat_history)):
                    if displayed_message.node_id == node_id:
                        break
                # `k` is now how many messages must be popped from the end to reach this one
                assert k < len(current_chat_history) - 3  # should have at least the system prompt, the AI's initial greeting, and the user's first message remaining
                # Rewind the linearized chat history in the GUI
                for _ in range(k):
                    old_displayed_message = current_chat_history.pop(-1)
                    old_displayed_message.demolish()

                # Handle the RAG query: find the latest user message (above this AI message)
                user_message_text = None
                for displayed_message in reversed(current_chat_history):  # ...what's remaining of the history, anyway
                    if displayed_message.role == "user":
                        user_message_text = displayed_message.text
                        break

                # Remove the AI message from GUI
                app_state["HEAD"] = datastore.get_parent(node_id)
                old_displayed_message = current_chat_history.pop(-1)  # once more, with feeling!
                old_displayed_message.demolish()

                # Generate new AI message
                ai_turn(docs_query=user_message_text)
            reroll_enabled = (node_id is not None and node_id != app_state["new_chat_HEAD"])  # The AI's initial greeting can't be rerolled
            if reroll_enabled:
                self.gui_button_callbacks["reroll"] = reroll_message_callback  # stash it so we can call it from the hotkey handler
            dpg.add_button(label=fa.ICON_RECYCLE,
                           callback=reroll_message_callback,
                           enabled=reroll_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_reroll_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_reroll_button_{self.gui_uuid}", themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_reroll_button_{self.gui_uuid}", "disablable_button_theme")  # tag
            reroll_tooltip = dpg.add_tooltip(f"message_reroll_button_{self.gui_uuid}")  # tag
            dpg.add_text("Reroll AI response (create new sibling) [Ctrl+R]", parent=reroll_tooltip)
        else:
            dpg.add_spacer(width=gui_config.toolbutton_w, height=1, parent=g)

        if role == "assistant":
            def speak_message_callback():
                if app_state["avatar_speech_enabled"]:
                    unused_message_role, unused_message_persona, message_text = get_node_message_text_without_persona(node_id)
                    # Send only non-thought message content to TTS
                    message_text = chatutil.scrub(persona=llm_settings.personas.get("assistant", None),
                                                  text=message_text,
                                                  thoughts_mode="discard",
                                                  markup=None,
                                                  add_persona=False)
                    avatar_controller.send_text_to_tts(message_text,
                                                       voice=librarian_config.avatar_config.voice,
                                                       voice_speed=librarian_config.avatar_config.voice_speed,
                                                       video_offset=librarian_config.avatar_config.video_offset)

                    # Acknowledge the action in the GUI.
                    gui_animation.animator.add(gui_animation.ButtonFlash(message="Sent to avatar!",
                                                                         target_button=speak_message_button,
                                                                         target_tooltip=speak_message_tooltip,
                                                                         target_text=speak_message_tooltip_text,
                                                                         original_theme=themes_and_fonts.global_theme,
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
            dpg.bind_item_font(speak_message_button, themes_and_fonts.icon_font_solid)
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
        dpg.bind_item_font(f"chat_edit_button_{self.gui_uuid}", themes_and_fonts.icon_font_solid)  # tag
        dpg.bind_item_theme(f"chat_edit_button_{self.gui_uuid}", "disablable_button_theme")  # tag
        edit_tooltip = dpg.add_tooltip(f"chat_edit_button_{self.gui_uuid}")  # tag
        dpg.add_text("Edit (revise)", parent=edit_tooltip)

        dpg.add_button(label=fa.ICON_CODE_BRANCH,
                       callback=lambda: None,  # TODO
                       enabled=False,
                       width=gui_config.toolbutton_w,
                       tag=f"message_new_branch_button_{self.gui_uuid}",
                       parent=g)
        dpg.bind_item_font(f"message_new_branch_button_{self.gui_uuid}", themes_and_fonts.icon_font_solid)  # tag
        dpg.bind_item_theme(f"message_new_branch_button_{self.gui_uuid}", "disablable_button_theme")  # tag
        new_branch_tooltip = dpg.add_tooltip(f"message_new_branch_button_{self.gui_uuid}")  # tag
        dpg.add_text("Branch from this node", parent=new_branch_tooltip)

        # NOTE: We disallow deleting the system prompt and the AI's initial greeting, as well as any message that is not linked to a chat node in the datastore.
        delete_enabled = (node_id is not None and node_id not in (app_state["system_prompt_node_id"], app_state["new_chat_HEAD"]))
        dpg.add_button(label=fa.ICON_TRASH_CAN,
                       callback=lambda: None,  # TODO
                       enabled=False,  # TODO: use `delete_enabled` once delete is implemented
                       width=gui_config.toolbutton_w,
                       tag=f"message_delete_branch_button_{self.gui_uuid}",
                       parent=g)
        dpg.bind_item_font(f"message_delete_branch_button_{self.gui_uuid}", themes_and_fonts.icon_font_solid)  # tag
        dpg.bind_item_theme(f"message_delete_branch_button_{self.gui_uuid}", "disablable_button_theme")  # tag
        delete_branch_tooltip = dpg.add_tooltip(f"message_delete_branch_button_{self.gui_uuid}")  # tag

        c_red = '<font color="(255, 96, 96)">'
        c_end = '</font>'
        dpg_markdown.add_text(f"Delete branch (this node and {c_red}**all**{c_end} descendants)", parent=delete_branch_tooltip)

        def make_navigate_to_prev_sibling(message_node_id: str) -> Callable:
            def navigate_to_prev_sibling_callback():
                node_id = get_next_or_prev_sibling(message_node_id, direction="prev")
                if node_id is not None:
                    app_state["HEAD"] = node_id
                    build_linearized_chat_panel()
            return navigate_to_prev_sibling_callback

        def make_navigate_to_next_sibling(message_node_id: str) -> Callable:
            def navigate_to_next_sibling_callback():
                node_id = get_next_or_prev_sibling(message_node_id, direction="next")
                if node_id is not None:
                    app_state["HEAD"] = node_id
                    build_linearized_chat_panel()
            return navigate_to_next_sibling_callback

        # Only messages attached to a datastore chat node can have siblings in the datastore
        if node_id is not None:
            siblings, node_index = datastore.get_siblings(node_id)
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
            dpg.bind_item_font(f"message_prev_branch_button_{self.gui_uuid}", themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_prev_branch_button_{self.gui_uuid}", "disablable_button_theme")  # tag
            prev_branch_tooltip = dpg.add_tooltip(f"message_prev_branch_button_{self.gui_uuid}")  # tag
            dpg.add_text("Switch to previous sibling [Ctrl+Left]", parent=prev_branch_tooltip)

            dpg.add_button(label=fa.ICON_ANGLE_RIGHT,
                           callback=navigate_to_next_sibling_callback,
                           enabled=next_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_next_branch_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_next_branch_button_{self.gui_uuid}", themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_next_branch_button_{self.gui_uuid}", "disablable_button_theme")  # tag
            next_branch_tooltip = dpg.add_tooltip(f"message_next_branch_button_{self.gui_uuid}")  # tag
            dpg.add_text("Switch to next sibling [Ctrl+Right]", parent=next_branch_tooltip)

            if siblings is not None:
                dpg.add_text(f"{node_index + 1} / {len(siblings)}", parent=g)
        else:
            # Add the two spacers separately so we get the same margins as with two separate buttons
            dpg.add_spacer(width=gui_config.toolbutton_w, height=1, parent=g)
            dpg.add_spacer(width=gui_config.toolbutton_w, height=1, parent=g)


class DisplayedCompleteChatMessage(DisplayedChatMessage):
    def __init__(self,
                 node_id: str,
                 gui_parent: Union[int, str]):
        """A complete chat message displayed in the linearized chat view, linked to a node ID in the datastore.

        `node_id`: The ID of the chat node, in the datastore, from which to extract the data to show.
        `gui_parent`: DPG tag or ID of the GUI widget (typically child window or group) to add the chat message to.
        """
        super().__init__(gui_parent=gui_parent)
        self.node_id = node_id  # reference to the chat node (to ORIGINAL node data, not a copy)
        self.build()

    def build(self) -> None:
        """Build (or rebuild) the GUI widgets for this chat message.

        Automatically parse the content from the chat node, and add the text to the GUI.
        """
        role, persona, text = get_node_message_text_without_persona(self.node_id)  # TODO: later (chat editing), we need to set the revision to load
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


class DisplayedStreamingChatMessage(DisplayedChatMessage):
    def __init__(self,
                 gui_parent: Union[int, str]):
        """A chat message being streamed live from the LLM, displayed in the linearized chat view."""
        super().__init__(gui_parent=gui_parent)
        self.build()

    def build(self):
        super().build(role="assistant",  # TODO: parameterize this?
                      persona=llm_settings.personas.get("assistant", None),
                      node_id=None)


def add_complete_chat_message_to_linearized_chat_panel(node_id: str,
                                                       scroll_to_end: bool = True) -> DisplayedCompleteChatMessage:
    """Append the given chat node to the end of the linearized chat view in the GUI."""
    global current_chat_history  # intent only; we write, but we don't replace the list itself.
    with current_chat_history_lock:
        displayed_chat_message = DisplayedCompleteChatMessage(gui_parent="chat_group",
                                                              node_id=node_id)
        current_chat_history.append(displayed_chat_message)
    if scroll_to_end:
        dpg.split_frame()
        _scroll_chat_view_to_end()


def build_linearized_chat_panel(head_node_id: Optional[str] = None) -> None:
    """Build the linearized chat view in the GUI, linearizing up from `head_node_id`.

    As side effects:

      - Update the global `current_chat_history`.
      - If `head_node_id` is an AI message, update the avatar's emotion from that
        (using the node's current payload revision).
    """
    global current_chat_history  # intent only; we write, but we don't replace the list itself.
    if head_node_id is None:  # use current HEAD from app_state?
        head_node_id = app_state["HEAD"]
    node_id_history = datastore.linearize_up(head_node_id)
    with current_chat_history_lock:
        current_chat_history.clear()
        dpg.delete_item("chat_group", children_only=True)  # clear old content from GUI
        for node_id in node_id_history:
            add_complete_chat_message_to_linearized_chat_panel(node_id=node_id,
                                                               scroll_to_end=False)  # we scroll just once, when done
    # Update avatar emotion from the message text (use only non-thought message content)
    role, unused_persona, text = get_node_message_text_without_persona(head_node_id)
    if role == "assistant":
        logger.info("build_linearized_chat_panel: linearized chat view new HEAD node is an AI message; updating avatar emotion from (non-thought) message content")
        text = chatutil.scrub(persona=llm_settings.personas.get("assistant", None),
                              text=text,
                              thoughts_mode="discard",
                              markup=None,
                              add_persona=False)
        avatar_controller.update_emotion_from_text(text)
    dpg.split_frame()
    _scroll_chat_view_to_end()


# --------------------------------------------------------------------------------
# Scaffold to GUI integration

def chat_round(user_message_text: str) -> None:  # message text comes from GUI
    """Run a chat round (user and AI).

    This spawns a background task to avoid hanging GUI event handlers,
    since the typical use case is to call `chat_round` from a GUI event handler.

    By sending empty `user_message_text`, it is possible to have the AI generate
    another message without the user writing in between.

    The RAG query is taken from the latest available user message.
    """
    def run_chat_round(task_env: env) -> None:
        if task_env.cancelled:  # while the task was in the queue
            return

        # Only add the user's message to the chat if the user entered any text.
        if user_message_text:
            user_turn(text=user_message_text)
            # NOTE: Rudimentary approach to RAG search, using the user's message text as the query. (Good enough to demonstrate the functionality. Improve later.)
            docs_query = user_message_text
        else:
            # Handle the RAG query: find the latest existing user message
            docs_query = None  # if no user message, send `None` as query to AI -> no docs search
            for displayed_message in reversed(current_chat_history):
                if displayed_message.role == "user":
                    docs_query = displayed_message.text
                    break
        if task_env.cancelled:  # during user turn
            return
        ai_turn(docs_query=docs_query)
    task_manager.submit(run_chat_round, env())

def user_turn(text: str) -> None:
    """Add the user's message to the chat, and append it to the linearized chat view in the GUI."""
    def run_user_turn(task_env: env) -> None:
        if task_env.cancelled:  # while the task was in the queue
            return

        new_head_node_id = scaffold.user_turn(llm_settings=llm_settings,
                                              datastore=datastore,
                                              head_node_id=app_state["HEAD"],
                                              user_message_text=text)
        app_state["HEAD"] = new_head_node_id  # as soon as possible, so that not affected by any errors during GUI building
        add_complete_chat_message_to_linearized_chat_panel(new_head_node_id)
    task_manager.submit(run_user_turn, env())

def ai_turn(docs_query: Optional[str]) -> None:  # TODO: implement continue mode
    """Run the AI's response part of a chat round.

    This spawns a background task to avoid hanging GUI event handlers,
    since the reroll GUI event handler calls `ai_turn` directly.
    """
    docs_query = docs_query if app_state["docs_enabled"] else None

    def run_ai_turn(task_env: env) -> None:
        global gui_alive  # intent only

        if task_env.cancelled:  # while the task was in the queue
            return

        if gui_alive:
            dpg.enable_item("chat_stop_generation_button")  # tag

        speech_enabled = app_state["avatar_speech_enabled"]  # grab once, in case the user toggles it while this AI turn is being processed

        try:
            streaming_chat_message = None
            def delete_streaming_chat_message():  # for replacing with completed message
                nonlocal streaming_chat_message
                if streaming_chat_message is not None:
                    streaming_chat_message.demolish()
                    streaming_chat_message = None

            def on_docs_start() -> None:
                global gui_alive  # intent only
                if gui_alive:
                    avatar_controller.start_data_eyes()
                    pulsating_gray_text_glow.reset()  # start new pulsation cycle
                    dpg.show_item(docs_indicator_group)

            def on_docs_done(matches: List[Dict]) -> None:
                global gui_alive  # intent only
                if gui_alive:
                    dpg.hide_item(docs_indicator_group)
                    avatar_controller.stop_data_eyes()

            def on_llm_start() -> None:
                global gui_alive  # intent only
                if gui_alive:
                    nonlocal streaming_chat_message
                    streaming_chat_message = DisplayedStreamingChatMessage(gui_parent="chat_group")
                    dpg.split_frame()
                    _scroll_chat_view_to_end()

                    pulsating_gray_text_glow.reset()  # start new pulsation cycle
                    dpg.show_item(llm_indicator_group)  # show prompt processing indicator

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
                    logger.info(f"ai_turn.run_ai_turn._update_avatar_emotion_from_incoming_text: updating emotion from {len(text)} characters of recent text")
                    avatar_controller.update_emotion_from_text(text)
                task_env.emotion_update_calls += 1

            def on_llm_progress(n_chunks: int, chunk_text: str) -> None:
                global gui_alive  # intent only

                if gui_alive and chunk_text:  # avoid triggering on the initial empty chunk (ACK)
                    dpg.hide_item(llm_indicator_group)  # hide prompt processing indicator

                # If the task is cancelled, interrupt the LLM, keeping the content received so far (the scaffold will automatically send the content to `on_llm_done`).
                # TODO: arrange for the GUI to actually cancel the task upon the user pressing an interrupt button
                if task_env.cancelled or not gui_alive:  # TODO: EAFP to avoid TOCTTOU
                    reason = "Cancelled" if task_env.cancelled else "App is shutting down"
                    logger.info(f"ai_turn.run_ai_turn.on_llm_progress: {reason}, stopping text generation.")
                    return llmclient.action_stop

                # Detect think block state (TODO: improve; very rudimentary and brittle for now)
                if "<think>" in chunk_text:
                    task_env.inside_think_block = True
                    logger.info("ai_turn.run_ai_turn.on_llm_progress: AI entered thinking state.")
                elif "</think>" in chunk_text:
                    logger.info("ai_turn.run_ai_turn.on_llm_progress: AI exited thinking state.")
                    task_env.inside_think_block = False

                    if not speech_enabled:  # If TTS is NOT enabled, show the generic talking animation while the LLM is writing
                        api.avatar_start_talking(avatar_instance_id)

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
                    #         avatar_add_text_to_preprocess_queue(paragraph_text)
                    streaming_chat_message.replace_last_paragraph(paragraph_text,
                                                                  is_thought=(task_env.inside_think_block or ("</think>" in chunk_text)))  # easiest to special-case the closing tag
                    streaming_chat_message.add_paragraph("",
                                                         is_thought=task_env.inside_think_block)
                    task_env.text = io.StringIO()
                    dpg.split_frame()
                    _scroll_chat_view_to_end()
                # - update at least every 0.5 sec
                # - update after every 10 chunks, but rate-limited (at least 0.1 sec must have passed since last update)
                elif dt >= 0.5 or (dt >= 0.25 and dchunks >= 10):  # commit changes to in-progress last paragraph
                    task_env.t0 = time_now
                    task_env.n_chunks0 = n_chunks
                    streaming_chat_message.replace_last_paragraph(task_env.text.getvalue(),
                                                                  is_thought=task_env.inside_think_block)  # at first paragraph, will auto-create it if not created yet
                    dpg.split_frame()
                    _scroll_chat_view_to_end()

                # Let the LLM keep generating (if it wants to).
                return llmclient.action_ack

            def on_done(node_id: str) -> None:   # For both `on_llm_done` and `on_nomatch_done`.
                global gui_alive  # intent only

                app_state["HEAD"] = node_id  # update just in case of Ctrl+C or crash during tool calls
                task_env.text = io.StringIO()  # for next AI message (in case of tool calls)
                if gui_alive:
                    if not speech_enabled:  # If TTS is NOT enabled, stop the generic talking animation now that the LLM is done
                        api.avatar_stop_talking(avatar_instance_id)

                    unused_role, unused_persona, text = get_node_message_text_without_persona(node_id)

                    # Keep only non-thought content for TTS and emotion update
                    text = chatutil.scrub(persona=llm_settings.personas.get("assistant", None),
                                          text=text,
                                          thoughts_mode="discard",
                                          markup=None,
                                          add_persona=False)

                    # Avatar speech and subtitling
                    logger.info("ai_turn.run_ai_turn.on_done: sending final (non-thought) message content for translation, TTS, and subtitling")
                    if speech_enabled:  # If TTS enabled, send final message text to TTS preprocess queue (this always uses lipsync)
                        avatar_controller.send_text_to_tts(text,
                                                           voice=librarian_config.avatar_config.voice,
                                                           voice_speed=librarian_config.avatar_config.voice_speed,
                                                           video_offset=librarian_config.avatar_config.video_offset)

                    # Update avatar emotion one last time, from the final message text
                    logger.info("ai_turn.run_ai_turn.on_done: updating emotion from final (non-thought) message content")
                    avatar_controller.update_emotion_from_text(text)

                    # Update linearized chat view
                    logger.info("ai_turn.run_ai_turn.on_done: updating chat view with final message")
                    delete_streaming_chat_message()  # if we are called by docs nomatch, the in-progress message shouldn't exist in the GUI; then this doesn't matter.
                    add_complete_chat_message_to_linearized_chat_panel(node_id)

                    logger.info("ai_turn.run_ai_turn.on_done: all done.")

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
                global gui_alive  # intent only
                if gui_alive:
                    avatar_controller.start_data_eyes()

                    # # HACK: If websearch is present *anywhere* among the tool calls in this message,
                    # #       light up the web access indicator for the whole tool call processing step.
                    # #       Often there is just one tool call, so it's fine.
                    # ids_and_names = [_parse_toolcall(request_record) for request_record in tool_calls]
                    # names = [name for _id, name in ids_and_names]
                    # if "websearch" in names:
                    #     pulsating_gray_text_glow.reset()  # start new pulsation cycle
                    #     dpg.show_item(web_indicator_group)

            def on_call_lowlevel_start(toolcall_id: str, function_name: str, arguments: Dict[str, Any]) -> None:
                global gui_alive  # intent only
                if gui_alive:
                    if function_name == "websearch":
                        pulsating_gray_text_glow.reset()  # start new pulsation cycle
                        dpg.show_item(web_indicator_group)

            def on_call_lowlevel_done(toolcall_id: str, function_name: str, status: str, text: str) -> None:
                global gui_alive  # intent only
                if gui_alive:
                    if function_name == "websearch":
                        dpg.hide_item(web_indicator_group)

            def on_tool_done(node_id: str) -> None:
                global gui_alive  # intent only

                app_state["HEAD"] = node_id  # update just in case of Ctrl+C or crash during tool calls
                task_env.text = io.StringIO()  # for next AI message (in case of tool calls)
                if gui_alive:
                    delete_streaming_chat_message()  # it shouldn't exist when this triggers, but robustness.
                    add_complete_chat_message_to_linearized_chat_panel(node_id)

            def on_tools_done() -> None:
                global gui_alive  # intent only
                if gui_alive:
                    # dpg.hide_item(web_indicator_group)
                    avatar_controller.stop_data_eyes()

            new_head_node_id = scaffold.ai_turn(llm_settings=llm_settings,
                                                datastore=datastore,
                                                retriever=retriever,
                                                head_node_id=app_state["HEAD"],
                                                tools_enabled=app_state["tools_enabled"],
                                                docs_query=docs_query,
                                                docs_num_results=librarian_config.docs_num_results,
                                                speculate=app_state["speculate_enabled"],
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
            app_state["HEAD"] = new_head_node_id
        finally:
            if gui_alive:
                dpg.disable_item("chat_stop_generation_button")  # tag
                avatar_controller.stop_data_eyes()  # make sure the data eyes effect ends (unless app shutting down, in which case we shouldn't start new GUI animations)
                if not speech_enabled:  # make sure the generic talking animation ends (if we invoked it)
                    api.avatar_stop_talking(avatar_instance_id)
                # Also make sure that the processing indicators hide
                dpg.hide_item(docs_indicator_group)
                dpg.hide_item(web_indicator_group)
                dpg.hide_item(llm_indicator_group)
    ai_turn_task_manager.submit(run_ai_turn, env())

def stop_ai_turn() -> None:
    """Interrupt the AI, i.e. stop ongoing text generation.

    Useful to have in case you see the AI has misunderstood your question,
    so that there's no need to wait for a complete response.
    """
    if gui_alive:
        dpg.disable_item("chat_stop_generation_button")  # tag
    # Cancelling all background tasks from the AI turn specific task manager stops the task (co-operatively, so it shuts down gracefully).
    ai_turn_task_manager.clear()

# --------------------------------------------------------------------------------
# Set up the main window

logger.info("Initial GUI setup...")
with timer() as tim:
    with dpg.window(show=True, modal=False, no_title_bar=False, tag="summarizer_window",
                    label="Raven-librarian main window",
                    no_scrollbar=True, autosize=True) as main_window:  # DPG "window" inside the app OS window ("viewport"), container for the whole GUI
        with dpg.group(horizontal=True):
            with dpg.group():
                chat_panel_h = gui_config.main_window_h - (gui_config.ai_warning_h + 16) - (gui_config.chat_controls_h + 16) + 8
                with dpg.child_window(tag="chat_panel",
                                      width=(gui_config.chat_panel_w + 16),  # 16 = round border (8 on each side)
                                      height=chat_panel_h):
                    # dummy chat item for testing  # TODO: make a class for this
                    with dpg.group(tag="chat_group"):
                        initial_message_container_height = 2 * gui_config.margin + gui_config.chat_icon_size
                        before_buttons_spacing = 1
                        message_spacing = 8

                    #     # We need to draw text using a text widget, not `draw_text`, so that we can use Markdown.
                    #     # But we want a visual frame, which needs a drawlist. The chat icon can also go into this drawlist.
                    #     # To draw the text on top of the drawlist, we add the drawlist first (so it will be below the text in z-order),
                    #     # and then, while adding the text widget, manually set the position (in child-window coordinates).
                    #     with dpg.drawlist(width=800, height=initial_message_container_height, tag="chat_text_drawlist_ai"):
                    #         dpg.draw_rectangle((0, 0), (800, initial_message_container_height), color=gui_config.chat_color_ai_front, fill=gui_config.chat_color_ai_back, rounding=8)
                    #         dpg.draw_image("icon_ai_texture", (gui_config.margin, gui_config.margin), (gui_config.margin + gui_config.chat_icon_size, gui_config.margin + gui_config.chat_icon_size), uv_min=(0, 0), uv_max=(1, 1))
                    #     dpg.add_spacer(height=before_buttons_spacing)
                    #     make_ai_message_buttons(gui_parent="chat_group", uuid="mockup_ai")
                    #     with dpg.group(horizontal=True):
                    #         dpg.add_spacer(tag="branch_count_spacer_ai")
                    #         dpg.add_text("1/1", color=(180, 180, 180), tag="branch_count_text_ai")
                    #         with dpg.tooltip("branch_count_text_ai"):  # tag
                    #             dpg.add_text("Current branch, number of branches at this point")
                    #     dpg.add_spacer(height=message_spacing)
                    #
                    #     with dpg.drawlist(width=800, height=initial_message_container_height, tag="chat_text_drawlist_user"):
                    #         dpg.draw_rectangle((0, 0), (800, initial_message_container_height), color=gui_config.chat_color_user_front, fill=gui_config.chat_color_user_back, rounding=8)
                    #         dpg.draw_image("icon_user_texture", (gui_config.margin, gui_config.margin), (gui_config.margin + gui_config.chat_icon_size, gui_config.margin + gui_config.chat_icon_size), uv_min=(0, 0), uv_max=(1, 1))
                    #     dpg.add_spacer(height=before_buttons_spacing)
                    #     make_user_message_buttons(gui_parent="chat_group", uuid="mockup_user")
                    #     with dpg.group(horizontal=True):
                    #         dpg.add_spacer(tag="branch_count_spacer_user")
                    #         dpg.add_text("1/1", color=(180, 180, 180), tag="branch_count_text_user")
                    #         with dpg.tooltip("branch_count_text_user"):  # tag
                    #             dpg.add_text("Current branch, number of branches at this point")
                    #     dpg.add_spacer(height=message_spacing)
                    #
                    # # We must wait for the drawlists to get a position before we can overlay a text widget on them.
                    # def add_chat_texts():
                    #     # Align branch counts to the right
                    #     w_header, h_header = dpg.get_item_rect_size("branch_count_text_ai")
                    #     dpg.set_item_width("branch_count_spacer_ai", 800 - (w_header + 8))
                    #
                    #     w_header, h_header = dpg.get_item_rect_size("branch_count_text_user")
                    #     dpg.set_item_width("branch_count_spacer_user", 800 - (w_header + 8))
                    #
                    #     w_header, h_header = dpg.get_item_rect_size("performance_stats_text_ai")
                    #     dpg.set_item_width("ai_message_buttons_spacer", 800 - 5 * (gui_config.toolbutton_w + 8) - (w_header + 8))
                    #
                    #     # Write the "chat messages" for the mockup
                    #     x0_local, y0_local = guiutils.get_widget_relative_pos("chat_text_drawlist_ai", reference="chat_panel")  # tag
                    #     dpg.add_text("Hello! I'll be your AI summarizer. To begin, select item(s) and click Summarize.",
                    #                  pos=(x0_local + 8 + 3 + gui_config.margin + gui_config.chat_icon_size, y0_local + 3 + gui_config.chat_icon_size // 2 - (gui_config.font_size // 2)),  # 8 = extra spacing; 3 = DPG inner margin
                    #                  color=(255, 255, 255), tag="chat_test_text_ai", parent="chat_group")
                    #
                    #     x0_local, y0_local = guiutils.get_widget_relative_pos("chat_text_drawlist_user", reference="chat_panel")  # tag
                    #     dpg.add_text("That's great. Testing 1 2 3?",
                    #                  pos=(x0_local + 8 + 3 + gui_config.margin + gui_config.chat_icon_size, y0_local + 3 + gui_config.chat_icon_size // 2 - (gui_config.font_size // 2)),  # 8 = extra spacing; 3 = DPG inner margin
                    #                  color=(255, 255, 255), tag="chat_test_text_user", parent="chat_group")
                    # dpg.set_frame_callback(11, add_chat_texts)

                    # def place():
                    #     x0, y0 = guiutils.get_widget_pos("chat_text_drawlist")  # tag
                    #     print(x0, y0)
                    #     # dpg.set_item_pos("chat_test_text", x0 + 16, y0 + 16)
                    # dpg.set_frame_callback(11, place)

                with dpg.child_window(tag="chat_controls",
                                      width=(gui_config.chat_panel_w + 16),  # 16 = round border (8 on each side)
                                      height=gui_config.chat_controls_h,
                                      no_scrollbar=True,
                                      no_scroll_with_mouse=True):
                    with dpg.group(horizontal=True):
                        def send_message_to_ai_callback() -> None:
                            user_message_text = dpg.get_value("chat_field")
                            dpg.set_value("chat_field", "")
                            chat_round(user_message_text)
                        dpg.add_input_text(tag="chat_field",
                                           default_value="",
                                           hint="[ask the AI questions here] [Ctrl+Space to focus]",
                                           width=gui_config.chat_panel_w - gui_config.toolbutton_w - 8)
                        dpg.add_button(label=fa.ICON_PAPER_PLANE,
                                       callback=send_message_to_ai_callback,
                                       width=gui_config.toolbutton_w,
                                       tag="chat_send_button")
                        dpg.bind_item_font("chat_send_button", themes_and_fonts.icon_font_solid)  # tag  # TODO: make this change into a cancel button while the LLM is writing.
                        dpg.bind_item_theme("chat_send_button", "disablable_button_theme")  # tag
                        with dpg.tooltip("chat_send_button"):  # tag
                            dpg.add_text("Send to AI [Enter]")

            with dpg.group():
                with dpg.child_window(tag="avatar_panel",
                                      width=-1,
                                      height=chat_panel_h,
                                      no_scrollbar=True,
                                      no_scroll_with_mouse=True):
                    # We all love magic numbers!
                    #
                    # The size of the avatar panel is not available at startup, until the GUI is rendered at least once,
                    # so we must compute the initial size.
                    avatar_panel_w = (gui_config.main_window_w - (gui_config.chat_panel_w + 16) - 3 * 8)  # the 3 * 8 are the outer borders outside the panels (between panel and window edge, and between the panels)
                    avatar_panel_h = chat_panel_h
                    dpg_avatar_renderer = DPGAvatarRenderer(texture_registry="librarian_app_textures",
                                                            gui_parent="avatar_panel",
                                                            avatar_x_center=(avatar_panel_w // 2),
                                                            avatar_y_bottom=avatar_panel_h - 8,
                                                            paused_text="[No video]",
                                                            task_manager=task_manager)
                    # DRY, just so that `_load_initial_animator_settings` at app bootup is guaranteed to use the same values
                    dpg_avatar_renderer.configure_live_texture(new_image_size=int(librarian_config.avatar_config.animator_settings_overrides["upscale"] * librarian_config.avatar_config.source_image_size))

                    with dpg.group(pos=(16, 16), show=False, horizontal=True) as llm_indicator_group:
                        dpg.add_text(fa.ICON_MICROCHIP, tag="llm_prompt_process_symbol")
                        dpg.bind_item_font("llm_prompt_process_symbol", themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("llm_prompt_process_symbol", "my_pulsating_gray_text_theme")  # tag
                        dpg.add_text("SYSTEM", tag="llm_prompt_process_text")

                    with dpg.group(pos=(16, 16), show=False, horizontal=True) as docs_indicator_group:
                        dpg.add_text(fa.ICON_DATABASE, tag="docs_access_symbol")
                        dpg.bind_item_font("docs_access_symbol", themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("docs_access_symbol", "my_pulsating_gray_text_theme")  # tag
                        dpg.add_text("DOCS", tag="docs_access_text")

                    with dpg.group(pos=(16, 16), show=False, horizontal=True) as web_indicator_group:
                        dpg.add_text(fa.ICON_GLOBE, tag="web_access_symbol")
                        dpg.bind_item_font("web_access_symbol", themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("web_access_symbol", "my_pulsating_gray_text_theme")  # tag
                        dpg.add_text("WEB", tag="web_access_text")

                    global subtitle_bottom_y0
                    subtitle_bottom_y0 = (avatar_panel_h - 24) + gui_config.subtitle_y0
                    dpg.add_text("",
                                 pos=(gui_config.subtitle_x0,
                                      subtitle_bottom_y0),  # Position doesn't really matter; the text is empty for now, and will be re-positioned when subtitles are generated.
                                 color=gui_config.subtitle_color,
                                 wrap=(avatar_panel_w - 16) - gui_config.subtitle_x0 - gui_config.subtitle_text_wrap_margin,
                                 tag="avatar_subtitle_text")
                    dpg.bind_item_font("avatar_subtitle_text", subtitle_font)  # tag

                with dpg.child_window(tag="mode_toggle_controls",
                                      width=-1,
                                      height=gui_config.chat_controls_h,
                                      no_scrollbar=True,
                                      no_scroll_with_mouse=True):
                    with dpg.group(horizontal=True):
                        def toggle_tools_enabled():
                            app_state["tools_enabled"] = not app_state["tools_enabled"]
                        def toggle_docs_enabled():
                            app_state["docs_enabled"] = not app_state["docs_enabled"]
                        def toggle_speculate_enabled():
                            app_state["speculate_enabled"] = not app_state["speculate_enabled"]
                        def toggle_speech_enabled():
                            app_state["avatar_speech_enabled"] = not app_state["avatar_speech_enabled"]
                        def toggle_subtitles_enabled():
                            app_state["avatar_subtitles_enabled"] = not app_state["avatar_subtitles_enabled"]
                            avatar_controller.avatar_controller_config.subtitles_enabled = app_state["avatar_subtitles_enabled"]
                        dpg.add_checkbox(label="Tools", default_value=app_state["tools_enabled"], callback=toggle_tools_enabled, tag="tools_enabled_checkbox")
                        dpg.add_tooltip("tools_enabled_checkbox", tag="tools_enabled_tooltip")  # tag
                        dpg.add_text("Provide tools to the AI, such as web search.", parent="tools_enabled_tooltip")  # tag

                        dpg.add_checkbox(label="Documents", default_value=app_state["docs_enabled"], callback=toggle_docs_enabled, tag="docs_enabled_checkbox")
                        dpg.add_tooltip("docs_enabled_checkbox", tag="docs_enabled_tooltip")  # tag
                        dpg.add_text("Before responding, search document database for relevant information.", parent="docs_enabled_tooltip")  # tag

                        dpg.add_checkbox(label="Speculation", default_value=app_state["speculate_enabled"], callback=toggle_speculate_enabled, tag="speculate_enabled_checkbox")
                        dpg.add_tooltip("speculate_enabled_checkbox", tag="speculate_enabled_tooltip")  # tag
                        dpg.add_text("ON: Let AI freely use its internal knowledge.\nOFF: Remind AI to use information from context only.\nOFF, and documents ON: As above, plus skip AI generation if no match in document database.", parent="speculate_enabled_tooltip")  # tag

                        dpg.add_checkbox(label="Speech", default_value=app_state["avatar_speech_enabled"], callback=toggle_speech_enabled, tag="speech_enabled_checkbox")
                        dpg.add_tooltip("speech_enabled_checkbox", tag="speech_enabled_tooltip")  # tag
                        dpg.add_text("Have the avatar speak the final response (TTS, text to speech).", parent="speech_enabled_tooltip")  # tag

                        dpg.add_checkbox(label="Subtitles", default_value=app_state["avatar_subtitles_enabled"], callback=toggle_subtitles_enabled, tag="avatar_subtitles_checkbox")
                        dpg.add_tooltip("avatar_subtitles_checkbox", tag="subtitles_enabled_tooltip")  # tag
                        if gui_config.translator_target_lang is not None:
                            subtitle_explanation_str = f"Subtitle the avatar's speech (language: {gui_config.translator_target_lang.upper()})."
                        else:
                            subtitle_explanation_str = "Closed-caption (CC) the avatar's speech."
                        dpg.add_text(f"{subtitle_explanation_str}\nUsed when TTS is ON.\nTakes effect from the AI's next chat message onward.", parent="subtitles_enabled_tooltip")  # tag

        with dpg.child_window(tag="chat_ai_warning",
                              height=gui_config.ai_warning_h,
                              no_scrollbar=True,
                              no_scroll_with_mouse=True):
            with dpg.group(horizontal=True):
                def start_new_chat_callback() -> None:
                    new_chat_head_node_id = app_state["new_chat_HEAD"]
                    app_state["HEAD"] = new_chat_head_node_id
                    build_linearized_chat_panel()
                    dpg.focus_item("chat_field")  # tag  # Focus the chat field for convenience, since the whole point of a new chat is to immediately start a new conversation.
                    # Acknowledge the action in the GUI.
                    gui_animation.animator.add(gui_animation.ButtonFlash(message="New chat started!",
                                                                         target_button=new_chat_button,
                                                                         target_tooltip=new_chat_tooltip,
                                                                         target_text=new_chat_tooltip_text,
                                                                         original_theme=themes_and_fonts.global_theme,
                                                                         duration=gui_config.acknowledgment_duration))

                def copy_chatlog_to_clipboard_as_markdown_callback() -> None:
                    shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
                    with current_chat_history_lock:
                        if not current_chat_history:
                            return

                        output_text = io.StringIO()
                        output_text.write(f"# Raven-librarian chatlog\n\n- *HEAD node ID*: `{current_chat_history[-1].node_id}`\n- *Log generated*: {chatutil.format_chatlog_datetime_now()}\n\n{'-' * 80}\n\n")
                        for message_number, displayed_chat_message in enumerate(current_chat_history):
                            node_payload = datastore.get_payload(displayed_chat_message.node_id)  # auto-selects active revision  TODO: later (chat editing), we need to set the revision to load
                            message = node_payload["message"]
                            role = message["role"]
                            persona = node_payload["general_metadata"]["persona"]  # stored persona for this chat message
                            text = message["content"]
                            formatted_message = format_chat_message_for_clipboard(message_number=message_number,
                                                                                  role=role,
                                                                                  persona=persona,
                                                                                  text=text,
                                                                                  add_heading=True)  # In the full chatlog, the message numbers and role names are important, so always include them.
                            if shift_pressed:
                                payload_datetime = node_payload["general_metadata"]["datetime"]  # of the active revision!
                                node_active_revision = datastore.get_revision(displayed_chat_message.node_id)
                                header = f"- *Node ID*: `{displayed_chat_message.node_id}`\n- *Revision date*: {payload_datetime}\n- *Revision number*: {node_active_revision}\n\n"  # yes, it'll say `None` when no node ID is available (incoming streaming message), which is exactly what we want.
                            else:
                                header = ""
                            output_text.write(f"{header}{formatted_message}\n\n{'-' * 80}\n\n")
                    dpg.set_clipboard_text(output_text.getvalue())
                    # Acknowledge the action in the GUI.
                    mode = "with node IDs" if shift_pressed else "as-is"
                    gui_animation.animator.add(gui_animation.ButtonFlash(message=f"Copied to clipboard! ({mode})",
                                                                         target_button=copy_chat_button,
                                                                         target_tooltip=copy_chat_tooltip,
                                                                         target_text=copy_chat_tooltip_text,
                                                                         original_theme=themes_and_fonts.global_theme,
                                                                         duration=gui_config.acknowledgment_duration))

                def stop_generation_callback() -> None:
                    stop_ai_turn()
                    # Acknowledge the action in the GUI.
                    gui_animation.animator.add(gui_animation.ButtonFlash(message="Interrupted!",
                                                                         target_button=stop_generation_button,
                                                                         target_tooltip=stop_generation_tooltip,
                                                                         target_text=stop_generation_tooltip_text,
                                                                         original_theme=themes_and_fonts.global_theme,
                                                                         duration=gui_config.acknowledgment_duration))

                def stop_speech_callback() -> None:
                    avatar_controller.stop_tts()
                    # Acknowledge the action in the GUI.
                    gui_animation.animator.add(gui_animation.ButtonFlash(message="Stopped speaking!",
                                                                         target_button=stop_speech_button,
                                                                         target_tooltip=stop_speech_tooltip,
                                                                         target_text=stop_speech_tooltip_text,
                                                                         original_theme=themes_and_fonts.global_theme,
                                                                         duration=gui_config.acknowledgment_duration))

                def toggle_fullscreen():
                    dpg.toggle_viewport_fullscreen()
                    # resize_gui()  # TODO: resiable GUI

                new_chat_button = dpg.add_button(label=fa.ICON_FILE,
                                                 callback=start_new_chat_callback,
                                                 width=gui_config.toolbutton_w,
                                                 tag="chat_new_button")
                dpg.bind_item_font("chat_new_button", themes_and_fonts.icon_font_solid)  # tag
                dpg.bind_item_theme("chat_new_button", "disablable_button_theme")  # tag
                new_chat_tooltip = dpg.add_tooltip("chat_new_button")  # tag
                new_chat_tooltip_text = dpg.add_text("Start new chat [Ctrl+N]", parent=new_chat_tooltip)

                dpg.add_button(label=fa.ICON_DIAGRAM_PROJECT,
                               callback=lambda: None,  # TODO
                               enabled=False,
                               width=gui_config.toolbutton_w,
                               tag="chat_open_graph_button")
                dpg.bind_item_font("chat_open_graph_button", themes_and_fonts.icon_font_solid)  # tag
                dpg.bind_item_theme("chat_open_graph_button", "disablable_button_theme")  # tag
                open_graph_tooltip = dpg.add_tooltip("chat_open_graph_button")  # tag
                dpg.add_text("Open graph view", parent=open_graph_tooltip)

                copy_chat_button = dpg.add_button(label=fa.ICON_COPY,
                                                  callback=copy_chatlog_to_clipboard_as_markdown_callback,
                                                  width=gui_config.toolbutton_w,
                                                  tag="chat_copy_to_clipboard_button")
                dpg.bind_item_font("chat_copy_to_clipboard_button", themes_and_fonts.icon_font_solid)  # tag
                dpg.bind_item_theme("chat_copy_to_clipboard_button", "disablable_button_theme")  # tag
                copy_chat_tooltip = dpg.add_tooltip("chat_copy_to_clipboard_button")  # tag
                copy_chat_tooltip_text = dpg.add_text("Copy this conversation to clipboard [F8]\n    no modifier: as-is\n    with Shift: include message node IDs", parent=copy_chat_tooltip)

                stop_generation_button = dpg.add_button(label=fa.ICON_SQUARE,
                                                        callback=stop_generation_callback,
                                                        enabled=False,
                                                        width=gui_config.toolbutton_w,
                                                        tag="chat_stop_generation_button")
                dpg.bind_item_font("chat_stop_generation_button", themes_and_fonts.icon_font_solid)  # tag
                dpg.bind_item_theme("chat_stop_generation_button", "disablable_button_theme")  # tag
                stop_generation_tooltip = dpg.add_tooltip("chat_stop_generation_button")  # tag
                stop_generation_tooltip_text = dpg.add_text("Interrupt the AI [Ctrl+G]\nThis stops the AI when it is writing.", parent=stop_generation_tooltip)

                stop_speech_button = dpg.add_button(label=fa.ICON_COMMENT_SLASH,
                                                    callback=stop_speech_callback,
                                                    enabled=False,
                                                    width=gui_config.toolbutton_w,
                                                    tag="chat_stop_speech_button")
                dpg.bind_item_font("chat_stop_speech_button", themes_and_fonts.icon_font_solid)  # tag
                dpg.bind_item_theme("chat_stop_speech_button", "disablable_button_theme")  # tag
                stop_speech_tooltip = dpg.add_tooltip("chat_stop_speech_button")  # tag
                stop_speech_tooltip_text = dpg.add_text("Stop speaking [Ctrl+S]", parent=stop_speech_tooltip)

                dpg.add_button(label=fa.ICON_EXPAND,
                               callback=toggle_fullscreen,
                               width=gui_config.toolbutton_w,
                               tag="fullscreen_button")
                dpg.bind_item_font("fullscreen_button", themes_and_fonts.icon_font_solid)  # tag
                with dpg.tooltip("fullscreen_button", tag="fullscreen_tooltip"):  # tag
                    dpg.add_text("Toggle fullscreen [F11]",
                                 tag="fullscreen_tooltip_text")

                # # DEBUG / TESTING button
                # _testing_data_eyes_enabled = False
                # def testing_callback() -> None:
                #     global _testing_data_eyes_enabled
                #     _testing_data_eyes_enabled = not _testing_data_eyes_enabled
                #     if _testing_data_eyes_enabled:
                #         avatar_controller.start_data_eyes()
                #     else:
                #         avatar_controller.stop_data_eyes()
                #     # Acknowledge the action in the GUI.
                #     gui_animation.animator.add(gui_animation.ButtonFlash(message="Ran the action being tested!",
                #                                                          target_button=testing_button,
                #                                                          target_tooltip=testing_tooltip,
                #                                                          target_text=testing_tooltip_text,
                #                                                          original_theme=themes_and_fonts.global_theme,
                #                                                          duration=gui_config.acknowledgment_duration))
                # testing_button = dpg.add_button(label=fa.ICON_VOLCANO,
                #                                 callback=testing_callback,
                #                                 width=gui_config.toolbutton_w,
                #                                 tag="chat_testing_button")
                # dpg.bind_item_font("chat_testing_button", themes_and_fonts.icon_font_solid)  # tag
                # dpg.bind_item_theme("chat_testing_button", "disablable_button_theme")  # tag
                # testing_tooltip = dpg.add_tooltip("chat_testing_button")  # tag
                # testing_tooltip_text = dpg.add_text("Developer button for testing purposes. What will it do today?!", parent=testing_tooltip)

                n_below_chat_buttons = 6
                avatar_panel_left = gui_config.chat_panel_w - n_below_chat_buttons * (gui_config.toolbutton_w + 8)
                dpg.add_spacer(width=avatar_panel_left + 60)

                with dpg.group(horizontal=True):
                    dpg.add_text(fa.ICON_TRIANGLE_EXCLAMATION, color=(255, 180, 120), tag="ai_warning_icon")  # orange
                    dpg.add_text("Response quality and factual accuracy depend on the connected AI. Always verify important facts independently.", color=(255, 180, 120), tag="ai_warning_text")  # orange
                dpg.bind_item_font("ai_warning_icon", themes_and_fonts.icon_font_solid)  # tag

# --------------------------------------------------------------------------------
# Animations, live updates

def update_animations():
    gui_animation.animator.render_frame()

# --------------------------------------------------------------------------------
# TODO: App window (viewport) resizing

# def toggle_fullscreen():
#     dpg.toggle_viewport_fullscreen()
#     resize_gui()  # see below
#
# def resize_gui():
#     """Wait for the viewport size to actually change, then resize dynamically sized GUI elements.
#
#     This is handy for toggling fullscreen, because the size changes at the next frame at the earliest.
#     For the viewport resize callback, that one fires (*almost* always?) after the size has already changed.
#     """
#     logger.debug("resize_gui: Entered. Waiting for viewport size change.")
#     if guiutils.wait_for_resize(gui_instance.window):
#         _resize_gui()
#     logger.debug("resize_gui: Done.")
#
# def _resize_gui():
#     if gui_instance is None:
#         return
#     gui_instance._resize_gui()
# dpg.set_viewport_resize_callback(_resize_gui)

# --------------------------------------------------------------------------------
# Hotkey support

combobox_choice_map = None   # DPG tag or ID -> (choice_strings, callback)
def librarian_hotkeys_callback(sender, app_data):
    # # Hotkeys while an "open file" or "save as" dialog is shown - fdialog handles its own hotkeys
    # if is_any_modal_window_visible():
    #     return

    key = app_data
    ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
    shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)

    # ------------------------------------------------------------
    # Helpers for operating on the most recent chat message

    def get_last_message() -> Optional[DisplayedChatMessage]:
        global current_chat_history  # for documenting the intent only
        if not current_chat_history:
            return None
        displayed_message = current_chat_history[-1]
        return displayed_message

    def fire_event_if_exists(action: str) -> None:
        displayed_message = get_last_message()
        if displayed_message is None:
            return
        if action in displayed_message.gui_button_callbacks:
            displayed_message.gui_button_callbacks[action]()

    # ------------------------------------------------------------

    # NOTE: If you update this, to make the hotkeys discoverable, update also:
    #  - The tooltips wherever the GUI elements are created or updated (search for e.g. "[F9]", may appear in multiple places)
    #  - The help window

    # Hotkeys that are always available, regardless of any dialogs (even if modal)
    if key == dpg.mvKey_F11:  # de facto standard hotkey for toggle fullscreen
        toggle_fullscreen()
    # elif some_modal_window_visible():
    #     ...

    # Hotkeys for main window, while no modal window is shown
    if key == dpg.mvKey_F8:  # NOTE: Shift is a modifier here
        copy_chatlog_to_clipboard_as_markdown_callback()
    # Ctrl+Shift+...
    elif ctrl_pressed and shift_pressed:
        # Some hidden debug features. Mnemonic: "Mr. T Lite" (Ctrl + Shift + M, R, T, L)
        if key == dpg.mvKey_M:
            dpg.show_metrics()
            dpg_avatar_renderer.configure_fps_counter(show=None)  # `None` = toggle
        elif key == dpg.mvKey_R:
            dpg.show_item_registry()
        elif key == dpg.mvKey_T:
            dpg.show_font_manager()
        elif key == dpg.mvKey_L:
            dpg.show_style_editor()
    # Ctrl+...
    elif ctrl_pressed:
        if key == dpg.mvKey_Spacebar:
            dpg.focus_item("chat_field")  # tag
        elif key == dpg.mvKey_R:
            fire_event_if_exists("reroll")
        elif key == dpg.mvKey_Left:
            fire_event_if_exists("prev")
        elif key == dpg.mvKey_Right:
            fire_event_if_exists("next")
        elif key == dpg.mvKey_N:
            start_new_chat_callback()
        elif key == dpg.mvKey_S:
            if api.tts_speaking():
                stop_speech_callback()
            else:
                fire_event_if_exists("speak")
        elif key == dpg.mvKey_G:
            if dpg.is_item_enabled("chat_stop_generation_button"):  # tag
                stop_generation_callback()

    # Bare key
    #
    # NOTE: These are global across the whole app (when no modal window is open) - be very careful here!
    else:
        if dpg.is_item_focused("chat_field"):
            if key == dpg.mvKey_Return:  # tag  # regardless of modifier state
                send_message_to_ai_callback()
                dpg.focus_item("chat_field")  # tag

    # # Ctrl+Shift+...
    # if ctrl_pressed and shift_pressed:
    #     if key == dpg.mvKey_E:  # emotions
    #         show_open_json_dialog()
    #     elif key == dpg.mvKey_A:  # load animator settings
    #         show_open_animator_settings_dialog()
    #     elif key == dpg.mvKey_S:  # save animator settings
    #         show_save_animator_settings_dialog()
    #
    # # Ctrl+...
    # elif ctrl_pressed:
    #     if key == dpg.mvKey_O:
    #         show_open_input_image_dialog()
    #     elif key == dpg.mvKey_R:
    #         gui_instance.on_reload_input_image(sender, app_data)
    #     elif key == dpg.mvKey_B:
    #         show_open_backdrop_image_dialog()
    #     elif key == dpg.mvKey_T:
    #         gui_instance.toggle_talking()
    #     elif key == dpg.mvKey_P:
    #         gui_instance.toggle_animator_paused()
    #     elif key == dpg.mvKey_E:
    #         dpg.focus_item(gui_instance.emotion_choice)
    #     elif key == dpg.mvKey_V:
    #         dpg.focus_item(gui_instance.voice_choice)
    #     elif key == dpg.mvKey_S:
    #         if not gui_instance.speaking:
    #             gui_instance.on_start_speaking(sender, app_data)
    #         else:
    #             gui_instance.on_stop_speaking(sender, app_data)
    #
    # # Bare key
    # #
    # # NOTE: These are global across the whole app (when no modal window is open) - be very careful here!
    # else:
    #     if key == dpg.mvKey_F11:
    #         toggle_fullscreen()
    #     else:
    #         # {widget_tag_or_id: list_of_choices}
    #         global combobox_choice_map
    #         if combobox_choice_map is None:  # build on first use
    #             combobox_choice_map = {gui_instance.emotion_choice: (gui_instance.emotion_names, gui_instance.on_send_emotion),
    #                                    gui_instance.voice_choice: (gui_instance.voice_names, None)}
    #         def browse(choice_widget, data):
    #             choices, callback = data
    #             index = choices.index(dpg.get_value(choice_widget))
    #             if key == dpg.mvKey_Down:
    #                 new_index = min(index + 1, len(choices) - 1)
    #             elif key == dpg.mvKey_Up:
    #                 new_index = max(index - 1, 0)
    #             elif key == dpg.mvKey_Home:
    #                 new_index = 0
    #             elif key == dpg.mvKey_End:
    #                 new_index = len(choices) - 1
    #             else:
    #                 new_index = None
    #             if new_index is not None:
    #                 dpg.set_value(choice_widget, choices[new_index])
    #                 if callback is not None:
    #                     callback(sender, app_data)  # the callback doesn't trigger automatically if we programmatically set the combobox value
    #         focused_item = dpg.get_focused_item()
    #         focused_item = dpg.get_item_alias(focused_item)
    #         if focused_item in combobox_choice_map.keys():
    #             browse(focused_item, combobox_choice_map[focused_item])
with dpg.handler_registry(tag="librarian_handler_registry"):  # global (whole viewport)
    dpg.add_key_press_handler(tag="librarian_hotkeys_handler", callback=librarian_hotkeys_callback)

# --------------------------------------------------------------------------------
# Set up app exit cleanup

def clear_background_tasks(wait: bool):
    """Stop (cancel) and delete all background tasks."""
    task_manager.clear(wait=wait)

def clean_up_at_exit():
    logger.info("App exiting.")
    clear_background_tasks(wait=True)
dpg.set_exit_callback(clean_up_at_exit)

# --------------------------------------------------------------------------------
# Start the app

logger.info("App bootup...")

avatar_instance_id = api.avatar_load(librarian_config.avatar_config.image_path)
api.avatar_load_emotion_templates(avatar_instance_id, {})  # send empty dict -> reset emotion templates to server defaults
api.avatar_start(avatar_instance_id)
dpg_avatar_renderer.start(avatar_instance_id)
avatar_controller.initialize(avatar_instance_id=avatar_instance_id,
                             data_eyes_fadeout_duration=librarian_config.avatar_config.data_eyes_fadeout_duration,
                             emotion_autoreset_interval=librarian_config.avatar_config.emotion_autoreset_interval,
                             emotion_blacklist=librarian_config.avatar_config.emotion_blacklist,
                             stop_tts_button_gui_widget="chat_stop_speech_button",  # tag
                             on_tts_idle=None,
                             tts_idle_check_interval=None,
                             subtitles_enabled=app_state["avatar_subtitles_enabled"],
                             subtitle_text_gui_widget="avatar_subtitle_text",  # tag
                             subtitle_left_x0=gui_config.subtitle_x0,
                             subtitle_bottom_y0=subtitle_bottom_y0,
                             translator_source_lang=gui_config.translator_source_lang,
                             translator_target_lang=gui_config.translator_target_lang,
                             main_window_w=gui_config.main_window_w,
                             main_window_h=gui_config.main_window_h,
                             executor=bg)  # use the same thread pool as our main task manager

gui_alive = True  # Global flag for app shutdown, for background tasks in our main task manager to detect if GUI teardown has started (so that updating GUI elements is no longer safe).
def gui_shutdown() -> None:
    """App exit: gracefully shut down parts that access DPG."""
    global gui_alive
    avatar_controller.stop_tts()  # Stop the TTS speaking so that the speech background thread (if any) exits.
    logger.info("gui_shutdown: entered")
    # Tell background tasks that GUI teardown is in progress (app is shutting down, so trying to update GUI elements may hang the app).
    # This also tells `run_ai_turn` to exit, so we don't need to clear the `ai_turn_task_manager`.
    # Same for `avatar_preprocess_task` in the `avatar_preprocess_task_manager`.
    gui_alive = False
    task_manager.clear(wait=True)  # Wait until background tasks actually exit.
    avatar_controller.shutdown()
    gui_animation.animator.clear()
    # global gui_instance  # TODO: maybe we need to encapsulate the main GUi into a class? Or maybe not?
    # gui_instance = None
    logger.info("gui_shutdown: done")
dpg.set_exit_callback(gui_shutdown)

def app_shutdown() -> None:
    """App exit: gracefully shut down parts that don't need DPG.

    This is guaranteed to run even if DPG shutdown never completes gracefully, as long as it doesn't hang the main thread, or segfault the process.

    Currently, we release server-side resources here.
    """
    logger.info("app_shutdown: entered")
    if avatar_instance_id is not None:
        try:
            api.avatar_unload(avatar_instance_id)  # delete the instance so the server can release the resources
        except requests.exceptions.ConnectionError:  # server has gone bye-bye
            pass
    logger.info("app_shutdown: done")
atexit.register(app_shutdown)

dpg.set_primary_window(main_window, True)  # Make this DPG "window" occupy the whole OS window (DPG "viewport").
dpg.set_viewport_vsync(True)
dpg.show_viewport()

# Load default animator settings from disk.
#
# We must defer loading the animator settings until after the GUI has been rendered at least once,
# so that if there are any issues during loading, we can open a modal dialog. (We don't currently do that, though.)
def _load_initial_animator_settings() -> None:
    animator_json_path = pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "avatar", "assets", "settings", "animator.json")).expanduser().resolve()

    try:
        with open(animator_json_path, "r", encoding="utf-8") as json_file:
            animator_settings = json.load(json_file)
    except FileNotFoundError:
        print(colorizer.colorize(f"AI avatar animator default config file not found at '{animator_json_path}'.", colorizer.Style.BRIGHT, colorizer.Fore.RED) + " Please run `raven-avatar-settings-editor` once to create it.")
        logger.error(f"_load_initial_animator_settings: AI avatar animator default config file not found at '{animator_json_path}'. Please run `raven-avatar-settings-editor` once to create it.")
        sys.exit(255)
    except BaseException as exc:  # yes, also Ctrl+C
        print(colorizer.colorize("Failed to load AI avatar animator default config file.", colorizer.Style.BRIGHT, colorizer.Fore.RED) + " Details follow.")
        logger.error(f"_load_initial_animator_settings: Failed, reason {type(exc)}: {exc}")
        traceback.print_exc()
        sys.exit(255)

    animator_settings.update(librarian_config.avatar_config.animator_settings_overrides)

    api.avatar_load_animator_settings(avatar_instance_id, animator_settings)  # send settings to server
    dpg_avatar_renderer.load_backdrop_image(animator_settings["backdrop_path"])
    dpg_avatar_renderer.configure_backdrop(new_width=avatar_panel_w - 16,
                                           new_height=avatar_panel_h - 16,
                                           new_blur_state=animator_settings["backdrop_blur"])

dpg.set_frame_callback(2, _load_initial_animator_settings)

def _build_initial_chat_view(sender, app_data) -> None:
    build_linearized_chat_panel()
dpg.set_frame_callback(3, _build_initial_chat_view)

logger.info("App render loop starting.")

try:
    # We control the render loop manually to have a convenient place to update our GUI animations just before rendering each frame.
    while dpg.is_dearpygui_running():
        update_animations()
        dpg.render_dearpygui_frame()
    # dpg.start_dearpygui()  # automatic render loop
except KeyboardInterrupt:
    clear_background_tasks(wait=False)  # signal background tasks to exit

logger.info("App render loop exited.")

dpg.destroy_context()

def main() -> None:  # TODO: we don't really need this; it's just for console_scripts.
    pass
