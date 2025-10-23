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
    import concurrent.futures
    import json
    import os
    import pathlib
    import platform
    import requests
    import sys
    import traceback
    from typing import Tuple, Union

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
    from ..client.avatar_controller import DPGAvatarController
    from ..client.avatar_renderer import DPGAvatarRenderer
    from ..client import config as client_config

    from ..common import bgtask

    from ..common.gui import animation as gui_animation
    from ..common.gui import helpcard
    from ..common.gui import utils as guiutils

    from . import appstate
    from .chat_controller import DPGChatController
    from . import config as librarian_config
    # from . import chattree
    from . import hybridir
    from . import llmclient

    gui_config = librarian_config.gui_config  # shorthand, this is used a lot
logger.info(f"Libraries loaded in {tim.dt:0.6g}s.")

# ----------------------------------------
# Module bootup

bg = concurrent.futures.ThreadPoolExecutor()
task_manager = bgtask.TaskManager(name="librarian",  # for avatar renderer
                                  mode="concurrent",
                                  executor=bg)
gui_resize_task_manager = bgtask.TaskManager(name="librarian_gui_resize",  # de-spammer for expensive parts of GUI resizing
                                             mode="sequential",
                                             executor=bg)
api.initialize(raven_server_url=client_config.raven_server_url,
               raven_api_key_file=client_config.raven_api_key_file,
               tts_playback_audio_device=client_config.tts_playback_audio_device,
               executor=bg)  # reuse our executor so the TTS audio player goes in the same thread pool  # TODO: there's currently a bug, because `llmclient` inits API first, with a default executor.

llm_backend_url = librarian_config.llm_backend_url

# These are initialized later, when the app starts
avatar_instance_id = None

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
    print(f"{colorizer.Fore.GREEN}{colorizer.Style.BRIGHT}Connected to Raven-server at {client_config.raven_server_url}{colorizer.Style.RESET_ALL}")
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
# Set up the main window

logger.info("Initial GUI setup...")
with timer() as tim:
    with dpg.window(show=True, modal=False, no_title_bar=False, tag="librarian_main_window",
                    label="Raven-librarian main window",
                    no_scrollbar=True, autosize=True) as main_window:  # DPG "window" inside the app OS window ("viewport"), container for the whole GUI
        # We all love magic numbers!
        #
        # The dynamic panel sizes are not available at startup, until the GUI is rendered at least once,
        # so we must compute the initial sizes explicitly. These is also needed for dynamic resizing.
        def _get_chat_panel_base_size() -> Tuple[int, int]:  # at initial view, with the window at its design size
            w = gui_config.chat_panel_w + 16  # 16 = round border (8 on each side)
            h = gui_config.main_window_h - (gui_config.ai_warning_h + 16) - (gui_config.chat_controls_h + 16) + 8
            return w, h
        def _get_chat_panel_size(main_window_w: int, main_window_h: int) -> Tuple[int, int]:  # at current window size
            extra_w = main_window_w - gui_config.main_window_w
            extra_h = main_window_h - gui_config.main_window_h
            base_w, base_h = _get_chat_panel_base_size()
            w = base_w + extra_w
            h = base_h + extra_h
            return w, h
        def _get_avatar_panel_base_size() -> Tuple[int, int]:
            chat_panel_base_w, chat_panel_base_h = _get_chat_panel_base_size()
            w = (gui_config.main_window_w - chat_panel_base_w - 3 * 8)  # the 3 * 8 are the outer borders outside the panels (between panel and window edge, and between the panels)
            h = chat_panel_base_h
            return w, h
        def _get_avatar_panel_size(main_window_w: int, main_window_h: int) -> Tuple[int, int]:
            extra_w = 0  # avatar panel keeps the same width
            extra_h = main_window_h - gui_config.main_window_h
            base_w, base_h = _get_avatar_panel_base_size()
            w = base_w + extra_w
            h = base_h + extra_h
            return w, h
        def _get_chat_field_base_width() -> int:
            return gui_config.chat_panel_w - gui_config.toolbutton_w - 8
        def _get_chat_field_width(main_window_w: int) -> int:
            extra_w = main_window_w - gui_config.main_window_w
            base_w = _get_chat_field_base_width()
            w = base_w + extra_w
            return w
        def _get_chat_controls_base_size() -> Tuple[int, int]:
            chat_panel_base_w, chat_panel_base_h = _get_chat_panel_base_size()
            w = chat_panel_base_w
            h = gui_config.chat_controls_h
            return w, h
        def _get_chat_controls_size(main_window_w: int, main_window_h: int) -> Tuple[int, int]:
            extra_w = main_window_w - gui_config.main_window_w
            extra_h = 0
            base_w, base_h = _get_chat_controls_base_size()
            w = base_w + extra_w
            h = base_h + extra_h
            return w, h

        with dpg.group(horizontal=True):
            with dpg.group():  # left column: linearized chat view
                # The `DPGChatController` goes into this panel when the app boots up.
                chat_panel_w, chat_panel_h = _get_chat_panel_base_size()
                chat_panel_widget = dpg.add_child_window(tag="chat_panel",
                                                         width=chat_panel_w,
                                                         height=chat_panel_h)

                chat_controls_w, chat_controls_h = _get_chat_controls_base_size()
                with dpg.child_window(tag="chat_controls",
                                      width=chat_controls_w,
                                      height=chat_controls_h,
                                      no_scrollbar=True,
                                      no_scroll_with_mouse=True):
                    with dpg.group(horizontal=True):
                        def send_message_to_ai_callback() -> None:
                            user_message_text = dpg.get_value("chat_field")
                            dpg.set_value("chat_field", "")
                            chat_controller.chat_round(user_message_text)
                        dpg.add_input_text(tag="chat_field",
                                           default_value="",
                                           hint="[ask the AI questions here] [Ctrl+Space to focus]",
                                           width=_get_chat_field_base_width())
                        dpg.add_button(label=fa.ICON_PAPER_PLANE,
                                       callback=send_message_to_ai_callback,
                                       width=gui_config.toolbutton_w,
                                       tag="chat_send_button")
                        dpg.bind_item_font("chat_send_button", themes_and_fonts.icon_font_solid)  # tag  # TODO: make this change into a cancel button while the LLM is writing.
                        dpg.bind_item_theme("chat_send_button", "disablable_button_theme")  # tag
                        with dpg.tooltip("chat_send_button"):  # tag
                            dpg.add_text("Send to AI [Enter]")

            with dpg.group():  # right column: AI avatar
                avatar_panel_w, avatar_panel_h = _get_avatar_panel_base_size()
                with dpg.child_window(tag="avatar_panel",
                                      width=avatar_panel_w,
                                      height=avatar_panel_h,
                                      no_scrollbar=True,
                                      no_scroll_with_mouse=True):
                    dpg_avatar_renderer = DPGAvatarRenderer(gui_parent="avatar_panel",
                                                            avatar_x_center=(avatar_panel_w // 2),
                                                            avatar_y_bottom=(avatar_panel_h - 8),
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
                            avatar_controller.subtitles_enabled = app_state["avatar_subtitles_enabled"]
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

        with dpg.child_window(tag="chat_global_buttons",
                              height=gui_config.ai_warning_h,
                              no_scrollbar=True,
                              no_scroll_with_mouse=True):
            with dpg.group(horizontal=True):
                def start_new_chat_callback() -> None:
                    new_chat_head_node_id = app_state["new_chat_HEAD"]
                    app_state["HEAD"] = new_chat_head_node_id
                    chat_controller.view.build()
                    dpg.focus_item("chat_field")  # tag  # Focus the chat field for convenience, since the whole point of a new chat is to immediately start a new conversation.
                    # Acknowledge the action in the GUI.
                    gui_animation.animator.add(gui_animation.ButtonFlash(message="New chat started!",
                                                                         target_button=new_chat_button,
                                                                         target_tooltip=new_chat_tooltip,
                                                                         target_text=new_chat_tooltip_text,
                                                                         original_theme=dpg.get_item_theme(new_chat_tooltip),
                                                                         duration=gui_config.acknowledgment_duration))

                def copy_chatlog_to_clipboard_as_markdown_callback(self) -> None:
                    shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
                    if (chatlog_text := chat_controller.view.get_chatlog_as_markdown(include_metadata=shift_pressed)) is not None:
                        dpg.set_clipboard_text(chatlog_text)
                    # Acknowledge the action in the GUI.
                    mode = "with node IDs" if shift_pressed else "as-is"
                    gui_animation.animator.add(gui_animation.ButtonFlash(message=f"Copied to clipboard! ({mode})",
                                                                         target_button=copy_chat_button,
                                                                         target_tooltip=copy_chat_tooltip,
                                                                         target_text=copy_chat_tooltip_text,
                                                                         original_theme=dpg.get_item_theme(copy_chat_tooltip),
                                                                         duration=gui_config.acknowledgment_duration))

                def stop_text_generation_callback() -> None:
                    chat_controller.stop_ai_turn()
                    # Acknowledge the action in the GUI.
                    gui_animation.animator.add(gui_animation.ButtonFlash(message="Interrupted!",
                                                                         target_button=stop_generation_button,
                                                                         target_tooltip=stop_generation_tooltip,
                                                                         target_text=stop_generation_tooltip_text,
                                                                         original_theme=dpg.get_item_theme(stop_generation_tooltip),
                                                                         duration=gui_config.acknowledgment_duration))

                def stop_speech_callback() -> None:
                    avatar_controller.stop_tts()
                    # Acknowledge the action in the GUI.
                    gui_animation.animator.add(gui_animation.ButtonFlash(message="Stopped speaking!",
                                                                         target_button=stop_speech_button,
                                                                         target_tooltip=stop_speech_tooltip,
                                                                         target_text=stop_speech_tooltip_text,
                                                                         original_theme=dpg.get_item_theme(stop_speech_tooltip),
                                                                         duration=gui_config.acknowledgment_duration))

                def toggle_fullscreen():
                    dpg.toggle_viewport_fullscreen()
                    resize_gui()

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
                                                        callback=stop_text_generation_callback,
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

                # We'll define and bind the callback later, when we set up the help window.
                dpg.add_button(label=fa.ICON_CIRCLE_QUESTION,
                               width=gui_config.toolbutton_w,
                               tag="help_button")
                dpg.bind_item_font("help_button", themes_and_fonts.icon_font_regular)  # tag
                with dpg.tooltip("help_button", tag="help_tooltip"):  # tag
                    dpg.add_text("Open the Help card [F1]",
                                 tag="help_tooltip_text")

                # # DEBUG / TESTING button
                # _testing_data_eyes_enabled = False
                # def testing_callback() -> None:
                #     global _testing_data_eyes_enabled
                #     _testing_data_eyes_enabled = not _testing_data_eyes_enabled
                #     if _testing_data_eyes_enabled:
                #         avatar_controller.start_data_eyes(config=avatar_record)
                #     else:
                #         avatar_controller.stop_data_eyes(config=avatar_record)
                #     # Acknowledge the action in the GUI.
                #     gui_animation.animator.add(gui_animation.ButtonFlash(message="Ran the action being tested!",
                #                                                          target_button=testing_button,
                #                                                          target_tooltip=testing_tooltip,
                #                                                          target_text=testing_tooltip_text,
                #                                                          original_theme=dpg.get_item_theme(testing_tooltip),
                #                                                          duration=gui_config.acknowledgment_duration))
                # testing_button = dpg.add_button(label=fa.ICON_VOLCANO,
                #                                 callback=testing_callback,
                #                                 width=gui_config.toolbutton_w,
                #                                 tag="chat_testing_button")
                # dpg.bind_item_font("chat_testing_button", themes_and_fonts.icon_font_solid)  # tag
                # dpg.bind_item_theme("chat_testing_button", "disablable_button_theme")  # tag
                # testing_tooltip = dpg.add_tooltip("chat_testing_button")  # tag
                # testing_tooltip_text = dpg.add_text("Developer button for testing purposes. What will it do today?!", parent=testing_tooltip)

                n_below_chat_buttons = 7
                ai_warning_spacer_base_size = gui_config.chat_panel_w - n_below_chat_buttons * (gui_config.toolbutton_w + 8) + 60
                dpg.add_spacer(width=ai_warning_spacer_base_size, tag="ai_warning_spacer")

                with dpg.group(horizontal=True):
                    dpg.add_text(fa.ICON_TRIANGLE_EXCLAMATION, color=(255, 180, 120), tag="ai_warning_icon")  # orange
                    dpg.add_text("Response quality and factual accuracy depend on the connected AI. Always verify important facts independently.", color=(255, 180, 120), tag="ai_warning_text")  # orange
                dpg.bind_item_font("ai_warning_icon", themes_and_fonts.icon_font_solid)  # tag

# --------------------------------------------------------------------------------
# Animations, live updates

def update_animations():
    gui_animation.animator.render_frame()

# --------------------------------------------------------------------------------
# Built-in help window

hotkey_info = (env(key_indent=0, key="Ctrl+Space", action_indent=0, action="Focus text entry field", notes=""),
               env(key_indent=1, key="Enter", action_indent=0, action="Send message to AI", notes="When text entry field focused"),
               env(key_indent=1, key="Esc", action_indent=0, action="Clear text and cancel", notes="When text entry field focused"),
               helpcard.hotkey_blank_entry,
               env(key_indent=0, key="Ctrl+G", action_indent=0, action="Stop AI text generation", notes="While the AI is writing"),
               env(key_indent=0, key="Ctrl+U", action_indent=0, action="Continue last AI message", notes="Creates new revision of same node"),
               env(key_indent=0, key="Ctrl+R", action_indent=0, action="Reroll last AI message", notes="Creates new sibling"),
               helpcard.hotkey_blank_entry,
               env(key_indent=0, key="Ctrl+T", action_indent=0, action="Show/hide last thinking trace", notes="For thinking models"),
               env(key_indent=0, key="Ctrl+S", action_indent=0, action="Speak last AI message / stop speaking", notes=""),
               helpcard.hotkey_blank_entry,
               env(key_indent=0, key="Ctrl+Right", action_indent=0, action="Next sibling of last message", notes=""),
               env(key_indent=0, key="Ctrl+Left", action_indent=0, action="Previous sibling of last message", notes=""),
               helpcard.hotkey_new_column,
               env(key_indent=0, key="Ctrl+N", action_indent=0, action="Start new chat", notes=""),
               helpcard.hotkey_blank_entry,
               env(key_indent=0, key="F8", action_indent=0, action="Copy chatlog to clipboard", notes="As-is"),
               env(key_indent=1, key="Shift+F8", action_indent=0, action="Copy chatlog to clipboard", notes="With chat node IDs"),
               helpcard.hotkey_blank_entry,
               env(key_indent=0, key="F11", action_indent=0, action="Toggle fullscreen mode", notes=""),
               env(key_indent=0, key="F1", action_indent=0, action="Open this Help card", notes=""),
               )
def render_help_extras(self: helpcard.HelpWindow,
                       gui_parent: Union[str, int]) -> None:
    """Render app-specific extra information into the help card.

    Called by `HelpWindow` when the help card is first rendered.
    """
    # Chat history
    dpg_markdown.add_text(f"{self.c_hed}**Chat history**{self.c_end}", parent=gui_parent)
    g = dpg.add_group(horizontal=True, parent=gui_parent)
    g1 = dpg.add_group(horizontal=False, parent=g)
    dpg_markdown.add_text(f"{self.c_txt}The chat history is **natively nonlinear**. Messages are stored as nodes in a tree. The current chat is the HEAD, plus its ancestor chain up to the system prompt. Continuing the chat adds a new child node below the latest message displayed.{self.c_end}",
                          parent=g1)
    dpg_markdown.add_text(f"{self.c_txt}Rerolling creates a new sibling and sets the HEAD pointer to that. Previous siblings remain stored in the tree. Starting a new chat, or branching the chat, only resets the HEAD pointer.{self.c_end}",
                          parent=g1)
    dpg_markdown.add_text(f"{self.c_txt}**This is a tech demo.** Currently old chats are stored, but there is no way to access them. We will add a graph view to navigate the chat tree later.{self.c_end}",
                          parent=g1)
    dpg.add_spacer(width=1, height=themes_and_fonts.font_size, parent=g)

    # Docs database
    dpg_markdown.add_text(f"{self.c_hed}**Document database**{self.c_end} (retrieval-augmented generation, RAG)", parent=gui_parent)
    g = dpg.add_group(horizontal=True, parent=gui_parent)
    g1 = dpg.add_group(horizontal=False, parent=g)
    dpg_markdown.add_text(f'{self.c_txt}You can put *.txt* documents for the AI to access in *{librarian_config.llm_docs_dir}*. The path can be configured in *raven.librarian.config*.{self.c_end}',
                          parent=g1)
    dpg_markdown.add_text(f'{self.c_txt}The documents are search-indexed automatically. The index is automatically kept up to date. The search index is stored in *{librarian_config.llm_database_dir}*. If you ever need to clear it manually, just delete that directory.{self.c_end}',
                          parent=g1)
    dpg_markdown.add_text(f'{self.c_txt}When the {self.c_end}{self.c_hig}**Documents**{self.c_end}{self.c_txt} checkbox in the app is **ON**, the document database is automatically searched, using your latest message to the AI as the search query. If {self.c_end}{self.c_hig}**Speculation**{self.c_end}{self.c_txt} is **OFF**, and there is no match, the LLM is bypassed.{self.c_end}',
                          parent=g1)
    dpg_markdown.add_text(f'{self.c_txt}To improve search result quality, Raven-librarian uses a hybrid method: Okapi BM25 for keywords, and vector embeddings for semantic search. Results are combined with RRF (reciprocal rank fusion).{self.c_end}',
                          parent=g1)

    # Tool use (tool-calling)
    dpg_markdown.add_text(f"{self.c_hed}**Tool use** (tool-calling){self.c_end}", parent=gui_parent)
    g = dpg.add_group(horizontal=True, parent=gui_parent)
    g1 = dpg.add_group(horizontal=False, parent=g)
    dpg_markdown.add_text(f'{self.c_txt}Raven-librarian provides some tools, such as websearch, for the AI to use if it wants to. These can be enabled/disabled with the {self.c_end}{self.c_hig}**Tools**{self.c_end}{self.c_txt} checkbox in the app.{self.c_end}',
                          parent=g1)
    dpg_markdown.add_text(f'{self.c_txt}**This is a tech demo.** We plan to add more tools later.{self.c_end}',
                          parent=g1)
help_window = helpcard.HelpWindow(hotkey_info=hotkey_info,
                                  width=gui_config.help_window_w,
                                  height=gui_config.help_window_h,
                                  reference_window=main_window,
                                  themes_and_fonts=themes_and_fonts,
                                  on_render_extras=render_help_extras,
                                  on_show=None,
                                  on_hide=None)
dpg.set_item_callback("help_button", help_window.show)  # tag

# --------------------------------------------------------------------------------
# GUI resizing handler

def resize_gui() -> None:
    """Wait for the viewport size to actually change, then resize dynamically sized GUI elements.

    This is handy for toggling fullscreen, because the size changes at the next frame at the earliest.
    For the viewport resize callback, that one fires (*almost* always?) after the size has already changed.
    """
    logger.debug("resize_gui: Entered. Waiting for viewport size change.")
    if guiutils.wait_for_resize(main_window):
        _resize_gui()
    logger.debug("resize_gui: Done.")

def _resize_panels() -> None:
    """Resize the panels in the main window RIGHT NOW, based on main window size."""
    global _animator_settings  # intent only; loaded during app startup

    w, h = guiutils.get_widget_size(main_window)

    chat_panel_w, chat_panel_h = _get_chat_panel_size(main_window_w=w, main_window_h=h)
    dpg.set_item_width("chat_panel", chat_panel_w)  # tag
    dpg.set_item_height("chat_panel", chat_panel_h)  # tag
    chat_controls_w, chat_controls_h = _get_chat_controls_size(main_window_w=w, main_window_h=h)
    dpg.set_item_width("chat_controls", chat_controls_w)
    dpg.set_item_height("chat_controls", chat_controls_h)
    chat_field_w = _get_chat_field_width(main_window_w=w)
    dpg.set_item_width("chat_field", chat_field_w)  # tag

    extra_w = w - gui_config.main_window_w
    dpg.set_item_width("ai_warning_spacer", ai_warning_spacer_base_size + extra_w)  # tag

    avatar_panel_w, avatar_panel_h = _get_avatar_panel_size(main_window_w=w, main_window_h=h)
    dpg.set_item_width("avatar_panel", avatar_panel_w)  # tag
    dpg.set_item_height("avatar_panel", avatar_panel_h)  # tag
    dpg_avatar_renderer.reposition(new_x_center=(avatar_panel_w // 2),
                                   new_y_bottom=(avatar_panel_h - 8))
    dpg_avatar_renderer.configure_backdrop(new_width=avatar_panel_w - 16,
                                           new_height=avatar_panel_h - 16,
                                           new_blur_state=_animator_settings["backdrop_blur"])

    # TODO: change upscale factor too? (need to update "upscale" in `librarian_config.avatar_config.animator_settings_overrides` and send config to server)

def _resize_gui_task(task_env: env) -> None:
    """We run this in the background. Expensive parts of the GUI update benefit from the "there can be only one" mechanism."""
    if task_env.cancelled:  # while waiting in queue
        return
    logger.debug(f"_resize_gui_task: {task_env.task_name}: Updating main window GUI element sizes.")
    _resize_panels()
    if task_env.cancelled:
        return
    logger.debug(f"_resize_gui_task: {task_env.task_name}: Re-rendering linearized chat view.")
    chat_controller.view.build()
    logger.debug(f"_resize_gui_task: {task_env.task_name}: Done.")

def _resize_gui() -> None:
    """Resize dynamically sized GUI elements, RIGHT NOW (unless overridden by another call shortly in succession)."""
    logger.debug("_resize_gui: Entered.")
    logger.debug("_resize_gui: Recentering help window.")
    help_window.reposition()
    logger.debug("_resize_gui: Submitting task for computationally expensive GUI updates.")
    task_view_rebuild_task = bgtask.ManagedTask(category="raven_librarian_chat_view_rebuild",
                                                entrypoint=_resize_gui_task,
                                                running_poll_interval=0.01,
                                                pending_wait_duration=0.1)
    gui_resize_task_manager.submit(task_view_rebuild_task, env(wait=True))
    logger.debug("_resize_gui: Done.")

dpg.set_viewport_resize_callback(_resize_gui)

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

    def fire_event_if_exists(action: str) -> None:
        dpg_chat_message = chat_controller.get_last_message()
        if dpg_chat_message is None:
            return
        if action in dpg_chat_message.gui_button_callbacks:
            dpg_chat_message.gui_button_callbacks[action]()

    # ------------------------------------------------------------

    # NOTE: If you update this, to make the hotkeys discoverable, update also:
    #  - The tooltips wherever the GUI elements are created or updated (search for e.g. "[F9]", may appear in multiple places)
    #  - The help window

    # Hotkeys that are always available, regardless of any dialogs (even if modal)
    if key == dpg.mvKey_F11:  # de facto standard hotkey for toggle fullscreen
        toggle_fullscreen()

    # Hotkeys while the Help card is shown - helpcard handles its own hotkeys
    elif help_window.is_visible():
        return

    elif key == dpg.mvKey_F1:  # de facto standard hotkey for help
        help_window.show()

    # Hotkeys for main window, while no modal window is shown
    elif key == dpg.mvKey_F8:  # NOTE: Shift is a modifier here
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
        elif key == dpg.mvKey_T:
            fire_event_if_exists("toggle_thinking_trace")
        elif key == dpg.mvKey_U:
            fire_event_if_exists("continue")
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
                stop_text_generation_callback()

    # Bare key
    #
    # NOTE: These are global across the whole app (when no modal window is open) - be very careful here!
    else:
        if dpg.is_item_focused("chat_field"):
            if key == dpg.mvKey_Return:  # tag  # regardless of modifier state
                send_message_to_ai_callback()
                dpg.focus_item("chat_field")  # tag
        # else:
        #     # {widget_tag_or_id: list_of_choices}
        #     global combobox_choice_map
        #     if combobox_choice_map is None:  # build on first use
        #         combobox_choice_map = {gui_instance.emotion_choice: (gui_instance.emotion_names, gui_instance.on_send_emotion),
        #                                gui_instance.voice_choice: (gui_instance.voice_names, None)}
        #     def browse(choice_widget, data):
        #         choices, callback = data
        #         index = choices.index(dpg.get_value(choice_widget))
        #         if key == dpg.mvKey_Down:
        #             new_index = min(index + 1, len(choices) - 1)
        #         elif key == dpg.mvKey_Up:
        #             new_index = max(index - 1, 0)
        #         elif key == dpg.mvKey_Home:
        #             new_index = 0
        #         elif key == dpg.mvKey_End:
        #             new_index = len(choices) - 1
        #         else:
        #             new_index = None
        #         if new_index is not None:
        #             dpg.set_value(choice_widget, choices[new_index])
        #             if callback is not None:
        #                 callback(sender, app_data)  # the callback doesn't trigger automatically if we programmatically set the combobox value
        #     focused_item = dpg.get_focused_item()
        #     focused_item = dpg.get_item_alias(focused_item)
        #     if focused_item in combobox_choice_map.keys():
        #         browse(focused_item, combobox_choice_map[focused_item])
with dpg.handler_registry(tag="librarian_handler_registry"):  # global (whole viewport)
    dpg.add_key_press_handler(tag="librarian_hotkeys_handler", callback=librarian_hotkeys_callback)

# --------------------------------------------------------------------------------
# Start the app

logger.info("App bootup...")

avatar_instance_id = api.avatar_load(librarian_config.avatar_config.image_path)
api.avatar_load_emotion_templates(avatar_instance_id, {})  # send empty dict -> reset emotion templates to server defaults
avatar_controller = DPGAvatarController(stop_tts_button_gui_widget="chat_stop_speech_button",  # tag
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
avatar_record = avatar_controller.register_avatar_instance(avatar_instance_id=avatar_instance_id,
                                                           voice=librarian_config.avatar_config.voice,
                                                           voice_speed=librarian_config.avatar_config.voice_speed,
                                                           emotion_autoreset_interval=librarian_config.avatar_config.emotion_autoreset_interval,
                                                           emotion_blacklist=librarian_config.avatar_config.emotion_blacklist,
                                                           data_eyes_fadeout_duration=librarian_config.avatar_config.data_eyes_fadeout_duration)
api.tts_warmup(voice=librarian_config.avatar_config.voice)

chat_controller = DPGChatController(llm_settings=llm_settings,
                                    datastore=datastore,
                                    retriever=retriever,
                                    app_state=app_state,
                                    avatar_image_path=librarian_config.avatar_config.image_path,
                                    avatar_controller=avatar_controller,
                                    avatar_record=avatar_record,
                                    themes_and_fonts=themes_and_fonts,
                                    chat_panel_widget=chat_panel_widget,
                                    chat_stop_generation_button_widget=stop_generation_button,
                                    indicator_glow_animation=pulsating_gray_text_glow,
                                    llm_indicator_widget=llm_indicator_group,
                                    docs_indicator_widget=docs_indicator_group,
                                    web_indicator_widget=web_indicator_group,
                                    executor=bg)

def gui_shutdown() -> None:
    """App exit: gracefully shut down parts that access DPG."""
    avatar_controller.stop_tts()  # Stop the TTS speaking so that the speech background thread (if any) exits.
    logger.info("gui_shutdown: entered")
    gui_resize_task_manager.clear(wait=True)
    task_manager.clear(wait=True)
    chat_controller.shutdown()
    avatar_controller.shutdown()
    dpg_avatar_renderer.stop()
    gui_animation.animator.clear()
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
_animator_settings = None
def _load_initial_animator_settings() -> None:
    global _animator_settings

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
    api.avatar_start(avatar_instance_id)
    dpg_avatar_renderer.start(avatar_instance_id)
    dpg_avatar_renderer.load_backdrop_image(animator_settings["backdrop_path"])
    dpg_avatar_renderer.configure_backdrop(new_width=avatar_panel_w - 16,
                                           new_height=avatar_panel_h - 16,
                                           new_blur_state=animator_settings["backdrop_blur"])
    _animator_settings = animator_settings  # for access from GUI event handlers

dpg.set_frame_callback(2, _load_initial_animator_settings)

def _build_initial_chat_view(sender, app_data) -> None:
    chat_controller.view.build()
dpg.set_frame_callback(3, _build_initial_chat_view)

logger.info("App render loop starting.")

try:
    # We control the render loop manually to have a convenient place to update our GUI animations just before rendering each frame.
    while dpg.is_dearpygui_running():
        update_animations()
        dpg.render_dearpygui_frame()
    # dpg.start_dearpygui()  # automatic render loop
except KeyboardInterrupt:
    pass  # cleanup will be handled by our DPG exit handler

logger.info("App render loop exited.")

dpg.destroy_context()

def main() -> None:  # TODO: we don't really need this; it's just for console_scripts.
    pass
