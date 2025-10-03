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
    from typing import Optional

    # WORKAROUND: Deleting a texture or image widget causes DPG to segfault on Nvidia/Linux.
    # https://github.com/hoffstadt/DearPyGui/issues/554
    if platform.system().upper() == "LINUX":
        os.environ["__GLVND_DISALLOW_PATCHING"] = "1"

    import dearpygui.dearpygui as dpg

    from mcpyrate import colorizer

    # Vendored libraries
    from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa  # https://github.com/juliettef/IconFontCppHeaders
    # from ..vendor.file_dialog.fdialog import FileDialog  # https://github.com/totallynotdrait/file_dialog, but with custom modifications

    from ..client import api  # Raven-server support
    from ..client.avatar_controller import DPGAvatarController
    from ..client.avatar_renderer import DPGAvatarRenderer
    from ..client import config as client_config

    from ..common import bgtask

    from ..common.gui import animation as gui_animation
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
task_manager = bgtask.TaskManager(name="librarian",  # for most tasks
                                  mode="concurrent",
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
# Set up the main window

logger.info("Initial GUI setup...")
with timer() as tim:
    with dpg.window(show=True, modal=False, no_title_bar=False, tag="summarizer_window",
                    label="Raven-librarian main window",
                    no_scrollbar=True, autosize=True) as main_window:  # DPG "window" inside the app OS window ("viewport"), container for the whole GUI
        with dpg.group(horizontal=True):
            with dpg.group():  # left column: linearized chat view
                # The `DPGChatController` goes into this panel when the app boots up.
                chat_panel_h = gui_config.main_window_h - (gui_config.ai_warning_h + 16) - (gui_config.chat_controls_h + 16) + 8
                chat_panel_widget = dpg.add_child_window(tag="chat_panel",
                                                         width=(gui_config.chat_panel_w + 16),  # 16 = round border (8 on each side)
                                                         height=chat_panel_h)

                with dpg.child_window(tag="chat_controls",
                                      width=(gui_config.chat_panel_w + 16),  # 16 = round border (8 on each side)
                                      height=gui_config.chat_controls_h,
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
                                           width=gui_config.chat_panel_w - gui_config.toolbutton_w - 8)
                        dpg.add_button(label=fa.ICON_PAPER_PLANE,
                                       callback=send_message_to_ai_callback,
                                       width=gui_config.toolbutton_w,
                                       tag="chat_send_button")
                        dpg.bind_item_font("chat_send_button", themes_and_fonts.icon_font_solid)  # tag  # TODO: make this change into a cancel button while the LLM is writing.
                        dpg.bind_item_theme("chat_send_button", "disablable_button_theme")  # tag
                        with dpg.tooltip("chat_send_button"):  # tag
                            dpg.add_text("Send to AI [Enter]")

            with dpg.group():  # right column: AI avatar
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

        with dpg.child_window(tag="chat_ai_warning",
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
                                                                         original_theme=themes_and_fonts.global_theme,
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
                                                                         original_theme=themes_and_fonts.global_theme,
                                                                         duration=gui_config.acknowledgment_duration))

                def stop_text_generation_callback() -> None:
                    chat_controller.stop_ai_turn()
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
                    # resize_gui()  # TODO: resizable GUI

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
                stop_text_generation_callback()

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
                                                           emotion_autoreset_interval=librarian_config.avatar_config.emotion_autoreset_interval,
                                                           emotion_blacklist=librarian_config.avatar_config.emotion_blacklist,
                                                           data_eyes_fadeout_duration=librarian_config.avatar_config.data_eyes_fadeout_duration)
chat_controller = DPGChatController(llm_settings=llm_settings,
                                    datastore=datastore,
                                    retriever=retriever,
                                    app_state=app_state,
                                    avatar_controller=avatar_controller,
                                    avatar_record=avatar_record,
                                    themes_and_fonts=themes_and_fonts,
                                    chat_panel_widget=chat_panel_widget,
                                    indicator_glow_animation=pulsating_gray_text_glow,
                                    llm_indicator_widget=llm_indicator_group,
                                    docs_indicator_widget=docs_indicator_group,
                                    web_indicator_widget=web_indicator_group,
                                    executor=bg)

def gui_shutdown() -> None:
    """App exit: gracefully shut down parts that access DPG."""
    avatar_controller.stop_tts()  # Stop the TTS speaking so that the speech background thread (if any) exits.
    logger.info("gui_shutdown: entered")
    task_manager.clear(wait=True)  # Wait until background tasks actually exit.
    chat_controller.shutdown()
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
    clear_background_tasks(wait=False)  # signal background tasks to exit

logger.info("App render loop exited.")

dpg.destroy_context()

def main() -> None:  # TODO: we don't really need this; it's just for console_scripts.
    pass
