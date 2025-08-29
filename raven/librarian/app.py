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
    import concurrent.futures
    import os
    import pathlib
    import platform
    import requests
    import sys
    from typing import Dict, Optional, Union
    import uuid

    # WORKAROUND: Deleting a texture or image widget causes DPG to segfault on Nvidia/Linux.
    # https://github.com/hoffstadt/DearPyGui/issues/554
    if platform.system().upper() == "LINUX":
        os.environ["__GLVND_DISALLOW_PATCHING"] = "1"

    import dearpygui.dearpygui as dpg

    from mcpyrate import colorizer

    # Vendored libraries
    from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa  # https://github.com/juliettef/IconFontCppHeaders
    from ..vendor import DearPyGui_Markdown as dpg_markdown  # https://github.com/IvanNazaruk/DearPyGui-Markdown
    # from ..vendor.file_dialog.fdialog import FileDialog  # https://github.com/totallynotdrait/file_dialog, but with custom modifications

    from ..common import bgtask

    from ..common.gui import animation as gui_animation
    from ..common.gui import utils as guiutils

    from . import appstate
    from . import config as librarian_config
    # from . import chattree
    from . import hybridir
    from . import llmclient

    gui_config = librarian_config.gui_config  # shorthand, this is used a lot
logger.info(f"Libraries loaded in {tim.dt:0.6g}s.")

# --------------------------------------------------------------------------------
# Set up DPG - basic startup, load fonts, set up global theme

# We do this as early as possible, because before the startup is complete, trying to `dpg.add_xxx` or `with dpg.xxx:` anything will segfault the app.

logger.info("DPG bootup...")
with timer() as tim:
    dpg.create_context()

    themes_and_fonts = guiutils.bootup(font_size=gui_config.font_size)

    # Initialize textures.
    with dpg.texture_registry(tag="app_textures"):
        w, h, c, data = dpg.load_image(str(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "icons", "ai.png")).expanduser().resolve()))
        icon_ai_texture = dpg.add_static_texture(w, h, data, tag="icon_ai_texture")

        w, h, c, data = dpg.load_image(str(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "icons", "user.png")).expanduser().resolve()))
        icon_user_texture = dpg.add_static_texture(w, h, data, tag="icon_user_texture")

    dpg.create_viewport(title=f"Raven-librarian {__version__}",
                        width=gui_config.main_window_w,
                        height=gui_config.main_window_h)  # OS window (DPG "viewport")
    dpg.setup_dearpygui()
logger.info(f"    Done in {tim.dt:0.6g}s.")

# --------------------------------------------------------------------------------
# Connect to LLM backend, load datastores

backend_url = librarian_config.llm_backend_url

# API key already loaded during module bootup; here, we just inform the user.
if "Authorization" in llmclient.headers:
    print()
    print(f"Loaded LLM API key from '{str(librarian_config.llm_api_key_file)}'.")
    print()
else:
    print()
    print(f"No LLM API key configured. If your LLM needs an API key to connect, put it into '{str(librarian_config.llm_api_key_file)}'.")
    print("This can be any plain-text data your LLM's API accepts in the 'Authorization' field of the HTTP headers.")
    print("For username/password, the format is 'user pass'. Do NOT use a plaintext password over an unencrypted http:// connection!")
    print()

try:
    llmclient.list_models(backend_url)  # just do something, to try to connect
except requests.exceptions.ConnectionError as exc:
    print(colorizer.colorize(f"Cannot connect to LLM backend at {backend_url}.", colorizer.Style.BRIGHT, colorizer.Fore.RED) + " Is the LLM server running?")
    msg = f"Failed to connect to LLM backend at {backend_url}, reason {type(exc)}: {exc}"
    logger.error(msg)
    sys.exit(255)
else:
    print(colorizer.colorize(f"Connected to LLM backend at {backend_url}", colorizer.Style.BRIGHT, colorizer.Fore.GREEN))
    settings = llmclient.setup(backend_url=backend_url)
    print()

logger.info("Loading chat datastore.")
with timer() as tim:
    datastore_file = librarian_config.llmclient_userdata_dir / "data.json"  # chat node datastore
    state_file = librarian_config.llmclient_userdata_dir / "state.json"     # important node IDs for the chat client state

    # Persistent, branching chat history, and app settings (these will auto-persist at app exit).
    datastore, app_state = appstate.load(settings, datastore_file, state_file)
logger.info(f"Datastore loaded in {tim.dt:0.6g}s.")

logger.info("Loading RAG (retrieval-augmented generation) document store.")
with timer() as tim:
    docs_dir = pathlib.Path(librarian_config.llm_docs_dir).expanduser().resolve()  # RAG documents (put your documents in this directory)
    db_dir = pathlib.Path(librarian_config.llm_database_dir).expanduser().resolve()  # RAG search indices datastore

    # Load RAG database (it will auto-persist at app exit).
    retriever, _unused_scanner = hybridir.setup(docs_dir=docs_dir,
                                                recursive=librarian_config.llm_docs_dir_recursive,
                                                db_dir=db_dir,
                                                embedding_model_name=librarian_config.qa_embedding_model)

    logger.info(f"RAG document store is at '{str(librarian_config.llm_docs_dir)}' (put your plain-text documents here).")
    # The retriever's `documents` attribute must be locked before accessing.
    with retriever.datastore_lock:
        plural_s = "s" if len(retriever.documents) != 1 else ""
        logger.info(f"RAG: {len(retriever.documents)} document{plural_s} loaded.")
    logger.info(f"RAG: Search indices are saved in '{str(librarian_config.llm_database_dir)}'.")
logger.info(f"RAG document store loaded in {tim.dt:0.6g}s.")


# --------------------------------------------------------------------------------
# Linear chat view (of current branch)

def make_ai_message_buttons(gui_parent: Union[int, str], uuid: str):
    g = dpg.add_group(horizontal=True, tag=f"ai_message_group_{uuid}", parent=gui_parent)

    # dpg.add_text("[0 t, 0 s, ∞ t/s]", color=(180, 180, 180), tag=f"performance_stats_text_ai_{uuid}", parent=g)  # TODO: add the performance stats

    # dpg.add_spacer(tag=f"ai_message_buttons_spacer_{uuid}",
    #                parent=g)

    dpg.add_button(label=fa.ICON_RECYCLE,
                   callback=lambda: None,  # TODO
                   enabled=False,
                   width=gui_config.toolbutton_w,
                   tag=f"chat_reroll_button_{uuid}",
                   parent=g)
    dpg.bind_item_font(f"chat_reroll_button_{uuid}", themes_and_fonts.icon_font_solid)  # tag
    dpg.bind_item_theme(f"chat_reroll_button_{uuid}", "disablable_button_theme")  # tag
    reroll_tooltip = dpg.add_tooltip(f"chat_reroll_button_{uuid}")  # tag
    dpg.add_text("Reroll (create new sibling)", parent=reroll_tooltip)

    dpg.add_button(label=fa.ICON_CODE_BRANCH,
                   callback=lambda: None,  # TODO
                   enabled=False,
                   width=gui_config.toolbutton_w,
                   tag=f"chat_new_branch_button_{uuid}",
                   parent=g)
    dpg.bind_item_font(f"chat_new_branch_button_{uuid}", themes_and_fonts.icon_font_solid)  # tag
    dpg.bind_item_theme(f"chat_new_branch_button_{uuid}", "disablable_button_theme")  # tag
    new_branch_tooltip = dpg.add_tooltip(f"chat_new_branch_button_{uuid}")  # tag
    dpg.add_text("Branch from here", parent=new_branch_tooltip)

    dpg.add_button(label=fa.ICON_TRASH_CAN,
                   callback=lambda: None,  # TODO
                   enabled=False,
                   width=gui_config.toolbutton_w,
                   tag=f"chat_delete_branch_button_{uuid}",
                   parent=g)
    dpg.bind_item_font(f"chat_delete_branch_button_{uuid}", themes_and_fonts.icon_font_solid)  # tag
    dpg.bind_item_theme(f"chat_delete_branch_button_{uuid}", "disablable_button_theme")  # tag
    delete_branch_tooltip = dpg.add_tooltip(f"chat_delete_branch_button_{uuid}")  # tag
    dpg.add_text("Delete current branch", parent=delete_branch_tooltip)

    dpg.add_button(label=fa.ICON_ANGLE_LEFT,
                   callback=lambda: None,  # TODO
                   enabled=False,
                   width=gui_config.toolbutton_w,
                   tag=f"chat_prevbranch_button_{uuid}",
                   parent=g)
    dpg.bind_item_font(f"chat_prevbranch_button_{uuid}", themes_and_fonts.icon_font_solid)  # tag
    dpg.bind_item_theme(f"chat_prevbranch_button_{uuid}", "disablable_button_theme")  # tag
    prevbranch_tooltip = dpg.add_tooltip(f"chat_prevbranch_button_{uuid}")  # tag
    dpg.add_text("Switch to previous sibling", parent=prevbranch_tooltip)

    dpg.add_button(label=fa.ICON_ANGLE_RIGHT,
                   callback=lambda: None,  # TODO
                   enabled=False,
                   width=gui_config.toolbutton_w,
                   tag=f"chat_nextbranch_button_{uuid}",
                   parent=g)
    dpg.bind_item_font(f"chat_nextbranch_button_{uuid}", themes_and_fonts.icon_font_solid)  # tag
    dpg.bind_item_theme(f"chat_nextbranch_button_{uuid}", "disablable_button_theme")  # tag
    nextbranch_tooltip = dpg.add_tooltip(f"chat_nextbranch_button_{uuid}")  # tag
    dpg.add_text("Switch to next sibling", parent=nextbranch_tooltip)

def make_user_message_buttons(gui_parent: Union[int, str], uuid: str):
    g = dpg.add_group(horizontal=True, tag=f"user_message_group_{uuid}", parent=gui_parent)

    # dpg.add_spacer(width=800 - 5 * (gui_config.toolbutton_w + 8),  # 8 = DPG outer margin
    #                tag=f"user_message_buttons_spacer_{uuid}",
    #                parent=g)

    dpg.add_button(label=fa.ICON_PENCIL,
                   callback=lambda: None,  # TODO
                   width=gui_config.toolbutton_w,
                   tag=f"chat_edit_button_{uuid}",
                   parent=g)
    dpg.bind_item_font(f"chat_edit_button_{uuid}", themes_and_fonts.icon_font_solid)  # tag
    edit_tooltip = dpg.add_tooltip(f"chat_edit_button_{uuid}")  # tag
    dpg.add_text("Edit (revise)", parent=edit_tooltip)

    dpg.add_button(label=fa.ICON_CODE_BRANCH,
                   callback=lambda: None,  # TODO
                   width=gui_config.toolbutton_w,
                   tag=f"chat_new_branch_button_{uuid}",
                   parent=g)
    dpg.bind_item_font(f"chat_new_branch_button_{uuid}", themes_and_fonts.icon_font_solid)  # tag
    new_branch_tooltip = dpg.add_tooltip(f"chat_new_branch_button_{uuid}")  # tag
    dpg.add_text("Branch from here", parent=new_branch_tooltip)

    dpg.add_button(label=fa.ICON_TRASH_CAN,
                   callback=lambda: None,  # TODO
                   width=gui_config.toolbutton_w,
                   tag=f"chat_delete_branch_button_{uuid}",
                   parent=g)
    dpg.bind_item_font(f"chat_delete_branch_button_{uuid}", themes_and_fonts.icon_font_solid)  # tag
    delete_branch_tooltip = dpg.add_tooltip(f"chat_delete_branch_button_{uuid}")  # tag
    dpg.add_text("Delete current branch", parent=delete_branch_tooltip)

    dpg.add_button(label=fa.ICON_ANGLE_LEFT,
                   callback=lambda: None,  # TODO
                   enabled=False,
                   width=gui_config.toolbutton_w,
                   tag=f"chat_prevbranch_button_{uuid}",
                   parent=g)
    dpg.bind_item_font(f"chat_prevbranch_button_{uuid}", themes_and_fonts.icon_font_solid)  # tag
    dpg.bind_item_theme(f"chat_prevbranch_button_{uuid}", "disablable_button_theme")  # tag
    prevbranch_tooltip = dpg.add_tooltip(f"chat_prevbranch_button_{uuid}")  # tag
    dpg.add_text("Switch to previous sibling", parent=prevbranch_tooltip)

    dpg.add_button(label=fa.ICON_ANGLE_RIGHT,
                   callback=lambda: None,  # TODO
                   enabled=False,
                   width=gui_config.toolbutton_w,
                   tag=f"chat_nextbranch_button_{uuid}",
                   parent=g)
    dpg.bind_item_font(f"chat_nextbranch_button_{uuid}", themes_and_fonts.icon_font_solid)  # tag
    dpg.bind_item_theme(f"chat_nextbranch_button_{uuid}", "disablable_button_theme")  # tag
    nextbranch_tooltip = dpg.add_tooltip(f"chat_nextbranch_button_{uuid}")  # tag
    dpg.add_text("Switch to next sibling", parent=nextbranch_tooltip)

class DisplayedChatMessage:
    def __init__(self,
                 gui_parent: Union[int, str],
                 node_payload: Dict):
        self.gui_parent = gui_parent  # GUI container to render in (DPG ID or tag)
        self.node_payload = node_payload  # reference to the chat node data (ORIGINAL, not a copy)
        self.gui_uuid = str(uuid.uuid4())  # used in GUI widget tags
        self.gui_container_group = dpg.add_group(tag=f"chat_item_container_group_{self.gui_uuid}",
                                                 parent=self.gui_parent)
        self.build()

    def build(self) -> None:
        dpg.delete_item(self.gui_container_group, children_only=True)  # clear old GUI content if any

        # lay out the role icon and the text content horizontally
        icon_and_text_container_group = dpg.add_group(horizontal=True,
                                                      tag=f"chat_icon_and_text_container_group_{self.gui_uuid}",
                                                      parent=self.gui_container_group)

        # ----------------------------------------
        # role icon

        # TODO: add icons for system and tool; improve user/AI icons
        message = self.node_payload["message"]
        message_role = message["role"]
        message_text = message["content"]
        gui_role_icons = {"assistant": "icon_ai_texture",
                          "user": "icon_user_texture"}
        dpg.add_drawlist(width=(2 * gui_config.margin + gui_config.chat_icon_size),
                         height=(2 * gui_config.margin + gui_config.chat_icon_size),
                         tag=f"chat_icon_drawlist_{self.gui_uuid}",
                         parent=icon_and_text_container_group)  # empty drawlist acts as placeholder if no icon
        if message_role in gui_role_icons:
            dpg.draw_image(gui_role_icons[message_role],
                           (gui_config.margin, gui_config.margin),
                           (gui_config.margin + gui_config.chat_icon_size, gui_config.margin + gui_config.chat_icon_size),
                           uv_min=(0, 0),
                           uv_max=(1, 1),
                           parent=f"chat_icon_drawlist_{self.gui_uuid}")

        # ----------------------------------------
        # text content

        # adjust text vertical positioning
        text_vertical_layout_group = dpg.add_group(tag=f"chat_text_vertical_layout_group_{self.gui_uuid}",
                                                   parent=icon_and_text_container_group)
        dpg.add_spacer(height=gui_config.margin,
                       parent=text_vertical_layout_group)

        # render the actual text
        colorized_message_text = f"<font color='#ffffff'>{message_text}</font>"
        chat_message_widget = dpg_markdown.add_text(colorized_message_text,
                                                    wrap=700,  # TODO: figure out the correct wrap width
                                                    parent=text_vertical_layout_group)
        dpg.set_item_alias(chat_message_widget, f"chat_message_{message_role}_{self.gui_uuid}")

        # ----------------------------------------
        # buttons

        dpg.add_spacer(height=2,
                       parent=text_vertical_layout_group)

        if message_role == "user":
            make_user_message_buttons(gui_parent=text_vertical_layout_group,
                                      uuid=self.gui_uuid)
        elif message_role == "assistant":
            make_ai_message_buttons(gui_parent=text_vertical_layout_group,
                                    uuid=self.gui_uuid)

def build_linearized_chat(head_node_id: Optional[str] = None) -> None:
    if head_node_id is None:  # use current HEAD from app_state?
        head_node_id = app_state["HEAD"]
    payload_history = datastore.linearize_up(head_node_id)
    for payload in payload_history:
        DisplayedChatMessage(gui_parent="chat_group",
                             node_payload=payload)

# --------------------------------------------------------------------------------
# Set up the main window

logger.info("Initial GUI setup...")
with timer() as tim:
    # TODO: GUI for AI summaries
    # TODO: hotkeys
    # TODO: separate hotkey mode while `chat_field` is focused
    with dpg.window(show=True, modal=False, no_title_bar=False, tag="summarizer_window",
                    label="Raven-librarian main window",
                    no_scrollbar=True, autosize=True) as main_window:  # DPG "window" inside the app OS window ("viewport"), container for the whole GUI
        with dpg.child_window(tag="chat_ai_warning",
                              height=42,
                              no_scrollbar=True,
                              no_scroll_with_mouse=True):
            with dpg.group(horizontal=True):
                dpg.add_text(fa.ICON_TRIANGLE_EXCLAMATION, color=(255, 180, 120), tag="ai_warning_icon")  # orange
                dpg.add_text("Response quality and factual accuracy ultimately depend on the AI. Check important facts independently.", color=(255, 180, 120), tag="ai_warning_text")  # orange
            dpg.bind_item_font("ai_warning_icon", themes_and_fonts.icon_font_solid)  # tag

        with dpg.child_window(tag="chat_panel",
                              width=816,  # 800 + round border (8 on each side)
                              height=600):
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
                              width=816,
                              height=42,
                              no_scrollbar=True,
                              no_scroll_with_mouse=True):
            with dpg.group(horizontal=True):
                dpg.add_input_text(tag="chat_field",
                                   default_value="",
                                   hint="[ask the AI questions here]",
                                   width=800 - gui_config.toolbutton_w - 8,
                                   callback=lambda: None)  # TODO
                dpg.add_button(label=fa.ICON_PAPER_PLANE,
                               callback=lambda: None,  # TODO
                               width=gui_config.toolbutton_w,
                               tag="chat_send_button")
                dpg.bind_item_font("chat_send_button", themes_and_fonts.icon_font_solid)  # tag  # TODO: make this change into a cancel button while the LLM is writing.
                with dpg.tooltip("chat_send_button"):  # tag
                    dpg.add_text("Send to AI")

# --------------------------------------------------------------------------------
# Animations, live updates

def update_animations():
    gui_animation.animator.render_frame()

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

bg = concurrent.futures.ThreadPoolExecutor()  # for info panel and tooltip annotation updates
task_manager = bgtask.TaskManager(name="annotation_update",
                                  mode="concurrent",
                                  executor=bg)

dpg.set_primary_window(main_window, True)  # Make this DPG "window" occupy the whole OS window (DPG "viewport").
dpg.set_viewport_vsync(True)
dpg.show_viewport()

def build_initial_chat_callback(sender, app_data):
    build_linearized_chat()
dpg.set_frame_callback(11, build_initial_chat_callback)

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

def main():  # TODO: we don't really need this; it's just for console_scripts.
    pass
