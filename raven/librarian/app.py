#!/usr/bin/env python
"""GUI LLM client with auto-persisted branching chat history and RAG (retrieval-augmented generation; query your plain-text documents)."""

import logging
logging.basicConfig(level=logging.DEBUG)
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

    # WORKAROUND: Deleting a texture or image widget causes DPG to segfault on Nvidia/Linux.
    # https://github.com/hoffstadt/DearPyGui/issues/554
    if platform.system().upper() == "LINUX":
        os.environ["__GLVND_DISALLOW_PATCHING"] = "1"

    import dearpygui.dearpygui as dpg

    # Vendored libraries
    from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa  # https://github.com/juliettef/IconFontCppHeaders
    # from ..vendor import DearPyGui_Markdown as dpg_markdown  # https://github.com/IvanNazaruk/DearPyGui-Markdown
    # from ..vendor.file_dialog.fdialog import FileDialog  # https://github.com/totallynotdrait/file_dialog, but with custom modifications

    from ..common import bgtask

    from ..common.gui import animation as gui_animation
    from ..common.gui import utils as guiutils

    from . import config as librarian_config

    gui_config = librarian_config.gui_config  # shorthand, this is used a lot
logger.info(f"    Done in {tim.dt:0.6g}s.")

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
                margin = 8
                icon_size = 32
                initial_message_container_height = 2 * margin + icon_size
                before_buttons_spacing = 1
                message_spacing = 8
                color_ai_front = (80, 80, 83)
                color_ai_back = (45, 45, 48)
                color_user_front = (70, 70, 90)
                color_user_back = (40, 40, 50)

                def make_ai_message_buttons():
                    with dpg.group(horizontal=True):
                        dpg.add_text("[0 t, 0 s, ∞ t/s]", color=(180, 180, 180), tag="performance_stats_text_ai")

                        dpg.add_spacer(tag="ai_message_buttons_spacer")

                        dpg.add_button(label=fa.ICON_RECYCLE,
                                       callback=lambda: None,  # TODO
                                       enabled=False,
                                       width=gui_config.toolbutton_w,
                                       tag="chat_reroll_button")
                        dpg.bind_item_font("chat_reroll_button", themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("chat_reroll_button", "disablable_button_theme")  # tag
                        with dpg.tooltip("chat_reroll_button"):  # tag
                            dpg.add_text("Regenerate (replace branch)")

                        dpg.add_button(label=fa.ICON_CODE_BRANCH,
                                       callback=lambda: None,  # TODO
                                       enabled=False,
                                       width=gui_config.toolbutton_w,
                                       tag="chat_new_branch_button")
                        dpg.bind_item_font("chat_new_branch_button", themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("chat_new_branch_button", "disablable_button_theme")  # tag
                        with dpg.tooltip("chat_new_branch_button"):  # tag
                            dpg.add_text("Stash and regenerate (new branch)")

                        dpg.add_button(label=fa.ICON_TRASH_CAN,
                                       callback=lambda: None,  # TODO
                                       enabled=False,
                                       width=gui_config.toolbutton_w,
                                       tag="chat_delete_branch_button")
                        dpg.bind_item_font("chat_delete_branch_button", themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("chat_delete_branch_button", "disablable_button_theme")  # tag
                        with dpg.tooltip("chat_delete_branch_button"):  # tag
                            dpg.add_text("Delete current branch")

                        dpg.add_button(label=fa.ICON_ANGLE_LEFT,
                                       callback=lambda: None,  # TODO
                                       enabled=False,
                                       width=gui_config.toolbutton_w,
                                       tag="chat_prevbranch_button")
                        dpg.bind_item_font("chat_prevbranch_button", themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("chat_prevbranch_button", "disablable_button_theme")  # tag
                        with dpg.tooltip("chat_prevbranch_button"):  # tag
                            dpg.add_text("Previous branch")

                        dpg.add_button(label=fa.ICON_ANGLE_RIGHT,
                                       callback=lambda: None,  # TODO
                                       enabled=False,
                                       width=gui_config.toolbutton_w,
                                       tag="chat_nextbranch_button")
                        dpg.bind_item_font("chat_nextbranch_button", themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("chat_nextbranch_button", "disablable_button_theme")  # tag
                        with dpg.tooltip("chat_nextbranch_button"):  # tag
                            dpg.add_text("Next branch")

                def make_user_message_buttons():
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=800 - 5 * (gui_config.toolbutton_w + 8), tag="user_message_buttons_spacer")  # 8 = DPG outer margin

                        dpg.add_button(label=fa.ICON_PENCIL,
                                       callback=lambda: None,  # TODO
                                       width=gui_config.toolbutton_w,
                                       tag="chat_edit_button")
                        dpg.bind_item_font("chat_edit_button", themes_and_fonts.icon_font_solid)  # tag
                        with dpg.tooltip("chat_edit_button"):  # tag
                            dpg.add_text("Edit (replace)")

                        dpg.add_button(label=fa.ICON_CODE_BRANCH,
                                       callback=lambda: None,  # TODO
                                       width=gui_config.toolbutton_w,
                                       tag="chat_new_branch_button_2")
                        dpg.bind_item_font("chat_new_branch_button_2", themes_and_fonts.icon_font_solid)  # tag
                        with dpg.tooltip("chat_new_branch_button_2"):  # tag
                            dpg.add_text("Stash and clear (new branch)")

                        dpg.add_button(label=fa.ICON_TRASH_CAN,
                                       callback=lambda: None,  # TODO
                                       width=gui_config.toolbutton_w,
                                       tag="chat_delete_branch_button_2")
                        dpg.bind_item_font("chat_delete_branch_button_2", themes_and_fonts.icon_font_solid)  # tag
                        with dpg.tooltip("chat_delete_branch_button_2"):  # tag
                            dpg.add_text("Delete current branch")

                        dpg.add_button(label=fa.ICON_ANGLE_LEFT,
                                       callback=lambda: None,  # TODO
                                       enabled=False,
                                       width=gui_config.toolbutton_w,
                                       tag="chat_prevbranch_button_2")
                        dpg.bind_item_font("chat_prevbranch_button_2", themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("chat_prevbranch_button_2", "disablable_button_theme")  # tag
                        with dpg.tooltip("chat_prevbranch_button_2"):  # tag
                            dpg.add_text("Previous branch")

                        dpg.add_button(label=fa.ICON_ANGLE_RIGHT,
                                       callback=lambda: None,  # TODO
                                       enabled=False,
                                       width=gui_config.toolbutton_w,
                                       tag="chat_nextbranch_button_2")
                        dpg.bind_item_font("chat_nextbranch_button_2", themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("chat_nextbranch_button_2", "disablable_button_theme")  # tag
                        with dpg.tooltip("chat_nextbranch_button_2"):  # tag
                            dpg.add_text("Next branch")

                # We need to draw text using a text widget, not `draw_text`, so that we can use Markdown.
                # But we want a visual frame, which needs a drawlist. The chat icon can also go into this drawlist.
                # To draw the text on top of the drawlist, we add the drawlist first (so it will be below the text in z-order),
                # and then, while adding the text widget, manually set the position (in child-window coordinates).
                with dpg.drawlist(width=800, height=initial_message_container_height, tag="chat_text_drawlist_ai"):
                    dpg.draw_rectangle((0, 0), (800, initial_message_container_height), color=color_ai_front, fill=color_ai_back, rounding=8)
                    dpg.draw_image("icon_ai_texture", (margin, margin), (margin + icon_size, margin + icon_size), uv_min=(0, 0), uv_max=(1, 1))
                dpg.add_spacer(height=before_buttons_spacing)
                make_ai_message_buttons()
                with dpg.group(horizontal=True):
                    dpg.add_spacer(tag="branch_count_spacer_ai")
                    dpg.add_text("1/1", color=(180, 180, 180), tag="branch_count_text_ai")
                    with dpg.tooltip("branch_count_text_ai"):  # tag
                        dpg.add_text("Current branch, number of branches at this point")
                dpg.add_spacer(height=message_spacing)

                with dpg.drawlist(width=800, height=initial_message_container_height, tag="chat_text_drawlist_user"):
                    dpg.draw_rectangle((0, 0), (800, initial_message_container_height), color=color_user_front, fill=color_user_back, rounding=8)
                    dpg.draw_image("icon_user_texture", (margin, margin), (margin + icon_size, margin + icon_size), uv_min=(0, 0), uv_max=(1, 1))
                dpg.add_spacer(height=before_buttons_spacing)
                make_user_message_buttons()
                with dpg.group(horizontal=True):
                    dpg.add_spacer(tag="branch_count_spacer_user")
                    dpg.add_text("1/1", color=(180, 180, 180), tag="branch_count_text_user")
                    with dpg.tooltip("branch_count_text_user"):  # tag
                        dpg.add_text("Current branch, number of branches at this point")
                dpg.add_spacer(height=message_spacing)

            # We must wait for the drawlists to get a position before we can overlay a text widget on them.
            def add_chat_texts():
                # Align branch counts to the right
                w_header, h_header = dpg.get_item_rect_size("branch_count_text_ai")
                dpg.set_item_width("branch_count_spacer_ai", 800 - (w_header + 8))

                w_header, h_header = dpg.get_item_rect_size("branch_count_text_user")
                dpg.set_item_width("branch_count_spacer_user", 800 - (w_header + 8))

                w_header, h_header = dpg.get_item_rect_size("performance_stats_text_ai")
                dpg.set_item_width("ai_message_buttons_spacer", 800 - 5 * (gui_config.toolbutton_w + 8) - (w_header + 8))

                # Write the "chat messages" for the mockup
                x0_local, y0_local = guiutils.get_widget_relative_pos("chat_text_drawlist_ai", reference="chat_panel")  # tag
                dpg.add_text("Hello! I'll be your AI summarizer. To begin, select item(s) and click Summarize.",
                             pos=(x0_local + 8 + 3 + margin + icon_size, y0_local + 3 + icon_size // 2 - (gui_config.font_size // 2)),  # 8 = extra spacing; 3 = DPG inner margin
                             color=(255, 255, 255), tag="chat_test_text_ai", parent="chat_group")

                x0_local, y0_local = guiutils.get_widget_relative_pos("chat_text_drawlist_user", reference="chat_panel")  # tag
                dpg.add_text("That's great. Testing 1 2 3?",
                             pos=(x0_local + 8 + 3 + margin + icon_size, y0_local + 3 + icon_size // 2 - (gui_config.font_size // 2)),  # 8 = extra spacing; 3 = DPG inner margin
                             color=(255, 255, 255), tag="chat_test_text_user", parent="chat_group")
            dpg.set_frame_callback(11, add_chat_texts)

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
