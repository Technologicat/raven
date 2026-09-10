#!/usr/bin/env python
"""GUI LLM client with auto-persisted branching chat history and RAG (retrieval-augmented generation; query your plain-text documents)."""

import argparse

from .. import __version__
from .. import avatar  # for `avatar.assets_path`
from ..common import replserver

parser = argparse.ArgumentParser(description="""GUI LLM client with auto-persisted branching chat history and RAG.""",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
parser.add_argument('--log', metavar='PATH', default=None,
                    help='mirror stderr log to this file (overwritten each run)')
parser.add_argument('--log-level', default='INFO',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    help='root logger level (default: INFO)')
parser.add_argument('--backend-url', metavar='URL', default=None,
                    help='LLM backend to talk to, overriding the configured one; e.g. http://localhost:1234. '
                         'Point it at nothing listening to see what the app does with no backend.')
parser.add_argument('--server-url', metavar='URL', default=None,
                    help='Raven server to talk to, overriding the configured one; e.g. http://localhost:5100. '
                         'The other endpoint this app depends on, and the other one worth pointing elsewhere.')
parser.add_argument('--qr', action='store_true',
                    help='show a "Get Raven" QR code in a corner of the window, for demoing at an exhibit')
replserver.add_argument(parser)
opts = parser.parse_args()

import logging
from ..common import logsetup
logsetup.configure(level=getattr(logging, opts.log_level),
                   logfile=opts.log)
logger = logging.getLogger(__name__)

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
    import textwrap
    import threading
    import time
    from collections.abc import Callable

    # WORKAROUND: Deleting a texture or image widget causes DPG to segfault on Nvidia/Linux.
    # https://github.com/hoffstadt/DearPyGui/issues/554
    if platform.system().upper() == "LINUX":
        os.environ["__GLVND_DISALLOW_PATCHING"] = "1"

    import dearpygui.dearpygui as dpg

    from mcpyrate import colorizer
    from unpythonic import call, sym
    from unpythonic.env import env

    # Vendored libraries
    from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa  # https://github.com/juliettef/IconFontCppHeaders
    from ..vendor import DearPyGui_Markdown as dpg_markdown  # https://github.com/IvanNazaruk/DearPyGui-Markdown
    from ..vendor.file_dialog.fdialog import FileDialog  # https://github.com/totallynotdrait/file_dialog, but with custom modifications

    from ..client import api  # Raven-server support
    from ..client.avatar_controller import DPGAvatarController
    from ..client.avatar_renderer import DPGAvatarRenderer
    from ..client import config as client_config

    from ..common import audio
    from ..common.audio import player as audio_player
    from ..common.audio import recorder as audio_recorder
    from ..common import bgtask
    from ..common import datastorelock
    from ..common import docextract
    from ..common import quitsignal
    from ..common import text as common_text
    from ..common import utils as common_utils

    from ..common.gui import animation as gui_animation
    from ..common.gui import helpcard
    from ..common.gui import keyboardmark
    from ..common.gui import messagebox
    from ..common.gui import filedrop
    from ..common.gui import qroverlay
    from ..common.gui import tooltip as gui_tooltip
    from ..common.gui import utils as guiutils
    from ..common.gui.vumeter import DPGVUMeter

    from . import appstate
    from . import audio_input_panel as audio_input  # module: the panel class, and the meter scale the toolbar's VU meter shares
    from . import chatgraph_panel
    from .chat_controller import DPGChatController
    from .cleanup_dialog import DPGCleanupDialog
    from . import config as librarian_config
    # from . import chattree
    from . import hybridir
    from . import imagestore
    from . import llmclient
    from . import textfilestore

    gui_config = librarian_config.gui_config  # shorthand, this is used a lot
logger.info(f"Libraries loaded in {tim.dt:0.6g}s.")

# ----------------------------------------
# Module bootup

# How far one Up/Down keypress scrolls the chat log, counted in font heights. Sized as a comfortable reading
# nudge — enough to bring the next couple of lines in without losing your place, and well short of what a
# page moves.
#
# The load-bearing property is not the count but the margin: it has to **clear the follow-tail floor**.
# `should_follow_tail` treats anything within `_PIN_TOLERANCE_PX` of the end as still at the end, so a
# keypress moving less than that would be undone by the next arriving chunk during a streaming reply. Five
# font heights against that floor's two is a 2.5x margin, and since both are counted in the same unit the
# margin holds at any font size rather than only at this one.
#
# The unit is the font height and not a *line*, which is why it is named that way: a rendered line also
# carries the item spacing, 26 px against a font height of 20 in the chat panel. So five of these are nearer
# four lines than five — harmless, since the property above rests on the ratio and both sides count the same
# unit, but not something the name should claim.
_SCROLL_FONT_HEIGHTS_PER_ARROW = 5

#: What `librarian_config.send_message_key` may say. Two and only two, because ImGui's multiline
#: `InputText` knows exactly two chords and whichever does not commit is the one that inserts the newline.
_SEND_MESSAGE_KEYS = ("enter", "ctrl+enter")

# Refuse to start on a value that is neither, rather than falling back to a default.
#
# The setting is read by string equality in five places — the two labels below, the composer widget's
# `ctrl_enter_for_new_line`, and the two global key handlers — and a value matching none of them does not
# degrade into a default. It degrades into *nothing*: both handlers test for a specific string, so neither
# fires, and the help card meanwhile names a chord with complete confidence. Sending stops working and the
# app says nothing about why.
#
# A silent fallback would be worse than it sounds, too. Nothing in the UI displays this setting, so a user
# who typed `"Enter"` and got Ctrl+Enter has nowhere to notice the difference; they would conclude the
# setting does not work. Failing here costs one restart and names the typo.
if librarian_config.send_message_key not in _SEND_MESSAGE_KEYS:
    expected = ", ".join(repr(key) for key in _SEND_MESSAGE_KEYS)
    print(colorizer.colorize(f"raven-librarian: config.send_message_key is {librarian_config.send_message_key!r}, "
                             f"which is not one of {expected}. Fix it in raven/librarian/config.py.",
                             colorizer.Style.BRIGHT, colorizer.Fore.RED))
    sys.exit(255)

def _send_key_label() -> str:
    """How to name the send chord in a tooltip, per `config.send_message_key`."""
    return "Enter" if librarian_config.send_message_key == "enter" else "Ctrl+Enter"

def _newline_keys_label() -> str:
    """How to name the newline chord in a tooltip, per `config.send_message_key`.

    Always the *other* one of the pair: ImGui's multiline `InputText` knows exactly two chords, Enter and
    Ctrl+Enter, and `ctrl_enter_for_new_line` decides which of them commits. The one that does not commit
    inserts the newline, so naming either half names the other by elimination.

    Shift+Enter is deliberately not mentioned, because it does nothing — the widget has no such binding, and
    saying otherwise sent at least one reader looking for a key that was not there.
    """
    return "Ctrl+Enter" if librarian_config.send_message_key == "enter" else "Enter"

bg = concurrent.futures.ThreadPoolExecutor()
gui_resize_task_manager = bgtask.TaskManager(name="librarian_gui_resize",  # de-spammer for expensive parts of GUI resizing
                                             mode="sequential",
                                             executor=bg)
# Reading an attached document's text. Concurrent, not sequential: a multi-select attaches several files at
# once, and each is an independent question about a different file — a later one must not cancel an earlier.
attachment_task_manager = bgtask.TaskManager(name="librarian_attachment_extract",
                                             mode="concurrent",
                                             executor=bg)
raven_server_url = opts.server_url if opts.server_url is not None else client_config.raven_server_url
if opts.server_url is not None:
    logger.info(f"Using Raven server '{raven_server_url}' from --server-url, overriding the configured '{client_config.raven_server_url}'.")
api.initialize(raven_server_url=raven_server_url,
               raven_api_key_file=client_config.raven_api_key_file,
               executor=bg)  # reuse our executor for client background tasks
audio.initialize(player={"device_name": client_config.tts_playback_audio_device},
                 recorder={"device_name": client_config.stt_capture_audio_device,
                           "executor": bg})

llm_backend_url = opts.backend_url if opts.backend_url is not None else librarian_config.llm_backend_url
if opts.backend_url is not None:
    logger.info(f"Using LLM backend '{llm_backend_url}' from --backend-url, overriding the configured '{librarian_config.llm_backend_url}'.")

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

    # "Send a link to the chat input" affordance: each URL in the chat history gets a small icon to
    # its left; clicking it appends the URL to the user's input draft (e.g. to forward a websearch
    # result to the AI for a webfetch). The markdown renderer is generic; this action is Librarian-
    # specific. The callback fires only on click, so referring to the not-yet-created `chat_field`
    # tag here is fine.
    def _append_url_to_chat_input(url: str) -> None:
        current = dpg.get_value("chat_field")  # tag
        separator = "" if (not current or current.endswith((" ", "\n"))) else " "
        dpg.set_value("chat_field", f"{current}{separator}{url}")  # tag
        dpg.focus_item("chat_field")  # tag
    dpg_markdown.set_url_secondary_action(_append_url_to_chat_input,
                                          glyph=fa.ICON_ARROW_RIGHT_TO_BRACKET,
                                          font=themes_and_fonts.icon_font_solid,
                                          tooltip="Send to chat input:\n{url}")

    # The app's "read this, but nothing has gone wrong" orange. Two owners: the AI-disclosure label below
    # the avatar, and the LLM backend status row above the composer. Named rather than written out at each
    # site, because two literals in one file are two literals that can drift, and these two are meant to be
    # recognizably the same voice — neither is an error, both want reading before the user sends anything.
    _CAUTION_COLOR = (255, 180, 120)

    # animation for document database and web access indicators (cyclic, runs in the background)
    with dpg.theme(tag="my_pulsating_gray_text_theme"):
        with dpg.theme_component(dpg.mvAll):
            pulsating_gray_color = dpg.add_theme_color(dpg.mvThemeCol_Text, (180, 180, 180))
        pulsating_gray_text_glow = gui_animation.PulsatingColor(cycle_duration=2.0,
                                                                theme_color_widget=pulsating_gray_color)
        gui_animation.animator.add(pulsating_gray_text_glow)

    # animation for mic button (cyclic, runs in the background)
    with dpg.theme(tag="my_pulsating_red_text_theme"):
        with dpg.theme_component(dpg.mvAll):
            pulsating_red_color = dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 96, 96))  # color-matching the rec button, "disablable_red_widget_theme"
        pulsating_red_text_glow = gui_animation.PulsatingColor(cycle_duration=2.0,
                                                               theme_color_widget=pulsating_red_color)
        gui_animation.animator.add(pulsating_red_text_glow)

    # animation for the DOCS indicator while RAG is *indexing* (cyclic, runs in the background).
    # Same color as the mic-recording theme — semantically both are "recording" — but a separate theme
    # and pulsator, so resetting the mic phase on recording start doesn't yank the DOCS-indexing pulsation.
    with dpg.theme(tag="my_pulsating_red_docs_theme"):
        with dpg.theme_component(dpg.mvAll):
            pulsating_red_docs_color = dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 96, 96))
        pulsating_red_docs_glow = gui_animation.PulsatingColor(cycle_duration=2.0,
                                                               theme_color_widget=pulsating_red_docs_color)
        gui_animation.animator.add(pulsating_red_docs_glow)

    # Steady (non-pulsating) themes for the long DOCS progress label. The icon and the "DOCS" label carry
    # the recording/reading pulsation cue; the progress label is too long for the eye to read inside one
    # pulsation cycle, so it gets a calm full-alpha variant matching the active state's color.
    with dpg.theme(tag="my_steady_red_docs_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 96, 96))
    with dpg.theme(tag="my_steady_gray_docs_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (180, 180, 180))

    # An attachment whose text could not be extracted. Steady, not pulsating: the pulsation means "working
    # on it", so a failed chip must stop moving or the two states read the same at a glance. Its own theme
    # rather than the DOCS red above, whose name says who owns it — and an owner's name on a shared theme is
    # how a later edit for one owner silently restyles the other.
    with dpg.theme(tag="my_attachment_error_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 96, 96))

    # Themes for the LLM backend status pill above the composer (`_refresh_backend_status_pill`). Its own
    # pulsator rather than the DOCS one, for the reason the DOCS indicator has its own: a pulsation whose
    # phase another owner may reset is a pulsation that jumps.
    #
    # Split steady/pulsating the same way the DOCS row is, and for the same reason — the icon pulsates to
    # say "act on me", the sentence beside it stays at full alpha because a sentence is too long to read
    # inside one cycle. The same caution orange for both bad states: which of the two it is, and what to do
    # about it, is what the words are for.
    with dpg.theme(tag="my_pulsating_caution_backend_theme"):
        with dpg.theme_component(dpg.mvAll):
            pulsating_caution_backend_color = dpg.add_theme_color(dpg.mvThemeCol_Text, _CAUTION_COLOR)
        pulsating_caution_backend_glow = gui_animation.PulsatingColor(cycle_duration=2.0,
                                                                      theme_color_widget=pulsating_caution_backend_color)
        gui_animation.animator.add(pulsating_caution_backend_glow)
    with dpg.theme(tag="my_steady_caution_backend_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, _CAUTION_COLOR)
    # The connected state, which appears only to announce itself and then leaves. Steady on both widgets:
    # nothing here is asking to be acted on.
    with dpg.theme(tag="my_steady_green_backend_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (96, 224, 96))

    # The thought bubble's cloud, in the same color the thinking trace is written in — so the cloud and what
    # it hides are visibly the same thing, whether or not the trace is open.
    #
    # It pulsates only on the reply being generated, and only while the model is actually reasoning; every
    # stored message's cloud is steady. That is the whole point of it: with the trace collapsed, an app that
    # showed nothing would look frozen for exactly as long as the model thinks, which on a thinking model is
    # most of the turn. Pulsating means "still going", in the same vocabulary as the INDEXING / DOCS /
    # READING / SYSTEM / WEB indicators, so it needs no explaining to anyone who has seen those.
    #
    # One shared pair of themes rather than one per message, which the pulsating half makes possible: at most
    # one reply is being generated at a time, so at most one cloud is ever pulsating.
    with dpg.theme(tag="my_pulsating_think_theme"):
        with dpg.theme_component(dpg.mvAll):
            pulsating_think_color = dpg.add_theme_color(dpg.mvThemeCol_Text, gui_config.chat_color_think_front)
        pulsating_think_glow = gui_animation.PulsatingColor(cycle_duration=2.0,
                                                            theme_color_widget=pulsating_think_color)
        gui_animation.animator.add(pulsating_think_glow)
    with dpg.theme(tag="my_steady_think_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, gui_config.chat_color_think_front)

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
# Idle throttle
#
# When nothing is happening — avatar paused (idle auto-off), no LLM streaming, no RAG
# indexing, no recent user input — drop to ~12 fps to conserve CPU/GPU. The headline
# win is the avatar-paused window: while the user is reading, the avatar pauses after
# `idle_timeout`, and from there onward we coast at the throttled rate.
#
# The indicator pulsations (gray for the LLM / DOCS / WEB indicators, red for the mic button,
# red for the DOCS indicator while indexing, red for the backend status pill) run for the
# lifetime of the app, so they are *ambient* and `transient_count` leaves them out. Only a
# button flash or a smooth scroll counts as being busy.

IDLE_SLEEP_S = 0.08   # ~12 fps when idle
INPUT_ACTIVE_S = 0.5  # stay at full fps for this long after last user input

# The AI-disclosure notice shown below the avatar. Module-level because both the widget that renders it and
# `_center_ai_warning`, which measures it to place it, need the exact same string.
_AI_WARNING_TEXT = "You are interacting with an AI system. Response quality and factual accuracy depend on the connected AI — always verify important facts independently."
_AI_WARNING_ICON_W = 23  # the warning glyph, plus the gap DPG leaves between members of a horizontal group
_WINDOW_PADDING = 8      # DPG's default WindowPadding; `setup_themes` overrides only rounding (see `config.py`)
# Residual offset after accounting for the padding, measured from a screenshot: the label still sat this many
# pixels right of the panel's center. Empirical, not derived - the group's reported width does not include the
# warning glyph's left side bearing, and DPG exposes no way to ask for it. Verified by re-measuring at two
# window widths; if the icon or the font changes, re-measure rather than trusting this number.
_AI_WARNING_CENTERING_BIAS = 7

_last_input_ns: int = 0  # monotonic_ns timestamp of last user input

def _is_busy() -> bool:
    """True when the render loop should run at full frame rate."""
    if (time.monotonic_ns() - _last_input_ns) < INPUT_ACTIVE_S * 1e9:
        return True
    if "dpg_avatar_renderer" in globals() and dpg_avatar_renderer.animator_running:
        return True
    if "chat_controller" in globals() and chat_controller.is_generating():
        return True
    if "retriever" in globals() and retriever.is_indexing():
        return True
    return gui_animation.animator.transient_count > 0

def _on_any_input(*_args) -> None:
    global _last_input_ns
    _last_input_ns = time.monotonic_ns()

def _on_mouse_wheel(*args) -> None:
    """Idle-throttle bookkeeping, plus the chat log's scroll-end flash.

    The wheel needs its own handling for the flash: DPG scrolls a child window internally, so no scroll
    animation exists to notice that the end was reached.
    """
    _on_any_input(*args)
    if "chat_controller" in globals() and guiutils.is_mouse_on_widget(chat_controller.view.gui_parent):
        chat_controller.view.note_wheel_scroll()

# --------------------------------------------------------------------------------
# Connect to servers, load datastores

if not api.test_connection():
    sys.exit(255)

# The LLM backend is deliberately not a startup gate, where Raven-server just above is one. Librarian is
# useful with no model in sight — past chats, the cleanup dialog, the settings — and a user who started the
# LLM server second is two clicks from fixing it. So `connect` where a batch tool would call `setup`: it
# reports which of the three states this is instead of raising, on the console here and in the composer's
# status pill (`_refresh_backend_status_pill`) once there is a GUI to put it in.
llm_settings = llmclient.connect(backend_url=llm_backend_url)
print()

# Claim the chat datastore before reading it. Librarian and minichat each hold the whole thing in memory
# and write it back at exit, so running both means the later exit silently discards the other's session.
#
# Bound to a module-level name deliberately: the lock lives as long as this object does, and the process
# needs it for its whole run. Nothing releases it — the OS does that at exit, crash included.
try:
    datastore_lock = datastorelock.acquire(librarian_config.llm_datastore_file, what="The chat datastore")
except datastorelock.DatastoreBusyError as exc:
    print(colorizer.colorize(str(exc), colorizer.Style.BRIGHT, colorizer.Fore.RED))
    sys.exit(255)

logger.info("Loading chat datastore.")
with timer() as tim:
    # Persistent, branching chat history, and app settings (these will auto-persist at app exit).
    datastore, app_state = appstate.load(llm_settings,
                                         librarian_config.llm_datastore_file,
                                         librarian_config.llm_state_file)
logger.info(f"Datastore loaded in {tim.dt:0.6g}s.")

@call
def _apply_stored_audio_capture_settings() -> None:
    """Hand the recorder the capture settings the user tuned last time.

    `audio.initialize` ran before the state file could be read, so the recorder started on the
    configured values; these are the remembered ones, which take precedence over them.

    The microphone is the one setting that can stop existing between runs, so it resolves in three
    steps: the one chosen in the GUI if it is plugged in, else the configured one if it is, else the
    first available. The last two are `audio.initialize`'s doing and have already happened by now;
    this only adds the first, and writes back whichever survived so the state file stops naming a
    microphone nobody has.
    """
    rec = audio_recorder.require()
    rec.silence_threshold = app_state["stt_silence_threshold"]
    rec.autostop_timeout = app_state["stt_autostop_timeout"]
    rec.vu_peak_hold = app_state["stt_vu_peak_hold"]

    # The microphone the user last chose may not be plugged in this time, and a missing one must not
    # stop the app from starting: fall back to whatever `audio.initialize` already opened, and write
    # that back so the state file names a device that exists.
    stored_device = app_state["stt_capture_audio_device"]
    if stored_device is not None and stored_device != rec.device_name:
        try:
            rec.set_device(stored_device)
        except ValueError:
            logger.warning(f"_apply_stored_audio_capture_settings: audio capture device '{stored_device}' from the app state is not present; staying on '{rec.device_name}'.")
    app_state["stt_capture_audio_device"] = rec.device_name

    logger.info(f"_apply_stored_audio_capture_settings: device '{rec.device_name}', silence threshold {rec.silence_threshold}dBFS, autostop timeout {rec.autostop_timeout}s, VU peak hold {rec.vu_peak_hold}s.")

logger.info("Loading RAG (retrieval-augmented generation) document store.")
with timer() as tim:
    docs_dir = pathlib.Path(librarian_config.llm_docs_dir).expanduser().resolve()  # RAG documents (put your documents in this directory)
    db_dir = pathlib.Path(librarian_config.llm_database_dir).expanduser().resolve()  # RAG search indices datastore

    # Load RAG database (it will auto-persist at app exit).
    #
    # `setup` takes a cross-process lock on the index, so this refuses rather than starting beside a
    # running `raven-indexer` and silently undoing its work. Same handling as the chat datastore above.
    try:
        retriever, scanner = hybridir.open_document_store()  # the configured store, with the configured defaults
    except datastorelock.DatastoreBusyError as exc:
        print(colorizer.colorize(str(exc), colorizer.Style.BRIGHT, colorizer.Fore.RED))
        sys.exit(255)

    logger.info(f"RAG document store is at '{str(librarian_config.llm_docs_dir)}' (put your text or PDF documents here).")
    # The retriever's `documents` attribute must be locked before accessing.
    with retriever.datastore_lock:
        plural_s = "s" if len(retriever.documents) != 1 else ""
        logger.info(f"RAG: {len(retriever.documents)} document{plural_s} loaded.")
    logger.info(f"RAG: Search indices are saved in '{str(librarian_config.llm_database_dir)}'.")
logger.info(f"RAG document store loaded in {tim.dt:0.6g}s.")

# --------------------------------------------------------------------------------
# Image attachment (composer staging)
#
# When a vision-capable model (VLM) is loaded, the user can attach images to the message being composed. An
# attachment is staged in memory here until send; on send, `chat_controller.chat_exchange` stores each as a
# datastore sidecar and the staging is cleared. The image bytes are snapshotted at attach time, so a file
# edited or removed on disk between attach and send still sends exactly what the user picked.

# Currently staged image attachments. Each entry is an `env` with `raw` (image bytes), `path`, `provenance_url`,
# `provenance_source` (consumed by `scaffold.user_turn`), plus `strip_group_tag` / `texture_tag` for GUI teardown.
staged_images = []

# Currently staged document attachments (plain text / PDF). Each entry is an `env` with `raw` (file bytes),
# `path`, `name`, `provenance_url`, `provenance_source` (consumed by `scaffold.user_turn`), plus `strip_group_tag`
# and the chip's other widget tags for GUI teardown and restyling. Documents need no vision model (their text is
# folded into the prompt at wire-build) and no thumbnail texture (they render as a chip), so this staging carries
# no texture — but unlike an image, a document must be *read* before it is known to be attachable at all, which
# is what the states below are for.
staged_files = []
_staged_file_counter = 0  # monotonic; keeps chip widget tags unique even across remove-then-re-add

# What a staged document's chip is currently saying. Extracting a large PDF takes seconds (measured: ~4 s for
# 8.5 MB), so the chip appears immediately and reports its own progress rather than the app going quiet — see
# `_add_staged_file`.
ATTACHMENT_EXTRACTING = sym("attachment_extracting")  # text is being read out; chip pulsates
ATTACHMENT_READY = sym("attachment_ready")            # text in hand; chip is calm, and the message can be sent
ATTACHMENT_FAILED = sym("attachment_failed")          # nothing usable in it; chip is red and blocks the send

# Thumbnails for the staged-image strip get their own texture registry (distinct from the chat log's inline-image
# registry, which the controller owns). These thumbnails ARE deleted — on remove, and on send — so this relies on
# the __GLVND_DISALLOW_PATCHING workaround (set at import time) that makes texture deletion safe on Nvidia/Linux.
_staged_image_texture_registry = dpg.add_texture_registry(tag="librarian_staged_image_textures")  # tag
_staged_image_counter = 0  # monotonic; keeps thumbnail widget/texture tags unique even across remove-then-re-add

def _decode_staged_thumbnail(raw: bytes) -> tuple[str, int, int]:
    """Decode `raw` image bytes to a strip-height thumbnail texture. Return `(texture_tag, width, height)`."""
    global _staged_image_counter
    from ..common.image import codec, lanczos  # deferred: pulls torch / Pillow only on an actual attach
    from ..common.image import utils as image_utils
    arr = image_utils.ensure_rgba(codec.decode(raw))  # (H, W, 4) uint8
    src_h, src_w = int(arr.shape[0]), int(arr.shape[1])
    max_h = gui_config.chat_attachments_h - 12  # leave a little vertical room inside the strip
    scale = min(max_h / src_h, 1.0)  # never upscale a small image
    disp_h = max(1, round(src_h * scale))
    disp_w = max(1, round(src_w * scale))
    tensor = image_utils.np_to_tensor(arr, device="cpu")  # (1, 4, H, W) float32
    tensor = lanczos.resize(tensor, disp_h, disp_w)
    flat = image_utils.tensor_to_dpg_flat(tensor)  # flat float32 RGBA in [0, 1]
    _staged_image_counter += 1
    texture_tag = f"staged_image_texture_{_staged_image_counter}"  # tag
    # `dynamic_texture`, not `static_texture`: these thumbnails are deleted at runtime (on remove / on send),
    # and dpg-notes.md's rule is dynamic-for-runtime-deleted, static-for-permanent. Two `split_frame`s because
    # DPG defers the OpenGL upload to a render frame and a single wait empirically doesn't guarantee completion
    # before the thumbnail is first drawn (the raven-cherrypick finding; see dpg-notes.md "Texture upload ordering").
    dpg.add_dynamic_texture(disp_w, disp_h, flat, tag=texture_tag, parent=_staged_image_texture_registry)  # tag
    dpg.split_frame()  # trigger the deferred upload...
    dpg.split_frame()  # ...and ensure it completed before the thumbnail draws
    return texture_tag, disp_w, disp_h

# --------------------------------------------------------------------------------
# LLM backend status
#
# Librarian opens whatever the LLM backend is doing (see the `connect` call above), so the window may be up
# while nothing can answer. The composer's status row is how that is reported, and what it reports is one of
# `llmclient.backend_status`'s three states.
#
# Its scope is the state Librarian *starts* in, and recovery from it. A backend that goes away mid-session
# — the user unloads the model, or stops the server — is not noticed here, because noticing it would mean
# polling a backend that is working, which is a request nobody asked for on every idle session. It is
# reported instead by the send that fails, which the user is present for and can reroll.

# How often to re-probe a backend that cannot answer, in seconds. This runs *only* while the row is up: a
# healthy Librarian sends nothing nobody asked for. Short, because the condition is one the user is actively
# fixing — they start the server or load a model and look back at the window — and the probe is one HTTP
# request, usually against the same machine.
_BACKEND_POLL_INTERVAL_S = 3.0
# How long the row stays up to announce a connection before it goes away.
_BACKEND_CONNECTED_LINGER_S = 4.0
# Slice length for the two waits above. They are waited out in slices so that closing the window does not
# have to sit through a whole interval first.
_BACKEND_POLL_TICK_S = 0.1

# Sequential, so that a click on the row supersedes the poll that is currently sleeping rather than racing
# it: submitting cancels whatever was in flight, which is exactly "retry now".
backend_status_task_manager = bgtask.TaskManager(name="librarian_backend_status",
                                                 mode="sequential",
                                                 executor=bg)

_backend_pill_shown = False  # mirrors the row's visibility; `_refresh_composer_layout` sizes the field from it

def _describe_backend_status(status: sym) -> tuple[str, str, str]:
    """Return `(icon, label, tooltip)` for the composer's status row in `status`.

    The three states get distinct wording because the user can act on the difference: starting a server and
    loading a model into one that is already running are different jobs, and a row that said only "not
    working" would leave them to guess which. All three icons are plug variants, so the glyph changes while
    the row stays recognizably about the connection.

    What is said comes from `llmclient.describe_backend_status`, which every frontend shares. Only the row's
    *label* is local, and only because it has to fit: the shared headline names the backend's address, which
    is more than a one-line row beside a message box can carry, so the address goes in the tooltip with the
    advice and the label says the short of it.
    """
    headline, advice = llmclient.describe_backend_status(status, llm_settings.backend_url)
    if status is llmclient.backend_unreachable:
        return (fa.ICON_PLUG_CIRCLE_XMARK,
                "LLM backend not connected",
                f"{headline}\n{advice}\n\nRetrying automatically. Click to retry now.")
    if status is llmclient.backend_has_no_model:
        return (fa.ICON_PLUG_CIRCLE_EXCLAMATION,
                "LLM backend has no model loaded",
                f"{headline}\n{advice}\n\nRetrying automatically. Click to retry now.")
    # Connected. The model identity is what just changed, so say it — unless the backend does not report one
    # (`NO_MODEL_INFO`), where naming it would turn an announcement into an apology.
    model_suffix = "" if llm_settings.model == llmclient.NO_MODEL_INFO else f" — {llm_settings.model}"
    return (fa.ICON_PLUG_CIRCLE_CHECK,
            f"LLM backend connected{model_suffix}",
            f"{headline}\n\nClick to check again.")

def _refresh_backend_status_pill(status: sym) -> None:
    """Put `status` into the composer's status row, and show the row."""
    global _backend_pill_shown
    icon, label, caption = _describe_backend_status(status)
    dpg.set_value("backend_status_icon", icon)  # tag
    dpg.configure_item("backend_status_button", label=label)  # tag
    # Through the flash rather than around it: clicking the pill flashes "Checking the LLM backend…" into
    # this same tooltip, and the probe it starts usually answers inside that second. A plain write would be
    # put back by the flash when it ends, leaving the tooltip describing the state before the click.
    gui_animation.set_text_under_flash(backend_status_tooltip, caption)
    if status is llmclient.backend_ready:
        dpg.bind_item_theme("backend_status_icon", "my_steady_green_backend_theme")  # tag
        dpg.bind_item_theme("backend_status_button", "my_steady_green_backend_theme")  # tag
    else:
        dpg.bind_item_theme("backend_status_icon", "my_pulsating_caution_backend_theme")  # tag
        dpg.bind_item_theme("backend_status_button", "my_steady_caution_backend_theme")  # tag
    _backend_pill_shown = True
    _refresh_composer_layout()

def _hide_backend_status_pill() -> None:
    """Take the composer's status row down, giving the text field its height back."""
    global _backend_pill_shown
    _backend_pill_shown = False
    _refresh_composer_layout()

def _backend_status_poll_task(task_env: env) -> None:
    """Re-probe the LLM backend until it can answer, then stand the status row down.

    This is the one place a poll is unavoidable, and it is worth being clear about why. A *send* against a
    backend that cannot answer already reports itself and is retried by reroll — user-initiated, free when
    idle, and already built. But nothing user-initiated makes a status readout go green, so the readout has
    to look. The cost is bounded by the duration of a condition the user is fixing: the task exists only
    while the row is up, and returns as soon as it is not.

    `task_env.delay_first_probe`: whether to wait out an interval before the first probe. `True` when the
                                  status was just established (at startup), `False` when the user clicked.
    """
    def keep_waiting(duration: float) -> bool:
        """Wait up to `duration` seconds in slices. Return whether the caller should carry on."""
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if task_env.cancelled or _shutting_down:
                return False
            time.sleep(_BACKEND_POLL_TICK_S)
        return not (task_env.cancelled or _shutting_down)

    previous_status = llmclient.backend_status(llm_settings)

    if task_env.delay_first_probe and not keep_waiting(_BACKEND_POLL_INTERVAL_S):
        return
    while True:
        logger.debug(f"_backend_status_poll_task: {task_env.task_name}: re-probing {llm_settings.backend_url}")
        # Quiet, which for `reconnect` covers the log as well as the console: this asks the same question
        # every few seconds, so what is worth recording is where the answer *changes*, which is here.
        status = llmclient.reconnect(llm_settings)
        if task_env.cancelled or _shutting_down:
            return
        if status is not previous_status:
            logger.info(f"_backend_status_poll_task: {task_env.task_name}: backend at {llm_settings.backend_url} went from {previous_status} to {status}.")
            previous_status = status
        if status is llmclient.backend_ready:
            break
        _refresh_backend_status_pill(status)  # the two bad states can turn into each other
        if not keep_waiting(_BACKEND_POLL_INTERVAL_S):
            return

    logger.info(f"_backend_status_poll_task: {task_env.task_name}: backend is ready, model is '{llm_settings.model}'.")
    # Nothing stored needs repairing, which is the point of stating the backend's facts as per-turn injects:
    # the card's text is determined by the configuration alone, so connecting cannot have made it wrong.
    #
    # Deliberately not calling `appstate.refresh_system_prompt` here. It is now match-or-create rather than
    # rewrite-in-place, so on a card that did not change it does nothing — and on one that did, it would
    # create a root mid-run with no greeting under it while `new_chat_HEAD` still pointed under the old one.
    # A card *can* change here, but only in a deployment whose own prose writes `{model}` or
    # `{context_length}` into it; that one picks up the new text at the next start.
    _refresh_backend_status_pill(llmclient.backend_ready)
    if not keep_waiting(_BACKEND_CONNECTED_LINGER_S):
        return
    _hide_backend_status_pill()

def _start_backend_status_poll(delay_first_probe: bool) -> None:
    """Start watching the LLM backend. Supersedes a watch already running."""
    if _shutting_down:
        return
    backend_status_task_manager.submit(_backend_status_poll_task, env(delay_first_probe=delay_first_probe))

# How often to ask Raven-server whether it is still there. Unlike the LLM backend's poll, this one runs for
# the whole session: the LLM row appears at startup or when a send fails, both of which announce themselves,
# where a Raven-server that dies mid-session announces nothing at all. Everything it serves — the avatar,
# speech, subtitles, translation, the embeddings behind document search — is called only on demand, so
# without this the news arrives as a failure the next time the user asks for one of them.
_SERVER_POLL_INTERVAL_S = 5.0
# How long the row stays up to announce a recovery before it goes away.
_SERVER_RECOVERED_LINGER_S = 4.0

# Sequential, so a click on the row supersedes the poll that is currently sleeping rather than racing it.
server_status_task_manager = bgtask.TaskManager(name="librarian_server_status",
                                                mode="sequential",
                                                executor=bg)

_server_was_available = True  # what the row currently claims; the poll only acts on changes

def _describe_server_status(available: bool) -> tuple[str, str, str]:
    """Return `(icon, label, tooltip)` for the utility panel's Raven-server row.

    `available`: whether the server is answering.

    The tooltip names what stops working rather than only that something did. Raven-server is where the
    avatar, the speech and the embeddings behind document search all live, and which of those the reader
    cares about decides whether this is an emergency or an annoyance. It also says what is *not* affected:
    the chat goes to a different server, so a reader watching their conversation still work would otherwise
    have to guess whether this row was about them.
    """
    if available:
        return (fa.ICON_PLUG_CIRCLE_CHECK,
                "Raven-server connected",
                f"Raven-server at {raven_server_url} is answering again.\n\nClick to check again.")
    return (fa.ICON_PLUG_CIRCLE_XMARK,
            "Raven-server not connected",
            f"Cannot reach Raven-server at {raven_server_url}.\n\n"
            "The avatar, speech, subtitles, translation, the search over your\n"
            "documents and the AI's internet access all run there, and will fail\n"
            "until it is back. The chat itself does not - that is the LLM backend,\n"
            "which is separate.\n\n"
            "Retrying automatically. Click to retry now.")

def _refresh_server_status_pill(available: bool) -> None:
    """Put the server's status into the utility panel's row, and show the row."""
    icon, label, caption = _describe_server_status(available)
    dpg.set_value("server_status_icon", icon)  # tag
    dpg.configure_item("server_status_button", label=label)  # tag
    # Through the flash rather than around it, as the backend row does: a click flashes its own message into
    # this tooltip, and a plain write would be put back when the flash ended.
    gui_animation.set_text_under_flash(server_status_tooltip, caption)
    if available:
        dpg.bind_item_theme("server_status_icon", "my_steady_green_backend_theme")  # tag
        dpg.bind_item_theme("server_status_button", "my_steady_green_backend_theme")  # tag
    else:
        dpg.bind_item_theme("server_status_icon", "my_pulsating_caution_backend_theme")  # tag
        dpg.bind_item_theme("server_status_button", "my_steady_caution_backend_theme")  # tag
    dpg.show_item("server_status_pill")  # tag

def _hide_server_status_pill() -> None:
    """Take the utility panel's server row down."""
    with guiutils.nonexistent_ok():
        dpg.hide_item("server_status_pill")  # tag

def _server_status_poll_task(task_env: env) -> None:
    """Watch Raven-server for the whole session, and put the row up when it goes away.

    Edge-triggered: the row is written when the answer *changes*, so a server that stays down does not have
    its caption rewritten every few seconds, and one that stays up costs nothing but the probe.

    `task_env.delay_first_probe`: whether to wait out an interval before the first probe. `True` for the
                                  standing watch, `False` when the user clicked and is waiting for an
                                  answer now.
    """
    global _server_was_available

    def keep_waiting(duration: float) -> bool:
        """Wait up to `duration` seconds in slices. Return whether the caller should carry on."""
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if task_env.cancelled or _shutting_down:
                return False
            time.sleep(_BACKEND_POLL_TICK_S)
        return not (task_env.cancelled or _shutting_down)

    if task_env.delay_first_probe and not keep_waiting(_SERVER_POLL_INTERVAL_S):
        return
    while True:
        available = api.raven_server_available()
        if task_env.cancelled or _shutting_down:  # the probe takes a network timeout; the app may be gone by now
            return
        if available != _server_was_available:
            logger.info(f"_server_status_poll_task: {task_env.task_name}: Raven-server at {raven_server_url} went from "
                        f"{'up' if _server_was_available else 'down'} to {'up' if available else 'down'}.")
            _server_was_available = available
            _refresh_server_status_pill(available)
            if available:
                # Announce the recovery, then stand down. A row that stayed up would become furniture, and
                # the next one to appear would not be noticed.
                if not keep_waiting(_SERVER_RECOVERED_LINGER_S):
                    return
                _hide_server_status_pill()
        if not keep_waiting(_SERVER_POLL_INTERVAL_S):
            return

def _start_server_status_poll(delay_first_probe: bool) -> None:
    """Start watching Raven-server. Supersedes a watch already running."""
    if _shutting_down:
        return
    server_status_task_manager.submit(_server_status_poll_task, env(delay_first_probe=delay_first_probe))

def _request_server_recheck() -> None:
    """Probe Raven-server now instead of at the poll's next tick. The server row's click action."""
    gui_animation.flash_button(button="server_status_button",  # tag
                               message="Checking Raven-server...",
                               duration=gui_config.acknowledgment_duration,
                               tooltip=server_status_tooltip)
    _start_server_status_poll(delay_first_probe=False)

def _request_backend_reconnect() -> None:
    """Re-probe the LLM backend now, instead of at the poll's next tick. The status row's click action."""
    gui_animation.flash_button(button="backend_status_button",  # tag
                               message="Checking the LLM backend...",
                               duration=gui_config.acknowledgment_duration,
                               tooltip=backend_status_tooltip)
    _start_backend_status_poll(delay_first_probe=False)

def _refresh_composer_layout() -> None:
    """Show or hide the composer's two optional rows, and give the text field the height they leave.

    The rows are the backend-status pill (above the field) and the staged-attachments strip (below it,
    shared by staged images and staged documents). The composer's outer height (`chat_controls_h`) is fixed,
    so a row that appears takes its height from the text field rather than adding to the composer — the chat
    and avatar panels never jump.

    One function for both rows rather than one per row: the field's height is a function of *both*
    conditions, so two writers would each set it as if it were the only one, and whichever ran last would
    win. Call after changing either condition.
    """
    attachments_shown = bool(staged_images or staged_files)
    field_h = gui_config.chat_field_h
    if _backend_pill_shown:
        dpg.show_item("backend_status_pill")  # tag
        field_h -= gui_config.chat_backend_pill_h
    else:
        dpg.hide_item("backend_status_pill")  # tag
    if attachments_shown:
        dpg.show_item("chat_attachments_strip")  # tag
        field_h -= gui_config.chat_attachments_h
    else:
        dpg.hide_item("chat_attachments_strip")  # tag
    dpg.configure_item("chat_field", height=field_h)  # tag

def _remove_staged_image(staged: env) -> None:
    """Remove one staged attachment: drop it from the list and delete its strip widgets + thumbnail texture."""
    if staged in staged_images:
        staged_images.remove(staged)
    with guiutils.nonexistent_ok():
        dpg.delete_item(staged.strip_group_tag)  # the thumbnail image + its remove button
    with guiutils.nonexistent_ok():
        dpg.delete_item(staged.texture_tag)  # free the thumbnail texture (safe under the GLVND workaround)
    _refresh_composer_layout()

def _clear_staged_images() -> None:
    """Remove all staged attachments (called after a send)."""
    for staged in list(staged_images):
        _remove_staged_image(staged)

def _add_staged_image(path: str) -> None:
    """Stage the image at `path`: snapshot its bytes, build a strip thumbnail, and register it for sending."""
    try:
        raw = pathlib.Path(path).read_bytes()
        texture_tag, thumb_w, thumb_h = _decode_staged_thumbnail(raw)
    except Exception as exc:  # noqa: BLE001 -- a bad file must not break the composer; just skip it with a log
        logger.error(f"_add_staged_image: failed to stage '{path}': {type(exc)}: {exc}")
        return
    idx = _staged_image_counter  # bumped inside `_decode_staged_thumbnail`; unique per thumbnail
    strip_group_tag = f"staged_image_group_{idx}"  # tag
    remove_button_tag = f"staged_image_remove_{idx}"  # tag
    staged = env(raw=raw,
                 path=path,
                 provenance_url=pathlib.Path(path).resolve().as_uri(),  # "file:///abs/path" — provenance only, never a live ref
                 provenance_source="user_attachment",
                 strip_group_tag=strip_group_tag,
                 texture_tag=texture_tag)
    image_tag = f"staged_image_thumb_{idx}"  # tag
    with dpg.group(parent="chat_attachments_strip", horizontal=True, tag=strip_group_tag):  # tag
        dpg.add_image(texture_tag, width=thumb_w, height=thumb_h, tag=image_tag)  # tag
        with dpg.tooltip(image_tag):  # tag  # filename only (the remove button's tooltip adds the action)
            dpg.add_text(pathlib.Path(path).name)
        dpg.add_button(label=fa.ICON_XMARK,
                       width=gui_config.toolbutton_w,
                       callback=lambda: _remove_staged_image(staged),
                       tag=remove_button_tag)  # tag
        dpg.bind_item_font(remove_button_tag, themes_and_fonts.icon_font_solid)  # tag
        dpg.bind_item_theme(remove_button_tag, "disablable_widget_theme")  # tag
        with dpg.tooltip(remove_button_tag):  # tag
            dpg.add_text(f"Remove attachment\n{pathlib.Path(path).name}")
    staged_images.append(staged)
    _refresh_composer_layout()

# What the attach tooltip names as accepted, asked for rather than typed out. Each kind declares its own
# formats (`imagestore` and `docextract`), so spelling them out again here would put the tooltip at the mercy
# of whoever remembers to update prose. The user reading it is holding a file and deciding whether to bother,
# so a list that lags the backend is the expensive kind of wrong.
_ATTACH_DOC_EXTS_TEXT = " ".join(docextract.supported_extensions())
_ATTACH_IMAGE_EXTS_TEXT = " ".join(imagestore.supported_extensions())


def _remove_staged_file(staged: env) -> None:
    """Remove one staged document: drop it from the list and delete its strip chip widgets.

    Removing is also how a failed attachment is cleared, and how a user gives up on one that is taking too
    long — so this is what unblocks the send, and re-checks it.
    """
    if staged in staged_files:
        staged_files.remove(staged)
    with guiutils.nonexistent_ok():
        dpg.delete_item(staged.strip_group_tag)  # the chip group (icon + filename + remove button)
    for tooltip in (staged.icon_tooltip, staged.label_tooltip):  # windows at the root; the chip group does not hold them
        if tooltip is not None:
            tooltip.destroy()
    _refresh_composer_layout()
    _refresh_send_gate()

def _clear_staged_files() -> None:
    """Remove all staged documents (called after a send)."""
    for staged in list(staged_files):
        _remove_staged_file(staged)

def _add_staged_file(path: str) -> None:
    """Stage the document at `path`: add its chip straight away, and read its text in the background.

    Reading is what takes the time — pypdf is pure Python, and a large paper costs seconds (measured: 4 s for
    an 8.5 MB PDF). Doing it here, inline, would do it on DPG's single callback thread, which is the one that
    runs every other callback in the app: for those seconds nothing would respond, not the composer, not a
    hotkey, not another button's own acknowledgment flash. So the chip goes up immediately and says what it is
    doing — pulsating while reading, calm when ready, red when the document turns out to hold no text.

    The bytes are snapshotted with the text, so a file edited on disk between attach and send still sends what
    the user picked. The text is handed to `textfilestore` rather than discarded, so the wire-build that needs
    it later does not read the same document a second time.
    """
    global _staged_file_counter
    name = pathlib.Path(path).name
    _staged_file_counter += 1
    idx = _staged_file_counter
    strip_group_tag = f"staged_file_group_{idx}"  # tag
    icon_tag = f"staged_file_icon_{idx}"  # tag
    label_tag = f"staged_file_label_{idx}"  # tag
    remove_button_tag = f"staged_file_remove_{idx}"  # tag
    staged = env(raw=None,  # filled in by the extraction task, along with `status`
                 text=None,
                 path=path,
                 name=name,
                 provenance_url=pathlib.Path(path).resolve().as_uri(),  # "file:///abs/path" — provenance only, never a live ref
                 provenance_source="user_attachment",
                 strip_group_tag=strip_group_tag,
                 status=ATTACHMENT_EXTRACTING,
                 error_message=None,
                 icon_tag=icon_tag,
                 label_tag=label_tag,
                 icon_tooltip=None,  # the two `Tooltip`s below, once the chip's widgets exist
                 label_tooltip=None)
    with dpg.group(parent="chat_attachments_strip", horizontal=True, tag=strip_group_tag):  # tag
        dpg.add_text(fa.ICON_FILE_LINES, tag=icon_tag)  # tag  # a document glyph stands in for the image thumbnail
        dpg.bind_item_font(icon_tag, themes_and_fonts.icon_font_solid)  # tag
        # Wrap the filename so a very long name can't push the remove button off the (fixed-width) composer panel
        # — bounded conservatively so the button stays reachable even on a narrow (~1080p-split) window. The full
        # name is always available via the tooltips.
        dpg.add_text(name, wrap=420, tag=label_tag)  # tag
        # Both the icon and the name carry the chip's state, so both answer when hovered. A failed chip's
        # colour says *that* something is wrong; only the tooltip can say what. Self-sizing, because the
        # caption grows from a bare filename to a filename plus an explanation while the chip is hovered —
        # reading a large PDF takes seconds, which is exactly when someone is hovering it to ask why.
        staged.icon_tooltip = gui_tooltip.Tooltip(icon_tag, name)  # tag
        staged.label_tooltip = gui_tooltip.Tooltip(label_tag, name)  # tag
        dpg.add_button(label=fa.ICON_XMARK,
                       width=gui_config.toolbutton_w,
                       callback=lambda: _remove_staged_file(staged),
                       tag=remove_button_tag)  # tag
        dpg.bind_item_font(remove_button_tag, themes_and_fonts.icon_font_solid)  # tag
        dpg.bind_item_theme(remove_button_tag, "disablable_widget_theme")  # tag
        with dpg.tooltip(remove_button_tag):  # tag
            dpg.add_text(f"Remove attachment\n{name}")
    _apply_staged_file_appearance(staged)
    staged_files.append(staged)
    _refresh_composer_layout()
    _refresh_send_gate()
    # `staged` reaches the task through the closure, and only its name goes in the task `env`. `TaskManager`
    # logs the env at DEBUG, and `staged.raw` is the whole document — passing it there put an 8 MB PDF in the
    # log per attachment, at the very log level this area asks people to run at.
    attachment_task_manager.submit(_make_extraction_task(staged), env(name=name))


def _apply_staged_file_appearance(staged: env) -> None:
    """Restyle a staged document's chip and retarget its tooltips to match `staged.status`.

    Called from the extraction task as well as from the GUI thread, which DPG allows. Every widget is
    guarded, because the user may remove the chip while its document is still being read — the task cannot
    be stopped mid-`extract_text` (pypdf offers no hook), so it finishes and reports into widgets that are
    no longer there.
    """
    if staged.status is ATTACHMENT_EXTRACTING:
        theme, detail = "my_pulsating_gray_text_theme", "Reading the text out of this document…"  # tag
    elif staged.status is ATTACHMENT_FAILED:
        theme, detail = "my_attachment_error_theme", staged.error_message  # tag
    else:
        theme, detail = None, None  # ready: the app's ordinary text colour, and nothing more to say
    caption = f"{staged.name}\n\n{detail}" if detail else staged.name
    with guiutils.nonexistent_ok():
        for item in (staged.icon_tag, staged.label_tag):
            dpg.bind_item_theme(item, theme)
    for tooltip in (staged.icon_tooltip, staged.label_tooltip):
        tooltip.text = caption


def _make_extraction_task(staged: env) -> Callable[[env], None]:
    """Build the background task that reads `staged`'s document and settles its chip into ready or failed.

    A closure rather than a plain function taking `staged` in its task `env`, so that the document's bytes
    never enter anything `TaskManager` prints: it logs the env at DEBUG, and an `env` repr recurses, so one
    attached paper became megabytes of log per attachment.
    """
    def extract_staged_file(task_env: env) -> None:
        try:
            raw = pathlib.Path(staged.path).read_bytes()
            text = docextract.extract_text(staged.path)
        except Exception as exc:  # noqa: BLE001 -- a bad file must not break the composer; report it on the chip
            logger.error(f"extract_staged_file: {task_env.task_name}: failed to read '{staged.path}': {type(exc)}: {exc}")
            _settle_staged_file(staged, error_message="This file could not be read as text.")
            return
        if task_env.cancelled:  # the chip is already gone; reporting into it would only log noise
            return
        if not text:
            logger.info(f"extract_staged_file: {task_env.task_name}: '{staged.path}' yielded no extractable text.")
            _settle_staged_file(staged, error_message="There is no text in this document.\n"
                                                      "If it is a scanned or image-only PDF, run it through OCR first.")
            return
        # Filed under the name these bytes will get as a sidecar, so the wire-build reuses it instead of
        # reading the document again.
        textfilestore.remember_extracted_text(staged.name, raw, text)
        staged.raw = raw
        staged.text = text
        _settle_staged_file(staged, error_message=None)
    return extract_staged_file


def _settle_staged_file(staged: env, *, error_message: str | None) -> None:
    """Move `staged` out of the extracting state, and bring the GUI up to date."""
    staged.status = ATTACHMENT_FAILED if error_message is not None else ATTACHMENT_READY
    staged.error_message = error_message
    if staged not in staged_files:  # removed while it was being read; nothing left to restyle or unblock
        return
    _apply_staged_file_appearance(staged)
    _refresh_send_gate()


def _count_staged_files(status: sym) -> int:
    """How many staged documents are currently in `status`."""
    return sum(1 for staged in staged_files if staged.status is status)


def _describe_send_gate() -> str | None:
    """Why the message cannot be sent right now, or `None` when it can.

    A multi-select attaches several documents at once, so every count here can be more than one — and the
    failed case is reported before the pending one, because it is the one the user has to act on.
    """
    # A turn in flight blocks sending, and is reported first because it is the one condition the user can
    # do nothing about except wait. Two turns on one branch would interleave their writes to HEAD, and both
    # would stream into the same view.
    if "chat_controller" in globals() and chat_controller.is_generating():
        return "The AI is still writing. Cancel with Ctrl+G, or wait for it to finish."
    n_failed = _count_staged_files(ATTACHMENT_FAILED)
    if n_failed:
        if n_failed == 1:
            return "Remove the attachment shown in red — its text could not be read."
        return f"Remove the {n_failed} attachments shown in red — their text could not be read."
    n_pending = _count_staged_files(ATTACHMENT_EXTRACTING)
    if n_pending:
        if n_pending == 1:
            return "Still reading an attached document."
        return f"Still reading {n_pending} attached documents."
    return None


# What the gate last said, so that polling it every frame costs a comparison rather than a DPG round-trip.
# `None` means "sending is allowed"; a string is the reason it is not. Starts at a value no reason can
# equal, so the first refresh always applies.
_send_gate_reason: str | None | object = object()

def _refresh_send_gate() -> None:
    """Enable or disable sending, according to the staged documents' states and whether a turn is in flight.

    A failed attachment blocks rather than being silently dropped: the user chose that document, and sending
    the message without it would answer a question they did not ask. One still being read blocks too, for the
    shorter reason that its text is not there yet.

    The button is disabled *and* the callback re-checks. Disabling the button says so visibly, but the send
    hotkey fires the text field's own callback and never touches the button — so the button is the
    affordance and the callback's own check is the rule.
    """
    global _send_gate_reason
    reason = _describe_send_gate()
    if reason == _send_gate_reason:  # nothing to say that is not already on screen
        return
    _send_gate_reason = reason
    with guiutils.nonexistent_ok():
        dpg.configure_item("chat_send_button", enabled=(reason is None))  # tag
    chat_send_tooltip.text = reason if reason is not None else f"Send to AI [{_send_key_label()}]"


# The frame a send was last requested in, so that the two key paths into one keypress produce one send.
# `None` before the first, which no frame number equals.
_last_send_frame: int | None = None

def _request_send() -> None:
    """Send the composer's message, at most once per frame. The entry point for both key paths.

    The chord that sends reaches two handlers, and which of them is the right one depends on state that
    cannot be read at the time. ImGui owns the chord while the composer is active and commits the edit
    itself, so the field's own callback is what sends — but the composer is often *not* active, and then
    only the global hotkey handler hears anything. Hence both, coalesced.

    Measured in `investigations/dpg-focus/commit_chord_dispatch_probe.py`, and coalescing is what that
    measurement leaves rather than what it was looking for. Both handlers fire on the same keypress, the
    global one first — and at the moment it runs, a composer that has just committed and one that was
    auto-focused at startup and never touched read *identically* (focused, not active) while wanting
    opposite answers. So no predicate separates them, and none has to: the first request through sends and
    the second arrives too soon after it to be a second send.

    Not locked, and deliberately: DPG runs every Python callback on one dedicated callback thread, so the
    two arrive in sequence rather than concurrently and there is no interleaving to protect against.

    **What the two share is a very short interval, not a frame number.** They run back to back on that one
    thread, microseconds apart against a frame of roughly sixteen milliseconds. The probe saw both report
    the same `get_frame_count()`, but that follows from the gap rather than bounding it: the main thread
    advances the counter independently, so a boundary can fall between the two reads and an equality test
    would then miss — a double send, from a race that reproduces about as often as one asks it to.

    Hence a window of one frame, which is the interval expressed in the only clock available here. One is
    enough because a gap that short admits at most one boundary. And it cannot swallow anything real: two
    *legitimate* sends that close together are not reachable — a human cannot produce them, and
    `_describe_send_gate` refuses while a turn is in flight.
    """
    global _last_send_frame
    frame = dpg.get_frame_count()
    if _last_send_frame is not None and frame - _last_send_frame <= 1:
        return
    _last_send_frame = frame
    send_message_to_ai_callback()

def _attach_callback(selected_files) -> None:
    """FileDialog callback: route each selected file to image or document staging by its extension.

    Documents attach on any model (whatever `docextract` can read). Images need a vision model; on a *confirmed* text-only
    model (`model_is_vlm is False`) a picked image is rejected with a dialog, since the model could not use it.

    The picker steers away from that case — `_attach_filter_list` offers no image formats on a text-only model —
    but this rejection remains the enforcement, because "All files" is still offered and a drag'n'drop does not
    go through the picker at all.
    """
    logger.debug(f"_attach_callback: {len(selected_files)} file{common_text.plural_s(len(selected_files))} selected.")
    rejected_images = []
    for selected_file in selected_files:
        if imagestore.is_supported(selected_file):
            if llm_settings.model_is_vlm is False:  # confirmed text-only: the model can't see an image
                rejected_images.append(pathlib.Path(selected_file).name)
            else:
                _add_staged_image(selected_file)
        elif docextract.is_supported(selected_file):
            _add_staged_file(selected_file)
        else:
            logger.info(f"_attach_callback: '{selected_file}': unsupported file type, skipping.")
    if rejected_images:
        names = "\n".join(f"  - {n}" for n in rejected_images)
        messagebox.modal_dialog(window_title="Images need a vision model",
                                message=f"The loaded model is text-only, so these images were not attached:\n\n{names}\n\n"
                                        "Load a vision model (VLM) to attach images. Documents attach on any model.",
                                buttons=["OK"], ok_button="OK", cancel_button="OK",
                                centering_reference_window="librarian_main_window")

def show_attach_dialog() -> None:
    """Composer button callback: open the attach dialog (images and documents).

    Flash the button on click as an acknowledgment. Building the file listing can take a moment on a large
    directory, and the dialog only appears afterward — without this the click looks like it did nothing. The
    flash animates on the render loop, independent of this callback thread doing the (possibly slow) dialog build,
    so the pulse shows immediately even while the listing is still being assembled."""
    if _filedialog_attach is None:
        return
    gui_animation.flash_button(button="chat_attach_button",  # tag
                               tooltip=chat_attach_tooltip,
                               message="Opening the file browser…",  # shown in the tooltip while it's hovered during the flash
                               duration=gui_config.acknowledgment_duration)
    _filedialog_attach.set_filter_list(_attach_filter_list())  # the loaded model may have changed since the dialog was built
    _filedialog_attach.show_file_dialog()

# The attach dialog manages its own window (created outside any window context).
#
# The filters are grouped rather than one item per extension: the default shows everything that can actually be
# attached and nothing else, and the narrower items are there for when you know which kind you are after.
# Both sets are asked for at startup rather than written out, so the picker cannot drift from what the ingester
# and the image store will accept.
_attachable_image_extensions = imagestore.supported_extensions()
_attachable_document_extensions = docextract.supported_extensions()

def _attach_filter_list() -> list:
    """The file type filters to offer, given what the loaded model can currently read.

    On a *confirmed* text-only model, image formats are not offered, so the common way of picking one no
    longer leads to a rejection. Recomputed per open rather than fixed at construction: `llmclient.reconnect`
    updates `model_is_vlm` in place when a model is loaded, so the answer changes while the app runs.

    "All files" stays offered either way — an unusual extension is a real thing to want, and taking it away
    would trade one refusal for another. So `_attach_callback` keeps its rejection as the backstop; what
    changes is that reaching it now takes a deliberate detour rather than being the default path.
    """
    if llm_settings.model_is_vlm is False:  # confirmed text-only
        return [("Documents", _attachable_document_extensions),
                ".*"]
    return [("Documents and images", (*_attachable_document_extensions, *_attachable_image_extensions)),
            ("Documents", _attachable_document_extensions),
            ("Images", _attachable_image_extensions),
            ".*"]

_filedialog_attach = FileDialog(title="Attach one or more files [Ctrl+click to multi-select]",
                                tag="attach_file_dialog",
                                callback=_attach_callback,
                                filter_list=_attach_filter_list(),
                                multi_selection=True,
                                default_path=os.path.expanduser("~"))

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
        def _get_chat_panel_base_size() -> tuple[int, int]:  # at initial view, with the window at its design size
            w = gui_config.chat_panel_w + 16  # 16 = round border (8 on each side)
            h = gui_config.main_window_h - (gui_config.ai_warning_h + 16) - (gui_config.chat_controls_h + 16) + 8
            return w, h
        def _get_chat_panel_size(main_window_w: int, main_window_h: int) -> tuple[int, int]:  # at current window size
            extra_w = main_window_w - gui_config.main_window_w
            extra_h = main_window_h - gui_config.main_window_h
            base_w, base_h = _get_chat_panel_base_size()
            w = base_w + extra_w
            h = base_h + extra_h
            return w, h
        def _get_avatar_panel_base_size() -> tuple[int, int]:
            chat_panel_base_w, chat_panel_base_h = _get_chat_panel_base_size()
            w = (gui_config.main_window_w - chat_panel_base_w - 3 * 8)  # the 3 * 8 are the outer borders outside the panels (between panel and window edge, and between the panels)
            h = chat_panel_base_h
            return w, h
        def _get_avatar_panel_size(main_window_w: int, main_window_h: int) -> tuple[int, int]:
            extra_w = 0  # avatar panel keeps the same width
            extra_h = main_window_h - gui_config.main_window_h
            base_w, base_h = _get_avatar_panel_base_size()
            w = base_w + extra_w
            h = base_h + extra_h
            return w, h
        def _get_subtitle_bottom_y0(avatar_panel_h: int):
            return (avatar_panel_h - 24) + gui_config.subtitle_y0
        def _get_chat_field_base_width() -> int:  # full width; the toolbar (send/mic/VU) sits below the field now, not beside it
            return gui_config.chat_panel_w  # `chat_controls` has `no_scrollbar=True`, so full width can't trip a horizontal scrollbar
        def _get_chat_field_width(main_window_w: int) -> int:
            extra_w = main_window_w - gui_config.main_window_w
            base_w = _get_chat_field_base_width()
            w = base_w + extra_w
            return w
        def _get_chat_controls_base_size() -> tuple[int, int]:
            chat_panel_base_w, chat_panel_base_h = _get_chat_panel_base_size()
            w = chat_panel_base_w
            h = gui_config.chat_controls_h
            return w, h
        def _get_chat_controls_size(main_window_w: int, main_window_h: int) -> tuple[int, int]:
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
                    with dpg.group():  # composer: vertical stack (backend status pill / multiline text field / staged-image strip / toolbar)
                        # LLM backend status. Hidden whenever the backend is answering and has something to
                        # answer with, which is the ordinary case — so this is a row that normally isn't
                        # there at all, appearing to report a condition the user can fix and leaving once
                        # they have. Its contents are set by `_refresh_backend_status_pill`.
                        #
                        # Above the field rather than beside the send button, because it is a precondition
                        # for the whole composer rather than a property of one control — and here it is in
                        # the reader's eye on the way to the thing they are about to type into.
                        #
                        # The icon is a plain text widget and only the label is a button, so the leftmost
                        # ~23 px of the row do not respond to a click. The alternative is no icon: DPG draws
                        # an item's label in one font, and the app font has no warning glyph (the same
                        # constraint that decides the jump-to-latest pill's font — see `chat_controller`).
                        # An icon that reads at a glance is worth more than those pixels.
                        with dpg.group(tag="backend_status_pill", horizontal=True, show=False):  # tag
                            dpg.add_text(fa.ICON_PLUG_CIRCLE_XMARK, tag="backend_status_icon")  # tag
                            dpg.bind_item_font("backend_status_icon", themes_and_fonts.icon_font_solid)  # tag
                            dpg.add_button(label="",
                                           callback=lambda: _request_backend_reconnect(),
                                           tag="backend_status_button")  # tag
                            # Self-sizing, because this caption is rewritten on every status change and
                            # again by the click flash — and a `dpg.tooltip` is drawn once at its previous
                            # size each time that happens.
                            backend_status_tooltip = gui_tooltip.Tooltip("backend_status_button", "")  # tag

                        def send_message_to_ai_callback() -> None:
                            # Grab and strip the message. The trailing newline the multiline widget inserts on the
                            # sending Enter, plus any stray whitespace, come off here. An empty result is intentional
                            # and still sends — an empty user message is Librarian's canonical "let the AI take
                            # another turn" gesture.
                            # An attachment still being read, or one that turned out to have no text, holds the
                            # send. The button is already disabled, but the send hotkey comes through the text
                            # field's own callback and never consults it — so the rule lives here.
                            gate_reason = _describe_send_gate()
                            if gate_reason is not None:
                                # The flash says only "not now". The reason is already standing in the
                                # send button's tooltip, which `_refresh_send_gate` keeps current, so
                                # repeating it here would write the same string over itself.
                                gui_animation.animator.add(gui_animation.WidgetFlash(target="chat_send_button",  # tag
                                                                                     duration=1.0,
                                                                                     flash_color=(255, 32, 32),
                                                                                     text_color=(255, 255, 255)))
                                return
                            user_message_text = dpg.get_value("chat_field").strip()  # tag
                            # Snapshot the staged attachments and hand them off, then clear the staging. `chat_exchange`
                            # stores each image (bytes and all) on a background thread from this snapshot, so clearing
                            # the strip and its textures right away can't pull the rug out from under the send.
                            outgoing_images = list(staged_images)
                            outgoing_files = list(staged_files)
                            chat_controller.chat_exchange(user_message_text,
                                                          staged_images=(outgoing_images or None),
                                                          staged_files=(outgoing_files or None))
                            _clear_staged_images()
                            _clear_staged_files()
                            # Clear the composer. ImGui owns the *active* (focused) multiline input's edit buffer
                            # and ignores an external `set_value` while it's focused, writing its own buffer back on
                            # deactivation — so a focused Enter-send can't be cleared by `set_value` alone. (A Send-
                            # button click clears trivially, but only because the click already moved focus off the
                            # field; cf. the Visualizer search field, which likewise never clears while focused.)
                            # So move focus off the field to deactivate it, let that frame apply the deactivation,
                            # then clear the now-inactive field. Safe: this runs on a DPG event-callback thread
                            # (Send button / key handler), never the render loop (where `split_frame` would
                            # deadlock).
                            #
                            # Focus is then left on the send button rather than returned to the composer.
                            # Sending is a departure from the field, and what follows a send is reading a reply
                            # — so the navigation keys should be live while it streams, which they are not while
                            # the composer is being edited. `Ctrl+Space` comes back for the next message. The
                            # field keeps its cleared value and reloads it whenever it is next activated.
                            #
                            # The button is a safe place to leave it: DPG does not enable ImGui's keyboard-nav
                            # activation, so a focused button ignores Space and Enter and cannot re-send.
                            # (Measured, because resting focus on a *send* button is the kind of thing that
                            # wants checking rather than assuming.) The chat panel is not an alternative —
                            # `dpg.focus_item` cannot focus a child window, and returns the caret to the
                            # composer when asked to; see `_build_initial_chat_view`.
                            dpg.focus_item("chat_send_button")  # tag  # deactivate the input's ImGui edit buffer
                            dpg.split_frame()
                            dpg.set_value("chat_field", "")  # tag  # field inactive now, so the clear sticks

                        def record_audio_message_callback() -> None:
                            if not audio_recorder.require().is_recording():
                                start_recording_audio_message()
                            else:
                                stop_recording_audio_message()
                        def start_recording_audio_message() -> None:
                            # The microphone is one device handle, so the audio input panel has to give it up
                            # before a message can be recorded. It takes it back in `stop_recording_audio_message`.
                            audio_input_panel.stop_monitoring()

                            # Start capturing before saying so. A refused start — the device still open —
                            # would otherwise leave the button glowing over a recorder that is not
                            # recording, and the user finds out when the transcript comes back empty.
                            if not audio_recorder.require().start(on_autostop=stop_recording_audio_message):
                                logger.error("start_recording_audio_message: The audio device is busy; not recording.")
                                gui_animation.flash_button(button="record_audio_message_button",  # tag
                                                           tooltip=record_audio_message_tooltip,
                                                           ok=False, message="Microphone busy",
                                                           duration=gui_config.acknowledgment_duration)
                                audio_input_panel.start_monitoring()  # give the panel its meter back
                                return

                            # Acknowledge in GUI
                            pulsating_red_text_glow.reset()  # start new pulsation cycle
                            dpg.bind_item_theme(record_audio_message_button, "my_pulsating_red_text_theme")  # tag
                            record_audio_message_tooltip.text = "Stop speaking and send to AI [Ctrl+Shift+Enter]"
                        def stop_recording_audio_message() -> None:
                            # Acknowledge in GUI
                            dpg.bind_item_theme(record_audio_message_button, "disablable_widget_theme")  # tag
                            record_audio_message_tooltip.text = "Speak to AI [Ctrl+Shift+Enter]"  # TODO: DRY the tooltip labels

                            # Stop recording (if still recording; we may have been triggered by autostop)
                            rec = audio_recorder.require()
                            logger.info("stop_recording_audio_message: Stopping audio recorder.")
                            if not rec.stop(wait=True):
                                logger.error("stop_recording_audio_message: Timed out while waiting for audio recorder to respond to stop command.")
                                return

                            # Get the captured audio
                            logger.info("stop_recording_audio_message: Getting recorded audio.")
                            audio_data = rec.get_recorded_audio()

                            # The device is free again, so the audio input panel can have it back — before
                            # the transcription below, which takes seconds during which the meter would
                            # otherwise sit dead. Monitoring keeps no audio, so it cannot disturb what we
                            # just took.
                            audio_input_panel.start_monitoring()

                            if audio_data is None:
                                logger.warning("stop_recording_audio_message: Got no audio. Cancelling.")
                                return
                            assert audio_data is not None

                            # Transcribe the audio
                            logger.info("stop_recording_audio_message: Transcribing recorded audio.")
                            user_message_text = api.stt_transcribe_array(audio_data=audio_data,
                                                                         sample_rate=rec.sample_rate,  # available after at least one recording has started
                                                                         prompt="This is a conversation between an AI and a user.")  # TODO: prompt-engineer the STT transcription prompt (e.g. detect proper names from chat log)
                            logger.info(f"Transcribed: '{user_message_text}'")  # TODO: privacy-sensitive log message? (The server has some, too.)

                            # Send the message to AI
                            logger.info("stop_recording_audio_message: Sending transcribed text to AI, as the user's message.")
                            chat_controller.chat_exchange(user_message_text)

                        # Sending is the field's own commit action *and* a global hotkey, and it has to be
                        # both. ImGui owns this chord while the field is active and will not hand it over: a
                        # multiline `InputText` natively validates on Ctrl+Enter and inserts a newline on
                        # Enter, `ctrl_enter_for_new_line` swaps the two, and validating deactivates the
                        # field — so a global handler gated on `is_item_active` or `is_item_focused` sees
                        # the chord only after the state it would have gated on is gone. Using the flag
                        # instead means the toolkit decides what "commit" is and `on_enter` reports it.
                        #
                        # But the field only commits while it holds the caret, which leaves the chord dead
                        # everywhere else — including right after a send, which parks focus on the send
                        # button. Hence the second path in `librarian_hotkeys_callback`, and `_request_send`
                        # to keep one keypress from becoming two sends when both fire.
                        #
                        # Note the widget offers exactly these two chords and no third one — Shift+Enter
                        # does nothing, whatever other chat apps have trained into the reader's fingers, so
                        # nothing user-facing should promise it.
                        #
                        # **`ctrl_enter_for_new_line` reads backwards**, so mind the mapping. It names what
                        # *Ctrl+Enter* does, not what sends: `True` means Ctrl+Enter inserts a newline and
                        # therefore **Enter sends**; `False` (ImGui's default) means Enter inserts the
                        # newline and **Ctrl+Enter sends**. Hence the comparison against `"enter"` below,
                        # which looks inverted and is not.
                        dpg.add_input_text(tag="chat_field",
                                           multiline=True,
                                           default_value="",
                                           on_enter=True,  # fire the callback on whichever chord commits
                                           # `True` <=> Enter sends; `False` (default) <=> Ctrl+Enter sends
                                           ctrl_enter_for_new_line=(librarian_config.send_message_key == "enter"),
                                           # ImGui's default for Escape is to *revert* the field to what it
                                           # held when the caret entered it, which in a composer means the
                                           # key clears or restores an older draft depending on where the
                                           # caret was last clicked — a distinction the reader cannot see.
                                           # Clearing is the one meaning worth having, and is what the help
                                           # card and this field's own tooltip promise.
                                           escape_clears_all=True,
                                           callback=lambda: _request_send(),
                                           width=_get_chat_field_base_width(),
                                           height=gui_config.chat_field_h)
                        # ImGui renders `hint` (placeholder text) for single-line inputs only, so a multiline
                        # field can't show one — the discoverability hint lives in this tooltip instead.
                        with dpg.tooltip("chat_field"):  # tag
                            dpg.add_text("Compose messages to the AI here.\n"
                                         f"    [{_newline_keys_label()}]: insert a new line\n"
                                         f"    [{_send_key_label()}]: send to the AI\n"
                                         "    [Esc]: clear the text; press again to unfocus\n"
                                         "    [Ctrl+Space]: focus this field")

                        # Staged-image thumbnail strip. Hidden until the user attaches an image; populated by the
                        # attach handler. Shown by stealing height from the text field (the composer's outer
                        # height is fixed, so the chat and avatar panels never jump when attachments appear).
                        with dpg.group(tag="chat_attachments_strip", horizontal=True, show=False):  # tag
                            pass

                        with dpg.group(horizontal=True):  # composer toolbar
                            # Attach button — documents and images. Always enabled: a document works
                            # on any model (its text is folded into the prompt), so only images depend on vision
                            # capability, and that is enforced at routing time (`_attach_callback`) rather than by
                            # disabling the button. The tooltip's image note tracks the model_is_vlm tri-state:
                            # True (confirmed VLM), None (backend didn't report — e.g. ooba, allowed on faith),
                            # False (confirmed text-only — images rejected on pick, documents still fine).
                            model_is_vlm = llm_settings.model_is_vlm
                            dpg.add_button(label=fa.ICON_PAPERCLIP,
                                           callback=show_attach_dialog,
                                           width=gui_config.toolbutton_w,
                                           tag="chat_attach_button")  # tag
                            dpg.bind_item_font("chat_attach_button", themes_and_fonts.icon_font_solid)  # tag
                            dpg.bind_item_theme("chat_attach_button", "disablable_widget_theme")  # tag
                            # One caption whichever branch runs, so the click-flash can briefly swap in an
                            # "opening…" acknowledgment and restore the help text after. Self-sizing, because
                            # that swap is a four-line caption turning into a one-line one and back.
                            if model_is_vlm is True:
                                chat_attach_caption = ("Attach file(s) to your message [Ctrl+Shift+O].\n\n"
                                                       "Documents and images are both accepted.\n"
                                                       f"    Documents: {_ATTACH_DOC_EXTS_TEXT}\n"
                                                       f"    Images: {_ATTACH_IMAGE_EXTS_TEXT}")
                            elif model_is_vlm is None:
                                chat_attach_caption = ("Attach file(s) to your message [Ctrl+Shift+O].\n\n"
                                                       f"    Documents: {_ATTACH_DOC_EXTS_TEXT}\n"
                                                       f"    Images: {_ATTACH_IMAGE_EXTS_TEXT}\n\n"
                                                       "Documents work with any model. Images require a vision model —\n"
                                                       "your LLM backend didn't report whether the loaded model can see images, so an\n"
                                                       "image is allowed on faith and the backend will error on send if it can't. LM\n"
                                                       "Studio reports the flag, so it can confirm capability up front.")
                            else:  # False — confirmed text-only; listing image formats would only offer what is refused
                                chat_attach_caption = ("Attach file(s) to your message [Ctrl+Shift+O].\n\n"
                                                       f"    Documents: {_ATTACH_DOC_EXTS_TEXT}\n\n"
                                                       "Documents work with any model. The loaded model is text-only, so\n"
                                                       "images can't be attached — load a vision model (VLM) at your LLM backend for those.")
                            chat_attach_tooltip = gui_tooltip.Tooltip("chat_attach_button", chat_attach_caption)  # tag

                            dpg.add_button(label=fa.ICON_PAPER_PLANE,
                                           callback=send_message_to_ai_callback,
                                           width=gui_config.toolbutton_w,
                                           tag="chat_send_button")  # TODO: disable this button while AI is writing
                            dpg.bind_item_font("chat_send_button", themes_and_fonts.icon_font_solid)  # tag
                            dpg.bind_item_theme("chat_send_button", "disablable_widget_theme")  # tag
                            # Self-sizing, because `_refresh_send_gate` swaps in the reason sending is
                            # blocked. A disabled button with an unchanged tooltip is a button that looks
                            # broken, and the reasons are longer than the resting caption.
                            chat_send_tooltip = gui_tooltip.Tooltip("chat_send_button", f"Send to AI [{_send_key_label()}]")  # tag

                            record_audio_message_button = dpg.add_button(label=fa.ICON_MICROPHONE,
                                                                         callback=record_audio_message_callback,
                                                                         width=gui_config.toolbutton_w,
                                                                         tag="record_audio_message_button")  # TODO: disable this button while AI is writing
                            dpg.bind_item_font("record_audio_message_button", themes_and_fonts.icon_font_solid)  # tag
                            dpg.bind_item_theme("record_audio_message_button", "disablable_widget_theme")  # tag
                            # Self-sizing: the caption says what a click does *now*, so it changes each time
                            # recording starts or stops.
                            record_audio_message_tooltip = gui_tooltip.Tooltip("record_audio_message_button",  # tag
                                                                               "Speak to AI [Ctrl+Shift+Enter]")  # TODO: DRY the tooltip labels
                            # The threshold comes from the recorder rather than from a literal here, so this
                            # line and the audio input panel's cannot disagree about where it is.
                            mic_vu_meter = DPGVUMeter(width=gui_config.vu_meter_w,
                                                      height=gui_config.vu_meter_h,
                                                      border=1,
                                                      min_value=audio_input.METER_MIN,
                                                      max_value=audio_input.METER_MAX,
                                                      yellow_start=audio_input.METER_YELLOW_START,
                                                      red_start=audio_input.METER_RED_START,
                                                      threshold_value=audio_recorder.require().silence_threshold,
                                                      tooltip_text=("Mic input level (dBFS)\n"
                                                                    f"Yellow = {audio_input.METER_YELLOW_START:0.6g}; "
                                                                    f"red = {audio_input.METER_RED_START:0.6g}; "
                                                                    "gray line = the silence threshold.\n"
                                                                    "Click the sliders button to set it [F9]."))
                            audio_recorder.require().connect_vu_readout(mic_vu_meter.update)

                            dpg.add_button(label=fa.ICON_SLIDERS,
                                           # Resolved when clicked, so the definition further down is in
                                           # place by then, as it is for the other toolbar callbacks.
                                           callback=lambda: _toggle_audio_input_panel(),
                                           width=gui_config.toolbutton_w,
                                           tag="audio_input_panel_button")  # tag
                            dpg.bind_item_font("audio_input_panel_button", themes_and_fonts.icon_font_solid)  # tag
                            dpg.bind_item_theme("audio_input_panel_button", "disablable_widget_theme")  # tag
                            gui_tooltip.Tooltip("audio_input_panel_button",  # tag
                                                "Set up the microphone [F9]\n\n"
                                                "Shows the input level, and sets how quiet counts as\n"
                                                "having stopped speaking. Worth doing in the room the\n"
                                                "system will be used in — that is what decides the number.")

            with dpg.group(tag="avatar_column"):  # tag  # right column: the avatar, or the chat graph in its place
                avatar_panel_w, avatar_panel_h = _get_avatar_panel_base_size()
                with dpg.child_window(tag="avatar_panel",
                                      width=avatar_panel_w,
                                      height=avatar_panel_h,
                                      no_scrollbar=True,
                                      no_scroll_with_mouse=True):
                    dpg_avatar_renderer = DPGAvatarRenderer(gui_parent="avatar_panel",
                                                            avatar_x_center=(avatar_panel_w // 2),
                                                            avatar_y_bottom=(avatar_panel_h - 8),
                                                            paused_text="[Video is off]",
                                                            executor=bg)
                    # DRY, just so that `_load_initial_animator_settings` at app bootup is guaranteed to use the same values
                    _initial_image_size = int(librarian_config.avatar_config.animator_settings_overrides["upscale"] * librarian_config.avatar_config.source_image_size)
                    dpg_avatar_renderer.configure_live_texture(_initial_image_size, _initial_image_size)

                    # Status indicators stack top-down via a vertical parent group anchored at (16, 16).
                    # Order — INDEXING, DOCS, READING, SYSTEM, WEB — mirrors the typical processing order
                    # of a query and places the longest-lived indicator (INDEXING) at the top so it stays
                    # in place when shorter-lived siblings appear below it. READING sits where it does
                    # because extracting an attachment's text is what happens between finding documents
                    # and handing the prompt to the backend. DPG's vertical group naturally
                    # hides any child whose `show=False`, so the visible siblings just repack with no
                    # overlap. Indexing and search have separate widgets — they can run concurrently
                    # (since the lock granularity work), so they're independent indicators rather than
                    # two states of one.
                    with dpg.group(pos=(16, 16)):
                        with dpg.group(show=False, horizontal=True) as docs_indexing_indicator_group:
                            dpg.add_text(fa.ICON_DATABASE, tag="docs_indexing_symbol")
                            dpg.bind_item_font("docs_indexing_symbol", themes_and_fonts.icon_font_solid)  # tag
                            dpg.bind_item_theme("docs_indexing_symbol", "my_pulsating_red_docs_theme")  # tag
                            dpg.add_text("INDEXING", tag="docs_indexing_text")
                            dpg.bind_item_theme("docs_indexing_text", "my_pulsating_red_docs_theme")  # tag
                            # Steady-red (non-pulsating) theme on the long progress label — pulsation
                            # kills readability for a label this long.
                            dpg.add_text("", tag="docs_indexing_progress_text")
                            dpg.bind_item_theme("docs_indexing_progress_text", "my_steady_red_docs_theme")  # tag

                        with dpg.group(show=False, horizontal=True) as docs_search_indicator_group:
                            dpg.add_text(fa.ICON_DATABASE, tag="docs_search_symbol")
                            dpg.bind_item_font("docs_search_symbol", themes_and_fonts.icon_font_solid)  # tag
                            dpg.bind_item_theme("docs_search_symbol", "my_pulsating_gray_text_theme")  # tag
                            dpg.add_text("DOCS", tag="docs_search_text")
                            dpg.bind_item_theme("docs_search_text", "my_pulsating_gray_text_theme")  # tag
                            dpg.add_text("", tag="docs_search_progress_text")
                            dpg.bind_item_theme("docs_search_progress_text", "my_steady_gray_docs_theme")  # tag

                        with dpg.group(show=False, horizontal=True) as attachment_read_indicator_group:
                            dpg.add_text(fa.ICON_BOOK_OPEN_READER, tag="attachment_read_symbol")
                            dpg.bind_item_font("attachment_read_symbol", themes_and_fonts.icon_font_solid)  # tag
                            dpg.bind_item_theme("attachment_read_symbol", "my_pulsating_gray_text_theme")  # tag
                            dpg.add_text("READING", tag="attachment_read_text")
                            dpg.bind_item_theme("attachment_read_text", "my_pulsating_gray_text_theme")  # tag

                        with dpg.group(show=False, horizontal=True) as llm_indicator_group:
                            dpg.add_text(fa.ICON_MICROCHIP, tag="llm_prompt_process_symbol")
                            dpg.bind_item_font("llm_prompt_process_symbol", themes_and_fonts.icon_font_solid)  # tag
                            dpg.bind_item_theme("llm_prompt_process_symbol", "my_pulsating_gray_text_theme")  # tag
                            dpg.add_text("SYSTEM", tag="llm_prompt_process_text")

                        with dpg.group(show=False, horizontal=True) as web_indicator_group:
                            dpg.add_text(fa.ICON_GLOBE, tag="web_access_symbol")
                            dpg.bind_item_font("web_access_symbol", themes_and_fonts.icon_font_solid)  # tag
                            dpg.bind_item_theme("web_access_symbol", "my_pulsating_gray_text_theme")  # tag
                            dpg.add_text("WEB", tag="web_access_text")

                    dpg.add_text("",
                                 pos=(gui_config.subtitle_x0,
                                      _get_subtitle_bottom_y0(avatar_panel_h)),  # Position doesn't really matter; the text is empty for now, and will be re-positioned when subtitles are generated.
                                 color=gui_config.subtitle_color,
                                 wrap=(avatar_panel_w - 16) - gui_config.subtitle_x0 - gui_config.subtitle_text_wrap_margin,
                                 tag="avatar_subtitle_text")
                    dpg.bind_item_font("avatar_subtitle_text", subtitle_font)  # tag

                # The chat graph shares the avatar's rect exactly, as a second child window shown in its
                # place rather than as an overlay with its own z-order. One rect with alternative
                # occupants: the same mechanism a no-avatar mode will use, where this one simply never
                # hides.
                #
                # `input_blocked` goes in as a lambda because `is_any_modal_window_visible` is defined
                # further down this module and the GUI is built at module scope, so the name does not exist
                # yet. The lambda defers the lookup to the first mouse event, by which time it does. Same
                # for `chat_controller`, which is built after the GUI it drives.
                def switch_to_chat_node(node_id: str) -> None:
                    """Move HEAD to `node_id` — the graph view's commit gesture."""
                    if datastore.get_parent(node_id) is None:
                        # A root is a character card, not a message, and HEAD resting on one is the single
                        # state the chat view can show nothing from: it builds upward from HEAD, so the
                        # chat under a card falls out of sight. The per-message branch button refuses this
                        # for the same reason.
                        logger.info(f"switch_to_chat_node: '{node_id}' is a system prompt node; not moving HEAD there")
                        return
                    if app_state["HEAD"] == node_id:
                        return
                    app_state["HEAD"] = node_id
                    # The same discontinuity a sibling switch is, and usually a larger one: a jump taken
                    # from the graph can cross to a conversation the avatar was never in.
                    chat_controller.mark_discontinuity()
                    chat_controller.view.build()

                chat_graph_panel = chatgraph_panel.DPGChatGraphPanel(
                    gui_parent="avatar_column",  # tag
                    datastore=datastore,
                    app_state=app_state,
                    themes_and_fonts=themes_and_fonts,
                    width=avatar_panel_w,
                    height=avatar_panel_h,
                    on_preview=lambda node_id: chat_controller.view.jump_to_node(node_id),
                    on_commit=lambda node_id: switch_to_chat_node(node_id),
                    input_blocked=lambda: is_any_modal_window_visible(),
                    # Through a lambda, as `input_blocked` above is and for the same reason: both are
                    # defined further down this module than the panel is built.
                    on_focus_requested=lambda: _give_keyboard_to_graph(),
                    # `None` when the setting is off, which is what gives the label back the room the
                    # glyph's gutter reserves -- the panel asks per rebuild, so the switch takes
                    # effect on the next one rather than needing the app restarted.
                    icon_for=(lambda: chat_controller.icon_texture_for
                              if gui_config.chat_graph_role_icons else None),
                    thumbnail_for=lambda name, size: chat_controller.get_graph_thumbnail_texture(name, size),
                    show=False)

                with dpg.child_window(tag="mode_toggle_controls",
                                      width=-1,
                                      height=gui_config.chat_controls_h,
                                      no_scrollbar=True,
                                      no_scroll_with_mouse=True):
                    with dpg.group(horizontal=True):
                        def toggle_internet_enabled():
                            app_state["internet_enabled"] = not app_state["internet_enabled"]
                        def toggle_docs_enabled():
                            app_state["docs_enabled"] = not app_state["docs_enabled"]
                        def toggle_speech_enabled():
                            app_state["avatar_speech_enabled"] = not app_state["avatar_speech_enabled"]
                        def toggle_subtitles_enabled():
                            app_state["avatar_subtitles_enabled"] = not app_state["avatar_subtitles_enabled"]
                            avatar_controller.subtitles_enabled = app_state["avatar_subtitles_enabled"]
                        def toggle_chat_graph():
                            # The checkbox is a *preference*, not the panel's state: the graph also stands
                            # in whenever the avatar has nothing to show. So this records what the user
                            # asked for and lets the one writer work out what that means right now.
                            app_state["chat_graph_shown"] = dpg.get_value("chat_graph_checkbox")  # tag
                            if not app_state["chat_graph_shown"]:
                                # Switching the graph off is a request for the avatar, and a request for
                                # the avatar is activity — so it wakes one that had gone to sleep, exactly
                                # as clicking into the chat would.
                                #
                                # Without this the switch does nothing whenever the avatar had already
                                # idled out *before* the graph took the panel: the graph is then standing
                                # in for an absence rather than covering anything, so unchecking recomputes
                                # to "still no video, keep the graph". The state that decides which of
                                # those two happened is invisible to the user, so the same gesture would
                                # work or not for reasons nobody can see.
                                #
                                # Here rather than in `_apply_panel_occupancy`, which runs five times a
                                # second: a ping there would keep the avatar awake forever.
                                avatar_controller.ping(avatar_record)
                            _apply_panel_occupancy()
                        def toggle_show_thinking():
                            app_state["show_thinking"] = not app_state["show_thinking"]
                        def toggle_thinking_enabled():
                            app_state["thinking_enabled"] = not app_state["thinking_enabled"]

                        # Three groups, divided: what the AI does when it answers, how the chat log is
                        # shown, and what the avatar does. They answer different questions of the user, and
                        # in a flat row the display preference reads as one more thing the AI does.
                        dpg.add_checkbox(label="Thinking", default_value=app_state["thinking_enabled"], callback=toggle_thinking_enabled, tag="thinking_enabled_checkbox")
                        dpg.add_tooltip("thinking_enabled_checkbox", tag="thinking_enabled_tooltip")  # tag
                        dpg.add_text("Let a thinking model reason before it answers. [Alt+T]\n\nWith this off, the same model answers immediately: replies arrive sooner\nand are shorter. Reasoning is how these models work through a hard\nquestion, so switching it off trades accuracy on those for speed.\n\nDoes nothing to a model that does not reason.\n\nTakes effect from the AI's next chat message onward.", parent="thinking_enabled_tooltip")  # tag

                        dpg.add_checkbox(label="Internet", default_value=app_state["internet_enabled"], callback=toggle_internet_enabled, tag="internet_enabled_checkbox")
                        dpg.add_tooltip("internet_enabled_checkbox", tag="internet_enabled_tooltip")  # tag
                        dpg.add_text("Let the AI reach the internet: web search, and fetching a page it finds\nor that you link to. [Alt+I]\n\nThis is the only switch that sends anything to a site the AI picked, so it\nis the one to turn off when the conversation should stay between you and\nthe machines you run. Your messages always go to the LLM backend you\nconfigured, and searching your documents may reach the Raven-server you\nconfigured; either of those can sit elsewhere on your network. Both are\nset in the config file, not here.\n\nWith this off, the AI can still read your document database (see next\ntoggle), do arithmetic, and ask what time it is.", parent="internet_enabled_tooltip")  # tag

                        dpg.add_checkbox(label="Documents", default_value=app_state["docs_enabled"], callback=toggle_docs_enabled, tag="docs_enabled_checkbox")
                        dpg.add_tooltip("docs_enabled_checkbox", tag="docs_enabled_tooltip")  # tag
                        dpg.add_text("Before responding, search document database for relevant information. [Alt+D]\nAlso lets the AI search the database itself; with this off, the document\ntools are not offered at all.\n\nWhile on, the AI is asked to ground claims about your documents in what\nwas actually retrieved, and any reply that got nothing to stand on is\nmarked [no sources retrieved].\n\nThe search always injects its best matches, even when the topic is not\nin the database and those matches are noise. That costs prompt-processing\ntime before each reply, so it is worth switching off while discussing\nsomething the database does not cover.", parent="docs_enabled_tooltip")  # tag

                        # No line, matching the toolbar below the chat, which separates its sections by
                        # spacing alone at every one of its call sites.
                        guiutils.add_toolbar_separator(horizontal=True,
                                                       toolbar_extent=gui_config.mode_toggle_row_h,
                                                       size=gui_config.toolbar_separator_w,
                                                       line=False)

                        dpg.add_checkbox(label="Show thinking", default_value=app_state["show_thinking"], callback=toggle_show_thinking, tag="show_thinking_checkbox")
                        dpg.add_tooltip("show_thinking_checkbox", tag="show_thinking_tooltip")  # tag
                        dpg.add_text("Start a thinking model's reasoning trace open instead of collapsed. [Alt+Shift+T]\n\nThis is about what you see, not about whether the AI reasons at all -\nthat is the Thinking switch, at the left of this row.\n\nTakes effect from the AI's next chat message onward. For a reply already\non screen, the cloud beside it opens its trace - or press Ctrl+T.", parent="show_thinking_tooltip")  # tag

                        # No line, matching the toolbar below the chat, which separates its sections by
                        # spacing alone at every one of its call sites.
                        guiutils.add_toolbar_separator(horizontal=True,
                                                       toolbar_extent=gui_config.mode_toggle_row_h,
                                                       size=gui_config.toolbar_separator_w,
                                                       line=False)

                        # A group of its own, between how the chat log is shown and what the avatar does,
                        # because it is neither: it governs *which panel* fills the right-hand side. That
                        # is the same widening a no-avatar mode makes — the panel is what a mode varies,
                        # and the avatar is one thing that can fill it. Grouped with Speech and Subtitles
                        # it read as a third avatar control, which is the one thing it is not.
                        dpg.add_checkbox(label="Chat graph",
                                         default_value=app_state["chat_graph_shown"],
                                         callback=toggle_chat_graph, tag="chat_graph_checkbox")  # tag
                        dpg.add_tooltip("chat_graph_checkbox", tag="chat_graph_tooltip")  # tag
                        dpg.add_text("Show the chat tree in place of the avatar. [Alt+G]\n\n"
                                     "Every chat ever started is in there, branching. Clicking a message\n"
                                     "shows it; clicking it again switches the conversation to it, so you\n"
                                     "can look around without changing anything.\n\n"
                                     "The avatar's video pauses while it is covered, and switching this\n"
                                     "off wakes it, even if it had gone to sleep meanwhile. With this off,\n"
                                     "the graph still stands in whenever the avatar has nothing to show:\n"
                                     "while its video starts up, and once it switches itself off.",
                                     parent="chat_graph_tooltip")  # tag

                        # No line, matching the toolbar below the chat, which separates its sections by
                        # spacing alone at every one of its call sites.
                        guiutils.add_toolbar_separator(horizontal=True,
                                                       toolbar_extent=gui_config.mode_toggle_row_h,
                                                       size=gui_config.toolbar_separator_w,
                                                       line=False)

                        dpg.add_checkbox(label="Speech", default_value=app_state["avatar_speech_enabled"], callback=toggle_speech_enabled, tag="speech_enabled_checkbox")
                        dpg.add_tooltip("speech_enabled_checkbox", tag="speech_enabled_tooltip")  # tag
                        dpg.add_text("Have the avatar speak the final response (TTS, text to speech). [Alt+S]", parent="speech_enabled_tooltip")  # tag

                        dpg.add_checkbox(label="Subtitles", default_value=app_state["avatar_subtitles_enabled"], callback=toggle_subtitles_enabled, tag="avatar_subtitles_checkbox")
                        dpg.add_tooltip("avatar_subtitles_checkbox", tag="subtitles_enabled_tooltip")  # tag
                        if gui_config.translator_target_lang is not None:
                            subtitle_explanation_str = f"Subtitle the avatar's speech (language: {gui_config.translator_target_lang.upper()})."
                        else:
                            subtitle_explanation_str = "Closed-caption (CC) the avatar's speech."
                        dpg.add_text(f"{subtitle_explanation_str} [Alt+C]\nUsed when TTS is ON.\nTakes effect from the AI's next chat message onward.", parent="subtitles_enabled_tooltip")  # tag

                    # Utility actions — one-shot actions, kept a visually distinct group from the
                    # persistent-state toggles above (their own rows, under a separator). The panel below the
                    # avatar has room to grow this into a collapsing header if more tools land here later.
                    dpg.add_separator()
                    with dpg.group(horizontal=True):
                        dpg.add_text("Open folder:")

                        def _make_open_folder_callback(*, get_dir, button_tag, tooltip, ok_message, ensure_exists=False):
                            """Build a click callback that opens a directory in the file manager, flashing the button on success/failure.

                            `get_dir` is called at click time (so a value like the active datastore path is read fresh, not
                            captured at GUI-build time). `ensure_exists` creates the directory first — for the documents drop
                            folder, which may not exist yet on a fresh install."""
                            def callback() -> None:
                                # The `try` covers opening the folder and nothing else, so that a fault in
                                # the acknowledgment is not caught here and reported as the folder having
                                # failed to open. One flash, outside it, for the same reason: a broken
                                # acknowledgment reported through a second acknowledgment breaks twice and
                                # says so once. Same shape as `chat_controller`'s action buttons.
                                try:
                                    directory = get_dir()
                                    if ensure_exists:
                                        common_utils.create_directory(directory)
                                    common_utils.open_in_file_manager(directory)
                                    ok, message = True, ok_message
                                except Exception as exc:  # noqa: BLE001 -- opening a folder must never crash the GUI
                                    logger.error(f"open-folder utility ({button_tag}): {type(exc)}: {exc}")
                                    ok, message = False, "Couldn't open folder"
                                gui_animation.flash_button(button=button_tag, tooltip=tooltip,
                                                           ok=ok, message=message, duration=gui_config.acknowledgment_duration)
                            return callback

                        # The callback is bound after the button rather than at creation, because it flashes
                        # a tooltip that does not exist until the button it belongs to does.
                        dpg.add_button(label=fa.ICON_FOLDER_TREE,
                                       width=gui_config.toolbutton_w,
                                       tag="util_open_docs_dir_button")  # tag
                        dpg.bind_item_font("util_open_docs_dir_button", themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("util_open_docs_dir_button", "disablable_widget_theme")  # tag
                        util_open_docs_dir_tooltip = gui_tooltip.Tooltip("util_open_docs_dir_button",  # tag
                                                                          "Open the documents folder\n(drop files in this folder for the AI to search)")
                        dpg.set_item_callback("util_open_docs_dir_button",  # tag
                                              _make_open_folder_callback(get_dir=lambda: librarian_config.llm_docs_dir,
                                                                         button_tag="util_open_docs_dir_button",
                                                                         tooltip=util_open_docs_dir_tooltip,
                                                                         ok_message="Opened documents folder",
                                                                         ensure_exists=True))

                        dpg.add_button(label=fa.ICON_DATABASE,
                                       width=gui_config.toolbutton_w,
                                       tag="util_open_datastore_dir_button")  # tag
                        dpg.bind_item_font("util_open_datastore_dir_button", themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("util_open_datastore_dir_button", "disablable_widget_theme")  # tag
                        util_open_datastore_dir_tooltip = gui_tooltip.Tooltip("util_open_datastore_dir_button",  # tag
                                                                               "Open the chat data folder\n(chat history + attached files)")
                        dpg.set_item_callback("util_open_datastore_dir_button",  # tag
                                              _make_open_folder_callback(get_dir=lambda: pathlib.Path(chat_controller.datastore.datastore_file).expanduser().resolve().parent,
                                                                         button_tag="util_open_datastore_dir_button",
                                                                         tooltip=util_open_datastore_dir_tooltip,
                                                                         ok_message="Opened chat data folder"))

                    # A destructive action gets its own row rather than a third seat on the folder row above:
                    # that row's label would start lying, and "delete things forever" should not sit a few
                    # pixels from two buttons whose worst outcome is a file manager opening.
                    with dpg.group(horizontal=True):
                        dpg.add_text("Maintenance:")

                        dpg.add_button(label=fa.ICON_BROOM,
                                       callback=lambda: cleanup_dialog.open(),
                                       width=gui_config.toolbutton_w,
                                       tag="util_cleanup_button")  # tag
                        dpg.bind_item_font("util_cleanup_button", themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("util_cleanup_button", "disablable_widget_theme")  # tag
                        util_cleanup_tooltip = gui_tooltip.Tooltip("util_cleanup_button",  # tag
                                                                    "Clean up and save the chat data\n(shows what would be deleted first)")

                    # Raven-server's status, on the same pattern as the composer's LLM-backend row and for
                    # the same reason: a server that has gone away is something the user has to be told,
                    # since nothing else in the window says so until they try to use it and it fails.
                    #
                    # Here rather than in either panel above, and that placement is the whole trick: this
                    # column sits *outside* the rect the avatar and the chat graph take turns in, so the row
                    # is visible whichever of them holds it — no occupancy test, and no reparenting a widget
                    # between two containers as the panel changes hands.
                    #
                    # Shown only when something is wrong, plus a moment on the way back. A row that said
                    # "connected" all day would be teaching the user to stop reading it.
                    # Parked at the bottom of the panel rather than following the utility rows, which leaves
                    # the empty middle as the gap between them: the row is not a third utility action and
                    # should not read as one.
                    #
                    # Positioned rather than spaced, and set once rather than tracked, because this child
                    # window's *height* is the fixed `chat_controls_h` while only its width follows the
                    # window. Right-aligning would be the other story entirely — the width does move, and
                    # the pill's own width differs between its two labels, so it would want measuring after
                    # every status change.
                    with dpg.group(tag="server_status_pill", show=False):  # tag
                        with dpg.group(horizontal=True):
                            dpg.add_text(fa.ICON_PLUG_CIRCLE_XMARK, tag="server_status_icon")  # tag
                            dpg.bind_item_font("server_status_icon", themes_and_fonts.icon_font_solid)  # tag
                            dpg.add_button(label="",
                                           callback=lambda: _request_server_recheck(),
                                           tag="server_status_button")  # tag
                            # Self-sizing, for the same reason as the backend row's: the caption is
                            # rewritten on every status change and again by the click flash, and a
                            # `dpg.tooltip` would be drawn at its previous size each time that happened.
                            server_status_tooltip = gui_tooltip.Tooltip("server_status_button", "")  # tag
                    # Bottom edge, less one row and the padding the child window keeps below it. Outside the
                    # group's own `with`, so it applies to the group rather than to a member of it.
                    dpg.set_item_pos("server_status_pill",  # tag
                                     (guiutils.DPG_WINDOW_PADDING,
                                      gui_config.chat_controls_h - gui_config.mode_toggle_row_h - 2 * guiutils.DPG_WINDOW_PADDING))

        # The bottom row is split into two child windows that mirror the panels above them: the chat-side
        # buttons sit under the chat panel, the AI-disclosure label under the avatar panel. Splitting is what
        # makes the label centerable at all - in one full-width row its position depended on the total width
        # of everything to its left, including the variable-width context-fill readout, so it drifted.
        with dpg.group(horizontal=True):
            with dpg.child_window(tag="chat_global_buttons",  # tag
                                  width=gui_config.chat_panel_w,
                                  height=gui_config.ai_warning_h,
                                  no_scrollbar=True,
                                  no_scroll_with_mouse=True):
                with dpg.group(horizontal=True):
                    def add_separator(*, width=None, line=True, line_offset=None):
                        if width is None:
                            width = gui_config.toolbar_separator_w
                        guiutils.add_toolbar_separator(horizontal=True,
                                                       toolbar_extent=gui_config.toolbar_inner_h,
                                                       size=width, line=line,
                                                       line_offset=line_offset)
                    if gui_config.toolbutton_indent is None:
                        toolbutton_h = gui_config.toolbutton_w  # square buttons
                        gui_config.toolbutton_indent = (gui_config.toolbar_inner_h - toolbutton_h) // 2  # pixels, to center the buttons

                    def start_new_chat_callback() -> None:
                        new_chat_head_node_id = app_state["new_chat_HEAD"]
                        app_state["HEAD"] = new_chat_head_node_id
                        # The same discontinuity a branch switch is, and a larger one: a sibling switch
                        # swaps one reply for its alternative, while this drops the whole conversation the
                        # avatar was in.
                        chat_controller.mark_discontinuity()
                        chat_controller.view.build()
                        dpg.focus_item("chat_field")  # tag  # Focus the chat field for convenience, since the whole point of a new chat is to immediately start a new conversation.
                        # Acknowledge the action in the GUI.
                        gui_animation.flash_button(button=new_chat_button,
                                                   message="New chat started!",
                                                   duration=gui_config.acknowledgment_duration,
                                                   tooltip=new_chat_tooltip)

                    def copy_chatlog_to_clipboard_as_markdown_callback() -> None:
                        shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
                        if (chatlog_text := chat_controller.view.get_chatlog_as_markdown(include_metadata=shift_pressed)) is not None:
                            dpg.set_clipboard_text(chatlog_text)
                        # Acknowledge the action in the GUI.
                        mode = "with node IDs" if shift_pressed else "as-is"
                        gui_animation.flash_button(button=copy_chat_button,
                                                   message=f"Copied to clipboard! ({mode})",
                                                   duration=gui_config.acknowledgment_duration,
                                                   tooltip=copy_chat_tooltip)

                    def stop_text_generation_callback() -> None:
                        chat_controller.stop_ai_turn()
                        # Acknowledge the action in the GUI.
                        gui_animation.flash_button(button=stop_generation_button,
                                                   message="Interrupted!",
                                                   duration=gui_config.acknowledgment_duration,
                                                   tooltip=stop_generation_tooltip)

                    def stop_speech_callback() -> None:
                        avatar_controller.stop_tts()
                        # Acknowledge the action in the GUI.
                        gui_animation.flash_button(button=stop_speech_button,
                                                   message="Stopped speaking!",
                                                   duration=gui_config.acknowledgment_duration,
                                                   tooltip=stop_speech_tooltip)

                    def toggle_fullscreen():
                        dpg.toggle_viewport_fullscreen()
                        resize_gui()

                    new_chat_button = dpg.add_button(label=fa.ICON_FILE,
                                                     callback=start_new_chat_callback,
                                                     width=gui_config.toolbutton_w,
                                                     tag="chat_new_button")
                    dpg.bind_item_font("chat_new_button", themes_and_fonts.icon_font_solid)  # tag
                    dpg.bind_item_theme("chat_new_button", "disablable_widget_theme")  # tag
                    new_chat_tooltip = gui_tooltip.Tooltip("chat_new_button", "Start new chat [Ctrl+N]")  # tag

                    add_separator(line=False)

                    copy_chat_button = dpg.add_button(label=fa.ICON_COPY,
                                                      callback=copy_chatlog_to_clipboard_as_markdown_callback,
                                                      width=gui_config.toolbutton_w,
                                                      tag="chat_copy_to_clipboard_button")
                    dpg.bind_item_font("chat_copy_to_clipboard_button", themes_and_fonts.icon_font_solid)  # tag
                    dpg.bind_item_theme("chat_copy_to_clipboard_button", "disablable_widget_theme")  # tag
                    copy_chat_tooltip = gui_tooltip.Tooltip("chat_copy_to_clipboard_button",  # tag
                                                             "Copy this conversation to clipboard [F8]\n    no modifier: as-is\n    with Shift: include message node IDs")

                    stop_generation_button = dpg.add_button(label=fa.ICON_SQUARE,
                                                            callback=stop_text_generation_callback,
                                                            enabled=False,
                                                            width=gui_config.toolbutton_w,
                                                            tag="chat_stop_generation_button")
                    dpg.bind_item_font("chat_stop_generation_button", themes_and_fonts.icon_font_solid)  # tag
                    dpg.bind_item_theme("chat_stop_generation_button", "disablable_widget_theme")  # tag
                    stop_generation_tooltip = gui_tooltip.Tooltip("chat_stop_generation_button",  # tag
                                                                   "Interrupt the AI [Ctrl+G]\nThis stops the AI when it is writing.")

                    stop_speech_button = dpg.add_button(label=fa.ICON_COMMENT_SLASH,
                                                        callback=stop_speech_callback,
                                                        enabled=False,
                                                        width=gui_config.toolbutton_w,
                                                        tag="chat_stop_speech_button")
                    dpg.bind_item_font("chat_stop_speech_button", themes_and_fonts.icon_font_solid)  # tag
                    dpg.bind_item_theme("chat_stop_speech_button", "disablable_widget_theme")  # tag
                    stop_speech_tooltip = gui_tooltip.Tooltip("chat_stop_speech_button", "Stop speaking [Ctrl+S]")  # tag

                    add_separator(line=False)

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
                        dpg.add_text("Open the help card [F1]",
                                     tag="help_tooltip_text")

                    add_separator(line=False)

                    # Context-fill readout: how full the model's loaded context window is for the current chat.
                    # Updated by `DPGChatController.update_context_fill_indicator`. Provisional placement in the
                    # bottom toolbar; revisit when the multiline input / file-upload work reshapes this area.
                    context_fill_text_widget = dpg.add_text("", color=(160, 160, 160), tag="context_fill_text")  # tag
                    with dpg.tooltip("context_fill_text"):  # tag
                        dpg.add_text("Conversation size vs the model's loaded context window.\n"
                                     "A leading '~' means an estimate (no exact tokenizer configured for this backend).",
                                     tag="context_fill_tooltip_text")

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
                    #     gui_animation.flash_button(button=testing_button,
                    #                                message="Ran the action being tested!",
                    #                                duration=gui_config.acknowledgment_duration,
                    #                                tooltip=testing_tooltip,
                    #                                text=testing_tooltip_text)
                    # testing_button = dpg.add_button(label=fa.ICON_VOLCANO,
                    #                                 callback=testing_callback,
                    #                                 width=gui_config.toolbutton_w,
                    #                                 tag="chat_testing_button")
                    # dpg.bind_item_font("chat_testing_button", themes_and_fonts.icon_font_solid)  # tag
                    # dpg.bind_item_theme("chat_testing_button", "disablable_widget_theme")  # tag
                    # testing_tooltip = dpg.add_tooltip("chat_testing_button")  # tag
                    # testing_tooltip_text = dpg.add_text("Developer button for testing purposes. What will it do today?!", parent=testing_tooltip)

            # The AI-disclosure label, in its own child window under the avatar panel so it can be centered there.
            #
            # The first clause is the disclosure proper: EU AI Act Article 50(1) asks that a person be told they are
            # interacting with an AI system, and its exception for cases where that is obvious is to be read narrowly -
            # so state it outright rather than leaving it to be inferred from "the connected AI". The second clause is
            # the older quality warning, which is good practice but not the disclosure. Deliberately always visible and
            # not dismissable: "at the start of the first interaction" is then satisfied trivially, and there is no way
            # to configure the app out of compliance.
            with dpg.child_window(tag="ai_warning_panel",  # tag
                                  width=_get_avatar_panel_base_size()[0],
                                  height=gui_config.ai_warning_h,
                                  no_scrollbar=True,
                                  no_scroll_with_mouse=True):
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=0, tag="ai_warning_centering_spacer")  # tag  # width set by `_center_ai_warning`
                    with dpg.group(horizontal=True, tag="ai_warning_block"):  # tag  # measured by `_center_ai_warning`
                        # The label wraps to two lines, so the icon has to drop to the block's vertical center
                        # rather than sit on the first line. Half a line would be 10 px at font size 20, but the
                        # triangle glyph reads a touch low there - its ink sits lower in the em box than the text's
                        # does - so 9 px is what actually looks centered. A vertical group adds item_spacing_y
                        # (4 px) after the spacer, so the spacer itself supplies the remaining 5.
                        with dpg.group():
                            dpg.add_spacer(height=5)
                            dpg.add_text(fa.ICON_TRIANGLE_EXCLAMATION, color=_CAUTION_COLOR, tag="ai_warning_icon")
                        dpg.add_text(_AI_WARNING_TEXT,
                                     color=_CAUTION_COLOR,
                                     wrap=gui_config.ai_warning_w,
                                     tag="ai_warning_text")
        dpg.bind_item_font("ai_warning_icon", themes_and_fonts.icon_font_solid)  # tag

# --------------------------------------------------------------------------------
# Animations, live updates

# Whether the composer held the caret on the previous frame; see `update_animations`.
_composer_was_active = False
def update_animations():
    gui_animation.animator.render_frame()
    # Mirror the retriever's progress-text channels (indexing + search) into their DPG widgets.
    # Indicator visibility is push-driven via callbacks; only the progress text strings are polled.
    chat_controller.update_docs_indicator_progress_text()
    # The jump-to-latest pill must be polled rather than pushed: the mouse wheel and the scrollbar move the
    # chat panel from inside ImGui and raise nothing we could hook, so "the reader has left the end" is only
    # observable by looking. See `DPGLinearizedChatView.update_jump_to_latest_pill`.
    chat_controller.view.update_jump_to_latest_pill()
    # The live thinking counter is drawn on the frame clock rather than as chunks arrive, so that it ticks
    # at a steady rate the reader can trust. See `DPGChatController.update_thinking_readout`.
    chat_controller.update_thinking_readout()
    # Which message the per-message hotkeys act on follows the scroll position, and is polled for the same
    # reason the pill is: nothing raises an event when the reader wheels the panel.
    chat_controller.update_current_message_mark()
    # Whether a turn is in flight gates sending, and is polled for the same reason: the turn starts and
    # finishes on a background task, which raises nothing this module hooks. The call is a comparison
    # unless the answer actually changed.
    _refresh_send_gate()
    # Who holds the keyboard is derived everywhere except one bit -- whether the graph has it -- which DPG
    # cannot answer and so is stored. That bit goes stale the moment the composer is entered by a route
    # that does not pass through `_cycle_keyboard_home`: a click on the field, or Ctrl+Space. Polled for
    # the reason its neighbours are, neither route raising anything this module hooks.
    #
    # Stale, it showed as two panes wearing the blue at once, which is not a state the design has: the
    # composer owns the caret, and the graph's bit is meaningful only while it does not.
    # Two frames, not one, and the second is what makes this correct rather than merely cautious. Tab
    # reaches our handler *and* still moves ImGui's nav, and the item it lands on reports itself activated
    # and then deactivated a frame later -- see `dpg-notes.md`, "Tab reaches a global handler and still
    # moves ImGui's nav". Read on a single frame, that phantom says the composer holds the caret when the
    # reader has just Tabbed *away* from it, and the graph's bit is revoked 16 ms after Tab granted it.
    # A genuine click or Ctrl+Space leaves the field active until something takes it away, so requiring
    # the answer twice running costs a frame nobody can see and rejects the transient outright.
    global _composer_was_active
    composer_active = dpg.is_item_active("chat_field")  # tag
    if chat_graph_panel.has_keyboard and composer_active and _composer_was_active:
        chat_graph_panel.has_keyboard = False
    _composer_was_active = composer_active

# --------------------------------------------------------------------------------
# Built-in help window

hotkey_info = (env(key_indent=0, key="Ctrl+Space", action_indent=0, action="Focus the message composer", notes=""),
               env(key_indent=0, key=_send_key_label(), action_indent=0, action="Send the message to the AI", notes="Empty message = the AI adds a reply"),
               env(key_indent=1, key=_newline_keys_label(), action_indent=0, action="Insert a new line", notes="While writing a message"),
               env(key_indent=1, key="Esc", action_indent=0, action="Clear the text", notes="While writing. Again to unfocus"),
               # No device name here, deliberately. This tuple is built once at import and `helpcard` renders
               # its table once, so anything interpolated into it freezes at startup — and the microphone
               # *can* change later, from the F9 panel. A card confidently naming the wrong device is worse
               # than one naming none, and the panel shows the current one live, a row below.
               env(key_indent=0, key="Ctrl+Shift+Enter", action_indent=0, action="Speak to the AI using your mic", notes=""),
               env(key_indent=1, key="F9", action_indent=0, action="Set up the microphone", notes="Device, input level and auto-off"),
               env(key_indent=2, key="D", action_indent=1, action="Choose the microphone", notes="Then Up, Down, Home, End, Esc"),
               env(key_indent=2, key="M", action_indent=1, action="Measure the room", notes=""),
               env(key_indent=2, key="A", action_indent=1, action="Measure at each recording", notes=""),
               env(key_indent=2, key="S", action_indent=1, action="Toggle stop on silence", notes=""),
               env(key_indent=2, key="R", action_indent=1, action="Reset to defaults", notes=""),
               env(key_indent=2, key="Esc", action_indent=1, action="Close the panel", notes="Twice if chooser focused"),
               env(key_indent=0, key="Ctrl+Shift+O", action_indent=0, action="Attach one or more files", notes="Documents; and images on a VLM"),
               helpcard.hotkey_blank_entry,
               env(key_indent=0, key="Ctrl+T", action_indent=0, action="Thinking trace of marked message", notes="For thinking models"),
               env(key_indent=0, key="Ctrl+S", action_indent=0, action="Start/stop AI speaking", notes="The blue mark shows which message"),
               env(key_indent=0, key="Ctrl+Right", action_indent=0, action="Next sibling of marked message", notes=""),
               env(key_indent=1, key="Ctrl+Shift+Right", action_indent=1, action="Same, but jump 10", notes=""),
               env(key_indent=1, key="Ctrl+End", action_indent=1, action="Same, but to the last", notes=""),
               env(key_indent=0, key="Ctrl+Left", action_indent=0, action="Previous sibling of marked message", notes=""),
               env(key_indent=1, key="Ctrl+Shift+Left", action_indent=1, action="Same, but jump 10", notes=""),
               env(key_indent=1, key="Ctrl+Home", action_indent=1, action="Same, but to the first", notes=""),
               env(key_indent=0, key="Ctrl+Down", action_indent=0, action="Show the chat continuation", notes="If any exists in chat datastore"),
               env(key_indent=0, key="Ctrl+B", action_indent=0, action="Branch the chat here", notes="Rolls back. Not while typing"),
               env(key_indent=0, key="Ctrl+Shift+Delete", action_indent=0, action="Delete it and all below it", notes="Twice to confirm. No undo"),
               helpcard.hotkey_blank_entry,
               env(key_indent=0, key="Ctrl+N", action_indent=0, action="Start a new chat", notes=""),
               helpcard.hotkey_new_column,
               # One row for a keyboard that has a dozen keys. Tab is the row that pays: it is the way
               # *into* the graph, and from inside it the arrows, Enter and Esc are what a reader tries
               # first. The rest are in the Chat graph section of the README.
               env(key_indent=0, key="Tab", action_indent=0, action="Move the keyboard between panes", notes="Composer, chat log, chat graph"),
               env(key_indent=1, key="Shift+Tab", action_indent=1, action="Same, but backwards", notes=""),
               env(key_indent=0, key="Page Up", action_indent=0, action="Scroll the chat up one page", notes="Also while typing"),
               env(key_indent=0, key="Page Down", action_indent=0, action="Scroll the chat down one page", notes="Also while typing"),
               env(key_indent=1, key="Up", action_indent=1, action="Same, but five lines", notes="Not while typing"),
               env(key_indent=1, key="Down", action_indent=1, action="Same, but five lines", notes="Not while typing"),
               env(key_indent=0, key="Home", action_indent=0, action="Jump to the start of the chat", notes="Not while typing"),
               env(key_indent=0, key="End", action_indent=0, action="Jump to the latest message", notes="Not while typing"),
               helpcard.hotkey_blank_entry,
               env(key_indent=0, key="Ctrl+G", action_indent=0, action="Stop the AI's text generation", notes="While the AI is writing"),
               # These three notes are a set, and are worded to be read as one: sending an empty message
               # *adds* a reply, continuing *extends* the one that is there, rerolling makes an
               # *alternative* to it. Each says what happens to the chat before naming the tree operation,
               # because the table is met before the prose that explains the tree.
               env(key_indent=0, key="Ctrl+U", action_indent=0, action="Continue the marked AI message", notes="The last one only (new revision)"),
               env(key_indent=0, key="Ctrl+R", action_indent=0, action="Reroll the marked AI message", notes="An alternative to it (new sibling)"),
               helpcard.hotkey_blank_entry,
               env(key_indent=0, key="F8", action_indent=0, action="Copy the chatlog to the clipboard", notes="As-is"),
               env(key_indent=1, key="Shift+F8", action_indent=0, action="Same, but with chat node IDs", notes=""),
               env(key_indent=0, key="Ctrl+C", action_indent=0, action="Copy the marked message", notes="Not while typing"),
               env(key_indent=1, key="Ctrl+Shift+C", action_indent=0, action="Same, but with its node ID", notes=""),
               helpcard.hotkey_blank_entry,
               # The mode switches, which had been left off for want of rows. Each is also written into
               # its own checkbox's tooltip, which is where a reader looking at the row finds it; what
               # that cannot do is answer "what can I switch?" before you know what to look at.
               env(key_indent=0, key="Alt+T", action_indent=0, action="Thinking", notes="AI reasoning mode on/off"),
               env(key_indent=1, key="Alt+Shift+T", action_indent=1, action="Show thinking", notes="Whether thinking traces arrive open"),
               env(key_indent=0, key="Alt+I", action_indent=0, action="Internet", notes="websearch, webfetch"),
               env(key_indent=0, key="Alt+D", action_indent=0, action="Documents", notes="Your document database"),
               env(key_indent=0, key="Alt+G", action_indent=0, action="Chat graph", notes="Graph or avatar in the side panel"),
               env(key_indent=0, key="Alt+S", action_indent=0, action="Speech", notes="Whether the avatar speaks replies"),
               env(key_indent=1, key="Alt+C", action_indent=1, action="Subtitles", notes="Used when Speech is on"),
               helpcard.hotkey_blank_entry,
               env(key_indent=0, key="F11", action_indent=0, action="Toggle fullscreen mode", notes=""),
               env(key_indent=0, key="F1", action_indent=0, action="Open this help card", notes=""),
               )

# The graph is a keyboard of its own — nineteen keys against the main page's forty — so it gets a page
# rather than a corner of one. Two-thirds of these are unreachable from anywhere else in the app, and
# until the card had pages they lived only in the README, which is not open while you are using the graph.
chat_graph_hotkey_info = (env(key_indent=0, key="Tab", action_indent=0, action="Move the keyboard to the graph", notes="Or click anywhere in it"),
                          # The cursor has to be conjured before it can be moved, and the first arrow press
                          # is what does it — planting it on HEAD rather than stepping. Its own row, because
                          # a reader whose first press "did nothing" is looking straight at the key that
                          # worked, and the rows below would have them pressing it again to no effect.
                          env(key_indent=0, key="Any arrow", action_indent=0, action="Show the cursor, on HEAD", notes="The first press only"),
                          env(key_indent=1, key="Up", action_indent=1, action="Then: move it up the conversation", notes="Up and down follow the branch"),
                          env(key_indent=1, key="Down", action_indent=1, action="Same, the other way", notes=""),
                          env(key_indent=1, key="Left", action_indent=1, action="Move it along the siblings", notes="Left and right stay on one level"),
                          env(key_indent=1, key="Right", action_indent=1, action="Same, the other way", notes=""),
                          env(key_indent=0, key="Enter", action_indent=0, action="Look at it", notes="Again to switch to it. A gap opens what it hides"),
                          env(key_indent=0, key="Esc", action_indent=0, action="Put the cursor away", notes="Without going anywhere"),
                          env(key_indent=0, key="Backspace", action_indent=0, action="Fold an opened tool round back up", notes="From anywhere inside the round"),
                          helpcard.hotkey_blank_entry,
                          env(key_indent=0, key="Ctrl+Right", action_indent=0, action="Next sibling, drawn or not", notes="Slides the window to follow"),
                          env(key_indent=1, key="Ctrl+Shift+Right", action_indent=1, action="Same, but jump 10", notes=""),
                          env(key_indent=1, key="Ctrl+End", action_indent=1, action="Same, but to the last", notes=""),
                          env(key_indent=0, key="Ctrl+Left", action_indent=0, action="Previous sibling, drawn or not", notes=""),
                          env(key_indent=1, key="Ctrl+Shift+Left", action_indent=1, action="Same, but jump 10", notes=""),
                          env(key_indent=1, key="Ctrl+Home", action_indent=1, action="Same, but to the first", notes=""),
                          helpcard.hotkey_new_column,
                          env(key_indent=0, key="Shift+Up", action_indent=0, action="Pan the view", notes="The mouse drags; the wheel zooms"),
                          env(key_indent=1, key="Shift+Down", action_indent=1, action="Same, the other way", notes=""),
                          env(key_indent=0, key="Shift+Left", action_indent=0, action="Pan the view sideways", notes=""),
                          env(key_indent=1, key="Shift+Right", action_indent=1, action="Same, the other way", notes=""),
                          helpcard.hotkey_blank_entry,
                          env(key_indent=0, key="F", action_indent=0, action="Zoom to fit the whole tree", notes=""),
                          env(key_indent=0, key="B", action_indent=0, action="Fit the current branch", notes=""),
                          env(key_indent=0, key="1", action_indent=0, action="Actual size (1:1)", notes="Main row or numpad"),
                          env(key_indent=0, key="Numpad +", action_indent=0, action="Zoom in", notes=""),
                          env(key_indent=0, key="Numpad -", action_indent=0, action="Zoom out", notes=""),
                          helpcard.hotkey_blank_entry,
                          env(key_indent=0, key="Home", action_indent=0, action="Back to where you are", notes="Takes you to HEAD"),
                          env(key_indent=0, key="Alt+Left", action_indent=0, action="Back to the previous view", notes=""),
                          env(key_indent=0, key="Alt+Right", action_indent=0, action="Forward again", notes=""),
                          # The app's own key, listed here because from this view it is another way of moving:
                          # it takes you to the box wearing the NEW pill. The graph does not bind it — an
                          # unhandled key falls through to the app — so this is a signpost, not a second binding.
                          env(key_indent=0, key="Ctrl+N", action_indent=0, action="Start a new chat", notes="Takes you to NEW"),
                          )
def render_chat_graph_help(self: helpcard.HelpWindow,
                           gui_parent: str | int) -> None:
    """Render the chat graph's explanation, below its hotkey table on the card's chat graph page.

    Called by `HelpWindow` when the help card is first rendered.
    """
    # Two columns, as page three is: a card this wide gives a single column lines too long to track back
    # to the start of. One section per column, so the reader crosses over once, at the bottom.
    self.prose_columns(
        gui_parent,
        [helpcard.section(
            None,
            "The graph draws the **whole chat tree**: the branch you are on runs down the middle, with a few of its siblings either side at each level. One of those levels is every chat ever started under the current character card, which is as close as this format comes to a list of recent chats.",
            "**Clicking is two steps, and the first changes nothing.** Click a message to look at it — the graph redraws around it if it is on another branch — and click it again to move the conversation there. Colour says where you *are* rather than what you are looking at, so the point where a branch you are considering left the one you are on stays visible.")],
        [helpcard.section(
            None,
            f'Anything left out is drawn as a dashed {self.c_hig}**...N more**{self.c_end} box, so a box with no visible links means the tree really does end there. Clicking one navigates: between siblings it jumps to the middle of what it hides, under an off-branch message it opens what continues below, and a round of three or more tool results opens into its own boxes.',
            f'The one at the very top walks through the *other* character cards — the only route to chats you held under an earlier system prompt. It wears the {self.c_hig}**HEAD**{self.c_end} pill while the chat you are actually in is behind it.')])
    dpg.add_spacer(height=themes_and_fonts.font_size, parent=gui_parent)

def render_help_extras(self: helpcard.HelpWindow,
                       gui_parent: str | int) -> None:
    """Render app-specific extra information into the help card.

    Called by `HelpWindow` when the help card is first rendered.
    """
    # Two newspaper columns, as page two is: a card this wide gives a single column lines too long to track
    # back to the start of, and a section split into a pair either side would be read out of order — the
    # eye runs a column to its end before it crosses over. So the left column is finished before the right
    # one starts, and the split is by eye, there being no way to measure a column before it is drawn.
    #
    # No locator for the Chat graph switch, deliberately. "Below the avatar" is where it is today and
    # nowhere in a no-avatar mode, which has no avatar to be below and no toggle either, the graph being
    # permanently up. A label is findable; a direction that is wrong in one mode is worse than none. That
    # mode will want that sentence to say something else entirely, which is its own work.
    self.prose_columns(
        gui_parent,
        [helpcard.section(
            "**Chat history**",
            "The chat history is **natively multiversal**. Messages are stored as nodes in a tree. The current chat is the HEAD, plus its ancestor chain up to the system prompt. Continuing the chat adds a new child node below the latest message displayed.",
            "Rerolling creates a new sibling and sets the HEAD pointer to that. Previous siblings remain stored in the tree. Starting a new chat, or branching the chat, only resets the HEAD pointer.",
            "Nothing is discarded on its own. Where a message has siblings, its arrow buttons step between them, so a rerolled reply can be compared against the one it replaced. To reach a *different* old chat, switch on **Chat graph** — which has a page of its own on this card.",
            "Deleting is the exception, and the only one: a message's trash button removes that message and everything below it, for good. It asks for a second click first, because there is no undo."),
         helpcard.section(
            f"**Document database** {self.c_txt}(retrieval-augmented generation, RAG){self.c_end}",
            f'You can put documents for the AI to access in {self.c_hig}{librarian_config.llm_docs_dir}{self.c_end}. The folder button on the {self.c_hig}**Open folder:**{self.c_end} row opens it in your file manager.',
            f'Plain text, Markdown, BibTeX, LaTeX, PDF, Word, PowerPoint, OpenDocument and saved web pages are read - the text layer only, so a scanned PDF needs OCR (e.g. **ocrmypdf**) before it can be indexed. Recognition is by file extension, listed as {self.c_hig}llm_docs_exts{self.c_end} in {self.c_hig}raven/librarian/config.py{self.c_end} — add to it if you keep notes in a plain-text format that is not there. That file sets the folder above, too.',
            f'The documents are search-indexed automatically, and the index is kept up to date. It is stored in {self.c_hig}{librarian_config.llm_database_dir}{self.c_end}. If you ever need to clear it manually, just delete that directory.',
            f'Indexing runs in the app as it goes, but {self.c_hig}raven-indexer{self.c_end} does the same from a terminal and then exits - for a folder you have just filled with hundreds of documents, or a machine you reach over SSH with no display to start a GUI on.',
            f'When the {self.c_hig}**Documents**{self.c_end} checkbox in the app is **ON**, the document database is automatically searched, using your latest message to the AI as the search query. The AI can also search it again itself, with a better query, once it has read those results.',
            f'A reply that had nothing to stand on - no document matches, no attachments, no tool results - is marked {self.c_hig}**[no sources retrieved]**{self.c_end}. The AI still answers; the marker reports what was **retrieved**, not whether it was used. With {self.c_hig}**Documents**{self.c_end} off it does not appear at all, since it would only be reporting what you just switched off.',
            'To improve search result quality, Raven-librarian uses a hybrid method: Okapi BM25 for keywords, and vector embeddings for semantic search. Results are combined with RRF (reciprocal rank fusion).')],
        [helpcard.section(
            "**Message attachments**",
            'The document database answers *what do my documents say about this*; an attachment answers *read this one, now* - the whole thing, rather than the snippets that matched.',
            textwrap.dedent(f"""
                Attach one or several with the paperclip button, with {self.c_hig}**Ctrl+Shift+O**{self.c_end}, or by dropping files on the window. Two kinds, asking different things of the model:

                - **Documents** — anything the database accepts. Any model can read one.
                - **Images** — seen only by a vision-capable model (a VLM).
            """).strip(),
            'Both ride along with the message you attached them to, so they stay in the branch where you asked about them.',
            '**The AI produces attachments too.** A web page it fetches is filed as one rather than pasted into the conversation: the chat log shows the opening and a chip to click, while the model reads the whole thing.',
            'Attachments are sized against the context window along with everything else, so several large ones share the room that is left rather than crowding each other out. One too big for its share is cut in the **middle**, keeping the beginning and the end with a marker saying how much went - the chip still opens the whole stored copy, whatever the AI saw.',
            f'They are stored beside the chat, **content-addressed** - identical bytes are kept once however many messages point at them. Anything no longer referenced by any message can be cleaned up with the broom button on the {self.c_hig}**Maintenance**{self.c_end} row, which shows what it would delete before deleting anything, and offers to rescue a copy first.'),
         helpcard.section(
            "**Tool use** (tool-calling)",
            textwrap.dedent("""
                The AI has these tools, and decides for itself which to use, if any:

                - **websearch** — search the web
                - **webfetch** — read a web page, one it found or one you linked
                - **search_documents** — search your document database
                - **fetch_document** — read one of those in full
                - **list_consulted_documents** — list what this chat has consulted
                - **get_current_time** — ask what time it is
                - **calculate** — do arithmetic
            """).strip(),
            f'The first two need {self.c_hig}**Internet**{self.c_end}, the next three need {self.c_hig}**Documents**{self.c_end}, and the last two are always available, reaching nothing outside this process. Each checkbox governs its own group, so switching one off never takes the other away.',
            'One reply may take several rounds of tool calls, up to a configurable ceiling.',
            'Either switch changes which tools are declared to the AI, and the declarations travel at the top of the conversation - so the reply after you flip one has to re-read the whole chat before it can start. Nothing is lost; on a long chat it is a pause.')])
# Three pages, split by scope rather than to find room. Page one is the app's keyboard and nothing else,
# so it is a reference a reader can screenshot and keep open on a second monitor while they learn it —
# which is what the card's header has been inviting all along, and what prose sharing the page took away.
# The graph is a keyboard of its own and gets page two, with the prose that explains it directly under its
# keys. Page three is what the app *does* — read once, where the keyboard pages are kept open — and is
# called Features rather than anything with "About" in it, which promises a version number and a credits
# list. If it outgrows one screen, the answer is a fourth page rather than shorter prose: the card has no
# scrollbar by design, and paging is what was built to replace one.
help_window = helpcard.HelpWindow(width=gui_config.help_window_w,
                                  height=gui_config.help_window_h,
                                  reference_window=main_window,
                                  themes_and_fonts=themes_and_fonts,
                                  pages=[helpcard.page("Keyboard", hotkey_info=hotkey_info),
                                         helpcard.page("Chat graph",
                                                       hotkey_info=chat_graph_hotkey_info,
                                                       on_render_extras=render_chat_graph_help),
                                         helpcard.page("Features", on_render_extras=render_help_extras)],
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

def _center_ai_warning(avatar_panel_w: int) -> None:
    """Horizontally center the AI-disclosure label within the avatar-side panel of the bottom row.

    `avatar_panel_w`: current width of the avatar panel, in pixels.

    Centering is done by widening a leading spacer rather than by positioning the text, because the label
    is a wrapped two-line block inside a horizontal group, and DPG has no "center this group" affordance.

    The block's width is taken from the laid-out widget (`get_item_rect_size` on the group holding the icon
    and the text) rather than reconstructed from its parts. Reconstructing it was off by a constant 18 px:
    it has to account for the child window's padding, for the gap DPG puts between group members, and for
    the fact that a wrapped text's allocated box is not its measured ink width (550 px of wrap allowance
    renders 521 px of glyphs). Measuring the assembled group gets all three for free, and cannot drift if
    the wording, the wrap width, or the theme's spacing changes.

    Before the first frame the group has no size yet; the estimate from the parts is used until then, and
    the next resize pass corrects it.
    """
    block_w, _ = guiutils.get_widget_size("ai_warning_block")  # tag
    if not block_w:  # not laid out yet (called before the first frame)
        text_w, _ = dpg.get_text_size(_AI_WARNING_TEXT, wrap_width=gui_config.ai_warning_w)
        block_w = _AI_WARNING_ICON_W + text_w
    # The spacer lives in the child window's *content* box, which is inset by WindowPadding on each side,
    # so the width to center within is the panel minus that padding - not the panel's outer width.
    content_w = avatar_panel_w - 2 * _WINDOW_PADDING
    dpg.set_item_width("ai_warning_centering_spacer", max(0, int((content_w - block_w) / 2) - _AI_WARNING_CENTERING_BIAS))  # tag


def _resize_panels() -> None:
    """Resize the panels in the main window RIGHT NOW, based on main window size."""
    global _animator_settings  # noqa: F824 -- intent only; loaded during app startup, never rebound here

    w, h = guiutils.get_widget_size(main_window)

    chat_panel_w, chat_panel_h = _get_chat_panel_size(main_window_w=w, main_window_h=h)
    dpg.set_item_width("chat_panel", chat_panel_w)  # tag
    dpg.set_item_height("chat_panel", chat_panel_h)  # tag
    chat_controls_w, chat_controls_h = _get_chat_controls_size(main_window_w=w, main_window_h=h)
    dpg.set_item_width("chat_controls", chat_controls_w)
    dpg.set_item_height("chat_controls", chat_controls_h)
    chat_field_w = _get_chat_field_width(main_window_w=w)
    dpg.set_item_width("chat_field", chat_field_w)  # tag

    dpg.set_item_width("chat_global_buttons", chat_panel_w)  # tag

    avatar_panel_w, avatar_panel_h = _get_avatar_panel_size(main_window_w=w, main_window_h=h)
    dpg.set_item_width("ai_warning_panel", avatar_panel_w)  # tag
    _center_ai_warning(avatar_panel_w)
    avatar_controller.subtitle_bottom_y0 = _get_subtitle_bottom_y0(avatar_panel_h)  # takes effect from next subtitle shown
    avatar_controller.reposition_subtitle()  # apply new position to current subtitle, if any
    dpg.set_item_width("avatar_panel", avatar_panel_w)  # tag
    dpg.set_item_height("avatar_panel", avatar_panel_h)  # tag
    chat_graph_panel.set_size(avatar_panel_w, avatar_panel_h)  # shares that rect exactly
    dpg_avatar_renderer.reposition(new_x_center=(avatar_panel_w // 2),
                                   new_y_bottom=(avatar_panel_h - 8))
    if _animator_settings is not None:  # may not be initialized yet at app startup on a 1920x1080 screen (triggers immediate resize)
        blur_state = _animator_settings["backdrop_blur"]
        logger.info(f"_resize_panels: `_animator_settings` exists, got `backdrop_blur={blur_state}`.")
    else:
        logger.warning("_resize_panels: `_animator_settings` not initialized, assuming `backdrop_blur=True`. Maybe GUI was resized before the app has finished booting up?")
        blur_state = True
    dpg_avatar_renderer.configure_backdrop(new_width=avatar_panel_w - 16,
                                           new_height=avatar_panel_h - 16,
                                           new_blur_state=blur_state)

    # TODO: change upscale factor too? (need to update "upscale" in `librarian_config.avatar_config.animator_settings_overrides` and send config to server)

def _resize_gui_task(task_env: env) -> None:
    """We run this in the background. Expensive parts of the GUI update benefit from the "there can be only one" mechanism."""
    if task_env.cancelled or _shutting_down:  # while waiting in queue, or app tearing down (this task can be submitted *after* the shutdown cancel)
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
    if _shutting_down:  # a resize event (incl. the window close itself) must not kick off GUI rebuilds during teardown
        return
    logger.debug("_resize_gui: Entered.")
    logger.debug("_resize_gui: Recentering help window.")
    help_window.reposition()
    logger.debug("_resize_gui: Recentering avatar.")
    dpg_avatar_renderer.reposition()  # reposition paused text
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

def is_attach_file_dialog_visible() -> bool:
    """Return whether the attach-file dialog is open.

    An abstraction over `dpg.is_item_visible`, not just a call to it, because the window might not exist yet.
    """
    if _filedialog_attach is None:
        return False
    return _filedialog_attach.is_visible()

def is_any_modal_window_visible() -> bool:
    """Return whether some window that owns the keyboard is open: the help card, the attach-file dialog,
    the cleanup dialog, or the messagebox.

    **It enumerates rather than asks, and the name overstates what it checks.** There is no modality test
    here — each of the four is asked whether it is *visible*, and the four are named. That is on purpose,
    DPG offering no way to enumerate the modals on its popup stack, but it has two consequences worth
    knowing before adding a window:

      - **A new window is not covered until it is added here**, whether or not it is modal. The symptom is
        chat hotkeys staying live behind it: `Enter` sending a message while the reader believes they are
        confirming a deletion.
      - **A window in this list stays covered if it is made non-modal.** `FileDialog` takes `modal` as a
        parameter and every Raven call site currently leaves it `True`; flipping one would not break this
        gate, because the gate never depended on the flag.

    A window that takes the keys only while the focus is on one of its own controls does *not* belong here
    — it answers for itself, and this gate would silence the whole app while it was merely on screen. The
    audio input panel is that shape, and so is the chat graph; both appear as their own branches in the key
    dispatcher, tested by "does it hold the keyboard?" rather than by "is it open?".
    """
    return (help_window.is_visible() or
            is_attach_file_dialog_visible() or
            cleanup_dialog.is_open or
            messagebox.is_visible())

combobox_choice_map = None   # DPG tag or ID -> (choice_strings, callback)
def _give_keyboard_to_graph() -> None:
    """Send the keyboard to the chat graph. What a click anywhere in it asks for, and where Tab lands.

    Two halves, and doing one without the other is the bug this exists to avoid: the graph is told it has
    the keys, *and* the composer is made to give up the caret. Leaving the composer active would light two
    panes blue at once and leave the typist's keys going to a pane they are not looking at.

    Parking on a button is what deactivates an ImGui text field from outside, and a focused button is safe
    -- DPG leaves ImGui's keyboard-navigation activation off, so it ignores Space and Enter rather than
    pressing itself.

    Release everything, then claim: the shape every claimant here uses, so that adding a pane with a flag
    of its own means editing `_release_manual_keyboard_claims` and nothing else.
    """
    dpg.focus_item("chat_send_button")  # tag
    _release_manual_keyboard_claims()
    chat_graph_panel.has_keyboard = True


def _release_manual_keyboard_claims() -> None:
    """Clear every "this pane has the keyboard" flag Raven maintains by hand. Grep for callers.

    **The list of such flags is here and nowhere else, which is the whole point of the function.** Most of
    the keyboard home is derived — `_cycle_keyboard_home` reads ImGui's caret and works the rest out — and
    what cannot be derived is kept in one flag per pane, because DPG cannot answer for a pane that has no
    widget to hold a caret. Those flags go stale the moment something *else* takes the keyboard, and
    nothing tells them so.

    So anything that takes the keyboard calls this first and then claims what it wants. A new pane with a
    flag of its own adds one line here, and every existing claimant releases it without being edited; a
    new *claimant* calls this and inherits the whole list. Written out at each opener instead, the two
    would drift the first time one of them was added without the other in view — which is exactly how the
    audio input panel came to leave the chat graph lit while it had the keys.

    Not a general "give the keyboard up": it clears the manual flags only, and leaves ImGui's caret alone.
    A caller that also needs the composer deactivated parks focus on a button, as `_give_keyboard_to_graph`
    does — that is DPG's half, and DPG owns it.

    TODO: See `TODO_DEFERRED.md`, "Nothing owns 'which pane has the keyboard'". This makes the obligation
    TODO: greppable rather than removing it; a single owner would be better and is not obviously reachable
    TODO: given what DPG reports. Read that item before replacing this with stored state.
    """
    chat_graph_panel.has_keyboard = False


def _toggle_audio_input_panel() -> None:
    """Open or close the audio input panel, releasing the graph's claim on the keyboard as it opens.

    The panel is a window of its own and parks the keyboard inside itself, so from the moment it opens the
    keys are no longer the graph's. Nothing else says so: `_cycle_keyboard_home` derives the home from
    ImGui's caret plus the graph's one stored bit, and that bit stays set until something clears it — so
    without this the graph goes on claiming the keyboard, and stays lit, while the panel has it.

    It used to be corrected by accident, on the first click landing outside the graph, which could be the
    click that closed the panel again. That is the right answer arriving at the wrong moment, and looks
    like the close doing something to the focus.

    Here rather than on the panel, which owns none of this: the keyboard model lives in this module, and a
    panel that had to know about the chat graph would be the wrong shape for the next panel too.
    """
    if not audio_input_panel.is_open:
        _release_manual_keyboard_claims()
    audio_input_panel.toggle()


def _cycle_keyboard_home(backwards: bool = False) -> None:
    """Move the keyboard to the next pane, or the previous one. What Tab and Shift+Tab do.

    Three panes at one level — the composer, the chat log, and the graph while it is on screen — so Tab
    cycles rather than descending into anything, and Shift+Tab is worth having because with three of them
    "the one before" is otherwise two presses away.

    **The current home is derived, not stored**, and that is deliberate: the composer can be entered by
    clicking it or with Ctrl+Space, neither of which goes through here, so a stored answer would be wrong
    whenever it was not the last thing to have set it. What is stored is only the bit that DPG cannot
    answer — whether the graph has the keys — and that is meaningful only while the composer does not.

    TODO: **The derivation knows about three panes, and a window that takes the keyboard is not one of**
    TODO: **them.** With the audio input panel up, this still answers "log" or "graph" — true of where the
    TODO: keys would go were the panel dismissed, and false about where they are. Harmless today, since
    TODO: nothing asks while a panel is up; a fourth pane, or anything that asks at the wrong moment,
    TODO: would find it. The flags this reads are listed in `_release_manual_keyboard_claims`; see
    TODO: `TODO_DEFERRED.md`, "Nothing owns 'which pane has the keyboard'".
    """
    graph_available = chat_graph_panel.is_shown
    homes = ["composer", "log"] + (["graph"] if graph_available else [])

    if dpg.is_item_active("chat_field"):  # tag
        current = "composer"
    elif graph_available and chat_graph_panel.has_keyboard:
        current = "graph"
    else:
        current = "log"

    # Which also settles where the keyboard lands when the graph *disappears* while holding it — the
    # toggle switched off, or, once the avatar hand-off is built, the video coming back. The graph stops
    # being available, so this falls through to the log, and that is the right place rather than an
    # accident of the ordering: the log is the neutral home. Nothing there grabs the caret, bare keys
    # merely scroll, and the message that would be acted on already wears the blue dot. Landing on the
    # *composer* instead would take the caret because a panel the reader dismissed went away, which is a
    # side effect nobody asked for.

    target = homes[(homes.index(current) + (-1 if backwards else 1)) % len(homes)]

    # The composer is left by parking focus on a real widget, which is the only thing that deactivates an
    # ImGui text field from the outside. A button is the safe place to park: DPG leaves ImGui's keyboard
    # navigation activation off, so a focused button ignores Space and Enter rather than pressing itself.
    if target == "composer":
        dpg.focus_item("chat_field")  # tag
        chat_graph_panel.has_keyboard = False
    elif target == "graph":
        _give_keyboard_to_graph()
    else:
        dpg.focus_item("chat_send_button")  # tag  # deactivate the composer's ImGui edit buffer
        chat_graph_panel.has_keyboard = False


def librarian_hotkeys_callback(sender, app_data):
    global _last_input_ns
    _last_input_ns = time.monotonic_ns()

    # The cleanup dialog is the one modal with no key handling of its own, so Esc is honored here. Everything
    # else it swallows, via the general guard below.
    if cleanup_dialog.is_open and app_data == dpg.mvKey_Escape:
        cleanup_dialog.close()
        return

    key = app_data
    ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
    shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
    alt_pressed = dpg.is_key_down(dpg.mvKey_LAlt) or dpg.is_key_down(dpg.mvKey_RAlt)

    # ------------------------------------------------------------
    # Helpers for operating on the most recent chat message

    def fire_event_if_exists(action: str) -> None:
        # The message on screen, not the last one in the chat. They are the same until the reader scrolls
        # back, and from then on a reroll or a sibling step aimed at the last message is an edit happening
        # somewhere they cannot see. The blue dot beside the button row says which message this is.
        dpg_chat_message = chat_controller.get_current_message()
        if dpg_chat_message is None:
            return
        if action in dpg_chat_message.gui_button_callbacks:
            dpg_chat_message.gui_button_callbacks[action]()

    # ------------------------------------------------------------

    # No shared keymap — bindings live here, and the surfaces that make them
    # discoverable mirror them by hand (KISS; hotkeys change rarely). If you add,
    # remove, or rebind a key, update those surfaces too:
    #   - the help card (search "HelpWindow")
    #   - any tooltip naming the key (search its bracketed hint, e.g. "[Ctrl+O]")

    # A global action's closure must take NO parameters. Below, such a closure is called directly by name,
    # while the toolbar binds the same object as a DPG `callback=` - and DPG passes `sender` positionally,
    # so a stray parameter is absorbed there and the button keeps working while the hotkey raises TypeError.
    # That asymmetry hid a dead F8 for months. The per-message actions are immune by construction: they go
    # through `fire_event_if_exists`, which invokes the very callable the button is bound to, so the two
    # surfaces cannot drift apart.

    # Hotkeys that are always available, regardless of any dialogs (even if modal)
    if key == dpg.mvKey_F11:  # de facto standard hotkey for toggle fullscreen
        toggle_fullscreen()

    # Hotkeys while a modal window is shown - each modal handles its own keys (the help card, the file
    # dialog and the messagebox have their own handlers; the cleanup dialog's Esc is handled above).
    #
    # A DPG modal blocks the mouse but NOT the keyboard, so without this the chat hotkeys stay live behind
    # whatever is on top: Enter would send a chat message while the user believes they are confirming a
    # deletion or picking a file. Nothing here is cosmetic.
    elif is_any_modal_window_visible():
        return

    elif key == dpg.mvKey_F1:  # de facto standard hotkey for help
        help_window.show()

    # The audio input panel is not modal, so it cannot claim the keyboard the way a dialog does — it
    # takes the keys only while the focus is on one of its own controls, and passes on anything else.
    # That is what lets it use bare letters without stealing them from the composer.
    elif audio_input_panel.has_keyboard() and audio_input_panel.handle_key(key, ctrl=ctrl_pressed, alt=alt_pressed):
        pass

    # The chat graph, on the same terms as the audio panel above: it takes the keys only while it is the
    # keyboard home, and passes on anything it does not claim, so F1 and the rest still work from inside
    # it. Above the composer's own branches, because a reader who has Tabbed to the graph means Enter to
    # commit a branch rather than to send an empty message.
    #
    # **Not a keyboard handler of its own, which is how the constellation usually gives a view its keys.**
    # That pattern belongs to *modals*: a modal owns the keyboard outright, so the main handler can punt
    # the moment one is open and neither side needs to know what the other binds. Neither of these two is
    # modal — the reader can be typing in the composer with the graph on screen — so "is it open?" is the
    # wrong question and "does it hold the keyboard?" is the right one. Which makes it a branch here, in
    # the chain that already answers that question for every other pane.
    # The composer's own guard is far below, and would be reached too late: the graph claims `Enter`, the
    # bare arrows and several letters, so a stale bit -- see `update_animations` -- meant a typist's Enter
    # switching branch instead of sending. Tested here rather than relied upon to have been cleared,
    # because a click and a keypress can land in one frame and the poll runs between frames.
    elif (not dpg.is_item_active("chat_field")  # tag
          and chat_graph_panel.has_keyboard
          and chat_graph_panel.handle_key(key, ctrl=ctrl_pressed, shift=shift_pressed, alt=alt_pressed)):
        pass

    # Hotkeys for main window, while no modal window is shown
    elif key == dpg.mvKey_F9:  # a bare key, because it is reached with a microphone in one hand
        _toggle_audio_input_panel()
    elif key == dpg.mvKey_F8:  # NOTE: Shift is a modifier here
        copy_chatlog_to_clipboard_as_markdown_callback()
    # Copying one message, the per-message counterpart of F8 above — and tested here, above the Ctrl+Shift
    # branch, for the same reason F8 is: the callback reads Shift itself, so one binding serves both
    # Ctrl+C and Ctrl+Shift+C, and Shift is what adds the node ID.
    #
    # Not while the composer holds the caret, where Ctrl+C is ImGui's own copy of the selected text. This
    # is the only per-message key needing that guard; the rest are chords a text field does not claim.
    elif ctrl_pressed and key == dpg.mvKey_C and not dpg.is_item_active("chat_field"):  # tag
        fire_event_if_exists("copy")
    # Ctrl+Shift+...
    elif ctrl_pressed and shift_pressed:
        if key == dpg.mvKey_Return:
            record_audio_message_callback()
        elif key == dpg.mvKey_Left:
            fire_event_if_exists("prev10")
        elif key == dpg.mvKey_Right:
            fire_event_if_exists("next10")
        elif key == dpg.mvKey_O:
            # Shift, so that plain Ctrl+O stays free for opening a chat datastore — a deferred item, and
            # the meaning a reader will expect of the unshifted chord.
            show_attach_dialog()
        # Two modifiers and a key well away from the letters, so it cannot be hit by accident: this is the
        # one hotkey in the app that destroys data. It still asks for the second press the button does.
        elif key == dpg.mvKey_Delete:
            fire_event_if_exists("delete")

        # Some hidden debug features. Mnemonic: "Mr. T Lite" (Ctrl + Shift + M, R, T, L)
        elif key == dpg.mvKey_M:
            # One state, "show me the numbers", not a toggle per overlay: each of these reports on the
            # view it is drawn on, and the graph is up exactly when the avatar is not — so both follow
            # the key, and whichever view is on screen answers it. DPG's own Metrics window is the third,
            # and is what carries the numbers that belong to the app rather than to either view.
            dpg.show_metrics()
            dpg_avatar_renderer.configure_fps_counter(show=None)  # `None` = toggle
            chat_graph_panel.configure_metrics_readout(show=None)  # ...ditto
        elif key == dpg.mvKey_R:
            dpg.show_item_registry()
        elif key == dpg.mvKey_T:
            dpg.show_font_manager()
        elif key == dpg.mvKey_L:
            dpg.show_style_editor()
    # Alt+... — the switches in the row below the avatar, one key each, mnemonic on the label. Alt is
    # otherwise unused here, which is what leaves seven letters free in one place; the graph claims
    # Alt+Left/Alt+Right for its history, and is offered the key further up, while it holds the keyboard.
    #
    # Live while the composer has the caret, like the Ctrl chords above and unlike the bare keys below:
    # Alt+letter types nothing into a text field, and these are exactly the switches a reader reaches for
    # mid-sentence, on noticing that the answer wants the web or the documents.
    elif alt_pressed:
        # Each flips its checkbox and runs whatever that checkbox is wired to, so a key and a click are the
        # same gesture and cannot drift apart if one of them is later rewired.
        #
        # `Alt+Shift+T` rather than a letter of its own, because the pair is the point: what the AI does
        # when it answers, and what you then see of it. The row is grouped that way and the README says so.
        if key == dpg.mvKey_T and shift_pressed:
            guiutils.toggle_checkbox("show_thinking_checkbox")  # tag
        elif key == dpg.mvKey_T:
            guiutils.toggle_checkbox("thinking_enabled_checkbox")  # tag
        elif key == dpg.mvKey_I:
            guiutils.toggle_checkbox("internet_enabled_checkbox")  # tag
        elif key == dpg.mvKey_D:
            guiutils.toggle_checkbox("docs_enabled_checkbox")  # tag
        elif key == dpg.mvKey_G:
            guiutils.toggle_checkbox("chat_graph_checkbox")  # tag
        elif key == dpg.mvKey_S:
            guiutils.toggle_checkbox("speech_enabled_checkbox")  # tag
        elif key == dpg.mvKey_C:  # C for captions; S is taken by Speech, which this one qualifies
            guiutils.toggle_checkbox("avatar_subtitles_checkbox")  # tag

    # Tab moves the keyboard between the panes, Shift+Tab the other way. Above the Ctrl branch and above
    # the composer's own branch, because it has to be reachable *from* the composer: Tab types nothing
    # into a multiline field, so it was doing nothing at all there.
    elif key == dpg.mvKey_Tab:
        _cycle_keyboard_home(backwards=shift_pressed)

    # Ctrl+...
    elif ctrl_pressed:
        if key == dpg.mvKey_Spacebar:
            dpg.focus_item("chat_field")  # tag
        # The send chord, when it is the composer that is *not* holding the caret. While it is, the field
        # commits and this fires on the same keypress; `_request_send` is what makes that one send.
        elif key == dpg.mvKey_Return and librarian_config.send_message_key == "ctrl+enter":
            _request_send()
        elif key == dpg.mvKey_R:
            fire_event_if_exists("reroll")
        elif key == dpg.mvKey_T:
            fire_event_if_exists("toggle_thinking_trace")
        elif key == dpg.mvKey_U:
            fire_event_if_exists("continue")
        elif key == dpg.mvKey_Left:
            fire_event_if_exists("prev1")
        elif key == dpg.mvKey_Down:
            fire_event_if_exists("show_chat_continuation")
        elif key == dpg.mvKey_Right:
            fire_event_if_exists("next1")
        # The ends of the run. Bare Home and End scroll the log, so these are free -- and the graph, which
        # claims the same two chords for the same verb, is offered the key first and only when it holds
        # the keyboard.
        elif key == dpg.mvKey_Home:
            fire_event_if_exists("prevend")
        elif key == dpg.mvKey_End:
            fire_event_if_exists("nextend")
        # Beside Ctrl+N rather than with the sibling steps above: branching and starting a new chat are the
        # two keys that answer "where does the conversation go from here", and both only move HEAD.
        #
        # Guarded like Ctrl+C above, for a different reason: Ctrl+B is the bold reflex every rich-text
        # editor has trained, and a plain-text field that ignores it should go on ignoring it rather than
        # rolling the conversation back to some other message.
        elif key == dpg.mvKey_B and not dpg.is_item_active("chat_field"):  # tag
            fire_event_if_exists("branch")
        elif key == dpg.mvKey_N:
            start_new_chat_callback()
        elif key == dpg.mvKey_S:
            if audio_player.require().is_playing():
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
        # Chat log scrolling. Page Up/Down page the log wherever focus is: the composer is a few lines tall,
        # so paging *within* it means nothing, and reaching for these keys while typing is how a reader looks
        # back at what they are replying to.
        #
        # The literal 517/518 are not magic numbers to be tidied into constants — they ARE the constants.
        # `dpg.mvKey_Prior` and `dpg.mvKey_Next` still carry their pre-2.0 Windows virtual-key values (266,
        # 267), which no longer match anything DPG delivers, so comparing against them silently never fires:
        # no error, just a dead key. See `dpg-notes.md`, "Keyboard input", and `briefs/reference/dpg-keycodes.md`.
        # The named constants are kept alongside the literals rather than dropped: they cost nothing, they say
        # which key this is meant to be, and if DPG ever regenerates them the binding starts matching on the
        # name too, with no edit here.
        if key in (517, dpg.mvKey_Prior):  # Page Up
            chat_controller.view.page_up()
        elif key in (518, dpg.mvKey_Next):  # Page Down
            chat_controller.view.page_down()

        # The send chord, under the setting that puts it on bare Enter. Sending is *also* the composer's own
        # commit action, wired at the widget (`on_enter` + `ctrl_enter_for_new_line`), because ImGui
        # consumes the chord itself while the field holds the caret. This branch is for every other moment —
        # notably just after a send, which parks focus on the send button, where an empty message is the
        # "let the AI take another turn" gesture and nothing was pressable.
        #
        # It sits above the `is_item_active` branch below so that it is reached while typing too. That costs
        # nothing: by the time this handler runs the commit has already deactivated the field, so the branch
        # below would not have caught the key anyway, and `_request_send` collapses the two into one send.
        #
        # Under the *other* setting Enter inserts a newline and must not send, which is what the config test
        # is for — not a redundant guard on a key that only means one thing.
        elif key == dpg.mvKey_Return and librarian_config.send_message_key == "enter":
            _request_send()

        # *Active*, not *focused*, and the distinction is the whole reason the navigation keys work at all.
        # ImGui hands nav focus to the first navigable item of a newly focused window on its own, so the
        # composer reports focused from the first frame with no user having gone near it — a gate on
        # `is_item_focused` therefore swallows Up/Down/Home/End until something else is clicked. *Active* is
        # the state that actually means "this field owns the caret": measured False when merely auto-focused
        # and after Escape, True from the click that enters the field until it is left. That is exactly the
        # condition under which these keys belong to the widget rather than to the log.
        elif dpg.is_item_active("chat_field"):  # tag
            # Empty on purpose, and load-bearing: this branch exists to *withhold* the log-navigation keys
            # below while someone is typing. Every key it would claim belongs to the widget instead.
            #
            # Escape is deliberately absent: ImGui's own `InputText` handles it. Under `escape_clears_all`
            # — set where the field is built — a press on a field with text in it clears the text, and a
            # press on an empty one deactivates the field. So Esc from a written message clears it, and Esc
            # again leaves; nothing needs parking afterwards, and an inactive field is what this branch
            # tests for, so the navigation keys are live from there.
            #
            # Up/Down/Home/End are likewise absent, and belong to the widget: in a multiline field
            # they move the caret between lines and to the ends of one, which is what a typist expects and
            # what the field already does. Claiming them would break ordinary text editing to add scrolling
            # that Page Up/Down already provides from inside the composer.
            pass

        else:
            # With the composer out of the way, the remaining navigation keys scroll the log. Bare Up/Down
            # were previously unbound; the modified arrows are sibling navigation (`Ctrl` +/- `Shift`) and
            # keep their meaning. See `_SCROLL_FONT_HEIGHTS_PER_ARROW` for why an arrow moves several.
            if key == dpg.mvKey_Up:
                chat_controller.view.scroll_by_font_heights(-_SCROLL_FONT_HEIGHTS_PER_ARROW)
            elif key == dpg.mvKey_Down:
                chat_controller.view.scroll_by_font_heights(_SCROLL_FONT_HEIGHTS_PER_ARROW)
            elif key == dpg.mvKey_Home:
                chat_controller.view.go_to_top()
            elif key == dpg.mvKey_End:
                chat_controller.view.go_to_bottom()
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
    # Input tracking for idle throttle. Mouse-move covers slider drags, scrolling, and general activity.
    dpg.add_mouse_move_handler(callback=_on_any_input)
    dpg.add_mouse_click_handler(callback=_on_any_input)
    dpg.add_mouse_wheel_handler(callback=_on_mouse_wheel)

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
                                        subtitle_bottom_y0=_get_subtitle_bottom_y0(avatar_panel_h),
                                        translator_source_lang=gui_config.translator_source_lang,
                                        translator_target_lang=gui_config.translator_target_lang,
                                        executor=bg)  # use the same thread pool as our main task manager
avatar_record = avatar_controller.register_avatar_instance(avatar_instance_id=avatar_instance_id,
                                                           avatar_renderer=dpg_avatar_renderer,
                                                           voice=librarian_config.avatar_config.voice,
                                                           voice_speed=librarian_config.avatar_config.voice_speed,
                                                           emotion_blacklist=librarian_config.avatar_config.emotion_blacklist,
                                                           emotion_autoreset_interval=librarian_config.avatar_config.emotion_autoreset_interval,
                                                           idle_timeout=librarian_config.avatar_config.idle_off_timeout,
                                                           # The panel changes hands here rather than at the
                                                           # watch's next tick, so the renderer's "video is
                                                           # off" text is drawn into a panel the graph has
                                                           # already taken. A lambda because the handler
                                                           # takes the instance and this one wants none of it.
                                                           on_idle=lambda config: _apply_panel_occupancy())
avatar_controller.tts.warmup(voice=librarian_config.avatar_config.voice)

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
                                    think_glow_animation=pulsating_think_glow,
                                    docs_indexing_glow_animation=pulsating_red_docs_glow,
                                    attachment_read_indicator_widget=attachment_read_indicator_group,
                                    llm_indicator_widget=llm_indicator_group,
                                    docs_indexing_indicator_widget=docs_indexing_indicator_group,
                                    docs_indexing_progress_text_widget="docs_indexing_progress_text",
                                    docs_search_indicator_widget=docs_search_indicator_group,
                                    docs_search_progress_text_widget="docs_search_progress_text",
                                    web_indicator_widget=web_indicator_group,
                                    is_any_modal_window_visible=is_any_modal_window_visible,
                                    # Asked of the renderer, which owns the answer, rather than of
                                    # whichever pane happens to have taken the space. The two agree
                                    # today; only one of them keeps agreeing when a third occupant of
                                    # the avatar column arrives.
                                    avatar_panel_covered=(lambda: not dpg_avatar_renderer.is_shown),
                                    executor=bg)

def _get_cleanup_roots() -> tuple[str, ...]:
    """The node IDs a cleanup must keep everything reachable from: **every** root, each of which is a system
    prompt node holding the chats that were written under it.

    Every root, not the configured one. The datastore keeps one card per variety, so the chats held under an
    older card hang off a different root — and a sweep given only the current one would call all of them
    unreachable and offer to delete them. Read fresh at each cleanup: roots come and go as the configuration
    changes, and a stale list here is the same mistake in slower motion."""
    return tuple(datastore.get_all_root_nodes())

def _on_cleanup_committed(result: env) -> None:
    """Acknowledge a completed cleanup on the button that started it (the dialog is gone by now)."""
    n_reclaimed = len(result.deleted_sidecars)
    message = (f"Reclaimed {n_reclaimed} attachment{common_text.plural_s(n_reclaimed)}" if result.deleted_sidecars
               else "Saved; nothing to reclaim")
    gui_animation.flash_button(button="util_cleanup_button",  # tag
                               tooltip=util_cleanup_tooltip,
                               ok=True, message=message, duration=gui_config.acknowledgment_duration)

audio_input_panel = audio_input.DPGAudioInputPanel(app_state=app_state,
                                                   configured_defaults=appstate.configured_defaults(),
                                                   themes_and_fonts=themes_and_fonts,
                                                   save_app_state=lambda: appstate.save(state_file=librarian_config.llm_state_file, state=app_state),
                                                   # The toolbar's mini meter draws the same threshold, and is not the panel's to know about.
                                                   on_threshold_changed=lambda value: setattr(mic_vu_meter, "threshold", value),
                                                   centering_reference_window="librarian_main_window")  # tag

cleanup_dialog = DPGCleanupDialog(datastore=datastore,
                                  get_roots=_get_cleanup_roots,
                                  executor=bg,  # use the same thread pool as our main task manager
                                  themes_and_fonts=themes_and_fonts,
                                  save_app_state=lambda: appstate.save(state_file=librarian_config.llm_state_file, state=app_state),
                                  on_committed=_on_cleanup_committed,
                                  centering_reference_window="librarian_main_window")  # tag

# Set in `_gui_cancel_tasks` (the DPG exit callback) and again, defensively, in `gui_shutdown`. The two
# startup frame callbacks (`_load_initial_animator_settings`, `_build_initial_chat_view`) run on DPG's
# callback thread and can race app teardown: if the user closes the window mid-boot, a callback may still be
# in flight while the context is being destroyed, and creating widgets then segfaults the process (no Python
# `try/except` can catch a crash in DPG's C side — the only safe move is to not make the call). The callbacks
# check this flag and bail.
_shutting_down = False

# Two-phase shutdown (the pattern raven-cherrypick uses; see `raven.cherrypick.app`):
#   1. `_gui_cancel_tasks` — the DPG exit callback. Runs inside `render_dearpygui_frame`, so it may only
#      *signal* cancellation, never wait (see its docstring).
#   2. `gui_shutdown` — called from the render loop's `finally`, on the main thread, once the loop has exited.
#      Does the blocking drains and resource teardown, then the caller destroys the context.
def _gui_cancel_tasks() -> None:
    """DPG exit callback: signal background work to stop, WITHOUT waiting.

    DPG dispatches the exit callback from inside `render_dearpygui_frame`, so this must NOT wait. A background
    task parked in `dpg.split_frame` — the avatar renderer's OpenGL task and the chat-streaming updater both
    do this — can only be released by the render loop completing one more frame, and the render loop is right
    now sitting in this callback. So we only *cancel* here. The frame that fired this callback then completes,
    releasing the `split_frame` waiters; the tasks observe their cancelled flags and exit. The blocking drain
    happens in `gui_shutdown`. Without this split, `destroy_context()` in the `finally` could run while the
    renderer thread is still touching OpenGL — destroying the context under it segfaults the process (notably
    when closing the window mid-boot, where the renderer was just started and is busy).
    """
    global _shutting_down
    _shutting_down = True  # also tells any in-flight startup frame callback to bail before it touches DPG
    chat_controller.cancel_tasks()        # cancel chat / AI-turn / context-prefill tasks (no wait)
    gui_resize_task_manager.clear(wait=False)  # cancel any in-flight GUI resize (it can use split_frame)
    cleanup_dialog.task_manager.clear(wait=False)  # cancel thumbnail loading (it too can use split_frame)
    backend_status_task_manager.clear(wait=False)  # stop watching the LLM backend (it rebuilds the chat view)
    panel_occupancy_task_manager.clear(wait=False)  # stop watching the avatar's video (it swaps panels, and can use split_frame)
    server_status_task_manager.clear(wait=False)  # stop watching Raven-server (it writes into the utility panel's row)
    dpg_avatar_renderer.stop(wait=False)  # signal the avatar renderer's background (OpenGL) task to stop (no wait)
    avatar_controller.stop_tts()          # stop TTS playback (no wait)
    audio_recorder.require().stop()       # the capture task writes the VU readout into DPG widgets (no wait)
dpg.set_exit_callback(_gui_cancel_tasks)

def gui_shutdown() -> None:
    """App exit, second phase: wait for background work to finish and release GUI/server resources.

    Call from the render loop's `finally`, on the main thread, AFTER the loop has exited and AFTER
    `_gui_cancel_tasks` (the exit callback) has already signalled cancellation during the final frame — so the
    `wait=True` drains below complete instead of deadlocking on `split_frame` waiters. Must run before
    `dpg.destroy_context()`, so no background thread is still touching DPG/OpenGL when the context goes away.
    """
    global _shutting_down
    _shutting_down = True  # defensive; normally already set by `_gui_cancel_tasks`
    avatar_controller.stop_tts()  # Stop the TTS speaking so that the speech background thread (if any) exits.
    logger.info("gui_shutdown: entered")
    # Silence the GUI side (idempotent; `_gui_cancel_tasks` already did this, via `chat_controller.cancel_tasks()`,
    # whose first action is `disable_gui_updates()`). The cancelled commit's `finally` will fire `on_indexing_done`
    # from a worker thread, and in-flight chat tasks can fire `on_docs_done` similarly — both would then call
    # `dpg.show/hide_item` on widgets that are already being torn down.
    chat_controller.disable_gui_updates()
    # The audio capture task writes levels into DPG widgets once per audio frame, so it has to be gone
    # before the context is destroyed under it — a DPG call into a dying context is a segfault, not an
    # exception, so the guards on those writes cannot help here.
    audio_recorder.require().stop(wait=True)
    # Stop the watchdog observer first so no new ingest/commit tasks land while we're tearing down,
    # then cancel any in-flight RAG indexing. `hybridir.shutdown` waits for the running commit to exit
    # its per-doc loop, partial-save what was applied, and release `datastore_lock`. This must run
    # before `chat_controller.shutdown()`: any chat task blocked inside `retriever.search` waits on
    # that same `datastore_lock`, so leaving hybridir running here would deadlock the wait=True drain.
    scanner.shutdown()
    hybridir.shutdown()
    gui_resize_task_manager.clear(wait=True)
    backend_status_task_manager.clear(wait=True)
    panel_occupancy_task_manager.clear(wait=True)
    server_status_task_manager.clear(wait=True)
    chat_controller.shutdown()
    avatar_controller.shutdown()
    dpg_avatar_renderer.stop(wait=True)
    # Before the animator is cleared, since the dialog cancels its cursor pulsation through it — and well
    # before `destroy_context`, because an opened dialog runs a tick thread that calls DPG. See
    # `FileDialog.destroy`, which joins that thread and therefore belongs in this phase rather than in the
    # exit callback.
    if _filedialog_attach is not None:
        _filedialog_attach.destroy()
    gui_animation.animator.clear()
    logger.info("gui_shutdown: done")

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

# Attach files dragged in from the file manager, exactly as the attach button does. Installed right after
# `show_viewport` because that call is what makes DPG's window reachable through GLFW on this thread.
#
if opts.qr:
    qroverlay.install()

# One rule rather than one per kind, because a drop mixing an image and a document is a *supported* attach
# and the router rejects drops that straddle two rules. Routing between the two kinds is `_attach_callback`'s
# job anyway — it already does it for the file browser, including the text-only-model gate on images.
filedrop.install(filedrop.make_router([filedrop.DropRule(matches=lambda path: (os.path.isfile(path) and
                                                                               (imagestore.is_supported(path) or
                                                                                docextract.is_supported(path))),
                                                         handler=_attach_callback,
                                                         label="images and documents")],
                                      reference_window="librarian_main_window",  # tag
                                      what="Raven-librarian",
                                      blocked=is_any_modal_window_visible))

# Load default animator settings from disk.
#
# We must defer loading the animator settings until after the GUI has been rendered at least once,
# so that if there are any issues during loading, we can open a modal dialog. (We don't currently do that, though.)
_animator_settings = None
def _load_initial_animator_settings() -> None:
    global _animator_settings

    if _shutting_down:  # window closed before this deferred startup callback even started
        return

    animator_json_path = avatar.assets_path("settings", "animator.json")

    try:
        with open(animator_json_path, "r", encoding="utf-8") as json_file:
            animator_settings = json.load(json_file)
    except FileNotFoundError:
        print(colorizer.colorize(f"AI avatar animator default config file not found at '{animator_json_path}'.", colorizer.Style.BRIGHT, colorizer.Fore.RED) + " Please run `raven-avatar-settings-editor` once to create it.")
        logger.error(f"_load_initial_animator_settings: AI avatar animator default config file not found at '{animator_json_path}'. Please run `raven-avatar-settings-editor` once to create it.")
        sys.exit(255)
    except BaseException:  # yes, also Ctrl+C
        print(colorizer.colorize("Failed to load AI avatar animator default config file.", colorizer.Style.BRIGHT, colorizer.Fore.RED) + " See the Librarian log for details.")
        logger.exception("_load_initial_animator_settings: Failed")
        sys.exit(255)

    animator_settings.update(librarian_config.avatar_config.animator_settings_overrides)

    # Re-check after the (possibly slow) JSON load: this callback runs on DPG's callback thread, so the user
    # may have closed the window while we were here. Everything below starts the avatar and creates DPG widgets
    # (e.g. `configure_backdrop` -> `add_raw_texture`); doing that against a context being torn down segfaults.
    if _shutting_down:
        return

    # Through the controller rather than straight at the API, so that it knows what the avatar's settings
    # are. Anything that changes them *temporarily* - the branch-switch glitch - has to put them back, and
    # the server offers no getter to read them from.
    avatar_controller.load_animator_settings(avatar_record, animator_settings)  # send settings to server
    api.avatar_start(avatar_instance_id)
    dpg_avatar_renderer.start(avatar_instance_id)
    dpg_avatar_renderer.load_backdrop_image(animator_settings["backdrop_path"])
    dpg_avatar_renderer.configure_backdrop(new_width=avatar_panel_w - 16,
                                           new_height=avatar_panel_h - 16,
                                           new_blur_state=animator_settings["backdrop_blur"])
    _animator_settings = animator_settings  # for access from GUI event handlers
    _resize_gui()  # force GUI resize just in case (app startup on 1920x1080 screen)

dpg.set_frame_callback(2, _load_initial_animator_settings)

def _build_initial_chat_view(sender, app_data) -> None:
    if _shutting_down:  # window closed during startup; building chat widgets now would race context teardown (segfault)
        return
    chat_controller.view.build()

    # Keyboard focus is deliberately left alone here, and the reason is worth recording because the obvious
    # thing to write instead — park focus on the chat panel, so the app starts ready to read — is not
    # available and does harm when attempted.
    #
    # `dpg.focus_item` cannot focus a child window. Asked to, it does not merely fail: focus lands on the
    # first navigable item of the enclosing window and is *activated*, which for a text field means it takes
    # the caret. So the instruction meant to send focus away from the composer is one of the few that can
    # reliably put it there.
    #
    # Nothing needs to replace it. ImGui gives the first navigable item nav focus of its own accord, but
    # leaves it *inactive* — no caret — and inactive is what the navigation keys are gated on, so the log is
    # scrollable from the first frame without anyone having been sent anywhere. `Ctrl+Space` activates the
    # composer when the reader wants it.

    # Report the LLM backend if it cannot answer yet, and keep watching until it can. Here rather than at
    # `connect` time because both the status row and the chat view the reconnect rebuilds are DPG widgets,
    # and neither exists until this frame.
    startup_backend_status = llmclient.backend_status(llm_settings)
    if startup_backend_status is not llmclient.backend_ready:
        _refresh_backend_status_pill(startup_backend_status)
        _start_backend_status_poll(delay_first_probe=True)
dpg.set_frame_callback(3, _build_initial_chat_view)

# How often to re-ask whether the avatar has video. Second-scale, because the one thing left for it to
# notice is a stream warming up: the checkbox applies itself directly, and the idle detector announces
# itself through `on_idle`.
_PANEL_OCCUPANCY_TICK_S = 0.2

# One rect, two occupants, and three things with an opinion about which one is in it: the user's
# preference, whether the avatar has video to show, and the poll below. One writer resolves all three,
# under this lock. Two writers would each show their own occupant, and the two panels being siblings in
# one group, the pair would stack rather than replace each other.
_panel_occupancy_lock = threading.RLock()

def _apply_panel_occupancy() -> None:
    """Put the right occupant in the avatar's panel, and stop the avatar's video when it is not the one.

    The chat graph holds the panel when the user asked for it, and also whenever the avatar has nothing to
    show: during the seconds a freshly started stream takes to deliver its first frame, and after the idle
    detector has switched the video off. The avatar gets the panel back when its video returns, if that is
    where the user's preference points.

    Call after anything that changes either half. It does nothing when the panel already holds the right
    occupant, so calling it more often than necessary is free.

    Not callable from the render thread: pausing the animator and re-measuring the subtitle both wait for
    a frame.
    """
    with _panel_occupancy_lock:
        avatar_has_video = avatar_controller.video_available(avatar_record)
        show_graph = app_state["chat_graph_shown"] or not avatar_has_video
        # Suppress only what there is to suppress. A stream that has not yet delivered a frame must be
        # left running: pausing it stops the frames, and the first of those frames is the very thing that
        # would end the warmup — so a suppression applied here would hold the avatar unavailable, and the
        # graph in front of it, for the rest of the session.
        suppress_video = show_graph and avatar_has_video

        if show_graph and not chat_graph_panel.is_shown:
            # Hide the outgoing occupant before showing the incoming one, in both directions.
            dpg_avatar_renderer.hide()
            chat_graph_panel.show()
        elif chat_graph_panel.is_shown and not show_graph:
            # Resume before the swap, so the panel does not come back holding the "[Video is off]" text
            # that pausing put in it.
            avatar_controller.set_video_suppressed(avatar_record, False)
            chat_graph_panel.hide()
            dpg_avatar_renderer.show()
            # A hidden item is not laid out at all, so the subtitle still carries whatever width it had
            # when the graph took the panel — and it is placed by measuring itself. Re-measure now that it
            # renders again, or a caption spoken while the graph was up comes back at the wrong height.
            avatar_controller.reposition_subtitle()

        # Last, and outside the swap: the occupant can stay the same while this changes. A stream
        # finishing its warmup under a graph the user asked for is exactly that case.
        avatar_controller.set_video_suppressed(avatar_record, suppress_video)

# Sequential, so a restart supersedes rather than races the watch already running.
panel_occupancy_task_manager = bgtask.TaskManager(name="librarian_panel_occupancy",
                                                  mode="sequential",
                                                  executor=bg)

def _panel_occupancy_task(task_env: env) -> None:
    """Watch for the avatar's video coming and going, and hand the panel over as it does.

    A poll, for the transitions that announce themselves to nobody: a stream delivering its first frame,
    and a `ping` waking a sleeping avatar. Reading a pair of flags five times a second cannot miss one and
    costs nothing measurable.

    The switch-*off* does not come through here — `on_idle` carries it, because that one is the transition
    where being a tick late is visible: the renderer draws "video is off" into the panel it is losing. The
    poll would still catch it, so this remains the safety net for both directions.
    """
    while not (task_env.cancelled or _shutting_down):
        _apply_panel_occupancy()
        time.sleep(_PANEL_OCCUPANCY_TICK_S)

def _start_panel_occupancy_watch() -> None:
    """Start watching who should hold the avatar's panel. Supersedes a watch already running."""
    if _shutting_down:
        return
    panel_occupancy_task_manager.submit(_panel_occupancy_task, env())

def _apply_saved_panel_choice(sender, app_data) -> None:
    """Start deciding who holds the right-hand panel, which also puts the chat graph back if that is where the user left it.

    Frame 4, which is after the avatar has started (frame 2) and the chat view has been built (frame 3).
    Later rather than at GUI build time for two reasons: the avatar renderer initializes into that panel
    and should not have to do it while the panel is hidden, and the graph's first build wants a datastore
    and a chat view that already exist.
    """
    if _shutting_down:
        return
    _start_panel_occupancy_watch()
    # Raven-server was reachable at startup or the app would have exited, so the watch starts believing that
    # and waits an interval before its first probe.
    _start_server_status_poll(delay_first_probe=True)

    # The blue that says where the keyboard is, on the composer. A *caret* follower, not a focus one:
    # ImGui gives nav focus to the first navigable item of a window by itself, so the composer reports
    # focused from the first frame and a focus-driven mark would be lit before anyone had touched it.
    keyboardmark.install_caret_follower(["chat_field"])  # tag
dpg.set_frame_callback(4, _apply_saved_panel_choice)

# A `SIGTERM` — a plain `kill`, a session manager logging out, a supervisor stopping the app — now leaves
# the loop the way the window's close button does, so the teardown below runs and the chat is saved. The
# datastore is written once, at exit, so the alternative is losing the whole session's messages.
#
# Installed here, after everything that initializes SDL: see `raven.common.quitsignal` for why that
# matters and what was swallowing the signal before.
quitsignal.install(dpg.stop_dearpygui)

# Started here, last, so that a session opens onto a fully built app: this module's globals are what it
# gets, and by this point they are every widget, controller and panel the app has, plus `dpg` itself.
replserver.maybe_start(opts.repl, globals(), f"Raven-librarian {__version__}")

logger.info("App render loop starting.")

exitcode = 0
try:
    # We control the render loop manually to have a convenient place to update our GUI animations just before rendering each frame.
    # The debug overlays — the avatar's crop overlay and FPS counter, the chat graph's metrics readout —
    # live in `front=True` viewport drawlists, which DPG draws above the whole window hierarchy, including
    # over a modal, whose input it correctly blocks but whose pixels it does not. That is not something DPG
    # can be asked to stop, and it is not something either overlay can see, so the app says.
    # See `DPGAvatarRenderer.set_overlays_suppressed`.
    while dpg.is_dearpygui_running():
        update_animations()
        # Each of the two occupants of the avatar column draws its debug overlay in a `front=True`
        # viewport drawlist, and gets told the one thing neither can see for itself. Whether its own
        # panel is up is not that thing — each owns it, and gates on it — so both are told the same.
        # Both no-op when the value is unchanged.
        modal_up = is_any_modal_window_visible()
        dpg_avatar_renderer.set_overlays_suppressed(modal_up)
        chat_graph_panel.set_overlays_suppressed(modal_up)
        dpg.render_dearpygui_frame()

        # Idle throttle: sleep when nothing needs updating (avatar paused, no LLM streaming, no RAG indexing, no recent input).
        if not _is_busy():
            time.sleep(IDLE_SLEEP_S)
    # dpg.start_dearpygui()  # automatic render loop
except KeyboardInterrupt:
    pass  # cleanup will be handled by our DPG exit handler
except Exception:
    exitcode = 1
    logger.exception("Unhandled exception in render loop")
finally:
    logger.info("App render loop exited.")

    # Drive BOTH shutdown phases here, on the main thread — we must NOT rely on DPG having run the exit
    # callback. On a fast (e.g. mid-boot) close, DPG's callback-thread slot can be occupied by a startup
    # frame callback parked in `split_frame`, so the exit callback never fires.
    #   1. `_gui_cancel_tasks` — signal cancellation, no waiting. Sets `_shutting_down` (so a late startup
    #      frame callback bails), flips `gui_updates_safe` off, and cancels the avatar renderer + chat tasks,
    #      so they stop before parking in `split_frame` (which would hang now that the loop is stopped). The
    #      renderer's `split_frame`s self-skip once its task is cancelled (see `_split_frame_unless_stopping`).
    #      Idempotent with the exit-callback invocation, if DPG did run it.
    #   2. `gui_shutdown` — the blocking drain + resource teardown. Safe to wait now: phase 1 already signalled
    #      everything, so nothing remains parked in `split_frame`.
    # Then destroy the context, with no background thread still touching DPG/OpenGL.
    _gui_cancel_tasks()
    gui_shutdown()

    try:
        dpg.destroy_context()
    except BaseException:
        logger.exception("dpg.destroy_context() failed")
    common_utils.bail(exitcode)

def main() -> None:  # TODO: we don't really need this; it's just for console_scripts.
    pass
