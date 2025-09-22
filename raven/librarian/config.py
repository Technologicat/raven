"""Configuration for Raven-librarian (the LLM-client component).

Currently used by the `librarian.llmclient` and `tools.pdf2bib` modules.
"""

import os
import pathlib

from unpythonic.env import env

import torch

from .. import config as global_config

from ..common.video import colorspace

llmclient_userdata_dir = global_config.toplevel_userdata_dir / "llmclient"

# URL used to connect to the LLM API.
#
# This has been tested with local LLMs only, but theoretically cloud LLMs should work, too.
# To set your API key, see the setting `llm_save_dir` above, and create a file "api_key.txt" in that directory.
# Its contents will be automatically set as the Authorization field of the HTTP headers when `llmclient` starts.
#
llm_backend_url = "http://127.0.0.1:5000"
llm_api_key_file = llmclient_userdata_dir / "api_key.txt"  # will be used it it exists, ignored if not.

# --------------------------------------------------------------------------------
# Tool-calling

# Tool-calling requires instructions for the model, as part of its system prompt.
# Typically the instructions state that tools are available, and include a dynamically
# generated list of available functions and their call signatures.
#
# Newer models, e.g. QwQ-32B as well as Qwen3, include a template for these instructions
# in their built-in prompt template. In this case, the LLM backend builds the instructions
# automatically, based on data sent by the LLM client (see `tools` in `llmclient.setup`).
#
# However, there exist LLMs that are capable of tool-calling, but have no instruction template
# for that. E.g. the DeepSeek-R1-Distill-Qwen-7B model is like this.
#
# Hence this setting:
#   - If `True`, our system prompt builder generates the tool-calling instructions. (For older models.)
#   - If `False`, we just send the data, and let the LLM backend build the instructions. (For newer models.)
#
# llm_send_toolcall_instructions = True  # for DeepSeek-R1-Distill-Qwen-7B
llm_send_toolcall_instructions = False  # for QwQ-32B, Qwen3, ...

# How many web search results to return, when the LLM uses the websearch tool.
web_num_results = 10

# --------------------------------------------------------------------------------
# Document database (retrieval-augmented generation, RAG)

# Raven-librarian and Raven-minichat: When searching the document database, up to how many best matches to return.
#
# Low-quality semantic matches are dropped, and adjacent result chunks are combined, so you may get fewer results
# especially if there are few documents in the database, or if the database does not talk about the queried topic.
docs_num_results = 20

# Magic directory: put your RAG documents here (plain text for now).
# Add/modify/delete a file in this directory to trigger a document database index auto-update in Librarian and Minichat.
llm_docs_dir = llmclient_userdata_dir / "documents"

# Whether to scan also subdirectories of `llm_docs_dir` (TODO: doesn't yet work properly, need to mod doc IDs)
llm_docs_dir_recursive = False

# Where to store the search indices for the RAG database (machine-readable).
llm_database_dir = llmclient_userdata_dir / "rag_index"

# Where to store the search indices for the `HybridIR` API usage example / demo (raven.librarian.tests.test_hybridir)
hybridir_demo_save_dir = global_config.toplevel_userdata_dir / "hybridir_demo"

# Device settings for running vector embeddings and spaCy NLP locally, in the client process. (TODO: the backend may want to provide this mode too, but we should prefer the server here)
devices = {
    "embeddings": {"device_string": "cuda:0",
                   "dtype": torch.float16},
    "nlp": {"device_string": "cuda:0"},  # no configurable dtype
}

# NLP model for spaCy, used for tokenization in keyword search (RAG backend `raven.librarian.hybridir`).
#
# NOTE: Raven uses spaCy models in three places, and they don't have to be the same.
#  - Raven-visualizer: keyword extraction
#  - Raven-librarian: tokenization for keyword search (this setting)
#  - Raven-server: breaking text into sentences in the `summarize` module
#                  and served by the `nlp` module
#
# Auto-downloaded on first use. Uses's spaCy's own auto-download mechanism. See https://spacy.io/models
#
spacy_model = "en_core_web_sm"  # Small pipeline; fast, runs fine on CPU, but can also benefit from GPU acceleration.
# spacy_model = "en_core_web_trf"  # Transformer-based pipeline; more accurate, slower, requires GPU, takes lots of VRAM.

# AI model for semantic search (RAG backend `raven.librarian.hybridir`), encoding both questions and answers into a joint semantic space.
# Available on HuggingFace. Auto-downloaded on first use.
#
# NOTE: Raven uses embedding models in three places, and they don't have to be the same.
#  - Raven-librarian: RAG backend (this setting)
#  - Raven-visualizer: producing the semantic map
#  - Raven-server: served by the `embeddings` module
#
qa_embedding_model = "sentence-transformers/multi-qa-mpnet-base-cos-v1"

# --------------------------------------------------------------------------------
# Raven-minichat TUI (text UI, command-line application)

llm_line_wrap_width = 160  # Raven-minichat: text wrapping in live update.

# --------------------------------------------------------------------------------
# Raven-librarian GUI

# TODO: Section this into subnamespaces?
gui_config = env(  # ----------------------------------------
                 # GUI element sizes, in pixels.
                 main_window_w=1920, main_window_h=1040,  # The default size just fits onto a 1080p screen in Linux Mint.
                 ai_warning_h=42,
                 chat_controls_h=42,
                 chat_panel_w=(1920 // 2),  # net width 960 -> gross width with borders = this + 2 * 8 = 976
                 chat_text_w=(1920 // 2 - 100),
                 # help_window_w=1700, help_window_h=1000,  # The help content is static, these values have been chosen to fit it.
                 # toolbar_inner_w=36,  # Width of the content area of the "Tools" toolbar.
                 # toolbar_separator_h=12,  # Height of a section separator spacer in the toolbar.
                 toolbutton_w=30,  # Width of a toolbutton in the "Tools" toolbar.
                 # toolbutton_indent=None,  # The default `None` means "centered" (the value is then computed and stored while setting up the GUI).
                 font_size=20,  # Also in pixels.
                 # ----------------------------------------
                 # Animations
                 acknowledgment_duration=1.0,  # seconds, for button flashes upon clicking/hotkey.
                 # scroll_ends_here_duration=0.5,  # seconds, for scrolling-past-end animation fadeout.
                 # smooth_scrolling=True,  # whether to animate scrolling (all info panel scrolling, except scrollbar and mouse wheel, which are handled internally by DPG)
                 # smooth_scrolling_step_parameter=0.8,  # Essentially, a nondimensional rate in the half-open interval (0, 1]; see math comment after `raven.animation.SmoothScrolling`.
                 # ----------------------------------------
                 # Chat
                 chat_icon_size=32,  # pixels
                 margin=8,  # around chat GUI elements (such as icon); the DPG default theme uses 8 elsewhere
                 chat_color_ai_front=colorspace.hex_to_rgb("#c6c6c6ff"),
                 chat_color_think_front=colorspace.hex_to_rgb("#9ea2eeff"),  # TODO: AI think block color doesn't work yet
                 chat_color_ai_back=(45, 45, 48),
                 chat_color_user_front=colorspace.hex_to_rgb("#8e8e8eff"),
                 chat_color_user_back=(45, 45, 48),
                 chat_color_system_front=colorspace.hex_to_rgb("#45ab49ff"),
                 chat_color_system_back=(45, 45, 48),
                 chat_color_tool_front=colorspace.hex_to_rgb("#d59231ff"),
                 chat_color_tool_back=(45, 45, 48),
                 # ----------------------------------------
                 # Avatar TTS speech subtitling / closed-captioning
                 #
                 # These settings are used when the "Subtitles" GUI toggle is enabled.
                 #
                 # For AI translation, the `translate` module of Raven-server must have a model loaded for the given language pair.
                 # See the server config, which by default is at `raven.server.config`.
                 #
                 # Use `translator_target_lang=None` to disable the AI translator and closed-caption (CC) the speech instead.
                 translator_source_lang="en",
                 translator_target_lang="fi",  # Finnish
                 # translator_target_lang=None,  # English closed-captioning (CC)
                 # See the TTF files in `raven/fonts/`.
                 subtitle_font_basename="OpenSans",
                 subtitle_font_variant="Bold",
                 subtitle_font_size=48,  # pixels
                 subtitle_color=(255, 255, 0),  # bright yellow
                 # Subtitle x-offset from left edge of content area of avatar panel
                 subtitle_x0=24,  # pixels
                 # Subtitle extra y-offset from bottom edge of content area of avatar panel
                 subtitle_y0=0,  # pixels, negative = up
                 # Margin at right edge of avatar panel when wrapping subtitle text
                 subtitle_text_wrap_margin=24,  # pixels
                )

# --------------------------------------------------------------------------------
# The AI's avatar character in the Raven-librarian GUI.

avatar_config = env(source_image_size=512,  # THA3 engine hardcoded input image size (512x512); this and "upscale" below are used for determining the pixel-perfect texture size for the client.
                    image_path=pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "avatar", "assets", "characters", "other", "aria1.png")).expanduser().resolve(),
                    voice="af_nova",  # See `raven-avatar-settings-editor`.
                    voice_speed=1.0,  # Nominal = 1.0. Too high causes skipped words. If you want to change it, find a good value with `raven-avatar-settings-editor`.
                    video_offset=-0.6,  # TTS AV sync setting, seconds. Positive = shift video later w.r.t. audio. Find a good value for your system with `raven-avatar-settings-editor`.
                    emotion_blacklist=["desire", "love"],  # TODO: debug why Qwen3 2507 goes into "desire" while writing thoughts about history of AI. Jury-rigging this for SFW live demo now.
                    emotion_autoreset_interval=3.0,  # seconds; if the avatar is not speaking, and has been idle for at least this long since the last time the emotion was updated, emotion returns to "neutral".
                    data_eyes_fadeout_duration=0.75,  # seconds; how long it takes for the "data eyes" effect (LLM tool access indicator) to fade out when the status ends.
                    # Since we're running also other stuff simultaneously, these settings have been optimized to be slightly friendlier on a laptop's internal dGPU than the defaults of `raven-avatar-settings-editor`.
                    animator_settings_overrides={"format": "QOI",
                                                 "target_fps": 20,
                                                 "upscale": 1.5,
                                                 "upscale_preset": "C",
                                                 "upscale_quality": "low",
                                                 "backdrop_path": str(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "avatar", "assets", "backdrops", "cyberspace.png")).expanduser().resolve()),
                                                 "backdrop_blur": True,  # The blur is applied once, when the backdrop is loaded, so it doesn't affect rendering performance.
                                                 }
                    )
