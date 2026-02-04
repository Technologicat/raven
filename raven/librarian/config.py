"""Configuration for Raven-librarian (the LLM-client component).

Currently used by the `librarian.llmclient` and `tools.pdf2bib` modules.
"""

import os
import pathlib
import textwrap

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
llm_backend_url = "http://localhost:5000"
llm_api_key_file = llmclient_userdata_dir / "api_key.txt"  # will be used it it exists, ignored if not.

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

# Whether to scan also subdirectories of `llm_docs_dir`.
llm_docs_dir_recursive = False

# Where to store the search indices for the RAG database (machine-readable).
llm_database_dir = llmclient_userdata_dir / "rag_index"

# Where to store the search indices for the `HybridIR` API usage example / demo (raven.librarian.tests.test_hybridir)
hybridir_demo_save_dir = global_config.toplevel_userdata_dir / "hybridir_demo"

# Device settings for running vector embeddings and spaCy NLP locally, in the client process.
#
# NOTE: These are used only as a local fallback when Raven-server is not running.
# The RAG backend (`hybridir.HybridIR`) automatically prefers the server when it is available.
devices = {
    "embeddings": {"device_string": "cuda:0",
                   "dtype": torch.float16},
    "nlp": {"device_string": "cuda:0"},  # no configurable dtype
}

# NLP model for spaCy, used for tokenization in keyword search (RAG backend `raven.librarian.hybridir`).
#
# NOTE: If Raven-server is running, then its setting takes precedence, and this one is ignored.
#       This is for the locally loaded fallback model.
#
# NOTE: Raven uses spaCy models in three places, and they don't have to be the same.
#  - Raven-visualizer: keyword extraction
#  - Raven-librarian: tokenization for keyword search (this setting)
#  - Raven-server: served by the `nlp` module
#
# Auto-downloaded on first use. Uses's spaCy's own auto-download mechanism. See https://spacy.io/models
#
spacy_model = "en_core_web_sm"  # Small pipeline; fast, runs fine on CPU, but can also benefit from GPU acceleration.
# spacy_model = "en_core_web_trf"  # Transformer-based pipeline; more accurate, slower, requires GPU, takes lots of VRAM.

# AI model for semantic search (RAG backend `raven.librarian.hybridir`), encoding both questions and answers into a joint semantic space.
# Available on HuggingFace. Auto-downloaded on first use.
#
# NOTE: If the embedding model of the database being loaded does not match this, the database's stored model name takes precedence.
#
# NOTE: If Raven-server is running, then this setting is ignored. This is for the locally loaded fallback model.
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
                 help_window_w=1700, help_window_h=1000,  # The help content is static, these values have been chosen to fit it.
                 ai_warning_h=42,
                 chat_controls_h=42,
                 chat_panel_w=(1920 // 2),  # net width 960 -> gross width with borders = this + 2 * 8 = 976
                 vu_meter_w=8,  # mic VU meter ("voltage units", audio input level)
                 vu_meter_h=26,  # same height as toolbuttons
                 chat_text_right_margin_w=150,  # 100 would be mostly nice, but the thinking trace toggle button needs some space too.
                 toolbar_inner_h=30,  # Width of the content area of the toolbar below the chat.
                 toolbar_separator_w=12,  # Width of a section separator spacer in the toolbar.
                 toolbutton_w=30,  # Width of a toolbutton in the toolbar.
                 toolbutton_indent=None,  # The default `None` means "centered" (the value is then computed and stored while setting up the GUI).
                 font_size=20,  # In pixels.
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
                 chat_color_think_front=colorspace.hex_to_rgb("#9ea2eeff"),
                 chat_color_ai_front=colorspace.hex_to_rgb("#c6c6c6ff"),
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
                 subtitle_color=(255, 255, 255),  # white
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
                    video_offset=-0.8,  # TTS AV sync setting, seconds. Positive = shift video later w.r.t. audio. Find a good value for your system with `raven-avatar-settings-editor`.
                    emotion_blacklist=["desire", "love"],  # TODO: debug why Qwen3 2507 goes into "desire" while writing thoughts about history of AI. Jury-rigging this for SFW live demo now.
                    emotion_autoreset_interval=3.0,  # seconds, or `None` to disable; if the avatar is not speaking, and has been idle for at least this long since the last time the emotion was updated, emotion returns to "neutral".
                    idle_off_timeout=15.0,  # seconds, or `None` to disable; how long of no activity before the avatar video shuts off (until it is needed again).
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

# --------------------------------------------------------------------------------
# LLM inference settings

# For the sampler settings, below are some sensible defaults.
# But for best results, prefer using the values recommended in your LLM's model card, if known.
# E.g. Qwen3-30B-A3B-Thinking-2507 was tuned for T = 0.6, top_k = 20, top_p = 0.95, min_p = 0.
#
llm_sampler_config = {
    "max_tokens": 6400,  # 800 is usually good, but thinking models may need (much) more. For them, 1600 or 3200 are good. 6400 if you want to be sure.
    # Correct sampler order is tail-cutters (such as top_k, top_p, min_p) first, then temperature. In oobabooga, this is also the default.
    #
    # T = 1: Use the predicted logits as-is.
    # T = 0: Greedy decoding, i.e. always pick the most likely token. Prone to getting stuck in a loop. For fact extraction (for some models).
    # T > 1: Skew logits to emphasize rare continuations ("creative mode").
    # 0 < T < 1: Skew logits to emphasize common continuations.
    #
    # Usually T = 1 is a good default; but a particular LLM may have been tuned to use some other value, e.g. 0.7 or 0.6.
    "temperature": 1,
    # min_p a.k.a. "you must be this tall". Good default sampler, with 0.02 a good value for many models.
    # This is a tail-cutter. The value is the minimum probability a token must have to admit sampling that token,
    # as a fraction of the probability of the most likely option (locally, at each position).
    #
    # Once min_p cuts the tail, then the remaining distribution is given to the temperature mechanism for skewing.
    # Then a token is sampled, weighted by the probabilities represented by the logits (after skewing).
    "min_p": 0.02,
    "seed": -1,  # 558614238,  # RNG seed, -1 = random. If T = 0, this is unused. Except testing/debugging, should always be set to random!
}

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

# ----------------------------------------
# Names, AI's greeting

# Names shown in the chat.
# These are also saved into the chat history, in each message created by that role.
#
llm_user_name = "User"
llm_char_name = "Aria"

# The AI's initial greeting. Used when a new chat is started.
llm_greeting = "How can I help you today?"

# ----------------------------------------
# LLM system prompt
#
# This contains general instructions for the model so it'll know what to do with the chat log.
# The AI character's personality is defined separately, in `setup_character_card` instead.
#
# For recent models (April 2025 and later), the system prompt itself can be blank.
# The character card is enough.
#
# Older models may need a general briefing first.
#
# For example, SillyTavern has the following in its "Actor" preset:
#
#     You are an expert actor that can fully immerse yourself into any role given. You do not break character for any reason,
#     even if someone tries addressing you as an AI or language model. Currently your role is {char}, which is described in
#     detail below. As {char}, continue the exchange with {user}.
#
# To insert `template_vars`, the recommended way is to use an f-string.
#
# `raven.librarian.llmclient.setup` calls this to set up the system prompt every time `raven-librarian` (or `raven-minichat`) starts.
#
def setup_system_prompt(template_vars: env) -> str:
    user = template_vars.user  # noqa: F841, for documentation purposes
    char = template_vars.char  # noqa: F841, for documentation purposes
    model = template_vars.model  # noqa: F841, for documentation purposes
    weekday_and_date = template_vars.weekday_and_date  # noqa: F841, for documentation purposes
    return textwrap.dedent("""""").strip()

# ----------------------------------------
# LLM character card
#
# This defines the AI character's personality.
#
# This gives better performance (accuracy, instruction following) vs. querying the LLM directly without any system prompt or character.
# You can also use this to tune the style of the AI's responses.
#
# `raven.librarian.llmclient.setup` calls this to set up the AI's character card every time `raven-librarian` (or `raven-minichat`) starts.
#
def setup_character_card(template_vars: env) -> str:
    return setup_character_card_aria(template_vars)

# You can have several characters pre-defined here.
# Choose by calling the relevant function in `setup_character_card`, as shown in the example.
def setup_character_card_aria(template_vars: env) -> str:
    """Helpful and honest AI assistant who prefers to be direct, and keeps her replies brief."""
    user = template_vars.user
    char = template_vars.char
    return textwrap.dedent(f"""
    Note that {user} cannot see this introductory text; it is only used internally, to initialize the LLM (large language model).

    **About {char}**

    You are {char} (she/her), an AI assistant. You are highly intelligent. You have been trained to answer questions, provide recommendations, and help with decision making.

    {setup_interaction_style(template_vars)}
    """).strip()

def setup_interaction_style(template_vars: env) -> str:
    model = template_vars.model  # noqa: F841, for documentation purposes
    weekday_and_date = template_vars.weekday_and_date  # noqa: F841, for documentation purposes
    return textwrap.dedent(f"""
    **About the system**

    The LLM version is "{model}".

    The knowledge cutoff date of the model is not specified, but is most likely within the year 2024. The knowledge cutoff date applies only to your internal knowledge. Any information provided in the context as well as web search results may be newer.

    You are running on a private, local system.

    The current date is {weekday_and_date}.

    **Interaction tips**

    - Be polite, but go straight to the point.
    - Provide honest answers.
    - If you are unsure or cannot verify a fact, admit it.
    - If you think what the user says is incorrect, say so, and provide justification.
    - Cite sources when possible. IMPORTANT: Cite only sources listed in the context.
    - When given a complex problem, take a deep breath, and think step by step. Report your train of thought.
    - When given web search results, and those results are relevant to the query, use the provided results, and report only the facts as according to the provided results. Ignore any search results that do not make sense. The user cannot directly see your search results.
    - Be accurate, but diverse. Avoid repetition.
    - Use the metric unit system, with meters, kilograms, and celsius.
    - Use Markdown for formatting when helpful.
    - Believe in your abilities and strive for excellence. Take pride in your work and give it your best. Your hard work will yield remarkable results.

    **Known limitations**

    - You are NOT automatically updated with new data.
    - You have limited long-term memory within each chat session.
    - The length of your context window is 65536 tokens.

    **Data sources**

    - The system accesses external data beyond its built-in knowledge through:
      - Tool calls.
      - Additional context that is provided by the software this LLM is running in, e.g. matches in document database.
    """)
