"""Configuration for raven-visualizer."""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch

from unpythonic.env import env

import dearpygui.dearpygui as dpg

# --------------------------------------------------------------------------------
# Torch config

# NOTE: This configures the client-side devices.
# See also `raven.server.config` for server-side devices.

# Which GPU to use in the BibTeX importer, if available. If not available, CPU fallback is used automatically.
# See also `run-on-internal-gpu.sh` for another way to select the GPU when starting the app, without modifying any files.
devices = {
    "embeddings": {"device_string": "cuda:0",
                   "dtype": torch.float16},
    "nlp": {"device_string": "cuda:0"},  # no configurable dtype
    "sanitize": {"device_string": "cuda:0"},  # used for dehyphenation; no configurable dtype
    "summarization": {"device_string": "cuda:0",
                      "dtype": torch.float16}
}

# --------------------------------------------------------------------------------
# BiBTeX import config

# AI model that produces the high-dimensional semantic vectors, for visualization in `raven-visualizer`.
# Available on HuggingFace. Auto-downloaded on first use.
#
# NOTE: Raven uses embedding models in three places, and they don't have to be the same.
#  - Raven-librarian: RAG backend
#  - Raven-visualizer: producing the semantic map (this setting)
#  - Raven-server: served by the `embeddings` module
#
embedding_model = "Snowflake/snowflake-arctic-embed-l"
# embedding_model = "Snowflake/snowflake-arctic-embed-m"
# embedding_model = "sentence-transformers/all-mpnet-base-v2"

# Dimension reduction method for hiD -> 2D conversion, used for generating the semantic map.
#
# vis_method = "umap"  # best quality, slow
vis_method = "tsne"  # good quality, fast (recommended)

# Whether to detect keywords, for visualizing common terms in clusters.
#
# Keyword extraction is somewhat expensive (requires NLP), so this can be disabled.
# It's useful when browsing the semantic map, so we recommend keeping it enabled.
#
extract_keywords = True
# extract_keywords = False

# NLP model for spaCy, used in keyword extraction.
#
# NOTE: Raven uses spaCy models in three places, and they don't have to be the same.
#  - Raven-visualizer: keyword extraction (this setting)
#  - Raven-librarian: tokenization for keyword search
#  - Raven-server: breaking text into sentences in the `summarize` module
#                  and served by the `nlp` module
#
# Auto-downloaded on first use. Uses's spaCy's own auto-download mechanism. See https://spacy.io/models
#
spacy_model = "en_core_web_sm"  # Small pipeline; fast, runs fine on CPU, but can also benefit from GPU acceleration.
# spacy_model = "en_core_web_trf"  # Transformer-based pipeline; more accurate, slower, requires GPU, takes lots of VRAM.

# Dehyphenate abstracts?
#
# This can fix PDF text bro-ken by hyp-he-na-tion, but may also cause paragraphs to run together
# (in those abstracts that have multiple paragraphs).
dehyphenate = True

# Character-level contextual embeddings by Flair-NLP. Used for dehyphenation of broken text (e.g. as extracted from a PDF file).
#
# NOTE: Raven uses dehyphenation mdoels in two places, and they don't have to be the same.
#  - Raven-visualizer: processing of abstracts during BibTeX import (this setting)
#  - Raven-server: served by the `sanitize` module
#
# This is NOT a HuggingFace model name, but is auto-downloaded (by Flair-NLP) on first use.
#
# This is installed into `~/.flair/embeddings/`.
#
# For available models, see:
#     https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md
#     https://github.com/flairNLP/flair/blob/master/flair/embeddings/token.py
#
# This model is loaded by the `dehyphen` package; omit the "-forward" or "-backward" part
# of the model name, those are added automatically.
#
# At first, try "multi", it should support 300+ languages. If that doesn't perform adequately, then look at the docs.
#
dehyphenation_model = "multi"

summarize = False  # Shorten abstracts using AI? VERY slow; experimental, the results are not used yet.

# AI model used by the `summarize` module, for abstractive summarization.
#
# This is a small AI model specialized to the task of summarization ONLY, not a general-purpose LLM.
#
# NOTE: Raven uses a summarizer model in two places, and they don't have to be the same.
#  - Raven-visualizer: tldr AI summarization of abstracts in importer (this setting)
#  - Raven-server: `summarize` module
#
# `summarization_prefix`: Some summarization models need input to be formatted like
#     "summarize: Actual text goes here...". This sets the prefix, which in this example is "summarize: ".
#     For whether you need this and what the value should be, see the model card for your particular model.
#
# summarization_model = "ArtifactAI/led_base_16384_arxiv_summarization"  # ~650 MB
# summarization_model = "ArtifactAI/led_large_16384_arxiv_summarization"  # ~1.8 GB
# summarization_model = "Falconsai/text_summarization"  # ~250 MB
summarization_model = "Qiliang/bart-large-cnn-samsum-ChatGPT_v3"  # ~1.6 GB, performs well
summarization_prefix = ""  # for all of the summarizers listed above

# summarization_model = "KipperDev/bart_summarizer_model"
# summarization_prefix = "summarize: "

# --------------------------------------------------------------------------------
# Stopword list for importer

# This is based on an automated frequency analysis (without these) on a few small datasets of research literature,
# with some manual editing afterward.
#
# When editing this, keep in mind the use case of this program, and keep the list field-agnostic.
# If the `embedding_model` is good, it will do the heavy lifting.
#
# E.g. don't filter:
#  - "best": e.g. "best practices" (software engineering)
#  - "change": e.g. "rate of change" (physics)
#  - "concern": e.g. "separation of concerns" (software engineering)
#  - "condition": e.g. "condition number" (linear algebra)
#  - "field": e.g. "field theory" (physics)
#  - "improved": some names of methods, e.g. "improved hypercube sampling (IHS)" (statistics)
#  - "lower": e.g. "lower triangular" (matrix algebra)
#  - "method": e.g. "finite element method" (applied mathematics, engineering sciences)
#  - "nearly": e.g. "nearly orthogonal" (machine learning, NLP)
#  - "new": geographic names, e.g. "New York"
#  - "paper": e.g. "paper materials" (papermaking)
#  - "performance": e.g. "high performance computing (HPC)"
#  - "potential": e.g. "potential flow" (fluid dynamics), "electric potential"
#  - "review" (also "survey"): vs. a regular study; a useful distinction
#  - "specfic": e.g. "specific heat capacity" (physics)
#  - "system": equation system vs. a single equation
#  - "term": some particular term in an equation (mathematics)
#  - "time": time-dependent analysis vs. steady-state analysis
#  - "used": recycling, green transition... but usually just "we used method X...". Borderline case?
#  - "value": e.g. "value network"
#  - "well": e.g. "oil well"
#  - "work": e.g. "principle of virtual work" (engineering sciences)
#  - "yield": e.g. "yield statement" (programming)

# filler words
filler_stopwords = ["additionally", "afterward", "already", "also", "although", "amid", "amidst", "among",
                    "amount",
                    "become", "beyond",
                    "certain", "certainly", "continuously", "could",
                    "directly", "due",
                    "especially",
                    "finally", "furthermore",
                    "firstly", "secondly", "thirdly",
                    "herein", "however",
                    "indeed",
                    "may", "moreover", "much", "must",
                    "nevertheless",
                    "often", "onto", "overall",
                    "particular",
                    "per",
                    "respectively",
                    "should",
                    "thereby", "though", "thus", "together", "toward", "towards",
                    "upon",
                    "via",
                    "way",
                    "within", "without", "would",
                    "yet"]

# typical scientific language
scilang_stopwords = ["ability", "able",
                     "academia", "academic",
                     "achieve", "achieved",
                     "advantage",
                     "assess",
                     "aim", "application", "analysis", "approach", "area", "article",
                     "approximately",
                     "available",
                     "benefit",
                     "better", "breakthrough",
                     "case",
                     "characterize", "characterise",
                     "compare", "compared", "comparison",
                     "consider", "consideration", "considered", "considering",
                     "corresponding",  # also "corresponding author"
                     "currently",  # but not "current" (as in electrical current)
                     "demonstrate", "demonstrated", "demonstration",
                     "describe",
                     "develop", "developed", "development",
                     "discussed",
                     "enhanced", "evaluate", "evaluated", "evaluation", "exhibit",
                     "establish",
                     "examine", "examined", "examination",
                     "facilitate",
                     "finding",
                     "given", "good",
                     "help", "helped",
                     "identified", "identify",
                     "importance", "important",
                     "improve",
                     "indicate", "indicated", "indication",
                     "introduce",  # what about "introduction"? Useful tutorial text type?
                     "investigate", "investigation",
                     "involve",
                     "main", "mainly",
                     "maintained",
                     "novel",
                     "offer",
                     "possible",
                     "present", "promising", "promote", "proposed",
                     "progress",
                     "prospect",  # what about "prospecting" as in looking for valuable metals?
                     "provide", "provides",
                     "publish", "published",
                     "recent", "related", "reported", "research",
                     "regard",
                     "result", "resulted", "resulting",
                     "reveal", "revealed",
                     "select",
                     "severe",
                     "show", "showed", "shown",
                     "significant", "significantly",
                     "similar", "similarly",
                     "study",
                     "substantial", "substantially",
                     "success", "successful", "successfully",
                     "suggest", "suggested",
                     "suitable", "suited",
                     "summarize", "summarized", "summary",
                     "tackle", "tackled", "tackling",
                     "technical",
                     "test", "typical", "typically",
                     "understand", "understanding", "understood",
                     "undertaken", "undertook",
                     "utilization", "utilize", "utilized", "utilizing",
                     "vary",
                     "widely", "widespread"]
metadata_stopwords = ["academy", "author", "conference", "journal", "institute", "proceedings", "report"]
copyright_stopwords = ["all", "right", "reserved"]

# some common publisher names
publisher_stopwords = ["elsevier", "elsevi",  # "elsevi" is an incorrect lemmatization of "elsevier", though I suppose it makes sense (and leaves me wondering what kind of publication would be the elseviest).
                       "springer",  # we can't do the same for "spring", even though "springer" is much springer than a regular spring.
                       "wiley",
                       "llc"]  # Some Company Name LLC

# uncategorized
misc_stopwords = ["center", "constructed", "containing",
                  "different",
                  "excellent",
                  "great",
                  "higher", "highly",
                  "include", "included", "including",
                  "increase", "increased",
                  "like",
                  "made", "make",
                  "role",
                  "sub",  # "<sub>...</sub>" in HTML abstracts
                  "sudden", "superior",
                  "superb",
                  "usage", "use", "using",
                  "various"]

# The final set just combines all of the above.
# This is the stopword set used by `raven.visualizer.importer` during keyword detection.
custom_stopwords = set(filler_stopwords + scilang_stopwords +
                       metadata_stopwords + copyright_stopwords +
                       publisher_stopwords + misc_stopwords)

# --------------------------------------------------------------------------------
# Raven-visualizer GUI

# For `word_cloud_colormap` below, see colormaps provided by Matplotlib:
#     https://matplotlib.org/stable/gallery/color/colormap_reference.html

# For `plotter_colormap` below, see colormaps provided by DPG:
#     https://dearpygui.readthedocs.io/en/latest/_modules/dearpygui/dearpygui.html?highlight=mvPlotColormap#
#
# From section "Constants":
#     mvPlotColormap_Default=internal_dpg.mvPlotColormap_Default
#     mvPlotColormap_Deep=internal_dpg.mvPlotColormap_Deep
#     mvPlotColormap_Dark=internal_dpg.mvPlotColormap_Dark
#     mvPlotColormap_Pastel=internal_dpg.mvPlotColormap_Pastel
#     mvPlotColormap_Paired=internal_dpg.mvPlotColormap_Paired
#     mvPlotColormap_Viridis=internal_dpg.mvPlotColormap_Viridis
#     mvPlotColormap_Plasma=internal_dpg.mvPlotColormap_Plasma
#     mvPlotColormap_Hot=internal_dpg.mvPlotColormap_Hot
#     mvPlotColormap_Cool=internal_dpg.mvPlotColormap_Cool
#     mvPlotColormap_Pink=internal_dpg.mvPlotColormap_Pink
#     mvPlotColormap_Jet=internal_dpg.mvPlotColormap_Jet
#     mvPlotColormap_Twilight=internal_dpg.mvPlotColormap_Twilight
#     mvPlotColormap_RdBu=internal_dpg.mvPlotColormap_RdBu
#     mvPlotColormap_BrBG=internal_dpg.mvPlotColormap_BrBG
#     mvPlotColormap_PiYG=internal_dpg.mvPlotColormap_PiYG
#     mvPlotColormap_Spectral=internal_dpg.mvPlotColormap_Spectral
#     mvPlotColormap_Greys=internal_dpg.mvPlotColormap_Greys


# TODO: Section this into subnamespaces?
gui_config = env(  # ----------------------------------------
                 # GUI element sizes, in pixels.
                 main_window_w=1920, main_window_h=1040,  # The default size just fits onto a 1080p screen in Linux Mint.
                 help_window_w=1700, help_window_h=1000,  # The help content is static, these values have been chosen to fit it.
                 importer_w=600,
                 info_panel_w=600,
                 info_panel_header_h=40,  # The title section and the navigation controls section both have this height.
                 title_wrap_w=500,  # Note there will be two columns of buttons to the left of each item title.
                 main_text_wrap_w=540,  # For wrapping the abstract.
                 info_panel_reserved_h=230,  # In the left part of the app window, how much vertical space to leave to GUI elements *other than* the item info content area.
                 toolbar_inner_w=36,  # Width of the content area of the "Tools" toolbar.
                 toolbar_separator_h=12,  # Height of a section separator spacer in the toolbar.
                 toolbutton_w=30,  # Width of a toolbutton in the "Tools" toolbar.
                 toolbutton_indent=None,  # The default `None` means "centered" (the value is then computed and stored while setting up the GUI).
                 info_panel_button_w=26,  # Width of the inline buttons in the info panel. Same width as a DPG arrow button so all buttons align properly.
                 annotation_tooltip_w=800,  # Just the width; height is automatic depending on content.
                 font_size=20,  # Also in pixels.
                 # ----------------------------------------
                 # Word cloud
                 #
                 word_cloud_w=768, word_cloud_h=768,
                 word_cloud_background_color="black",
                 word_cloud_colormap="viridis",  # Matplotlib colormap (name as string)
                 # ----------------------------------------
                 # Plotter
                 #
                 # default colors
                 plotter_background_color=(37, 37, 38),  # measured from DPG default theme using GIMP
                 plotter_grid_color=(60, 60, 64),  # measured from DPG default theme using GIMP, from the major tick grid
                 #
                 # # light colors
                 # plotter_background_color=(255, 255, 255),
                 # plotter_grid_color=(255, 128, 64),
                 #
                 plotter_colormap=dpg.mvPlotColormap_Viridis,  # DPG colormap (one of the `dpg.mvPlotColormap_*` constants)
                 plotter_search_results_highlight_color=(255, 96, 96),  # Raven default red
                 plotter_selection_highlight_color=(96, 255, 255),  # Raven default cyan
                 # ----------------------------------------
                 # Animations
                 n_many_searchresults=200,  # Number of data points to reach minimum per-datapoint glow highlight brightness for search results.
                 n_many_selection=30,  # Number of data points to reach minimum per-datapoint glow highlight brightness for selection.
                 glow_cycle_duration=2.0,  # seconds, for glow animations.
                 acknowledgment_duration=1.0,  # seconds, for button flashes upon clicking/hotkey.
                 scroll_ends_here_duration=0.5,  # seconds, for scrolling-past-end animation fadeout.
                 smooth_scrolling=True,  # whether to animate scrolling (all info panel scrolling, except scrollbar and mouse wheel, which are handled internally by DPG)
                 smooth_scrolling_step_parameter=0.8,  # Essentially, a nondimensional rate in the half-open interval (0, 1]; see math comment after `raven.common.gui.animation.SmoothScrolling`.
                 # ----------------------------------------
                 # Mouse
                 selection_brush_radius_pixels=10,
                 datapoints_at_mouse_max_neighbors=100,  # affects performance
                 # ----------------------------------------
                 # Max numbers of dynamic stuff to put into GUI.
                 # Approximate; we always show at least one item per cluster.
                 max_titles_in_tooltip=10,
                 max_items_in_info_panel=100)
