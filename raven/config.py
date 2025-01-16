"""Configuration for BibTeX extraction/visualization."""

import torch

from unpythonic.env import env

# --------------------------------------------------------------------------------
# Torch config

# see SillyTavern-extras/server.py
device_string = "cuda:0"
torch_dtype = torch.float16 if device_string.startswith("cuda") else torch.float32

# --------------------------------------------------------------------------------
# Raven importer config

# Model that produces the high-dimensional semantic vectors.
# embedding_model = "sentence-transformers/all-mpnet-base-v2"
# embedding_model = "Snowflake/snowflake-arctic-embed-m"
embedding_model = "Snowflake/snowflake-arctic-embed-l"

# Dimension reduction method for hiD -> 2D conversion, used in plotting the semantic point cloud.
# vis_method = "umap"  # best quality, slow-ish
vis_method = "tsne"  # fast

# Find keywords, for visualizing common terms in clusters. Keyword extraction is somewhat expensive (requires NLP), so this can be disabled.
extract_keywords = True
# extract_keywords = False

# Moved to a command-line option of `visualize.py`.
# Plot a word cloud? Requires `extract_keywords=True`.
# plot_wordcloud = True
# plot_wordcloud = False

# NLP model for spaCy, used in keyword extraction.
spacy_model = "en_core_web_sm"  # small pipeline; fast
# spacy_model = "en_core_web_trf"  # Transformer-based pipeline; more accurate, slower, requires GPU, takes lots of VRAM

summarize = False  # Shorten abstracts using AI; VERY expensive; currently, result is unused anyway.

# summarization_model = "KipperDev/bart_summarizer_model"  # https://huggingface.co/KipperDev/bart_summarizer_model
# summarization_prefix = "summarize: "  # some models need this (see the model card for your particular model)

# summarization_model = "Falconsai/text_summarization"
summarization_model = "ArtifactAI/led_base_16384_arxiv_summarization"
# summarization_model = "ArtifactAI/led_large_16384_arxiv_summarization"

summarization_prefix = ""

# --------------------------------------------------------------------------------
# Stopword list for importer

# This is based on an automated frequency analysis (without these) on a few small datasets of research literature,
# with some manual editing afterward.
#
# When editing this, keep in mind the use case of this program, and keep the list field-agnostic.
# If the embedding model is good, it will do the heavy lifting.
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
publisher_stopwords = ["elsevier", "elsevi",  # "elsevi" is an incorrect lemmatization of "elsevier"
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
custom_stopwords = set(filler_stopwords + scilang_stopwords +
                       metadata_stopwords + copyright_stopwords +
                       publisher_stopwords + misc_stopwords)

# --------------------------------------------------------------------------------
# String normalization, common to importer and GUI.

# Currently, unicode subscript and superscript characters.

regular_to_subscript_numbers = {"0": "₀",
                                "1": "₁",
                                "2": "₂",
                                "3": "₃",
                                "4": "₄",
                                "5": "₅",
                                "6": "₆",
                                "7": "₇",
                                "8": "₈",
                                "9": "₉"}
subscript_to_regular_numbers = {v: k for k, v in regular_to_subscript_numbers.items()}

regular_to_subscript_symbols = {"+": "₊",
                                "-": "₋",
                                "=": "₌",
                                "(": "₍",
                                ")": "₎"}
subscript_to_regular_symbols = {v: k for k, v in regular_to_subscript_symbols.items()}

regular_to_subscript_letters = {"a": "ₐ",
                                "e": "ₑ",
                                "ə": "ₔ",  # latin small letter schwa
                                "h": "ₕ",
                                "i": "ᵢ",
                                "j": "ⱼ",
                                "k": "ₖ",
                                "l": "ₗ",
                                "m": "ₘ",
                                "n": "ₙ",
                                "o": "ₒ",
                                "p": "ₚ",
                                "r": "ᵣ",
                                "s": "ₛ",
                                "t": "ₜ",
                                "u": "ᵤ",
                                "v": "ᵥ",
                                "x": "ₓ",
                                "β": "ᵦ",
                                "γ": "ᵧ",
                                "ρ": "ᵨ",
                                "ϕ": "ᵩ",  # symbol phi (0x3d5)
                                "φ": "ᵩ",  # letter phi (0x3c6)
                                "χ": "ᵪ"}
subscript_to_regular_letters = {v: k for k, v in regular_to_subscript_letters.items()}  # letter phi overrides symbol phi in this inverse

regular_to_subscript = {**regular_to_subscript_numbers,
                        **regular_to_subscript_symbols,
                        **regular_to_subscript_letters}
subscript_to_regular = {v: k for k, v in regular_to_subscript.items()}

regular_to_superscript_numbers = {"0": "⁰",
                                  "1": "¹",
                                  "2": "²",
                                  "3": "³",
                                  "4": "⁴",
                                  "5": "⁵",
                                  "6": "⁶",
                                  "7": "⁷",
                                  "8": "⁸",
                                  "9": "⁹"}
superscript_to_regular_numbers = {v: k for k, v in regular_to_superscript_numbers.items()}

regular_to_superscript_symbols = {"+": "⁺",
                                  "-": "⁻",
                                  "=": "⁼",
                                  "(": "⁽",
                                  ")": "⁾"}
superscript_to_regular_symbols = {v: k for k, v in regular_to_superscript_symbols.items()}

regular_to_superscript_letters = {"i": "ⁱ",
                                  "n": "ⁿ"}
superscript_to_regular_letters = {v: k for k, v in regular_to_superscript_letters.items()}

regular_to_superscript = {**regular_to_superscript_numbers,
                          **regular_to_superscript_symbols,
                          **regular_to_superscript_letters}
superscript_to_regular = {v: k for k, v in regular_to_superscript.items()}

# --------------------------------------------------------------------------------
# Raven-visualizer GUI

# TODO: Section this into subnamespaces?
gui_config = env(  # ----------------------------------------
                 # GUI element sizes, in pixels.
                 main_window_w=1920, main_window_h=1040,  # The default size just fits onto a 1080p screen in Linux Mint.
                 help_window_w=1700, help_window_h=1000,  # The help content is static, these values have been chosen to fit it.
                 word_cloud_w=768, word_cloud_h=768,
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
                 # Animations
                 n_many_searchresults=200,  # Number of data points to reach minimum per-datapoint glow highlight brightness for search results.
                 n_many_selection=30,  # Number of data points to reach minimum per-datapoint glow highlight brightness for selection.
                 glow_cycle_duration=2.0,  # seconds, for glow animations.
                 acknowledgment_duration=1.0,  # seconds, for button flashes upon clicking/hotkey.
                 scroll_ends_here_duration=0.5,  # seconds, for scrolling-past-end animation fadeout.
                 smooth_scrolling=True,  # whether to animate scrolling (all info panel scrolling, except scrollbar and mouse wheel, which are handled internally by DPG)
                 smooth_scrolling_step_parameter=0.8,  # Essentially, a nondimensional rate in the half-open interval (0, 1]; see math comment after `SmoothScrolling`.
                 # ----------------------------------------
                 # Mouse
                 selection_brush_radius_pixels=10,
                 datapoints_at_mouse_max_neighbors=100,  # affects performance
                 # ----------------------------------------
                 # Max numbers of dynamic stuff to put into GUI.
                 # Approximate; we always show at least one item per cluster.
                 max_titles_in_tooltip=10,
                 max_items_in_info_panel=100)
