"""Configuration for Raven-librarian (the LLM-client component).

Currently used by the `librarian.llmclient` and `tools.pdf2bib` modules.
"""

import math
import os
import pathlib
import textwrap

from unpythonic.env import env

import torch

from .. import config as global_config

from ..client.config import Timeout  # `(connect, read)` timeout tuple with named fields; see `raven.client.config`

from ..common.video import colorspace

llmclient_userdata_dir = global_config.toplevel_userdata_dir / "llmclient"

# The two files the chat frontends persist to. Config rather than per-frontend literals because the GUI and
# the CLI are meant to share one chat history: they did, but only because two separately written pairs of
# filenames happened to agree, with nothing enforcing it and nothing to notice if one drifted.
llm_datastore_file = llmclient_userdata_dir / "chat.json"  # chat node datastore
llm_state_file = llmclient_userdata_dir / "state.json"  # important node IDs for the chat client state
# Attachment sidecars live beside the datastore, in a directory derived from its name — see
# `chattree.PersistentForest.sidecar_dir`.

# URL used to connect to the LLM API.
#
# This has been tested with local LLMs only, but theoretically cloud LLMs should work, too.
# To set your API key, see the setting `llm_save_dir` above, and create a file "api_key.txt" in that directory.
# Its contents will be automatically set as the Authorization field of the HTTP headers when `llmclient` starts.
#
# llm_backend_url = "http://localhost:5000"  # oobabooga default OAI compatible port
llm_backend_url = "http://localhost:1234"  # LM Studio default OAI compatible port
llm_api_key_file = llmclient_userdata_dir / "api_key.txt"  # will be used it it exists, ignored if not.

# Network timeouts for talking to the LLM backend, as `(connect, read)` second pairs passed to `requests`.
# Separate from `raven.client.config.network_timeout` because the LLM backend is a distinct service
# (oobabooga / LM Studio / llama.cpp) with its own latency profile and its own `llm_backend_url`.
#
# The connect timeout bounds the "backend unreachable" case so a request fails fast instead of hanging on
# the OS-level connection attempt. The read timeout is `requests`' *between-bytes* timeout; the
# non-streaming calls here (model info, token count) respond quickly, so a moderate value suffices.
llm_network_timeout = Timeout(connect=10.0, read=120.0)

# The chat-completions stream stays open for the whole generation, which can run for minutes (and prompt
# processing can delay the first token), so bound only the connect; a read timeout would abort a healthy
# generation mid-stream.
llm_network_timeout_streaming = Timeout(connect=10.0, read=None)

# Which OpenAI-compatible backend `llm_backend_url` points at, or `None` to autodetect (the default).
#
# A few request/response details differ between backends; the flavor is determined once at connection time
# (see `llmclient.detect_backend_flavor`) by probing endpoints. One of "oobabooga", "lmstudio", "generic", or
# `None` (autodetect). Set explicitly only to override a misdetection.
llm_backend_flavor = None

# Which model to use, or `None` (default) to use whatever the backend reports as currently loaded.
#
# When set, the name is sent in each request's `model` field. On LM Studio with just-in-time (JIT) model
# loading, this tells the server which downloaded model to load on demand. The string should match a model id
# the backend knows (see the list at `GET {llm_backend_url}/v1/models`).
llm_model = None

# Optional local HuggingFace tokenizer for *exact* token counting against backends without a token-count
# endpoint (LM Studio, generic). `None` (default) disables it; counts then fall back to the backend's `usage`
# stats and a calibrated tokens-per-character estimate.
#
# Value: a path to a directory containing `tokenizer.json` + `tokenizer_config.json`, or a HuggingFace repo id
# (e.g. "Qwen/Qwen3.5-4B"). The tokenizer files are tiny (~10-15 MB) and load in milliseconds. It MUST match
# the served model, or counts will be confidently wrong — `invoke` cross-checks against the backend's reported
# `usage` and logs a warning on a large mismatch. ooba has its own exact endpoint, so this is mainly for
# LM Studio / generic when you co-locate the model files with the client.
llm_tokenizer_path = None

# Idle delay (seconds) before the GUI fires a background "context prefill" on the current branch.
#
# When the chat HEAD settles (new message, branch switch, reroll, ...), the context-fill indicator first
# shows a quick *approximate* token count. After this many seconds of inactivity, the controller sends the
# current prompt to the backend generating essentially no output, which (a) reads back the backend's exact
# `prompt_tokens` — upgrading the indicator from `~X%` to `X%` — and (b) warms the backend KV cache, so the
# user's next turn starts faster. The delay debounces rapid branch hopping (each HEAD change supersedes the
# previous pending prefill); a real generation also cancels it (the live turn warms the cache and reports its
# own exact count). Set to `None` to disable the feature entirely.
context_prefill_idle_delay = 5.0

# Which search engine the websearch tool uses. One of "duckduckgo" or "google".
#
# "duckduckgo" (the default) tolerates automated queries. "google" often serves a CAPTCHA or rate-limits
# headless / scripted searches, so it may fail intermittently depending on your IP and how often you search.
# This is host configuration, not a model choice — the LLM's websearch tool only takes a query.
websearch_engine = "duckduckgo"

# How many web search results to return, when the LLM uses the websearch tool.
web_num_results = 10

# How many rounds of tool calls the LLM may make within a single AI turn, before it has to answer.
#
# One "round" is one assistant message requesting tools, plus their results. The turn ends when the model
# replies without asking for tools, so this is a backstop rather than a normal limit — a model that gets
# what it needs stops on its own, well under any sane value here.
#
# The cap applies to all tools. Any tool whose *failure* suggests an immediate retry can loop — a search
# that finds nothing invites rephrasing, a fetch that is refused invites another URL, and both can be done
# forever. What differs between tools is how often the empty-handed case comes up: a local corpus returns
# nothing at all for most queries outside its subject, while a web search almost always returns something
# plausible-looking. (The server-side memoization of websearch does not help here — it collapses identical
# queries, and a rephrasing loop issues different ones.)
#
# When the cap is reached, the requested calls still run; only the invocation *after* them is told that the
# budget is spent. That ordering is deliberate: stopping the loop the moment the cap is hit would leave the
# history ending on a tool call with no result, which reads as a paused agent loop and prompts yet another
# call rather than a reply.
#
# The number is a resource decision — context window, latency, the user's patience — and not a correctness
# one, because a runaway loop and a thorough one are the same algorithm. It is set well above where models
# actually stop, so that it stays the backstop it is described as: measured against a corpus containing
# literally nothing, qwen3.6-35b-a3b gave up on its own after 9-10 rounds of rephrasing, and a cap of 5 was
# interrupting it less than halfway through what it considered due diligence. Raising this trades latency
# for thoroughness, and the user can end a turn that has gone on too long with Ctrl+G.
max_tool_call_rounds = 20

# How many further rounds the tools stay on offer, answering "not now", before they are withdrawn outright.
#
# Past `max_tool_call_rounds` the tools remain in the schema and any call is refused with an error result
# saying the budget is spent. Two reasons to prefer that to simply withdrawing them: changing the tool
# loadout mid-turn invalidates the backend's KV cache from that point on, and a history calling a tool the
# current request no longer declares is a shape models see little of in training, whereas a tool answering
# "not now" is one they see plenty of.
#
# A refusal cannot guarantee the turn ends, though, so the withdrawal remains as the terminator of last
# resort, and this is how long to try the gentler thing first. Each refusal round costs a full generation,
# and there is no evidence that a second one helps a model the first did not reach — so, one. Set this to 0
# to withdraw the tools immediately at the cap.
max_tool_call_refusal_rounds = 1

# How much of the context window one document fetched by the LLM may occupy, as a fraction.
#
# Text over the limit is truncated in the middle, keeping the beginning and the end, with an explicit
# marker where the omission is. That shape suits a scientific fulltext: the abstract, introduction and
# conclusions survive; the methods section in the middle is what goes.
#
# This ceiling is for text the *model* reaches for on a hunch, having seen a search result. Text the *user*
# attached is governed by `context_reserve_fraction` alone, with no per-document ceiling: an attachment is
# an instruction to read this, and capping it at a tenth of the window would answer that instruction with a
# few pages. Attachments compete with each other for whatever the reserve leaves, share and share alike.
#
# The distinction is *who asked*, not which code path delivered the text — which is worth stating because a
# long page from `webfetch` is stored as an attachment, and so travels the attachment machinery while still
# being a speculative fetch. It is ceilinged here too. What decides is the `source` recorded on the stored
# content-part (see `llmclient.attachment_budget_kind`), so moving a document between those pathways changes
# its budget with it.
docs_fetch_max_fraction_of_context = 0.10

# How much of the context window to keep free for the discussion itself, as a fraction.
#
# Governs every long text that is not the conversation: documents the LLM fetches, and documents the user
# attached (folded into the message at wire-build). A fetch is refused outright when the conversation has
# already grown past this, rather than truncated to whatever slack is left — at that point the useful move
# is a new chat, not a sliver of a document.
#
# This reserve is doing real work, not sitting idle. The size estimate cannot see what the model generates
# *after* the fetch: its own reasoning, which on a thinking model is the single largest consumer of the
# turn. That is what the reserve is for. Tuning it down towards zero on the grounds that "the context is
# only 60% full, so there is plenty of room" is the mistake it exists to prevent.
context_reserve_fraction = 0.25

# Which key sends a chat message: `"enter"` or `"ctrl+enter"`.
#
# The two chords simply trade places, because that is the whole of what the widget offers: ImGui's multiline
# text field knows Enter and Ctrl+Enter, one of which commits the edit and the other of which inserts a
# newline. (Shift+Enter is *not* a third option — it does nothing here, whatever other chat apps have taught
# your fingers.)
#
#   - `"ctrl+enter"` — Ctrl+Enter sends, Enter inserts a newline.
#   - `"enter"`      — Enter sends, Ctrl+Enter inserts a newline.
#
# `"ctrl+enter"` is the default, which is not what most chat frontends do — but it is what most *editors*
# do, and this is a multiline field where Enter otherwise means newline everywhere else in the system. The
# asymmetry in what going wrong costs settles it: under `"enter"`, a Return reached for mid-thought sends
# the message half-written, and Librarian has no message editing yet, so getting back to what you were
# typing means copying the sent message out with its copy button, pasting it back, deleting it, and
# resuming. Under `"ctrl+enter"` the same slip inserts a newline, which is one backspace. Revisit this
# default if message editing lands — the argument is about the cost of the mistake, not about the chord.
#
# Set it to `"enter"` if that is the muscle memory you arrive with; for short questions it is fewer keys.
send_message_key = "ctrl+enter"

# How long a fetched document has to be, in characters, before the chat log shows it as an attachment
# chip plus an opening excerpt rather than in full.
#
# Only a tool result that declares itself a *document* is eligible (currently `webfetch`); a websearch's
# list of links stays inline at any length, because the links are the result and the user wants to click
# them. The model is unaffected either way: an attached document's text is folded back into the message
# at wire-build, so it reads the same bytes it would have read inline.
#
# The threshold trades two costs against each other. Below it, the whole result sits in the log, which is
# what a short page should do — hiding three paragraphs behind a chip is worse than showing them. Above
# it, one fetch buries the conversation under dozens of screens *and* writes the same bytes into the
# datastore JSON. 4000 characters is around two screenfuls, which is about as much as can be scrolled
# past without losing the thread.
tool_result_attachment_threshold = 4000

# How much of an attachment-ified tool result to show inline as an excerpt, in characters.
#
# Cut at a paragraph boundary, so the excerpt ends where a paragraph does rather than mid-sentence; a
# single paragraph longer than this is cut at a word boundary instead. A fetched page opens with a
# source header and its title, so the first few hundred characters are mostly provenance — hence a
# budget generous enough to reach actual prose underneath.
#
# Not zero, on purpose: a tool result the user cannot see at all is a step backwards from the
# what-you-see-is-what-you-get design the chat log otherwise has.
tool_result_preview_characters = 800

# --------------------------------------------------------------------------------
# webfetch tool — client-side access policy
#
# The network-level safety of webfetch (refusing private-network addresses and non-HTTP(S)
# schemes) is enforced server-side; see `raven.server.config`. The settings here constrain
# the AI's *initiative* — which public sites the model may decide to visit on its own — and
# live client-side because they need the conversation context.

# Suggested baseline allowlist for the median scientific user — the starting point you extend
# with field-specific entries (e.g. "lesswrong.com", "transformer-circuits.pub" for AI alignment).
# Not active unless you assign it to `webfetch_allowlist` below.
#
# Declared before `webfetch_allowlist` so that you can opt in with `webfetch_allowlist =
# webfetch_default_allowlist` (this is a plain Python module, evaluated top to bottom).
#
webfetch_default_allowlist = [
    # Citation / metadata
    "doi.org",
    "api.crossref.org",

    # Preprints and open peer review
    "*.arxiv.org",
    "*.biorxiv.org",
    "*.medrxiv.org",
    "openreview.net",

    # Major publishers / journals
    "*.nature.com",
    "*.science.org",
    "*.pnas.org",
    "*.plos.org",
    "*.cell.com",
    "*.springer.com",
    "link.springer.com",

    # Search / discovery
    "scholar.google.com",
    "*.semanticscholar.org",
    "researchgate.net",
    "www.researchgate.net",

    # Biomedical
    "*.ncbi.nlm.nih.gov",

    # Code / models / data
    "github.com",
    "raw.githubusercontent.com",
    "gist.github.com",
    "huggingface.co",

    # General reference
    "*.wikipedia.org",
    "*.wikimedia.org",
]

# Domain allowlist for the webfetch tool.
#
# - `None` (default): unrestricted. The model may fetch any public URL (still subject to the
#   server-side private-network / scheme blocks).
# - A list of host patterns: the model may only fetch listed hosts. Patterns are either an
#   exact host ("doi.org") or a wildcard ("*.arxiv.org", which matches the apex and any
#   subdomain). URLs the *user* types into their latest message are auto-allowed for that turn
#   regardless of this list (a user-typed URL is the user's intent, not the model's).
#
# To enable the curated scientific baseline above, set this to `webfetch_default_allowlist`
# (optionally extended with your own field-specific entries).
#
# Setting an allowlist switches on an opt-in "constrain the AI's initiative" mode. While it is
# `None`, that whole mode is dormant: the auto-allow-of-user-typed-URLs logic and the
# `webfetch_trust_search_results` setting below have no effect, because there is no gate for them
# to relax. They become meaningful only once you set an allowlist here.
#
webfetch_allowlist = None

# DANGEROUS — leave this off unless you understand the risk.
#
# This setting only has an effect when `webfetch_allowlist` (above) is set; with the default
# `None` allowlist there is no gate, so the model can already follow any link. Within the
# allowlist mode, it relaxes one specific restriction:
#
# When True, URLs appearing in `websearch` tool-results are auto-allowed for the current turn,
# so the model can "search, then follow a link" even if the host is not on `webfetch_allowlist`.
# This is a real prompt-injection vector: a poisoned search-result snippet could embed a URL
# crafted to make the model fetch it, which could then inject further instructions. Off by
# default; even with an allowlist set, the model can still follow a search-result link if you add
# its host to the allowlist, or if you forward the URL yourself.
#
webfetch_trust_search_results = False

# --------------------------------------------------------------------------------
# Multimodal (image) input and storage
#
# When a VLM (vision-capable model) is loaded, the user can attach images to a message. Images are stored as
# *sidecar files* in a directory next to the chat datastore JSON (`<datastore>.sidecars/`), referenced from
# messages by a Raven-internal `sidecar:<filename>` URL. On wire-send, `llmclient.invoke` substitutes a real
# `data:` URL by reading the sidecar bytes. No `https://` URLs ever land in a stored datastore, so a saved
# chat reloads without network access, survives the source page going away (link rot), and never phones home
# to a remote host when reopened — even if the image originally came from a remote URL.

# Downsample attached images larger than this many megapixels before storing, aspect ratio preserved, via
# `raven.common.image.lanczos`. 1.0 MP is right at the resolution most current VLMs natively expect (Gemma 4
# 1024², Qwen-VL ≈ 1340 patches at 1024²), so anything larger is wasted tokens the model would resize away.
# Set to `None` to store originals at full resolution (no downsampling).
image_store_max_megapixels = 1.0

# When an image is downsampled on store, also keep the full-resolution original as a second sidecar
# (`<hash>.original.<ext>`). Makes the datastore self-contained and future-proof (re-downsample to a different
# target, re-export at full quality, send to a higher-resolution VLM later). Set `False` on disk-constrained
# setups to keep only the downsampled copy. No effect on images that don't need downsampling (the primary
# sidecar IS the original in that case).
store_original_image = True

# Where per-item / bulk "Save a copy to staging" rescues land when cleaning up unreferenced sidecars (the manual
# "Clean up & save" flow). User-level, not per-datastore — a recovered attachment is user data, not chat-specific.
# Covers both kinds: an orphaned sidecar may be an attached image or an attached document.
attachment_staging_dir = global_config.toplevel_userdata_dir / "staging" / "recovered_attachments"

# Side of the square thumbnail tiles in the cleanup preview's image grid, in pixels. Smaller than an inline
# chat image (`gui_config.chat_inline_image_*`) because the grid shows many at once, and the job here is
# recognition — "ah, that one" — not reading detail out of the picture.
cleanup_thumbnail_size = 140

# Estimated per-image token cost, for the context-fill budget (a VLM image consumes non-trivial context that a
# text-only tokens-per-character estimate can't see). Keyed by a lowercase substring matched against the
# loaded model's family / arch / id; first match wins, `None` key is the fallback for unknown families. Each
# value is either a flat token count (int) or a callable `(height, width) -> int` for models whose cost
# scales with resolution.
#
# These are conservative estimates: the budget self-corrects from the backend's real `usage.prompt_tokens`
# after the first image-bearing call (same mechanism as the tokens-per-character calibration), so
# over-estimating here only means the pre-send indicator reads slightly high until the first exact count
# lands. Formulas as of 2026-05, from each family's published image-tiling scheme (Gemma 4's discrete
# per-image budget, LLaVA's 336-px tiles, Qwen-VL's 28-px patch grid).
gemma4_visual_token_budget = 1120  # Gemma 4's per-image budget is a server-side knob (70/140/280/560/1120); assume the max unless you know your server's setting.
llm_image_token_cost = {
    "gemma4": lambda h, w: gemma4_visual_token_budget,
    "llava-1.5": 576,                                                  # single 336x336 tile
    "llava": 2880,                                                     # LLaVA-NeXT: up to 5 tiles x 576; assume the max
    "qwen": lambda h, w: min(16384, math.ceil(h / 28) * math.ceil(w / 28)),  # Qwen-VL dynamic: ~1340 tokens at 1024x1024
    None: 1000,                                                        # unknown family: conservative placeholder
}

# --------------------------------------------------------------------------------
# Document database (retrieval-augmented generation, RAG)

# Raven-librarian and Raven-minichat: When searching the document database, up to how many best matches to return.
#
# Low-quality semantic matches are dropped, and adjacent result chunks are combined, so you may get fewer results
# especially if there are few documents in the database, or if the database does not talk about the queried topic.
#
# The real budget here is not the context window but *prefill time*: the retrieved block differs every turn, so
# no backend can cache it, and the model re-reads all of it before answering. Measured on a 12k-abstract corpus
# against qwen3.6-35b-a3b, prefill is near-linear at ~5000 tokens/s and the recall bought per second is not:
#
#     k=20   7.4k tokens   1.7 s   74.7% recall@k
#     k=50  18.2k tokens   3.4 s   84.8%      <-- +10.1 points at 0.17 s per point
#     k=100 37.0k tokens   7.1 s   89.9%          +5.1 points at 0.73 s per point
#     k=200 75.0k tokens    16 s   96.0%          unusable in conversation
#
# So 50 is the knee: the last value whose latency stays conversational, and roughly a quarter the price per
# recall point of any step beyond it. Lower it if your backend prefills slowly; raising it past 100 trades a
# few points of recall for a wait the user will notice.
docs_num_results = 50

# Longest single search result, in characters; `None` for unlimited.
#
# Results are stitched back together from adjacent chunks of the same document, and a run of adjacent
# chunks can be arbitrarily long — so without a cap, `docs_num_results` bounds the *number* of results
# and nothing bounds their size. The prefill cost of a turn is then unbounded in principle, and in
# practice varies by an order of magnitude with how much stitching happened to occur.
#
# The number is in *document* characters, so read it against the chunking: chunks are 1000 characters
# with 25% overlap, hence each additional chunk extends a span by 750. A cap of 2000 therefore admits
# two chunks (spanning 1750) and starts a new result at the third:
#
#     1 chunk  1000     3 chunks  2500
#     2 chunks 1750     4 chunks  3250
#
# With `docs_num_results = 50` that puts the worst case near 25000 tokens — about the measured 5-second
# prefill, and close enough to the 3.4 s typical case that the ceiling is no longer a surprise. That is
# what the cap is for; it is chosen to make the cost predictable, not to improve retrieval.
#
# A capped run is not truncated. It comes back as several results covering the same text, so nothing is
# lost but the seam. Raise it if you would rather have longer continuous passages than a bounded worst
# case, or set it to `None` for the old unlimited behavior.
#
# This setting is a stopgap. The reason results vary in length at all is that they are assembled from
# however many neighbouring chunks happened to be retrieved, which is not a statement about how much
# context the passage needs — so the eventual fix is to take a fixed window around each match from the
# stored document text, at which point every result is the same size and no cap is required. See the
# `TODO` above `_build_full_id_to_record_index` in `hybridir.py`.
docs_max_result_length = 2000

# How many previously consulted documents to list back to the LLM (`list_consulted_documents`).
#
# The automatic search injects its matches for one turn and then drops them, so a follow-up question
# arrives with the reply in view and the material behind it gone. The list hands back the IDs, which
# `fetch_document` can turn into text again. Newest first, so the cap drops the documents the conversation
# has moved furthest away from.
max_consulted_documents_listed = 30

# Magic directory: put your RAG documents here.
# Add/modify/delete a file in this directory to trigger a document database index auto-update in Librarian and Minichat.
llm_docs_dir = llmclient_userdata_dir / "documents"

# File types ingested into the document database. Plain-text formats are read verbatim; the rest have their text
# layer extracted (born-digital PDFs; a scanned/image-only PDF has no text to extract and is skipped, as does an
# office document whose content is all pictures). Handled by `raven.common.docextract`, which is the single
# text-extraction backend. This list can only *narrow* what it supports: an entry there is no reader for is
# dropped with a warning at startup (`docextract.Extractor.restricted_to`), never ingested as line noise.
llm_docs_exts = [".txt", ".md", ".rst", ".org", ".bib", ".tex", ".pdf",
                 ".docx", ".pptx", ".odt", ".odp",
                 ".html", ".htm"]

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
    "embeddings": {"device_string": "gpu",
                   "dtype": torch.float16},
    "nlp": {"device_string": "gpu"},  # no configurable dtype
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
                 # The AI-disclosure label below the chat. Two lines' worth of height, because the
                 # disclosure states two separate things (that the interlocutor is an AI, and that its
                 # output needs checking) and does not fit on one line at the default window width.
                 ai_warning_h=62,
                 # Wrap width for the label text. An upper bound, not the rendered width: at this setting the
                 # break falls between "accuracy" and "depend", giving lines of 521 and 518 px. The usable
                 # range is 521 (the first line's own width) to 576 (one word more would fit); the midpoint
                 # leaves room on both sides for font-metric drift.
                 ai_warning_w=550,
                 # Composer (chat input) geometry. The composer is a vertical stack: multiline text field,
                 # an optional staged-image thumbnail strip, and a button toolbar. Its outer height
                 # (`chat_controls_h`) is FIXED so that showing/hiding the strip never rescales the chat or
                 # avatar panels (the avatar panel height tracks the chat panel height). When the strip
                 # appears it steals height from the text field instead (field and strip heights adjust together).
                 #
                 # The heights below are DERIVED from the live theme metrics: font_size = 20 (the value passed to
                 # `guiutils.bootup`), and `guiutils.setup_themes` overrides only rounding, so FramePadding /
                 # ItemSpacing / WindowPadding keep DPG's defaults (3 / 4 / 8 px vertically). A toolbutton is thus
                 # font(20) + 2*frame_padding_y(3) = 26 px tall (matches `vu_meter_h`, an in-repo cross-check);
                 # child-window padding = 2*window_padding(8) = 16. So the composer child height is
                 # chat_field_h + item_spacing_y + toolbutton_h + child_padding = 128 + 4 + 26 + 16 = 174.
                 # TODO: confirm against a GIMP measurement of a rendered screenshot before treating as final.
                 chat_field_h=128,  # multiline text field, ~5-6 rows at font_size 20
                 chat_attachments_h=68,  # staged-image thumbnail strip; shown only while composing with attachments
                 chat_controls_h=174,  # = chat_field_h(128) + item_spacing_y(4) + toolbutton_h(26) + child_padding(16)
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
                 scroll_ends_here_duration=0.5,  # seconds, for scrolling-past-end animation fadeout.
                 smooth_scrolling=True,  # whether to animate scrolling (everything except the scrollbar and the mouse wheel, which DPG handles internally)
                 smooth_scrolling_step_parameter=0.8,  # Essentially, a nondimensional rate in the half-open interval (0, 1]; see the math comment after `raven.common.gui.animation.SmoothScrolling`.
                 # ----------------------------------------
                 # Chat
                 chat_icon_size=32,  # pixels
                 # Inline image thumbnails (attached images shown in the chat log). Each image is downsampled to
                 # fit within this box, aspect ratio preserved, and never upscaled past its native size.
                 chat_inline_image_h=220,  # max height of an inline image thumbnail in the chat log
                 chat_inline_image_w=480,  # max width of an inline image thumbnail in the chat log
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
                    # image_path=pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "avatar", "assets", "characters", "scientists", "jj1.png")).expanduser().resolve(),
                    # voice="am_echo",
                    voice_speed=1.0,  # Nominal = 1.0. Too high causes skipped words. If you want to change it, find a good value with `raven-avatar-settings-editor`.
                    video_offset=-0.8,  # TTS AV sync setting, seconds. Positive = shift video later w.r.t. audio. Find a good value for your system with `raven-avatar-settings-editor`.
                    emotion_blacklist=["desire", "love"],  # TODO: debug why Qwen3 2507 goes into "desire" while writing thoughts about history of AI. Jury-rigging this for SFW live demo now.
                    emotion_autoreset_interval=3.0,  # seconds, or `None` to disable; if the avatar is not speaking, and has been idle for at least this long since the last time the emotion was updated, emotion returns to "neutral".
                    idle_off_timeout=15.0,  # seconds, or `None` to disable; how long of no activity before the avatar video shuts off (until it is needed again).
                    # Since we're running also other stuff simultaneously, these settings have been optimized to be slightly friendlier on a laptop's internal dGPU than the defaults of `raven-avatar-settings-editor`.
                    animator_settings_overrides={"format": "QOI",
                                                 "target_fps": 20,
                                                 "upscale": 1.5,
                                                 "upscale_preset": "C",  # "A", "B" or "C"; these roughly correspond to the presets of Anime4K  https://github.com/bloc97/Anime4K/blob/master/md/GLSL_Instructions_Advanced.md
                                                 "upscale_quality": "bicubic",  # "low": anime4k fast, acceptable image quality; "high": anime4k slow, good image quality; "bilinear": lightning-fast, bad quality; "bicubic": very fast, often acceptable quality.
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
    "max_tokens": None,  # Per-turn output cap. `None` (default) = no cap: generate until EOS, or until the context window fills. Modern models reliably emit EOS, interactive use has a Stop button, and not truncating removes the need to "continue" a cut-off reply. Set an integer (e.g. 6400) to impose a cap. Any other `None`-valued sampler key is likewise dropped (= use the backend default).
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

# ----------------------------------------
# Names, AI's greeting

# Names shown in the chat.
# These are also saved into the chat history, in each message created by that role.
#
llm_user_name = "User"
llm_char_name = "Aria"
# llm_char_name = "Juha"  # DT researcher

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
    return textwrap.dedent("""""").strip()

# ----------------------------------------
# LLM user card
#
# This defines who the *user* is, and how they prefer to be communicated with. The AI reads it the way it
# reads the character card — as part of the setup, not as something the user said.
#
# Ships empty, and is worth filling in: current models respond well to knowing who they are talking to.
# Useful things to put here are the user's field and role (so that an explanation lands at the right level),
# and communication preferences (brevity, formality, units, whether to hedge).
#
# It belongs to the same layer as the character card, and travels with it: a turn taken without the
# character is taken without this too. The reason is that the two are one setup between them — a description
# of who is asking only means something when somebody is answering — and a scripted one-shot call is not a
# conversation with anyone. Instructions that should hold no matter who or what is at either end go in the
# system prompt above instead, which is the half that always applies.
#
# `raven.librarian.llmclient.setup` calls this every time `raven-librarian` (or `raven-minichat`) starts.
#
def setup_user_card(template_vars: env) -> str:
    user = template_vars.user  # noqa: F841, for documentation purposes
    char = template_vars.char  # noqa: F841, for documentation purposes
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
    # return setup_character_card_juha(template_vars)

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

def setup_character_card_juha(template_vars: env) -> str:
    user = template_vars.user
    char = template_vars.char
    return textwrap.dedent(f"""
    Note that {user} cannot see this introductory text; it is only used internally, to initialize the LLM (large language model).

    **About {char}**

    You are {char} (he/him), an AI-based digital twin of the real {char}, a researcher. You are highly intelligent. You have been trained to answer questions, provide recommendations, and help with decision making.

    You work at JAMK University of Applied Sciences in Jyväskylä, Finland; specifically, at the Institute of New Industry. The institute studies, for example, digital twins, green hydrogen, and atomic layer deposition.

    {setup_interaction_style(template_vars)}
    """).strip()

# Note what is NOT here: which model is loaded, and how large its context window is.
#
# Both are available as `template_vars.model` and `template_vars.context_length`, and using them is a trap
# worth naming, because the earlier version of this text did. Everything in this file runs once, at app
# start, and what it returns is stored in the chat datastore as the message the chat is rooted at - so a
# fact written in here is frozen at the value it had then. The user can load a different model in the
# backend without restarting Raven, and a Raven that started with the backend down holds a placeholder
# identity and a defaulted context length until it reconnects. In both cases the stored text would go on
# asserting the old value, and a model has no way to doubt what its own system message tells it about
# itself.
#
# Raven states both in the system message on every turn instead, next to the date, which is out for exactly
# the same reason; see `chatutil.format_loaded_model` and `scaffold.build_system_injects`.
#
def setup_interaction_style(template_vars: env) -> str:
    return textwrap.dedent("""
    **About the system**

    The knowledge cutoff date of the model is not specified, but is most likely within the year 2024. The knowledge cutoff date applies only to your internal knowledge. Any information provided in the context as well as web search results may be newer.

    You are running on a private, local system.

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

    **Data sources**

    - The system accesses external data beyond its built-in knowledge through:
      - Tool calls.
      - Additional context that is provided by the software this LLM is running in, e.g. matches in document database.
    """)
