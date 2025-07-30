"""Configuration for Raven-librarian (the LLM-client component).

Currently used by the `librarian.llmclient` and `tools.pdf2bib` modules.
"""

import torch

from .. import config as global_config

llmclient_userdata_dir = global_config.toplevel_userdata_dir / "llmclient"

# URL used to connect to the LLM API.
#
# This has been tested with local LLMs only, but theoretically cloud LLMs should work, too.
# To set your API key, see the setting `llm_save_dir` above, and create a file "api_key.txt" in that directory.
# Its contents will be automatically set as the Authorization field of the HTTP headers when `llmclient` starts.
#
llm_backend_url = "http://127.0.0.1:5000"
llm_api_key_file = llmclient_userdata_dir / "api_key.txt"  # will be used it it exists, ignored if not.

llm_line_wrap_width = 160  # `llmclient`; for text wrapping in live update

# RAG / IR

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

# Magic directory: put your RAG documents here (plain text for now).
# Add/modify/delete a file in this directory to trigger a RAG index auto-update in the LLM client.
llm_docs_dir = llmclient_userdata_dir / "documents"

# Whether to scan also subdirectories of `llm_docs_dir` (TODO: doesn't yet work properly, need to mod doc IDs)
llm_docs_dir_recursive = False

# Where to store the search indices for the RAG database (machine-readable).
llm_database_dir = llmclient_userdata_dir / "rag_index"

# Where to store the search indices for the `HybridIR` API usage example / demo
hybridir_demo_save_dir = global_config.toplevel_userdata_dir / "hybridir_demo"

# Tool-calling

# Tool-calling requires instructions for the model, as part of its system prompt.
# Typically the instructions state that tools are available, and include a dynamically
# generated list of available functions and their call signatures.
#
# Newer models, e.g. QwQ-32B, include a templates for these instructions in their built-in
# prompt template. In this case, the LLM backend builds the instructions automatically,
# based on data sent by the LLM client (see `tools` in `llmclient.setup`).
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
