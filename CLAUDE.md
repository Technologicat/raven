# Raven - CLAUDE.md

## Project Overview
Local RAG-based research assistant constellation. Privacy-first, 100% local.

**Components:**
- **Visualizer** (`raven/visualizer/`): BibTeX topic analysis, semantic clustering, keyword extraction. The original app.
- **Librarian** (`raven/librarian/`): LLM chat frontend with tree-structured branching history, hybrid RAG, tool-calling, avatar integration.
- **Server** (`raven/server/`): Web API for GPU-bound ML models. All inference happens here.
- **Client** (`raven/client/`): Python bindings for Server API.
- **Avatar** (`raven/avatar/`): AI-animated anime character (THA3 engine, lipsync, cel animations, video postprocessor).
- **Common** (`raven/common/`): Shared utilities (video processing, audio, GUI widgets, networking).
- **Tools** (`raven/tools/`): CLI utilities (format converters, system checks).

## Architecture

### Server/Client Split
All ML inference in `raven/server/modules/`:
- `tts.py` - Kokoro TTS with phoneme timestamps (needed for lipsync)
- `stt.py` - Whisper speech recognition
- `embeddings.py` - Sentence embeddings (snowflake-arctic)
- `translate.py` - Neural machine translation
- `summarize.py` - Text summarization
- `classify.py` - Sentiment/emotion classification, to control avatar's facial expression
- `sanitize.py` - Text cleanup (dehyphenation etc.)
- `natlang.py` - spaCy NLP analysis
- `websearch.py` - Web search tool for LLM
- `avatar.py`, `avatarutil.py`, `imagefx.py` - Avatar rendering pipeline

Client apps call Server via `raven/client/api.py`. Server can run on different machine.

### Librarian Core Modules
- `chattree.py` - Tree-structured message storage (nodes with parent pointers)
- `chat_controller.py` - Linearized chat view GUI
- `llmclient.py` - LLM backend communication (text-generation-webui, OpenAI-compatible API)
- `hybridir.py` - Hybrid retrieval (BM25 + ChromaDB, reciprocal rank fusion)
- `scaffold.py` - Prompt construction, tool orchestration
- `appstate.py` - Application state management (persistent across sessions)
- `chatutil.py` - Chat utilities
- `app.py` - Main GUI application
- `minichat.py` - CLI mini-prototype, deprecated

### Common Subsystems
- `raven/common/video/` - Postprocessor, upscaler (PyTorch Anime4K), colorspace conversions, cel compositor
- `raven/common/audio/` - Player, recorder, codec (PyAV streaming)
- `raven/common/gui/` - Custom DearPyGui widgets (VU meter, GUI animation framework, messagebox)

### Vendored Dependencies
- `tha3/` - Talking Head Anime 3 neural network (avatar animation)
- `DearPyGui_Markdown/` - MD renderer (robustified for background threads, has one remaining URL highlight bug)
- `file_dialog/` - File dialog, extended (sortable, animated OK button, click twice when overwriting)
- `anime4k/` - PyTorch port of Anime4K upscaler (extracts kernels from GLSL), slightly cleaned up
- `kokoro_fastapi/` - Streaming audio writer for TTS over network
- `IconsFontAwesome6.py` - Icon font (note: outdated version)

## Code Style
- Impure functional, Lispy (closures, `unpythonic` patterns)
- `unpythonic` used for: `env` (namespace), `Timer` (benchmarking), occasionally `@call`
- OOP where appropriate (GUI components, stateful objects)
- Config via Python modules (`config.py` files, not YAML/JSON)
- Type hints encouraged but not enforced everywhere
- Contract-style preconditions/postconditions would be useful, but mostly not implemented yet

## Key Patterns

### Message Tree (Librarian)
Messages stored as nodes with `parent` pointers. Branch = HEAD pointer to leaf.
Linear history reconstructed by walking ancestor chain. Branching is cheap.

### Hybrid RAG
- Semantic: ChromaDB embeddings
- Keyword: bm25s (BM25 algorithm)
- Combined via reciprocal rank fusion

### Avatar Lipsync
TTS (Kokoro) provides timestamped phonemes → mapped to mouth morphs → THA3 animator.
This coupling limits TTS engine choices (most don't expose timestamped phoneme data).

## Current State

### Well-structured (target style)
- `raven/librarian/` - Clean module separation

### Needs refactoring
- `raven/visualizer/app.py` - ~4000 lines, monolithic, needs splitting

### Test coverage
- Minimal: only `raven/client/tests/test_api.py` and `raven/librarian/tests/test_hybridir.py`
- **Priority: expand test coverage**

## LLM Backend
Uses text-generation-webui with OpenAI-compatible API.
Recommended model: Qwen3-VL-30B-A3B (24GB+ VRAM) or Qwen3-VL-4B (8GB VRAM).

## Known Issues / TODOs
- Visualizer refactoring needed
- Test coverage very low
- DearPyGui_Markdown URL highlight bug (threading-related, untracked)
- FontAwesome version outdated
- Hindsight integration pending (PDM dependency conflicts)
- TTS engine expansion limited by phoneme timestamp requirement
