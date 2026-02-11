# Authors

*Last updated for v0.2.5.*

*Both human as well as AI authors are listed here.*

Juha Jeronen (@Technologicat):

- Main author.
- Librarian, Visualizer, and integration work.
- Raven-server:
  - Significant re-engineering and expansion.
  - New server modules `imagefx`, `natlang`, `sanitize`, `stt`, `translate`, and `tts`.
  - Significant expansion of server module `avatar`.
- Raven-avatar (both server and client):
  - Significant re-engineering and expansion.
  - The new avatar animation driver (`raven.server.modules.avatar`); a prototype was already included in the last version of SillyTavern-Extras, but has been significantly extended in Raven.
  - Video postprocessor (100% own code, re-licensed under BSD).
  - Postprocessor settings GUI app (`raven-avatar-settings-editor`).
  - Integration of Anime4K upscaler.
  - Porting to DearPyGUI, including the pose editor GUI app (`raven-avatar-pose-editor`).
  - Python client.
  - Lipsynced TTS.
  - Subtitler.
- Customization of vendored libraries (DearPyGUI-Markdown, file_dialog, Anime4K-PyTorch).
- Integration of a docstring into Anime4K-PyTorch and cleaning up the module slightly.
- Human review of AI-created changesets.

@cohee and SillyTavern-Extras contributors:

- The SillyTavern-Extras codebase, which was discontinued, and then formed the basis for the first version of Raven-server. Used under the AGPL license.
- The original version of the server modules `avatar`, `classify`, `embeddings`, and `websearch`.

Pramook Khungurn (@pkhungurn):

- Talking Head Anime 3 (THA3) AI posing engine (software and AI models).
- The original version of the THA3 pose editor GUI app (now `raven-avatar-pose-editor`).
- The original version of the THA3 demo app (with facial motion capture), which then evolved into `talkinghead` of SillyTavern-Extras, and then into Raven-avatar.

@IvanNazaruk:

- DearPyGUI-markdown.

@totallynotdrait:

- `file_dialog` for DearPyGUI.

Kokoro-FastAPI contributors:

- `raven/vendor/kokoro_fastapi/streaming_audio_writer.py`. Used under the Apache License, version 2.0.

Anime4K contributors:

- Anime4K-PyTorch. Used under the MIT License.

Claude Opus, 4.5 & 4.6:

- `raven.common.gui.xdotwidget`, `raven-xdot-viewer`.
- Unit tests.
- Refactoring.

Qwen3-30B-A3B, 2507 and VL:

- `raven-arxiv-download`: arXiv metadata and PDF download.
- `raven-check-cuda`.
- `raven.common.readcsv`.
- First draft for `raven.librarian.hybridir`, including the adjacent chunk combiner, and the RRF (reciprocal rank fusion) result set combiner.
