# Authors

*Last updated for v0.2.9.*

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

Claude Opus, 4.5 through 5:

Raven has been built as a human-AI collaboration since February 2026, so most of what is
listed here was co-created rather than written by either party alone; the human review that
goes with it is credited above. The model versions, in the order they worked on the tree:
4.5 and 4.6 (February to April 2026), 4.7 (April to May), 4.8 (June to July), 5 (from July).

- `raven.common.gui.xdotwidget`, `raven-xdot-viewer`.
- Unit tests. The suite grew from a handful of modules to 96, covering the library and
  utility layers, the shared GUI vocabulary, and the vendored file dialog.
- Refactoring, including the split of the Visualizer's `app.py` into `annotation`,
  `app_state`, `entry_renderer`, `info_panel`, `plotter`, `selection` and `word_cloud`.
- Two new apps: `raven-cherrypick` (image triage) and `raven-conference-timer`.
- `raven.papers`: arXiv search, identifier parsing, rate-limited metadata and PDF
  download, and the bibliography converters.
- Raven-librarian: the scripting surface (`agent`), the tool registry (`llmtools`), the
  attachment sidecar stores (`sidecarstore`, `imagestore`, `textfilestore`), datastore
  cleanup (`cleanup`, `cleanup_dialog`), exact token counting (`gguftokenizer`), and the
  `raven-indexer` CLI.
- Raven-server: the `webfetch` module.
- `raven.common`: the audio and speech layer (resampling, STT, TTS datatypes, lipsync,
  playback), document text extraction (`docextract`), the image layer, and the shared GUI
  vocabulary — `filedrop`, `filegrid`, `gridnav`, `keyboardmark`, `layout_math`,
  `qroverlay`, `tablecursor`, `thumbnailgrid`, `tileicons`, `tooltip`.
- Project documentation: the design briefs, the investigation write-ups, `dpg-notes.md`,
  and the style guide.

Qwen3-30B-A3B, 2507 and VL:

- `raven-arxiv-download`: arXiv metadata and PDF download.
- `raven-check-cuda`.
- `raven.common.readcsv`.
- First draft for `raven.librarian.hybridir`, including the adjacent chunk combiner, and the RRF (reciprocal rank fusion) result set combiner.
