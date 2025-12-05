# Changelog

**0.2.4** (December 2025, in progress):

**Added**:

- Tools:
  - New tool: *Raven-dehyphenate*.
    - This uses the `dehyphen` package to sanitize text bro-ken by hyp-he-na-tion.
    - This can be useful for `pdftotext` outputs, and for text obtained from PDF files by OCR (such as with `ocrmypdf --force-ocr input.pdf output.pdf`).
    - Raven-server's `sanitize` module is used automatically, if the server is reachable and the module is loaded on the server; else the dehyphenator model is loaded locally.

  - New tool: *Raven-arxiv-download*.
    - This takes arXiv paper IDs from the command line (e.g. 2511.22570, 2411.17075v5, cond-mat/0207270, math/0501001v2), and downloads the corresponding PDFs.
    - For instructions, see the [visualizer README](raven/visualizer/README.md).

- *Raven-visualizer*:
  - Importer: New keyword detection mode "llm".
    - This uses the LLM backend configured for *Librarian*.
      - When this mode is used, the LLM backend must be running when *Visualizer* (or the command-line tool `raven-importer`) is started.
    - To initialize the task, this uses the same system prompt and AI character as *Librarian* uses for its chat.
      - This gives results consistent with what *Librarian* would say, because the LLM operations are handled by the same AI simulacrum.
      - See `raven.librarian.config`.
    - The AI analyzes the titles and abstracts for each cluster (separately), and suggests keywords. These keywords are recorded as the cluster keywords for *Visualizer*.

- *Raven-librarian*:
  - New feature: STT (speech to text, speech recognition). Talk to the AI using your mic!
    - To start speaking to the AI, click the mic button next to the chat text entry field (hotkey Ctrl+Shift+Enter).
      - The mic starts glowing red, to indicate that Librarian is listening. The VU meter (audio input level) next to the mic button becomes active.
      - To stop speaking, and send the spoken message to the AI, click the mic button again, or wait until the recorder detects silence and stops automatically.
        - The gray line on the VU meter is the silence threshold level.
      - The mic stops glowing (and returns to its default white).
    - Librarian then runs the recorded audio through a locally hosted [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) speech recognizer, which lives in the `stt` module of Raven-server.
    - The transcribed text is then sent to the AI, just as if it was typed as text to the chat text entry field.
    - For now, this feature has certain limitations:
      - The recorder autostop settings are hardcoded: 1.5s of input audio signal level under -40.00 dBFS.
        - This should work under most circumstances, but if you are not in "most circumstances", you'll have to stop recording by clicking the mic button again.
      - It is not possible to edit the transcribed text before sending.
    - To choose your mic device, see `raven.client.config`.
      - By default, Librarian picks the first available NON-monitoring audio capture device, in the order listed by the command-line tool `raven-check-audio-devices`.
      - The default should work on laptops, and in general, most systems that have just one audio input device.
      - The help card (F1) shows which mic device is active. It is also printed to the client log at app startup.
  - Very rudimentary chat branch navigation added.
    - In the linearized chat view, each message has buttons for next/previous sibling, jump 10 siblings, jump to first/last sibling.
      - Hotkeys apply to the last message displayed in the view.
    - It's easier to show than explain (try it out yourself!); but when switching siblings in the linearized chat view:
      - The sibling node switched to becomes the candidate HEAD.
      - If the candidate HEAD has any child nodes (i.e. chat continuations):
        - The child node with the most recent payload (according to payload timestamp) is chosen.
        - That child node becomes the new candidate HEAD, and the process repeats.
      - Once no more child nodes are found (i.e. the candidate HEAD is a leaf node), the candidate HEAD becomes the final new HEAD.
      - The linearized chat view scrolls to the sibling node that was switched to, regardless of where the final HEAD is.
    - Contrast this with the branch button, which sets the chat HEAD to the given node, without scanning the subtree for continuations.
      - Just like in `git`, branching is cheap. Branching only sets the HEAD pointer; no data is copied.
      - If you branch, but then change your mind, click the "Show chat continuation" button on the last message (hotkey Ctrl+Down).
        This rescans the chat continuation just like when switching siblings.
  - The LLM system prompt, the AI's character card, persona names (AI and user), and the AI's greeting message can now be customized in `raven.librarian.config`.
    - Changes take effect when Librarian is restarted.
    - Limitation: for now, only one AI character icon is loaded. If you switch characters, old chats will show the current character's icon (the persona name is stored in the chat database, but the avatar and icon paths are not).

**Changed**:

- *Raven-visualizer*:
  - Configurable plotter colors (background, grid, colormap). Loaded from `raven.visualizer.config` at app startup.
  - Configurable word cloud colors (background, colormap). Loaded from `raven.visualizer.config` at app startup.
  - The section headings in the BibTeX import dialog are now clickable, and perform the same function as the icon buttons.
  - *Visualizer*'s importer now automatically uses *Raven-server* for embeddings and NLP if it is running.
    - If the server is not running, the AI models are loaded locally (in the client process) as before.
    - There is no visible difference from the user's perspective (other than saving some VRAM, if also *Librarian* is running at the same time).
  - The importer now sanitizes abstracts using the `sanitize` module of Raven-server. This feature is on by default.
    - This affects only new BibTeX imports. Existing datasets are not modified.
    - The feature can be turned off in `raven.visualizer.config`. See the `dehyphenate` setting.
    - For each abstract, all paragraphs are sent together for processing. This may cause paragraphs to run together, if an abstract contains multiple paragraphs,
      but is often the only way if the input text is REALLY broken and contains newlines at arbitrary places. It was felt this is preferable, because scientific
      abstracts are often just one 200-word paragraph.
    - Raven-server's `sanitize` module is used automatically, if the server is reachable and the module is loaded on the server; else the dehyphenator model is loaded locally.

- *Raven-librarian*:
  - The document database now uses *Raven-server* for embeddings and NLP.
    - This saves some VRAM, by avoiding loading another copy of the same models in the client process.
    - This also makes the `raven.librarian.hybridir` information retrieval backend fully client-server, allowing the AI components for this too to run on another machine.
    - Because *Librarian* requires *Raven-server* for other purposes, too, *Librarian* will not start if the server is not running.
  - The document database now ingests `.bib` files, too.
    - This allows using the `raven-burstbib` command-line tool to mass-feed abstracts into Librarian's document database.
      - The tool takes a `.bib` file and splits it into individual files, one per entry. Hence each entry becomes a separate document in Librarian's document database.
  - The app now recovers if `state.json` is missing or corrupt.
  - Many small UI improvements, for example:
    - Window resizing implemented.
    - Collapsible thinking traces.
    - Interrupt/continue.
    - Avatar idle off.
      - Configurable, optional. See `avatar_config.idle_off_timeout` in `raven.librarian.config`. Seconds as float, or `None` to disable.
      - This saves some GPU compute by switching off the avatar video after the AI avatar is idle for a while.
      - The avatar video switches back on when:
        - The AI starts processing (writing new message, continuing existing message, rerolling existing message).
        - The chat view is re-rendered (e.g. by switching chat branches, or resizing the window).
        - The AI starts speaking (Ctrl+S, send last message to TTS).
    - Help card added.
    - TTS audio playback device setup in `raven.client.config`:
      - For configuration symmetry reasons, `None` now means "use the first available playback device as listed by `raven-check-audio-devices`", not the system's default playback device.
      - The new special value "system-default" uses the system's default playback device, so that the playback goes to the same device as from other apps. (This is what `None` did before.)
      - The default configuration has been changed to use "system-default", so that behavior should remain the same.

- Tools:
  - *Raven-pdf2bib*: Overhauled. See updated instructions in [visualizer README](raven/visualizer/README.md).
    - To initialize each LLM task, this uses the same system prompt and AI character as *Librarian* uses for its chat.
      - This gives results consistent with what *Librarian* would say, because the LLM operations are handled by the same AI simulacrum.
      - See `raven.librarian.config`.


**Fixed**:

- *Raven-visualizer*:
  - Fix bug: "reset zoom" missed some datapoints (in a "select visible", hotkey F9), if they were exactly at the edges of the data bounding box.
    - Note that also loading a dataset resets the zoom, so the bug also affected the initial view upon loading a dataset.
    - Workaround for previous versions: after a "reset zoom", zoom out by one mouse wheel click before using "select visible".
  - Fix bug: wrong dtype in the embedder loader's CPU fallback.
    - The CPU fallback loader now always uses float32.
    - Workaround for previous versions: when working without a GPU, configure the embedder explicitly to use dtype `torch.float32`. See `raven.visualizer.config` and `raven.server.config`.
  - Fix UI bug: the plotter axes no longer light up when the mouse hovers on them.
    - The axes are not clickable, so the highlight was spurious.
    - This was broken when we upgraded to DearPyGUI 2.0, where the plotter changed to introduce that hover-highlight by default. Now we disable the highlight by theming the plotter.

- *Raven-avatar*:
  - Fix bug: Also the background image is now hidden while the avatar is paused.

- *Raven-librarian*:
  - Fix bug: The avatar's subtitle now re-positions itself correctly when the GUI is resized while the avatar is speaking.


---

**0.2.3** (7 October 2025):

**Added**:

- Prototype of *Raven-librarian*, a scientific LLM frontend GUI app.
  - Features an animated AI avatar with TTS and auto-translated subtitles, document database (plain text files for now), and tool-calling support (websearch for now).
    - Document database uses hybrid search (BM25 for keyword search, ChromaDB for semantic search).
  - In this prototype, chats are saved, but going back to previous chats is not yet possible because the GUI for that has not yet been developed.
  - When the *Documents* checkbox in the GUI is ON, the document database is autosearched, using the user's latest message to the AI as the query.
    - If, additionally, the *Speculation* checkbox is OFF, the LLM is bypassed when there is no match in the document database.
  - Websearch is enabled when the *Tools* checkbox in the GUI is ON.
  - Requires both *Raven-server* and the LLM backend (oobabooga/text-generation-webui, with `--api`) to be running.
  - For configuring Raven-librarian, for now, see `raven.librarian.config`.
    - The default location for the document database is `~/.config/raven/llmclient/documents`.
      - Librarian monitors this directory automatically, and also scans for offline changes at app startup.
      - Put `.txt` files there; they are search-indexed automatically. Replace files; the index is updated automatically. Remove files; they are removed from the index automatically.
      - If you need to force a manual index rebuild: make sure Librarian is not running, then delete `~/.config/raven/llmclient/rag_index`. It will be rebuilt at app startup.

- *Raven-avatar* now has a "data eyes" effect, for use as an LLM tool access indicator in Librarian.
  - Cel animation, up to 4 frames. Can be tested in `raven-avatar-settings-editor`.

- Speech video recording in `raven-avatar-settings-editor`.
  - Output goes in the `rec/` subdirectory.
  - TTS speech is saved as `.mp3` files, one per sentence.
  - Avatar video (avatar only, no background) is saved as as individual frames as `.qoi`.
    - For converting the video frames into a usable format, see the `raven-qoi2png` tool.
  - A speech timings list is saved as `.txt`.
  - These can be used to piece together a speech video in a video editor such as *OpenShot*.


**Changed**:

- *Raven-visualizer*'s importer now uses both the title and the abstract to cluster the inputs.
  - This requires *Snowflake-Arctic* or better as the embedding model; the older *mpnet* model tends to lead everything to become one cluster if the abstracts are used for clustering.
  - Old datasets must be imported again for the changes to take effect, because the embeddings are computed at import time.
    - Old embeddings caches must be deleted before re-importing!
    - For example, when `mydata.bib` is imported, Raven's importer produces:
      - `mydata_embeddings_cache.npz`: The vector embeddings. Delete this cache file!
      - `mydata_nlp_cache.pickle`: The natural language processing results, used for keyword detection. This cache is not affected by this change, so no need to delete.

- *Raven-server* now hosts all AI components, including embeddings and spaCy NLP.
  - spaCy NLP is only available for Python-based clients running the same versions of Python and spaCy, because it communicates in spaCy's internal format.


---

**0.2.2** (13 August 2025):

**Added**:

- First complete tech demo of *Raven-avatar*.
  - See the GUI apps `raven-avatar-settings-editor` (completely new postprocessor settings GUI) and `raven-avatar-pose-editor` (ported from the old THA3 pose editor).
  - The settings editor requires *Raven-server* to be running.

- *Raven-avatar* now has cel-blending and animefx support.


---

**0.2.1** (18 June 2025):

Otherwise the same as 0.2.0 (17 June 2025), but with the TODO cleaned up. Documenting both here.

**Added**:

- *Raven-server*: to provide an animated AI avatar, and to eventually host all AI components.
  - This is a web API server, initially ported and stripped from the discontinued *SillyTavern-extras*.
    - AGPL license! Affects the `raven.server` and `raven.avatar.pose_editor` directories.
    - All other components of *Raven* remain BSD-licensed. This includes `raven.avatar.settings_editor`.
  - Pose editor ported from wxPython to DPG, to match the rest of the *Raven* constellation, and to require only one GUI toolkit.
  - Added in d42d52356d61d290d4d9a1e5ffd0e1b6e0843c61, 22 May 2025. The entrypoint has since moved to `raven.server.app`.
  - Avatar has new features compared to the old Talkinghead:
    - Lipsync, to new TTS module based on Kokoro-82M.
    - Anime4k upscaling (super-resolution).


---


**0.1.x** and older

The project was started in December 2024.

No changelog was maintained.

These versions included only *Raven-visualizer*.
