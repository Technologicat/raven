# Raven-visualizer TODO


## v0.2.x (August 2025?)

*Preliminary plan, not final. Details may change.*

- Improve codebase maintainability:
  - Vendor our fixed wosfile library?
  - `vis_data` should really be called `entries` everywhere in this app constellation, also in importers. Something to BiBTeX importers in `raven.tools`, and the BibTeX to Raven importer in `raven.visualizer.importer`.

- Fdialog use site boilerplate reduction? We have lots of these dialogs in Raven.

- Word boundary mark (\b) for search. What's a good UX here? Use what character as a word boundary?

- App icon. `small_icon` and `large_icon` parameters of `create_viewport` (png or ico format).
   https://github.com/hoffstadt/DearPyGui/discussions/1688
   https://dearpygui.readthedocs.io/en/1.x/reference/dearpygui.html#dearpygui.dearpygui.create_viewport

- Flash the search field (both in main window and in file dialogs) when focused by hotkey. Need to generalize `ButtonFlash` for GUI elements other than buttons.

- fdialog: least surprise: if the user has picked a unique file extension in the file extension filter combo, use that as the default file extension when in save mode. If the current choice in that combo has several file extensions, and/or has one or more wildcards, then use the API-provided default file extension.

- Import BibTeX: use multiple columns for input file table in the GUI if we have very many input files.

- Make clustering hyperparameters configurable, preferably in the GUI. Put defaults into `raven.visualizer.config`.

- pdf2bib
  - prompt the author extraction step to return the exact words "No authors provided" (or similar) if the LLM detects this case.
  - prompt the title extraction step to return the exact words "No title provided" (or similar) if the LLM detects this case (or if it thinks the title is "Abstract", literally).
  - investigate how to avoid Qwen3 2507's failure mode of thinking that it's in a simulated environment (e.g. from system date being later than knowledge cutoff)
  - overthinking mitigation: once the output token limit is reached and the task aborts, executive function simulation via LLM?
    - should detect token-limit-exceeded in `raven.librarian.llmclient`, and return a status flag in the metadata

- Server
  - Finish the new `stt` module (speech to text)
    - Figure out what's going wrong with the end of the long example sent by `raven.client.tests.test_api` (spurious text generated after the speech in the long audio ends)
    - Test also `stt_transcribe_file` and `stt_transcribe_array`
    - whisper-large-v3-turbo needs another 1.6 GB of VRAM; check if we could use a quantized model (may need vLLM)
  - Option: check first for local model before checking on HF
    - Currently some modules do this (some due to using a non-default download location, in turn due to requirements of vendored code), others don't.
    - Can be important in case the model vanishes from HF (as suddenly happened with the old/ancient summarizer).
    - This would allow the modules of an existing Raven-server installation to start (with the locally existing model) even when that model is no longer available on HF.
  - Zip the avatar characters, for ease of use
    - Include all extra cels in the zip, as well as optional animator/postprocessor settings, and optional emotion templates
    - Implement zip loading on server side, add a new web API endpoint
  - `summarize`: add LLM summarization mode, with a configurable prompt. Allow using a separate small LLM for speed.

- Avatar
  - Update assets for all characters (add at least eye-waver effect, maybe other cel-blending cels too)
  - Move data eyes management to server side
  - Optional: hologram glitch effect when switching chat branches
  - Add help cards:
    - Avatar settings editor
    - Avatar pose editor

- Visualizer
  - Excel or CSV import to BibTeX
  - BibTeX export (must keep a copy of original full entries in the dataset file)
  - Show item slug (BibTeX identifier)
  - Add feature: author search
    - Show full author list if it fits
    - How to show author list if "Aaltonen et al." and the user is searching for "Virtanen" (200 names later in the same author list)
  - Add feature: text content search ("abstract" field in BibTeX)
  - Color data points by input BibTeX filename, to easily see new data
    - Store import-source BibTeX file metadata in the dataset
  - Importer: allow specifying a dataset to load the dimension reduction from
    - This is the easiest way to add the feature to add new data on top of an existing dataset
  - Keep the app state in top-level containers, and pass these in/out explicitly. More FP and facilitates adding unit tests later.
  - See if we can still refactor something to make `raven.visualizer.app` shorter (still too much of a "god object").
    - Refactoring the info panel (~2k SLOC, half of the app) would help a lot.
    - Info tooltip another good candidate, and needs many of the same data sources to be passed in. (Do these two need to work together?)
  - Make the layout switchable left/right (which side of the screen the info panel is on, for on-site collaboration accounting for physical placement constraints for laptop and users)
  - Improve keyword autodetection
    - Preprocess text by LLM before handing over to the simple keyword detector algorithm?
    - Invert embedding?
  - Account for BibTeX entry type: article, inproceedings, book, patent, ...
    - Show entry type for each entry
    - Show count by type in selection
    - Allow filtering by entry type
  - BUG: Search result highlight: "Can a" -> highlights whole word "Can", then highlights "a" inside it, breaking the outer highlight
  - Importer: Configurable hyperparameters
  - File extensions: dataset (.pickle) should have file extension different from nlp cache (.pickle)

- Librarian
  - Documentation:
    - Write Raven-Librarian user manual
    - Privacy: audio recording for STT (speech to text)
      - only recorded when the user clicks the mic button
      - only used for locally hosted STT, and then discarded
      - never saved to disk
      - never sent anywhere (except to your configured Raven-server for processing)
      - transcript shown *for the user's information* in the client log
      - see `raven.server.modules.stt`, and `stt_*` functions in `raven.client.api` (and their use sites)
    - Mention empirical observation: start LLM first (before Raven-server) to make it run faster. Possibly due to GPU memory management. Or start avatar first, to make stuttering less likely on a single GPU?

  - Maybe next:
    - Refactor chat message payload creation to a function that auto-populates the payload's `general_metadata`.

      Somthing like::

            timestamp, unused_weekday, isodate, isotime = chatutil.make_timestamp()
            greeting_node_id = datastore.create_node(payload={"message": currently_configured_greeting,
                                                              "general_metadata": {"timestamp": timestamp,
                                                                                   "datetime": f"{isodate} {isotime}",
                                                                                   "persona": llm_settings.personas.get("assistant", None)}},
                                                     parent_id=system_prompt_node_id)


    - STT (speech to text, speech recognition):
      - Configurable silence level, autostop timeout, VU peak hold time
      - `raven-transcribe`: command-line tool
        - from audio file or from mic
        - -p "prompt prompt prompt"
        - write transcribed text to stdout by default, to file by  -o filename.txt
      - do we need a GUI indicator for the autostop timeout? (mini progress bar or something)
      - Extract proper names from chat log (use spaCy NER), fill a comma-separated list of those into the STT prompt
      - Add voice command interface, e.g. "Raven command subtitles off"
        - Split transcribed text to words, check the first two words; on match, scrub those two words, and trigger the command processor for the rest.
        - Chaining commands: split to sentences?
      - Edit spoken message before sending?
    - Switch to next/previous branch, when switching a message that is not currently last in the linearized view:
      - Pick the most recent continuation and show it, if any continuations exist
        - To determine which is the most recent: for each possible continuation, scan subtree for the most recent revision timestamp anywhere in descendants
      - This gives us a *very* rudimentary way to navigate the chat database.
    - For model testing: User persona sampling / "impersonate"
    - For model testing: Send message as AI / "prefill"
    - Long subtitle splitter (we now have the audio length).
    - Add feature: Avatar on/off (for low VRAM)
      - What to put in the right panel when avatar is off? Recent chats list, once we implement that?
    - Add feature: Smooth scrolling for linearized chat view
      - Infra already exists (`raven.common.gui.animation.SmoothScrolling`), just connect it. See the info panel in `raven.visualizer.app` for an example.
    - Don't crash if e.g. `tts` module isn't running

  - Later:
    - Document database: scopes
      - Scope = a subdirectory of `llm_docs_dir` in `raven.librarian.config`
        - E.g. AI, hydrogen, MLP fanfics, ...
      - For filtering, add a new custom metadata field: "scope". Works with both ChromaDB as well as our wrapper logic over BM25s.
      - Add feature: scope selection for docs search.
        - Checkbox for each existing scope, autogenerated?
        - Button for select/unselect all
      - Add GUI element to choose number of matches; persist it in `app_state`.
      - Scoping is needed for long-term memory, too (by chat message tags), to avoid undesired cross-contamination, e.g. between work and hobby chats, when retrieving memories.
        - How to avoid moving files around when/if the user changes the tags on an existing message?
          - Solution: make tags the primary mechanism, and make the document ingestion mechanism automatically add a tag that matches the subdirectory name.

    - Import tool for importing a batch of documents into the document database (useful when importing lots of documents at once)
      - Just instantiate a `hybridir.setup` in the same datastore that Librarian uses, and wait for the scanner to finish updating. Once the rescan finishes, exit the tool.

    - Context fill meter

    - Add recent chats list view
      - This is useful to complement the functionality of the graph view (ideally, we need both); also easier to build out of existing GUI components.
      - "X chats (Y unique nodes) in database"
        - In a tree-native storage, what is a "sufficiently unique" chat to warrant inclusion in the list?
          - Obviously, not every possible HEAD (that would be almost every node in the storage, except the system prompt)
          - A chat continues for a while and then ends -> more depth better?
          - But in general, not all leaf nodes are sufficiently unique chats
            - If there is sibling collection of rerolls at leaf level, then just one should be shown, and "+N", where N is the number of other branches (excluding the one shown)
            - Make it optional to collapse the redundant siblings, it is sometimes useful to see them all
          - Siblings at non-maximal depth aren't sufficiently unique chats either, if at least one branch continues further down
            - Then the branch that continues down is part of a sufficiently unique chat?
            - Others are beginnings of alternative timelines that weren't picked (at least yet)
          - General challenge: show potentially interesting HEADs the user would want to navigate to
          - What else could be useful, ideally? Search, plus a topic index?
      - Click a chat to switch to it
      - Double-click to switch and close the chat list (to show the AI avatar again)
      - Timeline
        - Section separators and headers, e.g. by date
        - Get item timestamps from the most recent revision in the subtree
     - Show a "chat card" for each chat
        - Show something distinctive for each unique chat
          - E.g.:
            - User's initial message
            - Last branch point (e.g. where rerolling last happened)
            - The most recent message in the whole subtree
            - All tags in the linearized history (or in whole subtree?)
          - For each message shown, show the message number in the linearized history ("chat depth")
          - Summarize each shown message if too long?
            - Generate and save LLM summary and keywords on request (of chat so far)
              - This could live in the metadata for a chat node (but revision-specific)
      - Filter by stored persona names (e.g. @Aria @User)
      - Tags (e.g. #work #hydrogen #AI):
        - "Show items matching", "Exclude items matching" (allow using both simultaneously, with two text inputs)
          - Complement sets / de Morgan's laws: include should use AND, exclude should use OR?
        - Mass edit tags for all matching items
        - Tag autocomplete:
          - Under the edit field, show ~20? most common tags in order of number of occurrences, descending
          - Shade them, more common across database is brighter; show number of occurrences, too
          - Tab: cycle through currently possible completions (and select the autocompleted part if possible), Enter: accept, Esc: reject completion (and return input text to state prior to when Tab was pressed)
      - Recent chat search
        - Search by time interval
          - Chat-node-created vs. most-recent-revision-modified time?
            - Maybe either one is OK to declare a match; show both times, highlight why the item matches
        - HybridIR search engine search (since we'll search-index the chats anyway, for the memory feature)
          - Show matching snippet in the chat list (beside the distinctive message(s) from the chat), and its number in the linearized history (highlight; and insert to correct position among the distinctive messages)
          - Indicate matches also in collapsed sibling nodes
          - Keep a list of which nodes are hidden under each chat shown
          - Color the match differently (grayed-out shade? to indicate it's in a branch (or multiple branches?) not shown in the list
        - Incremental fragment search?
          - Maybe not needed for fulltext since we have the search engine
          - Search in message text in linearized history + in tags (#work) + in stored persona names (@Aria)
            - Search each chat node once, cache results (the same node may turn up in several linearized histories)
        - Persist state of search filters in `app_state`
        - Name and save search filters? Or better to keep things lightweight and as simple as reasonably possible?

    - Add nonlinear chat view / chat graph editor (this is part of where the true power of Librarian will come from)
      - View subtree
        - Include a virtual root node "view X more nodes" (unless already at an actual root node of the forest)
          - Warn (red text on node?) if about to expand to the whole chat storage (slow, too much data, not very useful, often not what the user wants)
      - Perhaps the easiest to do by porting XDottir to DPG, then writing the graph as XDot, and reusing its display mechanism
        - This would also give us a modernized Graphviz viewer as a bonus
      - zoom hack: https://github.com/iwatake2222/dear_ros_node_viewer/blob/main/src/dear_ros_node_viewer/graph_vewmodel.py#L206
      - how to get mouse position: https://github.com/hoffstadt/DearPyGui/issues/2164
      - simple examples:
        - https://github.com/DataExplorerUser/drag_drop_node_editor/blob/main/drag_and_drop_node_editor_dear_py_gui.py
        - https://github.com/hoshianaaa/DearPyGUI_NodeEditor_Template/tree/main
      - Maybe better to just use a the plotter, with custom tooltips? We don't need a node *editor* here, but rather just something to visualize a graph.
      - Need a "jump to chat node by chat node ID" feature (chatlog export reports the IDs)

    - Add feature: switch HEAD node by chat node ID (chatlog export reports the IDs)
    - IBM Granite OCR for PDF input
    - `DPGAvatarRenderer`, `DPGAvatarController`: isolate the DPG-specific parts
    - Support for non-thinking models
      - Librarian currently assumes in a few places (e.g. avatar speaking animation control) that the model will first emit a "<think>" tag.
      - `raven.librarian.llmclient.invoke` should inject the initial "<think>" tag if the model doesn't send it. Some models don't (e.g. QwQ-32B, which was a preview of Qwen3).
        - Have a "thinking model" toggle that, when enabled, does the initial "<think>" tag check at the start of the message (and only if NOT continuing a previous message).
      - `raven.librarian.chatutil.scrub` already fixes a missing initial think tag (if there is a closing tag but not an opening one), but that's only for the final message.
    - MCP support for loading tools from remote servers?
      - And/or skills support? Needs a sandboxed coding venv.
        https://simonwillison.net/2025/Oct/16/claude-skills/
    - Add feature: show prompt
      - Save it per-chat-message, from `on_prompt_ready` (in `raven.librarian.scaffold`).
      - Show prompt length as tokens (`raven.llmclient.token_count`).
    - Add feature: show RAG results (first step toward an attribution mechanism)
    - Add feature: file attachments
      - For detailed, full-content analysis of one or a small handful of documents
      - E.g. read a PDF, inject full (cleaned) content into LLM context
        - Should this be a tool, too?
      - It's still possible to RAG ingest the document, by adding it to the document database. File attachments are an orthogonal feature, for cases where you need to ensure the LLM sees the full document.
    - Draw assets:
        - Make per-character AI chat icons for all characters (now Librarian supports them; e.g. character `aria1.png` has `aria1_icon.png`, RGBA, 64x64)
    - Have three RAG stores:
      - documents: explicit, for user (exists)
      - long-term memory: implicit, managed by system; for recalling old chats (new)
      - explicit memory bank: explicit, for AI (new)
    - Add feature: long-term memory
      - A second RAG store that indexes chat messages
        - Should be able to use the chat node ID (in the forest) as the document ID (it's a UUID).
      - Provide explicit access via tool-calling
        - Search with a given query (get matching messages, with node IDs)
        - Retrieve a local neighborhood of a given node ID (e.g. subtree max 3 levels up/down)
      - Provide a feature to query the LLM for chats on a given topic
        - It could use the tools, and then format an output with links
        - Clicking a link could open that chat
        - Using the LLM here gives much more search power (accepting also vague queries) than a simple fragment search or even the raw hybridir search engine.
      - Automatic associative memory: autosearch with user's most recent message or two?
        - In autosearch, return also the AI's replies or only the user's messages? Similar commercial products return user's messages only, probably to keep the AI grounded.
      - Each chat message = a RAG document; store chat node ID, too.
        - Metadata side channel, or just use as a heading in the content? Content would be useful for the LLM to see the IDs, too, for tool-calling.
      - To avoid reindexing at every new message, commit changes when switching to another chat branch (or when creating a new chat), or when app is shutting down.
        - Ignore the chat nodes on the current branch, when searching the RAG store. (Exclude by document ID.)
      - Memory reinforcement
    - Add feature: explicit long-term memory bank
      - A third RAG store, for use by the AI
      - Provide tools to store/list/search/retrieve memories (title and content)
      - A customizable system message section for the AI to store things it wants to remember in every chat?
    - Add feature: tool-call access to RAG
      - Get full document, based on its ID (the current RAG autosearch already shows the document IDs).
      - Search database with given query, optionally disabling or enabling only given scopes.
        - Get document IDs, which correspond to relative path (scope + filename).
        - Titles would be nice, but we don't currently have a title field - the document content is completely arbitrary.
      - Get topics.
        - Auto-include, in the system message, a high-level summary of topics currently available in the document database?
        - Simplest possible approach: scope names.
        - One possible more sophisticated approach:
          - Use keyword detection to identify what the documents are about.
          - Preprocess keywords when tokenizing new/updated documents.
            - Avoid running the full document through the NLP pipeline twice (need to mod `hybridir.HybridIR._tokenize`).
            - Where to get the corpus data to compare against? Store the raw word frequency data for each document in the fulldocs database to avoid recomputing them each time a new document is added?
              - We still need to aggregate across the whole database at each commit, but that's probably acceptable (AI parts as well as the reindexing step are much more expensive anyway).
              - Ugh, we also need to update the whole database, to refresh the keyword sets for existing documents (they will change when the corpus changes).
    - Improve user text entry: multiline input
    - Upgrade translator
      - HPLT consortium, new version (8April 2025) of the earlier model by Helsinki-NLP that we use currently
      - Needs new infra at the backend: Marian format (not HuggingFace format)
        - https://huggingface.co/HPLT/translate-en-fi-v2.0-hplt_opus
    - Add feature: Switch chat (from all leaf nodes in datastore)
    - Add feature: Avatar: optional digital glitch effect when switching chat branches (change postprocessor config on the fly)
    - Add websearch toggle? (Need to regenerate system prompt with/without tools)
    - Improve chat panel
      - Add double-buffering for rebuilding, like in Raven-visualizer
    - Add feature: save full prompt with each AI message (get it from the `on_prompt_ready` event of `raven.librarian.scaffold.ai_turn`)
      - Add a GUI button and window to show the full prompt (render as Markdown) and to copy it to clipboard
    - Robustness: temporarily disable the relevant buttons while the AI is writing
      - Per-message buttons can be then re-enabled correctly by checking whether the relevant action has a callback stashed for that specific displayed chat message (need to stash button DPG IDs or tags, too)
    - Add feature: Ctrl+F find in current chat history, with highlighting
    - Add feature: search for chats (incremental fragment search for now)
    - Avatar: do more to eliminate stutter while receiving LLM response
      - Is the audio buffer size fine now, or do we need a larger one to eliminate xruns? See `raven.client.util`.
      - How to keep avatar video rendering smooth under high system load?
    - Avatar: vector emotions
      - Blend several emotions by classification values.
      - Boost neutral by (1 - sum(others)), or something. Think about normalization.
    - Add feature: message editing (use chattree's revision system)
    - Add feature: context rolling (and summarization?) when the context window runs out
    - Integration with *Raven-visualizer*: AI summary and synthesis of selected studies
      - The apps could talk to each other over the network? For example, *Raven-visualizer* could send its selection data to *Raven-server*, from which *Raven-librarian* could query the document names to enable.

  - Add a lockfile so that `raven-minichat` and `raven-librarian` can't be running at the same time (to prevent losing changes made in one of the apps)

  - RAG: list the chunk full-IDs in retrieval metadata for combined contiguous chunks?
  - minichat: when are retrieval results `null` in the chat datastore (`data.json`)? Did these come from an old bug that does not exist any more? Could fix while migrating, replacing each null with an empty list.
  - LLM context compaction
    - Drop and/or summarize old messages when the LLM's context window fills.
    - Use `raven.llmclient.token_count` to check token count from LLM backend. Should be possible to bisect the linearized history quickly to find the point where it just fits (plus account for max response length; get this from `settings.request_data["max_tokens"]`, where `settings` is the return value of `raven.llmclient.setup`).
  - RAG: PDF support
    - Use `pdftotext` (from `poppler-utils`) to extract the text, run it through `sanitize`, then add the result to the RAG index
      - May need to improve paragraph break detection in `sanitize`
    - Store a copy of the original document, and keep a link to it
      - In the current design, we can use the `callback` in `HybridIRFileSystemEventHandler` to extract the text; the PDFs can live in the "docs" directory
      - The HybridIR engine chunks the text and indexes those, and throws the original full text away (since not needed for search).
        If we want a copy of the extracted text (for debugging etc.), maybe make an "text" directory (`docs/file.pdf` -> `text/file.txt`), and save the text there in the callback
      - Rename `callback` to `add_or_update_callback`; add a `delete_callback` so we can delete the extracted texts as necessary
    - Upon retrieval, return also the link to the original document (useful for GUI: launch original in viewer, with viewer command configured in librarian settings)
    - This readily generalizes to other input formats
      - Images, too, if we use CLIP or something to generate a natural-language caption

- Improve MacOSX support
  - Raven already runs on MacOSX, but some things could be improved.
  - When running on MacOSX, instead of the Ctrl key, all hotkeys use the Cmd key instead.
    - We should detect the OS we're running on at app startup, and change help and tooltips accordingly.
  - Resolve hotkey conflicts with MacOSX builtin hotkeys.
    - As of v0.2.3, the hidden debug window (to show FPS stats) is now Cmd+Shift+M, which works. Note that bare Cmd+M is *Minimize window*.
  - Right-click or right-drag features on MacOSX with a one-button mouse or trackpad?
  - F-keys on MacOSX?
  - OS X 10.x support. In that environment:
    - ChromaDB won't install, because it depends on `onnxruntime`, which won't install. HybridIR won't work, preventing Raven-librarian from starting.
    - `av` won't install, so the TTS's mp3 audio compressor won't work. This may prevent Raven-server from starting, when it tries to import the `tts` module. We could add a `try`/`except` and disable just `tts`.

- Fix issues found via user feedback from initial testing:
  - Installation instructions: TL;DR version (without CUDA), walk through how to import the included dataset
  - Import and show the DOI / URL
    - Export list of DOIs / URLs (for fulltext internet search automation)
  - LATER: use HybridIR search backend
  - Word cloud window size: make it possible to scale the window and image
    - Make the word cloud window resizable
    - Add a 1:1 button to the word cloud window's toolbar to return to pixel-perfect size
    - DPG's image texture scaler seems to be just nearest-neighbor, which is a really bad algorithm (horrible frequency response / aliasing); perhaps we should use Pillow, it has a Lanczos scaler in `Image.resize`
      - Keep a separate copy of the full-resolution texture (word cloud output)
      - Keep a dynamic texture copy that is shown in the GUI
      - Lanczos-scale the data in the GUI texture in a bgtask (Lanczos scale to target size, then nearest-neighbor scale the result to the fixed texture size)
    - Selectable color scheme (for white background, for export to papers)
  - Toggle fullscreen -> Fullscreen mode / Exit fullscreen mode, with icons
  - Full report of all selected items that doesn't care whether the items fit into the info panel
  - Make the text headings clickable in the import window (same as clicking the corresponding button)
  - Make highlight visualization clearer, now it obscures which cluster each data point belongs to
    - Maybe just an outline, not a filled circle?
    - Brighten the data point's own color, don't use a separate color? (Difficult, DPG needs one data series per color)
  - Add a screenful of spacer at end of info panel, to be able to scroll to last cluster
  - Make the "Search" heading brighter to make it stand out
  - Highlight/color data points by year, so that newer research is brighter
    - What to do with Misc items which are scattered all over the semantic map? Toggle for show/hide them?
  - What if the AI models update?
    - Currently, we just auto-install them once, at first use after Raven itself is installed.
    - Do we need an "update AI models" feature?
    - Models:
      - Semantic embedder
      - spaCy NLP model
      - QA embedder (for upcoming intelligent search features)
  - BibTeX import:
    - handle umlauts Å, Ä, Ö, Ü, å, ä, ö, ü: e.g. {\"o} -> ö
    - drop BibTeX's "verbatim" braces: {GPU} clusters -> GPU clusters

  - Misc items: assign to closest cluster in 2D view?
  - Show full authors in abstract field in info panel?
  - spaCy NLP for arbitrary input language? (Especially Finnish?)
  - Make stopword list configurable.
  - Show the most common keywords in the import panel when the import finishes.
  - Smaller/scalable word cloud view?


- Save/load selection, for reproducible reports. (Needs some care to make it work for a dynamic dataset.)
  - This becomes especially important with the LLM client, as the selection will affect which documents are enabled for RAG, so chat histories will be selection-specific.
    So it will feel silly if there is no way to save/load selections without attaching an AI chat to them.
  - The GUI needs some thinking. What is a good UX here?

- Check if we can auto-spawn a server from raven-visualizer (and other end-user apps) if it's not already running.
  - Would need open a terminal to show the server's log messages.
  - OTOH, maybe no need if we can support a local (one process) mode instead.

- LLMClient, to prepare for interactive AI summarization:
  - Expand tool-calling functionality.
    - Add possibility for the user to call the tools, too? (Then control returns back to user. Use Python syntax?)
    - Finish websearch.
      - Final formatting for the results.
      - Store the raw results into the chat tree.
      - Add link crawling to retrieve the full result documents. (These should be safe to download.)
        - Persist the documents to the RAG database to avoid unnecessary re-downloading.
          - Set an expiry timeout for each downloaded page.
          - How to decide in which contexts the search result pages should be enabled as RAG data sources?
    - Add a "download web page" tool?
      - Infosec needs some consideration here.
        - Some web pages are not documents, but actions (e.g. wikipedia "Edit" link). Some pages may contain viruses that could corrupt or hijack the web driver.
        - User-provided links might be considered safe? ("Here, take a look at this: [URL]" -> allow the AI to download that page if it wants to)
    - Add a weather tool.
      - https://open-meteo.com/en/docs
    - Add a calendar tool (get one- or three-month calendar, like the `cal` command-line utility). See Python's `calendar` module.
    - Add a calculator tool, if this can be done securely.
      - I'd use Python restricted to the `math` module, but `eval` itself is unsafe: e.g. ().__class__.__base__.__subclasses__()[-1].__init__.__globals__['__builtins__']['__import__']('os').system('install ransomware or something')
        https://stackoverflow.com/questions/64618043/safely-using-eval-to-calculate-using-the-math-module-in-python
      - Maybe https://github.com/danthedeckie/simpleeval
  - Fix RAG document IDs so that they are unique across subdirectories of the RAG datastore
  - Add a pedigree field to `HybridIR` documents, so that the automatic rescan can auto-remove only documents added by that scanner (name the scanner instances).
    - There may be occasions we need to programmatically send data into the RAG index, e.g. web pages from websearch.
  - Source attribution for RAG search and websearch results.
    - In GUI: RAG: clickable snippets based on `document_id`, `offset`, length; plus a clickable link to open the full document (need to spawn external program depending on file type).
  - Inline citations?
  - See where to stuff the RAG search data in the chat tree, it's not part of the standard format.
  - Tune the system prompt, consider how it needs to be different when LLM speculation is on/off.


## v0.3 and later

### Large new features

- AMD GPU support.
  - We only use GPU compute via Torch and via some other libraries such as `sentence_transformers` and `spaCy`.
  - What needs to be done here?
    - What packages should be installed? (I don't have an AMD environment to test/develop in.)

- Visualizer: visualize links between documents if link/reference information available in data?

- **More import sources**.
  - We currently have:
    - Web of Science (working).
      - Fix a bug with character escapes (quotes, braces, etc.) that's currently breaking the import for several files in our test set.
    - PDF conference abstracts (WIP, in beta).
      - Improve robustness. Maybe needs a bigger LLM?
  - Could be useful:
    - Semantic Scholar
    - Scopus
    - ERIC (educational sciences / didactics)

- **Filtering**. Real-time view filtering, e.g. by authors or year range.
  - Color data points by year (instead of cluster number; what to do about outliers?).
  - Needs the full list of authors ("Author, Other, Someone"), not just the summarized version ("Author et al."). The proprocessor doesn't currently save that to the dataset.
  - Need an "inactive" scatter series *on the bottom* of the plot, so that it doesn't cover the active datapoints (that match the filter).
    - Maybe one series per cluster: grayscale each color separately, and use a monotonic-brightness color map. This gives the appearance of the datapoints retaining their identity, just becoming grayed out.
  - Any code handling or variables containing indices to `sorted_xxx` should now be tagged as such in the comments, so we can find what needs to change when we add filtering.
    - Actually, might not have to change much. Just yes/no filter (`unpythonic.partition`) the data into two scatter series, per cluster, using the filter condition, on the fly when the filter condition changes.
    - No need to reorder the data in `sorted_xxx`. The only time we even access the scatterplot (which needs separate data for each color) is when we load in the data, and it takes in a separate copy of the coordinates.
      Everything else is done directly on `sorted_xxx`.
    - We can use the original sorted numbering (as in v0.1) as our internal numbering for the datapoints, no need to change it because of filtering.
  - Year range.
    - Ideally, a range slider with two tabs. If not available, two single-tab sliders vertically near each other (top slider for start year, bottom slider for end year).
    - Show relative data mass for each year (some color, brightness) over/under the slider, to show the user see where the interesting years are.
    - Just above the slider, show year numbers in some reasonable interval (decade?). Show tick marks.
    - Snap the slider tab to years that actually exist in the dataset?
  - We still need to update quite many places that should/should not look at data that has been filtered away. Just add `if` or `np.where` as appropriate.
  - Add support features:
    - Convert filter to selection, and then clear the filter (allows e.g. easily selecting all datapoints from years 2020-2024)
    - Invert selection (continuing previous example, then look at data up to year 2019)
    - Convert selection to filter
      - Needs some thinking how to display the result in the GUI; an arbitrary selection is not a year-range filter.

- **AI summarize**: call an LLM to generate a summary report of items currently shown in info panel (or of the full selection).
  - Preprocess the per-datapoint summarization.
    - Condense each abstract into one sentence with just the most important main point.
    - Is it better to make abstractive summaries with an LLM, or a summarization-specific AI?
    - To evaluate summary accuracy, `seahorse-large` based on `mT5-Large` (6 models, 5 GB each)? https://github.com/google-research-datasets/seahorse
  - Scaffold to produce guaranteed-correct citations.
    - In AI summarization, process each document separately to eliminate cross-contamination.
     - Check each summary via LLM to guard against hallucinations. Does all the information in the summary come from the original text? (It's not published yet, but we already have a prototype prompt to do this.)
    - Whenever we perform a search (whether keyword-based on semantic vector lookup), we know which items matched. Keep track of their IDs in the scaffold.
    - When answering a question based on documents:
      - Single source document: in the scaffold, collect that document to an internal reference list. Paste the document to the LLM context and ask the question. Append a citation to the end of the text produced by the LLM. (e.g. "[1]")
      - Multiple source documents: in the scaffold, collect those documents to an internal reference list. Paste their *summaries* into the LLM context (to save on amount of context used) and ask the question. Append a citation to *all those documents* to the end of the text produced by the LLM. (e.g. "[1, 2, 3]")
      - Validate aggressively. Use heuristics in scaffold, plus LLM analysis. Use techniques from literature to improve quality of LLM analysis (ensembles, debate, multi-persona analysis, ...).
    - At the end, use the collected internal reference list to write the actual list of cited documents programmatically.
      - For each citation, show the matched fragment somewhere?

- **Extend existing dataset**.
  - Two separate new features:
     - 1) Update an existing semantic map, adding new datapoints to it (easy to implement).
          - Add an option to the BibTeX importer to add more data on the same topic to an existing dataset, using the already trained dimension reduction from that dataset.
          - Before adding each new item, check that it's not already in the dataset. Add only new items (and report them as such, in log messages).
            This allows re-scanning a BibTeX database for any new entries added since the last time it was imported. (What to do with changed entries? Removed entries?)
          - Produce a new dataset file (to avoid destroying the original).
          - Need to save the dimension reduction weights in the dataset file. See what OpenTSNE recommends for its serialization.
          - How to cluster the new datapoints? Re-run the 2D clustering step? Or snap to closest existing cluster (compare keyword frequencies from each new datapoint to each existing cluster)? Maybe an option to do either?
          - This should work as long as the original dataset covers the semantic space well enough to initialize the mapping,
            so that the new datapoints fall on or at least near the same manifold.
     - 2) Comparative analysis between datasets on the same topic (maybe more difficult).
          - E.g. see how the set of studies from one's own research group locates itself in the wider field of science.
          - This requires using two or more different color schemes in the plotter simultaneously. Also, which dataset should go on top (more visible)?

- **More flexible BibTeX import**.
  - Rethink what our native input format should be. BibTeX is nice for research literature, but the Raven core could be applicable to so much more: patent databases, Wikipedia, news articles, arbitrary text files, Linux system logs, ...
  - User-defined Python function: input record -> object to be embedded (allowing customization of which fields to use)
    - E.g. for scientific papers, could be useful to use also the abstracts, and author-supplied keywords, not only the titles. But depends on the quality of the embedding model; so far, the best clustering has been obtained using titles only.
  - Embedding model: object to be embedded -> high-dimensional vector (allowing embedding of different modalities of data: text, images, audio)

- **Other input modalities** beside text.
  - Raven operates on semantic vectors, so the core is really modality-agnostic. The input doesn't need to be text.
  - Images.
    - Use filename as title, generate text description via an AI model. Compute the sentence embedding from the generated text description.
    - Can use the same AI models that Stable Diffusion frontends do; CLIP to describe photos, and Deepbooru to describe anime/cartoon art. Make this configurable.
    - Could also use a multimodal embedding model (text/images → latent), if available.
    - Show the generated text description as the abstract. Show the images, too: in the info panel, and as a thumbnail (Lanczos downscaled for best quality) in the annotation tooltip.
    - This allows us to go truly multimodal: now, add to the same dataset some text datapoints that talk of the same topics that are shown in the images... if the text embedding model is good, then e.g. a Wikipedia article for "Apollo program" and an image of a Saturn V rocket should automatically end up near each other on the semantic map.
  - Audio.
    - Other than speech, e.g. music; speech can be converted to text.
    - See if we can source an audio sample embedder e.g. from some promptable music genAI or from a text description generator (if those exist for music).
    - Need some thinking about details for visualization. Maybe HDR-plot the audio waveform? (Data-adaptive dynamic range compression, as in the SAVU experiment back in 2010; better than a log-plot.)
  - To check: has anyone trained an embedder for mathematical equations?
  - Need a document type field, and GUI support for showing different kinds of assets in the annotation tooltip and in the info panel.
  - Large files (images, audio, full PDFs) shouldn't be embedded into the dataset file, but rather just linked to.

- Chatbot integration.
  - Would be useful to have a RAG-enabled LLM to "talk with the dataset". Needs major GUI work, though (to have usability on par with existing solutions such as SillyTavern).

- PDF import: OCR. Shop around for AI models.


### Small improvements

- Publish a ready-made dataset to allow users to quickly try out the tool, e.g. AI papers from arXiv.

- Find a good font for rendering scientific Unicode text.
  - InterTight renders subscript numbers as superscript numbers, which breaks the rendering of chemistry formulas.
  - OpenSans is missing the subscript-x glyph, which also breaks the rendering of chemistry formulas (e.g. "NOₓ").

- BibTeX importer:
  - Detect and report duplicate entry keys to ease debugging on BibTeX databases.
  - Run the abstracts through the `sanitize` module.
  - For cluster-level keyword detection, first de-duplicate words in each abstract?
    (This avoids "keyword spam" from a single abstract from dominating the cluster keywords.)
  - Make it configurable which fields to use for the semantic embedding.
  - Make the stopword list configurable (text file).
  - Investigate more advanced NLP methods to improve the quality of the automatically extracted keyword list.

- Fragment search for authors, year, abstract, ...so maybe make configurable which fields to search in. Add checkboxes (and a select/unselect all button) below the search bar?

- Semantic orienteering: Search for datapoints semantically similar to a given piece of text (type it in from keyboard).
  - Embed the input text, dimension-reduce it, highlight the resulting virtual datapoint in the plotter.
  - Later: add support for doing this for a user-given BibTeX entry or PDF file.

- We can now import items that have no abstract. Think of how to generalize this to arbitrary missing fields, when we eventually allow the user to choose which fields to embed in the BibTeX import step.

- The importer should detect duplicates (if reasonably possible), emit a warning, and ignore (or merge?) the duplicate entry.

- Generate report of full selection (without rendering it into the info panel). The info panel is currently a bottleneck for large selections of data.
  - Hotkey: add Ctrl to the current hotkeys: Ctrl+F8 for plain text (whole selection), Ctrl+Shift+F8 for Markdown (whole selection)?
  - Need to separate the report generator from the info panel renderer (`_update_info_panel`) so that we can easily get data in the same format for the full selection.

- BibTeX report/export to clipboard. Needs the entire original BibTeX records, currently not saved by the BibTeX importer. Could save them as-is into the dataset.

- Record also the DOI (if present) in the BibTeX importer. Useful for opening the webpage, and for external tools.

- Info panel: may need a full row of buttons per item. This would also act as a visual separator.
  - Add per-item button to use the DOI to open the official webpage of the paper in the default browser: "https://dx.doi.org/...".
  - Add per-item button to search for other items by the same author(s). Should rank (or filter) by number of shared authors, descending. Needs the full list of authors. The BibTeX importer currently saves it, but Raven-visualizer doesn't use it yet.

- Timeline granularity: not only publication year, but also month/day, useful e.g. for news analysis. Analysis of system logs needs a full timestamp, because milliseconds may matter.

- Add a GUI filter to search hotkeys in the help window? Fragment search, by key or action.

- Add pre-filtering of data at BibTeX import time, e.g. by year.

- Deployability: place user-configurable parts in `~/.config/raven.conf` (or something), not inside the source tree. Check also where it should go on OSs other than Linux (Windows, OS X).

- Data file format: `pickle` is not portable across Python versions or indeed even app versions. Use `npz` or something.

- Multiple datasets, to place one dataset into the wider context of another (e.g. one's own research within a whole field of science). How to color-code the datasets in the plot?

- Make all colors configurable. May need a lot of work. We must customize *every colorable item* in the theme, since the default theme cannot be queried for its colors in DPG.
  The app itself doesn't know e.g. the color of the info panel background, which makes it hard to color the dimmer correctly if the theme ever changes.
  Also, all the custom colors we use have been chosen to visually fit DPG's default color scheme.

- Somehow visualize how the selection was produced.
  - E.g. search "cat photo", add search "solar", subtract search "vehicle", ... -> results mostly solar panel related.
  - As of v0.1.0, there are already many ways to build the selection:
    - Search, add search, subtract search
    - Select all in view, intersect to those in view
    - Select all in same cluster (also add, subtract)
    - Paint by mouse (add, subtract)

- Add a settings window. Expose `gui_config`. See what triggers we need to reconfigure existing GUI elements. Try to avoid a need to restart the app when GUI settings change.

- Make the annotation tooltip (and info panel?) configurable - which fields to show, sort by which field, ...

- Drag'n'drop from the OS file manager into the Raven window to open a dataset.
  - As of DPG 2.0.0, drag'n'drop from an external application doesn't seem to be implemented for Linux. For Windows there's an add-on. We need a cross-platform solution. Keep an eye on this.

- Detect novelty of research, to automatically identify promising research directions. Maybe something like the inverse of the number of semantically nearby items in dataset? (Dense regions, less novelty; sparse regions, more novelty.)


### Technical improvements

- Test the `SONAR` sentence embedder for creating the semantic embedding; is this better than `arctic-snowflake-l`?
  - https://github.com/facebookresearch/SONAR
  - SONAR can read input text in 200 languages, as well as translate between them, and also convert speech to text in 37 languages.

- Possible NLP tools for cleaning up documents, if needed:
  - SaT: https://github.com/segment-any-text/wtpsplit
  - dehyphen: https://github.com/pd3f/dehyphen/

- "Detailed debug" logging level.
  - Some debug loggings are particularly spammy, but would be nice to have when specifically needed.
    - `SmoothScrolling.render_frame`
    - `_managed_task`
    - `binary_search_item`, `find_item_depth_first`, etc.

- Performance, both the render framerate and the speed of background tasks. Improve if possible, especially during info panel building.
 - As of November 2024, in the info panel 100 items is fine, but 400 is already horribly slow. Judging by eyeballing the progress indicator, the update seems to be O(n²).
 - That's already a lot of text to read, though: at 200 words per abstract, one item is ~1/4 of an A4 page, so 100 items ~ 25 pages, and 400 items ~ 100 pages.

- Post a PR of our vendored FileDialog fixes so that other projects can benefit, and so that we can benefit from upstream maintenance of FileDialog.


### Robustness, bug fixing

- Crash-proof the app, just in case:
  - Periodically save into a crash recovery file: which file was open, selection undo history, search status.
  - If a crash recovery file exists, load it on startup. Flash a non-blocking notification that the previous state was restored.

- Test again in DPG 2.0.0: Figure out where the keyboard focus is sometimes when the search field is not focused (at least visually), but the navigation keys still won't operate the info panel.

- Test again in DPG 2.0.0: At least with 1.x, there was a very rare race condition that crashed `hotkeys_callback`: looking up the search field failed, as if the GUI widget didn't exist. DPG attempted to look up widget 0, and it doesn't exist.

- Test again in DPG 2.0.0: DPG crash: App sometimes crashes if Ctrl+Z is pressed in the search bar, especially after clearing the search.

- The word cloud window is currently shown under the toolbutton highlight, because the highlight is in a viewport overlay. Also sometimes below info panel dimmer.  Figure out if we can fix this and how.

- fdialog: Ctrl+F hotkey to focus the file name field is not always working. Figure out the exact conditions and see if we can fix this.
