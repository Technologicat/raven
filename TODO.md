# Raven-visualizer TODO


## v0.1.1 (January 2025?)

*Preliminary plan, not final. Details may change.*

- Improve codebase maintainability:
  - Move the reusable animation components into their own module.
  - Move the background task manager into its own module.
  - The "General utilities" section could live in a separate module
  - Other generic utilities:
    - `compute_tooltip_position_scalar`
    - `_mouse_inside_widget`
  - `binary_search_item` and its friends could live in a separate module
  - Plotter utilities: `get_pixels_per_plotter_data_unit`, `get_data_idxs_at_mouse` (though the second one needs the KDtree, so maybe not that?)
  - Vendor our fixed wosfile library?
  - `vis_data` should really be called `entries` everywhere in this app constellation, also in importers and in the preprocessor.

- Make the Refresh and Go back buttons of the file dialog flash when clicked, to indicate that the action did indeed take.

- Word boundary mark (\b) for search.

- App icon. `small_icon` and `large_icon` parameters of `create_viewport` (png or ico format).
   https://github.com/hoffstadt/DearPyGui/discussions/1688
   https://dearpygui.readthedocs.io/en/1.x/reference/dearpygui.html#dearpygui.dearpygui.create_viewport

- Word cloud from selection? Render as image, display in a separate DPG window, offering a "save as PNG" button.


## v0.2 and later

### Large new features

- **Filtering**. Real-time view filtering, e.g. by authors or year range.
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

- **Extend existing dataset**.
  - Two separate new features:
     - 1) Update an existing semantic map, adding new datapoints to it (easy to implement).
          - Add an option to the preprocessor to add more data on the same topic to an existing dataset, using the already trained dimension reduction from that dataset.
          - Before adding each new item, check that it's not already in the dataset. Add only new items (and report them as such, in log messages).
            This allows re-scanning a BibTeX database for any new entries added since the last time it was preprocessed. (What to do with changed entries? Removed entries?)
          - Produce a new dataset file (to avoid destroying the original).
          - Need to save the dimension reduction weights in the dataset file. See what OpenTSNE recommends for its serialization.
          - How to cluster the new datapoints? Re-run the 2D clustering step? Or snap to closest existing cluster (compare keyword frequencies from each new datapoint to each existing cluster)? Maybe an option to do either?
          - This should work as long as the original dataset covers the semantic space well enough to initialize the mapping,
            so that the new datapoints fall on or at least near the same manifold.
     - 2) Comparative analysis between datasets on the same topic (maybe more difficult).
          - E.g. see how the set of studies from one's own research group locates itself in the wider field of science.
          - This requires using two or more different color schemes in the plotter simultaneously. Also, which dataset should go on top (more visible)?

- **AI summarize**: call an LLM to generate a summary report of items currently shown in info panel (or of the full selection).
  - Preprocess the per-datapoint summarization.
    - Condense each abstract into one sentence with just the most important main point.
    - Is it better to make abstractive summaries with an LLM, or a summarization-specific AI?
    - To evaluate summary accuracy, `seahorse-large` based on `mT5-Large` (6 models, 5 GB each)? https://github.com/google-research-datasets/seahorse

- **Integrated preprocessor**. Access the preprocessor from the GUI app, to quickly import small datasets with minimal fuss.
  - APIfy the preprocessor. Make it callable from the GUI app.
  - Make the preprocessor run in a background thread/process, to allow the user to explore previous existing datasets in the GUI while the preprocess is running. Notify when complete.
  - While a preprocess is running, show a progress bar in the GUI.
    - Progress bar in DPG: https://github.com/my1e5/dpg-examples/blob/main/threading/progress_bar.py

- **More import sources**.
 - We currently have:
   - Web of Science (working).
     - Fix a bug with character escapes that's currently breaking the import for one file in our test set.
   - PDF conference abstracts (WIP, in beta).
     - Improve robustness.
 - Could be useful:
   - arXiv, to stay on top of developments in AI.

- **More flexible preprocessing**.
  - Rethink what our native input format should be. BibTeX is nice for research literature, but the Raven core could be applicable to so much more: patent databases, Wikipedia, news articles, arbitrary text files, Linux system logs, ...
  - User-defined Python function: input record -> object to be embedded (allowing customization of which fields to use)
    - E.g. for scientific papers, could be useful to use also the abstracts, and keywords, not only the titles. But depends on the quality of the embedding model; so far, the best clustering has been obtained using titles only.
  - Embedding model: object to be embedded -> high-dimensional vector (allowing embedding of different modalities of data: text, images, audio)

- **Other input modalities** beside text.
  - Raven operates on semantic vectors, so the core is really modality-agnostic. The input doesn't need to be text.
  - Images.
    - Use filename as title, generate text description via an AI model. Compute the sentence embedding from the generated text description.
    - Can use the same AI models that Stable Diffusion frontends do; CLIP to describe photos, and Deepbooru to describe anime/cartoon art. Make this configurable.
    - Show the generated text description as the abstract. Show the images, too: in the info panel, and as a thumbnail (Lanczos downscaled for best quality) in the annotation tooltip.
    - This allows us to go truly multimodal: now, add to the same dataset some text datapoints that talk of the same topics that are shown in the images... if the text embedding model is good, then e.g. a Wikipedia article for "Apollo program" and an image of a Saturn V rocket should automatically end up near each other on the semantic map.
  - Audio.
    - Other than speech, e.g. music; speech can be converted to text.
    - See if we can source an audio sample embedder e.g. from some promptable music genAI or from a text description generator (if those exist for music).
    - Need some thinking about details for visualization. Maybe HDR-plot the audio waveform? (Data-adaptive dynamic range compression, as in the SAVU experiment back in 2010; better than a log-plot.)
  - To check: has anyone trained an embedder for mathematical equations?
  - Need a document type field, and GUI support for showing different kinds of assets in the annotation tooltip and in the info panel.
  - Large files (images, audio, full PDFs) shouldn't be embedded into the dataset file, but rather just linked to.


### Small improvements

- Publish a ready-made dataset to allow users to quickly try out the tool, e.g. AI papers from arXiv.

- Fragment search for authors, year, abstract, ...so maybe make configurable which fields to search in. Add checkboxes (and a select/unselect all button) below the search bar?

- Semantic orienteering: Search for datapoints semantically similar to a given piece of text (type it in from keyboard).
  - Embed the input text, dimension-reduce it, highlight the resulting virtual datapoint in the plotter.
  - Later: add support for doing this for a user-given BibTeX entry or PDF file.

- Import also items that have no abstract. We only really need a title to compute the embedding.
  - Think of how to generalize this (to arbitrary missing fields) when we allow the user to choose which fields to embed.

- Generate report of full selection (without rendering it into the info panel).
  - Hotkey: add Ctrl to the current hotkeys: Ctrl+F8 for plain text (whole selection), Ctrl+Shift+F8 for Markdown (whole selection)?
  - Need to separate the report generator from the info panel renderer (`_update_info_panel`) so that we can easily get data in the same format for the full selection.

- BibTeX report/export to clipboard. Needs the BibTeX records, currently not saved by the preprocessor. Could save them as-is into the dataset.

- Record also the DOI (if present) in the preprocessor. Useful for opening the webpage, and for external tools.

- Info panel: may need a full row of buttons per item. This would also act as a visual separator.
  - Add per-item button to use the DOI to open the official webpage of the paper in the default browser: "https://dx.doi.org/...".
  - Add per-item button to search for other items by the same author(s). Should rank (or filter) by number of shared authors, descending. Needs the full list of authors. The preprocessor doesn't currently save that.

- Timeline granularity: not only publication year, but also month/day, useful e.g. for news analysis. Analysis of system logs needs a full timestamp, because milliseconds may matter.

- Add a GUI filter to search hotkeys in the help window? Fragment search, by key or action.

- Add pre-filtering of data at preprocess time, e.g. by year.

- Deployability: place user-configurable parts in `~/.config/raven.conf` (or something), not inside the source tree. Check also where it should go on OSs other than Linux (Windows, OS X).

- Data file format: `pickle` is not portable across Python versions or indeed even app versions. Use `npz` or something.

- Save/load selection, for reproducible reports. (Needs some care to make it work for a dynamic dataset.)
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