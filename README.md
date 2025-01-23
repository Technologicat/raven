<p align="center">
<img src="img/logo.png" alt="Logo of Raven" height="200"/> <br/>
</p>

-----

<p align="center">
<img src="img/screenshot-main.png" alt="Screenshot of Raven's main window" width="800"/> <br/>
<i>12 000 studies on a semantic map. Items matching your search terms are highlighted as you type.</i>
</p>

# Introduction

**Raven** is an easy-to-use research literature visualization tool, powered by AI and [NLP](https://en.wikipedia.org/wiki/Natural_language_processing). It is intended to help a scholar or subject matter expert to stay up to date as well as to learn new topics, by helping to narrow down which texts from a large dataset form the most important background for a given topic or problem.

- **Graphical user interface** (GUI). Easy to use.
- **Fully local**. Your data never leaves your workstation/laptop.
- **Usability-focused**. Modern animated GUI with immediate visual feedback on user actions. Most functions accessible from keyboard.
- **Semantic clustering**. Discover **vertically**: See how a field of science splits into topic groups. Explore similar papers.
- **Fragment search**. Discover **horizontally**: E.g. find studies (across all topics) where some particular method has been used.
- **Info panel**. Read the abstracts (if available) of the studies you discover, right there in the GUI.
- **Open source**. 3-clause BSD license.

*Fragment search* means that e.g. *"cat photo"* matches *"photocatalytic"*. This is the same kind of search provided by the Firefox address bar, or by the `helm-swoop` function in Emacs. Currently the search looks only in the title field of the data; this may change in the future.

**Raven is NOT a search engine.** Rather, for its input, it uses research literature metadata (title, authors, year, abstract) for thousands of papers, as returned by a search engine, and plots that data in an interactive semantic visualization.

**:exclamation: Raven is currently in beta. :exclamation:**

The basic functionality is complete, the codebase should be in a semi-maintainable state, and most bugs have been squashed. If you find a bug that is not listed in [TODO.md](TODO.md), please open an issue.

We still plan to add important features later, such as filtering by time range to help discover trends, and abstractive AI summaries of a user-selected subset of data (based on the author-provided abstracts).

We believe that at the end of 2024, AI- and NLP-powered literature filtering tools are very much in the zeitgeist, and that demand for them is only rising. Thus, we release the version we have right now as a useful tool in its own right, but also as an appetizer for future developments to come.

<p align="center">
<img src="img/screenshot-help.png" alt="Screenshot of Raven's help card" width="800"/> <br/>
<i>The help card. Most functions are accessible from the keyboard.</i>
</p>

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Introduction](#introduction)
- [Import a dataset](#import-a-dataset)
    - [BibTeX](#bibtex)
    - [Other formats](#other-formats)
        - [WOS (Web of Science)](#wos-web-of-science)
        - [PDF (human-readable abstracts)](#pdf-human-readable-abstracts)
            - [How it works](#how-it-works)
            - [LLM requirements](#llm-requirements)
- [Visualize an imported dataset](#visualize-an-imported-dataset)
    - [Load a file in the GUI](#load-a-file-in-the-gui)
    - [Load a file from the command line, when starting the app](#load-a-file-from-the-command-line-when-starting-the-app)
    - [Create a word cloud](#create-a-word-cloud)
- [Install & uninstall](#install--uninstall)
    - [From PyPI](#from-pypi)
    - [From source](#from-source)
        - [Uninstall](#uninstall)
- [Limitations](#limitations)
- [Technologies](#technologies)
- [Other similar tools](#other-similar-tools)
- [License](#license)
- [Acknowledgements](#acknowledgements)

<!-- markdown-toc end -->

# Import a dataset

Raven uses the following workflow:

```
+-------+             +--------+                 +---------+
|  any  | --import--> | BibTeX | --preprocess--> | dataset | --> interactive visualization
+-------+             +--------+                 +---------+
```

where the `import` step is optional; BibTeX, widely used in the engineering sciences, is considered the native input format of Raven.

The input does not strictly have to be research literature. Anything that can be defined to have `title`, `authors`, `year`, and `abstract` fields, where *abstract* is any kind of human-readable short text summary, can be used as input. That said, the titles are used for linguistic analysis, so having precise titles (as in common in scientific papers) is likely to produce a more accurate semantic map.

Note that even BibTeX data needs to be preprocessed before it can be visualized.

The preprocessing step typically takes some time, so it is performed offline (in the sense of a batch job). All computationally expensive procedures, such as semantic embedding, clustering, keyword analysis, and training the dimension reduction for the dataset, are performed during preprocessing. Some of these, particularly the semantic embedding, support GPU acceleration.

The data is clustered automatically, and each cluster of data has its keywords automatically determined by a linguistic analysis. It is not possible to edit the clusters or keywords. If something is detected incorrectly, it is more in the spirit of Raven to improve the algorithms rather than hand-edit each dataset. Raven is intended to operate in an environment that has too much data, and where the data updates too quickly, for any kind of manual editing to be feasible at all.

The preprocessing step produces a **dataset file**, which can then be opened and explored in the GUI.


## BibTeX

BibTeX is considered the native input format of Raven.

To preprocess one or more BibTeX databases into a dataset file named `mydata.pickle`:

```bash
conda activate raven
python -m raven.preprocess mydata.pickle file1.bib file2.bib ...
```

The output filename can be anything, but Raven expects it to have the `.pickle` file extension. The file extension will likely change in the future once we move to a more portable data format.

The preprocessor caches its intermediate data per input file, so you can include e.g. `file1.bib` into multiple different dataset files, and the expensive computations specific to that input file will only happen once. The caching mechanism checks the timestamps; when e.g. `file1.bib` is processed, computations are re-done if `file1.bib` has changed after the cache was last updated.

Currently, the dimension reduction that produces the 2D semantic map is trained using up to 10k data items, picked at random if there are more.

Currently, it is not possible to add new data into an existing visualization dataset. This will likely change in the future.


## Other formats

First import your data into BibTeX, then preprocess the BibTeX data as above.

We plan to add more importers in the future. Particularly, an arXiv importer would be useful for following AI/CS research.


### WOS (Web of Science)

Web of Science indexes much of the engineering sciences. To import WOS into BibTeX:

```bash
conda activate raven
python -m raven.wos2bib input1.txt ... inputn.txt 1>output.bib 2>log.txt
```

where the input `.txt` files are WOS files exported from Web of Science.

In the example, the output is written to `output.bib`, and any log messages (such as warnings for broken input data) are written to `log.txt`.


### PDF (human-readable abstracts)

We include an AI-based importer for free-form, human-readable PDF abstracts. If you are organizing a scientific conference, this allows analyzing the submissions.

**:exclamation: This functionality is currently in beta. :exclamation:**

- The text content of the PDF is analyzed via an LLM (large language model).
- The PDF must have its text content readable by `pdftotext` (from `poppler-utils`).
- Each PDF should contain one abstract. Multiple abstracts are fed in as separate PDF files.
- The importer does not enforce a length limit, but its intended use case is a typical conference abstract, 1-2 pages in length.
- The abstract should have a human-recognizable title, authors, and main text. Exact formatting does not matter.
- If the abstract contains a line beginning with "*keywords:*" or "*key words:*", the importer will attempt to also extract keywords.

To import PDF into BibTeX:

```bash
conda activate raven
python -m raven.pdf2bib http://127.0.0.1:5000 -o done 1>output.bib 2>log.txt
```

The "*http://...*" argument is the URL of an LLM serving an OpenAI-compatible API (streaming mode).

The command imports all PDF files in the current directory, descending into subdirectories. The files are processed one directory at a time, in Unicode lexicographical order by filename. Output is written to `output.bib`, and log messages to `log.txt`.

The `-o done` moves each PDF file into a subdirectory named `done` after the file has been processed. This allows canceling the job and easily continuing it later, which is useful if there are lots of input files; the LLM analysis can be slow. An input PDF file is moved if and only if it was successfully processed, **after** printing its BibTeX entry.

The directory specified by `-o` is ignored while descending into subdirectories.

#### How it works

The PDF importer analyzes the human-readable text content of the PDF via an LLM. If the text contains a section title *"References"*, anything after that point is discarded before processing.

To improve reliability, the fields are processed one at a time. Some prompt engineering has gone both into the system prompt as well as each individual data-extracting prompt. The prompts have been engineered manually; we have not yet looked at automatic prompt optimization.

The extracted data is double-checked by some heuristics for fields for which this is reasonably possible. Any suspicious-looking LLM responses are flagged with a warning. It is **very strongly recommended** to manually double-check any entries that were flagged by comparing the generated BibTeX entry to the human-readable content of the original PDF file, because any flagged entries are **very likely** to be incorrect in one or more ways.

Note that people do actually sometimes submit PDF abstracts with no author list, or even no title. The importer attempts to catch such cases, but is not always successful at doing so.

As is well known, LLMs may make things up, may respond incorrectly, or may occasionally fail to follow instructions correctly. Hence this functionality is in beta.

#### LLM requirements

The PDF importer has been tested on a local Llama 3.1 8B instance running on [Oobabooga](https://github.com/oobabooga/text-generation-webui). This model fits into a laptop's 8 GB VRAM at 4 bits, e.g. in a Q4_K_M quantized format, while leaving enough VRAM for 24576 (24k) tokens of context.

Based on our own testing, accuracy with this LLM is ~80%, or in other words, on average, 8 out of 10 abstracts import without warnings (and also look correct by manual inspection).

Support for LLM authentication (API key) has not been implemented yet, so currently the LLM needs to be local (on the same network, no API key). Also, the importer expects the LLM to accept a custom system prompt (unlike cloud LLMs, whose system prompts are hidden and uneditable). Supporting cloud LLMs is not a high priority, but PRs are welcome.


# Visualize an imported dataset

First, if raven-visualizer is not yet running, start it:

```bash
conda activate raven  # see Installation below
python -m raven.app
```

Instead of `python -m raven.app`, you can also just use the command `raven-visualizer` (installed when you install the software).

For details on how to use the app, see the built-in Help card. To show the help, click the "?" button in the toolbar, or press F1.

## Load a file in the GUI

To load your dataset file, you can then click on the *Open dataset* button in the toolbar, or press Ctrl+O, thus bringing up this dialog:

<p align="center">
<img src="img/screenshot-open-file.png" alt="Screenshot of Raven's open dataset dialog" width="800"/> <br/>
<i>Opening an imported dataset for visualization.</i>
</p>

Double-clicking a directory in the list changes to that directory. Double-clicking the ".." directory goes one level up.

The buttons at the top of the dialog refresh the view of the current directory, and jump back to the default directory, respectively.

You can focus the *Search files* field by pressing Ctrl+F. Searching filters the view live, as you type. If the search has exactly one match in the current directory (i.e. when only one file is shown in the list, not counting the ".."), that file can then be opened by pressing Enter.

So in this example, to open `out.pickle`, you can press Ctrl+O, then Ctrl+F, type "out" (so that the other file `100.pickle` does not match the search filter), and press Enter.

Pressing Esc cancels the dialog.

**:exclamation: With exception to the search functionality, the open dataset dialog currently requires using the mouse to pick the file. This is a known issue. :exclamation:**

## Load a file from the command line, when starting the app

Raven can also open a dataset file when the app starts:

```bash
python -m raven.app mydata.pickle
```

or

```bash
raven-visualizer mydata.pickle
```


## Create a word cloud

You can make a word cloud from the auto-detected keywords of the studies in the current selection. The size of each word in the picture represents its relative number of occurrences within the selection.

Here is an example:

<p align="center">
<img src="img/screenshot-wordcloud.png" alt="Screenshot of Raven's wordcloud window" width="600"/> <br/>
<i>Word cloud window.</i>
</p>

The "hard disk" toolbutton (hotkey Ctrl+S) opens a "save as" dialog to save the word cloud image as PNG.

The word cloud window hotkey (F10) toggles the window. Note this window is **not** modal, so you can continue working with the app while the window is open, and pressing Esc will not close it.

If the word cloud window is open, it updates automatically whenever the selection changes. Just like in the info panel, the old content remains in the window until the new rendering finishes.

When the word cloud window is opened, Raven checks whether the selection has changed since the last word cloud was rendered. If there are no changes, the latest already rendered word cloud is just re-shown.

The rendering algorithm allocates regions and colors randomly, so even re-rendering with the same data (e.g. in another session later), you will get a different-looking result each time.

The word cloud renderer is Python-based, so it can be rather slow for large selections containing very many data points. The render runs in the background, so you can continue working with your data while the word cloud is being rendered.


# Install & uninstall

Raven is a traditional desktop app. It needs to be installed.

Currently, this takes the form of installing a `conda` environment, and then installing the app via PyPI. At least at this stage of development, app packaging into a single executable is not a priority.

Raven has been developed and tested on Linux Mint. It should work in any environment that has `bash` and `conda` ([miniconda](https://docs.anaconda.com/miniconda/) recommended), including Windows and Mac OS X.

## From PyPI

**Coming soon**

## From source

```bash
conda create -n "raven" python=3.10
conda activate raven
pip install -r requirements.txt
pip install .
```

### Uninstall

```bash
pip uninstall raven-visualizer
```


# Limitations

- Scalability? Beta version tested up to 12k entries, but datasets can be 100k entries in size.
- Hardware requirements, especially GPU. Tested on a laptop with an NVIDIA RTX 3070 Ti mobile, 8 GB VRAM.
- Clustering in high-dimensional spaces is an open problem in data science. Semantic vectors have upwards of 1k dimensions. This causes many entries to be placed into a catch-all "*Misc*" cluster.
- Hyperparameters of the clustering algorithm in the preprocessor may be dataset-dependent, but are not yet configurable. This will change in the future.
- Dataset files are currently **not** portable across different Python versions.
- We attempt to provide keyboard access to GUI features whenever reasonably possible, but there are currently some features where this is not reasonably possible; notably the plotter, and navigation within the *Open dataset* dialog window.
- Configuration is currently hardcoded. See `config.py` for the preprocessor, and the `gui_config` data structure in `main.py` for the app itself. We believe that `.py` files are as good a plaintext configuration format as any, but in the long term, we aim to have a GUI to configure at least the most important parts.


# Technologies

Raven builds upon several AI, NLP, statistical, numerical and software engineering technologies:

- Semantic embedding
  - AI model: [snowflake-arctic](https://huggingface.co/Snowflake/snowflake-arctic-embed-l).
  - Engine for running embedding models: [sentence_transformers](https://sbert.net/).
- Keyword extraction: [spaCy](https://spacy.io/).
- High-dimensional clustering: [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/index.html).
- Dimension reduction: [OpenTSNE](https://opentsne.readthedocs.io/en/stable/).
- AI-powered PDF import
  - AI model: a large language model (LLM), such as [Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B).
  - Engine for running LLMs: an LLM server such as [Oobabooga](https://github.com/oobabooga/text-generation-webui).
  - Communication with the LLM server: [sseclient-py](https://github.com/mpetazzoni/sseclient).
- File format support
  - BibTeX: [BibtexParser](https://bibtexparser.readthedocs.io/en/main/).
  - Web of Science: [wosfile](https://github.com/rafguns/wosfile).
- Graphical user interface: [DearPyGUI](https://github.com/hoffstadt/DearPyGui/).

Note that installing Raven will auto-install dependencies into the same `conda` environment. This list is here just to provide a flavor of the kinds of parts needed to build a tool like this.


# Other similar tools

To our knowledge, [LitStudy](https://nlesc.github.io/litstudy/) is the closest existing tool, but it is a Jupyter notebook, not a desktop app, and its analysis methods seem slightly different to what we use. Also, having existed since 2022, it does have many more importers than we do at the moment.


# License

[2-clause BSD](LICENSE.md).


# Acknowledgements

This work was financially supported by the [gH2ADDVA](https://www.jamk.fi/en/research-and-development/rdi-projects/adding-value-by-clean-hydrogen-production) (Adding Value by Clean Hydrogen production) project at JAMK, co-funded by the EU and the Regional Council of Central Finland.
