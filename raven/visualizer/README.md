<p align="center">
<img src="../../img/logo.png" alt="Logo of Raven" height="200"/> <br/>
</p>

-----

<p align="center">
<img src="../../img/screenshot-main.png" alt="Screenshot of Raven's main window" width="800"/> <br/>
<i>12 000 studies on a semantic map. Items matching your search terms are highlighted as you type.</i>
</p>

<p align="center">
<img src="../../img/raven-visualizer-infographic.png" alt="Raven-visualizer at a glance: a semantic map of about 12,000 studies at the centre, clusters in colour and the studies that fit no cluster in gray. On the left, how a bibliography becomes the map: titles and abstracts embedded by a language model, grouped into clusters, laid out in 2D, and described by keywords. On the right, what you do with it: search, select, read the abstracts by cluster, see a word cloud, and set noise aside. Along the bottom, four cards: &quot;Local and private&quot;, &quot;At scale&quot;, &quot;Honest about filters&quot; and &quot;Keyboard first&quot;." width="800"/> <br/>
<i>How a bibliography becomes the map, and what you do with it.</i>
</p>

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Introduction](#introduction)
- [Import](#import)
    - [What the semantic map is](#what-the-semantic-map-is)
    - [In the GUI](#in-the-gui)
        - [Save imported dataset as](#save-imported-dataset-as)
        - [Select input BibTeX files](#select-input-bibtex-files)
        - [Choose how clusters are found](#choose-how-clusters-are-found)
        - [Start the import](#start-the-import)
    - [From the command line](#from-the-command-line)
    - [Good to know](#good-to-know)
    - [Importing from other formats](#importing-from-other-formats)
        - [arXiv](#arxiv)
            - [Converting a list of arXiv IDs into a BibTeX file](#converting-a-list-of-arxiv-ids-into-a-bibtex-file)
            - [Extracting arXiv IDs from PDF filenames](#extracting-arxiv-ids-from-pdf-filenames)
            - [Auto-downloading arXiv fulltexts by IDs](#auto-downloading-arxiv-fulltexts-by-ids)
        - [WOS (Web of Science)](#wos-web-of-science)
        - [PDF (human-readable abstracts)](#pdf-human-readable-abstracts)
- [Visualize](#visualize)
    - [Load a dataset file in the GUI](#load-a-dataset-file-in-the-gui)
    - [Load a dataset file from the command line, when starting the app](#load-a-dataset-file-from-the-command-line-when-starting-the-app)
    - [Put the info panel on the right](#put-the-info-panel-on-the-right)
    - [How tight the clusters are, and showing loose ones as Misc](#how-tight-the-clusters-are-and-showing-loose-ones-as-misc)
    - [Gray out or hide Misc](#gray-out-or-hide-misc)
    - [Show only some years](#show-only-some-years)
    - [Create a word cloud](#create-a-word-cloud)
        - [Save the word cloud as PNG](#save-the-word-cloud-as-png)
- [Limitations](#limitations)
- [Other similar tools](#other-similar-tools)

<!-- markdown-toc end -->

# Introduction

**Raven-visualizer** is an easy-to-use research literature visualization tool, powered by AI and [NLP](https://en.wikipedia.org/wiki/Natural_language_processing). It is intended to help a scholar or subject matter expert to stay up to date as well as to learn new topics, by helping to narrow down which texts from a large dataset form the most important background for a given topic or problem.

- **Graphical user interface** (GUI). Easy to use.
- **Fully local**. Your data never leaves your workstation/laptop.
- **Usability-focused**. Modern animated GUI with immediate visual feedback on user actions. Most functions accessible from keyboard.
- **Semantic clustering**. Discover **vertically**: See how a field of science splits into topic groups. Explore similar papers.
- **Fragment search**. Discover **horizontally**: E.g. find studies (across all topics) where some particular method has been used.
- **Info panel**. Read the abstracts (if available) of the studies you discover, right there in the GUI.
- **Open source**. 2-clause BSD license.

*Fragment search* means that e.g. *"cat photo"* matches *"photocatalytic"*. This is the same kind of search provided by the Firefox address bar, or by the `helm-swoop` function in Emacs. Currently the search looks only in the title field of the data; this may change in the future.

To find a cluster by its number, search for `#` and the number: *#7* finds cluster 7's studies.

**Raven is NOT a search engine.** Rather, for its input, it uses research literature metadata (title, authors, year, abstract) for thousands of papers, as returned by a search engine, and plots that data in an interactive semantic visualization.

*Raven-visualizer* is fully operational, and under active development.

The basic functionality is complete, the codebase should be in a semi-maintainable state, and most bugs have been squashed. If you find a bug that is not listed in [TODO.md](../../TODO.md), please [open an issue](https://github.com/Technologicat/raven/issues).

We still plan to add important features later, such as filtering by time range to help discover trends, and abstractive AI summaries of a user-selected subset of data (based on the author-provided abstracts).

We believe that at the end of 2024, AI- and NLP-powered literature filtering tools are very much in the zeitgeist, and that demand for them is only rising. Thus, we release the version we have right now as a useful tool in its own right, but also as an appetizer for future developments to come.

<p align="center">
<img src="../../img/screenshot-help.png" alt="Screenshot of Raven's help card" width="800"/> <br/>
<i>The help card. Most functions are accessible from the keyboard.</i>
</p>

# Import

Raven uses the following workflow:

```
+-------+              +--------+             +---------+
|  any  | --convert--> | BibTeX | --import--> | dataset | --> interactive visualization
+-------+              +--------+             +---------+
```

where the `convert` step is optional; BibTeX, widely used in the engineering sciences, is considered the native input format of Raven.

The input does not strictly have to be research literature. Anything that can be defined to have `title`, `authors`, and `year` fields, and optionally an `abstract` field (where *abstract* is any kind of human-readable short text summary), can be used as input. That said, the titles are used for linguistic analysis, so having precise titles (as is common in scientific papers) is likely to produce a more accurate semantic map.

Note that even BibTeX data needs to be imported before it can be visualized.

The import step typically takes some time, so it is performed either offline (in the sense of a batch job) or in the background. All computationally expensive procedures, such as semantic embedding, clustering, keyword analysis, and training the dimension reduction for the dataset, are performed during import. Some of these, particularly the semantic embedding, support GPU acceleration.

The data is clustered automatically, and each cluster of data has its keywords automatically determined by an automated linguistic analysis. It is not possible to edit the clusters or keywords. If something is detected incorrectly, it is more in the spirit of Raven to improve the algorithms rather than hand-edit each dataset. Raven is intended to operate in an environment that has too much data, and where the data updates too quickly, for any kind of manual editing to be feasible at all.

The import step produces a **dataset file**, which can then be opened and explored in the GUI.

You can import BibTeX files into dataset files in Raven's GUI, as well as from the command line. How to do this is described in more detail below.

## What the semantic map is

The technology is roughly explained by the following figures.

<p align="center">
<img src="../../img/embedding_space_ai.png" alt="Semantic embedding maps text into high-dimensional vectors." height="200"/> <br/>
<i>The semantic embedding model is a pretrained AI component that transforms text into high-dimensional vectors (default: Snowflake/snowflake-arctic-embed-l, d = 1024). Normalization brings the vectors onto a d-1 dimensional hypersphere surface. Schematic illustrations shown in 3 dimensions. (a) A concept (here "hot") and its opposite ("cold") map to opposite directions. Other, unrelated concepts ("cat", "democracy") map to orthogonal directions. (b) Concepts that are semantically near each other (e.g. "physics" is a field of "science") map in directions near to each other. For any chosen pair of concepts, semantic similarity can be measured via the <a href="https://en.wikipedia.org/wiki/Cosine_similarity">cosine similarity</a> of the embedding vectors.</i>
</p>

<p align="center">
<img src="../../img/raven-data-processing-pipeline-ai.png" alt="Raven-visualizer's data processing pipeline." height="400"/> <br/>
<i>Raven-visualizer's data processing pipeline. (a) Semantic embedding onto the high-dimensional hypersphere. (b) <a href="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html">HDBSCAN</a> in the high-dimensional space detects initial clusters, whose representatives train the dimension reduction; this step is now used only with UMAP, as t-SNE (the default) is fitted to the whole dataset. (c) The dataset is dimension-reduced into 2D with <a href="https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding">t-SNE</a>. (d) HDBSCAN in 2D produces the final clusters, shown on the <b>semantic map</b> in Raven-visualizer. This is the default; see <a href="#choose-how-clusters-are-found">Choose how clusters are found</a> for the others.</i>
</p>

## In the GUI

To import one or more BibTeX databases into a dataset file, click on the *Import BibTeX files* button, or press Ctrl+I. Doing so opens the following **BibTeX import window**:

<p align="center">
<img src="../../img/screenshot-import-bibtex.png" alt="Screenshot of Raven's BibTeX importer" width="630"/> <br/>
<i>The importer. This functionality converts BibTeX files into a Raven dataset.</i>
</p>

Importing several *input BibTeX files* at once combines the data from all of them into the same *output dataset file*.

Note this window is **not** modal, so you can continue working with the app while the window is open, and pressing Esc will not close it.

Pressing the Ctrl+I hotkey again closes the window.

### Save imported dataset as

Click on the hard disk icon (next to the heading "*Output dataset file*") in the *BibTeX import window*, or press Ctrl+S while the *BibTeX import window* is open. A **save-as dialog** opens, for choosing the filename of the dataset to be created. It is the file dialog every Raven app shares, described in [its own manual](../vendor/file_dialog/file-dialog-manual.md).

Double-clicking a directory in the list goes into that directory. Double-clicking the ".." directory goes one level up.

The buttons at the top of the dialog refresh the view of the current directory, and jump back to the default directory, respectively.

The list can be sorted by clicking on the column headers. The date shown is the mtime (modification time).

The save-as filename field can be focused by pressing Ctrl+F. The field doubles as a search filter, so you can see what existing files in the current directory have names similar to the one you are saving.

The file extension (`.pickle`) is added automatically to the filename you specify. In future versions of Raven, the file extension will likely change, once we move to a more portable data format.

You can also pre-populate the filename by clicking a file in the list. This can be useful if you want to overwrite a file, or if you are saving a series of related files (`dataset1.pickle`, `dataset2.pickle`, ...).

If a file would be overwritten, the OK button flashes red, and the dialog asks to press it again (before the flash ends) to confirm.

Pressing Enter is the same as clicking the OK button. To overwrite a file, press Enter again (before the flash ends).

Pressing Esc cancels the save-as dialog.

**:exclamation: Navigating directories in the save-as dialog currently requires using the mouse. This is a known issue. :exclamation:**

### Select input BibTeX files

To select input files, click the folder icon (next to the heading "*Input BibTeX files*"), or press Ctrl+O while the *BibTeX import window* is open. A **file picker dialog** opens, for choosing the input BibTeX files. It is the file dialog every Raven app shares, described in [its own manual](../vendor/file_dialog/file-dialog-manual.md).

The file picker works similarly to the save-as dialog, but with a **Search files** field replacing the save-as filename field.

You can focus the *Search files* field by pressing Ctrl+F. The search filters the view live, as you type. All files matching the current search can be accepted by pressing Enter, or by clicking the OK button.

So for example, in the situation shown in the screenshot, to open `savedrecs-3.bib`, you can press Ctrl+F, type "-3" (so that only this one file matches the search filter), and press Enter.

On the other hand, if you want to accept *all* files whose name contains `savedrecs`, you can press Ctrl+F, type "savedrecs", and press Enter.

You can also hold down Ctrl and click files in the list to select multiple input files.

Accept the selection (whether one or more files) by clicking OK, or by pressing Enter. Accepting multiple files will import them all into the same dataset.

If you need just one input file, you can also double-click the file in the list to accept that one file.

Pressing Esc cancels the file picker.

**:exclamation: With the exception of the search functionality, the file picker currently requires using the mouse. This is a known issue. :exclamation:**

### Choose how clusters are found

The *Clustering* selector in the *BibTeX import window* chooses how this import finds the clusters the map shows. It starts at `clustering_method` in [`raven.visualizer.config`](config.py).

- **HDBSCAN on the 2D map**, the default: clusters are the dense regions of the map. Records between them go to *Misc*, and on some datasets that is many of them.
- **Agglomerative, in the embedding space**: records are grouped by meaning in the full embedding space, so the clusters do not depend on the map. This places more of the dataset in clusters. A cluster smaller than 5 records is shown as *Misc*.
  - Where the grouping stops is set by `clustering_n_clusters`, `clustering_min_cluster_size` and `clustering_min_mean_cluster_size` in [`raven.visualizer.config`](config.py). For a small dataset the number of clusters is lowered, so that clusters hold at least 10 records on average.
  - The map still positions the points, so a cluster can be spread over more than one place on it.
- **LLM themes, for a small corpus**: one call to the LLM reads every title and abstract and groups the records into themes, which it names. The names are the cluster keywords.
  - The map is a layout of the themes, one circle each, and not a semantic map: where a circle sits means nothing.
  - A record may belong under two themes; its point is drawn under the one it belongs to most.
  - It needs an LLM backend, and a model whose context window holds the whole dataset. A dataset too large stops the import before anything is asked, with an error suggesting *Agglomerative*. With no backend, the import window uses *Agglomerative* instead and says so; `raven-importer` stops, unless given `--llm optional`.

The *Aim* field says what the collection is for, and is given to the LLM as context. It is used only with *LLM themes*. Leave it empty to let the material speak for itself.

### Start the import

To start the import, click the play icon in the *BibTeX import window*, or press Ctrl+Enter while the *BibTeX import window* is open. A progress bar will appear. While the import is running, a brief status message at the bottom of the *BibTeX import window* will indicate what the importer is currently doing. More detailed status is printed into the terminal window from which you started Raven.

The import process runs in the background, so you can continue working while your new dataset is being imported.

**:exclamation: Some tools used internally by the importer have no way to report on their ongoing progress. It is normal for the progress bar to seem stuck for several minutes, particularly while the importer is training or applying the dimension reduction. :exclamation:**

**:exclamation: The BibTeX import process may take a very long time, from several minutes to hours, and how the importer has been configured in [`raven.visualizer.config`](config.py). :exclamation:**

## From the command line

You can also run BibTeX imports without opening the Raven GUI (e.g. on a headless server). Raven provides a command-line tool for this task. It uses the exact same mechanism as the GUI importer; only the user interface is different.

To import one or more BibTeX files into a dataset file named `mydata.pickle`:

```bash
$(pdm venv activate)
raven-importer mydata.pickle file1.bib file2.bib ...
```

Status messages are printed into the terminal window.

To choose how clusters are found for this import, overriding the configuration, use `--clustering hdbscan_2d`, `--clustering agglomerative` or `--clustering llm`; with the last, `--aim "what the collection is for"` or `--aim-file aim.txt` gives the LLM its context. See [Choose how clusters are found](#choose-how-clusters-are-found).

When the configuration asks for LLM cluster keywords or summaries and no LLM backend answers, `raven-importer` stops before it starts, where the GUI importer goes ahead without them. `--llm optional` makes it go ahead too, with phrase keywords and no summaries, and say so at the end; `--no-llm` leaves those stages out for this run without trying a backend.

**:exclamation: The BibTeX import process may take a very long time, from several minutes to hours, depending on how much data you have, and how the importer has been configured in [`raven.visualizer.config`](config.py). :exclamation:**

## What the importer does to your records

A database export is not tidy, and the importer makes a few decisions about it on the way in. They are worth
knowing because they change what you see in the word cloud and in the info panel.

**An incomplete record is imported, not dropped.** A missing title, author or year becomes
`[Title not specified]`, `[Author not specified]` or `[Year not specified]`, and the import log names every
one. Whole conference proceedings arrive without authors, and losing those records would lose the abstracts
that were the part worth reading.

- **The placeholders are shown, never analyzed.** They are Raven's words rather than the record's, and the
  same words on every such record — so feeding them to the keyword extractor and the semantic vector would
  gather those records into a cluster whose members share nothing but a field their database omitted. Both
  stages read the abstract alone instead. (Authors and year never reached those stages anyway.)
- **A record with neither a title nor an abstract is skipped**, with a warning naming it: there is nothing
  to read.
- **A re-export writes the record's own author field back out**, or nothing where it had none — never the
  placeholder, which would put Raven's words into somebody's bibliography as though a database had said
  them.

**A record repeating an earlier record's key is kept**, read under a new key (`key-2`), and the import log says so. Records sharing a key are often the same paper, though, and the importer does not merge anything.

- **When combining BibTeX files that may hold the same papers, deduplicate them first** and import the result: `raven-deduplicate -o combined.bib first.bib second.bib`. Several files go to it at once, with nothing to concatenate. It matches papers by DOI and title, never by key: a paper present in both files is merged into one record whatever its keys, and two different papers that happened to share a key both stay, under distinct keys. See [`raven-deduplicate`](../papers/README.md#raven-deduplicate--merging-a-multi-database-search).
- **This matters across files too**, where the importer sees no clash at all: each file is read on its own, so the same paper in two files is imported twice and shows twice on the map.

**Keywords count nouns and proper nouns, and leave the verbs out.** A topic keyword is nearly always a noun,
while the verbs of academic prose — *provide*, *improve*, *develop*, *investigate*, *propose* — describe what
a paper *does*, are the same in every field, and crowd the head of the frequency list without saying
anything. This is also why *learning* and *learn* are counted separately, and should be: as a noun it is the
topic, as a verb it is prose. Affects the word cloud's word mode and the per-entry keywords, for datasets
imported from now on. `nlptools.count_frequencies` takes `accepted_pos=None` for the
older, wider behaviour.

**A publisher's rights notice is stripped off an abstract.** `© 2022 IEEE.` and *This article is distributed
under the terms of the Creative Commons Attribution 4.0 License* stop being treated as part of what a paper
says — which is where publisher names in your word cloud were coming from.

- **A paper *about* copyright keeps every word**, which is the hard case: the phrases a notice is built from
  are also things an abstract on open licensing says. So the copyright sign is trusted on sight, while
  wording that is ordinary English (*All rights reserved*, *copyright held by*, a licence-grant clause)
  counts only where it *opens* a sentence — which is what appended boilerplate does and a clause inside an
  argument does not. A bare *copyright* is never a match, and only the tail of an abstract is examined.
- `publisher_stopwords` in [`raven.visualizer.config`](config.py) still exists and still works; it now has
  much less to do.

**Cluster keywords are noun phrases** (`clusters_keyword_method = "phrases"`, the default): *palm oil mill
effluent*, *proton exchange membrane*, *self-regulated learning*, where single words would give *palm*, *mill*
and *effluent*.

- **A label has two parts.** Its first phrase is the one that best says what the cluster is about. The other
  five say how it differs from the other clusters: phrases particular to the cluster, and close in meaning to
  it. So a phrase found in many clusters is never one of those five, though it can come first: in a corpus
  about generative AI, *generative AI* is what several clusters are mostly about, and the five after it say
  which part of the subject each one is.
- **An abbreviation and its long form count as one keyword**, so *SRL* and *self-regulated learning* do not
  take two places in a label. It is shown as written where the corpus mostly uses it unexplained (*ChatGPT*),
  and spelled out where papers define it wherever they use it (*FWP* becomes *fast weight programmer*).
- **So do spellings that differ only by a hyphen or a space**: *chain-of-thought* and *chain of thought* are
  one keyword, as are *bit-width* and *bitwidth*, spelled as most of the papers spell it.
- **Keywords are capitalized as the papers capitalize them**, away from the start of a sentence: *Solomonoff
  induction*, *LoRA*, and otherwise lowercase. LLM keywords get the same treatment, which undoes the title case
  a model applies at random while keeping the proper names it spells right.
- **The dataset keeps a longer list per cluster**, 30 phrases by default, each with the share of the cluster's
  records containing it.
- **Phrases quote the papers, and an LLM can abstract.** A phrase label is made of words the papers use, so
  it never names a topic the cluster does not have, but it cannot name what the papers share when each words
  it differently. An LLM (`clusters_keyword_method = "llm"`) can, and now and then names a theme the cluster
  does not have. Phrases need no LLM backend, and give the same labels on every import.
- **`"hybrid"` takes each for the part it is good at**: the LLM's first two keywords name the theme, and the
  phrases that add something new follow, in the papers' own words. It needs an LLM backend, as `"llm"` does.
  `clusters_keyword_hybrid_llm_count` sets how many of the LLM's keywords lead.
- **The LLM is shown the cluster's phrases as evidence**, with the share of the cluster's papers using each,
  beside the papers its prompt holds. For a large cluster those are a sample, where the phrases count every
  paper. Its keywords stay its own. `clusters_llm_keyword_evidence`, on by default.
- The settings are the `clusters_keyword_*` ones in [`raven.visualizer.config`](config.py). The method needs
  no LLM, and is what `"llm"` and `"hybrid"` fall back to when no backend answers. It replaces `"frequencies"`; a
  configuration still naming that is refused, with a message saying what to write instead.

**A cluster shows its themes and its terms**, where the LLM keyworded the dataset along with the phrases
(`"hybrid"`, or `"llm"` with the evidence on). Its header in the info panel and in the plot tooltip has two lines:
**Themes**, the LLM's keywords, which name what the cluster's studies share, and **Terms**, the phrases, which
quote what they say. The two catch different things, the themes the ideas the studies word differently, the terms
the names they use: *double descent* is a theme, *ImageNet* a term.

- **A theme the LLM gave most clusters is left out**, as is a term repeating a theme.
- **A tooltip over more than two clusters shows the themes alone**, to leave room for the titles.

**LLM cluster keywords are made comparable across clusters**, if you have set `clusters_keyword_method =
"llm"`. Each cluster is keyworded on its own, so one concept can come back under several spellings — an
acronym in one cluster, its expansion in another — and two clusters sharing a topic then look as if they do
not. A second pass over the whole vocabulary folds the variants together once every cluster has been seen.

- **The model is asked for a mapping, not for a rewritten list, so the result can be checked.** A
  replacement is applied only when it is itself one of the keywords the first pass extracted, so an invented
  or rephrased term cannot reach the dataset — it is dropped instead of trusted.
- The prompt is `clusters_llm_keyword_canonicalization_prompt`, and the log names every replacement applied.

## Good to know

The BibTeX importer caches its intermediate data per input file, in a folder named after it beside it (`file1/` for `file1.bib`), so you can include `file1.bib` into multiple different dataset files, and the expensive computations specific to `file1.bib` will only happen once, unless `file1.bib` itself changes. The caching mechanism checks the timestamps; when `file1.bib` is processed, computations are re-done if `file1.bib` has changed after the cache was last updated.

**The LLM's answers are kept too**, in `file1/llm/`, so an import stopped part way and started again asks the LLM only what it has not asked yet.
- **Summaries are kept per file**, so a file used in several datasets is summarized once.
- **Cluster keywords, from a dataset imported from several files, go in the first file's folder**: a cluster draws on all of them, so none of them owns it.

An answer is filed by its whole prompt, so anything that changes the prompt asks again, and so does another model.

With t-SNE, the default (`vis_method`), the 2D semantic map is fitted on every record. With UMAP it is fitted on a sample of representative records, a few from each of the clusters found in at most 10k records picked at random, and the rest of the records are placed by that fit.

Currently, it is not possible to add new data into an existing Raven-visualizer dataset (to overlay new data on an existing, already trained dimension reduction). This is currently a major usability drawback, particularly for the use case of following ongoing research trends, so this will likely change in the future.


## Importing from other formats

First convert your data into BibTeX format, then import the BibTeX data into a dataset as explained above.

Available converters are described in more detail below.

We plan to add more converters in the future.


### arXiv

Useful especially for AI/CS topics.

**:exclamation: Raven is third-party software, NOT affiliated with arXiv. The relevant command-line tools have `arxiv` in the command name only for discoverability reasons. :exclamation:**

#### Converting a list of arXiv IDs into a BibTeX file

We provide `raven-arxiv2bib`, which pulls the metadata from arXiv and writes a BibTeX file. For a short help message, run `raven-arxiv2bib -h`.

Usage:

```bash
$(pdm venv activate)
raven-arxiv2bib arxiv_ids.txt -o arxiv_papers.bib
```

where `arxiv_ids.txt` is a text file containing arXiv identifiers, one per line. Identifiers can also be given directly as arguments, or piped in on stdin.

This gives you a BibTeX bibliography (`arxiv_papers.bib`) that can be imported into *Raven-visualizer*.

Identifiers are sent in batches, so a list of hundreds costs a handful of requests rather than one per paper, and the three seconds arXiv asks between requests is paid once per batch. Identifiers arXiv returns nothing for are reported at the end rather than aborting the run.

The version arXiv answered with is recorded, whether or not you asked for one: request the bare `2410.07866` — which means "whatever is current" — and the entry says `2410.07866v5`. So the bibliography states which revision it describes. Pass `--strip-versions` if you want a bibliography for *citing* papers rather than for tracking a collection.

**Earlier versions of this document recommended the external `arxiv2bib` package** and described a workaround for an HTTP 414 (URI too long) error it hit on long identifier lists. Neither is needed now: `raven-arxiv2bib` replaces it, is rate-limited like Raven's other arXiv tools, and batches its requests so the 414 cannot arise. If you installed the external tool on this advice, it is no longer a Raven dependency.

#### Extracting arXiv IDs from PDF filenames

In case you have a directory full of PDFs downloaded from arXiv, with the identifier somewhere in the filename, we provide `raven-arxiv2id`, which extracts arXiv identifiers from filenames.

Only unique identifiers will be returned; where several versions of the same paper are present, only the newest is kept. For a short help message, run `raven-arxiv2id -h`.

Usage:

```bash
$(pdm venv activate)
raven-arxiv2id >arxiv_ids.txt  # run this in a directory that has arXiv PDF files
```

Then pipe that into `raven-arxiv2bib`, in one command or two:

```bash
raven-arxiv2id -i ~/papers | raven-arxiv2bib -o arxiv_papers.bib
```

and import the resulting `.bib` file into *Raven-visualizer*.

##### Refreshing a collection when papers get new versions

Preprints get revised, sometimes years later, and a collection assembled over time drifts out of date without saying so. `--strip-versions` is what refreshes it:

```bash
raven-arxiv2id -i ~/papers --strip-versions > ids.txt
raven-arxiv-download -o ~/papers --save-bib arxiv_papers.bib $(cat ids.txt)
```

The mechanism is arXiv's own: an identifier *with* a version means that version, and one *without* means whatever is current. So dropping the suffix is exactly the request "give me the latest", and both tools honour it.

`--save-bib` writes the bibliography from metadata the download already fetched in order to name the files, so it costs no extra requests and no extra waiting — where running `raven-arxiv2bib` over the same identifiers afterwards would ask arXiv for all of it a second time. Papers already present in the output directory are included in the `.bib` too: it describes the set you asked for, not just the part that had changed.

Note that the old versions stay on disk, and that a refreshed paper's *filename* changes with its version — so if you have measured anything against this collection by filename, those references need regenerating alongside the refresh.

#### Auto-downloading arXiv fulltexts by IDs

If you have a list of arXiv identifiers and you want to download the corresponding fulltexts from arXiv, we provide `raven-arxiv-download`.

This tool takes a list of arXiv IDs on the command line, and downloads and names the corresponding PDFs automatically. For a short help message, run `raven-arxiv-download -h`.

If you have a file of arXiv IDs, one per line (as above), then to download the fulltexts:

```bash
xargs -a arxiv_ids.txt raven-arxiv-download
```

This will save the PDFs into the current working directory. Use the `-o some_output_dir` option to customize the output path (which can be relative or absolute).

If an ID specifies a version, that version of the paper is downloaded; otherwise the latest version is downloaded. Each unique PDF file is downloaded only once.

The PDF files are named automatically using the metadata from the arXiv API. Output filename format for papers with 1, 2, and ≥ 3 authors are:

```
Author (2024) - Paper Title - yymm.xxxxxvx.pdf
Author and Coauthor (2024) - Another Paper Title - yymm.xxxxxvx.pdf
Author and Coauthor et al. (2024, revised 2025) - Yet Another Paper Title - yymm.xxxxxvx.pdf
```

In the filename, *yymm.xxxxxvx* is the arXiv ID, including the version. In old-format (pre-2007) IDs (e.g. `cond-mat/0207270`, `math/0501001v2`), in the filename, the "/" is replaced by "_".

The version included in the filename is always automatically determined from the API metadata, regardless of whether a version was specified for that paper on the command line.


### WOS (Web of Science)

Useful for the engineering sciences. Export plain text from Web of Science, convert it with `raven-wos2bib`, and import the resulting `.bib` here.

See [`raven-wos2bib`](../papers/README.md#raven-wos2bib--web-of-science) in the paper tools manual for how to run it and what it maps.


### PDF (human-readable abstracts)

Abstract submissions to scientific conferences sometimes arrive as free-form, human-readable PDF files. If you are a conference organizer who would like to semantically visualize the set of abstracts sent to you, `raven-pdf2bib` converts them into BibTeX you can import here. It needs an LLM.

**See [`raven-pdf2bib`](../papers/README.md#raven-pdf2bib--conference-abstracts-as-pdfs) in the paper tools manual** for what it expects of a PDF, how to run a directory of them, how it works, why flagged entries are worth checking, and which model to use.


# Visualize

First, if the `raven-visualizer` app is not yet running, start it:

```bash
$(pdm venv activate)  # see Installation below
raven-visualizer
```

**:exclamation: For details on how to use the app (including a list of hotkeys), see the built-in Help card. To show the help, click the "?" button in the toolbar, or press F1. :exclamation:**

## Load a dataset file in the GUI

To load your dataset file, click on the *Open dataset* button in the toolbar, or press Ctrl+O, which opens the [file dialog](../vendor/file_dialog/file-dialog-manual.md) for picking it.

The *Open dataset* dialog is a file picker, which works similarly to the file picker in the *BibTeX import window*.

The only difference is that here multi-select mode is not available, because only one dataset can be opened at a time. Thus, Ctrl+click is not available.

If you use the search feature (Ctrl+F) to open a file by typing a part of its name, and the search has exactly one match in the current directory (i.e. when only one file is shown in the list, not counting the ".."), that file can then be opened by pressing Enter.

**:exclamation: With the exception of the search functionality, the file picker currently requires using the mouse. This is a known issue. :exclamation:**

## Load a dataset file from the command line, when starting the app

Like many GUI apps, Raven also accepts a dataset file from the command line, when the app starts:

```bash
raven-visualizer mydata.pickle
```


## Put the info panel on the right

By default the info panel is on the left, the plot on the right, and the toolbar between them. Press **Ctrl+Shift+P**, or click the ↔ button in the toolbar, to swap the sides; the toolbar stays in the middle. Handy for pair work at one screen.

The app remembers the side when it closes, and starts with it the next time. To choose for one session, give it on the command line, which wins over what was remembered:

```bash
raven-visualizer --panel-side right mydata.pickle
```


## How tight the clusters are, and showing loose ones as Misc

The row above the plot gives the dataset's **cluster purity**, as in *"Cluster purity: 58% on average"*. For a record in a cluster, purity is the share of its closest neighbours in meaning (in the embedding space) that are in the same cluster; the row gives the average over every record in a cluster. Each cluster's header, in the info panel and in the plot's tooltip, gives that cluster's own, as *"purity 62%"*.

- **High purity** means the cluster is a tight topic: its records' nearest neighbours are each other.
- **Low purity** means the cluster overlaps its neighbours: its records are related, but each has closer neighbours elsewhere. Read such a cluster as a region of the map rather than as a topic.
- **Compare clusters within one dataset, not across datasets.** A collection on a single topic has low purity throughout, because everything in it is related; a broad one has high purity. There is no general threshold for a "good" cluster.

The **Misc cutoff** slider, at the right end of the row, shows every cluster below the chosen purity as *Misc*: on the map, in the info panel and in the tooltip. It works live, and your selection is kept as clusters come and go. At 0%, every cluster is shown as itself. Hover over the slider for a reminder of all this; Ctrl+click it to type a value.

The cutoff changes only what is shown. The dataset file is unchanged, and the slider starts at 0% whenever a dataset is loaded.

**A cluster can appear on the map in several separate pieces**, when the clusters were found in the embedding space (*Agglomerative*). Its purity says which of two things you are seeing:

- **High purity: the map has torn the cluster.** The records are each other's nearest neighbours in meaning, but the map cannot keep that everywhere. It flattens a space of a thousand dimensions onto a plane, which is like crumpling a globe flat: somewhere, neighbours end up apart. The cluster is real; its pieces belong together.
- **Low purity: the cluster is loose.** Its records are related to each other, but each one's closest neighbours are in other clusters, and the map places each record by its neighbours. The pieces are real structure: read them as parts of the neighbouring topics that this cluster also touches.

HDBSCAN finds its clusters on the map itself, so with that method each cluster is one dense region of the map.


## Gray out or hide Misc

The **Misc** selector, in the row above the plot, chooses how the records in *Misc* are shown: **shown** in colour, **grayed** out beneath everything else, or **hidden**. **F4** steps to the next choice, and **Shift+F4** back; or **Ctrl+M** puts the keyboard on the selector, where the arrow keys, **Home** and **End** choose, and **Esc** leaves. Clusters that the purity cutoff shows as *Misc* go with it, so the two together let you set aside whatever you take to be noise.

Graying out keeps the shape of the whole collection in view, while hiding leaves only the rest.

**Only records in colour can be hovered and selected.** The grayed-out and hidden ones stay out of the hover tooltip, the selection brush, *select visible* and *select this cluster*. Graying or hiding records that are selected takes them out of the selection, and showing them again brings them back; undo and redo keep them too.

**Changing the selection in between ends that**: a new selection is built from what is on screen, so if you brush in more records while some are grayed out, say, the grayed ones stay out of the selection when you show them again.

While records are grayed out, the **Grayed points selectable** checkbox, under the purity line, makes them hoverable and selectable as well, for asking why one of them is grayed. It applies to whatever is grayed, by the Misc selector or by the [year filter](#show-only-some-years).

**Search still finds every match**: one among hidden records is counted but not drawn, as in *"[7 matches, 3 hidden]"*, and one among grayed records is highlighted like any other. Selecting the matches takes only those that can be selected.

**Whenever records are grayed out or hidden, the app says so**: a pill beside the purity line reads, for example, *"1,422 of 2,032 in focus · Show all"* when records are grayed, or *"Showing 1,422 of 2,032"* when some are hidden. The plot's title carries the same count. **Click the pill to show everything again**: it resets the Misc selector, the purity cutoff and the years at once. With everything shown, neither the pill nor the count appears.

Like the purity cutoff, this changes only what is shown, and starts from showing everything whenever a dataset is loaded.

A dataset imported before Raven measured purity shows neither the row nor the cluster purities. Importing it again adds them; the embeddings are cached beside the BibTeX file, so that is quick.


## Show only some years

The **Years** fields, in the second row above the plot, show the records from a range of years in colour, both ends included. They start at the dataset's own span, which shows everything; narrow them with the − and + buttons, or type a year. Setting one end past the other moves the other along.

The selector beside them chooses how the records outside the range are shown: **grayed** out beneath the rest, or **hidden**. A record with no year counts as outside, so narrowing the range sets it aside too.

**From the keyboard**, **Ctrl+E** (for *era*) puts the keyboard on that selector. There, **Left** and **Right** move the whole range a year earlier or later, keeping its width, which is a quick way to watch a topic come and go. **Shift+Left** and **Shift+Right** move only the first year, and **Ctrl+Left** and **Ctrl+Right** only the last. **Up**, **Down**, **Home** and **End** choose how the other years are shown, and **Esc** leaves. **F5** switches that choice from anywhere.

**The year filter and the Misc selector work together**: a record is in colour only if both show it, hidden if either hides it, and grayed otherwise. So you can, say, hide the noise and gray the other years at once. Everything else is as for Misc above: what can be selected, how search counts, the pill, and *Show all*.

A dataset whose records carry no year has nothing to filter by, so the year fields do not appear.


## Create a word cloud

Raven can make a word cloud of the studies in the current selection, in one of two modes:

- **Phrases** (the default): the noun phrases of the selected studies, each sized by how many of them use it. Phrases are counted the way the cluster keywords are: an abbreviation and its long form are one phrase, and a longer phrase stands in for a shorter one it contains where it is found in at least half as many studies (*large language model* for *language model*).
- **Words**: their single words, each sized by how often it occurs in the selection.

The two radio buttons at the bottom of the window choose the mode, and **Shift+F10** switches it from anywhere; the choice is remembered between sessions. A dataset imported before phrase keywords existed has words only, and the radio buttons are then grayed out; import it again to get phrases.

<p align="center">
<img src="../../img/screenshot-wordcloud.png" alt="Screenshot of Raven's wordcloud window" width="600"/> <br/>
<i>Word cloud window.</i>
</p>

The word cloud window hotkey (F10) toggles the window. Note this window is **not** modal, so you can continue working with the app while the window is open, and pressing Esc will not close it.

If the word cloud window is open, it updates automatically whenever the selection changes. Just like in the info panel, the old content remains in the window until the new rendering finishes.

When the word cloud window is opened, Raven checks whether the selection has changed since the last word cloud was rendered. If there are no changes, the latest already rendered word cloud is re-shown.

The rendering algorithm allocates regions and colors randomly, so even re-rendering with the same data (e.g. in another session later), you will get a different-looking result each time.

The word cloud renderer is Python-based, so it can be rather slow for large selections containing very many data points. The render runs in the background, so you can continue working (as long as you don't change the selection) while the word cloud is being rendered.

### Save the word cloud as PNG

Click the "hard disk" button, or press Ctrl+S while the *word cloud window* is open. A *save-as dialog* opens, offering to save the word cloud image as PNG.

This dialog works similarly to the dataset save-as dialog in the *BibTeX import window*.

The file extension (`.png`) is added automatically to the filename you specify.


## Keyboard reference

Every hotkey in one place. What each one *does* is explained in the sections above and on the built-in help
card (**F1**); this says what exists, and **when each key is live**, which is the part that is hard to work
out by trying things.

Two terms the table leans on. The **current item** is the topmost entry *fully* visible in the info panel,
marked with a pulsating blue dot — most of the per-item keys act on it. The **selection** is the set of
entries you have gathered, which the plotter highlights. Several keys offer the usual set operations on it,
and the modifier is the same every time: bare replaces the selection, Shift adds to it, Ctrl subtracts from
it, and Ctrl+Shift intersects with it. The tables below spell each combination out.

### Always live

| Key | Action |
|---|---|
| `F1` | Help card |
| `F11` | Toggle fullscreen |

### Layout

| Key | Action |
|---|---|
| `Ctrl+Shift+P` | Move the info panel to the other side of the window |

### What the plot shows

| Key | Action |
|---|---|
| `F4` / `Shift+F4` | Show *Misc* the next / previous way: shown, grayed out, hidden |
| `Ctrl+M` | Put the keyboard on the *Misc* selector: then `Up`, `Down`, `Home` and `End` choose, and `Esc` leaves it |
| `F5` / `Shift+F5` | Show the records outside the years the other way: grayed out, hidden |
| `Ctrl+E` | Put the keyboard on the year selector: then `Up`, `Down`, `Home` and `End` choose, and `Esc` leaves it |
| `Left` / `Right` | With the year selector holding the keyboard: move the years back or forward, keeping the range's width |
| `Shift+Left` / `Shift+Right` | The same, but the first year only |
| `Ctrl+Left` / `Ctrl+Right` | The same, but the last year only |

### Datasets and search

| Key | Action |
|---|---|
| `Ctrl+O` | Open a dataset |
| `Ctrl+I` | Import BibTeX files — this is how a dataset is made |
| `Ctrl+F` | Put the caret in the search field |
| `Ctrl+Shift+F` | Clear the search — works whether or not you are typing in the field |
| `Enter` | Select the search matches and leave the field |
| `Shift+Enter` / `Ctrl+Enter` / `Ctrl+Shift+Enter` | ...adding to / subtracting from / intersecting with the selection |
| `Esc` | Cancel the edit and leave the field |
| `Tab` / `Shift+Tab` | Move the keyboard between the search field and the info panel, keeping what is typed |
| `F3` / `Shift+F3` | Scroll to the next / previous search match |

### Moving around the info panel

`Home` and `End` are live whenever the search field does not hold the caret, since the field uses them itself.
The rest work while typing too — the paging keys and the vertical arrows included, so the results of a search
can be read down without leaving the field.

| Key | Action |
|---|---|
| `Home` / `End` | Top / bottom |
| `Page Up` / `Page Down` | Page up / down |
| `Up arrow` / `Down arrow` | Scroll a little |
| `Ctrl+U` | To the start of the current cluster ("up") |
| `Ctrl+N` / `Ctrl+P` | To the next / previous cluster |
| `Ctrl+Home` | Reset the plotter's zoom |

### The current item, and the selection

| Key | Action |
|---|---|
| `F6` | Search for the current item, which highlights it in the plotter — `Shift+F6` selects only it, `Ctrl+F6` removes it from the selection |
| `F7` | Select the current cluster |
| `Shift+F7` / `Ctrl+F7` / `Ctrl+Shift+F7` | ...adding to / subtracting from / intersecting with the selection |
| `F9` | Select everything currently visible in the plotter |
| `Shift+F9` / `Ctrl+F9` / `Ctrl+Shift+F9` | ...adding to / subtracting from / intersecting with the selection |
| `Ctrl+Shift+C` | Copy the current item to the clipboard, as plain text for a web search |
| `Ctrl+Shift+Z` / `Ctrl+Shift+Y` | Undo / redo the last selection change |
| `F8` | Copy the report to the clipboard as plain text — `Shift+F8` for Markdown |
| `F10` | Toggle the word cloud window, built from the phrases or words of the selected items |
| `Shift+F10` | Switch the word cloud between phrases and words |

### Word cloud window

Live only while that window is open, which is what leaves `Ctrl+S` free everywhere else.

| Key | Action |
|---|---|
| `Ctrl+S` | Save the word cloud as a PNG |

### BibTeX importer window

Likewise live only while the importer window is open.

| Key | Action |
|---|---|
| `Ctrl+O` | Choose the input BibTeX files |
| `Ctrl+S` | Choose where to save the dataset |
| `Ctrl+Enter` | Start the import, or stop one in progress |

### Hidden debug keys

`Ctrl+Shift+` **M**, **R**, **T**, **L** — DPG's metrics window, item registry, font manager and style
editor. Mnemonic: *Mr. T Lite*.


# Limitations

- Scalability? Beta version tested up to 12k entries, but datasets can be 100k entries in size.
- Hardware requirements, especially GPU. Tested on a laptop with an NVIDIA RTX 3070 Ti mobile, 8 GB VRAM.
- Clustering in high-dimensional spaces is an open problem in data science. Semantic vectors have upwards of 1k dimensions. With the default clustering method, this causes many entries to be placed into a catch-all "*Misc*" cluster; see [Choose how clusters are found](#choose-how-clusters-are-found) for the alternatives. HDBSCAN's own hyperparameters may be dataset-dependent, and are not yet configurable.
- Dataset files are currently **not** portable across different Python versions.
- We attempt to provide keyboard access to GUI features whenever reasonably possible, but the plotter is currently one place where this is not reasonably possible. The *Open dataset* dialog is fully keyboard-operable: arrow keys walk the listing, Enter descends or picks, Tab completes the filename, and there are chords for sorting, filtering, the places panel and the path field. Press F1 inside it for the full list.
- As explained in the main README, configuration is currently fed in as a Python module, [`raven.visualizer.config`](config.py), which exists specifically as a configuration file.


# Other similar tools

[LitStudy](https://nlesc.github.io/litstudy/) is a Jupyter notebook with similar goals. Its analysis methods seem slightly different to what we use. Also, having existed since 2022, it has many more importers than we do at the moment.

[BERTopic](https://maartengr.github.io/BERTopic/index.html) is a library for *topic modeling*, to automatically extract topics from a large dataset of texts. BERTopic uses many of the same ideas and methods that *Raven-visualizer* uses to generate its semantic map (embed semantically, cluster with HDBSCAN, reduce the dimension), and the phrase keywords for the clusters borrow its class-based TF-IDF to rank a cluster's phrases. Raven was developed independently, before I became aware of BERTopic. While the library has existed since 2022, for some reason it didn't come up in my initial searches at the beginning of the Raven project. As of H2/2025, BERTopic has become very popular, and has been mentioned several times in various technically flavored AI news outlets. It seems that as of 2025, this kind of overview analysis of large text datasets is in the zeitgeist.
