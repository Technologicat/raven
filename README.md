<p align="center">
<img src="img/logo.png" alt="Logo of Raven" height="200"/> <br/>
</p>

-----

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [The Raven constellation](#the-raven-constellation)
- [Install & run](#install--run)
    - [From PyPI](#from-pypi)
    - [From source](#from-source)
        - [Install PDM in your Python environment](#install-pdm-in-your-python-environment)
        - [Install Raven via PDM](#install-raven-via-pdm)
            - [Install on an Intel Mac with MacOSX 10.x](#install-on-an-intel-mac-with-macosx-10x)
        - [Check that CUDA works (optional)](#check-that-cuda-works-optional)
        - [Activate the Raven venv (to run Raven commands such as `raven-visualizer`)](#activate-the-raven-venv-to-run-raven-commands-such-as-raven-visualizer)
        - [Activate GPU compute support (optional)](#activate-gpu-compute-support-optional)
        - [Exit from the Raven venv (optional)](#exit-from-the-raven-venv-optional)
- [Uninstall](#uninstall)
- [Technologies](#technologies)
- [License](#license)
- [Acknowledgements](#acknowledgements)

<!-- markdown-toc end -->

# The Raven constellation

As of 08/2025, *Raven* is now a constellation, no longer a single app. Until I properly update the documentation, here is a short overview:

- :white_check_mark: *Raven-visualizer*: **Research literature visualization tool**
  - **Documentation**: [User manual](raven/visualizer/README.md)
  - **Goal**: Take 10k+ studies, find the most relevant ones.
    - Status: Fully operational. Could still use more features; we plan to add some later.
  - **Features**: See [Raven-visualizer](#raven-visualizer) section below.
  - This was the original *Raven*.

- :white_check_mark: *Raven-server*: **Web API server for GPU-powered components: avatar, TTS, NLP**
  - **Documentation**: [Server manual](raven/server/README.md), [Avatar manual](raven/avatar/README.md)
  - **Goal**: Run all GPU processing in the server process, wherever it is on the local network.
    - Status: Fully operational. On the client side, `raven-importer` and the RAG subsystem of *Raven-librarian* have no server support yet.
  - **Features**:
    - AI-animated anime avatar, for PR stunts and for fun.
      - Represent your LLM as a custom, talking anime character with support for emotions. (The `classify` module can detect emotions from the generated text).
      - A test client / tech demo for the avatar is available as `raven-avatar-settings-editor`.
        - For this to start, `raven-server` must be running.
        - On the client side, set the server URL to connect to in [`raven/client/config.py`](raven/client/config.py). Default is to run both server and client on localhost.
      - To edit the emotion templates of the avatar, `raven-avatar-pose-editor`.
    - Speech synthesizer that the avatar **can lipsync to**.
    - Various [GPU-accelerated AI components for natural language processing](raven/server/README.md) so that all Raven apps can take advantage of their capabilities.
  - The server was originally developed as a continuation of the discontinued *SillyTavern-extras*, but has since been extended in various ways.
    - The `classify`, `embeddings`, `summarize`, and `websearch` modules are compatible with those modules of *SillyTavern-Extras*.
    - Additionally, the `tts` module provides an OpenAI compatible speech endpoint, which *SillyTavern* can use as its TTS.
    - It would be interesting to introduce `avatar` to replace the discontinued `talkinghead`, but at the moment, there are no development resources to write a JS client for the avatar. See [SillyTavern#4034](https://github.com/SillyTavern/SillyTavern/issues/4034) for details.
  - Python bindings for Raven-server are provided in [`raven/client/api.py`](raven/client/api.py). These double as documentation for how to call the web API endpoints.
    - The client API abstracts away the fact that you're calling a remote process - you just call regular Python functions.
  - The server can be configured in [`raven/server/config.py`](raven/server/config.py). Or make your own config file, and point to that using the `--config` command-line option when you start the server. See [`raven/server/config_lowvram.py`](raven/server/config_lowvram.py) for an example.

- :construction: *Raven-librarian*: **Scientific LLM frontend** (under development)
  - **Documentation**: under development
  - **Goal**: Efficiently interrogate a stack of 2k scientific papers. Talk with a local LLM for synthesis, clarifications, speculation, ...
    - Status: A command-line prototype `raven-minichat` is available.
      - We recommend having `raven-server` running; this allows the LLM to search the web.
      - A GUI application is planned, but not available yet.
  - **Features**:
    - Natively nonlinear branching chat history - think [Loom](https://github.com/cosmicoptima/loom) or [SillyTavern-Timelines](https://github.com/SillyTavern/SillyTavern-Timelines).
      - Chat messages are stored as nodes in a tree. A chat branch is just its HEAD pointer; the chat app follows the `parent` links to reconstruct the linear history for that branch.
      - The command-line prototype can create chat branches and switch between them, but not delete them. Complete control for chat branches will be in the GUI app.
    - RAG (retrieval-augmented generation) with hybrid (semantic + keyword) search.
      - Semantic backend: [ChromaDB](https://www.trychroma.com/) (with telemetry off, for maximum privacy).
      - Keyword backend: [bm25s](https://huggingface.co/blog/xhluca/bm25s), which implements the [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) ranking algorithm.
      - Results are combined with [reciprocal rank fusion](https://www.assembled.com/blog/better-rag-results-with-reciprocal-rank-fusion-and-hybrid-search).
    - Tool-calling (a.k.a. tool use)
      - Currently, a websearch tool is provided.
  - Uses [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) as the LLM backend through its OpenAI-compatible API.
    - We currently test our scaffolding with the Qwen series of LLMs. Recommended model: [Qwen3-30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507).


# Install & run

Raven is a traditional desktop app. It needs to be installed.

Currently, this takes the form of installing the app and dependencies into a venv (virtual environment). At least at this stage of development, app packaging into a single executable is not a priority.

Raven has been developed and tested on Linux Mint. It should work in any environment that has `bash` and `pdm`.

## From PyPI

**Coming soon**

## From source

Raven has the following requirements:

 - A Python environment for running the [PDM](https://pdm-project.org/en/latest/) installer. Linux OSs have one built-in; on other OSs it is possible to use tools such as [Conda](https://www.anaconda.com/docs/getting-started/miniconda/main) to install one.
 - An NVIDIA GPU for running AI models via CUDA. (This is subject to change in the future.)

### Install PDM in your Python environment

Raven uses [PDM](https://pdm-project.org/en/latest/) to manage its dependencies. This allows easy installation of the app and its dependencies into a venv (virtual environment) that is local to this one app, so that installing Raven will not break your other apps that use machine-learning libraries (which tend to be very version-sensitive).

If your Python environment does not have PDM, you will need to install it first:

```bash
python -m pip install pdm
```

Don't worry; it won't break `pip`, `poetry`, or other similar tools.

### Install Raven via PDM

Then, to install Raven, in a terminal that sees your Python environment, navigate to the Raven folder.

We will next initialize the new venv, installing the required Python version into it. This Python will be available for PDM venvs, and is independent of Python that PDM itself runs on.

Raven is currently developed against the minimum supported Python version, so we recommend to install that version, like this:

```bash
pdm python install --min
```

The venv will be installed in the `.venv` hidden subfolder of the Raven folder.

Then, install Raven's dependencies as follows. (If you are a seasoned pythonista, note that there is no `requirements.txt`; the dependency list lives in `pyproject.toml`.)

**Basic install without GPU compute support**:

```bash
pdm install
```

**Install with GPU compute support**:

:exclamation: **Help wanted!** Raven does not directly depend on CUDA, but only on PyTorch and on various AI libraries in the Python ecosystem. If you have an AMD system and would be willing to collaborate to get Raven working on it, [please chime in](https://github.com/Technologicat/raven/issues/1)! :exclamation:

This requires an NVIDIA GPU, the proprietary NVIDIA drivers, and CUDA. The GPU will be used for accelerating BibTeX imports.

```bash
pdm install --prod -G cuda
```

If you want to add GPU compute support later, you can run this install command on top of an already installed Raven.

Installing dependencies may take a long time (up to 15-30 minutes, depending on your internet connection), because `torch` and the NVIDIA packages are rather large (my `.venv` shows 11.1 GB in total).

Now the installation should be complete.

#### Install on an Intel Mac with MacOSX 10.x

Installing Raven may fail, if Torch cannot be installed.

On MacOSX, installing torch 2.3.0 or later requires an ARM64 processor and MacOSX 11.0 or later.

If you have an Intel Mac (x86_64) with MacOSX 10.x, to work around this, you can use Torch 2.2.x.

To do this, modify Raven's `pyproject.toml` in a text editor, so that the line

```
    "torch>=2.4.0",
```

becomes

```
    "torch>=2.2.0,<2.3.0",
```

Then run `pdm install` again.


### Check that CUDA works (optional)

If you want to use the optional GPU compute support, you will need an NVIDIA GPU and the proprietary NVIDIA drivers (which provide CUDA). How to install them depends on your OS.

Currently Raven uses GPU compute only in the preprocessor (BibTeX import), to accelerate the computation of semantic vectors and the NLP analysis. For large datasets, using a GPU can make these steps much faster.

**:exclamation: Currently Raven uses CUDA 12.x. Make sure your NVIDIA drivers support this version. :exclamation:**

Once you have the NVIDIA drivers, and you have installed Raven with GPU compute support, you can check if Raven detects your CUDA installation:

```bash
raven-check-cuda
```

This command will print some system info into the terminal, saying whether it found CUDA, and if it did, which device CUDA is running on. It will also check whether the `cupy` library loads successfully.

### Activate the Raven venv (to run Raven commands such as `raven-visualizer`)

**:exclamation: This is the (early) state of things as of August 2025. We aim to provide easier startup scripts in the future. :exclamation:**

In a terminal that sees your Python environment, navigate to the Raven folder.

Then, activate Raven's venv with the command:

```bash
$(pdm venv activate)
```

Note the Bash exec syntax `$(...)`; the command `pdm venv activate` just prints the actual internal activation command.

Whenever Raven's venv is active, you can use Raven commands. Most of the time you'll want `raven-visualizer`, which opens the GUI app.

### Activate GPU compute support (optional)

With the venv activated, and the terminal in the Raven folder, you can enable CUDA support by:

```bash
source env.sh
```

This sets up the library paths and `$PATH` so that Raven finds the CUDA libraries. This script is coded to look for them in Raven's `.venv` subfolder.

If you have multiple GPUs, you can use the `CUDA_VISIBLE_DEVICES` environment variable to set which GPU Raven should use. We provide an example script [`run-on-internal-gpu.sh`](run-on-internal-gpu.sh), meant for a laptop with a Thunderbolt eGPU (external GPU), when we want to force Raven to run on the *internal* GPU (useful e.g. if your eGPU is in use by a self-hosted LLM).

With the terminal still in the Raven folder, usage is:

```bash
source run-on-internal-gpu.sh
```

Then you can use Raven commands as usual.

### Exit from the Raven venv (optional)

If you want to exit from the Raven venv without exiting your terminal session, you can deactivate the venv like this:

```bash
deactivate
```

After this command completes, `python` again points to the Python in your Python environment (where e.g. PDM runs), **not** to Raven's app-local Python.

If you want to also exit your terminal session, you can just close the terminal window as usual; there is no need to deactivate the venv unless you want to continue working in the same terminal session.


# Uninstall

```bash
python -m pip uninstall raven-visualizer
```

Or just delete the venv, located in the `.venv` subfolder of the Raven folder.


# Technologies

Raven builds upon several AI, NLP, statistical, numerical and software engineering technologies:

- Semantic embedding
  - AI model: [snowflake-arctic](https://huggingface.co/Snowflake/snowflake-arctic-embed-l).
  - Engine for running embedding models: [sentence_transformers](https://sbert.net/).
- Low-level NLP analysis for keyword extraction: [spaCy](https://spacy.io/).
- High-dimensional clustering: [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/index.html).
- Dimension reduction: [OpenTSNE](https://opentsne.readthedocs.io/en/stable/).
- AI-powered PDF import
  - A large language model (LLM), such as:
    - For machines with at least 24 GB VRAM:
      - [Qwen3-30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507) (**recommended** as of 08/2025)
      - [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)
      - [Sky-T1 32B](https://huggingface.co/NovaSky-AI/Sky-T1-32B-Preview)
    - For machines with 8 GB VRAM (e.g. a laptop with an internal NVIDIA GPU):
      - [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) (**recommended** as of 08/2025; punches well above its size class)
      - [Deepseek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
      - [Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
  - LLM inference server; we recommend [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) (start it with the `--api` option to let Raven see it).
  - Communication with the LLM inference server: [sseclient-py](https://github.com/mpetazzoni/sseclient).
- File format support
  - BibTeX: [BibtexParser](https://bibtexparser.readthedocs.io/en/main/).
  - Web of Science: [wosfile](https://github.com/rafguns/wosfile).
- Graphical user interface: [DearPyGUI](https://github.com/hoffstadt/DearPyGui/).
  - "Open"/"Save as" dialog: [file_dialog](https://github.com/totallynotdrait/file_dialog), but customized for Raven, and some features added.
  - Markdown renderer: [DearPyGui-Markdown](https://github.com/IvanNazaruk/DearPyGui-Markdown).
  - Toolbutton icons: [Font Awesome](https://github.com/FortAwesome/Font-Awesome) v6.6.0.
  - Word cloud renderer: [word_cloud](https://github.com/amueller/word_cloud).

Note that installing Raven will auto-install dependencies into the same venv (virtual environment). This list is here just to provide a flavor of the kinds of parts needed to build a tool like this.


# License

[2-clause BSD](LICENSE.md).


# Acknowledgements

This work was financially supported by the [gH2ADDVA](https://www.jamk.fi/en/research-and-development/rdi-projects/adding-value-by-clean-hydrogen-production) (Adding Value by Clean Hydrogen production) project at JAMK, co-funded by the EU and the Regional Council of Central Finland.
