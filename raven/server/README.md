<p align="center">
<img src="../../img/logo.png" alt="Logo of Raven" height="200"/> <br/>
</p>

-----

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Introduction](#introduction)
- [Command-line options](#command-line-options)
- [Server modules](#server-modules)
    - [List of server modules](#list-of-server-modules)
- [Server configuration](#server-configuration)
    - [Low VRAM config (8 GB)](#low-vram-config-8-gb)
    - [Choosing which GPU to use (optional)](#choosing-which-gpu-to-use-optional)
- [SillyTavern compatibility](#sillytavern-compatibility)
    - [Raven-server TTS for SillyTavern](#raven-server-tts-for-sillytavern)
- [Python bindings (easy client API)](#python-bindings-easy-client-api)
- [Web API endpoints](#web-api-endpoints)

<!-- markdown-toc end -->

# Introduction

*Raven-server* is a web API server that hosts local, specialized AI models on the GPU:

- **Avatar**: AI-animated custom anime character for your LLM. This is the server side of [*Raven-avatar*](../avatar/README.md).
- **Speech synthesizer (TTS)**: built-in, locally hosted [Kokoro-82M](https://github.com/hexgrad/kokoro).
- **Natural language processing (NLP)**: various components for GPU-accelerated natural language analysis and processing.

Although the default is to run both the server and the client apps on localhost, the server can run anywhere on the local network. This allows a separate machine with a powerful GPU to host the server for one or more clients on the local network.

Most of the server functions are stateless; the only exception is *Raven-avatar*, which gives you a session ID.

For the speech synthesizer, we provide two web API endpoints: an OpenAI compatible one, and a custom one. The custom endpoint provides word timestamps and per-word phoneme data, which is needed for lipsyncing the avatar. The actual lipsync driver lives on the client side, in the [Python bindings](#python-bindings-easy-client-api), because the speech audio playback is done on the client side, too.

Historically, *Raven-server* began as a continuation of the discontinued *SillyTavern-extras*. One important reason was to keep the avatar technology alive; it was a promising, unique experiment that no other project seems to have followed up on. But also, a web API server for various specialized NLP functionality happened to be exactly what *Raven-visualizer* and the upcoming *Raven-librarian* needed. The server has since been extended in various ways: the avatar has several new features, the built-in TTS is new, and several new NLP modules have been added.


# Command-line options

For a guaranteed-up-to-date list of available command-line options, run `raven-server --help`.

It will print a usage summary and exit, like this (output as of v0.2.3):

```
usage: Raven-server [-h] [-v] [--config some.python.module] [--port PORT] [--listen] [--secure] [--max-content-length MAX_CONTENT_LENGTH]

Server for specialized local AI models, based on the discontinued SillyTavern-extras

options:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  --config some.python.module
                        Python module containing the server config (default is 'raven.server.config')
  --port PORT           Specify the port on which the application is hosted (default is set in the server config module)
  --listen              Host the app on the local network (if not set, the server is visible to localhost only)
  --secure              Require an API key (will be auto-created first time, and printed to console each time on server startup)
  --max-content-length MAX_CONTENT_LENGTH
                        Set the max content length for the Flask app config.
```

The default port is 5100.

**Important difference to SillyTavern-extras**

In *Raven-server*, server modules are enabled/disabled and configured **in the server config file**, not on the command line.

Thus, to support environments with varying use cases, we only provide **one** command-line option for server module configuration: namely, `--config`, to load a different config file.


# Server modules

:exclamation: *To be able to use GPU, be sure to install the CUDA optional dependencies of Raven (see [installation in the main README](../../README.md#install-raven-via-pdm)).* :exclamation:

The server's functionality is split into *modules* that can be enabled or disabled individually. Disabling server modules you don't need can save VRAM as well as allow the server to start up faster.

Modules are enabled/disabled in the *server config* ([see below](#server-configuration)). The modules' compute device and dtype (data type), and any HuggingFace models they use, can also be specified there.

Devices and dtypes follow the PyTorch format.

Typical compute devices are `"cuda:0"` (for the first visible NVIDIA GPU) and `"cpu"`.

Typical dtypes are `"float16"` (commonly used on GPU) and `"float32"` (used on CPU).

Note that not all modules have a dtype setting, and one (`websearch`) doesn't even have a device setting, as it doesn't do any heavy computation. In such cases, the existence of blank config record means that the module is enabled.

## List of server modules

*Last updated for v0.2.3.*

*This server module list is maintained on a best-effort basis, but sometimes, recent changes may be missing. The ground truth is [`raven.server.app`](../server/app.py), especially the function `init_server_modules`.*

*If you are feeling lucky, you can also just skim the `from .modules import ...` statements at the beginning of the file.*

We provide the following server modules:

- `avatar`: The server side of [*Raven-avatar*](../avatar/README.md).
- `classify`: Emotion classification from text, via distilBERT (28 emotions).
  - Useful for controlling the avatar's emotional state, by running `classify` on the last few sentences of LLM output, and then setting the avatar's emotion based on the result.
- `embeddings`: Semantic embeddings, a.k.a. vector embeddings. Supports several *roles* that may each use a different embedding model.
  - The `"default"` role is useful e.g. for sentence similarity and for semantic visualization.
  - The `"qa"` role maps questions and their answers near each other, so it is useful e.g. as RAG vector DB keys.
- `imagefx`: server-side Anime4K upscaling and image filters for still images. No relation to the Google product.
  - Essentially, `imagefx` exposes the parts used by the avatar's postprocessor.
  - Used by `raven-avatar-settings-editor` to blur the background.
  - The `imagefx` module is useful when you want to process still images on the server's GPU. But the network roundtrip time (including image encoding and decoding) means it is not fast enough for processing a video stream.
  - If you want to use the same features locally, from Python (on the client's GPU), then don't use this module. Instead, use `raven.common.video.postprocessor` and `raven.common.video.upscaler` directly; those are fast enough for realtime. The `imagefx` module is just a web API wrapper for those.
- `natlang`: server-side [spaCy](https://spacy.io/) NLP. Python clients only.
  - Just like calling spaCy locally in the client process, but the model runs on the server's GPU.
  - Supports e.g. part-of-speech (POS) tagging, lemmatization, named entity recognition, and splitting into sentences.
- `sanitize`: Fix text broken by hyphenation, such as that extracted from scientific paper PDFs.
- `summarize`: Abstractive summarization of text, using a small, specialized AI model.
  - This is **much faster** than an LLM, but the result quality may not be as good.
  - The module has an automatic internal splitter that handles input that is longer than the model's context window, which is automatically queried from the model.
- `translate`: Natural language translation using a small, specialized AI model.
  - The module has an automatic internal splitter that sends one sentence at a time.
    - Many translation-specific AI models expect this format, and may fail spectacularly (even sometimes ignoring large parts of the input) if sent several sentences at a time.
    - Sentence boundaries are detected with the loaded spaCy model.
    - Note the implication: context is not preserved between sentences, so this will not be able to translate texts where seeing several sentences at once is important for the meaning.
  - A default config for English → Finnish is included, based on [Helsinki-NLP/opus-mt-tc-big-en-fi](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-fi).
    - This was found to perform better than more recent solutions such as [EuroLLM](https://huggingface.co/collections/utter-project/eurollm-66b2bd5402f755e41c5d9c6d), or even [Qwen3-30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507).
    - As of 08/2025, open-weight LLMs aren't that good with even moderately low-resource languages, like Finnish.
  - Models for other language combinations can be found on HuggingFace: [opus-mt-tc-big](https://huggingface.co/models?sort=trending&search=helsinki+opus+mt+big); [others](https://huggingface.co/tasks/translation).
  - There exist [updated translation models](https://huggingface.co/collections/HPLT/hplt-20-uni-direction-translation-models-67f2fc7ae54845f9b182957a) from the [HPLT consortium](https://hplt-project.org/) (released in 2025), but as of 08/2025, these don't support Transformers yet.
  - Note that the `translate` module currently requires a model that **does not** need a prefix instruction or a language code specification in the text sent to the model, so e.g. `base-t5` is not compatible.
- `tts`: Built-in, locally hosted [Kokoro-82M](https://github.com/hexgrad/kokoro).
  - Provides per-word timestamps and corresponding per-word phoneme data, useful for lipsyncing.
  - :exclamation: *For `tts`, you may need to install `espeak-ng` and have it available on your `PATH`, because the Kokoro TTS uses it as a fallback phonemizer.* :exclamation:
  - :exclamation: *`espeak-ng` is **not** a Python package, but a separate command-line app; how to install it depends on your OS.* :exclamation:
  - :exclamation: *In a Debian-based Linux (such as Ubuntu or Mint), `sudo apt install espeak-ng`. This is the **only** part of installing Raven that needs admin privileges.* :exclamation:
  - :exclamation: *Raven only ever calls `espeak-ng` from its `tts` module, and only for those inputs for which the TTS's built-in [Misaki](https://github.com/hexgrad/misaki) phonemizer fails.* :exclamation:
- `websearch`: A local [SERP](https://en.wikipedia.org/wiki/Search_engine_results_page) processor, used by the `websearch` tool of *Raven-minichat* (and will be similarly used by the upcoming *Raven-librarian*).
  - Continuation of the `websearch` module of *SillyTavern-Extras*, with improvements ported from the newer extension [SillyTavern-WebSearch-Selenium](https://github.com/SillyTavern/SillyTavern-WebSearch-Selenium).
  - We also provide a custom web API endpoint that returns structured search results (to easily keep each link with the corresponding search result).

Many of the NLP modules have an automatic CPU fallback in their loader: `classify`, `embeddings`, `natlang`, `sanitize`, `summarize`, and `translate`. If loading on GPU fails, these modules will note this in the server log, and auto-retry on the CPU. The rule of thumb is that, for a given module, if a slow response won't completely break the UX, then that module has a loader with a CPU fallback.

The `natlang` module is the only one that only works with Python clients. It needs a compatible instance of spaCy on the client side, to read the internal binary format (which is essentially a Python *pickle*). It was felt this is necessary to support arbitrary use cases of spaCy without overcomplicating the web API, or increasing *Raven-server*'s need of maintenance too much.

All other modules can be used with a client written in any programming language.


# Server configuration

:exclamation: *The server config file is, technically, arbitrary Python code.* :exclamation:

:exclamation: ***Never** install a server config from the internet, unless you are sure what that particular config does.* :exclamation:

The server config file is a Python module, which by default is [`raven.server.config`](../server/config.py).

To start the server with a custom config, use the `--config` command-line option.

The server config file is only read **once** per server session, when the server starts up.

## Low VRAM config (8 GB)

If your machine has 8 GB or less VRAM, see [`raven.server.config_lowvram`](../server/config_lowvram.py). To use it, start the server as:

```
raven-server --config raven.server.config_lowvram
```

This is useful e.g. if you are on the road with a laptop, and you'd like to dedicate the whole GPU for an LLM.

The low-VRAM config doesn't bother loading the `avatar` module, as it's nearly useless on CPU (~2 FPS; not a typo). It loads all other modules on CPU.

The low-VRAM config also doubles as an example of how to make your own customized config.

Especially, note that you can just `import` the base config and customize only the parts you want to change.

## Choosing which GPU to use (optional)

If your machine has multiple GPUs, there are two ways to tell *Raven-server* which GPU to use.

If your system *permanently* has several GPUs connected (like a desktop server rig), and you want to use a different GPU *permanently*, you can adjust the device settings in [`raven.server.config`](raven/server/config.py). These are configurable per-module.

If you run all server modules on the same GPU, and switch GPUs only occasionally (e.g. a laptop that sometimes has an eGPU connected and sometimes doesn't), you can use the `CUDA_VISIBLE_DEVICES` environment variable to choose the GPU temporarily, for the duration of a command prompt session.

We provide an example script [`run-on-internal-gpu.sh`](run-on-internal-gpu.sh), meant for a laptop with a Thunderbolt eGPU (external GPU), which forces Raven to run on the *internal* GPU when the external is connected (which is useful e.g. if your eGPU is dedicated for a self-hosted LLM). On the machine where the script was tested, PyTorch sees the eGPU as GPU 0 when available, pushing the internal GPU to become GPU 1. When the eGPU is not connected, the internal is GPU 0.

With the venv activated, and the terminal in the Raven folder, run the following `bash` command:

```bash
source run-on-internal-gpu.sh
```

Then for the rest of the command prompt session, any Raven commands (such as `raven-server`) will only see the internal GPU, and `"cuda:0"` in the device settings will point to the only visible GPU.


# SillyTavern compatibility

By default, *Raven-server* listens on `http://localhost:5100`, just like *SillyTavern-extras* did.

The following modules work as drop-in replacements for the module with the same name in the discontinued *SillyTavern-extras*:

  - `classify`
  - `embeddings`
  - `summarize`
  - `websearch`

The `tts` module provides an OpenAI compatible TTS endpoint (`/v1/audio/speech`) you can use as a speech synthesizer in *SillyTavern*.

The endpoint `/v1/audio/voices`, to list supported voices, is also provided, but ST doesn't call it.

*Talkinghead* support has been discontinued in *SillyTavern*. It would be interesting to introduce *Raven-avatar* as an upgraded replacement, but at the moment, there are no development resources to write a JS client for the avatar. If interested, much of the porting should be straightforward; see [#2](https://github.com/Technologicat/raven/issues/2).

## Raven-server TTS for SillyTavern

To connect *SillyTavern* to the `tts` module of *Raven-server*, to use it as ST's speech synthesizer:

- Open ST, and go into _Extensions ⊳ TTS_.
- Set the TTS provider to _OpenAI Compatible_.
- Set the provider endpoint to `http://127.0.0.1:5100/v1/audio/speech`.

To test, you can use `af_nova` as the voice.

To view a full list of available voices, point a browser to `http://127.0.0.1:5100/v1/audio/voices`.

The first letter of a voice name is the language code; the second letter is 'f' for female, 'm' for male.

Kokoro's single-letter language codes are (from [its README](https://github.com/hexgrad/kokoro?tab=readme-ov-file#advanced-usage)):

- 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
- 🇪🇸 'e' => Spanish es
- 🇫🇷 'f' => French fr-fr
- 🇮🇳 'h' => Hindi hi
- 🇮🇹 'i' => Italian it
- 🇯🇵 'j' => Japanese
- 🇧🇷 'p' => Brazilian Portuguese pt-br
- 🇨🇳 'z' => Mandarin Chinese

*All* of the voices can speak English, regardless of which language they were designed for (at least as long as you have `espeak-ng` installed; I haven't tested without it).

For example, if you use `ff_siwis` as the voice, you'll get a French accent. (_Editor's note: Raven-avatar provides a maid character for testing._)

Similarly, `jf_alpha` yields a Japanese accent.


# Python bindings (easy client API)

*Last updated for v0.2.3.*

*This API documentation is maintained on a best-effort basis, but sometimes, recent changes may be missing. For the ground truth, see below.*

For integration with Python-based apps, we provide *Python bindings*, i.e. an easy-to-use API that abstracts away the fact that you're interfacing with a server. With the Python bindings, you can call the web API endpoints on the server by calling regular Python functions - your code does not need to care about JSON or HTTP.

The server URL that the Python bindings call, can be set in [`raven.client.config`](../client/config.py). The default is `http://localhost:5100`.

The Python bindings live in [`raven.client.api`](../client/api.py) and [`raven.client.tts`](../client/tts.py). This split is because the TTS client includes the avatar lipsync driver, which is a couple hundred SLOC, although rather simple.

To avoid duplication, most of the client API functions are not documented separately. The documentation lives on the server side, in the docstrings of [`raven.server.app`](../server/app.py), for the function that serves each specific web API endpoint. The parameters of the Python bindings are the natural Python equivalent of what goes into the web API as JSON.

Full list of Python API functions:

- **General**:
  - `initialize`: Must be called first.
    - This loads the client configuration, and starts the audio service so that TTS can work.
    - The implementation of this one client API function actually lives in `raven.client.util`; see function `initialize_api`.
  - `raven_server_available`: Check whether the client can connect to *Raven-server* (whose URL was specified when `initialize` was called).
  - `tts_server_available`: Same, but check the TTS server. This isn't needed when using *Raven-server*'s internal TTS.
- **Raven-avatar client** (AI-animated anime avatar):
  - `avatar_load`: Create an avatar session and load a character into it. You'll get a **session ID**, which is needed for **all other** avatar API functions; they operate on a specific session.
  - `avatar_reload`: Reload a character into the specified avatar session; useful e.g. if the image file has changed on disk, or if you want to switch to another character.
    - Used by `raven-avatar-settings-editor` for its character-loading and refresh features.
  - `avatar_unload`: End an avatar session.
  - `avatar_load_emotion_templates`: Load a set of avatar emotion templates, from a Python dictionary.
    - Format of the dictionary is:
        ```
        {"emotion0": {"morph0": value0,
                      ...}
         ...}
        ```
      Here morphs include cel blends (except animefx).

      See `Animator.load_emotion_templates` in [`raven.server.modules.avatar`](../server/modules/avatar.py).

      The factory-default emotions file [`raven/avatar/assets/emotions/_defaults.json`](../avatar/assets/emotions/_defaults.json) is a full example using this format.
  - `avatar_load_emotion_templates_from_file`: Load a set of avatar emotion templates, from an emotion JSON file.
    - Format as above. You could point this to the factory-default emotions file to load that.
  - `avatar_load_animator_settings`: Load animator, upscaler and postprocessor settings, from a Python dictionary.
    - Format of the dictionary is:
        ```
        {"name0": value0,
         ...}
        ```
      See `animator_defaults` (and `postprocessor_defaults`) in [`raven.server.config`](../server/config.py) for a full example (which also doubles as an authoritative list of supported settings, documented in comments).

      The settings files in [`raven/avatar/assets/settings/`](../avatar/assets/settings/) are examples using this format.

      The `raven-avatar-settings-editor` GUI app saves settings files with this format.
  - `avatar_load_animator_settings_from_file`: Load animator, upscaler and postprocessor settings, from a settings JSON file.
    - Format as above. You could point this to a settings file saved by the `raven-avatar-settings-editor` GUI app.
  - `avatar_start`: Start (resume) avatar animation.
  - `avatar_stop`: Stop (pause) avatar animation.
  - `avatar_start_talking`: Start a generic talking animation (randomized mouth) for no-audio environments. See also `tts_speak_lipsynced`.
  - `avatar_stop_talking`: Stop the generic talking animation.
  - `avatar_set_emotion`: Set the avatar character's current emotion.
  - `avatar_set_overrides`: Manually control specific morphs (including cel blends, except animefx).
    - Used by the lipsync driver to control the character's mouth based on timestamped phoneme data.
  - `avatar_result_feed`: Receive the video feed of the avatar, as a `multipart/x-mixed-replace` stream of images.
    - Image format and desired framerate are set in the animator settings.
    - The server will try hard to keep the desired framerate.
      - If rendering falls behind, so that a new frame is not available when needed, the latest available frame is re-sent.
      - If network transport falls behind, so that frames would pile up on the server, rendering auto-pauses until the latest frame has been sent.
      - In normal operation, the server works on three consecutive frames at once: while frame X is being sent, X+1 is being encoded, and X+2 is being rendered.
        - This increases parallelization at the cost of some latency.
    - In the Python API, this returns a generator that yields video frames from the server. See usage example in [`raven.avatar.settings_editor.app`](../avatar/settings_editor/app.py).
    - **NOTE**: In the web API, unlike most others, this is a `GET` endpoint. The session ID is sent as a URL parameter.
  - `avatar_get_available_filters`: Get list of available image filters in the postprocessor.
    - The same image filters are also exposed to the `imagefx` module. This is the only API function to get the list.
- **Text sentiment classification**:
  - `classify_labels`: Get list of emotions supported by the `classify` model loaded to the server.
    - By default, the model is distilBERT, with 28 emotions, compatible with the avatar's emotion templates.
  - `classify`: Classify the emotion from a piece of text.
- **Semantic embeddings of text**:
  - `embeddings_compute`: Compute semantic embeddings (vector embeddings).
    - This uses `sentence_transformers`.
    - You can optionally specify the *role*. By default, the `"default"` role is used. You may want `"qa"`, depending on the use case. See [the server config file](../server/config.py).
- **Image processing**:
  - `imagefx_process`: Apply postprocess filters (on the server) to an image from a filelike or a `bytes` object.
    - The filters are a filter chain, formatted as in `raven.server.config.postprocessor_defaults`.
    - Be sure to set some filters; the default is a blank list, which does nothing.
  - `imagefx_process_file`: Apply postprocess filters (on the server) to an image from a file.
  - `imagefx_process_array`: Apply postprocess filters (on the server) to an image from a NumPy array.
    - Array format `float32`, range `[0, 1]`, layout `[h, w, c]`, either RGB (3 channels) or RGBA (4 channels).
  - `imagefx_upscale`: Anime4K upscale (on the server) an image from a filelike or a `bytes` object.
    - Anime4K presets `A`, `B` and `C`, with high or low quality. For details, see docstring.
  - `imagefx_upscale_file`: Anime4K upscale (on the server) to an image from a file.
  - `imagefx_upscale_array`: Anime4K upscale (on the server) to an image from a NumPy array.
- **Server-side spaCy NLP**:
  - `natlang_analyze`: Run text through a spaCy pipeline on the server, using the loaded spaCy model, and send the results to the client.
    - The transport is spaCy's binary format (which uses Python's *pickle*), so the client must be running a compatible spaCy with a compatible version of Python.
    - The client loads an empty English pipeline to receive the results. This behaves as if the result came from a local spaCy instance in the client process: you can look at tokens, their parts of speech and lemmas, sentences, ...
    - You can optionally specify which spaCy pipes to enable. This is useful to speed up processing by skipping unnecessary pipes, if e.g. just sentence splitting is needed (`pipes=["tok2vec", "parser", "senter"]`).
- **Text cleanup**:
  - `sanitize_dehyphenate`: Fix text broken by hyphenation, such as that extracted from scientific paper PDFs.
- **Text summarization**:
  - `summarize_summarize`: Generate an abstractive summary for text.
    - This uses a small, specialized AI model, which is not as accurate as an LLM, but is much faster.
- **Natural language translation**:
  - `translate_translate`: Translate text from one natural language to another.
    - This uses a small, specialized AI model for sentence-level translation.
    - Default configuration for English to Finnish is provided.
- **Speech synthesizer (TTS)**:
  - `tts_list_voices`: Get a list of all voice names supported by the TTS.
  - `tts_speak`: Speak text using the TTS. No lipsync.
  - `tts_speak_lipsynced`: Speak text using the TTS. Lipsync the specified avatar session to the speech audio.
  - `tts_stop`: Stop speaking. Useful for canceling while speech in progress. (Will in any case stop automatically when the speech audio ends.)
- **Web search**:
  - `websearch_search`: Perform a web search and parse the [SERP](https://en.wikipedia.org/wiki/Search_engine_results_page).
    - As a new feature over what *SillyTavern-Extras* did, you'll now get the results in a structured format that preserves the information of which link belongs to which search result.
    - Search engines change things over time, so this is likely to break at some point. If you notice it doesn't work, please open an issue.


# Web API endpoints

*Last updated for v0.2.3.*

*This web API endpoint documentation is maintained on a best-effort basis, but sometimes, recent changes may be missing. The ground truth are the docstrings, and ultimately the actual implementation, both of which can be found in [`raven.server.app`](../server/app.py).*

For usage examples, look at the Python bindings of the web API in [`raven.client.api`](../client/api.py) and [`raven.client.tts`](../client/tts.py). These should be straightforward to port to other programming environments if needed. Porting the bindings specifically to JavaScript, for web app clients, is tracked in [#2](https://github.com/Technologicat/raven/issues/2).

**TODO: document each web API endpoint here; name, docstring from `raven.server.app`**

- New simple ping endpoint `/health` (note no `/api/...`), for a client to easily check that the server is up and running.
