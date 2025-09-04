<p align="center">
<img src="../../img/logo.png" alt="Logo of Raven" height="200"/> <br/>
</p>

-----

<p align="center">
<img src="../../img/screenshot-server.png" alt="Screenshot of Raven-server" width="800"/> <br/>
<i>Raven-server serving on localhost.</i>
</p>

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
    - [Important differences to SillyTavern-extras](#important-differences-to-sillytavern-extras)
        - [Overhauled command-line options](#overhauled-command-line-options)
        - [Removed modules](#removed-modules)
    - [Raven-server TTS for SillyTavern](#raven-server-tts-for-sillytavern)
- [Python bindings (easy client API)](#python-bindings-easy-client-api)
    - [General](#general)
    - [Raven-avatar client](#raven-avatar-client)
    - [Text sentiment classification](#text-sentiment-classification)
    - [Semantic embeddings](#semantic-embeddings)
    - [Image processing](#image-processing)
    - [Server-side spaCy NLP](#server-side-spacy-nlp)
    - [Text cleanup](#text-cleanup)
    - [Text summarization](#text-summarization)
    - [Natural language translation](#natural-language-translation)
    - [Speech synthesizer (TTS)](#speech-synthesizer-tts)
    - [Web search](#web-search)
- [Web API endpoints](#web-api-endpoints)
    - [General](#general-1)
        - [GET "/"](#get-)
        - [GET "/health"](#get-health)
        - [GET "/api/modules"](#get-apimodules)
    - [Raven-avatar client](#raven-avatar-client-1)
        - [POST "/api/avatar/load"](#post-apiavatarload)
        - [POST "/api/avatar/reload"](#post-apiavatarreload)
        - [POST "/api/avatar/unload"](#post-apiavatarunload)
        - [POST "/api/avatar/load_emotion_templates"](#post-apiavatarload_emotion_templates)
        - [POST "/api/avatar/load_animator_settings"](#post-apiavatarload_animator_settings)
        - [POST "/api/avatar/start"](#post-apiavatarstart)
        - [POST "/api/avatar/stop"](#post-apiavatarstop)
        - [POST "/api/avatar/start_talking"](#post-apiavatarstart_talking)
        - [POST "/api/avatar/stop_talking"](#post-apiavatarstop_talking)
        - [POST "/api/avatar/set_emotion"](#post-apiavatarset_emotion)
        - [POST "/api/avatar/set_overrides"](#post-apiavatarset_overrides)
        - [GET "/api/avatar/result_feed"](#get-apiavatarresult_feed)
        - [GET "/api/avatar/get_available_filters"](#get-apiavatarget_available_filters)
    - [Text sentiment classification](#text-sentiment-classification-1)
        - [POST "/api/classify"](#post-apiclassify)
        - [GET "/api/classify/labels"](#get-apiclassifylabels)
    - [Semantic embeddings](#semantic-embeddings-1)
        - [POST "/api/embeddings/compute"](#post-apiembeddingscompute)
    - [Image processing](#image-processing-1)
        - [POST "/api/imagefx/process"](#post-apiimagefxprocess)
        - [POST "/api/imagefx/upscale"](#post-apiimagefxupscale)
    - [Server-side spaCy NLP](#server-side-spacy-nlp-1)
        - [POST "/api/natlang/analyze"](#post-apinatlanganalyze)
    - [Text cleanup](#text-cleanup-1)
        - [POST "/api/sanitize/dehyphenate"](#post-apisanitizedehyphenate)
    - [Text summarization](#text-summarization-1)
        - [POST "/api/summarize"](#post-apisummarize)
    - [Natural language translation](#natural-language-translation-1)
        - [POST "/api/translate"](#post-apitranslate)
    - [Speech synthesizer (TTS)](#speech-synthesizer-tts-1)
        - [GET "/api/tts/list_voices"](#get-apittslist_voices)
        - [POST "/api/tts/speak"](#post-apittsspeak)
        - [GET "/v1/audio/voices"](#get-v1audiovoices)
        - [POST "/v1/audio/speech"](#post-v1audiospeech)
    - [Web search](#web-search-1)
        - [POST "/api/websearch"](#post-apiwebsearch)
        - [POST "/api/websearch2"](#post-apiwebsearch2)

<!-- markdown-toc end -->

# Introduction

*Raven-server* is a web API server that hosts local, specialized AI models on the GPU:

- **Avatar**: AI-animated custom anime character for your LLM. This is the server side of [*Raven-avatar*](../avatar/README.md).
- **Speech synthesizer (TTS)**: built-in, locally hosted [Kokoro-82M](https://github.com/hexgrad/kokoro).
- **Natural language processing (NLP)**: various components for GPU-accelerated natural language analysis and processing.

Although the default is to run both the server and the client apps on localhost, the server can run anywhere on the local network. This allows a separate machine with a powerful GPU to host the server for one or more clients on the local network.

Most of the server functions are stateless; the only exception is *Raven-avatar*, which gives you a session ID.

For the speech synthesizer, we provide two web API endpoints: an OpenAI compatible one, and a custom one. The custom endpoint provides word timestamps and per-word phoneme data, which is needed for lipsyncing the avatar. The actual lipsync driver lives on the client side, in the [Python bindings](#python-bindings-easy-client-api), because the speech audio playback is done on the client side, too.

Historically, *Raven-server* began as a continuation of the discontinued *SillyTavern-extras*. One important reason was to keep the avatar technology alive; it is a promising, unique experiment that no other project seems to have followed up on. But also, a web API server for various specialized NLP functionality happened to be exactly what *Raven-visualizer* and the upcoming *Raven-librarian* needed. The server has since been extended in various ways: the avatar has several new features, the built-in TTS is new, and several new NLP modules have been added.


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
  - :exclamation: *In practice, the fallback is used for out-of-dictionary words in English, as well as for some non-English languages.* :exclamation:
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

By default, *Raven-server* listens on `http://localhost:5100`, just like the discontinued *SillyTavern-extras* did.

The following modules work as drop-in replacements for the module with the same name in ST-Extras:

  - `classify`
  - `embeddings`
  - `summarize`
  - `websearch`

Additionally, the `tts` module provides an OpenAI compatible TTS endpoint (`/v1/audio/speech`) you can use as a speech synthesizer in ST ([see below](#raven-server-tts-for-sillytavern)).

The `embeddings` module is the most useful one, because the GPU-accelerated embedder runs much faster than ST's built-in one. Speed is crucial if you routinely upload 20-page PDFs to the data bank to discuss them with your LLM.

The recommended way to add web search to ST is [SillyTavern-WebSearch-Selenium](https://github.com/SillyTavern/SillyTavern-WebSearch-Selenium), which is an official extension by the ST developers. *Raven-server* provides a `websearch` module mainly because Raven itself needs that functionality.

*Talkinghead* support has been discontinued in ST. It would be interesting to introduce *Raven-avatar* as an upgraded replacement, but at the moment, there are no development resources to write a JS client for the avatar. If you are a developer interested in solving this, see [#2](https://github.com/Technologicat/raven/issues/2).


## Important differences to SillyTavern-extras

### Overhauled command-line options

In *Raven-server*, server modules are enabled/disabled and configured **in the server config file**, not on the command line.

Thus, to support environments with varying use cases, we only provide **one** command-line option for server module configuration: namely, `--config`, to load a different config file.

### Removed modules

Some modules have been removed:

- `caption`
- `chromadb`: use ST's Vector Storage instead.
- `coqui-tts`, `edge-tts`, `rvc`, `silero-tts`: replaced by the new `tts` module.
- `streaming-stt`, `vosk-stt`, `whisper-stt`
- `sd`
- `talkinghead`: replaced by the new `avatar` module (currently not supported by SillyTavern).

Particularly:

**No STT support.** Potentially interesting. We could add STT via [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo), but this needs some consideration of the UX. *Raven-server* might run on a different machine than the client, so the audio needs to be recorded on the client, and sent to the server for transcription.

**No image captioning support.** Potentially interesting, if we expand other components of Raven to handle image analysis later.

**No image generation support.** Out of scope for Raven.


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

There is also a DPG GUI driver in [`raven.client.avatar_renderer`](../client/avatar_renderer.py). If you use Python and DPG, this is the easy way to integrate the avatar to your app. If you use another GUI toolkit or another programming language, this serves as a complete example of the low-level client code needed to build a GUI driver for the avatar.

To avoid duplication, most of the client API functions are not documented separately. The documentation lives on the server side, in the docstrings of [`raven.server.app`](../server/app.py), for the function that serves each specific web API endpoint. The parameters of the Python bindings are the natural Python equivalent of what goes into the web API as JSON.

Full list of Python API functions follows.

## General

- `initialize`: **Must be called first.**
  - This loads the client configuration, and starts the audio service so that TTS can work.
  - The implementation of this one client API function actually lives in `raven.client.util`; see function `initialize_api`.
- `raven_server_available`: Check whether the client can connect to *Raven-server* (whose URL was specified when `initialize` was called).
- `tts_server_available`: Same, but check the TTS server. This isn't needed when using *Raven-server*'s internal TTS (recommended).

## Raven-avatar client

AI-animated anime avatar for your LLM.

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
  - In the Python API, this returns a generator that yields video frames from the server.
    - Note that `avatar_result_feed` is the low-level API, meant for building GUI drivers for the avatar.
    - We provide a GUI driver for DearPyGui (DPG); see `DPGAvatarRenderer` in [`raven.client.avatar_renderer`](../client/avatar_renderer.py). It also doubles as a usage example of `avatar_result_feed`; see especially the `start` method and the background task spawned there.
  - **NOTE**: In the web API, unlike most others, this is a `GET` endpoint. The session ID is sent as a URL parameter.
- `avatar_get_available_filters`: Get list of available image filters in the postprocessor.
  - The same image filters are also exposed to the `imagefx` module. This is the only API function to get the list.

**DPG GUI driver for avatar**

This lives in a separate module, [`raven.client.avatar_renderer`](../client/avatar_renderer.py). It is part of the API, but not imported automatically.

- `DPGAvatarRenderer`: Receive the avatar video stream, and render it in a DearPyGui (DPG) image widget.
  - This also serves as a starting point for porting the avatar's GUI driver to other GUI toolkits and to other programming languages.
  - Instantiate `DPGAvatarRenderer`, then configure it, then `start` it. See docstrings for details. See usage example in [`raven.avatar.settings_editor.app`](../avatar/settings_editor/app.py).

## Text sentiment classification

- `classify_labels`: Get list of emotions supported by the `classify` model loaded to the server.
  - By default, the model is distilBERT, with 28 emotions, compatible with the avatar's emotion templates.
- `classify`: Classify the emotion from a piece of text.

## Semantic embeddings

- `embeddings_compute`: Compute semantic embeddings (vector embeddings) of text.
  - This uses `sentence_transformers`.
  - You can optionally specify the *role*. By default, the `"default"` role is used. You may want `"qa"`, depending on the use case. See [the server config file](../server/config.py).

## Image processing

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

## Server-side spaCy NLP

- `natlang_analyze`: Run text through a spaCy pipeline on the server, using the loaded spaCy model, and send the results to the client.
  - The transport is spaCy's binary format (which uses Python's *pickle*), so the client must be running a compatible spaCy with a compatible version of Python.
  - The client loads an empty English pipeline to receive the results. This behaves as if the result came from a local spaCy instance in the client process: you can look at tokens, their parts of speech and lemmas, sentences, ...
  - You can optionally specify which spaCy pipes to enable. This is useful to speed up processing by skipping unnecessary pipes, if e.g. just sentence splitting is needed (`pipes=["tok2vec", "parser", "senter"]`).

## Text cleanup

- `sanitize_dehyphenate`: Fix text broken by hyphenation, such as that extracted from scientific paper PDFs.

## Text summarization

- `summarize_summarize`: Generate an abstractive summary for text.
  - This uses a small, specialized AI model, which is not as accurate as an LLM, but is much faster.

## Natural language translation

- `translate_translate`: Translate text from one natural language to another.
  - This uses a small, specialized AI model for sentence-level translation.
  - Default configuration for English to Finnish is provided.

## Speech synthesizer (TTS)

- `tts_list_voices`: Get a list of all voice names supported by the TTS.
- `tts_speak`: Speak text using the TTS. No lipsync.
- `tts_speak_lipsynced`: Speak text using the TTS. Lipsync the specified avatar session to the speech audio.
- `tts_stop`: Stop speaking. Useful for canceling while speech in progress. (Will in any case stop automatically when the speech audio ends.)

## Web search

- `websearch_search`: Perform a web search and parse the [SERP](https://en.wikipedia.org/wiki/Search_engine_results_page).
  - As a new feature over what *SillyTavern-Extras* did, you'll now get the results in a structured format that preserves the information of which link belongs to which search result.
  - Search engines change things over time, so this is likely to break at some point. If you notice it doesn't work, please open an issue.


# Web API endpoints

*Last updated for v0.2.3.*

*This web API endpoint documentation is maintained on a best-effort basis, but sometimes, recent changes may be missing. The ground truth are the docstrings, and ultimately the actual implementation, both of which can be found in [`raven.server.app`](../server/app.py).*

For usage examples, look at the Python bindings of the web API in [`raven.client.api`](../client/api.py) and [`raven.client.tts`](../client/tts.py). These should be straightforward to port to other programming environments if needed. Porting the bindings specifically to JavaScript, for web app clients, is tracked in [#2](https://github.com/Technologicat/raven/issues/2).


## General

### GET "/"

  Return this documentation.

  No inputs.

  Output is suitable for rendering in a web browser.


### GET "/health"

  A simple ping endpoint for clients to check that the server is running.

  No inputs, no outputs - if you get a 200 OK, it means the server heard you.


### GET "/api/modules"

  Get a list of enabled modules.

  No inputs.

  Output format is JSON:

      {"modules": ["modulename0",
                   ...]}


## Raven-avatar client

AI-animated anime avatar for your LLM.

### POST "/api/avatar/load"

Start a new avatar instance, and load the avatar sprite posted as a file in the request.

Input is POST, Content-Type `"multipart/form-data"`, with one file attachment, named `"file"`.

The file should be an RGBA image in a format that Pillow can read. It will be autoscaled to 512x512.

Optionally, there may be more file attachments, one for each add-on cel. The attachment name
for each is the cel name. For supported cels, see `supported_cels` in `raven.server.avatarutil`,
and for animefx cels, `raven.server.config`.

Output is JSON:

    {"instance_id": "some_important_string"}

Here the important string is the instance ID the new avatar instance. Use this instance ID in the
other avatar API endpoints to target the operations to this instance.


### POST "/api/avatar/reload"

For an existing avatar instance, load the avatar sprite posted as a file in the request, replacing the current sprite.

Input is POST, Content-Type `"multipart/form-data"`, with two file attachments, named `"file"` and `"json"`.

The `"file"` attachment should be an RGBA image in a format that Pillow can read. It will be autoscaled to 512x512.

Optionally, there may be more file attachments, one for each add-on cel. The attachment name
for each is the cel name. For supported cels, see `supported_cels` in `raven.server.avatarutil`,
and for animefx cels, `raven.server.config`.

The "json" attachment should contain the API call parameters as JSON:

    {"instance_id": "some_important_string"}

Here the important string is the instance ID you got from `api_avatar_load`.

No outputs.


### POST "/api/avatar/unload"

Unload (delete) the given avatar instance.

This automatically causes the `result_feed` for that instance to shut down.

Input is JSON:

    {"instance_id": "some_important_string"}

Here the important string is the instance ID you got from `api_avatar_load`.

No outputs.


### POST "/api/avatar/load_emotion_templates"

Load custom emotion templates for avatar, or reset to defaults.

Input is JSON:

    {"instance_id": "some_important_string",
     "emotions": {"emotion0": {"morph0": value0,
                               ...}
                  ...}
    }

For details, see `Animator.load_emotion_templates` in `raven/server/modules/avatar.py`.

To reload server defaults, send `"emotions": {}` or omit it.

Here the important string is the instance ID you got from `api_avatar_load`.

No outputs.


### POST "/api/avatar/load_animator_settings"

Load custom settings for avatar animator and postprocessor, or reset to defaults.

Input is JSON:

    {"instance_id": "some_important_string",
     "animator_settings": {"name0": value0,
                           ...}
    }

For details, see `Animator.load_animator_settings` in `animator.py`.

To reload server defaults, send `"animator_settings": {}` or omit it.

Here the important string is the instance ID you got from `api_avatar_load`.

No outputs.


### POST "/api/avatar/start"

Start the avatar animation.

Input is JSON::

    {"instance_id": "some_important_string"}

Here the important string is the instance ID you got from `api_avatar_load`.

No outputs.

A character must be loaded first; use `/api/avatar/load` to do that.

To pause, use `/api/avatar/stop`.


### POST "/api/avatar/stop"

Pause the avatar animation.

Input is JSON::

    {"instance_id": "some_important_string"}

Here the important string is the instance ID you got from `api_avatar_load`.

No outputs.

To resume, use `/api/avatar/start`.


### POST "/api/avatar/start_talking"

Start the mouth animation for talking.

Input is JSON:

    {"instance_id": "some_important_string"}

Here the important string is the instance ID you got from `api_avatar_load`.

No outputs.

This is the generic, non-lipsync animation that randomizes the mouth.

This is useful for applications without actual voiced audio, such as
an LLM when TTS is offline, or a low-budget visual novel.

For speech with automatic lipsync, see `tts_speak_lipsynced`.


### POST "/api/avatar/stop_talking"

Stop the mouth animation for talking.

Input is JSON::

    {"instance_id": "some_important_string"}

Here the important string is the instance ID you got from `api_avatar_load`.

No outputs.

This is the generic, non-lipsync animation that randomizes the mouth.

This is useful for applications without actual voiced audio, such as
an LLM when TTS is offline, or a low-budget visual novel.

For speech with automatic lipsync, see `tts_speak_lipsynced`.


### POST "/api/avatar/set_emotion"

Set avatar emotion to that posted in the request.

Input is JSON:

    {"instance_id": "some_important_string",
     "emotion_name": "curiosity"}

where the key "emotion_name" is literal, and the value is the emotion to set.

Here the important string is the instance ID you got from `api_avatar_load`.

No outputs.

There is no getter, by design. If the emotion state is meaningful to you,
keep a copy in your frontend, and sync that to the server.


### POST "/api/avatar/set_overrides"

Directly control the animator's morphs from the client side.

Useful for lipsyncing.

Input is JSON::

    {"instance_id": "some_important_string",
     "overrides": {"morph0": value0,
                   ...}
    }

To unset overrides, set `"overrides": {}` or omit it.

See `raven.avatar.editor` for available morphs. Value range for most morphs is [0, 1],
and for morphs taking also negative values, it is [-1, 1].

No outputs.

There is no getter, by design. If the override state is meaningful to you,
keep a copy in your frontend, and sync that to the server.


### GET "/api/avatar/result_feed"

Video output.

Example:

    GET /api/avatar/result_feed?instance_id=some_important_string

where `some_important_string` is what `/api/avatar/load` gave you.

The instance ID is an URL parameter to make it trivially easy to display the video stream
in a web browser. (You still have to get the ID from `/api/avatar/load`; it's an UUID,
so unfortunately it's not very human-friendly.)

Output is a `"multipart/x-mixed-replace"` stream of video frames, each as an image file.
The payload separator is `"--frame"`.

The file format can be set in the animator settings. The frames are always sent
with the Content-Type and Content-Length headers set.


### GET "/api/avatar/get_available_filters"

Get metadata of all available postprocessor filters and their available parameters.

The intended audience of this endpoint is developers; this is useful for dynamically
building an editor GUI for the postprocessor chain.

No inputs.

Output is JSON:

    {"filters": [
                  [filter_name, {"defaults": {param0_name: default_value0,
                                              ...},
                                 "ranges": {param0_name: [min_value0, max_value0],
                                            ...}}],
                   ...
                ]
    }

For any given parameter, the format of the parameter range depends on the parameter type:

  - numeric (int or float): [min_value, max_value]
  - bool: [true, false]
  - multiple-choice str: [choice0, choice1, ...]
  - RGB color: ["!RGB"]
  - safe to ignore in GUI: ["!ignore"]

In the case of an RGB color parameter, the default value is of the form [R, G, B],
where each component is a float in the range [0, 1].

You can detect the type from the default value.


## Text sentiment classification

### POST "/api/classify"

Perform sentiment analysis (emotion classification) on the text posted in the request. Return the result.

Input is JSON:

    {"text": "Blah blah blah."}

Output is JSON::

    {"classification": [{"label": emotion0, "score": confidence0},
                        ...]}

sorted by score, descending, so that the most probable emotion is first.


### GET "/api/classify/labels"

Return the available classifier labels for text sentiment (character emotion).

No inputs.

Output is JSON:

    {"labels": [emotion0,
                ...]}

The actual labels depend on the classifier model.


## Semantic embeddings

### POST "/api/embeddings/compute"

Compute the vector embedding of one or more sentences of text.

Input is JSON:

    {"text": "Blah blah blah.",
     "model": "default"}

or:

    {"text": ["Blah blah blah.",
              ...],
     "model": "default"}

The "model" field is optional. It selects the role:

  - If not specified, "default" is used.

  - If specified, the value must be one of the keys of `embedding_models` in the server config.
    The default config is `raven.server.config`, but note the server's `--config` command-line
    option, which can be used to specify a different config at server startup.

  - If specified but not present in server config, the request aborts with HTTP error 400.

This functionality is provided because different models may be good for different use cases;
e.g. beside a general-purpose embedder, having a separate specialized "qa" embedder that maps
questions and related answers near each other.

Output is also JSON:

    {"embedding": array}

or:

    {"embedding": [array0,
                   ...]}

respectively.


## Image processing

### POST "/api/imagefx/process"

Run an image through a postprocessor chain.

This can be used e.g. for blurring a client-side background for the AI avatar,
running the blur filter on the server's GPU.

Input is POST, Content-Type `"multipart/form-data"`, with two file attachments:

    "file": the actual image file (binary, any supported format)
    "json": the API call parameters, in JSON format.

The parameters are:

    {"format": "png",
     "filters": [[filter0, {param0_name: value0, ...}],
                 ...]}

Supported image formats (both input and output) are RGB/RGBA formats supported by Pillow,
and QOI (Quite OK Image).

If you need speed, and your client supports it, prefer the QOI format. Especially the
encoder is dozens of times faster than PNG's, and compresses almost as tightly.

To get supported filters, call the endpoint `/api/avatar/get_available_filters`.
Don't mind the name - the endpoint is available whenever at least one of `avatar`
or `imagefx` is loaded.

Output is an image with mimetype `"image/<format>"`.


### POST "/api/imagefx/upscale"

Upscale an image with Anime4K.

Input is POST, Content-Type `"multipart/form-data"`, with two file attachments:

    "file": the actual image file (binary, any supported format)
    "json": the API call parameters, in JSON format.

The parameters are:

    {"format": "png",
     "upscaled_width": 1920,
     "upscaled_height": 1080,
     "preset": "C",
     "quality": "high"}

Supported image formats (both input and output) are RGB/RGBA formats supported by Pillow,
and QOI (Quite OK Image).

Preset is "A", "B" or "C", corresponding to the Anime4K preset with the same letter;
for the meanings, see `raven.common.video.upscaler`.

Quality is "high" or "low".

If you need speed, and your client supports it, prefer the QOI format. Especially the
encoder is dozens of times faster than PNG's, and compresses almost as tightly.

Output is an image with mimetype `"image/<format>"`.


## Server-side spaCy NLP

### POST "/api/natlang/analyze"

Perform NLP analysis on the text posted in the request. Return the result.

:exclamation: *This endpoint returns Python spaCy data in binary format.* :exclamation:

:exclamation: *It can only be read by a Python client process running a Python version
that is compatible with the Python running the server. Specifically,
since spaCy transmits binary data in the pickle format, the client
Python version must be such that it can unpickle data pickled by
the server process.* :exclamation:

:exclamation: *The point is that you can use the server's GPU to perform the NLP analysis,
and then read the results in the client as if they came from a spaCy
instance running locally on the client.* :exclamation:

:exclamation: *The most convenient way to call this endpoint is `natlang_analyze`
in `raven.client.api`.* :exclamation:

The NLP analysis is done via spaCy using the `spacy_model` set in the server config,
which by default lives in `raven.server.config`.

The analysis currently includes part-of-speech tagging, lemmatization,
and named entity recognition.

Input is JSON:

    {"text": "Blah blah blah."}

To send multiple texts at once, use a list:

    {"text": ["Blah blah blah.",
              "The quick brown fox jumps over the lazy dog."]}

This is more efficient than sending one text at a time, as it allows the spaCy backend
to batch the texts.

The server runs the text(s) through the default set of pipes in the loaded spaCy model.
There is an optional "pipes" field you can use to enable only the pipes you want.
(Which ones exist depend on the spaCy model that is loaded.)

This works for both one text, or multiple texts. Here is a one-text example:

    {"text": "Blah blah blah.",
     "pipes": ["tok2vec", "parser", "senter"]}

This effectively does `with nlp.select_pipes(enable=pipes): ...` on the server side.

This can be useful to save processing time if you only need partial analysis,
e.g. to split the text into sentences (for the model `"en_core_web_sm"`, the pipes
in the example will do just that).

Output is binary data. It can be loaded on the client side by calling
`raven.common.nlptools.deserialize_spacy_docs`, which see.

The response contains a **custom header**, `"x-langcode"`, which is the language code
of the server's loaded spaCy model (e.g. `"en"` for English). The client needs the
language code to be able to deserialize the data correctly.

If you use `natlang_analyze` in `raven.client.api`, it already loads the data,
and behaves as if you had called `nlp.pipe(...)` (locally, on the client) on the text.


## Text cleanup

### POST "/api/sanitize/dehyphenate"

Dehyphenate the text posted in the request, using a small, specialized AI model.

The AI is a character-level contextual embeddings model from the
Flair-NLP project.

This can be used to clean up broken text e.g. as extracted
from a PDF file:

    Text that was bro-
    ken by hyphenation.

→

    Text that was broken by hyphenation.

If you intend to send the text to an LLM, having it broken
by hyphenation doesn't matter much in practice, but this
makes the text much nicer for humans to look at.

Be aware that this often causes paragraphs to run together,
because the likely-paragraph-split analyzer is not perfect.
We could analyze one paragraph at a time, but we currently don't,
because the broken input text could contain blank lines at
arbitrary positions, so these are not a reliable indicator
of actual paragraph breaks. If you have known paragraphs you
want to preserve, you can send them as a list to process each
separately.

The primary use case for this in Raven is English text; but the
backend (with the `"multi"` model) does autodetect 300+ languages,
so give it a try.

This is based on the `dehyphen` package. The analysis applies a small,
specialized AI model (not an LLM) to evaluate the perplexity of the
different possible hyphenation options (in the example, "bro ken",
"bro-ken", "broken") that could have produced the hyphenated text.
The engine automatically picks the choice with the minimal perplexity
(i.e. the most likely according to the model).

We then apply some heuristics to clean up the output.

Input is JSON:

    {"text": "Text that was bro-\nken by hyphenation."}

or

    {"text": ["Text that was bro-\nken by hyphenation.",
              "Some more hyp-\nhenated text."]}

Output is also JSON:

    {"text": "Text that was broken by hyphenation."}

or

    {"text": ["Text that was broken by hyphenation.",
              "Some more hyphenated text."]}

respectively.


## Text summarization

### POST "/api/summarize"

Summarize the text posted in the request. Return the summary.

This uses a small, specialized AI model (not an LLM) plus some
heuristics to clean up its output.

Input is JSON:

    {"text": "Blah blah blah."}

Output is also JSON:

    {"summary": "Blah."}


## Natural language translation

### POST "/api/translate"

Translate the text posted in the request. Return the translation.

This uses a small, specialized AI model (not an LLM).

Input is JSON::

    {"text": "The quick brown fox jumps over the lazy dog.",
     "source_lang": "en",
     "target_lang": "fi"}

Output is also JSON:

    {"translation": "Nopea ruskea kettu hyppää laiskan koiran yli."}

To send several texts, use a list:

    {"text": ["The quick brown fox jumps over the lazy dog.",
              "Please translate this text, too."],
     "source_lang": "en",
     "target_lang": "fi"}

→

    {"translation": ["Nopea ruskea kettu hyppää laiskan koiran yli.",
                     "Ole hyvä ja käännä tämäkin teksti."]}


## Speech synthesizer (TTS)

:exclamation: *To use TTS, `espeak-ng` must be installed manually on the server because it's a non-Python dependency of the TTS feature.* :exclamation:

:exclamation: *The TTS engine Kokoro-82M uses it as a phonemizer fallback for out-of-dictionary words
in English, as well as for some non-English languages.* :exclamation:

### GET "/api/tts/list_voices"

Text to speech.

Get list of voice names from the speech synthesizer.

No inputs.

Output is JSON:

    {"voices": [voice0, ...]}

See:

[https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)


### POST "/api/tts/speak"

:exclamation: *This endpoint is general, for both non-lipsynced and lipsynced speech. The avatar lipsync driver is on the client side.* :exclamation:

Text to speech.

Input is JSON:

    {"text": "Blah blah blah.",
     "voice": "af_bella",
     "speed": 1.0,
     "format": "mp3",
     "get_metadata": true,
     "stream": false}

Only the "text" field is mandatory.

For available voices, call the endpoint `/api/tts/list_voices`.

For available formats, see `tts.text_to_speech`.

The audio file is returned as the response content. Content-Type is `"audio/<format>"`, e.g. `"audio/mp3"`.

If `"get_metadata"` is true, an extra **custom header** `"x-word-timestamps"` is returned, with JSON data
containing word-level timestamps and phonemes:

    [{"word": "reasonably" (URL-encoded to ASCII with percent-escaped UTF-8),
      "phonemes": "ɹˈizənəbli" (URL-encoded to ASCII with percent-escaped UTF-8),
      "start_time": 2.15,
      "end_time": 2.75},
     ...]

The start and end times are measured in seconds from start of audio.

This data is useful for lipsyncing and captioning.

Note `"get_metadata"` currently only works in English (we use a local Kokoro with only English installed).


### GET "/v1/audio/voices"

OpenAI compatible endpoint, for SillyTavern; does the exact same thing as `"/api/tts/list_voices"`.


### POST "/v1/audio/speech"

OpenAI compatible endpoint, for SillyTavern; does the exact same thing as `"/api/tts/speak"`.

However, this endpoint does **not** support `"get_metadata"`, because it's not part of the OAI format.
If you need lipsyncing or captioning, use `"/api/tts/speak"` instead.

Input is JSON:

    {"input": "Blah blah blah.",
     "voice": "af_bella",
     "speed": 1.0,
     "response_format": "mp3",
     "stream": false}

The audio file is returned as the response content. Content-Type is `"audio/<format>"`, e.g. `"audio/mp3"`.


## Web search

### POST "/api/websearch"

:exclamation: *Legacy endpoint, for compatibility. For new clients, prefer `"/api/websearch2"`.* :exclamation:

Perform a web search with the query posted in the request.

This is the SillyTavern compatible legacy endpoint.
For new clients, prefer to use `"/api/websearch2"`, which gives
structured output and has a `"max_links"` option.

Input is JSON:

    {"query": "what is the airspeed velocity of an unladen swallow",
     "engine": "duckduckgo"}

In the input, `"engine"` is optional. Valid values are `"duckduckgo"` (default)
and `"google"`.

Technically, this endpoint can also accept `"max_links"` (like in `/api/websearch2`),
but SillyTavern doesn't send it.

Output is JSON:

    {"results": preformatted_text,
     "links": [link0, ...]}

where the `"links"` field contains a list of all links to the search results.


### POST "/api/websearch2"

Perform a web search with the query posted in the request.

Input is JSON:

    {"query": "what is the airspeed velocity of an unladen swallow",
     "engine": "duckduckgo",
     "max_links": 10}

In the input, some fields are optional:

  - `"engine"`: valid values are `"duckduckgo"` (default) and `"google"`.
  - `"max_links"`: default 10.

The `"max_links"` field is a hint; the search engine may return more
results, especially if you set it to a small value (e.g. 3).

Output is JSON:

    {"results": preformatted_text,
     "data": [{"title": ...,
               "link": ...,
               "text": ...}],
              ...}

In the output, the title field may be missing; not all search engines return it.

This format preserves the connection between the text of the result
and its corresponding link.
