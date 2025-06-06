#!/usr/bin/python
"""Talkinghead server, for rendering an animated avatar for the AI.

Customized from `server.py` in the discontinued SillyTavern-extras.
Stripped everything except the `talkinghead`, `classify`, and `embeddings` modules.

The first two are now always loaded, and `embeddings` can be enabled for use with SillyTavern.
(Note this is a separate process, and Raven runs its embeddings in its main process.)
"""

# TODO: convert prints to use logger where appropriate

import argparse
import gc
import os
import pathlib
import secrets
import sys
import time
from typing import List, Union

from colorama import Fore, Style, init as colorama_init
import markdown

from flask import Flask, jsonify, request, abort, render_template_string
from flask_cors import CORS
from flask_compress import Compress

import torch

from ..common import config  # default models
from ..common import postprocessor

from . import animator
from . import classify
from . import embed
from . import util
from . import websearch

# --------------------------------------------------------------------------------
# Inits that must run before we proceed any further

colorama_init()

app = Flask(__name__)
CORS(app)  # allow cross-domain requests
Compress(app)  # compress responses

# will be populated later
args = []  # command-line args

api_key = None  # secure mode
ignore_auth = []  # features that do not need an API key

# --------------------------------------------------------------------------------
# General utilities

def is_authorize_ignored(request):
    view_func = app.view_functions.get(request.endpoint)
    if view_func is not None:
        if view_func in ignore_auth:
            return True
    return False

# --------------------------------------------------------------------------------
# Web API and its support functions

@app.before_request
def before_request():
    # Request time measuring
    request.start_time = time.time()

    # Checks if an API key is present and valid, otherwise return unauthorized
    # The options check is required so CORS doesn't get angry
    try:
        if request.method != 'OPTIONS' and args.secure and not is_authorize_ignored(request) and getattr(request.authorization, 'token', '') != api_key:
            print(f"{Fore.RED}{Style.NORMAL}WARNING: Unauthorized API key access from {request.remote_addr}{Style.RESET_ALL}")
            response = jsonify({'error': '401: Invalid API key'})
            response.status_code = 401
            return response
    except Exception as e:
        print(f"API key check error: {e}")
        return "401 Unauthorized\n{}\n\n".format(e), 401

@app.after_request
def after_request(response):
    duration = time.time() - request.start_time
    response.headers["X-Request-Duration"] = str(duration)
    return response

# ----------------------------------------
# general utilities

@app.route("/", methods=["GET"])
def index():
    """Return usage documentation, to be rendered in a web browser."""
    with open(os.path.join(os.path.dirname(__file__), "..", "README.md"), "r", encoding="utf8") as f:
        content = f.read()
    return render_template_string(markdown.markdown(content, extensions=["tables"]))

@app.route("/health", methods=["GET"])
def health():
    """A simple ping endpoint for clients to check that the server is running.

    No inputs, no outputs - if you get a 200 OK, it means the server heard you.
    """
    return "OK"

@app.route("/api/modules", methods=["GET"])
def get_modules():
    """Get a list of enabled modules.

    Output format is JSON::

        {"modules": ["modulename0",
                     ...]}

    Unlike SillyTavern-extras, `raven.avatar.server` always enables the following modules:

    - classify
    - talkinghead
    - websearch

    If any of these are missing, it means that there has been an error during startup that has prevented the module from loading.

    Optionally, the following modules can be enabled via a command-line switch when the server is started:

    - embeddings
    """
    modules = []
    if animator.is_available():
        modules.append("talkinghead")
    if classify.is_available():
        modules.append("classify")
    if websearch.is_available():
        modules.append("websearch")
    if args.embeddings and embed.is_available():  # the only optional module; embeddings API endpoint enabled?
        modules.append("embeddings")
    return jsonify({"modules": modules})

# ----------------------------------------
# embeddings

@app.route("/api/embeddings/compute", methods=["POST"])
def api_embeddings_compute():
    """For making vector DB keys. Compute the vector embedding of one or more sentences of text.

    Input format is JSON::

        {"text": "Blah blah blah."}

    or::

        {"text": ["Blah blah blah.",
                  ...]}

    Output is also JSON::

        {"embedding": array}

    or::

        {"embedding": [array0,
                       ...]}

    respectively.

    This is the Extras backend for computing embeddings in the Vector Storage builtin extension.
    """
    if not embed.is_available():
        abort(403, "Module 'embeddings' not running")  # this is the only optional module
    data = request.get_json()
    if "text" not in data:
        abort(400, '"text" is required')
    sentences: Union[str, List[str]] = data["text"]
    if not (isinstance(sentences, str) or (isinstance(sentences, list) and all(isinstance(x, str) for x in sentences))):
        abort(400, '"text" must be string or array of strings')
    if isinstance(sentences, str):
        nitems = 1
    else:
        nitems = len(sentences)
    print(f"Computing vector embedding for {nitems} item{'s' if nitems != 1 else ''}")
    vectors = embed.embed_sentences(sentences)
    return jsonify({"embedding": vectors})

# ----------------------------------------
# websearch

def _websearch_impl():
    data = request.get_json()
    if "query" not in data or not isinstance(data["query"], str):
        abort(400, '"query" is required')

    query = data["query"]
    engine = data["engine"] if "engine" in data else "duckduckgo"
    max_links = data["max_links"] if "max_links" in data else 10

    if engine not in ("duckduckgo", "google"):
        abort(400, '"engine", if provided, must be one of "duckduckgo", "google"')

    return websearch.search(query, engine=engine, max_links=max_links)

# legacy ST-compatible websearch endpoint
@app.route("/api/websearch", methods=["POST"])
def api_websearch():
    """Perform a web search with the query posted in the request.

    SillyTavern compatible endpoint.

    Input format is JSON::

        {"query": "what is the airspeed velocity of an unladen swallow",
         "engine": "duckduckgo"}

    In the input, "engine" is optional. Valid values are "duckduckgo" (default)
    and "google".

    Output is also JSON:

        {"results": preformatted_text,
         "links": [link0, ...]}

    where the "links" field contains a list of all links to the search results.
    """
    if not websearch.is_available():
        abort(403, "Module 'websearch' not running")
    preformatted_text, structured_results = _websearch_impl()
    output = {"results": preformatted_text,
              "links": [item["link"] for item in structured_results]}
    return jsonify(output)

# for Raven
@app.route("/api/websearch2", methods=["POST"])
def api_websearch2():
    """Perform a web search with the query posted in the request.

    Input format is JSON::

        {"query": "what is the airspeed velocity of an unladen swallow",
         "engine": "duckduckgo",
         "max_links": 10}

    In the input, some fields are optional:

      - "engine": valid values are "duckduckgo" (default) and "google".
      - "max_links": default 10.

    The "max_links" field is a hint; the search engine may return more
    results, especially if you set it to a small value (e.g. 3).

    Output is also JSON:

        {"results": preformatted_text,
         "data": [{"title": ...,
                   "link": ...,
                   "text": ...}],
                  ...}

    In the output, title is optional; not all search engines return it.

    This format preserves the connection between the text of the result
    and its corresponding link, which is convenient for citation mechanisms.
    """
    if not websearch.is_available():
        abort(403, "Module 'websearch' not running")
    preformatted_text, structured_results = _websearch_impl()
    output = {"results": preformatted_text,
              "data": structured_results}
    return jsonify(output)

# ----------------------------------------
# classify

@app.route("/api/classify", methods=["POST"])
def api_classify():
    """Perform sentiment analysis (emotion classification) on the text posted in the request. Return the result.

    Output is JSON::

        {"classification": [{"label": emotion0, "score": confidence0},
                            ...]}

    sorted by score, descending, so that the most probable emotion is first.
    """
    if not classify.is_available():
        abort(403, "Module 'classify' not running")

    data = request.get_json()

    if "text" not in data or not isinstance(data["text"], str):
        abort(400, '"text" is required')

    print("Classification input:", data["text"], sep="\n")
    classification = classify.classify_text(data["text"])
    print("Classification output:", classification, sep="\n")
    gc.collect()
    return jsonify({"classification": classification})

@app.route("/api/classify/labels", methods=["GET"])
def api_classify_labels():
    """Return the available classifier labels for text sentiment (character emotion).

    Output is JSON::

        {"labels": [emotion0,
                    ...]}

    The actual labels depend on the classifier model.
    """
    if not classify.is_available():
        abort(403, "Module 'classify' not running")

    classification = classify.classify_text("")
    labels = [x["label"] for x in classification]
    return jsonify({"labels": labels})

# ----------------------------------------
# talkinghead

@app.route("/api/talkinghead/load", methods=["POST"])
def api_talkinghead_load():
    """Load the avatar sprite posted as a file in the request.

    The request should be posted as "multipart/form-data", with one file attachment, named "file".

    The file should be an RGBA image in a format that Pillow can read. It will be autoscaled to 512x512.
    """
    if not animator.is_available():
        abort(403, "Module 'talkinghead' not running")

    file = request.files['file']
    return animator.load_image_from_stream(file.stream)

@app.route("/api/talkinghead/load_emotion_templates", methods=["POST"])
def api_talkinghead_load_emotion_templates():
    """Load custom emotion templates for talkinghead, or reset to defaults.

    Input format is JSON::

        {"emotion0": {"morph0": value0,
                      ...}
         ...}

    For details, see `Animator.load_emotion_templates` in `animator.py`.

    To reload server defaults, send a blank JSON.

    This API endpoint becomes available after the talkinghead has been launched.
    """
    if not animator.is_available():
        abort(403, "Module 'talkinghead' not running")

    data = request.get_json()
    if not len(data):
        data = None  # sending `None` to talkinghead will reset to defaults
    animator.global_animator_instance.load_emotion_templates(data)
    return "OK"

@app.route("/api/talkinghead/load_animator_settings", methods=["POST"])
def api_talkinghead_load_animator_settings():
    """Load custom settings for talkinghead animator and postprocessor, or reset to defaults.

    Input format is JSON::

        {"name0": value0,
         ...}

    For details, see `Animator.load_animator_settings` in `animator.py`.

    To reload server defaults, send a blank JSON.

    This API endpoint becomes available after the talkinghead has been launched.
    """
    if not animator.is_available():
        abort(403, "Module 'talkinghead' not running")

    data = request.get_json()
    if not len(data):
        data = None  # sending `None` to talkinghead will reset to defaults
    animator.global_animator_instance.load_animator_settings(data)
    return "OK"

@app.route("/api/talkinghead/start")
def api_talkinghead_start():
    """Start the avatar animation.

    A character must be loaded first; use '/api/talkinghead/load'.

    To pause, use '/api/talkinghead/stop'.
    """
    if not animator.is_available():
        abort(403, "Module 'talkinghead' not running")
    return animator.start()

@app.route("/api/talkinghead/stop")
def api_talkinghead_stop():
    """Pause the avatar animation.

    To resume, use '/api/talkinghead/start'.
    """
    if not animator.is_available():
        abort(403, "Module 'talkinghead' not running")
    return animator.stop()

@app.route("/api/talkinghead/start_talking")
def api_talkinghead_start_talking():
    """Start the mouth animation for talking.

    This is the generic, non-lipsync animation that randomizes the mouth.

    This is useful for applications without actual voiced audio, such as
    an LLM when TTS is offline, or a low-budget visual novel.

    For speech with automatic lip sync, see `tts_speak_lipsynced`.
    """
    if not animator.is_available():
        abort(403, "Module 'talkinghead' not running")
    return animator.start_talking()

@app.route("/api/talkinghead/stop_talking")
def api_talkinghead_stop_talking():
    """Stop the mouth animation for talking.

    This is the generic, non-lipsync animation that randomizes the mouth.

    This is useful for applications without actual voiced audio, such as
    an LLM when TTS is offline, or a low-budget visual novel.

    For speech with automatic lip sync, see `tts_speak_lipsynced`.
    """
    if not animator.is_available():
        abort(403, "Module 'talkinghead' not running")
    return animator.stop_talking()

@app.route("/api/talkinghead/set_emotion", methods=["POST"])
def api_talkinghead_set_emotion():
    """Set talkinghead character emotion to that posted in the request.

    Input format is JSON::

        {"emotion_name": "curiosity"}

    where the key "emotion_name" is literal, and the value is the emotion to set.

    There is no getter, because SillyTavern keeps its state in the frontend
    and the plugins only act as slaves (in the technological sense of the word).
    """
    if not animator.is_available():
        abort(403, "Module 'talkinghead' not running")
    data = request.get_json()
    if "emotion_name" not in data or not isinstance(data["emotion_name"], str):
        abort(400, '"emotion_name" is required')
    emotion_name = data["emotion_name"]
    return animator.set_emotion(emotion_name)

@app.route("/api/talkinghead/set_overrides", methods=["POST"])
def api_talkinghead_set_overrides():
    """Directly control the animator's morphs from the client side.

    Useful for lipsyncing.

    Input format is JSON::

        {"morph0": value0,
         ...}

    To unset overrides, send a blank JSON.

    See `raven.avatar.editor` for available morphs. Value range for most morphs is [0, 1],
    and for morphs taking also negative values, it is [-1, 1].

    This API endpoint becomes available after the talkinghead has been launched.
    """
    if not animator.is_available():
        abort(403, "Module 'talkinghead' not running")
    data = request.get_json()
    if not len(data):
        data = {}
    try:
        animator.global_animator_instance.set_overrides(data)
    except Exception as exc:
        abort(400, f"api_talkinghead_set_overrides: failed, reason: {type(exc)}: {exc}")
    return "OK"

@app.route("/api/talkinghead/result_feed")
def api_talkinghead_result_feed():
    """Video output.

    A "multipart/x-mixed-replace" stream of video frames, each as an image file.
    The payload separator is "--frame".

    The file format can be set in the animator settings. The frames are always sent
    with the Content-Type and Content-Length headers set.
    """
    if not animator.is_available():
        abort(403, "Module 'talkinghead' not running")
    return animator.result_feed()

@app.route("/api/talkinghead/get_available_filters")
def api_talkinghead_get_available_filters():
    """Get metadata of all available postprocessor filters and their available parameters.

    The intended audience of this endpoint is developers; this is useful for dynamically
    building an editor GUI for the postprocessor chain.

    The output format is JSON::

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
    where each component is in the range [0, 1].

    You can detect the type from the default value.
    """
    if not animator.is_available():
        abort(403, "Module 'talkinghead' not running")
    return jsonify({"filters": postprocessor.Postprocessor.get_filters()})

# ----------------------------------------
# Script arguments

parser = argparse.ArgumentParser(
    prog="Raven-avatar", description="Talkinghead (THA3) server based on the discontinued SillyTavern-extras"
)
parser.add_argument(
    "--port", type=int, help=f"Specify the port on which the application is hosted (default {config.DEFAULT_PORT})"
)
parser.add_argument(
    "--listen", action="store_true", help="Host the app on the local network"
)
parser.add_argument(
    "--secure", action="store_true", help="Require an API key (will be auto-created first time, and printed to console each time on server startup)"
)

parser.add_argument("--cpu", action="store_true", help="Run the classify and embeddings models on the CPU")
parser.add_argument("--cuda", action="store_false", dest="cpu", help="Run the classify and embeddings models on the GPU (default)")
parser.add_argument("--cuda-device", help="Specify the CUDA device to use")
parser.add_argument("--mps", "--apple", "--m1", "--m2", action="store_false", dest="cpu", help="Run the classify and embeddings models on Apple Silicon")
parser.set_defaults(cpu=False)

parser.add_argument(
    "--classification-model", help=f"Load a custom text classification model (default '{config.DEFAULT_CLASSIFICATION_MODEL}')"
)

parser.add_argument("--embeddings", action="store_true", help="Load the text embedder (fast API endpoint for SillyTavern)")
parser.add_argument("--embedding-model", help=f"Load a custom text embedding model (default '{config.DEFAULT_EMBEDDING_MODEL}')")
parser.set_defaults(embeddings=False)

parser.add_argument("--talkinghead-cpu", action="store_true", help="Run the avatar animator on the CPU")
parser.add_argument("--talkinghead-gpu", dest="talkinghead_cpu", action="store_false", help="Run the avatar animator on the GPU (default)")
parser.add_argument(
    "--talkinghead-model", type=str, help="The THA3 model to use. 'float' models are fp32, 'half' are fp16. 'auto' (default) picks fp16 for GPU and fp32 for CPU.",
    required=False, default="auto",
    choices=["auto", "standard_float", "separable_float", "standard_half", "separable_half"],
)
parser.add_argument(
    "--talkinghead-models", metavar="HFREPO",
    type=str, help="If THA3 models are not yet installed, use the given HuggingFace repository to install them (default 'OktayAlpk/talking-head-anime-3').",
    default="OktayAlpk/talking-head-anime-3"
)

parser.add_argument("--max-content-length", help="Set the max content length for the Flask app config.")

args = parser.parse_args()

port = args.port if args.port else config.DEFAULT_PORT
host = "0.0.0.0" if args.listen else "localhost"
classification_model = args.classification_model if args.classification_model else config.DEFAULT_CLASSIFICATION_MODEL
embedding_model = args.embedding_model if args.embedding_model else config.DEFAULT_EMBEDDING_MODEL

# ----------------------------------------
# Flask init

app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
max_content_length = args.max_content_length if args.max_content_length else None
if max_content_length is not None:
    print("Setting MAX_CONTENT_LENGTH to", max_content_length, "Mb")
    app.config["MAX_CONTENT_LENGTH"] = int(max_content_length) * 1024 * 1024

# ----------------------------------------
# Modules init

cuda_device = config.DEFAULT_CUDA_DEVICE if not args.cuda_device else args.cuda_device
device_string = cuda_device if (torch.cuda.is_available() and not args.cpu) else 'mps' if (torch.backends.mps.is_available() and not args.cpu) else 'cpu'
torch_dtype = torch.float32 if (device_string != cuda_device) else torch.float16  # float32 on CPU, float16 on GPU

if not torch.cuda.is_available() and not args.cpu:
    print(f"{Fore.YELLOW}{Style.BRIGHT}torch-cuda is not supported on this device.{Style.RESET_ALL}")
    if not torch.backends.mps.is_available() and not args.cpu:
        print(f"{Fore.YELLOW}{Style.BRIGHT}torch-mps is not supported on this device.{Style.RESET_ALL}")

if device_string.startswith("cuda") and torch.cuda.is_available():
    print(f"Device info for GPU '{Fore.GREEN}{Style.BRIGHT}{device_string}{Style.RESET_ALL}' ({torch.cuda.get_device_name(device_string)}):")
    print(f"    {torch.cuda.get_device_properties(device_string)}")
    print(f"    Compute capability {'.'.join(str(x) for x in torch.cuda.get_device_capability(device_string))}")
    print(f"    Detected CUDA version {torch.version.cuda}")

# --------------------
# Websearch

websearch.init_module()

# --------------------
# Embeddings

# The "embeddings" module is only provided for compatibility with the discontinued SillyTavern-extras,
# to provide a fast (GPU-accelerated, or at least CPU-native) embeddings API endpoint for SillyTavern.
#
# Raven loads its embedding module in the main app, not in the `avatar` subapp.
#
# So this is optional, and off by default.
#
if args.embeddings:
    embed.init_module(embedding_model, device_string)

# --------------------
# Classify

classify.init_module(classification_model, device_string, torch_dtype)

# --------------------
# Talkinghead

talkinghead_path = pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "vendor")).expanduser().resolve()
print(f"Talkinghead is installed at '{str(talkinghead_path)}'")

sys.path.append(str(talkinghead_path))  # The vendored code from THA3 expects to find the `tha3` module at the top level of the module hierarchy

avatar_device = cuda_device if not args.talkinghead_cpu else "cpu"
model = args.talkinghead_model
if model == "auto":  # default
    # FP16 boosts the rendering performance by ~1.5x, but is only supported on GPU.
    model = "separable_half" if not args.talkinghead_cpu else "separable_float"
print(f"Initializing {Fore.GREEN}{Style.BRIGHT}avatar{Style.RESET_ALL} on device '{Fore.GREEN}{Style.BRIGHT}{avatar_device}{Style.RESET_ALL}' with model '{Fore.GREEN}{Style.BRIGHT}{model}{Style.RESET_ALL}'...")

# Install the THA3 models if needed
tha3_models_path = str(talkinghead_path / "tha3" / "models")
util.maybe_install_models(hf_reponame=args.talkinghead_models, modelsdir=tha3_models_path)

# avatar_device: choices='The device to use for PyTorch ("cuda" for GPU, "cpu" for CPU).'
# model: choices=['standard_float', 'separable_float', 'standard_half', 'separable_half'],
animator.init_module(avatar_device, model)

# ----------------------------------------
# Start app

print(f"{Fore.GREEN}{Style.BRIGHT}Starting server{Style.RESET_ALL}")

# Read an API key from an already existing file. If that file doesn't exist, create it.
if args.secure:
    config_dir = pathlib.Path(config.config_base_dir).expanduser().resolve()

    try:
        with open(config_dir / "api_key.txt", "r") as txt:
            api_key = txt.read().replace('\n', '')
    except Exception:
        api_key = secrets.token_hex(5)
        with open(config_dir / "api_key.txt", "w") as txt:
            txt.write(api_key)

    print(f"{Fore.YELLOW}{Style.BRIGHT}Your API key is {api_key}{Style.RESET_ALL}")
else:
    print(f"{Fore.YELLOW}{Style.BRIGHT}No API key, accepting all requests.{Style.RESET_ALL} (use --secure to require an API key)")

ignore_auth.append(api_talkinghead_result_feed)   # TODO: does this make sense?

from waitress import serve
serve(app, host=host, port=port)

def main():  # TODO: we don't really need this; it's just for console_scripts so that we can provide a command-line entrypoint.
    pass
