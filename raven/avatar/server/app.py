#!/usr/bin/python
"""Talkinghead server, for rendering an animated avatar for the AI.

Customized from `server.py` in the discontinued SillyTavern-extras.
Stripped everything except the `classify`, `embeddings`, `talkinghead` and `websearch` modules.

The `tts` module is new, based on Kokoro-82M. All old TTS options are gone.
This gives us lipsync for the talkinghead.
"""

# TODO: convert prints to use logger where appropriate

import argparse
import gc
import io
import json
import os
import pathlib
import secrets
import time
import traceback
from typing import Any, Dict, List, Union

from colorama import Fore, Style, init as colorama_init
import markdown

from flask import Flask, jsonify, request, abort, render_template_string, Response
from flask_cors import CORS
from flask_compress import Compress
from waitress import serve

import torch

from ..common import config  # default models
from ..common import postprocessor

from . import animator
from . import classify
from . import embed
from . import imagefx
from . import tts
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
ignore_auth = []  # endpoints (Python functions) whose web API we want not to require an API key

# --------------------------------------------------------------------------------
# Web API and its support functions

def is_authorize_ignored(request) -> bool:
    """Check whether the API endpoint called by `request` is in `ignore_auth`."""
    view_func = app.view_functions.get(request.endpoint)
    if view_func is not None:
        if view_func in ignore_auth:
            return True
    return False

def is_authorized(request) -> bool:
    """Check whether `request` is authorized.

    - When not in "--secure" mode, all requests are authorized.
    - HTTP OPTIONS requests are always authoriszed.
    - If the API endpoint being called is in the server's `ignore_auth`, the request is authorized.

    Otherwise: the correct API key must be present in the request for the request to be authorized.
    """
    if not args.secure:
        return True
    if request.method == "OPTIONS":  # The options check is required so CORS doesn't get angry
        return True
    if is_authorize_ignored(request):
        return True
    # Check if an API key is present and valid, otherwise return unauthorized
    return (getattr(request.authorization, "token", "") == api_key)

@app.before_request
def before_request():
    request.start_time = time.monotonic()
    try:
        if not is_authorized(request):
            print(f"{Fore.YELLOW}{Style.NORMAL}WARNING: Unauthorized access (missing or wrong API key) from {request.remote_addr}{Style.RESET_ALL}")
            response = jsonify({"error": "401: Invalid API key"})
            response.status_code = 401
            return response
    except Exception as e:
        print(f"{Fore.RED}{Style.NORMAL}Internal server error during API key check.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        return f"500 Internal Server Error\n{e}\n\n", 500

@app.after_request
def after_request(response):
    duration = time.monotonic() - request.start_time
    response.headers["X-Request-Duration"] = str(duration)  # seconds
    return response

# ----------------------------------------
# server metadata endpoints

@app.route("/", methods=["GET"])
def index():
    """Return usage documentation.

    No inputs.

    Output is suitable for rendering in a web browser.
    """
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

    No inputs.

    Output format is JSON::

        {"modules": ["modulename0",
                     ...]}

    Unlike SillyTavern-extras, `raven.avatar.server` always enables all modules.
    We provide the following:

    - classify
    - embeddings
    - talkinghead
    - tts (both a simple OpenAI-compatible speech endpoint, and our own)
    - websearch

    If any of these are missing, it means that there has been an error during startup that has prevented the module from loading.
    """
    modules = []
    if classify.is_available():
        modules.append("classify")
    if embed.is_available():
        modules.append("embeddings")
    if animator.is_available():
        modules.append("talkinghead")
    if tts.is_available():
        modules.append("tts")
    if websearch.is_available():
        modules.append("websearch")
    return jsonify({"modules": modules})

# ----------------------------------------
# module: classify

@app.route("/api/classify", methods=["POST"])
def api_classify():
    """Perform sentiment analysis (emotion classification) on the text posted in the request. Return the result.

    Input is JSON::

        {"text": "Blah blah blah."}

    Output is JSON::

        {"classification": [{"label": emotion0, "score": confidence0},
                            ...]}

    sorted by score, descending, so that the most probable emotion is first.
    """
    if not classify.is_available():
        abort(403, "Module 'classify' not running")

    data = request.get_json()

    if "text" not in data or not isinstance(data["text"], str):
        abort(400, 'api_classify: "text" is required')

    print("Classification input:", data["text"], sep="\n")
    classification = classify.classify_text(data["text"])
    print("Classification output:", classification, sep="\n")
    gc.collect()
    return jsonify({"classification": classification})

@app.route("/api/classify/labels", methods=["GET"])
def api_classify_labels():
    """Return the available classifier labels for text sentiment (character emotion).

    No inputs.

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
# module: embeddings

@app.route("/api/embeddings/compute", methods=["POST"])
def api_embeddings_compute():
    """For making vector DB keys. Compute the vector embedding of one or more sentences of text.

    Input is JSON::

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
    """
    if not embed.is_available():
        abort(403, "Module 'embeddings' not running")  # this is the only optional module
    data = request.get_json()
    if "text" not in data:
        abort(400, 'api_embeddings_compute: "text" is required')
    sentences: Union[str, List[str]] = data["text"]
    if not (isinstance(sentences, str) or (isinstance(sentences, list) and all(isinstance(x, str) for x in sentences))):
        abort(400, 'api_embeddings_compute: "text" must be string or array of strings')
    if isinstance(sentences, str):
        nitems = 1
    else:
        nitems = len(sentences)
    print(f"Computing vector embedding for {nitems} item{'s' if nitems != 1 else ''}")
    vectors = embed.embed_sentences(sentences)
    return jsonify({"embedding": vectors})

# ----------------------------------------
# module: imagefx

# TODO: add "/api/imagefx/upscale", too?

@app.route("/api/imagefx/process", methods=["POST"])
def api_imagefx_process():
    """Run an image through a postprocessor chain.

    This can be used e.g. for blurring a client-side background for the AI avatar,
    running the blur filter on the server's GPU.

    Input is POST, Content-Type "multipart/form-data", with two file attachments:

        "file": the actual image file (binary, any supported format)
        "json": the API call parameters, in JSON format.

    The parameters are::

        {"format": "png",
         "filters": [[filter0, {param0_name: value0, ...}],
                     ...]}

    Supported image formats (both input and output) are RGB/RGBA formats supported by Pillow,
    and QOI (Quite OK Image).

    If you need speed, and your client supports it, prefer the QOI format. Especially the
    encoder is dozens of times faster than PNG's, and compresses almost as tightly.

    To get supported filters, call the endpoint "/api/talkinghead/get_available_filters".
    Don't mind the name - the endpoint is available whenever at least one of "talkinghead"
    or "imagefx" is loaded.

    Output is an image with mimetype "image/<format>".
    """
    if not imagefx.is_available():
        abort(403, "Module 'imagefx' not running")

    try:
        file = request.files["file"]

        # TODO: Do we need to run this through a `BytesIO` to copy the data? Probably not?
        # The internet says that in some versions of Flask, touching most of the attributes
        # of a `FileStorage` causes a disk write to a temporary file, but `.stream` can be
        # safely accessed in-memory.
        parameters_filestorage = request.files["json"]
        buffer = io.BytesIO()
        buffer.write(parameters_filestorage.stream.read())
        parameters_bytes = buffer.getvalue()
        parameters_python = json.loads(parameters_bytes)

        # # Simpler way without `BytesIO`:
        # parameters_filestorage = request.files["json"]
        # parameters_bytes = parameters_filestorage.read()
        # parameters_python = json.loads(parameters_bytes)

        postprocessor_chain = parameters_python["filters"]
        format = parameters_python["format"]

        processed_image = imagefx.process(file.stream,
                                          output_format=format,
                                          postprocessor_chain=postprocessor_chain)
    except Exception as exc:
        abort(400, f"api_imagefx_process: failed, reason: {type(exc)}: {exc}")

    return Response(processed_image, mimetype=f"image/{format.lower()}")

# ----------------------------------------
# module: talkinghead

@app.route("/api/talkinghead/load", methods=["POST"])
def api_talkinghead_load():
    """Load the avatar sprite posted as a file in the request.

    Input is POST, Content-Type "multipart/form-data", with one file attachment, named "file".

    The file should be an RGBA image in a format that Pillow can read. It will be autoscaled to 512x512.

    No outputs.
    """
    if not animator.is_available():
        abort(403, "Module 'talkinghead' not running")

    file = request.files["file"]
    return animator.load_image_from_stream(file.stream)

@app.route("/api/talkinghead/load_emotion_templates", methods=["POST"])
def api_talkinghead_load_emotion_templates():
    """Load custom emotion templates for talkinghead, or reset to defaults.

    Input is JSON::

        {"emotion0": {"morph0": value0,
                      ...}
         ...}

    For details, see `Animator.load_emotion_templates` in `animator.py`.

    To reload server defaults, send a blank JSON.

    No outputs.
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

    No inputs, no outputs.

    A character must be loaded first; use '/api/talkinghead/load' to do that.

    To pause, use '/api/talkinghead/stop'.
    """
    if not animator.is_available():
        abort(403, "Module 'talkinghead' not running")
    return animator.start()

@app.route("/api/talkinghead/stop")
def api_talkinghead_stop():
    """Pause the avatar animation.

    No inputs, no outputs.

    To resume, use '/api/talkinghead/start'.
    """
    if not animator.is_available():
        abort(403, "Module 'talkinghead' not running")
    return animator.stop()

@app.route("/api/talkinghead/start_talking")
def api_talkinghead_start_talking():
    """Start the mouth animation for talking.

    No inputs, no outputs.

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

    No inputs, no outputs.

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

    Input is JSON::

        {"emotion_name": "curiosity"}

    where the key "emotion_name" is literal, and the value is the emotion to set.

    No outputs.

    There is no getter, by design. If the emotion state is meaningful to you,
    keep a copy in your frontend, and sync that to the server.
    """
    if not animator.is_available():
        abort(403, "Module 'talkinghead' not running")
    data = request.get_json()
    if "emotion_name" not in data or not isinstance(data["emotion_name"], str):
        abort(400, 'api_talkinghead_set_emotion: "emotion_name" is required')
    emotion_name = data["emotion_name"]
    return animator.set_emotion(emotion_name)

@app.route("/api/talkinghead/set_overrides", methods=["POST"])
def api_talkinghead_set_overrides():
    """Directly control the animator's morphs from the client side.

    Useful for lipsyncing.

    Input is JSON::

        {"morph0": value0,
         ...}

    To unset overrides, send a blank JSON.

    See `raven.avatar.editor` for available morphs. Value range for most morphs is [0, 1],
    and for morphs taking also negative values, it is [-1, 1].

    No outputs.

    There is no getter, by design. If the override state is meaningful to you,
    keep a copy in your frontend, and sync that to the server.
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

    No inputs.

    Output is a "multipart/x-mixed-replace" stream of video frames, each as an image file.
    The payload separator is "--frame".

    The file format can be set in the animator settings. The frames are always sent
    with the Content-Type and Content-Length headers set.
    """
    if not animator.is_available():
        abort(403, "Module 'talkinghead' not running")
    return animator.result_feed()
ignore_auth.append(api_talkinghead_result_feed)   # TODO: does this make sense?

@app.route("/api/talkinghead/get_available_filters")
def api_talkinghead_get_available_filters():
    """Get metadata of all available postprocessor filters and their available parameters.

    The intended audience of this endpoint is developers; this is useful for dynamically
    building an editor GUI for the postprocessor chain.

    No inputs.

    Output is JSON::

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
    if not (animator.is_available() or imagefx.is_available()):
        abort(403, "Neither of modules 'imagefx' or 'talkinghead' is running")
    return jsonify({"filters": postprocessor.Postprocessor.get_filters()})

# ----------------------------------------
# module: tts

def _list_voices():
    if not tts.is_available():
        abort(403, "Module 'tts' not running")
    return jsonify({"voices": tts.get_voices()})

@app.route("/api/tts/list_voices")
def api_tts_list_voices():
    """Text to speech.

    Get list of voice names from the speech synthesizer.

    No inputs.

    Output is JSON::

        {"voices": [voice0, ...]}

    See:

        https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
    """
    return _list_voices()

@app.route("/api/tts/speak", methods=["POST"])
def api_tts_speak():
    """Text to speech.

    NOTE: `espeak-ng` must be installed manually because it's a non-Python dependency.
    The TTS engine Kokoro-82M uses it as a phonemizer fallback for out-of-dictionary words
    in English, as well as for some non-English languages.

    Input is JSON::

        {"text": "Blah blah blah.",
         "voice": "af_bella",
         "speed": 1.0,
         "format": "mp3",
         "get_metadata": true,
         "stream": false}

    Only the "text" field is mandatory.

    For available voices, call the endpoint "/api/tts/list_voices".

    For available formats, see `tts.text_to_speech`.

    The audio file is returned as the response content. Content-Type is "audio/<format>", e.g. "audio/mp3".

    If "get_metadata" is true, an extra header "x-word-timestamps" is returned, with JSON data
    containing word-level timestamps and phonemes:

        [{"word": "reasonably",
          "phonemes": "ɹˈizənəbli" (URL-encoded to ASCII with percent-escaped UTF-8),
          "start_time": 2.15,
          "end_time": 2.75},
         ...]

    The start and end times are measured in seconds from start of audio.

    This data is useful for lipsyncing and captioning.

    Note "get_metadata" currently only works in English (we use a local Kokoro with only English installed).
    """
    if not tts.is_available():
        abort(403, "Module 'tts' not running")

    data = request.get_json()

    if "text" not in data or not isinstance(data["text"], str):
        abort(400, 'api_tts_speak: "text" is required')
    text = data["text"]

    voice = data.get("voice", "af_bella")  # TODO: sane default
    speed = float(data.get("speed", 1.0))
    format = data.get("format", "mp3")
    get_metadata = data.get("get_metadata", True)  # bool flag
    stream = data.get("stream", False)  # bool flag

    try:
        speed = float(speed)
    except ValueError:
        abort(400, 'api_tts_speak: "speed", if specified, should be a number')

    try:
        response = tts.text_to_speech(voice=voice,
                                      text=text,
                                      speed=speed,
                                      format=format,
                                      get_metadata=get_metadata,
                                      stream=stream)
        return response
    except Exception as exc:
        traceback.print_exc()
        abort(400, f"api_tts_speak: failed, reason: {type(exc)}: {exc}")

@app.route("/v1/audio/voices")
def api_v1_audio_voices():
    """OpenAI compatible endpoint, for SillyTavern; does the exact same thing as "/api/tts/list_voices"."""
    return _list_voices()

@app.route("/v1/audio/speech")
def api_v1_audio_speech():
    """OpenAI compatible endpoint, for SillyTavern; does the exact same thing as "/api/tts/speak".

    However, this endpoint does not support "get_metadata", because it's not part of the OAI format.
    If you need lipsyncing or captioning, use "/api/tts/speak" instead.

    Input is JSON::

        {"input": "Blah blah blah.",
         "voice": "af_bella",
         "speed": 1.0,
         "response_format": "mp3",
         "stream": false}

    The audio file is returned as the response content. Content-Type is "audio/<format>", e.g. "audio/mp3".
    """
    if not tts.is_available():
        abort(403, "Module 'tts' not running")

    data = request.get_json()

    if "input" not in data or not isinstance(data["input"], str):
        abort(400, 'api_v1_audio_speech: "input" is required')
    text = data["input"]

    voice = data.get("voice", "af_bella")  # TODO: sane default
    speed = float(data.get("speed", 1.0))
    format = data.get("response_format", "mp3")
    stream = data.get("stream", False)  # bool flag
    # Kokoro-FastAPI-like "return_download_link" flag is not supported

    try:
        speed = float(speed)
    except ValueError:
        abort(400, 'api_v1_audio_speech: "speed", if specified, should be a number')

    try:
        response = tts.text_to_speech(voice=voice,
                                      text=text,
                                      speed=speed,
                                      format=format,
                                      get_metadata=False,
                                      stream=stream)
        return response
    except Exception as exc:
        traceback.print_exc()
        abort(400, f"api_v1_audio_speech: failed, reason: {type(exc)}: {exc}")

# ----------------------------------------
# module: websearch

def _websearch_impl():
    data = request.get_json()

    if "query" not in data or not isinstance(data["query"], str):
        abort(400, '"query" is required')
    query = data["query"]

    engine = data.get("engine", "duckduckgo")
    max_links = data.get("max_links", 10)

    if engine not in ("duckduckgo", "google"):
        abort(400, '"engine", if provided, must be one of "duckduckgo", "google"')

    return websearch.search(query, engine=engine, max_links=max_links)

# legacy ST-compatible websearch endpoint
@app.route("/api/websearch", methods=["POST"])
def api_websearch():
    """Perform a web search with the query posted in the request.

    This is the SillyTavern compatible legacy endpoint.
    For new clients, prefer to use "/api/websearch2", which gives
    structured output and has a "max_links" option.

    Input is JSON::

        {"query": "what is the airspeed velocity of an unladen swallow",
         "engine": "duckduckgo"}

    In the input, "engine" is optional. Valid values are "duckduckgo" (default)
    and "google".

    Output is JSON:

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

    Input is JSON::

        {"query": "what is the airspeed velocity of an unladen swallow",
         "engine": "duckduckgo",
         "max_links": 10}

    In the input, some fields are optional:

      - "engine": valid values are "duckduckgo" (default) and "google".
      - "max_links": default 10.

    The "max_links" field is a hint; the search engine may return more
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
    """
    if not websearch.is_available():
        abort(403, "Module 'websearch' not running")
    preformatted_text, structured_results = _websearch_impl()
    output = {"results": preformatted_text,
              "data": structured_results}
    return jsonify(output)


# --------------------------------------------------------------------------------
# Main program

# NOTE SillyTavern-Extras users: settings for module enable/disable and which compute device to use for each module have been moved to `raven/avatar/common/config.py`.

# ----------------------------------------
# Parse command-line arguments

parser = argparse.ArgumentParser(
    prog="Raven-server", description="Server for specialized local AI models, based on the discontinued SillyTavern-extras"
)
parser.add_argument(
    "--port", type=int, help=f"Specify the port on which the application is hosted (default {config.DEFAULT_PORT})"
)
parser.add_argument(
    "--listen", action="store_true", help="Host the app on the local network (if not set, the server is visible to localhost only)"
)
parser.add_argument(
    "--secure", action="store_true", help="Require an API key (will be auto-created first time, and printed to console each time on server startup)"
)

parser.add_argument("--max-content-length", help="Set the max content length for the Flask app config.")

args = parser.parse_args()

port = args.port if args.port else config.DEFAULT_PORT
host = "0.0.0.0" if args.listen else "localhost"

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

# Configure Flask
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
max_content_length = args.max_content_length if args.max_content_length else None
if max_content_length is not None:
    print("Setting MAX_CONTENT_LENGTH to", max_content_length, "Mb")
    app.config["MAX_CONTENT_LENGTH"] = int(max_content_length) * 1024 * 1024

# ----------------------------------------
# Initialize enabled modules

cuda_info_shown = set()
def get_device_and_dtype(record: Dict[str, Any]) -> (str, torch.dtype):
    global cuda_info_shown

    device_string = record["device_string"]
    torch_dtype = record.get("dtype", None)  # not all modules have a specifiable dtype

    if device_string.startswith("cuda"):  # Nvidia
        if not torch.cuda.is_available():
            print(f"{Fore.YELLOW}{Style.BRIGHT}CUDA backend specified in config (device string '{device_string}'), but CUDA not available. Using CPU instead.{Style.RESET_ALL}")
            device_string = "cpu"
        else:
            if device_string not in cuda_info_shown:
                cuda_info_shown.add(device_string)
                print(f"Device info for GPU '{Fore.GREEN}{Style.BRIGHT}{device_string}{Style.RESET_ALL}' ({torch.cuda.get_device_name(device_string)}):")
                print(f"    {torch.cuda.get_device_properties(device_string)}")
                print(f"    Compute capability {'.'.join(str(x) for x in torch.cuda.get_device_capability(device_string))}")
                print(f"    Detected CUDA version {torch.version.cuda}")

    elif device_string.startswith("mps"):  # Mac, Apple Metal Performance Shaders
        if not torch.backends.mps.is_available():
            print(f"{Fore.YELLOW}{Style.BRIGHT}MPS backend specified in config (device string '{device_string}'), but MPS not available. Using CPU instead.{Style.RESET_ALL}")
            device_string = "cpu"
        # TODO: Torch MPS backend info?

    if device_string == "cpu":  # no "elif" because also as fallback if CUDA/MPS wasn't available
        if torch_dtype is torch.float16:
            print(f"{Fore.YELLOW}{Style.BRIGHT}dtype is set to torch.float16, but device 'cpu' does not support half precision. Using torch.float32 instead.{Style.RESET_ALL}")
            torch_dtype = torch.float32

    return device_string, torch_dtype

def init_server_modules():  # keep global namespace clean
    if (record := config.SERVER_ENABLED_MODULES.get("classify", None)) is not None:
        device_string, torch_dtype = get_device_and_dtype(record)
        classify.init_module(config.CLASSIFICATION_MODEL, device_string, torch_dtype)
    if (record := config.SERVER_ENABLED_MODULES.get("embeddings", None)) is not None:
        device_string, torch_dtype = get_device_and_dtype(record)
        embed.init_module(config.EMBEDDING_MODEL, device_string, torch_dtype)
    if (record := config.SERVER_ENABLED_MODULES.get("imagefx", None)) is not None:
        device_string, torch_dtype = get_device_and_dtype(record)
        imagefx.init_module(device_string, torch_dtype)
    if (record := config.SERVER_ENABLED_MODULES.get("talkinghead", None)) is not None:
        device_string, torch_dtype = get_device_and_dtype(record)
        # One of 'standard_float', 'separable_float', 'standard_half', 'separable_half'.
        # FP16 boosts the rendering performance by ~1.5x, but is only supported on GPU.
        tha3_model_variant = "separable_half" if torch_dtype is torch.float16 else "separable_float"
        animator.init_module(device_string, tha3_model_variant)
    if config.SERVER_ENABLED_MODULES.get("tts", None) is not None:
        device_string, _ = get_device_and_dtype(record)
        tts.init_module(device_string)
    if config.SERVER_ENABLED_MODULES.get("websearch", None) is not None:  # no device/dtype settings; if a blank record exists, this module is enabled.
        websearch.init_module()
init_server_modules()

# ----------------------------------------
# Start serving

print(f"{Fore.GREEN}{Style.BRIGHT}Starting server{Style.RESET_ALL}")

serve(app, host=host, port=port)

def main():  # TODO: we don't really need this; it's just for console_scripts so that we can provide a command-line entrypoint.
    pass
