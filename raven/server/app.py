#!/usr/bin/python
"""WSGI/HTTP server, serving various AI components, mainly to Raven-<app-of-the-day> on localhost.

Customized from `server.py` in the discontinued SillyTavern-extras.

Contains an animated avatar mechanism for an AI character (continued from Talkinghead in ST-Extras).

The `tts` module is new, based on Kokoro-82M. All old TTS options are gone. This gives us lipsync for the avatar.
"""

# TODO: convert prints to use logger where appropriate

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from .. import __version__

logger.info(f"Raven-server {__version__} starting.")

logger.info("Loading libraries...")
from unpythonic import timer
with timer() as tim:
    import argparse
    import gc
    import importlib
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

    from .. import __version__

    from ..common import deviceinfo
    from ..common.video.postprocessor import Postprocessor  # available image filters

    from .modules import avatar
    from .modules import classify
    from .modules import embeddings
    from .modules import imagefx
    from .modules import natlang
    from .modules import sanitize
    from .modules import summarize
    from .modules import translate
    from .modules import tts
    from .modules import websearch
logger.info(f"Libraries loaded in {tim.dt:0.6g}s.")

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
    - If the API endpoint being called is in the server's `ignore_auth`, the request is always authorized.

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

def unpack_parameters_from_json_file_attachment(stream) -> Dict[str, Any]:
    """Return API call parameters as `dict`, that came in the request as a JSON file.

    `stream`: the `request.files["my_tag"].stream`.

    Returns a dictionary `{param_name0: value0, ...}`.

    This is meant for endpoints that receive "multipart/form-data" because they need a file input,
    but also simultenously need a JSON input to pass some API call parameters.

    The counterpart is `raven.client.api.pack_parameters_into_json_file_attachment`.
    """
    # TODO: Do we need to run this through a `BytesIO` to copy the data? Probably not?
    # The internet says that in some versions of Flask, touching most of the attributes
    # of a `FileStorage` causes a disk write to a temporary file, but `.stream` can be
    # safely accessed in-memory.
    buffer = io.BytesIO()
    buffer.write(stream.read())
    parameters_bytes = buffer.getvalue()
    parameters_python = json.loads(parameters_bytes)

    # # Simpler way without `BytesIO`:
    # parameters_filestorage = request.files["json"]
    # parameters_bytes = parameters_filestorage.read()
    # parameters_python = json.loads(parameters_bytes)

    return parameters_python

# ----------------------------------------
# server metadata endpoints

@app.route("/", methods=["GET"])
def index():
    """Return usage documentation.

    No inputs.

    Output is suitable for rendering in a web browser.
    """
    with open(pathlib.Path(os.path.join(os.path.dirname(__file__), "README.md")).expanduser().resolve(), "r", encoding="utf8") as f:
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
    """
    modules = []
    if avatar.is_available():
        modules.append("avatar")
    if classify.is_available():
        modules.append("classify")
    if embeddings.is_available():
        modules.append("embeddings")
    if imagefx.is_available():
        modules.append("imagefx")
    if natlang.is_available():
        modules.append("natlang")
    if sanitize.is_available():
        modules.append("sanitize")
    if summarize.is_available():
        modules.append("summarize")
    if translate.is_available():
        modules.append("translate")
    if tts.is_available():
        modules.append("tts")
    if websearch.is_available():
        modules.append("websearch")
    return jsonify({"modules": modules})

# ----------------------------------------
# module: avatar

@app.route("/api/avatar/load", methods=["POST"])
def api_avatar_load():
    """Start a new avatar instance, and load the avatar sprite posted as a file in the request.

    Input is POST, Content-Type "multipart/form-data", with one file attachment, named "file".

    The file should be an RGBA image in a format that Pillow can read. It will be autoscaled to 512x512.

    Optionally, there may be more file attachments, one for each add-on cel. The attachment name
    for each is the cel name. For supported cels, see `supported_cels` in `raven.server.avatarutil`,
    and for animefx cels, `raven.server.config`.

    Output is JSON::

        {"instance_id": "some_important_string"}

    Here the important string is the instance ID the new avatar instance. Use this instance ID in the
    other avatar API endpoints to target the operations to this instance.
    """
    if not avatar.is_available():
        abort(403, "Module 'avatar' not running")

    try:
        stream = request.files["file"].stream  # the base image
        cel_streams = {celname: request.files[celname].stream
                       for celname in request.files
                       if celname not in ("file", "json")}  # add-on cels the client also sent, if any
        instance_id = avatar.load(stream=stream,
                                  cel_streams=cel_streams)
    except Exception as exc:
        abort(400, f"api_avatar_load: failed, reason: {type(exc)}: {exc}")

    return jsonify({"instance_id": instance_id})

@app.route("/api/avatar/reload", methods=["POST"])
def api_avatar_reload():
    """For an existing avatar instance, load the avatar sprite posted as a file in the request, replacing the current sprite.

    Input is POST, Content-Type "multipart/form-data", with two file attachments, named "file" and "json".

    The "file" attachment should be an RGBA image in a format that Pillow can read. It will be autoscaled to 512x512.

    Optionally, there may be more file attachments, one for each add-on cel. The attachment name
    for each is the cel name. For supported cels, see `supported_cels` in `raven.server.avatarutil`,
    and for animefx cels, `raven.server.config`.

    The "json" attachment should contain the API call parameters as JSON:

        {"instance_id": "some_important_string"}

    Here the important string is the instance ID you got from `api_avatar_load`.

    No outputs.
    """
    if not avatar.is_available():
        abort(403, "Module 'avatar' not running")

    try:
        parameters = unpack_parameters_from_json_file_attachment(request.files["json"].stream)
        if "instance_id" not in parameters or not isinstance(parameters["instance_id"], str):
            abort(400, 'api_avatar_reload: "instance_id" is required')

        stream = request.files["file"].stream  # the base image
        cel_streams = {celname: request.files[celname].stream
                       for celname in request.files
                       if celname not in ("file", "json")}  # add-on cels the client also sent, if any

        avatar.reload(instance_id=parameters["instance_id"],
                      stream=stream,
                      cel_streams=cel_streams)
    except Exception as exc:
        abort(400, f"api_avatar_reload: failed, reason: {type(exc)}: {exc}")
    return "OK"

@app.route("/api/avatar/unload", methods=["POST"])
def api_avatar_unload():
    """Unload (delete) the given avatar instance.

    This automatically causes the `result_feed` for that instance to shut down.

    Input is JSON::

        {"instance_id": "some_important_string"}

    Here the important string is the instance ID you got from `api_avatar_load`.

    No outputs.
    """
    if not avatar.is_available():
        abort(403, "Module 'avatar' not running")

    data = request.get_json()
    if "instance_id" not in data or not isinstance(data["instance_id"], str):
        abort(400, 'api_avatar_unload: "instance_id" is required')

    avatar.unload(data["instance_id"])
    return "OK"

@app.route("/api/avatar/load_emotion_templates", methods=["POST"])
def api_avatar_load_emotion_templates():
    """Load custom emotion templates for avatar, or reset to defaults.

    Input is JSON::

        {"instance_id": "some_important_string",
         "emotions": {"emotion0": {"morph0": value0,
                                   ...}
                      ...}
        }

    For details, see `Animator.load_emotion_templates` in `animator.py`.

    To reload server defaults, send `"emotions": {}` or omit it.

    Here the important string is the instance ID you got from `api_avatar_load`.

    No outputs.
    """
    if not avatar.is_available():
        abort(403, "Module 'avatar' not running")

    data = request.get_json()
    if "instance_id" not in data or not isinstance(data["instance_id"], str):
        abort(400, 'api_avatar_load_emotion_templates: "instance_id" is required')

    instance_id = data["instance_id"]
    emotions = data.get("emotions", {})
    avatar.load_emotion_templates(instance_id, emotions)
    return "OK"

@app.route("/api/avatar/load_animator_settings", methods=["POST"])
def api_avatar_load_animator_settings():
    """Load custom settings for avatar animator and postprocessor, or reset to defaults.

    Input is JSON::

        {"instance_id": "some_important_string",
         "animator_settings": {"name0": value0,
                               ...}
        }

    For details, see `Animator.load_animator_settings` in `animator.py`.

    To reload server defaults, send `"animator_settings": {}` or omit it.

    Here the important string is the instance ID you got from `api_avatar_load`.

    No outputs.
    """
    if not avatar.is_available():
        abort(403, "Module 'avatar' not running")

    data = request.get_json()
    if "instance_id" not in data or not isinstance(data["instance_id"], str):
        abort(400, 'api_avatar_load_animator_settings: "instance_id" is required')

    instance_id = data["instance_id"]
    animator_settings = data.get("animator_settings", {})
    avatar.load_animator_settings(instance_id, animator_settings)
    return "OK"

@app.route("/api/avatar/start", methods=["POST"])
def api_avatar_start():
    """Start the avatar animation.

    Input is JSON::

        {"instance_id": "some_important_string"}

    Here the important string is the instance ID you got from `api_avatar_load`.

    No outputs.

    A character must be loaded first; use '/api/avatar/load' to do that.

    To pause, use '/api/avatar/stop'.
    """
    if not avatar.is_available():
        abort(403, "Module 'avatar' not running")

    data = request.get_json()
    if "instance_id" not in data or not isinstance(data["instance_id"], str):
        abort(400, 'api_avatar_start: "instance_id" is required')

    avatar.start(data["instance_id"])
    return "OK"

@app.route("/api/avatar/stop", methods=["POST"])
def api_avatar_stop():
    """Pause the avatar animation.

    Input is JSON::

        {"instance_id": "some_important_string"}

    Here the important string is the instance ID you got from `api_avatar_load`.

    No outputs.

    To resume, use '/api/avatar/start'.
    """
    if not avatar.is_available():
        abort(403, "Module 'avatar' not running")

    data = request.get_json()
    if "instance_id" not in data or not isinstance(data["instance_id"], str):
        abort(400, 'api_avatar_stop: "instance_id" is required')

    avatar.stop(data["instance_id"])
    return "OK"

@app.route("/api/avatar/start_talking", methods=["POST"])
def api_avatar_start_talking():
    """Start the mouth animation for talking.

    Input is JSON::

        {"instance_id": "some_important_string"}

    Here the important string is the instance ID you got from `api_avatar_load`.

    No outputs.

    This is the generic, non-lipsync animation that randomizes the mouth.

    This is useful for applications without actual voiced audio, such as
    an LLM when TTS is offline, or a low-budget visual novel.

    For speech with automatic lipsync, see `tts_speak_lipsynced`.
    """
    if not avatar.is_available():
        abort(403, "Module 'avatar' not running")

    data = request.get_json()
    if "instance_id" not in data or not isinstance(data["instance_id"], str):
        abort(400, 'api_avatar_start_talking: "instance_id" is required')

    avatar.start_talking(data["instance_id"])
    return "OK"

@app.route("/api/avatar/stop_talking", methods=["POST"])
def api_avatar_stop_talking():
    """Stop the mouth animation for talking.

    Input is JSON::

        {"instance_id": "some_important_string"}

    Here the important string is the instance ID you got from `api_avatar_load`.

    No outputs.

    This is the generic, non-lipsync animation that randomizes the mouth.

    This is useful for applications without actual voiced audio, such as
    an LLM when TTS is offline, or a low-budget visual novel.

    For speech with automatic lipsync, see `tts_speak_lipsynced`.
    """
    if not avatar.is_available():
        abort(403, "Module 'avatar' not running")

    data = request.get_json()
    if "instance_id" not in data or not isinstance(data["instance_id"], str):
        abort(400, 'api_avatar_stop_talking: "instance_id" is required')

    avatar.stop_talking(data["instance_id"])
    return "OK"

@app.route("/api/avatar/set_emotion", methods=["POST"])
def api_avatar_set_emotion():
    """Set avatar emotion to that posted in the request.

    Input is JSON::

        {"instance_id": "some_important_string",
         "emotion_name": "curiosity"}

    where the key "emotion_name" is literal, and the value is the emotion to set.

    Here the important string is the instance ID you got from `api_avatar_load`.

    No outputs.

    There is no getter, by design. If the emotion state is meaningful to you,
    keep a copy in your frontend, and sync that to the server.
    """
    if not avatar.is_available():
        abort(403, "Module 'avatar' not running")
    data = request.get_json()
    if "instance_id" not in data or not isinstance(data["instance_id"], str):
        abort(400, 'api_avatar_set_emotion: "instance_id" is required')
    if "emotion_name" not in data or not isinstance(data["emotion_name"], str):
        abort(400, 'api_avatar_set_emotion: "emotion_name" is required')

    instance_id = data["instance_id"]
    emotion_name = data["emotion_name"]
    avatar.set_emotion(instance_id, emotion_name)
    return "OK"

@app.route("/api/avatar/set_overrides", methods=["POST"])
def api_avatar_set_overrides():
    """Directly control the animator's morphs from the client side.

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
    """
    if not avatar.is_available():
        abort(403, "Module 'avatar' not running")

    data = request.get_json()
    if "instance_id" not in data or not isinstance(data["instance_id"], str):
        abort(400, 'api_avatar_set_overrides: "instance_id" is required')

    instance_id = data["instance_id"]
    overrides = data.get("overrides", {})
    avatar.set_overrides(instance_id, overrides)
    return "OK"

@app.route("/api/avatar/result_feed")
def api_avatar_result_feed():
    """Video output.

    Example:

      GET /api/avatar/result_feed?instance_id=some_important_string

    where `some_important_string` is what "/api/avatar/load" gave you.

    The instance ID is an URL parameter to make it trivially easy to display the video stream
    in a web browser. (You still have to get the ID from "/api/avatar/load"; it's an UUID,
    so unfortunately it's not very human-friendly.)

    Output is a "multipart/x-mixed-replace" stream of video frames, each as an image file.
    The payload separator is "--frame".

    The file format can be set in the animator settings. The frames are always sent
    with the Content-Type and Content-Length headers set.
    """
    if not avatar.is_available():
        abort(403, "Module 'avatar' not running")
    return avatar.result_feed(instance_id=request.args.get("instance_id"))  # this will yell if the instance doesn't exist so we don't have to
ignore_auth.append(api_avatar_result_feed)   # TODO: does this make sense?

@app.route("/api/avatar/get_available_filters")
def api_avatar_get_available_filters():
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
    where each component is a float in the range [0, 1].

    You can detect the type from the default value.
    """
    if not (avatar.is_available() or imagefx.is_available()):
        abort(403, "Neither of modules 'avatar' or 'imagefx' is running")
    return jsonify({"filters": Postprocessor.get_filters()})

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
        abort(400, 'api_classify: "text" is required and must be a string')

    try:
        print("Classification input:", data["text"], sep="\n")
        classification = classify.classify_text(data["text"])
        print("Classification output:", classification, sep="\n")
        gc.collect()
        return jsonify({"classification": classification})
    except Exception as exc:
        traceback.print_exc()
        abort(400, f"api_classify: failed, reason: {type(exc)}: {exc}")

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

    try:
        classification = classify.classify_text("")
        labels = [x["label"] for x in classification]
        return jsonify({"labels": labels})
    except Exception as exc:
        traceback.print_exc()
        abort(400, f"api_classify_labels: failed, reason: {type(exc)}: {exc}")

# ----------------------------------------
# module: embeddings

@app.route("/api/embeddings/compute", methods=["POST"])
def api_embeddings_compute():
    """For making vector DB keys. Compute the vector embedding of one or more sentences of text.

    Input is JSON::

        {"text": "Blah blah blah.",
         "model": "default"}

    or::

        {"text": ["Blah blah blah.",
                  ...],
         "model": "default"}

    The "model" field is optional:

      - If not specified, "default" is used.

      - If specified, the value must be one of the keys of `embedding_models` in the server config.
        The default config is `raven.server.config`, but note the server's `--config` command-line
        option, which can be used to specify a different config at server startup.

      - If specified but not present in server config, the request aborts with HTTP error 400.

    This functionality is provided because different models may be good for different use cases;
    e.g. beside a general-purpose embedder, having a separate specialized "qa" embedder that maps
    questions and related answers near each other.

    Output is also JSON::

        {"embedding": array}

    or::

        {"embedding": [array0,
                       ...]}

    respectively.
    """
    if not embeddings.is_available():
        abort(403, "Module 'embeddings' not running")  # this is the only optional module
    data = request.get_json()
    if "text" not in data:
        abort(400, 'api_embeddings_compute: "text" is required')

    sentences: Union[str, List[str]] = data["text"]
    role: str = data["model"] if "model" in data else "default"  # one of the keys of `embedding_models` in `raven.server.config`

    if not (isinstance(sentences, str) or (isinstance(sentences, list) and all(isinstance(x, str) for x in sentences))):
        abort(400, 'api_embeddings_compute: "text" must be string or array of strings')
    if isinstance(sentences, str):
        nitems = 1
    else:
        nitems = len(sentences)

    print(f"Computing vector embedding for {nitems} item{'s' if nitems != 1 else ''} with model '{role}'")
    try:
        vectors = embeddings.embed_sentences(sentences, role=role)
        return jsonify({"embedding": vectors})
    except Exception as exc:
        traceback.print_exc()
        abort(400, f"api_embeddings_compute: failed, reason: {type(exc)}: {exc}")

# ----------------------------------------
# module: imagefx

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

    To get supported filters, call the endpoint "/api/avatar/get_available_filters".
    Don't mind the name - the endpoint is available whenever at least one of "avatar"
    or "imagefx" is loaded.

    Output is an image with mimetype "image/<format>".
    """
    if not imagefx.is_available():
        abort(403, "Module 'imagefx' not running")

    try:
        file = request.files["file"]
        parameters = unpack_parameters_from_json_file_attachment(request.files["json"].stream)
        postprocessor_chain = parameters["filters"]
        format = parameters["format"]

        processed_image = imagefx.process(file.stream,
                                          output_format=format,
                                          postprocessor_chain=postprocessor_chain)
    except Exception as exc:
        traceback.print_exc()
        abort(400, f"api_imagefx_process: failed, reason: {type(exc)}: {exc}")

    return Response(processed_image, mimetype=f"image/{format.lower()}")

@app.route("/api/imagefx/upscale", methods=["POST"])
def api_imagefx_upscale():
    """Upscale an image with Anime4K.

    Input is POST, Content-Type "multipart/form-data", with two file attachments:

        "file": the actual image file (binary, any supported format)
        "json": the API call parameters, in JSON format.

    The parameters are::

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

    Output is an image with mimetype "image/<format>".
    """
    if not imagefx.is_available():
        abort(403, "Module 'imagefx' not running")

    try:
        file = request.files["file"]
        parameters = unpack_parameters_from_json_file_attachment(request.files["json"].stream)

        format = parameters["format"]
        upscaled_width = parameters["upscaled_width"]
        upscaled_height = parameters["upscaled_height"]
        preset = parameters["preset"]
        quality = parameters["quality"]

        processed_image = imagefx.upscale(file.stream,
                                          output_format=format,
                                          upscaled_width=upscaled_width,
                                          upscaled_height=upscaled_height,
                                          preset=preset,
                                          quality=quality)
    except Exception as exc:
        traceback.print_exc()
        abort(400, f"api_imagefx_upscale: failed, reason: {type(exc)}: {exc}")

    return Response(processed_image, mimetype=f"image/{format.lower()}")

# ----------------------------------------
# module: natlang

@app.route("/api/natlang/analyze", methods=["POST"])
def api_natlang_analyze():
    """Perform NLP analysis on the text posted in the request. Return the result.

    NOTE: This endpoint returns Python spaCy data in binary format.

          It can only be read by a Python client process running a Python version
          that is compatible with the Python running the server. Specifically,
          since spaCy transmits binary data in the pickle format, the client
          Python version must be such that it can unpickle data pickled by
          the server process.

          The point is that you can use the server's GPU to perform the NLP analysis,
          and then read the results in the client as if they came from a spaCy
          instance running locally on the client.

          The most convenient way to call this endpoint is `natlang_analyze`
          in `raven.client.api`.

    The NLP analysis is done via spaCy using the `spacy_model` set in the server config,
    which by default lives in `raven.server.config`.

    The analysis currently includes part-of-speech tagging, lemmatization,
    and named entity recognition.

    Input is JSON::

        {"text": "Blah blah blah."}

    To send multiple texts at once, use a list::

        {"text": ["Blah blah blah.",
                  "The quick brown fox jumps over the lazy dog."]}

    This is more efficient than sending one text at a time, as it allows the spaCy backend
    to batch the texts.

    The server runs the text(s) through the default set of pipes in the loaded spaCy model.
    There is an optional "pipes" field you can use to enable only the pipes you want.
    (Which ones exist depend on the spaCy model that is loaded.)

    This works for both one text, or multiple texts. Here is a one-text example::

        {"text": "Blah blah blah.",
         "pipes": ["tok2vec", "parser", "senter"]}

    This effectively does `with nlp.select_pipes(enable=pipes): ...` on the server side.

    This can be useful to save processing time if you only need partial analysis,
    e.g. to split the text into sentences (for the model "en_core_web_sm", the pipes
    in the example will do just that).

    Output is binary data. It can be loaded on the client side by calling
    `raven.common.nlptools.deserialize_spacy_docs`, which see.

    The response contains a custom header, "x-langcode", which is the language code
    of the server's loaded spaCy model (e.g. "en" for English). The client needs the
    language code to be able to deserialize the data correctly.

    If you use `natlang_analyze` in `raven.client.api`, it already loads the data,
    and behaves as if you had called `nlp.pipe(...)` (locally, on the client) on the text.
    """
    if not natlang.is_available():
        abort(403, "Module 'natlang' not running")

    data = request.get_json()

    if "text" not in data:
        abort(400, 'api_natlang_analyze: "text" is required')

    text = data["text"]
    if not (isinstance(text, str) or (isinstance(text, list) and all(isinstance(item, str) for item in text))):
        abort(400, 'api_natlang_analyze: "text" must be a string or a list of strings')

    if "pipes" in data:
        pipes = data["pipes"]
        if not (isinstance(pipes, list) and all(isinstance(pipe_name, str) for pipe_name in pipes)):
            abort(400, 'api_natlang_analyze: "pipes", if provided, must be a list of strings')
    else:
        pipes = None  # use the default pipes for the loaded spaCy model

    try:
        response = natlang.analyze(data["text"], pipes)
        return response
    except Exception as exc:
        traceback.print_exc()
        abort(400, f"api_natlang_analyze: failed, reason: {type(exc)}: {exc}")

# ----------------------------------------
# module: sanitize

@app.route("/api/sanitize/dehyphenate", methods=["POST"])
def api_sanitize_dehyphenate():
    """Dehyphenate the text posted in the request.

    This can be used to clean up broken text e.g. as extracted
    from a PDF file:

        Text that was bro-
        ken by hyphenation.

    -->

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
    backend (with the "multi" model) does autodetect 300+ languages,
    so give it a try.

    This is based on the `dehyphen` package. The analysis applies a small,
    specialized AI model (not an LLM) to evaluate the perplexity of the
    different possible hyphenation options (in the example, "bro ken",
    "bro-ken", "broken") that could have produced the hyphenated text.
    The engine automatically picks the choice with the minimal perplexity
    (i.e. the most likely according to the model).

    The AI is a character-level contextual embeddings model from the
    Flair-NLP project.

    We then apply some heuristics to clean up the output.

    Input is JSON::

        {"text": "Text that was bro-\nken by hyphenation."}

    or

        {"text": ["Text that was bro-\nken by hyphenation.",
                  "Some more hyp-\nhenated text."]}

    Output is also JSON::

        {"text": "Text that was broken by hyphenation."}

    or

        {"text": ["Text that was broken by hyphenation.",
                  "Some more hyphenated text."]}

    respectively.
    """
    if not sanitize.is_available():
        abort(403, "Module 'sanitize' not running")

    data = request.get_json()

    if "text" not in data:
        abort(400, 'api_sanitize_dehyphenate: "text" is required')

    text = data["text"]
    if not (isinstance(text, str) or (isinstance(text, list) and all(isinstance(item, str) for item in text))):
        abort(400, 'api_sanitize_dehyphenate: "text" must be a string or a list of strings')

    try:
        output_text = sanitize.dehyphenate(text)
        return jsonify({"text": output_text})
    except Exception as exc:
        traceback.print_exc()
        abort(400, f"api_sanitize_dehyphenate: failed, reason: {type(exc)}: {exc}")

# ----------------------------------------
# module: summarize

@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    """Summarize the text posted in the request. Return the summary.

    This uses a small, specialized AI model (not an LLM) plus some
    heuristics to clean up its output.

    Input is JSON::

        {"text": "Blah blah blah."}

    Output is also JSON::

        {"summary": "Blah."}
    """
    if not summarize.is_available():
        abort(403, "Module 'summarize' not running")

    data = request.get_json()

    if "text" not in data:
        abort(400, 'api_summarize: "text" is required')

    text = data["text"]
    if not isinstance(text, str):
        abort(400, 'api_summarize: "text" must be a string')

    try:
        summary = summarize.summarize_text(text)
        return jsonify({"summary": summary})
    except Exception as exc:
        traceback.print_exc()
        abort(400, f"api_summarize: failed, reason: {type(exc)}: {exc}")

# ----------------------------------------
# module: translate

@app.route("/api/translate", methods=["POST"])
def api_translate():
    """Translate the text posted in the request. Return the translation.

    This uses a small, specialized AI model (not an LLM).

    Input is JSON::

        {"text": "The quick brown fox jumps over the lazy dog.",
         "source_lang": "en",
         "target_lang": "fi"}

    Output is also JSON::

        {"translation": "Nopea ruskea kettu hyppää laiskan koiran yli."}

    To send several texts, use a list::

        {"text": ["The quick brown fox jumps over the lazy dog.",
                  "Please translate this text, too."],
         "source_lang": "en",
         "target_lang": "fi"}

    -->

        {"translation": ["Nopea ruskea kettu hyppää laiskan koiran yli.",
                         "Ole hyvä ja käännä tämäkin teksti."]}
    """
    if not translate.is_available():
        abort(403, "Module 'translate' not running")

    data = request.get_json()

    if "text" not in data:
        abort(400, 'api_translate: "text" is required')
    if "source_lang" not in data:
        abort(400, 'api_translate: "source_lang" is required')
    if "target_lang" not in data:
        abort(400, 'api_translate: "target_lang" is required')

    text = data["text"]
    if not (isinstance(text, str) or (isinstance(text, list) and all(isinstance(item, str) for item in text))):
        abort(400, 'api_translate: "text" must be a string or a list of strings')

    source_lang = data["source_lang"]
    if not isinstance(source_lang, str):
        abort(400, 'api_translate: "source_lang" must be a string')

    target_lang = data["target_lang"]
    if not isinstance(target_lang, str):
        abort(400, 'api_translate: "target_lang" must be a string')

    try:
        translation = translate.translate_text(text=text,
                                               source_lang=source_lang,
                                               target_lang=target_lang)
        return jsonify({"translation": translation})
    except Exception as exc:
        traceback.print_exc()
        abort(400, f"api_translate: failed, reason: {type(exc)}: {exc}")

# ----------------------------------------
# module: tts

def _list_voices():
    if not tts.is_available():
        abort(403, "Module 'tts' not running")
    try:
        return jsonify({"voices": tts.get_voices()})
    except Exception as exc:
        traceback.print_exc()
        abort(400, f"_list_voices: failed, reason: {type(exc)}: {exc}")

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

@app.route("/v1/audio/speech", methods=["POST"])
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

    try:
        return websearch.search(query, engine=engine, max_links=max_links)
    except Exception as exc:
        traceback.print_exc()
        abort(400, f"_websearch_impl: failed, reason: {type(exc)}: {exc}")

# legacy ST-compatible websearch endpoint
@app.route("/api/websearch", methods=["POST"])
def api_websearch():
    """Perform a web search with the query posted in the request.

    This is the SillyTavern compatible legacy endpoint.
    For new clients, prefer to use "/api/websearch2", which gives
    structured output and has a "max_links" option.

    (Technically, this endpoint can also accept "max_links",
     but SillyTavern doesn't send it.)

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

# NOTE SillyTavern-Extras users: settings for module enable/disable and which compute device to use for each module have been moved to `raven/server/config.py`.

# ----------------------------------------
# Parse command-line arguments

parser = argparse.ArgumentParser(prog="Raven-server", description="Server for specialized local AI models, based on the discontinued SillyTavern-extras")
parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
parser.add_argument("--config", metavar='some.python.module', default="raven.server.config", type=str, help="Python module containing the server config (default is 'raven.server.config')")
parser.add_argument("--port", type=int, help="Specify the port on which the application is hosted (default is set in the server config module)")
parser.add_argument("--listen", action="store_true", help="Host the app on the local network (if not set, the server is visible to localhost only)")
parser.add_argument("--secure", action="store_true", help="Require an API key (will be auto-created first time, and printed to console each time on server startup)")
parser.add_argument("--max-content-length", help="Set the max content length for the Flask app config.")

args = parser.parse_args()

# Switchable config module, so the user can load different configs e.g. when running with an eGPU vs. just a laptop's internal GPU.
#
# Note the security implications: essentially, this is executing user-provided arbitrary code at server startup.
#   - If the OS user account running the server can run Python (which it needs to, to start the server), it can run arbitrary Python code anyway.
#   - This is done only once per session, at server startup. By design, there is no way to inject code into a running server.
#     - If you want to switch configs, you need to restart the server.
#   - The main risk comes from an attacker replacing the config file with something else; but that can happen with a hardcoded config path, too.
#   - The server is meant to run in a trusted environment only.
config_module_name = args.config
try:
    server_config = importlib.import_module(config_module_name)
    print(f"{Fore.GREEN}{Style.BRIGHT}Server config loaded from '{args.config}'.{Style.RESET_ALL}")
except ModuleNotFoundError:
    print(f"{Fore.RED}{Style.BRIGHT}Server config '{args.config}' (Python module) not found.{Style.RESET_ALL}")
    raise
try:
    deviceinfo.validate(server_config.enabled_modules)  # modifies in-place if CPU fallback needed
except AttributeError:  # very basic sanity check while at it
    print(f"{Fore.RED}{Style.BRIGHT}Server config '{args.config}' (Python module) does not seem to be a Raven server config module.{Style.RESET_ALL}")
    raise

port = args.port if args.port else server_config.default_port
host = "0.0.0.0" if args.listen else "localhost"

# Read an API key from an already existing file. If that file doesn't exist, create it.
if args.secure:
    server_userdata_dir = pathlib.Path(server_config.server_userdata_dir).expanduser().resolve()

    try:
        with open(server_config.server_api_key_file, "r") as txt:
            api_key = txt.read().replace('\n', '')
    except Exception:
        api_key = secrets.token_hex(5)
        with open(server_config.server_api_key_file, "w") as txt:
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

# TODO: Maybe clean up the API: pass `config_module_name` to all modules, and let each `init_module` get whatever it needs from there. For implementation, see the modules that already do this.
def init_server_modules():  # keep global namespace clean
    def get_maybe_shared_spacy_device_string(module_name):
        # Seems slightly faster to run sentence-splitting on the CPU, at least for short-ish (chat message) inputs.
        # But if "natlang" is already serving a copy of spaCy, re-use that to save memory.
        if (natlang_record := server_config.enabled_modules.get("natlang", None)) is not None:
            spacy_device_string = natlang_record["device_string"]
            print(f"Initializing {Fore.GREEN}{Style.BRIGHT}{module_name}{Style.RESET_ALL}: reusing {Fore.GREEN}{Style.BRIGHT}natlang{Style.RESET_ALL}'s already loaded spaCy on device '{Fore.GREEN}{Style.BRIGHT}{spacy_device_string}{Style.RESET_ALL}'")
        else:
            print("Initializing: {Fore.GREEN}{Style.BRIGHT}{module_name}{Style.RESET_ALL}: natlang not enabled, loading spaCy on device '{Fore.GREEN}{Style.BRIGHT}cpu{Style.RESET_ALL}'")
            spacy_device_string = "cpu"
        return spacy_device_string

    if (record := server_config.enabled_modules.get("avatar", None)) is not None:
        device_string, torch_dtype = record["device_string"], record["dtype"]
        # One of 'standard_float', 'separable_float', 'standard_half', 'separable_half'.
        # FP16 boosts the rendering performance by ~1.5x, but is only supported on GPU.
        tha3_model_variant = "separable_half" if torch_dtype is torch.float16 else "separable_float"
        avatar.init_module(config_module_name, device_string, tha3_model_variant)

    if (record := server_config.enabled_modules.get("classify", None)) is not None:
        device_string, torch_dtype = record["device_string"], record["dtype"]
        classify.init_module(server_config.classification_model, device_string, torch_dtype)

    if (record := server_config.enabled_modules.get("embeddings", None)) is not None:
        device_string, torch_dtype = record["device_string"], record["dtype"]
        embeddings.init_module(config_module_name, device_string, torch_dtype)

    if (record := server_config.enabled_modules.get("imagefx", None)) is not None:
        device_string, torch_dtype = record["device_string"], record["dtype"]
        imagefx.init_module(device_string, torch_dtype)

    if (record := server_config.enabled_modules.get("natlang", None)) is not None:
        device_string = record["device_string"]  # no configurable dtype
        natlang.init_module(server_config.spacy_model,
                            device_string)

    if (record := server_config.enabled_modules.get("sanitize", None)) is not None:
        device_string = record["device_string"]  # no configurable dtype
        sanitize.init_module(server_config.dehyphenation_model, device_string)

    if (record := server_config.enabled_modules.get("summarize", None)) is not None:
        device_string, torch_dtype = record["device_string"], record["dtype"]
        spacy_device_string = get_maybe_shared_spacy_device_string("summarize")
        summarize.init_module(server_config.summarization_model,
                              server_config.spacy_model,
                              device_string,
                              spacy_device_string,
                              torch_dtype,
                              summarization_prefix=server_config.summarization_prefix)

    if (record := server_config.enabled_modules.get("translate", None)) is not None:
        device_string, torch_dtype = record["device_string"], record["dtype"]
        spacy_device_string = get_maybe_shared_spacy_device_string("translate")
        translate.init_module(config_module_name,
                              device_string,
                              server_config.spacy_model,
                              spacy_device_string,
                              torch_dtype)

    if server_config.enabled_modules.get("tts", None) is not None:
        device_string = record["device_string"]  # no configurable dtype
        tts.init_module(config_module_name, device_string)

    if server_config.enabled_modules.get("websearch", None) is not None:  # no device/dtype settings; if a record exists (regardless of whether blank), this module is enabled.
        websearch.init_module(config_module_name)
init_server_modules()

# ----------------------------------------
# Start serving

print(f"{Fore.GREEN}{Style.BRIGHT}Starting server{Style.RESET_ALL}")
where = "all IPv4 addresses" if args.listen else "localhost"
print(f"{Fore.GREEN}{Style.BRIGHT}Listening for connections from {where}{Style.RESET_ALL}")

serve(app, host=host, port=port)

def main():  # TODO: we don't really need this; it's just for console_scripts so that we can provide a command-line entrypoint.
    pass
