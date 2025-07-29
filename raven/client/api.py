"""Python client for the Raven-server web API.

This talks with the server so you can just call regular Python functions.

For documentation of what the functions do, see the server side in `raven.server.app`.
The naming convention is as follows:

  - Client API function: `avatar_load` (in `raven.client.api`)
  - Server-side function: `api_avatar_load` (in `raven.server.app`)
  - Web API endpoint: "/api/avatar/load"

We support all modules served by `raven.server.app`:

  - avatar      - animated AI avatar as a video stream
  - classify    - text sentiment analysis
  - embeddings  - vector embeddings of text, useful for semantic visualization and RAG indexing
  - imagefx     - apply filter effects to an image (see `raven.common.video.postprocessor`)
  - tts         - text-to-speech with and without lipsyncing the AI avatar
  - websearch   - search the web, and parse results for consumption by an LLM

This module must be initialized before any API function is used; see `init_module`.
Suggested default settings for the parameters of `init_module` are provided
in `raven.client.config`.

This module is licensed under the 2-clause BSD license.
"""

__all__ = ["initialize",
           "raven_server_available",
           "tts_server_available",
           "classify_labels", "classify",
           "embeddings_compute",
           "imagefx_process", "imagefx_process_file", "imagefx_process_array",
           "imagefx_upscale", "imagefx_upscale_file", "imagefx_upscale_array",
           "avatar_load", "avatar_reload", "avatar_unload",
           "avatar_load_emotion_templates", "avatar_load_emotion_templates_from_file",
           "avatar_load_animator_settings", "avatar_load_animator_settings_from_file",
           "avatar_start", "avatar_stop",
           "avatar_start_talking", "avatar_stop_talking",  # generic animation for no-audio environments; see also `tts_speak_lipsynced`
           "avatar_set_emotion",
           "avatar_set_overrides",
           "avatar_result_feed",  # this reads the AI avatar video stream
           "avatar_get_available_filters",  # shared between "avatar" and "imagefx" modules
           "summarize_summarize",
           "tts_list_voices",
           "tts_speak", "tts_speak_lipsynced",
           "tts_stop",
           "websearch_search"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import copy
import io
import json
import pathlib
import re
import requests
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

# import PIL.Image
import qoi

import numpy as np

from ..common import netutil
from ..server.modules import avatarutil

from .tts import tts_list_voices, tts_speak, tts_speak_lipsynced, tts_stop  # noqa: F401: re-export
from . import util  # for the `api_initialized` flag (must be looked up on the `util` module each time it is used, because the flag is not boxed)  # TODO: box it, or wrap it in a property?

from .util import api_config, yell_on_error  # noqa: F401: re-export
from .util import initialize_api as initialize  # noqa: F401: re-export

# --------------------------------------------------------------------------------
# Internal utilities

def pack_parameters_into_json_file_attachment(parameters: Dict[str, Any]) -> str:
    """Pack API call parameters from a `dict`, for sending in the request as a JSON file attachment.

    The return value can be used as a value in the `files` argument of a `requests.post` call::

        files={"my_param_file": pack_parameters_into_json_file_attachment(...),
               "my_data_file": ...}

    This is meant for endpoints that on the server side receive "multipart/form-data" because
    they need a file input, but also simultenously need a JSON input to pass some API call parameters.

    The counterpart is `raven.server.app.unpack_parameters_from_json_file_attachment`.
    """
    return ("parameters.json", json.dumps(parameters, indent=4), "application/json")

# --------------------------------------------------------------------------------
# General utilities

def raven_server_available() -> bool:
    """Return whether Raven-server is available.

    Raven-server handles everything on the server side of Raven,
    except possibly TTS (speech synthesis), if that has been configured
    to use another server in `init_module`.
    """
    if not util.api_initialized:
        raise RuntimeError("raven_server_available: The `raven.client.api` module must be initialized before using the API.")
    if util.api_config.raven_server_url is None:
        return False
    headers = copy.copy(util.api_config.raven_default_headers)
    try:
        response = requests.get(f"{util.api_config.raven_server_url}/health", headers=headers)
    except requests.exceptions.ConnectionError as exc:
        logger.error(f"raven_server_available: {type(exc)}: {exc}")
        return False
    if response.status_code != 200:
        return False
    return True

def tts_server_available() -> bool:
    """Return whether the speech synthesizer is available.

    TTS may use either Raven-server, or a separate Kokoro-FastAPI server,
    depending on how it was configured in `init_module`.
    """
    if not util.api_initialized:
        raise RuntimeError("tts_server_available: The `raven.client.api` module must be initialized before using the API.")
    if util.api_config.tts_server_type is None:
        return False
    headers = copy.copy(util.api_config.tts_default_headers)
    try:
        response = requests.get(f"{util.api_config.tts_url}/health", headers=headers)
    except requests.exceptions.ConnectionError as exc:
        logger.error(f"tts_server_available: {type(exc)}: {exc}")
        return False
    if response.status_code != 200:
        return False
    return True

# --------------------------------------------------------------------------------
# Avatar

def _load_file(filename):
    with open(filename, "rb") as file:
        return file.read()

def avatar_load(filename: Union[pathlib.Path, str]) -> str:
    """Create a new avatar instance, loading a character image (512x512 RGBA PNG) from `filename`.

    Then, to start the animator, call `avatar_start`.

    Returns the instance ID (important; needed by all other `avatar` API functions to operate on that specific instance).

    Be sure to delete the avatar instance when you no longer need it, using `avatar_unload`. The same server instance
    may run for a long time, and may run out of resources if too many simultaneous instances are created.
    """
    if not util.api_initialized:
        raise RuntimeError("avatar_load: The `raven.client.api` module must be initialized before using the API.")

    # Base image file
    files = {"file": (str(filename), _load_file(filename), "application/octet-stream")}

    # Any supported cels, if present in the same folder
    for celname, cel_filename in avatarutil.scan_addon_cels(filename).items():
        files[celname] = (str(cel_filename), _load_file(cel_filename), "application/octet-stream")

    # Flask expects the file as multipart/form-data. `requests` sets this automatically when we send files, if we don't set a 'Content-Type' header.
    headers = copy.copy(util.api_config.raven_default_headers)
    response = requests.post(f"{util.api_config.raven_server_url}/api/avatar/load", headers=headers, files=files)
    util.yell_on_error(response)

    output = response.json()
    return output["instance_id"]

def avatar_reload(instance_id: str, filename: Union[pathlib.Path, str]) -> None:
    """Send a new character image to an existing avatar instance."""
    if not util.api_initialized:
        raise RuntimeError("avatar_reload: The `raven.client.api` module must be initialized before using the API.")

    # Web API call parameters and the base image file.
    # We must jump through some hoops to send parameters in the same request. A convenient way is to put the parameters into another (virtual) file.
    parameters = {"instance_id": instance_id}
    files = {"json": pack_parameters_into_json_file_attachment(parameters),
             "file": (str(filename), _load_file(filename), "application/octet-stream")}

    # Any supported cels, if present in the same folder
    for celname, cel_filename in avatarutil.scan_addon_cels(filename).items():
        files[celname] = (str(cel_filename), _load_file(cel_filename), "application/octet-stream")

    # Flask expects the file as multipart/form-data. `requests` sets this automatically when we send files, if we don't set a 'Content-Type' header.
    headers = copy.copy(util.api_config.raven_default_headers)
    response = requests.post(f"{util.api_config.raven_server_url}/api/avatar/reload", headers=headers, files=files)
    util.yell_on_error(response)

def avatar_unload(instance_id: str) -> None:
    """Unload (delete) the given avatar instance."""
    if not util.api_initialized:
        raise RuntimeError("avatar_unload: The `raven.client.api` module must be initialized before using the API.")
    headers = copy.copy(util.api_config.raven_default_headers)
    headers["Content-Type"] = "application/json"
    data = {"instance_id": instance_id}
    response = requests.post(f"{util.api_config.raven_server_url}/api/avatar/unload", json=data, headers=headers)
    util.yell_on_error(response)

def avatar_load_emotion_templates(instance_id: str, emotions: Dict) -> None:
    if not util.api_initialized:
        raise RuntimeError("avatar_load_emotion_templates: The `raven.client.api` module must be initialized before using the API.")
    headers = copy.copy(util.api_config.raven_default_headers)
    headers["Content-Type"] = "application/json"
    data = {"instance_id": instance_id,
            "emotions": emotions}
    response = requests.post(f"{util.api_config.raven_server_url}/api/avatar/load_emotion_templates", json=data, headers=headers)
    util.yell_on_error(response)

def avatar_load_emotion_templates_from_file(instance_id: str, filename: Union[pathlib.Path, str]) -> None:
    if not util.api_initialized:
        raise RuntimeError("avatar_load_emotion_templates_from_file: The `raven.client.api` module must be initialized before using the API.")
    with open(filename, "r", encoding="utf-8") as json_file:
        emotions = json.load(json_file)
    avatar_load_emotion_templates(instance_id, emotions)

def avatar_load_animator_settings(instance_id: str, animator_settings: Dict) -> None:
    if not util.api_initialized:
        raise RuntimeError("avatar_load_animator_settings: The `raven.client.api` module must be initialized before using the API.")
    headers = copy.copy(util.api_config.raven_default_headers)
    headers["Content-Type"] = "application/json"
    data = {"instance_id": instance_id,
            "animator_settings": animator_settings}
    response = requests.post(f"{util.api_config.raven_server_url}/api/avatar/load_animator_settings", json=data, headers=headers)
    util.yell_on_error(response)

def avatar_load_animator_settings_from_file(instance_id: str, filename: Union[pathlib.Path, str]) -> None:
    if not util.api_initialized:
        raise RuntimeError("avatar_load_animator_settings_from_file: The `raven.client.api` module must be initialized before using the API.")
    with open(filename, "r", encoding="utf-8") as json_file:
        animator_settings = json.load(json_file)
    avatar_load_animator_settings(instance_id, animator_settings)

def avatar_start(instance_id: str) -> None:
    """Start or resume the animator."""
    if not util.api_initialized:
        raise RuntimeError("avatar_start: The `raven.client.api` module must be initialized before using the API.")
    headers = copy.copy(util.api_config.raven_default_headers)
    headers["Content-Type"] = "application/json"
    data = {"instance_id": instance_id}
    response = requests.post(f"{util.api_config.raven_server_url}/api/avatar/start", json=data, headers=headers)
    util.yell_on_error(response)

def avatar_stop(instance_id: str) -> None:
    """Pause the animator."""
    if not util.api_initialized:
        raise RuntimeError("avatar_stop: The `raven.client.api` module must be initialized before using the API.")
    headers = copy.copy(util.api_config.raven_default_headers)
    headers["Content-Type"] = "application/json"
    data = {"instance_id": instance_id}
    response = requests.post(f"{util.api_config.raven_server_url}/api/avatar/stop", json=data, headers=headers)
    util.yell_on_error(response)

def avatar_start_talking(instance_id: str) -> None:
    if not util.api_initialized:
        raise RuntimeError("avatar_start_talking: The `raven.client.api` module must be initialized before using the API.")
    headers = copy.copy(util.api_config.raven_default_headers)
    headers["Content-Type"] = "application/json"
    data = {"instance_id": instance_id}
    response = requests.post(f"{util.api_config.raven_server_url}/api/avatar/start_talking", json=data, headers=headers)
    util.yell_on_error(response)

def avatar_stop_talking(instance_id: str) -> None:
    if not util.api_initialized:
        raise RuntimeError("avatar_stop_talking: The `raven.client.api` module must be initialized before using the API.")
    headers = copy.copy(util.api_config.raven_default_headers)
    headers["Content-Type"] = "application/json"
    data = {"instance_id": instance_id}
    response = requests.post(f"{util.api_config.raven_server_url}/api/avatar/stop_talking", json=data, headers=headers)
    util.yell_on_error(response)

def avatar_set_emotion(instance_id: str, emotion_name: str) -> None:
    if not util.api_initialized:
        raise RuntimeError("avatar_set_emotion: The `raven.client.api` module must be initialized before using the API.")
    headers = copy.copy(util.api_config.raven_default_headers)
    headers["Content-Type"] = "application/json"
    data = {"instance_id": instance_id,
            "emotion_name": emotion_name}
    response = requests.post(f"{util.api_config.raven_server_url}/api/avatar/set_emotion", json=data, headers=headers)
    util.yell_on_error(response)

def avatar_set_overrides(instance_id: str, overrides: Dict[str, float]) -> None:
    if not util.api_initialized:
        raise RuntimeError("avatar_set_overrides: The `raven.client.api` module must be initialized before using the API.")
    headers = copy.copy(util.api_config.raven_default_headers)
    headers["Content-Type"] = "application/json"
    data = {"instance_id": instance_id,
            "overrides": overrides}
    response = requests.post(f"{util.api_config.raven_server_url}/api/avatar/set_overrides", json=data, headers=headers)
    util.yell_on_error(response)

def avatar_result_feed(instance_id: str, chunk_size: int = 4096, expected_mimetype: Optional[str] = None) -> Generator[Tuple[Optional[str], bytes], None, None]:
    """Return a generator that yields video frames, in the image file format received from the server.

    The yielded value is the tuple `(received_mimetype, payload)`, where `received_mimetype` is set to whatever the server
    sent in the Content-Type header. Avatar always sends a mimetype, which specifies the file format of `payload`.

    `expected_mimetype`: If provided, string identifying the mimetype for video frames expected by your client, e.g. "image/png".
    If the server sends some other format, `ValueError` is raised. If not provided, no format checking is done.

    Due to the server's framerate control, the result feed attempts to feed data to the client at TARGET_FPS (default 25).
    New frames are not generated until the previous one has been consumed. Thus, while the animator is in the running state,
    it is recommended to continuously read the stream in a background thread.

    To close the connection (so that the server stops sending), call the `.close()` method of the generator.
    The connection also auto-closes when the generator is garbage-collected.
    """
    if not util.api_initialized:
        raise RuntimeError("avatar_result_feed: The `raven.client.api` module must be initialized before using the API.")
    headers = copy.copy(util.api_config.raven_default_headers)
    headers["Accept"] = "multipart/x-mixed-replace"
    stream_response = requests.get(f"{util.api_config.raven_server_url}/api/avatar/result_feed?instance_id={instance_id}", headers=headers, stream=True)
    util.yell_on_error(stream_response)

    stream_iterator = stream_response.iter_content(chunk_size=chunk_size)
    boundary = re.search(r"boundary=(\S+)", stream_response.headers["Content-Type"]).group(1)
    boundary_prefix = f"--{boundary}"  # e.g., '--frame'
    gen = netutil.multipart_x_mixed_replace_payload_extractor(source=stream_iterator,
                                                              boundary_prefix=boundary_prefix,
                                                              expected_mimetype=expected_mimetype)
    return gen

def avatar_get_available_filters() -> List[Tuple[str, Dict]]:
    """Get available postprocessor filters.

    Available whenever at least one of "avatar" or "imagefx" is.
    """
    if not util.api_initialized:
        raise RuntimeError("avatar_get_available_filters: The `raven.client.api` module must be initialized before using the API.")
    headers = copy.copy(util.api_config.raven_default_headers)
    response = requests.get(f"{util.api_config.raven_server_url}/api/avatar/get_available_filters", headers=headers)
    util.yell_on_error(response)
    output_data = response.json()
    return output_data["filters"]

# --------------------------------------------------------------------------------
# Classify

def classify_labels() -> List[str]:
    """Get list of emotion names from server.

    Return format is::

        [emotion0, ...]
    """
    if not util.api_initialized:
        raise RuntimeError("classify_labels: The `raven.client.api` module must be initialized before using the API.")
    headers = copy.copy(util.api_config.raven_default_headers)
    response = requests.get(f"{util.api_config.raven_server_url}/api/classify/labels", headers=headers)
    util.yell_on_error(response)
    output_data = response.json()  # -> {"labels": [emotion0, ...]}
    return list(sorted(output_data["labels"]))

def classify(text: str) -> Dict[str, float]:
    """Classify the emotion of `text`.

    Return format is::

        {emotion0: score0,
         ...}
    """
    if not util.api_initialized:
        raise RuntimeError("classify: The `raven.client.api` module must be initialized before using the API.")
    headers = copy.copy(util.api_config.raven_default_headers)
    headers["Content-Type"] = "application/json"
    input_data = {"text": text}
    response = requests.post(f"{util.api_config.raven_server_url}/api/classify", headers=headers, json=input_data)
    util.yell_on_error(response)
    output_data = response.json()  # -> ["classification": [{"label": "curiosity", "score": 0.5329479575157166}, ...]]

    sorted_records = output_data["classification"]  # sorted already
    return {record["label"]: record["score"] for record in sorted_records}

# --------------------------------------------------------------------------------
# Embeddings

def embeddings_compute(text: Union[str, List[str]]) -> np.array:
    """Compute vector embeddings (semantic embeddings).

    Useful e.g. for semantic similarity comparison and RAG search.

    Return format is `np.array`, with shape:

        - `(ndim,)` if `text` is a single string
        - `(nbatch, ndim)` if `text` is a list of strings.

    Here `ndim` is the dimensionality of the vector embedding model that Raven-server is using.
    """
    if not util.api_initialized:
        raise RuntimeError("embeddings_compute: The `raven.client.api` module must be initialized before using the API.")
    headers = copy.copy(util.api_config.raven_default_headers)
    headers["Content-Type"] = "application/json"
    input_data = {"text": text}
    response = requests.post(f"{util.api_config.raven_server_url}/api/embeddings/compute", json=input_data, headers=headers)
    util.yell_on_error(response)
    output_data = response.json()

    vectors = output_data["embedding"]
    return np.array(vectors)

# --------------------------------------------------------------------------------
# Imagefx

def imagefx_process(stream,
                    output_format: str = "png",
                    filters: List[Dict[str, Any]] = []) -> bytes:
    """Process a static image through the postprocessor.

    `stream`: The image to send, as a filelike or a `bytes` object. Filelikes are e.g.:
                - `with open("example.png", "rb") as stream`, or
                - a `BytesIO` object.
              File format is autodetected on the server side.
              It can be any RGB or RGBA image Pillow can read, with any resolution.
              Special case is "qoi", which is automatically decoded by a separate
              fast QOI decoder.

    `output_format`: format to encode output to (e.g. "png", "tga", "qoi").

    `filters`: Formatted as in `raven.server.config.postprocessor_defaults`.
               Be sure to populate this - default is a blank list, which does nothing.

    Returns a `bytes` object containing the processed image, encoded in `output_format`.
    Output resolution is the same as that of the input image.
    """
    if not util.api_initialized:
        raise RuntimeError("imagefx_process: The `raven.client.api` module must be initialized before using the API.")
    # Flask expects the file as multipart/form-data. `requests` sets this automatically when we send files, if we don't set a 'Content-Type' header.
    # We must jump through some hoops to send parameters in the same request - a convenient way is to put those into another (virtual) file.
    headers = copy.copy(util.api_config.raven_default_headers)
    parameters = {"format": output_format,
                  "filters": filters}
    files = {"json": pack_parameters_into_json_file_attachment(parameters),
             "file": ("image.bin", stream, "application/octet-stream")}
    response = requests.post(f"{util.api_config.raven_server_url}/api/imagefx/process", headers=headers, files=files)
    util.yell_on_error(response)

    return response.content  # image file encoded in requested format

def imagefx_process_file(filename: Union[pathlib.Path, str],
                         output_format: str = "png",
                         filters: List[Dict[str, Any]] = []) -> bytes:
    """Exactly like `imagefx_process`, but open `filename` for reading, and set the `stream` argument to the file handle."""
    if not util.api_initialized:
        raise RuntimeError("imagefx_process_file: The `raven.client.api` module must be initialized before using the API.")

    with open(filename, "rb") as image_file:
        return imagefx_process(image_file, output_format, filters)

def imagefx_process_array(image_data: np.array,
                          filters: List[Dict[str, Any]] = []) -> np.array:
    """Exactly like `imagefx_process`, but take image data from in-memory array, and return a new array.

    Array format is float32 [0, 1], layout [h, w, c], either RGB (3 channels) or RGBA (4 channels).
    """
    # QOI
    image_rgba = np.uint8(255.0 * image_data)
    encoded_image_bytes = qoi.encode(image_rgba.copy(order="C"))
    input_buffer = io.BytesIO()
    input_buffer.write(encoded_image_bytes)
    input_buffer.seek(0)

    # # PNG via Pillow
    # image_rgba = np.uint8(255.0 * image_data)
    # input_pil_image = PIL.Image.fromarray(np.uint8(image_rgba[:, :, :3]))
    # if image_rgba.shape[2] == 4:
    #     alpha_channel = image_rgba[:, :, 3]
    #     input_pil_image.putalpha(PIL.Image.fromarray(np.uint8(alpha_channel)))
    # input_buffer = io.BytesIO()
    # input_pil_image.save(input_buffer,
    #                      format="PNG",
    #                      compress_level=1)
    # input_buffer.seek(0)

    output_image_bytes = imagefx_process(input_buffer,
                                         output_format="QOI",
                                         filters=filters)

    output_image_rgba = qoi.decode(output_image_bytes)  # -> uint8 array of shape (h, w, c)
    output_image_rgba = np.array(output_image_rgba, dtype=np.float32) / 255  # uint8 -> float [0, 1]

    # # PNG via Pillow
    # output_buffer = io.BytesIO()
    # output_buffer.write(output_image_bytes)
    # output_buffer.seek(0)
    # output_pil_image = PIL.Image.open(output_buffer)
    # output_image_rgba = np.asarray(output_pil_image)
    # output_image_rgba = np.array(output_image_rgba, dtype=np.float32) / 255  # uint8 -> float [0, 1]

    return output_image_rgba

def imagefx_upscale(stream,
                    output_format: str = "png",
                    upscaled_width: int = 1920,
                    upscaled_height: int = 1080,
                    preset: str = "C",
                    quality: str = "high") -> bytes:
    """Upscale a static image with Anime4K.

    `stream`: The image to send, as a filelike or a `bytes` object. Filelikes are e.g.:
                - `with open("example.png", "rb") as stream`, or
                - a `BytesIO` object.
              File format is autodetected on the server side.
              It can be any RGB or RGBA image Pillow can read, with any resolution.
              Special case is "qoi", which is automatically decoded by a separate
              fast QOI decoder.

    `output_format`: format to encode output to (e.g. "png", "tga", "qoi").

    `upscaled_width`, `upscaled_height`: desired output image resolution.
    `preset`: One of "A", "B" or "C", corresponding to the Anime4K preset with the same letter;
             for the meanings, see `raven.common.video.upscaler`.
     `quality`: One of "high" or "low".

    Returns a `bytes` object containing the upscaled image, encoded in `output_format`.
    """
    if not util.api_initialized:
        raise RuntimeError("imagefx_process: The `raven.client.api` module must be initialized before using the API.")
    # Flask expects the file as multipart/form-data. `requests` sets this automatically when we send files, if we don't set a 'Content-Type' header.
    # We must jump through some hoops to send parameters in the same request - a convenient way is to put those into another (virtual) file.
    headers = copy.copy(util.api_config.raven_default_headers)
    parameters = {"format": output_format,
                  "upscaled_width": upscaled_width,
                  "upscaled_height": upscaled_height,
                  "preset": preset,
                  "quality": quality}
    files = {"json": pack_parameters_into_json_file_attachment(parameters),
             "file": ("image.bin", stream, "application/octet-stream")}
    response = requests.post(f"{util.api_config.raven_server_url}/api/imagefx/upscale", headers=headers, files=files)
    util.yell_on_error(response)

    return response.content  # image file encoded in requested format

def imagefx_upscale_file(filename: Union[pathlib.Path, str],
                         output_format: str = "png",
                         upscaled_width: int = 1920,
                         upscaled_height: int = 1080,
                         preset: str = "C",
                         quality: str = "high") -> bytes:
    """Exactly like `imagefx_upscale`, but open `filename` for reading, and set the `stream` argument to the file handle."""
    if not util.api_initialized:
        raise RuntimeError("imagefx_upscale_file: The `raven.client.api` module must be initialized before using the API.")

    with open(filename, "rb") as image_file:
        return imagefx_upscale(image_file,
                               output_format=output_format,
                               upscaled_width=upscaled_width,
                               upscaled_height=upscaled_height,
                               preset=preset,
                               quality=quality)

def imagefx_upscale_array(image_data: np.array,
                          upscaled_width: int = 1920,
                          upscaled_height: int = 1080,
                          preset: str = "C",
                          quality: str = "high") -> bytes:
    """Exactly like `imagefx_upscale`, but take image data from in-memory array, and return a new array.

    Array format is float32 [0, 1], layout [h, w, c], either RGB (3 channels) or RGBA (4 channels).
    """
    image_rgba = np.uint8(255.0 * image_data)
    encoded_image_bytes = qoi.encode(image_rgba.copy(order="C"))
    input_buffer = io.BytesIO()
    input_buffer.write(encoded_image_bytes)
    input_buffer.seek(0)

    output_image_bytes = imagefx_upscale(input_buffer,
                                         output_format="QOI",
                                         upscaled_width=upscaled_width,
                                         upscaled_height=upscaled_height,
                                         preset=preset,
                                         quality=quality)

    output_image_rgba = qoi.decode(output_image_bytes)  # -> uint8 array of shape (h, w, c)
    output_image_rgba = np.array(output_image_rgba, dtype=np.float32) / 255  # uint8 -> float [0, 1]

    return output_image_rgba

# --------------------------------------------------------------------------------
# Summarize

def summarize_summarize(text: str) -> str:
    """Return an abstractive summary of input text."""
    if not util.api_initialized:
        raise RuntimeError("summarize_summarize: The `raven.client.api` module must be initialized before using the API.")
    headers = copy.copy(util.api_config.raven_default_headers)
    headers["Content-Type"] = "application/json"
    input_data = {"text": text}
    response = requests.post(f"{util.api_config.raven_server_url}/api/summarize", json=input_data, headers=headers)
    util.yell_on_error(response)
    output_data = response.json()

    summary = output_data["summary"]
    return summary

# --------------------------------------------------------------------------------
# TTS (text to speech, AI speech synthesizer)

# This part is pretty long due to lipsync code; see `tts.py`.

# --------------------------------------------------------------------------------
# Websearch

def websearch_search(query: str, engine: str = "duckduckgo", max_links: int = 10) -> Tuple[str, Dict]:
    """Perform a websearch, using Raven-server to handle the interaction with the search engine and the parsing of the results page.

    Uses the "/api/websearch2" endpoint on the server, which see.
    """
    if not util.api_initialized:
        raise RuntimeError("websearch_search: The `raven.client.api` module must be initialized before using the API.")
    headers = copy.copy(util.api_config.raven_default_headers)
    headers["Content-Type"] = "application/json"
    input_data = {"query": query,
                  "engine": engine,
                  "max_links": max_links}
    response = requests.post(f"{util.api_config.raven_server_url}/api/websearch2", headers=headers, json=input_data)
    util.yell_on_error(response)

    output_data = response.json()
    return output_data
