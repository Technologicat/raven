"""Utilities for the Python bindings of Raven's web API."""

__all__ = ["api_config",  # configuration namespace
           "initialize_api",
           "require",
           "yell_on_error",
           "post_streamed_job"]

import logging
logger = logging.getLogger(__name__)

import atexit
import concurrent.futures
import json
import os
import pathlib
import requests
import traceback
from typing import TYPE_CHECKING, Optional, Union

from bs4 import BeautifulSoup  # for error message prettification (strip HTML from server's error response)

from unpythonic import equip_with_traceback
from unpythonic.env import env as envcls

from ..common import bgtask
from ..common import deviceinfo
from ..common import netutil

if TYPE_CHECKING:
    from .config import Timeout

api_initialized = False
api_config = envcls(raven_default_headers={})
def initialize_api(raven_server_url: str,
                   raven_api_key_file: Optional[Union[pathlib.Path, str]],
                   executor: Optional[concurrent.futures.Executor] = None):
    """Set up URLs and API keys, and create the client-side background task manager.

    Call this before calling any of the actual API functions in `raven.client.api`.

    Suggested values for the `raven_*` arguments are provided in `raven.client.config`.

    `executor`: `concurrent.futures.ThreadPoolExecutor` or something duck-compatible with it.
                Used for client-side background tasks (e.g. backgrounding TTS playback calls
                so they don't block the caller).

                If not provided, an executor is instantiated automatically.

    Note that audio playback and capture are local resources and live outside this init path.
    Apps that need audio should also call `raven.common.audio.initialize(...)`.
    """
    global api_initialized

    # HACK: Here it is very useful to know where the call came from, to debug mysterious extra initializations (since only the settings sent the first time will take).
    dummy_exc = Exception()
    dummy_exc = equip_with_traceback(dummy_exc, stacklevel=2)  # 2 = ignore `equip_with_traceback` itself, and its caller, i.e. us
    tb = traceback.extract_tb(dummy_exc.__traceback__)
    top_frame = tb[-1]
    called_from = f"{top_frame[0]}:{top_frame[1]}"  # e.g. "/home/xxx/foo.py:52"
    logger.info(f"initialize_api: called from: {called_from}")

    if api_initialized:  # initialize only once
        logger.info("initialize_api: `raven.client.api` is already initialized. Using existing initialization.")
        return

    logger.info(f"initialize_api: Initializing `raven.client.api` with raven_server_url = '{raven_server_url}', raven_api_key_file = '{str(raven_api_key_file)}', executor = {executor}.")

    if executor is None:
        executor = concurrent.futures.ThreadPoolExecutor()
    api_config.task_manager = bgtask.TaskManager(name="raven_client_api",
                                                 mode="concurrent",
                                                 executor=executor)
    def clear_background_tasks():
        api_config.task_manager.clear(wait=False)  # signal background tasks to exit
    atexit.register(clear_background_tasks)

    api_config.raven_server_url = raven_server_url

    if raven_api_key_file is not None and os.path.exists(raven_api_key_file):  # TODO: test this (I have no idea what I'm doing)
        with open(raven_api_key_file, "r", encoding="utf-8") as f:
            raven_api_key = f.read().replace('\n', '')
        # See `raven.server.app`.
        api_config.raven_default_headers["Authorization"] = raven_api_key.strip()

    # Validate local-mode fallback device settings (CUDA → CPU fallback, dtype adjustments,
    # device_name injection). Deferred import of `raven.client.config` — importing it at
    # module top-level would cycle via `raven.server.config`. Accessed in place: `validate`
    # modifies the config dicts so downstream readers see the validated values.
    from . import config as client_config  # noqa: PLC0415 -- intentional deferred import
    deviceinfo.validate(client_config.devices)

    # Network timeouts, sourced from config so `raven.client.api`/`tts` read them from the runtime
    # config namespace rather than importing `client_config` (which would risk the import cycle above).
    api_config.network_timeout = client_config.network_timeout
    api_config.network_timeout_streaming = client_config.network_timeout_streaming

    api_initialized = True

def require() -> None:
    """Raise `RuntimeError` if `raven.client.api` has not been initialized yet.

    Intended as a one-liner guard at the top of every API function. Pair with
    `raven.common.audio.player.require` / `raven.common.audio.recorder.require`
    for a consistent fail-fast shape across Raven's client layer.
    """
    if not api_initialized:
        raise RuntimeError("raven.client.util.require: `raven.client.api` has not been initialized. Call `raven.client.api.initialize(...)` first.")

def _strip_html(html: str) -> str:
    try:
        soup = BeautifulSoup(html, features='html.parser')
        return soup.get_text()
    except Exception:
        return html  # used for cleaning error messages; important to see the original text if HTML stripping fails

def yell_on_error(response: requests.Response) -> None:
    if response.status_code != 200:
        logger.error(f"Raven-server returned error: {response.status_code} {response.reason}. Content of error response follows.")
        logger.error(_strip_html(response.text))
        raise RuntimeError(f"While calling Raven-server: HTTP {response.status_code} {response.reason}")

def post_streamed_job(url: str,
                      input_data: dict,
                      timeout: "Timeout",
                      maybe_abort: netutil.Abort | None = None) -> dict:
    """POST `input_data` to a Raven-server endpoint that answers with a streamed job, and return its result.

    The server side is `raven.server.util.stream_job`: the headers arrive at once and the JSON result
    when the job is done.

    `timeout`: a `raven.client.config.Timeout`. Its read timeout is between bytes, and the server sends
               nothing while the job runs, so it bounds the job.

    `maybe_abort`: A `raven.common.netutil.Abort` handle, if the call should be abandonable from another
                   thread. Firing it closes the connection, which also tells the server to stop the job,
                   and raises `netutil.Aborted` here.

    Raises `RuntimeError` when the server refuses the request (a bad status), and when the job itself
    failed (an `"error"` in the body) — the same exception for both, since to the caller they are the same
    event.
    """
    if maybe_abort is not None and maybe_abort.aborted:
        raise netutil.Aborted("post_streamed_job: aborted before the request was sent")
    headers = dict(api_config.raven_default_headers)
    headers["Content-Type"] = "application/json"
    response = requests.post(url, headers=headers, json=input_data, timeout=timeout, stream=True)
    yell_on_error(response)
    if maybe_abort is not None:
        maybe_abort.arm(response)
    try:
        body = response.content
    except requests.RequestException as exc:
        if maybe_abort is not None and maybe_abort.aborted:
            raise netutil.Aborted("post_streamed_job: aborted while waiting for the result") from exc
        raise
    finally:
        if maybe_abort is not None:
            maybe_abort.disarm()
    output_data = json.loads(body)
    if "error" in output_data:
        logger.error(f"post_streamed_job: Raven-server reported a failure: {output_data['error']}")
        raise RuntimeError(f"While calling Raven-server: {output_data['error']}")
    return output_data
