"""Utilities for the Python bindings of Raven's web API."""

__all__ = ["api_config",  # configuration namespace
           "initialize_api",
           "require",
           "yell_on_error"]

import logging
logger = logging.getLogger(__name__)

import atexit
import concurrent.futures
import os
import pathlib
import requests
import traceback
from typing import Optional, Union

from bs4 import BeautifulSoup  # for error message prettification (strip HTML from server's error response)

from unpythonic import equip_with_traceback
from unpythonic.env import env as envcls

from ..common import bgtask
from ..common import deviceinfo

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
