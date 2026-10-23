"""Switching the model an LM Studio backend has loaded, between a run's steps.

A run whose steps want different models loads each in turn (`Switcher.load`): the loaded model is unloaded
first, since two large models seldom fit on one card at once, and the next is loaded with the context length
the run names, or else with the settings LM Studio has saved for it. Every switch is logged, and none asks for
confirmation: starting the run is the permission.

Only LM Studio is switched (`switcher_for`), through its REST API, `/api/v1/models/load` and `/unload`. Another
backend gets no switcher, and a step wanting a model other than the loaded one is refused before any work.
"""

__all__ = ["LOAD_TIMEOUT", "Switcher", "switcher_for"]

import logging
import time
from collections.abc import Callable

import requests
from unpythonic.env import env

from ...librarian import llmclient

logger = logging.getLogger(__name__)

#: Seconds a load may take: a large model from a slow disk takes minutes.
LOAD_TIMEOUT = 600


class Switcher:
    """Loads models on one LM Studio backend, one at a time.

    `backend_url`: the backend.
    `reconnect`: returns fresh LLM settings for whatever is now loaded, as `raven.librarian.llmclient.setup`
                 does; called after each load.
    `maybe_context_length`: the context length to load every model with; `None` for the one LM Studio has
                            saved for each.
    """

    def __init__(self, backend_url: str, *, reconnect: Callable[[], env], maybe_context_length: int | None = None):
        self.backend_url = backend_url.rstrip("/")
        self.reconnect = reconnect
        self.maybe_context_length = maybe_context_length

    def _get_models(self) -> list[dict]:
        response = requests.get(f"{self.backend_url}/api/v1/models", headers=llmclient.headers, verify=False, timeout=30)
        response.raise_for_status()
        return response.json().get("models", [])

    def available(self) -> set[str]:
        """The keys of every language model the backend can load."""
        return {model["key"] for model in self._get_models() if model.get("type") == "llm"}

    def loaded(self) -> list[str]:
        """The instance ids of the language models now loaded."""
        return [instance["id"] for model in self._get_models() if model.get("type") == "llm"
                for instance in model.get("loaded_instances") or []]

    def load(self, model: str) -> env:
        """Unload whatever is loaded, load `model`, and return fresh settings for it."""
        for instance in self.loaded():
            logger.info(f"Switcher.load: unloading '{instance}'")
            response = requests.post(f"{self.backend_url}/api/v1/models/unload", json={"instance_id": instance},
                                     headers=llmclient.headers, verify=False, timeout=LOAD_TIMEOUT)
            response.raise_for_status()
        payload = {"model": model, "parallel": 1}
        if self.maybe_context_length is not None:
            payload["context_length"] = self.maybe_context_length
        context = (f"a context of {self.maybe_context_length}" if self.maybe_context_length is not None
                   else "its saved settings")
        logger.info(f"Switcher.load: loading '{model}', with {context}")
        t0 = time.monotonic()
        response = requests.post(f"{self.backend_url}/api/v1/models/load", json=payload, headers=llmclient.headers,
                                 verify=False, timeout=LOAD_TIMEOUT)
        response.raise_for_status()
        logger.info(f"Switcher.load: '{model}' loaded in {time.monotonic() - t0:.0f} s")
        return self.reconnect()


def switcher_for(settings: env, *, reconnect: Callable[[], env],
                 maybe_context_length: int | None = None) -> Switcher | None:
    """A `Switcher` for the backend `settings` talk to, or `None` where it is not LM Studio."""
    if settings.backend_flavor != "lmstudio":
        return None
    return Switcher(settings.backend_url, reconnect=reconnect, maybe_context_length=maybe_context_length)
