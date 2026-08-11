"""Shared pytest fixtures for raven.librarian tests."""

import pytest

from unpythonic.env import env

from raven.librarian import llmclient


@pytest.fixture
def llm_settings(monkeypatch):
    """The real `llm_settings` env, built without contacting a backend.

    `llmclient.configure` is `setup` minus the two network queries that discover its arguments, so a caller
    holding the facts gets the genuine object — the same system prompt, character card, tool tables, sampler
    settings and personas a running app has. Forging one instead means tests assert against a replica that
    can drift from what the app actually uses, and the drift is invisible until something depends on a field
    the replica got wrong.

    That was not available until `llmclient` became importable without the client stack: it used to reach
    `spacy` and `transformers` through `raven.client.api`, and an import failure in a *conftest* takes the
    whole package's collection with it rather than skipping the few tests that need a backend. `api` is now
    imported on first use by the two tool wrappers that need it, so this import is safe — including in CI,
    which installs neither.

    Two things are pinned rather than taken from the local configuration, so that the fixture describes the
    same model on every machine:

      - `model_info`, which a running app discovers from the backend. Stated here.
      - `llm_tokenizer_path`, which selects exact token counting when set. Left unset, so `count_tokens`
        takes the estimate path, which needs no model files — the exactness of the count is not what any of
        these tests are about, and a configured tokenizer would otherwise be fetched.
    """
    monkeypatch.setattr(llmclient.librarian_config, "llm_tokenizer_path", None)
    return llmclient.configure(model_info=env(label="test-model",
                                              model_id="test-model",
                                              context_length=32768,
                                              is_vlm=None,  # "cannot tell", as a backend that does not report it gives
                                              loaded=True),  # a test that gets as far as a turn is testing one against a backend that could answer
                               backend_flavor="lmstudio",
                               backend_url="http://test-backend",
                               quiet=True)
