"""Integration tests for `raven.client.mayberemote` — the transparent local/remote dispatcher.

!!! Requires a running raven-server !!!

All tests in this module are automatically skipped when the server is unreachable.
Start `raven-server` before running these tests.

Scope: currently exercises `NLP` only. The other `MaybeRemoteService` subclasses
(`Dehyphenator`, `Embedder`, `STT`, `TTS`) lack coverage — see the deferred item
"MaybeRemote test coverage" in `TODO_DEFERRED.md`.
"""

import pytest

pytest.importorskip("spacy", reason="spacy not installed (needs full dependency stack)")

import numpy as np

from raven.client import api
from raven.client import config as client_config
from raven.client import mayberemote


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def initialized_api():
    """Initialize the API and check server availability.

    If the server is not running, the entire module is skipped.
    """
    api.initialize(raven_server_url=client_config.raven_server_url,
                   raven_api_key_file=client_config.raven_api_key_file,
                   tts_playback_audio_device=client_config.tts_playback_audio_device,
                   stt_capture_audio_device=client_config.stt_capture_audio_device)
    if not api.test_connection():
        pytest.skip("raven-server is not running")


@pytest.fixture(scope="module")
def remote_nlp(initialized_api):
    """`MaybeRemote.NLP` instance dispatching to the server (allow_local=False)."""
    return mayberemote.NLP(allow_local=False,
                           model_name="en_core_web_sm",
                           device_string="cpu")


# ---------------------------------------------------------------------------
# NLP — remote mode
# ---------------------------------------------------------------------------

class TestMaybeRemoteNLP:
    def test_analyze_single(self, remote_nlp):
        docs = remote_nlp.analyze("Hello world.")
        assert len(docs) == 1
        assert [t.text for t in docs[0]] == ["Hello", "world", "."]

    def test_analyze_multiple(self, remote_nlp):
        docs = remote_nlp.analyze(["First sentence.", "Second sentence."])
        assert len(docs) == 2
        assert docs[0][0].text == "First"
        assert docs[1][0].text == "Second"

    def test_analyze_pipes(self, remote_nlp):
        """Selective-pipes path, used by `avatar_controller` to split text into sentences."""
        docs = remote_nlp.analyze("First sentence. Second sentence. Third one.",
                                  pipes=["tok2vec", "parser", "senter"])
        assert len(docs) == 1
        assert [s.text for s in docs[0].sents] == ["First sentence.", "Second sentence.", "Third one."]

    def test_analyze_lemma_attributes(self, remote_nlp):
        """Attributes read by `hybridir.py` for BM25 tokenization."""
        docs = remote_nlp.analyze("The researchers investigated quantum gravity.")
        lemmas = [t.lemma_.lower() for t in docs[0] if t.is_alpha]
        assert lemmas == ["the", "researcher", "investigate", "quantum", "gravity"]

    def test_analyze_with_vectors(self, remote_nlp):
        """Feature-parity contract: `with_vectors=True` makes `token.vector` available on remote docs.

        In local mode the flag is a no-op (vectors are always there); in remote mode it opts
        into the `doc.tensor` roundtrip. Same call either way — that's the parity guarantee
        `MaybeRemote` is meant to provide.
        """
        docs = remote_nlp.analyze("Token vectors should work.", with_vectors=True)
        doc = docs[0]
        assert doc.tensor is not None and doc.tensor.size > 0
        assert np.any(doc.tensor != 0.0)
        # `token.vector` falls through to `doc.tensor[i]` on the blank-vocab reconstructed Doc.
        for token in doc:
            assert np.array_equal(token.vector, doc.tensor[token.i])

    def test_analyze_default_has_no_vectors(self, remote_nlp):
        """Default (`with_vectors=False`): `doc.tensor` is empty on the client side.

        Guards against someone flipping the default — which would silently balloon every
        natlang response on the wire.
        """
        docs = remote_nlp.analyze("No vectors today.")
        assert docs[0].tensor.size == 0

    def test_is_local_is_false_in_remote_mode(self, remote_nlp):
        assert remote_nlp.is_local() is False
