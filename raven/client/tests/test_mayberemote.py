"""Integration tests for `raven.client.mayberemote` — the transparent local/remote dispatcher.

!!! Requires a running raven-server !!!

All tests in this module are automatically skipped when the server is unreachable.
Start `raven-server` before running these tests.

Scope: remote-mode roundtrip tests for all five `MaybeRemoteService` subclasses —
`NLP`, `Dehyphenator`, `Embedder`, `TTS`, `STT`. Local-mode tests would require
loading the full model stack on the test host (hundreds of MB per engine) and
are out of scope here; the existing `raven/common/tests/test_nlptools.py` covers
the nlptools / local-loader layer directly.

Shared fixtures (`initialized_api`, `scientific_abstract_1`, …) live in the
sibling `conftest.py`.
"""

import pytest

import numpy as np

from raven.client import mayberemote


# ---------------------------------------------------------------------------
# Service fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def remote_nlp(initialized_api):
    """`MaybeRemote.NLP` instance dispatching to the server (allow_local=False)."""
    return mayberemote.NLP(allow_local=False,
                           model_name="en_core_web_sm",
                           device_string="cpu")


@pytest.fixture(scope="module")
def remote_classifier(initialized_api):
    """`MaybeRemote.Classifier` instance dispatching to the server."""
    return mayberemote.Classifier(allow_local=False)


@pytest.fixture(scope="module")
def remote_translator(initialized_api):
    """`MaybeRemote.Translator` instance (en → fi) dispatching to the server."""
    return mayberemote.Translator(allow_local=False,
                                  source_lang="en",
                                  target_lang="fi")


@pytest.fixture(scope="module")
def remote_postprocessor(initialized_api):
    """`MaybeRemote.Postprocessor` instance dispatching to the server."""
    return mayberemote.Postprocessor(allow_local=False)


@pytest.fixture(scope="module")
def remote_upscaler(initialized_api):
    """`MaybeRemote.Upscaler` instance dispatching to the server."""
    return mayberemote.Upscaler(allow_local=False)


@pytest.fixture(scope="module")
def remote_dehyphenator(initialized_api):
    """`MaybeRemote.Dehyphenator` instance dispatching to the server."""
    return mayberemote.Dehyphenator(allow_local=False,
                                    model_name=None,
                                    device_string=None)


@pytest.fixture(scope="module")
def remote_embedder(initialized_api):
    """`MaybeRemote.Embedder` instance dispatching to the server.

    The "default" role is the server's general-use embedding model (see
    `raven.server.config.embedding_models`). Role names resolve server-side,
    so we don't need to know the underlying HuggingFace model name.
    """
    return mayberemote.Embedder(allow_local=False,
                                model_name="default",
                                device_string=None,
                                dtype=None)


@pytest.fixture(scope="module")
def remote_tts(initialized_api):
    """`MaybeRemote.TTS` instance dispatching to the server."""
    return mayberemote.TTS(allow_local=False)


@pytest.fixture(scope="module")
def remote_stt(initialized_api):
    """`MaybeRemote.STT` instance dispatching to the server."""
    return mayberemote.STT(allow_local=False)


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


# ---------------------------------------------------------------------------
# Dehyphenator — remote mode
# ---------------------------------------------------------------------------

class TestMaybeRemoteDehyphenator:
    def test_joins_broken_word(self, remote_dehyphenator, scientific_abstract_1):
        # The perplexity model needs paragraph-scale context to score the
        # dehyphenated form higher than the broken one — sentence-scale input
        # isn't enough for the character-level LM.
        result = remote_dehyphenator.dehyphenate(scientific_abstract_1)
        assert isinstance(result, str)
        assert "significant" in result
        assert "signifi-" not in result

    def test_is_local_is_false_in_remote_mode(self, remote_dehyphenator):
        assert remote_dehyphenator.is_local() is False


# ---------------------------------------------------------------------------
# Embedder — remote mode
# ---------------------------------------------------------------------------

class TestMaybeRemoteEmbedder:
    def test_encode_single_returns_1d(self, remote_embedder):
        vec = remote_embedder.encode("Hello world.")
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert vec.shape[0] > 0
        assert np.any(vec != 0.0)

    def test_encode_batch_returns_2d(self, remote_embedder):
        vecs = remote_embedder.encode(["First sentence.", "Second sentence."])
        assert isinstance(vecs, np.ndarray)
        assert vecs.ndim == 2
        assert vecs.shape[0] == 2
        assert vecs.shape[1] > 0

    def test_single_and_batch_same_dimension(self, remote_embedder):
        single = remote_embedder.encode("Hello.")
        batch = remote_embedder.encode(["Hello."])
        assert single.shape[0] == batch.shape[1]

    def test_is_local_is_false_in_remote_mode(self, remote_embedder):
        assert remote_embedder.is_local() is False


# ---------------------------------------------------------------------------
# TTS — remote mode
# ---------------------------------------------------------------------------

class TestMaybeRemoteTTS:
    def test_list_voices(self, remote_tts):
        voices = remote_tts.list_voices()
        assert isinstance(voices, list)
        assert len(voices) > 0
        assert "af_nova" in voices  # default test voice used elsewhere

    def test_synthesize_raw_float_shape(self, remote_tts):
        """`format=None` returns a `TTSResult` with raw float audio in [-1, 1]."""
        result = remote_tts.synthesize(voice="af_nova",
                                       text="Hello world.",
                                       get_metadata=False,
                                       format=None)
        assert result.audio.ndim == 1
        assert result.audio.dtype == np.float32
        assert -1.0 <= result.audio.min() and result.audio.max() <= 1.0
        assert result.sample_rate == remote_tts.sample_rate
        assert result.duration > 0.0

    def test_synthesize_encoded_shape(self, remote_tts):
        """`format="flac"` returns an `EncodedTTSResult` with encoded bytes."""
        result = remote_tts.synthesize(voice="af_nova",
                                       text="Hello world.",
                                       get_metadata=False,
                                       format="flac")
        assert isinstance(result.audio_bytes, bytes)
        assert len(result.audio_bytes) > 0
        assert result.audio_format == "flac"
        assert result.duration > 0.0

    def test_synthesize_with_metadata(self, remote_tts):
        """Word-level timing metadata is populated when requested."""
        result = remote_tts.synthesize(voice="af_nova",
                                       text="Hello world.",
                                       get_metadata=True,
                                       format=None)
        assert isinstance(result.word_metadata, list)
        assert len(result.word_metadata) > 0

    def test_is_local_is_false_in_remote_mode(self, remote_tts):
        assert remote_tts.is_local() is False


# ---------------------------------------------------------------------------
# STT — remote mode
# ---------------------------------------------------------------------------

class TestMaybeRemoteSTT:
    def test_tts_stt_roundtrip(self, remote_tts, remote_stt):
        """Synthesize via `MaybeRemote.TTS`, transcribe via `MaybeRemote.STT`.

        Avoids shipping audio fixtures in the repo — the TTS service generates the
        sample at test time. Does require TTS to work first, but that's acceptable:
        STT coverage is valuable and the alternative (bundled audio) is a worse tax.
        """
        original = "The quick brown fox jumps over the lazy dog."
        synth = remote_tts.synthesize(voice="af_nova",
                                      text=original,
                                      get_metadata=False,
                                      format=None)
        assert synth.audio.size > 0  # synthesis actually happened

        transcribed = remote_stt.transcribe(audio=synth.audio,
                                            sample_rate=synth.sample_rate,
                                            prompt=original)
        assert isinstance(transcribed, str)
        assert len(transcribed) > 0
        # Whisper won't reproduce the sentence verbatim, but should catch the content words.
        transcribed_lower = transcribed.lower()
        assert "fox" in transcribed_lower or "dog" in transcribed_lower

    def test_is_local_is_false_in_remote_mode(self, remote_stt):
        assert remote_stt.is_local() is False


# ---------------------------------------------------------------------------
# Classifier — remote mode
# ---------------------------------------------------------------------------

class TestMaybeRemoteClassifier:
    def test_classify_returns_sorted_dict(self, remote_classifier, sample_text):
        result = remote_classifier.classify(sample_text)
        assert isinstance(result, dict)
        assert len(result) > 0
        scores = list(result.values())
        # Dict iteration order preserves insertion order; classify returns
        # scores sorted descending, so adjacent scores should be non-increasing.
        for earlier, later in zip(scores, scores[1:]):
            assert earlier >= later

    def test_labels_returns_nonempty_list(self, remote_classifier):
        labels = remote_classifier.labels()
        assert isinstance(labels, list)
        assert len(labels) > 0
        assert all(isinstance(label, str) for label in labels)

    def test_classify_labels_subset_of_labels(self, remote_classifier, sample_text):
        """Every label in a classify result should come from the declared label set."""
        all_labels = set(remote_classifier.labels())
        result = remote_classifier.classify(sample_text)
        assert set(result.keys()).issubset(all_labels)

    def test_is_local_is_false_in_remote_mode(self, remote_classifier):
        assert remote_classifier.is_local() is False


# ---------------------------------------------------------------------------
# Translator — remote mode
# ---------------------------------------------------------------------------

class TestMaybeRemoteTranslator:
    def test_translate_single_string(self, remote_translator):
        """Single-string input → single-string output."""
        result = remote_translator.translate("The quick brown fox jumps over the lazy dog.")
        assert isinstance(result, str)
        assert len(result) > 0
        # Sanity check: output should differ from input (translation actually happened).
        assert result != "The quick brown fox jumps over the lazy dog."

    def test_translate_list(self, remote_translator):
        """List-of-strings input → list-of-strings output."""
        result = remote_translator.translate(["Hello.", "Goodbye."])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, str) and len(r) > 0 for r in result)

    def test_is_local_is_false_in_remote_mode(self, remote_translator):
        assert remote_translator.is_local() is False


# ---------------------------------------------------------------------------
# Postprocessor — remote mode
# ---------------------------------------------------------------------------

class TestMaybeRemotePostprocessor:
    def test_process_preserves_shape_and_range(self, remote_postprocessor):
        """Postprocessor output is float32 [0, 1] with the same resolution as input."""
        # 64×64 RGBA mid-gray with a visible diagonal stripe.
        image = np.full((64, 64, 4), 0.5, dtype=np.float32)
        for i in range(64):
            image[i, i] = [1.0, 0.0, 0.0, 1.0]
        # No-op chain: empty filters list.
        result = remote_postprocessor.process(image, filters=[])
        assert result.dtype == np.float32
        assert result.shape == (64, 64, 4)
        assert result.min() >= 0.0 and result.max() <= 1.0

    def test_process_actually_filters(self, remote_postprocessor):
        """A simple filter chain should change the image (bloom noticeably brightens)."""
        # Start from a darkish image so bloom has something to do.
        image = np.full((64, 64, 4), 0.25, dtype=np.float32)
        image[..., 3] = 1.0  # opaque
        image[16:48, 16:48] = [1.0, 1.0, 1.0, 1.0]  # bright square
        filters = [("bloom", {"threshold": 0.5, "exposure": 1.0, "sigma": 7.0})]
        result = remote_postprocessor.process(image, filters=filters)
        # Result should differ from input (filter had an effect).
        assert not np.allclose(result, image, atol=1e-4)

    def test_is_local_is_false_in_remote_mode(self, remote_postprocessor):
        assert remote_postprocessor.is_local() is False


# ---------------------------------------------------------------------------
# Upscaler — remote mode
# ---------------------------------------------------------------------------

class TestMaybeRemoteUpscaler:
    def test_upscale_changes_resolution(self, remote_upscaler):
        """Upscaler output resolution matches the requested target."""
        # 64×64 source → 128×128 target. `bicubic` quality avoids the Anime4K
        # model-load cost; enough to exercise the dispatch path.
        image = np.zeros((64, 64, 4), dtype=np.float32)
        image[..., 3] = 1.0
        image[::2, ::2, 0] = 1.0  # red checkerboard
        result = remote_upscaler.upscale(image,
                                         upscaled_width=128,
                                         upscaled_height=128,
                                         preset="C",
                                         quality="bicubic")
        assert result.dtype == np.float32
        assert result.shape == (128, 128, 4)
        assert result.min() >= 0.0 and result.max() <= 1.0

    def test_is_local_is_false_in_remote_mode(self, remote_upscaler):
        assert remote_upscaler.is_local() is False


# ---------------------------------------------------------------------------
# TTS.speak / TTS.speak_lipsynced — remote-mode delegation
# ---------------------------------------------------------------------------
#
# `MaybeRemote.TTS.speak` and `.speak_lipsynced` in remote mode are thin
# delegators to `raven.client.api.tts_speak` / `tts_speak_lipsynced`. The
# tests below verify the delegation itself (correct args forwarded, `prep`
# union normalized) by monkeypatching the api-layer functions — this keeps
# the tests fast and isolated from audio hardware. Real end-to-end TTS
# playback is exercised by the visual smoke-test workflow, not by unit tests.

class TestMaybeRemoteTTSSpeak:
    def test_speak_delegates_to_api_in_remote_mode(self, remote_tts, monkeypatch):
        calls = []
        def fake_tts_speak(**kwargs):
            calls.append(kwargs)
        monkeypatch.setattr("raven.client.api.tts_speak", fake_tts_speak)
        remote_tts.speak(voice="af_nova", text="Hello world.", speed=1.0)
        assert len(calls) == 1
        kwargs = calls[0]
        assert kwargs["text"] == "Hello world."
        assert kwargs["voice"] == "af_nova"
        assert kwargs["speed"] == 1.0
        assert kwargs["prep"] is None  # no prep supplied

    def test_speak_encodes_ttsresult_prep_before_delegating(self, remote_tts, monkeypatch):
        """When `prep` is a `TTSResult`, it should be encoded to `EncodedTTSResult` before delegation."""
        from raven.common.audio.speech import datatypes as speech_datatypes
        calls = []
        def fake_tts_speak(**kwargs):
            calls.append(kwargs)
        monkeypatch.setattr("raven.client.api.tts_speak", fake_tts_speak)

        ttsresult = speech_datatypes.TTSResult(audio=np.zeros(240, dtype=np.float32),  # 10 ms at 24 kHz
                                               sample_rate=24000,
                                               duration=0.01,
                                               word_metadata=None)
        remote_tts.speak(voice="af_nova", text="Hello.", prep=ttsresult)

        assert len(calls) == 1
        prep_out = calls[0]["prep"]
        assert isinstance(prep_out, speech_datatypes.EncodedTTSResult)
        assert prep_out.audio_format == "flac"

    def test_speak_passes_encoded_prep_through(self, remote_tts, monkeypatch):
        """When `prep` is already `EncodedTTSResult`, it should pass through unchanged (identity)."""
        from raven.common.audio.speech import datatypes as speech_datatypes
        calls = []
        def fake_tts_speak(**kwargs):
            calls.append(kwargs)
        monkeypatch.setattr("raven.client.api.tts_speak", fake_tts_speak)

        encoded = speech_datatypes.EncodedTTSResult(audio_bytes=b"dummy flac bytes",
                                                    audio_format="flac",
                                                    sample_rate=24000,
                                                    duration=0.1,
                                                    word_metadata=None)
        remote_tts.speak(voice="af_nova", text="Hello.", prep=encoded)

        assert len(calls) == 1
        assert calls[0]["prep"] is encoded  # passed through, not re-encoded

    def test_speak_lipsynced_delegates_to_api_in_remote_mode(self, remote_tts, monkeypatch):
        calls = []
        def fake_tts_speak_lipsynced(**kwargs):
            calls.append(kwargs)
        monkeypatch.setattr("raven.client.api.tts_speak_lipsynced", fake_tts_speak_lipsynced)

        remote_tts.speak_lipsynced(instance_id="avatar-uuid-123",
                                   voice="af_nova",
                                   text="Hello world.",
                                   video_offset=-0.5)
        assert len(calls) == 1
        kwargs = calls[0]
        assert kwargs["instance_id"] == "avatar-uuid-123"
        assert kwargs["text"] == "Hello world."
        assert kwargs["voice"] == "af_nova"
        assert kwargs["video_offset"] == -0.5
        assert kwargs["prep"] is None

    def test_speak_lipsynced_encodes_ttsresult_prep(self, remote_tts, monkeypatch):
        """Same `prep` union normalization as `.speak`, for the lipsynced path."""
        from raven.common.audio.speech import datatypes as speech_datatypes
        calls = []
        def fake_tts_speak_lipsynced(**kwargs):
            calls.append(kwargs)
        monkeypatch.setattr("raven.client.api.tts_speak_lipsynced", fake_tts_speak_lipsynced)

        ttsresult = speech_datatypes.TTSResult(audio=np.zeros(240, dtype=np.float32),
                                               sample_rate=24000,
                                               duration=0.01,
                                               word_metadata=[])
        remote_tts.speak_lipsynced(instance_id="avatar-uuid-123",
                                   voice="af_nova",
                                   text="Hello.",
                                   prep=ttsresult)

        prep_out = calls[0]["prep"]
        assert isinstance(prep_out, speech_datatypes.EncodedTTSResult)
        assert prep_out.audio_format == "flac"
