"""Integration tests for raven.client.api — the Raven-server REST API.

!!! Requires a running raven-server !!!

All tests in this module are automatically skipped when the server is
unreachable. Start `raven-server` before running these tests.

Shared fixtures (`initialized_api`, `assets_base`, `sample_text`,
`scientific_abstract_1`, `scientific_abstract_2`) live in the sibling
`conftest.py` so that `test_mayberemote.py` can reuse them.
"""

import io

import numpy as np
import PIL.Image

from raven.client import api


# ---------------------------------------------------------------------------
# Server connection
# ---------------------------------------------------------------------------

class TestConnection:
    def test_connection_succeeds(self, initialized_api):
        assert api.test_connection()


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class TestModules:
    def test_returns_nonempty_list(self, initialized_api):
        result = api.modules()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_contains_expected_modules(self, initialized_api):
        result = api.modules()
        # At minimum, classify and natlang should be available.
        assert "classify" in result
        assert "natlang" in result


# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------

class TestClassify:
    def test_labels_returns_nonempty_list(self, initialized_api):
        labels = api.classify_labels()
        assert isinstance(labels, list)
        assert len(labels) > 0

    def test_classify_returns_dict(self, initialized_api, sample_text):
        result = api.classify(sample_text)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_classify_scores_are_floats(self, initialized_api, sample_text):
        result = api.classify(sample_text)
        for label, score in result.items():
            assert isinstance(label, str)
            assert isinstance(score, (int, float))

    def test_classify_labels_match(self, initialized_api, sample_text):
        """The classify result keys should be a subset of the known labels."""
        labels = set(api.classify_labels())
        result = api.classify(sample_text)
        assert set(result.keys()).issubset(labels)


# ---------------------------------------------------------------------------
# ImageFX
# ---------------------------------------------------------------------------

class TestImagefx:
    def test_available_filters_returns_list(self, initialized_api):
        filters = api.avatar_get_available_filters()
        assert isinstance(filters, list)

    def test_process_file_returns_valid_png(self, initialized_api, assets_base):
        processed_bytes = api.imagefx_process_file(
            assets_base / "backdrops" / "study.png",
            output_format="png",
            filters=[["analog_lowres", {"sigma": 3.0}]])
        assert isinstance(processed_bytes, bytes)
        assert len(processed_bytes) > 0
        image = PIL.Image.open(io.BytesIO(processed_bytes))
        assert image.size[0] > 0 and image.size[1] > 0

    def test_upscale_file_returns_4k_png(self, initialized_api, assets_base):
        processed_bytes = api.imagefx_upscale_file(
            assets_base / "backdrops" / "study.png",
            output_format="png",
            upscaled_width=3840,
            upscaled_height=2160,
            preset="C",
            quality="high")
        assert isinstance(processed_bytes, bytes)
        image = PIL.Image.open(io.BytesIO(processed_bytes))
        assert image.size == (3840, 2160)


# ---------------------------------------------------------------------------
# Natlang (NLP analysis)
# ---------------------------------------------------------------------------

class TestNatlang:
    def test_single_document(self, initialized_api):
        docs = api.natlang_analyze("This is a test document for NLP analysis.")
        assert len(docs) == 1

    def test_tokens_have_attributes(self, initialized_api):
        docs = api.natlang_analyze("The quick brown fox.")
        doc = docs[0]
        for token in doc:
            assert hasattr(token, "text")
            assert hasattr(token, "lemma_")
            assert hasattr(token, "pos_")

    def test_multiple_documents(self, initialized_api):
        docs = api.natlang_analyze(["The quick brown fox jumps over the lazy dog.",
                                    "This is another document."])
        assert len(docs) == 2

    def test_pipe_selection(self, initialized_api):
        docs = api.natlang_analyze("This is a multi-sentence document. It has two sentences, see.",
                                   pipes=["tok2vec", "parser", "senter"])
        assert len(docs) == 1
        doc = docs[0]
        sents = list(doc.sents)
        assert len(sents) == 2

    def test_named_entities(self, initialized_api):
        docs = api.natlang_analyze("Albert Einstein was born in Germany.")
        doc = docs[0]
        # The NER should find at least one entity.
        assert hasattr(doc, "ents")

    def test_with_vectors_round_trip(self, initialized_api):
        """When `with_vectors=True`, the reconstructed client-side docs expose `token.vector`.

        Feature-parity with in-process spaCy use: if the caller opts in, vectors are there
        on both local and remote `MaybeRemote.NLP` paths.
        """
        import numpy as np
        docs = api.natlang_analyze("The quick brown fox.", with_vectors=True)
        doc = docs[0]
        assert doc.tensor is not None and doc.tensor.size > 0
        # Vectors should be non-zero for content tokens in a model with Tok2Vec output.
        assert np.any(doc.tensor != 0.0)
        for token in doc:
            assert np.array_equal(token.vector, doc.tensor[token.i])


# ---------------------------------------------------------------------------
# Sanitize (dehyphenate)
# ---------------------------------------------------------------------------

class TestSanitize:
    def test_dehyphenate_joins_broken_words(self, initialized_api, scientific_abstract_1):
        result = api.sanitize_dehyphenate(scientific_abstract_1)
        assert isinstance(result, str)
        # "signifi-\ncant" should become "significant".
        assert "significant" in result
        assert "signifi-" not in result

    def test_dehyphenate_preserves_real_hyphens(self, initialized_api, scientific_abstract_1):
        result = api.sanitize_dehyphenate(scientific_abstract_1)
        # "power-law" is a real hyphenated compound and should be preserved.
        assert "power-law" in result

    def test_dehyphenate_second_abstract(self, initialized_api, scientific_abstract_2):
        result = api.sanitize_dehyphenate(scientific_abstract_2)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Translate
# ---------------------------------------------------------------------------

class TestTranslate:
    def test_translate_returns_string(self, initialized_api):
        result = api.translate_translate(
            "The quick brown fox jumps over the lazy dog.",
            source_lang="en", target_lang="fi")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_translate_scientific_text(self, initialized_api, scientific_abstract_1):
        # Use the dehyphenated version for cleaner translation input.
        clean_text = api.sanitize_dehyphenate(scientific_abstract_1)
        result = api.translate_translate(clean_text, source_lang="en", target_lang="fi")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_translate_fandom_text(self, initialized_api):
        text = ("Sharon Apple. Before Hatsune Miku, before VTubers, there was Sharon Apple. "
                "The digital diva of Macross Plus hailed from the in-universe mind of Myung Fang Lone, "
                "and sings tunes by legendary composer Yoko Kanno.")
        result = api.translate_translate(text, source_lang="en", target_lang="fi")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# TTS + STT round-trip
# ---------------------------------------------------------------------------

class TestTts:
    def test_list_voices_returns_nonempty(self, initialized_api):
        voices = api.tts_list_voices()
        assert isinstance(voices, list)
        assert len(voices) > 0

    def test_prepare_returns_audio_bytes(self, initialized_api):
        prep = api.tts_prepare(text="The quick brown fox jumps over the lazy dog.",
                               voice="af_nova",
                               speed=1.0,
                               get_metadata=False)
        assert isinstance(prep.audio_bytes, bytes)
        assert len(prep.audio_bytes) > 0
        assert prep.audio_format == "flac"
        assert prep.word_metadata is None

    def test_prepare_with_metadata(self, initialized_api):
        prep = api.tts_prepare(text="Hello world.",
                               voice="af_nova",
                               speed=1.0,
                               get_metadata=True)
        assert isinstance(prep.audio_bytes, bytes)
        assert isinstance(prep.word_metadata, list)
        assert len(prep.word_metadata) > 0

    def test_prepare_blank_input_returns_empty_result(self, initialized_api):
        # Blank text no longer returns None; it returns an EncodedTTSResult with
        # empty audio_bytes as the "cancelled" signal. Callers detect via `not prep.audio_bytes`.
        prep = api.tts_prepare(text="   ", voice="af_nova", speed=1.0, get_metadata=False)
        assert prep.audio_bytes == b""
        assert prep.duration == 0.0


class TestStt:
    def test_tts_stt_roundtrip(self, initialized_api):
        """Synthesize speech from text, then transcribe it back. The transcription should
        resemble the original."""
        original = "The quick brown fox jumps over the lazy dog."
        prep = api.tts_prepare(text=original, voice="af_nova", speed=1.0, get_metadata=False)
        assert prep.audio_bytes  # non-empty = synthesis happened

        audio_buffer = io.BytesIO(prep.audio_bytes)
        transcribed = api.stt_transcribe(stream=audio_buffer, prompt=original)
        assert isinstance(transcribed, str)
        assert len(transcribed) > 0
        # The transcription should contain at least some of the original words.
        transcribed_lower = transcribed.lower()
        assert "fox" in transcribed_lower or "dog" in transcribed_lower


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class TestEmbeddings:
    def test_single_text_returns_1d(self, initialized_api, sample_text):
        embedding = api.embeddings_compute(sample_text)
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert embedding.shape[0] > 0

    def test_batch_returns_2d(self, initialized_api, sample_text):
        embedding = api.embeddings_compute([sample_text, "Testing, 1, 2, 3."])
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 2
        assert embedding.shape[0] == 2
        assert embedding.shape[1] > 0

    def test_single_and_batch_same_dimension(self, initialized_api, sample_text):
        single = api.embeddings_compute(sample_text)
        batch = api.embeddings_compute([sample_text])
        assert single.shape[0] == batch.shape[1]


# ---------------------------------------------------------------------------
# Avatar
# ---------------------------------------------------------------------------

class TestAvatar:
    def test_avatar_lifecycle(self, initialized_api, assets_base):
        """Exercise the full avatar lifecycle: load, configure, start, render, stop, unload."""
        character_path = assets_base / "characters" / "other" / "example.png"
        animator_settings_path = assets_base / "settings" / "animator.json"
        emotion_templates_path = assets_base / "emotions" / "_defaults.json"

        instance_id = api.avatar_load(character_path)
        assert isinstance(instance_id, str)
        assert len(instance_id) > 0

        try:
            # Load optional settings.
            api.avatar_load_animator_settings_from_file(instance_id, animator_settings_path)
            api.avatar_load_emotion_templates_from_file(instance_id, emotion_templates_path)

            # Start the animator and begin receiving frames.
            api.avatar_start(instance_id)
            gen = api.avatar_result_feed(instance_id)

            # Start talking animation.
            api.avatar_start_talking(instance_id)

            # Set an emotion.
            api.avatar_set_emotion(instance_id, "surprise")

            # Receive a few frames.
            for _ in range(5):
                image_format, _headers, image_data = next(gen)
                assert isinstance(image_data, bytes)
                assert len(image_data) > 0
                # Verify the frame is a valid image.
                image = PIL.Image.open(io.BytesIO(image_data))
                assert image.size[0] > 0 and image.size[1] > 0

            # Stop talking, pause, resume.
            api.avatar_stop_talking(instance_id)
            api.avatar_stop(instance_id)
            api.avatar_start(instance_id)
        finally:
            api.avatar_unload(instance_id)
