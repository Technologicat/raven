"""Integration tests for raven.client.api — the Raven-server REST API.

!!! Requires a running raven-server !!!

All tests in this module are automatically skipped when the server is
unreachable. Start `raven-server` before running these tests.
"""

import io
import os
import pathlib
import textwrap

import numpy as np
import PIL.Image
import pytest

from raven.client import api
from raven.client import config as client_config


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
def assets_base():
    """Path to the avatar assets directory."""
    return pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "..", "avatar", "assets")).expanduser().resolve()


# Reusable sample text for classify/embeddings/etc.
SAMPLE_TEXT = "What is the airspeed velocity of an unladen swallow?"

# Neumann & Gros 2023, https://arxiv.org/abs/2210.00849
SCIENTIFIC_ABSTRACT_1 = textwrap.dedent("""
    The recent observation of neural power-law scaling relations has made a signifi-
    cant impact in the field of deep learning. A substantial amount of attention has
    been dedicated as a consequence to the description of scaling laws, although
    mostly for supervised learning and only to a reduced extent for reinforcement
    learning frameworks. In this paper we present an extensive study of performance
    scaling for a cornerstone reinforcement learning algorithm, AlphaZero. On the ba-
    sis of a relationship between Elo rating, playing strength and power-law scaling,
    we train AlphaZero agents on the games Connect Four and Pentago and analyze
    their performance. We find that player strength scales as a power law in neural
    network parameter count when not bottlenecked by available compute, and as a
    power of compute when training optimally sized agents. We observe nearly iden-
    tical scaling exponents for both games. Combining the two observed scaling laws
    we obtain a power law relating optimal size to compute similar to the ones ob-
    served for language models. We find that the predicted scaling of optimal neural
    network size fits our data for both games. We also show that large AlphaZero
    models are more sample efficient, performing better than smaller models with the
    same amount of training data.
""").strip()

# Brown et al. 2020, p. 40, https://arxiv.org/abs/2005.14165
SCIENTIFIC_ABSTRACT_2 = textwrap.dedent("""
    Giving multi-task models instructions in natural language was first formalized in a supervised setting with [MKXS18]
    and utilized for some tasks (such as summarizing) in a language model with [RWC+ 19]. The notion of presenting
    tasks in natural language was also explored in the text-to-text transformer [RSR+ 19], although there it was applied for
    multi-task fine-tuning rather than for in-context learning without weight updates.

    Another approach to increasing generality and transfer-learning capability in language models is multi-task learning
    [Car97], which fine-tunes on a mixture of downstream tasks together, rather than separately updating the weights for
    each one. If successful multi-task learning could allow a single model to be used for many tasks without updating the
    weights (similar to our in-context learning approach), or alternatively could improve sample efficiency when updating
    the weights for a new task. Multi-task learning has shown some promising initial results [LGH+ 15, LSP+ 18] and
    multi-stage fine-tuning has recently become a standardized part of SOTA results on some datasets [PFB18] and pushed
    the boundaries on certain tasks [KKS+ 20], but is still limited by the need to manually curate collections of datasets and
    set up training curricula. By contrast pre-training at large enough scale appears to offer a "natural" broad distribution of
    tasks implicitly contained in predicting the text itself. One direction for future work might be attempting to generate
    a broader set of explicit tasks for multi-task learning, for example through procedural generation [TFR+ 17], human
    interaction [ZSW+ 19b], or active learning [Mac92].

    Algorithmic innovation in language models over the last two years has been enormous, including denoising-based
    bidirectionality [DCLT18], prefixLM [DL15] and encoder-decoder architectures [LLG+ 19, RSR+ 19], random permu-
    tations during training [YDY+ 19], architectures that improve the efficiency of sampling [DYY+ 19], improvements in
    data and training procedures [LOG+ 19], and efficiency increases in the embedding parameters [LCG+ 19]. Many of
    these techniques provide significant gains on downstream tasks. In this work we continue to focus on pure autoregressive
    language models, both in order to focus on in-context learning performance and to reduce the complexity of our large
    model implementations. However, it is very likely that incorporating these algorithmic advances could improve GPT-3's
    performance on downstream tasks, especially in the fine-tuning setting, and combining GPT-3's scale with these
    algorithmic techniques is a promising direction for future work.
""").strip()


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

    def test_classify_returns_dict(self, initialized_api):
        result = api.classify(SAMPLE_TEXT)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_classify_scores_are_floats(self, initialized_api):
        result = api.classify(SAMPLE_TEXT)
        for label, score in result.items():
            assert isinstance(label, str)
            assert isinstance(score, (int, float))

    def test_classify_labels_match(self, initialized_api):
        """The classify result keys should be a subset of the known labels."""
        labels = set(api.classify_labels())
        result = api.classify(SAMPLE_TEXT)
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


# ---------------------------------------------------------------------------
# Sanitize (dehyphenate)
# ---------------------------------------------------------------------------

class TestSanitize:
    def test_dehyphenate_joins_broken_words(self, initialized_api):
        result = api.sanitize_dehyphenate(SCIENTIFIC_ABSTRACT_1)
        assert isinstance(result, str)
        # "signifi-\ncant" should become "significant".
        assert "significant" in result
        assert "signifi-" not in result

    def test_dehyphenate_preserves_real_hyphens(self, initialized_api):
        result = api.sanitize_dehyphenate(SCIENTIFIC_ABSTRACT_1)
        # "power-law" is a real hyphenated compound and should be preserved.
        assert "power-law" in result

    def test_dehyphenate_second_abstract(self, initialized_api):
        result = api.sanitize_dehyphenate(SCIENTIFIC_ABSTRACT_2)
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

    def test_translate_scientific_text(self, initialized_api):
        # Use the dehyphenated version for cleaner translation input.
        clean_text = api.sanitize_dehyphenate(SCIENTIFIC_ABSTRACT_1)
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
        assert prep is not None
        assert "audio_bytes" in prep
        assert isinstance(prep["audio_bytes"], bytes)
        assert len(prep["audio_bytes"]) > 0

    def test_prepare_with_metadata(self, initialized_api):
        prep = api.tts_prepare(text="Hello world.",
                               voice="af_nova",
                               speed=1.0,
                               get_metadata=True)
        assert prep is not None
        assert "audio_bytes" in prep
        assert "timestamps" in prep
        assert isinstance(prep["timestamps"], list)


class TestStt:
    def test_tts_stt_roundtrip(self, initialized_api):
        """Synthesize speech from text, then transcribe it back. The transcription should
        resemble the original."""
        original = "The quick brown fox jumps over the lazy dog."
        prep = api.tts_prepare(text=original, voice="af_nova", speed=1.0, get_metadata=False)
        assert prep is not None

        audio_buffer = io.BytesIO(prep["audio_bytes"])
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
    def test_single_text_returns_1d(self, initialized_api):
        embedding = api.embeddings_compute(SAMPLE_TEXT)
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert embedding.shape[0] > 0

    def test_batch_returns_2d(self, initialized_api):
        embedding = api.embeddings_compute([SAMPLE_TEXT, "Testing, 1, 2, 3."])
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 2
        assert embedding.shape[0] == 2
        assert embedding.shape[1] > 0

    def test_single_and_batch_same_dimension(self, initialized_api):
        single = api.embeddings_compute(SAMPLE_TEXT)
        batch = api.embeddings_compute([SAMPLE_TEXT])
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
                image_format, image_data = next(gen)
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
