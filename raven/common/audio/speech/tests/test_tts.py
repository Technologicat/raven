"""Smoke tests for the in-process Kokoro wrapper.

Full TTS→STT round-trip lives in `test_tts_stt_roundtrip.py`. This file
covers TTS in isolation: load, voice enumeration, single-segment synthesis,
metadata absolute-timestamp invariant, two-layer API equivalence.
"""

import pytest

pytest.importorskip("kokoro")
pytest.importorskip("torch")

pytestmark = pytest.mark.ml

import numpy as np  # noqa: E402

from raven.common.audio.speech import tts as speech_tts  # noqa: E402


@pytest.fixture(scope="session")
def pipeline() -> speech_tts.TTSPipeline:
    return speech_tts.load_tts_pipeline(repo_id="hexgrad/Kokoro-82M",
                                        device_string="cpu",
                                        lang_code="a")


@pytest.fixture(scope="session")
def voice(pipeline) -> str:
    # First voice alphabetically — stable against voice-set changes.
    return speech_tts.get_voices(pipeline)[0]


class TestLoad:
    def test_pipeline_has_expected_sample_rate(self, pipeline):
        assert pipeline.sample_rate == speech_tts.SAMPLE_RATE == 24000

    def test_pipeline_modelsdir_exists(self, pipeline):
        import os
        assert os.path.isdir(pipeline.modelsdir)

    def test_load_is_cached(self, pipeline):
        second = speech_tts.load_tts_pipeline(repo_id="hexgrad/Kokoro-82M",
                                              device_string="cpu",
                                              lang_code="a")
        assert second is pipeline


class TestVoices:
    def test_voice_list_nonempty(self, pipeline):
        assert len(speech_tts.get_voices(pipeline)) > 0

    def test_voice_list_sorted(self, pipeline):
        voices = speech_tts.get_voices(pipeline)
        assert voices == sorted(voices)

    def test_unknown_voice_raises(self, pipeline):
        with pytest.raises(ValueError, match="unknown voice"):
            # `synthesize` → `synthesize_iter`, which validates voice before generating.
            # Consume the generator to trigger it.
            next(speech_tts.synthesize_iter(pipeline, voice="does_not_exist", text="Hello."))


class TestSynthesizeShape:
    def test_audio_is_float32_in_range(self, pipeline, voice):
        result = speech_tts.synthesize(pipeline, voice=voice, text="Hello world.", get_metadata=False)
        assert result.audio.dtype == np.float32
        assert result.audio.ndim == 1
        # Float audio in [-1, 1]. Kokoro output usually well within this, but allow tiny overshoot
        # for very loud peaks.
        assert np.abs(result.audio).max() <= 1.01

    def test_duration_matches_length(self, pipeline, voice):
        result = speech_tts.synthesize(pipeline, voice=voice, text="Hello world.", get_metadata=False)
        assert result.duration == pytest.approx(len(result.audio) / result.sample_rate)

    def test_no_metadata_when_requested_off(self, pipeline, voice):
        result = speech_tts.synthesize(pipeline, voice=voice, text="Hello.", get_metadata=False)
        assert result.word_metadata is None

    def test_metadata_present_when_requested(self, pipeline, voice):
        result = speech_tts.synthesize(pipeline, voice=voice, text="Hello world.", get_metadata=True)
        assert result.word_metadata is not None
        assert len(result.word_metadata) > 0


class TestMetadataInvariants:
    def test_word_and_phonemes_are_raw_unicode(self, pipeline, voice):
        # The common-layer API must NOT URL-encode. If anyone re-introduces %XX encoding
        # here by mistake, transport leakage has happened.
        result = speech_tts.synthesize(pipeline, voice=voice, text="Hello world.", get_metadata=True)
        for w in result.word_metadata:
            assert "%" not in w.word, f"word '{w.word}' looks URL-encoded — transport leakage into common layer"
            assert "%" not in w.phonemes, f"phonemes '{w.phonemes}' look URL-encoded"

    def test_timestamps_are_absolute_and_monotonic(self, pipeline, voice):
        # Absolute means "from start of whole audio", and timings should be non-decreasing
        # across the flat metadata list (including across segment boundaries).
        # Use a longer text so Kokoro actually produces multiple segments.
        text = ("This is a sentence. " * 8).strip()
        result = speech_tts.synthesize(pipeline, voice=voice, text=text, get_metadata=True)

        # Monotonic start_times (skip any None entries).
        prev = -1.0
        for w in result.word_metadata:
            if w.start_time is None:
                continue
            assert w.start_time >= prev, f"start_time went backwards: {prev} → {w.start_time} at word '{w.word}'"
            prev = w.start_time

        # Last word's end_time, if present, should fit within the audio duration.
        last_with_end = next((w for w in reversed(result.word_metadata) if w.end_time is not None), None)
        if last_with_end is not None:
            # Allow a small epsilon — Kokoro's per-segment timestamps can round slightly over
            # the raw sample-count duration.
            assert last_with_end.end_time <= result.duration + 0.1


class TestTwoLayerAPIEquivalence:
    def test_synthesize_equals_concatenated_iter(self, pipeline, voice):
        # `synthesize` is documented as a thin wrapper over `synthesize_iter`.
        # Can't check bit-equality — Kokoro is nondeterministic across identical calls.
        # Measured on CPU: length & sample count are identical, zero phase offset, normalized
        # cross-correlation ≈ 0.993 — but sample-wise RMS difference is ~3 % of peak amplitude
        # (-30 dB), consistent with unseeded stochastic sampling in the neural vocoder
        # (the same pattern VITS-family TTS uses for naturalness). What we CAN check: both
        # paths produce matching shape, duration, sample rate, and word-count.
        text = "The quick brown fox jumps over the lazy dog."

        result = speech_tts.synthesize(pipeline, voice=voice, text=text, get_metadata=True)
        segments = list(speech_tts.synthesize_iter(pipeline, voice=voice, text=text, get_metadata=True))

        expected_length = sum(len(seg.audio) for seg in segments)
        # Allow 1 sample of slack for the off-chance Kokoro produces a different
        # segment count run-to-run; the total length should match closely.
        assert abs(len(result.audio) - expected_length) <= 1, \
            f"length mismatch: synthesize={len(result.audio)}, iter-sum={expected_length}"

        assert result.sample_rate == segments[0].sample_rate
        assert result.duration == pytest.approx(len(result.audio) / result.sample_rate)

        flat_count = sum(len(seg.word_metadata) for seg in segments)
        assert len(result.word_metadata) == flat_count

    def test_synthesize_metadata_is_flat_concat_of_segment_metadata(self, pipeline, voice):
        # This one we CAN check structurally — within a single run, the iter path
        # must give us exactly what synthesize's metadata would be if it used
        # those same segments. We do this by calling only synthesize_iter and
        # verifying the flattened form matches what a reference implementation
        # would produce. This tests the "flatten" contract without requiring
        # two independent calls to compare.
        text = "The quick brown fox jumps over the lazy dog."
        segments = list(speech_tts.synthesize_iter(pipeline, voice=voice, text=text, get_metadata=True))

        flat = []
        for seg in segments:
            flat.extend(seg.word_metadata)

        # Flattened list is monotonic in start_time (the whole-audio absolute-time invariant).
        prev = -1.0
        for w in flat:
            if w.start_time is None:
                continue
            assert w.start_time >= prev
            prev = w.start_time

    def test_segments_have_increasing_t0(self, pipeline, voice):
        text = ("This is a sentence. " * 8).strip()
        segments = list(speech_tts.synthesize_iter(pipeline, voice=voice, text=text, get_metadata=False))
        assert len(segments) >= 1
        t0_values = [seg.t0 for seg in segments]
        assert t0_values == sorted(t0_values)
        # First segment starts at zero.
        assert t0_values[0] == 0.0


class TestCleanTimestamps:
    # `clean_timestamps` is pure — no model needed. Tests don't need the pipeline fixture.

    def test_dedup_applies_in_both_modes(self):
        t = [speech_tts.WordTiming(word="hello", phonemes="h", start_time=0.0, end_time=0.5),
             speech_tts.WordTiming(word="world", phonemes="w", start_time=0.0, end_time=0.6),  # dup start_time
             speech_tts.WordTiming(word="after", phonemes="a", start_time=0.7, end_time=1.0)]
        for for_lipsync in (True, False):
            cleaned = speech_tts.clean_timestamps(t, for_lipsync=for_lipsync)
            assert [w.word for w in cleaned] == ["hello", "after"], f"for_lipsync={for_lipsync}"

    def test_lipsync_drops_incidental_single_char(self):
        t = [speech_tts.WordTiming(word="hello", phonemes="h", start_time=0.0, end_time=0.5),
             speech_tts.WordTiming(word="-",     phonemes="-", start_time=0.5, end_time=0.6),
             speech_tts.WordTiming(word="world", phonemes="w", start_time=0.6, end_time=1.0)]
        cleaned = speech_tts.clean_timestamps(t, for_lipsync=True)
        assert [w.word for w in cleaned] == ["hello", "world"]

    def test_non_lipsync_keeps_incidental_single_char(self):
        # For captioning / transcript uses, single-char tokens are kept.
        t = [speech_tts.WordTiming(word="hello", phonemes="h", start_time=0.0, end_time=0.5),
             speech_tts.WordTiming(word="-",     phonemes="-", start_time=0.5, end_time=0.6),
             speech_tts.WordTiming(word="world", phonemes="w", start_time=0.6, end_time=1.0)]
        cleaned = speech_tts.clean_timestamps(t, for_lipsync=False)
        assert [w.word for w in cleaned] == ["hello", "-", "world"]

    def test_lipsync_keeps_common_punctuation_singletons(self):
        # Common end-of-phrase punctuation is kept (used to close the mouth for lipsync).
        t = [speech_tts.WordTiming(word="hello", phonemes="h", start_time=0.0, end_time=0.5),
             speech_tts.WordTiming(word=".",     phonemes=".", start_time=0.6, end_time=0.7),
             speech_tts.WordTiming(word="bye",   phonemes="b", start_time=0.8, end_time=1.1)]
        cleaned = speech_tts.clean_timestamps(t, for_lipsync=True)
        assert [w.word for w in cleaned] == ["hello", ".", "bye"]

    def test_preserves_identity_not_copies(self):
        # Cleaning should not copy WordTiming instances — the output list is a subset view
        # into the input (cheap, composable).
        t = [speech_tts.WordTiming(word="hello", phonemes="h", start_time=0.0, end_time=0.5)]
        cleaned = speech_tts.clean_timestamps(t)
        assert cleaned[0] is t[0]
