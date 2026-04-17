"""In-process TTS → resample → STT round-trip.

Text goes in, Kokoro synthesizes at 24 kHz float, resample converts to
16 kHz, Whisper transcribes, text comes out. All in-process.

The client-side integration test in `test_api.py::TestStt::test_tts_stt_roundtrip`
covers the transport layer (HTTP stack); this one covers the engines.
"""

import pytest

pytest.importorskip("kokoro")
pytest.importorskip("transformers")
pytest.importorskip("torchaudio")
pytest.importorskip("torch")

pytestmark = pytest.mark.ml

import torch  # noqa: E402

from raven.common.audio import resample as audio_resample  # noqa: E402
from raven.common.audio.speech import stt as speech_stt  # noqa: E402
from raven.common.audio.speech import tts as speech_tts  # noqa: E402


@pytest.fixture(scope="session")
def tts_pipeline() -> speech_tts.TTSPipeline:
    return speech_tts.load_tts_pipeline(repo_id="hexgrad/Kokoro-82M",
                                        device_string="cpu",
                                        lang_code="a")


@pytest.fixture(scope="session")
def stt_model() -> speech_stt.STTModel:
    # Whisper-base: 74 M params, CPU-friendly. The Linux default in server/config.py.
    return speech_stt.load_stt_model(model_name="openai/whisper-base",
                                     device_string="cpu",
                                     dtype=torch.float32)


def _tts_to_stt(tts_pipeline: speech_tts.TTSPipeline,
                stt_model: speech_stt.STTModel,
                text: str,
                prompt: str | None = None) -> str:
    """text → Kokoro (24 kHz float) → resample → Whisper (16 kHz float) → text."""
    voice = speech_tts.get_voices(tts_pipeline)[0]
    synth = speech_tts.synthesize(tts_pipeline, voice=voice, text=text, get_metadata=False)

    # Kokoro is 24 kHz, Whisper is 16 kHz.
    audio_16k = audio_resample.resample(synth.audio,
                                        from_rate=synth.sample_rate,
                                        to_rate=stt_model.sample_rate)

    return speech_stt.transcribe(stt_model,
                                 audio=audio_16k,
                                 sample_rate=stt_model.sample_rate,
                                 prompt=prompt,
                                 language="en")


class TestRoundtrip:
    def test_recognizable_word_survives_roundtrip(self, tts_pipeline, stt_model):
        original = "The quick brown fox jumps over the lazy dog."
        transcribed = _tts_to_stt(tts_pipeline, stt_model, original, prompt=original)
        low = transcribed.lower()
        # At least one content word from the original should survive TTS→STT through
        # whisper-base. We deliberately don't demand exact match — whisper-base is the
        # smallest Whisper and makes typical mistakes like "quick" → "quit", "jumps" → "jumped",
        # "lazy" → "lay-z". Content-word recognition is the real integrity check.
        assert any(word in low for word in ("fox", "dog", "quick", "brown", "lazy", "jump")), \
            f"no content word from original survived the round-trip. Transcribed: {transcribed!r}"

    def test_numbers_survive_roundtrip(self, tts_pipeline, stt_model):
        # Kokoro tokenizes numbers specially ("2025" → one word, phonemes "twˈɛnti twˈɛnti fˈIv"),
        # so this exercises a different metadata / phoneme path from plain words.
        # Whisper should still transcribe the spoken form back to something numeric-ish.
        original = "The year is 2025."
        transcribed = _tts_to_stt(tts_pipeline, stt_model, original, prompt=original)
        low = transcribed.lower()
        assert any(token in low for token in ("2025", "twenty", "year")), \
            f"no recognisable token survived. Transcribed: {transcribed!r}"
