"""Smoke test for the in-process Whisper wrapper.

Full TTS→STT round-trip lives in `test_tts_stt_roundtrip.py` (added with
the TTS extraction). This file only verifies that STT alone loads, accepts
a float mono numpy array, and produces a string.

Uses `whisper-base` (74 M params) for CPU friendliness. The first run
downloads the model via `maybe_install_models` — subsequent runs hit the
HuggingFace cache.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

pytestmark = pytest.mark.ml

import numpy as np  # noqa: E402
import torch  # noqa: E402

from raven.common.audio.speech import stt as speech_stt  # noqa: E402


@pytest.fixture(scope="session")
def stt_model() -> speech_stt.STTModel:
    return speech_stt.load_stt_model(model_name="openai/whisper-base",
                                     device_string="cpu",
                                     dtype=torch.float32)


class TestLoad:
    def test_model_loads_with_expected_sample_rate(self, stt_model):
        # Whisper is canonically 16 kHz. If this ever changes upstream, every
        # caller relying on `sample_rate` to pick a resample target needs to know.
        assert stt_model.sample_rate == 16000

    def test_model_dtype_is_normalized_to_torch_dtype(self, stt_model):
        # Even when dtype is passed as a string or torch.dtype, the stored value
        # is a torch.dtype — guards MaybeRemote glue that might compare these.
        assert isinstance(stt_model.dtype, torch.dtype)

    def test_load_is_cached(self, stt_model):
        # Second call with the same key returns the exact same object.
        second = speech_stt.load_stt_model(model_name="openai/whisper-base",
                                           device_string="cpu",
                                           dtype=torch.float32)
        assert second is stt_model


class TestTranscribe:
    def test_silence_returns_string(self, stt_model):
        # 1 second of digital silence at 16 kHz. Whisper typically returns an
        # empty string or a token like "..." — we don't assert the content,
        # only that the call completes and produces a string.
        silence = np.zeros(stt_model.sample_rate, dtype=np.float32)
        result = speech_stt.transcribe(stt_model,
                                       audio=silence,
                                       sample_rate=stt_model.sample_rate)
        assert isinstance(result, str)

    def test_wrong_sample_rate_is_resampled_transparently(self, stt_model):
        # 1 s of silence at 24 kHz (Kokoro's native rate). The common layer should
        # resample to Whisper's 16 kHz internally — not raise.
        audio = np.zeros(24000, dtype=np.float32)
        result = speech_stt.transcribe(stt_model,
                                       audio=audio,
                                       sample_rate=24000)
        assert isinstance(result, str)

    def test_progress_callback_is_invoked(self, stt_model):
        silence = np.zeros(stt_model.sample_rate, dtype=np.float32)
        calls = []
        def on_progress(current: int, total: int) -> None:
            calls.append((current, total))
        speech_stt.transcribe(stt_model,
                              audio=silence,
                              sample_rate=stt_model.sample_rate,
                              progress_callback=on_progress)
        # At minimum the final (1, 1) completion signal from `transcribe` fires,
        # even if the HF `monitor_progress` path didn't emit anything on short input.
        assert len(calls) >= 1
        assert calls[-1] == (1, 1)
