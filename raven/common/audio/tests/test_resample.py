"""Unit tests for raven.common.audio.resample."""

import pytest

pytest.importorskip("torchaudio")

import numpy as np  # noqa: E402 -- after importorskip
import torch  # noqa: E402

from raven.common.audio.resample import resample  # noqa: E402


def _sine(freq_hz: float, duration_s: float, sample_rate: int, dtype=np.float32) -> np.ndarray:
    """Pure sine wave in [-1, 1], rank-1."""
    t = np.arange(int(duration_s * sample_rate), dtype=dtype) / sample_rate
    return np.sin(2 * np.pi * freq_hz * t).astype(dtype)


def _peak_frequency(audio: np.ndarray, sample_rate: int) -> float:
    """Bin-peak of the magnitude spectrum — good enough for sine-wave integrity checks."""
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)
    return float(freqs[int(np.argmax(spectrum))])


class TestNoop:
    def test_same_rate_returns_input_unchanged(self):
        audio = _sine(1000.0, 0.1, 24000)
        result = resample(audio, 24000, 24000)
        assert result is audio  # identity — no copy

    def test_same_rate_tensor_returns_input_unchanged(self):
        tensor = torch.from_numpy(_sine(1000.0, 0.1, 24000))
        result = resample(tensor, 16000, 16000)
        assert result is tensor


class TestTypePreservation:
    def test_numpy_in_numpy_out(self):
        audio = _sine(1000.0, 0.1, 24000)
        result = resample(audio, 24000, 16000)
        assert isinstance(result, np.ndarray)

    def test_tensor_in_tensor_out(self):
        tensor = torch.from_numpy(_sine(1000.0, 0.1, 24000))
        result = resample(tensor, 24000, 16000)
        assert isinstance(result, torch.Tensor)

    def test_tensor_stays_on_source_device(self):
        tensor = torch.from_numpy(_sine(1000.0, 0.1, 24000))  # CPU
        result = resample(tensor, 24000, 16000)
        assert result.device == tensor.device


class TestOutputLength:
    def test_downsample_length_matches_ratio(self):
        # Common speech case: 24 kHz → 16 kHz (ratio 2/3).
        audio = _sine(1000.0, 1.0, 24000)  # 24000 samples
        result = resample(audio, 24000, 16000)
        # torchaudio rounds via polyphase reduction; tolerate ±2 samples of rounding slack.
        assert abs(len(result) - 16000) <= 2

    def test_upsample_length_matches_ratio(self):
        # 16 kHz → 24 kHz (ratio 3/2).
        audio = _sine(1000.0, 1.0, 16000)
        result = resample(audio, 16000, 24000)
        assert abs(len(result) - 24000) <= 2

    def test_non_integer_ratio_length(self):
        # 44.1 kHz → 16 kHz (CD to Whisper). Nontrivial gcd.
        audio = _sine(1000.0, 0.5, 44100)
        result = resample(audio, 44100, 16000)
        expected = int(round(len(audio) * 16000 / 44100))
        assert abs(len(result) - expected) <= 2


class TestSignalIntegrity:
    # A 1 kHz sine must remain a 1 kHz sine after round-tripping 24k → 16k → 24k,
    # for every quality preset. This catches aliasing / off-by-one-rate bugs
    # that length checks alone would miss.
    @pytest.mark.parametrize("quality", ["default", "kaiser_fast", "kaiser_best"])
    def test_sine_frequency_preserved_through_roundtrip(self, quality):
        original = _sine(1000.0, 1.0, 24000)
        down = resample(original, 24000, 16000, quality=quality)
        up = resample(down, 16000, 24000, quality=quality)

        peak = _peak_frequency(up, sample_rate=24000)
        assert peak == pytest.approx(1000.0, abs=5.0)  # within 5 Hz of original

    def test_all_quality_presets_produce_finite_output(self):
        audio = _sine(1000.0, 0.1, 24000)
        for quality in ("default", "kaiser_fast", "kaiser_best"):
            result = resample(audio, 24000, 16000, quality=quality)
            assert np.all(np.isfinite(result)), f"quality={quality!r} produced non-finite samples"


class TestErrors:
    def test_unknown_quality_preset_raises(self):
        audio = _sine(1000.0, 0.1, 24000)
        with pytest.raises(ValueError, match="unknown quality preset"):
            resample(audio, 24000, 16000, quality="ultra")  # type: ignore[arg-type]

    def test_nonpositive_from_rate_raises(self):
        audio = _sine(1000.0, 0.1, 24000)
        with pytest.raises(ValueError, match="sample rates must be positive"):
            resample(audio, 0, 16000)

    def test_nonpositive_to_rate_raises(self):
        audio = _sine(1000.0, 0.1, 24000)
        with pytest.raises(ValueError, match="sample rates must be positive"):
            resample(audio, 24000, -1)

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="expected `audio` to be np.ndarray or torch.Tensor"):
            resample([0.0, 0.1, 0.2], 24000, 16000)  # type: ignore[arg-type]
