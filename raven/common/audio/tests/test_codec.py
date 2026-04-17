"""Unit tests for raven.common.audio.codec (round-trip synthetic audio)."""

import io

import numpy as np
import pytest

# Skip on CI (no `av` there): the codec module has `import av` at top.
pytest.importorskip("av")

from raven.common.audio import codec  # noqa: E402


SAMPLE_RATE = 16000


def _sine_s16(freq_hz: float, duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a mono sine wave as a rank-1 s16 numpy array."""
    n = int(round(duration_s * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    wave = 0.5 * np.sin(2 * np.pi * freq_hz * t)  # amplitude 0.5 → ~-6 dBFS
    return (wave * 32767.0).astype(np.int16)


def _sine_s16_stereo(freq_l: float, freq_r: float, duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a stereo sine wave as a rank-2 s16 numpy array of shape [n, 2]."""
    left = _sine_s16(freq_l, duration_s, sample_rate)
    right = _sine_s16(freq_r, duration_s, sample_rate)
    return np.stack([left, right], axis=-1)


# --------------------------------------------------------------------------------
# Shape validation

class TestEncodeShapeValidation:
    def test_rank1_mono_accepted(self):
        audio = _sine_s16(440, 0.1)
        # Should not raise.
        blob = codec.encode(audio, format="wav", sample_rate=SAMPLE_RATE)
        assert isinstance(blob, bytes) and len(blob) > 0

    def test_rank2_mono_accepted(self):
        audio = _sine_s16(440, 0.1)[:, np.newaxis]  # shape [n, 1]
        blob = codec.encode(audio, format="wav", sample_rate=SAMPLE_RATE)
        assert isinstance(blob, bytes) and len(blob) > 0

    def test_rank2_stereo_accepted(self):
        audio = _sine_s16_stereo(440, 660, 0.1)
        blob = codec.encode(audio, format="wav", sample_rate=SAMPLE_RATE)
        assert isinstance(blob, bytes) and len(blob) > 0

    def test_rank3_rejected(self):
        audio = np.zeros((100, 2, 1), dtype=np.int16)
        with pytest.raises(ValueError, match="mono .* or stereo"):
            codec.encode(audio, format="wav", sample_rate=SAMPLE_RATE)

    def test_rank2_three_channels_rejected(self):
        audio = np.zeros((100, 3), dtype=np.int16)
        with pytest.raises(ValueError, match="mono .* or stereo"):
            codec.encode(audio, format="wav", sample_rate=SAMPLE_RATE)


# --------------------------------------------------------------------------------
# Lossless round-trip (WAV and FLAC)

class TestLosslessRoundtrip:
    @pytest.mark.parametrize("format", ["wav", "flac"])
    def test_mono_roundtrip(self, format):
        original = _sine_s16(440, 0.2)
        blob = codec.encode(original, format=format, sample_rate=SAMPLE_RATE)

        metadata, decoded = codec.decode(io.BytesIO(blob))
        assert metadata["input_sample_rate"] == SAMPLE_RATE
        # pyav reports layout names as either "mono"/"stereo" or "N channels" depending on the
        # container — accept both spellings for single-channel audio.
        assert metadata["input_layout"] in ("mono", "1 channels")
        # Lossless → sample-for-sample match (lengths may include a tiny codec-introduced tail).
        assert decoded.dtype == np.int16
        assert len(decoded) >= len(original)
        np.testing.assert_array_equal(decoded[:len(original)], original)

    @pytest.mark.parametrize("format", ["wav", "flac"])
    def test_stereo_roundtrip_via_mono_downmix(self, format):
        # Raven's `decode` returns a rank-1 array, so to compare channels we downmix
        # to mono via resampler. The test verifies the stereo encode/decode pipeline
        # runs end-to-end and produces correct metadata.
        original = _sine_s16_stereo(440, 660, 0.2)
        blob = codec.encode(original, format=format, sample_rate=SAMPLE_RATE)

        metadata, decoded = codec.decode(io.BytesIO(blob), target_layout="mono")
        assert metadata["input_layout"] in ("stereo", "2 channels")
        assert metadata["input_sample_rate"] == SAMPLE_RATE
        assert decoded.ndim == 1
        assert len(decoded) > 0

    def test_multi_chunk_concatenates(self):
        # Splitting one wave into chunks then encoding as a list should produce the
        # same final audio as encoding the concatenation in a single shot.
        full = _sine_s16(440, 0.3)
        chunks = [full[:4000], full[4000:10000], full[10000:]]

        blob_chunked = codec.encode(chunks, format="wav", sample_rate=SAMPLE_RATE)
        blob_whole = codec.encode(full, format="wav", sample_rate=SAMPLE_RATE)

        _m1, audio_chunked = codec.decode(io.BytesIO(blob_chunked))
        _m2, audio_whole = codec.decode(io.BytesIO(blob_whole))

        # Both should match the original up to length.
        n = len(full)
        np.testing.assert_array_equal(audio_chunked[:n], full)
        np.testing.assert_array_equal(audio_whole[:n], full)


# --------------------------------------------------------------------------------
# Resampling

class TestDecodeResampling:
    def test_target_sample_rate(self):
        original = _sine_s16(440, 0.2, sample_rate=SAMPLE_RATE)
        blob = codec.encode(original, format="wav", sample_rate=SAMPLE_RATE)

        metadata, decoded = codec.decode(io.BytesIO(blob), target_sample_rate=8000)
        assert metadata["input_sample_rate"] == SAMPLE_RATE  # unchanged by resampling
        # Output length should be roughly halved (16k → 8k).
        assert abs(len(decoded) - len(original) // 2) <= 64

    def test_target_sample_format_float(self):
        original = _sine_s16(440, 0.1)
        blob = codec.encode(original, format="wav", sample_rate=SAMPLE_RATE)

        metadata, decoded = codec.decode(io.BytesIO(blob), target_sample_format="fltp")
        assert metadata["input_sample_format"] in ("s16", "s16p")
        assert decoded.dtype == np.float32
        # After resampling to float, amplitudes should be bounded in roughly [-1, 1].
        assert np.max(np.abs(decoded)) <= 1.0 + 1e-3


# --------------------------------------------------------------------------------
# Streaming API

class TestEncodeStreaming:
    def test_stream_yields_bytes_chunks_and_final(self):
        chunks = [_sine_s16(440, 0.1), _sine_s16(660, 0.1)]
        streamer = codec.encode(chunks, format="wav", sample_rate=SAMPLE_RATE, stream=True)
        # `encode(stream=True)` returns a generator-making function; call it to get the generator.
        produced = list(streamer())
        # One chunk per input array + one finalization chunk.
        assert len(produced) == len(chunks) + 1
        assert all(isinstance(piece, (bytes, bytearray)) for piece in produced)

    def test_stream_roundtrip_equivalent_to_non_stream(self):
        chunks = [_sine_s16(440, 0.1), _sine_s16(880, 0.1)]
        streamer = codec.encode(chunks, format="wav", sample_rate=SAMPLE_RATE, stream=True)
        streamed_bytes = b"".join(streamer())
        batched_bytes = codec.encode(chunks, format="wav", sample_rate=SAMPLE_RATE)
        # Streamed and batched should decode to the same audio.
        _m1, a1 = codec.decode(io.BytesIO(streamed_bytes))
        _m2, a2 = codec.decode(io.BytesIO(batched_bytes))
        np.testing.assert_array_equal(a1, a2)


# --------------------------------------------------------------------------------
# Lossy formats (shape only — no bit-exact comparison)

class TestLossyFormats:
    @pytest.mark.parametrize("format", ["mp3"])
    def test_encode_decode_recognizable(self, format):
        original = _sine_s16(440, 0.3)
        blob = codec.encode(original, format=format, sample_rate=SAMPLE_RATE)
        assert len(blob) > 0

        metadata, decoded = codec.decode(io.BytesIO(blob), target_sample_format="s16", target_layout="mono")
        assert metadata["input_sample_rate"] in (SAMPLE_RATE, 44100, 48000)  # some encoders resample
        # Decoded length should be within an order of magnitude of the original duration.
        # (Lossy codecs pad with silence at the ends.)
        assert 0.5 * len(original) < len(decoded) < 4 * len(original)
