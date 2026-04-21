"""Tests for `raven.common.image.codec` — image decode / encode."""

import io
import pathlib

import numpy as np
import pytest
from PIL import Image

from raven.common.image import codec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_dir(tmp_path):
    """Create a temp directory with synthetic test images in several formats."""
    # PNG RGBA — 200×100 red gradient.
    img = Image.new("RGBA", (200, 100), (0, 0, 0, 255))
    pixels = img.load()
    for x in range(200):
        for y in range(100):
            pixels[x, y] = (int(x / 200 * 255), 0, 0, 255)
    img.save(tmp_path / "red_gradient.png")

    # JPEG — 150×150 green. No alpha channel possible; natural RGB output.
    img = Image.new("RGB", (150, 150), (0, 200, 0))
    img.save(tmp_path / "green.jpg", quality=90)

    # JPEG — 300×200 blue. Big enough to exercise the turbojpeg `max_size` scaled-decode path.
    img = Image.new("RGB", (300, 200), (0, 0, 200))
    img.save(tmp_path / "blue.jpg", quality=90)

    # Small PNG RGBA — 16×16 white.
    img = Image.new("RGBA", (16, 16), (255, 255, 255, 255))
    img.save(tmp_path / "tiny.png")

    return tmp_path


# ---------------------------------------------------------------------------
# decode — input polymorphism and natural channel counts
# ---------------------------------------------------------------------------

class TestDecodeNaturalChannels:
    def test_png_rgba(self, sample_dir):
        arr = codec.decode(sample_dir / "red_gradient.png")
        assert arr.dtype == np.uint8
        assert arr.shape == (100, 200, 4)  # PNG RGBA → 4 channels

    def test_jpeg_rgb(self, sample_dir):
        """JPEG has no alpha; codec returns RGB (3 channels) without synthesizing one."""
        arr = codec.decode(sample_dir / "green.jpg")
        assert arr.dtype == np.uint8
        assert arr.shape == (150, 150, 3)


class TestDecodeInputPolymorphism:
    """`decode` accepts path, bytes, or binary file-like."""
    def test_accepts_path(self, sample_dir):
        arr = codec.decode(sample_dir / "tiny.png")
        assert arr.shape[:2] == (16, 16)

    def test_accepts_str_path(self, sample_dir):
        arr = codec.decode(str(sample_dir / "tiny.png"))
        assert arr.shape[:2] == (16, 16)

    def test_accepts_bytes(self, sample_dir):
        raw = pathlib.Path(sample_dir / "tiny.png").read_bytes()
        arr = codec.decode(raw)
        assert arr.shape[:2] == (16, 16)

    def test_accepts_bytesio(self, sample_dir):
        raw = pathlib.Path(sample_dir / "tiny.png").read_bytes()
        arr = codec.decode(io.BytesIO(raw))
        assert arr.shape[:2] == (16, 16)


class TestDecodeMaxSize:
    def test_max_size_hint_does_not_crash(self, sample_dir):
        """`max_size` is a hint — exact behavior depends on turbojpeg availability.

        With turbojpeg: dimensions scaled down to nearest factor ≥ max_size.
        Without: ignored; full-resolution image returned.
        """
        arr = codec.decode(sample_dir / "blue.jpg", max_size=64)
        assert arr.dtype == np.uint8
        assert arr.shape[2] == 3  # JPEG → RGB
        # With or without turbojpeg: at least one dimension should still be ≥ max_size.
        assert arr.shape[0] >= 64 or arr.shape[1] >= 64


class TestDecodeErrors:
    def test_nonexistent_path_raises(self, tmp_path):
        with pytest.raises(Exception):
            codec.decode(tmp_path / "nonexistent.png")


# ---------------------------------------------------------------------------
# encode — round-trip via decode
# ---------------------------------------------------------------------------

class TestEncodeRoundTrip:
    def test_png_round_trip(self):
        original = np.zeros((8, 16, 4), dtype=np.uint8)
        original[..., 0] = 200  # red channel
        original[..., 3] = 255  # opaque

        encoded = codec.encode(original, "png")
        assert isinstance(encoded, bytes) and len(encoded) > 0

        decoded = codec.decode(encoded)
        assert decoded.shape == original.shape
        # PNG is lossless, so round-trip should be bit-identical.
        assert np.array_equal(decoded, original)

    def test_qoi_round_trip(self):
        original = np.zeros((8, 16, 4), dtype=np.uint8)
        original[..., 1] = 128  # green channel
        original[..., 3] = 200  # partially transparent

        encoded = codec.encode(original, "qoi")
        assert isinstance(encoded, bytes) and len(encoded) > 0
        assert encoded.startswith(b"qoif")  # QOI file magic

        decoded = codec.decode(encoded)
        assert decoded.shape == original.shape
        assert np.array_equal(decoded, original)

    def test_encode_accepts_rgb_input(self):
        """Three-channel input encodes without alpha; round-trip preserves the 3 channels."""
        original = np.full((8, 16, 3), 100, dtype=np.uint8)
        encoded = codec.encode(original, "png")
        decoded = codec.decode(encoded)
        # PIL's PNG encoder preserves RGB-only when no alpha is provided.
        assert decoded.shape == original.shape
        assert np.array_equal(decoded, original)
