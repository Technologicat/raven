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
        pytest.importorskip("qoi")  # optional C extension; CI installs it, a bare checkout may not

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


# ---------------------------------------------------------------------------
# has_alpha_channel / has_transparency — routing predicates
# ---------------------------------------------------------------------------

class TestAlphaPredicates:
    """These two exist to route a dropped image to one destination or another, so their *difference* is the
    contract, not either one alone. An image exported as RGBA with nothing transparent in it is the case
    that separates them, and the one a "does it have an alpha channel?" test gets wrong.
    """

    def test_an_image_with_no_alpha_channel_answers_no_to_both(self, sample_dir):
        assert codec.has_alpha_channel(sample_dir / "green.jpg") is False
        assert codec.has_transparency(sample_dir / "green.jpg") is False

    def test_an_opaque_rgba_image_has_the_channel_but_no_transparency(self, sample_dir):
        """The distinguishing case: a backdrop saved as RGBA is fully opaque, and is not a cutout."""
        assert codec.has_alpha_channel(sample_dir / "tiny.png") is True
        assert codec.has_transparency(sample_dir / "tiny.png") is False

    def test_an_image_with_a_transparent_pixel_answers_yes_to_both(self, tmp_path):
        img = Image.new("RGBA", (8, 8), (255, 255, 255, 255))
        img.putpixel((0, 0), (255, 255, 255, 0))  # one transparent pixel is enough to make it a cutout
        img.save(tmp_path / "cutout.png")
        assert codec.has_alpha_channel(tmp_path / "cutout.png") is True
        assert codec.has_transparency(tmp_path / "cutout.png") is True

    def test_a_partially_transparent_pixel_counts(self, tmp_path):
        """Anti-aliased cutout edges are partial alpha, so the test cannot be for fully transparent pixels."""
        img = Image.new("RGBA", (8, 8), (255, 255, 255, 255))
        img.putpixel((0, 0), (255, 255, 255, 128))
        img.save(tmp_path / "soft.png")
        assert codec.has_transparency(tmp_path / "soft.png") is True

    def test_a_file_that_is_not_an_image_answers_no_rather_than_raising(self, tmp_path):
        """They are predicates over whatever the user dropped, so "unreadable" is an answer, not an error."""
        not_an_image = tmp_path / "notes.txt"
        not_an_image.write_text("hello")
        assert codec.has_alpha_channel(not_an_image) is False
        assert codec.has_transparency(not_an_image) is False

    def test_a_missing_file_answers_no_rather_than_raising(self, tmp_path):
        assert codec.has_alpha_channel(tmp_path / "gone.png") is False
        assert codec.has_transparency(tmp_path / "gone.png") is False

    def test_a_directory_answers_no_rather_than_raising(self, tmp_path):
        """Drops deliver directories too, and both predicates are asked before anything checks the kind."""
        d = tmp_path / "folder.png"
        d.mkdir()
        assert codec.has_alpha_channel(d) is False
        assert codec.has_transparency(d) is False
