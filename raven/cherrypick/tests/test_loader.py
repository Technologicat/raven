"""Tests for raven.cherrypick.loader — thumbnail pipeline."""

import time

import numpy as np
import pytest
import torch
from PIL import Image

from raven.cherrypick.loader import ThumbnailPipeline
from raven.common.image import codec as imagecodec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_dir(tmp_path):
    """Create a temp directory with a few synthetic test images."""
    img = Image.new("RGBA", (200, 100), (255, 0, 0, 255))
    img.save(tmp_path / "red.png")

    img = Image.new("RGB", (150, 150), (0, 200, 0))
    img.save(tmp_path / "green.jpg", quality=90)

    img = Image.new("RGB", (300, 200), (0, 0, 200))
    img.save(tmp_path / "blue.jpg", quality=90)

    img = Image.new("RGBA", (16, 16), (255, 255, 255, 255))
    img.save(tmp_path / "tiny.png")

    return tmp_path


@pytest.fixture
def image_paths(sample_dir):
    """Sorted list of image paths from sample_dir."""
    return sorted(f for f in sample_dir.iterdir()
                  if f.suffix.lower() in imagecodec.IMAGE_EXTENSIONS)


@pytest.fixture
def device():
    """Best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# ThumbnailPipeline
# ---------------------------------------------------------------------------

class TestThumbnailPipeline:
    def test_produces_all_thumbnails(self, image_paths, device):
        """Pipeline should produce one thumbnail per input image."""
        pipeline = ThumbnailPipeline(device=device, dtype=torch.float32, tile_size=32)
        pipeline.start(image_paths)

        results = {}
        deadline = time.monotonic() + 30.0
        while len(results) < len(image_paths) and time.monotonic() < deadline:
            for idx, flat in pipeline.poll():
                results[idx] = flat
            time.sleep(0.05)

        pipeline.shutdown()
        assert len(results) == len(image_paths), \
            f"Expected {len(image_paths)} thumbnails, got {len(results)}"

    def test_thumbnail_shape(self, image_paths, device):
        """Each thumbnail should be a flat float32 array of the right size."""
        tile_size = 48
        pipeline = ThumbnailPipeline(device=device, dtype=torch.float32,
                                     tile_size=tile_size)
        pipeline.start(image_paths)

        results = {}
        deadline = time.monotonic() + 30.0
        while len(results) < len(image_paths) and time.monotonic() < deadline:
            for idx, flat in pipeline.poll():
                results[idx] = flat
            time.sleep(0.05)

        pipeline.shutdown()

        expected_len = tile_size * tile_size * 4
        for idx, flat in results.items():
            assert flat.dtype == np.float32
            assert flat.shape == (expected_len,), \
                f"Index {idx}: expected shape ({expected_len},), got {flat.shape}"

    def test_thumbnail_values_in_range(self, image_paths, device):
        """Thumbnail values should be in [0, 1] (clamped)."""
        pipeline = ThumbnailPipeline(device=device, dtype=torch.float32, tile_size=32)
        pipeline.start(image_paths)

        results = {}
        deadline = time.monotonic() + 30.0
        while len(results) < len(image_paths) and time.monotonic() < deadline:
            for idx, flat in pipeline.poll():
                results[idx] = flat
            time.sleep(0.05)

        pipeline.shutdown()

        for idx, flat in results.items():
            assert flat.min() >= 0.0, f"Index {idx}: min value {flat.min()}"
            assert flat.max() <= 1.0, f"Index {idx}: max value {flat.max()}"

    def test_cancel(self, sample_dir, device):
        """Cancellation should stop the pipeline without hanging."""
        # Create many images so the pipeline has work to do.
        for i in range(50):
            img = Image.new("RGB", (100, 100), (i * 5, 0, 0))
            img.save(sample_dir / f"cancel_test_{i:03d}.jpg")

        paths = sorted(f for f in sample_dir.iterdir()
                       if f.suffix.lower() in imagecodec.IMAGE_EXTENSIONS)

        pipeline = ThumbnailPipeline(device=device, dtype=torch.float32, tile_size=32)
        pipeline.start(paths)
        time.sleep(0.2)  # let it start working
        pipeline.cancel()
        # Should not hang.
        pipeline.shutdown()

    def test_restart(self, image_paths, device):
        """Starting a new batch should cancel the previous one."""
        pipeline = ThumbnailPipeline(device=device, dtype=torch.float32, tile_size=32)

        # First batch.
        pipeline.start(image_paths)
        time.sleep(0.1)

        # Restart with same images.
        pipeline.start(image_paths)

        results = {}
        deadline = time.monotonic() + 30.0
        while len(results) < len(image_paths) and time.monotonic() < deadline:
            for idx, flat in pipeline.poll():
                results[idx] = flat
            time.sleep(0.05)

        pipeline.shutdown()
        assert len(results) == len(image_paths)

    def test_empty_start(self, device):
        """Starting with no images should not crash."""
        pipeline = ThumbnailPipeline(device=device, dtype=torch.float32, tile_size=32)
        pipeline.start([])
        assert pipeline.poll() == []
        pipeline.shutdown()

    def test_progress_tracking(self, image_paths, device):
        """total and completed properties should track progress."""
        pipeline = ThumbnailPipeline(device=device, dtype=torch.float32, tile_size=32)
        pipeline.start(image_paths)
        assert pipeline.total == len(image_paths)
        assert pipeline.completed == 0

        deadline = time.monotonic() + 30.0
        while pipeline.completed < len(image_paths) and time.monotonic() < deadline:
            pipeline.poll()
            time.sleep(0.05)

        pipeline.shutdown()
        assert pipeline.completed == len(image_paths)
