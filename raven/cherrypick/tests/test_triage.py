"""Tests for raven.cherrypick.triage — triage state management and file operations."""

import pytest

from raven.cherrypick.triage import (
    TriageState, ImageEntry, TriageManager,
    CHERRY_DIR, LEMON_DIR,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def image_dir(tmp_path):
    """Create a temporary directory with some test images."""
    for name in ["alpha.png", "bravo.jpg", "charlie.png", "delta.jpeg", "echo.qoi"]:
        (tmp_path / name).write_bytes(b"\x00")  # dummy content
    return tmp_path


@pytest.fixture
def triaged_dir(tmp_path):
    """Create a directory with pre-existing triage subdirectories."""
    for name in ["alpha.png", "charlie.png", "echo.qoi"]:
        (tmp_path / name).write_bytes(b"\x00")
    (tmp_path / CHERRY_DIR).mkdir()
    (tmp_path / CHERRY_DIR / "bravo.jpg").write_bytes(b"\x00")
    (tmp_path / LEMON_DIR).mkdir()
    (tmp_path / LEMON_DIR / "delta.jpeg").write_bytes(b"\x00")
    return tmp_path


# ---------------------------------------------------------------------------
# ImageEntry
# ---------------------------------------------------------------------------

class TestImageEntry:
    def test_neutral_path(self, tmp_path):
        e = ImageEntry("foo.png", TriageState.NEUTRAL, tmp_path)
        assert e.path == tmp_path / "foo.png"

    def test_cherry_path(self, tmp_path):
        e = ImageEntry("foo.png", TriageState.CHERRY, tmp_path)
        assert e.path == tmp_path / CHERRY_DIR / "foo.png"

    def test_lemon_path(self, tmp_path):
        e = ImageEntry("foo.png", TriageState.LEMON, tmp_path)
        assert e.path == tmp_path / LEMON_DIR / "foo.png"

    def test_repr(self, tmp_path):
        e = ImageEntry("foo.png", TriageState.CHERRY, tmp_path)
        assert "foo.png" in repr(e)
        assert "cherry" in repr(e)


# ---------------------------------------------------------------------------
# TriageManager — scanning
# ---------------------------------------------------------------------------

class TestTriageManagerScan:
    def test_scan_flat_directory(self, image_dir):
        tm = TriageManager(image_dir)
        assert len(tm) == 5
        # All neutral.
        assert all(e.state is TriageState.NEUTRAL for e in tm.images)
        # Sorted by filename.
        names = [e.filename for e in tm.images]
        assert names == sorted(names)

    def test_scan_with_existing_triage(self, triaged_dir):
        tm = TriageManager(triaged_dir)
        assert len(tm) == 5  # all 5 images found across 3 dirs
        # Check states.
        states = {e.filename: e.state for e in tm.images}
        assert states["alpha.png"] is TriageState.NEUTRAL
        assert states["bravo.jpg"] is TriageState.CHERRY
        assert states["charlie.png"] is TriageState.NEUTRAL
        assert states["delta.jpeg"] is TriageState.LEMON
        assert states["echo.qoi"] is TriageState.NEUTRAL

    def test_scan_ignores_non_images(self, tmp_path):
        (tmp_path / "readme.txt").write_bytes(b"hello")
        (tmp_path / "data.csv").write_bytes(b"1,2,3")
        (tmp_path / "photo.png").write_bytes(b"\x00")
        tm = TriageManager(tmp_path)
        assert len(tm) == 1
        assert tm[0].filename == "photo.png"

    def test_scan_empty_directory(self, tmp_path):
        tm = TriageManager(tmp_path)
        assert len(tm) == 0

    def test_sort_order_stable_across_states(self, triaged_dir):
        """Sort order is by filename, regardless of triage state."""
        tm = TriageManager(triaged_dir)
        names = [e.filename for e in tm.images]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# TriageManager — queries
# ---------------------------------------------------------------------------

class TestTriageManagerQueries:
    def test_index_of(self, image_dir):
        tm = TriageManager(image_dir)
        idx = tm.index_of("charlie.png")
        assert idx is not None
        assert tm[idx].filename == "charlie.png"

    def test_index_of_missing(self, image_dir):
        tm = TriageManager(image_dir)
        assert tm.index_of("nonexistent.png") is None

    def test_count(self, triaged_dir):
        tm = TriageManager(triaged_dir)
        assert tm.count(TriageState.NEUTRAL) == 3
        assert tm.count(TriageState.CHERRY) == 1
        assert tm.count(TriageState.LEMON) == 1


# ---------------------------------------------------------------------------
# TriageManager — state changes
# ---------------------------------------------------------------------------

class TestTriageManagerStateChanges:
    def test_mark_cherry(self, image_dir):
        tm = TriageManager(image_dir)
        idx = tm.index_of("bravo.jpg")
        err = tm.set_state(idx, TriageState.CHERRY)
        assert err is None
        assert tm[idx].state is TriageState.CHERRY
        # File moved.
        assert (image_dir / CHERRY_DIR / "bravo.jpg").exists()
        assert not (image_dir / "bravo.jpg").exists()

    def test_mark_lemon(self, image_dir):
        tm = TriageManager(image_dir)
        idx = tm.index_of("delta.jpeg")
        err = tm.set_state(idx, TriageState.LEMON)
        assert err is None
        assert tm[idx].state is TriageState.LEMON
        assert (image_dir / LEMON_DIR / "delta.jpeg").exists()

    def test_clear_mark(self, triaged_dir):
        tm = TriageManager(triaged_dir)
        idx = tm.index_of("bravo.jpg")
        assert tm[idx].state is TriageState.CHERRY
        err = tm.set_state(idx, TriageState.NEUTRAL)
        assert err is None
        assert tm[idx].state is TriageState.NEUTRAL
        assert (triaged_dir / "bravo.jpg").exists()
        assert not (triaged_dir / CHERRY_DIR / "bravo.jpg").exists()

    def test_change_cherry_to_lemon(self, triaged_dir):
        tm = TriageManager(triaged_dir)
        idx = tm.index_of("bravo.jpg")
        err = tm.set_state(idx, TriageState.LEMON)
        assert err is None
        assert tm[idx].state is TriageState.LEMON
        assert (triaged_dir / LEMON_DIR / "bravo.jpg").exists()
        assert not (triaged_dir / CHERRY_DIR / "bravo.jpg").exists()

    def test_noop_same_state(self, image_dir):
        tm = TriageManager(image_dir)
        idx = tm.index_of("alpha.png")
        err = tm.set_state(idx, TriageState.NEUTRAL)
        assert err is None
        # File didn't move.
        assert (image_dir / "alpha.png").exists()

    def test_collision_detection(self, tmp_path):
        """Refuse to overwrite if destination file already exists."""
        (tmp_path / "alpha.png").write_bytes(b"neutral")
        tm = TriageManager(tmp_path)
        idx = tm.index_of("alpha.png")

        # Plant a same-named blocker in lemons/.
        (tmp_path / LEMON_DIR).mkdir()
        (tmp_path / LEMON_DIR / "alpha.png").write_bytes(b"blocker")

        err = tm.set_state(idx, TriageState.LEMON)
        assert err is not None
        assert "already exists" in err

    def test_grid_position_stable(self, image_dir):
        """Triage state change doesn't alter the image's position in the list."""
        tm = TriageManager(image_dir)
        names_before = [e.filename for e in tm.images]
        idx = tm.index_of("charlie.png")
        tm.set_state(idx, TriageState.CHERRY)
        names_after = [e.filename for e in tm.images]
        assert names_before == names_after

    def test_batch_set_state(self, image_dir):
        tm = TriageManager(image_dir)
        indices = [tm.index_of("alpha.png"), tm.index_of("echo.qoi")]
        errors = tm.set_state(indices, TriageState.CHERRY)
        assert errors == []
        assert tm[indices[0]].state is TriageState.CHERRY
        assert tm[indices[1]].state is TriageState.CHERRY

    def test_creates_subdirs_on_first_use(self, image_dir):
        """Cherry/lemon subdirectories are created automatically."""
        assert not (image_dir / CHERRY_DIR).exists()
        tm = TriageManager(image_dir)
        tm.set_state(0, TriageState.CHERRY)
        assert (image_dir / CHERRY_DIR).is_dir()
