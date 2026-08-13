"""Tests for the shared fixture generators in `raven.common.tests` itself.

Most of them are exercised by whatever uses them — `make_docx` by `test_docextract`, and so on. The demo
image folder is the exception: its purpose is hand-testing a grid app, so no test would notice it breaking.
These assertions are what keep it honest.
"""

import pytest

from raven.common.tests import make_demo_image, write_demo_image_folder
from raven.common.image import codec as imagecodec


def test_demo_image_is_a_decodable_png():
    raw = make_demo_image(3)
    assert raw[:4] == b"\x89PNG"


def test_demo_folder_writes_the_requested_count_in_order(tmp_path):
    paths = write_demo_image_folder(tmp_path, n=5)
    assert [p.name for p in paths] == [f"demo_{i:02d}.png" for i in range(5)]
    assert all(p.exists() for p in paths)


def test_demo_folder_creates_a_directory_that_is_not_there_yet(tmp_path):
    """Called with a path to fill rather than a directory to find, which is how a hand-test uses it."""
    target = tmp_path / "not" / "yet"
    write_demo_image_folder(target, n=2)
    assert target.is_dir()


def test_demo_images_alternate_orientation(tmp_path):
    """The load-bearing property: a grid letterboxes, and uniform shapes cannot show whether it does."""
    paths = write_demo_image_folder(tmp_path, n=4)
    shapes = [imagecodec.decode(p).shape[:2] for p in paths]  # (height, width)
    assert shapes[0][1] > shapes[0][0], "even indices should be landscape"
    assert shapes[1][0] > shapes[1][1], "odd indices should be portrait"
    assert shapes[2] == shapes[0] and shapes[3] == shapes[1]


@pytest.mark.parametrize("index", [0, 5, 11, 12, 23])
def test_hues_cycle_without_running_off_the_end(index):
    """Indices past the palette wrap rather than raising, so a big folder is still generated."""
    assert make_demo_image(index)[:4] == b"\x89PNG"
