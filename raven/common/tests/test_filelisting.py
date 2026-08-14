"""Tests for `raven.common.filelisting` — what a file browser's listing must guarantee.

These are the first tests the file dialog's listing has ever had. It was untestable while it existed only as
DPG widgets, which is the reason this module was split out; the invariants below are the ones a listing has
to hold for a picker built on it to be usable at all.
"""

import os
import pathlib

import pytest

from raven.common import filelisting


@pytest.fixture
def tree(tmp_path):
    """A small directory: two subdirectories (one hidden), four files (one hidden), varied sizes."""
    (tmp_path / "sub").mkdir()
    (tmp_path / ".hiddendir").mkdir()
    (tmp_path / "b.txt").write_bytes(b"x" * 300)
    (tmp_path / "a.png").write_bytes(b"x" * 100)
    (tmp_path / "z.txt").write_bytes(b"x" * 200)
    (tmp_path / ".hidden").write_bytes(b"x")
    return tmp_path


def names(entries):
    return [entry.name for entry in entries]


# --------------------------------------------------------------------------------
# The parent entry

def test_the_parent_entry_comes_first(tree):
    entries = filelisting.list_directory(str(tree))
    assert entries[0].name == os.pardir
    assert entries[0].is_parent is True
    assert entries[0].is_dir is True


def test_the_parent_entry_survives_a_filter_that_matches_nothing(tree):
    """It is the only way up, so filtering it out would strand the user in the directory."""
    entries = filelisting.list_directory(str(tree), name_filter=lambda name: False)

    assert names(entries) == [os.pardir]


def test_the_parent_entry_points_at_the_parent(tree):
    entries = filelisting.list_directory(str(tree))
    assert entries[0].path == str(tree.parent)


def test_the_parent_entry_can_be_left_out(tree):
    assert os.pardir not in names(filelisting.list_directory(str(tree), include_parent=False))


# --------------------------------------------------------------------------------
# Paths are resolved against the argument, not the process

def test_paths_resolve_against_the_listed_directory_not_the_cwd(tree, tmp_path_factory, monkeypatch):
    """The failure this prevents: a listing is correct only while cwd happens to equal its argument.

    Every path a caller gets back — for opening the file, for re-anchoring a cursor — has to name the
    directory that was listed, whatever directory the process is sitting in.
    """
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    monkeypatch.chdir(elsewhere)

    entries = filelisting.list_directory(str(tree), include_parent=False)

    assert entries, "expected a non-empty listing"
    for entry in entries:
        assert pathlib.Path(entry.path).parent == tree
        assert os.path.isabs(entry.path)


def test_the_kinds_are_right_when_listing_a_directory_that_is_not_the_cwd(tree, tmp_path_factory, monkeypatch):
    """The same hazard, in the form that misclassifies rather than mislocates."""
    monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere"))

    entries = filelisting.list_directory(str(tree), include_parent=False)
    by_name = {entry.name: entry for entry in entries}

    assert by_name["sub"].is_dir is True
    assert by_name["b.txt"].is_dir is False


# --------------------------------------------------------------------------------
# Grouping and ordering

def test_directories_precede_files(tree):
    entries = filelisting.list_directory(str(tree), include_parent=False)
    kinds = [entry.kind for entry in entries]

    assert kinds == sorted(kinds, key=lambda kind: kind != filelisting.KIND_DIR)


def test_directories_still_precede_files_when_reversed(tree):
    """Reversing sorts *within* the groups; it does not interleave them."""
    entries = filelisting.list_directory(str(tree), include_parent=False, descending=True)

    assert entries[0].is_dir is True
    assert names(entries)[1:] == ["z.txt", "b.txt", "a.png"]


def test_sorting_by_name(tree):
    entries = filelisting.list_directory(str(tree), include_parent=False)
    assert names(entries) == ["sub", "a.png", "b.txt", "z.txt"]


def test_sorting_by_size(tree):
    entries = filelisting.list_directory(str(tree), include_parent=False,
                                         sort_key=filelisting.SortKey.SIZE)
    assert names(entries)[1:] == ["a.png", "z.txt", "b.txt"]  # 100, 200, 300 bytes


def test_sorting_by_date(tree):
    os.utime(tree / "a.png", (0, 0))  # oldest by a wide margin
    entries = filelisting.list_directory(str(tree), include_parent=False,
                                         sort_key=filelisting.SortKey.DATE)
    assert names(entries)[1] == "a.png"


# --------------------------------------------------------------------------------
# Filtering

def test_hidden_entries_are_omitted_by_default(tree):
    listed = names(filelisting.list_directory(str(tree)))
    assert ".hidden" not in listed
    assert ".hiddendir" not in listed


def test_hidden_entries_can_be_shown(tree):
    listed = names(filelisting.list_directory(str(tree), show_hidden=True))
    assert ".hidden" in listed
    assert ".hiddendir" in listed


def test_the_type_filter_applies_to_files_only(tree):
    """A type filter selects among files. Hiding the directories would remove the way to reach the files."""
    entries = filelisting.list_directory(str(tree), include_parent=False,
                                         type_filter=lambda name: name.endswith(".txt"))

    assert "sub" in names(entries)
    assert "a.png" not in names(entries)


def test_the_name_filter_applies_to_directories_too(tree):
    """Unlike the type filter — the find field is a search over everything shown."""
    entries = filelisting.list_directory(str(tree), include_parent=False,
                                         name_filter=lambda name: name.startswith("s"))

    assert names(entries) == ["sub"]


def test_dirs_only_omits_files(tree):
    entries = filelisting.list_directory(str(tree), include_parent=False, dirs_only=True)
    assert names(entries) == ["sub"]


# --------------------------------------------------------------------------------
# Entries that cannot be read

@pytest.fixture
def dangling_link(tree):
    """A symlink whose target does not exist, or a skip where the platform will not make one."""
    try:
        (tree / "dangling").symlink_to(tree / "nonexistent")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available here")
    return tree


def test_a_broken_link_is_listed_and_says_so(dangling_link):
    """It is in the directory, so omitting it makes the listing disagree with the filesystem — and it is
    exactly the thing a user goes looking for when a file they expected seems to be missing.
    """
    entries = filelisting.list_directory(str(dangling_link), include_parent=False)
    by_name = {entry.name: entry for entry in entries}

    assert by_name["dangling"].kind == filelisting.KIND_BROKEN_LINK
    assert by_name["dangling"].size is None
    assert by_name["dangling"].mtime is not None  # the link's own timestamp; the target has none


def test_a_broken_link_does_not_cost_the_rest_of_the_listing(dangling_link):
    """The failure this prevents: one unreadable entry raising out of the whole directory read."""
    assert "b.txt" in names(filelisting.list_directory(str(dangling_link), include_parent=False))


def test_a_broken_link_groups_with_the_files(dangling_link):
    """It is not somewhere you can navigate to, so it does not belong among the directories."""
    entries = filelisting.list_directory(str(dangling_link), include_parent=False)
    kinds = [entry.kind for entry in entries]

    assert kinds.index(filelisting.KIND_BROKEN_LINK) > kinds.index(filelisting.KIND_DIR)


def test_dirs_only_omits_broken_links(dangling_link):
    """A directory picker offers directories; a broken link is not one, whatever it points at."""
    entries = filelisting.list_directory(str(dangling_link), include_parent=False, dirs_only=True)
    assert names(entries) == ["sub"]


def test_sizes_are_none_for_directories_unless_asked_for(tree):
    entries = filelisting.list_directory(str(tree), include_parent=False)
    by_name = {entry.name: entry for entry in entries}

    assert by_name["sub"].size is None
    assert by_name["b.txt"].size == 300


def test_directory_sizes_are_computed_on_request(tree):
    (tree / "sub" / "inner.bin").write_bytes(b"x" * 50)

    entries = filelisting.list_directory(str(tree), include_parent=False, dir_sizes=True)
    by_name = {entry.name: entry for entry in entries}

    assert by_name["sub"].size == 50


# --------------------------------------------------------------------------------
# Display helpers

@pytest.mark.parametrize("size, expected", [(None, "-"),
                                            (0, "0 B"),
                                            (512, "512 B"),
                                            (2**20, "1 MB"),
                                            (3 * 2**30, "3 GB")])
def test_format_size(size, expected):
    assert filelisting.format_size(size) == expected


def test_format_mtime_has_an_answer_for_no_answer():
    assert filelisting.format_mtime(None) == "-"
    assert filelisting.format_mtime(0.0) != "-"
