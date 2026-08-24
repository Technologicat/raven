"""Tests for `raven.librarian.gguftokenizer`: picking the right GGUF, and building a tokenizer from it."""

import os
import pathlib

import pytest

from ..gguftokenizer import find_for_model, load


# --------------------------------------------------------------------------------
# Fixtures

def make_model(root: pathlib.Path, directory: str, filename: str, size: int = 1024) -> pathlib.Path:
    """Create a fake `.gguf` of `size` bytes at `root/directory/filename`, and return its path."""
    path = root / directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * size)
    return path


@pytest.fixture
def archive(tmp_path):
    """A model archive shaped like a real one: several families, several quantizations, a vision projector."""
    root = tmp_path / "archive"
    make_model(root, "Qwen3.5-9B", "Qwen3.5-9B-UD-Q4_K_XL.gguf", size=4096)
    make_model(root, "Qwen3.5-9B", "Qwen3.5-9B-mmproj-BF16.gguf", size=512)
    make_model(root, "Qwen3.5-4B", "Qwen3.5-4B-UD-Q4_K_XL.gguf", size=2048)
    make_model(root, "Gemma4-26B-A4B", "gemma-4-26B-A4B-it-Q4_K_M.gguf", size=8192)
    make_model(root, "Gemma4-26B-A4B", "mmproj-gemma-4-26B-A4B-it-BF16.gguf", size=256)
    return root


# --------------------------------------------------------------------------------
# Choosing the file

def test_the_model_the_backend_names_is_the_one_found(archive):
    found = find_for_model(archive, ["qwen3.5-9b"])
    assert found is not None and found.name == "Qwen3.5-9B-UD-Q4_K_XL.gguf"


def test_a_quantization_the_archive_does_not_have_still_matches(archive):
    """A backend names the quantization it loaded, which need not be the one on disk."""
    assert find_for_model(archive, ["qwen3.5-9b@q8_0"]).name == "Qwen3.5-9B-UD-Q4_K_XL.gguf"


def test_a_backend_that_describes_rather_than_names_still_matches(archive):
    """LM Studio's label is a sentence about the model, not an identifier."""
    assert find_for_model(archive, ["qwen3.5-9b, Q4_K_XL, 128 Ki context"]).name == "Qwen3.5-9B-UD-Q4_K_XL.gguf"


def test_a_vision_projector_is_not_a_model(archive):
    """It matches the model's name as well as the model does, carries no tokenizer, and is the smaller file."""
    assert "mmproj" not in find_for_model(archive, ["qwen3.5-9b"]).name.lower()
    assert "mmproj" not in find_for_model(archive, ["gemma4-26b-a4b"]).name.lower()

    projector = archive / "Qwen3.5-9B" / "Qwen3.5-9B-mmproj-BF16.gguf"
    model = archive / "Qwen3.5-9B" / "Qwen3.5-9B-UD-Q4_K_XL.gguf"
    assert projector.stat().st_size < model.stat().st_size, ("the projector is not the smaller file here, so this "
                                                             "fixture cannot tell the name filter from the size "
                                                             "tie-break")


def test_an_unrelated_model_matches_nothing(archive):
    assert find_for_model(archive, ["llama-3.1-8b"]) is None


def test_two_blends_from_one_publisher_do_not_match_each_other(tmp_path):
    """A shared publisher prefix says who packaged the file, and nothing about whose vocabulary is in it."""
    root = tmp_path / "archive"
    make_model(root, "blends", "Nous-Hermes-Llama3-8B-Q4_K_M.gguf")
    assert find_for_model(root, ["Nous-Hermes-Qwen3.5-9B"]) is None, ("a shared prefix must not be a match; it would "
                                                                     "count with another family's vocabulary")


def test_a_publisher_prefix_does_not_prevent_a_match(tmp_path):
    """The community convention puts the packager first, so the model name is not at the start."""
    root = tmp_path / "archive"
    make_model(root, "blends", "TheDrummer-Qwen3.5-9B-Tuned-Q4_K_M.gguf")
    assert find_for_model(root, ["TheDrummer-Qwen3.5-9B-Tuned"]) is not None


def test_the_smallest_of_equally_good_matches_wins(tmp_path):
    """Quantizations of one model carry the same tokenizer, so the cheapest to read is the one to read."""
    root = tmp_path / "archive"
    make_model(root, "Qwen3.5-9B", "Qwen3.5-9B-Q8_0.gguf", size=9000)
    make_model(root, "Qwen3.5-9B", "Qwen3.5-9B-Q4_K_XL.gguf", size=4000)
    assert find_for_model(root, ["qwen3.5-9b"]).name == "Qwen3.5-9B-Q4_K_XL.gguf"


def test_a_model_reachable_only_through_a_symlink_is_found(tmp_path):
    """A model archive shared between backends is typically a tree of links into one central copy."""
    central = tmp_path / "central"
    make_model(central, "Qwen3.5-9B", "Qwen3.5-9B-UD-Q4_K_XL.gguf")
    root = tmp_path / "archive"
    root.mkdir()
    os.symlink(central / "Qwen3.5-9B", root / "Qwen3.5-9B", target_is_directory=True)

    assert not list(root.glob("**/*.gguf")), ("the file is reachable without following symlinks, so this fixture "
                                              "cannot tell a symlink-following walk from a plain one")
    assert find_for_model(root, ["qwen3.5-9b"]) is not None


def test_a_directory_that_is_not_there_is_not_an_error(tmp_path):
    assert find_for_model(tmp_path / "no-such-archive", ["qwen3.5-9b"]) is None


# --------------------------------------------------------------------------------
# Building the tokenizer

gguf = pytest.importorskip("gguf", reason="needs the GGUF reader/writer")
tokenizers = pytest.importorskip("tokenizers", reason="needs the tokenizers library")


def write_gguf(path: pathlib.Path, *, tokenizer_class: str, pre: str | None, merges=("l l",)) -> pathlib.Path:
    """Write a minimal GGUF carrying a byte-level vocabulary and `merges`, and return its path.

    The vocabulary is every byte-level character plus whatever the merges produce, which is a valid
    byte-level BPE — enough for `load` to build, self-check and count with.
    """
    merged_tokens = [merge.replace(" ", "") for merge in merges]
    writer = gguf.GGUFWriter(str(path), "test-arch")
    writer.add_tokenizer_model(tokenizer_class)
    if pre is not None:
        writer.add_tokenizer_pre(pre)
    writer.add_token_list(sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet()) + merged_tokens)
    if merges:
        writer.add_token_merges(list(merges))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()
    return path


def test_a_verified_tokenizer_loads_and_counts(tmp_path):
    tokenizer = load(write_gguf(tmp_path / "model.gguf", tokenizer_class="gpt2", pre="qwen35"))
    assert tokenizer is not None
    # One token per byte-level character, except that "ll" merges: h, e, ll, o. Four rather than five is
    # also what says the merges were read — a vocabulary whose merges went missing counts every character.
    assert len(tokenizer.encode("hello", add_special_tokens=False).ids) == 4


def test_a_vocabulary_with_no_merges_is_declined(tmp_path):
    """It would build, round-trip, and quietly count one token per character — a wrong number without a `~`."""
    assert load(write_gguf(tmp_path / "model.gguf", tokenizer_class="gpt2", pre="qwen35", merges=())) is None


def test_an_unverified_pre_tokenizer_is_declined(tmp_path):
    """Building it anyway would produce counts that are wrong without being marked as estimates."""
    assert load(write_gguf(tmp_path / "model.gguf", tokenizer_class="gpt2", pre="something-we-have-not-measured")) is None


def test_a_tokenizer_class_we_cannot_build_is_declined(tmp_path):
    """Gemma's GGUFs carry no pre-tokenizer field at all."""
    assert load(write_gguf(tmp_path / "model.gguf", tokenizer_class="gemma4", pre=None)) is None


def test_a_file_that_is_not_a_gguf_is_declined(tmp_path):
    not_a_model = tmp_path / "model.gguf"
    not_a_model.write_bytes(b"this is not a GGUF at all")
    assert load(not_a_model) is None
