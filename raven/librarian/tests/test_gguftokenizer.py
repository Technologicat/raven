"""Tests for `raven.librarian.gguftokenizer`: picking the right GGUF, and building a tokenizer from it."""

import os
import pathlib

import pytest

from .. import gguftokenizer
from ..gguftokenizer import find_for_model, load, _PROBE_TOLERANCE_TOKENS


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
    """With no backend to ask, a construction nothing has measured is a guess nothing can check."""
    assert load(write_gguf(tmp_path / "model.gguf", tokenizer_class="gpt2", pre="something-we-have-not-measured")) is None


# --------------------------------------------------------------------------------
# Checking the built tokenizer against the backend

def counter_agreeing_with(tokenizer, framing: int = 0, ratio: float = 1.0):
    """A stand-in backend: counts `text` with `tokenizer`, scaled by `ratio`, plus `framing` for its template.

    `ratio` away from 1.0 is a backend serving a model whose vocabulary is not this one.
    """
    def count(text: str) -> int:
        return int(len(tokenizer.encode(text, add_special_tokens=False).ids) * ratio) + framing
    return count


@pytest.fixture
def reference(tmp_path):
    """A built tokenizer to speak for the backend, and the path of a file declaring an unmeasured pre."""
    measured = load(write_gguf(tmp_path / "measured.gguf", tokenizer_class="gpt2", pre="qwen35"))
    unmeasured = write_gguf(tmp_path / "unmeasured.gguf", tokenizer_class="gpt2", pre="not-measured-anywhere")
    return measured, unmeasured


def test_an_unmeasured_pre_tokenizer_is_accepted_when_the_backend_confirms_it(reference):
    """The whole point: the model being served is a better authority than a list compiled elsewhere."""
    tokenizer, unmeasured = reference
    assert load(unmeasured, counter_agreeing_with(tokenizer)) is not None


def test_the_backends_own_framing_does_not_matter(reference):
    """Comparing two probes' difference cancels the chat template, which a comparison of totals could not."""
    tokenizer, unmeasured = reference
    framing = 1000
    assert load(unmeasured, counter_agreeing_with(tokenizer, framing=framing)) is not None
    assert framing > _PROBE_TOLERANCE_TOKENS, ("the framing fits inside the tolerance, so this fixture would pass "
                                               "even if the check compared totals rather than the difference")


def test_a_measured_pre_tokenizer_is_declined_when_the_backend_disagrees(tmp_path, reference):
    """A file whose name matched but whose vocabulary belongs to another model builds and round-trips fine."""
    tokenizer, _ = reference
    measured = write_gguf(tmp_path / "measured-again.gguf", tokenizer_class="gpt2", pre="qwen35")
    assert load(measured, counter_agreeing_with(tokenizer, ratio=0.5)) is None
    assert load(measured) is not None, ("this file is declined even with a backend that agrees, so the disagreement "
                                        "above proves nothing")


def test_a_backend_that_cannot_answer_leaves_the_measured_list_deciding(reference):
    tokenizer, unmeasured = reference
    assert load(unmeasured, lambda text: None) is None


def test_a_backend_that_raises_is_not_itself_a_failure(tmp_path, reference):
    """An unreachable backend means "cannot say", so the offline list decides — it does not veto."""
    def unreachable(text: str) -> int:
        raise OSError("connection refused")

    measured = write_gguf(tmp_path / "measured-again.gguf", tokenizer_class="gpt2", pre="qwen35")
    assert load(measured, unreachable) is not None


def test_a_tokenizer_class_we_cannot_build_is_declined(tmp_path):
    """The pieces of an unknown construction fit together in more than one way; only some of them count right."""
    assert load(write_gguf(tmp_path / "model.gguf", tokenizer_class="rwkv", pre=None)) is None


def write_gemma_gguf(path: pathlib.Path, *, with_unk: bool = True) -> pathlib.Path:
    """Write a minimal GGUF shaped like Gemma's: word-mark pieces, byte fallback, no pre-tokenizer."""
    word_mark = "\u2581"  # U+2581 LOWER ONE EIGHTH BLOCK, spelled out: it reads as an underscore on the page
    specials = ["<pad>", "<eos>", "<bos>"] + (["<unk>"] if with_unk else [])
    byte_tokens = [f"<0x{value:02X}>" for value in range(256)]        # what byte fallback falls back to
    characters = [chr(code) for code in range(32, 127)] + [word_mark, "\n", "\t"]
    merges = [f"{word_mark} t"]
    writer = gguf.GGUFWriter(str(path), "test-arch")
    writer.add_tokenizer_model("gemma4")
    writer.add_token_list(specials + byte_tokens + characters + [m.replace(" ", "") for m in merges])
    writer.add_token_merges(merges)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()
    return path


def test_a_word_mark_tokenizer_builds_and_round_trips(tmp_path, monkeypatch):
    """Gemma's construction: a space becomes a word mark, and anything outside the vocabulary becomes bytes.

    Whether this construction is *trusted* with no backend to confirm it is a separate question, decided by
    `_VERIFIED_CONSTRUCTIONS` and tested below; patched here so this test is about the assembly alone.
    """
    monkeypatch.setattr(gguftokenizer, "_VERIFIED_CONSTRUCTIONS", {("gemma4", None)})
    tokenizer = load(write_gemma_gguf(tmp_path / "gemma.gguf"))
    assert tokenizer is not None
    # The vocabulary has no non-ASCII characters, so "ä" can only be represented through byte fallback —
    # which is what makes this a test of that, and not just of the merges.
    assert tokenizer.decode(tokenizer.encode("hä", add_special_tokens=False).ids) == "hä"
    assert tokenizer.decode(tokenizer.encode("a b", add_special_tokens=False).ids) == "a b", "the word mark did not decode back to a space"


def test_a_vocabulary_the_construction_cannot_use_is_declined(tmp_path, monkeypatch):
    """A file claiming Gemma's class while carrying a byte-level vocabulary: no `<unk>`, no byte-fallback pieces.

    `models.BPE` accepts that without complaint and raises only when something is encoded, so the decline has
    to come from the check that encodes rather than from the assembly.
    """
    monkeypatch.setattr(gguftokenizer, "_VERIFIED_CONSTRUCTIONS", {("gemma4", None)})
    assert load(write_gguf(tmp_path / "mismatched.gguf", tokenizer_class="gemma4", pre=None)) is None


def test_a_word_mark_vocabulary_needs_no_unknown_token(tmp_path, monkeypatch):
    """Its 256 byte-fallback pieces can spell anything, so `<unk>` is never reached."""
    monkeypatch.setattr(gguftokenizer, "_VERIFIED_CONSTRUCTIONS", {("gemma4", None)})
    assert load(write_gemma_gguf(tmp_path / "gemma.gguf", with_unk=False)) is not None


def test_both_measured_constructions_are_trusted_without_a_backend(tmp_path):
    """Measured against a served model of each family, so they need no confirmation to be used."""
    assert load(write_gemma_gguf(tmp_path / "gemma.gguf")) is not None
    assert load(write_gguf(tmp_path / "qwen.gguf", tokenizer_class="gpt2", pre="qwen35")) is not None


def test_a_file_that_is_not_a_gguf_is_declined(tmp_path):
    not_a_model = tmp_path / "model.gguf"
    not_a_model.write_bytes(b"this is not a GGUF at all")
    assert load(not_a_model) is None
