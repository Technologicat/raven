"""Read a tokenizer out of a GGUF file, and find the GGUF that goes with a served model.

A llama.cpp-family backend serves a `.gguf`, and that file carries the model's whole vocabulary and merge
list. So any machine keeping a copy of the model can count its tokens exactly, offline, without asking the
backend — which matters because a backend's `usage["prompt_tokens"]` is not reliably about the whole prompt
(measured twice, in `investigations/prompt-size-cache-relative/`).

Two jobs, because a machine that keeps several models has to answer "which file" before "which tokenizer":

  - `find_for_model` picks the file whose name matches what the backend says it is serving.
  - `load` builds a `tokenizers.Tokenizer` from that file.

**Only constructions that have been checked against a live backend are built.** A tokenizer assembled from
plausible-looking parts produces confidently wrong numbers, which is worse than the estimate it would
replace, because the readout stops saying `~`. `load` therefore declines anything outside `_PRE_TOKENIZER_REGEXES`
and lets the caller fall back. Adding an entry means measuring one: point
`investigations/prompt-size-cache-relative/measure_true_size.py` at a backend serving that model and check
its offline count against the backend's count of the same unique text.

Measured 2026-08-24 on `qwen3.5-9b`: 0.05% apart, the gap being the framing the backend request added.
Measured the same day: Qwen 3.5, 3.6 and 3.8 — dense and MoE — ship a byte-identical tokenizer, so a
near-miss inside that family costs nothing. Across families it would be silently wrong, which is what the
matching below is careful about.
"""

__all__ = ["find_for_model", "load"]

import logging
import os
import pathlib
import re
from typing import Any, Collection, Optional

logger = logging.getLogger(__name__)

# Quantization and file-format markers, dropped from a name before matching so that the same model at two
# bit depths reads as the same model. Whole tokens only, so a name that merely contains these letters is
# untouched. The backend names the quantization too (LM Studio reports `qwen3.8-27b@q4_k_xl`), and it need
# not be the one on disk.
_QUANTIZATION_NOISE = re.compile(r"(?<![a-z0-9])(?:i?q\d+(?:_[a-z0-9]+)*|ud|bf16|f16|f32|gguf|mtp)(?![a-z0-9])",
                                 re.IGNORECASE)

# GGUF names the tokenizer's *class* in `tokenizer.ggml.model` and its pre-tokenizer variant in
# `tokenizer.ggml.pre`. The class says how the pieces fit together; the pre says where the text is cut
# before they do, and getting that wrong shifts counts by a few percent — silently, since the result still
# tokenizes. Keyed on the pre, therefore, not on the model architecture: architectures come and go while
# a family keeps its pre-tokenizer, and `transformers`' own GGUF reader gates on architecture and so
# refuses a model it could otherwise handle ("architecture qwen35 is not supported yet", measured).
_GPT2_FAMILY_REGEX = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
                      r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
_PRE_TOKENIZER_REGEXES = {"qwen35": _GPT2_FAMILY_REGEX}

# A vision projector rides beside its model under a name that matches the model's just as well, and carries
# no tokenizer. Anywhere in the name, not just at the front: both `mmproj-gemma-4-26B-A4B-it-BF16.gguf` and
# `Qwen3.5-9B-mmproj-BF16.gguf` are in use, and a projector is also the smaller file, so it wins the
# size tie-break and disables the feature it was mistaken for.
_NOT_A_MODEL = re.compile(r"mmproj", re.IGNORECASE)

# Round-trip probe. Digits, punctuation runs, non-ASCII letters and newlines are where a mis-assembled
# byte-level BPE stops being reversible, so they are all in here.
_SELF_CHECK_SAMPLE = "Hello, world! 3.14159 — ei se mitään.\n\tKuinka monta? 42 tokens…  ✓"


def _normalize(name: str) -> str:
    """Reduce a model or file name to comparable form: quantization dropped, lowercase, alphanumerics only.

    `Qwen3.5-9B-UD-Q4_K_XL` and `qwen3.5-9b` both become `qwen359b`, which is what lets a name written by a
    backend match one written by whoever packaged the file.
    """
    return re.sub(r"[^a-z0-9]", "", _QUANTIZATION_NOISE.sub("", name.lower()))


def _match_score(candidate: pathlib.Path, wanted: Collection[str]) -> int:
    """How well `candidate` matches any of the model names in `wanted`. Higher is better; 0 is no match.

    A file's own name and its parent directory's are both tried, since a model archive usually spells the
    identity in the directory and appends the quantization to the file.
    """
    # Containment either way, and nothing weaker. Two names agreeing only in part is not evidence that they
    # are the same model: community blends put the publisher first, so a shared opening says the packager is
    # the same and nothing about the vocabulary, while a shared tail is often just the parameter count. The
    # cost of being wrong here is one-directional — a missed match falls back to the estimate and says so in
    # the log, where a wrong match silently counts with another model's vocabulary, and would pass the
    # round-trip check in `load` while doing it.
    names = [_normalize(candidate.stem), _normalize(candidate.parent.name)]
    best = 0
    for want in (_normalize(name) for name in wanted if name):
        if not want:
            continue
        for name in names:
            if name and (want in name or name in want):
                # Score by what actually agreed, which is the shorter of the two — a backend that describes
                # the model rather than naming it ("qwen3.5-9b, Q4_K_XL, 128 Ki context") would otherwise
                # score higher for containing a short file name than an exact match does.
                best = max(best, min(len(want), len(name)))
    return best


def _gguf_files_under(search_root: pathlib.Path):
    """Yield every `.gguf` under `search_root`, following symlinks, visiting no directory twice."""
    # Following them is the point: a model archive shared between backends is typically a tree of symlinks
    # into one central copy, and `pathlib.Path.glob("**/*.gguf")` does not follow those — measured on such a
    # tree, it found 2 files where this finds 13. Following symlinks can also walk in circles, hence the
    # set of directories already visited, compared by real path so two routes to one directory are one entry.
    seen = set()
    for directory, subdirectories, filenames in os.walk(search_root, followlinks=True):
        seen.add(os.path.realpath(directory))
        subdirectories[:] = [name for name in subdirectories
                             if os.path.realpath(os.path.join(directory, name)) not in seen]
        for filename in filenames:
            if filename.lower().endswith(".gguf"):
                yield pathlib.Path(directory) / filename


def find_for_model(search_root: pathlib.Path, model_names: Collection[str]) -> Optional[pathlib.Path]:
    """Find the `.gguf` under `search_root` that belongs to the model named by `model_names`. `None` if none does.

    `model_names`: what the backend calls the loaded model — pass every spelling available (its label and
                   its id), since backends differ about which one is descriptive.

    Symlinks are followed, so `search_root` may be a tree of links into a central model archive.
    """
    if not search_root.is_dir():
        logger.warning(f"find_for_model: '{search_root}' is not a directory; no local tokenizer.")
        return None

    scored = []
    for candidate in _gguf_files_under(search_root):
        if _NOT_A_MODEL.search(candidate.name):
            continue
        score = _match_score(candidate, model_names)
        if score:
            scored.append((score, candidate))

    if not scored:
        logger.info(f"find_for_model: nothing under '{search_root}' matches {sorted(set(model_names))}; keeping the token estimate.")
        return None

    # Among equally good matches, the smallest file. One model is often kept at several quantizations, and
    # those carry the same tokenizer (measured: two quantizations of one model, byte-identical vocabulary and
    # merges) — so the choice is free, and reading the smaller file is several seconds faster.
    scored.sort(key=lambda pair: (-pair[0], pair[1].stat().st_size, str(pair[1])))
    best_score, best = scored[0]
    runners_up = ", ".join(f"{path.name} ({score})" for score, path in scored[1:4])
    logger.info(f"find_for_model: {sorted(set(model_names))} -> '{best}' (score {best_score})"
                + (f"; also considered {runners_up}" if runners_up else ""))
    return best


def load(gguf_path: pathlib.Path) -> Optional[Any]:
    """Build the tokenizer stored in `gguf_path`. Returns a `tokenizers.Tokenizer`, or `None`.

    `None` means the file's tokenizer is not one this module has been verified to reproduce, or the file
    could not be read, or the result failed its own round-trip check. Every one of those is logged, and every
    one leaves the caller to fall back to estimating.

    Reading is slow enough to matter — measured at ~7 s, nearly all of it in the GGUF reader indexing the
    file's tensor metadata on the way past — so call this off any thread that must stay responsive.
    """
    try:
        import gguf  # noqa: PLC0415 -- heavy, and only this path needs it
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, Regex  # noqa: PLC0415
    except ImportError as exc:
        logger.warning(f"load: cannot read '{gguf_path}': {type(exc)}: {exc}. Falling back to token estimates.")
        return None

    try:
        reader = gguf.GGUFReader(str(gguf_path))

        def field(key: str) -> Any:
            return reader.fields[key].contents()

        tokenizer_class = field("tokenizer.ggml.model")
        pre = field("tokenizer.ggml.pre") if "tokenizer.ggml.pre" in reader.fields else None
        regex = _PRE_TOKENIZER_REGEXES.get(pre)
        if regex is None:
            logger.info(f"load: '{gguf_path.name}' has tokenizer class {tokenizer_class!r}, pre-tokenizer {pre!r}, "
                        f"which this module has not been verified against; keeping the token estimate. "
                        f"Verified: {sorted(_PRE_TOKENIZER_REGEXES)}.")
            return None

        vocabulary = {token: index for index, token in enumerate(field("tokenizer.ggml.tokens"))}
        merges = [tuple(merge.split(" ", 1)) for merge in field("tokenizer.ggml.merges")]

        tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=merges, fuse_unk=False, byte_fallback=False))
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(regex), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        tokenizer.decoder = decoders.ByteLevel()
    except Exception as exc:  # noqa: BLE001 -- any failure here just means "no local tokenizer"
        logger.warning(f"load: could not build a tokenizer from '{gguf_path}': {type(exc)}: {exc}. Falling back to token estimates.")
        return None

    # Ask the thing we just assembled whether it is reversible before anyone counts with it. A tokenizer
    # built from mismatched parts still returns a number, and that number would be shown without the `~`
    # that marks an estimate — so an unchecked build trades a known approximation for an unknown error.
    round_tripped = tokenizer.decode(tokenizer.encode(_SELF_CHECK_SAMPLE, add_special_tokens=False).ids)
    if round_tripped != _SELF_CHECK_SAMPLE:
        logger.warning(f"load: the tokenizer built from '{gguf_path}' failed its round-trip check "
                       f"({round_tripped!r} != {_SELF_CHECK_SAMPLE!r}); falling back to token estimates.")
        return None

    logger.info(f"load: tokenizer ready from '{gguf_path.name}': {len(vocabulary)} tokens, class {tokenizer_class!r}, pre-tokenizer {pre!r}.")
    return tokenizer
