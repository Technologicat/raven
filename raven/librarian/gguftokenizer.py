"""Read a tokenizer out of a GGUF file, and find the GGUF that goes with a served model.

A llama.cpp-family backend serves a `.gguf`, and that file carries the model's whole vocabulary and merge
list. So any machine keeping a copy of the model can count its tokens exactly, offline, without asking the
backend — which matters because a backend's `usage["prompt_tokens"]` is not reliably about the whole prompt
(measured twice, in `investigations/prompt-size-cache-relative/`).

Two jobs, because a machine that keeps several models has to answer "which file" before "which tokenizer":

  - `find_for_model` picks the file whose name matches what the backend says it is serving.
  - `load` builds a `tokenizers.Tokenizer` from that file.

**Nothing is trusted until something checks it.** A tokenizer assembled from plausible-looking parts
produces confidently wrong numbers, which is worse than the estimate it replaces, because the readout stops
saying `~`. So `load` builds optimistically and then asks the backend to confirm: two short probes, compared
by the *difference* between them so the chat template's framing cancels. That check is about the model
actually being served, which is the only authority that matters, and it also catches a wrong file — one
whose name matched while its vocabulary belongs to another model, which would build and round-trip perfectly.

When the backend cannot be asked, `_VERIFIED_CONSTRUCTIONS` is the fallback: `(class, pre-tokenizer)` pairs
measured in advance, on the grounds that a guess nothing can check should not be made. Measured 2026-08-24 on
`qwen3.5-9b`, whose offline count came within 0.05% of the backend's own, the gap being framing the
measurement added. The same day: Qwen 3.5, 3.6 and 3.8 — dense and MoE — ship a byte-identical tokenizer,
so a near-miss inside that family costs nothing. Across families it would be silently wrong, which is what
the matching below is careful about.

Two constructions are assembled here, and they differ in every part: the byte-level BPE that most current
families use (`tokenizer.ggml.model = 'gpt2'`), and the SentencePiece-derived one Gemma carries (`'gemma4'`),
where a space is a word mark rather than a byte and anything unknown falls back to single-byte pieces. Both
were checked against a backend serving that family and agreed exactly — 486 tokens against 486 for Qwen,
510 against 510 for Gemma, on the probes below.
"""

__all__ = ["find_for_model", "load"]

import logging
import os
import pathlib
import re
import uuid
from typing import Any, Callable, Collection, Optional

logger = logging.getLogger(__name__)

# Quantization and file-format markers, dropped from a name before matching so that the same model at two
# bit depths reads as the same model. Whole tokens only, so a name that merely contains these letters is
# untouched. The backend names the quantization too (LM Studio reports `qwen3.8-27b@q4_k_xl`), and it need
# not be the one on disk.
_QUANTIZATION_NOISE = re.compile(r"(?<![a-z0-9])(?:i?q\d+(?:_[a-z0-9]+)*|ud|bf16|f16|f32|gguf|mtp)(?![a-z0-9])",
                                 re.IGNORECASE)

# GGUF names the tokenizer's *class* in `tokenizer.ggml.model` and, for the byte-level ones, its
# pre-tokenizer variant in `tokenizer.ggml.pre`. The class says how the pieces fit together; the pre says
# where the text is cut before they do, and getting either wrong shifts counts silently, since the result
# still tokenizes. Keyed on those two rather than on the model architecture: architectures come and go while
# a family keeps its tokenizer, and `transformers`' own GGUF reader gates on architecture and so refuses a
# model it could otherwise handle ("architecture qwen35 is not supported yet", measured).
_GPT2_FAMILY_REGEX = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
                      r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
_PRE_TOKENIZER_REGEXES = {"qwen35": _GPT2_FAMILY_REGEX}

# The word mark a SentencePiece-derived vocabulary writes instead of a space: U+2581 LOWER ONE EIGHTH BLOCK.
# It is a visible character, so it needs no escape on those grounds — but at most sizes it is indistinguishable
# from an underscore, and it is load-bearing in both the normalizer and the decoder below. Written as an escape
# so that what it is can be read off the page rather than guessed at from its shape.
_WORD_MARK = "\u2581"

# `(tokenizer class, pre-tokenizer)` pairs whose construction has been checked against a backend serving that
# model, and may therefore be trusted when no backend is available to confirm. Anything else is built anyway
# and has to earn its place live; see `load`. Adding an entry means measuring one — point
# `investigations/prompt-size-cache-relative/measure_true_size.py` at a backend serving that model.
_VERIFIED_CONSTRUCTIONS = {("gpt2", "qwen35"),      # Qwen 3.5 / 3.6 / 3.8, measured 2026-08-24: 486 tokens against 486
                           ("gemma4", None)}       # Gemma 4, measured 2026-08-24: 510 against 510

# A vision projector rides beside its model under a name that matches the model's just as well, and carries
# no tokenizer. Anywhere in the name, not just at the front: both `mmproj-gemma-4-26B-A4B-it-BF16.gguf` and
# `Qwen3.5-9B-mmproj-BF16.gguf` are in use, and a projector is also the smaller file, so it wins the
# size tie-break and disables the feature it was mistaken for.
_NOT_A_MODEL = re.compile(r"mmproj", re.IGNORECASE)

# Round-trip probe. Digits, punctuation runs, non-ASCII letters and newlines are where a mis-assembled
# byte-level BPE stops being reversible, so they are all in here.
_SELF_CHECK_SAMPLE = "Hello, world! 3.14159 — ei se mitään.\n\tKuinka monta? 42 tokens…  ✓"

# Text for the backend comparison. Ordinary prose with the things vocabularies disagree about mixed in —
# digits, contractions, hyphenation, punctuation runs, a non-English sentence — since two tokenizers differ
# least on plain lowercase words. Repeated to make the longer probe, which keeps both probes one kind of
# text: a difference in *content* between them would be a difference in tokenization the comparison would
# then have to allow for.
_PROBE_TEXT = ("The 2026-08-24 run cost $1,234.56 and took 7.5 hours; it wasn't re-runnable. "
               "Kokeillaanpa myös ääkkösiä — ja pitkää ajatusviivaa. Values: 3.14159, 1e-9, 0xFF.\n")
_PROBE_REPEATS = 6                # ~900 characters of added text: enough that a wrong vocabulary misses by tens of tokens
_PROBE_TOLERANCE_TOKENS = 2       # the one boundary where the added text meets what precedes it

def _build_byte_level_bpe(vocabulary, merges, pre: Optional[str]) -> Any:
    """Assemble the byte-level BPE that `tokenizer.ggml.model = 'gpt2'` names. Used by the Qwen families."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, Regex  # noqa: PLC0415

    # An unrecognized pre-tokenizer gets the family's regex as a guess, which the backend is then asked to
    # confirm. Guessing is only safe because something checks; see the policy in `load`.
    regex = _PRE_TOKENIZER_REGEXES.get(pre, _GPT2_FAMILY_REGEX)
    tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=merges, fuse_unk=False, byte_fallback=False))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(regex), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def _build_word_mark_bpe(vocabulary, merges, pre: Optional[str]) -> Any:
    """Assemble the SentencePiece-derived BPE that Gemma's GGUFs carry (`tokenizer.ggml.model = 'gemma4'`).

    Three things differ from the byte-level construction, and all three change the count if got wrong: a
    space is written as a word mark rather than encoded as a byte, there is no pre-tokenizer splitting the
    text first, and anything outside the vocabulary falls back to the 256 single-byte tokens the vocabulary
    carries for the purpose (`<0x00>`…`<0xFF>`) rather than being unrepresentable.
    """
    from tokenizers import Tokenizer, models, normalizers, decoders  # noqa: PLC0415

    tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=merges,
                                     unk_token="<unk>", fuse_unk=True, byte_fallback=True))
    tokenizer.normalizer = normalizers.Replace(" ", _WORD_MARK)
    tokenizer.decoder = decoders.Sequence([decoders.Replace(_WORD_MARK, " "),
                                           decoders.ByteFallback(),
                                           decoders.Fuse()])
    return tokenizer


# What to do with each tokenizer class GGUF names in `tokenizer.ggml.model`. A class absent from here is one
# this module has no assembly for, which is a decline rather than a guess — the pieces of an unknown
# construction fit together in more than one way, and only some of them count correctly.
_TOKENIZER_BUILDERS = {"gpt2": _build_byte_level_bpe,
                       "gemma4": _build_word_mark_bpe}


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


def _agrees_with_backend(tokenizer: Any,
                         backend_counter: Optional[Callable[[str], Optional[int]]]) -> Optional[bool]:
    """Does `tokenizer` count the way the backend does? `True`, `False`, or `None` if it could not be asked."""
    if backend_counter is None:
        return None

    # Two probes, and what is compared is the *difference* between them. A backend counts a whole templated
    # prompt, so its figure includes framing this tokenizer never sees — a fixed overhead that cancels when
    # the same template wraps both probes. Comparing absolute counts would need that overhead to be known,
    # and would fail for exactly the reason this module exists.
    #
    # Short probes on purpose: they stay in the regime where the backend's reported size is dependable, which
    # a long conversation is measurably not (`investigations/prompt-size-cache-relative/`). Each probe is
    # unique per call, so nothing that happens to be cached can answer for it.
    tag = uuid.uuid4()
    short_probe = f"[{tag}]\n{_PROBE_TEXT}"
    long_probe = f"{short_probe}\n{_PROBE_TEXT * _PROBE_REPEATS}"

    try:
        backend_short = backend_counter(short_probe)
        backend_long = backend_counter(long_probe)
    except Exception as exc:  # noqa: BLE001 -- an unreachable backend is a "cannot say", not a failure here
        logger.info(f"_agrees_with_backend: could not ask the backend to count: {type(exc)}: {exc}.")
        return None
    if backend_short is None or backend_long is None:
        return None

    backend_difference = backend_long - backend_short
    local_difference = (len(tokenizer.encode(long_probe, add_special_tokens=False).ids)
                        - len(tokenizer.encode(short_probe, add_special_tokens=False).ids))
    # The added text is identical in both, so a correct tokenizer matches exactly; the slack is for the one
    # boundary where the addition meets what precedes it. A tokenizer for the wrong model misses by far more
    # than that — a different vocabulary changes the count of a paragraph by percent, not by tokens.
    #
    # What this cannot see is a *constant* error, since a fixed offset cancels along with the framing: a
    # tokenizer disagreeing about the leading-space convention would be off by about a token per message and
    # pass. That is the right thing to miss — it is immaterial against a context window, where the errors
    # worth catching are the ones that grow with the text.
    disagreement = abs(backend_difference - local_difference)
    agrees = disagreement <= _PROBE_TOLERANCE_TOKENS
    logger.info(f"_agrees_with_backend: backend counted {backend_difference} tokens of added text where this "
                f"tokenizer counts {local_difference} ({'agreed' if agrees else 'DISAGREED'}, "
                f"tolerance {_PROBE_TOLERANCE_TOKENS}).")
    return agrees


def load(gguf_path: pathlib.Path, backend_counter: Optional[Callable[[str], Optional[int]]] = None) -> Optional[Any]:
    """Build the tokenizer stored in `gguf_path`. Returns a `tokenizers.Tokenizer`, or `None`.

    `None` means the file could not be read, its tokenizer is of a class this module cannot assemble, the
    result failed its own round-trip check, or it disagreed with the backend about how many tokens a piece of
    text is. Every one of those is logged, and every one leaves the caller to fall back to estimating.

    `backend_counter`: how to ask the backend to count a piece of text — `text -> token count`, or `None`
                       when it cannot answer. Given one, the tokenizer is checked against the model that is
                       actually being served, and any pre-tokenizer is then allowed. Without one, only
                       constructions measured in advance (`_PRE_TOKENIZER_REGEXES`) are trusted, since
                       nothing else could catch a wrong guess.

    Reading is slow enough to matter — measured at ~7 s, nearly all of it in the GGUF reader indexing the
    file's tensor metadata on the way past — so call this off any thread that must stay responsive.
    """
    try:
        # Deferred because only this path needs them, not because they are dear: measured together at ~46 ms
        # against `llmclient`'s own 1282 ms to import. What the deferral does buy is that a broken install
        # degrades to estimating here, alongside every other reason this function declines, instead of making
        # `llmclient` unimportable.
        import gguf  # noqa: PLC0415
        import tokenizers  # noqa: PLC0415, F401 -- the builders import what they need; named here so a missing install reports itself as that, rather than as a failure to assemble
    except ImportError as exc:
        logger.warning(f"load: cannot read '{gguf_path}': {type(exc)}: {exc}. Falling back to token estimates.")
        return None

    try:
        reader = gguf.GGUFReader(str(gguf_path))

        def field(key: str) -> Any:
            return reader.fields[key].contents()

        tokenizer_class = field("tokenizer.ggml.model")
        build = _TOKENIZER_BUILDERS.get(tokenizer_class)
        if build is None:
            logger.info(f"load: '{gguf_path.name}' has tokenizer class {tokenizer_class!r}, which this module cannot "
                        f"assemble; keeping the token estimate. Buildable: {sorted(_TOKENIZER_BUILDERS)}.")
            return None

        pre = field("tokenizer.ggml.pre") if "tokenizer.ggml.pre" in reader.fields else None
        vocabulary = {token: index for index, token in enumerate(field("tokenizer.ggml.tokens"))}
        merges = [tuple(merge.split(" ", 1)) for merge in field("tokenizer.ggml.merges")]
        tokenizer = build(vocabulary, merges, pre)
    except Exception as exc:  # noqa: BLE001 -- any failure here just means "no local tokenizer"
        logger.warning(f"load: could not build a tokenizer from '{gguf_path}': {type(exc)}: {exc}. Falling back to token estimates.")
        return None

    # Ask the thing we just assembled whether it is reversible before anyone counts with it. A tokenizer
    # built from mismatched parts still returns a number, and that number would be shown without the `~`
    # that marks an estimate — so an unchecked build trades a known approximation for an unknown error.
    # In a try of its own because the assembly above is lazy: `models.BPE` accepts a vocabulary missing the
    # pieces it was told to rely on, and only says so when something is encoded. A build that fails the first
    # time it is used has to decline like any other, not raise into the caller.
    try:
        round_tripped = tokenizer.decode(tokenizer.encode(_SELF_CHECK_SAMPLE, add_special_tokens=False).ids)
    except Exception as exc:  # noqa: BLE001 -- an unusable tokenizer is a decline, whenever it announces itself
        logger.warning(f"load: the tokenizer built from '{gguf_path}' could not encode its own check sample: "
                       f"{type(exc)}: {exc}. Falling back to token estimates.")
        return None
    if round_tripped != _SELF_CHECK_SAMPLE:
        logger.warning(f"load: the tokenizer built from '{gguf_path}' failed its round-trip check "
                       f"({round_tripped!r} != {_SELF_CHECK_SAMPLE!r}); falling back to token estimates.")
        return None

    # Then ask the backend whether this tokenizer counts the way it does. That check subsumes the offline
    # list — it is about the model being served rather than about what someone measured once — and it also
    # catches the thing no amount of care in `find_for_model` can: a file whose *name* matched while its
    # vocabulary belongs to another model. Such a tokenizer builds and round-trips perfectly.
    agrees = _agrees_with_backend(tokenizer, backend_counter)
    if agrees is False:
        logger.warning(f"load: the tokenizer built from '{gguf_path.name}' does not count the way the backend does; "
                       f"falling back to token estimates. Is this the model the backend is serving?")
        return None
    if agrees is None and (tokenizer_class, pre) not in _VERIFIED_CONSTRUCTIONS:
        logger.info(f"load: '{gguf_path.name}' is tokenizer class {tokenizer_class!r} with pre-tokenizer {pre!r}, a "
                    f"combination that has not been measured, and the backend could not be asked to confirm it; "
                    f"keeping the token estimate. Measured offline: {sorted(_VERIFIED_CONSTRUCTIONS)}.")
        return None

    confirmation = "confirmed against the backend" if agrees else "matching a pre-tokenizer measured offline"
    logger.info(f"load: tokenizer ready from '{gguf_path.name}' ({confirmation}): {len(vocabulary)} tokens, "
                f"class {tokenizer_class!r}, pre-tokenizer {pre!r}.")
    return tokenizer
