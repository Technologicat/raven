#!/usr/bin/env python
"""Measure Kokoro's run-to-run nondeterminism.

Generate the same `(text, voice, speed)` multiple times and quantify the
sample-level differences. Helps distinguish between:

  (a) unseeded stochastic vocoder sampling (expected, intentional — the
      same pattern VITS-family TTS uses for natural-sounding speech),

  (b) accidental nondeterminism from threading / CUDA reduction order
      (would show up as different magnitudes on different hardware),

  (c) broken engine (would show up as phase offsets, length mismatches,
      or low cross-correlation).

Typical CPU result on whisper-base / Kokoro-82M / "af_alloy" voice:

    identical sample count, zero phase offset, xcorr ≈ 0.993,
    RMS diff ≈ 3% of peak (-30 dB), max diff ≈ 18% of peak (-15 dB),
    concentrated in voiced regions (silence / padding stays near machine epsilon).

Run from the project root with the venv activated:

    python -m raven.common.audio.speech.tests.bench_tts_nondeterminism
    python -m raven.common.audio.speech.tests.bench_tts_nondeterminism --runs 5
    python -m raven.common.audio.speech.tests.bench_tts_nondeterminism --text "Hello world." --device cuda:0

Not collected by pytest (no `test_` prefix).
"""

import argparse

import numpy as np

from scipy.signal import correlate

from raven.common.audio.speech import tts as speech_tts


DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."
DEFAULT_VOICE: str | None = None  # None → first voice alphabetically
DEFAULT_DEVICE = "cpu"
DEFAULT_RUNS = 3
DEFAULT_REPO = "hexgrad/Kokoro-82M"


def _regional_diff(a: np.ndarray, b: np.ndarray, start: int, length: int) -> tuple[float, float, float]:
    """Peak, max |diff|, RMS |diff| over `[start : start+length]`."""
    aa = a[start : start + length]
    bb = b[start : start + length]
    peak = float(np.max(np.abs(aa))) or 1e-12
    max_diff = float(np.max(np.abs(aa - bb)))
    rms_diff = float(np.sqrt(np.mean((aa - bb) ** 2)))
    return peak, max_diff, rms_diff


def _db_below(ratio: float) -> str:
    if ratio <= 0.0:
        return "  -inf dB"
    return f"{20 * np.log10(ratio):+7.2f} dB"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--text", default=DEFAULT_TEXT, help=f"Text to synthesize. Default: {DEFAULT_TEXT!r}")
    ap.add_argument("--voice", default=DEFAULT_VOICE, help="Voice name. Default: first voice alphabetically.")
    ap.add_argument("--device", default=DEFAULT_DEVICE, help=f"Torch device string. Default: {DEFAULT_DEVICE!r}")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS, help=f"Number of runs to compare pairwise. Default: {DEFAULT_RUNS}")
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"Kokoro HF repo. Default: {DEFAULT_REPO!r}")
    args = ap.parse_args()

    print(f"Loading Kokoro ({args.repo}) on {args.device}...")
    pipeline = speech_tts.load_tts_pipeline(repo_id=args.repo, device_string=args.device)

    voice = args.voice or speech_tts.get_voices(pipeline)[0]
    print(f"Voice: {voice}")
    print(f"Text:  {args.text!r}")
    print(f"Runs:  {args.runs}")
    print()

    # Generate all runs up front.
    print("Generating...")
    audios = []
    for i in range(args.runs):
        result = speech_tts.synthesize(pipeline, voice=voice, text=args.text, get_metadata=False)
        audios.append(result.audio)
        print(f"  run {i + 1}: {len(result.audio)} samples ({len(result.audio) / result.sample_rate:.3f} s)")
    sample_rate = 24000

    lengths = [len(a) for a in audios]
    print()
    print(f"Length — min: {min(lengths)}, max: {max(lengths)}, spread: {max(lengths) - min(lengths)} samples")

    # Truncate all to the shortest for comparison.
    n = min(lengths)
    audios = [a[:n] for a in audios]

    # Pairwise comparisons against run 0.
    ref = audios[0]
    peak = float(np.max(np.abs(ref)))
    print(f"Peak amplitude of reference run: {peak:.4e}  ({_db_below(peak)} FS)")
    print()

    print("Run-vs-run global diffs (against run 1):")
    print("  run          max |diff|                 RMS |diff|")
    for i, other in enumerate(audios[1:], start=2):
        max_diff = float(np.max(np.abs(ref - other)))
        rms_diff = float(np.sqrt(np.mean((ref - other) ** 2)))
        print(f"  {i:2d}    {max_diff:.4e} ({_db_below(max_diff / peak)})   {rms_diff:.4e} ({_db_below(rms_diff / peak)})")
    print()

    # Regional breakdown — first, mid, last. Shows whether diff concentrates in voiced regions.
    if args.runs >= 2:
        print("Regional diff (run 1 vs run 2, 1000-sample windows):")
        other = audios[1]
        print("  region       peak         max |diff|                 RMS |diff|")
        for label, start in [("first ", 0),
                             ("mid   ", n // 2),
                             ("last  ", max(0, n - 1000))]:
            p, md, rd = _regional_diff(ref, other, start, 1000)
            print(f"  {label}  {p:.3e}  {md:.3e} ({_db_below(md / p)})   {rd:.3e} ({_db_below(rd / p)})")
        print()

    # Cross-correlation against run 1 to rule out phase offset / length drift.
    if args.runs >= 2:
        print("Cross-correlation (first 20k samples, run 1 vs run 2):")
        n_xc = min(n, 20000)
        x = ref[:n_xc] - ref[:n_xc].mean()
        y = audios[1][:n_xc] - audios[1][:n_xc].mean()
        xcorr = correlate(x, y, mode="full")
        lag = int(np.argmax(xcorr) - (n_xc - 1))
        norm = xcorr.max() / (float(np.linalg.norm(ref[:n_xc])) * float(np.linalg.norm(audios[1][:n_xc])) + 1e-12)
        print(f"  best lag:        {lag} samples ({lag / sample_rate * 1000:+.3f} ms)")
        print(f"  normalized peak: {norm:.6f}  (1.0 = identical, 0.0 = unrelated)")
        print()

    print("Reference:")
    print(f"  16-bit quantization step: {1/32768:.3e}  (-96.3 dBFS, s16 LSB)")
    print(f"  float32 machine epsilon:  {float(np.finfo(np.float32).eps):.3e}  ({_db_below(float(np.finfo(np.float32).eps))})")


if __name__ == "__main__":
    main()
