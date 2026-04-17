"""Resample 1-D audio signals between sample rates.

Device-agnostic thin wrapper over `torchaudio.functional.resample`:

    numpy in  → numpy out (under the hood: tensor on CPU)
    tensor in → tensor out on the same device

Needed because different speech engines run at different native sample rates
(Kokoro TTS at 24 kHz, Whisper STT at 16 kHz) and the in-process round-trip
path passes audio between them as plain numpy arrays.

Also useful more generally — the `quality` preset covers both speech-grade
(fast) and music-grade (slow) resampling, so this helper can carry its
weight beyond the speech pipeline.
"""

__all__ = ["Quality", "AudioT", "resample"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Literal, TypeVar

import numpy as np

import torch
import torchaudio.functional

Quality = Literal["default", "kaiser_fast", "kaiser_best"]

AudioT = TypeVar("AudioT", np.ndarray, torch.Tensor)

# Preset parameters for `torchaudio.functional.resample`.
#
# "default" — Hann-windowed sinc, width 6. Fast; speech-grade. torchaudio's own default.
# "kaiser_fast" / "kaiser_best" — Kaiser-windowed sinc, matching librosa's presets of the same name.
#                                 Beta values are librosa's tuned constants (see `librosa.resample`
#                                 source): β = π·√((width/2 · roll)² − 0.5²) with roll ≈ 0.85 / 0.945.
_QUALITY_PARAMS = {"default": {"resampling_method": "sinc_interp_hann",
                               "lowpass_filter_width": 6},
                   "kaiser_fast": {"resampling_method": "sinc_interp_kaiser",
                                   "lowpass_filter_width": 16,
                                   "beta": 8.555504641634386},
                   "kaiser_best": {"resampling_method": "sinc_interp_kaiser",
                                   "lowpass_filter_width": 64,
                                   "beta": 14.769656459379492}}

def resample(audio: AudioT,
             from_rate: int,
             to_rate: int,
             quality: Quality = "default") -> AudioT:
    """Resample a 1-D float audio signal from `from_rate` to `to_rate` Hz.

    `audio`: rank-1 float32 or float64, either `np.ndarray` or `torch.Tensor`.

             A tensor stays on its original device; the resample runs there.
             An ndarray is processed on CPU via a zero-copy `torch.from_numpy` view.

    `from_rate`, `to_rate`: sample rates in Hz. Both must be positive integers.
                            When equal, this function is a no-op and returns the input as-is.

    `quality`: one of:

        "default"     — sinc_interp_hann,   width 6.   Fast; speech-grade.
        "kaiser_fast" — sinc_interp_kaiser, width 16.  librosa "fast" preset.
        "kaiser_best" — sinc_interp_kaiser, width 64.  librosa "best" preset; music-grade.

               For 24 kHz → 16 kHz speech feeding Whisper, "default" is plenty — Whisper's
               mel-spectrogram front-end bins at 80 filters over 0–8 kHz, so any subsample-level
               interpolation artefacts are lost in quantization well before the STT decoder
               sees them. Use "kaiser_best" for music or other content where stopband
               attenuation matters.

    Returns resampled audio, same type and (for tensors) device as `audio`.

    The output length is approximately ``len(audio) * to_rate / from_rate``; torchaudio
    rounds using ``gcd(from_rate, to_rate)`` reduction, so exact lengths follow the
    polyphase convention.
    """
    if from_rate <= 0 or to_rate <= 0:
        raise ValueError(f"resample: sample rates must be positive; got from_rate={from_rate}, to_rate={to_rate}.")

    if quality not in _QUALITY_PARAMS:
        raise ValueError(f"resample: unknown quality preset '{quality}'; expected one of {list(_QUALITY_PARAMS)}.")

    if from_rate == to_rate:  # no-op fast path
        return audio

    is_numpy = isinstance(audio, np.ndarray)
    if is_numpy:
        # torchaudio requires a float dtype. If caller passed an integer array, let torch raise the TypeError —
        # converting silently would mask an API misuse.
        tensor = torch.from_numpy(audio)
    elif isinstance(audio, torch.Tensor):
        tensor = audio
    else:
        raise TypeError(f"resample: expected `audio` to be np.ndarray or torch.Tensor; got {type(audio)}.")

    resampled = torchaudio.functional.resample(tensor,
                                               orig_freq=from_rate,
                                               new_freq=to_rate,
                                               **_QUALITY_PARAMS[quality])

    if is_numpy:
        return resampled.numpy()
    return resampled
