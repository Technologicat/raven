"""Audio-related utilities."""

__all__ = ["linear_to_dBFS", "dBFS_to_linear"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Union

import numpy as np

# TODO: vectorize these to make them work on `np.array` inputs

def linear_to_dBFS(level: Union[float, int, np.int16]) -> float:
    """Convert linear audio sample value to dBFS.

    `level`: one of:

        `int`: s16 (signed 16-bit) audio sample value, range [-32768, 32767].
        `float`: floating-point audio sample value, range [-1, 1].

        Note that if `level` is exactly zero, the corresponding dBFS is -∞,
        in which case this function returns `-np.inf`.

    dbFS = decibels full scale = a logarithmic scale, where 0.0 corresponds
    to the loudest representable signal. For example:

          0.00 dB -> maximum possible audio sample value
        -42.15 dB -> at this point the high byte in s16 (signed 16-bit) audio becomes zero
        -60.00 dB -> possibly useful silence threshold when there is some background noise
        -90.31 dB -> s16 audio is completely silent (in practice, will never happen in a recording)

    Returns the dBFS value corresponding to the linear audio sample `level`.
    Note that the sign of `level` is lost (this is an abs-logarithmic scale).
    """
    if isinstance(level, (int, np.int16)):
        fs = 32767 if level > 0 else 32768
    elif isinstance(level, float):
        fs = 1.0
    else:
        raise TypeError(f"linear_to_dBFS: expected `level` to be float or s16 int; got {type(level)}.")
    # We use "20" here because `level` is the signal amplitude, which is a root-power quantity (P ~ level²).
    #     https://en.wikipedia.org/wiki/Decibel#Root-power_(field)_quantities
    dB = 20 * np.log10(abs(level) / fs)  # yes, -np.inf is fine (and the correct answer) if `level` is exactly zero
    return dB

def dBFS_to_linear(dB: float, format: type) -> Union[float, int]:
    """Convert dBFS value to linear scale.

    `dB`: the decibel level to convert, in dBFS (decibels full scale).
          See `linear_to_dBFS` for explanation.

    `format`: one of:

        `int`: for 16-bit integer output, range [0, 32767].
        `float`: for floating-point output, range [0, 1].

    Returns the linear audio sample value corresponding to `dB`.
    """
    if format is int:
        fs = 32767
    elif format is float:
        fs = 1.0
    else:
        raise TypeError(f"dBFS_to_linear: expected `format` to be exactly `int` (for s16 output) or `float` (for float output); got {format}.")
    level = fs * 10**(dB / 20)
    return level
