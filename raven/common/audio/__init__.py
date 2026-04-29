"""Audio subsystem: playback (`player`), capture (`recorder`), codecs, resampling.

The submodules are independently importable — nothing here auto-loads them.

For app startup convenience, this package offers an aggregator `initialize(...)`
that inits the playback and/or capture singletons in one call. Apps that need
only one side (or neither) should pass `False` for the unneeded half, or call
the submodule-level `initialize` directly.
"""

__all__ = ["initialize"]

import logging
logger = logging.getLogger(__name__)

from typing import Union

def initialize(*,
               player: Union[bool, dict] = True,
               recorder: Union[bool, dict] = True) -> None:
    """Initialize the audio playback and/or capture singletons.

    `player`:
        - `True` (default): initialize the playback singleton with default settings.
        - `dict`: initialize with these keyword arguments, forwarded to
          `raven.common.audio.player.initialize`.
        - `False` / falsy: skip playback initialization entirely.

    `recorder`: same semantics, forwarded to `raven.common.audio.recorder.initialize`.

    After this call, readers access the singletons at `player.instance` and
    `recorder.instance` on their respective modules. Idempotent.
    """
    # Deferred imports keep pygame/pvrecorder out of processes that only
    # want lightweight audio utilities (codec, resample) or nothing at all.
    if player:
        kwargs = player if isinstance(player, dict) else {}
        from . import player as _player  # noqa: PLC0415 -- intentional deferred import
        _player.initialize(**kwargs)
    if recorder:
        kwargs = recorder if isinstance(recorder, dict) else {}
        from . import recorder as _recorder  # noqa: PLC0415 -- intentional deferred import
        _recorder.initialize(**kwargs)
