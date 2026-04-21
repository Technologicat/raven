"""Synchronous playback primitives for encoded speech audio.

Two entry points:

- `play_encoded`: straightforward playback of encoded audio bytes through a
  `raven.common.audio.player.Player`. Fires `on_audio_ready` / `on_start` /
  `on_stop` callbacks at the appropriate transitions.

- `play_encoded_with_lipsync`: same, plus a tick-driven loop (see
  `raven.common.audio.speech.lipsync.drive`) that fires a caller-supplied
  `on_tick` callback synced to playback time. Composes with the lookup
  helpers (`phoneme_at` / `word_at`) for avatar lipsync, word-level
  subtitles, per-phoneme captions, or any other per-tick consumer.

Both are synchronous. Callers that want fire-and-forget semantics submit them
to their own task manager (`raven.common.bgtask.TaskManager`, a thread pool,
etc.). Pure with respect to client singletons: the `Player` instance is
passed in explicitly, so these functions are usable from any layer that
holds a player (the client app, tests with a fake player, …).

The avatar-driving closure that a typical lipsync consumer builds — mapping
phoneme events to mouth-morph overrides and calling the server — lives in
the caller. Keeping it out of this module preserves the common-layer rule
that `raven.common.*` does not depend on `raven.client.api`.
"""

__all__ = ["play_encoded", "play_encoded_with_lipsync"]

import io
import logging
import time
import traceback
from typing import Callable, Optional

from unpythonic import sym

from . import lipsync

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------

def _fire(callback: Optional[Callable], label: str) -> None:
    """Call `callback()` if not `None`, logging + swallowing any exception.

    Keeps a misbehaving caller callback from tearing down the playback task
    mid-flight. The label identifies the callsite for diagnostics.
    """
    if callback is None:
        return
    try:
        callback()
    except Exception as exc:
        logger.error(f"{label}: {type(exc)}: {exc}")
        traceback.print_exc()


def play_encoded(audio_bytes: bytes,
                 player,
                 on_audio_ready: Optional[Callable] = None,
                 on_start: Optional[Callable] = None,
                 on_stop: Optional[Callable] = None) -> None:
    """Synchronously play encoded audio through `player`, firing callbacks.

    `audio_bytes`: encoded audio in any format `player` can load (FLAC, MP3, …).

    `player`: a `raven.common.audio.player.Player`-compatible object. Must
              expose `is_playing()`, `stop()`, `load(stream)`, `start()`.

    `on_audio_ready(audio_bytes)`: called right before playback, with the
                                   encoded bytes. Useful for saving the audio,
                                   wire-level logging, etc. Receives the
                                   `audio_bytes` argument.

    `on_start()`: called after `player.start()`. No arguments.

    `on_stop()`: called when playback ends. No arguments. Omitting this is
                 the fire-and-forget hint: the function returns immediately
                 after starting playback instead of waiting for it to finish.
                 (Fire-and-forget is still bounded — the player's own audio
                 thread carries on.)

                 When provided, `on_stop` fires via `finally` — guaranteed to
                 run exactly once, whether playback finishes normally or the
                 wait is cut short by an exception. Callers can rely on it
                 for "playback ended" state resets.

    Callback exceptions are logged and swallowed so a buggy callback doesn't
    tear down the playback task itself.
    """
    if on_audio_ready is not None:
        try:
            on_audio_ready(audio_bytes)
        except Exception as exc:
            logger.error(f"play_encoded: in on_audio_ready: {type(exc)}: {exc}")
            traceback.print_exc()

    logger.info("play_encoded: loading audio into mixer")
    if player.is_playing():
        player.stop()
    audio_buffer = io.BytesIO(audio_bytes)
    try:
        player.load(audio_buffer)
    except RuntimeError as exc:
        logger.error(f"play_encoded: failed to load audio into mixer, reason {type(exc)}: {exc}")
        return

    # Fire-and-forget shortcut: with `on_stop=None`, don't wait for playback to finish.
    if on_stop is None:
        logger.info("play_encoded: starting playback")
        _fire(on_start, "play_encoded: in on_start")
        player.start()
        return

    # Wait-for-end mode: `on_stop` fires in `finally` so callers can rely on
    # their "playback ended" handler running even if playback aborts unexpectedly.
    logger.info("play_encoded: starting playback")
    _fire(on_start, "play_encoded: in on_start")
    try:
        player.start()
        while player.is_playing():
            time.sleep(0.01)
    finally:
        logger.info("play_encoded: playback finished")
        _fire(on_stop, "play_encoded: in on_stop")


def play_encoded_with_lipsync(audio_bytes: bytes,
                              player,
                              on_tick: Callable[[float], sym],
                              clock: Optional[Callable[[], float]] = None,
                              tick_seconds: float = 0.01,
                              on_audio_ready: Optional[Callable] = None,
                              on_start: Optional[Callable] = None,
                              on_stop: Optional[Callable] = None) -> None:
    """Synchronously play encoded audio + run a tick-driven loop.

    Composes `play_encoded`'s playback machinery with `lipsync.drive`. The
    caller supplies `on_tick`, which is fired at `tick_seconds` intervals
    with the current playback time; typical consumers use `lipsync.phoneme_at`
    / `lipsync.word_at` inside it to look up per-tick data (avatar morph,
    subtitle text, …) and apply side effects.

    `audio_bytes`, `player`, `on_audio_ready`, `on_start`, `on_stop`:
    as in `play_encoded`. The `on_stop`-implied fire-and-forget hint does
    not apply here — playback time is what drives the tick loop, so the
    function necessarily waits for playback to finish.

    `on_tick(t)`: see `lipsync.drive` for the contract. Must return
                  `lipsync.action_continue` or `lipsync.action_finish`.
                  Returning `action_finish` breaks the loop early; by
                  default the loop exits when playback ends (i.e. when
                  `on_tick` notices `player.is_playing()` is `False`).

    `clock`: 0-arg callable returning the current playback time in seconds.
             Defaults to `player.get_position`. Callers wanting to offset
             the video relative to audio (lipsync `video_offset`) pass
             `lambda: player.get_position() + offset`.

    `tick_seconds`: as in `lipsync.drive`. ~10 ms is the sweet spot for
                    lipsync — fast enough for smooth mouth motion, slow
                    enough to keep CPU negligible.
    """
    if on_audio_ready is not None:
        try:
            on_audio_ready(audio_bytes)
        except Exception as exc:
            logger.error(f"play_encoded_with_lipsync: in on_audio_ready: {type(exc)}: {exc}")
            traceback.print_exc()

    logger.info("play_encoded_with_lipsync: loading audio into mixer")
    if player.is_playing():
        player.stop()
    audio_buffer = io.BytesIO(audio_bytes)
    try:
        player.load(audio_buffer)
    except RuntimeError as exc:
        logger.error(f"play_encoded_with_lipsync: failed to load audio into mixer, reason {type(exc)}: {exc}")
        return

    if clock is None:
        clock = player.get_position

    # `on_stop` fires in `finally` so callers can rely on their "playback
    # ended" handler running even if the drive loop aborts unexpectedly.
    # The drive loop exits when `on_tick` returns `action_finish`; tying that
    # return to `player.is_playing()` (as the avatar-lipsync callback does)
    # keeps the end-of-drive and end-of-audio events coincident.
    logger.info("play_encoded_with_lipsync: starting playback")
    _fire(on_start, "play_encoded_with_lipsync: in on_start")
    try:
        player.start()
        lipsync.drive(on_tick, clock, tick_seconds=tick_seconds)
    finally:
        logger.info("play_encoded_with_lipsync: playback finished")
        if on_stop is not None:
            _fire(on_stop, "play_encoded_with_lipsync: in on_stop")
