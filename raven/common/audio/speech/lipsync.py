"""Lipsync driver: word-level TTS timings → time-indexed dispatch.

Engine-agnostic: the math (phoneme interpolation, time-to-event lookup, tick loop)
lives here; the side effects (poke avatar morphs, update subtitles, log, …)
live in the caller-supplied `on_tick` callback.

Split of concerns:

- **Phoneme track** — for avatar lipsync, per-phoneme subtitles, or any other
  per-phoneme consumer.
  - `build_phoneme_stream(timings)`: word timings → `list[PhonemeEvent]`. Pure
    time slicing; every phoneme from every word is included, with `morph=None`.
  - `label_phoneme_stream(stream, vocabulary, drop_unknown=True)`: attach
    vocabulary-derived labels. Separate from time slicing so consumers that
    don't need a mapping (e.g. a researcher analyzing raw phoneme statistics)
    can skip it. The lipsync use case composes both.
  - `phoneme_at(stream, t)`: look up the event active at time `t`, or `None`
    if outside the stream / in a gap left by `drop_unknown=True`.

- **Word track** — for word-level subtitles.
  - `word_at(timings, t)`: look up the `WordTiming` active at time `t`.
    Operates directly on the `WordTiming` list; no separate dataclass (words
    already carry text, phonemes, and timings).

- **Tick loop** — shared across all tracks.
  - `drive(on_tick, clock, tick_seconds=0.01)`: tick at fixed interval, fire
    `on_tick(t)` each tick. Takes no stream, no event type — just a clock
    and a callback. Consumers compose tracks inside their `on_tick` closure,
    calling `phoneme_at` / `word_at` / both / neither as needed. The callback
    signals continuation via the return value (`action_continue` / `action_finish`).

The `morph` field of `PhonemeEvent` is what the consumer asked for in
`label_phoneme_stream`'s `vocabulary` — an avatar morph name, a subtitle
string, anything. The type stays `str`; multi-target consumers that want
several labels per phoneme build a dict in the `morph` field, or run
`label_phoneme_stream` once per label set.
"""

__all__ = ["action_continue",
           "action_finish",
           "PhonemeEvent",
           "build_phoneme_stream",
           "label_phoneme_stream",
           "phoneme_at",
           "word_at",
           "drive"]

import bisect
import logging
import time
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Optional, Sequence

from unpythonic import sym

from .tts import WordTiming

logger = logging.getLogger(__name__)


# Loop-control sentinels returned by the `on_tick` callback. Interned `sym`s:
# compare with `is`, and the string form is informative in tracebacks / logs.
action_continue = sym("continue")
action_finish = sym("finish")


@dataclass
class PhonemeEvent:
    """One phoneme's slice of the audio timeline.

    `phoneme`: raw IPA (or engine-specific) phoneme character, e.g. `"ʃ"`.
    `start_time`, `end_time`: seconds, absolute (from start of whole audio).
    `morph`: consumer-supplied label (avatar morph name, subtitle text, …),
             or `None` if the stream hasn't been labeled. Populated by
             `label_phoneme_stream`.
    """
    phoneme: str
    start_time: float
    end_time: float
    morph: Optional[str] = None


def build_phoneme_stream(timings: Sequence[WordTiming]) -> list[PhonemeEvent]:
    """Word-level timings → per-phoneme stream with interpolated start/end.

    Each word's timespan is split linearly across its phonemes. Every phoneme
    is included; the returned events carry `morph=None`. Use `label_phoneme_stream`
    to attach consumer-specific labels and optionally drop unwanted phonemes.

    Words with empty phonemes or missing start/end timestamps are skipped with
    a warning; this can happen if Kokoro fails to time-align a word.
    """
    stream: list[PhonemeEvent] = []
    for timing in timings:
        if not timing.phonemes:
            logger.warning(f"build_phoneme_stream: TTS returned empty phonemes for word '{timing.word}'. Skipping this word.")
            continue
        if timing.start_time is None or timing.end_time is None:
            logger.warning(f"build_phoneme_stream: TTS missing timestamps for word '{timing.word}', cannot compute phoneme timings. Skipping this word.")
            continue
        n_phonemes = len(timing.phonemes)
        dt = timing.end_time - timing.start_time
        for idx, phoneme in enumerate(timing.phonemes):  # e.g. mˈaɪnd → m, ˈ, a, ɪ, n, d
            t_start = timing.start_time + dt * idx / n_phonemes
            t_end = timing.start_time + dt * (idx + 1) / n_phonemes
            stream.append(PhonemeEvent(phoneme=phoneme,
                                       start_time=t_start,
                                       end_time=t_end))
    return stream


def label_phoneme_stream(stream: Sequence[PhonemeEvent],
                         vocabulary: Mapping[str, str],
                         drop_unknown: bool = True) -> list[PhonemeEvent]:
    """Attach vocabulary-derived labels to a phoneme stream.

    Returns a new list; input events are not mutated. `morph` in each output
    event is set from `vocabulary[phoneme]` for phonemes present in the map.

    `drop_unknown=True` (default): phonemes not in `vocabulary` are dropped
    entirely, leaving time gaps in the stream. This is the lipsync use case
    — the avatar driver only understands the mouth morphs in its vocabulary,
    and a gap means "don't move the mouth right now" (which is the correct
    behavior during stress marks, inter-word pauses, etc. that aren't in the
    avatar vocabulary but are in Kokoro's phoneme output).

    `drop_unknown=False`: phonemes not in `vocabulary` are kept with
    `morph=None`. Use for consumers that want to see every phoneme (e.g. a
    diagnostic overlay distinguishing "labeled" vs. "unlabeled" spans).
    """
    result: list[PhonemeEvent] = []
    for event in stream:
        if event.phoneme in vocabulary:
            result.append(replace(event, morph=vocabulary[event.phoneme]))
        elif not drop_unknown:
            result.append(replace(event, morph=None))
    return result


def phoneme_at(stream: Sequence[PhonemeEvent], t: float) -> Optional[PhonemeEvent]:
    """Return the phoneme event active at time `t`, or `None` if outside the stream.

    Outside means: `t` is before the first event's `start_time`, or after the
    last event's `end_time`, or the stream is empty. Gaps between events (from
    `drop_unknown=True` labeling) also return `None`.

    O(log n) via binary search on `start_time`.
    """
    if not stream:
        return None
    if t < stream[0].start_time or t > stream[-1].end_time:
        return None
    idx = bisect.bisect_right(stream, t, key=lambda e: e.start_time) - 1
    event = stream[idx]
    # Guard against gaps: t may land between two events if phonemes were dropped
    # by `drop_unknown=True`. `bisect_right` returns the event whose start is
    # latest-but-≤t, so we only need to check that t hasn't run past its end.
    if t > event.end_time:
        return None
    return event


def word_at(timings: Sequence[WordTiming], t: float) -> Optional[WordTiming]:
    """Return the word active at time `t`, or `None` if outside any word.

    Skips entries with `None` start/end timestamps (Kokoro occasionally emits
    these — see `WordTiming` docstring). The filter is rebuilt each call;
    word lists are short (tens per sentence), so call-site consistency with
    `phoneme_at` (binary search) wins over micro-optimizing.

    Consumers: subtitle/captioning renderers. Unlike `phoneme_at`, no separate
    event type — `WordTiming` already carries `word`, `phonemes`, and the
    timings.
    """
    timed = [w for w in timings if w.start_time is not None and w.end_time is not None]
    if not timed or t < timed[0].start_time or t > timed[-1].end_time:
        return None
    idx = bisect.bisect_right(timed, t, key=lambda w: w.start_time) - 1
    wt = timed[idx]
    if t > wt.end_time:  # in a gap between words
        return None
    return wt


def drive(on_tick: Callable[[float], sym],
          clock: Callable[[], float],
          tick_seconds: float = 0.01) -> None:
    """Pure tick loop. Fire `on_tick(t)` every `tick_seconds` until it says to stop.

    Intentionally stream- and event-agnostic: the loop doesn't know about
    phonemes or words. Consumers compose what they need inside `on_tick`:

        def on_tick(t: float) -> sym:
            if not audio_player.is_playing():
                return lipsync.action_finish
            phoneme_event = lipsync.phoneme_at(phoneme_stream, t)   # avatar
            word = lipsync.word_at(timings, t)                       # subtitle
            # apply morph, display word, either, neither — consumer's choice
            return lipsync.action_continue

        lipsync.drive(on_tick, clock)

    `clock`: returns the current media time (seconds from start of the audio).
             Typical: `util.api_config.audio_player.get_position`.

    `on_tick` must return either `action_continue` (run another tick) or
    `action_finish` (exit the loop cleanly). Any other return value raises
    `ValueError` — silent "default to continue" would mask callback bugs
    (forgotten return statements, typos) as infinite loops.

    `on_tick` is called every tick — consumers typically write idempotent
    "set state for now" bodies, not "fire once per change" bodies. That's
    the simplest model and matches what the real consumers (avatar driver,
    subtitle renderer) want. Distinguishing pre-stream from post-stream is
    the consumer's responsibility (compare `t` against the stream bounds,
    or track a "seen an event yet" flag).
    """
    while True:
        t = clock()
        action = on_tick(t)
        if action is action_finish:
            return
        if action is not action_continue:
            raise ValueError(f"drive: on_tick must return `action_continue` or `action_finish`; got {action!r}")
        time.sleep(tick_seconds)
