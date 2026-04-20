"""Unit tests for raven.common.audio.speech.lipsync."""

import pytest

from raven.common.audio.speech import lipsync
from raven.common.audio.speech.tts import WordTiming


# --------------------------------------------------------------------------------
# build_phoneme_stream


class TestBuildPhonemeStream:
    def test_single_word(self):
        timings = [WordTiming(word="abc", phonemes="abc", start_time=0.0, end_time=3.0)]
        stream = lipsync.build_phoneme_stream(timings)
        assert len(stream) == 3
        assert stream[0].phoneme == "a"
        assert stream[0].start_time == 0.0
        assert stream[0].end_time == 1.0
        assert stream[1].phoneme == "b"
        assert stream[1].start_time == 1.0
        assert stream[1].end_time == 2.0
        assert stream[2].phoneme == "c"
        assert stream[2].start_time == 2.0
        assert stream[2].end_time == 3.0

    def test_morph_defaults_to_none(self):
        # build_phoneme_stream is pure time slicing — morph stays None until
        # label_phoneme_stream is called.
        timings = [WordTiming(word="ab", phonemes="ab", start_time=0.0, end_time=1.0)]
        stream = lipsync.build_phoneme_stream(timings)
        assert all(e.morph is None for e in stream)

    def test_linear_interpolation_within_word(self):
        # 5 phonemes, 1.0 second → 0.2 s each
        timings = [WordTiming(word="abcab", phonemes="abcab", start_time=0.0, end_time=1.0)]
        stream = lipsync.build_phoneme_stream(timings)
        assert len(stream) == 5
        for i, event in enumerate(stream):
            assert abs(event.start_time - i * 0.2) < 1e-9
            assert abs(event.end_time - (i + 1) * 0.2) < 1e-9

    def test_multiple_words_with_gap_between(self):
        timings = [
            WordTiming(word="ab", phonemes="ab", start_time=0.0, end_time=1.0),
            WordTiming(word="ba", phonemes="ba", start_time=1.5, end_time=2.5),
        ]
        stream = lipsync.build_phoneme_stream(timings)
        assert len(stream) == 4
        assert [e.phoneme for e in stream] == ["a", "b", "b", "a"]
        assert stream[1].end_time == 1.0
        assert stream[2].start_time == 1.5  # gap preserved

    def test_empty_phonemes_skipped_with_warning(self, caplog):
        timings = [
            WordTiming(word="silent", phonemes="", start_time=0.0, end_time=1.0),
            WordTiming(word="ab", phonemes="ab", start_time=1.0, end_time=2.0),
        ]
        stream = lipsync.build_phoneme_stream(timings)
        assert [e.phoneme for e in stream] == ["a", "b"]
        assert any("empty phonemes" in rec.message for rec in caplog.records)

    def test_missing_timestamps_skipped_with_warning(self, caplog):
        timings = [
            WordTiming(word="untimed", phonemes="ab", start_time=None, end_time=None),
            WordTiming(word="timed", phonemes="ab", start_time=1.0, end_time=2.0),
        ]
        stream = lipsync.build_phoneme_stream(timings)
        assert len(stream) == 2
        assert any("missing timestamps" in rec.message for rec in caplog.records)

    def test_empty_input(self):
        assert lipsync.build_phoneme_stream([]) == []

    def test_ipa_characters_preserved(self):
        timings = [WordTiming(word="sh", phonemes="ʃa", start_time=0.0, end_time=1.0)]
        stream = lipsync.build_phoneme_stream(timings)
        assert [e.phoneme for e in stream] == ["ʃ", "a"]


# --------------------------------------------------------------------------------
# label_phoneme_stream


VOCAB = {"a": "A-morph", "b": "B-morph", "ʃ": "SH-morph"}


class TestLabelPhonemeStream:
    def _raw_stream(self):
        return [
            lipsync.PhonemeEvent(phoneme="a", start_time=0.0, end_time=1.0),
            lipsync.PhonemeEvent(phoneme="x", start_time=1.0, end_time=2.0),  # unknown
            lipsync.PhonemeEvent(phoneme="b", start_time=2.0, end_time=3.0),
        ]

    def test_known_phonemes_get_morph(self):
        labeled = lipsync.label_phoneme_stream(self._raw_stream(), VOCAB)
        morphs = {e.phoneme: e.morph for e in labeled}
        assert morphs["a"] == "A-morph"
        assert morphs["b"] == "B-morph"

    def test_drop_unknown_true_by_default(self):
        labeled = lipsync.label_phoneme_stream(self._raw_stream(), VOCAB)
        assert [e.phoneme for e in labeled] == ["a", "b"]
        # The time gap at t=[1.0, 2.0] is now a gap in the labeled stream.
        assert labeled[0].end_time == 1.0
        assert labeled[1].start_time == 2.0

    def test_drop_unknown_false_keeps_all_with_morph_none(self):
        labeled = lipsync.label_phoneme_stream(self._raw_stream(), VOCAB, drop_unknown=False)
        assert [e.phoneme for e in labeled] == ["a", "x", "b"]
        assert labeled[1].morph is None  # the unknown one
        assert labeled[0].morph == "A-morph"
        assert labeled[2].morph == "B-morph"

    def test_input_not_mutated(self):
        # `replace` returns new instances; originals should be untouched.
        raw = self._raw_stream()
        lipsync.label_phoneme_stream(raw, VOCAB)
        assert all(e.morph is None for e in raw)

    def test_empty_input(self):
        assert lipsync.label_phoneme_stream([], VOCAB) == []

    def test_ipa_character_labeled(self):
        raw = [lipsync.PhonemeEvent(phoneme="ʃ", start_time=0.0, end_time=1.0)]
        labeled = lipsync.label_phoneme_stream(raw, VOCAB)
        assert labeled[0].morph == "SH-morph"


# --------------------------------------------------------------------------------
# phoneme_at


class TestPhonemeAt:
    def _stream(self):
        return [
            lipsync.PhonemeEvent(phoneme="a", morph="A", start_time=0.0, end_time=1.0),
            lipsync.PhonemeEvent(phoneme="b", morph="B", start_time=1.0, end_time=2.0),
            lipsync.PhonemeEvent(phoneme="c", morph="C", start_time=2.0, end_time=3.0),
        ]

    def test_lookup_in_first_event(self):
        assert lipsync.phoneme_at(self._stream(), 0.5).phoneme == "a"

    def test_lookup_at_boundary_returns_later_event(self):
        # bisect_right convention: at t=1.0, the "b" event starts.
        assert lipsync.phoneme_at(self._stream(), 1.0).phoneme == "b"

    def test_before_stream_returns_none(self):
        assert lipsync.phoneme_at(self._stream(), -0.5) is None

    def test_after_stream_returns_none(self):
        assert lipsync.phoneme_at(self._stream(), 3.5) is None

    def test_empty_stream_returns_none(self):
        assert lipsync.phoneme_at([], 1.0) is None

    def test_gap_between_events_returns_none(self):
        # Simulates what label_phoneme_stream(..., drop_unknown=True) leaves behind.
        stream = [
            lipsync.PhonemeEvent(phoneme="a", morph="A", start_time=0.0, end_time=0.5),
            lipsync.PhonemeEvent(phoneme="b", morph="B", start_time=1.5, end_time=2.0),
        ]
        assert lipsync.phoneme_at(stream, 0.9) is None
        assert lipsync.phoneme_at(stream, 0.25) is not None
        assert lipsync.phoneme_at(stream, 1.75) is not None


# --------------------------------------------------------------------------------
# word_at


class TestWordAt:
    def _timings(self):
        return [
            WordTiming(word="The", phonemes="ðə", start_time=0.0, end_time=0.3),
            WordTiming(word="quick", phonemes="kwˈɪk", start_time=0.3, end_time=0.7),
            WordTiming(word="fox", phonemes="fˈɑks", start_time=0.7, end_time=1.1),
        ]

    def test_lookup_within_word(self):
        assert lipsync.word_at(self._timings(), 0.5).word == "quick"

    def test_at_start_boundary_returns_later_word(self):
        # Parallels phoneme_at's bisect_right convention.
        assert lipsync.word_at(self._timings(), 0.3).word == "quick"

    def test_before_all_words(self):
        assert lipsync.word_at(self._timings(), -0.1) is None

    def test_after_all_words(self):
        assert lipsync.word_at(self._timings(), 2.0) is None

    def test_empty_list(self):
        assert lipsync.word_at([], 0.5) is None

    def test_skips_none_timestamps(self):
        timings = [
            WordTiming(word="untimed", phonemes="xy", start_time=None, end_time=None),
            WordTiming(word="timed", phonemes="ab", start_time=0.5, end_time=1.0),
        ]
        assert lipsync.word_at(timings, 0.7).word == "timed"
        assert lipsync.word_at(timings, 0.1) is None

    def test_gap_between_words(self):
        # Non-contiguous words (typical — punctuation or silence between them).
        timings = [
            WordTiming(word="first", phonemes="ab", start_time=0.0, end_time=0.5),
            WordTiming(word="second", phonemes="cd", start_time=1.5, end_time=2.0),
        ]
        assert lipsync.word_at(timings, 1.0) is None  # in the gap


# --------------------------------------------------------------------------------
# drive


class TestDrive:
    def test_fires_on_tick_until_finish(self):
        times = iter([0.0, 0.5, 1.0, 1.5, 2.5])
        tick_count = [0]
        recorded = []

        def on_tick(t):
            recorded.append(t)
            tick_count[0] += 1
            if tick_count[0] >= 5:
                return lipsync.action_finish
            return lipsync.action_continue

        lipsync.drive(on_tick, lambda: next(times), tick_seconds=0.0)
        assert recorded == [0.0, 0.5, 1.0, 1.5, 2.5]

    def test_composes_with_phoneme_at(self):
        stream = [
            lipsync.PhonemeEvent(phoneme="a", morph="A", start_time=0.0, end_time=1.0),
            lipsync.PhonemeEvent(phoneme="b", morph="B", start_time=1.0, end_time=2.0),
        ]
        times = iter([0.5, 1.5, 2.5])
        tick_count = [0]
        recorded = []

        def on_tick(t):
            event = lipsync.phoneme_at(stream, t)
            recorded.append(event.phoneme if event is not None else None)
            tick_count[0] += 1
            return lipsync.action_finish if tick_count[0] >= 3 else lipsync.action_continue

        lipsync.drive(on_tick, lambda: next(times), tick_seconds=0.0)
        assert recorded == ["a", "b", None]

    def test_composes_with_word_at(self):
        timings = [
            WordTiming(word="hello", phonemes="hh", start_time=0.0, end_time=1.0),
            WordTiming(word="world", phonemes="ww", start_time=1.0, end_time=2.0),
        ]
        times = iter([0.5, 1.5, 2.5])
        tick_count = [0]
        recorded = []

        def on_tick(t):
            wt = lipsync.word_at(timings, t)
            recorded.append(wt.word if wt is not None else None)
            tick_count[0] += 1
            return lipsync.action_finish if tick_count[0] >= 3 else lipsync.action_continue

        lipsync.drive(on_tick, lambda: next(times), tick_seconds=0.0)
        assert recorded == ["hello", "world", None]

    def test_immediate_finish_fires_once(self):
        recorded = []

        def on_tick(t):
            recorded.append(t)
            return lipsync.action_finish

        lipsync.drive(on_tick, lambda: 0.0, tick_seconds=0.0)
        assert recorded == [0.0]  # fired once, then stopped

    def test_invalid_return_raises(self):
        def on_tick(t):
            return "keep going"  # not a sym

        with pytest.raises(ValueError, match="action_continue"):
            lipsync.drive(on_tick, lambda: 0.0, tick_seconds=0.0)

    def test_none_return_raises(self):
        # Forgotten `return` statement — callback implicitly returns None.
        # The strict check catches this as a bug rather than silently looping.
        def on_tick(t):
            pass  # oops, forgot to return

        with pytest.raises(ValueError, match="action_continue"):
            lipsync.drive(on_tick, lambda: 0.0, tick_seconds=0.0)
