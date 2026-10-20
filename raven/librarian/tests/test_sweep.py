"""Tests for `raven.librarian.sweep`: the fingerprint, the ledger, and a run's resume, failure and stop rules."""

import json
import logging

import pytest

from raven.common import netutil
from raven.librarian import llmclient, sweep


def _items(n):
    return [(f"item{i}", f"text {i}") for i in range(n)]

class _Turn:
    """Stands in for an `agent.TurnRecord`, which is all the sweep does with one: `to_dict` it."""
    def __init__(self, reply):
        self.reply = reply

    def to_dict(self):
        return {"reply": self.reply, "reasoning": [f"thought about {self.reply}"]}

def _echo(llm_settings, batch, maybe_abort):
    """Answers every item with its own text upper-cased, and one turn naming the batch."""
    return {item_id: item.upper() for item_id, item in batch}, [_Turn(f"{len(batch)} answers")]


class TestFingerprint:
    def test_each_part_of_the_instrument_changes_it(self, llm_settings):
        base = sweep.fingerprint(llm_settings, "the prompt")
        assert sweep.fingerprint(llm_settings, "the prompt") == base  # deterministic
        assert sweep.fingerprint(llm_settings, "another prompt") != base
        assert sweep.fingerprint(llm_settings, "the prompt", declared="KV cache q4_0") != base

        llm_settings.model_record = llmclient._model_info(**(dict(llm_settings.model_record) | {"model_id": "other"}))
        assert sweep.fingerprint(llm_settings, "the prompt") != base

    def test_quantization_is_in_it_and_the_load_configuration_is_not(self, llm_settings):
        record = dict(llm_settings.model_record)
        llm_settings.model_record = llmclient._model_info(**(record | {"quantization": "Q4_K_XL"}))
        quantized = sweep.fingerprint(llm_settings, "p")
        llm_settings.model_record = llmclient._model_info(**(record | {"quantization": "Q8_0"}))
        assert sweep.fingerprint(llm_settings, "p") != quantized

        llm_settings.model_record = llmclient._model_info(**(record | {"load_config": {"flash_attention": True}}))
        flash_on = sweep.fingerprint(llm_settings, "p")
        llm_settings.model_record = llmclient._model_info(**(record | {"load_config": {"flash_attention": False}}))
        assert sweep.fingerprint(llm_settings, "p") == flash_on

    def test_the_header_says_what_was_not_reported(self, llm_settings):
        header = sweep.instrument_header(llm_settings)
        assert header["model_id"] == "test-model"
        assert set(header["not_reported"]) == {"quantization", "bits_per_weight", "load_config"}
        assert header["declared"] == "not stated"
        assert sweep.instrument_header(llm_settings, declared="KV cache q4_0")["declared"] == "KV cache q4_0"


class TestLedger:
    def test_a_later_line_supersedes_an_earlier_one(self, tmp_path):
        ledger = sweep.Ledger(tmp_path, "run", "abcd1234")
        ledger.write_failure("a", "timed out")
        ledger.write_answer("b", {"x": 1})
        ledger.write_answer("a", {"x": 2})
        assert ledger.answers() == {"a": {"x": 2}, "b": {"x": 1}}
        assert ledger.path.name == "run-abcd1234.jsonl"

    def test_a_failure_after_an_answer_leaves_the_item_unanswered(self, tmp_path):
        ledger = sweep.Ledger(tmp_path, "run", "abcd1234")
        ledger.write_answer("a", 1)
        ledger.write_failure("a", "asked again and failed")
        assert ledger.answers() == {}
        assert ledger.latest()["a"]["status"] == "failed"

    def test_lines_from_another_instrument_are_not_read(self, tmp_path):
        ours = sweep.Ledger(tmp_path, "run", "abcd1234")
        ours.write_answer("a", 1)
        # A file renamed or concatenated: the name no longer says what is inside, the stamp still does.
        with ours.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": "b", "v": "ffff0000", "status": "ok", "answer": 2}) + "\n")
        assert ours.answers() == {"a": 1}

    def test_reasking_keeps_the_earlier_answer_in_the_file(self, tmp_path):
        ledger = sweep.Ledger(tmp_path, "run", "abcd1234")
        ledger.write_answer("a", "first answer")
        ledger.write_answer("b", "kept")
        ledger.reask(["a"])
        assert ledger.answers() == {"b": "kept"}
        assert "first answer" in ledger.path.read_text(), "the superseded answer is still there to compare with"

    def test_headers_are_read_back_in_order(self, tmp_path):
        ledger = sweep.Ledger(tmp_path, "run", "abcd1234")
        ledger.write_header({"model_id": "m1"})
        ledger.write_answer("a", 1)
        ledger.write_header({"model_id": "m1", "declared": "KV cache q4_0"})
        assert [entry["header"] for entry in ledger.headers()] == [{"model_id": "m1"},
                                                                   {"model_id": "m1", "declared": "KV cache q4_0"}]
        assert ledger.answers() == {"a": 1}  # a header is not an item


class TestSweep:
    def test_every_item_is_answered_and_its_trace_kept(self, llm_settings, tmp_path):
        ledger = sweep.Ledger(tmp_path, "run", "abcd1234")
        result = sweep.Sweep(llm_settings, _echo, _items(5), ledger, batch_size=2).run()
        assert (result.outcome, result.done, result.failed, result.total) == (sweep.completed, 5, 0, 5)
        assert ledger.answers() == {f"item{i}": f"TEXT {i}" for i in range(5)}
        traces = [json.loads(line) for line in ledger.traces_path.read_text().splitlines()]
        assert [trace["ids"] for trace in traces] == [["item0", "item1"], ["item2", "item3"], ["item4"]]
        assert traces[2]["turns"] == [{"reply": "1 answers", "reasoning": ["thought about 1 answers"]}]

    def test_a_batch_answered_with_nothing_still_keeps_its_turn(self, llm_settings, tmp_path):
        # The reply is what explains a batch whose answers did not parse, so it has to be on disk.
        ledger = sweep.Ledger(tmp_path, "run", "abcd1234")
        result = sweep.Sweep(llm_settings, lambda s, batch, a: ({}, [_Turn("not JSON at all")]),
                             _items(2), ledger, batch_size=2).run()
        assert result.failed == 2
        traces = [json.loads(line) for line in ledger.traces_path.read_text().splitlines()]
        assert traces[0]["turns"][0]["reply"] == "not JSON at all"

    def test_a_reasked_item_is_asked_again_and_only_it(self, llm_settings, tmp_path):
        ledger = sweep.Ledger(tmp_path, "run", "abcd1234")
        ledger.write_answer("item0", "first answer")
        ledger.write_answer("item1", "kept")
        ledger.reask(["item0"])
        asked = []

        def ask(llm_settings, batch, maybe_abort):
            asked.extend(item_id for item_id, _ in batch)
            return {item_id: "second answer" for item_id, _ in batch}, []
        sweep.Sweep(llm_settings, ask, _items(2), ledger).run()
        assert asked == ["item0"]
        assert ledger.answers() == {"item0": "second answer", "item1": "kept"}

    def test_a_rerun_asks_only_what_is_not_answered(self, llm_settings, tmp_path):
        ledger = sweep.Ledger(tmp_path, "run", "abcd1234")
        ledger.write_answer("item0", "done earlier")
        ledger.write_failure("item1", "failed earlier")
        asked = []

        def ask(llm_settings, batch, maybe_abort):
            asked.extend(item_id for item_id, _ in batch)
            return _echo(llm_settings, batch, maybe_abort)
        result = sweep.Sweep(llm_settings, ask, _items(3), ledger).run()
        assert asked == ["item1", "item2"], "a failed item must be retried, an answered one skipped"
        assert result.total == 2
        assert ledger.answers()["item0"] == "done earlier"

    def test_a_failed_batch_while_the_backend_is_fine_is_recorded_and_the_run_goes_on(
            self, llm_settings, tmp_path, caplog):
        ledger = sweep.Ledger(tmp_path, "run", "abcd1234")

        def ask(llm_settings, batch, maybe_abort):
            if batch[0][0] == "item0":
                raise ValueError("expected a JSON array")
            return _echo(llm_settings, batch, maybe_abort)
        with caplog.at_level(logging.WARNING, logger="raven.librarian.sweep"):
            result = sweep.Sweep(llm_settings, ask, _items(4), ledger, batch_size=2).run()
        assert (result.outcome, result.done, result.failed) == (sweep.completed, 2, 2)
        assert set(ledger.answers()) == {"item2", "item3"}
        assert ledger.latest()["item0"]["status"] == "failed"
        # Each failed item is logged on its own, which is what a watcher of the run wants to see.
        assert sum("item0" in r.message or "item1" in r.message for r in caplog.records) == 2

    def test_an_item_the_reply_left_out_is_recorded_as_failed(self, llm_settings, tmp_path):
        ledger = sweep.Ledger(tmp_path, "run", "abcd1234")

        def ask(llm_settings, batch, maybe_abort):
            answers, reasoning = _echo(llm_settings, batch, maybe_abort)
            answers.pop("item1", None)
            return answers, reasoning
        result = sweep.Sweep(llm_settings, ask, _items(3), ledger, batch_size=3).run()
        assert (result.done, result.failed) == (2, 1)
        assert ledger.latest()["item1"]["status"] == "failed"

    def test_an_item_answered_with_a_failure_is_recorded_with_its_reason_and_turns(self, llm_settings, tmp_path):
        ledger = sweep.Ledger(tmp_path, "run", "abcd1234")

        def ask(llm_settings, batch, maybe_abort):
            return ({"item0": "fine", "item1": sweep.Failure("the abstract came back empty three times")},
                    [_Turn("why it was empty")])
        result = sweep.Sweep(llm_settings, ask, _items(2), ledger, batch_size=2).run()
        assert (result.done, result.failed) == (1, 1)
        assert ledger.latest()["item1"] == {"id": "item1", "v": "abcd1234", "status": "failed",
                                            "error": "the abstract came back empty three times"}
        traces = [json.loads(line) for line in ledger.traces_path.read_text().splitlines()]
        assert traces[0]["turns"][0]["reply"] == "why it was empty"

    def test_a_backend_that_went_away_stops_the_run(self, llm_settings, tmp_path, monkeypatch):
        ledger = sweep.Ledger(tmp_path, "run", "abcd1234")
        asked = []

        def ask(llm_settings, batch, maybe_abort):
            asked.append(batch[0][0])
            # The backend dies during this batch: the call fails, and so does the probe after it.
            def refuse(backend_url, flavor):
                raise llmclient.requests.exceptions.ConnectionError("nothing listening")
            monkeypatch.setattr(llmclient, "resolve_model_info", refuse)
            raise llmclient.requests.exceptions.ConnectionError("nothing listening")
        result = sweep.Sweep(llm_settings, ask, _items(3), ledger).run()
        assert result.outcome is sweep.backend_unavailable
        assert asked == ["item0"], "the run went on asking a backend that was not there"
        assert result.message  # the operator is told why
        assert set(ledger.latest()) == {"item0"}  # the rest were never tried, so nothing is said about them

    def test_a_swapped_model_stops_the_run_after_the_batch_it_answered(self, llm_settings, tmp_path, monkeypatch):
        ledger = sweep.Ledger(tmp_path, "run", "abcd1234")
        swapped = llmclient._model_info(**(dict(llm_settings.model_record) | {"quantization": "Q8_0"}))

        def ask(llm_settings, batch, maybe_abort):
            monkeypatch.setattr(llmclient, "resolve_model_info", lambda backend_url, flavor: swapped)
            return _echo(llm_settings, batch, maybe_abort)
        result = sweep.Sweep(llm_settings, ask, _items(3), ledger).run()
        assert result.outcome is sweep.model_changed
        assert set(ledger.answers()) == {"item0"}, "the batch the old model answered is kept, and no more is asked"
        assert "'test-model' was reloaded with other settings" in result.message

    def test_the_run_names_the_model_before_asking_and_both_on_a_swap(self, llm_settings, tmp_path, monkeypatch,
                                                                       caplog):
        ledger = sweep.Ledger(tmp_path, "run", "abcd1234")
        other = llmclient._model_info(**(dict(llm_settings.model_record) | {"label": "small-model", "model_id": "small-model"}))

        def ask(llm_settings, batch, maybe_abort):
            monkeypatch.setattr(llmclient, "resolve_model_info", lambda backend_url, flavor: other)
            return _echo(llm_settings, batch, maybe_abort)
        with caplog.at_level(logging.INFO, logger="raven.librarian.sweep"):
            result = sweep.Sweep(llm_settings, ask, _items(3), ledger).run()
        assert any("asking 'test-model'" in r.message for r in caplog.records)
        assert "from 'test-model' to 'small-model'" in result.message

    def test_with_nothing_wrong_the_probe_does_not_stop_the_run(self, llm_settings, tmp_path):
        # The control for the two above: the shared fixture's backend keeps serving the same model, so a
        # run over it must go to the end. Without this, a probe that always stopped would pass both.
        result = sweep.Sweep(llm_settings, _echo, _items(3), sweep.Ledger(tmp_path, "run", "abcd1234")).run()
        assert result.outcome is sweep.completed and result.done == 3

    def test_cancel_reaches_the_call_in_flight_and_drops_its_batch(self, llm_settings, tmp_path):
        ledger = sweep.Ledger(tmp_path, "run", "abcd1234")
        holder = {}

        def ask(llm_settings, batch, maybe_abort):
            if batch[0][0] == "item1":
                holder["sweep"].cancel()  # as a GUI thread would, while this call is waiting on the backend
                assert maybe_abort.aborted, "cancel did not reach the handle of the call in flight"
                raise netutil.Aborted("abandoned mid-read")
            return _echo(llm_settings, batch, maybe_abort)
        holder["sweep"] = sweep.Sweep(llm_settings, ask, _items(3), ledger)
        result = holder["sweep"].run()
        assert result.outcome is sweep.cancelled
        assert set(ledger.latest()) == {"item0"}, "the abandoned batch must not be recorded, so a resume starts with it"

    def test_a_cancel_between_batches_stops_before_the_next(self, llm_settings, tmp_path):
        ledger = sweep.Ledger(tmp_path, "run", "abcd1234")
        holder = {}
        run = holder["sweep"] = sweep.Sweep(llm_settings, _echo, _items(3), ledger,
                                            on_item=lambda event: holder["sweep"].cancel())
        result = run.run()
        assert result.outcome is sweep.cancelled
        assert set(ledger.answers()) == {"item0"}

    def test_progress_counts_only_this_run_s_items(self, llm_settings, tmp_path):
        ledger = sweep.Ledger(tmp_path, "run", "abcd1234")
        for i in range(6):
            ledger.write_answer(f"item{i}", "done earlier")
        events = []
        run = sweep.Sweep(llm_settings, _echo, _items(10), ledger, batch_size=2, on_item=events.append)
        run.run()
        assert [(event.done, event.total) for event in events] == [(2, 4), (4, 4)]
        assert events[0].eta.total == 2, "the estimate must count the batches left, not the whole set"
        assert run.progress.done == 4 and "4 / 4" in run.progress.status_line

    def test_the_header_is_written_once_per_run_that_has_work(self, llm_settings, tmp_path):
        ledger = sweep.Ledger(tmp_path, "run", "abcd1234")
        header = sweep.instrument_header(llm_settings)
        sweep.Sweep(llm_settings, _echo, _items(2), ledger, header=header).run()
        sweep.Sweep(llm_settings, _echo, _items(2), ledger, header=header).run()  # nothing left to do
        assert len(ledger.headers()) == 1

    def test_a_batch_size_below_one_is_refused(self, llm_settings, tmp_path):
        with pytest.raises(ValueError):
            sweep.Sweep(llm_settings, _echo, _items(1), sweep.Ledger(tmp_path, "run", "x"), batch_size=0)
