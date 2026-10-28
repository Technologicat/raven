"""Tests for `raven.visualizer.savedstate`: the GUI choices kept between sessions."""

import json
import logging

import pytest

from raven.visualizer import savedstate


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "visualizer" / "state.json"


class TestLoad:
    def test_a_missing_file_gives_the_defaults(self, state_path):
        assert savedstate.load(state_path) == savedstate.DEFAULTS

    def test_a_saved_value_is_read_back(self, state_path):
        state_path.parent.mkdir()
        state_path.write_text(json.dumps({"panel_side": "right"}))
        assert savedstate.load(state_path)["panel_side"] == "right"

    def test_a_missing_entry_gets_its_default(self, state_path):
        state_path.parent.mkdir()
        state_path.write_text("{}")
        assert savedstate.load(state_path) == savedstate.DEFAULTS

    def test_a_value_outside_the_allowed_ones_gets_the_default(self, state_path, caplog):
        state_path.parent.mkdir()
        state_path.write_text(json.dumps({"panel_side": "top"}))
        with caplog.at_level(logging.WARNING):
            assert savedstate.load(state_path)["panel_side"] == "left"
        assert "panel_side" in caplog.text

    def test_an_unknown_entry_is_ignored(self, state_path):
        state_path.parent.mkdir()
        state_path.write_text(json.dumps({"panel_side": "right", "from_the_future": 42}))
        assert savedstate.load(state_path) == {"panel_side": "right"}

    @pytest.mark.parametrize("text", ["{not json", "[1, 2]", ""])
    def test_a_malformed_file_gives_the_defaults_with_a_warning(self, state_path, caplog, text):
        state_path.parent.mkdir()
        state_path.write_text(text)
        with caplog.at_level(logging.WARNING):
            assert savedstate.load(state_path) == savedstate.DEFAULTS
        assert caplog.records, "a malformed state file should be reported, not silently replaced"

    def test_the_defaults_are_a_copy(self, state_path):
        state = savedstate.load(state_path)
        state["panel_side"] = "right"
        assert savedstate.DEFAULTS["panel_side"] == "left"


class TestSave:
    def test_a_save_round_trips_through_load(self, state_path):
        savedstate.save(state_path, {"panel_side": "right"})
        assert savedstate.load(state_path) == {"panel_side": "right"}

    def test_the_directory_is_created(self, state_path):
        assert not state_path.parent.exists()
        savedstate.save(state_path, {"panel_side": "left"})
        assert state_path.exists()

    def test_only_the_known_entries_are_written(self, state_path):
        savedstate.save(state_path, {"panel_side": "right", "transient": object()})
        assert json.loads(state_path.read_text()) == {"panel_side": "right"}

    def test_a_str_path_works_as_well_as_a_pathlib_path(self, state_path):
        savedstate.save(str(state_path), {"panel_side": "right"})
        assert savedstate.load(str(state_path)) == {"panel_side": "right"}


class TestDeclarations:
    def test_every_default_is_one_of_its_allowed_values(self):
        assert set(savedstate.DEFAULTS) == set(savedstate.ALLOWED)
        for key, value in savedstate.DEFAULTS.items():
            assert value in savedstate.ALLOWED[key], key
