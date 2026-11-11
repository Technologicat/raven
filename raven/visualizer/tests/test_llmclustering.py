"""Tests for `raven.visualizer.llmclustering`: the prompt, reading the grouping back, repairs, the layout, the size check."""

import json
import math

import numpy as np
import pytest
from unpythonic.env import env

from raven.visualizer import llmclustering


def _entries(n):
    return [env(author=f"Author{i}", year="2024", title=f"Paper {i}", abstract=f"About topic {i % 3}.") for i in range(n)]


def _reply(themes):
    return json.dumps({"themes": themes})


def _theme(name, items, primary=True):
    return {"name": name, "description": f"{name}.", "papers": [{"item": i, "primary": primary, "reason": "fits"} for i in items]}


class TestPrompt:
    def test_every_paper_is_numbered_from_one(self):
        prompt = llmclustering.build_prompt(_entries(3))
        assert "[1] Author0 (2024). Paper 0" in prompt and "[3] Author2 (2024). Paper 2" in prompt
        assert "all 3 papers" in prompt

    def test_an_aim_is_given_as_context_only_when_there_is_one(self):
        with_aim = llmclustering.build_prompt(_entries(2), maybe_aim="Teaching with chatbots.")
        without = llmclustering.build_prompt(_entries(2), maybe_aim="  ")
        assert "Teaching with chatbots." in with_aim and "do not make one theme per part" in with_aim
        assert "aim" not in without


class TestReadingTheGroupingBack:
    def test_a_clean_reply_has_no_problems(self):
        themes, problems = llmclustering.read_grouping(_reply([_theme("A", [1, 2]), _theme("B", [3])]), 3)
        assert problems == []
        assert [[p["index"] for p in t["placements"]] for t in themes] == [[0, 1], [2]]

    def test_each_slip_is_reported(self):
        reply = _reply([_theme("A", [1, 1, 9])])  # paper 1 twice, item 9 out of range, papers 2 and 3 nowhere
        _, problems = llmclustering.read_grouping(reply, 3)
        assert any("twice" in p for p in problems)
        assert any("item 9 names no paper" in p for p in problems)
        assert sum("placed nowhere" in p for p in problems) == 2

    def test_a_paper_with_no_primary_placement_gets_its_first(self):
        themes, problems = llmclustering.read_grouping(_reply([_theme("A", [1], primary=False), _theme("B", [1], primary=False)]), 1)
        assert any("no placement marked primary" in p for p in problems)
        assert [p["primary"] for t in themes for p in t["placements"]] == [True, False]

    def test_a_paper_with_two_primaries_keeps_the_first(self):
        themes, problems = llmclustering.read_grouping(_reply([_theme("A", [1]), _theme("B", [1])]), 1)
        assert any("exactly one" in p for p in problems)
        assert [p["primary"] for t in themes for p in t["placements"]] == [True, False]

    def test_a_reply_that_is_not_json_of_the_asked_shape_is_refused(self):
        with pytest.raises(ValueError):
            llmclustering.read_grouping('{"clusters": []}', 3)


class TestRepairs:
    def _fake(self, monkeypatch, replies, truncated=False):
        asked = []

        def ask_record(settings, prompt, max_tokens=None, **kwargs):
            asked.append(prompt)
            return env(reply=replies[len(asked) - 1], reasoning=[], truncated=truncated)
        from raven.librarian import agent
        monkeypatch.setattr(agent, "ask_record", ask_record)
        return asked

    def test_a_slip_is_named_in_a_repair_turn_and_the_fixed_reply_is_used(self, monkeypatch):
        asked = self._fake(monkeypatch, [_reply([_theme("A", [1])]), _reply([_theme("A", [1, 2])])])
        themes, problems = llmclustering.ask_with_repairs(None, "PROMPT", 2)
        assert problems == [] and len(asked) == 2
        assert asked[1].startswith("PROMPT") and "paper [2] is placed nowhere" in asked[1]

    def test_what_is_left_after_the_last_repair_is_returned_fixed_up(self, monkeypatch):
        slipped = _reply([_theme("A", [1])])
        asked = self._fake(monkeypatch, [slipped] * (llmclustering.MAX_REPAIRS + 1))
        _, problems = llmclustering.ask_with_repairs(None, "PROMPT", 2)
        assert len(asked) == llmclustering.MAX_REPAIRS + 1 and problems == ["paper [2] is placed nowhere"]

    def test_a_reply_never_readable_raises(self, monkeypatch):
        self._fake(monkeypatch, ["not json"] * (llmclustering.MAX_REPAIRS + 1))
        with pytest.raises(RuntimeError, match="unreadable"):
            llmclustering.ask_with_repairs(None, "PROMPT", 2)

    def test_a_reply_cut_off_at_the_cap_raises(self, monkeypatch):
        self._fake(monkeypatch, [_reply([_theme("A", [1])])], truncated=True)
        with pytest.raises(RuntimeError, match="cut off"):
            llmclustering.ask_with_repairs(None, "PROMPT", 1)


class TestLayout:
    def _themes(self):
        themes, _ = llmclustering.read_grouping(_reply([_theme("A", range(1, 7)), _theme("B", range(7, 10)), _theme("C", [10])]), 11)
        return themes  # paper 11 is placed nowhere

    def test_each_paper_is_labelled_with_its_primary_theme_and_the_unplaced_as_outliers(self):
        points, labels = llmclustering.layout(self._themes(), 11)
        assert points.shape == (11, 2)
        assert list(labels) == [0] * 6 + [1] * 3 + [2] + [-1]

    def test_a_themes_members_sit_on_one_circle(self):
        points, labels = llmclustering.layout(self._themes(), 11)
        members = points[labels == 0]
        center = members.mean(axis=0)
        radii = np.linalg.norm(members - center, axis=1)
        assert np.allclose(radii, radii[0]) and radii[0] > 0

    def test_no_two_points_coincide_and_the_circles_do_not_overlap(self):
        points, labels = llmclustering.layout(self._themes(), 11)
        assert min(math.dist(a, b) for i, a in enumerate(points) for b in points[i + 1:]) > 0.5
        circles = []
        for label in sorted(set(labels.tolist())):
            members = points[labels == label]
            center = members.mean(axis=0)
            circles.append((center, float(np.max(np.linalg.norm(members - center, axis=1)))))
        for i, (c1, r1) in enumerate(circles):
            for c2, r2 in circles[i + 1:]:
                assert math.dist(c1, c2) > r1 + r2


class TestTheSizeCheck:
    def test_a_dataset_that_does_not_fit_is_refused_before_asking(self, monkeypatch):
        from raven.librarian import agent, llmclient
        monkeypatch.setattr(llmclient, "count_tokens", lambda settings, text: (50_000, True))
        monkeypatch.setattr(llmclient, "wait_for_tokenizer", lambda settings: True)
        monkeypatch.setattr(agent, "ask_record", lambda *a, **k: pytest.fail("asked although the prompt does not fit"))
        settings = env(context_length=llmclustering.MAX_REPLY_TOKENS + 10_000)
        with pytest.raises(llmclustering.PromptTooLarge, match="agglomerative"):
            llmclustering.cluster(settings, _entries(3))

    def test_a_dataset_that_fits_is_asked_about(self, monkeypatch):
        # The control for the test above: the same call with room to spare goes through to the model.
        from raven.librarian import agent, llmclient
        monkeypatch.setattr(llmclient, "count_tokens", lambda settings, text: (5_000, True))
        monkeypatch.setattr(llmclient, "wait_for_tokenizer", lambda settings: True)
        monkeypatch.setattr(agent, "ask_record",
                            lambda *a, **k: env(reply=_reply([_theme("A", [1, 2, 3])]), reasoning=[], truncated=False))
        labels, points, themes = llmclustering.cluster(env(context_length=llmclustering.MAX_REPLY_TOKENS + 10_000), _entries(3))
        assert list(labels) == [0, 0, 0] and themes[0]["name"] == "A"
