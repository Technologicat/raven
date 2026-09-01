"""Unit tests for raven.visualizer.importer.

The first tests in this package. `importer` is reachable from pytest because everything expensive in it
is loaded lazily -- the LLM connection is set up at import time only when the config asks for cluster
keywords or summaries, and the NLP and embedding models are loaded on first use -- so `_parse_input_files`
can be exercised against a `.bib` written into `tmp_path`, with the one remote service it touches
(dehyphenation) replaced.
"""

import itertools

import pytest

# `raven.visualizer.importer` reaches sklearn, torch and spaCy, none of which CI installs -- and a
# module-level import failure is a *collection* error rather than a skip, so it would turn the matrix
# red rather than quietly sitting out. Guarding on the module itself rather than on a list of packages
# keeps this correct as the import chain changes. `scripts/check_ci_imports.py` is what reports it.
importer = pytest.importorskip("raven.visualizer.importer")

pytestmark = pytest.mark.ml

from raven.visualizer import config as visualizer_config  # noqa: E402 -- must follow the guard above


TWO_RECORDS = """
@article{alpha2024,
  author = {Alpha, Anna},
  year = {2024},
  title = {A first paper about something},
  abstract = {An abstract that mentions auto-\nmatic hyphenation.}
}

@article{beta2024,
  author = {Beta, Bob},
  year = {2024},
  title = {A second paper about something else},
  abstract = {A second abstract, with no hyphenation in it at all.}
}
"""


class FakeRecord:
    """Stands in for `agent.turn`'s `TurnRecord`, of which only these two fields are read here."""
    def __init__(self, reply):
        self.reply = reply
        self.reasoning = []


class ExplodingDehyphenator:
    """Stands in for `mayberemote.Dehyphenator`, failing the way a real one has been seen to."""
    def __init__(self, *args, **kwargs):
        pass

    def dehyphenate(self, text):
        raise RuntimeError("dehyphenation exploded")


@pytest.fixture
def two_record_bib(tmp_path):
    path = tmp_path / "two_records.bib"
    path.write_text(TWO_RECORDS, encoding="utf-8")
    return path


def parsed_entries(input_data):
    """Flatten `_parse_input_files` output into one list, as `import_bibtex` does."""
    return list(itertools.chain.from_iterable(input_data.parsed_data_by_filename.values()))


def test_parse_input_files_reads_both_records(two_record_bib, monkeypatch):
    # Negative control for the test below: with dehyphenation off, nothing can throw, so a fixture that
    # only ever ran this way could not tell a surviving import from a lucky one.
    monkeypatch.setattr(visualizer_config, "dehyphenate", False)
    entries = parsed_entries(importer._parse_input_files(str(two_record_bib)))
    assert len(entries) == 2
    assert entries[0].title == "A first paper about something"
    assert entries[1].title == "A second paper about something else"


def test_parse_input_files_survives_a_failing_dehyphenator(two_record_bib, monkeypatch, caplog):
    # One malformed abstract used to abort the whole import, discarding every record already processed --
    # an hour's work on a large bibliography, with no way to skip the offending record. Dehyphenation is
    # cosmetic, so a failure now costs that one abstract its tidying and nothing else.
    monkeypatch.setattr(visualizer_config, "dehyphenate", True)
    monkeypatch.setattr(importer.mayberemote, "Dehyphenator", ExplodingDehyphenator)

    with caplog.at_level("WARNING"):
        entries = parsed_entries(importer._parse_input_files(str(two_record_bib)))

    # Both records survive: the run continues past the failure rather than stopping at it.
    assert len(entries) == 2, "a failing dehyphenator must not cost us any records"
    # The record that failed keeps its abstract, untidied rather than dropped.
    assert entries[0].abstract is not None
    assert "hyphenation" in entries[0].abstract
    # The failure is reported rather than swallowed, and names the record so it can be found.
    assert any("alpha2024" in record.message for record in caplog.records), \
        f"the warning should name the offending entry; got {[r.message for r in caplog.records]}"


def test_parse_input_files_skips_a_record_that_fails_anywhere_else(two_record_bib, monkeypatch, caplog):
    # The guard above is specific to dehyphenation, which is cosmetic. This one is the general case:
    # anything else that throws costs its own record and lets the run continue. Bibliographies arrive
    # from exporters we do not control, so the point is not this particular failure but that no single
    # record can end the import.
    monkeypatch.setattr(visualizer_config, "dehyphenate", False)

    real_format_authors = importer.common_utils.format_bibtex_authors

    def explode_on_alpha(author_field):
        # `fields["author"].value` is bibtexparser's parsed name list, not a string, so match on its
        # repr rather than on the value -- checking `"Alpha" in author_field` silently matches nothing
        # and the test then passes against the unfixed code, having exercised no failure at all.
        if "Alpha" in str(author_field):
            raise ValueError("author field exploded")
        return real_format_authors(author_field)

    monkeypatch.setattr(importer.common_utils, "format_bibtex_authors", explode_on_alpha)

    with caplog.at_level("WARNING"):
        entries = parsed_entries(importer._parse_input_files(str(two_record_bib)))

    # The good record survives, so the failure did not end the run. Note this is the assertion that
    # distinguishes "skipped it" from "crashed": both leave the bad record out.
    assert len(entries) == 1, "the record after the failing one must still be parsed"
    assert entries[0].title == "A second paper about something else"
    assert any("beta2024" not in record.message and "alpha2024" in record.message
               for record in caplog.records), \
        f"the warning should name the skipped entry; got {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Cluster keyword canonicalization


def test_canonicalization_mapping_keeps_only_verifiable_replacements():
    # The mapping is parsed rather than trusted, which is the whole reason the prompt asks for a mapping
    # instead of for a rewritten keyword list. Anything whose replacement was not itself extracted is
    # dropped, so a model that invents or rephrases a term cannot get it into the dataset.
    known = {"LLM", "Large Language Models", "Chain-of-Thought", "Reasoning"}
    reply = ("LLM -> Large Language Models\n"
             "Chain-of-Thought -> Chain of Thought Prompting\n"   # replacement was never extracted
             "Reasoning -> Reasoning\n"                           # self-mapping, no-op
             "Nonexistent -> Reasoning\n"                         # original was never extracted
             "this line has no arrow at all\n")
    mapping = importer._parse_canonicalization_mapping(reply, known)
    assert mapping == {"LLM": "Large Language Models"}


def test_canonicalization_mapping_drops_chains():
    # `a -> b` together with `b -> c` needs resolving in an order the model was never asked to give, so
    # the far end is dropped rather than guessed at.
    known = {"a", "b", "c"}
    mapping = importer._parse_canonicalization_mapping("a -> b\nb -> c\n", known)
    assert mapping == {"b": "c"}


def test_canonicalize_cluster_keywords_merges_variants_without_reordering(monkeypatch):
    clusters = [["LLM", "Reasoning"],
                ["Large Language Models", "Benchmarking"],
                ["Reasoning", "LLM"]]  # a cluster where the replacement collides with a keyword it has

    def fake_turn(settings, prompt, **kwargs):
        return FakeRecord("LLM -> Large Language Models")

    # `agent` and `llm_settings` are bound at import time only when the config asks for LLM work, so a
    # test that drives this path has to supply them.
    monkeypatch.setattr(importer, "agent", type("_FakeAgent", (), {"turn": staticmethod(fake_turn)}), raising=False)
    monkeypatch.setattr(importer, "llm_settings", object(), raising=False)
    monkeypatch.setattr(importer, "llmclient",
                        type("_FakeLLMClient", (),
                             {"make_console_progress_handler": staticmethod(lambda _: None)}),
                        raising=False)

    out = importer._canonicalize_cluster_keywords(clusters)

    assert out[0] == ["Large Language Models", "Reasoning"]
    assert out[1] == ["Large Language Models", "Benchmarking"], "an untouched keyword keeps its place"
    # The replacement collides with a keyword the cluster already had; the duplicate goes and the
    # original order survives, since that order is the model's ranking and reads as most-important-first.
    assert out[2] == ["Reasoning", "Large Language Models"]


def test_canonicalize_cluster_keywords_passes_through_when_nothing_to_do(monkeypatch):
    # Negative control for the test above: with one keyword there is nothing to canonicalize, so the LLM
    # is never consulted. Were it consulted, the fake below would raise and this would fail rather than
    # quietly agreeing with the treatment case.
    def explode(*args, **kwargs):
        raise AssertionError("the model should not be asked when there is nothing to canonicalize")

    monkeypatch.setattr(importer, "agent", type("_FakeAgent", (), {"turn": staticmethod(explode)}), raising=False)
    clusters = [["Large Language Models"]]
    assert importer._canonicalize_cluster_keywords(clusters) is clusters


# ---------------------------------------------------------------------------
# Oversized clusters


def test_oversized_cluster_is_sampled_from_its_center_outwards(monkeypatch):
    # A cluster with hundreds of entries builds a prompt no backend will take, so it is capped. The
    # sample is spread along the centrality ordering rather than taken from the top of it, because the
    # top of it describes the cluster's densest part and a big cluster is the one most likely to span
    # several subtopics.
    #
    # The fixture is built so the two behaviours disagree: nine entries share one direction and the
    # tenth is orthogonal to them, so it sorts last by centrality. Taking the five *most central* would
    # leave it out; spreading five picks across the ordering includes it. Asserting that it is present
    # is therefore an assertion about the spreading, not merely about the capping.
    import numpy as np
    from unpythonic.env import env

    vis_data = [env(title=f"central {i}", abstract="", cluster_id=0, cluster_probability=1.0)
                for i in range(9)]
    vis_data.append(env(title="the outlier", abstract="", cluster_id=0, cluster_probability=1.0))
    all_vectors = np.array([[1.0, 0.0]] * 9 + [[0.0, 1.0]])

    prompts = []

    def fake_turn(settings, prompt, **kwargs):
        prompts.append(prompt)
        return FakeRecord("Keyword A, Keyword B")

    monkeypatch.setattr(visualizer_config, "clusters_keyword_method", "llm")
    monkeypatch.setattr(importer, "agent", type("_FakeAgent", (), {"turn": staticmethod(fake_turn)}), raising=False)
    monkeypatch.setattr(importer, "llm_settings", object(), raising=False)
    monkeypatch.setattr(importer, "llmclient",
                        type("_FakeLLMClient", (),
                             {"make_console_progress_handler": staticmethod(lambda _: None)}),
                        raising=False)
    monkeypatch.setattr(importer, "_canonicalize_cluster_keywords", lambda keywords: keywords)

    importer._collect_cluster_keywords(vis_data, 1, {}, all_vectors, max_prompt_entries=5)

    assert len(prompts) == 1
    body = prompts[0].split("-----", 1)[1]
    assert body.count("\n\n\n") + 1 == 5, "the prompt should carry exactly the capped number of entries"
    assert "the outlier" in body, \
        "the least central entry should still be sampled; taking the most central ones would drop it"


def test_small_cluster_is_sent_whole(monkeypatch):
    # Negative control for the test above: below the cap nothing is sampled, so every entry is present.
    # Without this, a bug that sent one entry per cluster would still satisfy the assertions above.
    import numpy as np
    from unpythonic.env import env

    vis_data = [env(title=f"paper {i}", abstract="", cluster_id=0, cluster_probability=1.0)
                for i in range(4)]
    all_vectors = np.array([[1.0, 0.0]] * 4)

    prompts = []

    def fake_turn(settings, prompt, **kwargs):
        prompts.append(prompt)
        return FakeRecord("Keyword A")

    monkeypatch.setattr(visualizer_config, "clusters_keyword_method", "llm")
    monkeypatch.setattr(importer, "agent", type("_FakeAgent", (), {"turn": staticmethod(fake_turn)}), raising=False)
    monkeypatch.setattr(importer, "llm_settings", object(), raising=False)
    monkeypatch.setattr(importer, "llmclient",
                        type("_FakeLLMClient", (),
                             {"make_console_progress_handler": staticmethod(lambda _: None)}),
                        raising=False)
    monkeypatch.setattr(importer, "_canonicalize_cluster_keywords", lambda keywords: keywords)

    importer._collect_cluster_keywords(vis_data, 1, {}, all_vectors, max_prompt_entries=60)

    body = prompts[0].split("-----", 1)[1]
    for i in range(4):
        assert f"paper {i}" in body
