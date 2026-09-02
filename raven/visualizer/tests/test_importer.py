"""Unit tests for raven.visualizer.importer.

The first tests in this package. `importer` is reachable from pytest because everything expensive in it
is loaded lazily -- the LLM connection is set up at import time only when the config asks for cluster
keywords or summaries, and the NLP and embedding models are loaded on first use -- so `_parse_input_files`
can be exercised against a `.bib` written into `tmp_path`, with the one remote service it touches
(dehyphenation) replaced.
"""

import concurrent.futures
import itertools
import logging
import threading

import pytest

# `raven.visualizer.importer` reaches sklearn, torch and spaCy, none of which CI installs -- and a
# module-level import failure is a *collection* error rather than a skip, so it would turn the matrix
# red rather than quietly sitting out. Guarding on the module itself rather than on a list of packages
# keeps this correct as the import chain changes. `scripts/check_ci_imports.py` is what reports it.
importer = pytest.importorskip("raven.visualizer.importer")

pytestmark = pytest.mark.ml

from unpythonic import unbox  # noqa: E402 -- must follow the guard above

from raven.visualizer import config as visualizer_config  # noqa: E402 -- ditto


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


MISSING_FIELDS = """
@article{notitle2023,
  author = {Gamma, Gemma},
  year = {2023},
  abstract = {A record whose database exported everything except the title.}
}

@article{noauthor2023,
  year = {2023},
  title = {A paper whose database omitted its authors},
  abstract = {ECCOMAS 2024 exported records in this shape.}
}

@article{noyear2023,
  author = {Epsilon, Eve},
  title = {A paper whose database omitted its year}
}

@article{nothing2023,
  author = {Delta, Dee},
  year = {2023}
}
"""


@pytest.fixture
def missing_fields_bib(tmp_path):
    path = tmp_path / "missing_fields.bib"
    path.write_text(MISSING_FIELDS, encoding="utf-8")
    return path


def test_an_incomplete_record_is_imported_with_each_missing_field_named(missing_fields_bib, monkeypatch, caplog):
    # A database export can omit any of the bibliographic fields while carrying the rest, and dropping
    # the record loses everything it did carry. So each of the three is named as absent rather than
    # costing the record its place, and all three are handled alike.
    monkeypatch.setattr(visualizer_config, "dehyphenate", False)
    with caplog.at_level(logging.WARNING, logger="raven.visualizer.importer"):
        entries = parsed_entries(importer._parse_input_files(str(missing_fields_bib)))

    by_placeholder = {entry.title: entry for entry in entries}
    assert len(entries) == 3, "only the record with nothing to read should be skipped"

    no_title = by_placeholder[importer.MISSING_TITLE]
    assert "everything except the title" in no_title.abstract, "it kept what made it worth keeping"

    no_author = next(e for e in entries if e.author == importer.MISSING_AUTHOR)
    assert no_author.title == "A paper whose database omitted its authors"
    no_year = next(e for e in entries if e.year == importer.MISSING_YEAR)
    assert no_year.title == "A paper whose database omitted its year"

    # Each substitution says so. A reader meeting a placeholder later has no way to tell it from
    # something a database actually said.
    messages = [record.message for record in caplog.records]
    for key in ("notitle2023", "noauthor2023", "noyear2023"):
        assert any(key in message and "has no" in message for message in messages)


def test_a_record_with_neither_a_title_nor_an_abstract_is_skipped(missing_fields_bib, monkeypatch, caplog):
    # The one thing a record cannot do without is something to read. The title placeholder is the same
    # string on every such record, so importing them would cluster them on Raven's own boilerplate.
    monkeypatch.setattr(visualizer_config, "dehyphenate", False)
    with caplog.at_level(logging.WARNING, logger="raven.visualizer.importer"):
        entries = parsed_entries(importer._parse_input_files(str(missing_fields_bib)))

    assert "nothing2023" not in {entry.title for entry in entries}
    # The negative control: the other two incomplete records are also missing a mandatory-looking field
    # and *are* imported, so this fixture is showing that having nothing to read is what decides, rather
    # than incompleteness in general.
    assert len(entries) == 3
    assert any("nothing2023" in record.message and "no title and no abstract" in record.message
               for record in caplog.records)


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


# ---------------------------------------------------------------------------
# Records `bibtexparser` refused


STRAY_BRACE = """@article{stray2024,
  author = {Alpha, Anna},
  year = {2024},
  title = {A paper with a stray brace},
  abstract = {We consider the set {a, b and nothing closes it.}
}
"""

# Not a brace fault at all: the field has no `=`, so there is nothing for the brace repair to propose.
BEYOND_REPAIR = """@article{hopeless2024,
  author {Beta, Bob},
  year = {2024},
  title = {A paper that cannot be saved}
}
"""

GOOD_RECORD = """@article{fine2024,
  author = {Gamma, Gil},
  year = {2024},
  title = {A paper that parses},
  abstract = {Nothing wrong here.}
}
"""


def parse_bib(text):
    from raven.papers import bibtex
    return bibtex.parse_string(text)


def test_a_record_with_a_stray_brace_is_recovered_into_the_library():
    # A stray `{` in an abstract -- mathematics arriving through a PDF extractor is the usual source --
    # aborts the parse of the whole record, title and all. Escaping it recovers the record whole, and the
    # entry is added as though it had parsed.
    library = parse_bib(STRAY_BRACE + "\n" + GOOD_RECORD)
    assert [entry.key for entry in library.entries] == ["fine2024"], \
        "the stray-brace record parsed on its own, so this fixture cannot tell a recovery from a no-op"

    importer._report_unparseable_records("test.bib", library)
    assert sorted(entry.key for entry in library.entries) == ["fine2024", "stray2024"]


def test_a_recovered_record_keeps_the_fields_that_were_lost_with_it():
    # The point of recovering is the whole record, not just its key: everything downstream needs the
    # title, and the reason the record failed was a field the user cannot see is missing.
    library = parse_bib(STRAY_BRACE)
    importer._report_unparseable_records("test.bib", library)
    recovered = next(entry for entry in library.entries if entry.key == "stray2024")
    assert recovered["title"] == "A paper with a stray brace"
    assert set(recovered.fields_dict) >= {"author", "year", "title", "abstract"}
    assert "a, b" in recovered["abstract"], "the field the record died on should be there, brace and all"


def test_a_recovery_is_reported_naming_the_record_and_the_file(caplog):
    # Raven repairs its own reading of the file and never the file, so the user has to be told that a
    # record in their bibliography needs fixing and where -- `raven-fixbib` is what writes it back.
    library = parse_bib(STRAY_BRACE)
    with caplog.at_level("WARNING"):
        importer._report_unparseable_records("mybib.bib", library)
    messages = [record.message for record in caplog.records]
    assert any("stray2024" in message and "mybib.bib" in message for message in messages), messages
    assert any("raven-fixbib" in message for message in messages), messages


def test_a_record_that_cannot_be_recovered_is_reported_rather_than_vanishing(caplog):
    # The asymmetry this removes: a record that parses but lacks `author`, `year` or `title` is skipped
    # further down with a warning naming it, while a record that never became an entry at all would
    # otherwise disappear silently -- and that is the case the user has no other way to notice.
    library = parse_bib(BEYOND_REPAIR + "\n" + GOOD_RECORD)
    with caplog.at_level("WARNING"):
        importer._report_unparseable_records("mybib.bib", library)

    assert [entry.key for entry in library.entries] == ["fine2024"], "nothing should have been recovered here"
    messages = [record.message for record in caplog.records]
    assert any("hopeless2024" in message for message in messages), \
        f"the unparseable record should be named; got {messages}"


def test_the_report_counts_the_lost_records(caplog):
    library = parse_bib(BEYOND_REPAIR + "\n" + BEYOND_REPAIR.replace("hopeless2024", "hopeless2025"))
    with caplog.at_level("WARNING"):
        importer._report_unparseable_records("mybib.bib", library)
    assert any("2 records" in record.message for record in caplog.records), \
        [record.message for record in caplog.records]


def test_a_clean_file_is_reported_on_at_all(caplog):
    # Negative control for the four above: with nothing to report the pass returns immediately, so a
    # warning here would mean the check fires on healthy files too.
    library = parse_bib(GOOD_RECORD)
    with caplog.at_level("WARNING"):
        importer._report_unparseable_records("mybib.bib", library)
    assert not [record for record in caplog.records if record.levelname == "WARNING"]


# ---------------------------------------------------------------------------
# Progress reporting


@pytest.fixture
def progress(monkeypatch):
    """A fresh progress counter that believes a task is running, since that is what it reports for."""
    monkeypatch.setattr(importer, "has_task", lambda: True)
    counter = importer._Progress()
    return counter


def test_progress_is_none_when_no_task_is_running(monkeypatch):
    # The branch `importer_gui.update_status` handles by showing 0%: the counter has no meaning between
    # runs, and reporting a stale fraction would be worse than reporting nothing.
    monkeypatch.setattr(importer, "has_task", lambda: False)
    assert importer._Progress().value is None


def test_progress_starts_at_zero_and_each_macrostep_is_an_equal_share(progress):
    assert progress.value == pytest.approx(0.0)
    progress.tock()
    assert progress.value == pytest.approx(1 / 8), "eight macrosteps, so each is an eighth"
    progress.tock()
    assert progress.value == pytest.approx(2 / 8)


def test_microsteps_interpolate_within_the_current_macrostep(progress):
    progress.tock()
    progress.set_micro_count(4)
    progress.tick()
    progress.tick()
    # Half way through the second macrostep: one whole macrostep, plus half of the next one's share.
    assert progress.value == pytest.approx((1 + 0.5) / 8)


def test_a_macrostep_does_not_inherit_the_previous_one_s_microstep_count(progress):
    # Without the reset, a macrostep that never calls `set_micro_count` would divide its single tick by
    # whatever the previous step happened to use, and the bar would crawl instead of stepping.
    progress.set_micro_count(100)
    progress.tock()
    progress.tick()
    assert progress.value == pytest.approx((1 + 1) / 8)


def test_resetting_takes_the_counter_back_to_the_start(progress):
    progress.set_micro_count(4)
    progress.tick()
    progress.tock()
    assert progress.value > 0.0, "nothing advanced, so this fixture cannot tell a reset from a fresh counter"
    progress.reset()
    assert progress.value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Formatting an entry for keyword extraction


def test_an_entry_with_an_abstract_is_its_title_then_the_abstract():
    from unpythonic.env import env
    entry = env(title="A title", abstract="An abstract.", author="Alpha, Anna", year="2024")
    assert importer._format_entry_for_keyword_extraction(entry) == "A title.\n\nAn abstract."


def test_an_entry_without_an_abstract_is_its_bare_title():
    # No full stop is added here, unlike the case above, where the stop separates title from abstract.
    from unpythonic.env import env
    entry = env(title="A title", abstract="", author="Alpha, Anna", year="2024")
    assert importer._format_entry_for_keyword_extraction(entry) == "A title"


def test_an_entry_whose_title_is_the_placeholder_is_analyzed_from_its_abstract_alone():
    # `MISSING_TITLE` is Raven's word, not the record's, and it is the *same* word on every such record —
    # so letting it reach the embedder or the keyword extractor would gather those records into a cluster
    # whose members share nothing but a field their database omitted.
    from unpythonic.env import env
    entry = env(title=importer.MISSING_TITLE, abstract="An abstract.", author="Alpha, Anna", year="2024")
    assert importer._format_entry_for_keyword_extraction(entry) == "An abstract."

    # The negative control: an ordinary title in the same position *is* joined on, so this fixture is
    # showing that the placeholder is treated specially rather than that the abstract wins in general.
    ordinary = env(title="A title", abstract="An abstract.", author="Alpha, Anna", year="2024")
    assert importer._format_entry_for_keyword_extraction(ordinary) == "A title.\n\nAn abstract."


def test_authors_and_year_are_left_out():
    # They are not relevant to what a paper is about, and an author name repeated across a cluster's
    # entries would read to a frequency count as a keyword.
    from unpythonic.env import env
    entry = env(title="A title", abstract="An abstract.", author="Zzyzx, Quentin", year="1999")
    formatted = importer._format_entry_for_keyword_extraction(entry)
    assert "Zzyzx" not in formatted
    assert "1999" not in formatted


# ---------------------------------------------------------------------------
# Status updates


def test_a_status_update_goes_nowhere_when_nobody_is_listening():
    # The dynvar's default is a no-op, which is what lets the pipeline call this unconditionally --
    # `raven-importer` on the command line has no GUI to update.
    importer._update_status_and_log("Parsing input files...")


def test_a_status_update_reaches_the_listener_verbatim():
    from unpythonic import dyn
    seen = []
    with dyn.let(maybe_update_status=seen.append):
        importer._update_status_and_log("Parsing input files...")
    assert seen == ["Parsing input files..."]


def test_the_log_indent_does_not_reach_the_listener(caplog):
    # The indent is for reading the log as a tree of steps; the GUI's status line is one line wide and
    # would show it as leading blanks.
    from unpythonic import dyn
    seen = []
    with caplog.at_level("INFO"):
        with dyn.let(maybe_update_status=seen.append):
            importer._update_status_and_log("Clustering...", log_indent=2)
    assert seen == ["Clustering..."]
    assert any(record.message == "        Clustering..." for record in caplog.records), \
        f"the log line should carry four spaces per indent level; got {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# The background task


@pytest.fixture
def uninitialized(monkeypatch):
    """`importer` as it is on a fresh import: no executor, no task manager."""
    monkeypatch.setattr(importer, "bg", None)
    monkeypatch.setattr(importer, "task_manager", None)


@pytest.fixture
def initialized(uninitialized, monkeypatch):
    """`importer` with a real executor behind it, torn down afterwards.

    The executor is real because the point of these tests is the task wrapper -- callbacks, result codes,
    the status the GUI reads -- and a fake one would be asserting against the fake. What is replaced is
    `import_bibtex`, the hour-long part.

    `init` registers an `atexit` cleanup that reads the module-level `task_manager` when the interpreter
    exits -- long after this fixture has put it back to `None`. Left alone, every test using this fixture
    contributes an `AttributeError` to the end of the run. Collecting the registrations instead of making
    them keeps the teardown honest, and `test_initialization_registers_a_cleanup` is what covers the real
    one.
    """
    registered = []
    monkeypatch.setattr(importer.atexit, "register", registered.append)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    importer.init(executor)
    yield registered
    if importer.task_manager is not None:
        importer.task_manager.clear(wait=True)
    executor.shutdown(wait=True)


def run_one_task(fake_import_bibtex, monkeypatch, timeout=10.0):
    """Start an import whose pipeline is `fake_import_bibtex`, wait for it, and return its `task_env`."""
    monkeypatch.setattr(importer, "import_bibtex", fake_import_bibtex)
    finished = threading.Event()
    seen = []

    def done_callback(task_env):
        seen.append(task_env)
        finished.set()

    assert importer.start_task(None, done_callback, "/out/dataset.pickle", "/in/one.bib") is True
    assert finished.wait(timeout), "the importer task never finished"
    return seen[0]


def test_no_task_exists_before_the_module_is_initialized(uninitialized):
    assert importer.has_task() is False


def test_cancelling_before_initialization_is_a_no_op(uninitialized):
    # Teardown paths call this whatever killed the app, including a failure during bootup.
    importer.cancel_task()


def test_an_import_cannot_start_before_the_module_is_initialized(uninitialized):
    # The GUI's start button is live from the first frame, and `init` happens later in the app's bootup.
    assert importer.start_task(None, None, "/out/dataset.pickle", "/in/one.bib") is False


def test_initialization_registers_a_cleanup(initialized):
    # An import holds a worker thread, so a process exiting mid-import would otherwise hang waiting for it.
    assert len(initialized) == 1


def test_initialization_is_idempotent(initialized):
    # `init` is called from the app's bootup and from `raven-importer`; a second call must not swap the
    # executor out from under a task that is already using it.
    first = importer.task_manager
    importer.init(concurrent.futures.ThreadPoolExecutor(max_workers=1))
    assert importer.task_manager is first


def test_a_task_runs_the_pipeline_with_the_filenames_it_was_given(initialized, monkeypatch):
    called = []
    task_env = run_one_task(lambda status_cb, out, *ins: called.append((out, ins)), monkeypatch)
    assert called == [("/out/dataset.pickle", ("/in/one.bib",))]
    assert task_env.result_code is importer.result_successful


def test_only_one_import_runs_at_a_time(initialized, monkeypatch):
    # An import takes a lot of GPU and CPU, so a second one alongside the first would make both slower
    # and could exhaust VRAM.
    release = threading.Event()
    monkeypatch.setattr(importer, "import_bibtex", lambda status_cb, out, *ins: release.wait(10.0))
    assert importer.start_task(None, None, "/out/dataset.pickle", "/in/one.bib") is True
    try:
        assert importer.has_task(), "nothing was running, so this fixture cannot tell a refusal from a start"
        assert importer.start_task(None, None, "/out/other.pickle", "/in/two.bib") is False
    finally:
        release.set()


def test_the_started_callback_fires_before_the_pipeline_does(initialized, monkeypatch):
    # The GUI re-enables its stop button from this callback, so it must arrive while there is still
    # something to stop.
    order = []
    monkeypatch.setattr(importer, "import_bibtex", lambda status_cb, out, *ins: order.append("pipeline"))
    finished = threading.Event()
    importer.start_task(lambda task_env: order.append("started"), lambda task_env: finished.set(),
                        "/out/dataset.pickle", "/in/one.bib")
    assert finished.wait(10.0)
    assert order == ["started", "pipeline"]


def test_a_failing_import_is_reported_in_the_status_the_gui_reads(initialized, monkeypatch):
    # The task dies on a background thread, so the exception itself never reaches the user. The status
    # line is the only place they learn the import did not happen.
    def explode(status_cb, out, *ins):
        raise RuntimeError("the embedder went away")

    task_env = run_one_task(explode, monkeypatch)
    assert task_env.result_code is importer.result_errored
    assert isinstance(task_env.exc, RuntimeError)
    assert "the embedder went away" in unbox(importer.status_box)


def test_a_finished_import_says_so_and_says_how_to_start_another(initialized, monkeypatch):
    run_one_task(lambda status_cb, out, *ins: None, monkeypatch)
    assert "complete" in unbox(importer.status_box)


def test_a_finished_task_leaves_the_progress_counter_at_the_start(initialized, monkeypatch):
    def advance(status_cb, out, *ins):
        importer.progress.tock()
        importer.progress.tock()

    monkeypatch.setattr(importer, "has_task", importer.has_task)  # keep the real one; the fixture has a task
    run_one_task(advance, monkeypatch)
    # With no task running the counter reports `None` rather than a number, so the reset is checked on
    # the underlying macrostep count -- which is what the *next* run would otherwise inherit.
    assert importer.progress._macrosteps_done == 0


# ---------------------------------------------------------------------------
# Keyword replies that are not keyword lists


PROSE_REPLY = (
    "Based on the five abstracts provided, here is a structured summary of the research.\n"
    "*   Investigated physiological signal-based prediction.\n"
    "*   Applied standard machine learning algorithms and integrated explainability techniques.\n")


def test_parse_keyword_list_separates_declining_from_malformed():
    # Three outcomes, and the middle one is the reason this function exists. A model that declines has
    # answered the question; a model that returns prose has not, and only the second is worth retrying.
    assert importer._parse_keyword_list("Alpha, Beta Gamma, Delta") == ["Alpha", "Beta Gamma", "Delta"]
    assert importer._parse_keyword_list("  Keyword Extraction Failed  ") == []
    assert importer._parse_keyword_list(PROSE_REPLY) is None, \
        "a bulleted prose summary is not a keyword list"
    assert importer._parse_keyword_list(", ".join(f"kw{i}" for i in range(40))) is None, \
        "forty keywords is not an answer to a request for six"
    assert importer._parse_keyword_list("") is None


def test_keyword_extraction_retries_a_malformed_reply(monkeypatch, caplog):
    # The failure is occasional rather than systematic -- one cluster in 338 across five corpora -- so a
    # second ask is worth one request against a run that takes minutes either way.
    replies = iter([PROSE_REPLY, "Knowledge Distillation, Model Compression"])

    def fake_turn(settings, prompt, **kwargs):
        return FakeRecord(next(replies))

    keywords = run_keyword_extraction(monkeypatch, fake_turn, caplog)
    assert keywords == [["Knowledge Distillation", "Model Compression"]], \
        "the retry's answer should be used, not the prose"
    assert any("not a keyword list" in r.message for r in caplog.records)


def test_keyword_extraction_gives_up_and_logs_an_error(monkeypatch, caplog):
    # Out of attempts, the cluster is a hole in the map and somebody should see it in the log -- an
    # error, where every other cluster in the run got a real label.
    def always_prose(settings, prompt, **kwargs):
        return FakeRecord(PROSE_REPLY)

    keywords = run_keyword_extraction(monkeypatch, always_prose, caplog)
    assert keywords == [["<unknown topic>"]]
    assert any(r.levelname == "ERROR" and "Giving up" in r.message for r in caplog.records)


def test_keyword_extraction_does_not_retry_a_decline(monkeypatch, caplog):
    # Negative control for the two above: declining is a valid answer, so it must be taken at face value
    # rather than retried. Were it retried, the counter below would exceed one and this would fail --
    # which is what separates "handles malformed replies" from "asks repeatedly whenever it dislikes an
    # answer", and the second would pester the model into inventing a theme it had just said was absent.
    calls = []

    def declines(settings, prompt, **kwargs):
        calls.append(prompt)
        return FakeRecord("keyword extraction failed")

    keywords = run_keyword_extraction(monkeypatch, declines, caplog)
    assert keywords == [["<unknown topic>"]]
    assert len(calls) == 1, f"a decline should be accepted, not retried; got {len(calls)} calls"


def run_keyword_extraction(monkeypatch, fake_turn, caplog):
    """Drive `_collect_cluster_keywords`'s LLM branch over one two-record cluster."""
    import numpy as np
    from unpythonic.env import env

    vis_data = [env(title=f"paper {i}", abstract="", cluster_id=0, cluster_probability=1.0)
                for i in range(2)]
    all_vectors = np.array([[1.0, 0.0]] * 2)

    monkeypatch.setattr(visualizer_config, "clusters_keyword_method", "llm")
    monkeypatch.setattr(importer, "agent", type("_FakeAgent", (), {"turn": staticmethod(fake_turn)}), raising=False)
    monkeypatch.setattr(importer, "llm_settings", object(), raising=False)
    monkeypatch.setattr(importer, "llmclient",
                        type("_FakeLLMClient", (),
                             {"make_console_progress_handler": staticmethod(lambda _: None)}),
                        raising=False)
    monkeypatch.setattr(importer, "_canonicalize_cluster_keywords", lambda keywords: keywords)

    with caplog.at_level("WARNING"):
        return importer._collect_cluster_keywords(vis_data, 1, {}, all_vectors)


# ---------------------------------------------------------------------------
# Clustering
#
# These need scikit-learn but no model and no server, so they are the cheap half of the `ml` group. The
# vectors are synthetic and well separated on purpose: what is asserted is the shape of the answer and the
# alignment between the answer and the data, not HDBSCAN's judgement, which is the library's business.


def two_directions(n_per_cluster=40, dim=8, spread=0.05, seed=42):
    """Two tight groups of unit vectors pointing along different axes, as one `[2 * n, dim]` array.

    The high-dimensional pass compares by cosine, so what has to differ between the groups is *direction*.
    An isotropic blob around the origin has none — it comes back as no clusters at all, and the step
    raises rather than returning an empty answer. This is also the shape the step really sees: the
    embedder's output lives on the unit hypersphere.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    groups = []
    for axis in np.eye(dim)[:2]:
        vectors = axis + rng.normal(0.0, spread, size=(n_per_cluster, dim))
        groups.append(vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
    return np.concatenate(groups, axis=0)


def two_blobs(n_per_blob=40, dim=2, separation=10.0, seed=42):
    """Two well-separated Gaussian blobs, as one `[2 * n_per_blob, dim]` array.

    For the 2D pass, which compares by euclidean distance — unlike `two_directions` above.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    first = rng.normal(0.0, 1.0, size=(n_per_blob, dim))
    second = rng.normal(0.0, 1.0, size=(n_per_blob, dim)) + separation
    return np.concatenate([first, second], axis=0)


def test_the_highdim_pass_finds_the_clusters_that_are_there():
    _, n_clusters = importer._cluster_highdim_semantic_vectors(two_directions())
    assert n_clusters >= 2, "two groups pointing different ways should not come back as one cluster"


def test_the_highdim_pass_returns_a_bounded_sample_per_cluster():
    # What comes back is training data for the dimension reduction, so it is stratified rather than
    # complete: up to `_MAX_REPRESENTATIVES_PER_CLUSTER` from each detected cluster. `max_n` does not
    # bound it — that bounds how many vectors are *fitted*.
    cap = importer._MAX_REPRESENTATIVES_PER_CLUSTER
    n_per_cluster = 40  # a literal, deliberately: sizing the fixture from `cap` would grow it with the
    assert cap < n_per_cluster, ("the cap now exceeds what this fixture supplies per cluster, so it "
                                 "cannot bite and this test asserts nothing -- raise `n_per_cluster`")
    all_vectors = two_directions(n_per_cluster=n_per_cluster)
    unique_vs, n_clusters = importer._cluster_highdim_semantic_vectors(all_vectors, max_n=10000)

    assert unique_vs.shape[1] == all_vectors.shape[1], "the vectors keep their dimensionality"
    assert len(unique_vs) <= cap * n_clusters
    # Every cluster holds more than the cap, so a run that returned everything is the thing to rule out.
    assert len(unique_vs) < len(all_vectors)


def test_the_highdim_pass_fits_on_a_sample_when_the_dataset_is_large(caplog):
    # HDBSCAN runs out of memory somewhere around 50k vectors, so above `max_n` it sees a random subset.
    # The representative points still come from that subset, so `max_n` bounds them too.
    all_vectors = two_directions(n_per_cluster=100)
    with caplog.at_level("INFO"):
        unique_vs, _ = importer._cluster_highdim_semantic_vectors(all_vectors, max_n=120)
    assert any("Dataset is large" in record.message for record in caplog.records), \
        "the sampling branch was not taken, so this fixture says nothing about the cap"
    assert len(unique_vs) <= 120


def test_the_sample_a_large_dataset_is_fitted_on_holds_distinct_entries():
    # The subset is drawn without replacement, so no entry is fitted twice. Drawing independently instead
    # spends part of the sample on duplicates -- roughly a tenth of a 10k-of-50k draw -- and the duplicates
    # reach the representative points, where they weight the dimension reduction's training set toward
    # whichever entries happened to be drawn more than once.
    #
    # Observable from outside because the fixture has no repeated rows of its own: a duplicate row in the
    # output can only have come from a repeated index.
    import numpy as np
    all_vectors = two_directions(n_per_cluster=100)
    unique_vs, _ = importer._cluster_highdim_semantic_vectors(all_vectors, max_n=150)
    assert len(np.unique(unique_vs, axis=0)) == len(unique_vs), "the same entry was picked more than once"


def test_the_highdim_pass_refuses_a_dataset_it_cannot_cluster():
    # Documented behaviour: with no clusters there is nothing to train the dimension reduction on, so the
    # step stops rather than proceeding on an empty training set. A bibliography of a handful of entries
    # is the realistic way to get here -- the clusterer's `min_cluster_size` is 5, so four entries cannot
    # form one however they are arranged. (Scattering points at random does *not* get here: with
    # `min_samples=1` and leaf selection, HDBSCAN finds structure in noise.)
    with pytest.raises(RuntimeError):
        importer._cluster_highdim_semantic_vectors(two_directions(n_per_cluster=2))


def test_the_2d_pass_tags_each_entry_with_its_own_cluster():
    # The tagging walks `vis_data` and `labels` by the same index, so an off-by-one here would mislabel
    # every point in the dataset while looking perfectly healthy -- every entry still gets *a* cluster.
    from unpythonic.env import env

    lowdim = two_blobs(n_per_blob=40, dim=2)
    entries = [env(title=f"paper {i}", abstract="") for i in range(len(lowdim))]
    input_data = env(parsed_data_by_filename={"a.bib": entries})

    vis_data, labels, n_vis_clusters, n_vis_outliers = importer._cluster_lowdim_data(input_data, lowdim)

    assert vis_data == entries, "the concatenation should preserve order; the labels are indexed by it"
    assert len(labels) == len(lowdim)
    for entry, label in zip(vis_data, labels):
        assert entry.cluster_id == label
        assert 0.0 <= entry.cluster_probability <= 1.0
    # The two blobs are far apart, so the tags must actually separate them -- otherwise every assertion
    # above is satisfied by a run that put everything in one cluster.
    assert len({int(label) for label in labels} - {-1}) >= 2
    assert n_vis_clusters >= 2


def test_the_2d_pass_counts_outliers_as_the_points_it_left_unclustered():
    import numpy as np

    lowdim = two_blobs(n_per_blob=40, dim=2)
    entries_env = _entries_env(len(lowdim))
    _, labels, _, n_vis_outliers = importer._cluster_lowdim_data(entries_env, lowdim)
    assert n_vis_outliers == int(np.sum(labels == -1))


def test_the_2d_pass_concatenates_across_input_files_in_order():
    # A multi-file import is the normal case, and the 2D coordinates arrive as one array for the whole
    # dataset -- so the concatenation order here is what aligns entries to their points.
    from unpythonic.env import env

    lowdim = two_blobs(n_per_blob=40, dim=2)
    first = [env(title=f"first {i}", abstract="") for i in range(len(lowdim) // 2)]
    second = [env(title=f"second {i}", abstract="") for i in range(len(lowdim) // 2)]
    input_data = env(parsed_data_by_filename={"a.bib": first, "b.bib": second})

    vis_data, _, _, _ = importer._cluster_lowdim_data(input_data, lowdim)
    assert vis_data == first + second


def _entries_env(n):
    """An `input_data`-shaped namespace holding `n` blank entries in one file."""
    from unpythonic.env import env
    return env(parsed_data_by_filename={"a.bib": [env(title=f"paper {i}", abstract="") for i in range(n)]})


# ---------------------------------------------------------------------------
# The steps that need a model
#
# `mayberemote` with `allow_local=True` means these run whether or not raven-server is up: with a server
# they go over HTTP, without one they load the model in-process. Either way they need `raven.client.api`
# initialized first, and either way they are slow -- which is what the `ml` marker on this file is for.


@pytest.fixture(scope="module")
def initialized_api():
    """Initialize the client API, as every app does during its bootup, before any `mayberemote` call."""
    from raven.client import api
    from raven.client import config as client_config
    api.initialize(raven_server_url=client_config.raven_server_url,
                   raven_api_key_file=client_config.raven_api_key_file)


@pytest.fixture
def two_entry_input_data(tmp_path):
    """`_parse_input_files` output for two real entries, with its caches under `tmp_path`.

    The caches matter: both steps below write one beside each input file, keyed on the file's mtime. A
    fixture pointing at a repo path would leave those lying around and, worse, would let one test's cache
    answer another test's question.
    """
    from unpythonic.env import env
    path = tmp_path / "two_records.bib"
    path.write_text(TWO_RECORDS, encoding="utf-8")
    entries = [env(author="Alpha, Anna", bibtex_author="Alpha, Anna", year="2024",
                   title="Laser ablation of titanium alloys",
                   abstract="We study laser ablation thresholds for titanium alloys in Helsinki."),
               env(author="Beta, Bob", bibtex_author="Beta, Bob", year="2024",
                   title="A paper with no abstract at all",
                   abstract=None)]
    return env(parsed_data_by_filename={str(path): entries},
               n_entries_total=len(entries),
               resolved_filenames=[str(path)])


def test_embedding_vectors_land_on_the_unit_hypersphere(initialized_api, two_entry_input_data):
    # The module documents this as a property callers may rely on -- it is what makes cosine similarity
    # the right comparison downstream, and what the high-dimensional clustering assumes.
    import numpy as np
    all_vectors = importer._get_highdim_semantic_vectors(two_entry_input_data)
    assert all_vectors.shape[0] == 2
    norms = np.linalg.norm(all_vectors, axis=1)
    # The tolerance is fp16-sized on purpose: the configured embedding device may use half precision, and
    # measured norms then sit a few times 1e-4 off unity. Tightening this to 1e-4 fails on a correct run.
    assert np.allclose(norms, 1.0, atol=1e-3), f"vectors are not unit-length: {norms}"


def test_a_second_embedding_pass_reads_the_cache_rather_than_the_model(initialized_api, two_entry_input_data, caplog):
    # Caches are per input file so that adding a file to a dataset does not re-embed the ones already
    # done. On a real corpus this is the difference between seconds and an hour.
    first = importer._get_highdim_semantic_vectors(two_entry_input_data)
    with caplog.at_level("INFO"):
        second = importer._get_highdim_semantic_vectors(two_entry_input_data)

    import numpy as np
    assert np.allclose(first, second)
    assert not any("Computing embeddings" in record.message for record in caplog.records), \
        "the second pass recomputed the embeddings instead of reading its cache"


def test_keyword_extraction_fills_in_the_three_fields_the_dataset_needs(initialized_api, two_entry_input_data):
    all_keywords = importer._extract_keywords(two_entry_input_data)
    entries = two_entry_input_data.parsed_data_by_filename[two_entry_input_data.resolved_filenames[0]]

    assert isinstance(all_keywords, dict) and all_keywords, "the dataset-wide keyword counts should be populated"
    for entry in entries:
        assert isinstance(entry.keywords, dict)
        assert isinstance(entry.entities, set)
        assert isinstance(entry.vis_keywords, list)

    # The entry with an abstract has more to say than the one without; without this the assertions above
    # are satisfied by an extractor that returned empty containers for both.
    with_abstract, without_abstract = entries
    assert with_abstract.keywords, f"nothing was extracted from an abstract-bearing entry: {with_abstract.title}"
    assert len(with_abstract.keywords) > len(without_abstract.keywords)


def test_the_dataset_wide_keyword_counts_are_sorted_by_frequency(initialized_api, two_entry_input_data):
    all_keywords = importer._extract_keywords(two_entry_input_data)
    counts = list(all_keywords.values())
    assert counts == sorted(counts, reverse=True)


def test_the_visualized_keywords_per_entry_are_capped(initialized_api, two_entry_input_data):
    # `vis_keywords` is what reaches the GUI, so its length is a display decision rather than an analysis
    # one. Named entities are added on top of the cap, so the bound is on the frequency-ranked part.
    max_vis_kw = 2
    importer._extract_keywords(two_entry_input_data, max_vis_kw=max_vis_kw)
    entries = two_entry_input_data.parsed_data_by_filename[two_entry_input_data.resolved_filenames[0]]
    for entry in entries:
        frequency_ranked = [kw for kw in entry.vis_keywords if kw not in entry.entities]
        assert len(frequency_ranked) <= max_vis_kw, f"{entry.title}: {entry.vis_keywords}"


# ---------------------------------------------------------------------------
# Summarization, against a live LLM backend
#
# What a mock cannot say: whether a real model, given Raven's own prompt, produces something the importer
# stores as a summary rather than tripping its failure sentinel. Shape only -- the model differs between
# machines and over time, so asserting on wording would be a test of the model.


@pytest.fixture
def live_llm(request, monkeypatch):
    """Give `importer` a real LLM connection, whatever its config said when it was imported.

    `importer` sets `llm_settings` at *import* time, and only when the config asks for cluster keywords or
    summaries -- so a test that waited for that would skip on any machine configured without them, which is
    the common case and would make it a test that never runs. It is built here instead, exactly as the
    module builds its own, and injected the way the fake-LLM tests inject theirs.

    Only the *connection* is built here. The LLM call itself belongs to `_summarize`, which makes it
    through `agent.turn` -- so this fixture must not make one of its own, or the test would be asserting
    against its own request rather than against the one the importer sends. Same gate as
    `raven/librarian/tests/test_live_backend.py`, and for the same reason: a skip has to name what was
    wrong, since a skip and a pass look alike.
    """
    from raven.librarian import agent
    from raven.librarian import config as librarian_config
    from raven.librarian import llmclient

    backend_url = request.config.getoption("--backend-url") or librarian_config.llm_backend_url
    if not llmclient.test_connection(backend_url, quiet=True):
        pytest.skip(f"no LLM backend answering at {backend_url} -- start one, or point this somewhere "
                    f"else with: pytest --backend-url http://host:port")
    settings = llmclient.setup(backend_url, quiet=True)
    status = llmclient.backend_status(settings)
    if status is not llmclient.backend_ready:
        pytest.skip(f"backend at {backend_url} is not ready: "
                    f"{llmclient.describe_backend_status(status, backend_url)}")

    monkeypatch.setattr(importer, "agent", agent, raising=False)
    monkeypatch.setattr(importer, "llmclient", llmclient, raising=False)
    monkeypatch.setattr(importer, "llm_settings", settings, raising=False)
    return settings


@pytest.mark.llm
def test_an_abstract_is_summarized_and_a_missing_one_is_not(live_llm, two_entry_input_data, monkeypatch):
    monkeypatch.setattr(visualizer_config, "summarize", True)
    importer._summarize(two_entry_input_data)

    with_abstract, without_abstract = two_entry_input_data.parsed_data_by_filename[two_entry_input_data.resolved_filenames[0]]
    assert without_abstract.summary is None, "an entry with no abstract has nothing to summarize"
    # `None` here too would mean the model answered with the failure sentinel, which is the outcome worth
    # knowing about: the prompt no longer works on whatever model is loaded.
    assert isinstance(with_abstract.summary, str) and with_abstract.summary.strip(), \
        "the model returned no usable summary for an abstract; the prompt may no longer suit it"
