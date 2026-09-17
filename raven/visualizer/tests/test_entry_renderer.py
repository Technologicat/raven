"""Unit tests for raven.visualizer.entry_renderer.

The rendering vocabulary the annotation tooltip and the info panel share, which is what makes pinning it
worth doing before either grows: a change to how items are grouped, ordered or truncated is a change to
both consumers at once, and neither of them is under test.

Nothing here needs DPG. `entry_renderer` deliberately stops at the data layer, so the whole module is
plain functions over a dataset -- which the fixtures below stand up as `env` records shaped the way
`plotter.load_dataset` builds them.
"""

import pytest

from unpythonic.env import env

from raven.visualizer import entry_renderer
from raven.visualizer.app_state import app_state


def make_entry(title, cluster_id, **fields):
    """One record of `dataset.sorted_entries`, carrying the fields these tests sort or group by."""
    return env(title=title, cluster_id=cluster_id, **fields)


def make_dataset(entries, *, keywords_available=True, vis_keywords_by_cluster=None):
    """The slice of a Visualizer dataset that `entry_renderer` reads.

    `vis_keywords_by_cluster` is a list indexed by cluster ID, as the importer writes it -- which is
    what lets the misc-group test below say something: on a list, index -1 is the *last cluster's*
    keywords rather than a missing key, so a special case that failed to fire would return keywords
    instead of raising.
    """
    return env(sorted_entries=entries,
               file_content=env(keywords_available=keywords_available,
                                vis_keywords_by_cluster=vis_keywords_by_cluster or []))


@pytest.fixture
def two_clusters():
    """Six items in two clusters, deliberately neither alphabetical nor grouped in index order."""
    return make_dataset([make_entry("Zebras", 0),          # 0
                         make_entry("apples", 1),          # 1
                         make_entry("  Bananas ", 1),      # 2 -- leading/trailing space, capitalized
                         make_entry("mangoes", 0),         # 3
                         make_entry("Cherries", 1),        # 4
                         make_entry("Aardvarks", 0)],      # 5
                        vis_keywords_by_cluster=[["animals", "fauna"],
                                                 ["fruit"]])


# --------------------------------------------------------------------------------
# Grouping and ordering

def test_entries_are_grouped_by_cluster(two_clusters):
    entries_by_cluster, _ = entry_renderer.get_entries_for_selection([0, 1, 2, 3, 4, 5], dataset=two_clusters)
    assert set(entries_by_cluster.keys()) == {0, 1}
    assert {data_idx for data_idx, _ in entries_by_cluster[0]} == {0, 3, 5}
    assert {data_idx for data_idx, _ in entries_by_cluster[1]} == {1, 2, 4}


def test_only_the_selection_is_gathered(two_clusters):
    # Negative control for the test above: it selects everything, so it cannot tell "gathered the
    # selection" from "gathered the dataset".
    entries_by_cluster, _ = entry_renderer.get_entries_for_selection([0, 1], dataset=two_clusters)
    assert set(entries_by_cluster.keys()) == {0, 1}
    assert [data_idx for data_idx, _ in entries_by_cluster[0]] == [0]
    assert [data_idx for data_idx, _ in entries_by_cluster[1]] == [1]


def test_entries_are_alphabetized_within_each_cluster(two_clusters):
    # Case-insensitively and ignoring surrounding whitespace, which is the point: a title arriving from
    # BibTeX as "  Bananas " must sort where a reader glancing down the list expects to find it, not
    # ahead of everything because a space sorts low.
    entries_by_cluster, _ = entry_renderer.get_entries_for_selection([0, 1, 2, 3, 4, 5], dataset=two_clusters)
    assert [entry.title for _, entry in entries_by_cluster[0]] == ["Aardvarks", "mangoes", "Zebras"]
    assert [entry.title for _, entry in entries_by_cluster[1]] == ["apples", "  Bananas ", "Cherries"]


def test_sort_field_selects_which_field_orders_the_cluster():
    dataset = make_dataset([make_entry("Zebras", 0, year="1999"),
                            make_entry("Aardvarks", 0, year="2024")])
    entries_by_cluster, _ = entry_renderer.get_entries_for_selection([0, 1], sort_field="year", dataset=dataset)
    # Sorted by year, the alphabetically-last title comes first -- so this says the field was honored
    # rather than that some ordering happened.
    assert [entry.title for _, entry in entries_by_cluster[0]] == ["Zebras", "Aardvarks"]


def test_an_empty_selection_gathers_nothing_and_does_not_divide_by_zero(two_clusters):
    # `max_n` is shared out over the clusters present, so an empty selection is the case where that
    # division has no denominator.
    entries_by_cluster, _ = entry_renderer.get_entries_for_selection([], max_n=10, dataset=two_clusters)
    assert not entries_by_cluster


def test_the_dataset_defaults_to_the_live_one(two_clusters, monkeypatch):
    # Callers on the main thread omit `dataset`; only background workers capture a snapshot to pass in.
    monkeypatch.setattr(app_state, "dataset", two_clusters, raising=False)
    entries_by_cluster, _ = entry_renderer.get_entries_for_selection([0])
    assert [entry.title for _, entry in entries_by_cluster[0]] == ["Zebras"]


# --------------------------------------------------------------------------------
# Cluster annotations

def test_a_cluster_is_annotated_with_its_number_and_keywords(two_clusters):
    _, formatter = entry_renderer.get_entries_for_selection([0, 1], max_n=10, dataset=two_clusters)
    cluster_title, cluster_keywords, _, _ = formatter(0)
    assert cluster_title == "#0"
    assert cluster_keywords == "[animals, fauna]\n"


def test_the_misc_group_is_named_rather_than_numbered():
    # Cluster -1 collects the outliers, which have no keywords in common by construction. The dataset
    # here *does* have keywords available and a list filed under -1, so this says the misc group is
    # special-cased rather than that there was nothing to show.
    dataset = make_dataset([make_entry("Odd one out", -1)],
                           vis_keywords_by_cluster=[["should", "not", "appear"]])
    _, formatter = entry_renderer.get_entries_for_selection([0], max_n=10, dataset=dataset)
    cluster_title, cluster_keywords, _, _ = formatter(-1)
    assert cluster_title == "Misc"
    assert cluster_keywords == ""


def test_a_dataset_imported_without_keyword_extraction_still_annotates_its_clusters():
    # Keyword extraction is a config switch at import time, so a dataset can legitimately have none.
    dataset = make_dataset([make_entry("Zebras", 0)], keywords_available=False)
    _, formatter = entry_renderer.get_entries_for_selection([0], max_n=10, dataset=dataset)
    cluster_title, cluster_keywords, _, _ = formatter(0)
    assert cluster_title == "#0"
    assert cluster_keywords == ""


# --------------------------------------------------------------------------------
# The `max_n` budget

def cluster_of(n):
    """A dataset of `n` items in one cluster, titled so that alphabetical order is index order."""
    return make_dataset([make_entry(f"Item {i:02d}", 0) for i in range(n)],
                        keywords_available=False)  # these tests are about the budget, not the annotation


def test_a_selection_that_fits_the_budget_is_shown_whole():
    dataset = cluster_of(3)
    _, formatter = entry_renderer.get_entries_for_selection(range(3), max_n=5, dataset=dataset)
    _, _, cluster_content, more = formatter(0)
    assert len(cluster_content) == 3
    assert more == ""


def test_going_one_or_two_over_the_budget_is_tolerated_rather_than_cut():
    # Cutting two entries to save two lines reads as pedantic, and the "...N more..." line costs one of
    # them back. So the limit is soft by up to two entries.
    dataset = cluster_of(5)
    _, formatter = entry_renderer.get_entries_for_selection(range(5), max_n=3, dataset=dataset)
    _, _, cluster_content, more = formatter(0)
    assert len(cluster_content) == 5, "two entries over the limit should still be shown in full"
    assert more == ""


def test_going_three_or_more_over_the_budget_cuts_at_the_limit():
    # One more entry than the case above, and the tolerance stops applying: cut, and say how many were
    # hidden. The pair is what pins where the threshold sits.
    dataset = cluster_of(6)
    _, formatter = entry_renderer.get_entries_for_selection(range(6), max_n=3, dataset=dataset)
    _, _, cluster_content, more = formatter(0)
    assert [entry.title for _, entry in cluster_content] == ["Item 00", "Item 01", "Item 02"]
    assert more == "[...3 more entries in #0...]"


def test_every_cluster_gets_at_least_one_entry_however_small_the_budget():
    # The budget is shared out per cluster, rounding up, so a selection spanning more clusters than the
    # budget has entries still shows what it found in each -- a cluster silently missing from the panel
    # would read as "no items here", which is a different statement from "too many to list".
    dataset = make_dataset([make_entry(f"Item {i:02d}", cluster_id)
                            for cluster_id in range(3)
                            for i in range(4)],
                           keywords_available=False)
    entries_by_cluster, formatter = entry_renderer.get_entries_for_selection(range(12), max_n=2, dataset=dataset)
    assert set(entries_by_cluster.keys()) == {0, 1, 2}
    for cluster_id in (0, 1, 2):
        _, _, cluster_content, more = formatter(cluster_id)
        assert len(cluster_content) == 1
        assert more == f"[...3 more entries in #{cluster_id}...]"


def test_no_budget_means_no_limit():
    # `max_n=None` is the documented default, and the only caller of it is whoever wants the whole
    # selection. It used to raise `UnboundLocalError` from the formatter, the "how many more" text
    # being computed only on the truncating path.
    dataset = cluster_of(50)
    _, formatter = entry_renderer.get_entries_for_selection(range(50), dataset=dataset)
    _, _, cluster_content, more = formatter(0)
    assert len(cluster_content) == 50
    assert more == ""


# --------------------------------------------------------------------------------
# Cluster ID ordering

def test_cluster_ids_sort_ascending_with_misc_last():
    assert entry_renderer.order_cluster_ids([3, -1, 0, 12]) == [0, 3, 12, -1]


def test_cluster_ids_are_deduplicated():
    assert entry_renderer.order_cluster_ids([2, 0, 2, 0]) == [0, 2]


def test_ordering_cluster_ids_without_a_misc_group_leaves_the_tail_alone():
    # Negative control: the rule moves -1 specifically, rather than rotating the list.
    assert entry_renderer.order_cluster_ids([2, 0, 1]) == [0, 1, 2]


def test_an_empty_cluster_id_list_stays_empty():
    assert entry_renderer.order_cluster_ids([]) == []


def test_the_misc_group_alone_is_still_the_misc_group():
    assert entry_renderer.order_cluster_ids([-1]) == [-1]
