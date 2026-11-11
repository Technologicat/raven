"""Tests for `raven.visualizer.displaystate`: the label each record is shown with, and the purity cutoff."""

from unpythonic.env import env

from raven.visualizer import displaystate
from raven.visualizer import entry_renderer


def entries_of(*cluster_ids):
    entries = [env(title=f"paper {i}", cluster_id=cid) for i, cid in enumerate(cluster_ids)]
    displaystate.init(entries)
    return entries


class TestInit:
    def test_every_record_starts_under_the_importers_label(self):
        entries = entries_of(0, 0, 1, -1)
        assert [e.current_label_id for e in entries] == [0, 0, 1, -1]


class TestRelabelByPurity:
    PURITIES = [0.62, 0.30, None]  # cluster 2's purity is undefined

    def test_clusters_below_the_cutoff_are_shown_as_misc(self):
        entries = entries_of(0, 1, 1, 2, -1)
        assert displaystate.relabel_by_purity(entries, self.PURITIES, 0.5)
        assert [e.current_label_id for e in entries] == [0, -1, -1, 2, -1]

    def test_the_importers_label_is_kept_underneath(self):
        entries = entries_of(0, 1)
        displaystate.relabel_by_purity(entries, self.PURITIES, 0.5)
        assert [e.cluster_id for e in entries] == [0, 1]

    def test_lowering_the_cutoff_brings_a_cluster_back(self):
        entries = entries_of(0, 1, 1)
        displaystate.relabel_by_purity(entries, self.PURITIES, 0.5)
        assert displaystate.relabel_by_purity(entries, self.PURITIES, 0.0)
        assert [e.current_label_id for e in entries] == [0, 1, 1]

    def test_a_cutoff_that_changes_nothing_says_so(self):
        # The caller skips the redraw on False; the control is the first relabel, which does change something.
        entries = entries_of(0, 1)
        assert displaystate.relabel_by_purity(entries, self.PURITIES, 0.5)
        assert not displaystate.relabel_by_purity(entries, self.PURITIES, 0.4)

    def test_a_cluster_with_undefined_purity_keeps_its_label_at_any_cutoff(self):
        entries = entries_of(2)
        displaystate.relabel_by_purity(entries, self.PURITIES, 1.0)
        assert entries[0].current_label_id == 2

    def test_a_dataset_without_purities_is_left_alone(self):
        entries = entries_of(0, 1)
        assert not displaystate.relabel_by_purity(entries, None, 1.0)
        assert [e.current_label_id for e in entries] == [0, 1]


class TestPointsByLabel:
    def test_records_are_grouped_by_the_label_shown_keeping_their_indices(self):
        entries = entries_of(0, 1, 0, 1, -1)
        displaystate.relabel_by_purity(entries, [0.9, 0.1], 0.5)
        assert displaystate.points_by_label(entries) == {0: [0, 2], -1: [1, 3, 4]}


class TestTheInfoPanelAndTooltipGroupByTheLabelShown:
    def test_a_demoted_clusters_records_are_listed_under_misc(self):
        entries = entries_of(0, 1, 1)
        dataset = env(sorted_entries=entries,
                      file_content=env(keywords_available=False, vis_keywords_by_cluster=[]))
        displaystate.relabel_by_purity(entries, [0.9, 0.1], 0.5)
        entries_by_cluster, _ = entry_renderer.get_entries_for_selection(range(3), dataset=dataset)
        assert sorted(entries_by_cluster) == [-1, 0]
        assert [data_idx for data_idx, _ in entries_by_cluster[-1]] == [1, 2]
