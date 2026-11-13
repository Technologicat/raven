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


class TestMiscMode:
    def test_misc_is_shown_as_the_mode_says_and_everything_else_in_full(self):
        entries = entries_of(0, -1, 1, -1)
        assert displaystate.apply_misc_mode(entries, displaystate.GRAYED)
        assert [e.visibility for e in entries] == ["shown", "grayed", "shown", "grayed"]

    def test_a_cluster_moved_to_misc_by_purity_goes_with_misc(self):
        entries = entries_of(0, 1, -1)
        displaystate.relabel_by_purity(entries, [0.9, 0.1], 0.5)
        displaystate.apply_misc_mode(entries, displaystate.HIDDEN)
        assert [e.visibility for e in entries] == ["shown", "hidden", "hidden"]

    def test_showing_misc_again_restores_it(self):
        entries = entries_of(0, -1)
        displaystate.apply_misc_mode(entries, displaystate.HIDDEN)
        assert displaystate.apply_misc_mode(entries, displaystate.SHOWN)
        assert [e.visibility for e in entries] == ["shown", "shown"]
        assert not displaystate.apply_misc_mode(entries, displaystate.SHOWN), "nothing changed, and it said so"

    def test_an_unknown_mode_is_refused_by_name(self):
        import pytest
        with pytest.raises(ValueError, match="sideways"):
            displaystate.apply_misc_mode(entries_of(0), "sideways")


class TestSummary:
    def test_with_everything_shown_there_is_nothing_to_say(self):
        assert displaystate.summary(entries_of(0, -1)) is None

    def test_grayed_records_are_still_on_screen_so_the_count_is_of_those_in_focus(self):
        entries = entries_of(*([0] * 1422 + [-1] * 610))
        displaystate.apply_misc_mode(entries, displaystate.GRAYED)
        assert displaystate.summary(entries) == "1,422 of 2,032 in focus"

    def test_hidden_records_are_not_shown_so_the_count_is_of_those_on_screen(self):
        entries = entries_of(*([0] * 1422 + [-1] * 610))
        displaystate.apply_misc_mode(entries, displaystate.HIDDEN)
        assert displaystate.summary(entries) == "Showing 1,422 of 2,032"

    def test_grayed_and_hidden_together_give_both_counts(self):
        entries = entries_of(0, 0, -1, 1)
        entries[2].visibility = displaystate.HIDDEN
        entries[3].visibility = displaystate.GRAYED
        assert displaystate.summary(entries) == "Showing 3 of 4, 2 in focus"


class TestPointsByLabel:
    def test_records_are_grouped_by_the_label_shown_keeping_their_indices(self):
        entries = entries_of(0, 1, 0, 1, -1)
        displaystate.relabel_by_purity(entries, [0.9, 0.1], 0.5)
        assert displaystate.points_by_label(entries) == {0: [0, 2], -1: [1, 3, 4]}

    def test_grayed_and_hidden_records_are_in_no_colour_series(self):
        entries = entries_of(0, -1, 0, -1)
        displaystate.apply_misc_mode(entries, displaystate.GRAYED)
        assert displaystate.points_by_label(entries) == {0: [0, 2]}
        assert displaystate.points_with_visibility(entries, displaystate.GRAYED) == [1, 3]


class TestTheInfoPanelAndTooltipGroupByTheLabelShown:
    def test_a_demoted_clusters_records_are_listed_under_misc(self):
        entries = entries_of(0, 1, 1)
        dataset = env(sorted_entries=entries,
                      file_content=env(keywords_available=False, vis_keywords_by_cluster=[]))
        displaystate.relabel_by_purity(entries, [0.9, 0.1], 0.5)
        entries_by_cluster, _ = entry_renderer.get_entries_for_selection(range(3), dataset=dataset)
        assert sorted(entries_by_cluster) == [-1, 0]
        assert [data_idx for data_idx, _ in entries_by_cluster[-1]] == [1, 2]
