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
        assert displaystate.apply_filters(entries, misc_mode=displaystate.GRAYED)
        assert [e.visibility for e in entries] == ["shown", "grayed", "shown", "grayed"]

    def test_a_cluster_moved_to_misc_by_purity_goes_with_misc(self):
        entries = entries_of(0, 1, -1)
        displaystate.relabel_by_purity(entries, [0.9, 0.1], 0.5)
        displaystate.apply_filters(entries, misc_mode=displaystate.HIDDEN)
        assert [e.visibility for e in entries] == ["shown", "hidden", "hidden"]

    def test_showing_misc_again_restores_it(self):
        entries = entries_of(0, -1)
        displaystate.apply_filters(entries, misc_mode=displaystate.HIDDEN)
        assert displaystate.apply_filters(entries, misc_mode=displaystate.SHOWN)
        assert [e.visibility for e in entries] == ["shown", "shown"]
        assert not displaystate.apply_filters(entries, misc_mode=displaystate.SHOWN), "nothing changed, and it said so"

    def test_an_unknown_mode_is_refused_by_name(self):
        import pytest
        with pytest.raises(ValueError, match="sideways"):
            displaystate.apply_filters(entries_of(0), misc_mode="sideways")


def entries_with_years(*years, cluster_id=0):
    """Entries in one cluster, with `year` as given; `None` gives the importer's placeholder for a missing year."""
    entries = [env(title=f"paper {i}", cluster_id=cluster_id, year=year if year is not None else "[Year not specified]")
               for i, year in enumerate(years)]
    displaystate.init(entries)
    return entries


class TestYearFilter:
    def test_records_outside_the_range_are_shown_as_the_mode_says_and_the_ends_are_inside(self):
        entries = entries_with_years("2019", "2022", "2024", "2026")
        assert displaystate.apply_filters(entries, maybe_year_range=(2022, 2024), year_mode=displaystate.GRAYED)
        assert [e.visibility for e in entries] == ["grayed", "shown", "shown", "grayed"]

    def test_a_record_with_no_year_follows_the_records_outside(self):
        entries = entries_with_years("2023", None, "in press")
        displaystate.apply_filters(entries, maybe_year_range=(2022, 2024), year_mode=displaystate.HIDDEN)
        assert [e.visibility for e in entries] == ["shown", "hidden", "hidden"]

    def test_with_the_filter_off_a_record_with_no_year_is_shown(self):
        entries = entries_with_years("2023", None)
        # The control: with a range, the yearless record does go. Without one it must not.
        displaystate.apply_filters(entries, maybe_year_range=(2022, 2024))
        assert entries[1].visibility == displaystate.GRAYED, "this fixture's yearless record never left the range"
        assert displaystate.apply_filters(entries, maybe_year_range=None)
        assert [e.visibility for e in entries] == ["shown", "shown"]

    def test_the_bounds_are_the_dataset_s_own_and_a_missing_year_is_not_one_of_them(self):
        assert displaystate.year_bounds(entries_with_years("2024", None, "2019", "2022")) == (2019, 2024)
        assert displaystate.year_bounds(entries_with_years(None, "in press")) is None

    def test_an_unknown_year_mode_is_refused_and_shown_is_not_a_year_mode(self):
        import pytest
        with pytest.raises(ValueError, match="year mode"):
            displaystate.apply_filters(entries_with_years("2023"), maybe_year_range=(2022, 2024),
                                       year_mode=displaystate.SHOWN)


class TestSteppingTheYearRange:
    BOUNDS = (2012, 2026)

    def test_a_slide_keeps_the_width(self):
        assert displaystate.step_year_range((2015, 2018), delta=1, bounds=self.BOUNDS) == (2016, 2019)
        assert displaystate.step_year_range((2015, 2018), delta=-1, bounds=self.BOUNDS) == (2014, 2017)

    def test_a_slide_that_would_cross_a_bound_does_not_move(self):
        # Clamping one end instead would narrow the range, which is not what a slide is for.
        assert displaystate.step_year_range((2023, 2026), delta=1, bounds=self.BOUNDS) == (2023, 2026)
        assert displaystate.step_year_range((2012, 2014), delta=-1, bounds=self.BOUNDS) == (2012, 2014)

    def test_one_end_moves_alone_and_stops_at_the_bound_and_at_the_other_end(self):
        step = displaystate.step_year_range
        assert step((2015, 2018), delta=1, bounds=self.BOUNDS, ends="first") == (2016, 2018)
        assert step((2015, 2018), delta=-1, bounds=self.BOUNDS, ends="last") == (2015, 2017)
        assert step((2012, 2018), delta=-1, bounds=self.BOUNDS, ends="first") == (2012, 2018)
        assert step((2015, 2026), delta=1, bounds=self.BOUNDS, ends="last") == (2015, 2026)
        assert step((2018, 2018), delta=1, bounds=self.BOUNDS, ends="first") == (2018, 2018)
        assert step((2018, 2018), delta=-1, bounds=self.BOUNDS, ends="last") == (2018, 2018)

    def test_an_unknown_end_is_refused(self):
        import pytest
        with pytest.raises(ValueError, match="middle"):
            displaystate.step_year_range((2015, 2018), delta=1, bounds=self.BOUNDS, ends="middle")


class TestFiltersCombine:
    def test_the_most_restrictive_verdict_wins(self):
        S, G, H = displaystate.SHOWN, displaystate.GRAYED, displaystate.HIDDEN
        assert displaystate.combine() == S
        assert displaystate.combine(S, S) == S
        assert displaystate.combine(S, G) == displaystate.combine(G, S) == G
        assert displaystate.combine(G, H) == displaystate.combine(H, S) == H

    def test_misc_and_the_year_filter_apply_together(self):
        # Four records, one per corner of (Misc or not) x (in range or not), Misc grayed and years hidden.
        entries = entries_with_years("2023", "2019", "2023", "2019")
        entries[2].cluster_id = entries[3].cluster_id = -1
        displaystate.init(entries)
        displaystate.apply_filters(entries, misc_mode=displaystate.GRAYED,
                                   maybe_year_range=(2022, 2024), year_mode=displaystate.HIDDEN)
        assert [e.visibility for e in entries] == ["shown", "hidden", "grayed", "hidden"]


class TestSummary:
    def test_with_everything_shown_there_is_nothing_to_say(self):
        assert displaystate.summary(entries_of(0, -1)) is None

    def test_grayed_records_are_still_on_screen_so_the_count_is_of_those_in_focus(self):
        entries = entries_of(*([0] * 1422 + [-1] * 610))
        displaystate.apply_filters(entries, misc_mode=displaystate.GRAYED)
        assert displaystate.summary(entries) == "1,422 of 2,032 in focus"

    def test_hidden_records_are_not_shown_so_the_count_is_of_those_on_screen(self):
        entries = entries_of(*([0] * 1422 + [-1] * 610))
        displaystate.apply_filters(entries, misc_mode=displaystate.HIDDEN)
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
        displaystate.apply_filters(entries, misc_mode=displaystate.GRAYED)
        assert displaystate.points_by_label(entries) == {0: [0, 2]}
        assert displaystate.points_with_visibility(entries, displaystate.GRAYED) == [1, 3]


class TestGrouping:
    def years(self):
        entries = entries_of(0, 1, 0, -1)
        for entry, year in zip(entries, ("2021", "2022", "[Year not specified]", "2021")):
            entry.year = year
        return entries

    def test_grouping_by_cluster_is_the_label(self):
        entries = self.years()
        assert [displaystate.group_of(e, displaystate.CLUSTER) for e in entries] == [0, 1, 0, -1]

    def test_grouping_by_year_puts_a_record_with_no_year_in_misc(self):
        entries = self.years()
        assert [displaystate.group_of(e, displaystate.YEAR) for e in entries] == [2021, 2022, -1, 2021]

    def test_the_points_follow_the_grouping(self):
        entries = self.years()
        assert displaystate.points_by_label(entries, grouping=displaystate.YEAR) == {2021: [0, 3], 2022: [1], -1: [2]}
        assert displaystate.points_by_label(entries) == {0: [0, 2], 1: [1], -1: [3]}, \
            "the cluster grouping, for contrast: the same records, other groups"


class TestWhatCanBeInteractedWith:
    def mixed(self):
        entries = entries_of(0, 0, -1, 1)
        entries[1].visibility = displaystate.GRAYED
        entries[2].visibility = displaystate.HIDDEN
        return entries

    def test_only_records_in_full_colour_by_default(self):
        assert list(displaystate.interactable(self.mixed(), range(4), grayed_selectable=False)) == [0, 3]

    def test_grayed_records_too_when_asked_for_but_never_hidden_ones(self):
        assert list(displaystate.interactable(self.mixed(), range(4), grayed_selectable=True)) == [0, 1, 3]

    def test_the_order_given_is_kept(self):
        # The hover tooltip lists records nearest first, so a filter must not re-sort them.
        assert list(displaystate.interactable(self.mixed(), [3, 1, 0], grayed_selectable=True)) == [3, 1, 0]

    def test_nothing_left_is_still_an_index_array(self):
        result = displaystate.interactable(self.mixed(), [1, 2], grayed_selectable=False)
        assert len(result) == 0
        assert result.dtype.kind == "i", "an empty float array raises when used to slice with"

    def test_on_screen_is_everything_but_the_hidden(self):
        assert list(displaystate.on_screen(self.mixed(), range(4))) == [0, 1, 3]

class TestTheInfoPanelAndTooltipGroupByTheLabelShown:
    def test_a_demoted_clusters_records_are_listed_under_misc(self):
        entries = entries_of(0, 1, 1)
        dataset = env(sorted_entries=entries,
                      file_content=env(keywords_available=False, vis_keywords_by_cluster=[]))
        displaystate.relabel_by_purity(entries, [0.9, 0.1], 0.5)
        entries_by_cluster, _ = entry_renderer.get_entries_for_selection(range(3), dataset=dataset)
        assert sorted(entries_by_cluster) == [-1, 0]
        assert [data_idx for data_idx, _ in entries_by_cluster[-1]] == [1, 2]
