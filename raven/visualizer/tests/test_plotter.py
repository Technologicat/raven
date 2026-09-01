"""Unit tests for raven.visualizer.plotter.

Two halves of the module, and both are testable without a rendered plot:

  - **`parse_dataset_file`**, which turns a dataset file into the `sorted_xxx` arrays every other
    subsystem indexes into. Every "data_idx" in the Visualizer means "index into these", so what this
    pins is the meaning of the number the rest of the package passes around.
  - **the plotter-space queries**, `get_visible_datapoints` and `get_data_idxs_at_mouse`. They ask DPG
    where the viewport and the mouse are, and then decide; a recording stand-in answers the questions so
    a test can put the mouse where it likes.

What is *not* here is `load_dataset` and `create_highlight_series`, which build DPG widgets and themes
and decide nothing that can be read back without them. They want a real context, which per
`dpg-notes.md` is fine per module -- nobody has written them yet.
"""

import pickle

import numpy as np

import pytest

# `plotter` reaches scipy (for the kd-tree) and DearPyGui, and CI installs only the latter. A
# module-level import failure is a *collection* error rather than a skip, so it would redden the matrix.
plotter = pytest.importorskip("raven.visualizer.plotter")

from unpythonic.env import env  # noqa: E402 -- after importorskip by design

from raven.visualizer import config as visualizer_config  # noqa: E402 -- ditto
from raven.visualizer.app_state import app_state  # noqa: E402 -- ditto

gui_config = visualizer_config.gui_config


class RecordingDPG:
    """Stands in for `dearpygui.dearpygui`, answering the two questions the queries ask it.

    Anything not overridden here comes from the real module.
    """
    def __init__(self, real_dpg):
        self._real_dpg = real_dpg
        self.axis_limits = {"axis0": (0.0, 10.0), "axis1": (0.0, 10.0)}
        self.plot_mouse_pos = [5.0, 5.0]
        self.fitted = []  # axis tags, in call order
        self.deleted = []  # item ids, in call order

    def __getattr__(self, name):
        return getattr(self._real_dpg, name)

    def get_axis_limits(self, tag):
        return self.axis_limits[tag]

    def get_plot_mouse_pos(self):
        return list(self.plot_mouse_pos)

    def fit_axis_data(self, tag):
        self.fitted.append(tag)

    def delete_item(self, item):
        self.deleted.append(item)


# --------------------------------------------------------------------------------
# A dataset file, written the way the importer writes one

def make_entry(title, **fields):
    """One record of `vis_data`, carrying the fields the importer puts there."""
    return env(author="Author, An", bibtex_author="An Author", year="2024",
               title=title, abstract="An abstract.", **fields)


# Five points in three clusters, deliberately interleaved so that sorting by label has to move them.
# Coordinates are distinct and readable, so an assertion can say which point it is looking at.
UNSORTED_TITLES = ["Cluster one, first",     # 0, label  1
                   "Cluster zero, first",    # 1, label  0
                   "The outlier",            # 2, label -1
                   "Cluster zero, second",   # 3, label  0
                   "Cluster one, second"]    # 4, label  1
UNSORTED_LABELS = np.array([1, 0, -1, 0, 1])
UNSORTED_COORDS = np.array([[10.0, 100.0],
                            [20.0, 200.0],
                            [30.0, 300.0],
                            [40.0, 400.0],
                            [50.0, 500.0]])

# What the sort must produce: stable argsort of the labels, so -1 first and ties in original order.
EXPECTED_ORDER = [2, 1, 3, 0, 4]


def write_dataset(path, *, version=1, titles=None, labels=None, coords=None):
    """Write a dataset file with the full key set `importer.import_bibtex` saves."""
    titles = UNSORTED_TITLES if titles is None else titles
    labels = UNSORTED_LABELS if labels is None else labels
    coords = UNSORTED_COORDS if coords is None else coords
    vis_data = [make_entry(title, cluster_id=int(label), cluster_probability=1.0, keywords={})
                for title, label in zip(titles, labels)]
    data = {"version": version,
            "all_input_filenames_raw": ["/somewhere/refs.bib"],
            "all_input_filenames_list": ["refs.bib"],
            "all_input_filenames_str": "refs",
            "embedding_model": "a-model",
            "vis_method": "tsne",
            "n_vis_clusters": 2,
            "n_vis_outliers": 1,
            "labels": labels,
            "vis_data": vis_data,
            "lowdim_data": coords,
            "keywords_available": False,
            "all_keywords": {},
            "vis_keywords_by_cluster": []}
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


@pytest.fixture
def dataset_file(tmp_path):
    return write_dataset(tmp_path / "dataset.pickle")


@pytest.fixture
def dataset(dataset_file):
    return plotter.parse_dataset_file(str(dataset_file))


# --------------------------------------------------------------------------------
# Reading a dataset file

def test_a_dataset_of_an_unknown_version_is_refused(tmp_path):
    # The file is a pickle of whatever the importer of the day wrote, so the version is the only thing
    # standing between a newer file and a confusing failure deep in the plotting code.
    path = write_dataset(tmp_path / "from_the_future.pickle", version=2)
    with pytest.raises(NotImplementedError) as excinfo:
        plotter.parse_dataset_file(str(path))
    assert "from_the_future" in str(excinfo.value), "the error should name the file that cannot be read"


def test_a_dataset_of_the_supported_version_is_read(dataset):
    # Negative control for the test above: version 1 goes through, so the check is reading the version
    # rather than refusing everything.
    assert len(dataset.sorted_entries) == 5


def test_the_dataset_records_where_it_came_from(dataset, dataset_file):
    assert dataset.filename == str(dataset_file)
    assert dataset.absolute_filename == str(dataset_file.resolve())


# --------------------------------------------------------------------------------
# The cluster sort, which is what a "data_idx" means

def test_the_data_is_sorted_by_cluster(dataset):
    # DPG gives a scatter series one colour, so colouring by cluster means one series per cluster, which
    # means the data has to arrive in contiguous per-cluster blocks.
    assert list(dataset.sorted_labels) == [-1, 0, 0, 1, 1]
    assert list(dataset.sorted_orig_data_idxs) == EXPECTED_ORDER


def test_entries_and_coordinates_are_permuted_together(dataset):
    # The three `sorted_xxx` arrays are indexed by the same number everywhere in the package, so a sort
    # that moved one and not another would mislabel every point on the plot.
    assert [entry.title for entry in dataset.sorted_entries] == [UNSORTED_TITLES[i] for i in EXPECTED_ORDER]
    assert dataset.sorted_lowdim_data.tolist() == [UNSORTED_COORDS[i].tolist() for i in EXPECTED_ORDER]


def test_each_entry_knows_its_own_index(dataset):
    # The reverse lookup: given an entry, which `data_idx` is it? Widgets carry entries around and then
    # have to say which point to select.
    for data_idx, entry in enumerate(dataset.sorted_entries):
        assert entry.data_idx == data_idx


def test_entry_titles_are_normalized_for_searching(tmp_path):
    # Search normalizes its query the same way, so a title that was not normalized here cannot be found
    # by typing what is on the screen.
    path = write_dataset(tmp_path / "odd_titles.pickle",
                         titles=["  O₂  reduction  "], labels=np.array([0]), coords=np.array([[1.0, 1.0]]))
    ds = plotter.parse_dataset_file(str(path))
    assert ds.sorted_entries[0].title == "  O₂  reduction  ", "the title itself is left as the source had it"
    assert ds.sorted_entries[0].normalized_title == "O2 reduction"


def test_the_cluster_block_boundaries_slice_the_clusters(dataset):
    # These are used pairwise as slice bounds when building one scatter series per cluster, so what has
    # to hold is that consecutive pairs cut the sorted data at exactly the label changes.
    jumps = dataset.cluster_id_jump_data_idxs
    assert jumps[-1] is None, "the last block runs to the end, and `None` is what says so in a slice"
    blocks = [list(dataset.sorted_labels[start:end]) for start, end in zip(jumps, jumps[1:])]
    assert blocks == [[-1], [0, 0], [1, 1]]


def test_the_kdtree_is_built_over_the_sorted_data(dataset):
    # It answers with indices, and every consumer treats them as indices into `sorted_xxx`. Built over
    # the unsorted data it would answer with the original ones, which are a different set of numbers for
    # every point the sort moved -- and it moved four of these five.
    _, data_idx = dataset.kdtree.query(UNSORTED_COORDS[2], k=1)  # "The outlier", which sorts to the front
    assert data_idx == 0
    assert dataset.sorted_entries[data_idx].title == "The outlier"


def test_parsing_a_dataset_does_not_publish_it(dataset_file, monkeypatch):
    # The docstring's concurrency contract: the caller publishes with one atomic assignment, because
    # background workers read `app_state.dataset` while this runs and a half-built env would hand them
    # fields that do not exist yet -- `kdtree` is populated last.
    sentinel = env(marker="the dataset that was already loaded")
    monkeypatch.setattr(app_state, "dataset", sentinel, raising=False)
    plotter.parse_dataset_file(str(dataset_file))
    assert app_state.dataset is sentinel


# --------------------------------------------------------------------------------
# Which datapoints are on screen

@pytest.fixture
def gui(monkeypatch):
    """Replaces the plotter's DPG binding with a stand-in whose viewport and mouse a test can place."""
    fake_dpg = RecordingDPG(plotter.dpg)
    monkeypatch.setattr(plotter, "dpg", fake_dpg)
    return fake_dpg


def test_nothing_is_visible_when_no_dataset_is_loaded(gui, monkeypatch):
    # Answered before DPG is asked anything: the axes exist from startup, and their limits mean nothing
    # until a file is open.
    monkeypatch.setattr(app_state, "dataset", None, raising=False)
    assert len(plotter.get_visible_datapoints()) == 0


def test_only_the_datapoints_inside_the_viewport_are_visible(gui, dataset):
    gui.axis_limits = {"axis0": (15.0, 45.0), "axis1": (0.0, 1000.0)}
    visible = plotter.get_visible_datapoints(dataset=dataset)
    # x in [15, 45] keeps the points at x=20, 30 and 40, which sort to indices 1, 0 and 2.
    assert sorted(dataset.sorted_lowdim_data[visible, 0].tolist()) == [20.0, 30.0, 40.0]


def test_a_datapoint_exactly_on_the_edge_of_the_viewport_counts_as_visible(gui, dataset):
    # Fitting the axes to the data puts the outermost points exactly on the boundary, so a strict
    # comparison would drop the very points that just got zoomed to. Hence the epsilon.
    gui.axis_limits = {"axis0": (10.0, 50.0), "axis1": (100.0, 500.0)}
    visible = plotter.get_visible_datapoints(dataset=dataset)
    assert len(visible) == 5, "all five, including the two defining the bounding box"


def test_a_datapoint_outside_the_viewport_is_not_visible(gui, dataset):
    # Negative control for the test above: the edge tolerance is an epsilon, not "everything passes".
    gui.axis_limits = {"axis0": (10.0, 45.0), "axis1": (100.0, 500.0)}
    visible = plotter.get_visible_datapoints(dataset=dataset)
    assert 50.0 not in dataset.sorted_lowdim_data[visible, 0].tolist()


def test_the_dataset_can_be_named_explicitly(gui, dataset, monkeypatch):
    # Background workers pass a captured snapshot so a concurrent `open_file` cannot split one operation
    # across two datasets. Pinned by pointing the live one somewhere else entirely.
    monkeypatch.setattr(app_state, "dataset", None, raising=False)
    gui.axis_limits = {"axis0": (0.0, 1000.0), "axis1": (0.0, 1000.0)}
    assert len(plotter.get_visible_datapoints(dataset=dataset)) == 5


# --------------------------------------------------------------------------------
# Which datapoints are under the mouse

@pytest.fixture
def brush(monkeypatch, gui):
    """Puts the mouse over the plot with a known pixel scale, and hands back a way to change either.

    The scale is the interesting parameter: the brush is a radius in *pixels*, and the plot's aspect
    ratio is not square, so the same data-space distance is a different number of pixels per axis.
    """
    scale = {"x": 1.0, "y": 1.0}
    monkeypatch.setattr(plotter.guiutils, "get_pixels_per_plotter_data_unit",
                        lambda plot, xaxis, yaxis: (scale["x"], scale["y"]))
    monkeypatch.setattr(gui_config, "selection_brush_radius_pixels", 10)
    return env(scale=scale, dpg=gui)


def points_at_mouse(dataset, brush, position, *, scale=None):
    if scale is not None:
        brush.scale.update(scale)
    brush.dpg.plot_mouse_pos = list(position)
    return plotter.get_data_idxs_at_mouse(dataset=dataset)


def test_the_datapoint_under_the_mouse_is_found(dataset, brush):
    found = points_at_mouse(dataset, brush, (30.0, 300.0))
    assert [dataset.sorted_entries[i].title for i in found] == ["The outlier"]


def test_a_dataset_smaller_than_the_neighbor_budget_can_be_queried(dataset, brush):
    # The kd-tree is asked for the 100 nearest neighbours. A dataset with fewer points than that gets an
    # answer padded out to length 100 -- with an out-of-range index, and an infinite distance saying so.
    # Slicing with those raised `IndexError`, inside the mouse-move handler, so on a bibliography of a
    # few dozen entries the hover highlight, the tooltip and click-to-select all did nothing.
    assert len(dataset.sorted_entries) < gui_config.datapoints_at_mouse_max_neighbors, \
        "this fixture can only exercise the padding while it is smaller than the neighbor budget"
    found = points_at_mouse(dataset, brush, (30.0, 300.0))
    assert [dataset.sorted_entries[i].title for i in found] == ["The outlier"]


def test_a_datapoint_beyond_the_brush_is_not_found(dataset, brush):
    # Negative control for the test above: the brush has a radius rather than always answering with the
    # nearest point.
    assert len(points_at_mouse(dataset, brush, (300.0, 3000.0))) == 0


@pytest.fixture
def two_points_apart_in_y(tmp_path):
    """Two datapoints differing only in y, so a test can vary the y pixel scale and nothing else."""
    path = write_dataset(tmp_path / "apart_in_y.pickle",
                         titles=["Under the cursor", "Fifty units up"],
                         labels=np.array([0, 0]),
                         coords=np.array([[0.0, 100.0], [0.0, 150.0]]))
    return plotter.parse_dataset_file(str(path))


def test_the_brush_is_a_radius_in_pixels_not_in_data_units(two_points_apart_in_y, brush):
    # The plot's aspect ratio is not square, so a brush measured in data units would be an oval on
    # screen, reaching further along one axis than the other. Squash y by a hundred and the point fifty
    # data units away is half a pixel from the cursor, so the brush reaches it.
    found = points_at_mouse(two_points_apart_in_y, brush, (0.0, 100.0), scale={"x": 1.0, "y": 0.01})
    titles = {two_points_apart_in_y.sorted_entries[i].title for i in found}
    assert titles == {"Under the cursor", "Fifty units up"}


def test_the_same_point_is_out_of_reach_at_an_unsquashed_scale(two_points_apart_in_y, brush):
    # Negative control for the test above, and what makes it about the scale rather than the positions:
    # the same cursor and the same two points, an isotropic scale, and the far one drops out.
    found = points_at_mouse(two_points_apart_in_y, brush, (0.0, 100.0), scale={"x": 1.0, "y": 1.0})
    titles = {two_points_apart_in_y.sorted_entries[i].title for i in found}
    assert titles == {"Under the cursor"}


def test_nothing_is_under_the_mouse_when_no_dataset_is_loaded(brush, monkeypatch):
    monkeypatch.setattr(app_state, "dataset", None, raising=False)
    assert len(plotter.get_data_idxs_at_mouse()) == 0


def test_a_degenerate_plot_scale_finds_nothing_rather_than_dividing_by_it(dataset, brush):
    # The plot reports a zero pixel scale before it has been laid out; every pixel distance would then be
    # zero, so the brush would answer with the whole dataset.
    assert len(points_at_mouse(dataset, brush, (30.0, 300.0), scale={"x": 0.0, "y": 1.0})) == 0


@pytest.fixture
def point_at_the_origin(tmp_path):
    """A dataset with a datapoint at exactly (0, 0), which is the arrangement the workaround is about.

    The fixture has to be this way round: with the nearest point a hundred units off, the brush rejects
    it on distance whether or not the guard fires, and the test below passes against code that has no
    guard at all.
    """
    path = write_dataset(tmp_path / "at_origin.pickle",
                         titles=["At the origin", "Somewhere else"],
                         labels=np.array([0, 0]),
                         coords=np.array([[0.0, 0.0], [90.0, 90.0]]))
    return plotter.parse_dataset_file(str(path))


def test_the_mouse_reported_at_exactly_the_origin_is_not_believed(point_at_the_origin, brush):
    # DPG's `get_plot_mouse_pos` returns [0, 0] before the plot has seen the cursor, and t-SNE output
    # tends to have points near the origin -- so at startup the app would report a datapoint under a
    # cursor the user has not moved onto the plot yet.
    found = points_at_mouse(point_at_the_origin, brush, (0.0, 0.0))
    assert len(found) == 0, "a point is sitting right there, and that is exactly the trap"


def test_a_mouse_position_near_but_not_at_the_origin_is_believed(point_at_the_origin, brush):
    # Negative control for the test above: the guard is exact equality on both axes, not a dead zone
    # around the origin, so a dataset that genuinely lives near zero stays usable.
    found = points_at_mouse(point_at_the_origin, brush, (0.0, 0.001))
    assert [point_at_the_origin.sorted_entries[i].title for i in found] == ["At the origin"]


# --------------------------------------------------------------------------------
# The select-radius brush outline

def outline(radius_pixels=10.0, scale_x=1.0, scale_y=1.0, center=(0.0, 0.0)):
    return np.array(plotter.brush_outline_points(center, radius_pixels, scale_x, scale_y))


def test_the_brush_outline_is_a_circle_on_screen():
    # The user aims with a circle, so what has to be round is the *pixel* distance from the cursor to
    # every point of the outline -- which is the same statement as the brush's own test above.
    points = outline(radius_pixels=10.0, scale_x=4.0, scale_y=0.25)
    pixel_offsets = points * np.array([4.0, 0.25])  # data space -> pixels
    pixel_radii = np.hypot(pixel_offsets[:, 0], pixel_offsets[:, 1])
    assert np.allclose(pixel_radii, 10.0)


def test_the_brush_outline_is_an_ellipse_in_data_space_when_the_axes_differ():
    # Negative control for the test above: it is only round in pixels. Drawn as a data-space circle it
    # would sit somewhere other than where the brush reaches, which is the failure being prevented.
    points = outline(radius_pixels=10.0, scale_x=4.0, scale_y=0.25)
    assert not np.allclose(np.ptp(points[:, 0]), np.ptp(points[:, 1]))


def test_the_brush_outline_is_a_circle_in_data_space_when_the_axes_agree():
    # ...and the other half of that control: with equal scales the two readings coincide.
    points = outline(radius_pixels=10.0, scale_x=2.0, scale_y=2.0)
    assert np.allclose(np.ptp(points[:, 0]), np.ptp(points[:, 1]))


def test_the_brush_outline_is_centred_on_the_cursor():
    # By the bounding box rather than the mean of the points: the ring closes by repeating its first
    # point, so the mean is pulled towards that point by a sampling artifact rather than by the geometry.
    points = outline(center=(3.0, -7.0))
    assert np.allclose((points.max(axis=0) + points.min(axis=0)) / 2, [3.0, -7.0])


def test_a_bigger_brush_draws_a_bigger_outline():
    small = np.ptp(outline(radius_pixels=5.0)[:, 0])
    large = np.ptp(outline(radius_pixels=20.0)[:, 0])
    assert large > small


def test_the_brush_outline_is_a_closed_loop():
    # It is handed to `draw_polygon`, so the first and last points have to meet or the ring has a notch.
    points = outline()
    assert np.allclose(points[0], points[-1])


# --------------------------------------------------------------------------------
# The highlight brightness curve
#
# A calibrated perceptual curve, hand-tuned by eye. Nothing here asserts that the calibration is *right*
# -- that is a judgement about how it looks -- only that the shape it was tuned to have is the shape it
# still has, so a later change to the coefficients cannot quietly invert or flatten it.

N_MANY = 100


def test_the_highlight_brightens_with_the_animation_channel():
    dim = plotter.compute_highlight_alpha(0.0, n_data=10, n_many=N_MANY)
    bright = plotter.compute_highlight_alpha(1.0, n_data=10, n_many=N_MANY)
    assert dim < bright


def test_the_highlight_is_monotonic_in_the_animation_channel():
    # It is a pulsation, so a dip anywhere in the cycle would read as a stutter.
    alphas = [plotter.compute_highlight_alpha(x / 20, n_data=10, n_many=N_MANY) for x in range(21)]
    assert alphas == sorted(alphas)


def test_a_larger_set_is_drawn_more_faintly_per_datapoint():
    # The whole reason the curve takes `n_data`: translucent dots accumulate, so a crowd drawn at the
    # brightness of a handful becomes an opaque blob.
    few = plotter.compute_highlight_alpha(1.0, n_data=1, n_many=N_MANY)
    many = plotter.compute_highlight_alpha(1.0, n_data=N_MANY, n_many=N_MANY)
    assert many < few


def test_the_per_datapoint_alpha_stops_falling_once_the_set_is_large():
    # `n_many` is "this is already as many as it needs to be", so beyond it the curve is flat rather
    # than continuing down towards invisibility on a very large selection.
    at_the_limit = plotter.compute_highlight_alpha(1.0, n_data=N_MANY, n_many=N_MANY)
    far_beyond = plotter.compute_highlight_alpha(1.0, n_data=50 * N_MANY, n_many=N_MANY)
    assert at_the_limit == far_beyond


def test_the_alpha_stays_inside_the_range_a_colour_channel_has():
    # Sampled across both parameters, since the coefficients are interpolated between two hand-tuned
    # pairs and nothing else bounds the result.
    for n_data in (0, 1, 10, N_MANY, 10 * N_MANY):
        for x in (0.0, 0.25, 0.5, 0.75, 1.0):
            alpha = plotter.compute_highlight_alpha(x, n_data=n_data, n_many=N_MANY)
            assert 0 <= alpha <= 255, f"alpha {alpha} out of range at x={x}, n_data={n_data}"


def test_the_dimmest_a_highlight_gets_is_still_visible():
    # A highlight that faded to nothing at the bottom of its cycle would blink rather than pulsate.
    assert plotter.compute_highlight_alpha(0.0, n_data=10 * N_MANY, n_many=N_MANY) > 0


# --------------------------------------------------------------------------------
# Zoom and lifecycle

def test_resetting_the_zoom_fits_both_axes(gui):
    plotter.reset_zoom()
    assert gui.fitted == ["axis0", "axis1"]


def test_clearing_the_cluster_colour_themes_deletes_them_all(gui, monkeypatch):
    # One theme is created per cluster on every load, so a load that did not clear the previous set would
    # leak a few hundred DPG items per file opened.
    monkeypatch.setattr(plotter, "_cluster_color_themes", ["theme_a", "theme_b", "theme_c"])
    plotter.clear_cluster_color_themes()
    assert gui.deleted == ["theme_a", "theme_b", "theme_c"]
    assert plotter._cluster_color_themes == [], "...and the list is emptied, or the next clear deletes them twice"
