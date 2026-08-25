"""The exhibit QR overlay: what it encodes, where it sits, and that it is drawn the right way round.

Nothing here maps a window. The overlay draws into a viewport drawlist, and creating draw items needs no
rendered frame.
"""

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from raven.common.gui import animation as gui_animation  # noqa: E402 -- after importorskip by design
from raven.common.gui import qroverlay  # noqa: E402 -- after importorskip by design

DRAWLIST_SLOT = 2  # slot 1 holds none, and reads as "nothing was drawn"; see `dpg-notes.md`


@pytest.fixture(scope="module")
def dpg_context():
    """One DPG context for the whole module, with an unmapped viewport."""
    dpg.create_context()
    dpg.create_viewport(width=800, height=600)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


@pytest.fixture
def overlay(dpg_context):
    """An installed overlay, removed again afterwards.

    The animator is process-wide, so an overlay left registered would outlive this module's DPG context
    and be rendered into a destroyed one by whatever runs next.
    """
    instance = qroverlay.install(url="https://example.com/raven")
    yield instance
    qroverlay.uninstall(instance)


# --------------------------------------------------------------------------------
# The URL

def test_the_repository_url_comes_from_the_package_metadata():
    """The reason the code is generated at runtime: one source of truth for the URL."""
    assert qroverlay.get_project_url() == "https://github.com/Technologicat/raven"


def test_an_uninstalled_distribution_is_not_an_error():
    """A source checkout that was never installed is an ordinary way to run Raven, and an overlay is a
    demo convenience — so a missing distribution declines rather than raising."""
    assert qroverlay.get_project_url(distribution="no-such-distribution-exists") is None


# --------------------------------------------------------------------------------
# Rasterization

BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)


def pixel_at(rgba, width, x, y):
    """The RGBA tuple at (x, y) in a flat float array of the given width."""
    offset = 4 * (y * width + x)
    return tuple(rgba[offset:offset + 4])


def test_the_raster_is_the_matrix_plus_two_quiet_zones():
    width, height, _ = qroverlay.matrix_to_pixels([[True, False], [False, True]],
                                                  module_size=2, quiet_zone=4,
                                                  foreground=BLACK, background=WHITE)
    assert (width, height) == ((2 + 8) * 2, (2 + 8) * 2)


def test_the_code_is_rasterized_the_right_way_round():
    """A transposed matrix still looks like a QR code and does not scan.

    QR is not symmetric — its three finder squares sit in three corners, not four — so row-vs-column is a
    mistake with no visual tell, and it shows up only when someone points a phone at it. The fixture is
    set in exactly one place, so the two orientations disagree about where the dark pixel lands.
    """
    matrix = [[False, False],
              [True, False]]  # row 1, column 0
    width, _, rgba = qroverlay.matrix_to_pixels(matrix, module_size=1, quiet_zone=0,
                                                foreground=BLACK, background=WHITE)
    assert pixel_at(rgba, width, 0, 1) == (0.0, 0.0, 0.0, 1.0), "the set module should be at x=0, y=1"
    assert pixel_at(rgba, width, 1, 0) == (1.0, 1.0, 1.0, 1.0), "a transposed raster would darken this one"


def test_the_quiet_zone_is_background_coloured():
    """Not merely absent: a scanner needs the margin to be light, and `np.pad` defaults to zero, which in
    a boolean matrix means "not a module" and therefore background — worth pinning rather than assuming."""
    width, _, rgba = qroverlay.matrix_to_pixels([[True]], module_size=1, quiet_zone=2,
                                                foreground=BLACK, background=WHITE)
    assert pixel_at(rgba, width, 0, 0) == (1.0, 1.0, 1.0, 1.0)
    assert pixel_at(rgba, width, 2, 2) == (0.0, 0.0, 0.0, 1.0), "the single module should be inside the margin"


def test_each_module_becomes_module_size_pixels():
    width, height, rgba = qroverlay.matrix_to_pixels([[True]], module_size=3, quiet_zone=0,
                                                     foreground=BLACK, background=WHITE)
    assert (width, height) == (3, 3)
    assert all(pixel_at(rgba, width, x, y) == (0.0, 0.0, 0.0, 1.0)
               for x in range(3) for y in range(3))


# --------------------------------------------------------------------------------
# Drawing

def test_the_overlay_is_ambient(overlay):
    """Load-bearing: it never finishes, and the apps' idle-framerate throttles ask `transient_count`.
    Not ambient, and every app installing one would run at full framerate all evening."""
    assert overlay.ambient is True


def test_the_drawlist_holds_one_item_per_visible_part(overlay):
    """The whole point of rasterizing: the code is one textured quad, not a rectangle per module."""
    overlay._redraw(800, 600)
    items = dpg.get_item_children(overlay.drawlist, DRAWLIST_SLOT) or []
    assert len(items) == 2, f"expected the label and the image, got {len(items)} items"


def test_the_quiet_zone_is_left_clear(overlay):
    """A code drawn flush to its background does not scan; the standard wants four clear modules."""
    assert qroverlay._QUIET_ZONE_MODULES == 4
    modules = len(overlay.matrix)
    side, _ = overlay._get_size()
    assert side == (modules + 2 * qroverlay._QUIET_ZONE_MODULES) * overlay.module_size


@pytest.mark.parametrize("corner", list(qroverlay.Corner))
def test_every_corner_places_the_overlay_inside_the_viewport(dpg_context, corner):
    instance = qroverlay.install(url="https://example.com/raven", corner=corner, margin=16)
    try:
        instance._redraw(800, 600)
        width, height = instance._get_size()
        items = dpg.get_item_children(instance.drawlist, DRAWLIST_SLOT) or []
        assert items, "nothing was drawn, so this asserts nothing about placement"
        for item in items:
            configuration = dpg.get_item_configuration(item)
            x, y = configuration["pmin"][:2] if "pmin" in configuration else configuration["pos"][:2]
            assert 0 <= x <= 800 and 0 <= y <= 600, f"{corner} put something at ({x}, {y})"
    finally:
        qroverlay.uninstall(instance)


def test_uninstall_deregisters_from_the_animator(dpg_context):
    instance = qroverlay.install(url="https://example.com/raven")
    before = gui_animation.animator.active_count
    qroverlay.uninstall(instance)
    assert gui_animation.animator.active_count == before - 1
    assert not dpg.does_item_exist(instance.drawlist)


def test_uninstalling_nothing_is_allowed():
    """`install` returns `None` when no URL can be found, so callers should not need a guard."""
    qroverlay.uninstall(None)
