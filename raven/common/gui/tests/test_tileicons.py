"""Tests for `raven.common.gui.tileicons` — icon assets resampled to a grid's tile size.

The failure these guard against is quiet: a wrongly-sized or wrongly-cached texture still draws, it just
draws the wrong picture or the right one blocky. So the checks are on the identities and the sizes rather
than on the pixels.

No window is mapped, so nothing here takes keyboard focus and none of it carries the `gui` marker. Textures
are real DPG items, so a context is needed; no rendered frame is.
"""

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")
pytest.importorskip("torch", reason="torch not installed")

from raven.common.gui.tileicons import TileIconCache  # noqa: E402 -- after importorskip by design


def _solid(width: int, height: int, rgba=(1.0, 0.0, 0.0, 1.0)):
    """A flat RGBA source of the shape `dpg.load_image` hands back."""
    return list(rgba) * (width * height)


@pytest.fixture(scope="module")
def dpg_context():
    """One DPG context for the whole module, with an unmapped viewport."""
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


@pytest.fixture
def make_cache(dpg_context):
    built = []

    def build(tile_size: int = 32, **kwargs):
        cache = TileIconCache(tile_size, **kwargs)
        built.append(cache)
        return cache
    yield build
    for cache in built:
        cache.destroy()


# --------------------------------------------------------------------------------
# Building textures

def test_texture_is_built_at_the_tile_size(make_cache):
    cache = make_cache(tile_size=32)
    cache.add("folder", 16, 16, _solid(16, 16))
    tag = cache.texture("folder")
    assert dpg.does_item_exist(tag)  # tag
    assert dpg.get_item_configuration(tag)["width"] == 32  # tag
    assert dpg.get_item_configuration(tag)["height"] == 32  # tag


def test_a_non_square_source_is_padded_to_a_square(make_cache):
    """Whatever the source shape, the tile is square — otherwise the grid's lattice breaks."""
    cache = make_cache(tile_size=24)
    cache.add("wide", 32, 8, _solid(32, 8))
    tag = cache.texture("wide")
    assert dpg.get_item_configuration(tag)["width"] == 24  # tag
    assert dpg.get_item_configuration(tag)["height"] == 24  # tag


def test_the_same_icon_gives_the_same_texture(make_cache):
    """One texture serves every tile showing this icon; that is the whole point of the cache."""
    cache = make_cache()
    cache.add("folder", 16, 16, _solid(16, 16))
    assert cache.texture("folder") == cache.texture("folder")


def test_different_icons_give_different_textures(make_cache):
    cache = make_cache()
    cache.add("folder", 16, 16, _solid(16, 16))
    cache.add("document", 16, 16, _solid(16, 16, (0.0, 1.0, 0.0, 1.0)))
    assert cache.texture("folder") != cache.texture("document")


def test_an_unknown_name_is_answered_rather_than_raised(make_cache):
    """A caller mapping file types to icons will have types with no picture, which is not an error."""
    cache = make_cache()
    assert cache.texture("no such icon") is None


def test_a_source_of_the_wrong_length_is_rejected_at_registration(make_cache):
    """Caught where the mistake is, rather than as a mangled tile much later."""
    cache = make_cache()
    with pytest.raises(ValueError):
        cache.add("folder", 16, 16, _solid(8, 8))


# --------------------------------------------------------------------------------
# Tile size

def test_changing_the_tile_size_rebuilds_at_the_new_size(make_cache):
    cache = make_cache(tile_size=16)
    cache.add("folder", 16, 16, _solid(16, 16))
    old = cache.texture("folder")
    cache.set_tile_size(64)
    new = cache.texture("folder")
    assert new != old
    assert dpg.get_item_configuration(new)["width"] == 64  # tag
    assert cache.tile_size == 64


def test_the_old_textures_go_when_the_tile_size_changes(make_cache):
    """Otherwise every visited tile size leaks a texture per icon."""
    cache = make_cache(tile_size=16)
    cache.add("folder", 16, 16, _solid(16, 16))
    old = cache.texture("folder")
    cache.set_tile_size(64)
    assert not dpg.does_item_exist(old)  # tag


def test_setting_the_same_tile_size_keeps_the_texture(make_cache):
    cache = make_cache(tile_size=16)
    cache.add("folder", 16, 16, _solid(16, 16))
    old = cache.texture("folder")
    cache.set_tile_size(16)
    assert cache.texture("folder") == old


def test_re_registering_a_name_replaces_its_texture(make_cache):
    cache = make_cache()
    cache.add("folder", 16, 16, _solid(16, 16))
    old = cache.texture("folder")
    cache.add("folder", 16, 16, _solid(16, 16, (0.0, 0.0, 1.0, 1.0)))
    assert not dpg.does_item_exist(old)  # tag
    assert cache.texture("folder") != old


def test_add_all_registers_every_entry(make_cache):
    cache = make_cache()
    cache.add_all({"folder": (16, 16, _solid(16, 16)),
                   "document": (16, 16, _solid(16, 16))})
    assert cache.texture("folder") is not None
    assert cache.texture("document") is not None


# --------------------------------------------------------------------------------
# Teardown

def test_destroy_removes_the_textures_but_keeps_the_sources(make_cache):
    cache = make_cache()
    cache.add("folder", 16, 16, _solid(16, 16))
    tag = cache.texture("folder")
    cache.destroy()
    assert not dpg.does_item_exist(tag)  # tag
    assert cache.texture("folder") is not None  # rebuilt from the source it still holds
