"""Tests for `raven.common.gui.filegrid` — the join between a directory listing and a thumbnail grid.

Two things are worth pinning here, and both fail quietly rather than loudly.

The first is **which entries get a decoded preview**: an entry mis-sorted into the decode queue costs
milliseconds per frame and produces nothing, and one mis-sorted out of it shows an icon where a picture was
wanted. Neither raises.

The second is **when decoding starts**. The scheduler has to refuse three things — a set still changing
under a scroll, a set already in flight, and an entry a finished batch failed to produce — and each of
those, left out, is a decoder that restarts forever. Nothing about that is visible except as heat, so it is
tested against a stand-in pipeline rather than a real one.

No window is mapped, so nothing here takes keyboard focus and none of it carries the `gui` marker. Which
tiles are *actually* on screen needs rendered frames, so `visible_on_screen` is supplied by the test.
"""

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")
pytest.importorskip("torch", reason="torch not installed")

from raven.common import filelisting  # noqa: E402 -- after importorskip by design
from raven.common.gui.filegrid import FileGrid  # noqa: E402 -- after importorskip by design

TILE = 32


def _entry(name, kind=filelisting.KIND_FILE, **kwargs):
    return filelisting.FileEntry(name=name, path=f"/somewhere/{name}", kind=kind,
                                 is_hidden=False, mtime=1.0, size=1, **kwargs)


IMAGE_EXTENSIONS = (".png", ".jpg")


def _icon_name_for(entry):
    """The dialog's rule in miniature: directories get a folder, images get decoded, the rest get a page."""
    if entry.is_dir:
        return "folder"
    if entry.name.lower().endswith(IMAGE_EXTENSIONS):
        return None
    return "document"


def _solid(width, height):
    return [1.0, 1.0, 1.0, 1.0] * (width * height)


ICON_ASSETS = {"folder": (8, 8, _solid(8, 8)),
               "document": (8, 8, _solid(8, 8))}


class FakePipeline:
    """Records what it was asked to decode, and produces nothing unless told to."""

    def __init__(self):
        self.batches = []
        self.in_progress = False
        self._results = []
        self.tile_size = TILE

    def start(self, paths):
        self.batches.append(list(paths))
        self.in_progress = True

    def finish(self, results=()):
        """Complete the batch, optionally having produced thumbnails for these positions."""
        self._results = [(position, [0.0] * (TILE * TILE * 4)) for position in results]
        self.in_progress = False

    def poll(self):
        results, self._results = self._results, []
        return results

    def cancel(self):
        self.in_progress = False

    def shutdown(self):
        self.in_progress = False

    def set_tile_size(self, size):
        self.tile_size = size


@pytest.fixture(scope="module")
def dpg_context():
    """One DPG context for the whole module, with an unmapped viewport."""
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


@pytest.fixture
def make_grid(dpg_context, request):
    """A grid with a stand-in decoder and no placeholder pool, in a throwaway window."""
    built = []

    def build(on_screen=(), **kwargs):
        window = dpg.add_window(label="host", tag=f"host_{request.node.name}_{len(built)}")
        kwargs.setdefault("icon_name_for", _icon_name_for)
        grid = FileGrid(parent=window, width=400, height=300,
                        tile_size=TILE,
                        icon_assets=ICON_ASSETS,
                        thumbnail_device="cpu",
                        placeholder_count=0,
                        settle_time=0.0,
                        **kwargs)
        grid._pipeline.shutdown()
        grid._pipeline = FakePipeline()
        grid.visible_on_screen = lambda: list(grid.on_screen)
        grid.on_screen = list(on_screen)
        built.append((grid, window))
        return grid
    yield build
    for grid, window in built:
        grid.destroy()
        dpg.delete_item(window)


# --------------------------------------------------------------------------------
# The listing

def test_entries_that_are_not_images_get_an_icon(make_grid):
    grid = make_grid()
    grid.set_listing([_entry("..", kind=filelisting.KIND_DIR, is_parent=True),
                      _entry("subdir", kind=filelisting.KIND_DIR),
                      _entry("notes.txt"),
                      _entry("photo.png")])
    assert set(grid._shared_images) == {0, 1, 2}
    assert grid._decodable == {3}


def test_the_directories_are_listed_too(make_grid):
    """Removing them would remove the only way to navigate, which is the obvious version of this and wrong."""
    grid = make_grid()
    grid.set_listing([_entry("..", kind=filelisting.KIND_DIR, is_parent=True),
                      _entry("subdir", kind=filelisting.KIND_DIR),
                      _entry("photo.png")])
    assert [entry.name for entry in grid.entries] == ["..", "subdir", "photo.png"]
    assert grid.visible_count == 3


def test_an_icon_with_no_asset_falls_back_to_the_placeholder(make_grid):
    """A file type nobody has drawn yet should look unfinished, not crash the listing."""
    grid = make_grid(icon_name_for=lambda entry: "no such icon")
    grid.set_listing([_entry("notes.txt")])
    assert grid._shared_images == {}


def test_one_texture_serves_every_entry_with_the_same_icon(make_grid):
    """The alternative is a texture per file, which is what makes a large directory expensive."""
    grid = make_grid()
    grid.set_listing([_entry(f"notes{i}.txt") for i in range(5)])
    assert len(set(grid._shared_images.values())) == 1


# --------------------------------------------------------------------------------
# The cursor

def test_the_cursor_follows_the_file_across_a_re_sort(make_grid):
    """Indices move whenever the listing is re-filtered or re-sorted; the path is the stable identity."""
    grid = make_grid()
    grid.set_listing([_entry("a.txt"), _entry("b.txt"), _entry("c.txt")])
    grid.set_current(2)
    grid.set_listing([_entry("c.txt"), _entry("b.txt"), _entry("a.txt")])
    assert grid.current_entry.name == "c.txt"


def test_the_cursor_keeps_its_place_when_its_file_is_gone(make_grid):
    """Position is what survives when identity does not — it does not jump to the top of the listing.

    The case is a keystroke in the file dialog's find field narrowing the cursor's file out of the listing.
    Falling to the first entry would throw the user to the top of the directory on every such character,
    and would also put this grid somewhere different from the table showing the same listing.
    """
    grid = make_grid()
    grid.set_listing([_entry("a.txt"), _entry("b.txt")])
    grid.set_current(1)
    grid.set_listing([_entry("a.txt"), _entry("c.txt")])
    assert grid.current_entry.name == "c.txt"


def test_the_cursor_goes_home_when_its_file_comes_back(make_grid):
    """Filter the cursor's file out, then stop filtering: the cursor returns to it.

    While the file is missing the cursor holds its position, so a *different* file slides under it — but the
    user never chose that file, the list moved under a stationary cursor. Remembering the chosen entry apart
    from the displayed one is what lets the cursor go home. Typing a character and erasing it again is the
    shape this takes in the file dialog, and it is a round trip a user expects to be free.
    """
    grid = make_grid()
    grid.set_listing([_entry("a.txt"), _entry("b.txt"), _entry("c.txt")])
    grid.set_current(0)  # deliberate: the user chose a.txt

    grid.set_listing([_entry("b.txt"), _entry("c.txt")])  # a.txt filtered out
    assert grid.current_entry.name == "b.txt"  # held its position; the user did not pick this

    grid.set_listing([_entry("a.txt"), _entry("b.txt"), _entry("c.txt")])  # filter cleared
    assert grid.current_entry.name == "a.txt"


def test_moving_the_cursor_adopts_the_new_file(make_grid):
    """The escape hatch from the rule above: any deliberate move re-anchors, so the cursor stops going home.

    Without this, a user who filtered, then arrowed to what they were looking for, would be dragged back to
    where they started the moment the filter cleared.
    """
    grid = make_grid()
    grid.set_listing([_entry("a.txt"), _entry("b.txt"), _entry("c.txt")])
    grid.set_current(0)
    grid.set_listing([_entry("b.txt"), _entry("c.txt")])
    grid.set_current(1)  # deliberate: the user chose c.txt

    grid.set_listing([_entry("a.txt"), _entry("b.txt"), _entry("c.txt")])
    assert grid.current_entry.name == "c.txt"


def test_a_new_directory_starts_at_the_top(make_grid):
    """A changed `listing_key` says "this is somewhere else", and the cursor does not carry a position over.

    Carrying one across a `chdir` would drop the cursor at an index that named a file in the directory you
    left.
    """
    grid = make_grid()
    grid.set_listing([_entry("a.txt"), _entry("b.txt"), _entry("c.txt")], listing_key="/here")
    grid.set_current(2)
    grid.set_listing([_entry("x.txt"), _entry("y.txt"), _entry("z.txt")], listing_key="/elsewhere")
    assert grid.current_entry.name == "x.txt"


def test_the_same_directory_relisted_keeps_the_cursor(make_grid):
    """The other half of the same rule: an unchanged key is a rebuild in place, however much the list moved."""
    grid = make_grid()
    grid.set_listing([_entry("a.txt"), _entry("b.txt"), _entry("c.txt")], listing_key="/here")
    grid.set_current(2)
    grid.set_listing([_entry("c.txt"), _entry("a.txt")], listing_key="/here")
    assert grid.current_entry.name == "c.txt"


def test_selected_entries_come_back_in_display_order(make_grid):
    grid = make_grid()
    grid.set_listing([_entry("a.txt"), _entry("b.txt"), _entry("c.txt")])
    grid.toggle_select(2)
    grid.toggle_select(0)
    assert [entry.name for entry in grid.selected_entries] == ["a.txt", "c.txt"]


# --------------------------------------------------------------------------------
# Which thumbnails get decoded, and when

def test_only_the_images_on_screen_are_decoded(make_grid):
    """The whole budget argument: a listing of thousands must not decode what nobody is looking at."""
    grid = make_grid(on_screen=[0, 1])
    grid.set_listing([_entry("a.png"), _entry("b.txt"), _entry("c.png")])
    grid.tick()  # first tick records the wanted set
    grid.tick()  # second one finds it unchanged, and starts
    assert [str(p.name) for p in grid._pipeline.batches[-1]] == ["a.png"]


def test_a_set_still_changing_does_not_start_a_batch(make_grid):
    """Otherwise a scroll cancels and restarts the decoder at every row it passes."""
    grid = make_grid(on_screen=[0])
    grid.set_listing([_entry("a.png"), _entry("b.png")])
    grid.tick()
    grid.on_screen = [1]
    grid.tick()
    grid.on_screen = [0, 1]
    grid.tick()
    assert grid._pipeline.batches == []


def test_a_completing_thumbnail_does_not_restart_the_batch(make_grid):
    """The wanted set shrinks as results arrive; that must not read as a new request."""
    grid = make_grid(on_screen=[0, 1])
    grid.set_listing([_entry("a.png"), _entry("b.png")])
    grid.tick()
    grid.tick()
    assert len(grid._pipeline.batches) == 1
    grid._pipeline.finish(results=[0])  # the first of the two arrives; the batch is done
    grid.tick()
    grid.tick()
    assert len(grid._pipeline.batches) == 1


def test_an_entry_a_finished_batch_failed_to_produce_is_not_asked_for_again(make_grid):
    """A file the decoder cannot read would otherwise be retried for as long as it is on screen."""
    grid = make_grid(on_screen=[0])
    grid.set_listing([_entry("broken.png")])
    grid.tick()
    grid.tick()
    grid._pipeline.finish(results=[])  # ran to completion, produced nothing
    for _ in range(4):
        grid.tick()
    assert len(grid._pipeline.batches) == 1


def test_a_new_listing_asks_again(make_grid):
    """The refusals above are about one listing; opening a folder starts the question over."""
    grid = make_grid(on_screen=[0])
    grid.set_listing([_entry("a.png")])
    grid.tick()
    grid.tick()
    grid._pipeline.finish(results=[])
    grid.set_listing([_entry("a.png")])
    grid.tick()
    grid.tick()
    assert len(grid._pipeline.batches) == 2


def test_a_result_is_routed_to_the_entry_that_asked_for_it(make_grid):
    """The pipeline counts positions within its batch, which are not entry indices."""
    grid = make_grid(on_screen=[1, 3])
    entries = [_entry("a.txt"), _entry("b.png"), _entry("c.txt"), _entry("d.png")]
    grid.set_listing(entries)
    grid.tick()
    grid.tick()
    grid._pipeline.finish(results=[1])  # the second image of the batch, i.e. entry 3
    grid.tick()
    assert entries[3].path in grid._thumbnail_cache
    assert entries[1].path not in grid._thumbnail_cache


# --------------------------------------------------------------------------------
# The thumbnail cache

def test_a_re_listing_reuses_the_decoded_thumbnails(make_grid):
    """The find field re-lists on every keystroke; decoding again each time is the cost this removes.

    Keyed by path, because the index a file occupies moves whenever the listing is re-filtered or re-sorted
    — a texture remembered by index would be a picture of the wrong file.
    """
    grid = make_grid(on_screen=[0])
    entries = [_entry("a.png"), _entry("b.png")]
    grid.set_listing(entries)
    grid.tick()
    grid.tick()
    grid._pipeline.finish(results=[0])
    grid.tick()
    texture = grid._thumbnail_cache[entries[0].path]

    # Re-list with the order reversed, as a re-sort would: the same file, a different index.
    grid.set_listing([entries[1], entries[0]])
    assert grid._shared_images.get(1) == texture   # re-attached at its new index...
    assert grid._thumbnail_cache[entries[0].path] == texture   # ...from the same texture as before


def test_a_cached_thumbnail_is_not_decoded_again(make_grid):
    """Having the picture already is what keeps it out of the decode queue."""
    grid = make_grid(on_screen=[0])
    entries = [_entry("a.png")]
    grid.set_listing(entries)
    grid.tick()
    grid.tick()
    grid._pipeline.finish(results=[0])
    grid.tick()
    batches_after_first_decode = len(grid._pipeline.batches)

    grid.set_listing(entries)  # what a keystroke in the find field does
    for _ in range(4):
        grid.tick()
    assert len(grid._pipeline.batches) == batches_after_first_decode


def test_eviction_never_takes_a_thumbnail_the_listing_is_showing(make_grid):
    """Evicting an on-screen tile would blank it and decode it again, which is the opposite of the point.

    So a directory larger than the limit still displays in full; the limit bounds what is kept for folders
    that are no longer on screen.
    """
    grid = make_grid(on_screen=[0])
    grid._thumbnail_cache_limit = 1
    old = _entry("gone.png")
    shown = [_entry("here1.png"), _entry("here2.png")]
    grid._thumbnail_cache = {old.path: "tex_old", shown[0].path: "tex_1", shown[1].path: "tex_2"}

    grid.set_listing(shown)

    assert old.path not in grid._thumbnail_cache          # the one not on screen went...
    assert shown[0].path in grid._thumbnail_cache         # ...and the shown ones stayed, over the limit
    assert shown[1].path in grid._thumbnail_cache


def test_decoding_a_path_twice_does_not_strand_the_first_texture(make_grid):
    """Nothing else refers to it, so the cache is the only thing that would ever have deleted it."""
    grid = make_grid(on_screen=[0])
    entries = [_entry("a.png")]
    grid.set_listing(entries)
    grid.tick()
    grid.tick()
    grid._pipeline.finish(results=[0])
    grid.tick()
    first = grid._thumbnail_cache[entries[0].path]

    grid._store_thumbnail(0, [0.0] * (TILE * TILE * 4))  # as a re-decode of the same file would

    assert grid._thumbnail_cache[entries[0].path] != first
    assert not dpg.does_item_exist(first)  # tag


def test_changing_the_tile_size_empties_the_cache(make_grid):
    """Every cached tile is the wrong size afterwards, and a wrong-sized one is worse than none."""
    grid = make_grid(on_screen=[0])
    entries = [_entry("a.png")]
    grid.set_listing(entries)
    grid.tick()
    grid.tick()
    grid._pipeline.finish(results=[0])
    grid.tick()
    assert grid._thumbnail_cache

    grid.set_tile_size(TILE * 2)
    assert grid._thumbnail_cache == {}
