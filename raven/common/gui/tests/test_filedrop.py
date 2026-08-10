"""Tests for `raven.common.gui.filedrop`'s drop routing.

What is tested here is the router — the part that decides, from a set of dropped paths, which handler (if
any) should see them. That decision is pure: predicates over paths, then a dispatch. It needs no window, no
render loop and no drag, so it is ordinary test material even though the feature around it is not.

**The GLFW half is deliberately not tested**, and the reason is the same one that makes `dnd_probe.py` a
probe rather than a test: a real drop needs a human to drag a file onto a mapped window. What could be
asserted without one — that `install` refuses off the render thread, that binding degrades rather than
raising — is asserted below; the delivery path itself stays in `investigations/dpg-dnd/`.

The rejection path is exercised through `on_rejected` rather than the modal messagebox, since a modal wants
a viewport to center on and a render loop to wait for.
"""

import os
import threading

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed (GUI toolkit absent in CI)")

from raven.common.gui import filedrop  # noqa: E402 -- after importorskip by design


@pytest.fixture
def collector():
    """A rejection reporter that records instead of showing a dialog."""
    rejections = []
    def on_rejected(title, message):
        rejections.append((title, message))
    on_rejected.rejections = rejections
    return on_rejected


def make_files(tmp_path, *names):
    """Create empty files and return their absolute paths, in the order given."""
    paths = []
    for name in names:
        p = tmp_path / name
        p.write_text("")
        paths.append(str(p))
    return paths


# --------------------------------------------------------------------------------
# Predicates

def test_by_extension_ignores_case_and_leading_dot(tmp_path):
    bib, = make_files(tmp_path, "refs.BIB")
    assert filedrop.by_extension(".bib")(bib)
    assert filedrop.by_extension("bib")(bib)
    assert not filedrop.by_extension(".pickle")(bib)


def test_by_extension_does_not_match_a_directory(tmp_path):
    # A directory named like a file would otherwise be handed to a handler expecting file contents.
    d = tmp_path / "looks_like.bib"
    d.mkdir()
    assert not filedrop.by_extension(".bib")(str(d))


def test_by_extension_does_not_match_a_nonexistent_path(tmp_path):
    assert not filedrop.by_extension(".bib")(str(tmp_path / "never_created.bib"))


def test_all_of_requires_every_predicate(tmp_path):
    png, txt = make_files(tmp_path, "a.png", "b.txt")
    named_png = filedrop.by_extension(".png")
    big_enough = lambda path: os.path.getsize(path) >= 0  # noqa: E731 -- a one-line stand-in for a content test
    assert filedrop.all_of(named_png, big_enough)(png)
    assert not filedrop.all_of(named_png, big_enough)(txt)


def test_all_of_short_circuits_so_a_content_test_never_sees_the_wrong_file(tmp_path):
    """Ordering is the point: the cheap name test must spare unrelated drops a decode.

    The avatar apps pair `by_extension` with a predicate that opens the file, and a drop can carry anything.
    """
    opened = []
    def records_being_asked(path):
        opened.append(path)
        return True
    png, txt = make_files(tmp_path, "a.png", "b.txt")
    matches = filedrop.all_of(filedrop.by_extension(".png"), records_being_asked)
    assert matches(png)
    assert not matches(txt)
    assert opened == [png]  # the .txt never reached the expensive half


def test_is_directory(tmp_path):
    d = tmp_path / "folder"
    d.mkdir()
    f, = make_files(tmp_path, "file.txt")
    assert filedrop.is_directory(str(d))
    assert not filedrop.is_directory(f)


# --------------------------------------------------------------------------------
# Routing

def test_matching_files_reach_their_handler(tmp_path, collector):
    seen = []
    route = filedrop.make_router([filedrop.DropRule(matches=filedrop.by_extension(".bib"),
                                                    handler=seen.append,
                                                    label="BibTeX files")],
                                 reference_window="w", on_rejected=collector)
    paths = make_files(tmp_path, "a.bib", "b.bib")
    route(paths)
    assert seen == [paths]
    assert collector.rejections == []


def test_an_unmatched_file_rejects_the_whole_drop(tmp_path, collector):
    # Partial action on an ambiguous gesture is harder to undo than no action, so the matching half is
    # dropped too — the point of the assertion is that `seen` stays empty, not merely that we rejected.
    seen = []
    route = filedrop.make_router([filedrop.DropRule(matches=filedrop.by_extension(".bib"),
                                                    handler=seen.append,
                                                    label="BibTeX files")],
                                 reference_window="w", on_rejected=collector)
    route(make_files(tmp_path, "a.bib", "notes.txt"))
    assert seen == []
    assert len(collector.rejections) == 1
    title, message = collector.rejections[0]
    assert "notes.txt" in message
    assert "BibTeX files" in message  # the dialog says what would have worked


def test_a_drop_straddling_two_rules_is_rejected(tmp_path, collector):
    seen = []
    route = filedrop.make_router([filedrop.DropRule(matches=filedrop.by_extension(".bib"),
                                                    handler=lambda paths: seen.append(("bib", paths)),
                                                    label="BibTeX files"),
                                  filedrop.DropRule(matches=filedrop.by_extension(".pickle"),
                                                    handler=lambda paths: seen.append(("pickle", paths)),
                                                    label="dataset files")],
                                 reference_window="w", on_rejected=collector)
    route(make_files(tmp_path, "a.bib", "b.pickle"))
    assert seen == []
    assert len(collector.rejections) == 1


def test_several_files_are_rejected_by_a_single_file_rule(tmp_path, collector):
    seen = []
    route = filedrop.make_router([filedrop.DropRule(matches=filedrop.by_extension(".pickle"),
                                                    handler=seen.append,
                                                    label="a dataset file",
                                                    multiple=False)],
                                 reference_window="w", on_rejected=collector)
    two = make_files(tmp_path, "a.pickle", "b.pickle")
    route(two)
    assert seen == []
    assert len(collector.rejections) == 1

    route(two[:1])  # ...but one is fine
    assert seen == [two[:1]]


def test_the_first_matching_rule_wins(tmp_path, collector):
    """Rule order is the routing mechanism, not a tie-break detail.

    This is the shape `raven-avatar-settings-editor` relies on: "an image with transparency is a character,
    any other image is a backdrop" is two rules whose predicates deliberately overlap, resolved by order.
    Reordering them would silently send every character image to the backdrop slot.
    """
    seen = []
    narrow, broad = make_files(tmp_path, "special.png", "ordinary.png")
    route = filedrop.make_router([filedrop.DropRule(matches=lambda p: os.path.basename(p).startswith("special"),
                                                    handler=lambda paths: seen.append(("narrow", paths)),
                                                    label="special images"),
                                  filedrop.DropRule(matches=filedrop.by_extension(".png"),
                                                    handler=lambda paths: seen.append(("broad", paths)),
                                                    label="images")],
                                 reference_window="w", on_rejected=collector)
    route([narrow])
    route([broad])
    assert seen == [("narrow", [narrow]), ("broad", [broad])]


def test_a_drop_is_ignored_while_a_modal_is_open(tmp_path, collector):
    """Not rejected — ignored. Reporting would stack a second modal on the one already asking a question.

    The OS drop lands on the window, not on whatever DPG is drawing inside it, so this arrives even while a
    file dialog has the app's attention.
    """
    seen = []
    modal_open = True
    route = filedrop.make_router([filedrop.DropRule(matches=filedrop.by_extension(".bib"),
                                                    handler=seen.append,
                                                    label="BibTeX files")],
                                 reference_window="w", blocked=lambda: modal_open, on_rejected=collector)
    paths = make_files(tmp_path, "a.bib")
    route(paths)
    assert seen == []
    assert collector.rejections == []  # silence, not a dialog

    modal_open = False
    route(paths)
    assert seen == [paths]


def test_an_empty_drop_does_nothing(collector):
    seen = []
    route = filedrop.make_router([filedrop.DropRule(matches=lambda p: True,
                                                    handler=seen.append,
                                                    label="anything")],
                                 reference_window="w", on_rejected=collector)
    route([])
    assert seen == []
    assert collector.rejections == []


def test_many_dropped_names_are_abridged_in_the_dialog(tmp_path, collector):
    route = filedrop.make_router([filedrop.DropRule(matches=filedrop.by_extension(".bib"),
                                                    handler=lambda paths: None,
                                                    label="BibTeX files")],
                                 reference_window="w", on_rejected=collector)
    route(make_files(tmp_path, *[f"f{i}.txt" for i in range(12)]))
    _title, message = collector.rejections[0]
    assert "and 4 more" in message  # 12 dropped, 8 listed


# --------------------------------------------------------------------------------
# Installation guards

def test_install_refuses_off_the_render_thread():
    """GLFW's current context is per-thread, so a background thread cannot reach the window handle.

    Refusing loudly matters more than it looks: `glfwGetCurrentContext` would return NULL there, and
    installing against a NULL window is how this becomes a segfault instead of a log line.
    """
    result = {}
    t = threading.Thread(target=lambda: result.update(installed=filedrop.install(lambda paths: None)))
    t.start()
    t.join()
    assert result["installed"] is False


def test_availability_is_answerable_without_a_context():
    """`is_available` is a capability question, so it must not need a viewport — apps call it while deciding what to log."""
    assert isinstance(filedrop.is_available(), bool)


@pytest.mark.gui
def test_install_needs_show_viewport_and_succeeds_right_after_it():
    """Pins *when* DPG's window becomes reachable, which is what fixes where every app installs the handler.

    All six GUI apps install immediately after `dpg.show_viewport()`, on the strength of a measurement:
    GLFW's current context is NULL through `create_context`, `create_viewport` and `setup_dearpygui`, and
    non-NULL from `show_viewport()` on — no rendered frame required. Nothing documents that, so a DPG
    upgrade could move it, and the symptom would be drag-and-drop silently doing nothing everywhere at
    once. This fails instead.

    Marked `gui`: `show_viewport` maps a real window and takes keyboard focus for a moment. The
    complementary case — that installing off the render thread is refused — needs no window and is tested
    above.
    """
    dpg.create_context()
    try:
        dpg.create_viewport(title="raven filedrop install test", width=320, height=200)
        dpg.setup_dearpygui()
        with dpg.window(tag="main"):  # tag
            dpg.add_text("installing an OS file drop handler")
        dpg.set_primary_window("main", True)  # tag

        assert filedrop.install(lambda paths: None) is False, "before show_viewport there is no window to install against"

        dpg.show_viewport()
        assert filedrop.install(lambda paths: None) is True, "show_viewport must be enough; apps do not render a frame first"
    finally:
        filedrop.uninstall()
        dpg.destroy_context()
