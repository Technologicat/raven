"""Project-wide pytest configuration.

Two jobs, both about the tests that map a real GUI window.

Most of Raven's DPG tests never show their viewport, so they cost nothing and run with everything else (see
`raven/common/gui/tests/`). A few invariants cannot be checked that way — anything about layout, scrolling
extents or keyboard focus needs frames to actually render, and `dpg.render_dearpygui_frame()` aborts the
process outright when there is no window to render into. Those tests therefore map a window, which on a
normal desktop takes keyboard focus away from whatever the developer is typing into.

That is a surprise worth requiring consent for, so it is opt-in rather than opt-out: they are collected
always and skipped unless `--run-gui` is passed, so a bare `pytest` reports their existence without ever
grabbing the screen.

The second job is `mapped_gui_context`, the one mapped context those tests share.
"""

import pytest


def pytest_addoption(parser) -> None:
    parser.addoption("--run-gui",
                     action="store_true",
                     default=False,
                     help="Run tests that map a real GUI window. These take keyboard focus while they run.")
    parser.addoption("--run-llm",
                     action="store_true",
                     default=False,
                     help="Run tests that talk to the configured live LLM backend. They connect to it.")
    # Spelled as `raven-librarian`, `raven-minichat` and `raven-pdf2bib` spell it. The configured value is
    # already per-machine, so the `llm` tests need this only when the two disagree — pointing at a backend on
    # another host for an afternoon. Passing it also opts in, there being no reason to name one otherwise.
    parser.addoption("--backend-url",
                     action="store",
                     default=None,
                     metavar="URL",
                     help="OpenAI-compatible LLM backend for the `llm` tests. "
                          "Default: raven.librarian.config.llm_backend_url.")


def pytest_collection_modifyitems(config, items) -> None:
    # The `llm` tests are opt-in for a different reason than the `gui` ones: not focus, but the outbound
    # connection. Left to run by default they would have a CI runner open a socket to whatever the committed
    # `llm_backend_url` names — `localhost:1234` — on a machine we do not control. That a refused connection
    # is the near-certain outcome is beside the point: a test suite has no business connecting anywhere its
    # operator did not ask it to, and "we checked and it is probably nothing" is not a property anyone can
    # verify from the outside. Naming a backend counts as asking, so `--backend-url` opts in on its own.
    if not (config.getoption("--run-llm") or config.getoption("--backend-url")):
        skip_llm = pytest.mark.skip(reason="talks to a live LLM backend; pass --run-llm to use the configured "
                                           "one, or --backend-url URL to name another")
        for item in items:
            if item.get_closest_marker("llm") is not None:
                item.add_marker(skip_llm)

    if config.getoption("--run-gui"):
        # The shared context has to be the last one alive in the process. Almost every other DPG test builds
        # a context of its own and destroys it, and doing that while `mapped_gui_context` is up segfaults
        # inside `setup_dearpygui` — measured 2026-08-24, `test_fontsetup` dying at 10% of the group with
        # `filedrop._dispatch_loop` still on the stack. Since a session fixture lives from first use to the
        # end of the run, "not alive yet" is the only state that can be arranged, so every test that takes
        # it sorts to the very end. `sort` is stable, so nothing else moves.
        #
        # This does not reach `test_filedrop`'s two `gui` tests, which take no context fixture: they create
        # and destroy their own, that being what they are testing. They keep their place, ahead of the point
        # where the shared context comes up.
        shares_context = [item for item in items
                          if "mapped_gui_context" in getattr(item, "fixturenames", ())]
        owns_context = [item for item in items
                        if item.get_closest_marker("gui") is not None and item not in shares_context]

        # A DPG context appears not to survive being the second one in a process — whatever GLFW and ImGui
        # keep between `glfwInit` and `glfwTerminate` is process-global, and the mechanism has never been
        # identified (`dpg-notes.md`, "Context recreation is not reliably safe once real widgets have
        # rendered"). Measured 2026-08-24: `test_filedrop`'s two lifecycle tests pass alone (18 passed)
        # and, run before the rest, take down the next context created anywhere in the process — including
        # unmapped ones, `test_fontsetup` dying inside `setup_dearpygui`. Run after, they die in their own
        # `setup_dearpygui` instead, the shared context still being up. There is no position for them.
        #
        # So they run in a process of their own, and are skipped only when something they would break is
        # present. Targeting the file directly still runs them, which is what the skip reason says.
        if owns_context and shares_context:
            skip_owning = pytest.mark.skip(reason="creates and destroys DPG contexts, which no other context "
                                                  "in the process survives; run it alone: "
                                                  "pytest --run-gui raven/common/gui/tests/test_filedrop.py")
            for item in owns_context:
                item.add_marker(skip_owning)

        # The shared context lives from first use to the end of the run, so it is brought up as late as
        # possible: every test that takes it sorts last. `sort` is stable, so nothing else moves.
        items.sort(key=lambda item: item in shares_context)
        return
    skip_gui = pytest.mark.skip(reason="maps a real window and takes keyboard focus; pass --run-gui to run")
    for item in items:
        # `get_closest_marker`, not `"gui" in item.keywords`: keywords also carry the names of every parent
        # collector, and these tests live under `raven/common/gui/`, so the directory name alone matched all
        # fifty of its unmapped-viewport tests and skipped them too.
        if item.get_closest_marker("gui") is not None:
            item.add_marker(skip_gui)


GUI_VIEWPORT_TITLE = "raven gui tests"


@pytest.fixture(scope="session")
def mapped_gui_context():
    """One mapped DPG context, shared by every `gui` test that just needs a window to render into.

    Yields the viewport's title, which is what a test driving synthetic input searches for to find the
    window. Taking it from here rather than repeating the string is what keeps such a test from quietly
    *skipping* — `xdotool search` matching nothing is indistinguishable from the tools being absent.
    """
    # Session-scoped because the churn is what breaks, not the mapping. DPG keeps its context in
    # process-global state, and `destroy_context` leaves GLFW holding callbacks belonging to it: a focus
    # event arriving afterwards reaches a freed backend and takes the process down inside `glfwPollEvents`,
    # under `ImGui_ImplGlfw_WindowFocusCallback`. A context per test therefore turns the whole group red as
    # soon as a second one is mapped, wherever in the collection that happens to fall. One context for the
    # session removes the churn rather than reordering it.
    #
    # 1280x800 is DPG's own default, and the size is load-bearing rather than cosmetic: ImGui only pulls an
    # offscreen window back inside the viewport when there is an inside to pull it to, so a viewport smaller
    # than the windows drawn into it leaves the clamping tests unable to tell a clamp from a no-op.
    #
    # `dearpygui` is imported here rather than at module scope so that a bare `pytest` — which skips every
    # `gui` test and never builds this fixture — does not need the toolkit installed at all.
    import dearpygui.dearpygui as dpg

    dpg.create_context()
    dpg.create_viewport(title=GUI_VIEWPORT_TITLE, width=1280, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    yield GUI_VIEWPORT_TITLE
    dpg.destroy_context()
