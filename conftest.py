"""Project-wide pytest configuration.

Currently one job: keep the tests that map a real GUI window out of an ordinary `pytest` run.

Most of Raven's DPG tests never show their viewport, so they cost nothing and run with everything else (see
`raven/common/gui/tests/`). A few invariants cannot be checked that way — anything about layout, scrolling
extents or keyboard focus needs frames to actually render, and `dpg.render_dearpygui_frame()` aborts the
process outright when there is no window to render into. Those tests therefore map a window, which on a
normal desktop takes keyboard focus away from whatever the developer is typing into.

That is a surprise worth requiring consent for, so it is opt-in rather than opt-out: they are collected
always and skipped unless `--run-gui` is passed, so a bare `pytest` reports their existence without ever
grabbing the screen.
"""

import pytest


def pytest_addoption(parser) -> None:
    parser.addoption("--run-gui",
                     action="store_true",
                     default=False,
                     help="Run tests that map a real GUI window. These take keyboard focus while they run.")


def pytest_collection_modifyitems(config, items) -> None:
    if config.getoption("--run-gui"):
        return
    skip_gui = pytest.mark.skip(reason="maps a real window and takes keyboard focus; pass --run-gui to run")
    for item in items:
        # `get_closest_marker`, not `"gui" in item.keywords`: keywords also carry the names of every parent
        # collector, and these tests live under `raven/common/gui/`, so the directory name alone matched all
        # fifty of its unmapped-viewport tests and skipped them too.
        if item.get_closest_marker("gui") is not None:
            item.add_marker(skip_gui)
