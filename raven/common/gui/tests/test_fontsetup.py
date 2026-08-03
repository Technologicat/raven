"""Pins that Raven's font setup runs clean on the DPG version it declares.

DPG 2.3 made font atlas character ranges automatic and turned `add_font_range` into a deprecated no-op, so
every Raven GUI app opened with four `DeprecationWarning`s on stderr that named a call the reader could do
nothing about — and which no longer did anything either. The calls are gone, and `pyproject.toml` requires
`dearpygui>=2.3` so nothing has to configure ranges by hand.

What is asserted is the *absence*: loading the default font, the icon fonts and the Markdown renderer's font
must warn about nothing. A regression here is silent by nature — an app that warns on every start still runs
correctly, so the only thing lost is a reader's willingness to look at the log at all. That is what the
warning-free start is protecting, and it is why this is worth a test rather than a one-time cleanup.

No window is mapped: font loading needs a DPG context but not a rendered frame, so these run in the ordinary
suite rather than behind `--run-gui`.
"""

import warnings

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed (GUI toolkit absent in CI)")

from raven.common.gui import utils as guiutils  # noqa: E402 -- after importorskip by design


@pytest.fixture
def dpg_context():
    """A DPG context with an unmapped viewport, fresh per test so the font registry starts empty."""
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


def test_bootup_loads_fonts_without_warning(dpg_context):
    """The whole common startup path — default font, icon fonts, Markdown fonts, themes."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        themes_and_fonts = guiutils.bootup(font_size=20)
    assert [str(warning.message) for warning in caught] == []
    assert themes_and_fonts.font_registry


def test_load_extra_font_is_quiet_and_caches(dpg_context):
    """The on-demand path, used for fonts a single widget needs (Librarian's pill takes InterTight this way)."""
    themes_and_fonts = guiutils.bootup(font_size=20)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        key, font = guiutils.load_extra_font(themes_and_fonts, font_size=20,
                                             font_basename="InterTight", variant="Regular")
    assert [str(warning.message) for warning in caught] == []

    # Same request, same font: the key is the cache key, so a second call must not build a second atlas entry.
    assert guiutils.load_extra_font(themes_and_fonts, font_size=20,
                                    font_basename="InterTight", variant="Regular") == (key, font)
