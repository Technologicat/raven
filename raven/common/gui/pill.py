"""A status pill: a short coloured message with an icon, which is also a button.

What Raven's apps use to say something about their state in one line — the LLM backend is not connected, not
every record is shown — and offer the obvious thing to do about it with a click. A pill is an icon, a button
carrying the label, and a tooltip on the button for the longer explanation:

    p = pill.Pill(tag="backend_status", icon_font=themes_and_fonts.icon_font_solid,
                  callback=lambda: reconnect())
    p.set(label="LLM backend not connected", icon=fa.ICON_PLUG_CIRCLE_XMARK,
          tooltip="Nothing answers at http://localhost:5000.", kind=pill.ALERT)
    p.shown = True

Three kinds, which differ only in colour:

  - `OK`: green, steady. Something has come right; typically shown for a moment, then taken down.
  - `ALERT`: the caution colour, the icon pulsing. Something wants acting on.
  - `NOTICE`: the caution colour, steady. Read this, but nothing is wrong.

The label never pulses, as a sentence is too long to read inside one cycle. The themes and the pulsation are
shared by every pill in the app, so two pills alerting side by side pulse in phase.

Where the pill goes, what the click does and what the captions say are the app's.
"""

__all__ = ["OK", "ALERT", "NOTICE", "KINDS",
           "Pill"]

import dearpygui.dearpygui as dpg

from . import animation as gui_animation
from . import tooltip as gui_tooltip
from . import utils as guiutils

OK = "ok"
ALERT = "alert"
NOTICE = "notice"
KINDS = (OK, ALERT, NOTICE)

_OK_COLOR = (96, 224, 96)
_PULSE_CYCLE_DURATION = 2.0  # seconds, the period the apps' other glows use

# String tags, so that `_ensure_themes` can ask whether they exist in the current DPG context.
_OK_THEME = "raven_pill_ok_theme"  # tag
_CAUTION_THEME = "raven_pill_caution_theme"  # tag
_PULSING_CAUTION_THEME = "raven_pill_pulsing_caution_theme"  # tag

_pulsator = None  # the one `PulsatingColor` driving `_PULSING_CAUTION_THEME`


def _ensure_themes() -> None:
    """Build the shared pill themes, unless the current DPG context already has them."""
    global _pulsator
    if dpg.does_item_exist(_OK_THEME):  # tag
        return
    with dpg.theme(tag=_OK_THEME):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, _OK_COLOR)
    with dpg.theme(tag=_CAUTION_THEME):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, guiutils.CAUTION_COLOR)
    with dpg.theme(tag=_PULSING_CAUTION_THEME):
        with dpg.theme_component(dpg.mvAll):
            pulsing_color = dpg.add_theme_color(dpg.mvThemeCol_Text, guiutils.CAUTION_COLOR)
    # A previous context's pulsation would go on driving a theme colour that no longer exists.
    if _pulsator is not None:
        gui_animation.animator.cancel(_pulsator, finalize=False)
    _pulsator = gui_animation.animator.add(gui_animation.PulsatingColor(cycle_duration=_PULSE_CYCLE_DURATION,
                                                                         theme_color_widget=pulsing_color))


# (icon theme, label theme) per kind
_THEMES = {OK: (_OK_THEME, _OK_THEME),
           ALERT: (_PULSING_CAUTION_THEME, _CAUTION_THEME),
           NOTICE: (_CAUTION_THEME, _CAUTION_THEME)}


class Pill:
    def __init__(self, *, tag: str, icon_font: int | str, callback, show: bool = False):
        """Build a pill in the current DPG container, hidden unless `show`.

        `tag`: names its widgets: the group `"{tag}_pill"`, the icon `"{tag}_icon"` and the button
               `"{tag}_button"`, so app code can name them for placement or a flash.
        `icon_font`: the DPG font the icons are drawn in, typically `themes_and_fonts.icon_font_solid`.
        `callback`: called with no arguments when the pill is clicked.

        Shows nothing until `set` gives it a label.
        """
        _ensure_themes()
        self.group = f"{tag}_pill"
        self.icon = f"{tag}_icon"
        self.button = f"{tag}_button"
        # The icon is a text widget, and only the label is the button: DPG draws an item's label in one font,
        # and the app font has no icons. So the icon's few pixels at the left do not respond to a click.
        with dpg.group(tag=self.group, horizontal=True, show=show):
            dpg.add_text("", tag=self.icon, show=False)
            dpg.bind_item_font(self.icon, icon_font)
            dpg.add_button(label="", tag=self.button, callback=lambda: callback())
        # Self-sizing: a pill's caption is rewritten whenever the state it describes moves, and a `dpg.tooltip`
        # is drawn once at its previous size each time that happens.
        self.tooltip = gui_tooltip.Tooltip(self.button, "")
        self.kind = None

    def set(self, *, label: str, kind: str, icon: str | None = None, tooltip: str = "") -> None:
        """Say `label`, coloured as `kind` (one of `KINDS`), with `icon` (a glyph of the icon font) before it,
        or no icon if `None`. `tooltip`: the longer explanation, shown on hover.

        Does not show or hide the pill; that is `shown`.

        A click flash in progress survives it: the tooltip text becomes what the flash restores
        (`raven.common.gui.animation.set_text_under_flash`), and the colours are rebound only when `kind` changes.
        """
        if kind not in KINDS:
            raise ValueError(f"Pill.set: unknown kind {kind!r}; known: {KINDS}")
        dpg.configure_item(self.button, label=label)
        if icon is None:
            dpg.hide_item(self.icon)
        else:
            dpg.set_value(self.icon, icon)
            dpg.show_item(self.icon)
        gui_animation.set_text_under_flash(self.tooltip, tooltip)
        # Only when the kind changes: a click flash binds a theme of its own to the button for as long as it
        # lasts, and rebinding the button's would end it early. A pill re-set with every poll of what it
        # reports would otherwise never show its flash.
        if kind != self.kind:
            icon_theme, label_theme = _THEMES[kind]
            dpg.bind_item_theme(self.icon, icon_theme)
            dpg.bind_item_theme(self.button, label_theme)
            self.kind = kind

    def _get_shown(self) -> bool:
        """Whether the pill is shown: its own visibility, not whether it is on screen."""
        return bool(dpg.get_item_configuration(self.group)["show"])

    def _set_shown(self, shown: bool) -> None:
        """Show or hide the pill."""
        (dpg.show_item if shown else dpg.hide_item)(self.group)

    shown = property(fget=_get_shown, fset=_set_shown,
                     doc="Whether the pill is shown. Assign to show or hide it; reading answers what was last set.")
