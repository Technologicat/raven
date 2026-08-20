"""The blue that says where the keyboard is, shared by every widget that draws it.

Two different facts wear this colour, and they are deliberately not distinguished by it:

- **Where the cursor is** — the entry Enter would act on. Drawn as text colour by a table
  (`tablecursor`), as an inner border by a grid (`thumbnailgrid`).
- **Where the caret is parked** — which control the arrow keys are driving, in a GUI that routes them by
  hand because DPG draws nothing on a focused combo, a listing, or a panel of its own.

One hue and one rhythm across both, because they are one idea seen from two sides: *the keyboard is here*.
A reader glancing at any Raven app should recognize it without being taught, which is also why this lives in
`raven.common.gui` rather than in whichever widget happened to need it first — the values were `thumbnailgrid`'s
until the file dialog became the second caller and a combo mark was about to be the third, at which point a
grid would have been dictating the colour of a combo's border in an app with no grid in it.

The *drawing* is each widget's own: a theme colour where one reaches, a drawn rectangle where none does.
Only the vocabulary is shared. `raven.common.gui.animation.PulsatingColor` breathes a theme colour at
`PULSE_SECONDS`; a widget that draws its own mark ticks `pulsating_alpha` against the same period, which is
what keeps two marks on one screen in step.

This module is licensed under the 2-clause BSD license, to facilitate integration anywhere.
"""

__all__ = ["COLOR", "PULSE_SECONDS"]

# Names are bare because the module supplies the namespace: `keyboardmark.COLOR` at a call site, rather
# than a `KEYBOARD_MARK_` prefix repeated inside the module that already says it.
#
# **Which makes the import style load-bearing here rather than merely preferred.** Import the module —
# `from ..gui import keyboardmark` — and never the names inside it: `from ..gui.keyboardmark import COLOR`
# leaves a bare `COLOR` at every use site, saying nothing about *which* colour it is or why that widget
# should be wearing it. The bare names are readable only while the namespace is still attached to them, so
# a module whose public names are deliberately short is the one place that rule cannot be relaxed.
COLOR = (80, 160, 255, 255)

# How long one breath takes, in seconds. Shared for the same reason the colour is: two marks pulsating at
# different rates read as two things blinking at each other rather than as one mark meaning one thing.
PULSE_SECONDS = 2.0
