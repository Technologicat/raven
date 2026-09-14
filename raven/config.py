"""Global configuration for the Raven constellation.

Some components also have their own configurations, which see:

  - client.config
  - server.config
  - cherrypick.config
  - conference_timer.config
  - librarian.config
  - papers.config
  - visualizer.config
  - xdot_viewer.config
"""

import pathlib

from . import configoverrides

# Used for various things. E.g. the web API keys go here.
toplevel_userdata_dir = "~/.config/raven/"

# Convert to an absolute path, just once here.
toplevel_userdata_dir = pathlib.Path(toplevel_userdata_dir).expanduser().resolve()


# ---------------------------------------------------------------------------
# Idle throttle
# ---------------------------------------------------------------------------
#
# Every GUI app in the constellation runs its own render loop, and every one of them drops to a low frame
# rate while nothing is happening. The two numbers below are what that means, for all of them at once:
# `raven.common.gui.utils.sleep_until_next_frame` does the arithmetic, and each app decides for itself what
# counts as "something is happening".
#
# One pair of numbers rather than one pair per app, because they answer one question — how fast should an
# idle Raven window redraw — and an app that wanted its own answer would be saying something about itself
# that none of them has ever had to say.

GUI_IDLE_FRAMERATE = 12    # frames per second, while the app is idle
GUI_INPUT_ACTIVE_S = 0.5   # stay at full frame rate for this long after the last user input


# ---------------------------------------------------------------------------
# Window sizes for the full-size GUI apps
# ---------------------------------------------------------------------------
#
# Raven-librarian, Raven-visualizer, Raven-cherrypick and Raven-xdot-viewer: four apps that open a window
# meant to be worked in all day, and a help card in front of it. One set of numbers rather than one per
# app, for the reason the idle throttle above has one — they answer one question, and an app wanting its
# own answer would be saying something about itself that none of them has to say.
#
# **Not every app is one of the four.** The conference timer is a full-screen countdown with four keys, and
# the two avatar editors are laid out for windows of their own size; each keeps its own numbers, and the
# file dialog sizes its card to the dialog it belongs to.

GUI_MAIN_WINDOW_W = 1920
GUI_MAIN_WINDOW_H = 1040   # this pair just fits onto a 1080p screen in Linux Mint, taskbar included

# The help card floating in front of it. Wide enough that three column-groups of hotkeys need no
# telegraphic phrasing to fit: the house style treats a dropped article as a concession bought with
# horizontal space, so the width is what stops that concession being necessary.
#
# **The height is a starting value, not a size.** A card of two or more pages measures its tallest page and
# fits itself to it (`HelpWindow._fit_height_to_pages`), clamping to the viewport and logging if it has to.
# A *single*-page card gets no such fitting: it keeps the height it was given and silently clips whatever
# does not fit, which is how Raven-cherrypick's card came to omit its own `F1` row.
GUI_HELP_WINDOW_W = 1700
GUI_HELP_WINDOW_H = 1000


# ---------------------------------------------------------------------------
# The reference clock for time-varying effects
# ---------------------------------------------------------------------------
#
# Every animated quantity in the constellation is tuned *per frame at `CALIBRATION_FPS`*, and corrected at
# run time to whatever frame rate is actually achieved — a GUI animation's rate, the avatar's pose
# interpolator step and blink probability, a video postprocessor effect's durations. So this is the unit
# those numbers are quoted in, and re-exported here because this is where a reader looks for it.
#
# **Changing it redefines the unit rather than reconfiguring anything.** Nothing anywhere assumes the
# number: every consumer uses it as a ratio (`avg_fps / CALIBRATION_FPS`, `CALIBRATION_FPS * seconds`), so
# a different value is self-consistent — but every quantity that was *tuned* against it would then need
# re-tuning by the same factor, which is a day's work with a video capture and an eye, not an edit. It is
# here to make the assumption discoverable by anyone willing to take that on, not to invite it.
CALIBRATION_FPS = 25


# Machine-local overrides (`~/.config/raven/overrides.json`); applied last, so they can name anything above.
configoverrides.apply(__name__, globals())
