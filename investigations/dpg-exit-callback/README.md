# When DearPyGui runs the exit callback

**The question:** which ways of ending a render loop run the callback registered with
`dpg.set_exit_callback`, and when? Asked so that Librarian's log could say *why* its window went away, after
it vanished mid-session at an event (2026-10-08) and the log captured from the terminal could not tell.

**The answer, measured 2026-10-09 on dearpygui 2.3.1, Linux/X11:** all three ways out run it, and they all run
it at the same point, which is **during `dpg.destroy_context()`**. Neither the render loop ending nor the
final frame triggers it.

| how the loop ended | callback ran | when |
|---|---|---|
| `dpg.stop_dearpygui()` from the render loop, on the main thread | yes | inside `destroy_context` |
| SIGTERM, handled by calling `dpg.stop_dearpygui()` (what `raven.common.quitsignal` installs) | yes | inside `destroy_context` |
| the window manager closing the window (`wmctrl -i -c`) | yes | inside `destroy_context` |

The probe sleeps between the loop's exit and `destroy_context`, and the callback arrived exactly that long
after the loop ended — 1.0 s by default, and 2.5 s with `PROBE_DELAY=2.5`. It runs on DPG's callback thread,
not on the main thread.

## What follows

- **The exit callback cannot say why the app is closing.** It looks the same for every cause, so a log line
  that names the cause has to come from the paths that know it: the loop exiting normally (a window close, or
  the app's own `stop_dearpygui`), an exception propagating out of the loop, a signal handler.
- **It runs after anything in the loop's `finally` that comes before `destroy_context`.** An app whose
  `finally` already drives its own teardown, as Raven's do, gets the exit callback a second time, late,
  against a context being destroyed.
- **The callback's output is not a timestamp for "the loop stopped".** A log line written from it appears
  after the app's whole teardown, which is the misreading this probe was written to settle.

The probe's minimal app has no callbacks queued and no frame callbacks in flight. A busy app's callback
thread may behave differently, which this does not measure.

## Script

- `probe_exit_callback.py MODE` — maps a small window titled `exit-callback-probe` (it takes keyboard focus)
  and ends it by `MODE`: `stop`, `sigterm` or `wmclose`. For `wmclose` the caller closes the window, e.g.
  `wmctrl -i -c $(wmctrl -l | awk '/exit-callback-probe/ {print $1; exit}')` about a second after launch.
  Prints each event with its time and thread, and ends with a `RESULT:` line. `PROBE_DELAY` sets the pause
  before `destroy_context`.
