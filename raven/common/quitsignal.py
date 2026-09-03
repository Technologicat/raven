"""Make a `SIGTERM` ask the app to shut down, instead of killing it where it stands.

An app that persists at exit — through `atexit`, or through the teardown at the end of its render loop —
gets nothing from a signal that terminates it at the C level. Raven's Librarian writes its chat datastore
once, at clean exit, so a plain `kill` loses the whole session; the same signal orphans the server-side
avatar instance it would otherwise have released.

**The signal was not even reaching the process, and the reason is worth knowing before touching this.**
SDL installs its own `SIGINT` and `SIGTERM` handlers when it initializes, and Raven initializes SDL
wherever it plays audio (`raven.common.audio.player` calls `pygame.mixer.init`). SDL's handler pushes a
quit event onto the SDL event queue, which suits an SDL application with an SDL event loop and is inert
here: Raven uses SDL for audio only and never pumps that queue. So `SIGTERM` was caught and discarded, and
`kill` did nothing at all — measured 2026-09-03, and consistent with the report from 2026-08-04 that a
`kill` left the process alive and silent.

That is why `player` sets `SDL_NO_SIGNAL_HANDLERS` before importing pygame, and why the fix is not simply
to install a handler after SDL: the mixer can be re-initialized at runtime to change the sample rate, and
SDL reinstalls its handlers every time it does.

`SIGINT` needs nothing from this module. CPython's own handler raises `KeyboardInterrupt`, SDL does not
displace it (checked), and the render loops already catch it.
"""

__all__ = ["DEFAULT_SIGNALS",
           "install"]

import logging
import signal
import threading
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

# What a session manager, a supervisor, a logout and a plain `kill` send — of those this platform has.
# Built by lookup rather than written out, because `SIGHUP` does not exist on Windows and naming it in a
# default argument is enough to fail the *import*, which would take the whole app down on that platform
# rather than one signal. (It did, on CI, 2026-09-03.)
DEFAULT_SIGNALS = tuple(found for found in (getattr(signal, name, None)
                                            for name in ("SIGTERM", "SIGHUP"))
                        if found is not None)


def install(on_quit: Callable[[], None],
            signals: Sequence[signal.Signals] = DEFAULT_SIGNALS) -> None:
    """Ask `on_quit` to be called when the process is asked to terminate.

    `on_quit`: What to do about it. **Called from a signal handler**, so it must be cheap and must not
               wait for anything: set a flag, or ask the render loop to stop. A GUI app passes
               `dpg.stop_dearpygui`, which lets its ordinary teardown run on the way out of the loop.
    `signals`: Which signals to treat as "please stop". Defaults to `DEFAULT_SIGNALS`, which see.

    Must be called from the main thread — `signal.signal` accepts no other — and after anything that
    initializes SDL, or SDL's handlers replace these. In practice: at app start, before the render loop.

    Does nothing but log if a signal cannot be installed, which is the right failure: an app that cannot
    catch `SIGTERM` should still start.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("quitsignal.install: must be called from the main thread; "
                           "`signal.signal` accepts no other.")

    def handle(signum, frame) -> None:
        # Python runs this between bytecodes on the main thread, which is parked in the render loop's
        # frame call -- so it lands between frames, where calling into the GUI toolkit is as safe as it is
        # from the loop body itself.
        logger.info(f"quitsignal: {signal.Signals(signum).name} received; asking the app to shut down.")
        on_quit()

    for signum in signals:
        try:
            signal.signal(signum, handle)
        except (OSError, ValueError, RuntimeError) as exc:  # not every signal exists on every platform
            logger.warning(f"quitsignal.install: cannot handle {signum}: {type(exc)}: {exc}")
