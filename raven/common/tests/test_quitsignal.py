"""Unit tests for raven.common.quitsignal — turning a termination signal into a clean shutdown.

The interesting half is not that a handler can be installed; it is *what it is competing with*. SDL
installs handlers of its own the moment audio initializes, and they discard the signal for an app that
does not pump the SDL event queue — which Raven does not, using SDL for audio alone. So there is a
characterization test here for a third-party library's behaviour, of the same kind as the one pinning
DearPyGui's focus model: if pygame ever stops doing this, the workaround in `player` becomes dead weight
and we should find out from a red test rather than never.
"""

import os
import pathlib
import signal
import threading

import pytest

from raven.common import quitsignal


# Delivering a signal to ourselves is POSIX-only: `os.kill` on Windows cannot deliver `SIGTERM`, it calls
# `TerminateProcess`, so a test that tried would end the run rather than fail it.
posix_only = pytest.mark.skipif(os.name != "posix", reason="needs real signal delivery to self")

# And asking what handlers a *C library* installed needs the kernel's own view, which is Linux's `/proc`.
# `signal.getsignal` cannot answer it — it reports what Python installed, and the whole question here is
# what SDL did behind Python's back.
_PROC_STATUS = pathlib.Path(f"/proc/{os.getpid()}/status")
needs_proc = pytest.mark.skipif(not _PROC_STATUS.exists(),
                                reason="reads the kernel's signal mask from /proc (Linux)")


@pytest.fixture
def restore_signal_handlers():
    """Put back whatever handlers were installed, so a test cannot leak one into the rest of the run."""
    saved = {signum: signal.getsignal(signum) for signum in quitsignal.DEFAULT_SIGNALS}
    yield
    for signum, handler in saved.items():
        signal.signal(signum, handler)


def caught_signals() -> set:
    """Return the signals this process has handlers installed for, read from the kernel's own view."""
    for line in _PROC_STATUS.read_text().splitlines():
        if line.startswith("SigCgt:"):
            mask = int(line.split()[1], 16)
            return {s for s in signal.Signals if mask & (1 << (s.value - 1))}
    return set()


class TestItAsksTheAppToStop:
    @posix_only
    def test_the_signal_reaches_the_callback(self, restore_signal_handlers):
        asked = []
        quitsignal.install(lambda: asked.append(True))
        os.kill(os.getpid(), signal.SIGTERM)
        assert asked == [True], "SIGTERM did not reach the callback"

    @needs_proc
    def test_the_default_would_have_killed_us(self, restore_signal_handlers):
        # The negative control, and the reason this module exists: without the handler the same signal
        # terminates the process outright, so nothing that runs at exit runs. Asserted on the disposition
        # rather than by dying, for obvious reasons.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        assert signal.SIGTERM not in caught_signals(), \
            "something else is already catching SIGTERM, so this fixture proves nothing"
        quitsignal.install(lambda: None)
        assert signal.SIGTERM in caught_signals()

    def test_it_refuses_to_install_off_the_main_thread(self):
        # `signal.signal` raises there anyway; this turns a confusing ValueError from inside the stdlib
        # into a sentence naming the actual constraint.
        failed = []

        def try_it():
            try:
                quitsignal.install(lambda: None)
            except RuntimeError as exc:
                failed.append(str(exc))

        thread = threading.Thread(target=try_it)
        thread.start()
        thread.join()
        assert failed and "main thread" in failed[0]

    @needs_proc
    def test_a_signal_the_platform_lacks_is_survivable(self, restore_signal_handlers):
        # An app that cannot catch a signal should still start. Nothing here is more important than that.
        quitsignal.install(lambda: None, signals=(signal.SIGTERM, signal.SIGKILL))  # SIGKILL cannot be caught
        assert signal.SIGTERM in caught_signals(), "the survivable one was skipped along with the fatal one"

    def test_the_default_list_holds_only_signals_this_platform_has(self):
        # The Windows lesson, pinned: naming `SIGHUP` in a default argument is enough to fail the module's
        # *import* there, which takes the app down rather than one signal. Portable on purpose — this is
        # the test that has to run on the platform it is about.
        assert quitsignal.DEFAULT_SIGNALS, "no signals at all; a termination would go unhandled everywhere"
        for signum in quitsignal.DEFAULT_SIGNALS:
            assert getattr(signal, signum.name, None) is signum


_RENDER_LOOP_UNDER_SIGTERM = r'''
import os, signal, sys
import dearpygui.dearpygui as dpg
from raven.common import quitsignal

dpg.create_context()
dpg.create_viewport(width=200, height=100, title="raven quitsignal test")
dpg.setup_dearpygui()
dpg.show_viewport()   # `is_dearpygui_running` asks GLFW about a window, and asserts if there is none
quitsignal.install(dpg.stop_dearpygui)

frames = 0
try:
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        frames += 1
        if frames == 3:                      # a signal from outside would race the startup
            os.kill(os.getpid(), signal.SIGTERM)
        if frames > 600:                     # ten seconds at 60 fps; the loop never stopped
            print("LOOP-DID-NOT-STOP")
            sys.exit(2)
finally:
    print("TEARDOWN-RAN")                    # what an app saves its work in
dpg.destroy_context()
print(f"STOPPED-AFTER {frames}")
'''


@posix_only  # the subprocess signals itself, which on Windows would terminate it instead of asking
@pytest.mark.gui  # it maps a window for a fraction of a second, which takes focus like any other
def test_sigterm_leaves_the_render_loop_and_runs_the_teardown():
    """The whole point, end to end: a `kill` has to come out where the close button comes out.

    In a subprocess because it needs a real DearPyGui context and a real signal, and because the thing
    being asserted is that the process *ends* — neither of which a test sharing this interpreter can
    survive. Also the only place the loop-exit half is covered: the unit tests above stop at the callback.

    It needs a *shown* viewport, which is what puts it in the `gui` group. `is_dearpygui_running` asks
    GLFW whether the window should close, and with no window it does not return False — it fails an
    assertion and takes the process with it.
    """
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c", _RENDER_LOOP_UNDER_SIGTERM],
                            capture_output=True, text=True, timeout=120,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert "LOOP-DID-NOT-STOP" not in result.stdout, "SIGTERM did not end the render loop"
    assert "TEARDOWN-RAN" in result.stdout, "the loop ended without running its teardown"
    assert result.returncode == 0, f"exited {result.returncode}; stderr:\n{result.stderr}"


@pytest.mark.ml  # pygame comes with the audio stack rather than with the test subset
class TestWhatSDLDoesToOurSignals:
    """Characterizing pygame/SDL, because the fix in `player` is shaped entirely by it."""

    @needs_proc
    def test_the_sdl_hint_keeps_our_signals_ours(self, restore_signal_handlers):
        # `player` sets `SDL_NO_SIGNAL_HANDLERS` before importing pygame; importing it here therefore
        # exercises the shipped configuration rather than a hypothetical one.
        from raven.common.audio import player  # noqa: F401 -- imported for its import-time side effect
        import pygame

        assert os.environ.get("SDL_NO_SIGNAL_HANDLERS") == "1", \
            "player no longer sets the hint, and SDL will swallow SIGTERM again"

        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        pygame.mixer.init()
        try:
            assert signal.SIGTERM not in caught_signals(), \
                "SDL installed a SIGTERM handler despite the hint; a `kill` will be discarded again"
        finally:
            pygame.mixer.quit()

    @posix_only
    def test_ctrl_c_still_works(self, restore_signal_handlers):
        # SIGINT is deliberately left to CPython, whose handler raises `KeyboardInterrupt` — which the
        # render loops already catch. This pins that SDL does not displace it, since if it did, the apps
        # would need the same treatment for Ctrl+C and currently do not have it.
        from raven.common.audio import player  # noqa: F401
        import pygame

        pygame.mixer.init()
        try:
            with pytest.raises(KeyboardInterrupt):
                os.kill(os.getpid(), signal.SIGINT)
        finally:
            pygame.mixer.quit()
