"""An opt-in in-process REPL, so a running app can be asked questions its log cannot answer.

**DANGER — read this before wiring an app up, and before running one with `--repl` anywhere but your own
machine.** A REPL server is *remote code execution as a feature*. Everything it does is what it is for,
which is exactly why it deserves a warning rather than a caveat:

  - **No authentication of any kind.** Whoever connects gets a prompt. There is no password, no token, and
    no check on who is at the other end.
  - **No encryption.** The session is plaintext on the wire, so anything typed or printed — file contents,
    API keys held in the app's memory, the user's chat history — is readable by anything on the path.
  - **Full access to the process.** A session can call any function, read and rewrite any object, open
    files and sockets as the user running the app, and `os.system` whatever it likes. It is not a
    read-only inspection window and cannot be made into one.
  - **Localhost only, and that is load-bearing.** `maybe_start` binds `127.0.0.1` and offers no way to ask
    for anything else. Do not add one; forward a port over SSH if a remote session is genuinely needed, so
    that authentication and encryption are somebody's job rather than nobody's.
  - **For debugging, and off by default.** No app starts one unless `--repl` is passed on its command
    line. It has no place in a demo, an exhibit machine, or anything left running unattended.

Common Lisp's Swank, roughly: connect to a live process and inspect it while it keeps running. The
situation it answers is the one where nothing else helps — an instance that came up *wrong* and is still
running, where the next launch will be fine and killing this one destroys the evidence. Intermittent
font-atlas damage in the Markdown renderer is the standing example; a wedged unattended batch is the other.

Two calls wire an app up: `add_argument` where it builds its parser, and `maybe_start` once the app is
built. Both are deliberately separate — argparse happens before anything exists, and what a session should
be able to reach does not exist until the app is up.

**What a session can reach is whatever it is handed**, plus everything reachable from there — which is the
part to get right, because an empty room looks exactly like a working REPL until you ask it something.
Raven's apps come in both shapes:

  - **Built at module scope** (`librarian`, `visualizer`, both avatar editors): pass `globals()`. The
    session then shares the app's namespace *live*, so a name rebound later is visible.
  - **Built inside `main()`** (`xdot_viewer`, `cherrypick`, `conference_timer`): the interesting state is
    that function's locals, and `globals()` alone would reach almost none of it. Pass
    `{**globals(), **locals()}` from inside `main()`. That is a snapshot rather than a live view — good
    enough, since the objects in it are the live ones and only the *bindings* are frozen. Anything whose
    rebinding a session needs to see should be in an `unpythonic.box` regardless, and a box comes through
    a snapshot intact.

`python -m unpythonic.net.client localhost` is the other end — note the client takes its ports as separate
arguments (`localhost PORT CONTROL`), not as `host:port`, and defaults to the same pair this module does.
"""

__all__ = ["DEFAULT_PORT", "add_argument", "maybe_start"]

import argparse
import logging
from typing import Optional

logger = logging.getLogger(__name__)

#: Where a REPL lands when `--repl` is given with no port. `unpythonic.net`'s own default, and its client's
#: — so a bare `--repl` is reachable by a bare `python -m unpythonic.net.client localhost`, with nothing to
#: remember at either end.
DEFAULT_PORT = 1337

#: The control channel that goes with `DEFAULT_PORT`. Also `unpythonic.net`'s own, and *not* `DEFAULT_PORT
#: + 1`: both numbers are chosen rather than derived (1337 for the obvious reason, 8128 being the fourth
#: perfect number), and the client defaults to this one. Deriving it would silently move the control
#: channel out from under `client localhost`, which is the spelling everybody will actually type.
DEFAULT_CONTROL_PORT = 8128


def add_argument(parser: argparse.ArgumentParser) -> None:
    """Declare the `--repl` option on `parser`. Call this where the app builds its command line.

    `parser`: The app's `argparse.ArgumentParser`.

    The option's value is what `maybe_start` wants for `maybe_port`: `None` when the flag was not given,
    the port when it was.

    There is deliberately no per-app default port. The default's whole value is that both ends already
    know it, so an app choosing its own would be giving up the only thing it buys — and would then need a
    matching control port, which is a second chosen number and not a derivable one.
    """
    parser.add_argument("--repl", nargs="?", metavar="PORT", const=DEFAULT_PORT, type=int, default=None,
                        help=f"DEBUGGING AID, DANGEROUS: open an in-process REPL. A bare --repl uses "
                             f"{DEFAULT_PORT} with its control channel on {DEFAULT_CONTROL_PORT}, and is "
                             f"reached by `python -m unpythonic.net.client localhost`; --repl PORT uses "
                             f"PORT and PORT+1, reached by `python -m unpythonic.net.client localhost PORT "
                             f"PORT+1`. Anyone who can reach either port gets unauthenticated, "
                             f"unencrypted, arbitrary code execution inside this app, as you. Bound to "
                             f"localhost; use SSH port forwarding rather than exposing it. For debugging "
                             f"a running instance, where its log does not say enough.")


def maybe_start(maybe_port: Optional[int],
                session_locals: dict,
                app_name: str) -> bool:
    """Start the REPL server, if asked for. Returns whether it started.

    `maybe_port`: The port, or `None` to do nothing — i.e. the `--repl` option's value straight from
                  argparse, so that a caller needs no `if` of its own.
    `session_locals`: What a REPL session sees as its namespace. An app whose entry point runs at module
                      scope should pass `globals()`.
    `app_name`: Named in the banner, so a session opened onto the wrong window says so.

    The protocol needs two ports, a REPL channel and a control channel, and both are bound to localhost.
    `DEFAULT_PORT` keeps its own partner, `DEFAULT_CONTROL_PORT`; any other port takes the one above it,
    there being nothing else to derive it from. `unpythonic.net.server` registers its own `stop` at exit,
    so there is nothing to tear down here.

    Read this module's docstring before calling this. It opens an unauthenticated, unencrypted,
    arbitrary-code-execution port into the calling process.
    """
    if maybe_port is None:
        return False

    # Imported here rather than at module scope: an app that never asks for a REPL should not pay for the
    # import, and this module is imported by every app's startup path.
    from unpythonic.net import server as repl_server

    control_port = DEFAULT_CONTROL_PORT if maybe_port == DEFAULT_PORT else maybe_port + 1
    try:
        repl_server.start(locals=session_locals,
                          bind="127.0.0.1",  # an arbitrary-code-execution port by design; do not offer it to the network
                          repl_port=maybe_port,
                          control_port=control_port,
                          banner=f"{app_name}. The app's namespace is in scope.\n")
    except (OSError, RuntimeError) as exc:
        # A debugging aid must not be the thing that takes the app down. Two dull causes, and neither is
        # worth a traceback: another process already holds the port (`OSError`, the likely one — a second
        # app started with a bare `--repl`, so both asked for the default), or this process already has a
        # server, which `unpythonic.net.server` allows only one of (`RuntimeError`).
        logger.error(f"replserver.maybe_start: could not open a REPL on 127.0.0.1:{maybe_port}: "
                     f"{type(exc)}: {exc}. Continuing without one; pass a different port to `--repl` if "
                     f"something else has this one.")
        return False

    connect = ("python -m unpythonic.net.client localhost" if maybe_port == DEFAULT_PORT
               else f"python -m unpythonic.net.client localhost {maybe_port} {control_port}")
    logger.warning(f"replserver.maybe_start: REPL listening on 127.0.0.1:{maybe_port} (control channel "
                   f"127.0.0.1:{control_port}) for {app_name}. ANYONE who can reach those ports can run "
                   f"arbitrary code in this process, as you, unauthenticated and unencrypted. Debugging "
                   f"aid; do not leave it running. Connect with: {connect}")
    return True
