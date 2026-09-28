# Raven-server starts itself when an app needs it

**Status: direction and both policies decided (Juha, 2026-09-28); the mechanism below is a proposal where
marked [P].** Not before Yrityspäivä (8 October 2026) — the demo runs a hand-managed server either way.

## What the user gets

A desktop user who runs everything on one machine starts an app and nothing else. If no Raven-server is
running, the app starts one; when the last app that needs it has gone, it stops.

**Manual management stays fully supported.** An app never stops a server it did not start. Keeping one server
alive across app restarts is how development works, and may be the right shape for some deployments too — so
a server started by hand is left alone by every policy here.

## The two policies

**At most one instance: the port is the lock.** An app asks whether a server answers at its configured URL,
and starts one only if nothing does. Two apps starting one at the same moment is harmless: the second server
fails to bind the port and exits, and both apps find the first.

**Last one turns off the lights: leases, not a count.** A server started by an app is launched with a flag —
`--exit-when-idle`, say — and exits once no client has been in touch for some interval. A server started by
hand has no flag and never exits on its own, which is what makes the manual mode safe without any
coordination. A crashed app simply stops being in touch, so there is no reference count that can be left
stuck.

- [P] **The lease may already exist.** Librarian polls `/health` for the whole session, for its server status
  row, so "no request of any kind for N seconds" might be the whole lease with nothing new on the client
  side. The Visualizer does not poll yet; its status indication in the availability brief (below) would add
  one. Check what else would have to count as activity — a long-running request, say, during which no poll
  arrives from the process making it.

## What it depends on

**Librarian must boot before the server is up**, and currently does not re-check: a server that appears
after Librarian has started is not picked up. That is *Item 2: Librarian boots before Raven-server does* in
`briefs/raven-server-availability-brief.md`, and it comes first. Autostart makes the case routine rather than
rare, since the server's model loading takes far longer than an app's startup.

## Settled details

- **Which server.** The default config module, with the user's overrides applied — which
  `raven.configoverrides` already does for any config module, so the spawned server needs no special
  handling to get them.
- **Logs.** An ordinary user cannot write under `/var/log`, so a spawned server logs under
  `~/.config/raven/server/`. Apps take `--server-log` and `--server-log-level`, used only when that app is
  the one starting the server; the same spelling as the apps' own `--log` and `--log-level`.
- **Environment.** A Raven install is its own venv — a plain pip install into a shared environment is not a
  target, since Raven's ML stack would break the user's other apps. Activating the venv *should* be enough to
  run `raven-server` in its default config, but that has not been exercised since the `~/.bashrc` wrappers
  started sourcing `env.sh` for everything, so it is the first thing to check: a spawned server inherits the
  app's environment, and the CUDA library paths are the part most likely to be missing from it.

## Open

- **Windows.** A child that outlives its parent is spelled differently there (a detached process rather than
  a new session). Linux and macOS share the POSIX spelling.
- **The idle interval**, which has to outlast the longest gap between polls with some margin, and an app
  restart if keeping the server across one is wanted in the autostarted case too.
- **What the user sees while the spawned server loads.** The status rows already say what is unavailable;
  whether "starting" deserves its own state there is a GUI call.
- **The unified DB.** If the database moves behind the server (brief 13, *Where the database lives*), the
  server stops being optional for any app that touches the DB, and autostart becomes the thing that keeps
  the one-machine case painless. The two designs should be read together.
