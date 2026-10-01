# Sketch: installing and starting Raven without a terminal ritual

**Status: a discussion sketch, not an implementation brief.** Written 2026-10-01 from the maintainer's
notes. Decided: the direction — a user installs once and then starts Raven from a launcher. Open: most of
the mechanism, and three things to probe before anything is designed on top of them.

**What forces it is the pilot users** (maintainer, 2026-10-01): today each one who wants to pilot Raven
costs an on-site support visit, to install it and again to upgrade it. The lab installation benefits too,
but it is not the driver.

## What it takes today

Installing:

1. On Windows, ask an admin to install Anaconda.
2. `git pull`, `pdm python install --min`, `pdm install` (and possibly a `pdm use` in between).
3. Install LM Studio.
4. Open its Settings window and enable the Developer tab.
5. In the Developer tab, switch the server on.

Then **every time**, in three terminals:

- `cd` into the checkout, `$(pdm venv activate)`, `raven-server`;
- start LM Studio, open the Developer tab, load the model;
- `cd` into the checkout, `$(pdm venv activate)`, then the app, e.g. `raven-librarian`.

That is too propellerhead for most potential users (maintainer, 2026-10-01).

## What it should take

Install LM Studio and a model once. Install Raven with one command. Then start an app from the desktop's
menu, and have Raven-server and the model come up by themselves.

## The pieces

Most of them exist as separate items already; this sketch is what they add up to.

- **Raven-server starts itself when an app needs it** — `briefs/server-autostart-brief.md`, policies
  decided, gated on availability item 2 (`briefs/raven-server-availability-brief.md`).
- **Launchers.** A desktop entry per app on Linux, Start-menu shortcuts on Windows, the macOS equivalent.
  They need an installed command to point at, which a checkout plus a venv is not.
  - **And one for Raven-server alone, opening it in a terminal window** (maintainer, 2026-10-01): for
    processing imports in the background when no app is needed at that moment
    (`briefs/design/offline-and-remote-processing-sketch.md`). A server started that way counts as started
    by hand, so under the autostart brief's policies it never exits on idle — which is what a background
    job needs.
- **The LLM backend started and loaded the same way.** Nothing covers this yet. LM Studio ships a CLI,
  `lms`; its own error messages suggest `lms load`, so Raven could start the backend and load the
  configured model as it will start its own server.
  - **The model to load is already configurable**: `raven.librarian.config.llm_model`, whose comment says
    that on LM Studio with just-in-time loading, naming the model in a request makes the server load it.
    If that holds, Raven only has to *start* the server, and the first request does the rest.
  - **Only when the backend is local and declared to be LM Studio** (maintainer, 2026-10-01): `localhost` in
    `llm_backend_url`, and `llm_backend_flavor = "lmstudio"` set explicitly. Autodetection cannot answer
    here, `llmclient.detect_backend_flavor` working by asking the running server — which is the thing
    that is missing.
- **Installing as a tool, from PyPI.** One command, the console scripts on `PATH`, no `cd` and no
  activation. Waits on the `raven-lab` rename and the wheel audit (`TODO_DEFERRED.md`, both 0.2.11), and
  above all on the GPU install item below.
- **The GPU install** — `TODO_DEFERRED.md`, "GPU-accelerated install on any OS and GPU, without editing
  `pyproject.toml`". A package on PyPI cannot say which index its torch should come from, and PyPI's own
  torch wheels carry one CUDA version per release, so this is the hard part of installing from it.
- **A Python to install with.** Anaconda is there only to provide a Python for PDM to run in. `uv` (Astral)
  installs per user as a single binary and fetches its own Python, so for a user it could replace both
  Miniconda and PDM; development would stay on PDM.

## Upgrading

Half of the support cost above, so it needs as much design as installing. Two parts are already in place:
a user's own settings live outside the installed tree, in `~/.config/raven/overrides.json`, so an upgrade
does not overwrite them; and Librarian's chat datastore migrates itself on load. What is missing is the
upgrade itself — one command, once Raven installs as a tool rather than from a checkout.

**One thing does not survive an upgrade safely: the Visualizer's datasets.** They are pickles with no
migrator, not portable across Python or app versions, so an upgrade can leave a pilot user unable to open
their own data. The format is due a redesign anyway (`TODO.md`, "Data file format"), and brief 13's
unified DB may be where that happens; until then, an upgrade path for pilots should at least say so.

## To probe first

None of these has been checked; each decides how much of the above is cheap.

1. **Whether `uv` can choose the torch index from the machine's GPU.** Recalled, not verified: a recent `uv`
   gained an option that picks the PyTorch backend from the installed driver. If it holds for `uv tool
   install`, it answers the GPU install item for Linux and Windows.
2. **What `lms` can do headless**: start the server, and load a model by name.
3. **Whether LM Studio can start its server at login.** That would make the backend half nearly free. Its
   just-in-time loading is documented in our own config's `llm_model` comment; confirm it is on by default.
