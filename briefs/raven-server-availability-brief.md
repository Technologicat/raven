# Surviving Raven-server: when it goes away, and when it is not there yet

**Three items, from Juha, 2026-09-07** — his numbering, and the third of the two is his joke:

1. The error-catching work wherever Librarian contacts Raven-server.
2. Making Librarian bootable before Raven-server is running.
3. Making the Visualizer recover from Raven-server going down mid-session.

**Item 3 is deliberately not "the same as Librarian"** (Juha's correction, on reading the first draft). The
Visualizer needs no boot-without-server and no status row: the server is optional to it already, and one
part uses it. What it needs is the ability to notice the server leaving and carry on, which is a narrower
job with a different answer.

**What Raven-server is, for the purpose of this brief:** the avatar, speech, subtitles, translation, the
embeddings behind document search, and the AI's internet access. Not the chat — that goes to a separate LLM
backend, which has its own status row and its own reconnect. A reader who does not hold that distinction
will misread every symptom here.

## What already landed (2026-09-07)

Not preamble: these are the parts the three items build on, and two of them change what the items have to do.

- **A status row** in Librarian's utility panel, edge-triggered off a `raven_server_available()` poll that
  runs for the whole session. It names what stops working and what does not. Placed there rather than in
  either panel above because that column sits outside the rect the avatar and the chat graph take turns in
  — so it needs no occupancy test and no reparenting.
- **The panel switch already degrades correctly.** `video_available` answers `False` when the renderer has
  no avatar instance, which is what a lost or never-started stream leaves behind, so the chat graph takes
  the panel on its own. **Item 2 therefore has no GUI work to do about the missing avatar.**
- **`DPGAvatarRenderer.pause` no longer loses the animator's stopped state** when it cannot reach the
  server. That was the first real failure the relay found, and it is the shape to expect more of: a cleanup
  path that notifies the server *before* recording locally what it did, on a path reached precisely because
  the server has gone.
- **`investigations/backend-fault-injection/tcprelay.py`**, which is how any of this is tested. Kill it and
  the server has vanished; run it again and it is back. Do not reach for `faultproxy.py` here — it speaks
  HTTP, and Raven-server puts real data in headers.

## Item 1: error-catching at every call site

**The policy is Juha's and it is uniform: log a warning, and cancel the operation if it cannot be done
without the server.** What is not uniform is what "cancel" means, which is why this is a sweep with
judgement in it rather than a `try` around 37 lines.

**The map, from `grep -rn '\bapi\.[a-z_]*('` over `raven/librarian`, `raven/client` and `raven/common`:**
44 call sites, **37 with no `try` above them**. By subsystem, with what cancelling should mean:

- **`mayberemote.*` — 17 sites, and the one with a real design question.** See below; it is not a matter of
  adding a `try`.
- **`llmtools` — `websearch_search`, `webfetch_fetch`.** These are *tool calls*. The failure should reach
  the model as a tool error, so the reply can say the search could not be run, rather than aborting the
  turn. Check first whether the tool-call layer already converts an exception into a tool result; if it
  does, these two need nothing.
- **The avatar's cosmetics — `avatar_modify_overrides` (4, in `tts.py`), `avatar_start_data_eyes`,
  `avatar_stop_data_eyes`, `avatar_load_animator_settings` (3), `avatar_set_emotion`.** Cancel quietly: the
  user has lost the avatar entirely and does not need a second report per morph. Beware the per-frame ones:
  a warning inside a lipsync loop is a log flood, so these want log-once-per-outage rather than
  log-per-call.
- **`avatar_renderer.imagefx_process_array` and `avatar_result_feed`** — per frame, same caution. The
  result feed already has an error path; it is what surfaced the `pause` bug.
- **`stt_transcribe_array` in `app.py`.** The user pressed record and is waiting for words. This one needs
  a *visible* failure, not a log line.
- **`api.avatar_load` / `avatar_load_emotion_templates` / `avatar_start` in `app.py`'s startup.** These are
  item 2's business, not item 1's.

### The `mayberemote` question, which is a decision rather than a fix

**A `MaybeRemoteService` picks remote or local once, in its constructor**, by asking whether the server
answers. Every later call assumes that answer. So a server that dies mid-run leaves a service in remote
mode with no route back — which is exactly Juha's description of the Visualizer importer: *"if the server
was up when it was started, the importer assumes it stays up."*

**`allow_local` is what settles it, and it settles it differently for the two kinds of caller** — which is
why this reads as one question and is two.

- **`allow_local=True` means the caller has already declared it can work without the server**, and for the
  Visualizer that is not a concession but a *deployment*: local mode exists so that someone who only wants
  the Visualizer never has to set up Raven-server at all (Juha, 2026-09-07). Calling it a fallback
  undersells it — it is one of the two ways the app is meant to run.
  - Which makes the mid-session case easy to reason about. Losing the server does not degrade such a
    Visualizer into something unsupported; it lands it in the configuration a whole class of its users run
    all the time. **The once-only decision is the defect**, not the switch.
  - The cost the constructor's docstring warns about — entering local mode *loads the model locally*, which
    can mean fetching several gigabytes — is unchanged by *when* the decision is taken, and does not apply
    at all to the users local mode was built for, who have those models already. It bites exactly one
    person: a Visualizer user who has a server and has therefore never needed the local models. Worth
    keeping in view, but it is a narrower case than it first looks.
- **`allow_local=False` means the app needs the server for other things anyway**, so there is nothing to
  fall back to and the operation fails. That is Juha's stated policy verbatim, and it is the whole rule for
  these callers.

**So the shape is: re-decide the mode when a remote call fails, and let `allow_local` decide what
re-deciding means.** Whether that is a retry inside the service or a rebuild by the caller is an
implementation choice; the service is the better home, since the caller would otherwise need to know which
exceptions mean "the server left".

Either way `allow_local`'s docstring has to say *when* the decision is made. It currently does not, and that
silence is what let the once-only behaviour go unnoticed.

## Item 2: Librarian boots before Raven-server does

**Today it does not**: `api.test_connection()` at startup exits with 255 if the server does not answer. The
goal is to start anyway, with the status row up, and connect when the server appears.

**The connect path already exists as a startup step**, and the work is largely making it re-runnable:
`_load_initial_animator_settings` (frame callback 2) loads the animator settings, calls `api.avatar_start`,
starts the renderer, loads the backdrop, and warms up the TTS. That has to become something the server
watch can call when the server turns up, rather than a thing that happens once at frame 2.

**The trap Juha flagged, which nothing else would have caught: avatar sessions are persistent on the
server.** So a connect that follows an earlier connect must *unload the old session first*, or every
reconnect leaks one. That makes reconnect-shaped code the natural owner of the unload, rather than
teardown, which by definition does not run on the path that matters.

Open, and worth settling before building:

- **Automatic or asked-for?** The status row already offers "click to retry now". Whether the *connect*
  should also be automatic on first sighting, or offered, is a UX call — an app that suddenly grows an
  avatar ten minutes into a conversation may be a pleasant surprise or an intrusion.
- **What a chat started with no server should do** when the server appears mid-conversation. Nothing about
  the chat itself depends on Raven-server, so the honest answer is probably "nothing, except that the
  features light up".

## Item 3: the Visualizer, the same way

The importer is the only part that uses the server, and it uses it through `mayberemote` with
`allow_local=True` — `Dehyphenator`, `Embedder`, `NLP`. So its exposure is exactly the `mayberemote`
question above, and it inherits whatever is decided there.

**No status row and no boot work here.** The Visualizer is *designed* to run without Raven-server — that is
what local mode is for — so "the server is not there" is not a condition to report, it is a supported way
to run. What is missing is only the transition: the app can start without the server and cannot survive
losing it.

What is different is that **an import is a batch job**: a failure part-way means partial results, and
"cancel the operation" has to say something about what has already been imported. The dehyphenation crash
fixed earlier in this release cycle is the precedent — it could fail a run part-way and lose the whole
thing.

## Later, and deliberately not now

- **HTTPS.** Client↔server traffic is unencrypted, which is fine on `localhost` and nowhere else. Juha
  expects the hard part to be certificates: Raven is open-source software, so buying one is not an option,
  and the goal is not to make users jump through hoops.
- **Load balancers cannot be supported**, because avatar sessions are persistent — a second request routed
  to a different instance finds no session. Worth writing down so nobody proposes it as the scaling answer.
