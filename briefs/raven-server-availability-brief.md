# Surviving Raven-server: when it goes away, and when it is not there yet

**Three items, from Juha, 2026-09-07** — his numbering, and the third of the two is his joke:

1. The error-catching work wherever Librarian contacts Raven-server.
2. Making Librarian bootable before Raven-server is running.
3. Making the Visualizer recover from Raven-server going down mid-session.

**Item 3 is deliberately not "the same as Librarian"** (Juha's correction, on reading the first draft). The
Visualizer needs no boot-without-server: it is *designed* to run without Raven-server, that being what local
mode is for. What it needs is to survive losing the server mid-session, which is a narrower job.

It does still need a status indication, and an earlier draft of this brief said otherwise — reasoning from
"the server is optional to the Visualizer" to "so it never has to report on the server". True of a
Visualizer that *started* without one, false of a Visualizer that started with one: that app is in remote
mode, and when the server goes its importer is unusable until the server returns or the user asks for local
mode. Being unusable with no explanation is the thing a status row exists to prevent.

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

**Settled 2026-09-07, Juha's call, and it is neither of the two answers this section first proposed.** The
once-only decision **stays**: instantiation time is when the mode is declared, and nothing switches by
itself afterwards.

- **What is missing is not the switch, it is the declaration being invisible.** So: say which mode is in
  force, loudly in the log, and — for the Visualizer — in the GUI at startup. Today nothing tells a user
  which of the two ways their app is running.
- **If the server then goes down in remote mode, that is the user's to resolve**: bring the server back, or
  choose local mode. What must not happen is Raven choosing for them.
- **Choosing local mode should be a click, not a restart.** That is the part that needs building; a
  declaration you can only revise by relaunching is not much of a choice.

**Why not switch automatically**, since an earlier draft of this brief argued for it and the argument was
not silly: `allow_local=True` really does mean the caller can work without the server, and for the
Visualizer local mode is a *deployment* rather than a concession — it exists so that someone who only wants
the Visualizer never has to set up Raven-server at all. The trouble is that the switch is not free and not
invisible. It can mean fetching several gigabytes of models, and it changes where the work runs and how
long it takes. A user who has a server has never needed those models, and is exactly the person a silent
switch would surprise. Juha's summary: *"it's nontrivial which is better"* — which is the reason to put it
in front of the user rather than to pick harder.

**`allow_local=False` is unaffected**: the app needs the server for other things anyway, so there is
nothing to offer and the operation simply fails, with the warning. That is the stated policy verbatim, and
it is the whole rule for those callers.

Either way `allow_local`'s docstring has to say *when* the decision is made. It currently does not, and that
silence is what let the once-only behaviour read as an oversight rather than as the design.

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

**Better still, if it turns out to be reachable: re-adopt the old session rather than replace it** (Juha,
2026-09-11). Persistence is the property that makes the leak possible, and it is also the property that
would make this the *good* outcome — the session the client lost is, from the server's side, still sitting
there loaded and posed. A reconnect that resumes it costs no model load and no visible restart, where
unload-then-create pays for both and blinks the avatar.

Three things to establish first, and all three are about the server rather than about Librarian: whether a
session's id survives on the client across the outage, whether the server can be asked what it still holds,
and whether a session whose stream was dropped is still in a state worth resuming or has been torn down by
something else.

**"No" is a design input rather than an answer, because the server is ours** (Juha, 2026-09-11). If
resumability is not there, we are free to build it — and probably should, on the condition that it comes at
reasonable effort and does not drag in much complexity. So the investigation is not "is this reachable"
but "does it already work, and if not, what would making it work cost". Worth having in view while reading
the answers, since the first two questions have cheap fixes available if they come out wrong: an id the
client forgets can be persisted, and a server that cannot be asked what it holds can be given an endpoint
that says.

The unload path is needed either way, as the fallback for the disconnections that resumption cannot cover
and as the thing that stops the leak meanwhile, so it is not wasted whichever way this goes.

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

**No boot work here** — the Visualizer already starts happily with no server. What it needs is three things,
in what looks like increasing order of effort:

1. **Say which mode it is in**, loudly in the log and in the GUI at startup. Cheapest, and useful on its own:
   nothing currently tells a user whether their embeddings are being computed on a server or in-process.
2. **Say when remote mode has broken**, i.e. a status row like Librarian's, for the app that booted with a
   server and lost it. Its importer is unusable until the server is back, and unusable-with-no-explanation
   is what the row prevents.
3. **Offer the switch to local mode from the GUI.** The declaration is made at instantiation and stays made
   — the point is not to revise it automatically but to let the user revise it without relaunching. This
   is the piece that needs real design: the services are built during an import, so "switch now" has to
   mean something definite about a run in progress.
   - **Relaunching is the expensive alternative, which is what makes this worth building.** A restart loses
     the user's selection undo/redo history, selections not being saveable yet. So "just restart it" is not
     the cheap escape it sounds like, and a user whose server dies mid-session currently has no other move.

### Both actions live in the pill, and deliberately not as a matching pair

Settled 2026-09-07. The row carries two actions — *retry the connection now*, and *switch to local mode* —
and they are **asymmetric on purpose**: the body of the pill retries when clicked, as Librarian's already
does, and the switch is a button of its own.

**The asymmetry is the safety argument, not a layout accident.** The two actions differ sharply in
consequence: a retry is a cheap probe that changes nothing, while switching modes moves the work off the
server and can pull down several gigabytes for exactly the user who has a server and has therefore never
needed the local models. So the question is which mis-aimed click one would rather people make. With a big
forgiving target for the retry and a small deliberate button for the switch, a missed click costs a probe.
With two identical buttons, a missed click can start a download.

Symmetry is normally the least-surprise argument, which is why it was the tempting one; here it would put a
consequential action and a trivial one on equal footing, which is the thing least-surprise arguments usually
exist to prevent.

The cost of the asymmetry is discoverability — "click here to retry" is invisible without the tooltip. That
is already how Librarian's row works, so the tooltip carries it, and the same tooltip is where the switch
should say what it may cost, so that a user meets the download *before* clicking rather than after.

**No confirmation on the switch.** Raven does have a pattern for this — two clicks with a warning flash, as
in the file dialog's overwrite and Librarian's delete-subtree — and it is the house way to confirm
something. Judged overengineering here (Juha): the action is deliberate and tooltip-warned, which for a
button nobody reaches by accident is enough. Recorded because the pattern is the right one to reach for if
practice proves otherwise, not because the question is still open.

**What is *not* part of that argument: "and you can always restart".** Restarting the Visualizer loses the
user's selection undo/redo history, selections not being saveable yet — so a restart is a real cost rather
than the escape hatch it looks like, and an earlier draft of this brief leaned on it. That cuts the other
way too, and is the strongest argument for building the switch at all: without it, a user whose server dies
mid-session has *only* the expensive remedy.

**An import is a batch job**, which is what makes item 3 harder than its Librarian counterpart: a failure
part-way means partial results, and both "cancel" and "switch to local" have to say what happens to what has
already been imported. The dehyphenation crash fixed earlier in this release cycle is the precedent — it
could fail a run part-way and lose the whole thing.

## Later, and deliberately not now

- **HTTPS.** Client↔server traffic is unencrypted, which is fine on `localhost` and nowhere else. Juha
  expects the hard part to be certificates: Raven is open-source software, so buying one is not an option,
  and the goal is not to make users jump through hoops.
- **Load balancers cannot be supported**, because avatar sessions are persistent — a second request routed
  to a different instance finds no session. Worth writing down so nobody proposes it as the scaling answer.
