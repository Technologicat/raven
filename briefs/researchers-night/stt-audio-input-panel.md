# Brief: the audio input panel — tuning STT autostop in the room

Target file: `briefs/researchers-night/stt-audio-input-panel.md`

Band-2 item 10 of the Researchers' Night sprint. `TODO.md` → *STT / voice*, first item:
configurable silence level, autostop timeout and VU peak hold, **as a GUI rather than as config
knobs**, because the threshold has to be set in the room, on the day.

Speech input is new this year — last year the operator typed the visitors' questions in — so the
noise floor of a full lab during an open-doors evening is a quantity nobody here has ever measured.
That is the whole case for a control: there is no value to ship as a default, only a value to find
on the night.

## 1. What is there now

Read from the source, 2026-08-28.

`raven.common.audio.recorder.Recorder` already implements everything the *engine* side needs:

- `silence_threshold` — dBFS, or `None` to autodetect from the first 0.1 s of each recording, plus
  a 6 dB margin.
- `autostop_timeout` — seconds of continuous silence before the recorder stops itself, or `None`
  to disable.
- `vu_peak_hold` — how long the meter holds a peak.
- `connect_vu_readout(cb)` — `cb(instant, peak)`, both dBFS, called once per audio frame.

`initialize` passes `DEFAULT_SILENCE_THRESHOLD = -40.0`, so **autodetect is off in every Raven app**:
Librarian calls `audio.initialize(recorder={"device_name": ..., "executor": ...})` and takes the
default for the rest.

`raven.common.gui.vumeter.DPGVUMeter` already draws a threshold line. Librarian builds it with
`threshold_value=-40.0` beside a `# TODO: configurable autostop threshold`, and repeats the number
as a literal in the tooltip text.

So the feature is not missing an engine. It is missing three small things and a surface.

## 2. The three things that block a live control

- **The threshold is read once per recording.** In `record_task`, `self.silence_threshold` is copied
  into the local `silence_level_dBFS` on the first frame, and `silence_level_available` is never
  revisited. Changing the attribute mid-recording therefore does nothing. `autostop_timeout` and
  `vu_peak_hold` are re-read every frame and *are* already live.
- **`DPGVUMeter._threshold` is construction-time only** — no setter, and `_update_geometry` is
  private.
- **Nothing shows a level unless a message is being recorded.** There is no way to look at the
  room's floor without capturing a question and sending it to the AI.

## 3. Why the threshold has to clear the whole noise distribution, not its average

Worth stating before the design, because it decides what the calibration button computes and what
the operator should be looking at.

The autostop rule is: any frame whose instant level exceeds the threshold sets
`last_signal_timestamp = time_now`; the recorder stops when `time_now - last_signal_timestamp`
reaches `autostop_timeout`. A **single** frame above threshold restarts the clock. A frame is
`frame_length` samples — 512 by default — so it is a few tens of milliseconds.

Consequently a threshold sitting above the room's *typical* level but below its transients never
autostops at all: one chair scrape or one laugh per timeout window is enough to hold the recorder
open indefinitely. The threshold must clear essentially the whole distribution of room-noise frames,
not its centre.

Two things follow:

- **The calibration measures a maximum over its window**, not a mean or a median — which is what the
  existing autodetect already does (`np.max(np.abs(...))` + 6 dB), only over 0.1 s, which is too
  short to contain a room's transients.
- **Peak hold is the operator's window onto the same statistic.** The held peak line is the maximum
  over the last `vu_peak_hold` seconds, so watching it *is* watching the quantity the threshold has
  to clear. That is what makes peak hold a knob worth exposing rather than an aesthetic setting.

## 4. The panel

A **non-modal window**, titled *Audio input*, opened from a hotkey and from a toolbar button beside
the mic. Non-modal so the operator can watch the meter while a visitor asks a real question — that
is the calibration that matters, and a modal would hide the chat it is happening in.

Contents, top to bottom:

- **A large VU meter** — the existing `DPGVUMeter`, at panel size rather than the toolbar's 8×26 px,
  showing instant level, held peak, and the threshold line.
- **Numeric readouts** under it: `now`, `peak`, and `floor` (the maximum over the last few seconds of
  monitoring). Cheaper and more precise than tick marks on the meter, and the `floor` figure is the
  one the operator is actually trying to clear.
- **Threshold** — a slider over the meter's range, moving the meter's line with it.
- **Autostop** — a checkbox plus a timeout slider. Unchecked means the recording runs until the mic
  button is pressed again, which is the fallback if the room defeats a fixed threshold. `TODO.md`'s
  wake-word item asks for exactly this ("a push-to-talk fallback that can be switched to on the day
  without a restart"), and it costs nothing here beyond the checkbox.
- **Peak hold** — a slider, per §3.
- **Measure the room** — captures a couple of seconds, takes the maximum, adds a margin, writes the
  result into the threshold. The existing autodetect promoted from a per-recording guess to an
  operator-triggered calibration over a window long enough to be meaningful.
- **Reset to configured defaults** — puts back what `client/config.py` says, which is the escape
  hatch from a bad tuning at a moment when nobody wants to reason about precedence.
- **A status line**: whether monitoring is live, and the capture device name.

**Draggable threshold line** on the meter is the obvious refinement and is deliberately not in v1:
click handling on a drawlist is unmeasured here, and the slider does the job.

## 5. Monitoring

Opening the panel starts a capture that feeds the VU readout and nothing else; closing it stops.
This is what makes "tunable in the room" true — the floor is visible with the room noisy and nobody
speaking, at no cost in spurious questions to the AI.

It needs a mode on the recorder, `start(monitor=True)`, with two properties:

- **Autostop suppressed.** Otherwise monitoring stops itself after one timeout of silence, which is
  the state it exists to observe.
- **No accumulation.** The recording path appends every frame to `self.data`; monitoring has no end,
  so it must not append at all.

**Exclusivity.** `pvrecorder` is a single device handle, so monitoring and recording cannot overlap.
Resolution: the mic button and `Ctrl+Shift+Enter` stop monitoring, run the recording normally, and
resume monitoring afterwards if the panel is still open. The operator never sees the handover; what
they see is that the meter keeps working.

**The VU readout needs more than one listener.** `connect_vu_readout` holds a single callback and
Librarian has already used it for the toolbar meter, so the panel would displace it. Make it a
listener list — connect/disconnect rather than a slot. It is a contained change (one caller today),
and `TODO.md`'s wake-word item names three simultaneous consumers of this stream as its central
architectural cost, so the shape is wanted anyway.

## 6. Persistence

Tuned values go in the Librarian app-state file, alongside the existing flags, so a crash and
restart mid-evening does not cost the tuning.

`appstate._DEFAULT_FLAGS` currently holds booleans and is described as the toggles; `save` validates
against `("new_chat_HEAD", "HEAD") + tuple(_DEFAULT_FLAGS.keys())`. Add a sibling mapping for the
numeric settings and let `load`/`save` derive from the union, rather than widening what "flags" means.

**Precedence: a stored value wins over `client/config.py`.** Config supplies the value for a fresh
state file and for the *Reset to configured defaults* button; once the operator has tuned it, the
tuning is what the app starts with. Stated here because the other reading is defensible and the two
are indistinguishable until someone edits config and sees nothing happen.

New config knobs, beside the existing `stt_capture_audio_device`: `stt_silence_threshold`,
`stt_autostop_timeout`, `stt_vu_peak_hold`. Librarian forwards them through
`audio.initialize(recorder={...})`.

## 7. Making the decision testable

`Recorder` has no tests, because recording needs a device. The autostop decision does not: extract it
into a small stateful helper — fed a frame's level and a timestamp, answering whether to stop, and
owning the autodetect state and the live threshold. The record loop then calls it once per frame.

That is what lets the live-threshold behaviour be pinned at all, and it comes with its own negative
control: the same test run against the read-once shape must fail, with the threshold change having
no effect.

Tests to write:

- The gate stops after the timeout of continuous sub-threshold frames, and not before.
- A single above-threshold frame restarts the clock (§3, asserted rather than merely reasoned).
- A threshold changed mid-stream takes effect on the next frame. **Negative control:** the fixture
  must be able to tell that from the old behaviour.
- Autodetect fires once and holds; an explicit threshold set later overrides it.
- `autostop_timeout=None` never stops.
- App state: numbers round-trip, missing keys fill in from config, an old state file without them
  loads.

**What stays untested, and why it is not an oversight.** Monitor mode and the VU listener list are
`Recorder` behaviour, and constructing a `Recorder` opens an audio device — so they cannot run in CI,
which does not install `pvrecorder`, and cannot run unattended on a dev machine either. They are
covered by the step-6 live test. That is also the argument for the extraction above: the part with
the logic worth pinning is the part that needed no device.

## 8. What this is not, and what it leaves room for

The other two STT items are off the Researchers' Night path — the input-language selector was
dropped to [Medium] on 2026-08-25 (the operator asks visitors to speak English, which covers the
mixed audience at no cost), and the wake word was pushed past the deadline. `TODO.md` asks that the
three be designed together because they share a GUI surface. This brief builds that surface and
stops there: the panel has room below the peak-hold slider for a language combobox and for
wake-word controls, and the listener-list change in §5 is the piece the wake word needs.

## 9. Order of work

1. Recorder: extract the silence gate, read the threshold live, add monitor mode, make the VU
   readout multi-listener. Tests for all of it.
2. `DPGVUMeter`: a settable threshold.
3. Config knobs and app-state persistence, with tests.
4. The panel itself, plus the hotkey and the toolbar button.
5. Discoverability: the help card row, the mic tooltip (which currently states `-40` as a literal),
   and Librarian's `README.md`.
6. Live test with a real mic: floor, speech, calibration, autostop firing and not firing.

Step 1 is where the work is; 2–3 are small; 4 is ordinary DPG panel building.

## 10. Risks and levers

- **The room may defeat a fixed threshold.** If transients are dense enough that no threshold both
  autostops and survives a pause, the next lever is a hold-off — require *N* consecutive frames above
  threshold before the clock restarts, so a 30 ms scrape does not count as speech. It is a counter in
  the gate extracted in §7, so the cost is small; it is not in v1 because there is nothing to tune it
  against until the room exists. Named here so it is not re-derived under pressure on the night.
- **The hotkey.** `F1`, `F8` and `F11` are handled before any focus gating, so they fire while typing;
  a bare function key for this panel would behave the same way. `F9` is free. Confirm against the
  live keymap when wiring it, and update the help card in the same commit.
- **Untested against the real device.** Everything above is designed from source. The gate is
  testable without hardware; the panel, the handover between monitoring and recording, and the
  calibration are not, and want the step-6 live test rather than confidence.
