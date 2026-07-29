# Sketch: the avatar as the interface

**Status: a discussion sketch, not an implementation brief.** Written 2026-07-29 from a design conversation.
Sibling to `corpus-interrogation-sketch.md`, which works out a different angle on the same constellation; that
sketch names this one under "One angle among several" and does not develop it. This does.

Same caution applies: the workflow is clear, the mechanism mostly is not, and writing it as a brief now would
freeze decisions nobody has made.

## What the mode is

Hide the Librarian GUI except the avatar. What remains on screen is a character you talk to — as if you were
talking with a science-fiction AI, because at that point you effectively are. Anything reasonable to do by
voice goes through the avatar; the full GUI stays available on the machine Librarian runs on, for when you walk
over to the console and want to edit a message, inspect provenance, or read a long reply properly.

The talk's long-term vision already names the destination — a personal **co-researcher** with the avatar as an
on-site natural-language interface, literature monitoring with novelty detection, and lab-equipment status by
asking. This is what that looks like as a screen.

**The load-bearing change is who the user is.** Not someone at a keyboard, but someone across the room, hands
busy, glancing over. That single shift is the test to apply to every affordance below: *if it requires reading,
it has failed in this mode.* Speech, an icon, a glance-legible shape — those work. A paragraph does not.

## Most of it is already built, which is the surprising part

Worth establishing before treating any of this as new construction, because the cost intuition is wrong by a
large factor:

- **The avatar itself** — THA3 animation, lipsync driven by Kokoro's phoneme timestamps, emotion and expression
  control from the `classify` module, cel animations, subtitles.
- **Both directions of speech** — Whisper STT on the mic button, TTS with word timings.
- **Tool-calling with validation and dispatch** (`websearch`, `webfetch` are working examples), which is how any
  spoken request becomes an action.
- **A science-fiction visual-effects pipeline, already written and already GPU-resident.** The video
  postprocessor has `translucent_display`, `scanlines`, `banding`, `digital_glitches`, `bloom`,
  `chromatic_aberration`, `vignetting`, `zoom`, plus the whole analog-artifact family. "Pops up with a sci-fi
  effect" is *pointing existing filters at a different texture*, not building a compositor.
- **A GUI animation framework** (`raven.common.gui.animation`) for the timing and easing.

So the work concentrates in three places: a layout mode, a small set of new on-screen objects, and — the real
one — deciding what the avatar is allowed to interrupt you about.

## The affordances, roughly in order of how well they pay

**A quest marker over the avatar's head.** The strongest idea here, and not because of the reference. It solves
*notification without interruption*: an icon floating above the character is ambient, costs no attention until
you look, persists until acknowledged, and is instantly legible to anyone who has played a game in the last
twenty years. Novelty detection is already in the vision — a new paper that fits no existing cluster, hence
plausibly worth a look — and "there is something for you" is exactly the payload a marker carries well.

It generalizes past that one use, which is the sign it is the right primitive: indexing finished, an upload
arrived, a long-running interrogation completed. All the same shape — a state worth knowing, not worth
stopping for. Worth designing as *the* ambient-notification channel rather than as one feature's icon.

**A QR code the avatar offers.** From the upload-page design in the sibling sketch: the desktop shows a QR that
already encodes the upload's destination, so the scoping decision is made where the context is and the phone
only supplies bytes. In this mode the avatar presents it — *"here, scan this to send the file to me"* — as a
pop-up beside the character.

That closes a loop the other sketch left open. The destination has to be chosen somewhere, and if the QR is
summoned *by an utterance*, the utterance carries the scope: "send me a file for this conversation" and "add
something to the hydrogen papers" produce different codes. No picker anywhere, on either device.

**A representation of what arrived.** Once a file lands, render something for it and have it materialize beside
the avatar with the sci-fi treatment — hovering, slowly turning. Mostly this is a thumbnail with an effect
stack over it, which the pieces above already do: a PDF's first page, an image, a document icon. Cheap, and it
answers the question the phone cannot ("did it actually arrive, and is it the right file?") without a word of
text.

**The 3D-model case is the expensive one, and should not be assumed to come along for the ride.** Raven has no
scene renderer — THA3 is a neural renderer for one specific character, not a mesh-and-camera pipeline — so a
spinning model means a loader plus a renderer plus a format zoo, all of it new and none of it shared with
anything else Raven does. The cheap approximations get most of the effect: many formats carry an embedded
preview, and a turntable can be pre-rendered once on arrival and replayed as frames rather than rendered live.
Worth doing the cheap version first and finding out whether the expensive one is still wanted.

## What hiding the GUI costs, and has to be paid back

**Every error has to become speakable.** Today failures surface as modal messageboxes and status indicators —
the LLM backend is down, no model is loaded, a document could not be read, the server went away. In a mode
where the GUI is not on screen, a silent failure is indistinguishable from an assistant that is ignoring you.
So each error path needs a spoken form, and the avatar needs a way to look wrong — the expression machinery is
already there and is the obvious carrier.

This is a real constraint on the mode rather than a polish item: it means the mode cannot ship covering only
the happy path, because the happy path is the half that already works without it.

**Not everything should be voice.** The GUI stays on the console deliberately. Reading a long reply, comparing
two branches, checking where a claim came from, editing a message — these are better with a screen and a
pointer, and the mode is stronger for admitting it than for pretending otherwise. The design question is which
interactions are *reasonable* by voice, and that phrasing is doing real work: the answer is not "as many as
possible".

## Open questions

1. **What summons the mode, and what leaves it?** A toggle, a hotkey, launching differently, or automatically
   when the window loses focus and someone is talking? This decides whether it is a mode or a posture.
2. **Does the avatar speak unprompted?** A quest marker is ambient by design. Speaking is not — it takes the
   room. Probably: marker by default, speech only when addressed or when the user has asked to be told. But
   "your monitoring found something" is exactly the case where unprompted speech might earn its keep, and that
   is worth arguing about rather than assuming.
3. **How much of the reply does the avatar say?** Subtitles exist, and TTS on a thousand-word answer is not the
   same product as TTS on three sentences. Perhaps the spoken form is a summary with the full text waiting on
   the console — which makes the assistant's answer *two* artifacts, and that is a design decision.
4. **What does the phone see?** It has a screen too, and it is the device in your hand. Confirmation only, or
   is there a reason for it to show more?
5. **Where do the on-screen objects live in DPG terms?** Overlaying the avatar panel, a separate always-on-top
   window, or drawn into the avatar's own texture — the last would get the postprocessor for free but couples
   the objects to the render pipeline.

## What this is not

Not a replacement for the GUI, and not a demo. The test in "What the mode is" is the one to keep applying: this
is for the person across the room with their hands busy. If an affordance only makes sense to someone already
sitting at the keyboard, it belongs in the ordinary GUI, where it probably already is.
