# Sketch: what kind of product Raven is

**Status: a discussion sketch, not an implementation brief.** Opened 2026-07-30 (Juha), from the running
discussion of what Raven wants to be. Unlike its three siblings it describes no mechanism at all — it is the
stance the others are designed *in*, written down so that decisions elsewhere have something to be consistent
with.

## The one-line version

**Raven should feel like it fell on our desks from the future** — and specifically, perhaps, aesthetically
from a *1990s* retro-future: the one that era expected and did not get.

But don't bring any actual VHS tapes.

The product is local, it is free, and it exists.

## The tiebreak rule, stated first because it decides the hard cases

**When the (retro-)future and usefulness conflict, usefulness wins.** This is an actual product, used for actual
work, and a tool that is charming to look at and worse to use has failed at the main thing that matters. The
aesthetic is a register the product is *written in*, not a constraint the product serves.

It is first in this document rather than a footnote at the end because that is the order the conflicts arrive
in: the tempting effect always shows up before the readability cost does. So the rule needs to be already
in hand when the question comes up, not found afterwards.

Stated the other way round: **inside what a good tool would do anyway, the register is free.** Nearly all of it
costs nothing — a palette, a font, the shape of an animation, what the avatar's screen-space effects do.
That is where it belongs, and there is plenty of room there.

## What "from the future" means as a test

The framing earns its keep only if it can settle an argument, so here is the form it takes as a test. A
feature fits when both halves hold:

1. It is something you would expect a scifi machine to do.
2. It is not already available *in this form* — local, private, yours, and part of one workspace rather than
   one product per task.

The second half is what keeps this from being nostalgia. The interesting region is the gap between what that
future assumed was ordinary and what is actually on anyone's desk today — a research assistant that is local,
private, and yours; that talks and listens; that sees what you show it; that remembers the work; that is a
collaborator rather than a search box.

**Form, not category.** A category test would be the wrong one and would rule out the flagships: LLM frontends
are abundant, and topic-analysis tools exist. What does not exist is either of them in this form, or the two of
them as views on one corpus. So the gap has three shapes, and only the first is novelty:

- **Not built yet.**
- **Built, but not like this** — our own take, driven by a different vision. A crowded category is not a closed
  one.
- **Built, but never put together** — where the integration is the contribution rather than the parts.

**The avatar is the sharpest instance**, and worth spelling out, because it is where the crowded-category
reading misleads most. Talking-avatar systems exist and are good: Alibaba's
[TaoAvatar](https://arxiv.org/abs/2503.17032) (CVPR 2025) does full-body photoreal avatars for AR via 3D
Gaussian splatting, and Alibaba-Quark's [LiveAvatar](https://arxiv.org/abs/2512.04677) (ECCV 2026) streams
audio-driven avatar video of unbounded length from a 14B diffusion model — with demos wired to an LLM for live
dialogue. "Talking avatar driven by a language model" is not an empty category.

What is absent is the form. Both are avatar *engines* — photoreal humans, generation-first — and they differ
from each other in the way that makes the point:

- **LiveAvatar is heavy**: those real-time figures are on five H800s, where Raven's avatar runs on the desktop
  GPU that is already there. Against this one the contrast is locality and price.
- **TaoAvatar is light** — deliberately so, targeting on-device use on phones and headsets, 90 FPS on an Apple
  Vision Pro. So against this one the compute argument does not apply *at all*, and the distance is entirely
  about form: photoreal rather than anime, an AR avatar rather than an interface, and nothing resembling an
  LLM frontend around it.

Which is the useful half. Where the easy objection — "you only get to do that because you spend less compute"
— is unavailable, the gap is still there, and it is a gap in what anyone is *trying to build*.

**Character.AI is the strongest counter and deserves stating as such**, because for them the character is not a
feature of the product, it *is* the product — at scale, and not as a novelty. They have moved the same
direction: [AvatarFX](https://blog.character.ai/avatar-fx-cutting-edge-video-generation-by-character-ai/)
(April 2025) animates a character from a still image with synchronized speech and expression, in styles
including 2D cartoon, alongside Scenes and Streams. So "treats an animated character as central, and means it"
is not a claim Raven gets to make alone.

What separates it is the rest of the form, and the separation is clean rather than a matter of degree:

- **It is closed and cloud-hosted**, so it fails the form test at the first clause — not local, not private,
  not yours. Nothing to install, nothing to read, nothing to lift, and the conversations are on someone else's
  machine. That also puts it out of channels 1 and 2 entirely: it proves nothing about what you can run
  yourself, and there is nothing to borrow.
- **The use case is companionship and entertainment**, not a working instrument. The character is what you came
  for. In Raven the character is how you reach a corpus, a map, and a set of tools — the interface to work
  being done, not the work itself.

So the claim that survives all three, and the one this document actually makes: nobody else is putting **a
talking anime character into a local, open LLM frontend and treating it as the interface to a research
instrument**. SillyTavern briefly did, through ST-Extras, which hosted Raven-avatar's direct ancestor, and that
path closed when the project moved away from Python and dropped Extras.

Credit where it is due, since the lineage is easy to overstate: ST-Extras was an existing product with a
working if rudimentary THA3 demo — around 10 FPS on a 3070 Ti — and the emotion classifier already in place.
The classifier was lifted wholesale, its license permitting it. What is Juha's is the idle animation, the
lipsync driver — THA3 has always had the speech morphs, but nothing in Extras drove them — the performance
work that made the thing usable, and the postprocessor.

That is all three shapes at once: *not built* (a local anime-character interface for a research tool),
*not like this* (avatar engines exist; this form does not), and *never put together* (the character, lipsync
driven by TTS phoneme timings, classifier-driven expression, and a research frontend became one interface only
once assembled).

Most of Raven's existing direction already passes that test, which is a sign the framing is descriptive rather
than imposed: the avatar as an interface you speak to (`lab-assistant-hci-sketch.md`), screening ten thousand
papers down to the ones worth reading (`corpus-interrogation-sketch.md`), the whole thing running on one or two
local GPUs and sending nothing anywhere.

## The future, building itself

- **The artifact makes the future *possible*.** A local-first research assistant with a face, running on one
  or two consumer GPUs and phoning nowhere, is at present an argument most people would tell you loses — to
  scale, to the cloud, to the platforms. It stops being an argument the moment someone can run it.
  Demonstrated feasibility is a different object from asserted feasibility.
- **Making it *actual* needs uptake.** And in an open-source context, this means user base, or having the
  ideas spread within the community.

### Three channels, and the claim rests on the first two

Ordered by how much each depends on Raven itself succeeding:

1. **Existence proof.** Needs only the artifact to exist. Settles feasibility, permanently.
2. **The ideas spread and get reimplemented elsewhere.** Needs the artifact to be legible and borrowable — but
   *not* for Raven to win. Someone reads it, sees that it is possible, builds their own.
3. **Raven itself is adopted.** Needs users.

The claim is built not to need channel 3.

### What that reclassifies

The useful consequence, and the reason this is worth a document rather than a nice sentence: **if the channels
are the mechanism, then what feeds them is the vision's machinery rather than housekeeping around it.** The
channels want different things, so keep them apart:

- **Channel 2 wants legibility and extractability.** Public briefs and design sketches, notes like `dpg-notes.md`,
  comments that explain *why* — that is what makes an idea liftable by someone who is never going to install the
  product. The project already works this way, so this reframes an existing practice rather than asking for a new
  one. Permissive licensing (2-clause BSD) serves the same channel: a piece can be taken without adopting the whole.
- **Channel 3 wants adoption friction removed.** Install complexity, the hardware floor,
  discoverability, the public name. The deferred items on **easy install with a chosen CUDA version** and on
  **deciding the public name** belong here, and are load-bearing in a way their filing does not currently
  suggest.

So: Raven is part hyperstition, in the modest sense that building the thing makes the future *available*, and
what happens after that is whether anyone picks up either the product or the ideas that went into its creation.
The tilt is small. But it is real, it points the way we want, and it costs nothing beyond building the product
well — which was the plan regardless.

## Aesthetic influences

**Cyberpunk**, and where applicable **early-2000s anime — the fake-HDR / bloom era.**

Both are already operative rather than aspirational, which is the reason to write them down: the code has been
built to them for a while and the intent was never recorded outside conversation.

- The video postprocessor is a science-fiction effects stack that already exists and is already GPU-resident:
  `translucent_display`, `scanlines`, `banding`, `digital_glitches`, `chromatic_aberration`, `vignetting`,
  `monochrome_display`, plus the whole analog-artifact family (VHS head-switching, tracking, hsync rippling and
  runaway, NTSC chroma noise). Several have design briefs of their own — `crt-display.md`,
  `vhs-headswitching.md`, `vhs-ntsc-noise.md`, `atmospheric-dust.md`.
- `Postprocessor.bloom`'s own docstring reads *"Bloom effect (fake HDR) … Popular in early 2000s anime"* — the
  influence is named in the source already.
- `TODO.md` already calls the direction deliberate, in the avatar digital-glitch-on-branch-switch item:
  *"Fits Raven's deliberate cyberpunk aesthetic."*

**The reference is the technique, not a title.** The era is named here by what it did — "HDR" lighting as a
look rather than a pipeline, with bloom applied digitally to any bright part of the frame — and that is
exactly what `Postprocessor.bloom` implements, which makes the filter the most precise statement of the
influence available. Specific titles and studios are not recalled with enough confidence to name, so they are
left out rather than guessed at; anyone filling them in later should be someone who can check.

## Where the aesthetic applies, and where it stops

The tiebreak rule in concrete form, since "usefulness wins" needs a shape to be actionable.

- **The avatar and its screen space** are the natural home, and the influences apply at full strength. It is a
  character on a display; effects on a display are diegetic there, and the user can turn any of them off
  through the postprocessor config.
- **The working surfaces** — Visualizer's map, Librarian's chat log, dialogs, forms — are where people read and
  think, and readability wins outright. The influence shows there as palette, typography, iconography, and the
  character of a transition.
- **Motion is a special case worth naming**. Animated user interfaces make pair work feasible: when two people
  share one screen, a transition is what lets the one who did not press the key see *where the view went*, and
  a jump cut leaves them to re-find their place. The same argument extends to the human-and-agent pair, and
  gets stronger there — once tool calls can change the GUI, those transitions have to be animated, because the
  party that has to follow the change is a human who did not initiate it. However, an animation the user
  routinely outruns is a cost. A successful product strikes a balance.

## The product does not explain itself

No "retro-future theme" label in the settings, no in-app note about influences, no README section pointing at
its own references. It simply looks the way it looks.

This is the same principle as the naming rule in `CLAUDE.md` — a name may carry layered references, and the
surface reading has to work on its own regardless. A reader who recognizes where the look comes from gets that
for free; one who does not sees a well-made tool, which is what they came for.

## Where this bites next

From the aesthetic half:

- **Avatar: digital glitch on branch switch** (`TODO.md`, Avatar/Librarian-side). The first concrete claim this
  document makes, and already scoped there as a scripting task over existing filters.
- **Smooth scrolling across the constellation** (`TODO_DEFERRED.md`). The same principle in its least
  glamorous register: things that move should move alike.

From the channels, where the effect is a reclassification rather than a new task — each of these is already on
a list, filed as maintenance:

- **Extract `raven.common` into an upstream library** (`TODO_DEFERRED.md`, "corvid"). Channel 2.
- **Easy install with a chosen CUDA version** (`TODO_DEFERRED.md`), and the hardware floor generally. Channel 3.
- **The public-name decision** (`TODO_DEFERRED.md`, *"Decide the public name"*). Both channels — a name is how
  an idea is referred to as much as how a product is found — so decide it with this document in view rather
  than on availability alone.

## Open questions

- **How far into the working surfaces does the register reach?** Palette and typography are agreed; the
  boundary past that is undrawn.
- **Is the 1990s specifically right** as the aesthetic anchor, or is it the broader "future that was expected
  and did not arrive"? The decade may be a useful anchor or a false precision. Either way this is a question
  about the *look*: the product-scope question is settled above, and retro-futurism does not bound it.
- **Where does any of this get said out loud** — README, talk, or nowhere, and only enacted?
