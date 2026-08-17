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

1. It is something you would expect the *lab computer* to do — the machine a researcher talks to while doing
   their own work.
2. It is not already available *in this form* — local, private, yours, and part of one workspace rather than
   one product per task.

Both halves need unpacking, and they are different jobs. The two sections below take them in turn: what bounds
the machine in half 1, and why half 2 is about form rather than category.

## Which scifi machine, though

"A scifi machine" alone is too loose: this one does not control 3D printers, and the genre's AI systems
notoriously include several nobody should be building. "A system of this kind" would be circular. Three
properties bound it without circularity, and none of them is about the technology: two describe the
*relationship*, and the third says who decides what it can reach.

- **It assists someone doing their own work**, rather than doing the work instead of them. The researcher stays
  the researcher, and this is an instrument they use. That is also the line against a coding agent or a general
  agent harness, where handing the task over is precisely the point.
- **It reports rather than acting on its own initiative.** Note that this is *not* a bound on what it may
  reach: the lab-assistant direction takes it well outside your own files — asking instruments for their
  status, and an experiment controller for the results of a run, over MCP
  (`lab-assistant-hci-sketch.md`). The bound is that its job is to tell you things so that *you* can decide,
  not to pursue goals in the world by itself.
- **What it may reach is yours to grant.** Today this holds by inventory: the tool surface is `websearch`,
  `webfetch`, `search_documents`, `fetch_document` and `list_consulted_documents`, none of which write
  anywhere. But an inventory is not a guarantee, and the MCP client (brief 04) makes the surface whatever the
  user connects. So it has to be carried by policy instead, and the pattern already exists for `webfetch`,
  split across two layers: network-level refusals server-side (private-network addresses, non-HTTP(S)
  schemes), and a client-side allowlist constraining which public sites the model may visit on its own
  initiative.

None of this bounds how many steps it may take unattended, so the scriptable agent layer on the list
(`TODO_DEFERRED.md`, headless scaffold mode for `ai_turn`) sits inside it rather than against it.

## Form, not category

The second half of the test is what keeps this from being nostalgia. The interesting region is the gap between
what that future assumed was ordinary and what is actually on anyone's desk today — a research assistant that
is local, private, and yours; that talks and listens; that sees what you show it; that remembers the work; that
is a collaborator rather than a search box.

Read as a *category* test it would be the wrong one, and would rule out the flagships: LLM frontends are
abundant, and topic-analysis tools exist. What does not exist is either of them in this form, or the two of
them as views on one corpus. So the gap has three shapes, and only the first is novelty:

- **Not built yet.**
- **Built, but not like this** — our own take, driven by a different vision. A crowded category is not a closed
  one.
- **Built, but never put together** — where the integration is the contribution rather than the parts.

**The avatar is the sharpest instance**, since it is where the crowded-category reading misleads most. Three
systems sit near it, and each removes one objection the claim would otherwise have to survive:

- **[LiveAvatar](https://arxiv.org/abs/2512.04677)** (Alibaba-Quark, ECCV 2026) — streaming audio-driven avatar
  video of unbounded length from a 14B diffusion model, demos wired to an LLM for live dialogue. So "talking
  avatar driven by a language model" is not an empty category. But its real-time figures are on five H800s,
  against the desktop GPU that is already there.
- **[TaoAvatar](https://arxiv.org/abs/2503.17032)** (Alibaba, CVPR 2025) — full-body photoreal avatars for AR
  via 3D Gaussian splatting, and deliberately light: on-device on phones and headsets, 90 FPS on an Apple
  Vision Pro. So the compute objection does not apply here at all, and the whole distance is form — photoreal
  rather than anime, an AR avatar rather than an interface, nothing resembling an LLM frontend around it.
- **[Character.AI](https://blog.character.ai/avatar-fx-cutting-edge-video-generation-by-character-ai/)** — the
  strongest counter, because there the character *is* the product, at scale and in earnest; AvatarFX (April
  2025) animates it from a still image, in styles including 2D cartoon. What it is not is local, private, or
  yours: closed and cloud-hosted, nothing to install, read, or lift. And the character is what you came for,
  rather than how you reach a corpus, a map, and a set of tools.

What survives all three: nobody else is putting **a talking anime character into a local, open LLM frontend
and treating it as the interface to a research instrument**. SillyTavern briefly did, through ST-Extras, which
hosted Raven-avatar's direct ancestor; that path closed when the project moved away from Python and dropped
Extras. Credit where due, since the lineage is easy to overstate — Extras was an existing product with a
working if rudimentary THA3 demo (~10 FPS on a 3070 Ti) and the emotion classifier already in place. Our
improvements to the system - which then moved to Raven - are the idle animation, the lipsync driver
(THA3 has always had the speech morphs; nothing drove them), the performance work that made it usable,
and the postprocessor.

That is all three shapes at once: *not built* (a local anime-character interface for a research tool),
*not like this* (avatar engines exist; this form does not), and *never put together* (the character, lipsync
driven by TTS phoneme timings, classifier-driven expression, and a research frontend became one interface only
once assembled).

Most of Raven's existing direction already passes that test, which is a sign the framing is descriptive rather
than imposed: the avatar as an interface you speak to (`lab-assistant-hci-sketch.md`), screening ten thousand
papers down to the ones worth reading (`corpus-interrogation-sketch.md`), the whole thing running on one or two
local GPUs and sending nothing anywhere.

**Why the current shape is literature-heavy, and why that is not the definition.** The first-half-of-the-2020s
jump in language technology lands most naturally on text, and a corpus is made of text — so literature work is
where the capability arrives first. The need is real rather than assumed: researchers read a great deal, and
the volume of published work has outrun the reading (`corpus-interrogation-sketch.md` carries that argument in
full). But arriving first is not the same as being the boundary. Taken as the identity, "literature tool" would
make the lab-assistant direction look like drift, when it is the same product reaching the next thing the same
capability is good for.

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

**The register is a blend, and the two dates are not in tension.** Cyberpunk is a 1980s–90s genre, which is
precisely why it reads as retrofuturism today: the future it imagined has aged into one. The early-2000s
reference is not a second genre but a rendering technique — the digital bloom the postprocessor implements. So
the dates belong to different things: one to what is depicted, the other to how it is lit.

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
  - **Continuous motion on live state is the second kind, and it is admitted on the same grounds.** A
    transition animates a *change*; a pulsating selection animates a *condition* — this is the thing you have
    picked, this is where the cursor rests, this is the set a search matched. It aids the same reader for the
    same reason, and the Visualizer has done it since long before this document existed: `PlotterPulsatingGlow`
    breathes the search results and the selection at opposite phases, so the two stay distinguishable where they
    overlap. Librarian pulsates its recording, indexing and backend-caution indicators through the shared
    `PulsatingColor` in `raven.common.gui.animation`.
  - **It also happens to be the register's clearest tell, which is why it is worth spreading rather than
    merely keeping.** An immediate-mode toolkit redraws every frame whether or not anything moved, so a
    breathing highlight costs nothing beyond the theme edit — while a conventional desktop GUI, redrawing on
    invalidation, would have to be *made* to do it and therefore does not. A programmer who notices a selection
    that breathes has learned something true about what this is built on, without being told. The naming rule
    applies unchanged: it has to earn its place as a readability cue first, and it does.

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
- **Pulsating cursor and selection in `FileDialog`.** The keyboard work of 2026-08-17 gave the listing a cursor
  and a selection on separate visual axes (text colour and fill), both currently static. Making them breathe
  brings the dialog into line with the Visualizer's plotter, and `PulsatingColor` already does the work — the
  open part is which axis pulsates, and whether the two want opposite phases the way the plotter's do.
  - **The dialog is the sharpest case because it is already halfway there**, which is what makes the missing
    half legible as an omission rather than as a road not taken. Its thumbnail view resizes through
    `raven.common.image.lanczos` on the GPU and stands undecoded tiles in with VHS noise from
    `postprocessor.vhs_noise_pool` — the register is in the pixels already. What it has is the still frame of
    it; a file browser that grades its thumbnails through a video pipeline and then holds the selection
    perfectly still is showing its hand in one place and hiding it in the other.

From the channels, where the effect is a reclassification rather than a new task — each of these is already on
a list, filed as maintenance:

- **Extract `raven.common` into an upstream library** (`TODO_DEFERRED.md`, "corvid"). Channel 2.
- **Easy install with a chosen CUDA version** (`TODO_DEFERRED.md`), and the hardware floor generally. Channel 3.
- **The public-name decision** (`TODO_DEFERRED.md`, *"Decide the public name"*). Both channels — a name is how
  an idea is referred to as much as how a product is found — so decide it with this document in view rather
  than on availability alone.

## Open questions

- **How far into the working surfaces does the register reach?** Palette, typography and live-state motion are
  agreed; the boundary past that is undrawn. What the motion answer suggests is a shape for the rest of it —
  an effect gets in when it is doing a reader's job, and the aesthetic reading is a bonus a genre-savvy user
  collects on their own.
- **Where does any of this get said out loud** — README, talk, or nowhere, and only enacted?
- **What is the permission model once the tool surface is user-extensible?** Brief 04 covers the adapter,
  transports, namespacing and lifecycle, but has no gating section. "It reports rather than acting on its own
  initiative" stops being self-enforcing the moment a connected MCP server can offer a tool that acts — and
  the lab direction points at exactly the systems where that matters.
