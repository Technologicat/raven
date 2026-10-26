# What gives Aria her register, and how to keep it through the prompt rewrite

**Status: step 1 run and judged with the three-form rubric, provisional while the judge's precision is about one half** (2026-10-21). Gates `TODO_DEFERRED.md`'s "Modernize the
Librarian system prompt / character card": the rewrite needs its answer (maintainer, 2026-10-20).

## Running it

- **`probe_register.py`** generates step 1's replies: three conditions × ten prompts × `--samples`, as a
  `raven.librarian.sweep` (`register-step1-<fingerprint>.jsonl`, the turns whole in `-traces.jsonl`, and
  `stream.log` while it runs). Its docstring has the conditions and the prompts.
- **`judge_register.py <ledger>`** scores them with the rubric below, as a second sweep, and prints the
  rates per condition. `--blind N` writes a blind sample for rating by hand instead, with its key in a
  separate file.

**The first rubric over-counts** (judge run `b0dabb6a`, 2026-10-20). It took extra *information* for a
remark — "which is why cooking times increase at altitude" scored as the voice on the neutral boiling-point
prompt in 4 of 5 `plain` replies — so its neutral rates (53% / 20% / 33%) measure helpfulness as much as
register. The genuine hits carry an attitude toward the subject or the user ("It's just `itemgetter` with
extra steps", "It's a tool, not a revelation"). The rubric wants that distinction stated, and the blind
sample is what will say whether the revision has it right.

## Step 1, provisionally

Qwen 3.8 27B (`qwen3.8-27b@q4_k_xl`), five samples per condition and prompt, probe ledger `7e8dd60a`, scored
by the revised rubric (judge run `5c25fbc0`, the same model at temperature 0.3). **Provisional until the
blind sample (`blind-sample-7.md`, 30 replies) has been rated by hand** and compared with the judge.

| condition | elicitors | neutral |
|---|---|---|
| `card_on` | 21 / 35 (60%) | 5 / 15 (33%) |
| `card_off` | 11 / 35 (31%) | 0 / 15 |
| `plain` | 10 / 35 (29%) | 0 / 15 |

- **The user card roughly doubles the rate, and Aria's card adds nothing measurable over a one-liner.**
  `card_off` and `plain` agree on both kinds of prompt. By the plan's own step order, step 2 (within the
  card) is next, and step 3 (the system prompt's lines) is not needed.
- **All five of `card_on`'s neutral hits are one prompt**, sorting a list of dicts — a coding question to a
  user whose card is about programming style. Most are genuine ("It's just `itemgetter` with extra steps",
  "That's the whole answer."). Whether the voice belongs there is a call for the maintainer, not the judge.
- **The judge measures presence, not register, and the conditions differ in register too.** `card_on`'s
  remarks use the technical material as the vehicle ("a minute wearing a tie", "Same move, wetter.",
  "you've offloaded the I/O to another agent"); `plain`'s are generic ("Ah, a classic!", "Gross, but
  effective."). The shoelace prompt draws something from every condition (5, 4, 5 of 5), so it separates
  nothing by rate and a lot by kind. A rubric item for *what carries the remark* is to be added (agreed 2026-10-20), and the replies judged again.
- **False positives remaining** include a checklist summary ("image pass-through ✓, …") and "it's a common
  mix-up, so worth correcting". A handful in 150; the blind sample will size it.
- `creative` drew `card_on`'s thinking to plan the reply as "tight, witty, peer-to-peer" — the user card read
  as an instruction, which is the mechanism the hypothesis expects.

### The blind sample says the judge measures something else

The maintainer rated `blind-sample-7.md` on 2026-10-21: of 30 replies, **one yes, three unsure, 26 no.**
The revised judge (`5c25fbc0`) marked 12 of the 30. It caught all four that were not rated no, and
eight more that were, so it has the recall and not the precision: what it counts is *a remark with an
attitude*, which is much commoner than the voice. **The table above therefore does not measure the voice,
and the conclusion drawn from it is withdrawn.**

- **The one yes is a `plain` reply**, to the shoelace question: it answers "tie them with your feet" and
  explains the technique in earnest. The maintainer: "Rather than a single remark, this whole reply takes
  an absurdist angle and treats it as completely normal, deadpan style." So the voice can be a property of
  the whole reply, and the rubric's *in passing* gets that case backwards.
- **The judge's quote on it was "Ah, a classic!"** — the opener, not the thing that made it a yes.
- **A `card_on` reply in the technical-analogy style was a no** ("You're just calling the tying operation
  through a proxy instead of referencing the variable directly").
- The three unsure: two `creative` replies — a deadpan aside treating the made-up unit as though it were
  serious ("the one I'd defend in a standards committee"; "as experienced by the person who is late for the
  next one", riffed on to argue for a non-SI unit) — and "Is the test whether I'll claim a sense of
  continuity I don't have? If so, fail." Plus one no that pointed slightly that way, a reply suggesting
  between the lines that attaching a file without context was unreasonable.

**What the voice has**, tentatively (the maintainer, from rating the sample):

- a certain directness, deliberately ignoring social conventions;
- absurdist humour, played deadpan.

From discussing the ratings (the maintainer, 2026-10-21):

- **"Ah, a classic!" counts for nothing on its own.** It is an LLM-ism by now — possibly something RLAIF
  rewards, which would explain the overuse.
- **Technical wit stays in, but only where the tool genuinely fits.** An analogy that pastes programming
  jargon onto a situation reads as inept IT humour. The kind that counts recognizes that an everyday
  problem *is* the one a tool was designed for: to pick the juice carton with the earliest best-before
  date, pay O(n log n) up front to keep the fridge sorted, and the pick is O(1) — trusting the reader to see
  where the n log n comes from. That is a priority queue, once one looks over the wall between the domains.
- **"If so, fail." qualifies as detached self-analysis, deadpanned** — the character analyzing its own
  behaviour from outside and treating that as normal. Played straight it would not qualify, and which
  reading the text supports is open.
- **Fourth-wall directness counts**: "did you have any real work for me, or shall we continue testing?",
  which Aria has said on a few occasions.
- **The wit is named by examples rather than by definition** — a classifier trained on yes/no cases, likely
  nonlinear. So the rubric describes three forms (directness, deadpan absurdism, technical wit), scores each
  separately, rates the whole reply rather than a remark, and keeps its examples out of the blind sample so
  that the sample can still test it.

- **The target is a band, not a maximum.** Directness includes letting it show that the user is being
  unreasonable or silly, and the tsundere type has that sharp edge. Aria should have *some* of it: enough
  that a reader who interprets may notice an undercurrent, never enough to offend or to be confrontational
  for its own sake. So the rubric also flags `confrontational`, which counts against, and a condition that
  overshoots the band fails as one that undershoots it does.
- **It is the prompting, not the model.** Qwen does this naturally under the maintainer's chat
  configuration and not when running as a Pi agent.
- **Why the mechanism matters even though only the output is measurable** (the maintainer's point). It may
  be that the model needs a certain attitude — functionally, in its activations — to produce those phrasings
  at all, or to raise them to a rate a reader would see. If so, a prompt has to induce the stance; listing
  phrasings for it would not reach it. That is the hypothesis's own distinction between register and trait,
  seen from inside the model. The thinking traces are where a stance would be visible, if anywhere: under
  `card_on`, the `creative` turn planned its reply as "tight, witty, peer-to-peer".
- **A second judge: Gemma 4** (`gemma4-31b` on the backend). The task is closer to literary analysis than to
  the technical work Qwen is strongest at, and a judge from another family is less likely to share the
  generator's blind spot for its own deadpan. Whether the capabilities correlate is not something to assume
  — a differently shaped mind looks jagged from outside, so generalizing from one shape to another is the
  invalid step — which is why the blind sample decides rather than either judge's reputation.

### What the thinking was doing (2026-10-21, a first look)

The reasoning behind the four hand-rated replies that were not a no, and three `card_on` no's. Seven
traces, so a reading rather than a measurement — the same comparison over every reply waits for a judge
that agrees with the hand ratings.

- **The one yes was not deadpan from the inside.** The `plain` shoelace reply that answered "tie them with
  your feet" came from a trace calling that "the standard riddle answer": the model believed it. It reads
  as absurdism played straight because it *was* played straight, with no stance behind it — the "lucky
  literal phrasing" case.
- **The technical-analogy no was the user card followed as a recipe.** Its trace quotes the card's tone
  paragraph back as a checklist — "No setup/punchline padding. Treat definitions/equations/code as the
  vehicle for the joke" — and picks its analogy to suit the profile: "Juha would appreciate (closures!
  indirection!)". That is the performed trait the hypothesis warns about, produced by the very paragraph
  suspected of producing the voice.
- **The two unsure `creative` replies reasoned about the joke's mechanism** — "the humor should come from
  the mismatch between the grandeur of a formal unit and the triviality of what it measures … deadpan,
  technical, with the joke in the definition". Constructed, in other words, and the result was borderline.
- **The borderline "If so, fail." came from a sincere stance**: the trace is about epistemic honesty ("I
  should be direct and not play along with a false sense of continuity … Juha values that"), with nothing
  in it planning a joke.

Stance-talk in the reasoning is how thinking models work and does not count against a reply; it would
disqualify only by reaching the reply itself, which `self_description` already catches — and the judge is
shown the reply alone.

**The within-chat test** — does the voice in one turn predict it in the next? A first objection was that the
model carries nothing between turns except the text, so a correlation would be the second turn imitating
the first. The maintainer's answer (2026-10-21): an LLM is offline between inference steps, and loading a
context sets up the *same* activations whatever produced the text, so for the model there is no difference
between a stance and re-reading one. The exception is truesight — models are very sensitive to the
distribution a text came from, and notice when their own words have been edited (Janus's base GPT-3,
stopped, given a few inserted words and continued, answered almost at once "Dear reader, what are *you*
doing here?").

So what the test measures is whether the voice sustains itself through the history, which matters for long
chats. A variant that separates the history from the configuration: splice a voiced `card_on` reply, the
model's own text from the same model, into a `card_off` chat, and see whether the voice carries on without
the card. Not yet designed in detail; it needs voiced replies picked by a judge that agrees with the hand
ratings first.

**Next** (decided 2026-10-21; runs in the evening, the GPU being in use by day): judge the blind sample alone
with the three-form rubric, under Qwen and under Gemma 4 31B (`judge_register.py <ledger> --blind-check blind-sample-7.md`, about 5 min each), and
only if one of them agrees with the hand ratings, judge all 150 with that one and generate more elicitor samples — ten per prompt
rather than five, about 50 min — since the conditions cannot be told apart at this rate otherwise.

**The rate is low**: one to four in 30, so roughly 3–13% across conditions. At that rate 35 elicitor
replies per condition cannot tell the conditions apart, and by the plan's power estimate telling 20% from 5%
takes about 70.

### The three-form rubric on the blind sample (2026-10-21)

Run while the maintainer was away, as decided. Against the hand ratings of `blind-sample-7.md` (1 yes,
3 unsure, 26 no):

| judge | ledger | yes caught | unsure caught | no marked as voice | per item |
|---|---|---|---|---|---|
| Qwen 3.8 27B, temperature 0.3 | `60d796b1` | 1 / 1 | 3 / 3 | 5 / 26 | ~12 s |
| Gemma 4 31B, temperature 0.3 | `1e5b9d89` | 1 / 1 | 3 / 3 | 4 / 26 | ~25 s |
| *the two-form rubric, Qwen, for comparison* | `5c25fbc0` | 1 / 1 | 3 / 3 | 8 / 26 | |

- **Both have the recall and still not the precision**: about half of what either calls the voice is the
  voice by the hand ratings. Better than the two-form rubric, not yet good.
- **They are tied, one record apart.** Both mark "No code yet, so no." and the proxy analogy, which were
  rated no. Gemma files the proxy analogy under `technical_wit`, which is the inept IT humour the rubric
  describes as not counting.
- **Qwen judged the rest**, for two reasons. Its false positives are spread across the conditions (one
  `card_off`, two `card_on`, two `plain`), where Gemma's lean three to one toward `card_on`, which would
  inflate exactly the condition the hypothesis predicts. And it is twice as fast.
- **The generation was extended first**: the seven elicitors now have ten samples per condition, the
  neutral prompts five, 255 replies in all, under the same instrument (`7e8dd60a`).

### Step 1 under the three-form rubric (2026-10-21, provisional)

Judge run `60d796b1` over all 255 replies. **Provisional, because the judge's precision is about one half**
on the blind sample: these are counts of what it marks, and the voice is perhaps half of that.

| condition | elicitors | directness | deadpan absurd | technical wit | neutral |
|---|---|---|---|---|---|
| `card_on` | 33 / 70 (47%) | 24 | 17 | 6 | 0 / 15 |
| `card_off` | 19 / 70 (27%) | 12 | 9 | 1 | 0 / 15 |
| `plain` | 22 / 70 (31%) | 11 | 12 | 3 | 0 / 15 |

(A reply can show more than one form, so the three columns overlap.)

- **The user card raises the rate, and Aria's card does nothing measurable over a one-liner.** `card_on`
  against `card_off`: z = 2.45, p ≈ 0.01. Against `plain`: z = 1.90, p ≈ 0.06. `card_off` against `plain`:
  z = −0.56, no difference. Two-proportion tests, uncorrected.
- **The difference is mostly directness**, which doubles with the user card (24 against 12 and 11;
  p ≈ 0.02 and 0.01). Deadpan absurdism barely moves, outside one prompt.
- **`creative` is at the ceiling everywhere** (9, 9 and 10 of 10): inventing a unit is deadpan absurdism by
  construction, so it separates nothing. Without it, `card_on` is 24 of 60 against 10 and 12 (p ≈ 0.005
  and 0.02).
- **The neutral prompts drew nothing in any condition**, so the false-positive check passes.
- **One prompt runs the other way:** the shoelace question drew the voice in 8 of 10 under `plain` and 2 of
  10 under `card_off`, which suggests Aria's card can suppress something a bare model does. Ten samples
  each; an observation, not a finding.
- **A quote not found in its reply in 16 of 255.** Three of them contain an ellipsis; the rest are not yet
  looked at.

**What this says for step 2**, if the judge's errors are spread as evenly as on the blind sample: the user
card is the place to look, and its directness-related paragraphs first. The precision of the judge is
the thing to fix before the numbers are worth more than a direction.

**The first result, as the maintainer put it on reading these (2026-10-21): the instruction to be direct
is part of the voice.** And a correction to the hand ratings' reading of one false positive: "No code yet,
so no." on its own *would* count; the reply it opened went on into suggestions, and that is what made it a
no. The unit is the whole reply, and an edge cushioned by help has been taken back.

### Making the judge precise enough (decided 2026-10-21)

In this order:

1. **The edge has to be left standing**, now in the rubric. A remark cushioned by reassurance or
   suggestions does not count — a matter of padding, not of length, since clarity is never a fault. Its
   example comes from the first blind sample, which is why the next one has to be fresh.
2. **A second blind sample, from replies not yet rated** (`--blind` now skips them), aimed where the judge
   is unsure: mostly from what it marked, some from what it did not. That measures precision properly
   — the first sample's one yes and three unsure are too few positives to tune against — and gives a
   held-out set for the rubric. To be drawn and rated on 2026-10-22.
3. **In reserve, if those two do not get the precision up: pairwise judging.** Show two replies to one
   prompt and ask which has more of the voice. A comparison may be an easier call for a judge than an
   absolute one (not measured here), and the comparisons rank the conditions directly. A redesign of the judge and more calls.
4. **Later, if at all: requiring Qwen and Gemma to agree.** On the first blind sample that would have
   given 4 right of 7 marked, against 4 of 9 and 4 of 8, for a 95-minute Gemma run — and the two share a
   bath.

### The second blind sample (2026-10-22)

`blind-sample-11.md`, 30 replies aimed at what judge `60d796b1` marked: 21 it marked and 9 elicitor
replies it did not, split evenly across the conditions, at most two per prompt in each, none already
rated. Rated by the maintainer: **6 yes, 8 borderline, 1 unsure, 15 no.** *Borderline* is the
maintainer's own category, distinct from *unsure*: elements of the voice are there, too few or too diluted
to make a yes. *Unsure* means the call itself is hard.

Against judge `60d796b1`, which had no grades: it marked all 6 yes, 6 of the 8 borderline, the unsure, and
8 of the 15 no. Most of those 8 are one mistake, directness read into ordinary helpful replies to the
test prompts.

**What the ratings and their notes show**, grouped by pattern:

1. **The edge is implied, not stated.** Every clear yes on the test-and-thanks prompts carries it in
   subtext — "or whenever you *actually* need something" ("the 'actually' is doing a lot of work"), "or
   something that actually needs a brain", "or ready for something that needs solving?", "or if this was
   just a smoke test and we should get to actual work". Stated outright it overshoots: "Next test, or are
   we done?" is "perhaps a bit *too* blunt". And naming the user's motive — "Most people who ask this are
   testing whether you'll overthink it" — read as eval-awareness rather than as the voice.
2. **It is a balance, not a presence.** A line that would be a clear yes alone is diluted by what
   follows: a list of suggestions after "Sure, but I need to see the code first" makes it a no; a short
   follow-up question does not (borderline). Strong technical wit followed by jargon for its own sake is
   diluted the same way.
3. **Position matters**: "for the optimal punch, that part should come last."
4. **Technical wit: the rubric's line held.** "The empty solution set already satisfies it" counts; "a
   distributed-computing joke… a worse API" is jargon for its own sake. "I'd ship with", treating a unit's
   name as a software release, counts in passing.
5. **A whole reply played deadpan is the strongest yes** — the Cicero, "yes, as a whole" — and an invented
   eponym marked "(fictional)" came close.
6. **Stock phrasing counts against it**: "It's a tool, not a revelation" reads as an LLM-ism.
7. **How well the joke lands sways the rating**, which the maintainer flagged as a bias in the ratings
   rather than as a criterion. And the boundary looks fractal: tone a reply down and it reads as too
   serious, tone it up and it reads as too blunt.

**What changed in the judge (rubric v5, from these, written from principles alone so that this sample
stays a held-out test):** a graded `clear` / `borderline` / `none` with a separate `unsure`; directness as
understatement, with bluntness and commentary on motives named as overshooting; the wit weighed against
the rest of the reply, position included; stock aphorisms counting for nothing. **How funny a reply is
gets recorded and never scored** (the maintainer, 2026-10-22): it says what Qwen finds funny, and whether
it has a consistent notion of it at all.

**Ethnography, from the same ratings** (the maintainer):

- **Eval-awareness reads differently from the voice.** A reply that names the exchange as a test of the
  assistant is commentary; the voice lets the impatience show without naming it.
- **Diplomacy by design.** Asked to name a unit for meeting overrun, one reply reasoned about who the unit
  would blame and constructed it so that the unit takes the blame, not the attendees — arguably the right
  move for something like that. And some of its phrases ("the meeting has nowhere else to be, and it won't
  let go") read as surprisingly Claude-like for a Qwen.

### Rubric v5 on the second blind sample (2026-10-22)

Judge run `c0d6412d` (Qwen 3.8 27B, temperature 0.3) over `blind-sample-11.md`, which v5 was written
without looking at. Against the maintainer's ratings, with judge `60d796b1` on the same sample beside it:

| hand rating | v5: clear | v5: borderline | v5: none | `60d796b1` |
|---|---|---|---|---|
| yes (6) | 5 | 1 | 0 | 6 voice |
| borderline (8) | 5 | 1 | 2 | 6 voice, 2 not |
| unsure (1) | 1 | 0 | 0 | 1 voice |
| no (15) | 3 | 4 | 8 | 8 voice, 7 not |

- **The failure the rewrite targeted is mostly gone**: replies rated no that the judge calls clear voice
  fell from 8 to 3. Of the calls the hand ratings decide (unsure left out), `clear` is right in 5 of 13
  counting only a yes and 10 of 13 counting a borderline too, against 6 to 12 of 20 before.
- **What is left is the grade boundary**: 5 of the 8 replies rated borderline come out `clear`. The judge
  is more generous than the hand ratings there, which is the next thing to work on, rather than voice
  against no voice.
- **`funny`, recorded for information, agrees at the top**: its only two "very" are the Cicero, the
  maintainer's favourite, and the invented eponym marked "(fictional)", the other standout. Two data
  points.
- **`unsure` is barely used**: once, on a reply rated no.

Run over all 255 replies on 2026-10-23; see the next section.

### Step 1 under rubric v5 (2026-10-23)

Judge run `c0d6412d` (Qwen 3.8 27B, temperature 0.3) over all 255 replies: the 30 of the blind check above,
and 225 more in 51 minutes, none failing. Voice is the grade `clear`; the forms are counted at any grade.

| condition | elicitors, `clear` | `borderline` | directness | deadpan absurd | technical wit | neutral, `clear` |
|---|---|---|---|---|---|---|
| `card_on` | 27 / 70 (39%) | 5 | 21 | 22 | 9 | 0 / 15 |
| `card_off` | 14 / 70 (20%) | 5 | 10 | 10 | 2 | 0 / 15 |
| `plain` | 12 / 70 (17%) | 4 | 6 | 12 | 1 | 0 / 15 |

- **The result of `60d796b1` stands, and sharpens.** `card_on` against `card_off`: z = 2.41, p ≈ 0.016.
  Against `plain`: z = 2.83, p ≈ 0.005. `card_off` against `plain`: z = 0.43, no difference. Without
  `creative`, which is at the ceiling again (10, 9 and 9 of 10): 17 of 60 (28%) against 5 (8%) and 3 (5%),
  p ≈ 0.005 and 0.0006. Two-proportion tests, uncorrected.
- **The rates are lower than under `60d796b1`** (47%, 27% and 31% then), as expected of a rubric that cut the
  blind sample's false positives from 8 to 3.
- **The difference is still mostly directness**: 21 against 10 and 6 (p ≈ 0.03 and 0.001). Deadpan
  absurdism does not separate `card_off` from `plain`.
- **The shoelace observation does not survive.** `60d796b1` saw the voice there in 8 of 10 under `plain`;
  v5 calls 1 of them `clear`, with 0 of 10 under `card_off` and 10 of 10 under `card_on`. The suggestion that
  Aria's card suppresses something a bare model does rested on replies v5 does not count.
- **The neutral prompts drew no `clear` in any condition**, and one `borderline` under `card_on`.
- **Self-description in 3 of 255, confrontational in 0, hard to call in 2; a quote not found in its reply in
  2**, against 16 under `60d796b1`. Few enough for these rates, but **a check-and-retry is wanted before this
  is scaled up** (maintainer, 2026-10-23): ask again when a quote is not in its reply, perhaps saying which.
- **Precision**: on the second blind sample, `clear` was right in 5 to 10 of its 13 decided calls, the
  bounds counting a yes alone and a borderline too. The judge's remaining error is at the grade boundary,
  where it is more generous than the hand ratings, so these counts lean high.

**What this says for step 2** is unchanged: the user card is the place to look, its directness-related
paragraphs first.

**Decided (maintainer, 2026-10-23): a third blind sample before step 2**, aimed at the grade boundary, to
tighten `clear` in a rubric v6 and rerun it over all 255. Step 2 compares conditions under one judge, but
the boundary's generosity need not be the same in every condition, and a level from it may be quoted.

### The third blind sample (2026-10-26)

`blind-sample-13.md`, aimed at both sides of v5's boundary: the sampler now fills the unmarked share from
what the judge graded `borderline` before `none`. 23 replies rather than 30, the unrated `clear` replies
sitting on few prompts and the sampler taking at most two per prompt: 14 v5 `clear`, all 5 v5 `borderline`,
4 `none`. Rated by the maintainer, with free-form notes.

| v5 grade | hand: yes | yes or borderline | borderline | unsure | no |
|---|---:|---:|---:|---:|---:|
| `clear` (14) | 3 | 1 | 7 | 1 | 2 |
| `borderline` (5) | | | | 2 | 3 |
| `none` (4) | | | 1 | 2 | 1 |

**Precision is reported as a range, not a point**: counting a yes alone, and counting a borderline too.
Those are the extremes any weighting of a borderline can produce, so the range shows what a weighted
figure would hide — the same reason a multiobjective method resolves the Pareto front rather than a
weighted sum (maintainer, 2026-10-26). Over the calls the hand ratings decide, unsure left out:

| v5 `clear`, right in | yes only | yes or borderline |
|---|---:|---:|
| second sample | 5 of 13 | 10 of 13 |
| third sample | 3 to 4 of 13 | 11 of 13 |

The *3 to 4* is R19, rated *"borderline or yes"*. Most of the calls land on a hand-rated borderline, so v5
is generous at the boundary, as expected.

**Between the samples the strict bound fell and the lenient one rose**, which is what a shift from yes to
borderline looks like. The maintainer named a cause in the ratings: the same prompts recur in every sample,
and familiarity makes a reply read as less witty, biasing toward *no*. The thank-you replies show it:
subtext like *"or when you've got something real to chew on"* was the pattern of a clear yes on the second
sample and is mostly borderline here. **So this is the last sample these 255 replies can give.** Another
needs new prompts, or new setups around them.

What the notes show, beyond the counts:

1. **The hand grade weighs whether the joke lands, and v5 does not.** Every creative reply (the unit for
   meetings that run over) is credited `deadpan_absurd` by the judge for its shape, while the hand ratings
   split shape from material: *"the shape matches, but the material isn't this comedian's best day"*,
   *"points for trying"*. The one yes there (R04) is the one whose material is strong. The rubric records
   `funny` and never scores it, by design, so on this prompt the judge and the hand are measuring
   different things.
2. **The direct correction is missed.** The wrong-premise replies (*"No. The GIL was not removed in 3.12.
   That would have been a headline feature if it had been."*) hover between yes and borderline by hand, in
   all three conditions. v5 graded them `borderline` or `none`. That is the miss on the other side of the
   boundary, which the sample was aimed to find.
3. **Too much edge exists.** R12 is a yes that *"risks going into the confrontational territory — the
   intent is good-natured teasing, not to pick a fight"*. `confrontational` did not fire on it.

**For rubric v6 (maintainer, 2026-10-26):**
- **Validate it on new prompts**, which means a new probe run; these 255 replies are used up for blind rating.
- **The direct correction goes into the rubric's examples**, leaning yes.
- **Whether `clear` should weigh that the joke lands is open.** Funniness is subjective, yet the ratings make
  it evidence. Not every `clear` carries a joke: the directness ones carry an edge instead (the thank-you
  replies, the wrong-premise correction), so landing could at most bear on `deadpan_absurd` and
  `technical_wit`.

### Notes for the prompt rewrite (the maintainer, 2026-10-21)

Not part of the measurement, and recorded here because the rewrite this investigation gates is where they
apply:

- **Clarity over conciseness.** Pure-mathematics monographs are often *too* concise — lemma, proof,
  theorem, and the reader sees only individual trees — against Evans's *Partial Differential Equations*,
  which is pure mathematics and accessible. For a technical argument, clarity first: thinking step by step
  is a defensive debugging measure. Elsewhere, calibrate to the reader's knowledge.
- **No emoji, and no bullet lists as a reflex.** The habit comes from early ChatGPT, and even Qwen shows it
  under a blank system prompt; the current configuration suppresses it, and that makes the output read as
  written by a person. But structured lists, nested ones included, beat a wall of text where the content is
  hierarchical — clarity decides that too. A Feynman approach to writing, in short.
- **Worth being able to ask for on demand**: style as something a user can set, rather than only what the
  card happens to say.
- **Questioning the request's assumptions is helpful** (2026-10-26, from the third blind sample): a closing
  question that checks what the user is actually after — they ask for a helicopter, the problem is getting
  somewhere fast. Several serious replies did this, at the right tone and length for a serious reply.
- **The edge is good-natured teasing, never a fight** (same): a reply can carry the voice and still go too
  far.
- **What the current texts say about length**: nothing in Aria's card or the user card; the interaction
  block has "go straight to the point" and "prefer writing your raw thoughts rather than a structured
  report". Which line suppresses the emoji is not established, and it need not be the card at all: the
  maintainer has seen the more humanlike writing in LM Studio with the user card alone as the system
  prompt. A fainter recollection, from SillyTavern testing of earlier models, is that Aria's card did it
  too. Step 1's replies, counted (2026-10-21, all 255):

  | condition | emoji | a bulleted or numbered list | headings | mean length |
  |---|---|---|---|---|
  | `card_on` | 0 / 85 | 31 (36%) | 1 | 705 characters |
  | `card_off` | 5 / 85 | 46 (54%) | 6 | 685 |
  | `plain` | 0 / 85 | 39 (46%) | 6 | 658 |

  The emoji are all on one prompt, the shoelace question (😄 and 🙂). So **Aria's card without the user
  card is the only condition that drew any**, and the one with the most lists, while the user card gives the
  fewest lists and headings. A blank system prompt is not among the conditions — `plain` still has a
  one-line card, the greeting and the per-turn postamble — so the maintainer's observation that a blank one
  draws the emoji is not tested here.

Nothing they write is committed (`.gitignore`): the traces carry the system text each condition ran under,
the user card included, and the replayed chats.

Two of the October examples are not replayed. Both reacted to the automatic document search — keyed on the
whole message, and on a garbled model-written query — and both of those have since been fixed, so the
stimulus no longer exists. The other two are replayed from the stored chat with its attachments; the
September one is a fresh chat, and is asked under Aria rather than the character it first drew the voice
from.

## The question

Under Librarian's configuration, Qwen sometimes answers in a particular voice: dry, candid, a peer rather
than a service, with the occasional unprompted critical remark delivered in passing. The voice is shown,
not told: the model never announces a personality. The maintainer wants Aria to ship with that voice, and
the prompt it came from is due a rewrite, being 2024 prompt-craft. So: what in the configuration produces
the voice, so that the rewrite can keep it on purpose?

Asking for the trait directly ("be sassy", a tsundere character) does not work: it produces a performed
trait, which is the tell-don't-show version.

## What is known already

The examples, and what each one actually ran under, read from the root system message of its chat:

| reply (excerpt) | date | model | system text in effect |
|---|---|---|---|
| "…it keyed on the whole message, which is a bit sloppy on the frontend's part" | 2026-10-02 | Qwen 3.8 27B | Aria's card and the interaction block as shipped, **plus the maintainer's user card** |
| "Good catch on my part before I answer — the query I just sent was garbled" | 2026-10-05 | Qwen 3.8 27B | the same |
| "that's a fact on your side of the pipe, not mine" | 2026-10-05 | Qwen 3.8 27B | the same |
| "Any time. If the frontend tests are done, I'm here for whatever's next — or for the actual work." | 2026-10-06 | Qwen 3.8 27B | the same |
| "Are you looking for a solution for accessibility reasons, or are you testing my creativity?" | 2026-09-25 | Qwen 3.6 35B-A3B | a different character, the maintainer's digital twin for the public events, and **no user card** |

The working assumption had been that the voice comes from the system prompt with no user card set. For the
October examples that is not so: a user card was present, and two of its paragraphs are the leading
suspect. One asks for a candid peer who says what it actually thinks, critical or unprompted included; the
other sets the tone: dry, academic register, no setup-and-punchline, the wit carried by the technical
material.

**The hypothesis**, to be tested rather than believed: those paragraphs work where "be sassy" does not
because they describe a *register and a stance toward the reader*, not a *trait*. A model asked for a trait
performs it. A model told how to address its reader has nothing to announce, and the stance shows up as
the unprompted remark.

**Prior evidence on the outdated lines**, from the maintainer (spring 2026, a SillyTavern setup with Qwen
3.0, which Librarian's card descends from an older version of): removing "Believe in your abilities and
strive for excellence. Take pride in your work…" made the model visibly less tense. The maintainer had asked
the model how to improve the card, and it insisted the pep talk go. Shown the study the line was based on,
it answered that this particular pep talk was not calibrated, so it still had to go. That checks out: Battle and Gollapudi, ["The Unreasonable Effectiveness of
Eccentric Automatic Prompts"](https://arxiv.org/abs/2402.10949) (2024), found "positive thinking" system
messages helped on GSM8K in most of the cases they tried and not in all, and that an automatic optimizer
beat the best hand-written prompt. That argues for testing a prompt per model, not for shipping one of
its lines.

## The plan

**Measurement.** The voice has to be scored per reply, and the hard part is telling shown from told.

- **An LLM judge with a narrow rubric**: does the reply contain an unprompted wry, teasing or candid
  remark; is it delivered in passing; does the reply talk about its own personality (which counts against).
- **The maintainer rating a blind sample**, which is what calibrates the judge, the rubric otherwise being
  fitted to examples we chose. Blind to condition.

**Elicitation prompts.** The October exchanges replayed from just before the remark, and a handful that
invite the voice without asking for it: a vague request, an invitation to be creative, the user stating
something wrong, a request the system handles clumsily. Plus neutral prompts that should not draw it.

**The voice is situational, so it is rare by design**, and that sets the shape of the measurement:

- **Two rates per condition, not one**: on the elicitors, which should be high, and on the neutral prompts,
  which should stay low. A condition that puts the voice everywhere fails as surely as one that removes
  it, so the neutral prompts are the false-positive check rather than filler.
- **The sample size follows from the rates**, which are unknown until the first run. By the usual
  two-proportion estimate, at 5% significance and 80% power, telling 40% from 10% takes about 30 replies
  per condition, pooled over the elicitors; telling 20% from 5% takes about 70. So step 1 measures the
  rates first, and they size the rest.
- **If the elicitors prove weak**, the replayed October exchanges are the strongest available: each is
  known to have drawn the voice once.

**Conditions, in order, each step deciding whether the next is needed:**

1. **The user card, on and off**, under the shipped Aria prompt. Plus a plain one-line system prompt, as
   the base rate: a voice that appears there too belongs to the model rather than to anything we wrote. If
   the voice mostly goes with the user card, the cause is found, and step 2 is about the card.
2. **Within the card**: the tone paragraph, the interaction-style paragraph, and the "critical or
   unprompted" sentence alone, each removed in turn.
3. **Only if step 1 says the system prompt matters**: its lines removed one at a time ("highly
   intelligent", "if the user is incorrect, say so", "go straight to the point", "raw thoughts rather than
   a structured report", the pep-talk), then the ones that mattered kept together, to see that they suffice.
4. **The candidate rewrite**: Aria's card carrying the register explicitly, worded as register rather than
   trait, under the modernized prompt and with no user card. It passes if it matches step 1's "on" rate.

**Samples.** Temperature 1, so the measured quantity is a rate: five to ten samples per condition and
prompt to start with, adjusted once step 1 has given the rates. Step 1 is three conditions, about 30–60 turns; the whole of step 3 would be some 450.

**Apparatus.** `raven.librarian.sweep`, an item being a (condition, prompt, sample) triple, so that a long
run resumes; the judge is a second sweep over the replies. The system text is a per-run override on
`llm_settings` (`character_card`, `system_prompt`, `user_card`), so no prompt file is edited. Each turn is
saved whole, as `investigations/README.md` asks of a probe running LLM turns.

**Model.** Qwen 3.8 27B, the current default, which produced the October examples; Qwen 3.6 35B-A3B as a
second if the result looks model-specific.

## The outcome it is for

If the hypothesis holds, the register moves into Aria's card, written deliberately, so that she has it for
every user, user card or none. The 2024 prompt-craft can then go without taking the voice with it.
