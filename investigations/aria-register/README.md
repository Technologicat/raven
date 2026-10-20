# What gives Aria her register, and how to keep it through the prompt rewrite

**Status: step 1 run and judged, provisional until the blind sample is rated** (2026-10-20). Gates `TODO_DEFERRED.md`'s "Modernize the
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
