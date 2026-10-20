# What gives Aria her register, and how to keep it through the prompt rewrite

**Status: a plan, nothing run yet** (2026-10-20). For `TODO_DEFERRED.md`, "Modernize the Librarian system
prompt / character card".

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
strive for excellence. Take pride in your work…" made the model visibly less tense. Asked about it, the
model said such instructions are ineffective, and pointed out that the study they come from does not
recommend them as such. That checks out: Battle and Gollapudi, ["The Unreasonable Effectiveness of
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
something wrong, a request the system handles clumsily.

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
prompt. Step 1 is three conditions, about 30–60 turns; the whole of step 3 would be some 450.

**Apparatus.** `raven.librarian.sweep`, an item being a (condition, prompt, sample) triple, so that a long
run resumes; the judge is a second sweep over the replies. The system text is a per-run override on
`llm_settings` (`character_card`, `system_prompt`, `user_card`), so no prompt file is edited. Each turn is
saved whole, as `investigations/README.md` asks of a probe running LLM turns.

**Model.** Qwen 3.8 27B, the current default, which produced the October examples; Qwen 3.6 35B-A3B as a
second if the result looks model-specific.

## The outcome it is for

If the hypothesis holds, the register moves into Aria's card, written deliberately, so that she has it for
every user, user card or none. The 2024 prompt-craft can then go without taking the voice with it.
