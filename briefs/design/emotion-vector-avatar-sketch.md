# Sketch: the avatar's emotion read from the model's activations

**Status: a discussion sketch.** Written 2026-11-12 from a claude.ai design conversation of 2026-11-06.
**Decided** (Juha): the direction — drive the avatar's emotion from emotion vectors read off the LLM's
activations, as a replacement for or complement to the text classifier — and that the avatar should show
**the AI character's own state only**, not the user's or a story character's. **Chosen for testing**:
approaches 1 and 3 under *Whose emotion is it?*. **Open**: everything about the mechanism, the serving
stack included. Switching LLM serving stacks is acceptable if it brings feature upgrades (Juha).

**Scope, decided 2026-11-12** (Juha): activation readout is an **opt-in** for setups we run ourselves — tech
demos and the lab installation, where accurate emotions matter and the extra effort is acceptable. The text
classifier stays the default; the rest of the team will likely stay on LM Studio. So the readout stack does not
need to meet the one-command-install bar. **Not a gate for the lab installation**: the avatar already works off
text, so this lands as an upgrade whenever it is ready.

## What the user gets

Today the avatar's emotion comes from the generated text: `classify` (distilBERT, 28 emotions) runs on a
window of recent paragraphs (`common_text.EmotionWindow` in `chat_controller`'s AI-turn task), on the final
message, and on text sent to TTS, all through `avatar_controller.update_emotion_from_text`. So the face can
only perform what the text already says.

Read from the activations, the face can show what the text does not. People read faces as the channel that
leaks — the text says what is intended, the face shows what is going on underneath — so this is also closer
to how humans interpret a face (Juha, 2026-11-06). A face that looks nervous while the text says "certainly!"
is a UX oddity and a research datum at once, which makes Librarian an instrument as well as a frontend.

## Background

Anthropic, *Emotion concepts and their function in a large language model* (Apr 2026;
anthropic.com/research/emotion-concepts-function, with the full paper on transformer-circuits). Two results
matter here; re-read the article before relying on either.

- **Internal and expressed emotion can decouple.** Steering toward "desperate" produced cheating carried out
  in calm, methodical text. A text classifier on that output reads neutral; the vectors would not.
- **The vectors are local.** They encode the emotional content most relevant to the current or upcoming
  output — which may be the user's emotion, a character's, or the speaking character's own. In the paper's
  example, "desperate" tracked the CTO while the model read the CTO's email, and the model's own state once
  it reasoned as the agent. This is the problem the *Whose emotion* section addresses.

## Mechanism — three pieces

**Vectors.** Anthropic's are internal to Claude, so rebuild them for Qwen with their recipe: have the model
write short stories per emotion, run the stories back through, record the mean activation at a chosen layer.
Use distilBERT's 28 labels as the emotion set, so the avatar's emotion presets (`avatarutil`, the
`_defaults.json` in `raven/avatar/assets/emotions/`) and the expression mapping stay as they are. One
extraction per model; store the vectors with a record of the model, quant and layer they came from.

**Readout.** A dot product of the residual stream with N vectors at each generated token — negligible next to
the forward pass. A linear readout should be much less sensitive to curvature in the representation than
linear *steering* is (from the belief-state geometry discussion), so plain dot products should do for display.
Steering, if it ever comes up, would need the manifold-aware treatment.

**Plumbing — the real obstacle.** LM Studio's OpenAI-compatible API does not expose hidden states. Candidate
route: a thin llama.cpp-based server that taps intermediate tensors via llama.cpp's eval callback (the
`imatrix` tool uses this mechanism) and streams per-token emotion scores alongside the text. llama.cpp already
has the reverse direction built in (`llama-cvector-generator`, `--control-vector`), which suggests the
extraction side is close to existing tooling too. Where it runs is open: a raven-server module, or on the Strix
Halo box. To check: Qwen 3.8's hybrid layers (Gated DeltaNet interleaved with attention, 3:1) — which residual
stream to tap. The attention layers are the natural first candidate.

## Whose emotion is it?

Three approaches came up, cheapest first. **1 and 3 are the ones to test** (Juha).

1. **Sample at "about to speak" positions.** Anthropic measured just before the response. If the vectors
   encode what is relevant to the upcoming output, then at turn start and at sentence boundaries the upcoming
   output is the character's own reaction. Update the face only there, smoothed in between; this should filter
   out much of the mid-sentence tracking of the user or of characters. A hypothesis, cheap to test — and the
   face should not change expression every token anyway. Sentence boundaries are already available from the
   streaming path.

2. *(Not chosen.)* Gate by perspective: a small classifier on the sentence being generated, letting through
   first-person and plain assistant voice, blocking "you must be feeling…" and narration.

3. **Learn a self/other attribution direction.** Build each emotion vector twice: from stories where the
   narrating character feels it in first person, and from stories where someone else does, everything else
   held constant. The difference is the self/other direction, if there is one. Read the emotion vector and
   that direction together, and show only emotions with a high self component. Prior work on how models bind
   properties to entities in context (Feng & Steinhardt 2023, binding IDs) points toward attribution being
   represented separately from content; the CTO-to-agent shift above is at least consistent with that.

**The experiment inside approach 3.** With the Aria character card, recent Qwen CoTs say "I should respond as
Aria…" — the default character roleplaying another on top (Juha's observation). So "self" might resolve to
Aria, to Qwen, or to both on different layers. For the avatar, Aria is the target. Any of the three outcomes is
worth recording. A cheap first probe for the same question: compare readings with and without the card on
identical prompts.

## Things this touches that are easy to miss

- **Replay.** `_avatar_get_emotion_from_text` is LRU-cached so that "speak again" and switching between
  messages stay cheap. Activation readings exist only at generation time, so they would have to be stored
  with the message (per sentence, say) for replay to show the same face — or replay falls back to the text
  classifier.
- **Backends without hidden states.** Remote APIs and LM Studio will never expose them, so the text
  classifier stays as the fallback, and the avatar needs to know which source it is getting. **Log it**
  (Juha, 2026-11-12): add the source to the existing INFO line in `update_emotion_from_text`
  ("updating emotion to '…'"), and write one INFO line whenever the source changes, so one grep shows which
  signal drove a given face.
- **Thinking.** Whether readings during the reasoning channel drive the face, or only those during the
  visible answer, is a choice to make once there are readings to look at.

## What it would split into

Probably two briefs, each able to close on its own: (a) extraction plus an offline readout on recorded
conversations — answers the layer question and the Aria/Qwen question, needs no serving changes; (b) the
streaming server and the avatar wiring. (a) is also the cheap way to find out whether (b) is worth building.
