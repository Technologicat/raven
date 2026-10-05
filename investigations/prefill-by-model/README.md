# How fast each model processes a prompt

**Measured 2026-10-05**, LM Studio on the personal machine, every model loaded through the REST API at a
context of 32768 with `parallel` pinned to 1, flash attention on, thinking off (`reasoning_effort: "none"`).
The question came from Librarian's chat datastore, where Qwen 3.8 showed a median prefill of 9.7 s against
0.8 s for Qwen 3.6 35B-A3B, with Documents left on for the 3.8 replies, so that 50 matches rode along with
every prompt.

## The scripts

- **`probe_prefill.py`** — per model: three cold requests (a fresh nonce opens the system text, so nothing is
  cached) and two tail requests (the same nonce, a different question: only the tail is new, which is the
  shape of Librarian's query request). The prompt is about 9.6k tokens, the first 40000 characters of
  `raven/librarian/README.md`. Timings are LM Studio's own, from `/api/v0/chat/completions`' `stats`.
- **`probe_batch_and_usage.py`** — on one model: cold prefill at three physical batch sizes; whether the
  OpenAI-compatible endpoint's `usage.prompt_tokens` changes on a repeat; and the circle prompt with
  thinking on.
- **`results.jsonl`** — every response, whole, with every load's reported configuration.
- **`probe_extend.py`** — per model and prefix length: a cold request, an identical repeat, the same messages
  *extended* by a reply and a new question (Librarian's turn after its prefill), and a changed last message.
  `--parts` and `--tools` switch on the two ways Raven's requests differ in form. Writes
  `extend_results.jsonl`.

## Prompt processing

Cold, the same ~9.6k-token prompt, three runs each:

| model | kind | prefill | time to first token | generation |
|---|---|---|---|---|
| Qwen 3.6 35B-A3B (IQ4_NL_XL) | MoE, ~3B active | ~4200 t/s | 2.3 s | 87 t/s |
| Qwen 3.6 27B (Q4_K_XL) | dense | ~1700 t/s | 5.6 s | 32.5 t/s |
| Qwen 3.8 27B (Q4_K_XL) | dense | ~1500 t/s | 6.4 s | 37–50 t/s |

- **Dense against MoE is most of the gap**: the dense 27Bs process a prompt at 35–40% of the MoE's rate.
- **3.8 against 3.6 at 27B is small**: about 12% slower to process the prompt, cause unidentified.
- **The generation column is not like for like.** Multi-token prediction was on for 3.8 and off for both 3.6
  loads (`speculative_draft_mtp` in the recorded configurations; 3.8 ships the head, Juha 2026-10-05). The
  spread of 37–50 t/s across 3.8's runs fits a speedup that depends on how many drafts are accepted, which
  varies with the text; on the circle prompt it accepted 104 of 108.
- **The datastore's 9.7 s for 3.8 fits the Documents confound**: 50 matches are about 18k tokens, around
  12 s at 3.8's rate on their own.

With the prefix cached, only the tail is processed: 0.18 s on the MoE, 0.41 s and 0.52 s on the dense
27Bs. That is what Librarian's model-written search query costs once the conversation is cached.

## Batch size is not a lever here

Qwen 3.8, physical batch 512 (the default), 1024 and 2048, two cold runs each: ~1470, ~1455, ~1470 t/s. The
reloaded configurations report the requested sizes, so the setting took effect and made no difference.

## `usage.prompt_tokens` is the whole prompt on a cache hit

The same request three times to `/v1/chat/completions` (Qwen 3.5 4B): 1.57 s, then 0.16 s and 0.21 s, so the
repeats hit the cache, and every one reported 9556 prompt tokens. `/api/v0` reported whole counts on the
cached tail runs above, too. This agrees with `prompt-size-cache-relative/`'s finding that a repeat reports the
same figure; the short reports it saw on 0.3.x did not occur here, and why they happened there is still
unknown.

## Qwen 3.8 at low reasoning effort

The circle prompt ("draw an svg of a circle", from Simon Willison's
[post on Qwen 3.8 27B](https://simonwillison.net/2026/Aug/16/qwen-38-27b/), where the model's default
effort spent minutes on it): **61 reasoning tokens, 4.2 s in all**, answering with a plain circle. Thinking
on, no `reasoning_effort` sent, the effort set to low in the server's chat template (the maintainer's edit;
LM Studio takes no chat-template arguments from the client, so it cannot be set per request).

## A cached prompt is reused when extended, so Librarian's turns are missing it for another reason

**What was seen in Librarian** (2026-10-05, Qwen 3.8 27B): the context prefill between turns warmed the
cache, and the turn after it still processed its whole prompt. Part of that was Librarian's: the prefill sent
the stored system message, a turn sends it wrapped in the setup notice, the date and the model, so the two
differed at message 0. Fixed the same day (`scaffold.build_prefill_prompt`), and checked by recording the
request bodies in the running app: prefill and turn then agreed in every message and field but `max_tokens`.

**The turn still processed everything.** Measured from inside the app with Raven's own prompt builders, on
the chat's branch (one image) and on a text-only greeting branch:

| Branch | prefix | prefix again | prefix + a short message | that again |
|---|---|---|---|---|
| with an image, 6357 → 6443 tokens | 2.87 s | 0.50 s | 5.73 s | 0.43 s |
| text only, 1947 → 2034 tokens | 1.02 s | 0.38 s | 2.23 s | 0.29 s |

A prefill ending in a placeholder user message, so the reply above it rendered as it would in the turn, did
not help either (0.92 s, then 1.01 s for the turn).

**`probe_extend.py` says the backend is not the reason.** With plain prompts, all three models reuse an
extended prompt at every length tried, from about 2k to 24k tokens:

| Model | cold, ~2k → ~24k tokens | repeat | extend | changed last message |
|---|---|---|---|---|
| Qwen 3.8 27B (hybrid, MTP on) | 1.81 → 15.32 s | 0.26–0.38 s | 0.61–0.87 s | 0.45–0.65 s |
| Qwen 3.6 27B (hybrid) | 1.63 → 13.57 s | 0.25 s | 0.56–0.57 s | 0.41–0.42 s |
| Gemma 4 26B-A4B (sliding window) | 0.67 → 4.70 s | 0.07 s | 0.19–0.21 s | 0.13–0.15 s |

So an architecture that resumes only from checkpoints, considered first, is not it; nor is the prompt's
length. On Qwen 3.8 the extension stays cheap with Raven's form of request too: content as a list of parts,
a tool definition, both together, and through `/v1` streaming with a one-token cap and a `seed` (0.60–0.75 s
against 1.9–5.4 s cold).

**What is left is the content of Raven's prompts.** The leading suspect, untested: a turn's extension begins
with the clock's synthetic tool exchange, an assistant message with a tool call, directly after the prefix's
last assistant message, and a chat template may render an assistant message differently when another follows
it, moving the divergence back into the prefix. Next step: bisect with Raven's own builders against the
backend (`build_prefill_prompt`, then `build_turn_prompt` with the clock off, persona prefixes stripped, the
card shortened) until the extension turns cheap.
