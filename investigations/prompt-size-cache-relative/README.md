# A backend's `prompt_tokens` can be wrong in either direction, and the machine can answer without it

**Measured 2026-08-22 and re-measured 2026-08-24, LM Studio 0.3.x serving `qwen3.5-9b` Q4_K_XL at a loaded
context of 131072.** The 08-24 pass overturns the 08-22 conclusion; both are kept below, because which
number was believed and why is the whole lesson.

Librarian's context-fill readout is two-stage: a local character-ratio estimate shown as `~X%`, then the
backend's own figure shown as `X%`. The second comes from `llmclient.prefill`, which sends the real prompt
with `max_tokens=1` and reads `usage["prompt_tokens"]`.

**On LM Studio that figure can come back an order of magnitude short — or half as large as the truth — for
the same content, and nothing in the response says which.** The mechanism is still unidentified. What is
established is that the figure cannot be trusted on its own, and that a machine serving a GGUF already holds
everything needed to answer the question offline.

## The true size, established two independent ways

One chat branch, three attached PDF fulltexts. The folded attachment text is **296370 characters**, the whole
wire payload **301589 characters** of message text.

| method | tokens | chars/token |
|---|---|---|
| backend, asked about a **unique** (uncacheable) copy of the attachment text | 85937 | 3.45 |
| the served model's **own tokenizer**, read out of the GGUF, offline | 85894 | 3.45 |

They agree to **0.05%** — and the 43-token gap is accounted for: the backend request carried a UUID prefix
and the chat template's framing, which the offline count did not. So the attachment text is ~85.9k tokens,
the whole prompt ~88.5k, and the branch is **68% of a 131072 window**.

Linearity was checked rather than assumed — 10k, 50k, 100k, 200k and 296370-character slices of the same
corpus, each sent with a fresh UUID prefix so nothing could be cache-hit. The ratio drifts from 4.28 to 3.45
across that range, which is why a *slice* cannot be extrapolated: this corpus's early pages are ordinary
prose and its later ones are references and tables, which tokenize far worse.

## What the backend said about the same content

| request | 2026-08-22 | 2026-08-24 |
|---|---|---|
| as-is | 8745 | **88524** |
| the same again | 8745 | 88524 |
| attachments not resolved — the conversation alone | 2630 | 2630 |
| nonce **prepended at the front** | 56365 | 56392 |
| nonce appended at the end (~600 tokens, verified sent) | 8745 | — |

Against a true ~88500. **The as-is figure was an order of magnitude low on one day and exactly right on the
other**, and the prepended-nonce figure is reproducibly ~64% of the truth on both. Only the conversation-alone
figure is stable and correct, which is presumably why nothing looked wrong until a large attachment appeared.

**The two variants send the same bytes.** Checked directly, because the whole finding turns on it:
serializing the branch as-is and with the nonce prepended gives 301589 and 301627 characters — the 38-character
nonce and nothing else, with the attachment text identical at 296370 in both. So the spread is the backend's
accounting, not Raven sending different prompts.

## What was ruled out

The 08-22 pass ruled these out, and they stay ruled out — a re-investigation need not repeat them:

- **The model's context being smaller than advertised.** `GET /api/v0/models` reports
  `loaded_context_length: 131072`; nothing was truncated for lack of room. (The GGUF itself declares
  `qwen35.context_length = 262144`, so the ceiling is LM Studio's loaded setting, not the model's.)
- **The attachments not reaching the wire.** `serialize_history_for_wire` produced 301589 characters with the
  documents folded in, and the fold runs for user-role attachments as well as tool-role ones.
- **Raven mutating its own stored messages.** `serialize_history_for_wire` deep-copies first, so its in-place
  content rewrite touches only the copy; the stored `text_file` parts survive.
- **The extraction failing and caching a placeholder.** The PDFs have text layers and are found.
- **A transient state in the running app.** Everything here reproduces from scripts with no GUI.

The 08-24 pass adds one:

- **"It reports the tokens it had to *process*, the rest being in the KV cache."** This was the 08-22
  headline explanation, and it is wrong: sending a byte-identical prompt twice reports the *same* figure both
  times (4045, then 4045, on a small synthetic prompt), and appending ~90 characters to it moves the figure by
  21 tokens, as a straight count would. A warm repeat is not discounted.

## Why the 08-22 conclusion was wrong, which is the part worth keeping

That pass concluded the prepended-nonce figure (56365) was the whole-prompt truth, and wrote down two
consequences: that the local estimate "runs 44% high", and that the branch was "43% full". Both are false —
the estimate runs about **8% low** (81158 against ~88500), and the branch is 68% full.

Nothing was measured carelessly. The error was that **every figure in that pass came from the instrument under
suspicion.** Four numbers from one backend were cross-checked against each other, and the one that looked most
principled — the one where the cache had provably been busted — was promoted to ground truth. It took a
measurement from *outside* the backend to see that it was wrong, and that measurement was available the whole
time: the tokenizer is in the GGUF file on the same disk.

The general form, worth recognizing before the next one: *when an instrument is suspected of lying, its
readings cannot arbitrate between each other, however cleverly they are varied.*

## What to do about it

**Configure a local tokenizer.** `llmclient.count_tokens` already prefers one (`config.llm_tokenizer_path`,
tier 1) over every backend figure, and it is exact and offline on any backend. It is `None` today, so the tier
that would have prevented all of this is present and unused. See the deferred item; the one piece of work is
that `_load_local_tokenizer` currently expects a directory of Hugging Face tokenizer files, while a
llama.cpp-family user has a `.gguf` — which carries the vocabulary, the merges and the chat template
(248320 tokens, 247587 merges, `tokenizer.ggml.pre = 'qwen35'` here).

**Keep `prompt_size_report_looks_whole`.** It refuses a reported figure far below the local estimate, which is
exactly the 8745 case, and it correctly believed 88524. Its threshold rationale needs rewriting, though: the
bound was set loose *because* the estimate was thought to run 44% high, and it does not — so the bound has more
room than it was given credit for. It cannot catch a figure that is wrong on the *high* side, and nothing here
says such a figure is impossible.

**A local tokenizer cannot help a remote backend**, where the GGUF is on another machine. There the estimate
and the sanity check remain all there is, so neither is redundant.

## Files

- `probe_prompt_size.py` — the backend's four answers for one branch, plus the local estimate and the wire size.
  Defaults to Librarian's configured datastore and HEAD; `--datastore` and `--head` point it at a scratch chat.
  **It sends the chat's contents to the backend you name.**
- `measure_true_size.py` — the 08-24 pass: same-bytes check on the two variants, chars-per-token across slices
  of the corpus, and the offline count from the GGUF's own tokenizer. `--gguf` points at the model file.

Re-run them after an LM Studio upgrade, or when adding support for a backend whose usage reporting is unknown:
"is this figure about the whole prompt" has no answer in the OpenAI schema, so it is per-backend behaviour that
can change under us.
