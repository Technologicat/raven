# Which templates keep earlier thinking, and whether Librarian can tell

Measured 2026-10-20, while building `llm_retain_thinking` and the check that notices it set wrongly.

**The question.** Librarian sends every stored thinking trace when `llm_retain_thinking` is on, and counts
them in its context readout. Whether the model then sees them is up to the chat template, which neither
Librarian nor the backend can see. So: does the backend's reported prompt size say what the template did,
and does `llmclient.template_strips_thinking` read it correctly?

## Scripts

| script | what it answers |
|---|---|
| `probe_retention.py` | Against the configured backend: the prompt size with the earlier traces sent and without, the local count both ways (estimated and exact), and whether the rule agrees with the direct measurement |
| `render_template.py PATH.gguf` | What a GGUF's own template does with earlier thinking, rendered locally with Jinja, before any runtime sets its variables. Positive control: the same render with `preserve_thinking=True` |

## Results

LM Studio on the backend machine, three exchanges with a trace of about 2000 tokens across them:

| model | backend: traces sent | not sent | template | rule |
|---|---|---|---|---|
| Qwen 3.8 27B, Q4_K_XL | 2349 | 300 | keeps | keeps |
| Qwen 3.6 35B-A3B, IQ4_NL_XL | 2315 | 272 | keeps | keeps |
| Qwen 3.5 9B, Q4_K_XL | 257 | 257 | strips | strips |

Librarian's own count of the same messages: 2124 tokens with the traces and 81 without, exact (the GGUF
tokenizer); 2641 and 49 before the tokenizer had loaded, from the tokens-per-character estimate.

**What the numbers establish.**

- **The backend's figure is the local count plus a constant framing**: 300 − 81 = 219 without the traces,
  2349 − 2124 = 225 with them, on Qwen 3.8. That framing is what the prompt carries beyond the stored
  messages, the per-turn injects and the template's own markup, which the local count does not include. It is the premise of the rule, and it
  holds under exact counts.
- **The rule agrees with the direct measurement on all three**, on the exact count and on the estimate.
- **The stripping case is the one the check exists for.** On Qwen 3.5, the backend's 257 against a local
  2124 is less than half, so the short-count check alone (`prompt_size_report_looks_whole`) would have
  refused the correct figure and left the over-count on screen.
- **The estimate's error depends on the text.** It put the traces at 2592 tokens against 2043 exact, 27%
  high, and the short remainder at 49 against 81, 40% low. One sentence repeated forty times is a poor
  sample of real chats, so this says the error is large and varies by text, not how large it usually is.

## Why Qwen 3.6 keeps it: the runtime, not the template

Qwen 3.6's template strips earlier thinking unless `preserve_thinking` is true (`render_template.py`
confirms it on the GGUF the backend serves). No saved per-model config sets that variable there. The
backend's server log says what does, at every model load:

```
srv  init: chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)
```

So the runtime, llama.cpp's server going by the `srv` prefix and the flag, sets the variable for any
template that offers one. That is in the backend machine's logs from 2026-07-28 on, the earliest log
there mentioning it; no logs from the days before survive on that machine, so when it changed is not
established. `../thinking-toggle/` measured 3.6 stripping on 2026-07-27, which fits a change at about
that time, but that measurement may have been made on another machine.

What follows for a user: on a current LM Studio, a template with a `preserve_thinking`-style switch keeps
earlier thinking unless an override sets the switch false. A template with no such switch, as Qwen 3.5's,
does whatever it does regardless.

## Two side findings

- **The KV cache quantization is in LM Studio's saved per-model config**,
  `~/.lmstudio/.internal/user-concrete-model-default-config/PUBLISHER/MODEL/*.json`, as
  `llm.load.llama.kCacheQuantizationType` and `vCacheQuantizationType` (`q4_0` here). That is the setting
  `raven.librarian.sweep` has the operator declare, since `/api/v1/models` does not report it. Readable
  automatically only where the run is on the machine LM Studio is.
- **Next:** the estimate's error on real chats, measured over the short chats in the live datastore, as
  groundwork for fixing its overshoot.
