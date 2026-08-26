# Does the model's chat template already open the thinking block?

**Question.** When a backend hands Raven the raw generated stream — no server-side reasoning parsing — does a
thinking model's reasoning arrive with an opening `<think>` that `llmclient.StreamParser` could detect, or
does the chat template put the model *inside* the block before generation starts, so that only the closing
`</think>` ever reaches us?

**Answer: every Qwen in the archive opens the block in its template.** Measured 2026-08-26 against
`~/llms`: Qwen 3.5-4B, 3.5-9B, 3.6-27B, 3.6-35B-A3B and 3.8-27B. All five render a generation prompt ending

```
<|im_start|>assistant\n<think>\n
```

so the model begins already thinking and emits only the close. (The `mtp-` file is the 3.8-27B's MTP head,
shipped separately and paired with that model rather than a model of its own; it reports the same template,
as expected. The `mmproj` files carry no template at all.)

## Why it matters

`llmclient.StreamParser` starts in `_PS_TEXT` and enters `_PS_THINK` only on an *opening* tag. Against a
single-channel backend, therefore, the whole reasoning phase of every one of these models would arrive as
`content` events: rendered as the visible answer rather than as thinking, and — once the thinking phase is
timed — attributed to the answer.

**Nothing in Raven prevents that. The backend does.** LM Studio parses reasoning server-side and delivers it
on the `reasoning_content` channel, which is why none of this is visible in normal use. That parsing is
load-bearing for the exhibit, and this measurement is what says so: the exposure is not confined to unusual
models, it is every model we run, on any backend that leaves the parsing to us.

`TODO_DEFERRED.md`, "Streaming thinking shows as gray (not blue) for models that pre-fill the opening
`<think>` tag", is the item this corrects — it was filed as being about QwQ-32B and Qwen3-2507-as-served-by-ooba,
which reads as a curiosity.

## The second reading: the non-thinking branch

The probe renders both branches, because the pair is what the planned *Enable thinking* toggle acts on. With
`enable_thinking=False` all five templates end

```
<|im_start|>assistant\n<think>\n\n</think>\n\n
```

which is byte-identical to the prefill string recorded in `TODO.md` under "Thinking toggle". That claim was
established in July against the then-current models; it still holds on 3.5, 3.6 and 3.8. So the prefill
mechanism reproduces the template's own non-thinking mode exactly, rather than approximating it.

## Scope of the answer

The template ships *with the model*, and this reads it from the file. What a given server renders is a
separate question: a backend may override the template with its own copy. So this says what the model asks
for, not what any particular backend does with it — for the latter, ask the backend.

## Apparatus

| Script | What it answers |
|---|---|
| `probe_chat_template.py` | For every `.gguf` under a model directory (default `~/llms`): does its chat template open a thinking block at the generation prompt, what does the non-thinking branch render, and do the two branches differ. Reads `tokenizer.chat_template` out of the GGUF and renders it locally — no backend, no inference |

Re-run it whenever the model generation turns over; the question comes back each time.
