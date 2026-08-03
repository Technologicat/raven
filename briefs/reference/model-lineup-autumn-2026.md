# Model lineup, summer/autumn 2026

Which LLM Raven-librarian is developed and demonstrated against, decided 2026-07-28 on the strength of the
measurements in `context-inject-shape-measurements.md` rather than on reputation. **Qwen across the board**,
one model per hardware tier.

| tier | model | on disk | what it is for |
|---|---|---|---|
| 8 GB dGPU, mobile | **Qwen3.5-4B** | 3.59 GB | Working away from the desk. Leaves room for the avatar and a server module or two. |
| 16 GB dGPU, mobile | **Qwen3.5-9B** | 6.89 GB | The better mobile option where the card allows it. |
| 24 GB eGPU, at the desk | **Qwen3.6-27B** (dense) or **Qwen3.6-35B-A3B** (MoE) | 18.54 / 20.40 GB | Serious single-workstation use, whenever the external card is attached. |

The eGPU travels to the Researchers' Night demo (2026-09-26), so the demo runs on the top tier rather than on a
laptop-class model. See `../TODO.md` for the demo plan; the hardware shape recorded there — LLM alone on the
larger card, all nine raven-server modules on the internal one, the `config_dual_midvram` variant — is what this
lineup assumes.

## Why Qwen at every tier

Not a preference, an outcome. Every tier was measured against the alternative that was actually installed:

- **Qwen3.5-4B beat Gemma4-E4B outright at the 8 GB tier.** On retrieval it scored 24/24 — perfect at every
  corpus size from k=5 to k=40, in both the `user` and `tool` roles — where E4B managed 12/12 as `user` but only
  9/12 as `tool`, degrading above roughly ten results. The 4B also passed the vision checks that matter for
  attachments: it reads a Librarian screenshot's title bar (`Raven-librarian 0.2.8-dev`) and its smallest status
  readout (`~1% (913 / 131072)`) verbatim.
- **The 3.5-versus-3.6 generation gap did not appear** in anything measured. Qwen3.5-9B matched the 3.6 models on
  retrieval at both k=20 and k=100. That is what makes 3.5 acceptable at the mobile tiers rather than a
  compromise — though it is a statement about *these* task classes, not a general claim.
- **Gemma 4 is not dropped, only not first choice.** It works, it is the multilingual backup, and it now loads in
  every packaging under LM Studio 0.4.20. But it has one hard requirement the Qwens do not: it ignores a bare
  `tool` message entirely and confabulates a plausible answer in its place. Anything Raven sends it as tool
  output must carry a synthetic tool call. Since the chosen inject shape does that anyway, Gemma stays usable.

## What this decides, and what it doesn't

Decided: the models to develop against, and the model the September demo runs on. Raven stays backend-agnostic —
nothing here is wired into the code, and the lineup is a `config.py` matter.

Not decided, and deliberately left open:

- **What 3.6 actually improves over 3.5.** Nothing measured here distinguished them, which is a statement about
  the retrieval and date-handling tasks probed, not about the models. Worth reading the changelogs on a day when
  it matters.
- **Whether a ~12B model is worth installing** to fill the gap between the mobile and eGPU tiers. As of
  mid-2026 that size is **Gemma's alone** — Qwen 3.6 ships only at 27B and 35B-A3B, and the 4B/9B options in
  the table above are Qwen 3.5. So the candidate is **Gemma 12B**, not a Qwen, which changes the question:
  Gemma is already the multilingual backup rather than the first choice at its tier, so a 12B would have to
  earn the slot on quality rather than inherit it on family. Neither machine has one installed.
- **Anything above the single-workstation class.** Out of scope by construction.

One behavioural note that shaped the inject design and is worth remembering when picking prompts for these
models: Qwen takes instructions **literally**. A self-contradictory instruction does not get quietly smoothed
over — the 9B spent 52796 characters of reasoning refusing to resolve one and never answered. That literalism is
usually a feature, and it is the reason the reminder wording had to be fixed rather than worked around.
