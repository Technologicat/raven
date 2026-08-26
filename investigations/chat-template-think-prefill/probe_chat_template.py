#!/usr/bin/env python
"""Ask each local `.gguf` whether its chat template opens a thinking block at the generation prompt.

The question this answers: when a backend hands us the raw stream — no server-side reasoning parsing —
does the model's reasoning arrive with an opening `<think>` we could detect, or does the template already
put us inside the block, leaving only the closing tag on the wire?

Reads `tokenizer.chat_template` straight out of the GGUF and renders it here, so it needs no backend and
no inference: the template ships with the model, and what a *server* does with it is a separate question.
Renders both branches, because the pair is what the "Enable thinking" toggle acts on:

  - `enable_thinking=True`  -> the generation prompt Raven gets by default.
  - `enable_thinking=False` -> the template's own non-thinking branch, which is the string the prefill
                               mechanism reproduces (`TODO.md`, "Thinking toggle").

Usage::

    python probe_chat_template.py [MODEL_DIR ...]      # default: ~/llms

Two caveats about the scope of the answer. A backend may override the template with its own copy, so this
describes what the *model* ships rather than what a given server renders. And an mmproj or MTP file is a
companion to a model rather than a model, so its template is the parent's; they are reported, not filtered,
since which files are which is a fact about the archive rather than about the templates.
"""

import glob
import os
import sys

import gguf
import jinja2

__all__ = ["chat_template_of", "render_generation_prompt", "probe"]

# One user turn is enough: the generation prompt is the tail, and what precedes it does not change whether
# the template opens a thinking block.
_MESSAGES = [{"role": "user", "content": "hi"}]

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def chat_template_of(path: str) -> str | None:
    """Return the Jinja chat template stored in the `.gguf` at `path`, or `None` if it carries none."""
    reader = gguf.GGUFReader(path, "r")
    field = reader.fields.get("tokenizer.chat_template")
    if field is None:
        return None
    return str(bytes(field.parts[field.data[0]]), encoding="utf-8")


def render_generation_prompt(template: str, *, enable_thinking: bool) -> str:
    """Render `template` for one user turn, asking for a generation prompt.

    `enable_thinking` is passed to the template as a bare name, which is how the Qwen family spells it.
    A template that does not use the name ignores it, and renders the same string either way — which is
    itself an answer, so it is not worth guarding against.
    """
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
    return env.from_string(template).render(messages=_MESSAGES,
                                            tools=None,
                                            add_generation_prompt=True,
                                            enable_thinking=enable_thinking)


def probe(model_dirs: list[str]) -> None:
    """Print, for every `.gguf` under `model_dirs`, whether its generation prompt lands inside a think block."""
    paths = sorted(path
                   for model_dir in model_dirs
                   for path in glob.glob(os.path.join(os.path.expanduser(model_dir), "*", "*.gguf")))
    if not paths:
        print(f"No .gguf files found under {model_dirs}.")
        return
    for path in paths:
        name = os.path.basename(path)
        try:
            template = chat_template_of(path)
        except Exception as exc:  # noqa: BLE001 -- an unreadable file is a result to report, not a crash
            print(f"{name}: unreadable: {type(exc).__name__}: {exc}")
            continue
        if template is None:
            print(f"{name}: no chat template")
            continue
        try:
            on = render_generation_prompt(template, enable_thinking=True)
            off = render_generation_prompt(template, enable_thinking=False)
        except Exception as exc:  # noqa: BLE001 -- likewise: a template we cannot render is a result
            print(f"{name}: template did not render: {type(exc).__name__}: {exc}")
            continue
        # Inside the block when the last open is later than the last close (and there is an open at all).
        prefills = _THINK_OPEN in on and on.rfind(_THINK_OPEN) > on.rfind(_THINK_CLOSE)
        print(f"{name}")
        print(f"    thinking on,  tail: ...{on[-48:]!r}")
        print(f"    thinking off, tail: ...{off[-48:]!r}")
        print(f"    opens a think block at the generation prompt: {prefills}")
        print(f"    the two branches differ: {on != off}")


if __name__ == "__main__":
    probe(sys.argv[1:] or ["~/llms"])
