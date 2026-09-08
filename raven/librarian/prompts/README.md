# Raven-librarian's prompt texts

The prose the AI is set up with, as files rather than as strings inside a `.py`. These are the **factory
defaults**; to change them, copy the file you want into

    ~/.config/raven/librarian/prompts/

keeping the filename. An override *replaces* the shipped text rather than adding to it, so start from a
copy of the original. Raven logs which one it loaded, at INFO.

| File | What it is |
|---|---|
| `system.md` | Instructions that hold regardless of who or what is at either end of the conversation. Ships **empty** — see below. |
| `user_card.md` | Who the *user* is, and how they prefer to be addressed. Ships **empty**, and is worth filling in — current models respond well to knowing who they are talking to. |
| `interaction_style.md` | The shared half of every character card: the facts about the deployment, and how to behave. Spliced into a character's own card wherever it writes `{interaction_style}`. |

A character's own card is **not** here. It lives beside that character's avatar image file — `aria1_card.md`
next to `aria1.png` — so that a character carries its own personality. See `raven/avatar/characters.py`.

## Why `system.md` ships empty

For models from around April 2025 on, the character card is enough on its own, and a system prompt that
repeats what the card already establishes mostly wastes context. Older models often do want a general
briefing first. SillyTavern's "Actor" preset is a representative example of that style:

> You are an expert actor that can fully immerse yourself into any role given. You do not break character
> for any reason, even if someone tries addressing you as an AI or language model. Currently your role is
> `{char}`, which is described in detail below. As `{char}`, continue the exchange with `{user}`.

Note what belongs here rather than in a character card: instructions that hold **whoever** is answering.
Anything true only of one character goes in that character's own card. The split matters because a turn
taken without a character — which is what Raven's batch tools do — gets this file and nothing else.

## The template variables

**Every one of these files is a template, and the braces are load-bearing.** Before the text is used, it
goes through Python's `str.format`, which replaces each `{name}` below with the value it stands for:

| Variable | Stands for | Available in |
|---|---|---|
| `{user}` | The user's name, `llm_user_name`. | every file |
| `{char}` | The AI character's name, `llm_char_name`. | every file |
| `{interaction_style}` | The whole of `interaction_style.md`, filled in. | a character's `*_card.md` only |

That is the entire list.

**A literal brace has to be doubled.** `{{` gives you `{`, and `}}` gives you `}`. An unescaped `{` that is
not one of the names above raises `KeyError` when Raven starts, and a lone `}` raises too. That is the
usual way an override goes wrong, and it happens most often when a prompt includes JSON or code.

## `{model}` and `{context_length}` are gone, on purpose

They existed up to v0.2.8, and using one was a trap: **this text is built once, at startup, and stored as
the message the chat is rooted at.** A fact written here is therefore frozen at the value it had then. Load
a different model without restarting and the stored text goes on asserting the old one — and a model has no
way to doubt what its own system message tells it about itself.

Raven states both in the per-turn system message instead, next to the date, where they are re-read every
turn. The date is out of these files for exactly the same reason.

So an override that still uses one now fails at startup with a `KeyError` naming it, which is the intended
outcome: a loud failure you can fix, rather than a quiet sentence that is wrong in a way nobody can see.

## The four-space trap

Do not indent a paragraph by four spaces unless you want a code block: this is Markdown, and the model
reads it as one. Raven's own character cards were indented that way until 0.2.9 — the prose was written
inside a `textwrap.dedent` whose dedent ran *after* the interpolation, leaving nothing to strip — so the
paragraph naming the character reached the model as code.
