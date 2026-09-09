# Raven-librarian's prompt texts

The prose the AI is set up with, as files rather than as strings inside a `.py`. These are the **factory
defaults**; to change them, copy the file you want into

    ~/.config/raven/librarian/prompts/

keeping the filename. An override *replaces* the shipped text rather than adding to it, so start from a
copy of the original. Raven logs which one it loaded, at INFO.

| File | What it is |
|---|---|
| `system.md` | Instructions that hold regardless of who or what is at either end of the conversation. Ships **empty** — see below. |
| `interaction.md` | The shared half of every character card: the facts about the deployment, and how to behave. Spliced into a character's own card wherever it writes `{interaction}`. |

**Neither participant's own card is here**, because each travels with whoever it describes:

- The **AI character's** card is `aria1.md` beside `aria1.json`, among the avatar assets. See
  `raven/avatar/characters.py`.
- **Yours** is `juha.md` beside `juha.json` in `~/.config/raven/librarian/users/`, which nothing ships —
  it is yours to write. See `raven/librarian/userprofile.py`.

What is left here is what belongs to neither: instructions that hold whoever is answering and whoever is
asking.

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
| `{interaction}` | The whole of `interaction.md`, filled in. | a character's own `.md` only |

The same `{user}` and `{char}` work in a character's card and in your own profile's card.

That is the entire list.

**A literal brace has to be doubled.** `{{` gives you `{`, and `}}` gives you `}`. An unescaped `{` that
is not one of the names above stops Raven at startup, and so does a lone `}`. That is the usual way an
override goes wrong, and it happens most often when a prompt includes JSON or code — the error names your
file, the placeholder it did not recognize, and the ones it does.

## `{model}` and `{context_length}` are gone, on purpose

They existed up to v0.2.8, and using one was a trap: **this text is built once, at startup, and stored as
the message the chat is rooted at.** A fact written here is therefore frozen at the value it had then. Load
a different model without restarting and the stored text goes on asserting the old one — and a model has no
way to doubt what its own system message tells it about itself.

Raven states both in the per-turn system message instead, next to the date, where they are re-read every
turn. The date is out of these files for exactly the same reason.

So an override that still uses one now stops Raven at startup, naming it. That is the intended outcome: a
loud failure you can fix, rather than a quiet sentence that is wrong in a way nobody can see.

## Spacing between the parts

**You do not need to leave blank lines at the start or end of a file.** Raven strips each part and joins
them with exactly one blank line between, so the spacing is the same however your editor saves the file —
and a trailing newline, which most editors add, changes nothing.

To separate parts *visibly*, put a Markdown horizontal rule (`-----`) in the prose. The whole block already
ends with one, which is what closes it off from the conversation.

## The four-space trap

Do not indent a paragraph by four spaces unless you want a code block: this is Markdown, and the model
reads it as one. Raven's own character cards were indented that way until 0.2.9 — the prose was written
inside a `textwrap.dedent` whose dedent ran *after* the interpolation, leaving nothing to strip — so the
paragraph naming the character reached the model as code.
