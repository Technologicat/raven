---
name: helpcard
description: How Raven's F1 help cards are written and checked — pages and what earns one, the hotkey table's hard no-wrap rule, the continuation idiom, how a key is written in prose, card sizing and the silent clipping it can cause, and the audit scripts. Use BEFORE editing any `hotkey_info`, `helpcard.page`, `HelpWindow` call or card prose, before adding or rebinding a hotkey in an app that has a card, and before concluding that a card is correct — which cannot be done by reading the source.
---

# Help cards in Raven

**`raven-style-guide.md` → *User-facing text* is the reference. This skill is the index into it.**

Everything durable lives there because a card's rules are rules about user-facing prose, and because that
file is what a person reads in an IDE — where this skill is invisible. Nothing here restates it.

## The standing instruction

**A card cannot be checked by reading its source, so open it and look.** Whether a cell wraps depends on
column widths the table computes at run time, and a card has no scrollbar, so anything past the bottom edge
is cut off in silence. Raven-cherrypick's card omitted its own `F1` row for as long as it existed, and every
reading of `hotkey_info` said otherwise.

Driving it costs one window mapping and no synthetic keys — every GUI app takes `--repl`, and the client
takes piped input. **Read `CLAUDE.md` → "Live GUI testing on a shared desktop" before launching**: mapping a
window takes the user's keyboard, and that has to be announced before the decision, which is earlier than
this skill loads.

```bash
printf 'print("SHOW", _help_window.show())\n' | timeout 30 python -m unpythonic.net.client localhost
printf '_help_window.next_page()\nprint("P")\n' | timeout 20 python -m unpythonic.net.client localhost
import -window "$WID" shot.png
```

The `live-gui-testing` skill has the launch, wait-for-ready, screenshot and shutdown recipes.

## Where the answer lives

| If you are about to… | Read, in `raven-style-guide.md` → *User-facing text* |
|---|---|
| write or reword any row of a hotkey table | the opening bullets — articles, sentence case, a row readable in the order it is met, one key described the same way twice |
| write a row that varies the one above it | the continuation idiom (`key_indent=1` + `"...the same, …"`), and which duplicated action text is a defect |
| add rows, or wonder why a cell went to two lines | *A help card cannot be checked by reading its source* — the no-wrap rule, its mechanism, and buying width rather than cutting words |
| decide whether something needs a second page | *Pages* — split by scope, page one is the keyboard alone, a mode earns a page, the key that enters it does not |
| set or change a card's size | *Card size, and what fits itself* — the shared constants, which apps are exceptions, and what fits itself when |
| write the prose on a page | *Prose on a card* — two columns, say what no key table can, source it from the code |
| style a key, a path or a UI label in prose, or wonder why a page reads as too red | *Prose on a card* → the three stylings — bold for a key, backticks for a literal, the highlight for a label to find on screen — and what counts as too much of the last |
| finish, before committing | *Before you finish* — the two audit scripts, and reading the key handler yourself |

## Adjacent material

- `raven/common/gui/helpcard.py` — `HelpWindow`, `page`, `section`, `prose_columns`, and the height fitting.
- `raven/librarian/app.py` — the card the others are brought to: three pages, and the fullest prose.
- `CLAUDE.md` → *Help cards*, and → *Hotkeys: one meaning per key* for what may share a key.

## When you learn something new

Record it in `raven-style-guide.md`, in the section it belongs to — that file is authoritative and this
skill must not grow a second copy. Add a row above only if it introduces a *new question* a reader might
arrive with.
