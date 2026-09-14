---
name: helpcard
description: How Raven's F1 help cards are written and checked — pages and what earns one, the hotkey table's hard no-wrap rule, the continuation idiom, how a key is written in prose, card sizing and the silent clipping it can cause, and the audit scripts. Use BEFORE editing any `hotkey_info`, `helpcard.page`, `HelpWindow` call or card prose, before adding or rebinding a hotkey in an app that has a card, and before concluding that a card is correct — which cannot be done by reading the source.
---

# Writing a help card

Every GUI app in the constellation has one, on `F1`, built from `raven.common.gui.helpcard`. Raven-librarian's
is the one the others are brought to; Raven-visualizer's carries the clearest statement of *why* the shape is
what it is, in the comment above its `HelpWindow` call. Read one of those before writing a new card.

## The standing instruction

**A card cannot be checked by reading its source.** Whether a cell wraps depends on column widths the table
computes from the content, and whether a row is visible depends on a height that may or may not be fitted —
so the source can be perfectly correct while the card on screen is missing rows. Raven-cherrypick's card
omitted its own `F1` row for as long as it existed, and every reading of `hotkey_info` said otherwise.

**So: launch the app and look.** Driving it costs one window mapping and no synthetic keys, because every
GUI app takes `--repl` and the client is pipe-scriptable (`live-gui-testing` skill; `CLAUDE.md` carries the
focus etiquette, which must fire *before* the decision to launch):

```bash
printf 'print("SHOW", _help_window.show())\n' | timeout 30 python -m unpythonic.net.client localhost
# then, per page:
printf '_help_window.next_page()\nprint("P")\n' | timeout 20 python -m unpythonic.net.client localhost
import -window "$WID" shot.png
```

**What to look for, in this order** — the first two are invisible in the source:

1. **Rows in `hotkey_info` that are not on screen.** Compare the last row of each column-group against the
   source. A card has no scrollbar by design; overflow is simply cut off, with no warning unless the height
   fitter ran and had to clamp.
2. **Cells that wrap to two lines.** See the no-wrap rule below for why this is not cosmetic.
3. **Blank-row grouping lining up across the column-groups.** They share table rows, so one wrapped cell
   pushes every group down from there and the grouping stops meaning anything.

## Sizing, and the clipping it causes

The four full-size apps — librarian, visualizer, cherrypick, xdot_viewer — take their numbers from
`raven.config`: `GUI_HELP_WINDOW_W` / `_H` for the card, `GUI_MAIN_WINDOW_W` / `_H` for the app window. One
value each, because they answer one question. The conference timer and the two avatar editors keep their own
(laid out for windows of their own size), and the file dialog sizes its card to the dialog it belongs to.

**The height is a starting value, not a size.** `HelpWindow._fit_height_to_pages` measures each page while
the card is parked offscreen and grows the card to the tallest, clamping to the viewport and logging if it
has to. It is skipped only for a card whose owner sizes it itself through `on_parked` — `fdialog` is the one
that does.

- **It only ever grows.** A card configured taller than its content keeps the empty space.
- **Buy width rather than pay in clarity.** The house style treats a dropped article as a concession bought
  with horizontal space (`raven-style-guide.md`, *User-facing text*). If cells are wrapping, the first
  question is whether the card can simply be wider, not which words to cut.

## Pages

**Split by scope, not to find room** — though a split usually finds some. The Visualizer's comment is the
statement of it: page one is the app's keyboard and nothing else, so it is a reference a reader can
screenshot and keep beside the app, which is what the card's header invites and what prose sharing the page
took away. A later page is what the app *does*, read once.

- **A distinct view or mode with its own keys earns its own page**, with the prose that explains it.
  Librarian's chat graph; Raven-cherrypick's compare mode.
- **The key that *enters* a mode stays on the main page.** Librarian keeps `Alt+G — Chat graph` on its
  keyboard page. Which key reaches a mode is part of the main UI, even when a later page explains the mode.
- **A page is named for what it holds** — *Keyboard*, *Compare*, *Features* — never for its place in a
  sequence (*More*, *Page 2*), which the `2 / 3` beside it already says.
- **If a page outgrows one screen, add a page rather than cutting prose.**

## The hotkey table

**Nothing may wrap. This is a hard constraint, not a preference.** The column-groups share table rows, so a
cell that wraps heightens the row for *all* of them, and the blank rows that group related keys stop lining
up — which is the thing the layout spends its space to achieve. Measured on a three-group card at 1400 px:
writing in the articles the house style asks for made eight cells wrap. At 1700 px, none of them do.

- **Grammatical English, articles included** — *"Open a folder"*, not *"Open folder"*. Terse phrasing is a
  concession for space; make it only where a *measurement* says space is short.
- **Sentence case**, never Title Case.
- **A row must read in the order it is met.** The table sits above the prose, so a row that forward-
  references the prose explains nothing.
- **The same key on two pages: one Action string, different Notes.**
- **`F1` is `"Open this help card"`.** Every card has this row, so it is the one most likely to be compared
  across apps; it read three ways before 2026-09-14.

### The continuation idiom

A row that varies its neighbour is written `key_indent=1, action_indent=1` with an action beginning `"..."`:

```python
env(key_indent=0, key="Alt+Up",  action_indent=0, action="Up one level",            notes=""),
env(key_indent=1, key="Ctrl+Up", action_indent=1, action="...the same, one-handed", notes=""),
```

**It requires adjacency** — "the same" needs the row above to be what it refers to. Where two rows spell one
action and are *not* adjacent, that is the defect: group them so one can continue the other. (xdot's
`"Pan view"` appeared twice, unindented and three rows apart.)

**Repeating one continuation under several parents is not that defect.** Cherrypick's `"...all selected"`
appears under *Mark as cherry*, *Mark as lemon* and *Clear the mark*: one modifier applied to three parents,
each directly above its continuation, and each reads correctly in order.

## Prose

`self.prose_columns(gui_parent, left, right)` with `helpcard.section(heading, *paragraphs)`. Newspaper
columns — a reader finishes one column before crossing, so a section belongs wholly to one of them. A
section may have `heading=None` for an opening block.

- **A key named in prose is bold, in the text colour**: `**Ctrl+S**`. The highlight colour (`self.c_hig`) is
  for emphasis and contrast, not for keys — a key wearing it reads as a different kind of thing from the
  same key one card over.
- **Say what no key table can say.** That is what earns a prose page. Cherrypick's marks *move the user's
  files* into `cherries/` and `lemons/`; its zoom and pan survive between images of the same dimensions, so
  flicking between them compares one detail across near-identical shots. Neither is deducible from a key.
- **Say what the app is, and who does the work.** Most of this constellation has a model in it, so *"Raven-
  cherrypick triages a folder"* is read as a claim that it classifies them. It is a GUI for doing it by hand.
- **Source it from the code or an existing README**, not from what the keys imply. Raven-avatar-pose-editor's
  prose said `Ctrl+S` saves a JSON emotion template; `save_image` writes the PNG *and* drops the pose beside
  it, and the card had been right all along.

## Before you finish

- `python scripts/check_hotkey_tooltips.py` — is every key named on a control that triggers it? An app read
  through by hand goes in `SIGNED_OFF`, after which a newly bound key with no caption fails the run. An entry
  there claims a person looked and found no widget doing the same thing; adding a control for one is a
  *feature*, and the entry should then go.
- `python scripts/check_option_lists.py` — does the card offer every key its README documents?
- **Read the app's key handler**, which the scripts cannot. A card may legitimately omit keys: the
  `Ctrl+Shift+` M/R/T/L debug keys (DPG's own developer windows) are deliberately off every card and live in
  the READMEs instead.
- **Check the captions against the card.** A button's bracketed hint and the card's row describe one key and
  should agree — xdot's tooltips said `[numpad +]` where its card said `Numpad +`, which is both an
  inconsistency and why the scan could not match them.
