# Why an inline-code background vanishes on a help card

Backticked text in a help card's prose draws no background at all, while the same markup decorates
correctly in a chat message. Measured 2026-09-14, on DPG 2.3.1.

**The answer is visibility at decoration time, and nothing else.** `DearPyGui_Markdown` defers decoration
work to a worker thread (`CallInNextFrame`), and `text_attributes.Code.render` sizes its background quad
from `dpg.get_item_rect_size` of the text group. A widget DPG has not laid out has no metrics, so a text
group that is hidden when the worker reaches it reports `[0, 0]` — and the quad is built with zero area.
It is *built*: three items per span, present in the tree, occupying no pixels. That is why looking for a
misplaced box on screen finds nothing, and why brightening a screenshot finds nothing either.

## What was measured

`probe_nesting.py` renders one code span four ways, varying two things independently, and instruments
`Code.render` to report what it read:

| case | `rect_size` the quad was built from | on screen |
|---|---|---|
| flat parent, shown | `[211, 26]` | drawn, correctly placed |
| nested in a horizontal group holding a vertical one (the `HelpWindow.prose_columns` shape), shown | `[211, 26]` | drawn, correctly placed |
| flat parent, hidden at build time | `[0, 0]` | nothing |
| nested, hidden at build time | `[0, 0]` | nothing |

**Nesting makes no difference**, which is worth stating because it was the first hypothesis and it was
wrong: `prose_columns` was the visible difference between the cards that decorate and the cards that do
not, and it was a coincidence. Every multi-page card also uses it.

`get_widget_pos` reports `(0, 0)` for the hidden cases too, so this is not the `get_item_pos`-versus-
`rect_min` confusion that `guiutils.get_widget_pos` exists to paper over. There is no position to get.

**Revealing the widget afterwards does not repair it.** The probe shows the hidden cases, renders ninety
more frames, and the quads stay zero-sized: nothing re-runs the decoration, so the one chance to measure
is the frame the worker happened to wake on.

## Why this reads as "the card is broken"

A card of two or more pages builds every page and hides all but the current one, so a code span anywhere
past page one is decorated while hidden. Every observation we had fits that:

- Raven-avatar-pose-editor's card decorated correctly while it was a **single page** — shown, therefore
  measurable — and stopped the day it gained a second page.
- Raven-librarian's, Raven-cherrypick's and both avatar editors' multi-page cards decorate nothing.
- A chat message decorates fine: it is visible as it is built.

## What it does not explain

The *misplaced* box seen in a chat message on 2026-08-27 — a grey rectangle a couple of words to the right
of its text. That one is visible-but-wrong rather than absent, and re-rendering the message fixed it, which
says premature measurement rather than no measurement. **So there are two faults in `text_attributes.py`,
not one**, and only the second is the settling race that `TODO_DEFERRED.md` describes.

## What would fix it

Not measured, so treat as candidates rather than a plan:

- **Do not decorate a hidden widget** — defer until it is shown, which needs something to notice that it
  has been. The one chance to measure is the problem, not where the measurement goes.
- **Size from a build-time metric instead.** `dpg_markdown.get_text_size` answers from the font, not from
  the layout, and works whether or not the widget has been drawn.
- **Stop positioning the decoration at all**, and draw the span as an inline drawlist carrying both the
  quad and the text, laid out by DPG like any other item. This is the shape the URL secondary-action icon
  already uses in `text_attributes.py` — it is added with a plain `dpg.add_text(parent=parent)` and
  computes no position — and it removes the frame delay rather than working around it.

## How to run it

```bash
python investigations/dpg-markdown-decorations/probe_nesting.py
```

Needs a display, and maps a window for a few seconds. Prints to stdout — **do not pipe it**, and note it
leaves via `os._exit`, the renderer's worker thread not participating in DPG teardown.

## A second finding, which this probe fell into first

Building **wrapped** Markdown before the first frame used to hang the process rather than fail. A face
reaches the font atlas only between frames, and `get_text_size` retried without bound until it could
measure — which on the render thread is never, that thread being the one that would render the frame. It
raises there now, logging the stack that reached it, and `guiutils.bootup` carries the rule.

**The rule is about `wrap`, not about how many calls have been made**, which is worth stating because the
older wording said otherwise ("at most one `dpg_markdown.add_text` before the first frame", written against
DPG 1.11 where it presented as a segfault). Measured here:

| before any frame | outcome |
|---|---|
| two *unwrapped* `add_text` calls | both succeed |
| one *wrapped* `add_text`, plain words, no bold or italic | fails on the first call |

Wrapping is what forces a measurement; unwrapped text is never measured, so it needs nothing from the
atlas. That is also why the Visualizer's `markdown_font_loader_trigger_dummy` works as a preloader — it is
unwrapped, so it *loads* the four faces without measuring them.

So: build wrapped Markdown only after the loop is running. Any probe that builds it must do so after its
first frames, not inside the window construction.
