# Brief: block-level Markdown in the chat view

**Researchers' Night work.** Unnumbered pending your call on where it sits relative to 16 — both are
exhibit-critical, and this one is smaller than 16 by some margin.

> **Line numbers are as of 2026-08-10** and want verifying; `chat_controller.py` has been moving daily.
> Treat every `file.py:NNN` as a pointer to a thing that exists, not as a coordinate.

**Supersedes three items in `TODO_DEFERRED.md`** — *Markdown ATX headings don't render in the chat view*,
*Fenced code block support in the Markdown renderer*, *Markdown tables don't render in the chat view* — and
**corrects the diagnosis all three share**. They are one item plus a footnote.

Origin: `investigations/todo-sweep-2026-08-10/`, batch 6, with `markdown_block_probe.py` establishing the
mechanism.

## The finding

**The chat view puts two independent barriers in front of block-level Markdown, and the vendored renderer is
behind both.** Neither is in the renderer.

**Barrier 1 — the colour wrapper.** `chat_controller._render_text` (`:529`) emits
`f"<font color='{color}'>{text}</font>"` around every paragraph before handing it over. With the open tag on
the same line as the content, CommonMark makes the whole thing an ordinary paragraph containing inline raw
HTML. A heading is a *block* construct and cannot occur inside a paragraph, so the `#` markers survive
verbatim. Measured: `<font color='...'>### A heading</font>` → `<p><font ...>### A heading</font></p>`.

**Barrier 2 — the single-newline split.** `_render_text_paragraphs` (`:1512`) splits on `"\n"` and renders
each line as its own call, so any construct spanning lines — a fenced block, a table, a multi-line list —
cannot form even before the wrapper applies.

**Why it looked like a renderer bug.** `**bold**`, `*italic*` and `` `code` `` are *inline* constructs and
are unaffected by either barrier, so the chat view renders almost everything correctly. "The renderer must
not support headings" is the natural conclusion, and three separate items reached it.

**What the renderer actually has.** `<h1>`–`<h6>` mapped since its initial commit (`parser.py:283` onward,
`__init__.py:213` onward to `font_attributes.H1`–`H6`, consumed at `text_entities.py:49`). `MessageEntityPre`
for fenced code (`parser.py:61`, `:215`), with its own attribute class and post-render machinery. **`table`
is the one block construct with no `case`** — the only genuine gap of the three.

## The steps

Independent enough to land separately, and step 1 alone is visible.

### 1. A colour parameter on the vendored renderer — **done 2026-08-25**

Landed as designed, and headings render. Two things the design below did not anticipate, both now in the
code:

- **`recreate_attributes` had to carry the colour explicitly.** `LineEntity.append` calls it on every line
  it takes, rebuilding the `AttributeController` from scratch, so without the carry the colour survived an
  unwrapped render and was lost by a wrapped one — every chat message, while the help card looked fine.
- **The colour parsing moved to `font_attributes.parse_color`**, shared with `Font.__init__`, so the
  argument and the `<font color=...>` attribute accept the same spellings by construction.

Threaded through `MarkdownText.__init__` rather than set as a module-level default (question 1 below, now
settled): Raven renders messages concurrently on background threads and colours them per role, so a global
would race a user message against a thought bubble.

Tests in `raven/vendor/DearPyGui_Markdown/tests/test_document_color.py`, including the negative control that
pins what the wrapper used to do. Step 6 landed alongside.

`dpg_markdown.add_text` (`__init__.py:308`) takes only `markdown_text, wrap, parent, pos, tag`, which is why
the wrapper exists. But colour is already carried internally — `font_attributes.Font(self.entity.color, ...)`
at `:194` — and the `add_text_bold` / `add_text_italic` / `add_text_bold_italic` helpers each take a `color`
kwarg. Thread a default colour through `MarkdownText.__init__` (`:237`) as the initial state instead of
requiring a `<font>` tag in the source, and stop wrapping in `_render_text`.

**This alone fixes headings in both paths**, streaming included: a heading is a single-line construct, so it
survives the split. It just cannot survive the wrapper.

*Leave the XML-token replacement alone.* `_render_text` also rewrites `<tool_call>` and `<think>` into bold
markers because the renderer silently drops unknown tags. That is a different problem and still real.

### 2. Remove the dead inline-`<think>` handling

`_render_text_paragraphs` consolidates inline `<think>...</think>` into one collapsible paragraph, and its
own docstring records that this is dead:

> since the June 2026 `reasoning_content` migration, thinking is separated out before render (at load by
> `upgrade_datastore`, live by the stream parser), so `content` no longer carries inline `<think>`

It is also the only reason the splitter exists. There is an existing deferred item for this removal; it was
filed as a tidy-up and is in fact the unblocker for step 3.

### 3. Stop splitting

`_render_text_paragraphs` hands the renderer the whole text part. Fixes fenced code blocks and multi-line
lists.

### 4. Dedent the reasoning trace before rendering

**In scope** (Juha, 2026-08-10), and it is not downstream of the other steps — the deferred item *Reasoning
traces with indented bullets mis-render* is live today, which corrects an assumption an earlier draft of this
brief made.

Reasoning takes a different route to the same wrapper: `DPGCompleteChatMessage.build()` (`:1301`–`:1308`)
calls `add_paragraph(reasoning_content, is_thought=True)` — one paragraph carrying the whole trace, straight
to `_render_text`, never through the splitter. But `reasoning_content` *starts with a newline*, so the
wrapper emits `<font color='...'>` alone on a line with the indented content beneath it, which parses
differently from the heading case where the tag and the content share a line. **So block constructs are
already partly getting through the wrapper**, and "the wrapper flattens everything" is too strong as a model
of what is happening.

The item measured both manifestations of one input — Gemma 4 emitting `    *   Role: ...` with four leading
spaces, confirmed by grep to contain zero backticks and zero font tags:

- **On reload**, four-space indentation fires the indented-code-block rule, so the bullet list renders as a
  grey `Pre` box — whose border does not match its fill, because that is the stranded-`Pre` reflow bug
  arriving by a second route.
- **While streaming**, the code-block rule fires inconsistently, so `*` markers get mis-parsed as emphasis
  delimiters across lines (words tinted pink and teal) and a raw `</font>` leaks at the end.

**Step 1 fixes the leaked `</font>` directly** — there is no wrapper left to be broken by a list parse.
The indentation collision needs its own fix: strip the common leading whitespace from the trace before
handing it over, so indented bullets become real bullets rather than code.

**Dedent by common prefix, never per line.** Removing a uniform prefix preserves relative indentation;
stripping each line independently would flatten any structure that depends on it. This matters because a
reasoning trace *can* contain real code: the item's supporting grep found zero backticks in the stored
traces, which eliminates stray markup as a confound for the samples observed, but is a fact about the
questions asked rather than about reasoning traces in general (Juha, 2026-08-10 — "maybe it's just that I
haven't asked Librarian any coding questions"). Ask a coding question and the trace will carry fenced
blocks whose leading whitespace is content. A per-line dedent would corrupt it, and step 1 is what makes
that corruption visible, since fenced blocks in traces do not currently render at all.

One thing the item notes that is *not* in scope here: the vendored renderer's `Pre`-box position handling.
That is the stranded-box bug, settle-item 4, and it has its own entry.

### 5. Colour the list markers — **done 2026-08-25**

Taken immediately after step 1 rather than in its planned slot, because the prediction below came true the
moment step 1 shipped: green system-prompt text with white bullets down the left, reported from the running
app. The general version was built as argued, and step 1 had already supplied its fallback, so the two cases
really are one mechanism.

Three things worth carrying forward:

- **The colour is resolved on the raw entity spans, before flattening**, and handed to
  `_ConvertedMessageEntity.marker_color`. It cannot be set on the attribute object: `object` is a *property*
  that mints a fresh `List` on every access, one per segment the list spans.
- **`color` has to be set on every bullet case, not just the filled ones.** The hollow circle and hollow
  quad take their colour from the outline rather than the fill, so leaving those alone would have left every
  second nesting depth white.
- **`Blockquote` does not have this defect**, contrary to the note at the end of this step. It carries a
  deliberate `color = [50, 55, 65, 255]` — a dark bar meant to sit *behind* text rather than match it. Left
  alone; do not re-open it.

**In scope** (Juha, 2026-08-10). Bullets and numbers currently keep DPG's default text colour even when the
list renders correctly, which step 1 makes more visible rather than less, since more lists will render.

Two mechanisms in `line_atributes.List`, both missing the same plumbing:

- **Ordered** (`ordered_render`, `:229`): `dpg.add_text(text, pos=..., parent=...)` — no `color` argument, so
  the number falls back to the default. One kwarg.
- **Unordered** (`unordered_render`, `:246`–`:253`): the bullet is drawn into a `drawlist` via
  `dpg.draw_circle` and friends, which take their own colour parameter.

The reason it is not already done is that `List` has no idea what colour is in effect: colour lives in
`font_attributes.Font`, a different attribute applied to the *text span*, and the two do not talk.

**Do the general version; it is smaller than the special case looks.** An earlier draft of this brief said
full generality would need an attribute stack and was not worth it. **That was wrong — the stack is already
there.** `MarkdownText.__init__` (`:259`–`:284`) walks the sorted attribute points and, for each maximal
segment between them, collects every entity active there into `str_attributes` before calling
`str_entity.set_attributes(...)`. A list nested inside a `<font>` span therefore has segments carrying *both*
the `Font` and the `List`. Nothing needs building.

The gap is that `List` never looks at its siblings: colour goes to `font_attributes.Font`, and
`line_atributes.List` renders in a separate pass with no reference to it.

**Resolve from the enclosing span, not from co-occurring segment attributes.** An earlier draft said to take
the colour off the first segment carrying the `List` — wrong, and wrong in a way that shows up immediately: a
list item beginning with a coloured span would tint the marker. **A marker is structural**, so its colour
comes from the context the list *sits in*, never from anything inside an item.

That is simpler to compute than the segment version, because it works on the raw entity list *before*
flattening: entities carry `offset` and `end`, so the governing `Font` is the innermost one whose span
contains the whole `List` range. One containment test, resolved once per list. The document default from
step 1 is the fallback when nothing encloses it — which makes the default and the per-span case one
mechanism rather than two, and gives the right answer for a `<font>` that opens mid-list (it encloses
nothing, so the marker stays default).

Plus a `color` field on `List`, plus the two draw calls. Perhaps fifteen to twenty lines.

**Check `Blockquote` for the same defect while in there**, since it is the other `line_atributes` entry that
draws its own marker and would have the same blindness to `Font`.

### 6. Rename `line_atributes.py` — **done 2026-08-25**

Landed ahead of step 5 rather than riding along with it, at Juha's request. Nine references, not the ten
measured below (`__init__.py` eight, `text_entities.py` one); zero inside the module, as predicted. Renamed
through the edit tools rather than `sed`, the diff being small enough to read.

Typo in the vendored filename, inherited. `git mv` to `line_attributes.py`, then
`sed -i 's/line_atributes/line_attributes/g'` on the two files that reference it. No behaviour change, and
it rides along with step 5, the step that touches this module most.

**Scoped by measurement, 2026-08-10: ten references, and the identifiers are clean.** `__init__.py` has eight
(one import at `:94`, seven `line_atributes.X` uses), `text_entities.py` has one (`:9`). A grep for the
misspelling *inside* `line_atributes.py` returns nothing — it never reached a class or attribute name, so
this is a filename-and-import rename with no API surface. A scan of the rest of the vendored package for
sibling misspellings found none.

**Functional, not only cosmetic**: the misspelling means a grep for `attributes` across the project misses
this module entirely, while `text_attributes.py` and `font_attributes.py` are found. A search that silently
skips one of three sibling modules is the same failure the sweep kept running into — a query that cannot
succeed returns the same empty result as one that legitimately finds nothing.

### 7. A blank line should be a paragraph gap, and is currently deleted

**Added 2026-08-26.** Measured against the parser rather than inferred from the look of the chat:

```python
parser.parse("first para\n\nsecond para\nsame para continued")
→ 'first para\nsecond para\nsame para continued'
```

The blank line is **discarded**, so the two paragraphs merge into one block, and every single newline
becomes a hard line break. The renderer is line-oriented where CommonMark is paragraph-oriented, and it does
the opposite of CommonMark on both counts.

**This belongs to step 3 rather than standing alone, and that is the whole reason to write it down now.**
Today the renderer's behaviour is not what a reader sees: `_render_text_paragraphs` splits on *every* `\n`
and hands the renderer one line at a time, so paragraph separation is DPG's uniform item spacing between
sibling widgets, and a blank line — an empty paragraph, skipped by `_render_text`'s `if text:` — simply
contributes nothing. Every line is equally far from the next, and there is no paragraph-level gap at all.

Step 3 removes that splitter and hands over whole text parts. At that moment the renderer becomes the only
thing deciding, and it deletes blank lines — so paragraph separation would go from *absent* to
*unobtainable*. **Do step 3 and this together, or the chat log gets worse rather than better.**

**The other half is a trap worth naming so nobody "completes" it later.** Making a *single* newline render
as a space would be CommonMark-correct and wrong here — model output uses single newlines to mean something
(step-by-step working, list-ish lines), and reflowing those into prose costs readability rather than buying
it. Keep hard breaks; add paragraph gaps.

Consequence to expect once both land: messages get taller, so anything measuring a message's height —
scroll anchoring, the follow-the-tail sampling — sees different numbers. Nothing depends on the current
values, but it changes every chat log at once rather than one spot.

Until it lands, a tooltip or a card that wants two paragraphs adds them as separate
`dpg_markdown.add_text` calls with a spacer between (the help card's shape, and the thinking-stats tooltip's).
Writing `\n\n` in the source looks right and does nothing at all.

### 8. Tables

The one real renderer gap. Now optional rather than blocking, and it can slip past the exhibit without
holding anything else up.

## Non-goal: the streaming path

**Streaming is not touched, and must not be.**

`_render_text_paragraphs` is called only from `DPGCompleteChatMessage` (`:1344`, `:1505`, `:1507`). Streaming
uses `add_paragraph` / `replace_last_paragraph` (`:502`) and re-renders only the last paragraph per chunk; at
turn end the streaming message is demolished and a fresh `DPGCompleteChatMessage` is built (`:2593`). **A
full re-render at completion already happens today**, so step 3 lands where the message is built once, from
complete text, and introduces no new flicker.

The visible consequence: mid-stream, a fenced block stays literal until the turn completes, then snaps into
place. That is correct rather than a compromise — a half-arrived fence has no closing delimiter and there is
nothing correct to render. Headings, being single-line, will render live once step 1 lands.

## What this brief must settle before implementation

1. **Where the colour parameter lives.** A `MarkdownText.__init__` argument threaded to the initial entity
   state, or a module-level default the caller sets. The first is cleaner; the second is smaller. Check
   whether `_ConvertedMessageEntity` and `text_entities` need to know, or only the font attribute layer.
2. ~~How a reloaded complete message shows its reasoning~~ — **resolved 2026-08-10, and it makes step 3
   safer than drafted.** `DPGCompleteChatMessage.build()` (`:1301`–`:1308`) reads the `reasoning_content`
   sibling field and calls `add_paragraph(reasoning_content, is_thought=True)` — a single call that
   **bypasses `_render_text_paragraphs` entirely**. So the splitter never touches reasoning; step 3 affects
   `content` only, and the collapsible thought block is unchanged by it. This is also step 2's safety
   confirmed twice: the think branch is dead both because the migration removed inline tags and because
   reasoning does not travel that path.
3. **Where the reasoning trace's dedent goes.** See step 5 — the work is in scope; only its placement
   relative to steps 1 and 3 is open.
4. **Whether fenced code blocks land on the stranded-box bug.** `text_attributes.py` still captures absolute
   position via `get_item_pos` at five sites, which is the mechanism the inline-code item names. A `Pre` box
   is tall, so a stranded one is a slab rather than a smudge. The bug is currently dormant — reflow within a
   single message container is what triggers it, and both the resize path and streaming rebuild rather than
   reflow. Worth a short probe before step 3, not a blocker.
5. ~~Whether `is_thought` still needs to be per-paragraph~~ — **resolved by item 2**: reasoning is one
   paragraph from one call, and `content` paragraphs are never thoughts, so the flag stays per-paragraph
   without anything having to change.

## What this closes

Three deferred items merge into one, and the two that survive as sub-tasks are *tables* (a real renderer gap)
and the existing dead-`<think>`-removal item (absorbed as step 2).

Worth noting for the file: the three items were not wrong about the symptom. Each described what it saw
accurately. They were wrong about the cause, identically, for the same reason — and no amount of re-reading
the items would have surfaced it, because the evidence was in code none of them named.
