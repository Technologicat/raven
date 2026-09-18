# The chat graph's font ladder, against DPG's font atlas

Measured 2026-09-18, when the chat graph view grew a bold face so a search match inside a box label could be
painted red *and* bold, agreeing with how the chat log draws one. That took the graph from one font ladder
to four — `(regular, bold, italic, bold-italic)` at 4, 8, 16, 32 and 64 px — and the question was whether
twenty rungs are something an atlas minds.

## What is here

| Script | What it answers |
|---|---|
| `measure_font_ladder.py` | Do all twenty rungs exist and behave: does each measure, do widths grow with the atlas size, is bold wider than regular |
| `measure_size_ceiling.py` | How far up font sizes still work now that DPG picks the character ranges itself, and what loading one costs |

Both map a window. `dpg.get_text_size` has no answer before a frame is rendered, and rendering one needs a
viewport that is shown, so neither can run headless.

## What was found

**All twenty rungs are there.** Every one measures, widths scale with the atlas size to within a fraction of
a percent, bold comes out about 7.5% wider than regular at every size and italic slightly narrower. Nothing
fell back to another face.

**Loading a face costs nothing, at any size.** 0.1–0.2 ms each, the same at 1024 px as at 64. A 1024 px face
then measures Latin, Greek and Cyrillic correctly, at widths linear in the size across a sixteen-fold range
(5.943 to 5.953 px per size unit).

**So rasterization is demand-driven rather than eager over the font's range.** Rasterizing OpenSans' ~1150
glyphs at 1024 px would be 1.2 billion pixels, which no texture holds; it loads in a tenth of a millisecond
and works. That also disposes of an estimate made before these ran, which bounded the four-face ladder at
~25 M pixels by charging every glyph in the TTF a full em square at every rung: that is a bound on something
that does not happen. What an atlas actually holds follows what gets *drawn*, and graph labels draw a small
ASCII subset.

Worth keeping in view because the ceiling is expensive rather than distant: a 16384-px square atlas is
268 M pixels, which is around a gigabyte at four bytes each and a quarter of that if the upload is
alpha-only. Which of those DPG uses was not established here. On the one-card configuration that is VRAM the
LLM would otherwise have.

## What this does **not** show

**That glyphs rasterize.** Both scripts measure, and `get_text_size` can answer from the font's own metrics
without a glyph ever reaching the atlas — the clearest sign being that Greek and Cyrillic measure correctly
at 1024 px in a process that has never drawn either. A missing glyph is silent and looks like this. Only
looking at rendered text answers it, so that part belongs to the first live look at the graph rather than to
a probe.

## The bug this is *not* about

Raven has two font problems and they are easy to conflate — the first version of the write-up in
`briefs/researchers-night/done/16_chat-graph-view-brief.md` conflated them twice over:

- **Atlas overflow at several hundred pixels**, which needs the extended Unicode ranges and was answered by
  loading only the codepoints actually needed. DPG 2.3 made the ranges automatic and turned
  `add_font_range` into a deprecated no-op, so nothing configures them by hand any more; see
  `raven/common/gui/fontsetup.py` and `raven/common/gui/tests/test_fontsetup.py`.
- **The intermittent drop at ordinary sizes** — `TODO_DEFERRED.md`, *"The Markdown renderer drops text"* —
  which is unexplained, settled per launch, and has the atlas as its standing suspect. Nothing here
  reproduces it. What this work does is add twenty more `(face, size)` rungs to a process that already has
  an unexplained fault around building them, which is a reason to watch for a specimen rather than to expect
  one.
