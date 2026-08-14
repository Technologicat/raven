# Which symbols Raven's UI fonts actually have

Measured 2026-08-14, against the fonts in `raven/fonts/`.

**The question this answers:** *can I put this character in a GUI label?* A glyph the font lacks does not
fall back to another font in DPG — it renders as a missing-glyph box, and only in the running app, so it is
invisible in code review and in every test that does not map a window.

**The headline: OpenSans has no dingbats.** No arrows in any direction, no triangles, no check mark, no
cross, no star, no warning sign. It covers Latin-1, Greek, mathematical operators and typographic
punctuation well, and stops there — 1010 glyphs. InterTight has all of the above, at 2504.

This matters because **OpenSans is the default** (`raven.common.gui.utils.bootup`, `font_basename="OpenSans"`),
so it is what a GUI label is rendered in unless someone chose otherwise.

## Coverage

`Y` = present, `.` = absent.

| Codepoint | Glyph | OpenSans | InterTight | Name |
|---|---|---|---|---|
| U+2190 | ← | . | Y | LEFTWARDS ARROW |
| U+2191 | ↑ | . | Y | UPWARDS ARROW |
| U+2192 | → | . | Y | RIGHTWARDS ARROW |
| U+2193 | ↓ | . | Y | DOWNWARDS ARROW |
| U+2194 | ↔ | . | Y | LEFT RIGHT ARROW |
| U+2195 | ↕ | . | Y | UP DOWN ARROW |
| U+21B5 | ↵ | . | Y | DOWNWARDS ARROW WITH CORNER LEFTWARDS |
| U+21D2 | ⇒ | . | Y | RIGHTWARDS DOUBLE ARROW |
| U+27A1 | ➡ | . | . | BLACK RIGHTWARDS ARROW |
| U+00AB | « | Y | Y | LEFT-POINTING DOUBLE ANGLE QUOTATION MARK |
| U+00BB | » | Y | Y | RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK |
| U+2039 | ‹ | Y | Y | SINGLE LEFT-POINTING ANGLE QUOTATION MARK |
| U+203A | › | Y | Y | SINGLE RIGHT-POINTING ANGLE QUOTATION MARK |
| U+25B2 | ▲ | . | Y | BLACK UP-POINTING TRIANGLE |
| U+25BC | ▼ | . | Y | BLACK DOWN-POINTING TRIANGLE |
| U+25C0 | ◀ | . | Y | BLACK LEFT-POINTING TRIANGLE |
| U+25B6 | ▶ | . | Y | BLACK RIGHT-POINTING TRIANGLE |
| U+25B8 | ▸ | . | . | BLACK RIGHT-POINTING SMALL TRIANGLE |
| U+2023 | ‣ | . | Y | TRIANGULAR BULLET |
| U+2022 | • | Y | Y | BULLET |
| U+00B7 | · | Y | Y | MIDDLE DOT |
| U+2713 | ✓ | . | Y | CHECK MARK |
| U+2717 | ✗ | . | Y | BALLOT X |
| U+2716 | ✖ | . | . | HEAVY MULTIPLICATION X |
| U+2605 | ★ | . | Y | BLACK STAR |
| U+2606 | ☆ | . | Y | WHITE STAR |
| U+2665 | ♥ | . | Y | BLACK HEART SUIT |
| U+26A0 | ⚠ | . | Y | WARNING SIGN |
| U+2502 | │ | . | . | BOX DRAWINGS LIGHT VERTICAL |
| U+2500 | ─ | . | . | BOX DRAWINGS LIGHT HORIZONTAL |
| U+250C | ┌ | . | . | BOX DRAWINGS LIGHT DOWN AND RIGHT |
| U+2588 | █ | . | . | FULL BLOCK |
| U+007C | \| | Y | Y | VERTICAL LINE |
| U+00A6 | ¦ | Y | Y | BROKEN BAR |

Both fonts have all of these, so they need no lookup: `× ÷ − ± ≠ ≤ ≥ ∞ ≈ ∑ ∏ √ ∂`, `– — … ‘ ’ “ ” † ‡ § ¶ ′ ″`,
`© ® ™ ° ‰`, and Greek (`α μ Ω π`).

## What to reach for instead

- **A directional mark that works in OpenSans:** `»` or `›`. This is what
  `raven.common.filelisting.format_kind` uses, writing `Link»Dir` where a file manager would write "Link to
  directory".
- **An actual arrow, or a check, or a warning:** these live in the **FontAwesome icon font**, which Raven
  already loads (`themes_and_fonts.icon_font_solid`) and which does have U+2192. The catch is that binding
  a font applies to the whole widget, so an icon-font label cannot mix icons with prose — it works for a
  standalone glyph, not for `"Link→Dir"`.
- **Switching the app to InterTight is not available**, for the reason in the next section. Do not reach
  for it to get a symbol.

## Why OpenSans is the default, measured

InterTight draws **subscripts at superscript height**. Its `two.subs` and `two.sups` glyphs have identical
outlines in identical positions, so H₂O renders as H²O and x₁ as x¹ — chemistry and mathematics read
wrongly, which is disqualifying for a research tool.

| | subscript ₂ (U+2082) | superscript ² (U+00B2) | |
|---|---|---|---|
| OpenSans | y −266..629, below the baseline | y 852..1747 | distinct |
| InterTight | y 744..1650 | y 744..1650 | **identical** |

(Units are font units, `unitsPerEm` 2048 in both. Same result for `3`.)

**Codepoint coverage runs the other way**, which is the trap: InterTight has *more* of the subscript block
(28/29 of U+2080–209C, and 16/16 superscripts) than OpenSans (22/29 and 15/16). OpenSans is missing
`ₓ ₐ ₑ ₒ ₔ ₋` — including subscript x, which chemical formulae want. So a coverage check alone recommends
InterTight, and rendering disqualifies it.

Note this **corrects `raven.common.gui.utils.bootup`'s docstring**, which had the coverage claim backwards
— it says InterTight is the one missing subscript x. Its conclusion was right; the stated reason was not.

The general lesson, and the reason this section exists: **a font can have a glyph and still render it
wrongly.** `cmap` presence answers "will it show a box", not "will it show the right thing".

## Re-running this

Extend `GROUPS` and run from the repo root. Needs `fontTools`, which arrives with the font stack.

```python
from fontTools.ttLib import TTFont
import unicodedata

FONTS = {"OpenSans": "raven/fonts/OpenSans-Regular.ttf",
         "InterTight": "raven/fonts/InterTight-Regular.ttf"}
cmaps = {name: TTFont(path).getBestCmap() for name, path in FONTS.items()}

GROUPS = {
    "Arrows": [0x2190, 0x2191, 0x2192, 0x2193, 0x2194, 0x2195, 0x21B5, 0x21D2, 0x27A1],
    "Angle marks": [0x00AB, 0x00BB, 0x2039, 0x203A],
    "Triangles/bullets": [0x25B2, 0x25BC, 0x25C0, 0x25B6, 0x25B8, 0x2022, 0x00B7, 0x2023],
    "Marks": [0x2713, 0x2717, 0x2716, 0x2605, 0x2606, 0x2665, 0x26A0, 0x00B0],
    "Boxes/bars": [0x007C, 0x00A6, 0x2502, 0x2500, 0x250C, 0x2588],
}

for group, codepoints in GROUPS.items():
    print(f"### {group}")
    for cp in codepoints:
        ch = chr(cp)
        marks = " ".join(("Y" if cp in cmaps[name] else ".") for name in FONTS)
        print(f"  U+{cp:04X} {ch}  {marks}   {unicodedata.name(ch, '?')}")
    print()
print("columns:", list(FONTS), "| glyph counts:", {n: len(c) for n, c in cmaps.items()})
```

To ask about one character rather than a table: `0x2192 in TTFont(path).getBestCmap()`.

The same check answers it for the icon fonts (`fa-solid-900.ttf`, `fa-regular-400.ttf`) — `fa-solid-900`
does carry U+2192, which is where the "use the icon font" suggestion above comes from.
