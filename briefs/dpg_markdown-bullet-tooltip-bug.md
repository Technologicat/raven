# `dpg_markdown`: bullets/blockquotes inside tooltips

*2026-04-24, Claude Opus 4.7 (1M context)*

## Summary

`raven/vendor/DearPyGui_Markdown/` rendered unordered-list bullet glyphs (and
blockquote vertical bars) incorrectly when the markdown lived inside an
initially-hidden container — most visibly a `dpg.tooltip`. A list with N items
showed as plain text with **one** bullet glyph stray-drawn at the top-left of
the tooltip content.

## Root cause

In `line_atributes.py`, `List.unordered_render` / `List.ordered_render` and
`Blockquote.self_post_render` compute glyph positions with
`dpg.get_item_pos(spacer_group)`. These methods are scheduled via
`CallInNextFrame`, which only waits for one `dpg.split_frame()` before
running — it does **not** wait for the ancestor window/tooltip to become
visible.

DPG only lays out widgets after their ancestor container is shown. Inside a
hidden tooltip, `get_item_pos` and `get_item_rect_size` both return `[0, 0]`
for every child. All N bullet drawlists therefore get the same
`pos=(0+offset, 0+offset)` and stack on top of each other at the top-left of
`attributes_group`. When the tooltip eventually shows, the text flows in its
normal layout, but the bullets remain pinned at the origin — hence "one stray
bullet, items look like plain text".

Confirmed empirically. With an earlier ad-hoc probe, an inline list produced bullet positions `(34.0, 85.2), (34.0, 111.2), (34.0, 137.2)` (distinct), and a list inside a tooltip produced `(34.0, 0.0), (34.0, 0.0), (34.0, 0.0)` (stacked).

The committed `dpg_markdown_bullet_verify.py` in this directory uses an initially-hidden `dpg.window` as a test proxy rather than a live tooltip, since `dpg.configure_item(tooltip, show=True)` does **not** force a tooltip visible — DPG's tooltip visibility is hover-tied, and a headless script has no mouse hover to work with. A hidden window reproduces the identical root cause ("layout doesn't run until the container is shown") and is controllable from a script.

## Fix

Added `_run_when_laid_out(item, thunk)` in `line_atributes.py`:

- If `item` is already visible and has non-zero rect size, run `thunk`
  immediately (inline case — unchanged behaviour).
- Otherwise, attach a one-shot `item_visible_handler` that fires when `item`
  has been laid out by DPG, then unbinds itself.

The three affected methods wrap their bodies in a local `_draw()` function
and hand it to `_run_when_laid_out`. The deferred execution captures
`attributes_group`, `spacer_group`, and `text_height` via closure, so the
draw step still targets the right DPG parents when it finally runs.

With the fix, the probe reports tooltip bullets at
`(34.0, 43.2), (34.0, 69.2), (34.0, 95.2)` — three distinct rows.

## If we ever talk to upstream

The vendored copy diverges from
[upstream `DearPyGui_Markdown`](https://github.com/IvanNazaruk/DearPyGui-Markdown)
via the Raven-side robustification work; this fix is another such patch.
`dpg_markdown_bullet_verify.py` reproduces the issue against the library's public API and
checks the bullet positions — good starting point for an upstream bug report
or PR. The core observation ("`get_item_pos` returns 0 inside a hidden
container") applies to vanilla upstream too, since upstream's
`line_atributes.py` uses the same pattern.
