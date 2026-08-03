# DearPyGui keyboard focus: which predicate, and what `focus_item` really does

Started 2026-08-03 from a Librarian bug — the chat log's scrolling keys were dead at app start and worked
after clicking the log — and ended up rewriting the keyboard handling of both GUI apps. Two findings, neither
documented upstream and neither producing an error: the failure mode is a hotkey that silently does nothing.

## What was measured

**`is_item_focused` is true of a text field nobody has touched.** ImGui gives nav focus to the first
navigable item of a newly focused window on its own, so the field reports focused within a few frames of app
start. A bare-key branch gated on it is therefore dead from startup until something else is clicked.
`is_item_active` is the predicate that means "this field holds the caret", and it is the one a global hotkey
handler wants.

| composer state | `is_item_focused` | `is_item_active` |
|---|---|---|
| startup, no interaction | **True** | False |
| clicked in / typing | True | **True** |
| after Escape (`InputText`'s own cancel) | True | False |
| after clicking another widget | False | False |

`dpg.get_focused_item()` is not a usable cross-check: it kept naming the field on the last row, where
`is_item_focused` on that same field was `False`.

**`dpg.focus_item` cannot focus a child window, and does harm when asked to.** It works on ordinary items —
a button takes focus on the *next* frame. On a child window it does not merely fail: focus lands on the
enclosing window's first navigable item and is *activated*, which for a text field means it takes the caret.
So "park focus on the scrollable panel", the natural way to express *the reader is reading now*, is instead
a reliable way to hand a text field the caret. Both apps contained that call.

To move focus out of a text field, focus a real widget. **A button is safe**: DPG leaves ImGui's
keyboard-nav activation off, so a focused button ignores Space and Enter and cannot fire its callback.

## Files

- `focus_states_probe.py` — the four-state table. Clicks into a field, types, presses Escape, clicks away,
  printing both predicates at each step. Needs `xdotool` and a real X session; takes keyboard focus for
  about ten seconds.
- `button_activation_probe.py` — whether a focused button fires on Space or Enter. The answer is what makes
  a button a usable parking spot. Same requirements, about nine seconds.

Both are self-driving: run them and read the table. Re-run after a DPG upgrade that might have touched focus
handling — the send path in `raven/librarian/app.py` and the search-accept path in `raven/visualizer/app.py`
both rest on the button result.

## Where this ended up

- `dpg-notes.md`, "Keyboard input" — the two findings as reference, plus the investigation-history entry.
- `CLAUDE.md`, DPG Pitfall #7 — the one-paragraph index version.
- `raven/common/gui/tests/test_focus_semantics.py` — the input-free half, asserted continuously. Marked
  `gui`, so it runs under `pytest --run-gui` and is skipped otherwise (it maps a window, which takes focus).
  The rows needing synthetic input are why the two probes above stay as probes.
- Fixes: `raven/librarian/app.py` (gate on `is_item_active`; three `focus_item("chat_panel")` calls removed;
  the send path parks on the send button) and `raven/visualizer/app.py` (same gate; Enter parks on
  `clear_search_button`; the Escape branch deleted, since `InputText` deactivates itself).
