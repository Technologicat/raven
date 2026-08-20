# What an `InputText(on_enter=True)` tells you, and when

**Question.** The file dialog's path field commits with `on_enter=True`, and Ctrl+L wanted two things from
it that the flag appeared to rule out: a colour that updates as the path is typed, and an Enter that goes
to the field rather than also to the listing behind it. Both turn on when DPG says what, relative to the
frame the key arrives in.

**Answers, measured 2026-08-20 on `dearpygui` 2.3.1, X11.**

- **An item-edited handler fires per keystroke regardless of `on_enter`.** The field's own `callback` is
  spent on the commit, but `add_item_edited_handler` still reports every edit — one frame after the key
  press, carrying the new value. So a field can be a readout as it is typed into *and* commit on Enter.
- **The global key-press handler runs before the field's `on_enter` callback**, both within the same frame.
  A dialog that binds Enter globally therefore runs both on one press, in that order.
- **By the time the global handler sees Enter, the field is already deactivated** — `is_item_active` False,
  `is_item_focused` True. So `is_item_active` cannot answer "was this Enter the field's?", and neither can
  `is_item_focused`, which is True from startup on a field nobody has touched (ImGui auto-focuses the first
  navigable item). Which control has the caret has to be *tracked*; `fdialog.CaretHome` is how.

Recorded in `dpg-notes.md` under "What still reaches a global handler while a single-line field holds the
caret".

## Scripts

| Script | What it answers |
|---|---|
| `probe_input_text_enter.py` | All three questions in one run. Logs every handler and callback with its frame number and a monotonic timestamp, so the order is read rather than inferred. Needs driving — click the field, type, press Enter, click away — and prints its log when the window closes. |

The window is a primary window on purpose: no title bar, pinned to the viewport origin, so a driven click
lands where the arithmetic says. The first run of this probe had an ordinary window and aimed 20 px below
its top-left, which is the title bar — the drag that started went unnoticed, and the probe reported that
typing had gone nowhere.
