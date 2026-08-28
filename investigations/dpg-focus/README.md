# DearPyGui keyboard focus: which predicate, and what `focus_item` really does

Started 2026-08-03 from a Librarian bug — the chat log's scrolling keys were dead at app start and worked
after clicking the log — and ended up rewriting the keyboard handling of both GUI apps. Four findings, none
documented upstream and none producing an error: the failure mode throughout is a hotkey that silently does
nothing, which is why every one of them cost a live test to notice.

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

**A child window is unaskable, not unfocusable.** `focus_item` has no working spelling for it, but a click
does focus one — grabbing its scrollbar included — and the focus persists after release.

**Enter is the exception to all of the above, and the right predicate depends on the field's kind.** A
single-line `InputText` *deactivates itself* on Enter, because the key commits the edit; a multiline one
does not, because there Enter inserts a newline.

| after pressing Enter | `is_item_focused` | `is_item_active` |
|---|---|---|
| single-line | True | **False** |
| multiline | True | **True** |

So an app with a single-line field gates its Enter handler on `is_item_focused` while still gating its
bare-key branch on `is_item_active` — two different questions about one widget, each chosen for the state
the key arrives in. Raven's two GUI apps differ from each other for exactly this reason. Learned by
regression: switching the Visualizer's Enter gate to `is_item_active` silently killed its search, and it
took a live test to notice, because a dead hotkey raises nothing.

## A committing field and a global key handler both see the chord, and cannot be told apart

Measured 2026-08-28, DPG 2.3.1, for a second send path in `raven-librarian`: the composer sends on its own
commit callback, so the chord is dead whenever the composer is not focused — which is a real state, since a
send parks focus on the send button, and an empty message is Librarian's "take another turn" gesture.

Adding the chord to the global hotkey handler works only if the two paths can coexist. They do fire
together:

| phase | global handler sees | field's `on_enter` |
|---|---|---|
| auto-focused at startup, never touched | focused **True**, active False | — |
| caret in it, typed, then the chord | focused **True**, active False | **commits** |
| focus parked on a button | focused False, active False | — |

**The first two rows are the finding.** They are the same state and want opposite answers: the untouched
field does not commit, so the global path must send; the typed field does, so the global path must not. No
predicate separates them, and `is_item_focused` — the right gate for a *commit* handler, per the Enter
result above — would leave the chord dead from app start until something else is clicked. Which is the
original bug wearing a different hat.

**The global handler runs first, and in the same frame as the commit** — 232/232 and 602/602 across two
runs, the global row always ahead of the field row. So it cannot consult anything the commit sets either; by
the time the field could say "I sent that", the global path has already decided.

**But same-frame is what was observed, not what is guaranteed**, and the difference decides the fix. Both
callbacks run on DPG's single callback thread while the *main* thread goes on incrementing the frame
counter, so nothing stops a frame boundary from falling between the two reads. Two agreeing samples say the
gap is small; they say nothing about the tail. What bounds it is that the two run back to back on that one
thread, microseconds apart against a frame of roughly sixteen milliseconds — so at most **one** boundary can
land between them.

What that leaves is not a better gate but no gate: **let both paths request a send, and drop a request that
arrives within one frame of the last.** Every row above then yields exactly one send, and the question "did
the field handle this one?" is never asked. The window is one frame rather than zero for the reason above;
it cannot swallow anything real, since two legitimate sends that close are not reachable — a human cannot
produce them, and `_describe_send_gate` refuses while a turn is in flight.

Also confirms the same holds for the other setting: `send_message_key = "enter"` puts the commit on bare
Enter (`ctrl_enter_for_new_line=True`, which names what *Ctrl+Enter* does and so reads backwards from the
setting), and the dispatch pattern is identical.

## Holding the scrollbar does not hold your place

Found separately, testing whether a reader can hold a position by dragging the scrollbar while a reply
streams. They cannot, for long: it works over a quick drag and creeps over a long hold.

ImGui derives the scroll position from where the thumb sits in its track, which is a **fraction** of the
content. So holding the thumb still holds the *fraction*, and as `scroll_max` grows with each generated line
the absolute position slides down by (fraction × new content).

Measured from a Librarian session log, 288 consecutive samples over 2.7 s while the reader held the
scrollbar (2026-08-03):

| | y_scroll | max_y_scroll | y/max |
|---|---|---|---|
| start | 2375 | 3555 | 0.6681 |
| end | 2640 | 3971 | 0.6648 |

The ratio is flat to 0.5 % while both endpoints grow by hundreds of pixels — the fraction is what is being
preserved. Per step, `max += 52` and `y += 33`, and 33/52 = 0.63 ≈ the held fraction.

**Raven is not a party to this.** Over those 288 samples the app issued no scroll command of any kind — no
`scroll_view`, no `_set_y_scroll`, no animation frames — and `should_follow_tail` correctly returned `False`
throughout. The movement is entirely ImGui's.

A fix is available but not taken: `scrollbar_drag_probe.py` establishes that the drag is detectable, so a
per-frame compensator could hold the absolute offset across a change in `max_y_scroll` instead of letting
the fraction stand. The cost is that the thumb then drifts away from the cursor, which wants looking at
rather than reasoning about. Tracked in `TODO_DEFERRED.md`, alongside the `SmoothScrolling` refactor,
because both land in `raven/common/gui/animation.py` and are better done together.

## Files

- `focus_states_probe.py` — the four-state table. Clicks into a field, types, presses Escape, clicks away,
  printing both predicates at each step. Needs `xdotool` and a real X session; takes keyboard focus for
  about ten seconds.
- `button_activation_probe.py` — whether a focused button fires on Space or Enter. The answer is what makes
  a button a usable parking spot. Same requirements, about nine seconds.
- `enter_key_probe.py` — whether Enter deactivates a text field, single-line versus multiline. The answer
  decides which predicate an Enter handler may be gated on, and getting it wrong killed the Visualizer's
  search. About twelve seconds.
- `scrollbar_drag_probe.py` — whether a scrollbar drag is detectable at all, which decides whether the creep
  above can be compensated. Also where the click-focuses-a-child-window result comes from. About eight
  seconds. It deliberately does **not** reproduce the creep: its press lands on the scrollbar track rather
  than the thumb, so ImGui answers with auto-repeat paging instead of a drag, and the creep is better
  established from the session log above than from any synthetic drag.

- `focus_across_child_window_probe.py` — when `focus_item` refuses to move focus at all. Enumerates every
  source/target position across two sibling child windows and the enclosing window. The answer is that
  **only window-level to child is refused**; child to child, including between siblings, works. Added
  2026-08-18, when `FileDialog`'s Ctrl+F stopped returning the caret. Under a second.
  - **Measured on DearPyGui 2.3.1**, the version pinned at the time (`dearpygui>=2.3`). Recorded because
    this one is a bug rather than a design choice as far as anyone can tell, so it is the finding here
    most likely to change under upgrade — and a result with no version against it cannot be compared to a
    later run.

- `auto_focus_target_probe.py` — which item DPG auto-focuses in a child window, and whether it *activates*
  it. Kept for what it rules out rather than what it establishes: image buttons are **not** skipped in that
  order, and auto-focus focuses without activating. Both were the obvious explanation for the flash below,
  and both are wrong. Added 2026-08-18, DPG 2.3.1. Under a second.

- `commit_chord_dispatch_probe.py` — whether a global key handler also sees the keypress a text field
  commits on, in what order, and whether the states involved can be told apart. The answer decides whether a
  second send path can exist beside the field's own commit, and it is *no gate is possible* — see the
  section above. Two multiline fields, one per `send_message_key` setting. Added 2026-08-28, DPG 2.3.1.
  About fifteen seconds.

- `catch_visual_flash.sh` — not a probe but the instrument that worked when three probes did not. Records a
  running app's window with `ffmpeg -f x11grab` at 60 fps and ranks the frames by how much the region of
  interest stands out, so a 25–100 ms artifact can be found and looked at. Screenshots cannot: `import`
  costs 50–200 ms a frame and samples roughly one in five.
  - **The open question it was built for is still open.** `FileDialog`'s path field goes *active* for a few
    frames on every Tab — active meaning the caret, hence select-all, hence a blue flash in a field nobody
    touched. That much is measured, by this script and by logging focus state per frame from the grid's
    tick thread. What puts it there is not known, and three plausible mechanisms have been falsified.

The python probes are all self-driving: run one and read the table. Re-run after a DPG upgrade that might have touched focus
handling — the send path in `raven/librarian/app.py` and the search-accept path in `raven/visualizer/app.py`
both rest on the button result, and `FileDialog`'s choice of parking spot rests on the position result. If a
later DPG lifts the window-to-child restriction, that probe is where it will show, and the parking spot can go
back to being chosen for readability rather than for reachability.

## Where this ended up

- `dpg-notes.md`, "Keyboard input" — the findings as reference, plus the investigation-history entry.
- `CLAUDE.md`, DPG Pitfall #7 — the index version, including the Enter caveat, since the bare rule without
  it is what produced the Visualizer regression.
- `raven/common/gui/tests/test_focus_semantics.py` — the input-free half, asserted continuously. Marked
  `gui`, so it runs under `pytest --run-gui` and is skipped otherwise (it maps a window, which takes focus).
  Everything needing a click, a keystroke or a drag is why the probes above stay probes.
- Fixes: `raven/librarian/app.py` (bare keys and Enter both on `is_item_active`, its composer being
  multiline; three `focus_item("chat_panel")` calls removed; the send path parks on the send button) and
  `raven/visualizer/app.py` (bare keys on `is_item_active`, but Enter on `is_item_focused`, its search field
  being single-line; Enter parks on `clear_search_button`; the Escape branch deleted, since `InputText`
  deactivates itself).

**The two apps wore the same bug differently, which is worth knowing before hunting for it elsewhere.** In
Librarian it showed at startup: the composer auto-focuses, the chat log has content immediately, so the
scrolling keys were dead from the first frame. The Visualizer cannot fail that way — its info panel is empty
until something is selected, so there is nothing to scroll before the user has clicked, and by then focus has
moved off the search field of its own accord. Its symptom was the *other* half: after `Ctrl+F` and Enter,
`focus_item("item_information_panel")` bounced focus back into the search field and activated it, so the
navigation keys went dead after every search. Same two defects, opposite halves visible.

Consequently the Visualizer's startup case is not testable by hand — an empty panel scrolls the same whether
the keys reach it or not — and the fix is validated instead by `Ctrl+F` → type → Enter → arrows, and
`Ctrl+F` → Esc → arrows, both confirmed working 2026-08-03.
