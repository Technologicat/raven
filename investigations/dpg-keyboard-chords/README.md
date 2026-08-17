# Which chords survive a text field, and whether modals stack

Measured 2026-08-17, ahead of building `FileDialog`'s keyboard operation
(`briefs/researchers-night/filedialog-keyboard-brief.md`). That design parks focus in the find field and
gives every other key a job, which is only possible if the keys actually arrive — so this asks what a
global `add_key_press_handler` still receives while a single-line `InputText` holds the caret.

The conclusions are folded into `dpg-notes.md` (*Keyboard input*, and *Window sizing* for the modal
result). This directory keeps the apparatus, so the survey can be re-run against a future DPG.

## Apparatus

`probe_chords.py` — one modal window with a single-line `InputText`, a global key-press handler logging
every key with its modifier state and the field's focused/active state, a twice-a-second heartbeat so the
caret state is readable at any instant, and a timer that shows a second modal after 22 s.

Run it, click the field, then drive it. The driving matters more than it looks:

```bash
python probe_chords.py > probe.log 2>&1 &
# ... activate the window, click the field, confirm "BEAT active=1" in the log, then:
xdotool keydown ctrl; sleep 0.25; xdotool key Up; sleep 0.25; xdotool keyup ctrl
```

**Do not append your own markers to the probe's log.** The probe holds it open at its own write offset, so
shell appends and the probe's writes overwrite each other and the file stops being a sequence. Write
markers to a second file and correlate on the timestamps both sides print.

## Results

Every chord in the dialog's table arrives, with modifiers intact, while the field has the caret:
Ctrl+Enter, Alt+Up, Ctrl+Up, Ctrl+Space, Ctrl+Home, Ctrl+Shift+1, and bare Up / Down / Home / End /
Page Up / Page Down. Nothing intercepted Alt+Up under Cinnamon, which is what both dev machines run — so
whether another desktop eats it is for users elsewhere to report, and is the reason Ctrl+Up is offered as
an alias regardless.

**Tab arrives and ImGui does not spend it**: focus does not move, no character is inserted (typed
`readme`, pressed Tab, field still read `readme`). Tab pressed while the field is focused but *inactive*
re-activates it.

**Ctrl+Enter deactivates the field**, committing the edit exactly as bare Enter does on a single-line
field — `active` goes 1 → 0 in the same event.

**Page Up / Page Down are 517 / 518.** Confirmed against the live enum rather than by inference:
Tab=512, Up=515, Down=516, **517**, **518**, Home=519, End=520, while `mvKey_Prior` and `mvKey_Next` still
read 266 and 267.

**A modal does not stack over a modal.** `show_item` on the second one succeeded, raised nothing, and the
window never appeared — `is_item_visible` stayed `False` for the eight seconds observed, with the first
modal still up.

## Two ways this probe lies to you, both found the hard way

**Synthetic chords lose their modifier unless it is held across frames.** `is_key_down` is sampled when
the callback runs, and callbacks are dispatched per frame; `xdotool key ctrl+Up` releases both keys inside
one frame, so the `Up` is dispatched with Ctrl already up. The first run of this probe reported that
Ctrl+Up and Ctrl+Shift+1 carry no modifiers, which would have read as a DPG limitation and is purely an
artifact of the harness. Hold modifiers with `keydown` / `keyup`.

**A permission prompt mid-sequence hands focus back to the terminal**, and the rest of the chords go
there instead. The tell is `active=0` on keys you know you sent into an active field. Confirm the caret
from the log's heartbeat before sending anything, and treat a run without that confirmation as void.

Held modifiers also auto-repeat as key presses (~50 ms apart), each with a companion pseudo-key that no
`mvKey_*` constant names — 663 for Ctrl, 664 for Shift, 665 for Alt.
