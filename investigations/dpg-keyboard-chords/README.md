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

## Writing the field while the user is typing in it

`probe_setvalue.py`, added the same day, after the question came up: both Tab completion and the save-mode
arrow-fill write the find field *while it holds the caret*, and ImGui keeps its own edit buffer for an
active `InputText`. F2 writes the field, F3/F4/F5 try progressively more patient versions of the
unfocus → write → refocus dance, and the edit callback is logged so the revert is visible.

**On an inactive field, `set_value` works and fires nothing** — the baseline, and what the 2026-08-13
measurement recorded. Typing afterwards continues from the written value.

**On an active field the write is undone.** `get_value` immediately after `set_value` reports the new
string, and the *next frame* writes the old buffer back and fires the edit callback while doing so:

```
F2 set_value: before='abc' after='SETVALUE' active=1
CALLBACK app_data='abc' get_value='abc'      <- 17 ms later, ImGui reverting
```

Typing `Z` then yields `abcZ`. So both recorded rules invert on an active field: the write does not take,
and a callback fires anyway.

**The caret is not released on the calling frame.** Polling `is_item_active` once per frame after
`focus_item` on a button gives `[1, 0, 0, 0, 0, 0]` — but that is one sample on an idle app, and it is not
a bound. `focus_item` queues a change ImGui applies on its next NewFrame, so the number of *rendered*
frames it costs moves with what else is queued and where the vsyncs land. Code that waits a fixed number
of frames is a race that passes here and fails on a loaded app; poll `is_item_active` instead, bounded.

The easy walk-in: `focus_item` not taking effect until the next frame is already documented, so spending
one `split_frame` on it feels like the job is done.

**But refocusing arms select-all**, and that is the part with no answer. After the full dance the field
genuinely held `DANCED3`; the next typed character left it holding `Y`. For a completion this is the wrong
behaviour outright, and DPG exposes no caret or selection API to correct it.

**`configure_item(default_value=...)` is not a way around it** (F6). It fails harder than `set_value`:
`before='abc' after='abc'` on the very next line, where `set_value` at least reported the new string before
the revert. So there is no spelling of the write that survives an active field — a feature needing one has
to be redesigned rather than re-spelled. This is what closed Tab completion in the FileDialog brief.

## Can a `menu_item` hold focus?

No, and it is the third distinct case in DPG's focus model (F7). `get_item_state` on a `menu_item` returns
a dict with no `"focused"` key at all, so `dpg.is_item_focused` raises `KeyError: 'focused'` rather than
answering False — which is why the first run of this probe lost the whole log line to an exception, and why
the query is wrapped now. `focus_item` on one is a no-op: focus was on the text field before the call and
still on it afterwards, still active.

Harmless, then, unlike a child window, which takes the caret away when asked to do the impossible. But it
means anything built from menu items cannot use the focus-dispatch idiom.

## Is the dialog resizable if you ask it to be?

`probe_fd_resize.py` builds a real `FileDialog` with `no_resize=False` — which the parameter allows and no
Raven caller has ever passed — and the grip gets dragged both ways. Findings and the resulting work item are
in `TODO.md` under the fdialog work package; the short version is that enlarging works (the thumbnail grid
reflows for free, since `_resize_grid` measures instead of computing) and shrinking below the construction
width pushes the OK and Cancel buttons off the edge.

**Run it with `PROBE_FONT=1`.** Without it the dialog renders in DPG's built-in ProggyClean, which is
ASCII-only, and the grid's `…` truncation marker becomes a missing-glyph box. That looks exactly like a
Raven bug and is not one — it cost a detour here before `fontTools` confirmed OpenSans carries U+2026 and a
re-run with the font showed the ellipsis rendering correctly. `PROBE_PATH` and `PROBE_THUMBS=1` open a
picture directory in grid mode, which is the case worth watching during a resize.

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
