# The chat view stops following the tail mid-reply

**This bundle now holds three episodes of the same machinery**, newest last in the file but not in date
order: the 2026-08-11 drift below, the 2026-08-12 settle-wait fault after it, and the 2026-07-30 work that
made the view follow at all, which is the background to both. They share `should_follow_tail`,
`scroll_view` and `_commanded_y_scroll`, and each one's fix is visible in the next one's failure.

Observed in `raven-librarian` on 2026-08-11: while a reply was streaming, the view fell behind the text and
only jumped to the end once the message was finalized. Seen a handful of times before, intermittently, and
never reproduced on demand.

It did not need reproducing. `DPGLinearizedChatView.should_follow_tail` already reports a refusal near the
end as a NEAR MISS at INFO, carrying the numbers that say which test failed, so an ordinary run captured it:

```
NEAR MISS - settled_gap=52.0px exceeds tolerance=40px and the position has drifted 46.0px from the 4894
we last commanded (to_end=True, drift tolerance=40px including 0.0px of animation slack), so the view will
not follow.
```

Raw excerpts: `near-miss-2026-08-11.txt` (the run that recovered), `latched-2026-08-11.txt` (the one that did
not), and `fixed-2026-08-11.txt` (the trigger firing after the fix, with nothing following it).

**Fixed 2026-08-11**, confirmed in the log and in the view — the chat panel followed the reply as expected
(Juha). See the last section.

## What the numbers say

The scroll commands logged just before it, one per streamed update:

| time | max_y_scroll | commanded |
|---|---|---|
| 41.224 | 4718 | 4718 |
| 41.595 | 4744 | 4744 |
| 41.721 | 4796 | 4796 |
| 41.978 | 4848 | 4848 |
| 42.270 | 4900 | 4900 |

So the content grows about **52 px per update** — one wrapped line of markdown — and each update commands a
scroll to the new end.

Three things then hold at the moment of refusal, and only the third is surprising:

- The position sits at **4848**, which is the *previous* maximum. `settled_gap` is 4900 − 4848 = 52 px.
- `_commanded_y_scroll` reads **4894**, which is not a value any logged command used. Scrolling is animated
  (`gui_animation.SmoothScrolling`), and the animation writes a new commanded value every frame in the same
  breath as each `dpg.set_y_scroll` - so 4894 is a frame of an animation in flight between 4848 and 4900.
- **The drift tolerance included 0.0 px of animation slack.**

The drift the check objects to - 46 px, against a 40 px tolerance - is therefore the gap between where a
running animation had got to and where the panel actually was. That is what animation slack exists to
absorb, and it was zero.

## Where the zero comes from

```python
animation_slack = scroll_animation.last_step if scroll_animation is not None else 0.0
tolerance = max(_PIN_TOLERANCE_PX, animation_slack)
```

The slack is the animation's most recent step, so it is large early in an exponential decay and shrinks to
nothing as the animation converges - and it is `0.0` outright once the animation is retired. Which means
**the slack is smallest exactly when the animation is finishing**, and a residual lag of one frame is still
outstanding at that moment.

That is enough to explain the numbers without appealing to anything unobserved:

1. A streamed update grows the content and commands a scroll to the new end. The animation runs.
2. The animation's last written value is 4894. The panel's actual position is clamped to **4848**, the
   maximum that was in force when the write landed.
3. Content grows again, to 4900. Nothing re-writes the position, because the animation has converged.
4. The check runs. `expected = min(commanded, max_y_scroll) = 4894`, actual is 4848, drift 46 px - and the
   slack that would have covered it is now 0.0, because the animation it belonged to is done.

So the drift is the residue of *our own* scroll, scored against the reader. The position sitting exactly on
the previous maximum is the tell, and it is what a fix can key on: with `_commanded_scroll_was_to_end` set,
a position equal to an earlier maximum means the content grew under us, which is not somebody scrolling.

Still inference, and worth checking before building on it: that `scroll_animation` was `None` (retired)
rather than present-with-a-zero-step. The log line does not distinguish them, and it easily could.

**A tolerance bump is not the fix.** 40 px (`_PIN_TOLERANCE_PX`, two lines of text) is already less than
one growth step, so raising it past 52 px would hide this instance and still fail on a longer wrapped line -
and the bound is squeezed from the other side too, since too large means a deliberate one-line scroll away
from the end still counts as following, and the arrow keys look broken. The comment above the constant says
this. The thing to fix is the expectation, not the bound.

This is also not the first pass over this ground: the comment beside the slack records an earlier
measurement of 43 samples in 857 reading as user scrolls at 51-78 px drift, which is what the slack term
was introduced to absorb. What is left is the tail of the same problem, where the absorber has already
faded out.

## One bad sample costs the whole message

A second occurrence in the same session showed what a single false refusal actually costs: **it persists for
the rest of the reply.** Two runs are distinguishable by the value they froze at - `4894` (which recovered
when the message finalized) and `4863` (which never did).

Across the four seconds of the second run, `settled_gap` climbs monotonically - 52, 78, 104, … 780 px, in
steps of 26, one wrapped line per streamed update - while **the drift stays fixed at 47 px**. That pairing is
the whole story:

- Once `follow` is false, nothing commands a scroll. There are **zero scroll commands** in the log between
  the first refusal and the last, four seconds and several hundred pixels later.
- So `_commanded_y_scroll` stays 4863 and the position stays 4816, and `drift` is 47 forever. `undisturbed`
  can never become true again.
- And `settled_gap` only grows, so `at_end` can never become true either.

Both terms of `follow = at_end or (commanded_was_to_end and undisturbed)` are therefore pinned false for the
rest of the message. The only exit is the reader scrolling to the end by hand, which is exactly the recovery
that was observed the first time and absent the second.

That reframes the severity rather than the diagnosis. A false refusal is not a lost frame: it costs every
remaining line of the reply, and whether the message end rescues it is a matter of which sample the finalize
path happens to take. The persistence itself is correct behaviour - see below - so the whole cost lands on
getting the entry right.

The moment of capture, with the numbers that name every term:

| | |
|---|---|
| content grows | 4816 → 4868 (two lines at once) |
| commanded | 4868 by `scroll_view`; the animation had written **4863** when the check ran |
| actual position | **4816**, the previous maximum - the write had not landed |
| drift | \|4816 − 4863\| = **47** against a 40 px tolerance |
| animation slack | **0.0**, the animation having all but converged (5 px short of its target) |

## The latch is the feature; only the entry is the bug

Considered and rejected (Juha, 2026-08-11): making the refusal self-recovering. Not following **must**
persist - a reader who scrolls up to re-read something mid-reply has to stay where they put themselves, and
that is the entire intention this machinery serves. The exit is the same gesture in both cases, scrolling
back to the end, and it works. So there is no second defect here. There is one bug, entered spuriously.

That matters for where to look: the question is never "how do we get out of this state" but "how did we
conclude the reader had scrolled when they had not".

## No scroll event — but the *input* is hookable, which is the way out

`should_follow_tail` explains why it infers intent from position rather than from events: "of the three ways
this panel moves - scrollbar drag, mouse wheel, navigation keys - the drag is handled inside ImGui and
raises nothing we could hook". True of the **scroll**, and that is what the sentence is about.
`_user_scroll_generation` is authoritative only for the third way, the one going through `_set_y_scroll`, so
consulting it alone would miss the other two.

**But the gesture that caused the scroll is fully visible** (Juha, 2026-08-11), and that is a different
question from seeing the scroll. DPG exposes `add_mouse_down_handler`, `add_mouse_drag_handler` (with a
button filter and a movement threshold) and `add_mouse_release_handler` alongside the wheel and move
handlers already registered in `app.py`, plus `is_mouse_button_down` and `is_mouse_button_dragging` as
queries. `guiutils.is_mouse_inside_widget` narrows any of them to the chat panel, whose position and size
are known.

So both blind paths can be caught positively, at the input rather than at the scroll:

- **Wheel:** a wheel event with the pointer over the chat panel is a reader scrolling it - *unless a modal
  is up*, which takes the wheel while the pointer still reads as being over the panel underneath.
  `app.is_any_modal_window_visible` already answers that (help card, attach dialog, cleanup dialog,
  messagebox), so the test is the conjunction, and it is then exact.
- **Scrollbar drag:** a left button held, dragging, with the *press* inside the panel. The press is what has
  to be tested, not the pointer at each moment: once a drag has begun, the pointer routinely leaves the
  scrollbar strip and the panel entirely and the drag continues - it would be unusable otherwise. So this is
  a small state machine, opened by a button-down inside the strip and closed by the release, and a
  position test applied per-event would drop exactly the part of the drag where the reader is furthest from
  where they started.

That turns "did the reader scroll?" from an inference about position into an observation about input, which
is what the whole difficulty here has been.

**One idea that does not survive**, recorded because it suggests itself: *bias the ambiguous case toward the
reader.* There is no such direction. Deciding "theirs" stops following, which abandons a reader watching the
stream; deciding "ours" keeps following, which drags back a reader who scrolled away. Both are the reader,
so the test has to be accurate rather than conservative.

## What is actually broken, in the docstring's own words

The same docstring names this failure exactly:

> The comparison is only as good as the record it compares against, which is why `scroll_view` waits for its
> command to actually land rather than assuming it did. A command still in flight leaves the position
> disagreeing with the record, which is indistinguishable here from the user having scrolled.

That is the bug, and the mitigation does not reach the case: `scroll_view` waits for its own call to land,
but scrolling is *animated*, so `SmoothScrolling` goes on writing a new commanded value every frame after
`scroll_view` has returned. The record therefore runs ahead of the position by design, for as many frames as
the decay lasts, and the animation slack that was meant to cover that has already faded to zero by the end
of it (see above).

So the record and the position are being compared at moments when they are *guaranteed* to disagree. A fix
wants to compare the position against what the animation has actually had time to apply - its trajectory -
rather than against its most recent write. The position sitting exactly on an earlier maximum, as 4816 was
here, is the observable form of that: it is a value we wrote, not one a reader would land on.

## Attempted 2026-08-11, and what it turned up

**A dead end, recorded so nobody walks it twice.** The obvious fix is to record what DPG *will* adopt rather
than what was asked for - clamp the value written into `commanded_y_scroll` by the scrollable range, inside
`SmoothScrolling._set_y_scroll`, where the box and `dpg.set_y_scroll` are written together. It fails on the
query: `dpg.get_y_scroll_max` returns `0.0` when the panel has not been laid out, so the box records `0`
while the panel sits elsewhere, which reads as an enormous user scroll. That is the same bug by a worse
route, and `test_every_written_position_reaches_the_box` catches it immediately. Any fix of this shape needs
a maximum it can trust, and the write site does not have one.

**A real defect found on the way, independent of the above.** The tolerance is

```python
_PIN_TOLERANCE_PX = 2 * gui_config.font_size  # two lines of text
```

with `font_size = 20`, so 40 px. **A rendered line is 26 px**, measured from the log: each streamed update
grows the content by 26, or 52 for two lines. So the constant that says "two lines of text" is worth about
one and a half, and both observed drifts - 46 and 47 px - fall in the gap between what it allows and what
two lines actually are. `font_size` is the glyph size; the line box also carries the item spacing.

That is a genuine bug in the constant, and widening it to a real two lines would have prevented both
observed instances. **It is not obviously safe to widen on its own**, which is why it is recorded rather than
applied: the comment above it warns that too large a bound makes a deliberate one- or two-line scroll away
from the end still count as being at the end, so the arrow keys look broken. Deciding that needs the app, not
arithmetic.

**And the question flagged earlier is now the blocker**, as expected: whether `scroll_animation` was retired
or merely converged. The evidence says retired - `settled_gap` was computed from `y_scroll` rather than from
a target, which is the `scroll_animation is None` branch - but the animation's last written value (4863) is
not its target (4868), and an animation that *completes* writes its target exactly. So it ended without
completing: cancelled, retargeted, or timed out on `update_pending_frames`. Which of those it was decides
whether the fix belongs in the check or in the animation's teardown.

## Fixed and confirmed, 2026-08-11

The cause is inside `SmoothScrolling`, and the earlier sections circle it without naming it. In smooth mode
the animation advances only once DPG reports the position it was last given - but DPG clamps a write to the
scrollable range *as it stands*, so a scroll aimed at the end that lands a frame before the content grows is
clamped short, and that equality can never hold. The animation waits four frames, times out, and stops,
leaving `commanded_y_scroll` holding a value the panel never took. Everything above follows from that: the
frozen record, the drift that never changes, the slack that is zero because the animation is gone.

The fix is one line of intent: on timing out, record the position DPG reports rather than the one we asked
for. Observed rather than predicted - see the dead end above for why predicting it does not work.

**Confirmed live.** Three turns, 162 scroll commands, the trigger firing three times and no refusal:

| wrote | panel reached | gap | before the fix |
|---|---|---|---|
| 3557 | 3540 | 17 px | inside the 40 px tolerance - survives |
| 4394 | 4346 | **48 px** | **over tolerance - would have latched** |
| 4956 | 4918 | 38 px | inside the tolerance - survives |

That last column is also the explanation of the intermittency, which nothing before had accounted for: the
clamp gap varies with timing, and the old code failed only when it happened to land above the tolerance. The
two recorded failures measured 46 and 47 px, a hair over 40. A 48 came up in this run and produced nothing,
because the record now agrees with the panel whatever the gap was.

It also settles the tolerance question raised above: with the record correct, the drift is zero regardless
of how large the clamp was, so widening `_PIN_TOLERANCE_PX` is unnecessary. Its comment has been corrected
where it stands, since "two lines of text" was wrong independently of this.

## Is the tolerance still needed? (asked 2026-08-11, not acted on)

Yes, but less of it, and it is now doing two jobs that want different sizes.

- **`at_end`** — "is the reader effectively at the end?" Needs a tolerance for rounding at minimum, and its
  size is what decides how large a deliberate scroll must be to escape follow-mode. This is the arrow-key
  constraint, and it is unaffected by the fix.
- **`undisturbed`** — "did the reader move the position?" This is the one the fix changed. The large drift
  source is gone, and what remains is the documented one-frame window where `_set_y_scroll` has written the
  box but DPG has not applied it — which `animation_slack` already covers while an animation runs. So the
  floor this needs is now near rounding error rather than 40 px.

The two share `_PIN_TOLERANCE_PX` because they used to need the same large value, which is exactly why the
comment describes it as squeezed from both sides: that is two constraints on one number. Splitting them
would let the drift floor drop to a few pixels — making genuine small scrolls easier to detect — while
`at_end` keeps whatever the keyboard needs. Worth doing; not done, and it wants the app.

### The unit is used as a proxy in three places, and it is consistently low

`font_size` stands in for the line height in `_PIN_TOLERANCE_PX`, in the chat view's per-keypress scroll,
and in the arrow-key step. It is low by about a quarter — a rendered line measures 26 px against a font size
of 20 — so an arrow moves about 3.8 lines where the count said 5, and the tolerance allows about 1.5 lines
where its comment said 2.

**The two scroll names have since been corrected to say what they measure**: `scroll_by_font_heights` and
`_SCROLL_FONT_HEIGHTS_PER_ARROW`, since a font height is what the arithmetic actually uses. Only
`_PIN_TOLERANCE_PX`'s comment still has to explain the discrepancy, because that one is a length rather than
a count.

**Nothing breaks, because the error is consistent.** The design argument the two share is that the
per-keypress step must clear the follow-tail floor or a streaming chunk would undo it, and that holds on the
ratio rather than on the unit: five font-sizes against two is 100 px against 40, a 2.5x margin, and both
scale together with the font. The comment beside `_SCROLL_FONT_HEIGHTS_PER_ARROW` states this correctly.

So this is cosmetic — the constants do not mean quite what they say — and worth knowing mainly because
`_PIN_TOLERANCE_PX`'s share of it was implicated in the bug above, where 46 and 47 px of drift exceeded a
bound that was supposed to be worth two lines. That is moot now the drift is zero. Recorded rather than
applied: changing the unit would move a keyboard gesture, and the margin that makes it work is unaffected.

The failure is silent and reads as the app being broken: the reply scrolls out of view while the model is
writing, which is precisely when a viewer is watching it. The recovery - a jump to the end at finalize - is
itself a visible lurch.

## Later episode: the settle wait believed the first standstill (2026-08-12)

Fault 1 below — `get_y_scroll_max` lagging a content change — came back in a second costume, and the fix
for it turned out to be one frame short.

**Symptom**: the chat panel came up part-way down its own content. On startup, and after the ▼ button
rebuilt the view, the message the reader had come back to sat below the fold; pressing End found it there
all along. So the tree walk was right and only the scroll was short — which is what made it read as a
loading bug rather than a scrolling one.

**Cause**: the panel's content is laid out in *pieces*, because the Markdown renderer runs on its own
worker — so `get_y_scroll_max` does not climb monotonically to its final value. It stands still *between*
pieces. Fault 1's settle wait asked for a single unchanged frame, and a lull satisfies that.

Measured at startup on a real chat:

| observation | `max_y_scroll` |
|---|---|
| first settle candidate | 3051 |
| a few frames later | 3497 |
| final | 4147 |

The scroll went to **3051** and left the reader **1096 px short**.

**Fixed 2026-08-12.** A standstill now has to last three consecutive frames
(`chat_controller._SCROLL_SETTLE_FRAMES`) before it is taken for the end, and a full rebuild allows sixty
frames for that rather than ten (`_BUILD_SCROLL_WAIT_FRAMES`) — laying out a chat from nothing takes many
more frames than appending one message to a chat already on screen. Same measurement after the change: the
first scroll waits four frames and goes to 4147.

**It is a heuristic and the code says so.** The renderer reports no "finished" event, so there is nothing to
wait on that would make this exact. Three frames buys headroom over the one-frame lull that was actually
observed, and costs nothing when the content settles sooner, which is the ordinary case.

Two things worth carrying:

- **The tell was that pressing End found the message.** A short scroll and a failed load look identical on
  screen; one keystroke separates them, and it points at the scroll rather than at the tree walk.
- **A probe, not reasoning, found it.** `max_y_scroll` was printed on each of the four frames after the
  scroll; the sequence above is what the log showed, and no amount of reading the settle loop would have
  suggested that the maximum pauses mid-layout. The probe was removed once it had answered.

## Prior episode: making the view follow at all (2026-07-30)

Re-homed here 2026-08-12 from `TODO_DEFERRED.md`, where it had accumulated under "Chat view scroll position
jumps back down while the model is writing" — an item that is now merged into "Holding the chat view's
scrollbar does not hold your place while a reply streams", since what is left live is the ImGui drift and
not this. The record belongs with the apparatus rather than on a list of things to do.

**The original defect**: while a reply streamed, the scroll position kept being pulled back to the bottom, so
scrolling up to re-read an earlier message did not stay put. `chat_controller` called `scroll_view()` with no
target — which scrolls to the end — at four points during a streaming turn, unconditionally.

Fixed 2026-07-30 and confirmed live. Three faults, each found by a live test. The final run, over a long
reply with a thinking block and a multi-screenful `webfetch` answer: **zero near-miss refusals**, following
correct throughout, the position-wait firing 115 times and needing more than one extra frame only once.
Honouring a scrolled-away reader worked from the start, including across tool calls; making the view
*follow* took all three.

**Fault 1 — `dpg.get_y_scroll_max` lags a content change by more than one frame.** `scroll_view` read the
maximum before the newly added message had been laid out, so "scroll to the end" landed where the *previous*
message ended (on Send, the view stayed on the greeting). Fixed with a settle-wait: the loop used to stop as
soon as `max_y_scroll > 0` and now stops only once that value is also unchanged from the previous frame,
still bounded by `max_wait_frames`. Same lag `SmoothScrolling` budgets four frames for
(`update_pending_threshold = 4`). The wait lives in `scroll_view` alone — `add_complete_message` and
`follow_tail` no longer `split_frame` on their own account, since one owner of the timing is the point.

**Fault 2 — the predicate could not tell arriving content from a user scroll, and getting it wrong latched.**
This was the one that kept the view frozen, and the log made it unmistakable: over a single reply the gap
grew 52 → 68 → 120 → 146 → 172 → 198 → 224 px and never recovered, with `scroll_view` never called once.

The mechanism: `is_pinned_to_bottom` compared the position against `max_y_scroll`. But two endpoints move
independently — the user moves the position, arriving content moves the maximum — so both causes produced the
same gap, and the view read its own content arriving as a reason to stop following. Because the verdict is
sampled once per chunk *before* that chunk renders, one false answer guarantees the next sample is taken from
a view one chunk further behind: monotonically worse, no recovery. A displacement of two lines was enough to
disable following for a whole turn.

Fixed by comparing against **the position we last commanded**, not the maximum — the shadow-state pattern,
and the same `_commanded_y_scroll` whose drift is the subject of the sections above. Content arrival cannot
change that relationship; a user scroll is exactly a change to the position we did not ask for. All of the
view's own scrolling goes through one private setter that records the commanded value and whether it was a
scroll-to-end, so the two causes separate with one remembered integer and no scroll events. Renamed to
`should_follow_tail`, because "is it at the bottom" is no longer the question it answers — it deliberately
returns `True` for a view that is *not* at the bottom but is still following.

**Fault 3 — `dpg.get_y_scroll` does not reflect a `dpg.set_y_scroll` for more than one frame,** so the
comparison introduced by fault 2's fix was reading our own in-flight command as a discrepancy. That is what
produced the one remaining dropout (mid chain-of-thought): `gap=52.0px ... drifted 52.0px from the 533.0 we
last commanded` — the panel was simply still at the previous position. Fixed by waiting in `scroll_view` for
the panel to report the position asked for, bounded by a round count and re-issuing the recomputed target
each round. Measured after the fix: one extra frame sufficed 114 times out of 115, two once, three never.

An earlier hypothesis — that DPG had clamped the command to a content height momentarily shortened by
`replace_last_paragraph`'s delete-then-add — **did not survive the log**: the first wait of the session read
a position of `0.0` against a maximum of `692.0`, where nothing had shrunk. That clamp window is real (the
`dpg.mutex()` that would make the swap atomic is disabled because holding it hangs the app) and recomputing
the target each round covers it for free, but it was not the cause of any measured case. Recorded because the
wrong mechanism was briefly written into the code comments and `dpg-notes.md`.

**Also dropped in the same pass: the refusal was initially made *sticky*, which looked careful and was the
opposite.** A reader who genuinely scrolls away keeps failing the drift test unaided, because they stay put
and we issue no further commands — so stickiness added no protection, only amplification, turning one wrong
refusal into a dead view for the rest of the reply. The log showed exactly that: every later refusal in the
affected turn reported `drift 0.0` with the flag already cleared. Each sample now decides on current evidence
and stores no verdict, so a wrong answer costs one chunk.

Diagnostics kept in place, and they are what made the 2026-08-11 episode above diagnosable from an ordinary
run: `should_follow_tail` logs both comparisons and the deciding branch at DEBUG, a near-miss refusal at
INFO, and `scroll_view` logs each wait round. For a future regression the number to read is the *drift* — a
nonzero drift with no user scrolling means something moved the position behind our back, which is a different
bug from a tolerance being too small.

### Two design constraints from that work, still live

**"Was", not "is", and that is the whole trick.** The at-the-bottom test has to be sampled *before* the new
content is added, and acted on after. Appending text grows the container, so `max_y_scroll` increases and a
view that was pinned to the bottom is no longer at the bottom the instant the chunk lands. Testing after the
append therefore reports "the user has scrolled away" every single time, autoscroll never engages, and the
view freezes wherever the stream began — a fix that fails in exactly the opposite direction from the bug, and
one that would look correct in the code.

**The same hazard reaches `ScrollEndFlasher`, so the predicate is shared in *form* but not in timing.** A
user-initiated scroll is not a quiet moment — the user can scroll *while* the model streams — so a chunk can
land between the flasher's sample and its act, and "you are at the end" becomes false as it is drawn. What
differs is the consequence, not the exposure: the flasher's failure is one wrong flash, the autoscroll's is a
view that never follows again. Do not fold them into a single "am I at the bottom" helper on the assumption
that one of them is safe; either pass the sampled state in, or take the size change into account explicitly.
