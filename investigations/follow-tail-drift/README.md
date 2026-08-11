# The chat view stops following the tail mid-reply

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

Raw excerpts: `near-miss-2026-08-11.txt` (the run that recovered) and `latched-2026-08-11.txt` (the one that
did not).

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

## There is no scroll event to consult, and the docstring already says so

`should_follow_tail` explains why it infers intent from position rather than from events: "of the three ways
this panel moves - scrollbar drag, mouse wheel, navigation keys - the drag is handled inside ImGui and
raises nothing we could hook". The wheel is the same. `_user_scroll_generation` is authoritative only for
the third way, the one that goes through `_set_y_scroll`, so consulting it would miss both of the others.

**Two ideas that do not survive contact with that**, recorded because they are the ones that suggest
themselves:

- *Bump the user-scroll signal from the input handlers `app.py` already registers.* Weaker than it looks,
  though not dead. The handlers are viewport-global, but `guiutils.is_mouse_inside_widget` can narrow a
  wheel event to the chat panel, which makes **the wheel case precise**: a wheel event with the pointer over
  the panel is a reader scrolling it, and nothing else is. The drag is the hard half - it raises nothing to
  hook at all, so it would have to be reconstructed from a click inside the panel plus subsequent motion,
  and the mouse-move handler fires on any pointer motion anywhere. Worth having for the wheel; not a
  complete answer on its own.
- *Bias the ambiguous case toward the reader.* There is no such direction. Deciding "theirs" stops
  following, which abandons a reader who is watching the stream; deciding "ours" keeps following, which
  drags back a reader who scrolled away. Both are the reader. The test has to be accurate rather than
  conservative, which is why this is worth diagnosing properly instead of tuning.

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

## Why this is worth fixing before a demo

The failure is silent and reads as the app being broken: the reply scrolls out of view while the model is
writing, which is precisely when a viewer is watching it. The recovery - a jump to the end at finalize - is
itself a visible lurch.
