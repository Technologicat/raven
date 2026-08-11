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

Raw excerpt: `near-miss-2026-08-11.txt`.

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

## Why this is worth fixing before a demo

The failure is silent and reads as the app being broken: the reply scrolls out of view while the model is
writing, which is precisely when a viewer is watching it. The recovery - a jump to the end at finalize - is
itself a visible lurch.
