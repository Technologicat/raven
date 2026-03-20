"""Framerate-independent exponential decay animation.

Provides `SmoothValue` (float) and `SmoothInt` (integer with subpixel
tracking) for animated transitions that adapt to sudden target changes.

Both classes use Newton's law of cooling (first-order ODE) with
FPS-corrected stepping — the animation speed is constant in wall-clock
time regardless of the render frame rate.

The `rate` parameter, in the half-open interval (0, 1], controls
how quickly the value approaches the target. Higher values mean faster
animation. For example, a rate of 0.8 means 80% of the remaining
distance is covered per frame at *calibration FPS* (25 FPS).
The animator compensates for actual FPS automatically.

Terminology: ``rate`` is the user-facing parameter you configure — the fraction
of remaining distance covered per frame at ``CALIBRATION_FPS``.  ``step`` is
the FPS-corrected value actually applied each frame (the output of
``fps_corrected_step``).  The distinction matters because ``rate`` is constant
while ``step`` varies with frame rate.
"""

__all__ = ["SmoothValue", "SmoothInt", "fps_corrected_step", "CALIBRATION_FPS"]

import math
import time
from typing import Optional


CALIBRATION_FPS = 25  # FPS for which `rate` was calibrated


def fps_corrected_step(rate: float, dt: float) -> float:
    """Compute FPS-corrected interpolation step size.

    Given a `rate` calibrated at `CALIBRATION_FPS`, return the FPS-corrected
    step that produces equivalent wall-clock animation speed at the current
    actual frame rate (inferred from `dt`).
    """
    # The `rate` parameter is calibrated against animation at `CALIBRATION_FPS`, so we must
    # scale it appropriately, taking into account the actual FPS.
    #
    # How to do this requires some explanation. Numericist hat on. Let's do a quick
    # back-of-the-envelope calculation. The interpolator is essentially a solver for
    # the first-order ODE:
    #
    #   u' = f(u, t)
    #
    # Consider the most common case, where the target remains constant over several
    # animation frames. Then our ODE is Newton's law of cooling:
    #
    #   u' = -β [u - u∞]
    #
    # where `u = u(t)` is the temperature, `u∞` is the constant temperature of the
    # external environment, and `β > 0` is a material-dependent cooling coefficient.
    #
    # But instead of numerical simulation at a constant timestep size, as would be typical
    # in computational science, we instead read off points off the analytical solution curve.
    # The `rate` parameter is *not* the timestep size; instead, it controls the relative
    # distance along the *u* axis that should be covered in one simulation step, so it is
    # actually related to the cooling coefficient β.
    #
    # (How exactly: write the left-hand side as `[unew - uold] / Δt + O([Δt]²)`, drop the
    #  error term, and decide whether to use `uold` (forward Euler) or `unew` (backward Euler)
    #  as `u` on the right-hand side. Then compare to our update formula. But those details
    #  don't matter here.)
    #
    # To match the notation used in the code, let us denote the temperature (actually
    # current value) as `x` (instead of `u`). And to keep notation shorter, let
    # `β := rate` (although it's not exactly the `β` of the continuous-in-time case above).
    #
    # To scale the animation speed linearly with regard to FPS, we must invert the relation
    # between simulation step number `n` and the solution value `x`. For an initial value `x0`,
    # a constant target value `x∞`, and constant step `β ∈ (0, 1]`, the interpolator produces
    # the sequence:
    #
    #   x1 = x0 + β [x∞ - x0] = [1 - β] x0 + β x∞
    #   x2 = x1 + β [x∞ - x1] = [1 - β] x1 + β x∞
    #   x3 = x2 + β [x∞ - x2] = [1 - β] x2 + β x∞
    #   ...
    #
    # Note that with exact arithmetic, if `β < 1`, the final value is only reached in the
    # limit `n → ∞`. For floating point, this is not the case. Eventually the increment
    # becomes small enough that when it is added, nothing happens. After sufficiently many
    # steps, in practice `x` will stop just slightly short of `x∞` (on the side it approached
    # the target from).
    #
    # Also, for performance reasons, when approaching zero, one may need to beware of denormals,
    # because those are usually implemented in (slow!) software on modern CPUs. So especially
    # if the target is zero, it is useful to have some very small cutoff (inside the normal
    # floating-point range) after which we make `x` instantly jump to the target value.
    #
    # Inserting the definition of `x1` to the formula for `x2`, we can express `x2` in terms
    # of `x0` and `x∞`:
    #
    #   x2 = [1 - β] ([1 - β] x0 + β x∞) + β x∞
    #      = [1 - β]² x0 + [1 - β] β x∞ + β x∞
    #      = [1 - β]² x0 + [[1 - β] + 1] β x∞
    #
    # Then inserting this to the formula for `x3`:
    #
    #   x3 = [1 - β] ([1 - β]² x0 + [[1 - β] + 1] β x∞) + β x∞
    #      = [1 - β]³ x0 + [1 - β]² β x∞ + [1 - β] β x∞ + β x∞
    #
    # To simplify notation, define:
    #
    #   α := 1 - β
    #
    # We have:
    #
    #   x1 = α  x0 + [1 - α] x∞
    #   x2 = α² x0 + [1 - α] [1 + α] x∞
    #      = α² x0 + [1 - α²] x∞
    #   x3 = α³ x0 + [1 - α] [1 + α + α²] x∞
    #      = α³ x0 + [1 - α³] x∞
    #
    # This suggests that the general pattern is (as can be proven by induction on `n`):
    #
    #   xn = α**n x0 + [1 - α**n] x∞
    #
    # This allows us to determine `x` as a function of simulation step number `n`. Now the
    # scaling question becomes: if we want to reach a given value `xn` by some given step
    # `n_scaled` (instead of the original step `n`), how must we change the step size `β`
    # (or equivalently, the parameter `α`)?
    #
    # To simplify further, observe:
    #
    #   x1 = α x0 + [1 - α] [[x∞ - x0] + x0]
    #      = [α + [1 - α]] x0 + [1 - α] [x∞ - x0]
    #      = x0 + [1 - α] [x∞ - x0]
    #
    # Rearranging yields:
    #
    #   [x1 - x0] / [x∞ - x0] = 1 - α
    #
    # which gives us the relative distance from `x0` to `x∞` that is covered in one step.
    # This isn't yet much to write home about (it's essentially just a rearrangement of the
    # definition of `x1`), but next, let's treat `x2` the same way:
    #
    #   x2 = α² x0 + [1 - α] [1 + α] [[x∞ - x0] + x0]
    #      = [α² x0 + [1 - α²] x0] + [1 - α²] [x∞ - x0]
    #      = [α² + 1 - α²] x0 + [1 - α²] [x∞ - x0]
    #      = x0 + [1 - α²] [x∞ - x0]
    #
    # We obtain
    #
    #   [x2 - x0] / [x∞ - x0] = 1 - α²
    #
    # which is the relative distance, from the original `x0` toward the final `x∞`, that is
    # covered in two steps using the original step size `β = 1 - α`. Next up, `x3`:
    #
    #   x3 = α³ x0 + [1 - α³] [[x∞ - x0] + x0]
    #      = α³ x0 + [1 - α³] [x∞ - x0] + [1 - α³] x0
    #      = x0 + [1 - α³] [x∞ - x0]
    #
    # Rearranging,
    #
    #   [x3 - x0] / [x∞ - x0] = 1 - α³
    #
    # which is the relative distance covered in three steps. Hence, we have:
    #
    #   xrel := [xn - x0] / [x∞ - x0] = 1 - α**n
    #
    # so that
    #
    #   α**n = 1 - xrel              (**)
    #
    # and (taking the natural logarithm of both sides)
    #
    #   n log α = log [1 - xrel]
    #
    # Finally,
    #
    #   n = [log [1 - xrel]] / [log α]
    #
    # Given `α`, this gives the `n` where the interpolator has covered the fraction `xrel`
    # of the original distance. On the other hand, we can also solve (**) for `α`:
    #
    #   α = (1 - xrel)**(1 / n)
    #
    # which, given desired `n`, gives us the `α` that makes the interpolator cover the
    # fraction `xrel` of the original distance in `n` steps.
    alpha_orig = 1.0 - rate
    if 0 < alpha_orig < 1:
        avg_fps = (1.0 / dt) if dt > 0 else CALIBRATION_FPS

        xrel = 0.5
        n_orig = math.log(1.0 - xrel) / math.log(alpha_orig)
        n_scaled = (avg_fps / CALIBRATION_FPS) * n_orig
        if n_scaled > 0:
            alpha_scaled = (1.0 - xrel) ** (1.0 / n_scaled)
        else:
            alpha_scaled = alpha_orig
    else:
        alpha_scaled = alpha_orig

    return 1.0 - alpha_scaled


class SmoothValue:
    """An animated float value using first-order ODE solution.

    The animation depends only on current and target positions, using
    the analytical solution to Newton's law of cooling (exponential decay
    toward target). Hence this adapts to sudden target value changes.

    Provides smooth transitions between values, without a fixed duration.
    """

    EPSILON = 1e-3  # Snap-to-target threshold; also denormal guard

    def __init__(self, value: float = 0.0, rate: float = 0.3):
        """Initialize with a starting value.

        `value`: Initial value.
        `rate`: Animation rate, in (0, 1]. Higher = faster. Default 0.3.
        """
        self._current = value
        self._target = value
        self._rate = rate
        self._last_time = time.time()

    @property
    def current(self) -> float:
        """The current (animated) value."""
        return self._current

    @property
    def target(self) -> float:
        """The target value the animation is moving toward."""
        return self._target

    @target.setter
    def target(self, value: float) -> None:
        """Set a new target value."""
        # Reset the time reference so the first animation frame has a
        # reasonable dt, not a stale delta from the last update() call
        # (which may have been seconds/minutes ago if nothing was animating).
        if not self.is_animating():
            self._last_time = time.time()
        self._target = value

    def set_immediate(self, value: float) -> None:
        """Set both current and target value immediately (no animation)."""
        self._current = value
        self._target = value

    def is_animating(self) -> bool:
        """Return True if the value is still animating toward target."""
        return abs(self._target - self._current) > self.EPSILON

    def update(self, dt: Optional[float] = None) -> bool:
        """Advance the animation by one frame.

        `dt`: Time delta in seconds. If None, computed from wall clock.

        Returns True if still animating, False if reached target.
        """
        if not self.is_animating():
            self._current = self._target
            return False

        now = time.time()
        if dt is None:
            dt = now - self._last_time
        self._last_time = now

        step = fps_corrected_step(self._rate, dt)
        remaining = self._target - self._current
        new_value = self._current + step * remaining

        if abs(self._target - new_value) <= self.EPSILON:
            self._current = self._target
            return False
        self._current = new_value
        return True


class SmoothInt:
    """An animated integer value with subpixel tracking.

    Like `SmoothValue`, but ``current`` returns an integer (truncated).
    Internally tracks a float to prevent quantization drift during
    animation — fractional deltas accumulate correctly across frames.

    Suitable for scroll positions, pixel coordinates, and other
    integer-valued quantities that need smooth animation.
    """

    EPSILON = 1.0  # Snap at 1 unit — sub-pixel animation is invisible

    def __init__(self, value: int = 0, rate: float = 0.8):
        """Initialize with a starting value.

        `value`: Initial value.
        `rate`: Animation rate, in (0, 1]. Higher = faster. Default 0.8.
        """
        self._current_float = float(value)
        self._target = value
        self._rate = rate
        self._last_time = time.time()

    @property
    def current(self) -> int:
        """The current (animated) value, truncated to int."""
        return int(self._current_float)

    @property
    def current_exact(self) -> float:
        """The current value at full float precision (for diagnostics)."""
        return self._current_float

    @property
    def target(self) -> int:
        """The target value the animation is moving toward."""
        return self._target

    @target.setter
    def target(self, value: int) -> None:
        """Set a new target value."""
        if not self.is_animating():
            self._last_time = time.time()
        self._target = value

    def set_immediate(self, value: int) -> None:
        """Set both current and target value immediately (no animation)."""
        self._current_float = float(value)
        self._target = value

    def is_animating(self) -> bool:
        """Return True if the value is still animating toward target."""
        return abs(self._target - self._current_float) > self.EPSILON

    def update(self, dt: Optional[float] = None) -> bool:
        """Advance the animation by one frame.

        `dt`: Time delta in seconds. If None, computed from wall clock.

        Returns True if still animating, False if reached target.
        """
        if not self.is_animating():
            self._current_float = float(self._target)
            return False

        now = time.time()
        if dt is None:
            dt = now - self._last_time
        self._last_time = now

        step = fps_corrected_step(self._rate, dt)
        remaining = self._target - self._current_float
        new_value = self._current_float + step * remaining

        if abs(self._target - new_value) <= self.EPSILON:
            self._current_float = float(self._target)
            return False
        self._current_float = new_value
        return True
