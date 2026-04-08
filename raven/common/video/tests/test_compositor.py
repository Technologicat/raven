"""Tests for raven.common.video.compositor — cel animation and alpha blending."""

import time

import torch

from raven.common.video import compositor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_rgba(r, g, b, a, h=4, w=4):
    """Create a uniform RGBA image tensor."""
    img = torch.zeros(4, h, w)
    img[0] = r; img[1] = g; img[2] = b; img[3] = a  # noqa: E702
    return img


# ---------------------------------------------------------------------------
# Tests: render_celstack — alpha blending
# ---------------------------------------------------------------------------

class TestRenderCelstack:
    """Alpha compositing (over operator) correctness."""

    def test_empty_celstack_returns_clone(self):
        """Empty celstack returns a clone of the base image."""
        base = _solid_rgba(1.0, 0.0, 0.0, 1.0)
        result = compositor.render_celstack(base, [], {})
        assert torch.equal(result, base)
        assert result is not base  # must be a clone

    def test_opaque_over_opaque(self):
        """Fully opaque cel completely occludes the base."""
        base = _solid_rgba(1.0, 0.0, 0.0, 1.0)  # red
        cel = _solid_rgba(0.0, 1.0, 0.0, 1.0)  # green
        stack = [("overlay", 1.0)]
        cels = {"overlay": cel}
        result = compositor.render_celstack(base, stack, cels)
        # Green should dominate
        assert torch.allclose(result[1], torch.ones(4, 4), atol=1e-4)  # G ≈ 1
        assert torch.allclose(result[0], torch.zeros(4, 4), atol=1e-4)  # R ≈ 0

    def test_transparent_cel_is_noop(self):
        """Fully transparent cel doesn't affect base."""
        base = _solid_rgba(1.0, 0.0, 0.0, 1.0)
        cel = _solid_rgba(0.0, 1.0, 0.0, 0.0)  # transparent green
        stack = [("overlay", 1.0)]
        cels = {"overlay": cel}
        result = compositor.render_celstack(base, stack, cels)
        assert torch.allclose(result[:3], base[:3], atol=1e-4)

    def test_zero_strength_skips_cel(self):
        """Cel with strength 0.0 is skipped entirely."""
        base = _solid_rgba(1.0, 0.0, 0.0, 1.0)
        cel = _solid_rgba(0.0, 1.0, 0.0, 1.0)
        stack = [("overlay", 0.0)]
        cels = {"overlay": cel}
        result = compositor.render_celstack(base, stack, cels)
        assert torch.allclose(result[:3], base[:3], atol=1e-4)

    def test_half_strength_blends(self):
        """Cel at half strength produces partial blending."""
        base = _solid_rgba(1.0, 0.0, 0.0, 1.0)
        cel = _solid_rgba(0.0, 1.0, 0.0, 1.0)  # opaque green
        stack = [("overlay", 0.5)]
        cels = {"overlay": cel}
        result = compositor.render_celstack(base, stack, cels)
        # Strength 0.5 halves the cel's alpha → partial blend
        # Neither pure red nor pure green
        assert result[0, 0, 0].item() > 0.1
        assert result[1, 0, 0].item() > 0.1

    def test_missing_cel_skipped(self):
        """Cel not present in torch_cels dict is silently skipped."""
        base = _solid_rgba(1.0, 0.0, 0.0, 1.0)
        stack = [("nonexistent", 1.0)]
        result = compositor.render_celstack(base, stack, {})
        assert torch.allclose(result[:3], base[:3], atol=1e-4)

    def test_does_not_mutate_base(self):
        """render_celstack must not modify the base image."""
        base = _solid_rgba(1.0, 0.0, 0.0, 1.0)
        base_copy = base.clone()
        cel = _solid_rgba(0.0, 1.0, 0.0, 1.0)
        stack = [("overlay", 1.0)]
        cels = {"overlay": cel}
        compositor.render_celstack(base, stack, cels)
        assert torch.equal(base, base_copy)

    def test_stacking_order(self):
        """Later cels in the stack sit on top (occlude earlier ones)."""
        base = _solid_rgba(0.0, 0.0, 0.0, 1.0)  # black
        red = _solid_rgba(1.0, 0.0, 0.0, 1.0)
        blue = _solid_rgba(0.0, 0.0, 1.0, 1.0)
        stack = [("red", 1.0), ("blue", 1.0)]
        cels = {"red": red, "blue": blue}
        result = compositor.render_celstack(base, stack, cels)
        # Blue is on top, should dominate
        assert result[2, 0, 0].item() > 0.9
        assert result[0, 0, 0].item() < 0.1


# ---------------------------------------------------------------------------
# Tests: get_cel_index_in_stack
# ---------------------------------------------------------------------------

class TestGetCelIndex:
    def test_found(self):
        stack = [("a", 1.0), ("b", 0.5), ("c", 0.0)]
        assert compositor.get_cel_index_in_stack("b", stack) == 1

    def test_not_found(self):
        stack = [("a", 1.0), ("b", 0.5)]
        assert compositor.get_cel_index_in_stack("z", stack) == -1

    def test_first_occurrence(self):
        """Returns the index of the first occurrence."""
        stack = [("a", 1.0), ("b", 0.5), ("a", 0.0)]
        assert compositor.get_cel_index_in_stack("a", stack) == 0

    def test_empty_stack(self):
        assert compositor.get_cel_index_in_stack("a", []) == -1


# ---------------------------------------------------------------------------
# Tests: animation drivers — stateless, time-based
# ---------------------------------------------------------------------------

class TestAnimateCelCycle:
    """animate_cel_cycle: looping cel animation."""

    def test_disabled_when_zero_duration(self):
        """cycle_duration=0.0 disables animation (no-op)."""
        stack = [("a", 0.0), ("b", 0.0)]
        epoch, result = compositor.animate_cel_cycle(0.0, 0, 1.0, ["a", "b"], stack)
        # Nothing should change
        assert result == stack

    def test_disabled_when_no_cels(self):
        """Empty cels list disables animation."""
        stack = [("a", 0.0)]
        epoch, result = compositor.animate_cel_cycle(1.0, 0, 1.0, [], stack)
        assert result == stack

    def test_activates_one_cel(self):
        """At any point in the cycle, exactly one cel should have nonzero strength."""
        cels = ["a", "b", "c"]
        stack = [(name, 0.0) for name in cels]
        # epoch = right now, so we're at the start of the cycle
        epoch = time.monotonic_ns()
        _, result = compositor.animate_cel_cycle(10.0, epoch, 0.8, cels, stack)
        active = [(name, s) for name, s in result if s > 0.0]
        assert len(active) == 1
        assert active[0][1] == 0.8

    def test_does_not_mutate_input(self):
        stack = [("a", 0.5), ("b", 0.5)]
        original = list(stack)
        compositor.animate_cel_cycle(1.0, time.monotonic_ns(), 1.0, ["a", "b"], stack)
        assert stack == original


class TestAnimateCelSequence:
    """animate_cel_sequence: one-shot cel animation."""

    def test_disabled_when_zero_duration(self):
        stack = [("a", 0.0)]
        result = compositor.animate_cel_sequence(0, 0.0, 1.0, ["a"], stack)
        assert result == stack

    def test_after_duration_no_change(self):
        """After the sequence duration, no cel should be activated."""
        stack = [("a", 0.0), ("b", 0.0)]
        # t0 far in the past
        t0 = time.monotonic_ns() - int(10 * 10**9)
        result = compositor.animate_cel_sequence(t0, 1.0, 1.0, ["a", "b"], stack)
        # All strengths should remain 0
        for _, s in result:
            assert s == 0.0


class TestAnimateCelFadeout:
    """animate_cel_fadeout: fade cels to zero over time."""

    def test_at_start_full_strength(self):
        """At t=0, fadeout hasn't started — strength should be preserved."""
        stack = [("a", 1.0)]
        t0 = time.monotonic_ns()
        result = compositor.animate_cel_fadeout(t0, 10.0, ["a"], stack)
        # Should be very close to original strength (we're at ~t=0)
        assert result[0][1] > 0.95

    def test_after_duration_zero_strength(self):
        """After the fadeout duration, strength should be 0."""
        stack = [("a", 1.0)]
        t0 = time.monotonic_ns() - int(10 * 10**9)
        result = compositor.animate_cel_fadeout(t0, 1.0, ["a"], stack)
        assert result[0][1] == 0.0

    def test_disabled_when_zero_duration(self):
        stack = [("a", 1.0)]
        result = compositor.animate_cel_fadeout(0, 0.0, ["a"], stack)
        assert result == stack
