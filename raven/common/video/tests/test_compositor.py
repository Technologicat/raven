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

    def test_strength_scales_cel_alpha_not_base(self):
        """Strength multiplies the cel's alpha, not the base's."""
        base = _solid_rgba(1.0, 0.0, 0.0, 1.0)
        cel = _solid_rgba(0.0, 1.0, 0.0, 0.5)  # half-transparent green
        stack = [("overlay", 0.5)]
        cels = {"overlay": cel}
        result = compositor.render_celstack(base, stack, cels)
        # Effective cel alpha = 0.5 * 0.5 = 0.25. Base alpha = 1.0.
        # Base should still dominate heavily (red > green)
        assert result[0, 0, 0].item() > result[1, 0, 0].item()

    def test_does_not_mutate_cel_at_full_strength(self):
        """At strength=1.0, the cel tensor must not be modified."""
        base = _solid_rgba(0.0, 0.0, 0.0, 1.0)
        cel = _solid_rgba(0.0, 1.0, 0.0, 1.0)
        cel_copy = cel.clone()
        stack = [("overlay", 1.0)]
        cels = {"overlay": cel}
        compositor.render_celstack(base, stack, cels)
        assert torch.equal(cel, cel_copy)

    def test_does_not_mutate_cel_at_partial_strength(self):
        """At strength < 1.0, the original cel tensor must not be modified."""
        base = _solid_rgba(0.0, 0.0, 0.0, 1.0)
        cel = _solid_rgba(0.0, 1.0, 0.0, 1.0)
        cel_copy = cel.clone()
        stack = [("overlay", 0.5)]
        cels = {"overlay": cel}
        compositor.render_celstack(base, stack, cels)
        assert torch.equal(cel, cel_copy)


class TestRenderCelstackAlphaMath:
    """Numerical correctness of the Porter-Duff 'over' operator."""

    def test_semitransparent_over_opaque(self):
        """50% alpha cel over opaque base: known analytical result.

        'over' formula:  alpha_o = alpha_a + alpha_b * (1 - alpha_a)
                         RGB_o   = (RGB_a * alpha_a + RGB_b * alpha_b * (1 - alpha_a)) / alpha_o

        With alpha_a=0.5 (green), alpha_b=1.0 (red):
            alpha_o = 0.5 + 1.0 * 0.5 = 1.0
            R_o = (0.0 * 0.5 + 1.0 * 1.0 * 0.5) / 1.0 = 0.5
            G_o = (1.0 * 0.5 + 0.0 * 1.0 * 0.5) / 1.0 = 0.5
            B_o = 0.0
        """
        base = _solid_rgba(1.0, 0.0, 0.0, 1.0)
        cel = _solid_rgba(0.0, 1.0, 0.0, 0.5)
        stack = [("overlay", 1.0)]
        cels = {"overlay": cel}
        result = compositor.render_celstack(base, stack, cels)
        assert torch.allclose(result[0, 0, 0], torch.tensor(0.5), atol=1e-4)
        assert torch.allclose(result[1, 0, 0], torch.tensor(0.5), atol=1e-4)
        assert torch.allclose(result[2, 0, 0], torch.tensor(0.0), atol=1e-4)
        assert torch.allclose(result[3, 0, 0], torch.tensor(1.0), atol=1e-4)

    def test_semitransparent_over_semitransparent(self):
        """Both layers semitransparent: check output alpha.

        alpha_a=0.5, alpha_b=0.5:
            alpha_o = 0.5 + 0.5 * 0.5 = 0.75
        """
        base = _solid_rgba(1.0, 0.0, 0.0, 0.5)
        cel = _solid_rgba(0.0, 1.0, 0.0, 0.5)
        stack = [("overlay", 1.0)]
        cels = {"overlay": cel}
        result = compositor.render_celstack(base, stack, cels)
        assert torch.allclose(result[3, 0, 0], torch.tensor(0.75), atol=1e-4)

    def test_transparent_over_transparent(self):
        """Both fully transparent: output should be transparent too."""
        base = _solid_rgba(1.0, 0.0, 0.0, 0.0)
        cel = _solid_rgba(0.0, 1.0, 0.0, 0.0)
        stack = [("overlay", 1.0)]
        cels = {"overlay": cel}
        result = compositor.render_celstack(base, stack, cels)
        assert torch.allclose(result[3, 0, 0], torch.tensor(0.0), atol=1e-4)


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

    def test_cycles_through_all_cels(self):
        """Over a full cycle, every cel should get its turn."""
        cels = ["a", "b", "c"]
        stack = [(name, 0.0) for name in cels]
        cycle_duration = 3.0  # seconds
        seen = set()
        # Sample at 10 evenly spaced points across the cycle.
        # Use fake epochs to control the cycle position deterministically.
        now = time.monotonic_ns()
        for i in range(10):
            # Place epoch in the past so that (now - epoch) / cycle_duration = i/10
            elapsed_ns = int((i / 10) * cycle_duration * 10**9)
            epoch = now - elapsed_ns
            _, result = compositor.animate_cel_cycle(cycle_duration, epoch, 1.0, cels, stack)
            for name, s in result:
                if s > 0.0:
                    seen.add(name)
        assert seen == {"a", "b", "c"}

    def test_epoch_resets_after_full_cycle(self):
        """After a full cycle, the returned epoch should advance."""
        cels = ["a", "b"]
        stack = [(name, 0.0) for name in cels]
        # Epoch far in the past — well past one cycle
        old_epoch = time.monotonic_ns() - int(5 * 10**9)
        new_epoch, _ = compositor.animate_cel_cycle(1.0, old_epoch, 1.0, cels, stack)
        assert new_epoch > old_epoch

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

    def test_activates_cel_during_sequence(self):
        """During the sequence, exactly one cel should be active."""
        cels = ["a", "b"]
        stack = [(name, 0.0) for name in cels]
        t0 = time.monotonic_ns()  # just started
        result = compositor.animate_cel_sequence(t0, 10.0, 0.8, cels, stack)
        active = [(name, s) for name, s in result if s > 0.0]
        assert len(active) == 1
        assert active[0][1] == 0.8

    def test_does_not_mutate_input(self):
        stack = [("a", 0.0), ("b", 0.0)]
        original = list(stack)
        compositor.animate_cel_sequence(time.monotonic_ns(), 10.0, 1.0, ["a"], stack)
        assert stack == original


class TestAnimateCelFadeout:
    """animate_cel_fadeout: fade cels to zero over time."""

    def test_at_start_full_strength(self):
        """At t=0, fadeout hasn't started — strength should be preserved."""
        stack = [("a", 1.0)]
        t0 = time.monotonic_ns()
        result = compositor.animate_cel_fadeout(t0, 10.0, ["a"], stack)
        # Should be very close to original strength (we're at ~t=0)
        assert result[0][1] > 0.95

    def test_mid_fade_partial_strength(self):
        """Halfway through the fadeout, strength should be roughly halved."""
        stack = [("a", 1.0)]
        # Place t0 so that half the duration has elapsed
        duration = 2.0
        t0 = time.monotonic_ns() - int(duration / 2 * 10**9)
        result = compositor.animate_cel_fadeout(t0, duration, ["a"], stack)
        # Linear fade: at t=duration/2, r = 0.5, so output strength ≈ 0.5
        assert 0.3 < result[0][1] < 0.7

    def test_after_duration_zero_strength(self):
        """After the fadeout duration, strength should be 0."""
        stack = [("a", 1.0)]
        t0 = time.monotonic_ns() - int(10 * 10**9)
        result = compositor.animate_cel_fadeout(t0, 1.0, ["a"], stack)
        assert result[0][1] == 0.0

    def test_scales_existing_strength(self):
        """Fadeout multiplies the existing strength, doesn't replace it."""
        stack = [("a", 0.6)]
        t0 = time.monotonic_ns()  # just started, r ≈ 1.0
        result = compositor.animate_cel_fadeout(t0, 10.0, ["a"], stack)
        # Should preserve the original strength (r ≈ 1.0, so output ≈ 0.6)
        assert result[0][1] > 0.5

    def test_disabled_when_zero_duration(self):
        stack = [("a", 1.0)]
        result = compositor.animate_cel_fadeout(0, 0.0, ["a"], stack)
        assert result == stack

    def test_does_not_mutate_input(self):
        stack = [("a", 1.0)]
        original = list(stack)
        compositor.animate_cel_fadeout(time.monotonic_ns(), 10.0, ["a"], stack)
        assert stack == original

    def test_only_affects_listed_cels(self):
        """Cels not in the fadeout list should be unaffected."""
        stack = [("a", 1.0), ("b", 1.0)]
        t0 = time.monotonic_ns() - int(10 * 10**9)  # far past duration
        result = compositor.animate_cel_fadeout(t0, 1.0, ["a"], stack)
        assert result[0][1] == 0.0  # "a" faded
        assert result[1][1] == 1.0  # "b" untouched


# ---------------------------------------------------------------------------
# Tests: combined animation drivers
# ---------------------------------------------------------------------------

class TestAnimateCelCycleWithFadeout:
    """animate_cel_cycle_with_fadeout: cycling + fading out."""

    def test_disabled_when_cycle_disabled(self):
        """Zero cycle duration disables the cycle part; fadeout still applies."""
        stack = [("a", 1.0), ("b", 1.0)]
        t0 = time.monotonic_ns() - int(10 * 10**9)
        _, result = compositor.animate_cel_cycle_with_fadeout(
            cycle_duration=0.0, epoch=0, strength=1.0,
            fadeout_t0=t0, fadeout_duration=1.0,
            cels=["a", "b"], celstack=stack)
        # Cycle disabled → strengths untouched by cycle.
        # But fadeout has elapsed → strengths zeroed.
        assert result[0][1] == 0.0
        assert result[1][1] == 0.0

    def test_active_cel_fades(self):
        """The active cycling cel should have its strength reduced by the fadeout."""
        cels = ["a", "b"]
        stack = [(name, 0.0) for name in cels]
        now = time.monotonic_ns()
        # Cycle just started, fadeout halfway through
        fadeout_duration = 2.0
        fadeout_t0 = now - int(fadeout_duration / 2 * 10**9)
        _, result = compositor.animate_cel_cycle_with_fadeout(
            cycle_duration=10.0, epoch=now, strength=1.0,
            fadeout_t0=fadeout_t0, fadeout_duration=fadeout_duration,
            cels=cels, celstack=stack)
        active = [(name, s) for name, s in result if s > 0.0]
        # One cel should be active, but at reduced strength (~0.5)
        assert len(active) == 1
        assert active[0][1] < 0.7

    def test_does_not_mutate_input(self):
        stack = [("a", 0.0), ("b", 0.0)]
        original = list(stack)
        compositor.animate_cel_cycle_with_fadeout(
            cycle_duration=10.0, epoch=time.monotonic_ns(), strength=1.0,
            fadeout_t0=time.monotonic_ns(), fadeout_duration=10.0,
            cels=["a", "b"], celstack=stack)
        assert stack == original


class TestAnimateCelSequenceWithFadeout:
    """animate_cel_sequence_with_fadeout: one-shot sequence + fade."""

    def test_after_duration_all_zero(self):
        """After both sequence and fadeout have elapsed, all strengths should be 0."""
        cels = ["a", "b"]
        stack = [(name, 0.0) for name in cels]
        t0 = time.monotonic_ns() - int(10 * 10**9)
        result = compositor.animate_cel_sequence_with_fadeout(
            t0=t0, duration=1.0, strength=1.0,
            cels=cels, celstack=stack)
        for _, s in result:
            assert s == 0.0

    def test_active_cel_fades_during_sequence(self):
        """During the sequence, the active cel should have faded strength."""
        cels = ["a", "b"]
        stack = [(name, 0.0) for name in cels]
        # Start now — we're at t≈0, so fadeout factor r ≈ 1.0
        t0 = time.monotonic_ns()
        result = compositor.animate_cel_sequence_with_fadeout(
            t0=t0, duration=10.0, strength=1.0,
            cels=cels, celstack=stack)
        active = [(name, s) for name, s in result if s > 0.0]
        # At t≈0: sequence activates first cel, fadeout r≈1.0 → strength ≈ 1.0
        assert len(active) == 1
        assert active[0][1] > 0.9

    def test_does_not_mutate_input(self):
        stack = [("a", 0.0), ("b", 0.0)]
        original = list(stack)
        compositor.animate_cel_sequence_with_fadeout(
            t0=time.monotonic_ns(), duration=10.0, strength=1.0,
            cels=["a", "b"], celstack=stack)
        assert stack == original
