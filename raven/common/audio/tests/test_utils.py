"""Unit tests for raven.common.audio.utils."""

import math

import numpy as np
import pytest

from raven.common.audio.utils import dBFS_to_linear, linear_to_dBFS


class TestLinearToDBFS:
    def test_full_scale_int_positive(self):
        assert linear_to_dBFS(32767) == pytest.approx(0.0)

    def test_full_scale_int_negative(self):
        # -32768 is also full scale in s16 (the asymmetric range uses 32768 as fs for negatives).
        assert linear_to_dBFS(-32768) == pytest.approx(0.0)

    def test_full_scale_float_positive(self):
        assert linear_to_dBFS(1.0) == pytest.approx(0.0)

    def test_full_scale_float_negative(self):
        assert linear_to_dBFS(-1.0) == pytest.approx(0.0)

    def test_zero_int_is_minus_inf(self):
        result = linear_to_dBFS(0)
        assert math.isinf(result) and result < 0

    def test_zero_float_is_minus_inf(self):
        result = linear_to_dBFS(0.0)
        assert math.isinf(result) and result < 0

    def test_half_amplitude_float_is_minus_6dB(self):
        # 20 log10(0.5) ≈ -6.0206 dB
        assert linear_to_dBFS(0.5) == pytest.approx(-6.0206, abs=1e-3)

    def test_sign_is_dropped(self):
        assert linear_to_dBFS(-0.5) == pytest.approx(linear_to_dBFS(0.5))

    def test_s16_high_byte_threshold(self):
        # Documented landmark: the high byte of s16 goes to zero at about -42.15 dB.
        assert linear_to_dBFS(255) == pytest.approx(-42.15, abs=0.05)

    def test_numpy_int16_accepted(self):
        assert linear_to_dBFS(np.int16(32767)) == pytest.approx(0.0)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            linear_to_dBFS("not a number")


class TestDBFSToLinear:
    def test_zero_db_int_is_full_scale(self):
        assert dBFS_to_linear(0.0, int) == pytest.approx(32767.0)

    def test_zero_db_float_is_one(self):
        assert dBFS_to_linear(0.0, float) == pytest.approx(1.0)

    def test_minus_6db_is_half_amplitude_float(self):
        assert dBFS_to_linear(-6.0206, float) == pytest.approx(0.5, abs=1e-3)

    def test_minus_inf_is_zero_float(self):
        # 10^(-inf / 20) → 0.
        assert dBFS_to_linear(-math.inf, float) == 0.0

    def test_unsupported_format_raises(self):
        with pytest.raises(TypeError):
            dBFS_to_linear(-20.0, str)

    def test_numpy_int_is_not_accepted_as_format(self):
        # `format` must be exactly `int` or `float` — a numpy dtype is not equivalent.
        with pytest.raises(TypeError):
            dBFS_to_linear(-20.0, np.int16)


class TestRoundtrip:
    @pytest.mark.parametrize("level", [1.0, 0.5, 0.1, 0.001, 1e-6])
    def test_float_roundtrip(self, level):
        dB = linear_to_dBFS(level)
        recovered = dBFS_to_linear(dB, float)
        assert recovered == pytest.approx(level, rel=1e-6)

    @pytest.mark.parametrize("level", [32767, 16384, 1000, 100, 1])
    def test_int_roundtrip(self, level):
        dB = linear_to_dBFS(level)
        recovered = dBFS_to_linear(dB, int)
        assert recovered == pytest.approx(level, rel=1e-4)
