"""Tests for ggplot2_py.scale — scale system."""

import pytest
from ggplot2_py import (
    Scale,
    ScaleContinuousPosition,
    ScaleDiscrete,
    scale_x_continuous,
    scale_y_continuous,
    scale_colour_hue,
    scale_fill_manual,
    scale_colour_gradient,
    scale_color_gradient,
    scale_colour_viridis_c,
    expansion,
    is_scale,
)


class TestScaleXContinuous:
    """Test scale_x_continuous."""

    def test_creates_correct_type(self):
        s = scale_x_continuous()
        assert isinstance(s, ScaleContinuousPosition)

    def test_is_scale(self):
        s = scale_x_continuous()
        assert is_scale(s) is True


class TestScaleColourHue:
    """Test scale_colour_hue."""

    def test_creates_scale(self):
        s = scale_colour_hue()
        assert is_scale(s)

    def test_is_discrete(self):
        s = scale_colour_hue()
        assert isinstance(s, ScaleDiscrete)


class TestScaleFillManual:
    """Test scale_fill_manual with dict values."""

    def test_with_dict_values(self):
        s = scale_fill_manual(values={"a": "red", "b": "blue"})
        assert is_scale(s)

    def test_is_discrete(self):
        s = scale_fill_manual(values={"a": "red", "b": "blue"})
        assert isinstance(s, ScaleDiscrete)


class TestAmericanSpellingAliases:
    """Test American spelling aliases for colour scales."""

    def test_color_gradient_is_colour_gradient(self):
        assert scale_color_gradient is scale_colour_gradient


class TestExpansion:
    """Test expansion() helper."""

    def test_returns_array_like(self):
        e = expansion(mult=0.05)
        assert len(e) == 4  # [mult_lower, add_lower, mult_upper, add_upper]

    def test_mult_values(self):
        e = expansion(mult=0.05)
        assert float(e[0]) == pytest.approx(0.05)
        assert float(e[2]) == pytest.approx(0.05)

    def test_add_values(self):
        e = expansion(mult=0, add=1)
        assert float(e[1]) == pytest.approx(1.0)
        assert float(e[3]) == pytest.approx(1.0)


class TestIsScale:
    """Test is_scale predicate."""

    def test_true_for_scale(self):
        assert is_scale(scale_x_continuous()) is True

    def test_false_for_string(self):
        assert is_scale("not a scale") is False

    def test_false_for_none(self):
        assert is_scale(None) is False
