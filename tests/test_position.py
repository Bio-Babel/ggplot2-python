"""Tests for ggplot2_py.position — position adjustments."""

import pytest
from ggplot2_py import (
    PositionIdentity,
    PositionDodge,
    PositionStack,
    PositionJitter,
    PositionFill,
    PositionNudge,
    position_identity,
    position_dodge,
    position_stack,
    position_jitter,
    position_fill,
    position_nudge,
    is_position,
)


class TestPositionIdentity:
    """Test position_identity."""

    def test_creates_position_identity(self):
        p = position_identity()
        assert isinstance(p, PositionIdentity)


class TestPositionDodge:
    """Test position_dodge."""

    def test_creates_position_dodge(self):
        p = position_dodge(width=0.9)
        assert isinstance(p, PositionDodge)


class TestPositionStack:
    """Test position_stack."""

    def test_creates_position_stack(self):
        p = position_stack()
        assert isinstance(p, PositionStack)


class TestPositionJitter:
    """Test position_jitter."""

    def test_creates_position_jitter(self):
        p = position_jitter(width=0.3, height=0.3)
        assert isinstance(p, PositionJitter)


class TestPositionFill:
    """Test position_fill."""

    def test_creates_position_fill(self):
        p = position_fill()
        assert isinstance(p, PositionFill)


class TestIsPosition:
    """Test is_position predicate."""

    def test_true_for_identity(self):
        assert is_position(position_identity()) is True

    def test_true_for_dodge(self):
        assert is_position(position_dodge()) is True

    def test_true_for_stack(self):
        assert is_position(position_stack()) is True

    def test_true_for_jitter(self):
        assert is_position(position_jitter()) is True

    def test_false_for_string(self):
        assert is_position("identity") is False

    def test_false_for_none(self):
        assert is_position(None) is False
