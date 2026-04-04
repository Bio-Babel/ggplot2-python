"""Tests for ggplot2_py.guide — guide system."""

import pytest
from ggplot2_py import (
    GuideLegend,
    GuideColourbar,
    GuideNone,
    guide_legend,
    guide_colourbar,
    guide_colorbar,
    guide_none,
    guides,
    is_guide,
)
from ggplot2_py.guide import Guides


class TestGuideLegend:
    """Test guide_legend."""

    def test_creates_guide_legend(self):
        g = guide_legend()
        assert isinstance(g, GuideLegend)

    def test_is_guide(self):
        g = guide_legend()
        assert is_guide(g) is True


class TestGuideColourbar:
    """Test guide_colourbar."""

    def test_creates_guide_colourbar(self):
        g = guide_colourbar()
        assert isinstance(g, GuideColourbar)

    def test_colorbar_alias(self):
        # American spelling alias
        assert guide_colorbar is guide_colourbar


class TestGuideNone:
    """Test guide_none."""

    def test_creates_guide_none(self):
        g = guide_none()
        assert isinstance(g, GuideNone)


class TestGuides:
    """Test guides() container."""

    def test_creates_guides(self):
        g = guides(colour=guide_legend())
        assert isinstance(g, Guides)


class TestIsGuide:
    """Test is_guide predicate."""

    def test_true_for_legend(self):
        assert is_guide(guide_legend()) is True

    def test_true_for_colourbar(self):
        assert is_guide(guide_colourbar()) is True

    def test_true_for_none_guide(self):
        assert is_guide(guide_none()) is True

    def test_false_for_string(self):
        assert is_guide("legend") is False

    def test_false_for_none(self):
        assert is_guide(None) is False
