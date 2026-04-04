"""Tests for ggplot2_py.labels — label helpers and limits."""

import pytest
from ggplot2_py import labs, xlab, ylab, ggtitle, xlim, ylim
from ggplot2_py.labels import Labels


class TestLabs:
    """Test labs() function."""

    def test_creates_labels(self):
        l = labs(title="T", x="X", y="Y")
        assert isinstance(l, Labels)

    def test_title_stored(self):
        l = labs(title="My Title")
        assert l["title"] == "My Title"

    def test_x_stored(self):
        l = labs(x="X axis")
        assert l["x"] == "X axis"

    def test_y_stored(self):
        l = labs(y="Y axis")
        assert l["y"] == "Y axis"

    def test_multiple_keys(self):
        l = labs(title="T", x="X", y="Y", subtitle="S")
        assert "title" in l
        assert "x" in l
        assert "y" in l
        assert "subtitle" in l


class TestXlab:
    """Test xlab() function."""

    def test_creates_labels_with_x(self):
        l = xlab("X axis")
        assert isinstance(l, Labels)
        assert l["x"] == "X axis"


class TestYlab:
    """Test ylab() function."""

    def test_creates_labels_with_y(self):
        l = ylab("Y axis")
        assert isinstance(l, Labels)
        assert l["y"] == "Y axis"


class TestGgtitle:
    """Test ggtitle() function."""

    def test_creates_labels_with_title(self):
        l = ggtitle("My Title")
        assert isinstance(l, Labels)
        assert l["title"] == "My Title"


class TestYlim:
    """Test ylim() function."""

    def test_creates_continuous_scale(self):
        from ggplot2_py import ScaleContinuousPosition
        s = ylim(0, 100)
        assert isinstance(s, ScaleContinuousPosition)


class TestXlim:
    """Test xlim() function."""

    def test_creates_discrete_scale_for_strings(self):
        from ggplot2_py import ScaleDiscretePosition
        s = xlim("a", "b", "c")
        assert isinstance(s, ScaleDiscretePosition)

    def test_creates_continuous_scale_for_numbers(self):
        from ggplot2_py import ScaleContinuousPosition
        s = xlim(0, 10)
        assert isinstance(s, ScaleContinuousPosition)
