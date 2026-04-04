"""Tests for ggplot2_py.plot — plot construction."""

import pytest
import pandas as pd
from ggplot2_py import (
    ggplot,
    is_ggplot,
    aes,
    geom_point,
    geom_line,
    geom_bar,
    scale_x_continuous,
    coord_flip,
    facet_wrap,
    theme_bw,
    labs,
)
from ggplot2_py.plot import GGPlot
from ggplot2_py.aes import Mapping
from ggplot2_py.labels import Labels


class TestGGPlotConstructor:
    """Test ggplot() constructor."""

    def test_creates_ggplot(self):
        p = ggplot()
        assert isinstance(p, GGPlot)

    def test_with_data(self, mpg):
        p = ggplot(mpg)
        assert isinstance(p.data, pd.DataFrame)
        assert p.data.shape == mpg.shape

    def test_with_mapping(self, mpg):
        p = ggplot(mpg, aes(x="displ"))
        assert isinstance(p.mapping, Mapping)
        assert p.mapping["x"] == "displ"

    def test_with_xy_mapping(self, mpg):
        p = ggplot(mpg, aes(x="displ", y="hwy"))
        assert p.mapping["x"] == "displ"
        assert p.mapping["y"] == "hwy"

    def test_empty_layers(self):
        p = ggplot()
        assert len(p.layers) == 0


class TestIsGGPlot:
    """Test is_ggplot predicate."""

    def test_true_for_ggplot(self):
        assert is_ggplot(ggplot()) is True

    def test_false_for_string(self):
        assert is_ggplot("not a plot") is False

    def test_false_for_none(self):
        assert is_ggplot(None) is False


class TestPlusOperator:
    """Test + operator for adding components to plots."""

    def test_add_layer(self, mpg):
        p = ggplot(mpg, aes(x="displ", y="hwy"))
        p2 = p + geom_point()
        assert len(p2.layers) == 1

    def test_add_scale(self, mpg):
        p = ggplot(mpg, aes(x="displ", y="hwy")) + geom_point()
        p2 = p + scale_x_continuous()
        assert len(p2.layers) == 1  # layers unchanged

    def test_add_coord(self, mpg):
        p = ggplot(mpg, aes(x="displ", y="hwy")) + geom_point()
        p2 = p + coord_flip()
        assert hasattr(p2, "coordinates")
        from ggplot2_py import CoordFlip
        assert isinstance(p2.coordinates, CoordFlip)

    def test_add_facet(self, mpg):
        p = ggplot(mpg, aes(x="displ", y="hwy")) + geom_point()
        p2 = p + facet_wrap("class")
        from ggplot2_py import FacetWrap
        assert isinstance(p2.facet, FacetWrap)

    def test_add_theme(self, mpg):
        p = ggplot(mpg, aes(x="displ", y="hwy")) + geom_point()
        p2 = p + theme_bw()
        from ggplot2_py.theme import Theme
        assert isinstance(p2.theme, Theme)

    def test_add_labels(self, mpg):
        p = ggplot(mpg, aes(x="displ", y="hwy")) + geom_point()
        p2 = p + labs(title="Test")
        assert "title" in p2.labels
        assert p2.labels["title"] == "Test"

    def test_multiple_layers(self, mpg):
        p = ggplot(mpg, aes(x="displ", y="hwy"))
        p2 = p + geom_point() + geom_line()
        assert len(p2.layers) == 2

    def test_original_unchanged(self, mpg):
        """Adding layers should not mutate the original plot."""
        p = ggplot(mpg, aes(x="displ", y="hwy"))
        _ = p + geom_point()
        assert len(p.layers) == 0
