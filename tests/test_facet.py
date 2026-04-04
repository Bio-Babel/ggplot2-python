"""Tests for ggplot2_py.facet — faceting system."""

import pytest
from ggplot2_py import (
    FacetWrap,
    FacetGrid,
    FacetNull,
    facet_wrap,
    facet_grid,
    facet_null,
    is_facet,
)


class TestFacetWrap:
    """Test facet_wrap."""

    def test_creates_facet_wrap(self):
        f = facet_wrap("class")
        assert isinstance(f, FacetWrap)

    def test_is_facet(self):
        f = facet_wrap("class")
        assert is_facet(f) is True


class TestFacetGrid:
    """Test facet_grid."""

    def test_creates_facet_grid(self):
        f = facet_grid(rows="drv", cols="cyl")
        assert isinstance(f, FacetGrid)

    def test_is_facet(self):
        f = facet_grid(rows="drv")
        assert is_facet(f) is True


class TestFacetNull:
    """Test facet_null."""

    def test_creates_facet_null(self):
        f = facet_null()
        assert isinstance(f, FacetNull)

    def test_is_facet(self):
        f = facet_null()
        assert is_facet(f) is True


class TestIsFacet:
    """Test is_facet predicate."""

    def test_true_for_wrap(self):
        assert is_facet(facet_wrap("x")) is True

    def test_true_for_null(self):
        assert is_facet(facet_null()) is True

    def test_false_for_string(self):
        assert is_facet("wrap") is False

    def test_false_for_none(self):
        assert is_facet(None) is False
