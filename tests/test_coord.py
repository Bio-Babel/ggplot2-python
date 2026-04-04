"""Tests for ggplot2_py.coord — coordinate systems."""

import pytest
from ggplot2_py import (
    CoordCartesian,
    CoordFlip,
    CoordPolar,
    CoordFixed,
    coord_cartesian,
    coord_flip,
    coord_polar,
    coord_fixed,
    is_coord,
)


class TestCoordCartesian:
    """Test coord_cartesian."""

    def test_creates_coord_cartesian(self):
        c = coord_cartesian()
        assert isinstance(c, CoordCartesian)

    def test_is_linear(self):
        c = coord_cartesian()
        assert c.is_linear() is True


class TestCoordFlip:
    """Test coord_flip."""

    def test_creates_coord_flip(self):
        c = coord_flip()
        assert isinstance(c, CoordFlip)


class TestCoordPolar:
    """Test coord_polar."""

    def test_creates_coord_polar(self):
        c = coord_polar()
        assert isinstance(c, CoordPolar)


class TestCoordFixed:
    """Test coord_fixed."""

    def test_creates_coord_fixed(self):
        c = coord_fixed()
        assert isinstance(c, CoordFixed)


class TestIsCoord:
    """Test is_coord predicate."""

    def test_true_for_coord(self):
        assert is_coord(coord_cartesian()) is True

    def test_true_for_flip(self):
        assert is_coord(coord_flip()) is True

    def test_true_for_polar(self):
        assert is_coord(coord_polar()) is True

    def test_false_for_string(self):
        assert is_coord("cartesian") is False

    def test_false_for_none(self):
        assert is_coord(None) is False
