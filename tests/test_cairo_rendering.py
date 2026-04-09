"""Tests for Cairo rendering pipeline — _repr_png_, print_plot, ggsave."""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

from ggplot2_py import ggplot, aes, geom_point, geom_line, geom_bar, labs
from ggplot2_py.plot import GGPlot, ggplot_build, ggplot_gtable, print_plot
from ggplot2_py.save import ggsave


# ------------------------------------------------------------------ #
# Fixtures                                                           #
# ------------------------------------------------------------------ #


@pytest.fixture
def simple_plot(mpg):
    """A minimal scatter plot for rendering tests."""
    return ggplot(mpg, aes(x="displ", y="hwy")) + geom_point()


@pytest.fixture
def labeled_plot(mpg):
    """A plot with title and axis labels."""
    return (
        ggplot(mpg, aes(x="displ", y="hwy"))
        + geom_point()
        + labs(title="Test Plot", x="Displacement", y="Highway MPG")
    )


# ------------------------------------------------------------------ #
# _repr_png_                                                         #
# ------------------------------------------------------------------ #


class TestReprPng:
    """_repr_png_ should produce valid PNG bytes via Cairo."""

    def test_returns_bytes(self, simple_plot):
        data = simple_plot._repr_png_()
        assert data is not None
        assert isinstance(data, bytes)

    def test_png_magic(self, simple_plot):
        data = simple_plot._repr_png_()
        assert data[:4] == b"\x89PNG"

    def test_nonempty(self, simple_plot):
        data = simple_plot._repr_png_()
        assert len(data) > 500  # A real plot image, not just a header

    def test_with_labels(self, labeled_plot):
        data = labeled_plot._repr_png_()
        assert data is not None
        assert len(data) > 500

    def test_custom_dimensions(self, mpg):
        p = ggplot(mpg, aes(x="displ", y="hwy")) + geom_point()
        p.fig_width = 10.0
        p.fig_height = 8.0
        p.fig_dpi = 72
        data = p._repr_png_()
        assert data is not None


# ------------------------------------------------------------------ #
# print_plot                                                         #
# ------------------------------------------------------------------ #


class TestPrintPlot:
    """print_plot should render without error via Cairo."""

    def test_no_error(self, simple_plot):
        result = print_plot(simple_plot)
        assert result is simple_plot

    def test_with_labels(self, labeled_plot):
        result = print_plot(labeled_plot)
        assert result is labeled_plot


# ------------------------------------------------------------------ #
# ggsave — PNG                                                       #
# ------------------------------------------------------------------ #


class TestGgsavePng:
    """ggsave with PNG output via Cairo."""

    def test_creates_file(self, simple_plot):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            ggsave(path, plot=simple_plot, width=4, height=3, dpi=72)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 500
        finally:
            os.unlink(path)

    def test_png_magic(self, simple_plot):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            ggsave(path, plot=simple_plot, width=4, height=3, dpi=72)
            with open(path, "rb") as fh:
                assert fh.read(4) == b"\x89PNG"
        finally:
            os.unlink(path)

    def test_custom_dpi(self, simple_plot):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            ggsave(path, plot=simple_plot, width=4, height=3, dpi=300)
            assert os.path.getsize(path) > 1000
        finally:
            os.unlink(path)

    def test_create_dir(self, simple_plot):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "sub", "dir")
            path = os.path.join(subdir, "plot.png")
            ggsave(path, plot=simple_plot, width=3, height=2, dpi=72,
                   create_dir=True)
            assert os.path.exists(path)


# ------------------------------------------------------------------ #
# ggsave — PDF                                                       #
# ------------------------------------------------------------------ #


class TestGgsavePdf:
    """ggsave with PDF output via Cairo."""

    def test_creates_file(self, simple_plot):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            ggsave(path, plot=simple_plot, width=4, height=3)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 100
        finally:
            os.unlink(path)

    def test_pdf_magic(self, simple_plot):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            ggsave(path, plot=simple_plot, width=4, height=3)
            with open(path, "rb") as fh:
                assert fh.read(5) == b"%PDF-"
        finally:
            os.unlink(path)


# ------------------------------------------------------------------ #
# ggsave — SVG                                                       #
# ------------------------------------------------------------------ #


class TestGgsaveSvg:
    """ggsave with SVG output via Cairo."""

    def test_creates_file(self, simple_plot):
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            path = f.name
        try:
            ggsave(path, plot=simple_plot, width=4, height=3)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 100
        finally:
            os.unlink(path)

    def test_svg_content(self, simple_plot):
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            path = f.name
        try:
            ggsave(path, plot=simple_plot, width=4, height=3)
            with open(path, "r") as fh:
                content = fh.read()
            assert "<svg" in content or "<?xml" in content
        finally:
            os.unlink(path)


# ------------------------------------------------------------------ #
# ggsave — dimension and unit handling                               #
# ------------------------------------------------------------------ #


class TestGgsaveDimensions:
    """ggsave dimension and unit options."""

    def test_cm_units(self, simple_plot):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            ggsave(path, plot=simple_plot, width=10, height=8,
                   units="cm", dpi=72)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_mm_units(self, simple_plot):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            ggsave(path, plot=simple_plot, width=100, height=80,
                   units="mm", dpi=72)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_scale(self, simple_plot):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            ggsave(path, plot=simple_plot, width=4, height=3,
                   dpi=72, scale=2.0)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_limitsize_raises(self, simple_plot):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(Exception):
                ggsave(path, plot=simple_plot, width=60, height=60,
                       dpi=72, limitsize=True)
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ------------------------------------------------------------------ #
# ggplot_build + ggplot_gtable pipeline                              #
# ------------------------------------------------------------------ #


class TestBuildPipeline:
    """ggplot_build and ggplot_gtable should work with Cairo backend."""

    def test_build_returns_object(self, simple_plot):
        built = ggplot_build(simple_plot)
        assert built is not None
        assert hasattr(built, "data")
        assert hasattr(built, "layout")

    def test_gtable_from_build(self, simple_plot):
        built = ggplot_build(simple_plot)
        gtable = ggplot_gtable(built)
        assert gtable is not None
