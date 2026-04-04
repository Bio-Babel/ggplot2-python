"""Tests for ggplot2_py.geom — geom construction and types."""

import pytest
from ggplot2_py import (
    Layer,
    is_layer,
    is_geom,
    Geom,
    GeomPoint,
    GeomBar,
    GeomLine,
    GeomBoxplot,
    GeomHistogram,
    GeomSmooth,
    GeomText,
    GeomRect,
    GeomTile,
    GeomRaster,
    GeomViolin,
    GeomArea,
    GeomRibbon,
    GeomPolygon,
    GeomDensity,
    GeomAbline,
    GeomHline,
    GeomVline,
    GeomRug,
    GeomBlank,
    GeomErrorbar,
    GeomSegment,
    GeomCol,
    GeomPath,
    GeomStep,
    GeomLabel,
    geom_point,
    geom_bar,
    geom_line,
    geom_boxplot,
    geom_histogram,
    geom_smooth,
    geom_text,
    geom_rect,
    geom_tile,
    geom_raster,
    geom_violin,
    geom_area,
    geom_ribbon,
    geom_polygon,
    geom_density,
    geom_abline,
    geom_hline,
    geom_vline,
    geom_rug,
    geom_blank,
    geom_errorbar,
    geom_segment,
    geom_col,
    geom_path,
    geom_step,
    geom_label,
    geom_hex,
    geom_bin2d,
    geom_contour,
    geom_jitter,
    geom_freqpoly,
    geom_count,
    geom_crossbar,
    geom_linerange,
    geom_pointrange,
    geom_curve,
    geom_dotplot,
)


# All geom constructor functions to test
GEOM_CONSTRUCTORS = [
    geom_point,
    geom_bar,
    geom_line,
    geom_boxplot,
    geom_histogram,
    geom_smooth,
    geom_text,
    geom_rect,
    geom_tile,
    geom_violin,
    geom_area,
    geom_ribbon,
    geom_polygon,
    geom_density,
    geom_abline,
    geom_hline,
    geom_vline,
    geom_rug,
    geom_blank,
    geom_errorbar,
    geom_segment,
    geom_col,
    geom_path,
    geom_step,
    geom_label,
    geom_hex,
    geom_bin2d,
    geom_contour,
    geom_jitter,
    geom_freqpoly,
    geom_count,
    geom_crossbar,
    geom_linerange,
    geom_pointrange,
    geom_curve,
    geom_dotplot,
]


class TestGeomConstructors:
    """Test that geom_*() functions return Layer objects."""

    @pytest.mark.parametrize("geom_fn", GEOM_CONSTRUCTORS, ids=lambda f: f.__name__)
    def test_returns_layer(self, geom_fn):
        layer = geom_fn()
        assert isinstance(layer, Layer), f"{geom_fn.__name__} did not return a Layer"

    @pytest.mark.parametrize("geom_fn", GEOM_CONSTRUCTORS, ids=lambda f: f.__name__)
    def test_is_layer(self, geom_fn):
        layer = geom_fn()
        assert is_layer(layer)


class TestGeomPointDefaults:
    """Test GeomPoint default aesthetics."""

    def test_default_aes_has_xy(self):
        defaults = GeomPoint.default_aes
        assert "x" not in defaults  # x,y are required, not default
        # But default_aes should have shape, colour, etc.
        assert "shape" in defaults or "colour" in defaults

    def test_geom_point_has_geom(self):
        layer = geom_point()
        assert hasattr(layer, "geom")


class TestGeomStatDefaults:
    """Test default stat assignments on geoms."""

    def test_geom_bar_default_stat_is_count(self):
        layer = geom_bar()
        assert layer.stat == "count"

    def test_geom_histogram_default_stat_is_bin(self):
        layer = geom_histogram()
        assert layer.stat == "bin"

    def test_geom_point_default_stat_is_identity(self):
        layer = geom_point()
        assert layer.stat == "identity"

    def test_geom_smooth_default_stat_is_smooth(self):
        layer = geom_smooth()
        assert layer.stat == "smooth"

    def test_geom_boxplot_default_stat_is_boxplot(self):
        layer = geom_boxplot()
        assert layer.stat == "boxplot"


class TestIsGeom:
    """Test is_geom predicate."""

    def test_true_for_instance(self):
        assert is_geom(GeomPoint()) is True

    def test_true_for_class(self):
        assert is_geom(GeomPoint) is True

    def test_false_for_string(self):
        assert is_geom("point") is False

    def test_false_for_none(self):
        assert is_geom(None) is False
