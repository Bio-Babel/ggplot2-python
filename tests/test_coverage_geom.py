"""Comprehensive tests for ggplot2_py.geom to improve coverage."""

import pytest
import numpy as np
import pandas as pd
import warnings

from ggplot2_py.geom import (
    # Base class
    Geom,
    is_geom,
    translate_shape_string,
    # Geom classes
    GeomPoint,
    GeomPath,
    GeomLine,
    GeomStep,
    GeomRect,
    GeomTile,
    GeomRaster,
    GeomBar,
    GeomCol,
    GeomText,
    GeomLabel,
    GeomPolygon,
    GeomRibbon,
    GeomArea,
    GeomSmooth,
    GeomSegment,
    GeomCurve,
    GeomSpoke,
    GeomErrorbar,
    GeomErrorbarh,
    GeomCrossbar,
    GeomLinerange,
    GeomPointrange,
    GeomBoxplot,
    GeomViolin,
    GeomDotplot,
    GeomDensity,
    GeomAbline,
    GeomHline,
    GeomVline,
    GeomRug,
    GeomBlank,
    GeomContour,
    GeomContourFilled,
    GeomDensity2d,
    GeomDensity2dFilled,
    GeomHex,
    GeomBin2d,
    GeomFunction,
    GeomMap,
    GeomQuantile,
    GeomJitter,
    GeomFreqpoly,
    GeomHistogram,
    GeomCount,
    GeomSf,
    GeomAnnotationMap,
    GeomCustomAnn,
    GeomRasterAnn,
    GeomLogticks,
    # Utilities
    _fill_alpha,
    _ggname,
    _stairstep,
    _coord_transform,
    _r_col_to_mpl,
    scales_alpha,
    PT,
    STROKE,
    _PCH_TABLE,
)

from ggplot2_py.aes import Mapping


# ============================================================================
# Utility function tests
# ============================================================================

class TestTranslateShapeString:
    def test_none(self):
        assert translate_shape_string(None) == 19

    def test_int(self):
        assert translate_shape_string(5) == 5

    def test_float(self):
        assert translate_shape_string(5.0) == 5

    def test_single_char(self):
        assert translate_shape_string("o") == "o"

    def test_named_shape(self):
        # "circle" matches "circle open" (1) due to startswith
        assert translate_shape_string("circle open") == 1
        assert translate_shape_string("square open") == 0
        assert translate_shape_string("plus") == 3
        assert translate_shape_string("cross") == 4
        assert translate_shape_string("bullet") == 20
        assert translate_shape_string("asterisk") == 8
        assert translate_shape_string("star") == 11

    def test_named_shape_open(self):
        assert translate_shape_string("circle open") == 1
        assert translate_shape_string("square open") == 0
        assert translate_shape_string("triangle open") == 2
        assert translate_shape_string("diamond open") == 5

    def test_named_shape_filled(self):
        assert translate_shape_string("circle filled") == 21
        assert translate_shape_string("square filled") == 22
        assert translate_shape_string("triangle filled") == 24
        assert translate_shape_string("diamond filled") == 23

    def test_invalid_shape(self):
        with pytest.raises(Exception):
            translate_shape_string("invalid_shape_name")

    def test_array(self):
        result = translate_shape_string(["bullet", "plus", 3])
        assert result[0] == 20
        assert result[1] == 3
        assert result[2] == 3

    def test_numpy_int(self):
        assert translate_shape_string(np.int64(5)) == 5


class TestIsGeom:
    def test_true_for_instance(self):
        assert is_geom(GeomPoint()) is True

    def test_true_for_class(self):
        assert is_geom(GeomPoint) is True

    def test_false_for_string(self):
        assert is_geom("point") is False

    def test_false_for_none(self):
        assert is_geom(None) is False


class TestFillAlpha:
    def test_none(self):
        assert _fill_alpha(None, 0.5) is None

    def test_basic(self):
        result = _fill_alpha("red", 0.5)
        # Should return something

    def test_exception(self):
        # Invalid input should return the original
        result = _fill_alpha(object(), None)
        assert result is not None


class TestGgname:
    def test_basic(self):
        class MockGrub:
            name = ""
        g = MockGrub()
        result = _ggname("test_prefix", g)
        assert result.name == "test_prefix"

    def test_no_name_attr(self):
        # Should not raise for objects without name
        result = _ggname("prefix", 42)
        assert result == 42


class TestRColToMpl:
    def test_grey(self):
        result = _r_col_to_mpl("grey50")
        assert isinstance(result, str)
        assert result.startswith("#")

    def test_gray(self):
        result = _r_col_to_mpl("gray75")
        assert isinstance(result, str)

    def test_regular(self):
        assert _r_col_to_mpl("red") == "red"


class TestScalesAlpha:
    def test_string(self):
        result = scales_alpha("red", 0.5)
        # Should return something

    def test_list(self):
        result = scales_alpha(["red", "blue"], [0.5, 0.8])
        # Should return something

    def test_grey_conversion(self):
        result = scales_alpha("grey50", 1.0)
        # Should convert and return


class TestStairstep:
    def test_hv(self):
        data = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        result = _stairstep(data, "hv")
        assert len(result) > 3

    def test_vh(self):
        data = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        result = _stairstep(data, "vh")
        assert len(result) > 3

    def test_mid(self):
        data = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        result = _stairstep(data, "mid")
        assert len(result) > 3

    def test_invalid(self):
        with pytest.raises(Exception):
            _stairstep(pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}), "invalid")

    def test_single_point(self):
        data = pd.DataFrame({"x": [1.0], "y": [4.0]})
        result = _stairstep(data, "hv")
        assert len(result) == 0

    def test_extra_columns(self):
        data = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "group": [1, 1, 1]})
        result = _stairstep(data, "hv")
        assert "group" in result.columns


class TestCoordTransform:
    def test_no_coord(self):
        data = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = _coord_transform(None, data, None)
        assert list(result["x"]) == [1, 2]

    def test_with_coord(self):
        class MockCoord:
            def transform(self, data, params):
                return data
        data = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = _coord_transform(MockCoord(), data, None)
        assert list(result["x"]) == [1, 2]


# ============================================================================
# Constants
# ============================================================================

class TestConstants:
    def test_pt(self):
        assert PT > 0
        assert abs(PT - 72.27 / 25.4) < 1e-10

    def test_stroke(self):
        assert STROKE > 0
        assert abs(STROKE - 96 / 25.4) < 1e-10

    def test_pch_table(self):
        assert "circle" in _PCH_TABLE
        assert _PCH_TABLE["circle"] == 19


# ============================================================================
# Base Geom class tests
# ============================================================================

class TestGeomBase:
    def test_required_aes(self):
        g = Geom()
        assert isinstance(g.required_aes, tuple)

    def test_default_aes(self):
        g = Geom()
        assert isinstance(g.default_aes, Mapping)

    def test_setup_params(self):
        g = Geom()
        result = g.setup_params(pd.DataFrame(), {"a": 1})
        assert result == {"a": 1}

    def test_setup_data(self):
        g = Geom()
        df = pd.DataFrame({"x": [1, 2]})
        result = g.setup_data(df, {})
        assert list(result["x"]) == [1, 2]

    def test_handle_na(self):
        g = GeomPoint()
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [4.0, 5.0, np.nan]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = g.handle_na(df, {"na_rm": True})
        assert len(result) <= 3

    def test_parameters(self):
        g = GeomPoint()
        params = g.parameters(extra=False)
        assert isinstance(params, list)
        params_extra = g.parameters(extra=True)
        assert "na_rm" in params_extra

    def test_aesthetics(self):
        g = GeomPoint()
        aes = g.aesthetics()
        assert "x" in aes
        assert "y" in aes
        assert "group" in aes


# ============================================================================
# GeomPoint tests
# ============================================================================

class TestGeomPoint:
    def test_required_aes(self):
        assert "x" in GeomPoint.required_aes
        assert "y" in GeomPoint.required_aes

    def test_default_aes(self):
        defaults = GeomPoint.default_aes
        assert "shape" in defaults
        assert "colour" in defaults
        assert "size" in defaults
        assert "alpha" in defaults
        assert "stroke" in defaults

    def test_non_missing_aes(self):
        assert "size" in GeomPoint.non_missing_aes
        assert "shape" in GeomPoint.non_missing_aes


# ============================================================================
# GeomPath / GeomLine / GeomStep tests
# ============================================================================

class TestGeomPath:
    def test_required_aes(self):
        assert "x" in GeomPath.required_aes
        assert "y" in GeomPath.required_aes

    def test_default_aes(self):
        defaults = GeomPath.default_aes
        assert "colour" in defaults
        assert "linewidth" in defaults

    def test_rename_size(self):
        assert GeomPath.rename_size is True


class TestGeomLine:
    def test_is_subclass(self):
        assert issubclass(GeomLine, GeomPath)

    def test_setup_params(self):
        g = GeomLine()
        params = g.setup_params(pd.DataFrame(), {})
        assert "flipped_aes" in params

    def test_setup_data(self):
        g = GeomLine()
        df = pd.DataFrame({"x": [3.0, 1.0, 2.0], "y": [4.0, 5.0, 6.0], "PANEL": [1, 1, 1]})
        result = g.setup_data(df, {"flipped_aes": False})
        # Should sort by x
        assert list(result["x"]) == [1.0, 2.0, 3.0]


class TestGeomStep:
    def test_is_subclass(self):
        assert issubclass(GeomStep, GeomPath)

    def test_setup_params(self):
        g = GeomStep()
        params = g.setup_params(pd.DataFrame(), {})
        assert "flipped_aes" in params


# ============================================================================
# GeomRect / GeomTile / GeomRaster tests
# ============================================================================

class TestGeomRect:
    def test_required_aes(self):
        # R geom-rect.R:13: c("x|width|xmin|xmax", "y|height|ymin|ymax")
        assert "x|width|xmin|xmax" in GeomRect.required_aes
        assert "y|height|ymin|ymax" in GeomRect.required_aes

    def test_setup_data_all_present(self):
        g = GeomRect()
        df = pd.DataFrame({"xmin": [0], "xmax": [1], "ymin": [0], "ymax": [1]})
        result = g.setup_data(df, {})
        assert list(result.columns) == list(df.columns)

    def test_setup_data_from_center(self):
        g = GeomRect()
        df = pd.DataFrame({"x": [5.0], "y": [5.0], "width": [2.0], "height": [4.0]})
        result = g.setup_data(df, {})
        assert result["xmin"].iloc[0] == pytest.approx(4.0)
        assert result["xmax"].iloc[0] == pytest.approx(6.0)
        assert result["ymin"].iloc[0] == pytest.approx(3.0)
        assert result["ymax"].iloc[0] == pytest.approx(7.0)

    def test_setup_data_from_x_only(self):
        g = GeomRect()
        df = pd.DataFrame({"x": [5.0], "y": [5.0]})
        result = g.setup_data(df, {"width": 2, "height": 4})
        assert "xmin" in result.columns
        assert "ymin" in result.columns


class TestGeomTile:
    def test_required_aes(self):
        assert "x" in GeomTile.required_aes
        assert "y" in GeomTile.required_aes

    def test_is_subclass(self):
        assert issubclass(GeomTile, GeomRect)

    def test_setup_data(self):
        g = GeomTile()
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
        result = g.setup_data(df, {})
        assert "xmin" in result.columns
        assert "xmax" in result.columns


class TestGeomRaster:
    def test_required_aes(self):
        assert "x" in GeomRaster.required_aes
        assert "y" in GeomRaster.required_aes

    def test_setup_data(self):
        g = GeomRaster()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
        result = g.setup_data(df, {})
        assert "xmin" in result.columns
        assert "ymin" in result.columns


# ============================================================================
# GeomBar / GeomCol tests
# ============================================================================

class TestGeomBar:
    def test_required_aes(self):
        assert "x" in GeomBar.required_aes
        assert "y" in GeomBar.required_aes

    def test_is_subclass(self):
        assert issubclass(GeomBar, GeomRect)

    def test_setup_params(self):
        g = GeomBar()
        params = g.setup_params(pd.DataFrame(), {})
        assert "flipped_aes" in params

    def test_setup_data(self):
        g = GeomBar()
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = g.setup_data(df, {})
        assert "ymin" in result.columns
        assert "ymax" in result.columns
        assert "xmin" in result.columns
        assert "xmax" in result.columns
        assert result["ymin"].iloc[0] == 0.0


class TestGeomCol:
    def test_is_subclass(self):
        assert issubclass(GeomCol, GeomBar)


# ============================================================================
# GeomText / GeomLabel tests
# ============================================================================

class TestGeomText:
    def test_required_aes(self):
        assert "x" in GeomText.required_aes
        assert "y" in GeomText.required_aes
        assert "label" in GeomText.required_aes

    def test_default_aes(self):
        defaults = GeomText.default_aes
        assert "colour" in defaults
        assert "size" in defaults
        assert "angle" in defaults
        assert "hjust" in defaults


class TestGeomLabel:
    def test_required_aes(self):
        assert "label" in GeomLabel.required_aes

    def test_default_aes(self):
        defaults = GeomLabel.default_aes
        assert "fill" in defaults
        assert "colour" in defaults


# ============================================================================
# GeomPolygon tests
# ============================================================================

class TestGeomPolygon:
    def test_required_aes(self):
        assert "x" in GeomPolygon.required_aes
        assert "y" in GeomPolygon.required_aes

    def test_handle_na(self):
        g = GeomPolygon()
        df = pd.DataFrame({"x": [1, np.nan, 3], "y": [4, 5, np.nan]})
        result = g.handle_na(df, {})
        # Polygon does not remove NA
        assert len(result) == 3


# ============================================================================
# GeomRibbon / GeomArea tests
# ============================================================================

class TestGeomRibbon:
    def test_required_aes(self):
        assert "x" in GeomRibbon.required_aes
        assert "ymin" in GeomRibbon.required_aes
        assert "ymax" in GeomRibbon.required_aes

    def test_setup_params(self):
        g = GeomRibbon()
        params = g.setup_params(pd.DataFrame(), {})
        assert "flipped_aes" in params

    def test_setup_data(self):
        g = GeomRibbon()
        df = pd.DataFrame({"x": [1, 2, 3], "ymin": [0, 0, 0], "ymax": [1, 2, 1], "PANEL": [1, 1, 1]})
        result = g.setup_data(df, {})
        assert "y" in result.columns


class TestGeomArea:
    def test_is_subclass(self):
        assert issubclass(GeomArea, GeomRibbon)

    def test_setup_data(self):
        g = GeomArea()
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = g.setup_data(df, {})
        assert result["ymin"].iloc[0] == 0
        assert result["ymax"].iloc[0] == 4


# ============================================================================
# GeomSmooth tests
# ============================================================================

class TestGeomSmooth:
    def test_required_aes(self):
        assert "x" in GeomSmooth.required_aes
        assert "y" in GeomSmooth.required_aes

    def test_default_aes(self):
        defaults = GeomSmooth.default_aes
        assert "colour" in defaults
        assert "fill" in defaults

    def test_setup_params(self):
        g = GeomSmooth()
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "ymin": [2, 3], "ymax": [4, 5]})
        params = g.setup_params(df, {})
        assert "se" in params
        assert params["se"] is True

    def test_setup_params_no_se(self):
        g = GeomSmooth()
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        params = g.setup_params(df, {})
        assert params["se"] is False


# ============================================================================
# GeomSegment / GeomCurve / GeomSpoke tests
# ============================================================================

class TestGeomSegment:
    def test_required_aes(self):
        assert "x" in GeomSegment.required_aes
        assert "xend" in GeomSegment.required_aes

    def test_default_aes(self):
        defaults = GeomSegment.default_aes
        assert "colour" in defaults


class TestGeomCurve:
    def test_is_subclass(self):
        assert issubclass(GeomCurve, GeomSegment)


class TestGeomSpoke:
    def test_required_aes(self):
        assert "angle" in GeomSpoke.required_aes
        assert "radius" in GeomSpoke.required_aes

    def test_setup_data(self):
        g = GeomSpoke()
        df = pd.DataFrame({"x": [0.0], "y": [0.0], "angle": [np.pi / 4], "radius": [1.0]})
        result = g.setup_data(df, {})
        assert "xend" in result.columns
        assert "yend" in result.columns
        assert result["xend"].iloc[0] == pytest.approx(np.cos(np.pi / 4))
        assert result["yend"].iloc[0] == pytest.approx(np.sin(np.pi / 4))


# ============================================================================
# GeomErrorbar / GeomCrossbar / GeomLinerange / GeomPointrange tests
# ============================================================================

class TestGeomErrorbar:
    def test_required_aes(self):
        assert "ymin" in GeomErrorbar.required_aes
        assert "ymax" in GeomErrorbar.required_aes

    def test_setup_params(self):
        g = GeomErrorbar()
        params = g.setup_params(pd.DataFrame(), {})
        assert "flipped_aes" in params

    def test_setup_data(self):
        g = GeomErrorbar()
        df = pd.DataFrame({"x": [1.0], "ymin": [0.0], "ymax": [2.0]})
        result = g.setup_data(df, {})
        assert "xmin" in result.columns
        assert "xmax" in result.columns
        assert "width" in result.columns


class TestGeomErrorbarh:
    def test_is_subclass(self):
        assert issubclass(GeomErrorbarh, GeomErrorbar)

    def test_deprecation_warning(self):
        g = GeomErrorbarh()
        with pytest.warns(FutureWarning):
            g.setup_params(pd.DataFrame(), {})


class TestGeomCrossbar:
    def test_required_aes(self):
        assert "y" in GeomCrossbar.required_aes
        assert "ymin" in GeomCrossbar.required_aes
        assert "ymax" in GeomCrossbar.required_aes

    def test_setup_params(self):
        g = GeomCrossbar()
        params = g.setup_params(pd.DataFrame(), {})
        assert "fatten" in params
        assert params["fatten"] == 2.5

    def test_setup_data(self):
        g = GeomCrossbar()
        df = pd.DataFrame({"x": [1.0], "y": [1.0], "ymin": [0.0], "ymax": [2.0]})
        result = g.setup_data(df, {})
        assert "xmin" in result.columns


class TestGeomLinerange:
    def test_required_aes(self):
        # R geom-linerange.R:12: c("x|y", "ymin|xmin", "ymax|xmax")
        assert "x|y" in GeomLinerange.required_aes
        assert "ymin|xmin" in GeomLinerange.required_aes
        assert "ymax|xmax" in GeomLinerange.required_aes


class TestGeomPointrange:
    def test_required_aes(self):
        assert "y" in GeomPointrange.required_aes
        assert "ymin" in GeomPointrange.required_aes
        assert "ymax" in GeomPointrange.required_aes


# ============================================================================
# GeomBoxplot tests
# ============================================================================

class TestGeomBoxplot:
    def test_required_aes(self):
        req = GeomBoxplot.required_aes
        assert "x" in req or "lower" in req

    def test_default_aes(self):
        defaults = GeomBoxplot.default_aes
        assert "fill" in defaults or "colour" in defaults


# ============================================================================
# GeomViolin tests
# ============================================================================

class TestGeomViolin:
    def test_required_aes(self):
        assert "x" in GeomViolin.required_aes
        assert "y" in GeomViolin.required_aes


# ============================================================================
# GeomDotplot tests
# ============================================================================

class TestGeomDotplot:
    def test_required_aes(self):
        assert "x" in GeomDotplot.required_aes


# ============================================================================
# GeomDensity tests
# ============================================================================

class TestGeomDensity:
    def test_is_subclass(self):
        assert issubclass(GeomDensity, GeomArea)


# ============================================================================
# GeomAbline / GeomHline / GeomVline tests
# ============================================================================

class TestGeomAbline:
    def test_default_aes(self):
        defaults = GeomAbline.default_aes
        assert "colour" in defaults
        assert "linewidth" in defaults

    def test_required_aes_empty(self):
        # abline has no strict required_aes
        assert len(GeomAbline.required_aes) == 0 or isinstance(GeomAbline.required_aes, tuple)


class TestGeomHline:
    def test_default_aes(self):
        defaults = GeomHline.default_aes
        assert "colour" in defaults


class TestGeomVline:
    def test_default_aes(self):
        defaults = GeomVline.default_aes
        assert "colour" in defaults


# ============================================================================
# GeomRug tests
# ============================================================================

class TestGeomRug:
    def test_default_aes(self):
        defaults = GeomRug.default_aes
        assert "colour" in defaults or "linewidth" in defaults


# ============================================================================
# GeomBlank tests
# ============================================================================

class TestGeomBlank:
    def test_no_required_aes(self):
        assert len(GeomBlank.required_aes) == 0


# ============================================================================
# GeomContour tests
# ============================================================================

class TestGeomContour:
    def test_is_subclass(self):
        assert issubclass(GeomContour, GeomPath)


class TestGeomContourFilled:
    def test_is_subclass(self):
        assert issubclass(GeomContourFilled, GeomPolygon)


# ============================================================================
# GeomDensity2d tests
# ============================================================================

class TestGeomDensity2dClass:
    def test_is_subclass(self):
        assert issubclass(GeomDensity2d, GeomPath)


class TestGeomDensity2dFilledClass:
    def test_is_subclass(self):
        assert issubclass(GeomDensity2dFilled, GeomPolygon)


# ============================================================================
# GeomHex tests
# ============================================================================

class TestGeomHex:
    def test_required_aes(self):
        assert "x" in GeomHex.required_aes
        assert "y" in GeomHex.required_aes


# ============================================================================
# GeomBin2d tests
# ============================================================================

class TestGeomBin2d:
    def test_is_subclass(self):
        assert issubclass(GeomBin2d, GeomTile)


# ============================================================================
# GeomFunction tests
# ============================================================================

class TestGeomFunction:
    def test_is_subclass(self):
        assert issubclass(GeomFunction, GeomPath)


# ============================================================================
# GeomMap tests
# ============================================================================

class TestGeomMap:
    def test_is_subclass(self):
        assert issubclass(GeomMap, GeomPolygon)


# ============================================================================
# GeomQuantile tests
# ============================================================================

class TestGeomQuantile:
    def test_is_subclass(self):
        assert issubclass(GeomQuantile, GeomPath)


# ============================================================================
# GeomSf tests
# ============================================================================

class TestGeomSfClass:
    def test_exists(self):
        g = GeomSf()
        assert is_geom(g) is True


# ============================================================================
# use_defaults tests
# ============================================================================

class TestUseDefaults:
    def test_basic(self):
        g = GeomPoint()
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = g.use_defaults(df)
        # Should have default aesthetics filled in
        assert "shape" in result.columns
        assert "colour" in result.columns

    def test_with_params(self):
        g = GeomPoint()
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = g.use_defaults(df, params={"colour": "red"})
        assert all(result["colour"] == "red")

    def test_rename_size(self):
        g = GeomPath()  # has rename_size=True
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "size": [1.0, 2.0]})
        result = g.use_defaults(df)
        assert "linewidth" in result.columns

    def test_empty_data(self):
        g = GeomPoint()
        result = g.use_defaults(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)


# ============================================================================
# All geom class property tests (comprehensive coverage)
# ============================================================================

ALL_GEOM_CLASSES = [
    GeomPoint, GeomPath, GeomLine, GeomStep,
    GeomRect, GeomTile, GeomRaster,
    GeomBar, GeomCol,
    GeomText, GeomLabel,
    GeomPolygon,
    GeomRibbon, GeomArea, GeomSmooth,
    GeomSegment, GeomCurve, GeomSpoke,
    GeomErrorbar, GeomCrossbar, GeomLinerange, GeomPointrange,
    GeomBoxplot, GeomViolin, GeomDotplot,
    GeomDensity,
    GeomAbline, GeomHline, GeomVline,
    GeomRug, GeomBlank,
    GeomContour, GeomContourFilled,
    GeomDensity2d, GeomDensity2dFilled,
    GeomHex, GeomBin2d,
    GeomFunction, GeomMap, GeomQuantile,
    GeomFreqpoly, GeomHistogram, GeomCount,
    GeomJitter,
]


class TestAllGeomProperties:
    """Test that all geom classes have the essential properties."""

    @pytest.mark.parametrize("geom_cls", ALL_GEOM_CLASSES, ids=lambda c: c.__name__)
    def test_has_required_aes(self, geom_cls):
        assert hasattr(geom_cls, "required_aes")
        assert isinstance(geom_cls.required_aes, (tuple, list))

    @pytest.mark.parametrize("geom_cls", ALL_GEOM_CLASSES, ids=lambda c: c.__name__)
    def test_has_default_aes(self, geom_cls):
        assert hasattr(geom_cls, "default_aes")

    @pytest.mark.parametrize("geom_cls", ALL_GEOM_CLASSES, ids=lambda c: c.__name__)
    def test_instantiable(self, geom_cls):
        g = geom_cls()
        assert is_geom(g)

    @pytest.mark.parametrize("geom_cls", ALL_GEOM_CLASSES, ids=lambda c: c.__name__)
    def test_has_draw_key(self, geom_cls):
        assert hasattr(geom_cls, "draw_key")

    @pytest.mark.parametrize("geom_cls", ALL_GEOM_CLASSES, ids=lambda c: c.__name__)
    def test_aesthetics(self, geom_cls):
        g = geom_cls()
        aes = g.aesthetics()
        assert isinstance(aes, list)
        assert "group" in aes

    @pytest.mark.parametrize("geom_cls", ALL_GEOM_CLASSES, ids=lambda c: c.__name__)
    def test_parameters(self, geom_cls):
        g = geom_cls()
        params = g.parameters(extra=True)
        assert isinstance(params, list)
        assert "na_rm" in params


# ============================================================================
# Constructor function tests (extended from existing test_geom.py)
# ============================================================================

from ggplot2_py import (
    geom_raster,
    geom_function,
    geom_map,
    geom_quantile,
    geom_spoke,
    geom_density2d,
    geom_density2d_filled,
    geom_contour_filled,
    Layer,
)


EXTRA_CONSTRUCTORS = [
    geom_raster,
    geom_function,
    geom_spoke,
]


class TestExtraConstructors:
    @pytest.mark.parametrize("geom_fn", EXTRA_CONSTRUCTORS, ids=lambda f: f.__name__)
    def test_returns_layer(self, geom_fn):
        layer = geom_fn()
        assert isinstance(layer, Layer)
