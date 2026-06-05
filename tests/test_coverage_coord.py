"""Extended coverage tests for ggplot2_py.coord."""

import math
import pytest
import numpy as np
import pandas as pd

from ggplot2_py.coord import (
    Coord,
    CoordCartesian,
    CoordFixed,
    CoordFlip,
    CoordPolar,
    CoordRadial,
    CoordTransform,
    CoordTrans,
    coord_cartesian,
    coord_equal,
    coord_fixed,
    coord_flip,
    coord_polar,
    coord_radial,
    coord_trans,
    coord_munch,
    is_coord,
    _rescale,
    _squish_infinite,
    _dist_euclidean,
    _dist_polar,
    _theta_rescale,
    _theta_rescale_no_clip,
    _r_rescale,
    _parse_coord_expand,
    _transform_position,
    _flip_axis_labels,
    _polar_bbox,
    _in_arc,
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

class TestRescale:
    def test_basic(self):
        x = np.array([0, 5, 10])
        result = _rescale(x, to=(0, 1))
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])

    def test_custom_range(self):
        x = np.array([0, 10])
        result = _rescale(x, to=(0, 100), from_=(0, 10))
        np.testing.assert_array_almost_equal(result, [0, 100])

    def test_zero_range(self):
        x = np.array([5, 5])
        result = _rescale(x, to=(0, 1))
        np.testing.assert_array_almost_equal(result, [0.5, 0.5])


class TestSquishInfinite:
    def test_neg_inf(self):
        x = np.array([-np.inf, 0, np.inf])
        result = _squish_infinite(x, range_=(0, 1))
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 1.0])

    def test_no_range(self):
        x = np.array([-np.inf, np.inf])
        result = _squish_infinite(x)
        np.testing.assert_array_almost_equal(result, [0.0, 1.0])


class TestDistEuclidean:
    def test_basic(self):
        x = np.array([0, 3])
        y = np.array([0, 4])
        result = _dist_euclidean(x, y)
        np.testing.assert_array_almost_equal(result, [5.0])

    def test_single_point(self):
        result = _dist_euclidean(np.array([1]), np.array([2]))
        assert result[0] == 0.0

    def test_scalar(self):
        result = _dist_euclidean(np.array(1.0), np.array(2.0))
        assert result[0] == 0.0


class TestDistPolar:
    def test_basic(self):
        # R gold standard: dist_polar is the polar arc-length metric
        # (spiral/circular/ray), NOT the straight-line law of cosines.
        # R: dist_polar(c(1, 1), c(0, pi)) == 0.9286289 (a circular arc,
        # r constant -> r1 * (t2 - t1) / max_dist).
        r = np.array([1.0, 1.0])
        theta = np.array([0.0, math.pi])
        result = _dist_polar(r, theta)
        assert result[0] == pytest.approx(0.9286289, abs=1e-6)

    def test_single_point(self):
        # R: dist_polar returns length n-1, so a single point -> length 0.
        result = _dist_polar(np.array([1.0]), np.array([0.0]))
        assert len(result) == 0


class TestThetaRescale:
    def test_basic(self):
        x = np.array([0, 0.5, 1.0])
        result = _theta_rescale(x, (0, 1))
        assert len(result) == 3

    def test_no_clip(self):
        x = np.array([0.5])
        result = _theta_rescale_no_clip(x, (0, 1))
        assert len(result) == 1


class TestRRescale:
    def test_basic(self):
        x = np.array([0, 0.5, 1.0])
        result = _r_rescale(x, (0, 1))
        assert len(result) == 3
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(0.4)


class TestParseCoordExpand:
    def test_bool_true(self):
        result = _parse_coord_expand(True)
        assert result == [True, True, True, True]

    def test_bool_false(self):
        result = _parse_coord_expand(False)
        assert result == [False, False, False, False]

    def test_list(self):
        result = _parse_coord_expand([True, False])
        assert result == [True, False, False, False]

    def test_list_four(self):
        result = _parse_coord_expand([True, False, True, False])
        assert result == [True, False, True, False]

    def test_none(self):
        result = _parse_coord_expand(None)
        assert result == [True, True, True, True]


class TestTransformPosition:
    def test_transforms_both(self):
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        result = _transform_position(df, lambda v: v * 2, lambda v: v * 3)
        assert result["x"].iloc[0] == 2.0
        assert result["y"].iloc[0] == 6.0


class TestFlipAxisLabels:
    def test_dict(self):
        d = {"x_range": [0, 1], "y_range": [2, 3]}
        result = _flip_axis_labels(d)
        assert "y_range" in result
        assert "x_range" in result

    def test_dataframe(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        result = _flip_axis_labels(df)
        assert "y" in result.columns
        assert "x" in result.columns

    def test_other(self):
        result = _flip_axis_labels(42)
        assert result == 42


class TestPolarBbox:
    def test_full_circle(self):
        result = _polar_bbox((0, 2 * math.pi))
        assert result == {"x": [0.0, 1.0], "y": [0.0, 1.0]}

    def test_half_circle(self):
        result = _polar_bbox((0, math.pi))
        assert isinstance(result, dict)
        assert "x" in result and "y" in result


class TestInArc:
    def test_full_circle(self):
        theta = np.array([0, math.pi / 2, math.pi])
        result = _in_arc(theta, (0, 2 * math.pi))
        assert all(result)

    def test_partial(self):
        theta = np.array([0, math.pi])
        result = _in_arc(theta, (0, math.pi / 2))
        assert result[0] is np.True_


# ---------------------------------------------------------------------------
# Base Coord
# ---------------------------------------------------------------------------

class TestCoordBase:
    def test_default_is_not_linear(self):
        c = Coord()
        assert c.is_linear() is False

    def test_default_is_not_free(self):
        c = Coord()
        assert c.is_free() is False

    def test_default_aspect(self):
        c = Coord()
        assert c.aspect({}) is None

    def test_setup_params(self):
        c = Coord()
        params = c.setup_params(None)
        assert "expand" in params

    def test_setup_data(self):
        c = Coord()
        data = [pd.DataFrame({"x": [1]})]
        assert c.setup_data(data) is data

    def test_setup_layout(self):
        c = Coord()
        layout = pd.DataFrame({
            "PANEL": [1],
            "ROW": [1],
            "COL": [1],
            "SCALE_X": [1],
            "SCALE_Y": [1],
        })
        result = c.setup_layout(layout)
        assert "COORD" in result.columns

    def test_modify_scales(self):
        c = Coord()
        c.modify_scales([], [])  # should not raise

    def test_setup_panel_params(self):
        c = Coord()
        result = c.setup_panel_params(None, None)
        assert result == {}

    def test_setup_panel_guides(self):
        c = Coord()
        pp = {}
        result = c.setup_panel_guides(pp, "mock_guides")
        assert result["guides"] == "mock_guides"

    def test_train_panel_guides(self):
        c = Coord()
        pp = {"some": "data"}
        result = c.train_panel_guides(pp, [])
        assert result is pp

    def test_labels_passthrough(self):
        c = Coord()
        labels = {"x": "X", "y": "Y"}
        assert c.labels(labels, {}) == labels


# ---------------------------------------------------------------------------
# CoordCartesian
# ---------------------------------------------------------------------------

class TestCoordCartesianExtended:
    def test_is_linear(self):
        c = coord_cartesian()
        assert c.is_linear() is True

    def test_is_free_no_ratio(self):
        c = coord_cartesian()
        assert c.is_free() is True

    def test_is_free_with_ratio(self):
        c = coord_cartesian(ratio=1.0)
        assert c.is_free() is False

    def test_aspect_none(self):
        c = coord_cartesian()
        assert c.aspect({}) is None

    def test_aspect_with_ratio(self):
        c = coord_cartesian(ratio=2.0)
        ranges = {"x.range": [0, 10], "y.range": [0, 20]}
        aspect = c.aspect(ranges)
        assert aspect == pytest.approx(4.0)

    def test_distance(self):
        c = coord_cartesian()
        pp = {"x_range": [0, 10], "y_range": [0, 10]}
        d = c.distance(np.array([0, 3]), np.array([0, 4]), pp)
        assert len(d) == 1
        assert d[0] > 0

    def test_distance_zero_range(self):
        c = coord_cartesian()
        pp = {"x_range": [5, 5], "y_range": [5, 5]}
        d = c.distance(np.array([0, 1]), np.array([0, 1]), pp)
        assert len(d) == 1

    def test_range(self):
        c = coord_cartesian()
        pp = {"x_range": [0, 10], "y_range": [0, 20]}
        r = c.range(pp)
        assert r == {"x": [0, 10], "y": [0, 20]}

    def test_backtransform_range(self):
        c = coord_cartesian()
        pp = {"x_range": [0, 10], "y_range": [0, 20]}
        r = c.backtransform_range(pp)
        assert r == {"x": [0, 10], "y": [0, 20]}

    def test_transform(self):
        c = coord_cartesian()
        df = pd.DataFrame({"x": [0.0, 5.0, 10.0], "y": [0.0, 10.0, 20.0]})
        pp = {"x_range": [0, 10], "y_range": [0, 20]}
        result = c.transform(df, pp)
        np.testing.assert_array_almost_equal(result["x"].values, [0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(result["y"].values, [0.0, 0.5, 1.0])

    def test_transform_with_reverse_x(self):
        c = coord_cartesian(reverse="x")
        df = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 20.0]})
        pp = {"x_range": [0, 10], "y_range": [0, 20], "reverse": "x"}
        result = c.transform(df, pp)
        assert result["x"].iloc[0] == pytest.approx(1.0)
        assert result["x"].iloc[1] == pytest.approx(0.0)

    def test_transform_with_reverse_y(self):
        c = coord_cartesian(reverse="y")
        df = pd.DataFrame({"x": [5.0], "y": [0.0]})
        pp = {"x_range": [0, 10], "y_range": [0, 20], "reverse": "y"}
        result = c.transform(df, pp)
        assert result["y"].iloc[0] == pytest.approx(1.0)

    def test_transform_with_reverse_xy(self):
        c = coord_cartesian(reverse="xy")
        df = pd.DataFrame({"x": [0.0], "y": [0.0]})
        pp = {"x_range": [0, 10], "y_range": [0, 20], "reverse": "xy"}
        result = c.transform(df, pp)
        assert result["x"].iloc[0] == pytest.approx(1.0)
        assert result["y"].iloc[0] == pytest.approx(1.0)

    def test_transform_squishes_inf(self):
        c = coord_cartesian()
        df = pd.DataFrame({"x": [-np.inf, np.inf], "y": [0.0, 20.0]})
        pp = {"x_range": [0, 10], "y_range": [0, 20]}
        result = c.transform(df, pp)
        assert np.isfinite(result["x"].values).all()

    def test_setup_panel_params_no_scales(self):
        c = coord_cartesian()
        result = c.setup_panel_params(None, None)
        assert result["x_range"] == [0, 1]
        assert result["y_range"] == [0, 1]

    def test_setup_panel_params_with_limits(self):
        c = coord_cartesian(xlim=[0, 5], ylim=[0, 15])
        result = c.setup_panel_params(None, None)
        assert result["x_range"] == [0, 5]
        assert result["y_range"] == [0, 15]

    def test_constructor_params(self):
        c = coord_cartesian(xlim=[0, 10], ylim=[0, 20], clip="off", reverse="x", default=True)
        assert c.clip == "off"
        assert c.reverse == "x"
        assert c.default is True


# ---------------------------------------------------------------------------
# CoordFixed
# ---------------------------------------------------------------------------

class TestCoordFixedExtended:
    def test_constructor(self):
        c = coord_fixed(ratio=2.0)
        assert isinstance(c, CoordFixed)
        assert c.ratio == 2.0

    def test_is_linear(self):
        c = coord_fixed()
        assert c.is_linear() is True

    def test_is_not_free(self):
        c = coord_fixed()
        assert c.is_free() is False

    def test_aspect(self):
        c = coord_fixed(ratio=1.0)
        ranges = {"y.range": [0, 10], "x.range": [0, 10]}
        assert c.aspect(ranges) == pytest.approx(1.0)

    def test_coord_equal_alias(self):
        assert coord_equal is coord_fixed


# ---------------------------------------------------------------------------
# CoordFlip
# ---------------------------------------------------------------------------

class TestCoordFlipExtended:
    def test_constructor(self):
        c = coord_flip()
        assert isinstance(c, CoordFlip)

    def test_range_flipped(self):
        c = coord_flip()
        pp = {"x_range": [0, 10], "y_range": [0, 20]}
        r = c.range(pp)
        assert r["x"] == [0, 20]
        assert r["y"] == [0, 10]

    def test_backtransform_range(self):
        c = coord_flip()
        pp = {"x_range": [0, 10], "y_range": [0, 20]}
        r = c.backtransform_range(pp)
        assert "x" in r and "y" in r

    def test_transform(self):
        c = coord_flip()
        df = pd.DataFrame({"x": [5.0], "y": [10.0]})
        pp = {"x_range": [0, 10], "y_range": [0, 20]}
        result = c.transform(df, pp)
        assert len(result) == 1

    def test_labels_flipped(self):
        c = coord_flip()
        labels = {"x": "X label", "y": "Y label"}
        result = c.labels(labels, {})
        assert result.get("y") == "X label" or result.get("x") == "Y label"

    def test_setup_layout_swaps_scales(self):
        c = coord_flip()
        layout = pd.DataFrame({
            "PANEL": [1],
            "ROW": [1],
            "COL": [1],
            "SCALE_X": [1],
            "SCALE_Y": [2],
        })
        result = c.setup_layout(layout)
        assert result["SCALE_X"].iloc[0] == 2
        assert result["SCALE_Y"].iloc[0] == 1

    def test_with_limits(self):
        c = coord_flip(xlim=[0, 5], ylim=[0, 15])
        assert isinstance(c, CoordFlip)


# ---------------------------------------------------------------------------
# CoordPolar
# ---------------------------------------------------------------------------

class TestCoordPolarExtended:
    def test_constructor(self):
        c = coord_polar()
        assert isinstance(c, CoordPolar)
        assert c.theta == "x"
        assert c.r == "y"

    def test_constructor_theta_y(self):
        c = coord_polar(theta="y")
        assert c.theta == "y"
        assert c.r == "x"

    def test_aspect(self):
        c = coord_polar()
        assert c.aspect({}) == 1.0

    def test_is_free(self):
        c = coord_polar()
        assert c.is_free() is True

    def test_range(self):
        c = coord_polar()
        pp = {"theta.range": [0, 6], "r.range": [0, 10]}
        r = c.range(pp)
        assert r["x"] == [0, 6]
        assert r["y"] == [0, 10]

    def test_backtransform_range(self):
        c = coord_polar()
        pp = {"theta.range": [0, 6], "r.range": [0, 10]}
        r = c.backtransform_range(pp)
        assert r == c.range(pp)

    def test_distance(self):
        c = coord_polar()
        pp = {"r.range": [0, 1], "theta.range": [0, 2 * math.pi]}
        d = c.distance(np.array([0, 0.5]), np.array([0.5, 0.5]), pp)
        assert len(d) == 1

    def test_transform(self):
        c = coord_polar()
        df = pd.DataFrame({"x": [0.0, 0.5], "y": [0.5, 1.0]})
        pp = {"theta.range": [0, 1], "r.range": [0, 1]}
        result = c.transform(df, pp)
        assert "x" in result.columns
        assert "y" in result.columns

    def test_transform_theta_y(self):
        c = coord_polar(theta="y")
        df = pd.DataFrame({"x": [0.5], "y": [0.5]})
        pp = {"theta.range": [0, 1], "r.range": [0, 1]}
        result = c.transform(df, pp)
        assert len(result) == 1

    def test_setup_panel_params(self):
        c = coord_polar()
        result = c.setup_panel_params(None, None)
        assert "theta.range" in result
        assert "r.range" in result

    def test_setup_panel_guides(self):
        c = coord_polar()
        pp = {}
        result = c.setup_panel_guides(pp, None)
        assert result is pp

    def test_train_panel_guides(self):
        c = coord_polar()
        pp = {}
        result = c.train_panel_guides(pp, [])
        assert result is pp

    def test_labels_default(self):
        c = coord_polar()
        labels = {"x": "X", "y": "Y"}
        result = c.labels(labels, {})
        assert result == labels

    def test_labels_theta_y(self):
        c = coord_polar(theta="y")
        labels = {"x": "X", "y": "Y"}
        result = c.labels(labels, {})
        assert result["x"] == "Y"
        assert result["y"] == "X"

    def test_with_start_and_direction(self):
        c = coord_polar(start=math.pi / 4, direction=-1)
        assert c.start == math.pi / 4
        assert c.direction == -1


# ---------------------------------------------------------------------------
# CoordRadial
# ---------------------------------------------------------------------------

class TestCoordRadialExtended:
    def test_constructor(self):
        c = coord_radial()
        assert isinstance(c, CoordRadial)
        assert c.theta == "x"
        assert c.r == "y"

    def test_constructor_theta_y(self):
        c = coord_radial(theta="y")
        assert c.theta == "y"
        assert c.r == "x"

    def test_aspect(self):
        c = coord_radial()
        details = {"bbox": {"x": [0, 1], "y": [0, 1]}}
        assert c.aspect(details) == pytest.approx(1.0)

    def test_is_free(self):
        c = coord_radial()
        assert c.is_free() is True

    def test_range(self):
        c = coord_radial()
        pp = {"theta.range": [0, 6], "r.range": [0, 10]}
        r = c.range(pp)
        assert r["x"] == [0, 6]

    def test_distance(self):
        c = coord_radial()
        pp = {"r.range": [0, 1], "theta.range": [0, 1]}
        d = c.distance(np.array([0, 0.5]), np.array([0.5, 0.5]), pp)
        assert len(d) == 1

    def test_distance_theta_y(self):
        c = coord_radial(theta="y")
        pp = {"r.range": [0, 1], "theta.range": [0, 1]}
        d = c.distance(np.array([0.5, 0.5]), np.array([0, 0.5]), pp)
        assert len(d) == 1

    def test_setup_panel_params(self):
        c = coord_radial()
        result = c.setup_panel_params(None, None)
        assert "theta.range" in result
        assert "r.range" in result
        assert "bbox" in result

    def test_transform(self):
        c = coord_radial()
        df = pd.DataFrame({"x": [0.5], "y": [0.5]})
        pp = {"theta.range": [0, 1], "r.range": [0, 1],
              "bbox": {"x": [0, 1], "y": [0, 1]},
              "arc": (0, 2 * math.pi),
              "inner_radius": (0.0, 0.4)}
        result = c.transform(df, pp)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# CoordTransform
# ---------------------------------------------------------------------------

class TestCoordTransformExtended:
    def test_alias(self):
        assert CoordTrans is CoordTransform

    def test_constructor(self):
        c = coord_trans()
        assert isinstance(c, CoordTransform)

    def test_is_free(self):
        c = coord_trans()
        assert c.is_free() is True

    def test_range(self):
        c = coord_trans()
        pp = {"x.range": [0, 10], "y.range": [0, 20]}
        r = c.range(pp)
        assert r == {"x": [0, 10], "y": [0, 20]}

    def test_backtransform_range_no_trans(self):
        c = coord_trans()
        pp = {"x.range": [0, 10], "y.range": [0, 20]}
        r = c.backtransform_range(pp)
        assert r == {"x": [0, 10], "y": [0, 20]}

    def test_distance(self):
        c = coord_trans()
        pp = {"x.range": [0, 10], "y.range": [0, 10]}
        d = c.distance(np.array([0, 3]), np.array([0, 4]), pp)
        assert len(d) == 1

    def test_transform_no_trans(self):
        c = coord_trans()
        df = pd.DataFrame({"x": [0.0, 5.0, 10.0], "y": [0.0, 10.0, 20.0]})
        pp = {"x.range": [0, 10], "y.range": [0, 20]}
        result = c.transform(df, pp)
        np.testing.assert_array_almost_equal(result["x"].values, [0.0, 0.5, 1.0])

    def test_setup_panel_params(self):
        c = coord_trans()
        result = c.setup_panel_params(None, None)
        assert "x.range" in result
        assert "y.range" in result


# ---------------------------------------------------------------------------
# coord_munch
# ---------------------------------------------------------------------------

class TestCoordMunch:
    def test_linear_coord(self):
        c = coord_cartesian()
        df = pd.DataFrame({"x": [0.0, 5.0, 10.0], "y": [0.0, 10.0, 20.0]})
        pp = {"x_range": [0, 10], "y_range": [0, 20]}
        result = coord_munch(c, df, pp)
        assert len(result) == 3

    def test_nonlinear_single_point(self):
        c = coord_polar()
        df = pd.DataFrame({"x": [0.5], "y": [0.5]})
        pp = {"theta.range": [0, 1], "r.range": [0, 1]}
        result = coord_munch(c, df, pp)
        assert len(result) == 1

    def test_nonlinear_two_points(self):
        c = coord_polar()
        df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.5, 0.5]})
        pp = {"theta.range": [0, 1], "r.range": [0, 1]}
        result = coord_munch(c, df, pp)
        assert len(result) >= 2

    def test_nonlinear_same_points(self):
        c = coord_polar()
        df = pd.DataFrame({"x": [0.5, 0.5], "y": [0.5, 0.5]})
        pp = {"theta.range": [0, 1], "r.range": [0, 1]}
        result = coord_munch(c, df, pp)
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# Constructor functions
# ---------------------------------------------------------------------------

class TestConstructorFunctions:
    def test_coord_cartesian_defaults(self):
        c = coord_cartesian()
        assert c.limits == {"x": None, "y": None}

    def test_coord_cartesian_with_limits(self):
        c = coord_cartesian(xlim=[0, 10], ylim=[0, 20])
        assert c.limits["x"] == [0, 10]
        assert c.limits["y"] == [0, 20]

    def test_coord_cartesian_clip_off(self):
        c = coord_cartesian(clip="off")
        assert c.clip == "off"

    def test_coord_flip_with_limits(self):
        c = coord_flip(xlim=[0, 5], ylim=[0, 15])
        assert isinstance(c, CoordFlip)

    def test_coord_polar_defaults(self):
        c = coord_polar()
        assert c.theta == "x"
        assert c.direction == 1
        assert c.start == 0.0

    def test_coord_polar_custom(self):
        c = coord_polar(theta="y", start=math.pi, direction=-1)
        assert c.theta == "y"
        assert c.start == math.pi
        assert c.direction == -1

    def test_coord_fixed_default_ratio(self):
        c = coord_fixed()
        assert c.ratio == 1.0

    def test_coord_trans_no_transforms(self):
        c = coord_trans()
        assert isinstance(c, CoordTransform)


# ---------------------------------------------------------------------------
# is_coord
# ---------------------------------------------------------------------------

class TestIsCoordExtended:
    def test_coord_transform(self):
        assert is_coord(coord_trans()) is True

    def test_coord_radial(self):
        assert is_coord(coord_radial()) is True

    def test_not_coord(self):
        assert is_coord("cartesian") is False

    def test_none(self):
        assert is_coord(None) is False
