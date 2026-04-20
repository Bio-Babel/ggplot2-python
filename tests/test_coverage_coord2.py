"""Additional tests for ggplot2_py.coord."""

import pytest
import numpy as np
import pandas as pd
import math

from ggplot2_py.coord import (
    Coord, CoordCartesian, CoordFixed, CoordFlip, CoordPolar,
    CoordRadial, CoordTransform, coord_munch, _polar_bbox,
)


class _FT:
    pass


class _FS:
    # Minimal stand-in for a continuous positional scale.  Real scales
    # inherit ``expand = waiver()`` and an ``is_discrete()`` method from
    # the ``Scale`` base class; mirror those here so ``default_expansion``
    # can resolve the expansion spec without bailing out.
    from ggplot2_py._compat import waiver as _waiver
    expand = _waiver()
    del _waiver

    def get_limits(self):
        return [0, 10]
    def dimension(self):
        return [0, 10]
    def is_discrete(self):
        return False


class TestCoordAbstract:
    def test_transform(self):
        with pytest.raises(Exception):
            Coord().transform(pd.DataFrame({"x": [1]}), {})

    def test_distance(self):
        with pytest.raises(Exception):
            Coord().distance(np.array([1]), np.array([1]), {})

    def test_backtransform_range(self):
        with pytest.raises(Exception):
            Coord().backtransform_range({})

    def test_range(self):
        with pytest.raises(Exception):
            Coord().range({})

    def test_render_bg(self):
        with pytest.raises(Exception):
            Coord().render_bg({}, _FT())


class TestCoordCartesianRender:
    def test_draw_panel(self):
        from grid_py import null_grob
        pp = {"x_range": [0, 10], "y_range": [0, 10]}
        assert CoordCartesian().draw_panel([null_grob()], pp, _FT()) is not None

    def test_render_fg(self):
        assert CoordCartesian().render_fg({}, _FT()) is not None

    def test_render_axis_h(self):
        r = CoordCartesian().render_axis_h({}, _FT())
        assert "top" in r

    def test_render_axis_v(self):
        r = CoordCartesian().render_axis_v({}, _FT())
        assert "left" in r


class TestCoordFlipMethods:
    def test_transform(self):
        data = pd.DataFrame({"x": [1.0], "y": [3.0]})
        assert isinstance(CoordFlip().transform(data, {"x_range": [0, 10], "y_range": [0, 10]}), pd.DataFrame)

    def test_backtransform_range(self):
        r = CoordFlip().backtransform_range({"x_range": [0, 10], "y_range": [0, 5]})
        assert "x" in r

    def test_range(self):
        r = CoordFlip().range({"x_range": [0, 10], "y_range": [0, 5]})
        assert "x" in r

    def test_setup_panel_params(self):
        assert CoordFlip().setup_panel_params(_FS(), _FS(), params={"expand": [True]*4}) is not None


class TestCoordPolarMethods:
    def test_transform(self):
        data = pd.DataFrame({"x": [0.5], "y": [0.5]})
        assert "x" in CoordPolar(theta="x").transform(data, {"theta.range": [0, 1], "r.range": [0, 1]}).columns

    def test_distance_theta_x(self):
        assert CoordPolar(theta="x").distance(np.array([0.1, 0.5]), np.array([0.5, 0.8]),
            {"theta.range": [0, 1], "r.range": [0, 1]}) is not None

    def test_distance_theta_y(self):
        assert CoordPolar(theta="y").distance(np.array([0.1, 0.5]), np.array([0.5, 0.8]),
            {"theta.range": [0, 1], "r.range": [0, 1]}) is not None

    def test_render_bg(self):
        assert CoordPolar().render_bg({"theta.range": [0, 1]}, _FT()) is not None

    def test_render_axis_h(self):
        assert "top" in CoordPolar().render_axis_h({}, _FT())

    def test_render_axis_v(self):
        assert "left" in CoordPolar().render_axis_v({}, _FT())

    def test_setup_panel_params(self):
        r = CoordPolar().setup_panel_params(_FS(), _FS())
        assert r is not None


class TestCoordRadialMethods:
    def test_transform(self):
        pp = {"theta.range": [0, 1], "r.range": [0, 1],
              "bbox": {"x": [0, 1], "y": [0, 1]},
              "arc": (0, 2 * math.pi), "inner_radius": (0, 0.4)}
        assert "x" in CoordRadial().transform(pd.DataFrame({"x": [0.5], "y": [0.5]}), pp).columns

    def test_setup_panel_params(self):
        assert "theta.range" in CoordRadial().setup_panel_params(_FS(), _FS())

    def test_setup_panel_params_theta_y(self):
        assert "theta.range" in CoordRadial(theta="y").setup_panel_params(_FS(), _FS())

    def test_render_bg(self):
        assert CoordRadial().render_bg({}, _FT()) is not None

    def test_render_axis_h(self):
        assert "top" in CoordRadial().render_axis_h({}, _FT())

    def test_render_axis_v(self):
        assert "left" in CoordRadial().render_axis_v({}, _FT())

    def test_labels_theta_y(self):
        assert CoordRadial(theta="y").labels({"x": "X", "y": "Y"}, {}) is not None


class TestCoordTransformMethods:
    def test_transform_no_trans(self):
        data = pd.DataFrame({"x": [1.0, 5.0], "y": [2.0, 8.0]})
        assert isinstance(CoordTransform().transform(data, {"x.range": [0, 10], "y.range": [0, 10]}), pd.DataFrame)

    def test_setup_panel_params(self):
        r = CoordTransform().setup_panel_params(_FS(), _FS())
        assert "x.range" in r or "x_range" in r

    def test_setup_panel_params_with_limits(self):
        c = CoordTransform(limits={"x": [0, 5], "y": [0, 5]})
        assert c.setup_panel_params(_FS(), _FS()) is not None

    def test_render_bg(self):
        assert CoordTransform().render_bg({}, _FT()) is not None

    def test_render_axis_h(self):
        assert "top" in CoordTransform().render_axis_h({}, _FT())


class TestCoordMunch:
    def test_basic(self):
        data = pd.DataFrame({"x": [0.0, 5.0, 10.0], "y": [0.0, 5.0, 10.0]})
        result = coord_munch(CoordCartesian(), data, {"x_range": [0, 10], "y_range": [0, 10]}, n=5)
        assert isinstance(result, pd.DataFrame)


class TestPolarBbox:
    def test_full_circle(self):
        assert _polar_bbox((0, 2 * math.pi))["x"] == [0.0, 1.0]

    def test_partial_arc(self):
        r = _polar_bbox((0, math.pi / 2))
        assert "x" in r
