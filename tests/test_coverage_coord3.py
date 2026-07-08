"""Targeted coverage tests for ggplot2_py.coord – missing lines."""

import pytest
import numpy as np
import pandas as pd
import math

from ggplot2_py.coord import (
    Coord,
    CoordCartesian,
    CoordFlip,
    CoordFixed,
    CoordPolar,
    CoordRadial,
    CoordTransform,
    coord_cartesian,
    coord_flip,
    coord_fixed,
    coord_polar,
    coord_radial,
    coord_trans,
    coord_munch,
)


def _find_rect_children(grob):
    """Recursively collect ``rect``-class grobs under *grob* (test helper)."""
    out = []
    cls = getattr(grob, "_grid_class", None)
    if cls == "rect":
        out.append(grob)
    for child in (getattr(grob, "get_children", None) or (lambda: []))():
        out.extend(_find_rect_children(child))
    return out


# ===========================================================================
# Base Coord abstract methods (lines 352, 371, 385, 399)
# ===========================================================================

class TestCoordBase:
    def test_transform_raises(self):
        with pytest.raises(Exception):
            Coord().transform(pd.DataFrame(), {})

    def test_distance_raises(self):
        with pytest.raises(Exception):
            Coord().distance(np.array([0]), np.array([0]), {})

    def test_backtransform_range_raises(self):
        with pytest.raises(Exception):
            Coord().backtransform_range({})

    def test_range_raises(self):
        with pytest.raises(Exception):
            Coord().range({})

    def test_render_fg(self):
        result = Coord().render_fg({}, None)
        assert result is not None

    def test_render_axis_h(self):
        result = Coord().render_axis_h({}, None)
        assert "top" in result and "bottom" in result

    def test_render_axis_v(self):
        result = Coord().render_axis_v({}, None)
        assert "left" in result and "right" in result

    def test_draw_panel(self):
        """Base Coord.draw_panel (lines 416-420)."""
        from grid_py import null_grob
        c = CoordCartesian()
        pp = {"x_range": [0, 10], "y_range": [0, 10], "x.range": [0, 10], "y.range": [0, 10]}
        result = c.draw_panel(null_grob(), pp, None)
        assert result is not None


# ===========================================================================
# CoordCartesian: render_bg, render_fg, render_axis_h/v (lines 694-703)
# ===========================================================================

class TestCoordCartesian:
    def test_render_fg(self):
        c = CoordCartesian()
        pp = {"x_range": [0, 10], "y_range": [0, 10], "x.range": [0, 10], "y.range": [0, 10]}
        result = c.render_fg(pp, None)
        assert result is not None

    def test_render_fg_draws_panel_border_when_visible(self):
        # R Coord$render_fg (coord-.R:532-534): element_render(theme,
        # "panel.border", fill = NA) — theme_bw()'s border must be a
        # real (non-null) grob, and its fill must be transparent (NA),
        # not the element's own opaque fill (which would paint over
        # every data point drawn under it). Regression test for a bug
        # where `fill=None` collided with "argument omitted" in
        # _grob_from_rect and fell through to element.fill='white'.
        import ggplot2_py as gg
        c = CoordCartesian()
        theme = gg.theme_bw()
        border = c.render_fg({}, theme)
        assert getattr(border, "_grid_class", None) != "null"
        assert border.gp.get("fill") in (None, [None])

    def test_render_fg_blank_for_default_theme(self):
        # theme_grey()'s panel.border is element_blank() — must stay a
        # null grob (no visual change for the default theme).
        import ggplot2_py as gg
        c = CoordCartesian()
        border = c.render_fg({}, gg.theme_grey())
        assert getattr(border, "_grid_class", None) == "null"

    def test_render_axis_h(self):
        c = CoordCartesian()
        pp = {"x_range": [0, 10], "y_range": [0, 10], "x.range": [0, 10], "y.range": [0, 10]}
        result = c.render_axis_h(pp, None)
        assert isinstance(result, dict)

    def test_render_axis_v(self):
        c = CoordCartesian()
        pp = {"x_range": [0, 10], "y_range": [0, 10], "x.range": [0, 10], "y.range": [0, 10]}
        result = c.render_axis_v(pp, None)
        assert isinstance(result, dict)


# ===========================================================================
# CoordPolar: transform, distance, range (lines 847-848, 876, 880, 887-890)
# ===========================================================================

class TestCoordPolar:
    def test_render_fg_border_is_transparent_not_opaque(self):
        # Regression: CoordPolar$render_fg's panel.border draw used
        # fill=None, which _grob_from_rect treats as "argument
        # omitted" and falls through to the element's own (opaque
        # white) fill under theme_bw() -- hiding every wedge/bar drawn
        # underneath. Must be fill=NA (transparent), matching R's
        # element_render(theme, "panel.border", fill = NA).
        import ggplot2_py as gg
        c = CoordPolar()
        pp = {"theta.range": [0, 1], "r.range": [0, 1]}
        fg = c.render_fg(pp, gg.theme_bw())
        found = _find_rect_children(fg)
        assert found, "expected a panel.border rect grob in render_fg output"
        for rect in found:
            assert rect.gp.get("fill") in (None, [None])

    def test_transform(self):
        c = CoordPolar()
        pp = {"theta.range": [0, 1], "r.range": [0, 1], "arc": (0, 2 * math.pi),
              "bbox": {"x": [0, 1], "y": [0, 1]}, "inner_radius": (0, 0.4)}
        df = pd.DataFrame({"x": [0.5], "y": [0.5]})
        result = c.transform(df, pp)
        assert isinstance(result, pd.DataFrame)

    def test_distance(self):
        c = CoordPolar()
        pp = {"theta.range": [0, 1], "r.range": [0, 1]}
        result = c.distance(np.array([0.0, 0.5]), np.array([0.5, 0.5]), pp)
        assert len(result) > 0

    def test_range(self):
        c = CoordPolar()
        pp = {"theta.range": [0, 1], "r.range": [0, 10]}
        result = c.range(pp)
        assert isinstance(result, dict)

    def test_backtransform_range(self):
        c = CoordPolar()
        pp = {"theta.range": [0, 1], "r.range": [0, 10]}
        result = c.backtransform_range(pp)
        assert isinstance(result, dict)

    def test_setup_panel_params_no_scales(self):
        c = CoordPolar()
        pp = c.setup_panel_params(None, None)
        assert "theta.range" in pp
        assert "r.range" in pp


# ===========================================================================
# CoordRadial: transform, distance, range (lines 1032, 1054-1056, 1060, 1064, 1068, 1072)
# ===========================================================================

class TestCoordRadial:
    def test_transform(self):
        c = coord_radial()
        pp = {"theta.range": [0, 1], "r.range": [0, 1], "arc": (0, 2 * math.pi),
              "bbox": {"x": [0, 1], "y": [0, 1]}, "inner_radius": (0, 0.4)}
        df = pd.DataFrame({"x": [0.5], "y": [0.5]})
        result = c.transform(df, pp)
        assert isinstance(result, pd.DataFrame)

    def test_distance(self):
        c = coord_radial()
        pp = {"theta.range": [0, 1], "r.range": [0, 1]}
        result = c.distance(np.array([0.0, 0.5]), np.array([0.5, 0.5]), pp)
        assert len(result) > 0

    def test_range(self):
        c = coord_radial()
        pp = {"theta.range": [0, 6.28], "r.range": [0, 10]}
        result = c.range(pp)
        assert isinstance(result, dict)

    def test_backtransform_range(self):
        c = coord_radial()
        pp = {"theta.range": [0, 6.28], "r.range": [0, 10]}
        result = c.backtransform_range(pp)
        assert isinstance(result, dict)

    def test_setup_panel_params_with_theta_y(self):
        c = coord_radial(theta="y")
        pp = c.setup_panel_params(None, None)
        assert "theta.range" in pp

    def test_setup_panel_params_no_scales(self):
        c = coord_radial()
        pp = c.setup_panel_params(None, None)
        assert "theta.range" in pp


# ===========================================================================
# CoordTransform (lines 1227, 1251, 1253, 1268, 1281, 1299, 1303, 1306, 1310)
# ===========================================================================

class TestCoordTransform:
    def test_distance(self):
        c = CoordTransform()
        pp = {"x.range": [0, 10], "y.range": [0, 10]}
        result = c.distance(np.array([0.0, 5.0]), np.array([0.0, 5.0]), pp)
        assert len(result) > 0

    def test_backtransform_range(self):
        c = CoordTransform()
        pp = {"x.range": [0, 10], "y.range": [0, 10]}
        result = c.backtransform_range(pp)
        assert "x" in result and "y" in result

    def test_transform(self):
        c = CoordTransform()
        pp = {"x.range": [0, 10], "y.range": [0, 10], "x_range": [0, 10], "y_range": [0, 10],
              "reverse": "none"}
        df = pd.DataFrame({"x": [2.0, 5.0], "y": [3.0, 7.0]})
        result = c.transform(df, pp)
        assert isinstance(result, pd.DataFrame)

    def test_transform_reversed(self):
        c = CoordTransform()
        pp = {"x.range": [0, 10], "y.range": [0, 10], "x_range": [0, 10], "y_range": [0, 10],
              "reverse": "xy"}
        df = pd.DataFrame({"x": [2.0, 5.0], "y": [3.0, 7.0]})
        result = c.transform(df, pp)
        assert isinstance(result, pd.DataFrame)

    def test_setup_panel_params(self):
        c = CoordTransform()
        pp = c.setup_panel_params(None, None)
        assert "x.range" in pp


# ===========================================================================
# coord_munch (lines 1379, 1387-1410)
# ===========================================================================

class TestCoordMunch:
    def test_munch_basic(self):
        c = CoordCartesian()
        pp = {"x_range": [0, 10], "y_range": [0, 10], "x.range": [0, 10], "y.range": [0, 10]}
        df = pd.DataFrame({"x": [0.0, 5.0, 10.0], "y": [0.0, 5.0, 10.0]})
        result = coord_munch(c, df, pp, n=5)
        assert isinstance(result, pd.DataFrame)

    def test_munch_single_row(self):
        c = CoordCartesian()
        pp = {"x_range": [0, 10], "y_range": [0, 10], "x.range": [0, 10], "y.range": [0, 10]}
        df = pd.DataFrame({"x": [5.0], "y": [5.0]})
        result = coord_munch(c, df, pp)
        assert isinstance(result, pd.DataFrame)

    def test_munch_with_interpolation(self):
        """Should interpolate segments (lines 1387-1410)."""
        c = CoordCartesian()
        pp = {"x_range": [0, 100], "y_range": [0, 100], "x.range": [0, 100], "y.range": [0, 100]}
        df = pd.DataFrame({
            "x": [0.0, 50.0, 100.0],
            "y": [0.0, 100.0, 0.0],
            "colour": ["red", "red", "red"],
        })
        result = coord_munch(c, df, pp, n=10)
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 3  # At least original points

    def test_munch_two_points(self):
        c = CoordCartesian()
        pp = {"x_range": [0, 10], "y_range": [0, 10], "x.range": [0, 10], "y.range": [0, 10]}
        df = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 10.0]})
        result = coord_munch(c, df, pp, n=3)
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# coord_* constructors (lines 1533, 1581, 1583, 1589-1590, 1593, 1598, 1700)
# ===========================================================================

class TestCoordConstructors:
    def test_coord_polar_theta_x(self):
        c = coord_polar(theta="x")
        assert c is not None

    def test_coord_polar_theta_y(self):
        c = coord_polar(theta="y")
        assert c is not None

    def test_coord_radial_default(self):
        c = coord_radial()
        assert c is not None

    def test_coord_radial_reversed_theta(self):
        c = coord_radial(reverse="theta")
        assert c is not None

    def test_coord_radial_reversed_r(self):
        c = coord_radial(reverse="r")
        assert c is not None

    def test_coord_radial_reversed_thetar(self):
        c = coord_radial(reverse="thetar")
        assert c is not None

    def test_coord_radial_arc_wrap(self):
        c = coord_radial(start=10.0, end=5.0)
        assert c is not None

    def test_coord_trans(self):
        c = coord_trans()
        assert c is not None

    def test_coord_radial_inner_radius(self):
        c = coord_radial(inner_radius=0.3)
        assert c is not None

    def test_coord_radial_theta_y(self):
        c = coord_radial(theta="y")
        pp = {"theta.range": [0, 1], "r.range": [0, 1],
              "arc": (0, 2 * math.pi),
              "bbox": {"x": [0, 1], "y": [0, 1]},
              "inner_radius": (0, 0.4)}
        df = pd.DataFrame({"x": [0.5], "y": [0.5]})
        result = c.transform(df, pp)
        assert isinstance(result, pd.DataFrame)

    def test_coord_transform_reverse_x(self):
        c = CoordTransform()
        pp = {"x.range": [0, 10], "y.range": [0, 10],
              "x_range": [0, 10], "y_range": [0, 10],
              "reverse": "x"}
        df = pd.DataFrame({"x": [2.0], "y": [3.0]})
        result = c.transform(df, pp)
        assert isinstance(result, pd.DataFrame)

    def test_coord_transform_reverse_y(self):
        c = CoordTransform()
        pp = {"x.range": [0, 10], "y.range": [0, 10],
              "x_range": [0, 10], "y_range": [0, 10],
              "reverse": "y"}
        df = pd.DataFrame({"x": [2.0], "y": [3.0]})
        result = c.transform(df, pp)
        assert isinstance(result, pd.DataFrame)
