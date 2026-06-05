"""CoordPolar panel/axis rendering + coord_munch correctness.

Workstreams B + C of coord_polar_plan.md.

R gold standard: R/coord-polar.R, R/coord-munch.R (ggplot2 4.0.2).  Every
numeric expectation here was produced by running the corresponding R internal
(``dist_polar``, ``coord_munch``, ``CoordPolar$setup_panel_params``,
``CoordPolar$render_bg``) on the same inputs (both-side verification).
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

import ggplot2_py as gg
from ggplot2_py.coord import (
    CoordPolar,
    coord_polar,
    coord_munch,
    _dist_polar,
    _spiral_arc_length,
    _interp,
    _close_poly,
    _vec_interleave,
)


def _unit_values(u):
    """Return numeric values of a grid Unit (or plain array)."""
    if hasattr(u, "values"):
        v = u.values
        return np.asarray(v() if callable(v) else v, dtype=float)
    return np.asarray(u, dtype=float)


# ---------------------------------------------------------------------------
# Workstream B — dist_polar (== R dist_polar, spiral arc length)
# ---------------------------------------------------------------------------

class TestDistPolar:
    def test_spiral_circle_ray_mixed(self):
        # R dist_polar(c(0.1,0.5,0.5,0.2,0.2), c(0.3,1.2,1.2,1.2,0.3))
        out = _dist_polar(
            np.array([0.1, 0.5, 0.5, 0.2, 0.2]),
            np.array([0.3, 1.2, 1.2, 1.2, 0.3]),
        )
        exp = np.array([0.14494771, np.nan, 0.08867753, 0.05320652])
        assert np.allclose(out, exp, equal_nan=True, atol=1e-7)

    def test_rect_corners(self):
        # R dist_polar(c(0,0.4,0.4,0), c(0,0,1.5,1.5))
        out = _dist_polar(
            np.array([0.0, 0.4, 0.4, 0.0]), np.array([0.0, 0.0, 1.5, 1.5])
        )
        assert np.allclose(out, [0.1182367, 0.17735505, 0.1182367], atol=1e-7)

    def test_single_spiral_segment(self):
        # R dist_polar(c(0.1,0.45), c(0.2,2.7))
        out = _dist_polar(np.array([0.1, 0.45]), np.array([0.2, 2.7]))
        assert np.allclose(out, [0.23095939], atol=1e-7)

    def test_circular_arc(self):
        # r constant -> circular arc: R 0.08867753, 0.1773551, 0.08867753
        out = _dist_polar(
            np.array([0.1, 0.4, 0.4, 0.1]), np.array([0.5, 0.5, 2.0, 2.0])
        )
        assert np.allclose(out, [0.08867753, 0.17735505, 0.08867753], atol=1e-7)

    def test_length(self):
        # dist has n-1 entries (R semantics).
        out = _dist_polar(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.5, 0.9]))
        assert len(out) == 2

    def test_spiral_arc_length_matches_formula(self):
        # spiral_arc_length(a=0.2, theta1=0.5*pi, theta2=pi) — R value.
        v = _spiral_arc_length(
            np.array(0.2), np.array(0.5 * math.pi), np.array(math.pi)
        )
        # Reproduce R closed form independently.
        a, t1, t2 = 0.2, 0.5 * math.pi, math.pi
        exp = 0.5 * a * (
            (t1 * math.sqrt(1 + t1 * t1) + math.asinh(t1))
            - (t2 * math.sqrt(1 + t2 * t2) + math.asinh(t2))
        )
        assert math.isclose(float(v), exp, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Workstream B — coord_munch (absolute segment_length = 0.01, == R)
# ---------------------------------------------------------------------------

class TestCoordMunch:
    def test_interp(self):
        # R interp(0, 10, 4) = 0, 2.5, 5, 7.5 (end never included).
        assert np.allclose(_interp(0.0, 10.0, 4), [0.0, 2.5, 5.0, 7.5])
        # n == 1 -> just start.
        assert np.allclose(_interp(3.0, 9.0, 1), [3.0])

    def test_close_poly_repeats_first(self):
        df = pd.DataFrame({"x": [0.0, 1.0, 1.0], "y": [0.0, 0.0, 1.0],
                           "group": [1, 1, 1]})
        out = _close_poly(df)
        # First row repeated after the group (closing the polygon).
        assert len(out) == 4
        assert out.iloc[-1]["x"] == 0.0 and out.iloc[-1]["y"] == 0.0

    def test_rect_to_wedge_matches_R(self):
        # R coord_munch(coord_polar('x'), rect-corner data, is_closed=TRUE)
        # produced 244 rows; spot-check head/tail/sums.
        c = coord_polar(theta="x")
        data = pd.DataFrame({
            "x": [0.0, 1.0, 1.0, 0.0],
            "y": [0.0, 0.0, 1.0, 1.0],
            "group": [1, 1, 1, 1],
        })
        pp = {"theta.range": [0, 1], "r.range": [0, 1]}
        m = coord_munch(c, data, pp, is_closed=True)
        assert len(m) == 244
        # All x of the first spoke collapse to centre 0.5 (theta=0 ray).
        assert np.allclose(m["x"].values[:8], 0.5)
        assert np.allclose(
            m["y"].values[2:8],
            [0.513793, 0.527586, 0.541379, 0.555172, 0.568966, 0.582759],
            atol=1e-5,
        )
        assert math.isclose(m["x"].sum(), 122.0, abs_tol=1e-4)
        assert math.isclose(m["y"].sum(), 133.6, abs_tol=1e-4)

    def test_absolute_segment_length(self):
        # A long arc must be subdivided into many pieces (~ dist / 0.01).
        c = coord_polar(theta="x")
        data = pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 1.0],
                             "group": [1, 1]})
        pp = {"theta.range": [0, 1], "r.range": [0, 1]}
        m = coord_munch(c, data, pp, is_closed=False)
        # The outer ring sweeps the full circle; many subdivisions expected.
        assert len(m) > 10

    def test_linear_coord_passthrough(self):
        from ggplot2_py.coord import CoordCartesian
        c = CoordCartesian()
        data = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
        pp = {"x.range": [0, 1], "y.range": [0, 1]}
        out = coord_munch(c, data, pp)
        # Linear: no subdivision, just transform (2 rows in, 2 rows out).
        assert len(out) == 2

    def test_backcompat_n_kwarg_ignored(self):
        # Old call sites pass n=...; it must be accepted and ignored.
        c = coord_polar(theta="x")
        data = pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 1.0], "group": [1, 1]})
        pp = {"theta.range": [0, 1], "r.range": [0, 1]}
        a = coord_munch(c, data, pp, n=5)
        b = coord_munch(c, data, pp)
        assert len(a) == len(b)


# ---------------------------------------------------------------------------
# Workstream C — setup_panel_params (R expansion)
# ---------------------------------------------------------------------------

class TestSetupPanelParams:
    def _pp(self, theta="x"):
        c = coord_polar(theta=theta)
        sx = gg.scale_x_discrete(); sx.train(["a", "b", "c", "d"])
        sy = gg.scale_y_continuous(); sy.train([0, 10])
        return c, c.setup_panel_params(sx, sy)

    def test_theta_expansion_closes_ring(self):
        # Discrete theta gets add=0.5 each side -> [0.5, 4.5] (R value).
        _, pp = self._pp("x")
        assert np.allclose(pp["theta.range"], [0.5, 4.5])

    def test_r_no_expansion(self):
        _, pp = self._pp("x")
        assert np.allclose(pp["r.range"], [0.0, 10.0])

    def test_breaks_and_labels(self):
        _, pp = self._pp("x")
        assert np.allclose(pp["theta.major"], [1, 2, 3, 4])
        assert list(pp["theta.labels"]) == ["a", "b", "c", "d"]
        assert np.allclose(pp["r.major"], [0, 2.5, 5, 7.5, 10])

    def test_r_arrange(self):
        _, pp = self._pp("x")
        assert pp["r.arrange"] == ["primary", "secondary"]

    def test_theta_y_swaps_axes(self):
        # When theta='y', the y scale drives theta and x drives r.
        c = coord_polar(theta="y")
        sx = gg.scale_x_continuous(); sx.train([0, 10])
        sy = gg.scale_y_discrete(); sy.train(["a", "b", "c", "d"])
        pp = c.setup_panel_params(sx, sy)
        assert np.allclose(pp["theta.range"], [0.5, 4.5])
        assert np.allclose(pp["r.range"], [0.0, 10.0])


# ---------------------------------------------------------------------------
# Workstream C — render_bg / render_fg / axes
# ---------------------------------------------------------------------------

class TestRenderBg:
    def _bg(self):
        c = coord_polar(theta="x")
        sx = gg.scale_x_discrete(); sx.train(["a", "b", "c", "d"])
        sy = gg.scale_y_continuous(); sy.train([0, 10])
        pp = c.setup_panel_params(sx, sy)
        return c.render_bg(pp, gg.theme_grey())

    def test_is_grill_tree(self):
        bg = self._bg()
        assert getattr(bg, "name", None) == "grill"
        # background rect + theta spokes + r circles.
        assert bg.n_children() == 3

    def test_spokes_geometry(self):
        bg = self._bg()
        spokes = list(bg.get_children())[1]
        x = _unit_values(spokes.x)
        y = _unit_values(spokes.y)
        # 4 spokes -> 8 endpoints; each starts at centre (0.5, 0.5).
        assert len(x) == 8
        assert np.allclose(x[0::2], 0.5)
        assert np.allclose(y[0::2], 0.5)
        # R reference endpoints.
        assert np.allclose(
            x, [0.5, 0.81820, 0.5, 0.81820, 0.5, 0.18180, 0.5, 0.18180],
            atol=1e-4,
        )
        assert np.allclose(
            y, [0.5, 0.81820, 0.5, 0.18180, 0.5, 0.18180, 0.5, 0.81820],
            atol=1e-4,
        )

    def test_concentric_circles(self):
        bg = self._bg()
        circ = list(bg.get_children())[2]
        x = _unit_values(circ.x)
        # 6 radii (5 r.major + outer 0.45) * 100 thetafine = 600 points.
        assert len(x) == 600
        ids = list(circ.id_lengths)
        assert ids == [100] * 6


class TestRenderFg:
    def _fg(self, theme=None):
        c = coord_polar(theta="x")
        sx = gg.scale_x_discrete(); sx.train(["a", "b", "c", "d"])
        sy = gg.scale_y_continuous(); sy.train([0, 10])
        pp = c.setup_panel_params(sx, sy)
        return c.render_fg(pp, theme or gg.theme_grey())

    def test_theta_labels_present(self):
        fg = self._fg()
        children = list(fg.get_children())
        # text grob + panel.border grob.
        text = children[0]
        assert list(text.label) == ["a", "b", "c", "d"]

    def test_combine_close_ends(self):
        # Continuous theta wrapping 0..1 -> first & last close -> "0/1".
        c = coord_polar(theta="y")
        sx = gg.scale_x_continuous(); sx.train([0, 1])
        sy = gg.scale_y_continuous(); sy.train([0, 1])
        pp = c.setup_panel_params(sx, sy)
        fg = c.render_fg(pp, gg.theme_grey())
        text = list(fg.get_children())[0]
        labels = list(text.label)
        # The combined "first/last" label appears (R combine-close-ends rule).
        assert any("/" in str(l) for l in labels)

    def test_none_theta_returns_border(self):
        c = coord_polar(theta="x")
        out = c.render_fg({"theta.major": None}, gg.theme_grey())
        # No labels -> just the panel.border element grob (not a crash).
        assert out is not None


class TestRenderAxes:
    def _pp(self):
        c = coord_polar(theta="x")
        sx = gg.scale_x_discrete(); sx.train(["a", "b", "c", "d"])
        sy = gg.scale_y_continuous(); sy.train([0, 10])
        return c, c.setup_panel_params(sx, sy)

    def test_axis_v_keys(self):
        c, pp = self._pp()
        out = c.render_axis_v(pp, gg.theme_grey())
        assert set(out.keys()) == {"left", "right"}

    def test_axis_h_keys(self):
        c, pp = self._pp()
        out = c.render_axis_h(pp, gg.theme_grey())
        assert set(out.keys()) == {"top", "bottom"}

    def test_aspect_is_one(self):
        c = coord_polar(theta="x")
        assert c.aspect({}) == 1.0


# ---------------------------------------------------------------------------
# setup_panel_guides warns (R parity)
# ---------------------------------------------------------------------------

class TestSetupPanelGuides:
    def test_warns_on_unsupported_guide(self):
        c = coord_polar(theta="x")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c.setup_panel_guides({}, {"x": object()})
        assert any("guide" in str(rec.message).lower() for rec in w)

    def test_no_warn_without_guides(self):
        c = coord_polar(theta="x")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c.setup_panel_guides({}, {})
        assert len(w) == 0


# ---------------------------------------------------------------------------
# End-to-end: a pie builds and renders with a polar (circular) panel
# ---------------------------------------------------------------------------

class TestPolarPlotBuild:
    def test_pie_builds_to_gtable(self):
        df = pd.DataFrame({
            "cat": ["a", "b", "c", "d"],
            "val": [10, 20, 30, 40],
            "x": ["all"] * 4,
        })
        p = (
            gg.ggplot(df, gg.aes(x="x", y="val", fill="cat"))
            + gg.geom_col(position="fill")
            + gg.coord_polar(theta="y")
        )
        built = gg.ggplot_build(p)
        table = gg.ggplot_gtable(built)
        assert table is not None

    def test_render_fg_wired_in_draw_panel(self):
        # The base Coord.draw_panel draws bg behind + fg in front; ensure
        # CoordPolar inherits it and produces a tree containing both.
        c = coord_polar(theta="x")
        sx = gg.scale_x_discrete(); sx.train(["a", "b", "c", "d"])
        sy = gg.scale_y_continuous(); sy.train([0, 10])
        pp = c.setup_panel_params(sx, sy)
        from grid_py import null_grob
        decorated = c.draw_panel([null_grob()], pp, gg.theme_grey())
        # bg (grill) first, fg (labels/border) last.
        kids = list(decorated.get_children())
        names = [getattr(k, "name", "") for k in kids]
        assert "grill" in names
        assert "fg" in names
        assert names.index("grill") < names.index("fg")
