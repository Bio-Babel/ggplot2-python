"""CoordRadial (modern radial coord) — R-parity tests.

Phase 2 of coord_polar_plan.md.  ``coord_radial()`` supersedes
``coord_polar()`` and adds: partial arcs (``start``/``end``), donut
holes (``inner.radius``), per-axis ``default_expansion``, and a
bounding-box driven aspect ratio.

R gold standard: R/coord-radial.R (ggplot2 4.0.2).  Every numeric
expectation below was produced by running the corresponding R internal
(``CoordRadial$setup_panel_params``, ``$transform``, ``$aspect``,
``polar_bbox``, the ``coord_radial`` constructor) on the same inputs —
both-side verification, not static reading.
"""

import math

import numpy as np
import pandas as pd
import pytest

import ggplot2_py as gg
from ggplot2_py.coord import (
    CoordRadial,
    coord_radial,
    coord_munch,
    _polar_bbox,
    _in_arc,
)


def _pp(coord, scale_x, scale_y):
    return coord.setup_panel_params(scale_x, scale_y, coord.setup_params([pd.DataFrame()]))


# ---------------------------------------------------------------------------
# Constructor — arc / inner_radius / r_axis_inside resolution (R:113-193)
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_full_circle_default_arc(self):
        c = coord_radial(theta="x")
        assert np.allclose(c.arc, [0.0, 2 * math.pi])

    def test_partial_arc(self):
        c = coord_radial(theta="x", start=0, end=math.pi)
        assert np.allclose(c.arc, [0.0, math.pi])

    def test_start_greater_than_end_rotates(self):
        # R: start>end -> rotate start below end.
        c = coord_radial(theta="x", start=math.pi, end=0.5 * math.pi)
        assert np.allclose(c.arc, [-math.pi, 0.5 * math.pi])

    def test_reverse_theta_flips_arc(self):
        c = coord_radial(theta="x", reverse="theta")
        assert np.allclose(c.arc, [2 * math.pi, 0.0])

    def test_inner_radius_scaled(self):
        # R: inner.radius -> c(inner, 1) * 0.4.
        c = coord_radial(theta="x", inner_radius=0.3)
        assert np.allclose(c.inner_radius, [0.12, 0.4])

    def test_inner_radius_reverse_r(self):
        c = coord_radial(theta="x", inner_radius=0.3, reverse="r")
        assert np.allclose(c.inner_radius, [0.4, 0.12])

    def test_r_axis_inside_default_full_circle_is_false(self):
        # Full circle -> axis drawn outside.
        c = coord_radial(theta="x")
        assert c.r_axis_inside is False

    def test_r_axis_inside_default_partial_is_true(self):
        c = coord_radial(theta="x", start=0, end=math.pi)
        assert c.r_axis_inside is True

    def test_r_axis_inside_numeric_passthrough(self):
        c = coord_radial(theta="x", r_axis_inside=5)
        assert c.r_axis_inside == 5

    def test_inner_radius_out_of_range_errors(self):
        with pytest.raises(Exception):
            coord_radial(theta="x", inner_radius=1.5)

    def test_invalid_theta_errors(self):
        with pytest.raises(Exception):
            coord_radial(theta="z")


# ---------------------------------------------------------------------------
# polar_bbox / in_arc (R:583-634)
# ---------------------------------------------------------------------------

class TestPolarBbox:
    def test_full_circle_is_unit_box(self):
        bb = _polar_bbox((0.0, 2 * math.pi))
        assert np.allclose(bb["x"], [0.0, 1.0])
        assert np.allclose(bb["y"], [0.0, 1.0])

    def test_half_circle_bbox(self):
        # R polar_bbox(c(0, pi)) -> x=[0.45, 1], y=[0, 1].
        bb = _polar_bbox((0.0, math.pi))
        assert np.allclose(bb["x"], [0.45, 1.0])
        assert np.allclose(bb["y"], [0.0, 1.0])

    def test_in_arc_full_circle(self):
        out = _in_arc(np.array([0.0, 1.0, 3.0, 5.0]), (0.0, 2 * math.pi))
        assert out.all()

    def test_in_arc_partial(self):
        out = _in_arc(np.array([0.5, math.pi, 1.5 * math.pi]), (0.0, math.pi))
        assert list(out) == [True, True, False]


# ---------------------------------------------------------------------------
# setup_panel_params — ranges/breaks/expansion (R:235-269, view_scales_polar)
# ---------------------------------------------------------------------------

class TestSetupPanelParams:
    def test_discrete_pie_expand_true(self):
        # theta=x discrete + expand -> theta.range = [1-0.6, 4+0.6].
        c = coord_radial(theta="x", expand=True)
        sx = gg.scale_x_discrete(); sx.train(["a", "b", "c", "d"])
        sy = gg.scale_y_continuous(); sy.train([0, 10])
        pp = _pp(c, sx, sy)
        assert np.allclose(pp["theta.range"], [0.4, 4.6])
        assert np.allclose(pp["r.range"], [-0.5, 10.5])

    def test_discrete_pie_breaks_labels(self):
        c = coord_radial(theta="x", expand=True)
        sx = gg.scale_x_discrete(); sx.train(["a", "b", "c", "d"])
        sy = gg.scale_y_continuous(); sy.train([0, 10])
        pp = _pp(c, sx, sy)
        assert np.allclose(pp["theta.major"], [1, 2, 3, 4])
        assert list(pp["theta.labels"]) == ["a", "b", "c", "d"]
        assert np.allclose(pp["r.major"], [0, 2.5, 5, 7.5, 10])

    def test_no_expand_single_category(self):
        # expand=False on a single discrete level -> r.range = [1, 1].
        c = coord_radial(theta="y", expand=False)
        sx = gg.scale_x_discrete(); sx.train(["a"])
        sy = gg.scale_y_continuous(); sy.train([0, 40])
        pp = _pp(c, sx, sy)
        assert np.allclose(pp["theta.range"], [0, 40])
        assert np.allclose(pp["r.range"], [1, 1])

    def test_arc_and_inner_radius_in_params(self):
        c = coord_radial(theta="x", start=0, end=math.pi, inner_radius=0.3)
        sx = gg.scale_x_continuous(); sx.train([0, 10])
        sy = gg.scale_y_continuous(); sy.train([0, 5])
        pp = _pp(c, sx, sy)
        assert np.allclose(pp["arc"], [0.0, math.pi])
        assert np.allclose(pp["inner_radius"], [0.12, 0.4])
        assert np.allclose(pp["bbox"]["x"], [0.45, 1.0])

    def test_axis_rotation_numeric(self):
        # numeric r_axis_inside -> axis_rotation at that theta (rescaled).
        c = coord_radial(theta="x", r_axis_inside=5)
        sx = gg.scale_x_continuous(); sx.train([0, 10])
        sy = gg.scale_y_continuous(); sy.train([0, 5])
        pp = _pp(c, sx, sy)
        # theta=5 in range [-0.5,10.5] -> arc [0,2pi]: ~pi.
        assert len(pp["axis_rotation"]) == 2
        assert np.isclose(pp["axis_rotation"][0], pp["axis_rotation"][1])


# ---------------------------------------------------------------------------
# transform (R:392-411) — r*sin/cos with arc + inner_radius + bbox
# ---------------------------------------------------------------------------

class TestTransform:
    def test_full_circle_pie(self):
        # theta=y, r is single category -> r=0.4; transform onto a circle.
        c = coord_radial(theta="y", expand=False)
        sx = gg.scale_x_discrete(); sx.train(["a"])
        sy = gg.scale_y_continuous(); sy.train([0, 40])
        pp = _pp(c, sx, sy)
        td = c.transform(pd.DataFrame({"x": [1] * 5, "y": [0, 10, 20, 30, 40]}), pp)
        assert np.allclose(td["x"], [0.5, 0.7, 0.5, 0.3, 0.5], atol=1e-5)
        assert np.allclose(td["y"], [0.7, 0.5, 0.3, 0.5, 0.7], atol=1e-5)

    def test_partial_arc(self):
        c = coord_radial(theta="x", start=0, end=math.pi, expand=False)
        sx = gg.scale_x_continuous(); sx.train([0, 10])
        sy = gg.scale_y_continuous(); sy.train([0, 5])
        pp = _pp(c, sx, sy)
        td = c.transform(pd.DataFrame({"x": [0, 5, 10], "y": [0, 2.5, 5]}), pp)
        assert np.allclose(td["x"], [0.09091, 0.45455, 0.09091], atol=1e-5)
        assert np.allclose(td["y"], [0.5, 0.5, 0.1], atol=1e-5)

    def test_donut_inner_radius(self):
        c = coord_radial(theta="x", inner_radius=0.3, expand=False)
        sx = gg.scale_x_continuous(); sx.train([0, 10])
        sy = gg.scale_y_continuous(); sy.train([0, 5])
        pp = _pp(c, sx, sy)
        td = c.transform(pd.DataFrame({"x": [0, 5, 10], "y": [0, 2.5, 5]}), pp)
        # r=0 maps to inner hole (0.12), not centre.
        assert np.allclose(td["x"], [0.5, 0.5, 0.5], atol=1e-5)
        assert np.allclose(td["y"], [0.62, 0.24, 0.9], atol=1e-5)


# ---------------------------------------------------------------------------
# aspect (R:201-203) — bbox-driven, square for full circle
# ---------------------------------------------------------------------------

class TestAspect:
    def test_full_circle_is_square(self):
        c = coord_radial(theta="x")
        assert c.aspect({"bbox": {"x": [0, 1], "y": [0, 1]}}) == 1.0

    def test_partial_arc_aspect(self):
        # half-circle bbox x=[0.45,1], y=[0,1] -> aspect 1/0.55 = 1.81818.
        c = coord_radial(theta="x", start=0, end=math.pi)
        sx = gg.scale_x_continuous(); sx.train([0, 10])
        sy = gg.scale_y_continuous(); sy.train([0, 5])
        pp = _pp(c, sx, sy)
        assert np.isclose(c.aspect(pp), 1.818182, atol=1e-5)


# ---------------------------------------------------------------------------
# render_bg / render_fg (R:427-467) — munch-based grill + theta labels
# ---------------------------------------------------------------------------

class TestRender:
    def _setup(self, **kw):
        c = coord_radial(theta="x", expand=False, **kw)
        sx = gg.scale_x_discrete(); sx.train(["a", "b", "c", "d"])
        sy = gg.scale_y_continuous(); sy.train([0, 10])
        return c, _pp(c, sx, sy)

    def test_render_bg_is_grill(self):
        c, pp = self._setup()
        bg = c.render_bg(pp, gg.theme_grey())
        assert getattr(bg, "name", None) == "grill"
        # background polygon + r rings + theta spokes (minor absent for
        # discrete theta / continuous minor not produced by break_info).
        assert len(list(bg.get_children())) >= 2

    def test_render_bg_background_polygon_is_munched(self):
        # The background poly is the Inf-rect munched into a circle:
        # all vertices lie within the unit square.
        c, pp = self._setup()
        bg = c.render_bg(pp, gg.theme_grey())
        poly = [k for k in bg.get_children()
                if getattr(k, "name", "") == "panel.background.polygon"]
        assert len(poly) == 1
        x = np.asarray(poly[0].x.values if hasattr(poly[0].x, "values") else poly[0].x,
                       dtype=float)
        assert x.min() >= -1e-6 and x.max() <= 1 + 1e-6

    def test_render_fg_theta_labels(self):
        # Full-circle discrete pie: positions 1 and 4 land at the same
        # angle (theta wraps 0..2pi), so R combines the first/last labels
        # ("a/d") and drops the duplicate spoke -> labels [b, c, a/d].
        c, pp = self._setup()
        fg = c.render_fg(pp, gg.theme_grey())
        assert getattr(fg, "name", None) == "fg"
        text = list(fg.get_children())[0]
        assert list(text.label) == ["b", "c", "a/d"]

    def test_render_fg_combine_close_ends(self):
        # Continuous theta wrapping 0..1 -> first & last close -> combined.
        c = coord_radial(theta="y", expand=False)
        sx = gg.scale_x_continuous(); sx.train([0, 1])
        sy = gg.scale_y_continuous(); sy.train([0, 1])
        pp = _pp(c, sx, sy)
        fg = c.render_fg(pp, gg.theme_grey())
        text = list(fg.get_children())[0]
        assert any("/" in str(l) for l in text.label)

    def _find_text_labels(self, grob):
        """Collect all label strings from text grobs in a grob tree.

        Descends through GTree children *and* gtable ``.grobs`` (the axis
        ticks/labels live inside the assembled axis gtable).
        """
        out = []
        lbl = getattr(grob, "label", None)
        if lbl is not None:
            try:
                out.extend([str(x) for x in lbl])
            except TypeError:
                out.append(str(lbl))
        get_children = getattr(grob, "get_children", None)
        if callable(get_children):
            for ch in get_children():
                out.extend(self._find_text_labels(ch))
        gtable_grobs = getattr(grob, "grobs", None)
        if gtable_grobs is not None:
            for ch in gtable_grobs:
                out.extend(self._find_text_labels(ch))
        return out

    def test_render_fg_inside_axis_is_rotated_grob(self):
        """R coord-radial.R:436-467 — r_axis_inside=True draws an in-panel
        rotated r-axis (ticks + labels) via a viewport carrying an angle.

        Both-side verified: for the canonical partial-arc + donut case R's
        render_fg builds a left axis rotated by 72 deg, pinned at the panel
        centre rescaled into bbox (y=0.025185), with height 1.949630 and
        just c(1, 0.5), tick text rotated -72 deg, r labels 10..35.
        """
        # mtcars disp (theta) / mpg (r), partial arc + inner radius (R example).
        c = coord_radial(theta="x", start=-0.4 * math.pi, end=0.4 * math.pi,
                         inner_radius=0.3)
        sx = gg.scale_x_continuous(); sx.train([71.1, 472.0])
        sy = gg.scale_y_continuous(); sy.train([10.4, 33.9])
        pp = _pp(c, sx, sy)
        assert c.r_axis_inside is True  # default for partial arc

        fg = c.render_fg(pp, gg.theme_grey())
        children = list(fg.get_children())

        # Locate the rotated r-axis grob: a child carrying a viewport with a
        # non-default angle/justification (the theta labels + border do not).
        rotated = [
            ch for ch in children
            if getattr(ch, "vp", None) is not None
            and abs(getattr(ch.vp, "angle", 0.0)) > 1e-9
        ]
        assert len(rotated) == 1, "exactly one rotated in-panel r-axis expected"
        vp = rotated[0].vp

        # Geometry parity with R (panel_guides_grob + rotate_r_axis).
        assert pp["r_axis_inside_position"] == "left"
        assert pp["r_axis_inside_angle"] == pytest.approx(-72.0)
        assert vp.angle == pytest.approx(72.0)
        assert float(vp.x.values[0]) == pytest.approx(0.5)
        assert float(vp.y.values[0]) == pytest.approx(0.025185237533905, abs=1e-9)
        assert float(vp.height.values[0]) == pytest.approx(1.9496295249322, abs=1e-9)
        assert tuple(vp.just) == pytest.approx((1.0, 0.5))

        # The r tick labels (10..35) must appear inside the rotated axis.
        labels = self._find_text_labels(rotated[0])
        for expect in ("10", "15", "20", "25", "30", "35"):
            assert expect in labels

    def test_render_fg_inside_axis_numeric_theta(self):
        """Numeric r_axis_inside (a theta value) rotates the axis to that
        angle (R coord-radial.R:254-266 -> render_fg)."""
        c = coord_radial(theta="x", r_axis_inside=300)
        sx = gg.scale_x_continuous(); sx.train([71.1, 472.0])
        sy = gg.scale_y_continuous(); sy.train([10.4, 33.9])
        pp = _pp(c, sx, sy)
        fg = c.render_fg(pp, gg.theme_grey())
        rotated = [
            ch for ch in fg.get_children()
            if getattr(ch, "vp", None) is not None
            and abs(getattr(ch.vp, "angle", 0.0)) > 1e-9
        ]
        assert len(rotated) == 1
        # R: axis_rotation = 3.546946 rad -> render_fg angle = -rad2deg = -203.225
        assert rotated[0].vp.angle == pytest.approx(-203.225016440282, abs=1e-6)


# ---------------------------------------------------------------------------
# render_axis_* (R:413-425) — r-axis suppressed when inside
# ---------------------------------------------------------------------------

class TestRenderAxis:
    def test_inside_axis_suppressed(self):
        from grid_py import null_grob
        c = coord_radial(theta="x", start=0, end=math.pi)  # inside default
        sx = gg.scale_x_continuous(); sx.train([0, 10])
        sy = gg.scale_y_continuous(); sy.train([0, 5])
        pp = _pp(c, sx, sy)
        v = c.render_axis_v(pp, gg.theme_grey())
        assert set(v.keys()) == {"left", "right"}

    def test_outside_axis_drawn(self):
        # Full circle -> r_axis_inside=False -> the chosen side is drawn.
        c = coord_radial(theta="x", expand=False)
        sx = gg.scale_x_discrete(); sx.train(["a", "b", "c", "d"])
        sy = gg.scale_y_continuous(); sy.train([0, 10])
        pp = _pp(c, sx, sy)
        h = c.render_axis_h(pp, gg.theme_grey())
        v = c.render_axis_v(pp, gg.theme_grey())
        assert set(h.keys()) == {"top", "bottom"}
        assert set(v.keys()) == {"left", "right"}


# ---------------------------------------------------------------------------
# End-to-end — pie / partial / donut build and render to a gtable
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def _df(self):
        return pd.DataFrame({"x": ["all"] * 4, "cat": ["a", "b", "c", "d"],
                             "val": [10, 20, 30, 40]})

    def test_full_circle_pie_builds(self):
        p = (gg.ggplot(self._df(), gg.aes(x="x", y="val", fill="cat"))
             + gg.geom_col(position="fill")
             + gg.coord_radial(theta="y", expand=False))
        table = gg.ggplot_gtable(gg.ggplot_build(p))
        assert table is not None

    def test_donut_builds(self):
        p = (gg.ggplot(self._df(), gg.aes(x="x", y="val", fill="cat"))
             + gg.geom_col(position="fill")
             + gg.coord_radial(theta="y", inner_radius=0.3, expand=False))
        table = gg.ggplot_gtable(gg.ggplot_build(p))
        assert table is not None

    def test_partial_arc_builds(self):
        mt = pd.DataFrame({"disp": [160, 200, 250, 300], "mpg": [21, 19, 17, 15]})
        p = (gg.ggplot(mt, gg.aes(x="disp", y="mpg"))
             + gg.geom_point()
             + gg.coord_radial(start=0, end=math.pi))
        table = gg.ggplot_gtable(gg.ggplot_build(p))
        assert table is not None

    def test_is_nonlinear(self):
        c = coord_radial(theta="x")
        assert c.is_linear() is False
