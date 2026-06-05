"""Non-linear (polar) geom drawing — Workstream A of coord_polar.

These tests verify that under a non-linear coordinate system (``coord_polar``)
the position-based geoms *munch* their straight edges into arcs/wedges (R's
``if (!coord$is_linear())`` branch), while the Cartesian (linear) draw path is
left byte-for-byte unchanged.

R gold standard: geom-rect.R, geom-polygon.R, geom-path.R, geom-ribbon.R,
geom-segment.R, geom-raster.R + coord-munch.R.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from ggplot2_py.geom import (
    GeomRect,
    GeomPolygon,
    GeomPath,
    GeomLine,
    GeomSegment,
    GeomRibbon,
    GeomArea,
    GeomRaster,
    GeomTile,
    GeomBar,
    _rect_to_polygon,
    _coord_is_linear,
)
from ggplot2_py.coord import CoordCartesian, CoordPolar, coord_munch


# ---------------------------------------------------------------------------
# Helpers to extract grob coordinates / class regardless of nesting.
# ---------------------------------------------------------------------------

def _unit_values(u):
    """Return the numeric values of a grid ``Unit`` (or array)."""
    return np.asarray(getattr(u, "values", u), dtype=float)


def _find_grobs(grob, classes=("polygon", "path", "pathgrob", "polyline", "lines", "rect")):
    """Recursively collect grobs of the given ``_grid_class`` values."""
    out = []

    def walk(o):
        cls = getattr(o, "_grid_class", None)
        if cls in classes:
            out.append(o)
        kids = []
        if hasattr(o, "get_children"):
            try:
                kids = list(o.get_children())
            except Exception:
                kids = []
        if not kids:
            gr = getattr(o, "grobs", None)
            if gr is not None:
                kids = list(gr.values()) if hasattr(gr, "values") else list(gr)
        for c in kids:
            walk(c)

    walk(grob)
    return out


POLAR = CoordPolar(theta="x")
POLAR_Y = CoordPolar(theta="y")
CART = CoordCartesian()
PP = {"theta.range": [0, 1], "r.range": [0, 1], "x.range": [0, 1], "y.range": [0, 1]}


def _rect_df():
    return pd.DataFrame({
        "xmin": [0.0], "xmax": [0.5], "ymin": [0.0], "ymax": [1.0],
        "fill": ["red"], "colour": [None], "linewidth": [0.5],
        "linetype": [1], "alpha": [None],
    })


# ---------------------------------------------------------------------------
# _coord_is_linear guard
# ---------------------------------------------------------------------------

def test_coord_is_linear_guard():
    assert _coord_is_linear(CART) is True
    assert _coord_is_linear(POLAR) is False
    # Missing coord is treated as linear (legend keys, defensive).
    assert _coord_is_linear(None) is True


# ---------------------------------------------------------------------------
# _rect_to_polygon helper (R geom-rect.R:50-58)
# ---------------------------------------------------------------------------

def test_rect_to_polygon_interleave():
    df = pd.DataFrame({
        "xmin": [0.0], "xmax": [2.0], "ymin": [1.0], "ymax": [3.0],
        "fill": ["blue"],
    })
    poly = _rect_to_polygon(df)
    assert len(poly) == 4
    # R: x = interleave(xmin, xmax, xmax, xmin); y = (ymax, ymax, ymin, ymin)
    assert list(poly["x"]) == [0.0, 2.0, 2.0, 0.0]
    assert list(poly["y"]) == [3.0, 3.0, 1.0, 1.0]
    # group is the originating row index
    assert list(poly["group"]) == [0, 0, 0, 0]
    # non-position aesthetics carried over, repeated 4x
    assert list(poly["fill"]) == ["blue"] * 4


def test_rect_to_polygon_two_rects():
    df = pd.DataFrame({
        "xmin": [0.0, 5.0], "xmax": [1.0, 6.0],
        "ymin": [0.0, 0.0], "ymax": [1.0, 2.0],
        "fill": ["a", "b"],
    })
    poly = _rect_to_polygon(df)
    assert len(poly) == 8
    assert list(poly["group"]) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert list(poly["fill"]) == ["a"] * 4 + ["b"] * 4


# ---------------------------------------------------------------------------
# GeomRect — non-linear branch bends into a wedge; linear branch unchanged.
# ---------------------------------------------------------------------------

def test_geomrect_polar_produces_wedge_polygon():
    grob = GeomRect.draw_panel(GeomRect(), _rect_df(), PP, POLAR)
    polys = _find_grobs(grob, classes=("polygon",))
    assert len(polys) == 1, "rect under coord_polar must be a polygon grob"
    x = _unit_values(polys[0].x)
    # A flat rect would have 4 corners; a munched wedge has many more.
    assert len(x) > 20, "wedge polygon should be munched (many points)"


def test_geomrect_polar_wedge_is_curved():
    grob = GeomRect.draw_panel(GeomRect(), _rect_df(), PP, POLAR)
    poly = _find_grobs(grob, classes=("polygon",))[0]
    x = _unit_values(poly.x)
    y = _unit_values(poly.y)
    rad = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    # Many points share (approximately) the max radius → a circular arc
    # (a flat axis-aligned rect would have only the 2 outer corners there).
    near_max = np.sum(rad >= rad.max() * 0.99)
    assert near_max > 5, "outer wedge edge should trace a curved arc"


def test_geomrect_cartesian_stays_rect():
    grob = GeomRect.draw_panel(GeomRect(), _rect_df(), PP, CART)
    # Linear fast path: a single rect grob, NOT a polygon.
    assert grob._grid_class == "rect"
    assert not _find_grobs(grob, classes=("polygon",))


# ---------------------------------------------------------------------------
# GeomPolygon — per-group gpar + munched edges under polar.
# ---------------------------------------------------------------------------

def test_geompolygon_polar_munches_and_per_group_gpar():
    # Two triangles with different fills.
    df = pd.DataFrame({
        "x": [0.1, 0.4, 0.25, 0.6, 0.9, 0.75],
        "y": [0.1, 0.1, 0.5, 0.1, 0.1, 0.5],
        "group": [1, 1, 1, 2, 2, 2],
        "fill": ["red", "red", "red", "blue", "blue", "blue"],
        "colour": [None] * 6, "linewidth": [0.5] * 6,
        "linetype": [1] * 6, "alpha": [None] * 6,
    })
    grob = GeomPolygon.draw_panel(GeomPolygon(), df, PP, POLAR)
    poly = _find_grobs(grob, classes=("polygon",))[0]
    ids = np.asarray(poly.id)
    assert set(ids.tolist()) == {1, 2}, "two distinct polygon ids"
    fills = poly.gp.params.get("fill")
    assert len(fills) == 2, "one gpar fill entry per group (R first_rows)"
    # munched → many more points than the 6 input vertices
    assert len(_unit_values(poly.x)) > 6


def test_geompolygon_cartesian_unchanged():
    df = pd.DataFrame({
        "x": [0.1, 0.4, 0.25], "y": [0.1, 0.1, 0.5],
        "group": [1, 1, 1], "fill": ["red"] * 3, "colour": [None] * 3,
        "linewidth": [0.5] * 3, "linetype": [1] * 3, "alpha": [None] * 3,
    })
    grob = GeomPolygon.draw_panel(GeomPolygon(), df, PP, CART)
    poly = _find_grobs(grob, classes=("polygon",))[0]
    # Cartesian: exactly the input vertices, no subdivision.
    assert len(_unit_values(poly.x)) == 3


def test_geompolygon_subgroup_holes_use_pathgrob():
    # Outer ring (subgroup 1) + inner hole (subgroup 2), same group.
    outer = pd.DataFrame({"x": [0.1, 0.9, 0.9, 0.1], "y": [0.1, 0.1, 0.9, 0.9],
                          "subgroup": [1, 1, 1, 1]})
    hole = pd.DataFrame({"x": [0.4, 0.6, 0.6, 0.4], "y": [0.4, 0.4, 0.6, 0.6],
                         "subgroup": [2, 2, 2, 2]})
    df = pd.concat([outer, hole], ignore_index=True)
    df["group"] = 1
    df["fill"] = "green"
    df["colour"] = None
    df["linewidth"] = 0.5
    df["linetype"] = 1
    df["alpha"] = None
    grob = GeomPolygon.draw_panel(GeomPolygon(), df, PP, POLAR)
    paths = _find_grobs(grob, classes=("path", "pathgrob"))
    assert len(paths) == 1, "subgroup holes draw via path_grob"


# ---------------------------------------------------------------------------
# GeomPath / GeomLine — munched polyline under polar.
# ---------------------------------------------------------------------------

def test_geompath_polar_munches():
    df = pd.DataFrame({
        "x": [0.1, 0.5, 0.9], "y": [0.2, 0.8, 0.3],
        "group": [1, 1, 1], "colour": ["black"] * 3,
        "linewidth": [0.5] * 3, "linetype": [1] * 3, "alpha": [None] * 3,
    })
    grob = GeomPath.draw_panel(GeomPath(), df, PP, POLAR)
    lines = _find_grobs(grob, classes=("polyline", "lines"))
    assert lines, "polar path should draw a polyline"
    assert len(_unit_values(lines[0].x)) > 3, "path edges should be munched"


def test_geompath_cartesian_unchanged():
    df = pd.DataFrame({
        "x": [0.1, 0.5, 0.9], "y": [0.2, 0.8, 0.3],
        "group": [1, 1, 1], "colour": ["black"] * 3,
        "linewidth": [0.5] * 3, "linetype": [1] * 3, "alpha": [None] * 3,
    })
    grob = GeomPath.draw_panel(GeomPath(), df, PP, CART)
    line = _find_grobs(grob, classes=("polyline", "lines"))[0]
    assert len(_unit_values(line.x)) == 3


# ---------------------------------------------------------------------------
# GeomSegment — each segment is a munched 2-point path under polar.
# ---------------------------------------------------------------------------

def test_geomsegment_polar_munches_each_segment():
    df = pd.DataFrame({
        "x": [0.1, 0.2], "y": [0.1, 0.2],
        "xend": [0.8, 0.9], "yend": [0.7, 0.3],
        "colour": ["black"] * 2, "linewidth": [0.5] * 2,
        "linetype": [1] * 2, "alpha": [None] * 2,
    })
    grob = GeomSegment.draw_panel(GeomSegment(), df, PP, POLAR)
    lines = _find_grobs(grob, classes=("polyline", "lines"))
    # Two segments → two munched polylines, each more than 2 points.
    assert len(lines) == 2
    for ln in lines:
        assert len(_unit_values(ln.x)) > 2


def test_geomsegment_cartesian_stays_segments():
    df = pd.DataFrame({
        "x": [0.1], "y": [0.1], "xend": [0.8], "yend": [0.7],
        "colour": ["black"], "linewidth": [0.5], "linetype": [1], "alpha": [None],
    })
    grob = GeomSegment.draw_panel(GeomSegment(), df, PP, CART)
    # Linear path: a single segments grob, no polyline subdivision.
    assert grob._grid_class in ("segments", "segment")


# ---------------------------------------------------------------------------
# GeomRibbon / GeomArea — munched polygon under polar.
# ---------------------------------------------------------------------------

def test_geomribbon_polar_munches():
    df = pd.DataFrame({
        "x": [0.1, 0.4, 0.7, 0.9],
        "ymin": [0.0, 0.1, 0.05, 0.0],
        "ymax": [0.5, 0.8, 0.6, 0.4],
        "fill": ["grey"] * 4, "colour": [None] * 4,
        "linewidth": [0.5] * 4, "linetype": [1] * 4, "alpha": [None] * 4,
    })
    grob = GeomRibbon.draw_group(GeomRibbon(), df, PP, POLAR)
    poly = _find_grobs(grob, classes=("polygon",))
    assert poly, "polar ribbon draws a polygon"
    # upper + lower edges both munched → far more than 2*4 vertices
    assert len(_unit_values(poly[0].x)) > 8


def test_geomribbon_cartesian_unchanged():
    df = pd.DataFrame({
        "x": [0.1, 0.4, 0.7], "ymin": [0.0, 0.1, 0.05], "ymax": [0.5, 0.8, 0.6],
        "fill": ["grey"] * 3, "colour": [None] * 3,
        "linewidth": [0.5] * 3, "linetype": [1] * 3, "alpha": [None] * 3,
    })
    grob = GeomRibbon.draw_group(GeomRibbon(), df, PP, CART)
    poly = _find_grobs(grob, classes=("polygon",))[0]
    # Cartesian: upper(3) + lower(3) = 6 vertices, no munching.
    assert len(_unit_values(poly.x)) == 6


# ---------------------------------------------------------------------------
# GeomRaster — R-faithful: warns + falls back to GeomRect (NOT an abort).
# ---------------------------------------------------------------------------

def test_geomraster_polar_falls_back_to_rect_polygon():
    df = pd.DataFrame({
        "x": [1, 2, 1, 2], "y": [1, 1, 2, 2],
        "xmin": [0.5, 1.5, 0.5, 1.5], "xmax": [1.5, 2.5, 1.5, 2.5],
        "ymin": [0.5, 0.5, 1.5, 1.5], "ymax": [1.5, 1.5, 2.5, 2.5],
        "fill": ["#111111", "#222222", "#333333", "#444444"],
        "alpha": [None] * 4,
    })
    ppr = {"theta.range": [0, 3], "r.range": [0, 3]}
    grob = GeomRaster.draw_panel(GeomRaster(), df, ppr, POLAR)
    # R falls back to GeomRect → under polar that munches to polygons.
    polys = _find_grobs(grob, classes=("polygon",))
    assert polys, "geom_raster under polar must fall back to a (munched) polygon"
    # It must NOT raise / abort.


def test_geomraster_cartesian_stays_raster():
    df = pd.DataFrame({
        "x": [1, 2, 1, 2], "y": [1, 1, 2, 2],
        "xmin": [0.5, 1.5, 0.5, 1.5], "xmax": [1.5, 2.5, 1.5, 2.5],
        "ymin": [0.5, 0.5, 1.5, 1.5], "ymax": [1.5, 1.5, 2.5, 2.5],
        "fill": ["#111111", "#222222", "#333333", "#444444"],
        "alpha": [None] * 4,
    })
    ppr = {"x.range": [0, 3], "y.range": [0, 3]}
    grob = GeomRaster.draw_panel(GeomRaster(), df, ppr, CART)
    assert grob._grid_class in ("raster", "rastergrob")


# ---------------------------------------------------------------------------
# End-to-end: a polar bar chart is a pie of munched wedges (the linchpin).
# ---------------------------------------------------------------------------

def test_polar_bar_pie_builds_and_bends():
    from ggplot2_py import ggplot, aes, geom_bar, coord_polar
    from ggplot2_py.plot_render import ggplotGrob

    df = pd.DataFrame({"cat": ["a", "b", "c", "a", "b", "c"]})
    df["xcol"] = 1
    p = (ggplot(df, aes(x="xcol", fill="cat"))
         + geom_bar(position="fill")
         + coord_polar("y"))
    grob = ggplotGrob(p)  # must build without error

    polys = _find_grobs(grob, classes=("polygon",))
    # Find the bar layer polygon (many munched points, 3 wedge ids).
    bar_polys = [pg for pg in polys
                 if len(_unit_values(pg.x)) > 20
                 and pg.id is not None
                 and len(set(np.asarray(pg.id).tolist())) == 3]
    assert bar_polys, "polar bar layer must be a 3-wedge munched polygon"
    bar = bar_polys[0]
    fills = bar.gp.params.get("fill")
    assert len(set(fills)) == 3, "each wedge keeps its own fill"


def test_discrete_position_dimension_includes_range_c():
    """``ScaleDiscretePosition.dimension`` must union the continuous range.

    R ``Scale$dimension`` for discrete position scales is
    ``expand_limits_scale`` → ``expand_limits_discrete_trans``: it unions
    the (expanded) discrete range with the *un-expanded* continuous
    ``range_c`` (trained from a bar's ``xmin``/``xmax`` widths).  An
    earlier port dropped ``range_c``; with the polar r-axis expansion
    ``(0, 0)`` that collapsed the r-range to a single point, producing
    zero-area wedges.  Here a single level (range_c = [0.55, 1.45]) with
    *no* expansion must report ``[0.55, 1.45]`` rather than ``[1, 1]``.
    """
    from ggplot2_py.scales import scale_x_discrete

    sc = scale_x_discrete()
    sc.train(["all"])          # discrete level -> position 1
    sc.train([0.55, 1.45])     # continuous bar extent -> range_c
    dim = np.asarray(sc.dimension(expand=np.array([0.0, 0.0, 0.0, 0.0])),
                     dtype=float)
    assert abs(dim[0] - 0.55) < 1e-9
    assert abs(dim[1] - 1.45) < 1e-9


def test_polar_pie_rrange_matches_r():
    """``coord_polar('y')`` r.range for a single-category fill-bar == R.

    R (ggplot_build of ``geom_bar(position='fill') + coord_polar('y')``
    on a single discrete level): ``panel_params$r.range == c(0.55, 1.45)``
    (the bar's xmin/xmax extent, since the r-axis expansion is (0, 0)).
    """
    from ggplot2_py import ggplot, aes, geom_bar, coord_polar
    from ggplot2_py.plot import ggplot_build

    df = pd.DataFrame({"cat": ["a", "b", "c", "a", "b", "c"]})
    df["xcol"] = "all"
    p = (ggplot(df, aes(x="xcol", fill="cat"))
         + geom_bar(position="fill")
         + coord_polar("y"))
    built = ggplot_build(p)
    rr = np.asarray(built.layout.panel_params[0]["r.range"], dtype=float)
    assert abs(rr[0] - 0.55) < 1e-9
    assert abs(rr[1] - 1.45) < 1e-9


def test_polar_pie_wedges_have_nonzero_area_inside_circle():
    """The fix: single-category pie wedges fill the ring (non-zero area).

    Regression guard for the wedges-don't-show bug.  When the discrete
    r-axis collapsed to a zero-width range, every munched wedge traced
    out-and-back along the same radius (shoelace area == 0) and rendered
    nothing.  With the range_c fix each wedge:

      * encloses non-trivial area (shoelace > 0), and
      * lies inside the panel unit circle (radius from centre (0.5,0.5)
        never exceeds R's outer ring radius 0.45 by more than rounding).
    """
    from ggplot2_py import ggplot, aes, geom_bar, coord_polar
    from ggplot2_py.plot_render import ggplotGrob

    df = pd.DataFrame({"cat": ["a", "b", "c", "a", "b", "c", "a"]})
    df["xcol"] = "all"
    p = (ggplot(df, aes(x="xcol", fill="cat"))
         + geom_bar(position="fill")
         + coord_polar("y"))
    grob = ggplotGrob(p)

    polys = _find_grobs(grob, classes=("polygon",))
    bar_polys = [pg for pg in polys
                 if pg.id is not None
                 and len(_unit_values(pg.x)) > 20]
    assert bar_polys, "polar bar layer must be a munched wedge polygon"
    bar = bar_polys[0]
    x = _unit_values(bar.x)
    y = _unit_values(bar.y)
    ids = np.asarray(bar.id)

    radius = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    # Wedges fill the ring: the outer edge reaches near R's 0.45 radius
    # (R r_rescale maps r into the donut c(0, 0.4) over r.range, + 0.5
    # offset; outer ring of render_bg is 0.45).  Must clearly exceed the
    # collapsed-bug radius of ~0.0.
    assert radius.max() > 0.25, "wedges must extend out toward the ring"
    # And stay inside the panel unit circle / grill (radius <= 0.45-ish).
    assert radius.max() <= 0.45 + 1e-6, "wedges must stay within the panel circle"

    # Every wedge encloses non-zero area (the actual bug: shoelace == 0).
    for uid in np.unique(ids):
        m = ids == uid
        xs, ys = x[m], y[m]
        area = 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
        assert area > 1e-4, f"wedge {uid} collapsed to zero area"


def test_polar_bar_munch_matches_rect_wedge_geometry():
    """The munched wedge for a quarter rect matches R's coord_munch shape.

    R reference (coord_munch(coord_polar('x'), rect-as-polygon, is_closed=TRUE)):
        head x = 0.5, 0.51358, 0.52715, ...
        head y = 0.9, 0.89977, 0.89908, ...
    The point *count* is owned by Workstream B (segment_length); here we
    assert the trajectory (shape) agrees to a loose tolerance.
    """
    df = pd.DataFrame({"x": [0.0, 0.5, 0.5, 0.0],
                       "y": [1.0, 1.0, 0.0, 0.0],
                       "group": [1, 1, 1, 1]})
    m = coord_munch(POLAR, df, PP, is_closed=True)
    x = m["x"].to_numpy()
    y = m["y"].to_numpy()
    # Starts at the top of the outer arc.
    assert abs(x[0] - 0.5) < 1e-6
    assert abs(y[0] - 0.9) < 1e-6
    # Sweeps an outer arc (x increases, y barely drops) — matches R head.
    assert x[3] > x[1] > x[0]
    assert y[1] < y[0] and y[3] < y[1]
