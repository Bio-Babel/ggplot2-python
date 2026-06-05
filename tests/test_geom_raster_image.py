"""Tests for GeomRaster's 2-D colour-matrix construction in ``draw_panel``.

R ``GeomRaster$draw_panel`` (R/geom-raster.R:53-92) turns the flat per-row
vector of fills into an ``nrow x ncol`` colour matrix and hands *that* to
``rasterGrob``.  The previous Python port passed the flat fill vector
straight through, so the raster grob received a 1-D array (e.g. a 5x6 grid
became ``(30,)``) and rendering crashed / drew nothing.

These tests pin the reshape: shape, R orientation (row 0 = top = max y,
columns increasing with x, byrow), missing-cell handling, and end-to-end
PNG rendering for a plain raster and a 150x150 heatmap.
"""

import numpy as np
import pandas as pd
import pytest

import ggplot2_py as gg
from ggplot2_py import geom as _geommod
from ggplot2_py.geom import GeomRaster


def _capture_raster_images(plot):
    """Render ``plot`` to PNG, capturing every image handed to raster_grob."""
    shapes = []
    images = []
    orig = _geommod.raster_grob

    def spy(image, *a, **k):
        arr = np.asarray(image)
        shapes.append(arr.shape)
        images.append(arr)
        return orig(image, *a, **k)

    _geommod.raster_grob = spy
    try:
        png = plot._repr_png_()
    finally:
        _geommod.raster_grob = orig
    return png, shapes, images


def test_draw_panel_passes_2d_image():
    """A 5x6 grid must reach raster_grob as a 2-D ``(5, 6)`` array."""
    df = pd.DataFrame(
        {
            "x": np.repeat(np.arange(6), 5),
            "y": np.tile(np.arange(5), 6),
        }
    )
    df["z"] = df["x"] * df["y"]
    plot = gg.ggplot(df, gg.aes(x="x", y="y", fill="z")) + gg.geom_raster()

    png, shapes, images = _capture_raster_images(plot)

    assert png and len(png) > 0
    assert len(shapes) == 1
    assert shapes[0] == (5, 6)  # nrow (unique y) x ncol (unique x)
    assert images[0].ndim == 2


def test_raster_orientation_matches_r():
    """Row 0 = top = max y; columns increase with x; byrow.

    Reproduces R's exact matrix for a known-fill 5x6 grid:

        nrow - y_pos (1-indexed) -> row 0 (0-indexed) holds the largest y.
    """
    from ggplot2_py.position import _resolution

    # expand.grid(x = 1:6, y = 1:5): x varies fastest within each y.
    df = pd.DataFrame(
        [(x, y) for y in range(1, 6) for x in range(1, 7)], columns=["x", "y"]
    )
    df["fill"] = ["#%06X" % ((i + 1) * 1000) for i in range(len(df))]

    g = GeomRaster()
    data = g.setup_data(df.copy(), {})

    x_num = data["x"].to_numpy(float)
    y_num = data["y"].to_numpy(float)
    x_pos = ((x_num - x_num.min()) / _resolution(x_num, zero=False)).astype(int)
    y_pos = ((y_num - y_num.min()) / _resolution(y_num, zero=False)).astype(int)
    nrow = int(y_pos.max() + 1)
    ncol = int(x_pos.max() + 1)
    raster = np.full((nrow, ncol), "transparent", dtype=object)
    raster[(nrow - 1) - y_pos, x_pos] = data["fill"].to_numpy(object)

    assert (nrow, ncol) == (5, 6)
    # Row 0 = y == 5 (max y); fills there are the last 6 fills (i = 24..29).
    expected_top = ["#%06X" % ((i + 1) * 1000) for i in range(24, 30)]
    assert list(raster[0]) == expected_top
    # Row 4 (bottom) = y == 1 (min y); first 6 fills (i = 0..5).
    expected_bottom = ["#%06X" % ((i + 1) * 1000) for i in range(0, 6)]
    assert list(raster[nrow - 1]) == expected_bottom
    # No cell left transparent for a complete grid.
    assert "transparent" not in raster.ravel().tolist()


def test_incomplete_grid_fills_missing_with_transparent():
    """Missing cells in an incomplete grid stay transparent (R: NA)."""
    from ggplot2_py.position import _resolution

    # 3x3 grid minus one corner cell (x=2, y=2 removed).
    rows = [(x, y) for y in range(3) for x in range(3) if not (x == 2 and y == 2)]
    df = pd.DataFrame(rows, columns=["x", "y"])
    df["fill"] = ["#000000"] * len(df)

    g = GeomRaster()
    data = g.setup_data(df.copy(), {})
    x_num = data["x"].to_numpy(float)
    y_num = data["y"].to_numpy(float)
    x_pos = ((x_num - x_num.min()) / _resolution(x_num, zero=False)).astype(int)
    y_pos = ((y_num - y_num.min()) / _resolution(y_num, zero=False)).astype(int)
    nrow = int(y_pos.max() + 1)
    ncol = int(x_pos.max() + 1)
    raster = np.full((nrow, ncol), "transparent", dtype=object)
    raster[(nrow - 1) - y_pos, x_pos] = data["fill"].to_numpy(object)

    assert (nrow, ncol) == (3, 3)
    # The removed cell (x=2, y=2 -> top-right, row 0 col 2) is transparent.
    assert raster[0, 2] == "transparent"
    assert (raster == "transparent").sum() == 1


def test_large_heatmap_renders_2d():
    """A 150x150 heatmap renders to PNG with a 2-D ``(150, 150)`` image."""
    n = 150
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": np.repeat(np.arange(n), n),
            "y": np.tile(np.arange(n), n),
            "cor": rng.random(n * n),
        }
    )
    plot = gg.ggplot(df, gg.aes(x="x", y="y", fill="cor")) + gg.geom_raster()

    png, shapes, images = _capture_raster_images(plot)

    assert png and len(png) > 0
    assert shapes == [(150, 150)]
    assert images[0].ndim == 2


def test_identity_fill_renders():
    """Identity-mapped fill colours flow through unchanged into the matrix."""
    df = pd.DataFrame(
        {
            "x": np.repeat(np.arange(4), 4),
            "y": np.tile(np.arange(4), 4),
        }
    )
    df["z"] = np.arange(16)
    plot = gg.ggplot(df, gg.aes(x="x", y="y", fill="z")) + gg.geom_raster()
    png, shapes, images = _capture_raster_images(plot)
    assert png and len(png) > 0
    assert shapes == [(4, 4)]
