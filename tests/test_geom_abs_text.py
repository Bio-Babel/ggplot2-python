"""Tests for ``GeomAbsText`` / ``geom_abs_text`` — npc-positioned text badges."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ggplot2_py as gg
from ggplot2_py.geom import GeomAbsText


def test_geom_abs_text_registered_and_aes():
    g = GeomAbsText()
    assert g.required_aes == ("xpos", "ypos", "label")
    # inherits text default aes (so colour/size/hjust/... are available)
    for a in ("colour", "size", "hjust", "vjust", "angle"):
        assert a in g.default_aes


def test_geom_abs_text_paints_at_npc_corner(tmp_path):
    """The badge must paint at its npc corner and leave the rest untouched."""
    Image = pytest.importorskip("PIL.Image")
    df = pd.DataFrame({"x": np.arange(10), "y": np.arange(10)})
    base = gg.ggplot(df, gg.aes("x", "y")) + gg.geom_point()
    badge = pd.DataFrame({"xpos": [0.02], "ypos": [0.95], "label": ["BADGE"]})
    withb = base + gg.geom_abs_text(
        gg.aes(xpos="xpos", ypos="ypos", label="label"),
        data=badge, inherit_aes=False, hjust=0.0, size=6,
    )
    nb, wb = tmp_path / "nb.png", tmp_path / "wb.png"
    gg.ggsave(str(nb), base, width=4, height=3, dpi=100)
    gg.ggsave(str(wb), withb, width=4, height=3, dpi=100)
    a = np.asarray(Image.open(nb).convert("L"), float)
    b = np.asarray(Image.open(wb).convert("L"), float)
    diff = np.abs(a - b)
    H, W = diff.shape
    assert (diff[: H // 4, : W // 2] > 20).sum() > 30  # painted top-left
    assert (diff[3 * H // 4 :, W // 2 :] > 20).sum() == 0  # nothing leaked
