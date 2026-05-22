"""Standalone OO-path tests for Guide subclasses.

Each test asserts that calling ``guide.train(scale=...)`` then
``guide.draw(theme=...)`` produces a non-empty ``Gtable`` — i.e. the OO
pipeline (the R-equivalent ``Guide$draw`` orchestration via ``self.method``
dispatch) is wired correctly for each guide class.

These tests do NOT exercise plot_render; they probe the Guide classes
directly. Visual rendering identity vs the procedural plot_render path is
verified separately (see compare_internals workflow on the ggnewscale port).
"""

from __future__ import annotations

import pandas as pd
import pytest

import ggplot2_py as gg
from ggplot2_py.theme_defaults import theme_grey


def _build_with_scale(df, mapping, scale):
    g = gg.ggplot(df, mapping) + gg.geom_point(size=4) + scale
    return gg.ggplot_build(g)


class TestGuideColourbarOO:
    """guide_colourbar().train(...).draw(...) end-to-end."""

    def test_returns_gtable_with_bar_labels_background(self):
        df = pd.DataFrame({"x": range(10), "y": range(10), "z": range(10)})
        built = _build_with_scale(
            df, gg.aes("x", "y", color="z"), gg.scale_color_viridis_c()
        )
        sc = built.plot.scales.scales[0]
        guide = sc.guide if hasattr(sc.guide, "_class_name") else gg.GuideColourbar()
        params = guide.train(scale=sc, aesthetic="colour", params=dict(guide.params))
        assert params is not None
        params["direction"] = "vertical"
        gt = guide.draw(
            theme=theme_grey(), position="right", direction="vertical", params=params,
        )
        assert gt is not None
        assert type(gt).__name__ == "Gtable"
        names = [getattr(g, "name", "") for g in gt.grobs]
        assert any("bar" in n for n in names), names
        assert any("labels" in n for n in names), names

    def test_key_value_rescaled_to_unit_range(self):
        df = pd.DataFrame({"x": range(10), "y": range(10), "z": range(10)})
        built = _build_with_scale(
            df, gg.aes("x", "y", color="z"), gg.scale_color_viridis_c()
        )
        sc = built.plot.scales.scales[0]
        guide = gg.GuideColourbar()
        params = guide.train(scale=sc, aesthetic="colour", params=dict(guide.params))
        # R extract_params rescales .value to (0.5/nbin, (nbin-0.5)/nbin) for
        # non-gradient display.
        v = params["key"][".value"].values
        assert v.min() >= 0.0
        assert v.max() <= 1.0


class TestGuideLegendOO:
    """guide_legend().train(...).process_layers(...).draw(...) end-to-end."""

    def test_returns_gtable_with_keys_labels_background(self):
        df = pd.DataFrame(
            {"x": [1, 2, 3, 4], "y": [1, 3, 2, 4], "cat": ["a", "b", "a", "b"]}
        )
        g = (
            gg.ggplot(df, gg.aes("x", "y"))
            + gg.geom_point(gg.aes(color="cat"), size=4)
            + gg.scale_color_brewer(palette="Set1")
        )
        built = gg.ggplot_build(g)
        sc = built.plot.scales.scales[0]
        guide = gg.GuideLegend()
        params = guide.train(scale=sc, aesthetic="colour", params=dict(guide.params))
        assert params is not None
        params["direction"] = "vertical"
        params = guide.process_layers(params, layers=built.plot.layers, theme=theme_grey())
        gt = guide.draw(
            theme=theme_grey(), position="right", direction="vertical", params=params,
        )
        assert gt is not None
        assert type(gt).__name__ == "Gtable"
        assert len(gt.grobs) > 0

    def test_setup_params_computes_nrow_ncol(self):
        # R guide-legend.R:286-298: vertical direction defaults to
        # ncol = ceiling(n_breaks / 20); horizontal to nrow = ceiling(n_breaks / 5).
        params = gg.GuideLegend.setup_params(
            {"direction": "horizontal", "key": list(range(13))}
        )
        # horizontal, 13 breaks → nrow = ceil(13/5) = 3, ncol = ceil(13/3) = 5
        assert params["nrow"] == 3
        assert params["ncol"] == 5

        params = gg.GuideLegend.setup_params(
            {"direction": "vertical", "key": list(range(25))}
        )
        # vertical, 25 breaks → ncol = ceil(25/20) = 2, nrow = ceil(25/2) = 13
        assert params["ncol"] == 2
        assert params["nrow"] == 13


class TestGuideColourbarParamsParity:
    """extract_params behaviour against R semantics."""

    def test_extract_params_picks_scale_name_when_no_user_title(self):
        df = pd.DataFrame({"x": range(5), "y": range(5), "z": range(5)})
        g = (
            gg.ggplot(df, gg.aes("x", "y"))
            + gg.geom_point(gg.aes(color="z"))
            + gg.scale_color_viridis_c(name="my-title")
        )
        built = gg.ggplot_build(g)
        sc = built.plot.scales.scales[0]
        guide = gg.GuideColourbar()
        params = guide.train(scale=sc, aesthetic="colour", params=dict(guide.params))
        assert params["title"] == "my-title"
