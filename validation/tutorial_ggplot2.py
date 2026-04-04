#!/usr/bin/env python
"""Validation: ggplot2 tutorial (vignettes/ggplot2.qmd)

Compares Python port outputs against R reference values for the introductory
tutorial.  Uses the canonical ResultRecorder to write
``validation/results_ggplot2.csv``.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from _recorder import ResultRecorder

import numpy as np
import pandas as pd

from ggplot2_py import (
    ggplot, is_ggplot, aes, is_mapping,
    geom_point, geom_smooth, geom_bar, geom_histogram,
    scale_colour_viridis_c, scale_colour_viridis_d,
    facet_grid, facet_wrap,
    coord_fixed,
    theme, theme_minimal, theme_bw,
    element_blank, element_line, element_text,
    labs,
    Mapping, Layer,
)
from ggplot2_py.plot import GGPlot
from ggplot2_py.datasets import mpg
from ggplot2_py.theme import Theme


def _geom_name(layer) -> str:
    g = layer.geom
    return getattr(g, "__name__", type(g).__name__)


def _stat_name(layer) -> str:
    s = layer.stat
    if isinstance(s, str):
        return "Stat" + s.capitalize()
    return getattr(s, "__name__", type(s).__name__)


def main():
    rec = ResultRecorder("ggplot2")

    # ==== Section 1: Data (mpg dataset) ====
    rec.record("mpg_nrow", "exact", value=len(mpg), threshold=234, tier=1)
    rec.record("mpg_ncol", "exact", value=len(mpg.columns), threshold=11, tier=1)

    r_cols = ["manufacturer", "model", "displ", "year", "cyl", "trans",
              "drv", "cty", "hwy", "fl", "class"]
    rec.record("mpg_colnames", "exact",
               value=list(mpg.columns), threshold=r_cols, tier=1)

    rec.record("mpg_displ_head5", "exact",
               value=mpg["displ"].head(5).tolist(),
               threshold=[1.8, 1.8, 2.0, 2.0, 2.8], tier=1)

    rec.record("mpg_hwy_min", "exact",
               value=int(mpg["hwy"].min()), threshold=12, tier=1)
    rec.record("mpg_hwy_max", "exact",
               value=int(mpg["hwy"].max()), threshold=44, tier=1)

    rec.record("mpg_cty_mean", "numeric_tol",
               value=float(mpg["cty"].mean()),
               threshold=(16.85897, 0.001), tier=1)

    rec.record("mpg_drv_unique", "exact",
               value=sorted(mpg["drv"].unique().tolist()),
               threshold=sorted(["f", "4", "r"]), tier=1)

    rec.record("mpg_year_unique", "exact",
               value=sorted(mpg["year"].unique().tolist()),
               threshold=[1999, 2008], tier=1)

    # ==== Section 2: Mapping (aes()) ====
    m = aes(x="cty", y="hwy")
    rec.record("aes_is_mapping", "bool",
               value=is_mapping(m), threshold=True, tier=1)
    rec.record("aes_x_value", "exact",
               value=m.get("x"), threshold="cty", tier=1)
    rec.record("aes_y_value", "exact",
               value=m.get("y"), threshold="hwy", tier=1)
    rec.record("aes_len", "exact",
               value=len(m), threshold=2, tier=1)

    m2 = aes(x="cty", y="hwy", colour="class")
    rec.record("aes_colour_value", "exact",
               value=m2.get("colour"), threshold="class", tier=1)
    rec.record("aes_colour_len", "exact",
               value=len(m2), threshold=3, tier=1)

    # ==== Section 3: Layers ====
    p1 = ggplot(mpg, aes(x="cty", y="hwy")) + geom_point()
    rec.record("scatter_is_ggplot", "bool",
               value=is_ggplot(p1), threshold=True, tier=1)
    rec.record("scatter_nlayers", "exact",
               value=len(p1.layers), threshold=1, tier=1)
    rec.record("scatter_geom", "exact",
               value=_geom_name(p1.layers[0]), threshold="GeomPoint", tier=1)
    rec.record("scatter_stat", "exact",
               value=_stat_name(p1.layers[0]), threshold="StatIdentity", tier=1)

    p2 = ggplot(mpg, aes(x="cty", y="hwy")) + geom_point() + geom_smooth()
    rec.record("smooth_nlayers", "exact",
               value=len(p2.layers), threshold=2, tier=1)
    rec.record("smooth_geom0", "exact",
               value=_geom_name(p2.layers[0]), threshold="GeomPoint", tier=1)
    rec.record("smooth_geom1", "exact",
               value=_geom_name(p2.layers[1]), threshold="GeomSmooth", tier=1)
    rec.record("smooth_stat1", "exact",
               value=_stat_name(p2.layers[1]), threshold="StatSmooth", tier=1)

    rec.record("plot_mapping_x", "exact",
               value=p2.mapping.get("x"), threshold="cty", tier=1)
    rec.record("plot_mapping_y", "exact",
               value=p2.mapping.get("y"), threshold="hwy", tier=1)
    rec.record("plot_data_nrow", "exact",
               value=len(p2.data), threshold=234, tier=1)

    # ==== Section 4: Scales ====
    p3 = (ggplot(mpg, aes(x="cty", y="hwy", colour="class"))
          + geom_point() + scale_colour_viridis_d())
    has_colour = any(
        s.aesthetics[0] == "colour"
        if hasattr(s, "aesthetics") and s.aesthetics else False
        for s in (p3.scales.scales if hasattr(p3.scales, "scales") else [])
    )
    rec.record("viridis_d_colour_scale", "bool",
               value=has_colour, threshold=True, tier=1)

    # ==== Section 5: Facets ====
    p4 = (ggplot(mpg, aes(x="cty", y="hwy"))
          + geom_point() + facet_grid(rows="year", cols="drv"))
    rec.record("facet_grid_type", "exact",
               value=type(p4.facet).__name__, threshold="FacetGrid", tier=1)
    fparams = getattr(p4.facet, "params", {})
    rec.record("facet_grid_has_rows", "bool",
               value="rows" in (fparams or {}), threshold=True, tier=1)
    rec.record("facet_grid_has_cols", "bool",
               value="cols" in (fparams or {}), threshold=True, tier=1)

    # ==== Section 6: Coordinates ====
    p5 = ggplot(mpg, aes(x="cty", y="hwy")) + geom_point() + coord_fixed()
    rec.record("coord_fixed_type", "exact",
               value=type(p5.coordinates).__name__,
               threshold="CoordFixed", tier=1)
    ratio = getattr(p5.coordinates, "ratio", None)
    rec.record("coord_fixed_ratio", "numeric_tol",
               value=float(ratio) if ratio is not None else -1,
               threshold=(1.0, 0.001), tier=1)

    # ==== Section 7: Theme ====
    t_min = theme_minimal()
    rec.record("theme_minimal_is_theme", "bool",
               value=isinstance(t_min, Theme), threshold=True, tier=1)
    rec.record("theme_minimal_complete", "bool",
               value=getattr(t_min, "complete", False), threshold=True, tier=1)

    t_custom = theme(legend_position="top")
    rec.record("theme_custom_is_theme", "bool",
               value=isinstance(t_custom, Theme), threshold=True, tier=1)

    eb = element_blank()
    rec.record("element_blank_type", "exact",
               value=type(eb).__name__, threshold="ElementBlank", tier=2)

    el = element_line(linewidth=0.75)
    rec.record("element_line_is_element", "bool",
               value="Element" in type(el).__name__, threshold=True, tier=2)

    # ==== Section 8: Combined plot ====
    p_final = (
        ggplot(mpg, aes(x="cty", y="hwy"))
        + geom_point(aes(colour="displ"))
        + geom_smooth()
        + scale_colour_viridis_c()
        + facet_grid(rows="year", cols="drv")
        + coord_fixed()
        + theme_minimal()
        + theme(panel_grid_minor=element_blank())
    )
    rec.record("final_is_ggplot", "bool",
               value=is_ggplot(p_final), threshold=True, tier=1)
    rec.record("final_nlayers", "exact",
               value=len(p_final.layers), threshold=2, tier=1)
    rec.record("final_geom0", "exact",
               value=_geom_name(p_final.layers[0]), threshold="GeomPoint", tier=1)
    rec.record("final_geom1", "exact",
               value=_geom_name(p_final.layers[1]), threshold="GeomSmooth", tier=1)
    rec.record("final_coord", "exact",
               value=type(p_final.coordinates).__name__,
               threshold="CoordFixed", tier=1)
    rec.record("final_facet", "exact",
               value=type(p_final.facet).__name__,
               threshold="FacetGrid", tier=1)
    rec.record("final_has_theme", "bool",
               value=(p_final.theme is not None
                      and isinstance(p_final.theme, (dict, Theme))
                      and len(p_final.theme) > 0),
               threshold=True, tier=1)
    rec.record("final_mapping_x", "exact",
               value=p_final.mapping.get("x"), threshold="cty", tier=1)
    rec.record("final_mapping_y", "exact",
               value=p_final.mapping.get("y"), threshold="hwy", tier=1)

    layer0_mapping = p_final.layers[0].mapping or {}
    rec.record("final_layer0_colour", "exact",
               value=layer0_mapping.get("colour"),
               threshold="displ", tier=1)

    # ==== Save ====
    rec.save()
    print(rec.summary())
    failed = sum(1 for r in rec.records if r["pass"] != "true")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
