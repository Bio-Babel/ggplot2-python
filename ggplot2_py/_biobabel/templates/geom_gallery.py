"""Recipe: a geom gallery — one figure, many diverse geoms ("各种图性的画图").

Six panels on the `mpg` dataset (plus a couple of synthetic frames for the
continuous geoms), each showcasing a different geom family:

    point + colour | line + ribbon | col (bar)
    boxplot        | violin        | tile (heatmap)

ggplot2_py has no first-class plot-composition operator (``p1 + p2`` is an
error — see the ``plus_between_two_plots`` anti-pattern). So each panel is
built as its own ``GGPlot``, rendered to a ``Gtable`` via
``ggplot_gtable(ggplot_build(p))``, and the gtables are placed into the
cells of an outer ``Gtable`` with ``gtable_add_grob``. ``ggsave`` accepts any
gtable, so the composite is written like a normal plot.

Geom choices follow ``tutorials/geoms_gallery.ipynb``. Runs in well under a
second.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from grid_py import Unit
from gtable_py import Gtable, gtable_add_grob

from ggplot2_py import (
    aes,
    element_blank,
    element_text,
    geom_boxplot,
    geom_col,
    geom_line,
    geom_point,
    geom_ribbon,
    geom_tile,
    geom_violin,
    ggplot,
    ggsave,
    labs,
    scale_fill_viridis_c,
    theme,
    theme_minimal,
)
from ggplot2_py.datasets import mpg
from ggplot2_py.plot import ggplot_build, ggplot_gtable


def _as_gtable(p):
    """Render a GGPlot to a Gtable so it can be nested in a layout grid."""
    return ggplot_gtable(ggplot_build(p))


def _panels():
    base = theme_minimal()

    # Synthetic frames for the continuous geoms.
    summ = mpg.groupby("class", as_index=False)["hwy"].mean()
    x = np.linspace(0, 10, 80)
    line_df = pd.DataFrame({"x": x, "y": np.sin(x) + 2.0})
    band = pd.DataFrame({"x": x, "lo": np.sin(x) + 1.3, "hi": np.sin(x) + 2.7})
    tile_df = pd.DataFrame(
        {
            "x": np.repeat(range(6), 6),
            "y": np.tile(range(6), 6),
            "z": np.random.RandomState(1).randn(36),
        }
    )

    return [
        (
            ggplot(mpg, aes("displ", "hwy"))
            + geom_point(aes(color="drv"), size=0.9)
            + labs(title="point + colour")
            + base
            + theme(legend_position="none")
        ),
        (
            ggplot(line_df, aes("x", "y"))
            + geom_ribbon(
                data=band,
                mapping=aes(x="x", ymin="lo", ymax="hi"),
                inherit_aes=False,
                fill="steelblue",
                alpha=0.3,
            )
            + geom_line(color="firebrick")
            + labs(title="line + ribbon")
            + base
        ),
        (
            ggplot(summ, aes("class", "hwy"))
            + geom_col(fill="#4C72B0")
            + labs(title="col")
            + base
            + theme(axis_text_x=element_text(angle=45, hjust=1))
        ),
        (
            ggplot(mpg, aes("class", "hwy", fill="class"))
            + geom_boxplot()
            + labs(title="boxplot")
            + base
            + theme(legend_position="none", axis_text_x=element_blank())
        ),
        (
            ggplot(mpg, aes("class", "hwy", fill="class"))
            + geom_violin(alpha=0.7)
            + labs(title="violin")
            + base
            + theme(legend_position="none", axis_text_x=element_blank())
        ),
        (
            ggplot(tile_df, aes("x", "y", fill="z"))
            + geom_tile()
            + scale_fill_viridis_c()
            + labs(title="tile")
            + base
        ),
    ]


def main(out_path: Path = Path("geom_gallery.png")) -> Path:
    panels = _panels()
    ncol, nrow = 3, 2
    outer = Gtable(
        widths=Unit([1] * ncol, "null"),
        heights=Unit([1] * nrow, "null"),
        name="geom_gallery",
    )
    for i, p in enumerate(panels):
        r, c = i // ncol + 1, i % ncol + 1
        outer = gtable_add_grob(outer, [_as_gtable(p)], t=r, l=c, name=f"panel_{i}")

    ggsave(str(out_path), plot=outer, width=12, height=7, dpi=110)
    return out_path


if __name__ == "__main__":
    print(f"wrote {main().resolve()}")
