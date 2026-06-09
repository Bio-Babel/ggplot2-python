"""Recipe: scatter + smoother + facet_wrap on the mpg dataset.

The four-line ggplot template that 70%+ of publication-quality ggplots
follow: data + geom + facet + theme. Idiom `ggplot2_py.ggplot_geom_facet_theme`.
"""

from __future__ import annotations

from pathlib import Path

from ggplot2_py import (
    aes,
    facet_wrap,
    geom_point,
    geom_smooth,
    ggplot,
    ggsave,
    labs,
    theme_minimal,
)
from ggplot2_py.datasets import mpg


def main(out_path: Path = Path("scatter_smooth_facet.png")) -> Path:
    p = (
        ggplot(mpg, aes(x="displ", y="hwy", color="class"))
        + geom_point(alpha=0.6)
        + geom_smooth(method="loess", se=False)
        + facet_wrap("drv")
        + labs(
            title="Fuel efficiency by engine displacement",
            x="Displacement (L)",
            y="Highway MPG",
            color="Vehicle class",
        )
        + theme_minimal()
    )
    ggsave(str(out_path), plot=p, width=9, height=4, dpi=150)
    return out_path


if __name__ == "__main__":
    print(f"wrote {main().resolve()}")
