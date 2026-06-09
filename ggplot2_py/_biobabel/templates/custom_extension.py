"""Recipe: extend ggplot2_py — a custom Stat + a custom `+`-type.

Two extension mechanisms in one runnable figure, both from
``tutorials/extending_ggplot2.ipynb`` and the README:

1. **Custom Stat** — subclass ``ggplot2_py.stat.Stat``, declare
   ``required_aes``, and override ``compute_group(self, data, scales,
   **params)`` to return a NEW DataFrame whose columns become the geom's
   aesthetics. Here ``StatLm`` fits a least-squares polynomial per group and
   returns the fitted line. A thin ``stat_lm()`` wraps it in a ``layer()`` so
   it reads like a built-in ``stat_*()`` constructor.

2. **Custom `+`-type** — ``GGPlot.__add__`` dispatches through the
   ``update_ggplot`` ``functools.singledispatch`` generic, so ANY plain
   Python class can be made addable with ``+`` by registering a handler with
   ``@update_ggplot.register(MyType)``. ``Watermark`` stamps a caption when
   added — no subclassing, no metaclass.

Both are combined in one plot and written with ``ggsave``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ggplot2_py import aes, geom_point, ggplot, ggsave, labs, theme_minimal
from ggplot2_py.datasets import mpg
from ggplot2_py.layer import layer as Layer
from ggplot2_py.plot import update_ggplot
from ggplot2_py.stat import Stat


# --- 1. Custom Stat: override compute_group ------------------------------
class StatLm(Stat):
    """Least-squares polynomial fit, computed per group."""

    required_aes = ["x", "y"]

    def compute_group(self, data, scales, n=100, degree=1, **kwargs):
        x = data["x"].values.astype(float)
        y = data["y"].values.astype(float)
        coeffs = np.polyfit(x, y, degree)
        x_grid = np.linspace(x.min(), x.max(), n)
        return pd.DataFrame({"x": x_grid, "y": np.polyval(coeffs, x_grid)})


def stat_lm(
    mapping=None,
    data=None,
    geom="line",
    position="identity",
    n=100,
    degree=1,
    **kwargs,
):
    """``stat_*()``-style layer wrapper around :class:`StatLm`."""
    return Layer(
        stat=StatLm,
        geom=geom,
        data=data,
        mapping=mapping,
        position=position,
        params={"n": n, "degree": degree, **kwargs},
    )


# --- 2. Custom `+`-type: register a handler for the + operator -----------
class Watermark:
    """A plain class that, when added to a plot, stamps a caption."""

    def __init__(self, text: str):
        self.text = text


@update_ggplot.register(Watermark)
def _add_watermark(obj: Watermark, plot, object_name: str = ""):
    """Fires on ``plot + Watermark(...)``; injects the text as a caption."""
    return plot + labs(caption=f"[{obj.text}]")


def main(out_path: Path = Path("custom_extension.png")) -> Path:
    p = (
        ggplot(mpg, aes("displ", "hwy"))
        + geom_point(size=1.2, alpha=0.6)
        + stat_lm(degree=2, color="firebrick", linewidth=1.0)  # custom Stat
        + Watermark("custom stat + custom +-type")             # custom +-type
        + labs(
            title="Extending ggplot2_py",
            x="Engine displacement (L)",
            y="Highway MPG",
        )
        + theme_minimal()
    )
    assert p.labels.get("caption") == "[custom stat + custom +-type]"
    ggsave(str(out_path), plot=p, width=7, height=5, dpi=120)
    return out_path


if __name__ == "__main__":
    print(f"wrote {main().resolve()}")
