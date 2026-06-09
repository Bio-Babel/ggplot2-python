---
name: use-ggplot2-py
description: Grammar of graphics for Python — the canonical layered plot DSL ported 1:1 from R ggplot2.
---

# ggplot2-python

Python port of [tidyverse/ggplot2](https://github.com/tidyverse/ggplot2) (v4.0.2.9000). Same grammar, same operators, same dataset (`mpg`, `diamonds`, `economics`, etc.). The one user-facing departure: aesthetics use **string column names with keyword args**, because Python has no Non-Standard Evaluation.

## Mental model — 60 seconds

ggplot2 is a *builder DSL*. Five truths:

1. **A plot is a value, not a side effect.** `p = ggplot(df, aes(x="x", y="y"))` is a `GGPlot`. Nothing is drawn until `print(p)` or `p.draw()`.

2. **`+` is accumulation, not addition.** `p + geom_point()` returns a *new* GGPlot with the layer appended. Each `+ X` chooses a slot based on the type of `X` (Layer, Scale, Theme, Facet, Coord, Labels, Mapping).

3. **`aes()` binds columns to visual properties.** Inside `aes()`: strings (column references). Outside `aes()`: constants. Confusing the two is the #1 ggplot bug — see anti-pattern `ggplot2_py.constant_inside_aes`.

4. **Layers inherit by default.** `ggplot(df, aes(...))` defines the *default* data and mapping; each layer can either inherit, override, or fully isolate with `inherit_aes=False`.

5. **Scales come in three pieces.** Customizing an axis means transform + breaks + labels together, not one in isolation. Reach for `scale_x_log10()` etc. as ergonomic shortcuts.

## The two cardinal sins

1. **Unquoted column names**: `aes(displ, hwy)` works in R via NSE, fails or silently misbehaves in Python. Always `aes(x="displ", y="hwy")`.

2. **Constants inside aes()**: `aes(color="red")` does *not* paint points red — it maps to a phantom "red" group and produces a one-entry legend. The fix is to move the kwarg outside aes: `geom_point(color="red")`.

## Quick reference — the canonical 4-line plot

```python
from ggplot2_py import ggplot, aes, geom_point, geom_smooth, facet_wrap, theme_minimal, labs
from ggplot2_py.datasets import mpg

p = (
    ggplot(mpg, aes(x="displ", y="hwy", color="class"))
    + geom_point(alpha=0.6)
    + geom_smooth(method="loess", se=False)
    + facet_wrap("drv")
    + labs(title="Fuel efficiency by engine displacement",
           x="Displacement (L)", y="Highway MPG")
    + theme_minimal()
)
print(p)  # writes to display / file
```

## When NOT to reach for ggplot2-python

- **One-off scatter of two NumPy arrays** — matplotlib is faster to type.
- **Interactive widgets / web app** — use plotly / bokeh.
- **3D plots** — ggplot2 is 2D only by design.
- **Very large data (>1M points)** — geom_hex / geom_density_2d, or downsample first.

For more: `biobabel.list_idioms(package="ggplot2_py")`, `biobabel.describe_concept("ggplot2_py.plus_accumulation")`, `biobabel.describe_symbol(symbol_id="ggplot2_py.<name>")`.
