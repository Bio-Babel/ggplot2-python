# ggplot2_py

A Python port of the R ggplot2 package — Create Elegant Data Visualisations Using the Grammar of Graphics.

## Overview

ggplot2_py implements the grammar of graphics in Python, providing a layered approach to creating statistical visualizations. It is a faithful port of R's ggplot2 package, using pandas DataFrames as the primary data container and matplotlib as the rendering backend.

Beyond a direct port, ggplot2_py adds **Python-exclusive features** that extend the Grammar of Graphics with Python-native idioms.

## Quick Start

```python
from ggplot2_py import *
from ggplot2_py.datasets import mpg

p = (ggplot(mpg, aes(x='displ', y='hwy', colour='class'))
     + geom_point()
     + theme_minimal()
     + labs(title='Engine Displacement vs Highway MPG'))
```

## GOG Components (R-compatible)

| Component | Coverage |
|-----------|----------|
| **Aesthetics** | `aes()`, `after_stat()`, `after_scale()`, `stage()` |
| **Geoms** | 47 geometry layers (`geom_point`, `geom_bar`, `geom_boxplot`, ...) |
| **Stats** | 32 statistical transformations (`stat_bin`, `stat_smooth`, `stat_density`, ...) |
| **Scales** | 130+ scale functions for colour, size, shape, position, etc. |
| **Coordinates** | `coord_cartesian`, `coord_flip`, `coord_polar`, `coord_fixed`, `coord_trans` |
| **Faceting** | `facet_wrap`, `facet_grid` |
| **Themes** | 10 complete themes + full theme element system |
| **Guides** | Axis, legend, colourbar, coloursteps, bins, and custom guide types |

## Python-Exclusive Features

These have no R equivalent and leverage Python-specific language capabilities:

| Feature | Python mechanism | Example |
|---------|-----------------|---------|
| Callable `aes()` | First-class functions | `aes(y=lambda d: np.log(d["mpg"]))` |
| Callable `after_stat()` / `after_scale()` | Same | `after_stat(lambda d: d["count"] / d["count"].sum())` |
| `singledispatch` extensibility | `functools.singledispatch` | `@update_ggplot.register(MyClass)` |
| Build hooks | Dict-keyed callbacks | `plot.add_build_hook("after", BuildStage.COMPUTE_STAT, fn)` |
| Auto-registration | `__init_subclass__` | `class GeomStar(Geom): ...` auto-registers |
| Protocol contracts | `typing.Protocol` | `isinstance(my_geom, GeomProtocol)` |
| Scoped defaults | `contextvars.ContextVar` | `with ggplot_defaults(theme=theme_minimal()): ...` |

## Dependencies

- numpy, pandas, matplotlib (core)
- grid_py, gtable_py, scales (pre-ported R packages)
- scipy (statistical computations)
- contourpy (contour computation)

## Datasets

11 built-in datasets: diamonds, economics, faithfuld, luv_colours, midwest, mpg, msleep, presidential, seals, txhousing.
