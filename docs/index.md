# ggplot2_py

A Python port of the R ggplot2 package — Create Elegant Data Visualisations Using the Grammar of Graphics.

## Overview

ggplot2_py implements the grammar of graphics in Python, providing a layered approach to creating statistical visualizations. It is a faithful port of R's ggplot2 package, using pandas DataFrames as the primary data container and matplotlib as the rendering backend.

## Quick Start

```python
from ggplot2_py import *
from ggplot2_py.datasets import mpg

p = (ggplot(mpg, aes(x='displ', y='hwy', colour='class'))
     + geom_point()
     + theme_minimal()
     + labs(title='Engine Displacement vs Highway MPG'))
```

## Key Components

- **Aesthetics**: `aes()`, `after_stat()`, `after_scale()`, `stage()`
- **Geoms**: 40+ geometry layers (`geom_point`, `geom_bar`, `geom_boxplot`, ...)
- **Stats**: 30+ statistical transformations (`stat_bin`, `stat_smooth`, `stat_density`, ...)
- **Scales**: 130+ scale functions for colour, size, shape, position, etc.
- **Coordinates**: `coord_cartesian`, `coord_flip`, `coord_polar`, `coord_fixed`, ...
- **Faceting**: `facet_wrap`, `facet_grid`
- **Themes**: 10 complete themes + full theme element system
- **Guides**: Axis, legend, colourbar, and custom guide types

## Dependencies

- numpy, pandas, matplotlib (core)
- grid_py, gtable_py, scales (pre-ported R packages)
- scipy (statistical computations)
- contourpy (contour computation)

## Datasets

11 built-in datasets: diamonds, economics, faithfuld, luv_colours, midwest, mpg, msleep, presidential, seals, txhousing.
