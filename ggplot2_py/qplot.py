"""
Quick plotting shortcut (deprecated).

Provides :func:`qplot` / :func:`quickplot` for creating simple plots
with less typing.  This interface is deprecated in favour of the full
``ggplot() + geom_*()`` pipeline.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ggplot2_py._compat import cli_warn

__all__ = [
    "qplot",
    "quickplot",
]


def qplot(
    x: Any = None,
    y: Any = None,
    *,
    data: Optional[pd.DataFrame] = None,
    geom: Union[str, List[str]] = "auto",
    facets: Any = None,
    xlim: Optional[Sequence] = None,
    ylim: Optional[Sequence] = None,
    log: str = "",
    main: Optional[str] = None,
    xlab: Optional[str] = None,
    ylab: Optional[str] = None,
    asp: Optional[float] = None,
    colour: Any = None,
    color: Any = None,
    fill: Any = None,
    size: Any = None,
    shape: Any = None,
    alpha: Any = None,
    **kwargs: Any,
) -> Any:
    """Create a quick plot (deprecated).

    ``qplot`` is a shorthand for ``ggplot() + geom_*()``.  It is
    deprecated; prefer the full ggplot pipeline.

    Parameters
    ----------
    x, y : str or array-like, optional
        Variables for the x and y axes.  If strings, they are interpreted
        as column names in *data*.
    data : DataFrame, optional
        Dataset.
    geom : str or list of str, optional
        Geom name(s) to use (e.g. ``"point"``, ``"histogram"``).
        ``"auto"`` guesses from the supplied aesthetics.
    facets : str or None, optional
        Faceting formula (stub).
    xlim, ylim : sequence, optional
        Axis limits.
    log : str, optional
        Which axes to log-transform (``"x"``, ``"y"``, or ``"xy"``).
    main : str, optional
        Plot title.
    xlab, ylab : str, optional
        Axis labels.
    asp : float, optional
        Aspect ratio (y/x).
    colour, color, fill, size, shape, alpha : str or array-like, optional
        Additional aesthetics.
    **kwargs
        Other parameters passed to the geom.

    Returns
    -------
    GGPlot
        A ggplot object.

    Warnings
    --------
    ``qplot()`` is deprecated since ggplot2 3.4.0.  Use ``ggplot()``
    instead.
    """
    warnings.warn(
        "qplot() is deprecated. Use ggplot() + geom_*() instead.",
        FutureWarning,
        stacklevel=2,
    )

    from ggplot2_py.aes import aes
    from ggplot2_py.plot import ggplot, GGPlot
    from ggplot2_py.labels import labs

    # Normalise colour/color
    if color is not None and colour is None:
        colour = color

    # Build data if not provided
    if data is None:
        data_dict: dict = {}
        if x is not None and not isinstance(x, str):
            data_dict["x"] = np.asarray(x)
            x = "x"
        if y is not None and not isinstance(y, str):
            data_dict["y"] = np.asarray(y)
            y = "y"
        if colour is not None and not isinstance(colour, str):
            data_dict["colour"] = np.asarray(colour)
            colour = "colour"
        if fill is not None and not isinstance(fill, str):
            data_dict["fill"] = np.asarray(fill)
            fill = "fill"
        if size is not None and not isinstance(size, str):
            data_dict["size"] = np.asarray(size)
            size = "size"
        if shape is not None and not isinstance(shape, str):
            data_dict["shape"] = np.asarray(shape)
            shape = "shape"
        if alpha is not None and not isinstance(alpha, str):
            data_dict["alpha"] = np.asarray(alpha)
            alpha = "alpha"
        if data_dict:
            data = pd.DataFrame(data_dict)
        else:
            data = pd.DataFrame()

    # Build mapping
    mapping_kwargs: dict = {}
    if x is not None:
        mapping_kwargs["x"] = x
    if y is not None:
        mapping_kwargs["y"] = y
    if colour is not None:
        mapping_kwargs["colour"] = colour
    if fill is not None:
        mapping_kwargs["fill"] = fill
    if size is not None:
        mapping_kwargs["size"] = size
    if shape is not None:
        mapping_kwargs["shape"] = shape
    if alpha is not None:
        mapping_kwargs["alpha"] = alpha

    mapping = aes(**mapping_kwargs)

    # Auto-detect geom
    if isinstance(geom, str):
        geoms = [geom]
    else:
        geoms = list(geom)

    if "auto" in geoms:
        if y is not None:
            geoms = ["point" if g == "auto" else g for g in geoms]
        elif x is not None:
            # Check if x is discrete
            if isinstance(data, pd.DataFrame) and x in data.columns:
                col = data[x]
                if col.dtype == object or hasattr(col, "cat"):
                    geoms = ["bar" if g == "auto" else g for g in geoms]
                else:
                    geoms = ["histogram" if g == "auto" else g for g in geoms]
            else:
                geoms = ["histogram" if g == "auto" else g for g in geoms]
        else:
            geoms = ["point" if g == "auto" else g for g in geoms]

    # Determine axis labels
    if xlab is None:
        xlab = str(x) if x is not None else ""
    if ylab is None:
        ylab = str(y) if y is not None else ""

    # Build plot
    p = ggplot(data, mapping)

    # Add geoms
    import ggplot2_py.geoms as _geoms_mod

    for g in geoms:
        geom_fn = getattr(_geoms_mod, f"geom_{g}", None)
        if geom_fn is None:
            cli_warn(f"Unknown geom: 'geom_{g}'.")
            continue
        p = p + geom_fn(**kwargs)

    # Log transforms
    if "x" in log:
        try:
            from ggplot2_py.scales import scale_x_log10
            p = p + scale_x_log10()
        except ImportError:
            pass
    if "y" in log:
        try:
            from ggplot2_py.scales import scale_y_log10
            p = p + scale_y_log10()
        except ImportError:
            pass

    # Aspect ratio
    if asp is not None:
        from ggplot2_py.theme import theme
        p = p + theme(aspect_ratio=asp)

    # Labels
    label_kwargs: dict = {}
    if xlab:
        label_kwargs["x"] = xlab
    if ylab:
        label_kwargs["y"] = ylab
    if main:
        label_kwargs["title"] = main
    if label_kwargs:
        p = p + labs(**label_kwargs)

    # Axis limits
    if xlim is not None:
        from ggplot2_py.limits import xlim as _xlim
        p = p + _xlim(*xlim)
    if ylim is not None:
        from ggplot2_py.limits import ylim as _ylim
        p = p + _ylim(*ylim)

    return p


quickplot = qplot
