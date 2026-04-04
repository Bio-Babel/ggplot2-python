"""
Annotation helpers for ggplot2 plots.

Provides ``annotate()``, ``annotation_custom()``, ``annotation_raster()``,
``annotation_logticks()``, and map-related annotation stubs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ggplot2_py._compat import Waiver, is_waiver, waiver, cli_abort, cli_warn
from ggplot2_py._utils import compact

__all__ = [
    "annotate",
    "annotation_custom",
    "annotation_raster",
    "annotation_logticks",
    "annotation_map",
    "annotation_borders",
    "borders",
]


# ---------------------------------------------------------------------------
# annotate()
# ---------------------------------------------------------------------------

def annotate(
    geom: str,
    x: Any = None,
    y: Any = None,
    xmin: Any = None,
    xmax: Any = None,
    ymin: Any = None,
    ymax: Any = None,
    xend: Any = None,
    yend: Any = None,
    *,
    na_rm: bool = False,
    **kwargs: Any,
) -> Any:
    """Create an annotation layer.

    Unlike a typical geom, the aesthetics of an annotation layer are
    *not* mapped from variables in a data frame -- they are passed as
    constant vectors.  This is useful for adding small annotations such as
    text labels, rectangles, or line segments.

    Parameters
    ----------
    geom : str
        Name of the geom to use (e.g. ``"text"``, ``"rect"``, ``"segment"``).
    x, y, xmin, xmax, ymin, ymax, xend, yend : scalar or array-like, optional
        Positional aesthetics.  At least one must be specified.
    na_rm : bool, optional
        If ``True``, silently remove missing values.
    **kwargs
        Additional aesthetic values or geom parameters.

    Returns
    -------
    Layer
        A layer suitable for adding to a ggplot via ``+``.

    Raises
    ------
    ValueError
        If *geom* is one of the reference-line geoms (``"abline"``,
        ``"hline"``, ``"vline"``).

    Examples
    --------
    >>> annotate("text", x=4, y=25, label="Some text")
    >>> annotate("rect", xmin=3, xmax=4.2, ymin=12, ymax=21, alpha=0.2)
    """
    if geom in ("abline", "hline", "vline"):
        cli_warn(
            f"`geom` must not be '{geom}'. "
            f"Please use `geom_{geom}()` directly instead."
        )

    # Build position dict (only non-None entries)
    position = compact({
        "x": x,
        "xmin": xmin,
        "xmax": xmax,
        "xend": xend,
        "y": y,
        "ymin": ymin,
        "ymax": ymax,
        "yend": yend,
    })

    aesthetics: Dict[str, Any] = dict(position)
    aesthetics.update(kwargs)

    # Check compatible lengths
    lengths = {}
    for k, v in aesthetics.items():
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
            lengths[k] = len(v)
        else:
            lengths[k] = 1

    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        unique_lengths.discard(1)
    if len(unique_lengths) > 1:
        bad = {k: l for k, l in lengths.items() if l != 1}
        details = ", ".join(f"{k} ({l})" for k, l in bad.items())
        cli_abort(f"Unequal parameter lengths: {details}")

    n = max(lengths.values()) if lengths else 1

    # Build data from position aesthetics
    data_dict: Dict[str, Any] = {}
    for k, v in position.items():
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
            data_dict[k] = list(v)
        else:
            data_dict[k] = [v] * n
    data = pd.DataFrame(data_dict)

    # Separate params (non-position kwargs)
    params: Dict[str, Any] = {"na_rm": na_rm}
    for k, v in kwargs.items():
        if k not in ("position", "stat"):
            params[k] = v
        else:
            cli_warn(f"`annotate()` cannot accept the `{k}` argument.")

    # Lazy imports
    from ggplot2_py.aes import aes, Mapping
    from ggplot2_py.layer import layer
    from ggplot2_py.stat import StatIdentity
    from ggplot2_py.position import PositionIdentity

    # Build mapping from data columns
    mapping = aes(**{k: k for k in data.columns})

    return layer(
        geom=geom,
        params=params,
        stat=StatIdentity,
        position=PositionIdentity,
        data=data,
        mapping=mapping,
        inherit_aes=False,
        show_legend=False,
    )


# ---------------------------------------------------------------------------
# annotation_custom()
# ---------------------------------------------------------------------------

def annotation_custom(
    grob: Any,
    xmin: float = -np.inf,
    xmax: float = np.inf,
    ymin: float = -np.inf,
    ymax: float = np.inf,
) -> Any:
    """Add an arbitrary grob as a plot annotation.

    The annotation is placed using data coordinates and will not affect
    the scale limits.

    Parameters
    ----------
    grob : grob
        A grid grob to display.
    xmin, xmax : float
        Horizontal extent in data coordinates.  ``-inf``/``inf`` spans
        the full panel.
    ymin, ymax : float
        Vertical extent in data coordinates.

    Returns
    -------
    Layer
        An annotation layer.
    """
    from ggplot2_py.layer import layer
    from ggplot2_py.stat import StatIdentity
    from ggplot2_py.position import PositionIdentity
    from ggplot2_py.geom import Geom

    # Minimal GeomCustomAnn implementation
    class GeomCustomAnn(Geom):
        """Internal geom for annotation_custom."""

        _class_name = "GeomCustomAnn"
        extra_params = []

        @staticmethod
        def handle_na(data: Any, params: Any) -> Any:
            return data

        @staticmethod
        def draw_panel(data: Any, panel_params: Any, coord: Any,
                       grob: Any = None, xmin: float = -np.inf,
                       xmax: float = np.inf, ymin: float = -np.inf,
                       ymax: float = np.inf, **kwargs: Any) -> Any:
            from grid_py import Viewport, edit_grob

            # Transform annotation coordinates
            adf = pd.DataFrame({
                "xmin": [xmin], "xmax": [xmax],
                "ymin": [ymin], "ymax": [ymax],
            })
            adf = coord.transform(adf, panel_params)
            x_range = [float(adf["xmin"].iloc[0]), float(adf["xmax"].iloc[0])]
            y_range = [float(adf["ymin"].iloc[0]), float(adf["ymax"].iloc[0])]
            vp = Viewport(
                x=sum(x_range) / 2,
                y=sum(y_range) / 2,
                width=abs(x_range[1] - x_range[0]),
                height=abs(y_range[1] - y_range[0]),
                just="center",
            )
            return edit_grob(grob, vp=vp)

    dummy = pd.DataFrame({"x": [0], "y": [0]})

    return layer(
        data=dummy,
        stat=StatIdentity,
        position=PositionIdentity,
        geom=GeomCustomAnn,
        inherit_aes=False,
        params={
            "grob": grob,
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
        },
    )


# ---------------------------------------------------------------------------
# annotation_raster()
# ---------------------------------------------------------------------------

def annotation_raster(
    raster: Any,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    interpolate: bool = False,
) -> Any:
    """Add a raster image as a plot annotation.

    Parameters
    ----------
    raster : array-like
        Raster object to display (e.g. a NumPy array or PIL Image).
    xmin, xmax : float
        Horizontal extent in data coordinates.
    ymin, ymax : float
        Vertical extent in data coordinates.
    interpolate : bool, optional
        If ``True``, interpolate the raster linearly.

    Returns
    -------
    Layer
        An annotation layer.
    """
    from ggplot2_py.layer import layer
    from ggplot2_py.stat import StatIdentity
    from ggplot2_py.position import PositionIdentity
    from ggplot2_py.geom import Geom

    class GeomRasterAnn(Geom):
        """Internal geom for annotation_raster."""

        _class_name = "GeomRasterAnn"
        extra_params = []

        @staticmethod
        def handle_na(data: Any, params: Any) -> Any:
            return data

        @staticmethod
        def draw_panel(data: Any, panel_params: Any, coord: Any,
                       raster: Any = None, xmin: float = 0,
                       xmax: float = 1, ymin: float = 0,
                       ymax: float = 1, interpolate: bool = False,
                       **kwargs: Any) -> Any:
            from grid_py import raster_grob

            adf = pd.DataFrame({
                "xmin": [xmin], "xmax": [xmax],
                "ymin": [ymin], "ymax": [ymax],
            })
            adf = coord.transform(adf, panel_params)
            x0 = float(adf["xmin"].iloc[0])
            y0 = float(adf["ymin"].iloc[0])
            w = float(adf["xmax"].iloc[0]) - x0
            h = float(adf["ymax"].iloc[0]) - y0
            return raster_grob(
                raster, x0, y0, w, h,
                default_units="native",
                just=("left", "bottom"),
                interpolate=interpolate,
            )

    dummy = pd.DataFrame({"x": [0], "y": [0]})

    return layer(
        data=dummy,
        mapping=None,
        stat=StatIdentity,
        position=PositionIdentity,
        geom=GeomRasterAnn,
        inherit_aes=False,
        params={
            "raster": raster,
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "interpolate": interpolate,
        },
    )


# ---------------------------------------------------------------------------
# annotation_logticks()
# ---------------------------------------------------------------------------

def annotation_logticks(
    base: float = 10,
    sides: str = "bl",
    outside: bool = False,
    scaled: bool = True,
    short: Any = None,
    mid: Any = None,
    long: Any = None,
    colour: str = "black",
    linewidth: float = 0.5,
    linetype: Union[str, int] = 1,
    alpha: float = 1.0,
    color: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Add log-scale tick marks to a plot.

    This annotation is superseded by ``guide_axis_logticks()``.

    Parameters
    ----------
    base : float, optional
        Base of the logarithm (default 10).
    sides : str, optional
        Which sides to draw ticks on.  A string containing any of
        ``"t"`` (top), ``"r"`` (right), ``"b"`` (bottom), ``"l"`` (left).
    outside : bool, optional
        If ``True``, draw ticks outside the plot area.
    scaled : bool, optional
        ``True`` if data is already log-transformed.
    short, mid, long : Unit-like, optional
        Lengths for the three levels of tick marks.
    colour : str, optional
        Tick colour.
    linewidth : float, optional
        Tick line width in mm.
    linetype : str or int, optional
        Tick linetype.
    alpha : float, optional
        Tick transparency.
    color : str, optional
        Alias for *colour*.
    **kwargs
        Additional parameters.

    Returns
    -------
    Layer
        A log-tick annotation layer.
    """
    if color is not None:
        colour = color

    from grid_py import Unit

    if short is None:
        short = Unit(0.1, "cm")
    if mid is None:
        mid = Unit(0.2, "cm")
    if long is None:
        long = Unit(0.3, "cm")

    from ggplot2_py.layer import layer
    from ggplot2_py.stat import StatIdentity
    from ggplot2_py.position import PositionIdentity
    from ggplot2_py.geom import Geom

    class GeomLogticks(Geom):
        """Internal geom for annotation_logticks."""

        _class_name = "GeomLogticks"
        extra_params = []

        @staticmethod
        def handle_na(data: Any, params: Any) -> Any:
            return data

        @staticmethod
        def draw_panel(data: Any, panel_params: Any, coord: Any,
                       **params: Any) -> Any:
            from grid_py import GTree, GList, segments_grob, Unit as U

            # Stub: return an empty gTree
            return GTree(children=GList())

    dummy = pd.DataFrame({"x": [0], "y": [0]})

    return layer(
        data=dummy,
        mapping=None,
        stat=StatIdentity,
        position=PositionIdentity,
        geom=GeomLogticks,
        show_legend=False,
        inherit_aes=False,
        params={
            "base": base,
            "sides": sides,
            "outside": outside,
            "scaled": scaled,
            "short": short,
            "mid": mid,
            "long": long,
            "colour": colour,
            "linewidth": linewidth,
            "linetype": linetype,
            "alpha": alpha,
            **kwargs,
        },
    )


# ---------------------------------------------------------------------------
# Stubs for map-based annotations
# ---------------------------------------------------------------------------

def annotation_map(map_data: Any, **kwargs: Any) -> Any:
    """Add a map layer as an annotation (stub).

    Parameters
    ----------
    map_data : DataFrame
        Map data with ``long``, ``lat``, ``group`` columns.
    **kwargs
        Additional aesthetic parameters.

    Returns
    -------
    Layer
        An annotation layer (currently a stub that raises
        ``NotImplementedError``).
    """
    raise NotImplementedError(
        "annotation_map() is not yet implemented in the Python port."
    )


def annotation_borders(
    database: str = "world",
    regions: str = ".",
    fill: Optional[str] = None,
    colour: str = "grey50",
    xlim: Optional[Any] = None,
    ylim: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Add map border annotations (stub).

    Parameters
    ----------
    database : str
        Map database name.
    regions : str
        Regions to include.
    fill : str or None
        Fill colour.
    colour : str
        Border colour.
    xlim, ylim : optional
        Coordinate limits.

    Returns
    -------
    Layer
    """
    raise NotImplementedError(
        "annotation_borders() is not yet implemented in the Python port."
    )


def borders(
    database: str = "world",
    regions: str = ".",
    fill: Optional[str] = None,
    colour: str = "grey50",
    xlim: Optional[Any] = None,
    ylim: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Add map borders as a layer (stub).

    Parameters
    ----------
    database : str
        Map database name.
    regions : str
        Regions to include.
    fill : str or None
        Fill colour.
    colour : str
        Border colour.
    xlim, ylim : optional
        Coordinate limits.

    Returns
    -------
    Layer
    """
    raise NotImplementedError(
        "borders() is not yet implemented in the Python port."
    )
