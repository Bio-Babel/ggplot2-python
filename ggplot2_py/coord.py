"""
Coordinate systems for ggplot2.

Coordinate systems control how position aesthetics are mapped to the 2-D
plane of the plot. They also provide axes, panel backgrounds, and
foreground decorations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ggplot2_py._compat import Waiver, is_waiver, waiver, cli_abort, cli_warn


def _is_waiver_like(x: Any) -> bool:
    """Check if x is a Waiver or waiver-like sentinel."""
    return is_waiver(x) or x is None
from ggplot2_py.ggproto import GGProto, ggproto
from ggplot2_py._utils import snake_class, modify_list, compact

__all__ = [
    "Coord",
    "CoordCartesian",
    "CoordFixed",
    "CoordFlip",
    "CoordPolar",
    "CoordRadial",
    "CoordSf",
    "CoordTrans",
    "CoordTransform",
    "coord_cartesian",
    "coord_equal",
    "coord_fixed",
    "coord_flip",
    "coord_polar",
    "coord_radial",
    "coord_sf",
    "coord_trans",
    "coord_transform",
    "coord_munch",
    "is_coord",
    "is_Coord",
]


# ---------------------------------------------------------------------------
# Break computation helpers for panel_params
# ---------------------------------------------------------------------------


def _scale_numeric_range(scale: Any, fallback: Optional[list] = None) -> list:
    """Return the **expanded** numeric range for *scale*.

    R (coord-cartesian-.R:175-189 ``view_scales_from_scale``):

        expansion <- default_expansion(scale, expand = TRUE)
        continuous_range <- expand_limits_scale(scale, expansion, limits)

    R's ``Scale$dimension()`` itself defaults to ``expansion(0, 0)``
    (i.e. no expansion).  Expansion is applied *at the call site* —
    ``view_scales_from_scale`` passes the per-scale
    ``default_expansion`` explicitly.  Python previously baked a 5%
    expansion into ``dimension()`` as the default, which corrupted
    any caller that needed raw limits (e.g. ``hex_binwidth``).  With
    that default now matching R (no expansion), we have to apply
    the expansion here.
    """
    if scale is None:
        return list(fallback or [0, 1])

    if hasattr(scale, "dimension"):
        try:
            # Compute the scale-specific expansion vector (continuous
            # mult=0.05, discrete add=0.6, honouring a user-supplied
            # ``expand`` on the scale) and ask dimension() to apply it.
            from ggplot2_py.scale import default_expansion as _def_exp
            exp_vec = _def_exp(scale, expand=True)
            d = list(scale.dimension(expand=exp_vec))
            if len(d) >= 2:
                float(d[0])
                float(d[1])
                return d
        except (ValueError, TypeError, ImportError):
            pass

    # Fallback to get_limits for scales without dimension()
    if hasattr(scale, "get_limits"):
        try:
            lim = list(scale.get_limits())
            float(lim[0])
            float(lim[1])
            return lim
        except (ValueError, TypeError, IndexError):
            pass

    return list(fallback or [0, 1])


def _is_discrete_scale(scale: Any) -> bool:
    """Return True if *scale* is a discrete position scale."""
    cls_name = type(scale).__name__ if scale is not None else ""
    return "Discrete" in cls_name


def _compute_mapped_breaks(
    scale: Any,
    range_: list,
    n: int = 5,
) -> np.ndarray:
    """Compute major breaks and rescale to [0, 1] NPC.

    If the scale provides ``get_breaks()``, use it; otherwise fall back
    to ``numpy.linspace``.  For discrete position scales the breaks are
    placed at the integer positions corresponding to each level.
    """
    try:
        r0, r1 = float(range_[0]), float(range_[1])
    except (ValueError, TypeError):
        return np.array([])

    breaks = None
    if scale is not None and hasattr(scale, "get_breaks"):
        try:
            if _is_discrete_scale(scale):
                # For discrete scales, call get_breaks() without numeric
                # limits so it returns the category labels, then map to
                # integer positions 1..N.
                raw = scale.get_breaks()
                if raw is not None and len(raw) > 0:
                    breaks = np.arange(1, len(raw) + 1, dtype=float)
            else:
                raw = scale.get_breaks(range_)
                if raw is not None and len(raw) > 0:
                    breaks = np.asarray(raw, dtype=float)
        except Exception:
            pass
    if breaks is None or (hasattr(breaks, "__len__") and len(breaks) == 0):
        breaks = np.linspace(r0, r1, n + 2)[1:-1]
    try:
        breaks = np.asarray(breaks, dtype=float)
    except (ValueError, TypeError):
        return np.array([])
    breaks = breaks[np.isfinite(breaks)]
    # Rescale to [0, 1]
    rng = r1 - r0
    if rng == 0:
        return np.array([0.5] * len(breaks))
    return (breaks - r0) / rng


def _compute_break_labels(scale: Any, range_: list) -> Tuple[np.ndarray, List[str]]:
    """Return (break_positions_in_npc, labels) for axis rendering.

    This supplements ``_compute_mapped_breaks`` by also returning the
    text labels that should appear at each break.
    """
    try:
        r0, r1 = float(range_[0]), float(range_[1])
    except (ValueError, TypeError):
        return np.array([]), []

    if scale is None or not hasattr(scale, "get_breaks"):
        return np.array([]), []

    if _is_discrete_scale(scale):
        raw_breaks = scale.get_breaks()  # string labels
        if raw_breaks is None or len(raw_breaks) == 0:
            return np.array([]), []
        labels = [str(b) for b in raw_breaks]
        positions = np.arange(1, len(raw_breaks) + 1, dtype=float)
    else:
        raw_breaks = scale.get_breaks(range_)
        if raw_breaks is None or len(raw_breaks) == 0:
            return np.array([]), []
        try:
            positions = np.asarray(raw_breaks, dtype=float)
        except (ValueError, TypeError):
            return np.array([]), []
        # Get labels
        if hasattr(scale, "get_labels"):
            try:
                labels = scale.get_labels(raw_breaks)
            except Exception:
                labels = [str(b) for b in raw_breaks]
        else:
            labels = [str(b) for b in raw_breaks]

    # Filter out non-finite
    finite = np.isfinite(positions)
    positions = positions[finite]
    labels = [l for l, f in zip(labels, finite) if f]

    # Filter to range (keep only breaks within [r0, r1])
    in_range = (positions >= r0) & (positions <= r1)
    positions = positions[in_range]
    labels = [l for l, f in zip(labels, in_range) if f]

    # Rescale to [0, 1]
    rng = r1 - r0
    if rng == 0:
        npc = np.full(len(positions), 0.5)
    else:
        npc = (positions - r0) / rng

    return npc, labels


def _compute_mapped_minor_breaks(
    scale: Any,
    range_: list,
    major: np.ndarray,
    n: int = 2,
) -> np.ndarray:
    """Compute minor breaks in [0, 1] NPC, excluding positions that
    coincide with major breaks."""
    minor = None
    if scale is not None and hasattr(scale, "get_breaks_minor"):
        try:
            minor = scale.get_breaks_minor(range_, n=n)
        except Exception:
            pass
    if minor is None or (hasattr(minor, "__len__") and len(minor) == 0):
        # Default: one minor break between each pair of major breaks
        if len(major) >= 2:
            mids = (major[:-1] + major[1:]) / 2.0
            minor = mids
        else:
            return np.array([])
    else:
        minor = np.asarray(minor, dtype=float)
        rng = range_[1] - range_[0]
        if rng != 0:
            minor = (minor - range_[0]) / rng
    minor = minor[np.isfinite(minor)]
    # Remove minor breaks that coincide with major breaks
    if len(major) > 0 and len(minor) > 0:
        keep = np.array([not np.any(np.abs(major - m) < 1e-8) for m in minor])
        minor = minor[keep]
    return minor


# ---------------------------------------------------------------------------
# guide_grid — panel background and grid lines
# ---------------------------------------------------------------------------


def guide_grid(
    theme: Any,
    panel_params: Dict[str, Any],
    coord: Any,
) -> Any:
    """Render the panel background rectangle and grid lines.

    Mirrors R's ``guide_grid()`` from ``guides-grid.R``.  Produces a
    ``GTree`` containing:

    1. Panel background (``panel.background`` theme element)
    2. Minor grid lines (``panel.grid.minor.x/y``)
    3. Major grid lines (``panel.grid.major.x/y``)

    Parameters
    ----------
    theme : Theme
        The plot theme.
    panel_params : dict
        Panel parameters (must contain ``x_major``, ``x_minor``,
        ``y_major``, ``y_minor`` arrays in [0, 1] NPC).
    coord : Coord
        The coordinate system.

    Returns
    -------
    GTree
        A grob tree with background + grid lines.
    """
    from grid_py import (
        rect_grob, polyline_grob, grob_tree,
        null_grob, Gpar, Unit, GTree, GList,
    )
    # R: guide_grid() (guides-grid.R:6-41) — always uses element_render(),
    # no try/except fallback.  Theme element → grob via element_grob().
    from ggplot2_py.theme_elements import element_render

    children = []

    # 1. Panel background (R: guides-grid.R:32)
    bg = element_render(theme, "panel.background")
    if bg is not None:
        children.append(bg)

    x_major = panel_params.get("x_major", np.array([]))
    x_minor = panel_params.get("x_minor", np.array([]))
    y_major = panel_params.get("y_major", np.array([]))
    y_minor = panel_params.get("y_minor", np.array([]))

    # 2. Minor grid lines (R: breaks_as_grid, guides-grid.R:43-60)
    if len(y_minor) > 0:
        grob = element_render(
            theme, "panel.grid.minor.y",
            x=np.tile([0.0, 1.0], len(y_minor)),
            y=np.repeat(y_minor, 2),
            id_lengths=[2] * len(y_minor),
        )
        if grob is not None:
            children.append(grob)

    if len(x_minor) > 0:
        grob = element_render(
            theme, "panel.grid.minor.x",
            x=np.repeat(x_minor, 2),
            y=np.tile([0.0, 1.0], len(x_minor)),
            id_lengths=[2] * len(x_minor),
        )
        if grob is not None:
            children.append(grob)

    # 3. Major grid lines
    if len(y_major) > 0:
        grob = element_render(
            theme, "panel.grid.major.y",
            x=np.tile([0.0, 1.0], len(y_major)),
            y=np.repeat(y_major, 2),
            id_lengths=[2] * len(y_major),
        )
        if grob is not None:
            children.append(grob)

    if len(x_major) > 0:
        grob = element_render(
            theme, "panel.grid.major.x",
            x=np.repeat(x_major, 2),
            y=np.tile([0.0, 1.0], len(x_major)),
            id_lengths=[2] * len(x_major),
        )
        if grob is not None:
            children.append(grob)

    if not children:
        return null_grob()

    return grob_tree(*children, name="grill")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _rescale(
    x: np.ndarray,
    to: Tuple[float, float] = (0.0, 1.0),
    from_: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Linearly rescale *x* from *from_* range to *to* range."""
    x = np.asarray(x, dtype=float)
    if from_ is None:
        from_ = (float(np.nanmin(x)), float(np.nanmax(x)))
    rng = from_[1] - from_[0]
    if rng == 0:
        return np.full_like(x, (to[0] + to[1]) / 2.0)
    return (x - from_[0]) / rng * (to[1] - to[0]) + to[0]


def _squish_infinite(
    x: np.ndarray,
    range_: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Squish infinite values to the range endpoints."""
    x = np.asarray(x, dtype=float)
    if range_ is not None:
        x = np.where(np.isneginf(x), range_[0], x)
        x = np.where(np.isposinf(x), range_[1], x)
    else:
        x = np.where(np.isneginf(x), 0.0, x)
        x = np.where(np.isposinf(x), 1.0, x)
    return x


def _dist_euclidean(
    x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Euclidean distance between successive points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 0 or len(x) < 2:
        return np.array([0.0])
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sqrt(dx ** 2 + dy ** 2)


def _dist_polar(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Distance in polar coordinates between successive points."""
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if len(r) < 2:
        return np.array([0.0])
    dr = np.diff(r)
    dtheta = np.diff(theta)
    r1 = r[:-1]
    r2 = r[1:]
    return np.sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * np.cos(dtheta))


def _theta_rescale(
    x: np.ndarray,
    range_: Tuple[float, float],
    arc: Tuple[float, float] = (0, 2 * math.pi),
    direction: int = 1,
) -> np.ndarray:
    """Rescale theta to arc range, squishing and wrapping."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, range_[0], range_[1])
    out = _rescale(x, to=arc, from_=range_)
    return (out % (2 * math.pi)) * direction


def _theta_rescale_no_clip(
    x: np.ndarray,
    range_: Tuple[float, float],
    arc: Tuple[float, float] = (0, 2 * math.pi),
    direction: int = 1,
) -> np.ndarray:
    """Rescale theta without clipping."""
    x = np.asarray(x, dtype=float)
    return _rescale(x, to=arc, from_=range_) * direction


def _r_rescale(
    x: np.ndarray,
    range_: Tuple[float, float],
    donut: Tuple[float, float] = (0.0, 0.4),
) -> np.ndarray:
    """Rescale radius to donut range."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, range_[0], range_[1])
    return _rescale(x, to=donut, from_=range_)


def _parse_coord_expand(expand: Any) -> List[bool]:
    """Expand argument to a length-4 list of booleans (top, right, bottom, left)."""
    if isinstance(expand, bool):
        return [expand] * 4
    if isinstance(expand, (list, tuple)):
        result = list(expand)
        while len(result) < 4:
            result.append(result[-1] if result else True)
        return [bool(v) for v in result[:4]]
    return [True, True, True, True]


def _transform_position(
    data: pd.DataFrame,
    trans_x: Any = None,
    trans_y: Any = None,
) -> pd.DataFrame:
    """Apply transformation functions to position aesthetics.

    Parameters
    ----------
    data : pd.DataFrame
        Data to transform.
    trans_x, trans_y : callable, optional
        Transformation functions for x and y families.

    Returns
    -------
    pd.DataFrame
        Transformed data.
    """
    data = data.copy()
    x_cols = [c for c in data.columns if c in ("x", "xmin", "xmax", "xend", "xintercept")]
    y_cols = [c for c in data.columns if c in ("y", "ymin", "ymax", "yend", "yintercept")]
    if trans_x is not None:
        for c in x_cols:
            data[c] = trans_x(data[c].values)
    if trans_y is not None:
        for c in y_cols:
            data[c] = trans_y(data[c].values)
    return data


# ---------------------------------------------------------------------------
# Base Coord
# ---------------------------------------------------------------------------

class Coord(GGProto):
    """Base coordinate system.

    Attributes
    ----------
    default : bool
        Whether this is the default coordinate system.
    clip : str
        Clipping setting: ``"on"``, ``"off"``, or ``"inherit"``.
    reverse : str
        Which directions to reverse: ``"none"``, ``"x"``, ``"y"``, or ``"xy"``.
    """

    # --- Auto-registration registry (Python-exclusive) -------------------
    _registry: Dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        name = cls.__name__
        if name.startswith("Coord") and len(name) > 5:
            key = name[5:]
            Coord._registry[key] = cls
            Coord._registry[key.lower()] = cls

    default: bool = False
    clip: str = "on"
    reverse: str = "none"

    # -- setup ---------------------------------------------------------------

    def setup_params(self, data: Any) -> Dict[str, Any]:
        """Modify or check parameters based on data.

        Parameters
        ----------
        data : list of DataFrames
            Global data followed by layer data.

        Returns
        -------
        dict
            Parameters, including parsed ``expand``.
        """
        expand = getattr(self, "expand", True)
        return {"expand": _parse_coord_expand(expand)}

    def setup_data(self, data: Any, params: Optional[Dict[str, Any]] = None) -> Any:
        """Hook for modifying data before defaults are added.

        Parameters
        ----------
        data : list of DataFrames
        params : dict

        Returns
        -------
        list of DataFrames
        """
        return data

    def setup_layout(self, layout: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Hook for the coord to influence layout.

        Parameters
        ----------
        layout : pd.DataFrame
            Layout table with ``ROW``, ``COL``, ``PANEL``, ``SCALE_X``, ``SCALE_Y``.
        params : dict

        Returns
        -------
        pd.DataFrame
            Layout with an added ``COORD`` column.
        """
        layout = layout.copy()
        scales = layout[["SCALE_X", "SCALE_Y"]]
        unique_scales = scales.drop_duplicates().reset_index(drop=True)
        unique_scales = unique_scales.copy()
        unique_scales["COORD"] = range(1, len(unique_scales) + 1)
        layout = layout.drop(columns="COORD", errors="ignore")
        layout = pd.merge(layout, unique_scales, on=["SCALE_X", "SCALE_Y"], how="left")
        return layout

    # -- panel params --------------------------------------------------------

    def modify_scales(self, scales_x: list, scales_y: list) -> None:
        """Optionally modify scales in-place.

        Parameters
        ----------
        scales_x, scales_y : list
            Lists of trained x and y scales.
        """
        pass

    def setup_panel_params(
        self,
        scale_x: Any,
        scale_y: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create panel parameters for one panel.

        Parameters
        ----------
        scale_x, scale_y : Scale
            Trained position scales.
        params : dict

        Returns
        -------
        dict
            Panel parameters including view scales and ranges.
        """
        return {}

    def setup_panel_guides(
        self,
        panel_params: Dict[str, Any],
        guides: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Initiate position guides for a panel.

        Parameters
        ----------
        panel_params : dict
            Output from ``setup_panel_params``.
        guides : Guides
            Guides ggproto.
        params : dict

        Returns
        -------
        dict
            ``panel_params`` with guides appended.
        """
        panel_params["guides"] = guides
        return panel_params

    def train_panel_guides(
        self,
        panel_params: Dict[str, Any],
        layers: list,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train and transform position guides.

        Parameters
        ----------
        panel_params : dict
        layers : list
        params : dict

        Returns
        -------
        dict
        """
        return panel_params

    # -- transform -----------------------------------------------------------

    def transform(self, data: pd.DataFrame, panel_params: Dict[str, Any]) -> pd.DataFrame:
        """Transform data coordinates to [0, 1] range.

        Parameters
        ----------
        data : pd.DataFrame
            Data with numeric position columns.
        panel_params : dict
            Panel parameters.

        Returns
        -------
        pd.DataFrame
            Transformed data.
        """
        cli_abort(f"{snake_class(self)} has not implemented transform().")

    def distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        panel_params: Dict[str, Any],
    ) -> np.ndarray:
        """Compute distances between successive points.

        Parameters
        ----------
        x, y : array-like
        panel_params : dict

        Returns
        -------
        np.ndarray
        """
        cli_abort(f"{snake_class(self)} has not implemented distance().")

    def backtransform_range(self, panel_params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ranges back to data coordinates.

        Parameters
        ----------
        panel_params : dict

        Returns
        -------
        dict
            With ``x`` and ``y`` ranges.
        """
        cli_abort(f"{snake_class(self)} has not implemented backtransform_range().")

    def range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        """Extract x/y ranges from panel_params.

        Parameters
        ----------
        panel_params : dict

        Returns
        -------
        dict
            ``{"x": [lo, hi], "y": [lo, hi]}``.
        """
        cli_abort(f"{snake_class(self)} has not implemented range().")

    # -- render --------------------------------------------------------------

    def draw_panel(self, panel: Any, params: Dict[str, Any], theme: Any) -> Any:
        """Decorate panel with foreground and background.

        Parameters
        ----------
        panel : grob
        params : dict
        theme : Theme

        Returns
        -------
        grob
        """
        from grid_py import GTree, GList, Viewport
        fg = self.render_fg(params, theme)
        bg = self.render_bg(params, theme)
        children = [bg] + (list(panel) if isinstance(panel, (list, tuple)) else [panel]) + [fg]

        # The panel viewport maps NPC [0,1] to the panel sub-region,
        # matching R's Coord$draw_panel which wraps content in a
        # clipping viewport.
        return GTree(
            children=GList(*children),
            vp=Viewport(clip=self.clip),
        )

    def render_fg(self, panel_params: Dict[str, Any], theme: Any) -> Any:
        """Render panel foreground.

        Parameters
        ----------
        panel_params : dict
        theme : Theme

        Returns
        -------
        grob
        """
        from grid_py import null_grob
        return null_grob()

    def render_bg(self, panel_params: Dict[str, Any], theme: Any) -> Any:
        """Render panel background.

        Parameters
        ----------
        panel_params : dict
        theme : Theme

        Returns
        -------
        grob
        """
        cli_abort(f"{snake_class(self)} has not implemented render_bg().")

    def render_axis_h(self, panel_params: Dict[str, Any], theme: Any) -> Dict[str, Any]:
        """Render horizontal axes.

        Parameters
        ----------
        panel_params : dict
        theme : Theme

        Returns
        -------
        dict
            ``{"top": grob, "bottom": grob}``.
        """
        from grid_py import null_grob
        return {"top": null_grob(), "bottom": null_grob()}

    def render_axis_v(self, panel_params: Dict[str, Any], theme: Any) -> Dict[str, Any]:
        """Render vertical axes.

        Parameters
        ----------
        panel_params : dict
        theme : Theme

        Returns
        -------
        dict
            ``{"left": grob, "right": grob}``.
        """
        from grid_py import null_grob
        return {"left": null_grob(), "right": null_grob()}

    def labels(self, labels: Dict[str, Any], panel_params: Dict[str, Any]) -> Dict[str, Any]:
        """Format axis labels.

        Parameters
        ----------
        labels : dict
            Label structure with ``x`` and ``y`` sub-dicts.
        panel_params : dict

        Returns
        -------
        dict
        """
        return labels

    def aspect(self, ranges: Any) -> Optional[float]:
        """Return the aspect ratio for panels.

        Parameters
        ----------
        ranges : dict
            Panel parameters.

        Returns
        -------
        float or None
        """
        return None

    # -- utilities -----------------------------------------------------------

    def is_linear(self) -> bool:
        """Whether this coordinate system is linear.

        Returns
        -------
        bool
        """
        return False

    def is_free(self) -> bool:
        """Whether this coord supports free-scaling in facets.

        Returns
        -------
        bool
        """
        return False


# ---------------------------------------------------------------------------
# CoordCartesian
# ---------------------------------------------------------------------------

class CoordCartesian(Coord):
    """Cartesian coordinate system.

    Attributes
    ----------
    limits : dict
        ``{"x": (lo, hi) or None, "y": (lo, hi) or None}``.
    ratio : float or None
        Aspect ratio ``y/x``.
    """

    limits: Dict[str, Any] = {"x": None, "y": None}
    ratio: Optional[float] = None

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def is_linear(self) -> bool:
        return True

    def is_free(self) -> bool:
        return self.ratio is None

    def aspect(self, ranges: Any) -> Optional[float]:
        """Compute aspect ratio from ranges.

        Parameters
        ----------
        ranges : dict
            Must have ``x.range`` and ``y.range`` or similar.

        Returns
        -------
        float or None
        """
        if self.ratio is None:
            return None
        y_range = ranges.get("y.range") or ranges.get("y_range", [0, 1])
        x_range = ranges.get("x.range") or ranges.get("x_range", [0, 1])
        return (y_range[1] - y_range[0]) / max(x_range[1] - x_range[0], 1e-10) * self.ratio

    def distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        panel_params: Dict[str, Any],
    ) -> np.ndarray:
        """Euclidean distance normalised by the panel diagonal."""
        x_dim = panel_params.get("x_range") or panel_params.get("x.range", [0, 1])
        y_dim = panel_params.get("y_range") or panel_params.get("y.range", [0, 1])
        max_dist = np.sqrt((x_dim[1] - x_dim[0]) ** 2 + (y_dim[1] - y_dim[0]) ** 2)
        if max_dist == 0:
            max_dist = 1.0
        return _dist_euclidean(np.asarray(x), np.asarray(y)) / max_dist

    def range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        x_range = panel_params.get("x_range") or panel_params.get("x.range", [0, 1])
        y_range = panel_params.get("y_range") or panel_params.get("y.range", [0, 1])
        return {"x": list(x_range), "y": list(y_range)}

    def backtransform_range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        return self.range(panel_params)

    def transform(self, data: pd.DataFrame, panel_params: Dict[str, Any]) -> pd.DataFrame:
        """Rescale x/y into [0, 1].

        Parameters
        ----------
        data : pd.DataFrame
        panel_params : dict

        Returns
        -------
        pd.DataFrame
        """
        reverse = panel_params.get("reverse") or getattr(self, "reverse", "none")
        x_range = panel_params.get("x_range") or panel_params.get("x.range", [0, 1])
        y_range = panel_params.get("y_range") or panel_params.get("y.range", [0, 1])

        def rescale_x(vals: np.ndarray) -> np.ndarray:
            r = x_range
            if reverse in ("x", "xy"):
                r = list(reversed(r))
            return _rescale(vals, to=(0, 1), from_=tuple(r))

        def rescale_y(vals: np.ndarray) -> np.ndarray:
            r = y_range
            if reverse in ("y", "xy"):
                r = list(reversed(r))
            return _rescale(vals, to=(0, 1), from_=tuple(r))

        data = _transform_position(data, rescale_x, rescale_y)
        data = _transform_position(data, _squish_infinite, _squish_infinite)
        return data

    def setup_panel_params(
        self,
        scale_x: Any,
        scale_y: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build panel parameters from scales.

        Extracts limits and computes breaks/minor breaks so that
        ``render_bg`` can draw grid lines.

        Parameters
        ----------
        scale_x, scale_y : Scale
        params : dict

        Returns
        -------
        dict
        """
        params = params or {}
        x_limits = self.limits.get("x")
        y_limits = self.limits.get("y")

        x_range = _scale_numeric_range(scale_x, [0, 1])
        y_range = _scale_numeric_range(scale_y, [0, 1])

        # Apply coord limits as zoom
        if x_limits is not None:
            x_range = list(x_limits)
        if y_limits is not None:
            y_range = list(y_limits)

        # Compute breaks and rescale to [0, 1] NPC for grid lines
        x_major = _compute_mapped_breaks(scale_x, x_range)
        x_minor = _compute_mapped_minor_breaks(scale_x, x_range, x_major)
        y_major = _compute_mapped_breaks(scale_y, y_range)
        y_minor = _compute_mapped_minor_breaks(scale_y, y_range, y_major)

        # Break labels for axis rendering
        x_major_pos, x_labels = _compute_break_labels(scale_x, x_range)
        y_major_pos, y_labels = _compute_break_labels(scale_y, y_range)

        result = {
            "x_range": x_range,
            "y_range": y_range,
            "x.range": x_range,
            "y.range": y_range,
            "x_major": x_major_pos if len(x_major_pos) > 0 else x_major,
            "x_minor": x_minor,
            "y_major": y_major_pos if len(y_major_pos) > 0 else y_major,
            "y_minor": y_minor,
            "x_labels": x_labels,
            "y_labels": y_labels,
            "reverse": getattr(self, "reverse", "none"),
        }

        # Secondary axes — compute breaks via the AxisSecondary transform
        for axis, scale, rng in [("x", scale_x, x_range), ("y", scale_y, y_range)]:
            sec = getattr(scale, "secondary_axis", None)
            if sec is None or _is_waiver_like(sec):
                continue
            trans_fn = getattr(sec, "trans", None)
            if trans_fn is None:
                continue
            try:
                primary_breaks = np.array([float(b) for b in result[f"{axis}_major"]])
                # Transform break NPC positions back to data, apply sec trans,
                # then rescale back to NPC.  For dup_axis (identity), this
                # produces the same positions with (optionally) different labels.
                data_breaks = primary_breaks * (rng[1] - rng[0]) + rng[0]
                sec_data = np.array([float(trans_fn(b)) for b in data_breaks])
                sec_rng = [float(trans_fn(rng[0])), float(trans_fn(rng[1]))]
                if sec_rng[1] != sec_rng[0]:
                    sec_npc = (sec_data - sec_rng[0]) / (sec_rng[1] - sec_rng[0])
                else:
                    sec_npc = primary_breaks
                sec_labels = getattr(sec, "labels", None)
                if (sec_labels is None or _is_waiver_like(sec_labels)
                        or not hasattr(sec_labels, "__len__")):
                    # derive() / waiver / None → generate from break values
                    sec_labels = [str(round(v, 2)) for v in sec_data]
                elif callable(sec_labels):
                    sec_labels = sec_labels(sec_data)
                result[f"{axis}_sec_major"] = sec_npc
                result[f"{axis}_sec_labels"] = sec_labels
            except Exception:
                pass

        return result

    def render_bg(self, panel_params: Dict[str, Any], theme: Any) -> Any:
        """Render panel background (grid lines, background fill).

        Mirrors R's ``CoordCartesian$render_bg`` which delegates to
        ``guide_grid()``.
        """
        return guide_grid(theme, panel_params, self)

    def render_axis_h(self, panel_params: Dict[str, Any], theme: Any) -> Dict[str, Any]:
        """Render horizontal axes using the GuideAxis pipeline.

        Mirrors R's ``CoordCartesian$render_axis_h``.
        """
        from grid_py import null_grob
        from ggplot2_py.guide_axis import draw_axis

        breaks = panel_params.get("x_major", np.array([]))
        labels = panel_params.get("x_labels", [])
        minor = panel_params.get("x_minor", None)

        bottom = draw_axis(
            breaks, labels, "bottom", theme,
            minor_positions=minor,
        )

        top = null_grob()
        if panel_params.get("x_sec_major") is not None:
            sec_labels = panel_params.get("x_sec_labels", [])
            top = draw_axis(
                panel_params["x_sec_major"], sec_labels, "top", theme,
            )
        return {"top": top, "bottom": bottom}

    def render_axis_v(self, panel_params: Dict[str, Any], theme: Any) -> Dict[str, Any]:
        """Render vertical axes using the GuideAxis pipeline.

        Mirrors R's ``CoordCartesian$render_axis_v``.
        """
        from grid_py import null_grob
        from ggplot2_py.guide_axis import draw_axis

        breaks = panel_params.get("y_major", np.array([]))
        labels = panel_params.get("y_labels", [])
        minor = panel_params.get("y_minor", None)

        left = draw_axis(
            breaks, labels, "left", theme,
            minor_positions=minor,
        )

        right = null_grob()
        if panel_params.get("y_sec_major") is not None:
            sec_labels = panel_params.get("y_sec_labels", [])
            right = draw_axis(
                panel_params["y_sec_major"], sec_labels, "right", theme,
            )
        return {"left": left, "right": right}


def _resolve_element(element_name: str, theme: Any, fallback: dict) -> dict:
    """Resolve a theme element via calc_element, returning a flat dict.

    Falls back to *fallback* if the element is blank or missing.
    """
    from ggplot2_py.theme_elements import calc_element, ElementBlank
    try:
        el = calc_element(element_name, theme)
    except Exception:
        return dict(fallback)
    if el is None or isinstance(el, ElementBlank):
        return dict(fallback)
    out = dict(fallback)
    for key in fallback:
        val = getattr(el, key, None)
        if val is not None:
            # Resolve Rel values to float
            if hasattr(val, "x"):  # Rel wrapper
                val = float(val.x) * fallback.get(key, 1)
            out[key] = val
    return out



# NOTE: _render_axis has been removed and replaced by guide_axis.draw_axis.
# See guide_axis.py and the render_axis_h/render_axis_v methods above.


# ---------------------------------------------------------------------------
# CoordFixed
# ---------------------------------------------------------------------------

class CoordFixed(CoordCartesian):
    """Fixed-ratio Cartesian coordinate system.

    Attributes
    ----------
    ratio : float
        Aspect ratio (y per x unit).  Default is 1.
    """

    ratio: float = 1.0

    def is_free(self) -> bool:
        return False

    def aspect(self, ranges: Any) -> float:
        y_range = ranges.get("y.range") or ranges.get("y_range", [0, 1])
        x_range = ranges.get("x.range") or ranges.get("x_range", [0, 1])
        return (y_range[1] - y_range[0]) / max(x_range[1] - x_range[0], 1e-10) * self.ratio


# ---------------------------------------------------------------------------
# CoordFlip
# ---------------------------------------------------------------------------

class CoordFlip(CoordCartesian):
    """Flipped Cartesian coordinates (swap x and y)."""

    def transform(self, data: pd.DataFrame, panel_params: Dict[str, Any]) -> pd.DataFrame:
        data = _flip_axis_labels(data)
        return super().transform(data, panel_params)

    def backtransform_range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        r = self.range(panel_params)
        return r

    def range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        un_flipped = super().range(panel_params)
        return {"x": un_flipped["y"], "y": un_flipped["x"]}

    def setup_panel_params(
        self,
        scale_x: Any,
        scale_y: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = params or {}
        expand = params.get("expand", [True, True, True, True])
        if len(expand) >= 4:
            params["expand"] = [expand[1], expand[0], expand[3], expand[2]]
        pp = super().setup_panel_params(scale_x, scale_y, params)
        return _flip_axis_labels(pp) if isinstance(pp, dict) else pp

    def labels(self, labels: Dict[str, Any], panel_params: Dict[str, Any]) -> Dict[str, Any]:
        return _flip_axis_labels(labels)

    def setup_layout(self, layout: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        layout = super().setup_layout(layout, params)
        layout = layout.copy()
        layout[["SCALE_X", "SCALE_Y"]] = layout[["SCALE_Y", "SCALE_X"]].values
        return layout


def _flip_axis_labels(x: Any) -> Any:
    """Swap x/y prefixed names in a dict or DataFrame."""
    if isinstance(x, dict):
        new = {}
        for k, v in x.items():
            nk = k.replace("x", "__Z__").replace("y", "x").replace("__Z__", "y") if k.startswith(("x", "y")) else k
            new[nk] = v
        return new
    if isinstance(x, pd.DataFrame):
        rename_map = {}
        for c in x.columns:
            if c.startswith("x"):
                rename_map[c] = "y" + c[1:]
            elif c.startswith("y"):
                rename_map[c] = "x" + c[1:]
        return x.rename(columns=rename_map)
    return x


# ---------------------------------------------------------------------------
# CoordSf — simple-features / spatial coord system
#
# Faithful port of R's ``CoordSf`` (coord-sf.R:5-606), including:
#   * CRS reprojection of both sf geometry and plain (x, y) data
#     (``_sf_transform_xy``, ``_sf_rescale01``).
#   * Limit-projection methods cross / box / orthogonal / geometry_bbox
#     (``_calc_limits_bbox``).
#   * Graticule generation (``_st_graticule``) — port of
#     ``sf::st_graticule()`` covering meridian (E) and parallel (N)
#     LineStrings clipped to the panel bbox in target CRS, with start /
#     end / angle metadata.
#   * Axis-label viewscales (``_view_scales_from_graticule``) honouring
#     ``label_axes`` and ``label_graticule``.
#   * Aspect-ratio correction for longlat CRS
#     (``aspect`` returning ``Δy / Δx / cos(mid_lat)``).
#   * Bounding-box accumulation from sf layers via ``record_bbox``.
#   * ``render_bg`` drawing projected graticule lines as panel grid.
#
# Spatial dependencies (``shapely`` / ``pyproj`` / ``geopandas``) are
# imported lazily inside the helpers that need them so the ``coord``
# module loads without sf installed; calling ``coord_sf()`` itself only
# fails if a method that needs reprojection is invoked.
# ---------------------------------------------------------------------------


def _coerce_crs(value: Any) -> Any:
    """Coerce *value* to a ``pyproj.CRS`` instance (or ``None``).

    Mirrors ``sf::st_crs()`` coercion: ``None``/``NaN`` → ``None``,
    integers → ``EPSG:N``, strings/dicts → ``CRS.from_user_input``.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    from pyproj import CRS  # lazy
    if isinstance(value, CRS):
        return value
    return CRS.from_user_input(value)


def _sf_transform_xy(
    data: Any,
    target_crs: Any,
    source_crs: Any,
    authority_compliant: bool = False,
) -> Any:
    """Project ``x`` / ``y`` columns of *data* from source to target CRS.

    Port of R ``sf_transform_xy()`` (coord-sf.R:392-413). Operates on
    dict-like or DataFrame inputs and returns the same structure with
    projected coordinates. Returns *data* unchanged if either CRS is
    ``None``, the CRSs are equal, or *data* lacks ``x`` and ``y``.
    """
    if data is None:
        return data
    target = _coerce_crs(target_crs)
    source = _coerce_crs(source_crs)
    if target is None or source is None or target == source:
        return data
    # sf's identical() check: use the proj equality test
    if not all(k in data for k in ("x", "y")):
        return data

    from pyproj import Transformer  # lazy
    transformer = Transformer.from_crs(
        source, target, always_xy=not authority_compliant,
    )
    xs = np.asarray(data["x"], dtype=float)
    ys = np.asarray(data["y"], dtype=float)
    new_x, new_y = transformer.transform(xs, ys)
    new_x = np.where(np.isfinite(new_x), new_x, np.nan)
    new_y = np.where(np.isfinite(new_y), new_y, np.nan)

    if isinstance(data, pd.DataFrame):
        out = data.copy()
        out["x"] = new_x
        out["y"] = new_y
        return out
    out = dict(data)
    out["x"] = new_x
    out["y"] = new_y
    return out


def _sf_rescale01(geom: Any, x_range: Sequence[float], y_range: Sequence[float]) -> Any:
    """Normalise geometry coordinates to ``[0, 1] x [0, 1]`` panel space.

    Port of R ``sf_rescale01()`` (coord-sf.R:420-438). Wraps
    ``sf::st_normalize`` with handling for reversed ranges (which flip
    the corresponding axis after normalisation).
    """
    if geom is None:
        return geom
    from shapely import affinity  # lazy

    mult_x, mult_y = 1, 1
    xr0, xr1 = float(x_range[0]), float(x_range[1])
    yr0, yr1 = float(y_range[0]), float(y_range[1])
    if xr0 > xr1:
        xr0, xr1 = xr1, xr0
        mult_x = -1
    if yr0 > yr1:
        yr0, yr1 = yr1, yr0
        mult_y = -1

    dx = xr1 - xr0 if xr1 != xr0 else 1.0
    dy = yr1 - yr0 if yr1 != yr0 else 1.0

    def _normalize_one(g: Any) -> Any:
        if g is None or (hasattr(g, "is_empty") and g.is_empty):
            return g
        out = affinity.translate(g, xoff=-xr0, yoff=-yr0)
        out = affinity.scale(out, xfact=1.0 / dx, yfact=1.0 / dy, origin=(0, 0))
        if mult_x != 1 or mult_y != 1:
            out = affinity.scale(out, xfact=mult_x, yfact=mult_y, origin=(0, 0))
            out = affinity.translate(
                out, xoff=max(-mult_x, 0), yoff=max(-mult_y, 0),
            )
        return out

    if hasattr(geom, "__iter__") and not hasattr(geom, "is_empty"):
        return [_normalize_one(g) for g in geom]
    return _normalize_one(geom)


def _calc_limits_bbox(
    method: str,
    xlim: Sequence[float],
    ylim: Sequence[float],
    crs: Any,
    default_crs: Any,
) -> Dict[str, np.ndarray]:
    """Project the scale-limit polygon under ``lims_method`` and return points.

    Port of R ``calc_limits_bbox()`` (coord-sf.R:441-486). Returns a dict
    ``{"x": ..., "y": ...}`` whose ``min`` / ``max`` define the bounding
    box of the limits in target CRS.
    """
    finite = all(np.isfinite([xlim[0], xlim[1], ylim[0], ylim[1]]))
    if not finite and method != "geometry_bbox":
        cli_abort(
            "Scale limits cannot be mapped onto spatial coordinates in "
            "coord_sf(). Consider setting lims_method='geometry_bbox' or "
            "default_crs=None."
        )

    if method == "box":
        x = np.concatenate([
            np.repeat(xlim[0], 20),
            np.linspace(xlim[0], xlim[1], 20),
            np.repeat(xlim[1], 20),
            np.linspace(xlim[1], xlim[0], 20),
        ])
        y = np.concatenate([
            np.linspace(ylim[0], ylim[1], 20),
            np.repeat(ylim[1], 20),
            np.linspace(ylim[1], ylim[0], 20),
            np.repeat(ylim[0], 20),
        ])
    elif method == "geometry_bbox":
        x = np.array([np.nan, np.nan])
        y = np.array([np.nan, np.nan])
    elif method == "orthogonal":
        x = np.array([float(xlim[0]), float(xlim[1])])
        y = np.array([float(ylim[0]), float(ylim[1])])
    else:  # "cross" — also the default in R
        x = np.concatenate([
            np.repeat((xlim[0] + xlim[1]) / 2.0, 20),
            np.linspace(xlim[0], xlim[1], 20),
        ])
        y = np.concatenate([
            np.linspace(ylim[0], ylim[1], 20),
            np.repeat((ylim[0] + ylim[1]) / 2.0, 20),
        ])

    bbox = {"x": x, "y": y}
    return _sf_transform_xy(bbox, crs, default_crs)


def _format_degree_label(degree: float, type_: str) -> str:
    """Format a degree value as ``'120°W'`` / ``'30°N'`` (sf's format_lonlat)."""
    if not np.isfinite(degree):
        return ""
    abs_deg = abs(float(degree))
    if type_ == "E":
        suffix = "°E" if degree > 0 else ("°W" if degree < 0 else "°")
    else:  # "N"
        suffix = "°N" if degree > 0 else ("°S" if degree < 0 else "°")
    if abs_deg == int(abs_deg):
        return f"{int(abs_deg)}{suffix}"
    return f"{abs_deg:g}{suffix}"


def _st_graticule(
    bbox: Sequence[float],
    crs: Any = None,
    lat: Optional[Sequence[float]] = None,
    lon: Optional[Sequence[float]] = None,
    datum: Any = None,
    ndiscr: int = 100,
) -> pd.DataFrame:
    """Generate meridian / parallel LineStrings clipped to *bbox*.

    Port of ``sf::st_graticule()``. Returns a ``pandas.DataFrame`` with
    columns ``type`` (``"E"``/``"N"``), ``degree``, ``degree_label``,
    ``geometry`` (shapely ``LineString``), ``x_start``, ``y_start``,
    ``x_end``, ``y_end``, ``angle_start``, ``angle_end``. Empty rows
    (lines that fully fall outside *bbox* in target space) are dropped.
    """
    from shapely.geometry import LineString  # lazy

    crs_obj = _coerce_crs(crs)
    datum_obj = _coerce_crs(datum) if datum is not None else _coerce_crs(4326)

    # Compute datum-space bbox (long/lat by default) by densely sampling
    # the target-space bbox edges and forward-projecting.
    if crs_obj is not None and datum_obj is not None and crs_obj != datum_obj:
        from pyproj import Transformer  # lazy
        edge_x = np.concatenate([
            np.linspace(bbox[0], bbox[2], 50),
            np.repeat(bbox[2], 50),
            np.linspace(bbox[2], bbox[0], 50),
            np.repeat(bbox[0], 50),
        ])
        edge_y = np.concatenate([
            np.repeat(bbox[1], 50),
            np.linspace(bbox[1], bbox[3], 50),
            np.repeat(bbox[3], 50),
            np.linspace(bbox[3], bbox[1], 50),
        ])
        t_to_datum = Transformer.from_crs(crs_obj, datum_obj, always_xy=True)
        lon_edge, lat_edge = t_to_datum.transform(edge_x, edge_y)
        good = np.isfinite(lon_edge) & np.isfinite(lat_edge)
        if good.any():
            datum_bbox = (
                float(lon_edge[good].min()), float(lat_edge[good].min()),
                float(lon_edge[good].max()), float(lat_edge[good].max()),
            )
        else:
            datum_bbox = (-180.0, -90.0, 180.0, 90.0)
    else:
        datum_bbox = (float(bbox[0]), float(bbox[1]),
                      float(bbox[2]), float(bbox[3]))

    # Compute lon/lat breaks (R-style pretty()).  scales_py provides a
    # bit-for-bit port of R's pretty algorithm.
    def _pretty_breaks(lo: float, hi: float, n: int = 5) -> np.ndarray:
        try:
            from scales.breaks import _pretty as _scales_pretty
            return np.asarray(_scales_pretty(lo, hi, n=n), dtype=float)
        except ImportError:
            return np.linspace(lo, hi, n + 2)[1:-1]

    if lon is None:
        lon_breaks = _pretty_breaks(datum_bbox[0], datum_bbox[2], n=5)
    else:
        lon_breaks = np.asarray(lon, dtype=float)
    if lat is None:
        lat_breaks = _pretty_breaks(datum_bbox[1], datum_bbox[3], n=5)
    else:
        lat_breaks = np.asarray(lat, dtype=float)

    lon_breaks = lon_breaks[(lon_breaks >= datum_bbox[0])
                            & (lon_breaks <= datum_bbox[2])]
    lat_breaks = lat_breaks[(lat_breaks >= datum_bbox[1])
                            & (lat_breaks <= datum_bbox[3])]

    # Forward transformer from datum back to target CRS for line densification.
    if crs_obj is not None and datum_obj is not None and crs_obj != datum_obj:
        from pyproj import Transformer  # lazy
        t_to_target = Transformer.from_crs(datum_obj, crs_obj, always_xy=True)
    else:
        t_to_target = None

    n = max(int(ndiscr), 2)
    rows: List[Dict[str, Any]] = []

    from shapely.geometry import box as _shapely_box  # lazy
    bbox_poly = _shapely_box(bbox[0], bbox[1], bbox[2], bbox[3])

    def _add_line(deg: float, type_: str, lon_seq: np.ndarray,
                  lat_seq: np.ndarray) -> None:
        if t_to_target is not None:
            xs, ys = t_to_target.transform(lon_seq, lat_seq)
        else:
            xs, ys = lon_seq, lat_seq
        good = np.isfinite(xs) & np.isfinite(ys)
        if good.sum() < 2:
            return
        xs = np.asarray(xs)[good]; ys = np.asarray(ys)[good]
        line = LineString(list(zip(xs.tolist(), ys.tolist())))

        # Geometric clip to the plot bbox so endpoints land on bbox edges,
        # matching sf::st_graticule's behaviour.
        clipped = line.intersection(bbox_poly)
        if clipped.is_empty:
            return
        # intersection can yield a (Multi)LineString or even a Point.
        if clipped.geom_type == "MultiLineString":
            # Take the longest piece (sf chooses the dominant one too).
            parts = list(clipped.geoms)
            clipped = max(parts, key=lambda g: g.length)
        if clipped.geom_type != "LineString" or len(clipped.coords) < 2:
            return

        cxs, cys = clipped.coords.xy
        cxs = np.asarray(cxs, dtype=float)
        cys = np.asarray(cys, dtype=float)

        ang_start = math.degrees(math.atan2(cys[1] - cys[0], cxs[1] - cxs[0]))
        ang_end = math.degrees(math.atan2(cys[-1] - cys[-2], cxs[-1] - cxs[-2]))
        rows.append({
            "type": type_,
            "degree": float(deg),
            "degree_label": _format_degree_label(deg, type_),
            "geometry": clipped,
            "x_start": float(cxs[0]), "y_start": float(cys[0]),
            "x_end": float(cxs[-1]), "y_end": float(cys[-1]),
            "angle_start": float(ang_start),
            "angle_end": float(ang_end),
        })

    # Meridians: constant longitude, latitude varying from south to north
    for lon_v in lon_breaks:
        lat_seq = np.linspace(datum_bbox[1], datum_bbox[3], n)
        lon_seq = np.full_like(lat_seq, lon_v)
        _add_line(lon_v, "E", lon_seq, lat_seq)

    # Parallels: constant latitude, longitude varying from west to east
    for lat_v in lat_breaks:
        lon_seq = np.linspace(datum_bbox[0], datum_bbox[2], n)
        lat_seq = np.full_like(lon_seq, lat_v)
        _add_line(lat_v, "N", lon_seq, lat_seq)

    if not rows:
        return pd.DataFrame(columns=[
            "type", "degree", "degree_label", "geometry",
            "x_start", "y_start", "x_end", "y_end",
            "angle_start", "angle_end",
        ])
    return pd.DataFrame(rows)


def _sf_breaks(scale_x: Any, scale_y: Any,
               bbox: Sequence[float], crs: Any) -> Dict[str, Any]:
    """Choose break sets for graticule generation.

    Port of R ``sf_breaks()`` (coord-sf.R:623-664). Returns a dict
    ``{"x": <array|None|waiver>, "y": <array|None|waiver>}``:

    * ``None`` instructs ``_st_graticule`` to suppress that direction
      (scale set ``breaks = NULL``).
    * ``waiver`` means "let st_graticule pick defaults".
    * an array means "use these explicit breaks" (scale supplied a
      break vector or ``n.breaks``).
    """

    def _scale_breaks(scale: Any) -> Any:
        # Match R's `scale$breaks` access — None when explicitly suppressed.
        return getattr(scale, "breaks", None)

    def _scale_n_breaks(scale: Any) -> Any:
        # ggplot2_py uses ``n_breaks`` (Pythonic); be tolerant of either.
        return getattr(scale, "n_breaks", None) or getattr(scale, "n.breaks", None)

    x_brk = _scale_breaks(scale_x)
    y_brk = _scale_breaks(scale_y)
    x_n = _scale_n_breaks(scale_x)
    y_n = _scale_n_breaks(scale_y)

    has_x = x_brk is not None or x_n is not None
    has_y = y_brk is not None or y_n is not None

    x_breaks: Any = waiver() if has_x else None
    y_breaks: Any = waiver() if has_y else None

    if not (has_x or has_y):
        return {"x": x_breaks, "y": y_breaks}

    # Project bbox to long/lat (4326) so user-supplied breaks (assumed
    # in long/lat) line up with the scale's ``get_breaks`` domain.
    bbox_ll = list(bbox)
    if crs is not None:
        try:
            crs_obj = _coerce_crs(crs)
            if crs_obj is not None:
                from pyproj import CRS, Transformer  # lazy
                target = CRS.from_user_input(4326)
                if crs_obj != target:
                    t = Transformer.from_crs(crs_obj, target, always_xy=True)
                    edge_x = np.array([bbox[0], bbox[2], bbox[0], bbox[2]])
                    edge_y = np.array([bbox[1], bbox[1], bbox[3], bbox[3]])
                    lon_e, lat_e = t.transform(edge_x, edge_y)
                    if np.all(np.isfinite(lon_e)) and np.all(np.isfinite(lat_e)):
                        bbox_ll = [float(lon_e.min()), float(lat_e.min()),
                                   float(lon_e.max()), float(lat_e.max())]
                    else:
                        bbox_ll = [-180.0, -90.0, 180.0, 90.0]
        except Exception:
            bbox_ll = list(bbox)

    # R: only override x_breaks when scale supplied non-default breaks
    # OR an n.breaks setting (coord-sf.R:650-660).
    if has_x and not (_is_waiver_like(x_brk) and x_n is None):
        try:
            raw = scale_x.get_breaks(limits=[bbox_ll[0], bbox_ll[2]])
        except TypeError:
            raw = scale_x.get_breaks([bbox_ll[0], bbox_ll[2]])
        if raw is None or len(np.asarray(raw)) == 0:
            x_breaks = None
        else:
            arr = np.asarray(raw, dtype=float)
            arr = arr[np.isfinite(arr)]
            x_breaks = arr if arr.size else None

    if has_y and not (_is_waiver_like(y_brk) and y_n is None):
        try:
            raw = scale_y.get_breaks(limits=[bbox_ll[1], bbox_ll[3]])
        except TypeError:
            raw = scale_y.get_breaks([bbox_ll[1], bbox_ll[3]])
        if raw is None or len(np.asarray(raw)) == 0:
            y_breaks = None
        else:
            arr = np.asarray(raw, dtype=float)
            arr = arr[np.isfinite(arr)]
            y_breaks = arr if arr.size else None

    return {"x": x_breaks, "y": y_breaks}


def _view_scales_from_graticule(
    graticule: pd.DataFrame,
    scale: Any,
    aesthetic: str,
    label: str,
    label_graticule: Sequence[str],
    bbox: Sequence[float],
) -> Dict[str, Any]:
    """Convert a graticule into per-axis tick positions and labels.

    Port of R ``view_scales_from_graticule()`` (coord-sf.R:685-812).
    Python coords store axis breaks as ``x_major``/``y_major`` arrays
    rather than ``ViewScale`` objects, so this returns the dict shape
    consumed by ``CoordCartesian.render_axis_*``.
    """
    if graticule is None or len(graticule) == 0:
        return {
            "position": _aes_to_position(aesthetic),
            "limits": _aes_limits(aesthetic, bbox),
            "breaks": np.array([]),
            "labels": [],
        }

    position = _aes_to_position(aesthetic)
    axis = aesthetic.replace(".sec", "")
    if axis == "x":
        orth = "y"
        thres = (float(bbox[1]), float(bbox[3]))
        limits = (float(bbox[0]), float(bbox[2]))
    else:
        orth = "x"
        thres = (float(bbox[0]), float(bbox[2]))
        limits = (float(bbox[1]), float(bbox[3]))

    axis_start = f"{axis}_start"
    axis_end = f"{axis}_end"
    orth_start = f"{orth}_start"
    orth_end = f"{orth}_end"

    span = thres[1] - thres[0]
    if position in ("top", "right"):
        thr = thres[0] + 0.999 * span
        accept_start = graticule[orth_start].values > thr
        accept_end = graticule[orth_end].values > thr
    else:
        thr = thres[0] + 0.001 * span
        accept_start = graticule[orth_start].values < thr
        accept_end = graticule[orth_end].values < thr

    if not (accept_start | accept_end).any():
        eps = math.sqrt(np.finfo(float).tiny)
        # For top/bottom we expect meridians angled near 90° (vertical);
        # for left/right we expect parallels near 0° (horizontal).
        subtract = 90.0 if position in ("top", "bottom") else 0.0
        straight = (
            (np.abs(graticule["angle_start"].values - subtract) < eps)
            & (np.abs(graticule["angle_end"].values - subtract) < eps)
        )
        accept_start = straight

    types = graticule["type"].values
    idx_start: List[int] = list(np.where((types == label) & accept_start)[0])
    idx_end: List[int] = list(np.where((types == label) & accept_end)[0])

    if "S" in label_graticule:
        idx_start += list(np.where((types == "E") & accept_start)[0])
    if "N" in label_graticule:
        idx_end += list(np.where((types == "E") & accept_end)[0])
    if "W" in label_graticule:
        idx_start += list(np.where((types == "N") & accept_start)[0])
    if "E" in label_graticule:
        idx_end += list(np.where((types == "N") & accept_end)[0])

    idx_start = sorted(set(idx_start))
    idx_end = sorted(set(idx_end))

    positions = list(graticule.iloc[idx_start][axis_start].values) \
        + list(graticule.iloc[idx_end][axis_end].values)
    labels = list(graticule.iloc[idx_start]["degree_label"].values) \
        + list(graticule.iloc[idx_end]["degree_label"].values)

    if positions:
        ord_idx = np.argsort(positions)
        positions = [positions[i] for i in ord_idx]
        labels = [labels[i] for i in ord_idx]

    return {
        "position": position,
        "limits": limits,
        "breaks": np.asarray(positions, dtype=float),
        "labels": labels,
    }


def _aes_to_position(aesthetic: str) -> str:
    return {"x": "bottom", "x.sec": "top",
            "y": "left", "y.sec": "right"}[aesthetic]


def _aes_limits(aesthetic: str, bbox: Sequence[float]) -> Tuple[float, float]:
    if aesthetic.startswith("x"):
        return (float(bbox[0]), float(bbox[2]))
    return (float(bbox[1]), float(bbox[3]))


def _parse_axes_labeling(x: Any) -> Dict[str, str]:
    """Port of R ``parse_axes_labeling()`` (coord-sf.R:608-616)."""
    if isinstance(x, str):
        chars = list(x)
        chars += [""] * (4 - len(chars))
        return {"top": chars[0], "right": chars[1],
                "bottom": chars[2], "left": chars[3]}
    if isinstance(x, dict):
        return {k: x.get(k, "") for k in ("top", "right", "bottom", "left")}
    cli_abort("Panel labeling format not recognized.")


def _detect_geom_column(data: Any) -> Optional[str]:
    """Find the active geometry column on a (Geo)DataFrame, else ``None``."""
    if data is None:
        return None
    name = getattr(data, "_geometry_column_name", None)
    if name and name in getattr(data, "columns", []):
        return name
    if "geometry" in getattr(data, "columns", []):
        return "geometry"
    return None


def _is_sf_data(data: Any) -> bool:
    """True if *data* carries a shapely-geometry column (analogue of ``is_sf``)."""
    if data is None:
        return False
    cls_name = type(data).__name__
    if cls_name in ("GeoDataFrame", "GeoSeries"):
        return True
    geom_col = _detect_geom_column(data)
    if geom_col is None:
        return False
    series = data[geom_col]
    for v in series:
        if v is not None and hasattr(v, "geom_type"):
            return True
        # only inspect the first non-null cell
        if v is not None:
            return False
    return False


def _is_transform_immune(data: Any, coord_name: str) -> bool:
    """Port of R ``is_transform_immune()`` (coord-.R:749-768).

    Python data has no ``AsIs`` analogue, so this is always ``False``.
    Kept for parity with the R surface so future ``I()`` support hooks
    in cleanly.
    """
    return False


# ---- CoordSf class ---------------------------------------------------------


class CoordSf(CoordCartesian):
    """Spatial-features coordinate system.

    Port of R ``CoordSf`` (coord-sf.R:5-359). Inherits ``CoordCartesian``
    for ``setup_layout`` / ``setup_panel_guides`` / ``modify_scales`` /
    rendering helpers, and overrides every method whose semantics differ
    under sf — ``setup_params``, ``setup_data``, ``transform``,
    ``setup_panel_params``, ``aspect``, ``is_linear``, ``is_free``,
    ``distance``, ``backtransform_range``, ``range``, and ``render_bg``.
    """

    # --- Class fields, mirroring R CoordSf (coord-sf.R:5-22) -------------
    default: bool = False
    clip: str = "on"
    lims_method: str = "cross"
    ndiscr: int = 100
    crs: Any = None
    default_crs: Any = None
    datum: Any = None
    label_graticule: Any = None
    label_axes: Any = None
    expand: Any = True
    reverse: str = "none"
    limits: Dict[str, Any] = {"x": None, "y": None}

    def __init__(self, **kwargs: Any) -> None:
        # Per-instance params dict so different panels do not share state.
        self.params: Dict[str, Any] = {}
        super().__init__(**kwargs)

    # ---- bbox accumulation hook used by GeomSf draw_panel (coord-sf.R:70-77)

    def record_bbox(self, xmin: float, xmax: float,
                    ymin: float, ymax: float) -> None:
        bbox = self.params.get("bbox") or {
            "xmin": math.inf, "xmax": -math.inf,
            "ymin": math.inf, "ymax": -math.inf,
        }
        bbox["xmin"] = min(bbox["xmin"], float(xmin))
        bbox["xmax"] = max(bbox["xmax"], float(xmax))
        bbox["ymin"] = min(bbox["ymin"], float(ymin))
        bbox["ymax"] = max(bbox["ymax"], float(ymax))
        self.params["bbox"] = bbox

    def get_default_crs(self) -> Any:
        return self.default_crs if self.default_crs is not None \
            else self.params.get("default_crs")

    # ---- setup_params / determine_crs (coord-sf.R:20-51) ----------------

    def setup_params(self, data: Any) -> Dict[str, Any]:
        params = Coord.setup_params(self, data)
        params["crs"] = self.determine_crs(data)
        params["default_crs"] = self.default_crs
        self.params.update(params)
        return params

    def determine_crs(self, data: Any) -> Any:
        if self.crs is not None:
            return self.crs
        if data is None:
            return None
        for layer_data in data:
            if not _is_sf_data(layer_data):
                continue
            geom_col = _detect_geom_column(layer_data)
            if geom_col is None:
                continue
            series = layer_data[geom_col]
            crs = getattr(series, "crs", None)
            if crs is None:
                continue
            return _coerce_crs(crs)
        return None

    # ---- setup_data: project all sf layers to common CRS (coord-sf.R:54-67)

    def setup_data(self, data: Any, params: Optional[Dict[str, Any]] = None) -> Any:
        params = params or self.params
        target_crs = params.get("crs")
        if target_crs is None or data is None:
            return data
        target = _coerce_crs(target_crs)
        if target is None:
            return data

        out = []
        for layer_data in data:
            if not _is_sf_data(layer_data):
                out.append(layer_data)
                continue
            geom_col = _detect_geom_column(layer_data)
            if geom_col is None:
                out.append(layer_data)
                continue
            try:
                import geopandas as gpd  # lazy
            except ImportError:
                out.append(layer_data)
                continue
            series = layer_data[geom_col]
            if not isinstance(series, gpd.GeoSeries):
                series = gpd.GeoSeries(series, crs=getattr(series, "crs", None))
            if series.crs is None or series.crs == target:
                out.append(layer_data)
                continue
            new_geom = series.to_crs(target)
            if isinstance(layer_data, gpd.GeoDataFrame):
                new_layer = layer_data.copy()
                new_layer[geom_col] = new_geom
            else:
                new_layer = layer_data.copy()
                new_layer[geom_col] = list(new_geom)
            out.append(new_layer)
        return out

    # ---- transform: rescale geometry + reproject + rescale x/y (coord-sf.R:79-107)

    def transform(self, data: pd.DataFrame, panel_params: Dict[str, Any]) -> pd.DataFrame:
        if _is_transform_immune(data, snake_class(self)):
            return data

        source_crs = panel_params.get("default_crs")
        target_crs = panel_params.get("crs")
        reverse = panel_params.get("reverse") or getattr(self, "reverse", "none")
        x_range = list(panel_params.get("x_range", [0, 1]))
        y_range = list(panel_params.get("y_range", [0, 1]))
        if reverse in ("xy", "x"):
            x_range = list(reversed(x_range))
        if reverse in ("xy", "y"):
            y_range = list(reversed(y_range))

        # Rescale geometry column (already in target CRS) to NPC.
        geom_col = _detect_geom_column(data) if isinstance(data, pd.DataFrame) else None
        if geom_col is not None:
            data = data.copy()
            data[geom_col] = _sf_rescale01(list(data[geom_col]), x_range, y_range)

        # Reproject + rescale plain x/y data.
        if isinstance(data, pd.DataFrame) and {"x", "y"}.issubset(data.columns):
            data = _sf_transform_xy(data, target_crs, source_crs)

            def rescale_x(vals: np.ndarray) -> np.ndarray:
                return _rescale(vals, to=(0, 1), from_=tuple(x_range))

            def rescale_y(vals: np.ndarray) -> np.ndarray:
                return _rescale(vals, to=(0, 1), from_=tuple(y_range))

            data = _transform_position(data, rescale_x, rescale_y)
            data = _transform_position(data, _squish_infinite, _squish_infinite)
        return data

    # ---- panel-params with graticule (coord-sf.R:177-278) ---------------

    def setup_panel_params(self, scale_x: Any, scale_y: Any,
                           params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from ggplot2_py.scale import default_expansion, expand_range4

        coord_params: Dict[str, Any] = self.params.copy()
        if params:
            coord_params.update(params)
        expand_vec = _parse_coord_expand(getattr(self, "expand", True))
        # R: expansion_x uses positions [4, 2] (left, right); expansion_y
        # uses positions [3, 1] (bottom, top).  Our parsed expand is
        # (top, right, bottom, left).
        try:
            expansion_x = default_expansion(scale_x, expand=bool(expand_vec[3] or expand_vec[1]))
            expansion_y = default_expansion(scale_y, expand=bool(expand_vec[2] or expand_vec[0]))
        except Exception:
            from ggplot2_py.scale import expansion as _expansion
            expansion_x = _expansion(mult=0.05) if any(expand_vec) else _expansion()
            expansion_y = _expansion(mult=0.05) if any(expand_vec) else _expansion()

        scale_xlim = list(scale_x.get_limits()) if hasattr(scale_x, "get_limits") else [0.0, 1.0]
        scale_ylim = list(scale_y.get_limits()) if hasattr(scale_y, "get_limits") else [0.0, 1.0]
        coord_xlim = list(self.limits.get("x") or [np.nan, np.nan])
        coord_ylim = list(self.limits.get("y") or [np.nan, np.nan])

        scale_xlim = [scale_xlim[i] if not np.isfinite(coord_xlim[i]) else coord_xlim[i]
                      for i in range(2)]
        scale_ylim = [scale_ylim[i] if not np.isfinite(coord_ylim[i]) else coord_ylim[i]
                      for i in range(2)]

        scales_bbox = _calc_limits_bbox(
            self.lims_method, scale_xlim, scale_ylim,
            coord_params.get("crs"), coord_params.get("default_crs"),
        )

        coord_bbox = self.params.get("bbox") or {}

        if (self.limits.get("x") is None and self.limits.get("y") is None
                and getattr(scale_x, "limits", None) is None
                and getattr(scale_y, "limits", None) is None):
            xs = np.concatenate([
                np.asarray(scales_bbox["x"], dtype=float),
                np.asarray([coord_bbox.get("xmin", np.nan),
                            coord_bbox.get("xmax", np.nan)], dtype=float),
            ])
            ys = np.concatenate([
                np.asarray(scales_bbox["y"], dtype=float),
                np.asarray([coord_bbox.get("ymin", np.nan),
                            coord_bbox.get("ymax", np.nan)], dtype=float),
            ])
            scales_xrange = [float(np.nanmin(xs)), float(np.nanmax(xs))]
            scales_yrange = [float(np.nanmin(ys)), float(np.nanmax(ys))]
        elif (np.any(~np.isfinite(np.asarray(scales_bbox["x"], dtype=float)))
              or np.any(~np.isfinite(np.asarray(scales_bbox["y"], dtype=float)))):
            if self.lims_method != "geometry_bbox":
                cli_warn(
                    "Projection of x or y limits failed in coord_sf(). "
                    "Consider setting lims_method='geometry_bbox' or default_crs=None."
                )
            xmin = coord_bbox.get("xmin", 0.0)
            xmax = coord_bbox.get("xmax", 0.0)
            ymin = coord_bbox.get("ymin", 0.0)
            ymax = coord_bbox.get("ymax", 0.0)
            scales_xrange = [float(xmin), float(xmax)]
            scales_yrange = [float(ymin), float(ymax)]
        else:
            scales_xrange = [float(np.nanmin(scales_bbox["x"])),
                             float(np.nanmax(scales_bbox["x"]))]
            scales_yrange = [float(np.nanmin(scales_bbox["y"])),
                             float(np.nanmax(scales_bbox["y"]))]

        x_range = list(expand_range4(scales_xrange, expansion_x))
        y_range = list(expand_range4(scales_yrange, expansion_y))
        bbox = (x_range[0], y_range[0], x_range[1], y_range[1])

        breaks = _sf_breaks(scale_x, scale_y, bbox, coord_params.get("crs"))

        graticule = _st_graticule(
            bbox,
            crs=coord_params.get("crs"),
            lat=breaks["y"] if not _is_waiver_like(breaks["y"]) else None,
            lon=breaks["x"] if not _is_waiver_like(breaks["x"]) else None,
            datum=self.datum,
            ndiscr=self.ndiscr,
        )

        if breaks["x"] is None and len(graticule):
            graticule = graticule[graticule["type"] != "E"].reset_index(drop=True)
        if breaks["y"] is None and len(graticule):
            graticule = graticule[graticule["type"] != "N"].reset_index(drop=True)

        graticule = self.fixup_graticule_labels(graticule, scale_x, scale_y, coord_params)

        label_axes = self.label_axes or {"top": "", "right": "", "bottom": "E", "left": "N"}
        label_graticule = self.label_graticule or []

        view_x = _view_scales_from_graticule(
            graticule, scale_x, "x", label_axes.get("bottom", ""),
            label_graticule, bbox,
        )
        view_y = _view_scales_from_graticule(
            graticule, scale_y, "y", label_axes.get("left", ""),
            label_graticule, bbox,
        )
        view_x_sec = _view_scales_from_graticule(
            graticule, scale_x, "x.sec", label_axes.get("top", ""),
            label_graticule, bbox,
        )
        view_y_sec = _view_scales_from_graticule(
            graticule, scale_y, "y.sec", label_axes.get("right", ""),
            label_graticule, bbox,
        )

        # Convert view-scale tick positions (in target CRS) to NPC for axis
        # rendering, matching CoordCartesian's panel_params contract.
        def _to_npc(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                return arr
            span = hi - lo
            if span == 0:
                return np.full_like(arr, 0.5)
            return (arr - lo) / span

        x_major_npc = _to_npc(view_x["breaks"], x_range[0], x_range[1])
        y_major_npc = _to_npc(view_y["breaks"], y_range[0], y_range[1])
        x_sec_npc = _to_npc(view_x_sec["breaks"], x_range[0], x_range[1])
        y_sec_npc = _to_npc(view_y_sec["breaks"], y_range[0], y_range[1])

        panel_params = {
            "x_range": x_range,
            "y_range": y_range,
            "x.range": x_range,
            "y.range": y_range,
            "crs": coord_params.get("crs"),
            "default_crs": coord_params.get("default_crs"),
            "reverse": getattr(self, "reverse", "none"),
            "x_major": x_major_npc,
            "y_major": y_major_npc,
            "x_minor": np.array([]),
            "y_minor": np.array([]),
            "x_labels": list(view_x["labels"]),
            "y_labels": list(view_y["labels"]),
            "x_sec_major": x_sec_npc if x_sec_npc.size else None,
            "y_sec_major": y_sec_npc if y_sec_npc.size else None,
            "x_sec_labels": list(view_x_sec["labels"]),
            "y_sec_labels": list(view_y_sec["labels"]),
            "view_scales": {
                "x": view_x, "y": view_y,
                "x.sec": view_x_sec, "y.sec": view_y_sec,
            },
            "graticule": graticule,
        }

        # Rescale graticule LineStrings into NPC for render_bg.
        if len(graticule):
            normalized_lines = _sf_rescale01(
                list(graticule["geometry"]), x_range, y_range,
            )
            graticule = graticule.copy()
            graticule["geometry"] = normalized_lines
            panel_params["graticule"] = graticule
        return panel_params

    # ---- backtransform (coord-sf.R:290-299) ------------------------------

    def backtransform_range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        target_crs = panel_params.get("default_crs")
        source_crs = panel_params.get("crs")
        x = panel_params.get("x_range", [0, 1])
        y = panel_params.get("y_range", [0, 1])
        data = {"x": np.array([x[0], x[1], x[0], x[1]]),
                "y": np.array([y[0], y[1], y[1], y[0]])}
        data = _sf_transform_xy(data, target_crs, source_crs)
        xs = np.asarray(data["x"], dtype=float)
        ys = np.asarray(data["y"], dtype=float)
        xs = xs[np.isfinite(xs)]
        ys = ys[np.isfinite(ys)]
        if xs.size == 0:
            xs = np.asarray(x, dtype=float)
        if ys.size == 0:
            ys = np.asarray(y, dtype=float)
        return {"x": [float(xs.min()), float(xs.max())],
                "y": [float(ys.min()), float(ys.max())]}

    def range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        return {"x": list(panel_params.get("x_range", [0, 1])),
                "y": list(panel_params.get("y_range", [0, 1]))}

    def is_free(self) -> bool:
        return False

    def is_linear(self) -> bool:
        # Non-linear when default_crs is specified (so plain x/y get reprojected).
        return self.get_default_crs() is None

    def distance(self, x: np.ndarray, y: np.ndarray,
                 panel_params: Dict[str, Any]) -> np.ndarray:
        d = self.backtransform_range(panel_params)
        max_dist = math.hypot(d["x"][1] - d["x"][0], d["y"][1] - d["y"][0])
        if max_dist == 0:
            max_dist = 1.0
        return _dist_euclidean(np.asarray(x), np.asarray(y)) / max_dist

    def aspect(self, panel_params: Dict[str, Any]) -> Optional[float]:
        crs_obj = _coerce_crs(panel_params.get("crs"))
        is_longlat = bool(getattr(crs_obj, "is_geographic", False))
        y_range = list(panel_params.get("y_range", [0, 1]))
        x_range = list(panel_params.get("x_range", [0, 1]))
        if is_longlat:
            mid_y = (y_range[0] + y_range[1]) / 2.0
            ratio = math.cos(mid_y * math.pi / 180.0)
        else:
            ratio = 1.0
        if ratio == 0:
            ratio = 1e-10
        dx = x_range[1] - x_range[0]
        if dx == 0:
            return None
        return (y_range[1] - y_range[0]) / dx / ratio

    def labels(self, labels: Any, panel_params: Dict[str, Any]) -> Any:
        return labels

    # ---- graticule label override (coord-sf.R:112-175) ------------------

    def fixup_graticule_labels(
        self, graticule: pd.DataFrame, scale_x: Any, scale_y: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        if graticule is None or len(graticule) == 0:
            return graticule
        graticule = graticule.copy()

        for axis_type, scale in (("E", scale_x), ("N", scale_y)):
            mask = graticule["type"].values == axis_type
            n_breaks = int(mask.sum())
            if n_breaks == 0:
                continue
            user_labels = getattr(scale, "labels", None)
            if user_labels is None:
                # rep(NA, length(x_breaks)) — clear labels
                graticule.loc[mask, "degree_label"] = ""
            elif _is_waiver_like(user_labels):
                # leave sf's default labels (already populated)
                continue
            else:
                breaks = graticule.loc[mask, "degree"].values
                if callable(user_labels):
                    new_labels = user_labels(breaks)
                else:
                    new_labels = user_labels
                new_labels = list(new_labels)
                if len(new_labels) != n_breaks:
                    cli_abort(
                        f"breaks and labels along {axis_type} direction "
                        f"have different lengths."
                    )
                graticule.loc[mask, "degree_label"] = [
                    str(v) for v in new_labels
                ]
        return graticule

    # ---- render_bg (coord-sf.R:339-358) ---------------------------------

    def render_bg(self, panel_params: Dict[str, Any], theme: Any) -> Any:
        from grid_py import (
            polyline_grob, grob_tree, null_grob, Gpar,
        )
        from ggplot2_py.theme_elements import (
            calc_element, element_render, ElementBlank,
        )

        try:
            el = calc_element("panel.grid.major", theme)
        except Exception:
            el = None

        children = []
        bg = element_render(theme, "panel.background")
        if bg is not None:
            children.append(bg)

        graticule = panel_params.get("graticule")
        if (el is not None and not isinstance(el, ElementBlank)
                and graticule is not None and len(graticule)):
            colour = getattr(el, "colour", None)
            linewidth = getattr(el, "linewidth", None)
            linetype = getattr(el, "linetype", None)
            gp = Gpar(col=colour, lwd=linewidth, lty=linetype)

            x_arr: List[float] = []
            y_arr: List[float] = []
            id_lengths: List[int] = []
            for line in graticule["geometry"]:
                if line is None or line.is_empty:
                    continue
                xs, ys = line.coords.xy
                xs = list(xs); ys = list(ys)
                if len(xs) < 2:
                    continue
                x_arr.extend(xs)
                y_arr.extend(ys)
                id_lengths.append(len(xs))
            if id_lengths:
                children.append(polyline_grob(
                    x=np.asarray(x_arr, dtype=float),
                    y=np.asarray(y_arr, dtype=float),
                    id_lengths=id_lengths,
                    gp=gp, name="graticule",
                ))

        if not children:
            return null_grob()
        return grob_tree(*children, name="grill")


def coord_sf(
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    expand: Union[bool, List[bool]] = True,
    crs: Any = None,
    default_crs: Any = None,
    datum: Any = 4326,
    label_graticule: Any = waiver(),
    label_axes: Any = waiver(),
    lims_method: str = "cross",
    ndiscr: int = 100,
    default: bool = False,
    clip: str = "on",
    reverse: str = "none",
) -> CoordSf:
    """R port of ``coord_sf()`` (coord-sf.R:555-606).

    The ``default=True`` flag is used by ``geom_sf()`` to auto-inject a
    coord that a user-supplied coord will replace (matching R ``c(layer,
    coord_sf(default = TRUE))`` semantics).

    Parameters
    ----------
    xlim, ylim : sequence of float, optional
        Limits for the x / y axes in default-CRS units.
    expand : bool or list of bool
    crs : pyproj.CRS or str/int, optional
        CRS into which all data will be projected.
    default_crs : pyproj.CRS or str/int, optional
        CRS used for non-sf layers and scale limits.
    datum : pyproj.CRS or str/int, default ``4326``
        CRS providing the datum for graticule generation.
    label_graticule : str
        Which graticule end-points to label (subset of ``"NESW"``).
    label_axes : str or dict
        Which graticule lines to label per side (e.g. ``"--EN"``).
    lims_method : {'cross', 'box', 'orthogonal', 'geometry_bbox'}
    ndiscr : int
        Number of segments for graticule discretisation.
    default : bool
        If ``True``, this coord can be replaced by a later coord+ call.
    clip : {'on', 'off'}
    reverse : {'none', 'x', 'y', 'xy'}
    """
    if _is_waiver_like(label_graticule) and _is_waiver_like(label_axes):
        # If both are waiver, apply the standard default (no labels on top
        # / right; meridians on bottom; parallels on left).
        label_graticule_resolved: Any = ""
        label_axes_resolved: Any = "--EN"
    else:
        label_graticule_resolved = "" if _is_waiver_like(label_graticule) else label_graticule
        label_axes_resolved = "" if _is_waiver_like(label_axes) else label_axes

    label_axes_parsed = _parse_axes_labeling(label_axes_resolved)
    if isinstance(label_graticule_resolved, str):
        label_graticule_parsed = list(label_graticule_resolved)
    else:
        cli_abort("Graticule labeling format not recognized.")

    # R: switch limit method to "orthogonal" if not specified and
    # default_crs indicates projected coords (default_crs=NULL).
    if default_crs is None and lims_method == "cross":
        lims_method_resolved = "orthogonal"
    elif lims_method not in ("cross", "box", "orthogonal", "geometry_bbox"):
        cli_abort(
            f"lims_method must be one of 'cross', 'box', 'orthogonal', "
            f"'geometry_bbox'; got {lims_method!r}."
        )
    else:
        lims_method_resolved = lims_method

    return CoordSf(
        limits={"x": list(xlim) if xlim is not None else None,
                "y": list(ylim) if ylim is not None else None},
        expand=expand,
        crs=_coerce_crs(crs) if crs is not None else None,
        default_crs=_coerce_crs(default_crs) if default_crs is not None else None,
        datum=datum,
        label_graticule=label_graticule_parsed,
        label_axes=label_axes_parsed,
        lims_method=lims_method_resolved,
        ndiscr=int(ndiscr),
        default=bool(default),
        clip=clip,
        reverse=reverse,
    )


# ---------------------------------------------------------------------------
# CoordPolar
# ---------------------------------------------------------------------------

class CoordPolar(Coord):
    """Polar coordinate system.

    Attributes
    ----------
    theta : str
        Which variable to map to angle (``"x"`` or ``"y"``).
    r : str
        Which variable to map to radius.
    start : float
        Offset from 12 o'clock in radians.
    direction : int
        1 for clockwise, -1 for anticlockwise.
    """

    theta: str = "x"
    r: str = "y"
    start: float = 0.0
    direction: int = 1
    limits: Dict[str, Any] = {"x": None, "y": None}

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.theta == "x":
            self.r = "y"
        else:
            self.r = "x"

    def aspect(self, ranges: Any) -> float:
        return 1.0

    def is_free(self) -> bool:
        return True

    def distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        panel_params: Dict[str, Any],
        boost: float = 0.75,
    ) -> np.ndarray:
        arc = (self.start, self.start + 2 * math.pi)
        if self.theta == "x":
            r = _rescale(np.asarray(y), from_=tuple(panel_params.get("r.range", [0, 1])))
            theta = _theta_rescale_no_clip(
                np.asarray(x),
                tuple(panel_params.get("theta.range", [0, 1])),
                arc,
                self.direction,
            )
        else:
            r = _rescale(np.asarray(x), from_=tuple(panel_params.get("r.range", [0, 1])))
            theta = _theta_rescale_no_clip(
                np.asarray(y),
                tuple(panel_params.get("theta.range", [0, 1])),
                arc,
                self.direction,
            )
        return _dist_polar(r ** boost, theta)

    def backtransform_range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        return self.range(panel_params)

    def range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        return {
            self.theta: list(panel_params.get("theta.range", [0, 1])),
            self.r: list(panel_params.get("r.range", [0, 1])),
        }

    def setup_panel_params(
        self,
        scale_x: Any,
        scale_y: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = params or {}
        result: Dict[str, Any] = {}
        for name, scale in [("x", scale_x), ("y", scale_y)]:
            limits = self.limits.get(name)
            rng = _scale_numeric_range(scale, [0, 1])
            if limits is not None:
                rng = list(limits)

            is_theta = (self.theta == name)
            prefix = "theta" if is_theta else "r"

            result[f"{prefix}.range"] = rng
            if hasattr(scale, "break_info"):
                info = scale.break_info(rng)
                result[f"{prefix}.major"] = info.get("major_source")
                result[f"{prefix}.minor"] = info.get("minor_source")
                result[f"{prefix}.labels"] = info.get("labels")
            else:
                result[f"{prefix}.major"] = None
                result[f"{prefix}.minor"] = None
                result[f"{prefix}.labels"] = None

        return result

    def setup_panel_guides(
        self,
        panel_params: Dict[str, Any],
        guides: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # CoordPolar cannot render standard guides
        return panel_params

    def train_panel_guides(
        self,
        panel_params: Dict[str, Any],
        layers: list,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return panel_params

    def transform(self, data: pd.DataFrame, panel_params: Dict[str, Any]) -> pd.DataFrame:
        arc = (self.start, self.start + 2 * math.pi)
        direction = self.direction
        data = data.copy()

        # Rename x/y to theta/r based on self.theta
        if self.theta == "x":
            theta_col, r_col = "x", "y"
        else:
            theta_col, r_col = "y", "x"

        r_range = panel_params.get("r.range", [0, 1])
        theta_range = panel_params.get("theta.range", [0, 1])

        if r_col in data.columns:
            data["__r__"] = _r_rescale(data[r_col].values, tuple(r_range))
        else:
            data["__r__"] = 0.0

        if theta_col in data.columns:
            data["__theta__"] = _theta_rescale(
                data[theta_col].values, tuple(theta_range), arc, direction
            )
        else:
            data["__theta__"] = 0.0

        data["x"] = data["__r__"] * np.sin(data["__theta__"]) + 0.5
        data["y"] = data["__r__"] * np.cos(data["__theta__"]) + 0.5
        data.drop(columns=["__r__", "__theta__"], inplace=True, errors="ignore")
        return data

    def render_bg(self, panel_params: Dict[str, Any], theme: Any) -> Any:
        return guide_grid(theme, panel_params, self)

    def render_axis_h(self, panel_params: Dict[str, Any], theme: Any) -> Dict[str, Any]:
        from grid_py import null_grob
        return {"top": null_grob(), "bottom": null_grob()}

    def render_axis_v(self, panel_params: Dict[str, Any], theme: Any) -> Dict[str, Any]:
        from grid_py import null_grob
        return {"left": null_grob(), "right": null_grob()}

    def labels(self, labels: Dict[str, Any], panel_params: Dict[str, Any]) -> Dict[str, Any]:
        if self.theta == "y":
            return {"x": labels.get("y", {}), "y": labels.get("x", {})}
        return labels


# ---------------------------------------------------------------------------
# CoordRadial
# ---------------------------------------------------------------------------

class CoordRadial(Coord):
    """Modern radial (polar) coordinate system.

    Attributes
    ----------
    theta : str
        Angle variable (``"x"`` or ``"y"``).
    r : str
        Radius variable.
    arc : tuple of float
        Start and end of the arc in radians.
    r_axis_inside : bool or float or None
        Whether the r-axis is drawn inside the panel.
    rotate_angle : bool
        Whether to transform the ``angle`` aesthetic.
    inner_radius : tuple of float
        Inner and outer radius proportions.
    """

    theta: str = "x"
    r: str = "y"
    arc: Tuple[float, float] = (0.0, 2 * math.pi)
    r_axis_inside: Any = None
    rotate_angle: bool = False
    inner_radius: Tuple[float, float] = (0.0, 0.4)
    limits: Dict[str, Any] = {"theta": None, "r": None}

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.theta == "x":
            self.r = "y"
        else:
            self.r = "x"

    def aspect(self, details: Any) -> float:
        bbox = details.get("bbox", {"x": [0, 1], "y": [0, 1]})
        dx = bbox["x"][1] - bbox["x"][0]
        dy = bbox["y"][1] - bbox["y"][0]
        return dy / max(dx, 1e-10)

    def is_free(self) -> bool:
        return True

    def distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        details: Dict[str, Any],
        boost: float = 0.75,
    ) -> np.ndarray:
        arc = details.get("arc") or self.arc
        inner = self.inner_radius
        if self.theta == "x":
            r = _rescale(np.asarray(y), from_=tuple(details.get("r.range", [0, 1])),
                         to=(inner[0] / 0.4, inner[1] / 0.4))
            theta = _theta_rescale_no_clip(np.asarray(x), tuple(details.get("theta.range", [0, 1])), arc)
        else:
            r = _rescale(np.asarray(x), from_=tuple(details.get("r.range", [0, 1])),
                         to=(inner[0] / 0.4, inner[1] / 0.4))
            theta = _theta_rescale_no_clip(np.asarray(y), tuple(details.get("theta.range", [0, 1])), arc)
        return _dist_polar(r ** boost, theta)

    def backtransform_range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        return self.range(panel_params)

    def range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        return {
            self.theta: list(panel_params.get("theta.range", [0, 1])),
            self.r: list(panel_params.get("r.range", [0, 1])),
        }

    def setup_panel_params(
        self,
        scale_x: Any,
        scale_y: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = params or {}
        result: Dict[str, Any] = {}

        if self.theta == "x":
            theta_limits = self.limits.get("theta")
            r_limits = self.limits.get("r")
            theta_scale, r_scale = scale_x, scale_y
        else:
            theta_limits = self.limits.get("theta")
            r_limits = self.limits.get("r")
            theta_scale, r_scale = scale_y, scale_x

        # Theta
        theta_range = _scale_numeric_range(theta_scale, [0, 1])
        if theta_limits is not None:
            theta_range = list(theta_limits)

        # R
        r_range = _scale_numeric_range(r_scale, [0, 1])
        if r_limits is not None:
            r_range = list(r_limits)

        result["theta.range"] = theta_range
        result["r.range"] = r_range
        result["bbox"] = _polar_bbox(self.arc, inner_radius=self.inner_radius)
        result["arc"] = self.arc
        result["inner_radius"] = self.inner_radius

        return result

    def transform(self, data: pd.DataFrame, panel_params: Dict[str, Any]) -> pd.DataFrame:
        data = data.copy()
        bbox = panel_params.get("bbox", {"x": [0, 1], "y": [0, 1]})
        arc = panel_params.get("arc", self.arc)
        inner_radius = panel_params.get("inner_radius", self.inner_radius)

        if self.theta == "x":
            theta_col, r_col = "x", "y"
        else:
            theta_col, r_col = "y", "x"

        r_range = panel_params.get("r.range", [0, 1])
        theta_range = panel_params.get("theta.range", [0, 1])

        if r_col in data.columns:
            data["__r__"] = _r_rescale(data[r_col].values, tuple(r_range), donut=inner_radius)
        else:
            data["__r__"] = 0.0

        if theta_col in data.columns:
            data["__theta__"] = _theta_rescale(
                data[theta_col].values, tuple(theta_range), arc
            )
        else:
            data["__theta__"] = 0.0

        raw_x = data["__r__"] * np.sin(data["__theta__"]) + 0.5
        raw_y = data["__r__"] * np.cos(data["__theta__"]) + 0.5
        data["x"] = _rescale(raw_x, from_=tuple(bbox["x"]))
        data["y"] = _rescale(raw_y, from_=tuple(bbox["y"]))
        data.drop(columns=["__r__", "__theta__"], inplace=True, errors="ignore")
        return data

    def render_bg(self, panel_params: Dict[str, Any], theme: Any) -> Any:
        return guide_grid(theme, panel_params, self)

    def render_axis_h(self, panel_params: Dict[str, Any], theme: Any) -> Dict[str, Any]:
        from grid_py import null_grob
        return {"top": null_grob(), "bottom": null_grob()}

    def render_axis_v(self, panel_params: Dict[str, Any], theme: Any) -> Dict[str, Any]:
        from grid_py import null_grob
        return {"left": null_grob(), "right": null_grob()}

    def labels(self, labels: Dict[str, Any], panel_params: Dict[str, Any]) -> Dict[str, Any]:
        if self.theta == "y":
            return {"x": labels.get("y", {}), "y": labels.get("x", {})}
        return labels


def _polar_bbox(
    arc: Tuple[float, float],
    margin: Tuple[float, float, float, float] = (0.05, 0.05, 0.05, 0.05),
    inner_radius: Tuple[float, float] = (0.0, 0.4),
) -> Dict[str, list]:
    """Compute bounding box for a partial polar chart.

    Parameters
    ----------
    arc : tuple of float
        Start and end angles in radians.
    margin : tuple
        Margins (top, right, bottom, left).
    inner_radius : tuple
        Inner and outer radii.

    Returns
    -------
    dict
        ``{"x": [xmin, xmax], "y": [ymin, ymax]}``.
    """
    if abs(arc[1] - arc[0]) >= 2 * math.pi:
        return {"x": [0.0, 1.0], "y": [0.0, 1.0]}

    sorted_arc = (min(arc), max(arc))
    angles = np.array([sorted_arc[0], sorted_arc[1]])
    x_outer = 0.5 * np.sin(angles) + 0.5
    y_outer = 0.5 * np.cos(angles) + 0.5

    # Check cardinal directions
    cardinal = np.array([0, 0.5 * math.pi, math.pi, 1.5 * math.pi])
    in_sector = _in_arc(cardinal, sorted_arc)

    # top, right, bottom, left extremes
    bounds = [
        1.0 if in_sector[0] else max(float(np.max(y_outer)), 0.5 + margin[0]),
        1.0 if in_sector[1] else max(float(np.max(x_outer)), 0.5 + margin[1]),
        0.0 if in_sector[2] else min(float(np.min(y_outer)), 0.5 - margin[2]),
        0.0 if in_sector[3] else min(float(np.min(x_outer)), 0.5 - margin[3]),
    ]
    return {"x": [bounds[3], bounds[1]], "y": [bounds[2], bounds[0]]}


def _in_arc(theta: np.ndarray, arc: Tuple[float, float]) -> np.ndarray:
    """Test whether angles are inside an arc."""
    theta = np.asarray(theta)
    if abs(arc[1] - arc[0]) >= 2 * math.pi - 1e-8:
        return np.ones(len(theta), dtype=bool)
    a0 = arc[0] % (2 * math.pi)
    a1 = arc[1] % (2 * math.pi)
    if a0 < a1:
        return (theta >= a0) & (theta <= a1)
    else:
        return ~((theta < a0) & (theta > a1))


# ---------------------------------------------------------------------------
# CoordTransform
# ---------------------------------------------------------------------------

class CoordTransform(Coord):
    """Transformed Cartesian coordinate system.

    Applies arbitrary transformations to x and y.

    Attributes
    ----------
    trans : dict
        ``{"x": transform, "y": transform}`` where each transform has
        ``transform()`` and ``inverse()`` methods.
    limits : dict
        Coordinate limits.
    """

    trans: Dict[str, Any] = {"x": None, "y": None}
    limits: Dict[str, Any] = {"x": None, "y": None}

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def is_free(self) -> bool:
        return True

    def distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        panel_params: Dict[str, Any],
    ) -> np.ndarray:
        x_range = panel_params.get("x.range", [0, 1])
        y_range = panel_params.get("y.range", [0, 1])
        max_dist = np.sqrt((x_range[1] - x_range[0]) ** 2 + (y_range[1] - y_range[0]) ** 2)
        if max_dist == 0:
            max_dist = 1.0
        tx = self.trans["x"].transform(np.asarray(x)) if self.trans.get("x") else np.asarray(x)
        ty = self.trans["y"].transform(np.asarray(y)) if self.trans.get("y") else np.asarray(y)
        return _dist_euclidean(tx, ty) / max_dist

    def backtransform_range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        x_range = panel_params.get("x.range", [0, 1])
        y_range = panel_params.get("y.range", [0, 1])
        inv_x = self.trans["x"].inverse(np.array(x_range)) if self.trans.get("x") else x_range
        inv_y = self.trans["y"].inverse(np.array(y_range)) if self.trans.get("y") else y_range
        return {"x": list(inv_x), "y": list(inv_y)}

    def range(self, panel_params: Dict[str, Any]) -> Dict[str, list]:
        return {
            "x": list(panel_params.get("x.range", [0, 1])),
            "y": list(panel_params.get("y.range", [0, 1])),
        }

    def transform(self, data: pd.DataFrame, panel_params: Dict[str, Any]) -> pd.DataFrame:
        reverse = panel_params.get("reverse") or getattr(self, "reverse", "none")
        x_range = list(panel_params.get("x.range", [0, 1]))
        y_range = list(panel_params.get("y.range", [0, 1]))

        if reverse in ("x", "xy"):
            x_range = list(reversed(x_range))
        if reverse in ("y", "xy"):
            y_range = list(reversed(y_range))

        trans_x = self.trans.get("x")
        trans_y = self.trans.get("y")

        def apply_trans_x(vals: np.ndarray) -> np.ndarray:
            vals = np.asarray(vals, dtype=float)
            finite = np.isfinite(vals)
            if trans_x is not None and np.any(finite):
                vals[finite] = _rescale(
                    trans_x.transform(vals[finite]),
                    to=(0, 1),
                    from_=tuple(x_range),
                )
            else:
                vals = _rescale(vals, to=(0, 1), from_=tuple(x_range))
            return vals

        def apply_trans_y(vals: np.ndarray) -> np.ndarray:
            vals = np.asarray(vals, dtype=float)
            finite = np.isfinite(vals)
            if trans_y is not None and np.any(finite):
                vals[finite] = _rescale(
                    trans_y.transform(vals[finite]),
                    to=(0, 1),
                    from_=tuple(y_range),
                )
            else:
                vals = _rescale(vals, to=(0, 1), from_=tuple(y_range))
            return vals

        data = _transform_position(data, apply_trans_x, apply_trans_y)
        data = _transform_position(data, _squish_infinite, _squish_infinite)
        return data

    def setup_panel_params(
        self,
        scale_x: Any,
        scale_y: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = params or {}
        x_limits = self.limits.get("x")
        y_limits = self.limits.get("y")

        x_range = _scale_numeric_range(scale_x, [0, 1])
        if x_limits is not None:
            x_range = list(x_limits)

        y_range = _scale_numeric_range(scale_y, [0, 1])
        if y_limits is not None:
            y_range = list(y_limits)

        return {
            "x_range": x_range,
            "y_range": y_range,
            "x.range": x_range,
            "y.range": y_range,
            "reverse": getattr(self, "reverse", "none"),
        }

    def render_bg(self, panel_params: Dict[str, Any], theme: Any) -> Any:
        return guide_grid(theme, panel_params, self)


# Alias for backward compatibility
CoordTrans = CoordTransform


# ---------------------------------------------------------------------------
# coord_munch
# ---------------------------------------------------------------------------

def coord_munch(
    coord: Coord,
    data: pd.DataFrame,
    range_: Dict[str, Any],
    n: int = 50,
    is_closed: bool = False,
) -> pd.DataFrame:
    """Interpolate path data for non-linear coordinate systems.

    For linear coordinates, the data is returned unchanged (after
    transformation).  For non-linear coordinates, points are
    interpolated so that straight lines in data space become curves
    in plot space.

    Parameters
    ----------
    coord : Coord
        Coordinate system.
    data : pd.DataFrame
        Data with ``x`` and ``y`` columns (at minimum).
    range_ : dict
        Panel parameters / ranges.
    n : int
        Maximum number of interpolation points per segment.
    is_closed : bool
        Whether the path is closed (polygon).

    Returns
    -------
    pd.DataFrame
        Transformed (and possibly interpolated) data.
    """
    if coord.is_linear():
        return coord.transform(data, range_)

    # For non-linear coords, interpolate
    if len(data) < 2:
        return coord.transform(data, range_)

    # Compute distances to determine segment counts
    x = data["x"].values
    y = data["y"].values
    dist = coord.distance(x, y, range_)

    # Interpolate segments that are long
    if len(dist) == 0:
        return coord.transform(data, range_)

    # Determine how many points each segment needs
    max_dist = float(np.nanmax(dist)) if len(dist) > 0 else 0.0
    if max_dist == 0:
        return coord.transform(data, range_)

    # Simple approach: subdivide each segment proportionally
    segments = np.ceil(dist / max_dist * n).astype(int)
    segments = np.clip(segments, 1, n)

    rows = []
    for i in range(len(data) - 1):
        nseg = int(segments[i]) if i < len(segments) else 1
        row_start = data.iloc[i]
        row_end = data.iloc[i + 1]
        for j in range(nseg):
            t = j / nseg
            new_row = {}
            for col in data.columns:
                v0 = row_start[col]
                v1 = row_end[col]
                if isinstance(v0, (int, float, np.integer, np.floating)):
                    new_row[col] = v0 + (v1 - v0) * t
                else:
                    new_row[col] = v0
            rows.append(new_row)
    # Last point
    rows.append(dict(data.iloc[-1]))

    munched = pd.DataFrame(rows)
    return coord.transform(munched, range_)


# ---------------------------------------------------------------------------
# Constructor functions
# ---------------------------------------------------------------------------

def coord_cartesian(
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    expand: Union[bool, List[bool]] = True,
    default: bool = False,
    clip: str = "on",
    reverse: str = "none",
    ratio: Optional[float] = None,
) -> CoordCartesian:
    """Create a Cartesian coordinate system.

    Parameters
    ----------
    xlim, ylim : sequence of float or None
        Limits for zooming (does not filter data).
    expand : bool or list of bool
        Whether to expand limits to avoid data/axis overlap.
    default : bool
        Whether this is the default coord.
    clip : str
        Clipping: ``"on"`` or ``"off"``.
    reverse : str
        ``"none"``, ``"x"``, ``"y"``, or ``"xy"``.
    ratio : float or None
        Fixed aspect ratio.

    Returns
    -------
    CoordCartesian
    """
    return CoordCartesian(
        limits={"x": list(xlim) if xlim is not None else None,
                "y": list(ylim) if ylim is not None else None},
        expand=expand,
        default=default,
        clip=clip,
        reverse=reverse,
        ratio=ratio,
    )


def coord_fixed(ratio: float = 1.0, **kwargs: Any) -> CoordFixed:
    """Create a fixed-ratio coordinate system.

    Parameters
    ----------
    ratio : float
        Aspect ratio (y/x).
    **kwargs
        Passed to :class:`CoordFixed`.

    Returns
    -------
    CoordFixed
    """
    obj = CoordFixed(ratio=ratio, **kwargs)
    if "limits" not in kwargs:
        obj.limits = {
            "x": kwargs.get("xlim"),
            "y": kwargs.get("ylim"),
        }
    return obj


coord_equal = coord_fixed


def coord_flip(
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    expand: Union[bool, List[bool]] = True,
    clip: str = "on",
) -> CoordFlip:
    """Create a flipped Cartesian coordinate system.

    Parameters
    ----------
    xlim, ylim : sequence of float or None
    expand : bool or list
    clip : str

    Returns
    -------
    CoordFlip
    """
    return CoordFlip(
        limits={"x": list(xlim) if xlim is not None else None,
                "y": list(ylim) if ylim is not None else None},
        expand=expand,
        clip=clip,
    )


def coord_polar(
    theta: str = "x",
    start: float = 0.0,
    direction: int = 1,
    clip: str = "on",
) -> CoordPolar:
    """Create a polar coordinate system.

    Parameters
    ----------
    theta : str
        ``"x"`` or ``"y"``.
    start : float
        Offset from 12 o'clock in radians.
    direction : int
        1 for clockwise, -1 for anticlockwise.
    clip : str

    Returns
    -------
    CoordPolar
    """
    if theta not in ("x", "y"):
        cli_abort("theta must be 'x' or 'y'.")
    return CoordPolar(
        theta=theta,
        start=start,
        direction=int(np.sign(direction)),
        clip=clip,
    )


def coord_radial(
    theta: str = "x",
    start: float = 0.0,
    end: Optional[float] = None,
    thetalim: Optional[Sequence[float]] = None,
    rlim: Optional[Sequence[float]] = None,
    expand: Union[bool, List[bool]] = True,
    clip: str = "off",
    r_axis_inside: Any = None,
    rotate_angle: bool = False,
    inner_radius: float = 0.0,
    reverse: str = "none",
) -> CoordRadial:
    """Create a radial coordinate system.

    Parameters
    ----------
    theta : str
        ``"x"`` or ``"y"``.
    start : float
        Start angle in radians.
    end : float or None
        End angle.  Defaults to ``start + 2*pi``.
    thetalim, rlim : sequence or None
        Limits for theta and r.
    expand : bool or list
    clip : str
    r_axis_inside : bool, float, or None
    rotate_angle : bool
    inner_radius : float
        Between 0 and 1.
    reverse : str
        ``"none"``, ``"theta"``, ``"r"``, or ``"thetar"``.

    Returns
    -------
    CoordRadial
    """
    if theta not in ("x", "y"):
        cli_abort("theta must be 'x' or 'y'.")
    if reverse not in ("none", "theta", "r", "thetar"):
        cli_abort("reverse must be 'none', 'theta', 'r', or 'thetar'.")

    arc_end = end if end is not None else (start + 2 * math.pi)
    arc = (start, arc_end)

    if arc[0] > arc[1]:
        n_rot = int((arc[0] - arc[1]) // (2 * math.pi)) + 1
        arc = (arc[0] - n_rot * 2 * math.pi, arc[1])

    if reverse in ("theta", "thetar"):
        arc = (arc[1], arc[0])

    inner = (inner_radius, 1.0)
    inner = (inner[0] * 0.4, inner[1] * 0.4)
    if reverse in ("r", "thetar"):
        inner = (inner[1], inner[0])

    return CoordRadial(
        theta=theta,
        arc=arc,
        limits={"theta": list(thetalim) if thetalim is not None else None,
                "r": list(rlim) if rlim is not None else None},
        expand=expand,
        clip=clip,
        r_axis_inside=r_axis_inside,
        rotate_angle=rotate_angle,
        inner_radius=inner,
        reverse=reverse,
    )


def coord_transform(
    x: Any = "identity",
    y: Any = "identity",
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    clip: str = "on",
    expand: Union[bool, List[bool]] = True,
    reverse: str = "none",
) -> CoordTransform:
    """Create a transformed coordinate system.

    Parameters
    ----------
    x, y : str or transform
        Transformations for x and y.
    xlim, ylim : sequence or None
    clip : str
    expand : bool or list
    reverse : str

    Returns
    -------
    CoordTransform
    """
    from scales import as_transform

    if isinstance(x, str):
        x = as_transform(x)
    if isinstance(y, str):
        y = as_transform(y)

    return CoordTransform(
        trans={"x": x, "y": y},
        limits={"x": list(xlim) if xlim is not None else None,
                "y": list(ylim) if ylim is not None else None},
        expand=expand,
        reverse=reverse,
        clip=clip,
    )


def coord_trans(**kwargs: Any) -> CoordTransform:
    """Deprecated alias for :func:`coord_transform`.

    Parameters
    ----------
    **kwargs
        Passed to :func:`coord_transform`.

    Returns
    -------
    CoordTransform
    """
    cli_warn("coord_trans() is deprecated; use coord_transform().")
    return coord_transform(**kwargs)


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

def is_coord(x: Any) -> bool:
    """Test whether *x* is a Coord.

    Parameters
    ----------
    x : object

    Returns
    -------
    bool
    """
    return isinstance(x, Coord)


def is_Coord(x: Any) -> bool:
    """Deprecated alias for :func:`is_coord`.

    Parameters
    ----------
    x : object

    Returns
    -------
    bool
    """
    return is_coord(x)
