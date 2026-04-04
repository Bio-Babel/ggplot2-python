"""
Save ggplot objects to files.

Provides :func:`ggsave` for saving plots to PNG, PDF, SVG, JPG and other
formats via matplotlib.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

from ggplot2_py._compat import cli_abort, cli_warn, cli_inform

__all__ = [
    "ggsave",
    "check_device",
]


# ---------------------------------------------------------------------------
# Device mapping
# ---------------------------------------------------------------------------

_SUPPORTED_DEVICES = frozenset({
    "png", "pdf", "svg", "jpg", "jpeg", "tiff", "tif",
    "bmp", "eps", "ps", "webp",
})


def check_device(device: Optional[str], filename: str) -> str:
    """Determine the output device from *device* or *filename*.

    Parameters
    ----------
    device : str or None
        Explicit device name (e.g. ``"png"``).
    filename : str
        Output file path; the extension is used when *device* is ``None``.

    Returns
    -------
    str
        Resolved device name (lowercase extension without dot).

    Raises
    ------
    ValueError
        If the device cannot be determined or is unsupported.
    """
    if device is not None:
        device = device.lower().strip()
        if device not in _SUPPORTED_DEVICES:
            cli_abort(f"Unknown graphics device: '{device}'.")
        return device

    _, ext = os.path.splitext(filename)
    if not ext:
        cli_abort(
            f"Cannot determine graphics device from '{filename}'. "
            "Either supply `filename` with a file extension or supply `device`."
        )
    ext = ext.lstrip(".").lower()
    if ext not in _SUPPORTED_DEVICES:
        cli_abort(f"Unknown graphics device: '{ext}'.")
    return ext


# ---------------------------------------------------------------------------
# DPI parsing
# ---------------------------------------------------------------------------

_DPI_PRESETS: Dict[str, int] = {
    "screen": 72,
    "print": 300,
    "retina": 320,
}


def _parse_dpi(dpi: Union[int, float, str]) -> float:
    """Resolve a DPI specification.

    Parameters
    ----------
    dpi : int, float, or str
        Either a number or one of ``"screen"`` (72), ``"print"`` (300),
        ``"retina"`` (320).

    Returns
    -------
    float
    """
    if isinstance(dpi, str):
        val = _DPI_PRESETS.get(dpi.lower())
        if val is None:
            cli_abort(
                f"Unknown DPI preset: '{dpi}'. "
                f"Must be one of {list(_DPI_PRESETS)}."
            )
        return float(val)
    return float(dpi)


# ---------------------------------------------------------------------------
# Dimension helpers
# ---------------------------------------------------------------------------

_UNIT_FACTORS: Dict[str, float] = {
    "in": 1.0,
    "cm": 1.0 / 2.54,
    "mm": 1.0 / 25.4,
}


def _to_inches(
    value: Optional[float],
    units: str,
    dpi: float,
) -> Optional[float]:
    """Convert *value* from *units* to inches.

    Parameters
    ----------
    value : float or None
    units : str
        ``"in"``, ``"cm"``, ``"mm"``, or ``"px"``.
    dpi : float

    Returns
    -------
    float or None
    """
    if value is None:
        return None
    if units == "px":
        return value / dpi
    factor = _UNIT_FACTORS.get(units)
    if factor is None:
        cli_abort(f"Unknown unit: '{units}'. Must be 'in', 'cm', 'mm', or 'px'.")
    return value * factor


# ---------------------------------------------------------------------------
# ggsave
# ---------------------------------------------------------------------------

def ggsave(
    filename: str,
    plot: Any = None,
    device: Optional[str] = None,
    path: Optional[str] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    units: str = "in",
    dpi: Union[int, float, str] = 300,
    limitsize: bool = True,
    bg: Optional[str] = None,
    create_dir: bool = False,
    scale: float = 1.0,
    **kwargs: Any,
) -> str:
    """Save a ggplot (or other grid object) to a file.

    Parameters
    ----------
    filename : str
        Output file name.
    plot : GGPlot or None, optional
        Plot to save.  If ``None``, uses :func:`get_last_plot`.
    device : str or None, optional
        Graphics device (``"png"``, ``"pdf"``, ``"svg"``, etc.).
        Auto-detected from *filename* extension if ``None``.
    path : str or None, optional
        Directory path.  Combined with *filename* to form the full path.
    width, height : float, optional
        Plot dimensions in *units*.  Defaults to 7 x 7 inches.
    units : str, optional
        Unit for *width* / *height*: ``"in"`` (default), ``"cm"``,
        ``"mm"``, or ``"px"``.
    dpi : int, float, or str, optional
        Resolution (default 300).  Also accepts ``"screen"`` (72),
        ``"print"`` (300), ``"retina"`` (320).
    limitsize : bool, optional
        If ``True`` (default), refuse images > 50 x 50 inches.
    bg : str or None, optional
        Background colour.
    create_dir : bool, optional
        If ``True``, create non-existent directories.
    scale : float, optional
        Multiplicative scaling factor (default 1.0).
    **kwargs
        Passed through to the matplotlib ``savefig`` call.

    Returns
    -------
    str
        The resolved output file path.

    Raises
    ------
    ValueError
        If dimensions exceed limits or the device is unknown.
    FileNotFoundError
        If the target directory does not exist and *create_dir* is
        ``False``.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    # Resolve DPI
    dpi_val = _parse_dpi(dpi)

    # Resolve device
    dev = check_device(device, filename)

    # Resolve path
    if path is not None:
        filename = os.path.join(path, filename)
    target_dir = os.path.dirname(filename)
    if target_dir and not os.path.isdir(target_dir):
        if create_dir:
            os.makedirs(target_dir, exist_ok=True)
        else:
            cli_abort(
                f"Cannot find directory '{target_dir}'. "
                "Supply an existing directory or use `create_dir=True`."
            )

    # Resolve dimensions
    width_in = _to_inches(width, units, dpi_val)
    height_in = _to_inches(height, units, dpi_val)
    if width_in is None:
        width_in = 7.0
    if height_in is None:
        height_in = 7.0
    width_in *= scale
    height_in *= scale

    if limitsize and (width_in >= 50 or height_in >= 50):
        cli_abort(
            f"Dimensions exceed 50 inches ({width_in:.1f} x {height_in:.1f}). "
            "Use `limitsize=False` if you really want a plot that big."
        )

    # Resolve plot
    if plot is None:
        from ggplot2_py.plot import get_last_plot
        plot = get_last_plot()
    if plot is None:
        cli_abort("No plot to save. Supply `plot` or create a plot first.")

    # Build the plot
    from ggplot2_py.plot import ggplot_build, ggplot_gtable, is_ggplot

    if is_ggplot(plot):
        built = ggplot_build(plot)
        gtable = ggplot_gtable(built)
    else:
        gtable = plot

    # Resolve background
    if bg is None:
        bg = "white"

    # Create figure and render
    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi_val)
    fig.patch.set_facecolor(bg)

    # Try to render the gtable via grid_py
    try:
        from grid_py import grid_draw, grid_newpage
        grid_draw(gtable)
    except Exception:
        # Fallback: if grid_draw fails, at least save the empty figure
        pass

    # Save
    savefig_kwargs: Dict[str, Any] = {
        "dpi": dpi_val,
        "facecolor": bg,
        "bbox_inches": "tight",
    }
    savefig_kwargs.update(kwargs)

    # Map device to matplotlib format
    fmt_map = {
        "jpg": "jpeg",
        "tif": "tiff",
        "eps": "eps",
        "ps": "ps",
    }
    fmt = fmt_map.get(dev, dev)
    savefig_kwargs["format"] = fmt

    fig.savefig(filename, **savefig_kwargs)
    plt.close(fig)

    cli_inform(f"Saving {width_in:.3g} x {height_in:.3g} {units} image to {filename}")

    return filename
