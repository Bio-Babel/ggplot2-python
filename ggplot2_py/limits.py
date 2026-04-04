"""
Scale-limit shortcuts for ggplot2.

Provides ``xlim()``, ``ylim()``, ``lims()``, and ``expand_limits()`` for
quickly setting axis and aesthetic limits.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ggplot2_py._compat import cli_abort

__all__ = [
    "lims",
    "xlim",
    "ylim",
    "expand_limits",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_scale(
    scale_type: str,
    var: str,
    *,
    limits: Any = None,
    transform: str = "identity",
) -> Any:
    """Construct a scale via name look-up.

    Parameters
    ----------
    scale_type : str
        One of ``"continuous"``, ``"discrete"``, ``"date"``, ``"datetime"``.
    var : str
        Aesthetic variable name (e.g. ``"x"``, ``"y"``, ``"colour"``).
    limits : object
        The limits to pass to the scale constructor.
    transform : str
        Transform name (only meaningful for continuous scales).

    Returns
    -------
    Scale
        A configured Scale instance.
    """
    # Lazy imports to avoid circular dependency
    import ggplot2_py.scales as _scales_mod

    name = f"scale_{var}_{scale_type}"
    scale_fn = getattr(_scales_mod, name, None)
    if scale_fn is None:
        cli_abort(f"Unknown scale function: {name}")

    kwargs: dict = {"limits": limits}
    if scale_type == "continuous":
        kwargs["transform"] = transform

    return scale_fn(**kwargs)


def _limits_numeric(lims: Sequence, var: str) -> Any:
    """Create a continuous scale with numeric limits.

    Parameters
    ----------
    lims : sequence of float
        Two-element sequence ``[lower, upper]``.  ``None``/``np.nan``
        elements are treated as auto-computed.
    var : str
        Aesthetic name.

    Returns
    -------
    Scale
    """
    if len(lims) != 2:
        cli_abort(
            f"`{var}` must be a two-element numeric sequence, "
            f"got length {len(lims)}."
        )
    low, high = lims
    # Determine transform
    has_low = low is not None and not (isinstance(low, float) and np.isnan(low))
    has_high = high is not None and not (isinstance(high, float) and np.isnan(high))
    if has_low and has_high and low > high:
        transform = "reverse"
    else:
        transform = "identity"
    return _make_scale("continuous", var, limits=list(lims), transform=transform)


def _limits_character(lims: Sequence[str], var: str) -> Any:
    """Create a discrete scale with character limits.

    Parameters
    ----------
    lims : sequence of str
        Factor / category levels.
    var : str
        Aesthetic name.

    Returns
    -------
    Scale
    """
    return _make_scale("discrete", var, limits=list(lims))


def _limits_date(lims: Sequence, var: str) -> Any:
    """Create a date scale with date limits.

    Parameters
    ----------
    lims : sequence
        Two-element date sequence.
    var : str
        Aesthetic name.

    Returns
    -------
    Scale
    """
    if len(lims) != 2:
        cli_abort(
            f"`{var}` must be a two-element date sequence, "
            f"got length {len(lims)}."
        )
    return _make_scale("date", var, limits=list(lims))


def _limits_dispatch(lims: Any, var: str) -> Any:
    """Dispatch to the correct limit constructor based on *lims* type.

    Parameters
    ----------
    lims : object
        Limits specification.
    var : str
        Aesthetic name.

    Returns
    -------
    Scale
    """
    if isinstance(lims, (list, tuple)):
        if len(lims) == 0:
            cli_abort(f"`{var}` limits must be non-empty.")
        first = next((v for v in lims if v is not None), None)
        if first is None or isinstance(first, (int, float, np.integer, np.floating)):
            return _limits_numeric(lims, var)
        elif isinstance(first, str):
            return _limits_character(lims, var)
        elif isinstance(first, (pd.Timestamp, np.datetime64)):
            return _limits_date(lims, var)
        else:
            # Try numeric
            return _limits_numeric(lims, var)
    elif isinstance(lims, np.ndarray):
        return _limits_numeric(list(lims), var)
    else:
        cli_abort(
            f"Cannot create limits for `{var}` from type {type(lims).__name__}."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def xlim(*args: Any) -> Any:
    """Set x-axis limits.

    Parameters
    ----------
    *args
        Either a single two-element sequence or two scalar values
        specifying the lower and upper bounds.  ``None`` leaves the
        corresponding limit to be computed from the data.

    Returns
    -------
    Scale
        A position scale with the specified limits.

    Examples
    --------
    >>> xlim(0, 10)
    >>> xlim(10, 0)    # reversed
    >>> xlim(None, 20) # auto lower bound
    """
    if len(args) == 1 and isinstance(args[0], (list, tuple, np.ndarray)):
        values = list(args[0])
    else:
        values = list(args)
    return _limits_dispatch(values, "x")


def ylim(*args: Any) -> Any:
    """Set y-axis limits.

    Parameters
    ----------
    *args
        Either a single two-element sequence or two scalar values
        specifying the lower and upper bounds.

    Returns
    -------
    Scale
        A position scale with the specified limits.

    Examples
    --------
    >>> ylim(0, 50)
    """
    if len(args) == 1 and isinstance(args[0], (list, tuple, np.ndarray)):
        values = list(args[0])
    else:
        values = list(args)
    return _limits_dispatch(values, "y")


def lims(**kwargs: Any) -> List[Any]:
    """Set limits for one or more aesthetics.

    Parameters
    ----------
    **kwargs
        Keyword arguments mapping aesthetic names to limit specifications.
        Numeric limits create continuous scales; string limits create
        discrete scales.

    Returns
    -------
    list of Scale
        A list of scale objects suitable for adding to a ggplot via ``+``.

    Examples
    --------
    >>> lims(x=(0, 10), y=(0, 50), colour=["red", "blue"])
    """
    result: List[Any] = []
    for var, lim in kwargs.items():
        if isinstance(lim, (list, tuple)):
            result.append(_limits_dispatch(lim, var))
        else:
            cli_abort(
                f"Limits for `{var}` must be a list or tuple, "
                f"got {type(lim).__name__}."
            )
    return result


def expand_limits(**kwargs: Any) -> Any:
    """Expand plot limits to include specified values.

    Creates a ``geom_blank`` layer with data containing the specified
    values, which forces the scales to expand.

    Parameters
    ----------
    **kwargs
        Aesthetic-name = value pairs.  Each value (or list of values)
        will be included in the scale training for the corresponding
        aesthetic.

    Returns
    -------
    Layer
        A ``geom_blank`` layer with the specified data.

    Examples
    --------
    >>> expand_limits(x=0, y=[0, 100])
    """
    # Build a DataFrame from the kwargs, repeating shorter vectors
    data: dict = {}
    for k, v in kwargs.items():
        if isinstance(v, (list, tuple, np.ndarray)):
            data[k] = list(v)
        elif isinstance(v, pd.Series):
            data[k] = list(v)
        else:
            data[k] = [v]

    # Repeat vectors up to the max length
    if data:
        max_len = max(len(v) for v in data.values())
        for k, v in data.items():
            if len(v) < max_len:
                # Repeat cyclically
                reps = (max_len // len(v)) + 1
                data[k] = (v * reps)[:max_len]

    df = pd.DataFrame(data)

    # Lazy imports
    from ggplot2_py.aes import aes, Mapping

    # Build a mapping from the column names
    mapping = aes(**{k: k for k in df.columns})

    # Use geom_blank
    from ggplot2_py.geom import geom_blank

    return geom_blank(mapping=mapping, data=df, inherit_aes=False)
