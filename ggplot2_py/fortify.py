"""
Fortify: convert model objects to DataFrames.

In R, ``fortify()`` converts model objects (lm, etc.) to data frames for
plotting.  In this Python port we focus on converting common Python data
types (dicts, arrays, None) to pandas DataFrames.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ggplot2_py._compat import Waiver, is_waiver, waiver, cli_abort

__all__ = ["fortify"]


def fortify(
    model: Any,
    data: Optional[pd.DataFrame] = None,
    **kwargs: Any,
) -> Any:
    """Convert *model* to a :class:`~pandas.DataFrame` suitable for plotting.

    Parameters
    ----------
    model : object
        The object to convert.  Supported types:

        * ``None`` -- returns a :class:`~ggplot2_py._compat.Waiver` sentinel.
        * ``pandas.DataFrame`` -- returned as-is.
        * ``dict`` -- converted via ``pd.DataFrame(model)``.
        * ``numpy.ndarray`` -- converted via ``pd.DataFrame(model)``.
        * ``callable`` -- returned as-is (for functional data specification).
        * Any object with ``__dataframe__`` or ``to_pandas()`` methods.
    data : DataFrame, optional
        Original dataset (unused in the generic implementation but kept
        for signature compatibility).
    **kwargs
        Additional keyword arguments (unused in the generic implementation).

    Returns
    -------
    pandas.DataFrame or Waiver or callable
        A DataFrame representation of *model*, a ``Waiver`` (for ``None``),
        or the original callable.

    Raises
    ------
    ValueError
        If *model* cannot be coerced into a DataFrame.
    """
    # None -> waiver (like R's fortify.NULL)
    if model is None:
        return waiver()

    # Already a DataFrame
    if isinstance(model, pd.DataFrame):
        return model

    # Waiver pass-through
    if is_waiver(model):
        return model

    # Callable (function / lambda) -- returned as-is (like R's fortify.function)
    if callable(model) and not isinstance(model, (type, np.ndarray)):
        return model

    # dict -> DataFrame
    if isinstance(model, dict):
        return pd.DataFrame(model)

    # numpy array -> DataFrame
    if isinstance(model, np.ndarray):
        return pd.DataFrame(model)

    # Objects that support the pandas interchange protocol
    if hasattr(model, "to_pandas"):
        return model.to_pandas()

    if hasattr(model, "__dataframe__"):
        return pd.api.interchange.from_dataframe(model)

    # Generic fallback -- try pd.DataFrame()
    try:
        return pd.DataFrame(model)
    except Exception as exc:
        cli_abort(
            "`data` must be a DataFrame, or an object coercible by fortify(), "
            f"not {type(model).__name__}.",
            cls=TypeError,
        )
