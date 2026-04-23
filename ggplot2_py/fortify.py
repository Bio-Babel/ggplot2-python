"""
Fortify: convert model objects to DataFrames.

In R, ``fortify()`` converts model objects (lm, etc.) to data frames for
plotting.  In this Python port we focus on converting common Python data
types (dicts, arrays, None) to pandas DataFrames.

Model-fortify methods (``fortify.lm`` and friends) are registered using
:func:`functools.singledispatch`. The ``lm`` path targets
:class:`statsmodels.regression.linear_model.OLSResults`; the ``glht`` /
``cld`` paths raise :class:`NotImplementedError` with a pointer to the
equivalent Python tooling (these are R-specific ``multcomp`` classes with
no direct Python analogue).

R references
------------
* ``ggplot2/R/fortify-models.R`` -- ``fortify.lm``, ``fortify.glht``,
  ``fortify.confint.glht``, ``fortify.summary.glht``, ``fortify.cld``.
"""

from __future__ import annotations

from functools import singledispatch
from typing import Any, Optional

import numpy as np
import pandas as pd

from ggplot2_py._compat import Waiver, is_waiver, waiver, cli_abort

__all__ = ["fortify", "fortify_lm"]


def _is_lm_results(model: Any) -> bool:
    """Duck-type detection for a statsmodels OLS/GLS RegressionResults.

    We check for ``fittedvalues``, ``resid``, and ``get_influence``; all
    three are required by :func:`fortify_lm`. This avoids a hard import of
    statsmodels at the top of the module (so the package still imports if
    statsmodels is missing).
    """
    return (
        hasattr(model, "fittedvalues")
        and hasattr(model, "resid")
        and hasattr(model, "get_influence")
    )


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

    # Linear-model results (statsmodels OLSResults, duck-typed)
    if _is_lm_results(model):
        return fortify_lm(model, data=data, **kwargs)

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


# ---------------------------------------------------------------------------
# fortify.lm — statsmodels RegressionResults adapter
# ---------------------------------------------------------------------------

def fortify_lm(
    model: Any,
    data: Optional[pd.DataFrame] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Supplement linear-model data with fit statistics.

    Python adaptation of R's ``fortify.lm``. Accepts any object that
    duck-types like :class:`statsmodels.regression.linear_model.RegressionResults`
    -- specifically it must have ``fittedvalues``, ``resid`` and
    ``get_influence()``. Returns a :class:`~pandas.DataFrame` with the same
    columns R emits:

    ============ =======================================================
    Column       Meaning (R equivalent)
    ============ =======================================================
    ``.fitted``  Fitted values (``predict(model)``)
    ``.resid``   Residuals (``resid(model)``)
    ``.hat``     Diagonal of the hat matrix (``hatvalues(model)``)
    ``.sigma``   Leave-one-out residual sd (``influence(model)$sigma``)
    ``.cooksd``  Cook's distance (``cooks.distance(model)``)
    ``.stdresid`` Standardised residuals (``rstandard(model)``)
    ============ =======================================================

    Deviation from R
    ----------------
    R's ``fortify.lm`` also re-attaches the original ``model$model`` frame
    (or the *data* argument) to the output. In Python we only attach *data*
    when it is explicitly supplied; otherwise we return just the six
    ``.*`` columns -- statsmodels' ``RegressionResults`` does not expose a
    ``model$model`` frame with the exact same layout.

    Parameters
    ----------
    model : statsmodels RegressionResults (or duck-type)
        The fitted model.
    data : DataFrame, optional
        If supplied, its columns are concatenated in front of the fit
        statistics (mirrors R's ``data`` argument).
    **kwargs
        Ignored (R-signature parity).

    Returns
    -------
    pandas.DataFrame
        Original *data* (if any) plus ``.fitted``, ``.resid``, ``.hat``,
        ``.sigma``, ``.cooksd``, ``.stdresid``.
    """
    del kwargs  # accepted for signature parity with R
    infl = model.get_influence()

    fitted = np.asarray(model.fittedvalues, dtype=float)
    resid = np.asarray(model.resid, dtype=float)
    hat = np.asarray(infl.hat_matrix_diag, dtype=float)
    # R's influence(model)$sigma is the leave-one-out residual sd.
    sigma = np.sqrt(np.asarray(infl.sigma2_not_obsi, dtype=float))
    cooksd = np.asarray(infl.cooks_distance[0], dtype=float)
    stdresid = np.asarray(infl.resid_studentized_internal, dtype=float)

    stats_df = pd.DataFrame({
        ".fitted": fitted,
        ".resid": resid,
        ".hat": hat,
        ".sigma": sigma,
        ".cooksd": cooksd,
        ".stdresid": stdresid,
    })

    if data is None:
        return stats_df

    data_reset = data.reset_index(drop=True)
    return pd.concat([data_reset, stats_df], axis=1)


# ---------------------------------------------------------------------------
# fortify for multcomp (glht / confint.glht / summary.glht / cld)
# ---------------------------------------------------------------------------

def _multcomp_stub(name: str):
    """Raise a NotImplementedError referencing the R-specific source."""
    raise NotImplementedError(
        f"fortify.{name} requires R's multcomp package and has no direct "
        "Python port. Use statsmodels.stats.multicomp (e.g. `pairwise_tukeyhsd`) "
        "and convert its results manually to a DataFrame, or call the original "
        "R function via rpy2."
    )


def fortify_glht(model: Any, data: Any = None, **kwargs: Any) -> pd.DataFrame:
    """Stub for R ``fortify.glht`` -- deferred, no Python equivalent."""
    _multcomp_stub("glht")


def fortify_confint_glht(model: Any, data: Any = None, **kwargs: Any) -> pd.DataFrame:
    """Stub for R ``fortify.confint.glht`` -- deferred, no Python equivalent."""
    _multcomp_stub("confint_glht")


def fortify_summary_glht(model: Any, data: Any = None, **kwargs: Any) -> pd.DataFrame:
    """Stub for R ``fortify.summary.glht`` -- deferred, no Python equivalent."""
    _multcomp_stub("summary_glht")


def fortify_cld(model: Any, data: Any = None, **kwargs: Any) -> pd.DataFrame:
    """Stub for R ``fortify.cld`` -- deferred, no Python equivalent."""
    _multcomp_stub("cld")


# ---------------------------------------------------------------------------
# Optional: singledispatch registration on the statsmodels class.
# We register lazily so that missing statsmodels does not break import.
# ---------------------------------------------------------------------------

@singledispatch
def fortify_dispatch(model: Any, data: Any = None, **kwargs: Any):
    """Singledispatch generic mirroring R's S3 generic for ``fortify``.

    Users extend via ``@fortify_dispatch.register(MyClass)``. The default
    path defers to :func:`fortify`.
    """
    return fortify(model, data=data, **kwargs)


def _register_statsmodels_dispatch() -> bool:
    """Best-effort: register OLSResults on the singledispatch. No-op if
    statsmodels is unavailable. Returns True on success."""
    try:
        from statsmodels.regression.linear_model import RegressionResults
    except ImportError:
        return False

    @fortify_dispatch.register(RegressionResults)
    def _(model, data=None, **kwargs):
        return fortify_lm(model, data=data, **kwargs)

    return True


_register_statsmodels_dispatch()
