"""
Built-in datasets for ggplot2_py.

Provides lazy-loaded access to the classic ggplot2 example datasets.
CSV files are stored in the ``resources/`` subdirectory and loaded on
first access via module-level ``__getattr__``.

Examples
--------
>>> from ggplot2_py.datasets import mpg
>>> mpg.head()

>>> from ggplot2_py.datasets import load_dataset
>>> df = load_dataset("diamonds")
"""

from __future__ import annotations

import importlib.resources as _resources
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

__all__ = [
    "load_dataset",
    "diamonds",
    "economics",
    "economics_long",
    "faithfuld",
    "luv_colours",
    "midwest",
    "mpg",
    "msleep",
    "presidential",
    "seals",
    "txhousing",
]

# ---------------------------------------------------------------------------
# Dataset names recognised by this module
# ---------------------------------------------------------------------------

_DATASET_NAMES = frozenset(
    {
        "diamonds",
        "economics",
        "economics_long",
        "faithfuld",
        "luv_colours",
        "midwest",
        "mpg",
        "msleep",
        "presidential",
        "seals",
        "txhousing",
    }
)

# Cache: dataset name -> DataFrame (populated on first access)
_cache: Dict[str, pd.DataFrame] = {}

# ---------------------------------------------------------------------------
# Resource path helper
# ---------------------------------------------------------------------------

def _resources_dir() -> Path:
    """Return the path to the ``resources/`` directory shipped with the package.

    Returns
    -------
    Path
        Absolute path to the resources directory.
    """
    return Path(__file__).resolve().parent / "resources"


# ---------------------------------------------------------------------------
# Post-load fixups
# ---------------------------------------------------------------------------

def _fixup_diamonds(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ordered-categorical dtypes to the *diamonds* dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw diamonds data.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``cut``, ``color``, and ``clarity`` as ordered
        categoricals.
    """
    cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    color_order = ["D", "E", "F", "G", "H", "I", "J"]
    clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

    df["cut"] = pd.Categorical(df["cut"], categories=cut_order, ordered=True)
    df["color"] = pd.Categorical(df["color"], categories=color_order, ordered=True)
    df["clarity"] = pd.Categorical(
        df["clarity"], categories=clarity_order, ordered=True
    )
    return df


def _fixup_economics(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the ``date`` column in *economics* as ``datetime64``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw economics data.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``date`` parsed as datetime.
    """
    df["date"] = pd.to_datetime(df["date"])
    return df


def _fixup_economics_long(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the ``date`` column in *economics_long* as ``datetime64``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw economics_long data.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``date`` parsed as datetime.
    """
    df["date"] = pd.to_datetime(df["date"])
    return df


def _fixup_presidential(df: pd.DataFrame) -> pd.DataFrame:
    """Parse ``start`` and ``end`` columns in *presidential* as ``datetime64``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw presidential data.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``start`` and ``end`` parsed as datetime.
    """
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def _fixup_txhousing(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``year`` and ``month`` are integer columns in *txhousing*.

    Parameters
    ----------
    df : pd.DataFrame
        Raw txhousing data.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``year`` and ``month`` as integers.
    """
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    return df


# Registry of fixup functions keyed by dataset name.
_FIXUPS = {
    "diamonds": _fixup_diamonds,
    "economics": _fixup_economics,
    "economics_long": _fixup_economics_long,
    "presidential": _fixup_presidential,
    "txhousing": _fixup_txhousing,
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dataset(name: str) -> pd.DataFrame:
    """Load a built-in ggplot2 dataset by name.

    The dataset is read from CSV on the first call and cached for subsequent
    requests.

    Parameters
    ----------
    name : str
        Name of the dataset.  Must be one of: ``diamonds``, ``economics``,
        ``economics_long``, ``faithfuld``, ``luv_colours``, ``midwest``,
        ``mpg``, ``msleep``, ``presidential``, ``seals``, ``txhousing``.

    Returns
    -------
    pd.DataFrame
        The requested dataset as a pandas DataFrame.

    Raises
    ------
    ValueError
        If *name* is not a recognised dataset name.

    Examples
    --------
    >>> df = load_dataset("mpg")
    >>> df.shape
    (234, 11)
    """
    if name not in _DATASET_NAMES:
        raise ValueError(
            f"Unknown dataset {name!r}. Available datasets: "
            f"{sorted(_DATASET_NAMES)}"
        )

    if name not in _cache:
        csv_path = _resources_dir() / f"{name}.csv"
        df = pd.read_csv(csv_path)

        fixup = _FIXUPS.get(name)
        if fixup is not None:
            df = fixup(df)

        _cache[name] = df

    return _cache[name].copy()


# ---------------------------------------------------------------------------
# Module-level lazy loading via __getattr__
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> pd.DataFrame:
    """Lazily load a dataset when accessed as a module attribute.

    Parameters
    ----------
    name : str
        Attribute name (must match a dataset name).

    Returns
    -------
    pd.DataFrame
        The requested dataset.

    Raises
    ------
    AttributeError
        If *name* is not a recognised dataset name.
    """
    if name in _DATASET_NAMES:
        return load_dataset(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
