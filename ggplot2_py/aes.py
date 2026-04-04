"""
Aesthetic mapping system for ggplot2_py.

This module implements the core aesthetic-mapping machinery that translates
user-specified column references (and computed-variable references) into a
structured :class:`Mapping` object consumed by layers, scales, and stats.

In R's ggplot2, ``aes()`` uses non-standard evaluation (quosures).  In this
Python port we use plain strings for column names and the helper classes
:class:`AfterStat`, :class:`AfterScale`, and :class:`Stage` for deferred
references.

Examples
--------
>>> from ggplot2_py.aes import aes, after_stat, after_scale, stage
>>> aes(x="displ", y="hwy", colour="class")
>>> aes(x="displ", y=after_stat("count"))
>>> aes(colour=stage(start="class", after_scale="fill"))
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)

__all__ = [
    "aes",
    "after_stat",
    "after_scale",
    "stage",
    "vars",
    "is_mapping",
    "standardise_aes_names",
    "rename_aes",
    "Mapping",
    "AfterStat",
    "AfterScale",
    "Stage",
    "AESTHETIC_ALIASES",
]

# ---------------------------------------------------------------------------
# Aesthetic-alias lookup (R name -> canonical ggplot2 name)
# ---------------------------------------------------------------------------

AESTHETIC_ALIASES: Dict[str, str] = {
    "color": "colour",
    "pch": "shape",
    "cex": "size",
    "lwd": "linewidth",
    "lty": "linetype",
    "bg": "fill",
}
"""Mapping of common R aesthetic aliases to their canonical ggplot2 names."""

# ---------------------------------------------------------------------------
# Helper classes for deferred / staged aesthetics
# ---------------------------------------------------------------------------


class AfterStat:
    """Reference to a variable computed by the stat layer.

    Parameters
    ----------
    x : str
        Name of the computed-stat variable (e.g. ``"count"``, ``"density"``).

    Examples
    --------
    >>> AfterStat("count")
    AfterStat('count')
    """

    __slots__ = ("x",)

    def __init__(self, x: str) -> None:
        if not isinstance(x, str):
            raise TypeError(f"AfterStat expects a str, got {type(x).__name__}")
        self.x: str = x

    # -- repr / eq / hash ---------------------------------------------------

    def __repr__(self) -> str:
        return f"AfterStat({self.x!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AfterStat):
            return self.x == other.x
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("AfterStat", self.x))


class AfterScale:
    """Reference to a variable available after scale transformation.

    Parameters
    ----------
    x : str
        Name of the post-scale variable (e.g. ``"fill"``).

    Examples
    --------
    >>> AfterScale("fill")
    AfterScale('fill')
    """

    __slots__ = ("x",)

    def __init__(self, x: str) -> None:
        if not isinstance(x, str):
            raise TypeError(f"AfterScale expects a str, got {type(x).__name__}")
        self.x: str = x

    def __repr__(self) -> str:
        return f"AfterScale({self.x!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AfterScale):
            return self.x == other.x
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("AfterScale", self.x))


class Stage:
    """Staged aesthetic with potentially different values at each pipeline stage.

    At most one of *start*, *after_stat*, or *after_scale* should be the
    "primary" mapping; the others provide overrides for later stages.

    Parameters
    ----------
    start : str or None, optional
        Column name used at the initial data stage.
    after_stat : str or AfterStat or None, optional
        Reference used after the stat computation.
    after_scale : str or AfterScale or None, optional
        Reference used after scale transformation.

    Examples
    --------
    >>> Stage(start="class", after_scale="fill")
    Stage(start='class', after_stat=None, after_scale='fill')
    """

    __slots__ = ("start", "after_stat", "after_scale")

    def __init__(
        self,
        start: Optional[str] = None,
        after_stat: Optional[Union[str, AfterStat]] = None,
        after_scale: Optional[Union[str, AfterScale]] = None,
    ) -> None:
        self.start = start

        # Normalise bare strings to wrapper objects for convenience.
        if isinstance(after_stat, str):
            after_stat = AfterStat(after_stat)
        self.after_stat: Optional[AfterStat] = after_stat  # type: ignore[assignment]

        if isinstance(after_scale, str):
            after_scale = AfterScale(after_scale)
        self.after_scale: Optional[AfterScale] = after_scale  # type: ignore[assignment]

    def __repr__(self) -> str:
        return (
            f"Stage(start={self.start!r}, after_stat={self.after_stat!r}, "
            f"after_scale={self.after_scale!r})"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Stage):
            return (
                self.start == other.start
                and self.after_stat == other.after_stat
                and self.after_scale == other.after_scale
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("Stage", self.start, self.after_stat, self.after_scale))


# ---------------------------------------------------------------------------
# Mapping class (dict subclass)
# ---------------------------------------------------------------------------

#: Type alias for a single aesthetic value.
AesValue = Union[str, AfterStat, AfterScale, Stage, Callable[..., Any], int, float]


class Mapping(dict):
    """Aesthetic mapping object, analogous to the output of R's ``aes()``.

    A thin :class:`dict` subclass whose keys are canonical aesthetic names
    (e.g. ``"x"``, ``"colour"``) and whose values describe how to map data
    to that aesthetic.

    Values may be:

    * **str** — column-name reference (most common).
    * **AfterStat** — computed variable from a stat layer.
    * **AfterScale** — variable available after scale transformation.
    * **Stage** — staged aesthetic with per-stage overrides.
    * **callable** — a lambda or function applied to the data.
    * **scalar** (int, float, str) — constant aesthetic value.

    Examples
    --------
    >>> m = Mapping(x="displ", y="hwy", colour="class")
    >>> m["x"]
    'displ'
    """

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self.items())
        return f"aes({inner})"

    # Convenience: attribute access mirrors key access (read-only).
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            ) from None


# ---------------------------------------------------------------------------
# Public convenience constructors
# ---------------------------------------------------------------------------


def after_stat(x: str) -> AfterStat:
    """Create an :class:`AfterStat` reference.

    Parameters
    ----------
    x : str
        Name of a stat-computed variable (e.g. ``"count"``).

    Returns
    -------
    AfterStat
        A deferred reference resolved after stat computation.

    Examples
    --------
    >>> after_stat("density")
    AfterStat('density')
    """
    return AfterStat(x)


def after_scale(x: str) -> AfterScale:
    """Create an :class:`AfterScale` reference.

    Parameters
    ----------
    x : str
        Name of a post-scale variable (e.g. ``"fill"``).

    Returns
    -------
    AfterScale
        A deferred reference resolved after scale transformation.

    Examples
    --------
    >>> after_scale("fill")
    AfterScale('fill')
    """
    return AfterScale(x)


def stage(
    start: Optional[str] = None,
    after_stat: Optional[Union[str, AfterStat]] = None,
    after_scale: Optional[Union[str, AfterScale]] = None,
) -> Stage:
    """Create a :class:`Stage` aesthetic with per-stage overrides.

    Parameters
    ----------
    start : str or None, optional
        Column name used at the initial data stage.
    after_stat : str, AfterStat, or None, optional
        Reference used after the stat computation.
    after_scale : str, AfterScale, or None, optional
        Reference used after scale transformation.

    Returns
    -------
    Stage
        A staged aesthetic mapping.

    Examples
    --------
    >>> stage(start="class", after_scale="fill")
    Stage(start='class', after_stat=None, after_scale='fill')
    """
    return Stage(start=start, after_stat=after_stat, after_scale=after_scale)


# ---------------------------------------------------------------------------
# aes() — main entry point
# ---------------------------------------------------------------------------


def aes(
    x: Optional[AesValue] = None,
    y: Optional[AesValue] = None,
    **kwargs: AesValue,
) -> Mapping:
    """Create an aesthetic mapping.

    This is the Python equivalent of R's ``aes()`` function.  Column
    references are passed as plain strings; deferred references use
    :func:`after_stat`, :func:`after_scale`, or :func:`stage`.

    Parameters
    ----------
    x : str, callable, or scalar, optional
        Mapping for the *x* aesthetic.
    y : str, callable, or scalar, optional
        Mapping for the *y* aesthetic.
    **kwargs
        Additional aesthetic mappings.  Names are automatically
        standardised (e.g. ``color`` becomes ``colour``).

    Returns
    -------
    Mapping
        An aesthetic-mapping dictionary.

    Examples
    --------
    >>> aes(x="displ", y="hwy")
    aes(x='displ', y='hwy')

    >>> aes(x="displ", y="hwy", color="class")
    aes(x='displ', y='hwy', colour='class')

    >>> aes(x="displ", y=after_stat("count"))
    aes(x='displ', y=AfterStat('count'))
    """
    raw: Dict[str, AesValue] = {}
    if x is not None:
        raw["x"] = x
    if y is not None:
        raw["y"] = y
    raw.update(kwargs)

    # Standardise names (e.g. "color" -> "colour").
    result = Mapping()
    for key, value in raw.items():
        canonical = _standardise_single(key)
        result[canonical] = value

    return result


# ---------------------------------------------------------------------------
# Name standardisation
# ---------------------------------------------------------------------------


def _standardise_single(name: str) -> str:
    """Canonicalise a single aesthetic name.

    Parameters
    ----------
    name : str
        Raw aesthetic name.

    Returns
    -------
    str
        Canonical aesthetic name.
    """
    return AESTHETIC_ALIASES.get(name, name)


def standardise_aes_names(aes_names: Iterable[str]) -> List[str]:
    """Standardise a sequence of aesthetic names to their canonical forms.

    Parameters
    ----------
    aes_names : Iterable[str]
        Aesthetic names, possibly containing aliases.

    Returns
    -------
    list of str
        Canonical aesthetic names in the same order.

    Examples
    --------
    >>> standardise_aes_names(["color", "lwd", "x"])
    ['colour', 'linewidth', 'x']
    """
    return [_standardise_single(n) for n in aes_names]


def rename_aes(mapping: Mapping) -> Mapping:
    """Return a copy of *mapping* with all keys standardised.

    Parameters
    ----------
    mapping : Mapping
        An aesthetic mapping (or plain dict).

    Returns
    -------
    Mapping
        New mapping with canonical aesthetic names as keys.

    Examples
    --------
    >>> rename_aes(Mapping(color="class"))
    aes(colour='class')
    """
    return Mapping(
        {_standardise_single(k): v for k, v in mapping.items()}
    )


# ---------------------------------------------------------------------------
# vars() helper (for facets)
# ---------------------------------------------------------------------------


def vars(*args: str, **kwargs: str) -> List[str]:
    """Specify variables for faceting.

    A thin helper that collects positional and keyword variable names into a
    flat list of strings, suitable for passing to ``facet_wrap`` or
    ``facet_grid``.

    Parameters
    ----------
    *args : str
        Variable names given positionally.
    **kwargs : str
        Variable names given as keyword arguments (values are used, keys
        are ignored).

    Returns
    -------
    list of str
        Flat list of variable-name strings.

    Examples
    --------
    >>> vars("cyl", "drv")
    ['cyl', 'drv']

    >>> vars(rows="cyl")
    ['cyl']
    """
    result: List[str] = list(args)
    result.extend(kwargs.values())
    return result


# ---------------------------------------------------------------------------
# Type guard
# ---------------------------------------------------------------------------


def is_mapping(x: Any) -> bool:
    """Test whether *x* is an aesthetic :class:`Mapping`.

    Parameters
    ----------
    x : object
        Object to test.

    Returns
    -------
    bool
        ``True`` if *x* is a :class:`Mapping` instance.

    Examples
    --------
    >>> is_mapping(aes(x="a"))
    True
    >>> is_mapping({"x": "a"})
    False
    """
    return isinstance(x, Mapping)
