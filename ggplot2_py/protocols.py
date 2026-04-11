"""
Structural typing protocols for ggplot2_py GOG components.

These :class:`~typing.Protocol` definitions specify the **contracts** that
custom Geom, Stat, Scale, Coord, Facet, and Position classes should satisfy.
They are ``@runtime_checkable``, so you can use ``isinstance()`` to verify
compliance without requiring inheritance from the base classes.

This is a **Python-exclusive** feature — R's ggplot2 has no equivalent
compile-time or runtime contract checking.

Usage
-----
Mypy / pyright will flag violations statically::

    class MyStat(Stat):
        required_aes = ("x",)
        # Missing compute_group → type error

At runtime you can check::

    from ggplot2_py.protocols import StatProtocol
    assert isinstance(my_stat, StatProtocol)

Notes
-----
These protocols describe the *minimum* interface required for each
component to participate in the GOG pipeline.  They do **not** replace
the base classes (``Geom``, ``Stat``, etc.) — they complement them by
enabling structural (duck-typed) checking.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import pandas as pd

__all__ = [
    "GeomProtocol",
    "StatProtocol",
    "ScaleProtocol",
    "CoordProtocol",
    "FacetProtocol",
    "PositionProtocol",
]


# ---------------------------------------------------------------------------
# Geom
# ---------------------------------------------------------------------------

@runtime_checkable
class GeomProtocol(Protocol):
    """Contract for geometry objects.

    A conforming Geom must declare its aesthetic requirements and provide
    at least ``draw_panel`` or ``draw_group`` for rendering.
    """

    required_aes: Union[Tuple[str, ...], List[str]]
    default_aes: Any  # Mapping or dict
    draw_key: Any  # callable

    def setup_params(self, data: pd.DataFrame, params: dict) -> dict: ...
    def setup_data(self, data: pd.DataFrame, params: dict) -> pd.DataFrame: ...
    def draw_panel(self, data: pd.DataFrame, panel_params: dict,
                   coord: Any, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Stat
# ---------------------------------------------------------------------------

@runtime_checkable
class StatProtocol(Protocol):
    """Contract for statistical transformation objects.

    A conforming Stat must declare its aesthetic requirements and provide
    ``compute_group`` (or ``compute_panel`` / ``compute_layer``).
    """

    required_aes: Union[Tuple[str, ...], List[str]]
    default_aes: Any

    def setup_params(self, data: pd.DataFrame, params: dict) -> dict: ...
    def setup_data(self, data: pd.DataFrame, params: dict) -> pd.DataFrame: ...
    def compute_group(self, data: pd.DataFrame, scales: Any,
                      **params: Any) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------

@runtime_checkable
class ScaleProtocol(Protocol):
    """Contract for scale objects.

    A conforming Scale mediates between data space and aesthetic space
    via train / transform / map.
    """

    aesthetics: Any  # list of str

    def train(self, x: Any) -> None: ...
    def transform(self, x: Any) -> Any: ...
    def map(self, x: Any) -> Any: ...
    def get_breaks(self) -> Any: ...
    def get_labels(self, breaks: Any = None) -> Any: ...
    def clone(self) -> Any: ...


# ---------------------------------------------------------------------------
# Coord
# ---------------------------------------------------------------------------

@runtime_checkable
class CoordProtocol(Protocol):
    """Contract for coordinate system objects.

    A conforming Coord transforms data positions into viewport positions
    and renders background / axes.
    """

    def setup_params(self, data: list) -> dict: ...
    def transform(self, data: pd.DataFrame, panel_params: dict) -> pd.DataFrame: ...
    def setup_panel_params(self, scale_x: Any, scale_y: Any,
                           params: dict = ...) -> dict: ...


# ---------------------------------------------------------------------------
# Facet
# ---------------------------------------------------------------------------

@runtime_checkable
class FacetProtocol(Protocol):
    """Contract for faceting specification objects.

    A conforming Facet computes panel layout and assigns data to panels.
    """

    def compute_layout(self, data: list, params: dict) -> pd.DataFrame: ...
    def map_data(self, data: pd.DataFrame, layout: pd.DataFrame,
                 params: dict) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

@runtime_checkable
class PositionProtocol(Protocol):
    """Contract for position adjustment objects.

    A conforming Position adjusts data coordinates (e.g. dodge, stack)
    after stat computation but before coordinate transformation.
    """

    def setup_params(self, data: pd.DataFrame) -> dict: ...
    def compute_layer(self, data: pd.DataFrame, params: dict,
                      layout: Any) -> pd.DataFrame: ...
