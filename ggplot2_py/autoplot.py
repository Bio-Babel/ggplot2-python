"""
autoplot / autolayer — generic plot-and-layer factories for user classes.

Port of R's S3 generics ``autoplot()`` and ``autolayer()`` (autoplot.R,
autolayer.R). Downstream packages register methods for their own classes
so a single call ``autoplot(obj)`` produces a full ggplot, or
``autolayer(obj)`` produces a layer.

In R these are dispatched via ``UseMethod``; here we use
:func:`functools.singledispatch`. Users extend via
``@autoplot.register(MyClass)`` / ``@autolayer.register(MyClass)``.

The default method raises :class:`TypeError` — matching the user-visible
semantics of R's default, which calls ``cli::cli_abort(...)``.

R references
------------
* ``ggplot2/R/autoplot.R``  -- autoplot generic + default.
* ``ggplot2/R/autolayer.R`` -- autolayer generic + default.
"""

from __future__ import annotations

from functools import singledispatch
from typing import Any

__all__ = ["autoplot", "autolayer"]


@singledispatch
def autoplot(obj: Any, *args: Any, **kwargs: Any):
    """Create a complete ggplot appropriate to a particular data type.

    Mirrors R's ``autoplot()`` S3 generic. The default method raises
    :class:`TypeError`; downstream packages extend it via
    ``@autoplot.register(MyClass)``.

    Parameters
    ----------
    obj : any
        An object whose type determines the dispatched method.
    *args, **kwargs
        Forwarded to the registered method.

    Returns
    -------
    GGPlot
        A full ggplot, as produced by the registered method.
    """
    raise TypeError(
        f"No autoplot method defined for {type(obj).__name__!r}. "
        "Have you registered one via `@autoplot.register(...)`?"
    )


@singledispatch
def autolayer(obj: Any, *args: Any, **kwargs: Any):
    """Create a ggplot layer appropriate to a particular data type.

    Mirrors R's ``autolayer()`` S3 generic. The default method raises
    :class:`TypeError`; downstream packages extend it via
    ``@autolayer.register(MyClass)``.

    Parameters
    ----------
    obj : any
        An object whose type determines the dispatched method.
    *args, **kwargs
        Forwarded to the registered method.

    Returns
    -------
    Layer
        A ggplot layer, as produced by the registered method.
    """
    raise TypeError(
        f"No autolayer method defined for {type(obj).__name__!r}. "
        "Have you registered one via `@autolayer.register(...)`?"
    )
