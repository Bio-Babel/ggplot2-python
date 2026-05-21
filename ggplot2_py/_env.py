"""
PlotEnv — layered namespace lookup for scale/guide constructors.

Port of R's ``find_global(name, env, mode = "function")`` (R ref:
``ggplot2/R/scale-type.R:39-54``).  Quoting the R source:

    Look for object first in parent environment and if not found, then in
    ggplot2 namespace environment.  This makes it possible to override
    default scales by setting them in the parent environment.

In R, ``plot@plot_env`` is the environment where the plot was constructed
(typically ``parent.frame()`` at ``ggplot()`` call time), and a chain of
environments terminates in ``asNamespace("ggplot2")``.  ggnewscale
exploits this by **injecting** ``scale_<bumped_aes>_<type>`` functions
into ``plot@plot_env`` (R ref:
``ggnewscale/R/rename-aes.R:87-104``), so that when build-time
default-scale resolution runs, the injected constructor wins.

In Python, the closest equivalent is a list-of-namespaces walked in
order, with the ``ggplot2_py.scales`` module as the implicit fallback
layer.  :class:`PlotEnv` encapsulates that — call sites pass a
``PlotEnv`` to :func:`find_scale` / ``ScalesList.add_defaults`` /
``ScalesList.add_missing``, and lookups walk the user-pushed layers
first.

A *namespace* is anything that responds to attribute access **or**
mapping ``__getitem__``.  ``dict``, ``types.ModuleType``,
``types.SimpleNamespace``, ``argparse.Namespace``, and arbitrary
objects implementing ``__getattr__`` all work.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

__all__ = ["PlotEnv"]


class PlotEnv:
    """Layered scale/guide-constructor lookup table for a plot.

    Mirrors the *effective* behaviour of R's ``plot@plot_env`` + the
    ``find_global`` walk: lookups proceed through the user-pushed
    layers in **last-pushed-first-checked** order (LIFO), then fall
    back to the ``ggplot2_py.scales`` module.

    Each layer may be any object that supports either attribute access
    (``getattr(layer, name)``) or mapping access (``layer[name]``); the
    first that yields a non-``None`` value wins.

    Examples
    --------
    >>> env = PlotEnv()
    >>> env.push({"scale_colour_continuous": my_factory})
    >>> env.lookup("scale_colour_continuous") is my_factory
    True

    The implicit fallback to ``ggplot2_py.scales`` is intentionally
    **not** included in :attr:`layers` so that :meth:`clone` produces
    an env with the same user-visible chain.
    """

    def __init__(self, *layers: Any) -> None:
        # Layers are stored in **push order**.  Lookup walks them in
        # reverse so that the most recently pushed layer wins, matching
        # R's environment-chain semantics (the immediate parent is
        # checked before more ancestral ones).
        self._layers: List[Any] = [l for l in layers if l is not None]

    # -- Mutation ----------------------------------------------------------

    def push(self, layer: Any) -> None:
        """Add a new lookup layer.  Last-pushed wins on conflicts.

        Parameters
        ----------
        layer : object
            Any dict-like or attribute-bearing namespace.
        """
        if layer is None:
            return
        self._layers.append(layer)

    def pop(self) -> Optional[Any]:
        """Remove and return the most-recently-pushed layer, or ``None``
        if the chain is empty.
        """
        if not self._layers:
            return None
        return self._layers.pop()

    # -- Query -------------------------------------------------------------

    def lookup(self, name: str) -> Optional[Callable]:
        """Resolve *name* through the layer chain, then the
        ``ggplot2_py.scales`` fallback module.

        Returns the first hit, or ``None`` if nothing matches.

        Notes
        -----
        The fallback layer is searched **last** even when the lookup
        chain is non-empty — matching R's
        ``c(env, list(as_namespace("ggplot2")))`` (R ref:
        ``scale-type.R:44``).
        """
        for layer in reversed(self._layers):
            val = _ns_get(layer, name)
            if val is not None:
                return val
        # Fallback: the package's own scale-constructor module.  Import
        # lazily to avoid an import cycle with scale.py at load time —
        # the cycle only matters during module construction; at call
        # time every module is loaded so the import never raises.
        from ggplot2_py import scales as _scales_mod
        return getattr(_scales_mod, name, None)

    def __contains__(self, name: str) -> bool:
        return self.lookup(name) is not None

    # -- Plumbing ----------------------------------------------------------

    def clone(self) -> "PlotEnv":
        """Return a new :class:`PlotEnv` with the same layer chain.

        The layers themselves are **not** deep-copied — mirrors R env
        semantics where ``plot_clone`` does not snapshot the
        environment contents.
        """
        new = PlotEnv()
        new._layers = list(self._layers)
        return new

    @property
    def layers(self) -> List[Any]:
        """Read-only view of the layer chain (in push order)."""
        return list(self._layers)

    def __repr__(self) -> str:
        return f"<PlotEnv layers={len(self._layers)}>"


def _ns_get(layer: Any, name: str) -> Any:
    """Resolve *name* on *layer*, trying mapping then attribute access.

    Returns the found value, or ``None`` if absent.  Catches only the
    natural "key absent" / "unsupported indexing" cases — anything else
    propagates so real bugs surface (per the project rule against
    over-broad fallback logic).
    """
    # Mapping path (dict and dict-like).  ``str``/``bytes`` are
    # excluded because their ``__getitem__`` is positional and would
    # raise ``TypeError`` for non-integer keys.
    if hasattr(layer, "__getitem__") and not isinstance(layer, (str, bytes)):
        try:
            return layer[name]
        except (KeyError, TypeError):
            pass
    # Attribute path (modules, SimpleNamespace, regular objects).
    return getattr(layer, name, None)
