"""
ggproto object system for ggplot2.

In R, ggproto is a prototype-based OOP system built on top of
environments.  This Python port uses standard classes with a thin
compatibility layer so that:

* ``ggproto("ClassName", ParentClass, method=..., attr=...)`` dynamically
  creates a new class (like ``type()``).
* Instances can override class-level members (prototype semantics).
* ``ggproto_parent(Parent, self)`` provides explicit parent-method
  dispatch (similar to ``ggproto_parent()`` in R).
"""

from __future__ import annotations

import types
from typing import Any, Callable, Dict, Optional, Type, Union

__all__ = [
    "GGProto",
    "ggproto",
    "ggproto_parent",
    "is_ggproto",
    "fetch_ggproto",
]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class GGProtoMeta(type):
    """Metaclass for GGProto that makes class-level attribute access
    work like instance-level access for method binding.

    This supports the R pattern where e.g. ``GeomPoint`` is used both
    as a class *and* as a singleton object.
    """

    def __repr__(cls) -> str:
        return f"<ggproto class: {cls.__name__}>"


class GGProto(metaclass=GGProtoMeta):
    """Base class for ggplot2's proto-based objects.

    ``GGProto`` is the foundation of ggplot2's OOP.  Subclasses
    represent geoms, stats, scales, coords, facets, etc.  Both classes
    and instances are used interchangeably in R; this Python port
    preserves that duality.

    Attributes
    ----------
    _class_name : str or None
        Optional explicit name set by the ``ggproto()`` factory.

    Examples
    --------
    Defining a geom-like object:

    >>> MyGeom = ggproto(
    ...     "MyGeom", GGProto,
    ...     required_aes={"x", "y"},
    ...     draw_panel=lambda self, data, params: data,
    ... )
    """

    _class_name: Optional[str] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    def __repr__(self) -> str:
        cls_name = self._class_name or type(self).__name__
        return f"<ggproto object: {cls_name}>"

    # Allow instance-level member override (prototype semantics).
    def _set(self, **members: Any) -> None:
        """Override members on this instance.

        Parameters
        ----------
        **members : Any
            Name/value pairs.  Callables are stored as-is (they will be
            bound when accessed via ``__getattribute__``).
        """
        for name, value in members.items():
            object.__setattr__(self, name, value)

    def __getattribute__(self, name: str) -> Any:
        """Attribute access with auto-binding of plain functions.

        When a plain function (not already a bound method) is retrieved
        from the instance ``__dict__`` or from the class, it is bound so
        that ``self`` is passed automatically — matching R ggproto
        behaviour.
        """
        value = super().__getattribute__(name)
        # Only auto-bind plain functions, not built-ins, lambdas stored
        # intentionally without self, classmethods, staticmethods, etc.
        if isinstance(value, types.FunctionType) and not name.startswith("_"):
            # Check how many arguments the function expects.
            code = value.__code__
            nargs = code.co_argcount
            # Only bind if the first parameter is named 'self'.
            varnames = code.co_varnames[:nargs]
            if nargs > 0 and varnames[0] == "self":
                return types.MethodType(value, self)
        return value


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def ggproto(
    _class: str,
    _inherit: Optional[Type[GGProto]] = None,
    **members: Any,
) -> Type[GGProto]:
    """Create a new ``GGProto`` subclass dynamically.

    This mirrors the R ``ggproto()`` call.  The returned object is a
    **class** (not an instance), but thanks to the metaclass it can be
    used in both class-like and instance-like ways.

    Parameters
    ----------
    _class : str
        Name for the new class.
    _inherit : type, optional
        Parent class.  Defaults to ``GGProto``.
    **members : Any
        Attributes and methods to set on the class.  Functions whose
        first parameter is named ``self`` will behave as instance
        methods.

    Returns
    -------
    type
        A new ``GGProto`` subclass.

    Examples
    --------
    >>> Geom = ggproto("Geom", GGProto, draw_panel=lambda self, data: data)
    >>> Point = ggproto("Point", Geom, shape="circle")
    """
    if _inherit is None:
        _inherit = GGProto

    # Build the class namespace.
    namespace: Dict[str, Any] = {"_class_name": _class}
    namespace.update(members)

    # Create the class via the metaclass.
    new_cls = GGProtoMeta(_class, (_inherit,), namespace)
    return new_cls


# ---------------------------------------------------------------------------
# Parent-method dispatch
# ---------------------------------------------------------------------------

class _GGProtoParentProxy:
    """Proxy object for calling parent-class methods with a bound *self*.

    Returned by ``ggproto_parent()``.  Attribute access on this proxy
    retrieves the method from the parent class and binds it to the
    original *self* instance (or class).
    """

    __slots__ = ("_parent", "_self")

    def __init__(self, parent: Type[GGProto], self_obj: Any) -> None:
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_self", self_obj)

    def __repr__(self) -> str:
        parent = object.__getattribute__(self, "_parent")
        return f"<ggproto parent proxy: {parent.__name__}>"

    def __getattr__(self, name: str) -> Any:
        parent = object.__getattribute__(self, "_parent")
        self_obj = object.__getattribute__(self, "_self")

        # Walk the parent's MRO to find the attribute.
        for cls in parent.__mro__:
            if name in cls.__dict__:
                value = cls.__dict__[name]
                # Bind functions to self_obj.
                if isinstance(value, types.FunctionType):
                    return types.MethodType(value, self_obj)
                return value
        raise AttributeError(
            f"'{parent.__name__}' ggproto object has no member '{name}'"
        )


def ggproto_parent(
    parent: Type[GGProto],
    self: Any,
) -> _GGProtoParentProxy:
    """Get a proxy for calling parent-class methods.

    This is the Python equivalent of R's ``ggproto_parent(Parent, self)``.
    Methods accessed through the proxy will be bound to *self*.

    Parameters
    ----------
    parent : type
        The parent ``GGProto`` class whose methods should be called.
    self : GGProto
        The current object (``self`` in the calling method).

    Returns
    -------
    _GGProtoParentProxy
        A proxy that dispatches attribute access to *parent*'s methods.

    Examples
    --------
    Inside a ggproto method:

    >>> def draw_panel(self, data, params):
    ...     data = ggproto_parent(Geom, self).draw_panel(data, params)
    ...     return data
    """
    return _GGProtoParentProxy(parent, self)


# ---------------------------------------------------------------------------
# Predicates / accessors
# ---------------------------------------------------------------------------

def is_ggproto(x: Any) -> bool:
    """Check whether *x* is a ``GGProto`` class or instance.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        ``True`` if *x* is a ``GGProto`` instance **or** a subclass of
        ``GGProto``.
    """
    if isinstance(x, GGProto):
        return True
    if isinstance(x, type) and issubclass(x, GGProto):
        return True
    return False


def fetch_ggproto(x: Any, name: str) -> Any:
    """Retrieve a member from a ``GGProto`` object.

    Parameters
    ----------
    x : GGProto
        A ``GGProto`` class or instance.
    name : str
        Member name.

    Returns
    -------
    Any
        The member value.

    Raises
    ------
    AttributeError
        If the member does not exist.
    """
    if not is_ggproto(x):
        raise TypeError(f"Expected a GGProto object, got {type(x).__name__}")
    return getattr(x, name)
