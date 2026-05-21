"""
ggproto object system for ggplot2.

In R, ggproto is a prototype-based OOP system built on top of
environments (R ref: ``ggplot2/R/ggproto.R``).  This Python port emulates
the same semantics on top of standard classes:

* ``ggproto("ClassName", ParentClass, method=..., attr=...)`` dynamically
  creates a new ``GGProto`` subclass (the class-as-parent path).
* ``ggproto("Name", parent_instance, method=...)`` creates a new instance
  whose super-chain is the parent **instance** (the instance-as-parent
  path) — this is the idiom used by extension packages such as
  ggnewscale to clone-with-overrides.  See R ``ggproto.R:67-97`` where
  ``_inherit`` can be either a class-singleton or an instance.
* Instances can override class-level members via :meth:`GGProto._set`
  (prototype semantics).
* :func:`ggproto_parent` provides explicit parent-method dispatch,
  matching R's ``ggproto_parent(Parent, self)``.  When *parent* is an
  instance, the proxy walks the instance's own ``__dict__`` first, then
  the class MRO — matching R's ``fetch_ggproto`` recursion.
* :func:`bind_method` explicitly installs a callable as a bound method
  regardless of its first-argument name, complementing the automatic
  ``self``-named binding done by :meth:`GGProto.__getattribute__`.
"""

from __future__ import annotations

import copy as _copy
import types
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

__all__ = [
    "GGProto",
    "ggproto",
    "ggproto_parent",
    "is_ggproto",
    "fetch_ggproto",
    "bind_method",
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

    def __dir__(cls) -> list:
        """Enable tab-completion on classes used as singletons.

        Mirrors the instance ``__dir__`` (R ``.DollarNames.ggproto``):
        returns the union of the class's own names and its bases',
        excluding ``super``.
        """
        base = set(super().__dir__())
        for c in cls.__mro__:
            base.update(c.__dict__.keys())
        base.discard("super")
        return sorted(base)


def _bind_if_self_method(value: Any, self_obj: Any, name: str) -> Any:
    """Bind *value* as a method on *self_obj* when it is a plain function
    whose first parameter is named ``self``.

    R analogue: ``make_proto_method`` (R ref: ``ggproto.R:182-198``),
    which inspects the function's formals and inserts ``self`` only when
    a ``self`` argument is declared.  Names starting with ``_`` are
    skipped to keep dunder / private fields raw.
    """
    if isinstance(value, types.FunctionType) and not name.startswith("_"):
        code = value.__code__
        if code.co_argcount > 0 and code.co_varnames[0] == "self":
            return types.MethodType(value, self_obj)
    return value


def _resolve_shadowed(obj: "GGProto", name: str) -> Any:
    """Return the *unbound* value that ``obj.name`` would currently
    resolve to, walking own ``__dict__`` → ``_super_inst`` chain →
    class MRO.

    Used by :meth:`GGProto._set` to detect the mis-bind footgun.
    Returns ``None`` if no such value exists.  Symmetric with the
    lookup order in :meth:`GGProto.__getattribute__` — checking
    ``obj.__dict__`` first is what lets the footgun warning catch
    sequences like ``inst._set(m=self_fn)`` immediately followed by
    ``inst._set(m=non_self_fn)``, where the shadowed value lives on
    the instance itself rather than on a parent or class.
    """
    own = object.__getattribute__(obj, "__dict__")
    if name in own:
        return own[name]
    sup = own.get("_super_inst")
    seen = {id(obj)}
    while sup is not None and id(sup) not in seen:
        seen.add(id(sup))
        sup_dict = object.__getattribute__(sup, "__dict__")
        if name in sup_dict:
            return sup_dict[name]
        sup = sup_dict.get("_super_inst")
    return getattr(type(obj), name, None)


def _is_self_bearing(existing: Any) -> bool:
    """Return ``True`` when *existing* would be auto-bound (or has
    been explicitly bound) to receive ``self`` as its first argument.

    Recognises:

    * a plain :class:`types.FunctionType` whose first formal parameter
      is named ``self`` — the value that :func:`_bind_if_self_method`
      would wrap on access;
    * a :class:`types.MethodType` previously installed via
      :func:`bind_method` — its ``__func__`` is already bound to a
      receiver, so replacing it with a plain function loses the
      binding silently.

    Used by :meth:`GGProto._set` to decide whether the footgun warning
    should fire.
    """
    if isinstance(existing, types.FunctionType):
        code = existing.__code__
        varnames = code.co_varnames[: code.co_argcount]
        return bool(varnames) and varnames[0] == "self"
    if isinstance(existing, types.MethodType):
        # Explicit ``bind_method`` install — by definition self-bearing.
        return True
    return False


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
    _super_inst : GGProto or None
        When this object was created via the instance-as-parent path
        (``ggproto("Name", parent_instance, ...)``), this slot holds the
        parent **instance** so that :func:`ggproto_parent` and
        :meth:`super` can walk it.  ``None`` for class-as-parent
        constructions (the super chain is then expressed in the class
        MRO).  Mirrors R's ``e$super`` (R ref: ``ggproto.R:90``).

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
    _super_inst: Optional["GGProto"] = None

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
            Name/value pairs.  Callables stored as plain ``FunctionType``
            objects are bound automatically on access when their first
            parameter is named ``self`` (see :meth:`__getattribute__`).
            If you want a function whose first parameter is **not**
            named ``self`` to behave as a method, use
            :func:`bind_method` instead.

        Notes
        -----
        Emits a :class:`UserWarning` when the assignment looks like a
        silent-mis-bind: replacing an existing bound method with a
        plain function whose first arg is not ``self`` would otherwise
        produce a member that calls without ``self`` injected.  The
        shadowed value is resolved through the same path as
        :meth:`__getattribute__` (own ``__dict__`` → ``_super_inst``
        chain → class MRO), so the warning catches mis-binds against
        either class-level defaults or methods defined on a
        parent instance.
        """
        for name, value in members.items():
            if (
                isinstance(value, types.FunctionType)
                and not name.startswith("_")
            ):
                code = value.__code__
                varnames = code.co_varnames[: code.co_argcount]
                first = varnames[0] if varnames else None
                if first != "self" and _is_self_bearing(
                    _resolve_shadowed(self, name)
                ):
                    warnings.warn(
                        (
                            f"_set({name}=...): replacement function's "
                            f"first argument is {first!r}, not 'self'. "
                            f"It will NOT be auto-bound. Use "
                            f"bind_method() for explicit binding."
                        ),
                        UserWarning,
                        stacklevel=2,
                    )
            object.__setattr__(self, name, value)

    def super(self) -> Optional["GGProto"]:
        """Return the parent instance/class for ``ggproto_parent``-style
        traversal.

        Mirrors R's ``self$super()`` (R ref: ``ggproto.R:83-90``).  For
        instances built via :func:`ggproto` with an instance-as-parent,
        returns the captured parent instance.  Otherwise, returns the
        first non-``GGProto`` ancestor class in the MRO that is itself
        a ``GGProto`` subclass — emulating R's "class chain super".

        Returns
        -------
        GGProto or None
        """
        sup = object.__getattribute__(self, "_super_inst")
        if sup is not None:
            return sup
        # Class-chain fall-back: first ``GGProto`` ancestor (skipping self's
        # class) — corresponds to ``class(e) <- c(_class, class(super))``
        # (R: ggproto.R:91) when ``super`` was a class.
        for cls in type(self).__mro__[1:]:
            if cls is GGProto:
                continue
            if issubclass(cls, GGProto):
                return cls  # type: ignore[return-value]
        return None

    def __getattribute__(self, name: str) -> Any:
        """Attribute access mirroring R's ``fetch_ggproto`` recursion.

        Lookup order (R ref: ``ggproto.R:118-142`` — ``fetch_ggproto``
        recursively walks ``x$super()``):

        1. This instance's own ``__dict__``.
        2. The ``_super_inst`` chain — each ancestor's ``__dict__``
           in turn.  This is the *instance* chain, which in R env
           semantics shadows class-level defaults.
        3. The class MRO (Python descriptor protocol applies).

        Plain functions whose first parameter is named ``self`` are
        bound to **this** receiver (not the ancestor where they were
        defined), matching R's ``make_proto_method`` (R ref:
        ``ggproto.R:182-198``) which always binds to the originating
        receiver.

        The instance chain takes precedence over class MRO so that
        the R idiom ``parent$add(10)`` (mutating ``parent$x``)
        followed by ``child <- ggproto("Child", parent)`` exposes the
        mutated value via ``child$x`` — not the class-level default.
        """
        # Fast path: bypass for dunders / private to avoid recursion.
        if name.startswith("__"):
            return super().__getattribute__(name)

        # 1. Own instance __dict__ (raw — same as R env-local lookup).
        inst_dict = object.__getattribute__(self, "__dict__")
        if name in inst_dict:
            return _bind_if_self_method(inst_dict[name], self, name)

        # 2. Walk the ``_super_inst`` chain (R env-chain semantics).
        #    ``seen`` guards against pathological cycles.
        sup = inst_dict.get("_super_inst")
        seen = {id(self)}
        while sup is not None and id(sup) not in seen:
            seen.add(id(sup))
            sup_dict = object.__getattribute__(sup, "__dict__")
            if name in sup_dict:
                return _bind_if_self_method(sup_dict[name], self, name)
            sup = sup_dict.get("_super_inst")

        # 3. Class MRO fall-back (Python descriptor protocol applies).
        value = super().__getattribute__(name)
        return _bind_if_self_method(value, self, name)

    def __dir__(self) -> list:
        """Return the list of accessible member names.

        Port of R's ``.DollarNames.ggproto`` (R ref: ``ggproto.R:146-158``),
        which returns the union of the object's own names and its
        parents', excluding ``super``. Enables tab-completion of
        ``obj.<tab>`` and ``g$<tab>`` parity with R.
        """
        base = set(super().__dir__())
        for cls in type(self).__mro__:
            base.update(cls.__dict__.keys())
        # Walk instance super-chain too (R: Recall(x$super())).
        sup = object.__getattribute__(self, "_super_inst")
        seen = {id(self)}
        while sup is not None and id(sup) not in seen:
            seen.add(id(sup))
            base.update(dir(sup))
            sup = object.__getattribute__(sup, "_super_inst")
        base.discard("super")
        return sorted(base)

    def to_list(self) -> Dict[str, Any]:
        """Return a dict of all public members (R: ``as.list.ggproto``).

        Port of R's ``as.list.ggproto`` semantics: returns a named mapping
        of every accessible field and method, including inherited ones.
        Callables are returned as bound method references (not invoked).
        """
        out: Dict[str, Any] = {}
        for name in dir(self):
            if name.startswith("_"):
                continue
            if name == "super":
                continue
            out[name] = getattr(self, name)
        return out


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def ggproto(
    _class: Optional[str] = None,
    _inherit: Optional[Union[Type["GGProto"], "GGProto"]] = None,
    **members: Any,
) -> Union[Type["GGProto"], "GGProto"]:
    """Create a new ggproto class or instance.

    Mirrors R's ``ggproto()`` (R ref: ``ggproto.R:67-97``).  The
    ``_inherit`` argument may be:

    * ``None`` — defaults to :class:`GGProto`; returns a fresh subclass.
    * a :class:`GGProto` subclass — returns a fresh subclass inheriting
      from it (the **class-as-parent** path, original behaviour).
    * a :class:`GGProto` instance — returns a fresh **instance** whose
      class is a dynamic subclass of the parent instance's class.
      Only the explicit *members* are written onto the new
      instance; everything else is resolved lazily through the
      ``_super_inst`` chain, matching R's env-chain semantics
      (R ref: ``ggproto.R:118-142``: ``fetch_ggproto`` recurses into
      ``x$super()`` on miss).  This means **post-clone mutations on
      the parent are visible through the child**, matching R env
      modify-in-place behaviour — and conversely, writing to the
      child never touches the parent.

    Parameters
    ----------
    _class : str, optional
        Class name to assign.  When ``_inherit`` is an instance and
        *_class* is ``None``, the parent's class name is reused
        (no anonymous wrapping).  When *_class* is provided it is also
        attached to ``_class_name`` for ``repr`` parity with R's
        ``class(e) <- c(_class, ...)``.
    _inherit : type or GGProto, optional
        Parent class **or** parent instance.  See above.
    **members : Any
        Attributes and methods to install.  Functions whose first
        parameter is named ``self`` will behave as instance methods.

    Returns
    -------
    type or GGProto
        A new subclass (class-as-parent) or instance (instance-as-parent).

    Examples
    --------
    Class-as-parent (original behaviour):

    >>> Geom = ggproto("Geom", GGProto, draw_panel=lambda self, data: data)
    >>> Point = ggproto("Point", Geom, shape="circle")

    Instance-as-parent (ggnewscale-style clone-with-overrides):

    >>> point_inst = Point()
    >>> patched = ggproto("PatchedPoint", point_inst, shape="square")
    >>> patched.shape
    'square'
    >>> patched.super() is point_inst
    True
    """
    if _inherit is None:
        _inherit = GGProto

    # ---- Instance-as-parent path (R: _inherit is itself a ggproto env) ----
    if isinstance(_inherit, GGProto) and not isinstance(_inherit, type):
        parent_inst: GGProto = _inherit
        parent_cls = type(parent_inst)
        cls_name = _class if _class is not None else parent_cls.__name__
        # Dynamic subclass so isinstance(new, parent_cls) still holds.
        new_cls = GGProtoMeta(cls_name, (parent_cls,), {"_class_name": cls_name})
        new_obj = object.__new__(new_cls)
        # R env-chain semantics: do NOT snapshot the parent's
        # ``__dict__``.  Lookups walk ``_super_inst`` lazily so the
        # child sees post-clone parent mutations (R: ``fetch_ggproto``
        # recurses into ``x$super()`` on every lookup — R env-chain is
        # lazy, not eager).  Writes via ``setattr(child, k, v)`` /
        # ``child._set(...)`` create slots on the child only,
        # mirroring R env modify-in-place: assignments touch the
        # current env, not the super.
        object.__setattr__(new_obj, "_super_inst", parent_inst)
        # Apply explicit overrides at the instance level (R:
        # list2env(members, envir = e)).  Route through _set() so that
        # the footgun warning fires when applicable.
        if members:
            new_obj._set(**members)
        return new_obj

    # ---- Class-as-parent path (original behaviour) ----
    if not (isinstance(_inherit, type) and issubclass(_inherit, GGProto)):
        raise TypeError(
            "ggproto(): `_inherit` must be a GGProto class, a GGProto "
            f"instance, or None; got {type(_inherit).__name__}."
        )
    cls_name = _class if _class is not None else _inherit.__name__
    namespace: Dict[str, Any] = {"_class_name": cls_name}
    namespace.update(members)
    return GGProtoMeta(cls_name, (_inherit,), namespace)


# ---------------------------------------------------------------------------
# Parent-method dispatch
# ---------------------------------------------------------------------------

class _GGProtoParentProxy:
    """Proxy returned by :func:`ggproto_parent`.

    Attribute access retrieves a member from *parent* (a class **or**
    instance) and, if it is a plain function, binds it to *self_obj*.
    The two cases mirror R's ``$.ggproto_parent`` (R ref:
    ``ggproto.R:172-179``):

    * If *parent* is an instance, walk ``parent.__dict__`` first, then
      ``type(parent).__mro__`` (i.e. ``fetch_ggproto`` recursion).
    * If *parent* is a class, walk its MRO.
    """

    __slots__ = ("_parent", "_self")

    def __init__(self, parent: Union[Type[GGProto], GGProto], self_obj: Any) -> None:
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_self", self_obj)

    def __repr__(self) -> str:
        parent = object.__getattribute__(self, "_parent")
        name = getattr(parent, "__name__", None) or getattr(
            parent, "_class_name", None
        ) or type(parent).__name__
        return f"<ggproto parent proxy: {name}>"

    def __getattr__(self, name: str) -> Any:
        parent = object.__getattribute__(self, "_parent")
        self_obj = object.__getattribute__(self, "_self")

        # Instance-parent: check parent's own dict, then walk class MRO,
        # then recurse into parent._super_inst (R: fetch_ggproto).
        if isinstance(parent, GGProto) and not isinstance(parent, type):
            current: Optional[GGProto] = parent
            seen: set = set()
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                if name in current.__dict__:
                    value = current.__dict__[name]
                    if isinstance(value, types.FunctionType):
                        return types.MethodType(value, self_obj)
                    return value
                for cls in type(current).__mro__:
                    if name in cls.__dict__:
                        value = cls.__dict__[name]
                        if isinstance(value, types.FunctionType):
                            return types.MethodType(value, self_obj)
                        return value
                current = object.__getattribute__(current, "_super_inst")
            raise AttributeError(
                f"'{type(parent).__name__}' ggproto instance has no member "
                f"'{name}'"
            )

        # Class-parent path: walk the class MRO.
        for cls in parent.__mro__:  # type: ignore[union-attr]
            if name in cls.__dict__:
                value = cls.__dict__[name]
                if isinstance(value, types.FunctionType):
                    return types.MethodType(value, self_obj)
                return value
        raise AttributeError(
            f"'{parent.__name__}' ggproto object has no member '{name}'"  # type: ignore[union-attr]
        )


def ggproto_parent(
    parent: Union[Type[GGProto], GGProto],
    self: Any,
) -> _GGProtoParentProxy:
    """Get a proxy for calling parent-class (or parent-instance) methods.

    R ref: ``ggproto.R:102-104`` (``ggproto_parent <- function(parent, self)``).
    Methods accessed through the proxy will be bound to *self*.  The
    proxy resolves methods by:

    1. Looking up in *parent*'s own ``__dict__`` (for instance parents).
    2. Walking *parent*'s class MRO.
    3. Recursing into ``parent._super_inst`` (for instance parents).

    Parameters
    ----------
    parent : type or GGProto
        The parent (class or instance) whose methods should be called.
    self : GGProto
        The current object (``self`` in the calling method).

    Returns
    -------
    _GGProtoParentProxy
        A proxy that dispatches attribute access to *parent*'s methods,
        binding plain functions to *self*.

    Examples
    --------
    Inside a ggproto method:

    >>> def draw_panel(self, data, params):
    ...     data = ggproto_parent(Geom, self).draw_panel(data, params)
    ...     return data
    """
    return _GGProtoParentProxy(parent, self)


# ---------------------------------------------------------------------------
# Explicit method binding (PR-4)
# ---------------------------------------------------------------------------

def bind_method(obj: "GGProto", name: str, fn: Callable) -> None:
    """Install *fn* as a bound method on *obj* under *name*.

    Counterpart to R's ``ggproto(NULL, obj, name = fn)`` when *fn*'s
    first parameter is not literally called ``self``.  Without this,
    :meth:`GGProto.__getattribute__`'s auto-bind would leave *fn*
    unbound (it requires ``self`` as the first arg name).

    Unlike :meth:`GGProto._set`, this:

    1. Always produces a :class:`types.MethodType` bound to *obj*.
    2. Accepts any callable (lambdas, partials, closures, callables
       with arbitrary first-arg names).
    3. Does not emit the "first arg isn't 'self'" warning, because the
       binding is explicit.

    Parameters
    ----------
    obj : GGProto
        Instance to install the method on.
    name : str
        Attribute name.
    fn : callable
        Implementation.  Will be called as ``fn(obj, *args, **kwargs)``.

    Examples
    --------
    >>> def patched(this, data, params):
    ...     ...
    >>> bind_method(geom_inst, "handle_na", patched)
    >>> geom_inst.handle_na(df, {})   # calls patched(geom_inst, df, {})
    """
    if not isinstance(obj, GGProto):
        raise TypeError(
            f"bind_method(): obj must be a GGProto instance, got "
            f"{type(obj).__name__}"
        )
    if not callable(fn):
        raise TypeError(
            f"bind_method(): fn must be callable, got {type(fn).__name__}"
        )
    bound = types.MethodType(fn, obj)
    object.__setattr__(obj, name, bound)


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

    Mirrors R's ``fetch_ggproto`` recursion (R ref: ``ggproto.R:118-142``):
    look up *name* on *x*; if absent, walk into ``x.super()``.

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
        If the member does not exist anywhere along the super-chain.
    """
    if not is_ggproto(x):
        raise TypeError(f"Expected a GGProto object, got {type(x).__name__}")
    return getattr(x, name)
