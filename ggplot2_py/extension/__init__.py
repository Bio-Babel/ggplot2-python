"""
Horizontal-extension toolkit for ggplot2-python.

This subpackage exposes the helpers that **cross-cutting** extension
packages (e.g. a ggnewscale-style aes-bumping layer) need but should
not have to re-implement.  None of these names are part of the
top-level ``ggplot2_py`` namespace — extension authors opt in
explicitly::

    from ggplot2_py.extension import (
        clone_layer, rename_aes_in_mapping, protect, is_protected,
        palette_for_aes, register_pre_add_hook,
    )

Every helper documents its R reference in ggplot2 or ggnewscale.

Inventory
---------

Layer / geom / stat cloning
    :func:`clone_layer`, :func:`clone_geom`, :func:`clone_stat`

Aes renaming
    :func:`rename_aes_in_mapping`, :func:`rename_aes_in_dict`,
    :func:`rename_aes_in_seq`

Protection stamps (one-shot guards)
    :func:`protect`, :func:`is_protected`

Palette fallback
    :func:`palette_for_aes`

Pre-add hook plumbing
    :func:`register_pre_add_hook`, :func:`unregister_pre_add_hook`
    (re-exports from ``ggplot2_py.plot`` for one-stop import)
"""

from __future__ import annotations

import copy as _copy
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping as _MappingABC,
    Optional,
    Tuple,
)

from ggplot2_py.aes import Mapping
from ggplot2_py.ggproto import GGProto, ggproto, bind_method
from ggplot2_py.plot import register_pre_add_hook, unregister_pre_add_hook

__all__ = [
    # Cloning
    "clone_layer",
    "clone_geom",
    "clone_stat",
    # Aes renaming
    "rename_aes_in_mapping",
    "rename_aes_in_dict",
    "rename_aes_in_seq",
    # Protection
    "protect",
    "is_protected",
    # Palette fallback
    "palette_for_aes",
    # Hook plumbing (re-export)
    "register_pre_add_hook",
    "unregister_pre_add_hook",
]


# ---------------------------------------------------------------------------
# Cloning helpers
# ---------------------------------------------------------------------------

def clone_layer(layer: Any) -> Any:
    """Bounded-deep clone of a ``Layer`` so per-instance mutation is safe.

    Mirrors ggnewscale's ``new_layer <- ggproto(NULL, layer)`` followed
    by the field-by-field rewrites in ``bump_aes_layer`` (R ref:
    ``ggnewscale/R/bump-aes-layers.R:7``).  Python's ``copy.copy`` is
    too shallow on its own — ``mapping`` and the ``*_params`` dicts
    would still alias the source.  This helper produces a layer
    whose:

    * ``mapping``         is a fresh :class:`~ggplot2_py.aes.Mapping`,
    * ``aes_params``,
      ``geom_params``,
      ``stat_params``    are fresh ``dict`` copies,
    * ``geom`` and
      ``stat``           are fresh ``ggproto(None, inst)`` instance-
                         clones (so per-instance method overrides via
                         :func:`~ggplot2_py.ggproto.bind_method` won't
                         leak),
    * ``position``        is shallow-copied so per-layer mutable
                         position state (e.g. ``PositionJitter.seed``
                         resolved in ``setup_params`` at R ref:
                         ``position-jitter.R:61``) is isolated.

    Parameters
    ----------
    layer : Layer

    Returns
    -------
    Layer
        New, fully-isolated Layer.
    """
    new = _copy.copy(layer)
    # Mapping is a dict subclass; rebuilding it via its own constructor
    # preserves the subclass identity.
    if new.mapping is not None:
        new.mapping = type(new.mapping)(new.mapping)
    if new.aes_params:
        new.aes_params = dict(new.aes_params)
    if new.geom_params:
        new.geom_params = dict(new.geom_params)
    if new.stat_params:
        new.stat_params = dict(new.stat_params)
    if new.geom is not None and isinstance(new.geom, GGProto):
        new.geom = clone_geom(new.geom)
    if new.stat is not None and isinstance(new.stat, GGProto):
        new.stat = clone_stat(new.stat)
    # Position can hold per-layer mutable state (e.g.
    # ``PositionJitter`` resolves its ``seed`` lazily in
    # ``setup_params``).  Use ``ggproto(None, ...)`` when it's a
    # ``GGProto`` (so per-instance method overrides via
    # :func:`bind_method` don't leak) and a shallow ``copy.copy`` for
    # anything else.
    pos = getattr(new, "position", None)
    if pos is not None:
        if isinstance(pos, GGProto):
            new.position = ggproto(None, pos)
        else:
            new.position = _copy.copy(pos)
    return new


def clone_geom(geom: GGProto) -> GGProto:
    """Instance-level clone of a Geom for per-layer method overrides.

    Mirrors ggnewscale ``new_geom <- ggproto(paste0("New", class(...)),
    old_geom, ...)`` (R ref: ``bump-aes-layers.R:38-42``).  Uses
    :func:`~ggplot2_py.ggproto.ggproto` with an instance parent so the
    resulting object preserves ``isinstance`` against the source's
    class and can carry overrides on its own ``__dict__``.

    Parameters
    ----------
    geom : GGProto
        Source geom instance.

    Returns
    -------
    GGProto
        Fresh instance whose super-chain points back to *geom*.
    """
    if not isinstance(geom, GGProto):
        raise TypeError(f"clone_geom: expected GGProto, got {type(geom).__name__}")
    return ggproto(None, geom)  # type: ignore[return-value]


def clone_stat(stat: GGProto) -> GGProto:
    """Same as :func:`clone_geom`, for Stat (R ref:
    ``bump-aes-layers.R:85,88``).
    """
    if not isinstance(stat, GGProto):
        raise TypeError(f"clone_stat: expected GGProto, got {type(stat).__name__}")
    return ggproto(None, stat)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Aes renaming (R ref: ggnewscale/R/utils.R:23-37 change_name generic)
# ---------------------------------------------------------------------------

def rename_aes_in_mapping(mapping: Mapping, old: str, new: str) -> Mapping:
    """Return a copy of *mapping* with key *old* renamed to *new*.

    R ref: ``change_name.default`` in ``ggnewscale/R/utils.R:32-37``,
    applied to a ``Mapping`` (named list).  No-op if *old* is absent.

    Parameters
    ----------
    mapping : Mapping
    old, new : str
    """
    if mapping is None:
        return mapping  # type: ignore[return-value]
    if old not in mapping:
        # Return a same-type copy so callers can always treat the
        # result as a fresh object.
        return type(mapping)(mapping)
    return type(mapping)({(new if k == old else k): v for k, v in mapping.items()})


def rename_aes_in_dict(d: Optional[Dict[str, Any]], old: str, new: str) -> Optional[Dict[str, Any]]:
    """Same as :func:`rename_aes_in_mapping`, for a plain ``dict``.

    Used to rename keys inside ``aes_params``, ``default_aes`` (a dict
    subclass) and similar (R ref: ``change_name.default``).  Returns
    ``None`` unchanged.
    """
    if d is None:
        return None
    if old not in d:
        return dict(d)
    return {(new if k == old else k): v for k, v in d.items()}


def rename_aes_in_seq(seq: Iterable[str], old: str, new: str) -> List[str]:
    """Rename occurrences of *old* with *new* in a string sequence.

    Used for fields like ``required_aes`` / ``non_missing_aes`` /
    ``optional_aes`` which are tuples or lists of aesthetic names
    (R ref: ``change_name.character`` in ``utils.R:27-30``).

    Returns a freshly-allocated list; callers that need a tuple
    should wrap with ``tuple(...)``.
    """
    if seq is None:
        return []
    return [new if s == old else s for s in seq]


# ---------------------------------------------------------------------------
# Protection stamps (R ref: ggnewscale/R/utils.R:44-72)
# ---------------------------------------------------------------------------

_PROTECT_ATTR = "_ggnewscale_renamed"


def protect(obj: Any, aes_name: str, *, attr: str = _PROTECT_ATTR) -> Any:
    """Stamp *obj* so :func:`is_protected` returns ``True`` for *aes_name*.

    Mirrors R's S3 ``protect`` generic (``utils.R:44-58``).  The R
    version stores either ``$ggnewscale_renamed`` (env path) or
    ``attr(., "ggnewscale_renamed")`` (character vector path).  Python
    has unified attribute access so we always use an attribute named
    *attr* (defaults to the ggnewscale-compatible name).

    Idempotent and additive: stamping *aes_name* twice leaves a single
    entry.  Returns *obj* for chaining.

    Parameters
    ----------
    obj : object
        Anything that accepts ``setattr``.  Built-in immutable types
        (e.g. plain ``str``) are not supported — callers wrap in a
        custom carrier object before stamping.
    aes_name : str
    attr : str, keyword-only
        Attribute name to use.  Override only when implementing a
        rival extension that must not collide with ggnewscale's
        stamps.
    """
    current = list(getattr(obj, attr, []) or [])
    if aes_name not in current:
        current.append(aes_name)
    try:
        setattr(obj, attr, current)
    except AttributeError as e:
        raise TypeError(
            f"protect(): cannot stamp object of type "
            f"{type(obj).__name__}: {e}"
        )
    return obj


def is_protected(obj: Any, aes_name: str, *, attr: str = _PROTECT_ATTR) -> bool:
    """Reverse of :func:`protect`.

    R ref: ``is_protected`` in ``utils.R:62-72``.

    Parameters
    ----------
    obj : object
    aes_name : str
    attr : str, keyword-only

    Returns
    -------
    bool
    """
    stamps = getattr(obj, attr, None)
    if not stamps:
        return False
    return aes_name in stamps


# ---------------------------------------------------------------------------
# Palette fallback (R ref: ggnewscale/R/bump-aes-scales.R:73-96)
# ---------------------------------------------------------------------------

def palette_for_aes(
    original_aes: str,
    scale_template: Any = None,
    theme: Any = None,
) -> Optional[Callable]:
    """Resolve the palette R would assign to a scale of aesthetic
    *original_aes* under *theme*.

    Reproduces the trick in ggnewscale's ``use_fallback_palette``
    (R ref: ``bump-aes-scales.R:73-96``): construct a temporary
    ``ggplot() + dummy_scale``, run ``ScalesList.set_palettes`` against
    *theme*, and read the assigned palette back off the dummy scale.
    The original scale's aesthetics have already been bumped to a
    non-standard name (e.g. ``"colour_ggnewscale_1"``) so the theme's
    ``palette.<aes>.<dom>`` keys would no longer match — this helper
    fetches the palette that the *original* aesthetic name would have
    gotten, so callers can copy it onto the renamed scale.

    Parameters
    ----------
    original_aes : str
        Aesthetic name to look up (e.g. ``"colour"``, ``"fill"``).
    scale_template : Scale, optional
        Used to determine ``is_discrete``-ness.  If ``None``, the
        helper picks a generic continuous scale (matching R's default
        path when no template is supplied).
    theme : Theme, optional
        Theme to use for palette resolution.  ``None`` means use the
        package default.

    Returns
    -------
    callable or None
        The resolved palette, or ``None`` if no scale matched.
    """
    from ggplot2_py.plot import ggplot
    from ggplot2_py.theme import Theme, theme_get
    from ggplot2_py.scale import ScalesList
    if theme is None:
        try:
            theme = theme_get()
        except Exception:
            theme = Theme()
    # Build a minimal dummy scale that targets ``original_aes``.
    if scale_template is not None:
        # Clone the user's scale at the instance level and reset its
        # aesthetics to the original aes name — exactly mirroring R:
        # ``dummy_scale <- ggproto(NULL, scale, aesthetics = original_aes)``.
        dummy = ggproto(None, scale_template, aesthetics=[original_aes])
    else:
        from ggplot2_py.scales import scale_colour_continuous
        dummy = scale_colour_continuous(aesthetics=original_aes)
        # Generic fallback covers the common ``colour`` / ``fill``
        # case; callers needing other aesthetics should pass a
        # template.
    dummy_plot = ggplot()
    dummy_plot.scales = ScalesList()
    dummy_plot.scales.add(dummy)
    try:
        dummy_plot.scales.set_palettes(theme)
    except Exception:
        return None
    # The first (and only) scale on the dummy list now carries the
    # resolved palette.
    for s in dummy_plot.scales.scales:
        if original_aes in s.aesthetics:
            return getattr(s, "palette", None)
    return None
