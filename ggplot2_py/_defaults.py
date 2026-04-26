"""Geom / stat default-aesthetic modifiers — port of R's
``geom-update-defaults.R``.

Lets users override ``Geom*.default_aes`` (and the analogous Stat
defaults) globally for the lifetime of a Python session, with a cache
that lets the user roll back via ``reset_geom_defaults`` /
``reset_stat_defaults``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ggplot2_py.aes import rename_aes
from ggplot2_py.geom import Geom
from ggplot2_py.stat import Stat

__all__ = [
    "update_geom_defaults",
    "update_stat_defaults",
    "reset_geom_defaults",
    "reset_stat_defaults",
    "get_geom_defaults",
]


# Cached pristine copies — populated lazily on first override so reset
# can restore the originals. R's equivalent: ``cache_defaults``.
_cache_geom: Dict[str, Any] = {}
_cache_stat: Dict[str, Any] = {}


def _resolve(name_or_cls: Any, registry: Dict[str, Any], kind: str) -> Any:
    """Resolve a name or class to the registered Geom/Stat class.

    Mirrors R ``validate_subclass``: accepts a string (snake-case
    suffix), a Geom/Stat subclass, or a layer-constructor function.
    """
    if isinstance(name_or_cls, str):
        cls = registry.get(name_or_cls) or registry.get(name_or_cls.lower())
        if cls is None:
            raise ValueError(f"No registered {kind} named {name_or_cls!r}.")
        return cls
    if isinstance(name_or_cls, type):
        return name_or_cls
    # A bound layer constructor returns a Layer; in R the function is
    # passed to validate_subclass which sniffs the geom/stat off it.
    raise TypeError(
        f"Expected a string name or {kind} class; got {type(name_or_cls).__name__}."
    )


def _snake_key(cls: Any, prefix: str) -> str:
    """``GeomPoint -> "point"``, used as the cache key."""
    name = cls.__name__
    if name.startswith(prefix):
        name = name[len(prefix):]
    # CamelCase → snake_case
    out = []
    for i, c in enumerate(name):
        if c.isupper() and i > 0:
            out.append("_")
        out.append(c.lower())
    return "".join(out) or name


def _update(cls: Any, new: Any, cache: Dict[str, Any]) -> Any:
    """Common update path for both geom and stat defaults."""
    key = _snake_key(cls, "Geom" if cache is _cache_geom else "Stat")
    old = dict(cls.default_aes) if hasattr(cls.default_aes, "items") else dict(cls.default_aes or {})

    if new is None:
        # Reset path — restore from cache, drop entry.
        cached = cache.pop(key, None)
        if cached is not None:
            cls.default_aes = type(cls.default_aes)(**cached) if hasattr(type(cls.default_aes), "__init__") else dict(cached)
        return old

    # First override: snapshot the pristine defaults so we can roll back.
    if key not in cache:
        cache[key] = old

    new_dict = rename_aes_dict(new)
    merged = {**old}
    for k, v in new_dict.items():
        merged[k] = v
    # Replace in the same container type the class originally used.
    cls.default_aes = type(cls.default_aes)(**merged) if hasattr(type(cls.default_aes), "__init__") else merged
    return old


def rename_aes_dict(new: Any) -> Dict[str, Any]:
    """Apply :func:`rename_aes` to a Mapping or plain dict."""
    if hasattr(new, "items"):
        items = list(new.items())
    else:
        items = list(new)
    keys = [k for k, _ in items]
    canonical = rename_aes({k: None for k in keys})
    name_map = {k: c for k, c in zip(keys, list(canonical.keys()))}
    return {name_map[k]: v for k, v in items}


def update_geom_defaults(geom: Any, new: Any) -> Any:
    """Modify a Geom's ``default_aes``.  Pass ``None`` to roll back.

    Mirrors R ``update_geom_defaults`` (geom-update-defaults.R:57-59).
    """
    cls = _resolve(geom, Geom._registry, "geom")
    return _update(cls, new, _cache_geom)


def update_stat_defaults(stat: Any, new: Any) -> Any:
    """Modify a Stat's ``default_aes``.  Pass ``None`` to roll back.

    Mirrors R ``update_stat_defaults`` (geom-update-defaults.R:63-65).
    """
    cls = _resolve(stat, Stat._registry, "stat")
    return _update(cls, new, _cache_stat)


def reset_geom_defaults() -> None:
    """Roll back every prior ``update_geom_defaults`` call.

    Mirrors R ``reset_geom_defaults`` (geom-update-defaults.R:120).
    """
    for key in list(_cache_geom.keys()):
        cls = Geom._registry.get(key) or Geom._registry.get(key.lower())
        if cls is not None:
            _update(cls, None, _cache_geom)


def reset_stat_defaults() -> None:
    """Roll back every prior ``update_stat_defaults`` call.

    Mirrors R ``reset_stat_defaults`` (geom-update-defaults.R:124).
    """
    for key in list(_cache_stat.keys()):
        cls = Stat._registry.get(key) or Stat._registry.get(key.lower())
        if cls is not None:
            _update(cls, None, _cache_stat)


def get_geom_defaults(geom: Any, theme: Optional[Any] = None) -> Dict[str, Any]:
    """Return the resolved default aesthetics for *geom*.

    Mirrors R ``get_geom_defaults`` (geom-update-defaults.R:96-116).
    Accepts a class, a registered name, or a layer-constructor
    function.  ``theme`` is honoured for ``FromTheme`` defaults.
    """
    if callable(geom) and not isinstance(geom, type):
        # Layer constructor (geom_point) — call to obtain the Layer
        layer = geom()
        cls = type(getattr(layer, "geom", None) or geom)
        if hasattr(layer, "aes_params"):
            base = dict(cls.default_aes) if hasattr(cls.default_aes, "items") else dict(cls.default_aes or {})
            base.update(layer.aes_params or {})
            return base
    if isinstance(geom, str):
        cls = _resolve(geom, Geom._registry, "geom")
    elif isinstance(geom, type):
        cls = geom
    else:
        cls = type(geom)

    base = dict(cls.default_aes) if hasattr(cls.default_aes, "items") else dict(cls.default_aes or {})
    if theme is not None:
        from ggplot2_py.geom import _eval_from_theme
        base = dict(_eval_from_theme(cls.default_aes, theme))
    return base
