"""
Entry-point-based plugin discovery for ggplot2_py.

This module scans ``importlib.metadata.entry_points`` for packages that
declare extensions under the following groups:

- ``ggplot2_py.geoms``     — custom Geom subclasses
- ``ggplot2_py.stats``     — custom Stat subclasses
- ``ggplot2_py.positions`` — custom Position subclasses
- ``ggplot2_py.scales``    — custom Scale subclasses
- ``ggplot2_py.coords``    — custom Coord subclasses
- ``ggplot2_py.facets``    — custom Facet subclasses

Extension packages declare entry points in their ``pyproject.toml``::

    [project.entry-points."ggplot2_py.geoms"]
    star = "my_ext.geoms:GeomStar"

    [project.entry-points."ggplot2_py.stats"]
    chull = "my_ext.stats:StatChull"

At ggplot2_py import time, :func:`discover_extensions` scans all installed
packages, loads the declared classes, and registers them in the
corresponding ``_registry`` dictionaries.  If a plugin fails to load
(e.g. missing dependency), a warning is emitted but import is not blocked.

This is a **Python-exclusive** extension mechanism — R has no equivalent.
"""

from __future__ import annotations

import importlib
import warnings
from importlib.metadata import entry_points
from typing import Any, Dict, List, Tuple

__all__ = ["discover_extensions", "list_extensions"]

# (entry_point_group, module_path, base_class_name)
_EXTENSION_GROUPS: List[Tuple[str, str, str]] = [
    ("ggplot2_py.geoms", "ggplot2_py.geom", "Geom"),
    ("ggplot2_py.stats", "ggplot2_py.stat", "Stat"),
    ("ggplot2_py.positions", "ggplot2_py.position", "Position"),
    ("ggplot2_py.scales", "ggplot2_py.scale", "Scale"),
    ("ggplot2_py.coords", "ggplot2_py.coord", "Coord"),
    ("ggplot2_py.facets", "ggplot2_py.facet", "Facet"),
]

_discovered: Dict[str, List[str]] = {}  # group -> list of names


def discover_extensions() -> Dict[str, List[str]]:
    """Scan installed packages for ggplot2_py entry-point extensions.

    For each declared entry point, the class is loaded and registered in
    the corresponding base class's ``_registry``.  Classes that are
    already registered (e.g. via ``__init_subclass__``) are skipped.

    Returns
    -------
    dict
        Mapping of ``{group: [name, ...]}`` for all discovered extensions.

    Examples
    --------
    ::

        from ggplot2_py._plugins import discover_extensions
        found = discover_extensions()
        print(found)
        # {'ggplot2_py.geoms': ['star'], 'ggplot2_py.stats': ['chull'], ...}
    """
    global _discovered
    result: Dict[str, List[str]] = {}

    eps = entry_points()

    for group, mod_path, base_name in _EXTENSION_GROUPS:
        group_eps = list(eps.select(group=group))
        if not group_eps:
            continue

        # Import the base module to access the registry
        try:
            mod = importlib.import_module(mod_path)
            base_cls = getattr(mod, base_name)
            registry = getattr(base_cls, "_registry", None)
        except Exception:
            continue

        names: List[str] = []
        for ep in group_eps:
            try:
                cls = ep.load()
                # Register under both the entry-point name and CamelCase
                if registry is not None:
                    if ep.name not in registry:
                        registry[ep.name] = cls
                    # Also register CamelCase form
                    camel = ep.name[0].upper() + ep.name[1:] if ep.name else ep.name
                    if camel not in registry:
                        registry[camel] = cls
                names.append(ep.name)
            except Exception as exc:
                warnings.warn(
                    f"ggplot2_py: failed to load extension '{ep.name}' "
                    f"from group '{group}': {exc}",
                    stacklevel=2,
                )

        if names:
            result[group] = names

    _discovered = result
    return result


def list_extensions() -> Dict[str, List[str]]:
    """Return previously discovered extensions (without re-scanning).

    Call :func:`discover_extensions` first, or rely on the automatic
    scan at ggplot2_py import time.

    Returns
    -------
    dict
        Mapping of ``{group: [name, ...]}`` for discovered extensions.
    """
    return dict(_discovered)
