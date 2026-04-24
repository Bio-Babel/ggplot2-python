"""
Plot rendering functions — conversion from built plot to gtable.

Extracted from plot.py to match R's separation of
plot-build.R (build pipeline) from plot-render.R (rendering).

Contains:
- ggplot_gtable() — convert built plot to gtable
- _table_add_legends() — build legends from scales
- _table_add_titles() — add title/subtitle/caption
- ggplotGrob() — build + render convenience
- find_panel() / panel_rows() / panel_cols() — panel location
- print_plot() — render to device

R references
------------
* ggplot2/R/plot-render.R
"""

from __future__ import annotations

import re
from functools import singledispatch
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from scales.colour_manip import to_rgba as _scales_to_rgba

from ggplot2_py._compat import Waiver, is_waiver, waiver

_R_GREY_RE = re.compile(r"^gr[ae]y(\d{1,3})$", re.IGNORECASE)

__all__ = [
    "ggplot_gtable",
    "ggplotGrob",
    "_safe_colour",
    "_table_add_legends",
    "_table_add_titles",
    "find_panel",
    "panel_rows",
    "panel_cols",
    "print_plot",
]


def _legend_label_width_cm(labels: List[Any], fontsize: float = 6.0) -> float:
    """Measure max label width in cm using Cairo font metrics.

    Replaces ``max(len(str(l)) for l in labels) * 0.18`` with actual
    text measurement, matching R's ``width_cm(grobs$labels)`` pattern.
    """
    from grid_py._size import calc_string_metric
    from grid_py import Gpar
    max_w = 0.0
    for l in labels:
        m = calc_string_metric(str(l), Gpar(fontsize=fontsize))
        max_w = max(max_w, m["width"] * 2.54)  # inches → cm
    return max(max_w, 0.3)  # minimum width 0.3 cm


# ---------------------------------------------------------------------------
# Layer-to-guide filtering (ports of R's matched_aes / include_layer_in_guide,
# ``ggplot2/R/guides-.R:871-912``).
#
# R's GuideLegend$process_layers only forwards layers whose aesthetic
# mapping actually maps one of the guide's aesthetics, unless the user
# explicitly set ``show.legend=TRUE``. Without this filter, a legend
# picks up ``draw_key`` from any convenient layer (e.g. a backbone
# ``geom_segment`` with a fixed black colour) and renders black path
# glyphs instead of the colour-scale dots it should show.
# ---------------------------------------------------------------------------

_AES_SYNONYMS: Dict[str, str] = {"color": "colour"}


def _canon_aes(name: str) -> str:
    return _AES_SYNONYMS.get(name, name)


def _aes_key_set(obj: Any) -> set:
    """Return the canonicalised set of aesthetic names in a mapping-like obj."""
    if obj is None:
        return set()
    try:
        keys = obj.keys() if hasattr(obj, "keys") else list(obj)
    except Exception:
        return set()
    return {_canon_aes(str(k)) for k in keys}


def _matched_aes(layer: Any, guide_aes: set) -> set:
    """Port of R's ``matched_aes`` (``guides-.R:871-880``).

    Returns the canonical aesthetic names that are *mapped* by this
    layer's ``aes()`` and also part of the guide's key columns, excluding
    aesthetics that are fixed (``aes_params``/``computed_geom_params``).
    """
    mapping_keys = _aes_key_set(getattr(layer, "computed_mapping", None)
                                or getattr(layer, "mapping", None))
    stat = getattr(layer, "stat", None)
    stat_default = _aes_key_set(getattr(stat, "default_aes", None))
    all_names = mapping_keys | stat_default

    geom = getattr(layer, "geom", None)
    geom_required = set()
    geom_default = set()
    if geom is not None:
        req = getattr(geom, "required_aes", None)
        if req is not None:
            geom_required = {_canon_aes(str(a)) for a in req}
        geom_default = _aes_key_set(getattr(geom, "default_aes", None))
    geom_names = geom_required | geom_default
    # R's rename_size shim: size-renaming geoms contribute to size
    # legends even without mapping "size" explicitly.
    if geom is not None and getattr(geom, "rename_size", False):
        if "size" in all_names and "linewidth" not in all_names:
            geom_names = geom_names | {"size"}

    matched = (all_names & geom_names) & {_canon_aes(a) for a in guide_aes}
    matched -= _aes_key_set(getattr(layer, "computed_geom_params", None))
    matched -= _aes_key_set(getattr(layer, "aes_params", None))
    return matched


def _include_layer_in_guide(layer: Any, matched: set) -> bool:
    """Port of R's ``include_layer_in_guide`` (``guides-.R:885-912``)."""
    show = getattr(layer, "show_legend", None)
    # Non-logical values: R warns and treats as FALSE. Python accepts
    # None (= NA) and bool; anything else is coerced to False.
    if show is not None and not isinstance(show, (bool, np.bool_)):
        # Named-dict form (``show.legend=c(colour=TRUE)``) — uncommon in
        # ggplot2_py but supported for completeness.
        if isinstance(show, dict):
            if not matched:
                return False
            picks = {_canon_aes(k): v for k, v in show.items()}
            vals = [picks[a] for a in matched if a in picks and picks[a] is not None]
            return len(vals) == 0 or any(vals)
        return False

    if matched:
        # Layer maps at least one of the guide's aesthetics:
        # include unless show.legend is explicitly FALSE.
        if show is None:
            return True
        return bool(show)
    # Layer does not map any guide aesthetic: include only if show.legend
    # is explicitly TRUE.
    return show is True


def _resolve_draw_key_for_entry(
    entry: Dict[str, Any], layers: Any,
) -> tuple[Any, List[Any]]:
    """Pick the ``draw_key`` and layer subset for a single legend entry.

    Mirrors R's ``GuideLegend$process_layers`` filtering combined with
    ``get_layer_key``'s first-layer-wins behaviour for glyph selection.
    Returns ``(draw_key_fn, included_layers)`` — the included layer
    list is forwarded to ``build_legend_decor`` so ``aes_params`` /
    ``default_aes`` resolution also uses only qualifying layers.
    """
    from ggplot2_py.draw_key import draw_key_point as _draw_key_point

    guide_aes = {_canon_aes(a) for a in (entry.get("aes_mapped") or {}).keys()}
    if not guide_aes:
        guide_aes = {_canon_aes(str(entry.get("aesthetic", "")))}

    included: List[Any] = []
    if layers:
        for layer in layers:
            matched = _matched_aes(layer, guide_aes)
            if _include_layer_in_guide(layer, matched):
                included.append(layer)

    draw_key_fn = _draw_key_point
    for layer in included:
        geom = getattr(layer, "geom", None)
        if geom is not None and hasattr(geom, "draw_key"):
            draw_key_fn = geom.draw_key
            break
    return draw_key_fn, included


@singledispatch
def ggplot_gtable(data: Any) -> Any:
    """Convert a built ggplot to a gtable for rendering.

    This is a :func:`functools.singledispatch` generic (R ref:
    ``plot-render.R:22``, ``UseMethod("ggplot_gtable")``).  Extension
    packages can register custom built-plot types::

        @ggplot_gtable.register(MyBuiltPlot)
        def _gtable_my_plot(data):
            ...

    Parameters
    ----------
    data : BuiltGGPlot
        Output from :func:`ggplot_build`.

    Returns
    -------
    gtable
        A gtable suitable for drawing with ``grid_draw()``.
    """
    raise TypeError(
        f"Cannot render object of type {type(data).__name__}. "
        "Expected a BuiltGGPlot instance."
    )


def _ggplot_gtable_impl(data):
    """Core ggplot_gtable implementation for BuiltGGPlot objects."""
    from gtable_py import (
        Gtable,
        gtable_add_grob,
        gtable_add_rows,
        gtable_add_cols,
        gtable_width,
        gtable_height,
    )
    from grid_py import null_grob

    plot = data.plot
    layout = data.layout
    layer_data = data.data
    theme = plot.theme
    labels = plot.labels

    # Draw geom grobs for each layer
    geom_grobs: List[Any] = []
    for i, layer in enumerate(plot.layers):
        if hasattr(layer, "draw_geom"):
            geom_grobs.append(layer.draw_geom(layer_data[i], layout))
        else:
            geom_grobs.append(null_grob())

    # Render panels via layout
    plot_table = layout.render(geom_grobs, layer_data, theme, labels)

    # Legends — build directly from trained non-position scales. Pass
    # ``plot.guides`` so ``guides(<aes>='none')`` user overrides suppress
    # the corresponding legend (R parity with ``plot-render.R``).
    plot_table = _table_add_legends(
        plot_table, plot.scales, labels, theme, layers=plot.layers,
        guides=plot.guides,
    )
    # R parity post-pass: ``table_add_legends`` in R (plot-render.R:70-73)
    # always emits all five guide-box cells (``right`` / ``left`` /
    # ``top`` / ``bottom`` / ``inside``), holding a zeroGrob when no
    # legend occupies the slot. ``_table_add_legends`` has many early
    # returns that skip this emission; backfill any missing slots here
    # so downstream consumers (patchwork's ``add_guides``) can detect
    # the modern 3.5+ layout by ``len(guide-box-*) == 5``.
    plot_table = _ensure_five_guide_box_slots(plot_table)

    # Title / subtitle / caption / tag annotations
    plot_table = _table_add_titles(plot_table, labels, theme)
    # Port of R's ``table_add_tag`` (plot-render.R:46, 228-340). Emits
    # a ``tag`` layout row so downstream consumers (patchwork's
    # ``recurse_tags`` → ``labs(tag=...)``) get the per-plot tag cell
    # R's API contract promises. Runs unconditionally — R always
    # allocates the tag margin slot even when no label is set.
    plot_table = _table_add_tag(plot_table, labels.get("tag"), theme)

    # Add plot margin (R: table_add_background, plot-render.R:342-345)
    # R: margin <- calc_element("plot.margin", theme) %||% margin()
    #    table  <- gtable_add_padding(table, margin)
    # Margin.unit preserves the original unit type (default "pt").
    if hasattr(plot_table, "_widths"):
        from gtable_py import gtable_add_padding
        from grid_py import Unit
        from ggplot2_py.theme_elements import Margin, ElementBlank
        try:
            from ggplot2_py.theme_elements import calc_element as _calc_el
            margin = _calc_el("plot.margin", theme)
            if margin is None or isinstance(margin, ElementBlank):
                margin = Margin(5.5, 5.5, 5.5, 5.5, unit="pt")
            elif not isinstance(margin, Margin):
                margin = Margin(5.5, 5.5, 5.5, 5.5, unit="pt")
            plot_table = gtable_add_padding(plot_table, margin.unit)
        except Exception:
            plot_table = gtable_add_padding(
                plot_table, Unit([0.2, 0.2, 0.2, 0.2], "cm"),
            )

    # Add alt-text attribute
    if hasattr(plot_table, "__dict__"):
        plot_table._alt_label = labels.get("alt", "")

    return plot_table


def _safe_colour(colour: Any) -> str:
    """Validate a colour value, returning 'grey50' for invalid inputs.

    Accepts anything ``scales.colour_manip.to_rgba`` can parse (hex strings,
    CSS/R named colours) plus R's ``grey<N>`` / ``gray<N>`` family
    (0-100 inclusive), which the scales parser does not special-case.
    """
    if colour is None:
        return "grey50"
    s = str(colour)
    m = _R_GREY_RE.match(s)
    if m and 0 <= int(m.group(1)) <= 100:
        return s
    try:
        _scales_to_rgba(s)
        return s
    except (ValueError, TypeError):
        return "grey50"


def _ensure_five_guide_box_slots(table: Any) -> Any:
    """Ensure every rendered ggplotGrob carries five guide-box cells.

    Port of R ``table_add_legends`` behaviour (plot-render.R:70-134):
    R unconditionally emits ``guide-box-left`` / ``-right`` / ``-top``
    / ``-bottom`` / ``-inside`` with zeroGrob placeholders when no legend
    occupies the slot. ``_table_add_legends`` in this module has many
    early-return paths (no scales, ``legend.position="none"``, etc.) so
    this post-pass backfills missing slots. Downstream consumers —
    notably patchwork's ``add_guides`` — key off ``len(guide-box-*) == 5``
    to dispatch to the modern ggplot2 ≥ 3.5 layout branch.
    """
    if not hasattr(table, "_widths"):
        return table

    names = list(table.layout.get("name", []))
    present = set()
    for n in names:
        if n.startswith("guide-box-"):
            present.add(n.split("guide-box-", 1)[1])
    missing = {"right", "left", "top", "bottom", "inside"} - present
    if not missing:
        return table

    from grid_py import Unit, null_grob
    from gtable_py import (
        gtable_add_cols,
        gtable_add_grob,
        gtable_add_rows,
    )

    # Find the panel span once; it's stable across our zero-sized
    # insertions on the outer frame.
    place = find_panel(table)

    if "right" in missing:
        table = gtable_add_cols(table, Unit([0.0], ["pt"]), pos=-1)
        table = gtable_add_cols(table, Unit([0.0], ["pt"]), pos=-1)
        ncol_t = len(table._widths)
        table = gtable_add_grob(
            table, null_grob(),
            t=place["t"], b=place["b"], l=ncol_t,
            clip="off", name="guide-box-right",
        )
    if "left" in missing:
        table = gtable_add_cols(table, Unit([0.0], ["pt"]), pos=0)
        table = gtable_add_cols(table, Unit([0.0], ["pt"]), pos=0)
        place = find_panel(table)
        table = gtable_add_grob(
            table, null_grob(),
            t=place["t"], b=place["b"], l=1,
            clip="off", name="guide-box-left",
        )
    if "bottom" in missing:
        table = gtable_add_rows(table, Unit([0.0], ["pt"]), pos=-1)
        table = gtable_add_rows(table, Unit([0.0], ["pt"]), pos=-1)
        nrow_t = len(table._heights)
        place = find_panel(table)
        table = gtable_add_grob(
            table, null_grob(),
            t=nrow_t, b=nrow_t, l=place["l"], r=place["r"],
            clip="off", name="guide-box-bottom",
        )
    if "top" in missing:
        table = gtable_add_rows(table, Unit([0.0], ["pt"]), pos=0)
        table = gtable_add_rows(table, Unit([0.0], ["pt"]), pos=0)
        place = find_panel(table)
        table = gtable_add_grob(
            table, null_grob(),
            t=1, b=1, l=place["l"], r=place["r"],
            clip="off", name="guide-box-top",
        )
    if "inside" in missing:
        place = find_panel(table)
        table = gtable_add_grob(
            table, null_grob(),
            t=place["t"], b=place["b"],
            l=place["l"], r=place["r"],
            clip="off", name="guide-box-inside",
        )
    return table


def _table_add_legends(
    table: Any, scales_list: Any, labels: Dict[str, Any], theme: Any,
    layers: Any = None, guides: Any = None,
) -> Any:
    """Build legends from trained non-position scales and add to the gtable.

    Each legend is built as an independent :class:`~gtable_py.Gtable` with
    its own viewport-based cell layout, faithfully mirroring R's
    ``GuideLegend`` pipeline.  Scales sharing the same title and breaks
    are merged into a single legend (R's guide-merge semantics).

    Mirrors R's ``table_add_legends`` in ``plot-render.R`` and the
    ``GuideLegend`` class in ``guide-legend.R``.

    Parameters
    ----------
    table : gtable
    scales_list : ScalesList
    labels : dict
    theme : Theme
    layers : list of Layer, optional
        Plot layers — used to determine the ``draw_key`` function for each
        aesthetic.

    Returns
    -------
    gtable
    """
    if not hasattr(table, "_widths"):
        return table

    import math
    from gtable_py import (
        gtable_add_grob,
        gtable_add_cols,
        gtable_add_rows,
        gtable_width,
        gtable_height,
    )
    from grid_py import Unit as unit, text_grob, Gpar

    from ggplot2_py.guide_legend import (
        build_legend_decor,
        build_legend_labels,
        measure_legend_grobs,
        arrange_legend_layout,
        assemble_legend,
        package_legend_box,
    )

    # ------------------------------------------------------------------
    # Theme legend.position resolution — R's ``default_position`` from
    # ``Guides$assemble`` (``guides-.R:480-486``).  When set to
    # ``"none"`` R short-circuits and no legends are drawn.  When a
    # numeric 2-vector is supplied it means inside placement.
    # ------------------------------------------------------------------
    from ggplot2_py.theme_elements import calc_element as _calc_legend_pos_el

    _default_position: Any = None
    if theme is not None:
        try:
            _default_position = _calc_legend_pos_el("legend.position", theme)
        except Exception:
            _default_position = None
        if _default_position is None:
            _default_position = theme.get("legend.position") if hasattr(
                theme, "get"
            ) else None
    if _default_position is None:
        _default_position = "right"

    # Numeric 2-vector → "inside"
    if isinstance(_default_position, (list, tuple, np.ndarray)) and len(
        _default_position
    ) == 2 and all(
        isinstance(v, (int, float, np.integer, np.floating))
        for v in _default_position
    ):
        _default_position = "inside"

    # R parity: legend.position="none" suppresses every legend.
    if _default_position == "none":
        return table

    if _default_position not in ("top", "right", "bottom", "left", "inside"):
        # Unknown position — fall back to right (R's behaviour is to emit
        # a zeroGrob at ``assemble`` time).
        _default_position = "right"

    # ------------------------------------------------------------------
    # 1. Collect raw legend entries from non-position scales
    # ------------------------------------------------------------------
    raw_entries: List[Dict[str, Any]] = []
    np_scales = (
        scales_list.non_position_scales()
        if hasattr(scales_list, "non_position_scales")
        else None
    )
    if np_scales is None or np_scales.n() == 0:
        return table

    # Build a per-aesthetic user-override map from ``plot.guides``. When
    # a user passes ``guides(<aes>='none')`` (or ``GuideNone()``) the
    # corresponding legend must be suppressed — R parity. ``Guides.train``
    # records the set on ``suppressed_aesthetics`` before dropping the
    # GuideNone entries, so read from there. Also support the pre-build
    # dict form for robustness.
    suppressed_aes: set = set()
    user_guide_by_aes: Dict[str, Any] = {}
    if guides is not None:
        from ggplot2_py.guide import GuideNone

        suppressed_aes |= set(getattr(guides, "suppressed_aesthetics", set()) or set())

        guides_dict = getattr(guides, "guides", None)
        if isinstance(guides_dict, dict):
            for ak, av in guides_dict.items():
                if av is None:
                    continue
                if (
                    av == "none"
                    or av is GuideNone
                    or isinstance(av, GuideNone)
                ):
                    suppressed_aes.add(ak)
                else:
                    user_guide_by_aes[ak] = av

    # ------------------------------------------------------------------
    # Per-entry position resolution. Mirrors R's ``Guides$assemble``
    # (guides-.R:488-493) where each guide's ``params$position[1]`` is
    # used, falling back to ``default_position``.  For Python we read
    # ``params["position"]`` on the user-supplied guide object (if any)
    # or on the scale-attached guide object.
    # ------------------------------------------------------------------
    def _resolve_entry_position(aes_name: str, scale_obj: Any) -> str:
        # User ``guides(<aes>=guide_legend(position=...))`` override.
        g_user = user_guide_by_aes.get(aes_name)
        if g_user is not None and hasattr(g_user, "params"):
            pos = g_user.params.get("position")
            if pos:
                return pos
        # Scale-level ``scale_<aes>_*(guide = guide_legend(position=...))``.
        sc_guide = getattr(scale_obj, "guide", None) if scale_obj is not None else None
        if sc_guide is not None and hasattr(sc_guide, "params"):
            pos = sc_guide.params.get("position")
            if pos:
                return pos
        return _default_position

    for sc in np_scales.scales:
        aes_name = sc.aesthetics[0] if sc.aesthetics else "unknown"

        # User asked for no legend for this aesthetic — skip.
        if aes_name in suppressed_aes or any(
            a in suppressed_aes for a in (sc.aesthetics or [])
        ):
            continue

        breaks = getattr(sc, "get_breaks", lambda: None)()
        if breaks is None or (hasattr(breaks, "__len__") and len(breaks) == 0):
            continue

        mapped = breaks
        if hasattr(sc, "map"):
            try:
                mapped = sc.map(breaks)
            except (TypeError, ValueError):
                pass

        if hasattr(sc, "get_labels"):
            try:
                labs = sc.get_labels(breaks)
            except (TypeError, ValueError, AttributeError):
                labs = [str(b) for b in breaks]
        else:
            labs = [str(b) for b in breaks]

        # Drop NA/NaN-mapped entries
        keep: List[int] = []
        mapped_arr = np.asarray(mapped) if not isinstance(mapped, np.ndarray) else mapped
        for j in range(len(breaks)):
            val = mapped_arr[j] if j < len(mapped_arr) else None
            try:
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    keep.append(j)
            except (TypeError, ValueError):
                keep.append(j)
        if not keep:
            continue
        breaks = [breaks[j] for j in keep]
        mapped = [mapped_arr[j] for j in keep]
        labs = [labs[j] for j in keep if j < len(labs)]

        # Scale-level ``name=`` overrides the aesthetic-derived default
        # (R parity: ``scale.name %||% labels[[aes]]``).
        scale_title = getattr(sc, "name", None)
        if scale_title is not None and not (
            hasattr(scale_title, "__class__")
            and scale_title.__class__.__name__ == "Waiver"
        ):
            title = scale_title
        else:
            title = labels.get(aes_name, aes_name)
            if hasattr(title, "__class__") and title.__class__.__name__ == "Waiver":
                title = aes_name

        raw_entries.append({
            "aesthetic": aes_name,
            "breaks": breaks,
            "mapped": mapped,
            "labels": labs,
            "title": str(title),
            "scale": sc,
            "is_continuous": not getattr(sc, "is_discrete", lambda: True)(),
            "is_binned": sc.__class__.__name__.startswith("ScaleBinned") or
                         getattr(sc, "guide", None) in ("bins", "coloursteps"),
            "position": _resolve_entry_position(aes_name, sc),
        })

    if not raw_entries:
        return table

    # ------------------------------------------------------------------
    # 2. Merge entries that share the same title + number of breaks
    #    (R merges guides whose hash — based on title and breaks — match)
    # ------------------------------------------------------------------
    merged: Dict[str, Dict[str, Any]] = {}
    for entry in raw_entries:
        key = entry["title"]
        if key in merged and len(merged[key]["breaks"]) == len(entry["breaks"]):
            merged[key]["aes_mapped"][entry["aesthetic"]] = entry["mapped"]
        else:
            merged[key] = {
                "title": entry["title"],
                "breaks": entry["breaks"],
                "labels": entry["labels"],
                "aes_mapped": {entry["aesthetic"]: entry["mapped"]},
                "scale": entry.get("scale"),
                "is_continuous": entry.get("is_continuous", False),
                "is_binned": entry.get("is_binned", False),
                "position": entry.get("position", _default_position),
            }
    entries = list(merged.values())

    # ------------------------------------------------------------------
    # 3. Resolve theme elements (R: calc_element for proper inheritance)
    #    R always has a complete theme.  If Python's theme is None or
    #    incomplete, reset the element tree and use theme_grey().
    # ------------------------------------------------------------------
    from ggplot2_py.theme_elements import calc_element as _calc_theme_el

    if theme is None:
        from ggplot2_py.theme_defaults import theme_grey
        theme = theme_grey()

    _ltitle_raw = _calc_theme_el("legend.title", theme)
    if _ltitle_raw is None:
        from ggplot2_py.theme_elements import reset_theme_settings
        reset_theme_settings()
        from ggplot2_py.theme_defaults import theme_grey as _tg
        theme = _tg()
        _ltitle_raw = _calc_theme_el("legend.title", theme)
    _ltext_raw = _calc_theme_el("legend.text", theme)

    title_size = float(_ltitle_raw.size)
    label_size = float(_ltext_raw.size)
    _ltitle_colour = _ltitle_raw.colour
    _ltext_colour = _ltext_raw.colour

    # Resolve legend key dimensions from theme
    # (R: GuideLegend$override_elements → width_cm/height_cm of theme units)
    from ggplot2_py.theme_elements import calc_element as _calc_el
    from grid_py import Unit as _Unit, convert_width, convert_height

    def _unit_to_cm(u, axis="height"):
        """Convert a theme Unit to cm using grid's **device-default** gp.

        Mirrors R's ``convertUnit(u, "cm", valueOnly=TRUE)`` called at
        gtable-construction time (pre-draw, no viewport active): R's
        grid falls back to the device default gp (``fontsize=12``,
        ``lineheight=1.2``).  R's ggplot2 uses this device default —
        **not** the theme's ``text`` element — when computing static
        layout sizes such as ``legend.key.width``.  grid_py's
        ``convert_*`` with no active viewport reproduces the same
        behaviour, so we just call it directly.
        """
        if u is None or not isinstance(u, _Unit):
            return None
        fn = convert_height if axis == "height" else convert_width
        cm = fn(u, "cm", valueOnly=True)
        val = float(np.sum(cm))
        return val if val > 0 else None

    key_size = _unit_to_cm(_calc_el("legend.key.size", theme))
    key_w = _unit_to_cm(_calc_el("legend.key.width", theme), "width")
    key_h = _unit_to_cm(_calc_el("legend.key.height", theme))
    spacing_x = _unit_to_cm(_calc_el("legend.key.spacing.x", theme), "width")
    spacing_y = _unit_to_cm(_calc_el("legend.key.spacing.y", theme))
    legend_spacing = _unit_to_cm(_calc_el("legend.spacing", theme))

    KEY_W_CM = key_w or key_size
    KEY_H_CM = key_h or key_size
    SPACING_X_CM = spacing_x
    SPACING_Y_CM = spacing_y
    PADDING_CM = 0.15  # R: legend.margin default padding

    # ------------------------------------------------------------------
    # 4. ``draw_key`` is now resolved per-entry inside the loop below
    # (R's ``GuideLegend$process_layers`` filters layers against each
    # guide's aesthetics via ``matched_aes`` / ``include_layer_in_guide``,
    # ``guide-legend.R:219-231``). See ``_resolve_draw_key_for_entry``.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 5. Build each guide as an independent Gtable
    #    Dispatch: continuous colour/fill → colourbar; else → legend
    # ------------------------------------------------------------------
    from ggplot2_py.guide_colourbar import (
        extract_colourbar_decor,
        extract_coloursteps_decor,
        build_colourbar_decor,
        build_coloursteps_decor,
        build_colourbar_labels,
        build_colourbar_ticks,
        assemble_colourbar,
    )

    # Helper: build a legend title grob with position_margin injection
    # (R guide-legend.R:326-334).  Applies equally to discrete-legend
    # titles and colourbar / coloursteps titles so that all three guide
    # flavours have the same visible gap between title and body.
    from ggplot2_py.theme_elements import (
        element_render as _el_render_t,
        calc_element as _calc_el_t,
        Margin as _Margin_t,
    )

    def _build_legend_title_grob(title_text: str, title_position: str = "top") -> Any:
        _title_el = _calc_el_t("legend.title", theme)
        _gap_pt = 0.0
        try:
            _sp = _calc_el_t("legend.key.spacing.x", theme) or _calc_el_t(
                "legend.key.spacing", theme
            )
            if _sp is not None:
                _gap_pt = float(np.sum(convert_width(_sp, "pt", valueOnly=True)))
        except Exception:
            _gap_pt = 5.5

        _bm = getattr(_title_el, "margin", None)
        if isinstance(_bm, _Margin_t):
            _mt, _mr, _mb, _ml = float(_bm.t), float(_bm.r), float(_bm.b), float(_bm.l)
            _mu = _bm.unit_str
            if _mu != "pt":
                _gap_val = float(np.sum(convert_width(_Unit(_gap_pt, "pt"), _mu, valueOnly=True)))
            else:
                _gap_val = _gap_pt
        else:
            _mt = _mr = _mb = _ml = 0.0
            _mu = "pt"
            _gap_val = _gap_pt

        if title_position == "top":
            _mb += _gap_val
        elif title_position == "bottom":
            _mt += _gap_val
        elif title_position == "left":
            _mr += _gap_val
        elif title_position == "right":
            _ml += _gap_val

        return _el_render_t(
            theme, "legend.title",
            label=str(title_text),
            margin=_Margin_t(t=_mt, r=_mr, b=_mb, l=_ml, unit=_mu),
            margin_x=True, margin_y=True,
        )

    legend_gtables: List[Any] = []
    legend_positions: List[str] = []

    for entry in entries:
        n_breaks = len(entry["breaks"])
        if n_breaks == 0:
            continue

        aes_names = list(entry["aes_mapped"].keys())
        is_colour_fill = any(a in ("colour", "color", "fill") for a in aes_names)
        is_continuous = entry.get("is_continuous", False)
        is_binned = entry.get("is_binned", False)
        sc = entry.get("scale")
        entry_pos = entry.get("position") or _default_position
        # R parity (guides-.R:574-577): direction defaults to "horizontal"
        # for top/bottom, "vertical" for left/right/inside.
        _entry_direction = (
            "horizontal" if entry_pos in ("top", "bottom") else "vertical"
        )

        # --- Coloursteps path: binned colour/fill scale ---
        if is_colour_fill and is_binned and sc is not None:
            title_grob = _build_legend_title_grob(entry["title"])

            # Extract stepped colour bins
            decor = extract_coloursteps_decor(
                sc, entry["breaks"], even_steps=True,
            )

            # Build stepped rectangle bar
            bar_parts = build_coloursteps_decor(decor, direction="vertical")

            # Labels and ticks (same as colourbar)
            limits = sc.get_limits()
            cb_labels = build_colourbar_labels(
                entry["breaks"], entry["labels"], limits,
                direction="vertical",
                label_size=label_size, label_colour=_ltext_colour,
            )
            ticks = build_colourbar_ticks(
                entry["breaks"], limits, direction="vertical",
            )

            label_w_cm = _legend_label_width_cm(entry["labels"], label_size)

            legend_gt = assemble_colourbar(
                bar_grob=bar_parts["bar"],
                frame_grob=bar_parts["frame"],
                ticks_grob=ticks,
                label_grobs=cb_labels,
                title_grob=title_grob,
                direction="vertical",
                bar_width_cm=KEY_W_CM,
                bar_height_cm=KEY_H_CM * 5,
                label_width_cm=label_w_cm,
                padding_cm=PADDING_CM,
                bg_colour="white",
            )
            legend_gtables.append(legend_gt)
            legend_positions.append(entry_pos)
            continue

        # --- Colourbar path: continuous colour/fill scale ---
        if is_colour_fill and is_continuous and sc is not None:
            title_grob = _build_legend_title_grob(entry["title"])

            # Extract dense colour sequence
            decor = extract_colourbar_decor(sc, nbin=300)

            # Build bar grob (raster mode)
            bar_parts = build_colourbar_decor(decor, direction="vertical",
                                              display="raster")

            # Build tick labels
            limits = sc.get_limits()
            cb_labels = build_colourbar_labels(
                entry["breaks"], entry["labels"], limits,
                direction="vertical",
                label_size=label_size, label_colour=_ltext_colour,
            )

            # Build tick marks
            ticks = build_colourbar_ticks(
                entry["breaks"], limits, direction="vertical",
            )

            # Estimate label width
            label_w_cm = _legend_label_width_cm(entry["labels"], label_size)

            # Assemble
            legend_gt = assemble_colourbar(
                bar_grob=bar_parts["bar"],
                frame_grob=bar_parts["frame"],
                ticks_grob=ticks,
                label_grobs=cb_labels,
                title_grob=title_grob,
                direction="vertical",
                bar_width_cm=KEY_W_CM,
                bar_height_cm=KEY_H_CM * 5,
                label_width_cm=label_w_cm,
                padding_cm=PADDING_CM,
                bg_colour="white",
            )
            legend_gtables.append(legend_gt)
            legend_positions.append(entry_pos)
            continue

        # --- Legend path: discrete scales ---
        # Mirror R ``GuideLegend$setup_params`` (``guide-legend.R:286-298``):
        # vertical direction defaults to ``ncol = ceiling(n_breaks / 20)``,
        # then ``nrow = ceiling(n_breaks / ncol)``. Previously hardcoded
        # ``ncol = 1`` caused any legend with more than 20 entries to pile
        # all wrapped entries into a single physical column (multi-column
        # positions were computed by ``arrange_legend_layout`` but only
        # one column width was allocated in the gtable), producing
        # overlapping key + label glyphs per row.
        # R ``GuideLegend$setup_params`` (``guide-legend.R:286-298``):
        # horizontal direction defaults to ``nrow = ceiling(n_breaks / 20)``
        # then ``ncol = ceiling(n_breaks / nrow)``; vertical mirrors the
        # transpose (ncol first).  ``_entry_direction`` was set above
        # based on the resolved legend position.
        if _entry_direction == "horizontal":
            nrow = max(1, math.ceil(n_breaks / 20))
            ncol = max(1, math.ceil(n_breaks / nrow))
        else:
            ncol = max(1, math.ceil(n_breaks / 20))
            nrow = max(1, math.ceil(n_breaks / ncol))

        # Per-entry draw_key: mirror R's ``matched_aes`` /
        # ``include_layer_in_guide`` so ``geom_segment`` / ``geom_path``
        # layers that don't map the guide's aesthetic can't hijack the
        # legend key glyph.
        entry_draw_key_fn, entry_layers = _resolve_draw_key_for_entry(
            entry, layers,
        )

        decor = build_legend_decor(
            entry, entry_draw_key_fn, entry_layers,
            key_width_cm=KEY_W_CM, key_height_cm=KEY_H_CM,
            theme=theme,
        )

        # R (guide-legend.R:433-450): labels are ``titleGrob``s with
        # the ``legend.text`` element's margin baked in, so
        # ``width_cm(label)`` includes the left/right margins — this is
        # what creates the visible gap between each key and its label.
        # Threading the theme through here gives us that behaviour.
        label_grobs = build_legend_labels(
            entry, label_size=label_size, label_colour=_ltext_colour,
            theme=theme, text_position="right",
        )

        sizes = measure_legend_grobs(
            decor, label_grobs, n_breaks,
            nrow=nrow, ncol=ncol,
            key_width_cm=KEY_W_CM, key_height_cm=KEY_H_CM,
            spacing_x=SPACING_X_CM, spacing_y=SPACING_Y_CM,
            text_position="right",
            label_size=label_size,
        )

        layout = arrange_legend_layout(
            n_breaks, nrow=nrow, ncol=ncol,
            text_position="right",
        )

        # Discrete-legend title with R's position_margin gap injection.
        title_grob = _build_legend_title_grob(entry["title"])
        _title_position = "top"

        legend_gt = assemble_legend(
            decor, label_grobs, title_grob,
            layout, sizes,
            title_position=_title_position,
            padding_cm=PADDING_CM,
            bg_colour="white",
        )
        legend_gtables.append(legend_gt)
        legend_positions.append(entry_pos)

    # R parity note (plot-render.R:70-73): ``table_add_legends``
    # unconditionally emits all five guide-box cells (left, right, top,
    # bottom, inside) — when no legend exists for a position, the cell
    # holds a zeroGrob and the spacing row/col is zero-sized. Downstream
    # patchwork's ``add_guides`` detects the modern 3.5+ layout by
    # counting ``guide-box-*`` == 5. Fall through with an empty
    # ``legend_gtables`` so the placement loop below still fires.

    # ------------------------------------------------------------------
    # 6. Package legends into per-position guide boxes — R parity with
    #    ``Guides$assemble`` (``guides-.R:474-568``) which groups drawn
    #    grobs by position and routes each group through ``package_box``.
    # ------------------------------------------------------------------
    from ggplot2_py.guide_legend import _gtable_total_cm

    # Group legend gtables by their resolved position.
    position_groups: Dict[str, List[Any]] = {}
    for _gt, _pos in zip(legend_gtables, legend_positions):
        position_groups.setdefault(_pos, []).append(_gt)

    # Package each position group (``Guides$package_box``,
    # ``guides-.R:592-757``).  Non-empty R positions get a real gtable;
    # positions with no guides get a zero placeholder.
    packaged_boxes: Dict[str, Any] = {}
    for _pos, _grobs in position_groups.items():
        packaged_boxes[_pos] = package_legend_box(
            _grobs, position=_pos, spacing_cm=legend_spacing,
        )

    # ------------------------------------------------------------------
    # 7. Place packaged guide boxes into the plot table — R parity with
    #    ``table_add_legends`` (``plot-render.R:68-145``).  R inserts
    #    right, left, bottom, top boxes at the plot-table extrema and
    #    the "inside" box at the panel cell.
    # ------------------------------------------------------------------
    # For every direction that has no legend, fall back to a
    # ``null_grob`` placeholder with zero-sized spacing + slot — matches
    # R's ``table_add_legends`` (plot-render.R:70-134) which always
    # emits all four outer cells.
    from grid_py import null_grob as _null_grob

    # Right legend ---------------------------------------------------------
    if "right" in packaged_boxes:
        box = packaged_boxes["right"]
        w_cm = max(_gtable_total_cm(box.widths), 1.0)
        place = find_panel(table)
        table = gtable_add_cols(table, unit([legend_spacing], "cm"), pos=-1)
        table = gtable_add_cols(table, unit([w_cm], "cm"), pos=-1)
        ncol_t = len(table._widths)
        table = gtable_add_grob(
            table, box, t=place["t"], b=place["b"], l=ncol_t,
            clip="off", name="guide-box-right",
        )
    else:
        place = find_panel(table)
        table = gtable_add_cols(table, unit([0], "cm"), pos=-1)
        table = gtable_add_cols(table, unit([0], "cm"), pos=-1)
        ncol_t = len(table._widths)
        table = gtable_add_grob(
            table, _null_grob(), t=place["t"], b=place["b"], l=ncol_t,
            clip="off", name="guide-box-right",
        )

    # Left legend ----------------------------------------------------------
    if "left" in packaged_boxes:
        box = packaged_boxes["left"]
        w_cm = max(_gtable_total_cm(box.widths), 1.0)
        # R inserts spacing at pos=0 first, then the legend column at
        # pos=0 — so the final order left-to-right is [legend, spacing,
        # ...existing...]. The ``find_panel`` call happens BEFORE the
        # column insertions so the panel rows are still accurate.
        place = find_panel(table)
        table = gtable_add_cols(table, unit([legend_spacing], "cm"), pos=0)
        table = gtable_add_cols(table, unit([w_cm], "cm"), pos=0)
        table = gtable_add_grob(
            table, box, t=place["t"], b=place["b"], l=1,
            clip="off", name="guide-box-left",
        )
    else:
        place = find_panel(table)
        table = gtable_add_cols(table, unit([0], "cm"), pos=0)
        table = gtable_add_cols(table, unit([0], "cm"), pos=0)
        table = gtable_add_grob(
            table, _null_grob(), t=place["t"], b=place["b"], l=1,
            clip="off", name="guide-box-left",
        )

    # Bottom legend --------------------------------------------------------
    if "bottom" in packaged_boxes:
        box = packaged_boxes["bottom"]
        h_cm = max(_gtable_total_cm(box.heights), 0.5)
        place = find_panel(table)
        table = gtable_add_rows(table, unit([legend_spacing], "cm"), pos=-1)
        table = gtable_add_rows(table, unit([h_cm], "cm"), pos=-1)
        nrow_t = len(table._heights)
        table = gtable_add_grob(
            table, box, t=nrow_t, b=nrow_t, l=place["l"], r=place["r"],
            clip="off", name="guide-box-bottom",
        )
    else:
        place = find_panel(table)
        table = gtable_add_rows(table, unit([0], "cm"), pos=-1)
        table = gtable_add_rows(table, unit([0], "cm"), pos=-1)
        nrow_t = len(table._heights)
        table = gtable_add_grob(
            table, _null_grob(), t=nrow_t, b=nrow_t, l=place["l"], r=place["r"],
            clip="off", name="guide-box-bottom",
        )

    # Top legend -----------------------------------------------------------
    if "top" in packaged_boxes:
        box = packaged_boxes["top"]
        h_cm = max(_gtable_total_cm(box.heights), 0.5)
        # R: ``table <- gtable_add_rows(table, spacing$top, pos = 0)``
        # then ``gtable_add_rows(table, heights$top, pos = 0)`` — so
        # final ordering top-to-bottom is [legend, spacing, ...existing...].
        place = find_panel(table)
        table = gtable_add_rows(table, unit([legend_spacing], "cm"), pos=0)
        table = gtable_add_rows(table, unit([h_cm], "cm"), pos=0)
        table = gtable_add_grob(
            table, box, t=1, b=1, l=place["l"], r=place["r"],
            clip="off", name="guide-box-top",
        )
    else:
        place = find_panel(table)
        table = gtable_add_rows(table, unit([0], "cm"), pos=0)
        table = gtable_add_rows(table, unit([0], "cm"), pos=0)
        table = gtable_add_grob(
            table, _null_grob(), t=1, b=1, l=place["l"], r=place["r"],
            clip="off", name="guide-box-top",
        )

    # Inside legend --------------------------------------------------------
    # R uses ``legend.position.inside`` / ``legend.justification.inside``
    # to place the inside box at the panel cell without altering the plot
    # gtable's row/column count.  The semantics of inside placement rely
    # on viewport justification within ``Guides$package_box`` which is
    # only partially wired in the Python port — see TODO below.
    if "inside" in packaged_boxes:
        box = packaged_boxes["inside"]
        place = find_panel(table)
        table = gtable_add_grob(
            table, box,
            t=place["t"], b=place["b"],
            l=place["l"], r=place["r"],
            clip="off", name="guide-box-inside",
        )
    else:
        # R parity: ``guide-box-inside`` is always emitted so
        # ``len(guide-box-*) == 5`` signals the modern layout to
        # patchwork's ``add_guides``. The null placeholder occupies the
        # panel cell but carries no grob content.
        place = find_panel(table)
        table = gtable_add_grob(
            table, _null_grob(),
            t=place["t"], b=place["b"],
            l=place["l"], r=place["r"],
            clip="off", name="guide-box-inside",
        )
        # TODO: ``legend.position.inside`` + ``legend.justification.inside``
        # viewport resolution is deferred — Python places the inside
        # guide-box at the panel cell only, not at a user-supplied
        # ``c(x, y)`` coordinate.  Matches R's gtable placement but
        # without the viewport-based x/y justification.
        from ggplot2_py._compat import cli_warn

        cli_warn(
            "Inside legend placement uses panel-cell geometry only. "
            "`legend.position.inside = c(x, y)` and "
            "`legend.justification.inside` are not yet wired through "
            "the viewport pipeline — the guide-box will appear at the "
            "panel centre."
        )

    return table


def _table_add_titles(table: Any, labels: Dict[str, Any], theme: Any) -> Any:
    """Add title, subtitle, caption annotations to the plot table.

    Mirrors R's ``table_add_titles()`` / ``table_add_caption()`` in
    ``plot-render.R`` (lines 147-224):
      1. Render the text via ``element_render(theme, element_name, label, ...)``
      2. Measure actual rendered height via ``grob_height(grob)``
      3. Add a row of that measured height to the gtable

    Parameters
    ----------
    table : gtable
        The plot gtable.
    labels : dict
        Plot labels (``title``, ``subtitle``, ``caption``).
    theme : Theme
        Complete theme.

    Returns
    -------
    gtable
        Modified table.
    """
    from gtable_py import gtable_add_grob, gtable_add_rows
    from grid_py import grob_height
    from ggplot2_py.theme_elements import element_render, calc_element

    if not hasattr(table, "_widths"):
        return table

    ncol = len(table._widths)

    # --- Caption (bottom) --- (R: plot-render.R:193-224)
    caption = labels.get("caption")
    if caption:
        caption_grob = element_render(
            theme, "plot.caption", label=str(caption),
            margin_y=True, margin_x=True,
        )
        caption_height = grob_height(caption_grob)
        table = gtable_add_rows(table, caption_height, pos=-1)
        nrow = len(table._heights)
        table = gtable_add_grob(
            table, caption_grob,
            t=nrow, l=1, r=ncol, clip="off", name="caption",
        )

    # --- Subtitle (top, added first so title goes above) ---
    # (R: plot-render.R:157-161, 182-184)
    subtitle = labels.get("subtitle")
    if subtitle:
        subtitle_grob = element_render(
            theme, "plot.subtitle", label=str(subtitle),
            margin_y=True, margin_x=True,
        )
        subtitle_height = grob_height(subtitle_grob)
        table = gtable_add_rows(table, subtitle_height, pos=0)
        table = gtable_add_grob(
            table, subtitle_grob,
            t=1, l=1, r=ncol, clip="off", name="subtitle",
        )

    # --- Title (top) --- (R: plot-render.R:150-154, 186-188)
    title = labels.get("title")
    if title:
        title_grob = element_render(
            theme, "plot.title", label=str(title),
            margin_y=True, margin_x=True,
        )
        title_height = grob_height(title_grob)
        table = gtable_add_rows(table, title_height, pos=0)
        table = gtable_add_grob(
            table, title_grob,
            t=1, l=1, r=ncol, clip="off", name="title",
        )

    return table


def _table_add_tag(table: Any, label: Any, theme: Any) -> Any:
    """Emit a ``tag`` layout row per R's ``table_add_tag``.

    Ports ``plot-render.R:228-340`` verbatim. The function has four
    behaviourally distinct paths driven by theme elements:

    1. Zero-padding on all four sides is added *unconditionally* via
       ``gtable_add_padding(table, unit(0, "pt"))`` (R:230). This gives
       the tag a stable margin slot even when the plot carries no tag.
    2. Early exit if ``label`` is missing or ``plot.tag`` is blank.
    3. Resolve ``plot.tag.position`` (one of the 8 corner / side
       keywords, or a numeric c(x, y)) and ``plot.tag.location``
       (``"margin"`` / ``"plot"`` / ``"panel"``).
    4. Render the tag grob, compute placement, call
       ``gtable_add_grob(..., name="tag", clip="off")``.

    Downstream in patchwork, ``recurse_tags`` emits
    ``p + labs(tag=...)`` per patch; rendering that patch through
    ``ggplotGrob`` hits this function, which is what makes the
    per-patch ``tag`` row visible to composition-level collectors.
    """
    from grid_py import Unit, grob_height, grob_width, unit_c
    from gtable_py import gtable_add_grob, gtable_add_padding
    from ggplot2_py.theme_elements import (
        ElementBlank,
        calc_element,
        element_render,
    )

    if not hasattr(table, "_widths"):
        return table

    # R:230 — always-on zero-pt padding on all four sides.
    table = gtable_add_padding(table, Unit([0.0], ["pt"]))

    # R:233-234 — no label: early exit (table keeps the zero padding).
    if label is None:
        return table
    if isinstance(label, str) and label == "":
        return table

    # R:236-239 — resolve plot.tag; blank element means "don't draw".
    try:
        element = calc_element("plot.tag", theme)
    except Exception:
        return table
    if element is None or isinstance(element, ElementBlank):
        return table

    # R:242-243 — resolve position + location. Defaults:
    #   position = "topleft"
    #   location = "margin" for keyword positions, "plot" for numeric.
    try:
        position = calc_element("plot.tag.position", theme)
    except Exception:
        position = None
    if position is None:
        position = "topleft"

    try:
        location = calc_element("plot.tag.location", theme)
    except Exception:
        location = None
    is_numeric_pos = (
        isinstance(position, (list, tuple))
        and len(position) == 2
        and all(isinstance(v, (int, float)) for v in position)
    )
    if location is None:
        location = "plot" if is_numeric_pos else "margin"

    # R:259-272 — derive top/left/right/bottom booleans from keyword.
    if is_numeric_pos:
        top = left = right = bottom = False
    else:
        valid_positions = (
            "topleft", "top", "topright", "left",
            "right", "bottomleft", "bottom", "bottomright",
        )
        pos_str = position if isinstance(position, str) else "topleft"
        if pos_str not in valid_positions:
            pos_str = "topleft"
        top    = pos_str in ("topleft",    "top",    "topright")
        left   = pos_str in ("topleft",    "left",   "bottomleft")
        right  = pos_str in ("topright",   "right",  "bottomright")
        bottom = pos_str in ("bottomleft", "bottom", "bottomright")

    # R:275-277 — render tag and measure.
    tag = element_render(
        theme, "plot.tag", label=str(label),
        margin_x=True, margin_y=True,
    )
    height = grob_height(tag)
    width  = grob_width(tag)

    # R:279-314 — "plot" / "panel" location: re-render with manual
    # (x, y) anchor. For "plot" we return immediately placing at the
    # full outer span; "panel" falls through to the common placement.
    if location in ("plot", "panel"):
        if location == "plot":
            n_rows = len(table._heights)
            n_cols = len(table._widths)
            return gtable_add_grob(
                table, tag,
                name="tag", clip="off",
                t=1, b=n_rows, l=1, r=n_cols,
            )
        # location == "panel" — use find_panel for placement.
        place = find_panel(table)
        t_ = place["t"]; l_ = place["l"]
        b_ = place["b"]; r_ = place["r"]
    else:
        # R:320-328 — "margin" location: resize the four padding cells
        # so the tag occupies its half of the corner. Native Unit
        # slicing (grid_py.Unit.__getitem__, R port of [.unit) carries
        # the per-entry grob references in ``data`` through every
        # operation, so the title / xlab / ylab lazy ``grobheight``
        # entries retain their grob link across the head/tail splice.
        n_col = len(table._widths)
        n_row = len(table._heights)
        if top:
            # R: table$heights <- unit.c(height, table$heights[-1])
            table.heights = unit_c(height, table._heights[1:])
        if left:
            table.widths = unit_c(width, table._widths[1:])
        if right:
            table.widths = unit_c(table._widths[:-1], width)
        if bottom:
            table.heights = unit_c(table._heights[:-1], height)
        t_, l_, b_, r_ = 1, 1, n_row, n_col

    # R:331-334 — shrink the placement to the correct edge/corner.
    if top:    b_ = t_
    if left:   r_ = l_
    if right:  l_ = r_
    if bottom: t_ = b_

    return gtable_add_grob(
        table, tag,
        name="tag", clip="off",
        t=t_, l=l_, b=b_, r=r_,
    )


# ---------------------------------------------------------------------------
# ggplotGrob
# ---------------------------------------------------------------------------

def ggplotGrob(plot: "GGPlot") -> Any:
    """Build and convert a ggplot to a gtable grob.

    Parameters
    ----------
    plot : GGPlot
        A ggplot object.

    Returns
    -------
    gtable
    """
    from ggplot2_py.plot import ggplot_build
    return ggplot_gtable(ggplot_build(plot))


def find_panel(table: Any) -> Dict[str, Any]:
    """Find the panel area in a gtable.

    Mirrors R's ``find_panel()`` in ``layout.R``.  Supports gtable layouts
    stored as either a ``pd.DataFrame`` or a plain dict-of-lists.

    Parameters
    ----------
    table : gtable
        A gtable object.

    Returns
    -------
    dict
        ``{"t": int, "l": int, "b": int, "r": int}`` panel bounds.
    """
    layout = getattr(table, "layout", None)
    if layout is None:
        return {"t": 1, "l": 1, "b": 1, "r": 1}

    # --- DataFrame path ---
    if isinstance(layout, pd.DataFrame):
        panel_rows = layout.loc[
            layout["name"].str.contains("panel", case=False, na=False)
        ]
        if not panel_rows.empty:
            return {
                "t": int(panel_rows["t"].min()),
                "l": int(panel_rows["l"].min()),
                "b": int(panel_rows["b"].max()),
                "r": int(panel_rows["r"].max()),
            }

    # --- dict-of-lists path (gtable_py stores layout this way) ---
    elif isinstance(layout, dict) and "name" in layout:
        names = layout["name"]
        indices = [i for i, n in enumerate(names)
                   if isinstance(n, str) and "panel" in n.lower()]
        if indices:
            return {
                "t": min(layout["t"][i] for i in indices),
                "l": min(layout["l"][i] for i in indices),
                "b": max(layout["b"][i] for i in indices),
                "r": max(layout["r"][i] for i in indices),
            }

    return {"t": 1, "l": 1, "b": 1, "r": 1}


def panel_rows(table: Any) -> Dict[str, int]:
    """Return the row range of panels in a gtable.

    Parameters
    ----------
    table : gtable

    Returns
    -------
    dict
        ``{"t": int, "b": int}``
    """
    p = find_panel(table)
    return {"t": p["t"], "b": p["b"]}


def panel_cols(table: Any) -> Dict[str, int]:
    """Return the column range of panels in a gtable.

    Parameters
    ----------
    table : gtable

    Returns
    -------
    dict
        ``{"l": int, "r": int}``
    """
    p = find_panel(table)
    return {"l": p["l"], "r": p["r"]}


# ---------------------------------------------------------------------------
# Matplotlib label helpers
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# print_plot
# ---------------------------------------------------------------------------

def print_plot(
    plot: "GGPlot",
    newpage: bool = True,
    vp: Any = None,
) -> "GGPlot":
    """Render a ggplot to the current device.

    Parameters
    ----------
    plot : GGPlot
        The plot to display.
    newpage : bool, optional
        If ``True``, create a new page / figure first.
    vp : Viewport, optional
        Viewport to draw in.

    Returns
    -------
    GGPlot
        The original plot (invisibly).
    """
    from grid_py import grid_draw, grid_newpage
    from ggplot2_py.plot import ggplot_build, set_last_plot

    set_last_plot(plot)

    if newpage and vp is None:
        grid_newpage()

    built = ggplot_build(plot)
    gtable = ggplot_gtable(built)

    if vp is None:
        grid_draw(gtable)
    else:
        from grid_py import push_viewport, up_viewport
        push_viewport(vp)
        grid_draw(gtable)
        up_viewport()

    return plot


# ---------------------------------------------------------------------------
# Deferred singledispatch registration for ggplot_gtable
# ---------------------------------------------------------------------------

def _register_ggplot_gtable_types():
    """Register BuiltGGPlot for ggplot_gtable dispatch.

    Called from plot.py after BuiltGGPlot is defined.
    """
    from ggplot2_py.plot import BuiltGGPlot
    ggplot_gtable.register(BuiltGGPlot)(_ggplot_gtable_impl)


_register_ggplot_gtable_types()
