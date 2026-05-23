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
from typing import Any, Dict, List, Optional, Tuple

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

    # R: ``table_add_background`` (plot-render.R:342-357) — first add
    # the plot margin via ``gtable_add_padding`` then drop the
    # ``plot.background`` element_grob into the full table at z=-Inf so
    # it sits behind everything else.
    if hasattr(plot_table, "_widths"):
        from gtable_py import gtable_add_padding, gtable_add_grob
        from ggplot2_py.theme_elements import (
            Margin, ElementBlank, calc_element as _calc_el,
            element_grob, is_theme_element,
        )

        margin = _calc_el("plot.margin", theme)
        if not isinstance(margin, Margin):
            margin = Margin(5.5, 5.5, 5.5, 5.5, unit="pt")
        # ``Margin`` IS-A ``Unit``; no ``.unit`` extraction needed.
        plot_table = gtable_add_padding(plot_table, margin)

        background = _calc_el("plot.background", theme)
        if background is not None and not isinstance(background, ElementBlank):
            bg_grob = element_grob(background)
            plot_table = gtable_add_grob(
                plot_table, bg_grob,
                t=1, l=1,
                b=len(plot_table._heights),
                r=len(plot_table._widths),
                clip="off", name="background", z=float("-inf"),
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
    # Build per-position guide-boxes via the OO Guides container — R parity
    # with ``Guides$setup → train → merge → process_layers → assemble``
    # (guides-.R:331-587). The container resolves user-supplied + scale-
    # default guides per aesthetic, trains each on its scale, merges by
    # hash, threads layer info into each guide's params, then routes the
    # resulting grobs into packaged guide-boxes keyed by position.
    # ------------------------------------------------------------------
    from ggplot2_py.guide import (
        Guides as _Guides,
        GuideLegend as _GL,
        guide_none as _gn,
    )
    from ggplot2_py.guide_legend import _gtable_total_cm

    np_scales = (
        scales_list.non_position_scales()
        if hasattr(scales_list, "non_position_scales")
        else None
    )
    if np_scales is None or np_scales.n() == 0:
        return table

    scales = list(np_scales.scales)
    aesthetics = [
        s.aesthetics[0] for s in scales
        if getattr(s, "aesthetics", None)
    ]

    user_guides = guides if isinstance(guides, _Guides) else _Guides()
    trained = user_guides.setup(
        scales,
        aesthetics=aesthetics,
        default=_GL(),
        missing=_gn(),
    )
    trained.train(scales, labels or {})
    trained.merge()
    trained.process_layers(layers or [], theme=theme)
    packaged_boxes = trained.assemble(theme) or {}
    packaged_boxes = {
        k: v for k, v in packaged_boxes.items() if v is not None
    }

    # Spacing between panel and legend column — R: ``legend.spacing`` theme.
    from ggplot2_py.theme_elements import calc_element as _calc_legend_spacing_el
    legend_spacing = 0.4
    if theme is not None:
        try:
            ls_el = _calc_legend_spacing_el("legend.spacing", theme)
            if ls_el is not None:
                from grid_py import convert_height as _convert_h
                legend_spacing = float(np.sum(_convert_h(ls_el, "cm", valueOnly=True)))
        except Exception:
            pass

    # Local imports used by the placement section below.
    from gtable_py import (
        gtable_add_grob,
        gtable_add_cols,
        gtable_add_rows,
    )
    from grid_py import Unit as unit

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
    # R guides-.R:621-625 + 716-720 — for ``position == "inside"``,
    # ``Guides$package_box`` wraps the guide-box gtable in a viewport at
    # ``legend.position.inside`` (NPC anchor point inside the panel) with
    # ``legend.justification.inside`` (which corner of the box anchors).
    # We replicate the same: build a viewport on the box, then drop it
    # into the panel cell.
    if "inside" in packaged_boxes:
        box = packaged_boxes["inside"]
        place = find_panel(table)

        from ggplot2_py.theme_elements import calc_element as _calc_el
        from grid_py import (
            Viewport as _Viewport,
            Unit as _Unit,
            edit_grob as _edit_grob,
            valid_just as _valid_just,
        )

        # ``legend.justification.inside`` falls back to
        # ``legend.justification`` (R guides-.R:618-619).  Default in both
        # is c(0.5, 0.5) (centre).
        just_inside = (
            _calc_el("legend.justification.inside", theme)
            or _calc_el("legend.justification", theme)
            or (0.5, 0.5)
        )
        # ``legend.position.inside`` falls back to the justification when
        # not set (R guides-.R:623).
        pos_inside = _calc_el("legend.position.inside", theme) or just_inside

        # R parity: ``grid:::valid.just`` handles both single-string
        # ("left" → c(0, 0.5)) and two-element ("right", "top") forms;
        # grid_py.valid_just is its faithful port and is bit-exact across
        # the 16-case parity matrix in validation/_verify_valid_just.{R,py}.
        # The previous local helper symmetrised single strings ("left" →
        # (0, 0)), which silently broke ``legend.justification = "top"``.
        xjust, yjust = _valid_just(just_inside)
        try:
            xpos, ypos = float(pos_inside[0]), float(pos_inside[1])
        except (TypeError, IndexError, ValueError):
            xpos, ypos = xjust, yjust

        # Use the box's measured size so the viewport doesn't expand to
        # fill the panel cell — matching R's ``height = vp_height``,
        # ``width = total_width`` from package_box (guides-.R:716-720).
        try:
            box_w_cm = max(_gtable_total_cm(box.widths), 0.0) or None
            box_h_cm = max(_gtable_total_cm(box.heights), 0.0) or None
        except Exception:
            box_w_cm = box_h_cm = None
        vp_kwargs: Dict[str, Any] = dict(
            x=_Unit([xpos], "npc"),
            y=_Unit([ypos], "npc"),
            just=(xjust, yjust),
        )
        if box_w_cm is not None:
            vp_kwargs["width"] = _Unit([box_w_cm], "cm")
        if box_h_cm is not None:
            vp_kwargs["height"] = _Unit([box_h_cm], "cm")

        box = _edit_grob(box, vp=_Viewport(**vp_kwargs))

        table = gtable_add_grob(
            table, box,
            t=place["t"], b=place["b"],
            l=place["l"], r=place["r"],
            clip="off", name="guide-box-inside",
        )
    else:
        # R parity: ``guide-box-inside`` is always emitted so
        # ``len(guide-box-*) == 5`` signals the modern layout to
        # patchwork's ``add_guides``. A null placeholder renders nothing
        # visible — no warning needed, because the user never asked for
        # an inside legend in the first place.
        place = find_panel(table)
        table = gtable_add_grob(
            table, _null_grob(),
            t=place["t"], b=place["b"],
            l=place["l"], r=place["r"],
            clip="off", name="guide-box-inside",
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

    # R always adds title/subtitle/caption rows even when the label is
    # NULL — element_render returns a zeroGrob and the row gets a 0-cm
    # height (plot-render.R:147-225). Downstream consumers (patchwork,
    # collect_axis_titles) rely on those rows existing for layout
    # stability, so we mirror the unconditional emission.

    def _label_or_none(key: str) -> Any:
        v = labels.get(key)
        if v is None:
            return None
        v = str(v)
        return v if v else None

    # --- Caption (bottom) --- (R: plot-render.R:193-224)
    caption_grob = element_render(
        theme, "plot.caption", label=_label_or_none("caption"),
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
    subtitle_grob = element_render(
        theme, "plot.subtitle", label=_label_or_none("subtitle"),
        margin_y=True, margin_x=True,
    )
    subtitle_height = grob_height(subtitle_grob)
    table = gtable_add_rows(table, subtitle_height, pos=0)
    table = gtable_add_grob(
        table, subtitle_grob,
        t=1, l=1, r=ncol, clip="off", name="subtitle",
    )

    # --- Title (top) --- (R: plot-render.R:150-154, 186-188)
    title_grob = element_render(
        theme, "plot.title", label=_label_or_none("title"),
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
