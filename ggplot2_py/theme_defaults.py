"""
Built-in complete themes for ggplot2.

Each function returns a ``Theme`` with all elements defined.
``theme_grey()`` is the base default; all others derive from it.
"""

from __future__ import annotations

from typing import Optional

from grid_py import Unit

from ggplot2_py.theme_elements import (
    element_blank,
    element_line,
    element_rect,
    element_text,
    element_point,
    element_polygon,
    element_geom,
    margin,
    margin_auto,
    rel,
)
from ggplot2_py.theme import Theme, theme, theme_replace_op

__all__ = [
    "theme_grey",
    "theme_gray",
    "theme_bw",
    "theme_linedraw",
    "theme_light",
    "theme_dark",
    "theme_minimal",
    "theme_classic",
    "theme_void",
    "theme_test",
    "theme_sub_axis",
    "theme_sub_axis_x",
    "theme_sub_axis_y",
    "theme_sub_axis_top",
    "theme_sub_axis_bottom",
    "theme_sub_axis_left",
    "theme_sub_axis_right",
    "theme_sub_legend",
    "theme_sub_panel",
    "theme_sub_plot",
    "theme_sub_strip",
]


# ---------------------------------------------------------------------------
# Colour mixing helper
# ---------------------------------------------------------------------------

def _col_mix(ink: str, paper: str, amount: float) -> str:
    """Mix *ink* and *paper* colours.

    Parameters
    ----------
    ink : str
        Foreground colour (e.g. ``"black"``).
    paper : str
        Background colour (e.g. ``"white"``).
    amount : float
        Fraction of *paper* to blend in (0 = pure ink, 1 = pure paper).

    Returns
    -------
    str
        A hex colour string.
    """
    import matplotlib.colors as mcolors
    import numpy as np

    try:
        c_ink = np.array(mcolors.to_rgba(ink))
        c_paper = np.array(mcolors.to_rgba(paper))
    except ValueError:
        # Fallback for colours matplotlib cannot parse
        return ink

    mixed = c_ink * (1 - amount) + c_paper * amount
    mixed = np.clip(mixed, 0, 1)
    return mcolors.to_hex(mixed, keep_alpha=True)


# ---------------------------------------------------------------------------
# Helper to build an all-None theme (mimics ggplot_global$theme_all_null)
# ---------------------------------------------------------------------------

def _theme_all_null() -> Theme:
    """Return a complete theme with every element set to ``None``.

    This is used by complete themes so that elements not explicitly set
    in the theme definition become ``None`` rather than being missing.

    Returns
    -------
    Theme
    """
    from ggplot2_py.theme_elements import get_element_tree

    elements = {name: None for name in get_element_tree()}
    return Theme(elements=elements, complete=True, validate=False)


# ---------------------------------------------------------------------------
# theme_grey / theme_gray
# ---------------------------------------------------------------------------

def theme_grey(
    base_size: float = 11,
    base_family: str = "",
    header_family: Optional[str] = None,
    base_line_size: Optional[float] = None,
    base_rect_size: Optional[float] = None,
    ink: str = "black",
    paper: str = "white",
    accent: str = "#3366FF",
) -> Theme:
    """The default ggplot2 theme with grey background and white gridlines.

    Parameters
    ----------
    base_size : float
        Base font size in points (default 11).
    base_family : str
        Base font family.
    header_family : str or None
        Font family for titles and headers.
    base_line_size : float or None
        Base size for line elements (default ``base_size / 22``).
    base_rect_size : float or None
        Base size for rect elements (default ``base_size / 22``).
    ink : str
        Foreground colour (default ``"black"``).
    paper : str
        Background colour (default ``"white"``).
    accent : str
        Accent colour (default ``"#3366FF"``).

    Returns
    -------
    Theme
        A complete theme.
    """
    if base_line_size is None:
        base_line_size = base_size / 22
    if base_rect_size is None:
        base_rect_size = base_size / 22

    half_line = base_size / 2

    t = theme(
        complete=True,
        # Root elements
        line=element_line(
            colour=ink,
            linewidth=base_line_size,
            linetype=1,
            lineend="butt",
            linejoin="round",
        ),
        rect=element_rect(
            fill=paper,
            colour=ink,
            linewidth=base_rect_size,
            linetype=1,
            linejoin="round",
        ),
        text=element_text(
            family=base_family,
            face="plain",
            colour=ink,
            size=base_size,
            lineheight=0.9,
            hjust=0.5,
            vjust=0.5,
            angle=0,
            margin=margin(),
            debug=False,
        ),
        title=element_text(family=header_family),
        spacing=Unit(half_line, "pt"),
        margins=margin_auto(half_line),
        point=element_point(
            colour=ink,
            shape=19,
            fill=paper,
            size=(base_size / 11) * 1.5,
            stroke=base_line_size,
        ),
        polygon=element_polygon(
            fill=paper,
            colour=ink,
            linewidth=base_rect_size,
            linetype=1,
            linejoin="round",
        ),
        geom=element_geom(
            ink=ink,
            paper=paper,
            accent=accent,
            linewidth=base_line_size,
            borderwidth=base_line_size,
            linetype=1,
            bordertype=1,
            family=base_family,
            fontsize=base_size,
            pointsize=(base_size / 11) * 1.5,
            pointshape=19,
        ),

        # Axis lines
        axis_line=element_blank(),
        axis_line_x=None,
        axis_line_y=None,

        # Axis text
        axis_text=element_text(
            size=rel(0.8),
            colour=_col_mix(ink, paper, 0.302),
        ),
        axis_text_x=element_text(
            margin=margin(t=0.8 * half_line / 2),
            vjust=1,
        ),
        axis_text_x_top=element_text(
            margin=margin(b=0.8 * half_line / 2),
            vjust=0,
        ),
        axis_text_y=element_text(
            margin=margin(r=0.8 * half_line / 2),
            hjust=1,
        ),
        axis_text_y_right=element_text(
            margin=margin(l=0.8 * half_line / 2),
            hjust=0,
        ),
        axis_text_r=element_text(
            margin=margin(l=0.8 * half_line / 2, r=0.8 * half_line / 2),
            hjust=0.5,
        ),

        # Axis ticks
        axis_ticks=element_line(colour=_col_mix(ink, paper, 0.2)),
        axis_ticks_length=rel(0.5),
        axis_ticks_length_x=None,
        axis_ticks_length_x_top=None,
        axis_ticks_length_x_bottom=None,
        axis_ticks_length_y=None,
        axis_ticks_length_y_left=None,
        axis_ticks_length_y_right=None,
        axis_minor_ticks_length=rel(0.75),

        # Axis titles
        axis_title_x=element_text(
            margin=margin(t=half_line / 2),
            vjust=1,
        ),
        axis_title_x_top=element_text(
            margin=margin(b=half_line / 2),
            vjust=0,
        ),
        axis_title_y=element_text(
            angle=90,
            margin=margin(r=half_line / 2),
            vjust=1,
        ),
        axis_title_y_right=element_text(
            angle=-90,
            margin=margin(l=half_line / 2),
            vjust=1,
        ),

        # Legend
        legend_background=element_rect(colour=None),
        legend_spacing=rel(2),
        legend_spacing_x=None,
        legend_spacing_y=None,
        legend_margin=None,
        legend_key=None,
        legend_key_size=Unit(1.2, "lines"),
        legend_key_height=None,
        legend_key_width=None,
        legend_key_spacing=None,
        legend_text=element_text(size=rel(0.8)),
        legend_title=element_text(hjust=0),
        legend_ticks_length=rel(0.2),
        legend_position="right",
        legend_direction=None,
        legend_justification="center",
        legend_box=None,
        legend_box_margin=margin_auto(0),
        legend_box_background=element_blank(),
        legend_box_spacing=rel(2),

        # Panel
        panel_background=element_rect(
            fill=_col_mix(ink, paper, 0.92),
            colour=None,
        ),
        panel_border=element_blank(),
        panel_grid=element_line(colour=paper),
        panel_grid_minor=element_line(linewidth=rel(0.5)),
        panel_spacing=None,
        panel_spacing_x=None,
        panel_spacing_y=None,
        panel_ontop=False,

        # Strip
        strip_background=element_rect(
            fill=_col_mix(ink, paper, 0.85),
            colour=None,
        ),
        strip_clip="on",
        strip_text=element_text(
            colour=_col_mix(ink, paper, 0.1),
            size=rel(0.8),
            margin=margin_auto(0.8 * half_line),
        ),
        strip_text_x=None,
        strip_text_y=element_text(angle=-90),
        strip_text_y_left=element_text(angle=90),
        strip_placement="inside",
        strip_placement_x=None,
        strip_placement_y=None,
        strip_switch_pad_grid=Unit(half_line / 2, "pt"),
        strip_switch_pad_wrap=Unit(half_line / 2, "pt"),

        # Plot
        plot_background=element_rect(colour=paper),
        plot_title=element_text(
            size=rel(1.2),
            hjust=0,
            vjust=1,
            margin=margin(b=half_line),
        ),
        plot_title_position="panel",
        plot_subtitle=element_text(
            hjust=0,
            vjust=1,
            margin=margin(b=half_line),
        ),
        plot_caption=element_text(
            size=rel(0.8),
            hjust=1,
            vjust=1,
            margin=margin(t=half_line),
        ),
        plot_caption_position="panel",
        plot_tag=element_text(
            size=rel(1.2),
            hjust=0.5,
            vjust=0.5,
        ),
        plot_tag_position="topleft",
        plot_margin=None,
    )

    # Merge onto the all-null base so unset elements become None
    base = _theme_all_null()
    return theme_replace_op(base, t)


# Alias
theme_gray = theme_grey


# ---------------------------------------------------------------------------
# theme_bw
# ---------------------------------------------------------------------------

def theme_bw(
    base_size: float = 11,
    base_family: str = "",
    header_family: Optional[str] = None,
    base_line_size: Optional[float] = None,
    base_rect_size: Optional[float] = None,
    ink: str = "black",
    paper: str = "white",
    accent: str = "#3366FF",
) -> Theme:
    """Classic dark-on-light theme with a white panel and dark border.

    Parameters
    ----------
    base_size : float
        Base font size in points.
    base_family : str
        Base font family.
    header_family : str or None
        Header font family.
    base_line_size : float or None
        Base line size.
    base_rect_size : float or None
        Base rect size.
    ink : str
        Foreground colour.
    paper : str
        Background colour.
    accent : str
        Accent colour.

    Returns
    -------
    Theme
    """
    base = theme_grey(
        base_size=base_size,
        base_family=base_family,
        header_family=header_family,
        base_line_size=base_line_size,
        base_rect_size=base_rect_size,
        ink=ink,
        paper=paper,
        accent=accent,
    )
    override = theme(
        complete=True,
        panel_background=element_rect(fill=paper, colour=None),
        panel_border=element_rect(colour=_col_mix(ink, paper, 0.2)),
        panel_grid=element_line(colour=_col_mix(ink, paper, 0.92)),
        panel_grid_minor=element_line(linewidth=rel(0.5)),
        strip_background=element_rect(
            fill=_col_mix(ink, paper, 0.851),
            colour=_col_mix(ink, paper, 0.2),
        ),
    )
    return theme_replace_op(base, override)


# ---------------------------------------------------------------------------
# theme_linedraw
# ---------------------------------------------------------------------------

def theme_linedraw(
    base_size: float = 11,
    base_family: str = "",
    header_family: Optional[str] = None,
    base_line_size: Optional[float] = None,
    base_rect_size: Optional[float] = None,
    ink: str = "black",
    paper: str = "white",
    accent: str = "#3366FF",
) -> Theme:
    """A theme with only black lines of various widths on white backgrounds.

    Parameters
    ----------
    base_size : float
        Base font size in points.
    base_family : str
        Base font family.
    header_family : str or None
        Header font family.
    base_line_size : float or None
        Base line size.
    base_rect_size : float or None
        Base rect size.
    ink : str
        Foreground colour.
    paper : str
        Background colour.
    accent : str
        Accent colour.

    Returns
    -------
    Theme
    """
    half_line = base_size / 2

    base = theme_bw(
        base_size=base_size,
        base_family=base_family,
        header_family=header_family,
        base_line_size=base_line_size,
        base_rect_size=base_rect_size,
        ink=ink,
        paper=paper,
        accent=accent,
    )
    override = theme(
        complete=True,
        axis_text=element_text(colour=ink, size=rel(0.8)),
        axis_ticks=element_line(colour=ink, linewidth=rel(0.5)),
        panel_border=element_rect(colour=ink, linewidth=rel(1)),
        panel_grid=element_line(colour=ink),
        panel_grid_major=element_line(linewidth=rel(0.1)),
        panel_grid_minor=element_line(linewidth=rel(0.05)),
        strip_background=element_rect(fill=ink),
        strip_text=element_text(
            colour=paper,
            size=rel(0.8),
            margin=margin_auto(0.8 * half_line),
        ),
    )
    return theme_replace_op(base, override)


# ---------------------------------------------------------------------------
# theme_light
# ---------------------------------------------------------------------------

def theme_light(
    base_size: float = 11,
    base_family: str = "",
    header_family: Optional[str] = None,
    base_line_size: Optional[float] = None,
    base_rect_size: Optional[float] = None,
    ink: str = "black",
    paper: str = "white",
    accent: str = "#3366FF",
) -> Theme:
    """A theme similar to ``theme_linedraw`` but with light grey lines.

    Parameters
    ----------
    base_size : float
        Base font size in points.
    base_family : str
        Base font family.
    header_family : str or None
        Header font family.
    base_line_size : float or None
        Base line size.
    base_rect_size : float or None
        Base rect size.
    ink : str
        Foreground colour.
    paper : str
        Background colour.
    accent : str
        Accent colour.

    Returns
    -------
    Theme
    """
    half_line = base_size / 2

    base = theme_grey(
        base_size=base_size,
        base_family=base_family,
        header_family=header_family,
        base_line_size=base_line_size,
        base_rect_size=base_rect_size,
        ink=ink,
        paper=paper,
        accent=accent,
    )
    override = theme(
        complete=True,
        panel_background=element_rect(fill=paper, colour=None),
        panel_border=element_rect(
            colour=_col_mix(ink, paper, 0.702),
            linewidth=rel(1),
        ),
        panel_grid=element_line(colour=_col_mix(ink, paper, 0.871)),
        panel_grid_major=element_line(linewidth=rel(0.5)),
        panel_grid_minor=element_line(linewidth=rel(0.25)),
        axis_ticks=element_line(
            colour=_col_mix(ink, paper, 0.702),
            linewidth=rel(0.5),
        ),
        strip_background=element_rect(
            fill=_col_mix(ink, paper, 0.702),
            colour=None,
        ),
        strip_text=element_text(
            colour=paper,
            size=rel(0.8),
            margin=margin_auto(0.8 * half_line),
        ),
    )
    return theme_replace_op(base, override)


# ---------------------------------------------------------------------------
# theme_dark
# ---------------------------------------------------------------------------

def theme_dark(
    base_size: float = 11,
    base_family: str = "",
    header_family: Optional[str] = None,
    base_line_size: Optional[float] = None,
    base_rect_size: Optional[float] = None,
    ink: str = "black",
    paper: str = "white",
    accent: str = "#3366FF",
) -> Theme:
    """The dark cousin of ``theme_light`` with a dark panel background.

    Parameters
    ----------
    base_size : float
        Base font size in points.
    base_family : str
        Base font family.
    header_family : str or None
        Header font family.
    base_line_size : float or None
        Base line size.
    base_rect_size : float or None
        Base rect size.
    ink : str
        Foreground colour.
    paper : str
        Background colour.
    accent : str
        Accent colour.

    Returns
    -------
    Theme
    """
    half_line = base_size / 2

    base = theme_grey(
        base_size=base_size,
        base_family=base_family,
        header_family=header_family,
        base_line_size=base_line_size,
        base_rect_size=base_rect_size,
        ink=ink,
        paper=paper,
        accent=accent,
    )
    override = theme(
        complete=True,
        panel_background=element_rect(
            fill=_col_mix(ink, paper, 0.499),
            colour=None,
        ),
        panel_grid=element_line(colour=_col_mix(ink, paper, 0.42)),
        panel_grid_major=element_line(linewidth=rel(0.5)),
        panel_grid_minor=element_line(linewidth=rel(0.25)),
        axis_ticks=element_line(
            colour=_col_mix(ink, paper, 0.2),
            linewidth=rel(0.5),
        ),
        strip_background=element_rect(
            fill=_col_mix(ink, paper, 0.15),
            colour=None,
        ),
        strip_text=element_text(
            colour=_col_mix(ink, paper, 0.899),
            size=rel(0.8),
            margin=margin_auto(0.8 * half_line),
        ),
    )
    return theme_replace_op(base, override)


# ---------------------------------------------------------------------------
# theme_minimal
# ---------------------------------------------------------------------------

def theme_minimal(
    base_size: float = 11,
    base_family: str = "",
    header_family: Optional[str] = None,
    base_line_size: Optional[float] = None,
    base_rect_size: Optional[float] = None,
    ink: str = "black",
    paper: str = "white",
    accent: str = "#3366FF",
) -> Theme:
    """A minimalistic theme with no background annotations.

    Parameters
    ----------
    base_size : float
        Base font size in points.
    base_family : str
        Base font family.
    header_family : str or None
        Header font family.
    base_line_size : float or None
        Base line size.
    base_rect_size : float or None
        Base rect size.
    ink : str
        Foreground colour.
    paper : str
        Background colour.
    accent : str
        Accent colour.

    Returns
    -------
    Theme
    """
    base = theme_bw(
        base_size=base_size,
        base_family=base_family,
        header_family=header_family,
        base_line_size=base_line_size,
        base_rect_size=base_rect_size,
        ink=ink,
        paper=paper,
        accent=accent,
    )
    override = theme(
        complete=True,
        axis_ticks=element_blank(),
        axis_text_x_bottom=element_text(margin=margin(t=0.45 * base_size)),
        axis_text_x_top=element_text(margin=margin(b=0.45 * base_size)),
        axis_text_y_left=element_text(margin=margin(r=0.45 * base_size)),
        axis_text_y_right=element_text(margin=margin(l=0.45 * base_size)),
        legend_background=element_blank(),
        legend_key=element_blank(),
        panel_background=element_blank(),
        panel_border=element_blank(),
        strip_background=element_blank(),
        plot_background=element_rect(fill=paper, colour=None),
    )
    return theme_replace_op(base, override)


# ---------------------------------------------------------------------------
# theme_classic
# ---------------------------------------------------------------------------

def theme_classic(
    base_size: float = 11,
    base_family: str = "",
    header_family: Optional[str] = None,
    base_line_size: Optional[float] = None,
    base_rect_size: Optional[float] = None,
    ink: str = "black",
    paper: str = "white",
    accent: str = "#3366FF",
) -> Theme:
    """A classic theme with axis lines and no gridlines.

    Parameters
    ----------
    base_size : float
        Base font size in points.
    base_family : str
        Base font family.
    header_family : str or None
        Header font family.
    base_line_size : float or None
        Base line size.
    base_rect_size : float or None
        Base rect size.
    ink : str
        Foreground colour.
    paper : str
        Background colour.
    accent : str
        Accent colour.

    Returns
    -------
    Theme
    """
    base = theme_bw(
        base_size=base_size,
        base_family=base_family,
        header_family=header_family,
        base_line_size=base_line_size,
        base_rect_size=base_rect_size,
        ink=ink,
        paper=paper,
        accent=accent,
    )
    override = theme(
        complete=True,
        panel_border=element_blank(),
        panel_grid=element_blank(),
        axis_text=element_text(size=rel(0.8)),
        axis_line=element_line(lineend="square"),
        axis_ticks=element_line(),
        strip_background=element_rect(linewidth=rel(2)),
    )
    return theme_replace_op(base, override)


# ---------------------------------------------------------------------------
# theme_void
# ---------------------------------------------------------------------------

def theme_void(
    base_size: float = 11,
    base_family: str = "",
    header_family: Optional[str] = None,
    base_line_size: Optional[float] = None,
    base_rect_size: Optional[float] = None,
    ink: str = "black",
    paper: Optional[str] = None,
    accent: str = "#3366FF",
) -> Theme:
    """A completely empty theme.

    Parameters
    ----------
    base_size : float
        Base font size in points.
    base_family : str
        Base font family.
    header_family : str or None
        Header font family.
    base_line_size : float or None
        Base line size.
    base_rect_size : float or None
        Base rect size.
    ink : str
        Foreground colour.
    paper : str or None
        Background colour (default is transparent).
    accent : str
        Accent colour.

    Returns
    -------
    Theme
    """
    if base_line_size is None:
        base_line_size = base_size / 22
    if base_rect_size is None:
        base_rect_size = base_size / 22
    if paper is None:
        # Transparent paper by default
        paper = "#00000000"

    half_line = base_size / 2

    t = theme(
        complete=True,
        line=element_blank(),
        rect=element_rect(
            fill=paper,
            colour=None,
            linewidth=0,
            linetype=1,
            linejoin="round",
        ),
        polygon=element_blank(),
        point=element_blank(),
        text=element_text(
            family=base_family,
            face="plain",
            colour=ink,
            size=base_size,
            lineheight=0.9,
            hjust=0.5,
            vjust=0.5,
            angle=0,
            margin=margin(),
            debug=False,
        ),
        title=element_text(family=header_family),
        spacing=Unit(half_line, "pt"),
        margins=margin_auto(half_line),
        geom=element_geom(
            ink=ink,
            paper=paper,
            accent=accent,
            linewidth=base_line_size,
            borderwidth=base_line_size,
            linetype=1,
            bordertype=1,
            family=base_family,
            fontsize=base_size,
            pointsize=(base_size / 11) * 1.5,
            pointshape=19,
        ),
        axis_text=element_blank(),
        axis_title=element_blank(),
        axis_ticks_length=rel(0),
        axis_ticks_length_x=None,
        axis_ticks_length_x_top=None,
        axis_ticks_length_x_bottom=None,
        axis_ticks_length_y=None,
        axis_ticks_length_y_left=None,
        axis_ticks_length_y_right=None,
        axis_minor_ticks_length=None,
        legend_box=None,
        legend_key_size=Unit(1.2, "lines"),
        legend_position="right",
        legend_text=element_text(size=rel(0.8)),
        legend_title=element_text(hjust=0),
        legend_key_spacing=rel(1),
        legend_margin=margin_auto(0),
        legend_box_margin=margin_auto(0),
        legend_box_spacing=Unit(0.2, "cm"),
        legend_ticks_length=rel(0.2),
        legend_background=element_blank(),
        legend_box_background=element_blank(),
        strip_clip="on",
        strip_text=element_text(size=rel(0.8)),
        strip_switch_pad_grid=rel(0.5),
        strip_switch_pad_wrap=rel(0.5),
        strip_background=element_blank(),
        panel_ontop=False,
        panel_spacing=None,
        panel_background=element_blank(),
        panel_border=element_blank(),
        plot_margin=margin_auto(0),
        plot_title=element_text(
            size=rel(1.2),
            hjust=0,
            vjust=1,
            margin=margin(t=half_line),
        ),
        plot_title_position="panel",
        plot_subtitle=element_text(
            hjust=0,
            vjust=1,
            margin=margin(t=half_line),
        ),
        plot_caption=element_text(
            size=rel(0.8),
            hjust=1,
            vjust=1,
            margin=margin(t=half_line),
        ),
        plot_caption_position="panel",
        plot_tag=element_text(
            size=rel(1.2),
            hjust=0.5,
            vjust=0.5,
        ),
        plot_tag_position="topleft",
        plot_background=element_rect(),
    )

    base = _theme_all_null()
    return theme_replace_op(base, t)


# ---------------------------------------------------------------------------
# theme_test
# ---------------------------------------------------------------------------

def theme_test(
    base_size: float = 11,
    base_family: str = "",
    header_family: Optional[str] = None,
    base_line_size: Optional[float] = None,
    base_rect_size: Optional[float] = None,
    ink: str = "black",
    paper: str = "white",
    accent: str = "#3366FF",
) -> Theme:
    """A theme for visual unit tests.

    Parameters
    ----------
    base_size : float
        Base font size in points.
    base_family : str
        Base font family.
    header_family : str or None
        Header font family.
    base_line_size : float or None
        Base line size.
    base_rect_size : float or None
        Base rect size.
    ink : str
        Foreground colour.
    paper : str
        Background colour.
    accent : str
        Accent colour.

    Returns
    -------
    Theme
    """
    if base_line_size is None:
        base_line_size = base_size / 22
    if base_rect_size is None:
        base_rect_size = base_size / 22

    half_line = base_size / 2

    t = theme(
        complete=True,
        line=element_line(
            colour=ink,
            linewidth=base_line_size,
            linetype=1,
            lineend="butt",
            linejoin="round",
        ),
        rect=element_rect(
            fill=paper,
            colour=ink,
            linewidth=base_rect_size,
            linetype=1,
            linejoin="round",
        ),
        text=element_text(
            family=base_family,
            face="plain",
            colour=ink,
            size=base_size,
            lineheight=0.9,
            hjust=0.5,
            vjust=0.5,
            angle=0,
            margin=margin(),
            debug=False,
        ),
        point=element_point(
            colour=ink,
            shape=19,
            fill=paper,
            size=(base_size / 11) * 1.5,
            stroke=base_line_size,
        ),
        polygon=element_polygon(
            fill=paper,
            colour=ink,
            linewidth=base_rect_size,
            linetype=1,
            linejoin="round",
        ),
        title=element_text(family=header_family),
        spacing=Unit(half_line, "pt"),
        margins=margin_auto(half_line),
        geom=element_geom(
            ink=ink,
            paper=paper,
            accent=accent,
            linewidth=base_line_size,
            borderwidth=base_line_size,
            linetype=1,
            family=base_family,
            fontsize=base_size,
            pointsize=(base_size / 11) * 1.5,
            pointshape=19,
        ),

        # Axis
        axis_line=element_blank(),
        axis_line_x=None,
        axis_line_y=None,
        axis_text=element_text(
            size=rel(0.8),
            colour=_col_mix(ink, paper, 0.302),
        ),
        axis_text_x=element_text(
            margin=margin(t=0.8 * half_line / 2),
            vjust=1,
        ),
        axis_text_x_top=element_text(
            margin=margin(b=0.8 * half_line / 2),
            vjust=0,
        ),
        axis_text_y=element_text(
            margin=margin(r=0.8 * half_line / 2),
            hjust=1,
        ),
        axis_text_y_right=element_text(
            margin=margin(l=0.8 * half_line / 2),
            hjust=0,
        ),
        axis_ticks=element_line(colour=_col_mix(ink, paper, 0.2)),
        axis_ticks_length=rel(0.5),
        axis_ticks_length_x=None,
        axis_ticks_length_x_top=None,
        axis_ticks_length_x_bottom=None,
        axis_ticks_length_y=None,
        axis_ticks_length_y_left=None,
        axis_ticks_length_y_right=None,
        axis_minor_ticks_length=rel(0.75),
        axis_title_x=element_text(
            margin=margin(t=half_line / 2),
            vjust=1,
        ),
        axis_title_x_top=element_text(
            margin=margin(b=half_line / 2),
            vjust=0,
        ),
        axis_title_y=element_text(
            angle=90,
            margin=margin(r=half_line / 2),
            vjust=1,
        ),
        axis_title_y_right=element_text(
            angle=-90,
            margin=margin(l=half_line / 2),
            vjust=1,
        ),

        # Legend
        legend_background=element_rect(colour=None),
        legend_spacing=rel(2),
        legend_spacing_x=None,
        legend_spacing_y=None,
        legend_margin=margin_auto(0, unit="cm"),
        legend_key=None,
        legend_key_size=Unit(1.2, "lines"),
        legend_key_height=None,
        legend_key_width=None,
        legend_key_spacing=None,
        legend_key_spacing_x=None,
        legend_key_spacing_y=None,
        legend_text=element_text(size=rel(0.8)),
        legend_title=element_text(hjust=0),
        legend_ticks_length=rel(0.2),
        legend_position="right",
        legend_direction=None,
        legend_justification="center",
        legend_box=None,
        legend_box_margin=margin_auto(0, unit="cm"),
        legend_box_background=element_blank(),
        legend_box_spacing=rel(2),

        # Panel
        panel_background=element_rect(fill=paper, colour=None),
        panel_border=element_rect(colour=_col_mix(ink, paper, 0.2)),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_spacing=None,
        panel_spacing_x=None,
        panel_spacing_y=None,
        panel_ontop=False,

        # Strip
        strip_background=element_rect(
            fill=_col_mix(ink, paper, 0.85),
            colour=_col_mix(ink, paper, 0.2),
        ),
        strip_clip="on",
        strip_text=element_text(
            colour=_col_mix(ink, paper, 0.1),
            size=rel(0.8),
            margin=margin_auto(0.8 * half_line),
        ),
        strip_text_x=None,
        strip_text_y=element_text(angle=-90),
        strip_text_y_left=element_text(angle=90),
        strip_placement="inside",
        strip_placement_x=None,
        strip_placement_y=None,
        strip_switch_pad_grid=rel(0.5),
        strip_switch_pad_wrap=rel(0.5),

        # Plot
        plot_background=element_rect(colour=paper),
        plot_title=element_text(
            size=rel(1.2),
            hjust=0,
            vjust=1,
            margin=margin(b=half_line),
        ),
        plot_title_position="panel",
        plot_subtitle=element_text(
            hjust=0,
            vjust=1,
            margin=margin(b=half_line),
        ),
        plot_caption=element_text(
            size=rel(0.8),
            hjust=1,
            vjust=1,
            margin=margin(t=half_line),
        ),
        plot_caption_position="panel",
        plot_tag=element_text(
            size=rel(1.2),
            hjust=0.5,
            vjust=0.5,
        ),
        plot_tag_position="topleft",
        plot_margin=None,
    )

    base = _theme_all_null()
    return theme_replace_op(base, t)


# ---------------------------------------------------------------------------
# Sub-theme helpers
# ---------------------------------------------------------------------------

def theme_sub_axis(**kwargs: Any) -> Theme:
    """Create a partial theme modifying axis elements.

    Parameters
    ----------
    **kwargs
        Axis-related theme element overrides (e.g.
        ``axis_text=element_text(size=8)``).

    Returns
    -------
    Theme
    """
    return theme(**kwargs)


def theme_sub_axis_x(**kwargs: Any) -> Theme:
    """Create a partial theme modifying x-axis elements.

    Parameters
    ----------
    **kwargs
        X-axis theme element overrides.

    Returns
    -------
    Theme
    """
    return theme(**kwargs)


def theme_sub_axis_y(**kwargs: Any) -> Theme:
    """Create a partial theme modifying y-axis elements.

    Parameters
    ----------
    **kwargs
        Y-axis theme element overrides.

    Returns
    -------
    Theme
    """
    return theme(**kwargs)


def theme_sub_axis_top(**kwargs: Any) -> Theme:
    """Create a partial theme modifying top-axis elements.

    Parameters
    ----------
    **kwargs
        Top-axis theme element overrides.

    Returns
    -------
    Theme
    """
    return theme(**kwargs)


def theme_sub_axis_bottom(**kwargs: Any) -> Theme:
    """Create a partial theme modifying bottom-axis elements.

    Parameters
    ----------
    **kwargs
        Bottom-axis theme element overrides.

    Returns
    -------
    Theme
    """
    return theme(**kwargs)


def theme_sub_axis_left(**kwargs: Any) -> Theme:
    """Create a partial theme modifying left-axis elements.

    Parameters
    ----------
    **kwargs
        Left-axis theme element overrides.

    Returns
    -------
    Theme
    """
    return theme(**kwargs)


def theme_sub_axis_right(**kwargs: Any) -> Theme:
    """Create a partial theme modifying right-axis elements.

    Parameters
    ----------
    **kwargs
        Right-axis theme element overrides.

    Returns
    -------
    Theme
    """
    return theme(**kwargs)


def theme_sub_legend(**kwargs: Any) -> Theme:
    """Create a partial theme modifying legend elements.

    Parameters
    ----------
    **kwargs
        Legend-related theme element overrides.

    Returns
    -------
    Theme
    """
    return theme(**kwargs)


def theme_sub_panel(**kwargs: Any) -> Theme:
    """Create a partial theme modifying panel elements.

    Parameters
    ----------
    **kwargs
        Panel-related theme element overrides.

    Returns
    -------
    Theme
    """
    return theme(**kwargs)


def theme_sub_plot(**kwargs: Any) -> Theme:
    """Create a partial theme modifying plot elements.

    Parameters
    ----------
    **kwargs
        Plot-related theme element overrides.

    Returns
    -------
    Theme
    """
    return theme(**kwargs)


def theme_sub_strip(**kwargs: Any) -> Theme:
    """Create a partial theme modifying strip elements.

    Parameters
    ----------
    **kwargs
        Strip-related theme element overrides.

    Returns
    -------
    Theme
    """
    return theme(**kwargs)
