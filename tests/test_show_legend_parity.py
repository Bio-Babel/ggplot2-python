"""Parity tests for layer ``show_legend`` legend suppression.

Gold standard: R ggplot2 4.0.2.  A layer with ``show.legend = FALSE`` does
NOT contribute its mapped aesthetics to any guide; if an aesthetic ends up
with no contributing layer its guide is dropped (no legend).  ``NA`` (Python
``None``) and ``TRUE`` contribute / force inclusion.

R behaviour these mirror (verified both-side with Rscript ggplot2 4.0.2)::

    ggplot(d, aes(f1, fill=class)) + geom_bar()                  -> legend
    ggplot(d, aes(f1, fill=class)) + geom_bar(show.legend=FALSE) -> NO legend
    ggplot(d, aes(f1, fill=class)) + geom_bar(show.legend=TRUE)  -> legend

R source: ``guide-legend.R:219-231`` (process_layers/get_layer_key),
``guides-.R:871-912`` (matched_aes / include_layer_in_guide).
"""

import pandas as pd
import pytest

from ggplot2_py import (
    ggplot,
    aes,
    geom_bar,
    geom_point,
    geom_line,
    ggplot_build,
    ggplot_gtable,
)


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "f1": ["a", "b", "c", "a", "b", "c"],
            "class": ["x", "y", "z", "x", "y", "z"],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def _legend_present(plot) -> bool:
    """True iff the rendered gtable holds a non-empty guide-box cell.

    R always emits all five ``guide-box-*`` cells but fills them with a
    zeroGrob when no legend occupies the slot; a *present* legend is a
    populated gtable (non-zero width list).
    """
    table = ggplot_gtable(ggplot_build(plot))
    names = list(table.layout.get("name", []))
    for name, grob in zip(names, table.grobs):
        if "guide-box" in str(name):
            widths = getattr(grob, "_widths", None)
            if widths is not None and len(widths) > 0:
                return True
    return False


def _guide_box_cell_count(plot) -> int:
    table = ggplot_gtable(ggplot_build(plot))
    names = [str(n) for n in table.layout.get("name", [])]
    return sum(1 for n in names if "guide-box" in n)


# ---------------------------------------------------------------------------
# Single-layer mapped aesthetic
# ---------------------------------------------------------------------------

class TestSingleLayerShowLegend:
    def test_default_shows_legend(self, df):
        # R: geom_bar() with mapped fill -> legend present.
        p = ggplot(df, aes("f1", fill="class")) + geom_bar()
        assert _legend_present(p) is True

    def test_false_suppresses_legend(self, df):
        # R: geom_bar(show.legend=FALSE) -> NO legend. (The bug.)
        p = ggplot(df, aes("f1", fill="class")) + geom_bar(show_legend=False)
        assert _legend_present(p) is False

    def test_true_forces_legend(self, df):
        # R: geom_bar(show.legend=TRUE) -> legend present.
        p = ggplot(df, aes("f1", fill="class")) + geom_bar(show_legend=True)
        assert _legend_present(p) is True

    def test_none_default_shows_legend(self, df):
        # show_legend=None is the NA default -> include if mapped.
        p = ggplot(df, aes("f1", fill="class")) + geom_bar(show_legend=None)
        assert _legend_present(p) is True

    def test_point_colour_false_suppresses(self, df):
        p = ggplot(df, aes("v", "v", colour="class")) + geom_point(
            show_legend=False
        )
        assert _legend_present(p) is False

    def test_point_colour_default_shows(self, df):
        p = ggplot(df, aes("v", "v", colour="class")) + geom_point()
        assert _legend_present(p) is True


# ---------------------------------------------------------------------------
# Multi-layer: a hidden layer is excluded but others still contribute
# ---------------------------------------------------------------------------

class TestMultiLayerShowLegend:
    def test_one_hidden_one_shown_keeps_legend(self, df):
        # R: point(show.legend=FALSE)+line(show.legend=NA) -> legend present
        # (the line still contributes the colour guide).
        p = (
            ggplot(df, aes("v", "v", colour="class"))
            + geom_point(show_legend=False)
            + geom_line(show_legend=None)
        )
        assert _legend_present(p) is True

    def test_all_hidden_drops_legend(self, df):
        # R: both layers show.legend=FALSE -> NO legend.
        p = (
            ggplot(df, aes("v", "v", colour="class"))
            + geom_point(show_legend=False)
            + geom_line(show_legend=False)
        )
        assert _legend_present(p) is False


# ---------------------------------------------------------------------------
# Five guide-box cells are always emitted (R parity, plot-render.R:70-134)
# ---------------------------------------------------------------------------

class TestGuideBoxSlots:
    def test_five_slots_when_suppressed(self, df):
        p = ggplot(df, aes("f1", fill="class")) + geom_bar(show_legend=False)
        assert _guide_box_cell_count(p) == 5

    def test_five_slots_when_present(self, df):
        p = ggplot(df, aes("f1", fill="class")) + geom_bar()
        assert _guide_box_cell_count(p) == 5
