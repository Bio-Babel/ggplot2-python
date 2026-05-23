"""Tests for the gtable-based legend, colourbar, coloursteps, and labeller systems.

Covers all features implemented in the legend/guide refactor:
- Legend rendering as independent Gtable (_guide_legend.py)
- Colourbar rendering for continuous scales (_guide_colourbar.py)
- Coloursteps rendering for binned scales (_guide_colourbar.py)
- Labeller functions (labeller.py)
- GuideAxis pipeline (_guide_axis.py)
- Plot split (plot.py / plot_render.py)
- Scale aliases (scale_linetype_ordinal)
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mpg():
    from ggplot2_py.datasets import mpg
    return mpg


@pytest.fixture
def random_df():
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "x": rng.randn(100),
        "y": rng.randn(100),
        "z": rng.randn(100),
    })


@pytest.fixture
def tmp_png(tmp_path):
    return str(tmp_path / "test.png")


# ---------------------------------------------------------------------------
# Legend tests (discrete scales)
# ---------------------------------------------------------------------------

class TestGuideLegend:
    """Test the gtable-based legend rendering."""

    def test_colour_shape_same_var(self, mpg, tmp_png):
        from ggplot2_py import ggplot, aes, geom_point
        from ggplot2_py.save import ggsave
        p = ggplot(mpg, aes('cty', 'hwy', colour='drv', shape='drv')) + geom_point(size=2)
        ggsave(tmp_png, p, width=7, height=5, dpi=100)
        assert os.path.exists(tmp_png)

    def test_shape_only(self, mpg, tmp_png):
        from ggplot2_py import ggplot, aes, geom_point
        from ggplot2_py.save import ggsave
        p = ggplot(mpg, aes('cty', 'hwy', shape='drv')) + geom_point(size=2)
        ggsave(tmp_png, p, width=7, height=5, dpi=100)
        assert os.path.exists(tmp_png)

    def test_colour_only(self, mpg, tmp_png):
        from ggplot2_py import ggplot, aes, geom_point
        from ggplot2_py.save import ggsave
        p = ggplot(mpg, aes('cty', 'hwy', colour='drv')) + geom_point(size=2)
        ggsave(tmp_png, p, width=7, height=5, dpi=100)
        assert os.path.exists(tmp_png)

    def test_colour_shape_different_vars(self, mpg, tmp_png):
        from ggplot2_py import ggplot, aes, geom_point
        from ggplot2_py.save import ggsave
        p = ggplot(mpg, aes('cty', 'hwy', colour='drv', shape='fl')) + geom_point(size=2)
        ggsave(tmp_png, p, width=7, height=5, dpi=100)
        assert os.path.exists(tmp_png)

    def test_fill_bar(self, mpg, tmp_png):
        from ggplot2_py import ggplot, aes, geom_bar
        from ggplot2_py.save import ggsave
        p = ggplot(mpg, aes(x='drv', fill='drv')) + geom_bar()
        ggsave(tmp_png, p, width=7, height=5, dpi=100)
        assert os.path.exists(tmp_png)

    def test_colour_brewer(self, mpg, tmp_png):
        from ggplot2_py import ggplot, aes, geom_point, scale_colour_brewer
        from ggplot2_py.save import ggsave
        p = ggplot(mpg, aes('cty', 'hwy', colour='drv')) + geom_point(size=2) + scale_colour_brewer(palette='Set1')
        ggsave(tmp_png, p, width=7, height=5, dpi=100)
        assert os.path.exists(tmp_png)


# ---------------------------------------------------------------------------
# Colourbar tests (continuous scales)
# ---------------------------------------------------------------------------

class TestGuideColourbar:
    """Test the colourbar rendering for continuous colour scales."""

    def test_continuous_colour(self, mpg, tmp_png):
        from ggplot2_py import ggplot, aes, geom_point
        from ggplot2_py.save import ggsave
        p = ggplot(mpg, aes('cty', 'hwy', colour='displ')) + geom_point()
        ggsave(tmp_png, p, width=7, height=5, dpi=100)
        assert os.path.exists(tmp_png)

    def test_gradient2(self, random_df, tmp_png):
        from ggplot2_py import ggplot, aes, geom_point
        from ggplot2_py.scales import scale_colour_gradient2
        from ggplot2_py.save import ggsave
        p = ggplot(random_df, aes('x', 'y', colour='z')) + geom_point() + scale_colour_gradient2()
        ggsave(tmp_png, p, width=7, height=5, dpi=100)
        assert os.path.exists(tmp_png)


# ---------------------------------------------------------------------------
# Coloursteps tests (binned colour scales)
# ---------------------------------------------------------------------------

class TestGuideColoursteps:
    """Test the coloursteps rendering for binned colour scales."""

    def test_colour_steps(self, mpg, tmp_png):
        from ggplot2_py import ggplot, aes, geom_point
        from ggplot2_py.scales import scale_colour_steps
        from ggplot2_py.save import ggsave
        p = ggplot(mpg, aes('cty', 'hwy', colour='displ')) + geom_point() + scale_colour_steps()
        ggsave(tmp_png, p, width=7, height=5, dpi=100)
        assert os.path.exists(tmp_png)

    def test_colour_steps2(self, random_df, tmp_png):
        from ggplot2_py import ggplot, aes, geom_point
        from ggplot2_py.scales import scale_colour_steps2
        from ggplot2_py.save import ggsave
        p = ggplot(random_df, aes('x', 'y', colour='z')) + geom_point() + scale_colour_steps2()
        ggsave(tmp_png, p, width=7, height=5, dpi=100)
        assert os.path.exists(tmp_png)


# ---------------------------------------------------------------------------
# Labeller tests
# ---------------------------------------------------------------------------

class TestLabeller:
    """Test facet labeller functions."""

    def test_label_value(self):
        from ggplot2_py.labeller import label_value
        result = label_value({"drv": ["4", "f", "r"]})
        assert result == ["4", "f", "r"]

    def test_label_both(self):
        from ggplot2_py.labeller import label_both
        result = label_both({"drv": ["4", "f", "r"]})
        assert result == ["drv: 4", "drv: f", "drv: r"]

    def test_label_context_single(self):
        from ggplot2_py.labeller import label_context
        result = label_context({"drv": ["4", "f"]})
        assert result == ["4", "f"]  # single var → label_value

    def test_label_context_multi(self):
        from ggplot2_py.labeller import label_context
        result = label_context({"drv": ["4", "f"], "cyl": ["4", "6"]})
        # multi var → label_both
        assert "drv: 4" in result[0]
        assert "cyl: 4" in result[0]

    def test_label_wrap_gen(self):
        from ggplot2_py.labeller import label_wrap_gen
        labeller = label_wrap_gen(width=5)
        result = labeller({"class": ["compact car"]})
        assert "\n" in result[0]

    def test_as_labeller_string(self):
        from ggplot2_py.labeller import as_labeller, label_both
        fn = as_labeller("label_both")
        assert fn is label_both

    def test_as_labeller_dict(self):
        from ggplot2_py.labeller import as_labeller
        fn = as_labeller({"4": "Four", "f": "Front"})
        result = fn({"drv": ["4", "f", "r"]})
        assert result == ["Four", "Front", "r"]

    def test_facet_wrap_with_labeller(self, mpg, tmp_png):
        from ggplot2_py import ggplot, aes, geom_point, facet_wrap
        from ggplot2_py.labeller import label_both
        from ggplot2_py.save import ggsave
        p = ggplot(mpg, aes('cty', 'hwy')) + geom_point() + facet_wrap('drv', labeller=label_both)
        ggsave(tmp_png, p, width=9, height=4, dpi=100)
        assert os.path.exists(tmp_png)


# ---------------------------------------------------------------------------
# GuideAxis tests
# ---------------------------------------------------------------------------

class TestGuideAxis:
    """Test axis rendering via the GuideAxis pipeline."""

    def test_basic_axes(self, mpg, tmp_png):
        from ggplot2_py import ggplot, aes, geom_point
        from ggplot2_py.save import ggsave
        p = ggplot(mpg, aes('cty', 'hwy')) + geom_point()
        ggsave(tmp_png, p, width=7, height=5, dpi=100)
        assert os.path.exists(tmp_png)

    def test_draw_axis_function(self):
        from ggplot2_py._guide_axis import draw_axis
        import numpy as np
        breaks = np.array([0.2, 0.4, 0.6, 0.8])
        labels = ["0.2", "0.4", "0.6", "0.8"]
        grob = draw_axis(breaks, labels, "bottom", theme=None)
        assert grob is not None


# ---------------------------------------------------------------------------
# Plot split tests
# ---------------------------------------------------------------------------

class TestPlotSplit:
    """Test that plot.py / plot_render.py split works correctly."""

    def test_ggplot_gtable_importable_from_plot(self):
        from ggplot2_py.plot import ggplot_gtable
        assert callable(ggplot_gtable)

    def test_ggplot_gtable_importable_from_plot_render(self):
        from ggplot2_py.plot_render import ggplot_gtable
        assert callable(ggplot_gtable)

    def test_same_function(self):
        from ggplot2_py.plot import ggplot_gtable as f1
        from ggplot2_py.plot_render import ggplot_gtable as f2
        assert f1 is f2

    def test_find_panel(self):
        from ggplot2_py.plot import find_panel
        from ggplot2_py.plot_render import find_panel as fp2
        assert find_panel is fp2

    def test_full_pipeline(self, mpg, tmp_png):
        from ggplot2_py import ggplot, aes, geom_point
        from ggplot2_py.save import ggsave
        p = ggplot(mpg, aes('cty', 'hwy')) + geom_point()
        ggsave(tmp_png, p, width=7, height=5, dpi=100)
        assert os.path.exists(tmp_png)


# ---------------------------------------------------------------------------
# Scale alias tests
# ---------------------------------------------------------------------------

class TestScaleAliases:
    """Test that scale aliases are importable and functional."""

    def test_scale_shape_discrete(self):
        from ggplot2_py.scales import scale_shape_discrete, scale_shape
        assert scale_shape_discrete is scale_shape

    def test_scale_linetype_discrete(self):
        from ggplot2_py.scales import scale_linetype_discrete, scale_linetype
        assert scale_linetype_discrete is scale_linetype

    def test_scale_linetype_ordinal(self):
        from ggplot2_py.scales import scale_linetype_ordinal, scale_linetype
        assert scale_linetype_ordinal is scale_linetype

    def test_scale_shape_ordinal(self):
        from ggplot2_py.scales import scale_shape_ordinal
        assert callable(scale_shape_ordinal)
