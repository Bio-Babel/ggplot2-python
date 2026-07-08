"""Tests for ggplot2_py.facet — faceting system."""

import pytest
import pandas as pd
from ggplot2_py import (
    FacetWrap,
    FacetGrid,
    FacetNull,
    facet_wrap,
    facet_grid,
    facet_null,
    is_facet,
)


class TestFacetWrap:
    """Test facet_wrap."""

    def test_creates_facet_wrap(self):
        f = facet_wrap("class")
        assert isinstance(f, FacetWrap)

    def test_is_facet(self):
        f = facet_wrap("class")
        assert is_facet(f) is True


class TestFacetGrid:
    """Test facet_grid."""

    def test_creates_facet_grid(self):
        f = facet_grid(rows="drv", cols="cyl")
        assert isinstance(f, FacetGrid)

    def test_is_facet(self):
        f = facet_grid(rows="drv")
        assert is_facet(f) is True


class TestFacetNull:
    """Test facet_null."""

    def test_creates_facet_null(self):
        f = facet_null()
        assert isinstance(f, FacetNull)

    def test_is_facet(self):
        f = facet_null()
        assert is_facet(f) is True


class TestIsFacet:
    """Test is_facet predicate."""

    def test_true_for_wrap(self):
        assert is_facet(facet_wrap("x")) is True

    def test_true_for_null(self):
        assert is_facet(facet_null()) is True

    def test_false_for_string(self):
        assert is_facet("wrap") is False

    def test_false_for_none(self):
        assert is_facet(None) is False


class TestFacetGridLayoutParity:
    """Layout encoding validated against gold-standard R runs
    (facet-grid-.R compute_layout + compat-plyr.R id())."""

    def _df(self):
        import numpy as np
        rng = np.random.default_rng(1)
        return pd.DataFrame({
            "x": rng.normal(size=60), "y": rng.normal(size=60),
            "g": np.tile([2, 10, 100], 20),
            "h": pd.Categorical(np.repeat(["low", "mid", "high"], 20),
                                categories=["low", "mid", "high"]),
        })

    def _layout(self, facet, df=None):
        from ggplot2_py.plot import ggplot_build
        from ggplot2_py import ggplot, aes, geom_point
        p = ggplot(df if df is not None else self._df(), aes("x", "y")) + geom_point() + facet
        return ggplot_build(p).layout.layout

    def test_numeric_var_orders_by_value(self):
        # R: g = 2/10/100 → ROW 1/2/3 (value order, not "10"<"2")
        ld = self._layout(facet_grid(rows="g"))
        assert list(ld.sort_values("ROW")["g"]) == [2, 10, 100]

    def test_factor_var_orders_by_levels(self):
        ld = self._layout(facet_grid(cols="h"))
        assert list(ld.sort_values("COL")["h"]) == ["low", "mid", "high"]

    def test_as_table_false_reverses_rows(self):
        ld = self._layout(facet_grid(rows="g", as_table=False))
        assert list(ld.sort_values("ROW")["g"]) == [100, 10, 2]

    def test_margins_layout(self):
        # R: h ~ g with margins=TRUE → 4x4 grid, "(all)" panels last
        ld = self._layout(facet_grid("h ~ g", margins=True))
        assert len(ld) == 16
        assert int(ld["ROW"].max()) == 4 and int(ld["COL"].max()) == 4
        corner = ld[(ld["ROW"] == 4) & (ld["COL"] == 4)]
        assert str(corner["g"].iloc[0]) == "(all)" and str(corner["h"].iloc[0]) == "(all)"

    def test_margins_data_replication(self):
        # R: 60 rows × 4 (original + row margin + col margin + both)
        from ggplot2_py.plot import ggplot_build
        from ggplot2_py import ggplot, aes, geom_point
        p = ggplot(self._df(), aes("x", "y")) + geom_point() + facet_grid("h ~ g", margins=True)
        built = ggplot_build(p)
        assert len(built.data[0]) == 240
        assert built.data[0]["PANEL"].value_counts()[16] == 60  # (all)/(all)

    def test_annotation_layer_in_every_panel(self):
        # R facet-.R:1494-1510: layers without facet vars replicate
        from ggplot2_py.plot import ggplot_build
        from ggplot2_py import ggplot, aes, geom_point
        ann = pd.DataFrame({"x": [0.0], "y": [0.0]})
        p = (ggplot(self._df(), aes("x", "y")) + geom_point()
             + geom_point(data=ann, colour="red")
             + facet_grid(cols="h"))
        built = ggplot_build(p)
        counts = built.data[1]["PANEL"].value_counts()
        assert sorted(counts.index.astype(int)) == [1, 2, 3]
        assert (counts == 1).all()

    def test_duplicated_rows_cols_aborts(self):
        from ggplot2_py.plot import ggplot_build
        from ggplot2_py import ggplot, aes, geom_point
        p = ggplot(self._df(), aes("x", "y")) + geom_point() + facet_grid(rows="h", cols="h")
        with pytest.raises(Exception):
            ggplot_build(p)
