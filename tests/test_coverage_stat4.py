"""Targeted coverage tests for ggplot2_py.stat – round 4."""

import pytest
import numpy as np
import pandas as pd

from ggplot2_py.aes import aes
from ggplot2_py.plot import ggplot, ggplot_build


# ===========================================================================
# StatSummary with string function names
# ===========================================================================

class TestStatSummaryFunctions:
    def test_summary_mean_se(self):
        """Cover lines 844-859: string fun_data lookup."""
        from ggplot2_py.stat import stat_summary
        from ggplot2_py.geom import geom_point
        df = pd.DataFrame({"x": [1, 1, 2, 2, 3, 3], "y": [1, 3, 2, 4, 3, 5]})
        p = ggplot(df, aes("x", "y")) + stat_summary(fun_data="mean_se")
        built = ggplot_build(p)

    def test_summary_numpy_func(self):
        """Cover the numpy-function resolution branch of _make_summary_fun
        (``hasattr(np, f) → getattr(np, f)``).  ``fun_y`` was the
        ggplot2 ≤3.2 spelling and is silently dropped in 3.3+, so passing
        it would never exercise the resolver — use ``fun`` (current name)
        and assert the resolver actually returned ``np.mean``.
        """
        from ggplot2_py.stat import stat_summary, _make_summary_fun
        # First check the resolver directly: "mean" must resolve to np.mean.
        summarise = _make_summary_fun(fun="mean")
        out = summarise(pd.DataFrame({"y": [1.0, 3.0]}))
        assert out["y"].iloc[0] == 2.0
        # Then the integration call uses the same resolver path through
        # stat_summary, so a build round-trip must also succeed.
        df = pd.DataFrame({"x": [1, 1, 2, 2], "y": [1, 3, 2, 4]})
        p = ggplot(df, aes("x", "y")) + stat_summary(fun="mean")
        built = ggplot_build(p)
        assert built is not None


# ===========================================================================
# Density – bandwidth methods
# ===========================================================================

class TestStatDensityBW:
    def test_density_default(self):
        """Cover density compute with default settings."""
        from ggplot2_py.geom import geom_density
        df = pd.DataFrame({"x": np.random.randn(50)})
        p = ggplot(df, aes("x")) + geom_density()
        built = ggplot_build(p)
        assert built is not None

    def test_density_bw_nrd(self):
        """Cover line 1096: nrd bandwidth."""
        from ggplot2_py.stat import _precompute_bw
        x = np.random.randn(20)
        result = _precompute_bw(x, "nrd")
        assert result > 0

    def test_density_bw_sj(self):
        """Cover lines 1103-1104: sj bandwidth."""
        from ggplot2_py.stat import _precompute_bw
        x = np.random.randn(20)
        result = _precompute_bw(x, "sj")
        assert result > 0

    def test_density_bw_ucv(self):
        """Cover lines 1109-1110: ucv bandwidth."""
        from ggplot2_py.stat import _precompute_bw
        x = np.random.randn(20)
        result = _precompute_bw(x, "ucv")
        assert result > 0


# ===========================================================================
# StatBin edge cases
# ===========================================================================

class TestStatBinEdges:
    def test_bin_breaks_edge(self):
        """Cover lines 364, 472: bin breaks edge cases."""
        from ggplot2_py.stat import _Bins, _bin_breaks_width
        bins = _bin_breaks_width(np.array([0.0, 1.0]), width=0.5)
        assert bins is not None

    def test_stat_bin_zero_count(self):
        """Cover lines 579-581: zero count/density."""
        from ggplot2_py.geom import geom_histogram
        # Single value -> all same bin
        df = pd.DataFrame({"x": [5.0] * 5})
        p = ggplot(df, aes("x")) + geom_histogram(bins=3)
        built = ggplot_build(p)
        assert built is not None


# ===========================================================================
# StatSmooth – different methods
# ===========================================================================

class TestStatSmoothMethods:
    def test_smooth_lm(self):
        """Cover lines 2324: lm method."""
        from ggplot2_py.geom import geom_smooth
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]})
        p = ggplot(df, aes("x", "y")) + geom_smooth(method="lm", se=False)
        built = ggplot_build(p)

    def test_smooth_loess(self):
        """Cover lines 2422-2428: loess/lowess fallback."""
        from ggplot2_py.geom import geom_smooth
        df = pd.DataFrame({"x": np.arange(10), "y": np.random.randn(10)})
        p = ggplot(df, aes("x", "y")) + geom_smooth(method="loess", se=False)
        built = ggplot_build(p)

    def test_smooth_glm(self):
        """Cover line 2324: glm fallback."""
        from ggplot2_py.geom import geom_smooth
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]})
        p = ggplot(df, aes("x", "y")) + geom_smooth(method="glm", se=False)
        built = ggplot_build(p)


# ===========================================================================
# StatBin2d
# ===========================================================================

class TestStatBin2d:
    def test_bin2d(self):
        """Cover StatBin2d compute."""
        from ggplot2_py.geom import geom_bin2d
        df = pd.DataFrame({
            "x": np.random.randn(20),
            "y": np.random.randn(20),
        })
        p = ggplot(df, aes("x", "y")) + geom_bin2d(bins=5)
        built = ggplot_build(p)


# ===========================================================================
# StatContour
# ===========================================================================

class TestStatContour:
    def test_contour(self):
        """Cover StatContour lines 4243-4269."""
        from ggplot2_py.stat import stat_contour
        # Grid data for contour
        x = np.tile(np.arange(5), 5)
        y = np.repeat(np.arange(5), 5)
        z = np.sin(x) + np.cos(y)
        df = pd.DataFrame({"x": x, "y": y, "z": z})
        p = ggplot(df, aes("x", "y", z="z")) + stat_contour()
        built = ggplot_build(p)


# ===========================================================================
# StatQuantile
# ===========================================================================

class TestStatQuantile:
    def test_quantile(self):
        """Cover StatQuantile lines 5718-5727."""
        from ggplot2_py.stat import stat_quantile
        df = pd.DataFrame({"x": np.arange(20), "y": np.random.randn(20)})
        p = ggplot(df, aes("x", "y")) + stat_quantile()
        built = ggplot_build(p)


# ===========================================================================
# StatSf
# ===========================================================================

class TestStatSf:
    def test_stat_sf_bounds(self):
        """Cover StatSf lines 5826-5831 fallback.

        StatSf requires a 'geometry' column and geopandas; without them the
        compute_panel method simply returns data unchanged.  Test the fallback
        path by calling compute_panel directly with data that lacks geometry.
        """
        from ggplot2_py.stat import StatSf
        sf = StatSf()
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = sf.compute_panel(df, scales={})
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# _is_discrete helper (lives in scale.py)
# ===========================================================================

class TestIsDiscrete:
    def test_is_discrete_categorical(self):
        """Cover line 257: pd.CategoricalDtype."""
        from ggplot2_py.scale import _is_discrete
        assert _is_discrete(pd.Categorical(["a", "b"])) is True

    def test_is_discrete_object(self):
        from ggplot2_py.scale import _is_discrete
        s = pd.Series(["a", "b"], dtype=object)
        assert _is_discrete(s) is True

    def test_is_discrete_numeric(self):
        from ggplot2_py.scale import _is_discrete
        s = pd.Series([1.0, 2.0])
        assert _is_discrete(s) is False


# ===========================================================================
# StatECDF
# ===========================================================================

class TestStatEcdf:
    def test_ecdf_via_geom(self):
        """Cover StatEcdf compute."""
        from ggplot2_py.stat import stat_ecdf
        df = pd.DataFrame({"x": np.random.randn(20)})
        p = ggplot(df, aes("x")) + stat_ecdf()
        built = ggplot_build(p)


# ===========================================================================
# StatSummaryBin
# ===========================================================================

class TestStatSummaryBin:
    def test_summary_bin(self):
        from ggplot2_py.stat import stat_summary_bin
        df = pd.DataFrame({"x": np.random.randn(20), "y": np.random.randn(20)})
        p = ggplot(df, aes("x", "y")) + stat_summary_bin(bins=5)
        built = ggplot_build(p)


# ===========================================================================
# More direct stat method tests
# ===========================================================================

class TestStatDirect:
    def test_stat_density_compute(self):
        """Cover StatDensity.compute_group."""
        from ggplot2_py.stat import StatDensity
        stat = StatDensity()
        df = pd.DataFrame({
            "x": np.random.randn(20),
            "weight": np.ones(20),
        })
        result = stat.compute_group(df, scales={},
            bw="nrd0", adjust=1, kernel="gaussian",
            n=64, trim=False, na_rm=False,
            bounds=(-np.inf, np.inf), flipped_aes=False,
        )
        assert isinstance(result, pd.DataFrame)
        assert "density" in result.columns

    def test_stat_smooth_compute_lm(self):
        """Cover StatSmooth.compute_group with lm."""
        from ggplot2_py.stat import StatSmooth
        stat = StatSmooth()
        df = pd.DataFrame({
            "x": np.arange(10, dtype=float),
            "y": np.arange(10, dtype=float) + np.random.randn(10) * 0.1,
            "weight": np.ones(10),
        })
        result = stat.compute_group(df, scales={},
            method="lm", formula=None, se=True,
            level=0.95, n=80, span=0.75,
            fullrange=False, xseq=None, method_args={},
            na_rm=False, flipped_aes=False,
        )
        assert result is not None

    def test_stat_smooth_compute_loess(self):
        """Cover StatSmooth.compute_group with loess (fallback)."""
        from ggplot2_py.stat import StatSmooth
        stat = StatSmooth()
        df = pd.DataFrame({
            "x": np.arange(20, dtype=float),
            "y": np.random.randn(20),
            "weight": np.ones(20),
        })
        result = stat.compute_group(df, scales={},
            method="loess", formula=None, se=False,
            level=0.95, n=50, span=0.75,
            fullrange=False, xseq=None, method_args={},
            na_rm=False, flipped_aes=False,
        )

    def test_stat_boxplot_compute(self):
        """Cover StatBoxplot.compute_group."""
        from ggplot2_py.stat import StatBoxplot
        stat = StatBoxplot()
        df = pd.DataFrame({
            "x": [1.0] * 10,
            "y": np.random.randn(10),
            "weight": np.ones(10),
        })
        result = stat.compute_group(df, scales={},
            width=0.75, na_rm=False,
            coef=1.5, flipped_aes=False,
        )
        assert isinstance(result, pd.DataFrame)

    def test_stat_ydensity_compute(self):
        """Cover StatYdensity.compute_group."""
        from ggplot2_py.stat import StatYdensity
        stat = StatYdensity()
        df = pd.DataFrame({
            "x": [1.0] * 20,
            "y": np.random.randn(20),
            "weight": np.ones(20),
        })
        result = stat.compute_group(df, scales={},
            bw="nrd0", adjust=1, kernel="gaussian",
            n=64, trim=True, na_rm=False,
            bounds=(-np.inf, np.inf), scale="area",
            flipped_aes=False, quantiles=[0.25, 0.5, 0.75],
        )
        assert isinstance(result, pd.DataFrame)

    def test_stat_summary_compute(self):
        """Cover StatSummary compute_panel (StatSummary uses panel-level compute)."""
        from ggplot2_py.stat import StatSummary
        stat = StatSummary()
        df = pd.DataFrame({
            "x": [1, 1, 2, 2],
            "y": [1.0, 3.0, 2.0, 4.0],
        })
        result = stat.compute_panel(df, scales={},
            fun=None,
            na_rm=False,
            flipped_aes=False,
        )
        assert isinstance(result, pd.DataFrame)

    def test_stat_function_compute(self):
        """Cover StatFunction.compute_group."""
        from ggplot2_py.stat import StatFunction
        stat = StatFunction()
        df = pd.DataFrame({"x": [0.0, 3.14]})
        result = stat.compute_group(df, scales={},
            fun=np.sin, n=20,
            xlim=None, args={},
            na_rm=False,
        )
        assert isinstance(result, pd.DataFrame)

    def test_stat_ecdf_compute(self):
        """Cover StatEcdf.compute_group."""
        from ggplot2_py.stat import StatEcdf
        stat = StatEcdf()
        df = pd.DataFrame({
            "x": np.random.randn(20),
            "weight": np.ones(20),
        })
        result = stat.compute_group(df, scales={},
            n=None, pad=True,
            na_rm=False, flipped_aes=False,
        )
        assert isinstance(result, pd.DataFrame)
