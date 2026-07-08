"""Targeted coverage tests for ggplot2_py.stat – missing lines."""

import pytest
import numpy as np
import pandas as pd
import warnings

from ggplot2_py.stat import (
    Stat,
    StatIdentity,
    StatBin,
    StatCount,
    StatDensity,
    StatBoxplot,
    StatSummary,
    StatSummaryBin,
    StatFunction,
    StatEcdf,
    StatQq,
    StatQqLine,
    StatContour,
    StatContourFilled,
    StatDensity2d,
    StatDensity2dFilled,
    StatEllipse,
    StatUnique,
    StatSum,
    StatYdensity,
    StatBindot,
    StatAlign,
    StatConnect,
    StatManual,
    StatQuantile,
    StatSf,
    StatSfCoordinates,
    StatBin2d,
    StatBinhex,
    # Constructors
    stat_identity,
    stat_bin,
    stat_count,
    stat_density,
    stat_boxplot,
    stat_summary,
    stat_function,
    stat_ecdf,
    stat_qq,
    stat_qq_line,
    stat_contour,
    stat_contour_filled,
    stat_unique,
    stat_sum,
    stat_ydensity,
    stat_align,
    stat_connect,
    stat_manual,
    stat_quantile,
    stat_sf,
    stat_sf_coordinates,
    # Helpers
    is_stat,
    mean_se,
    mean_cl_boot,
    mean_cl_normal,
    mean_sdl,
    median_hilow,
)

# Try importing density/smooth-related constructors
try:
    from ggplot2_py.stat import (
        stat_density_2d,
        stat_density_2d_filled,
        stat_density2d,
        stat_density2d_filled,
        stat_bin2d,
        stat_bin_2d,
        stat_bin_hex,
        stat_binhex,
        stat_summary_bin,
        stat_summary2d,
        stat_summary_2d,
        stat_summary_hex,
        stat_spoke,
    )
    HAS_EXTRA = True
except ImportError:
    HAS_EXTRA = False

# Import helpers from their correct modules
from ggplot2_py.stat import (
    _layer,
    _layer_sf,
    _flip_data,
    _has_flipped_aes,
    _precompute_bw,
    _contour_breaks,
    _check_required_aesthetics,
)
from ggplot2_py.scale import _is_discrete
HAS_INTERNALS = True


# ===========================================================================
# Internal helper tests
# ===========================================================================

@pytest.mark.skipif(not HAS_INTERNALS, reason="internals not exported")
class TestInternalHelpers:
    def test_layer_returns_layer(self):
        result = _layer(stat=StatIdentity, geom="point", data=None,
                        mapping=None, position="identity",
                        show_legend=None, inherit_aes=True, params={})
        assert result is not None

    def test_layer_sf_returns_layer(self):
        result = _layer_sf(stat=StatSf, geom="rect", data=None,
                           mapping=None, position="identity",
                           show_legend=None, inherit_aes=True, params={})
        assert result is not None

    def test_flip_data_no_flip(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = _flip_data(df, False)
        assert list(result["x"]) == [1, 2]

    def test_flip_data_flipped(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = _flip_data(df, True)
        assert list(result["x"]) == [3, 4]

    def test_is_discrete_none(self):
        assert _is_discrete(None) is False

    def test_is_discrete_categorical(self):
        s = pd.Categorical(["a", "b"])
        assert _is_discrete(s) is True

    def test_is_discrete_object_series(self):
        s = pd.Series(["a", "b"])
        assert _is_discrete(s) is True

    def test_is_discrete_numeric(self):
        s = pd.Series([1.0, 2.0])
        assert _is_discrete(s) is False

    def test_precompute_bw_nrd0(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _precompute_bw(x, "nrd0")
        assert result > 0

    def test_precompute_bw_nrd(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _precompute_bw(x, "nrd")
        assert result > 0

    def test_precompute_bw_sj(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _precompute_bw(x, "sj")
        assert result > 0

    def test_precompute_bw_ucv(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _precompute_bw(x, "ucv")
        assert result > 0

    def test_precompute_bw_unknown(self):
        with pytest.raises(ValueError):
            _precompute_bw(np.array([1.0, 2.0, 3.0]), "unknown_bw_method")

    def test_contour_breaks_with_callable(self):
        result = _contour_breaks([0, 10], breaks=lambda r: np.linspace(r[0], r[1], 5))
        assert len(result) == 5

    def test_contour_breaks_default_bins(self):
        result = _contour_breaks([0, 10])
        assert len(result) > 0

    def test_contour_breaks_with_bins(self):
        result = _contour_breaks([0, 10], bins=5)
        assert len(result) >= 5

    def test_contour_breaks_with_binwidth(self):
        result = _contour_breaks([0, 10], binwidth=2.0)
        assert len(result) > 0

    def test_contour_breaks_with_bins_1(self):
        result = _contour_breaks([0, 10], bins=1)
        assert len(result) == 2

    def test_has_flipped_aes(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = _has_flipped_aes(df, {})
        assert isinstance(result, bool)


# ===========================================================================
# Stat base class: calculate / compute_panel (lines 1479-1528, 1555-1575)
# ===========================================================================

class TestStatComputeLayer:
    def test_compute_layer_with_panel(self):
        """Test Stat.compute_layer with PANEL column."""
        class MockLayout:
            def get_scales(self, panel):
                return {}
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "group": [1, 1, 1],
            "PANEL": [1, 1, 1],
        })
        stat = StatIdentity()
        result = stat.compute_layer(df, {}, MockLayout())
        assert not result.empty

    def test_compute_layer_without_panel(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
        stat = StatIdentity()
        result = stat.compute_layer(df, {}, None)
        assert not result.empty

    def test_compute_layer_empty_result(self):
        """Computation returning empty df."""
        class FailStat(Stat):
            required_aes = ["x"]
            def compute_group(self, data, scales, **params):
                return pd.DataFrame()
        df = pd.DataFrame({"x": [1.0], "group": [1]})
        stat = FailStat()
        result = stat.compute_layer(df, {}, None)
        assert result.empty

    def test_compute_layer_exception_in_panel(self):
        """Exception in compute_panel -> warning."""
        class ErrorStat(Stat):
            required_aes = ["x"]
            def compute_panel(self, data, scales, **params):
                raise RuntimeError("fail")
        df = pd.DataFrame({"x": [1.0], "group": [1]})
        stat = ErrorStat()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = stat.compute_layer(df, {}, None)
        assert result.empty

    def test_compute_panel_exception_handling(self):
        """compute_group raising -> warning, empty result."""
        class ErrorStat(Stat):
            required_aes = ["x"]
            def compute_group(self, data, scales, **params):
                raise RuntimeError("fail")
        df = pd.DataFrame({"x": [1.0], "group": [1]})
        stat = ErrorStat()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = stat.compute_panel(df, {})
        assert result.empty

    def test_compute_panel_returns_non_df(self):
        """compute_group returning dict -> converted to DataFrame."""
        class DictStat(Stat):
            required_aes = ["x"]
            def compute_group(self, data, scales, **params):
                return {"x": [1.0], "y": [2.0]}
        df = pd.DataFrame({"x": [1.0], "group": [1]})
        stat = DictStat()
        result = stat.compute_panel(df, {})
        assert not result.empty

    def test_compute_panel_preserves_constant_cols(self):
        """Constant columns from group data should be preserved."""
        class SimpleStat(Stat):
            required_aes = ["x"]
            def compute_group(self, data, scales, **params):
                return pd.DataFrame({"x": [data["x"].mean()], "y": [1.0]})
        df = pd.DataFrame({"x": [1.0, 2.0], "colour": ["red", "red"], "group": [1, 1]})
        stat = SimpleStat()
        result = stat.compute_panel(df, {})
        assert "colour" in result.columns


# ===========================================================================
# StatFunction compute_group (lines 3367, 3409-3411)
# ===========================================================================

class TestStatFunction:
    def test_compute_group(self):
        df = pd.DataFrame({"x": [0.0, 1.0], "group": [1, 1]})
        result = StatFunction().compute_group(df, {}, fun=lambda x: x ** 2, n=5)
        assert "x" in result.columns
        assert "y" in result.columns
        assert len(result) == 5

    def test_compute_group_with_xlim(self):
        df = pd.DataFrame({"x": [0.0], "group": [1]})
        result = StatFunction().compute_group(df, {}, fun=np.sin, xlim=(0, np.pi), n=5)
        assert len(result) == 5

    def test_compute_group_no_fun(self):
        df = pd.DataFrame({"x": [0.0], "group": [1]})
        with pytest.raises(ValueError):
            StatFunction().compute_group(df, {})

    def test_stat_function_constructor(self):
        result = stat_function(fun=lambda x: x)
        assert result is not None


# ===========================================================================
# StatEcdf compute_group (lines 3604-3605, 3618-3621)
# ===========================================================================

class TestStatEcdf:
    def test_compute_group(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "group": [1, 1, 1, 1]})
        result = StatEcdf().compute_group(df, {})
        assert "x" in result.columns
        assert "y" in result.columns

    def test_compute_group_with_n(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "group": [1, 1, 1, 1]})
        result = StatEcdf().compute_group(df, {}, n=10)
        assert len(result) >= 10  # May include padding rows

    def test_stat_ecdf_constructor(self):
        result = stat_ecdf()
        assert result is not None


# ===========================================================================
# StatContour compute_group (lines 4113-4159)
# ===========================================================================

class TestStatContour:
    def test_compute_group_tiny(self):
        # 3x3 grid
        x = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=float)
        z = np.array([1, 2, 1, 2, 4, 2, 1, 2, 1], dtype=float)
        df = pd.DataFrame({"x": x, "y": y, "z": z, "group": 1})
        result = StatContour().compute_group(df, {})
        # May be empty if contour computation fails, but should not raise
        assert isinstance(result, pd.DataFrame)

    def test_stat_contour_constructor(self):
        result = stat_contour()
        assert result is not None

    def test_stat_contour_filled_constructor(self):
        result = stat_contour_filled()
        assert result is not None


# ===========================================================================
# StatDensity2d compute_group (lines 4425, 4432, 4437, 4450-4451)
# ===========================================================================

class TestStatDensity2d:
    def test_compute_group_tiny(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.randn(5),
            "y": np.random.randn(5),
            "group": 1,
        })
        result = StatDensity2d().compute_group(df, {}, n=5)
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# StatEllipse compute_group (lines 4668, 4672-4673, 4722)
# ===========================================================================

class TestStatEllipse:
    def test_compute_group(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.randn(5),
            "y": np.random.randn(5),
            "group": 1,
        })
        result = StatEllipse().compute_group(df, {}, segments=10)
        assert isinstance(result, pd.DataFrame)

    def test_compute_group_euclid(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.randn(5),
            "y": np.random.randn(5),
            "group": 1,
        })
        result = StatEllipse().compute_group(df, {}, type="euclid", segments=10)
        assert isinstance(result, pd.DataFrame)

    def test_compute_group_unknown_type(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "group": [1, 1]})
        result = StatEllipse().compute_group(df, {}, type="unknown")
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# StatUnique compute_panel (line 4788)
# ===========================================================================

class TestStatUnique:
    def test_compute_panel(self):
        df = pd.DataFrame({"x": [1, 1, 2], "y": [1, 1, 2]})
        result = StatUnique().compute_panel(df, {})
        assert len(result) == 2

    def test_stat_unique_constructor(self):
        result = stat_unique()
        assert result is not None


# ===========================================================================
# StatSum compute_group (line 4877)
# ===========================================================================

class TestStatSum:
    def test_compute_panel(self):
        df = pd.DataFrame({"x": [1, 1, 2, 2, 2], "y": [1, 1, 2, 2, 2], "group": 1})
        result = StatSum().compute_panel(df, {})
        assert isinstance(result, pd.DataFrame)
        assert "n" in result.columns

    def test_stat_sum_constructor(self):
        result = stat_sum()
        assert result is not None


# ===========================================================================
# StatYdensity compute_group (lines 4995, 5013-5016)
# ===========================================================================

class TestStatYdensity:
    def test_compute_group(self):
        df = pd.DataFrame({
            "x": [1.0, 1.0, 1.0, 1.0, 1.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "group": 1,
        })
        result = StatYdensity().compute_group(df, {}, n=10)
        assert isinstance(result, pd.DataFrame)
        assert "density" in result.columns

    def test_compute_group_with_quantiles(self):
        """Test with quantiles (lines 4995, 5013-5016)."""
        df = pd.DataFrame({
            "x": [1.0] * 10,
            "y": np.random.randn(10),
            "group": 1,
        })
        result = StatYdensity().compute_group(df, {}, n=10, quantiles=[0.25, 0.5, 0.75])
        assert isinstance(result, pd.DataFrame)

    def test_stat_ydensity_constructor(self):
        result = stat_ydensity()
        assert result is not None


# ===========================================================================
# StatBindot compute_group (lines 5167, 5183, 5196)
# ===========================================================================

class TestStatBindot:
    def test_compute_group_histodot(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "group": 1,
        })
        result = StatBindot().compute_group(df, {}, method="histodot", binwidth=1.0)
        assert isinstance(result, pd.DataFrame)

    def test_compute_group_dotdensity(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "group": 1,
        })
        result = StatBindot().compute_group(df, {}, method="dotdensity", binwidth=1.0)
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# StatAlign compute_group (lines 5284-5285, 5306-5308)
# ===========================================================================

class TestStatAlign:
    def test_compute_group(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 4.0, 2.0, 3.0, 5.0],
            "group": 1,
        })
        result = StatAlign().compute_group(df, {})
        assert isinstance(result, pd.DataFrame)

    def test_stat_align_constructor(self):
        result = stat_align()
        assert result is not None


# ===========================================================================
# StatConnect compute_group (lines 5421-5466)
# ===========================================================================

class TestStatConnect:
    def test_compute_group_string_connection(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 3.0, 2.0],
            "group": 1,
        })
        result = StatConnect().compute_group(df, {}, connection="hv")
        assert isinstance(result, pd.DataFrame)

    def test_compute_group_array_connection(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 3.0, 2.0],
            "group": 1,
        })
        conn = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        result = StatConnect().compute_group(df, {}, connection=conn)
        assert isinstance(result, pd.DataFrame)

    def test_compute_group_single_row(self):
        df = pd.DataFrame({"x": [1.0], "y": [1.0], "group": [1]})
        result = StatConnect().compute_group(df, {})
        assert result.empty or len(result) <= 1

    def test_stat_connect_constructor(self):
        result = stat_connect()
        assert result is not None


# ===========================================================================
# StatManual compute_group (lines 5532-5535, 5561-5567)
# ===========================================================================

class TestStatManual:
    def test_compute_group(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "group": [1, 1]})
        result = StatManual().compute_group(df, {}, fun=lambda d: d.head(1))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_compute_group_no_fun(self):
        df = pd.DataFrame({"x": [1.0], "y": [1.0], "group": [1]})
        result = StatManual().compute_group(df, {})
        assert isinstance(result, pd.DataFrame)

    def test_setup_params_non_callable(self):
        with pytest.raises(ValueError):
            StatManual().setup_params(pd.DataFrame(), {"fun": "not_callable"})

    def test_stat_manual_constructor(self):
        result = stat_manual()
        assert result is not None


# ===========================================================================
# StatQuantile compute_group (lines 5673-5674, 5684, 5718-5727)
# ===========================================================================

class TestStatQuantile:
    def test_compute_group(self):
        np.random.seed(42)
        x = np.linspace(0, 1, 5)
        y = x + np.random.randn(5) * 0.1
        df = pd.DataFrame({"x": x, "y": y, "group": 1})
        result = StatQuantile().compute_group(df, {}, quantiles=[0.5])
        assert isinstance(result, pd.DataFrame)
        assert "quantile" in result.columns

    def test_stat_quantile_constructor(self):
        result = stat_quantile()
        assert result is not None


# ===========================================================================
# StatSf / StatSfCoordinates compute_panel (lines 5811-5833, 5869-5889)
# ===========================================================================

class TestStatSf:
    def test_compute_panel_no_geometry(self):
        df = pd.DataFrame({"x": [1.0]})
        result = StatSf().compute_panel(df, {})
        assert isinstance(result, pd.DataFrame)

    def test_compute_panel_with_geometry(self):
        df = pd.DataFrame({"geometry": [None, None]})
        result = StatSf().compute_panel(df, {})
        assert "xmin" in result.columns

    def test_stat_sf_constructor(self):
        result = stat_sf()
        assert result is not None


class TestStatSfCoordinates:
    def test_compute_panel_no_geometry(self):
        df = pd.DataFrame({"x": [1.0]})
        result = StatSfCoordinates().compute_panel(df, {})
        assert isinstance(result, pd.DataFrame)

    def test_compute_panel_with_geometry(self):
        df = pd.DataFrame({"geometry": [None, None]})
        result = StatSfCoordinates().compute_panel(df, {})
        assert "x" in result.columns

    def test_stat_sf_coordinates_constructor(self):
        result = stat_sf_coordinates()
        assert result is not None


# ===========================================================================
# stat_* constructor functions for coverage
# ===========================================================================

@pytest.mark.skipif(not HAS_EXTRA, reason="extra stat constructors not available")
class TestExtraStatConstructors:
    def test_stat_density_2d(self):
        assert stat_density_2d() is not None

    def test_stat_density_2d_filled(self):
        assert stat_density_2d_filled() is not None

    def test_stat_bin2d(self):
        assert stat_bin2d() is not None

    def test_stat_bin_2d(self):
        assert stat_bin_2d() is not None

    def test_stat_binhex(self):
        assert stat_binhex() is not None

    def test_stat_bin_hex(self):
        assert stat_bin_hex() is not None

    def test_stat_summary_bin(self):
        assert stat_summary_bin() is not None

    def test_stat_summary2d(self):
        assert stat_summary2d() is not None

    def test_stat_summary_2d(self):
        assert stat_summary_2d() is not None

    def test_stat_summary_hex(self):
        assert stat_summary_hex() is not None

    def test_stat_spoke(self):
        assert stat_spoke() is not None


# ===========================================================================
# StatDensity compute_group branches (lines 953, 961-968)
# ===========================================================================

class TestStatDensityBranches:
    def test_compute_group_few_points(self):
        """< 2 points should warn and return NaN frame."""
        df = pd.DataFrame({"x": [1.0], "group": [1]})
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = StatDensity().compute_group(df, {})
        assert isinstance(result, pd.DataFrame)

    def test_compute_group_with_bounds(self):
        """Data with bounds parameter."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "group": 1})
        result = StatDensity().compute_group(df, {}, bounds=(0, 10))
        assert isinstance(result, pd.DataFrame)

    def test_compute_group_with_tight_bounds(self):
        """Data outside bounds should trigger warning."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 20.0], "group": 1})
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = StatDensity().compute_group(df, {}, bounds=(0, 5))
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# StatSmooth branches (lines 2324-2327, 2330, 2422-2428, 2440-2441, 2450-2452)
# These are slow, so we test with tiny data only
# ===========================================================================

class TestStatSmoothBranches:
    def test_compute_group_loess_tiny(self):
        """Test loess method with tiny data."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 2.0, 1.5, 3.0, 2.5],
            "group": 1,
        })
        from ggplot2_py.stat import StatSmooth
        result = StatSmooth().compute_group(df, {}, method="loess", n=3, se=False)
        assert isinstance(result, pd.DataFrame)

    def test_compute_group_lm_no_se(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "group": 1,
        })
        from ggplot2_py.stat import StatSmooth
        result = StatSmooth().compute_group(df, {}, method="lm", n=3, se=False)
        assert isinstance(result, pd.DataFrame)
        assert "ymin" in result.columns

    def test_compute_group_glm(self):
        """Test glm method (lines 2324)."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 2.0, 1.5, 3.0, 2.5],
            "group": 1,
        })
        from ggplot2_py.stat import StatSmooth
        result = StatSmooth().compute_group(df, {}, method="glm", n=3, se=False)
        assert isinstance(result, pd.DataFrame)

    def test_compute_group_loess_with_se(self):
        """Test loess with se=True (lines 2422-2428, 2440-2441)."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 2.0, 1.5, 3.0, 2.5],
            "group": 1,
        })
        from ggplot2_py.stat import StatSmooth
        result = StatSmooth().compute_group(df, {}, method="loess", n=3, se=True)
        assert isinstance(result, pd.DataFrame)

    def test_compute_group_lm_with_se(self):
        """Test lm with se=True."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 2.0, 1.5, 3.0, 2.5],
            "group": 1,
        })
        from ggplot2_py.stat import StatSmooth
        result = StatSmooth().compute_group(df, {}, method="lm", n=3, se=True, level=0.95)
        assert isinstance(result, pd.DataFrame)
        assert "ymin" in result.columns
        assert "ymax" in result.columns


# ===========================================================================
# Summary helpers (lines 844, 857-859, 877)
# ===========================================================================

# ===========================================================================
# StatQq/StatQqLine compute_group (lines 3604-3605, 3618-3621, 3736-3741, 3751)
# ===========================================================================

class TestStatQq:
    def test_compute_group(self):

        df = pd.DataFrame({"sample": [1.0, 2.0, 3.0, 4.0, 5.0], "group": 1})
        result = StatQq().compute_group(df, {})
        assert "sample" in result.columns
        assert "theoretical" in result.columns

    def test_compute_group_with_quantiles(self):

        df = pd.DataFrame({"sample": [1.0, 2.0, 3.0], "group": 1})
        result = StatQq().compute_group(df, {}, quantiles=np.array([0.25, 0.5, 0.75]))
        assert len(result) == 3

    def test_stat_qq_constructor(self):
        result = stat_qq()
        assert result is not None


class TestStatQqLine:
    def test_compute_group(self):

        df = pd.DataFrame({"sample": np.random.randn(10), "group": 1})
        result = StatQqLine().compute_group(df, {})
        assert "x" in result.columns
        assert "y" in result.columns

    def test_compute_group_callable_distribution(self):
        """Test with callable distribution (lines 3736-3741)."""

        df = pd.DataFrame({"sample": np.random.randn(10), "group": 1})
        # Use a callable instead of a distribution object
        result = StatQqLine().compute_group(df, {}, distribution=lambda q: q * 2)
        assert isinstance(result, pd.DataFrame)

    def test_stat_qq_line_constructor(self):
        result = stat_qq_line()
        assert result is not None


# ===========================================================================
# StatContourFilled compute_group (lines 4217-4269)
# ===========================================================================

class TestStatContourFilled:
    def test_compute_group_tiny(self):
        # 5x5 grid for more reliable contouring
        xx, yy = np.meshgrid(np.arange(5, dtype=float), np.arange(5, dtype=float))
        z = np.exp(-((xx-2)**2 + (yy-2)**2) / 2)
        df = pd.DataFrame({
            "x": xx.ravel(), "y": yy.ravel(), "z": z.ravel(), "group": 1
        })
        result = StatContourFilled().compute_group(df, {})
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# StatBin2d compute_group (lines 3878, 3923)
# ===========================================================================

class TestStatBin2d:
    def test_compute_group(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.randn(10),
            "y": np.random.randn(10),
            "group": 1,
        })
        result = StatBin2d().compute_group(df, {}, bins=3)
        assert isinstance(result, pd.DataFrame)

    def test_stat_bin2d_constructor(self):
        result = stat_bin2d()
        assert result is not None


# ===========================================================================
# StatBinhex compute_group (lines 3989, 3995, 4038)
# ===========================================================================

class TestStatBinhex:
    def test_compute_group(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.randn(10),
            "y": np.random.randn(10),
            "group": 1,
        })
        result = StatBinhex().compute_group(df, {}, bins=3)
        assert isinstance(result, pd.DataFrame)


class TestSummaryHelpers:
    def test_mean_se(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mean_se(y)
        assert isinstance(result, pd.DataFrame)
        assert "y" in result.columns

    def test_mean_cl_boot(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mean_cl_boot(y)
        assert isinstance(result, pd.DataFrame)

    def test_mean_cl_normal(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mean_cl_normal(y)
        assert isinstance(result, pd.DataFrame)

    def test_mean_sdl(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mean_sdl(y)
        assert isinstance(result, pd.DataFrame)

    def test_median_hilow(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = median_hilow(y)
        assert isinstance(result, pd.DataFrame)


class TestStatBindotParity:
    """R stat-bindot.R semantics validated against gold-standard runs."""

    def test_default_aes_y_is_count(self):
        # R stat-bindot.R:8 aes(y = after_stat(count)) — geom_dotplot
        # works from aes(x) alone.
        from ggplot2_py.aes import AfterStat
        assert isinstance(StatBindot.default_aes.get("y"), AfterStat)

    def test_weight_must_be_whole_nonnegative(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "weight": [0.5, 1.0], "group": [1, 1]})
        with pytest.raises(ValueError):
            StatBindot().compute_panel(df, {}, binwidth=1.0)
        df2 = pd.DataFrame({"x": [1.0, 2.0], "weight": [-1, 1], "group": [1, 1]})
        with pytest.raises(ValueError):
            StatBindot().compute_panel(df2, {}, binwidth=1.0)

    def test_binaxis_y_sets_midline(self):
        df = pd.DataFrame({"x": [1.0, 3.0, 1.0], "y": [1.0, 2.0, 5.0]})
        result = StatBindot().compute_group(df, {}, binaxis="y", binwidth=1.0)
        # R: data$x <- midline = mean(range(data$x)) = 2
        assert (result["x"] == 2.0).all()
        assert "y" in result.columns

    def test_binpositions_all_aligns_bins(self):
        # bins computed over the whole panel: both groups share centers
        df = pd.DataFrame({
            "x": [0.0, 0.0, 0.0, 0.0], "y": [1.0, 1.2, 1.1, 1.3],
            "group": [1, 1, 2, 2],
        })
        result = StatBindot().compute_panel(
            df, {}, binaxis="y", method="dotdensity",
            binpositions="all", binwidth=0.5,
        )
        centers = sorted(result["y"].unique())
        assert len(centers) == 1  # all four values fall in one shared bin

    def test_dotdensity_counts(self):
        # Wilkinson binning: [1, 1.2] bin, [2.5] bin with binwidth 0.5
        df = pd.DataFrame({"x": [1.0, 1.2, 2.5]})
        result = StatBindot().compute_group(df, {}, binwidth=0.5)
        assert list(result["count"]) == [2, 1]
        assert list(result["x"]) == [1.1, 2.5]
        assert (result["binwidth"] == 0.5).all()
