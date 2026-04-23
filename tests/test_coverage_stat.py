"""Comprehensive tests for ggplot2_py.stat to improve coverage."""

import pytest
import numpy as np
import pandas as pd
import warnings

from ggplot2_py.stat import (
    # Base class
    Stat,
    StatIdentity,
    # Concrete stats
    StatBin,
    StatCount,
    StatDensity,
    StatSmooth,
    StatBoxplot,
    StatSummary,
    StatSummaryBin,
    StatSummary2d,
    StatSummaryHex,
    StatFunction,
    StatEcdf,
    StatQq,
    StatQqLine,
    StatBin2d,
    StatBinhex,
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
    # Utilities
    is_stat,
    # Summary helpers
    mean_se,
    mean_cl_boot,
    mean_cl_normal,
    mean_sdl,
    median_hilow,
    # Helper functions (internal but we test for coverage)
    _flip_data,
    _has_flipped_aes,
    _is_mapped_discrete,
    _rescale_max,
    _check_required_aesthetics,
    _inner_runs,
    _Bins,
    _compute_bins,
    _bin_breaks_width,
    _bin_breaks_bins,
    _bin_vector,
    _bin_cut,
    _bin_loc,
    _dual_param,
    _compute_density,
    _precompute_bw,
    _wecdf,
    _densitybin,
    _contour_breaks,
    _hex_binwidth,
    _hex_bin_summarise,
    _make_summary_fun,
)


# ============================================================================
# Helper function tests
# ============================================================================

class TestFlipData:
    def test_no_flip(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = _flip_data(df, False)
        assert list(result["x"]) == [1, 2]
        assert list(result["y"]) == [3, 4]

    def test_flip(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = _flip_data(df, True)
        assert list(result["x"]) == [3, 4]
        assert list(result["y"]) == [1, 2]

    def test_flip_with_min_max(self):
        df = pd.DataFrame({"x": [1], "y": [2], "xmin": [0], "ymin": [1], "xmax": [2], "ymax": [3]})
        result = _flip_data(df, True)
        assert result["xmin"].iloc[0] == 1
        assert result["ymin"].iloc[0] == 0

    def test_flip_with_boxplot_cols(self):
        df = pd.DataFrame({
            "x": [1], "y": [2],
            "xlower": [0.5], "lower": [1.5],
            "xupper": [1.5], "upper": [2.5],
            "xmiddle": [1.0], "middle": [2.0],
        })
        result = _flip_data(df, True)
        assert result["xlower"].iloc[0] == 1.5
        assert result["lower"].iloc[0] == 0.5


class TestHasFlippedAes:
    def test_orientation_y(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        assert _has_flipped_aes(df, {"orientation": "y"}) is True

    def test_orientation_x(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        assert _has_flipped_aes(df, {"orientation": "x"}) is False

    def test_explicit_flipped_aes(self):
        df = pd.DataFrame({"x": [1]})
        assert _has_flipped_aes(df, {"flipped_aes": True}) is True
        assert _has_flipped_aes(df, {"flipped_aes": False}) is False

    def test_main_is_orthogonal_x_only(self):
        df = pd.DataFrame({"x": [1]})
        assert _has_flipped_aes(df, {}, main_is_orthogonal=True) is True

    def test_main_is_orthogonal_y_only(self):
        df = pd.DataFrame({"y": [1]})
        assert _has_flipped_aes(df, {}, main_is_orthogonal=True) is False

    def test_main_is_continuous_x_only(self):
        df = pd.DataFrame({"x": [1]})
        assert _has_flipped_aes(df, {}, main_is_continuous=True) is False

    def test_main_is_continuous_y_only(self):
        df = pd.DataFrame({"y": [1]})
        assert _has_flipped_aes(df, {}, main_is_continuous=True) is True

    def test_default(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        assert _has_flipped_aes(df, {}, default=True) is True
        assert _has_flipped_aes(df, {}, default=False) is False


class TestIsMappedDiscrete:
    def test_none(self):
        assert _is_mapped_discrete(None) is False

    def test_categorical(self):
        assert _is_mapped_discrete(pd.Categorical(["a", "b"])) is True

    def test_object_series(self):
        s = pd.Series(["a", "b", "c"])
        assert _is_mapped_discrete(s) is True

    def test_numeric_series(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert _is_mapped_discrete(s) is False

    def test_bool_series(self):
        s = pd.Series([True, False])
        assert _is_mapped_discrete(s) is True


class TestRescaleMax:
    def test_basic(self):
        result = _rescale_max(np.array([1, 2, 4]))
        assert result[-1] == pytest.approx(1.0)

    def test_zeros(self):
        result = _rescale_max(np.array([0, 0, 0]))
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_negative(self):
        result = _rescale_max(np.array([-4, 2, 4]))
        assert np.max(np.abs(result)) == pytest.approx(1.0)


class TestCheckRequiredAesthetics:
    def test_all_present(self):
        _check_required_aesthetics(["x", "y"], ["x", "y", "colour"], "test")

    def test_missing(self):
        with pytest.raises(Exception):
            _check_required_aesthetics(["x", "y"], ["x"], "test")

    def test_alternatives(self):
        # "x|y" means either x or y must be present
        _check_required_aesthetics(["x|y"], ["y"], "test")

    def test_alternatives_missing(self):
        with pytest.raises(Exception):
            _check_required_aesthetics(["x|y"], ["z"], "test")


class TestInnerRuns:
    def test_basic(self):
        x = np.array([False, True, True, False])
        result = _inner_runs(x)
        assert len(result) == len(x)

    def test_empty(self):
        result = _inner_runs(np.array([], dtype=bool))
        assert len(result) == 0

    def test_all_true(self):
        result = _inner_runs(np.array([True, True, True]))
        assert all(result)


# ============================================================================
# Binning helper tests
# ============================================================================

class TestBins:
    def test_construction(self):
        b = _Bins(np.array([0, 1, 2, 3]), "right")
        assert len(b.breaks) == 4
        assert b.right_closed is True
        assert len(b.fuzzy) == 4

    def test_left_closed(self):
        b = _Bins(np.array([0, 1, 2]), "left")
        assert b.right_closed is False


class TestComputeBins:
    def test_default_bins(self):
        x = np.random.normal(0, 1, 100)
        b = _compute_bins(x)
        assert len(b.breaks) > 2

    def test_explicit_breaks(self):
        x = np.arange(10, dtype=float)
        b = _compute_bins(x, breaks=[0, 3, 6, 9])
        assert len(b.breaks) == 4

    def test_callable_breaks(self):
        x = np.arange(10, dtype=float)
        b = _compute_bins(x, breaks=lambda x: [0, 5, 10])
        assert len(b.breaks) == 3

    def test_explicit_binwidth(self):
        x = np.arange(10, dtype=float)
        b = _compute_bins(x, binwidth=2.0)
        assert len(b.breaks) > 2

    def test_callable_binwidth(self):
        x = np.arange(10, dtype=float)
        b = _compute_bins(x, binwidth=lambda x: 2.0)
        assert len(b.breaks) > 2

    def test_explicit_bins(self):
        x = np.arange(10, dtype=float)
        b = _compute_bins(x, bins=5)
        assert len(b.breaks) > 2

    def test_callable_bins(self):
        x = np.arange(10, dtype=float)
        b = _compute_bins(x, bins=lambda x: 5)
        assert len(b.breaks) > 2

    def test_center(self):
        x = np.arange(10, dtype=float)
        b = _compute_bins(x, binwidth=2.0, center=5.0)
        assert len(b.breaks) > 2

    def test_boundary(self):
        x = np.arange(10, dtype=float)
        b = _compute_bins(x, binwidth=2.0, boundary=0.0)
        assert len(b.breaks) > 2

    def test_center_and_boundary_errors(self):
        with pytest.raises(Exception):
            _compute_bins(np.arange(10, dtype=float), center=5.0, boundary=0.0)

    def test_empty_finite(self):
        x = np.array([np.nan, np.nan])
        b = _compute_bins(x)
        assert len(b.breaks) > 0


class TestBinBreaksBins:
    def test_single_bin(self):
        b = _bin_breaks_bins(np.array([0.0, 10.0]), bins=1)
        assert len(b.breaks) >= 2

    def test_zero_range(self):
        b = _bin_breaks_bins(np.array([5.0, 5.0]), bins=10)
        assert len(b.breaks) >= 2

    def test_with_boundary(self):
        b = _bin_breaks_bins(np.array([0.0, 10.0]), bins=5, boundary=0.0)
        assert len(b.breaks) >= 2


class TestBinVector:
    def test_basic(self):
        x = np.arange(10, dtype=float)
        bins_obj = _Bins(np.array([0, 5, 10.0]), "right")
        result = _bin_vector(x, bins_obj)
        assert "count" in result.columns
        assert "density" in result.columns

    def test_with_weights(self):
        x = np.arange(10, dtype=float)
        bins_obj = _Bins(np.array([0, 5, 10.0]), "right")
        weights = np.ones(10) * 2
        result = _bin_vector(x, bins_obj, weight=weights)
        assert result["count"].sum() > 10

    def test_with_pad(self):
        x = np.arange(10, dtype=float)
        bins_obj = _Bins(np.array([0, 5, 10.0]), "right")
        result = _bin_vector(x, bins_obj, pad=True)
        assert len(result) == 4  # 2 bins + 2 padding

    def test_left_closed(self):
        x = np.arange(10, dtype=float)
        bins_obj = _Bins(np.array([0, 5, 10.0]), "left")
        result = _bin_vector(x, bins_obj)
        assert result["count"].sum() > 0

    def test_nan_weights(self):
        x = np.array([1.0, 2.0, 3.0])
        bins_obj = _Bins(np.array([0, 2, 4.0]), "right")
        weights = np.array([1.0, np.nan, 1.0])
        result = _bin_vector(x, bins_obj, weight=weights)
        assert isinstance(result, pd.DataFrame)


class TestBinCut:
    def test_basic(self):
        x = np.array([1.0, 3.0, 7.0])
        bins_obj = _Bins(np.array([0, 5, 10.0]), "right")
        result = _bin_cut(x, bins_obj)
        assert len(result) == 3
        assert result[0] == 1  # 1-based
        assert result[2] == 2

    def test_left_closed(self):
        x = np.array([0.0, 5.0, 10.0])
        bins_obj = _Bins(np.array([0, 5, 10.0]), "left")
        result = _bin_cut(x, bins_obj)
        assert len(result) == 3


class TestBinLoc:
    def test_basic(self):
        breaks = np.array([0.0, 5.0, 10.0])
        idx = np.array([1, 2])  # 1-based
        result = _bin_loc(breaks, idx)
        assert result["mid"][0] == pytest.approx(2.5)
        assert result["mid"][1] == pytest.approx(7.5)
        assert result["length"][0] == pytest.approx(5.0)


class TestDualParam:
    def test_none(self):
        result = _dual_param(None)
        assert result == {"x": None, "y": None}

    def test_dict(self):
        result = _dual_param({"x": 1, "y": 2})
        assert result == {"x": 1, "y": 2}

    def test_scalar(self):
        result = _dual_param(5)
        assert result == {"x": 5, "y": 5}

    def test_list(self):
        result = _dual_param([3, 4])
        assert result == {"x": 3, "y": 4}

    def test_custom_default(self):
        result = _dual_param(None, {"x": 10, "y": 20})
        assert result == {"x": 10, "y": 20}


# ============================================================================
# Summary function tests
# ============================================================================

class TestMeanClBoot:
    def test_basic(self):
        result = mean_cl_boot([1, 2, 3, 4, 5], B=100)
        assert isinstance(result, pd.DataFrame)
        assert "y" in result.columns
        assert "ymin" in result.columns
        assert "ymax" in result.columns
        assert result["ymin"].iloc[0] <= result["y"].iloc[0]
        assert result["y"].iloc[0] <= result["ymax"].iloc[0]

    def test_empty(self):
        result = mean_cl_boot([])
        assert np.isnan(result["y"].iloc[0])


class TestMeanClNormal:
    def test_basic(self):
        result = mean_cl_normal([1, 2, 3, 4, 5])
        assert isinstance(result, pd.DataFrame)
        assert abs(result["y"].iloc[0] - 3.0) < 1e-10
        assert result["ymin"].iloc[0] < result["y"].iloc[0]
        assert result["y"].iloc[0] < result["ymax"].iloc[0]

    def test_empty(self):
        result = mean_cl_normal([])
        assert np.isnan(result["y"].iloc[0])


class TestMeanSdl:
    def test_basic(self):
        result = mean_sdl([1, 2, 3, 4, 5])
        assert isinstance(result, pd.DataFrame)
        assert abs(result["y"].iloc[0] - 3.0) < 1e-10

    def test_mult(self):
        r1 = mean_sdl([1, 2, 3, 4, 5], mult=1.0)
        r2 = mean_sdl([1, 2, 3, 4, 5], mult=2.0)
        # Wider interval with higher mult
        assert (r2["ymax"].iloc[0] - r2["ymin"].iloc[0]) > (r1["ymax"].iloc[0] - r1["ymin"].iloc[0])

    def test_empty(self):
        result = mean_sdl([])
        assert np.isnan(result["y"].iloc[0])


class TestMeanSeExtended:
    def test_single_value(self):
        result = mean_se([5])
        assert result["y"].iloc[0] == pytest.approx(5.0)
        # SE is 0 for single value
        assert result["ymin"].iloc[0] == result["ymax"].iloc[0]

    def test_empty(self):
        result = mean_se([])
        assert np.isnan(result["y"].iloc[0])

    def test_with_nan(self):
        result = mean_se([1, 2, np.nan, 4, 5])
        assert abs(result["y"].iloc[0] - 3.0) < 1e-10

    def test_mult(self):
        r1 = mean_se([1, 2, 3, 4, 5], mult=1.0)
        r2 = mean_se([1, 2, 3, 4, 5], mult=2.0)
        spread1 = r1["ymax"].iloc[0] - r1["ymin"].iloc[0]
        spread2 = r2["ymax"].iloc[0] - r2["ymin"].iloc[0]
        assert spread2 == pytest.approx(spread1 * 2.0)


class TestMedianHilowExtended:
    def test_confidence(self):
        r1 = median_hilow(np.arange(100), confidence=0.5)
        r2 = median_hilow(np.arange(100), confidence=0.95)
        # Wider for higher confidence
        spread1 = r1["ymax"].iloc[0] - r1["ymin"].iloc[0]
        spread2 = r2["ymax"].iloc[0] - r2["ymin"].iloc[0]
        assert spread2 > spread1

    def test_empty(self):
        result = median_hilow([])
        assert np.isnan(result["y"].iloc[0])


class TestMakeSummaryFun:
    def test_default(self):
        fun = _make_summary_fun()
        df = pd.DataFrame({"y": [1, 2, 3, 4, 5]})
        result = fun(df)
        assert "y" in result.columns

    def test_fun_data_string(self):
        fun = _make_summary_fun(fun_data="mean_se")
        df = pd.DataFrame({"y": [1, 2, 3, 4, 5]})
        result = fun(df)
        assert abs(result["y"].iloc[0] - 3.0) < 1e-10

    def test_fun_data_callable(self):
        fun = _make_summary_fun(fun_data=mean_se)
        df = pd.DataFrame({"y": [1, 2, 3, 4, 5]})
        result = fun(df)
        assert "y" in result.columns

    def test_fun_min_max(self):
        fun = _make_summary_fun(fun=np.mean, fun_min=np.min, fun_max=np.max)
        df = pd.DataFrame({"y": [1, 2, 3, 4, 5]})
        result = fun(df)
        assert result["ymin"].iloc[0] == 1.0
        assert result["y"].iloc[0] == 3.0
        assert result["ymax"].iloc[0] == 5.0

    def test_fun_string_lookup(self):
        fun = _make_summary_fun(fun_data="median_hilow")
        df = pd.DataFrame({"y": [1, 2, 3, 4, 5]})
        result = fun(df)
        assert abs(result["y"].iloc[0] - 3.0) < 1e-10


# ============================================================================
# Density helper tests
# ============================================================================

class TestPrecomputeBw:
    def test_numeric(self):
        result = _precompute_bw(np.arange(100, dtype=float), 0.5)
        assert result == 0.5

    def test_nrd0(self):
        x = np.random.normal(0, 1, 100)
        result = _precompute_bw(x, "nrd0")
        assert result > 0

    def test_nrd(self):
        x = np.random.normal(0, 1, 100)
        result = _precompute_bw(x, "nrd")
        assert result > 0

    def test_sj(self):
        x = np.random.normal(0, 1, 100)
        result = _precompute_bw(x, "sj")
        assert result > 0

    def test_ucv(self):
        x = np.random.normal(0, 1, 100)
        result = _precompute_bw(x, "ucv")
        assert result > 0


class TestComputeDensity:
    def test_basic(self):
        x = np.random.normal(0, 1, 100)
        result = _compute_density(x, from_=-3, to=3)
        assert "x" in result.columns
        assert "density" in result.columns
        assert all(result["density"] >= 0)

    def test_with_weights(self):
        x = np.random.normal(0, 1, 100)
        w = np.ones(100)
        result = _compute_density(x, w=w, from_=-3, to=3)
        assert len(result) > 0

    def test_small_sample(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _compute_density(np.array([1.0]), from_=0, to=2)
        assert "density" in result.columns

    def test_with_bounds(self):
        x = np.random.uniform(0, 1, 100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _compute_density(x, from_=0, to=1, bounds=(0, 1))
        assert len(result) > 0


class TestWecdf:
    def test_basic(self):
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        ecdf_fn = _wecdf(x)
        assert ecdf_fn(np.array([0.0]))[0] == 0.0
        assert ecdf_fn(np.array([6.0]))[0] == 1.0
        assert ecdf_fn(np.array([3.0]))[0] == pytest.approx(0.6)

    def test_weighted(self):
        x = np.array([1, 2, 3], dtype=float)
        w = np.array([1, 2, 1], dtype=float)
        ecdf_fn = _wecdf(x, w)
        result = ecdf_fn(np.array([3.0]))
        assert result[0] == pytest.approx(1.0)

    def test_single_weight(self):
        x = np.array([1, 2, 3], dtype=float)
        w = np.array([1.0])  # broadcasted
        ecdf_fn = _wecdf(x, w)
        assert ecdf_fn(np.array([3.0]))[0] == pytest.approx(1.0)


class TestDensitybin:
    def test_basic(self):
        x = np.random.normal(0, 1, 50)
        result = _densitybin(x, binwidth=0.5)
        assert "x" in result.columns
        assert "bin" in result.columns
        assert "bincenter" in result.columns

    def test_with_weights(self):
        x = np.arange(10, dtype=float)
        w = np.ones(10) * 2
        result = _densitybin(x, weight=w, binwidth=2.0)
        assert len(result) == 10

    def test_all_nan(self):
        x = np.array([np.nan, np.nan, np.nan])
        result = _densitybin(x)
        assert result.empty


class TestContourBreaks:
    def test_explicit_breaks(self):
        result = _contour_breaks([0, 10], breaks=[2, 4, 6, 8])
        np.testing.assert_array_equal(result, [2, 4, 6, 8])

    def test_callable_breaks(self):
        result = _contour_breaks([0, 10], breaks=lambda x: [2, 5, 8])
        np.testing.assert_array_equal(result, [2, 5, 8])

    def test_bins(self):
        result = _contour_breaks([0, 10], bins=5)
        assert len(result) >= 2

    def test_single_bin(self):
        result = _contour_breaks([0, 10], bins=1)
        np.testing.assert_array_equal(result, [0, 10])

    def test_binwidth(self):
        result = _contour_breaks([0, 10], binwidth=2.0)
        assert len(result) >= 2


class TestHexBinwidth:
    def test_basic(self):
        class MockScales:
            pass
        scales = MockScales()
        result = _hex_binwidth(10, scales)
        assert len(result) == 2

    def test_with_dimension(self):
        class MockScale:
            def dimension(self):
                return [0, 100]
        class MockScales:
            x = MockScale()
            y = MockScale()
        scales = MockScales()
        result = _hex_binwidth(10, scales)
        assert result[0] == pytest.approx(10.0)


class TestHexBinSummarise:
    def test_basic(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 50)
        y = np.random.normal(0, 1, 50)
        z = np.ones(50)
        result = _hex_bin_summarise(x, y, z, (0.5, 0.5))
        assert "x" in result.columns
        assert "y" in result.columns
        assert "value" in result.columns

    def test_empty(self):
        result = _hex_bin_summarise(
            np.array([]), np.array([]), np.array([]),
            (0.5, 0.5)
        )
        assert len(result) == 0


# ============================================================================
# Base Stat class tests
# ============================================================================

class TestStat:
    def test_is_stat(self):
        assert is_stat(Stat) is True
        assert is_stat(Stat()) is True
        assert is_stat(StatBin) is True
        assert is_stat("not_stat") is False
        assert is_stat(42) is False

    def test_base_compute_group_raises(self):
        s = Stat()
        with pytest.raises(NotImplementedError):
            s.compute_group(pd.DataFrame(), None)

    def test_setup_params(self):
        s = Stat()
        result = s.setup_params(pd.DataFrame(), {"a": 1})
        assert result == {"a": 1}

    def test_setup_data(self):
        s = Stat()
        df = pd.DataFrame({"x": [1, 2]})
        result = s.setup_data(df, {})
        assert list(result["x"]) == [1, 2]

    def test_finish_layer(self):
        s = Stat()
        df = pd.DataFrame({"x": [1, 2]})
        result = s.finish_layer(df, {})
        assert list(result["x"]) == [1, 2]

    def test_parameters(self):
        s = StatBin()
        params = s.parameters(extra=False)
        assert isinstance(params, list)
        params_extra = s.parameters(extra=True)
        assert "na_rm" in params_extra

    def test_aesthetics(self):
        s = StatBin()
        aes = s.aesthetics()
        assert "group" in aes

    def test_compute_panel_empty(self):
        s = StatBin()
        result = s.compute_panel(pd.DataFrame(), None)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_compute_panel_no_group(self):
        s = StatBin()
        df = pd.DataFrame({"x": np.random.normal(0, 1, 50)})
        result = s.compute_panel(df, None)
        assert "count" in result.columns


# ============================================================================
# StatIdentity tests
# ============================================================================

class TestStatIdentity:
    def test_compute_layer(self):
        s = StatIdentity()
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = s.compute_layer(df, {}, None)
        assert list(result["x"]) == [1, 2, 3]


# ============================================================================
# StatBin extended tests
# ============================================================================

class TestStatBinExtended:
    def test_setup_params_x_only(self):
        sb = StatBin()
        df = pd.DataFrame({"x": [1, 2, 3]})
        params = sb.setup_params(df, {})
        assert "bins" in params or "binwidth" in params or "breaks" in params

    def test_setup_params_y_only(self):
        sb = StatBin()
        df = pd.DataFrame({"y": [1, 2, 3]})
        params = sb.setup_params(df, {})
        assert "flipped_aes" in params

    def test_setup_params_no_xy_errors(self):
        sb = StatBin()
        with pytest.raises(Exception):
            sb.setup_params(pd.DataFrame({"z": [1]}), {})

    def test_setup_params_both_xy_errors(self):
        sb = StatBin()
        with pytest.raises(Exception):
            sb.setup_params(pd.DataFrame({"x": [1], "y": [2]}), {})

    def test_compute_group_with_binwidth(self):
        sb = StatBin()
        df = pd.DataFrame({"x": np.arange(100, dtype=float)})
        result = sb.compute_group(df, None, binwidth=10.0)
        assert len(result) > 0
        assert result["count"].sum() == pytest.approx(100)

    def test_compute_group_with_breaks(self):
        sb = StatBin()
        df = pd.DataFrame({"x": np.arange(10, dtype=float)})
        result = sb.compute_group(df, None, breaks=[0, 3, 6, 10])
        assert len(result) == 3

    def test_compute_group_with_pad(self):
        sb = StatBin()
        df = pd.DataFrame({"x": np.arange(10, dtype=float)})
        result = sb.compute_group(df, None, bins=3, pad=True)
        assert len(result) > 3  # padded

    def test_compute_group_drop_all(self):
        sb = StatBin()
        df = pd.DataFrame({"x": np.array([1.0, 2.0, 3.0])})
        result = sb.compute_group(df, None, breaks=[0, 1, 5, 10], drop="all")
        # Empty bins are dropped
        assert all(result["count"] > 0)

    def test_compute_group_drop_extremes(self):
        sb = StatBin()
        df = pd.DataFrame({"x": np.array([5.0, 5.5, 6.0])})
        result = sb.compute_group(df, None, breaks=[0, 2, 4, 6, 8, 10], drop="extremes")
        assert isinstance(result, pd.DataFrame)

    def test_compute_group_drop_bool_true(self):
        sb = StatBin()
        df = pd.DataFrame({"x": np.arange(10, dtype=float)})
        params = sb.setup_params(df, {"drop": True})
        assert params["drop"] == "all"

    def test_compute_group_drop_bool_false(self):
        sb = StatBin()
        df = pd.DataFrame({"x": np.arange(10, dtype=float)})
        params = sb.setup_params(df, {"drop": False})
        assert params["drop"] == "none"

    def test_compute_group_with_weight(self):
        sb = StatBin()
        df = pd.DataFrame({"x": np.arange(10, dtype=float), "weight": np.ones(10) * 2})
        result = sb.compute_group(df, None, bins=5)
        assert result["count"].sum() == pytest.approx(20)

    def test_compute_group_flipped(self):
        sb = StatBin()
        df = pd.DataFrame({"x": np.arange(10, dtype=float)})
        result = sb.compute_group(df, None, bins=5, flipped_aes=True)
        assert "y" in result.columns or "x" in result.columns

    def test_compute_group_closed_left(self):
        sb = StatBin()
        df = pd.DataFrame({"x": np.arange(10, dtype=float)})
        result = sb.compute_group(df, None, bins=5, closed="left")
        assert len(result) > 0

    def test_compute_group_center(self):
        sb = StatBin()
        df = pd.DataFrame({"x": np.arange(10, dtype=float)})
        result = sb.compute_group(df, None, binwidth=2.0, center=5.0)
        assert len(result) > 0


# ============================================================================
# StatCount extended tests
# ============================================================================

class TestStatCountExtended:
    def test_setup_params(self):
        sc = StatCount()
        df = pd.DataFrame({"x": ["a", "b", "a"]})
        params = sc.setup_params(df, {})
        assert "width" in params

    def test_setup_params_errors_no_xy(self):
        sc = StatCount()
        with pytest.raises(Exception):
            sc.setup_params(pd.DataFrame({"z": [1]}), {})

    def test_setup_params_errors_both_xy(self):
        sc = StatCount()
        with pytest.raises(Exception):
            sc.setup_params(pd.DataFrame({"x": [1], "y": [2]}), {})

    def test_with_width(self):
        sc = StatCount()
        df = pd.DataFrame({"x": ["a", "b", "a"]})
        result = sc.compute_group(df, None, width=0.5)
        assert result["width"].iloc[0] == 0.5

    def test_with_weight(self):
        sc = StatCount()
        df = pd.DataFrame({"x": ["a", "b", "a"], "weight": [1.0, 2.0, 3.0]})
        result = sc.compute_group(df, None)
        # a: 1+3=4, b: 2
        assert result["count"].sum() == pytest.approx(6.0)

    def test_flipped(self):
        sc = StatCount()
        df = pd.DataFrame({"x": ["a", "b", "a"]})
        result = sc.compute_group(df, None, flipped_aes=True)
        assert isinstance(result, pd.DataFrame)


# ============================================================================
# StatDensity extended tests
# ============================================================================

class TestStatDensityExtended:
    def test_setup_params(self):
        sd = StatDensity()
        df = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        params = sd.setup_params(df, {})
        assert "flipped_aes" in params

    def test_trim(self):
        sd = StatDensity()
        df = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        result = sd.compute_group(df, None, trim=True)
        assert result["x"].min() >= df["x"].min() - 1e-10
        assert result["x"].max() <= df["x"].max() + 1e-10

    def test_no_trim(self):
        sd = StatDensity()
        df = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        result = sd.compute_group(df, None, trim=False)
        assert len(result) > 0

    def test_different_bw(self):
        sd = StatDensity()
        df = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        r1 = sd.compute_group(df, None, bw="nrd0")
        r2 = sd.compute_group(df, None, bw=0.1)
        assert len(r1) == len(r2)

    def test_adjust(self):
        sd = StatDensity()
        df = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        result = sd.compute_group(df, None, adjust=2.0)
        assert len(result) > 0

    def test_n_param(self):
        sd = StatDensity()
        df = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        result = sd.compute_group(df, None, n=64)
        assert len(result) == 64

    def test_flipped(self):
        sd = StatDensity()
        df = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        result = sd.compute_group(df, None, flipped_aes=True)
        assert len(result) > 0

    def test_output_columns(self):
        sd = StatDensity()
        df = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        result = sd.compute_group(df, None)
        for col in ["x", "density", "scaled", "ndensity", "count", "n"]:
            assert col in result.columns

    def test_with_weight(self):
        sd = StatDensity()
        df = pd.DataFrame({"x": np.random.normal(0, 1, 100), "weight": np.ones(100)})
        result = sd.compute_group(df, None)
        assert len(result) > 0


# ============================================================================
# StatSmooth extended tests
# ============================================================================

class TestStatSmoothExtended:
    def test_setup_params_auto_gam_for_large_n(self):
        """R stat-smooth.R:14-20 — auto at n >= 1000 per group selects
        ``gam``; only if mgcv (statsmodels.gam here) is unavailable does
        it fall back to ``loess``."""
        ss = StatSmooth()
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.arange(2000, dtype=float),
            "y": np.arange(2000, dtype=float),
            "group": 1,
            "PANEL": 1,
        })
        params = ss.setup_params(df, {"method": "auto"})
        assert params["method"] == "gam"

    def test_setup_params_auto_loess(self):
        ss = StatSmooth()
        df = pd.DataFrame({
            "x": np.arange(50, dtype=float),
            "y": np.arange(50, dtype=float),
            "group": 1,
            "PANEL": 1,
        })
        params = ss.setup_params(df, {"method": None})
        assert params["method"] == "loess"

    def test_lm(self):
        ss = StatSmooth()
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2 * x + 1 + np.random.normal(0, 0.5, 50)
        df = pd.DataFrame({"x": x, "y": y})
        result = ss.compute_group(df, None, method="lm", n=20)
        assert "x" in result.columns
        assert "y" in result.columns
        assert "ymin" in result.columns
        assert "ymax" in result.columns
        assert len(result) == 20

    def test_lm_no_se(self):
        ss = StatSmooth()
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2 * x + 1 + np.random.normal(0, 0.5, 50)
        df = pd.DataFrame({"x": x, "y": y})
        result = ss.compute_group(df, None, method="lm", se=False, n=20)
        assert all(np.isnan(result["se"]))

    def test_lowess(self):
        ss = StatSmooth()
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = np.sin(x) + np.random.normal(0, 0.1, 50)
        df = pd.DataFrame({"x": x, "y": y})
        result = ss.compute_group(df, None, method="loess", n=20)
        assert len(result) == 20

    def test_glm(self):
        ss = StatSmooth()
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2 * x + 1
        df = pd.DataFrame({"x": x, "y": y})
        result = ss.compute_group(df, None, method="glm", n=20)
        assert len(result) == 20

    def test_insufficient_data(self):
        ss = StatSmooth()
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        result = ss.compute_group(df, None, method="lm")
        assert result.empty

    def test_fullrange(self):
        ss = StatSmooth()
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2 * x + 1
        df = pd.DataFrame({"x": x, "y": y})
        result = ss.compute_group(df, None, method="lm", fullrange=True, n=20)
        assert len(result) == 20

    def test_xseq(self):
        ss = StatSmooth()
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2 * x + 1
        df = pd.DataFrame({"x": x, "y": y})
        xseq = np.array([2, 4, 6, 8], dtype=float)
        result = ss.compute_group(df, None, method="lm", xseq=xseq)
        assert len(result) == 4


# ============================================================================
# StatBoxplot extended tests
# ============================================================================

class TestStatBoxplotExtended:
    def test_setup_params(self):
        sb = StatBoxplot()
        df = pd.DataFrame({"y": np.random.normal(0, 1, 50), "x": [1] * 50})
        params = sb.setup_params(df, {})
        assert "width" in params

    def test_setup_data_no_x(self):
        sb = StatBoxplot()
        df = pd.DataFrame({"y": np.random.normal(0, 1, 50)})
        result = sb.setup_data(df, {"flipped_aes": False})
        assert "x" in result.columns

    def test_outliers(self):
        sb = StatBoxplot()
        # Create data with clear outliers
        y = np.concatenate([np.random.normal(0, 1, 48), [100, -100]])
        df = pd.DataFrame({"y": y, "x": [1] * 50})
        result = sb.compute_group(df, None)
        outliers = result["outliers"].iloc[0]
        assert len(outliers) >= 2

    def test_coef(self):
        sb = StatBoxplot()
        y = np.random.normal(0, 1, 100)
        df = pd.DataFrame({"y": y, "x": [1] * 100})
        # Very large coef -> no outliers
        result = sb.compute_group(df, None, coef=100)
        assert len(result["outliers"].iloc[0]) == 0

    def test_notch(self):
        sb = StatBoxplot()
        y = np.random.normal(0, 1, 50)
        df = pd.DataFrame({"y": y, "x": [1] * 50})
        result = sb.compute_group(df, None)
        assert "notchupper" in result.columns
        assert "notchlower" in result.columns
        assert result["notchlower"].iloc[0] < result["notchupper"].iloc[0]

    def test_relvarwidth(self):
        sb = StatBoxplot()
        y = np.random.normal(0, 1, 50)
        df = pd.DataFrame({"y": y, "x": [1] * 50})
        result = sb.compute_group(df, None)
        assert result["relvarwidth"].iloc[0] == pytest.approx(np.sqrt(50))

    def test_empty_y(self):
        sb = StatBoxplot()
        df = pd.DataFrame({"y": pd.array([], dtype=float), "x": pd.array([], dtype=float)})
        result = sb.compute_group(df, None)
        assert result.empty

    def test_flipped(self):
        sb = StatBoxplot()
        y = np.random.normal(0, 1, 50)
        df = pd.DataFrame({"y": y, "x": [1] * 50})
        result = sb.compute_group(df, None, flipped_aes=True)
        assert isinstance(result, pd.DataFrame)


# ============================================================================
# StatSummary tests
# ============================================================================

class TestStatSummaryExtended:
    def test_setup_params(self):
        ss = StatSummary()
        df = pd.DataFrame({"x": [1, 1, 2, 2], "y": [1, 2, 3, 4]})
        params = ss.setup_params(df, {})
        assert "fun" in params

    def test_compute_panel_default(self):
        ss = StatSummary()
        df = pd.DataFrame({"x": [1, 1, 2, 2], "y": [1, 2, 3, 4], "group": [1, 1, 1, 1]})
        result = ss.compute_panel(df, None)
        assert "y" in result.columns

    def test_with_fun_data(self):
        ss = StatSummary()
        df = pd.DataFrame({"x": [1, 1, 2, 2], "y": [1, 2, 3, 4], "group": [1, 1, 1, 1]})
        fun = _make_summary_fun(fun_data="mean_se")
        result = ss.compute_panel(df, None, fun=fun)
        assert "y" in result.columns

    def test_with_groups(self):
        ss = StatSummary()
        df = pd.DataFrame({
            "x": [1, 1, 2, 2, 1, 1, 2, 2],
            "y": [1, 2, 3, 4, 5, 6, 7, 8],
            "group": [1, 1, 1, 1, 2, 2, 2, 2],
        })
        result = ss.compute_panel(df, None)
        assert len(result) == 4  # 2 groups x 2 x-values


# ============================================================================
# StatSummaryBin tests
# ============================================================================

class TestStatSummaryBin:
    def test_basic(self):
        ss = StatSummaryBin()
        df = pd.DataFrame({"x": np.arange(100, dtype=float), "y": np.random.normal(0, 1, 100)})
        result = ss.compute_group(df, None, bins=5)
        assert "y" in result.columns
        assert "x" in result.columns

    def test_setup_params(self):
        ss = StatSummaryBin()
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        params = ss.setup_params(df, {})
        assert "fun" in params


# ============================================================================
# StatSummary2d tests
# ============================================================================

class TestStatSummary2d:
    def test_basic(self):
        ss = StatSummary2d()
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 100),
            "y": np.random.normal(0, 1, 100),
            "z": np.random.normal(0, 1, 100),
        })
        result = ss.compute_group(df, None, bins=5)
        assert "x" in result.columns
        assert "y" in result.columns
        assert "value" in result.columns

    def test_fun_string(self):
        ss = StatSummary2d()
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 100),
            "y": np.random.normal(0, 1, 100),
            "z": np.random.normal(0, 1, 100),
        })
        for fun_name in ["mean", "sum", "median", "min", "max", "count"]:
            result = ss.compute_group(df, None, bins=5, fun=fun_name)
            assert len(result) > 0

    def test_drop_false(self):
        ss = StatSummary2d()
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 50),
            "y": np.random.normal(0, 1, 50),
            "z": np.random.normal(0, 1, 50),
        })
        result = ss.compute_group(df, None, bins=3, drop=False)
        assert isinstance(result, pd.DataFrame)


# ============================================================================
# StatSummaryHex tests
# ============================================================================

class TestStatSummaryHex:
    def test_basic(self):
        ss = StatSummaryHex()
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 50),
            "y": np.random.normal(0, 1, 50),
            "z": np.random.normal(0, 1, 50),
        })
        result = ss.compute_group(df, None, binwidth=(0.5, 0.5))
        assert "x" in result.columns

    def test_fun_string(self):
        ss = StatSummaryHex()
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 50),
            "y": np.random.normal(0, 1, 50),
            "z": np.random.normal(0, 1, 50),
        })
        result = ss.compute_group(df, None, bins=5, fun="sum")
        assert len(result) > 0


# ============================================================================
# StatFunction tests
# ============================================================================

class TestStatFunction:
    def test_basic(self):
        sf = StatFunction()
        df = pd.DataFrame({"x": [0]})
        result = sf.compute_group(df, {"x": None}, fun=np.sin, xlim=(0, 2 * np.pi), n=50)
        assert "x" in result.columns
        assert "y" in result.columns
        assert len(result) == 50

    def test_no_fun_raises(self):
        sf = StatFunction()
        with pytest.raises(Exception):
            sf.compute_group(pd.DataFrame(), None)

    def test_with_args(self):
        def my_func(x, a=1):
            return a * x
        sf = StatFunction()
        result = sf.compute_group(pd.DataFrame({"x": [0]}), {"x": None}, fun=my_func, xlim=(0, 10), n=20, args={"a": 3})
        assert result["y"].iloc[-1] == pytest.approx(30.0, rel=0.1)


# ============================================================================
# StatEcdf tests
# ============================================================================

class TestStatEcdf:
    def test_basic(self):
        se = StatEcdf()
        df = pd.DataFrame({"x": np.arange(10, dtype=float)})
        result = se.compute_group(df, None)
        assert "y" in result.columns
        assert "ecdf" in result.columns

    def test_no_pad(self):
        se = StatEcdf()
        df = pd.DataFrame({"x": np.arange(10, dtype=float)})
        result = se.compute_group(df, None, pad=False)
        assert not np.isinf(result["x"]).any()

    def test_with_n(self):
        se = StatEcdf()
        df = pd.DataFrame({"x": np.arange(100, dtype=float)})
        result = se.compute_group(df, None, n=20, pad=False)
        assert len(result) == 20

    def test_setup_params(self):
        se = StatEcdf()
        df = pd.DataFrame({"x": [1, 2, 3]})
        params = se.setup_params(df, {})
        assert "flipped_aes" in params

    def test_ecdf_values(self):
        se = StatEcdf()
        df = pd.DataFrame({"x": np.array([1, 2, 3, 4, 5], dtype=float)})
        result = se.compute_group(df, None, pad=False)
        # Last value should have ecdf=1
        assert result["ecdf"].iloc[-1] == pytest.approx(1.0)

    def test_flipped(self):
        se = StatEcdf()
        df = pd.DataFrame({"x": np.arange(10, dtype=float)})
        result = se.compute_group(df, None, flipped_aes=True)
        assert isinstance(result, pd.DataFrame)


# ============================================================================
# StatQq tests
# ============================================================================

class TestStatQq:
    def test_basic(self):
        sq = StatQq()
        df = pd.DataFrame({"sample": np.random.normal(0, 1, 50)})
        result = sq.compute_group(df, None)
        assert "sample" in result.columns
        assert "theoretical" in result.columns
        assert len(result) == 50

    def test_custom_distribution(self):
        from scipy import stats
        sq = StatQq()
        df = pd.DataFrame({"sample": np.random.exponential(1, 50)})
        result = sq.compute_group(df, None, distribution=stats.expon)
        assert len(result) == 50

    def test_with_dparams(self):
        from scipy import stats
        sq = StatQq()
        df = pd.DataFrame({"sample": np.random.normal(5, 2, 50)})
        result = sq.compute_group(df, None, dparams={"loc": 5, "scale": 2})
        assert len(result) == 50


# ============================================================================
# StatQqLine tests
# ============================================================================

class TestStatQqLine:
    def test_basic(self):
        sql = StatQqLine()
        df = pd.DataFrame({"sample": np.random.normal(0, 1, 50)})
        result = sql.compute_group(df, None)
        assert "x" in result.columns
        assert "y" in result.columns
        assert "slope" in result.columns
        assert "intercept" in result.columns
        assert len(result) == 2

    def test_fullrange(self):
        sql = StatQqLine()
        df = pd.DataFrame({"sample": np.random.normal(0, 1, 50)})
        result = sql.compute_group(df, None, fullrange=True)
        assert len(result) == 2


# ============================================================================
# StatBin2d tests
# ============================================================================

class TestStatBin2d:
    def test_basic(self):
        sb = StatBin2d()
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 100),
            "y": np.random.normal(0, 1, 100),
        })
        result = sb.compute_group(df, None, bins=5)
        assert "count" in result.columns
        assert "ncount" in result.columns
        assert "density" in result.columns

    def test_drop(self):
        sb = StatBin2d()
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 50),
            "y": np.random.normal(0, 1, 50),
        })
        result = sb.compute_group(df, None, bins=3, drop=True)
        assert all(result["count"] > 0)


# ============================================================================
# StatBinhex tests
# ============================================================================

class TestStatBinhex:
    def test_basic(self):
        sb = StatBinhex()
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 50),
            "y": np.random.normal(0, 1, 50),
        })
        result = sb.compute_group(df, None, binwidth=(0.5, 0.5))
        assert "count" in result.columns or "x" in result.columns

    def test_with_bins_default(self):
        sb = StatBinhex()
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 50),
            "y": np.random.normal(0, 1, 50),
        })
        result = sb.compute_group(df, None, bins=5)
        assert isinstance(result, pd.DataFrame)


# ============================================================================
# StatDensity2d tests
# ============================================================================

class TestStatDensity2d:
    def test_basic(self):
        sd = StatDensity2d()
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 50),
            "y": np.random.normal(0, 1, 50),
            "group": 1,
        })
        result = sd.compute_group(df, None, n=20)
        assert "density" in result.columns
        assert len(result) == 20 * 20  # n x n grid

    def test_adjust_scalar(self):
        sd = StatDensity2d()
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 50),
            "y": np.random.normal(0, 1, 50),
            "group": 1,
        })
        result = sd.compute_group(df, None, n=10, adjust=2.0)
        assert len(result) > 0


class TestStatDensity2dFilled:
    def test_subclass(self):
        sd = StatDensity2dFilled()
        assert sd.contour_type == "bands"
        assert sd.default_aes.get("fill") is None


# ============================================================================
# StatEllipse tests
# ============================================================================

class TestStatEllipse:
    def test_basic(self):
        se = StatEllipse()
        np.random.seed(42)
        x = np.random.normal(0, 1, 50)
        y = np.random.normal(0, 1, 50)
        df = pd.DataFrame({"x": x, "y": y})
        result = se.compute_group(df, None)
        assert "x" in result.columns
        assert "y" in result.columns

    def test_type_norm(self):
        se = StatEllipse()
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 50),
            "y": np.random.normal(0, 1, 50),
        })
        result = se.compute_group(df, None, type="norm")
        assert len(result) > 0

    def test_type_euclid(self):
        se = StatEllipse()
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 50),
            "y": np.random.normal(0, 1, 50),
        })
        result = se.compute_group(df, None, type="euclid")
        assert len(result) > 0

    def test_too_few_points(self):
        se = StatEllipse()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = se.compute_group(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), None)
        assert np.isnan(result["x"].iloc[0])

    def test_segments(self):
        se = StatEllipse()
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 50),
            "y": np.random.normal(0, 1, 50),
        })
        result = se.compute_group(df, None, segments=20)
        assert len(result) == 21  # segments + 1


# ============================================================================
# StatUnique tests
# ============================================================================

class TestStatUnique:
    def test_basic(self):
        su = StatUnique()
        df = pd.DataFrame({"x": [1, 2, 1, 3, 2], "y": [1, 2, 1, 3, 2]})
        result = su.compute_panel(df, None)
        assert len(result) == 3


# ============================================================================
# StatSum tests
# ============================================================================

class TestStatSum:
    def test_basic(self):
        ss = StatSum()
        df = pd.DataFrame({
            "x": [1, 1, 2, 2, 2],
            "y": [1, 1, 2, 2, 2],
        })
        result = ss.compute_panel(df, None)
        assert "n" in result.columns
        assert "prop" in result.columns

    def test_with_weight(self):
        ss = StatSum()
        df = pd.DataFrame({
            "x": [1, 1, 2],
            "y": [1, 1, 2],
            "weight": [1, 2, 3],
        })
        result = ss.compute_panel(df, None)
        assert "n" in result.columns

    def test_with_groups(self):
        ss = StatSum()
        df = pd.DataFrame({
            "x": [1, 1, 2, 2],
            "y": [1, 1, 2, 2],
            "group": [1, 1, 2, 2],
        })
        result = ss.compute_panel(df, None)
        assert "prop" in result.columns


# ============================================================================
# StatYdensity tests
# ============================================================================

class TestStatYdensity:
    def test_basic(self):
        sy = StatYdensity()
        df = pd.DataFrame({"x": [1] * 50, "y": np.random.normal(0, 1, 50)})
        result = sy.compute_group(df, None)
        assert "density" in result.columns
        assert len(result) > 0

    def test_trim(self):
        sy = StatYdensity()
        df = pd.DataFrame({"x": [1] * 50, "y": np.random.normal(0, 1, 50)})
        result = sy.compute_group(df, None, trim=True)
        assert len(result) > 0

    def test_no_trim(self):
        sy = StatYdensity()
        df = pd.DataFrame({"x": [1] * 50, "y": np.random.normal(0, 1, 50)})
        result = sy.compute_group(df, None, trim=False)
        assert len(result) > 0

    def test_too_few_drop(self):
        sy = StatYdensity()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sy.compute_group(pd.DataFrame({"x": [1], "y": [1.0]}), None, drop=True)
        assert result.empty

    def test_too_few_no_drop(self):
        sy = StatYdensity()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sy.compute_group(pd.DataFrame({"x": [1], "y": [1.0]}), None, drop=False)
        assert len(result) == 1

    def test_quantiles(self):
        sy = StatYdensity()
        df = pd.DataFrame({"x": [1] * 50, "y": np.random.normal(0, 1, 50)})
        result = sy.compute_group(df, None, quantiles=(0.25, 0.5, 0.75))
        # Should have extra rows for quantiles
        assert "quantile" in result.columns or len(result) > 512

    def test_no_quantiles(self):
        sy = StatYdensity()
        df = pd.DataFrame({"x": [1] * 50, "y": np.random.normal(0, 1, 50)})
        result = sy.compute_group(df, None, quantiles=None)
        assert len(result) > 0

    def test_setup_params(self):
        sy = StatYdensity()
        df = pd.DataFrame({"x": [1] * 50, "y": np.random.normal(0, 1, 50)})
        params = sy.setup_params(df, {})
        assert "flipped_aes" in params


# ============================================================================
# StatBindot tests
# ============================================================================

class TestStatBindot:
    def test_setup_params(self):
        sb = StatBindot()
        df = pd.DataFrame({"x": np.arange(10, dtype=float)})
        params = sb.setup_params(df, {})
        assert isinstance(params, dict)

    def test_dotdensity(self):
        sb = StatBindot()
        df = pd.DataFrame({"x": np.random.normal(0, 1, 50)})
        result = sb.compute_group(df, None, binwidth=0.5, method="dotdensity")
        assert len(result) > 0

    def test_histodot(self):
        sb = StatBindot()
        df = pd.DataFrame({"x": np.random.normal(0, 1, 50)})
        result = sb.compute_group(df, None, binwidth=0.5, method="histodot")
        assert len(result) > 0

    def test_binaxis_y(self):
        sb = StatBindot()
        df = pd.DataFrame({"x": [1] * 50, "y": np.random.normal(0, 1, 50)})
        result = sb.compute_group(df, None, binaxis="y", binwidth=0.5)
        assert len(result) > 0


# ============================================================================
# StatAlign tests
# ============================================================================

class TestStatAlign:
    def test_basic(self):
        sa = StatAlign()
        df = pd.DataFrame({"x": np.linspace(0, 10, 20), "y": np.sin(np.linspace(0, 10, 20))})
        result = sa.compute_group(df, None)
        assert "x" in result.columns
        assert "y" in result.columns

    def test_setup_params(self):
        sa = StatAlign()
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        params = sa.setup_params(df, {})
        assert "flipped_aes" in params

    def test_insufficient_data(self):
        sa = StatAlign()
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        result = sa.compute_group(df, None)
        assert result.empty


# ============================================================================
# StatConnect tests
# ============================================================================

class TestStatConnect:
    def test_setup_params_string_connections(self):
        sc = StatConnect()
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        for conn in ["hv", "vh", "mid", "linear"]:
            params = sc.setup_params(df, {"connection": conn})
            assert "connection" in params
            assert isinstance(params["connection"], np.ndarray)

    def test_setup_params_unknown_connection(self):
        sc = StatConnect()
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        with pytest.raises(Exception):
            sc.setup_params(df, {"connection": "unknown"})


# ============================================================================
# StatManual tests
# ============================================================================

class TestStatManual:
    def test_basic(self):
        sm = StatManual()
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = sm.compute_group(df, None)
        assert isinstance(result, pd.DataFrame)


# ============================================================================
# StatQuantile tests
# ============================================================================

class TestStatQuantile:
    def test_basic(self):
        sq = StatQuantile()
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2 * x + np.random.normal(0, 2, 100)
        df = pd.DataFrame({"x": x, "y": y})
        result = sq.compute_group(df, None)
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "x" in result.columns
            assert "y" in result.columns


# ============================================================================
# StatContour tests
# ============================================================================

class TestStatContour:
    def test_setup_params(self):
        sc = StatContour()
        x, y = np.meshgrid(np.arange(5), np.arange(5))
        z = x.ravel() + y.ravel()
        df = pd.DataFrame({"x": x.ravel(), "y": y.ravel(), "z": z.astype(float)})
        params = sc.setup_params(df, {})
        assert "z_range" in params


class TestStatContourFilled:
    def test_setup_params(self):
        sc = StatContourFilled()
        x, y = np.meshgrid(np.arange(5), np.arange(5))
        z = x.ravel() + y.ravel()
        df = pd.DataFrame({"x": x.ravel(), "y": y.ravel(), "z": z.astype(float)})
        params = sc.setup_params(df, {})
        assert "z_range" in params
