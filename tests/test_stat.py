"""Tests for ggplot2_py.stat — stat computation."""

import pytest
import numpy as np
import pandas as pd
from ggplot2_py import (
    StatBin,
    StatCount,
    StatDensity,
    StatBoxplot,
    mean_se,
    median_hilow,
)


class TestStatBin:
    """Test StatBin compute_group."""

    def test_compute_group_output_shape(self):
        sb = StatBin()
        data = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        result = sb.compute_group(data, scales=None)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] > 0

    def test_compute_group_has_count_column(self):
        sb = StatBin()
        data = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        result = sb.compute_group(data, scales=None)
        assert "count" in result.columns

    def test_compute_group_has_density_column(self):
        sb = StatBin()
        data = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        result = sb.compute_group(data, scales=None)
        assert "density" in result.columns

    def test_compute_group_has_x_xmin_xmax(self):
        sb = StatBin()
        data = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        result = sb.compute_group(data, scales=None)
        assert "x" in result.columns
        assert "xmin" in result.columns
        assert "xmax" in result.columns


class TestStatCount:
    """Test StatCount compute_group."""

    def test_produces_count_column(self):
        sc = StatCount()
        data = pd.DataFrame({"x": ["a", "b", "a", "c", "b", "a"]})
        result = sc.compute_group(data, scales=None)
        assert "count" in result.columns

    def test_correct_number_of_categories(self):
        sc = StatCount()
        data = pd.DataFrame({"x": ["a", "b", "a", "c", "b", "a"]})
        result = sc.compute_group(data, scales=None)
        assert result.shape[0] == 3  # three unique values

    def test_produces_prop_column(self):
        sc = StatCount()
        data = pd.DataFrame({"x": ["a", "b", "c"]})
        result = sc.compute_group(data, scales=None)
        assert "prop" in result.columns


class TestStatDensity:
    """Test StatDensity compute_group."""

    def test_produces_density_curve(self):
        sd = StatDensity()
        data = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        result = sd.compute_group(data, scales=None)
        assert isinstance(result, pd.DataFrame)
        assert "density" in result.columns
        assert result.shape[0] > 0

    def test_density_is_nonnegative(self):
        sd = StatDensity()
        data = pd.DataFrame({"x": np.random.normal(0, 1, 200)})
        result = sd.compute_group(data, scales=None)
        assert (result["density"] >= 0).all()


class TestStatBoxplot:
    """Test StatBoxplot compute_group."""

    def test_produces_quantile_columns(self):
        sbp = StatBoxplot()
        data = pd.DataFrame({"x": [1] * 20, "y": np.random.normal(0, 1, 20)})
        result = sbp.compute_group(data, scales=None)
        for col in ["ymin", "lower", "middle", "upper", "ymax"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_quantile_ordering(self):
        sbp = StatBoxplot()
        data = pd.DataFrame({"x": [1] * 50, "y": np.random.normal(0, 1, 50)})
        result = sbp.compute_group(data, scales=None)
        row = result.iloc[0]
        assert row["ymin"] <= row["lower"]
        assert row["lower"] <= row["middle"]
        assert row["middle"] <= row["upper"]
        assert row["upper"] <= row["ymax"]


class TestMeanSe:
    """Test mean_se summary function."""

    def test_correct_mean(self):
        result = mean_se([1, 2, 3, 4, 5])
        assert abs(result["y"].iloc[0] - 3.0) < 1e-10

    def test_returns_dataframe(self):
        result = mean_se([1, 2, 3, 4, 5])
        assert isinstance(result, pd.DataFrame)

    def test_has_ymin_ymax(self):
        result = mean_se([1, 2, 3, 4, 5])
        assert "ymin" in result.columns
        assert "ymax" in result.columns

    def test_ymin_less_than_ymax(self):
        result = mean_se([1, 2, 3, 4, 5])
        assert result["ymin"].iloc[0] < result["ymax"].iloc[0]


class TestMedianHilow:
    """Test median_hilow summary function."""

    def test_correct_median(self):
        result = median_hilow([1, 2, 3, 4, 5])
        assert abs(result["y"].iloc[0] - 3.0) < 1e-10

    def test_returns_dataframe(self):
        result = median_hilow([1, 2, 3, 4, 5])
        assert isinstance(result, pd.DataFrame)

    def test_has_ymin_ymax(self):
        result = median_hilow([1, 2, 3, 4, 5])
        assert "ymin" in result.columns
        assert "ymax" in result.columns
