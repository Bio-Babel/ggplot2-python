"""Tests for ggplot2_py.fortify — data coercion."""

import pytest
import pandas as pd
from ggplot2_py import fortify
from ggplot2_py._compat import Waiver


class TestFortifyDataFrame:
    """Test fortify with DataFrames."""

    def test_returns_same_df(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = fortify(df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a"]
        assert list(result["a"]) == [1, 2]


class TestFortifyDict:
    """Test fortify with dicts."""

    def test_returns_dataframe(self):
        result = fortify({"a": [1, 2]})
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a"]


class TestFortifyNone:
    """Test fortify with None."""

    def test_returns_waiver(self):
        result = fortify(None)
        assert isinstance(result, Waiver)
