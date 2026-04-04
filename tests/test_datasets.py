"""Tests for ggplot2_py.datasets — data loading and integrity."""

import pytest
import pandas as pd
from ggplot2_py.datasets import load_dataset


# Expected shapes for all 11 datasets
EXPECTED_SHAPES = {
    "mpg": (234, 11),
    "diamonds": (53940, 10),
    "economics": (574, 6),
    "economics_long": (2870, 4),
    "faithfuld": (5625, 3),
    "luv_colours": (657, 4),
    "midwest": (437, 28),
    "msleep": (83, 11),
    "presidential": (12, 4),
    "seals": (1155, 4),
    "txhousing": (8602, 9),
}

EXPECTED_COLUMNS = {
    "mpg": [
        "manufacturer", "model", "displ", "year", "cyl",
        "trans", "drv", "cty", "hwy", "fl", "class",
    ],
    "diamonds": [
        "carat", "cut", "color", "clarity", "depth",
        "table", "price", "x", "y", "z",
    ],
    "economics": ["date", "pce", "pop", "psavert", "uempmed", "unemploy"],
    "economics_long": ["date", "variable", "value", "value01"],
    "faithfuld": ["eruptions", "waiting", "density"],
    "luv_colours": ["L", "u", "v", "col"],
    "presidential": ["name", "start", "end", "party"],
    "seals": ["lat", "long", "delta_long", "delta_lat"],
    "txhousing": [
        "city", "year", "month", "sales", "volume",
        "median", "listings", "inventory", "date",
    ],
}


class TestLoadDataset:
    """Test load_dataset for all 11 datasets."""

    @pytest.mark.parametrize("name,expected_shape", list(EXPECTED_SHAPES.items()))
    def test_shape(self, name, expected_shape):
        df = load_dataset(name)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == expected_shape, f"{name}: expected {expected_shape}, got {df.shape}"

    @pytest.mark.parametrize("name,expected_cols", list(EXPECTED_COLUMNS.items()))
    def test_columns(self, name, expected_cols):
        df = load_dataset(name)
        assert list(df.columns) == expected_cols

    def test_diamonds_ordered_categoricals(self):
        df = load_dataset("diamonds")
        assert df["cut"].dtype.name == "category"
        assert df["cut"].cat.ordered is True
        assert df["color"].dtype.name == "category"
        assert df["color"].cat.ordered is True
        assert df["clarity"].dtype.name == "category"
        assert df["clarity"].cat.ordered is True

    def test_economics_date_dtype(self):
        df = load_dataset("economics")
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_economics_long_date_dtype(self):
        df = load_dataset("economics_long")
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_presidential_date_dtype(self):
        df = load_dataset("presidential")
        assert pd.api.types.is_datetime64_any_dtype(df["start"])
        assert pd.api.types.is_datetime64_any_dtype(df["end"])

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent_dataset")

    def test_load_returns_copy(self):
        """Each call returns a fresh copy so mutations do not propagate."""
        df1 = load_dataset("mpg")
        df2 = load_dataset("mpg")
        df1.iloc[0, 0] = "MUTATED"
        assert df2.iloc[0, 0] != "MUTATED"
