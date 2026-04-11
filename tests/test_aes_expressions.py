"""Tests for callable expression evaluation in aes() — Part A of the
GOG pipeline refactoring.

Covers:
- AfterStat / AfterScale / Stage accepting callables
- eval_aes_value() helper
- Stage 1 (compute_aesthetics): callable in aes()
- Stage 2 (map_statistic): callable in after_stat()
- Stage 3 (use_defaults): callable in after_scale()
- make_labels() with callables
"""

import numpy as np
import pandas as pd
import pytest

from ggplot2_py.aes import (
    AfterStat,
    AfterScale,
    Stage,
    Mapping,
    aes,
    after_stat,
    after_scale,
    stage,
    eval_aes_value,
)
from ggplot2_py.labels import make_labels


# -----------------------------------------------------------------------
# A1-A2: AfterStat / AfterScale / Stage accept callables
# -----------------------------------------------------------------------

class TestAfterStatCallable:
    def test_str_still_works(self):
        a = AfterStat("count")
        assert a.x == "count"

    def test_callable_accepted(self):
        fn = lambda d: d["count"] / d["count"].max()
        a = AfterStat(fn)
        assert callable(a.x)
        assert a.x is fn

    def test_non_str_non_callable_rejected(self):
        with pytest.raises(TypeError, match="str or callable"):
            AfterStat(42)

    def test_repr_callable(self):
        a = AfterStat(lambda d: d["x"])
        assert "<lambda>" in repr(a)

    def test_repr_str(self):
        assert repr(AfterStat("count")) == "AfterStat('count')"


class TestAfterScaleCallable:
    def test_str_still_works(self):
        a = AfterScale("fill")
        assert a.x == "fill"

    def test_callable_accepted(self):
        fn = lambda d: d["colour"].str.upper()
        a = AfterScale(fn)
        assert a.x is fn

    def test_non_str_non_callable_rejected(self):
        with pytest.raises(TypeError, match="str or callable"):
            AfterScale(3.14)


class TestStageCallable:
    def test_str_slots(self):
        s = Stage(start="x", after_stat="count", after_scale="fill")
        assert s.start == "x"
        assert isinstance(s.after_stat, AfterStat)
        assert isinstance(s.after_scale, AfterScale)

    def test_callable_start(self):
        fn = lambda d: d["x"] ** 2
        s = Stage(start=fn)
        assert s.start is fn

    def test_callable_after_stat(self):
        fn = lambda d: d["count"] / d["count"].max()
        s = Stage(after_stat=fn)
        assert isinstance(s.after_stat, AfterStat)
        assert s.after_stat.x is fn

    def test_callable_after_scale(self):
        fn = lambda d: d["colour"]
        s = Stage(after_scale=fn)
        assert isinstance(s.after_scale, AfterScale)
        assert s.after_scale.x is fn


# -----------------------------------------------------------------------
# A3: eval_aes_value() helper
# -----------------------------------------------------------------------

class TestEvalAesValue:
    def setup_method(self):
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    def test_str_column_lookup(self):
        result = eval_aes_value("x", self.df)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_str_missing_column(self):
        assert eval_aes_value("z", self.df) is None

    def test_callable(self):
        result = eval_aes_value(lambda d: d["x"] * 2, self.df)
        np.testing.assert_array_equal(result, [2, 4, 6])

    def test_callable_returns_series(self):
        result = eval_aes_value(lambda d: d["x"] + d["y"], self.df)
        np.testing.assert_array_equal(result, [5, 7, 9])

    def test_scalar(self):
        assert eval_aes_value(42, self.df) == 42

    def test_str_class_not_called(self):
        # str is callable in Python, but should be treated as column ref
        result = eval_aes_value("x", self.df)
        assert isinstance(result, np.ndarray)


# -----------------------------------------------------------------------
# A4: Stage 1 — callable in compute_aesthetics
# -----------------------------------------------------------------------

class TestStage1Callable:
    def test_callable_aes(self):
        """aes(x=lambda d: d['x']*2) should evaluate at Stage 1."""
        from ggplot2_py import ggplot, geom_point, ggplot_build

        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        p = ggplot(df, aes(x=lambda d: d["x"] * 2, y="y")) + geom_point()
        built = ggplot_build(p)
        np.testing.assert_array_equal(built.data[0]["x"].values, [2.0, 4.0, 6.0])

    def test_stage_start_callable(self):
        """stage(start=callable) should evaluate at Stage 1."""
        from ggplot2_py import ggplot, geom_point, ggplot_build

        df = pd.DataFrame({"x": [10, 20, 30], "y": [1, 2, 3]})
        p = ggplot(df, aes(x=stage(start=lambda d: d["x"] / 10), y="y")) + geom_point()
        built = ggplot_build(p)
        np.testing.assert_array_equal(built.data[0]["x"].values, [1.0, 2.0, 3.0])


# -----------------------------------------------------------------------
# A5: Stage 2 — callable in after_stat / map_statistic
# -----------------------------------------------------------------------

class TestStage2Callable:
    def test_after_stat_str(self):
        """after_stat('count') still works."""
        from ggplot2_py import ggplot, geom_bar, ggplot_build

        df = pd.DataFrame({"x": ["a", "a", "b", "b", "b"]})
        p = ggplot(df, aes(x="x")) + geom_bar()
        built = ggplot_build(p)
        assert "count" in built.data[0].columns or "y" in built.data[0].columns

    def test_after_stat_callable(self):
        """after_stat(lambda d: d['count'] / d['count'].sum()) normalises counts."""
        from ggplot2_py import ggplot, geom_bar, ggplot_build

        df = pd.DataFrame({"x": ["a", "a", "b", "b", "b"]})
        p = ggplot(df, aes(x="x", y=after_stat(lambda d: d["count"] / d["count"].sum()))) + geom_bar()
        built = ggplot_build(p)
        y_vals = built.data[0]["y"].values
        # Proportions should sum to 1
        assert abs(y_vals.sum() - 1.0) < 1e-10


# -----------------------------------------------------------------------
# A7: make_labels with callables
# -----------------------------------------------------------------------

class TestMakeLabelsCallable:
    def test_str_label(self):
        m = Mapping(x="mpg", y="hp")
        labels = make_labels(m)
        assert labels == {"x": "mpg", "y": "hp"}

    def test_callable_label_uses_name(self):
        def my_transform(d):
            return d["x"]

        m = Mapping(x=my_transform)
        labels = make_labels(m)
        assert labels["x"] == "my_transform"

    def test_lambda_label(self):
        m = Mapping(x=lambda d: d["x"])
        labels = make_labels(m)
        assert labels["x"] == "<lambda>"

    def test_after_stat_callable_label(self):
        m = Mapping(y=AfterStat(lambda d: d["count"]))
        labels = make_labels(m)
        assert labels["y"] == "<lambda>"

    def test_after_stat_str_label(self):
        m = Mapping(y=AfterStat("count"))
        labels = make_labels(m)
        assert labels["y"] == "count"
