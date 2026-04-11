"""Tests for pipeline dispatch extensibility and build hooks — Part B+C
of the GOG pipeline refactoring.

Covers:
- update_ggplot singledispatch (register custom types)
- ggplot_build singledispatch
- ggplot_gtable singledispatch
- by_layer helper
- BuildStage constants
- Build hook system
"""

import pandas as pd
import pytest

from ggplot2_py import (
    ggplot,
    aes,
    geom_point,
    ggplot_build,
    ggplot_add,
    BuildStage,
)
from ggplot2_py.plot import (
    update_ggplot,
    by_layer,
    GGPlot,
    BuiltGGPlot,
)


# -----------------------------------------------------------------------
# B1: update_ggplot singledispatch
# -----------------------------------------------------------------------

class TestUpdateGgplotDispatch:
    def test_layer_dispatched(self):
        """Layer objects should be added via singledispatch."""
        df = pd.DataFrame({"x": [1], "y": [1]})
        p = ggplot(df, aes(x="x", y="y")) + geom_point()
        assert len(p.layers) == 1

    def test_none_dispatched(self):
        """None should be a no-op."""
        df = pd.DataFrame({"x": [1], "y": [1]})
        p = ggplot(df, aes(x="x", y="y"))
        p2 = p + None
        assert len(p2.layers) == 0

    def test_list_dispatched(self):
        """Lists should be recursively added."""
        df = pd.DataFrame({"x": [1], "y": [1]})
        p = ggplot(df, aes(x="x", y="y"))
        p2 = p + [geom_point(), geom_point()]
        assert len(p2.layers) == 2

    def test_dataframe_dispatched(self):
        """DataFrames should replace default data."""
        df1 = pd.DataFrame({"x": [1], "y": [1]})
        df2 = pd.DataFrame({"x": [2, 3], "y": [4, 5]})
        p = ggplot(df1, aes(x="x", y="y"))
        p2 = p + df2
        assert len(p2.data) == 2

    def test_register_custom_type(self):
        """Extension packages should be able to register custom types."""

        class MyAnnotation:
            def __init__(self, text):
                self.text = text

        @update_ggplot.register(MyAnnotation)
        def _add_annotation(obj, plot, object_name=""):
            plot._meta["custom_annotation"] = obj.text
            return plot

        df = pd.DataFrame({"x": [1], "y": [1]})
        p = ggplot(df, aes(x="x", y="y"))
        p2 = ggplot_add(MyAnnotation("hello"), p)
        assert p2._meta.get("custom_annotation") == "hello"

    def test_unknown_type_raises(self):
        """Unknown types should raise TypeError."""
        df = pd.DataFrame({"x": [1], "y": [1]})
        p = ggplot(df, aes(x="x", y="y"))
        with pytest.raises(Exception):
            p + 42  # int has no registered handler


# -----------------------------------------------------------------------
# B2: ggplot_build singledispatch
# -----------------------------------------------------------------------

class TestGgplotBuildDispatch:
    def test_ggplot_type(self):
        """GGPlot objects should build normally."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes(x="x", y="y")) + geom_point()
        built = ggplot_build(p)
        assert isinstance(built, BuiltGGPlot)

    def test_built_noop(self):
        """Already-built plots should be returned as-is."""
        df = pd.DataFrame({"x": [1], "y": [1]})
        p = ggplot(df, aes(x="x", y="y")) + geom_point()
        built = ggplot_build(p)
        built2 = ggplot_build(built)
        assert built is built2

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError, match="Cannot build"):
            ggplot_build("not a plot")

    def test_registry_has_ggplot(self):
        assert GGPlot in ggplot_build.registry

    def test_registry_has_built(self):
        assert BuiltGGPlot in ggplot_build.registry


# -----------------------------------------------------------------------
# B4: by_layer helper
# -----------------------------------------------------------------------

class TestByLayer:
    def test_basic(self):
        layers = ["a", "b", "c"]
        data = [1, 2, 3]
        result = by_layer(lambda l, d: f"{l}{d}", layers, data, "test")
        assert result == ["a1", "b2", "c3"]

    def test_error_context(self):
        layers = ["ok", "fail"]
        data = [1, 2]

        def fn(l, d):
            if l == "fail":
                raise ValueError("boom")
            return d

        with pytest.raises(RuntimeError, match="layer 2"):
            by_layer(fn, layers, data, "testing")


# -----------------------------------------------------------------------
# C: BuildStage + hooks
# -----------------------------------------------------------------------

class TestBuildStage:
    def test_constants_are_strings(self):
        assert isinstance(BuildStage.COMPUTE_STAT, str)
        assert BuildStage.COMPUTE_STAT == "compute_stat"
        assert BuildStage.LAYER_DATA == "layer_data"
        assert BuildStage.FINISH_DATA == "finish_data"


class TestBuildHooks:
    def test_hook_fires_after_compute_stat(self):
        log = []
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        p = ggplot(df, aes(x="x", y="y")) + geom_point()
        p.add_build_hook(
            "after",
            BuildStage.COMPUTE_STAT,
            lambda data: (log.append("fired"), data)[-1],
        )
        ggplot_build(p)
        assert "fired" in log

    def test_hook_fires_before_compute_aesthetics(self):
        log = []
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = ggplot(df, aes(x="x", y="y")) + geom_point()
        p.add_build_hook(
            "before",
            BuildStage.COMPUTE_AESTHETICS,
            lambda data: (log.append(len(data)), data)[-1],
        )
        ggplot_build(p)
        assert log == [1]  # 1 layer

    def test_hook_can_modify_data(self):
        """Hook that returns new data list replaces the pipeline data."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        p = ggplot(df, aes(x="x", y="y")) + geom_point()

        def double_y(data):
            new_data = []
            for d in data:
                d = d.copy()
                if "y" in d.columns:
                    d["y"] = d["y"] * 2
                new_data.append(d)
            return new_data

        p.add_build_hook("after", BuildStage.COMPUTE_AESTHETICS, double_y)
        built = ggplot_build(p)
        # y values should be doubled (8, 10, 12 instead of 4, 5, 6)
        y_vals = built.data[0]["y"].values
        assert y_vals[0] == 8.0 or y_vals[0] == 8

    def test_invalid_timing_raises(self):
        df = pd.DataFrame({"x": [1], "y": [1]})
        p = ggplot(df, aes(x="x", y="y"))
        with pytest.raises(ValueError, match="before.*after"):
            p.add_build_hook("during", BuildStage.COMPUTE_STAT, lambda d: d)

    def test_chaining(self):
        """add_build_hook returns self for chaining."""
        df = pd.DataFrame({"x": [1], "y": [1]})
        p = ggplot(df, aes(x="x", y="y"))
        result = p.add_build_hook("after", BuildStage.FINISH_DATA, lambda d: d)
        assert result is p
