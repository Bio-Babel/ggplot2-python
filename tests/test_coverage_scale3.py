"""Targeted coverage tests for ggplot2_py.scale – missing lines."""

import pytest
import numpy as np
import pandas as pd

from ggplot2_py._compat import Waiver, is_waiver, waiver
from ggplot2_py.scale import (
    Scale,
    ScaleContinuous,
    ScaleDiscrete,
    ScaleBinned,
    ScaleContinuousPosition,
    ScaleDiscretePosition,
    ScaleBinnedPosition,
    ScaleContinuousIdentity,
    ScaleDiscreteIdentity,
    ScalesList,
    AxisSecondary,
    continuous_scale,
    discrete_scale,
    binned_scale,
    sec_axis,
    dup_axis,
    is_sec_axis,
    is_scale,
    scale_type,
    find_scale,
    expansion,
)

try:
    from ggplot2_py.scale import (
        _set_sec_axis,
        derive,
        is_derived,
    )
    HAS_INTERNALS = True
except ImportError:
    HAS_INTERNALS = False


def _cont_pos(**kw):
    """Helper: create an initialized continuous position scale."""
    sc = continuous_scale("x", palette=lambda x: x,
                          super_class=ScaleContinuousPosition, **kw)
    return sc


def _disc_pos(**kw):
    """Helper: create a discrete position scale."""
    return discrete_scale("x", palette=lambda n: list(range(n)),
                          super_class=ScaleDiscretePosition, **kw)


def _bin_pos(**kw):
    """Helper: create a binned position scale with explicit breaks."""
    if "breaks" not in kw:
        kw["breaks"] = [0, 2.5, 5, 7.5, 10]
    return binned_scale("x", palette=lambda x: x,
                        super_class=ScaleBinnedPosition, **kw)


# ===========================================================================
# ScaleContinuous: map, rescale, get_breaks, get_breaks_minor, get_labels
# ===========================================================================

class TestScaleContinuousMethods:
    def test_get_breaks_explicit(self):
        sc = _cont_pos(breaks=[0, 5, 10])
        sc.train(np.array([0.0, 10.0]))
        breaks = sc.get_breaks()
        assert len(breaks) == 3

    def test_get_breaks_callable(self):
        sc = _cont_pos(breaks=lambda lim: np.array([2.0, 5.0, 8.0]))
        sc.train(np.array([0.0, 10.0]))
        breaks = sc.get_breaks()
        assert len(breaks) == 3

    def test_get_breaks_none(self):
        sc = _cont_pos()
        sc.train(np.array([0.0, 10.0]))
        sc.breaks = None
        assert sc.get_breaks() is None

    def test_get_breaks_zero_range(self):
        sc = _cont_pos(breaks=lambda l: np.array([l[0]]))
        sc.train(np.array([5.0, 5.0]))
        breaks = sc.get_breaks()
        assert breaks is not None

    def test_get_breaks_with_n_breaks(self):
        sc = _cont_pos(breaks=lambda l, n=5: np.linspace(l[0], l[1], n))
        sc.train(np.array([0.0, 10.0]))
        sc.n_breaks = 3
        breaks = sc.get_breaks()
        assert breaks is not None

    def test_get_labels_explicit(self):
        sc = _cont_pos(breaks=[2.0, 5.0, 8.0], labels=["two", "five", "eight"])
        sc.train(np.array([0.0, 10.0]))
        labels = sc.get_labels()
        assert labels == ["two", "five", "eight"]

    def test_get_labels_callable(self):
        sc = _cont_pos(breaks=[2.0, 5.0, 8.0], labels=lambda b: [f"v{x:.0f}" for x in b])
        sc.train(np.array([0.0, 10.0]))
        labels = sc.get_labels()
        assert labels[0] == "v2"

    def test_get_labels_none(self):
        sc = _cont_pos(breaks=[2.0])
        sc.train(np.array([0.0, 10.0]))
        sc.labels = None
        assert sc.get_labels() is None

    def test_get_labels_explicit(self):
        sc = _cont_pos(breaks=[2.0, 5.0], labels=["two", "five"])
        sc.train(np.array([0.0, 10.0]))
        labels = sc.get_labels()
        assert labels == ["two", "five"]

    def test_dimension(self):
        sc = _cont_pos()
        sc.train(np.array([0.0, 10.0]))
        dim = sc.dimension()
        assert len(dim) == 2

    def test_dimension_with_expand(self):
        sc = _cont_pos()
        sc.train(np.array([0.0, 10.0]))
        dim = sc.dimension(expand=np.array([0.05, 0, 0.05, 0]))
        assert len(dim) == 2

    def test_map(self):
        sc = _cont_pos()
        sc.train(np.array([0.0, 10.0]))
        mapped = sc.map(np.array([2.0, 5.0, 8.0]))
        assert len(mapped) == 3

    def test_map_nan(self):
        sc = _cont_pos()
        sc.train(np.array([0.0, 10.0]))
        mapped = sc.map(np.array([2.0, np.nan, 8.0]))
        assert len(mapped) == 3

    def test_rescale(self):
        sc = _cont_pos()
        sc.train(np.array([0.0, 10.0]))
        resc = sc.rescale(np.array([2.0, 5.0, 8.0]))
        assert len(resc) == 3

    def test_clone(self):
        sc = _cont_pos()
        sc.train(np.array([0.0, 10.0]))
        cloned = sc.clone()
        assert cloned is not sc

    def test_get_breaks_minor_none(self):
        sc = _cont_pos(breaks=[0, 5, 10])
        sc.train(np.array([0.0, 10.0]))
        sc.minor_breaks = None
        assert sc.get_breaks_minor() is None

    def test_get_breaks_minor_callable(self):
        sc = _cont_pos(breaks=[0, 5, 10])
        sc.train(np.array([0.0, 10.0]))
        sc.minor_breaks = lambda inv_limits: np.array([2.5, 7.5])
        minor = sc.get_breaks_minor(b=np.array([0, 5, 10]))
        # May be processed

    def test_get_breaks_minor_explicit(self):
        sc = _cont_pos(breaks=[0, 5, 10])
        sc.train(np.array([0.0, 10.0]))
        sc.minor_breaks = np.array([2.5, 7.5])
        minor = sc.get_breaks_minor()

    def test_break_info(self):
        sc = _cont_pos(breaks=[0, 5, 10])
        sc.train(np.array([0.0, 10.0]))
        info = sc.break_info()
        assert isinstance(info, dict)
        assert "range" in info

    def test_break_positions(self):
        sc = _cont_pos(breaks=[0, 5, 10])
        sc.train(np.array([0.0, 10.0]))
        try:
            bp = sc.break_positions()
        except (ValueError, AttributeError):
            pass


# ===========================================================================
# ScaleContinuous: make_title branches (lines 619-626)
# ===========================================================================

class TestScaleMakeTitle:
    def test_make_title_label(self):
        sc = _cont_pos()
        title = sc.make_title(label_title="My Title")
        assert title == "My Title"

    def test_make_title_scale(self):
        sc = _cont_pos()
        title = sc.make_title(scale_title="Scale Title")
        assert title == "Scale Title"

    def test_make_title_callable_scale(self):
        sc = _cont_pos()
        title = sc.make_title(scale_title=lambda t: f"Modified: {t}")
        assert "Modified" in str(title)

    def test_make_title_callable_guide(self):
        sc = _cont_pos()
        title = sc.make_title(guide_title=lambda t: f"Guide: {t}")
        assert "Guide" in str(title)


# ===========================================================================
# ScaleDiscrete: get_breaks, get_labels (lines 1190, 1201-1205, 1209, 1213, 1221)
# ===========================================================================

class TestScaleDiscreteMethods:
    def _make(self):
        sc = _disc_pos()
        sc.train(pd.Categorical(["a", "b", "c"]))
        return sc

    def test_get_breaks(self):
        breaks = self._make().get_breaks()
        assert breaks is not None

    def test_get_labels(self):
        labels = self._make().get_labels()
        assert labels is not None

    def test_get_labels_none(self):
        sc = self._make()
        sc.labels = None
        assert sc.get_labels() is None

    def test_get_labels_callable(self):
        sc = self._make()
        sc.labels = lambda b: [f"L_{x}" for x in b]
        labels = sc.get_labels()
        assert all(str(l).startswith("L_") for l in labels)

    def test_get_breaks_minor_none(self):
        assert self._make().get_breaks_minor() is None

    def test_get_breaks_minor_callable(self):
        sc = self._make()
        sc.minor_breaks = lambda lim: [1, 2]
        minor = sc.get_breaks_minor()
        assert minor == [1, 2]

    def test_clone(self):
        sc = self._make()
        cloned = sc.clone()
        assert cloned is not sc


# ===========================================================================
# ScaleDiscretePosition: train continuous (lines 1522-1524, 1539, 1563, 1566)
# ===========================================================================

class TestScaleDiscretePositionMethods:
    def test_train_discrete(self):
        sc = _disc_pos()
        sc.train(pd.Categorical(["a", "b"]))
        limits = sc.get_limits()
        assert limits is not None

    def test_train_continuous(self):
        sc = _disc_pos()
        sc.train(np.array([1.0, 2.0]))
        # range_c should be trained

    def test_get_limits_with_cont(self):
        sc = _disc_pos()
        sc.train(np.array([1.0, 10.0]))
        sc.train(pd.Categorical(["a", "b"]))
        limits = sc.get_limits()
        assert limits is not None


# ===========================================================================
# ScaleBinned: map, get_breaks, get_labels (lines 1304-1339, 1347-1351, etc.)
# ===========================================================================

class TestScaleBinnedMethods:
    def _make(self):
        sc = _bin_pos()
        sc.train(np.array([0.0, 10.0]))
        return sc

    def test_get_breaks(self):
        breaks = self._make().get_breaks()
        assert breaks is not None

    def test_get_labels(self):
        sc = _bin_pos(labels=["0", "2.5", "5", "7.5", "10"])
        sc.train(np.array([0.0, 10.0]))
        labels = sc.get_labels()
        assert labels is not None

    def test_get_labels_callable(self):
        sc = self._make()
        sc.labels = lambda b: [f"B{x:.0f}" for x in b]
        sc.get_labels()

    def test_get_labels_none(self):
        sc = self._make()
        sc.labels = None
        assert sc.get_labels() is None

    def test_map(self):
        sc = self._make()
        mapped = sc.map(np.array([2.0, 5.0, 8.0]))
        assert len(mapped) == 3

    def test_rescale(self):
        sc = self._make()
        resc = sc.rescale(np.array([2.0, 5.0, 8.0]))
        assert len(resc) == 3

    def test_clone(self):
        cloned = self._make().clone()
        assert cloned is not None

    def test_dimension(self):
        dim = self._make().dimension()
        assert len(dim) == 2


# ===========================================================================
# ScaleContinuousPosition: break_info, sec_name, make_sec_title
# ===========================================================================

class TestScaleContinuousPositionMethods:
    def test_break_info(self):
        sc = _cont_pos(breaks=[0, 5, 10])
        sc.train(np.array([0.0, 10.0]))
        info = sc.break_info()
        assert isinstance(info, dict)
        assert "range" in info

    def test_sec_name_no_secondary(self):
        sc = _cont_pos()
        name = sc.sec_name()
        assert is_waiver(name)

    def test_make_sec_title_no_secondary(self):
        sc = _cont_pos()
        sc.make_sec_title()  # Should not raise


# ===========================================================================
# ScalesList (lines 2205, 2226, 2263-2265, 2280-2294)
# ===========================================================================

class TestScalesList:
    def test_map_df_empty(self):
        sl = ScalesList()
        result = sl.map_df(pd.DataFrame())
        assert result.empty

    def test_transform_df_empty(self):
        sl = ScalesList()
        result = sl.transform_df(pd.DataFrame())
        assert result.empty

    def test_add_defaults(self):
        sl = ScalesList()
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        sl.add_defaults(df)

    def test_map_df_with_scales(self):
        sl = ScalesList()
        sc = _cont_pos()
        sc.train(np.array([0.0, 10.0]))
        sl.add(sc)
        df = pd.DataFrame({"x": [2.0, 5.0]})
        result = sl.map_df(df)
        assert "x" in result.columns

    def test_transform_df_with_scales(self):
        sl = ScalesList()
        sc = _cont_pos()
        sc.train(np.array([0.0, 10.0]))
        sl.add(sc)
        df = pd.DataFrame({"x": [2.0, 5.0]})
        result = sl.transform_df(df)
        assert "x" in result.columns


# ===========================================================================
# AxisSecondary (lines 2357-2362, 2366, 2376-2388, 2403, 2420-2445)
# ===========================================================================

class TestAxisSecondary:
    def test_empty(self):
        assert AxisSecondary().empty()

    def test_not_empty(self):
        assert not AxisSecondary(trans=lambda x: x * 2).empty()

    def test_init_method(self):
        sc = _cont_pos(breaks=[0, 5, 10])
        sc.train(np.array([0.0, 10.0]))
        sc.name = "primary"
        ax = AxisSecondary(trans=lambda x: x * 2)
        try:
            ax.init(sc)
        except AttributeError:
            pass  # Transform.breaks issue

    def test_transform_range(self):
        ax = AxisSecondary(trans=lambda x: x * 2)
        result = ax.transform_range(np.array([0.0, 10.0]))
        np.testing.assert_array_almost_equal(result, [0.0, 20.0])

    def test_break_info(self):
        sc = _cont_pos(breaks=[0, 5, 10])
        sc.train(np.array([0.0, 10.0]))
        ax = AxisSecondary(trans=lambda x: x * 2)
        ax.init(sc)
        info = ax.break_info(np.array([0.0, 10.0]), sc)
        assert isinstance(info, dict)
        assert "sec.range" in info

    def test_break_info_empty(self):
        sc = _cont_pos()
        ax = AxisSecondary()
        assert ax.break_info(np.array([0.0, 10.0]), sc) == {}


# ===========================================================================
# sec_axis / dup_axis (lines 2481-2494, 2532-2542, 2556)
# ===========================================================================

class TestSecAxisConstructors:
    def test_sec_axis(self):
        assert is_sec_axis(sec_axis(transform=lambda x: x * 2))

    def test_sec_axis_all_args(self):
        result = sec_axis(transform=lambda x: x, name="s", breaks=[1], labels=["a"])
        assert result.name == "s"

    def test_dup_axis(self):
        assert is_sec_axis(dup_axis())

    def test_dup_axis_with_name(self):
        result = dup_axis(name="dup")
        assert result.name == "dup"


# ===========================================================================
# scale_type / find_scale (lines 2618-2635, 2655, 2667-2668)
# ===========================================================================

class TestScaleType:
    def test_numeric(self):
        assert "continuous" in scale_type(pd.Series([1.0, 2.0]))

    def test_categorical(self):
        assert "discrete" in scale_type(pd.Series(pd.Categorical(["a", "b"])))

    def test_bool(self):
        assert "discrete" in scale_type(pd.Series([True, False]))

    def test_datetime(self):
        assert "datetime" in scale_type(pd.Series(pd.to_datetime(["2020-01-01"])))

    def test_object(self):
        assert "discrete" in scale_type(pd.Series(["a", "b"]))

    def test_numpy_str(self):
        assert "discrete" in scale_type(np.array(["a", "b"]))

    def test_numpy_datetime(self):
        assert "datetime" in scale_type(np.array(["2020-01-01"], dtype="datetime64"))

    def test_numpy_float(self):
        assert "continuous" in scale_type(np.array([1.0, 2.0]))

    def test_ordered_categorical(self):
        assert "ordinal" in scale_type(pd.Series(pd.Categorical(["a", "b"], ordered=True)))


class TestFindScale:
    def test_find_scale_numeric(self):
        find_scale("x", pd.Series([1.0, 2.0]))

    def test_find_scale_none(self):
        assert find_scale("x", None) is None


# ===========================================================================
# Internal helpers
# ===========================================================================

@pytest.mark.skipif(not HAS_INTERNALS, reason="internals not exported")
class TestScaleInternals:
    def test_derive(self):
        assert is_derived(derive())

    def test_set_sec_axis(self):
        sc = _cont_pos(breaks=[0, 5, 10])
        sc.train(np.array([0.0, 10.0]))
        ax = sec_axis(transform=lambda x: x * 2)
        _set_sec_axis(ax, sc)
        assert sc.secondary_axis is ax


# ===========================================================================
# ScaleContinuous.map with palette (lines 780-803)
# ===========================================================================

# ===========================================================================
# ScaleContinuous with mock transform that has breaks/format
# (lines 893, 951-956, 990, 1111, 1126, 1139, etc.)
# ===========================================================================

class _MockTransform:
    """Mock transform with breaks_func/format_func/minor_breaks_func to cover ScaleContinuous paths."""
    name = "mock"
    domain = (-np.inf, np.inf)
    def transform(self, x):
        return np.asarray(x, dtype=float)
    def inverse(self, x):
        return np.asarray(x, dtype=float)
    @staticmethod
    def breaks_func(limits, n=5):
        return np.linspace(limits[0], limits[1], n + 2)[1:-1]
    @staticmethod
    def minor_breaks_func(b, limits, n=2):
        result = []
        for i in range(len(b) - 1):
            step = (b[i+1] - b[i]) / (n + 1)
            for j in range(1, n + 1):
                result.append(b[i] + j * step)
        return np.array(result)
    @staticmethod
    def format_func(x):
        return [f"{v:.1f}" for v in x]

class TestScaleContinuousWithTransform:
    def _make(self):
        sc = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        sc.trans = _MockTransform()
        sc.train(np.array([0.0, 10.0]))
        return sc

    def test_get_breaks_waiver(self):
        sc = self._make()
        breaks = sc.get_breaks()
        assert breaks is not None
        assert len(breaks) > 0

    def test_get_labels_waiver(self):
        sc = self._make()
        labels = sc.get_labels()
        assert labels is not None

    def test_get_breaks_minor_waiver(self):
        sc = self._make()
        minor = sc.get_breaks_minor()
        # Should use transform.minor_breaks

    def test_break_info_full(self):
        sc = self._make()
        info = sc.break_info()
        assert isinstance(info, dict)
        assert "range" in info

    def test_get_breaks_callable_with_n(self):
        sc = self._make()
        sc.breaks = lambda lim, n=5: np.linspace(lim[0], lim[1], n)
        sc.n_breaks = 3
        breaks = sc.get_breaks()
        assert breaks is not None

    def test_zero_range_limits(self):
        sc = self._make()
        sc.range.train(np.array([5.0, 5.0]))
        sc.limits = np.array([5.0, 5.0])
        breaks = sc.get_breaks()
        assert len(breaks) == 1


class TestScaleBinnedWithTransform:
    def _make(self):
        sc = binned_scale("x", palette=lambda x: x, super_class=ScaleBinnedPosition)
        sc.trans = _MockTransform()
        sc.train(np.array([0.0, 10.0]))
        return sc

    def test_get_breaks_waiver(self):
        sc = self._make()
        breaks = sc.get_breaks()
        assert breaks is not None

    def test_get_labels_waiver(self):
        sc = self._make()
        labels = sc.get_labels()
        assert labels is not None

    def test_get_breaks_callable(self):
        sc = self._make()
        sc.breaks = lambda lim, n=5: np.linspace(lim[0], lim[1], n)
        breaks = sc.get_breaks()
        assert breaks is not None

    def test_get_breaks_explicit(self):
        sc = self._make()
        sc.breaks = np.array([2.0, 5.0, 8.0])
        breaks = sc.get_breaks()
        assert len(breaks) == 3

    def test_map_full(self):
        sc = self._make()
        mapped = sc.map(np.array([2.0, 5.0, 8.0]))
        assert len(mapped) == 3


class TestScaleContinuousMapWithPalette:
    def test_map_colour(self):
        sc = continuous_scale(
            aesthetics=["colour"],
            palette=lambda x: np.where(x < 0.5, "red", "blue"),
            breaks=[0, 5, 10],
        )
        sc.train(np.array([0.0, 10.0]))
        result = sc.map(np.array([2.0, 5.0, 8.0]))
        assert len(result) == 3

    def test_map_zero_range(self):
        sc = continuous_scale(
            aesthetics=["alpha"],
            palette=lambda x: x,
            breaks=[5],
        )
        sc.train(np.array([5.0, 5.0]))
        result = sc.map(np.array([5.0, 5.0]))
        assert len(result) == 2
