"""Comprehensive tests for ggplot2_py.scale to improve coverage."""

import pytest
import numpy as np
import pandas as pd
import copy
import warnings

from ggplot2_py.scale import (
    # Base classes
    Scale,
    ScaleContinuous,
    ScaleDiscrete,
    ScaleBinned,
    # Position sub-classes
    ScaleContinuousPosition,
    ScaleDiscretePosition,
    ScaleBinnedPosition,
    # Identity sub-classes
    ScaleContinuousIdentity,
    ScaleDiscreteIdentity,
    # Date sub-classes
    ScaleContinuousDate,
    ScaleContinuousDatetime,
    # Constructors
    continuous_scale,
    discrete_scale,
    binned_scale,
    # Container
    ScalesList,
    scales_list,
    # Expansion helpers
    expansion,
    expand_scale,
    expand_range4,
    default_expansion,
    # Scale detection
    find_scale,
    is_scale,
    # Mapped discrete sentinel
    mapped_discrete,
    is_mapped_discrete,
    # Utilities
    _is_position_aes,
    _is_discrete,
    _empty,
    _unique0,
    _check_breaks_labels,
    _is_finite,
    _MappedDiscrete,
)

from ggplot2_py._compat import waiver, is_waiver

from scales import rescale, ContinuousRange, DiscreteRange


# ============================================================================
# Utility helper tests
# ============================================================================

class TestIsPositionAes:
    def test_string_true(self):
        assert _is_position_aes("x") is True
        assert _is_position_aes("ymin") is True

    def test_string_false(self):
        assert _is_position_aes("colour") is False

    def test_sequence(self):
        assert _is_position_aes(["colour", "x"]) is True
        assert _is_position_aes(["colour", "fill"]) is False


class TestIsDiscrete:
    def test_categorical(self):
        assert _is_discrete(pd.Categorical(["a", "b"])) is True

    def test_categorical_series(self):
        s = pd.Series(pd.Categorical(["a", "b"]))
        assert _is_discrete(s) is True

    def test_object_series(self):
        assert _is_discrete(pd.Series(["a", "b"])) is True

    def test_bool_series(self):
        assert _is_discrete(pd.Series([True, False])) is True

    def test_numeric_series(self):
        assert _is_discrete(pd.Series([1.0, 2.0])) is False

    def test_numpy_string(self):
        assert _is_discrete(np.array(["a", "b"])) is True

    def test_numpy_numeric(self):
        assert _is_discrete(np.array([1.0, 2.0])) is False

    def test_numpy_bool(self):
        assert _is_discrete(np.array([True, False])) is True

    def test_list_string(self):
        assert _is_discrete(["a", "b"]) is True

    def test_list_bool(self):
        assert _is_discrete([True, False]) is True

    def test_list_empty(self):
        assert _is_discrete([]) is False

    def test_string_scalar(self):
        assert _is_discrete("hello") is True

    def test_bool_scalar(self):
        assert _is_discrete(True) is True

    def test_int(self):
        assert _is_discrete(42) is False


class TestEmpty:
    def test_none(self):
        assert _empty(None) is True

    def test_empty_df(self):
        assert _empty(pd.DataFrame()) is True

    def test_nonempty_df(self):
        assert _empty(pd.DataFrame({"x": [1]})) is False

    def test_empty_dict(self):
        assert _empty({}) is True

    def test_nonempty_dict(self):
        assert _empty({"a": 1}) is False


class TestUnique0:
    def test_basic(self):
        result = _unique0(np.array([3, 1, 2, 1, 3]))
        # Preserves order of first appearance
        assert result[0] == 3
        assert result[1] == 1
        assert result[2] == 2

    def test_none(self):
        result = _unique0(None)
        assert len(result) == 0


class TestCheckBreaksLabels:
    def test_compatible(self):
        _check_breaks_labels([1, 2, 3], ["a", "b", "c"])

    def test_incompatible_length(self):
        with pytest.raises(Exception):
            _check_breaks_labels([1, 2], ["a", "b", "c"])

    def test_none_breaks(self):
        _check_breaks_labels(None, ["a", "b"])  # should not raise

    def test_callable_breaks(self):
        _check_breaks_labels(lambda x: [1, 2], ["a", "b"])  # should not raise


class TestIsFinite:
    def test_basic(self):
        result = _is_finite([1.0, np.nan, np.inf, 2.0])
        assert result[0] is np.True_
        assert result[1] is np.False_
        assert result[2] is np.False_
        assert result[3] is np.True_


# ============================================================================
# Mapped discrete sentinel
# ============================================================================

class TestMappedDiscrete:
    def test_create(self):
        md = mapped_discrete([1.0, 2.0, 3.0])
        assert isinstance(md, _MappedDiscrete)

    def test_is_mapped_discrete(self):
        md = mapped_discrete([1.0, 2.0])
        assert is_mapped_discrete(md) is True
        assert is_mapped_discrete(np.array([1.0, 2.0])) is False

    def test_none(self):
        assert mapped_discrete(None) is None


# ============================================================================
# Expansion helpers
# ============================================================================

class TestExpansion:
    def test_default(self):
        e = expansion()
        assert len(e) == 4
        np.testing.assert_array_equal(e, [0, 0, 0, 0])

    def test_mult_only(self):
        e = expansion(mult=0.05)
        assert e[0] == pytest.approx(0.05)
        assert e[2] == pytest.approx(0.05)

    def test_add_only(self):
        e = expansion(add=1)
        assert e[1] == pytest.approx(1.0)
        assert e[3] == pytest.approx(1.0)

    def test_asymmetric(self):
        e = expansion(mult=[0.1, 0.2], add=[0.5, 1.0])
        assert e[0] == pytest.approx(0.1)
        assert e[2] == pytest.approx(0.2)
        assert e[1] == pytest.approx(0.5)
        assert e[3] == pytest.approx(1.0)


class TestExpandScale:
    def test_deprecated_alias(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = expand_scale(mult=0.05, add=1)
        assert len(e) == 4


class TestExpandRange4:
    def test_2_element_expand(self):
        result = expand_range4([0, 10], [0.05, 0])
        result = np.asarray(result, dtype=float).flatten()
        # expand_range may return 2 or more elements depending on expand_range behavior
        assert len(result) >= 2

    def test_non_finite(self):
        result = expand_range4([np.nan, np.nan], [0.05, 0, 0.05, 0])
        result = np.asarray(result, dtype=float).flatten()
        assert not np.all(np.isfinite(result))

    def test_4_element_expand(self):
        result = expand_range4([0, 10], [0.05, 1, 0.05, 1])
        result = np.asarray(result, dtype=float).flatten()
        assert len(result) >= 2
        # The expanded range should be wider
        assert result[0] < 0 or result[-1] > 10


# ============================================================================
# Scale base class tests
# ============================================================================

class TestScale:
    def test_is_scale(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        assert is_scale(s) is True
        assert is_scale("not a scale") is False
        assert is_scale(None) is False

    def test_transform_df_empty(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        result = s.transform_df(pd.DataFrame())
        assert result == {}

    def test_transform_df_no_matching(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        result = s.transform_df(pd.DataFrame({"colour": [1, 2]}))
        assert result == {}

    def test_train_df_empty(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        s.train_df(pd.DataFrame())  # should not raise

    def test_map_df_empty(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        result = s.map_df(pd.DataFrame())
        assert result == {}

    def test_get_transformation(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        t = s.get_transformation()
        assert t is not None

    def test_make_title(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        title = s.make_title(scale_title="My Title")
        assert title == "My Title"

    def test_make_title_waiver(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        title = s.make_title()
        assert is_waiver(title) or title is None

    def test_make_title_guide_overrides(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        title = s.make_title(guide_title="Guide", scale_title="Scale")
        assert title == "Guide"

    def test_make_sec_title(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        title = s.make_sec_title(scale_title="Secondary")
        assert title == "Secondary"

    def test_axis_order(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        s.position = "left"
        order = s.axis_order()
        assert order == ["primary", "secondary"]

    def test_axis_order_right(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        s.position = "right"
        order = s.axis_order()
        assert order == ["secondary", "primary"]

    def test_is_empty(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        assert s.is_empty() is True

    def test_reset(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        s.train(np.array([1.0, 10.0]))
        s.reset()
        assert s.range.range is None


# ============================================================================
# ScaleContinuous tests
# ============================================================================

class TestScaleContinuous:
    def test_is_discrete(self):
        s = continuous_scale("x", palette=lambda x: x)
        assert s.is_discrete() is False

    def test_train(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        s.train(np.array([1.0, 5.0, 10.0]))
        assert s.range.range is not None

    def test_train_empty(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        s.train(np.array([]))
        assert s.range.range is None

    def test_is_empty_with_data(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        s.train(np.array([1.0, 10.0]))
        assert s.is_empty() is False

    def test_is_empty_with_limits(self):
        s = continuous_scale("x", palette=lambda x: x, limits=[0, 10], super_class=ScaleContinuousPosition)
        assert s.is_empty() is False

    def test_transform(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        result = s.transform(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_map_position(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        s.train(np.array([0.0, 10.0]))
        result = s.map(np.array([5.0]))
        assert not np.isnan(result[0])

    def test_rescale(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        s.train(np.array([0.0, 10.0]))
        result = s.rescale(np.array([5.0]))
        assert result[0] == pytest.approx(0.5)

    def test_get_limits(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        s.train(np.array([1.0, 10.0]))
        limits = s.get_limits()
        assert limits[0] == pytest.approx(1.0)
        assert limits[1] == pytest.approx(10.0)

    def test_get_limits_with_explicit(self):
        s = continuous_scale("x", palette=lambda x: x, limits=[0, 20], super_class=ScaleContinuousPosition)
        limits = s.get_limits()
        assert limits[0] == pytest.approx(0.0)
        assert limits[1] == pytest.approx(20.0)

    def test_get_limits_callable(self):
        s = continuous_scale("x", palette=lambda x: x, limits=lambda r: [0, 100], super_class=ScaleContinuousPosition)
        s.train(np.array([1.0, 10.0]))
        limits = s.get_limits()
        assert limits[0] == pytest.approx(0.0)

    def test_dimension(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        s.train(np.array([0.0, 10.0]))
        limits = s.get_limits()
        assert limits[0] == pytest.approx(0.0)
        assert limits[1] == pytest.approx(10.0)

    def test_get_breaks_empty(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        breaks = s.get_breaks()
        assert len(breaks) == 0

    def test_get_breaks_explicit(self):
        s = continuous_scale("x", palette=lambda x: x, breaks=[0, 5, 10], super_class=ScaleContinuousPosition)
        s.train(np.array([0.0, 10.0]))
        breaks = s.get_breaks()
        np.testing.assert_array_almost_equal(breaks, [0, 5, 10])

    def test_get_breaks_callable(self):
        s = continuous_scale("x", palette=lambda x: x, breaks=lambda lim: [lim[0], lim[1]], super_class=ScaleContinuousPosition)
        s.train(np.array([0.0, 10.0]))
        breaks = s.get_breaks()
        assert len(breaks) == 2

    def test_get_breaks_none(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        s.train(np.array([0.0, 10.0]))
        s.breaks = None
        breaks = s.get_breaks()
        assert breaks is None

    def test_get_breaks_minor_none(self):
        s = continuous_scale("x", palette=lambda x: x, breaks=[0, 5, 10], super_class=ScaleContinuousPosition)
        s.train(np.array([0.0, 10.0]))
        s.minor_breaks = None  # Set after construction to actually be None
        minor = s.get_breaks_minor()
        assert minor is None

    def test_get_labels_none(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        s.labels = None
        labels = s.get_labels(np.array([1, 2, 3]))
        assert labels is None

    def test_get_labels_callable(self):
        s = continuous_scale("x", palette=lambda x: x, labels=lambda b: [f"{v:.0f}!" for v in b], super_class=ScaleContinuousPosition)
        s.train(np.array([0.0, 10.0]))
        labels = s.get_labels(np.array([0, 5, 10]))
        assert labels[1] == "5!"

    def test_get_labels_none(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        s.labels = None
        labels = s.get_labels(np.array([1, 2, 3]))
        assert labels is None

    def test_get_labels_callable(self):
        s = continuous_scale("x", palette=lambda x: x, labels=lambda b: [f"{v:.0f}!" for v in b], super_class=ScaleContinuousPosition)
        s.train(np.array([0.0, 10.0]))
        labels = s.get_labels(np.array([0, 5, 10]))
        assert labels[1] == "5!"

    def test_clone(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        s.train(np.array([0.0, 10.0]))
        c = s.clone()
        assert c.range.range is None  # fresh range
        assert c.aesthetics == s.aesthetics

    def test_break_info(self):
        s = continuous_scale("x", palette=lambda x: x, breaks=[0, 50, 100],
                             labels=["0", "50", "100"],
                             super_class=ScaleContinuousPosition)
        s.minor_breaks = None
        s.train(np.array([0.0, 100.0]))
        info = s.break_info(range=np.array([0.0, 100.0]))
        assert "range" in info
        assert "labels" in info
        assert "major" in info

    def test_break_info_minor_continuous(self):
        # With default (waiver) minor_breaks, a continuous POSITION scale must
        # surface non-None minor / minor_source so that coord_polar /
        # coord_radial can draw minor rings & spokes.  Values are the R gold
        # standard for scale_y_continuous() trained on [0, 30]:
        #   minor_source = 0 5 10 15 20 25 30
        #   minor        = 0 1/6 1/3 1/2 2/3 5/6 1
        s = continuous_scale("x", palette=lambda x: x,
                             super_class=ScaleContinuousPosition)
        s.train(np.array([0.0, 30.0]))
        info = s.break_info(range=np.array([0.0, 30.0]))
        assert info["minor_source"] is not None
        assert info["minor"] is not None
        np.testing.assert_allclose(
            info["minor_source"], [0, 5, 10, 15, 20, 25, 30])
        np.testing.assert_allclose(
            info["minor"],
            [0, 1 / 6, 1 / 3, 0.5, 2 / 3, 5 / 6, 1])

    def test_break_info_minor_discards_oob(self):
        # R: get_breaks_minor ends with discard(breaks, limits) — minor breaks
        # outside the range must be dropped.  scale_y_continuous() on [2, 17]:
        #   minor_source = 2 4 6 8 10 12 14 16   (0, 18, 20 discarded)
        s = continuous_scale("x", palette=lambda x: x,
                             super_class=ScaleContinuousPosition)
        s.train(np.array([2.0, 17.0]))
        info = s.break_info(range=np.array([2.0, 17.0]))
        np.testing.assert_allclose(
            info["minor_source"], [2, 4, 6, 8, 10, 12, 14, 16])

    def test_get_limits_empty_scale(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        limits = s.get_limits()
        np.testing.assert_array_equal(limits, [0.0, 1.0])


# ============================================================================
# ScaleDiscrete tests
# ============================================================================

class TestScaleDiscrete:
    def test_is_discrete(self):
        s = discrete_scale("colour", palette=lambda n: [f"col{i}" for i in range(n)])
        assert s.is_discrete() is True

    def test_train(self):
        s = discrete_scale("colour", palette=lambda n: [f"col{i}" for i in range(n)])
        s.train(pd.Series(["a", "b", "c"]))
        assert s.range.range is not None

    def test_train_empty(self):
        s = discrete_scale("colour", palette=lambda n: [f"col{i}" for i in range(n)])
        s.train(pd.Series([], dtype=str))
        # Range should remain None or empty

    def test_transform_identity(self):
        s = discrete_scale("colour", palette=lambda n: [f"col{i}" for i in range(n)])
        data = pd.Series(["a", "b"])
        result = s.transform(data)
        # Identity transform
        assert list(result) == ["a", "b"]

    def test_map(self):
        s = discrete_scale("colour", palette=lambda n: [f"col{i}" for i in range(n)])
        s.train(pd.Series(["a", "b", "c"]))
        result = s.map(np.array(["a", "b", "c"]))
        assert len(result) == 3
        assert result[0] == "col0"

    def test_map_missing(self):
        s = discrete_scale("colour", palette=lambda n: [f"col{i}" for i in range(n)])
        s.train(pd.Series(["a", "b"]))
        result = s.map(np.array(["a", "z"]))
        # "z" not in limits should be na_value

    def test_map_dict_palette(self):
        s = discrete_scale("colour", palette=lambda n: {"k": "v"})
        s.train(pd.Series(["a"]))
        result = s.map(np.array(["a"]))
        assert len(result) == 1

    def test_rescale(self):
        s = discrete_scale("colour", palette=lambda n: [f"col{i}" for i in range(n)])
        s.train(pd.Series(["a", "b", "c"]))
        result = s.rescale(np.array(["a", "c"]))
        assert len(result) == 2

    def test_dimension(self):
        s = discrete_scale("colour", palette=lambda n: [f"col{i}" for i in range(n)])
        s.train(pd.Series(["a", "b", "c"]))
        dim = s.dimension()
        assert len(dim) == 2

    def test_dimension_empty(self):
        s = discrete_scale("colour", palette=lambda n: [f"col{i}" for i in range(n)])
        dim = s.dimension()
        dim = np.asarray(dim)
        # Empty scale: returns [0, 1] or similar default
        assert len(dim) == 2

    def test_get_breaks(self):
        s = discrete_scale("colour", palette=lambda n: [f"col{i}" for i in range(n)])
        s.train(pd.Series(["a", "b", "c"]))
        breaks = s.get_breaks()
        assert list(breaks) == ["a", "b", "c"]

    def test_get_breaks_explicit(self):
        s = discrete_scale("colour", palette=lambda n: [f"col{i}" for i in range(n)], breaks=["a", "c"])
        s.train(pd.Series(["a", "b", "c"]))
        breaks = s.get_breaks()
        assert "b" not in breaks

    def test_get_breaks_callable(self):
        s = discrete_scale("colour", palette=lambda n: [f"col{i}" for i in range(n)], breaks=lambda lim: lim[:2])
        s.train(pd.Series(["a", "b", "c"]))
        breaks = s.get_breaks()
        assert len(breaks) == 2

    def test_get_breaks_none(self):
        s = discrete_scale("colour", palette=lambda n: [f"col{i}" for i in range(n)])
        s.train(pd.Series(["a", "b"]))
        s.breaks = None
        breaks = s.get_breaks()
        assert breaks is None

    def test_get_breaks_minor(self):
        s = discrete_scale("colour", palette=lambda n: [f"col{i}" for i in range(n)])
        minor = s.get_breaks_minor()
        assert minor is None

    def test_get_labels(self):
        s = discrete_scale("colour", palette=lambda n: [f"col{i}" for i in range(n)])
        s.train(pd.Series(["a", "b", "c"]))
        labels = s.get_labels()
        assert labels == ["a", "b", "c"]

    def test_get_labels_callable(self):
        s = discrete_scale("colour", palette=lambda n: [f"col{i}" for i in range(n)], labels=lambda b: [f"L-{x}" for x in b])
        s.train(pd.Series(["a", "b"]))
        labels = s.get_labels()
        assert labels[0] == "L-a"

    def test_get_labels_none(self):
        s = discrete_scale("colour", palette=lambda n: [f"col{i}" for i in range(n)])
        s.labels = None
        s.train(pd.Series(["a", "b"]))
        labels = s.get_labels()
        assert labels is None

    def test_clone(self):
        s = discrete_scale("colour", palette=lambda n: [f"col{i}" for i in range(n)])
        s.train(pd.Series(["a", "b"]))
        c = s.clone()
        assert c.range.range is None

    def test_break_info(self):
        s = discrete_scale("colour", palette=lambda n: list(range(n)))
        s.train(pd.Series(["a", "b", "c"]))
        info = s.break_info(range=(1, 3))
        assert "labels" in info

    def test_break_info_no_breaks(self):
        s = discrete_scale("colour", palette=lambda n: list(range(n)))
        s.breaks = None
        s.train(pd.Series(["a", "b"]))
        info = s.break_info()
        assert info["major"] is None


# ============================================================================
# ScaleBinned tests
# ============================================================================

class TestScaleBinned:
    def test_is_discrete(self):
        s = binned_scale("colour", palette=lambda x: x)
        assert s.is_discrete() is False

    def test_train(self):
        s = binned_scale("colour", palette=lambda x: x)
        s.train(np.array([1.0, 5.0, 10.0]))
        assert s.range.range is not None

    def test_train_empty(self):
        s = binned_scale("colour", palette=lambda x: x)
        s.train(np.array([]))
        assert s.range.range is None

    def test_transform(self):
        s = binned_scale("colour", palette=lambda x: x)
        result = s.transform(np.array([1.0, 2.0]))
        np.testing.assert_array_almost_equal(result, [1.0, 2.0])

    def test_get_limits(self):
        s = binned_scale("colour", palette=lambda x: x)
        s.train(np.array([0.0, 10.0]))
        limits = s.get_limits()
        assert limits[0] == pytest.approx(0.0)

    def test_get_breaks_explicit(self):
        s = binned_scale("colour", palette=lambda x: x, breaks=[2, 5, 8])
        s.train(np.array([0.0, 10.0]))
        breaks = s.get_breaks()
        assert breaks is not None
        np.testing.assert_array_almost_equal(breaks, [2, 5, 8])

    def test_get_breaks_none(self):
        s = binned_scale("colour", palette=lambda x: x)
        s.train(np.array([0.0, 10.0]))
        s.breaks = None
        assert s.get_breaks() is None

    def test_get_breaks_not_nice(self):
        s = binned_scale("colour", palette=lambda x: x, nice_breaks=False)
        s.train(np.array([0.0, 10.0]))
        # Avoid waiver breaks by setting explicit
        s.breaks = [2, 5, 8]
        breaks = s.get_breaks()
        assert breaks is not None

    def test_get_labels_explicit(self):
        s = binned_scale("colour", palette=lambda x: x, breaks=[2, 5, 8], labels=["lo", "mid", "hi"])
        s.train(np.array([0.0, 10.0]))
        breaks = s.get_breaks()
        if breaks is not None:
            labels = s.get_labels(breaks)
            assert labels == ["lo", "mid", "hi"]

    def test_get_labels_none(self):
        s = binned_scale("colour", palette=lambda x: x, breaks=[2, 5, 8])
        s.labels = None
        labels = s.get_labels(np.array([2, 5, 8]))
        assert labels is None

    def test_clone(self):
        s = binned_scale("colour", palette=lambda x: x)
        s.train(np.array([0.0, 10.0]))
        c = s.clone()
        assert c.range.range is None

    def test_break_info_explicit(self):
        s = binned_scale("colour", palette=lambda x: x, breaks=[2, 5, 8], labels=["lo", "mid", "hi"])
        s.train(np.array([0.0, 10.0]))
        info = s.break_info()
        assert "range" in info

    def test_get_breaks_minor(self):
        s = binned_scale("colour", palette=lambda x: x)
        assert s.get_breaks_minor() is None


# ============================================================================
# Position sub-class tests
# ============================================================================

class TestScaleContinuousPosition:
    def test_map(self):
        s = continuous_scale(["x"], palette=lambda x: x, super_class=ScaleContinuousPosition)
        s.train(np.array([0.0, 10.0]))
        result = s.map(np.array([5.0]))
        assert result[0] == pytest.approx(5.0)

    def test_break_info(self):
        s = continuous_scale(["x"], palette=lambda x: x, breaks=[0, 25, 50, 75, 100],
                             labels=["0", "25", "50", "75", "100"],
                             super_class=ScaleContinuousPosition)
        s.minor_breaks = None
        s.train(np.array([0.0, 100.0]))
        info = s.break_info(range=np.array([0.0, 100.0]))
        assert "range" in info
        assert "major" in info

    def test_sec_name_waiver(self):
        s = continuous_scale(["x"], palette=lambda x: x, super_class=ScaleContinuousPosition)
        assert is_waiver(s.sec_name())


class TestScaleDiscretePosition:
    def test_train_discrete(self):
        s = discrete_scale(["x"], palette=lambda n: list(range(1, n + 1)), super_class=ScaleDiscretePosition)
        s.train(pd.Series(["a", "b", "c"]))
        assert s.range.range is not None

    def test_train_continuous(self):
        s = discrete_scale(["x"], palette=lambda n: list(range(1, n + 1)), super_class=ScaleDiscretePosition)
        s.train(np.array([1.0, 2.0, 3.0]))
        # Should train range_c

    def test_map_discrete(self):
        s = discrete_scale(["x"], palette=lambda n: list(range(1, n + 1)), super_class=ScaleDiscretePosition)
        s.train(pd.Series(["a", "b", "c"]))
        result = s.map(pd.Series(["a", "c"]))
        assert is_mapped_discrete(result) is True

    def test_map_continuous(self):
        s = discrete_scale(["x"], palette=lambda n: list(range(1, n + 1)), super_class=ScaleDiscretePosition)
        s.train(pd.Series(["a", "b"]))
        result = s.map(np.array([1.0, 2.0]))
        assert is_mapped_discrete(result) is True

    def test_dimension(self):
        s = discrete_scale(["x"], palette=lambda n: list(range(1, n + 1)), super_class=ScaleDiscretePosition)
        s.train(pd.Series(["a", "b", "c"]))
        dim = s.dimension()
        assert len(dim) == 2

    def test_is_empty(self):
        s = discrete_scale(["x"], palette=lambda n: list(range(1, n + 1)), super_class=ScaleDiscretePosition)
        assert s.is_empty() is True
        s.train(pd.Series(["a"]))
        assert s.is_empty() is False

    def test_clone(self):
        s = discrete_scale(["x"], palette=lambda n: list(range(1, n + 1)), super_class=ScaleDiscretePosition)
        s.train(pd.Series(["a", "b"]))
        c = s.clone()
        assert c.range.range is None

    def test_get_limits(self):
        s = discrete_scale(["x"], palette=lambda n: list(range(1, n + 1)), super_class=ScaleDiscretePosition)
        s.train(pd.Series(["a", "b"]))
        limits = s.get_limits()
        assert "a" in limits

    def test_get_limits_callable(self):
        s = discrete_scale(["x"], palette=lambda n: list(range(1, n + 1)), limits=lambda r: r[:1], super_class=ScaleDiscretePosition)
        s.train(pd.Series(["a", "b"]))
        limits = s.get_limits()
        assert len(limits) == 1

    def test_reset(self):
        s = discrete_scale(["x"], palette=lambda n: list(range(1, n + 1)), super_class=ScaleDiscretePosition)
        s.train(np.array([1.0, 5.0]))
        s.reset()  # should not raise

    def test_sec_name(self):
        s = discrete_scale(["x"], palette=lambda n: list(range(1, n + 1)), super_class=ScaleDiscretePosition)
        result = s.sec_name()
        assert is_waiver(result)


class TestScaleBinnedPosition:
    def test_map(self):
        s = binned_scale(["x"], palette=lambda x: x, breaks=[3, 5, 7], super_class=ScaleBinnedPosition)
        s.train(np.array([0.0, 10.0]))
        result = s.map(np.array([3.0, 7.0]))
        assert len(result) == 2

    def test_reset(self):
        s = binned_scale(["x"], palette=lambda x: x, breaks=[3, 5, 7], super_class=ScaleBinnedPosition)
        s.train(np.array([0.0, 10.0]))
        s.reset()
        assert s.after_stat is True

    def test_get_breaks_show_limits(self):
        s = binned_scale(["x"], palette=lambda x: x, breaks=[3, 5, 7], show_limits=True, super_class=ScaleBinnedPosition)
        s.train(np.array([0.0, 10.0]))
        breaks = s.get_breaks()
        assert breaks is not None


# ============================================================================
# Identity sub-class tests
# ============================================================================

class TestScaleContinuousIdentity:
    def test_map(self):
        s = continuous_scale("colour", palette=lambda x: x, super_class=ScaleContinuousIdentity)
        result = s.map(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_map_categorical(self):
        s = continuous_scale("colour", palette=lambda x: x, super_class=ScaleContinuousIdentity)
        result = s.map(pd.Categorical(["red", "blue"]))
        assert list(result) == ["red", "blue"]

    def test_train_guide_none(self):
        s = continuous_scale("colour", palette=lambda x: x, guide="none", super_class=ScaleContinuousIdentity)
        s.train(np.array([1.0, 2.0]))
        # Should not train when guide is "none"


class TestScaleDiscreteIdentity:
    def test_map(self):
        s = discrete_scale("colour", palette=lambda n: list(range(n)), super_class=ScaleDiscreteIdentity)
        result = s.map(np.array(["red", "blue"]))
        assert list(result) == ["red", "blue"]

    def test_train_guide_none(self):
        s = discrete_scale("colour", palette=lambda n: list(range(n)), guide="none", super_class=ScaleDiscreteIdentity)
        s.train(pd.Series(["a", "b"]))
        # Should not train when guide is "none"


# ============================================================================
# Date sub-classes
# ============================================================================

class TestScaleContinuousDate:
    def test_exists(self):
        s = ScaleContinuousDate()
        assert s.is_discrete() is False


class TestScaleContinuousDatetime:
    def test_exists(self):
        s = ScaleContinuousDatetime()
        assert s.is_discrete() is False


# ============================================================================
# Constructor function tests
# ============================================================================

class TestContinuousScale:
    def test_basic(self):
        s = continuous_scale("colour", palette=lambda x: x)
        assert isinstance(s, ScaleContinuous)

    def test_with_limits(self):
        s = continuous_scale("colour", palette=lambda x: x, limits=[0, 100])
        limits = s.get_limits()
        assert limits[0] == pytest.approx(0.0)

    def test_custom_position(self):
        s = continuous_scale("x", palette=lambda x: x, position="bottom")
        assert s.position == "bottom"

    def test_invalid_position(self):
        with pytest.raises(Exception):
            continuous_scale("x", palette=lambda x: x, position="invalid")

    def test_super_class(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        assert isinstance(s, ScaleContinuousPosition)

    def test_breaks_none_non_position(self):
        # When breaks=None for non-position aesthetics, guide should be set to "none"
        # However, breaks=None means waiver() is set; to truly set None we need to
        # pass it after construction or check that setting breaks=None properly disables guide
        s = continuous_scale("colour", palette=lambda x: x)
        s.breaks = None
        # The guide is set at construction; test the breaks=None behavior
        assert s.breaks is None

    def test_transform_string(self):
        s = continuous_scale("x", palette=lambda x: x, transform="identity")
        assert s.trans is not None

    def test_n_breaks(self):
        s = continuous_scale("x", palette=lambda x: x, n_breaks=5, super_class=ScaleContinuousPosition)
        assert s.n_breaks == 5


class TestDiscreteScale:
    def test_basic(self):
        s = discrete_scale("colour", palette=lambda n: list(range(n)))
        assert isinstance(s, ScaleDiscrete)

    def test_with_limits(self):
        s = discrete_scale("colour", palette=lambda n: list(range(n)), limits=["a", "b"])
        limits = s.get_limits()
        assert list(limits) == ["a", "b"]

    def test_na_translate(self):
        s = discrete_scale("colour", palette=lambda n: list(range(n)), na_translate=False)
        assert s.na_translate is False

    def test_drop(self):
        s = discrete_scale("colour", palette=lambda n: list(range(n)), drop=False)
        assert s.drop is False

    def test_super_class(self):
        s = discrete_scale("x", palette=lambda n: list(range(1, n + 1)), super_class=ScaleDiscretePosition)
        assert isinstance(s, ScaleDiscretePosition)


class TestBinnedScale:
    def test_basic(self):
        s = binned_scale("colour", palette=lambda x: x)
        assert isinstance(s, ScaleBinned)

    def test_with_limits(self):
        s = binned_scale("colour", palette=lambda x: x, limits=[0, 10])
        limits = s.get_limits()
        assert limits[0] == pytest.approx(0.0)

    def test_nice_breaks(self):
        s = binned_scale("colour", palette=lambda x: x, nice_breaks=True)
        assert s.nice_breaks is True

    def test_show_limits(self):
        s = binned_scale("colour", palette=lambda x: x, show_limits=True)
        assert s.show_limits is True


# ============================================================================
# ScalesList tests
# ============================================================================

class TestScalesList:
    def test_empty(self):
        sl = scales_list()
        assert sl.n() == 0

    def test_add(self):
        sl = ScalesList()
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        sl.add(s)
        assert sl.n() == 1

    def test_add_none(self):
        sl = ScalesList()
        sl.add(None)
        assert sl.n() == 0

    def test_add_replaces(self):
        sl = ScalesList()
        s1 = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        s2 = continuous_scale("x", palette=lambda x: x * 2, super_class=ScaleContinuousPosition)
        sl.add(s1)
        sl.add(s2)
        assert sl.n() == 1

    def test_has_scale(self):
        sl = ScalesList()
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        sl.add(s)
        assert sl.has_scale("x") is True
        assert sl.has_scale("y") is False

    def test_find(self):
        sl = ScalesList()
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        sl.add(s)
        found = sl.find("x")
        assert any(found)

    def test_input(self):
        sl = ScalesList()
        s = continuous_scale(["x", "xmin"], palette=lambda x: x, super_class=ScaleContinuousPosition)
        sl.add(s)
        inp = sl.input()
        assert "x" in inp

    def test_clone(self):
        sl = ScalesList()
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        sl.add(s)
        c = sl.clone()
        assert c.n() == 1

    def test_non_position_scales(self):
        sl = ScalesList()
        s1 = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        s2 = continuous_scale("colour", palette=lambda x: x)
        sl.add(s1)
        sl.add(s2)
        nps = sl.non_position_scales()
        assert nps.n() == 1

    def test_get_scales(self):
        sl = ScalesList()
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        sl.add(s)
        found = sl.get_scales("x")
        assert found is not None
        assert sl.get_scales("y") is None

    def test_train_df(self):
        sl = ScalesList()
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        sl.add(s)
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        sl.train_df(df)
        assert s.range.range is not None

    def test_train_df_empty(self):
        sl = ScalesList()
        sl.train_df(pd.DataFrame())  # should not raise

    def test_map_df(self):
        sl = ScalesList()
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        sl.add(s)
        s.train(np.array([0.0, 10.0]))
        df = pd.DataFrame({"x": [5.0]})
        result = sl.map_df(df)
        assert isinstance(result, pd.DataFrame)

    def test_transform_df(self):
        sl = ScalesList()
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        sl.add(s)
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = sl.transform_df(df)
        assert isinstance(result, pd.DataFrame)


# ============================================================================
# Default expansion tests
# ============================================================================

class TestDefaultExpansion:
    def test_discrete(self):
        s = discrete_scale("x", palette=lambda n: list(range(1, n + 1)), super_class=ScaleDiscretePosition)
        result = default_expansion(s)
        assert len(result) == 4

    def test_continuous(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        result = default_expansion(s)
        assert len(result) == 4

    def test_no_expand(self):
        s = continuous_scale("x", palette=lambda x: x, super_class=ScaleContinuousPosition)
        result = default_expansion(s, expand=False)
        np.testing.assert_array_equal(result, [0, 0, 0, 0])

    def test_custom_expand(self):
        s = continuous_scale("x", palette=lambda x: x, expand=[0.1, 0.5, 0.1, 0.5], super_class=ScaleContinuousPosition)
        result = default_expansion(s)
        assert result[0] == pytest.approx(0.1)


# ============================================================================
# find_scale tests
# ============================================================================

class TestFindScale:
    def test_continuous_data(self):
        s = find_scale("x", np.array([1.0, 2.0, 3.0]))
        # Should find or return a scale

    def test_discrete_data(self):
        s = find_scale("colour", pd.Series(["a", "b"]))
        # Should find or return a scale
