"""Targeted coverage tests for ggplot2_py.scale – round 4 (remaining gaps)."""

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
    expand_scale,
    expand_range4,
    default_expansion,
    _empty,
    _check_breaks_labels,
    _MappedDiscrete,
    mapped_discrete,
)

try:
    from ggplot2_py.scale import (
        _set_sec_axis,
        derive,
        is_derived,
        _is_position_aes,
        _unique0,
    )
    HAS_INTERNALS = True
except ImportError:
    HAS_INTERNALS = False


def _cont_pos(**kw):
    return continuous_scale("x", palette=lambda x: x,
                            super_class=ScaleContinuousPosition, **kw)


def _disc_pos(**kw):
    return discrete_scale("x", palette=lambda n: list(range(n)),
                          super_class=ScaleDiscretePosition, **kw)


def _bin_pos(**kw):
    kw.setdefault("breaks", [3.0, 7.0])
    return binned_scale("x", palette=lambda x: x,
                        super_class=ScaleBinnedPosition, **kw)


def _cont_colour(**kw):
    """Continuous colour-like scale for non-position tests."""
    return continuous_scale("colour", palette=lambda x: np.array(["#000000"] * len(x)),
                            **kw)


def _disc_colour(**kw):
    return discrete_scale("colour", palette=lambda n: [f"#{i:06x}" for i in range(n)],
                          **kw)


# ===========================================================================
# Helpers: _empty, _check_breaks_labels, expand_scale, expand_range4
# ===========================================================================

class TestHelpers:
    def test_empty_none(self):
        assert _empty(None) is True

    def test_empty_dataframe_zero(self):
        assert _empty(pd.DataFrame()) is True

    def test_empty_dataframe_nonempty(self):
        assert _empty(pd.DataFrame({"x": [1]})) is False

    def test_empty_dict_empty(self):
        assert _empty({}) is True

    def test_empty_dict_nonempty(self):
        assert _empty({"a": 1}) is False

    def test_empty_other(self):
        # Returns False for non-None non-container
        assert _empty(42) is False

    def test_check_breaks_labels_length_mismatch(self):
        with pytest.raises(Exception):
            _check_breaks_labels([1, 2], ["a"])

    def test_check_breaks_labels_both_none(self):
        _check_breaks_labels(None, None)  # Should not raise

    def test_expand_scale_deprecated(self):
        with pytest.warns(Warning):
            result = expand_scale(mult=0.05, add=0)
        assert len(result) == 4

    def test_expand_range4_two_element(self):
        result = expand_range4([0.0, 10.0], [0.05, 0.5])
        assert len(result) == 2

    def test_expand_range4_non_finite(self):
        result = expand_range4([np.nan, np.nan], [0.05, 0.0, 0.05, 0.0])
        assert np.isinf(result[0])

    def test_expand_range4_bad_expand(self):
        with pytest.raises(Exception):
            expand_range4([0, 10], [1, 2, 3])

    def test_expansion_bad_lengths(self):
        with pytest.raises(Exception):
            expansion(mult=[1, 2, 3], add=0)


class TestMappedDiscrete:
    def test_mapped_discrete_creation(self):
        md = mapped_discrete(np.array([1.0, 2.0, 3.0]))
        assert isinstance(md, _MappedDiscrete)
        assert len(md) == 3


class TestUniqueAndPositionAes:
    @pytest.mark.skipif(not HAS_INTERNALS, reason="internals not exported")
    def test_unique0_none(self):
        result = _unique0(None)
        assert len(result) == 0

    @pytest.mark.skipif(not HAS_INTERNALS, reason="internals not exported")
    def test_unique0_values(self):
        result = _unique0([3, 1, 2, 1, 3])
        assert len(result) == 3

    @pytest.mark.skipif(not HAS_INTERNALS, reason="internals not exported")
    def test_is_position_aes(self):
        assert _is_position_aes(["x"]) is True
        assert _is_position_aes(["colour"]) is False


# ===========================================================================
# default_expansion
# ===========================================================================

class TestDefaultExpansion:
    def test_default_expansion_continuous(self):
        sc = _cont_pos()
        sc.train(np.array([0.0, 10.0]))
        exp = default_expansion(sc)
        assert len(exp) == 4

    def test_default_expansion_discrete(self):
        sc = _disc_pos()
        sc.train(np.array(["a", "b"]))
        exp = default_expansion(sc)
        assert len(exp) == 4

    def test_default_expansion_no_expand(self):
        sc = _cont_pos()
        sc.train(np.array([0.0, 10.0]))
        exp = default_expansion(sc, expand=False)
        assert exp[0] == 0.0  # no expansion


# ===========================================================================
# ScaleContinuous – deeper coverage
# ===========================================================================

class TestScaleContinuousDeep:
    def test_map_with_palette(self):
        """Cover lines 786-798: continuous map with palette lookup."""
        sc = continuous_scale(
            "colour",
            palette=lambda x: np.array(["red", "blue"]) if len(x) >= 2
                    else np.array(["red"]),
        )
        sc.train(np.array([0.0, 10.0]))
        mapped = sc.map(np.array([0.0, 5.0, 10.0]))
        assert mapped is not None

    def test_get_breaks_n_breaks_fallback(self):
        """Cover lines 905-908: callable breaks with n_breaks that fails."""
        def breaks_no_n(lim):
            return np.linspace(lim[0], lim[1], 5)
        sc = _cont_pos(breaks=breaks_no_n)
        sc.train(np.array([0.0, 10.0]))
        sc.n_breaks = 3
        brk = sc.get_breaks()
        assert brk is not None

    def test_get_breaks_minor_none(self):
        """Cover lines 947-948: minor_breaks=None returns None."""
        sc = _cont_pos(breaks=[2.0, 5.0, 8.0])
        sc.train(np.array([0.0, 10.0]))
        sc.minor_breaks = None
        minor = sc.get_breaks_minor()
        assert minor is None

    def test_get_breaks_minor_zero_range(self):
        """Cover line 942: zero range returns None."""
        sc = _cont_pos(breaks=[5.0])
        sc.train(np.array([5.0, 5.0]))
        sc.limits = np.array([5.0, 5.0])
        minor = sc.get_breaks_minor()
        assert minor is None

    def test_get_breaks_minor_callable(self):
        """Cover lines 957-961: callable minor breaks."""
        sc = _cont_pos(breaks=[2.0, 5.0, 8.0])
        sc.train(np.array([0.0, 10.0]))
        sc.minor_breaks = lambda lim: np.linspace(lim[0], lim[1], 3)
        minor = sc.get_breaks_minor(b=np.array([2.0, 5.0, 8.0]))
        assert minor is not None

    def test_get_breaks_minor_explicit_array(self):
        """Cover lines 962-964: explicit minor breaks array."""
        sc = _cont_pos(breaks=[2.0, 5.0, 8.0])
        sc.train(np.array([0.0, 10.0]))
        sc.minor_breaks = np.array([2.5, 7.5])
        minor = sc.get_breaks_minor(b=np.array([2.0, 5.0, 8.0]))
        assert minor is not None

    def test_get_labels_callable_via_get_labels(self):
        """Cover line 981, 992: callable labels with explicit breaks."""
        sc = _cont_pos(breaks=[2.0, 5.0, 8.0],
                        labels=lambda b: [f"{v:.0f}" for v in b])
        sc.train(np.array([0.0, 10.0]))
        labels = sc.get_labels(breaks=np.array([2.0, 5.0]))
        assert labels == ["2", "5"]

    def test_get_labels_returns_none_when_breaks_none(self):
        """Cover line 980-981: breaks resolves to None -> returns None."""
        sc = _cont_pos()
        sc.train(np.array([0.0, 10.0]))
        sc.breaks = None
        result = sc.get_labels()
        assert result is None

    def test_get_labels_none_labels(self):
        """Cover line 987-988: labels=None."""
        sc = _cont_pos(breaks=[2.0], labels=None)
        # pass explicit None labels
        sc.train(np.array([0.0, 10.0]))
        sc.labels = None
        result = sc.get_labels(breaks=np.array([2.0]))
        assert result is None

    def test_get_labels_explicit_list(self):
        """Cover line 993: explicit label list."""
        sc = _cont_pos(breaks=[2.0, 5.0], labels=["two", "five"])
        sc.train(np.array([0.0, 10.0]))
        labels = sc.get_labels(breaks=np.array([2.0, 5.0]))
        assert labels == ["two", "five"]

    def test_break_info(self):
        """Cover lines 1016-1042: break_info on continuous scale."""
        sc = _cont_pos(breaks=[2.0, 5.0, 8.0],
                        labels=lambda b: [f"{v:.0f}" for v in b])
        sc.train(np.array([0.0, 10.0]))
        sc.minor_breaks = None  # avoid transformation.minor_breaks bug
        info = sc.break_info(range=np.array([0.0, 10.0]))
        assert "range" in info
        assert "labels" in info
        assert "major" in info

    def test_break_info_with_explicit_minor(self):
        """Cover line 1020: minor breaks containing NaN."""
        sc = _cont_pos(breaks=[2.0, 5.0, 8.0],
                        labels=lambda b: [f"{v:.0f}" for v in b])
        sc.train(np.array([0.0, 10.0]))
        sc.minor_breaks = np.array([2.5, np.nan, 7.5])
        info = sc.break_info(range=np.array([0.0, 10.0]))
        # Should succeed with NaN filtered out

    def test_break_info_major_none(self):
        """Cover line 1030: major is None."""
        sc = _cont_pos(breaks=[2.0, 5.0],
                        labels=lambda b: [f"{v:.0f}" for v in b])
        sc.train(np.array([0.0, 10.0]))
        sc.breaks = None
        sc.minor_breaks = None
        info = sc.break_info(range=np.array([0.0, 10.0]))
        assert info["major"] is None

    def test_zero_range_breaks(self):
        """Cover line 899: zero range returns single break."""
        sc = _cont_pos(breaks=[5.0])
        sc.train(np.array([5.0, 5.0]))
        sc.limits = np.array([5.0, 5.0])
        brk = sc.get_breaks()
        if brk is not None:
            assert len(brk) >= 1

    def test_get_limits_callable(self):
        """Cover the callable limits branch in ScaleContinuous."""
        sc = _cont_pos(limits=lambda r: np.array([0.0, 100.0]))
        sc.train(np.array([1.0, 5.0]))
        lim = sc.get_limits()
        assert lim is not None


# ===========================================================================
# ScaleDiscrete – deeper coverage
# ===========================================================================

class TestScaleDiscreteDeep:
    def test_map_empty_limits(self):
        """Cover line 1106: empty limits."""
        sc = _disc_colour()
        mapped = sc.map(np.array(["a"]), limits=[])
        assert len(mapped) == 1

    def test_map_with_na(self):
        """Cover line 1130: na_translate False."""
        sc = _disc_colour()
        sc.na_translate = False
        sc.train(np.array(["a", "b"]))
        mapped = sc.map(np.array(["a", "c"]))
        assert len(mapped) == 2

    def test_map_cached_palette(self):
        """Cover lines 1113-1114: palette cache hit."""
        sc = _disc_colour()
        sc.train(np.array(["a", "b"]))
        sc.map(np.array(["a"]))  # first call populates cache
        sc.map(np.array(["a"]))  # second call uses cache

    def test_map_dict_palette(self):
        """Cover line 1126: dict palette."""
        sc = discrete_scale("colour", palette=lambda n: {i: f"#{i:06x}" for i in range(n)})
        sc.train(np.array(["x", "y"]))
        mapped = sc.map(np.array(["x"]))
        assert len(mapped) == 1

    def test_rescale(self):
        """Cover ScaleDiscrete.rescale."""
        sc = _disc_colour()
        sc.train(np.array(["a", "b", "c"]))
        result = sc.rescale(np.array(["a", "b"]))
        assert len(result) == 2

    def test_dimension_empty(self):
        """Cover line 1171: empty limits."""
        sc = _disc_colour()
        dim = sc.dimension()
        assert len(dim) == 2

    def test_get_breaks_empty(self):
        """Cover line 1176: empty scale."""
        sc = _disc_colour()
        brk = sc.get_breaks()
        assert brk is not None

    def test_get_breaks_callable(self):
        """Cover line 1190: callable breaks (discrete)."""
        sc = _disc_colour(breaks=lambda lim: lim[:1])
        sc.train(np.array(["a", "b", "c"]))
        brk = sc.get_breaks()
        assert len(brk) == 1

    def test_get_breaks_explicit_filter(self):
        """Cover line 1190: explicit breaks filtered to limits."""
        sc = _disc_colour(breaks=["a", "z"])
        sc.train(np.array(["a", "b"]))
        brk = sc.get_breaks()
        assert "a" in brk

    def test_get_breaks_minor_callable(self):
        """Cover line 1205: callable minor breaks (discrete)."""
        sc = _disc_colour()
        sc.minor_breaks = lambda lim: ["minor1"]
        sc.train(np.array(["a", "b"]))
        result = sc.get_breaks_minor()
        assert result == ["minor1"]

    def test_get_labels_empty(self):
        """Cover line 1209: empty scale returns []."""
        sc = _disc_colour()
        labels = sc.get_labels()
        assert labels == []

    def test_get_labels_none(self):
        """Cover line 1213: breaks None -> None."""
        sc = _disc_colour()
        sc.train(np.array(["a"]))
        sc.breaks = None
        assert sc.get_labels() is None

    def test_get_labels_callable(self):
        """Cover line 1221: callable labels."""
        sc = _disc_colour(labels=lambda brk: [f"L-{b}" for b in brk])
        sc.train(np.array(["a", "b"]))
        labels = sc.get_labels()
        assert labels[0].startswith("L-")

    def test_clone(self):
        sc = _disc_colour()
        sc.train(np.array(["a", "b"]))
        c = sc.clone()
        assert c.range is not sc.range

    def test_break_info(self):
        """Use numeric palette for break_info."""
        sc = discrete_scale("x", palette=lambda n: list(range(1, n + 1)),
                            super_class=ScaleDiscretePosition)
        sc.train(np.array(["a", "b"]))
        info = sc.break_info(range=(1.0, 2.0))
        assert "labels" in info


# ===========================================================================
# ScaleBinned – cover all methods via binned_scale factory
# ===========================================================================

class TestScaleBinnedDeep:
    def test_binned_train_and_breaks_explicit(self):
        """Cover ScaleBinned.get_breaks paths with explicit breaks."""
        sc = binned_scale("x", palette=lambda x: x, super_class=ScaleBinned,
                          breaks=[3.0, 7.0])
        sc.train(np.array([1.0, 5.0, 10.0]))
        brk = sc.get_breaks()
        assert brk is not None

    def test_binned_get_breaks_not_nice(self):
        """Cover lines 1400-1401: not nice -> linspace."""
        sc = binned_scale("x", palette=lambda x: x, super_class=ScaleBinned)
        sc.train(np.array([0.0, 100.0]))
        sc.nice_breaks = False
        brk = sc.get_breaks()
        assert brk is not None

    def test_binned_get_breaks_callable(self):
        """Cover lines 1406-1410: callable breaks."""
        sc = binned_scale("x", palette=lambda x: x, super_class=ScaleBinned,
                          breaks=lambda lim: np.array([3.0, 7.0]))
        sc.train(np.array([0.0, 10.0]))
        brk = sc.get_breaks()
        assert len(brk) == 2

    def test_binned_get_breaks_callable_no_n(self):
        """Cover lines 1406-1410: callable that doesn't accept n."""
        def no_n_breaks(lim):
            return np.array([3.0])
        sc = binned_scale("x", palette=lambda x: x, super_class=ScaleBinned,
                          breaks=no_n_breaks)
        sc.train(np.array([0.0, 10.0]))
        brk = sc.get_breaks()
        assert brk is not None

    def test_binned_get_labels_callable(self):
        """Cover line 1432: callable labels."""
        sc = binned_scale("x", palette=lambda x: x, super_class=ScaleBinned,
                          breaks=[3.0, 7.0],
                          labels=lambda brk: [f"{v:.0f}" for v in brk])
        sc.train(np.array([0.0, 10.0]))
        labels = sc.get_labels()
        assert labels is not None

    def test_binned_get_labels_explicit(self):
        """Cover line 1433: explicit labels."""
        sc = binned_scale("x", palette=lambda x: x, super_class=ScaleBinned,
                          breaks=[3.0, 7.0], labels=["low", "high"])
        sc.train(np.array([0.0, 10.0]))
        labels = sc.get_labels()
        assert labels == ["low", "high"]

    def test_binned_get_labels_none_breaks(self):
        """Cover line 1422-1423: breaks None -> labels None."""
        sc = binned_scale("x", palette=lambda x: x, super_class=ScaleBinned,
                          breaks=[3.0])
        sc.train(np.array([0.0, 10.0]))
        sc.breaks = None  # set after creation to avoid waiver
        brk = sc.get_breaks()
        assert brk is None
        labels = sc.get_labels()
        assert labels is None

    def test_binned_get_labels_none_labels(self):
        """Cover line 1428: labels=None."""
        sc = binned_scale("x", palette=lambda x: x, super_class=ScaleBinned,
                          breaks=[3.0])
        sc.train(np.array([0.0, 10.0]))
        sc.labels = None
        labels = sc.get_labels()
        assert labels is None

    def test_binned_clone(self):
        sc = binned_scale("x", palette=lambda x: x, super_class=ScaleBinned,
                          breaks=[3.0, 7.0])
        sc.train(np.array([0.0, 10.0]))
        c = sc.clone()
        assert c.range is not sc.range

    def test_binned_break_info(self):
        sc = binned_scale("x", palette=lambda x: x, super_class=ScaleBinned,
                          breaks=[3.0, 7.0],
                          labels=lambda b: [f"{v:.0f}" for v in b])
        sc.train(np.array([0.0, 10.0]))
        info = sc.break_info()
        assert "range" in info

    def test_binned_get_limits_callable(self):
        """Cover line 1371-1374: callable limits in binned."""
        sc = binned_scale("x", palette=lambda x: x, super_class=ScaleBinned,
                          limits=lambda r: np.array([0.0, 50.0]))
        sc.train(np.array([1.0, 20.0]))
        lim = sc.get_limits()
        assert lim is not None

    def test_binned_get_limits_with_nan(self):
        """Cover line 1367: limits with NaN filled from range."""
        sc = binned_scale("x", palette=lambda x: x, super_class=ScaleBinned,
                          limits=[np.nan, 50.0])
        sc.train(np.array([1.0, 20.0]))
        lim = sc.get_limits()
        assert not np.isnan(lim[0])

    def test_binned_dimension(self):
        sc = binned_scale("x", palette=lambda x: x, super_class=ScaleBinned,
                          breaks=[3.0, 7.0])
        sc.train(np.array([0.0, 10.0]))
        dim = sc.dimension()
        assert len(dim) == 2

    def test_binned_map(self):
        """Cover ScaleBinned.map lines 1304-1339."""
        sc = binned_scale("colour", palette=lambda x: np.linspace(0, 1, len(x)),
                          super_class=ScaleBinned, breaks=[3.0, 7.0])
        sc.train(np.array([0.0, 10.0]))
        mapped = sc.map(np.array([2.0, 5.0, 8.0]))
        assert len(mapped) == 3

    def test_binned_rescale(self):
        sc = binned_scale("x", palette=lambda x: x, super_class=ScaleBinned,
                          breaks=[3.0, 7.0])
        sc.train(np.array([0.0, 10.0]))
        r = sc.rescale(np.array([2.0, 8.0]))
        assert len(r) == 2


# ===========================================================================
# ScaleBinnedPosition
# ===========================================================================

class TestScaleBinnedPosition:
    def test_train_and_map(self):
        sc = _bin_pos()
        sc.train(np.array([0.0, 10.0]))
        mapped = sc.map(np.array([2.0, 8.0]))
        assert len(mapped) == 2

    def test_reset(self):
        sc = _bin_pos()
        sc.train(np.array([0.0, 10.0]))
        sc.reset()
        assert sc.after_stat is True

    def test_get_breaks_show_limits(self):
        sc = _bin_pos()
        sc.train(np.array([0.0, 10.0]))
        sc.show_limits = True
        brk = sc.get_breaks()
        assert brk is not None


# ===========================================================================
# ScaleContinuousPosition – break_info with secondary
# ===========================================================================

class TestScaleContinuousPositionDeep:
    def test_break_info_no_secondary(self):
        sc = _cont_pos(breaks=[2.0, 5.0, 8.0],
                        labels=lambda b: [f"{v:.0f}" for v in b])
        sc.train(np.array([0.0, 10.0]))
        sc.minor_breaks = None
        info = sc.break_info(range=np.array([0.0, 10.0]))
        assert "range" in info

    def test_sec_name_waiver(self):
        sc = _cont_pos()
        name = sc.sec_name()
        assert is_waiver(name)


# ===========================================================================
# ScaleDiscretePosition
# ===========================================================================

class TestScaleDiscretePositionDeep:
    def test_train_continuous(self):
        """Cover line 1563: train with continuous data."""
        sc = _disc_pos()
        sc.train(np.array(["a", "b"]))
        sc.train(np.array([1.0, 2.0]))  # continuous
        # range_c should be trained

    def test_map_discrete(self):
        """Cover line 1566+: map discrete values."""
        sc = _disc_pos()
        sc.train(np.array(["a", "b"]))
        mapped = sc.map(np.array(["a", "b"]))
        assert len(mapped) == 2

    def test_map_continuous(self):
        """Cover line 1588: map continuous (non-discrete)."""
        sc = _disc_pos()
        sc.train(np.array(["a", "b"]))
        mapped = sc.map(np.array([1.0, 2.0]))
        assert len(mapped) == 2

    def test_dimension(self):
        sc = _disc_pos()
        sc.train(np.array(["a", "b"]))
        dim = sc.dimension()
        assert len(dim) == 2

    def test_dimension_empty(self):
        """Cover line 1588: empty mapped."""
        sc = _disc_pos()
        dim = sc.dimension()
        assert len(dim) == 2

    def test_clone(self):
        sc = _disc_pos()
        sc.train(np.array(["a", "b"]))
        c = sc.clone()
        assert hasattr(c, "range_c")

    def test_sec_name(self):
        sc = _disc_pos()
        name = sc.sec_name()
        assert is_waiver(name)

    def test_reset_with_range_c(self):
        sc = _disc_pos()
        sc.train(np.array(["a"]))
        sc.reset()

    def test_is_empty(self):
        sc = _disc_pos()
        assert sc.is_empty()

    def test_get_limits_callable(self):
        sc = _disc_pos(limits=lambda r: ["a"])
        sc.train(np.array(["a", "b"]))
        lim = sc.get_limits()
        assert lim == ["a"]

    def test_map_empty_limits(self):
        sc = _disc_pos()
        mapped = sc.map(np.array(["a"]), limits=[])
        assert len(mapped) == 0


# ===========================================================================
# Identity scales
# ===========================================================================

class TestIdentityScales:
    def test_continuous_identity_map_categorical(self):
        """Cover line 1668: Categorical input."""
        sc = ScaleContinuousIdentity()
        sc.aesthetics = ["colour"]
        sc.guide = "legend"
        cat = pd.Categorical(["red", "blue"])
        mapped = sc.map(cat)
        assert mapped is not None

    def test_continuous_identity_train_no_guide(self):
        """Cover line 1668: guide='none' skips training."""
        sc = ScaleContinuousIdentity()
        sc.aesthetics = ["colour"]
        sc.guide = "none"
        sc.range = type("R", (), {"train": lambda s, x: None, "range": None, "reset": lambda s: None})()
        sc.train(np.array([1.0, 2.0]))

    def test_discrete_identity_map_categorical(self):
        """Cover line 1677: Categorical input."""
        sc = ScaleDiscreteIdentity()
        sc.aesthetics = ["fill"]
        sc.guide = "legend"
        cat = pd.Categorical(["red", "blue"])
        mapped = sc.map(cat)
        assert mapped is not None

    def test_discrete_identity_train_no_guide(self):
        """Cover line 1683: guide='none' skips training."""
        sc = ScaleDiscreteIdentity()
        sc.aesthetics = ["fill"]
        sc.guide = "none"
        sc.range = type("R", (), {"train": lambda s, x, drop=True: None, "range": None, "reset": lambda s: None})()
        sc.train(np.array(["a", "b"]))


# ===========================================================================
# AxisSecondary
# ===========================================================================

class TestAxisSecondaryDeep:
    @pytest.mark.skipif(not HAS_INTERNALS, reason="internals not exported")
    def test_init_with_derive(self):
        """Cover lines 2377-2388: init with derived settings."""
        ax = sec_axis(transform=lambda x: x * 2, name=derive(), breaks=derive(),
                      labels=derive(), guide=derive())
        primary = _cont_pos(breaks=[2.0, 5.0, 8.0],
                             labels=lambda b: [f"{v:.0f}" for v in b])
        primary.train(np.array([0.0, 10.0]))
        primary.name = "X axis"
        ax.init(primary)
        assert ax.name == "X axis"
        assert ax.breaks == [2.0, 5.0, 8.0]

    @pytest.mark.skipif(not HAS_INTERNALS, reason="internals not exported")
    def test_init_waiver_breaks(self):
        """Cover lines 2382-2384: waiver breaks get trans.breaks (may fail if transform lacks .breaks)."""
        ax = sec_axis(transform=lambda x: x * 2, breaks=[1.0, 5.0])
        primary = _cont_pos(breaks=[2.0, 5.0, 8.0])
        primary.train(np.array([0.0, 10.0]))
        ax.init(primary)
        # breaks should remain as explicitly set
        assert ax.breaks is not None

    def test_break_info(self):
        """Cover lines 2423-2445: secondary break_info."""
        ax = sec_axis(transform=lambda x: x * 2, breaks=[1.0, 5.0, 9.0])
        primary = _cont_pos(breaks=[2.0, 5.0, 8.0])
        primary.train(np.array([0.0, 10.0]))
        info = ax.break_info(np.array([0.0, 10.0]), primary)
        assert "sec.range" in info
        assert "sec.major" in info

    def test_break_info_empty(self):
        """Cover line 2420: empty secondary axis."""
        ax = AxisSecondary()
        assert ax.empty()
        info = ax.break_info(np.array([0.0, 10.0]), _cont_pos())
        assert info == {}

    def test_transform_range(self):
        ax = sec_axis(transform=lambda x: x * 3)
        result = ax.transform_range(np.array([0.0, 10.0]))
        assert result[1] == 30.0

    def test_sec_axis_trans_deprecated(self):
        """Cover line 2482-2483: trans= deprecated."""
        with pytest.warns(Warning):
            ax = sec_axis(trans=lambda x: x * 2)
        assert ax.trans is not None


# ===========================================================================
# sec_axis / dup_axis / _set_sec_axis
# ===========================================================================

class TestSecAxisHelpers:
    def test_sec_axis_defaults(self):
        ax = sec_axis(transform=lambda x: x)
        assert not ax.empty()

    def test_dup_axis(self):
        ax = dup_axis()
        assert not ax.empty()

    @pytest.mark.skipif(not HAS_INTERNALS, reason="internals not exported")
    def test_set_sec_axis(self):
        sc = _cont_pos()
        ax = sec_axis(transform=lambda x: x)
        result = _set_sec_axis(ax, sc)
        assert hasattr(result, "secondary_axis")

    @pytest.mark.skipif(not HAS_INTERNALS, reason="internals not exported")
    def test_set_sec_axis_waiver(self):
        sc = _cont_pos()
        result = _set_sec_axis(waiver(), sc)
        # Should not modify


# ===========================================================================
# scale_type
# ===========================================================================

class TestScaleType:
    def test_categorical_series(self):
        s = pd.Series(pd.Categorical(["a", "b"]))
        types = scale_type(s)
        assert "discrete" in types

    def test_ordered_categorical(self):
        s = pd.Series(pd.Categorical(["a", "b"], ordered=True))
        types = scale_type(s)
        assert "ordinal" in types

    def test_bool_series(self):
        s = pd.Series([True, False])
        types = scale_type(s)
        assert "discrete" in types

    def test_datetime_series(self):
        s = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02"]))
        types = scale_type(s)
        assert "datetime" in types

    def test_numeric_series(self):
        s = pd.Series([1.0, 2.0])
        types = scale_type(s)
        assert "continuous" in types

    def test_object_series(self):
        s = pd.Series(["a", "b"], dtype=object)
        types = scale_type(s)
        assert "discrete" in types

    def test_numpy_string(self):
        arr = np.array(["a", "b"])
        types = scale_type(arr)
        assert "discrete" in types

    def test_numpy_datetime(self):
        arr = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64")
        types = scale_type(arr)
        assert "datetime" in types

    def test_numpy_bool(self):
        arr = np.array([True, False])
        types = scale_type(arr)
        assert "discrete" in types


# ===========================================================================
# find_scale
# ===========================================================================

class TestFindScale:
    def test_find_scale_none_data(self):
        result = find_scale("x", None)
        assert result is None

    def test_find_scale_continuous(self):
        result = find_scale("x", pd.Series([1.0, 2.0]))
        # Might return None if scales module doesn't have exact function

    def test_find_scale_discrete(self):
        result = find_scale("colour", pd.Series(["a", "b"], dtype=object))


# ===========================================================================
# ScalesList
# ===========================================================================

class TestScalesListDeep:
    def test_non_position_scales(self):
        sl = ScalesList()
        sc1 = _cont_pos()
        sc2 = _cont_colour()
        sl.add(sc1)
        sl.add(sc2)
        nps = sl.non_position_scales()
        assert nps.n() >= 1

    def test_has_scale(self):
        sl = ScalesList()
        sc = _cont_pos()
        sl.add(sc)
        assert sl.has_scale("x") is True
        assert sl.has_scale("colour") is False


# ===========================================================================
# Factory functions: deeper branches
# ===========================================================================

class TestFactoryFunctions:
    def test_continuous_scale_string_transform(self):
        """Cover line 1798-1799: string transform -> as_transform."""
        sc = continuous_scale("x", palette=lambda x: x,
                              transform="identity",
                              super_class=ScaleContinuousPosition)
        assert sc.trans is not None

    def test_continuous_scale_with_limits(self):
        """Cover lines 1782-1783: limits transform."""
        sc = continuous_scale("x", palette=lambda x: x, limits=[0, 10],
                              super_class=ScaleContinuousPosition)
        assert sc.limits is not None

    def test_continuous_scale_trans_deprecated(self):
        """Cover line 1908/1912: trans= deprecated."""
        with pytest.warns(Warning):
            sc = continuous_scale("x", palette=lambda x: x, trans="identity",
                                  super_class=ScaleContinuousPosition)

    def test_discrete_scale_bad_position(self):
        """Cover discrete_scale position validation."""
        with pytest.raises(Exception):
            discrete_scale("x", palette=lambda n: list(range(n)),
                           position="center")

    def test_binned_scale_trans_deprecated(self):
        """Cover lines 2018-2019: trans= deprecated."""
        with pytest.warns(Warning):
            sc = binned_scale("x", palette=lambda x: x, trans="identity",
                              super_class=ScaleBinned)

    def test_binned_scale_limits(self):
        """Cover lines 2028, 2031: binned with limits."""
        sc = binned_scale("x", palette=lambda x: x, limits=[0, 10],
                          super_class=ScaleBinned)
        assert sc.limits is not None

    def test_binned_scale_bad_position(self):
        """Cover line 2028: bad position."""
        with pytest.raises(Exception):
            binned_scale("x", palette=lambda x: x, position="center")


# ===========================================================================
# Scale base class abstract methods
# ===========================================================================

class TestScaleAbstractMethods:
    def test_transform_not_implemented(self):
        sc = Scale()
        sc.aesthetics = ["x"]
        sc.range = type("R", (), {"range": None, "reset": lambda s: None})()
        sc.limits = None
        with pytest.raises(Exception):
            sc.transform(np.array([1.0]))

    def test_train_not_implemented(self):
        sc = Scale()
        sc.aesthetics = ["x"]
        sc.range = type("R", (), {"range": None, "reset": lambda s: None})()
        sc.limits = None
        with pytest.raises(Exception):
            sc.train(np.array([1.0]))

    def test_map_not_implemented(self):
        sc = Scale()
        sc.aesthetics = ["x"]
        sc.range = type("R", (), {"range": None, "reset": lambda s: None})()
        sc.limits = None
        with pytest.raises(Exception):
            sc.map(np.array([1.0]))

    def test_rescale_not_implemented(self):
        sc = Scale()
        sc.aesthetics = ["x"]
        sc.range = type("R", (), {"range": None, "reset": lambda s: None})()
        sc.limits = None
        with pytest.raises(Exception):
            sc.rescale(np.array([1.0]))

    def test_dimension_not_implemented(self):
        sc = Scale()
        sc.aesthetics = ["x"]
        sc.range = type("R", (), {"range": None, "reset": lambda s: None})()
        sc.limits = None
        with pytest.raises(Exception):
            sc.dimension()

    def test_get_breaks_not_implemented(self):
        sc = Scale()
        sc.aesthetics = ["x"]
        sc.range = type("R", (), {"range": None, "reset": lambda s: None})()
        sc.limits = None
        with pytest.raises(Exception):
            sc.get_breaks()

    def test_get_labels_not_implemented(self):
        sc = Scale()
        sc.aesthetics = ["x"]
        sc.range = type("R", (), {"range": None, "reset": lambda s: None})()
        sc.limits = None
        with pytest.raises(Exception):
            sc.get_labels()

    def test_clone_not_implemented(self):
        sc = Scale()
        sc.aesthetics = ["x"]
        sc.range = type("R", (), {"range": None, "reset": lambda s: None})()
        sc.limits = None
        with pytest.raises(Exception):
            sc.clone()

    def test_is_discrete_not_implemented(self):
        sc = Scale()
        sc.aesthetics = ["x"]
        sc.range = type("R", (), {"range": None, "reset": lambda s: None})()
        sc.limits = None
        with pytest.raises(Exception):
            sc.is_discrete()

    def test_break_info_not_implemented(self):
        sc = Scale()
        sc.aesthetics = ["x"]
        sc.range = type("R", (), {"range": None, "reset": lambda s: None})()
        sc.limits = None
        with pytest.raises(Exception):
            sc.break_info()

    def test_break_positions(self):
        """Cover line 582: break_positions calls map(get_breaks(range))."""
        sc = _cont_pos(breaks=[2.0, 5.0, 8.0])
        sc.train(np.array([0.0, 10.0]))
        bp = sc.break_positions()
        assert bp is not None

    def test_get_transformation_default(self):
        """Cover line 586 base get_transformation."""
        sc = Scale()
        t = sc.get_transformation()
        assert t is not None
