"""Tests for ggplot2_py.aes — aesthetic mapping system."""

import pytest
from ggplot2_py.aes import (
    aes,
    after_stat,
    after_scale,
    stage,
    vars,
    is_mapping,
    standardise_aes_names,
    Mapping,
    AfterStat,
    AfterScale,
    Stage,
)


class TestAes:
    """Test aes() constructor."""

    def test_returns_mapping(self):
        m = aes()
        assert isinstance(m, Mapping)

    def test_xy_keys(self):
        m = aes(x="displ", y="hwy")
        assert "x" in m
        assert "y" in m
        assert m["x"] == "displ"
        assert m["y"] == "hwy"

    def test_color_alias(self):
        m = aes(color="class")
        assert "colour" in m
        assert "color" not in m
        assert m["colour"] == "class"

    def test_additional_kwargs(self):
        m = aes(x="a", y="b", size="c", fill="d")
        assert m["size"] == "c"
        assert m["fill"] == "d"

    def test_empty_aes(self):
        m = aes()
        assert len(m) == 0


class TestAfterStat:
    """Test after_stat helper."""

    def test_returns_after_stat(self):
        result = after_stat("count")
        assert isinstance(result, AfterStat)
        assert result.x == "count"

    def test_repr(self):
        result = after_stat("density")
        assert repr(result) == "AfterStat('density')"

    def test_equality(self):
        assert after_stat("count") == after_stat("count")
        assert after_stat("count") != after_stat("density")


class TestAfterScale:
    """Test after_scale helper."""

    def test_returns_after_scale(self):
        result = after_scale("fill")
        assert isinstance(result, AfterScale)
        assert result.x == "fill"

    def test_repr(self):
        result = after_scale("fill")
        assert repr(result) == "AfterScale('fill')"


class TestStage:
    """Test stage helper."""

    def test_returns_stage(self):
        result = stage(start="x", after_stat="count")
        assert isinstance(result, Stage)
        assert result.start == "x"
        assert isinstance(result.after_stat, AfterStat)
        assert result.after_stat.x == "count"

    def test_stage_with_after_scale(self):
        result = stage(start="class", after_scale="fill")
        assert result.start == "class"
        assert isinstance(result.after_scale, AfterScale)


class TestVars:
    """Test vars helper."""

    def test_positional(self):
        result = vars("a", "b")
        assert result == ["a", "b"]

    def test_empty(self):
        result = vars()
        assert result == []


class TestStandardiseAesNames:
    """Test standardise_aes_names."""

    def test_alias_mapping(self):
        result = standardise_aes_names(["color", "pch"])
        assert result == ["colour", "shape"]

    def test_passthrough(self):
        result = standardise_aes_names(["x", "y", "fill"])
        assert result == ["x", "y", "fill"]


class TestIsMapping:
    """Test is_mapping."""

    def test_true_for_mapping(self):
        assert is_mapping(aes()) is True

    def test_false_for_dict(self):
        assert is_mapping({"x": "a"}) is False

    def test_false_for_none(self):
        assert is_mapping(None) is False
