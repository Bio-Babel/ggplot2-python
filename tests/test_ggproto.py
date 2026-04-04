"""Tests for ggplot2_py.ggproto — the ggproto object system."""

import pytest
from ggplot2_py.ggproto import GGProto, ggproto, ggproto_parent, is_ggproto
from ggplot2_py import Geom, GeomPoint


class TestGGProto:
    """Test ggproto factory and GGProto base class."""

    def test_ggproto_creates_subclass(self):
        MyGeom = ggproto("MyGeom", Geom)
        assert issubclass(MyGeom, Geom)
        assert issubclass(MyGeom, GGProto)

    def test_ggproto_class_name(self):
        MyGeom = ggproto("MyGeom", Geom)
        assert MyGeom._class_name == "MyGeom"

    def test_ggproto_with_members(self):
        MyObj = ggproto("MyObj", GGProto, colour="red", size=10)
        assert MyObj.colour == "red"
        assert MyObj.size == 10

    def test_inheritance_override(self):
        Parent = ggproto("Parent", GGProto, value="parent")
        Child = ggproto("Child", Parent, value="child")
        assert Parent.value == "parent"
        assert Child.value == "child"


class TestIsGGProto:
    """Test is_ggproto predicate."""

    def test_true_for_class(self):
        assert is_ggproto(GeomPoint) is True

    def test_true_for_instance(self):
        assert is_ggproto(GeomPoint()) is True

    def test_true_for_base(self):
        assert is_ggproto(GGProto) is True

    def test_false_for_string(self):
        assert is_ggproto("not a ggproto") is False

    def test_false_for_none(self):
        assert is_ggproto(None) is False


class TestGGProtoRepr:
    """Test repr for ggproto classes and instances."""

    def test_class_repr(self):
        result = repr(GeomPoint)
        assert "GeomPoint" in result

    def test_instance_repr(self):
        gp = GeomPoint()
        result = repr(gp)
        assert "GeomPoint" in result


class TestGGProtoParent:
    """Test ggproto_parent for parent method dispatch."""

    def test_parent_proxy(self):
        Parent = ggproto("Parent", GGProto, value="parent_val")
        Child = ggproto("Child", Parent, value="child_val")
        inst = Child()
        proxy = ggproto_parent(Parent, inst)
        assert "parent proxy" in repr(proxy).lower() or "Parent" in repr(proxy)
