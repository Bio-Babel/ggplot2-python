"""Tests for ggplot2_py.annotation — annotation helpers."""

import pytest
from ggplot2_py import annotate, annotation_custom, Layer, is_layer


class TestAnnotate:
    """Test annotate() function."""

    def test_creates_layer(self):
        layer = annotate("text", x=1, y=2, label="hello")
        assert isinstance(layer, Layer)

    def test_is_layer(self):
        layer = annotate("text", x=1, y=2, label="hello")
        assert is_layer(layer)

    def test_rect_annotation(self):
        layer = annotate("rect", xmin=1, xmax=2, ymin=3, ymax=4, alpha=0.2)
        assert isinstance(layer, Layer)


class TestAnnotationCustom:
    """Test annotation_custom() function."""

    def test_creates_layer(self):
        layer = annotation_custom(None)
        assert isinstance(layer, Layer)

    def test_is_layer(self):
        layer = annotation_custom(None)
        assert is_layer(layer)
