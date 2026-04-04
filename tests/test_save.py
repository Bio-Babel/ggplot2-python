"""Tests for ggplot2_py.save — save and device detection."""

import pytest
from ggplot2_py import check_device


class TestCheckDevice:
    """Test check_device() function."""

    def test_png_from_filename(self):
        assert check_device(None, "plot.png") == "png"

    def test_pdf_from_filename(self):
        assert check_device(None, "plot.pdf") == "pdf"

    def test_svg_from_filename(self):
        assert check_device(None, "plot.svg") == "svg"

    def test_jpg_from_filename(self):
        assert check_device(None, "plot.jpg") == "jpg"

    def test_explicit_device(self):
        assert check_device("png", "whatever.txt") == "png"

    def test_no_extension_raises(self):
        with pytest.raises(Exception):
            check_device(None, "plot_no_ext")

    def test_unsupported_extension_raises(self):
        with pytest.raises(Exception):
            check_device(None, "plot.xyz")
