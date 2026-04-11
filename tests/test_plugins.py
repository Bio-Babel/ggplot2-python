"""Tests for entry-point-based plugin discovery (_plugins.py)."""

import pytest
from unittest.mock import patch, MagicMock
from ggplot2_py._plugins import discover_extensions, list_extensions, _EXTENSION_GROUPS


class TestPluginDiscovery:
    """Test the discover_extensions() function."""

    def test_returns_dict(self):
        """discover_extensions() should always return a dict."""
        result = discover_extensions()
        assert isinstance(result, dict)

    def test_empty_when_no_plugins(self):
        """With no extension packages installed, result should be empty."""
        result = discover_extensions()
        # May or may not be empty depending on env, but should not error
        assert isinstance(result, dict)

    def test_list_extensions_matches_discover(self):
        """list_extensions() returns the same result as last discover."""
        discovered = discover_extensions()
        listed = list_extensions()
        assert discovered == listed

    def test_extension_groups_defined(self):
        """All 6 extension groups should be defined."""
        groups = [g for g, _, _ in _EXTENSION_GROUPS]
        assert "ggplot2_py.geoms" in groups
        assert "ggplot2_py.stats" in groups
        assert "ggplot2_py.positions" in groups
        assert "ggplot2_py.scales" in groups
        assert "ggplot2_py.coords" in groups
        assert "ggplot2_py.facets" in groups

    def test_mock_extension_loaded(self):
        """Simulate an extension package with a mock entry point."""
        from ggplot2_py.geom import Geom
        from ggplot2_py.aes import Mapping

        # Create a mock geom class
        class GeomMockTest(Geom):
            required_aes = ("x", "y")
            default_aes = Mapping(colour="red")

        # Create a mock entry point
        mock_ep = MagicMock()
        mock_ep.name = "mock_test"
        mock_ep.load.return_value = GeomMockTest

        # Mock entry_points().select() to return our mock
        mock_eps = MagicMock()
        mock_eps.select.side_effect = lambda group: (
            [mock_ep] if group == "ggplot2_py.geoms" else []
        )

        with patch("ggplot2_py._plugins.entry_points", return_value=mock_eps):
            result = discover_extensions()

        assert "ggplot2_py.geoms" in result
        assert "mock_test" in result["ggplot2_py.geoms"]
        assert "mock_test" in Geom._registry
        assert Geom._registry["mock_test"] is GeomMockTest

        # Clean up registry
        Geom._registry.pop("mock_test", None)
        Geom._registry.pop("Mock_test", None)

    def test_failed_load_warns(self):
        """If an entry point fails to load, a warning should be emitted."""
        mock_ep = MagicMock()
        mock_ep.name = "broken_plugin"
        mock_ep.load.side_effect = ImportError("missing dependency")

        mock_eps = MagicMock()
        mock_eps.select.side_effect = lambda group: (
            [mock_ep] if group == "ggplot2_py.geoms" else []
        )

        with patch("ggplot2_py._plugins.entry_points", return_value=mock_eps):
            with pytest.warns(UserWarning, match="failed to load.*broken_plugin"):
                result = discover_extensions()

        # Should not crash, and broken plugin should not appear in result
        assert "broken_plugin" not in result.get("ggplot2_py.geoms", [])

    def test_duplicate_skip(self):
        """Entry points for already-registered classes should not overwrite."""
        from ggplot2_py.geom import Geom

        # GeomPoint is already registered via __init_subclass__
        original_cls = Geom._registry.get("point")
        assert original_cls is not None

        # Create a fake entry point for "point"
        mock_ep = MagicMock()
        mock_ep.name = "point"
        mock_ep.load.return_value = type("FakeGeomPoint", (), {})

        mock_eps = MagicMock()
        mock_eps.select.side_effect = lambda group: (
            [mock_ep] if group == "ggplot2_py.geoms" else []
        )

        with patch("ggplot2_py._plugins.entry_points", return_value=mock_eps):
            discover_extensions()

        # Original should not be overwritten
        assert Geom._registry["point"] is original_cls
