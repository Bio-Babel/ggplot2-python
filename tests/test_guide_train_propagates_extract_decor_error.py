"""Regression test: ``Guide.train`` must propagate ``extract_decor``
errors (R does not silence them).

Earlier ``Guide.train`` wrapped the ``extract_decor`` call in
``try/except Exception: pass`` and silently set ``params["decor"] =
None`` on any failure. This violates R parity (R uses ``inject(...)``
which lets errors propagate) and global CLAUDE.md rule 2 (avoid silent
``try/except``).
"""

from __future__ import annotations

import pytest

from ggplot2_py.guide import Guide, GuideLegend


class _BrokenGuide(GuideLegend):
    """GuideLegend subclass whose ``extract_decor`` intentionally raises."""

    def extract_decor(self, scale, aesthetic=None, **kwargs):
        raise RuntimeError("intentional extract_decor failure")


class TestExtractDecorErrorPropagates:
    def test_error_propagates_not_silenced(self):
        """Guide.train must not swallow the RuntimeError."""
        import ggplot2_py as gg
        # Build a minimal continuous colour scale to feed train()
        sc = gg.scale_color_viridis_c(name="x")
        sc.train([0.0, 1.0])  # ensure scale has limits/breaks
        guide = _BrokenGuide()
        with pytest.raises(RuntimeError, match="intentional extract_decor failure"):
            guide.train(scale=sc, aesthetic="colour")
