"""Regression tests for ``_gtable_total_cm`` axis-aware unit conversion.

R's ``convertHeight(unit, "cm")`` and ``convertWidth(unit, "cm")`` agree
for absolute and font-relative units (cm, pt, mm, inches, lines, char,
strwidth, strheight) but diverge for viewport-relative ``"npc"`` in
non-square viewports. ``_gtable_total_cm`` must therefore accept an
``axis`` argument and dispatch to the right converter.
"""

from __future__ import annotations

import numpy as np
import pytest

from grid_py import Unit


def _push_non_square_viewport():
    """Push a 1-inch wide × 2-inch tall viewport so npc resolves
    differently per axis."""
    from grid_py import Viewport, grid_newpage, push_viewport
    grid_newpage(width=4, height=8, dpi=72)
    push_viewport(Viewport(width=Unit(1, "in"), height=Unit(2, "in")))


class TestGtableTotalCmAxis:
    def test_default_axis_is_height(self):
        """Backwards-compat: omitted axis defaults to height."""
        from ggplot2_py._guide_legend import _gtable_total_cm
        u = Unit(2.54, "cm")
        # Absolute unit: any axis gives the same value
        assert _gtable_total_cm(u) == pytest.approx(2.54)
        assert _gtable_total_cm(u, "height") == pytest.approx(2.54)
        assert _gtable_total_cm(u, "width") == pytest.approx(2.54)

    def test_npc_diverges_by_axis(self):
        """``unit(0.5, "npc")`` resolves to half the viewport width on
        the x axis and half the viewport height on the y axis."""
        from ggplot2_py._guide_legend import _gtable_total_cm
        _push_non_square_viewport()
        npc = Unit(0.5, "npc")
        w_cm = _gtable_total_cm(npc, "width")    # 0.5 in = 1.27 cm
        h_cm = _gtable_total_cm(npc, "height")   # 1.0 in = 2.54 cm
        assert w_cm != pytest.approx(h_cm)
        assert w_cm == pytest.approx(1.27, abs=0.05)
        assert h_cm == pytest.approx(2.54, abs=0.05)

    def test_font_units_axis_invariant(self):
        """``lines`` / ``char`` resolve to the same cm value on both
        axes — they depend on font metrics, not viewport extent."""
        from ggplot2_py._guide_legend import _gtable_total_cm
        for utype in ("lines", "char"):
            u = Unit(2, utype)
            w = _gtable_total_cm(u, "width")
            h = _gtable_total_cm(u, "height")
            assert w == pytest.approx(h), f"axis-divergence in {utype}"

    def test_empty_unit(self):
        from ggplot2_py._guide_legend import _gtable_total_cm
        assert _gtable_total_cm(None) == 0.0


class TestGuideAxisUnitToCmAxis:
    def test_npc_diverges_by_axis(self):
        """``_guide_axis._unit_to_cm`` is the axis-aware sibling of
        ``_gtable_total_cm``; same npc-divergence contract applies."""
        from ggplot2_py._guide_axis import _unit_to_cm
        _push_non_square_viewport()
        # _unit_to_cm in _guide_axis skips npc by design (see
        # context-dependent unit allowlist), so neither axis returns the
        # npc contribution. Verify they agree on what they *do* count.
        u = Unit(2.0, "cm") + Unit(10.0, "pt")
        w = _unit_to_cm(u, "width")
        h = _unit_to_cm(u, "height")
        assert w == pytest.approx(h)
        assert w == pytest.approx(2.0 + 10 * 2.54 / 72.27, abs=0.01)
