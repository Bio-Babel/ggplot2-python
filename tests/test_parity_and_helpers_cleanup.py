"""Regression tests for the LOW + NEW-4..NEW-10 batch.

Covers:
* NEW-4: ``_bin_breaks_bins`` boundary correction (numeric parity is
  enforced by ``validation/validate_NEW4_bin_breaks_bins.py``; this test
  only exercises the corrected branch to guard against regression).
* NEW-5: StatEllipse docstring no longer mentions NotImplementedError
  for ``type='t'``.
* NEW-6: ``CoordQuickmap`` / ``CoordSf`` re-exported from
  ``ggplot2_py.coords``.
* NEW-7: ``guide_old_colourbar`` forwards to ``guide_colourbar`` and
  emits a ``FutureWarning``.
* NEW-10: ``sf_transform_xy`` is exported (public alias).
* L3: ``__dir__`` on ggproto includes members, excludes ``super``.
* L4: ``to_list`` returns dict of public members.
* L7: ``check_aesthetics`` / ``add_group`` / ``has_groups`` ports.
* L8: ``combine_elements`` and ``from_theme`` exported from ``theme``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


class TestNEW4BinBreaksBinsCorrection:
    def test_boundary_alignment_triggers(self):
        """Case documented in R bin.R:110-114: x_range = (0, 100), bins = 10,
        boundary = 0 — ``x_range %% width == boundary %% width`` fires, width
        switches from 100/9 to 100/10 = 10."""
        from ggplot2_py.stat import _bin_breaks_bins
        out = _bin_breaks_bins(
            np.array([0.0, 100.0]), bins=10, boundary=0.0, closed="right",
        )
        # Expect 11 break points spaced by 10.
        assert len(out.breaks) == 11
        np.testing.assert_allclose(np.diff(out.breaks), 10.0)
        np.testing.assert_allclose(out.breaks[0], 0.0)
        np.testing.assert_allclose(out.breaks[-1], 100.0)

    def test_bins_one(self):
        from ggplot2_py.stat import _bin_breaks_bins
        out = _bin_breaks_bins(
            np.array([-5.0, 5.0]), bins=1, closed="right",
        )
        assert len(out.breaks) == 2

    def test_zero_range(self):
        from ggplot2_py.stat import _bin_breaks_bins
        out = _bin_breaks_bins(
            np.array([4.2, 4.2]), bins=20, closed="right",
        )
        # width collapses to 0.1
        assert len(out.breaks) >= 2


class TestNEW5StatEllipseDocstring:
    def test_no_not_implemented_mention(self):
        from ggplot2_py.stat import StatEllipse
        doc = StatEllipse.compute_group.__doc__ or ""
        assert "NotImplementedError" not in doc
        # The docstring should describe the three supported types.
        assert "norm" in doc and "euclid" in doc and '"t"' in doc


class TestNEW6CoordsReexports:
    def test_coord_quickmap(self):
        from ggplot2_py.coords import CoordQuickmap, coord_quickmap
        assert CoordQuickmap is not None
        assert callable(coord_quickmap)

    def test_coord_sf(self):
        from ggplot2_py.coords import CoordSf, coord_sf
        assert CoordSf is not None
        assert callable(coord_sf)


class TestNEW7GuideOldColourbar:
    def test_emits_future_warning(self):
        # Import directly from the submodule to avoid shadowing by the
        # ``guide_colourbar.py`` submodule when other tests have loaded it.
        from ggplot2_py.guide import guide_old_colourbar, guide_colourbar
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            g = guide_old_colourbar(title="hello")
            assert any(issubclass(rec.category, FutureWarning) for rec in w)
        # Verify it forwarded — result is a guide spec like guide_colourbar's.
        ref = guide_colourbar(title="hello")
        assert type(g) is type(ref)

    def test_color_alias_exists(self):
        from ggplot2_py.guide import guide_old_colorbar, guide_old_colourbar
        assert guide_old_colorbar is guide_old_colourbar


class TestNEW10SfTransformXyExport:
    def test_public_alias(self):
        from ggplot2_py.coord import sf_transform_xy, _sf_transform_xy
        assert sf_transform_xy is _sf_transform_xy

    def test_reexported_from_coords(self):
        from ggplot2_py.coords import sf_transform_xy as s
        from ggplot2_py.coord import _sf_transform_xy
        assert s is _sf_transform_xy


class TestL3L4DirToList:
    def _make(self):
        from ggplot2_py.ggproto import ggproto, GGProto
        A = ggproto("A", GGProto, x=1, y=2, greet=lambda self: "hi")
        return A

    def test_dir_instance(self):
        A = self._make()
        a = A()
        d = dir(a)
        assert "x" in d and "y" in d and "greet" in d
        assert "super" not in d

    def test_dir_class(self):
        A = self._make()
        d = dir(A)
        assert "x" in d and "y" in d

    def test_to_list(self):
        A = self._make()
        a = A()
        lst = a.to_list()
        assert lst["x"] == 1 and lst["y"] == 2
        assert callable(lst["greet"])
        # No private keys.
        assert all(not k.startswith("_") for k in lst.keys())


class TestL7AesHelpers:
    def test_rename_aes_dup_warns(self):
        from ggplot2_py.aes import rename_aes, Mapping
        m = Mapping(color="x", colour="y")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = rename_aes(m)
        assert any("Duplicated aesthetics" in str(rec.message) for rec in w)
        # Later entry wins — matches R ``names(x) <- ...``.
        assert out["colour"] == "y"

    def test_check_aesthetics_ok(self):
        from ggplot2_py.geom import check_aesthetics
        # Mix of length-1 and length-n.
        check_aesthetics({"x": [1, 2, 3], "y": 5}, 3)  # no exception

    def test_check_aesthetics_raises(self):
        from ggplot2_py.geom import check_aesthetics
        with pytest.raises(ValueError, match="Aesthetics must be either length 1"):
            check_aesthetics({"x": [1, 2, 3], "y": [1, 2]}, 3)

    def test_add_group_discrete(self):
        from ggplot2_py.layer import add_group, has_groups, NO_GROUP
        df = pd.DataFrame({"x": [1, 2, 3], "cat": ["a", "b", "a"]})
        out = add_group(df)
        assert "group" in out.columns
        assert has_groups(out) is True
        assert NO_GROUP == -1

    def test_add_group_no_discrete(self):
        from ggplot2_py.layer import add_group, has_groups, NO_GROUP
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        out = add_group(df)
        assert (out["group"] == NO_GROUP).all()
        assert has_groups(out) is False

    def test_add_group_preserves_existing(self):
        from ggplot2_py.layer import add_group, has_groups
        df = pd.DataFrame({"x": [1, 2, 3], "group": [1, 1, 2]})
        out = add_group(df)
        assert out["group"].tolist() == [1, 1, 2]
        assert has_groups(out) is True

    def test_add_group_excludes_label(self):
        """R excludes 'label' and 'PANEL' columns from discrete detection."""
        from ggplot2_py.layer import add_group, has_groups, NO_GROUP
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "label": ["a", "b", "a"],
            "PANEL": ["p1", "p1", "p1"],
        })
        out = add_group(df)
        assert (out["group"] == NO_GROUP).all()


class TestL8ThemeExports:
    def test_combine_elements_exported(self):
        from ggplot2_py.theme import combine_elements
        assert callable(combine_elements)

    def test_from_theme_exported(self):
        from ggplot2_py.theme import from_theme
        assert from_theme(42) == 42
