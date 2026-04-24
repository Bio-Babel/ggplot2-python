"""Extended coverage tests for ggplot2_py.facet."""

import pytest
import numpy as np
import pandas as pd

from ggplot2_py.facet import (
    Facet,
    FacetNull,
    FacetGrid,
    FacetWrap,
    facet_null,
    facet_grid,
    facet_wrap,
    is_facet,
    _layout_null,
    wrap_dims,
    _resolve_facet_vars,
    _combine_vars,
    _map_facet_data,
    _wrap_layout,
)
from ggplot2_py._compat import waiver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestLayoutNull:
    def test_returns_dataframe(self):
        result = _layout_null()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert set(result.columns) >= {"PANEL", "ROW", "COL", "SCALE_X", "SCALE_Y"}

    def test_values(self):
        result = _layout_null()
        assert result["ROW"].iloc[0] == 1
        assert result["COL"].iloc[0] == 1


class TestWrapDims:
    def test_auto(self):
        nrow, ncol = wrap_dims(4)
        assert nrow * ncol >= 4

    def test_nrow_given(self):
        nrow, ncol = wrap_dims(6, nrow=2)
        assert nrow == 2
        assert ncol == 3

    def test_ncol_given(self):
        nrow, ncol = wrap_dims(6, ncol=3)
        assert nrow == 2
        assert ncol == 3

    def test_both_given(self):
        nrow, ncol = wrap_dims(6, nrow=2, ncol=3)
        assert nrow == 2
        assert ncol == 3

    def test_too_small(self):
        with pytest.raises(Exception):
            wrap_dims(10, nrow=2, ncol=2)

    def test_single_panel(self):
        nrow, ncol = wrap_dims(1)
        assert nrow * ncol >= 1

    def test_prime_number(self):
        nrow, ncol = wrap_dims(7)
        assert nrow * ncol >= 7


class TestResolveFacetVars:
    def test_none(self):
        assert _resolve_facet_vars(None) == []

    def test_single_string(self):
        assert _resolve_facet_vars("class") == ["class"]

    def test_formula_style(self):
        result = _resolve_facet_vars("drv ~ cyl")
        assert "drv" in result
        assert "cyl" in result

    def test_plus_style(self):
        result = _resolve_facet_vars("drv + cyl")
        assert "drv" in result
        assert "cyl" in result

    def test_list(self):
        result = _resolve_facet_vars(["drv", "cyl"])
        assert result == ["drv", "cyl"]

    def test_tuple(self):
        result = _resolve_facet_vars(("drv", "cyl"))
        assert result == ["drv", "cyl"]

    def test_dict(self):
        result = _resolve_facet_vars({"drv": True, "cyl": True})
        assert result == ["drv", "cyl"]

    def test_dot_ignored(self):
        result = _resolve_facet_vars(". ~ class")
        assert result == ["class"]


class TestCombineVars:
    def test_empty_vars(self):
        result = _combine_vars([pd.DataFrame()], [])
        assert len(result) == 0

    def test_single_df(self):
        df = pd.DataFrame({"class": ["suv", "compact", "suv"]})
        result = _combine_vars([df], ["class"])
        assert len(result) == 2

    def test_multiple_dfs(self):
        df1 = pd.DataFrame({"class": ["suv", "compact"]})
        df2 = pd.DataFrame({"class": ["midsize", "compact"]})
        result = _combine_vars([df1, df2], ["class"])
        assert len(result) == 3

    def test_missing_column(self):
        df = pd.DataFrame({"x": [1, 2]})
        result = _combine_vars([df], ["nonexistent"])
        assert "nonexistent" in result.columns

    def test_none_in_data_list(self):
        df = pd.DataFrame({"class": ["a", "b"]})
        result = _combine_vars([None, df, None], ["class"])
        assert len(result) == 2

    def test_empty_df(self):
        result = _combine_vars([pd.DataFrame()], ["class"])
        assert "class" in result.columns


class TestMapFacetData:
    def test_empty_data(self):
        result = _map_facet_data(pd.DataFrame(), _layout_null(), {}, [])
        assert "PANEL" in result.columns

    def test_waiver_data(self):
        result = _map_facet_data(waiver(), _layout_null(), {}, [])
        assert "PANEL" in result.columns

    def test_no_facet_vars(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = _map_facet_data(df, _layout_null(), {}, [])
        assert len(result) == 3
        assert (result["PANEL"] == 1).all()

    def test_with_facet_vars(self):
        df = pd.DataFrame({"class": ["a", "a", "b"], "x": [1, 2, 3]})
        layout = pd.DataFrame({
            "PANEL": pd.Categorical([1, 2]),
            "class": ["a", "b"],
            "ROW": [1, 1],
            "COL": [1, 2],
        })
        result = _map_facet_data(df, layout, {}, ["class"])
        assert "PANEL" in result.columns
        assert len(result) == 3

    def test_no_matching_vars(self):
        df = pd.DataFrame({"x": [1, 2]})
        layout = pd.DataFrame({
            "PANEL": pd.Categorical([1]),
            "class": ["a"],
        })
        result = _map_facet_data(df, layout, {}, ["class"])
        assert "PANEL" in result.columns


class TestWrapLayout:
    def test_lt(self):
        ids = np.arange(1, 5)
        row, col = _wrap_layout(ids, (2, 2), "lt")
        assert list(row) == [1, 1, 2, 2]
        assert list(col) == [1, 2, 1, 2]

    def test_tl(self):
        ids = np.arange(1, 5)
        row, col = _wrap_layout(ids, (2, 2), "tl")
        assert list(row) == [1, 2, 1, 2]
        assert list(col) == [1, 1, 2, 2]

    def test_lb(self):
        ids = np.arange(1, 5)
        row, col = _wrap_layout(ids, (2, 2), "lb")
        assert all(r in [1, 2] for r in row)

    def test_bl(self):
        ids = np.arange(1, 5)
        row, col = _wrap_layout(ids, (2, 2), "bl")
        assert all(r in [1, 2] for r in row)

    def test_rt(self):
        ids = np.arange(1, 5)
        row, col = _wrap_layout(ids, (2, 2), "rt")
        assert all(c in [1, 2] for c in col)

    def test_rb(self):
        ids = np.arange(1, 5)
        row, col = _wrap_layout(ids, (2, 2), "rb")
        assert all(c in [1, 2] for c in col)

    def test_tr(self):
        ids = np.arange(1, 5)
        row, col = _wrap_layout(ids, (2, 2), "tr")
        assert all(r in [1, 2] for r in row)

    def test_br(self):
        ids = np.arange(1, 5)
        row, col = _wrap_layout(ids, (2, 2), "br")
        assert all(r in [1, 2] for r in row)

    def test_unknown_defaults(self):
        ids = np.arange(1, 5)
        row, col = _wrap_layout(ids, (2, 2), "xx")
        assert len(row) == 4


# ---------------------------------------------------------------------------
# Base Facet
# ---------------------------------------------------------------------------


class _ScaleStub:
    # Stand-in for a position scale: R's Facet$init_scales (facet-.R:225-234)
    # calls x_scale$clone() once per SCALE_X level, so the prototype must be
    # clone-able. Each clone() returns a fresh instance so tests can verify
    # per-panel independence (the invariant behind commit 06cfc03).
    def clone(self):
        return _ScaleStub()


class TestFacetBase:
    def test_setup_params(self):
        f = Facet()
        df1 = pd.DataFrame({"x": [1], "y": [2]})
        params = f.setup_params([df1], {})
        assert "_possible_columns" in params

    def test_setup_data(self):
        f = Facet()
        data = [pd.DataFrame({"x": [1]})]
        result = f.setup_data(data, {})
        assert result is data

    def test_compute_layout_raises(self):
        f = Facet()
        with pytest.raises(Exception):
            f.compute_layout([], {})

    def test_map_data_raises(self):
        f = Facet()
        with pytest.raises(Exception):
            f.map_data(pd.DataFrame(), pd.DataFrame(), {})

    def test_init_scales_x(self):
        f = Facet()
        layout = pd.DataFrame({"SCALE_X": [1], "SCALE_Y": [1]})
        proto = _ScaleStub()
        result = f.init_scales(layout, x_scale=proto)
        assert "x" in result
        assert len(result["x"]) == 1
        assert result["x"][0] is not proto

    def test_init_scales_y(self):
        f = Facet()
        layout = pd.DataFrame({"SCALE_X": [1], "SCALE_Y": [1]})
        proto = _ScaleStub()
        result = f.init_scales(layout, y_scale=proto)
        assert "y" in result
        assert len(result["y"]) == 1
        assert result["y"][0] is not proto

    def test_init_scales_both(self):
        f = Facet()
        layout = pd.DataFrame({"SCALE_X": [1, 2], "SCALE_Y": [1, 2]})
        proto_x = _ScaleStub()
        proto_y = _ScaleStub()
        result = f.init_scales(layout, x_scale=proto_x, y_scale=proto_y)
        assert len(result["x"]) == 2
        assert len(result["y"]) == 2
        # R parity: each panel gets an independent clone, never an alias.
        assert result["x"][0] is not result["x"][1]
        assert result["y"][0] is not result["y"][1]

    def test_finish_data(self):
        f = Facet()
        df = pd.DataFrame({"x": [1, 2]})
        result = f.finish_data(df, pd.DataFrame(), [], [])
        pd.testing.assert_frame_equal(result, df)

    def test_vars_empty(self):
        f = Facet()
        assert f.vars() == []

    def test_draw_labels(self):
        # Facet.draw_labels ports R's Facet$draw_labels (facet-.R:824-850):
        # unconditionally emits xlab-t, xlab-b, ylab-l, ylab-r on the
        # panels gtable so downstream consumers (patchwork's
        # collect_axis_titles) can find axis titles by directional name.
        from grid_py import Unit, rect_grob, null_grob
        from gtable_py import Gtable, gtable_add_grob
        from ggplot2_py.theme import theme_get

        panels = Gtable(
            widths=Unit([1.0], ["null"]),
            heights=Unit([1.0], ["null"]),
        )
        panels = gtable_add_grob(panels, rect_grob(), t=1, l=1, name="panel-1-1")

        f = Facet()
        labels = {"x": [null_grob(), null_grob()],
                  "y": [null_grob(), null_grob()]}
        result = f.draw_labels(
            panels, pd.DataFrame(), [], [], [], None, None, theme_get(),
            labels, {},
        )
        names = list(result.layout["name"])
        assert "xlab-t" in names
        assert "xlab-b" in names
        assert "ylab-l" in names
        assert "ylab-r" in names


# ---------------------------------------------------------------------------
# FacetNull
# ---------------------------------------------------------------------------

class TestFacetNullExtended:
    def test_shrink(self):
        f = facet_null()
        assert f.shrink is True

    def test_shrink_false(self):
        f = facet_null(shrink=False)
        assert f.shrink is False

    def test_compute_layout(self):
        f = facet_null()
        layout = f.compute_layout([], {})
        assert len(layout) == 1
        assert layout["PANEL"].iloc[0] == 1

    def test_map_data(self):
        f = facet_null()
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = f.map_data(df, _layout_null(), {})
        assert "PANEL" in result.columns
        assert len(result) == 3
        assert (result["PANEL"] == 1).all()

    def test_map_data_empty(self):
        f = facet_null()
        df = pd.DataFrame(columns=["x", "y"])
        result = f.map_data(df, _layout_null(), {})
        assert "PANEL" in result.columns
        assert len(result) == 0

    def test_map_data_waiver(self):
        f = facet_null()
        result = f.map_data(waiver(), _layout_null(), {})
        assert "PANEL" in result.columns
        assert len(result) == 0


# ---------------------------------------------------------------------------
# FacetGrid
# ---------------------------------------------------------------------------

class TestFacetGridExtended:
    def test_constructor(self):
        f = facet_grid(rows="drv", cols="cyl")
        assert isinstance(f, FacetGrid)
        assert f.params["rows"] == "drv"
        assert f.params["cols"] == "cyl"

    def test_shrink(self):
        f = facet_grid(rows="drv", shrink=False)
        assert f.shrink is False

    def test_compute_layout_rows_only(self):
        f = facet_grid(rows="class")
        df = pd.DataFrame({"class": ["a", "b", "c"]})
        layout = f.compute_layout([df], f.params)
        assert len(layout) == 3
        assert "ROW" in layout.columns
        assert layout["COL"].nunique() == 1

    def test_compute_layout_cols_only(self):
        f = facet_grid(cols="class")
        df = pd.DataFrame({"class": ["a", "b"]})
        layout = f.compute_layout([df], f.params)
        assert len(layout) == 2
        assert layout["ROW"].nunique() == 1

    def test_compute_layout_both(self):
        f = facet_grid(rows="drv", cols="cyl")
        df = pd.DataFrame({
            "drv": ["f", "f", "r", "r"],
            "cyl": [4, 6, 4, 6],
        })
        layout = f.compute_layout([df], f.params)
        assert len(layout) == 4
        assert layout["ROW"].nunique() == 2
        assert layout["COL"].nunique() == 2

    def test_compute_layout_neither(self):
        f = facet_grid()
        layout = f.compute_layout([pd.DataFrame()], f.params)
        assert len(layout) == 1

    def test_free_scales(self):
        f = facet_grid(rows="class", scales="free")
        assert f.params["free"]["x"] is True
        assert f.params["free"]["y"] is True

    def test_free_x_only(self):
        f = facet_grid(rows="class", scales="free_x")
        assert f.params["free"]["x"] is True
        assert f.params["free"]["y"] is False

    def test_free_y_only(self):
        f = facet_grid(rows="class", scales="free_y")
        assert f.params["free"]["x"] is False
        assert f.params["free"]["y"] is True

    def test_space_free(self):
        f = facet_grid(rows="class", space="free")
        assert f.params["space_free"]["x"] is True
        assert f.params["space_free"]["y"] is True

    def test_map_data(self):
        f = facet_grid(rows="class")
        df = pd.DataFrame({"class": ["a", "a", "b"], "x": [1, 2, 3]})
        layout = f.compute_layout([df], f.params)
        result = f.map_data(df, layout, f.params)
        assert "PANEL" in result.columns

    def test_vars(self):
        f = facet_grid(rows="drv", cols="cyl")
        v = f.vars()
        assert "drv" in v
        assert "cyl" in v

    def test_switch(self):
        f = facet_grid(rows="class", switch="x")
        assert f.params.get("switch") == "x"

    def test_margins(self):
        f = facet_grid(rows="class", margins=True)
        assert f.params.get("margins") is True

    def test_draw_axes(self):
        f = facet_grid(rows="class", axes="all")
        assert f.params["draw_axes"]["x"] is True
        assert f.params["draw_axes"]["y"] is True

    def test_axis_labels(self):
        f = facet_grid(rows="class", axis_labels="all")
        assert f.params["axis_labels"]["x"] is True
        assert f.params["axis_labels"]["y"] is True

    def test_compute_layout_free_scale_ids(self):
        f = facet_grid(rows="drv", cols="cyl", scales="free")
        df = pd.DataFrame({
            "drv": ["f", "r"],
            "cyl": [4, 6],
        })
        layout = f.compute_layout([df], f.params)
        # With free scales, SCALE_X and SCALE_Y should vary
        assert layout["SCALE_X"].nunique() >= 1
        assert layout["SCALE_Y"].nunique() >= 1


# ---------------------------------------------------------------------------
# FacetWrap
# ---------------------------------------------------------------------------

class TestFacetWrapExtended:
    def test_constructor(self):
        f = facet_wrap("class")
        assert isinstance(f, FacetWrap)
        assert f.params["facets"] == "class"

    def test_shrink(self):
        f = facet_wrap("class", shrink=False)
        assert f.shrink is False

    def test_compute_layout(self):
        f = facet_wrap("class")
        df = pd.DataFrame({"class": ["a", "b", "c"]})
        layout = f.compute_layout([df], f.params)
        assert len(layout) == 3
        assert "ROW" in layout.columns
        assert "COL" in layout.columns

    def test_compute_layout_with_nrow(self):
        f = facet_wrap("class", nrow=2)
        df = pd.DataFrame({"class": ["a", "b", "c", "d"]})
        layout = f.compute_layout([df], f.params)
        assert layout["ROW"].max() <= 2

    def test_compute_layout_with_ncol(self):
        f = facet_wrap("class", ncol=2)
        df = pd.DataFrame({"class": ["a", "b", "c", "d"]})
        layout = f.compute_layout([df], f.params)
        assert layout["COL"].max() <= 2

    def test_compute_layout_no_facet_vars(self):
        f = FacetWrap()
        f.params = {"facets": None, "nrow": None, "ncol": None,
                     "free": {"x": False, "y": False}, "drop": True, "dir": "lt"}
        layout = f.compute_layout([pd.DataFrame()], f.params)
        assert len(layout) == 1

    def test_compute_layout_empty_base(self):
        f = facet_wrap("class")
        layout = f.compute_layout([pd.DataFrame()], f.params)
        assert len(layout) == 1

    def test_map_data(self):
        f = facet_wrap("class")
        df = pd.DataFrame({"class": ["a", "a", "b"], "x": [1, 2, 3]})
        layout = f.compute_layout([df], f.params)
        result = f.map_data(df, layout, f.params)
        assert "PANEL" in result.columns

    def test_vars(self):
        f = facet_wrap("class")
        assert f.vars() == ["class"]

    def test_free_scales(self):
        f = facet_wrap("class", scales="free")
        assert f.params["free"]["x"] is True
        assert f.params["free"]["y"] is True

    def test_dir_horizontal(self):
        f = facet_wrap("class", dir="h")
        assert f.params["dir"] in ("lt", "lb")

    def test_dir_vertical(self):
        f = facet_wrap("class", dir="v")
        assert f.params["dir"] in ("tl", "tr")

    def test_dir_two_letter(self):
        f = facet_wrap("class", dir="tl")
        assert f.params["dir"] == "tl"

    def test_strip_position(self):
        f = facet_wrap("class", strip_position="bottom")
        assert f.params.get("strip_position") == "bottom"

    def test_invalid_strip_position(self):
        with pytest.raises(Exception):
            facet_wrap("class", strip_position="center")

    def test_as_table_false(self):
        f = facet_wrap("class", dir="h", as_table=False)
        assert f.params["dir"] == "lb"

    def test_as_table_false_vertical(self):
        f = facet_wrap("class", dir="v", as_table=False)
        assert f.params["dir"] == "tr"

    def test_space_free(self):
        f = facet_wrap("class", space="free_x")
        assert f.params["space_free"]["x"] is True
        assert f.params["space_free"]["y"] is False

    def test_axes(self):
        f = facet_wrap("class", axes="all")
        assert f.params["draw_axes"]["x"] is True
        assert f.params["draw_axes"]["y"] is True

    def test_free_scale_ids(self):
        f = facet_wrap("class", scales="free")
        df = pd.DataFrame({"class": ["a", "b", "c"]})
        layout = f.compute_layout([df], f.params)
        # With free, each panel gets its own scale id
        assert layout["SCALE_X"].nunique() == 3
        assert layout["SCALE_Y"].nunique() == 3

    def test_list_facets(self):
        f = facet_wrap(["drv", "cyl"])
        df = pd.DataFrame({
            "drv": ["f", "f", "r"],
            "cyl": [4, 6, 4],
        })
        layout = f.compute_layout([df], f.params)
        assert len(layout) >= 2


# ---------------------------------------------------------------------------
# Facet.train_scales
# ---------------------------------------------------------------------------

class TestFacetTrainScales:
    def test_basic(self):
        f = Facet()
        layout = pd.DataFrame({
            "PANEL": pd.Categorical([1]),
            "SCALE_X": [1],
            "SCALE_Y": [1],
        })
        layer_data = pd.DataFrame({
            "x": [1, 2],
            "y": [3, 4],
            "PANEL": pd.Categorical([1, 1]),
        })

        class MockScale:
            def train_df(self, data):
                self.trained = True

        x_scales = [MockScale()]
        y_scales = [MockScale()]
        f.train_scales(x_scales, y_scales, layout, [layer_data])
        assert hasattr(x_scales[0], "trained")
        assert hasattr(y_scales[0], "trained")

    def test_empty_layer_data(self):
        f = Facet()
        layout = pd.DataFrame({
            "PANEL": pd.Categorical([1]),
            "SCALE_X": [1],
            "SCALE_Y": [1],
        })
        f.train_scales([], [], layout, [None])  # should not raise

    def test_no_panel_column(self):
        f = Facet()
        layout = pd.DataFrame({
            "PANEL": pd.Categorical([1]),
            "SCALE_X": [1],
            "SCALE_Y": [1],
        })
        layer_data = pd.DataFrame({"x": [1, 2]})
        f.train_scales([], [], layout, [layer_data])  # should not raise


# ---------------------------------------------------------------------------
# is_facet
# ---------------------------------------------------------------------------

class TestIsFacetExtended:
    def test_facet_grid(self):
        assert is_facet(facet_grid(rows="class")) is True

    def test_facet_base(self):
        assert is_facet(Facet()) is True

    def test_int(self):
        assert is_facet(42) is False

    def test_list(self):
        assert is_facet([]) is False
