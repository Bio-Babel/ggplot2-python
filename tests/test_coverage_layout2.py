"""Additional tests for ggplot2_py.layout."""

import pytest
import numpy as np
import pandas as pd

from ggplot2_py.layout import Layout, create_layout
from ggplot2_py.coord import CoordCartesian
from ggplot2_py.facet import FacetNull


from ggplot2_py._compat import waiver as _waiver


class _FS:
    aesthetics = ["x"]
    # Real scales inherit these from ``Scale``; the stubs mirror that so
    # ``default_expansion`` can resolve an expansion spec.
    expand = _waiver()

    def get_limits(self):
        return [0, 10]
    def dimension(self, expand=None, limits=None):
        return [0, 10]
    def train_df(self, df):
        pass
    def map_df(self, df, i=None):
        return {}
    def transform_df(self, df):
        return {}
    def break_info(self, range=None):
        return {"range": [0, 10], "labels": [], "major": None, "minor": None}
    def clone(self):
        return _FS()
    def reset(self):
        pass
    def is_discrete(self):
        return False


class TestLayout:
    def test_create(self):
        assert create_layout(FacetNull(), CoordCartesian()) is not None

    def test_setup(self):
        layout = create_layout(FacetNull(), CoordCartesian())
        data = [pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})]
        result = layout.setup(data, pd.DataFrame({"x": [1.0], "y": [2.0]}), {})
        assert isinstance(result, list)

    def test_train_position(self):
        layout = create_layout(FacetNull(), CoordCartesian())
        data = [pd.DataFrame({"x": [1.0], "y": [2.0], "PANEL": [1]})]
        layout.setup(data, pd.DataFrame({"x": [1.0], "y": [2.0]}), {})
        layout.train_position(data, _FS(), _FS())

    def test_map_position_empty(self):
        layout = create_layout(FacetNull(), CoordCartesian())
        layout.setup([pd.DataFrame()], pd.DataFrame(), {})
        assert isinstance(layout.map_position([pd.DataFrame()]), list)

    def test_setup_panel_params(self):
        layout = create_layout(FacetNull(), CoordCartesian())
        data = [pd.DataFrame({"x": [1.0], "y": [2.0], "PANEL": [1]})]
        layout.setup(data, pd.DataFrame({"x": [1.0], "y": [2.0]}), {})
        layout.panel_scales_x = [_FS()]
        layout.panel_scales_y = [_FS()]
        layout.setup_panel_params()
        assert layout.panel_params is not None

    def test_reset_scales(self):
        layout = create_layout(FacetNull(), CoordCartesian())
        layout.panel_scales_x = [_FS()]
        layout.panel_scales_y = [_FS()]
        layout.reset_scales()

    def test_get_scales(self):
        layout = create_layout(FacetNull(), CoordCartesian())
        data = [pd.DataFrame({"x": [1.0], "y": [2.0], "PANEL": [1]})]
        layout.setup(data, pd.DataFrame({"x": [1.0], "y": [2.0]}), {})
        layout.panel_scales_x = [_FS()]
        layout.panel_scales_y = [_FS()]
        assert isinstance(layout.get_scales(1), dict)
