"""Extended coverage tests for ggplot2_py.guide."""

import pytest
import numpy as np
import pandas as pd

from ggplot2_py.guide import (
    Guide,
    GuideNone,
    GuideAxis,
    GuideAxisLogticks,
    GuideAxisStack,
    GuideAxisTheta,
    GuideBins,
    GuideColourbar,
    GuideColoursteps,
    GuideCustom,
    GuideLegend,
    GuideOld,
    Guides,
    guide_none,
    guide_axis,
    guide_axis_logticks,
    guide_axis_stack,
    guide_axis_theta,
    guide_bins,
    guide_colourbar,
    guide_colorbar,
    guide_coloursteps,
    guide_colorsteps,
    guide_custom,
    guide_legend,
    guides,
    new_guide,
    is_guide,
    is_guides,
    _hash_object,
    _defaults,
    _validate_guide,
    _resolve_guide_name,
)
from ggplot2_py._compat import waiver, is_waiver
from ggplot2_py.guide import (
    old_guide,
    guide_train,
    guide_merge,
    guide_geom,
    guide_transform,
    guide_gengrob,
)


def _contributing_colour_layer():
    """A real layer that maps ``colour`` so it contributes to a colour guide.

    ``GuideLegend.process_layers`` ports R's ``include`` computation
    (``guide-legend.R:221-228``): a layer is only included when it
    ``matched_aes`` against the guide's key and its ``show.legend`` is not
    FALSE.  Tests that exercise ``process_layers`` therefore need a layer
    whose mapping matches the guide aesthetic, else (like R) the guide is
    dropped (``process_layers`` returns ``None``).
    """
    from ggplot2_py import geom_point, aes
    return geom_point(mapping=aes(x="x", y="y", colour="g"))


def _colour_key_params(extra=None):
    """Guide params carrying a ``colour`` key (mirrors a trained legend)."""
    params = {
        "key": pd.DataFrame(
            {"colour": ["red", "blue"], ".value": [1, 2], ".label": ["a", "b"]}
        ),
        "override.aes": {},
    }
    if extra:
        params.update(extra)
    return params


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

class TestHashObject:
    def test_deterministic(self):
        h1 = _hash_object("test")
        h2 = _hash_object("test")
        assert h1 == h2

    def test_different_inputs(self):
        assert _hash_object("a") != _hash_object("b")

    def test_list(self):
        h = _hash_object([1, 2, 3])
        assert isinstance(h, str)


class TestDefaults:
    def test_fills_missing(self):
        result = _defaults({"a": 1}, {"a": 0, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_empty_target(self):
        result = _defaults({}, {"x": 10})
        assert result == {"x": 10}


class TestValidateGuide:
    def test_string_legend(self):
        g = _validate_guide("legend")
        assert isinstance(g, GuideLegend)

    def test_string_colourbar(self):
        g = _validate_guide("colourbar")
        assert isinstance(g, GuideColourbar)

    def test_string_colorbar(self):
        g = _validate_guide("colorbar")
        assert isinstance(g, GuideColourbar)

    def test_string_none(self):
        g = _validate_guide("none")
        assert isinstance(g, GuideNone)

    def test_string_axis(self):
        g = _validate_guide("axis")
        assert isinstance(g, GuideAxis)

    def test_string_bins(self):
        g = _validate_guide("bins")
        assert isinstance(g, GuideBins)

    def test_string_coloursteps(self):
        g = _validate_guide("coloursteps")
        assert isinstance(g, GuideColoursteps)

    def test_string_colorsteps(self):
        g = _validate_guide("colorsteps")
        assert isinstance(g, GuideColoursteps)

    def test_string_custom(self):
        g = _validate_guide("custom")
        assert isinstance(g, GuideCustom)

    def test_string_axis_logticks(self):
        g = _validate_guide("axis_logticks")
        assert isinstance(g, GuideAxisLogticks)

    def test_string_axis_theta(self):
        g = _validate_guide("axis_theta")
        assert isinstance(g, GuideAxisTheta)

    def test_string_axis_stack(self):
        g = _validate_guide("axis_stack")
        assert isinstance(g, GuideAxisStack)

    def test_instance(self):
        g = guide_legend()
        assert _validate_guide(g) is g

    def test_class(self):
        g = _validate_guide(GuideLegend)
        assert isinstance(g, GuideLegend)

    def test_invalid(self):
        with pytest.raises(Exception):
            _validate_guide(42)

    def test_unknown_string(self):
        with pytest.raises(Exception):
            _validate_guide("nonexistent")


class TestResolveGuideName:
    # _resolve_guide_name is now R-faithful (Issue #4): it resolves via the
    # extensible registry and *calls* the registered guide_* constructor, so it
    # returns a resolved guide instance (or a bare class only when a class was
    # registered, e.g. "custom"). These tests check the resolved guide type.
    def test_dash_to_underscore(self):
        resolved = _resolve_guide_name("axis-logticks")
        if isinstance(resolved, type):
            resolved = resolved()
        assert isinstance(resolved, GuideAxisLogticks)

    def test_case_insensitive(self):
        resolved = _resolve_guide_name("LEGEND")
        if isinstance(resolved, type):
            resolved = resolved()
        assert isinstance(resolved, GuideLegend)


# ---------------------------------------------------------------------------
# Guide base class
# ---------------------------------------------------------------------------

class TestGuideBase:
    def test_class_name(self):
        g = Guide()
        assert g._class_name == "Guide"

    def test_params_has_title(self):
        g = Guide()
        assert "title" in g.params

    def test_extract_key_no_scale(self):
        result = Guide.extract_key(None, "colour")
        assert result is None

    def test_extract_decor_returns_none(self):
        assert Guide.extract_decor(None, "colour") is None

    def test_extract_params_sets_title(self):
        class MockScale:
            name = "test_scale"
        params = {"title": waiver()}
        result = Guide.extract_params(MockScale(), params)
        assert result["title"] == "test_scale"

    def test_extract_params_explicit_title(self):
        class MockScale:
            name = "test_scale"
        params = {"title": waiver()}
        result = Guide.extract_params(MockScale(), params, title="Custom")
        assert result["title"] == "Custom"

    def test_train_no_scale(self):
        g = Guide()
        result = g.train()
        assert isinstance(result, dict)

    def test_train_with_params(self):
        g = Guide()
        result = g.train(params={"title": "test", "key": None, "hash": ""})
        assert result["title"] == "test"

    def test_transform_with_key(self):
        class MockCoord:
            def transform(self, data, pp):
                return data
        params = {"key": pd.DataFrame({"x": [1, 2]})}
        result = Guide.transform(params, MockCoord(), {})
        assert "key" in result

    def test_transform_empty_key(self):
        params = {"key": pd.DataFrame()}
        result = Guide.transform(params, None, {})
        assert "key" in result

    def test_get_layer_key(self):
        g = Guide()
        params = {"test": True}
        assert g.get_layer_key(params, []) == params

    def test_process_layers(self):
        g = Guide()
        params = {"test": True}
        assert g.process_layers(params, []) == params

    def test_setup_params(self):
        result = Guide.setup_params({"a": 1})
        assert result == {"a": 1}

    def test_override_elements(self):
        result = Guide.override_elements({}, {"bg": "legend.bg"}, None)
        assert result == {"bg": "legend.bg"}

    def test_setup_elements(self):
        g = Guide()
        result = g.setup_elements({})
        assert isinstance(result, dict)

    def test_build_title(self):
        assert Guide.build_title("Title", {}, {}) is None

    def test_build_labels(self):
        assert Guide.build_labels(pd.DataFrame(), {}, {}) is None

    def test_build_decor(self):
        assert Guide.build_decor(None, {}, {}, {}) is None

    def test_build_ticks(self):
        assert Guide.build_ticks(pd.DataFrame(), {}, {}) is None

    def test_measure_grobs(self):
        result = Guide.measure_grobs({}, {}, {})
        assert result == {"width": None, "height": None}

    def test_arrange_layout(self):
        result = Guide.arrange_layout(pd.DataFrame(), {}, {}, {})
        assert result == {}

    def test_assemble_drawing(self):
        assert Guide.assemble_drawing({}, {}, {}, {}, {}) is None

    def test_merge(self):
        g = Guide()
        params = {"key": pd.DataFrame({".value": [1], "colour": ["red"]})}
        new_params = {"key": pd.DataFrame({".value": [1], "fill": ["blue"]})}
        result = g.merge(params, g, new_params)
        assert result["guide"] is g
        assert "fill" in result["params"]["key"].columns

    def test_merge_no_key(self):
        g = Guide()
        params = {"key": pd.DataFrame({"x": [1]})}
        new_params = {"key": None}
        result = g.merge(params, g, new_params)
        assert result["guide"] is g


# ---------------------------------------------------------------------------
# GuideNone
# ---------------------------------------------------------------------------

class TestGuideNoneExtended:
    def test_class_name(self):
        g = guide_none()
        assert g._class_name == "GuideNone"

    def test_train(self):
        g = guide_none()
        result = g.train()
        assert isinstance(result, dict)

    def test_train_with_params(self):
        g = guide_none()
        params = {"test": 42}
        result = g.train(params=params)
        assert result["test"] == 42

    def test_transform(self):
        g = guide_none()
        params = {"key": None}
        result = g.transform(params)
        assert result is params

    def test_draw_returns_none(self):
        g = guide_none()
        assert g.draw() is None

    def test_available_aes(self):
        g = guide_none()
        assert "any" in g.available_aes

    def test_with_title(self):
        g = guide_none(title="test")
        assert isinstance(g, GuideNone)


# ---------------------------------------------------------------------------
# GuideAxis
# ---------------------------------------------------------------------------

class TestGuideAxisExtended:
    def test_class_name(self):
        g = guide_axis()
        assert g._class_name == "GuideAxis"

    def test_available_aes(self):
        g = guide_axis()
        assert "x" in g.available_aes or "y" in g.available_aes

    def test_params_defaults(self):
        g = guide_axis()
        assert g.params.get("cap") == "none"
        assert g.params.get("n.dodge") == 1

    def test_with_check_overlap(self):
        g = guide_axis(check_overlap=True)
        assert g.params.get("check.overlap") is True

    def test_with_minor_ticks(self):
        g = guide_axis(minor_ticks=True)
        assert g.params.get("minor.ticks") is True

    def test_with_n_dodge(self):
        g = guide_axis(n_dodge=3)
        assert g.params.get("n.dodge") == 3

    def test_cap_bool_true(self):
        g = guide_axis(cap=True)
        assert g.params.get("cap") == "both"

    def test_cap_bool_false(self):
        g = guide_axis(cap=False)
        assert g.params.get("cap") == "none"

    def test_cap_string(self):
        g = guide_axis(cap="upper")
        assert g.params.get("cap") == "upper"

    def test_cap_invalid(self):
        with pytest.raises(Exception):
            guide_axis(cap="invalid")

    def test_extract_params_sets_name(self):
        params = {"aesthetic": "x", "name": "axis"}
        result = GuideAxis.extract_params(None, params)
        assert "x" in result["name"]

    def test_extract_decor_default(self):
        result = GuideAxis.extract_decor(None, "x")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_extract_decor_cap_both(self):
        key = pd.DataFrame({"x": [0.2, 0.8]})
        result = GuideAxis.extract_decor(None, "x", key=key, cap="both")
        assert result["x"].iloc[0] == 0.2
        assert result["x"].iloc[1] == 0.8

    def test_extract_decor_cap_upper(self):
        key = pd.DataFrame({"x": [0.2, 0.8]})
        result = GuideAxis.extract_decor(None, "x", key=key, cap="upper")
        assert result["x"].iloc[1] == 0.8
        assert result["x"].iloc[0] == -np.inf

    def test_extract_decor_cap_lower(self):
        key = pd.DataFrame({"x": [0.2, 0.8]})
        result = GuideAxis.extract_decor(None, "x", key=key, cap="lower")
        assert result["x"].iloc[0] == 0.2
        assert result["x"].iloc[1] == np.inf

    def test_extract_key_no_minor(self):
        class MockScale:
            def get_breaks(self):
                return [0, 1, 2]
            def map(self, x):
                return x
            def get_labels(self, x):
                return ["0", "1", "2"]
        result = GuideAxis.extract_key(MockScale(), "x")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_extract_key_with_minor(self):
        class MockScale:
            def get_breaks(self):
                return [0, 1, 2]
            def map(self, x):
                return x
            def get_labels(self, x):
                return ["0", "1", "2"]
            def get_breaks_minor(self):
                return [0.5, 1.5]
        result = GuideAxis.extract_key(MockScale(), "x", minor_ticks=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5  # 3 major + 2 minor

    def test_transform_with_coord(self):
        class MockCoord:
            def transform(self, data, pp):
                return data
        params = {
            "key": pd.DataFrame({"x": [0.2, 0.8]}),
            "aesthetic": "x",
            "position": "bottom",
            "decor": pd.DataFrame({"x": [-np.inf, np.inf]}),
        }
        result = GuideAxis.transform(params, MockCoord(), {})
        assert "key" in result
        assert "y" in result["key"].columns


# ---------------------------------------------------------------------------
# GuideLegend
# ---------------------------------------------------------------------------

class TestGuideLegendExtended:
    def test_class_name(self):
        g = guide_legend()
        assert g._class_name == "GuideLegend"

    def test_with_nrow(self):
        g = guide_legend(nrow=2)
        assert g.params.get("nrow") == 2

    def test_with_ncol(self):
        g = guide_legend(ncol=3)
        assert g.params.get("ncol") == 3

    def test_with_reverse(self):
        g = guide_legend(reverse=True)
        assert g.params.get("reverse") is True

    def test_with_direction(self):
        g = guide_legend(direction="horizontal")
        assert g.params.get("direction") == "horizontal"

    def test_with_override_aes(self):
        g = guide_legend(override_aes={"size": 5})
        assert g.params.get("override.aes") == {"size": 5}

    def test_with_position(self):
        g = guide_legend(position="top")
        assert g.params.get("position") == "top"

    def test_invalid_position(self):
        with pytest.raises(Exception):
            guide_legend(position="center")

    def test_with_order(self):
        g = guide_legend(order=3)
        assert g.params.get("order") == 3

    def test_extract_params_reverse(self):
        key = pd.DataFrame({"colour": ["red", "blue"], ".value": [1, 2], ".label": ["a", "b"]})
        params = {"title": waiver(), "reverse": True, "key": key}
        result = GuideLegend.extract_params(None, params)
        assert result["key"][".label"].iloc[0] == "b"

    def test_extract_params_sets_title_from_scale(self):
        class MockScale:
            name = "my_scale"
        params = {"title": waiver(), "reverse": False, "key": None}
        result = GuideLegend.extract_params(MockScale(), params)
        assert result["title"] == "my_scale"

    def test_elements(self):
        g = guide_legend()
        assert "background" in g.elements
        assert "key" in g.elements


# ---------------------------------------------------------------------------
# GuideColourbar
# ---------------------------------------------------------------------------

class TestGuideColourbarExtended:
    def test_class_name(self):
        g = guide_colourbar()
        assert g._class_name == "GuideColourbar"

    def test_alias(self):
        assert guide_colorbar is guide_colourbar

    def test_default_nbin(self):
        g = guide_colourbar()
        assert g.params.get("nbin") == 300

    def test_gradient_nbin(self):
        g = guide_colourbar(display="gradient")
        assert g.params.get("nbin") == 15

    def test_with_alpha(self):
        g = guide_colourbar(alpha=0.5)
        assert g.params.get("alpha") == 0.5

    def test_with_reverse(self):
        g = guide_colourbar(reverse=True)
        assert g.params.get("reverse") is True

    def test_with_direction(self):
        g = guide_colourbar(direction="horizontal")
        assert g.params.get("direction") == "horizontal"

    def test_invalid_display(self):
        with pytest.raises(Exception):
            guide_colourbar(display="invalid")

    def test_invalid_position(self):
        with pytest.raises(Exception):
            guide_colourbar(position="center")

    def test_available_aes(self):
        g = guide_colourbar()
        assert "colour" in g.available_aes or "fill" in g.available_aes

    def test_with_draw_limits(self):
        g = guide_colourbar(draw_ulim=False, draw_llim=False)
        assert g.params.get("draw_lim") == [False, False]

    def test_with_position(self):
        g = guide_colourbar(position="right")
        assert g.params.get("position") == "right"


# ---------------------------------------------------------------------------
# GuideColoursteps
# ---------------------------------------------------------------------------

class TestGuideColourstepsExtended:
    def test_class_name(self):
        g = guide_coloursteps()
        assert g._class_name == "GuideColoursteps"

    def test_alias(self):
        assert guide_colorsteps is guide_coloursteps

    def test_even_steps(self):
        g = guide_coloursteps(even_steps=True)
        assert g.params.get("even.steps") is True

    def test_show_limits(self):
        g = guide_coloursteps(show_limits=True)
        assert g.params.get("show.limits") is True

    def test_with_alpha(self):
        g = guide_coloursteps(alpha=0.7)
        assert g.params.get("alpha") == 0.7


# ---------------------------------------------------------------------------
# GuideBins
# ---------------------------------------------------------------------------

class TestGuideBinsExtended:
    def test_class_name(self):
        g = guide_bins()
        assert g._class_name == "GuideBins"

    def test_with_override_aes(self):
        g = guide_bins(override_aes={"size": 3})
        assert g.params.get("override.aes") == {"size": 3}

    def test_with_show_limits(self):
        g = guide_bins(show_limits=True)
        assert g.params.get("show.limits") is True

    def test_with_angle(self):
        g = guide_bins(angle=45)
        assert g.params.get("angle") == 45

    def test_invalid_position(self):
        with pytest.raises(Exception):
            guide_bins(position="center")

    def test_with_direction(self):
        g = guide_bins(direction="vertical")
        assert g.params.get("direction") == "vertical"


# ---------------------------------------------------------------------------
# GuideCustom
# ---------------------------------------------------------------------------

class TestGuideCustomExtended:
    def test_class_name(self):
        g = guide_custom(grob="test_grob")
        assert g._class_name == "GuideCustom"

    def test_train_returns_params(self):
        g = guide_custom(grob="test")
        result = g.train()
        assert isinstance(result, dict)

    def test_transform_passthrough(self):
        params = {"key": None}
        result = GuideCustom.transform(params)
        assert result is params

    def test_draw_returns_grob(self):
        g = guide_custom(grob="my_grob")
        result = g.draw(params={"grob": "my_grob"})
        assert result == "my_grob"

    def test_draw_default_params(self):
        g = guide_custom(grob="my_grob")
        result = g.draw()
        assert result == "my_grob"

    def test_with_title(self):
        g = guide_custom(grob="test", title="My Guide")
        assert isinstance(g, GuideCustom)

    def test_with_dimensions(self):
        g = guide_custom(grob="test", width=10, height=20)
        assert g.params.get("width") == 10
        assert g.params.get("height") == 20


# ---------------------------------------------------------------------------
# GuideAxisLogticks
# ---------------------------------------------------------------------------

class TestGuideAxisLogticksExtended:
    def test_class_name(self):
        g = guide_axis_logticks()
        assert g._class_name == "GuideAxisLogticks"

    def test_params(self):
        g = guide_axis_logticks(long=3.0, mid=2.0, short=1.0)
        assert g.params.get("long") == 3.0
        assert g.params.get("mid") == 2.0
        assert g.params.get("short") == 1.0

    def test_cap_bool(self):
        g = guide_axis_logticks(cap=True)
        assert g.params.get("cap") == "both"

    def test_expanded(self):
        g = guide_axis_logticks(expanded=False)
        assert g.params.get("expanded") is False


# ---------------------------------------------------------------------------
# GuideAxisStack
# ---------------------------------------------------------------------------

class TestGuideAxisStackExtended:
    def test_class_name(self):
        g = guide_axis_stack("axis")
        assert g._class_name == "GuideAxisStack"

    def test_with_multiple_axes(self):
        g = guide_axis_stack("axis", "axis")
        assert isinstance(g, GuideAxisStack)
        assert len(g.params.get("guides", [])) == 2


# ---------------------------------------------------------------------------
# GuideAxisTheta
# ---------------------------------------------------------------------------

class TestGuideAxisThetaExtended:
    def test_class_name(self):
        g = guide_axis_theta()
        assert g._class_name == "GuideAxisTheta"

    def test_available_aes(self):
        g = guide_axis_theta()
        assert "theta" in g.available_aes or "x" in g.available_aes

    def test_transform_adds_theta(self):
        class MockCoord:
            def transform(self, data, pp):
                return data
        key = pd.DataFrame({"x": [0.2, 0.8]})
        params = {
            "key": key,
            "aesthetic": "x",
            "position": "bottom",
            "decor": None,
        }
        result = GuideAxisTheta.transform(params, MockCoord(), {})
        assert "theta" in result["key"].columns


# ---------------------------------------------------------------------------
# GuideOld
# ---------------------------------------------------------------------------

class TestGuideOld:
    def test_class_name(self):
        g = GuideOld()
        assert g._class_name == "GuideOld"


# ---------------------------------------------------------------------------
# new_guide
# ---------------------------------------------------------------------------

class TestNewGuide:
    def test_creates_legend(self):
        g = new_guide(super=GuideLegend, available_aes="any", name="legend")
        assert isinstance(g, GuideLegend)

    def test_sets_available_aes_string(self):
        g = new_guide(super=GuideNone, available_aes="colour")
        assert g.available_aes == ["colour"]

    def test_sets_available_aes_list(self):
        g = new_guide(super=GuideNone, available_aes=["colour", "fill"])
        assert g.available_aes == ["colour", "fill"]

    def test_order_is_int(self):
        g = new_guide(super=GuideNone, available_aes="any", order=3)
        assert g.params["order"] == 3

    def test_merges_defaults(self):
        g = new_guide(super=GuideLegend, available_aes="any")
        assert "title" in g.params


# ---------------------------------------------------------------------------
# Guides container
# ---------------------------------------------------------------------------

class TestGuidesContainer:
    def test_creates_guides(self):
        g = guides(colour=guide_legend())
        assert isinstance(g, Guides)

    def test_empty_guides(self):
        g = guides()
        # When no args, guides() may return None or an empty Guides
        assert g is None or isinstance(g, Guides)

    def test_multiple_aesthetics(self):
        g = guides(colour=guide_legend(), fill=guide_colourbar())
        assert isinstance(g, Guides)


# ---------------------------------------------------------------------------
# is_guide / is_guides
# ---------------------------------------------------------------------------

class TestIsGuideExtended:
    def test_guide_axis(self):
        assert is_guide(guide_axis()) is True

    def test_guide_bins(self):
        assert is_guide(guide_bins()) is True

    def test_guide_coloursteps(self):
        assert is_guide(guide_coloursteps()) is True

    def test_guide_custom(self):
        assert is_guide(guide_custom(grob="test")) is True

    def test_guide_axis_logticks(self):
        assert is_guide(guide_axis_logticks()) is True

    def test_guide_axis_theta(self):
        assert is_guide(guide_axis_theta()) is True

    def test_int(self):
        assert is_guide(42) is False

    def test_dict(self):
        assert is_guide({}) is False


class TestIsGuides:
    def test_true_for_guides(self):
        g = guides(colour=guide_legend())
        assert is_guides(g) is True

    def test_false_for_guide(self):
        assert is_guides(guide_legend()) is False

    def test_false_for_none(self):
        assert is_guides(None) is False

    def test_false_for_dict(self):
        assert is_guides({}) is False


# ---------------------------------------------------------------------------
# Guide.draw flow (with key)
# ---------------------------------------------------------------------------

class TestGuideDraw:
    def test_draw_with_no_key(self):
        g = Guide()
        result = g.draw(params={"key": None, "title": "test", "position": "right",
                                 "direction": "vertical"})
        assert result is None

    def test_draw_with_key(self):
        g = Guide()
        key = pd.DataFrame({"colour": ["red"], ".value": [1], ".label": ["a"]})
        result = g.draw(params={
            "key": key, "title": "test", "position": "right",
            "direction": "vertical", "hash": "",
        })
        # assemble_drawing returns None by default
        assert result is None

    def test_draw_position_override(self):
        g = Guide()
        key = pd.DataFrame({"x": [1]})
        result = g.draw(position="left", direction="horizontal",
                        params={"key": key, "title": waiver(), "hash": ""})
        assert result is None


# ---------------------------------------------------------------------------
# Legacy S3 compatibility functions
# ---------------------------------------------------------------------------

class TestLegacyFunctions:
    def test_old_guide(self):
        g = old_guide("mock_guide")
        assert isinstance(g, GuideOld)
        assert g._legacy == "mock_guide"

    def test_guide_train(self):
        g = guide_legend()
        result = guide_train(g, scale=None)
        assert isinstance(result, dict)

    def test_guide_merge(self):
        g1 = guide_legend()
        g2 = guide_legend()
        result = guide_merge(g1, g2)
        assert isinstance(result, dict)

    def test_guide_geom(self):
        # A contributing layer (maps ``colour``) keeps the guide → dict.
        g = guide_legend()
        g.params = _colour_key_params()
        result = guide_geom(g, [_contributing_colour_layer()])
        assert isinstance(result, dict)

    def test_guide_geom_no_contributing_layer_drops(self):
        # R parity (guide-legend.R:226-228): no layer contributes → None.
        g = guide_legend()
        g.params = _colour_key_params()
        result = guide_geom(g, [])
        assert result is None

    def test_guide_transform(self):
        g = guide_legend()
        result = guide_transform(g, None, None)
        assert isinstance(result, dict)

    def test_guide_gengrob(self):
        g = guide_none()
        result = guide_gengrob(g, None)
        assert result is None


# ---------------------------------------------------------------------------
# Guides container methods
# ---------------------------------------------------------------------------

class TestGuidesContainerMethods:
    def test_repr(self):
        g = Guides({"colour": guide_legend()})
        r = repr(g)
        assert "colour" in r

    def test_add_dict(self):
        g = Guides({"colour": guide_legend()})
        g.add({"fill": guide_colourbar()})
        assert "fill" in g.guides

    def test_add_none(self):
        g = Guides()
        g.add(None)  # should not raise

    def test_add_guides_object(self):
        g1 = Guides({"colour": guide_legend()})
        g2 = Guides({"fill": guide_colourbar()})
        g1.add(g2)
        assert "fill" in g1.guides

    def test_get_guide_by_string_dict(self):
        g = Guides({"colour": guide_legend()})
        result = g.get_guide("colour")
        assert isinstance(result, GuideLegend)

    def test_get_guide_by_string_not_found(self):
        g = Guides({"colour": guide_legend()})
        result = g.get_guide("fill")
        assert result is None

    def test_get_guide_by_index(self):
        g = Guides({"colour": guide_legend()})
        result = g.get_guide(0)
        assert isinstance(result, GuideLegend)

    def test_get_guide_by_index_out_of_range(self):
        g = Guides({"colour": guide_legend()})
        result = g.get_guide(10)
        assert result is None

    def test_get_guide_by_string_from_aesthetics(self):
        g = Guides()
        g.guides = [guide_legend()]
        g.aesthetics = ["colour"]
        result = g.get_guide("colour")
        assert isinstance(result, GuideLegend)

    def test_get_params_by_string(self):
        g = Guides()
        g.guides = [guide_legend()]
        g.params = [{"title": "test"}]
        g.aesthetics = ["colour"]
        result = g.get_params("colour")
        assert result["title"] == "test"

    def test_get_params_not_found(self):
        g = Guides()
        g.guides = []
        g.params = []
        g.aesthetics = []
        assert g.get_params("colour") is None

    def test_get_params_by_index(self):
        g = Guides()
        g.params = [{"title": "a"}, {"title": "b"}]
        result = g.get_params(1)
        assert result["title"] == "b"

    def test_get_params_index_out_of_range(self):
        g = Guides()
        g.params = []
        assert g.get_params(5) is None

    def test_subset_guides_dict(self):
        g = Guides({"colour": guide_legend(), "fill": guide_colourbar()})
        g.params = [{}, {}]
        g.aesthetics = ["colour", "fill"]
        g.subset_guides([True, False])
        assert len(g.params) == 1
        assert len(g.aesthetics) == 1

    def test_subset_guides_list(self):
        g = Guides()
        g.guides = [guide_legend(), guide_colourbar()]
        g.params = [{}, {}]
        g.aesthetics = ["colour", "fill"]
        g.subset_guides([False, True])
        assert len(g.guides) == 1
        assert len(g.params) == 1

    def test_setup(self):
        g = Guides({"colour": guide_legend()})

        class MockScale:
            aesthetics = ["colour"]
            guide = waiver()

        result = g.setup([MockScale()], aesthetics=["colour"])
        assert isinstance(result, Guides)
        assert len(result.guides) == 1

    def test_setup_default_aes_from_scales(self):
        g = Guides()

        class MockScale:
            aesthetics = ["colour"]
            guide = waiver()

        result = g.setup([MockScale()])
        assert len(result.aesthetics) == 1

    def test_train_basic(self):
        g = Guides()
        gl = guide_none()
        g.guides = [gl]
        g.params = [dict(gl.params)]
        g.aesthetics = ["colour"]
        g.train([None], {"colour": "Colour"})
        # After training, GuideNone gets removed
        assert len(g.guides) == 0

    def test_merge_single(self):
        g = Guides()
        g.guides = [guide_legend()]
        g.params = [{"order": 0, "hash": "abc"}]
        g.aesthetics = ["colour"]
        g.merge()
        assert len(g.guides) == 1

    def test_merge_multiple_same_hash(self):
        g = Guides()
        g1 = guide_legend()
        g2 = guide_legend()
        g.guides = [g1, g2]
        g.params = [
            {"order": 0, "hash": "same_hash", "key": pd.DataFrame({".value": [1], "colour": ["red"]})},
            {"order": 0, "hash": "same_hash", "key": pd.DataFrame({".value": [1], "fill": ["blue"]})},
        ]
        g.aesthetics = ["colour", "fill"]
        g.merge()
        assert len(g.guides) == 1

    def test_merge_different_hash(self):
        g = Guides()
        g.guides = [guide_legend(), guide_legend()]
        g.params = [
            {"order": 0, "hash": "hash_a"},
            {"order": 0, "hash": "hash_b"},
        ]
        g.aesthetics = ["colour", "fill"]
        g.merge()
        assert len(g.guides) == 2

    def test_process_layers(self):
        # A contributing layer keeps the colour guide (R parity).
        g = Guides()
        gl = guide_legend()
        g.guides = [gl]
        g.params = [_colour_key_params()]
        g.aesthetics = ["colour"]
        g.process_layers([_contributing_colour_layer()], None, None)
        assert len(g.guides) == 1

    def test_process_layers_drops_when_no_layer_contributes(self):
        # R parity (guide-legend.R:226-228 / guides-.R:449-455):
        # ``!any(include)`` → guide returns NULL and is subset out.
        g = Guides()
        gl = guide_legend()
        g.guides = [gl]
        g.params = [_colour_key_params()]
        g.aesthetics = ["colour"]
        g.process_layers([], None, None)
        assert len(g.guides) == 0

    def test_draw(self):
        g = Guides()
        gn = guide_none()
        g.guides = [gn]
        g.params = [dict(gn.params)]
        g.aesthetics = ["colour"]
        result = g.draw(None, ["right"])
        assert len(result) == 1
        assert result[0] is None

    def test_draw_top_position(self):
        g = Guides()
        gn = guide_none()
        g.guides = [gn]
        g.params = [dict(gn.params)]
        g.aesthetics = ["colour"]
        result = g.draw(None, ["top"])
        assert len(result) == 1

    def test_assemble_empty(self):
        g = Guides()
        g.guides = {}
        result = g.assemble(None)
        assert result is None

    def test_assemble_with_guide(self):
        g = Guides()
        gn = guide_none()
        g.guides = [gn]
        g.params = [dict(gn.params)]
        g.aesthetics = ["colour"]
        result = g.assemble(None)
        assert isinstance(result, dict)

    def test_build_empty_scales(self):
        g = Guides()
        result = g.build([], [], {})
        assert isinstance(result, Guides)

    def test_guides_with_false_value(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            g = guides(colour=False)
            assert isinstance(g, Guides)


# ---------------------------------------------------------------------------
# Guide.train with scale (extract_key, hash computation)
# ---------------------------------------------------------------------------

class TestGuideTrainWithScale:
    def test_train_extract_key(self):
        """Test that Guide.extract_key works with a mock scale."""
        class MockScale:
            name = "test_scale"
            def get_breaks(self):
                return [1, 2, 3]
            def map(self, x):
                return x
            def get_labels(self, x):
                return ["a", "b", "c"]

        key = Guide.extract_key(MockScale(), "colour")
        assert isinstance(key, pd.DataFrame)
        assert len(key) == 3
        assert "colour" in key.columns
        assert ".value" in key.columns
        assert ".label" in key.columns

    def test_train_extract_key_no_breaks(self):
        """Test extract_key returns None when no breaks."""
        class MockScale:
            def get_breaks(self):
                return None
        key = Guide.extract_key(MockScale(), "colour")
        assert key is None

    def test_train_extract_key_empty_breaks(self):
        """Test extract_key with empty breaks."""
        class MockScale:
            def get_breaks(self):
                return []
            def map(self, x):
                return x
            def get_labels(self, x):
                return []
        key = Guide.extract_key(MockScale(), "colour")
        assert isinstance(key, pd.DataFrame)
        assert len(key) == 0

    def test_train_no_scale(self):
        g = GuideLegend()
        result = g.train(params=dict(g.params), scale=None, aesthetic="colour")
        assert result is not None
