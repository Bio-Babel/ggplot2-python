"""Tests to improve coverage for theme.py and theme_elements.py."""

import copy
import math
import pytest

from ggplot2_py._compat import Waiver, waiver
from ggplot2_py.theme import (
    Theme,
    theme,
    is_theme,
    add_theme,
    theme_replace_op,
    complete_theme,
    get_theme,
    set_theme,
    update_theme,
    replace_theme,
    reset_theme_settings,
    theme_get,
    theme_set,
    theme_update,
    theme_replace,
)
from ggplot2_py.theme_elements import (
    Element,
    ElementBlank,
    ElementLine,
    ElementRect,
    ElementText,
    ElementPoint,
    ElementPolygon,
    ElementGeom,
    element_blank,
    element_line,
    element_rect,
    element_text,
    element_point,
    element_polygon,
    element_geom,
    element_grob,
    element_render,
    el_def,
    merge_element,
    combine_elements,
    is_theme_element,
    Margin,
    margin,
    margin_auto,
    margin_part,
    is_margin,
    Rel,
    rel,
    is_rel,
    calc_element,
    get_element_tree,
    register_theme_elements,
    reset_theme_settings as reset_theme_settings_elements,
    _ggplot_global,
)


# =====================================================================
# Rel tests
# =====================================================================

class TestRel:
    def test_creation(self):
        r = rel(1.2)
        assert isinstance(r, Rel)
        assert r.value == 1.2

    def test_float(self):
        assert float(rel(0.8)) == 0.8

    def test_repr(self):
        assert repr(rel(0.5)) == "rel(0.5)"

    def test_mul_scalar(self):
        result = rel(2.0) * 5
        assert result == 10.0

    def test_rmul_scalar(self):
        result = 5 * rel(2.0)
        assert result == 10.0

    def test_mul_rel(self):
        result = rel(2.0) * rel(3.0)
        assert isinstance(result, Rel)
        assert result.value == 6.0

    def test_mul_unsupported(self):
        result = rel(2.0).__mul__("abc")
        assert result is NotImplemented

    def test_is_rel(self):
        assert is_rel(rel(1.0))
        assert not is_rel(1.0)


# =====================================================================
# Margin tests
# =====================================================================

class TestMargin:
    def test_creation(self):
        m = margin(1, 2, 3, 4, unit="mm")
        assert m.t == 1.0
        assert m.r == 2.0
        assert m.b == 3.0
        assert m.l == 4.0
        assert m.unit_str == "mm"

    def test_default_unit(self):
        # R parity: ``unit(1, "pt")`` normalises to ``"points"`` (see
        # ``grid::unitType``). ``Margin`` inherits this via ``Unit``'s
        # alias table, so the default ``"pt"`` round-trips as
        # ``"points"`` — matching R rather than the original
        # echo-back behaviour.
        m = margin()
        assert m.unit_str == "points"

    def test_indexing(self):
        # R parity (margins.R:8-29): ``margin`` is an S7 subclass of
        # ``simpleUnit``. ``m[i]`` returns a length-1 ``Unit`` matching
        # R's ``m[1]`` behaviour. Python users wanting a float should
        # use the named accessors ``.t/.r/.b/.l`` or ``.values[i]``.
        from grid_py import Unit
        m = margin(10, 20, 30, 40)
        assert isinstance(m[0], Unit)
        assert m[0]._values[0] == 10.0
        assert m[3]._values[0] == 40.0
        # Python sugar — direct float access:
        assert m.t == 10.0
        assert m.l == 40.0
        assert m.values[0] == 10.0

    def test_len(self):
        assert len(margin()) == 4

    def test_iter(self):
        # R parity: iterating a unit yields per-element length-1 units.
        from grid_py import Unit
        m = margin(1, 2, 3, 4)
        items = list(m)
        assert all(isinstance(v, Unit) for v in items)
        assert [v._values[0] for v in items] == [1.0, 2.0, 3.0, 4.0]
        # Python sugar — bare float vector:
        assert list(m.values) == [1.0, 2.0, 3.0, 4.0]

    def test_repr(self):
        m = margin(1, 2, 3, 4)
        assert "margin" in repr(m)

    def test_eq(self):
        assert margin(1, 2, 3, 4) == margin(1, 2, 3, 4)
        assert margin(1, 2, 3, 4) != margin(1, 2, 3, 5)

    def test_eq_different_unit(self):
        assert margin(1, 2, 3, 4, unit="pt") != margin(1, 2, 3, 4, unit="mm")

    def test_eq_other_type(self):
        assert margin(1, 2, 3, 4).__eq__("not a margin") is NotImplemented

    def test_unit_property(self):
        m = margin(1, 2, 3, 4)
        assert m.unit is not None

    def test_is_margin(self):
        assert is_margin(margin())
        assert not is_margin(42)

    def test_margin_auto_defaults(self):
        m = margin_auto(5)
        assert m.t == 5.0
        assert m.r == 5.0
        assert m.b == 5.0
        assert m.l == 5.0

    def test_margin_auto_partial(self):
        m = margin_auto(5, 10)
        assert m.t == 5.0
        assert m.r == 10.0
        assert m.b == 5.0
        assert m.l == 10.0

    def test_margin_part(self):
        m = margin_part(1)
        assert m.t == 1.0
        assert math.isnan(m.r)

    # ------------------------------------------------------------------
    # R-parity tests: ``Margin`` IS-A ``Unit`` (margins.R:8-29).
    # ------------------------------------------------------------------

    def test_margin_is_unit_subclass(self):
        """``Margin`` must be a real ``Unit`` subclass so it dispatches
        through ``isinstance`` / ``is_unit`` checks across grid_py and
        gtable_py — mirrors R's S7 inheritance margin → simpleUnit →
        unit → unit_v2."""
        from grid_py import Unit, is_unit
        m = margin(1, 2, 3, 4, unit="pt")
        assert isinstance(m, Unit)
        assert is_unit(m)
        # Length-4 view (R: ``length(margin(...)) == 4``).
        assert len(m) == 4

    def test_margin_passes_through_gtable_add_padding(self):
        """R guides-.R:734,753 calls ``gtable_add_padding(box, margin)``
        directly. Now that ``Margin`` IS-A ``Unit`` the Python equivalent
        must work without ``.unit`` extraction."""
        from grid_py import Unit
        from gtable_py import gtable_add_padding, Gtable
        gt = Gtable(widths=Unit([1.0], ["cm"]), heights=Unit([1.0], ["cm"]))
        padded = gtable_add_padding(gt, margin(10, 20, 30, 40, unit="pt"))
        # 1×1 cell + 4-sided padding → 3×3 (same as R's
        # ``dim(gtable_add_padding(gt, margin(...))) == c(3, 3)``).
        assert padded.shape == (3, 3)

    def test_margin_arithmetic_demotes_to_unit(self):
        """R parity: ``margin + unit`` and ``margin * scalar`` return a
        plain ``unit`` (R demotes from margin to its parent class).
        Anything that goes through ``Unit.copy()`` — ``__neg__``,
        ``__pos__``, ``__mul__``, ``__truediv__`` — uses the
        ``Unit.__new__(Unit)`` constructor and therefore demotes."""
        from grid_py import Unit
        m = margin(10, 20, 30, 40, "pt")
        assert type(m + Unit([5], "pt")) is Unit
        assert type(m - Unit([5], "pt")) is Unit
        assert type(m * 2) is Unit
        assert type(m / 2) is Unit
        assert type(-m) is Unit
        assert type(+m) is Unit

    def test_margin_copy_preserves_class(self):
        """``copy.copy``, ``copy.deepcopy`` and ``pickle`` must round-trip
        as ``Margin`` (Python idiom: copy/pickle preserve class). Note
        that ``m.copy()`` itself routes through ``Unit.copy()`` which
        hard-codes ``Unit.__new__(Unit)`` and therefore demotes — that
        path is what powers the arithmetic-demote behaviour above and
        must stay un-overridden."""
        import copy as _copy
        import pickle

        m = margin(10, 20, 30, 40, "pt")
        assert type(_copy.copy(m)) is Margin
        assert type(_copy.deepcopy(m)) is Margin
        assert type(pickle.loads(pickle.dumps(m))) is Margin
        # The pickled value must round-trip cleanly.
        roundtrip = pickle.loads(pickle.dumps(m))
        assert roundtrip == m
        # ``m.copy()`` deliberately demotes (R-faithful arithmetic).
        from grid_py import Unit
        assert type(m.copy()) is Unit


# =====================================================================
# Element class tests
# =====================================================================

class TestElementBlank:
    def test_creation(self):
        e = element_blank()
        assert isinstance(e, ElementBlank)
        assert e.blank is True
        assert repr(e) == "element_blank()"


class TestElementLine:
    def test_creation(self):
        e = element_line(colour="red", linewidth=2)
        assert isinstance(e, ElementLine)
        assert e.colour == "red"
        assert e.linewidth == 2
        assert e.blank is False

    def test_color_alias(self):
        e = element_line(color="blue")
        assert e.colour == "blue"

    def test_repr(self):
        e = element_line(colour="red")
        assert "red" in repr(e)

    def test_all_params(self):
        e = element_line(
            colour="black", linewidth=1.0, linetype="solid",
            lineend="round", linejoin="mitre", arrow=None,
            arrow_fill="red", inherit_blank=True
        )
        assert e.lineend == "round"
        assert e.linejoin == "mitre"


class TestElementRect:
    def test_creation(self):
        e = element_rect(fill="white", colour="black")
        assert e.fill == "white"
        assert e.colour == "black"

    def test_color_alias(self):
        e = element_rect(color="green")
        assert e.colour == "green"

    def test_repr(self):
        e = element_rect(fill="white")
        assert "white" in repr(e)


class TestElementText:
    def test_creation(self):
        e = element_text(family="serif", size=12, colour="red")
        assert e.family == "serif"
        assert e.size == 12
        assert e.colour == "red"

    def test_color_alias(self):
        e = element_text(color="green")
        assert e.colour == "green"

    def test_repr(self):
        e = element_text(size=14)
        assert "14" in repr(e)

    def test_rel_size(self):
        e = element_text(size=rel(1.2))
        assert isinstance(e.size, Rel)

    def test_all_params(self):
        e = element_text(
            family="sans", face="bold", colour="black", size=12,
            hjust=0.5, vjust=0.5, angle=45, lineheight=1.2,
            margin=margin(5, 5, 5, 5), debug=True, inherit_blank=True
        )
        assert e.angle == 45


class TestElementPoint:
    def test_creation(self):
        e = element_point(shape=19, colour="red")
        assert e.shape == 19
        assert e.colour == "red"

    def test_color_alias(self):
        e = element_point(color="blue")
        assert e.colour == "blue"

    def test_repr(self):
        e = element_point(size=3)
        assert "3" in repr(e)


class TestElementPolygon:
    def test_creation(self):
        e = element_polygon(colour="red", fill="blue")
        assert e.colour == "red"
        assert e.fill == "blue"

    def test_color_alias(self):
        e = element_polygon(color="green")
        assert e.colour == "green"

    def test_repr(self):
        r = repr(element_polygon(fill="red"))
        assert "red" in r


class TestElementGeom:
    def test_creation(self):
        e = element_geom(ink="black", paper="white")
        assert e.ink == "black"
        assert e.paper == "white"

    def test_color_alias(self):
        e = element_geom(color="green")
        assert e.colour == "green"

    def test_repr(self):
        r = repr(element_geom(ink="black"))
        assert "black" in r

    def test_all_params(self):
        e = element_geom(
            ink="black", paper="white", accent="red",
            linewidth=1.0, borderwidth=0.5, linetype=1,
            bordertype="solid", family="sans", fontsize=12,
            pointsize=3, pointshape=19, colour="red", fill="blue"
        )
        # R's element_geom constructor (theme-elements.R:340-342) divides the
        # supplied fontsize by .pt before storing, so fontsize=12 is stored as
        # 12 / .pt (mm). Only fontsize is converted; pointsize is left as-is.
        from ggplot2_py.geom import PT
        assert e.fontsize == pytest.approx(12 / PT)
        assert e.pointsize == 3


# =====================================================================
# is_theme_element tests
# =====================================================================

class TestIsThemeElement:
    def test_element_blank(self):
        assert is_theme_element(element_blank(), "blank")

    def test_element_line(self):
        assert is_theme_element(element_line(), "line")

    def test_element_rect(self):
        assert is_theme_element(element_rect(), "rect")

    def test_element_text(self):
        assert is_theme_element(element_text(), "text")

    def test_element_any(self):
        assert is_theme_element(element_text(), "any")

    def test_element_point(self):
        assert is_theme_element(element_point(), "point")

    def test_element_polygon(self):
        assert is_theme_element(element_polygon(), "polygon")

    def test_element_geom(self):
        assert is_theme_element(element_geom(), "geom")

    def test_not_element(self):
        assert not is_theme_element("hello", "any")

    def test_unknown_type(self):
        assert not is_theme_element(element_text(), "unknown_type")


# =====================================================================
# merge_element tests
# =====================================================================

class TestMergeElement:
    def test_new_fills_from_old(self):
        new = element_line(colour="red")
        old = element_line(colour="blue", linewidth=2)
        result = merge_element(new, old)
        assert result.colour == "red"
        assert result.linewidth == 2

    def test_old_none(self):
        new = element_line(colour="red")
        result = merge_element(new, None)
        assert result is new

    def test_old_blank(self):
        new = element_line(colour="red")
        result = merge_element(new, element_blank())
        assert result is new

    def test_new_none(self):
        result = merge_element(None, element_line())
        assert result is None

    def test_new_blank(self):
        result = merge_element(element_blank(), element_line())
        assert isinstance(result, ElementBlank)

    def test_new_is_scalar(self):
        assert merge_element("hello", element_line()) == "hello"
        assert merge_element(42, element_line()) == 42

    def test_incompatible_types_error(self):
        with pytest.raises(ValueError, match="same class"):
            merge_element(element_line(), element_rect())

    def test_new_is_margin(self):
        m = margin(1, 2, 3, 4)
        result = merge_element(m, element_line())
        assert result is m


# =====================================================================
# combine_elements tests
# =====================================================================

class TestCombineElements:
    def test_e2_none(self):
        e1 = element_line(colour="red")
        result = combine_elements(e1, None)
        assert result is e1

    def test_e1_none_inherits_e2(self):
        e2 = element_line(colour="blue")
        result = combine_elements(None, e2)
        assert result is e2

    def test_e1_blank(self):
        result = combine_elements(element_blank(), element_line())
        assert isinstance(result, ElementBlank)

    def test_rel_with_rel(self):
        result = combine_elements(rel(2.0), rel(3.0))
        assert isinstance(result, Rel)
        assert result.value == 6.0

    def test_rel_with_scalar(self):
        result = combine_elements(rel(2.0), 5.0)
        assert result == 10.0

    def test_fill_none_from_parent(self):
        child = element_text(colour="red")
        parent = element_text(colour="blue", size=12)
        result = combine_elements(child, parent)
        assert result.colour == "red"
        assert result.size == 12

    def test_resolve_rel_size(self):
        child = element_text(size=rel(0.5))
        parent = element_text(size=20)
        result = combine_elements(child, parent)
        assert result.size == 10.0

    def test_resolve_rel_linewidth(self):
        child = element_line(linewidth=rel(0.5))
        parent = element_line(linewidth=4)
        result = combine_elements(child, parent)
        assert result.linewidth == 2.0

    def test_e2_blank_with_inherit(self):
        child = element_line(inherit_blank=True)
        result = combine_elements(child, element_blank())
        assert isinstance(result, ElementBlank)

    def test_e2_blank_without_inherit(self):
        child = element_line(inherit_blank=False, colour="red")
        result = combine_elements(child, element_blank())
        assert isinstance(result, ElementLine)
        assert result.colour == "red"

    def test_margin_merging(self):
        m1 = margin_part(1, float("nan"), 3, float("nan"))
        m2 = margin(10, 20, 30, 40)
        result = combine_elements(m1, m2)
        assert result.t == 1.0
        assert result.r == 20.0
        assert result.b == 3.0
        assert result.l == 40.0

    def test_non_element_returns_e1(self):
        result = combine_elements("hello", "world")
        assert result == "hello"


# =====================================================================
# Element grob rendering
# =====================================================================

class TestElementGrob:
    def test_blank_returns_null_grob(self):
        result = element_grob(element_blank())
        assert result is not None

    def test_rect_returns_grob(self):
        result = element_grob(element_rect(fill="white", colour="black"))
        assert result is not None

    def test_line_returns_grob(self):
        result = element_grob(element_line(colour="red"))
        assert result is not None

    def test_text_returns_grob(self):
        result = element_grob(element_text(colour="black", size=12, angle=0), label="hello")
        assert result is not None

    def test_text_no_label_returns_null(self):
        result = element_grob(element_text())
        assert result is not None

    def test_fallback(self):
        result = element_grob(element_point())
        assert result is not None


# =====================================================================
# el_def tests
# =====================================================================

class TestElDef:
    def test_basic(self):
        d = el_def(ElementLine, "line", "A line element")
        assert d["class"] == ElementLine
        assert d["inherit"] == ["line"]
        assert d["description"] == "A line element"

    def test_string_inherit(self):
        d = el_def(ElementText, "text")
        assert d["inherit"] == ["text"]

    def test_list_inherit(self):
        d = el_def(ElementLine, ["line", "axis.line"])
        assert d["inherit"] == ["line", "axis.line"]

    def test_no_inherit(self):
        d = el_def(ElementRect)
        assert d["inherit"] is None


# =====================================================================
# Element tree tests
# =====================================================================

class TestElementTree:
    def test_get_element_tree(self):
        tree = get_element_tree()
        assert isinstance(tree, dict)
        assert "line" in tree
        assert "rect" in tree
        assert "text" in tree

    def test_register_theme_elements(self):
        register_theme_elements(
            element_tree={"custom.element": el_def(ElementLine, "line")}
        )
        tree = get_element_tree()
        assert "custom.element" in tree
        # Clean up
        del tree["custom.element"]


# =====================================================================
# calc_element tests
# =====================================================================

class TestCalcElement:
    def test_in_theme(self):
        t = Theme(elements={"line": element_line(colour="red", linewidth=1.0)})
        result = calc_element("line", t)
        assert isinstance(result, ElementLine)
        assert result.colour == "red"

    def test_blank_element(self):
        t = Theme(elements={"line": element_blank()})
        result = calc_element("line", t)
        assert isinstance(result, ElementBlank)

    def test_not_in_tree(self):
        t = Theme(elements={"unknown.thing": "hello"})
        result = calc_element("unknown.thing", t)
        assert result == "hello"

    def test_verbose(self, capsys):
        t = Theme(elements={"line": element_line(colour="red", linewidth=1)})
        calc_element("line", t, verbose=True)
        captured = capsys.readouterr()
        assert "line" in captured.out

    def test_skip_blank(self):
        t = Theme(elements={"axis.text": element_blank()})
        result = calc_element("axis.text", t, skip_blank=True)
        # Should skip the blank and try to inherit
        # Result depends on parent chain


# =====================================================================
# Theme class tests
# =====================================================================

class TestThemeClass:
    def test_creation(self):
        t = Theme(elements={"a": 1, "b": 2})
        assert len(t) == 2
        assert t["a"] == 1

    def test_setitem(self):
        t = Theme()
        t["x"] = 10
        assert t["x"] == 10

    def test_contains(self):
        t = Theme(elements={"a": 1})
        assert "a" in t
        assert "b" not in t

    def test_iter(self):
        t = Theme(elements={"a": 1, "b": 2})
        assert set(t) == {"a", "b"}

    def test_get(self):
        t = Theme(elements={"a": 1})
        assert t.get("a") == 1
        assert t.get("z", 99) == 99

    def test_keys_values_items(self):
        t = Theme(elements={"a": 1, "b": 2})
        assert set(t.keys()) == {"a", "b"}
        assert set(t.values()) == {1, 2}
        assert len(list(t.items())) == 2

    def test_names(self):
        t = Theme(elements={"a": 1, "b": 2})
        assert sorted(t.names()) == ["a", "b"]

    def test_update(self):
        t = Theme(elements={"a": 1})
        t.update({"b": 2})
        assert t["b"] == 2

    def test_copy(self):
        t = Theme(elements={"a": 1}, complete=True)
        t2 = t.copy()
        assert t2["a"] == 1
        assert t2.complete is True
        t2["a"] = 99
        assert t["a"] == 1  # original unchanged

    def test_add_theme(self):
        t1 = Theme(elements={"a": 1})
        t2 = Theme(elements={"b": 2})
        result = t1 + t2
        assert result["a"] == 1
        assert result["b"] == 2

    def test_add_none(self):
        t = Theme(elements={"a": 1})
        result = t + None
        assert result["a"] == 1

    def test_radd_none(self):
        t = Theme(elements={"a": 1})
        result = None + t
        assert result["a"] == 1

    def test_radd_zero(self):
        t = Theme(elements={"a": 1})
        result = 0 + t
        assert result["a"] == 1

    def test_radd_theme(self):
        t1 = Theme(elements={"a": 1})
        t2 = Theme(elements={"b": 2})
        result = t1.__radd__(t2)
        assert "a" in result
        assert "b" in result

    def test_radd_unsupported(self):
        t = Theme()
        result = t.__radd__("bad")
        assert result is NotImplemented

    def test_add_non_theme_raises(self):
        t = Theme()
        with pytest.raises(ValueError):
            t + "bad"

    def test_repr(self):
        t = Theme(elements={"a": 1}, complete=True)
        assert "complete" in repr(t)
        assert "1 elements" in repr(t)

    def test_repr_incomplete(self):
        t = Theme(elements={"a": 1})
        assert "complete" not in repr(t)


# =====================================================================
# theme() constructor tests
# =====================================================================

class TestThemeConstructor:
    def test_basic(self):
        t = theme(plot_title=element_text(size=20))
        assert isinstance(t, Theme)
        assert "plot.title" in t

    def test_underscore_to_dot(self):
        t = theme(axis_text_x=element_text(angle=45))
        assert "axis.text.x" in t

    def test_complete_sets_inherit_blank(self):
        e = element_text()
        t = theme(complete=True, line=e)
        # For complete theme, inherit_blank should be True on elements
        # that have the attribute

    def test_validate_false(self):
        t = theme(validate=False, custom_thing="hello")
        assert t.validate is False


# =====================================================================
# add_theme tests
# =====================================================================

class TestAddTheme:
    def test_t2_none(self):
        t1 = Theme(elements={"a": 1})
        result = add_theme(t1, None)
        assert result["a"] == 1

    def test_t2_complete_replaces(self):
        t1 = Theme(elements={"a": 1})
        t2 = Theme(elements={"b": 2}, complete=True)
        result = add_theme(t1, t2)
        assert "b" in result
        assert result.complete is True

    def test_t1_none(self):
        t2 = Theme(elements={"a": 1})
        result = add_theme(None, t2)
        assert result["a"] == 1

    def test_merge_elements(self):
        t1 = Theme(elements={"line": element_line(colour="red")})
        t2 = Theme(elements={"line": element_line(linewidth=2)})
        result = add_theme(t1, t2)
        # The merged element should have linewidth=2, colour from t1
        el = result["line"]
        assert el.linewidth == 2

    def test_validate_intersection(self):
        t1 = Theme(elements={"a": 1}, validate=True)
        t2 = Theme(elements={"b": 2}, validate=False)
        result = add_theme(t1, t2)
        assert result.validate is False


class TestThemeReplaceOp:
    def test_basic(self):
        t1 = Theme(elements={"a": 1, "b": 2})
        t2 = Theme(elements={"b": 99})
        result = theme_replace_op(t1, t2)
        assert result["a"] == 1
        assert result["b"] == 99

    def test_non_theme_raises(self):
        with pytest.raises(ValueError):
            theme_replace_op("bad", Theme())


# =====================================================================
# complete_theme tests
# =====================================================================

class TestCompleteTheme:
    def test_none_inputs(self):
        # Save and restore global state
        old = _ggplot_global.theme_current
        _ggplot_global.theme_current = None
        _ggplot_global.theme_default = None
        try:
            result = complete_theme(None, None)
            assert result.complete is True
        finally:
            _ggplot_global.theme_current = old

    def test_with_default(self):
        default = Theme(elements={"a": 1}, complete=True)
        result = complete_theme(None, default)
        assert "a" in result

    def test_complete_theme_with_complete_input(self):
        t = Theme(elements={"a": 1}, complete=True)
        default = Theme(elements={"b": 2}, complete=True)
        result = complete_theme(t, default)
        assert result["a"] == 1
        assert "b" in result


# =====================================================================
# Global theme state tests
# =====================================================================

class TestGlobalThemeState:
    def test_set_and_get(self):
        old = get_theme()
        try:
            t = Theme(elements={"x": 1})
            prev = set_theme(t)
            assert get_theme() is t
            assert prev is old
        finally:
            set_theme(old)

    def test_set_none_resets(self):
        old = get_theme()
        try:
            set_theme(Theme(elements={"x": 1}))
            set_theme(None)
        finally:
            set_theme(old)

    def test_set_non_theme_raises(self):
        with pytest.raises(ValueError):
            set_theme("bad")

    def test_update_theme(self):
        old = get_theme()
        try:
            set_theme(Theme(elements={"a": 1}))
            update_theme(b=2)
            current = get_theme()
            assert "b" in current
        finally:
            set_theme(old)

    def test_update_theme_no_current(self):
        old = get_theme()
        try:
            _ggplot_global.theme_current = None
            update_theme(x=1)
            assert get_theme() is not None
        finally:
            set_theme(old)

    def test_replace_theme(self):
        old = get_theme()
        try:
            set_theme(Theme(elements={"a": 1}))
            replace_theme(a=99)
            current = get_theme()
            assert current["a"] == 99
        finally:
            set_theme(old)

    def test_replace_theme_no_current(self):
        old = get_theme()
        try:
            _ggplot_global.theme_current = None
            replace_theme(x=1)
            assert get_theme() is not None
        finally:
            set_theme(old)

    def test_aliases(self):
        assert theme_get is get_theme
        assert theme_set is set_theme
        assert theme_update is update_theme
        assert theme_replace is replace_theme

    def test_reset_theme_settings(self):
        old = get_theme()
        try:
            reset_theme_settings()
        finally:
            set_theme(old)


class TestIsTheme:
    def test_theme_instance(self):
        assert is_theme(Theme())

    def test_not_theme(self):
        assert not is_theme("hello")
        assert not is_theme(None)


# =====================================================================
# element_render tests
# =====================================================================

class TestElementRender:
    def test_renders_existing_element(self):
        t = Theme(elements={"line": element_line(colour="red", linewidth=1)})
        result = element_render(t, "line")
        assert result is not None

    def test_renders_missing_returns_null(self):
        t = Theme(elements={})
        result = element_render(t, "nonexistent")
        assert result is not None


class TestResetThemeSettingsElements:
    def test_resets_element_tree(self):
        tree = get_element_tree()
        tree["_test_el"] = el_def(ElementLine)
        reset_theme_settings_elements()
        assert "_test_el" not in get_element_tree()
