"""Tests for ggplot2_py.theme — theme system."""

import pytest
from ggplot2_py import (
    theme,
    is_theme,
    theme_grey,
    theme_gray,
    theme_bw,
    theme_linedraw,
    theme_light,
    theme_dark,
    theme_minimal,
    theme_classic,
    theme_void,
    theme_test,
    element_blank,
    element_line,
    element_rect,
    element_text,
    margin,
    Margin,
)
from ggplot2_py.theme import Theme
from ggplot2_py.theme_elements import ElementBlank, ElementLine, ElementRect, ElementText


class TestThemeConstructor:
    """Test theme() constructor."""

    def test_creates_theme(self):
        t = theme()
        assert isinstance(t, Theme)

    def test_is_theme(self):
        t = theme()
        assert is_theme(t) is True


class TestThemeGrey:
    """Test theme_grey (complete theme)."""

    def test_is_complete(self):
        t = theme_grey()
        assert t.complete is True

    def test_is_theme(self):
        t = theme_grey()
        assert is_theme(t) is True

    def test_gray_alias(self):
        # theme_gray is an alias for theme_grey
        tg = theme_grey()
        ta = theme_gray()
        assert type(tg) == type(ta)


class TestThemeBw:
    """Test theme_bw."""

    def test_is_complete(self):
        t = theme_bw()
        assert t.complete is True

    def test_differs_from_grey(self):
        tg = theme_grey()
        tb = theme_bw()
        # They should be different Theme objects
        assert tg is not tb


class TestThemeAddition:
    """Test theme + operator."""

    def test_add_theme_element(self):
        result = theme_grey() + theme(axis_text_x=element_text(angle=45))
        assert isinstance(result, Theme)

    def test_add_preserves_complete(self):
        result = theme_grey() + theme(axis_text_x=element_text(angle=45))
        assert result.complete is True


class TestElementBlank:
    """Test element_blank."""

    def test_creates_element_blank(self):
        eb = element_blank()
        assert isinstance(eb, ElementBlank)


class TestElementLine:
    """Test element_line."""

    def test_creates_element_line(self):
        el = element_line(colour="red")
        assert isinstance(el, ElementLine)

    def test_colour_attribute(self):
        el = element_line(colour="red")
        assert el.colour == "red"


class TestElementText:
    """Test element_text."""

    def test_creates_element_text(self):
        et = element_text(size=12)
        assert isinstance(et, ElementText)


class TestElementRect:
    """Test element_rect."""

    def test_creates_element_rect(self):
        er = element_rect(fill="white")
        assert isinstance(er, ElementRect)


class TestIsTheme:
    """Test is_theme predicate."""

    def test_true_for_theme(self):
        assert is_theme(theme()) is True

    def test_true_for_complete_theme(self):
        assert is_theme(theme_grey()) is True

    def test_false_for_string(self):
        assert is_theme("theme") is False

    def test_false_for_none(self):
        assert is_theme(None) is False


class TestMargin:
    """Test margin() helper."""

    def test_creates_margin(self):
        m = margin(5, 10, 5, 10)
        assert isinstance(m, Margin)

    def test_margin_values(self):
        m = margin(5, 10, 15, 20)
        assert m.t == 5.0
        assert m.r == 10.0
        assert m.b == 15.0
        assert m.l == 20.0

    def test_margin_repr(self):
        m = margin(5, 10, 5, 10)
        r = repr(m)
        assert "margin" in r.lower() or "Margin" in r


class TestAllCompleteThemes:
    """Verify all complete theme functions create Theme objects."""

    @pytest.mark.parametrize(
        "theme_fn",
        [
            theme_grey, theme_gray, theme_bw, theme_linedraw,
            theme_light, theme_dark, theme_minimal, theme_classic,
            theme_void, theme_test,
        ],
        ids=lambda fn: fn.__name__,
    )
    def test_creates_complete_theme(self, theme_fn):
        t = theme_fn()
        assert isinstance(t, Theme)
        assert t.complete is True
