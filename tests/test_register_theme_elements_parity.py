"""R-parity tests for ``register_theme_elements`` default element values.

Covers Issue #2: ``register_theme_elements`` used to silently drop the default
element *values* passed via ``**kwargs`` (it only merged the ``element_tree``).
After the fix it mirrors R's ``register_theme_elements()`` (R/theme-elements.R)::

    register_theme_elements <- function(..., element_tree = NULL, complete = TRUE) {
      old <- ggplot_global$theme_default
      t <- theme(..., complete = complete)
      ggplot_global$theme_default <- ggplot_global$theme_default %+replace% t
      check_element_tree(element_tree)
      ggplot_global$element_tree <- defaults(element_tree, ggplot_global$element_tree)
      invisible(old)
    }

The R reference values these tests assert were captured from ggplot2 4.0.2 via
``tools/r_parity.py`` (both-side internal-computation verification):

    register_theme_elements(
        ggh4x.facet.nestline = element_blank(),
        element_tree = list(ggh4x.facet.nestline = el_def("element_line", "line")))
    calc_element("ggh4x.facet.nestline", complete_theme(theme_grey()))
    #> element_blank          (registered default surfaces, line OFF)

    # element_tree only, no default value:
    #> element_line           (inherits the visible "line" ancestor)

    # user override theme(ggh4x.facet.nestline = element_line(colour="blue")):
    #> element_line, colour = "blue"
"""

import pytest

from ggplot2_py import (
    register_theme_elements,
    reset_theme_settings,
    el_def,
    complete_theme,
    calc_element,
    get_element_tree,
    element_blank,
    element_line,
    element_rect,
    theme,
    theme_grey,
)
from ggplot2_py.theme_elements import (
    ElementBlank,
    ElementLine,
    ElementRect,
    _ggplot_global,
    _ELEMENT_TREE,
)


@pytest.fixture(autouse=True)
def _reset_theme():
    """Ensure global theme/element-tree state is pristine around every test."""
    reset_theme_settings()
    yield
    reset_theme_settings()


class TestRegisterDefaultValues:
    """The dropped-default-values bug and its R parity."""

    def test_registered_blank_default_surfaces_in_completed_theme(self):
        # R: completed theme resolves the registered element_blank (line OFF).
        register_theme_elements(
            element_tree={"ggh4x.facet.nestline": el_def(ElementLine, "line")},
            **{"ggh4x.facet.nestline": element_blank()},
        )
        ct = complete_theme(theme_grey())
        resolved = calc_element("ggh4x.facet.nestline", ct)
        assert isinstance(resolved, ElementBlank)

    def test_without_default_inherits_visible_line(self):
        # R: registering only the element_tree (no default value) leaves the
        # element to inherit the visible "line" ancestor -> element_line.
        register_theme_elements(
            element_tree={"ggh4x.facet.nestline": el_def(ElementLine, "line")},
        )
        ct = complete_theme(theme_grey())
        resolved = calc_element("ggh4x.facet.nestline", ct)
        assert isinstance(resolved, ElementLine)
        assert not isinstance(resolved, ElementBlank)

    def test_user_override_turns_element_on(self):
        # R: a user theme override beats the registered blank default.
        register_theme_elements(
            element_tree={"ggh4x.facet.nestline": el_def(ElementLine, "line")},
            **{"ggh4x.facet.nestline": element_blank()},
        )
        ct = complete_theme(
            theme_grey()
            + theme(**{"ggh4x.facet.nestline": element_line(colour="blue")})
        )
        resolved = calc_element("ggh4x.facet.nestline", ct)
        assert isinstance(resolved, ElementLine)
        assert not isinstance(resolved, ElementBlank)
        assert getattr(resolved, "colour", None) == "blue"

    def test_registered_default_stored_in_theme_default(self):
        # The %+replace% merge must land the value on ggplot_global$theme_default.
        register_theme_elements(
            element_tree={"ggxyz.custom.rect": el_def(ElementRect, "rect")},
            **{"ggxyz.custom.rect": element_rect(fill="red")},
        )
        td = _ggplot_global.theme_default
        assert "ggxyz.custom.rect" in td
        assert isinstance(td["ggxyz.custom.rect"], ElementRect)
        assert td["ggxyz.custom.rect"].fill == "red"

    def test_custom_default_resolves_with_inherited_props(self):
        # R: a registered default for a brand-new element (inheriting "rect")
        # resolves with its own fill plus inherited props -> fill stays "red".
        register_theme_elements(
            element_tree={"ggxyz.custom.rect": el_def(ElementRect, "rect")},
            **{"ggxyz.custom.rect": element_rect(fill="red")},
        )
        resolved = calc_element("ggxyz.custom.rect", complete_theme(theme_grey()))
        assert isinstance(resolved, ElementRect)
        assert resolved.fill == "red"


class TestElementTreeMerge:
    """The element_tree side: new-wins merge == R defaults(new, old)."""

    def test_element_tree_added(self):
        register_theme_elements(
            element_tree={"ggzzz.only.tree": el_def(ElementLine, "line")},
        )
        assert "ggzzz.only.tree" in get_element_tree()

    def test_element_tree_new_wins(self):
        # Re-registering the same key overwrites (new-wins), matching R.
        register_theme_elements(
            element_tree={"ggzzz.dup": el_def(ElementLine, "line")},
        )
        register_theme_elements(
            element_tree={"ggzzz.dup": el_def(ElementRect, "rect")},
        )
        entry = get_element_tree()["ggzzz.dup"]
        assert entry["class"] is ElementRect
        assert entry["inherit"] == ["rect"]


class TestBackCompatAndSignature:
    """Signature additions and back-compat."""

    def test_tree_only_call_still_works(self):
        # Old call style (element_tree only, no kwargs) must not error.
        register_theme_elements(
            element_tree={"ggaaa.tree.only": el_def(ElementLine, "line")},
        )
        assert "ggaaa.tree.only" in get_element_tree()

    def test_no_args_is_noop(self):
        # Calling with nothing should be a harmless no-op.
        before = dict(get_element_tree())
        register_theme_elements()
        assert dict(get_element_tree()) == before

    def test_complete_param_accepted(self):
        register_theme_elements(complete=False, **{"plot.title": element_blank()})
        assert "plot.title" in _ggplot_global.theme_default
        assert isinstance(
            _ggplot_global.theme_default["plot.title"], ElementBlank
        )


class TestThemeAllNullParity:
    """theme_all_null must read the STATIC element tree, not the live one.

    R's theme_all_null() comment: "We read from `.element_tree` instead of
    `ggplot_global$element_tree` because we don't want to change our results
    just because a user has defined new theme elements." Without this, a
    runtime-registered element gets a None placeholder in every complete theme,
    blocking the theme_default fallback fill in complete_theme.
    """

    def test_theme_grey_excludes_runtime_registered_element(self):
        register_theme_elements(
            element_tree={"ggh4x.facet.nestline": el_def(ElementLine, "line")},
            **{"ggh4x.facet.nestline": element_blank()},
        )
        # A fresh theme_grey() must NOT contain the runtime-registered key
        # (parity with R, where theme_grey() never lists it).
        assert "ggh4x.facet.nestline" not in theme_grey()
        # ... but theme_default (the fallback) must.
        assert "ggh4x.facet.nestline" in _ggplot_global.theme_default

    def test_theme_all_null_uses_static_tree(self):
        from ggplot2_py.theme_defaults import _theme_all_null

        register_theme_elements(
            element_tree={"ggbbb.extra": el_def(ElementLine, "line")},
        )
        null_theme = _theme_all_null()
        assert "ggbbb.extra" not in null_theme
        # All static built-in keys are present.
        assert "line" in null_theme
        assert "plot.background" in null_theme
        assert set(null_theme.keys()) == set(_ELEMENT_TREE.keys())
