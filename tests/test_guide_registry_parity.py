"""R-parity tests for the extensible guide registry (Issue #4).

Covers ``ggplot2_py.guide``'s ``_validate_guide`` / ``_resolve_guide_name`` /
``register_guide`` and the ``Guide`` auto-registration via ``__init_subclass__``.

The gold standard is R's ``ggplot2::validate_guide`` (``R/guides-.R``), which
resolves a guide *string* dynamically (``find_global("guide_" + name)``) and
*calls* the found constructor, making the set of resolvable guide strings
open/extensible -- any exported ``guide_*`` function works. Before this fix the
Python port used a closed, hardcoded local registry so third-party guides could
not be referenced by string. These tests assert both R parity for the built-in
strings and the new extensibility.

The R-side assertions run through the differential harness in ``tools/r_parity``
when an R installation is available; they are skipped otherwise so the suite
stays runnable in pure-Python environments.
"""
from __future__ import annotations

import os
import sys

import pytest

from ggplot2_py.guide import (
    Guide,
    GuideAxis,
    GuideAxisLogticks,
    GuideAxisStack,
    GuideAxisTheta,
    GuideBins,
    GuideColourbar,
    GuideColoursteps,
    GuideCustom,
    GuideLegend,
    GuideNone,
    guide_legend,
    register_guide,
)
from ggplot2_py.guide import _resolve_guide_name, _validate_guide


# --------------------------------------------------------------------------- #
# R differential harness (optional)
# --------------------------------------------------------------------------- #

_TOOLS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"
)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

try:  # the harness imports cleanly even without R present
    from r_parity import run_r as _run_r  # type: ignore
    _HAVE_HARNESS = True
except Exception:  # pragma: no cover - harness missing
    _run_r = None
    _HAVE_HARNESS = False


def _r_available() -> bool:
    """True if Rscript + ggplot2 can answer a trivial ``validate_guide`` call."""
    if not _HAVE_HARNESS:
        return False
    try:
        out = _run_r(
            'cat(class(validate_guide("legend"))[1])', fns=("validate_guide",)
        )
        return out.strip() == "GuideLegend"
    except Exception:  # pragma: no cover - R missing / broken
        return False


requires_r = pytest.mark.skipif(
    not _r_available(), reason="Rscript/ggplot2 not available for parity check"
)


# Canonical built-in string -> expected resolved class. These are the strings
# R's validate_guide resolves; "custom" is handled separately (R errors there).
BUILTIN_STRINGS = [
    ("legend", GuideLegend),
    ("colourbar", GuideColourbar),
    ("colorbar", GuideColourbar),
    ("coloursteps", GuideColoursteps),
    ("colorsteps", GuideColoursteps),
    ("bins", GuideBins),
    ("axis", GuideAxis),
    ("none", GuideNone),
    ("axis_logticks", GuideAxisLogticks),
    ("axis_theta", GuideAxisTheta),
    ("axis_stack", GuideAxisStack),
]


# --------------------------------------------------------------------------- #
# Built-in string resolution (Python-only invariants)
# --------------------------------------------------------------------------- #

class TestBuiltinStrings:
    """Every currently-supported built-in string still resolves correctly."""

    @pytest.mark.parametrize("name,cls", BUILTIN_STRINGS)
    def test_resolves_to_expected_class(self, name, cls):
        guide = _validate_guide(name)
        assert isinstance(guide, cls)

    def test_custom_string_preserved(self):
        # R's validate_guide("custom") errors (guide_custom needs a grob), but
        # the Python port has always resolved the bare class; preserve that.
        assert isinstance(_validate_guide("custom"), GuideCustom)

    def test_case_insensitive(self):
        assert isinstance(_validate_guide("Legend"), GuideLegend)
        assert isinstance(_validate_guide("COLOURBAR"), GuideColourbar)

    def test_dash_treated_as_underscore(self):
        assert isinstance(_validate_guide("axis-logticks"), GuideAxisLogticks)

    def test_string_resolution_uses_constructor_defaults(self):
        # R calls guide_<name>(); registering the *constructor* (not the bare
        # class) means string resolution yields constructor defaults.
        via_string = _validate_guide("legend")
        via_ctor = guide_legend()
        assert via_string.params == via_ctor.params

    def test_unknown_guide_aborts_clearly(self):
        with pytest.raises(Exception) as exc:
            _validate_guide("totally_not_a_guide")
        assert "totally_not_a_guide" in str(exc.value)

    def test_empty_string_aborts(self):
        with pytest.raises(Exception):
            _validate_guide("")


# --------------------------------------------------------------------------- #
# Pass-through of objects/classes (unchanged behaviour)
# --------------------------------------------------------------------------- #

class TestPassThrough:
    def test_instance_passthrough(self):
        g = guide_legend()
        assert _validate_guide(g) is g

    def test_class_instantiated(self):
        out = _validate_guide(GuideLegend)
        assert isinstance(out, GuideLegend)


# --------------------------------------------------------------------------- #
# Extensibility: register_guide + auto-registration
# --------------------------------------------------------------------------- #

class TestRegisterGuide:
    """A third-party guide becomes referenceable by string."""

    def test_register_constructor_mirrors_r_guide_myfake(self):
        # R: define guide_myfake <- function() guide_legend(); then
        # validate_guide("myfake") resolves to GuideLegend. Mirror that.
        def guide_myfake():
            return guide_legend()

        register_guide("myfake_ctor", guide_myfake)
        try:
            out = _validate_guide("myfake_ctor")
            assert isinstance(out, GuideLegend)
        finally:
            Guide._registry.pop("myfake_ctor", None)

    def test_register_class(self):
        register_guide("legendish_class", GuideLegend)
        try:
            out = _validate_guide("legendish_class")
            assert isinstance(out, GuideLegend)
        finally:
            Guide._registry.pop("legendish_class", None)

    def test_register_is_case_insensitive(self):
        register_guide("MixedCaseGuide", GuideLegend)
        try:
            assert isinstance(_validate_guide("mixedcaseguide"), GuideLegend)
        finally:
            Guide._registry.pop("mixedcaseguide", None)

    def test_register_rejects_empty_name(self):
        with pytest.raises(Exception):
            register_guide("", GuideLegend)

    def test_register_rejects_non_callable(self):
        with pytest.raises(Exception):
            register_guide("bad", object())

    def test_constructor_overrides_autoregistered_class(self):
        sentinel = guide_legend()

        def my_ctor():
            return sentinel

        register_guide("override_test", my_ctor)
        try:
            # registered constructor is *called*, returning our sentinel
            assert _validate_guide("override_test") is sentinel
        finally:
            Guide._registry.pop("override_test", None)


class TestAutoRegistration:
    """``Guide.__init_subclass__`` registers subclasses by name."""

    def test_subclass_auto_registers(self):
        class GuideStringlegendTest(Guide):
            pass

        try:
            out = _validate_guide("stringlegendtest")
            assert isinstance(out, GuideStringlegendTest)
        finally:
            Guide._registry.pop("stringlegendtest", None)

    def test_builtin_classes_present_in_registry(self):
        # Auto-registration ran at import for every built-in subclass.
        for key in ("legend", "colourbar", "coloursteps", "bins", "axis",
                    "none", "custom"):
            assert key in Guide._registry


class TestNamespaceForm:
    """``pkg::name`` resolves on the bare name (R strips the namespace)."""

    def test_pkg_prefixed_name(self):
        assert isinstance(_validate_guide("anypkg::legend"), GuideLegend)

    def test_pkg_prefixed_custom(self):
        def guide_nsfake():
            return guide_legend()

        register_guide("nsfake", guide_nsfake)
        try:
            assert isinstance(_validate_guide("somepkg::nsfake"), GuideLegend)
        finally:
            Guide._registry.pop("nsfake", None)


# --------------------------------------------------------------------------- #
# Both-side R parity (skipped when R is unavailable)
# --------------------------------------------------------------------------- #

@requires_r
class TestRParity:
    @pytest.mark.parametrize("name,cls", BUILTIN_STRINGS)
    def test_builtin_matches_r(self, name, cls):
        r_class = _run_r(
            "cat(class(validate_guide(%r))[1])" % name, fns=("validate_guide",)
        ).strip()
        py_class = type(_validate_guide(name)).__name__
        assert r_class == py_class == cls.__name__

    def test_custom_guide_resolution_matches_r(self):
        # R: guide_myfake <- function() guide_legend(); validate_guide("myfake")
        r_class = _run_r(
            'guide_myfake <- function() guide_legend()\n'
            'cat(class(validate_guide("myfake"))[1])',
            fns=("validate_guide",),
        ).strip()

        def guide_myfake():
            return guide_legend()

        register_guide("myfake", guide_myfake)
        try:
            py_class = type(_validate_guide("myfake")).__name__
        finally:
            Guide._registry.pop("myfake", None)
        assert r_class == py_class == "GuideLegend"
