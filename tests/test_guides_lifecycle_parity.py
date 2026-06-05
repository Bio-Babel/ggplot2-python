"""R-parity tests for the Guides build lifecycle (Issue #3).

Covers the two-layer fix for ``guides(<aes>=...)`` crashing for every guide:

Layer 1 -- a SINGLE guide build site (Design B(ii)).
    R's ``ggplot_build`` builds the guides exactly once
    (``plot@guides <- plot@guides$build(...)``, ``R/plot-build.R``) and
    ``ggplot_gtable`` only ``assemble``s them (``plot@guides$assemble(theme)``,
    ``R/plot-render.R``).  The Python port used to re-run the whole
    ``setup -> train -> merge -> process_layers`` lifecycle in
    ``_table_add_legends`` on an already-built container whose ``guides`` had
    become a bare list, so ``setup`` did ``list.get(aes)`` -> ``AttributeError``.
    After the fix a built ``Guides`` (flagged ``_built``) is assembled directly;
    only the common no-``guides()`` case (``plot.guides is None``) is built from
    scales.  Either path runs exactly ONE build.

Layer 2 -- representation unification (R named-list parity).
    ``Guides.guides`` is a single insertion-ordered NAMED structure across the
    entire lifecycle: a dict keyed by aesthetic at ``setup`` and re-keyed by
    ``{order}_{hash}`` at ``merge`` (= R's ``names(self$guides) <- hashes``),
    kept parallel to ``params`` / ``aesthetics``.  No method reassigns it to a
    bare list, and ``setup`` can never ``.get`` a list.

R reference values are captured from ggplot2 4.0.2 via ``tools/r_parity.py``
(both-side internal-computation verification) and skipped when R is absent.
"""
from __future__ import annotations

import os
import sys

import pytest
import pandas as pd

from ggplot2_py import (
    ggplot,
    aes,
    geom_point,
    guides,
    guide_legend,
    guide_colourbar,
)
from ggplot2_py.guide import Guides, GuideLegend, GuideNone, guide_none
from ggplot2_py.plot import ggplot_build
from ggplot2_py.plot_render import ggplotGrob


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
    """True if Rscript + ggplot2 can build a trivial legend plot."""
    if not _HAVE_HARNESS:
        return False
    try:
        out = _run_r(
            "p <- ggplot(mtcars, aes(mpg, wt, colour=factor(cyl))) + geom_point();"
            'cat(class(ggplotGrob(p))[1])'
        )
        return "gtable" in out
    except Exception:  # pragma: no cover - R missing / broken
        return False


requires_r = pytest.mark.skipif(
    not _r_available(), reason="Rscript/ggplot2 not available for parity check"
)


DF = pd.DataFrame(
    {
        "x": [1, 2, 3, 4],
        "y": [4, 3, 2, 1],
        "g": ["a", "b", "a", "b"],
        "s": ["p", "q", "p", "q"],
        "v": [1.0, 2.0, 3.0, 4.0],
    }
)


def _right_box_kind(grob) -> str:
    """Return 'REAL' if the right guide-box holds a legend, else 'ZERO'."""
    for nm, gr in zip(grob.layout["name"], grob.grobs):
        if nm == "guide-box-right":
            cls = type(gr).__name__
            return "ZERO" if cls in ("Grob", "ZeroGrob") else "REAL"
    return "MISSING"


def _guide_box_kinds(grob) -> dict:
    """Map every guide-box cell name -> 'REAL'/'ZERO'."""
    out = {}
    for nm, gr in zip(grob.layout["name"], grob.grobs):
        if "guide-box" in nm:
            cls = type(gr).__name__
            out[nm] = "ZERO" if cls in ("Grob", "ZeroGrob") else "REAL"
    return out


# --------------------------------------------------------------------------- #
# Layer 1 -- the crash is gone; every guide path builds (Python invariants)
# --------------------------------------------------------------------------- #

class TestGuidesBuildNoCrash:
    """``+ guides(...)`` used to raise ``'list' object has no attribute 'get'``."""

    def test_guides_legend_builds(self):
        p = ggplot(DF, aes("x", "y", colour="g")) + geom_point() + guides(
            colour=guide_legend()
        )
        gt = ggplotGrob(p)  # must not raise
        assert _right_box_kind(gt) == "REAL"

    def test_guides_colourbar_builds(self):
        p = ggplot(DF, aes("x", "y", colour="v")) + geom_point() + guides(
            colour=guide_colourbar()
        )
        gt = ggplotGrob(p)
        assert _right_box_kind(gt) == "REAL"

    def test_guides_none_suppresses(self):
        # ``guides(colour='none')`` must SUPPRESS the otherwise-present legend.
        p = ggplot(DF, aes("x", "y", colour="g")) + geom_point() + guides(
            colour="none"
        )
        gt = ggplotGrob(p)
        assert _right_box_kind(gt) == "ZERO"

    def test_two_guides_at_once(self):
        p = (
            ggplot(DF, aes("x", "y", colour="g", shape="s"))
            + geom_point()
            + guides(colour=guide_legend(), shape=guide_legend())
        )
        gt = ggplotGrob(p)
        assert _right_box_kind(gt) == "REAL"

    def test_common_no_guides_call_still_legends(self):
        # The majority case: legend springs from ``aes(colour=)`` with no
        # ``guides()`` call. Must keep working (build-from-scales path).
        p = ggplot(DF, aes("x", "y", colour="g")) + geom_point()
        gt = ggplotGrob(p)
        assert _right_box_kind(gt) == "REAL"

    def test_scale_guide_path_regression(self):
        # ``scale_*(guide=...)`` with no ``guides()`` call still works.
        from ggplot2_py import scale_colour_discrete

        p = (
            ggplot(DF, aes("x", "y", colour="g"))
            + geom_point()
            + scale_colour_discrete(guide=guide_legend(reverse=True))
        )
        gt = ggplotGrob(p)
        assert _right_box_kind(gt) == "REAL"

    def test_scale_guide_overridden_by_user_none(self):
        # User ``guides(colour='none')`` wins over a scale-level guide.
        from ggplot2_py import scale_colour_discrete

        p = (
            ggplot(DF, aes("x", "y", colour="g"))
            + geom_point()
            + scale_colour_discrete(guide=guide_legend())
            + guides(colour="none")
        )
        gt = ggplotGrob(p)
        assert _right_box_kind(gt) == "ZERO"

    def test_exactly_five_guide_box_cells(self):
        # R 3.5+ topology: all five cells always present.
        p = ggplot(DF, aes("x", "y", colour="g")) + geom_point() + guides(
            colour=guide_legend()
        )
        kinds = _guide_box_kinds(ggplotGrob(p))
        assert set(kinds) == {
            "guide-box-right",
            "guide-box-left",
            "guide-box-bottom",
            "guide-box-top",
            "guide-box-inside",
        }


# --------------------------------------------------------------------------- #
# Single build site: ``plot.guides`` is built once and marked ``_built``
# --------------------------------------------------------------------------- #

class TestSingleBuildSite:
    def test_build_marks_built(self):
        p = ggplot(DF, aes("x", "y", colour="g")) + geom_point() + guides(
            colour=guide_legend()
        )
        built = ggplot_build(p).plot.guides
        assert isinstance(built, Guides)
        assert getattr(built, "_built", False) is True

    def test_no_guides_call_leaves_none(self):
        # The common case never pre-builds: ``plot.guides`` stays None so
        # ``_table_add_legends`` is the single (build-from-scales) builder.
        p = ggplot(DF, aes("x", "y", colour="g")) + geom_point()
        assert ggplot_build(p).plot.guides is None

    def test_built_guides_assemble_directly(self):
        # An already-built container assembles without a second setup/train.
        p = ggplot(DF, aes("x", "y", colour="g")) + geom_point() + guides(
            colour=guide_legend()
        )
        built = ggplot_build(p).plot.guides
        boxes = built.assemble({}) or {}
        assert boxes.get("right") is not None


# --------------------------------------------------------------------------- #
# Layer 2 -- representation unification (NAMED structure throughout)
# --------------------------------------------------------------------------- #

class TestRepresentationUnification:
    def test_setup_returns_named_dict(self):
        # A fresh user Guides, set up against a scale, yields a dict keyed by
        # aesthetic with parallel params/aesthetics.
        p = ggplot(DF, aes("x", "y", colour="g")) + geom_point()
        b = ggplot_build(p)
        scales = b.plot.scales.non_position_scales().scales
        aes_names = [s.aesthetics[0] for s in scales]
        child = Guides().setup(scales, aesthetics=aes_names, default=GuideLegend())
        assert isinstance(child.guides, dict)
        assert list(child.guides.keys()) == aes_names
        assert len(child.params) == len(child.guides)
        assert child.aesthetics == aes_names

    def test_lifecycle_never_produces_bare_list(self):
        # setup -> train -> merge -> process_layers: guides stays a dict.
        p = ggplot(DF, aes("x", "y", colour="g")) + geom_point()
        b = ggplot_build(p)
        scales = list(b.plot.scales.non_position_scales().scales)
        aes_names = [s.aesthetics[0] for s in scales]
        g = Guides().setup(scales, aesthetics=aes_names, default=GuideLegend())
        assert isinstance(g.guides, dict)
        g.train(scales, {})
        assert isinstance(g.guides, dict)
        g.merge()
        assert isinstance(g.guides, dict)
        g.process_layers([], theme={})
        assert isinstance(g.guides, dict)

    def test_merge_rekeys_by_hash(self):
        # After merge the dict keys are the ``{order}_{hash}`` combos, not the
        # aesthetic names (R: ``names(self$guides) <- hashes``).
        p = ggplot(DF, aes("x", "y", colour="g")) + geom_point()
        b = ggplot_build(p)
        scales = list(b.plot.scales.non_position_scales().scales)
        aes_names = [s.aesthetics[0] for s in scales]
        g = Guides().setup(scales, aesthetics=aes_names, default=GuideLegend())
        g.train(scales, {})
        g.merge()
        # The key encodes the order-padded hash; aesthetic is preserved in the
        # parallel ``aesthetics`` field.
        assert all("_" in k for k in g.guides.keys())
        assert g.aesthetics == ["colour"]

    def test_setup_never_get_on_list(self):
        # Hardening: calling setup on a Guides whose ``guides`` is a bare list
        # must not raise (defensive; mirrors a post-build re-entry).
        p = ggplot(DF, aes("x", "y", colour="g")) + geom_point()
        b = ggplot_build(p)
        scales = list(b.plot.scales.non_position_scales().scales)
        aes_names = [s.aesthetics[0] for s in scales]
        legacy = Guides()
        legacy.guides = [guide_legend()]  # bare list (legacy / external)
        legacy.aesthetics = ["colour"]
        child = legacy.setup(scales, aesthetics=aes_names, default=GuideLegend())
        assert isinstance(child.guides, dict)

    def test_get_guide_resolves_user_dict_and_aesthetics(self):
        # String lookup works both at the user stage (dict key) and after the
        # lifecycle populates ``aesthetics``.
        user = Guides({"colour": guide_legend()})
        assert isinstance(user.get_guide("colour"), GuideLegend)
        assert user.get_guide("fill") is None

    def test_clone_preserves_built_flag(self):
        from ggplot2_py.guide import _clone_guides

        g = Guides({"colour": guide_legend()})
        g._built = True
        clone = _clone_guides(g)
        assert clone._built is True
        # COW: mutating the clone's dict does not touch the original.
        clone.guides["fill"] = guide_legend()
        assert "fill" not in g.guides


# --------------------------------------------------------------------------- #
# Copy-on-write: ``p2 = p1 + guides(...)`` must not mutate ``p1``
# --------------------------------------------------------------------------- #

class TestCopyOnWrite:
    def test_adding_guides_does_not_mutate_base(self):
        p1 = ggplot(DF, aes("x", "y", colour="g")) + geom_point()
        p2 = p1 + guides(colour="none")
        # p1 still shows a legend; p2 suppresses it.
        assert _right_box_kind(ggplotGrob(p1)) == "REAL"
        assert _right_box_kind(ggplotGrob(p2)) == "ZERO"


# --------------------------------------------------------------------------- #
# R layout parity (both-side, harness-gated)
# --------------------------------------------------------------------------- #

@requires_r
class TestRLayoutParity:
    """The legend gtable topology matches R cell-for-cell."""

    def _r_guide_box_layout(self, r_plot_expr: str) -> list:
        out = _run_r(
            f"""
            df <- data.frame(x=c(1,2,3,4), y=c(4,3,2,1),
                             g=c("a","b","a","b"), s=c("p","q","p","q"),
                             v=c(1,2,3,4))
            p <- {r_plot_expr}
            gt <- ggplotGrob(p)
            lay <- gt$layout
            gb <- lay[grepl("guide-box", lay$name),]
            gb <- gb[order(gb$name),]
            for (i in seq_len(nrow(gb))) {{
              idx <- which(lay$name==gb$name[i])
              g <- gt$grobs[[idx]]
              cat(gb$name[i], gb$t[i], gb$l[i], gb$b[i], gb$r[i],
                  ifelse(inherits(g,"zeroGrob"),"ZERO","REAL"), "\\n")
            }}
            """
        )
        rows = []
        for line in out.strip().splitlines():
            parts = line.split()
            if len(parts) == 6:
                rows.append(
                    (parts[0], int(parts[1]), int(parts[2]),
                     int(parts[3]), int(parts[4]), parts[5])
                )
        return sorted(rows)

    def _py_guide_box_layout(self, grob) -> list:
        rows = []
        lay = grob.layout
        for nm, t, l, b, r, gr in zip(
            lay["name"], lay["t"], lay["l"], lay["b"], lay["r"], grob.grobs
        ):
            if "guide-box" in nm:
                cls = type(gr).__name__
                kind = "ZERO" if cls in ("Grob", "ZeroGrob") else "REAL"
                rows.append((str(nm), int(t), int(l), int(b), int(r), kind))
        return sorted(rows)

    def test_two_legends_layout_matches(self):
        r_layout = self._r_guide_box_layout(
            "ggplot(df, aes(x,y,colour=g,shape=s)) + geom_point() + "
            "guides(colour=guide_legend(), shape=guide_legend())"
        )
        p = (
            ggplot(DF, aes("x", "y", colour="g", shape="s"))
            + geom_point()
            + guides(colour=guide_legend(), shape=guide_legend())
        )
        py_layout = self._py_guide_box_layout(ggplotGrob(p))
        assert py_layout == r_layout

    def test_suppressed_legend_layout_matches(self):
        r_layout = self._r_guide_box_layout(
            'ggplot(df, aes(x,y,colour=g)) + geom_point() + guides(colour="none")'
        )
        p = ggplot(DF, aes("x", "y", colour="g")) + geom_point() + guides(
            colour="none"
        )
        py_layout = self._py_guide_box_layout(ggplotGrob(p))
        assert py_layout == r_layout

    def test_colourbar_layout_matches(self):
        r_layout = self._r_guide_box_layout(
            "ggplot(df, aes(x,y,colour=v)) + geom_point() + "
            "guides(colour=guide_colourbar())"
        )
        p = ggplot(DF, aes("x", "y", colour="v")) + geom_point() + guides(
            colour=guide_colourbar()
        )
        py_layout = self._py_guide_box_layout(ggplotGrob(p))
        assert py_layout == r_layout
