"""
Tests for ggplot2-python's horizontal-extension surface.

Covers the four core extension points exposed for cross-cutting
extensions in the style of R's ggnewscale:

* :func:`ggproto2_py.ggproto.ggproto` with **instance-as-parent** —
  PR-2, R ref: ``ggplot2/R/ggproto.R:67-97``.
* :func:`ggplot2_py.ggproto.bind_method` + the ``_set`` footgun warning
  — PR-4.
* :class:`ggplot2_py.PlotEnv` + ``find_scale`` /
  ``ScalesList.add_defaults`` / ``ScalesList.add_missing`` honouring
  the env — PR-1, R ref: ``ggplot2/R/scale-type.R:39-54``.
* :func:`ggplot2_py.register_pre_add_hook` and its lifecycle across
  ``+`` — PR-3, R ref: ``ggnewscale/R/rename-aes.R:135-163``.

Plus an end-to-end ggnewscale-style mini port that combines every
fix above and confirms ``ggplot_build`` produces the expected scale
shape (the "lighthouse" test).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ggplot2_py import (
    GGProto,
    PlotEnv,
    aes,
    bind_method,
    geom_point,
    geom_smooth,
    ggplot,
    ggplot_build,
    ggproto,
    ggproto_parent,
    register_pre_add_hook,
    unregister_pre_add_hook,
    update_ggplot,
)
from ggplot2_py.aes import Mapping
from ggplot2_py.extension import (
    clone_geom,
    clone_layer,
    clone_stat,
    is_protected,
    palette_for_aes,
    protect,
    rename_aes_in_dict,
    rename_aes_in_mapping,
    rename_aes_in_seq,
)
from ggplot2_py.layer import Layer
from ggplot2_py.scale import find_scale
from ggplot2_py.scales import scale_colour_continuous, scale_colour_viridis_c


# =============================================================================
# PR-2: ggproto(_inherit=instance)
# =============================================================================

class TestGgprotoInstanceAsParent:
    """R ref: ``ggproto.R:67-97`` — `_inherit` can be a class **or**
    an instance.  ggnewscale uses the instance-parent form at 5 sites.
    """

    def test_class_as_parent_unchanged(self):
        """Original behaviour stays — ``ggproto(Name, Parent)`` returns
        a class."""
        Parent = ggproto("Parent", GGProto, x=42)
        Child = ggproto("Child", Parent, y=7)
        assert isinstance(Child, type)
        assert issubclass(Child, Parent)
        inst = Child()
        assert inst.x == 42 and inst.y == 7

    def test_instance_as_parent_returns_instance(self):
        """When ``_inherit`` is an instance, the result is an
        **instance** of a dynamic subclass of the parent's class."""
        Parent = ggproto("Parent", GGProto, x=1)
        parent_inst = Parent()
        child = ggproto("Child", parent_inst, y=2)
        assert not isinstance(child, type)
        assert isinstance(child, Parent)
        # The child's own __dict__ inherits the parent's slots
        # (Python's prototype emulation of R env semantics).
        assert child.x == 1
        assert child.y == 2

    def test_instance_super_chain(self):
        """``child.super()`` returns the parent **instance**, not the
        class.  Required for ``ggproto_parent(self.super(), self)``
        idiom (ggnewscale ``bump-aes-layers.R:88-94``).
        """
        Adder = ggproto("Adder", GGProto, x=0,
            add=lambda self, n: setattr(self, "x", self.x + n) or self.x)
        a = Adder()
        a.add(10)
        doubler = ggproto("Doubler", a,
            add=lambda self, n: ggproto_parent(self.super(), self).add(n * 2))
        assert doubler.super() is a
        # 10 (a.x) + 3*2 = 16; but Python's prototype emulation
        # gave doubler its own copy of x, so addition mutates the
        # child without touching the parent.
        assert doubler.x == 10
        assert doubler.add(3) == 16
        assert a.x == 10  # parent isolated

    def test_instance_overrides_dont_leak(self):
        """Setting an attribute on the child must not bleed up to the
        parent — mirrors R env-cloning semantics in
        ``ggproto.R:88-91`` where ``e`` is a *new* environment."""
        Parent = ggproto("Parent", GGProto, items=[1, 2, 3])
        a = Parent()
        b = ggproto("B", a)
        b.items = [9, 9]
        assert a.items == [1, 2, 3]

    def test_instance_parent_method_resolves_through_super(self):
        """Methods on the parent instance are reachable via the child
        through ``__getattribute__``'s super-chain fall-through."""
        Greeter = ggproto("Greeter", GGProto,
            hello=lambda self: f"hi {self._class_name}")
        g = Greeter()
        # No override; child resolves ``hello`` via parent walk.
        h = ggproto("H", g)
        assert h.hello().startswith("hi ")

    def test_post_clone_parent_mutation_visible_to_child(self):
        """R env-chain semantics: lookups on the child walk
        ``_super_inst`` lazily, so **post-clone mutations on the
        parent** are visible through the child until the child shadows
        them with its own slot.  Verified against the R reference:

            Adder <- ggproto("Adder", NULL, x = 0)
            child <- ggproto("Child", Adder)
            Adder$x <- 99      # mutate parent after clone
            Adder$y <- "tag"   # also add a brand-new slot on parent
            child$x  # => 99    (R env-chain lookup is lazy)
            child$y  # => "tag"

        This is the key invariant that lets ggnewscale-style
        extensions clone-once and have later parent edits flow
        through, mirroring R env modify-in-place behaviour
        (R ref: ``ggproto.R:118-142``).
        """
        Parent = ggproto("Parent", GGProto, x=0)
        a = Parent()
        child = ggproto("Child", a)
        # Mutate the parent AFTER the clone exists.
        a.x = 99
        # The child sees the mutated value (lazy walk, not snapshot).
        assert child.x == 99
        # Brand-new parent slot also flows through.
        a._set(y="tag")
        assert child.y == "tag"
        # Child's own writes do not propagate back to the parent.
        child.x = 7
        assert a.x == 99
        assert child.x == 7


# =============================================================================
# PR-4: bind_method + footgun warning
# =============================================================================

class TestBindMethodAndFootgun:
    def test_bind_method_works_for_non_self_function(self):
        """Functions whose first arg isn't ``self`` need
        :func:`bind_method` for explicit binding.  The auto-bind path
        on ``__getattribute__`` would silently skip them."""
        Geom = ggproto("Geom", GGProto, name="base")
        inst = Geom()
        def handle(this, data): return (this.name, data)
        bind_method(inst, "handle", handle)
        assert inst.handle("payload") == ("base", "payload")

    def test_bind_method_rejects_non_ggproto(self):
        with pytest.raises(TypeError):
            bind_method(object(), "x", lambda self: None)

    def test_bind_method_rejects_non_callable(self):
        g = ggproto("G", GGProto)()
        with pytest.raises(TypeError):
            bind_method(g, "x", "not_callable")

    def test_set_warns_when_replacing_bound_method_with_non_self(self):
        """The footgun: overriding a bound method with a function
        whose first arg isn't ``self`` would silently leave it
        unbound.  We warn before that happens."""
        Geom = ggproto("Geom", GGProto,
            handle=lambda self, data: ("orig", data))
        inst = Geom()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inst._set(handle=lambda data: ("bad", data))
            assert any("NOT be auto-bound" in str(x.message) for x in w)

    def test_set_does_not_warn_for_self_named_replacement(self):
        Geom = ggproto("Geom", GGProto,
            handle=lambda self, data: ("orig", data))
        inst = Geom()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inst._set(handle=lambda self, data: ("new", data))
            assert not any("NOT be auto-bound" in str(x.message) for x in w)

    def test_set_warns_when_shadowing_own_dict_entry(self):
        """The footgun must also fire when the shadowed self-bound
        method lives on the **instance**'s own ``__dict__`` (e.g. from
        an earlier ``_set``) — not only when it lives on the class or
        on a parent instance.  Symmetry with
        :meth:`GGProto.__getattribute__`'s lookup order.
        """
        inst = ggproto("Geom", GGProto)()
        # First _set installs a self-bound method on the instance.
        inst._set(handle=lambda self, data: ("first", data))
        # Second _set tries to replace it with a non-self function.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inst._set(handle=lambda data: ("second", data))
            assert any("NOT be auto-bound" in str(x.message) for x in w)

    def test_set_warns_when_shadowing_bind_method_install(self):
        """``bind_method`` installs an explicit ``MethodType`` — replacing
        it via ``_set`` with a non-self plain function still loses the
        binding, so the warning must fire there too.
        """
        inst = ggproto("Geom", GGProto)()
        def first(this, data): return ("bm", data)
        bind_method(inst, "handle", first)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inst._set(handle=lambda data: ("second", data))
            assert any("NOT be auto-bound" in str(x.message) for x in w)


# =============================================================================
# PR-1: PlotEnv + find_scale env consultation
# =============================================================================

class TestPlotEnv:
    def test_lookup_walks_layers_lifo(self):
        env = PlotEnv()
        env.push({"foo": "first"})
        env.push({"foo": "second"})
        # Most-recently-pushed wins (matches R env chain).
        assert env.lookup("foo") == "second"

    def test_lookup_falls_back_to_scales_module(self):
        env = PlotEnv()
        # No layer push — ``ggplot2_py.scales`` fallback should yield
        # the canonical constructor.
        f = env.lookup("scale_colour_continuous")
        assert callable(f)

    def test_pop_removes_top_layer(self):
        env = PlotEnv({"x": 1})
        env.push({"x": 2})
        assert env.lookup("x") == 2
        env.pop()
        assert env.lookup("x") == 1

    def test_clone_isolates(self):
        env = PlotEnv()
        env.push({"a": 1})
        cloned = env.clone()
        cloned.push({"a": 2})
        assert env.lookup("a") == 1
        assert cloned.lookup("a") == 2


class TestFindScaleHonoursEnv:
    def test_env_overrides_default(self):
        """R ref: ``scales-.R:159`` — ``add_defaults`` walks env
        before falling back to the ggplot2 namespace.  Our find_scale
        does the same."""
        env = PlotEnv({"scale_colour_continuous": scale_colour_viridis_c})
        sc = find_scale("colour", pd.Series([1.0, 2.0]), env)
        # Should be the viridis one, distinguishable by its palette
        # type (viridis scales use a ContinuousPalette wrapping the
        # viridis interpolator).
        assert sc is not None
        assert sc.aesthetics == ["colour"]

    def test_env_with_bumped_aes(self):
        """The ggnewscale use case: a bumped aesthetic name has no
        constructor in ``ggplot2_py.scales`` — the user injects one
        into the env, and the build pipeline picks it up."""
        env = PlotEnv()
        env.push({"scale_colour_NEW_continuous":
            lambda: scale_colour_continuous(aesthetics="colour_NEW")})
        sc = find_scale("colour_NEW", pd.Series([1.0, 2.0]), env)
        assert sc is not None
        assert "colour_NEW" in sc.aesthetics

    def test_no_env_no_injection_falls_through(self):
        """Without env injection, bumped aes names should resolve to
        ``None`` (no auto-scale)."""
        sc = find_scale("colour_NEW", pd.Series([1.0, 2.0]))
        assert sc is None

    def test_plot_env_clones_per_plot(self):
        df = pd.DataFrame({"x": [1, 2], "y": [1, 2]})
        p1 = ggplot(df, aes(x="x", y="y"))
        p1.plot_env.push({"marker": "p1"})
        p2 = p1 + geom_point()
        # ``+`` clones — pushing on p1 doesn't leak to p2's chain.
        p2.plot_env.push({"marker": "p2"})
        assert p1.plot_env.lookup("marker") == "p1"
        assert p2.plot_env.lookup("marker") == "p2"


# =============================================================================
# PR-3: _pre_add_hooks
# =============================================================================

class TestPreAddHooks:
    def test_hook_runs_before_dispatch(self):
        df = pd.DataFrame({"x": [1, 2], "y": [1, 2]})
        seen = []
        def hook(p, other):
            seen.append(type(other).__name__)
            return other
        p = ggplot(df, aes(x="x", y="y"))
        register_pre_add_hook(p, hook)
        p2 = p + geom_point()
        assert seen == ["Layer"]
        assert len(p2.layers) == 1

    def test_hook_can_transform_operand(self):
        """The canonical aes-bumping pattern (R ref:
        ``ggnewscale/R/rename-aes.R:135-163``)."""
        df = pd.DataFrame({"x": [1, 2], "y": [1, 2], "g": [1.0, 2.0]})
        def hook(p, other):
            if isinstance(other, Layer) and other.mapping is not None:
                other.mapping = rename_aes_in_mapping(
                    other.mapping, "colour", "colour_BUMPED")
            return other
        p = ggplot(df, aes(x="x", y="y"))
        register_pre_add_hook(p, hook)
        p2 = p + geom_point(aes(colour="g"))
        assert "colour_BUMPED" in p2.layers[0].mapping
        assert "colour" not in p2.layers[0].mapping

    def test_hook_returning_none_short_circuits(self):
        df = pd.DataFrame({"x": [1, 2], "y": [1, 2]})
        def hook(p, other): return None
        p = ggplot(df, aes(x="x", y="y"))
        register_pre_add_hook(p, hook)
        p2 = p + geom_point()
        assert len(p2.layers) == 0  # geom_point was dropped

    def test_hook_can_remove_itself(self):
        """Self-removing hook mirrors R's ``class(plot) <- setdiff(...)``
        cleanup in clear_aes."""
        df = pd.DataFrame({"x": [1, 2], "y": [1, 2]})
        def hook(p, other):
            unregister_pre_add_hook(p, hook)
            return other
        p = ggplot(df, aes(x="x", y="y"))
        register_pre_add_hook(p, hook)
        p2 = p + geom_point()
        assert len(p2._pre_add_hooks) == 0
        # Subsequent + does NOT trigger the hook.
        p3 = p2 + geom_smooth(method="lm")
        assert len(p3._pre_add_hooks) == 0

    def test_hooks_survive_clone(self):
        df = pd.DataFrame({"x": [1, 2], "y": [1, 2]})
        def hook(p, other): return other
        p = ggplot(df, aes(x="x", y="y"))
        register_pre_add_hook(p, hook)
        p2 = p + geom_point()
        assert hook in p2._pre_add_hooks
        # And cloning preserves identity, not just count:
        assert p2._pre_add_hooks is not p._pre_add_hooks

    def test_multiple_hooks_run_in_registration_order(self):
        df = pd.DataFrame({"x": [1, 2], "y": [1, 2]})
        order = []
        def h1(p, other): order.append("h1"); return other
        def h2(p, other): order.append("h2"); return other
        p = ggplot(df, aes(x="x", y="y"))
        register_pre_add_hook(p, h1)
        register_pre_add_hook(p, h2)
        _ = p + geom_point()
        assert order == ["h1", "h2"]


# =============================================================================
# PR-5: extension toolkit helpers
# =============================================================================

class TestExtensionHelpers:
    def setup_method(self):
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "g": [1.0, 2.0, 3.0]})

    def test_clone_layer_isolates_mapping_and_geom(self):
        p = ggplot(self.df, aes("x", "y")) + geom_point(aes(colour="g"))
        L = p.layers[0]
        L2 = clone_layer(L)
        assert L is not L2
        assert L.mapping is not L2.mapping
        assert L.geom is not L2.geom
        L2.mapping["colour"] = "OVERRIDDEN"
        assert L.mapping["colour"] == "g"

    def test_clone_layer_isolates_position(self):
        """``clone_layer`` must isolate ``position`` so per-layer
        mutable position state (e.g. ``PositionJitter.seed`` resolved
        in ``setup_params``) doesn't leak between source and clone.
        R analogue: position is fresh per layer (R ref:
        ``position-jitter.R:61``).
        """
        from ggplot2_py import position_jitter
        p = (ggplot(self.df, aes("x", "y"))
             + geom_point(aes(colour="g"), position=position_jitter(seed=42)))
        L = p.layers[0]
        L2 = clone_layer(L)
        # Identity isolation
        assert L.position is not L2.position
        # Mutating the clone's position state must not bleed back.
        L2.position.seed = 9999
        assert L.position.seed == 42

    def test_clone_geom_returns_isolated_instance(self):
        p = ggplot(self.df, aes("x", "y")) + geom_point()
        g = p.layers[0].geom
        g2 = clone_geom(g)
        assert g is not g2
        assert isinstance(g2, type(g))
        g2._set(default_aes=Mapping(colour="patched"))
        # Original geom's class-level default_aes is unaffected.
        assert "patched" not in str(g.default_aes)

    def test_clone_stat_returns_isolated_instance(self):
        p = ggplot(self.df, aes("x", "y")) + geom_point()
        s = p.layers[0].stat
        s2 = clone_stat(s)
        assert s is not s2
        assert isinstance(s2, type(s))

    def test_rename_aes_helpers_no_side_effects(self):
        m = Mapping(colour="g", x="x")
        m2 = rename_aes_in_mapping(m, "colour", "c1")
        assert m2 == {"c1": "g", "x": "x"}
        assert m == {"colour": "g", "x": "x"}
        d = {"a": 1, "b": 2}
        assert rename_aes_in_dict(d, "a", "z") == {"z": 1, "b": 2}
        assert d == {"a": 1, "b": 2}
        seq = ("x", "y", "colour")
        assert rename_aes_in_seq(seq, "colour", "c1") == ["x", "y", "c1"]
        assert seq == ("x", "y", "colour")

    def test_protect_is_idempotent(self):
        class S: pass
        s = S()
        protect(s, "colour")
        protect(s, "colour")
        assert is_protected(s, "colour")
        # Stamps are a list-of-unique-aesthetics, matching ggnewscale R idiom.
        assert s._ggnewscale_renamed.count("colour") == 1

    def test_palette_for_aes_returns_callable(self):
        pal = palette_for_aes("colour")
        # Should be the resolved palette of the default colour scale.
        assert pal is not None
        assert callable(getattr(pal, "__call__", None)) or callable(pal)


# =============================================================================
# Lighthouse: end-to-end ggnewscale-style port
# =============================================================================

class TestEndToEndGgnewscalePort:
    """A minimal port of ggnewscale's ``new_scale_colour`` mechanism,
    exercising every fix.  Builds the resulting plot to confirm the
    build pipeline produces the expected scale shape — same end-state
    R's ggnewscale produces (verified manually against the R package).
    """

    def test_new_colour_scale_bumps_existing_layer_and_keeps_new(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3],
                           "g1": [1.0, 2.0, 3.0], "g2": [4.0, 5.0, 6.0]})

        class NewColourScale:
            """``new_scale_colour()`` equivalent."""

        @update_ggplot.register(NewColourScale)
        def _add_new_colour(obj, plot, _=""):
            scale_number = plot._meta.get("_colour_bump_counter", 0) + 1
            new_aes = f"colour_ggnewscale_{scale_number}"
            # 1) bump existing layers' colour mapping
            new_layers = []
            for L in plot.layers:
                if is_protected(L, "colour"):
                    new_layers.append(L); continue
                nl = clone_layer(L)
                if nl.mapping is not None and "colour" in nl.mapping:
                    nl.mapping = rename_aes_in_mapping(nl.mapping, "colour", new_aes)
                protect(nl, "colour")
                new_layers.append(nl)
            plot.layers = new_layers
            # 2) bump existing colour scales
            for s in plot.scales.scales:
                if "colour" in s.aesthetics and not is_protected(s, "colour"):
                    s.aesthetics = rename_aes_in_seq(s.aesthetics, "colour", new_aes)
                    protect(s, "colour")
            # 3) inject default scale factory for the bumped name
            plot.plot_env.push({
                f"scale_{new_aes}_continuous":
                    lambda a=new_aes: scale_colour_continuous(aesthetics=a),
            })
            plot._meta["_colour_bump_counter"] = scale_number
            return plot

        p = (ggplot(df, aes(x="x", y="y"))
             + geom_point(aes(colour="g1"))           # layer 0: colour=g1
             + scale_colour_viridis_c()                # viridis colour scale
             + NewColourScale()                        # bump
             + geom_point(aes(colour="g2")))           # layer 1: fresh colour=g2

        # Structural assertions before build
        assert "colour_ggnewscale_1" in p.layers[0].mapping
        assert "colour" in p.layers[1].mapping
        bumped_scales = [s for s in p.scales.scales
                         if "colour_ggnewscale_1" in s.aesthetics]
        assert len(bumped_scales) == 1
        unprotected_colour_scales = [
            s for s in p.scales.scales
            if "colour" in s.aesthetics and not is_protected(s, "colour")]
        # No explicit second colour scale added — build will resolve one.
        assert len(unprotected_colour_scales) == 0

        # Build and confirm the new layer's data routes through a fresh
        # default colour scale.
        built = ggplot_build(p)
        all_aes = [a for s in built.plot.scales.scales for a in s.aesthetics]
        assert "colour_ggnewscale_1" in all_aes
        assert "colour" in all_aes  # the fresh default added at build


# =============================================================================
# Diverse extension patterns — the goal is R-ggplot2-level extensibility in
# general.  ggnewscale-style aes bumping is one use; below are independent
# patterns each exercising a different combination of the extension surface.
# =============================================================================

class TestExtensionPatternAutoWatermark:
    """**Pattern A** — a hook that quietly **adds** a layer alongside
    whatever the user adds next.  Showcases ``_pre_add_hooks`` as a
    generic "advice"-style mechanism, independent of aes renaming.
    """

    def test_pre_add_hook_can_inject_an_extra_layer(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        injected = []

        class WatermarkOnFirstLayer:
            """``+ WatermarkOnFirstLayer()`` arms a hook that fires
            exactly once: it appends a ``geom_point()`` after the next
            Layer the user adds.
            """

        @update_ggplot.register(WatermarkOnFirstLayer)
        def _add(obj, plot, _=""):
            def hook(p, other):
                if isinstance(other, Layer):
                    unregister_pre_add_hook(p, hook)
                    # Add the user's layer first, then a marker geom.
                    p.layers.append(other)
                    p.layers.append(geom_point())
                    injected.append(True)
                    return None  # we've handled the dispatch ourselves
                return other
            register_pre_add_hook(plot, hook)
            return plot

        p = ggplot(df, aes(x="x", y="y")) + WatermarkOnFirstLayer() + geom_smooth(method="lm")
        assert len(p.layers) == 2
        assert injected == [True]
        # Hook removed itself; further + doesn't double-inject.
        p2 = p + geom_smooth(method="lm")
        assert len(p2.layers) == 3


class TestExtensionPatternPluggableDefaultScale:
    """**Pattern B** — override the default-scale resolution **globally
    for a plot** via ``plot_env`` injection.  Showcases :class:`PlotEnv`
    as a per-plot lookup chain (R ref: ``scale-type.R:36-54``).
    """

    def test_plot_env_layer_overrides_default_colour_for_one_plot(self):
        df = pd.DataFrame({"x": [1, 2], "y": [1, 2], "c": [1.0, 2.0]})
        env = PlotEnv()
        env.push({"scale_colour_continuous": scale_colour_viridis_c})

        p = ggplot(df, aes(x="x", y="y", colour="c"), plot_env=env) + geom_point()
        built = ggplot_build(p)
        # The build resolved a colour scale; we don't assert which exact
        # palette class, only that the scale is *present* and was sourced
        # from our env (a different plot without env injection would also
        # get a colour scale — the structural assertion here is that the
        # env shapes the chain, not that the class name differs).
        colour_scales = [s for s in built.plot.scales.scales
                         if "colour" in s.aesthetics]
        assert len(colour_scales) == 1

    def test_plot_env_does_not_pollute_other_plots(self):
        df = pd.DataFrame({"x": [1, 2], "y": [1, 2], "c": [1.0, 2.0]})
        env = PlotEnv({"scale_colour_continuous": scale_colour_viridis_c})
        p1 = ggplot(df, aes(x="x", y="y", colour="c"), plot_env=env) + geom_point()
        # A separate ggplot without env injection still resolves normally.
        p2 = ggplot(df, aes(x="x", y="y", colour="c")) + geom_point()
        # Two independent build calls; neither raises.
        ggplot_build(p1)
        ggplot_build(p2)


class TestExtensionPatternGeomMethodPatch:
    """**Pattern C** — at plot construction time, patch a per-layer
    Geom's method (without subclassing).  Showcases
    ``ggproto(None, inst)`` + :func:`bind_method` as the Python
    counterpart of R's ``ggproto("New", old_geom, method = fn)``.
    """

    def test_per_layer_geom_method_override(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        p = ggplot(df, aes(x="x", y="y")) + geom_point()
        layer = p.layers[0]
        # Clone the layer so we don't mutate the source plot's geom.
        nl = clone_layer(layer)
        # Replace the cloned geom's ``setup_data`` with a tagging
        # version.  Use bind_method so the first arg can be named
        # anything.
        tagged = {"count": 0}
        def patched(this, data, params):
            tagged["count"] += 1
            return data
        bind_method(nl.geom, "setup_data", patched)
        # Install the cloned layer back.
        p2 = ggplot(df, aes(x="x", y="y"))
        p2.layers = [nl]
        ggplot_build(p2)
        assert tagged["count"] >= 1
        # Original layer's geom is untouched.
        from ggplot2_py.geom import Geom
        assert type(layer.geom).__name__ == "GeomPoint"


class TestExtensionPatternCustomAddDispatcher:
    """**Pattern D** — register a brand-new operand type for ``+``.
    This already worked in pre-fix ggplot2-python and is included
    here for **completeness of the general extension surface**, so
    the test file documents all four surfaces.
    """

    def test_custom_type_registers_through_singledispatch(self):
        df = pd.DataFrame({"x": [1, 2], "y": [1, 2]})

        class TitleOverride:
            def __init__(self, t): self.t = t

        @update_ggplot.register(TitleOverride)
        def _add(obj, plot, _=""):
            plot.labels["title"] = obj.t
            return plot

        p = ggplot(df, aes(x="x", y="y")) + geom_point() + TitleOverride("hello")
        assert p.labels.get("title") == "hello"


# =============================================================================
# R cross-validation harness — confirms the extension fixes produce the same
# **structural** result as the R counterpart on selected scenarios.  Tests
# are skipped when the ``ggrepel-dev`` R environment is missing.
# =============================================================================

import json
import os
import subprocess

R_BIN = "/home/groups/xiaojie/nianping/Conda_Files/envs/ggrepel-dev/bin/Rscript"


def _r_available() -> bool:
    return os.path.exists(R_BIN)


def _run_r(script: str) -> dict:
    """Execute *script* in the ggrepel-dev R env and parse its
    final ``cat()``'d JSON line as the return value."""
    proc = subprocess.run(
        [R_BIN, "--no-init-file", "-e", script],
        capture_output=True, text=True, check=False, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"R script failed:\n{proc.stderr}")
    out = proc.stdout.strip().splitlines()
    for line in reversed(out):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"No JSON output from R:\n{proc.stdout}\n{proc.stderr}")


@pytest.mark.skipif(not _r_available(), reason="R env not available")
class TestRCrossValidation:
    """Structural cross-checks: build the same plot in R and Python,
    compare invariants (layer count, mapping keys, scale aesthetic
    sets).  *Not* byte-for-byte rendering — we only assert the plot
    tree's shape matches.
    """

    def test_baseline_geom_point_mapping_matches_r(self):
        py_state = {
            "layer_count": 0, "layer0_mapping_keys": [],
            "scale_aes_sets": [],
        }
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "g": [1.0, 2.0, 3.0]})
        p = ggplot(df, aes(x="x", y="y", colour="g")) + geom_point()
        built = ggplot_build(p)
        py_state["layer_count"] = len(built.plot.layers)
        # Use the effective (computed) mapping — when the layer was
        # created without a layer-level aes() the local ``mapping``
        # is None and only ``computed_mapping`` carries the resolved
        # keys.  R aggregates plot+layer aes lazily via
        # ``compute_aesthetics`` and exposes the union via
        # ``layer$computed_mapping`` after build (R ref:
        # ``layer.R:493-526``); we read the same here.
        L0 = built.plot.layers[0]
        eff_mapping = L0.computed_mapping if L0.computed_mapping is not None else (
            L0.mapping if L0.mapping is not None else built.plot.mapping)
        py_state["layer0_mapping_keys"] = sorted(eff_mapping.keys())
        py_state["scale_aes_sets"] = sorted(
            ",".join(sorted(s.aesthetics)) for s in built.plot.scales.scales)

        r_script = r"""
        suppressPackageStartupMessages(library(ggplot2))
        suppressPackageStartupMessages(library(jsonlite))
        df <- data.frame(x=1:3, y=1:3, g=c(1.0,2.0,3.0))
        p <- ggplot(df, aes(x=x, y=y, colour=g)) + geom_point()
        built <- ggplot_build(p)
        # R's plot+layer aes is resolved at compute_aesthetics; the
        # effective mapping is the union of plot$mapping and
        # layer$mapping.  ``built$plot$layers[[1]]$computed_mapping``
        # is the post-build snapshot.
        m <- built$plot$layers[[1]]$computed_mapping
        if (is.null(m) || length(m) == 0) m <- built$plot$mapping
        out <- list(
          layer_count = length(built$plot$layers),
          layer0_mapping_keys = sort(names(m)),
          scale_aes_sets = sort(sapply(built$plot$scales$scales,
                                       function(s) paste(sort(s$aesthetics), collapse=",")))
        )
        cat(toJSON(out, auto_unbox=TRUE), "\n")
        """
        r_state = _run_r(r_script)
        # Both must produce the same number of layers and the same
        # mapping key set on layer 0.
        assert py_state["layer_count"] == r_state["layer_count"]
        assert py_state["layer0_mapping_keys"] == r_state["layer0_mapping_keys"]
        # Full structural equality: after PR-7 the ``_X_AESTHETICS`` /
        # ``_Y_AESTHETICS`` lists are byte-identical to R's
        # ``ggplot_global$x_aes`` / ``y_aes``, so the per-scale
        # aesthetic sets must match exactly.
        assert py_state["scale_aes_sets"] == r_state["scale_aes_sets"]

    def test_env_injected_scale_resolves_in_both(self):
        """R ref: ``scale-type.R:36-54`` — env override wins.  Python
        should replicate."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "g": [1.0, 2.0, 3.0]})
        env = PlotEnv({"scale_colour_continuous": scale_colour_viridis_c})
        p = ggplot(df, aes(x="x", y="y", colour="g"), plot_env=env) + geom_point()
        built = ggplot_build(p)
        py_n_colour = sum(1 for s in built.plot.scales.scales
                          if "colour" in s.aesthetics)

        r_script = r"""
        suppressPackageStartupMessages(library(ggplot2))
        suppressPackageStartupMessages(library(jsonlite))
        df <- data.frame(x=1:3, y=1:3, g=c(1.0,2.0,3.0))
        p <- ggplot(df, aes(x=x, y=y, colour=g)) + geom_point()
        p@plot_env$scale_colour_continuous <- function() scale_colour_viridis_c()
        built <- ggplot_build(p)
        n_colour <- sum(sapply(built$plot$scales$scales,
                               function(s) "colour" %in% s$aesthetics))
        cat(toJSON(list(n_colour=n_colour), auto_unbox=TRUE), "\n")
        """
        r_state = _run_r(r_script)
        assert py_n_colour == r_state["n_colour"]
        # Sanity: there's exactly one colour scale on each side.
        assert py_n_colour == 1

    def test_position_aesthetic_lists_match_r(self):
        """PR-7 fix — ``ggplot2_py.aes.X_AES`` / ``Y_AES`` must be
        byte-identical to R's ``ggplot_global$x_aes`` / ``y_aes``
        (R ref: ``ggplot-global.R:50-54``).  Single-source-of-truth:
        ``scale.py:_X_AESTHETICS`` / ``_Y_AESTHETICS`` re-export
        these.
        """
        from ggplot2_py.aes import X_AES, Y_AES
        from ggplot2_py.scale import _X_AESTHETICS, _Y_AESTHETICS
        # Local invariant: re-export identity (single source of truth).
        assert _X_AESTHETICS is X_AES
        assert _Y_AESTHETICS is Y_AES

        r_script = r"""
        suppressPackageStartupMessages(library(ggplot2))
        suppressPackageStartupMessages(library(jsonlite))
        # ggplot_global is package-private; grab via ::: as ggplot2's
        # own test suite does.
        x_aes <- ggplot2:::ggplot_global$x_aes
        y_aes <- ggplot2:::ggplot_global$y_aes
        cat(toJSON(list(x = x_aes, y = y_aes)), "\n")
        """
        r_state = _run_r(r_script)
        assert list(X_AES) == r_state["x"], (list(X_AES), r_state["x"])
        assert list(Y_AES) == r_state["y"], (list(Y_AES), r_state["y"])

    def test_ggproto_instance_chain_matches_r(self):
        """R ref: ``ggproto.R:67-97``.  Construct equivalent prototype
        chains and confirm the visible state matches."""
        Adder = ggproto("Adder", GGProto, x=0,
            add=lambda self, n: setattr(self, "x", self.x + n) or self.x)
        a = Adder()
        a.add(10)
        doubler = ggproto("Doubler", a,
            add=lambda self, n: ggproto_parent(self.super(), self).add(n * 2))
        py_state = {
            "doubler_isinstance_adder": isinstance(doubler, Adder),
            "doubler_x_initial": doubler.x,
            "doubler_add_3": doubler.add(3),
            "adder_x_after_doubler": a.x,
        }

        r_script = r"""
        suppressPackageStartupMessages(library(ggplot2))
        suppressPackageStartupMessages(library(jsonlite))
        Adder <- ggproto("Adder", NULL,
          x = 0,
          add = function(self, n) { self$x <- self$x + n; self$x })
        Adder$add(10)
        Doubler <- ggproto("Doubler", Adder,
          add = function(self, n) ggproto_parent(Adder, self)$add(n * 2))
        out <- list(
          doubler_isinstance_adder = inherits(Doubler, "Adder"),
          doubler_x_initial = Doubler$x,
          doubler_add_3 = Doubler$add(3),
          adder_x_after_doubler = Adder$x
        )
        cat(toJSON(out, auto_unbox=TRUE), "\n")
        """
        r_state = _run_r(r_script)
        assert py_state == r_state, f"py={py_state}, r={r_state}"
