import sympy as sp

import mathgr
from mathgr.tensor import DN, Pd, PdT, PdVars, Simp, declare_idx, tensor
from mathgr import util
from mathgr.util import Eps, LocalToK, MomentumLabel, OO, SS, TPower, TSeries, k


def test_tpower_preserves_raw_power_labels_like_upstream():
    f = tensor("f")
    g = tensor("g")

    assert TPower(f(DN("a")), 2) == f(DN("a")) ** 2
    assert TPower(f(DN("a")), -2) == f(DN("a")) ** -2
    assert TPower(f(DN("a")) * g(DN("a")), 2) == f(DN("a")) ** 2 * g(DN("a")) ** 2


def test_tpower_negative_dummy_contraction_matches_upstream_unrenamed_denominator():
    f = tensor("fTPowerNegative")
    g = tensor("gTPowerNegative")

    assert TPower(f(DN("a")) * g(DN("a")), -2) == 1 / (f(DN("a")) ** 2 * g(DN("a")) ** 2)


def test_ss_and_oo_expand_in_default_perturbation_parameter():
    x, y, z = sp.symbols("x y z")
    expr = (1 + Eps * x) * (1 + Eps * y) * (1 + Eps**2 * z)

    assert sp.expand(SS(2)(expr) - (Eps**2 * (x * y + z) + Eps * (x + y) + 1)) == 0
    assert OO(1)(expr) == x + y
    assert OO(2)(expr) == x * y + z


def test_eps_is_treated_as_partial_derivative_constant_like_upstream_util():
    assert Pd(Eps, DN("i")) == 0


def test_collect_eps_collects_perturbation_then_requested_symbols():
    x, y = sp.symbols("x y")

    assert hasattr(util, "CollectEps")
    assert util.CollectEps([x])(Eps * x * y + Eps * x**2 + y) == Eps * x * (x + y) + y
    assert mathgr.CollectEps is util.CollectEps


def test_tseries_preserves_free_index_and_raw_dummy_contraction_powers():
    f = tensor("fTSeries")
    g = tensor("gTSeries")

    assert TSeries((1 + Eps) * f(DN("a")) ** 2, (Eps, 0, 1)) == (
        Eps * f(DN("a")) ** 2 + f(DN("a")) ** 2
    )
    assert SS(1)((1 + Eps) * f(DN("a")) ** 2) == (
        Eps * f(DN("a")) ** 2 + f(DN("a")) ** 2
    )
    assert TSeries((1 + Eps) * (f(DN("a")) * g(DN("a"))) ** 2, (Eps, 0, 1)) == (
        Eps * f(DN("a")) ** 2 * g(DN("a")) ** 2
        + f(DN("a")) ** 2 * g(DN("a")) ** 2
    )


def test_tseries_preserves_raw_dummy_power_labels_like_upstream_protected_product():
    f = tensor("fTSeriesRawPower")
    g = tensor("gTSeriesRawPower")

    assert TSeries((1 + Eps) ** 2 * (f(DN("a")) * g(DN("a"))) ** 3, (Eps, 0, 1)) == (
        2 * Eps * f(DN("a")) ** 3 * g(DN("a")) ** 3
        + f(DN("a")) ** 3 * g(DN("a")) ** 3
    )


def test_tseries_expands_partial_derivative_bases_and_treats_series_symbol_as_constant():
    x = sp.Symbol("x")

    assert TSeries(PdT((1 + Eps * x) ** 2, PdVars(DN("i"))), (Eps, 0, 2)) == (
        2 * Eps * Pd(x, DN("i")) + 2 * Eps**2 * x * Pd(x, DN("i"))
    )


def test_ss_and_oo_preserve_explicit_integer_derivative_indices():
    x = sp.Symbol("xExplicitSeriesIndex")
    expr = (1 + Eps) * Pd(x, mathgr.DE(0))

    assert SS(1)(expr).has(mathgr.DE(0))
    assert not SS(1)(expr).has(mathgr.DE("0"))
    assert OO(1)(expr) == Pd(x, mathgr.DE(0))


def test_tseries_preserves_registered_indices_inside_indexed_tensor_sums():
    f = tensor("fTSeriesIndexedSum")
    g = tensor("gTSeriesIndexedSum")

    assert TSeries((f(DN("a")) + Eps * g(DN("a"))) ** 2, (Eps, 0, 1)) == (
        f(DN("a")) ** 2 + 2 * Eps * f(DN("a")) * g(DN("a"))
    )


def test_tseries_matches_upstream_simp_through_seriesdata_cell():
    xx, yy, zz = sp.symbols("xx yy zz")
    ff = tensor("ffTSeriesUpstreamSeriesData")

    series = TSeries((1 + Eps * xx) * (1 + Eps * yy) * (1 + Eps * zz) * ff(DN("a")), (Eps, 0, 1))

    assert Simp(series).xreplace({Eps: 1}) == sp.expand((1 + xx + yy + zz) * ff(DN("a")))


def test_local_to_k_matches_upstream_single_term_example():
    x, y = sp.symbols("x y")

    assert LocalToK(2 * x * PdT(y, PdVars(DN("a"), DN("b")))) == (
        2 * k(2)(DN("a")) * k(2)(DN("b")) * tensor("x")(k(1)) * tensor("y")(k(2))
    )


def test_local_to_k_assigns_fresh_momentum_to_each_repeated_scalar_field_occurrence():
    x = sp.Symbol("xLocalToKRepeated")

    assert LocalToK(x * PdT(x, PdVars(DN("i")))) == (
        k(2)(DN("i")) * tensor("xLocalToKRepeated")(k(1)) * tensor("xLocalToKRepeated")(k(2))
    )


def test_local_to_k_assigns_momentum_to_indexed_tensor_fields_like_upstream():
    f = tensor("fLocalToKIndexed")

    assert LocalToK(PdT(f(DN("i")), PdVars(DN("a")))) == k(1)(DN("a")) * f(k(1), DN("i"))


def test_local_to_k_assigns_momentum_to_undifferentiated_indexed_tensor_fields_like_upstream():
    f = tensor("fLocalToKUndifferentiatedIndexed")

    assert LocalToK(f(DN("i"))) == f(k(1), DN("i"))


def test_local_to_k_accepts_custom_index_family_and_momentum_head_like_upstream_option():
    _up, down = declare_idx("localToKCustomU", "localToKCustomD", dim=3)
    q = tensor("qLocalToK")
    x = sp.Symbol("xLocalToK")
    q1 = MomentumLabel("qLocalToK", 1)

    assert LocalToK(PdT(x, PdVars(down("a"))), down, Momentum=q) == (
        q(1, down("a")) * tensor("xLocalToK")(q1)
    )


def test_solve_expr_solves_for_compound_expressions_like_mathematica_wrapper():
    x, y, z = sp.symbols("x y z")

    assert util.SolveExpr([sp.Eq(2 * (x + y) + z, 7)], [x + y]) == [{x + y: (7 - z) / 2}]
    assert util.SolveExpr([sp.Eq(x + y, 3), sp.Eq(x - y, 1)], [x + y, x - y]) == [
        {x + y: sp.Integer(3), x - y: sp.Integer(1)}
    ]


def test_treplace_replaces_tensor_factors_and_preserves_raw_dummy_power_labels():
    x, y = sp.symbols("x y")
    f = tensor("f")
    g = tensor("g")

    expr = f(DN("a")) ** 2 + y
    expected = g(DN("a")) ** 2 + y

    assert util.TReplace(expr, {f(DN("a")): g(DN("a"))}) == expected
    assert util.TReplace({f(DN("a")): g(DN("a"))})(expr) == expected
    assert util.TReplace(x + y, {x: 2 * y}) == 3 * y


def test_treplace_supports_sympy_wild_index_patterns_like_mathematica_rules():
    f = tensor("fTReplacePattern")
    g = tensor("gTReplacePattern")
    i = sp.Wild("i")

    expr = f(DN("a")) + 2 * f(DN("b"))

    assert util.TReplace(expr, {f(DN(i)): g(DN(i))}) == g(DN("a")) + 2 * g(DN("b"))


def test_treplace_supports_callable_wild_replacements_like_mathematica_rule_delayed():
    f = tensor("fTReplaceDelayedPattern")
    g = tensor("gTReplaceDelayedPattern")
    i = sp.Wild("i")

    expr = f(DN("a")) + f(DN("b"))

    assert util.TReplace(expr, [(f(DN(i)), lambda i: g(DN(f"{i}_delayed")))]) == g(
        DN("a_delayed")
    ) + g(DN("b_delayed"))


def test_treplace_accepts_single_rule_pair_like_mathematica_rule():
    x, y = sp.symbols("xSingleRule ySingleRule")

    assert util.TReplace(x + 1, (x, y)) == y + 1
    assert util.TReplace((x, y))(x + 1) == y + 1


def test_solve_expr_and_treplace_are_exported_from_package_root():
    assert mathgr.SolveExpr is util.SolveExpr
    assert mathgr.TReplace is util.TReplace
