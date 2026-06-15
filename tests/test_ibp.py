import importlib

import mathgr
from mathgr.ibp import Ibp, IbpCountPt2, IbpVar, IdHold, PdHold
from mathgr.tensor import DE, DN, DeclareSym, Dta, Pd, PdT, PdVars, Pm2, Simp, Symmetric, tensor
import sympy as sp


def test_ibp_var_moves_derivative_off_requested_variable():
    x, y = sp.symbols("x y")

    assert Ibp(y * Pd(x, DN("i")), Rank=IbpVar(x)) == -x * Pd(y, DN("i")) + PdHold(x * y, DN("i"))


def test_ibp_default_rank_simplifies_whole_expression_like_upstream_countleaf_example():
    x, y, xx, yy, zz, xxx, yyy = sp.symbols("x y xx yy zz xxx yyy")
    f = tensor("fIbpCountLeaf")

    expr = (
        y * Pd(x, DN("a"))
        + x * Pd(y, DN("a"))
        + f(DN("a"))
        + xx * yy * Pd(zz, DN("a"))
        + xx * zz * Pd(yy, DN("a"))
        + xxx * Pd(yyy, DN("a"))
    )

    assert Ibp(expr) == (
        f(DN("a"))
        - yy * zz * Pd(xx, DN("a"))
        + xxx * Pd(yyy, DN("a"))
        + PdHold(x * y, DN("a"))
        + PdHold(xx * yy * zz, DN("a"))
    )


def test_ibp_count_pt2_reduces_second_time_derivative():
    x, y = sp.symbols("x y")

    assert Ibp(y * Pd(Pd(x, DE(0)), DE(0)), Rank=IbpCountPt2) == (
        -Pd(x, DE(0)) * Pd(y, DE(0)) + PdHold(y * Pd(x, DE(0)), DE(0))
    )


def test_ibp_power_rule_matches_upstream_one_derivative_shortcut():
    x, y = sp.symbols("x y")

    assert Ibp(y * x**2 * Pd(x, DN("i"))) == (
        -x**3 * Pd(y, DN("i")) / 3 + PdHold(x**3 * y / 3, DN("i"))
    )


def test_ibp_rules_use_first_derivative_power_rule_like_upstream():
    a, x = sp.symbols("aFirstDerivativePower xFirstDerivativePower")
    i = DN("i")

    expected = Simp(PdHold(a * x * Pd(x, i), i) - x * Pd(a * Pd(x, i), i))

    assert mathgr.IbpRules(a * Pd(x, i) ** 2) == expected


def test_ibp_rules_use_reduced_derivative_power_rule_like_upstream():
    a, x = sp.symbols("aReducedDerivativePower xReducedDerivativePower")
    i = DN("i")
    j = DN("j")

    expr = a * PdT(x, PdVars(j)) ** 2 * PdT(x, PdVars(i, j))
    expected = Simp(PdHold(a * PdT(x, PdVars(j)) ** 3 / 3, i) - PdT(x, PdVars(j)) ** 3 * Pd(a, i) / 3)

    assert mathgr.IbpRules(expr) == expected


def test_ibp_rules_use_higher_derivative_power_rule_like_upstream():
    a, x = sp.symbols("aHigherDerivativePower xHigherDerivativePower")
    i = DN("i")
    j = DN("j")

    derivative = PdT(x, PdVars(i, j))
    reduced = PdT(x, PdVars(j))
    expr = a * derivative**3
    expected = Simp(PdHold(a * reduced * derivative**2, i) - reduced * Pd(a * derivative**2, i))

    assert mathgr.IbpRules(expr) == expected


def test_ibp_rules_use_two_first_derivative_product_rule_like_upstream():
    x, y = sp.symbols("x y")
    i = DN("i")
    j = DN("j")

    expected = PdHold(x * Pd(y, j), i) - PdHold(x * Pd(y, i), j) + Pd(x, j) * Pd(y, i)

    assert mathgr.IbpRules(Pd(x, i) * Pd(y, j)) == expected


def test_ibp_rules_use_squared_second_derivative_rule_like_upstream():
    g, x = sp.symbols("g x")
    i = DN("i")
    j = DN("j")

    expected = mathgr.Simp(
        PdHold(g * Pd(Pd(x, i), j) * Pd(x, j), i)
        - PdHold(g * Pd(Pd(x, i), i) * Pd(x, j), j)
        - Pd(g, i) * Pd(Pd(x, i), j) * Pd(x, j)
        + Pd(g, j) * Pd(Pd(x, i), i) * Pd(x, j)
        + g * Pd(Pd(x, i), i) * Pd(Pd(x, j), j)
    )

    assert mathgr.IbpRules(g * PdT(x, PdVars(i, j)) ** 2) == expected


def test_ibp_rules_use_symmetric_mixed_second_derivative_rule_like_upstream():
    z = sp.Symbol("zSymmetricMixedSecondDerivative")
    weight = tensor("weightSymmetricMixedSecondDerivative")
    i = DN("i")
    j = DN("j")
    k = DN("k")
    DeclareSym(weight, (DN, DN), Symmetric((1, 2)))

    expr = weight(i, j) * PdT(z, PdVars(i)) * PdT(z, PdVars(j, k))
    expected = Simp(
        PdHold(weight(i, j) * PdT(z, PdVars(i)) * PdT(z, PdVars(j)) / 2, k)
        - Pd(weight(i, j), k) * PdT(z, PdVars(i)) * PdT(z, PdVars(j)) / 2
    )

    assert mathgr.IbpRules(expr) == expected


def test_ibp_rules_use_symmetric_first_second_derivative_rule_like_upstream():
    z = sp.Symbol("zSymmetricFirstSecondDerivative")
    weight = tensor("weightSymmetricFirstSecondDerivative")
    i = DN("i")
    j = DN("j")
    k = DN("k")
    DeclareSym(weight, (DN, DN), Symmetric((1, 2)))

    expr = weight(i, j) * PdT(z, PdVars(k)) * PdT(z, PdVars(i, j))
    expected = Simp(
        PdHold(weight(i, j) * PdT(z, PdVars(k)) * PdT(z, PdVars(j)), i)
        - PdHold(weight(i, j) * PdT(z, PdVars(i)) * PdT(z, PdVars(j)) / 2, k)
        - Pd(weight(i, j), i) * PdT(z, PdVars(k)) * PdT(z, PdVars(j))
        + Pd(weight(i, j), k) * PdT(z, PdVars(i)) * PdT(z, PdVars(j)) / 2
    )

    assert mathgr.IbpRules(expr) == expected


def test_ibp_rules_use_cubic_second_derivative_shortcut_like_upstream():
    f, h = sp.symbols("fCubicSecondDerivative hCubicSecondDerivative")
    a = DN("a")
    b = DN("b")
    c = DN("c")
    g = h

    expr = f * PdT(h, PdVars(c)) * PdT(h, PdVars(a, b)) ** 2
    expected = Simp(
        PdHold(
            f * Pd(g, c) * Pd(g, b) * Pd(Pd(g, a), b)
            - f * Pd(g, b) ** 2 * Pd(Pd(g, a), c) / 2
            - Pd(f, a) * Pd(g, b) ** 2 * Pd(g, c) / 2,
            a,
        )
        - PdHold(f * Pd(g, c) * Pd(g, b) * Pd(Pd(g, a), a), b)
        + PdHold(f * Pd(g, b) ** 2 * Pd(Pd(g, a), a) / 2, c)
        + f * Pd(g, c) * Pd(Pd(g, a), a) * Pd(Pd(g, b), b)
        - Pd(f, c) * Pd(g, b) ** 2 * Pd(Pd(g, a), a) / 2
        + Pd(f, b) * Pd(g, c) * Pd(g, b) * Pd(Pd(g, a), a)
        + Pd(Pd(f, a), a) * Pd(g, c) * Pd(g, b) ** 2 / 2
        + Pd(f, a) * Pd(Pd(g, a), c) * Pd(g, b) ** 2
    )

    assert mathgr.IbpRules(expr) == expected


def test_ibp_uses_pm2_self_adjoint_rule_to_cancel_symmetric_terms():
    f, g = sp.symbols("f g")

    assert Ibp(f * Pm2(g, DN) - g * Pm2(f, DN)) == 0


def test_public_pm2rules_and_ibprules_can_be_used_as_try_simp_rules():
    f, g, x, y = sp.symbols("f g x y")
    ibp_expected = -x * Pd(y, DN("i")) + PdHold(x * y, DN("i"))

    assert mathgr.TrySimp(f * Pm2(g, DN) - g * Pm2(f, DN), mathgr.Pm2Rules) == 0
    assert mathgr.Ibp(
        y * Pd(x, DN("i")),
        Rule=mathgr.IbpRules,
        Rank=lambda expr: 0 if expr == ibp_expected else 1,
    ) == ibp_expected


def test_ibp_uses_pm2_laplacian_rule_and_cancels_cross_term():
    f, g = sp.symbols("f g")

    expr = Pm2(g * PdT(f, PdVars(DE(0), DN("i"), DN("i"))), DN) + 2 * Pm2(
        PdT(f, PdVars(DE(0), DN("i"))) * PdT(g, PdVars(DN("i"))), DN
    )

    assert Ibp(expr) == (
        g * PdT(f, PdVars(DE(0)))
        - Pm2(PdT(f, PdVars(DE(0))) * PdT(g, PdVars(DN("a"), DN("a"))), DN)
    )


def test_try_simp_accepts_only_rank_improving_recursive_rule_candidates():
    x, y = sp.symbols("x y")

    def rule(expr):
        if expr == x + y:
            return x
        if expr == x:
            return x + y
        return expr

    assert mathgr.TrySimp((x + y) ** 2, rule) == x**2
    assert mathgr.TrySimp(x, rule) == x


def test_try_simp_supports_custom_rank_function():
    x, y = sp.symbols("x y")

    def rule(expr):
        if expr == x:
            return x + y
        return expr

    assert mathgr.TrySimp(x, rule, Rank=lambda expr: 0 if expr == x + y else 1) == x + y


def test_try_simp_accepts_sympy_wild_index_rule_dictionaries_like_upstream_patterns():
    f = tensor("fTrySimpPattern")
    g = tensor("gTrySimpPattern")
    i = sp.Wild("i")
    expr = f(DN("a"))
    expected = g(DN("a"))

    assert mathgr.TrySimp(expr, {f(DN(i)): g(DN(i))}, Rank=lambda candidate: 0 if candidate == expected else 1) == expected


def test_try_simp_accepts_single_rule_pair_like_mathematica_rule():
    x, y = sp.symbols("xTrySimpSingleRule yTrySimpSingleRule")

    assert mathgr.TrySimp(x + 1, (x, y), Rank=lambda candidate: 0 if candidate == y + 1 else 1) == y + 1


def test_try_simp_accepts_callable_wild_replacements_like_upstream_rule_delayed():
    f = tensor("fTrySimpDelayedPattern")
    g = tensor("gTrySimpDelayedPattern")
    i = sp.Wild("i")
    expr = f(DN("a"))
    expected = g(DN("a_delayed"))

    assert (
        mathgr.TrySimp(
            expr,
            [(f(DN(i)), lambda i: g(DN(f"{i}_delayed")))],
            Rank=lambda candidate: 0 if candidate == expected else 1,
        )
        == expected
    )


def test_try_simp_simplifies_context_after_callable_wild_replacement_like_upstream():
    from mathgr.decomp import D1, U2

    vector = tensor("vectorTrySimpContext")
    field_strength = tensor("fieldStrengthTrySimpContext")
    weight = sp.Symbol("weightTrySimpContext")
    m = sp.Wild("m")
    α = sp.Wild("α")
    β = sp.Wild("β")
    expr = weight * Pd(vector(U2("i"), D1("a")), D1("b")) - weight * Pd(vector(U2("i"), D1("b")), D1("a"))
    rule = [
        (
            Pd(vector(U2(m), D1(α)), D1(β)),
            lambda m, α, β: field_strength(U2(m), D1(α), D1(β))
            + Pd(vector(U2(m), D1(β)), D1(α)),
        )
    ]

    assert mathgr.TrySimp(expr, rule) == weight * field_strength(U2("i"), D1("a"), D1("b"))


def test_try_simp_simplifies_large_replacement_context_without_expanding_unrelated_products():
    from mathgr.decomp import D1, U2

    vector = tensor("vectorTrySimpLargeContext")
    field_strength = tensor("fieldStrengthTrySimpLargeContext")
    weight = sp.Symbol("weightTrySimpLargeContext")
    m = sp.Wild("m")
    α = sp.Wild("α")
    β = sp.Wild("β")
    noise = sum((sp.Symbol(f"trySimpNoiseX{n}") + 1) * (sp.Symbol(f"trySimpNoiseY{n}") + 2) for n in range(80))
    expr = (
        noise
        + weight * Pd(vector(U2("i"), D1("a")), D1("b"))
        - weight * Pd(vector(U2("i"), D1("b")), D1("a"))
    )
    rule = [
        (
            Pd(vector(U2(m), D1(α)), D1(β)),
            lambda m, α, β: field_strength(U2(m), D1(α), D1(β))
            + Pd(vector(U2(m), D1(β)), D1(α)),
        )
    ]

    assert mathgr.TrySimp(expr, rule) == noise + weight * field_strength(U2("i"), D1("a"), D1("b"))


def test_try_simp_level_two_can_take_temporary_rank_regression():
    x, y, z = sp.symbols("x y z")
    ranks = {x: 1, y: 2, z: 0}

    def rule(expr):
        if expr == x:
            return y
        if expr == y:
            return z
        return expr

    rank = lambda expr: ranks.get(expr, 99)

    assert mathgr.TrySimp(x, rule, Rank=rank) == x
    assert mathgr.TrySimp(x, rule, Rank=rank, Level=2) == z
    assert mathgr.TrySimp2(x, rule, Rank=rank) == z


def test_try_simp_preferred_pattern_biases_equal_rank_candidates_like_upstream():
    ibp_module = importlib.import_module("mathgr.ibp")
    x, y, z = sp.symbols("x y z")
    previous = tuple(ibp_module.TrySimpPreferredPattern)
    previous_strength = ibp_module.TrySimpPreferredPatternStrength
    ibp_module.TrySimpPreferredPattern[:] = [z]
    ibp_module.TrySimpPreferredPatternStrength = sp.Integer(1)

    def rule(expr):
        if expr == x:
            return [y, z]
        return expr

    try:
        assert mathgr.TrySimp(x, rule, Rank=lambda _expr: 1) == z
        assert mathgr.TrySimpPreferredPattern is ibp_module.TrySimpPreferredPattern
    finally:
        ibp_module.TrySimpPreferredPattern[:] = previous
        ibp_module.TrySimpPreferredPatternStrength = previous_strength


def test_try_simp_preferred_pattern_accepts_sympy_wild_tensor_patterns_like_upstream():
    ibp_module = importlib.import_module("mathgr.ibp")
    f = tensor("fTrySimpPreferredPattern")
    g = tensor("gTrySimpPreferredPattern")
    x = sp.Symbol("xTrySimpPreferredPattern")
    i = sp.Wild("i")
    previous = tuple(ibp_module.TrySimpPreferredPattern)
    previous_strength = ibp_module.TrySimpPreferredPatternStrength
    ibp_module.TrySimpPreferredPattern[:] = [g(DN(i))]
    ibp_module.TrySimpPreferredPatternStrength = sp.Integer(1)

    def rule(expr):
        if expr == x:
            return [f(DN("a")), g(DN("a"))]
        return expr

    try:
        assert mathgr.TrySimp(x, rule, Rank=lambda _expr: 1) == g(DN("a"))
    finally:
        ibp_module.TrySimpPreferredPattern[:] = previous
        ibp_module.TrySimpPreferredPatternStrength = previous_strength


def test_ibp_rule_option_uses_ranked_try_simp_like_upstream():
    x, y = sp.symbols("x y")

    def rule(expr):
        if expr == x + y:
            return x
        return expr

    assert Ibp(x + y, Rule=rule, Rank=lambda expr: 0 if expr == x else 1) == x


def test_ibp_nb_drops_total_derivative_boundary_terms():
    x, y = sp.symbols("x y")

    assert mathgr.IbpNB(y * Pd(x, DN("i"))) == -x * Pd(y, DN("i"))


def test_id_hold_is_public_and_ignored_by_ibp_rank_helpers_like_upstream():
    x, y = sp.symbols("x y")

    assert mathgr.IdHold(0) == 0
    assert mathgr.IbpCountLeaf(mathgr.IdHold(Pd(y, DN("i"))) + x) == mathgr.IbpCountLeaf(x)


def test_pdhold_algebra_matches_upstream_boundary_term_rules():
    x, y = sp.symbols("x y")
    f = tensor("fPdHold")
    z = sp.Symbol("zPdHold")

    assert -PdHold(x, DN("i")) == PdHold(-x, DN("i"))
    assert -z * PdHold(x, DN("i")) == z * PdHold(-x, DN("i"))
    assert 2 * PdHold(x, DN("i")) == PdHold(2 * x, DN("i"))
    assert PdHold(x, DN("i")) + PdHold(y, DN("i")) == PdHold(x + y, DN("i"))
    assert PdHold(f(DN("i")), DN("i")) == PdHold(f(DN("(PdId)")), DN("(PdId)"))


def test_simp_enters_pdhold_and_idhold_like_upstream_boundary_rules():
    g = tensor("gHoldSimp")

    assert Simp(PdHold(g(DN("a")) * Dta(DN("a"), DN("c")), DN("i"))) == PdHold(g(DN("c")), DN("i"))
    assert Simp(IdHold(g(DN("a")) * Dta(DN("a"), DN("c")))) == IdHold(g(DN("c")))


def test_ibp_variation_eliminates_derivatives_on_target_without_boundary_terms():
    x, y = sp.symbols("x y")

    assert mathgr.IbpVariation(y * Pd(x, DN("i")), x) == -x * Pd(y, DN("i"))
    assert mathgr.IbpVariation(y * Pd(Pd(x, DN("i")), DN("j")), x) == x * Pd(Pd(y, DN("i")), DN("j"))


def test_ibp_rank_helpers_count_relevant_derivative_shapes():
    x, y = sp.symbols("x y")

    assert mathgr.IbpCountPt2(Pd(Pd(x, DE(0)), DE(0))) > mathgr.IbpCountPt2(Pd(x, DE(0)))
    assert mathgr.IbpCountPd2(Pd(Pd(x, DN("i")), DN("j"))) > mathgr.IbpCountPd2(Pd(x, DN("i")))
    assert mathgr.IbpCountTerm(x + y) > mathgr.IbpCountTerm(x)
    assert mathgr.IbpStd2(Pd(Pd(x, DE(0)), DE(0))) > mathgr.IbpStd2(Pd(x, DE(0)))


def test_ibp_var_is_callable_rank_helper_like_upstream():
    x, y = sp.symbols("x y")
    rank = mathgr.IbpVar(x)

    assert rank(Pd(Pd(x, DN("i")), DN("j"))) > rank(Pd(x, DN("i")))
    assert rank(Pd(y, DN("i"))) < rank(Pd(x, DN("i")))


def test_ibp_reduce_order_exposes_upstream_power_aware_rank_helper():
    x, y = sp.symbols("x y")

    rank = mathgr.IbpReduceOrder([x])

    assert rank(x**2) < rank(x)
    assert rank(x + y) > rank(x)


def test_ibp_rule_with_forbidden_pattern_exposes_upstream_rank_wrapper():
    x = sp.Symbol("x")
    forbidden = PdHold(x, DN("i"))
    rank = mathgr.IbpRuleWithForbiddenPatterrn(lambda expr: sp.Integer(7), forbidden)

    assert rank(x) == 7
    assert rank(forbidden) == sp.oo
    assert rank(1 + forbidden) == sp.oo


def test_ibp_rule_with_forbidden_pattern_accepts_sympy_wild_tensor_patterns_like_upstream():
    f = tensor("fForbiddenPattern")
    i = sp.Wild("i")
    forbidden = f(DN(i))
    rank = mathgr.IbpRuleWithForbiddenPatterrn(lambda expr: sp.Integer(7), forbidden)

    assert rank(f(DN("a"))) == sp.oo
    assert rank(1 + f(DN("b"))) == sp.oo


def test_pm2_simp_exposes_pm2_rule_simplification_without_manual_ibp_call():
    f, g = sp.symbols("f g")

    assert mathgr.Pm2Simp(f * Pm2(g, DN) - g * Pm2(f, DN)) == 0


def test_pm2_simp_moves_laplacian_out_of_partial_derivative_like_upstream_pm2rules():
    a, b = sp.symbols("a b")

    assert mathgr.Pm2Simp(a * PdT(Pm2(b, DN), PdVars(DE(0)))) == Pm2(a, DN) * PdT(b, PdVars(DE(0)))


def test_pm2_simp_uses_square_laplacian_rule_like_upstream_pm2rules():
    a, f = sp.symbols("a f")

    expected = (
        a * f**2
        - Pm2(
            Pd(Pd(a, DN("a")), DN("a")) * f**2
            + 4 * Pd(a, DN("a")) * f * Pd(f, DN("a"))
            + 2 * a * Pd(f, DN("a")) ** 2,
            DN,
        )
    ) / 2

    assert mathgr.Pm2Simp(Pm2(a * f * PdT(f, PdVars(DN("i"), DN("i"))), DN)) == expected
