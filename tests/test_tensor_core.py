import importlib

import sympy as sp

import mathgr
from mathgr.decomp import D1, D2, DTot, U1, U2, UTot
from mathgr.gr import UseMetric

tensor_module = importlib.import_module("mathgr.tensor")

from mathgr.tensor import (
    DE,
    DN,
    UE,
    UP,
    DeclareExplicitIdx,
    Dta,
    Pd,
    P,
    PdT,
    PdVars,
    Pdts,
    Pm2,
    Simp,
    SimpInto1,
    SimpSelect,
    SimpUq,
    AntiSym,
    Antisymmetric,
    DeclareIdx,
    DeclareSym,
    DeleteSym,
    DtaGen,
    ShowSym,
    Sym,
    Symmetric,
    TensorReplace,
    Dim,
    IdxColor,
    IdxDnList,
    IdxDnPtn,
    IdxDual,
    IdxHeadPtn,
    IdxList,
    IdxPtn,
    IdxSet,
    IdxUpList,
    IdxUpPtn,
    Uniq,
    UniqueIdx,
    Uq,
    declare_idx,
    dummy,
    free,
    idx,
    is_pdt,
    is_pm2,
    pm2_parts,
    pd2pdts,
    pdts2pd,
    rmE,
    tensor,
    tensor_head_name,
)


def test_declare_idx_records_duals_dimension_and_sets():
    dim_test = sp.Symbol("dimTest")
    u, d = declare_idx("u", "d", dim=dim_test, index_set=["a", "b", "c"], color="Blue")

    assert u("a").dual() == d("a")
    assert d("a").dual() == u("a")
    assert u.dim == dim_test
    assert d.dim == dim_test
    assert u.index_set[:3] == ("a", "b", "c")
    assert d.color == "Blue"


def test_public_index_name_pools_are_exported_from_package_root_like_upstream():
    assert mathgr.LatinIdx is tensor_module.LatinIdx


def test_tensor_state_snapshot_restores_uniq_counter():
    state = tensor_module._snapshot_tensor_registry_state()
    first = Uq(1)

    tensor_module._restore_tensor_registry_state(state)
    second = Uq(1)

    assert second == first
    assert mathgr.GreekIdx is tensor_module.GreekIdx
    assert mathgr.LatinCapitalIdx is tensor_module.LatinCapitalIdx


def test_DeclareIdx_accepts_upstream_style_positional_arguments():
    dim_test = sp.Symbol("DimDeclareIdxCompat")
    u, d = DeclareIdx(("DeclareIdxCompatU", "DeclareIdxCompatD"), dim_test, ["i", "j"], "Green")

    assert u in IdxList
    assert d in IdxList
    assert u in IdxUpList
    assert d in IdxDnList
    assert IdxDual(u) is d
    assert IdxDual(d) is u
    assert IdxSet(u) == ("i", "j")
    assert IdxColor(d) == "Green"
    assert Dim(u) == dim_test
    assert u("i").dual() == d("i")
    assert u.dim == dim_test
    assert u.index_set == ("i", "j")
    assert d.color == "Green"


def test_index_pattern_predicates_track_declared_index_registry():
    dim_test = sp.Symbol("DimDeclareIdxPatternCompat")
    u, d = DeclareIdx(("DeclareIdxPatternCompatU", "DeclareIdxPatternCompatD"), dim_test, ["i", "j"], "Green")

    assert IdxHeadPtn(u)
    assert IdxHeadPtn(d)
    assert IdxPtn(u("i"))
    assert IdxPtn(d("i"))
    assert IdxUpPtn(u("i"))
    assert not IdxUpPtn(d("i"))
    assert IdxDnPtn(d("i"))
    assert not IdxDnPtn(u("i"))


def test_declare_explicit_idx_matches_upstream_explicit_index_metadata():
    eu, ed = DeclareExplicitIdx(("ExplicitCompatU", "ExplicitCompatD"), "Purple")

    assert eu in IdxUpList
    assert ed in IdxDnList
    assert IdxDual(eu) is ed
    assert IdxColor(eu) == "Purple"
    assert eu(0).head.explicit is True
    assert Dta(eu(0), ed(0)) == 1
    assert Dta(eu(0), ed(1)) == 0


def test_uniq_and_uq_generate_fresh_labels_usable_as_indices():
    first, second = Uq(2)
    more = Uniq(2)

    assert first != second
    assert len(more) == 2
    assert len({first, second, *more}) == 4
    assert UP(first).label == first


def test_delta_contracts_nested_tensor_indices_and_canonicalizes_dummy():
    f = tensor("f")
    f1 = tensor("f1")

    expr = f(UP("a"), f1(UP("b"))) * Dta(UP("b"), UP("c"))
    assert Simp(expr) == f(UP("a"), f1(UP("c")))

    expr2 = f(UP("x"), DN("b")) * f1(DN("b"))
    assert Simp(expr2) == f(UP("x"), DN("a")) * f1(DN("a"))


def test_delta_handles_variance_combinations_sums_and_explicit_indices():
    f = tensor("f")
    x = sp.Symbol("xDeltaPowerScalar")

    assert Simp(f(UP("a"), DN("b")) * Dta(UP("b"), DN("c"))) == f(UP("a"), DN("c"))
    assert Simp(Dta(UP("a"), UP("c")) * Dta(UP("b"), UP("c"))) == Dta(UP("a"), UP("b"))
    assert Simp(Dta(UP("a"), UP("b")) * Dta(UP("a"), UP("b"))) == UP.dim
    assert Simp(x * Dta(UP("a"), UP("b")) ** 2) == x * UP.dim
    assert Dta(UE(1), DE(0)) == 0
    assert Dta(UE(1), DE(1)) == 1


def test_delta_contraction_does_not_rewrite_same_label_in_other_index_families():
    f = tensor("fDeltaFamilyIsolation")
    g = tensor("gDeltaFamilyIsolation")

    expr = Dta(DN("a"), DN("b")) * f(D1("b")) * g(DN("b"))

    assert Simp(expr) == f(D1("b")) * g(DN("a"))


def test_dta_unsupported_signatures_remain_symbolic_like_mathematica():
    dta_head = tensor("Dta")

    assert _unsupported_call_as_value(Dta) == dta_head()
    assert _unsupported_call_as_value(Dta, DN("a")) == dta_head(DN("a"))


def test_delta_matches_upstream_tensor_and_closed_chain_sum_cases():
    f = tensor("fDtaUpstreamCases")

    assert Simp(f(DN("a"), DN("b")) * Dta(DN("b"), DN("c"))) == f(DN("a"), DN("c"))
    assert Simp(f(UP("a"), UP("b")) * Dta(DN("b"), DN("c"))) == f(UP("a"), UP("c"))
    assert Simp(Dta(UP("a"), DN("b")) * Dta(UP("a"), DN("b"))) == UP.dim
    assert (
        Simp(
            Dta(UP("a"), UP("b"))
            * Dta(UP("b"), UP("c"))
            * Dta(UP("c"), UP("d"))
            * Dta(UP("d"), UP("a"))
        )
        == UP.dim
    )


def test_metric_delta_auto_evaluation_is_limited_to_registered_index_family():
    metric = tensor("metricFamilyIsolation")
    UseMetric(metric, (UP, DN), SetAsDefault=False)

    assert metric(UP("i"), DN("j")) == Dta(UP("i"), DN("j"))

    total_metric_call = metric(UTot("mu"), DTot("mu"))
    assert tensor_module.tensor_head_name(total_metric_call) == "metricFamilyIsolation"
    assert tensor_module.tensor_args(total_metric_call) == (UTot("mu"), DTot("mu"))


def test_total_metric_delta_auto_evaluation_includes_decomposed_dual_families():
    metric = tensor("totalMetricFamilyCompatibility")
    UseMetric(metric, (UTot, DTot), SetAsDefault=False)

    assert metric(U1("alpha"), D1("beta")) == Dta(U1("alpha"), D1("beta"))

    mixed_family_call = metric(UTot("mu"), D1("alpha"))
    assert tensor_module.tensor_head_name(mixed_family_call) == "totalMetricFamilyCompatibility"
    assert tensor_module.tensor_args(mixed_family_call) == (UTot("mu"), D1("alpha"))


def test_generalized_delta_and_symmetrization():
    f = tensor("f")

    assert DtaGen(UP("a"), UP("b"), DN("m"), DN("n")) == (
        Dta(UP("a"), DN("m")) * Dta(UP("b"), DN("n"))
        - Dta(UP("a"), DN("n")) * Dta(UP("b"), DN("m"))
    )
    assert Sym(f(DN("a"), DN("b"))) == f(DN("a"), DN("b")) + f(DN("b"), DN("a"))
    assert AntiSym(f(DN("a"), DN("b"))) == f(DN("a"), DN("b")) - f(DN("b"), DN("a"))


def test_sym_and_antisym_unsupported_signatures_remain_symbolic_like_mathematica():
    f = tensor("fSymArity")
    sym_head = tensor("Sym")
    antisym_head = tensor("AntiSym")
    expr = f(DN("a"), DN("b"))

    assert _unsupported_call_as_value(Sym) == sym_head()
    assert _unsupported_call_as_value(AntiSym) == antisym_head()
    assert _unsupported_call_as_value(Sym, expr, DN("a")) == expr
    assert _unsupported_call_as_value(AntiSym, expr, DN("a")) == expr
    assert _unsupported_call_as_value(Sym, expr, DN("a"), DN("b")) == sym_head(expr, DN("a"), DN("b"))
    assert _unsupported_call_as_value(AntiSym, expr, DN("a"), DN("b")) == antisym_head(expr, DN("a"), DN("b"))


def test_dta_gen_accepts_upstream_dtagendta_option_name():
    metric = tensor("metricDtaGenOption")

    assert DtaGen(UP("a"), UP("b"), DN("m"), DN("n"), DtaGenDta=metric) == (
        metric(UP("a"), DN("m")) * metric(UP("b"), DN("n"))
        - metric(UP("a"), DN("n")) * metric(UP("b"), DN("m"))
    )


def test_dta_gen_empty_signature_remains_symbolic_like_mathematica():
    assert DtaGen() == tensor("DtaGen")()


def test_idx_free_dummy_and_partial_derivative_product_rule():
    f = tensor("f")
    a = tensor("a")

    expr = a(UP("a"), DN("b")) * Pd(f(UP("c"), UP("b")), DN("a"))
    assert idx(expr) == ["a", "b", "c", "b", "a"]
    assert free(expr) == ["c"]
    assert dummy(expr) == ["a", "b"]

    g = sp.Symbol("g")
    h = tensor("h")
    assert Pd(f(UP("a")) * g + h(UP("a")), DN("b")) == (
        Pd(f(UP("a")), DN("b")) * g
        + f(UP("a")) * Pd(g, DN("b"))
        + Pd(h(UP("a")), DN("b"))
    )


def test_pd_unsupported_signatures_remain_symbolic_like_mathematica():
    phi = sp.Symbol("phi_pd_inert")
    pd_head = tensor("Pd")

    assert _unsupported_call_as_value(Pd) == pd_head()
    assert _unsupported_call_as_value(Pd, phi) == pd_head(phi)
    assert _unsupported_call_as_value(Pd, phi, DN("a"), DN("b")) == pd_head(phi, DN("a"), DN("b"))


def test_idx_free_dummy_order_powered_indexed_factors_like_upstream():
    f = tensor("f")
    a = tensor("a")

    expr = a(UP("a"), DN("b")) * Pd(f(UP("c"), UP("b")), DN("a")) * f(DN("x")) ** 2

    assert idx(expr) == ["a", "b", "x", "x", "c", "b", "a"]
    assert free(expr) == ["c"]
    assert dummy(expr) == ["a", "b", "x"]


def test_rmE_removes_explicit_indices_like_upstream_private_helper():
    u, d = declare_idx("rmEu", "rmEd", dim=4, index_set=["a", "b", "c"], color="Blue")

    assert rmE([UE(0), u("a"), d("b"), DE(1), d("c")]) == [u("a"), d("b"), d("c")]


def test_pd2pdts_and_pdts2pd_round_trip_upstream_private_derivative_representation():
    f = tensor("fPdts")

    scalar_pdts = Pdts(1, sp.Symbol("phi_pdts"), DN("a"))
    indexed_pdts = Pdts(1, f, UP("a"), DN("a"))

    assert pd2pdts(Pd(sp.Symbol("phi_pdts"), DN("a"))) == scalar_pdts
    assert pdts2pd(scalar_pdts) == Pd(sp.Symbol("phi_pdts"), DN("a"))
    assert pd2pdts(Pd(f(UP("a")), DN("a"))) == indexed_pdts
    assert pdts2pd(indexed_pdts) == Pd(f(UP("a")), DN("a"))


def test_pd2pdts_and_pdts2pd_rewrite_nested_derivatives_recursively():
    f = tensor("fPdtsRecursive")
    x = sp.Symbol("x_pdts_recursive")

    expr = x * Pd(f(DN("i")), DN("j")) + 1
    converted = x * Pdts(1, f, DN("i"), DN("j")) + 1

    assert pd2pdts(expr) == converted
    assert pdts2pd(converted) == expr


def test_pdts2pd_reconstructs_tensor_base_with_orderless_head_like_upstream():
    f = tensor("fPdtsOrderless")
    DeclareSym(f, (DN, DN), Symmetric("All"))

    assert pdts2pd(Pdts(1, f, DN("b"), DN("a"), DN("k"))) == PdT(
        f(DN("a"), DN("b")), PdVars(DN("k"))
    )


def test_simp_canonicalizes_direct_pdts_derivative_tensor_symmetries_like_upstream():
    f_sym = tensor("fDirectPdtsSym")
    f_anti = tensor("fDirectPdtsAnti")
    DeclareSym(f_sym, (DN, DN), Symmetric((1, 2)))
    DeclareSym(f_anti, (DN, DN), Antisymmetric((1, 2)))

    assert Simp(Pdts(2, f_sym, DN("j"), DN("i"), DN("b"), DN("a"))) == PdT(
        f_sym(DN("i"), DN("j")), PdVars(DN("a"), DN("b"))
    )
    assert Simp(Pdts(2, f_anti, DN("j"), DN("i"), DN("b"), DN("a"))) == -PdT(
        f_anti(DN("i"), DN("j")), PdVars(DN("a"), DN("b"))
    )
    assert Simp(Pdts(1, f_anti, DN("i"), DN("i"), DN("k"))) == 0


def test_partial_derivative_treats_declared_dimensions_and_deltas_as_constants():
    dim_test = sp.Symbol("dimTest2")
    u, d = declare_idx("u2", "d2", dim=dim_test, index_set=["a", "b"], color="Blue")

    assert Pd(Dta(u("a"), d("b")), d("c")) == 0
    assert Pd(sp.sin(1), d("c")) == 0
    assert Pd(dim_test, d("c")) == 0


def test_partial_derivative_variables_are_orderless_like_upstream():
    f = tensor("fPdVarsOrderless")

    assert PdVars(DN("b"), DN("a")) == PdVars(DN("a"), DN("b"))
    assert Pd(Pd(f(DN("x")), DN("b")), DN("a")) == Pd(Pd(f(DN("x")), DN("a")), DN("b"))


def test_pdt_direct_constructor_folds_numeric_and_composite_bases_like_upstream():
    x, y = sp.symbols("x y")

    assert PdT(3, PdVars(DN("i"))) == 0
    assert PdT(x + y, PdVars(DN("i"))) == Pd(x, DN("i")) + Pd(y, DN("i"))
    assert PdT(x * y, PdVars(DN("i"))) == x * Pd(y, DN("i")) + y * Pd(x, DN("i"))


def test_pdt_unsupported_signatures_remain_symbolic_like_mathematica():
    phi = sp.Symbol("phi_pdt_inert")
    pdt_head = tensor("PdT")

    assert _unsupported_call_as_value(PdT) == pdt_head()
    assert _unsupported_call_as_value(PdT, phi) == pdt_head(phi)
    assert _unsupported_call_as_value(PdT, phi, DN("a")) == pdt_head(phi, DN("a"))


def test_pdt_direct_constructor_appends_variables_to_existing_pdt_like_upstream():
    x = sp.Symbol("x")

    assert PdT(PdT(x, PdVars(DN("i"))), PdVars(DN("j"))) == PdT(x, PdVars(DN("i"), DN("j")))


def test_pdt_direct_constructor_treats_delta_as_derivative_constant_like_upstream():
    assert PdT(Dta(UP("a"), DN("b")), PdVars(DN("i"))) == 0


def test_P_shorthand_matches_upstream_curried_partial_derivative_operator():
    f = tensor("fCurriedP")

    assert P(DN("a"), DE(0))(f(UP("b"))) == PdT(f(UP("b")), PdVars(DN("a"), DE(0)))


def test_tensor_replace_matches_upstream_name_and_preserves_dummy_powers():
    f = tensor("fTensorReplace")
    g = tensor("gTensorReplace")

    assert TensorReplace(f(DN("a")) ** 2, {f(DN("a")): g(DN("a"))}) == g(DN("a")) ** 2
    assert TensorReplace({f(DN("a")): g(DN("a"))})(f(DN("a")) ** 2) == g(DN("a")) ** 2
    assert hasattr(mathgr, "TensorReplace")
    assert mathgr.TensorReplace is TensorReplace


def test_global_simp_hook_matches_upstream_simp_hook_variable():
    x, y = sp.symbols("x y")
    previous = tuple(tensor_module.SimpHook)
    tensor_module.SimpHook[:] = [lambda expr: y if expr == x else expr]

    try:
        assert Simp(x) == y
        assert mathgr.SimpHook is tensor_module.SimpHook
    finally:
        tensor_module.SimpHook[:] = previous


def test_simp_hook_accepts_sympy_wild_pattern_rules_like_upstream_replacement_hooks():
    eta = tensor("etaSimpHookPattern")
    i = sp.Wild("i")
    j = sp.Wild("j")
    previous = tuple(tensor_module.SimpHook)
    tensor_module.SimpHook[:] = [(Pd(eta(DN(i)), DN(j)), sp.Integer(0))]

    try:
        assert Simp(Pd(eta(DN("a")), DN("b"))) == 0
    finally:
        tensor_module.SimpHook[:] = previous


def test_simp_hook_simplifies_derivative_zero_rules_inside_products_like_upstream():
    eta = tensor("etaSimpHookProduct")
    f = tensor("fSimpHookProduct")
    i = sp.Wild("i")
    j = sp.Wild("j")
    previous = tuple(tensor_module.SimpHook)
    tensor_module.SimpHook[:] = [(Pd(eta(DN(i)), DN(j)), sp.Integer(0))]

    try:
        assert Simp(f(DN("a")) * Pd(eta(DN("b")), DN("c"))) == 0
    finally:
        tensor_module.SimpHook[:] = previous


def test_callable_simp_hook_simplifies_derivative_zero_rules_inside_products_like_upstream():
    eta = tensor("etaCallableSimpHookProduct")
    f = tensor("fCallableSimpHookProduct")
    previous = tuple(tensor_module.SimpHook)

    def hook(expr):
        if is_pdt(expr) and tensor_head_name(expr.args[0]) == "etaCallableSimpHookProduct":
            return sp.Integer(0)
        return expr

    tensor_module.SimpHook[:] = [hook]
    try:
        assert Simp(f(DN("a")) * Pd(eta(DN("b")), DN("c"))) == 0
    finally:
        tensor_module.SimpHook[:] = previous


def test_simp_uq_simplifies_with_fresh_dummy_labels_like_upstream():
    f = tensor("fSimpUq")
    g = tensor("gSimpUq")

    result = SimpUq(f(DN("i")) * g(DN("i")))

    labels = tensor_module.dummy(result)
    assert len(labels) == 1
    assert labels[0].startswith("uq")
    assert result.has(f(DN(labels[0])))
    assert result.has(g(DN(labels[0])))
    assert mathgr.SimpUq is SimpUq


def test_simp_accepts_custom_dummy_label_pool_like_upstream_option():
    f = tensor("fSimpDummyOption")
    g = tensor("gSimpDummyOption")

    assert Simp(f(DN("z")) * g(DN("z")), Dummy=("p", "q")) == f(DN("p")) * g(DN("p"))
    unique_pool = UniqueIdx()
    unique_result = Simp(f(DN("z")) * g(DN("z")), Dummy=unique_pool)

    assert len(unique_pool) == 50
    assert len(set(unique_pool)) == 50
    assert unique_result == f(DN(unique_pool[0])) * g(DN(unique_pool[0]))


def test_simp_enters_upstream_single_argument_functions_from_simpinto1():
    f = tensor("fSimpInto")

    assert sp.exp in SimpInto1
    assert Simp(sp.exp(f(UP("a")) * Dta(UP("a"), UP("b")))) == sp.exp(f(UP("b")))
    assert mathgr.SimpInto1 is SimpInto1


def test_simp_enters_power_bases_with_dummy_indices_like_upstream():
    f = tensor("fSimpPowerInto")
    g = tensor("gSimpPowerInto")

    expr = sp.sqrt(1 + f(UP("a")) * Dta(UP("a"), UP("b")) * g(UP("b")))

    assert Simp(expr) == sp.sqrt(1 + f(UP("a")) * g(UP("a")))


def test_simp_preserves_sympy_wild_index_labels_like_upstream_patterns():
    f = tensor("fSimpPattern")
    x = tensor("xSimpPattern")
    a_wild, b_wild, c_wild, d_wild = (sp.Wild(name) for name in ("a", "b", "c", "d"))

    expr = f(DN(a_wild), DN(c_wild)) * x(UP(d_wild), UP(b_wild))

    assert DN(a_wild).label == a_wild
    assert Simp(expr) == expr
    assert Simp(expr).has(a_wild, b_wild, c_wild, d_wild)


def test_simp_select_filters_expanded_terms_like_upstream_global():
    f = tensor("fSimpSelect")
    selector_delta = Dta(UP("a"), UP("b"))
    previous = tensor_module.SimpSelect
    tensor_module.SimpSelect = lambda terms: [term for term in terms if term.has(selector_delta)]

    try:
        assert Simp(f(UP("a")) * selector_delta + f(UP("c"))) == f(UP("b"))
        assert mathgr.SimpSelect is tensor_module.SimpSelect
    finally:
        tensor_module.SimpSelect = previous


def test_pm2_expand_constant_and_derivative_rules():
    a, b, x = sp.symbols("a b x")
    assert Simp(x * Pm2((a + b) ** 2, DN)) == (
        x * Pm2(a**2, DN) + 2 * x * Pm2(a * b, DN) + x * Pm2(b**2, DN)
    )

    dim_pm2 = sp.Symbol("dimPm2Constant")
    declare_idx("pm2u", "pm2d", dim=dim_pm2, index_set=["a", "b"], color="Blue")
    g = sp.Symbol("g_pm2_constant")
    assert Pm2(dim_pm2 * g, DN) == dim_pm2 * Pm2(g, DN)

    f = tensor("f")
    assert Pd(Pm2(PdT(f(DN("a")), PdVars(DN("b"), DE(0))), DN), DN("b")) == PdT(
        f(DN("a")), PdVars(DE(0))
    )


def test_pm2_plus_power_expansion_stops_at_upstream_degree_cap():
    a, b = sp.symbols("a_pm2_cap b_pm2_cap")

    result = Pm2((a + b) ** 5, DN)
    inner, index_type = pm2_parts(result)

    assert is_pm2(result)
    assert inner == (a + b) ** 5
    assert index_type is DN


def test_pm2_constant_factor_detection_uses_simp_hook_derivative_zero_rules():
    f0, g = sp.symbols("f0_pm2_hook_constant g_pm2_hook_constant")
    i = sp.Wild("i")
    previous = tuple(tensor_module.SimpHook)
    tensor_module.SimpHook[:] = [(Pd(f0, DN(i)), sp.Integer(0))]

    try:
        assert Pm2(f0 * g, DN) == f0 * Pm2(g, DN)
    finally:
        tensor_module.SimpHook[:] = previous


def test_pm2_commutes_with_partial_derivatives_like_upstream_rule():
    f = tensor("fPm2DerivativeCommute")

    assert Pm2(PdT(f(DN("a")), PdVars(DE(0))), DN) == PdT(Pm2(f(DN("a")), DN), PdVars(DE(0)))


def test_pm2_unsupported_signatures_remain_symbolic_like_mathematica():
    phi = sp.Symbol("phi_pm2_inert")
    pm2_head = tensor("Pm2")

    assert _unsupported_call_as_value(Pm2) == pm2_head()
    assert Pm2(phi) == pm2_head(phi)


def test_declare_show_and_delete_tensor_symmetries():
    f_tmp = tensor("fTmp")

    assert DeclareSym(f_tmp, (UP, UP, DE(0), DN, DN), Symmetric((1, 2))) == [Symmetric((1, 2))]
    assert DeclareSym(f_tmp, (UP, UP, DE(0), DN, DN), Symmetric((3, 4))) == [
        Symmetric((1, 2)),
        Symmetric((3, 4)),
    ]
    assert ShowSym(f_tmp, (UP, UP, DE(0), DN, DN)) == [Symmetric((1, 2)), Symmetric((3, 4))]
    assert DeleteSym(f_tmp, (UP, UP, DE(0), DN, DN)) is None
    assert ShowSym(f_tmp, (UP, UP, DE(0), DN, DN)) == []


def test_declare_sym_rejects_symmetric_slots_with_different_index_identifiers_like_upstream():
    f_mixed = tensor("fMixedSymmetryRejected")

    assert DeclareSym(f_mixed, (UP, DN), Symmetric((1, 2))) is None
    assert DeclareSym(f_mixed, (UP, DN), Antisymmetric((1, 2))) is None
    assert ShowSym(f_mixed, (UP, DN)) == []


def test_declare_sym_accepts_multiple_symmetry_declarations_in_one_call():
    f_multi = tensor("fMultipleSymmetries")
    symmetries = [Symmetric((1, 2)), Antisymmetric((3, 4))]

    assert DeclareSym(f_multi, (DN, DN, DN, DN), symmetries) == symmetries
    assert ShowSym(f_multi, (DN, DN, DN, DN)) == symmetries


def test_declare_sym_expands_all_to_non_explicit_slots_like_upstream():
    f_all = tensor("fAllSymmetry")

    assert DeclareSym(f_all, (DN, DN, DE(0), DN), Antisymmetric("All")) == [Antisymmetric((1, 2, 3))]
    assert ShowSym(f_all, (DN, DN, DE(0), DN)) == [Antisymmetric((1, 2, 3))]
    assert Simp(f_all(DN("a"), DN("b"), DE(0), DN("a"))) == 0


def test_fully_symmetric_declaration_makes_tensor_head_orderless_like_upstream():
    f_orderless = tensor("fFullySymmetricOrderless")

    assert DeclareSym(f_orderless, (DN, DN), Symmetric("All")) == [Symmetric((1, 2))]
    assert f_orderless(DN("b"), DN("a")) == f_orderless(DN("a"), DN("b"))


def test_declare_sym_accepts_slot_permutation_symmetry():
    f_perm = tensor("fPermutationSymmetry")
    symmetry = tensor_module.PermutationSymmetry((2, 3, 1))

    assert DeclareSym(f_perm, (DN, DN, DN), symmetry) == [symmetry]
    assert ShowSym(f_perm, (DN, DN, DN)) == [symmetry]
    assert Simp(f_perm(DN("c"), DN("a"), DN("b"))) == f_perm(DN("a"), DN("b"), DN("c"))
    assert Simp(f_perm(DN("b"), DN("c"), DN("a")) - f_perm(DN("a"), DN("b"), DN("c"))) == 0


def test_declare_sym_accepts_cycles_symmetry():
    f_cycle = tensor("fCyclesSymmetry")
    symmetry = tensor_module.Cycles(((1, 2, 3),))

    assert DeclareSym(f_cycle, (DN, DN, DN), symmetry) == [symmetry]
    assert ShowSym(f_cycle, (DN, DN, DN)) == [symmetry]
    assert Simp(f_cycle(DN("c"), DN("a"), DN("b"))) == f_cycle(DN("a"), DN("b"), DN("c"))


def test_simp_uses_declared_symmetric_and_antisymmetric_slots_to_zero_terms():
    f3 = tensor("f3")
    DeclareSym(f3, (DN, DN, DN, DN), Symmetric((1, 2)))
    DeclareSym(f3, (DN, DN, DN, DN), Antisymmetric((3, 4)))

    assert Simp(f3(DN("i"), DN("j"), DN("i"), DN("j"))) == 0
    assert Simp(Pd(f3(DN("i"), DN("j"), DN("i"), DN("j")), DN("k"))) == 0


def test_simp_canonicalizes_declared_symmetric_slots_for_cancellation():
    f_sym = tensor("fSymCanonical")
    DeclareSym(f_sym, (DN, DN), Symmetric((1, 2)))

    assert Simp(f_sym(DN("b"), DN("a"))) == f_sym(DN("a"), DN("b"))
    assert Simp(f_sym(DN("b"), DN("a")) - f_sym(DN("a"), DN("b"))) == 0


def test_simp_declared_symmetry_slots_skip_explicit_indices_like_upstream():
    f_explicit_slot = tensor("fExplicitSlotSymmetry")
    DeclareSym(f_explicit_slot, (UP, UP, DE(0), DN, DN), Symmetric((3, 4)))

    assert Simp(f_explicit_slot(UP("p"), UP("q"), DE(0), DN("b"), DN("a"))) == f_explicit_slot(
        UP("p"), UP("q"), DE(0), DN("a"), DN("b")
    )


def test_simp_keeps_upstream_free_index_factor_ordering_after_dummy_sorting():
    c = tensor("cFreeIndexSorted")
    vector = tensor("AFreeIndexSorted")

    assert Simp(c(D2("b"), D2("c"), U2("a")) * vector(U2("b"), DN("mu")) * vector(U2("c"), DN("nu"))) == (
        vector(U2("b"), DN("mu")) * vector(U2("c"), DN("nu")) * c(D2("b"), D2("c"), U2("a"))
    )


def test_simp_canonicalizes_declared_antisymmetric_slots_with_sign():
    f_anti = tensor("fAntiCanonical")
    DeclareSym(f_anti, (DN, DN), Antisymmetric((1, 2)))

    assert Simp(f_anti(DN("b"), DN("a"))) == -f_anti(DN("a"), DN("b"))
    assert Simp(f_anti(DN("a"), DN("a"))) == 0
    assert Simp(f_anti(DN("b"), DN("a")) + f_anti(DN("a"), DN("b"))) == 0
    assert Simp(Pd(f_anti(DN("b"), DN("a")), DN("k"))) == -Pd(f_anti(DN("a"), DN("b")), DN("k"))


def test_simp_tensorreduce_zeroes_symmetric_antisymmetric_dummy_contraction():
    f_sym = tensor("fSymAntiProduct")
    f_anti = tensor("fAntiSymProduct")
    DeclareSym(f_sym, (DN, DN), Symmetric((1, 2)))
    DeclareSym(f_anti, (DN, DN), Antisymmetric((1, 2)))

    assert Simp(f_sym(DN("i"), DN("j")) * f_anti(DN("i"), DN("j"))) == 0
    assert Simp(f_sym(DN("i"), DN("j")) * f_anti(DN("j"), DN("i"))) == 0


def test_simp_tensorreduce_zeroes_bridged_symmetric_antisymmetric_contraction():
    f_sym = tensor("fSymAntiBridgeProduct")
    f_anti = tensor("fAntiSymBridgeProduct")
    bridge = tensor("xSymAntiBridgeProduct")
    DeclareSym(f_sym, (DN, DN), Symmetric((1, 2)))
    DeclareSym(f_anti, (DN, DN), Antisymmetric((1, 2)))

    expr = (
        f_sym(DN("i"), DN("j"))
        * f_anti(DN("k"), DN("l"))
        * bridge(DN("i"), DN("k"))
        * bridge(DN("j"), DN("l"))
    )

    assert Simp(expr) == 0


def test_simp_fast_method_stops_before_tensorreduce_style_product_symmetry():
    f_sym = tensor("fSymAntiBridgeFastProduct")
    f_anti = tensor("fAntiSymBridgeFastProduct")
    bridge = tensor("xSymAntiBridgeFastProduct")
    DeclareSym(f_sym, (DN, DN), Symmetric((1, 2)))
    DeclareSym(f_anti, (DN, DN), Antisymmetric((1, 2)))

    expr = (
        f_sym(DN("i"), DN("j"))
        * f_anti(DN("k"), DN("l"))
        * bridge(DN("i"), DN("k"))
        * bridge(DN("j"), DN("l"))
    )

    assert Simp(expr, Method="Fast") == (
        f_anti(DN("a"), DN("b"))
        * f_sym(DN("c"), DN("d"))
        * bridge(DN("c"), DN("a"))
        * bridge(DN("d"), DN("b"))
    )


def test_simp_tensorreduce_zeroes_dual_family_symmetric_antisymmetric_contraction():
    f_sym = tensor("fDualSymAntiProduct")
    f_anti = tensor("fDualAntiSymProduct")
    DeclareSym(f_sym, (UP, UP), Symmetric((1, 2)))
    DeclareSym(f_anti, (DN, DN), Antisymmetric((1, 2)))

    assert Simp(f_sym(UP("i"), UP("j")) * f_anti(DN("i"), DN("j"))) == 0
    assert Simp(Pd(f_sym(UP("i"), UP("j")), DE(0)) * f_anti(DN("i"), DN("j"))) == 0


def test_simp_tensorreduce_canonicalizes_dummy_order_through_declared_product_symmetry():
    f_sym = tensor("fSymProductCanonical")
    f_anti = tensor("fAntiProductCanonical")
    x = tensor("xProductCanonical")
    DeclareSym(f_sym, (DN, DN), Symmetric((1, 2)))
    DeclareSym(f_anti, (DN, DN), Antisymmetric((1, 2)))

    assert Simp(f_sym(DN("i"), DN("j")) * x(DN("j"), DN("i"))) == f_sym(DN("a"), DN("b")) * x(
        DN("a"), DN("b")
    )
    assert Simp(f_anti(DN("i"), DN("j")) * x(DN("j"), DN("i"))) == -f_anti(DN("a"), DN("b")) * x(
        DN("a"), DN("b")
    )


def test_simp_tensorreduce_canonicalizes_dummy_order_through_derivative_product_symmetry():
    f_sym = tensor("fSymDerivativeProductCanonical")
    f_anti = tensor("fAntiDerivativeProductCanonical")
    x = tensor("xDerivativeProductCanonical")
    DeclareSym(f_sym, (DN, DN), Symmetric((1, 2)))
    DeclareSym(f_anti, (DN, DN), Antisymmetric((1, 2)))

    assert Simp(Pd(f_sym(DN("i"), DN("j")), DE(0)) * x(DN("j"), DN("i"))) == Pd(
        f_sym(DN("a"), DN("b")), DE(0)
    ) * x(DN("a"), DN("b"))
    assert Simp(Pd(f_anti(DN("i"), DN("j")), DE(0)) * x(DN("j"), DN("i"))) == -Pd(
        f_anti(DN("a"), DN("b")), DE(0)
    ) * x(DN("a"), DN("b"))


def test_simp_tensorreduce_canonicalizes_dummy_order_with_one_symmetric_derivative_factor():
    f_sym = tensor("fSymTwoDerivativeProductCanonical")
    x = tensor("xTwoDerivativeProductCanonical")
    DeclareSym(f_sym, (DN, DN), Symmetric((1, 2)))

    assert Simp(Pd(f_sym(DN("i"), DN("j")), DE(0)) * Pd(x(DN("j"), DN("i")), DE(0))) == Pd(
        f_sym(DN("a"), DN("b")), DE(0)
    ) * Pd(x(DN("a"), DN("b")), DE(0))


def test_simp_tensorreduce_canonicalizes_multiple_symmetric_direct_factors_with_derivatives():
    eta = tensor("etaMultiMetricDerivativeCanonical")
    vector = tensor("vectorMultiMetricDerivativeCanonical")
    DeclareSym(eta, (U1, U1), Symmetric((1, 2)))

    expr = (
        eta(U1("a"), U1("c"))
        * eta(U1("b"), U1("d"))
        * Pd(vector(D1("a")), D1("b"))
        * Pd(vector(D1("c")), D1("d"))
    )
    relabeled = (
        eta(U1("a"), U1("d"))
        * eta(U1("b"), U1("c"))
        * Pd(vector(D1("a")), D1("c"))
        * Pd(vector(D1("d")), D1("b"))
    )

    assert Simp(expr - relabeled) == 0


def test_simp_tensorreduce_canonicalizes_large_symmetric_product_without_full_variant_expansion():
    sym = tensor("symLargeProductCanonical")
    left = tensor("leftLargeProductCanonical")
    right = tensor("rightLargeProductCanonical")
    DeclareSym(sym, (DN, DN), Symmetric((1, 2)))
    labels = tuple((f"i{pos}", f"j{pos}") for pos in range(9))

    symmetric_factors = sp.Mul(*(sym(DN(first), DN(second)) for first, second in labels))
    expr = symmetric_factors * left(*(DN(first) for first, _second in labels)) * right(
        *(DN(second) for _first, second in labels)
    )
    swapped = tuple((second, first) if pos in {0, 3, 7} else (first, second) for pos, (first, second) in enumerate(labels))
    relabeled = symmetric_factors * left(*(DN(first) for first, _second in swapped)) * right(
        *(DN(second) for _first, second in swapped)
    )

    assert Simp(expr - relabeled) == 0


def test_simp_tensorreduce_zeroes_large_product_with_bridged_symmetric_antisymmetric_contraction():
    f_sym = tensor("fSymAntiLargeBridgeProduct")
    f_anti = tensor("fAntiSymLargeBridgeProduct")
    bridge = tensor("xSymAntiLargeBridgeProduct")
    extra_sym = tensor("symLargeBridgeProduct")
    extra_bridge = tensor("yLargeBridgeProduct")
    DeclareSym(f_sym, (DN, DN), Symmetric((1, 2)))
    DeclareSym(f_anti, (DN, DN), Antisymmetric((1, 2)))
    DeclareSym(extra_sym, (DN, DN), Symmetric((1, 2)))

    expr = (
        f_sym(DN("i"), DN("j"))
        * f_anti(DN("k"), DN("l"))
        * bridge(DN("i"), DN("k"))
        * bridge(DN("j"), DN("l"))
    )
    for pos in range(1, 9):
        expr *= extra_sym(DN(f"p{pos}"), DN(f"q{pos}")) * extra_bridge(DN(f"p{pos}"), DN(f"q{pos}"))

    assert Simp(expr) == 0


def test_simp_tensorreduce_canonicalizes_multiple_symmetric_derivative_factors():
    f_sym = tensor("fMultiDerivativeSymmetryCanonical")
    g_sym = tensor("gMultiDerivativeSymmetryCanonical")
    x = tensor("xMultiDerivativeSymmetryCanonical")
    DeclareSym(f_sym, (DN, DN), Symmetric((1, 2)))
    DeclareSym(g_sym, (DN, DN), Symmetric((1, 2)))

    expr = (
        Pd(f_sym(DN("i"), DN("j")), DE(0))
        * Pd(g_sym(DN("i"), DN("k")), DE(0))
        * x(DN("j"), DN("k"))
    )

    assert Simp(expr) == (
        x(DN("a"), DN("b"))
        * Pd(f_sym(DN("a"), DN("c")), DE(0))
        * Pd(g_sym(DN("b"), DN("c")), DE(0))
    )


def test_simp_recanonicalizes_metric_derivative_slots_after_dummy_relabeling():
    u, d = declare_idx(
        "metricDerivativeCanonicalU",
        "metricDerivativeCanonicalD",
        dim=4,
        color="Black",
    )
    metric = tensor("metricDerivativeCanonical")
    UseMetric(metric, (u, d), SetAsDefault=False)

    expr = (
        metric(u("a"), u("c"))
        * metric(u("b"), u("d"))
        * Pd(metric(d("a"), d("b")), d("e"))
        * Pd(metric(d("c"), d("d")), d("e"))
    )
    relabeled = (
        metric(u("a"), u("d"))
        * metric(u("b"), u("c"))
        * Pd(metric(d("a"), d("b")), d("e"))
        * Pd(metric(d("c"), d("d")), d("e"))
    )

    assert Simp(expr - relabeled) == 0


def test_simp_tensorreduce_zeroes_derivative_symmetric_antisymmetric_contraction():
    f_sym = tensor("fSymAntiDerivativeProduct")
    f_anti = tensor("fAntiSymDerivativeProduct")
    DeclareSym(f_sym, (DN, DN), Symmetric((1, 2)))
    DeclareSym(f_anti, (DN, DN), Antisymmetric((1, 2)))

    assert Simp(Pd(f_sym(DN("i"), DN("j")), DE(0)) * f_anti(DN("i"), DN("j"))) == 0


def test_simp_does_not_rename_dummies_onto_overused_labels():
    f = tensor("fOverusedDummy")
    g = tensor("gOverusedDummy")
    h = tensor("hOverusedDummy")

    assert Simp(f(DN("a")) ** 3 * g(DN("b")) * h(DN("b"))) == (
        f(DN("a")) ** 3 * g(DN("b")) * h(DN("b"))
    )


def test_simp_keeps_explicit_indices_while_canonicalizing_einstein_dummies():
    f3 = tensor("f3Explicit")
    _u_exp, d_exp = declare_idx(
        "u_explicit",
        "d_explicit",
        dim=sp.Symbol("dimExplicit"),
        index_set=["a", "b", "c"],
        color="Blue",
    )

    expr = Pd(f3(d_exp("i"), DE(0), d_exp("i"), DE(0)), d_exp("k"))

    assert Simp(expr) == Pd(f3(d_exp("a"), DE(0), d_exp("a"), DE(0)), d_exp("k"))


def test_simp_splits_high_even_powers_of_indexed_free_tensor_into_dummy_pairs():
    scalar_a = sp.Symbol("a")
    f_power = tensor("fPower")

    expr = scalar_a**2 * (f_power(UP("a")) ** 4 + 1)

    assert Simp(expr) == scalar_a**2 + scalar_a**2 * f_power(UP("a")) ** 2 * f_power(UP("b")) ** 2


def test_simp_splits_powers_of_scalar_dummy_contractions_like_upstream():
    f_power = tensor("fScalarContractionPower")
    g_power = tensor("gScalarContractionPower")

    expr = (f_power(DN("i")) * g_power(DN("i"))) ** 2
    expected = f_power(DN("a")) * g_power(DN("a")) * f_power(DN("b")) * g_power(DN("b"))

    assert Simp(expr) == expected


def test_simp_power_split_keeps_same_label_different_families_separate():
    f = tensor("fPowerFamilyIsolation")
    g = tensor("gPowerFamilyIsolation")

    expr = (f(D1("a")) * g(DN("a"))) ** 2

    assert Simp(expr) == f(D1("α")) ** 2 * g(DN("a")) ** 2


def test_simp_splits_powers_of_scalar_trace_contractions_like_upstream():
    f_power = tensor("fScalarTracePower")

    expr = f_power(DN("i"), DN("i")) ** 2
    expected = f_power(DN("a"), DN("a")) * f_power(DN("b"), DN("b"))

    assert Simp(expr) == expected


def test_simp_renames_multiple_dummy_labels_simultaneously_inside_derivatives():
    f = tensor("fDerivativeDummySwap")
    g = tensor("gDerivativeDummySwap")

    expr = PdT(f(DN("b")), PdVars(DN("a"))) * PdT(g(DN("b")), PdVars(DN("a")))

    assert Simp(expr) == PdT(f(DN("a")), PdVars(DN("b"))) * PdT(g(DN("a")), PdVars(DN("b")))


def _unsupported_call_as_value(func, *args):
    try:
        return func(*args)
    except (TypeError, ValueError) as exc:
        return exc
