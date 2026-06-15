from collections import Counter
import importlib

import sympy as sp

import mathgr
import mathgr.gr as gr_module
from mathgr.gr import Affine, CovD, DG, Dsquare, G, K, KK, LapseN, RADM, ShiftN, T, UG, MetricContract, R, Rsimp, UseMetric, V, X
from mathgr.tensor import DE, DN, UP, Dta, LatinIdx, Pd, ShowSym, Simp, Symmetric, declare_idx, tensor

tensor_module = importlib.import_module("mathgr.tensor")


def test_default_metric_g_is_initialized_like_upstream_gr_module():
    assert mathgr.g is gr_module.g
    assert gr_module.Metric is gr_module.g
    assert R() == Rsimp()


def test_package_root_metric_state_tracks_gr_module_like_upstream_globals():
    original_metric = gr_module.Metric
    original_indices = gr_module.IdxOfMetric
    u, d = declare_idx("rootMetricU", "rootMetricD", dim=4, index_set=LatinIdx, color="Black")
    metric = tensor("root_metric_state")

    try:
        UseMetric(metric, (u, d))

        assert mathgr.Metric is gr_module.Metric
        assert mathgr.Metric is metric
        assert mathgr.IdxOfMetric == (u, d)
    finally:
        UseMetric(original_metric, original_indices)


def test_metric_contract_pairs_marked_slots_with_current_metric():
    u, d = declare_idx("gu", "gd", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_metric_contract")
    f = tensor("f")
    f1 = tensor("f1")
    UseMetric(g, (u, d))

    assert Simp(MetricContract(f(UG(1), DG(2)) * f1(UG(1), DG(2)))) == (
        f(u("a"), d("b"))
        * f1(u("c"), d("d"))
        * g(d("a"), d("c"))
        * g(u("b"), u("d"))
    )


def test_metric_contract_mixed_marked_slots_use_mixed_metric_like_upstream():
    u, d = declare_idx("mixedMcu", "mixedMcd", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_metric_contract_mixed")
    f = tensor("f_metric_contract_mixed")
    UseMetric(g, (u, d))

    assert MetricContract(f(UG(1), DG(1))) == Dta(u("b"), d("a")) * f(u("a"), d("b"))
    assert Simp(MetricContract(f(UG(1), DG(1)))) == f(u("a"), d("a"))


def test_metric_contract_lone_marked_slots_emit_metric_factors_like_upstream():
    u, d = declare_idx("loneMcu", "loneMcd", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_metric_contract_lone")
    f = tensor("f_metric_contract_lone")
    UseMetric(g, (u, d))

    assert MetricContract(f(UG(1))) == g(d("a")) * f(u("a"))
    assert MetricContract(f(DG(1))) == g(u("a")) * f(d("a"))
    assert MetricContract(f(UG(1), DG(2))) == g(d("a")) * g(u("b")) * f(u("a"), d("b"))


def test_use_metric_declares_same_variance_metric_symmetries_even_when_not_default():
    u, d = declare_idx("gmsu", "gmsd", dim=4, index_set=LatinIdx, color="Black")
    metric = tensor("g_metric_symmetry")

    UseMetric(metric, (u, d), SetAsDefault=False)

    assert ShowSym(metric, (u, u)) == [Symmetric((1, 2))]
    assert ShowSym(metric, (d, d)) == [Symmetric((1, 2))]
    assert Simp(metric(u("b"), u("a"))) == metric(u("a"), u("b"))
    assert Simp(metric(d("b"), d("a")) - metric(d("a"), d("b"))) == 0


def test_use_metric_rewrites_inverse_metric_partial_derivative():
    u, d = declare_idx("gimu", "gimd", dim=4, index_set=LatinIdx, color="Black")
    metric = tensor("g_inverse_metric_derivative")

    UseMetric(metric, (u, d), SetAsDefault=False)

    assert Pd(metric(u("m"), u("n")), d("l")) == (
        -metric(u("a"), u("n")) * metric(u("b"), u("m")) * Pd(metric(d("a"), d("b")), d("l"))
    )


def test_inverse_metric_derivative_avoids_product_context_free_labels():
    u, d = declare_idx("gimcu", "gimcd", dim=4, index_set=LatinIdx, color="Black")
    metric = tensor("g_inverse_metric_derivative_context")
    field = tensor("f_inverse_metric_derivative_context")

    UseMetric(metric, (u, d), SetAsDefault=False)

    assert Pd(metric(u("a"), u("e")) * field(d("b")), d("c")) == (
        -metric(u("d"), u("e"))
        * metric(u("f"), u("a"))
        * field(d("b"))
        * Pd(metric(d("d"), d("f")), d("c"))
        + metric(u("a"), u("e")) * Pd(field(d("b")), d("c"))
    )


def test_use_metric_contracts_inverse_metric_product_to_delta():
    u, d = declare_idx("gipu", "gipd", dim=4, index_set=LatinIdx, color="Black")
    metric = tensor("g_inverse_product")

    UseMetric(metric, (u, d), SetAsDefault=False)

    assert Simp(metric(u("a"), u("c")) * metric(d("c"), d("b"))) == Dta(u("a"), d("b"))


def test_with_metric_temporarily_scopes_metric_evaluation_and_restores_default():
    assert hasattr(gr_module, "WithMetric")

    default_u, default_d = declare_idx("wmu", "wmd", dim=4, index_set=LatinIdx, color="Black")
    default_metric = tensor("g_with_metric_default")
    scoped_u, scoped_d = declare_idx("wmsu", "wmsd", dim=4, index_set=LatinIdx, color="Black")
    scoped_metric = tensor("g_with_metric_scoped")
    field = sp.Symbol("phi_with_metric")

    UseMetric(default_metric, (default_u, default_d))

    assert gr_module.WithMetric(scoped_metric, (scoped_u, scoped_d), lambda: X(field)) == (
        -scoped_metric(scoped_u("a"), scoped_u("b")) * Pd(field, scoped_d("a")) * Pd(field, scoped_d("b")) / 2
    )
    assert X(field) == (
        -default_metric(default_u("a"), default_u("b")) * Pd(field, default_d("a")) * Pd(field, default_d("b")) / 2
    )
    assert mathgr.WithMetric is gr_module.WithMetric


def test_affine_connection_uses_current_metric_and_partial_derivatives():
    u, d = declare_idx("au", "ad", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_affine")
    UseMetric(g, (u, d))

    assert Simp(Affine(u("a"), d("b"), d("c"))) == (
        -g(u("a"), u("d")) * Pd(g(d("b"), d("c")), d("d")) / 2
        + g(u("a"), u("d")) * Pd(g(d("b"), d("d")), d("c")) / 2
        + g(u("a"), u("d")) * Pd(g(d("c"), d("d")), d("b")) / 2
    )


def test_ricci_scalar_matches_upstream_presimplified_expression():
    u, d = declare_idx("ru", "rd", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_ricci")
    UseMetric(g, (u, d))

    assert Simp(R()) == _expected_ricci_scalar(g, u, d)


def test_ricci_scalar_helper_exposes_upstream_metric_contracted_definition():
    u, d = declare_idx("rrawu", "rrawd", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_ricci_raw")
    UseMetric(g, (u, d))

    assert mathgr.RicciScalar() == MetricContract(R(DG(1), DG(1)))
    assert mathgr.RicciScalar() == sp.expand(g(u("a"), u("b")) * R(d("a"), d("b")))


def test_ricci_tensor_accepts_distinct_metric_contract_slots_like_upstream():
    u, d = declare_idx("rdgu", "rdgd", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_ricci_metric_slots")
    UseMetric(g, (u, d))

    assert R(DG(1), DG(2)) == R(d("a"), d("b"))


def test_ricci_tensor_raises_indices_with_current_metric():
    u, d = declare_idx("rvaru", "rvard", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_ricci_variance")
    UseMetric(g, (u, d))

    assert R(u("a"), d("b")) == g(u("a"), u("c")) * R(d("c"), d("b"))
    assert R(d("a"), u("b")) == g(u("b"), u("c")) * R(d("a"), d("c"))
    assert R(u("a"), u("b")) == g(u("a"), u("c")) * g(u("b"), u("d")) * R(d("c"), d("d"))
    assert R(UG(1), DG(2)) == R(u("a"), d("b"))


def test_unsupported_curvature_signatures_remain_symbolic_like_mathematica():
    r_head = tensor("R")
    g_head = tensor("G")
    rsimp_head = tensor("Rsimp")

    assert R(DN("a")) == r_head(DN("a"))
    assert R(DN("a"), DN("b"), DN("c")) == r_head(DN("a"), DN("b"), DN("c"))
    assert R(DE(0), DE(0)) == r_head(DE(0), DE(0))
    assert G(DN("a")) == g_head(DN("a"))
    assert Rsimp(DN("a")) == rsimp_head(DN("a"))


def test_riemann_component_uses_collision_free_affine_definition():
    u, d = declare_idx("rmu", "rmd", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_riemann")
    UseMetric(g, (u, d))

    assert R(u("a"), d("b"), d("c"), d("d")) == (
        Pd(gr_module._affine_with_dummy(u("a"), d("b"), d("d"), "e", u, d), d("c"))
        - Pd(gr_module._affine_with_dummy(u("a"), d("b"), d("c"), "f", u, d), d("d"))
        + gr_module._affine_with_dummy(u("g"), d("b"), d("d"), "h", u, d)
        * gr_module._affine_with_dummy(u("a"), d("g"), d("c"), "i", u, d)
        - gr_module._affine_with_dummy(u("g"), d("b"), d("c"), "j", u, d)
        * gr_module._affine_with_dummy(u("a"), d("g"), d("d"), "k", u, d)
    )


def test_lower_riemann_accepts_distinct_metric_contract_slots_like_upstream():
    u, d = declare_idx("rslotu", "rslotd", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_riemann_metric_slots")
    UseMetric(g, (u, d))

    assert R(DG(1), DG(2), DG(3), DG(4)) == R(d("a"), d("b"), d("c"), d("d"))


def test_riemann_raises_nonfirst_indices_with_current_metric():
    u, d = declare_idx("rvar4u", "rvar4d", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_riemann_variance")
    UseMetric(g, (u, d))

    assert R(d("a"), u("b"), d("c"), d("d")) == g(u("b"), u("e")) * R(d("a"), d("e"), d("c"), d("d"))
    assert R(u("a"), u("b"), d("c"), d("d")) == g(u("b"), u("e")) * R(u("a"), d("e"), d("c"), d("d"))
    assert R(DG(1), UG(2), DG(3), DG(4)) == R(d("a"), u("b"), d("c"), d("d"))


def test_lower_riemann_expansion_avoids_dummy_labels_colliding_with_free_indices():
    u, d = declare_idx("rfreeu", "rfreed", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_riemann_free_labels")
    UseMetric(g, (u, d))

    terms = sp.expand(Simp(R(d("a"), d("b"), d("c"), d("d")))).args

    assert all(
        max(Counter(index.label for index in tensor_module._iter_indices(term, include_explicit=False)).values()) <= 2
        for term in terms
    )


def test_lower_riemann_first_pair_antisymmetry_simplifies_like_upstream_tensorreduce():
    u, d = declare_idx("rfirstu", "rfirstd", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_riemann_first_pair")
    UseMetric(g, (u, d))

    assert Simp(R(d("a"), d("b"), d("c"), d("d")) + R(d("b"), d("a"), d("c"), d("d"))) == 0
    assert Simp(R(d("a"), d("a"), d("c"), d("d"))) == 0


def test_lower_riemann_pair_exchange_canonicalizes_like_upstream_tensorreduce():
    u, d = declare_idx("rpairu", "rpaird", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_riemann_pair_exchange")
    UseMetric(g, (u, d))

    assert Simp(R(d("a"), d("b"), d("c"), d("d")) - R(d("c"), d("d"), d("a"), d("b"))) == 0


def test_lower_riemann_metric_contraction_over_antisymmetric_pair_vanishes_like_upstream_tensorreduce():
    u, d = declare_idx("rmetricantiu", "rmetricantid", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_riemann_metric_antisymmetric_pair")
    UseMetric(g, (u, d))

    assert Simp(g(u("a"), u("b")) * R(d("a"), d("b"), d("c"), d("d"))) == 0
    assert Simp(R(DG(1), DG(1), DG(2), DG(2))) == 0


def test_lower_riemann_algebraic_bianchi_canonicalizes_like_upstream_tensorreduce():
    u, d = declare_idx("rbianchi1u", "rbianchi1d", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_riemann_first_bianchi")
    UseMetric(g, (u, d))

    expr = (
        R(d("a"), d("b"), d("c"), d("d"))
        + R(d("a"), d("c"), d("d"), d("b"))
        + R(d("a"), d("d"), d("b"), d("c"))
    )

    assert Simp(expr) == 0


def test_second_bianchi_identity_for_lower_riemann_simplifies_like_upstream():
    u, d = declare_idx("bianchiu", "bianchid", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_bianchi")
    UseMetric(g, (u, d))

    expr = (
        CovD(R(d("a"), d("b"), d("c"), d("d")), d("e"))
        + CovD(R(d("a"), d("b"), d("d"), d("e")), d("c"))
        + CovD(R(d("a"), d("b"), d("e"), d("c")), d("d"))
    )

    assert Simp(expr) == 0


def test_presimplified_ricci_tensor_matches_upstream_metric_expression():
    u, d = declare_idx("rsu", "rsd", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_ricci_tensor")
    UseMetric(g, (u, d))

    assert Rsimp(d("m"), d("n")) == _expected_presimplified_ricci_tensor(g, u, d, "m", "n")


def test_presimplified_ricci_tensor_avoids_free_label_collisions_and_raises_indices():
    u, d = declare_idx("rsvaru", "rsvard", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_ricci_tensor_variance")
    UseMetric(g, (u, d))

    assert Rsimp(d("a"), d("b")) == _expected_presimplified_ricci_tensor(g, u, d, "a", "b", "cdef")
    assert Rsimp(u("a"), d("b")) == g(u("a"), u("c")) * _expected_presimplified_ricci_tensor(
        g, u, d, "c", "b", "defg"
    )
    assert Rsimp(d("a"), u("b")) == g(u("b"), u("c")) * _expected_presimplified_ricci_tensor(
        g, u, d, "a", "c", "defg"
    )
    assert Rsimp(u("a"), u("b")) == g(u("a"), u("c")) * g(u("b"), u("d")) * _expected_presimplified_ricci_tensor(
        g, u, d, "c", "d", "efgh"
    )
    assert Rsimp(UG(1), DG(2)) == Rsimp(u("a"), d("b"))


def test_covariant_derivative_of_scalar_mixed_tensor_and_covector_matches_upstream():
    u, d = declare_idx("cu", "cd", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_covd")
    f = tensor("f_covd")
    f1 = tensor("f1_covd")
    f2 = sp.Symbol("f2_covd")
    UseMetric(g, (u, d))

    expr = f2 * f(u("a"), d("b")) * f1(d("c"))
    expected = sp.expand(
        f(u("a"), d("b")) * f1(d("c")) * Pd(f2, d("d"))
        + f2 * f1(d("c")) * Pd(f(u("a"), d("b")), d("d"))
        + f2 * f(u("a"), d("b")) * Pd(f1(d("c")), d("d"))
        + f2 * f(u("a"), d("e")) * f1(d("c")) * _lower_connection(g, u, d, "b", "d", "e")
        + f2 * f(u("a"), d("b")) * f1(d("e")) * _lower_connection(g, u, d, "c", "d", "e")
        + f2 * f(u("e"), d("b")) * f1(d("c")) * _upper_connection(g, u, d, "a", "d", "e")
    )

    assert Simp(CovD(expr, d("d"))) == expected


def test_covariant_derivative_with_upper_derivative_index_raises_with_metric_like_upstream():
    u, d = declare_idx("cuu", "cud", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_covd_upper")
    phi = sp.Symbol("phi_covd_upper")
    UseMetric(g, (u, d))

    assert CovD(phi, u("m")) == g(u("a"), u("m")) * Pd(phi, d("a"))


def test_affine_and_covd_unsupported_signatures_remain_symbolic_like_mathematica():
    phi = sp.Symbol("phi_affine_covd_inert")
    affine_head = tensor("Affine")
    covd_head = tensor("CovD")

    assert _unsupported_call_as_value(Affine) == affine_head()
    assert Affine(DN("a"), DN("b"), DN("c")) == affine_head(DN("a"), DN("b"), DN("c"))
    assert Affine(UP("a"), UP("b"), DN("c")) == affine_head(UP("a"), UP("b"), DN("c"))
    assert _unsupported_call_as_value(CovD, phi) == covd_head(phi)
    assert _unsupported_call_as_value(CovD, phi, DE(0)) == covd_head(phi, DE(0))


def test_einstein_tensor_down_down_matches_upstream_definition():
    u, d = declare_idx("guu", "gdd", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_einstein")
    UseMetric(g, (u, d))

    assert G(d("a"), d("b")) == R(d("a"), d("b")) - g(d("a"), d("b")) * R() / 2


def test_einstein_tensor_accepts_distinct_metric_contract_slots_like_upstream():
    u, d = declare_idx("gdgu", "gdgd", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_einstein_metric_slots")
    UseMetric(g, (u, d))

    assert G(DG(1), DG(2)) == G(d("a"), d("b"))


def test_einstein_tensor_accepts_mixed_metric_contract_slots_like_upstream():
    u, d = declare_idx("gmcu", "gmcd", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_einstein_mixed_metric_slots")
    UseMetric(g, (u, d))

    assert G(UG(1), DG(2)) == G(u("a"), d("b"))


def test_scalar_field_helpers_match_upstream_definitions():
    u, d = declare_idx("xgu", "xgd", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_scalar")
    phi = sp.Symbol("phi")
    UseMetric(g, (u, d))

    assert X(phi) == -g(u("a"), u("b")) * Pd(phi, d("a")) * Pd(phi, d("b")) / 2
    assert Dsquare(phi) == MetricContract(CovD(CovD(phi, DG(1)), DG(1)))
    assert T(phi)(d("a"), d("b")) == (
        g(d("a"), d("b")) * (X(phi) - V(phi)) + Pd(phi, d("a")) * Pd(phi, d("b"))
    )


def test_scalar_stress_tensor_unsupported_signatures_remain_symbolic_like_mathematica():
    phi = sp.Symbol("phi_stress_inert")
    stress_head = tensor("T")

    assert T(phi)(UP("a"), DN("b")) == stress_head(phi, UP("a"), DN("b"))
    assert T(phi)(DN("a")) == stress_head(phi, DN("a"))


def test_scalar_operator_unsupported_signatures_remain_symbolic_like_mathematica():
    phi = sp.Symbol("phi_scalar_operator_inert")
    x_head = tensor("X")
    dsquare_head = tensor("Dsquare")

    assert _unsupported_call_as_value(X) == x_head()
    assert _unsupported_call_as_value(X, phi, DN("a")) == x_head(phi, DN("a"))
    assert _unsupported_call_as_value(Dsquare) == dsquare_head()
    assert _unsupported_call_as_value(Dsquare, phi, DN("a")) == dsquare_head(phi, DN("a"))


def test_adm_extrinsic_curvature_helpers_match_upstream_definitions():
    u, d = declare_idx("kgu", "kgd", dim=4, index_set=LatinIdx, color="Black")
    g = tensor("g_adm")
    UseMetric(g, (u, d))

    assert K(d("a"), d("b")) == (
        Pd(g(d("a"), d("b")), DE(0)) - CovD(ShiftN(d("a")), d("b")) - CovD(ShiftN(d("b")), d("a"))
    ) / (2 * LapseN)
    assert K() == MetricContract(K(DG(1), DG(1)))
    assert KK() == MetricContract(K(DG(1), DG(2)) * K(DG(1), DG(2)))
    assert RADM() == R() - K() * K() + KK()


def test_adm_helper_unsupported_signatures_remain_symbolic_like_mathematica():
    k_head = tensor("K")
    kk_head = tensor("KK")
    radm_head = tensor("RADM")

    assert K(DN("a")) == k_head(DN("a"))
    assert K(UP("a"), DN("b")) == k_head(UP("a"), DN("b"))
    assert KK(DN("a")) == kk_head(DN("a"))
    assert RADM(DN("a")) == radm_head(DN("a"))


def _upper_connection(g, u, d, free, deriv, replacement):
    return (
        -g(u(free), u("f")) * Pd(g(d(deriv), d(replacement)), d("f")) / 2
        + g(u(free), u("f")) * Pd(g(d(deriv), d("f")), d(replacement)) / 2
        + g(u(free), u("f")) * Pd(g(d(replacement), d("f")), d(deriv)) / 2
    )


def _lower_connection(g, u, d, free, deriv, replacement):
    return (
        g(u(replacement), u("f")) * Pd(g(d(free), d(deriv)), d("f")) / 2
        - g(u(replacement), u("f")) * Pd(g(d(free), d("f")), d(deriv)) / 2
        - g(u(replacement), u("f")) * Pd(g(d(deriv), d("f")), d(free)) / 2
    )


def _unsupported_call_as_value(func, *args):
    try:
        return func(*args)
    except (TypeError, ValueError) as exc:
        return exc


def _expected_ricci_scalar(g, u, d):
    a, b, c, dd, e, f = [label for label in "abcdef"]

    return (
        3 * g(u(a), u(b)) * g(u(c), u(dd)) * g(u(e), u(f)) * Pd(g(d(a), d(c)), d(e)) * Pd(g(d(b), d(dd)), d(f)) / 4
        - g(u(a), u(b)) * g(u(c), u(dd)) * g(u(e), u(f)) * Pd(g(d(a), d(c)), d(f)) * Pd(g(d(b), d(e)), d(dd)) / 2
        - g(u(a), u(b)) * g(u(c), u(dd)) * g(u(e), u(f)) * Pd(g(d(a), d(c)), d(dd)) * Pd(g(d(b), d(e)), d(f))
        - g(u(a), u(b)) * g(u(c), u(dd)) * g(u(e), u(f)) * Pd(g(d(a), d(b)), d(e)) * Pd(g(d(c), d(dd)), d(f)) / 4
        + g(u(a), u(b)) * g(u(c), u(dd)) * g(u(e), u(f)) * Pd(g(d(a), d(b)), d(dd)) * Pd(g(d(c), d(e)), d(f))
        - g(u(a), u(b)) * g(u(c), u(dd)) * Pd(Pd(g(d(a), d(b)), d(c)), d(dd))
        + g(u(a), u(b)) * g(u(c), u(dd)) * Pd(Pd(g(d(a), d(c)), d(b)), d(dd))
    )


def _expected_presimplified_ricci_tensor(g, u, d, left, right, dummy_labels="abcd"):
    a, b, c, dd = [label for label in dummy_labels]

    return (
        -g(u(a), u(b)) * g(u(c), u(dd)) * Pd(g(d(left), d(right)), d(dd)) * Pd(g(d(a), d(b)), d(c)) / 4
        + g(u(a), u(b)) * g(u(c), u(dd)) * Pd(g(d(left), d(right)), d(dd)) * Pd(g(d(a), d(c)), d(b)) / 2
        - g(u(a), u(b)) * g(u(c), u(dd)) * Pd(g(d(a), d(c)), d(dd)) * Pd(g(d(b), d(left)), d(right)) / 2
        + g(u(a), u(b)) * g(u(c), u(dd)) * Pd(g(d(a), d(right)), d(c)) * Pd(g(d(b), d(left)), d(dd)) / 2
        - g(u(a), u(b)) * g(u(c), u(dd)) * Pd(g(d(a), d(c)), d(dd)) * Pd(g(d(b), d(right)), d(left)) / 2
        + g(u(a), u(b)) * g(u(c), u(dd)) * Pd(g(d(a), d(c)), d(right)) * Pd(g(d(b), d(dd)), d(left)) / 4
        + g(u(a), u(b)) * g(u(c), u(dd)) * Pd(g(d(a), d(b)), d(dd)) * Pd(g(d(c), d(left)), d(right)) / 4
        - g(u(a), u(b)) * g(u(c), u(dd)) * Pd(g(d(a), d(right)), d(dd)) * Pd(g(d(c), d(left)), d(b)) / 2
        + g(u(a), u(b)) * g(u(c), u(dd)) * Pd(g(d(a), d(b)), d(dd)) * Pd(g(d(c), d(right)), d(left)) / 4
        - g(u(a), u(b)) * Pd(Pd(g(d(left), d(right)), d(a)), d(b)) / 2
        + g(u(a), u(b)) * Pd(Pd(g(d(a), d(left)), d(right)), d(b)) / 2
        + g(u(a), u(b)) * Pd(Pd(g(d(a), d(right)), d(left)), d(b)) / 2
        - g(u(a), u(b)) * Pd(Pd(g(d(a), d(b)), d(left)), d(right)) / 2
    )
