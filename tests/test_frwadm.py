import sympy as sp

from mathgr.decomp import DTot
from mathgr.frwadm import (
    Fourier2,
    DecompG2H,
    H,
    K,
    KK,
    LapseN,
    RADM,
    ShiftN,
    Simp as FRWSimp,
    Sqrtg,
    a,
    alpha,
    b,
    beta,
    epsilon,
    eta,
    eta2,
    eta3,
    h,
    k,
    zeta,
)
from mathgr.gr import R
from mathgr.tensor import DE, DN, UE, UP, Dta, Pd, PdT, PdVars, tensor, tensor_head_name
from mathgr.util import Eps


def test_fourier2_replaces_repeated_spatial_derivatives_with_minus_k_squared():
    f = sp.Symbol("f")

    assert Fourier2(PdT(f, PdVars(DN("i"), DN("i"), DE(0)))) == -k**2 * PdT(f, PdVars(DE(0)))


def test_fourier2_replaces_paired_spatial_gradients_with_k_squared():
    f, g = sp.symbols("f g")

    assert Fourier2(PdT(f, PdVars(DN("i"))) * PdT(g, PdVars(DN("i"), DE(0)))) == (
        k**2 * f * PdT(g, PdVars(DE(0)))
    )


def test_fourier2_replaces_squared_spatial_gradient_with_k_squared():
    f = sp.Symbol("f")

    assert Fourier2(PdT(f, PdVars(DN("i"), DE(0))) ** 2) == k**2 * PdT(f, PdVars(DE(0))) ** 2


def test_fourier2_replaces_gradient_vector_contraction_with_momentum_vector():
    f = sp.Symbol("f")
    vector = tensor("vector_fourier")

    assert Fourier2(PdT(f, PdVars(DN("i"), DE(0))) * vector(DN("i"))) == (
        -sp.I * tensor("k")(DN("i")) * PdT(f, PdVars(DE(0))) * vector(DN("i"))
    )


def test_frw_shift_vector_is_transverse_like_upstream():
    f = sp.Symbol("f")

    assert FRWSimp(PdT(b(DN("i")), PdVars(DN("i")))) == 0
    assert Fourier2(tensor("k")(DN("i")) * b(DN("i"))) == 0
    assert Fourier2(PdT(f, PdVars(DN("i"), DE(0))) * b(DN("i"))) == 0


def test_frw_background_lapse_shift_sqrtg_and_spatial_metric_definitions():
    assert LapseN == 1 + Eps * alpha
    assert ShiftN(DN("i")) == Eps * Pd(beta, DN("i")) + Eps * b(DN("i"))
    assert Sqrtg == LapseN * sp.exp(3 * Eps * zeta) * a**3
    assert h(DN("i"), DN("j")) == a**2 * sp.exp(2 * Eps * zeta) * Dta(DN("i"), DN("j"))
    assert h(UP("i"), UP("j")) == sp.exp(-2 * Eps * zeta) * Dta(DN("i"), DN("j")) / a**2


def test_frw_shiftn_unsupported_signatures_remain_symbolic_like_mathematica():
    shift_head = tensor("ShiftN")

    assert ShiftN(UP("a")) == shift_head(UP("a"))
    assert ShiftN() == shift_head()


def test_decompg2h_replaces_decomposed_four_metric_components():
    g = tensor("g")

    assert DecompG2H(g(DTot("i"), DTot("i"))) == (
        -LapseN**2
        + h(UP("a"), UP("b")) * ShiftN(DN("a")) * ShiftN(DN("b"))
        + h(DN("i"), DN("i"))
    )
    assert DecompG2H(g(DN("i"), DN("j"))) == h(DN("i"), DN("j"))
    assert DecompG2H(g(DE(0), DN("i"))) == ShiftN(DN("i"))
    assert DecompG2H(g(UE(0), UE(0))) == -LapseN**-2


def test_decompg2h_replaces_mixed_time_space_components_in_either_order():
    g = tensor("g")

    assert DecompG2H(g(DN("i"), DE(0))) == ShiftN(DN("i"))
    assert DecompG2H(g(UP("i"), UE(0))) == h(UP("i"), UP("a")) * ShiftN(DN("a")) / LapseN**2


def test_decompg2h_accepts_lazy_metric_expressions_like_upstream_holdall():
    assert DecompG2H(lambda: R()).has(h(DN("a"), DN("b")))


def test_frw_radm_uses_spatial_metric_without_generic_metric_leak():
    from examples import second_order_pert

    radm_heads = {tensor_head_name(node) for node in sp.preorder_traversal(RADM())}
    action_heads = {tensor_head_name(node) for node in sp.preorder_traversal(second_order_pert.action_density(simplify=False))}

    assert "g" not in radm_heads
    assert "h" not in radm_heads
    assert "g" not in action_heads
    assert "h" not in action_heads


def test_frw_adm_helper_unsupported_signatures_remain_symbolic_like_mathematica():
    k_head = tensor("K")
    kk_head = tensor("KK")
    radm_head = tensor("RADM")

    assert K(DN("a")) == k_head(DN("a"))
    assert K(UP("a"), DN("b")) == k_head(UP("a"), DN("b"))
    assert KK(DN("a")) == kk_head(DN("a"))
    assert RADM(DN("a")) == radm_head(DN("a"))


def test_frw_simp_applies_background_time_derivative_hooks():
    assert FRWSimp(Pd(a, DE(0))) == a * H
    assert FRWSimp(PdT(a, PdVars(DE(0), DE(0)))) == a * H**2 - a * H**2 * epsilon
    assert FRWSimp(Pd(H, DE(0))) == -epsilon * H**2
    assert FRWSimp(PdT(H, PdVars(DE(0), DE(0)))) == 2 * H**3 * epsilon**2 - H**3 * epsilon * eta
    assert FRWSimp(PdT(H, PdVars(DE(0), DE(0), DE(0)))) == (
        -6 * H**4 * epsilon**3
        + 7 * H**4 * epsilon**2 * eta
        - H**4 * epsilon * eta**2
        - H**4 * epsilon * eta * eta2
    )
    assert FRWSimp(Pd(epsilon, DE(0))) == H * epsilon * eta
    assert FRWSimp(Pd(eta, DE(0))) == H * eta2 * eta
    assert FRWSimp(Pd(eta2, DE(0))) == H * eta3 * eta2


def test_frw_simp_applies_background_spatial_derivative_and_dimension_hooks():
    assert FRWSimp(Pd(a, DN("i"))) == 0
    assert FRWSimp(Pd(H, DN("i"))) == 0
    assert FRWSimp(Pd(epsilon, DN("i"))) == 0
    assert FRWSimp(Pd(eta, DN("i"))) == 0
    assert FRWSimp(Dta(DN("i"), DN("i"))) == 3


def test_frw_momentum_magnitude_is_partial_derivative_constant_like_upstream():
    assert Pd(k, DN("i")) == 0
    assert PdT(k, PdVars(DN("i"), DE(0))) == 0
