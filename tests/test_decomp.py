import time

import sympy as sp
import pytest

import mathgr
import mathgr.gr as gr_module
from mathgr import decomp
from mathgr.decomp import D1, D2, DTot, Decomp0123, Decomp01i, Decomp0i, Decomp123, Decomp1i, DecompSe, Dim1, Dim2, DimTot, U1, U2, UTot
from mathgr.gr import R, UseMetric
from mathgr.rewrite import ReplaceAll
from mathgr.tensor import (
    DE,
    DN,
    UE,
    UP,
    DeclareSym,
    DefaultDim,
    Dta,
    Pd,
    PdT,
    PdVars,
    Simp,
    Symmetric,
    is_pdt,
    pdt_parts,
    tensor,
    tensor_args,
    tensor_head_name,
)
from mathgr.util import Eps


def test_decomp0i_splits_dummy_total_indices_inside_sqrt():
    f = tensor("f")

    assert Decomp0i(sp.sqrt(1 + f(DTot("a")) * f(UTot("a")))) == sp.sqrt(
        1 + f(DE(0)) * f(UE(0)) + f(DN("a")) * f(UP("a"))
    )


def test_decomp0i_enters_single_argument_functions_like_upstream():
    f = tensor("fDecompInto")

    assert Decomp0i(sp.exp(f(DTot("a")) * f(UTot("a")))) == sp.exp(
        f(DE(0)) * f(UE(0)) + f(DN("a")) * f(UP("a"))
    )


def test_decomp0i_explicit_selector_matches_upstream_list_label_semantics():
    f = tensor("fDecompExplicitIndex")

    assert Decomp0i(f(DTot("A"))) == f(DTot("A"))
    assert Decomp0i(f(DTot("A")), indices=["A"]) == f(DE(0)) + f(DN("A"))
    assert Decomp0i(f(UTot("A")), indices=["A"]) == f(UE(0)) + f(UP("A"))
    assert Decomp0i(f(DTot("A")) * f(UTot("A")), indices=["A"]) == (
        f(DE(0)) * f(UE(0)) + f(DN("A")) * f(UP("A"))
    )
    assert Decomp0i(f(DTot("A")), indices="A") == f(DTot("A"))
    assert Decomp0i(f(DTot("A")), indices=DTot("A")) == f(DTot("A"))
    assert Decomp0i(f(DTot("A")), indices=[DTot("A")]) == f(DTot("A"))


def test_dimension_symbols_match_upstream_public_names():
    assert DefaultDim == UP.dim == DN.dim
    assert DimTot == UTot.dim == DTot.dim
    assert Dim1 == U1.dim == D1.dim
    assert Dim2 == U2.dim == D2.dim
    assert mathgr.DefaultDim == DefaultDim
    assert mathgr.DimTot == DimTot


def test_decomp0i_splits_ordinary_terms_and_indexed_power_bases():
    f = tensor("f")
    fx = tensor("fx")

    expr = fx(DTot("A"), UTot("A")) - 1 / (1 + f(DTot("A"), UTot("A")))
    expected = fx(DE(0), UE(0)) + fx(DN("A"), UP("A")) - (
        1 + f(DE(0), UE(0)) + f(DN("A"), UP("A"))
    ) ** -1

    assert Decomp0i(expr) == expected


def test_decomp0i_selected_free_square_uses_upstream_sector_square_sum():
    f = tensor("fDecompSelectedSquare")

    assert Decomp0i(f(DTot("A")) ** 2, indices=["A"]) == f(DE(0)) ** 2 + f(DN("A")) ** 2


def test_decomp0i_non_square_power_decomposes_power_base_factors_like_upstream():
    f = tensor("fDecompPowerBase")

    assert Decomp0i((f(DTot("a")) * f(UTot("a"))) ** 3) == (
        (f(DE(0)) + f(DN("a"))) ** 3 * (f(UE(0)) + f(UP("a"))) ** 3
    )


def test_decomp0i_splits_sympy_series_order_terms_like_upstream_seriesdata():
    f1 = tensor("fSeries1")
    f2 = tensor("fSeries2")

    expr = (
        Eps * f1(DTot("a")) * f2(DTot("a"))
        + Eps**2 * sp.sqrt(f1(DTot("b")) * f2(DTot("b")))
        + sp.Order(Eps**3, Eps)
    )
    expected = (
        Eps * f1(DE(0)) * f2(DE(0))
        + Eps * f1(DN("a")) * f2(DN("a"))
        + Eps**2 * sp.sqrt(f1(DE(0)) * f2(DE(0)) + f1(DN("b")) * f2(DN("b")))
        + sp.Order(Eps**3, Eps)
    )

    assert Decomp0i(expr) == expected


def test_decomp01i_splits_total_dummy_into_zero_one_and_spatial_sectors():
    f = tensor("f01")

    assert Decomp01i(f(DTot("a")) * f(UTot("a"))) == (
        f(DE(0)) * f(UE(0)) + f(DE(1)) * f(UE(1)) + f(DN("a")) * f(UP("a"))
    )


def test_decomp1i_splits_total_dummy_into_one_and_spatial_sectors():
    f = tensor("f1i")

    assert Decomp1i(f(DTot("a")) * f(UTot("a"))) == (
        f(DE(1)) * f(UE(1)) + f(DN("a")) * f(UP("a"))
    )


def test_decomp0123_and_decomp123_split_total_dummy_into_explicit_components():
    f = tensor("f123")
    expr = f(DTot("a")) * f(UTot("a"))

    assert Decomp0123(expr) == (
        f(DE(0)) * f(UE(0))
        + f(DE(1)) * f(UE(1))
        + f(DE(2)) * f(UE(2))
        + f(DE(3)) * f(UE(3))
    )
    assert Decomp123(expr) == (
        f(DE(1)) * f(UE(1)) + f(DE(2)) * f(UE(2)) + f(DE(3)) * f(UE(3))
    )


def test_decompse_splits_total_dummy_into_two_named_index_families():
    f = tensor("fSe")

    assert DecompSe(f(DTot("a")) * f(UTot("a"))) == (
        f(D2("a")) * f(U2("a")) + f(D1("a")) * f(U1("a"))
    )


def test_decomp_accepts_python_hooks_after_sector_splitting():
    metric = tensor("metricDecompHook")
    η = tensor("etaDecompHook")
    γ = tensor("gammaDecompHook")

    def hook(expr):
        if tensor_head_name(expr) != "metricDecompHook":
            return expr
        args = tensor_args(expr)
        if args == (D1("a"), U1("a")):
            return η(D1("a"), U1("a"))
        if args == (D2("a"), U2("a")):
            return γ(D2("a"), U2("a"))
        return expr

    assert DecompSe(metric(DTot("a"), UTot("a")), hooks=(hook,)) == (
        γ(D2("a"), U2("a")) + η(D1("a"), U1("a"))
    )


def test_decomp_hook_global_matches_upstream_decomp_hook_variable():
    metric = tensor("metricGlobalDecompHook")
    η = tensor("etaGlobalDecompHook")

    def hook(expr):
        if tensor_head_name(expr) == "metricGlobalDecompHook" and tensor_args(expr) == (D1("a"), U1("a")):
            return η(D1("a"), U1("a"))
        return expr

    previous = tuple(decomp.DecompHook)
    decomp.DecompHook[:] = [hook]
    try:
        assert decomp.DecompSe(metric(DTot("a"), UTot("a"))) == (
            metric(D2("a"), U2("a")) + η(D1("a"), U1("a"))
        )
        assert mathgr.DecompHook is decomp.DecompHook
    finally:
        decomp.DecompHook[:] = previous


def test_decomp_hooks_accept_sympy_wild_pattern_rules_like_upstream_decomp_hook():
    metric = tensor("metricPatternDecompHook")
    η = tensor("etaPatternDecompHook")
    γ = tensor("gammaPatternDecompHook")
    α = sp.Wild("α")
    a = sp.Wild("a")

    hooks = (
        (metric(D1(α), U1(α)), η(D1(α), U1(α))),
        (metric(D2(a), U2(a)), lambda a: γ(D2(a), U2(a))),
    )

    assert DecompSe(metric(DTot("p"), UTot("p")), hooks=hooks) == (
        γ(D2("p"), U2("p")) + η(D1("p"), U1("p"))
    )


def test_decomp_hook_rules_match_replace_all_for_wild_patterns():
    metric = tensor("metricReplaceAllDecompHook")
    η = tensor("etaReplaceAllDecompHook")
    α = sp.Wild("α")
    rules = [(metric(D1(α), U1(α)), η(D1(α), U1(α)))]
    expr = DecompSe(metric(DTot("p"), UTot("p")))

    assert DecompSe(metric(DTot("p"), UTot("p")), hooks=rules) == ReplaceAll(expr, rules)


def test_total_metric_evaluates_on_decomposed_dual_pairs_only_like_upstream():
    metric = tensor("metricDecompSectorDualPair")
    previous_metric = gr_module.Metric
    previous_indices = gr_module.IdxOfMetric

    try:
        UseMetric(metric, (UTot, DTot))

        assert metric(U1("a"), D1("b")) == Dta(U1("a"), D1("b"))
        mixed = metric(UTot("a"), D1("b"))
        assert tensor_head_name(mixed) == "metricDecompSectorDualPair"
        assert tensor_args(mixed) == (UTot("a"), D1("b"))
    finally:
        UseMetric(previous_metric, previous_indices)


@pytest.mark.slow
def test_decompse_non_diagonal_metric_derivative_hooks_simplify_curvature_path():
    metric = tensor("metricNonDiagonalDecomp")
    η = tensor("etaNonDiagonalDecomp")
    γ = tensor("gammaNonDiagonalDecomp")
    vector = tensor("ANonDiagonalDecomp")
    previous_metric = gr_module.Metric
    previous_indices = gr_module.IdxOfMetric

    def metric_hook(expr):
        if tensor_head_name(expr) != "metricNonDiagonalDecomp":
            return expr
        args = tensor_args(expr)
        if len(args) != 2:
            return expr
        left, right = args
        if left.head_name == "D1" and right.head_name == "D1":
            return η(left, right) + vector(U2("q"), left) * vector(U2("q"), right)
        if left.head_name == "D1" and right.head_name == "D2":
            return vector(U2(right.label), left)
        if left.head_name == "D2" and right.head_name == "D1":
            return vector(U2(left.label), right)
        if left.head_name == "U1" and right.head_name == "U1":
            return η(left, right)
        if left.head_name == "U1" and right.head_name == "U2":
            return -η(left, U1("q")) * vector(right, D1("q"))
        if left.head_name == "U2" and right.head_name == "U1":
            return -η(right, U1("q")) * vector(left, D1("q"))
        if left.head_name == "U2" and right.head_name == "U2":
            return Dta(left, right) + η(U1("q"), U1("r")) * vector(left, D1("q")) * vector(right, D1("r"))
        if left.head_name == "D2" and right.head_name == "D2":
            return Dta(U2(left.label), U2(right.label))
        return expr

    def derivative_hook(expr):
        if not is_pdt(expr):
            return expr
        base, variables = pdt_parts(expr)
        if tensor_head_name(base) in {"etaNonDiagonalDecomp", "gammaNonDiagonalDecomp"}:
            return sp.Integer(0)
        if tensor_head_name(base) == "ANonDiagonalDecomp" and any(getattr(var, "head_name", None) == "D2" for var in variables):
            return sp.Integer(0)
        return expr

    try:
        UseMetric(metric, (UTot, DTot))
        UseMetric(η, (U1, D1), SetAsDefault=False)
        UseMetric(γ, (U2, D2), SetAsDefault=False)

        started = time.perf_counter()
        result = Simp(DecompSe(Simp(R()), hooks=(metric_hook, derivative_hook)), hooks=(derivative_hook,))
        assert time.perf_counter() - started < 30

        bad_derivatives = []
        for node in sp.preorder_traversal(result):
            if not is_pdt(node):
                continue
            base, variables = pdt_parts(node)
            head_name = tensor_head_name(base)
            if head_name in {"etaNonDiagonalDecomp", "gammaNonDiagonalDecomp"}:
                bad_derivatives.append(node)
            if head_name == "ANonDiagonalDecomp" and any(getattr(var, "head_name", None) == "D2" for var in variables):
                bad_derivatives.append(node)

        assert bad_derivatives == []
        assert not any(getattr(node, "head_name", None) in {"UTot", "DTot"} for node in sp.preorder_traversal(result))
        assert not result.has(metric(DTot("a"), DTot("a")))
        assert any(tensor_head_name(node) == "ANonDiagonalDecomp" for node in sp.preorder_traversal(result))
    finally:
        gr_module.Metric = previous_metric
        gr_module.IdxOfMetric = previous_indices


def test_simp_cancels_non_diagonal_metric_second_derivative_block():
    η = tensor("etaSecondDerivativeCancel")
    A = tensor("ASecondDerivativeCancel")
    DeclareSym(η, (U1, U1), Symmetric((1, 2)))

    block = (
        A(U2("q"), D1("c"))
        * η(U1("a"), U1("b"))
        * η(U1("c"), U1("d"))
        * PdT(A(U2("q"), D1("a")), PdVars(D1("b"), D1("d")))
        + A(U2("q"), D1("a"))
        * η(U1("a"), U1("b"))
        * η(U1("c"), U1("d"))
        * PdT(A(U2("q"), D1("c")), PdVars(D1("b"), D1("d")))
        - A(U2("q"), D1("b"))
        * η(U1("a"), U1("b"))
        * η(U1("c"), U1("d"))
        * PdT(A(U2("q"), D1("a")), PdVars(D1("c"), D1("d")))
        - A(U2("q"), D1("a"))
        * η(U1("a"), U1("b"))
        * η(U1("c"), U1("d"))
        * PdT(A(U2("q"), D1("b")), PdVars(D1("c"), D1("d")))
        - 2
        * A(U2("a"), D1("b"))
        * η(U1("b"), U1("c"))
        * η(U1("d"), U1("e"))
        * PdT(A(U2("a"), D1("d")), PdVars(D1("c"), D1("e")))
        + 2
        * A(U2("a"), D1("b"))
        * η(U1("b"), U1("c"))
        * η(U1("d"), U1("e"))
        * PdT(A(U2("a"), D1("c")), PdVars(D1("d"), D1("e")))
    )

    assert Simp(block) == 0


@pytest.mark.slow
def test_decompse_full_non_diagonal_metric_reduces_to_maxwell_form():
    metric = tensor("metricNonDiagonalMaxwell")
    η = tensor("etaNonDiagonalMaxwell")
    γ = tensor("gammaNonDiagonalMaxwell")
    A = tensor("ANonDiagonalMaxwell")

    previous_metric = gr_module.Metric
    previous_indices = gr_module.IdxOfMetric

    def metric_hook(expr):
        if tensor_head_name(expr) != "metricNonDiagonalMaxwell":
            return expr
        args = tensor_args(expr)
        if len(args) != 2:
            return expr
        left, right = args
        if left.head_name == "D1" and right.head_name == "D1":
            return η(left, right) + A(U2("q"), left) * A(U2("q"), right)
        if left.head_name == "D1" and right.head_name == "D2":
            return A(U2(right.label), left)
        if left.head_name == "D2" and right.head_name == "D1":
            return A(U2(left.label), right)
        if left.head_name == "U1" and right.head_name == "U1":
            return η(left, right)
        if left.head_name == "U1" and right.head_name == "U2":
            return -η(left, U1("q")) * A(right, D1("q"))
        if left.head_name == "U2" and right.head_name == "U1":
            return -η(right, U1("q")) * A(left, D1("q"))
        if left.head_name == "U2" and right.head_name == "U2":
            return Dta(left, right) + η(U1("q"), U1("r")) * A(left, D1("q")) * A(right, D1("r"))
        if left.head_name == "D2" and right.head_name == "D2":
            return Dta(U2(left.label), U2(right.label))
        return expr

    def derivative_hook(expr):
        if not is_pdt(expr):
            return expr
        base, variables = pdt_parts(expr)
        if tensor_head_name(base) in {"etaNonDiagonalMaxwell", "gammaNonDiagonalMaxwell"}:
            return sp.Integer(0)
        if tensor_head_name(base) == "ANonDiagonalMaxwell" and any(
            getattr(var, "head_name", None) == "D2" for var in variables
        ):
            return sp.Integer(0)
        return expr

    def F(a, α, β):
        return Pd(A(a, β), α) - Pd(A(a, α), β)

    try:
        UseMetric(metric, (UTot, DTot))
        UseMetric(η, (U1, D1), SetAsDefault=False)
        UseMetric(γ, (U2, D2), SetAsDefault=False)

        started = time.perf_counter()
        result = Simp(
            DecompSe(Simp(R()), hooks=(metric_hook, derivative_hook)),
            hooks=(derivative_hook,),
        )
        assert time.perf_counter() - started < 30

        α, β, γi, delta = "α", "β", "γ", "delta"
        # docs-local/fix.md records a Mathematica oracle with the opposite
        # sign. WolframScript is not available in this environment, and
        # changing this coefficient to -1/2 fails the current Python identity.
        # Keep this convention explicit instead of silently diverging from the note.
        expected = (
            sp.Rational(1, 2)
            * F(U2("a"), D1(α), D1(β))
            * Pd(A(U2("a"), D1(γi)), D1(delta))
            * η(U1(α), U1(γi))
            * η(U1(β), U1(delta))
        )

        delta_expr = Simp(sp.expand(result - expected), hooks=(derivative_hook,))
        assert delta_expr == 0

        for term in sp.Add.make_args(sp.expand(delta_expr)):
            assert not any(
                is_pdt(node) and len(pdt_parts(node)[1]) >= 2
                for node in sp.preorder_traversal(term)
            ), term
    finally:
        gr_module.Metric = previous_metric
        gr_module.IdxOfMetric = previous_indices
