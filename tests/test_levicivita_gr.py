from mathgr.gr import UseMetric
from mathgr.tensor import DN, Dta, DtaGen, LatinIdx, LeviCivita, Simp, declare_idx, tensor


def test_levicivita_opposite_variance_products_reduce_to_generalized_delta():
    u3, d3 = declare_idx("u3", "d3", dim=3, index_set=LatinIdx, color="Brown")

    expr = LeviCivita(u3("a"), u3("b"), u3("c")) * LeviCivita(d3("d"), d3("e"), d3("f"))
    assert Simp(expr) == DtaGen(u3("a"), u3("b"), u3("c"), d3("d"), d3("e"), d3("f"))

    assert Simp(LeviCivita(u3("a"), u3("b"), u3("c")) * LeviCivita(d3("a"), d3("d"), d3("e"))) == (
        Dta(u3("b"), d3("d")) * Dta(u3("c"), d3("e"))
        - Dta(u3("b"), d3("e")) * Dta(u3("c"), d3("d"))
    )
    assert Simp(LeviCivita(u3("a"), u3("b"), u3("c")) * LeviCivita(d3("a"), d3("b"), d3("d"))) == (
        2 * Dta(u3("c"), d3("d"))
    )
    assert Simp(LeviCivita(u3("a"), u3("b"), u3("c")) * LeviCivita(d3("a"), d3("b"), d3("c"))) == 6


def test_levicivita_products_contract_before_fast_simp_method_boundary():
    u3, d3 = declare_idx("u3fast", "d3fast", dim=3, index_set=LatinIdx, color="Brown")

    expr = LeviCivita(u3("a"), u3("b"), u3("c")) * LeviCivita(d3("a"), d3("b"), d3("d"))

    assert Simp(expr, Method="Fast") == 2 * Dta(u3("c"), d3("d"))


def test_levicivita_repeated_indices_vanish_by_declared_antisymmetry():
    u3, _d3 = declare_idx("u3repeat", "d3repeat", dim=3, index_set=LatinIdx, color="Brown")

    assert Simp(LeviCivita(u3("a"), u3("a"), u3("b"))) == 0


def test_levicivita_default_dimension_signatures_remain_symbolic_like_mathematica():
    levi = tensor("LeviCivita")

    assert LeviCivita() == levi()
    assert LeviCivita(DN("a"), DN("a")) == levi(DN("a"), DN("a"))
    assert LeviCivita(DN("a"), DN("a"), DN("b")) == levi(DN("a"), DN("a"), DN("b"))


def test_levicivita_same_variance_products_use_registered_metric():
    u3, d3 = declare_idx("u3m", "d3m", dim=3, index_set=LatinIdx, color="Brown")
    g3 = tensor("g3")
    UseMetric(g3, (u3, d3))

    expr = LeviCivita(u3("a"), u3("b"), u3("c")) * LeviCivita(u3("d"), u3("e"), u3("f"))
    assert Simp(expr) == (
        -g3(u3("a"), u3("f")) * g3(u3("b"), u3("e")) * g3(u3("c"), u3("d"))
        + g3(u3("a"), u3("e")) * g3(u3("b"), u3("f")) * g3(u3("c"), u3("d"))
        + g3(u3("a"), u3("f")) * g3(u3("b"), u3("d")) * g3(u3("c"), u3("e"))
        - g3(u3("a"), u3("d")) * g3(u3("b"), u3("f")) * g3(u3("c"), u3("e"))
        - g3(u3("a"), u3("e")) * g3(u3("b"), u3("d")) * g3(u3("c"), u3("f"))
        + g3(u3("a"), u3("d")) * g3(u3("b"), u3("e")) * g3(u3("c"), u3("f"))
    )


def test_use_metric_evaluates_only_dual_index_pairs_to_delta():
    u, d = declare_idx("um", "dm", dim=3, index_set=LatinIdx, color="Black")
    ux, dx = declare_idx("ux", "dx", dim=3, index_set=LatinIdx, color="Black")
    metric = tensor("metric_m")

    UseMetric(metric, (u, d))

    assert metric(u("a"), d("b")) == Dta(u("a"), d("b"))
    assert metric(d("a"), u("b")) == Dta(d("a"), u("b"))
    assert metric(u("a"), dx("b")).func == metric(u("a"), dx("b")).func
    assert metric(u("a"), dx("b")) != Dta(u("a"), dx("b"))
