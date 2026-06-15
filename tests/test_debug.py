import importlib

import sympy as sp

from mathgr.state import isolated_state
from mathgr.tensor import DN, UP, Dta, Simp, tensor


def test_simplify_profile_returns_result_phases_and_hot_counters():
    from mathgr.debug import simplify_profile

    with isolated_state():
        f = tensor("debugProfileVector")
        expr = Dta(UP("a"), DN("b")) * f(UP("b"))

        profile = simplify_profile(expr)

    assert profile.result == Simp(expr)
    assert profile.total_seconds >= 0
    assert [phase.name for phase in profile.phases] == ["input", "total_simp"]
    assert profile.phases[0].output_terms == 1
    assert profile.phases[1].elapsed_seconds >= 0
    assert profile.counters["sp.expand"] >= 1
    assert profile.counters["_apply_simp_hooks"] >= 1
    assert profile.counters["_split_indexed_powers"] >= 1
    assert profile.counters["_canonicalize_tensor_product"] >= 1
    assert profile.timings["sp.expand"] >= 0


def test_simplify_profile_counts_add_term_selection():
    from mathgr.debug import simplify_profile

    with isolated_state():
        f = tensor("debugProfileAddVector")
        expr = Dta(UP("a"), DN("b")) * f(UP("b")) + Dta(UP("a"), DN("c")) * f(UP("c"))

        profile = simplify_profile(expr)

    assert profile.result == 2 * f(UP("a"))
    assert profile.counters["_selected_simp_terms"] >= 1


def test_simplify_profile_restores_wrapped_functions_after_success():
    from mathgr.debug import simplify_profile

    tensor_module = importlib.import_module("mathgr.tensor")
    original_tensor_product = tensor_module._canonicalize_tensor_product
    original_expand = sp.expand

    with isolated_state():
        f = tensor("debugProfileRestoreVector")
        simplify_profile(Dta(UP("a"), DN("b")) * f(UP("b")))

    assert tensor_module._canonicalize_tensor_product is original_tensor_product
    assert sp.expand is original_expand


def test_simplify_profile_trace_env_prints_compact_report(monkeypatch, capsys):
    from mathgr.debug import simplify_profile

    monkeypatch.setenv("MATHGR_TRACE_SIMPLIFY", "1")

    with isolated_state():
        f = tensor("debugProfileTraceVector")
        simplify_profile(Dta(UP("a"), DN("b")) * f(UP("b")))

    captured = capsys.readouterr()
    assert "MATHGR simplify profile" in captured.err
    assert "total_simp" in captured.err
    assert "_canonicalize_tensor_product" in captured.err
