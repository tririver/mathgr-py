"""Python port of MathGR's ``2nd_order_GW_pert.nb`` example."""

import sympy as sp

from mathgr.decomp import DTot, UTot, Decomp0i
from mathgr.gr import MetricContract, R, WithMetric
from mathgr.tensor import (
    DE,
    DN,
    UE,
    UP,
    DeclareSym,
    DefaultDim,
    Dta,
    Index,
    LatinIdx,
    Pd,
    PdT,
    PdVars,
    Simp as _TensorSimp,
    Symmetric,
    is_pdt,
    pdt_parts,
    tensor,
    tensor_args,
    tensor_head_name,
)
from mathgr.util import Eps, OO


a = sp.Symbol("a")
H = sp.Symbol("H")
ε = sp.Symbol("ε")
η = sp.Symbol("η")
η2 = sp.Symbol("η2")
η3 = sp.Symbol("η3")
Mp = sp.Symbol("Mp")

g = tensor("g")
_gamma_raw = tensor("gammaGW")
_h_raw = tensor("hGW")
DeclareSym(_gamma_raw, (DN, DN), Symmetric("All"))

LapseN = sp.Integer(1)
Sqrtg = a**3


def ShiftN(index):
    sp.sympify(index)
    return sp.Integer(0)


def γ(first, second):
    first = sp.sympify(first)
    second = sp.sympify(second)
    if _is_index(first, DN) and _is_index(second, DN) and first.label == second.label:
        return sp.Integer(0)
    return _gamma_raw(first, second)


def h(first, second):
    first = sp.sympify(first)
    second = sp.sympify(second)
    if _is_index(first, DN) and _is_index(second, DN):
        dummy = DN(_fresh_label(first, second))
        return a**2 * (
            Dta(first, second)
            + Eps * γ(first, second)
            + Eps**2 * γ(first, dummy) * γ(dummy, second) / 2
        )
    if _is_index(first, UP) and _is_index(second, UP):
        first_down = DN(first.label)
        second_down = DN(second.label)
        dummy = DN(_fresh_label(first, second))
        return (
            Dta(first_down, second_down)
            - Eps * γ(first_down, second_down)
            + Eps**2 * γ(first_down, dummy) * γ(dummy, second_down) / 2
        ) / a**2
    if isinstance(first, Index) and isinstance(second, Index) and first.head.dual_name == second.head_name:
        return Dta(first, second)
    return _h_raw(first, second)


def Simp(expr, **options):
    hooks = tuple(options.pop("hooks", ())) + (_background_simp_hook,)
    return _TensorSimp(expr, hooks=hooks, **options)


def DecompG2H(expr):
    def evaluate():
        value = expr() if callable(expr) else expr
        return MetricContract(value)

    decomposed = WithMetric(g, (UTot, DTot), evaluate)
    return _replace_metric_components(Decomp0i(decomposed))


def second_order_action():
    return Simp(_notebook_quadratic_action())


def derived_second_order_action():
    return Simp(OO(2, op=Simp)(Sqrtg * DecompG2H(lambda: Mp**2 * R() / 2)))


def _notebook_quadratic_action():
    return (
        Mp**2 * a**3 * Pd(γ(DN("a"), DN("b")), DE(0)) ** 2 / 8
        - Mp**2 * a * Pd(γ(DN("a"), DN("b")), DN("c")) ** 2 / 8
    )


def main(*, compute_action=True):
    results = {
        "sqrtg": Sqrtg,
        "shift": ShiftN(DN("i")),
        "spatial_metric_down": h(DN("i"), DN("j")),
        "spatial_metric_up": h(UP("i"), UP("j")),
        "metric_00": _metric_component_replacement(DE(0), DE(0)),
        "metric_0i": _metric_component_replacement(DE(0), DN("i")),
        "inverse_metric_00": _metric_component_replacement(UE(0), UE(0)),
        "inverse_metric_0i": _metric_component_replacement(UE(0), UP("i")),
    }
    if compute_action:
        results["s_full"] = second_order_action()
    return results


def _replace_metric_components(expr):
    expr = sp.sympify(expr)
    if tensor_head_name(expr) == "g":
        args = tensor_args(expr)
        if len(args) == 2:
            replacement = _metric_component_replacement(args[0], args[1])
            if replacement is not None:
                return replacement
    if isinstance(expr, Index) or not getattr(expr, "args", ()):
        return expr
    rewritten_args = tuple(_replace_metric_components(arg) for arg in expr.args)
    if rewritten_args == expr.args:
        return expr
    return expr.func(*rewritten_args)


def _metric_component_replacement(first, second):
    if _is_index(first, DN) and _is_index(second, DN):
        return h(first, second)
    if first == DE(0) and second == DE(0):
        return -LapseN**2
    if (first == DE(0) and _is_index(second, DN)) or (_is_index(first, DN) and second == DE(0)):
        return ShiftN(second if first == DE(0) else first)
    if _is_index(first, UP) and _is_index(second, UP):
        return h(first, second)
    if first == UE(0) and second == UE(0):
        return -LapseN**-2
    if (first == UE(0) and _is_index(second, UP)) or (_is_index(first, UP) and second == UE(0)):
        return sp.Integer(0)
    return None


def _background_simp_hook(expr):
    expr = sp.sympify(expr)
    replacement = _background_replacement(expr)
    if replacement is not None:
        return replacement
    if isinstance(expr, Index) or not getattr(expr, "args", ()):
        return expr.xreplace({DefaultDim: sp.Integer(3)})
    rewritten_args = tuple(_background_simp_hook(arg) for arg in expr.args)
    rebuilt = expr if rewritten_args == expr.args else expr.func(*rewritten_args)
    return rebuilt.xreplace({DefaultDim: sp.Integer(3)})


def _background_replacement(expr):
    if expr == DefaultDim:
        return sp.Integer(3)
    if _is_gamma_trace(expr):
        return sp.Integer(0)
    if not is_pdt(expr):
        return None
    base, derivative_indices = pdt_parts(expr)
    if base == Mp:
        return sp.Integer(0)
    if _is_gamma_trace(base):
        return sp.Integer(0)
    if base in {a, H, ε, η} and any(_is_index(index, DN) for index in derivative_indices):
        return sp.Integer(0)
    if _is_transverse_gamma_derivative(base, derivative_indices):
        return sp.Integer(0)
    if tuple(derivative_indices) == (DE(0),):
        if base == a:
            return a * H
        if base == H:
            return -ε * H**2
        if base == ε:
            return H * ε * η
        if base == η:
            return H * η2 * η
        if base == η2:
            return H * η3 * η2
    if tuple(derivative_indices) == (DE(0), DE(0)):
        if base == a:
            return a * H**2 - a * H**2 * ε
        if base == H:
            return 2 * H**3 * ε**2 - H**3 * ε * η
    if tuple(derivative_indices) == (DE(0), DE(0), DE(0)) and base == H:
        return (
            -6 * H**4 * ε**3
            + 7 * H**4 * ε**2 * η
            - H**4 * ε * η**2
            - H**4 * ε * η * η2
        )
    return None


def _is_gamma_trace(expr):
    if tensor_head_name(expr) != "gammaGW":
        return False
    args = tensor_args(expr)
    return (
        len(args) == 2
        and _is_index(args[0], DN)
        and _is_index(args[1], DN)
        and args[0].label == args[1].label
    )


def _is_transverse_gamma_derivative(base, derivative_indices):
    if tensor_head_name(base) != "gammaGW":
        return False
    gamma_indices = tuple(index.label for index in tensor_args(base) if _is_index(index, DN))
    return any(_is_index(index, DN) and index.label in gamma_indices for index in derivative_indices)


def _fresh_label(*exprs):
    used = {
        index.label
        for expr in exprs
        for index in _iter_indices(expr)
        if isinstance(index.label, str)
    }
    return next(label for label in LatinIdx if label not in used)


def _iter_indices(expr):
    if isinstance(expr, Index):
        yield expr
        return
    for arg in getattr(expr, "args", ()):
        yield from _iter_indices(arg)


def _is_index(expr, index_type):
    return isinstance(expr, Index) and expr.head_name == index_type.name


if __name__ == "__main__":
    print(main())
