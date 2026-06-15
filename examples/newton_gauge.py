"""Python port of the setup cells in MathGR's newton_gauge.nb example."""

import sympy as sp

from mathgr.decomp import DTot, Decomp0i, UTot
from mathgr.gr import MetricContract, R, V, WithMetric, X
from mathgr.rewrite import ReplaceAll, Rule, RuleDelayed
from mathgr.tensor import DE, DN, UE, UP, DefaultDim, Dta, Index, Pd, PdT, PdVars
from mathgr.tensor import Simp as _TensorSimp
from mathgr.tensor import is_pdt, pdt_parts, tensor, tensor_args, tensor_head_name
from mathgr.util import Eps, OO


a = sp.Symbol("a")
H = sp.Symbol("H")
dH = sp.Symbol("dH")
φ = sp.Symbol("φ")
ψ = sp.Symbol("ψ")
φ0 = sp.Symbol("φ0")
δφ = sp.Symbol("δφ")

g = tensor("g")
_h_raw = tensor("hNewtonGauge")
Sqrtg = a**4 * sp.sqrt((1 + 2 * Eps * φ) * (1 - 2 * Eps * ψ) ** 3)
_i = sp.Wild("i")
_j = sp.Wild("j")


def h(first, second):
    first = sp.sympify(first)
    second = sp.sympify(second)
    if _is_index(first, DN) and _is_index(second, DN):
        return -a**2 * (1 - 2 * Eps * ψ) * Dta(first, second)
    if _is_index(first, UP) and _is_index(second, UP):
        return -Dta(DN(first.label), DN(second.label)) / (a**2 * (1 - 2 * Eps * ψ))
    return _h_raw(first, second)


def Simp(expr, **options):
    hooks = tuple(options.pop("hooks", ())) + (_background_simp_hook,)
    return _TensorSimp(expr, hooks=hooks, **options)


metric_rules = [
    RuleDelayed(g(DN(_i), DN(_j)), lambda i, j: h(DN(i), DN(j))),
    Rule(g(DE(0), DE(0)), a**2 * (1 + 2 * Eps * φ)),
    Rule(g(DE(0), DN(_i)), sp.Integer(0)),
    Rule(g(DN(_i), DE(0)), sp.Integer(0)),
    RuleDelayed(g(UP(_i), UP(_j)), lambda i, j: h(UP(i), UP(j))),
    Rule(g(UE(0), UE(0)), 1 / (a**2 * (1 + 2 * Eps * φ))),
    Rule(g(UE(0), UP(_i)), sp.Integer(0)),
    Rule(g(UP(_i), UE(0)), sp.Integer(0)),
]


def decomp_hook(expr):
    return ReplaceAll(expr, metric_rules)


def DecompG2H(expr):
    def evaluate():
        value = expr() if callable(expr) else expr
        return MetricContract(value)

    decomposed = WithMetric(g, (UTot, DTot), evaluate)
    return Simp(Decomp0i(decomposed, hooks=(decomp_hook,)))


def action_density(*, simplify=True):
    field = φ0 + Eps * δφ
    expr = Sqrtg * (DecompG2H(lambda: -R() / 2 - X(field)) - V(field))
    return Simp(expr) if simplify else expr


def action_order(order):
    expr = OO(order)(action_density(simplify=True))
    expr = _collapse_scalar_spatial_delta_traces(expr)
    return Simp(_canonicalize_potential_derivatives(expr))


def main(*, compute_action=False):
    results = {
        "sqrtg": Sqrtg,
        "spatial_metric_down": h(DN("i"), DN("j")),
        "spatial_metric_up": h(UP("i"), UP("j")),
        "metric_00": decomp_hook(g(DE(0), DE(0))),
        "metric_0i": decomp_hook(g(DE(0), DN("i"))),
        "inverse_metric_00": decomp_hook(g(UE(0), UE(0))),
        "inverse_metric_0i": decomp_hook(g(UE(0), UP("i"))),
        "background_time_derivative": Simp(Pd(a, DE(0))),
        "background_second_time_derivative": Simp(PdT(a, PdVars(DE(0), DE(0)))),
        "background_spatial_derivative": Simp(Pd(a, DN("i"))),
        "action_density": action_density(simplify=compute_action),
    }
    if compute_action:
        results["s0"] = action_order(0)
        results["s1"] = action_order(1)
        results["s2"] = action_order(2)
    return results


def _background_simp_hook(expr):
    expr = sp.sympify(expr)
    if expr == DefaultDim:
        return sp.Integer(3)
    if not is_pdt(expr):
        if not getattr(expr, "args", ()):
            return expr
        rewritten_args = tuple(_background_simp_hook(arg) for arg in expr.args)
        if rewritten_args == expr.args:
            return expr
        return expr.func(*rewritten_args)
    base, derivative_indices = pdt_parts(expr)
    if base in {a, H, dH, φ0} and any(_is_index(index, DN) for index in derivative_indices):
        return sp.Integer(0)
    if tuple(derivative_indices) == (DE(0),):
        if base == a:
            return a * H
    if tuple(derivative_indices) == (DE(0), DE(0)):
        if base == a:
            return a * H**2 + a * dH
    return expr


def _collapse_scalar_spatial_delta_traces(expr):
    expr = sp.sympify(expr)
    if expr.func.__name__ == "_Dta":
        left, right = expr.args
        if (
            ((_is_index(left, DN) and _is_index(right, DN)) or (_is_index(left, UP) and _is_index(right, UP)))
            and left.label == right.label
        ):
            return DefaultDim
        return expr
    if isinstance(expr, Index) or not getattr(expr, "args", ()):
        return expr
    rewritten_args = tuple(_collapse_scalar_spatial_delta_traces(arg) for arg in expr.args)
    if rewritten_args == expr.args:
        return expr
    return expr.func(*rewritten_args)


def _canonicalize_potential_derivatives(expr):
    expr = sp.sympify(expr)
    if isinstance(expr, sp.Subs) and len(expr.variables) == 1 and len(expr.point) == 1:
        variable = expr.variables[0]
        point = expr.point[0]
        derivative = expr.expr
        if isinstance(derivative, sp.Derivative) and derivative.variables == (variable,):
            inner = derivative.expr
            if tensor_head_name(inner) == "V" and tensor_args(inner) == (variable,):
                return sp.Derivative(V(point), point)
    if isinstance(expr, Index) or not getattr(expr, "args", ()):
        return expr
    rewritten_args = tuple(_canonicalize_potential_derivatives(arg) for arg in expr.args)
    if rewritten_args == expr.args:
        return expr
    return expr.func(*rewritten_args)


def _is_index(expr, index_type):
    return isinstance(expr, Index) and expr.head_name == index_type.name


if __name__ == "__main__":
    print(main())
