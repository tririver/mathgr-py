"""Python port of setup/helper cells in ``gauge_trans_df_to_zeta.nb``."""

import sympy as sp

from mathgr.tensor import DE, DN, DefaultDim, Index, Pd, PdT, PdVars, Simp as _TensorSimp, is_pdt, pdt_parts, tensor
from mathgr.util import CollectEps, Eps


a = sp.Symbol("a")
H = sp.Symbol("H")
epsilon = sp.Symbol("epsilon")
eta = sp.Symbol("eta")
eta2 = sp.Symbol("eta2")
phi0 = sp.Symbol("phi0")
varphi = sp.Symbol("varphi")
delta_t = sp.Symbol("delta_t")
delta_x = tensor("delta_x")


def Simp(expr, **options):
    hooks = tuple(options.pop("hooks", ())) + (_background_simp_hook,)
    return _TensorSimp(expr, hooks=hooks, **options)


def expand_at_xzg(field, label="a"):
    index = DN(label)
    return Simp(
        field
        + Eps * Pd(field, DE(0)) * delta_t
        + Eps**2 * PdT(field, PdVars(DE(0), DE(0))) * delta_t**2 / 2
        + Eps**2 * Pd(field, index) * delta_x(index)
    )


def taylor_solve(eq_raw, var, max_order):
    coeff_solutions = {}
    series_var = sum(Eps ** (order - 1) * sp.Symbol(f"{var}_{order}") for order in range(1, max_order + 1))
    equation = sp.expand(sp.sympify(eq_raw).xreplace({var: series_var}))
    for order in range(max_order + 1):
        order_equation = sp.expand(equation.xreplace(coeff_solutions)).coeff(Eps, order)
        unknown = sp.Symbol(f"{var}_{order}")
        solution = sp.solve(order_equation, unknown, dict=True)
        if solution and unknown in solution[0]:
            coeff_solutions[unknown] = solution[0][unknown]
    return sp.expand(series_var.xreplace(coeff_solutions))


def col_simp(expr):
    return CollectEps(op=Simp)(expr)


def main():
    return {
        "default_dim": Simp(DefaultDim),
        "expanded_a": expand_at_xzg(a),
        "expanded_field": expand_at_xzg(varphi),
        "collected": col_simp,
    }


def _background_simp_hook(expr):
    expr = sp.sympify(expr)
    if expr == DefaultDim:
        return sp.Integer(3)
    if not is_pdt(expr):
        if isinstance(expr, Index) or not getattr(expr, "args", ()):
            return expr
        rewritten_args = tuple(_background_simp_hook(arg) for arg in expr.args)
        if rewritten_args == expr.args:
            return expr
        return expr.func(*rewritten_args)
    base, derivative_indices = pdt_parts(expr)
    if base in {phi0, a, H, epsilon} and any(_is_index(index, DN) for index in derivative_indices):
        return sp.Integer(0)
    if tuple(derivative_indices) == (DE(0),):
        if base == a:
            return a * H
        if base == H:
            return -epsilon * H**2
        if base == epsilon:
            return H * epsilon * eta
        if base == eta:
            return H * eta2 * eta
    if tuple(derivative_indices) == (DE(0), DE(0)):
        if base == a:
            return a * H**2 - a * H**2 * epsilon
        if base == H:
            return 2 * H**3 * epsilon**2 - H**3 * epsilon * eta
    return expr


def _is_index(expr, index_type):
    return isinstance(expr, Index) and expr.head_name == index_type.name


if __name__ == "__main__":
    print(main())
