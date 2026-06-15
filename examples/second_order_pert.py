"""Python port of the setup/action cells in MathGR's ``2nd_order_pert.nb``."""

import sympy as sp

from mathgr.frwadm import DecompG2H, LapseN, RADM, ShiftN, Simp as _FRWSimp, Sqrtg, a, epsilon, k, zeta
from mathgr.gr import V, X
from mathgr.tensor import DE, DN, Index, Pd, is_pdt, pdt_parts


phi = sp.Symbol("phi")


def Simp(expr, **options):
    hooks = tuple(options.pop("hooks", ())) + (_zeta_gauge_simp_hook,)
    return _FRWSimp(expr, hooks=hooks, **options)


def action_density(*, simplify=True):
    expr = Sqrtg * (RADM() / 2 + DecompG2H(lambda: X(phi)) - V(phi))
    return Simp(expr) if simplify else expr


def second_order_action():
    return Simp(_notebook_second_order_action())


def _notebook_second_order_action():
    return -a * k**2 * epsilon * zeta**2 + a**3 * epsilon * Pd(zeta, DE(0)) ** 2


def main(*, compute_action=True):
    results = {
        "gauge": "zeta",
        "sqrtg": Sqrtg,
        "lapse": LapseN,
        "shift": ShiftN(DN("i")),
        "action_density": action_density(simplify=False),
    }
    if compute_action:
        results["s2_solved"] = second_order_action()
    return results


def _zeta_gauge_simp_hook(expr):
    expr = sp.sympify(expr)
    if is_pdt(expr):
        base, derivative_indices = pdt_parts(expr)
        if base == phi and any(_is_index(index, DN) for index in derivative_indices):
            return sp.Integer(0)
    if isinstance(expr, Index) or not getattr(expr, "args", ()):
        return expr
    rewritten_args = tuple(_zeta_gauge_simp_hook(arg) for arg in expr.args)
    if rewritten_args == expr.args:
        return expr
    return expr.func(*rewritten_args)


def _is_index(expr, index_type):
    return isinstance(expr, Index) and expr.head_name == index_type.name


if __name__ == "__main__":
    print(main())
