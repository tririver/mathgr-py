"""Python port of setup/action cells in ``2nd_order_pert_deltaphi_gauge.nb``."""

import sympy as sp

from mathgr.frwadm import DecompG2H, LapseN, RADM, ShiftN, Simp as _FRWSimp, Sqrtg, zeta
from mathgr.gr import V, X
from mathgr.tensor import DN, Index, is_pdt, pdt_parts
from mathgr.util import Eps


phi0 = sp.Symbol("phi0")
varphi = sp.Symbol("varphi")
phi = phi0 + Eps * varphi


def Simp(expr, **options):
    hooks = tuple(options.pop("hooks", ())) + (_delta_phi_gauge_simp_hook,)
    return _FRWSimp(expr.xreplace({zeta: sp.Integer(0)}), hooks=hooks, **options)


def action_density(*, simplify=True):
    expr = Sqrtg * (RADM() / 2 + DecompG2H(lambda: X(phi)) - V(phi))
    expr = expr.xreplace({zeta: sp.Integer(0)})
    return Simp(expr) if simplify else expr


def main(*, compute_action=True):
    return {
        "gauge": "delta_phi",
        "phi": phi,
        "zeta": sp.Integer(0),
        "sqrtg": Sqrtg.xreplace({zeta: sp.Integer(0)}),
        "lapse": LapseN,
        "shift": ShiftN(DN("i")),
        "action_density": action_density(simplify=compute_action),
    }


def _delta_phi_gauge_simp_hook(expr):
    expr = sp.sympify(expr)
    if is_pdt(expr):
        base, derivative_indices = pdt_parts(expr)
        if base == phi0 and any(_is_index(index, DN) for index in derivative_indices):
            return sp.Integer(0)
    if isinstance(expr, Index) or not getattr(expr, "args", ()):
        return expr
    rewritten_args = tuple(_delta_phi_gauge_simp_hook(arg) for arg in expr.args)
    if rewritten_args == expr.args:
        return expr
    return expr.func(*rewritten_args)


def _is_index(expr, index_type):
    return isinstance(expr, Index) and expr.head_name == index_type.name


if __name__ == "__main__":
    print(main())
