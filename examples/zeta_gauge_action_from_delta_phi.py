"""Python port of setup/action cells in ``zeta_gauge_action_from_delta_phi.nb``."""

import sympy as sp

from mathgr.frwadm import DecompG2H, RADM, Simp as _FRWSimp, Sqrtg, H, a, alpha, b, beta, epsilon, eta, eta2, eta3, zeta
from mathgr.gr import V, X
from mathgr.tensor import DE, DN, Index, Pd, PdT, PdVars, Pm2, is_pdt, pdt_parts
import mathgr.typeset as typeset
from mathgr.util import Eps


phi0 = sp.Symbol("phi0")
varphi = sp.Symbol("varphi")
phi0_dot = sp.Symbol("phi0_dot")
varphi_dot = sp.Symbol("varphi_dot")
zeta_n = sp.Symbol("zeta_n")
phi = phi0 + Eps * varphi


def Simp(expr, **options):
    hooks = tuple(options.pop("hooks", ())) + (_delta_phi_simp_hook,)
    return _FRWSimp(sp.sympify(expr).xreplace({zeta: sp.Integer(0)}), hooks=hooks, **options)


def constraints():
    beta_numerator = (
        6 * a**2 * H**3 * epsilon * varphi
        - 2 * a**2 * H**3 * epsilon**2 * varphi
        + a**2 * H**3 * epsilon * eta * varphi
        - 3 * a**2 * H * varphi * phi0_dot**2
        + a**2 * H * epsilon * varphi * phi0_dot**2
        - a**2 * varphi_dot * phi0_dot**2
    )
    return {
        alpha: varphi * phi0_dot / (2 * H),
        beta: Pm2(beta_numerator / (2 * H * phi0_dot), DN),
        b(DN("a")): sp.Integer(0),
    }


def action_density(*, simplify=True):
    expr = Sqrtg * (RADM() / 2 + DecompG2H(lambda: X(phi)) - V(phi))
    expr = expr.xreplace({zeta: sp.Integer(0)})
    return Simp(expr) if simplify else expr


def count_slow_roll_order(term, sr_list=(epsilon, eta, eta2, eta3)):
    probe = sp.Symbol("_slow_roll_probe")
    polynomial = sp.sympify(term).xreplace({symbol: probe for symbol in sr_list})
    return max(0, len(sp.Poly(polynomial, probe).all_coeffs()) - 1)


def selected_cubic_terms():
    return (
        -2
        * a**3
        * epsilon**2
        * Pd(Pd(Pm2(zeta_n, DN), DE(0)), DN("a"))
        * Pd(zeta_n, DE(0))
        * Pd(zeta_n, DN("a"))
        + a**3 * epsilon**2 * zeta_n * Pd(zeta_n, DE(0)) ** 2
        + a * epsilon**2 * zeta_n * Pd(zeta_n, DN("a")) ** 2
    )


def selected_cubic_terms_tex():
    previous_template = typeset.ToTeXTemplate
    try:
        typeset.ToTeXTemplate = True
        return typeset.ToTeXString(selected_cubic_terms())
    finally:
        typeset.ToTeXTemplate = previous_template


def main(*, compute_action=True):
    return {
        "phi": phi,
        "zeta": sp.Integer(0),
        "constraints": constraints(),
        "action_density": action_density(simplify=compute_action),
        "s3_select": selected_cubic_terms(),
        "s3_select_tex": selected_cubic_terms_tex(),
    }


def _delta_phi_simp_hook(expr):
    expr = sp.sympify(expr)
    if is_pdt(expr):
        base, derivative_indices = pdt_parts(expr)
        if base == phi0 and any(_is_index(index, DN) for index in derivative_indices):
            return sp.Integer(0)
        if base == phi0 and tuple(derivative_indices) == (DE(0),):
            return phi0_dot
        if base == varphi and tuple(derivative_indices) == (DE(0),):
            return varphi_dot
    if isinstance(expr, Index) or not getattr(expr, "args", ()):
        return expr
    rewritten_args = tuple(_delta_phi_simp_hook(arg) for arg in expr.args)
    if rewritten_args == expr.args:
        return expr
    return expr.func(*rewritten_args)


def _is_index(expr, index_type):
    return isinstance(expr, Index) and expr.head_name == index_type.name


if __name__ == "__main__":
    print(main(compute_action=False))
