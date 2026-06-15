"""Python port of setup/action cells in ``zeta_gauge_action_from_delta_phi.nb``."""

import sympy as sp

from mathgr.frwadm import DecompG2H, RADM, Simp as _FRWSimp, Sqrtg, H, a, α, b, β, ε, η, η2, η3, ζ
from mathgr.gr import V, X
from mathgr.tensor import DE, DN, Index, Pd, PdT, PdVars, Pm2, is_pdt, pdt_parts
import mathgr.typeset as typeset
from mathgr.util import Eps


φ0 = sp.Symbol("φ0")
δφ = sp.Symbol("δφ")
φ0dot = sp.Symbol("φ0dot")
δφdot = sp.Symbol("δφdot")
ζn = sp.Symbol("ζn")
φ = φ0 + Eps * δφ


def Simp(expr, **options):
    hooks = tuple(options.pop("hooks", ())) + (_delta_phi_simp_hook,)
    return _FRWSimp(sp.sympify(expr).xreplace({ζ: sp.Integer(0)}), hooks=hooks, **options)


def constraints():
    beta_numerator = (
        6 * a**2 * H**3 * ε * δφ
        - 2 * a**2 * H**3 * ε**2 * δφ
        + a**2 * H**3 * ε * η * δφ
        - 3 * a**2 * H * δφ * φ0dot**2
        + a**2 * H * ε * δφ * φ0dot**2
        - a**2 * δφdot * φ0dot**2
    )
    return {
        α: δφ * φ0dot / (2 * H),
        β: Pm2(beta_numerator / (2 * H * φ0dot), DN),
        b(DN("a")): sp.Integer(0),
    }


def action_density(*, simplify=True):
    expr = Sqrtg * (RADM() / 2 + DecompG2H(lambda: X(φ)) - V(φ))
    expr = expr.xreplace({ζ: sp.Integer(0)})
    return Simp(expr) if simplify else expr


def count_slow_roll_order(term, sr_list=(ε, η, η2, η3)):
    probe = sp.Symbol("_slow_roll_probe")
    polynomial = sp.sympify(term).xreplace({symbol: probe for symbol in sr_list})
    return max(0, len(sp.Poly(polynomial, probe).all_coeffs()) - 1)


def selected_cubic_terms():
    return (
        -2
        * a**3
        * ε**2
        * Pd(Pd(Pm2(ζn, DN), DE(0)), DN("a"))
        * Pd(ζn, DE(0))
        * Pd(ζn, DN("a"))
        + a**3 * ε**2 * ζn * Pd(ζn, DE(0)) ** 2
        + a * ε**2 * ζn * Pd(ζn, DN("a")) ** 2
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
        "φ": φ,
        "ζ": sp.Integer(0),
        "constraints": constraints(),
        "action_density": action_density(simplify=compute_action),
        "s3_select": selected_cubic_terms(),
        "s3_select_tex": selected_cubic_terms_tex(),
    }


def _delta_phi_simp_hook(expr):
    expr = sp.sympify(expr)
    if is_pdt(expr):
        base, derivative_indices = pdt_parts(expr)
        if base == φ0 and any(_is_index(index, DN) for index in derivative_indices):
            return sp.Integer(0)
        if base == φ0 and tuple(derivative_indices) == (DE(0),):
            return φ0dot
        if base == δφ and tuple(derivative_indices) == (DE(0),):
            return δφdot
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
