"""Python port of the initial symbolic cells in MathGR's equilateral.nb."""

import sympy as sp

from mathgr.util import Eps


k1, k2, k3 = sp.symbols("k1 k2 k3", positive=True)
ε = sp.Symbol("ε", positive=True)
τ = sp.Symbol("τ", negative=True)
cs = sp.Symbol("cs", positive=True)
x, y, z = sp.symbols("x y z")
H, Σ, λ, φ0dot = sp.symbols("H Σ λ φ0dot")

a = sp.Function("a")
φ0 = sp.Function("φ0")
δφ = sp.Function("δφ")
ζfield = sp.Function("ζ")

δX1, δX2 = sp.symbols("δX1 δX2")
P, PX, PXX, PXXX = sp.symbols("P PX PXX PXXX")


def XExpand(order):
    φ = φ0(τ) + Eps * δφ(x, y, z, τ)
    kinetic = (
        sp.diff(φ, τ) ** 2
        - sp.diff(φ, x) ** 2
        - sp.diff(φ, y) ** 2
        - sp.diff(φ, z) ** 2
    ) / (2 * a(τ) ** 2)
    return sp.simplify(sp.series(kinetic, Eps, 0, order + 1).removeO().coeff(Eps, order))


def lag(order):
    coefficients = {
        0: P,
        1: PX * δX1,
        2: PX * δX2 + PXX * δX1**2 / 2,
        3: δX1 * (PXXX * δX1**2 + 6 * PXX * δX2) / 6,
    }
    try:
        return a(τ) ** 4 * coefficients[order]
    except KeyError as exc:
        raise NotImplementedError("Only lag[0], lag[1], lag[2], and lag[3] equilateral.nb cells are ported.") from exc


def lag3_delta_phi_gauge():
    expr = lag(3).subs({δX1: XExpand(1), δX2: XExpand(2)})
    return to_canonical(expr)


def lag3_zeta_gauge():
    return to_canonical(_replace_delta_phi_with_zeta(lag3_delta_phi_gauge()))


def to_canonical(expr):
    result = sp.expand(expr)
    derivative_subs = {
        sp.diff(φ0(τ), τ): a(τ) * φ0dot,
        sp.diff(φ0(τ), (τ, 2)): a(τ) ** 2 * H * φ0dot,
        sp.diff(a(τ), τ): a(τ) ** 2 * H,
    }
    for _ in range(8):
        previous = result
        result = result.subs(derivative_subs)
        result = result.subs(PXXX, 12 * (λ - φ0dot**4 * PXX / 4) / φ0dot**6)
        result = result.subs(PXX, PX * (1 / cs**2 - 1) / φ0dot**2)
        result = _reduce_phi0_dot_powers(sp.cancel(sp.expand(result)))
        result = result.subs(ε, Σ * cs**2 / H**2)
        result = sp.cancel(sp.expand(result))
        if result == previous:
            break
    return sp.expand(result)


def _reduce_phi0_dot_powers(expr):
    phi0_dot_squared = 2 * ε * H**2 / PX

    def is_phi0_dot_power(node):
        return isinstance(node, sp.Pow) and node.base == φ0dot and node.exp.is_integer

    def replace_phi0_dot_power(node):
        exponent = int(node.exp)
        if exponent % 2 == 0:
            return phi0_dot_squared ** (exponent // 2)
        return φ0dot * phi0_dot_squared ** ((exponent - 1) // 2)

    return expr.replace(is_phi0_dot_power, replace_phi0_dot_power)


def _replace_delta_phi_with_zeta(expr):
    def is_delta_phi_call(node):
        return getattr(node, "func", None) == δφ

    def replace_delta_phi_call(node):
        return -φ0dot * ζfield(*node.args) / H

    def is_delta_phi_derivative(node):
        return isinstance(node, sp.Derivative) and getattr(node.expr, "func", None) == δφ

    def replace_delta_phi_derivative(node):
        replacement = -φ0dot * ζfield(*node.expr.args) / H
        return sp.diff(replacement, *node.variables)

    return expr.replace(is_delta_phi_derivative, replace_delta_phi_derivative).replace(
        is_delta_phi_call, replace_delta_phi_call
    )


def main():
    return {
        "x_expand_1": XExpand(1),
        "x_expand_2": XExpand(2),
        "lag_0": lag(0),
        "lag_1": lag(1),
        "lag_2": lag(2),
        "lag_3": lag(3),
        "lag3_delta_phi": lag3_delta_phi_gauge(),
        "lag3_zeta": lag3_zeta_gauge(),
    }


if __name__ == "__main__":
    print(main())
