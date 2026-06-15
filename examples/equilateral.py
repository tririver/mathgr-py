"""Python port of the initial symbolic cells in MathGR's equilateral.nb."""

import sympy as sp

from mathgr.util import Eps


k1, k2, k3 = sp.symbols("k1 k2 k3", positive=True)
epsilon = sp.Symbol("epsilon", positive=True)
tau = sp.Symbol("tau", negative=True)
cs = sp.Symbol("cs", positive=True)
x, y, z = sp.symbols("x y z")
H, Sigma, lambda_, phi0_dot = sp.symbols("H Sigma lambda phi0_dot")

a = sp.Function("a")
phi0 = sp.Function("phi0")
delta_phi = sp.Function("delta_phi")
zeta_field = sp.Function("zeta")

delta_X1, delta_X2 = sp.symbols("delta_X1 delta_X2")
P, PX, PXX, PXXX = sp.symbols("P PX PXX PXXX")


def XExpand(order):
    phi = phi0(tau) + Eps * delta_phi(x, y, z, tau)
    kinetic = (
        sp.diff(phi, tau) ** 2
        - sp.diff(phi, x) ** 2
        - sp.diff(phi, y) ** 2
        - sp.diff(phi, z) ** 2
    ) / (2 * a(tau) ** 2)
    return sp.simplify(sp.series(kinetic, Eps, 0, order + 1).removeO().coeff(Eps, order))


def lag(order):
    coefficients = {
        0: P,
        1: PX * delta_X1,
        2: PX * delta_X2 + PXX * delta_X1**2 / 2,
        3: delta_X1 * (PXXX * delta_X1**2 + 6 * PXX * delta_X2) / 6,
    }
    try:
        return a(tau) ** 4 * coefficients[order]
    except KeyError as exc:
        raise NotImplementedError("Only lag[0], lag[1], lag[2], and lag[3] equilateral.nb cells are ported.") from exc


def lag3_delta_phi_gauge():
    expr = lag(3).subs({delta_X1: XExpand(1), delta_X2: XExpand(2)})
    return to_canonical(expr)


def lag3_zeta_gauge():
    return to_canonical(_replace_delta_phi_with_zeta(lag3_delta_phi_gauge()))


def to_canonical(expr):
    result = sp.expand(expr)
    derivative_subs = {
        sp.diff(phi0(tau), tau): a(tau) * phi0_dot,
        sp.diff(phi0(tau), (tau, 2)): a(tau) ** 2 * H * phi0_dot,
        sp.diff(a(tau), tau): a(tau) ** 2 * H,
    }
    for _ in range(8):
        previous = result
        result = result.subs(derivative_subs)
        result = result.subs(PXXX, 12 * (lambda_ - phi0_dot**4 * PXX / 4) / phi0_dot**6)
        result = result.subs(PXX, PX * (1 / cs**2 - 1) / phi0_dot**2)
        result = _reduce_phi0_dot_powers(sp.cancel(sp.expand(result)))
        result = result.subs(epsilon, Sigma * cs**2 / H**2)
        result = sp.cancel(sp.expand(result))
        if result == previous:
            break
    return sp.expand(result)


def _reduce_phi0_dot_powers(expr):
    phi0_dot_squared = 2 * epsilon * H**2 / PX

    def is_phi0_dot_power(node):
        return isinstance(node, sp.Pow) and node.base == phi0_dot and node.exp.is_integer

    def replace_phi0_dot_power(node):
        exponent = int(node.exp)
        if exponent % 2 == 0:
            return phi0_dot_squared ** (exponent // 2)
        return phi0_dot * phi0_dot_squared ** ((exponent - 1) // 2)

    return expr.replace(is_phi0_dot_power, replace_phi0_dot_power)


def _replace_delta_phi_with_zeta(expr):
    def is_delta_phi_call(node):
        return getattr(node, "func", None) == delta_phi

    def replace_delta_phi_call(node):
        return -phi0_dot * zeta_field(*node.args) / H

    def is_delta_phi_derivative(node):
        return isinstance(node, sp.Derivative) and getattr(node.expr, "func", None) == delta_phi

    def replace_delta_phi_derivative(node):
        replacement = -phi0_dot * zeta_field(*node.expr.args) / H
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
