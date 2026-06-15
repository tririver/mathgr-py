"""Python port of setup cells in MathGR's ``3rd_order_pert.nb`` example."""

from mathgr.frwadm import DecompG2H, RADM, Simp, Sqrtg
from mathgr.gr import V, X
from mathgr.util import OO

from .second_order_pert import φ


def action_density(*, simplify=True):
    expr = Sqrtg * (RADM() / 2 + DecompG2H(lambda: X(φ)) - V(φ))
    return Simp(expr) if simplify else expr


def third_order_action():
    return OO(3, op=Simp)(action_density(simplify=True))


def main(*, compute_orders=True):
    results = {
        "gauge": "ζ",
        "sqrtg": Sqrtg,
        "action_density": action_density(simplify=compute_orders),
        "order_operator": OO(3, op=Simp),
    }
    if compute_orders:
        results["s3"] = third_order_action()
    return results


if __name__ == "__main__":
    print(main(compute_orders=False))
