"""Python port of setup cells in MathGR's ``3rd_order_pert_pm2.nb`` example."""

from mathgr.frwadm import Simp
from mathgr.tensor import DN, Pd, Pm2
from mathgr.util import OO

from .third_order_pert import action_density, phi


def chi_constraint():
    return Pm2(Pd(phi, DN("i")), DN)


def third_order_action():
    return OO(3, op=Simp)(action_density(simplify=True))


def main(*, compute_orders=True):
    results = {
        "gauge": "zeta",
        "uses_pm2_constraints": True,
        "action_density": action_density(simplify=compute_orders),
        "constraint_kernel": chi_constraint(),
        "order_operator": OO(3, op=Simp),
    }
    if compute_orders:
        results["s3"] = third_order_action()
    return results


if __name__ == "__main__":
    print(main(compute_orders=False))
