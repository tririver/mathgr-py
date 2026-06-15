"""Python port of the lightweight Decomp0i cells in MathGR's decompExample.nb."""

import sympy as sp

from mathgr.decomp import DTot, Decomp0i, UTot
from mathgr.gr import R, WithMetric
from mathgr.tensor import DE, DN, UE, UP, Dta, Simp, tensor, tensor_args, tensor_head_name


def main():
    f = tensor("f")
    h = tensor("hDecompExampleHook")
    metric = tensor("metricDecompExample")
    h0, h1 = sp.symbols("h0 h1")

    simple = Decomp0i(f(UTot("mu"), DTot("mu")))
    sqrt_expr = sp.sqrt(1 + f(DTot("a")) * f(UTot("a")))
    metric_decomposition = Simp(Decomp0i(Simp(WithMetric(metric, (UTot, DTot), lambda: R()))))

    return {
        "simple": simple,
        "simple_simplified": Simp(simple),
        "hook_replacement": Simp(Decomp0i(h(UTot("mu"), DTot("mu")), hooks=(_decomp_hook(h, h0, h1),))),
        "sqrt": Decomp0i(sqrt_expr),
        "metric_noop": Decomp0i(Simp(R())),
        "metric_decomposition": metric_decomposition,
    }


def _decomp_hook(head, h0, h1):
    def hook(expr):
        if tensor_head_name(expr) != head.name:
            return expr
        args = tensor_args(expr)
        if args == (UE(0), DE(0)):
            return h0
        if len(args) == 2 and args[0].head_name == UP.name and args[1].head_name == DN.name:
            return h1 * Dta(args[0], args[1])
        return expr

    return hook


if __name__ == "__main__":
    print(main())
