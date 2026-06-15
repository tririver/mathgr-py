"""Python port of early GR/TeX cells in general_presentation.nb."""

import sympy as sp

from mathgr.gr import CovD, R
from mathgr.tensor import DN, Simp
import mathgr.typeset as typeset


def main():
    ricci_scalar = Simp(R())
    second_bianchi_expr = second_bianchi()
    derivative_second_bianchi_expr = derivative_second_bianchi()
    previous_template = typeset.ToTeXTemplate
    try:
        typeset.ToTeXTemplate = True
        ricci_scalar_tex = typeset.ToTeXString(ricci_scalar)
    finally:
        typeset.ToTeXTemplate = previous_template
    return {
        "ricci_scalar": ricci_scalar,
        "ricci_scalar_tex": ricci_scalar_tex,
        "second_bianchi": second_bianchi_expr,
        "second_bianchi_simplified": Simp(second_bianchi_expr),
        "derivative_second_bianchi": derivative_second_bianchi_expr,
        "derivative_second_bianchi_simplified": Simp(derivative_second_bianchi_expr),
    }


def second_bianchi():
    return sp.expand(
        CovD(R(DN("a"), DN("b"), DN("c"), DN("d")), DN("e"))
        + CovD(R(DN("a"), DN("b"), DN("d"), DN("e")), DN("c"))
        + CovD(R(DN("a"), DN("b"), DN("e"), DN("c")), DN("d"))
    )


def derivative_second_bianchi():
    return sp.expand(CovD(_second_bianchi_unexpanded(), DN("f")))


def _second_bianchi_unexpanded():
    return (
        CovD(R(DN("a"), DN("b"), DN("c"), DN("d")), DN("e"))
        + CovD(R(DN("a"), DN("b"), DN("d"), DN("e")), DN("c"))
        + CovD(R(DN("a"), DN("b"), DN("e"), DN("c")), DN("d"))
    )


if __name__ == "__main__":
    print(main())
