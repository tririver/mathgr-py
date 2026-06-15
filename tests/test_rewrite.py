import sympy as sp

import mathgr
from mathgr import ReplaceAll, RuleDelayed
from mathgr.decomp import D1, U2
from mathgr.tensor import tensor


def test_replace_all_supports_tensor_wild_labels():
    A = tensor("AReplaceAll")
    B = tensor("BReplaceAll")
    i = sp.Wild("i")
    mu = sp.Wild("mu")

    expr = A(U2("a"), D1("b")) + A(U2("c"), D1("d"))

    assert ReplaceAll(expr, {A(U2(i), D1(mu)): B(U2(i), D1(mu))}) == (
        B(U2("a"), D1("b")) + B(U2("c"), D1("d"))
    )


def test_rule_delayed_supports_tensor_wild_labels():
    A = tensor("ADelayed")
    B = tensor("BDelayed")
    i = sp.Wild("i")
    mu = sp.Wild("mu")

    expr = A(U2("a"), D1("b"))
    out = ReplaceAll(expr, [RuleDelayed(A(U2(i), D1(mu)), lambda i, mu: B(U2(i), D1(mu)))])

    assert out == B(U2("a"), D1("b"))


def test_replace_all_is_single_pass_unlike_replace_repeated():
    x, y, z = sp.symbols("x y z")

    assert ReplaceAll(x, {x: y, y: z}) == y


def test_rewrite_helpers_are_exported_from_package_root():
    assert mathgr.ReplaceAll is ReplaceAll
