import sympy as sp

from mathgr.util_private import (
    apply2term,
    expand2list,
    get_sample_term,
    plus2list,
    prod,
    prod2times,
    replace_to,
    times2prod,
)


def test_term_list_helpers_match_mathematica_util_private():
    a, b, c = sp.symbols("a b c")
    testf = sp.Function("testf")

    assert plus2list(1 + a + c) == [sp.Integer(1), a, c]
    assert expand2list(1 + 2 * (a + c)) == [sp.Integer(1), 2 * a, 2 * c]
    assert apply2term(testf, 1 + 2 * (a + c)) == testf(1) + testf(2 * a) + testf(2 * c)
    assert get_sample_term(2 * (a + c)) == 2 * a


def test_times2prod_prod2times_and_replace_to():
    a, b, c, d, n, x, y, z = sp.symbols("a b c d n x y z")
    f = sp.Function("f")

    assert times2prod(a + b**3 * c + d**n) == a + prod(b, b, b, c) + d**n
    assert prod2times(a + prod(b, b, b, c) + d**n) == a + b**3 * c + d**n
    assert f(a, b, c).xreplace(replace_to([a, b, c], [x, y, z])) == f(x, y, z)
