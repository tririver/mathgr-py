from __future__ import annotations

from collections.abc import Callable, Iterable

import sympy as sp


class _Prod(sp.Function):
    nargs = None


prod = _Prod


def plus2list(expr):
    expr = sp.sympify(expr)
    if isinstance(expr, sp.Add):
        return list(expr.args)
    if isinstance(expr, (list, tuple)):
        return list(expr)
    return [expr]


def expand2list(expr):
    return plus2list(sp.expand(expr))


def apply2term(func: Callable[[sp.Expr], sp.Expr], expr):
    return sp.Add(*(func(term) for term in expand2list(expr)))


def get_sample_term(expr):
    expanded = sp.expand(expr)
    if isinstance(expanded, sp.Add):
        return expanded.args[0]
    return expanded


def _ordered_mul_args(expr: sp.Mul) -> list[sp.Expr]:
    args = list(expr.args)
    return sorted(args, key=lambda arg: (0 if _positive_integer_power(arg) else 1, sp.default_sort_key(arg)))


def _positive_integer_power(expr) -> bool:
    return isinstance(expr, sp.Pow) and expr.exp.is_Integer and expr.exp > 0


def times2prod(expr, product=prod):
    expr = sp.sympify(expr)
    if isinstance(expr, sp.Add):
        return sp.Add(*(times2prod(arg, product) for arg in expr.args))
    if isinstance(expr, sp.Mul):
        factors = []
        for arg in _ordered_mul_args(expr):
            factors.extend(_expand_power_factor(arg, product))
        return product(*factors)
    if _positive_integer_power(expr):
        return product(*[times2prod(expr.base, product) for _ in range(int(expr.exp))])
    if expr.args:
        return expr.func(*(times2prod(arg, product) for arg in expr.args))
    return expr


def _expand_power_factor(expr, product):
    if _positive_integer_power(expr):
        return [times2prod(expr.base, product) for _ in range(int(expr.exp))]
    return [times2prod(expr, product)]


def prod2times(expr, product=prod):
    expr = sp.sympify(expr)
    if expr.func == product:
        return sp.Mul(*(prod2times(arg, product) for arg in expr.args))
    if expr.args:
        return expr.func(*(prod2times(arg, product) for arg in expr.args))
    return expr


def replace_to(source: Iterable, target: Iterable):
    return dict(zip(source, target, strict=True))
