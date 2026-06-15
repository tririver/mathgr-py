from __future__ import annotations

import sympy as sp

from .util import _coerce_replacements, _replace_once, prod2times, times2prod


_MAX_REPLACE_REPEATED_ITERATIONS = 20


def Rule(lhs, rhs):
    """Immediate replacement rule: lhs -> rhs."""
    return (lhs, rhs)


def RuleDelayed(lhs, func):
    """Delayed replacement rule: lhs :> func(**wild_matches)."""
    return (lhs, func)


def ReplaceAll(expr, rules):
    """MathGR/Mathematica-style replacement. Equivalent to expr /. rules."""
    replacements = _coerce_replacements(rules)
    return prod2times(_replace_once(times2prod(expr), replacements))


def ReplaceRepeated(expr, rules, *, max_iter=_MAX_REPLACE_REPEATED_ITERATIONS):
    """Repeated replacement until stable, bounded like Mathematica //."""
    current = sp.sympify(expr)
    for _ in range(max_iter):
        new = ReplaceAll(current, rules)
        if new == current:
            return current
        current = new
    return current


def LabelWild(name, *, exclude=(), properties=()):
    return sp.Wild(name, exclude=exclude, properties=properties)


def IndexWild(index_type, name):
    return index_type(sp.Wild(name))
