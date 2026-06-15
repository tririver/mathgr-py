from __future__ import annotations

from collections import OrderedDict

import sympy as sp


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
    return _replace_once(expr, replacements)


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


def _coerce_replacements(rule):
    if isinstance(rule, dict):
        items = rule.items()
    elif _is_rule_pair(rule):
        items = (rule,)
    elif isinstance(rule, (list, tuple)):
        items = rule
    else:
        raise TypeError("Replacement rule must be a dict or a sequence of (old, new) pairs.")
    return OrderedDict((sp.sympify(old), new if callable(new) else sp.sympify(new)) for old, new in items)


def _is_rule_pair(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    left, right = value
    return not (_looks_like_rule_pair(left) and _looks_like_rule_pair(right))


def _looks_like_rule_pair(value):
    return isinstance(value, (list, tuple)) and len(value) == 2


def _replace_once(expr, replacements):
    exact_replacements = {
        old: new
        for old, new in replacements.items()
        if not callable(new) and not old.has(sp.Wild)
    }
    current = sp.sympify(expr).xreplace(exact_replacements)
    for old, new in replacements.items():
        if callable(new):
            if old.has(sp.Wild):
                current = current.replace(old, lambda **matches: sp.sympify(new(**matches)))
            else:
                current = current.replace(lambda node, old=old: node == old, lambda node: sp.sympify(new(node)))
            continue
        if old.has(sp.Wild):
            current = current.replace(old, new)
    return current
