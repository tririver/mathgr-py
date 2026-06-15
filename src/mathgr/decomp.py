from __future__ import annotations

from collections import Counter

import sympy as sp

from .tensor import (
    DE,
    DN,
    UE,
    UP,
    GreekIdx,
    Index,
    LatinCapitalIdx,
    LatinIdx,
    SimpInto1,
    _freshen_hook_result_dummies,
    declare_idx,
)


DimTot = sp.Symbol("DimTot")
Dim1 = sp.Symbol("Dim1")
Dim2 = sp.Symbol("Dim2")

UTot, DTot = declare_idx("UTot", "DTot", dim=DimTot, index_set=LatinCapitalIdx, color="Blue")
U1, D1 = declare_idx("U1", "D1", dim=Dim1, index_set=GreekIdx, color="Black")
U2, D2 = declare_idx("U2", "D2", dim=Dim2, index_set=LatinIdx, color="Red")

_TOTAL_INDEX_NAMES = {UTot.name, DTot.name}
DecompHook = []
_MAX_HOOK_ITERATIONS = 10


def Decomp(expr, sectors, indices=None, hooks=None):
    expr = sp.sympify(expr)
    hooks = DecompHook if hooks is None else hooks
    if indices is None and isinstance(expr, sp.Add):
        return _apply_decomp_hooks(sp.Add(*(Decomp(arg, sectors, hooks=hooks) for arg in expr.args)), hooks)
    labels = _coerce_decomp_labels(indices) if indices is not None else _dummy_total_labels(expr)
    result = expr
    for label in labels:
        result = _decomp_label(result, label, _sector_rules(label, sectors))
        result = _apply_decomp_hooks(result, hooks)
    return _apply_decomp_hooks(result, hooks)


def Decomp0i(expr, indices=None, hooks=None):
    return Decomp(expr, (_explicit_sector(0), _index_family(DN, UP)), indices, hooks=hooks)


def Decomp01i(expr, indices=None, hooks=None):
    return Decomp(expr, (_explicit_sector(0), _explicit_sector(1), _index_family(DN, UP)), indices, hooks=hooks)


def Decomp0123(expr, indices=None, hooks=None):
    return Decomp(expr, tuple(_explicit_sector(component) for component in range(4)), indices, hooks=hooks)


def Decomp1i(expr, indices=None, hooks=None):
    return Decomp(expr, (_explicit_sector(1), _index_family(DN, UP)), indices, hooks=hooks)


def Decomp123(expr, indices=None, hooks=None):
    return Decomp(expr, tuple(_explicit_sector(component) for component in range(1, 4)), indices, hooks=hooks)


def DecompSe(expr, indices=None, hooks=None):
    return Decomp(expr, (_index_family(D2, U2), _index_family(D1, U1)), indices, hooks=hooks)


def _explicit_sector(component):
    return (lambda _label: DE(component), lambda _label: UE(component))


def _index_family(down, up):
    return (lambda label: down(label), lambda label: up(label))


def _sector_rules(label, sectors):
    return [{DTot(label): down(label), UTot(label): up(label)} for down, up in sectors]


def _coerce_decomp_labels(indices):
    if isinstance(indices, (str, int, sp.Symbol)):
        return []
    if isinstance(indices, Index):
        return []
    try:
        selectors = list(indices)
    except TypeError:
        return []
    labels = []
    for selector in selectors:
        label = _coerce_decomp_label_selector(selector)
        if label is not None:
            labels.append(label)
    return labels


def _coerce_decomp_label_selector(selector):
    if isinstance(selector, Index):
        return None
    if isinstance(selector, str):
        return selector
    if isinstance(selector, int):
        return selector
    if isinstance(selector, sp.Symbol):
        return str(selector)
    return None


def _dummy_total_labels(expr):
    labels = [index.label for index in _iter_total_indices(sp.sympify(expr))]
    counts = Counter(labels)
    seen = set()
    result = []
    for label in labels:
        if counts[label] == 2 and label not in seen:
            result.append(label)
            seen.add(label)
    return result


def _decomp_label(expr, label, alternatives):
    expr = sp.sympify(expr)
    if isinstance(expr, sp.Add):
        return sp.Add(*(_decomp_label(arg, label, alternatives) for arg in expr.args))
    if isinstance(expr, sp.Pow):
        base, exponent = expr.args
        if _contains_total_label(base, label) or _contains_total_label(exponent, label):
            if exponent == 2:
                return sp.Add(*(expr.xreplace(rule) for rule in alternatives))
            return _decomp_power_base(base, label, alternatives) ** _decomp_label(exponent, label, alternatives)
        return expr
    if expr.func in SimpInto1 and len(expr.args) == 1 and _contains_total_label(expr.args[0], label):
        return expr.func(_decomp_label(expr.args[0], label, alternatives))
    if isinstance(expr, sp.Mul):
        indexed_positions = [pos for pos, arg in enumerate(expr.args) if _contains_total_label(arg, label)]
        if len(indexed_positions) == 1:
            pos = indexed_positions[0]
            factors = list(expr.args)
            factors[pos] = _decomp_label(factors[pos], label, alternatives)
            return sp.Mul(*factors)
        if _should_decomp_split_power_product(expr, indexed_positions):
            return sp.Mul(*(_decomp_label(factor, label, alternatives) for factor in expr.args))
    if _contains_total_label(expr, label):
        return sp.Add(*(expr.xreplace(rule) for rule in alternatives))
    return expr


def _decomp_power_base(base, label, alternatives):
    base = sp.sympify(base)
    if isinstance(base, sp.Mul) and _has_dual_total_heads(base.args, label):
        return sp.Mul(*(_decomp_label(factor, label, alternatives) for factor in base.args))
    return _decomp_label(base, label, alternatives)


def _should_decomp_split_power_product(expr, indexed_positions):
    if len(indexed_positions) < 2:
        return False
    indexed_factors = [expr.args[pos] for pos in indexed_positions]
    return _has_dual_total_heads(indexed_factors, None) and all(
        isinstance(factor, sp.Pow)
        and factor.exp.is_Integer
        and factor.exp != 2
        for factor in indexed_factors
    )


def _has_dual_total_heads(expressions, label):
    head_names = {
        index.head_name
        for expr in expressions
        for index in _iter_total_indices(expr)
        if label is None or index.label == label
    }
    return UTot.name in head_names and DTot.name in head_names


def _apply_decomp_hooks(expr, hooks):
    current = sp.sympify(expr)
    for _ in range(_MAX_HOOK_ITERATIONS):
        replaced = _apply_decomp_hooks_once(current, hooks)
        if replaced == current:
            return current
        current = replaced
    return current


def _apply_decomp_hooks_once(expr, hooks):
    expr = sp.sympify(expr)
    if expr.args:
        rewritten_args = tuple(_apply_decomp_hooks_once(arg, hooks) for arg in expr.args)
        if rewritten_args != expr.args:
            expr = expr.func(*rewritten_args)
    for hook in hooks:
        expr = _apply_single_decomp_hook(expr, hook)
    return expr


def _apply_single_decomp_hook(expr, hook):
    if callable(hook):
        return _freshen_hook_result_dummies(sp.sympify(hook(expr)), expr)
    if isinstance(hook, dict):
        return _apply_hook_rules(expr, hook.items())
    if _is_hook_rule_pair(hook):
        return _apply_hook_rules(expr, (hook,))
    if isinstance(hook, (list, tuple)):
        return _apply_hook_rules(expr, hook)
    raise TypeError("Decomp hooks must be callables, dicts, or sequences of (old, new) pairs.")


def _is_hook_rule_pair(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    left, right = value
    return not (_looks_like_rule_pair(left) and _looks_like_rule_pair(right))


def _looks_like_rule_pair(value):
    return isinstance(value, (list, tuple)) and len(value) == 2


def _apply_hook_rules(expr, rules):
    replacements = tuple((sp.sympify(old), new if callable(new) else sp.sympify(new)) for old, new in rules)
    exact = {
        old: new
        for old, new in replacements
        if not callable(new) and not old.has(sp.Wild)
    }
    current = sp.sympify(expr).xreplace(exact)
    for old, new in replacements:
        if callable(new):
            if old.has(sp.Wild):
                current = current.replace(old, lambda **matches: sp.sympify(new(**matches)))
            else:
                current = current.replace(lambda node, old=old: node == old, lambda node: sp.sympify(new(node)))
            continue
        if old.has(sp.Wild):
            current = current.replace(old, new)
    return current


def _contains_total_label(expr, label):
    return any(index.label == label for index in _iter_total_indices(expr))


def _iter_total_indices(expr):
    if isinstance(expr, Index) and expr.head_name in _TOTAL_INDEX_NAMES:
        yield expr
        return
    for arg in getattr(expr, "args", ()):
        yield from _iter_total_indices(arg)
