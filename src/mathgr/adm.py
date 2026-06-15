from __future__ import annotations

import sympy as sp

from .decomp import DTot, UTot, Decomp0i
from .gr import MetricContract, UseMetric, WithMetric
from .tensor import DE, DN, UE, UP, Dta, Index, LatinIdx, Simp as _TensorSimp, tensor
from .tensor import tensor_args, tensor_head_name


a = sp.Symbol("a")
Sqrth = sp.Symbol("Sqrth")
ScriptCapitalN = sp.Symbol("ScriptCapitalN")
ScriptCapitalNVector = tensor("ScriptCapitalN")
_SHIFT_HEAD = tensor("ShiftN")
LapseN = ScriptCapitalN
Sqrtg = ScriptCapitalN * Sqrth * a**3
g = tensor("g")
h = tensor("h")
_DEFAULT_DIM = sp.Symbol("DefaultDim")


def ShiftN(*indices):
    if len(indices) != 1:
        return _SHIFT_HEAD(*(sp.sympify(index) for index in indices))
    index = sp.sympify(indices[0])
    if not _is_index(index, DN):
        return _SHIFT_HEAD(index)
    return ScriptCapitalNVector(sp.sympify(index))


def Simp(expr, **options):
    hooks = tuple(options.pop("hooks", ())) + (_adm_simp_hook,)
    return _TensorSimp(expr, hooks=hooks, **options)


def DecompG2H(expr):
    def evaluate():
        value = expr() if callable(expr) else expr
        return MetricContract(value)

    decomposed = WithMetric(g, (UTot, DTot), evaluate)
    return _replace_metric_components(Decomp0i(decomposed))


def _replace_metric_components(expr):
    expr = sp.sympify(expr)
    head_name = tensor_head_name(expr)
    if head_name == "g":
        args = tensor_args(expr)
        if len(args) == 2:
            replacement = _metric_component_replacement(args[0], args[1])
            if replacement is not None:
                return replacement
    if not getattr(expr, "args", ()):
        return expr
    new_args = tuple(_replace_metric_components(arg) for arg in expr.args)
    if new_args == expr.args:
        return expr
    return expr.func(*new_args)


def _metric_component_replacement(first, second):
    if _is_index(first, DN) and _is_index(second, DN):
        return h(first, second)
    if first == DE(0) and second == DE(0):
        left, right = _metric_dummy_labels((first, second))
        return -LapseN**2 + h(UP(left), UP(right)) * ShiftN(DN(left)) * ShiftN(DN(right))
    if _is_covariant_time_space(first, second):
        return ShiftN(second if _is_index(second, DN) else first)
    if _is_index(first, UP) and _is_index(second, UP):
        left, right = _metric_dummy_labels((first, second))
        return h(first, second) - ShiftN(DN(left)) * ShiftN(DN(right)) * h(UP(left), first) * h(
            UP(right), second
        ) / LapseN**2
    if first == UE(0) and second == UE(0):
        return -LapseN**-2
    if _is_contravariant_time_space(first, second):
        spatial = second if _is_index(second, UP) else first
        label = _metric_dummy_labels((first, second), count=1)[0]
        return h(spatial, UP(label)) * ShiftN(DN(label)) / LapseN**2
    return None


def _is_covariant_time_space(first, second):
    return (first == DE(0) and _is_index(second, DN)) or (_is_index(first, DN) and second == DE(0))


def _is_contravariant_time_space(first, second):
    return (first == UE(0) and _is_index(second, UP)) or (_is_index(first, UP) and second == UE(0))


def _adm_simp_hook(expr):
    return sp.sympify(expr).xreplace({_DEFAULT_DIM: sp.Integer(3)})


def _is_index(expr, index_type):
    return isinstance(expr, Index) and expr.head_name == index_type.name


def _metric_dummy_labels(expressions, *, count=2):
    used = {
        index.label
        for expr in expressions
        for index in _iter_indices(expr)
        if isinstance(index.label, str)
    }
    return tuple(label for label in LatinIdx if label not in used)[:count]


def _iter_indices(expr):
    if isinstance(expr, Index):
        yield expr
        return
    for arg in getattr(expr, "args", ()):
        yield from _iter_indices(arg)


UseMetric(h)


__all__ = [
    "DecompG2H",
    "LapseN",
    "ScriptCapitalN",
    "ScriptCapitalNVector",
    "ShiftN",
    "Simp",
    "Sqrtg",
    "Sqrth",
    "a",
    "g",
    "h",
]
