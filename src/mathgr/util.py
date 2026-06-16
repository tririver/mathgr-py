from __future__ import annotations

from collections import Counter, OrderedDict
from functools import lru_cache
from itertools import count

import sympy as sp

from .tensor import (
    DN,
    Index,
    LatinIdx,
    Pd,
    PdT,
    PdVars,
    Simp,
    _CONSTANTS,
    is_pdt,
    pdt_parts,
    tensor,
    tensor_args,
    tensor_head_name,
)
from .util_private import apply2term, prod


_SERIES_CACHE_SIZE = 1024
_SERIES_CACHED_FUNCTIONS = []


def _series_cache(func):
    cached = lru_cache(maxsize=_SERIES_CACHE_SIZE)(func)
    _SERIES_CACHED_FUNCTIONS.append(cached)
    return cached


def _clear_series_caches():
    for func in _SERIES_CACHED_FUNCTIONS:
        func.cache_clear()


def _series_cache_infos():
    return {func.__name__: func.cache_info() for func in _SERIES_CACHED_FUNCTIONS}


Eps = sp.Symbol("Eps")
_CONSTANTS.add(Eps)


class MomentumLabel(sp.Expr):
    is_commutative = True

    def __new__(cls, head: str, label):
        return sp.Expr.__new__(cls, sp.Symbol(head), sp.sympify(label))

    @property
    def head(self):
        return tensor(str(self.args[0]))

    @property
    def label(self):
        return self.args[1]

    def __call__(self, index):
        return self.head(self.label, index)

    def _sympystr(self, printer):
        return f"{self.args[0]}({printer.doprint(self.label)})"


def k(label):
    return MomentumLabel("k", label)


def SolveExpr(eqs, exprs_raw):
    exprs = [sp.sympify(expr) for expr in _flatten_exprs(exprs_raw)]
    replacements = [sp.Dummy("mathgr_solve") for _ in exprs]
    replaced_eqs = [_replace_exact(eq, dict(zip(exprs, replacements, strict=True))) for eq in _as_list(eqs)]
    solutions = sp.solve(replaced_eqs, replacements, dict=True)
    if not _solves_all_replacements(solutions, replacements):
        solutions = _solve_exprs_by_definitions(eqs, exprs, replacements)
    return [_restore_solution(solution, exprs, replacements) for solution in solutions]


def TReplace(expr_or_rule, rule=None):
    if rule is None:
        return lambda expr: TReplace(expr, expr_or_rule)
    from .rewrite import ReplaceAll

    return ReplaceAll(expr_or_rule, rule)


def _flatten_exprs(exprs_raw):
    if isinstance(exprs_raw, (list, tuple, set, sp.Tuple)):
        result = []
        for item in exprs_raw:
            result.extend(_flatten_exprs(item))
        return result
    return [exprs_raw]


def _as_list(value):
    if isinstance(value, (list, tuple, set, sp.Tuple)):
        return list(value)
    return [value]


def _replace_exact(expr, replacements):
    return sp.sympify(expr).xreplace(replacements)


def _solves_all_replacements(solutions, replacements):
    return bool(solutions) and all(all(replacement in solution for replacement in replacements) for solution in solutions)


def _solve_exprs_by_definitions(eqs, exprs, replacements):
    defining_equations = [sp.Eq(replacement, expr) for expr, replacement in zip(exprs, replacements, strict=True)]
    unknowns = list(replacements) + _symbols_inside(exprs)
    return sp.solve(_as_list(eqs) + defining_equations, unknowns, dict=True)


def _symbols_inside(exprs):
    symbols = OrderedDict()
    for expr in exprs:
        for symbol in sorted(sp.sympify(expr).free_symbols, key=sp.default_sort_key):
            symbols[symbol] = None
    return list(symbols)


def _restore_solution(solution, exprs, replacements):
    reverse = dict(zip(replacements, exprs, strict=True))
    return {
        expr: sp.sympify(solution[replacement]).xreplace(reverse)
        for expr, replacement in zip(exprs, replacements, strict=True)
        if replacement in solution
    }


def _rename_repeated_product_factors(expr):
    expr = sp.sympify(expr)
    if expr.func == prod:
        return _rename_repeated_product(expr)
    if expr.args:
        return expr.func(*(_rename_repeated_product_factors(arg) for arg in expr.args))
    return expr


def _rename_repeated_product(expr):
    rewritten_args = []
    seen = {}
    used_labels = {index.label for arg in expr.args for index in _iter_indices(arg)}
    for arg in expr.args:
        rewritten = _rename_repeated_product_factors(arg)
        indices = list(_iter_indices(rewritten))
        if indices and rewritten in seen:
            rewritten = _rename_all_indices(rewritten, used_labels)
            used_labels.update(index.label for index in _iter_indices(rewritten))
        else:
            used_labels.update(index.label for index in indices)
        seen[rewritten] = seen.get(rewritten, 0) + 1
        rewritten_args.append(rewritten)
    return prod(*rewritten_args)


def _rename_all_indices(expr, used_labels):
    indices = list(_iter_indices(expr))
    label_map = {}
    fresh_labels = (label for label in LatinIdx if label not in used_labels)
    for index in indices:
        if index.label in label_map:
            continue
        label_map[index.label] = next(fresh_labels)
    replacements = {
        index: index.with_label(label_map[index.label])
        for index in set(indices)
        if index.label in label_map
    }
    return sp.sympify(expr).xreplace(replacements)


def TPower(expr, n: int):
    if not isinstance(n, int):
        raise TypeError("TPower exponent must be an integer.")
    expr = sp.sympify(expr)
    product = sp.Mul(*([expr] * abs(n))) if n else sp.Integer(1)
    return product if n >= 0 else product**-1


def _tensor_power_factors(expr, n: int):
    dummy_labels = _dummy_labels(expr)
    if not dummy_labels:
        return [expr] * n

    factors = []
    used_labels = {
        index.label
        for index in _iter_indices(expr)
        if isinstance(index.label, str)
    }
    pools = _dummy_label_pools(expr, dummy_labels)
    for copy_pos in range(n):
        if copy_pos == 0:
            factors.append(expr)
            continue
        label_map = {}
        for label in dummy_labels:
            new_label = _next_available_label(pools[label], used_labels)
            label_map[label] = new_label
            used_labels.add(new_label)
        factors.append(_replace_index_labels(expr, label_map))
    return factors


def _dummy_labels(expr):
    labels = [
        index.label
        for index in _iter_indices(expr)
        if isinstance(index.label, str) and not index.head.explicit
    ]
    counts = Counter(labels)
    return _ordered_unique(label for label in labels if counts[label] == 2)


def _dummy_label_pools(expr, labels):
    pools = {}
    label_set = set(labels)
    for index in _iter_indices(expr):
        if index.label in label_set and index.label not in pools:
            pools[index.label] = tuple(index.head.index_set)
    return pools


def _next_available_label(pool, used_labels):
    for label in pool:
        if label not in used_labels:
            return label
    suffix = 1
    while True:
        for label in pool:
            candidate = f"{label}{suffix}"
            if candidate not in used_labels:
                return candidate
        suffix += 1


def _replace_index_labels(expr, label_map):
    replacements = {
        index: index.with_label(label_map[index.label])
        for index in set(_iter_indices(expr))
        if index.label in label_map
    }
    return expr.xreplace(replacements)


def _iter_indices(expr):
    if isinstance(expr, Index):
        yield expr
        return
    for arg in getattr(expr, "args", ()):
        yield from _iter_indices(arg)


def TSeries(expr, series_spec):
    return _TSeries_cached(sp.sympify(expr), tuple(series_spec))


@_series_cache
def _TSeries_cached(expr, series_spec):
    symbol, start, order = series_spec
    prepared = _prepare_series_expr(expr, symbol, start, order)
    protected, restore_indices = _protect_indices_for_series(prepared)
    expanded = sp.series(protected, symbol, start, order + 1).removeO().xreplace(restore_indices)
    return _expand_tensor_powers_in_series(expanded)


def _protect_indices_for_series(expr):
    indices = list(dict.fromkeys(_iter_indices(expr)))
    if not indices:
        return expr, {}
    replacements = {index: sp.Dummy(f"mathgr_idx_{pos}") for pos, index in enumerate(indices)}
    return expr.xreplace(replacements), {replacement: index for index, replacement in replacements.items()}


@_series_cache
def _prepare_series_expr(expr, symbol, start, order):
    expr = sp.sympify(expr)
    if is_pdt(expr):
        return _series_partial_derivative(expr, symbol, start, order)
    if isinstance(expr, Index) or not getattr(expr, "args", ()):
        return expr
    rewritten_args = tuple(_prepare_series_expr(arg, symbol, start, order) for arg in expr.args)
    if rewritten_args == expr.args:
        return expr
    return expr.func(*rewritten_args)


def _series_partial_derivative(expr, symbol, start, order):
    base, derivative_indices = pdt_parts(expr)
    differentiated = TSeries(base, (symbol, start, order))
    for index in derivative_indices:
        differentiated = _drop_series_symbol_derivatives(Pd(differentiated, index), symbol)
    return differentiated


@_series_cache
def _drop_series_symbol_derivatives(expr, symbol):
    expr = sp.sympify(expr)
    if is_pdt(expr):
        base, _derivative_indices = pdt_parts(expr)
        if base == symbol:
            return sp.Integer(0)
    if isinstance(expr, Index) or not getattr(expr, "args", ()):
        return expr
    rewritten_args = tuple(_drop_series_symbol_derivatives(arg, symbol) for arg in expr.args)
    if rewritten_args == expr.args:
        return expr
    return expr.func(*rewritten_args)


@_series_cache
def _expand_tensor_powers_in_series(expr):
    expr = sp.sympify(expr)
    if isinstance(expr, sp.Pow) and expr.exp.is_Integer and expr.exp > 1 and any(True for _ in _iter_indices(expr.base)):
        return TPower(expr.base, int(expr.exp))
    if isinstance(expr, Index):
        return expr
    if not getattr(expr, "args", ()):
        return expr
    rewritten_args = tuple(_expand_tensor_powers_in_series(arg) for arg in expr.args)
    return expr if rewritten_args == expr.args else expr.func(*rewritten_args)


def _expand_scalar_contraction_power_product_in_series(expr):
    factors = list(expr.args)
    powered_positions = [
        pos
        for pos, factor in enumerate(factors)
        if isinstance(factor, sp.Pow)
        and factor.exp.is_Integer
        and factor.exp > 1
        and any(True for _ in _iter_indices(factor.base))
    ]
    if len(powered_positions) < 2:
        return None

    exponents = {int(factors[pos].exp) for pos in powered_positions}
    if len(exponents) != 1:
        return None
    exponent = exponents.pop()

    base_product = sp.Mul(*(factors[pos].base for pos in powered_positions), evaluate=False)
    dummy_labels = _dummy_labels(base_product)
    if not dummy_labels:
        return None

    rest_factors = [factor for pos, factor in enumerate(factors) if pos not in set(powered_positions)]
    rest_labels = {
        index.label
        for factor in rest_factors
        for index in _iter_indices(factor)
        if isinstance(index.label, str)
    }
    if rest_labels & set(dummy_labels):
        return None

    split_factors = _tensor_power_factors(base_product, exponent)
    return sp.Mul(*(rest_factors + split_factors))


def _ordered_unique(values):
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def CollectEps(vars=None, op=Simp):
    variables = _as_list(sp.Symbol("tmp") if vars is None else vars)

    def _collect(expr):
        collected = sp.collect(sp.expand(sp.sympify(expr)), Eps, evaluate=False)
        terms = []
        for eps_power, coeff in collected.items():
            terms.append(eps_power * _collect_requested_variables(op(coeff), variables))
        return sp.Add(*terms)

    return _collect


def _collect_requested_variables(expr, variables):
    expr = sp.sympify(expr)
    if not variables:
        return expr
    protected, restore_indices = _protect_indices_for_series(expr)
    try:
        return sp.factor_terms(protected, *variables).xreplace(restore_indices)
    except TypeError:
        return expr


def SS(order: int, vars=None, op=Simp):
    def _ss(expr):
        return CollectEps(vars, op)(TSeries(expr, (Eps, 0, order)))

    return _ss


def OO(order: int, vars=None, op=Simp):
    def _oo(expr):
        return CollectEps(vars, op)(sp.expand(SS(order, vars, op)(expr)).coeff(Eps, order))

    return _oo


def LocalToK(expr, index_type=DN, *, Momentum=k):
    return apply2term(lambda term: _local_to_k_term(term, index_type, Momentum), expr)


def _local_to_k_term(term, index_type=DN, momentum=k):
    term = sp.sympify(term)
    variables = set(_differentiated_variables(term, index_type))
    transformed = _assign_local_momenta(term, variables, momentum, count(1))
    return _rewrite_pdt_to_momentum(transformed, index_type)


def _assign_local_momenta(expr, variables, momentum, positions):
    expr = sp.sympify(expr)
    if isinstance(expr, (Index, MomentumLabel)):
        return expr
    if expr in variables:
        momentum_label = _coerce_momentum_label(momentum, next(positions))
        return _momentum_replacement(expr, momentum_label)
    if is_pdt(expr):
        base, derivative_indices = pdt_parts(expr)
        return PdT(_assign_local_momenta(base, variables, momentum, positions), PdVars(*derivative_indices))
    if not getattr(expr, "args", ()):
        return expr
    rewritten_args = tuple(_assign_local_momenta(arg, variables, momentum, positions) for arg in expr.args)
    if rewritten_args == expr.args:
        return expr
    return expr.func(*rewritten_args)


def _coerce_momentum_label(momentum, position):
    label = momentum(position)
    if isinstance(label, MomentumLabel):
        return label
    head_name = tensor_head_name(label)
    args = tensor_args(label)
    if head_name is not None and len(args) == 1:
        return MomentumLabel(head_name, args[0])
    return label


def _differentiated_variables(expr, index_type):
    variables = OrderedDict()
    for node in sp.preorder_traversal(expr):
        if is_pdt(node):
            base, derivative_indices = pdt_parts(node)
            if any(_is_index_of_type(index, index_type) for index in derivative_indices):
                variables[_base_variable(base)] = None
        elif tensor_head_name(node) is not None and _is_local_indexed_tensor_variable(node):
            variables[node] = None
    for symbol in _scalar_symbols(expr):
        if _symbol_has_derivative(symbol, index_type):
            variables[symbol] = None
    return sorted(variables.keys(), key=str)


def _scalar_symbols(expr):
    found = OrderedDict()

    def visit(node):
        if isinstance(node, Index):
            return
        if isinstance(node, MomentumLabel):
            return
        if is_pdt(node):
            base, _derivative_indices = pdt_parts(node)
            visit(base)
            return
        head_name = tensor_head_name(node)
        if head_name is not None:
            for arg in tensor_args(node):
                visit(arg)
            return
        if isinstance(node, sp.Symbol):
            found[node] = None
            return
        for arg in getattr(node, "args", ()):
            visit(arg)

    visit(sp.sympify(expr))
    return sorted(found.keys(), key=lambda item: item.name)


def _is_local_indexed_tensor_variable(expr):
    args = tensor_args(expr)
    return bool(args) and any(isinstance(arg, Index) for arg in args) and not any(
        isinstance(arg, MomentumLabel) for arg in args
    )


def _base_variable(base):
    head_name = tensor_head_name(base)
    if head_name is not None:
        return base
    return base


def _momentum_replacement(var, momentum_label):
    head_name = tensor_head_name(var)
    if head_name is not None:
        return tensor(head_name)(momentum_label, *tensor_args(var))
    return tensor(str(var))(momentum_label)


def _symbol_has_derivative(symbol, index_type):
    return PdT(symbol, PdVars(index_type("__mathgr_test__"))) != 0


def _is_index_of_type(value, index_type):
    return isinstance(value, Index) and value.head_name == index_type.name


def _rewrite_pdt_to_momentum(expr, index_type):
    if isinstance(expr, (Index, MomentumLabel)):
        return expr
    if is_pdt(expr):
        base, derivative_indices = pdt_parts(expr)
        rewritten_base = _rewrite_pdt_to_momentum(base, index_type)
        momentum_factors = []
        remaining_indices = []
        for index in derivative_indices:
            if _is_index_of_type(index, index_type):
                momentum_factors.append(_momentum_for_base(rewritten_base)(index))
            else:
                remaining_indices.append(index)
        derivative = PdT(rewritten_base, PdVars(*remaining_indices)) if remaining_indices else rewritten_base
        return sp.Mul(*(momentum_factors + [derivative]))
    if not getattr(expr, "args", ()):
        return expr
    rewritten_args = tuple(_rewrite_pdt_to_momentum(arg, index_type) for arg in expr.args)
    if rewritten_args == expr.args:
        return expr
    return expr.func(*rewritten_args)


def _momentum_for_base(base):
    args = tensor_args(base)
    if args:
        for arg in args:
            if isinstance(arg, MomentumLabel):
                return arg
    return tensor("k")(0)
