from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .tensor import Index, Pd, PdT, PdVars, Pm2, Simp, is_pdt, is_pm2, pdt_parts, pm2_parts, tensor_head_name


TrySimpPreferredPattern = []
TrySimpPreferredPatternStrength = sp.Rational(1, 10000)
_TRY_SIMP_CONTEXT_SIMPLIFY_OP_LIMIT = 200


class _PdHold(sp.Function):
    nargs = 2
    _op_priority = 1000

    @classmethod
    def eval(cls, expr, index):
        if expr == 0:
            return sp.Integer(0)
        return None

    def __neg__(self):
        return PdHold(-self.args[0], self.args[1])

    def __mul__(self, other):
        other = sp.sympify(other)
        if other.is_number:
            return PdHold(other * self.args[0], self.args[1])
        if other.could_extract_minus_sign():
            return -other * PdHold(-self.args[0], self.args[1])
        return super().__mul__(other)

    def __rmul__(self, other):
        other = sp.sympify(other)
        if other.is_number:
            return PdHold(other * self.args[0], self.args[1])
        if other.could_extract_minus_sign():
            return -other * PdHold(-self.args[0], self.args[1])
        return super().__rmul__(other)

    def __add__(self, other):
        other = sp.sympify(other)
        if isinstance(other, _PdHold) and other.args[1] == self.args[1]:
            return PdHold(self.args[0] + other.args[0], self.args[1])
        return super().__add__(other)

    def __radd__(self, other):
        other = sp.sympify(other)
        if isinstance(other, _PdHold) and other.args[1] == self.args[1]:
            return PdHold(other.args[0] + self.args[0], self.args[1])
        return super().__radd__(other)


def PdHold(expr, index):
    expr, index = _normalize_pdhold_dummy(sp.sympify(expr), sp.sympify(index))
    return _PdHold(expr, index)


def _normalize_pdhold_dummy(expr, index):
    if not isinstance(index, Index) or index.head_name == "DE" or index.label == "(PdId)":
        return expr, index
    replacements = {
        node: node.with_label("(PdId)")
        for node in sp.preorder_traversal(expr)
        if isinstance(node, Index) and node.label == index.label
    }
    if not replacements:
        return expr, index
    return expr.xreplace(replacements), index.with_label("(PdId)")


class _IdHold(sp.Function):
    nargs = 1

    @classmethod
    def eval(cls, expr):
        if expr == 0:
            return sp.Integer(0)
        return None


def IdHold(expr):
    return _IdHold(sp.sympify(expr))


@dataclass(frozen=True)
class _IbpVar:
    target: sp.Expr

    def __call__(self, expr):
        expr = _drop_boundary_terms(expr)
        return sp.Integer(
            10000 * _count_target_derivatives(expr, self.target, 3)
            + 100 * _count_target_derivatives(expr, self.target, 2)
            + _count_target_derivatives(expr, self.target, 1)
        ) + IbpCountLeaf(expr)


def IbpVar(target):
    return _IbpVar(sp.sympify(target))


def IbpCountLeaf(expr):
    expr = _drop_boundary_terms(expr)
    return sp.Integer(sp.count_ops(expr) + _count_pm2(expr) * 100 + _count_time_derivative_fields(expr))


def IbpCountTerm(expr):
    expr = _drop_boundary_terms(expr)
    expanded = sp.expand(expr)
    term_count = len(expanded.args) if isinstance(expanded, sp.Add) else 1
    return sp.Integer(term_count + _count_pm2(expr) * 100)


def IbpCountPt2(expr):
    expr = _drop_boundary_terms(expr)
    return sp.Integer(_count_pdt_matching(expr, _has_two_time_derivatives) * 1000 + IbpCountLeaf(expr))


def IbpCountPd2(expr):
    expr = _drop_boundary_terms(expr)
    return sp.Integer(_count_pdt_matching(expr, _has_two_spatial_derivatives) * 1000 + IbpCountLeaf(expr))


def IbpStd2(expr):
    expr = _drop_boundary_terms(expr)
    mixed_first_derivative_count = sum(1 for node in _iter_rank_nodes(expr) if _is_self_first_derivative_product(node))
    return sp.Integer(IbpCountPt2(expr) * 1000 + IbpCountPd2(expr) * 100 + mixed_first_derivative_count * 10 + IbpCountLeaf(expr))


def IbpReduceOrder(vars):
    targets = tuple(sp.sympify(var) for var in (vars if isinstance(vars, (list, tuple, set, sp.Tuple)) else [vars]))

    def _rank(expr):
        terms = _expanded_terms(_drop_boundary_terms(expr))
        order_rank = sum(sp.Integer(100) ** (5 - _count_target_order(term, targets)) for term in terms)
        return sp.sympify(order_rank + IbpCountLeaf(expr) + IbpCountTerm(expr))

    return _rank


def IbpRuleWithForbiddenPatterrn(rule, ptn):
    def _rank(expr):
        expr = sp.sympify(expr)
        if _contains_forbidden(expr, ptn):
            return sp.oo
        return sp.sympify(rule(expr))

    return _rank


IbpRuleWithForbiddenPattern = IbpRuleWithForbiddenPatterrn


def TrySimp(expr, rule, *, Rank=None, Level=1):
    rank = Rank or _default_rank
    current = sp.sympify(expr)
    depth = max(1, int(Level))
    for _ in range(max(1, int(Level)) * 20):
        candidates = _try_simp_candidates(current, rule, depth=depth)
        if not candidates:
            return current
        candidates = _try_simp_rank_candidates(candidates)
        best = min(candidates, key=lambda candidate: (_rank_value(rank, candidate), sp.default_sort_key(candidate)))
        if _rank_value(rank, best) >= _rank_value(rank, current):
            return current
        current = best
    return current


def TrySimp2(expr, rule, *, Rank=None, Level=2):
    return TrySimp(expr, rule, Rank=Rank, Level=max(2, int(Level)))


def IbpNB(expr, **options):
    return _drop_boundary_terms(Ibp(expr, **options))


def IbpVariation(expr, target):
    target = sp.sympify(target)
    current = Simp(expr)
    for _ in range(20):
        replaced = _ibp_variation_step(current, target)
        if replaced == current:
            return replaced
        current = Simp(replaced)
    return current


def Pm2Simp(expr):
    simplified, changed = _simplify_pm2_rules(sp.expand(Simp(expr)))
    return Simp(simplified) if changed else simplified


def Pm2Rules(expr):
    simplified, changed = _simplify_pm2_rules(sp.expand(sp.sympify(expr)))
    return Simp(simplified) if changed else sp.sympify(expr)


def IbpRules(expr):
    expr = sp.sympify(expr)
    if is_pdt(expr):
        return expr
    pm2_simplified, pm2_changed = _simplify_pm2_rules(expr)
    if pm2_changed:
        return Simp(pm2_simplified)
    reduced_derivative_power = _ibp_reduced_derivative_power_term(expr)
    if reduced_derivative_power != expr:
        return Simp(reduced_derivative_power)
    first_derivative_power = _ibp_first_derivative_power_term(expr)
    if first_derivative_power != expr:
        return Simp(first_derivative_power)
    cubic_second_derivative = _ibp_cubic_second_derivative_term(expr)
    if cubic_second_derivative != expr:
        return Simp(cubic_second_derivative)
    squared_second_derivative = _ibp_squared_second_derivative_term(expr)
    if squared_second_derivative != expr:
        return Simp(squared_second_derivative)
    symmetric_mixed_second_derivative = _ibp_symmetric_mixed_second_derivative_term(expr)
    if symmetric_mixed_second_derivative != expr:
        return Simp(symmetric_mixed_second_derivative)
    symmetric_first_second_derivative = _ibp_symmetric_first_second_derivative_term(expr)
    if symmetric_first_second_derivative != expr:
        return Simp(symmetric_first_second_derivative)
    first_derivative_product = _ibp_two_first_derivative_product(expr)
    if first_derivative_product != expr:
        return Simp(first_derivative_product)
    second_derivative = _ibp_second_derivative_term(expr)
    if second_derivative != expr:
        return Simp(second_derivative)
    first_derivative = _ibp_first_derivative_term(expr)
    if first_derivative != expr:
        return Simp(first_derivative)
    return expr


def Ibp(expr, *, Rank=None, Rule=None, Level=1):
    expr = sp.expand(sp.sympify(expr))
    if Rule is not None:
        return TrySimp(Simp(expr), Rule, Rank=Rank or IbpCountLeaf, Level=Level)
    if isinstance(expr, sp.Add):
        return TrySimp(Simp(expr), IbpRules, Rank=Rank or IbpCountLeaf, Level=Level)
    expr, pm2_changed = _simplify_pm2_rules(expr)
    if pm2_changed:
        return Simp(expr)
    if isinstance(Rank, _IbpVar):
        return _ibp_var_term(expr, Rank.target)
    if Rank is IbpCountPt2:
        return _ibp_second_derivative_term(expr)
    return _ibp_first_derivative_term(expr)


def Ibp2(expr, *, Rank=None, Rule=None, Level=2):
    return Ibp(expr, Rank=Rank, Rule=Rule, Level=max(2, int(Level)))


def _simplify_pm2_rules(expr):
    if isinstance(expr, sp.Add):
        changed = False
        terms = []
        for term in expr.args:
            new_term, term_changed = _simplify_pm2_term(term)
            changed = changed or term_changed
            terms.append(new_term)
        return sp.expand(sp.Add(*terms)), changed
    return _simplify_pm2_term(expr)


def _simplify_pm2_term(term):
    term = sp.sympify(term)
    transformed, changed = _apply_pm2_derivative_self_adjoint_rule(term)
    if changed:
        return transformed, True
    transformed, changed = _apply_pm2_square_laplacian_rule(term)
    if changed:
        return transformed, True
    transformed, changed = _apply_pm2_laplacian_rule(term)
    if changed:
        return transformed, True
    return _canonicalize_pm2_self_adjoint_term(term)


def _apply_pm2_derivative_self_adjoint_rule(term):
    pdt_pm2_factor = _single_pdt_pm2_factor(term)
    if pdt_pm2_factor is None:
        return term, False
    coeff, pdt_expr = pdt_pm2_factor
    inner_pm2, variables = pdt_parts(pdt_expr)
    inner, index_type = pm2_parts(inner_pm2)
    if Pd(coeff, index_type("__mathgr_pm2_test__")) == 0:
        return term, False
    return Pm2(coeff, index_type) * PdT(inner, PdVars(*variables)), True


def _canonicalize_pm2_self_adjoint_term(term):
    pm2_factor = _single_pm2_factor(term)
    if pm2_factor is None:
        return term, False
    coeff, pm2_expr = pm2_factor
    numeric, nonnumeric_coeff = coeff.as_coeff_Mul()
    if nonnumeric_coeff == 1:
        return term, False
    inner, index_type = pm2_parts(pm2_expr)
    if Pd(nonnumeric_coeff, index_type("__mathgr_pm2_test__")) == 0:
        return term, False
    if sp.default_sort_key(nonnumeric_coeff) <= sp.default_sort_key(inner):
        return term, False
    return numeric * inner * Pm2(nonnumeric_coeff, index_type), True


def _apply_pm2_square_laplacian_rule(term):
    pm2_factor = _single_pm2_factor(term)
    if pm2_factor is None:
        return term, False
    outer_coeff, pm2_expr = pm2_factor
    inner, index_type = pm2_parts(pm2_expr)
    laplacian = _find_pm2_laplacian_factor(inner, index_type)
    if laplacian is None:
        return term, False
    coeff_inside_pm2, base, reduced_vars, _one_removed_vars, laplacian_index = laplacian
    reduced = PdT(base, PdVars(*reduced_vars)) if reduced_vars else base
    coefficient = _remove_single_factor(coeff_inside_pm2, reduced)
    if coefficient is None:
        return term, False
    reduced_derivative = Pd(reduced, laplacian_index)
    correction = (
        Pd(Pd(coefficient, laplacian_index), laplacian_index) * reduced**2
        + 4 * Pd(coefficient, laplacian_index) * reduced * reduced_derivative
        + 2 * coefficient * reduced_derivative**2
    )
    return outer_coeff * (coefficient * reduced**2 - Pm2(correction, index_type)) / 2, True


def _apply_pm2_laplacian_rule(term):
    pm2_factor = _single_pm2_factor(term)
    if pm2_factor is None:
        return term, False
    outer_coeff, pm2_expr = pm2_factor
    inner, index_type = pm2_parts(pm2_expr)
    laplacian = _find_pm2_laplacian_factor(inner, index_type)
    if laplacian is None:
        return term, False
    coeff_inside_pm2, base, reduced_vars, one_removed_vars, laplacian_index = laplacian
    reduced = PdT(base, PdVars(*reduced_vars)) if reduced_vars else base
    one_removed = PdT(base, PdVars(*one_removed_vars)) if one_removed_vars else base
    transformed = (
        reduced * coeff_inside_pm2
        - 2 * Pm2(one_removed * Pd(coeff_inside_pm2, laplacian_index), index_type)
        - Pm2(reduced * Pd(Pd(coeff_inside_pm2, laplacian_index), laplacian_index), index_type)
    )
    return outer_coeff * transformed, True


def _single_pm2_factor(term):
    factors = list(term.args) if isinstance(term, sp.Mul) else [term]
    matches = [(pos, factor) for pos, factor in enumerate(factors) if is_pm2(factor)]
    if len(matches) != 1:
        return None
    pos, factor = matches[0]
    coeff_factors = factors[:pos] + factors[pos + 1 :]
    coeff = sp.Mul(*coeff_factors) if coeff_factors else sp.Integer(1)
    return coeff, factor


def _single_pdt_pm2_factor(term):
    factors = list(term.args) if isinstance(term, sp.Mul) else [term]
    matches = []
    for pos, factor in enumerate(factors):
        if not is_pdt(factor):
            continue
        base, _variables = pdt_parts(factor)
        if is_pm2(base):
            matches.append((pos, factor))
    if len(matches) != 1:
        return None
    pos, factor = matches[0]
    coeff_factors = factors[:pos] + factors[pos + 1 :]
    coeff = sp.Mul(*coeff_factors) if coeff_factors else sp.Integer(1)
    return coeff, factor


def _remove_single_factor(expr, target):
    factors = list(expr.args) if isinstance(expr, sp.Mul) else [expr]
    for pos, factor in enumerate(factors):
        if factor != target:
            continue
        rest = factors[:pos] + factors[pos + 1 :]
        return sp.Mul(*rest) if rest else sp.Integer(1)
    return None


def _find_pm2_laplacian_factor(expr, index_type):
    factors = list(expr.args) if isinstance(expr, sp.Mul) else [expr]
    for pos, factor in enumerate(factors):
        if not is_pdt(factor):
            continue
        base, variables = pdt_parts(factor)
        pair = _find_repeated_derivative_pair(variables, index_type)
        if pair is None:
            continue
        first, second = pair
        coeff_factors = factors[:pos] + factors[pos + 1 :]
        coeff = sp.Mul(*coeff_factors) if coeff_factors else sp.Integer(1)
        reduced_vars = tuple(var for var_pos, var in enumerate(variables) if var_pos not in {first, second})
        one_removed_vars = tuple(var for var_pos, var in enumerate(variables) if var_pos != second)
        return coeff, base, reduced_vars, one_removed_vars, variables[first]
    return None


def _find_repeated_derivative_pair(variables, index_type):
    seen = {}
    for pos, variable in enumerate(variables):
        if not _is_index_of_type(variable, index_type):
            continue
        if variable.label in seen:
            return seen[variable.label], pos
        seen[variable.label] = pos
    return None


def _is_index_of_type(value, index_type):
    return isinstance(value, Index) and value.head_name == index_type.name


def _default_rank(expr):
    return sp.count_ops(expr)


def _rank_value(rank, expr):
    expr = sp.sympify(expr)
    return sp.sympify(rank(expr)) - sp.sympify(TrySimpPreferredPatternStrength) * _preferred_pattern_count(expr)


def _expanded_terms(expr):
    expanded = sp.expand(sp.sympify(expr))
    if expanded == 0:
        return []
    return list(expanded.args) if isinstance(expanded, sp.Add) else [expanded]


def _count_target_order(expr, targets):
    expr = sp.sympify(expr)
    if any(expr == target for target in targets):
        return 1
    if isinstance(expr, sp.Pow) and expr.exp.is_Integer and expr.exp > 0:
        return int(expr.exp) * _count_target_order(expr.base, targets)
    return sum(_count_target_order(arg, targets) for arg in expr.args)


def _count_target_derivatives(expr, target, order):
    count = 0
    for node in _iter_rank_nodes(expr):
        if not is_pdt(node):
            continue
        base, indices = pdt_parts(node)
        if len(indices) >= order and _contains_target(base, target):
            count += 1
    return count


def _contains_forbidden(expr, ptn):
    if isinstance(ptn, (list, tuple, set, sp.Tuple)):
        return any(_contains_forbidden(expr, item) for item in ptn)
    if isinstance(ptn, type):
        return any(isinstance(node, ptn) for node in sp.preorder_traversal(expr))
    if isinstance(ptn, sp.Expr):
        return any(_expr_matches_pattern(node, ptn) for node in sp.preorder_traversal(expr))
    if callable(ptn):
        for node in sp.preorder_traversal(expr):
            try:
                if ptn(node):
                    return True
            except TypeError:
                continue
        return False
    return False


def _preferred_pattern_count(expr):
    patterns = tuple(TrySimpPreferredPattern)
    if not patterns:
        return 0
    expr = _drop_boundary_terms(expr)
    return sum(_count_pattern(expr, pattern) for pattern in patterns)


def _count_pattern(expr, pattern):
    if isinstance(pattern, (list, tuple, set, sp.Tuple)):
        return sum(_count_pattern(expr, item) for item in pattern)
    if isinstance(pattern, type):
        return sum(1 for node in sp.preorder_traversal(expr) if isinstance(node, pattern))
    if isinstance(pattern, sp.Expr):
        return sum(1 for node in sp.preorder_traversal(expr) if _expr_matches_pattern(node, pattern))
    if callable(pattern):
        total = 0
        for node in sp.preorder_traversal(expr):
            try:
                total += 1 if pattern(node) else 0
            except TypeError:
                continue
        return total
    return 0


def _expr_matches_pattern(expr, pattern):
    expr = sp.sympify(expr)
    pattern = sp.sympify(pattern)
    if not pattern.has(sp.Wild):
        return expr == pattern
    try:
        return expr.match(pattern) is not None
    except (TypeError, ValueError):
        return False


def _try_simp_candidates(expr, rule, *, depth=1):
    expr = sp.sympify(expr)
    candidates = _try_simp_one_step_candidates(expr, rule)
    if depth > 1:
        for candidate in tuple(candidates):
            candidates.extend(_try_simp_candidates(candidate, rule, depth=depth - 1))
    return _unique_candidates(candidate for candidate in candidates if candidate != expr)


def _try_simp_rank_candidates(candidates):
    expanded = []
    for candidate in candidates:
        expanded.append(candidate)
        expanded.append(_simplify_replacement_context(candidate))
        if sp.count_ops(candidate) <= _TRY_SIMP_CONTEXT_SIMPLIFY_OP_LIMIT:
            expanded.append(Simp(candidate))
    return _unique_candidates(expanded)


def _simplify_replacement_context(expr):
    expr = sp.sympify(expr)
    if isinstance(expr, sp.Add):
        return sp.Add(*(_simplify_replacement_context(arg) for arg in expr.args))
    if isinstance(expr, sp.Mul):
        rewritten_args = tuple(_simplify_replacement_context(arg) for arg in expr.args)
        current = expr if rewritten_args == expr.args else sp.Mul(*rewritten_args)
        if _contains_mathgr_structure(current) and any(isinstance(arg, sp.Add) for arg in current.args):
            return sp.expand_mul(current)
        return current
    if isinstance(expr, (Index,)) or not getattr(expr, "args", ()):
        return expr
    rewritten_args = tuple(_simplify_replacement_context(arg) for arg in expr.args)
    if rewritten_args == expr.args:
        return expr
    return expr.func(*rewritten_args)


def _contains_mathgr_structure(expr):
    return any(is_pdt(node) or is_pm2(node) or tensor_head_name(node) is not None for node in sp.preorder_traversal(expr))


def _try_simp_one_step_candidates(expr, rule):
    candidates = _direct_rule_candidates(expr, rule)
    for pos, arg in enumerate(expr.args):
        for arg_candidate in _try_simp_one_step_candidates(arg, rule):
            args = list(expr.args)
            args[pos] = arg_candidate
            candidates.append(expr.func(*args))
    return _unique_candidates(candidate for candidate in candidates if candidate != expr)


def _direct_rule_candidates(expr, rule):
    if callable(rule):
        result = rule(expr)
        return _coerce_rule_result(result)
    if isinstance(rule, dict):
        return _replacement_rule_candidates(expr, rule.items())
    if _is_rule_pair(rule):
        return _replacement_rule_candidates(expr, (rule,))
    if isinstance(rule, (list, tuple)):
        return _replacement_rule_candidates(expr, rule)
    raise TypeError("TrySimp rule must be a callable, dict, or sequence of (old, new) pairs.")


def _is_rule_pair(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    left, right = value
    return not (_looks_like_rule_pair(left) and _looks_like_rule_pair(right))


def _looks_like_rule_pair(value):
    return isinstance(value, (list, tuple)) and len(value) == 2


def _replacement_rule_candidates(expr, items):
    replacements = tuple((sp.sympify(old), new if callable(new) else sp.sympify(new)) for old, new in items)
    exact_replacements = {
        old: new
        for old, new in replacements
        if not callable(new) and not old.has(sp.Wild)
    }
    candidates = [expr.xreplace(exact_replacements)]
    for old, new in replacements:
        if callable(new):
            if old.has(sp.Wild):
                candidates.append(expr.replace(old, lambda **matches: sp.sympify(new(**matches))))
            else:
                candidates.append(expr.replace(lambda node, old=old: node == old, lambda node: sp.sympify(new(node))))
            continue
        if old.has(sp.Wild):
            candidates.append(expr.replace(old, new))
    return candidates


def _coerce_rule_result(result):
    if result is None:
        return []
    if isinstance(result, (list, tuple, set, sp.Tuple)):
        return [sp.sympify(candidate) for candidate in result]
    return [sp.sympify(result)]


def _unique_candidates(candidates):
    unique = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _drop_boundary_terms(expr):
    expr = sp.sympify(expr)
    if isinstance(expr, (_PdHold, _IdHold)):
        return sp.Integer(0)
    if expr.args:
        return expr.func(*(_drop_boundary_terms(arg) for arg in expr.args))
    return expr


def _ibp_variation_step(expr, target):
    expr = sp.expand(sp.sympify(expr))
    if isinstance(expr, sp.Add):
        return sp.Add(*(_ibp_variation_step(term, target) for term in expr.args))
    match = _find_variation_derivative_factor(expr, target)
    if match is None:
        return expr
    coeff, base, indices = match
    derivative_index = indices[-1]
    rest = indices[:-1]
    reduced = PdT(base, PdVars(*rest)) if rest else base
    return -reduced * Pd(coeff, derivative_index)


def _find_variation_derivative_factor(term, target):
    factors = list(term.args) if isinstance(term, sp.Mul) else [term]
    for pos, factor in enumerate(factors):
        if not is_pdt(factor):
            continue
        base, indices = pdt_parts(factor)
        if not _contains_target(base, target):
            continue
        coeff_factors = factors[:pos] + factors[pos + 1 :]
        coeff = sp.Mul(*coeff_factors) if coeff_factors else sp.Integer(1)
        if _contains_target(coeff, target):
            continue
        return coeff, base, indices
    return None


def _contains_target(expr, target):
    expr = sp.sympify(expr)
    return expr == target or expr.has(target)


def _iter_rank_nodes(expr):
    expr = sp.sympify(expr)
    if isinstance(expr, _PdHold):
        return
    yield expr
    for arg in expr.args:
        yield from _iter_rank_nodes(arg)


def _count_pm2(expr):
    return sum(1 for node in _iter_rank_nodes(expr) if is_pm2(node))


def _count_time_derivative_fields(expr):
    return sum(1 for node in _iter_rank_nodes(expr) if is_pdt(node) and _pdt_has_time_derivative_field(node))


def _count_pdt_matching(expr, predicate):
    return sum(1 for node in _iter_rank_nodes(expr) if is_pdt(node) and predicate(pdt_parts(node)[1]))


def _pdt_has_time_derivative_field(node):
    base, _indices = pdt_parts(node)
    return any(isinstance(arg, Index) and arg.head_name == "DE" and arg.label == 0 for arg in getattr(base, "args", ()))


def _has_two_time_derivatives(indices):
    return len(indices) >= 2 and all(isinstance(index, Index) and index.head_name == "DE" for index in indices[:2])


def _has_two_spatial_derivatives(indices):
    return len(indices) >= 2 and all(isinstance(index, Index) and not index.head.explicit for index in indices[:2])


def _is_self_first_derivative_product(node):
    if not isinstance(node, sp.Mul):
        return False
    pdt_factors = [factor for factor in node.args if is_pdt(factor)]
    if len(pdt_factors) < 2:
        return False
    first_base, first_indices = pdt_parts(pdt_factors[0])
    return any(pdt_parts(factor)[0] == first_base and len(pdt_parts(factor)[1]) == len(first_indices) for factor in pdt_factors[1:])


def _ibp_var_term(term, target):
    match = _find_derivative_factor(term, lambda base, _indices: base.has(target) or base == target)
    if match is None:
        return term
    coeff, base, indices = match
    first, rest = indices[0], indices[1:]
    reduced = PdT(base, PdVars(*rest)) if rest else base
    return -reduced * Pd(coeff, first) + PdHold(coeff * reduced, first)


def _ibp_second_derivative_term(term):
    match = _find_derivative_factor(term, lambda _base, indices: len(indices) >= 2)
    if match is None:
        return term
    coeff, base, indices = match
    first, rest = indices[0], indices[1:]
    reduced = PdT(base, PdVars(*rest)) if rest else base
    return -reduced * Pd(coeff, first) + PdHold(coeff * reduced, first)


def _ibp_two_first_derivative_product(term):
    factors = list(term.args) if isinstance(term, sp.Mul) else [term]
    derivative_factors = []
    for pos, factor in enumerate(factors):
        if not is_pdt(factor):
            continue
        base, indices = pdt_parts(factor)
        if len(indices) >= 1:
            derivative_factors.append((pos, base, indices))
    if len(derivative_factors) < 2:
        return term

    (first_pos, first_base, first_indices), (second_pos, second_base, second_indices) = derivative_factors[:2]
    a, first_rest = first_indices[0], first_indices[1:]
    b, second_rest = second_indices[0], second_indices[1:]
    first_reduced = PdT(first_base, PdVars(*first_rest)) if first_rest else first_base
    second_reduced = PdT(second_base, PdVars(*second_rest)) if second_rest else second_base
    h_factors = [factor for pos, factor in enumerate(factors) if pos not in {first_pos, second_pos}]
    h = sp.Mul(*h_factors) if h_factors else sp.Integer(1)

    return (
        PdHold(first_reduced * Pd(second_reduced, b) * h, a)
        - PdHold(first_reduced * Pd(second_reduced, a) * h, b)
        + Pd(first_reduced, b) * Pd(second_reduced, a) * h
        + first_reduced * Pd(second_reduced, a) * Pd(h, b)
        - first_reduced * Pd(second_reduced, b) * Pd(h, a)
    )


def _ibp_squared_second_derivative_term(term):
    factors = list(term.args) if isinstance(term, sp.Mul) else [term]
    for pos, factor in enumerate(factors):
        if not isinstance(factor, sp.Pow) or factor.exp != 2 or not is_pdt(factor.base):
            continue
        base, indices = pdt_parts(factor.base)
        if len(indices) < 2:
            continue
        a, b, *rest = indices
        reduced = PdT(base, PdVars(*rest)) if rest else base
        coeff_factors = factors[:pos] + factors[pos + 1 :]
        coeff = sp.Mul(*coeff_factors) if coeff_factors else sp.Integer(1)
        return (
            PdHold(coeff * Pd(Pd(reduced, a), b) * Pd(reduced, b), a)
            - PdHold(coeff * Pd(Pd(reduced, a), a) * Pd(reduced, b), b)
            - Pd(coeff, a) * Pd(Pd(reduced, a), b) * Pd(reduced, b)
            + Pd(coeff, b) * Pd(Pd(reduced, a), a) * Pd(reduced, b)
            + coeff * Pd(Pd(reduced, a), a) * Pd(Pd(reduced, b), b)
        )
    return term


def _ibp_reduced_derivative_power_term(term):
    factors = list(term.args) if isinstance(term, sp.Mul) else [term]
    for pos, factor in enumerate(factors):
        if isinstance(factor, sp.Pow) and factor.exp.is_Integer and factor.exp > 2 and is_pdt(factor.base):
            base, indices = pdt_parts(factor.base)
            if len(indices) < 2:
                continue
            first, rest = indices[0], indices[1:]
            reduced = PdT(base, PdVars(*rest))
            coeff_factors = factors[:pos] + factors[pos + 1 :]
            coeff = sp.Mul(*coeff_factors) if coeff_factors else sp.Integer(1)
            derivative_power = factor.base ** (factor.exp - 1)
            return PdHold(coeff * reduced * derivative_power, first) - reduced * Pd(coeff * derivative_power, first)
        if not is_pdt(factor):
            continue
        base, indices = pdt_parts(factor)
        if len(indices) < 2:
            continue
        first, rest = indices[0], indices[1:]
        reduced = PdT(base, PdVars(*rest))
        coeff_factors = factors[:pos] + factors[pos + 1 :]
        coeff = sp.Mul(*coeff_factors) if coeff_factors else sp.Integer(1)
        power_rule = _ibp_power_rule(coeff, reduced, first)
        if power_rule is not None:
            return power_rule
    return term


def _ibp_first_derivative_power_term(term):
    factors = list(term.args) if isinstance(term, sp.Mul) else [term]
    for pos, factor in enumerate(factors):
        if not isinstance(factor, sp.Pow) or not factor.exp.is_Integer or factor.exp <= 1:
            continue
        if not is_pdt(factor.base):
            continue
        base, indices = pdt_parts(factor.base)
        if len(indices) != 1:
            continue
        derivative_index = indices[0]
        coeff_factors = factors[:pos] + factors[pos + 1 :]
        coeff = sp.Mul(*coeff_factors) if coeff_factors else sp.Integer(1)
        reduced = base
        derivative_power = factor.base ** (factor.exp - 1)
        return PdHold(coeff * reduced * derivative_power, derivative_index) - reduced * Pd(
            coeff * derivative_power, derivative_index
        )
    return term


def _ibp_cubic_second_derivative_term(term):
    factors = list(term.args) if isinstance(term, sp.Mul) else [term]
    for first_pos, first_factor in enumerate(factors):
        if not is_pdt(first_factor):
            continue
        first_base, first_indices = pdt_parts(first_factor)
        if len(first_indices) < 1:
            continue
        for square_pos, square_factor in enumerate(factors):
            if square_pos == first_pos:
                continue
            if not isinstance(square_factor, sp.Pow) or square_factor.exp != 2 or not is_pdt(square_factor.base):
                continue
            second_base, second_indices = pdt_parts(square_factor.base)
            if second_base != first_base or len(second_indices) < 2:
                continue
            c, *first_rest = first_indices
            a, b, *second_rest = second_indices
            if tuple(first_rest) != tuple(second_rest):
                continue
            coeff_factors = [
                factor for pos, factor in enumerate(factors) if pos not in {first_pos, square_pos}
            ]
            coeff = sp.Mul(*coeff_factors) if coeff_factors else sp.Integer(1)
            reduced = PdT(first_base, PdVars(*first_rest)) if first_rest else first_base
            return (
                PdHold(
                    coeff * Pd(reduced, c) * Pd(reduced, b) * Pd(Pd(reduced, a), b)
                    - coeff * Pd(reduced, b) ** 2 * Pd(Pd(reduced, a), c) / 2
                    - Pd(coeff, a) * Pd(reduced, b) ** 2 * Pd(reduced, c) / 2,
                    a,
                )
                - PdHold(coeff * Pd(reduced, c) * Pd(reduced, b) * Pd(Pd(reduced, a), a), b)
                + PdHold(coeff * Pd(reduced, b) ** 2 * Pd(Pd(reduced, a), a) / 2, c)
                + coeff * Pd(reduced, c) * Pd(Pd(reduced, a), a) * Pd(Pd(reduced, b), b)
                - Pd(coeff, c) * Pd(reduced, b) ** 2 * Pd(Pd(reduced, a), a) / 2
                + Pd(coeff, b) * Pd(reduced, c) * Pd(reduced, b) * Pd(Pd(reduced, a), a)
                + Pd(Pd(coeff, a), a) * Pd(reduced, c) * Pd(reduced, b) ** 2 / 2
                + Pd(coeff, a) * Pd(Pd(reduced, a), c) * Pd(reduced, b) ** 2
            )
    return term


def _ibp_symmetric_mixed_second_derivative_term(term):
    factors = list(term.args) if isinstance(term, sp.Mul) else [term]
    derivative_factors = []
    for pos, factor in enumerate(factors):
        if not is_pdt(factor):
            continue
        base, indices = pdt_parts(factor)
        derivative_factors.append((pos, factor, base, indices))

    for first in derivative_factors:
        for second in derivative_factors:
            if first[0] == second[0]:
                continue
            match = _symmetric_mixed_second_derivative_match(first, second)
            if match is None:
                continue
            first_pos, second_pos, first_factor, second_reduced, derivative_index, left, right = match
            coeff_factors = [factor for pos, factor in enumerate(factors) if pos not in {first_pos, second_pos}]
            coeff = sp.Mul(*coeff_factors) if coeff_factors else sp.Integer(1)
            if not _contains_index(coeff, left) or not _contains_index(coeff, right):
                continue
            if Simp(coeff - _swap_index_labels(coeff, left, right)) != 0:
                continue
            reduced_product = first_factor * second_reduced
            return PdHold(coeff * reduced_product / 2, derivative_index) - Pd(coeff, derivative_index) * reduced_product / 2
    return term


def _symmetric_mixed_second_derivative_match(first, second):
    first_pos, first_factor, first_base, first_indices = first
    second_pos, _second_factor, second_base, second_indices = second
    if first_base != second_base:
        return None
    if len(first_indices) < 1 or len(second_indices) != len(first_indices) + 1:
        return None
    left = first_indices[0]
    right = second_indices[0]
    derivative_index = second_indices[1]
    if not _is_implicit_index(left) or not _is_implicit_index(right):
        return None
    if tuple(first_indices[1:]) != tuple(second_indices[2:]):
        return None
    second_reduced_indices = (right,) + tuple(second_indices[2:])
    second_reduced = PdT(second_base, PdVars(*second_reduced_indices))
    return first_pos, second_pos, first_factor, second_reduced, derivative_index, left, right


def _ibp_symmetric_first_second_derivative_term(term):
    factors = list(term.args) if isinstance(term, sp.Mul) else [term]
    derivative_factors = []
    for pos, factor in enumerate(factors):
        if not is_pdt(factor):
            continue
        base, indices = pdt_parts(factor)
        derivative_factors.append((pos, factor, base, indices))

    for first in derivative_factors:
        for second in derivative_factors:
            if first[0] == second[0]:
                continue
            match = _symmetric_first_second_derivative_match(first, second)
            if match is None:
                continue
            first_pos, second_pos, first_factor, left_reduced, right_reduced, left, right, derivative_index = match
            coeff_factors = [factor for pos, factor in enumerate(factors) if pos not in {first_pos, second_pos}]
            coeff = sp.Mul(*coeff_factors) if coeff_factors else sp.Integer(1)
            if not _contains_index(coeff, left) or not _contains_index(coeff, right):
                continue
            if Simp(coeff - _swap_index_labels(coeff, left, right)) != 0:
                continue
            return (
                PdHold(coeff * first_factor * right_reduced, left)
                - PdHold(coeff * left_reduced * right_reduced / 2, derivative_index)
                - Pd(coeff, left) * first_factor * right_reduced
                + Pd(coeff, derivative_index) * left_reduced * right_reduced / 2
            )
    return term


def _symmetric_first_second_derivative_match(first, second):
    first_pos, first_factor, first_base, first_indices = first
    second_pos, _second_factor, second_base, second_indices = second
    if first_base != second_base:
        return None
    if len(first_indices) < 1 or len(second_indices) != len(first_indices) + 1:
        return None
    derivative_index = first_indices[0]
    left = second_indices[0]
    right = second_indices[1]
    if not _is_implicit_index(left) or not _is_implicit_index(right):
        return None
    if tuple(first_indices[1:]) != tuple(second_indices[2:]):
        return None
    rest = tuple(first_indices[1:])
    left_reduced = PdT(second_base, PdVars(left, *rest))
    right_reduced = PdT(second_base, PdVars(right, *rest))
    return first_pos, second_pos, first_factor, left_reduced, right_reduced, left, right, derivative_index


def _is_implicit_index(value):
    return isinstance(value, Index) and not value.head.explicit


def _contains_index(expr, target):
    return any(
        isinstance(node, Index) and node.head_name == target.head_name and node.label == target.label
        for node in sp.preorder_traversal(sp.sympify(expr))
    )


def _swap_index_labels(expr, left, right):
    replacements = {}
    for node in sp.preorder_traversal(sp.sympify(expr)):
        if not isinstance(node, Index) or node.head_name != left.head_name:
            continue
        if node.label == left.label:
            replacements[node] = node.with_label(right.label)
        elif node.label == right.label:
            replacements[node] = node.with_label(left.label)
    return sp.sympify(expr).xreplace(replacements)


def _ibp_first_derivative_term(term):
    match = _find_derivative_factor(term, lambda _base, indices: len(indices) >= 1)
    if match is None:
        return term
    coeff, base, indices = match
    first, rest = indices[0], indices[1:]
    reduced = PdT(base, PdVars(*rest)) if rest else base
    power_rule = _ibp_power_rule(coeff, reduced, first)
    if power_rule is not None:
        return power_rule
    return -reduced * Pd(coeff, first) + PdHold(coeff * reduced, first)


def _ibp_power_rule(coeff, reduced, derivative_index):
    factored = _extract_power_factor(coeff, reduced)
    if factored is None:
        return None
    remaining_coeff, exponent = factored
    denominator = exponent + 1
    if denominator == 0:
        return None
    integrated_power = reduced**denominator
    return -integrated_power * Pd(remaining_coeff, derivative_index) / denominator + PdHold(
        remaining_coeff * integrated_power / denominator, derivative_index
    )


def _extract_power_factor(expr, base):
    factors = list(expr.args) if isinstance(expr, sp.Mul) else [expr]
    for pos, factor in enumerate(factors):
        if factor == base:
            exponent = sp.Integer(1)
        elif isinstance(factor, sp.Pow) and factor.base == base:
            exponent = factor.exp
        else:
            continue
        remaining = sp.Mul(*(factors[:pos] + factors[pos + 1 :])) if len(factors) > 1 else sp.Integer(1)
        return remaining, sp.sympify(exponent)
    return None


def _find_derivative_factor(term, predicate):
    factors = list(term.args) if isinstance(term, sp.Mul) else [term]
    for pos, factor in enumerate(factors):
        if not is_pdt(factor):
            continue
        base, indices = pdt_parts(factor)
        if not predicate(base, indices):
            continue
        coeff_factors = factors[:pos] + factors[pos + 1 :]
        coeff = sp.Mul(*coeff_factors) if coeff_factors else sp.Integer(1)
        return coeff, base, indices
    return None
