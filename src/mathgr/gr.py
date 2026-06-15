from __future__ import annotations

from itertools import count

import sympy as sp

from .tensor import DE, DN, UP, DeclareSym, Index, LatinIdx, Pd, Symmetric, tensor, register_metric
from .util_private import apply2term


g = tensor("g")
_R_HEAD = tensor("R")
_G_HEAD = tensor("G")
_RSIMP_HEAD = tensor("Rsimp")
_AFFINE_HEAD = tensor("Affine")
_COVD_HEAD = tensor("CovD")
_X_HEAD = tensor("X")
_DSQUARE_HEAD = tensor("Dsquare")
_T_HEAD = tensor("T")
_K_HEAD = tensor("K")
_KK_HEAD = tensor("KK")
_RADM_HEAD = tensor("RADM")
Metric = None
IdxOfMetric = (UP, DN)
V = tensor("V")
LapseN = sp.Symbol("LapseN")
ShiftN = tensor("ShiftN")
_slot_counter = count(1)


class _MetricSlot(sp.Expr):
    is_commutative = True

    def __new__(cls, kind: str, label, occurrence=None):
        if occurrence is None:
            occurrence = next(_slot_counter)
        return sp.Expr.__new__(cls, sp.Symbol(kind), sp.sympify(label), sp.Integer(occurrence))

    @property
    def kind(self) -> str:
        return str(self.args[0])

    @property
    def label(self):
        label = self.args[1]
        if label.is_Integer:
            return int(label)
        return str(label)

    def _sympystr(self, printer):
        return f"{self.kind}({printer.doprint(self.args[1])})"


def UG(label):
    return _MetricSlot("UG", label)


def DG(label):
    return _MetricSlot("DG", label)


class _LowerRiemann(sp.Expr):
    is_commutative = True

    def __new__(cls, *indices):
        indices = tuple(sp.sympify(index) for index in indices)
        if len(indices) != 4:
            raise TypeError("Lower Riemann tensor expects four indices.")
        canonical = _canonical_lower_riemann_indices(indices)
        if canonical is None:
            return sp.Integer(0)
        sign, canonical_indices = canonical
        if canonical_indices != indices:
            result = cls(*canonical_indices)
            return result if sign == 1 else -result
        bianchi_terms = _canonical_lower_riemann_bianchi_terms(indices)
        dependent = max((term_indices for _term_sign, term_indices in bianchi_terms), key=_riemann_indices_sort_key)
        if indices == dependent:
            dependent_coeff = sum(term_sign for term_sign, term_indices in bianchi_terms if term_indices == dependent)
            other_terms = [
                term_sign * cls(*term_indices)
                for term_sign, term_indices in bianchi_terms
                if term_indices != dependent
            ]
            return -sp.Add(*other_terms) / dependent_coeff
        return sp.Expr.__new__(cls, *indices)

    def _sympystr(self, printer):
        args = ", ".join(printer.doprint(arg) for arg in self.args)
        return f"RiemannLower({args})"


class _CovDLowerRiemannAtom(sp.Expr):
    is_commutative = True

    def __new__(cls, *indices):
        indices = tuple(sp.sympify(index) for index in indices)
        if len(indices) != 5:
            raise TypeError("Covariant derivative of lower Riemann expects five indices.")
        canonical = _canonical_lower_riemann_indices(indices[:4])
        if canonical is None:
            return sp.Integer(0)
        sign, canonical_riemann = canonical
        canonical_indices = canonical_riemann + (indices[4],)
        if canonical_indices != indices:
            result = cls(*canonical_indices)
            return result if sign == 1 else -result
        return sp.Expr.__new__(cls, *indices)

    def _sympystr(self, printer):
        args = ", ".join(printer.doprint(arg) for arg in self.args)
        return f"CovDLowerRiemann({args})"


def UseMetric(metric, indices=(UP, DN), *, SetAsDefault=True):
    up, down = indices
    DeclareSym(metric, (up, up), Symmetric((1, 2)))
    DeclareSym(metric, (down, down), Symmetric((1, 2)))
    register_metric(metric, indices)
    if SetAsDefault:
        global Metric, IdxOfMetric
        Metric = metric
        IdxOfMetric = tuple(indices)
    return None


UseMetric(g)


def WithMetric(metric, indices=(UP, DN), expr=None):
    if expr is None:
        if not callable(indices):
            raise TypeError("WithMetric requires a callable expression when index types are omitted.")
        callback = indices
        indices = (UP, DN)
    else:
        callback = expr if callable(expr) else (lambda: expr)

    UseMetric(metric, indices, SetAsDefault=False)
    global Metric, IdxOfMetric
    previous_metric = Metric
    previous_indices = IdxOfMetric
    try:
        Metric = metric
        IdxOfMetric = tuple(indices)
        return callback()
    finally:
        Metric = previous_metric
        IdxOfMetric = previous_indices


def MetricContract(expr):
    return apply2term(_metric_contract_term, expr)


def _metric_contract_term(term):
    term = sp.sympify(term)
    slots = list(_iter_metric_slots(term))
    if not slots:
        return term
    up, down = IdxOfMetric
    free_labels = {index.label for index in _iter_indices(term) if isinstance(index.label, str)}
    concrete_labels = [label for label in LatinIdx if label not in free_labels]
    slot_to_index = {}
    groups = {}
    for slot, label in zip(slots, concrete_labels, strict=False):
        slot_to_index[slot] = up(label) if slot.kind == "UG" else down(label)
        groups.setdefault(slot.label, []).append(slot)

    metric_factors = []
    for group in groups.values():
        metric_factors.append(
            Metric(*(_dual_metric_index_for_slot(slot, slot_to_index[slot], up, down) for slot in group))
        )

    return term.xreplace(slot_to_index) * sp.Mul(*metric_factors)


def _dual_metric_index_for_slot(slot, concrete_index, up, down):
    return down(concrete_index.label) if slot.kind == "UG" else up(concrete_index.label)


def _iter_metric_slots(expr):
    if isinstance(expr, _MetricSlot):
        yield expr
        return
    for arg in getattr(expr, "args", ()):
        yield from _iter_metric_slots(arg)


def _iter_indices(expr, *, include_metric_slots=False):
    if include_metric_slots and isinstance(expr, _MetricSlot):
        yield expr
        return
    if isinstance(expr, Index):
        yield expr
        return
    for arg in getattr(expr, "args", ()):
        yield from _iter_indices(arg, include_metric_slots=include_metric_slots)


def Affine(*indices):
    if len(indices) != 3:
        return _AFFINE_HEAD(*(sp.sympify(index) for index in indices))
    i, m, n = (sp.sympify(index) for index in indices)
    up, down = IdxOfMetric
    if not (_is_metric_upper_like(i, up) and _is_metric_lower_like(m, down) and _is_metric_lower_like(n, down)):
        return _AFFINE_HEAD(i, m, n)
    free_labels = {idx.label for idx in (i, m, n) if isinstance(idx, Index)}
    dummy = next(label for label in LatinIdx if label not in free_labels)
    return _affine_with_dummy(i, m, n, dummy, up, down)


def _affine_with_dummy(i, m, n, dummy, up, down):
    q_up = up(dummy)
    q_down = down(dummy)
    return (
        Metric(i, q_up)
        * (Pd(Metric(m, q_down), n) + Pd(Metric(n, q_down), m) - Pd(Metric(m, n), q_down))
        / 2
    )


def CovD(*args):
    if len(args) != 2:
        return _COVD_HEAD(*(sp.sympify(arg) for arg in args))
    expr, index = (sp.sympify(arg) for arg in args)
    up, down = IdxOfMetric
    if not _is_covd_index(index, up, down):
        return _COVD_HEAD(expr, index)
    if isinstance(expr, sp.Add):
        return sp.Add(*(CovD(arg, index) for arg in expr.args))
    lower_riemann_term = _extract_lower_riemann_term(expr)
    if lower_riemann_term is not None and _is_index_of_type(index, down):
        coefficient, riemann = lower_riemann_term
        return coefficient * _covd_lower_riemann(*riemann.args, index)
    if isinstance(index, _MetricSlot):
        if index.kind == "UG":
            dummy_slot = UG(index.label)
            return Metric(index, dummy_slot) * CovD(expr, DG(index.label))
        if index.kind == "DG":
            return _covd_lower(expr, index, up, down)
        return _COVD_HEAD(expr, index)
    if index.head_name == up.name:
        dummy = _first_available_label(expr, index.label)
        return Metric(index, up(dummy)) * CovD(expr, down(dummy))
    return _covd_lower(expr, index, up, down)


def _is_covd_index(index, up, down):
    return _is_metric_upper_like(index, up) or _is_metric_lower_like(index, down)


def _covd_lower(expr, index, up, down):
    expr = _rename_conflicting_dummy_indices(expr, {_index_label(index)})
    free_upper, free_lower = _free_metric_indices(expr, up, down)
    replacement_label = _first_available_label(expr, _index_label(index))
    affine_dummy = _first_available_label(expr, index.label, replacement_label)
    replacement_up = up(replacement_label)
    replacement_down = down(replacement_label)

    result = Pd(expr, index)
    for free_index in free_upper:
        result += (
            _affine_with_dummy(free_index, index, replacement_down, affine_dummy, up, down)
            * expr.xreplace({free_index: replacement_up})
        )
    for free_index in free_lower:
        result += (
            _lower_covariant_connection(free_index, index, replacement_up, affine_dummy, up, down)
            * expr.xreplace({free_index: replacement_down})
        )
    return result


def _rename_conflicting_dummy_indices(expr, forbidden_labels):
    forbidden_labels = set(forbidden_labels)
    indices = list(_iter_indices(expr, include_metric_slots=True))
    counts = {}
    for index in indices:
        counts[_index_label(index)] = counts.get(_index_label(index), 0) + 1

    used_labels = set(counts) | forbidden_labels
    replacements = {}
    for label, count in counts.items():
        if count < 2 or label not in forbidden_labels:
            continue
        replacement_label = next(candidate for candidate in LatinIdx if candidate not in used_labels)
        used_labels.add(replacement_label)
        for index in indices:
            if _index_label(index) == label and isinstance(index, Index):
                replacements[index] = index.with_label(replacement_label)
    if not replacements:
        return expr
    return expr.xreplace(replacements)


def _lower_covariant_connection(free_index, derivative_index, replacement_up, dummy, up, down):
    dummy_up = up(dummy)
    dummy_down = down(dummy)
    return (
        Metric(replacement_up, dummy_up) * Pd(Metric(free_index, derivative_index), dummy_down) / 2
        - Metric(replacement_up, dummy_up) * Pd(Metric(free_index, dummy_down), derivative_index) / 2
        - Metric(replacement_up, dummy_up) * Pd(Metric(derivative_index, dummy_down), free_index) / 2
    )


def _free_metric_indices(expr, up, down):
    indices = [
        index
        for index in _iter_indices(expr, include_metric_slots=True)
        if _is_metric_upper_like(index, up) or _is_metric_lower_like(index, down)
    ]
    counts = {}
    for index in indices:
        counts[_index_label(index)] = counts.get(_index_label(index), 0) + 1
    free_upper = []
    free_lower = []
    for index in indices:
        if counts[_index_label(index)] != 1:
            continue
        target = free_upper if _is_metric_upper_like(index, up) else free_lower
        if index not in target:
            target.append(index)
    return free_upper, free_lower


def _first_available_label(expr, *extra_labels):
    used = {_index_label(index) for index in _iter_indices(expr, include_metric_slots=True)}
    used.update(extra_labels)
    return next(label for label in LatinIdx if label not in used)


def _claim_available_label(used_labels):
    label = next(label for label in LatinIdx if label not in used_labels)
    used_labels.add(label)
    return label


def _index_label(index):
    return index.label


def _is_metric_upper_like(index, up):
    return (isinstance(index, Index) and index.head_name == up.name) or (
        isinstance(index, _MetricSlot) and index.kind == "UG"
    )


def _is_metric_lower_like(index, down):
    return (isinstance(index, Index) and index.head_name == down.name) or (
        isinstance(index, _MetricSlot) and index.kind == "DG"
    )


def R(*indices):
    if len(indices) == 0:
        return Rsimp()
    indices = tuple(sp.sympify(index) for index in indices)
    up, down = IdxOfMetric
    if len(indices) == 4:
        if all(isinstance(index, _MetricSlot) for index in indices):
            return _metric_slot_tensor(R, indices, up, down)
        first, second, third, fourth = indices
        if all(_is_index_of_type(index, down) for index in indices):
            canonical = _canonical_lower_riemann_indices(indices)
            if canonical is None:
                return sp.Integer(0)
            sign, canonical_indices = canonical
            if canonical_indices != indices:
                return sign * R(*canonical_indices)
            return _LowerRiemann(*indices)
        if _is_index_of_type(first, up) and all(_is_index_of_type(index, down) for index in indices[1:]):
            return _upper_riemann_component(first, second, third, fourth, up, down)
        if any(_is_index_of_type(index, up) for index in indices[1:]) and (
            _is_index_of_type(first, up) or _is_index_of_type(first, down)
        ):
            context = sp.Mul(*indices, evaluate=False)
            used_labels = []
            metric_factors = []
            lowered = [first, second, third, fourth]
            for position, index in enumerate(indices[1:], start=1):
                if _is_index_of_type(index, down):
                    continue
                if not _is_index_of_type(index, up):
                    break
                dummy = _first_available_label(context, *used_labels)
                used_labels.append(dummy)
                metric_factors.append(Metric(index, up(dummy)))
                lowered[position] = down(dummy)
            else:
                return sp.Mul(*metric_factors) * R(*lowered)
    if len(indices) == 2 and all(isinstance(index, _MetricSlot) for index in indices):
        return _metric_slot_tensor(R, indices, up, down)
    if len(indices) == 2 and all(_is_index_of_type(index, down) for index in indices):
        first, second = indices
        dummy = _first_available_label(sp.Mul(*indices, evaluate=False))
        return R(up(dummy), first, down(dummy), second)
    if len(indices) == 2:
        first, second = indices
        if _is_index_of_type(first, up) and _is_index_of_type(second, down):
            dummy = _first_available_label(sp.Mul(first, second, evaluate=False))
            return Metric(first, up(dummy)) * R(down(dummy), second)
        if _is_index_of_type(first, down) and _is_index_of_type(second, up):
            dummy = _first_available_label(sp.Mul(first, second, evaluate=False))
            return Metric(second, up(dummy)) * R(first, down(dummy))
        if _is_index_of_type(first, up) and _is_index_of_type(second, up):
            dummy_left = _first_available_label(sp.Mul(first, second, evaluate=False))
            dummy_right = _first_available_label(sp.Mul(first, second, evaluate=False), dummy_left)
            return Metric(first, up(dummy_left)) * Metric(second, up(dummy_right)) * R(
                down(dummy_left), down(dummy_right)
            )
    return _R_HEAD(*indices)


def _extract_lower_riemann_term(expr):
    if isinstance(expr, _LowerRiemann):
        return sp.Integer(1), expr
    if isinstance(expr, sp.Mul):
        riemann_factors = [factor for factor in expr.args if isinstance(factor, _LowerRiemann)]
        if len(riemann_factors) != 1:
            return None
        riemann = riemann_factors[0]
        rest = sp.Mul(*(factor for factor in expr.args if factor is not riemann), evaluate=False)
        if any(True for _ in _iter_indices(rest, include_metric_slots=True)):
            return None
        return rest, riemann
    return None


def _covd_lower_riemann(first, second, third, fourth, derivative):
    terms = [
        _CovDLowerRiemannAtom(first, second, third, fourth, derivative),
        _CovDLowerRiemannAtom(first, second, fourth, derivative, third),
        _CovDLowerRiemannAtom(first, second, derivative, third, fourth),
    ]
    current = terms[0]
    dependent = max(terms, key=sp.default_sort_key)
    if current == dependent:
        return -sp.Add(*(term for term in terms[1:] if term != dependent))
    return current


def _upper_riemann_component(first, second, third, fourth, up, down, *extra_avoid_labels):
    used_labels = {_index_label(index) for index in (first, second, third, fourth)}
    used_labels.update(extra_avoid_labels)
    avoid_exprs = tuple(down(label) for label in extra_avoid_labels)
    first_affine_dummy = _claim_available_label(used_labels)
    second_affine_dummy = _claim_available_label(used_labels)
    contraction_dummy = _claim_available_label(used_labels)
    third_affine_dummy = _claim_available_label(used_labels)
    fourth_affine_dummy = _claim_available_label(used_labels)
    fifth_affine_dummy = _claim_available_label(used_labels)
    sixth_affine_dummy = _claim_available_label(used_labels)
    return (
        Pd(_affine_with_dummy(first, second, fourth, first_affine_dummy, up, down), third, avoid=avoid_exprs)
        - Pd(_affine_with_dummy(first, second, third, second_affine_dummy, up, down), fourth, avoid=avoid_exprs)
        + _affine_with_dummy(up(contraction_dummy), second, fourth, third_affine_dummy, up, down)
        * _affine_with_dummy(first, down(contraction_dummy), third, fourth_affine_dummy, up, down)
        - _affine_with_dummy(up(contraction_dummy), second, third, fifth_affine_dummy, up, down)
        * _affine_with_dummy(first, down(contraction_dummy), fourth, sixth_affine_dummy, up, down)
    )


def _riemann_pair_sort_key(index):
    return sp.default_sort_key(index.label)


def _riemann_pair_key(pair):
    return tuple(_riemann_pair_sort_key(index) for index in pair)


def _riemann_indices_sort_key(indices):
    return sp.default_sort_key(sp.Tuple(*indices))


def _canonical_lower_riemann_bianchi_terms(indices):
    first, second, third, fourth = indices
    raw_terms = (
        (first, second, third, fourth),
        (first, third, fourth, second),
        (first, fourth, second, third),
    )
    terms = []
    for raw_term in raw_terms:
        canonical = _canonical_lower_riemann_indices(raw_term)
        if canonical is None:
            continue
        sign, canonical_indices = canonical
        terms.append((sign, canonical_indices))
    return tuple(terms)


def _canonical_lower_riemann_indices(indices):
    first, second, third, fourth = indices
    if first == second or third == fourth:
        return None
    sign = sp.Integer(1)
    left_pair = (first, second)
    right_pair = (third, fourth)
    if _riemann_pair_sort_key(left_pair[1]) < _riemann_pair_sort_key(left_pair[0]):
        left_pair = (left_pair[1], left_pair[0])
        sign = -sign
    if _riemann_pair_sort_key(right_pair[1]) < _riemann_pair_sort_key(right_pair[0]):
        right_pair = (right_pair[1], right_pair[0])
        sign = -sign
    if _riemann_pair_key(right_pair) < _riemann_pair_key(left_pair):
        left_pair, right_pair = right_pair, left_pair
    return sign, left_pair + right_pair


def RicciScalar():
    return MetricContract(R(DG(1), DG(1)))


def _is_index_of_type(value, index_type):
    return isinstance(value, Index) and value.head_name == index_type.name


def G(*indices):
    indices = tuple(sp.sympify(index) for index in indices)
    if len(indices) != 2:
        return _G_HEAD(*indices)
    first, second = indices
    up, down = IdxOfMetric
    if all(isinstance(index, _MetricSlot) for index in (first, second)):
        return _metric_slot_tensor(G, (first, second), up, down)
    if _is_index_of_type(first, down) and _is_index_of_type(second, down):
        return R(first, second) - Metric(first, second) * R() / 2
    if _is_index_of_type(first, up) and _is_index_of_type(second, down):
        dummy = _first_available_label(sp.Mul(first, second, evaluate=False))
        return Metric(first, up(dummy)) * G(down(dummy), second)
    if _is_index_of_type(first, down) and _is_index_of_type(second, up):
        dummy = _first_available_label(sp.Mul(first, second, evaluate=False))
        return Metric(second, up(dummy)) * G(first, down(dummy))
    if _is_index_of_type(first, up) and _is_index_of_type(second, up):
        dummy_left = _first_available_label(sp.Mul(first, second, evaluate=False))
        dummy_right = _first_available_label(sp.Mul(first, second, evaluate=False), dummy_left)
        return Metric(first, up(dummy_left)) * Metric(second, up(dummy_right)) * G(down(dummy_left), down(dummy_right))
    return _G_HEAD(first, second)


def _metric_slot_tensor(callback, slots, up, down):
    slots = tuple(slots)
    labels = []
    expr = sp.Mul(*slots, evaluate=False)
    for _ in slots:
        labels.append(_first_available_label(expr, *labels))
    concrete = tuple(
        up(label) if slot.kind == "UG" else down(label)
        for slot, label in zip(slots, labels, strict=True)
    )
    result = callback(*concrete)

    groups = {}
    for slot, concrete_index in zip(slots, concrete, strict=True):
        groups.setdefault(slot.label, []).append((slot, concrete_index))

    metric_factors = []
    for group in groups.values():
        if len(group) < 2:
            continue
        metric_factors.append(
            Metric(
                *(
                    _dual_metric_index_for_slot(slot, concrete_index, up, down)
                    for slot, concrete_index in group
                )
            )
        )
    return sp.Mul(*metric_factors) * result


def X(*fields):
    if len(fields) != 1:
        return _X_HEAD(*(sp.sympify(field) for field in fields))
    field = sp.sympify(fields[0])
    up, down = IdxOfMetric
    left, right = _first_two_available_labels(field)
    return -Metric(up(left), up(right)) * Pd(field, down(left)) * Pd(field, down(right)) / 2


def Dsquare(*fields):
    if len(fields) != 1:
        return _DSQUARE_HEAD(*(sp.sympify(field) for field in fields))
    field = sp.sympify(fields[0])
    return MetricContract(CovD(CovD(field, DG(1)), DG(1)))


def T(field):
    def _stress(*indices):
        indices = tuple(sp.sympify(index) for index in indices)
        if len(indices) != 2:
            return _T_HEAD(field, *indices)
        first, second = indices
        _up, down = IdxOfMetric
        if not (_is_index_of_type(first, down) and _is_index_of_type(second, down)):
            return _T_HEAD(field, first, second)
        return Metric(first, second) * (X(field) - V(field)) + Pd(field, first) * Pd(field, second)

    return _stress


def K(*indices):
    if len(indices) == 0:
        return MetricContract(K(DG(1), DG(1)))
    if len(indices) != 2:
        return _K_HEAD(*(sp.sympify(index) for index in indices))
    first, second = (sp.sympify(index) for index in indices)
    _up, down = IdxOfMetric
    if not (_is_index_of_type(first, down) and _is_index_of_type(second, down)):
        return _K_HEAD(first, second)
    return (Pd(Metric(first, second), DE(0)) - CovD(ShiftN(first), second) - CovD(ShiftN(second), first)) / (
        2 * LapseN
    )


def KK(*indices):
    if indices:
        return _KK_HEAD(*(sp.sympify(index) for index in indices))
    return MetricContract(K(DG(1), DG(2)) * K(DG(1), DG(2)))


def RADM(*indices):
    if indices:
        return _RADM_HEAD(*(sp.sympify(index) for index in indices))
    return R() - K() * K() + KK()


def _first_two_available_labels(expr):
    first = _first_available_label(sp.sympify(expr))
    second = _first_available_label(sp.sympify(expr), first)
    return first, second


def _available_labels(expr, count, *extra_labels):
    labels = []
    for _ in range(count):
        labels.append(_first_available_label(expr, *extra_labels, *labels))
    return labels


def _lower_rsimp_component(first, second, *extra_labels):
    up, down = IdxOfMetric
    a, b, c, d_label = _available_labels(sp.Mul(first, second, evaluate=False), 4, *extra_labels)
    return (
        -Metric(up(a), up(b))
        * Metric(up(c), up(d_label))
        * Pd(Metric(first, second), down(d_label))
        * Pd(Metric(down(a), down(b)), down(c))
        / 4
        + Metric(up(a), up(b))
        * Metric(up(c), up(d_label))
        * Pd(Metric(first, second), down(d_label))
        * Pd(Metric(down(a), down(c)), down(b))
        / 2
        - Metric(up(a), up(b))
        * Metric(up(c), up(d_label))
        * Pd(Metric(down(a), down(c)), down(d_label))
        * Pd(Metric(down(b), first), second)
        / 2
        + Metric(up(a), up(b))
        * Metric(up(c), up(d_label))
        * Pd(Metric(down(a), second), down(c))
        * Pd(Metric(down(b), first), down(d_label))
        / 2
        - Metric(up(a), up(b))
        * Metric(up(c), up(d_label))
        * Pd(Metric(down(a), down(c)), down(d_label))
        * Pd(Metric(down(b), second), first)
        / 2
        + Metric(up(a), up(b))
        * Metric(up(c), up(d_label))
        * Pd(Metric(down(a), down(c)), second)
        * Pd(Metric(down(b), down(d_label)), first)
        / 4
        + Metric(up(a), up(b))
        * Metric(up(c), up(d_label))
        * Pd(Metric(down(a), down(b)), down(d_label))
        * Pd(Metric(down(c), first), second)
        / 4
        - Metric(up(a), up(b))
        * Metric(up(c), up(d_label))
        * Pd(Metric(down(a), second), down(d_label))
        * Pd(Metric(down(c), first), down(b))
        / 2
        + Metric(up(a), up(b))
        * Metric(up(c), up(d_label))
        * Pd(Metric(down(a), down(b)), down(d_label))
        * Pd(Metric(down(c), second), first)
        / 4
        - Metric(up(a), up(b)) * Pd(Pd(Metric(first, second), down(a)), down(b)) / 2
        + Metric(up(a), up(b)) * Pd(Pd(Metric(down(a), first), second), down(b)) / 2
        + Metric(up(a), up(b)) * Pd(Pd(Metric(down(a), second), first), down(b)) / 2
        - Metric(up(a), up(b)) * Pd(Pd(Metric(down(a), down(b)), first), second) / 2
    )


def Rsimp(*indices):
    up, down = IdxOfMetric
    if len(indices) == 2:
        first, second = (sp.sympify(index) for index in indices)
        if all(isinstance(index, _MetricSlot) for index in (first, second)):
            return _metric_slot_tensor(Rsimp, (first, second), up, down)
        if _is_index_of_type(first, down) and _is_index_of_type(second, down):
            return _lower_rsimp_component(first, second)
        context = sp.Mul(first, second, evaluate=False)
        if _is_index_of_type(first, up) and _is_index_of_type(second, down):
            dummy = _first_available_label(context)
            return Metric(first, up(dummy)) * _lower_rsimp_component(down(dummy), second, _index_label(first))
        if _is_index_of_type(first, down) and _is_index_of_type(second, up):
            dummy = _first_available_label(context)
            return Metric(second, up(dummy)) * _lower_rsimp_component(first, down(dummy), _index_label(second))
        if _is_index_of_type(first, up) and _is_index_of_type(second, up):
            dummy_left = _first_available_label(context)
            dummy_right = _first_available_label(context, dummy_left)
            return Metric(first, up(dummy_left)) * Metric(second, up(dummy_right)) * _lower_rsimp_component(
                down(dummy_left), down(dummy_right), _index_label(first), _index_label(second)
            )
    if len(indices) != 0:
        return _RSIMP_HEAD(*(sp.sympify(index) for index in indices))
    a, b, c, d_label, e, f = [label for label in "abcdef"]
    return (
        3
        * Metric(up(a), up(b))
        * Metric(up(c), up(d_label))
        * Metric(up(e), up(f))
        * Pd(Metric(down(a), down(c)), down(e))
        * Pd(Metric(down(b), down(d_label)), down(f))
        / 4
        - Metric(up(a), up(b))
        * Metric(up(c), up(d_label))
        * Metric(up(e), up(f))
        * Pd(Metric(down(a), down(c)), down(f))
        * Pd(Metric(down(b), down(e)), down(d_label))
        / 2
        - Metric(up(a), up(b))
        * Metric(up(c), up(d_label))
        * Metric(up(e), up(f))
        * Pd(Metric(down(a), down(c)), down(d_label))
        * Pd(Metric(down(b), down(e)), down(f))
        - Metric(up(a), up(b))
        * Metric(up(c), up(d_label))
        * Metric(up(e), up(f))
        * Pd(Metric(down(a), down(b)), down(e))
        * Pd(Metric(down(c), down(d_label)), down(f))
        / 4
        + Metric(up(a), up(b))
        * Metric(up(c), up(d_label))
        * Metric(up(e), up(f))
        * Pd(Metric(down(a), down(b)), down(d_label))
        * Pd(Metric(down(c), down(e)), down(f))
        - Metric(up(a), up(b))
        * Metric(up(c), up(d_label))
        * Pd(Pd(Metric(down(a), down(b)), down(c)), down(d_label))
        + Metric(up(a), up(b))
        * Metric(up(c), up(d_label))
        * Pd(Pd(Metric(down(a), down(c)), down(b)), down(d_label))
    )
