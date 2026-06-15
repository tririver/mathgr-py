from __future__ import annotations

import sympy as sp

from . import gr as _gr
from .decomp import DTot, UTot, Decomp0i
from .gr import DG, MetricContract, WithMetric
from .tensor import DE, DN, UE, UP, Dta, Index, LatinIdx, Pd, PdT, PdVars, Simp as _TensorSimp, _CONSTANTS, is_pdt, pdt_parts, tensor
from .tensor import tensor_args, tensor_head_name
from .util import Eps


k = sp.Symbol("k")
_CONSTANTS.add(k)
a = sp.Symbol("a")
H = sp.Symbol("H")
alpha = sp.Symbol("alpha")
beta = sp.Symbol("beta")
zeta = sp.Symbol("zeta")
epsilon = sp.Symbol("epsilon")
eta = sp.Symbol("eta")
eta2 = sp.Symbol("eta2")
eta3 = sp.Symbol("eta3")
Mp = sp.Symbol("Mp")
b = tensor("b")
g = tensor("g")
_h_raw = tensor("h")
_SHIFT_HEAD = tensor("ShiftN")
_K_HEAD = tensor("K")
_KK_HEAD = tensor("KK")
_RADM_HEAD = tensor("RADM")
LapseN = 1 + Eps * alpha
Sqrtg = LapseN * sp.exp(3 * Eps * zeta) * a**3
_DEFAULT_DIM = sp.Symbol("DefaultDim")


def ShiftN(*indices):
    if len(indices) != 1:
        return _SHIFT_HEAD(*(sp.sympify(index) for index in indices))
    index = sp.sympify(indices[0])
    if not _is_frw_shift_index(index):
        return _SHIFT_HEAD(index)
    return Eps * Pd(beta, index) + Eps * b(index)


def K(*indices):
    if _unsupported_k_signature(indices):
        return _K_HEAD(*(sp.sympify(index) for index in indices))
    return _with_frw_metric(lambda: _K(*indices))


def _K(*indices):
    if len(indices) == 0:
        return MetricContract(_K(DG(1), DG(1)))
    if len(indices) != 2:
        return _K_HEAD(*(sp.sympify(index) for index in indices))
    first, second = (sp.sympify(index) for index in indices)
    if not (_is_frw_k_index(first) and _is_frw_k_index(second)):
        return _K_HEAD(first, second)
    return (
        Pd(_gr.Metric(first, second), DE(0))
        - _gr.CovD(ShiftN(first), second)
        - _gr.CovD(ShiftN(second), first)
    ) / (2 * LapseN)


def KK(*indices):
    if indices:
        return _KK_HEAD(*(sp.sympify(index) for index in indices))
    return _with_frw_metric(_KK)


def _KK():
    return MetricContract(_K(DG(1), DG(2)) * _K(DG(1), DG(2)))


def RADM(*indices):
    if indices:
        return _RADM_HEAD(*(sp.sympify(index) for index in indices))
    return _with_frw_metric(lambda: _gr.R() - _K() * _K() + _KK())


def _with_frw_metric(callback):
    return _replace_spatial_metric_components(_TensorSimp(WithMetric(_h_raw, (UP, DN), callback)))


def _unsupported_k_signature(indices):
    if len(indices) == 0:
        return False
    if len(indices) != 2:
        return True
    return not all(_is_frw_k_index(sp.sympify(index)) for index in indices)


def _is_frw_k_index(index):
    return _is_index(index, DN) or isinstance(index, _gr._MetricSlot)


def _is_frw_shift_index(index):
    return _is_index(index, DN) or (isinstance(index, _gr._MetricSlot) and index.kind == "DG")


def Simp(expr, **options):
    hooks = tuple(options.pop("hooks", ())) + (_background_simp_hook,)
    return _TensorSimp(expr, hooks=hooks, **options)


def h(first, second):
    first = sp.sympify(first)
    second = sp.sympify(second)
    if isinstance(first, Index) and isinstance(second, Index):
        if first.head_name == DN.name and second.head_name == DN.name:
            return a**2 * sp.exp(2 * Eps * zeta) * Dta(first, second)
        if first.head_name == UP.name and second.head_name == UP.name:
            return sp.exp(-2 * Eps * zeta) * Dta(DN(first.label), DN(second.label)) / a**2
        if first.head.dual_name == second.head_name:
            return Dta(first, second)
    return _h_raw(first, second)


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


def _replace_spatial_metric_components(expr):
    expr = sp.sympify(expr)
    if tensor_head_name(expr) == "h":
        replacement = h(*tensor_args(expr))
        if replacement != expr:
            return _replace_spatial_metric_components(replacement)
    if not getattr(expr, "args", ()):
        return expr
    new_args = tuple(_replace_spatial_metric_components(arg) for arg in expr.args)
    if new_args == expr.args:
        return expr
    return expr.func(*new_args)


def _background_simp_hook(expr):
    expr = sp.sympify(expr)
    replacement = _background_replacement(expr)
    if replacement is not None:
        return replacement
    if not getattr(expr, "args", ()):
        return expr.xreplace({_DEFAULT_DIM: sp.Integer(3)})
    new_args = tuple(_background_simp_hook(arg) for arg in expr.args)
    rebuilt = expr if new_args == expr.args else expr.func(*new_args)
    return rebuilt.xreplace({_DEFAULT_DIM: sp.Integer(3)})


def _background_replacement(expr):
    delta_trace = _frw_delta_trace_replacement(expr)
    if delta_trace is not None:
        return delta_trace
    if expr == _DEFAULT_DIM:
        return sp.Integer(3)
    if not is_pdt(expr):
        return None
    base, derivative_indices = pdt_parts(expr)
    if base == Mp:
        return sp.Integer(0)
    if _is_transverse_shift_derivative(base, derivative_indices):
        return sp.Integer(0)
    if base in {a, H, epsilon, eta} and any(_is_index(index, DN) for index in derivative_indices):
        return sp.Integer(0)
    if tuple(derivative_indices) == (DE(0),):
        if base == a:
            return a * H
        if base == H:
            return -epsilon * H**2
        if base == epsilon:
            return H * epsilon * eta
        if base == eta:
            return H * eta2 * eta
        if base == eta2:
            return H * eta3 * eta2
    if tuple(derivative_indices) == (DE(0), DE(0)):
        if base == a:
            return a * H**2 - a * H**2 * epsilon
        if base == H:
            return 2 * H**3 * epsilon**2 - H**3 * epsilon * eta
    if tuple(derivative_indices) == (DE(0), DE(0), DE(0)) and base == H:
        return (
            -6 * H**4 * epsilon**3
            + 7 * H**4 * epsilon**2 * eta
            - H**4 * epsilon * eta**2
            - H**4 * epsilon * eta * eta2
        )
    return None


def _frw_delta_trace_replacement(expr):
    if not isinstance(expr, sp.Mul):
        return None
    factors = list(expr.args)
    for pos, factor in enumerate(factors):
        delta_power = _spatial_delta_power_factor(factor)
        if delta_power is None:
            continue
        delta, exponent = delta_power
        left, right = delta.args
        rest = sp.Mul(*(factors[:pos] + factors[pos + 1 :]))
        collapsed = _replace_index_label(rest, right.label, left.label)
        if exponent == 3:
            return sp.Integer(3) * collapsed
        if exponent == 2 and _has_transverse_shift_divergence(collapsed):
            return sp.Integer(0)
    return None


def _spatial_delta_power_factor(factor):
    if isinstance(factor, sp.Pow) and factor.exp in {2, 3} and getattr(factor.base, "func", None).__name__ == "_Dta":
        delta = factor.base
        left, right = delta.args
        if _is_index(left, DN) and _is_index(right, DN):
            return delta, int(factor.exp)
    return None


def _replace_index_label(expr, old_label, new_label):
    replacements = {
        index: index.with_label(new_label)
        for index in set(_iter_indices(expr))
        if index.label == old_label
    }
    return sp.sympify(expr).xreplace(replacements)


def _has_transverse_shift_divergence(expr):
    for node in sp.preorder_traversal(expr):
        if not is_pdt(node):
            continue
        base, derivative_indices = pdt_parts(node)
        if _is_transverse_shift_derivative(base, derivative_indices):
            return True
    return False


def _metric_component_replacement(first, second):
    if _is_index(first, DN) and _is_index(second, DN):
        return h(first, second)
    if first == DE(0) and second == DE(0):
        left, right = _metric_dummy_pair((first, second))
        return -LapseN**2 + h(UP(left), UP(right)) * ShiftN(DN(left)) * ShiftN(DN(right))
    if _is_covariant_time_space(first, second):
        return ShiftN(second if _is_index(second, DN) else first)
    if _is_index(first, UP) and _is_index(second, UP):
        left, right = _metric_dummy_pair((first, second))
        return h(first, second) - ShiftN(DN(left)) * ShiftN(DN(right)) * h(UP(left), first) * h(
            UP(right), second
        ) / LapseN**2
    if first == UE(0) and second == UE(0):
        return -LapseN**-2
    if _is_contravariant_time_space(first, second):
        spatial = second if _is_index(second, UP) else first
        label = _metric_dummy_pair((first, second), count=1)[0]
        return h(spatial, UP(label)) * ShiftN(DN(label)) / LapseN**2
    return None


def _is_covariant_time_space(first, second):
    return (first == DE(0) and _is_index(second, DN)) or (_is_index(first, DN) and second == DE(0))


def _is_contravariant_time_space(first, second):
    return (first == UE(0) and _is_index(second, UP)) or (_is_index(first, UP) and second == UE(0))


def _is_index(expr, index_type):
    return isinstance(expr, Index) and expr.head_name == index_type.name


def _metric_dummy_pair(expressions, *, count=2):
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


def Fourier2(expr):
    previous = None
    current = sp.expand(sp.sympify(expr))
    while current != previous:
        previous = current
        current = sp.expand(_fourier2_once(current))
    return current


def _fourier2_once(expr):
    if isinstance(expr, sp.Add):
        return sp.Add(*(_fourier2_once(arg) for arg in expr.args))
    if isinstance(expr, sp.Pow):
        base, exponent = expr.args
        if exponent == 2 and is_pdt(base):
            reduced = _gradient_square(base)
            if reduced is not None:
                return reduced
        return expr.func(_fourier2_once(base), exponent)
    if isinstance(expr, sp.Mul):
        if _has_transverse_shift_momentum_pair(expr):
            return sp.Integer(0)
        shift_derivative_zero = _transverse_shift_derivative_product(expr)
        if shift_derivative_zero is not None:
            return shift_derivative_zero
        pair_reduced = _paired_gradient_product(expr)
        if pair_reduced is not None:
            return pair_reduced
        vector_reduced = _gradient_vector_product(expr)
        if vector_reduced is not None:
            return vector_reduced
        return sp.Mul(*(_fourier2_once(arg) for arg in expr.args))
    if is_pdt(expr):
        reduced = _laplacian(expr)
        if reduced is not None:
            return reduced
    return expr


def _laplacian(expr):
    base, derivative_indices = pdt_parts(expr)
    first_pair = _first_repeated_spatial_pair(derivative_indices)
    if first_pair is None:
        return None
    first, second = first_pair
    remaining = [index for pos, index in enumerate(derivative_indices) if pos not in {first, second}]
    return -(k**2) * PdT(base, PdVars(*remaining))


def _gradient_square(expr):
    base, derivative_indices = pdt_parts(expr)
    spatial = _first_spatial_derivative(derivative_indices)
    if spatial is None:
        return None
    spatial_pos, _spatial_index = spatial
    remaining = [index for pos, index in enumerate(derivative_indices) if pos != spatial_pos]
    return k**2 * PdT(base, PdVars(*remaining)) ** 2


def _paired_gradient_product(expr):
    factors = list(expr.args)
    pdt_positions = [pos for pos, factor in enumerate(factors) if is_pdt(factor)]
    for left_pos_index, left_pos in enumerate(pdt_positions):
        left_base, left_indices = pdt_parts(factors[left_pos])
        left_spatial = _first_spatial_derivative(left_indices)
        if left_spatial is None:
            continue
        left_spatial_pos, left_spatial_index = left_spatial
        for right_pos in pdt_positions[left_pos_index + 1 :]:
            right_base, right_indices = pdt_parts(factors[right_pos])
            right_spatial = _first_spatial_derivative(right_indices)
            if right_spatial is None:
                continue
            right_spatial_pos, right_spatial_index = right_spatial
            if left_spatial_index.label != right_spatial_index.label:
                continue
            replacements = {
                left_pos: PdT(
                    left_base,
                    PdVars(*(index for pos, index in enumerate(left_indices) if pos != left_spatial_pos)),
                ),
                right_pos: PdT(
                    right_base,
                    PdVars(*(index for pos, index in enumerate(right_indices) if pos != right_spatial_pos)),
                ),
            }
            new_factors = [replacements.get(pos, factor) for pos, factor in enumerate(factors)]
            return k**2 * sp.Mul(*new_factors)
    return None


def _gradient_vector_product(expr):
    factors = list(expr.args)
    for pdt_pos, factor in enumerate(factors):
        if not is_pdt(factor):
            continue
        base, derivative_indices = pdt_parts(factor)
        spatial = _first_spatial_derivative(derivative_indices)
        if spatial is None:
            continue
        spatial_pos, contracted = spatial
        for tensor_pos, candidate in enumerate(factors):
            if tensor_pos == pdt_pos:
                continue
            if _single_matching_index_tensor(candidate, contracted):
                if _is_shift_vector_with_index(candidate, contracted):
                    return sp.Integer(0)
                new_factors = list(factors)
                new_factors[pdt_pos] = PdT(
                    base,
                    PdVars(*(index for pos, index in enumerate(derivative_indices) if pos != spatial_pos)),
                )
                new_factors.append(tensor("k")(contracted))
                return -sp.I * sp.Mul(*new_factors)
    return None


def _is_transverse_shift_derivative(base, derivative_indices):
    return any(_is_shift_vector_with_index(base, index) for index in derivative_indices)


def _has_transverse_shift_momentum_pair(term):
    factors = list(term.args) if isinstance(term, sp.Mul) else [term]
    for shift in factors:
        if tensor_head_name(shift) != "b":
            continue
        for index in _dn_indices(shift):
            if any(tensor_head_name(candidate) == "k" and _single_matching_index_tensor(candidate, index) for candidate in factors):
                return True
    return False


def _transverse_shift_derivative_product(term):
    factors = list(term.args) if isinstance(term, sp.Mul) else [term]
    shift_derivatives = []
    pdt_factors = []
    for pos, factor in enumerate(factors):
        if not is_pdt(factor):
            continue
        base, derivative_indices = pdt_parts(factor)
        pdt_factors.append((pos, base, tuple(derivative_indices)))
        shift_index = _single_shift_index(base)
        if shift_index is not None:
            shift_derivatives.append((pos, shift_index, tuple(derivative_indices)))
    for pos, shift_index, derivative_indices in shift_derivatives:
        derivative_labels = _spatial_index_labels(derivative_indices)
        if not derivative_labels:
            continue
        for other_pos, other_base, other_derivatives in pdt_factors:
            if other_pos == pos:
                continue
            other_shift_index = _single_shift_index(other_base)
            other_labels = _spatial_index_labels(other_derivatives)
            if other_shift_index is not None:
                if shift_index.label in other_labels and other_shift_index.label in derivative_labels:
                    return sp.Integer(0)
                continue
            if shift_index.label in other_labels and derivative_labels & other_labels:
                return sp.Integer(0)
    return None


def _single_shift_index(expr):
    if tensor_head_name(expr) != "b":
        return None
    indices = _dn_indices(expr)
    return indices[0] if len(indices) == 1 else None


def _spatial_index_labels(indices):
    return {
        index.label
        for index in indices
        if _is_dn_index(index)
    }


def _is_shift_vector_with_index(expr, index):
    return tensor_head_name(expr) == "b" and _single_matching_index_tensor(expr, index)


def _dn_indices(expr):
    return [arg for arg in getattr(expr, "args", ()) if isinstance(arg, Index) and arg.head_name == DN.name]


def _single_matching_index_tensor(expr, index):
    return len(getattr(expr, "args", ())) >= 2 and any(
        isinstance(arg, Index) and arg.head_name == index.head_name and arg.label == index.label
        for arg in expr.args
    )


def _first_repeated_spatial_pair(indices):
    for first, left in enumerate(indices):
        if not _is_dn_index(left):
            continue
        for second in range(first + 1, len(indices)):
            right = indices[second]
            if _is_dn_index(right) and right.label == left.label:
                return first, second
    return None


def _first_spatial_derivative(indices):
    for pos, index in enumerate(indices):
        if _is_dn_index(index):
            return pos, index
    return None


def _is_dn_index(index):
    return isinstance(index, Index) and index.head_name == DN.name
